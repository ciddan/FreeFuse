"""
FreeFuse attention bias hosted in comfy's own extension point.

Every attention function in comfy/ldm/modules/attention.py is wrapped by
`wrap_attn`, which delegates the whole call to
`transformer_options["optimized_attention_override"](func, *args, **kwargs)`
when one is installed (comfy >= the 2025-09-12 "Enable Runtime Selection
of Attention Functions" change). The override receives q/k/v *already
post-rope and post-GQA*, plus the live transformer_options.

That is everything the bias needs, and nothing about a particular
model's attention block — so this host replaces the per-family forward
mirrors (krea2_flex_bias's `_make_forward`, and the ZiT/Flux twins that
would otherwise have to be written). It also removes a standing hazard:
a forward mirror is a *copy* of comfy's attention block and silently
diverges when comfy edits theirs (Lumina's block was rewritten to use a
fused rms-rope kernel on 2026-07-22).

Sequence layout is the usual joint [txt(cap), img]. Lengths come from
whatever the family publishes:

  * Krea 2 sets transformer_options["img_slice"] = [txtlen, total]
  * families that publish nothing need an explicit cap_len source
    (see `cap_len_provider`) — Lumina/Z-Image publishes none

Everything that cannot be handled falls through to the original
attention function, so a miss degrades to "no bias on that call" rather
than to wrong math. The cases that fall through are counted and logged
once each, because a silently unbiased run is exactly the failure this
codebase has been bitten by before.

Escape hatches:
  FREEFUSE_ATTN_OVERRIDE=1   opt in (default off while it is validated)
  FREEFUSE_OVERRIDE_VALIDATE=1  on the first biased call, also compute
                             the dense-bias SDPA result and log max|diff|
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .flex_bias_core import (
    FLEX_AVAILABLE,
    BiasVectorCache,
    MAX_ADAPTERS,
    get_compiled_flex,
)

VALIDATE_MAX_SEQ = 9000  # above this the dense reference is itself an OOM


def _lengths_from_img_slice(transformer_options) -> Optional[Tuple[int, int]]:
    """Krea 2 publishes [txtlen, total] — cap_len and img_len for free."""
    sl = transformer_options.get("img_slice")
    if not sl or len(sl) < 2:
        return None
    cap_len = int(sl[0])
    img_len = int(sl[1]) - cap_len
    if cap_len <= 0 or img_len <= 0:
        return None
    return cap_len, img_len


class FreeFuseAttentionOverride:
    """Callable installed as transformer_options['optimized_attention_override']."""

    def __init__(
        self,
        lora_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List[List[int]]],
        config,
        block_indices: Sequence[int],
        latent_size: Optional[Tuple[int, int]] = None,
        cap_len_provider: Optional[Callable[[Any], Optional[Tuple[int, int]]]] = None,
        previous_override: Optional[Callable] = None,
    ):
        self.cache = BiasVectorCache(
            lora_masks, token_pos_maps, config, latent_size,
            log_prefix="[FreeFuse attn-override]")
        self.blocks = {int(i) for i in block_indices}
        self.cap_len_provider = cap_len_provider or _lengths_from_img_slice
        self.previous_override = previous_override
        self.biased_calls = 0
        self._skips: Dict[str, int] = {}
        self._validated = False

    # ---- bookkeeping ----------------------------------------------------

    def _skip(self, reason: str, func, args, kwargs):
        n = self._skips.get(reason, 0) + 1
        self._skips[reason] = n
        if n == 1:
            logging.info(f"[FreeFuse attn-override] passing through: {reason}")
        if self.previous_override is not None:
            return self.previous_override(func, *args, **kwargs)
        return func(*args, **kwargs)

    def stats(self) -> Dict[str, Any]:
        return {"biased_calls": self.biased_calls, "skipped": dict(self._skips)}

    # ---- the override ---------------------------------------------------

    def __call__(self, func, *args, **kwargs):
        to = kwargs.get("transformer_options")
        if not isinstance(to, dict):
            return self._skip("no transformer_options", func, args, kwargs)

        block_index = to.get("block_index")
        if block_index is None or int(block_index) not in self.blocks:
            # includes the refiner/txt-fusion passes, which run before the
            # main loop publishes a block index
            return self._skip("block not selected", func, args, kwargs)

        if len(args) < 3:
            return self._skip("unexpected attention signature", func, args, kwargs)
        q, k, v = args[0], args[1], args[2]
        heads = args[3] if len(args) > 3 else kwargs.get("heads")
        mask = args[4] if len(args) > 4 else kwargs.get("mask")
        skip_reshape = kwargs.get("skip_reshape", False)
        skip_output_reshape = kwargs.get("skip_output_reshape", False)

        if mask is not None:
            # The bias would have to be combined with the model's own mask.
            # Krea 2's main blocks pass None; ZiT does not (see module doc).
            return self._skip("model supplied its own attention mask",
                              func, args, kwargs)
        if heads is None:
            return self._skip("heads not resolvable", func, args, kwargs)

        lengths = self.cap_len_provider(to)
        if lengths is None:
            return self._skip("sequence lengths unavailable", func, args, kwargs)
        cap_len, img_len = lengths

        # normalise to (B, H, S, D)
        if skip_reshape:
            if q.dim() != 4:
                return self._skip("unexpected q layout", func, args, kwargs)
            qh, kh = q.shape[1], k.shape[1]
            b, s = q.shape[0], q.shape[2]
        else:
            if q.dim() != 3:
                return self._skip("unexpected q layout", func, args, kwargs)
            b, s = q.shape[0], q.shape[1]
            dim_head = q.shape[-1] // int(heads)
            q = q.view(b, -1, int(heads), dim_head).transpose(1, 2)
            k = k.view(b, -1, k.shape[-1] // dim_head, dim_head).transpose(1, 2)
            v = v.view(b, -1, v.shape[-1] // dim_head, dim_head).transpose(1, 2)
            qh, kh = q.shape[1], k.shape[1]

        if s != cap_len + img_len:
            # refiner blocks, reference-image passes, anything whose
            # sequence is not the joint [cap, img] the masks describe
            return self._skip(
                f"sequence {s} != cap {cap_len} + img {img_len}",
                func, args, kwargs)

        if kh != qh:  # GQA that the caller did not expand
            if qh % kh:
                return self._skip("non-divisible GQA", func, args, kwargs)
            rep = qh // kh
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        try:
            score_mod = self.cache.get(cap_len, img_len, q.device)
        except RuntimeError as e:
            # adapter-count ceiling and mask/length mismatches land here;
            # falling back keeps the render alive and says why, loudly
            return self._skip(f"bias unavailable ({e})", func, args, kwargs)

        out = get_compiled_flex()(q.contiguous(), k.contiguous(),
                                  v.contiguous(), score_mod=score_mod)

        if os.environ.get("FREEFUSE_OVERRIDE_VALIDATE") == "1" \
                and not self._validated:
            self._validated = True
            self._validate(q, k, v, out, cap_len, img_len)

        self.biased_calls += 1
        if skip_output_reshape:
            return out
        return out.transpose(1, 2).reshape(b, s, -1)

    def _validate(self, q, k, v, flex_out, cap_len, img_len):
        if q.shape[2] > VALIDATE_MAX_SEQ:
            logging.info(
                "[FreeFuse attn-override] VALIDATE skipped: fp32 dense "
                f"reference at S={q.shape[2]} would OOM")
            return
        from .attention_bias import construct_attention_bias
        cfg = self.cache.config
        bias = construct_attention_bias(
            lora_masks={n: m for n, m in self.cache.lora_masks.items()
                        if not n.startswith("_")},
            token_pos_maps=self.cache.token_pos_maps,
            txt_seq_len=cap_len, img_seq_len=img_len,
            bias_scale=cfg.bias_scale,
            positive_bias_scale=cfg.positive_bias_scale,
            bidirectional=cfg.bidirectional,
            use_positive_bias=cfg.use_positive_bias,
            img_img_bias_scale=getattr(cfg, "img_img_bias_scale", 0.0),
            device=q.device, dtype=torch.float32)
        if bias.dim() == 3:
            bias = bias.unsqueeze(1)
        ref = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=bias)
        diff = (ref - flex_out.float()).abs().max().item()
        logging.info(f"[FreeFuse attn-override] VALIDATE max|flex-dense| "
                     f"= {diff:.2e} (cap={cap_len} img={img_len})")


def apply_freefuse_attention_override(
    model,
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    config,
    block_indices: Sequence[int],
    latent_size: Optional[Tuple[int, int]] = None,
    cap_len_provider: Optional[Callable] = None,
) -> Optional[FreeFuseAttentionOverride]:
    """Install the bias host on this model clone. Returns the host, or
    None when the override route is unavailable (caller falls back)."""
    if os.environ.get("FREEFUSE_ATTN_OVERRIDE") != "1":
        return None
    if not FLEX_AVAILABLE:
        logging.info("[FreeFuse attn-override] flex_attention unavailable")
        return None

    live = len([n for n in lora_masks if not n.startswith("_")])
    if live > MAX_ADAPTERS:
        logging.info(
            f"[FreeFuse attn-override] {live} adapters exceeds the "
            f"unrolled score_mod's {MAX_ADAPTERS}; leaving the dense path "
            "in place")
        return None

    to = model.model_options.setdefault("transformer_options", {})
    host = FreeFuseAttentionOverride(
        lora_masks, token_pos_maps, config, block_indices,
        latent_size=latent_size,
        cap_len_provider=cap_len_provider,
        previous_override=to.get("optimized_attention_override"),
    )
    to["optimized_attention_override"] = host
    logging.info(
        f"[FreeFuse attn-override] installed for blocks "
        f"{min(host.blocks)}..{max(host.blocks)} ({len(host.blocks)} of them), "
        f"{live} adapters — O(S) bias in comfy's attention override")
    return host


__all__ = [
    "FreeFuseAttentionOverride",
    "apply_freefuse_attention_override",
]

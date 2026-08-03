"""
Krea 2 attention bias via FlexAttention — no dense (S, S) matrix, ever.

The FreeFuse bias is structurally low-rank: every quadrant rule reduces
to three scalars per (q, k) pair computed from a handful of captured
1-D vectors (numerically validated against construct_attention_bias to
float epsilon, all flag combinations):

    m_L : per-lora membership over the full [txt, img] sequence
          (txt: one-hot token ownership, img: mask values — soft masks
          included), L < 4, zero-padded
    cover = sum_L m_L
    dot(q, k) = sum_L m_L[q] * m_L[k]

    q img, k txt:  -neg * (cover_q * cover_k - dot) + pos * dot
    q txt, k img:  bidirectional * (-neg * (cover_q - dot) + pos * dot)
    q img, k img:  -ii  * (cover_q * cover_k - dot)
    q txt, k txt:  0

flex_attention computes this inside the fused kernel from the captured
vectors: peak memory is O(S) instead of O(S^2), and the dense-mask SDPA
fallback path (which can transiently expand to (B, heads, S, S)) is
avoided entirely. At 2 MP that replaces a ~134 MB matrix — plus its
per-head expansion transient — with six 8K-float vectors.

Lifecycle mirrors krea2_support: forward replacements are installed per
model call and removed in a finally block, so nothing leaks into other
generations sharing the checkpoint. The compiled kernel and the bias
vectors persist on a state object owned by this model clone's wrapper,
so torch.compile runs once per (cap_len, img_len), not once per step.

Escape hatches:
  FREEFUSE_KREA2_DENSE=1    force the legacy dense-mask path
  FREEFUSE_FLEX_VALIDATE=1  on the first hooked forward, also compute
                            the dense-bias SDPA result and log max|diff|
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# The bias math, the dynamo guard and the compiled-kernel handle are
# family-agnostic and shared with attention_override.py — see
# flex_bias_core. Re-exported under their original private names so this
# module's internals (and the validators) keep working unchanged.
from .flex_bias_core import (  # noqa: F401
    FLEX_AVAILABLE as _FLEX_AVAILABLE,
    ensure_dynamo_limits as _ensure_dynamo_limits,
    get_compiled_flex as _get_compiled_flex,
    build_bias_vectors,
    make_score_mod,
    resize_flat_mask as _resize_flat_mask,
)

try:
    from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
except Exception:
    flex_attention = None


class _FlexBiasState:
    """Per-model-clone persistent state: vectors + score_mod + cap_len."""

    def __init__(self, lora_masks, token_pos_maps, config,
                 latent_size: Optional[Tuple[int, int]] = None):
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.latent_size = latent_size  # mask grid (H, W) at build time
        self.cap_len: Optional[int] = None  # captured from txtfusion per call
        self._key: Optional[Tuple[int, int]] = None
        self.score_mod: Optional[Callable] = None
        self.vectors = None
        self._validated = False

    def ensure(self, seqlen: int, device: torch.device):
        if self.cap_len is None:
            raise RuntimeError(
                "[FreeFuse Krea2 flex] cap_len was not captured before "
                "bias injection (txtfusion hook missing?)")
        cap_len = int(self.cap_len)
        img_len = seqlen - cap_len
        if img_len <= 0:
            raise RuntimeError(
                f"[FreeFuse Krea2 flex] invalid lengths: total={seqlen}, "
                f"cap_len={cap_len}")
        key = (cap_len, img_len)
        if self._key != key or self.score_mod is None:
            _ensure_dynamo_limits()  # re-assert: other nodes clobber these
            masks = {
                name: _resize_flat_mask(mask.reshape(-1).float(),
                                        self.latent_size, img_len)
                for name, mask in self.lora_masks.items()
                if not name.startswith("_")
            }
            self.vectors = build_bias_vectors(
                masks, self.token_pos_maps, cap_len, img_len, device)
            self.score_mod = make_score_mod(self.vectors, self.config)
            self._key = key
            logging.info(
                f"[FreeFuse Krea2 flex] bias vectors built: cap={cap_len} "
                f"img={img_len} (O(S) memory; kernel compiles per shape)")
        return self.score_mod


class FreeFuseKrea2FlexBias:
    """Per-call attention forward replacement using flex_attention."""

    def __init__(self, state: _FlexBiasState, diffusion_model,
                 block_indices: List[int]):
        self.state = state
        self.diffusion_model = diffusion_model
        self.block_indices = sorted({int(i) for i in block_indices})
        self._originals: List[Tuple[Any, Callable]] = []

    def install(self):
        blocks = getattr(self.diffusion_model, "blocks", None)
        if blocks is None:
            raise RuntimeError(
                "[FreeFuse Krea2 flex] diffusion model has no `blocks`")
        txtfusion = getattr(self.diffusion_model, "txtfusion", None)
        if txtfusion is None:
            raise RuntimeError(
                "[FreeFuse Krea2 flex] cannot find `txtfusion` for cap_len "
                "capture")
        state = self.state

        def _cap_hook(module, args, kwargs, output):
            if torch.is_tensor(output):
                state.cap_len = int(output.shape[1])

        self._cap_handle = txtfusion.register_forward_hook(
            _cap_hook, with_kwargs=True)
        installed = 0
        for idx in self.block_indices:
            if idx < 0 or idx >= len(blocks):
                continue
            attn = getattr(blocks[idx], "attn", None)
            if attn is None:
                raise RuntimeError(
                    f"[FreeFuse Krea2 flex] blocks[{idx}] has no `attn` "
                    "- comfy's Krea2 block layout changed")
            self._originals.append((attn, attn.forward))
            attn.forward = self._make_forward(attn)
            installed += 1
        if installed == 0:
            raise RuntimeError(
                f"[FreeFuse Krea2 flex] no valid blocks from "
                f"{self.block_indices}")

    def remove(self):
        for attn, orig in self._originals:
            attn.forward = orig
        self._originals.clear()
        if getattr(self, "_cap_handle", None) is not None:
            self._cap_handle.remove()
            self._cap_handle = None

    def _make_forward(self, attn) -> Callable:
        # Mirrors comfy.ldm.krea2.model.Attention.forward, swapping
        # optimized_attention_masked for flex_attention with the bias
        # computed in-kernel.
        from comfy.ldm.krea2.model import apply_rope
        state = self.state

        def forward(x, freqs=None, mask=None, transformer_options={}):
            bsz, seqlen, _ = x.shape
            score_mod = state.ensure(seqlen, x.device)

            patches = transformer_options.get("patches", {})
            if "attn1_patch" in patches and not getattr(
                    state, "_warned_patches", False):
                state._warned_patches = True
                logging.warning(
                    "[FreeFuse Krea2 flex] attn1_patch present but not "
                    "supported on the flex path; set FREEFUSE_KREA2_DENSE=1 "
                    "if that patch matters")

            q, k, v, gate = attn.wq(x), attn.wk(x), attn.wv(x), attn.gate(x)
            q = q.view(bsz, seqlen, attn.heads, attn.headdim).transpose(1, 2)
            k = k.view(bsz, seqlen, attn.kvheads, attn.headdim).transpose(1, 2)
            v = v.view(bsz, seqlen, attn.kvheads, attn.headdim).transpose(1, 2)
            q, k = attn.qknorm(q, k)
            if freqs is not None:
                q, k = apply_rope(q, k, freqs)
            if attn.kvheads != attn.heads:
                rep = attn.heads // attn.kvheads
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)

            out = _get_compiled_flex()(q, k, v, score_mod=score_mod)

            if os.environ.get("FREEFUSE_FLEX_VALIDATE") == "1" \
                    and not state._validated:
                state._validated = True
                if seqlen > 9000:
                    print("[FreeFuse Krea2 flex] VALIDATE skipped: fp32 "
                          f"dense reference at S={seqlen} would transiently "
                          "allocate multiple GB")
                else:
                    with torch.no_grad():
                        cap = int(state.cap_len)
                        img_len = seqlen - cap
                        from .attention_bias import construct_attention_bias
                        masks = {
                            n: _resize_flat_mask(
                                mk.reshape(-1).float(), state.latent_size,
                                img_len).unsqueeze(0)
                            for n, mk in state.lora_masks.items()
                            if not n.startswith("_")}
                        dense = construct_attention_bias(
                            lora_masks=masks,
                            token_pos_maps=state.token_pos_maps,
                            txt_seq_len=cap, img_seq_len=img_len,
                            bias_scale=state.config.bias_scale,
                            positive_bias_scale=state.config.positive_bias_scale,
                            bidirectional=state.config.bidirectional,
                            use_positive_bias=state.config.use_positive_bias,
                            img_img_bias_scale=getattr(
                                state.config, "img_img_bias_scale", 0.0),
                            device=q.device, dtype=torch.float32)
                        ref = F.scaled_dot_product_attention(
                            q.float(), k.float(), v.float(),
                            attn_mask=dense.unsqueeze(1))
                        diff = (out.float() - ref).abs().max().item()
                        print(f"[FreeFuse Krea2 flex] VALIDATE "
                              f"max|flex-dense| = {diff:.3e}")

            out = out.transpose(1, 2).reshape(bsz, seqlen, -1)
            return attn.wo(out * F.sigmoid(gate))

        return forward


def apply_krea2_flex_bias_patches(
    model,
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    config,
    block_indices: Optional[List[int]] = None,
    latent_size: Optional[Tuple[int, int]] = None,
) -> bool:
    """Register the per-call flex wrapper. Returns False if flex is
    unavailable (caller should fall back to the dense path)."""
    if not _FLEX_AVAILABLE or os.environ.get("FREEFUSE_KREA2_DENSE") == "1":
        return False

    diffusion_model = model.model.diffusion_model
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        raise RuntimeError("[FreeFuse Krea2 flex] cannot find `blocks`")
    if block_indices is None:
        block_indices = list(range(len(blocks)))

    state = _FlexBiasState(lora_masks, token_pos_maps, config,
                           latent_size=latent_size)
    previous_wrapper = model.model_options.get("model_function_wrapper")

    def krea2_flex_wrapper(apply_model_fn, args):
        replacer = FreeFuseKrea2FlexBias(state, diffusion_model,
                                         block_indices)
        replacer.install()
        try:
            if previous_wrapper is not None:
                return previous_wrapper(apply_model_fn, args)
            return apply_model_fn(args["input"], args["timestep"],
                                  **args["c"])
        finally:
            replacer.remove()

    model.model_options["model_function_wrapper"] = krea2_flex_wrapper
    logging.info(
        f"[FreeFuse Krea2 flex] Registered flex-attention bias wrapper for "
        f"blocks {block_indices[0]}..{block_indices[-1]} "
        f"({len(block_indices)} blocks, O(S) bias memory)")
    return True


__all__ = [
    "apply_krea2_flex_bias_patches",
    "build_bias_vectors",
    "make_score_mod",
]

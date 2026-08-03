"""
Family-agnostic core of the FreeFuse FlexAttention bias.

Everything here depends only on the lora masks, the token position maps
and the two sequence lengths — never on a particular model's attention
block. It is shared by:

  * krea2_flex_bias.py      — attention forward replacement (legacy path)
  * attention_override.py   — comfy's optimized_attention_override host

The bias is structurally low-rank: every quadrant rule reduces to three
scalars per (q, k) pair computed from a handful of captured 1-D vectors
(numerically validated against construct_attention_bias to float
epsilon, all flag combinations):

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
avoided entirely.
"""

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention
    FLEX_AVAILABLE = True
except Exception:
    flex_attention = None
    FLEX_AVAILABLE = False


def ensure_dynamo_limits():
    """Re-assert dynamo limits. Process-global config is a commons:
    other custom nodes (e.g. SeedVR2's TorchCompileSettings) legitimately
    set their own values and clobber ours between runs, after which the
    next new flex shape silently falls back to the eager S^2 fp32 math
    path. Called at import AND before every potential (re)compile."""
    try:
        import torch._dynamo as _dynamo
        # torch >= 2.10 renamed the operative knob to recompile_limit;
        # cache_size_limit still exists but is no longer consulted.
        for _name, _val in (("recompile_limit", 64),
                            ("accumulated_recompile_limit", 512),
                            ("cache_size_limit", 64),
                            ("accumulated_cache_size_limit", 512)):
            if hasattr(_dynamo.config, _name):
                setattr(_dynamo.config, _name,
                        max(getattr(_dynamo.config, _name) or 0, _val))
        if hasattr(_dynamo.config, "fail_on_recompile_limit_hit"):
            _dynamo.config.fail_on_recompile_limit_hit = True
    except Exception as e:
        logging.warning(f"[FreeFuse flex] could not raise dynamo "
                        f"recompile limit: {e}")


ensure_dynamo_limits()

_flex_compiled = None


def get_compiled_flex():
    global _flex_compiled
    if _flex_compiled is None:
        _flex_compiled = torch.compile(flex_attention, dynamic=False)
    return _flex_compiled


MAX_ADAPTERS = 4  # score_mod is unrolled over m0..m3


def build_bias_vectors(
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    cap_len: int,
    img_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Membership/cover/segment vectors over the [txt, img] sequence.

    Mirrors construct_attention_bias exactly: background ("_"-prefixed)
    excluded, batch-0 positions, out-of-range positions dropped,
    last-writer-wins on shared tokens.
    """
    names = [n for n in lora_masks.keys() if not n.startswith("_")]
    if len(names) > MAX_ADAPTERS:
        raise RuntimeError(
            f"[FreeFuse flex] {len(names)} adapters; the unrolled "
            f"score_mod supports at most {MAX_ADAPTERS}")
    S = cap_len + img_len
    m = torch.zeros(MAX_ADAPTERS, S, dtype=torch.float32, device=device)

    owner = torch.full((cap_len,), -1, dtype=torch.long)
    for name, positions_list in token_pos_maps.items():
        if name.startswith("_") or name not in names:
            continue
        li = names.index(name)
        if not positions_list:
            continue
        flat = positions_list[0] if isinstance(positions_list[0], list) \
            else positions_list
        for p in flat:
            if isinstance(p, int) and 0 <= p < cap_len:
                owner[p] = li
    for j in range(cap_len):
        if owner[j] >= 0:
            m[owner[j], j] = 1.0

    for li, name in enumerate(names):
        mask = lora_masks[name]
        flat = mask.reshape(-1).float()
        if flat.numel() != img_len:
            raise RuntimeError(
                f"[FreeFuse flex] mask '{name}' has {flat.numel()} "
                f"tokens, expected img_len={img_len}")
        m[li, cap_len:] = flat.to(device)

    cover = m.sum(dim=0)
    is_img = torch.zeros(S, dtype=torch.float32, device=device)
    is_img[cap_len:] = 1.0
    return {"m": m.contiguous(), "cover": cover.contiguous(),
            "is_img": is_img.contiguous()}


def make_score_mod(vectors: Dict[str, torch.Tensor], config) -> Callable:
    m0, m1, m2, m3 = vectors["m"][0], vectors["m"][1], vectors["m"][2], vectors["m"][3]
    cover, is_img = vectors["cover"], vectors["is_img"]
    neg = float(config.bias_scale)
    pos = float(config.positive_bias_scale) if config.use_positive_bias else 0.0
    bidir = 1.0 if config.bidirectional else 0.0
    ii = float(getattr(config, "img_img_bias_scale", 0.0))

    def score_mod(score, b, h, q_idx, kv_idx):
        cq = cover[q_idx]
        ck = cover[kv_idx]
        qi = is_img[q_idx]
        ki = is_img[kv_idx]
        dot = (m0[q_idx] * m0[kv_idx] + m1[q_idx] * m1[kv_idx]
               + m2[q_idx] * m2[kv_idx] + m3[q_idx] * m3[kv_idx])
        bias = (qi * (1.0 - ki) * (-neg * (cq * ck - dot) + pos * dot)
                + (1.0 - qi) * ki * bidir * (-neg * (cq - dot) + pos * dot)
                + qi * ki * (-ii) * (cq * ck - dot))
        return score + bias

    return score_mod


def resize_flat_mask(flat: torch.Tensor, src_hw: Optional[Tuple[int, int]],
                     target_len: int) -> torch.Tensor:
    """Area-resize a flat token mask to a new token count (grid aspect
    preserved). Mirrors the dense path's per-forward mask resize so the
    same patched model reuses correctly at a different resolution
    (e.g. a confined high-res pass after upscale)."""
    n = flat.numel()
    if n == target_len:
        return flat
    if src_hw is not None and src_hw[0] * src_hw[1] == n:
        h, w = int(src_hw[0]), int(src_hw[1])
    else:
        # infer a plausible grid for the source
        h = int(n ** 0.5)
        while h > 1 and n % h:
            h -= 1
        w = n // h
    ratio = w / h
    th = max(1, int(round((target_len / ratio) ** 0.5)))
    tw = target_len // th
    while th > 1 and th * tw != target_len:
        th -= 1
        tw = target_len // th
    if th * tw != target_len:
        th, tw = 1, target_len
    m2 = flat.view(1, 1, h, w).float()
    return F.interpolate(m2, size=(th, tw), mode="area").reshape(-1)


class BiasVectorCache:
    """Builds and caches score_mods keyed by (cap_len, img_len).

    Owned by one model clone. The compiled flex kernel and the vectors
    persist across sampling steps, so torch.compile runs once per shape
    rather than once per step.
    """

    def __init__(self, lora_masks, token_pos_maps, config,
                 latent_size: Optional[Tuple[int, int]] = None,
                 log_prefix: str = "[FreeFuse flex]"):
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.latent_size = latent_size  # mask grid (H, W) at build time
        self.log_prefix = log_prefix
        self._key: Optional[Tuple[int, int]] = None
        self.score_mod: Optional[Callable] = None
        self.vectors = None

    def adapter_count(self) -> int:
        return len([n for n in self.lora_masks if not n.startswith("_")])

    def get(self, cap_len: int, img_len: int, device: torch.device) -> Callable:
        key = (int(cap_len), int(img_len))
        if self._key == key and self.score_mod is not None:
            return self.score_mod
        ensure_dynamo_limits()  # re-assert: other nodes clobber these
        masks = {
            name: resize_flat_mask(mask.reshape(-1).float(),
                                   self.latent_size, img_len)
            for name, mask in self.lora_masks.items()
            if not name.startswith("_")
        }
        self.vectors = build_bias_vectors(
            masks, self.token_pos_maps, key[0], key[1], device)
        self.score_mod = make_score_mod(self.vectors, self.config)
        self._key = key
        logging.info(
            f"{self.log_prefix} bias vectors built: cap={key[0]} "
            f"img={key[1]} (O(S) memory; kernel compiles per shape)")
        return self.score_mod


__all__ = [
    "FLEX_AVAILABLE", "MAX_ADAPTERS", "ensure_dynamo_limits",
    "get_compiled_flex", "build_bias_vectors", "make_score_mod",
    "resize_flat_mask", "BiasVectorCache",
]

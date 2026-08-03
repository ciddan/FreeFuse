"""
Z-Image (Lumina2 / NextDiT) attention-bias injection.

comfy's Lumina2 forward consults no patches_replace, so the
set_model_patch_replace("dit", "layer", i) route used by
FreeFuseZImageBiasBlockReplace registers patches nothing ever reads —
bias-on and bias-off renders are bit-identical (verified 2026-08-02,
comfy 0.29). This module ports the krea2_support approach instead:
a model_function_wrapper installs forward pre-hooks on each layer's
JointAttention for exactly one model call and removes them in a
finally block, so hook lifetime is scoped to this model clone and
nothing leaks into unrelated generations sharing the checkpoint.

The pre-hook replaces the attention's x_mask argument with the additive
FreeFuse bias (combined with any existing mask). Layout is ComfyUI's
NextDiT unified sequence [cap(txt), img]; img_len comes from the lora
masks' token count and cap_len is inferred per forward as
seq_len - img_len, matching the bypass-hook mask logic.

Everything here fails LOUDLY: missing layers/attention modules raise at
install time instead of logging success and doing nothing.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


class FreeFuseZImageBiasHooks:
    """Per-call bias injection on NextDiT layers (install/remove pairs)."""

    def __init__(
        self,
        lora_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List[List[int]]],
        config,
        diffusion_model,
        layer_indices: List[int],
    ):
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.diffusion_model = diffusion_model
        self.layer_indices = {int(i) for i in layer_indices}
        self._bias_cache: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
        self._hook_handles: List[Any] = []
        self._construct_attention_bias = None

        # img token count is fixed by the masks; cap_len varies per prompt
        first_mask = next(iter(lora_masks.values()), None)
        if first_mask is None:
            raise RuntimeError("[FreeFuse Z-Image] No lora masks provided for bias injection")
        self._img_len = int(first_mask.numel() // max(1, first_mask.shape[0])) \
            if first_mask.dim() > 1 else int(first_mask.numel())

    def install(self) -> None:
        from .attention_bias import construct_attention_bias
        self._construct_attention_bias = construct_attention_bias

        layers = getattr(self.diffusion_model, "layers", None)
        if layers is None:
            raise RuntimeError(
                "[FreeFuse Z-Image] Cannot find `layers` on the diffusion model "
                f"({type(self.diffusion_model).__name__}) for bias injection"
            )

        installed = 0
        for idx in sorted(self.layer_indices):
            if idx < 0 or idx >= len(layers):
                continue
            attn = getattr(layers[idx], "attention", None)
            if attn is None:
                raise RuntimeError(
                    f"[FreeFuse Z-Image] layers[{idx}] has no `attention` module "
                    "- comfy's Lumina2 block layout changed; bias injection needs updating"
                )
            self._hook_handles.append(
                attn.register_forward_pre_hook(
                    self._make_bias_prehook(idx), with_kwargs=True
                )
            )
            installed += 1
        if installed == 0:
            raise RuntimeError(
                f"[FreeFuse Z-Image] No valid bias layers from requested indices "
                f"{sorted(self.layer_indices)} (model has {len(layers)} layers)"
            )

    def remove(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _get_or_build_bias(self, img_len, cap_len, device, dtype):
        key = (img_len, cap_len)
        if key in self._bias_cache:
            cached = self._bias_cache[key]
            return cached.to(device=device, dtype=dtype) if cached is not None else None

        bias = self._construct_attention_bias(
            lora_masks=self.lora_masks,
            token_pos_maps=self.token_pos_maps,
            txt_seq_len=cap_len,
            img_seq_len=img_len,
            bias_scale=self.config.bias_scale,
            positive_bias_scale=self.config.positive_bias_scale,
            bidirectional=self.config.bidirectional,
            use_positive_bias=self.config.use_positive_bias,
            img_img_bias_scale=getattr(self.config, "img_img_bias_scale", 0.0),
            device=device,
            dtype=dtype,
        )
        self._bias_cache[key] = bias
        return bias

    def _make_bias_prehook(self, layer_index: int) -> Callable:
        def _prehook(module, args, kwargs):
            # JointAttention.forward(self, x, x_mask, freqs_cis, transformer_options={})
            if not args:
                return None
            x = args[0]
            seqlen = int(x.shape[1])
            img_len = self._img_len
            cap_len = seqlen - img_len
            if cap_len <= 0:
                raise RuntimeError(
                    f"[FreeFuse Z-Image] Invalid bias sequence lengths: total={seqlen}, "
                    f"img_len={img_len} (from lora masks), cap_len={cap_len}"
                )

            bias = self._get_or_build_bias(img_len, cap_len, x.device, x.dtype)
            if bias is None:
                return None
            if bias.dim() == 3:
                bias = bias.unsqueeze(1)  # (B, 1, S, S), broadcasts over heads

            if os.environ.get("FREEFUSE_DEBUG_ZIMAGE") == "1" and layer_index == min(self.layer_indices):
                if not hasattr(self, "_dbg_logged"):
                    self._dbg_logged = True
                    print(f"[FreeFuse Z-Image DBG] S={seqlen} cap_len={cap_len} "
                          f"img_len={img_len} bias min={bias.min().item():.2f} "
                          f"max={bias.max().item():.2f}")

            # Combine with the existing x_mask (positional index 1 or kwarg)
            new_args = list(args)
            new_kwargs = dict(kwargs) if kwargs else {}
            existing = new_args[1] if len(new_args) >= 2 else new_kwargs.get("x_mask")
            combined = bias
            if existing is not None and torch.is_tensor(existing):
                if existing.dtype == torch.bool:
                    # key-padding mask (B, S) -> additive
                    add = torch.zeros(
                        existing.shape, device=x.device, dtype=bias.dtype)
                    add.masked_fill_(~existing, torch.finfo(bias.dtype).min)
                    combined = bias + add.view(add.shape[0], 1, 1, add.shape[-1])
                else:
                    add = existing.to(device=x.device, dtype=bias.dtype)
                    while add.dim() < 4:
                        add = add.unsqueeze(1) if add.dim() == 3 else add.unsqueeze(0)
                    combined = bias + add

            if len(new_args) >= 2:
                new_args[1] = combined
            else:
                new_kwargs["x_mask"] = combined
            return tuple(new_args), new_kwargs

        return _prehook


def resolve_zimage_bias_layers(diffusion_model, bias_blocks: str) -> List[int]:
    """Map bias_blocks presets to NextDiT layer indices (loud on failure)."""
    layers = getattr(diffusion_model, "layers", None)
    if layers is None:
        raise RuntimeError(
            "[FreeFuse Z-Image] Cannot resolve bias layers: diffusion model has no `layers`"
        )
    n = len(layers)
    preset = bias_blocks
    # Flux-specific stream presets have no meaning here; map to last_half
    if preset in ("double_stream_only", "single_stream_only", "last_half_double"):
        preset = "last_half"
    if preset in ("all", None):
        return list(range(n))
    if preset == "last_half":
        return list(range(n // 2, n))
    raise ValueError(
        f"[FreeFuse Z-Image] bias_blocks='{bias_blocks}' is not valid for Z-Image. "
        "Use 'all', 'last_half', or 'none'."
    )


def apply_zimage_bias_patches(
    model,
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    config,
    layer_indices: Optional[List[int]] = None,
):
    """Register the per-call bias wrapper on this model clone (krea2-style)."""
    diffusion_model = model.model.diffusion_model
    layers = getattr(diffusion_model, "layers", None)
    if layers is None:
        raise RuntimeError(
            "[FreeFuse Z-Image] Cannot find `layers` on the diffusion model "
            f"({type(diffusion_model).__name__})"
        )

    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    previous_wrapper = model.model_options.get("model_function_wrapper")

    def zimage_bias_wrapper(apply_model_fn, args):
        hooks = FreeFuseZImageBiasHooks(
            lora_masks, token_pos_maps, config, diffusion_model, layer_indices
        )
        hooks.install()
        try:
            if previous_wrapper is not None:
                return previous_wrapper(apply_model_fn, args)
            return apply_model_fn(args["input"], args["timestep"], **args["c"])
        finally:
            hooks.remove()

    model.model_options["model_function_wrapper"] = zimage_bias_wrapper
    logging.info(
        f"[FreeFuse Z-Image] Registered per-call bias wrapper for layers "
        f"{sorted(set(int(i) for i in layer_indices))} (scoped to this model clone only)"
    )
    return zimage_bias_wrapper


__all__ = [
    "FreeFuseZImageBiasHooks",
    "resolve_zimage_bias_layers",
    "apply_zimage_bias_patches",
]

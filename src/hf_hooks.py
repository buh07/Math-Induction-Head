"""Forward-hook helpers for Hugging Face transformer models."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional

import torch

from .hooks import AttentionHookConfig, NeuronHookConfig, summarize_attention_configs


def _scale_output(output, scale: float):
    if torch.is_tensor(output):
        return output * scale
    if isinstance(output, tuple):
        return tuple(_scale_output(item, scale) for item in output)
    return output


class HFHookApplier:
    """Register attention/neuron scaling hooks on supported HF architectures."""

    def __init__(self, model):
        self.model = model
        self.handles: List = []
        self.layers = self._locate_layers(model)

    def clear(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def apply_attention_hooks(self, configs: Iterable[AttentionHookConfig]) -> None:
        grouped = {}
        for cfg in configs:
            grouped.setdefault(cfg.layer, []).append(cfg)
        for layer, layer_configs in grouped.items():
            module = self._get_attention_module(layer)
            handle = module.register_forward_hook(
                lambda mod, _inp, output, cfgs=layer_configs: self._apply_attention_scaling(mod, output, cfgs)
            )
            self.handles.append(handle)

    def apply_neuron_hooks(self, configs: Iterable[NeuronHookConfig]) -> None:
        for cfg in configs:
            module = self._get_mlp_module(cfg.layer)
            handle = module.register_forward_hook(
                lambda _m, _inp, output, scale=cfg.scale: _scale_output(output, scale)
            )
            self.handles.append(handle)

    def _get_attention_module(self, layer_index: int):
        layer = self.layers[layer_index]
        for attr in ("self_attn", "attention", "attn"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise AttributeError(f"Layer {layer_index} lacks a recognized attention module")

    def _get_mlp_module(self, layer_index: int):
        layer = self.layers[layer_index]
        for attr in ("mlp", "feed_forward", "ffn", "parallel_attn"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise AttributeError(f"Layer {layer_index} lacks a recognized MLP/FFN module")

    def _apply_attention_scaling(self, module, output, configs: Iterable[AttentionHookConfig]):
        base_scale, head_scales, downscale = summarize_attention_configs(configs)
        attn_tensor, remainder = self._extract_attention_tensor(output)
        if attn_tensor is None:
            return output
        scaled = self._scale_attention_tensor(module, attn_tensor, base_scale, head_scales, downscale)
        if remainder is None:
            return scaled
        return (scaled, *remainder)

    def _extract_attention_tensor(self, output):
        if torch.is_tensor(output):
            return output, None
        if isinstance(output, tuple) and output:
            first, *rest = output
            if torch.is_tensor(first):
                return first, tuple(rest)
        return None, None

    def _scale_attention_tensor(
        self,
        module,
        tensor: torch.Tensor,
        base_scale: float,
        head_scales: Dict[int, float],
        downscale: Optional[float],
    ) -> torch.Tensor:
        default_scale = base_scale * (downscale if downscale is not None else 1.0)
        if not head_scales:
            return tensor * default_scale
        head_view = self._view_as_heads(module, tensor)
        if head_view is None:
            return tensor * default_scale
        scaled = head_view * default_scale
        for head, factor in head_scales.items():
            if head < head_view.shape[-2]:
                scaled[..., head, :] = head_view[..., head, :] * base_scale * factor
        return scaled.reshape(tensor.shape)

    def _view_as_heads(self, module, tensor: torch.Tensor):
        head_count = self._infer_head_count(module)
        if head_count is None:
            return None
        hidden = tensor.shape[-1]
        if hidden % head_count != 0:
            return None
        head_dim = hidden // head_count
        new_shape = tensor.shape[:-1] + (head_count, head_dim)
        try:
            return tensor.view(*new_shape)
        except RuntimeError:
            return None

    @staticmethod
    def _infer_head_count(module) -> Optional[int]:
        for attr in ("num_heads", "num_attention_heads", "n_head", "n_heads"):
            value = getattr(module, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    @staticmethod
    def _locate_layers(model):
        for attr in ("model", "transformer", "gpt_neox"):
            if hasattr(model, attr):
                container = getattr(model, attr)
                if hasattr(container, "layers"):
                    return container.layers
                if hasattr(container, "h"):
                    return container.h
        raise AttributeError("Could not locate layer stack on model")


@contextmanager
def apply_hooks(
    model,
    attention_configs: Optional[Iterable[AttentionHookConfig]] = None,
    neuron_configs: Optional[Iterable[NeuronHookConfig]] = None,
):
    applier = HFHookApplier(model)
    try:
        if attention_configs:
            applier.apply_attention_hooks(attention_configs)
        if neuron_configs:
            applier.apply_neuron_hooks(neuron_configs)
        yield
    finally:
        applier.clear()

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


def _scale_tensor_index(tensor: torch.Tensor, index: int, scale: float) -> torch.Tensor:
    if tensor.shape[-1] == 0:
        return tensor
    if index < 0 or index >= tensor.shape[-1]:
        raise IndexError(
            f"Neuron index {index} out of bounds for tensor with last dimension {tensor.shape[-1]}"
        )
    scaled = tensor.clone()
    scaled[..., index] = scaled[..., index] * scale
    return scaled


def _scale_output_index(output, index: int, scale: float):
    if torch.is_tensor(output):
        return _scale_tensor_index(output, index, scale)
    if isinstance(output, tuple) and output:
        first, *rest = output
        if torch.is_tensor(first):
            return (_scale_tensor_index(first, index, scale), *rest)
    return output


class HFHookApplier:
    """Register attention/neuron scaling hooks on supported HF architectures."""

    def __init__(
        self,
        model,
        *,
        strict_attention_heads: bool = False,
        hook_debug_counters: Optional[Dict[str, Dict[int, int]]] = None,
    ):
        self.model = model
        self.handles: List = []
        self.layers = self._locate_layers(model)
        self.strict_attention_heads = strict_attention_heads
        if hook_debug_counters is None:
            hook_debug_counters = {}
        self.hook_debug_counters = hook_debug_counters
        self.hook_debug_counters.setdefault("attention_targeted_pre_proj", {})

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
            if any(cfg.head is not None for cfg in layer_configs):
                proj = self._get_attention_output_projection(module)
                if proj is None:
                    raise AttributeError(
                        f"Layer {layer} attention module lacks an output projection required "
                        "for true head-targeted scaling"
                    )
                handle = proj.register_forward_pre_hook(
                    lambda _proj, inputs, attn_module=module, cfgs=layer_configs, layer_index=layer: self._apply_head_scaling_pre_proj(
                        attn_module, inputs, cfgs, layer_index=layer_index
                    )
                )
            else:
                handle = module.register_forward_hook(
                    lambda mod, _inp, output, cfgs=layer_configs: self._apply_attention_scaling(
                        mod, output, cfgs
                    )
                )
            self.handles.append(handle)

    def apply_neuron_hooks(self, configs: Iterable[NeuronHookConfig]) -> None:
        for cfg in configs:
            module = self._get_mlp_module(cfg.layer)
            proj = self._get_mlp_output_projection(module)
            if proj is not None:
                handle = proj.register_forward_pre_hook(
                    lambda _proj, inputs, scale=cfg.scale, neuron_index=cfg.neuron_index: self._apply_neuron_scaling_pre_proj(
                        inputs, neuron_index, scale
                    )
                )
            else:
                handle = module.register_forward_hook(
                    lambda _m, _inp, output, scale=cfg.scale, neuron_index=cfg.neuron_index: _scale_output_index(
                        output, neuron_index, scale
                    )
                )
            self.handles.append(handle)

    def _get_attention_module(self, layer_index: int):
        layer = self.layers[layer_index]
        for attr in ("self_attn", "attention", "attn"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise AttributeError(f"Layer {layer_index} lacks a recognized attention module")

    def _get_attention_output_projection(self, module):
        for attr in ("o_proj", "c_proj", "dense", "out_proj"):
            if hasattr(module, attr):
                return getattr(module, attr)
        return None

    def _get_mlp_module(self, layer_index: int):
        layer = self.layers[layer_index]
        for attr in ("mlp", "feed_forward", "ffn", "parallel_attn"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise AttributeError(f"Layer {layer_index} lacks a recognized MLP/FFN module")

    def _get_mlp_output_projection(self, module):
        for attr in ("down_proj", "c_proj", "dense_4h_to_h", "fc2", "proj_out"):
            if hasattr(module, attr):
                return getattr(module, attr)
        return None

    def _apply_attention_scaling(self, module, output, configs: Iterable[AttentionHookConfig]):
        base_scale, head_scales, downscale = summarize_attention_configs(configs)
        attn_tensor, remainder = self._extract_attention_tensor(output)
        if attn_tensor is None:
            return output
        scaled = self._scale_attention_tensor(module, attn_tensor, base_scale, head_scales, downscale)
        if remainder is None:
            return scaled
        return (scaled, *remainder)

    def _apply_head_scaling_pre_proj(
        self,
        attn_module,
        inputs,
        configs: Iterable[AttentionHookConfig],
        *,
        layer_index: int,
    ):
        if not inputs:
            return inputs
        first, *rest = inputs
        if not torch.is_tensor(first):
            return inputs
        counters = self.hook_debug_counters.get("attention_targeted_pre_proj")
        if counters is not None:
            counters[layer_index] = counters.get(layer_index, 0) + 1
        base_scale, head_scales, downscale = summarize_attention_configs(configs)
        scaled_first = self._scale_concat_head_tensor(
            attn_module,
            first,
            base_scale=base_scale,
            head_scales=head_scales,
            downscale=downscale,
            strict=self.strict_attention_heads,
        )
        return (scaled_first, *rest)

    def _apply_neuron_scaling_pre_proj(self, inputs, neuron_index: int, scale: float):
        if not inputs:
            return inputs
        first, *rest = inputs
        if not torch.is_tensor(first):
            return inputs
        return (_scale_tensor_index(first, neuron_index, scale), *rest)

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

    def _scale_concat_head_tensor(
        self,
        module,
        tensor: torch.Tensor,
        *,
        base_scale: float,
        head_scales: Dict[int, float],
        downscale: Optional[float],
        strict: bool = False,
    ) -> torch.Tensor:
        default_scale = base_scale * (downscale if downscale is not None else 1.0)
        if not head_scales:
            return tensor * default_scale

        head_view = self._view_as_heads(module, tensor)
        if head_view is None:
            if strict:
                raise RuntimeError(
                    "Could not reshape attention projection input into heads for targeted head scaling. "
                    f"tensor_shape={tuple(tensor.shape)} head_count={self._infer_head_count(module)}"
                )
            return tensor * default_scale

        scaled = head_view * default_scale
        head_axis = self._head_axis(head_view.shape, self._infer_head_count(module))
        if head_axis is None:
            return tensor * default_scale
        head_count = head_view.shape[head_axis]
        for head, factor in head_scales.items():
            if 0 <= head < head_count:
                index = [slice(None)] * head_view.ndim
                index[head_axis] = head
                scaled[tuple(index)] = head_view[tuple(index)] * base_scale * factor
        return scaled.reshape(tensor.shape)

    def _view_as_heads(self, module, tensor: torch.Tensor):
        head_count = self._infer_head_count(module)
        if head_count is None:
            return None
        if tensor.ndim >= 2 and tensor.shape[-2] == head_count:
            return tensor
        hidden = tensor.shape[-1]
        if hidden % head_count != 0:
            return None
        head_dim = hidden // head_count
        new_shape = tensor.shape[:-1] + (head_count, head_dim)
        try:
            return tensor.reshape(*new_shape)
        except RuntimeError:
            return None

    @staticmethod
    def _head_axis(shape, head_count: Optional[int]) -> Optional[int]:
        if head_count is None:
            return None
        if len(shape) >= 2 and shape[-2] == head_count:
            return len(shape) - 2
        for axis in range(len(shape) - 1):
            if shape[axis] == head_count:
                return axis
        return None

    @staticmethod
    def _infer_head_count(module) -> Optional[int]:
        for attr in ("num_heads", "num_attention_heads", "n_head", "n_heads"):
            value = getattr(module, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        config = getattr(module, "config", None)
        if config is not None:
            for attr in ("num_attention_heads", "num_heads", "n_head", "n_heads"):
                value = getattr(config, attr, None)
                if isinstance(value, int) and value > 0:
                    return value
        head_dim = getattr(module, "head_dim", None)
        if isinstance(head_dim, int) and head_dim > 0:
            for proj_attr, dim_attr in (("o_proj", "in_features"), ("q_proj", "out_features")):
                proj = getattr(module, proj_attr, None)
                dim = getattr(proj, dim_attr, None) if proj is not None else None
                if isinstance(dim, int) and dim > 0 and dim % head_dim == 0:
                    return dim // head_dim
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
    *,
    strict_attention_heads: bool = False,
    hook_debug_counters: Optional[Dict[str, Dict[int, int]]] = None,
):
    applier = HFHookApplier(
        model,
        strict_attention_heads=strict_attention_heads,
        hook_debug_counters=hook_debug_counters,
    )
    try:
        if attention_configs:
            applier.apply_attention_hooks(attention_configs)
        if neuron_configs:
            applier.apply_neuron_hooks(neuron_configs)
        yield
    finally:
        applier.clear()

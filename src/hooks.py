"""Hook configuration primitives for attention heads and MLP neurons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _ensure_index(within: Sequence[float], index: int) -> None:
    if index < 0 or index >= len(within):
        raise IndexError(f"Index {index} out of bounds for sequence of length {len(within)}")


@dataclass
class AttentionHookConfig:
    """Describe how to scale a specific attention head or entire module output."""

    layer: int
    head: Optional[int] = None
    scale: float = 1.0
    downscale_others: Optional[float] = None

    def apply(self, head_outputs: Sequence[float]) -> List[float]:
        data = list(head_outputs)
        if self.head is None:
            return [value * self.scale for value in data]
        _ensure_index(data, self.head)
        updated: List[float] = []
        for idx, value in enumerate(data):
            if idx == self.head:
                updated.append(value * self.scale)
            else:
                factor = self.downscale_others if self.downscale_others is not None else 1.0
                updated.append(value * factor)
        return updated


@dataclass
class NeuronHookConfig:
    """Describe how to scale an MLP neuron activation."""

    layer: int
    neuron_index: int
    scale: float = 1.0

    def apply(self, activations: Sequence[float]) -> List[float]:
        data = list(activations)
        _ensure_index(data, self.neuron_index)
        data[self.neuron_index] = data[self.neuron_index] * self.scale
        return data


@dataclass
class HookManager:
    """Apply configured hooks to abstract tensors (lists of floats in tests)."""

    attention_hooks: Iterable[AttentionHookConfig] = field(default_factory=list)
    neuron_hooks: Iterable[NeuronHookConfig] = field(default_factory=list)

    def apply_attention(self, layer: int, head_outputs: Sequence[float]) -> List[float]:
        configs = [cfg for cfg in self.attention_hooks if cfg.layer == layer]
        if not configs:
            return list(head_outputs)
        base_scale, head_scales, downscale = summarize_attention_configs(configs)
        data = list(head_outputs)
        default_scale = base_scale * (downscale if downscale is not None else 1.0)
        scaled = [value * default_scale for value in data]
        for head, factor in head_scales.items():
            _ensure_index(data, head)
            scaled[head] = data[head] * base_scale * factor
        return scaled

    def apply_neurons(self, layer: int, activations: Sequence[float]) -> List[float]:
        updated = list(activations)
        for cfg in self.neuron_hooks:
            if cfg.layer == layer:
                updated = cfg.apply(updated)
        return updated


def _merge_downscale(current: Optional[float], new: Optional[float]) -> Optional[float]:
    if new is None:
        return current
    if current is None:
        return new
    if abs(current - new) <= 1e-9:
        return current
    raise ValueError(
        f"Conflicting downscale factors: existing={current}, new={new}."
    )


def summarize_attention_configs(
    configs: Iterable[AttentionHookConfig],
) -> Tuple[float, Dict[int, float], Optional[float]]:
    base_scale = 1.0
    head_scales: Dict[int, float] = {}
    downscale: Optional[float] = None
    for cfg in configs:
        if cfg.head is None:
            base_scale *= cfg.scale
        else:
            head_scales[cfg.head] = head_scales.get(cfg.head, 1.0) * cfg.scale
            downscale = _merge_downscale(downscale, cfg.downscale_others)
    return base_scale, head_scales, downscale

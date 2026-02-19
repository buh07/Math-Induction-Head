"""Staged ablation runner for Week 2 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

NumberVector = Sequence[float]
BaselineFn = Callable[[NumberVector], List[float]]


def mean_baseline(values: NumberVector) -> List[float]:
    if not values:
        return []
    mean_val = sum(values) / len(values)
    return [mean_val for _ in values]


def zero_baseline(values: NumberVector) -> List[float]:
    return [0.0 for _ in values]


def identity_baseline(values: NumberVector) -> List[float]:
    return list(values)


DEFAULT_BASELINES: Dict[str, BaselineFn] = {
    "mean": mean_baseline,
    "zero": zero_baseline,
    "identity": identity_baseline,
}


@dataclass
class AblationStage:
    name: str
    layers: Sequence[int]
    baseline: str = "mean"


class StagedAblationRunner:
    """Orchestrate multi-stage ablations with configurable baselines."""

    def __init__(self, baselines: Mapping[str, BaselineFn] | None = None):
        self.baselines = dict(baselines or DEFAULT_BASELINES)

    def run(
        self,
        activations: Mapping[int, NumberVector],
        stages: Iterable[AblationStage],
    ) -> Dict[str, Dict[str, List[float]]]:
        results: Dict[str, Dict[str, List[float]]] = {}
        for stage in stages:
            if stage.baseline not in self.baselines:
                raise KeyError(f"Unknown baseline '{stage.baseline}'")
            baseline_fn = self.baselines[stage.baseline]
            stage_result: Dict[str, List[float]] = {}
            for layer in stage.layers:
                if layer not in activations:
                    raise KeyError(f"Layer {layer} missing from activations")
                stage_result[str(layer)] = baseline_fn(activations[layer])
            results[stage.name] = stage_result
        return results

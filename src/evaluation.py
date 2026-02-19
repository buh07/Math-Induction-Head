"""Evaluation helpers for baselines and intervention sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable, Dict, List, Mapping, Sequence

from .datasets import DatasetBundle


EvaluationFn = Callable[[List[str], int], float]
SweepFn = Callable[[Dict[str, float]], float]


@dataclass
class BaselineReport:
    dataset_name: str
    dataset_hash: str
    scores: List[float]

    @property
    def mean_score(self) -> float:
        return mean(self.scores) if self.scores else 0.0

    @property
    def std_dev(self) -> float:
        if not self.scores:
            return 0.0
        mu = self.mean_score
        variance = sum((score - mu) ** 2 for score in self.scores) / len(self.scores)
        return variance ** 0.5


class BaselineEvaluator:
    """Run multiple evaluations on a dataset to estimate stability."""

    def __init__(self, evaluation_fn: EvaluationFn):
        self.evaluation_fn = evaluation_fn

    def run(self, bundle: DatasetBundle, repeats: int = 3, seed_offset: int = 0) -> BaselineReport:
        scores = [
            self.evaluation_fn(bundle.prompts, seed_offset + idx) for idx in range(repeats)
        ]
        return BaselineReport(
            dataset_name=bundle.name,
            dataset_hash=bundle.dataset_hash,
            scores=scores,
        )


@dataclass
class SweepResult:
    params: Dict[str, float]
    score: float


def run_parameter_sweep(
    grid: Mapping[str, Sequence[float]],
    evaluation_fn: SweepFn,
) -> List[SweepResult]:
    keys = list(grid.keys())
    if not keys:
        return []
    sequences = [list(grid[key]) for key in keys]

    results: List[SweepResult] = []

    def _backtrack(idx: int, current: Dict[str, float]) -> None:
        if idx == len(keys):
            score = evaluation_fn(current)
            results.append(SweepResult(params=current.copy(), score=score))
            return
        key = keys[idx]
        for value in sequences[idx]:
            current[key] = value
            _backtrack(idx + 1, current)
        current.pop(key, None)

    _backtrack(0, {})
    return results

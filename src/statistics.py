"""Statistical summaries for validation reports."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


def _bootstrap_samples(data: Sequence[float], num_samples: int, rng: random.Random) -> List[float]:
    return [
        sum(rng.choice(data) for _ in range(len(data))) / len(data)
        for _ in range(num_samples)
    ]


def quantile_index(sample_count: int, quantile: float) -> int:
    """Return a stable 0-based quantile index using floor(q * (n - 1))."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    q = min(max(float(quantile), 0.0), 1.0)
    return int(math.floor(q * (sample_count - 1)))


@dataclass
class StatisticalSummary:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    effect_size: float

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "ci": [self.ci_low, self.ci_high],
            "effect_size": self.effect_size,
        }


def summarize(
    baseline_scores: Sequence[float],
    intervention_scores: Sequence[float],
    num_bootstrap: int = 1000,
    seed: int = 0,
) -> StatisticalSummary:
    if not baseline_scores or not intervention_scores:
        raise ValueError("Scores must be non-empty")
    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    intervention_mean = sum(intervention_scores) / len(intervention_scores)
    baseline_var = sum((x - baseline_mean) ** 2 for x in baseline_scores) / len(baseline_scores)
    intervention_var = sum((x - intervention_mean) ** 2 for x in intervention_scores) / len(
        intervention_scores
    )
    std = math.sqrt(intervention_var)
    effect_size = (intervention_mean - baseline_mean) / math.sqrt(
        (baseline_var + intervention_var) / 2
    )

    rng = random.Random(seed)
    bootstrap_data = _bootstrap_samples(intervention_scores, num_bootstrap, rng)
    bootstrap_data.sort()
    ci_low = bootstrap_data[quantile_index(num_bootstrap, 0.025)]
    ci_high = bootstrap_data[quantile_index(num_bootstrap, 0.975)]
    return StatisticalSummary(
        mean=intervention_mean,
        std=std,
        ci_low=ci_low,
        ci_high=ci_high,
        effect_size=effect_size,
    )

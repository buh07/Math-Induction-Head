"""Power-analysis helpers for Phase 2 preregistration artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Optional


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def required_n_per_arm_two_proportion(
    *,
    baseline_rate: float,
    effect_size: float,
    alpha: float,
    power: float,
) -> int:
    """Approximate per-arm sample size for a two-sided two-proportion z-test."""
    baseline_rate = min(max(float(baseline_rate), 1e-6), 1 - 1e-6)
    effect_size = abs(float(effect_size))
    if effect_size <= 0:
        return 0
    p1 = baseline_rate
    p2 = min(max(p1 + effect_size, 1e-6), 1 - 1e-6)
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2.0)
    z_beta = NormalDist().inv_cdf(power)
    variance = p1 * (1 - p1) + p2 * (1 - p2)
    n = ((z_alpha + z_beta) ** 2) * variance / (effect_size**2)
    return int(n + 0.999999)


def build_power_analysis_report(
    preregistration: Mapping[str, Any],
    *,
    dataset_manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    alpha = _safe_float(preregistration.get("alpha"), 0.05)
    target_power = _safe_float(preregistration.get("target_power"), 0.8)
    mesi_map = preregistration.get("minimum_effect_size_of_interest")
    if not isinstance(mesi_map, dict) or not mesi_map:
        mesi_map = {"delta_vs_random_accuracy_all": 0.05}

    primary_grid = preregistration.get("primary_comparison_grid")
    if not isinstance(primary_grid, dict):
        primary_grid = {}
    interventions = primary_grid.get("interventions") if isinstance(primary_grid.get("interventions"), list) else ["ablation", "amplification"]
    k_values = primary_grid.get("k_values") if isinstance(primary_grid.get("k_values"), list) else [5, 10]
    scales = primary_grid.get("scales") if isinstance(primary_grid.get("scales"), list) else [0.0, 1.25]

    n_primary = max(1, len(interventions) * len(k_values) * len(scales) * len(mesi_map))
    multiplicity = preregistration.get("multiplicity_policy") if isinstance(preregistration.get("multiplicity_policy"), dict) else {}
    method = str(multiplicity.get("method", "bh_fdr"))
    if method == "none":
        effective_alpha = alpha
    else:
        # Planning-time conservative proxy for multiple tests.
        effective_alpha = alpha / n_primary

    assumptions = preregistration.get("assumptions") if isinstance(preregistration.get("assumptions"), dict) else {}
    baseline_rate = _safe_float(assumptions.get("baseline_accuracy"), 0.4)

    planned = preregistration.get("planned_sample_sizes") if isinstance(preregistration.get("planned_sample_sizes"), dict) else {}
    counts_per_bucket = _safe_int(planned.get("counts_per_bucket"), 0)
    operator_planned = planned.get("operators") if isinstance(planned.get("operators"), dict) else {}

    comparisons: List[Dict[str, Any]] = []
    for metric_name, mesi in sorted(mesi_map.items()):
        required_n = required_n_per_arm_two_proportion(
            baseline_rate=baseline_rate,
            effect_size=_safe_float(mesi, 0.05),
            alpha=effective_alpha,
            power=target_power,
        )
        operator_rows = []
        operator_counts = {}
        if dataset_manifest and isinstance(dataset_manifest.get("counts_by_operator"), dict):
            operator_counts = {str(k): int(v) for k, v in dataset_manifest.get("counts_by_operator", {}).items()}
        operators = sorted(set(operator_planned.keys()) | set(operator_counts.keys()))
        for operator in operators:
            planned_n = _safe_int(operator_planned.get(operator), counts_per_bucket)
            observed_n = int(operator_counts.get(operator, 0)) if operator_counts else None
            operator_rows.append(
                {
                    "operator": operator,
                    "planned_per_bucket": planned_n,
                    "observed_total_examples": observed_n,
                    "meets_required_per_arm": planned_n >= required_n,
                }
            )

        comparisons.append(
            {
                "metric": metric_name,
                "minimum_effect_size_of_interest": _safe_float(mesi, 0.05),
                "assumed_baseline_accuracy": baseline_rate,
                "required_n_per_arm": required_n,
                "operator_coverage": operator_rows,
            }
        )

    return {
        "schema_version": "power_analysis_report_v1",
        "generated_at_utc": _timestamp_utc(),
        "preregistration_schema_version": preregistration.get("schema_version"),
        "alpha": alpha,
        "target_power": target_power,
        "effective_alpha_for_planning": effective_alpha,
        "multiplicity_method": method,
        "n_primary_comparisons": n_primary,
        "primary_grid": {
            "interventions": list(interventions),
            "k_values": list(k_values),
            "scales": list(scales),
        },
        "comparisons": comparisons,
    }

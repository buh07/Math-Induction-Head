"""Parser-audit helpers for arithmetic extraction validity checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .experiment_runner import _extract_int, _extract_int_relaxed_last_number, _extract_int_strict_final


@dataclass(frozen=True)
class ParserAuditSample:
    source_run: str
    dataset: str
    operator: str
    bucket: str
    prompt: str
    output: str
    target: Any


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float


def _numeric_equal(a: Any, b: Any, *, tol: float = 1e-9) -> bool:
    a_num = _coerce_number(a)
    b_num = _coerce_number(b)
    if a_num is None or b_num is None:
        return False
    return abs(a_num - b_num) <= tol


def collect_parser_audit_samples_from_intervention_runs(
    runs: Mapping[str, Dict[str, Any]],
    *,
    per_dataset_limit: int = 16,
) -> List[ParserAuditSample]:
    samples: List[ParserAuditSample] = []
    for run_name, run_payload in sorted(runs.items()):
        for condition in run_payload.get("results", []):
            cond = condition.get("condition", {})
            condition_label = str(cond.get("component_set_name") or "")
            for dataset_name, dataset_payload in (condition.get("datasets") or {}).items():
                operator = str(dataset_payload.get("operator") or "")
                bucket = str(dataset_payload.get("bucket") or "")
                sample_rows = list(dataset_payload.get("prediction_samples") or [])
                for row in sample_rows[: max(0, int(per_dataset_limit))]:
                    output = str(row.get("output") or row.get("generated") or "")
                    prompt = str(row.get("prompt") or "")
                    target = row.get("target")
                    samples.append(
                        ParserAuditSample(
                            source_run=f"{run_name}:{condition_label}",
                            dataset=str(dataset_name),
                            operator=operator,
                            bucket=bucket,
                            prompt=prompt,
                            output=output,
                            target=target,
                        )
                    )
    return samples


def build_parser_audit_report(
    samples: Sequence[ParserAuditSample],
    *,
    source_label: str,
    adjudication_cap: int = 64,
) -> Dict[str, Any]:
    both_none = 0
    both_equal = 0
    default_only = 0
    strict_only = 0
    both_different = 0
    default_parsed = 0
    strict_parsed = 0
    relaxed_parsed = 0
    default_correct = 0
    strict_correct = 0
    relaxed_correct = 0
    target_available = 0
    adjudication: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        default_val = _extract_int(sample.output)
        strict_val = _extract_int_strict_final(sample.output)
        relaxed_val = _extract_int_relaxed_last_number(sample.output)
        default_has = default_val is not None
        strict_has = strict_val is not None
        relaxed_has = relaxed_val is not None
        if default_has:
            default_parsed += 1
        if strict_has:
            strict_parsed += 1
        if relaxed_has:
            relaxed_parsed += 1

        if not default_has and not strict_has:
            both_none += 1
        elif default_has and strict_has and _numeric_equal(default_val, strict_val):
            both_equal += 1
        elif default_has and not strict_has:
            default_only += 1
        elif strict_has and not default_has:
            strict_only += 1
        else:
            both_different += 1

        target_num = _coerce_number(sample.target)
        if target_num is not None:
            target_available += 1
            if _numeric_equal(default_val, target_num):
                default_correct += 1
            if _numeric_equal(strict_val, target_num):
                strict_correct += 1
            if _numeric_equal(relaxed_val, target_num):
                relaxed_correct += 1

        disagreement = not (
            (default_val is None and strict_val is None)
            or (default_val is not None and strict_val is not None and _numeric_equal(default_val, strict_val))
        )
        if disagreement and len(adjudication) < max(0, int(adjudication_cap)):
            adjudication.append(
                {
                    "sample_index": idx,
                    "source_run": sample.source_run,
                    "dataset": sample.dataset,
                    "operator": sample.operator,
                    "bucket": sample.bucket,
                    "target": sample.target,
                    "default_parse": default_val,
                    "strict_final_numeric_parse": strict_val,
                    "relaxed_last_number_parse": relaxed_val,
                    "prompt": sample.prompt,
                    "output": sample.output,
                }
            )

    total = len(samples)
    agreements = both_none + both_equal
    disagreements = total - agreements
    parse_rate_default = (default_parsed / total) if total else 0.0
    parse_rate_strict = (strict_parsed / total) if total else 0.0
    parse_rate_relaxed = (relaxed_parsed / total) if total else 0.0
    accuracy_default = (default_correct / target_available) if target_available else None
    accuracy_strict = (strict_correct / target_available) if target_available else None
    accuracy_relaxed = (relaxed_correct / target_available) if target_available else None

    return {
        "schema_version": "parser_audit_v1",
        "source": source_label,
        "generated_at_utc": _timestamp_utc(),
        "sample_count": total,
        "target_available_count": target_available,
        "parse_mode_agreement": {
            "agreement_rate": (agreements / total) if total else 1.0,
            "disagreement_rate": (disagreements / total) if total else 0.0,
            "both_none": both_none,
            "both_equal_numeric": both_equal,
            "default_only": default_only,
            "strict_only": strict_only,
            "both_different_numeric": both_different,
        },
        "parse_metrics": {
            "default": {
                "parse_rate": parse_rate_default,
                "accuracy": accuracy_default,
            },
            "strict_final_numeric": {
                "parse_rate": parse_rate_strict,
                "accuracy": accuracy_strict,
            },
            "relaxed_last_number": {
                "parse_rate": parse_rate_relaxed,
                "accuracy": accuracy_relaxed,
            },
            "delta": {
                "strict_minus_default_parse_rate": parse_rate_strict - parse_rate_default,
                "strict_minus_default_accuracy": (
                    (accuracy_strict - accuracy_default)
                    if accuracy_default is not None and accuracy_strict is not None
                    else None
                ),
                "strict_minus_relaxed_parse_rate": parse_rate_strict - parse_rate_relaxed,
                "strict_minus_relaxed_accuracy": (
                    (accuracy_strict - accuracy_relaxed)
                    if accuracy_strict is not None and accuracy_relaxed is not None
                    else None
                ),
            },
        },
        "ambiguity_rate": (disagreements / total) if total else 0.0,
        "adjudication_samples": adjudication,
    }

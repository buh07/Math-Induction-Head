"""Runtime batch-size autotuning utilities for Phase 2 GPU stages."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


NumericTree = Mapping[str, Any] | Dict[str, Any]
ProbeFn = Callable[[int], NumericTree]


@dataclass(frozen=True)
class BatchAutotuneConfig:
    enabled: bool = True
    min_batch_size: int = 4
    max_batch_size: Optional[int] = None
    growth_factor: float = 1.5
    safety_backoff: float = 0.85
    max_retries_after_oom: int = 3
    equivalence_check_enabled: bool = True
    max_abs_logit_diff: float = 1e-4
    max_metric_diff: float = 1e-4


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "out of memory" in text:
        return True
    if "cuda" in text and "memory" in text:
        return True
    return exc.__class__.__name__.lower().endswith("outofmemoryerror")


def _cleanup_after_oom() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def is_oom_error(exc: BaseException) -> bool:
    """Public helper for OOM classification used by stage runners."""
    return _is_oom_error(exc)


def cleanup_after_oom() -> None:
    """Public helper to reclaim CUDA/CPU memory after OOM."""
    _cleanup_after_oom()


def _flatten_numeric_tree(payload: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numeric_tree(value, key_path))
        return out
    if isinstance(payload, (list, tuple)):
        for idx, value in enumerate(payload):
            key_path = f"{prefix}[{idx}]"
            out.update(_flatten_numeric_tree(value, key_path))
        return out
    if isinstance(payload, bool):
        return out
    try:
        numeric = float(payload)
    except Exception:
        return out
    if math.isfinite(numeric):
        out[prefix or "root"] = numeric
    return out


def _max_abs_diff(reference: Any, candidate: Any, *, key_filter: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    ref = _flatten_numeric_tree(reference)
    cand = _flatten_numeric_tree(candidate)
    if key_filter is not None:
        allowed_prefixes = [str(k) for k in key_filter]
        ref = {k: v for k, v in ref.items() if any(k == pref or k.startswith(f"{pref}[") or k.startswith(f"{pref}.") for pref in allowed_prefixes)}
        cand = {k: v for k, v in cand.items() if any(k == pref or k.startswith(f"{pref}[") or k.startswith(f"{pref}.") for pref in allowed_prefixes)}
    common = sorted(set(ref.keys()) & set(cand.keys()))
    if not common:
        return {"max_abs_diff": None, "n_common": 0, "missing_in_candidate": sorted(set(ref.keys()) - set(cand.keys()))}
    max_key = common[0]
    max_diff = abs(ref[max_key] - cand[max_key])
    for key in common[1:]:
        diff = abs(ref[key] - cand[key])
        if diff > max_diff:
            max_diff = diff
            max_key = key
    return {
        "max_abs_diff": float(max_diff),
        "max_abs_diff_key": max_key,
        "n_common": len(common),
        "missing_in_candidate": sorted(set(ref.keys()) - set(cand.keys())),
        "missing_in_reference": sorted(set(cand.keys()) - set(ref.keys())),
    }


def _run_probe_with_status(run_probe_fn: ProbeFn, batch_size: int) -> Dict[str, Any]:
    t0 = time.time()
    try:
        payload = run_probe_fn(int(batch_size))
        return {
            "status": "ok",
            "batch_size": int(batch_size),
            "probe_seconds": float(time.time() - t0),
            "payload": payload,
        }
    except Exception as exc:  # pragma: no cover - error path depends on runtime backend
        oom = _is_oom_error(exc)
        if oom:
            _cleanup_after_oom()
        return {
            "status": "oom" if oom else "error",
            "batch_size": int(batch_size),
            "probe_seconds": float(time.time() - t0),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }


def autotune_batch_size(
    *,
    stage_name: str,
    device: str,
    baseline_batch_size: int,
    run_probe_fn: ProbeFn,
    config: BatchAutotuneConfig,
) -> Dict[str, Any]:
    """Find a safe, high-throughput batch size and optionally verify equivalence.

    `run_probe_fn` must execute a deterministic probe at a given batch size and return a
    numeric-like payload (nested dict/list/scalars) that can be diffed for equivalence.
    """

    baseline_batch_size = max(int(baseline_batch_size), 1)
    min_bs = max(int(config.min_batch_size), 1)
    growth = max(float(config.growth_factor), 1.1)
    safety_backoff = min(max(float(config.safety_backoff), 0.1), 1.0)
    max_retries = max(int(config.max_retries_after_oom), 0)
    max_bs = int(config.max_batch_size) if config.max_batch_size is not None else None
    if max_bs is None:
        max_bs = max(baseline_batch_size, min_bs) * 16
    max_bs = max(max_bs, min_bs)

    result: Dict[str, Any] = {
        "stage": stage_name,
        "device": device,
        "enabled": bool(config.enabled),
        "baseline_batch_size": baseline_batch_size,
        "min_batch_size": min_bs,
        "max_batch_size": max_bs,
        "growth_factor": growth,
        "safety_backoff": safety_backoff,
        "attempts": [],
        "status": "disabled" if not config.enabled else "ok",
        "tuned_batch_size": baseline_batch_size,
        "equivalence_check": {
            "enabled": bool(config.equivalence_check_enabled),
            "passed": None,
            "max_abs_logit_diff": None,
            "max_metric_diff": None,
            "thresholds": {
                "max_abs_logit_diff": float(config.max_abs_logit_diff),
                "max_metric_diff": float(config.max_metric_diff),
            },
        },
    }
    if not config.enabled:
        return result

    start_bs = min(max(baseline_batch_size, min_bs), max_bs)
    baseline_probe = _run_probe_with_status(run_probe_fn, start_bs)
    result["attempts"].append({k: v for k, v in baseline_probe.items() if k != "payload"})
    if baseline_probe["status"] != "ok":
        # Try reducing to min batch before giving up.
        fallback_bs = min_bs
        retries = 0
        probe = baseline_probe
        while probe["status"] != "ok" and fallback_bs > 1 and retries < max_retries:
            fallback_bs = max(1, int(math.floor(fallback_bs * 0.5)))
            probe = _run_probe_with_status(run_probe_fn, fallback_bs)
            result["attempts"].append({k: v for k, v in probe.items() if k != "payload"})
            retries += 1
        if probe["status"] != "ok":
            result["status"] = "fallback_baseline_failed"
            result["fallback_reason"] = "probe_failed_for_all_attempted_batch_sizes"
            result["tuned_batch_size"] = baseline_batch_size
            return result
        start_bs = int(probe["batch_size"])
        baseline_probe = probe

    best_ok_bs = int(start_bs)
    best_ok_payload = baseline_probe.get("payload")
    had_failure = False

    current = best_ok_bs
    while current < max_bs:
        next_bs = min(max_bs, max(current + 1, int(math.ceil(current * growth))))
        probe = _run_probe_with_status(run_probe_fn, next_bs)
        result["attempts"].append({k: v for k, v in probe.items() if k != "payload"})
        if probe["status"] == "ok":
            best_ok_bs = int(next_bs)
            best_ok_payload = probe.get("payload")
            current = next_bs
            continue
        had_failure = True
        break

    tuned = best_ok_bs
    if had_failure:
        backed_off = max(min_bs, int(math.floor(best_ok_bs * safety_backoff)))
        tuned = min(best_ok_bs, max(backed_off, min_bs))
    result["tuned_batch_size"] = int(tuned)

    if not config.equivalence_check_enabled:
        return result
    if tuned == start_bs:
        result["equivalence_check"]["passed"] = True
        result["equivalence_check"]["reason"] = "baseline_equals_tuned"
        return result

    tuned_probe = _run_probe_with_status(run_probe_fn, tuned)
    result["attempts"].append({k: v for k, v in tuned_probe.items() if k != "payload"})
    if tuned_probe["status"] != "ok":
        result["status"] = "fallback_tuned_probe_failed"
        result["fallback_reason"] = "tuned_probe_failed"
        result["tuned_batch_size"] = int(start_bs)
        result["equivalence_check"]["passed"] = False
        result["equivalence_check"]["failure_reason"] = "tuned_probe_failed"
        return result

    ref_payload = baseline_probe.get("payload")
    cand_payload = tuned_probe.get("payload")
    logit_diff = _max_abs_diff(ref_payload, cand_payload, key_filter=["logit_signature"])
    metric_diff = _max_abs_diff(ref_payload, cand_payload, key_filter=["metric_signature"])
    logit_max = float(logit_diff["max_abs_diff"] or 0.0)
    metric_max = float(metric_diff["max_abs_diff"] or 0.0)
    passed = bool(
        logit_diff["n_common"] > 0
        and metric_diff["n_common"] > 0
        and logit_max <= float(config.max_abs_logit_diff)
        and metric_max <= float(config.max_metric_diff)
    )
    result["equivalence_check"].update(
        {
            "passed": passed,
            "max_abs_logit_diff": logit_max,
            "max_metric_diff": metric_max,
            "logit_diff_detail": logit_diff,
            "metric_diff_detail": metric_diff,
        }
    )
    if not passed:
        result["status"] = "fallback_equivalence_failed"
        result["fallback_reason"] = "equivalence_check_failed"
        result["tuned_batch_size"] = int(start_bs)
    return result


__all__ = [
    "BatchAutotuneConfig",
    "autotune_batch_size",
    "is_oom_error",
    "cleanup_after_oom",
]

#!/usr/bin/env python3
"""Audit legacy Phase 2 gate summaries without mutating original artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_REQUIRED_GATES = [
    "dataset_bucket_gate",
    "localization_validity_gate",
    "operator_specificity_gate",
    "intervention_sanity_gate",
    "cot_gating_evidence_gate",
]


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit legacy Phase 2 run artifacts and emit readiness overrides.")
    parser.add_argument(
        "--results-root",
        default="results/phase2",
        help="Directory containing phase2 run folders.",
    )
    parser.add_argument(
        "--run-path",
        action="append",
        default=None,
        help="Optional run directory relative to results root (can be repeated).",
    )
    parser.add_argument(
        "--index-output",
        default="legacy_audit_index.json",
        help="Index filename written under results root.",
    )
    parser.add_argument(
        "--fail-if-v1-missing-sidecar",
        action="store_true",
        help="Exit non-zero if any v1 run is missing legacy_audit.json.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gather_run_dirs(results_root: Path, requested: List[str] | None) -> List[Path]:
    if requested:
        return [results_root / rel for rel in requested]
    run_dirs: List[Path] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "phase2_gate_summary.json").exists():
            run_dirs.append(child)
    return run_dirs


def _audit_run(run_dir: Path) -> Dict[str, Any] | None:
    gate_path = run_dir / "phase2_gate_summary.json"
    if not gate_path.exists():
        return None
    gate = _load_json(gate_path)
    schema = str(gate.get("schema_version") or "")
    if not schema.endswith("_v1"):
        return None

    phases = gate.get("phases") or {}
    overall = gate.get("overall") or {}
    required_policy = gate.get("required_gates_policy") or {}
    required = required_policy.get("required_for_readiness")
    if not isinstance(required, list) or not required:
        required = list(DEFAULT_REQUIRED_GATES)

    blocking_reasons: List[str] = []
    known_defects: List[str] = []

    for gate_name in required:
        if not bool((phases.get(gate_name) or {}).get("passes")):
            blocking_reasons.append(f"required_gate_failed:{gate_name}")

    specificity = phases.get("operator_specificity_gate") or {}
    for check in specificity.get("checks") or []:
        if check.get("target_operator") and not bool(check.get("passes")):
            blocking_reasons.append(f"specificity_check_failed:{check.get('run')}")
    if bool(specificity.get("passes")):
        checks = list(specificity.get("checks") or [])
        if checks:
            best_target = checks[0].get("best_target") or {}
            condition = str(best_target.get("condition") or "")
            if ":bottom" in condition:
                known_defects.append("specificity_best_target_from_non_primary_set")

    localization = phases.get("localization_validity_gate") or {}
    if bool(localization.get("passes")):
        for check in localization.get("checks") or []:
            if float(check.get("effect_nonzero_rate_max", 0.0) or 0.0) <= 0.0 and float(
                check.get("answer_token_prob_delta_abs_max", 0.0) or 0.0
            ) <= 1e-6:
                known_defects.append("localization_gate_passed_with_near_zero_effect")
                break

    if bool(overall.get("ready_for_multimodel_next_tranche")) and blocking_reasons:
        known_defects.append("legacy_readiness_true_despite_failed_required_gates")

    audited_ready = not blocking_reasons
    return {
        "schema_version": "legacy_audit_v1",
        "run_path": str(run_dir),
        "legacy_schema_version": schema,
        "legacy_readiness_value": bool(overall.get("ready_for_multimodel_next_tranche")),
        "audited_ready_for_multimodel": bool(audited_ready),
        "blocking_reasons": sorted(set(blocking_reasons)),
        "known_legacy_defects": sorted(set(known_defects)),
        "generated_at_utc": _timestamp_utc(),
    }


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Missing results root: {results_root}")

    run_dirs = _gather_run_dirs(results_root, args.run_path)
    index_rows: List[Dict[str, Any]] = []
    missing_sidecar: List[str] = []

    for run_dir in run_dirs:
        gate_path = run_dir / "phase2_gate_summary.json"
        if not gate_path.exists():
            continue
        gate = _load_json(gate_path)
        schema = str(gate.get("schema_version") or "")
        sidecar_path = run_dir / "legacy_audit.json"
        if args.fail_if_v1_missing_sidecar and schema.endswith("_v1") and not sidecar_path.exists():
            missing_sidecar.append(str(run_dir))
        audit_payload = _audit_run(run_dir)
        if audit_payload is None:
            continue
        sidecar_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
        index_rows.append(
            {
                "run_path": str(run_dir),
                "schema_version": audit_payload["legacy_schema_version"],
                "legacy_readiness_value": audit_payload["legacy_readiness_value"],
                "audited_ready_for_multimodel": audit_payload["audited_ready_for_multimodel"],
                "sidecar": str(sidecar_path.relative_to(results_root)),
            }
        )

    index_payload = {
        "schema_version": "legacy_audit_index_v1",
        "generated_at_utc": _timestamp_utc(),
        "results_root": str(results_root),
        "runs": index_rows,
    }
    index_path = results_root / args.index_output
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    print(str(index_path))

    if args.fail_if_v1_missing_sidecar and missing_sidecar:
        joined = "\n".join(missing_sidecar)
        raise SystemExit(f"Missing legacy_audit.json for v1 run(s):\n{joined}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2.run_operator_bottleneck_suite import (  # noqa: E402
    _evaluate_cot_gate,
    _evaluate_intervention_sanity_gate,
    _evaluate_localization_gate,
    _evaluate_specificity_gate,
    _phase2_gate_template,
    _required_gates_for_readiness,
)
from src.operator_interventions import (  # noqa: E402
    build_cross_operator_specificity_matrix,
    build_specificity_matrix_from_intervention_results,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge operator-sharded Phase 2 outputs into a canonical run layout.")
    parser.add_argument(
        "--shard-dirs",
        nargs="+",
        required=True,
        help="Shard output directories produced by run_operator_bottleneck_suite.py",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Merged output directory.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_effective_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(cfg))
    runtime = out.get("runtime", {})
    if isinstance(runtime, dict):
        for key in ("devices", "batch_size", "operator_filter", "operator_shard_mode"):
            runtime.pop(key, None)
        runtime.pop("batch_autotune", None)
    datasets = out.get("datasets", {})
    if isinstance(datasets, dict):
        datasets.pop("operator_buckets", None)
    return out


def _config_hash(cfg: Mapping[str, Any]) -> str:
    payload = json.dumps(_canonical_effective_config(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _operator_bucket_signature_from_map(op_map: Mapping[str, Any]) -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, Tuple[str, ...]] = {}
    for op, buckets in (op_map or {}).items():
        if isinstance(buckets, list):
            out[str(op)] = tuple(sorted(str(b) for b in buckets))
    return out


def _validate_operator_bucket_signatures(shard_payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Validate per-operator bucket definitions and return signature metadata."""
    signatures: Dict[str, Dict[str, Any]] = {}
    for payload in shard_payloads:
        shard_dir = str(payload["dir"])
        manifest_sig = _operator_bucket_signature_from_map(
            (
                payload.get("dataset_manifest", {})
                .get("config_snapshot", {})
                .get("operator_buckets", {})
            )
        )
        cfg_sig = _operator_bucket_signature_from_map(
            (
                payload.get("run_manifest", {})
                .get("effective_config", {})
                .get("datasets", {})
                .get("operator_buckets", {})
            )
        )
        # Ensure each shard's own metadata is internally consistent.
        for op, sig in cfg_sig.items():
            if op in manifest_sig and manifest_sig[op] != sig:
                raise ValueError(
                    f"Shard operator bucket mismatch for operator '{op}' in {shard_dir}: "
                    f"dataset_manifest={manifest_sig[op]} effective_config={sig}"
                )
        # Cross-shard consistency check for duplicated operators.
        for op, sig in manifest_sig.items():
            entry = signatures.setdefault(
                op,
                {
                    "signature": list(sig),
                    "sources": [],
                },
            )
            prev_sig = tuple(entry.get("signature", []))
            if prev_sig and prev_sig != sig:
                raise ValueError(
                    f"Operator bucket signature mismatch for operator '{op}' across shards: "
                    f"{prev_sig} vs {sig}"
                )
            entry["sources"].append(shard_dir)
    return signatures


def _require_same_schema(label: str, payloads: Sequence[Mapping[str, Any]]) -> None:
    if not payloads:
        return
    ref = payloads[0]
    ref_keys = set(ref.keys())
    for idx, payload in enumerate(payloads[1:], start=1):
        keys = set(payload.keys())
        if keys != ref_keys:
            raise ValueError(f"{label} schema mismatch between shard 0 and shard {idx}: {sorted(ref_keys ^ keys)}")
        for key in sorted(ref_keys):
            if type(payload.get(key)) is not type(ref.get(key)):  # noqa: E721
                raise ValueError(f"{label} type mismatch for key '{key}' between shard 0 and shard {idx}")


def _merge_dataset_manifest(manifests: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = {
        "schema_version": manifests[0].get("schema_version", "operator_bucket_suite_v1"),
        "suite_seed": manifests[0].get("suite_seed"),
        "config_snapshot": {"operator_buckets": {}, "counts_per_bucket": None, "representation_variants": []},
        "datasets": [],
        "counts_by_operator": {},
        "counts_by_bucket": {},
    }
    by_name: Dict[str, Dict[str, Any]] = {}
    operator_buckets: Dict[str, set[str]] = {}
    rep_variants: set[str] = set()
    counts_per_bucket: Optional[int] = None
    for manifest in manifests:
        snapshot = manifest.get("config_snapshot", {})
        for op, buckets in (snapshot.get("operator_buckets") or {}).items():
            operator_buckets.setdefault(str(op), set()).update(str(b) for b in buckets)
        if counts_per_bucket is None and snapshot.get("counts_per_bucket") is not None:
            counts_per_bucket = int(snapshot.get("counts_per_bucket"))
        rep_variants.update(str(v) for v in (snapshot.get("representation_variants") or []))
        for row in manifest.get("datasets", []):
            name = str(row.get("name"))
            if name in by_name:
                if row.get("dataset_hash") != by_name[name].get("dataset_hash"):
                    raise ValueError(f"Dataset collision with mismatched hash for {name}")
                continue
            by_name[name] = dict(row)
        for op, count in (manifest.get("counts_by_operator") or {}).items():
            merged["counts_by_operator"][str(op)] = int(merged["counts_by_operator"].get(str(op), 0)) + int(count)
        for bucket, count in (manifest.get("counts_by_bucket") or {}).items():
            merged["counts_by_bucket"][str(bucket)] = int(merged["counts_by_bucket"].get(str(bucket), 0)) + int(count)
    merged["datasets"] = [by_name[name] for name in sorted(by_name.keys())]
    merged["config_snapshot"]["operator_buckets"] = {op: sorted(list(buckets)) for op, buckets in sorted(operator_buckets.items())}
    merged["config_snapshot"]["counts_per_bucket"] = counts_per_bucket
    merged["config_snapshot"]["representation_variants"] = sorted(rep_variants)
    return merged


def _merge_dataset_diagnostics(diagnostics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = {
        "schema_version": diagnostics[0].get("schema_version", "dataset_diagnostics_v1"),
        "suite_seed": diagnostics[0].get("suite_seed"),
        "counts_by_operator": {},
        "counts_by_bucket": {},
        "datasets": {},
    }
    for diag in diagnostics:
        for op, count in (diag.get("counts_by_operator") or {}).items():
            merged["counts_by_operator"][str(op)] = int(merged["counts_by_operator"].get(str(op), 0)) + int(count)
        for bucket, count in (diag.get("counts_by_bucket") or {}).items():
            merged["counts_by_bucket"][str(bucket)] = int(merged["counts_by_bucket"].get(str(bucket), 0)) + int(count)
        for name, row in (diag.get("datasets") or {}).items():
            merged["datasets"][str(name)] = dict(row)
    return merged


def _merge_localization_payloads(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    merged_runs: Dict[str, Dict[str, Any]] = {}
    merged_sets: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    calibration = {
        "enabled": True,
        "null_runs": {},
        "coverage_by_run": {},
        "derived_thresholds": {},
        "localizer_null_policy": None,
        "family_heldout_coverage_policy": None,
    }
    selection_eval_split: Dict[str, Any] = {"enabled": True, "summary_output": "phase2_selection_eval_split.json"}
    for payload in payloads:
        runs = payload.get("runs", {})
        merged_runs.update({str(k): dict(v) for k, v in runs.items()})
        comp_sets = payload.get("component_sets", {})
        for op, op_payload in comp_sets.items():
            merged_sets.setdefault(str(op), {})
            merged_sets[str(op)].update({str(k): dict(v) for k, v in op_payload.items()})
        cal = payload.get("calibration", {})
        if isinstance(cal, dict):
            calibration["null_runs"].update({str(k): str(v) for k, v in (cal.get("null_runs") or {}).items()})
            calibration["coverage_by_run"].update({str(k): dict(v) for k, v in (cal.get("coverage_by_run") or {}).items()})
            if cal.get("derived_thresholds"):
                calibration["derived_thresholds"] = dict(cal["derived_thresholds"])
            if cal.get("localizer_null_policy") is not None:
                calibration["localizer_null_policy"] = cal.get("localizer_null_policy")
            if cal.get("family_heldout_coverage_policy") is not None:
                calibration["family_heldout_coverage_policy"] = cal.get("family_heldout_coverage_policy")
        if payload.get("selection_eval_split"):
            selection_eval_split = dict(payload.get("selection_eval_split"))
    return {
        "schema_version": "phase2_localization_summary_v1",
        "runs": merged_runs,
        "component_sets": merged_sets,
        "selection_eval_split": selection_eval_split,
        "calibration": calibration,
    }


def _merge_interventions_payloads(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    runs: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        runs.update({str(k): dict(v) for k, v in (payload.get("runs") or {}).items()})
    return {
        "schema_version": "phase2_interventions_summary_v1",
        "runs": runs,
    }


def _merge_cot_payloads(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    runs: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        runs.update({str(k): dict(v) for k, v in (payload.get("runs") or {}).items()})
    return {
        "schema_version": "phase2_cot_compare_summary_v1",
        "runs": runs,
    }


def _merge_cross_operator_verify_payloads(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    runs: Dict[str, Dict[str, Any]] = {}
    sources: List[str] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key, row in (payload.get("runs") or {}).items():
            runs[str(key)] = dict(row)
        src = payload.get("source_component_sets")
        if src:
            sources.append(str(src))
    status = "ok" if runs else "not_implemented"
    return {
        "schema_version": "phase2_cross_operator_verify_v1",
        "status": status,
        "evidence_source": "post_merge_cross_operator_verify",
        "source_component_sets": sorted(set(sources)),
        "runs": runs,
    }


def _build_merged_cross_operator_matrix(intervention_runs: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    cols: set[str] = set()
    for key, payload in intervention_runs.items():
        matrix = build_specificity_matrix_from_intervention_results(payload)
        for col in matrix.get("cols", []):
            cols.add(col)
        for row_entry in matrix.get("matrix", []):
            rows[f"{key}::{row_entry['row']}"] = row_entry.get("cells", {})
    return build_cross_operator_specificity_matrix(rows=list(rows.keys()), cols=sorted(cols), values=rows)


def _max_or_default(values: Sequence[float], default: float) -> float:
    if not values:
        return float(default)
    return float(max(values))


def main() -> None:
    args = _parse_args()
    shard_dirs = [Path(p).resolve() for p in args.shard_dirs]
    if len(shard_dirs) < 2:
        raise ValueError("Provide at least two shard directories.")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    shard_payloads: List[Dict[str, Any]] = []
    for shard in shard_dirs:
        if not shard.exists():
            raise FileNotFoundError(f"Missing shard directory: {shard}")
        payload = {
            "dir": shard,
            "run_manifest": _load_json(shard / "run_manifest.json"),
            "gate_summary": _load_json(shard / "phase2_gate_summary.json"),
            "dataset_manifest": _load_json(shard / "dataset_manifest.json"),
            "dataset_diagnostics": _load_json(shard / "dataset_diagnostics.json"),
            "phase2_localization": _load_json(shard / "phase2_localization.json"),
            "phase2_interventions": _load_json(shard / "phase2_interventions.json"),
            "phase2_cot_compare": _load_json(shard / "phase2_cot_recruitment_compare.json"),
            "phase2_cross_operator_verify": (
                _load_json(shard / "phase2_cross_operator_verify.json")
                if (shard / "phase2_cross_operator_verify.json").exists()
                else {"schema_version": "phase2_cross_operator_verify_v1", "status": "not_implemented", "runs": {}}
            ),
        }
        split_path = shard / "phase2_selection_eval_split.json"
        payload["phase2_selection_eval_split"] = _load_json(split_path) if split_path.exists() else None
        shard_payloads.append(payload)

    model_names = {str(p["run_manifest"].get("effective_config", {}).get("model", {}).get("name")) for p in shard_payloads}
    if len(model_names) != 1:
        raise ValueError(f"Shard model mismatch: {sorted(model_names)}")
    seed_lists = {
        tuple(p["run_manifest"].get("effective_config", {}).get("runtime", {}).get("seeds", []))
        for p in shard_payloads
    }
    if len(seed_lists) != 1:
        raise ValueError(f"Shard seed-list mismatch: {sorted(seed_lists)}")
    cfg_hashes = {_config_hash(p["run_manifest"].get("effective_config", {})) for p in shard_payloads}
    if len(cfg_hashes) != 1:
        raise ValueError("Shard effective config mismatch (after canonicalization).")
    operator_bucket_signatures = _validate_operator_bucket_signatures(shard_payloads)

    _require_same_schema("run_manifest", [p["run_manifest"] for p in shard_payloads])
    _require_same_schema("phase2_gate_summary", [p["gate_summary"] for p in shard_payloads])
    _require_same_schema("phase2_localization", [p["phase2_localization"] for p in shard_payloads])
    _require_same_schema("phase2_interventions", [p["phase2_interventions"] for p in shard_payloads])
    _require_same_schema("phase2_cot_recruitment_compare", [p["phase2_cot_compare"] for p in shard_payloads])

    merged_dataset_manifest = _merge_dataset_manifest([p["dataset_manifest"] for p in shard_payloads])
    merged_dataset_diagnostics = _merge_dataset_diagnostics([p["dataset_diagnostics"] for p in shard_payloads])
    merged_localization = _merge_localization_payloads([p["phase2_localization"] for p in shard_payloads])
    merged_interventions = _merge_interventions_payloads([p["phase2_interventions"] for p in shard_payloads])
    merged_cot = _merge_cot_payloads([p["phase2_cot_compare"] for p in shard_payloads])
    merged_cross_verify = _merge_cross_operator_verify_payloads([p["phase2_cross_operator_verify"] for p in shard_payloads])
    merged_cross_matrix = _build_merged_cross_operator_matrix(merged_cross_verify.get("runs", {}))

    _dump_json(output_root / "dataset_manifest.json", merged_dataset_manifest)
    _dump_json(output_root / "dataset_diagnostics.json", merged_dataset_diagnostics)
    _dump_json(output_root / "phase2_localization.json", merged_localization)
    _dump_json(output_root / "phase2_interventions.json", merged_interventions)
    _dump_json(output_root / "phase2_cot_recruitment_compare.json", merged_cot)
    _dump_json(output_root / "phase2_cross_operator_verify.json", merged_cross_verify)
    _dump_json(output_root / "phase2_cross_operator_specificity_matrix.json", merged_cross_matrix)

    base_manifest = dict(shard_payloads[0]["run_manifest"])
    merged_cfg = json.loads(json.dumps(base_manifest.get("effective_config", {})))
    merged_cfg["datasets"]["operator_buckets"] = merged_dataset_manifest.get("config_snapshot", {}).get("operator_buckets", {})
    merged_cfg["runtime"]["operator_filter"] = None
    merged_cfg["runtime"]["operator_shard_mode"] = False

    gate_summary = _phase2_gate_template()
    gate_summary["run_metadata"] = {
        "timestamp_utc": _timestamp_utc(),
        "model": merged_cfg.get("model", {}).get("name"),
        "output_root": str(output_root),
        "stage": "full",
        "source_shards": [str(p["dir"]) for p in shard_payloads],
        "is_merged_from_shards": True,
    }
    gate_summary["scope"] = {
        "operator_coverage": sorted(merged_dataset_manifest.get("counts_by_operator", {}).keys()),
        "is_sharded_run": False,
        "merge_required_for_full_claims": False,
    }

    gate_summary["required_gates_policy"] = {
        "cot_required_for_readiness": bool(merged_cfg.get("gates", {}).get("cot_required_for_readiness", True)),
        "require_all_component_types": bool(merged_cfg.get("gates", {}).get("require_all_component_types", True)),
        "require_non_target_operator_evidence": bool(
            merged_cfg.get("gates", {}).get("require_non_target_operator_evidence", True)
        ),
        "anomaly_blocking_enabled": bool((merged_cfg.get("gates", {}).get("anomaly_blocking") or {}).get("enabled", True)),
        "required_for_readiness": _required_gates_for_readiness(merged_cfg),
    }
    counts = merged_dataset_manifest.get("counts_by_bucket", {})
    min_required = int(merged_cfg.get("gates", {}).get("dataset_bucket_min_examples", 1))
    gate_summary["phases"]["dataset_bucket_gate"] = {
        "ran": True,
        "passes": all(int(v) >= min_required for v in counts.values()),
        "skipped": False,
        "counts_by_bucket": counts,
        "min_required": min_required,
        "dataset_manifest": "dataset_manifest.json",
        "dataset_diagnostics": "dataset_diagnostics.json",
    }

    localizer_thresholds = []
    specificity_thresholds = []
    for payload in shard_payloads:
        derived = payload["gate_summary"].get("derived_thresholds", {})
        localizer = derived.get("localizer", {})
        specificity = derived.get("specificity", {})
        if localizer:
            localizer_thresholds.append(localizer)
        if specificity:
            specificity_thresholds.append(specificity)
    loc_nonzero_min = _max_or_default(
        [float(x.get("localization_nonzero_min", 0.0)) for x in localizer_thresholds],
        float(merged_cfg.get("gates", {}).get("localization_nonzero_min", 0.0)),
    )
    loc_prob_min = _max_or_default(
        [float(x.get("localization_prob_delta_abs_min", 0.0)) for x in localizer_thresholds],
        float(merged_cfg.get("gates", {}).get("localization_prob_delta_abs_min_floor", 0.0)),
    )
    spec_ci_min = _max_or_default(
        [float(x.get("specificity_ci_low_min", 0.0)) for x in specificity_thresholds],
        float(merged_cfg.get("gates", {}).get("specificity_ci_low_min", 0.0)),
    )
    gate_summary["derived_thresholds"]["localizer"] = {
        "policy": "merged_max_across_shards",
        "localization_nonzero_min": loc_nonzero_min,
        "localization_prob_delta_abs_min": loc_prob_min,
    }
    gate_summary["derived_thresholds"]["specificity"] = {
        "policy": "merged_max_across_shards",
        "specificity_ci_low_min": spec_ci_min,
    }

    loc_gate = _evaluate_localization_gate(
        merged_localization.get("runs", {}),
        nonzero_min=loc_nonzero_min,
        prob_delta_abs_min=loc_prob_min,
        required_robustness_modes=list((merged_cfg.get("localization") or {}).get("stability_modes", [])),
        require_all_runs=bool((merged_cfg.get("gates") or {}).get("require_all_component_types", True)),
        score_aggregation=str((merged_cfg.get("localization") or {}).get("score_aggregation", "quantile")),
        score_quantile=float((merged_cfg.get("localization") or {}).get("score_quantile", 0.95)),
        min_valid_target_count=int((merged_cfg.get("localization") or {}).get("min_valid_target_count", 0)),
        min_components_passing=int((merged_cfg.get("localization") or {}).get("min_components_passing", 1)),
        min_answer_target_valid_rate=float((merged_cfg.get("localization") or {}).get("min_answer_target_valid_rate", 0.0)),
    )
    loc_gate.update({"ran": True, "skipped": False, "output": "phase2_localization.json"})
    gate_summary["phases"]["localization_validity_gate"] = loc_gate

    if merged_cross_verify.get("status") != "ok" or not merged_cross_verify.get("runs"):
        int_gate = {
            "passes": False,
            "checks": [],
            "failure_reason": "missing_non_target_evidence",
            "ran": True,
            "skipped": False,
            "output": "phase2_cross_operator_verify.json",
            "matrix_output": "phase2_cross_operator_specificity_matrix.json",
            "evidence_source": "post_merge_cross_operator_verify",
        }
    else:
        int_gate = _evaluate_specificity_gate(
            merged_cross_verify.get("runs", {}),
            ci_low_min=spec_ci_min,
            mean_gap_min=float((merged_cfg.get("gates") or {}).get("specificity_mean_gap_min", 0.01)),
            require_non_target_operator_evidence=bool(
                (merged_cfg.get("gates") or {}).get("require_non_target_operator_evidence", True)
            ),
            require_primary_set=bool((merged_cfg.get("gates") or {}).get("specificity_requires_primary_set", True)),
            primary_set_name=str((merged_cfg.get("gates") or {}).get("specificity_primary_set", "top")),
            require_directionality=bool((merged_cfg.get("gates") or {}).get("specificity_requires_directionality", True)),
            require_both_primary_interventions=bool(
                (merged_cfg.get("gates") or {}).get("specificity_requires_both_primary_interventions", True)
            ),
            primary_interventions=list((merged_cfg.get("interventions") or {}).get("primary_interventions", ["ablation", "amplification"])),
            primary_scales=list((merged_cfg.get("interventions") or {}).get("primary_scales", [])),
            primary_k_values=list((merged_cfg.get("interventions") or {}).get("primary_k_values", [])),
            condition_policy=str((merged_cfg.get("gates") or {}).get("specificity_condition_policy", "fixed_preregistered_grid")),
            sign_policy=str((merged_cfg.get("gates") or {}).get("specificity_sign_policy", "intervention_signed")),
            ablation_delta_vs_random_max=float(
                ((merged_cfg.get("gates") or {}).get("anomaly_blocking") or {}).get("top_ablation_delta_vs_random_max", 0.1)
            ),
            amplification_delta_vs_random_min=float(
                ((merged_cfg.get("gates") or {}).get("anomaly_blocking") or {}).get("top_amplification_delta_vs_random_min", -0.1)
            ),
            multiplicity_blocking_enabled=bool(
                ((merged_cfg.get("gates") or {}).get("multiplicity_blocking") or {}).get("enabled", False)
            ),
            multiplicity_q_max=float(
                ((merged_cfg.get("gates") or {}).get("multiplicity_blocking") or {}).get("q_max", 0.1)
            ),
            multiplicity_require_complete_primary_coverage=bool(
                (merged_cfg.get("gates") or {}).get("multiplicity_require_complete_primary_coverage", True)
            ),
        )
        int_gate.update(
            {
                "ran": True,
                "skipped": False,
                "output": "phase2_cross_operator_verify.json",
                "matrix_output": "phase2_cross_operator_specificity_matrix.json",
                "evidence_source": "post_merge_cross_operator_verify",
            }
        )
    gate_summary["phases"]["operator_specificity_gate"] = int_gate

    sanity_gate = _evaluate_intervention_sanity_gate(
        merged_interventions.get("runs", {}),
        enabled=bool(((merged_cfg.get("gates") or {}).get("anomaly_blocking") or {}).get("enabled", True)),
    )
    sanity_gate.update({"ran": True, "skipped": False, "output": "phase2_interventions.json"})
    gate_summary["phases"]["intervention_sanity_gate"] = sanity_gate

    cot_cfg = merged_cfg.get("cot_compare", {})
    cot_gate = _evaluate_cot_gate(
        merged_cot.get("runs", {}),
        effect_abs_min=float(cot_cfg.get("effect_abs_min", 0.01)),
        parse_rate_delta_abs_max=float(cot_cfg.get("parse_rate_delta_abs_max", 0.05)),
        min_pairs=int(cot_cfg.get("min_pairs", 0)),
        parse_rate_min=float(cot_cfg.get("parse_rate_min", 0.0)),
        require_accuracy_ci_excludes_zero=bool(cot_cfg.get("require_accuracy_ci_excludes_zero", False)),
    )
    cot_gate.update({"ran": True, "skipped": False, "output": "phase2_cot_recruitment_compare.json"})
    gate_summary["phases"]["cot_gating_evidence_gate"] = cot_gate

    if bool((merged_cfg.get("gates") or {}).get("require_non_target_operator_evidence", True)):
        if len(merged_dataset_manifest.get("counts_by_operator", {})) < 2:
            gate_summary["scope_blocks"].append("single_operator_scope_blocks_specificity")

    required_gates = list(gate_summary["required_gates_policy"]["required_for_readiness"])
    readiness_block_reasons: List[str] = []
    for gate_name in required_gates:
        if not bool(gate_summary["phases"].get(gate_name, {}).get("passes")):
            readiness_block_reasons.append(f"required_gate_failed:{gate_name}")
    for scope_block in gate_summary.get("scope_blocks", []):
        readiness_block_reasons.append(f"scope_block:{scope_block}")
    gate_summary["overall"] = {
        "ready_for_multimodel_next_tranche": len(readiness_block_reasons) == 0,
        "phase2_status": "full_pipeline_complete",
        "readiness_block_reasons": readiness_block_reasons,
    }

    merged_manifest = dict(base_manifest)
    merged_manifest["timestamp_utc"] = _timestamp_utc()
    merged_manifest["stage"] = "full"
    merged_manifest["cli"] = {
        "merge_operator_shards": True,
        "shard_dirs": [str(p["dir"]) for p in shard_payloads],
        "output_root": str(output_root),
    }
    merged_manifest["effective_config"] = merged_cfg
    merged_manifest["operator_scope"] = "full" if len(merged_dataset_manifest.get("counts_by_operator", {})) >= 3 else "subset"
    merged_manifest["operator_coverage"] = sorted(merged_dataset_manifest.get("counts_by_operator", {}).keys())
    merged_manifest["is_operator_shard"] = False
    merged_manifest["merge_required_for_full_claims"] = False
    merged_manifest["required_gates_policy"] = gate_summary.get("required_gates_policy", {})
    merged_manifest["derived_thresholds"] = gate_summary.get("derived_thresholds", {})
    merged_manifest["merge_sources"] = [str(p["dir"]) for p in shard_payloads]
    merged_manifest["tuned_batch_sizes_by_stage"] = {
        str(p["dir"]): dict(p["run_manifest"].get("tuned_batch_sizes_by_stage", {}))
        for p in shard_payloads
    }
    merged_manifest["autotune_probe_history"] = {
        str(p["dir"]): dict(p["run_manifest"].get("autotune_probe_history", {}))
        for p in shard_payloads
    }
    merged_manifest["equivalence_check_results"] = {
        str(p["dir"]): dict(p["run_manifest"].get("equivalence_check_results", {}))
        for p in shard_payloads
    }
    merged_manifest["operator_bucket_signatures"] = operator_bucket_signatures

    _dump_json(output_root / "run_manifest.json", merged_manifest)
    _dump_json(output_root / "phase2_gate_summary.json", gate_summary)
    _dump_json(
        output_root / "merge_manifest.json",
        {
            "schema_version": "phase2_operator_shard_merge_manifest_v1",
            "timestamp_utc": _timestamp_utc(),
            "output_root": str(output_root),
            "source_shards": [str(p["dir"]) for p in shard_payloads],
            "model": next(iter(model_names)),
            "seed_list": list(next(iter(seed_lists))),
            "effective_config_hash": next(iter(cfg_hashes)),
            "operator_coverage": sorted(merged_dataset_manifest.get("counts_by_operator", {}).keys()),
            "operator_bucket_signatures": operator_bucket_signatures,
            "notes": [
                "Merged artifact schemas match shard schemas at the top level.",
                "Per-operator bucket signatures validated across shard metadata.",
                "Gate summary is recomputed on merged runs and scope.",
            ],
        },
    )
    print(f"Merged {len(shard_dirs)} shard directories into {output_root}")
    print(f"Operators covered: {sorted(merged_dataset_manifest.get('counts_by_operator', {}).keys())}")
    print(f"Ready for multimodel next tranche: {gate_summary['overall']['ready_for_multimodel_next_tranche']}")


if __name__ == "__main__":
    main()

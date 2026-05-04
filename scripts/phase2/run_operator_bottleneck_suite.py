#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
import random
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config_file
from src.model_loader import load_local_model
from src.operator_buckets import OperatorBucketSuite, generate_operator_bucket_suite, suite_diagnostics
from src.arithmetic_localization import (
    LocalizationConfig,
    annotate_localization_rank_stability,
    component_sets_from_localization,
    run_arithmetic_localization,
    topk_rank_stability_spearman_localization,
)
from src.operator_interventions import (
    build_specificity_matrix_from_intervention_results,
    evaluate_operator_bucket_dataset,
    run_operator_intervention_sweeps,
)
from src.cot_recruitment import CoTRecruitmentComparisonConfig, run_cot_recruitment_compare
from src.parser_audit import (
    build_parser_audit_report,
    collect_parser_audit_samples_from_intervention_runs,
)
from src.power_analysis import build_power_analysis_report
from src.runtime_batch_autotune import (
    BatchAutotuneConfig,
    autotune_batch_size,
    cleanup_after_oom,
    is_oom_error,
)
from src.model_introspection import get_attention_module, infer_head_count, locate_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 operator heuristic bottleneck suite (CPU-first capable, GPU-backed stages supported)."
    )
    parser.add_argument("--model", default=None, help="HF model name (defaults to config file model.name).")
    parser.add_argument(
        "--cache-dir",
        default="../LLM Second-Order Effects/models",
        help="Model cache dir for local HF weights.",
    )
    parser.add_argument("--devices", default=None, help="CUDA_VISIBLE_DEVICES override.")
    parser.add_argument(
        "--dataset-config",
        default="configs/phase2/operator_buckets_llama3.yaml",
        help="Phase 2 YAML config path.",
    )
    parser.add_argument("--output-root", default=None, help="Output directory for this Phase 2 run.")
    parser.add_argument("--seed-list", default=None, help="Comma-separated seeds override (e.g. '0,1').")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override for GPU stages.")
    parser.add_argument(
        "--operators",
        default=None,
        help="Comma-separated operator subset to run (e.g. addition,subtraction).",
    )
    parser.add_argument(
        "--operator-shard-mode",
        action="store_true",
        help="Mark this run as an operator shard (readiness is scope-limited until merged).",
    )
    parser.add_argument(
        "--batch-autotune",
        dest="batch_autotune",
        action="store_true",
        help="Enable batch-size autotuning for GPU stages.",
    )
    parser.add_argument(
        "--no-batch-autotune",
        dest="batch_autotune",
        action="store_false",
        help="Disable batch-size autotuning for GPU stages.",
    )
    parser.set_defaults(batch_autotune=None)
    parser.add_argument("--batch-autotune-min", type=int, default=None, help="Minimum batch size considered by autotune.")
    parser.add_argument("--batch-autotune-max", type=int, default=None, help="Maximum batch size considered by autotune.")
    parser.add_argument(
        "--batch-autotune-stages",
        default=None,
        help="Comma-separated stages to autotune (localize,intervene,cot).",
    )
    parser.add_argument(
        "--batch-equivalence-check",
        dest="batch_equivalence_check",
        action="store_true",
        help="Enable baseline-vs-tuned numeric equivalence checks.",
    )
    parser.add_argument(
        "--no-batch-equivalence-check",
        dest="batch_equivalence_check",
        action="store_false",
        help="Disable baseline-vs-tuned numeric equivalence checks.",
    )
    parser.set_defaults(batch_equivalence_check=None)
    parser.add_argument("--smoke", action="store_true", help="Run a reduced-size Phase 2 smoke configuration.")
    parser.add_argument(
        "--stage",
        default="full",
        choices=["datasets", "localize", "intervene", "cross_operator_verify", "cot_compare", "full"],
        help="Stage to run. Dependency stages are run automatically as needed.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        choices=["datasets", "localize", "intervene", "cross_operator_verify", "cot_compare"],
        help="Resume hint recorded in manifest (full checkpoint resume is not yet implemented).",
    )
    parser.add_argument(
        "--cross-operator-source",
        default=None,
        help=(
            "Optional source path for component sets used by --stage cross_operator_verify. "
            "Accepts either a phase2_localization.json file or a run directory containing it."
        ),
    )
    parser.add_argument(
        "--low-cpu-mode",
        action="store_true",
        help="Cap thread usage for torch/BLAS/tokenizers to reduce CPU contention.",
    )
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=None,
        help="Explicit torch thread cap (applies in addition to low-cpu-mode if set).",
    )
    parser.add_argument(
        "--scaffold-gpu-stages",
        action="store_true",
        help="Write schema-valid placeholder outputs for GPU-backed stages instead of loading a model (useful for CPU-only validation/tests).",
    )
    return parser.parse_args()


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_seed_list(text: Optional[str], default: List[int]) -> List[int]:
    if text is None:
        return list(default)
    values: List[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if piece:
            values.append(int(piece))
    if not values:
        raise ValueError("No seeds parsed from --seed-list")
    return values


def _parse_operator_list(text: Optional[str], default: Optional[Sequence[str]] = None) -> Optional[List[str]]:
    if text is None:
        if default is None:
            return None
        return [str(x) for x in default]
    values: List[str] = []
    for piece in str(text).split(","):
        item = piece.strip()
        if item:
            values.append(item)
    if not values:
        return None
    return values


def _batch_autotune_template(runtime_cfg: Mapping[str, Any], *, smoke: bool) -> Dict[str, Any]:
    cfg = runtime_cfg.get("batch_autotune") if isinstance(runtime_cfg.get("batch_autotune"), dict) else {}
    stages = cfg.get("stages", ["localize", "intervene", "cot"])
    if not isinstance(stages, list) or not stages:
        stages = ["localize", "intervene", "cot"]
    min_batch = int(cfg.get("min_batch_size", 4))
    if smoke:
        min_batch = min(min_batch, 2)
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "min_batch_size": min_batch,
        "max_batch_size": cfg.get("max_batch_size"),
        "growth_factor": float(cfg.get("growth_factor", 1.5)),
        "safety_backoff": float(cfg.get("safety_backoff", 0.85)),
        "max_retries_after_oom": int(cfg.get("max_retries_after_oom", 3)),
        "stages": [str(s) for s in stages],
        "equivalence_check": {
            "enabled": bool((cfg.get("equivalence_check") or {}).get("enabled", True)),
            "sample_size": int((cfg.get("equivalence_check") or {}).get("sample_size", 16)),
            "max_abs_logit_diff": float((cfg.get("equivalence_check") or {}).get("max_abs_logit_diff", 1e-4)),
            "max_metric_diff": float((cfg.get("equivalence_check") or {}).get("max_metric_diff", 1e-4)),
        },
    }


def _apply_cpu_runtime_controls(low_cpu_mode: bool, max_cpu_threads: Optional[int]) -> Dict[str, Any]:
    applied: Dict[str, Any] = {
        "low_cpu_mode": bool(low_cpu_mode),
        "max_cpu_threads": max_cpu_threads,
        "env": {},
        "torch_threads": {},
    }
    if low_cpu_mode:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if max_cpu_threads is not None:
        if max_cpu_threads <= 0:
            raise ValueError("--max-cpu-threads must be positive")
        os.environ["OMP_NUM_THREADS"] = str(max_cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(max_cpu_threads)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        if key in os.environ:
            applied["env"][key] = os.environ[key]
    try:
        import torch

        if max_cpu_threads is not None:
            torch.set_num_threads(max_cpu_threads)
            try:
                torch.set_num_interop_threads(max_cpu_threads)
            except RuntimeError:
                pass
        elif low_cpu_mode:
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        applied["torch_threads"] = {
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        }
    except Exception as exc:  # pragma: no cover
        applied["torch_threads_error"] = str(exc)
    return applied


def _normalize_phase2_config(cfg: Dict[str, Any], *, smoke: bool) -> Dict[str, Any]:
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    runtime_cfg = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
    datasets_cfg = cfg.get("datasets") if isinstance(cfg.get("datasets"), dict) else {}
    loc_cfg = cfg.get("localization") if isinstance(cfg.get("localization"), dict) else {}
    int_cfg = cfg.get("interventions") if isinstance(cfg.get("interventions"), dict) else {}
    cot_cfg = cfg.get("cot_compare") if isinstance(cfg.get("cot_compare"), dict) else {}
    gates_cfg = cfg.get("gates") if isinstance(cfg.get("gates"), dict) else {}
    analysis_cfg = cfg.get("analysis") if isinstance(cfg.get("analysis"), dict) else {}

    operators_cfg = datasets_cfg.get("operators") if isinstance(datasets_cfg.get("operators"), dict) else {}
    operator_buckets = {
        str(op): [str(b) for b in buckets]
        for op, buckets in operators_cfg.items()
        if isinstance(buckets, list) and buckets
    }
    if not operator_buckets:
        operator_buckets = {"addition": ["no_carry", "single_carry", "cascading_carry"]}

    counts_per_bucket = int(datasets_cfg.get("counts_per_bucket", 64))
    if smoke:
        counts_per_bucket = min(counts_per_bucket, 8)
    rep_variants = datasets_cfg.get("representation_variants", ["plain"])
    if not isinstance(rep_variants, list) or not rep_variants:
        rep_variants = ["plain"]

    seeds = runtime_cfg.get("seeds", [0])
    if not isinstance(seeds, list) or not seeds:
        seeds = [0]
    seeds = [int(s) for s in seeds]
    batch_size = int(runtime_cfg.get("batch_size", 8))
    if smoke:
        batch_size = min(batch_size, 4)

    component_types = loc_cfg.get("component_types", ["attention_heads"])
    if not isinstance(component_types, list) or not component_types:
        component_types = ["attention_heads"]
    metric_targets = loc_cfg.get("metric_targets", ["answer_token"])
    if not isinstance(metric_targets, list) or not metric_targets:
        metric_targets = ["answer_token"]
    stability_modes = loc_cfg.get("stability_modes", ["same_set_shuffle_invariance"])
    if not isinstance(stability_modes, list) or not stability_modes:
        stability_modes = ["same_set_shuffle_invariance"]
    component_options = loc_cfg.get("component_options") if isinstance(loc_cfg.get("component_options"), dict) else {}
    # Set conservative defaults so Phase 2 is runnable on Llama without accidentally enumerating
    # every MLP neuron in every layer unless the user explicitly requests it.
    component_options = {
        "attention_heads": {
            **({"head_limit_per_layer": 8} if not smoke else {}),
            **(component_options.get("attention_heads") if isinstance(component_options.get("attention_heads"), dict) else {}),
        },
        "mlp_neurons": {
            **({"sample_per_layer": 32} if not smoke else {}),
            **(component_options.get("mlp_neurons") if isinstance(component_options.get("mlp_neurons"), dict) else {}),
        },
        "layer_blocks": {
            **(component_options.get("layer_blocks") if isinstance(component_options.get("layer_blocks"), dict) else {}),
        },
    }
    if smoke:
        # Keep smoke scans tiny and CPU-friendly.
        component_options = {
            **component_options,
            "attention_heads": {
                "layer_indices": [0],
                "head_limit_per_layer": 2,
                **(component_options.get("attention_heads") if isinstance(component_options.get("attention_heads"), dict) else {}),
            },
            "mlp_neurons": {
                "layer_indices": [0],
                "sample_per_layer": 4,
                **(component_options.get("mlp_neurons") if isinstance(component_options.get("mlp_neurons"), dict) else {}),
            },
            "layer_blocks": {
                "layer_indices": [0],
                **(component_options.get("layer_blocks") if isinstance(component_options.get("layer_blocks"), dict) else {}),
            },
        }
    min_valid_target_count = int(loc_cfg.get("min_valid_target_count", 64))
    if smoke:
        min_valid_target_count = min(min_valid_target_count, max(2, counts_per_bucket // 2))

    tests = int_cfg.get("tests", ["ablation", "amplification"])
    if not isinstance(tests, list) or not tests:
        tests = ["ablation", "amplification"]
    scales = int_cfg.get("scales", [0.0, 1.25])
    if not isinstance(scales, list) or not scales:
        scales = [0.0, 1.25]
    scales = [float(s) for s in scales]
    controls = int_cfg.get("controls", ["matched_random"])
    if not isinstance(controls, list) or not controls:
        controls = ["matched_random"]
    primary_interventions = int_cfg.get("primary_interventions", ["ablation", "amplification"])
    if not isinstance(primary_interventions, list) or not primary_interventions:
        primary_interventions = ["ablation", "amplification"]
    primary_scales = int_cfg.get("primary_scales", [])
    if not isinstance(primary_scales, list):
        primary_scales = []
    if not primary_scales:
        # If not explicitly preregistered, default to the configured sweep scales.
        primary_scales = list(scales)
    primary_k_values = int_cfg.get("primary_k_values", [])
    if not isinstance(primary_k_values, list):
        primary_k_values = []
    if not primary_k_values:
        primary_k_values = list(loc_cfg.get("k_values", [5, 10]))
    split_cfg = datasets_cfg.get("target_operator_selection_eval_split")
    if not isinstance(split_cfg, dict):
        split_cfg = {}
    split_min_examples = int(split_cfg.get("min_examples_per_split", 2 if smoke else 8))
    if smoke:
        split_min_examples = min(split_min_examples, max(2, counts_per_bucket // 2))

    return {
        "run_name": str(cfg.get("run_name", "operator_bottleneck_phase2")),
        "phase": str(cfg.get("phase", "phase2")),
        "model": {"name": str(model_cfg.get("name", "meta-llama/Meta-Llama-3-8B"))},
        "runtime": {
            "devices": runtime_cfg.get("devices"),
            "batch_size": batch_size,
            "seeds": seeds,
            "low_cpu_mode": bool(runtime_cfg.get("low_cpu_mode", False)),
            "deterministic_generation": bool(runtime_cfg.get("deterministic_generation", True)),
            "allow_sampling_fallback": bool(runtime_cfg.get("allow_sampling_fallback", False)),
            "operator_shard_mode": bool(runtime_cfg.get("operator_shard_mode", False)),
            "operator_filter": (
                [str(x) for x in runtime_cfg.get("operator_filter", [])]
                if isinstance(runtime_cfg.get("operator_filter"), list)
                else None
            ),
            "batch_autotune": _batch_autotune_template(runtime_cfg, smoke=smoke),
        },
        "datasets": {
            "operator_buckets": operator_buckets,
            "counts_per_bucket": counts_per_bucket,
            "representation_variants": [str(v) for v in rep_variants],
            "target_operator_selection_eval_split": {
                "enabled": bool(split_cfg.get("enabled", True)),
                "holdout_fraction": float(split_cfg.get("holdout_fraction", 0.5)),
                "min_examples_per_split": split_min_examples,
                "seed_offset": int(split_cfg.get("seed_offset", 101)),
                "require_strict": bool(split_cfg.get("require_strict", True)),
                "hard_stop_on_failure": bool(split_cfg.get("hard_stop_on_failure", True)),
            },
        },
        "localization": {
            "component_types": [str(x) for x in component_types],
            "metric_targets": [str(x) for x in metric_targets],
            "stability_modes": [str(x) for x in stability_modes],
            "strict_attention_heads": bool(loc_cfg.get("strict_attention_heads", True)),
            "component_options": component_options,
            "epsilon": float(loc_cfg.get("epsilon", 1e-4)),
            "max_examples_per_dataset": loc_cfg.get("max_examples_per_dataset"),
            "subsample_fraction": float(loc_cfg.get("subsample_fraction", 0.8)),
            "score_key": str(loc_cfg.get("score_key", "answer_token_prob_delta_mean")),
            "k_values": [int(x) for x in loc_cfg.get("k_values", [5, 10])],
            "component_sampling_seed": (
                int(loc_cfg.get("component_sampling_seed"))
                if loc_cfg.get("component_sampling_seed") is not None
                else None
            ),
            "score_aggregation": str(loc_cfg.get("score_aggregation", "quantile")),
            "score_quantile": float(loc_cfg.get("score_quantile", 0.95)),
            "min_valid_target_count": min_valid_target_count,
            "min_components_passing": int(loc_cfg.get("min_components_passing", 1)),
            "min_answer_target_valid_rate": float(loc_cfg.get("min_answer_target_valid_rate", 0.5 if not smoke else 0.25)),
        },
        "interventions": {
            "tests": [str(x) for x in tests],
            "scales": scales,
            "controls": [str(x) for x in controls],
            "patching": bool(int_cfg.get("patching", False)),
            "bootstrap_samples": int(int_cfg.get("bootstrap_samples", 500 if smoke else 1000)),
            "primary_component_set": str(int_cfg.get("primary_component_set", "top")),
            "primary_interventions": [str(x) for x in primary_interventions],
            "primary_scales": [float(x) for x in primary_scales],
            "primary_k_values": [int(x) for x in primary_k_values],
        },
        "cot_compare": {
            "enabled": bool(cot_cfg.get("enabled", False)),
            "direct_instruction": str(cot_cfg.get("direct_instruction", "Give only the final numeric answer.")),
            "cot_instruction": str(cot_cfg.get("cot_instruction", "Think step by step, then give the final answer.")),
            "format_lock": {
                "enabled": bool((cot_cfg.get("format_lock") or {}).get("enabled", True)),
                "answer_marker": str((cot_cfg.get("format_lock") or {}).get("answer_marker", "####")),
            },
            "relaxed_parser_diagnostics": {
                "enabled": bool((cot_cfg.get("relaxed_parser_diagnostics") or {}).get("enabled", True)),
            },
            "paired_count": int(cot_cfg.get("paired_count", 64)),
            "sensitivity_component_set": str(cot_cfg.get("sensitivity_component_set", "top")),
            "sensitivity_k": int(cot_cfg.get("sensitivity_k", 5)),
            "sensitivity_scale": float(cot_cfg.get("sensitivity_scale", 0.0)),
            "evaluation_parse_mode": str(cot_cfg.get("evaluation_parse_mode", "strict_final_numeric")),
            "max_new_tokens": int(cot_cfg.get("max_new_tokens", 64)),
            "effect_abs_min": float(cot_cfg.get("effect_abs_min", 0.01)),
            "parse_rate_delta_abs_max": float(cot_cfg.get("parse_rate_delta_abs_max", 0.05)),
            "min_pairs": int(cot_cfg.get("min_pairs", 32)),
            "parse_rate_min": float(cot_cfg.get("parse_rate_min", 0.8)),
            "require_accuracy_ci_excludes_zero": bool(cot_cfg.get("require_accuracy_ci_excludes_zero", True)),
            "stratify_by_dataset": bool(cot_cfg.get("stratify_by_dataset", True)),
            "sampling_seed": int(cot_cfg.get("sampling_seed", 0)),
            "dataset_pair_allocation": str(cot_cfg.get("dataset_pair_allocation", "weighted_by_dataset_size")),
        },
        "gates": {
            "dataset_bucket_min_examples": int(gates_cfg.get("dataset_bucket_min_examples", 1)),
            "localization_nonzero_min": float(gates_cfg.get("localization_nonzero_min", 0.01)),
            "localization_prob_delta_abs_min_floor": float(gates_cfg.get("localization_prob_delta_abs_min_floor", 1e-5)),
            "specificity_ci_low_min": float(gates_cfg.get("specificity_ci_low_min", 0.01)),
            "specificity_mean_gap_min": float(gates_cfg.get("specificity_mean_gap_min", 0.01)),
            "require_all_component_types": bool(gates_cfg.get("require_all_component_types", True)),
            "require_non_target_operator_evidence": bool(gates_cfg.get("require_non_target_operator_evidence", True)),
            "cot_required_for_readiness": bool(gates_cfg.get("cot_required_for_readiness", True)),
            "specificity_requires_primary_set": bool(gates_cfg.get("specificity_requires_primary_set", True)),
            "specificity_primary_set": str(gates_cfg.get("specificity_primary_set", "top")),
            "specificity_requires_directionality": bool(gates_cfg.get("specificity_requires_directionality", True)),
            "specificity_requires_both_primary_interventions": bool(
                gates_cfg.get("specificity_requires_both_primary_interventions", True)
            ),
            "specificity_condition_policy": str(gates_cfg.get("specificity_condition_policy", "fixed_preregistered_grid")),
            "specificity_sign_policy": str(gates_cfg.get("specificity_sign_policy", "intervention_signed")),
            "multiplicity_require_complete_primary_coverage": bool(
                gates_cfg.get("multiplicity_require_complete_primary_coverage", True)
            ),
            "scope_block_on_single_operator": bool(
                gates_cfg.get(
                    "scope_block_on_single_operator",
                    bool(gates_cfg.get("require_non_target_operator_evidence", True)),
                )
            ),
            "multiplicity_blocking": {
                "enabled": bool((gates_cfg.get("multiplicity_blocking") or {}).get("enabled", False)),
                "q_max": float((gates_cfg.get("multiplicity_blocking") or {}).get("q_max", 0.1)),
            },
            "calibration": {
                "enabled": bool((gates_cfg.get("calibration") or {}).get("enabled", True)),
                "localizer_null_policy": str(
                    (gates_cfg.get("calibration") or {}).get("localizer_null_policy", "target_shuffle+family_heldout")
                ),
                "quantile": float((gates_cfg.get("calibration") or {}).get("quantile", 0.95)),
                "max_heldout_buckets": int((gates_cfg.get("calibration") or {}).get("max_heldout_buckets", 1)),
                "family_heldout_coverage": str((gates_cfg.get("calibration") or {}).get("family_heldout_coverage", "exhaustive")),
            },
            "anomaly_blocking": {
                "enabled": bool((gates_cfg.get("anomaly_blocking") or {}).get("enabled", True)),
                "baseline_near_floor_max": float((gates_cfg.get("anomaly_blocking") or {}).get("baseline_near_floor_max", 0.15)),
                "high_accuracy_min": float((gates_cfg.get("anomaly_blocking") or {}).get("high_accuracy_min", 0.9)),
                "min_accuracy_delta": float((gates_cfg.get("anomaly_blocking") or {}).get("min_accuracy_delta", 0.5)),
                "top_ablation_delta_vs_random_max": float(
                    (gates_cfg.get("anomaly_blocking") or {}).get("top_ablation_delta_vs_random_max", 0.1)
                ),
                "top_amplification_delta_vs_random_min": float(
                    (gates_cfg.get("anomaly_blocking") or {}).get("top_amplification_delta_vs_random_min", -0.1)
                ),
                "ablation_large_positive_delta_min": float(
                    (gates_cfg.get("anomaly_blocking") or {}).get("ablation_large_positive_delta_min", 0.5)
                ),
                "prediction_sample_size": int((gates_cfg.get("anomaly_blocking") or {}).get("prediction_sample_size", 8)),
            },
        },
        "analysis": {
            "multiplicity_reporting": str(analysis_cfg.get("multiplicity_reporting", "bh_fdr")),
            "preregistration_config": str(
                analysis_cfg.get("preregistration_config", "configs/phase2/preregistration.yaml")
            ),
            "parser_audit": {
                "enabled": bool((analysis_cfg.get("parser_audit") or {}).get("enabled", True)),
                "per_dataset_limit": int((analysis_cfg.get("parser_audit") or {}).get("per_dataset_limit", 16)),
                "adjudication_cap": int((analysis_cfg.get("parser_audit") or {}).get("adjudication_cap", 64)),
            },
        },
    }


def _write_replication_protocol(output_root: Path) -> None:
    text = """# Phase 2 Replication Protocol (Operator Heuristic Bottleneck Mainline)

Status: Phase 2 implementation with CPU-first orchestration and GPU-backed localization/intervention/CoT stages.

Recommended sequence:
1. Run `--stage datasets` and validate `dataset_manifest.json` + `dataset_diagnostics.json`.
2. Run `--stage localize` (prefer smoke first) and validate `phase2_localization*.json` + localization gate.
3. Run `--stage intervene` and inspect the specificity matrix + operator-specificity gate.
4. Run `--stage cot_compare` to test CoT gating/composition hypotheses.
5. Use `--stage full` for end-to-end artifacts once CPU/GPU availability allows.
6. Only then mark `ready_for_multimodel_next_tranche=true`.
"""
    (output_root / "replication_protocol.md").write_text(text, encoding="utf-8")


def _phase2_gate_template() -> Dict[str, Any]:
    return {
        "schema_version": "phase2_operator_bottleneck_gate_summary_v2",
        "schema_revision": "2.1",
        "phases": {
            "dataset_bucket_gate": {"ran": False, "passes": False, "skipped": True},
            "localization_validity_gate": {"ran": False, "passes": False, "skipped": True},
            "operator_specificity_gate": {"ran": False, "passes": False, "skipped": True},
            "intervention_sanity_gate": {"ran": False, "passes": False, "skipped": True},
            "cot_gating_evidence_gate": {"ran": False, "passes": False, "skipped": True},
        },
        "derived_thresholds": {},
        "required_gates_policy": {},
        "scope_warnings": [],
        "scope_blocks": [],
        "scope": {
            "operator_coverage": [],
            "is_sharded_run": False,
            "merge_required_for_full_claims": False,
        },
        "overall": {"ready_for_multimodel_next_tranche": False, "readiness_block_reasons": []},
    }


def _load_preregistration_payload(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / config_path
    if not path.exists():
        raise FileNotFoundError(f"Missing preregistration config: {path}")
    payload = load_config_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Preregistration config must be a mapping: {path}")
    parsed = dict(payload)
    parsed["source_path"] = str(path)
    return parsed


def _required_gates_for_readiness(cfg: Mapping[str, Any]) -> List[str]:
    required_gate_names = ["dataset_bucket_gate", "localization_validity_gate", "operator_specificity_gate"]
    if bool(cfg["gates"]["anomaly_blocking"]["enabled"]):
        required_gate_names.append("intervention_sanity_gate")
    if bool(cfg["gates"]["cot_required_for_readiness"]):
        required_gate_names.append("cot_gating_evidence_gate")
    return required_gate_names


def _run_dataset_stage(output_root: Path, cfg: Dict[str, Any], *, seed: int) -> Dict[str, Any]:
    ds_cfg = cfg["datasets"]
    suite = generate_operator_bucket_suite(
        ds_cfg["operator_buckets"],
        counts_per_bucket=int(ds_cfg["counts_per_bucket"]),
        seed=seed,
        representation_variants=ds_cfg.get("representation_variants", ["plain"]),
    )
    manifest = suite.to_manifest()
    diagnostics = suite_diagnostics(suite)
    _json_dump(output_root / "dataset_manifest.json", manifest)
    _json_dump(output_root / "dataset_diagnostics.json", diagnostics)
    sample_examples = {
        name: [example.to_dict() for example in dataset.examples[:3]]
        for name, dataset in sorted(suite.datasets.items())
    }
    _json_dump(
        output_root / "dataset_examples_preview.json",
        {
            "schema_version": "operator_bucket_examples_preview_v1",
            "n_preview_per_dataset": 3,
            "datasets": sample_examples,
        },
    )
    return {"suite": suite, "manifest": manifest, "diagnostics": diagnostics}


def _operator_datasets_view(suite: OperatorBucketSuite) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, ds in suite.datasets.items():
        out.setdefault(ds.operator, {})[name] = ds
    return out


def _parse_policy_terms(text: str) -> List[str]:
    return [term.strip() for term in str(text).split("+") if term.strip()]


def _split_target_operator_datasets_for_selection_eval(
    suite: OperatorBucketSuite,
    *,
    operator: str,
    holdout_fraction: float,
    min_examples_per_split: int,
    seed: int,
) -> Dict[str, Any]:
    holdout_fraction = min(max(float(holdout_fraction), 0.1), 0.9)
    min_examples = max(int(min_examples_per_split), 1)
    rng = random.Random(int(seed))
    selection_target: Dict[str, Any] = {}
    evaluation_target: Dict[str, Any] = {}
    evaluation_all: Dict[str, Any] = {}
    split_records: Dict[str, Any] = {}

    for name, dataset in sorted(suite.datasets.items()):
        if dataset.operator != operator:
            evaluation_all[name] = dataset
            continue
        examples = list(dataset.examples)
        n = len(examples)
        indices = list(range(n))
        rng.shuffle(indices)
        requested_eval = int(round(n * holdout_fraction))
        max_eval = n - min_examples
        eval_size = min(max(requested_eval, min_examples), max_eval) if max_eval >= min_examples else 0
        if eval_size <= 0:
            # Not enough examples for a leakage-safe split; keep full dataset and mark warning.
            selection_target[name] = dataset
            evaluation_target[name] = dataset
            evaluation_all[name] = dataset
            split_records[name] = {
                "operator": dataset.operator,
                "bucket": dataset.bucket,
                "total_examples": n,
                "selection_examples": n,
                "evaluation_examples": n,
                "split_applied": False,
                "reason": "insufficient_examples_for_split",
            }
            continue
        eval_idx = set(indices[:eval_size])
        sel_examples = [ex for i, ex in enumerate(examples) if i not in eval_idx]
        eval_examples = [ex for i, ex in enumerate(examples) if i in eval_idx]
        selection_target[name] = type(dataset)(
            operator=dataset.operator,
            bucket=dataset.bucket,
            examples=sel_examples,
            seed=dataset.seed,
            representation_variant=dataset.representation_variant,
            metadata={**dict(dataset.metadata), "selection_eval_split": "selection"},
        )
        evaluation_all[name] = type(dataset)(
            operator=dataset.operator,
            bucket=dataset.bucket,
            examples=eval_examples,
            seed=dataset.seed,
            representation_variant=dataset.representation_variant,
            metadata={**dict(dataset.metadata), "selection_eval_split": "evaluation"},
        )
        split_records[name] = {
            "operator": dataset.operator,
            "bucket": dataset.bucket,
            "total_examples": n,
            "selection_examples": len(sel_examples),
            "evaluation_examples": len(eval_examples),
            "split_applied": True,
            "holdout_fraction": holdout_fraction,
        }
        evaluation_target[name] = evaluation_all[name]

    return {
        "selection_target_datasets": selection_target,
        "evaluation_target_datasets": evaluation_target,
        "evaluation_datasets_all_operators": evaluation_all,
        "split_records": split_records,
    }


def _maybe_load_induction_baseline_sets() -> Optional[Dict[str, List[str]]]:
    head_sets_path = ROOT / "results" / "phase1" / "canonical" / "head_validity_run_20260225_120553_gpu01" / "head_sets.json"
    if not head_sets_path.exists():
        return None
    try:
        payload = json.loads(head_sets_path.read_text())
    except Exception:
        return None
    # Use K5 top and random as compact induction baseline controls if present.
    sets = payload.get("head_sets", {}).get("K5", {})
    out: Dict[str, List[str]] = {}
    for name in ("top", "random_matched", "bottom"):
        if name not in sets:
            continue
        head_rows = sets[name]
        converted = []
        for row in head_rows:
            try:
                converted.append(f"attn_head:L{int(row['layer'])}:H{int(row['head'])}")
            except Exception:
                continue
        if converted:
            out[f"induction_{name}"] = converted
    return out or None


def _sanitize_attention_component_ids_for_model(
    model,
    component_ids: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Drop attention component IDs that are invalid for the current model topology."""
    try:
        layers = locate_layers(model)
    except Exception:
        return list(component_ids), []
    n_layers = len(layers)
    head_count_cache: Dict[int, Optional[int]] = {}
    valid: List[str] = []
    dropped: List[str] = []
    for cid in component_ids:
        if not str(cid).startswith("attn_head:"):
            valid.append(str(cid))
            continue
        try:
            rest = str(cid).split(":", 1)[1]
            layer = int(rest.split(":H")[0].replace("L", ""))
            head = int(rest.split(":H")[1])
        except Exception:
            dropped.append(str(cid))
            continue
        if layer < 0 or layer >= n_layers:
            dropped.append(str(cid))
            continue
        if layer not in head_count_cache:
            attn = get_attention_module(layers[layer])
            head_count_cache[layer] = infer_head_count(attn)
        head_count = head_count_cache[layer]
        if head_count is not None and (head < 0 or head >= int(head_count)):
            dropped.append(str(cid))
            continue
        valid.append(str(cid))
    return valid, dropped


def _sanitize_induction_baseline_sets_for_model(
    model,
    sets: Optional[Dict[str, List[str]]],
    *,
    scope_warnings: List[str],
) -> Optional[Dict[str, List[str]]]:
    if not sets:
        return None
    cleaned: Dict[str, List[str]] = {}
    for set_name, component_ids in sets.items():
        valid, dropped = _sanitize_attention_component_ids_for_model(model, component_ids)
        if dropped:
            scope_warnings.append(
                "Filtered incompatible induction baseline components for this model "
                f"({set_name}: dropped={len(dropped)}, kept={len(valid)})."
            )
        if valid:
            cleaned[set_name] = valid
    if not cleaned:
        scope_warnings.append(
            "All induction baseline component IDs were incompatible with this model; "
            "induction baseline controls disabled for this run."
        )
    return cleaned or None


def _run_with_oom_backoff(
    *,
    stage_name: str,
    initial_batch_size: int,
    min_batch_size: int,
    max_retries_after_oom: int,
    safety_backoff: float,
    run_fn,
    scope_warnings: List[str],
    context_label: str,
) -> Tuple[Any, int]:
    """Execute `run_fn(batch_size)` with bounded OOM backoff retries."""
    batch_size = max(int(initial_batch_size), 1)
    min_batch = max(int(min_batch_size), 1)
    retries = 0
    while True:
        try:
            return run_fn(batch_size), batch_size
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            if retries >= int(max_retries_after_oom) or batch_size <= min_batch:
                raise
            prev = batch_size
            next_batch = max(min_batch, int(math.floor(float(batch_size) * float(safety_backoff))))
            if next_batch >= batch_size:
                next_batch = max(min_batch, batch_size - 1)
            cleanup_after_oom()
            retries += 1
            batch_size = next_batch
            scope_warnings.append(
                f"OOM during {stage_name} ({context_label}); reduced batch size {prev}->{batch_size} "
                f"and retrying ({retries}/{max_retries_after_oom})."
            )


def _compute_localization_robustness(
    model,
    tokenizer,
    *,
    model_name: str,
    datasets: Mapping[str, Any],
    base_config: LocalizationConfig,
    component_options: Dict[str, Any],
    base_result: Dict[str, Any],
    stability_modes: Sequence[str],
    epsilon: float,
    max_examples_per_dataset: Optional[int],
    subsample_fraction: float,
    seeds: Sequence[int],
    oom_min_batch_size: int,
    oom_max_retries: int,
    oom_safety_backoff: float,
    scope_warnings: List[str],
    context_label: str,
) -> Dict[str, Any]:
    same_set = None
    subsample = None
    family_heldout = None
    seed_robust = None

    if "same_set_shuffle_invariance" in stability_modes:
        shuffled, _ = _run_with_oom_backoff(
            stage_name="localize",
            initial_batch_size=int(base_config.batch_size),
            min_batch_size=oom_min_batch_size,
            max_retries_after_oom=oom_max_retries,
            safety_backoff=oom_safety_backoff,
            run_fn=lambda bs: run_arithmetic_localization(
                model,
                tokenizer,
                model_name=model_name,
                datasets=datasets,
                config=LocalizationConfig(**{**asdict(base_config), "seed": base_config.seed + 1, "batch_size": int(bs)}),
                component_options=component_options,
                epsilon=epsilon,
                max_examples_per_dataset=max_examples_per_dataset,
                shuffle_records=True,
            ),
            scope_warnings=scope_warnings,
            context_label=f"{context_label}:same_set_shuffle_invariance",
        )
        same_set = topk_rank_stability_spearman_localization(
            base_result, shuffled, score_key="answer_token_prob_delta_mean", top_k=50
        )

    if "subsample_stability" in stability_modes:
        subsampled, _ = _run_with_oom_backoff(
            stage_name="localize",
            initial_batch_size=int(base_config.batch_size),
            min_batch_size=oom_min_batch_size,
            max_retries_after_oom=oom_max_retries,
            safety_backoff=oom_safety_backoff,
            run_fn=lambda bs: run_arithmetic_localization(
                model,
                tokenizer,
                model_name=model_name,
                datasets=datasets,
                config=LocalizationConfig(**{**asdict(base_config), "seed": base_config.seed + 11, "batch_size": int(bs)}),
                component_options=component_options,
                epsilon=epsilon,
                max_examples_per_dataset=max_examples_per_dataset,
                subsample_fraction=subsample_fraction,
            ),
            scope_warnings=scope_warnings,
            context_label=f"{context_label}:subsample_stability",
        )
        subsample = topk_rank_stability_spearman_localization(
            base_result, subsampled, score_key="answer_token_prob_delta_mean", top_k=50
        )

    if "family_heldout_stability" in stability_modes:
        buckets = sorted({ds.bucket for ds in datasets.values()})
        values: List[float] = []
        if len(buckets) >= 2:
            for heldout in buckets:
                heldout_run, _ = _run_with_oom_backoff(
                    stage_name="localize",
                    initial_batch_size=int(base_config.batch_size),
                    min_batch_size=oom_min_batch_size,
                    max_retries_after_oom=oom_max_retries,
                    safety_backoff=oom_safety_backoff,
                    run_fn=lambda bs, heldout_bucket=heldout: run_arithmetic_localization(
                        model,
                        tokenizer,
                        model_name=model_name,
                        datasets=datasets,
                        config=LocalizationConfig(
                            **{**asdict(base_config), "seed": base_config.seed + 101, "batch_size": int(bs)}
                        ),
                        component_options=component_options,
                        epsilon=epsilon,
                        max_examples_per_dataset=max_examples_per_dataset,
                        heldout_buckets=[heldout_bucket],
                    ),
                    scope_warnings=scope_warnings,
                    context_label=f"{context_label}:family_heldout_stability:{heldout}",
                )
                rho = topk_rank_stability_spearman_localization(
                    base_result, heldout_run, score_key="answer_token_prob_delta_mean", top_k=50
                )
                if rho is not None:
                    values.append(float(rho))
        family_heldout = (sum(values) / len(values)) if values else None

    if "seed_robustness" in stability_modes and len(seeds) >= 2:
        alt_seed = int(seeds[1])
        alt_run, _ = _run_with_oom_backoff(
            stage_name="localize",
            initial_batch_size=int(base_config.batch_size),
            min_batch_size=oom_min_batch_size,
            max_retries_after_oom=oom_max_retries,
            safety_backoff=oom_safety_backoff,
            run_fn=lambda bs: run_arithmetic_localization(
                model,
                tokenizer,
                model_name=model_name,
                datasets=datasets,
                config=LocalizationConfig(**{**asdict(base_config), "seed": alt_seed, "batch_size": int(bs)}),
                component_options=component_options,
                epsilon=epsilon,
                max_examples_per_dataset=max_examples_per_dataset,
            ),
            scope_warnings=scope_warnings,
            context_label=f"{context_label}:seed_robustness:{alt_seed}",
        )
        seed_robust = topk_rank_stability_spearman_localization(
            base_result, alt_run, score_key="answer_token_prob_delta_mean", top_k=50
        )

    return {
        "same_set_shuffle_invariance": same_set,
        "subsample_stability": subsample,
        "family_heldout_stability": family_heldout,
        "seed_robustness": seed_robust,
    }


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    q = min(max(float(q), 0.0), 1.0)
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    idx = q * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _missing_robustness_modes(robustness_summary: Mapping[str, Any], required_modes: Sequence[str]) -> List[str]:
    missing: List[str] = []
    for mode in required_modes:
        value = robustness_summary.get(mode)
        if value is None:
            missing.append(mode)
            continue
        try:
            if not math.isfinite(float(value)):
                missing.append(mode)
        except Exception:
            missing.append(mode)
    return missing


def _derive_localization_thresholds(
    calibration_runs: Mapping[str, Dict[str, Any]],
    *,
    policy: str,
    quantile: float,
    nonzero_floor: float,
    abs_prob_floor: float,
) -> Dict[str, Any]:
    abs_prob_samples: List[float] = []
    nonzero_samples: List[float] = []
    for payload in calibration_runs.values():
        for metric in payload.get("metrics", []):
            abs_prob_samples.append(abs(float(metric.get("answer_token_prob_delta_mean") or 0.0)))
            nonzero_samples.append(float(metric.get("effect_nonzero_rate") or 0.0))
    derived_abs = _quantile(abs_prob_samples, quantile)
    derived_nonzero = _quantile(nonzero_samples, quantile)
    final_abs = max(abs_prob_floor, float(derived_abs or 0.0))
    final_nonzero = max(nonzero_floor, float(derived_nonzero or 0.0))
    return {
        "policy": policy,
        "quantile": quantile,
        "samples_count": len(abs_prob_samples),
        "abs_prob_quantile": derived_abs,
        "nonzero_quantile": derived_nonzero,
        "localization_prob_delta_abs_min": final_abs,
        "localization_nonzero_min": final_nonzero,
    }


def _derive_specificity_threshold_from_random_controls(
    intervention_payloads: Mapping[str, Dict[str, Any]],
    *,
    quantile: float,
    ci_low_floor: float,
) -> Dict[str, Any]:
    random_delta_means: List[float] = []
    for payload in intervention_payloads.values():
        for record in payload.get("results", []):
            condition = record.get("condition", {})
            set_name = str(condition.get("component_set_name", ""))
            if not set_name.endswith(":random_matched"):
                continue
            for ds_payload in record.get("datasets", {}).values():
                delta = ds_payload.get("metrics", {}).get("delta_vs_baseline_accuracy_all")
                if not isinstance(delta, dict):
                    continue
                random_delta_means.append(abs(float(delta.get("mean") or 0.0)))
    derived = _quantile(random_delta_means, quantile)
    final = max(ci_low_floor, float(derived or 0.0))
    return {
        "policy": "matched_random_delta_vs_baseline",
        "quantile": quantile,
        "samples_count": len(random_delta_means),
        "specificity_ci_low_quantile": derived,
        "specificity_ci_low_min": final,
    }


def _evaluate_localization_gate(
    localization_index: Mapping[str, Dict[str, Any]],
    *,
    nonzero_min: float,
    prob_delta_abs_min: float,
    required_robustness_modes: Sequence[str],
    require_all_runs: bool,
    score_aggregation: str = "max",
    score_quantile: float = 0.95,
    min_valid_target_count: int = 0,
    min_components_passing: int = 1,
    min_answer_target_valid_rate: float = 0.0,
) -> Dict[str, Any]:
    if score_aggregation not in {"max", "quantile"}:
        raise ValueError(f"Unsupported localization score_aggregation: {score_aggregation}")
    checks = []
    passes_all = True
    for key, result in localization_index.items():
        metrics = result.get("metrics", [])
        effect_values = [float(m.get("effect_nonzero_rate") or 0.0) for m in metrics]
        prob_values = [abs(float(m.get("answer_token_prob_delta_mean") or 0.0)) for m in metrics]
        valid_target_counts: List[int] = []
        component_passes: List[bool] = []
        for metric in metrics:
            meta = metric.get("metadata", {})
            valid_targets = int((meta or {}).get("answer_valid_count") or 0)
            valid_target_counts.append(valid_targets)
            component_passes.append(
                valid_targets >= min_valid_target_count
                and float(metric.get("effect_nonzero_rate") or 0.0) >= nonzero_min
                and abs(float(metric.get("answer_token_prob_delta_mean") or 0.0)) >= prob_delta_abs_min
            )
        n_components_passing = sum(1 for flag in component_passes if flag)
        n_components_with_valid_targets = sum(1 for n in valid_target_counts if n >= min_valid_target_count)
        max_nonzero = max((float(m.get("effect_nonzero_rate") or 0.0) for m in metrics), default=0.0)
        max_prob_delta = max((abs(float(m.get("answer_token_prob_delta_mean") or 0.0)) for m in metrics), default=0.0)
        quant_nonzero = _quantile(effect_values, score_quantile)
        quant_prob = _quantile(prob_values, score_quantile)
        if score_aggregation == "max":
            agg_nonzero = max_nonzero
            agg_prob = max_prob_delta
        else:
            agg_nonzero = float(quant_nonzero or 0.0)
            agg_prob = float(quant_prob or 0.0)
        aggregation_passes = bool(agg_nonzero >= nonzero_min and agg_prob >= prob_delta_abs_min)
        robustness = result.get("robustness_summary", {})
        answer_valid_rate = float((result.get("prompt_set") or {}).get("answer_target_valid_rate") or 0.0)
        missing_robustness = _missing_robustness_modes(robustness, required_robustness_modes)
        failure_reasons: List[str] = []
        if result.get("status") != "ok":
            failure_reasons.append("status_not_ok")
        if len(metrics) == 0:
            failure_reasons.append("empty_metrics")
        if n_components_with_valid_targets < min_components_passing:
            failure_reasons.append("insufficient_valid_targets")
        if n_components_passing < min_components_passing:
            failure_reasons.append("insufficient_components_passing")
        if not aggregation_passes:
            failure_reasons.append("aggregation_threshold_not_met")
        if answer_valid_rate < min_answer_target_valid_rate:
            failure_reasons.append("low_target_coverage")
        if missing_robustness:
            failure_reasons.append("missing_robustness_modes")
        run_passes = len(failure_reasons) == 0
        check = {
            "run": key,
            "status": result.get("status"),
            "n_metrics": len(metrics),
            "effect_nonzero_rate_max": max_nonzero,
            "answer_token_prob_delta_abs_max": max_prob_delta,
            "effect_nonzero_rate_quantile": quant_nonzero,
            "answer_token_prob_delta_abs_quantile": quant_prob,
            "aggregation_policy": {
                "mode": score_aggregation,
                "quantile": score_quantile,
                "aggregated_effect_nonzero_rate": agg_nonzero,
                "aggregated_answer_token_prob_delta_abs": agg_prob,
            },
            "n_components_with_valid_targets": n_components_with_valid_targets,
            "n_components_passing": n_components_passing,
            "answer_target_valid_rate": answer_valid_rate,
            "robustness_summary": robustness,
            "missing_robustness_modes": missing_robustness,
            "thresholds": {
                "nonzero_min": nonzero_min,
                "prob_delta_abs_min": prob_delta_abs_min,
                "required_robustness_modes": list(required_robustness_modes),
                "min_valid_target_count": min_valid_target_count,
                "min_components_passing": min_components_passing,
                "min_answer_target_valid_rate": min_answer_target_valid_rate,
            },
            "failure_reasons": failure_reasons,
            "passes": run_passes,
        }
        checks.append(check)
        passes_all = passes_all and run_passes
    overall_passes = passes_all if require_all_runs else any(bool(c["passes"]) for c in checks)
    return {
        "passes": bool(overall_passes),
        "checks": checks,
        "aggregation_policy": {
            "mode": score_aggregation,
            "quantile": score_quantile,
            "min_valid_target_count": min_valid_target_count,
            "min_components_passing": min_components_passing,
            "min_answer_target_valid_rate": min_answer_target_valid_rate,
        },
    }


def _parse_k_from_component_set_name(component_set_name: str) -> Optional[int]:
    prefix = str(component_set_name or "").split(":", 1)[0]
    if prefix.startswith("K") and prefix[1:].isdigit():
        return int(prefix[1:])
    return None


def _round_scale(value: Any) -> Optional[float]:
    try:
        return round(float(value), 8)
    except Exception:
        return None


def _evaluate_specificity_gate(
    intervention_payloads: Mapping[str, Dict[str, Any]],
    *,
    ci_low_min: float,
    mean_gap_min: float,
    require_non_target_operator_evidence: bool,
    require_primary_set: bool = False,
    primary_set_name: str = "top",
    require_directionality: bool = False,
    ablation_delta_vs_random_max: float = 0.1,
    amplification_delta_vs_random_min: float = -0.1,
    require_both_primary_interventions: bool = False,
    primary_interventions: Sequence[str] = ("ablation", "amplification"),
    primary_scales: Optional[Sequence[float]] = None,
    primary_k_values: Optional[Sequence[int]] = None,
    condition_policy: str = "fixed_preregistered_grid",
    sign_policy: str = "intervention_signed",
    multiplicity_blocking_enabled: bool = False,
    multiplicity_q_max: float = 0.1,
    multiplicity_require_complete_primary_coverage: bool = True,
) -> Dict[str, Any]:
    if sign_policy not in {"intervention_signed", "absolute"}:
        raise ValueError(f"Unsupported specificity sign_policy: {sign_policy}")
    if condition_policy not in {"fixed_preregistered_grid"}:
        raise ValueError(f"Unsupported specificity condition_policy: {condition_policy}")
    primary_interventions_set = {str(x) for x in primary_interventions}
    if not primary_interventions_set:
        primary_interventions_set = {"ablation", "amplification"}
    primary_scale_set: Optional[set[float]] = None
    if primary_scales:
        primary_scale_set = {round(float(x), 8) for x in primary_scales if _round_scale(x) is not None}
    primary_k_set: Optional[set[int]] = None
    if primary_k_values:
        primary_k_set = {int(x) for x in primary_k_values}

    def _signed_mean(intervention: str, mean: float) -> float:
        if sign_policy == "absolute":
            return abs(float(mean))
        if intervention == "ablation":
            return -float(mean)
        return float(mean)

    def _signed_ci_support(intervention: str, ci: Sequence[float]) -> float:
        if not isinstance(ci, (list, tuple)) or len(ci) < 2:
            return float("-inf")
        ci_low = float(ci[0])
        ci_high = float(ci[1])
        if sign_policy == "absolute":
            # If CI crosses 0, the absolute-effect lower support is 0.
            if ci_low <= 0.0 <= ci_high:
                return 0.0
            return min(abs(ci_low), abs(ci_high))
        if intervention == "ablation":
            # For ablation, evidence is stronger when the upper bound is below 0.
            return -ci_high
        return ci_low

    def _intervention_matches_scale(intervention: str, scale: float) -> bool:
        if intervention == "ablation":
            return abs(float(scale)) <= 1e-8
        if intervention == "amplification":
            return float(scale) > 1.0
        return True

    def _row_key_to_dict(key: Tuple[Optional[int], str, float]) -> Dict[str, Any]:
        return {"k_value": key[0], "intervention": key[1], "scale": key[2]}

    checks = []
    passes_all = True
    for key, payload in intervention_payloads.items():
        if payload.get("status") != "ok":
            checks.append(
                {
                    "run": key,
                    "status": payload.get("status"),
                    "passes": False,
                    "failure_reason": "run_not_ok",
                }
            )
            passes_all = False
            continue
        target_operator = key.split("::", 1)[0]
        run_primary_scale_set = set(primary_scale_set) if primary_scale_set is not None else None
        run_primary_k_set = set(primary_k_set) if primary_k_set is not None else None
        primary_target_records: List[Dict[str, Any]] = []
        primary_non_target_records: List[Dict[str, Any]] = []
        observed_rows: Dict[Tuple[Optional[int], str, float], Dict[str, Any]] = {}
        for rec in payload.get("results", []):
            cond = rec.get("condition", {})
            intervention_name = str(cond.get("intervention") or "")
            if intervention_name not in primary_interventions_set:
                continue
            cond_scale = _round_scale(cond.get("scale"))
            if cond_scale is None:
                continue
            if run_primary_scale_set is not None and cond_scale not in run_primary_scale_set:
                continue
            component_set_name = str(cond.get("component_set_name") or "")
            set_label = component_set_name.split(":", 1)[1] if ":" in component_set_name else component_set_name
            if require_primary_set and set_label != primary_set_name:
                continue
            k_value = _parse_k_from_component_set_name(component_set_name)
            if run_primary_k_set is not None and (k_value is None or k_value not in run_primary_k_set):
                continue
            row_key = (k_value, intervention_name, cond_scale)
            row_bucket = observed_rows.setdefault(
                row_key,
                {
                    "component_set_name": component_set_name,
                    "set_label": set_label,
                    "k_value": k_value,
                    "intervention": intervention_name,
                    "scale": cond_scale,
                    "target_rows": [],
                    "non_target_rows": [],
                },
            )
            for ds_name, ds_payload in rec.get("datasets", {}).items():
                ds_operator = ds_payload.get("operator")
                delta_rand = ds_payload.get("metrics", {}).get("delta_vs_random_accuracy_all")
                if not delta_rand:
                    continue
                ci_vals = delta_rand.get("ci", [0.0, 0.0])
                if not isinstance(ci_vals, (list, tuple)):
                    ci_vals = [0.0, 0.0]
                ci_low = float(ci_vals[0]) if len(ci_vals) > 0 else 0.0
                ci_high = float(ci_vals[1]) if len(ci_vals) > 1 else ci_low
                mean = float(delta_rand.get("mean", 0.0))
                item = {
                    "condition": cond.get("component_set_name"),
                    "intervention": cond.get("intervention"),
                    "scale": cond.get("scale"),
                    "dataset": ds_name,
                    "operator": ds_operator,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "set_label": set_label,
                    "k_value": k_value,
                }
                if ds_operator == target_operator:
                    primary_target_records.append(item)
                    row_bucket["target_rows"].append(item)
                else:
                    primary_non_target_records.append(item)
                    row_bucket["non_target_rows"].append(item)

        if run_primary_scale_set is None:
            run_primary_scale_set = {row_key[2] for row_key in observed_rows.keys()}
        expected_k_values: List[Optional[int]]
        if run_primary_k_set is not None:
            expected_k_values = sorted(run_primary_k_set)
        else:
            observed_k_values = sorted({row_key[0] for row_key in observed_rows.keys() if row_key[0] is not None})
            expected_k_values = observed_k_values if observed_k_values else [None]

        expected_interventions: List[str]
        if require_both_primary_interventions:
            expected_interventions = sorted(primary_interventions_set)
        else:
            observed_target_interventions = sorted(
                {
                    row_key[1]
                    for row_key, row_payload in observed_rows.items()
                    if row_payload.get("target_rows")
                }
            )
            expected_interventions = observed_target_interventions or sorted(primary_interventions_set)

        expected_condition_keys: List[Tuple[Optional[int], str, float]] = []
        for k_value in expected_k_values:
            for intervention_name in expected_interventions:
                for scale in sorted(run_primary_scale_set):
                    if not _intervention_matches_scale(intervention_name, scale):
                        continue
                    expected_condition_keys.append((k_value, intervention_name, scale))

        row_checks: List[Dict[str, Any]] = []
        missing_primary_rows: List[Dict[str, Any]] = []
        missing_non_target_rows: List[Dict[str, Any]] = []
        directionality_violations: List[Dict[str, Any]] = []
        expected_primary_q_keys: set[Tuple[Optional[int], str, float, str, str]] = set()
        observed_condition_keys: set[Tuple[Optional[int], str, float]] = set()

        passes = True
        failure_reason: Optional[str] = None
        for row_key in expected_condition_keys:
            row_payload = observed_rows.get(row_key)
            target_rows = list((row_payload or {}).get("target_rows") or [])
            non_target_rows = list((row_payload or {}).get("non_target_rows") or [])
            row_passes = True
            row_failure_reason: Optional[str] = None
            target_signed_vals = [_signed_mean(row_key[1], float(row.get("mean", 0.0))) for row in target_rows]
            non_target_signed_vals = [_signed_mean(row_key[1], float(row.get("mean", 0.0))) for row in non_target_rows]
            target_ci_support_vals = [
                _signed_ci_support(row_key[1], [row.get("ci_low", 0.0), row.get("ci_high", row.get("ci_low", 0.0))])
                for row in target_rows
            ]
            target_signed_mean = (
                sum(target_signed_vals) / len(target_signed_vals) if target_signed_vals else None
            )
            non_target_signed_mean = (
                sum(non_target_signed_vals) / len(non_target_signed_vals) if non_target_signed_vals else None
            )
            target_ci_support_min = min(target_ci_support_vals) if target_ci_support_vals else None
            gap = (
                target_signed_mean - non_target_signed_mean
                if target_signed_mean is not None and non_target_signed_mean is not None
                else None
            )

            if row_payload is None or not target_rows:
                row_passes = False
                row_failure_reason = "missing_target_evidence_for_intervention"
                missing_primary_rows.append(_row_key_to_dict(row_key))
            else:
                observed_condition_keys.add(row_key)
                for t_row in target_rows:
                    expected_primary_q_keys.add(
                        (
                            row_key[0],
                            row_key[1],
                            row_key[2],
                            str(t_row.get("dataset")),
                            str(target_operator),
                        )
                    )
                if target_ci_support_min is None or not (float(target_ci_support_min) > ci_low_min):
                    row_passes = False
                    row_failure_reason = "target_ci_low_below_threshold"
                elif require_non_target_operator_evidence and not non_target_rows:
                    row_passes = False
                    row_failure_reason = "missing_non_target_evidence_for_intervention"
                    missing_non_target_rows.append(_row_key_to_dict(row_key))
                elif (
                    require_non_target_operator_evidence
                    and non_target_rows
                    and not (gap is not None and gap > mean_gap_min)
                ):
                    row_passes = False
                    row_failure_reason = "target_non_target_gap_below_threshold"
                if row_passes and require_directionality:
                    violations_for_row: List[Dict[str, Any]] = []
                    for t_row in target_rows:
                        mean = float(t_row.get("mean", 0.0))
                        if row_key[1] == "ablation" and mean > ablation_delta_vs_random_max:
                            violations_for_row.append(
                                {
                                    **t_row,
                                    "reason": "ablation_improves_vs_random_above_max",
                                    "threshold": ablation_delta_vs_random_max,
                                }
                            )
                        elif row_key[1] == "amplification" and mean < amplification_delta_vs_random_min:
                            violations_for_row.append(
                                {
                                    **t_row,
                                    "reason": "amplification_worse_vs_random_below_min",
                                    "threshold": amplification_delta_vs_random_min,
                                }
                            )
                    if violations_for_row:
                        directionality_violations.extend(violations_for_row)
                        row_passes = False
                        row_failure_reason = "directionality_violation"

            row_checks.append(
                {
                    "row_key": _row_key_to_dict(row_key),
                    "n_target_rows": len(target_rows),
                    "n_non_target_rows": len(non_target_rows),
                    "target_signed_mean": target_signed_mean,
                    "non_target_signed_mean": non_target_signed_mean,
                    "target_signed_ci_support_min": target_ci_support_min,
                    "target_minus_non_target_gap": gap,
                    "passes": row_passes,
                    "failure_reason": None if row_passes else row_failure_reason,
                }
            )
            if not row_passes and failure_reason is None:
                failure_reason = row_failure_reason
            passes = passes and row_passes

        if require_non_target_operator_evidence and not primary_non_target_records:
            passes = False
            if failure_reason is None:
                failure_reason = "insufficient_non_target_scope"

        multiplicity_report = ((payload.get("analysis") or {}).get("multiplicity_report") or {})
        multiplicity_rows = multiplicity_report.get("rows") if isinstance(multiplicity_report, dict) else None
        observed_primary_q: Dict[Tuple[Optional[int], str, float, str, str], Optional[float]] = {}
        if isinstance(multiplicity_rows, list):
            for row in multiplicity_rows:
                if not isinstance(row, dict):
                    continue
                if require_primary_set and row.get("set_label") != primary_set_name:
                    continue
                if row.get("operator") != target_operator:
                    continue
                if row.get("intervention") not in primary_interventions_set:
                    continue
                row_scale = _round_scale(row.get("scale"))
                if row_scale is None:
                    continue
                if run_primary_scale_set is not None and row_scale not in run_primary_scale_set:
                    continue
                row_k = None
                if row.get("k_value") is not None:
                    try:
                        row_k = int(row.get("k_value"))
                    except Exception:
                        row_k = None
                if row_k is None and row.get("k_label") is not None:
                    text = str(row.get("k_label"))
                    if text.startswith("K") and text[1:].isdigit():
                        row_k = int(text[1:])
                if row_k is None:
                    row_k = _parse_k_from_component_set_name(str(row.get("condition") or ""))
                if run_primary_k_set is not None and (row_k is None or row_k not in run_primary_k_set):
                    continue
                is_primary = row.get("is_primary_comparison")
                if is_primary is False:
                    continue
                dataset_name = str(row.get("dataset") or "")
                q_key = (row_k, str(row.get("intervention")), row_scale, dataset_name, str(target_operator))
                q_val = row.get("q_value_primary")
                if q_val is None:
                    q_val = row.get("q_value")
                q_float: Optional[float] = None
                try:
                    if q_val is not None:
                        q_float = float(q_val)
                        if not math.isfinite(q_float):
                            q_float = None
                except Exception:
                    q_float = None
                observed_primary_q[q_key] = q_float

        expected_primary_q_keys_sorted = sorted(expected_primary_q_keys)
        primary_q_values: List[float] = []
        missing_primary_q_rows: List[Dict[str, Any]] = []
        for q_key in expected_primary_q_keys_sorted:
            q_val = observed_primary_q.get(q_key)
            if q_val is None:
                missing_primary_q_rows.append(
                    {
                        "k_value": q_key[0],
                        "intervention": q_key[1],
                        "scale": q_key[2],
                        "dataset": q_key[3],
                        "operator": q_key[4],
                    }
                )
                continue
            primary_q_values.append(float(q_val))

        best_primary_q = min(primary_q_values) if primary_q_values else None
        worst_primary_q = max(primary_q_values) if primary_q_values else None
        multiplicity_passes = True
        multiplicity_failure_reason = None
        if multiplicity_blocking_enabled:
            if not expected_primary_q_keys_sorted:
                multiplicity_passes = False
                multiplicity_failure_reason = "missing_primary_q_coverage"
            elif multiplicity_require_complete_primary_coverage and missing_primary_q_rows:
                multiplicity_passes = False
                multiplicity_failure_reason = "missing_primary_q_coverage"
            elif best_primary_q is None:
                multiplicity_passes = False
                multiplicity_failure_reason = "missing_primary_q_coverage"
            elif worst_primary_q is not None and worst_primary_q > multiplicity_q_max:
                multiplicity_passes = False
                multiplicity_failure_reason = "primary_q_above_threshold"
        if not multiplicity_passes:
            passes = False
            if failure_reason is None:
                failure_reason = multiplicity_failure_reason

        representative_target = (
            max(
                primary_target_records,
                key=lambda row: _signed_mean(str(row.get("intervention")), float(row.get("mean", 0.0))),
            )
            if primary_target_records
            else None
        )
        representative_non_target = (
            max(
                primary_non_target_records,
                key=lambda row: _signed_mean(str(row.get("intervention")), float(row.get("mean", 0.0))),
            )
            if primary_non_target_records
            else None
        )
        checks.append({
            "run": key,
            "target_operator": target_operator,
            "best_target": representative_target,
            "best_non_target": representative_non_target,
            "primary_target_records_count": len(primary_target_records),
            "primary_non_target_records_count": len(primary_non_target_records),
            "condition_policy": condition_policy,
            "primary_grid": {
                "primary_k_values": sorted(run_primary_k_set) if run_primary_k_set is not None else None,
                "primary_scales": sorted(run_primary_scale_set) if run_primary_scale_set is not None else None,
                "primary_interventions": sorted(expected_interventions),
                "n_expected_rows": len(expected_condition_keys),
                "n_observed_rows": len(observed_condition_keys),
            },
            "row_checks": row_checks,
            "intervention_checks": row_checks,
            "missing_primary_rows": missing_primary_rows,
            "missing_non_target_rows": missing_non_target_rows,
            "directionality_violations": directionality_violations,
            "multiplicity_summary": {
                "blocking_enabled": multiplicity_blocking_enabled,
                "best_primary_q": best_primary_q,
                "worst_primary_q": worst_primary_q,
                "n_expected_primary_q_rows": len(expected_primary_q_keys_sorted),
                "n_observed_primary_q_rows": len(primary_q_values),
                "missing_primary_q_rows": missing_primary_q_rows,
                "n_primary_q_values": len(primary_q_values),
                "q_max": multiplicity_q_max,
                "require_complete_primary_coverage": multiplicity_require_complete_primary_coverage,
                "passes": multiplicity_passes,
                "failure_reason": multiplicity_failure_reason,
            },
            "thresholds": {
                "ci_low_min": ci_low_min,
                "mean_gap_min": mean_gap_min,
                "require_non_target_operator_evidence": require_non_target_operator_evidence,
                "require_primary_set": require_primary_set,
                "primary_set_name": primary_set_name,
                "require_directionality": require_directionality,
                "require_both_primary_interventions": require_both_primary_interventions,
                "primary_interventions": sorted(primary_interventions_set),
                "primary_scales": sorted(run_primary_scale_set) if run_primary_scale_set is not None else None,
                "primary_k_values": sorted(run_primary_k_set) if run_primary_k_set is not None else None,
                "condition_policy": condition_policy,
                "sign_policy": sign_policy,
                "ablation_delta_vs_random_max": ablation_delta_vs_random_max,
                "amplification_delta_vs_random_min": amplification_delta_vs_random_min,
                "multiplicity_blocking_enabled": multiplicity_blocking_enabled,
                "multiplicity_q_max": multiplicity_q_max,
                "multiplicity_require_complete_primary_coverage": multiplicity_require_complete_primary_coverage,
            },
            "passes": passes,
            "failure_reason": None if passes else failure_reason,
        })
        passes_all = passes_all and passes
    return {
        "passes": bool(checks) and passes_all,
        "checks": checks,
        "primary_set_policy": {
            "enabled": require_primary_set,
            "primary_set_name": primary_set_name,
            "require_directionality": require_directionality,
            "require_both_primary_interventions": require_both_primary_interventions,
            "primary_interventions": sorted(primary_interventions_set),
            "primary_scales": sorted(primary_scale_set) if primary_scale_set is not None else None,
            "primary_k_values": sorted(primary_k_set) if primary_k_set is not None else None,
            "condition_policy": condition_policy,
            "sign_policy": sign_policy,
        },
        "multiplicity_policy": {
            "blocking_enabled": multiplicity_blocking_enabled,
            "q_max": multiplicity_q_max,
            "require_complete_primary_coverage": multiplicity_require_complete_primary_coverage,
        },
    }


def _evaluate_intervention_sanity_gate(
    intervention_payloads: Mapping[str, Dict[str, Any]],
    *,
    enabled: bool,
) -> Dict[str, Any]:
    checks = []
    if not enabled:
        for key in sorted(intervention_payloads.keys()):
            checks.append({"run": key, "passes": True, "status": "disabled"})
        return {"passes": True, "checks": checks, "status": "disabled"}
    passes_all = True
    for key, payload in intervention_payloads.items():
        summary = payload.get("sanity_summary", {})
        flagged = int(summary.get("total_flagged_datasets") or 0)
        check_pass = flagged == 0
        checks.append(
            {
                "run": key,
                "status": payload.get("status"),
                "flagged_datasets": flagged,
                "flag_type_counts": summary.get("flag_type_counts", {}),
                "passes": check_pass,
            }
        )
        passes_all = passes_all and check_pass
    return {"passes": bool(checks) and passes_all, "checks": checks}


def _build_intervention_anomaly_report(
    intervention_payloads: Mapping[str, Dict[str, Any]],
    *,
    sample_cap_per_dataset: int = 3,
) -> Dict[str, Any]:
    report_runs: Dict[str, Any] = {}
    total_flagged_conditions = 0
    total_flagged_datasets = 0
    flag_type_counts: Dict[str, int] = {}
    for run_key, payload in intervention_payloads.items():
        run_entry = {
            "status": payload.get("status"),
            "flagged_conditions": [],
        }
        for record in payload.get("results", []):
            condition_flags = list(record.get("sanity_flags") or [])
            if not condition_flags:
                continue
            datasets_payload: Dict[str, Any] = {}
            for ds_name, ds_payload in (record.get("datasets") or {}).items():
                ds_flags = list(ds_payload.get("sanity_flags") or [])
                if not ds_flags:
                    continue
                metrics = ds_payload.get("metrics", {})
                datasets_payload[ds_name] = {
                    "flags": ds_flags,
                    "operator": ds_payload.get("operator"),
                    "bucket": ds_payload.get("bucket"),
                    "accuracy_all": metrics.get("accuracy_all"),
                    "parse_rate": metrics.get("parse_rate"),
                    "delta_vs_baseline_accuracy_all": metrics.get("delta_vs_baseline_accuracy_all"),
                    "delta_vs_random_accuracy_all": metrics.get("delta_vs_random_accuracy_all"),
                    "prediction_samples": list(ds_payload.get("prediction_samples") or [])[:sample_cap_per_dataset],
                }
                total_flagged_datasets += 1
                for flag in ds_flags:
                    flag_type_counts[flag] = flag_type_counts.get(flag, 0) + 1
            run_entry["flagged_conditions"].append(
                {
                    "condition": record.get("condition"),
                    "flags": condition_flags,
                    "datasets": datasets_payload,
                }
            )
            total_flagged_conditions += 1
        report_runs[run_key] = run_entry
    return {
        "schema_version": "phase2_intervention_anomaly_report_v1",
        "runs": report_runs,
        "summary": {
            "total_runs": len(report_runs),
            "total_flagged_conditions": total_flagged_conditions,
            "total_flagged_datasets": total_flagged_datasets,
            "flag_type_counts": flag_type_counts,
        },
    }


def _evaluate_cot_gate(
    cot_results: Mapping[str, Dict[str, Any]],
    *,
    effect_abs_min: float,
    parse_rate_delta_abs_max: float,
    min_pairs: int = 0,
    parse_rate_min: float = 0.0,
    require_accuracy_ci_excludes_zero: bool = False,
) -> Dict[str, Any]:
    checks = []
    passes_all = True
    for key, payload in cot_results.items():
        if payload.get("status") != "ok":
            checks.append(
                {
                    "run": key,
                    "status": payload.get("status"),
                    "reason": "; ".join(payload.get("notes", [])[:2]) if isinstance(payload.get("notes"), list) else None,
                    "passes": False,
                    "failure_reason": "cot_not_executed",
                }
            )
            passes_all = False
            continue
        direct = payload.get("direct_metrics", {})
        cot = payload.get("cot_metrics", {})
        sens = payload.get("sensitivity_deltas", {})
        n_pairs = int(payload.get("n_pairs") or 0)
        parse_control_present = ("parse_rate" in direct) and ("parse_rate" in cot)
        accuracy_delta = sens.get("baseline_direct_vs_cot", {}).get("accuracy_all_delta")
        accuracy_delta_ci = sens.get("baseline_direct_vs_cot", {}).get("accuracy_all_delta_ci")
        component_delta = sens.get("component_intervention", {}).get("cot_delta_vs_baseline")
        parse_rate_delta = sens.get("baseline_direct_vs_cot", {}).get("parse_rate_delta")
        direct_parse_rate = direct.get("parse_rate")
        cot_parse_rate = cot.get("parse_rate")
        if parse_rate_delta is None and parse_control_present:
            parse_rate_delta = (cot.get("parse_rate") or 0.0) - (direct.get("parse_rate") or 0.0)
        component_abs = abs(float(component_delta)) if component_delta is not None else None
        accuracy_abs = abs(float(accuracy_delta)) if accuracy_delta is not None else None
        ci_excludes_zero: Optional[bool] = None
        if isinstance(accuracy_delta_ci, (list, tuple)) and len(accuracy_delta_ci) >= 2:
            ci_low = float(accuracy_delta_ci[0])
            ci_high = float(accuracy_delta_ci[1])
            ci_excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
        effect_present = bool(
            (accuracy_abs is not None and accuracy_abs >= effect_abs_min)
            or (component_abs is not None and component_abs >= effect_abs_min)
        )
        parse_control_ok = bool(
            parse_control_present
            and parse_rate_delta is not None
            and abs(float(parse_rate_delta)) <= parse_rate_delta_abs_max
        )
        parse_rate_floor_ok = bool(
            parse_control_present
            and direct_parse_rate is not None
            and cot_parse_rate is not None
            and float(direct_parse_rate) >= parse_rate_min
            and float(cot_parse_rate) >= parse_rate_min
        )
        pair_count_ok = bool(n_pairs >= int(min_pairs))
        ci_ok = True if not require_accuracy_ci_excludes_zero else bool(ci_excludes_zero)
        passes = bool(effect_present and parse_control_ok and parse_rate_floor_ok and pair_count_ok and ci_ok)
        failure_reason = None
        if not pair_count_ok:
            failure_reason = "insufficient_pairs"
        elif not parse_control_present:
            failure_reason = "missing_parse_control"
        elif parse_rate_delta is None:
            failure_reason = "missing_parse_rate_delta"
        elif not parse_rate_floor_ok:
            failure_reason = "parse_rate_below_min"
        elif not parse_control_ok:
            failure_reason = "parse_rate_delta_too_large"
        elif not effect_present:
            failure_reason = "effect_below_threshold"
        elif not ci_ok:
            failure_reason = "accuracy_ci_includes_zero"
        checks.append({
            "run": key,
            "n_pairs": n_pairs,
            "parse_control_present": parse_control_present,
            "accuracy_all_delta": accuracy_delta,
            "accuracy_all_delta_ci": accuracy_delta_ci,
            "accuracy_ci_excludes_zero": ci_excludes_zero,
            "component_cot_delta_vs_baseline": component_delta,
            "direct_parse_rate": direct_parse_rate,
            "cot_parse_rate": cot_parse_rate,
            "parse_rate_delta": parse_rate_delta,
            "thresholds": {
                "effect_abs_min": effect_abs_min,
                "parse_rate_delta_abs_max": parse_rate_delta_abs_max,
                "min_pairs": min_pairs,
                "parse_rate_min": parse_rate_min,
                "require_accuracy_ci_excludes_zero": require_accuracy_ci_excludes_zero,
            },
            "passes": passes,
            "failure_reason": None if passes else failure_reason,
        })
        passes_all = passes_all and passes
    return {
        "passes": bool(checks) and passes_all,
        "checks": checks,
        "thresholds": {
            "effect_abs_min": effect_abs_min,
            "parse_rate_delta_abs_max": parse_rate_delta_abs_max,
            "min_pairs": min_pairs,
            "parse_rate_min": parse_rate_min,
            "require_accuracy_ci_excludes_zero": require_accuracy_ci_excludes_zero,
        },
    }


def _load_model_if_needed(args: argparse.Namespace, cfg: Dict[str, Any]):
    return load_local_model(cfg["model"]["name"], cache_dir=args.cache_dir, local_files_only=True)


def _build_localization_config(
    *,
    component_type: str,
    operator: str,
    buckets: Sequence[str],
    cfg: Dict[str, Any],
    seed: int,
    batch_size: Optional[int] = None,
) -> LocalizationConfig:
    metric_targets = cfg["localization"].get("metric_targets", ["answer_token"])
    if "both" in metric_targets:
        mt = "both"
    elif "per_digit" in metric_targets and "answer_token" in metric_targets:
        mt = "both"
    elif "per_digit" in metric_targets:
        mt = "per_digit"
    else:
        mt = "answer_token"
    stability_modes = cfg["localization"].get("stability_modes", ["same_set_shuffle_invariance"])
    stability_mode = "+".join(stability_modes)
    return LocalizationConfig(
        component_type=component_type,
        operator_filters=[operator],
        bucket_filters=list(buckets),
        metric_targets=mt,
        batch_size=int(batch_size if batch_size is not None else cfg["runtime"]["batch_size"]),
        seed=seed,
        component_sampling_seed=(
            int(cfg["localization"]["component_sampling_seed"])
            if cfg["localization"].get("component_sampling_seed") is not None
            else int(seed)
        ),
        stability_mode=stability_mode,
        strict_attention_heads=bool(cfg["localization"].get("strict_attention_heads", True)),
    )


def _localization_component_options(cfg: Dict[str, Any], component_type: str) -> Dict[str, Any]:
    options = cfg["localization"].get("component_options", {})
    if isinstance(options, dict):
        sub = options.get(component_type)
        if isinstance(sub, dict):
            return dict(sub)
    return {}


def _resolve_cross_operator_source(path_text: Optional[str], output_root: Path) -> Path:
    if path_text:
        candidate = Path(path_text).expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
    else:
        candidate = output_root / "phase2_localization.json"
    if candidate.is_dir():
        candidate = candidate / "phase2_localization.json"
    return candidate


def _load_component_sets_index_from_localization(path: Path) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    component_sets = payload.get("component_sets")
    if not isinstance(component_sets, dict):
        raise ValueError(f"Missing component_sets in localization source: {path}")
    out: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for operator, op_payload in component_sets.items():
        if not isinstance(op_payload, dict):
            continue
        out[str(operator)] = {}
        for component_type, sets_for_type in op_payload.items():
            if not isinstance(sets_for_type, dict):
                continue
            out[str(operator)][str(component_type)] = {
                str(k): [str(cid) for cid in cids]
                for k, cids in sets_for_type.items()
                if isinstance(cids, list)
            }
    return out


def _filter_component_sets_for_primary_k(
    component_sets: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    primary_k_values: Sequence[int],
) -> Dict[str, Dict[str, List[str]]]:
    if not primary_k_values:
        return {
            str(k): {str(name): [str(cid) for cid in ids] for name, ids in sets.items()}
            for k, sets in component_sets.items()
            if isinstance(sets, Mapping)
        }
    allowed = {f"K{int(k)}" for k in primary_k_values}
    out: Dict[str, Dict[str, List[str]]] = {}
    for key, sets in component_sets.items():
        if str(key) not in allowed:
            continue
        if not isinstance(sets, Mapping):
            continue
        out[str(key)] = {str(name): [str(cid) for cid in ids] for name, ids in sets.items()}
    return out


def _scaffold_stage_outputs(output_root: Path, cfg: Dict[str, Any], stage: str) -> None:
    # Preserve the CPU-only validation behavior for tests and environments without model access.
    from src.arithmetic_localization import build_localization_not_implemented_result
    from src.operator_interventions import build_cross_operator_specificity_matrix, build_intervention_not_implemented_result
    from src.cot_recruitment import build_cot_compare_not_implemented_result

    if stage in {"localize", "full", "intervene", "cot_compare", "cross_operator_verify"}:
        loc_cfg = LocalizationConfig(
            component_type="attention_heads",
            operator_filters=list(cfg["datasets"]["operator_buckets"].keys()),
            bucket_filters=[b for bs in cfg["datasets"]["operator_buckets"].values() for b in bs],
            metric_targets="answer_token",
            batch_size=int(cfg["runtime"]["batch_size"]),
            seed=int(cfg["runtime"]["seeds"][0]),
            stability_mode="same_set_shuffle_invariance",
            strict_attention_heads=True,
        )
        _json_dump(
            output_root / "phase2_localization.json",
            build_localization_not_implemented_result(
                model=cfg["model"]["name"],
                config=loc_cfg,
                prompt_set={"source": "dataset_manifest.json"},
                reason="Scaffold mode enabled (--scaffold-gpu-stages).",
            ),
        )
    if stage in {"intervene", "full", "cross_operator_verify"}:
        buckets = [f"{op}__{b}" for op, lst in cfg["datasets"]["operator_buckets"].items() for b in lst]
        _json_dump(
            output_root / "phase2_interventions.json",
            build_intervention_not_implemented_result(
                model=cfg["model"]["name"],
                component_set_source="phase2_localization.json",
                task_buckets=buckets,
                reason="Scaffold mode enabled (--scaffold-gpu-stages).",
            ),
        )
        _json_dump(
            output_root / "phase2_cross_operator_specificity_matrix.json",
            build_cross_operator_specificity_matrix(
                rows=["add", "sub", "mul", "random", "induction_baseline"],
                cols=buckets,
                values={},
            ),
        )
        _json_dump(
            output_root / "phase2_cross_operator_verify.json",
            {
                "schema_version": "phase2_cross_operator_verify_v1",
                "status": "not_implemented",
                "evidence_source": "post_merge_cross_operator_verify",
                "runs": {},
                "notes": ["Scaffold mode enabled (--scaffold-gpu-stages)."],
            },
        )
    if stage in {"cot_compare", "full"}:
        _json_dump(
            output_root / "phase2_cot_recruitment_compare.json",
            build_cot_compare_not_implemented_result(
                model=cfg["model"]["name"],
                reason="Scaffold mode enabled (--scaffold-gpu-stages).",
            ),
        )


def _dataset_subset(dataset: Any, *, max_examples: int, split_label: str) -> Any:
    clipped = list(dataset.examples[: max(1, int(max_examples))])
    return type(dataset)(
        operator=dataset.operator,
        bucket=dataset.bucket,
        examples=clipped,
        seed=dataset.seed,
        representation_variant=dataset.representation_variant,
        metadata={**dict(dataset.metadata), "autotune_probe_split": split_label},
    )


def _autotune_localization_signature(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = list(payload.get("metrics", []))
    # Stable ordering gives deterministic equivalence checks.
    metrics = sorted(metrics, key=lambda row: str(row.get("component_id")))[:16]
    return {
        "metric_signature": [
            float(row.get("answer_token_prob_delta_mean") or 0.0)
            for row in metrics
        ],
        "logit_signature": [
            float(row.get("next_token_kl_mean") or 0.0)
            for row in metrics
        ],
    }


def _autotune_eval_signature(payload: Mapping[str, Any]) -> Dict[str, Any]:
    per_prompt = list(payload.get("results", []))[:16]
    parsed_vals = []
    correct_vals = []
    for row in per_prompt:
        parsed = row.get("parsed")
        try:
            parsed_vals.append(float(parsed))
        except Exception:
            parsed_vals.append(0.0)
        correct_vals.append(1.0 if row.get("correct") else 0.0)
    return {
        "metric_signature": [
            float(payload.get("accuracy_all") or 0.0),
            float(payload.get("parse_rate") or 0.0),
            float(payload.get("generation_policy", {}).get("sampling_fallback_count") or 0.0),
        ]
        + correct_vals,
        "logit_signature": parsed_vals,
    }


def _build_batch_autotune_runtime(
    cfg: Mapping[str, Any],
    *,
    cli_enabled: Optional[bool],
    cli_min: Optional[int],
    cli_max: Optional[int],
    cli_stage_text: Optional[str],
    cli_equivalence: Optional[bool],
) -> Dict[str, Any]:
    bt = dict(cfg["runtime"]["batch_autotune"])
    if cli_enabled is not None:
        bt["enabled"] = bool(cli_enabled)
    if cli_min is not None:
        bt["min_batch_size"] = int(cli_min)
    if cli_max is not None:
        bt["max_batch_size"] = int(cli_max)
    if cli_stage_text is not None:
        parsed_stages = [item.strip() for item in cli_stage_text.split(",") if item.strip()]
        bt["stages"] = parsed_stages or list(bt.get("stages", []))
    if cli_equivalence is not None:
        eq = dict(bt.get("equivalence_check", {}))
        eq["enabled"] = bool(cli_equivalence)
        bt["equivalence_check"] = eq
    return bt


def _autotune_stage_enabled(bt_cfg: Mapping[str, Any], stage: str) -> bool:
    if not bool(bt_cfg.get("enabled", False)):
        return False
    stages = bt_cfg.get("stages", [])
    if not isinstance(stages, list):
        return False
    return stage in {str(s) for s in stages}


def main() -> None:
    args = parse_args()
    cfg_raw = load_config_file(args.dataset_config)
    cfg = _normalize_phase2_config(cfg_raw, smoke=args.smoke)

    model_name = args.model or cfg["model"]["name"]
    cfg["model"]["name"] = model_name
    if args.devices is not None:
        cfg["runtime"]["devices"] = args.devices
    if args.batch_size is not None:
        cfg["runtime"]["batch_size"] = int(args.batch_size)
    cfg["runtime"]["seeds"] = _parse_seed_list(args.seed_list, default=list(cfg["runtime"]["seeds"]))
    cfg["runtime"]["batch_autotune"] = _build_batch_autotune_runtime(
        cfg,
        cli_enabled=args.batch_autotune,
        cli_min=args.batch_autotune_min,
        cli_max=args.batch_autotune_max,
        cli_stage_text=args.batch_autotune_stages,
        cli_equivalence=args.batch_equivalence_check,
    )
    if args.operator_shard_mode:
        cfg["runtime"]["operator_shard_mode"] = True

    configured_operator_buckets = dict(cfg["datasets"]["operator_buckets"])
    available_operators = sorted(configured_operator_buckets.keys())
    cli_operator_filter = _parse_operator_list(args.operators)
    cfg_operator_filter = _parse_operator_list(None, default=cfg["runtime"].get("operator_filter"))
    operator_filter = cli_operator_filter if cli_operator_filter is not None else cfg_operator_filter
    if operator_filter is not None:
        unknown = sorted(set(operator_filter) - set(available_operators))
        if unknown:
            raise ValueError(f"Unknown operators requested via --operators/operator_filter: {unknown}")
        cfg["datasets"]["operator_buckets"] = {
            op: configured_operator_buckets[op]
            for op in available_operators
            if op in set(operator_filter)
        }
        cfg["runtime"]["operator_filter"] = sorted(cfg["datasets"]["operator_buckets"].keys())
    else:
        cfg["runtime"]["operator_filter"] = None
    operator_coverage = sorted(cfg["datasets"]["operator_buckets"].keys())
    operator_scope = "full" if operator_coverage == available_operators else "subset"
    is_operator_shard = bool(cfg["runtime"].get("operator_shard_mode", False))
    merge_required_for_full_claims = bool(is_operator_shard and operator_scope != "full")

    if cfg["runtime"].get("devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["runtime"]["devices"])

    low_cpu_mode = bool(args.low_cpu_mode or cfg["runtime"].get("low_cpu_mode", False))
    cpu_runtime = _apply_cpu_runtime_controls(low_cpu_mode=low_cpu_mode, max_cpu_threads=args.max_cpu_threads)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else (ROOT / "results" / "phase2" / f"operator_bottleneck_run_{ts}")
    output_root.mkdir(parents=True, exist_ok=True)
    prereg = _load_preregistration_payload(cfg["analysis"]["preregistration_config"])
    _json_dump(output_root / "preregistration_used.json", prereg)

    run_manifest = {
        "schema_version": "phase2_operator_bottleneck_run_manifest_v1",
        "schema_revision": "2.1",
        "timestamp_utc": _timestamp_utc(),
        "stage": args.stage,
        "resume_from": args.resume_from,
        "smoke": bool(args.smoke),
        "scaffold_gpu_stages": bool(args.scaffold_gpu_stages),
        "cli": {
            "model": args.model,
            "cache_dir": args.cache_dir,
            "devices": args.devices,
            "dataset_config": args.dataset_config,
            "output_root": str(output_root),
            "seed_list": args.seed_list,
            "batch_size": args.batch_size,
            "operators": args.operators,
            "operator_shard_mode": bool(args.operator_shard_mode),
            "batch_autotune": args.batch_autotune,
            "batch_autotune_min": args.batch_autotune_min,
            "batch_autotune_max": args.batch_autotune_max,
            "batch_autotune_stages": args.batch_autotune_stages,
            "batch_equivalence_check": args.batch_equivalence_check,
            "low_cpu_mode": args.low_cpu_mode,
            "max_cpu_threads": args.max_cpu_threads,
            "cross_operator_source": args.cross_operator_source,
        },
        "effective_config": cfg,
        "cpu_runtime_controls": cpu_runtime,
        "operator_scope": operator_scope,
        "operator_coverage": operator_coverage,
        "is_operator_shard": is_operator_shard,
        "merge_required_for_full_claims": merge_required_for_full_claims,
        "tuned_batch_sizes_by_stage": {},
        "autotune_probe_history": {},
        "equivalence_check_results": {},
        "preregistered_primary_comparisons": {
            "primary_component_set": str(cfg["interventions"]["primary_component_set"]),
            "primary_interventions": list(cfg["interventions"]["primary_interventions"]),
            "primary_scales": list(cfg["interventions"]["primary_scales"]),
            "primary_k_values": list(cfg["interventions"]["primary_k_values"]),
            "primary_target_metric": "delta_vs_random_accuracy_all",
            "specificity_primary_set": str(cfg["gates"]["specificity_primary_set"]),
        },
        "preregistration": {
            "config_path": str(prereg.get("source_path")),
            "artifact": "preregistration_used.json",
        },
        "analysis_artifacts": {},
    }
    _json_dump(output_root / "run_manifest.json", run_manifest)

    gate_summary = _phase2_gate_template()
    gate_summary["run_metadata"] = {
        "timestamp_utc": _timestamp_utc(),
        "output_root": str(output_root),
        "stage": args.stage,
        "resume_from": args.resume_from,
        "model": model_name,
        "cpu_runtime_controls": cpu_runtime,
        "scaffold_gpu_stages": bool(args.scaffold_gpu_stages),
        "operator_scope": operator_scope,
        "operator_coverage": operator_coverage,
        "is_operator_shard": is_operator_shard,
        "tuned_batch_sizes_by_stage": {},
        "autotune_probe_history": {},
        "equivalence_check_results": {},
    }
    gate_summary["required_gates_policy"] = {
        "cot_required_for_readiness": bool(cfg["gates"]["cot_required_for_readiness"]),
        "require_all_component_types": bool(cfg["gates"]["require_all_component_types"]),
        "require_non_target_operator_evidence": bool(cfg["gates"]["require_non_target_operator_evidence"]),
        "anomaly_blocking_enabled": bool(cfg["gates"]["anomaly_blocking"]["enabled"]),
        "specificity_requires_primary_set": bool(cfg["gates"]["specificity_requires_primary_set"]),
        "specificity_primary_set": str(cfg["gates"]["specificity_primary_set"]),
        "specificity_requires_directionality": bool(cfg["gates"]["specificity_requires_directionality"]),
        "specificity_requires_both_primary_interventions": bool(cfg["gates"]["specificity_requires_both_primary_interventions"]),
        "specificity_condition_policy": str(cfg["gates"]["specificity_condition_policy"]),
        "specificity_sign_policy": str(cfg["gates"]["specificity_sign_policy"]),
        "primary_interventions": list(cfg["interventions"]["primary_interventions"]),
        "primary_scales": list(cfg["interventions"]["primary_scales"]),
        "primary_k_values": list(cfg["interventions"]["primary_k_values"]),
        "scope_block_on_single_operator": bool(cfg["gates"]["scope_block_on_single_operator"]),
        "multiplicity_blocking_enabled": bool(cfg["gates"]["multiplicity_blocking"]["enabled"]),
        "multiplicity_q_max": float(cfg["gates"]["multiplicity_blocking"]["q_max"]),
        "multiplicity_require_complete_primary_coverage": bool(
            cfg["gates"]["multiplicity_require_complete_primary_coverage"]
        ),
        "split_hard_stop_on_failure": bool(
            cfg["datasets"]["target_operator_selection_eval_split"]["hard_stop_on_failure"]
        ),
        "batch_autotune": cfg["runtime"]["batch_autotune"],
    }
    gate_summary["required_gates_policy"]["required_for_readiness"] = _required_gates_for_readiness(cfg)
    gate_summary["scope"] = {
        "operator_coverage": list(operator_coverage),
        "is_sharded_run": bool(is_operator_shard),
        "merge_required_for_full_claims": bool(merge_required_for_full_claims),
    }
    if merge_required_for_full_claims:
        if "operator_shard_requires_merge" not in gate_summary["scope_blocks"]:
            gate_summary["scope_blocks"].append("operator_shard_requires_merge")
        gate_summary["scope_warnings"].append(
            "Operator shard mode is enabled on a subset scope; merge shard outputs before making readiness claims."
        )

    if args.stage == "cross_operator_verify":
        prior_gate_path = output_root / "phase2_gate_summary.json"
        if prior_gate_path.exists():
            try:
                prior = json.loads(prior_gate_path.read_text(encoding="utf-8"))
            except Exception:
                prior = {}
            prior_phases = prior.get("phases", {}) if isinstance(prior, dict) else {}
            for phase_name in (
                "dataset_bucket_gate",
                "localization_validity_gate",
                "intervention_sanity_gate",
                "cot_gating_evidence_gate",
            ):
                if isinstance(prior_phases.get(phase_name), dict):
                    gate_summary["phases"][phase_name] = dict(prior_phases[phase_name])
            if isinstance(prior.get("derived_thresholds"), dict):
                gate_summary["derived_thresholds"].update(dict(prior["derived_thresholds"]))
            if isinstance(prior.get("scope_warnings"), list):
                gate_summary["scope_warnings"] = list(dict.fromkeys(list(prior["scope_warnings"]) + gate_summary["scope_warnings"]))
            if isinstance(prior.get("scope_blocks"), list):
                gate_summary["scope_blocks"] = list(dict.fromkeys(list(prior["scope_blocks"]) + gate_summary["scope_blocks"]))

    seeds = cfg["runtime"]["seeds"]
    primary_seed = int(seeds[0])
    tuned_batch_sizes_by_stage: Dict[str, int] = {}
    autotune_probe_history: Dict[str, Any] = {}
    equivalence_check_results: Dict[str, Any] = {}

    # Stage 1: datasets (always needed for downstream stages)
    dataset_stage_output = _run_dataset_stage(output_root, cfg, seed=primary_seed)
    suite: OperatorBucketSuite = dataset_stage_output["suite"]
    min_required = int(cfg["gates"]["dataset_bucket_min_examples"])
    counts_ok = all(int(v) >= min_required for v in dataset_stage_output["manifest"].get("counts_by_bucket", {}).values())
    gate_summary["phases"]["dataset_bucket_gate"] = {
        "ran": True,
        "passes": bool(counts_ok),
        "skipped": False,
        "dataset_manifest": "dataset_manifest.json",
        "dataset_diagnostics": "dataset_diagnostics.json",
        "min_required": min_required,
        "counts_by_bucket": dataset_stage_output["manifest"].get("counts_by_bucket", {}),
    }
    power_report = build_power_analysis_report(prereg, dataset_manifest=dataset_stage_output["manifest"])
    _json_dump(output_root / "power_analysis_report.json", power_report)
    run_manifest["analysis_artifacts"]["power_analysis_report"] = "power_analysis_report.json"
    gate_summary["run_metadata"]["preregistration_used"] = "preregistration_used.json"
    gate_summary["run_metadata"]["power_analysis_report"] = "power_analysis_report.json"
    operators_present = sorted(dataset_stage_output["manifest"].get("counts_by_operator", {}).keys())
    if bool(cfg["gates"]["require_non_target_operator_evidence"]) and len(operators_present) < 2:
        gate_summary["scope_warnings"].append(
            "Run scope includes only one operator family; full cross-operator specificity evidence is unavailable."
        )
        if bool(cfg["gates"]["scope_block_on_single_operator"]):
            gate_summary["scope_blocks"].append("single_operator_scope_blocks_specificity")

    if args.stage == "datasets":
        readiness_block_reasons = ["stage_incomplete:datasets_only"]
        for scope_block in gate_summary.get("scope_blocks", []):
            readiness_block_reasons.append(f"scope_block:{scope_block}")
        gate_summary["overall"].update(
            {
                "ready_for_multimodel_next_tranche": False,
                "phase2_status": "dataset_stage_only_complete",
                "readiness_block_reasons": readiness_block_reasons,
            }
        )
        run_manifest["required_gates_policy"] = gate_summary.get("required_gates_policy", {})
        run_manifest["derived_thresholds"] = gate_summary.get("derived_thresholds", {})
        run_manifest["tuned_batch_sizes_by_stage"] = dict(tuned_batch_sizes_by_stage)
        run_manifest["autotune_probe_history"] = dict(autotune_probe_history)
        run_manifest["equivalence_check_results"] = dict(equivalence_check_results)
        _json_dump(output_root / "run_manifest.json", run_manifest)
        _json_dump(output_root / "phase2_gate_summary.json", gate_summary)
        _write_replication_protocol(output_root)
        print(f"Wrote Phase 2 outputs to {output_root}")
        print("Stage: datasets")
        print(f"Dataset gate pass: {gate_summary['phases']['dataset_bucket_gate']['passes']}")
        return

    operator_views = _operator_datasets_view(suite)
    split_cfg = cfg["datasets"]["target_operator_selection_eval_split"]
    operator_run_views: Dict[str, Dict[str, Any]] = {}
    selection_eval_split_summary: Dict[str, Any] = {}
    split_leakage_failures: List[Dict[str, Any]] = []
    split_require_strict = bool(split_cfg.get("require_strict", True))
    split_hard_stop_on_failure = bool(split_cfg.get("hard_stop_on_failure", True))
    for op_name, _ in sorted(operator_views.items()):
        if bool(split_cfg.get("enabled", True)):
            split_payload = _split_target_operator_datasets_for_selection_eval(
                suite,
                operator=op_name,
                holdout_fraction=float(split_cfg.get("holdout_fraction", 0.5)),
                min_examples_per_split=int(split_cfg.get("min_examples_per_split", 8)),
                seed=primary_seed + int(split_cfg.get("seed_offset", 101)),
            )
            operator_run_views[op_name] = {
                "selection_target_datasets": split_payload["selection_target_datasets"],
                "evaluation_target_datasets": split_payload["evaluation_target_datasets"],
                "evaluation_datasets": split_payload["evaluation_datasets_all_operators"],
            }
            selection_eval_split_summary[op_name] = split_payload["split_records"]
            for rec in split_payload["split_records"].values():
                if not rec.get("split_applied"):
                    gate_summary["scope_warnings"].append(
                        f"Selection/evaluation split not applied for {op_name}:{rec.get('bucket')} due to insufficient examples."
                    )
                    if split_require_strict:
                        split_leakage_failures.append(
                            {
                                "operator": op_name,
                                "bucket": rec.get("bucket"),
                                "reason": rec.get("reason", "insufficient_examples_for_split"),
                                "total_examples": rec.get("total_examples"),
                                "selection_examples": rec.get("selection_examples"),
                                "evaluation_examples": rec.get("evaluation_examples"),
                            }
                        )
        else:
            operator_run_views[op_name] = {
                "selection_target_datasets": dict(operator_views[op_name]),
                "evaluation_target_datasets": dict(operator_views[op_name]),
                "evaluation_datasets": {name: suite.datasets[name] for name in sorted(suite.datasets.keys())},
            }
            selection_eval_split_summary[op_name] = {
                name: {
                    "operator": ds.operator,
                    "bucket": ds.bucket,
                    "total_examples": len(ds.examples),
                    "selection_examples": len(ds.examples),
                    "evaluation_examples": len(ds.examples),
                    "split_applied": False,
                    "reason": "split_disabled",
                }
                for name, ds in operator_views[op_name].items()
            }
    if split_require_strict and split_leakage_failures:
        if "selection_eval_split_not_applied" not in gate_summary["scope_blocks"]:
            gate_summary["scope_blocks"].append("selection_eval_split_not_applied")
        gate_summary["scope_warnings"].append(
            "Strict selection/evaluation split policy failed on one or more target-operator buckets."
        )
        gate_summary["phases"]["dataset_bucket_gate"]["passes"] = False
        gate_summary["phases"]["dataset_bucket_gate"]["failure_reason"] = "selection_eval_split_not_applied"
        gate_summary["phases"]["dataset_bucket_gate"]["split_leakage_failures"] = split_leakage_failures
    _json_dump(
        output_root / "phase2_selection_eval_split.json",
        {
            "schema_version": "phase2_selection_eval_split_v1",
            "config": split_cfg,
            "operators": selection_eval_split_summary,
            "strict_policy": {
                "require_strict": split_require_strict,
                "leakage_failures": split_leakage_failures,
            },
        },
    )
    run_manifest["selection_eval_split"] = {
        "config": split_cfg,
        "output": "phase2_selection_eval_split.json",
    }

    if split_require_strict and split_hard_stop_on_failure and split_leakage_failures:
        readiness_block_reasons = ["required_gate_failed:dataset_bucket_gate"]
        for scope_block in gate_summary.get("scope_blocks", []):
            readiness_block_reasons.append(f"scope_block:{scope_block}")
        gate_summary["overall"].update(
            {
                "ready_for_multimodel_next_tranche": False,
                "phase2_status": "blocked_pre_gpu_split_failure",
                "readiness_block_reasons": readiness_block_reasons,
            }
        )
        run_manifest["required_gates_policy"] = gate_summary.get("required_gates_policy", {})
        run_manifest["derived_thresholds"] = gate_summary.get("derived_thresholds", {})
        run_manifest["tuned_batch_sizes_by_stage"] = dict(tuned_batch_sizes_by_stage)
        run_manifest["autotune_probe_history"] = dict(autotune_probe_history)
        run_manifest["equivalence_check_results"] = dict(equivalence_check_results)
        _json_dump(output_root / "run_manifest.json", run_manifest)
        _json_dump(output_root / "phase2_gate_summary.json", gate_summary)
        _write_replication_protocol(output_root)
        print(f"Wrote Phase 2 outputs to {output_root}")
        print(f"Stage: {args.stage}")
        print("Strict selection/evaluation split failed; hard-stopped before GPU stages.")
        return

    if args.scaffold_gpu_stages:
        _scaffold_stage_outputs(output_root, cfg, args.stage)
        gate_summary["phases"]["localization_validity_gate"] = {
            "ran": args.stage in {"localize", "intervene", "cot_compare", "cross_operator_verify", "full"},
            "passes": False,
            "skipped": False,
            "status": "not_implemented",
            "output": "phase2_localization.json",
            "reason": "Scaffold mode enabled (--scaffold-gpu-stages).",
        }
        gate_summary["phases"]["operator_specificity_gate"] = {
            "ran": args.stage in {"intervene", "cross_operator_verify", "full"},
            "passes": False,
            "skipped": args.stage not in {"intervene", "cross_operator_verify", "full"},
            "status": "not_implemented",
            "output": "phase2_cross_operator_verify.json",
            "evidence_source": "post_merge_cross_operator_verify",
            "failure_reason": "cross_operator_verify_not_run",
        }
        gate_summary["phases"]["intervention_sanity_gate"] = {
            "ran": args.stage in {"intervene", "cross_operator_verify", "full"},
            "passes": False,
            "skipped": args.stage not in {"intervene", "cross_operator_verify", "full"},
            "status": "not_implemented",
            "output": "phase2_interventions.json",
        }
        gate_summary["phases"]["cot_gating_evidence_gate"] = {
            "ran": args.stage in {"cot_compare", "full"},
            "passes": False,
            "skipped": args.stage not in {"cot_compare", "full"},
            "status": "not_implemented",
            "output": "phase2_cot_recruitment_compare.json",
            "failure_reason": "cot_not_executed",
        }
        readiness_block_reasons = ["scaffold_gpu_stages_enabled"]
        for gate_name in gate_summary["required_gates_policy"]["required_for_readiness"]:
            if not bool(gate_summary["phases"].get(gate_name, {}).get("passes")):
                readiness_block_reasons.append(f"required_gate_failed:{gate_name}")
        for scope_block in gate_summary.get("scope_blocks", []):
            readiness_block_reasons.append(f"scope_block:{scope_block}")
        gate_summary["overall"].update(
            {
                "ready_for_multimodel_next_tranche": False,
                "phase2_status": "scaffold_outputs_written",
                "readiness_block_reasons": readiness_block_reasons,
            }
        )
        run_manifest["required_gates_policy"] = gate_summary.get("required_gates_policy", {})
        run_manifest["derived_thresholds"] = gate_summary.get("derived_thresholds", {})
        run_manifest["tuned_batch_sizes_by_stage"] = dict(tuned_batch_sizes_by_stage)
        run_manifest["autotune_probe_history"] = dict(autotune_probe_history)
        run_manifest["equivalence_check_results"] = dict(equivalence_check_results)
        _json_dump(output_root / "run_manifest.json", run_manifest)
        _json_dump(output_root / "phase2_gate_summary.json", gate_summary)
        _write_replication_protocol(output_root)
        print(f"Wrote Phase 2 outputs to {output_root}")
        print(f"Stage: {args.stage} (scaffold mode)")
        return

    # GPU-backed stages
    model, tokenizer = _load_model_if_needed(args, cfg)
    device_label = str(cfg["runtime"].get("devices") or os.environ.get("CUDA_VISIBLE_DEVICES", "auto"))
    base_batch_size = int(cfg["runtime"]["batch_size"])
    runtime_bt_cfg = dict(cfg["runtime"]["batch_autotune"])
    oom_min_batch_size = int(runtime_bt_cfg.get("min_batch_size", 1))
    oom_max_retries = int(runtime_bt_cfg.get("max_retries_after_oom", 3))
    oom_safety_backoff = float(runtime_bt_cfg.get("safety_backoff", 0.85))

    def _record_autotune(stage_name: str, tune_result: Mapping[str, Any]) -> int:
        tuned = int(tune_result.get("tuned_batch_size") or base_batch_size)
        tuned_batch_sizes_by_stage[stage_name] = tuned
        autotune_probe_history[stage_name] = dict(tune_result)
        equivalence_check_results[stage_name] = dict((tune_result.get("equivalence_check") or {}))
        return tuned

    def _record_runtime_backoff(stage_name: str, previous_batch: int, new_batch: int, *, context: str) -> None:
        if int(new_batch) >= int(previous_batch):
            return
        tuned_batch_sizes_by_stage[stage_name] = int(new_batch)
        history = autotune_probe_history.setdefault(stage_name, {})
        runtime_events = history.setdefault("runtime_oom_backoff_events", [])
        runtime_events.append(
            {
                "context": context,
                "previous_batch_size": int(previous_batch),
                "new_batch_size": int(new_batch),
            }
        )

    localize_batch_size = base_batch_size
    intervene_batch_size = base_batch_size
    cot_batch_size = base_batch_size

    localization_results: Dict[str, Dict[str, Any]] = {}
    localization_null_results: Dict[str, Dict[str, Any]] = {}
    localization_null_artifacts: Dict[str, str] = {}
    localization_calibration_coverage: Dict[str, Dict[str, Any]] = {}
    component_sets_index: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    cross_operator_verify_results: Dict[str, Dict[str, Any]] = {}
    cross_operator_verify_status = "not_implemented"
    cross_operator_verify_source: Optional[str] = None
    localization_nonzero_min = float(cfg["gates"]["localization_nonzero_min"])
    localization_prob_delta_abs_min = float(cfg["gates"]["localization_prob_delta_abs_min_floor"])
    calibration_cfg = cfg["gates"]["calibration"]
    gate_summary["derived_thresholds"]["localizer"] = {
        "policy": "configured_floor_only",
        "localization_nonzero_min": localization_nonzero_min,
        "localization_prob_delta_abs_min": localization_prob_delta_abs_min,
    }

    if args.stage in {"localize", "intervene", "cot_compare", "full"} and _autotune_stage_enabled(runtime_bt_cfg, "localize"):
        probe_operator = sorted(operator_run_views.keys())[0]
        probe_views = operator_run_views[probe_operator]
        probe_datasets = {
            name: _dataset_subset(dataset, max_examples=int(runtime_bt_cfg["equivalence_check"]["sample_size"]), split_label="selection")
            for name, dataset in sorted(probe_views["selection_target_datasets"].items())
        }
        probe_component_type = str(cfg["localization"]["component_types"][0])
        probe_buckets = sorted({ds.bucket for ds in probe_datasets.values()})
        probe_component_options = _localization_component_options(cfg, probe_component_type)
        # Keep probe options aligned with real localization options so memory behavior
        # is representative of the workload being tuned.
        probe_component_options = dict(probe_component_options)

        def _localize_probe(batch_size: int) -> Dict[str, Any]:
            probe_cfg = _build_localization_config(
                component_type=probe_component_type,
                operator=probe_operator,
                buckets=probe_buckets,
                cfg=cfg,
                seed=primary_seed,
                batch_size=int(batch_size),
            )
            probe_result = run_arithmetic_localization(
                model,
                tokenizer,
                model_name=model_name,
                datasets=probe_datasets,
                config=probe_cfg,
                component_options=probe_component_options,
                epsilon=float(cfg["localization"]["epsilon"]),
                max_examples_per_dataset=int(runtime_bt_cfg["equivalence_check"]["sample_size"]),
            )
            return _autotune_localization_signature(probe_result)

        localize_tune = autotune_batch_size(
            stage_name="localize",
            device=device_label,
            baseline_batch_size=base_batch_size,
            run_probe_fn=_localize_probe,
            config=BatchAutotuneConfig(
                enabled=bool(runtime_bt_cfg.get("enabled", True)),
                min_batch_size=int(runtime_bt_cfg.get("min_batch_size", 4)),
                max_batch_size=runtime_bt_cfg.get("max_batch_size"),
                growth_factor=float(runtime_bt_cfg.get("growth_factor", 1.5)),
                safety_backoff=float(runtime_bt_cfg.get("safety_backoff", 0.85)),
                max_retries_after_oom=int(runtime_bt_cfg.get("max_retries_after_oom", 3)),
                equivalence_check_enabled=bool(runtime_bt_cfg.get("equivalence_check", {}).get("enabled", True)),
                max_abs_logit_diff=float(runtime_bt_cfg.get("equivalence_check", {}).get("max_abs_logit_diff", 1e-4)),
                max_metric_diff=float(runtime_bt_cfg.get("equivalence_check", {}).get("max_metric_diff", 1e-4)),
            ),
        )
        localize_batch_size = _record_autotune("localize", localize_tune)

    if args.stage in {"localize", "intervene", "cot_compare", "full"}:
        loc_dir = output_root / "phase2_localization"
        loc_dir.mkdir(parents=True, exist_ok=True)
        for operator, views in sorted(operator_run_views.items()):
            ds_map = views["selection_target_datasets"]
            buckets = sorted({ds.bucket for ds in ds_map.values()})
            component_sets_index[operator] = {}
            for component_type in cfg["localization"]["component_types"]:
                loc_cfg = _build_localization_config(
                    component_type=component_type,
                    operator=operator,
                    buckets=buckets,
                    cfg=cfg,
                    seed=primary_seed,
                    batch_size=localize_batch_size,
                )
                component_options = _localization_component_options(cfg, component_type)
                max_examples_per_dataset = (
                    None
                    if cfg["localization"].get("max_examples_per_dataset") is None
                    else int(cfg["localization"].get("max_examples_per_dataset"))
                )
                base_result, used_localize_batch = _run_with_oom_backoff(
                    stage_name="localize",
                    initial_batch_size=localize_batch_size,
                    min_batch_size=oom_min_batch_size,
                    max_retries_after_oom=oom_max_retries,
                    safety_backoff=oom_safety_backoff,
                    run_fn=lambda bs: run_arithmetic_localization(
                        model,
                        tokenizer,
                        model_name=model_name,
                        datasets=ds_map,
                        config=LocalizationConfig(**{**asdict(loc_cfg), "batch_size": int(bs)}),
                        component_options=component_options,
                        epsilon=float(cfg["localization"]["epsilon"]),
                        max_examples_per_dataset=max_examples_per_dataset,
                    ),
                    scope_warnings=gate_summary["scope_warnings"],
                    context_label=f"{operator}:{component_type}:base",
                )
                if used_localize_batch != localize_batch_size:
                    _record_runtime_backoff(
                        "localize",
                        localize_batch_size,
                        used_localize_batch,
                        context=f"{operator}:{component_type}:base",
                    )
                    localize_batch_size = int(used_localize_batch)
                    loc_cfg = LocalizationConfig(**{**asdict(loc_cfg), "batch_size": int(localize_batch_size)})
                robustness = _compute_localization_robustness(
                    model,
                    tokenizer,
                    model_name=model_name,
                    datasets=ds_map,
                    base_config=loc_cfg,
                    component_options=component_options,
                    base_result=base_result,
                    stability_modes=cfg["localization"]["stability_modes"],
                    epsilon=float(cfg["localization"]["epsilon"]),
                    max_examples_per_dataset=max_examples_per_dataset,
                    subsample_fraction=float(cfg["localization"]["subsample_fraction"]),
                    seeds=seeds,
                    oom_min_batch_size=oom_min_batch_size,
                    oom_max_retries=oom_max_retries,
                    oom_safety_backoff=oom_safety_backoff,
                    scope_warnings=gate_summary["scope_warnings"],
                    context_label=f"{operator}:{component_type}:robustness",
                )
                result = annotate_localization_rank_stability(base_result, **robustness)
                key = f"{operator}::{component_type}"
                localization_results[key] = result
                _json_dump(loc_dir / f"{operator}_{component_type}.json", result)
                null_policy_terms = set(_parse_policy_terms(str(calibration_cfg.get("localizer_null_policy", ""))))
                if calibration_cfg.get("enabled") and "target_shuffle" in null_policy_terms:
                    null_result, used_localize_batch = _run_with_oom_backoff(
                        stage_name="localize",
                        initial_batch_size=localize_batch_size,
                        min_batch_size=oom_min_batch_size,
                        max_retries_after_oom=oom_max_retries,
                        safety_backoff=oom_safety_backoff,
                        run_fn=lambda bs: run_arithmetic_localization(
                            model,
                            tokenizer,
                            model_name=model_name,
                            datasets=ds_map,
                            config=LocalizationConfig(
                                **{**asdict(loc_cfg), "seed": primary_seed + 911, "batch_size": int(bs)}
                            ),
                            component_options=component_options,
                            epsilon=float(cfg["localization"]["epsilon"]),
                            max_examples_per_dataset=max_examples_per_dataset,
                            shuffle_target_ids=True,
                        ),
                        scope_warnings=gate_summary["scope_warnings"],
                        context_label=f"{operator}:{component_type}:target_shuffle",
                    )
                    if used_localize_batch != localize_batch_size:
                        _record_runtime_backoff(
                            "localize",
                            localize_batch_size,
                            used_localize_batch,
                            context=f"{operator}:{component_type}:target_shuffle",
                        )
                        localize_batch_size = int(used_localize_batch)
                    null_key = f"{key}::target_shuffle"
                    localization_null_results[null_key] = null_result
                    target_shuffle_path = f"phase2_localization/{operator}_{component_type}_null_target_shuffle.json"
                    localization_null_artifacts[null_key] = target_shuffle_path
                    _json_dump(output_root / target_shuffle_path, null_result)
                if calibration_cfg.get("enabled") and "family_heldout" in null_policy_terms:
                    heldout_buckets = sorted({ds.bucket for ds in ds_map.values()})
                    max_heldout = max(1, int(calibration_cfg.get("max_heldout_buckets", 1)))
                    heldout_coverage_mode = str(calibration_cfg.get("family_heldout_coverage", "exhaustive"))
                    if len(heldout_buckets) < 2:
                        gate_summary["scope_warnings"].append(
                            f"Family-heldout null calibration requested but operator '{operator}' has fewer than 2 buckets in selection split."
                        )
                    heldout_seed = primary_seed + 1201 + sum(ord(ch) for ch in f"{operator}:{component_type}")
                    heldout_rng = random.Random(heldout_seed)
                    if heldout_coverage_mode == "exhaustive":
                        selected_heldout = list(heldout_buckets)
                    else:
                        heldout_candidates = list(heldout_buckets)
                        heldout_rng.shuffle(heldout_candidates)
                        selected_heldout = heldout_candidates[:max_heldout]
                    coverage_key = f"{operator}::{component_type}"
                    localization_calibration_coverage[coverage_key] = {
                        "operator": operator,
                        "component_type": component_type,
                        "coverage_mode": heldout_coverage_mode,
                        "total_buckets": len(heldout_buckets),
                        "heldout_buckets_run": list(selected_heldout),
                        "coverage_ratio": (
                            float(len(selected_heldout)) / float(len(heldout_buckets))
                            if heldout_buckets
                            else 0.0
                        ),
                        "max_heldout_buckets": max_heldout,
                    }
                    for heldout_bucket in selected_heldout:
                        heldout_null, used_localize_batch = _run_with_oom_backoff(
                            stage_name="localize",
                            initial_batch_size=localize_batch_size,
                            min_batch_size=oom_min_batch_size,
                            max_retries_after_oom=oom_max_retries,
                            safety_backoff=oom_safety_backoff,
                            run_fn=lambda bs, hb=heldout_bucket: run_arithmetic_localization(
                                model,
                                tokenizer,
                                model_name=model_name,
                                datasets=ds_map,
                                config=LocalizationConfig(
                                    **{**asdict(loc_cfg), "seed": primary_seed + 1201, "batch_size": int(bs)}
                                ),
                                component_options=component_options,
                                epsilon=float(cfg["localization"]["epsilon"]),
                                max_examples_per_dataset=max_examples_per_dataset,
                                heldout_buckets=[hb],
                                shuffle_target_ids=True,
                            ),
                            scope_warnings=gate_summary["scope_warnings"],
                            context_label=f"{operator}:{component_type}:family_heldout:{heldout_bucket}",
                        )
                        if used_localize_batch != localize_batch_size:
                            _record_runtime_backoff(
                                "localize",
                                localize_batch_size,
                                used_localize_batch,
                                context=f"{operator}:{component_type}:family_heldout:{heldout_bucket}",
                            )
                            localize_batch_size = int(used_localize_batch)
                        heldout_key = f"{key}::family_heldout::{heldout_bucket}"
                        localization_null_results[heldout_key] = heldout_null
                        heldout_path = (
                            f"phase2_localization/{operator}_{component_type}_null_family_heldout_{heldout_bucket}.json"
                        )
                        localization_null_artifacts[heldout_key] = heldout_path
                        _json_dump(output_root / heldout_path, heldout_null)
                component_sets_index[operator][component_type] = component_sets_from_localization(
                    result,
                    k_values=cfg["localization"]["k_values"],
                    score_key=cfg["localization"]["score_key"],
                    seed=primary_seed,
                )

        if localization_null_results:
            localizer_thresholds = _derive_localization_thresholds(
                localization_null_results,
                policy=str(calibration_cfg.get("localizer_null_policy", "target_shuffle")),
                quantile=float(calibration_cfg.get("quantile", 0.95)),
                nonzero_floor=float(cfg["gates"]["localization_nonzero_min"]),
                abs_prob_floor=float(cfg["gates"]["localization_prob_delta_abs_min_floor"]),
            )
            localization_nonzero_min = float(localizer_thresholds["localization_nonzero_min"])
            localization_prob_delta_abs_min = float(localizer_thresholds["localization_prob_delta_abs_min"])
            gate_summary["derived_thresholds"]["localizer"] = localizer_thresholds

        localization_summary = {
            "schema_version": "phase2_localization_summary_v1",
            "runs": localization_results,
            "component_sets": component_sets_index,
            "selection_eval_split": {
                "enabled": bool(split_cfg.get("enabled", True)),
                "summary_output": "phase2_selection_eval_split.json",
            },
            "calibration": {
                "enabled": bool(calibration_cfg.get("enabled")),
                "localizer_null_policy": calibration_cfg.get("localizer_null_policy"),
                "null_runs": dict(localization_null_artifacts),
                "derived_thresholds": gate_summary["derived_thresholds"].get("localizer"),
                "family_heldout_coverage_policy": str(calibration_cfg.get("family_heldout_coverage", "exhaustive")),
                "coverage_by_run": localization_calibration_coverage,
            },
        }
        _json_dump(output_root / "phase2_localization.json", localization_summary)
        loc_gate = _evaluate_localization_gate(
            localization_results,
            nonzero_min=localization_nonzero_min,
            prob_delta_abs_min=localization_prob_delta_abs_min,
            required_robustness_modes=list(cfg["localization"]["stability_modes"]),
            require_all_runs=bool(cfg["gates"]["require_all_component_types"]),
            score_aggregation=str(cfg["localization"]["score_aggregation"]),
            score_quantile=float(cfg["localization"]["score_quantile"]),
            min_valid_target_count=int(cfg["localization"]["min_valid_target_count"]),
            min_components_passing=int(cfg["localization"]["min_components_passing"]),
            min_answer_target_valid_rate=float(cfg["localization"]["min_answer_target_valid_rate"]),
        )
        loc_gate.update({"ran": True, "skipped": False, "output": "phase2_localization.json"})
        gate_summary["phases"]["localization_validity_gate"] = loc_gate

    if args.stage == "cross_operator_verify":
        source_path = _resolve_cross_operator_source(args.cross_operator_source, output_root)
        if not source_path.exists():
            raise FileNotFoundError(
                "Cross-operator verify requires localization component sets. "
                f"Missing source: {source_path}"
            )
        component_sets_index = _load_component_sets_index_from_localization(source_path)
        cross_operator_verify_source = str(source_path)
        gate_summary["scope_warnings"].append(
            f"Loaded component sets for cross-operator verification from {source_path}."
        )

    intervention_results: Dict[str, Dict[str, Any]] = {}
    intervention_anomaly_report: Optional[Dict[str, Any]] = None
    parser_audit_payload: Optional[Dict[str, Any]] = None
    specificity_ci_low_min = float(cfg["gates"]["specificity_ci_low_min"])
    gate_summary["derived_thresholds"]["specificity"] = {
        "policy": "configured_floor_only",
        "specificity_ci_low_min": specificity_ci_low_min,
    }
    if args.stage in {"intervene", "cross_operator_verify", "full"} and _autotune_stage_enabled(runtime_bt_cfg, "intervene"):
        probe_operator = sorted(operator_run_views.keys())[0]
        probe_eval_views = operator_run_views[probe_operator]["evaluation_target_datasets"]
        probe_dataset = _dataset_subset(
            sorted(probe_eval_views.values(), key=lambda ds: ds.name)[0],
            max_examples=int(runtime_bt_cfg["equivalence_check"]["sample_size"]),
            split_label="evaluation",
        )

        def _intervene_probe(batch_size: int) -> Dict[str, Any]:
            probe_eval = evaluate_operator_bucket_dataset(
                model,
                tokenizer,
                probe_dataset,
                parse_mode=str(cfg["cot_compare"]["evaluation_parse_mode"]),
                max_new_tokens=int(cfg["cot_compare"]["max_new_tokens"]),
                deterministic_generation=bool(cfg["runtime"]["deterministic_generation"]),
                allow_sampling_fallback=bool(cfg["runtime"]["allow_sampling_fallback"]),
                batch_size=int(batch_size),
            )
            return _autotune_eval_signature(probe_eval["evaluation"] | {"results": probe_eval.get("results", [])})

        intervene_tune = autotune_batch_size(
            stage_name="intervene",
            device=device_label,
            baseline_batch_size=base_batch_size,
            run_probe_fn=_intervene_probe,
            config=BatchAutotuneConfig(
                enabled=bool(runtime_bt_cfg.get("enabled", True)),
                min_batch_size=int(runtime_bt_cfg.get("min_batch_size", 4)),
                max_batch_size=runtime_bt_cfg.get("max_batch_size"),
                growth_factor=float(runtime_bt_cfg.get("growth_factor", 1.5)),
                safety_backoff=float(runtime_bt_cfg.get("safety_backoff", 0.85)),
                max_retries_after_oom=int(runtime_bt_cfg.get("max_retries_after_oom", 3)),
                equivalence_check_enabled=bool(runtime_bt_cfg.get("equivalence_check", {}).get("enabled", True)),
                max_abs_logit_diff=float(runtime_bt_cfg.get("equivalence_check", {}).get("max_abs_logit_diff", 1e-4)),
                max_metric_diff=float(runtime_bt_cfg.get("equivalence_check", {}).get("max_metric_diff", 1e-4)),
            ),
        )
        intervene_batch_size = _record_autotune("intervene", intervene_tune)
    if args.stage in {"intervene", "full"}:
        int_dir = output_root / "phase2_interventions"
        int_dir.mkdir(parents=True, exist_ok=True)
        induction_baseline_sets = _maybe_load_induction_baseline_sets() if "induction_baseline" in cfg["interventions"]["controls"] else None
        induction_baseline_sets = _sanitize_induction_baseline_sets_for_model(
            model,
            induction_baseline_sets,
            scope_warnings=gate_summary["scope_warnings"],
        )
        for operator, views in sorted(operator_run_views.items()):
            eval_datasets = views["evaluation_datasets"]
            for component_type, component_sets in component_sets_index.get(operator, {}).items():
                key = f"{operator}::{component_type}"
                payload, used_intervene_batch = _run_with_oom_backoff(
                    stage_name="intervene",
                    initial_batch_size=intervene_batch_size,
                    min_batch_size=oom_min_batch_size,
                    max_retries_after_oom=oom_max_retries,
                    safety_backoff=oom_safety_backoff,
                    run_fn=lambda bs: run_operator_intervention_sweeps(
                        model,
                        tokenizer,
                        model_name=model_name,
                        datasets=eval_datasets,
                        component_sets=component_sets,
                        operator_target=operator,
                        scales=cfg["interventions"]["scales"],
                        interventions=cfg["interventions"]["tests"],
                        strict_attention_heads=bool(cfg["localization"]["strict_attention_heads"]),
                        bootstrap_samples=int(cfg["interventions"]["bootstrap_samples"]),
                        seed=primary_seed,
                        induction_baseline_sets=induction_baseline_sets if component_type == "attention_heads" else None,
                        sanity_policy=cfg["gates"]["anomaly_blocking"],
                        primary_component_set=str(cfg["interventions"]["primary_component_set"]),
                        primary_interventions=list(cfg["interventions"]["primary_interventions"]),
                        primary_scales=list(cfg["interventions"]["primary_scales"]),
                        primary_k_values=list(cfg["interventions"]["primary_k_values"]),
                        multiplicity_reporting=str(cfg["analysis"]["multiplicity_reporting"]),
                        deterministic_generation=bool(cfg["runtime"]["deterministic_generation"]),
                        allow_sampling_fallback=bool(cfg["runtime"]["allow_sampling_fallback"]),
                        evaluation_parse_mode=str(cfg["cot_compare"]["evaluation_parse_mode"]),
                        batch_size=int(bs),
                    ),
                    scope_warnings=gate_summary["scope_warnings"],
                    context_label=f"{operator}:{component_type}",
                )
                if used_intervene_batch != intervene_batch_size:
                    _record_runtime_backoff(
                        "intervene",
                        intervene_batch_size,
                        used_intervene_batch,
                        context=f"{operator}:{component_type}",
                    )
                    intervene_batch_size = int(used_intervene_batch)
                intervention_results[key] = payload
                _json_dump(int_dir / f"{operator}_{component_type}.json", payload)
                matrix = build_specificity_matrix_from_intervention_results(payload)
                _json_dump(int_dir / f"{operator}_{component_type}_specificity_matrix.json", matrix)

        _json_dump(
            output_root / "phase2_interventions.json",
            {
                "schema_version": "phase2_interventions_summary_v1",
                "runs": intervention_results,
            },
        )
        parser_cfg = cfg["analysis"]["parser_audit"]
        if bool(parser_cfg.get("enabled", True)):
            parser_samples = collect_parser_audit_samples_from_intervention_runs(
                intervention_results,
                per_dataset_limit=int(parser_cfg.get("per_dataset_limit", 16)),
            )
            parser_audit_payload = build_parser_audit_report(
                parser_samples,
                source_label="phase2_interventions.json",
                adjudication_cap=int(parser_cfg.get("adjudication_cap", 64)),
            )
            _json_dump(output_root / "parser_audit.json", parser_audit_payload)
            run_manifest["analysis_artifacts"]["parser_audit"] = "parser_audit.json"
        # Build a global merged matrix for convenience.
        rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
        cols: set[str] = set()
        for key, payload in intervention_results.items():
            matrix = build_specificity_matrix_from_intervention_results(payload)
            for col in matrix.get("cols", []):
                cols.add(col)
            for row_entry in matrix.get("matrix", []):
                row_key = f"{key}::{row_entry['row']}"
                rows[row_key] = row_entry.get("cells", {})
        from src.operator_interventions import build_cross_operator_specificity_matrix

        merged_matrix = build_cross_operator_specificity_matrix(rows=list(rows.keys()), cols=sorted(cols), values=rows)
        _json_dump(output_root / "phase2_cross_operator_specificity_matrix.json", merged_matrix)

        if calibration_cfg.get("enabled"):
            specificity_thresholds = _derive_specificity_threshold_from_random_controls(
                intervention_results,
                quantile=float(calibration_cfg.get("quantile", 0.95)),
                ci_low_floor=float(cfg["gates"]["specificity_ci_low_min"]),
            )
            specificity_ci_low_min = float(specificity_thresholds["specificity_ci_low_min"])
            gate_summary["derived_thresholds"]["specificity"] = specificity_thresholds

        int_gate = _evaluate_specificity_gate(
            intervention_results,
            ci_low_min=specificity_ci_low_min,
            mean_gap_min=float(cfg["gates"]["specificity_mean_gap_min"]),
            require_non_target_operator_evidence=bool(cfg["gates"]["require_non_target_operator_evidence"]),
            require_primary_set=bool(cfg["gates"]["specificity_requires_primary_set"]),
            primary_set_name=str(cfg["gates"]["specificity_primary_set"]),
            require_directionality=bool(cfg["gates"]["specificity_requires_directionality"]),
            require_both_primary_interventions=bool(cfg["gates"]["specificity_requires_both_primary_interventions"]),
            primary_interventions=list(cfg["interventions"]["primary_interventions"]),
            primary_scales=list(cfg["interventions"]["primary_scales"]),
            primary_k_values=list(cfg["interventions"]["primary_k_values"]),
            condition_policy=str(cfg["gates"]["specificity_condition_policy"]),
            sign_policy=str(cfg["gates"]["specificity_sign_policy"]),
            ablation_delta_vs_random_max=float(cfg["gates"]["anomaly_blocking"]["top_ablation_delta_vs_random_max"]),
            amplification_delta_vs_random_min=float(cfg["gates"]["anomaly_blocking"]["top_amplification_delta_vs_random_min"]),
            multiplicity_blocking_enabled=bool(cfg["gates"]["multiplicity_blocking"]["enabled"]),
            multiplicity_q_max=float(cfg["gates"]["multiplicity_blocking"]["q_max"]),
            multiplicity_require_complete_primary_coverage=bool(
                cfg["gates"]["multiplicity_require_complete_primary_coverage"]
            ),
        )
        int_gate.update(
            {
                "ran": True,
                "skipped": False,
                "output": "phase2_interventions.json",
                "matrix_output": "phase2_cross_operator_specificity_matrix.json",
            }
        )
        gate_summary["phases"]["operator_specificity_gate"] = int_gate
        if any(check.get("failure_reason") == "insufficient_non_target_scope" for check in int_gate.get("checks", [])):
            if "single_operator_scope_blocks_specificity" not in gate_summary["scope_blocks"]:
                gate_summary["scope_blocks"].append("single_operator_scope_blocks_specificity")
        sanity_gate = _evaluate_intervention_sanity_gate(
            intervention_results,
            enabled=bool(cfg["gates"]["anomaly_blocking"]["enabled"]),
        )
        sanity_gate.update(
            {
                "ran": True,
                "skipped": False,
                "output": "phase2_interventions.json",
            }
        )
        intervention_anomaly_report = _build_intervention_anomaly_report(
            intervention_results,
            sample_cap_per_dataset=int(cfg["gates"]["anomaly_blocking"]["prediction_sample_size"]),
        )
        _json_dump(output_root / "phase2_intervention_anomaly_report.json", intervention_anomaly_report)
        sanity_gate["anomaly_report_output"] = "phase2_intervention_anomaly_report.json"
        gate_summary["phases"]["intervention_sanity_gate"] = sanity_gate
        if not bool(sanity_gate.get("passes")):
            gate_summary["scope_warnings"].append(
                "Intervention sanity gate flagged anomalies; inspect phase2_intervention_anomaly_report.json before interpreting specificity."
            )

    if args.stage in {"cross_operator_verify", "full"}:
        verify_runs: Dict[str, Dict[str, Any]] = {}
        if args.stage == "full" and intervention_results:
            verify_runs = dict(intervention_results)
            cross_operator_verify_status = "ok"
            cross_operator_verify_source = "phase2_interventions.json"
        else:
            if not component_sets_index:
                source_path = _resolve_cross_operator_source(args.cross_operator_source, output_root)
                if not source_path.exists():
                    raise FileNotFoundError(
                        "Cross-operator verify requires localization component sets. "
                        f"Missing source: {source_path}"
                    )
                component_sets_index = _load_component_sets_index_from_localization(source_path)
                cross_operator_verify_source = str(source_path)
            for operator, views in sorted(operator_run_views.items()):
                eval_datasets = views["evaluation_datasets"]
                for component_type, component_sets in component_sets_index.get(operator, {}).items():
                    filtered_sets = _filter_component_sets_for_primary_k(
                        component_sets,
                        primary_k_values=list(cfg["interventions"]["primary_k_values"]),
                    )
                    if not filtered_sets:
                        continue
                    key = f"{operator}::{component_type}"
                    payload, used_intervene_batch = _run_with_oom_backoff(
                        stage_name="cross_operator_verify",
                        initial_batch_size=intervene_batch_size,
                        min_batch_size=oom_min_batch_size,
                        max_retries_after_oom=oom_max_retries,
                        safety_backoff=oom_safety_backoff,
                        run_fn=lambda bs: run_operator_intervention_sweeps(
                            model,
                            tokenizer,
                            model_name=model_name,
                            datasets=eval_datasets,
                            component_sets=filtered_sets,
                            operator_target=operator,
                            scales=cfg["interventions"]["primary_scales"],
                            interventions=cfg["interventions"]["primary_interventions"],
                            strict_attention_heads=bool(cfg["localization"]["strict_attention_heads"]),
                            bootstrap_samples=int(cfg["interventions"]["bootstrap_samples"]),
                            seed=primary_seed,
                            induction_baseline_sets=None,
                            sanity_policy=cfg["gates"]["anomaly_blocking"],
                            primary_component_set=str(cfg["interventions"]["primary_component_set"]),
                            primary_interventions=list(cfg["interventions"]["primary_interventions"]),
                            primary_scales=list(cfg["interventions"]["primary_scales"]),
                            primary_k_values=list(cfg["interventions"]["primary_k_values"]),
                            multiplicity_reporting=str(cfg["analysis"]["multiplicity_reporting"]),
                            deterministic_generation=bool(cfg["runtime"]["deterministic_generation"]),
                            allow_sampling_fallback=bool(cfg["runtime"]["allow_sampling_fallback"]),
                            evaluation_parse_mode=str(cfg["cot_compare"]["evaluation_parse_mode"]),
                            max_new_tokens=int(cfg["cot_compare"]["max_new_tokens"]),
                            batch_size=int(bs),
                        ),
                        scope_warnings=gate_summary["scope_warnings"],
                        context_label=f"{operator}:{component_type}:cross_verify",
                    )
                    if used_intervene_batch != intervene_batch_size:
                        _record_runtime_backoff(
                            "cross_operator_verify",
                            intervene_batch_size,
                            used_intervene_batch,
                            context=f"{operator}:{component_type}",
                        )
                        intervene_batch_size = int(used_intervene_batch)
                    verify_runs[key] = payload
            cross_operator_verify_status = "ok" if verify_runs else "not_implemented"
            if cross_operator_verify_source is None:
                cross_operator_verify_source = "phase2_localization.json"

        cross_operator_verify_results = dict(verify_runs)
        gate_summary["run_metadata"]["cross_operator_verify_source"] = cross_operator_verify_source
        _json_dump(
            output_root / "phase2_cross_operator_verify.json",
            {
                "schema_version": "phase2_cross_operator_verify_v1",
                "status": cross_operator_verify_status,
                "evidence_source": "post_merge_cross_operator_verify",
                "source_component_sets": cross_operator_verify_source,
                "runs": cross_operator_verify_results,
            },
        )
        run_manifest["analysis_artifacts"]["cross_operator_verify"] = "phase2_cross_operator_verify.json"
        rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
        cols: set[str] = set()
        for key, payload in cross_operator_verify_results.items():
            matrix = build_specificity_matrix_from_intervention_results(payload)
            for col in matrix.get("cols", []):
                cols.add(col)
            for row_entry in matrix.get("matrix", []):
                rows[f"{key}::{row_entry['row']}"] = row_entry.get("cells", {})
        from src.operator_interventions import build_cross_operator_specificity_matrix

        merged_matrix = build_cross_operator_specificity_matrix(rows=list(rows.keys()), cols=sorted(cols), values=rows)
        _json_dump(output_root / "phase2_cross_operator_specificity_matrix.json", merged_matrix)
        run_manifest["analysis_artifacts"]["cross_operator_specificity_matrix"] = "phase2_cross_operator_specificity_matrix.json"

        if not cross_operator_verify_results:
            int_gate = {
                "passes": False,
                "checks": [],
                "failure_reason": "missing_non_target_evidence",
                "evidence_source": "post_merge_cross_operator_verify",
                "ran": True,
                "skipped": False,
                "output": "phase2_cross_operator_verify.json",
                "matrix_output": "phase2_cross_operator_specificity_matrix.json",
            }
        else:
            int_gate = _evaluate_specificity_gate(
                cross_operator_verify_results,
                ci_low_min=specificity_ci_low_min,
                mean_gap_min=float(cfg["gates"]["specificity_mean_gap_min"]),
                require_non_target_operator_evidence=bool(cfg["gates"]["require_non_target_operator_evidence"]),
                require_primary_set=bool(cfg["gates"]["specificity_requires_primary_set"]),
                primary_set_name=str(cfg["gates"]["specificity_primary_set"]),
                require_directionality=bool(cfg["gates"]["specificity_requires_directionality"]),
                require_both_primary_interventions=bool(cfg["gates"]["specificity_requires_both_primary_interventions"]),
                primary_interventions=list(cfg["interventions"]["primary_interventions"]),
                primary_scales=list(cfg["interventions"]["primary_scales"]),
                primary_k_values=list(cfg["interventions"]["primary_k_values"]),
                condition_policy=str(cfg["gates"]["specificity_condition_policy"]),
                sign_policy=str(cfg["gates"]["specificity_sign_policy"]),
                ablation_delta_vs_random_max=float(cfg["gates"]["anomaly_blocking"]["top_ablation_delta_vs_random_max"]),
                amplification_delta_vs_random_min=float(cfg["gates"]["anomaly_blocking"]["top_amplification_delta_vs_random_min"]),
                multiplicity_blocking_enabled=bool(cfg["gates"]["multiplicity_blocking"]["enabled"]),
                multiplicity_q_max=float(cfg["gates"]["multiplicity_blocking"]["q_max"]),
                multiplicity_require_complete_primary_coverage=bool(
                    cfg["gates"]["multiplicity_require_complete_primary_coverage"]
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
        if any(check.get("failure_reason") == "insufficient_non_target_scope" for check in int_gate.get("checks", [])):
            if "single_operator_scope_blocks_specificity" not in gate_summary["scope_blocks"]:
                gate_summary["scope_blocks"].append("single_operator_scope_blocks_specificity")

    cot_results: Dict[str, Dict[str, Any]] = {}
    if args.stage in {"cot_compare", "full"} and _autotune_stage_enabled(runtime_bt_cfg, "cot"):
        probe_operator = sorted(operator_run_views.keys())[0]
        probe_eval_views = operator_run_views[probe_operator]["evaluation_target_datasets"]
        probe_dataset = _dataset_subset(
            sorted(probe_eval_views.values(), key=lambda ds: ds.name)[0],
            max_examples=int(runtime_bt_cfg["equivalence_check"]["sample_size"]),
            split_label="evaluation",
        )

        def _cot_probe(batch_size: int) -> Dict[str, Any]:
            probe_eval = evaluate_operator_bucket_dataset(
                model,
                tokenizer,
                probe_dataset,
                parse_mode=str(cfg["cot_compare"]["evaluation_parse_mode"]),
                max_new_tokens=int(cfg["cot_compare"]["max_new_tokens"]),
                deterministic_generation=bool(cfg["runtime"]["deterministic_generation"]),
                allow_sampling_fallback=bool(cfg["runtime"]["allow_sampling_fallback"]),
                batch_size=int(batch_size),
            )
            return _autotune_eval_signature(probe_eval["evaluation"] | {"results": probe_eval.get("results", [])})

        cot_tune = autotune_batch_size(
            stage_name="cot",
            device=device_label,
            baseline_batch_size=base_batch_size,
            run_probe_fn=_cot_probe,
            config=BatchAutotuneConfig(
                enabled=bool(runtime_bt_cfg.get("enabled", True)),
                min_batch_size=int(runtime_bt_cfg.get("min_batch_size", 4)),
                max_batch_size=runtime_bt_cfg.get("max_batch_size"),
                growth_factor=float(runtime_bt_cfg.get("growth_factor", 1.5)),
                safety_backoff=float(runtime_bt_cfg.get("safety_backoff", 0.85)),
                max_retries_after_oom=int(runtime_bt_cfg.get("max_retries_after_oom", 3)),
                equivalence_check_enabled=bool(runtime_bt_cfg.get("equivalence_check", {}).get("enabled", True)),
                max_abs_logit_diff=float(runtime_bt_cfg.get("equivalence_check", {}).get("max_abs_logit_diff", 1e-4)),
                max_metric_diff=float(runtime_bt_cfg.get("equivalence_check", {}).get("max_metric_diff", 1e-4)),
            ),
        )
        cot_batch_size = _record_autotune("cot", cot_tune)
    if args.stage in {"cot_compare", "full"}:
        cot_dir = output_root / "phase2_cot_compare"
        cot_dir.mkdir(parents=True, exist_ok=True)
        cot_cfg = CoTRecruitmentComparisonConfig(
            enabled=bool(cfg["cot_compare"]["enabled"]),
            max_pairs=int(cfg["cot_compare"]["paired_count"]),
            direct_instruction=str(cfg["cot_compare"]["direct_instruction"]),
            cot_instruction=str(cfg["cot_compare"]["cot_instruction"]),
            format_lock_enabled=bool(cfg["cot_compare"]["format_lock"]["enabled"]),
            answer_marker=str(cfg["cot_compare"]["format_lock"]["answer_marker"]),
            relaxed_parser_diagnostics_enabled=bool(
                cfg["cot_compare"]["relaxed_parser_diagnostics"]["enabled"]
            ),
            evaluation_parse_mode=str(cfg["cot_compare"]["evaluation_parse_mode"]),
            max_new_tokens=int(cfg["cot_compare"]["max_new_tokens"]),
            min_pairs=int(cfg["cot_compare"]["min_pairs"]),
            parse_rate_min=float(cfg["cot_compare"]["parse_rate_min"]),
            require_accuracy_ci_excludes_zero=bool(cfg["cot_compare"]["require_accuracy_ci_excludes_zero"]),
            stratify_by_dataset=bool(cfg["cot_compare"]["stratify_by_dataset"]),
            sampling_seed=int(cfg["cot_compare"]["sampling_seed"]),
            dataset_pair_allocation=str(cfg["cot_compare"]["dataset_pair_allocation"]),
            evaluation_batch_size=int(cot_batch_size),
        )
        sensitivity_set_name = str(cfg["cot_compare"]["sensitivity_component_set"])
        sensitivity_k = int(cfg["cot_compare"]["sensitivity_k"])
        sensitivity_scale = float(cfg["cot_compare"]["sensitivity_scale"])
        for operator, views in sorted(operator_run_views.items()):
            ds_map = views["evaluation_target_datasets"]
            sensitivity_ids = None
            for component_type in ("attention_heads", "mlp_neurons", "layer_blocks"):
                comp_sets = component_sets_index.get(operator, {}).get(component_type)
                if not comp_sets:
                    continue
                key = f"K{sensitivity_k}"
                if key in comp_sets and sensitivity_set_name in comp_sets[key]:
                    sensitivity_ids = comp_sets[key][sensitivity_set_name]
                    break
            payload, used_cot_batch = _run_with_oom_backoff(
                stage_name="cot",
                initial_batch_size=cot_batch_size,
                min_batch_size=oom_min_batch_size,
                max_retries_after_oom=oom_max_retries,
                safety_backoff=oom_safety_backoff,
                run_fn=lambda bs: run_cot_recruitment_compare(
                    model,
                    tokenizer,
                    model_name=model_name,
                    datasets=ds_map,
                    config=CoTRecruitmentComparisonConfig(**{**asdict(cot_cfg), "evaluation_batch_size": int(bs)}),
                    sensitivity_component_ids=sensitivity_ids,
                    sensitivity_scale=sensitivity_scale,
                    strict_attention_heads=bool(cfg["localization"]["strict_attention_heads"]),
                    deterministic_generation=bool(cfg["runtime"]["deterministic_generation"]),
                    allow_sampling_fallback=bool(cfg["runtime"]["allow_sampling_fallback"]),
                    evaluation_batch_size=int(bs),
                ),
                scope_warnings=gate_summary["scope_warnings"],
                context_label=f"{operator}:cot_compare",
            )
            if used_cot_batch != cot_batch_size:
                _record_runtime_backoff(
                    "cot",
                    cot_batch_size,
                    used_cot_batch,
                    context=f"{operator}:cot_compare",
                )
                cot_batch_size = int(used_cot_batch)
                cot_cfg = CoTRecruitmentComparisonConfig(**{**asdict(cot_cfg), "evaluation_batch_size": int(cot_batch_size)})
            cot_results[operator] = payload
            _json_dump(cot_dir / f"{operator}.json", payload)
        _json_dump(output_root / "phase2_cot_recruitment_compare.json", {"schema_version": "phase2_cot_compare_summary_v1", "runs": cot_results})
        cot_gate = _evaluate_cot_gate(
            cot_results,
            effect_abs_min=float(cfg["cot_compare"]["effect_abs_min"]),
            parse_rate_delta_abs_max=float(cfg["cot_compare"]["parse_rate_delta_abs_max"]),
            min_pairs=int(cfg["cot_compare"]["min_pairs"]),
            parse_rate_min=float(cfg["cot_compare"]["parse_rate_min"]),
            require_accuracy_ci_excludes_zero=bool(cfg["cot_compare"]["require_accuracy_ci_excludes_zero"]),
        )
        cot_gate.update({"ran": True, "skipped": False, "output": "phase2_cot_recruitment_compare.json"})
        gate_summary["phases"]["cot_gating_evidence_gate"] = cot_gate
        if any(check.get("status") == "not_implemented" for check in cot_gate.get("checks", [])):
            gate_summary["scope_warnings"].append("CoT stage is disabled or not implemented for this run; CoT gate remains unresolved.")

    if parser_audit_payload is not None:
        gate_summary["run_metadata"]["parser_audit"] = {
            "output": "parser_audit.json",
            "sample_count": int(parser_audit_payload.get("sample_count", 0)),
            "ambiguity_rate": float(parser_audit_payload.get("ambiguity_rate", 0.0)),
        }
    gate_summary["run_metadata"]["tuned_batch_sizes_by_stage"] = dict(tuned_batch_sizes_by_stage)
    gate_summary["run_metadata"]["autotune_probe_history"] = dict(autotune_probe_history)
    gate_summary["run_metadata"]["equivalence_check_results"] = dict(equivalence_check_results)

    required_gate_names = list(gate_summary["required_gates_policy"]["required_for_readiness"])
    if bool(cfg["gates"]["cot_required_for_readiness"]):
        cot_phase = gate_summary["phases"]["cot_gating_evidence_gate"]
        if not cot_phase.get("ran") or cot_phase.get("status") == "not_implemented":
            gate_summary["scope_warnings"].append("CoT-required readiness policy is enabled; unresolved CoT gate blocks readiness.")
    readiness_block_reasons: List[str] = []
    for key in required_gate_names:
        if not bool(gate_summary["phases"][key].get("passes")):
            readiness_block_reasons.append(f"required_gate_failed:{key}")
    for scope_block in gate_summary.get("scope_blocks", []):
        readiness_block_reasons.append(f"scope_block:{scope_block}")
    ready = len(readiness_block_reasons) == 0
    gate_summary["overall"].update(
        {
            "ready_for_multimodel_next_tranche": ready,
            "phase2_status": "full_pipeline_complete" if args.stage == "full" else f"{args.stage}_complete",
            "readiness_block_reasons": readiness_block_reasons,
        }
    )
    run_manifest["derived_thresholds"] = gate_summary.get("derived_thresholds", {})
    run_manifest["required_gates_policy"] = gate_summary.get("required_gates_policy", {})
    run_manifest["tuned_batch_sizes_by_stage"] = dict(tuned_batch_sizes_by_stage)
    run_manifest["autotune_probe_history"] = dict(autotune_probe_history)
    run_manifest["equivalence_check_results"] = dict(equivalence_check_results)
    _json_dump(output_root / "run_manifest.json", run_manifest)
    _json_dump(output_root / "phase2_gate_summary.json", gate_summary)
    _write_replication_protocol(output_root)

    print(f"Wrote Phase 2 outputs to {output_root}")
    print(f"Stage: {args.stage}")
    print(f"Dataset gate pass: {gate_summary['phases']['dataset_bucket_gate']['passes']}")
    if args.stage in {"localize", "intervene", "cross_operator_verify", "cot_compare", "full"}:
        print(f"Localization gate pass: {gate_summary['phases']['localization_validity_gate']['passes']}")
    if args.stage in {"intervene", "cross_operator_verify", "full"}:
        print(f"Operator specificity gate pass: {gate_summary['phases']['operator_specificity_gate']['passes']}")
        print(f"Intervention sanity gate pass: {gate_summary['phases']['intervention_sanity_gate']['passes']}")
    if args.stage in {"cot_compare", "full"}:
        print(f"CoT gate pass: {gate_summary['phases']['cot_gating_evidence_gate']['passes']}")


if __name__ == "__main__":
    main()

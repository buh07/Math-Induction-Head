"""Phase 2 operator-specific intervention runners, schemas, and analysis helpers."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import inspect
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .experiment_runner import evaluate_bundle
from .datasets import DatasetBundle
from .hf_hooks import apply_hooks
from .hooks import AttentionHookConfig, NeuronHookConfig
from .model_introspection import get_mlp_module as _get_mlp_module, locate_layers as _locate_layers
from .arithmetic_error_taxonomy import summarize_bucket_predictions
from .operator_buckets import OperatorBucketDataset
from .statistics import quantile_index


@dataclass
class InterventionCondition:
    component_set_name: str
    operator_target: str
    intervention: str
    scale: Optional[float] = None
    patching: bool = False
    control_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionCellResult:
    evaluation_bucket: str
    accuracy_all: float
    accuracy_parsed: float
    parse_rate: float
    delta_vs_baseline: Dict[str, float]
    delta_vs_random: Optional[Dict[str, float]] = None
    per_digit_metrics: Optional[Dict[str, Any]] = None


def build_operator_intervention_sweep_result(
    *,
    model: str,
    component_set_source: str,
    task_buckets: Sequence[str],
    conditions: Sequence[InterventionCondition],
    condition_results: Sequence[Dict[str, Any]],
    notes: Optional[List[str]] = None,
    status: str = "ok",
) -> Dict[str, Any]:
    return {
        "schema_version": "operator_intervention_sweep_v1",
        "schema_revision": "1.1",
        "status": status,
        "model": model,
        "component_set_source": component_set_source,
        "task_buckets": list(task_buckets),
        "conditions": [asdict(c) for c in conditions],
        "results": list(condition_results),
        "run_metadata": {"timestamp_utc": datetime.now(timezone.utc).isoformat()},
        "notes": list(notes or []),
    }


def build_cross_operator_specificity_matrix(
    rows: Sequence[str],
    cols: Sequence[str],
    values: Mapping[str, Mapping[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    matrix_rows = []
    for row in rows:
        row_cells = {}
        for col in cols:
            row_cells[col] = dict(values.get(row, {}).get(col, {}))
        matrix_rows.append({"row": row, "cells": row_cells})
    return {
        "schema_version": "cross_operator_specificity_matrix_v1",
        "rows": list(rows),
        "cols": list(cols),
        "matrix": matrix_rows,
    }


def build_intervention_not_implemented_result(
    *,
    model: str,
    component_set_source: str,
    task_buckets: Sequence[str],
    reason: str,
) -> Dict[str, Any]:
    return build_operator_intervention_sweep_result(
        model=model,
        component_set_source=component_set_source,
        task_buckets=task_buckets,
        conditions=[],
        condition_results=[],
        notes=[reason],
        status="not_implemented",
    )


def _bootstrap_ci(values: Sequence[float], *, seed: int = 0, num_samples: int = 1000) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    vals = list(values)
    samples: List[float] = []
    for _ in range(num_samples):
        draw = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    return {
        "mean": sum(vals) / len(vals),
        "ci_low": samples[quantile_index(num_samples, 0.025)],
        "ci_high": samples[quantile_index(num_samples, 0.975)],
    }


def _paired_diff_ci(a: Sequence[float], b: Sequence[float], *, seed: int = 0, num_samples: int = 1000) -> Dict[str, float]:
    if len(a) != len(b):
        raise ValueError("Paired CI expects equal-length vectors")
    diffs = [x - y for x, y in zip(a, b)]
    return _bootstrap_ci(diffs, seed=seed, num_samples=num_samples)


def _approx_two_sided_p_from_ci(*, mean: float, ci_low: float, ci_high: float) -> Optional[float]:
    width = float(ci_high) - float(ci_low)
    if width <= 0:
        return None
    # Approximate normal-theory p-value from bootstrap CI width for reporting only.
    se = width / (2.0 * 1.96)
    if se <= 0:
        return None
    z = abs(float(mean)) / se
    return float(math.erfc(z / math.sqrt(2.0)))


def _call_with_supported_kwargs(fn, *args, **kwargs):
    supports_var_kw = False
    allowed: set[str] = set()
    try:
        sig = inspect.signature(fn)
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                supports_var_kw = True
            allowed.add(name)
    except (TypeError, ValueError):
        allowed = set()
    if supports_var_kw:
        filtered = dict(kwargs)
    else:
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(*args, **filtered)


def _bh_fdr(rows: Sequence[Dict[str, Any]], *, p_key: str = "p_value") -> List[Dict[str, Any]]:
    with_p = [(idx, float(row[p_key])) for idx, row in enumerate(rows) if row.get(p_key) is not None]
    if not with_p:
        return [dict(row, q_value=None) for row in rows]
    m = len(with_p)
    with_p_sorted = sorted(with_p, key=lambda x: x[1])
    q_vals = [0.0] * m
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        p = with_p_sorted[i][1]
        q = min(running, p * m / rank)
        running = q
        q_vals[i] = q
    idx_to_q = {with_p_sorted[i][0]: q_vals[i] for i in range(m)}
    return [dict(row, q_value=idx_to_q.get(i)) for i, row in enumerate(rows)]


def _bundle_from_operator_dataset(dataset: OperatorBucketDataset) -> DatasetBundle:
    answers: List[int] = []
    for t in dataset.targets:
        if not isinstance(t, int):
            raise ValueError("OperatorBucketDataset targets must be ints for evaluation")
        answers.append(t)
    return DatasetBundle(name=dataset.name, prompts=list(dataset.prompts), answers=answers)


def evaluate_operator_bucket_dataset(
    model,
    tokenizer,
    dataset: OperatorBucketDataset,
    *,
    parse_mode: str = "strict_final_numeric",
    max_new_tokens: int = 16,
    deterministic_generation: bool = True,
    allow_sampling_fallback: bool = False,
    batch_size: int = 1,
) -> Dict[str, Any]:
    bundle = _bundle_from_operator_dataset(dataset)
    eval_result = evaluate_bundle(
        model,
        tokenizer,
        bundle,
        parse_mode=parse_mode,
        max_new_tokens=max_new_tokens,
        deterministic_generation=deterministic_generation,
        allow_sampling_fallback=allow_sampling_fallback,
        batch_size=batch_size,
    )
    preds = [row.get("parsed") for row in eval_result.get("results", [])]
    taxonomy = summarize_bucket_predictions(dataset.examples, preds)
    per_prompt_correct = [1.0 if row.get("correct") else 0.0 for row in eval_result.get("results", [])]
    return {
        "dataset_name": dataset.name,
        "operator": dataset.operator,
        "bucket": dataset.bucket,
        "representation_variant": dataset.representation_variant,
        "evaluation": {k: v for k, v in eval_result.items() if k != "results"},
        "taxonomy": taxonomy,
        "results": eval_result.get("results", []),
        "per_prompt": {
            "correct": per_prompt_correct,
            "parsed_values": preds,
        },
    }


def _evaluate_operator_bucket_dataset_with_compat(
    model,
    tokenizer,
    dataset: OperatorBucketDataset,
    *,
    parse_mode: str,
    max_new_tokens: int,
    deterministic_generation: bool,
    allow_sampling_fallback: bool,
    batch_size: int,
) -> Dict[str, Any]:
    return _call_with_supported_kwargs(
        evaluate_operator_bucket_dataset,
        model,
        tokenizer,
        dataset,
        parse_mode=parse_mode,
        max_new_tokens=max_new_tokens,
        deterministic_generation=deterministic_generation,
        allow_sampling_fallback=allow_sampling_fallback,
        batch_size=batch_size,
    )


def _prediction_samples(eval_rows: Sequence[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    mismatches = [row for row in eval_rows if row.get("correct") is False]
    selected = list(mismatches[:limit])
    if len(selected) < limit:
        for row in eval_rows:
            if row in selected:
                continue
            selected.append(row)
            if len(selected) >= limit:
                break
    return [
        {
            "prompt": row.get("prompt"),
            # Keep both keys during the transition to preserve backward compatibility
            # with parser-audit readers that previously expected only one name.
            "output": row.get("generated"),
            "generated": row.get("generated"),
            "parsed": row.get("parsed"),
            "target": row.get("target"),
            "correct": row.get("correct"),
        }
        for row in selected
    ]


def _extract_set_label(component_set_name: str) -> str:
    if ":" not in component_set_name:
        return component_set_name
    return component_set_name.split(":", 1)[1]


def _dataset_sanity_flags(
    *,
    condition: Dict[str, Any],
    ds_payload: Dict[str, Any],
    baseline_eval: Dict[str, Any],
    policy: Mapping[str, Any],
) -> List[str]:
    flags: List[str] = []
    metrics = ds_payload.get("metrics", {})
    baseline_acc = float(baseline_eval.get("accuracy_all") or 0.0)
    cond_acc = float(metrics.get("accuracy_all") or 0.0)
    delta_vs_baseline = float(metrics.get("delta_vs_baseline_accuracy_all", {}).get("mean") or 0.0)
    delta_vs_random = metrics.get("delta_vs_random_accuracy_all")
    delta_vs_random_mean = float(delta_vs_random.get("mean") or 0.0) if isinstance(delta_vs_random, dict) else None

    if (
        baseline_acc <= float(policy.get("baseline_near_floor_max", 0.15))
        and cond_acc >= float(policy.get("high_accuracy_min", 0.9))
        and delta_vs_baseline >= float(policy.get("min_accuracy_delta", 0.5))
    ):
        flags.append("impossible_jump_near_floor_to_high_accuracy")

    set_label = _extract_set_label(str(condition.get("component_set_name") or ""))
    intervention = str(condition.get("intervention") or "")
    if set_label == "top" and intervention == "ablation" and delta_vs_random_mean is not None:
        if delta_vs_random_mean > float(policy.get("top_ablation_delta_vs_random_max", 0.1)):
            flags.append("directionality_violation_top_ablation_improves_vs_random")
    if set_label == "top" and intervention == "amplification" and delta_vs_random_mean is not None:
        if delta_vs_random_mean < float(policy.get("top_amplification_delta_vs_random_min", -0.1)):
            flags.append("directionality_violation_top_amplification_worse_vs_random")

    if (
        baseline_acc <= float(policy.get("baseline_near_floor_max", 0.15))
        and intervention == "ablation"
        and set_label in {"bottom", "top"}
        and delta_vs_baseline >= float(policy.get("ablation_large_positive_delta_min", 0.5))
    ):
        flags.append("ablation_large_positive_delta_requires_leakage_review")

    return sorted(set(flags))


def _parse_component_id(component_id: str) -> Tuple[str, Dict[str, int]]:
    if component_id.startswith("attn_head:"):
        rest = component_id.split(":", 1)[1]
        layer = int(rest.split(":H")[0].replace("L", ""))
        head = int(rest.split(":H")[1])
        return "attn_head", {"layer": layer, "head": head}
    if component_id.startswith("mlp_neuron:"):
        rest = component_id.split(":", 1)[1]
        layer_str, neuron_str = rest.split(":N")
        layer = int(layer_str.replace("L", ""))
        neuron = int(neuron_str)
        return "mlp_neuron", {"layer": layer, "neuron_index": neuron}
    if component_id.startswith("attn_layer:"):
        layer = int(component_id.split(":L", 1)[1])
        return "attn_layer", {"layer": layer}
    if component_id.startswith("mlp_layer:"):
        layer = int(component_id.split(":L", 1)[1])
        return "mlp_layer", {"layer": layer}
    raise ValueError(f"Unrecognized component_id: {component_id}")


@contextmanager
def _apply_mlp_layer_scale_hook(model, *, layer_index: int, scale: float):
    layers = _locate_layers(model)
    mlp = _get_mlp_module(layers[layer_index])
    if mlp is None:
        yield
        return

    def _hook(_m, _inp, output):
        if hasattr(output, "shape"):
            return output * scale
        if isinstance(output, tuple) and output:
            first, *rest = output
            if hasattr(first, "shape"):
                return (first * scale, *rest)
        return output

    handle = mlp.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def _apply_component_set_hooks(model, component_ids: Sequence[str], *, scale: float, strict_attention_heads: bool):
    attn_cfgs: List[AttentionHookConfig] = []
    neuron_cfgs: List[NeuronHookConfig] = []
    mlp_layer_scalers: List[Tuple[int, float]] = []
    for cid in component_ids:
        kind, vals = _parse_component_id(cid)
        if kind == "attn_head":
            attn_cfgs.append(AttentionHookConfig(layer=vals["layer"], head=vals["head"], scale=scale))
        elif kind == "attn_layer":
            attn_cfgs.append(AttentionHookConfig(layer=vals["layer"], head=None, scale=scale))
        elif kind == "mlp_neuron":
            neuron_cfgs.append(NeuronHookConfig(layer=vals["layer"], neuron_index=vals["neuron_index"], scale=scale))
        elif kind == "mlp_layer":
            mlp_layer_scalers.append((vals["layer"], scale))
    base_ctx = apply_hooks(
        model,
        attention_configs=attn_cfgs if attn_cfgs else None,
        neuron_configs=neuron_cfgs if neuron_cfgs else None,
        strict_attention_heads=strict_attention_heads,
    )
    if not mlp_layer_scalers:
        return base_ctx

    @contextmanager
    def _stacked():
        with base_ctx:
            with ExitStack() as stack:
                for layer, sc in mlp_layer_scalers:
                    stack.enter_context(_apply_mlp_layer_scale_hook(model, layer_index=layer, scale=sc))
                yield

    return _stacked()


def evaluate_component_set_on_datasets(
    model,
    tokenizer,
    *,
    datasets: Mapping[str, OperatorBucketDataset],
    component_ids: Sequence[str],
    scale: float,
    strict_attention_heads: bool = True,
    parse_mode: str = "strict_final_numeric",
    max_new_tokens: int = 16,
    deterministic_generation: bool = True,
    allow_sampling_fallback: bool = False,
    batch_size: int = 1,
) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {"datasets": {}}
    with _apply_component_set_hooks(model, component_ids, scale=scale, strict_attention_heads=strict_attention_heads):
        for name, dataset in sorted(datasets.items()):
            outputs["datasets"][name] = evaluate_operator_bucket_dataset(
                model,
                tokenizer,
                dataset,
                parse_mode=parse_mode,
                max_new_tokens=max_new_tokens,
                deterministic_generation=deterministic_generation,
                allow_sampling_fallback=allow_sampling_fallback,
                batch_size=batch_size,
            )
    return outputs


def _evaluate_component_set_on_datasets_with_compat(
    model,
    tokenizer,
    *,
    datasets: Mapping[str, OperatorBucketDataset],
    component_ids: Sequence[str],
    scale: float,
    strict_attention_heads: bool,
    parse_mode: str,
    max_new_tokens: int,
    deterministic_generation: bool,
    allow_sampling_fallback: bool,
    batch_size: int,
) -> Dict[str, Any]:
    return _call_with_supported_kwargs(
        evaluate_component_set_on_datasets,
        model,
        tokenizer,
        datasets=datasets,
        component_ids=component_ids,
        scale=scale,
        strict_attention_heads=strict_attention_heads,
        parse_mode=parse_mode,
        max_new_tokens=max_new_tokens,
        deterministic_generation=deterministic_generation,
        allow_sampling_fallback=allow_sampling_fallback,
        batch_size=batch_size,
    )


def _dataset_baseline_vectors(results_by_dataset: Dict[str, Any]) -> Dict[str, List[float]]:
    return {
        name: [float(x) for x in payload["per_prompt"]["correct"]]
        for name, payload in results_by_dataset.items()
    }


def run_operator_intervention_sweeps(
    model,
    tokenizer,
    *,
    model_name: str,
    datasets: Mapping[str, OperatorBucketDataset],
    component_sets: Mapping[str, Mapping[str, List[str]]],
    operator_target: str,
    scales: Sequence[float],
    interventions: Sequence[str],
    strict_attention_heads: bool = True,
    bootstrap_samples: int = 500,
    seed: int = 0,
    induction_baseline_sets: Optional[Mapping[str, List[str]]] = None,
    sanity_policy: Optional[Mapping[str, Any]] = None,
    primary_component_set: str = "top",
    primary_interventions: Optional[Sequence[str]] = None,
    primary_scales: Optional[Sequence[float]] = None,
    primary_k_values: Optional[Sequence[int]] = None,
    multiplicity_reporting: str = "bh_fdr",
    evaluation_parse_mode: str = "strict_final_numeric",
    max_new_tokens: int = 16,
    deterministic_generation: bool = True,
    allow_sampling_fallback: bool = False,
    batch_size: int = 1,
) -> Dict[str, Any]:
    if primary_interventions is None:
        primary_interventions = ("ablation", "amplification")
    primary_interventions_set = {str(name) for name in primary_interventions}
    primary_scale_set = {round(float(scale), 8) for scale in (primary_scales or [])}
    primary_k_set = {int(k) for k in (primary_k_values or [])}
    baseline_by_dataset = {
        name: _evaluate_operator_bucket_dataset_with_compat(
            model,
            tokenizer,
            dataset,
            parse_mode=evaluation_parse_mode,
            max_new_tokens=max_new_tokens,
            deterministic_generation=deterministic_generation,
            allow_sampling_fallback=allow_sampling_fallback,
            batch_size=batch_size,
        )
        for name, dataset in sorted(datasets.items())
    }
    baseline_vectors = _dataset_baseline_vectors(baseline_by_dataset)

    conditions: List[InterventionCondition] = []
    condition_results: List[Dict[str, Any]] = []
    sanity_policy = dict(sanity_policy or {})
    prediction_sample_size = int(sanity_policy.get("prediction_sample_size", 8))

    def _condition_record(
        *,
        k_label: str,
        set_name: str,
        component_ids: List[str],
        scale: float,
        intervention: str,
        control_type: Optional[str] = None,
    ) -> None:
        if not component_ids:
            return
        run = _evaluate_component_set_on_datasets_with_compat(
            model,
            tokenizer,
            datasets=datasets,
            component_ids=component_ids,
            scale=scale,
            strict_attention_heads=strict_attention_heads,
            parse_mode=evaluation_parse_mode,
            max_new_tokens=max_new_tokens,
            deterministic_generation=deterministic_generation,
            allow_sampling_fallback=allow_sampling_fallback,
            batch_size=batch_size,
        )
        cond = InterventionCondition(
            component_set_name=f"{k_label}:{set_name}",
            operator_target=operator_target,
            intervention=intervention,
            scale=scale,
            patching=False,
            control_type=control_type,
            metadata={"k_label": k_label, "n_components": len(component_ids)},
        )
        conditions.append(cond)

        ds_payload: Dict[str, Any] = {}
        for ds_name, ds_run in run["datasets"].items():
            eval_metrics = ds_run["evaluation"]
            taxonomy = ds_run["taxonomy"]
            cond_vec = [float(x) for x in ds_run["per_prompt"]["correct"]]
            base_vec = baseline_vectors[ds_name]
            delta_ci = _paired_diff_ci(cond_vec, base_vec, seed=seed + len(condition_results), num_samples=bootstrap_samples)
            ds_payload[ds_name] = {
                "operator": ds_run["operator"],
                "bucket": ds_run["bucket"],
                "representation_variant": ds_run["representation_variant"],
                "metrics": {
                    "accuracy_all": eval_metrics.get("accuracy_all"),
                    "accuracy_parsed": eval_metrics.get("accuracy"),
                    "parse_rate": eval_metrics.get("parse_rate"),
                    "delta_vs_baseline_accuracy_all": {
                        "mean": delta_ci["mean"],
                        "ci": [delta_ci["ci_low"], delta_ci["ci_high"]],
                    },
                    "per_digit_accuracy_mean": taxonomy.get("per_digit_accuracy_mean"),
                    "carry_position_error_rate": taxonomy.get("carry_position_error_rate"),
                    "borrow_position_error_rate": taxonomy.get("borrow_position_error_rate"),
                },
                "taxonomy": taxonomy,
                "per_prompt": ds_run["per_prompt"],
                "prediction_samples": _prediction_samples(ds_run.get("results", []), limit=prediction_sample_size),
            }
        condition_results.append({
            "condition": asdict(cond),
            "datasets": ds_payload,
        })

    for k_label, sets_for_k in sorted(component_sets.items()):
        for set_name, component_ids in sorted(sets_for_k.items()):
            for scale in scales:
                intervention = "ablation" if abs(scale - 0.0) < 1e-9 else "amplification" if scale > 1.0 else "baseline_scale"
                if intervention not in interventions and intervention != "baseline_scale":
                    continue
                control_type = "matched_random" if set_name == "random_matched" else None
                _condition_record(
                    k_label=k_label,
                    set_name=set_name,
                    component_ids=list(component_ids),
                    scale=float(scale),
                    intervention=intervention,
                    control_type=control_type,
                )

    if induction_baseline_sets:
        for set_name, component_ids in induction_baseline_sets.items():
            for scale in scales:
                intervention = "ablation" if abs(scale - 0.0) < 1e-9 else "amplification" if scale > 1.0 else "baseline_scale"
                if intervention not in interventions and intervention != "baseline_scale":
                    continue
                _condition_record(
                    k_label="induction_baseline",
                    set_name=set_name,
                    component_ids=list(component_ids),
                    scale=float(scale),
                    intervention=intervention,
                    control_type="induction_baseline",
                )

    # Fill delta_vs_random by matching same k_label/scale against random_matched if available.
    index: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for record in condition_results:
        cond = record["condition"]
        k_label = cond["metadata"].get("k_label")
        set_name = cond["component_set_name"].split(":", 1)[1]
        if set_name == "random_matched":
            index[(str(k_label), float(cond.get("scale") or 0.0))] = record
    for record in condition_results:
        cond = record["condition"]
        k_label = str(cond["metadata"].get("k_label"))
        set_name = cond["component_set_name"].split(":", 1)[1]
        if set_name == "random_matched":
            continue
        rand_record = index.get((k_label, float(cond.get("scale") or 0.0)))
        if rand_record is None:
            continue
        for ds_name, ds_payload in record["datasets"].items():
            if ds_name not in rand_record["datasets"]:
                continue
            cond_vec = [float(x) for x in ds_payload["per_prompt"]["correct"]]
            rand_vec = [float(x) for x in rand_record["datasets"][ds_name]["per_prompt"]["correct"]]
            ci = _paired_diff_ci(cond_vec, rand_vec, seed=seed + 101, num_samples=bootstrap_samples)
            ds_payload["metrics"]["delta_vs_random_accuracy_all"] = {
                "mean": ci["mean"],
                "ci": [ci["ci_low"], ci["ci_high"]],
            }

    total_flagged_datasets = 0
    flag_type_counts: Dict[str, int] = {}
    flagged_conditions: List[Dict[str, Any]] = []
    for record in condition_results:
        cond = record["condition"]
        condition_flags: List[str] = []
        flagged_dataset_names: List[str] = []
        for ds_name, ds_payload in record["datasets"].items():
            baseline_eval = baseline_by_dataset.get(ds_name, {}).get("evaluation", {})
            flags = _dataset_sanity_flags(
                condition=cond,
                ds_payload=ds_payload,
                baseline_eval=baseline_eval,
                policy=sanity_policy,
            )
            ds_payload["sanity_flags"] = flags
            if flags:
                flagged_dataset_names.append(ds_name)
                condition_flags.extend(flags)
                total_flagged_datasets += 1
                for flag in flags:
                    flag_type_counts[flag] = flag_type_counts.get(flag, 0) + 1
        deduped_flags = sorted(set(condition_flags))
        record["sanity_flags"] = deduped_flags
        if deduped_flags:
            flagged_conditions.append(
                {
                    "condition": cond,
                    "datasets": flagged_dataset_names,
                    "flags": deduped_flags,
                }
            )

    result = build_operator_intervention_sweep_result(
        model=model_name,
        component_set_source="phase2_localization.json",
        task_buckets=list(sorted(datasets.keys())),
        conditions=conditions,
        condition_results=condition_results,
        notes=[],
        status="ok",
    )
    result["baseline"] = {
        name: {
            "evaluation": payload["evaluation"],
            "taxonomy": payload["taxonomy"],
        }
        for name, payload in baseline_by_dataset.items()
    }
    result["sanity_summary"] = {
        "schema_version": "operator_intervention_sanity_v1",
        "policy": sanity_policy,
        "total_flagged_datasets": total_flagged_datasets,
        "flag_type_counts": flag_type_counts,
        "flagged_conditions": flagged_conditions,
    }
    primary_rows: List[Dict[str, Any]] = []
    directionality_checks: List[Dict[str, Any]] = []
    multiplicity_rows: List[Dict[str, Any]] = []
    for record in condition_results:
        cond = record.get("condition", {})
        set_name_full = str(cond.get("component_set_name") or "")
        set_label = _extract_set_label(set_name_full)
        intervention = str(cond.get("intervention") or "")
        for ds_name, ds_payload in (record.get("datasets") or {}).items():
            metrics = (ds_payload or {}).get("metrics", {})
            delta_rand = metrics.get("delta_vs_random_accuracy_all")
            if not isinstance(delta_rand, dict):
                continue
            mean_val = float(delta_rand.get("mean") or 0.0)
            ci_vals = delta_rand.get("ci") or [0.0, 0.0]
            ci_low = float(ci_vals[0]) if len(ci_vals) > 0 else 0.0
            ci_high = float(ci_vals[1]) if len(ci_vals) > 1 else 0.0
            row = {
                "condition": set_name_full,
                "set_label": set_label,
                "intervention": intervention,
                "scale": cond.get("scale"),
                "k_label": cond.get("metadata", {}).get("k_label"),
                "dataset": ds_name,
                "operator": ds_payload.get("operator"),
                "bucket": ds_payload.get("bucket"),
                "mean": mean_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": _approx_two_sided_p_from_ci(mean=mean_val, ci_low=ci_low, ci_high=ci_high),
                "is_primary_comparison": bool(
                    set_label == primary_component_set
                    and intervention in primary_interventions_set
                    and (
                        not primary_scale_set
                        or round(float(cond.get("scale") or 0.0), 8) in primary_scale_set
                    )
                    and (
                        not primary_k_set
                        or (
                            str(cond.get("metadata", {}).get("k_label", "")).startswith("K")
                            and str(cond.get("metadata", {}).get("k_label", ""))[1:].isdigit()
                            and int(str(cond.get("metadata", {}).get("k_label", ""))[1:]) in primary_k_set
                        )
                    )
                ),
            }
            multiplicity_rows.append(row)
            if set_label == primary_component_set and intervention in {"ablation", "amplification"}:
                primary_rows.append(row)
            if set_label == primary_component_set and intervention in {"ablation", "amplification"}:
                if intervention == "ablation" and mean_val > 0.0:
                    directionality_checks.append(
                        {
                            **row,
                            "expected_direction": "<= 0.0",
                            "violation": "ablation_improves_vs_random",
                        }
                    )
                if intervention == "amplification" and mean_val < 0.0:
                    directionality_checks.append(
                        {
                            **row,
                            "expected_direction": ">= 0.0",
                            "violation": "amplification_worse_vs_random",
                        }
                    )

    primary_results_by_dataset: Dict[str, Dict[str, Any]] = {}

    def _primary_row_score(row: Mapping[str, Any]) -> float:
        intervention = str(row.get("intervention") or "")
        mean_val = float(row.get("mean") or 0.0)
        # Necessity-style ablation evidence is stronger for more negative deltas.
        if intervention == "ablation":
            return -mean_val
        return mean_val

    for row in primary_rows:
        dataset_key = str(row["dataset"])
        int_key = str(row["intervention"])
        existing = primary_results_by_dataset.get(dataset_key, {}).get(int_key)
        if existing is None or _primary_row_score(row) > _primary_row_score(existing):
            primary_results_by_dataset.setdefault(dataset_key, {})[int_key] = row

    multiplicity_report_rows = multiplicity_rows
    primary_multiplicity_rows = [row for row in multiplicity_rows if bool(row.get("is_primary_comparison"))]
    multiplicity_method = str(multiplicity_reporting or "none")
    if multiplicity_method == "bh_fdr":
        multiplicity_report_rows = _bh_fdr(multiplicity_rows, p_key="p_value")
        primary_report_rows = _bh_fdr(primary_multiplicity_rows, p_key="p_value")
    else:
        multiplicity_report_rows = [dict(row, q_value=None) for row in multiplicity_rows]
        primary_report_rows = [dict(row, q_value=None) for row in primary_multiplicity_rows]

    # Gate blocking should use preregistered-primary multiplicity only.
    primary_q_by_key: Dict[tuple[Any, ...], Any] = {}
    for row in primary_report_rows:
        row_key = (
            row.get("set_label"),
            row.get("intervention"),
            row.get("scale"),
            row.get("dataset"),
            row.get("operator"),
        )
        primary_q_by_key[row_key] = row.get("q_value")

    multiplicity_report_rows_with_primary = []
    for row in multiplicity_report_rows:
        row_key = (
            row.get("set_label"),
            row.get("intervention"),
            row.get("scale"),
            row.get("dataset"),
            row.get("operator"),
        )
        row_copy = dict(row)
        row_copy["q_value_primary"] = primary_q_by_key.get(row_key)
        multiplicity_report_rows_with_primary.append(row_copy)

    result["analysis"] = {
        "primary_set_policy": {
            "primary_component_set": primary_component_set,
            "primary_target_metric": "delta_vs_random_accuracy_all",
            "interventions": sorted(primary_interventions_set),
            "primary_scales": sorted(primary_scale_set) if primary_scale_set else None,
            "primary_k_values": sorted(primary_k_set) if primary_k_set else None,
        },
        "primary_set_results": {
            "rows": primary_rows,
            "best_by_dataset_and_intervention": primary_results_by_dataset,
        },
        "directionality_checks": {
            "violations": directionality_checks,
            "total_violations": len(directionality_checks),
            "passes": len(directionality_checks) == 0,
        },
        "multiplicity_report": {
            "method": multiplicity_method,
            "n_tests": len(multiplicity_report_rows_with_primary),
            "n_primary_tests": len(primary_report_rows),
            "rows": multiplicity_report_rows_with_primary,
            "primary_only_rows": primary_report_rows,
        },
    }
    return result


def build_specificity_matrix_from_intervention_results(
    intervention_result: Dict[str, Any],
    *,
    include_only_interventions: Sequence[str] = ("ablation", "amplification"),
) -> Dict[str, Any]:
    rows: List[str] = []
    cols = list(intervention_result.get("task_buckets", []))
    values: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for record in intervention_result.get("results", []):
        cond = record.get("condition", {})
        if cond.get("intervention") not in include_only_interventions:
            continue
        row_key = f"{cond.get('component_set_name')}@{cond.get('intervention')}@{cond.get('scale')}"
        if row_key not in rows:
            rows.append(row_key)
        values.setdefault(row_key, {})
        for ds_name, ds_payload in record.get("datasets", {}).items():
            metrics = ds_payload.get("metrics", {})
            values[row_key][ds_name] = {
                "accuracy_all": metrics.get("accuracy_all"),
                "delta_vs_baseline_accuracy_all": metrics.get("delta_vs_baseline_accuracy_all"),
                "delta_vs_random_accuracy_all": metrics.get("delta_vs_random_accuracy_all"),
                "per_digit_accuracy_mean": metrics.get("per_digit_accuracy_mean"),
                "carry_position_error_rate": metrics.get("carry_position_error_rate"),
                "borrow_position_error_rate": metrics.get("borrow_position_error_rate"),
            }
    return build_cross_operator_specificity_matrix(rows=rows, cols=cols, values=values)


__all__ = [
    "InterventionCondition",
    "InterventionCellResult",
    "build_operator_intervention_sweep_result",
    "build_cross_operator_specificity_matrix",
    "build_intervention_not_implemented_result",
    "evaluate_operator_bucket_dataset",
    "evaluate_component_set_on_datasets",
    "run_operator_intervention_sweeps",
    "build_specificity_matrix_from_intervention_results",
]

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import DatasetBundle, load_tiered_suite
from src.experiment_runner import evaluate_bundle
from src.hf_hooks import apply_hooks
from src.hooks import AttentionHookConfig
from src.induction_detection import (
    PromptRecord,
    detect_induction_heads_with_model,
    filter_single_token_target_records,
    generate_control_prompt_suite,
    load_builtin_prompt_suite,
    topk_rank_stability_spearman,
)
from src.model_loader import load_local_model


@dataclass
class ControlBatch:
    inputs_cpu: Dict[str, torch.Tensor]
    baseline_last_logits_cpu: torch.Tensor
    target_ids_cpu: torch.Tensor
    target_valid_mask_cpu: torch.Tensor
    family_labels: List[str]


@dataclass
class ControlCache:
    name: str
    records: List[PromptRecord]
    batches: List[ControlBatch]
    filter_meta: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the induction-head validity-first suite (Plan A)")
    parser.add_argument("--model", required=True, help="Model name or path (default plan target: Meta-Llama-3-8B)")
    parser.add_argument("--cache-dir", default="../LLM Second-Order Effects/models")
    parser.add_argument("--model-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output dir (default: results/phase1/failed_or_partial/head_validity_<timestamp>)",
    )
    parser.add_argument("--devices", default=None, help="CUDA_VISIBLE_DEVICES override (e.g. 5,6,7)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed-list", default="0,1", help="Comma-separated seeds for detector/rank stability")
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--phase0-prompts", type=int, default=10)
    parser.add_argument("--control-count-per-family", type=int, default=200)
    parser.add_argument("--smoke", action="store_true", help="Reduced prompt counts and sweep sizes for smoke validation")
    parser.add_argument("--strict-head-hooks", dest="strict_head_hooks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--k-values", default="5,10")
    parser.add_argument("--scales", default="0.0,0.5,1.0,1.25,1.5,2.0")
    parser.add_argument("--downscale-others", default="none,0.9", help="Comma list; use none for no downscale")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--skip-gsm-detector", action="store_true", help="Skip gsm8k plain/cot detector comparison")
    parser.add_argument("--run-arithmetic-even-if-gates-fail", action="store_true")
    return parser.parse_args()


def _parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_float_list(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_optional_float_list(text: str) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for part in text.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"none", "null"}:
            values.append(None)
        else:
            values.append(float(token))
    return values


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bootstrap_ci(values: Sequence[float], *, seed: int = 0, num_samples: int = 1000) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    means: List[float] = []
    n = len(values)
    for _ in range(num_samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {
        "mean": sum(values) / len(values),
        "ci_low": means[int(0.025 * num_samples)],
        "ci_high": means[int(0.975 * num_samples)],
    }


def _bootstrap_median_diff_ci(
    values_a: Sequence[float], values_b: Sequence[float], *, seed: int = 0, num_samples: int = 1000
) -> Dict[str, float]:
    if len(values_a) != len(values_b):
        raise ValueError("Paired median-diff CI requires equal lengths")
    if not values_a:
        return {"median_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    paired = [a - b for a, b in zip(values_a, values_b)]
    rng = random.Random(seed)
    n = len(paired)
    medians: List[float] = []
    for _ in range(num_samples):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        sorted_sample = sorted(sample)
        medians.append(sorted_sample[n // 2] if n % 2 else 0.5 * (sorted_sample[n // 2 - 1] + sorted_sample[n // 2]))
    medians.sort()
    paired_sorted = sorted(paired)
    median = paired_sorted[n // 2] if n % 2 else 0.5 * (paired_sorted[n // 2 - 1] + paired_sorted[n // 2])
    return {
        "median_diff": float(median),
        "ci_low": float(medians[int(0.025 * num_samples)]),
        "ci_high": float(medians[int(0.975 * num_samples)]),
    }


def _zscore_map(values_by_key: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    if not values_by_key:
        return {}
    values = list(values_by_key.values())
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(var)
    if std <= 1e-12:
        return {key: 0.0 for key in values_by_key}
    return {key: (value - mean) / std for key, value in values_by_key.items()}


def _metrics_by_head(result: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    return {(int(m["layer"]), int(m["head"])): m for m in result.get("metrics", [])}


def _annotate_validated_composite(
    pos_result: Dict[str, Any],
    neg_result: Dict[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    pos = _metrics_by_head(pos_result)
    neg = _metrics_by_head(neg_result)
    common_keys = [key for key in pos.keys() if key in neg]
    pos_match = {key: float(pos[key].get("match_score", 0.0) or 0.0) for key in common_keys}
    pos_entropy = {key: float(pos[key].get("entropy", 0.0) or 0.0) for key in common_keys}
    pos_causal = {
        key: float(pos[key].get("copy_target_prob_delta_mean") or 0.0)
        for key in common_keys
    }
    neg_causal = {
        key: float(neg[key].get("copy_target_prob_delta_mean") or 0.0)
        for key in common_keys
    }
    z_match = _zscore_map(pos_match)
    z_entropy = _zscore_map(pos_entropy)
    z_pos_causal = _zscore_map(pos_causal)
    z_neg_causal = _zscore_map(neg_causal)

    for key in common_keys:
        score = z_match[key] - z_entropy[key] + z_pos_causal[key] - 0.5 * z_neg_causal[key]
        pos[key]["composite_validated_score"] = float(score)
        pos[key]["negative_control_copy_target_prob_delta_mean"] = float(neg_causal[key])

    pos_values = [pos_causal[key] for key in common_keys]
    neg_values = [neg_causal[key] for key in common_keys]
    median_ci = _bootstrap_median_diff_ci(
        pos_values,
        neg_values,
        seed=seed,
        num_samples=bootstrap_samples,
    )
    pos_median = sorted(pos_values)[len(pos_values) // 2] if pos_values else 0.0
    neg_median = sorted(neg_values)[len(neg_values) // 2] if neg_values else 0.0
    pos_result.setdefault("controls_summary", {})
    pos_result["controls_summary"].update(
        {
            "positive_vs_negative_copy_target_prob_delta": {
                "positive_median": float(pos_median),
                "negative_median": float(neg_median),
                **median_ci,
            }
        }
    )

    pos_result["rankings"]["composite_validated_top10"] = _top_heads_from_metrics(
        pos_result["metrics"], "composite_validated_score", limit=10
    )
    return pos_result


def _top_heads_from_metrics(metrics: Sequence[Dict[str, Any]], score_key: str, limit: int) -> List[Dict[str, Any]]:
    def sort_key(metric: Dict[str, Any]) -> Tuple[float, float, float]:
        score = metric.get(score_key)
        if score is None:
            score_val = float("-inf")
        else:
            score_val = float(score)
        return (score_val, float(metric.get("match_score", 0.0)), -float(metric.get("entropy", 0.0)))

    out: List[Dict[str, Any]] = []
    for metric in sorted(metrics, key=sort_key, reverse=True)[:limit]:
        out.append({
            "layer": int(metric["layer"]),
            "head": int(metric["head"]),
            score_key: metric.get(score_key),
            "match_score": metric.get("match_score"),
            "copy_target_prob_delta_mean": metric.get("copy_target_prob_delta_mean"),
            "entropy": metric.get("entropy"),
        })
    return out


def _sort_metrics(metrics: Sequence[Dict[str, Any]], score_key: str, descending: bool = True) -> List[Dict[str, Any]]:
    def key(metric: Dict[str, Any]) -> Tuple[float, float, float]:
        raw = metric.get(score_key)
        if raw is None:
            score = float("-inf") if descending else float("inf")
        else:
            score = float(raw)
        return (score, float(metric.get("match_score", 0.0)), -float(metric.get("entropy", 0.0)))

    return sorted(metrics, key=key, reverse=descending)


def _average_scores_across_runs(
    runs: Sequence[Dict[str, Any]], score_key: str
) -> Dict[Tuple[int, int], float]:
    sums: Dict[Tuple[int, int], float] = {}
    counts: Dict[Tuple[int, int], int] = {}
    for run in runs:
        for metric in run.get("metrics", []):
            key = (int(metric["layer"]), int(metric["head"]))
            value = metric.get(score_key)
            if value is None:
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums if counts.get(key, 0) > 0}


def _merge_metric_reference(runs: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    merged: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for run in runs:
        for metric in run.get("metrics", []):
            key = (int(metric["layer"]), int(metric["head"]))
            merged.setdefault(key, {}).update(metric)
    return merged


def _select_head_set(
    merged_metrics: Dict[Tuple[int, int], Dict[str, Any]],
    avg_scores: Dict[Tuple[int, int], float],
    *,
    score_key: str,
    k: int,
    mode: str,
) -> List[Tuple[int, int]]:
    if mode == "top":
        ranked = sorted(avg_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [key for key, _ in ranked[:k]]
    if mode == "bottom":
        ranked = sorted(avg_scores.items(), key=lambda kv: kv[1])
        return [key for key, _ in ranked[:k]]
    if mode == "high_match_low_causal":
        candidates = []
        match_values = []
        causal_values = []
        for key, metric in merged_metrics.items():
            if key not in avg_scores:
                continue
            match = float(metric.get("match_score", 0.0) or 0.0)
            causal = float(metric.get("copy_target_prob_delta_mean") or 0.0)
            candidates.append((key, match, causal))
            match_values.append(match)
            causal_values.append(causal)
        if not candidates:
            return []
        match_threshold = sorted(match_values)[int(0.75 * (len(match_values) - 1))]
        causal_threshold = sorted(causal_values)[len(causal_values) // 2]
        filtered = [
            (key, avg_scores[key])
            for key, match, causal in candidates
            if match >= match_threshold and causal <= causal_threshold and key in avg_scores
        ]
        filtered.sort(key=lambda kv: kv[1], reverse=True)
        return [key for key, _ in filtered[:k]]
    raise ValueError(f"Unknown head set mode: {mode}")


def _sample_matched_random_heads(
    selected: Sequence[Tuple[int, int]],
    all_heads: Sequence[Tuple[int, int]],
    *,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    selected_set = set(selected)
    by_layer: Dict[int, List[int]] = {}
    for layer, head in all_heads:
        by_layer.setdefault(layer, []).append(head)
    sampled: List[Tuple[int, int]] = []
    used: set[Tuple[int, int]] = set()
    for layer, _head in selected:
        candidates = [
            (layer, h)
            for h in by_layer.get(layer, [])
            if (layer, h) not in selected_set and (layer, h) not in used
        ]
        if not candidates:
            candidates = [
                (ly, h)
                for ly, h in all_heads
                if (ly, h) not in selected_set and (ly, h) not in used
            ]
        if not candidates:
            break
        choice = rng.choice(candidates)
        sampled.append(choice)
        used.add(choice)
    return sampled


def _build_control_cache(
    model,
    tokenizer,
    *,
    name: str,
    records: List[PromptRecord],
    batch_size: int,
) -> ControlCache:
    filtered_records, filter_meta = filter_single_token_target_records(tokenizer, records)
    if not filtered_records:
        raise ValueError(f"No control prompts survive tokenization filter for {name}")
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = torch.device("cpu")

    batches: List[ControlBatch] = []
    for start in range(0, len(filtered_records), batch_size):
        batch_records = filtered_records[start : start + batch_size]
        prompts = [record.prompt for record in batch_records]
        toks = tokenizer(prompts, padding=True, return_tensors="pt")
        inputs_cpu = {k: v.detach().cpu() for k, v in toks.items()}
        inputs = {k: v.to(input_device) for k, v in inputs_cpu.items()}
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False, return_dict=True)
        attn_mask = inputs["attention_mask"]
        last_idx = attn_mask.to(dtype=torch.long).sum(dim=-1) - 1
        last_idx = last_idx.clamp_min(0)
        rows = torch.arange(attn_mask.shape[0], device=attn_mask.device)
        last_logits = outputs.logits[rows, last_idx, :].detach().to("cpu", dtype=torch.float32)
        target_ids = torch.tensor(
            [int(record.metadata["expected_next_token_id"]) for record in batch_records], dtype=torch.long
        )
        target_valid = torch.ones(len(batch_records), dtype=torch.bool)
        batches.append(
            ControlBatch(
                inputs_cpu=inputs_cpu,
                baseline_last_logits_cpu=last_logits,
                target_ids_cpu=target_ids,
                target_valid_mask_cpu=target_valid,
                family_labels=[record.family for record in batch_records],
            )
        )
    return ControlCache(name=name, records=filtered_records, batches=batches, filter_meta=filter_meta)


def _evaluate_control_condition(
    model,
    cache: ControlCache,
    *,
    heads: Sequence[Tuple[int, int]],
    scale: float,
    downscale_others: Optional[float],
    batch_size: int,
    strict_head_hooks: bool,
    epsilon: float,
) -> Dict[str, Any]:
    del batch_size  # cached already with this batching granularity
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = torch.device("cpu")

    configs = [
        AttentionHookConfig(layer=layer, head=head, scale=scale, downscale_others=downscale_others)
        for layer, head in heads
    ]
    hook_debug = {"attention_targeted_pre_proj": {}}
    prompt_copy_correct: List[float] = []
    prompt_target_prob: List[float] = []
    prompt_target_prob_delta: List[float] = []
    prompt_kl: List[float] = []

    for batch in cache.batches:
        inputs = {k: v.to(input_device) for k, v in batch.inputs_cpu.items()}
        if configs:
            ctx = apply_hooks(
                model,
                attention_configs=configs,
                strict_attention_heads=strict_head_hooks,
                hook_debug_counters=hook_debug,
            )
        else:
            ctx = apply_hooks(model)
        with ctx:
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False, return_dict=True)
        attn_mask = inputs["attention_mask"]
        last_idx = attn_mask.to(dtype=torch.long).sum(dim=-1) - 1
        last_idx = last_idx.clamp_min(0)
        rows = torch.arange(attn_mask.shape[0], device=attn_mask.device)
        logits = outputs.logits[rows, last_idx, :].detach().to(dtype=torch.float32)

        baseline_logits = batch.baseline_last_logits_cpu.to(logits.device)
        target_ids = batch.target_ids_cpu.to(logits.device)
        probs = F.softmax(logits, dim=-1)
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        target_prob = probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
        baseline_target_prob = baseline_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
        target_prob_delta = target_prob - baseline_target_prob
        pred_ids = logits.argmax(dim=-1)
        copy_correct = (pred_ids == target_ids).to(dtype=torch.float32)

        baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        kl = (baseline_log_probs.exp() * (baseline_log_probs - log_probs)).sum(dim=-1)

        prompt_copy_correct.extend(float(x) for x in copy_correct.detach().cpu().tolist())
        prompt_target_prob.extend(float(x) for x in target_prob.detach().cpu().tolist())
        prompt_target_prob_delta.extend(float(x) for x in target_prob_delta.detach().cpu().tolist())
        prompt_kl.extend(float(x) for x in kl.detach().cpu().tolist())

    copy_accuracy = sum(prompt_copy_correct) / len(prompt_copy_correct) if prompt_copy_correct else 0.0
    copy_target_prob = sum(prompt_target_prob) / len(prompt_target_prob) if prompt_target_prob else 0.0
    next_token_kl = sum(prompt_kl) / len(prompt_kl) if prompt_kl else 0.0
    effect_nonzero_rate = (
        sum(1 for x in prompt_target_prob_delta if abs(x) > epsilon) / len(prompt_target_prob_delta)
        if prompt_target_prob_delta
        else 0.0
    )
    is_negative = "negative" in cache.name
    off_target_error_rate = copy_accuracy if is_negative else (1.0 - copy_accuracy)

    return {
        "copy_accuracy": copy_accuracy,
        "copy_target_prob": copy_target_prob,
        "copy_target_prob_delta_vs_baseline": (
            sum(prompt_target_prob_delta) / len(prompt_target_prob_delta) if prompt_target_prob_delta else 0.0
        ),
        "next_token_kl": next_token_kl,
        "off_target_error_rate": off_target_error_rate,
        "effect_nonzero_rate": effect_nonzero_rate,
        "n_prompts": len(prompt_copy_correct),
        "hook_debug_counters": hook_debug,
        "per_prompt": {
            "copy_correct": prompt_copy_correct,
            "copy_target_prob": prompt_target_prob,
            "copy_target_prob_delta_vs_baseline": prompt_target_prob_delta,
        },
    }


def _evaluate_control_sweeps(
    model,
    *,
    positive_cache: ControlCache,
    negative_cache: ControlCache,
    head_sets: Dict[str, Dict[str, List[Tuple[int, int]]]],
    k_values: Sequence[int],
    scales: Sequence[float],
    downscale_values: Sequence[Optional[float]],
    batch_size: int,
    strict_head_hooks: bool,
    epsilon: float,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"runs": []}
    for k in k_values:
        key = f"K{k}"
        sets_for_k = head_sets.get(key, {})
        for set_name, heads in sets_for_k.items():
            if not heads:
                results["runs"].append({"k": k, "set": set_name, "skipped": True, "reason": "empty_head_set"})
                continue
            for downscale in downscale_values:
                for scale in scales:
                    pos_metrics = _evaluate_control_condition(
                        model,
                        positive_cache,
                        heads=heads,
                        scale=scale,
                        downscale_others=downscale,
                        batch_size=batch_size,
                        strict_head_hooks=strict_head_hooks,
                        epsilon=epsilon,
                    )
                    neg_metrics = _evaluate_control_condition(
                        model,
                        negative_cache,
                        heads=heads,
                        scale=scale,
                        downscale_others=downscale,
                        batch_size=batch_size,
                        strict_head_hooks=strict_head_hooks,
                        epsilon=epsilon,
                    )
                    results["runs"].append(
                        {
                            "k": k,
                            "set": set_name,
                            "scale": scale,
                            "downscale_others": downscale,
                            "positive": pos_metrics,
                            "negative": neg_metrics,
                        }
                    )
    return results


def _paired_diff_ci(a: Sequence[float], b: Sequence[float], *, seed: int, num_samples: int) -> Dict[str, float]:
    if len(a) != len(b):
        raise ValueError("Paired CI expects equal lengths")
    diffs = [x - y for x, y in zip(a, b)]
    return _bootstrap_ci(diffs, seed=seed, num_samples=num_samples)


def _unpaired_mean_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    seed: int,
    num_samples: int,
) -> Dict[str, float]:
    if not a or not b:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    a = list(a)
    b = list(b)
    diffs: List[float] = []
    for _ in range(num_samples):
        sample_a = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sample_b = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append((sum(sample_a) / len(sample_a)) - (sum(sample_b) / len(sample_b)))
    diffs.sort()
    mean_diff = (sum(a) / len(a)) - (sum(b) / len(b))
    return {
        "mean": mean_diff,
        "ci_low": diffs[int(0.025 * num_samples)],
        "ci_high": diffs[int(0.975 * num_samples)],
    }


def _evaluate_phase3_gates(control_results: Dict[str, Any], *, bootstrap_samples: int, seed: int) -> Dict[str, Any]:
    runs = control_results.get("runs", [])
    top_vs_random_checks: List[Dict[str, Any]] = []
    passed = False

    for run in runs:
        if run.get("skipped") or run["set"] != "top":
            continue
        k = run["k"]
        downscale = run.get("downscale_others")
        scale = float(run["scale"])
        rand_run = next(
            (
                r
                for r in runs
                if not r.get("skipped")
                and r["k"] == k
                and r.get("downscale_others") == downscale
                and abs(float(r["scale"]) - scale) < 1e-9
                and r["set"] == "random_matched"
            ),
            None,
        )
        if rand_run is None:
            continue

        check: Dict[str, Any] = {
            "k": k,
            "scale": scale,
            "downscale_others": downscale,
            "criteria": {},
        }

        # Ablation sensitivity via paired prompt correctness difference (rand - top)
        if abs(scale - 0.0) < 1e-9:
            top_correct = run["positive"]["per_prompt"]["copy_correct"]
            rand_correct = rand_run["positive"]["per_prompt"]["copy_correct"]
            ci = _paired_diff_ci(rand_correct, top_correct, seed=seed + k, num_samples=bootstrap_samples)
            mean_diff = ci["mean"]
            criterion = {
                "mean_rand_minus_top_copy_accuracy": mean_diff,
                "ci": [ci["ci_low"], ci["ci_high"]],
                "passes": (mean_diff >= 0.10 and ci["ci_low"] > 0.0),
            }
            check["criteria"]["ablation_sensitivity"] = criterion
            passed = passed or bool(criterion["passes"])

        # Amplification sensitivity via paired target prob difference (top - rand)
        if scale > 1.0:
            top_prob = run["positive"]["per_prompt"]["copy_target_prob"]
            rand_prob = rand_run["positive"]["per_prompt"]["copy_target_prob"]
            ci = _paired_diff_ci(top_prob, rand_prob, seed=seed + int(scale * 100), num_samples=bootstrap_samples)
            criterion = {
                "mean_top_minus_rand_copy_target_prob": ci["mean"],
                "ci": [ci["ci_low"], ci["ci_high"]],
                "passes": ci["ci_low"] > 0.0,
            }
            check["criteria"]["amplification_sensitivity"] = criterion
            passed = passed or bool(criterion["passes"])

            # Specificity: positive effect > negative effect for same head set
            pos_delta = run["positive"]["per_prompt"]["copy_target_prob_delta_vs_baseline"]
            neg_delta = run["negative"]["per_prompt"]["copy_target_prob_delta_vs_baseline"]
            ci_spec = _unpaired_mean_diff_ci(
                pos_delta,
                neg_delta,
                seed=seed + 17 + int(scale * 100),
                num_samples=bootstrap_samples,
            )
            criterion_spec = {
                "mean_pos_minus_neg_delta": ci_spec["mean"],
                "ci": [ci_spec["ci_low"], ci_spec["ci_high"]],
                "passes": ci_spec["ci_low"] > 0.0,
            }
            check["criteria"]["specificity"] = criterion_spec
            passed = passed or bool(criterion_spec["passes"])

        top_vs_random_checks.append(check)

    return {
        "passes": passed,
        "checks": top_vs_random_checks,
    }


def _strip_per_prompt(control_results: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"runs": []}
    for run in control_results.get("runs", []):
        if run.get("skipped"):
            payload["runs"].append(run)
            continue
        run_copy = dict(run)
        for side in ("positive", "negative"):
            metrics = dict(run_copy[side])
            metrics.pop("per_prompt", None)
            run_copy[side] = metrics
        payload["runs"].append(run_copy)
    return payload


def _evaluate_arithmetic_sanity(
    model,
    tokenizer,
    *,
    head_sets: Dict[str, Dict[str, List[Tuple[int, int]]]],
    k_values: Sequence[int],
    scales: Sequence[float],
    dataset_seed: int,
    bootstrap_samples: int,
    strict_head_hooks: bool,
) -> Dict[str, Any]:
    suite = load_tiered_suite(seed=dataset_seed)
    bundles = {
        "tier1_in_distribution": suite.get("tier1_in_distribution"),
        "tier2_near_ood": suite.get("tier2_near_ood"),
    }

    results: Dict[str, Any] = {"datasets": list(bundles.keys()), "conditions": []}

    baseline_cache: Dict[str, Dict[str, Any]] = {}
    for dataset_name, bundle in bundles.items():
        baseline_cache[dataset_name] = evaluate_bundle(model, tokenizer, bundle)

    def _correct_vector(bundle: DatasetBundle, result: Dict[str, Any]) -> List[float]:
        vec: List[float] = []
        for item in result["results"]:
            target = item.get("target")
            parsed = item.get("parsed")
            vec.append(1.0 if (target is not None and parsed == target) else 0.0)
        if len(vec) != len(bundle.prompts):
            raise RuntimeError("Correctness vector length mismatch")
        return vec

    baseline_correct = {
        name: _correct_vector(bundle, baseline_cache[name]) for name, bundle in bundles.items()
    }

    for k in k_values:
        key = f"K{k}"
        for set_name, heads in head_sets.get(key, {}).items():
            if set_name == "high_match_low_causal" and not heads:
                continue
            for scale in [s for s in scales if abs(s - 0.5) > 1e-9 and abs(s - 2.0) > 1e-9]:
                # Per plan phase 4 uses [0.0, 1.0, 1.25, 1.5]
                if scale not in {0.0, 1.0, 1.25, 1.5}:
                    continue
                cfgs = [AttentionHookConfig(layer=ly, head=hd, scale=scale) for ly, hd in heads] if heads else []
                condition_record = {
                    "k": k,
                    "set": set_name,
                    "scale": scale,
                    "datasets": {},
                }
                for dataset_name, bundle in bundles.items():
                    if cfgs:
                        hook_debug = {"attention_targeted_pre_proj": {}}
                        with apply_hooks(
                            model,
                            attention_configs=cfgs,
                            strict_attention_heads=strict_head_hooks,
                            hook_debug_counters=hook_debug,
                        ):
                            result = evaluate_bundle(model, tokenizer, bundle)
                    else:
                        hook_debug = {"attention_targeted_pre_proj": {}}
                        result = evaluate_bundle(model, tokenizer, bundle)
                    cond_correct = _correct_vector(bundle, result)
                    base_correct = baseline_correct[dataset_name]
                    delta_ci = _paired_diff_ci(cond_correct, base_correct, seed=dataset_seed + k, num_samples=bootstrap_samples)
                    condition_record["datasets"][dataset_name] = {
                        **{k_: v for k_, v in result.items() if k_ != "results"},
                        "delta_vs_baseline_accuracy_all": {
                            "mean": delta_ci["mean"],
                            "ci": [delta_ci["ci_low"], delta_ci["ci_high"]],
                        },
                        "hook_debug_counters": hook_debug,
                    }
                results["conditions"].append(condition_record)

    results["baseline"] = {
        dataset_name: {k: v for k, v in data.items() if k != "results"}
        for dataset_name, data in baseline_cache.items()
    }
    return results


def _write_replication_protocol(output_root: Path) -> None:
    text = """# Head Validity Replication Protocol (Next Tranche)

Default next replication target: `google/gemma-2b`.

Prerequisites before multi-model replication:
- Phase 2 detector gates pass on Llama-3-8B (non-zero causal metric + positive vs negative separation)
- Phase 3 control steering gate passes on Llama-3-8B (Top-K beats matched-random on at least one criterion)
- Rank stability Spearman >= 0.7 across seeds for Top-50 validated composite ranking

Replication steps (same code path, no redesign):
1. Re-run `scripts/run_head_validity_suite.py` with `--model google/gemma-2b` and same seed list.
2. Keep prompt suites, K values, scales, and gate thresholds identical.
3. Compare gate outcomes, top-head overlap statistics, and control-steering effect sizes.
4. Only after control validity replicates, run arithmetic/GSM extensions.
"""
    (output_root / "replication_protocol.md").write_text(text, encoding="utf-8")


def _finalize_early(output_root: Path, gate_summary: Dict[str, Any], *, reason: str) -> None:
    gate_summary.setdefault("overall", {})
    gate_summary["overall"].update(
        {
            "terminated_early": True,
            "reason": reason,
            "phase0_hook_efficacy_gate": bool(gate_summary.get("phases", {}).get("phase0", {}).get("passes")),
            "phase1_prompt_filter_gate": bool(gate_summary.get("phases", {}).get("phase1", {}).get("passes")),
            "phase2_detector_validity_gate": bool(gate_summary.get("phases", {}).get("phase2", {}).get("passes")),
            "phase3_steering_validity_gate": bool(gate_summary.get("phases", {}).get("phase3", {}).get("passes")),
            "ready_for_multimodel_next_tranche": False,
        }
    )
    _json_dump(output_root / "gate_summary.json", gate_summary)
    _write_replication_protocol(output_root)
    print(f"Early exit: {reason}")
    print(f"Wrote partial outputs to {output_root}")


def main() -> None:
    args = parse_args()
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    seeds = _parse_int_list(args.seed_list)
    if not seeds:
        raise ValueError("No seeds parsed from --seed-list")
    k_values = _parse_int_list(args.k_values)
    scales = _parse_float_list(args.scales)
    downscale_values = _parse_optional_float_list(args.downscale_others)
    if not downscale_values:
        downscale_values = [None, 0.9]

    if args.smoke:
        args.control_count_per_family = min(args.control_count_per_family, 50)
        if args.phase0_prompts > 10:
            args.phase0_prompts = 10
        scales = [0.0, 1.0, 1.25]
        downscale_values = [None]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else (ROOT / "results" / "phase1" / "failed_or_partial" / f"head_validity_{ts}")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "schema_version": "head_validity_suite_v1",
        "plan": "Plan A (single-model validity-first)",
        "model": args.model,
        "cache_dir": args.cache_dir,
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "devices": args.devices,
        "seeds": seeds,
        "dataset_seed": args.dataset_seed,
        "phase0_prompts": args.phase0_prompts,
        "control_count_per_family": args.control_count_per_family,
        "batch_size": args.batch_size,
        "k_values": k_values,
        "scales": scales,
        "downscale_others": downscale_values,
        "strict_head_hooks": args.strict_head_hooks,
        "epsilon": args.epsilon,
        "smoke": args.smoke,
    }
    _json_dump(output_root / "run_manifest.json", run_manifest)

    model, tokenizer = load_local_model(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=True,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
    )

    gate_summary: Dict[str, Any] = {"phases": {}, "overall": {}}

    # Phase 0: debug causal metric sanity on small positive/negative/arithmetic prompt sets.
    phase0_positive_records = generate_control_prompt_suite("synthetic_repeat", args.phase0_prompts, seed=seeds[0])
    phase0_negative_records = generate_control_prompt_suite("synthetic_negative", args.phase0_prompts, seed=seeds[0])
    phase0_suite = load_tiered_suite(seed=args.dataset_seed)
    phase0_arith_prompts = phase0_suite.get("tier1_in_distribution").prompts[: args.phase0_prompts]

    try:
        phase0 = {
            "positive": detect_induction_heads_with_model(
                model,
                tokenizer,
                model_name=args.model,
                prompt_count=args.phase0_prompts,
                seed=seeds[0],
                prompts=None,
                prompt_records=phase0_positive_records,
                prompt_suite="synthetic_repeat",
                batch_size=args.batch_size,
                strict_head_hooks=args.strict_head_hooks,
                effect_token_policy="explicit_copy_target",
                metrics_mode="full",
                epsilon=args.epsilon,
                save_per_prompt_effects=False,
            ),
            "negative": detect_induction_heads_with_model(
                model,
                tokenizer,
                model_name=args.model,
                prompt_count=args.phase0_prompts,
                seed=seeds[0],
                prompts=None,
                prompt_records=phase0_negative_records,
                prompt_suite="synthetic_negative",
                batch_size=args.batch_size,
                strict_head_hooks=args.strict_head_hooks,
                effect_token_policy="explicit_copy_target",
                metrics_mode="full",
                epsilon=args.epsilon,
                save_per_prompt_effects=False,
            ),
            "arithmetic": detect_induction_heads_with_model(
                model,
                tokenizer,
                model_name=args.model,
                prompt_count=args.phase0_prompts,
                seed=seeds[0],
                prompts=phase0_arith_prompts,
                prompt_records=None,
                prompt_suite="phase0_tier1_sanity",
                batch_size=args.batch_size,
                strict_head_hooks=args.strict_head_hooks,
                effect_token_policy="baseline_argmax",
                metrics_mode="full",
                epsilon=args.epsilon,
                save_per_prompt_effects=False,
            ),
        }
    except ValueError as exc:
        phase0 = {"error": str(exc)}
        _json_dump(output_root / "phase0_debug.json", phase0)
        gate_summary["phases"]["phase0"] = {"passes": False, "error": str(exc)}
        _finalize_early(output_root, gate_summary, reason="phase0_prompt_filter_or_detector_failure")
        return
    _json_dump(output_root / "phase0_debug.json", phase0)

    positive_metrics0 = phase0["positive"]["metrics"]
    phase0_max = {
        "copy_target_prob_delta_abs_max": max(abs(float(m.get("copy_target_prob_delta_mean") or 0.0)) for m in positive_metrics0),
        "next_token_kl_max": max(float(m.get("next_token_kl_mean") or 0.0) for m in positive_metrics0),
        "logit_l1_delta_max": max(float(m.get("logit_l1_delta_mean") or 0.0) for m in positive_metrics0),
    }
    phase0_pass = (
        phase0_max["copy_target_prob_delta_abs_max"] > args.epsilon
        or phase0_max["next_token_kl_max"] > args.epsilon
        or phase0_max["logit_l1_delta_max"] > args.epsilon
    )
    gate_summary["phases"]["phase0"] = {"passes": phase0_pass, "max_effects": phase0_max}

    # Phase 1: build full prompt suites and tokenization filter stats.
    count_total = args.control_count_per_family * 3
    positive_records_full = generate_control_prompt_suite("synthetic_repeat", count_total, seed=seeds[0])
    negative_records_full = generate_control_prompt_suite("synthetic_negative", count_total, seed=seeds[0])
    try:
        positive_cache = _build_control_cache(
            model, tokenizer, name="positive_controls", records=positive_records_full, batch_size=args.batch_size
        )
        negative_cache = _build_control_cache(
            model, tokenizer, name="negative_controls", records=negative_records_full, batch_size=args.batch_size
        )
    except ValueError as exc:
        phase1 = {"error": str(exc)}
        _json_dump(output_root / "phase1_prompt_suites.json", phase1)
        gate_summary["phases"]["phase1"] = {"passes": False, "error": str(exc)}
        _finalize_early(output_root, gate_summary, reason="phase1_prompt_filter_failure")
        return
    phase1 = {
        "positive_filter": positive_cache.filter_meta,
        "negative_filter": negative_cache.filter_meta,
        "positive_families": sorted({record.family for record in positive_cache.records}),
        "negative_families": sorted({record.family for record in negative_cache.records}),
    }
    _json_dump(output_root / "phase1_prompt_suites.json", phase1)
    phase1_pass = (
        float(positive_cache.filter_meta.get("single_token_target_filter_rate", 0.0)) >= 0.90
        and float(negative_cache.filter_meta.get("single_token_target_filter_rate", 0.0)) >= 0.90
    )
    gate_summary["phases"]["phase1"] = {"passes": phase1_pass, **phase1}

    # Phase 2: detector runs and validated ranking.
    phase2_dir = output_root / "phase2_detector"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    pos_runs: List[Dict[str, Any]] = []
    neg_runs: List[Dict[str, Any]] = []
    gsm_plain_runs: List[Dict[str, Any]] = []
    gsm_cot_runs: List[Dict[str, Any]] = []
    for seed in seeds:
        pos = detect_induction_heads_with_model(
            model,
            tokenizer,
            model_name=args.model,
            prompt_count=len(positive_cache.records),
            seed=seed,
            prompts=None,
            prompt_records=positive_cache.records,
            prompt_suite="synthetic_repeat",
            batch_size=args.batch_size,
            strict_head_hooks=args.strict_head_hooks,
            effect_token_policy="explicit_copy_target",
            metrics_mode="full",
            epsilon=args.epsilon,
            save_per_prompt_effects=False,
        )
        neg = detect_induction_heads_with_model(
            model,
            tokenizer,
            model_name=args.model,
            prompt_count=len(negative_cache.records),
            seed=seed,
            prompts=None,
            prompt_records=negative_cache.records,
            prompt_suite="synthetic_negative",
            batch_size=args.batch_size,
            strict_head_hooks=args.strict_head_hooks,
            effect_token_policy="explicit_copy_target",
            metrics_mode="full",
            epsilon=args.epsilon,
            save_per_prompt_effects=False,
        )
        _annotate_validated_composite(pos, neg, bootstrap_samples=args.bootstrap_samples, seed=seed)
        pos_runs.append(pos)
        neg_runs.append(neg)
        _json_dump(phase2_dir / f"positive_seed{seed}.json", pos)
        _json_dump(phase2_dir / f"negative_seed{seed}.json", neg)

        if not args.skip_gsm_detector:
            plain = detect_induction_heads_with_model(
                model,
                tokenizer,
                model_name=args.model,
                prompt_count=50,
                seed=seed,
                prompts=None,
                prompt_records=None,
                prompt_suite="gsm8k_plain",
                batch_size=args.batch_size,
                strict_head_hooks=args.strict_head_hooks,
                effect_token_policy="baseline_argmax",
                metrics_mode="full",
                epsilon=args.epsilon,
                save_per_prompt_effects=False,
            )
            cot = detect_induction_heads_with_model(
                model,
                tokenizer,
                model_name=args.model,
                prompt_count=50,
                seed=seed,
                prompts=None,
                prompt_records=None,
                prompt_suite="gsm8k_cot",
                batch_size=args.batch_size,
                strict_head_hooks=args.strict_head_hooks,
                effect_token_policy="baseline_argmax",
                metrics_mode="full",
                epsilon=args.epsilon,
                save_per_prompt_effects=False,
            )
            gsm_plain_runs.append(plain)
            gsm_cot_runs.append(cot)
            _json_dump(phase2_dir / f"gsm8k_plain_seed{seed}.json", plain)
            _json_dump(phase2_dir / f"gsm8k_cot_seed{seed}.json", cot)

    rank_stability = None
    if len(pos_runs) >= 2:
        rank_stability = topk_rank_stability_spearman(
            pos_runs[0], pos_runs[1], score_key="composite_validated_score", top_k=50
        )
        if rank_stability is not None:
            for run in pos_runs:
                for metric in run.get("metrics", []):
                    metric["rank_stability_spearman"] = rank_stability

    pos_effect_nonzero_max = max(
        float(metric.get("effect_nonzero_rate") or 0.0)
        for run in pos_runs
        for metric in run.get("metrics", [])
    ) if pos_runs else 0.0

    separability_records = [run.get("controls_summary", {}).get("positive_vs_negative_copy_target_prob_delta", {}) for run in pos_runs]
    separability_pass = all(rec and float(rec.get("ci_low", -1.0)) > 0.0 for rec in separability_records)
    stability_pass = rank_stability is not None and rank_stability >= 0.7
    nonsilent_pass = pos_effect_nonzero_max > 0.0
    phase2_pass = bool(separability_pass and stability_pass and nonsilent_pass)

    phase2_summary = {
        "passes": phase2_pass,
        "rank_stability_spearman_top50": rank_stability,
        "separability": separability_records,
        "effect_nonzero_rate_max": pos_effect_nonzero_max,
        "nonsilent_pass": nonsilent_pass,
        "separability_pass": separability_pass,
        "stability_pass": stability_pass,
    }
    if gsm_plain_runs and gsm_cot_runs:
        phase2_summary["gsm_plain_vs_cot_rank_stability"] = topk_rank_stability_spearman(
            gsm_plain_runs[0], gsm_cot_runs[0], score_key="match_score", top_k=10
        )
    _json_dump(phase2_dir / "phase2_summary.json", phase2_summary)
    gate_summary["phases"]["phase2"] = phase2_summary

    # Phase 3: steering validity on controls.
    phase3_results: Optional[Dict[str, Any]] = None
    phase3_gate: Dict[str, Any] = {"passes": False, "skipped": True, "reason": "phase2_failed"}
    head_sets: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    if phase2_pass:
        merged_metrics = _merge_metric_reference(pos_runs)
        avg_validated = _average_scores_across_runs(pos_runs, "composite_validated_score")
        all_heads = sorted(avg_validated.keys())
        rng = random.Random(seeds[0] + 100)
        for k in k_values:
            top_heads = _select_head_set(merged_metrics, avg_validated, score_key="composite_validated_score", k=k, mode="top")
            bottom_heads = _select_head_set(merged_metrics, avg_validated, score_key="composite_validated_score", k=k, mode="bottom")
            high_match_low_causal = _select_head_set(
                merged_metrics,
                avg_validated,
                score_key="composite_validated_score",
                k=k,
                mode="high_match_low_causal",
            )
            random_matched = _sample_matched_random_heads(top_heads, all_heads, rng=rng)
            head_sets[f"K{k}"] = {
                "top": top_heads,
                "random_matched": random_matched,
                "bottom": bottom_heads,
                "high_match_low_causal": high_match_low_causal,
            }
        _json_dump(
            output_root / "head_sets.json",
            {
                "schema_version": "head_sets_v1",
                "model": args.model,
                "source": "phase2_validated_composite",
                "seeds": seeds,
                "head_sets": {
                    k: {name: [{"layer": ly, "head": hd} for ly, hd in heads] for name, heads in sets.items()}
                    for k, sets in head_sets.items()
                },
            },
        )

        phase3_results = _evaluate_control_sweeps(
            model,
            positive_cache=positive_cache,
            negative_cache=negative_cache,
            head_sets=head_sets,
            k_values=k_values,
            scales=scales,
            downscale_values=downscale_values,
            batch_size=args.batch_size,
            strict_head_hooks=args.strict_head_hooks,
            epsilon=args.epsilon,
        )
        _json_dump(output_root / "phase3_control_sweeps.json", _strip_per_prompt(phase3_results))
        phase3_gate = _evaluate_phase3_gates(phase3_results, bootstrap_samples=args.bootstrap_samples, seed=seeds[0])
        phase3_gate["skipped"] = False
        _json_dump(output_root / "phase3_gate_summary.json", phase3_gate)
    else:
        _json_dump(output_root / "phase3_gate_summary.json", phase3_gate)
    gate_summary["phases"]["phase3"] = phase3_gate

    # Phase 4: minimal arithmetic sanity rerun.
    run_phase4 = phase3_gate.get("passes", False) or args.run_arithmetic_even_if_gates_fail
    phase4_summary: Dict[str, Any]
    if run_phase4 and head_sets:
        arithmetic_results = _evaluate_arithmetic_sanity(
            model,
            tokenizer,
            head_sets=head_sets,
            k_values=k_values,
            scales=scales,
            dataset_seed=args.dataset_seed,
            bootstrap_samples=args.bootstrap_samples,
            strict_head_hooks=args.strict_head_hooks,
        )
        _json_dump(output_root / "phase4_arithmetic_sanity.json", arithmetic_results)
        phase4_summary = {
            "passes": True,
            "ran": True,
            "reason": "phase3_passed" if phase3_gate.get("passes") else "forced_by_flag",
            "output": "phase4_arithmetic_sanity.json",
        }
    else:
        phase4_summary = {
            "passes": False,
            "ran": False,
            "reason": "phase3_failed" if not args.run_arithmetic_even_if_gates_fail else "missing_head_sets",
        }
        _json_dump(output_root / "phase4_arithmetic_sanity.json", phase4_summary)
    gate_summary["phases"]["phase4"] = phase4_summary

    # Phase 5: replication-ready packaging.
    _write_replication_protocol(output_root)
    gate_summary["phases"]["phase5"] = {
        "passes": True,
        "replication_protocol": "replication_protocol.md",
        "head_sets_saved": (output_root / "head_sets.json").exists(),
    }

    gate_summary["overall"] = {
        "phase0_hook_efficacy_gate": bool(gate_summary["phases"]["phase0"]["passes"]),
        "phase1_prompt_filter_gate": bool(gate_summary["phases"]["phase1"]["passes"]),
        "phase2_detector_validity_gate": bool(gate_summary["phases"]["phase2"]["passes"]),
        "phase3_steering_validity_gate": bool(gate_summary["phases"]["phase3"]["passes"]),
        "ready_for_multimodel_next_tranche": bool(
            gate_summary["phases"]["phase2"].get("passes") and gate_summary["phases"]["phase3"].get("passes")
        ),
    }
    _json_dump(output_root / "gate_summary.json", gate_summary)

    print(f"Wrote head-validity suite outputs to {output_root}")
    print(json.dumps(gate_summary["overall"], indent=2))


if __name__ == "__main__":
    main()

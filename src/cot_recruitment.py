"""Phase 2 CoT vs direct-answer recruitment comparison schemas, prompt helpers, and runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import inspect
import random
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from .datasets import DatasetBundle
from .experiment_runner import evaluate_bundle
from .operator_buckets import OperatorBucketDataset, OperatorBucketExample
from .operator_interventions import evaluate_component_set_on_datasets
from .statistics import quantile_index


@dataclass
class PairedPromptExample:
    instance_id: str
    operator: str
    bucket: str
    expected_answer: int | str
    direct_prompt: str
    cot_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoTRecruitmentComparisonConfig:
    enabled: bool = True
    max_pairs: int = 100
    cot_instruction: str = "Think step by step, then give the final answer."
    direct_instruction: str = "Give only the final numeric answer."
    evaluation_parse_mode: str = "strict_final_numeric"
    max_new_tokens: int = 64
    min_pairs: int = 32
    parse_rate_min: float = 0.8
    require_accuracy_ci_excludes_zero: bool = True
    stratify_by_dataset: bool = True
    sampling_seed: int = 0
    dataset_pair_allocation: Literal["weighted_by_dataset_size", "equal"] = "weighted_by_dataset_size"
    evaluation_batch_size: int = 1
    format_lock_enabled: bool = True
    answer_marker: str = "####"
    relaxed_parser_diagnostics_enabled: bool = True


def make_direct_and_cot_prompt_pair(
    example: OperatorBucketExample,
    *,
    instance_id: str,
    direct_instruction: str = "Give only the final numeric answer.",
    cot_instruction: str = "Think step by step, then give the final answer.",
    format_lock_enabled: bool = True,
    answer_marker: str = "####",
) -> PairedPromptExample:
    base = example.prompt.strip()
    marker = str(answer_marker).strip() or "####"
    if format_lock_enabled:
        format_clause = (
            f"Return the final answer on its own line as: {marker} <integer>. "
            "Do not add any text after that line."
        )
        direct_prompt = f"{direct_instruction}\n{format_clause}\n{base}"
        cot_prompt = f"{cot_instruction}\n{format_clause}\n{base}"
    else:
        direct_prompt = f"{direct_instruction}\n{base}"
        cot_prompt = f"{cot_instruction}\n{base}"
    return PairedPromptExample(
        instance_id=instance_id,
        operator=example.operator,
        bucket=example.bucket,
        expected_answer=example.expected_answer,
        direct_prompt=direct_prompt,
        cot_prompt=cot_prompt,
        metadata={
            "representation_variant": example.representation_variant,
            "operands": list(example.operands),
        },
    )


def build_cot_recruitment_compare_result(
    *,
    model: str,
    pairs: Sequence[PairedPromptExample],
    config: CoTRecruitmentComparisonConfig,
    direct_metrics: Optional[Dict[str, Any]] = None,
    cot_metrics: Optional[Dict[str, Any]] = None,
    sensitivity_deltas: Optional[Dict[str, Any]] = None,
    notes: Optional[List[str]] = None,
    status: str = "ok",
) -> Dict[str, Any]:
    return {
        "schema_version": "cot_recruitment_compare_v1",
        "status": status,
        "model": model,
        "config": asdict(config),
        "n_pairs": len(pairs),
        "pair_manifest": [
            {
                "instance_id": pair.instance_id,
                "operator": pair.operator,
                "bucket": pair.bucket,
                "expected_answer": pair.expected_answer,
                "metadata": dict(pair.metadata),
            }
            for pair in pairs
        ],
        "direct_metrics": direct_metrics or {},
        "cot_metrics": cot_metrics or {},
        "sensitivity_deltas": sensitivity_deltas or {},
        "interpretation_guardrails": {
            "parse_rate_control_required": True,
            "formatting_vs_correctness_separation_required": True,
        },
        "run_metadata": {"timestamp_utc": datetime.now(timezone.utc).isoformat()},
        "notes": list(notes or []),
    }


def build_cot_compare_not_implemented_result(*, model: str, reason: str) -> Dict[str, Any]:
    return build_cot_recruitment_compare_result(
        model=model,
        pairs=[],
        config=CoTRecruitmentComparisonConfig(enabled=False),
        notes=[reason],
        status="not_implemented",
    )


def build_paired_prompt_examples(
    datasets: Mapping[str, OperatorBucketDataset],
    *,
    max_pairs: int,
    direct_instruction: str,
    cot_instruction: str,
    stratify_by_dataset: bool = True,
    sampling_seed: int = 0,
    dataset_pair_allocation: str = "weighted_by_dataset_size",
    format_lock_enabled: bool = True,
    answer_marker: str = "####",
) -> List[PairedPromptExample]:
    if max_pairs <= 0:
        return []
    rng = random.Random(int(sampling_seed))
    dataset_items = list(sorted(datasets.items()))
    if not dataset_items:
        return []
    if not stratify_by_dataset:
        pooled: List[tuple[str, int, OperatorBucketExample]] = []
        for dataset_name, dataset in dataset_items:
            for idx, example in enumerate(dataset.examples):
                pooled.append((dataset_name, idx, example))
        rng.shuffle(pooled)
        chosen = pooled[:max_pairs]
        return [
            make_direct_and_cot_prompt_pair(
                example,
                instance_id=f"{dataset_name}:{idx}",
                direct_instruction=direct_instruction,
                cot_instruction=cot_instruction,
                format_lock_enabled=bool(format_lock_enabled),
                answer_marker=str(answer_marker),
            )
            for dataset_name, idx, example in chosen
        ]

    # Stratified sampling by dataset with deterministic allocation policy.
    quotas: Dict[str, int] = {}
    capacities: Dict[str, int] = {}
    sizes = {dataset_name: len(dataset.examples) for dataset_name, dataset in dataset_items}
    allocation = str(dataset_pair_allocation or "weighted_by_dataset_size")
    if allocation not in {"weighted_by_dataset_size", "equal"}:
        raise ValueError(f"Unsupported dataset_pair_allocation: {allocation}")

    if allocation == "equal":
        base = max_pairs // len(dataset_items)
        for dataset_name, dataset in dataset_items:
            quota = min(base, len(dataset.examples))
            quotas[dataset_name] = quota
            capacities[dataset_name] = max(0, len(dataset.examples) - quota)
        remaining = max_pairs - sum(quotas.values())
        while remaining > 0:
            candidates = [name for name in capacities if capacities[name] > 0]
            if not candidates:
                break
            rng.shuffle(candidates)
            for dataset_name in candidates:
                if remaining <= 0:
                    break
                quotas[dataset_name] += 1
                capacities[dataset_name] -= 1
                remaining -= 1
    else:
        total_available = sum(sizes.values())
        if total_available <= 0:
            return []
        expected_total = min(max_pairs, total_available)
        residuals: Dict[str, float] = {}
        tie_break: Dict[str, float] = {}
        for dataset_name, dataset in dataset_items:
            raw_quota = expected_total * (len(dataset.examples) / total_available)
            base_quota = min(len(dataset.examples), int(raw_quota))
            quotas[dataset_name] = base_quota
            capacities[dataset_name] = max(0, len(dataset.examples) - base_quota)
            residuals[dataset_name] = raw_quota - int(raw_quota)
            tie_break[dataset_name] = rng.random()
        remaining = expected_total - sum(quotas.values())
        while remaining > 0:
            candidates = [name for name in capacities if capacities[name] > 0]
            if not candidates:
                break
            candidates.sort(key=lambda name: (-residuals.get(name, 0.0), tie_break.get(name, 0.0), name))
            chosen = candidates[0]
            quotas[chosen] += 1
            capacities[chosen] -= 1
            remaining -= 1

    selected: List[tuple[str, int, OperatorBucketExample]] = []
    leftovers: List[tuple[str, int, OperatorBucketExample]] = []
    for dataset_name, dataset in dataset_items:
        rows = list(enumerate(dataset.examples))
        rng.shuffle(rows)
        q = quotas[dataset_name]
        selected.extend((dataset_name, idx, ex) for idx, ex in rows[:q])
        leftovers.extend((dataset_name, idx, ex) for idx, ex in rows[q:])

    if len(selected) < max_pairs and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_pairs - len(selected)])
    selected = selected[:max_pairs]

    pairs: List[PairedPromptExample] = []
    for dataset_name, idx, example in selected:
        pairs.append(
            make_direct_and_cot_prompt_pair(
                example,
                instance_id=f"{dataset_name}:{idx}",
                direct_instruction=direct_instruction,
                cot_instruction=cot_instruction,
                format_lock_enabled=bool(format_lock_enabled),
                answer_marker=str(answer_marker),
            )
        )
    return pairs


def _bundle_from_pairs(pairs: Sequence[PairedPromptExample], *, mode: str) -> DatasetBundle:
    if mode not in {"direct", "cot"}:
        raise ValueError("mode must be 'direct' or 'cot'")
    prompts = [p.direct_prompt if mode == "direct" else p.cot_prompt for p in pairs]
    answers = [int(p.expected_answer) for p in pairs]
    return DatasetBundle(name=f"paired_{mode}", prompts=prompts, answers=answers)


def _metrics_from_eval(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "accuracy": eval_result.get("accuracy"),
        "accuracy_all": eval_result.get("accuracy_all"),
        "evaluated": eval_result.get("evaluated"),
        "total": eval_result.get("total"),
        "parse_rate": eval_result.get("parse_rate"),
    }


def _parse_agreement(strict_eval: Mapping[str, Any], relaxed_eval: Mapping[str, Any]) -> Dict[str, Any]:
    strict_rows = list(strict_eval.get("results", []))
    relaxed_rows = list(relaxed_eval.get("results", []))
    n = min(len(strict_rows), len(relaxed_rows))
    if n == 0:
        return {
            "n_compared": 0,
            "agreement_rate": 1.0,
            "disagreement_rate": 0.0,
            "strict_only_parsed": 0,
            "relaxed_only_parsed": 0,
            "both_parsed_disagree": 0,
        }
    strict_only = 0
    relaxed_only = 0
    both_disagree = 0
    agree = 0
    for idx in range(n):
        strict_val = strict_rows[idx].get("parsed")
        relaxed_val = relaxed_rows[idx].get("parsed")
        strict_has = strict_val is not None
        relaxed_has = relaxed_val is not None
        if strict_has and relaxed_has:
            try:
                equal = float(strict_val) == float(relaxed_val)
            except Exception:
                equal = strict_val == relaxed_val
            if equal:
                agree += 1
            else:
                both_disagree += 1
        elif strict_has and not relaxed_has:
            strict_only += 1
        elif relaxed_has and not strict_has:
            relaxed_only += 1
        else:
            agree += 1
    disagreements = strict_only + relaxed_only + both_disagree
    return {
        "n_compared": n,
        "agreement_rate": (agree / n) if n else 1.0,
        "disagreement_rate": (disagreements / n) if n else 0.0,
        "strict_only_parsed": strict_only,
        "relaxed_only_parsed": relaxed_only,
        "both_parsed_disagree": both_disagree,
    }


def _paired_correct_vector(eval_result: Dict[str, Any]) -> List[float]:
    return [1.0 if row.get("correct") else 0.0 for row in eval_result.get("results", [])]


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


def _paired_diff_ci(a: Sequence[float], b: Sequence[float], *, seed: int = 0, num_samples: int = 500) -> Optional[List[float]]:
    if len(a) != len(b) or not a:
        return None
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(num_samples):
        draw = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    low = samples[quantile_index(num_samples, 0.025)]
    high = samples[quantile_index(num_samples, 0.975)]
    return [low, high]


def run_cot_recruitment_compare(
    model,
    tokenizer,
    *,
    model_name: str,
    datasets: Mapping[str, OperatorBucketDataset],
    config: CoTRecruitmentComparisonConfig,
    sensitivity_component_ids: Optional[Sequence[str]] = None,
    sensitivity_scale: float = 0.0,
    strict_attention_heads: bool = True,
    deterministic_generation: bool = True,
    allow_sampling_fallback: bool = False,
    evaluation_batch_size: int = 1,
) -> Dict[str, Any]:
    if not config.enabled:
        return build_cot_compare_not_implemented_result(model=model_name, reason="CoT comparison disabled in config")

    pairs = build_paired_prompt_examples(
        datasets,
        max_pairs=config.max_pairs,
        direct_instruction=config.direct_instruction,
        cot_instruction=config.cot_instruction,
        stratify_by_dataset=bool(config.stratify_by_dataset),
        sampling_seed=int(config.sampling_seed),
        dataset_pair_allocation=str(config.dataset_pair_allocation),
        format_lock_enabled=bool(config.format_lock_enabled),
        answer_marker=str(config.answer_marker),
    )
    if not pairs:
        return build_cot_compare_not_implemented_result(model=model_name, reason="No paired prompts available")

    direct_bundle = _bundle_from_pairs(pairs, mode="direct")
    cot_bundle = _bundle_from_pairs(pairs, mode="cot")
    eval_kwargs = {
        "parse_mode": str(config.evaluation_parse_mode),
        "max_new_tokens": int(config.max_new_tokens),
        "deterministic_generation": bool(deterministic_generation),
        "allow_sampling_fallback": bool(allow_sampling_fallback),
        "batch_size": int(evaluation_batch_size),
    }
    direct_eval = _call_with_supported_kwargs(evaluate_bundle, model, tokenizer, direct_bundle, **eval_kwargs)
    cot_eval = _call_with_supported_kwargs(evaluate_bundle, model, tokenizer, cot_bundle, **eval_kwargs)

    direct_metrics = _metrics_from_eval(direct_eval)
    cot_metrics = _metrics_from_eval(cot_eval)
    direct_correct = _paired_correct_vector(direct_eval)
    cot_correct = _paired_correct_vector(cot_eval)
    acc_ci = _paired_diff_ci(cot_correct, direct_correct, seed=0, num_samples=500)
    sensitivity_deltas: Dict[str, Any] = {
        "baseline_direct_vs_cot": {
            "accuracy_all_delta": (cot_metrics.get("accuracy_all") or 0.0) - (direct_metrics.get("accuracy_all") or 0.0),
            "accuracy_all_delta_ci": acc_ci,
            "parse_rate_delta": (cot_metrics.get("parse_rate") or 0.0) - (direct_metrics.get("parse_rate") or 0.0),
        }
    }

    relaxed_parser_diagnostics: Optional[Dict[str, Any]] = None
    if bool(config.relaxed_parser_diagnostics_enabled):
        relaxed_kwargs = dict(eval_kwargs)
        relaxed_kwargs["parse_mode"] = "relaxed_last_number"
        direct_relaxed = _call_with_supported_kwargs(evaluate_bundle, model, tokenizer, direct_bundle, **relaxed_kwargs)
        cot_relaxed = _call_with_supported_kwargs(evaluate_bundle, model, tokenizer, cot_bundle, **relaxed_kwargs)
        relaxed_parser_diagnostics = {
            "enabled": True,
            "strict_parse_mode": str(config.evaluation_parse_mode),
            "relaxed_parse_mode": "relaxed_last_number",
            "direct": {
                "strict": _metrics_from_eval(direct_eval),
                "relaxed": _metrics_from_eval(direct_relaxed),
                "agreement": _parse_agreement(direct_eval, direct_relaxed),
            },
            "cot": {
                "strict": _metrics_from_eval(cot_eval),
                "relaxed": _metrics_from_eval(cot_relaxed),
                "agreement": _parse_agreement(cot_eval, cot_relaxed),
            },
        }

    # Minimal recruitment proxy in v1: compare intervention sensitivity on the same paired examples.
    if sensitivity_component_ids:
        direct_dataset = OperatorBucketDataset(
            operator="mixed",
            bucket="paired_direct",
            examples=[
                OperatorBucketExample(
                    prompt=p.direct_prompt,
                    expected_answer=int(p.expected_answer),
                    operator=p.operator,
                    bucket=p.bucket,
                    operands=list(p.metadata.get("operands", [])),
                    representation_variant=str(p.metadata.get("representation_variant", "plain")),
                    metadata={"instance_id": p.instance_id},
                )
                for p in pairs
            ],
            seed=0,
            representation_variant="paired_direct",
            metadata={"paired": True},
        )
        cot_dataset = OperatorBucketDataset(
            operator="mixed",
            bucket="paired_cot",
            examples=[
                OperatorBucketExample(
                    prompt=p.cot_prompt,
                    expected_answer=int(p.expected_answer),
                    operator=p.operator,
                    bucket=p.bucket,
                    operands=list(p.metadata.get("operands", [])),
                    representation_variant=str(p.metadata.get("representation_variant", "plain")),
                    metadata={"instance_id": p.instance_id},
                )
                for p in pairs
            ],
            seed=0,
            representation_variant="paired_cot",
            metadata={"paired": True},
        )
        intervention_runs = evaluate_component_set_on_datasets(
            model,
            tokenizer,
            datasets={"direct": direct_dataset, "cot": cot_dataset},
            component_ids=list(sensitivity_component_ids),
            scale=sensitivity_scale,
            strict_attention_heads=strict_attention_heads,
            parse_mode=str(config.evaluation_parse_mode),
            max_new_tokens=int(config.max_new_tokens),
            deterministic_generation=bool(deterministic_generation),
            allow_sampling_fallback=bool(allow_sampling_fallback),
            batch_size=int(evaluation_batch_size),
        )
        direct_int = intervention_runs["datasets"]["direct"]["evaluation"]
        cot_int = intervention_runs["datasets"]["cot"]["evaluation"]
        sensitivity_deltas["component_intervention"] = {
            "n_components": len(sensitivity_component_ids),
            "scale": sensitivity_scale,
            "direct_accuracy_all": direct_int.get("accuracy_all"),
            "cot_accuracy_all": cot_int.get("accuracy_all"),
            "direct_delta_vs_baseline": (direct_int.get("accuracy_all") or 0.0) - (direct_metrics.get("accuracy_all") or 0.0),
            "cot_delta_vs_baseline": (cot_int.get("accuracy_all") or 0.0) - (cot_metrics.get("accuracy_all") or 0.0),
            "direct_parse_rate_delta": (direct_int.get("parse_rate") or 0.0) - (direct_metrics.get("parse_rate") or 0.0),
            "cot_parse_rate_delta": (cot_int.get("parse_rate") or 0.0) - (cot_metrics.get("parse_rate") or 0.0),
        }

    notes = [
        "Step-level perturbation tests are not yet implemented in v1; current output covers paired direct-vs-CoT performance and optional component sensitivity.",
    ]
    out = build_cot_recruitment_compare_result(
        model=model_name,
        pairs=pairs,
        config=config,
        direct_metrics=direct_metrics,
        cot_metrics=cot_metrics,
        sensitivity_deltas=sensitivity_deltas,
        notes=notes,
        status="ok",
    )
    if relaxed_parser_diagnostics is not None:
        out["relaxed_parser_diagnostics"] = relaxed_parser_diagnostics
    out.setdefault("format_lock", {})
    out["format_lock"] = {
        "enabled": bool(config.format_lock_enabled),
        "answer_marker": str(config.answer_marker),
    }
    return out


__all__ = [
    "PairedPromptExample",
    "CoTRecruitmentComparisonConfig",
    "make_direct_and_cot_prompt_pair",
    "build_cot_recruitment_compare_result",
    "build_cot_compare_not_implemented_result",
    "build_paired_prompt_examples",
    "run_cot_recruitment_compare",
]

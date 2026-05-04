"""Phase 2 arithmetic-specific causal localization primitives and runners."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .hf_hooks import apply_hooks
from .hooks import AttentionHookConfig, NeuronHookConfig
from .induction_detection import _gather_last_valid_logits, _logit_l1_mean, _next_token_kl_mean
from .model_introspection import (
    get_attention_module as _get_attention_module,
    get_mlp_module as _get_mlp_module,
    infer_head_count as _infer_head_count,
    locate_layers as _locate_layers,
)
from .operator_buckets import OperatorBucketDataset, OperatorBucketExample, OperatorBucketSuite


@dataclass
class LocalizationConfig:
    component_type: str
    operator_filters: List[str]
    bucket_filters: List[str]
    metric_targets: str = "answer_token"
    batch_size: int = 8
    seed: int = 0
    component_sampling_seed: Optional[int] = None
    stability_mode: str = "same_set_shuffle"
    strict_attention_heads: bool = True


@dataclass
class LocalizationMetric:
    component_id: str
    component_type: str
    answer_token_logit_delta_mean: float = 0.0
    answer_token_prob_delta_mean: float = 0.0
    next_token_kl_mean: float = 0.0
    logit_l1_delta_mean: float = 0.0
    per_digit_logit_delta_mean: Optional[float] = None
    per_digit_prob_delta_mean: Optional[float] = None
    effect_nonzero_rate: float = 0.0
    rank_stability_spearman: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _TargetPromptRecord:
    prompt: str
    component_family: str  # e.g., answer_token or per_digit
    dataset_name: str
    operator: str
    bucket: str
    example_index: int
    target_id: Optional[int]
    target_valid: bool


@dataclass
class _PreparedBatch:
    inputs_cpu: Dict[str, torch.Tensor]
    baseline_last_logits_cpu: torch.Tensor
    target_ids_cpu: torch.Tensor
    target_valid_mask_cpu: torch.Tensor
    weights_cpu: torch.Tensor
    families: List[str]


@dataclass
class _PreparedCaches:
    answer_batches: List[_PreparedBatch]
    digit_batches: List[_PreparedBatch]
    prompt_set_meta: Dict[str, Any]
    dataset_names: List[str]
    records_meta: Dict[str, Any]


@dataclass
class ComponentSpec:
    component_id: str
    component_type: str
    layer: Optional[int] = None
    head: Optional[int] = None
    neuron_index: Optional[int] = None
    scale: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def answer_token_logit_delta_mean(
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    base = baseline_logits.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    abl = ablated_logits.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    return (base - abl).mean()


def answer_token_prob_delta_mean(
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    base_probs = torch.softmax(baseline_logits, dim=-1)
    abl_probs = torch.softmax(ablated_logits, dim=-1)
    base = base_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    abl = abl_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    return (base - abl).mean()


def gather_last_answer_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Alias with arithmetic-localization semantics (last valid token)."""
    return _gather_last_valid_logits(logits, attention_mask)


def build_robustness_summary(
    *,
    same_set_shuffle_invariance: Optional[float] = None,
    subsample_stability: Optional[float] = None,
    family_heldout_stability: Optional[float] = None,
    seed_robustness: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "same_set_shuffle_invariance": same_set_shuffle_invariance,
        "subsample_stability": subsample_stability,
        "family_heldout_stability": family_heldout_stability,
        "seed_robustness": seed_robustness,
    }


def _rankings(metrics: Sequence[LocalizationMetric], score_key: str = "answer_token_prob_delta_mean") -> Dict[str, Any]:
    ordered = sorted(metrics, key=lambda m: float(getattr(m, score_key)), reverse=True)
    top = [
        {
            "component_id": m.component_id,
            score_key: getattr(m, score_key),
            "component_type": m.component_type,
        }
        for m in ordered[:10]
    ]
    bottom = [
        {
            "component_id": m.component_id,
            score_key: getattr(m, score_key),
            "component_type": m.component_type,
        }
        for m in ordered[-10:]
    ]
    return {"score_key": score_key, "top10": top, "bottom10": bottom}


def build_localization_result(
    *,
    model: str,
    prompt_set: Dict[str, Any],
    config: LocalizationConfig,
    metrics: Sequence[LocalizationMetric],
    robustness_summary: Optional[Dict[str, Any]] = None,
    status: str = "ok",
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload = {
        "schema_version": "operator_localization_v1",
        "status": status,
        "model": model,
        "prompt_set": prompt_set,
        "metric_config": asdict(config),
        "metrics": [asdict(m) for m in metrics],
        "rankings": _rankings(metrics)
        if metrics
        else {"score_key": "answer_token_prob_delta_mean", "top10": [], "bottom10": []},
        "robustness_summary": robustness_summary or build_robustness_summary(),
        "run_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    if notes:
        payload["notes"] = list(notes)
    return payload


def build_localization_not_implemented_result(
    *,
    model: str,
    config: LocalizationConfig,
    prompt_set: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return build_localization_result(
        model=model,
        prompt_set=prompt_set,
        config=config,
        metrics=[],
        robustness_summary=build_robustness_summary(),
        status="not_implemented",
        notes=[reason],
    )


def summarize_logits_shift(
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> Dict[str, float]:
    return {
        "answer_token_logit_delta_mean": float(
            answer_token_logit_delta_mean(baseline_logits, ablated_logits, target_ids).item()
        ),
        "answer_token_prob_delta_mean": float(
            answer_token_prob_delta_mean(baseline_logits, ablated_logits, target_ids).item()
        ),
        "next_token_kl_mean": float(_next_token_kl_mean(baseline_logits, ablated_logits).item()),
        "logit_l1_delta_mean": float(_logit_l1_mean(baseline_logits, ablated_logits).item()),
    }


def _iter_slices(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _prepare_inputs(tokenizer, prompts: List[str], device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    batch = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = batch.get("input_ids")
    if input_ids is None:
        raise RuntimeError("Tokenizer output is missing required key 'input_ids'.")
    if input_ids.ndim < 2 or input_ids.shape[-1] <= 0:
        raise RuntimeError(
            "Tokenizer produced empty input_ids for localization prompts. "
            "Check tokenizer assets and model/tokenizer compatibility."
        )
    if device is None:
        return dict(batch)
    return {k: v.to(device) for k, v in batch.items()}


def _run_model(model, inputs: Dict[str, torch.Tensor]) -> Any:
    with torch.no_grad():
        return model(
            **inputs,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )


def _single_token_id_for_text(tokenizer, text: str) -> Optional[int]:
    candidates = []
    raw = str(text)
    if raw:
        candidates.append(raw)
        if not raw.startswith(" "):
            candidates.append(" " + raw)
    for cand in candidates:
        ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return int(ids[0])
    return None


def _example_answer_target_records(dataset_name: str, example: OperatorBucketExample, idx: int, tokenizer) -> _TargetPromptRecord:
    token_id = _single_token_id_for_text(tokenizer, str(example.expected_answer))
    return _TargetPromptRecord(
        prompt=example.prompt,
        component_family="answer_token",
        dataset_name=dataset_name,
        operator=example.operator,
        bucket=example.bucket,
        example_index=idx,
        target_id=token_id,
        target_valid=token_id is not None,
    )


def _example_digit_target_records(dataset_name: str, example: OperatorBucketExample, idx: int, tokenizer) -> List[_TargetPromptRecord]:
    out: List[_TargetPromptRecord] = []
    answer_str = str(example.expected_answer)
    if answer_str.startswith("-"):
        digits_only = answer_str[1:]
        prefix_seed = " -"
    else:
        digits_only = answer_str
        prefix_seed = " "
    if not digits_only.isdigit():
        return out
    prefix = prefix_seed
    for pos, digit in enumerate(digits_only):
        prompt = f"{example.prompt} {prefix.strip()}".rstrip()
        # Ensure one space before the next generated digit when prefix is empty after strip.
        if prompt == example.prompt:
            prompt = f"{example.prompt} "
        token_id = _single_token_id_for_text(tokenizer, digit)
        out.append(
            _TargetPromptRecord(
                prompt=prompt,
                component_family="per_digit",
                dataset_name=dataset_name,
                operator=example.operator,
                bucket=example.bucket,
                example_index=idx,
                target_id=token_id,
                target_valid=token_id is not None,
            )
        )
        prefix = (prefix + digit).strip()
    return out


def _select_examples(
    datasets: Mapping[str, OperatorBucketDataset],
    *,
    operator_filters: Sequence[str],
    bucket_filters: Sequence[str],
) -> List[Tuple[str, OperatorBucketExample, int]]:
    operator_set = set(operator_filters)
    bucket_set = set(bucket_filters)
    selected: List[Tuple[str, OperatorBucketExample, int]] = []
    for dataset_name, dataset in sorted(datasets.items()):
        if operator_set and dataset.operator not in operator_set:
            continue
        if bucket_set and dataset.bucket not in bucket_set:
            continue
        for idx, example in enumerate(dataset.examples):
            selected.append((dataset_name, example, idx))
    return selected


def _build_prepared_batches(
    model,
    tokenizer,
    records: List[_TargetPromptRecord],
    *,
    batch_size: int,
) -> List[_PreparedBatch]:
    if not records:
        return []
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    batches: List[_PreparedBatch] = []
    for chunk in _iter_slices(records, batch_size):
        prompts = [r.prompt for r in chunk]
        inputs_cpu = _prepare_inputs(tokenizer, prompts, device=None)
        inputs = {k: v.to(device) for k, v in inputs_cpu.items()}
        outputs = _run_model(model, inputs)
        last_logits = _gather_last_valid_logits(outputs.logits, inputs["attention_mask"]).detach().to("cpu", dtype=torch.float32)
        target_ids = torch.tensor([int(r.target_id or 0) for r in chunk], dtype=torch.long)
        valid_mask = torch.tensor([bool(r.target_valid) for r in chunk], dtype=torch.bool)
        # default uniform weights; can be changed later if balancing is needed
        weights = torch.ones(len(chunk), dtype=torch.float32)
        batches.append(
            _PreparedBatch(
                inputs_cpu={k: v.detach().cpu() for k, v in inputs_cpu.items()},
                baseline_last_logits_cpu=last_logits,
                target_ids_cpu=target_ids,
                target_valid_mask_cpu=valid_mask,
                weights_cpu=weights,
                families=[r.component_family for r in chunk],
            )
        )
    return batches


def prepare_localization_caches(
    model,
    tokenizer,
    datasets: Mapping[str, OperatorBucketDataset],
    *,
    operator_filters: Sequence[str],
    bucket_filters: Sequence[str],
    metric_targets: str,
    batch_size: int,
    seed: int = 0,
    shuffle_records: bool = False,
    max_examples_per_dataset: Optional[int] = None,
    subsample_fraction: Optional[float] = None,
    heldout_buckets: Optional[Sequence[str]] = None,
) -> _PreparedCaches:
    selected = _select_examples(datasets, operator_filters=operator_filters, bucket_filters=bucket_filters)
    if heldout_buckets:
        heldout = set(heldout_buckets)
        selected = [(name, ex, i) for (name, ex, i) in selected if ex.bucket not in heldout]
    if max_examples_per_dataset is not None:
        capped: List[Tuple[str, OperatorBucketExample, int]] = []
        counts: Dict[str, int] = {}
        for item in selected:
            name = item[0]
            if counts.get(name, 0) >= max_examples_per_dataset:
                continue
            counts[name] = counts.get(name, 0) + 1
            capped.append(item)
        selected = capped
    if subsample_fraction is not None and 0.0 < subsample_fraction < 1.0:
        rng = random.Random(seed)
        grouped: Dict[str, List[Tuple[str, OperatorBucketExample, int]]] = {}
        for item in selected:
            grouped.setdefault(item[0], []).append(item)
        sampled: List[Tuple[str, OperatorBucketExample, int]] = []
        for name, rows in grouped.items():
            rows = list(rows)
            rng.shuffle(rows)
            k = max(1, int(round(len(rows) * subsample_fraction)))
            sampled.extend(rows[:k])
        selected = sampled
    if shuffle_records:
        rng = random.Random(seed)
        rng.shuffle(selected)

    answer_records: List[_TargetPromptRecord] = []
    digit_records: List[_TargetPromptRecord] = []
    families: Dict[str, int] = {}
    for dataset_name, example, idx in selected:
        answer_rec = _example_answer_target_records(dataset_name, example, idx, tokenizer)
        answer_records.append(answer_rec)
        families[example.bucket] = families.get(example.bucket, 0) + 1
        if metric_targets in {"per_digit", "both"}:
            digit_records.extend(_example_digit_target_records(dataset_name, example, idx, tokenizer))

    answer_batches = _build_prepared_batches(model, tokenizer, answer_records, batch_size=batch_size)
    digit_batches = _build_prepared_batches(model, tokenizer, digit_records, batch_size=batch_size) if digit_records else []

    answer_valid = sum(1 for r in answer_records if r.target_valid)
    digit_valid = sum(1 for r in digit_records if r.target_valid)
    meta = {
        "n_examples": len(selected),
        "n_answer_records": len(answer_records),
        "n_answer_records_valid": answer_valid,
        "answer_target_valid_rate": (answer_valid / len(answer_records)) if answer_records else 0.0,
        "n_digit_records": len(digit_records),
        "n_digit_records_valid": digit_valid,
        "digit_target_valid_rate": (digit_valid / len(digit_records)) if digit_records else None,
        "families": families,
    }
    prompt_set_meta = {
        "count_examples": len(selected),
        "dataset_names": sorted({name for name, _, _ in selected}),
        "operators": sorted({ex.operator for _, ex, _ in selected}),
        "buckets": sorted({ex.bucket for _, ex, _ in selected}),
        **meta,
    }
    return _PreparedCaches(
        answer_batches=answer_batches,
        digit_batches=digit_batches,
        prompt_set_meta=prompt_set_meta,
        dataset_names=prompt_set_meta["dataset_names"],
        records_meta=meta,
    )


def _infer_mlp_neuron_count(mlp_module) -> Optional[int]:
    if mlp_module is None:
        return None
    for attr in ("down_proj", "fc2", "dense_4h_to_h", "c_proj", "proj_out"):
        proj = getattr(mlp_module, attr, None)
        in_features = getattr(proj, "in_features", None)
        if isinstance(in_features, int) and in_features > 0:
            return in_features
        weight = getattr(proj, "weight", None)
        if hasattr(weight, "shape") and len(getattr(weight, "shape", ())) >= 2:
            # GPT-2 Conv1D stores [in_features, out_features].
            in_dim = int(weight.shape[0])
            if in_dim > 0:
                return in_dim
    for attr in ("up_proj", "fc1", "dense_h_to_4h", "c_fc"):
        proj = getattr(mlp_module, attr, None)
        out_features = getattr(proj, "out_features", None)
        if isinstance(out_features, int) and out_features > 0:
            return out_features
        weight = getattr(proj, "weight", None)
        if hasattr(weight, "shape") and len(getattr(weight, "shape", ())) >= 2:
            # GPT-2 Conv1D stores [in_features, out_features].
            out_dim = int(weight.shape[1])
            if out_dim > 0:
                return out_dim
    return None


def enumerate_components(
    model,
    *,
    component_type: str,
    component_options: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> List[ComponentSpec]:
    component_options = dict(component_options or {})
    layers = _locate_layers(model)
    layer_indices = component_options.get("layer_indices")
    if layer_indices is None:
        layer_indices = list(range(len(layers)))
    layer_indices = [int(x) for x in layer_indices]
    rng = random.Random(seed)

    specs: List[ComponentSpec] = []
    if component_type == "attention_heads":
        head_limit_per_layer = component_options.get("head_limit_per_layer")
        explicit_heads = component_options.get("heads_by_layer") or {}
        for layer_idx in layer_indices:
            attn = _get_attention_module(layers[layer_idx])
            head_count = _infer_head_count(attn) or 0
            if layer_idx in explicit_heads:
                heads = [int(h) for h in explicit_heads[layer_idx]]
            else:
                heads = list(range(head_count))
                if head_limit_per_layer is not None:
                    heads = heads[: int(head_limit_per_layer)]
            for head in heads:
                specs.append(
                    ComponentSpec(
                        component_id=f"attn_head:L{layer_idx}:H{head}",
                        component_type="attention_heads",
                        layer=layer_idx,
                        head=head,
                    )
                )
        return specs

    if component_type == "mlp_neurons":
        explicit = component_options.get("neuron_indices_by_layer") or {}
        sample_per_layer = component_options.get("sample_per_layer")
        stride = component_options.get("stride")
        for layer_idx in layer_indices:
            mlp = _get_mlp_module(layers[layer_idx])
            width = _infer_mlp_neuron_count(mlp) or 0
            if width <= 0:
                continue
            if layer_idx in explicit:
                neuron_indices = [int(i) for i in explicit[layer_idx] if 0 <= int(i) < width]
            else:
                if stride is not None and int(stride) > 0:
                    neuron_indices = list(range(0, width, int(stride)))
                else:
                    neuron_indices = list(range(width))
                if sample_per_layer is not None and len(neuron_indices) > int(sample_per_layer):
                    neuron_indices = sorted(rng.sample(neuron_indices, int(sample_per_layer)))
            for neuron_index in neuron_indices:
                specs.append(
                    ComponentSpec(
                        component_id=f"mlp_neuron:L{layer_idx}:N{neuron_index}",
                        component_type="mlp_neurons",
                        layer=layer_idx,
                        neuron_index=neuron_index,
                    )
                )
        return specs

    if component_type == "layer_blocks":
        include_attention = bool(component_options.get("include_attention", True))
        include_mlp = bool(component_options.get("include_mlp", True))
        for layer_idx in layer_indices:
            if include_attention:
                specs.append(
                    ComponentSpec(
                        component_id=f"attn_layer:L{layer_idx}",
                        component_type="layer_blocks",
                        layer=layer_idx,
                        metadata={"block": "attention"},
                    )
                )
            if include_mlp:
                specs.append(
                    ComponentSpec(
                        component_id=f"mlp_layer:L{layer_idx}",
                        component_type="layer_blocks",
                        layer=layer_idx,
                        metadata={"block": "mlp"},
                    )
                )
        return specs

    raise ValueError(f"Unsupported component_type: {component_type}")


def _context_for_component(model, spec: ComponentSpec, *, strict_attention_heads: bool):
    if spec.component_type == "attention_heads":
        cfg = AttentionHookConfig(layer=int(spec.layer), head=int(spec.head), scale=spec.scale)
        return apply_hooks(model, attention_configs=[cfg], strict_attention_heads=strict_attention_heads)
    if spec.component_type == "mlp_neurons":
        cfg = NeuronHookConfig(layer=int(spec.layer), neuron_index=int(spec.neuron_index), scale=spec.scale)
        return apply_hooks(model, neuron_configs=[cfg], strict_attention_heads=strict_attention_heads)
    if spec.component_type == "layer_blocks":
        block = spec.metadata.get("block")
        if block == "attention":
            cfg = AttentionHookConfig(layer=int(spec.layer), head=None, scale=spec.scale)
            return apply_hooks(model, attention_configs=[cfg], strict_attention_heads=strict_attention_heads)
        if block == "mlp":
            return _apply_mlp_layer_scale_hook(model, layer_index=int(spec.layer), scale=spec.scale)
    return nullcontext()


@contextmanager
def _apply_mlp_layer_scale_hook(model, *, layer_index: int, scale: float):
    layers = _locate_layers(model)
    mlp = _get_mlp_module(layers[layer_index])
    if mlp is None:
        yield
        return

    def _hook(_m, _inp, output):
        if torch.is_tensor(output):
            return output * scale
        if isinstance(output, tuple) and output:
            first, *rest = output
            if torch.is_tensor(first):
                return (first * scale, *rest)
        return output

    handle = mlp.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def _score_batches(
    model,
    batches: Sequence[_PreparedBatch],
    *,
    spec: ComponentSpec,
    strict_attention_heads: bool,
    epsilon: float,
    shuffle_target_ids: bool = False,
    shuffle_seed: int = 0,
) -> Dict[str, Optional[float]]:
    if not batches:
        return {
            "target_logit_delta_mean": None,
            "target_prob_delta_mean": None,
            "next_token_kl_mean": 0.0,
            "logit_l1_delta_mean": 0.0,
            "effect_nonzero_rate": 0.0,
            "valid_count": 0,
        }
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    target_logit_delta_sum = 0.0
    target_prob_delta_sum = 0.0
    target_count = 0
    effect_nonzero_count = 0
    effect_total_count = 0
    kl_sum = 0.0
    kl_count = 0
    l1_sum = 0.0
    l1_count = 0

    for batch in batches:
        inputs = {k: v.to(device) for k, v in batch.inputs_cpu.items()}
        with _context_for_component(model, spec, strict_attention_heads=strict_attention_heads):
            outputs = _run_model(model, inputs)
        ablated_last_logits = _gather_last_valid_logits(outputs.logits, inputs["attention_mask"]).detach()
        baseline_last_logits = batch.baseline_last_logits_cpu.to(ablated_last_logits.device)
        target_ids = batch.target_ids_cpu.to(ablated_last_logits.device)
        valid_mask = batch.target_valid_mask_cpu.to(ablated_last_logits.device)
        if shuffle_target_ids and valid_mask.any():
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            if valid_indices.numel() > 1:
                gen = torch.Generator(device=target_ids.device)
                gen.manual_seed(
                    int(shuffle_seed) + int(spec.layer or 0) * 10007 + int(spec.head or spec.neuron_index or 0) * 31337
                )
                perm = valid_indices[torch.randperm(valid_indices.numel(), generator=gen, device=target_ids.device)]
                shuffled = target_ids.clone()
                shuffled[valid_indices] = target_ids[perm]
                target_ids = shuffled

        if valid_mask.any():
            base_target_logits = baseline_last_logits.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
            abl_target_logits = ablated_last_logits.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
            logit_delta = (base_target_logits - abl_target_logits)[valid_mask]

            base_probs = torch.softmax(baseline_last_logits, dim=-1)
            abl_probs = torch.softmax(ablated_last_logits, dim=-1)
            base_target_probs = base_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
            abl_target_probs = abl_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
            prob_delta = (base_target_probs - abl_target_probs)[valid_mask]

            target_logit_delta_sum += float(logit_delta.sum().item())
            target_prob_delta_sum += float(prob_delta.sum().item())
            target_count += int(logit_delta.numel())
            effect_nonzero_count += int((prob_delta.abs() > epsilon).sum().item())
            effect_total_count += int(prob_delta.numel())

        kl_sum += float(_next_token_kl_mean(baseline_last_logits, ablated_last_logits).item())
        kl_count += 1
        l1_sum += float(_logit_l1_mean(baseline_last_logits, ablated_last_logits).item())
        l1_count += 1

    return {
        "target_logit_delta_mean": (target_logit_delta_sum / target_count) if target_count else None,
        "target_prob_delta_mean": (target_prob_delta_sum / target_count) if target_count else None,
        "next_token_kl_mean": (kl_sum / kl_count) if kl_count else 0.0,
        "logit_l1_delta_mean": (l1_sum / l1_count) if l1_count else 0.0,
        "effect_nonzero_rate": (effect_nonzero_count / effect_total_count) if effect_total_count else 0.0,
        "valid_count": target_count,
    }


def _component_spec_to_metric(
    spec: ComponentSpec,
    answer_scores: Dict[str, Optional[float]],
    digit_scores: Dict[str, Optional[float]],
) -> LocalizationMetric:
    return LocalizationMetric(
        component_id=spec.component_id,
        component_type=spec.component_type,
        answer_token_logit_delta_mean=float(answer_scores.get("target_logit_delta_mean") or 0.0),
        answer_token_prob_delta_mean=float(answer_scores.get("target_prob_delta_mean") or 0.0),
        next_token_kl_mean=float(answer_scores.get("next_token_kl_mean") or 0.0),
        logit_l1_delta_mean=float(answer_scores.get("logit_l1_delta_mean") or 0.0),
        per_digit_logit_delta_mean=(
            None if digit_scores.get("target_logit_delta_mean") is None else float(digit_scores["target_logit_delta_mean"])
        ),
        per_digit_prob_delta_mean=(
            None if digit_scores.get("target_prob_delta_mean") is None else float(digit_scores["target_prob_delta_mean"])
        ),
        effect_nonzero_rate=float(answer_scores.get("effect_nonzero_rate") or 0.0),
        metadata={
            **dict(spec.metadata),
            "layer": spec.layer,
            "head": spec.head,
            "neuron_index": spec.neuron_index,
            "answer_valid_count": int(answer_scores.get("valid_count") or 0),
            "digit_valid_count": int(digit_scores.get("valid_count") or 0),
        },
    )


def run_arithmetic_localization(
    model,
    tokenizer,
    *,
    model_name: str,
    datasets: Mapping[str, OperatorBucketDataset],
    config: LocalizationConfig,
    component_options: Optional[Dict[str, Any]] = None,
    epsilon: float = 1e-4,
    max_examples_per_dataset: Optional[int] = None,
    subsample_fraction: Optional[float] = None,
    heldout_buckets: Optional[Sequence[str]] = None,
    shuffle_records: bool = False,
    shuffle_target_ids: bool = False,
) -> Dict[str, Any]:
    caches = prepare_localization_caches(
        model,
        tokenizer,
        datasets,
        operator_filters=config.operator_filters,
        bucket_filters=config.bucket_filters,
        metric_targets=config.metric_targets,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle_records=shuffle_records,
        max_examples_per_dataset=max_examples_per_dataset,
        subsample_fraction=subsample_fraction,
        heldout_buckets=heldout_buckets,
    )
    component_sampling_seed = int(config.component_sampling_seed if config.component_sampling_seed is not None else config.seed)
    specs = enumerate_components(
        model,
        component_type=config.component_type,
        component_options=component_options,
        seed=component_sampling_seed,
    )
    metrics: List[LocalizationMetric] = []
    for spec in specs:
        answer_scores = _score_batches(
            model,
            caches.answer_batches,
            spec=spec,
            strict_attention_heads=config.strict_attention_heads,
            epsilon=epsilon,
            shuffle_target_ids=shuffle_target_ids,
            shuffle_seed=config.seed,
        )
        digit_scores = (
            _score_batches(
                model,
                caches.digit_batches,
                spec=spec,
                strict_attention_heads=config.strict_attention_heads,
                epsilon=epsilon,
                shuffle_target_ids=shuffle_target_ids,
                shuffle_seed=config.seed + 1,
            )
            if config.metric_targets in {"per_digit", "both"}
            else {
                "target_logit_delta_mean": None,
                "target_prob_delta_mean": None,
                "valid_count": 0,
            }
        )
        metrics.append(_component_spec_to_metric(spec, answer_scores, digit_scores))

    notes: List[str] = []
    if config.metric_targets in {"per_digit", "both"} and caches.records_meta.get("n_digit_records", 0) == 0:
        notes.append("No per-digit target records were generated; per-digit metrics are unavailable for this prompt set.")
    if config.metric_targets in {"per_digit", "both"} and caches.records_meta.get("n_digit_records", 0) and not caches.records_meta.get("n_digit_records_valid"):
        notes.append("Per-digit target records were generated but none survived single-token target filtering.")

    result = build_localization_result(
        model=model_name,
        prompt_set=caches.prompt_set_meta,
        config=config,
        metrics=metrics,
        robustness_summary=build_robustness_summary(),
        status="ok",
        notes=notes,
    )
    result["calibration"] = {
        "target_shuffle_policy_applied": bool(shuffle_target_ids),
    }
    result["component_inventory"] = {
        "count": len(specs),
        "component_type": config.component_type,
        "component_options": component_options or {},
        "component_sampling_seed": component_sampling_seed,
        "component_catalog_hash": hashlib.sha256(
            "\n".join(spec.component_id for spec in specs).encode("utf-8")
        ).hexdigest(),
    }
    return result


def _rank_map(metrics: Sequence[Dict[str, Any]], score_key: str) -> Dict[str, int]:
    ordered = sorted(
        metrics,
        key=lambda m: (
            float("-inf") if m.get(score_key) is None else float(m.get(score_key)),
            float(m.get("next_token_kl_mean", 0.0)),
            -float(m.get("logit_l1_delta_mean", 0.0)),
        ),
        reverse=True,
    )
    return {str(m["component_id"]): idx + 1 for idx, m in enumerate(ordered)}


def _spearman_from_rank_maps(rank_a: Mapping[str, int], rank_b: Mapping[str, int], keys: Sequence[str]) -> Optional[float]:
    if len(keys) < 2:
        return None
    n = len(keys)
    diffsq = 0.0
    for key in keys:
        if key not in rank_a or key not in rank_b:
            return None
        d = rank_a[key] - rank_b[key]
        diffsq += float(d * d)
    return 1.0 - (6.0 * diffsq) / (n * (n * n - 1.0))


def topk_rank_stability_spearman_localization(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
    *,
    score_key: str = "answer_token_prob_delta_mean",
    top_k: int = 50,
) -> Optional[float]:
    metrics_a = result_a.get("metrics", [])
    metrics_b = result_b.get("metrics", [])
    if not metrics_a or not metrics_b:
        return None
    rank_a = _rank_map(metrics_a, score_key)
    rank_b = _rank_map(metrics_b, score_key)
    ordered_a = sorted(metrics_a, key=lambda m: float(m.get(score_key) or -1e30), reverse=True)[:top_k]
    ordered_b = sorted(metrics_b, key=lambda m: float(m.get(score_key) or -1e30), reverse=True)[:top_k]
    keys = list({str(m["component_id"]) for m in ordered_a + ordered_b})
    return _spearman_from_rank_maps(rank_a, rank_b, keys)


def annotate_localization_rank_stability(
    base_result: Dict[str, Any],
    *,
    same_set_shuffle_invariance: Optional[float] = None,
    subsample_stability: Optional[float] = None,
    family_heldout_stability: Optional[float] = None,
    seed_robustness: Optional[float] = None,
) -> Dict[str, Any]:
    out = dict(base_result)
    out["robustness_summary"] = build_robustness_summary(
        same_set_shuffle_invariance=same_set_shuffle_invariance,
        subsample_stability=subsample_stability,
        family_heldout_stability=family_heldout_stability,
        seed_robustness=seed_robustness,
    )
    mean_rho = None
    vals = [
        v
        for v in (
            same_set_shuffle_invariance,
            subsample_stability,
            family_heldout_stability,
            seed_robustness,
        )
        if v is not None and math.isfinite(float(v))
    ]
    if vals:
        mean_rho = float(sum(vals) / len(vals))
    metrics = []
    for m in out.get("metrics", []):
        mc = dict(m)
        mc["rank_stability_spearman"] = mean_rho
        metrics.append(mc)
    out["metrics"] = metrics
    return out


def component_sets_from_localization(
    localization_result: Dict[str, Any],
    *,
    k_values: Sequence[int] = (5, 10),
    score_key: str = "answer_token_prob_delta_mean",
    seed: int = 0,
) -> Dict[str, Dict[str, List[str]]]:
    metrics = list(localization_result.get("metrics", []))
    if not metrics:
        return {}
    ordered = sorted(metrics, key=lambda m: float(m.get(score_key) or -1e30), reverse=True)
    rng = random.Random(seed)
    all_ids = [str(m["component_id"]) for m in ordered]
    out: Dict[str, Dict[str, List[str]]] = {}
    for k in k_values:
        top = all_ids[:k]
        bottom = list(reversed(all_ids[-k:])) if k <= len(all_ids) else all_ids[::-1]
        top_set = set(top)
        remaining = [cid for cid in all_ids if cid not in top_set]
        random_ids = rng.sample(remaining, min(k, len(remaining))) if remaining else []
        out[f"K{k}"] = {
            "top": top,
            "random_matched": random_ids,
            "bottom": bottom,
        }
    return out


__all__ = [
    "LocalizationConfig",
    "LocalizationMetric",
    "ComponentSpec",
    "gather_last_answer_logits",
    "answer_token_logit_delta_mean",
    "answer_token_prob_delta_mean",
    "build_robustness_summary",
    "build_localization_result",
    "build_localization_not_implemented_result",
    "summarize_logits_shift",
    "prepare_localization_caches",
    "enumerate_components",
    "run_arithmetic_localization",
    "topk_rank_stability_spearman_localization",
    "annotate_localization_rank_stability",
    "component_sets_from_localization",
]

"""Detect and validate induction heads using attention and causal metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .datasets import generate_prompt_batch
from .hf_hooks import apply_hooks
from .hooks import AttentionHookConfig
from .model_loader import load_local_model


PROMPT_FILE_MAP = {
    "gsm8k_plain": Path("prompts/phase1/gsm8k_plain.txt"),
    "gsm8k_cot": Path("prompts/phase1/gsm8k_cot.txt"),
}


@dataclass
class PromptRecord:
    prompt: str
    family: str = "custom"
    expected_next_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadMetric:
    layer: int
    head: int
    entropy: float
    match_score: float
    logit_delta: float = 0.0
    copy_target_logit_delta_mean: Optional[float] = None
    copy_target_prob_delta_mean: Optional[float] = None
    next_token_kl_mean: float = 0.0
    logit_l1_delta_mean: float = 0.0
    effect_nonzero_rate: float = 0.0
    rank_stability_spearman: Optional[float] = None
    prompt_family_stats: Optional[Dict[str, float]] = None
    composite_score: Optional[float] = None


@dataclass
class _PreparedBatch:
    inputs_cpu: Dict[str, torch.Tensor]
    attention_mask_cpu: torch.Tensor
    last_logits_cpu: torch.Tensor
    baseline_argmax_ids_cpu: torch.Tensor
    explicit_target_ids_cpu: Optional[torch.Tensor]
    explicit_target_valid_mask_cpu: torch.Tensor
    records: List[PromptRecord]


@dataclass
class _AttentionAccumulator:
    entropy_sum: float = 0.0
    entropy_count: int = 0
    match_sum: float = 0.0
    match_count: int = 0


@dataclass
class _CausalAccumulator:
    target_logit_delta_sum: float = 0.0
    target_prob_delta_sum: float = 0.0
    target_count: int = 0
    kl_sum: float = 0.0
    kl_count: int = 0
    l1_sum: float = 0.0
    l1_count: int = 0
    effect_nonzero_count: int = 0
    effect_total_count: int = 0
    per_prompt_target_logit_deltas: List[float] = field(default_factory=list)
    per_prompt_target_prob_deltas: List[float] = field(default_factory=list)


def _attention_entropy(attn: torch.Tensor, mask: torch.Tensor) -> float:
    """Average attention entropy for all valid (non-padding) query tokens."""
    total, count = _attention_entropy_sum_count(attn, mask)
    return total / count if count else 0.0


def _attention_entropy_sum_count(attn: torch.Tensor, mask: torch.Tensor) -> Tuple[float, int]:
    if not mask.any():
        return 0.0, 0
    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
    attn = attn.clamp_min(1e-12)
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    token_entropy = -(attn * attn.log()).sum(dim=-1)
    valid = token_entropy[mask]
    return float(valid.sum().item()), int(valid.numel())


def _previous_token_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    if seq_len >= 2:
        idx = torch.arange(seq_len - 1, device=device)
        mask[idx + 1, idx] = 1.0
    return mask


def _previous_token_match(attn: torch.Tensor, mask: torch.Tensor, prev_mask: torch.Tensor) -> float:
    """Measure how much mass goes to the immediate previous token."""
    total, count = _previous_token_match_sum_count(attn, mask, prev_mask)
    return total / count if count else 0.0


def _previous_token_match_sum_count(
    attn: torch.Tensor,
    mask: torch.Tensor,
    prev_mask: torch.Tensor,
) -> Tuple[float, int]:
    if attn.shape[-1] < 2 or not mask.any():
        return 0.0, 0
    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
    prev_scores = (attn * prev_mask).sum(dim=-1)
    prev_available = mask.clone()
    prev_available[:, 0] = False
    prev_available[:, 1:] &= mask[:, :-1]
    if not prev_available.any():
        return 0.0, 0
    valid = prev_scores[prev_available]
    return float(valid.sum().item()), int(valid.numel())


def _prepare_inputs(tokenizer, prompts: List[str], device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    batch = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    )
    if device is None:
        return dict(batch)
    return {key: value.to(device) for key, value in batch.items()}


def _run_model(
    model,
    inputs: Dict[str, torch.Tensor],
    *,
    output_attentions: bool,
) -> Any:
    with torch.no_grad():
        return model(
            **inputs,
            output_attentions=output_attentions,
            use_cache=False,
            return_dict=True,
        )


def _gather_last_valid_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Gather logits at the final non-padding token for each example."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [batch, seq, vocab], got {tuple(logits.shape)}")
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected attention_mask with shape [batch, seq], got {tuple(attention_mask.shape)}"
        )
    last_indices = attention_mask.to(dtype=torch.long).sum(dim=-1) - 1
    last_indices = last_indices.clamp_min(0)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_idx, last_indices, :]


def _target_prob_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits.to(dtype=torch.float32), dim=-1)
    return probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)


def _target_logit_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    return logits.to(dtype=torch.float32).gather(1, target_ids.unsqueeze(-1)).squeeze(-1)


def _next_token_kl_mean(baseline_logits: torch.Tensor, ablated_logits: torch.Tensor) -> torch.Tensor:
    baseline_log_probs = F.log_softmax(baseline_logits.to(dtype=torch.float32), dim=-1)
    ablated_log_probs = F.log_softmax(ablated_logits.to(dtype=torch.float32), dim=-1)
    baseline_probs = baseline_log_probs.exp()
    kl = (baseline_probs * (baseline_log_probs - ablated_log_probs)).sum(dim=-1)
    return kl.mean()


def _logit_l1_mean(baseline_logits: torch.Tensor, ablated_logits: torch.Tensor) -> torch.Tensor:
    return (ablated_logits.to(dtype=torch.float32) - baseline_logits.to(dtype=torch.float32)).abs().mean(dim=-1).mean()


def _repo_git_sha() -> Optional[str]:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
        return proc.stdout.strip() or None
    except Exception:
        return None


def _infer_model_revision(model) -> Optional[str]:
    for attr in ("_commit_hash",):
        value = getattr(getattr(model, "config", object()), attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _iter_slices(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _normalize_prompt_input(prompts: Optional[List[str]], prompt_records: Optional[List[PromptRecord]]) -> List[PromptRecord]:
    if prompt_records is not None and prompts is not None:
        raise ValueError("Provide either prompts or prompt_records, not both")
    if prompt_records is not None:
        return [
            record if isinstance(record, PromptRecord) else PromptRecord(**record)  # type: ignore[arg-type]
            for record in prompt_records
        ]
    if prompts is None:
        return []
    return [PromptRecord(prompt=text, family="synthetic") for text in prompts]


def _truncate_and_shuffle_records(
    records: List[PromptRecord],
    *,
    prompt_count: int,
    seed: int,
) -> List[PromptRecord]:
    if seed is not None:
        rng = random.Random(seed)
        records = records.copy()
        rng.shuffle(records)
    if prompt_count > 0:
        return records[:prompt_count]
    return records


def _load_prompt_file(path: Path) -> List[PromptRecord]:
    text = path.read_text(encoding="utf-8")
    return [PromptRecord(prompt=line.strip(), family=path.stem) for line in text.splitlines() if line.strip()]


def load_builtin_prompt_suite(prompt_suite: str) -> Optional[List[PromptRecord]]:
    rel = PROMPT_FILE_MAP.get(prompt_suite)
    if rel is None:
        return None
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / rel
    if not path.exists() and len(rel.parts) >= 3 and rel.parts[0] == "prompts" and rel.parts[1] == "phase1":
        legacy_path = repo_root / Path("prompts") / rel.name
        if legacy_path.exists():
            path = legacy_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt suite file not found: {path}")
    return _load_prompt_file(path)


def _repeat_token_choices(rng: random.Random) -> Tuple[str, str, str]:
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return rng.choice(alphabet), rng.choice(alphabet), rng.choice(alphabet)


def _numeric_token_choices(rng: random.Random) -> Tuple[int, int, int]:
    return rng.randint(1, 49), rng.randint(1, 49), rng.randint(1, 49)


def generate_control_prompt_suite(prompt_suite: str, count: int, seed: int = 0) -> List[PromptRecord]:
    """Generate synthetic control prompt suites with explicit expected-next targets."""
    rng = random.Random(seed)
    records: List[PromptRecord] = []

    def add_short_positive(n: int) -> None:
        for _ in range(n):
            a, b, _ = _repeat_token_choices(rng)
            prompt = f"Copy pattern: {a} {b} {a}"
            records.append(
                PromptRecord(
                    prompt=f"{prompt} ",
                    family="repeat_short",
                    expected_next_text=f"{b}",
                    metadata={"a": a, "b": b},
                )
            )

    def add_delim_positive(n: int) -> None:
        for _ in range(n):
            a, b, _ = _repeat_token_choices(rng)
            prompt = f"Pattern [{a}|{b}|{a}] ->"
            records.append(
                PromptRecord(
                    prompt=f"{prompt} ",
                    family="repeat_delim",
                    expected_next_text=f"{b}",
                    metadata={"a": a, "b": b},
                )
            )

    def add_numeric_positive(n: int) -> None:
        for _ in range(n):
            x, y, _ = _numeric_token_choices(rng)
            prompt = f"Sequence: {x}, {y}, {x},"
            records.append(
                PromptRecord(
                    prompt=f"{prompt} ",
                    family="repeat_numeric",
                    expected_next_text=f"{y}",
                    metadata={"x": x, "y": y},
                )
            )

    def add_short_negative(n: int) -> None:
        for _ in range(n):
            a, b, c = _repeat_token_choices(rng)
            if c == a:
                c = chr(((ord(c) - 65 + 1) % 26) + 65)
            prompt = f"Copy pattern: {a} {b} {c}"
            records.append(
                PromptRecord(
                    prompt=f"{prompt} ",
                    family="negative_short",
                    expected_next_text=f"{b}",
                    metadata={"a": a, "b": b, "c": c},
                )
            )

    def add_delim_negative(n: int) -> None:
        for _ in range(n):
            a, b, c = _repeat_token_choices(rng)
            if c == a:
                c = chr(((ord(c) - 65 + 2) % 26) + 65)
            prompt = f"Pattern [{a}|{b}|{c}] ->"
            records.append(
                PromptRecord(
                    prompt=f"{prompt} ",
                    family="negative_delim",
                    expected_next_text=f"{b}",
                    metadata={"a": a, "b": b, "c": c},
                )
            )

    def add_numeric_negative(n: int) -> None:
        for _ in range(n):
            x, y, z = _numeric_token_choices(rng)
            if z == x:
                z = ((z + 7 - 1) % 49) + 1
            prompt = f"Sequence: {x}, {y}, {z},"
            records.append(
                PromptRecord(
                    prompt=f"{prompt} ",
                    family="negative_numeric",
                    expected_next_text=f"{y}",
                    metadata={"x": x, "y": y, "z": z},
                )
            )

    if prompt_suite == "synthetic_repeat":
        base = count // 3
        rem = count % 3
        sizes = [base + (1 if idx < rem else 0) for idx in range(3)]
        add_short_positive(sizes[0])
        add_delim_positive(sizes[1])
        add_numeric_positive(sizes[2])
    elif prompt_suite == "synthetic_repeat_numeric":
        add_numeric_positive(count)
    elif prompt_suite == "synthetic_negative":
        base = count // 3
        rem = count % 3
        sizes = [base + (1 if idx < rem else 0) for idx in range(3)]
        add_short_negative(sizes[0])
        add_delim_negative(sizes[1])
        add_numeric_negative(sizes[2])
    elif prompt_suite == "synthetic_repeat_short":
        add_short_positive(count)
    elif prompt_suite == "synthetic_repeat_delim":
        add_delim_positive(count)
    elif prompt_suite == "synthetic_negative_short":
        add_short_negative(count)
    elif prompt_suite == "synthetic_negative_delim":
        add_delim_negative(count)
    elif prompt_suite == "synthetic_negative_numeric":
        add_numeric_negative(count)
    else:
        raise ValueError(f"Unknown synthetic prompt suite: {prompt_suite}")

    return records


def _resolve_prompt_records(
    *,
    prompt_suite: Optional[str],
    prompt_count: int,
    seed: int,
    prompts: Optional[List[str]],
    prompt_records: Optional[List[PromptRecord]],
) -> Tuple[List[PromptRecord], str]:
    records = _normalize_prompt_input(prompts, prompt_records)
    suite_name = prompt_suite or "custom"
    if records:
        return _truncate_and_shuffle_records(records, prompt_count=prompt_count, seed=seed), suite_name

    if prompt_suite in PROMPT_FILE_MAP:
        records = load_builtin_prompt_suite(prompt_suite or "") or []
        return _truncate_and_shuffle_records(records, prompt_count=prompt_count, seed=seed), prompt_suite or "custom"

    if prompt_suite and prompt_suite.startswith("synthetic"):
        return generate_control_prompt_suite(prompt_suite, count=prompt_count, seed=seed), prompt_suite

    generated = generate_prompt_batch(prompt_count, seed=seed)
    return [PromptRecord(prompt=text, family="arithmetic_synthetic") for text in generated], "synthetic_arithmetic"


def _contextual_target_token_ids(
    tokenizer,
    *,
    prompt: str,
    target_text: str,
) -> Optional[List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    combined_ids = tokenizer(prompt + target_text, add_special_tokens=False)["input_ids"]
    if (
        isinstance(prompt_ids, list)
        and isinstance(combined_ids, list)
        and len(combined_ids) >= len(prompt_ids)
        and combined_ids[: len(prompt_ids)] == prompt_ids
    ):
        return list(combined_ids[len(prompt_ids) :])
    return None


def _filter_single_token_targets(tokenizer, records: List[PromptRecord]) -> Tuple[List[PromptRecord], Dict[str, Any]]:
    kept: List[PromptRecord] = []
    dropped_multi = 0
    dropped_empty = 0
    dropped_context_mismatch = 0
    for record in records:
        if record.expected_next_text is None:
            kept.append(record)
            continue
        token_ids = _contextual_target_token_ids(
            tokenizer,
            prompt=record.prompt,
            target_text=record.expected_next_text,
        )
        if token_ids is None:
            dropped_context_mismatch += 1
            token_ids = tokenizer(record.expected_next_text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            dropped_empty += 1
            continue
        if len(token_ids) != 1:
            dropped_multi += 1
            continue
        new_record = PromptRecord(
            prompt=record.prompt,
            family=record.family,
            expected_next_text=record.expected_next_text,
            metadata={**record.metadata, "expected_next_token_id": int(token_ids[0])},
        )
        kept.append(new_record)
    total_with_target = sum(1 for record in records if record.expected_next_text is not None)
    kept_with_target = sum(1 for record in kept if record.expected_next_text is not None)
    filter_rate = (kept_with_target / total_with_target) if total_with_target else 1.0
    return kept, {
        "total_records": len(records),
        "records_with_explicit_target": total_with_target,
        "records_kept": len(kept),
        "explicit_target_kept": kept_with_target,
        "dropped_multi_token_target": dropped_multi,
        "dropped_empty_target": dropped_empty,
        "dropped_context_mismatch_fallback": dropped_context_mismatch,
        "single_token_target_filter_rate": filter_rate,
    }


def filter_single_token_target_records(tokenizer, records: List[PromptRecord]) -> Tuple[List[PromptRecord], Dict[str, Any]]:
    """Public wrapper for tokenization filtering used by validity-suite orchestration."""
    return _filter_single_token_targets(tokenizer, records)


def _compute_attention_metrics_from_attentions(
    attn_acc: MutableMapping[Tuple[int, int], _AttentionAccumulator],
    attentions: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
) -> None:
    mask = attention_mask.to(dtype=torch.bool)
    seq_len = attention_mask.shape[-1]
    prev_mask = _previous_token_mask(seq_len, device=mask.device, dtype=torch.float32)
    for layer_idx, layer_attn in enumerate(attentions):
        layer_tensor = layer_attn.to(dtype=torch.float32)
        head_count = layer_tensor.shape[1]
        for head_idx in range(head_count):
            head_attn = layer_tensor[:, head_idx, :, :]
            entropy_sum, entropy_count = _attention_entropy_sum_count(head_attn, mask)
            match_sum, match_count = _previous_token_match_sum_count(head_attn, mask, prev_mask)
            acc = attn_acc.setdefault((layer_idx, head_idx), _AttentionAccumulator())
            acc.entropy_sum += entropy_sum
            acc.entropy_count += entropy_count
            acc.match_sum += match_sum
            acc.match_count += match_count


def _detect_with_loaded_model(
    model,
    tokenizer,
    *,
    model_name: str,
    prompt_count: int,
    seed: int,
    prompts: Optional[List[str]],
    prompt_records: Optional[List[PromptRecord]],
    prompt_suite: Optional[str],
    batch_size: int,
    strict_head_hooks: bool,
    effect_token_policy: str,
    metrics_mode: str,
    epsilon: float,
    save_per_prompt_effects: bool,
) -> Dict[str, Any]:
    model.config.output_attentions = True
    records, resolved_suite_name = _resolve_prompt_records(
        prompt_suite=prompt_suite,
        prompt_count=prompt_count,
        seed=seed,
        prompts=prompts,
        prompt_records=prompt_records,
    )
    if not records:
        raise ValueError("No prompts provided for induction-head detection")

    filtered_records, filter_meta = _filter_single_token_targets(tokenizer, records)
    records = filtered_records
    if not records:
        raise ValueError("No prompts remain after tokenization filter")

    try:
        input_device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - defensive
        input_device = torch.device("cpu")

    compute_attention_metrics = metrics_mode in {"basic", "causal", "full"}
    compute_causal_metrics = metrics_mode in {"causal", "full"}

    prepared_batches: List[_PreparedBatch] = []
    attention_acc: Dict[Tuple[int, int], _AttentionAccumulator] = {}
    layer_head_counts: List[int] = []
    hook_debug_counters: Dict[str, Dict[int, int]] = {"attention_targeted_pre_proj": {}}

    for batch_records in _iter_slices(records, batch_size):
        prompts_batch = [record.prompt for record in batch_records]
        inputs_cpu = _prepare_inputs(tokenizer, prompts_batch, device=None)
        attention_mask_cpu = inputs_cpu["attention_mask"].detach().cpu()
        inputs = {key: value.to(input_device) for key, value in inputs_cpu.items()}
        baseline_outputs = _run_model(model, inputs, output_attentions=compute_attention_metrics)
        if compute_attention_metrics and baseline_outputs.attentions is None:
            raise RuntimeError("Model did not return attention weights; enable output_attentions.")

        if compute_attention_metrics:
            attentions = tuple(
                attn.detach().to("cpu", dtype=torch.float32) for attn in baseline_outputs.attentions
            )
            if not layer_head_counts:
                layer_head_counts = [layer.shape[1] for layer in attentions]
            _compute_attention_metrics_from_attentions(attention_acc, attentions, attention_mask_cpu)
        elif not layer_head_counts:
            # Fallback: infer head counts from model architecture if attentions are disabled.
            layer_head_counts = [
                getattr(getattr(layer, "self_attn", None), "num_heads", 0) or 0
                for layer in getattr(getattr(model, "model", model), "layers", [])
            ]

        last_logits = _gather_last_valid_logits(baseline_outputs.logits, inputs["attention_mask"]).detach().to(
            "cpu", dtype=torch.float32
        )
        baseline_argmax_ids = last_logits.argmax(dim=-1).to(dtype=torch.long)

        explicit_ids: Optional[torch.Tensor] = None
        explicit_mask = torch.zeros(len(batch_records), dtype=torch.bool)
        if any(record.expected_next_text is not None for record in batch_records):
            explicit_ids_list = []
            for idx, record in enumerate(batch_records):
                token_id = record.metadata.get("expected_next_token_id")
                if isinstance(token_id, int):
                    explicit_ids_list.append(token_id)
                    explicit_mask[idx] = True
                else:
                    explicit_ids_list.append(0)
            explicit_ids = torch.tensor(explicit_ids_list, dtype=torch.long)

        prepared_batches.append(
            _PreparedBatch(
                inputs_cpu={key: value.detach().cpu() for key, value in inputs_cpu.items()},
                attention_mask_cpu=attention_mask_cpu,
                last_logits_cpu=last_logits,
                baseline_argmax_ids_cpu=baseline_argmax_ids.detach().cpu(),
                explicit_target_ids_cpu=explicit_ids.detach().cpu() if explicit_ids is not None else None,
                explicit_target_valid_mask_cpu=explicit_mask.detach().cpu(),
                records=list(batch_records),
            )
        )
        del baseline_outputs

    if not layer_head_counts:
        raise RuntimeError("Could not infer layer/head counts")

    causal_by_head: Dict[Tuple[int, int], _CausalAccumulator] = {}
    per_prompt_effects: Dict[str, Dict[str, List[float]]] = {}

    if compute_causal_metrics:
        if effect_token_policy not in {"baseline_argmax", "explicit_copy_target"}:
            raise ValueError(f"Unknown effect_token_policy: {effect_token_policy}")

        for layer_idx, head_count in enumerate(layer_head_counts):
            for head_idx in range(head_count):
                cfg = AttentionHookConfig(layer=layer_idx, head=head_idx, scale=0.0)
                acc = _CausalAccumulator()
                for batch in prepared_batches:
                    inputs = {key: value.to(input_device) for key, value in batch.inputs_cpu.items()}
                    with apply_hooks(
                        model,
                        attention_configs=[cfg],
                        strict_attention_heads=strict_head_hooks,
                        hook_debug_counters=hook_debug_counters,
                    ):
                        outputs = _run_model(model, inputs, output_attentions=False)
                    ablated_last_logits = _gather_last_valid_logits(outputs.logits, inputs["attention_mask"]).detach()
                    baseline_last_logits = batch.last_logits_cpu.to(ablated_last_logits.device)
                    if effect_token_policy == "explicit_copy_target" and batch.explicit_target_ids_cpu is not None:
                        target_ids = batch.explicit_target_ids_cpu.to(ablated_last_logits.device)
                        valid_mask = batch.explicit_target_valid_mask_cpu.to(ablated_last_logits.device)
                    else:
                        target_ids = batch.baseline_argmax_ids_cpu.to(ablated_last_logits.device)
                        valid_mask = torch.ones_like(target_ids, dtype=torch.bool)

                    baseline_target_logits = _target_logit_from_logits(baseline_last_logits, target_ids)
                    ablated_target_logits = _target_logit_from_logits(ablated_last_logits, target_ids)
                    target_logit_delta = baseline_target_logits - ablated_target_logits

                    baseline_target_probs = _target_prob_from_logits(baseline_last_logits, target_ids)
                    ablated_target_probs = _target_prob_from_logits(ablated_last_logits, target_ids)
                    target_prob_delta = baseline_target_probs - ablated_target_probs

                    if valid_mask.any():
                        valid_logit_delta = target_logit_delta[valid_mask]
                        valid_prob_delta = target_prob_delta[valid_mask]
                        acc.target_logit_delta_sum += float(valid_logit_delta.sum().item())
                        acc.target_prob_delta_sum += float(valid_prob_delta.sum().item())
                        acc.target_count += int(valid_mask.sum().item())
                        acc.effect_nonzero_count += int((valid_prob_delta.abs() > epsilon).sum().item())
                        acc.effect_total_count += int(valid_prob_delta.numel())
                        if save_per_prompt_effects:
                            acc.per_prompt_target_logit_deltas.extend(
                                float(x) for x in valid_logit_delta.detach().cpu().tolist()
                            )
                            acc.per_prompt_target_prob_deltas.extend(
                                float(x) for x in valid_prob_delta.detach().cpu().tolist()
                            )

                    kl_value = _next_token_kl_mean(baseline_last_logits, ablated_last_logits)
                    l1_value = _logit_l1_mean(baseline_last_logits, ablated_last_logits)
                    acc.kl_sum += float(kl_value.item())
                    acc.kl_count += 1
                    acc.l1_sum += float(l1_value.item())
                    acc.l1_count += 1

                causal_by_head[(layer_idx, head_idx)] = acc
                if save_per_prompt_effects and (acc.per_prompt_target_logit_deltas or acc.per_prompt_target_prob_deltas):
                    per_prompt_effects[f"{layer_idx}:{head_idx}"] = {
                        "target_logit_deltas": acc.per_prompt_target_logit_deltas,
                        "target_prob_deltas": acc.per_prompt_target_prob_deltas,
                    }

    metrics: List[Dict[str, Any]] = []
    for layer_idx, head_count in enumerate(layer_head_counts):
        for head_idx in range(head_count):
            attn = attention_acc.get((layer_idx, head_idx), _AttentionAccumulator())
            causal = causal_by_head.get((layer_idx, head_idx), _CausalAccumulator())
            entropy = (attn.entropy_sum / attn.entropy_count) if attn.entropy_count else 0.0
            match_score = (attn.match_sum / attn.match_count) if attn.match_count else 0.0
            target_logit_delta_mean = (
                causal.target_logit_delta_sum / causal.target_count if causal.target_count else 0.0
            )
            target_prob_delta_mean = (
                causal.target_prob_delta_sum / causal.target_count if causal.target_count else 0.0
            )
            next_token_kl_mean = causal.kl_sum / causal.kl_count if causal.kl_count else 0.0
            logit_l1_delta_mean = causal.l1_sum / causal.l1_count if causal.l1_count else 0.0
            effect_nonzero_rate = (
                causal.effect_nonzero_count / causal.effect_total_count if causal.effect_total_count else 0.0
            )

            explicit_metric_available = effect_token_policy == "explicit_copy_target" and causal.target_count > 0
            head_metric = HeadMetric(
                layer=layer_idx,
                head=head_idx,
                entropy=float(entropy),
                match_score=float(match_score),
                logit_delta=float(target_logit_delta_mean),
                copy_target_logit_delta_mean=float(target_logit_delta_mean)
                if explicit_metric_available and compute_causal_metrics
                else None,
                copy_target_prob_delta_mean=float(target_prob_delta_mean)
                if explicit_metric_available and compute_causal_metrics
                else None,
                next_token_kl_mean=float(next_token_kl_mean) if compute_causal_metrics else 0.0,
                logit_l1_delta_mean=float(logit_l1_delta_mean) if compute_causal_metrics else 0.0,
                effect_nonzero_rate=float(effect_nonzero_rate) if compute_causal_metrics else 0.0,
            )
            metrics.append(
                {
                    "layer": head_metric.layer,
                    "head": head_metric.head,
                    "entropy": head_metric.entropy,
                    "match_score": head_metric.match_score,
                    "logit_delta": head_metric.logit_delta,
                    "copy_target_logit_delta_mean": head_metric.copy_target_logit_delta_mean,
                    "copy_target_prob_delta_mean": head_metric.copy_target_prob_delta_mean,
                    "next_token_kl_mean": head_metric.next_token_kl_mean,
                    "logit_l1_delta_mean": head_metric.logit_l1_delta_mean,
                    "effect_nonzero_rate": head_metric.effect_nonzero_rate,
                    "rank_stability_spearman": None,
                    "prompt_family_stats": None,
                    "composite_score": None,
                }
            )

    _annotate_scores_and_rankings(metrics)

    result: Dict[str, Any] = {
        "schema_version": "induction_detection_v2",
        "model": model_name,
        "model_revision": _infer_model_revision(model),
        "repo_git_sha": _repo_git_sha(),
        "prompt_set": {
            "suite": resolved_suite_name,
            "count_requested": prompt_count,
            "count_after_filter": len(records),
            **filter_meta,
            "families": sorted({record.family for record in records}),
        },
        "metric_config": {
            "effect_token_policy": effect_token_policy,
            "metrics_mode": metrics_mode,
            "epsilon": epsilon,
            "batch_size": batch_size,
            "strict_head_hooks": strict_head_hooks,
            "legacy_logit_delta": "deprecated alias for selected target-logit delta mean",
        },
        "metrics": metrics,
        "rankings": _build_rankings(metrics),
        "controls_summary": {},
        "run_metadata": {
            "seed": seed,
            "device": str(input_device),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hook_debug_counters": hook_debug_counters,
        },
    }
    if save_per_prompt_effects:
        result["per_prompt_effects"] = per_prompt_effects
    return result


def _values_with_default(metrics: Sequence[Dict[str, Any]], key: str, default: float = 0.0) -> List[float]:
    values = []
    for metric in metrics:
        value = metric.get(key)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            values.append(default)
        else:
            values.append(float(value))
    return values


def _zscores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(var)
    if std <= 1e-12:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def _annotate_scores_and_rankings(metrics: List[Dict[str, Any]]) -> None:
    match_z = _zscores(_values_with_default(metrics, "match_score", 0.0))
    entropy_z = _zscores(_values_with_default(metrics, "entropy", 0.0))
    causal_values = [
        0.0 if metric.get("copy_target_prob_delta_mean") is None else float(metric["copy_target_prob_delta_mean"])
        for metric in metrics
    ]
    causal_z = _zscores(causal_values)
    for metric, mz, ez, cz in zip(metrics, match_z, entropy_z, causal_z):
        metric["composite_score"] = float(mz - ez + cz)


def _top_heads(metrics: Sequence[Dict[str, Any]], score_key: str, limit: int = 10) -> List[Dict[str, Any]]:
    def key(metric: Dict[str, Any]) -> Tuple[float, float, float]:
        value = metric.get(score_key)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            score = float("-inf")
        else:
            score = float(value)
        return (score, float(metric.get("match_score", 0.0)), -float(metric.get("entropy", 0.0)))

    ordered = sorted(metrics, key=key, reverse=True)
    out: List[Dict[str, Any]] = []
    for metric in ordered[:limit]:
        out.append(
            {
                "layer": int(metric["layer"]),
                "head": int(metric["head"]),
                score_key: metric.get(score_key),
                "match_score": metric.get("match_score"),
                "entropy": metric.get("entropy"),
            }
        )
    return out


def _build_rankings(metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "composite_top10": _top_heads(metrics, "composite_score", limit=10),
        "match_only_top10": _top_heads(metrics, "match_score", limit=10),
        "causal_only_top10": _top_heads(metrics, "copy_target_prob_delta_mean", limit=10),
    }


def _rank_map(metrics: Sequence[Dict[str, Any]], score_key: str) -> Dict[Tuple[int, int], int]:
    ordered = sorted(
        metrics,
        key=lambda m: (
            float("-inf")
            if m.get(score_key) is None
            else float(m.get(score_key)),
            float(m.get("match_score", 0.0)),
            -float(m.get("entropy", 0.0)),
        ),
        reverse=True,
    )
    return {(int(m["layer"]), int(m["head"])): idx + 1 for idx, m in enumerate(ordered)}


def _spearman_from_rank_maps(
    rank_a: Mapping[Tuple[int, int], int],
    rank_b: Mapping[Tuple[int, int], int],
    keys: Sequence[Tuple[int, int]],
) -> Optional[float]:
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


def topk_rank_stability_spearman(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
    *,
    score_key: str = "composite_score",
    top_k: int = 50,
) -> Optional[float]:
    metrics_a = result_a.get("metrics", [])
    metrics_b = result_b.get("metrics", [])
    if not metrics_a or not metrics_b:
        return None
    rank_a = _rank_map(metrics_a, score_key)
    rank_b = _rank_map(metrics_b, score_key)
    top_a = sorted(metrics_a, key=lambda m: float(m.get(score_key) or -1e30), reverse=True)[:top_k]
    top_b = sorted(metrics_b, key=lambda m: float(m.get(score_key) or -1e30), reverse=True)[:top_k]
    keys = list({(int(m["layer"]), int(m["head"])) for m in top_a + top_b})
    return _spearman_from_rank_maps(rank_a, rank_b, keys)


def aggregate_detection_runs(
    runs: Sequence[Dict[str, Any]],
    *,
    score_key: str = "composite_score",
    top_k: int = 50,
) -> Dict[str, Any]:
    if not runs:
        raise ValueError("No runs provided")
    stability: List[float] = []
    for idx in range(len(runs) - 1):
        rho = topk_rank_stability_spearman(runs[idx], runs[idx + 1], score_key=score_key, top_k=top_k)
        if rho is not None:
            stability.append(float(rho))
    annotated_runs: List[Dict[str, Any]] = []
    for run in runs:
        copied = dict(run)
        copied_metrics = []
        rho_value = None
        if stability:
            rho_value = sum(stability) / len(stability)
        for metric in run.get("metrics", []):
            metric_copy = dict(metric)
            metric_copy["rank_stability_spearman"] = rho_value
            copied_metrics.append(metric_copy)
        copied["metrics"] = copied_metrics
        annotated_runs.append(copied)
    return {
        "schema_version": "induction_detection_aggregate_v1",
        "score_key": score_key,
        "top_k": top_k,
        "rank_stability_spearman": {
            "pairwise": stability,
            "mean": (sum(stability) / len(stability)) if stability else None,
        },
        "runs": annotated_runs,
    }


def detect_induction_heads(
    model_name: str,
    cache_dir: str,
    prompt_count: int = 50,
    seed: int = 0,
    prompts: Optional[List[str]] = None,
    prompt_records: Optional[List[PromptRecord]] = None,
    prompt_suite: Optional[str] = None,
    batch_size: int = 8,
    strict_head_hooks: bool = False,
    effect_token_policy: str = "baseline_argmax",
    metrics_mode: str = "full",
    epsilon: float = 1e-4,
    save_per_prompt_effects: bool = False,
) -> Dict[str, Any]:
    model, tokenizer = load_local_model(model_name, cache_dir=cache_dir, local_files_only=True)
    return _detect_with_loaded_model(
        model,
        tokenizer,
        model_name=model_name,
        prompt_count=prompt_count,
        seed=seed,
        prompts=prompts,
        prompt_records=prompt_records,
        prompt_suite=prompt_suite,
        batch_size=batch_size,
        strict_head_hooks=strict_head_hooks,
        effect_token_policy=effect_token_policy,
        metrics_mode=metrics_mode,
        epsilon=epsilon,
        save_per_prompt_effects=save_per_prompt_effects,
    )


# Backwards-compatible alias for external callers that want a loaded-model path.
detect_induction_heads_with_model = _detect_with_loaded_model

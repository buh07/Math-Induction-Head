"""Detect induction heads using attention entropy and pattern matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .datasets import generate_prompt_batch
from .model_loader import load_local_model
from .hf_hooks import HFHookApplier


@dataclass
class HeadMetric:
    layer: int
    head: int
    entropy: float
    mean_match_score: float


def _attention_entropy(attn: torch.Tensor) -> float:
    attn = attn + 1e-12
    entropy = -(attn * attn.log()).sum(dim=-1).mean()
    return entropy.item()


def _previous_token_match(attn: torch.Tensor) -> float:
    # Look for strong attention from token i to token i-1
    if attn.size(-1) < 2:
        return 0.0
    eye = torch.zeros_like(attn)
    eye[..., 1:, :-1] = 1.0
    score = (attn * eye).sum(dim=-1).mean()
    return score.item()


def detect_induction_heads(
    model_name: str,
    cache_dir: str,
    prompt_count: int = 50,
    seed: int = 0,
) -> Dict[str, Any]:
    model, tokenizer = load_local_model(model_name, cache_dir=cache_dir, local_files_only=True)
    model.config.output_attentions = True
    prompts = generate_prompt_batch(prompt_count, seed=seed)
    device = next(model.parameters()).device
    head_metrics: List[HeadMetric] = []

    def hook_attention(module, input, output):
        attn = None
        if isinstance(output, tuple):
            # GPT-style: output = (attn_output, present, attn_weights)
            for candidate in reversed(output):
                if torch.is_tensor(candidate) and candidate.dim() == 4:
                    attn = candidate
                    break
        elif torch.is_tensor(output):
            attn = output
        if attn is None:
            return
        batch, heads, seq, _ = attn.shape
        for head in range(heads):
            attn_slice = attn[:, head, :, :].detach().mean(dim=0)
            entropy = _attention_entropy(attn_slice)
            match = _previous_token_match(attn_slice)
            head_metrics.append(
                HeadMetric(layer=module.layer_idx, head=head, entropy=entropy, mean_match_score=match)
            )

    applier = HFHookApplier(model)
    handles = []
    for idx, layer in enumerate(applier.layers):
        module = _resolve_attention_module(layer)
        if module is None:
            continue
        module.layer_idx = idx
        handles.append(module.register_forward_hook(hook_attention))

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            model(**inputs)

    for handle in handles:
        handle.remove()

    metrics = [
        {
            "layer": metric.layer,
            "head": metric.head,
            "entropy": metric.entropy,
            "match_score": metric.mean_match_score,
        }
        for metric in head_metrics
    ]
    metrics.sort(key=lambda m: (m["match_score"], -m["entropy"]), reverse=True)
    return {"model": model_name, "metrics": metrics}


def _resolve_attention_module(layer) -> Optional[torch.nn.Module]:
    for attr in ("self_attn", "attention", "attn"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None

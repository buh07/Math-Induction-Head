import math

import torch
import torch.nn as nn

from src.induction_detection import (
    _attention_entropy,
    _contextual_target_token_ids,
    _gather_last_valid_logits,
    _previous_token_mask,
    _previous_token_match,
    aggregate_detection_runs,
    detect_induction_heads,
    filter_single_token_target_records,
    generate_control_prompt_suite,
    PromptRecord,
)


def test_attention_entropy_ignores_padding_rows():
    attn = torch.tensor(
        [
            [
                [0.5, 0.5, 0.0],
                [0.1, 0.9, 0.0],
                [0.33, 0.33, 0.34],
            ]
        ]
    )
    mask = torch.tensor([[True, True, False]])
    entropy = _attention_entropy(attn, mask)
    row0 = -(0.5 * math.log(0.5) + 0.5 * math.log(0.5))
    row1 = -(0.1 * math.log(0.1) + 0.9 * math.log(0.9))
    expected = (row0 + row1) / 2.0
    assert math.isclose(entropy, expected, rel_tol=1e-5)


def test_previous_token_match_targets_single_offset():
    attn = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.1, 0.6, 0.3, 0.0],
                [0.9, 0.0, 0.1, 0.0],
            ]
        ]
    )
    mask = torch.tensor([[True, True, True, False]])
    prev_mask = _previous_token_mask(attn.shape[-1], attn.device, attn.dtype)
    score = _previous_token_match(attn, mask, prev_mask)
    expected = (0.8 + 0.6) / 2.0
    assert math.isclose(score, expected, rel_tol=1e-6)


def test_attention_entropy_sanitizes_nan_values():
    attn = torch.tensor(
        [
            [
                [float("nan"), 1.0, 0.0],
                [0.2, 0.8, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    mask = torch.tensor([[True, True, False]])
    entropy = _attention_entropy(attn, mask)
    assert math.isfinite(entropy)


def test_gather_last_valid_logits_uses_last_non_padding_position():
    logits = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    gathered = _gather_last_valid_logits(logits, attention_mask)
    assert torch.allclose(gathered, torch.tensor([[2.0, 0.0], [6.0, 0.0]]))


class _FakeTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.eos_token_id = 0
        self._prompt_map = {
            "P1": [1, 2, 1],
            "P2": [2, 1],
            "N1": [1, 3, 2],
            "N2": [3, 1],
            "P1 A": [1, 2, 1, 1],
            "P2 A": [2, 1, 1],
            "N1 B": [1, 3, 2, 2],
            "N2 B": [3, 1, 2],
        }
        self._target_map = {
            " A": [1],
            " B": [2],
            " C": [3],
            " multi": [1, 2],  # used to exercise tokenization filter
        }

    def __call__(self, text, padding=False, return_tensors=None, add_special_tokens=True):
        if isinstance(text, list):
            ids = [self._prompt_map[item] for item in text]
            max_len = max(len(seq) for seq in ids)
            padded = [seq + [self.eos_token_id] * (max_len - len(seq)) for seq in ids]
            mask = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in ids]
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(padded, dtype=torch.long),
                    "attention_mask": torch.tensor(mask, dtype=torch.long),
                }
            return {"input_ids": padded, "attention_mask": mask}
        if text in self._prompt_map:
            token_ids = self._prompt_map[text]
        else:
            token_ids = None
            for prompt, prompt_ids in sorted(self._prompt_map.items(), key=lambda kv: len(kv[0]), reverse=True):
                if text.startswith(prompt):
                    suffix = text[len(prompt) :]
                    if suffix in self._target_map:
                        token_ids = prompt_ids + self._target_map[suffix]
                        break
            if token_ids is None:
                token_ids = self._target_map[text]
        return {"input_ids": token_ids}


class _ToySelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.c_proj = nn.Identity()
        self._last_attn = None

    def forward(self, x):
        # x shape [B, S, 4]; head 0 copies dims 0:2, head 1 copies dims 2:4.
        bsz, seq, _ = x.shape
        head0 = x[..., :2]
        head1 = x[..., 2:4]
        concat = torch.cat([head0, head1], dim=-1)

        # Build simple attention maps: head0 previous-token heavy, head1 uniform.
        attn = torch.zeros(bsz, 2, seq, seq, device=x.device, dtype=x.dtype)
        for i in range(seq):
            if i == 0:
                attn[:, 0, i, i] = 1.0
            else:
                attn[:, 0, i, i - 1] = 1.0
            attn[:, 1, i, : i + 1] = 1.0 / float(i + 1)
        self._last_attn = attn
        return self.c_proj(concat)


class _ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _ToySelfAttention()
        self.mlp = nn.Identity()

    def forward(self, x):
        x = self.self_attn(x)
        return self.mlp(x)


class _ToyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer()])


class _ToyOutput:
    def __init__(self, logits, attentions):
        self.logits = logits
        self.attentions = attentions


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ToyInner()
        self.embed = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 4, bias=False)
        self.config = type("Cfg", (), {"output_attentions": False, "_commit_hash": "toysha"})()

        with torch.no_grad():
            self.embed.weight.zero_()
            # Token 1 strongly activates head0 dims.
            self.embed.weight[1] = torch.tensor([3.0, 0.0, 0.0, 0.0])
            # Token 2 strongly activates head1 dims.
            self.embed.weight[2] = torch.tensor([0.0, 0.0, 3.0, 0.0])
            self.embed.weight[3] = torch.tensor([0.5, 0.0, 0.5, 0.0])
            self.embed.weight[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])  # pad/eos
            self.lm_head.weight.zero_()
            self.lm_head.weight[1, 0] = 1.0  # token A depends on head0
            self.lm_head.weight[2, 2] = 1.0  # token B depends on head1
            self.lm_head.weight[3, 0] = 0.5
            self.lm_head.weight[3, 2] = 0.5

    def forward(self, input_ids, attention_mask, output_attentions=False, use_cache=False, return_dict=True):
        del use_cache, return_dict
        x = self.embed(input_ids)
        attentions = []
        for layer in self.model.layers:
            x = layer(x)
            attentions.append(layer.self_attn._last_attn)
        logits = self.lm_head(x)
        return _ToyOutput(logits=logits, attentions=tuple(attentions) if output_attentions else None)


def test_filter_single_token_target_records_filters_multi_token_targets():
    tokenizer = _FakeTokenizer()
    records = [
        PromptRecord(prompt="P1", family="pos", expected_next_text=" A"),
        PromptRecord(prompt="P2", family="pos", expected_next_text=" multi"),
    ]
    kept, meta = filter_single_token_target_records(tokenizer, records)
    assert len(kept) == 1
    assert kept[0].metadata["expected_next_token_id"] == 1
    assert meta["dropped_multi_token_target"] == 1
    assert meta["single_token_target_filter_rate"] == 0.5


def test_contextual_target_token_ids_uses_prompt_plus_target_delta():
    tokenizer = _FakeTokenizer()
    token_ids = _contextual_target_token_ids(tokenizer, prompt="P1", target_text=" A")
    assert token_ids == [1]


def test_generate_control_prompt_suite_includes_explicit_targets_and_families():
    records = generate_control_prompt_suite("synthetic_repeat", 9, seed=0)
    assert len(records) == 9
    assert all(record.expected_next_text is not None for record in records)
    families = {record.family for record in records}
    assert {"repeat_short", "repeat_delim", "repeat_numeric"} <= families


def test_detect_induction_heads_emits_v2_schema_and_nonzero_causal_metrics(monkeypatch):
    model = _ToyModel()
    tokenizer = _FakeTokenizer()

    def fake_load_local_model(*args, **kwargs):
        return model, tokenizer

    monkeypatch.setattr("src.induction_detection.load_local_model", fake_load_local_model)
    prompt_records = [
        PromptRecord(prompt="P1", family="repeat_short", expected_next_text=" A"),
        PromptRecord(prompt="P2", family="repeat_short", expected_next_text=" A"),
        PromptRecord(prompt="N1", family="negative_short", expected_next_text=" B"),
        PromptRecord(prompt="N2", family="negative_short", expected_next_text=" B"),
    ]
    result = detect_induction_heads(
        model_name="toy",
        cache_dir=".",
        prompt_records=prompt_records,
        prompt_suite="synthetic_repeat",
        batch_size=2,
        strict_head_hooks=True,
        effect_token_policy="explicit_copy_target",
        metrics_mode="full",
    )
    assert result["schema_version"] == "induction_detection_v2"
    assert "metric_config" in result
    assert "controls_summary" in result
    assert result["metric_config"]["effect_token_policy"] == "explicit_copy_target"
    assert any(abs(float(m.get("logit_delta", 0.0))) > 1e-6 for m in result["metrics"])
    assert any(float(m.get("next_token_kl_mean", 0.0)) > 1e-6 for m in result["metrics"])


def test_detect_induction_heads_explicit_target_policy_uses_explicit_metrics(monkeypatch):
    model = _ToyModel()
    tokenizer = _FakeTokenizer()

    monkeypatch.setattr("src.induction_detection.load_local_model", lambda *a, **k: (model, tokenizer))
    result = detect_induction_heads(
        model_name="toy",
        cache_dir=".",
        prompt_records=[PromptRecord(prompt="P1", family="repeat_short", expected_next_text=" A")],
        prompt_suite="synthetic_repeat",
        batch_size=1,
        effect_token_policy="explicit_copy_target",
        metrics_mode="full",
    )
    assert all("copy_target_prob_delta_mean" in metric for metric in result["metrics"])
    assert any(metric["copy_target_prob_delta_mean"] is not None for metric in result["metrics"])


def test_negative_control_decoy_targets_show_lower_causal_signal_than_positive(monkeypatch):
    model = _ToyModel()
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr("src.induction_detection.load_local_model", lambda *a, **k: (model, tokenizer))

    positive = detect_induction_heads(
        model_name="toy",
        cache_dir=".",
        prompt_records=[
            PromptRecord(prompt="P1", family="repeat_short", expected_next_text=" A"),
            PromptRecord(prompt="P2", family="repeat_short", expected_next_text=" A"),
        ],
        prompt_suite="synthetic_repeat",
        batch_size=2,
        effect_token_policy="explicit_copy_target",
        metrics_mode="full",
    )
    negative = detect_induction_heads(
        model_name="toy",
        cache_dir=".",
        prompt_records=[
            PromptRecord(prompt="P1", family="negative_short", expected_next_text=" B"),
            PromptRecord(prompt="P2", family="negative_short", expected_next_text=" B"),
        ],
        prompt_suite="synthetic_negative",
        batch_size=2,
        effect_token_policy="explicit_copy_target",
        metrics_mode="full",
    )
    pos_vals = sorted(float(m.get("copy_target_prob_delta_mean") or 0.0) for m in positive["metrics"])
    neg_vals = sorted(float(m.get("copy_target_prob_delta_mean") or 0.0) for m in negative["metrics"])
    pos_median = pos_vals[len(pos_vals) // 2]
    neg_median = neg_vals[len(neg_vals) // 2]
    assert pos_median >= neg_median


def test_aggregate_detection_runs_adds_rank_stability_field():
    run1 = {
        "metrics": [
            {"layer": 0, "head": 0, "composite_score": 2.0, "match_score": 0.2, "entropy": 1.0},
            {"layer": 0, "head": 1, "composite_score": 1.0, "match_score": 0.1, "entropy": 1.1},
        ]
    }
    run2 = {
        "metrics": [
            {"layer": 0, "head": 0, "composite_score": 1.5, "match_score": 0.25, "entropy": 1.0},
            {"layer": 0, "head": 1, "composite_score": 0.5, "match_score": 0.05, "entropy": 1.2},
        ]
    }
    agg = aggregate_detection_runs([run1, run2], top_k=2)
    assert agg["schema_version"] == "induction_detection_aggregate_v1"
    assert agg["rank_stability_spearman"]["mean"] is not None
    assert "rank_stability_spearman" in agg["runs"][0]["metrics"][0]

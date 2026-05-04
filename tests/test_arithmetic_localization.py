import pytest
import torch
import torch.nn as nn

from src.arithmetic_localization import (
    LocalizationConfig,
    LocalizationMetric,
    _infer_mlp_neuron_count,
    answer_token_logit_delta_mean,
    answer_token_prob_delta_mean,
    build_localization_not_implemented_result,
    build_robustness_summary,
    summarize_logits_shift,
    run_arithmetic_localization,
)
from src.operator_buckets import OperatorBucketDataset, OperatorBucketExample


def test_localization_metric_helpers_produce_nonzero_shifts():
    baseline = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    ablated = torch.tensor([[2.0, 1.5, 0.0], [0.0, 1.0, 2.0]])
    targets = torch.tensor([0, 1])
    logit_delta = answer_token_logit_delta_mean(baseline, ablated, targets)
    prob_delta = answer_token_prob_delta_mean(baseline, ablated, targets)
    summary = summarize_logits_shift(baseline, ablated, targets)
    assert float(logit_delta) != 0.0
    assert float(prob_delta) != 0.0
    assert summary["next_token_kl_mean"] > 0.0
    assert summary["logit_l1_delta_mean"] > 0.0


def test_robustness_summary_uses_same_set_shuffle_label():
    summary = build_robustness_summary(
        same_set_shuffle_invariance=1.0,
        subsample_stability=0.8,
        family_heldout_stability=0.6,
        seed_robustness=0.75,
    )
    assert summary["same_set_shuffle_invariance"] == 1.0
    assert summary["subsample_stability"] == 0.8
    assert "family_heldout_stability" in summary


def test_localization_stub_result_schema_contains_required_fields():
    config = LocalizationConfig(
        component_type="attention_heads",
        operator_filters=["addition"],
        bucket_filters=["no_carry"],
        metric_targets="answer_token",
    )
    result = build_localization_not_implemented_result(
        model="meta-llama/Meta-Llama-3-8B",
        config=config,
        prompt_set={"suite": "addition_mvp"},
        reason="stub",
    )
    assert result["schema_version"] == "operator_localization_v1"
    assert result["status"] == "not_implemented"
    assert result["metric_config"]["component_type"] == "attention_heads"
    assert "run_metadata" in result
    assert "robustness_summary" in result


class _SimpleTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 0
        self._vocab = {self.pad_token: 0}

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = len(self._vocab)
        return self._vocab[tok]

    def _encode(self, text: str) -> list[int]:
        # whitespace tokenization makes single-digit answers single-token targets
        toks = text.strip().split()
        if not toks:
            return []
        return [self._id(tok) for tok in toks]

    def __call__(self, text, padding=False, return_tensors=None, add_special_tokens=True):
        del add_special_tokens
        if isinstance(text, list):
            ids = [self._encode(t) for t in text]
            max_len = max((len(seq) for seq in ids), default=0)
            padded = [seq + [self.pad_token_id] * (max_len - len(seq)) for seq in ids]
            mask = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in ids]
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(padded, dtype=torch.long),
                    "attention_mask": torch.tensor(mask, dtype=torch.long),
                }
            return {"input_ids": padded, "attention_mask": mask}
        return {"input_ids": self._encode(str(text))}


class _AlwaysEmptyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, padding=False, return_tensors=None, add_special_tokens=True):
        del padding, add_special_tokens
        if isinstance(text, list):
            if return_tensors == "pt":
                batch = len(text)
                return {
                    "input_ids": torch.zeros((batch, 0), dtype=torch.long),
                    "attention_mask": torch.zeros((batch, 0), dtype=torch.long),
                }
            return {"input_ids": [[] for _ in text], "attention_mask": [[] for _ in text]}
        return {"input_ids": []}


class _MiniSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.c_proj = nn.Identity()

    def forward(self, x):
        # concat two head-like chunks
        h0 = x[..., :2]
        h1 = x[..., 2:4]
        return self.c_proj(torch.cat([h0, h1], dim=-1))


class _MiniMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


class _MiniLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _MiniSelfAttention()
        self.mlp = _MiniMLP()

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class _MiniInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_MiniLayer()])


class _MiniOutput:
    def __init__(self, logits):
        self.logits = logits


class _MiniModel(nn.Module):
    def __init__(self, vocab_size=64):
        super().__init__()
        torch.manual_seed(0)
        self.model = _MiniInner()
        self.embed = nn.Embedding(vocab_size, 4)
        self.lm_head = nn.Linear(4, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask, output_attentions=False, use_cache=False, return_dict=True):
        del output_attentions, use_cache, return_dict
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return _MiniOutput(logits)


def _tiny_operator_dataset():
    examples = [
        OperatorBucketExample(prompt="Compute: 1 + 2 =", expected_answer=3, operator="addition", bucket="no_carry", operands=[1, 2]),
        OperatorBucketExample(prompt="Compute: 2 + 3 =", expected_answer=5, operator="addition", bucket="no_carry", operands=[2, 3]),
        OperatorBucketExample(prompt="Compute: 4 + 5 =", expected_answer=9, operator="addition", bucket="no_carry", operands=[4, 5]),
    ]
    return {"addition__no_carry": OperatorBucketDataset(operator="addition", bucket="no_carry", examples=examples, seed=0)}


def test_run_arithmetic_localization_attention_heads_smoke_on_tiny_model():
    model = _MiniModel()
    tokenizer = _SimpleTokenizer()
    datasets = _tiny_operator_dataset()
    config = LocalizationConfig(
        component_type="attention_heads",
        operator_filters=["addition"],
        bucket_filters=["no_carry"],
        metric_targets="answer_token",
        batch_size=2,
        seed=0,
        strict_attention_heads=True,
    )
    result = run_arithmetic_localization(
        model,
        tokenizer,
        model_name="mini",
        datasets=datasets,
        config=config,
        component_options={"layer_indices": [0], "head_limit_per_layer": 2},
    )
    assert result["schema_version"] == "operator_localization_v1"
    assert result["status"] == "ok"
    assert len(result["metrics"]) == 2
    assert any(float(m["next_token_kl_mean"]) >= 0.0 for m in result["metrics"])


def test_run_arithmetic_localization_mlp_neurons_smoke_on_tiny_model():
    model = _MiniModel()
    tokenizer = _SimpleTokenizer()
    datasets = _tiny_operator_dataset()
    config = LocalizationConfig(
        component_type="mlp_neurons",
        operator_filters=["addition"],
        bucket_filters=["no_carry"],
        metric_targets="answer_token",
        batch_size=2,
        seed=0,
        strict_attention_heads=True,
    )
    result = run_arithmetic_localization(
        model,
        tokenizer,
        model_name="mini",
        datasets=datasets,
        config=config,
        component_options={"layer_indices": [0], "sample_per_layer": 3},
    )
    assert result["status"] == "ok"
    assert len(result["metrics"]) == 3


def test_run_arithmetic_localization_raises_for_empty_tokenized_inputs():
    model = _MiniModel()
    tokenizer = _AlwaysEmptyTokenizer()
    datasets = _tiny_operator_dataset()
    config = LocalizationConfig(
        component_type="attention_heads",
        operator_filters=["addition"],
        bucket_filters=["no_carry"],
        metric_targets="answer_token",
        batch_size=2,
        seed=0,
        strict_attention_heads=True,
    )
    with pytest.raises(RuntimeError, match="Tokenizer produced empty input_ids"):
        run_arithmetic_localization(
            model,
            tokenizer,
            model_name="mini",
            datasets=datasets,
            config=config,
            component_options={"layer_indices": [0], "head_limit_per_layer": 2},
        )


def test_component_sampling_seed_keeps_component_catalog_stable():
    model = _MiniModel()
    tokenizer = _SimpleTokenizer()
    datasets = _tiny_operator_dataset()
    cfg_a = LocalizationConfig(
        component_type="mlp_neurons",
        operator_filters=["addition"],
        bucket_filters=["no_carry"],
        metric_targets="answer_token",
        batch_size=2,
        seed=0,
        component_sampling_seed=123,
    )
    cfg_b = LocalizationConfig(
        component_type="mlp_neurons",
        operator_filters=["addition"],
        bucket_filters=["no_carry"],
        metric_targets="answer_token",
        batch_size=2,
        seed=99,
        component_sampling_seed=123,
    )
    kwargs = {"component_options": {"layer_indices": [0], "sample_per_layer": 3}}
    res_a = run_arithmetic_localization(model, tokenizer, model_name="mini", datasets=datasets, config=cfg_a, **kwargs)
    res_b = run_arithmetic_localization(model, tokenizer, model_name="mini", datasets=datasets, config=cfg_b, **kwargs)
    ids_a = [row["component_id"] for row in res_a["metrics"]]
    ids_b = [row["component_id"] for row in res_b["metrics"]]
    assert ids_a == ids_b
    assert res_a["component_inventory"]["component_catalog_hash"] == res_b["component_inventory"]["component_catalog_hash"]


def test_infer_mlp_neuron_count_supports_conv1d_weight_shapes():
    class _Proj:
        def __init__(self):
            self.weight = torch.zeros((16, 64))

    class _Mlp:
        def __init__(self):
            self.c_fc = _Proj()
            self.c_proj = _Proj()

    width = _infer_mlp_neuron_count(_Mlp())
    assert width == 16

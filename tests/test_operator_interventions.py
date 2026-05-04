import pytest

from src.operator_interventions import (
    InterventionCondition,
    build_cross_operator_specificity_matrix,
    build_intervention_not_implemented_result,
    run_operator_intervention_sweeps,
)
from src.operator_buckets import OperatorBucketDataset, OperatorBucketExample


def test_build_intervention_not_implemented_result_has_schema():
    result = build_intervention_not_implemented_result(
        model="meta-llama/Meta-Llama-3-8B",
        component_set_source="phase2_localization.json",
        task_buckets=["addition__no_carry"],
        reason="stub",
    )
    assert result["schema_version"] == "operator_intervention_sweep_v1"
    assert result["status"] == "not_implemented"
    assert result["task_buckets"] == ["addition__no_carry"]


def test_cross_operator_specificity_matrix_schema():
    matrix = build_cross_operator_specificity_matrix(
        rows=["add", "random"],
        cols=["addition__no_carry", "subtraction__no_borrow"],
        values={
            "add": {"addition__no_carry": {"ablation_delta": -0.2}},
            "random": {"addition__no_carry": {"ablation_delta": -0.01}},
        },
    )
    assert matrix["schema_version"] == "cross_operator_specificity_matrix_v1"
    assert matrix["rows"] == ["add", "random"]
    assert matrix["matrix"][0]["cells"]["addition__no_carry"]["ablation_delta"] == -0.2


def test_run_operator_intervention_sweeps_smoke_with_mocked_eval(monkeypatch):
    examples = [
        OperatorBucketExample(prompt="Compute: 1 + 2 =", expected_answer=3, operator="addition", bucket="no_carry", operands=[1, 2]),
        OperatorBucketExample(prompt="Compute: 2 + 3 =", expected_answer=5, operator="addition", bucket="no_carry", operands=[2, 3]),
    ]
    ds = OperatorBucketDataset(operator="addition", bucket="no_carry", examples=examples, seed=0)

    def _dataset_payload(dataset, name):
        return {
            "dataset_name": name,
            "operator": dataset.operator,
            "bucket": dataset.bucket,
            "representation_variant": dataset.representation_variant,
            "evaluation": {
                "accuracy": 0.5,
                "accuracy_all": 0.5,
                "evaluated": len(dataset.examples),
                "total": len(dataset.examples),
                "parse_rate": 1.0,
            },
            "taxonomy": {
                "schema_version": "arithmetic_error_taxonomy_summary_v1",
                "per_digit_accuracy_mean": 0.5,
                "carry_position_error_rate": None,
                "borrow_position_error_rate": None,
            },
            "per_prompt": {
                "correct": [1.0, 0.0][: len(dataset.examples)],
                "parsed_values": [ex.expected_answer for ex in dataset.examples],
            },
        }

    def _fake_eval_operator_bucket_dataset(model, tokenizer, dataset):
        del model, tokenizer
        return _dataset_payload(dataset, dataset.name)

    def _fake_eval_component_set_on_datasets(model, tokenizer, *, datasets, component_ids, scale, strict_attention_heads):
        del model, tokenizer, component_ids, scale, strict_attention_heads
        payload = {"datasets": {}}
        for name, dataset in datasets.items():
            payload["datasets"][name] = _dataset_payload(dataset, name)
        return payload

    monkeypatch.setattr("src.operator_interventions.evaluate_operator_bucket_dataset", _fake_eval_operator_bucket_dataset)
    monkeypatch.setattr("src.operator_interventions.evaluate_component_set_on_datasets", _fake_eval_component_set_on_datasets)

    result = run_operator_intervention_sweeps(
        model=object(),
        tokenizer=object(),
        model_name="mini",
        datasets={"addition__no_carry": ds},
        component_sets={"K1": {"top": ["attn_head:L0:H0"], "random_matched": ["attn_head:L0:H1"], "bottom": ["attn_head:L0:H1"]}},
        operator_target="addition",
        scales=[0.0, 1.25],
        interventions=["ablation", "amplification"],
        bootstrap_samples=20,
        seed=0,
        sanity_policy={"enabled": True, "prediction_sample_size": 2},
        primary_scales=[0.0, 1.25],
        primary_k_values=[1],
    )
    assert result["schema_version"] == "operator_intervention_sweep_v1"
    assert result["status"] == "ok"
    assert result["results"]
    assert "sanity_summary" in result
    assert "analysis" in result
    assert "primary_set_results" in result["analysis"]
    assert "directionality_checks" in result["analysis"]
    assert "multiplicity_report" in result["analysis"]
    assert result["analysis"]["primary_set_policy"]["primary_scales"] == [0.0, 1.25]
    assert result["analysis"]["primary_set_policy"]["primary_k_values"] == [1]
    first_dataset = next(iter(result["results"][0]["datasets"].values()))
    assert "prediction_samples" in first_dataset
    assert "sanity_flags" in first_dataset


def test_run_operator_intervention_sweeps_does_not_swallow_non_signature_typeerror(monkeypatch):
    examples = [
        OperatorBucketExample(prompt="Compute: 1 + 2 =", expected_answer=3, operator="addition", bucket="no_carry", operands=[1, 2]),
    ]
    ds = OperatorBucketDataset(operator="addition", bucket="no_carry", examples=examples, seed=0)

    def _raising_eval(*args, **kwargs):
        del args, kwargs
        raise TypeError("unexpected arithmetic failure")

    monkeypatch.setattr("src.operator_interventions.evaluate_operator_bucket_dataset", _raising_eval)
    with pytest.raises(TypeError, match="unexpected arithmetic failure"):
        run_operator_intervention_sweeps(
            model=object(),
            tokenizer=object(),
            model_name="mini",
            datasets={"addition__no_carry": ds},
            component_sets={"K1": {"top": ["attn_head:L0:H0"]}},
            operator_target="addition",
            scales=[0.0],
            interventions=["ablation"],
            bootstrap_samples=10,
            seed=0,
        )


def test_primary_set_results_choose_most_harmful_ablation_row(monkeypatch):
    examples = [
        OperatorBucketExample(prompt="Compute: 1 + 2 =", expected_answer=3, operator="addition", bucket="no_carry", operands=[1, 2]),
        OperatorBucketExample(prompt="Compute: 2 + 3 =", expected_answer=5, operator="addition", bucket="no_carry", operands=[2, 3]),
    ]
    ds = OperatorBucketDataset(operator="addition", bucket="no_carry", examples=examples, seed=0)

    def _dataset_payload(dataset, name, acc):
        return {
            "dataset_name": name,
            "operator": dataset.operator,
            "bucket": dataset.bucket,
            "representation_variant": dataset.representation_variant,
            "evaluation": {
                "accuracy": acc,
                "accuracy_all": acc,
                "evaluated": len(dataset.examples),
                "total": len(dataset.examples),
                "parse_rate": 1.0,
            },
            "taxonomy": {
                "schema_version": "arithmetic_error_taxonomy_summary_v1",
                "per_digit_accuracy_mean": acc,
                "carry_position_error_rate": None,
                "borrow_position_error_rate": None,
            },
            "per_prompt": {
                "correct": [1.0 if i < int(round(acc * len(dataset.examples))) else 0.0 for i in range(len(dataset.examples))],
                "parsed_values": [ex.expected_answer for ex in dataset.examples],
            },
        }

    def _fake_eval_operator_bucket_dataset(model, tokenizer, dataset):
        del model, tokenizer
        return _dataset_payload(dataset, dataset.name, acc=0.5)

    def _fake_eval_component_set_on_datasets(model, tokenizer, *, datasets, component_ids, scale, strict_attention_heads):
        del model, tokenizer, strict_attention_heads
        cid = component_ids[0]
        if cid == "top_K1":
            acc = 0.3  # weaker harm (delta_vs_random ~= -0.1)
        elif cid == "top_K2":
            acc = 0.1  # stronger harm (delta_vs_random ~= -0.3)
        elif cid.startswith("rand_"):
            acc = 0.4
        else:
            acc = 0.5
        payload = {"datasets": {}}
        for name, dataset in datasets.items():
            payload["datasets"][name] = _dataset_payload(dataset, name, acc=acc)
        return payload

    monkeypatch.setattr("src.operator_interventions.evaluate_operator_bucket_dataset", _fake_eval_operator_bucket_dataset)
    monkeypatch.setattr("src.operator_interventions.evaluate_component_set_on_datasets", _fake_eval_component_set_on_datasets)

    result = run_operator_intervention_sweeps(
        model=object(),
        tokenizer=object(),
        model_name="mini",
        datasets={"addition__no_carry": ds},
        component_sets={
            "K1": {"top": ["top_K1"], "random_matched": ["rand_K1"]},
            "K2": {"top": ["top_K2"], "random_matched": ["rand_K2"]},
        },
        operator_target="addition",
        scales=[0.0],
        interventions=["ablation"],
        bootstrap_samples=20,
        seed=0,
        sanity_policy={"enabled": True, "prediction_sample_size": 2},
    )
    best = result["analysis"]["primary_set_results"]["best_by_dataset_and_intervention"]["addition__no_carry"]["ablation"]
    assert str(best["condition"]).startswith("K2:top")
    assert float(best["mean"]) < 0.0

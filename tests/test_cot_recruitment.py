from src.cot_recruitment import (
    CoTRecruitmentComparisonConfig,
    build_paired_prompt_examples,
    build_cot_compare_not_implemented_result,
    make_direct_and_cot_prompt_pair,
    run_cot_recruitment_compare,
)
from src.operator_buckets import generate_operator_bucket_dataset


def test_make_direct_and_cot_prompt_pair_preserves_instance_metadata():
    ds = generate_operator_bucket_dataset("addition", "no_carry", count=1, seed=1)
    ex = ds.examples[0]
    pair = make_direct_and_cot_prompt_pair(ex, instance_id="ex-0")
    assert pair.instance_id == "ex-0"
    assert ex.prompt in pair.direct_prompt
    assert ex.prompt in pair.cot_prompt
    assert "#### <integer>" in pair.direct_prompt
    assert "#### <integer>" in pair.cot_prompt
    assert pair.operator == "addition"


def test_build_cot_compare_not_implemented_result_schema():
    result = build_cot_compare_not_implemented_result(
        model="meta-llama/Meta-Llama-3-8B",
        reason="stub",
    )
    assert result["schema_version"] == "cot_recruitment_compare_v1"
    assert result["status"] == "not_implemented"
    assert result["config"]["enabled"] is False
    assert "interpretation_guardrails" in result


def test_run_cot_recruitment_compare_smoke_with_mocked_eval(monkeypatch):
    ds = generate_operator_bucket_dataset("addition", "no_carry", count=2, seed=2)

    def _fake_evaluate_bundle(model, tokenizer, bundle):
        del model, tokenizer
        rows = []
        for target in bundle.answers:
            rows.append({"parsed": target, "target": target, "correct": True})
        return {
            "accuracy": 1.0,
            "accuracy_all": 1.0,
            "evaluated": len(rows),
            "total": len(rows),
            "parse_rate": 1.0,
            "results": rows,
        }

    monkeypatch.setattr("src.cot_recruitment.evaluate_bundle", _fake_evaluate_bundle)
    monkeypatch.setattr(
        "src.cot_recruitment.evaluate_component_set_on_datasets",
        lambda *args, **kwargs: {
            "datasets": {
                "direct": {"evaluation": {"accuracy_all": 1.0, "parse_rate": 1.0}},
                "cot": {"evaluation": {"accuracy_all": 1.0, "parse_rate": 1.0}},
            }
        },
    )

    result = run_cot_recruitment_compare(
        model=object(),
        tokenizer=object(),
        model_name="mini",
        datasets={"addition__no_carry": ds},
        config=CoTRecruitmentComparisonConfig(enabled=True, max_pairs=2),
        sensitivity_component_ids=["attn_head:L0:H0"],
        sensitivity_scale=0.0,
    )
    assert result["schema_version"] == "cot_recruitment_compare_v1"
    assert result["status"] == "ok"
    assert result["n_pairs"] == 2
    assert result["format_lock"]["enabled"] is True
    assert result["relaxed_parser_diagnostics"]["enabled"] is True


def test_build_paired_prompt_examples_stratified_is_order_robust():
    add_ds = generate_operator_bucket_dataset("addition", "no_carry", count=6, seed=11)
    sub_ds = generate_operator_bucket_dataset("subtraction", "no_borrow", count=6, seed=12)

    pairs_a = build_paired_prompt_examples(
        {"addition__no_carry": add_ds, "subtraction__no_borrow": sub_ds},
        max_pairs=6,
        direct_instruction="Give only the final numeric answer.",
        cot_instruction="Think step by step, then give the final answer.",
        stratify_by_dataset=True,
        sampling_seed=7,
    )
    pairs_b = build_paired_prompt_examples(
        {"subtraction__no_borrow": sub_ds, "addition__no_carry": add_ds},
        max_pairs=6,
        direct_instruction="Give only the final numeric answer.",
        cot_instruction="Think step by step, then give the final answer.",
        stratify_by_dataset=True,
        sampling_seed=7,
    )
    assert sorted(pair.instance_id for pair in pairs_a) == sorted(pair.instance_id for pair in pairs_b)


def test_build_paired_prompt_examples_stratified_remainder_is_order_robust():
    add_ds = generate_operator_bucket_dataset("addition", "no_carry", count=4, seed=21)
    sub_ds = generate_operator_bucket_dataset("subtraction", "no_borrow", count=4, seed=22)
    mul_ds = generate_operator_bucket_dataset("multiplication", "table_lookup", count=4, seed=23)

    args = dict(
        max_pairs=5,
        direct_instruction="Give only the final numeric answer.",
        cot_instruction="Think step by step, then give the final answer.",
        stratify_by_dataset=True,
        sampling_seed=11,
    )
    pairs_a = build_paired_prompt_examples(
        {
            "addition__no_carry": add_ds,
            "subtraction__no_borrow": sub_ds,
            "multiplication__table_lookup": mul_ds,
        },
        **args,
    )
    pairs_b = build_paired_prompt_examples(
        {
            "multiplication__table_lookup": mul_ds,
            "subtraction__no_borrow": sub_ds,
            "addition__no_carry": add_ds,
        },
        **args,
    )
    def _counts(pairs):
        counts = {}
        for pair in pairs:
            dataset_name = str(pair.instance_id).split(":", 1)[0]
            counts[dataset_name] = counts.get(dataset_name, 0) + 1
        return counts

    assert _counts(pairs_a) == _counts(pairs_b)


def test_build_paired_prompt_examples_weighted_allocation_tracks_dataset_size():
    add_ds = generate_operator_bucket_dataset("addition", "no_carry", count=12, seed=31)
    sub_ds = generate_operator_bucket_dataset("subtraction", "no_borrow", count=4, seed=32)

    pairs = build_paired_prompt_examples(
        {"addition__no_carry": add_ds, "subtraction__no_borrow": sub_ds},
        max_pairs=8,
        direct_instruction="Give only the final numeric answer.",
        cot_instruction="Think step by step, then give the final answer.",
        stratify_by_dataset=True,
        sampling_seed=13,
        dataset_pair_allocation="weighted_by_dataset_size",
    )
    counts = {}
    for pair in pairs:
        dataset_name = str(pair.instance_id).split(":", 1)[0]
        counts[dataset_name] = counts.get(dataset_name, 0) + 1
    # 12:4 prevalence with 8 pairs should allocate approximately 6:2.
    assert counts["addition__no_carry"] == 6
    assert counts["subtraction__no_borrow"] == 2


def test_build_paired_prompt_examples_weighted_allocation_is_order_robust():
    add_ds = generate_operator_bucket_dataset("addition", "no_carry", count=12, seed=41)
    sub_ds = generate_operator_bucket_dataset("subtraction", "no_borrow", count=4, seed=42)
    mul_ds = generate_operator_bucket_dataset("multiplication", "table_lookup", count=8, seed=43)

    args = dict(
        max_pairs=9,
        direct_instruction="Give only the final numeric answer.",
        cot_instruction="Think step by step, then give the final answer.",
        stratify_by_dataset=True,
        sampling_seed=23,
        dataset_pair_allocation="weighted_by_dataset_size",
    )
    pairs_a = build_paired_prompt_examples(
        {
            "addition__no_carry": add_ds,
            "subtraction__no_borrow": sub_ds,
            "multiplication__table_lookup": mul_ds,
        },
        **args,
    )
    pairs_b = build_paired_prompt_examples(
        {
            "multiplication__table_lookup": mul_ds,
            "subtraction__no_borrow": sub_ds,
            "addition__no_carry": add_ds,
        },
        **args,
    )
    assert sorted(pair.instance_id for pair in pairs_a) == sorted(pair.instance_id for pair in pairs_b)


def test_build_paired_prompt_examples_equal_allocation_strategy_supported():
    add_ds = generate_operator_bucket_dataset("addition", "no_carry", count=12, seed=51)
    sub_ds = generate_operator_bucket_dataset("subtraction", "no_borrow", count=4, seed=52)

    pairs = build_paired_prompt_examples(
        {"addition__no_carry": add_ds, "subtraction__no_borrow": sub_ds},
        max_pairs=8,
        direct_instruction="Give only the final numeric answer.",
        cot_instruction="Think step by step, then give the final answer.",
        stratify_by_dataset=True,
        sampling_seed=7,
        dataset_pair_allocation="equal",
    )
    counts = {}
    for pair in pairs:
        dataset_name = str(pair.instance_id).split(":", 1)[0]
        counts[dataset_name] = counts.get(dataset_name, 0) + 1
    assert counts["addition__no_carry"] == counts["subtraction__no_borrow"] == 4

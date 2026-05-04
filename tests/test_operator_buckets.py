from src.operator_buckets import (
    TABLE_LOOKUP_MAX_OPERAND,
    generate_operator_bucket_dataset,
    generate_operator_bucket_suite,
    suite_diagnostics,
)


def test_addition_bucket_generation_labels_match_requested_bucket():
    for bucket in ("no_carry", "single_carry", "cascading_carry"):
        ds = generate_operator_bucket_dataset("addition", bucket, count=12, seed=123)
        assert len(ds.examples) == 12
        assert ds.bucket == bucket
        for ex in ds.examples:
            carries = ex.carry_positions or []
            if bucket == "no_carry":
                assert len(carries) == 0
            elif bucket == "single_carry":
                assert len(carries) == 1
            else:
                assert len(carries) >= 2


def test_subtraction_bucket_generation_labels_match_requested_bucket():
    for bucket in ("no_borrow", "single_borrow", "cascading_borrow"):
        ds = generate_operator_bucket_dataset("subtraction", bucket, count=10, seed=321)
        for ex in ds.examples:
            borrows = ex.borrow_positions or []
            if bucket == "no_borrow":
                assert len(borrows) == 0
            elif bucket == "single_borrow":
                assert len(borrows) == 1
            else:
                assert len(borrows) >= 2


def test_operator_bucket_dataset_is_deterministic_by_seed():
    a = generate_operator_bucket_dataset("addition", "single_carry", count=16, seed=7)
    b = generate_operator_bucket_dataset("addition", "single_carry", count=16, seed=7)
    assert a.dataset_hash == b.dataset_hash
    assert a.prompts == b.prompts
    assert a.targets == b.targets


def test_multiplication_bucket_generation_labels_match_requested_bucket():
    for bucket in ("table_lookup", "partial_product", "carry_partial_sum"):
        ds = generate_operator_bucket_dataset("multiplication", bucket, count=10, seed=99)
        for ex in ds.examples:
            a, b = ex.operands
            if bucket == "table_lookup":
                assert a <= TABLE_LOOKUP_MAX_OPERAND and b <= TABLE_LOOKUP_MAX_OPERAND
            elif bucket == "partial_product":
                assert (a <= TABLE_LOOKUP_MAX_OPERAND) ^ (b <= TABLE_LOOKUP_MAX_OPERAND)
            else:
                assert a > TABLE_LOOKUP_MAX_OPERAND and b > TABLE_LOOKUP_MAX_OPERAND


def test_generate_operator_bucket_suite_manifest_has_counts_and_hashes():
    suite = generate_operator_bucket_suite(
        {"addition": ["no_carry", "single_carry"], "multiplication": ["table_lookup"]},
        counts_per_bucket=8,
        seed=0,
    )
    manifest = suite.to_manifest()
    assert manifest["schema_version"] == "operator_bucket_suite_v1"
    assert manifest["counts_by_bucket"]["addition__no_carry"] == 8
    assert manifest["counts_by_bucket"]["addition__single_carry"] == 8
    assert manifest["counts_by_bucket"]["multiplication__table_lookup"] == 8
    assert all(item["dataset_hash"] for item in manifest["datasets"])


def test_table_lookup_supports_high_count_generation():
    ds = generate_operator_bucket_dataset("multiplication", "table_lookup", count=256, seed=11)
    assert len(ds.examples) == 256


def test_suite_diagnostics_reports_answer_length_bounds():
    suite = generate_operator_bucket_suite(
        {
            "addition": ["single_carry"],
            "subtraction": ["single_borrow"],
            "multiplication": ["partial_product"],
        },
        counts_per_bucket=6,
        seed=3,
    )
    diagnostics = suite_diagnostics(suite)
    assert diagnostics["schema_version"] == "dataset_diagnostics_v1"
    for dataset_name, payload in diagnostics["datasets"].items():
        assert payload["count"] == 6
        assert payload["answer_length_min"] >= 1
        assert payload["answer_length_max"] >= payload["answer_length_min"], dataset_name

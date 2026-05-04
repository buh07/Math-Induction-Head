from src.arithmetic_error_taxonomy import (
    assess_prediction,
    parse_numeric_prediction,
    per_digit_accuracy,
    summarize_bucket_predictions,
)
from src.operator_buckets import generate_operator_bucket_dataset


def test_parse_numeric_prediction_handles_text_and_commas():
    assert parse_numeric_prediction("Answer: 1,234") == 1234
    assert parse_numeric_prediction("-12.0") == -12
    assert parse_numeric_prediction("no number") is None


def test_per_digit_accuracy_aligns_from_right():
    assert per_digit_accuracy(1234, 1234) == 1.0
    # 1234 vs 1204 differs at the tens place only
    assert per_digit_accuracy(1234, 1204) == 0.75
    # length mismatch is padded and scored from the right
    assert per_digit_accuracy(105, 5) == 1 / 3


def test_assess_prediction_reports_carry_position_error_rate():
    ds = generate_operator_bucket_dataset("addition", "single_carry", count=1, seed=9)
    ex = ds.examples[0]
    good = assess_prediction(ex, str(ex.expected_answer))
    bad = assess_prediction(ex, str(int(ex.expected_answer) + 1))
    assert good.parse_ok and good.is_correct
    assert good.carry_position_error_rate in (0.0, None)
    assert bad.parse_ok
    assert bad.carry_position_error_rate is None or bad.carry_position_error_rate >= 0.0


def test_summarize_bucket_predictions_separates_parse_rate_and_accuracy():
    ds = generate_operator_bucket_dataset("addition", "no_carry", count=4, seed=11)
    preds = [str(ds.targets[0]), "n/a", str(int(ds.targets[2]) + 1), str(ds.targets[3])]
    summary = summarize_bucket_predictions(ds.examples, preds)
    assert summary["schema_version"] == "arithmetic_error_taxonomy_summary_v1"
    assert summary["n_examples"] == 4
    assert summary["parse_rate"] == 0.75
    assert summary["accuracy_all"] == 0.5
    assert summary["accuracy_parsed"] == (2 / 3)
    assert summary["per_digit_accuracy_mean"] is not None

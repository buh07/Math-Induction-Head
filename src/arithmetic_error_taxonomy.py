"""Error taxonomy helpers for Phase 2 arithmetic operator-bucket experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .operator_buckets import OperatorBucketExample


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class PredictionAssessment:
    parsed: Optional[int | float]
    is_correct: bool
    parse_ok: bool
    digit_accuracy: Optional[float]
    carry_position_error_rate: Optional[float]
    borrow_position_error_rate: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parsed": self.parsed,
            "is_correct": self.is_correct,
            "parse_ok": self.parse_ok,
            "digit_accuracy": self.digit_accuracy,
            "carry_position_error_rate": self.carry_position_error_rate,
            "borrow_position_error_rate": self.borrow_position_error_rate,
        }


def parse_numeric_prediction(text: str | int | float | None) -> Optional[int | float]:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return text
    raw = str(text).strip().replace(",", "")
    if not raw:
        return None
    match = _NUMERIC_RE.search(raw)
    if not match:
        return None
    token = match.group(0)
    try:
        if "." in token:
            value = float(token)
            if abs(value - round(value)) < 1e-9:
                return int(round(value))
            return value
        return int(token)
    except ValueError:
        return None


def _digits(value: int | float) -> Optional[List[int]]:
    if not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and (not math.isfinite(value) or abs(value - round(value)) > 1e-9):
        return None
    ivalue = int(round(float(value)))
    return [int(ch) for ch in str(abs(ivalue))]


def per_digit_accuracy(expected: int | float, predicted: Optional[int | float]) -> Optional[float]:
    expected_digits = _digits(expected)
    predicted_digits = _digits(predicted) if predicted is not None else None
    if expected_digits is None or predicted_digits is None:
        return None
    max_len = max(len(expected_digits), len(predicted_digits))
    exp_rev = list(reversed(expected_digits)) + [None] * (max_len - len(expected_digits))
    pred_rev = list(reversed(predicted_digits)) + [None] * (max_len - len(predicted_digits))
    correct = sum(1 for e, p in zip(exp_rev, pred_rev) if e == p)
    return correct / max_len if max_len else 1.0


def _position_error_rate(
    expected: int | float,
    predicted: Optional[int | float],
    positions_from_right: Optional[Sequence[int]],
) -> Optional[float]:
    if not positions_from_right:
        return None
    expected_digits = _digits(expected)
    predicted_digits = _digits(predicted) if predicted is not None else None
    if expected_digits is None or predicted_digits is None:
        return 1.0
    exp_rev = list(reversed(expected_digits))
    pred_rev = list(reversed(predicted_digits))
    errors = 0
    total = 0
    for pos in positions_from_right:
        total += 1
        e = exp_rev[pos] if pos < len(exp_rev) else None
        p = pred_rev[pos] if pos < len(pred_rev) else None
        if e != p:
            errors += 1
    return errors / total if total else None


def assess_prediction(example: OperatorBucketExample, prediction_text: str | int | float | None) -> PredictionAssessment:
    parsed = parse_numeric_prediction(prediction_text)
    parse_ok = parsed is not None
    expected = example.expected_answer
    is_correct = bool(parse_ok and parsed == expected)
    digit_acc = per_digit_accuracy(expected, parsed)
    carry_err = _position_error_rate(expected, parsed, example.carry_positions)
    borrow_err = _position_error_rate(expected, parsed, example.borrow_positions)
    return PredictionAssessment(
        parsed=parsed,
        is_correct=is_correct,
        parse_ok=parse_ok,
        digit_accuracy=digit_acc,
        carry_position_error_rate=carry_err,
        borrow_position_error_rate=borrow_err,
    )


def summarize_bucket_predictions(
    examples: Sequence[OperatorBucketExample],
    predictions: Sequence[str | int | float | None],
) -> Dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions length mismatch")
    assessments = [assess_prediction(ex, pred) for ex, pred in zip(examples, predictions)]
    total = len(assessments)
    parsed = [a for a in assessments if a.parse_ok]
    correct = [a for a in assessments if a.is_correct]
    digit_acc_values = [a.digit_accuracy for a in assessments if a.digit_accuracy is not None]
    carry_err_values = [a.carry_position_error_rate for a in assessments if a.carry_position_error_rate is not None]
    borrow_err_values = [a.borrow_position_error_rate for a in assessments if a.borrow_position_error_rate is not None]
    operators = sorted({ex.operator for ex in examples})
    buckets = sorted({ex.bucket for ex in examples})

    return {
        "schema_version": "arithmetic_error_taxonomy_summary_v1",
        "n_examples": total,
        "operators": operators,
        "buckets": buckets,
        "accuracy_all": (len(correct) / total) if total else 0.0,
        "accuracy_parsed": (len(correct) / len(parsed)) if parsed else 0.0,
        "parse_rate": (len(parsed) / total) if total else 0.0,
        "per_digit_accuracy_mean": (sum(digit_acc_values) / len(digit_acc_values)) if digit_acc_values else None,
        "carry_position_error_rate": (sum(carry_err_values) / len(carry_err_values)) if carry_err_values else None,
        "borrow_position_error_rate": (sum(borrow_err_values) / len(borrow_err_values)) if borrow_err_values else None,
        "assessment_counts": {
            "parsed": len(parsed),
            "correct": len(correct),
            "with_carry_positions": len(carry_err_values),
            "with_borrow_positions": len(borrow_err_values),
        },
    }


__all__ = [
    "PredictionAssessment",
    "parse_numeric_prediction",
    "per_digit_accuracy",
    "assess_prediction",
    "summarize_bucket_predictions",
]

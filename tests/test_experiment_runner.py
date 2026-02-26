from src.datasets import DatasetBundle
from src.experiment_runner import _extract_int, evaluate_bundle


def test_extract_int_handles_commas_and_prefers_first_line_answer():
    text = "1,302\nThe number 1,302 is composite."
    assert _extract_int(text) == 1302


def test_extract_int_handles_fraction_and_negative():
    assert abs(_extract_int("Answer: 3/2") - 1.5) < 1e-9
    assert _extract_int("-48\n-48 - 97 = -145") == -48


def test_evaluate_bundle_reports_parse_rate_and_accuracy_all(monkeypatch):
    outputs = iter(["10", "No numeric answer here", "1,302"])

    def fake_generate(_model, _tokenizer, _prompt, max_new_tokens=16):
        return next(outputs)

    monkeypatch.setattr("src.experiment_runner._generate_answer", fake_generate)
    bundle = DatasetBundle(
        name="demo",
        prompts=["a", "b", "c"],
        answers=[10, 5, 1302],
    )
    result = evaluate_bundle(object(), object(), bundle)
    assert result["evaluated"] == 2
    assert result["total"] == 3
    assert abs(result["parse_rate"] - (2 / 3)) < 1e-9
    assert result["accuracy"] == 1.0
    assert abs(result["accuracy_all"] - (2 / 3)) < 1e-9

import pytest

from src.datasets import DatasetBundle
from src.experiment_runner import _extract_int, _extract_int_strict_final, evaluate_bundle


def test_extract_int_handles_commas_and_prefers_first_line_answer():
    text = "1,302\nThe number 1,302 is composite."
    assert _extract_int(text) == 1302


def test_extract_int_handles_fraction_and_negative():
    assert abs(_extract_int("Answer: 3/2") - 1.5) < 1e-9
    assert _extract_int("-48\n-48 - 97 = -145") == -48


def test_extract_int_strict_final_rejects_ambiguous_multi_number_text():
    text = "We tried 12 then 15 then 18."
    assert _extract_int(text) == 18
    assert _extract_int_strict_final(text) is None
    assert _extract_int_strict_final("Final answer: 18") == 18


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


def test_evaluate_bundle_does_not_swallow_non_signature_typeerror(monkeypatch):
    def fake_generate(_model, _tokenizer, _prompt, max_new_tokens=16):
        del _model, _tokenizer, _prompt, max_new_tokens
        raise TypeError("boom")

    monkeypatch.setattr("src.experiment_runner._generate_answer", fake_generate)
    bundle = DatasetBundle(name="demo", prompts=["a"], answers=[1])
    with pytest.raises(TypeError, match="boom"):
        evaluate_bundle(object(), object(), bundle)


def test_evaluate_bundle_batch_mode_uses_batched_generation(monkeypatch):
    def fake_batch_generate(*_args, **_kwargs):
        prompts = _kwargs.get("prompts") if "prompts" in _kwargs else _args[2]
        return [(str(9 + idx), {"sampling_fallback_used": False, "empty_generation": False}) for idx, _ in enumerate(prompts)]

    monkeypatch.setattr("src.experiment_runner._generate_answers_batch", fake_batch_generate)
    bundle = DatasetBundle(name="demo", prompts=["a", "b"], answers=[9, 9])
    result = evaluate_bundle(object(), object(), bundle, parse_mode="strict_final_numeric", batch_size=2)
    assert result["total"] == 2
    assert result["evaluated"] == 2
    assert result["accuracy_all"] == 0.5
    assert result["generation_policy"]["batch_size"] == 2

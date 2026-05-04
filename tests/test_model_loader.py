import sys
import types

import pytest

from src.model_loader import _tokenizer_is_usable, load_local_model


class _DummyModel:
    def eval(self):
        return self


class _BadTokenizer:
    vocab_size = 0
    pad_token = None
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return {"input_ids": []}


class _GoodTokenizer:
    vocab_size = 128
    pad_token = None
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return {"input_ids": [1]}


def test_tokenizer_health_check_flags_empty_tokenization():
    ok, reason = _tokenizer_is_usable(_BadTokenizer())
    assert ok is False
    assert reason in {"invalid_vocab_size:0", "empty_tokenization_for_sentinel"}


def test_tokenizer_health_check_accepts_nonempty_tokenization():
    ok, reason = _tokenizer_is_usable(_GoodTokenizer())
    assert ok is True
    assert reason == "ok"


def test_load_local_model_retries_tokenizer_with_remote_fallback(monkeypatch, tmp_path):
    calls = []
    tokenizer_instances = [_BadTokenizer(), _GoodTokenizer()]

    def _fake_model_from_pretrained(*args, **kwargs):
        calls.append(("model", kwargs.get("local_files_only")))
        return _DummyModel()

    def _fake_tokenizer_from_pretrained(*args, **kwargs):
        calls.append(("tokenizer", kwargs.get("local_files_only")))
        return tokenizer_instances.pop(0)

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=_fake_model_from_pretrained),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=_fake_tokenizer_from_pretrained),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    _model, tokenizer = load_local_model(
        "gpt2",
        cache_dir=str(tmp_path),
        local_files_only=True,
        allow_tokenizer_download_fallback=True,
    )
    assert isinstance(tokenizer, _GoodTokenizer)
    tokenizer_calls = [row for row in calls if row[0] == "tokenizer"]
    assert tokenizer_calls == [("tokenizer", True), ("tokenizer", False)]


def test_load_local_model_raises_when_tokenizer_invalid_and_no_fallback(monkeypatch, tmp_path):
    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: _DummyModel()),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: _BadTokenizer()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="Tokenizer for 'gpt2' is unusable"):
        load_local_model(
            "gpt2",
            cache_dir=str(tmp_path),
            local_files_only=True,
            allow_tokenizer_download_fallback=False,
        )

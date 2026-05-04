"""Load Hugging Face causal language models from a local cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch


def _tokenizer_is_usable(tokenizer) -> tuple[bool, str]:
    """Return whether a tokenizer can produce non-empty token ids."""
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size <= 0:
        return False, f"invalid_vocab_size:{vocab_size}"
    try:
        encoded = tokenizer("hello", add_special_tokens=False)
    except Exception as exc:
        return False, f"encode_error:{exc}"
    token_ids: Any = None
    if isinstance(encoded, Mapping):
        token_ids = encoded.get("input_ids")
    elif hasattr(encoded, "input_ids"):
        token_ids = getattr(encoded, "input_ids")
    if token_ids is None:
        return False, "missing_input_ids_for_sentinel"
    if hasattr(token_ids, "numel"):
        if int(token_ids.numel()) <= 0:
            return False, "empty_tokenization_for_sentinel"
    elif isinstance(token_ids, (list, tuple)):
        if len(token_ids) <= 0:
            return False, "empty_tokenization_for_sentinel"
    else:
        return False, "empty_tokenization_for_sentinel"
    return True, "ok"


def load_local_model(
    model_name: str,
    cache_dir: str,
    local_files_only: bool = True,
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    allow_tokenizer_download_fallback: bool = True,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        import accelerate  # noqa: F401
        has_accelerate = True
    except ImportError:  # pragma: no cover - optional dependency
        has_accelerate = False

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    model_load_path = model_path or model_name
    model_kwargs = {
        "local_files_only": local_files_only,
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if model_path is None:
        model_kwargs["cache_dir"] = str(cache_path)
    model_kwargs["attn_implementation"] = "eager"
    if torch.cuda.is_available() and has_accelerate:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_load_path, **model_kwargs)

    tokenizer_load_path = tokenizer_path or model_load_path
    tokenizer_kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    if tokenizer_path is None:
        tokenizer_kwargs["cache_dir"] = str(cache_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, **tokenizer_kwargs)
    tokenizer_ok, tokenizer_reason = _tokenizer_is_usable(tokenizer)
    if (
        not tokenizer_ok
        and local_files_only
        and allow_tokenizer_download_fallback
    ):
        # Some local cache snapshots can be missing GPT-style tokenizer vocab/merges.
        # Retry with remote lookup enabled to repair tokenizer assets in cache.
        fallback_kwargs = dict(tokenizer_kwargs)
        fallback_kwargs["local_files_only"] = False
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, **fallback_kwargs)
        tokenizer_ok, tokenizer_reason = _tokenizer_is_usable(tokenizer)
    if not tokenizer_ok:
        raise RuntimeError(
            f"Tokenizer for '{model_name}' is unusable ({tokenizer_reason}). "
            "If this model should run offline, refresh tokenizer assets in cache."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

"""Load Hugging Face causal language models from a local cache."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch


def load_local_model(
    model_name: str,
    cache_dir: str,
    local_files_only: bool = True,
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    model_load_path = model_path or model_name
    model_kwargs = {
        "local_files_only": local_files_only,
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if model_path is None:
        model_kwargs["cache_dir"] = str(cache_path)
    model_kwargs["attn_implementation"] = "eager"
    if torch.cuda.is_available():
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

"""Shared model introspection utilities used across hook/localization modules."""

from __future__ import annotations

from typing import Optional


def locate_layers(model):
    """Return the transformer block stack for supported model families."""
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(model, attr):
            container = getattr(model, attr)
            if hasattr(container, "layers"):
                return container.layers
            if hasattr(container, "h"):
                return container.h
    raise AttributeError("Could not locate layer stack on model")


def get_attention_module(layer):
    """Return the attention module for a transformer block, if present."""
    for attr in ("self_attn", "attention", "attn"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


def get_mlp_module(layer):
    """Return the feed-forward/MLP module for a transformer block, if present."""
    for attr in ("mlp", "feed_forward", "ffn", "parallel_attn"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


def infer_head_count(attn_module) -> Optional[int]:
    """Infer number of attention heads from module/config attributes."""
    if attn_module is None:
        return None
    for attr in ("num_heads", "num_attention_heads", "n_head", "n_heads"):
        value = getattr(attn_module, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    cfg = getattr(attn_module, "config", None)
    if cfg is not None:
        for attr in ("num_attention_heads", "num_heads", "n_head", "n_heads"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    head_dim = getattr(attn_module, "head_dim", None)
    if isinstance(head_dim, int) and head_dim > 0:
        for proj_attr, dim_attr in (("o_proj", "in_features"), ("q_proj", "out_features")):
            proj = getattr(attn_module, proj_attr, None)
            dim = getattr(proj, dim_attr, None) if proj is not None else None
            if isinstance(dim, int) and dim > 0 and dim % head_dim == 0:
                return dim // head_dim
    return None


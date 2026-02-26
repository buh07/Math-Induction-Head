import pytest
import torch
import torch.nn as nn

from src.hf_hooks import apply_hooks, HFHookApplier
from src.hooks import AttentionHookConfig, NeuronHookConfig


class DummySelfAttention(nn.Module):
    def __init__(self, features=4, heads=2):
        super().__init__()
        self.in_proj = nn.Linear(features, features, bias=False)
        self.c_proj = nn.Identity()
        self.num_heads = heads

    def forward(self, x):
        return self.c_proj(self.in_proj(x))


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummySelfAttention()
        self.mlp = nn.Linear(4, 4)

    def forward(self, x):
        x = self.self_attn(x)
        return self.mlp(x)


class DummyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer()])


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyInner()

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class DummyProjectedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Identity()
        self.c_proj = nn.Identity()

    def forward(self, x):
        return self.c_proj(self.expand(x))


class DummyLayerWithProjectedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummySelfAttention()
        self.mlp = DummyProjectedMLP()

    def forward(self, x):
        x = self.self_attn(x)
        return self.mlp(x)


class DummyModelWithProjectedMLP(DummyModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = DummyInner()
        self.model.layers = nn.ModuleList([DummyLayerWithProjectedMLP()])


class DummySelfAttentionUnknownHeads(DummySelfAttention):
    def __init__(self, features=4):
        super().__init__(features=features, heads=2)
        delattr(self, "num_heads")


class DummyLayerUnknownHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummySelfAttentionUnknownHeads()
        self.mlp = nn.Linear(4, 4)

    def forward(self, x):
        x = self.self_attn(x)
        return self.mlp(x)


class DummyModelUnknownHeads(DummyModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = DummyInner()
        self.model.layers = nn.ModuleList([DummyLayerUnknownHeads()])


class DummySelfAttentionConfigHeads(nn.Module):
    def __init__(self, features=4, heads=2):
        super().__init__()
        self.c_proj = nn.Identity()
        self.o_proj = nn.Linear(features, features, bias=False)
        self.head_dim = features // heads
        self.config = type("Cfg", (), {"num_attention_heads": heads})()

    def forward(self, x):
        return self.c_proj(self.o_proj(x))


class DummyLayerConfigHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummySelfAttentionConfigHeads()
        self.mlp = nn.Linear(4, 4)

    def forward(self, x):
        x = self.self_attn(x)
        return self.mlp(x)


class DummyModelConfigHeads(DummyModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = DummyInner()
        self.model.layers = nn.ModuleList([DummyLayerConfigHeads()])


def test_attention_hook_scales_output():
    model = DummyModel()
    applier = HFHookApplier(model)
    applier.apply_attention_hooks([AttentionHookConfig(layer=0, head=None, scale=0.5)])
    x = torch.ones(1, 4)
    with torch.no_grad():
        out_full = model.model.layers[0].self_attn(x)
    applier.clear()
    with torch.no_grad():
        out_scaled = model.model.layers[0].self_attn(x)
    assert torch.allclose(out_full, out_scaled * 0.5, atol=1e-5)


def test_neuron_hook_scales_output():
    model = DummyModel()
    x = torch.ones(1, 4)
    with torch.no_grad():
        base = model.model.layers[0].mlp(x)
    with apply_hooks(
        model,
        neuron_configs=[NeuronHookConfig(layer=0, neuron_index=0, scale=0.0)],
    ):
        out = model.model.layers[0].mlp(x)
    assert torch.allclose(out[..., 0], torch.zeros_like(out[..., 0]))
    assert torch.allclose(out[..., 1:], base[..., 1:])


def test_attention_hook_scales_specific_head_and_downscales_others():
    model = DummyModel()
    x = torch.ones(1, 4)
    base = model.model.layers[0].self_attn(x).view(1, 2, 2)
    configs = [
        AttentionHookConfig(layer=0, head=None, scale=0.75),
        AttentionHookConfig(layer=0, head=1, scale=2.0, downscale_others=0.5),
    ]
    with apply_hooks(model, attention_configs=configs):
        out = model.model.layers[0].self_attn(x)
    reshaped = out.view(1, 2, 2)
    assert torch.allclose(reshaped[..., 0, :], base[..., 0, :] * 0.375)
    assert torch.allclose(reshaped[..., 1, :], base[..., 1, :] * 1.5)


def test_neuron_hook_targets_projection_input_dimension_when_available():
    model = DummyModelWithProjectedMLP()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    base = model.model.layers[0].mlp(x)
    with apply_hooks(
        model,
        neuron_configs=[NeuronHookConfig(layer=0, neuron_index=2, scale=0.0)],
    ):
        out = model.model.layers[0].mlp(x)
    expected = base.clone()
    expected[..., 2] = 0.0
    assert torch.allclose(out, expected)


def test_attention_hook_strict_mode_raises_on_unknown_head_decomposition():
    model = DummyModelUnknownHeads()
    x = torch.ones(1, 4)
    with pytest.raises(RuntimeError):
        with apply_hooks(
            model,
            attention_configs=[AttentionHookConfig(layer=0, head=0, scale=0.0)],
            strict_attention_heads=True,
        ):
            _ = model.model.layers[0].self_attn(x)


def test_attention_hook_debug_counter_tracks_targeted_preproj_calls():
    model = DummyModel()
    x = torch.ones(1, 4)
    counters = {"attention_targeted_pre_proj": {}}
    with apply_hooks(
        model,
        attention_configs=[AttentionHookConfig(layer=0, head=1, scale=0.0)],
        hook_debug_counters=counters,
    ):
        _ = model.model.layers[0].self_attn(x)
    assert counters["attention_targeted_pre_proj"].get(0, 0) >= 1


def test_attention_hook_infers_head_count_from_module_config():
    model = DummyModelConfigHeads()
    x = torch.ones(1, 4)
    with apply_hooks(
        model,
        attention_configs=[AttentionHookConfig(layer=0, head=1, scale=0.0)],
        strict_attention_heads=True,
    ):
        out = model.model.layers[0].self_attn(x)
    assert out.shape == x.shape

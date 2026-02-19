import torch
import torch.nn as nn

from src.hf_hooks import apply_hooks, HFHookApplier
from src.hooks import AttentionHookConfig, NeuronHookConfig


class DummySelfAttention(nn.Linear):
    def __init__(self, features=4, heads=2):
        super().__init__(features, features)
        self.num_heads = heads


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
    with apply_hooks(
        model,
        neuron_configs=[NeuronHookConfig(layer=0, neuron_index=0, scale=0.0)],
    ):
        x = torch.ones(1, 4)
        out = model.model.layers[0].mlp(x)
    assert torch.allclose(out, torch.zeros_like(out))


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

from src.hooks import AttentionHookConfig, HookManager, NeuronHookConfig


def test_attention_hook_scales_target_head():
    cfg = AttentionHookConfig(layer=1, head=0, scale=0.5)
    updated = cfg.apply([2.0, 4.0])
    assert updated == [1.0, 4.0]


def test_neuron_hook_scales_target_neuron():
    cfg = NeuronHookConfig(layer=3, neuron_index=1, scale=2.0)
    updated = cfg.apply([1.0, 3.0])
    assert updated == [1.0, 6.0]


def test_hook_manager_applies_hooks_for_matching_layers():
    manager = HookManager(
        attention_hooks=[AttentionHookConfig(layer=2, head=1, scale=0.1)],
        neuron_hooks=[NeuronHookConfig(layer=5, neuron_index=0, scale=0.0)],
    )
    attn = manager.apply_attention(2, [10.0, 20.0])
    neurons = manager.apply_neurons(5, [7.5, 3.0])
    assert attn == [10.0, 2.0]
    assert neurons == [0.0, 3.0]


def test_attention_hook_downscale_others():
    manager = HookManager(
        attention_hooks=[
            AttentionHookConfig(layer=0, head=0, scale=2.0, downscale_others=0.5),
            AttentionHookConfig(layer=0, head=1, scale=1.5, downscale_others=0.5),
        ]
    )
    values = manager.apply_attention(0, [1.0, 1.0, 1.0])
    assert values == [2.0, 1.5, 0.5]


def test_attention_hook_combines_layer_and_head_scales():
    manager = HookManager(
        attention_hooks=[
            AttentionHookConfig(layer=1, head=None, scale=0.8),
            AttentionHookConfig(layer=1, head=0, scale=1.5, downscale_others=0.6),
        ]
    )
    values = manager.apply_attention(1, [10.0, 10.0])
    # default: 10 * 0.8 * 0.6 = 4.8
    assert values == [12.0, 4.8]

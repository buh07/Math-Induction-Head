from scripts.phase2.run_operator_bottleneck_suite import (
    _sanitize_attention_component_ids_for_model,
    _sanitize_induction_baseline_sets_for_model,
)


class _DummyAttention:
    num_heads = 8


class _DummyLayer:
    def __init__(self):
        self.self_attn = _DummyAttention()


class _DummyModelContainer:
    def __init__(self, n_layers: int):
        self.layers = [_DummyLayer() for _ in range(n_layers)]


class _DummyModel:
    def __init__(self, n_layers: int):
        self.model = _DummyModelContainer(n_layers)


def test_sanitize_attention_component_ids_for_model_drops_invalid_layer_and_head():
    model = _DummyModel(n_layers=2)
    component_ids = [
        "attn_head:L0:H0",
        "attn_head:L1:H7",
        "attn_head:L2:H0",
        "attn_head:L1:H8",
        "mlp_neuron:L0:N3",
    ]
    valid, dropped = _sanitize_attention_component_ids_for_model(model, component_ids)

    assert "attn_head:L0:H0" in valid
    assert "attn_head:L1:H7" in valid
    assert "mlp_neuron:L0:N3" in valid
    assert "attn_head:L2:H0" in dropped
    assert "attn_head:L1:H8" in dropped


def test_sanitize_induction_baseline_sets_for_model_removes_empty_sets():
    model = _DummyModel(n_layers=1)
    warnings = []
    cleaned = _sanitize_induction_baseline_sets_for_model(
        model,
        {
            "induction_top": ["attn_head:L0:H1", "attn_head:L0:H9"],
            "induction_bottom": ["attn_head:L3:H0"],
        },
        scope_warnings=warnings,
    )
    assert cleaned is not None
    assert cleaned["induction_top"] == ["attn_head:L0:H1"]
    assert "induction_bottom" not in cleaned
    assert warnings

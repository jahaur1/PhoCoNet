"""Structural and gradient checks for PhoCoNet."""

from types import SimpleNamespace

import torch

from ts_benchmark.baselines.phoconet.models.phoconet import Model


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        revin=1,
        enc_in=5,
        period=6,
        seq_len=24,
        horizon=4,
        d_model=16,
        d_ff=32,
        n_heads=4,
        attn_dropout=0.0,
        dropout=0.0,
        stable_len=2,
        activation="gelu",
        attn_mode="none",
        fusion_mode="dual",
        alpha_mode="learnable",
        local_gate_mode="adaptive",
        local_gate_fixed=0.5,
        global_gate_mode="adaptive",
        global_gate_fixed=0.5,
        ia_layers=0,
        ca_layers=0,
        layer_order="int_coint",
        series_dim=1,
        use_future_exog=True,
        use_history_target=True,
        use_history_exog=True,
        infer_use_future=True,
        future_exog_mode="normal",
        exog_drop=[],
        q_hidden=16,
        q_gate_bias=-2.0,
        q_mode="aligned",
        q_variable_attention=False,
        q_aligned_variable_attention=False,
        q_aligned_variable_residual=False,
        q_aligned_variable_modulation=True,
        q_aligned_modulation_dim=16,
        q_aligned_modulation_hidden=16,
        q_aligned_modulation_max=0.25,
        q_aligned_modulation_output_scale=0.01,
        late_aggregation="attention",
        late_dim=16,
        late_heads=4,
        late_hidden=16,
        late_output_scale=1e-2,
        late_gate_init=0.1,
    )


def test_position_free_correction_dimensions() -> None:
    model = Model(make_config())
    expected = 2 * 4 + 5
    assert model.q_feature_norm.normalized_shape == (expected,)
    assert model.q_delta[0].in_features == expected
    assert model.q_gate[0].in_features == expected


def test_forward_shape_and_gradients() -> None:
    torch.manual_seed(211)
    model = Model(make_config())
    model.init_fusion_components(exog_dim=4, alpha_init=0.0)
    history = torch.randn(2, 24, 5)
    marks = torch.randn(2, 24, 6)
    current = torch.randn(2, 4, 4)
    output = model(history, marks, current)
    assert output.shape == (2, 4, 5)
    output.square().mean().backward()
    assert model.q_delta[0].weight.grad is not None
    assert model.q_gate[0].weight.grad is not None
    assert model.aligned_q_variable_modulation.score.weight.grad is not None

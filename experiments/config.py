"""Shared protocol for the PhoCoNet experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MODEL_NAME = "phoconet.PhoCoNet"
MODEL_LABEL = "PhoCoNet"
DATASETS = ("juzizhou", "sanjiaozhou", "laodaohe")
HORIZONS = (4, 8, 12, 16, 24)
SEEDS = (2021, 2022, 2023, 2024, 2025)
TARGET_CHANNEL = 7
DATASET_ALPHA_INIT = {
    "juzizhou": 0.0,
    "sanjiaozhou": -0.5,
    "laodaohe": 3.0,
}

BASE_PARAMS: dict[str, Any] = {
    "batch_size": 64,
    "seq_len": 96,
    "period": 6,
    "d_model": 128,
    "d_ff": 128,
    "n_heads": 4,
    "dropout": 0.0,
    "attn_dropout": 0.15,
    "activation": "gelu",
    "stable_len": 2,
    "revin": 1,
    "ia_layers": 0,
    "ca_layers": 0,
    "attn_mode": "none",
    "layer_order": "int_coint",
    "fusion_mode": "dual",
    "alpha_mode": "learnable",
    "local_gate_mode": "adaptive",
    "global_gate_mode": "adaptive",
    "use_future_exog": True,
    "use_history_target": True,
    "use_history_exog": True,
    "infer_use_future": True,
    "future_exog_mode": "normal",
    "exog_drop": [],
    "q_hidden": 32,
    "q_gate_bias": -2.0,
    "q_mode": "aligned",
    "q_variable_attention": False,
    "q_aligned_variable_attention": False,
    "q_aligned_variable_residual": False,
    "q_aligned_variable_modulation": True,
    "q_aligned_modulation_dim": 32,
    "q_aligned_modulation_hidden": 32,
    "q_aligned_modulation_max": 0.25,
    "q_aligned_modulation_output_scale": 0.01,
    "late_aggregation": "attention",
    "late_dim": 32,
    "late_heads": 4,
    "late_hidden": 32,
    "late_output_scale": 1e-2,
    "late_gate_init": 0.1,
    "num_epochs": 100,
    "patience": 10,
    "lradj": "type1",
    "lr": 0.001,
    "loss": "MSE",
}


def params_for(dataset: str, horizon: int, epochs: int = 100) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported horizon: {horizon}")
    params = {
        **BASE_PARAMS,
        "horizon": horizon,
        "alpha_init": DATASET_ALPHA_INIT[dataset],
        "num_epochs": epochs,
    }
    validate_params(params)
    return params


def validate_params(params: Mapping[str, Any]) -> None:
    expected = {
        "period": 6,
        "attn_mode": "none",
        "ia_layers": 0,
        "ca_layers": 0,
        "alpha_mode": "learnable",
        "q_aligned_variable_modulation": True,
        "q_aligned_variable_attention": False,
        "q_aligned_variable_residual": False,
        "q_variable_attention": False,
        "loss": "MSE",
    }
    for key, value in expected.items():
        if params.get(key) != value:
            raise ValueError(
                f"PhoCoNet requires {key}={value!r}; "
                f"got {params.get(key)!r}"
            )

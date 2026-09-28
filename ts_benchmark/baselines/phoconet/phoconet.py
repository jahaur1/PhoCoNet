"""Benchmark wrapper for PhoCoNet."""

import torch.nn as nn
from torch import optim

from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)
from ts_benchmark.baselines.phoconet.models.phoconet import Model


MODEL_HYPER_PARAMS = {
    "seq_len": 96,
    "horizon": 24,
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
    "batch_size": 64,
    "lr": 0.001,
    "lradj": "type1",
    "loss": "MSE",
}


class PhoCoNet(DeepForecastingModelBase):
    """PhoCoNet for block soft sensing with delayed laboratory assays."""

    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "PhoCoNet"

    def _adjust_lr(self, optimizer, epoch, config):
        if config.lradj == "type1":
            schedule = {epoch: config.lr * (0.5 ** (epoch - 1))}
        elif config.lradj == "type3":
            schedule = {
                epoch: config.lr
                if epoch < 3
                else config.lr * (0.9 ** (epoch - 3))
            }
        else:
            schedule = {}
        if epoch in schedule:
            for group in optimizer.param_groups:
                group["lr"] = schedule[epoch]

    def _init_criterion(self):
        return nn.MSELoss() if self.config.loss == "MSE" else nn.L1Loss()

    def _init_model(self):
        if not hasattr(self.config, "output_dim"):
            self.config.output_dim = 1
        if not hasattr(self.config, "input_dim"):
            self.config.input_dim = 1
        return Model(self.config)

    def _init_optimizer(self, CovariateFusion=None):
        enc_in = getattr(self.config, "enc_in", 1)
        series_dim = getattr(self.config, "series_dim", enc_in)
        exog_dim = max(enc_in - series_dim, 0)
        self.model.c_in = enc_in
        self.model.exog_dim = exog_dim
        if exog_dim > 0 and not self.model._fusion_initialized:
            self.model.init_fusion_components(
                exog_dim, getattr(self.config, "alpha_init", 0.0)
            )

        if CovariateFusion is not None:
            parameters = [
                {"params": self.model.parameters(), "lr": self.config.lr},
                {"params": CovariateFusion.parameters(), "lr": self.config.lr},
            ]
            return optim.Adam(parameters)
        return optim.Adam(self.model.parameters(), lr=self.config.lr)

    def _process(
        self, input, target, input_mark, target_mark, exog_future=None
    ):
        output = self.model(
            input, x_mark_enc=input_mark, exog_future=exog_future
        )
        return {"output": output}


__all__ = ["PhoCoNet", "MODEL_HYPER_PARAMS"]

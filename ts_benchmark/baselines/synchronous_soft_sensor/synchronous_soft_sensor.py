"""Shared pointwise estimators for the synchronous soft-sensing control.

For every query time t, these models estimate TP as y_t = f(x_t).  The same
function f is applied independently at all 24 query positions; historical TP,
historical covariates, positional interactions, and information from other
query positions are deliberately unavailable.
"""

import torch.nn as nn
from torch import optim

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase


MODEL_HYPER_PARAMS = {
    "seq_len": 96,
    "horizon": 24,
    "hidden_dim": 64,
    "dropout": 0.1,
    "num_epochs": 100,
    "patience": 10,
    "batch_size": 64,
    "lr": 0.001,
    "lradj": "type1",
    "loss": "MSE",
}


class _PointwiseMLP(nn.Module):
    def __init__(self, exog_dim, output_dim, hidden_dim, dropout, linear_only=False):
        super().__init__()
        if exog_dim <= 0:
            raise ValueError("Synchronous soft sensing requires online covariates.")
        if linear_only:
            self.estimator = nn.Linear(exog_dim, output_dim)
        else:
            self.estimator = nn.Sequential(
                nn.Linear(exog_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, query_covariates):
        if query_covariates is None:
            raise ValueError("Query-window online covariates were not supplied.")
        return self.estimator(query_covariates)


class _SynchronousBase(DeepForecastingModelBase):
    linear_only = False

    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    def _init_model(self):
        exog_dim = self.config.input_dim - self.config.output_dim
        return _PointwiseMLP(
            exog_dim=exog_dim,
            output_dim=self.config.output_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            linear_only=self.linear_only,
        )

    def _init_optimizer(self, CovariateFusion=None):
        return optim.Adam(self.model.parameters(), lr=self.config.lr)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        # Intentionally ignore history: this is the y_t = f(x_t) control.
        return {"output": self.model(exog_future)}


class SynchronousLinear(_SynchronousBase):
    linear_only = True

    @property
    def model_name(self):
        return "SynchronousLinear"


class SynchronousMLP(_SynchronousBase):
    @property
    def model_name(self):
        return "SynchronousMLP"


class SynchronousMLPEndpoint(SynchronousMLP):
    """Pointwise MLP fitted using only the last label of each query window."""

    @property
    def model_name(self):
        return "SynchronousMLPEndpoint"

    def _post_process(self, output, target):
        return output[:, -1:, :], target[:, -1:, :]

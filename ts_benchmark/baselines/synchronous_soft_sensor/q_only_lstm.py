from torch import nn

from ts_benchmark.baselines.synchronous_soft_sensor.synchronous_soft_sensor import (
    _SynchronousBase,
)


class _QOnlyLSTM(nn.Module):
    def __init__(self, exog_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.encoder = nn.LSTM(exog_dim, hidden_dim, batch_first=True)
        self.readout = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, query_covariates):
        encoded, _ = self.encoder(query_covariates)
        return self.readout(encoded)


class QOnlyLSTM(_SynchronousBase):
    @property
    def model_name(self):
        return "QOnlyLSTM"

    def _init_model(self):
        return _QOnlyLSTM(
            self.config.input_dim - self.config.output_dim,
            self.config.hidden_dim,
            self.config.output_dim,
            self.config.dropout,
        )

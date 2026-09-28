import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.pred_len = config.pred_len
        self.output_dim = getattr(config, "output_dim", config.c_out)
        input_dim = getattr(config, "input_dim", config.enc_in)
        hidden_size = getattr(config, "hidden_size", 128)
        num_layers = getattr(config, "num_layers", 2)
        dropout = getattr(config, "dropout", 0.1)
        bidirectional = getattr(config, "bidirectional", False)
        direction_factor = 2 if bidirectional else 1

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.pred_len * self.output_dim),
        )

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        if self.encoder.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]
        output = self.readout(hidden)
        return output.view(x.shape[0], self.pred_len, self.output_dim)

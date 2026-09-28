from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.lstm.models.lstm_model import LSTMModel


MODEL_HYPER_PARAMS = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "bidirectional": False,
    "batch_size": 64,
    "lr": 0.001,
    "num_epochs": 20,
    "patience": 5,
    "loss": "MSE",
}


class LSTM(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super(LSTM, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "LSTM"

    def _init_model(self):
        return LSTMModel(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        return {"output": self.model(input)}

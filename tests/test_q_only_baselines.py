import torch

from ts_benchmark.baselines.synchronous_soft_sensor.synchronous_soft_sensor import SynchronousMLP
from ts_benchmark.baselines.synchronous_soft_sensor.q_only_lstm import QOnlyLSTM


def test_q_only_models_ignore_history_and_calendar():
    torch.set_num_threads(2)
    for cls in (SynchronousMLP, QOnlyLSTM):
        wrapper = cls()
        wrapper.config.input_dim = 9
        wrapper.config.output_dim = 1
        wrapper.model = wrapper._init_model().eval()
        query = torch.randn(2, 24, 8, requires_grad=True)
        history = torch.randn(2, 96, 9)
        marks = torch.randn(2, 96, 4)
        first = wrapper._process(history, None, marks, None, query)['output']
        second = wrapper._process(history + 100, None, marks - 100, None, query)['output']
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert first.shape == (2, 24, 1)
        gradient = torch.autograd.grad(first.sum(), query)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_q_only_lstm_does_not_read_later_query_positions():
    wrapper = QOnlyLSTM()
    wrapper.config.input_dim = 9
    wrapper.config.output_dim = 1
    model = wrapper._init_model().eval()
    query = torch.randn(2, 24, 8)
    changed = query.clone()
    changed[:, 12:] += 100
    torch.testing.assert_close(model(query)[:, :12], model(changed)[:, :12], rtol=0, atol=0)

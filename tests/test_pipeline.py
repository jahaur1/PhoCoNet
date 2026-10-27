"""Regression checks for the actual PhoCoNet explanation and execution pipeline."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    def test_upstream_model_checks(self):
        for filename in ("test_phoconet.py", "test_model_names.py"):
            spec = importlib.util.spec_from_file_location(filename[:-3], ROOT / "tests" / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name in sorted(vars(module)):
                if name.startswith("test_"):
                    with self.subTest(check=name):
                        getattr(module, name)()

    def test_shap_forward_and_gradient(self):
        from experiments.config import params_for
        from experiments.run_shap import differentiable_forward
        from ts_benchmark.baselines.phoconet.models.phoconet import Model
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = SimpleNamespace(**{**params_for("juzizhou", 24), "enc_in": 9, "series_dim": 1})
        model = Model(config).double().eval()
        model.init_fusion_components(8)
        # Nonzero trained-like correction ensures the gate and modulation paths
        # are checked, not just the zero-initialized output.
        with torch.no_grad():
            model.q_delta[-1].weight.normal_(std=0.03)
            model.q_delta[-1].bias.normal_(std=0.01)
        history = torch.randn(2, 96, 9, dtype=torch.float64, requires_grad=True)
        query = torch.randn(2, 24, 8, dtype=torch.float64, requires_grad=True)
        marks = torch.randn(2, 96, 4, dtype=torch.float64)
        actual = model(history, marks, query)
        equivalent = differentiable_forward(model, history, marks, query)
        torch.testing.assert_close(actual, equivalent, rtol=1e-10, atol=1e-10)
        scalar = equivalent[..., 0].mean()
        gh, gq = torch.autograd.grad(scalar, (history, query))
        epsilon = 1e-5
        for value, gradient, coordinate in ((history, gh, (0, 20, 0)), (query, gq, (0, 3, 2))):
            plus, minus = value.detach().clone(), value.detach().clone()
            plus[coordinate] += epsilon
            minus[coordinate] -= epsilon
            def evaluate(x):
                h, q = (x, query) if value is history else (history, x)
                return differentiable_forward(model, h, marks, q)[..., 0].mean()
            numerical = (evaluate(plus) - evaluate(minus)) / (2 * epsilon)
            torch.testing.assert_close(gradient[coordinate], numerical, rtol=1e-3, atol=1e-6)
        self.assertTrue(torch.isfinite(gh).all())
        self.assertTrue(torch.isfinite(gq).all())

    def test_expected_gradients_completeness_for_linear_output(self):
        from unittest.mock import patch
        from experiments import run_shap
        model = torch.nn.Linear(1, 1, bias=False)
        samples = np.ones((2, 120, 9), dtype=np.float32)
        samples[:, 96:, 0] = 0  # Future TP is unavailable.
        background = np.zeros((2, 120, 9), dtype=np.float32)
        marks = np.zeros((2, 96, 4), dtype=np.float32)
        def scalar(module, values, time):
            return values[:, :96, 0].mean(1) + 2 * values[:, 96:, 1].mean(1)
        with patch.object(run_shap, "forward_scalar", scalar):
            attribution, diagnostic = run_shap.expected_gradients(model, samples, marks, background, 4)
        np.testing.assert_allclose(attribution.sum((1, 2)), 3.0, rtol=1e-6)
        np.testing.assert_array_equal(attribution[:, 96:, 0], 0)
        self.assertLess(diagnostic["max_absolute_completeness_residual"], 1e-5)

if __name__ == "__main__":
    unittest.main()

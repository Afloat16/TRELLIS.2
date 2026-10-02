import importlib.util
import unittest
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("native_loss_utils", Path(__file__).resolve().parents[1] / "trellis2/utils/loss_utils.py")
loss_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loss_utils)


class SmoothL1ZeroBetaTest(unittest.TestCase):
    def test_zero_beta_is_l1_with_finite_exact_match_gradient(self):
        pred = torch.tensor([-2., 0., 3.], dtype=torch.float64, requires_grad=True)
        target = torch.zeros_like(pred)
        loss = loss_utils.smooth_l1_loss(pred, target, beta=0.)
        self.assertEqual(loss.item(), 5 / 3)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([-1., 0., 1.], dtype=torch.float64) / 3)

    def test_positive_beta_matches_piecewise_oracle_and_gradient(self):
        residual = [-2., -.5, 0., .25, 3.]
        for beta in [.1, 1., 4.]:
            pred = torch.tensor(residual, dtype=torch.float64, requires_grad=True)
            expected = sum(.5 * x * x / beta if abs(x) < beta else abs(x) - .5 * beta for x in residual) / len(residual)
            expected_grad = [x / beta if abs(x) < beta else (1. if x > 0 else -1.) for x in residual]
            loss = loss_utils.smooth_l1_loss(pred, torch.zeros_like(pred), beta)
            self.assertAlmostEqual(loss.item(), expected)
            loss.backward()
            torch.testing.assert_close(pred.grad, torch.tensor(expected_grad, dtype=torch.float64) / len(residual))

    def test_zero_beta_exact_matches_have_zero_gradient(self):
        pred = torch.zeros((2, 3, 4), requires_grad=True)
        loss = loss_utils.smooth_l1_loss(pred, torch.zeros_like(pred), beta=0.)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


if __name__ == "__main__":
    unittest.main()

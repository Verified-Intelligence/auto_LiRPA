import torch

from auto_LiRPA.operators.bivariate import BoundMul
from auto_LiRPA.operators.convex_concave import BoundReciprocal
from auto_LiRPA.operators.base import Interval
from testcase import TestCase


class TestReciprocal(TestCase):

    def test_interval_propagate_across_zero(self):
        node = BoundReciprocal.__new__(BoundReciprocal)
        lower = torch.tensor([[1.0, -2.0, 0.0, -2.0, -2.0, 0.0]])
        upper = torch.tensor([[2.0, -1.0, 2.0, 0.0, 3.0, 0.0]])

        output = node.interval_propagate(Interval(lower, upper))
        out_lower, out_upper = output.lower, output.upper

        expected_lower = torch.tensor([
            [0.5, -1.0, 0.5, -torch.inf, -torch.inf, -torch.inf]])
        expected_upper = torch.tensor([
            [1.0, -0.5, torch.inf, -0.5, torch.inf, torch.inf]])
        torch.testing.assert_close(out_lower, expected_lower)
        torch.testing.assert_close(out_upper, expected_upper)

    def test_softmax_mul_interval_handles_zero_times_infinity(self):
        node = BoundMul()
        node.is_linear_op = False
        node._is_softmax = lambda: True

        output = node.interval_propagate(
            Interval(torch.tensor([0.0]), torch.tensor([2.0])),
            Interval(torch.tensor([0.5]), torch.tensor([torch.inf])))
        lower, upper = output.lower, output.upper

        torch.testing.assert_close(lower, torch.tensor([0.0]))
        torch.testing.assert_close(upper, torch.tensor([1.0]))

    def test_softmax_mul_interval_preserves_finite_bounds(self):
        node = BoundMul()
        node.is_linear_op = False
        node._is_softmax = lambda: True

        output = node.interval_propagate(
            Interval(torch.tensor([0.2]), torch.tensor([0.4])),
            Interval(torch.tensor([1.0]), torch.tensor([2.0])))
        lower, upper = output.lower, output.upper

        torch.testing.assert_close(lower, torch.tensor([0.2]))
        torch.testing.assert_close(upper, torch.tensor([0.8]))

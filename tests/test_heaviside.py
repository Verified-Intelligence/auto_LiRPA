"""Bounds of torch.heaviside must account for its value at zero."""

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


class Heaviside(nn.Module):
    def __init__(self, value_at_zero, shift=0.):
        super().__init__()
        self.register_buffer('value_at_zero', torch.tensor(value_at_zero))
        self.shift = shift

    def forward(self, x):
        return torch.heaviside(x - self.shift, self.value_at_zero)


def _bounds(model, lower, upper, method):
    bounded = BoundedModule(model, torch.zeros(1, 1))
    ptb = PerturbationLpNorm(
        norm=float('inf'), x_L=torch.tensor([[lower]]), x_U=torch.tensor([[upper]]))
    x = BoundedTensor(torch.tensor([[(lower + upper) / 2]]), ptb)
    lb, ub = bounded.compute_bounds(x=(x,), method=method)
    return lb.item(), ub.item()


def _true_range(lower, upper, value_at_zero):
    z = torch.cat([torch.linspace(lower, upper, 101), torch.tensor([lower, upper])])
    if lower <= 0 <= upper:
        z = torch.cat([z, torch.zeros(1)])
    y = torch.heaviside(z, torch.tensor(value_at_zero))
    return y.min().item(), y.max().item()


INTERVALS = [(-1., 0.), (0., 1.), (0., 0.), (-1., 1.), (-2., -1.), (1., 2.)]
VALUES_AT_ZERO = [0., 0.5, 1., 2., -1.]
METHODS = ['IBP', 'CROWN', 'CROWN-Optimized']


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('value_at_zero', VALUES_AT_ZERO)
@pytest.mark.parametrize('lower,upper', INTERVALS)
def test_heaviside_bounds_are_sound(lower, upper, value_at_zero, method):
    lb, ub = _bounds(Heaviside(value_at_zero), lower, upper, method)
    true_lb, true_ub = _true_range(lower, upper, value_at_zero)
    assert lb <= true_lb + 1e-6 and ub >= true_ub - 1e-6


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('lower,upper,value_at_zero,expected', [
    (-2., -1., 0.5, (0., 0.)),
    (1., 2., 0.5, (1., 1.)),
    (-1., 0., 0., (0., 0.)),   # heaviside(0, 0) = 0: still stably 0
    (0., 1., 1., (1., 1.)),    # heaviside(0, 1) = 1: still stably 1
    (0., 0., 0., (0., 0.)),
    (0., 0., 1., (1., 1.)),
])
def test_heaviside_stable_bounds_are_exact(lower, upper, value_at_zero, expected, method):
    lb, ub = _bounds(Heaviside(value_at_zero), lower, upper, method)
    assert (lb, ub) == pytest.approx(expected)


@pytest.mark.parametrize('method', ['CROWN', 'CROWN-Optimized'])
def test_heaviside_bound_exactly_zero_after_linear_op(method):
    # z = x - 1 with x in [0, 1]: the upper bound of z is exactly 0, and x = 1 gives
    # heaviside(0, 1) = 1, so the upper bound of the output must be 1.
    lb, ub = _bounds(Heaviside(1., shift=1.), 0., 1., method)
    assert lb <= 0. and ub >= 1.

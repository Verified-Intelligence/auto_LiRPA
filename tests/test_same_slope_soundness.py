"""Soundness checks for same-slope S-shaped relaxations."""

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.linear_bound import LinearBound
from auto_LiRPA.operators.s_shaped import BoundAtan, BoundSigmoid, BoundTanh
from auto_LiRPA.perturbations import PerturbationLpNorm
from testcase import DEFAULT_DEVICE


class Atan(nn.Module):
    def forward(self, x):
        return torch.atan(x)


class MirrorTanh(nn.Module):
    """A network whose maximum occurs inside the perturbed interval."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 2)
        self.linear2 = nn.Linear(2, 1)
        with torch.no_grad():
            self.linear1.weight.copy_(torch.tensor([[1.], [-1.]]))
            self.linear1.bias.fill_(1.3)
            self.linear2.weight.fill_(1.)
            self.linear2.bias.zero_()

    def forward(self, x):
        return self.linear2(torch.tanh(self.linear1(x)))


ACTIVATIONS = (
    pytest.param(nn.Tanh, torch.tanh, id='tanh'),
    pytest.param(nn.Sigmoid, torch.sigmoid, id='sigmoid'),
    pytest.param(Atan, torch.atan, id='atan'),
)


def _bounded_input(lower, upper):
    value = (lower + upper) / 2
    return BoundedTensor(
        value, PerturbationLpNorm(x_L=lower, x_U=upper))


def _same_slope_A(model, lower, upper):
    bounded_model = BoundedModule(
        model, torch.empty_like(lower),
        bound_opts={'activation_bound_option': 'same-slope'},
        device=DEFAULT_DEVICE)
    output_name = bounded_model.output_name[0]
    input_name = bounded_model.input_name[0]
    _, _, A = bounded_model.compute_bounds(
        x=(_bounded_input(lower, upper),), method='CROWN', return_A=True,
        needed_A_dict={output_name: [input_name]})
    return A[output_name][input_name]


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('module,function', ACTIVATIONS)
def test_same_slope_lines_enclose_s_shaped_function(module, function, dtype):
    """Parallel lines enclose each activation across narrow and wide boxes."""
    centers = torch.tensor(
        [-13.6517, -3., -1.3, -0.658479, 0., 0.658479, 1.3, 3., 13.6517],
        dtype=dtype, device=DEFAULT_DEVICE)
    widths = torch.tensor(
        [0., 1e-6, 2e-3, 9.7966e-3, 0.2, 2., 10.],
        dtype=dtype, device=DEFAULT_DEVICE)
    centers, widths = torch.meshgrid(centers, widths, indexing='ij')
    centers = centers.reshape(-1, 1)
    half_widths = widths.reshape(-1, 1) / 2
    lower, upper = centers - half_widths, centers + half_widths

    A = _same_slope_A(module(), lower, upper)
    lA = A['lA'].reshape(-1)
    uA = A['uA'].reshape(-1)
    lbias = A['lbias'].reshape(-1)
    ubias = A['ubias'].reshape(-1)
    assert torch.equal(lA, uA)

    offsets = torch.linspace(
        -1., 1., 201, dtype=dtype, device=DEFAULT_DEVICE)
    samples = centers + half_widths * offsets
    values = function(samples)
    lower_lines = lA[:, None] * samples + lbias[:, None]
    upper_lines = uA[:, None] * samples + ubias[:, None]
    tolerance = 2e-5 if dtype == torch.float32 else 1e-12
    assert (lower_lines <= values + tolerance).all()
    assert (upper_lines >= values - tolerance).all()


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize(
    'operator,derivative_at_zero',
    ((BoundTanh, 1.), (BoundSigmoid, 0.25), (BoundAtan, 1.)))
def test_same_slope_bounds_have_finite_gradients(
        operator, derivative_at_zero, dtype):
    """Both affine bounds support backpropagation through input endpoints."""
    # At zero and on narrow boxes around it, the chord slope can equal the
    # maximum derivative. Ordinary boxes also exercise both curvature regions.
    lower = torch.tensor(
        [[0., 0., -1e-6, -2., 0.2]], dtype=dtype, requires_grad=True)
    upper = torch.tensor(
        [[0., 1e-6, 1e-6, -0.2, 2.]], dtype=dtype, requires_grad=True)
    node = operator(options={'activation_bound_option': 'same-slope'})
    node.bound_relax(LinearBound(lower=lower, upper=upper), init=True)

    for bound in (node.lw * lower + node.lb, node.uw * upper + node.ub):
        assert torch.isfinite(bound).all()
        gradients = torch.autograd.grad(
            bound.sum(), (lower, upper), retain_graph=True)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        # Moving both endpoints of an exact interval differentiates f itself.
        torch.testing.assert_close(
            gradients[0][0, 0] + gradients[1][0, 0],
            lower.new_tensor(derivative_at_zero))


def test_same_slope_bounds_contain_interior_maximum():
    """CROWN contains the reachable maximum of two mirrored tanh neurons."""
    dtype = torch.float64
    lower = torch.tensor([[-1e-3]], dtype=dtype, device=DEFAULT_DEVICE)
    upper = -lower
    model = MirrorTanh().to(dtype)
    bounded_model = BoundedModule(
        model, torch.empty_like(lower),
        bound_opts={'activation_bound_option': 'same-slope'},
        device=DEFAULT_DEVICE)
    _, output_upper = bounded_model.compute_bounds(
        x=(_bounded_input(lower, upper),), method='CROWN')
    reachable = model(torch.zeros_like(lower))
    assert output_upper >= reachable - 1e-12


def test_saturated_same_slope_bounds_shrink_with_input_box():
    """Nested boxes around a saturated tanh avoid abrupt relaxation growth."""
    dtype = torch.float64
    widths = torch.tensor(
        [1e-3, 7.4989e-4, 5.6234e-4, 3.1623e-4,
         2.3714e-4, 1.7783e-4, 1.3335e-4],
        dtype=dtype, device=DEFAULT_DEVICE).reshape(-1, 1)
    centers = torch.full_like(widths, 13.6517)
    lower, upper = centers - widths / 2, centers + widths / 2
    bounded_model = BoundedModule(
        nn.Tanh(), torch.empty_like(lower),
        bound_opts={'activation_bound_option': 'same-slope'},
        device=DEFAULT_DEVICE)
    output_lower, output_upper = bounded_model.compute_bounds(
        x=(_bounded_input(lower, upper),), method='CROWN')
    bound_widths = (output_upper - output_lower).reshape(-1)
    assert (bound_widths[1:] <= bound_widths[:-1] + 1e-12).all()

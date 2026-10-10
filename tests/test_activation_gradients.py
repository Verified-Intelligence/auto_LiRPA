"""Tests for bounds on activation gradients in Jacobian graphs."""
import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.jacobian import JacobianOP
from auto_LiRPA.operators.base import OperatorCapabilityError
from auto_LiRPA.operators.s_shaped import (
    BoundSigmoidGrad, BoundTanhGrad, SigmoidGradOp, TanhGradOp,
)
from auto_LiRPA.perturbations import PerturbationLpNorm


class JacobianModel(nn.Module):
    """Compute the Jacobian of one elementwise activation."""

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return JacobianOP.apply(self.activation(x), x)


class MixedJacobianModel(nn.Module):
    """Combine activation Jacobians whose inputs have different dtypes."""

    def forward(self, x_float, x_double):
        tanh_jacobian = JacobianOP.apply(torch.tanh(x_float), x_float)
        sigmoid_jacobian = JacobianOP.apply(
            torch.sigmoid(x_double), x_double)
        return tanh_jacobian.to(torch.float64) + sigmoid_jacobian


class NegatedGradientModel(nn.Module):
    """Apply a negative upstream coefficient to an activation gradient."""

    def __init__(self, gradient):
        super().__init__()
        self.gradient = gradient

    def forward(self, x):
        return -self.gradient(x)


def _tanh_gradient(x):
    return 1 - torch.tanh(x).square()


def _sigmoid_gradient(x):
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def _gradient_extrema(function, lower, upper):
    """Return exact extrema of a bell-shaped gradient on each interval."""
    endpoint_values = torch.stack((function(lower), function(upper)))
    exact_lower = endpoint_values.amin(dim=0)
    endpoint_upper = endpoint_values.amax(dim=0)
    contains_zero = torch.logical_and(lower <= 0, upper >= 0)
    exact_upper = torch.where(
        contains_zero, function(torch.zeros_like(lower)), endpoint_upper)
    return exact_lower, exact_upper


def _bounded_inputs(values, lowers, uppers):
    return tuple(
        BoundedTensor(
            value,
            PerturbationLpNorm(x_L=lower, x_U=upper),
        )
        for value, lower, upper in zip(values, lowers, uppers)
    )


def _compute_with_default_dtype(
        model, values, lowers, uppers, method, default_dtype):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        bounded_model = BoundedModule(model, values)
        return bounded_model.compute_bounds(
            x=_bounded_inputs(values, lowers, uppers), method=method)
    finally:
        torch.set_default_dtype(previous_dtype)


def _assert_default_independent_and_sound(
        model, values, lowers, uppers, method, exact_lower, exact_upper):
    results = [
        _compute_with_default_dtype(
            model, values, lowers, uppers, method, default_dtype)
        for default_dtype in (torch.float32, torch.float64)
    ]
    for actual, expected in zip(results[0], results[1]):
        assert torch.equal(actual, expected)

    lower, upper = results[0]
    exact_lower = exact_lower.reshape_as(lower)
    exact_upper = exact_upper.reshape_as(upper)
    tolerance = 1e-5 if lower.dtype == torch.float32 else 1e-12
    assert torch.all(lower <= exact_lower + tolerance)
    assert torch.all(upper >= exact_upper - tolerance)
    return lower, upper


ACTIVATIONS = (
    pytest.param(torch.tanh, _tanh_gradient, id='tanh'),
    pytest.param(torch.sigmoid, _sigmoid_gradient, id='sigmoid'),
)


@pytest.mark.parametrize('activation,gradient', ACTIVATIONS)
@pytest.mark.parametrize('method', ('CROWN', 'forward'))
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_activation_jacobian_bounds_ignore_default_dtype(
        activation, gradient, method, dtype):
    lower = torch.tensor([[-2.0, -0.3, 0.2]], dtype=dtype)
    upper = torch.tensor([[-0.4, 0.6, 2.5]], dtype=dtype)
    center = (lower + upper) / 2
    gradient_lower, gradient_upper = _gradient_extrema(
        gradient, lower, upper)
    exact_lower = torch.diag_embed(gradient_lower)
    exact_upper = torch.diag_embed(gradient_upper)

    actual_lower, actual_upper = _assert_default_independent_and_sound(
        JacobianModel(activation), (center,), (lower,), (upper,), method,
        exact_lower, exact_upper)

    assert actual_lower.dtype == dtype
    assert actual_upper.dtype == dtype


def test_mixed_dtype_activation_jacobians_are_node_owned():
    lowers = (
        torch.tensor([[0.0, -0.3]], dtype=torch.float32),
        torch.tensor([[0.1, -0.2]], dtype=torch.float64),
    )
    uppers = (
        torch.tensor([[0.2, -0.1]], dtype=torch.float32),
        torch.tensor([[0.3, 0.0]], dtype=torch.float64),
    )
    centers = tuple(
        (lower + upper) / 2 for lower, upper in zip(lowers, uppers))
    tanh_lower, tanh_upper = _gradient_extrema(
        _tanh_gradient, lowers[0], uppers[0])
    sigmoid_lower, sigmoid_upper = _gradient_extrema(
        _sigmoid_gradient, lowers[1], uppers[1])
    exact_lower = torch.diag_embed(tanh_lower.to(torch.float64)
                                   + sigmoid_lower)
    exact_upper = torch.diag_embed(tanh_upper.to(torch.float64)
                                   + sigmoid_upper)

    actual_lower, actual_upper = _assert_default_independent_and_sound(
        MixedJacobianModel(), centers, lowers, uppers, 'CROWN',
        exact_lower, exact_upper)

    assert actual_lower.dtype == torch.float64
    assert actual_upper.dtype == torch.float64


@pytest.mark.parametrize(
    'gradient,exact_gradient,lower_value,upper_value',
    (
        # The upper bound is just above -0.6585026, Tanh's negative
        # inflection point, and lookup selects the preceding 0.01 grid point,
        # which was zero-initialized before the fix.
        pytest.param(
            TanhGradOp.apply, _tanh_gradient,
            -0.744972619858304, -0.6558182477951053,
            id='tanh-negative-inflection'),
        # The same boundary case for Sigmoid, whose negative inflection point
        # is -1.3169614.
        pytest.param(
            SigmoidGradOp.apply, _sigmoid_gradient,
            -1.403, -1.314, id='sigmoid-negative-inflection'),
    ),
)
def test_gradient_bounds_are_sound_near_negative_inflection(
        gradient, exact_gradient, lower_value, upper_value):
    """A negative upstream coefficient must not expose an invalid tangent."""
    lower = torch.tensor([[lower_value]], dtype=torch.float64)
    upper = torch.tensor([[upper_value]], dtype=torch.float64)
    center = (lower + upper) / 2
    bounded_model = BoundedModule(
        NegatedGradientModel(gradient), center)
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(norm=float('inf'), x_L=lower, x_U=upper),
    )

    actual_lower, actual_upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='CROWN')
    exact_lower, exact_upper = _gradient_extrema(
        exact_gradient, lower, upper)
    expected_lower = -exact_upper
    expected_upper = -exact_lower
    tolerance = 1e-12

    assert torch.all(actual_lower <= expected_lower + tolerance)
    assert torch.all(actual_upper >= expected_upper - tolerance)


@pytest.mark.parametrize('method', ('CROWN', 'forward'))
def test_gradient_relaxation_covers_all_interval_regions(method):
    """Central, one-sided, and crossing intervals are all classified."""
    # Central, right, left, right-crossing, left-crossing, and wide-crossing.
    lower = torch.tensor([[-0.4, 0.7, -1.0, -1.0, -1.0, -2.0]])
    upper = torch.tensor([[0.4, 1.0, -0.7, 1.0, 0.4, 2.0]])
    center = (lower + upper) / 2
    bounded_model = BoundedModule(
        NegatedGradientModel(TanhGradOp.apply), center)
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(norm=float('inf'), x_L=lower, x_U=upper),
    )

    actual_lower, actual_upper = bounded_model.compute_bounds(
        x=(bounded_input,), method=method)
    exact_lower, exact_upper = _gradient_extrema(
        _tanh_gradient, lower, upper)

    assert torch.all(actual_lower <= -exact_upper + 1e-5)
    assert torch.all(actual_upper >= -exact_lower - 1e-5)


@pytest.mark.parametrize('operator_cls', (BoundTanhGrad, BoundSigmoidGrad))
def test_gradient_relaxation_regions_form_a_partition(operator_cls):
    """Every interval, including inflection boundaries, has one region."""
    bound_operator = operator_cls()
    inflection = bound_operator.inflection_point
    # Breakpoints, tails, and interior points generate every interval geometry.
    points = inflection * torch.tensor(
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        dtype=torch.float64)
    lower, upper = torch.meshgrid(points, points, indexing='ij')
    valid = lower <= upper
    lower, upper = lower[valid], upper[valid]

    regions = torch.stack(
        bound_operator._relaxation_region_masks(lower, upper))

    assert torch.equal(
        regions.sum(dim=0), torch.ones_like(lower, dtype=torch.int64))
    assert torch.all(regions.flatten(1).any(dim=1))
    at_inflection = (lower == upper) & (lower.abs() == inflection)
    assert torch.all(regions[0, at_inflection])


@pytest.mark.parametrize(
    'lower_value,upper_value',
    ((-501.0, -500.0), (500.0, 501.0)),
    ids=('below-table', 'above-table'),
)
def test_gradient_relaxation_rejects_bounds_outside_precomputed_range(
        lower_value, upper_value):
    """Finite bounds outside the tangent-table range fail explicitly."""
    lower = torch.tensor([[lower_value]])
    upper = torch.tensor([[upper_value]])
    center = (lower + upper) / 2
    bounded_model = BoundedModule(
        NegatedGradientModel(TanhGradOp.apply), center)
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(norm=float('inf'), x_L=lower, x_U=upper),
    )

    with pytest.raises(OperatorCapabilityError, match='precomputed relaxation'):
        bounded_model.compute_bounds(x=(bounded_input,), method='CROWN')

"""Restricted-domain operators report bound capability at the operator layer."""

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.linear_bound import LinearBound
from auto_LiRPA.operators.base import (
    Bound,
    Interval,
    OperatorCapabilityError,
    OperatorDomainError,
)
from auto_LiRPA.operators.bivariate import BoundMul
from auto_LiRPA.operators.convex_concave import (
    BoundLog,
    BoundReciprocal,
    BoundSqrt,
)
from auto_LiRPA.operators.cut_ops import CutModule
from auto_LiRPA.operators.gelu import GELU
from auto_LiRPA.operators.s_shaped import (
    BoundTan,
    SigmoidGradOp,
    TanhGradOp,
)
from auto_LiRPA.operators.trigonometric import BoundSec
from auto_LiRPA.perturbations import PerturbationLpNorm


class UnaryModel(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class BinaryModel(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x, y):
        return self.function(x, y)


class HeavisideModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('value_at_zero', torch.tensor(0.5))

    def forward(self, x):
        return torch.heaviside(x, self.value_at_zero)


class MatMulModel(nn.Module):
    def forward(self, x, weight):
        return x.matmul(weight)


def _compute_bounds(function, lower, upper, method, C=None):
    ptb = PerturbationLpNorm(
        norm=float('inf'), x_L=lower, x_U=upper)
    bounded_input = BoundedTensor(lower, ptb)
    model = BoundedModule(UnaryModel(function), lower)
    return model.compute_bounds(x=(bounded_input,), method=method, C=C)


def _compute_unbounded_model(model, centers, method):
    if isinstance(centers, torch.Tensor):
        centers = (centers,)
    example_inputs = centers[0] if len(centers) == 1 else centers
    bounded_model = BoundedModule(model, example_inputs)
    bounded_inputs = tuple(
        BoundedTensor(
            center,
            PerturbationLpNorm(
                norm=float('inf'),
                x_L=torch.full_like(center, -torch.inf),
                x_U=torch.full_like(center, torch.inf),
            ),
        )
        for center in centers
    )
    return bounded_model.compute_bounds(x=bounded_inputs, method=method)


UNARY_LINEAR_RELAXATIONS = [
    pytest.param(torch.relu, id='relu'),
    pytest.param(torch.abs, id='abs'),
    pytest.param(torch.square, id='square'),
    pytest.param(torch.exp, id='exp'),
    pytest.param(torch.tanh, id='tanh'),
    pytest.param(torch.sigmoid, id='sigmoid'),
    pytest.param(torch.atan, id='atan'),
    pytest.param(torch.sin, id='sin'),
    pytest.param(torch.cos, id='cos'),
    pytest.param(torch.floor, id='floor'),
    pytest.param(torch.sign, id='sign'),
]


@pytest.mark.parametrize('method', ['CROWN', 'forward'])
@pytest.mark.parametrize('function', UNARY_LINEAR_RELAXATIONS)
def test_unary_linear_relaxations_reject_unbounded_input(function, method):
    with pytest.raises(OperatorCapabilityError, match='finite input bounds'):
        _compute_unbounded_model(
            UnaryModel(function), torch.zeros(1, 2), method)


def test_relu_dynamic_forward_rejects_unbounded_input():
    with pytest.raises(OperatorCapabilityError, match='finite input bounds'):
        _compute_unbounded_model(
            UnaryModel(torch.relu), torch.zeros(1, 2),
            method='dynamic-forward')


@pytest.mark.parametrize(
    'model',
    [
        pytest.param(GELU(), id='gelu'),
        pytest.param(UnaryModel(TanhGradOp.apply), id='tanh_grad'),
        pytest.param(UnaryModel(SigmoidGradOp.apply), id='sigmoid_grad'),
        pytest.param(nn.Hardtanh(), id='hardtanh'),
        pytest.param(HeavisideModel(), id='heaviside'),
        pytest.param(nn.Softmax(dim=-1), id='softmax'),
    ],
)
def test_crown_only_relaxations_reject_unbounded_input(model):
    with pytest.raises(OperatorCapabilityError, match='finite input bounds'):
        _compute_unbounded_model(model, torch.zeros(1, 2), method='CROWN')


def test_max_pool_relaxations_reject_unbounded_input():
    model = nn.MaxPool2d(kernel_size=2)
    center = torch.zeros(1, 1, 2, 2)

    for method in ('CROWN', 'forward'):
        with pytest.raises(
                OperatorCapabilityError, match='finite input bounds'):
            _compute_unbounded_model(model, center, method)


@pytest.mark.parametrize('function', [torch.mul, torch.maximum, torch.minimum])
def test_binary_crown_relaxations_reject_unbounded_inputs(function):
    center = torch.zeros(1, 2)
    with pytest.raises(OperatorCapabilityError, match='finite input bounds'):
        _compute_unbounded_model(
            BinaryModel(function), (center, center), method='CROWN')


def test_mul_forward_relaxation_rejects_unbounded_inputs():
    center = torch.zeros(1, 2)
    with pytest.raises(OperatorCapabilityError, match='finite input bounds'):
        _compute_unbounded_model(
            BinaryModel(torch.mul), (center, center), method='forward')


def test_softmax_mul_handles_unbounded_intermediate_bounds_by_method():
    model = BoundedModule(
        nn.Softmax(dim=-1), torch.zeros(1, 2),
        bound_opts={
            'softmax': 'complex',
            'fixed_reducemax_index': True,
        })
    mul = next(node for node in model.nodes() if isinstance(node, BoundMul))

    finite = LinearBound(
        lower=torch.tensor([[0.0]]), upper=torch.tensor([[1.0]]))
    unbounded = LinearBound(
        lower=torch.tensor([[0.0]]), upper=torch.tensor([[torch.inf]]))

    mul.bound_relax(finite, unbounded)
    torch.testing.assert_close(mul.lb, torch.zeros_like(mul.lb))
    torch.testing.assert_close(mul.ub, torch.ones_like(mul.ub))

    with pytest.raises(OperatorCapabilityError, match='finite input bounds'):
        mul.bound_forward(1, finite, unbounded)


@pytest.mark.parametrize('method', ['CROWN', 'forward'])
def test_perturbed_matmul_relaxation_rejects_unbounded_inputs(method):
    data = torch.zeros(1, 2)
    weight = torch.ones(1, 2, 1)
    with pytest.raises(OperatorCapabilityError, match='finite input bounds'):
        _compute_unbounded_model(
            MatMulModel(), (data, weight), method=method)


@pytest.mark.parametrize('method', ['IBP', 'CROWN', 'forward'])
def test_nan_input_bound_violates_bound_invariant(method):
    with pytest.raises(RuntimeError, match='NaN detected') as error:
        _compute_bounds(
            torch.relu,
            torch.tensor([[torch.nan]]),
            torch.tensor([[1.0]]),
            method=method,
        )
    assert not isinstance(error.value, OperatorCapabilityError)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
def test_finite_relaxation_check_preserves_overflow_and_infeasible_domains(dtype):
    node = Bound()
    largest = torch.finfo(dtype).max
    for lower, upper in [(largest / 2, largest), (-largest, -largest / 2),
                         (torch.inf, -torch.inf)]:
        node.require_finite_relaxation_bounds(
            torch.tensor([lower], dtype=dtype),
            torch.tensor([upper], dtype=dtype), operator='test')
    for lower, upper in [([-torch.inf], [torch.inf]),
                         ([2., -torch.inf], [1., 0.])]:
        with pytest.raises(OperatorCapabilityError):
            node.require_finite_relaxation_bounds(
                torch.tensor(lower, dtype=dtype),
                torch.tensor(upper, dtype=dtype), operator='test')


@pytest.mark.parametrize('dtype', [torch.bool, torch.int64])
def test_finite_relaxation_check_accepts_nonfloating_endpoints(dtype):
    Bound().require_finite_relaxation_bounds(
        torch.tensor([0], dtype=dtype), torch.tensor([1], dtype=dtype),
        operator='test')


@pytest.mark.parametrize('nan_side', ['lower', 'upper', 'both'])
def test_bound_invariant_check_reports_nan_endpoint(nan_side):
    node = Bound()
    lower = torch.tensor([torch.nan if nan_side != 'upper' else 0.0])
    upper = lower if nan_side == 'both' else torch.tensor([
        torch.nan if nan_side == 'upper' else torch.inf])
    with pytest.raises(RuntimeError, match=f'{"upper" if nan_side == "upper" else "lower"} bound'):
        node.check_bound_invariants(lower, upper, context='test')
    with pytest.raises(RuntimeError, match=f'{"upper" if nan_side == "upper" else "lower"} bound'):
        node.require_finite_relaxation_bounds(lower, upper, operator='test')
    # Infinite endpoints are allowed by the NaN invariant, even when their sum
    # is indeterminate. The affine-relaxation capability check is separate.
    node.check_bound_invariants(
        torch.tensor([-torch.inf]), torch.tensor([torch.inf]), context='test')


def test_nan_intermediate_bound_violates_bound_invariant():
    values = torch.zeros(1, 1)
    model = BoundedModule(nn.ReLU(), values)
    bounded_input = BoundedTensor(
        values,
        PerturbationLpNorm(
            norm=float('inf'), x_L=values - 1.0, x_U=values + 1.0),
    )
    intermediate_bounds = {
        model.final_node().name: (torch.full_like(values, torch.nan), values)
    }

    with pytest.raises(RuntimeError, match='NaN detected'):
        model.compute_bounds(
            x=(bounded_input,), method='IBP',
            interm_bounds=intermediate_bounds)


@pytest.mark.parametrize('method', ['forward', 'CROWN', 'dynamic-forward'])
def test_nan_concretization_violates_bound_invariant(method):
    values = torch.zeros(1, 1, 2)

    with pytest.raises(RuntimeError, match='NaN detected'):
        _compute_unbounded_model(nn.Flatten(), values, method=method)


def test_cut_bias_rejects_nan_propagation():
    values = torch.zeros(1, 1)
    model = BoundedModule(nn.ReLU(), values)
    start_node = model.final_node()
    cut_module = CutModule(
        general_beta={
            start_node.name: torch.full((2, 1, 1, 1), torch.nan)
        },
        active_cuts={start_node.name: torch.tensor([0])},
        cut_bias=torch.ones(1),
    )

    with pytest.raises(RuntimeError, match='NaN detected'):
        cut_module.bias_cut(start_node, values, values)


def test_cut_coefficients_follow_beta_computation_dtype():
    """Cut coefficients must not force restored beta to their storage dtype."""
    cut_module = CutModule()
    general_beta = torch.ones((2, 1, 1, 1), dtype=torch.float64)
    coefficients = torch.full((1, 1), 2.0, dtype=torch.float32)
    lA = torch.zeros((1, 1, 1), dtype=torch.float64)

    new_lA, _ = cut_module.general_beta_coeffs_addmm_to_A(
        lA, None, general_beta, coefficients, current_layer_shape=(1,))

    assert new_lA.dtype == torch.float64
    torch.testing.assert_close(new_lA, torch.full_like(new_lA, -2.0))


def test_ibp_widens_nan_operator_output():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    values = torch.zeros(1, 1)
    input_lower = torch.full_like(values, -torch.inf)
    input_upper = torch.full_like(values, torch.inf)
    ptb = PerturbationLpNorm(
        norm=float('inf'), x_L=input_lower, x_U=input_upper)
    bounded_input = BoundedTensor(values, ptb)
    bounded_model = BoundedModule(model, values)

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='IBP')

    assert torch.isneginf(lower).all()
    assert torch.isposinf(upper).all()


def test_ibp_widens_nan_specification_output():
    lower, upper = _compute_bounds(
        torch.relu,
        torch.tensor([[-torch.inf]]),
        torch.tensor([[torch.inf]]),
        method='IBP',
        C=torch.ones(1, 1, 1),
    )

    assert torch.isneginf(lower).all()
    assert torch.isposinf(upper).all()


def test_log_ibp_preserves_function_and_rejects_nonpositive_domain():
    node = BoundLog(options={})
    lower = torch.tensor([0.5])
    upper = torch.tensor([2.0])

    output = node.interval_propagate(Interval(lower, upper))
    out_lower, out_upper = output.lower, output.upper

    torch.testing.assert_close(out_lower, torch.log(lower))
    torch.testing.assert_close(out_upper, torch.log(upper))
    assert torch.isnan(node.forward(torch.tensor([-1.0]))).all()
    with pytest.raises(OperatorDomainError, match="strictly positive"):
        node.interval_propagate(
            Interval(torch.tensor([-1.0]), upper))


def test_sqrt_ibp_accepts_zero_and_rejects_negative_domain():
    node = BoundSqrt(options={})
    lower = torch.tensor([0.0])
    upper = torch.tensor([4.0])

    output = node.interval_propagate(Interval(lower, upper))
    out_lower, out_upper = output.lower, output.upper

    torch.testing.assert_close(out_lower, torch.tensor([0.0]))
    torch.testing.assert_close(out_upper, torch.tensor([2.0]))
    with pytest.raises(OperatorDomainError, match="non-negative"):
        node.interval_propagate(
            Interval(torch.tensor([-1.0]), upper))


@pytest.mark.parametrize(
    'function,lower', [(torch.log, 1.0), (torch.sqrt, 0.0)])
def test_log_and_sqrt_reject_unbounded_linear_relaxation(function, lower):
    input_lower = torch.tensor([[lower]])
    input_upper = torch.tensor([[torch.inf]])

    out_lower, out_upper = _compute_bounds(
        function, input_lower, input_upper, method='IBP')

    torch.testing.assert_close(out_lower, function(input_lower))
    assert torch.isposinf(out_upper).all()
    for method in ('CROWN', 'forward'):
        with pytest.raises(
                OperatorCapabilityError, match="finite input bounds"):
            _compute_bounds(
                function,
                input_lower,
                input_upper,
                method=method,
            )


@pytest.mark.parametrize('method', ['CROWN', 'forward', 'alpha-CROWN'])
def test_sqrt_linear_bounds_handle_fixed_zero_exactly(method):
    values = torch.tensor([[0.0, 4.0]])

    lower, upper = _compute_bounds(
        torch.sqrt, values, values, method=method)

    expected = torch.tensor([[0.0, 2.0]])
    torch.testing.assert_close(lower, expected)
    torch.testing.assert_close(upper, expected)


def test_reciprocal_bound_methods_distinguish_supported_and_invalid_domains():
    lower = torch.tensor([[-1.0]])
    upper = torch.tensor([[1.0]])

    out_lower, out_upper = _compute_bounds(
        torch.reciprocal, lower, upper, method='IBP')

    assert torch.isneginf(out_lower).all()
    assert torch.isposinf(out_upper).all()
    for method in ('CROWN', 'forward'):
        with pytest.raises(OperatorDomainError, match="contains zero"):
            _compute_bounds(torch.reciprocal, lower, upper, method=method)

    one_sided_lower = torch.tensor([[-torch.inf, -4.0, 1.0, 2.0]])
    one_sided_upper = torch.tensor([[-1.0, -2.0, torch.inf, 4.0]])
    expected_lower = torch.tensor([[-1.0, -0.5, 0.0, 0.25]])
    expected_upper = torch.tensor([[0.0, -0.25, 1.0, 0.5]])
    node = BoundReciprocal(options={})
    output = node.interval_propagate(
        Interval(one_sided_lower, one_sided_upper))
    ibp_lower, ibp_upper = output.lower, output.upper
    torch.testing.assert_close(ibp_lower, expected_lower)
    torch.testing.assert_close(ibp_upper, expected_upper)

    node.bound_relax(
        LinearBound(lower=one_sided_lower, upper=one_sided_upper),
        init=True)
    torch.testing.assert_close(
        node.lw, torch.tensor([[0.0, -1.0 / 8.0, 0.0, -1.0 / 9.0]]))
    torch.testing.assert_close(
        node.lb, torch.tensor([[-1.0, -3.0 / 4.0, 0.0, 2.0 / 3.0]]))
    torch.testing.assert_close(
        node.uw, torch.tensor([[0.0, -1.0 / 9.0, 0.0, -1.0 / 8.0]]))
    torch.testing.assert_close(
        node.ub, torch.tensor([[0.0, -2.0 / 3.0, 1.0, 3.0 / 4.0]]))

    for method in ('CROWN', 'forward', 'alpha-CROWN'):
        bound_lower, bound_upper = _compute_bounds(
            torch.reciprocal,
            torch.tensor([[-4.0]]),
            torch.tensor([[-2.0]]),
            method=method,
        )
        assert torch.isfinite(bound_lower).all()
        assert torch.isfinite(bound_upper).all()
        assert (bound_lower <= -0.5).all()
        assert (bound_upper >= -0.25).all()


def test_tan_reports_pole_crossing_as_operator_domain_failure():
    input_lower = torch.tensor([[-1.0]])
    input_upper = torch.tensor([[1.0]])

    out_lower, out_upper = _compute_bounds(
        torch.tan, input_lower, input_upper, method='CROWN')

    assert torch.isfinite(out_lower).all()
    assert torch.isfinite(out_upper).all()
    assert (out_lower <= torch.tan(input_lower)).all()
    assert (out_upper >= torch.tan(input_upper)).all()
    for method in ('CROWN', 'forward'):
        with pytest.raises(OperatorDomainError, match="cross a pole"):
            _compute_bounds(
                torch.tan,
                torch.tensor([[1.0]]),
                torch.tensor([[2.0]]),
                method=method,
            )
    with pytest.raises(OperatorDomainError, match="cross a pole"):
        _compute_bounds(
            torch.tan,
            torch.tensor([[1.0]]),
            torch.tensor([[2.0]]),
            method='IBP',
        )
    with pytest.raises(OperatorDomainError, match="cross a pole"):
        pole = torch.tensor([torch.pi / 2])
        _compute_bounds(torch.tan, pole, pole, method='CROWN')


def test_sec_distinguishes_poles_from_unsupported_periods():
    node = BoundSec(options={})

    with pytest.raises(OperatorDomainError, match='cross a pole'):
        node.interval_propagate(
            Interval(torch.tensor([-2.0]), torch.tensor([0.0])))
    with pytest.raises(OperatorCapabilityError, match='only within'):
        node.interval_propagate(
            Interval(
                torch.tensor([torch.pi]), torch.tensor([torch.pi])))

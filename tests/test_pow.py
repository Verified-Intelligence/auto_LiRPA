"""Test BoundPow."""
import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators.base import (
    OperatorCapabilityError,
    OperatorDomainError,
)
from auto_LiRPA.operators.s_shaped import BoundPow
from auto_LiRPA.perturbations import PerturbationLpNorm
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class PowModel(nn.Module):
    """Simple model that computes x^exponent."""
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent

    def forward(self, x):
        return torch.pow(x, self.exponent)


class VariablePowModel(nn.Module):
    def forward(self, x, exponent):
        return torch.pow(x, exponent)


class MixedPowModel(nn.Module):
    """Combine odd-power nodes with distinct exponents and input dtypes."""

    def forward(self, x_float, x_double):
        return x_float.pow(3).to(torch.float64) + x_double.pow(5)


# Non-integer powers are only supported for x >= 0. On this domain x^p is
# monotone, so endpoint values give the exact interval bounds.
NONINTEGER_IBP_CROWN_CASES = [
    # Concave, 0 < p < 1.
    (0.5, 1, 4),
    (0.5, 0.1, 10),
    (0.3, 1, 5),
    (0.7, 2, 8),
    (0.9, 0.5, 3),
    (0.5, 0, 4),
    # Convex, p > 1.
    (1.5, 1, 4),
    (1.5, 0.5, 3),
    (2.5, 1, 3),
    (3.7, 0.5, 2),
    (1.1, 1, 5),
    (1.5, 0, 4),
    (2.5, 0, 3),
]

NONINTEGER_ALPHA_SWEEP_CASES = [
    # Representative exponents across concave, near-linear, and convex ranges.
    (0.1, 1, 5),
    (0.25, 0.5, 8),
    (0.5, 0, 4),
    (0.75, 0, 6),
    (0.99, 1, 3),
    (1.01, 1, 3),
    (1.5, 0, 4),
    (2.5, 0, 3),
    (3.5, 0.1, 2),
    (4.7, 0.5, 2),
    # Narrow intervals where optimized tangent points matter more.
    (0.5, 2, 2.5),
    (1.5, 3, 3.2),
    (2.5, 1, 1.1),
    # Wide intervals for numerical stability.
    (0.5, 0.01, 100),
    (1.5, 0.1, 10),
    (2.5, 0.01, 10),
]

EVEN_INTEGER_EXPONENTS = [4, 6]


def even_integer_input_bounds(device=None, dtype=None):
    input_lb = torch.tensor([[-7.681, -6.174, -2.0, 0.2, 3.919]],
                            device=device, dtype=dtype)
    input_ub = torch.tensor([[-4.414, -1.939, 3.0, 2.5, 5.997]],
                            device=device, dtype=dtype)
    return input_lb, input_ub


def test_odd_power_same_slope_lines_enclose_function():
    """Parallel bounds enclose an odd power on each curvature region."""
    lower = torch.tensor([[-2.], [-2.], [1.]], dtype=torch.float64)
    upper = torch.tensor([[-1.], [2.], [2.]], dtype=torch.float64)
    bounded_input = BoundedTensor(
        (lower + upper) / 2,
        PerturbationLpNorm(x_L=lower, x_U=upper))
    bounded_model = BoundedModule(
        PowModel(3), lower,
        bound_opts={'activation_bound_option': 'same-slope'})
    output_name = bounded_model.output_name[0]
    input_name = bounded_model.input_name[0]
    _, _, A = bounded_model.compute_bounds(
        x=(bounded_input,), method='CROWN', return_A=True,
        needed_A_dict={output_name: [input_name]})
    coefficients = A[output_name][input_name]

    offsets = torch.linspace(0., 1., 201, dtype=lower.dtype)
    samples = lower + (upper - lower) * offsets
    values = samples.pow(3)
    lower_line = (coefficients['lA'].reshape(-1, 1) * samples
                  + coefficients['lbias'].reshape(-1, 1))
    upper_line = (coefficients['uA'].reshape(-1, 1) * samples
                  + coefficients['ubias'].reshape(-1, 1))
    assert (lower_line <= values + 1e-12).all()
    assert (upper_line >= values - 1e-12).all()


def true_even_integer_bounds(input_lb, input_ub, exponent):
    true_lb = torch.where(
        torch.logical_and(input_lb <= 0, input_ub >= 0),
        torch.zeros_like(input_lb),
        torch.minimum(input_lb.pow(exponent), input_ub.pow(exponent)))
    true_ub = torch.maximum(input_lb.pow(exponent), input_ub.pow(exponent))
    return true_lb, true_ub


def compute_bounds(exponent, input_lb, input_ub, method='CROWN',
                   optimize_iterations=0, device='auto'):
    model = PowModel(exponent)
    ptb_data = BoundedTensor(
        (input_lb + input_ub) / 2,
        PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub))
    bounded_model = BoundedModule(model, ptb_data, device=device)
    if optimize_iterations > 0:
        bounded_model.set_bound_opts({
            'optimize_bound_args': {
                'iteration': optimize_iterations, 'lr_alpha': 0.1}
        })
    lb, ub = bounded_model.compute_bounds(x=(ptb_data,), method=method)
    return bounded_model, lb, ub


def check_monotone_pow_soundness(exponent, x_low, x_high, method='CROWN',
                                  atol=1e-5, optimize_iterations=0):
    assert x_low >= 0, f'x_low must be >= 0 for monotonicity; got {x_low}'
    assert exponent > 0, f'exponent must be > 0; got {exponent}'
    input_lb = torch.ones(1, 10) * x_low
    input_ub = torch.ones(1, 10) * x_high
    _, lb, ub = compute_bounds(
        exponent, input_lb, input_ub, method, optimize_iterations)
    true_lb = x_low ** exponent
    true_ub = x_high ** exponent
    return lb.min().item() <= true_lb + atol and ub.max().item() >= true_ub - atol


class TestPowElementwise:
    """Elementwise BoundPow soundness checks with analytically known bounds."""

    @pytest.mark.parametrize('method', ['IBP', 'CROWN'])
    @pytest.mark.parametrize('exponent,x_low,x_high', NONINTEGER_IBP_CROWN_CASES)
    def test_noninteger_ibp_crown_soundness(self, exponent, x_low, x_high, method):
        assert check_monotone_pow_soundness(exponent, x_low, x_high, method=method), \
            f'Bounds not sound for x^{exponent} on [{x_low}, {x_high}] with {method}'

    @pytest.mark.parametrize('method', ['CROWN', 'alpha-CROWN'])
    @pytest.mark.parametrize('exponent', EVEN_INTEGER_EXPONENTS)
    def test_even_integer_boundpow_soundness(self, exponent, method):
        input_lb, input_ub = even_integer_input_bounds()
        bounded_model, lb, ub = compute_bounds(
            exponent, input_lb, input_ub, method,
            optimize_iterations=5 if method == 'alpha-CROWN' else 0)
        assert any(type(node).__name__ == 'BoundPow' for node in bounded_model.nodes())

        true_lb, true_ub = true_even_integer_bounds(input_lb, input_ub, exponent)
        assert torch.all(lb <= true_lb + 1e-5), \
            f'x^{exponent} {method} lower bound is unsound'
        assert torch.all(ub >= true_ub - 1e-4), \
            f'x^{exponent} {method} upper bound is unsound'


class TestPowTableOwnership:
    """Odd-power tangent tables follow their node and cache dependencies."""

    @staticmethod
    def _bounded_input(dtype):
        """Create a two-dimensional perturbed input box."""
        lower = torch.tensor([[-0.7, -0.2]], dtype=dtype)
        upper = torch.tensor([[0.4, 0.9]], dtype=dtype)
        value = (lower + upper) / 2
        return BoundedTensor(
            value, PerturbationLpNorm(x_L=lower, x_U=upper))

    @staticmethod
    def _pow_node(bounded_model):
        """Return the only BoundPow node in a test graph."""
        return next(
            node for node in bounded_model.nodes()
            if isinstance(node, BoundPow))

    @staticmethod
    def _table_tensors(node):
        """Return all tangent-table tensors owned by one BoundPow node."""
        return node.d_lower, node.d_upper

    @staticmethod
    def _assert_odd_power_bounds(lower, upper, bounded_input, exponent):
        """Check output bounds against the exact range of an odd power."""
        input_lower = bounded_input.ptb.x_L
        input_upper = bounded_input.ptb.x_U
        rounding_margin = 8 * torch.finfo(input_lower.dtype).eps
        assert (lower <= input_lower.pow(exponent) + rounding_margin).all()
        assert (upper >= input_upper.pow(exponent) - rounding_margin).all()

    @pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        'method,activation_bound_option',
        [
            # Cover both tangent lookup modes without forming an unrelated
            # Cartesian product with alpha optimization.
            pytest.param('CROWN', 'adaptive', id='crown-adaptive'),
            pytest.param('CROWN', 'same-slope', id='crown-same-slope'),
            pytest.param(
                'alpha-CROWN', 'adaptive', id='alpha-crown-adaptive'),
            pytest.param('forward', 'adaptive', id='forward-adaptive'),
            pytest.param('forward', 'same-slope', id='forward-same-slope'),
        ],
    )
    def test_bounds_are_default_independent(
            self, dtype, method, activation_bound_option):
        """Odd-power bounds are sound and independent of the global default."""
        options = {'activation_bound_option': activation_bound_option}
        if method == 'alpha-CROWN':
            # Two iterations include one optimizer update and one final bound.
            options['optimize_bound_args'] = {'iteration': 2}

        results = []
        previous_dtype = torch.get_default_dtype()
        try:
            for default_dtype in (torch.float32, torch.float64):
                torch.set_default_dtype(default_dtype)
                bounded_input = self._bounded_input(dtype)
                bounded_model = BoundedModule(
                    PowModel(3), bounded_input, bound_opts=options)

                lower, upper = bounded_model.compute_bounds(
                    x=(bounded_input,), method=method)
                assert lower.dtype == dtype
                assert upper.dtype == dtype
                self._assert_odd_power_bounds(
                    lower, upper, bounded_input, exponent=3)
                results.append((lower, upper))
        finally:
            torch.set_default_dtype(previous_dtype)

        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)

    def test_bound_methods_share_the_same_cached_tables(self):
        """CROWN, forward, and optimized bounds reuse one valid table set."""
        bounded_input = self._bounded_input(torch.float32)
        bounded_model = BoundedModule(PowModel(3), bounded_input)
        node = self._pow_node(bounded_model)
        assert all(table is None for table in self._table_tensors(node))

        bounded_model.compute_bounds(x=(bounded_input,), method='CROWN')
        cached_tables = self._table_tensors(node)
        assert all(
            table.dtype == bounded_input.dtype for table in cached_tables)
        assert all(
            table.device == bounded_input.device for table in cached_tables)

        bounded_model.compute_bounds(x=(bounded_input,), method='forward')
        assert all(
            current is cached for current, cached in zip(
                self._table_tensors(node), cached_tables))

        bounded_model.set_bound_opts({
            'optimize_bound_args': {'iteration': 2}})
        bounded_model.compute_bounds(
            x=(bounded_input,), method='alpha-CROWN')
        assert all(
            current is cached for current, cached in zip(
                self._table_tensors(node), cached_tables))

    def test_exponent_change_invalidates_tables(self):
        """Changing the exponent rebuilds mathematically valid tables once."""
        bounded_input = self._bounded_input(torch.float32)
        exponent_3 = torch.tensor(3.0)
        exponent_5 = torch.tensor(5.0)
        bounded_model = BoundedModule(
            VariablePowModel(), (bounded_input, exponent_3))
        node = self._pow_node(bounded_model)

        cubic_lower, cubic_upper = bounded_model.compute_bounds(
            x=(bounded_input, exponent_3), method='CROWN')
        cubic_tables = self._table_tensors(node)
        self._assert_odd_power_bounds(
            cubic_lower, cubic_upper, bounded_input, exponent=3)

        quintic_lower, quintic_upper = bounded_model.compute_bounds(
            x=(bounded_input, exponent_5), method='CROWN')
        quintic_tables = self._table_tensors(node)
        assert all(
            current is not previous
            for current, previous in zip(quintic_tables, cubic_tables))
        assert any(
            not torch.equal(cubic, quintic)
            for cubic, quintic in zip(cubic_tables, quintic_tables))
        self._assert_odd_power_bounds(
            quintic_lower, quintic_upper, bounded_input, exponent=5)

        bounded_model.compute_bounds(
            x=(bounded_input, exponent_5), method='CROWN')
        assert all(
            current is cached
            for current, cached in zip(
                self._table_tensors(node), quintic_tables))

    def test_each_node_owns_tables_in_mixed_dtype_graph(self):
        """Each odd-power node follows its own exponent, dtype, and device."""
        inputs = (
            torch.tensor([[0.1, -0.2]], dtype=torch.float32),
            torch.tensor([[0.2, -0.1]], dtype=torch.float64),
        )
        bounded_inputs = tuple(
            BoundedTensor(
                value,
                PerturbationLpNorm(
                    x_L=value - 0.1, x_U=value + 0.1),
            )
            for value in inputs
        )
        bounded_model = BoundedModule(MixedPowModel(), inputs)

        lower, upper = bounded_model.compute_bounds(
            x=bounded_inputs, method='CROWN')
        nodes = [
            node for node in bounded_model.nodes()
            if isinstance(node, BoundPow)
        ]

        assert len(nodes) == 2
        nodes_by_exponent = {node.exponent: node for node in nodes}
        assert set(nodes_by_exponent) == {3, 5}
        assert all(
            table.dtype == torch.float32
            for table in self._table_tensors(nodes_by_exponent[3]))
        assert all(
            table.dtype == torch.float64
            for table in self._table_tensors(nodes_by_exponent[5]))
        assert all(
            table.device.type == 'cpu'
            for node in nodes for table in self._table_tensors(node))
        exact_lower = (
            bounded_inputs[0].ptb.x_L.pow(3).to(torch.float64)
            + bounded_inputs[1].ptb.x_L.pow(5))
        exact_upper = (
            bounded_inputs[0].ptb.x_U.pow(3).to(torch.float64)
            + bounded_inputs[1].ptb.x_U.pow(5))
        rounding_margin = 8 * torch.finfo(torch.float32).eps
        assert (lower <= exact_lower + rounding_margin).all()
        assert (upper >= exact_upper - rounding_margin).all()


class TestPowAlphaOptimization:
    """Behavior specific to alpha-CROWN tangent optimization."""

    @pytest.mark.parametrize('exponent,x_low,x_high', NONINTEGER_ALPHA_SWEEP_CASES)
    def test_alpha_crown_soundness_sweep(self, exponent, x_low, x_high):
        assert check_monotone_pow_soundness(
            exponent, x_low, x_high, method='alpha-CROWN',
            optimize_iterations=20), \
            f'alpha-CROWN not sound for x^{exponent} on [{x_low}, {x_high}]'

    def test_alpha_crown_tighter_than_crown(self):
        for exponent, x_low, x_high in [(1.5, 1, 4), (0.5, 1, 4), (2.5, 0, 3)]:
            input_lb = torch.ones(1, 10) * x_low
            input_ub = torch.ones(1, 10) * x_high
            _, crown_lb, crown_ub = compute_bounds(exponent, input_lb, input_ub, 'CROWN')
            _, alpha_lb, alpha_ub = compute_bounds(
                exponent, input_lb, input_ub, 'alpha-CROWN', optimize_iterations=20)

            assert alpha_lb.min().item() >= crown_lb.min().item() - 1e-5, \
                f'alpha-CROWN lb looser than CROWN for x^{exponent}'
            assert alpha_ub.max().item() <= crown_ub.max().item() + 1e-5, \
                f'alpha-CROWN ub looser than CROWN for x^{exponent}'

    def test_alpha_crown_optimization_converges(self):
        for exponent, x_low, x_high in [(1.5, 1, 4), (0.5, 1, 4), (2.5, 0, 3)]:
            input_lb = torch.ones(1, 10) * x_low
            input_ub = torch.ones(1, 10) * x_high
            results = {}
            for iters in [5, 50]:
                _, lb, ub = compute_bounds(
                    exponent, input_lb, input_ub, 'alpha-CROWN',
                    optimize_iterations=iters)
                results[iters] = (lb.min().item(), ub.max().item())

            assert results[50][0] >= results[5][0] - 1e-5, \
                f'x^{exponent}: 50-iter lb worse than 5-iter'
            assert results[50][1] <= results[5][1] + 1e-5, \
                f'x^{exponent}: 50-iter ub worse than 5-iter'


class TestPowComposedModels:
    """Sampling-based checks where Pow is not directly applied to the input."""

    def test_alpha_crown_multi_layer_convex(self):
        class LinearPowModel(nn.Module):
            def __init__(self, exponent):
                super().__init__()
                self.linear = nn.Linear(10, 10, bias=True)
                self.exponent = exponent
                nn.init.uniform_(self.linear.weight, 0.01, 0.1)
                nn.init.uniform_(self.linear.bias, 0.5, 1.0)

            def forward(self, x):
                return torch.pow(self.linear(x), self.exponent)

        for exponent in [1.5, 2.5]:
            self._check_alpha_crown_by_sampling(LinearPowModel, exponent)

    def test_alpha_crown_multi_layer_concave(self):
        class LinearPowModel(nn.Module):
            def __init__(self, exponent):
                super().__init__()
                self.linear = nn.Linear(10, 10, bias=True)
                self.exponent = exponent
                nn.init.uniform_(self.linear.weight, 0.01, 0.1)
                nn.init.uniform_(self.linear.bias, 0.5, 1.0)

            def forward(self, x):
                return torch.pow(self.linear(x), self.exponent)

        for exponent in [0.3, 0.5, 0.7]:
            self._check_alpha_crown_by_sampling(LinearPowModel, exponent)

    def test_alpha_crown_multi_layer_zero_lower(self):
        class ReLUPowModel(nn.Module):
            def __init__(self, exponent):
                super().__init__()
                self.linear = nn.Linear(10, 10, bias=True)
                self.relu = nn.ReLU()
                self.exponent = exponent
                nn.init.uniform_(self.linear.weight, -0.2, 0.2)
                nn.init.zeros_(self.linear.bias)

            def forward(self, x):
                return torch.pow(self.relu(self.linear(x)), self.exponent)

        for exponent in [0.5, 1.5, 2.5]:
            self._check_alpha_crown_by_sampling(
                ReLUPowModel, exponent, input_lb=torch.zeros(1, 10),
                input_ub=torch.ones(1, 10), center=torch.ones(1, 10) * 0.5)

    @staticmethod
    def _check_alpha_crown_by_sampling(model_cls, exponent, input_lb=None,
                                       input_ub=None, center=None):
        torch.manual_seed(42)
        model = model_cls(exponent)
        if input_lb is None:
            input_lb = torch.ones(1, 10)
            input_ub = torch.ones(1, 10) * 3.0
            center = torch.ones(1, 10) * 2.0

        bm = BoundedModule(model, center)
        bm.set_bound_opts({
            'optimize_bound_args': {'iteration': 50, 'lr_alpha': 0.1}
        })
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
        ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)
        lb, ub = bm.compute_bounds(x=(ptb_data,), method='alpha-CROWN')

        torch.manual_seed(0)
        samples = input_lb + (input_ub - input_lb) * torch.rand(1000, 10)
        with torch.no_grad():
            true_values = model(samples)
        assert lb.min().item() <= true_values.min().item() + 1e-4, \
            f'Multi-layer x^{exponent}: lb not sound'
        assert ub.max().item() >= true_values.max().item() - 1e-4, \
            f'Multi-layer x^{exponent}: ub not sound'


class TestPowErrors:
    """Unsupported-domain checks."""

    @pytest.mark.parametrize('method', ['IBP', 'CROWN', 'forward'])
    def test_negative_input_raises_error(self, method):
        model = PowModel(0.5)
        bounded_model = BoundedModule(model, torch.ones(1, 10))
        input_lb = torch.ones(1, 10) * -1
        input_ub = torch.ones(1, 10) * 4
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
        ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)

        with pytest.raises(
                OperatorDomainError, match='non-negative base') as error:
            bounded_model.compute_bounds(x=(ptb_data,), method=method)
        assert error.value.operator == 'pow'

    @pytest.mark.parametrize('method', ['IBP', 'CROWN', 'forward'])
    def test_negative_exponent_raises_error(self, method):
        model = PowModel(-0.5)
        bounded_model = BoundedModule(model, torch.ones(1, 10) * 2)
        input_lb = torch.ones(1, 10)
        input_ub = torch.ones(1, 10) * 4
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
        ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)

        with pytest.raises(
                OperatorCapabilityError,
                match='does not support exponent') as error:
            bounded_model.compute_bounds(x=(ptb_data,), method=method)
        assert error.value.operator == 'pow'

    @pytest.mark.parametrize('method', ['IBP', 'CROWN', 'forward'])
    def test_perturbed_exponent_raises_capability_error(self, method):
        base = torch.ones(1, 10) * 2
        exponent_lower = torch.ones(1)
        exponent_upper = torch.ones(1) * 2
        exponent = BoundedTensor(
            (exponent_lower + exponent_upper) / 2,
            PerturbationLpNorm(
                norm=float('inf'),
                x_L=exponent_lower,
                x_U=exponent_upper,
            ),
        )
        bounded_model = BoundedModule(
            VariablePowModel(), (base, exponent))

        with pytest.raises(OperatorCapabilityError) as error:
            bounded_model.compute_bounds(x=(base, exponent), method=method)
        assert error.value.operator == 'pow'

    @pytest.mark.parametrize('exponent', [torch.inf, -torch.inf])
    @pytest.mark.parametrize('method', ['IBP', 'CROWN', 'forward'])
    def test_nonfinite_exponent_raises_capability_error(
            self, exponent, method):
        input_lb = torch.ones(1, 10)
        input_ub = torch.ones(1, 10) * 2

        with pytest.raises(
                OperatorCapabilityError, match='finite exponent') as error:
            compute_bounds(exponent, input_lb, input_ub, method=method)
        assert error.value.operator == 'pow'

    @pytest.mark.parametrize('method', ['IBP', 'CROWN', 'forward'])
    def test_nan_exponent_violates_bound_invariant(self, method):
        input_lb = torch.ones(1, 10)
        input_ub = torch.ones(1, 10) * 2

        with pytest.raises(RuntimeError, match='NaN detected') as error:
            compute_bounds(torch.nan, input_lb, input_ub, method=method)
        assert not isinstance(error.value, OperatorCapabilityError)

    @pytest.mark.parametrize('method', ['CROWN', 'forward'])
    @pytest.mark.parametrize(
        'exponent,operator', [(0.5, 'pow'), (2, 'square'), (3, 'pow')])
    def test_unbounded_nonlinear_base_raises_capability_error(
            self, method, exponent, operator):
        center = torch.ones(1, 10)
        input_lb = torch.zeros_like(center)
        input_ub = torch.full_like(center, torch.inf)
        ptb_data = BoundedTensor(
            center,
            PerturbationLpNorm(
                norm=float('inf'), x_L=input_lb, x_U=input_ub))
        bounded_model = BoundedModule(PowModel(exponent), center)

        with pytest.raises(
                OperatorCapabilityError, match='finite .* bounds') as error:
            bounded_model.compute_bounds(x=(ptb_data,), method=method)
        assert error.value.operator == operator


class TestPowRef(TestCase):
    """Reference-based regression test for representative BoundPow outputs."""

    def __init__(self, methodName='runTest', generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName, seed=1, ref_name='pow_test_data',
            generate=generate, device=device, dtype=dtype)

    def _append_bounds(self, exponent, input_lb, input_ub, method,
                       optimize_iterations=0):
        input_lb = input_lb.to(device=self.default_device, dtype=self.default_dtype)
        input_ub = input_ub.to(device=self.default_device, dtype=self.default_dtype)
        _, lb, ub = compute_bounds(
            exponent, input_lb, input_ub, method, optimize_iterations,
            device=self.default_device)
        self.result.append((lb, ub))
        return lb, ub

    def _append_monotone_noninteger_cases(self):
        cases = [
            (0.5, 1, 4),
            (0.3, 1, 5),
            (1.5, 1, 4),
            (2.5, 1, 3),
            (0.5, 0, 4),
            (1.5, 0, 4),
        ]
        for exponent, x_low, x_high in cases:
            input_lb = torch.ones(1, 10) * x_low
            input_ub = torch.ones(1, 10) * x_high
            for method in ['IBP', 'CROWN']:
                self._append_bounds(exponent, input_lb, input_ub, method)

        for exponent, x_low, x_high in [(0.5, 1, 4), (1.5, 1, 4), (2.5, 1, 3)]:
            input_lb = torch.ones(1, 10) * x_low
            input_ub = torch.ones(1, 10) * x_high
            self._append_bounds(
                exponent, input_lb, input_ub, 'alpha-CROWN',
                optimize_iterations=20)

    def _append_even_integer_cases(self):
        input_lb, input_ub = even_integer_input_bounds()
        for exponent in EVEN_INTEGER_EXPONENTS:
            true_lb, true_ub = true_even_integer_bounds(input_lb, input_ub, exponent)
            for method in ['CROWN', 'alpha-CROWN']:
                lb, ub = self._append_bounds(
                    exponent, input_lb, input_ub, method,
                    optimize_iterations=5 if method == 'alpha-CROWN' else 0)
                assert torch.all(lb.cpu() <= true_lb + 1e-5)
                assert torch.all(ub.cpu() >= true_ub - 1e-4)

    def test(self):
        self.result = []
        self._append_monotone_noninteger_cases()
        self._append_even_integer_cases()
        self.check()


if __name__ == '__main__':
    testcase = TestPowRef(generate=False)
    testcase.setUp()
    testcase.test()

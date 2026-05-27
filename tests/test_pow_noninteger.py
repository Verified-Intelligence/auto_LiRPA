"""Test non-integer exponent support for BoundPow."""
import torch
import torch.nn as nn
import pytest
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class PowModel(nn.Module):
    """Simple model that computes x^exponent."""
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent

    def forward(self, x):
        return torch.pow(x, self.exponent)


def check_bounds_soundness(exponent, x_low, x_high, method='CROWN', atol=1e-5,
                           optimize_iterations=0):
    """
    Test that computed bounds are sound (contain all true values).

    Preconditions: x_low >= 0 and exponent > 0.  Under these conditions x^y is
    monotonically increasing, so the true lower/upper bounds are simply the
    function values at the interval endpoints.  Do NOT use this helper for
    negative x or non-positive exponents, where this monotonicity assumption
    does not hold.

    Args:
        exponent: The power exponent (must be > 0)
        x_low: Lower bound on input (must be >= 0)
        x_high: Upper bound on input
        method: Bound computation method ('IBP', 'CROWN', or 'alpha-CROWN')
        atol: Absolute tolerance for comparison
        optimize_iterations: Number of optimization iterations for alpha-CROWN

    Returns:
        True if bounds are sound, False otherwise
    """
    assert x_low >= 0, f"x_low must be >= 0 for monotonicity; got {x_low}"
    assert exponent > 0, f"exponent must be > 0; got {exponent}"
    model = PowModel(exponent)
    x = torch.ones(1, 10) * ((x_low + x_high) / 2)
    bounded_model = BoundedModule(model, x)

    if optimize_iterations > 0:
        bounded_model.set_bound_opts({
            'optimize_bound_args': {'iteration': optimize_iterations, 'lr_alpha': 0.1}
        })

    input_lb = torch.ones(1, 10) * x_low
    input_ub = torch.ones(1, 10) * x_high
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
    ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)

    # Compute bounds
    lb, ub = bounded_model.compute_bounds(x=(ptb_data,), method=method)

    # x^y is monotonically increasing for x >= 0, y > 0, so exact bounds are at endpoints
    true_lb = x_low ** exponent
    true_ub = x_high ** exponent

    # Check soundness: computed lower bound <= true lower, computed upper bound >= true upper
    lb_sound = lb.min().item() <= true_lb + atol
    ub_sound = ub.max().item() >= true_ub - atol

    return lb_sound and ub_sound


class TestPowNonInteger:
    """Test cases for non-integer exponents in BoundPow."""

    @pytest.mark.parametrize("method", ["IBP", "CROWN"])
    @pytest.mark.parametrize("exponent,x_low,x_high", [
        # Concave cases (0 < y < 1)
        (0.5, 1, 4),
        (0.5, 0.1, 10),
        (0.3, 1, 5),
        (0.7, 2, 8),
        (0.9, 0.5, 3),
        # Convex cases (y > 1)
        (1.5, 1, 4),
        (1.5, 0.5, 3),
        (2.5, 1, 3),
        (3.7, 0.5, 2),
        (1.1, 1, 5),
    ])
    def test_noninteger_exponent_soundness(self, exponent, x_low, x_high, method):
        """Test that bounds for non-integer exponents are sound."""
        assert check_bounds_soundness(exponent, x_low, x_high, method=method), \
            f"Bounds not sound for x^{exponent} on [{x_low}, {x_high}] with method {method}"

    def test_zero_lower_bound_convex(self):
        """Test that x^y with y > 1 works when lower bound includes zero."""
        assert check_bounds_soundness(2.5, 0, 3, method='IBP')
        assert check_bounds_soundness(2.5, 0, 3, method='CROWN')
        assert check_bounds_soundness(1.5, 0, 4, method='IBP')
        assert check_bounds_soundness(1.5, 0, 4, method='CROWN')

    def test_zero_lower_bound_concave(self):
        """Test that x^y with 0 < y < 1 works when lower bound includes zero."""
        assert check_bounds_soundness(0.5, 0, 4, method='IBP')
        assert check_bounds_soundness(0.5, 0, 4, method='CROWN')

    @pytest.mark.parametrize("exponent,x_low,x_high", [
        # Concave cases (0 < y < 1)
        (0.5, 1, 4),
        (0.3, 1, 5),
        # Convex cases (y > 1)
        (1.5, 1, 4),
        (2.5, 1, 3),
    ])
    def test_alpha_crown_soundness(self, exponent, x_low, x_high):
        """Test that alpha-CROWN bounds are sound and at least as tight as CROWN."""
        assert check_bounds_soundness(exponent, x_low, x_high,
                                      method='alpha-CROWN', optimize_iterations=20)

    def test_alpha_crown_zero_lower_bound(self):
        """Test alpha-CROWN with zero lower bound for both convex and concave."""
        # Convex (y > 1): tangent at x=0 is well-defined
        assert check_bounds_soundness(2.5, 0, 3, method='alpha-CROWN', optimize_iterations=20)
        assert check_bounds_soundness(1.5, 0, 4, method='alpha-CROWN', optimize_iterations=20)
        # Concave (0 < y < 1): derivative clamped near x=0
        assert check_bounds_soundness(0.5, 0, 4, method='alpha-CROWN', optimize_iterations=20)

    def test_alpha_crown_tighter_than_crown(self):
        """Test that alpha-CROWN produces bounds at least as tight as CROWN."""
        for exponent, x_low, x_high in [(1.5, 1, 4), (0.5, 1, 4), (2.5, 0, 3)]:
            model = PowModel(exponent)
            x = torch.ones(1, 10) * ((x_low + x_high) / 2)

            # CROWN bounds
            bm_crown = BoundedModule(model, x)
            input_lb = torch.ones(1, 10) * x_low
            input_ub = torch.ones(1, 10) * x_high
            ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
            ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)
            crown_lb, crown_ub = bm_crown.compute_bounds(x=(ptb_data,), method='CROWN')

            # alpha-CROWN bounds
            bm_alpha = BoundedModule(model, x)
            bm_alpha.set_bound_opts({
                'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}
            })
            ptb2 = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
            ptb_data2 = BoundedTensor((input_lb + input_ub) / 2, ptb2)
            alpha_lb, alpha_ub = bm_alpha.compute_bounds(x=(ptb_data2,), method='alpha-CROWN')

            # alpha-CROWN should be at least as tight (higher lb, lower ub)
            assert alpha_lb.min().item() >= crown_lb.min().item() - 1e-5, \
                f"alpha-CROWN lb looser than CROWN for x^{exponent} on [{x_low}, {x_high}]"
            assert alpha_ub.max().item() <= crown_ub.max().item() + 1e-5, \
                f"alpha-CROWN ub looser than CROWN for x^{exponent} on [{x_low}, {x_high}]"

    @pytest.mark.parametrize("exponent,x_low,x_high", [
        # Wide sweep of exponents
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
        # Narrow intervals (alpha optimization matters more)
        (0.5, 2, 2.5),
        (1.5, 3, 3.2),
        (2.5, 1, 1.1),
        # Wide intervals
        (0.5, 0.01, 100),
        (1.5, 0.1, 10),
        (2.5, 0.01, 10),
    ])
    def test_alpha_crown_extensive_soundness(self, exponent, x_low, x_high):
        """Extensive soundness test for alpha-CROWN across many exponent/range combos."""
        assert check_bounds_soundness(exponent, x_low, x_high,
                                      method='alpha-CROWN', optimize_iterations=20), \
            f"alpha-CROWN not sound for x^{exponent} on [{x_low}, {x_high}]"

    def test_alpha_crown_optimization_converges(self):
        """Test that more iterations produce tighter (or equal) bounds."""
        for exponent, x_low, x_high in [(1.5, 1, 4), (0.5, 1, 4), (2.5, 0, 3)]:
            model = PowModel(exponent)
            x = torch.ones(1, 10) * ((x_low + x_high) / 2)
            input_lb = torch.ones(1, 10) * x_low
            input_ub = torch.ones(1, 10) * x_high

            results = {}
            for iters in [5, 50]:
                bm = BoundedModule(model, x)
                bm.set_bound_opts({
                    'optimize_bound_args': {'iteration': iters, 'lr_alpha': 0.1}
                })
                ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
                ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)
                lb, ub = bm.compute_bounds(x=(ptb_data,), method='alpha-CROWN')
                results[iters] = (lb.min().item(), ub.max().item())

            # More iterations should give tighter or equal bounds
            assert results[50][0] >= results[5][0] - 1e-5, \
                f"x^{exponent}: 50-iter lb ({results[50][0]}) worse than 5-iter ({results[5][0]})"
            assert results[50][1] <= results[5][1] + 1e-5, \
                f"x^{exponent}: 50-iter ub ({results[50][1]}) worse than 5-iter ({results[5][1]})"

    def test_alpha_crown_multi_layer_convex(self):
        """Test alpha-CROWN with Linear -> Pow (convex) multi-layer network."""
        class LinearPowModel(nn.Module):
            def __init__(self, exponent):
                super().__init__()
                self.linear = nn.Linear(10, 10, bias=True)
                self.exponent = exponent
                # Initialize weights to ensure positive output for Pow
                nn.init.uniform_(self.linear.weight, 0.01, 0.1)
                nn.init.uniform_(self.linear.bias, 0.5, 1.0)

            def forward(self, x):
                return torch.pow(self.linear(x), self.exponent)

        for exponent in [1.5, 2.5]:
            torch.manual_seed(42)
            model = LinearPowModel(exponent)
            x = torch.ones(1, 10) * 2.0
            bm = BoundedModule(model, x)
            bm.set_bound_opts({
                'optimize_bound_args': {'iteration': 50, 'lr_alpha': 0.1}
            })
            input_lb = torch.ones(1, 10) * 1.0
            input_ub = torch.ones(1, 10) * 3.0
            ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
            ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)
            lb, ub = bm.compute_bounds(x=(ptb_data,), method='alpha-CROWN')

            # Verify soundness by sampling
            torch.manual_seed(0)
            samples = input_lb + (input_ub - input_lb) * torch.rand(1000, 10)
            with torch.no_grad():
                true_values = model(samples)
            assert lb.min().item() <= true_values.min().item() + 1e-4, \
                f"Multi-layer x^{exponent}: lb not sound"
            assert ub.max().item() >= true_values.max().item() - 1e-4, \
                f"Multi-layer x^{exponent}: ub not sound"

    def test_alpha_crown_multi_layer_concave(self):
        """Test alpha-CROWN with Linear -> Pow (concave) multi-layer network."""
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
            torch.manual_seed(42)
            model = LinearPowModel(exponent)
            x = torch.ones(1, 10) * 2.0
            bm = BoundedModule(model, x)
            bm.set_bound_opts({
                'optimize_bound_args': {'iteration': 50, 'lr_alpha': 0.1}
            })
            input_lb = torch.ones(1, 10) * 1.0
            input_ub = torch.ones(1, 10) * 3.0
            ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
            ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)
            lb, ub = bm.compute_bounds(x=(ptb_data,), method='alpha-CROWN')

            # Verify soundness by sampling
            torch.manual_seed(0)
            samples = input_lb + (input_ub - input_lb) * torch.rand(1000, 10)
            with torch.no_grad():
                true_values = model(samples)
            assert lb.min().item() <= true_values.min().item() + 1e-4, \
                f"Multi-layer x^{exponent}: lb not sound"
            assert ub.max().item() >= true_values.max().item() - 1e-4, \
                f"Multi-layer x^{exponent}: ub not sound"

    def test_alpha_crown_multi_layer_zero_lower(self):
        """Test alpha-CROWN multi-layer where Pow input range includes zero."""
        class ReLUPowModel(nn.Module):
            """ReLU guarantees x >= 0, then Pow."""
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
            torch.manual_seed(42)
            model = ReLUPowModel(exponent)
            x = torch.ones(1, 10) * 0.5
            bm = BoundedModule(model, x)
            bm.set_bound_opts({
                'optimize_bound_args': {'iteration': 50, 'lr_alpha': 0.1}
            })
            input_lb = torch.zeros(1, 10)
            input_ub = torch.ones(1, 10)
            ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
            ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)
            lb, ub = bm.compute_bounds(x=(ptb_data,), method='alpha-CROWN')

            # Verify soundness by sampling
            torch.manual_seed(0)
            samples = input_lb + (input_ub - input_lb) * torch.rand(1000, 10)
            with torch.no_grad():
                true_values = model(samples)
            assert lb.min().item() <= true_values.min().item() + 1e-4, \
                f"ReLU+Pow x^{exponent}: lb not sound"
            assert ub.max().item() >= true_values.max().item() - 1e-4, \
                f"ReLU+Pow x^{exponent}: ub not sound"

    def test_negative_input_raises_error(self):
        """Test that non-integer exponent with x < 0 raises an error."""
        model = PowModel(0.5)
        x = torch.ones(1, 10)
        bounded_model = BoundedModule(model, x)

        # Input bounds that include negative values
        input_lb = torch.ones(1, 10) * -1
        input_ub = torch.ones(1, 10) * 4
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
        ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)

        with pytest.raises(AssertionError, match="must be non-negative"):
            bounded_model.compute_bounds(x=(ptb_data,), method='IBP')

    def test_negative_exponent_raises_error(self):
        """Test that negative exponent raises an error."""
        model = PowModel(-0.5)
        x = torch.ones(1, 10) * 2  # positive input
        bounded_model = BoundedModule(model, x)

        input_lb = torch.ones(1, 10) * 1
        input_ub = torch.ones(1, 10) * 4
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
        ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)

        with pytest.raises(AssertionError, match="must be positive"):
            bounded_model.compute_bounds(x=(ptb_data,), method='IBP')


class TestPowNonIntegerRef(TestCase):
    """Reference-based regression test for non-integer BoundPow."""

    def __init__(self, methodName='runTest', generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName, seed=1, ref_name='pow_noninteger_test_data',
            generate=generate, device=device, dtype=dtype)

    def _compute_bounds(self, exponent, x_low, x_high, method,
                        optimize_iterations=0):
        """Compute bounds and append to self.result."""
        model = PowModel(exponent).to(
            device=self.default_device, dtype=self.default_dtype)
        x = torch.ones(1, 10) * ((x_low + x_high) / 2)
        bounded_model = BoundedModule(model, x, device=self.default_device)

        if optimize_iterations > 0:
            bounded_model.set_bound_opts({
                'optimize_bound_args': {
                    'iteration': optimize_iterations, 'lr_alpha': 0.1}
            })

        input_lb = torch.ones(1, 10) * x_low
        input_ub = torch.ones(1, 10) * x_high
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=input_lb, x_U=input_ub)
        ptb_data = BoundedTensor((input_lb + input_ub) / 2, ptb)
        lb, ub = bounded_model.compute_bounds(x=(ptb_data,), method=method)
        self.result.append((lb, ub))

    def test(self):
        self.result = []

        test_cases = [
            # (exponent, x_low, x_high)
            (0.5, 1, 4),    # concave
            (0.3, 1, 5),    # concave
            (1.5, 1, 4),    # convex
            (2.5, 1, 3),    # convex
            (0.5, 0, 4),    # concave, zero lower bound
            (1.5, 0, 4),    # convex, zero lower bound
        ]

        for exponent, x_low, x_high in test_cases:
            for method in ['IBP', 'CROWN']:
                self._compute_bounds(exponent, x_low, x_high, method)

        # Also test alpha-CROWN
        for exponent, x_low, x_high in [(0.5, 1, 4), (1.5, 1, 4), (2.5, 1, 3)]:
            self._compute_bounds(exponent, x_low, x_high,
                                 'alpha-CROWN', optimize_iterations=20)

        self.check()


if __name__ == '__main__':
    # Change to generate=True when generating reference results
    testcase = TestPowNonIntegerRef(generate=False)
    testcase.setUp()
    testcase.test()

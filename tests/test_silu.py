import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators.silu import (
    SiLU, BoundSiLU, silu, dsilu, d2silu,
    _SILU_X_MIN, _SILU_INFLECTION,
)
from auto_LiRPA.parse_graph import parse_module
from auto_LiRPA.perturbations import PerturbationLpNorm


class SiLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = SiLU()

    def forward(self, x):
        return self.act(x)


def _bounded_model():
    model = SiLUModel().eval()
    x = torch.zeros(1, 1)
    return BoundedModule(model, x, device='cpu')


def _interval_bounds(method, lower, upper):
    bounded_model = _bounded_model()
    x_lb = torch.tensor([[lower]], dtype=torch.float32)
    x_ub = torch.tensor([[upper]], dtype=torch.float32)
    x_center = (x_lb + x_ub) / 2
    ptb = PerturbationLpNorm(norm=float('inf'), eps=None, x_L=x_lb, x_U=x_ub)
    bounded_x = BoundedTensor(x_center, ptb)
    lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method=method)
    return lb.item(), ub.item()


def _sample_reference(lower, upper, num_samples=50001):
    xs = torch.linspace(lower, upper, steps=num_samples)
    ys = F.silu(xs)
    return ys.min().item(), ys.max().item()


def test_silu_custom_module_matches_torch_silu():
    x = torch.linspace(-8, 8, steps=257)
    model = SiLUModel().eval()
    assert torch.allclose(model(x), F.silu(x), atol=1e-7, rtol=1e-7)


def test_silu_symbolic_op_is_registered():
    model = SiLUModel().eval()
    x = torch.randn(1, 4)
    ops, _, _, _ = parse_module(model, (x,))
    assert [node.op for node in ops] == ['custom::SiLU']


def test_silu_first_derivative_matches_autograd():
    x = torch.linspace(-6, 6, steps=101, requires_grad=True)
    y = F.silu(x)
    grad = torch.autograd.grad(y.sum(), x)[0]
    assert torch.allclose(dsilu(x.detach()), grad.detach(), atol=1e-6, rtol=1e-5)


def test_silu_second_derivative_matches_autograd():
    x = torch.linspace(-4, 4, steps=81, requires_grad=True)
    y = F.silu(x)
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    grad2 = torch.autograd.grad(grad.sum(), x)[0]
    assert torch.allclose(d2silu(x.detach()), grad2.detach(), atol=1e-5, rtol=1e-4)


def test_silu_known_extreme_and_inflection_points():
    x_min = torch.tensor(_SILU_X_MIN)
    inflection = torch.tensor(_SILU_INFLECTION)
    assert abs(dsilu(x_min).item()) < 1e-6
    assert abs(d2silu(-inflection).item()) < 1e-6
    assert abs(d2silu(inflection).item()) < 1e-6
    assert d2silu(torch.tensor(0.0)).item() > 0
    assert d2silu(torch.tensor(-3.0)).item() < 0
    assert d2silu(torch.tensor(3.0)).item() < 0


def test_silu_bound_node_split_mask_focuses_on_hard_ranges():
    node = BoundSiLU(attr={'device': 'cpu'}, inputs=[], output_index=0, options={})
    lower = torch.tensor([[-6.0, -3.0, 3.0, -0.001]])
    upper = torch.tensor([[-5.0, 3.0, 5.0, 0.001]])
    mask = node.get_split_mask(lower, upper, input_index=0)
    expected = torch.tensor([[False, True, False, False]])
    assert torch.equal(mask, expected)


def test_silu_ibp_uses_internal_minimum():
    lb, ub = _interval_bounds('IBP', -2.0, 0.0)
    expected_lb = silu(torch.tensor(_SILU_X_MIN)).item()
    expected_ub = max(silu(torch.tensor(-2.0)).item(), silu(torch.tensor(0.0)).item())
    assert abs(lb - expected_lb) < 1e-5
    assert abs(ub - expected_ub) < 1e-5


def test_silu_ibp_matches_monotonic_endpoint_cases():
    dec_lb, dec_ub = _interval_bounds('IBP', -5.0, -2.0)
    assert abs(dec_lb - silu(torch.tensor(-2.0)).item()) < 1e-5
    assert abs(dec_ub - silu(torch.tensor(-5.0)).item()) < 1e-5

    inc_lb, inc_ub = _interval_bounds('IBP', -1.0, 3.0)
    assert abs(inc_lb - silu(torch.tensor(-1.0)).item()) < 1e-5
    assert abs(inc_ub - silu(torch.tensor(3.0)).item()) < 1e-5


def test_silu_crown_and_ibp_run_on_vector_input():
    model = SiLUModel().eval()
    x = torch.randn(1, 8)
    bounded_model = BoundedModule(model, x, device='cpu')
    ptb = PerturbationLpNorm(norm=float('inf'), eps=0.1)
    bounded_x = BoundedTensor(x, ptb)

    for method in ['IBP', 'CROWN']:
        lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method=method)
        assert lb.shape == x.shape
        assert ub.shape == x.shape
        assert torch.all(lb <= ub)


def test_silu_crown_bounds_cover_sampled_truth_on_representative_intervals():
    intervals = [
        (-5.0, -3.0),   # concave left
        (-1.0, 1.0),    # convex middle
        (3.0, 5.0),     # concave right
        (-5.0, 5.0),    # cross both inflections
        (-5.0, 2.0),    # cross minimum and left inflection
        (-2.5, 2.5),    # cross both inflections tightly
    ]
    for lower, upper in intervals:
        lb, ub = _interval_bounds('CROWN', lower, upper)
        ref_lb, ref_ub = _sample_reference(lower, upper)
        assert lb <= ref_lb + 1e-5, (lower, upper, lb, ref_lb)
        assert ub >= ref_ub - 1e-5, (lower, upper, ub, ref_ub)


def test_silu_ibp_bounds_cover_sampled_truth_on_representative_intervals():
    intervals = [
        (-5.0, -3.0),
        (-2.0, 0.0),
        (-1.0, 1.0),
        (0.0, 4.0),
        (-5.0, 5.0),
    ]
    for lower, upper in intervals:
        lb, ub = _interval_bounds('IBP', lower, upper)
        ref_lb, ref_ub = _sample_reference(lower, upper)
        assert abs(lb - ref_lb) < 5e-5, (lower, upper, lb, ref_lb)
        assert abs(ub - ref_ub) < 5e-5, (lower, upper, ub, ref_ub)


def test_silu_tiny_interval_is_numerically_stable():
    center = -0.75
    radius = 1e-6
    lb, ub = _interval_bounds('CROWN', center - radius, center + radius)
    true_value = silu(torch.tensor(center)).item()
    assert lb <= true_value <= ub
    assert ub - lb < 1e-4


def test_silu_wide_intervals_remain_valid():
    for lower, upper in [(-20.0, 20.0), (-30.0, -10.0), (10.0, 30.0)]:
        lb, ub = _interval_bounds('CROWN', lower, upper)
        ref_lb, ref_ub = _sample_reference(lower, upper, num_samples=20001)
        assert lb <= ref_lb + 1e-5
        assert ub >= ref_ub - 1e-5

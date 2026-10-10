""" Test different Perturbation classes"""
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import (
    PerturbationL0Norm,
    PerturbationLinear,
    PerturbationLpNorm,
)
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


BATCH = 2
IN_DIM = 3
OUT_DIM = 4


class ToyModel(nn.Module):
    """Small model with two MatMuls and ReLU."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OUT_DIM, 8)
        self.fc2 = nn.Linear(8, OUT_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SubtractInputs(nn.Module):
    def forward(self, first, second):
        return first - second


class SumDifferentSizedInputs(nn.Module):
    def forward(self, first, second):
        return first.sum(dim=-1, keepdim=True) + 3 * second


def _bounded_box(lower, upper):
    return BoundedTensor(
        (lower + upper) / 2,
        PerturbationLpNorm(norm=torch.inf, x_L=lower, x_U=upper))


class TestPerturbation(TestCase):
    """
    Tests for:
    - PerturbationLinear
    - PerturbationLpNorm
    """
    def __init__(self, methodName='runTest', seed=1, generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName, seed, 'test_perturbation_data', generate, device=device, dtype=dtype)

    def test(self):
        device = self.default_device
        dtype = self.default_dtype

        model = ToyModel().to(device=device, dtype=dtype)

        # Prepare base input interval
        input_lb = torch.rand(BATCH, IN_DIM, device=device, dtype=dtype)
        input_ub = input_lb + torch.rand_like(input_lb)    # ensure ub > lb

        self.result = []

        # =================================================================
        # Test PerturbationLinear
        # =================================================================
        # Build A matrices
        lower_A = torch.randn(BATCH, OUT_DIM, IN_DIM, device=device, dtype=dtype)
        upper_A = lower_A + torch.rand_like(lower_A)
        # biases
        lower_b = torch.randn(BATCH, OUT_DIM, device=device, dtype=dtype)
        upper_b = lower_b + torch.rand_like(lower_b)

        # Manual concretization
        mid = ((input_lb + input_ub) / 2.0).unsqueeze(-1)   # (B, IN_DIM, 1)
        diff = ((input_ub - input_lb) / 2.0).unsqueeze(-1)   # (B, IN_DIM, 1)

        manual_L = (lower_A @ mid - torch.abs(lower_A) @ diff).squeeze(-1) + lower_b
        manual_U = (upper_A @ mid + torch.abs(upper_A) @ diff).squeeze(-1) + upper_b
        assert (manual_L < manual_U).all(), "Invalid manual bounds construction."

        ptb_linear = PerturbationLinear(
            lower_A=lower_A, upper_A=upper_A, lower_b=lower_b, upper_b=upper_b,
            input_lb=input_lb, input_ub=input_ub,
            x_L=manual_L, x_U=manual_U
        )
        bounded_x = BoundedTensor((manual_L + manual_U) / 2, ptb_linear)
        lirpa_model = BoundedModule(model, bounded_x)
        lb_linear, ub_linear = lirpa_model.compute_bounds(bounded_x, method='backward')
        assert (lb_linear <= ub_linear).all(), "Invalid bounds from PerturbationLinear."
        self.result.append((lb_linear, ub_linear))


        # =================================================================
        # Test PerturbationLpNorm
        # =================================================================
        # We directly use manual concretization here for testing
        ptb_linf = PerturbationLpNorm(x_L=manual_L, x_U=manual_U)
        bounded_x = BoundedTensor((manual_L + manual_U) / 2, ptb_linf)
        lirpa_model = BoundedModule(model, bounded_x)
        lb_linf, ub_linf = lirpa_model.compute_bounds(bounded_x, method='backward')
        assert (lb_linf <= ub_linf).all(), "Invalid bounds from PerturbationLpNorm."
        self.result.append((lb_linf, ub_linf))

        # Notice that with the same x_L and x_U, PerturbationLinear should give
        # tighter bounds than PerturbationLpNorm. This is because
        # PerturbationLinear uses additional information (A matrices and biases).
        assert (lb_linear >= lb_linf).all() and (ub_linear <= ub_linf).all(
        ), "PerturbationLinear should give tighter bounds than PerturbationLpNorm."

        self.check()


def test_sparse_linf_forward_preserves_float64_bounds():
    center = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    lower = center.clone()
    upper = center.clone()
    upper[0, 0] += 0.1
    bounded_model = BoundedModule(nn.Identity().double(), center)
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(norm=float('inf'), x_L=lower, x_U=upper),
    )

    actual_lower, actual_upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='forward')

    torch.testing.assert_close(actual_lower, lower)
    torch.testing.assert_close(actual_upper, upper)


def test_l0_forward_preserves_float64_bounds():
    center = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    bounded_model = BoundedModule(nn.Identity().double(), center)
    bounded_input = BoundedTensor(center, PerturbationL0Norm(eps=1))

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='forward')

    torch.testing.assert_close(lower, torch.zeros_like(center))
    torch.testing.assert_close(upper, torch.ones_like(center))


def test_forward_concretization_uses_each_input_root_coefficients():
    first_lower = torch.tensor([[1.0]])
    first_upper = torch.tensor([[2.0]])
    second_lower = torch.tensor([[3.0]])
    second_upper = torch.tensor([[5.0]])
    bounded_model = BoundedModule(
        SubtractInputs(),
        ((first_lower + first_upper) / 2,
         (second_lower + second_upper) / 2),
        device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(_bounded_box(first_lower, first_upper),
           _bounded_box(second_lower, second_upper)),
        method='forward')

    torch.testing.assert_close(lower, torch.tensor([[-4.0]]))
    torch.testing.assert_close(upper, torch.tensor([[-1.0]]))


def test_forward_concretization_slices_different_sized_input_roots():
    first_lower = torch.tensor([[1.0, -1.0]])
    first_upper = torch.tensor([[2.0, 1.0]])
    second_lower = torch.tensor([[2.0]])
    second_upper = torch.tensor([[4.0]])
    bounded_model = BoundedModule(
        SumDifferentSizedInputs(),
        ((first_lower + first_upper) / 2,
         (second_lower + second_upper) / 2),
        device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(_bounded_box(first_lower, first_upper),
           _bounded_box(second_lower, second_upper)),
        method='forward')

    torch.testing.assert_close(lower, torch.tensor([[6.0]]))
    torch.testing.assert_close(upper, torch.tensor([[15.0]]))


if __name__ == '__main__':
    testcase = TestPerturbation(generate=False)
    testcase.test()

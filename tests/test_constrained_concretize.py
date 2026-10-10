"""Test optimized bounds in simple_verification."""
import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.concretize_func import (
    constraints_solving,
    sort_out_constr_batches,
)
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import eyeC

from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE

class ConstrainedConcretizeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.tensor([[1., -1.], [2., -1.]])
        self.w2 = torch.tensor([[1., -1.]])

    def forward(self, x):
        z1 = x.matmul(self.w1.t())
        hz1 = torch.nn.functional.relu(z1)
        z2 = hz1.matmul(self.w2.t())
        return z2

class TestConstrainedConcretize(TestCase):
    def __init__(self, methodName='runTest', generate=False, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName, 1, "test_constrained_concretize", generate, device=device, dtype=dtype)

    def test(self):
        model = ConstrainedConcretizeModel().to(self.default_device).to(self.default_dtype)
        # Input x.
        x = torch.tensor([[1., 1.]], dtype=self.default_dtype, device=self.default_device)
        # Lower and upper bounds of x.
        lower = torch.tensor([[-1., -2.]], dtype=self.default_dtype, device=self.default_device)
        upper = torch.tensor([[2., 1.]], dtype=self.default_dtype, device=self.default_device)

        # Wrap model with auto_LiRPA for bound computation.
        # The second parameter is for constructing the trace of the computational graph,
        # and its content is not important.

        lirpa_model = BoundedModule(model, torch.empty_like(x))
        pred = lirpa_model(x)
        print(f'Model prediction: {pred.item()}')

        # Compute bounds using LiRPA using the given lower and upper bounds.
        norm = float("inf")
        ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
        bounded_x = BoundedTensor(x, ptb)

        # Compute bounds.
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
        print(f'CROWN bounds: lower={lb.item()}, upper={ub.item()}')

        # Add a new constraint of :
        #    1*x_0 + 1*x_1 + 2 <= 0
        constraint_a = torch.tensor([[[1.0, 1.0]]], dtype=self.default_dtype, device=self.default_device)
        constraint_b = torch.tensor([[2.0]], dtype=self.default_dtype, device=self.default_device)
        constraints = (constraint_a, constraint_b)

        norm = float("inf")
        ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper, constraints=constraints)
        bounded_x = BoundedTensor(x, ptb)
        # Compute bounds.
        constrained_lb, constrained_ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
        print(f'CROWN bounds (with constraints): lower={constrained_lb.item()}, upper={constrained_ub.item()}')

        self.result = (lb, ub, constrained_lb, constrained_ub)
        self.check()


def test_float64_constrained_crown_matches_exact_box_extrema():
    center = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    constraints = (
        torch.tensor([[[1.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[2.0]], dtype=torch.float64),
    )
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(
            norm=float('inf'),
            x_L=torch.tensor([[-1.0, -2.0]], dtype=torch.float64),
            x_U=torch.tensor([[2.0, 1.0]], dtype=torch.float64),
            constraints=constraints,
        ),
    )
    bounded_model = BoundedModule(torch.nn.Identity().double(), center)

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='CROWN')

    torch.testing.assert_close(
        lower, torch.tensor([[-1.0, -2.0]], dtype=torch.float64))
    torch.testing.assert_close(
        upper, torch.tensor([[0.0, -1.0]], dtype=torch.float64))


def test_mixed_input_and_constraint_dtypes_promote_constrained_crown():
    class ToDouble(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float64)

    center = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    constraints = (
        torch.tensor([[[1.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[2.0]], dtype=torch.float64),
    )
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(
            norm=float('inf'),
            x_L=torch.tensor([[-1.0, -2.0]], dtype=torch.float32),
            x_U=torch.tensor([[2.0, 1.0]], dtype=torch.float32),
            constraints=constraints,
        ),
    )
    bounded_model = BoundedModule(ToDouble(), center)

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='CROWN')

    torch.testing.assert_close(
        lower, torch.tensor([[-1.0, -2.0]], dtype=torch.float64))
    torch.testing.assert_close(
        upper, torch.tensor([[0.0, -1.0]], dtype=torch.float64))


def test_cpu_eye_c_constraint_solving_promotes_operands():
    x_lower = torch.tensor(
        [[-1.0, -2.0], [-2.0, -1.0]], dtype=torch.float32)
    x_upper = torch.tensor(
        [[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    raw_constraints = (
        torch.tensor(
            [[[1.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[2.0], [2.0]], dtype=torch.float64),
    )
    constraints, sorted_batches = sort_out_constr_batches(
        x_lower, x_upper, raw_constraints)

    lower = constraints_solving(
        x_lower,
        x_upper,
        eyeC([2, 2, 2], x_lower.device),
        constraints,
        sign=-1.0,
        sorted_out_batches=sorted_batches,
        max_chunk_size=1,
    ).squeeze(-1)
    upper = constraints_solving(
        x_lower,
        x_upper,
        eyeC([2, 2, 2], x_lower.device),
        constraints,
        sign=1.0,
        sorted_out_batches=sorted_batches,
        max_chunk_size=1,
    ).squeeze(-1)

    torch.testing.assert_close(
        lower,
        torch.tensor([[-1.0, -2.0], [-2.0, -1.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        upper,
        torch.tensor([[0.0, -1.0], [-1.0, 0.0]], dtype=torch.float64),
    )


def test_sparse_linf_constraint_mapping_promotes_constraints_only():
    center = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    lower = center.clone()
    upper = center.clone()
    upper[0, 0] += 0.1
    constraints = (
        torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[10.0]], dtype=torch.float64),
    )
    perturbation = PerturbationLpNorm(
        norm=float('inf'), x_L=lower, x_U=upper, constraints=constraints)

    linear, _, _ = perturbation.init(center, forward=True)

    assert perturbation.constraints_sparse[0].dtype == torch.float64
    assert perturbation.constraints_sparse[1].dtype == torch.float64
    assert linear.lw.dtype == torch.float32

if __name__ == '__main__':
    testcase = TestConstrainedConcretize(generate=True)
    testcase.setUp()
    testcase.test()

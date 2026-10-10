# pylint: disable=wrong-import-position
"""Test Jacobian bounds."""
import sys
import torch
import torch.nn as nn

sys.path.append('../examples/vision')
from jacobian import compute_jacobians
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.utils import Flatten
from auto_LiRPA.jacobian import JacobianOP
from auto_LiRPA.perturbations import PerturbationLpNorm
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class TestJacobian(TestCase):
    def __init__(self, methodName='runTest', generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName, seed=1, ref_name='jacobian_test_data',
            generate=generate,
            device=device, dtype=dtype)

    def test(self):
        in_dim, linear_size = 8, 100
        model = nn.Sequential(
            Flatten(),
            nn.Linear(3*in_dim**2, linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, linear_size),
            nn.Tanh(),
            nn.Linear(linear_size, linear_size),
            nn.Sigmoid(),
            nn.Linear(linear_size, 10),
        )
        model = model.to(device=self.default_device, dtype=self.default_dtype)
        x0 = torch.randn(1, 3, in_dim, in_dim,
                         device=self.default_device, dtype=self.default_dtype)
        self.result = compute_jacobians(model, x0)
        self.check()

    def test_concat_jacobian(self):
        '''
        Test JacobianOP with Concat operation. This needs some special handling
        in auto_LiRPA to make it work properly. (See parse_graph.py for details.)
        '''
        class ConcatModule(nn.Module):
            def forward(self, x):
                return JacobianOP.apply(torch.cat([x, x], dim=1), x)
        concatmodel = ConcatModule().to(device=self.default_device, dtype=self.default_dtype)
        x0 = torch.randn(1, 5, device=self.default_device, dtype=self.default_dtype)
        BoundedModule(concatmodel, x0)
        print('Concat JacobianOP test passed.')


def test_jacobian_placeholder_preserves_output_dtype():
    value = torch.tensor([[1.0]], dtype=torch.float64)
    variable = torch.tensor([[0.5]], dtype=torch.float64)

    placeholder = JacobianOP.apply(value, variable)

    assert placeholder.dtype == value.dtype


def test_float64_jacobian_bounds_preserve_dtype():
    class SquareJacobian(nn.Module):
        def forward(self, x):
            return JacobianOP.apply(x * x, x)

    center = torch.tensor([[0.5]], dtype=torch.float64)
    bounded_model = BoundedModule(SquareJacobian(), center)
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(norm=float('inf'), eps=0.1),
    )

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='CROWN')

    assert lower.dtype == torch.float64
    assert upper.dtype == torch.float64
    assert torch.allclose(lower, torch.tensor([[[0.8]]], dtype=torch.float64))
    assert torch.allclose(upper, torch.tensor([[[1.2]]], dtype=torch.float64))


if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestJacobian(generate=False)
    testcase.setUp()
    testcase.test()

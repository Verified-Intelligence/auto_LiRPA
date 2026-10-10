"""Test a model with an nn.Identity layer only"""
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE 

class TestIdentity(TestCase):
    def __init__(self, methodName='runTest', device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName, device=device, dtype=dtype)

    def test(self):
        model = nn.Sequential(nn.Identity())
        x = torch.randn(2, 10, device=self.default_device,
                        dtype=self.default_dtype)
        y = model(x)
        eps = 0.1
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        x = BoundedTensor(x, ptb)
        model = BoundedModule(model, x, device=self.default_device)
        y_l, y_u = model.compute_bounds()
        self.assertEqual(torch.Tensor(x), y)
        self.assertEqual(y_l, x - eps)
        self.assertEqual(y_u, x + eps)


def test_float64_default_specification_matches_output_dtype():
    center = torch.tensor([[0.3, 0.4]], dtype=torch.float64)
    bounded_model = BoundedModule(nn.Identity(), center)
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(norm=float('inf'), eps=0.1),
    )

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='CROWN')

    assert lower.dtype == torch.float64
    assert upper.dtype == torch.float64
    assert torch.allclose(lower, center - 0.1)
    assert torch.allclose(upper, center + 0.1)


if __name__ == '__main__':
    testcase = TestIdentity()
    testcase.test()

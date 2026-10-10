"""Test bounds for gather/index_select operators."""

import torch
from torch import nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class IndexSelectModel(nn.Module):
    def __init__(self, axis, indices):
        super().__init__()
        self.axis = axis
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))

    def forward(self, x):
        return torch.index_select(x, self.axis, self.indices)


class TestGather(TestCase):
    def __init__(self, methodName='runTest', generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName, seed=1,
            generate=generate, device=device, dtype=dtype)

    def _run_index_select_case(
            self, axis, indices, x_l, x_u, expected_lb, expected_ub):
        x_l = x_l.to(device=self.default_device, dtype=self.default_dtype)
        x_u = x_u.to(device=self.default_device, dtype=self.default_dtype)
        expected_lb = expected_lb.to(
            device=self.default_device, dtype=self.default_dtype)
        expected_ub = expected_ub.to(
            device=self.default_device, dtype=self.default_dtype)
        x = (x_l + x_u) / 2

        for method in ("IBP", "forward", "backward", "CROWN"):
            model = IndexSelectModel(axis, indices).to(
                device=self.default_device)
            lirpa_model = BoundedModule(model, x, device=self.default_device)
            bounded_x = BoundedTensor(
                x, PerturbationLpNorm(norm=float("inf"), x_L=x_l, x_U=x_u))

            lb, ub = lirpa_model.compute_bounds(
                x=(bounded_x,), method=method)

            self.assertEqual(lb, expected_lb)
            self.assertEqual(ub, expected_ub)

    def test(self):
        # The gather bound should follow the selected input intervals, including
        # repeated indices.
        self._run_index_select_case(
            axis=1,
            indices=[2, 0, 2, 1],
            x_l=torch.tensor([[0.8, -2.3, 2.5, 0.2],
                              [-1.0, 0.1, 1.5, 2.0]]),
            x_u=torch.tensor([[1.2, -1.7, 3.5, 0.8],
                              [-0.5, 0.4, 2.5, 3.0]]),
            expected_lb=torch.tensor([[2.5, 0.8, 2.5, -2.3],
                                      [1.5, -1.0, 1.5, 0.1]]),
            expected_ub=torch.tensor([[3.5, 1.2, 3.5, -1.7],
                                      [2.5, -0.5, 2.5, 0.4]]))


if __name__ == '__main__':
    testcase = TestGather(generate=False)
    testcase.setUp()
    testcase.test()

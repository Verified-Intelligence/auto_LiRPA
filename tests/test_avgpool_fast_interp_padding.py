import numpy as np
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE

class TestAvgPoolPadding(TestCase):
    """
    Regressions for AvgPool2d with padding in patches mode.

    - test_avg_equals_conv_quarter_with_padding:
        AvgPool2d(k=2,s=2,p=1) ⇔ Conv2d(k=2,s=2,p=1, weight=¼) under backward bounds,
        checked in both 'matrix' and 'patches' conv_mode.

    - test_matrix_equals_patches_for_avgpool_pad_relu:
        Reproduces the original buggy case:
            AvgPool2d(k=2,s=2,p=1) -> Conv2d(3x3,p=1) -> ReLU
        and checks that bounds from 'matrix' and 'patches' modes match.

      NOTE: This test fails on the buggy implementation and passes with the fix.
    """

    def __init__(self, methodName="runTest", generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName,
            ref_name=None,
            generate=generate,
            device=device,
            dtype=dtype,
        )

    # helpers
    def _compute_bounds(self, model: nn.Module, x: torch.Tensor,
                        eps: float, mode: str):
        """Utility to get (lb, ub) bounds for one model/input in a given conv_mode."""

        model = model.to(self.default_device, self.default_dtype)
        x = x.to(self.default_device, self.default_dtype)

        bm = BoundedModule(
            model, x,
            device=self.default_device,
            bound_opts={"conv_mode": mode, "bound_every_node": True}
        )
        xb = BoundedTensor(x, PerturbationLpNorm(norm=np.inf, eps=eps))
        lb, ub = bm.compute_bounds(x=(xb,), method="backward")
        return lb, ub

    # tests
    def test_avg_equals_conv_quarter_with_padding(self):
        """
        Check AvgPool(k=2,s=2,p=1) ≡ Conv(¼, k=2,s=2,p=1) w.r.t. bounds, in both modes.
        """

        # Small 3x3 input.
        x = torch.tensor([[[
            [-0.1526, -0.0750, -0.0654],
            [-0.0100, -0.0609, -0.0980],
            [-0.0712,  0.0304, -0.0777],
        ]]]).to(self.default_dtype)

        # Second conv after pool (2x2, stride 1).
        conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
        with torch.no_grad():
            conv2.weight[:] = torch.tensor([[[
                [ 0.0483, -0.0013],
                [ 0.1037, -0.1241],
            ]]], dtype=self.default_dtype)
            conv2.bias[:] = torch.tensor([0.0160], dtype=self.default_dtype)

        # A) AvgPool -> Conv2
        avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        model_avg = nn.Sequential(avg, conv2)

        # B) Conv(1/4) -> Conv2
        conv1 = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=1, bias=False)
        with torch.no_grad():
            conv1.weight.fill_(0.25)
        model_conv = nn.Sequential(conv1, conv2)

        # Sanity: nominal forward should match.
        with torch.no_grad():
            ya = model_avg(x)
            yb = model_conv(x)
        self.assertEqual(ya, yb)

        # Bounds: both modes should agree (conv vs avg equivalence).
        for mode in ("matrix", "patches"):
            lbA, ubA = self._compute_bounds(model_avg, x, eps=1.0, mode=mode)
            lbB, ubB = self._compute_bounds(model_conv, x, eps=1.0, mode=mode)
            self.assertEqual(lbA, lbB)
            self.assertEqual(ubA, ubB)

    def test_matrix_equals_patches_for_avgpool_pad_relu(self):
        """
        Reproduce the original buggy case:
            AvgPool2d(k=2,s=2,p=1) -> Conv2d(3x3,p=1) -> ReLU
        Compare bounds from 'matrix' vs 'patches' modes under eps=1.0.

        This failed before the fix (patches ≠ matrix) and should pass after.
        """

        # 4x4 input from the original reproducer.
        x4 = torch.tensor([[[
           [-0.1526, -0.0750, -0.0654, -0.1609],
           [-0.0100, -0.0609, -0.0980, -0.1609],
           [-0.0712,  0.0304, -0.0777, -0.0251],
           [-0.0222,  0.1687,  0.0228,  0.0468],
        ]]], dtype=self.default_dtype)

        # Conv 3x3 (padding=1) + ReLU.
        conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        with torch.no_grad():
            conv3.weight[:] = torch.tensor([[[
                [ 0.0483, -0.0013,  0.2914],
                [ 0.1037, -0.1241, -0.2013],
                [-0.0559, -0.1438, -0.1068],
            ]]],dtype=self.default_dtype)
            conv3.bias[:] = torch.tensor([0.0160], dtype=self.default_dtype)

        model = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
            conv3,
            nn.ReLU(),
        )

        # Compare matrix vs patches modes (eps=1).
        lbM, ubM = self._compute_bounds(model, x4, eps=1.0, mode="matrix")
        lbP, ubP = self._compute_bounds(model, x4, eps=1.0, mode="patches")

        self.assertEqual(lbM, lbP)
        self.assertEqual(ubM, ubP)


if __name__ == "__main__":
    t = TestAvgPoolPadding()
    t.test_avg_equals_conv_quarter_with_padding()
    t.test_matrix_equals_patches_for_avgpool_pad_relu()

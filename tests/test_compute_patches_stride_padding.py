"""Focused regression test for non-square patch stride propagation."""

import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class TestComputePatchesStridePadding(TestCase):
    def __init__(self, methodName="runTest", generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName,
            ref_name=None,
            generate=generate,
            device=device,
            dtype=dtype,
        )

    def test_non_square_stride_helper_regression(self):
        # This witness is intentionally small and add-free:
        #
        #   Conv2d(1x1) -> ReLU -> Conv2d(3x2, stride=(2,1), padding=(1,0))
        #   -> ReLU -> Flatten -> Linear(1)
        #
        # The second convolution asks `compute_patches_stride_padding()` to
        # combine its non-square stride with the incoming patch metadata.
        class TinyStrideCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(
                        1, 1,
                        kernel_size=(3, 2),
                        stride=(2, 1),
                        padding=(1, 0),
                        bias=False,
                    ),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(18, 1, bias=False),
                )

            def forward(self, x):
                return self.net(x)

        torch.manual_seed(0)
        model = TinyStrideCNN().to(self.default_device, self.default_dtype)
        x0 = torch.rand(
            1, 1, 5, 7,
            device=self.default_device,
            dtype=self.default_dtype,
        )
        x = BoundedTensor(
            x0,
            PerturbationLpNorm(norm=torch.inf, eps=0.1),
        )

        def compute_bounds(conv_mode):
            bounded_model = BoundedModule(
                model,
                (x0,),
                device=self.default_device,
                bound_opts={"conv_mode": conv_mode},
            )
            return bounded_model.compute_bounds(x=(x,), method="backward")

        # The complete patches pipeline agrees with matrix mode on this
        # non-square stride witness.
        lb_matrix, ub_matrix = compute_bounds("matrix")
        lb_patches, ub_patches = compute_bounds("patches")
        self.assertTrue(torch.allclose(lb_patches, lb_matrix))
        self.assertTrue(torch.allclose(ub_patches, ub_matrix))


if __name__ == "__main__":
    t = TestComputePatchesStridePadding()
    t.test_non_square_stride_helper_regression()

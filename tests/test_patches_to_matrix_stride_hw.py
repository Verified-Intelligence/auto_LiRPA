"""End-to-end regression test for 2D stride in `patches_to_matrix()`."""

import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class _IdentityReshape(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])


class TestPatchesToMatrixStrideHW(TestCase):
    def __init__(self, methodName="runTest", generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName,
            ref_name=None,
            generate=generate,
            device=device,
            dtype=dtype,
        )

    def test_end_to_end_non_square_stride_reshape_regression(self):
        # This witness is intentionally small:
        #
        #   Conv2d(1x1) -> identity Reshape -> Conv2d(3x2, stride=(2,1))
        #   -> ReLU
        #
        # BoundReshape converts the incoming patch operator to a dense matrix
        # by calling `patches_to_matrix()`, which must preserve separate height
        # and width strides.
        class TinyReshapeStrideCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, kernel_size=1, bias=False)
                self.reshape = _IdentityReshape()
                self.conv2 = nn.Conv2d(
                    1, 1,
                    kernel_size=(3, 2),
                    stride=(2, 1),
                    padding=(1, 0),
                    bias=False,
                )
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.reshape(x)
                x = self.conv2(x)
                x = self.relu(x)
                return x

        torch.manual_seed(0)
        model = TinyReshapeStrideCNN().to(self.default_device, self.default_dtype)
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
    t = TestPatchesToMatrixStrideHW()
    t.test_end_to_end_non_square_stride_reshape_regression()

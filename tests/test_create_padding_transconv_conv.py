import numpy as np
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.patches import Patches
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class _DisablePaddedAttr:
    """Simulate missing boundary padding masking by returning raw patches."""

    def __enter__(self):
        self._orig = Patches.create_padding

        def hacked(self, output_shape):
            return self.patches

        Patches.create_padding = hacked
        return self

    def __exit__(self, exc_type, exc, tb):
        Patches.create_padding = self._orig
        return False


class TestCreatePaddingTransConvConv(TestCase):
    """
    Regression for transposed-convolution chains in patches mode.

    Two consecutive ConvTranspose2d layers both alter valid spatial support.
    When the backward merge uses the unmasked patches from the first transposed
    conv, coefficients from padded positions can leak and contaminate valid
    bounds in the second transposed conv.

    This is a mirror/dual of the normal-conv bug: if patch-padding masking is
    skipped, patch-mode bounds deviate from matrix-mode bounds.
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

    def _bounds(self, model, x0, x, conv_mode):
        model = model.to(self.default_device, self.default_dtype)
        x0 = x0.to(self.default_device, self.default_dtype)
        x = x.to(self.default_device, self.default_dtype)
        bm = BoundedModule(
            model,
            x0,
            bound_opts={
                "conv_mode": conv_mode,
                "verbosity": 0,
                "sparse_intermediate_bounds_with_ibp": False,
                "sparse_intermediate_bound_ibp": False,
            },
        )
        lb, ub = bm.compute_bounds(
            x=(x,),
            method="backward",
        )
        return lb.detach(), ub.detach()

    def test_create_padding_transconv_transconv(self):
        torch.manual_seed(42)

        eps = 0.0
        x0 = torch.randn(1, 1, 8, 8)
        x = BoundedTensor(
            x0,
            PerturbationLpNorm(norm=np.inf, eps=eps),
        )

        model = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=0, bias=True),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=2, bias=True),
            nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=True),
            nn.ReLU()
        )

        with torch.no_grad():
            for m in model:
                if isinstance(m, nn.ConvTranspose2d):
                    m.weight.copy_(torch.randn_like(m.weight))
                    m.bias.copy_(torch.randn_like(m.bias))

        lb_m, ub_m = self._bounds(model, x0, x, "matrix")
        lb_p, ub_p = self._bounds(model, x0, x, "patches")
        with _DisablePaddedAttr():
            lb_bug, ub_bug = self._bounds(model, x0, x, "patches")

        diff = max(
            (lb_m - lb_p).abs().max().item(),
            (ub_m - ub_p).abs().max().item(),
        )
        self.assertLess(diff, 1e-6)
        diff_bug = max(
            (lb_m - lb_bug).abs().max().item(),
            (ub_m - ub_bug).abs().max().item(),
        )
        self.assertGreater(diff_bug, 1e-4)


if __name__ == "__main__":
    t = TestCreatePaddingTransConvConv()
    t.test_create_padding_transconv_transconv()

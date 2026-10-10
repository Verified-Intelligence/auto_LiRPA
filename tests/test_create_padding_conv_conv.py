import torch
import torch.nn as nn
import numpy as np

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.patches import Patches
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class _DisablePaddedAttr:
    """Simulate missing padding_pos_masked attribute by returning raw patches tensors."""

    def __enter__(self):
        self._orig = Patches.create_padding

        def hacked(self, output_shape):
            return self.patches

        Patches.create_padding = hacked
        return self

    def __exit__(self, exc_type, exc, tb):
        Patches.create_padding = self._orig
        return False


class TestCreatePaddingConvConv(TestCase):
    """
    Regression for conv-conv pattern in patches mode.

    The second conv merges patches from the first via a transposed-conv-like
    operation. If padding positions are not masked (create_padding), those
    padding coefficients leak into valid positions and bounds diverge.
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
        lb, ub = bm.compute_bounds(x=(x,), method="backward")
        return lb.detach(), ub.detach()

    def test_create_padding_conv_conv_monkeypatch(self):
        torch.manual_seed(0)

        eps = 0.0
        x0 = torch.randn(1, 1, 8, 8)
        x = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=eps))

        model = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
        )

        with torch.no_grad():
            for m in model:
                if isinstance(m, nn.Conv2d):
                    m.weight.copy_(torch.randn_like(m.weight))

        lb_m, ub_m = self._bounds(model, x0, x, "matrix")
        lb_p, ub_p = self._bounds(model, x0, x, "patches")

        with _DisablePaddedAttr():
            lb_bug, ub_bug = self._bounds(model, x0, x, "patches")

        diff_ok = max(
            (lb_m - lb_p).abs().max().item(),
            (ub_m - ub_p).abs().max().item(),
        )
        diff_bug = max(
            (lb_m - lb_bug).abs().max().item(),
            (ub_m - ub_bug).abs().max().item(),
        )

        self.assertLess(diff_ok, 1e-6)
        self.assertGreater(diff_bug, 1e-4)


if __name__ == "__main__":
    t = TestCreatePaddingConvConv()
    t.test_create_padding_conv_conv_monkeypatch()

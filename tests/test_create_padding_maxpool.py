import numpy as np
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import auto_LiRPA.patches as patches
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class _DisableUnfoldPadding:
    """Disable patch padding mask by forcing unfold to return all-ones."""

    def __enter__(self):
        self._orig = patches.inplace_unfold

        def hacked(*args, **kwargs):
            out = self._orig(*args, **kwargs)
            return torch.ones_like(out)

        patches.inplace_unfold = hacked
        return self

    def __exit__(self, exc_type, exc, tb):
        patches.inplace_unfold = self._orig
        return False


class TestCreatePaddingMaxPoolCnn(TestCase):
    """
    Regression for conv-maxpool pattern in patches mode.

    Key idea: maxpooling multiplies patches by an unfolded all-ones tensor,
    which implicitly applies create_padding (i.e., zeros padded positions).
    Therefore we can mark Patches.padding_pos_masked = True after maxpool.

    This test shows the converse: if we monkeypatch the unfolded 1-matrix to be
    all ones (including padding positions), then treating the result as boundary-masked
    is incorrect and the bounds diverge.
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

    def _bounds(self, model, x0, x):
        model = model.to(self.default_device, self.default_dtype)
        x0 = x0.to(self.default_device, self.default_dtype)
        x = x.to(self.default_device, self.default_dtype)
        bm = BoundedModule(
            model,
            x0,
            bound_opts={
                "conv_mode": "patches",
                "bound_every_node": True,
                "verbosity": 0,
                "sparse_intermediate_bounds_with_ibp": False,
                "sparse_intermediate_bound_ibp": False,
            },
        )
        lb, ub = bm.compute_bounds(x=(x,), method="backward")
        return lb.detach(), ub.detach()

    def test_create_padding_maxpool_cnn_monkeypatch(self):
        torch.manual_seed(0)

        x0 = torch.randn(1, 1, 6, 6)
        x = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=1.0))

        model = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )

        with torch.no_grad():
            for m in model:
                if isinstance(m, nn.Conv2d):
                    m.weight.copy_(torch.randn_like(m.weight))

        lb_p, ub_p = self._bounds(model, x0, x)

        with _DisableUnfoldPadding():
            lb_bug, ub_bug = self._bounds(model, x0, x)

        diff_bug = max(
            (lb_p - lb_bug).abs().max().item(),
            (ub_p - ub_bug).abs().max().item(),
        )
        self.assertGreater(diff_bug, 1e-4)


if __name__ == "__main__":
    t = TestCreatePaddingMaxPoolCnn()
    t.test_create_padding_maxpool_cnn_monkeypatch()

import math

import torch
import torch.nn as nn

import auto_LiRPA.operators.activation_base as activation_base
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.patches import Patches
from auto_LiRPA.perturbations import *
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE

class cnn_model(nn.Module):
    def __init__(self, layers, padding, stride, linear):
        super(cnn_model, self).__init__()
        self.module_list = []
        channel = 1
        length = 28
        for i in range(layers):
            self.module_list.append(nn.Conv2d(channel, 3, 4, stride = stride, padding = padding))
            channel = 3
            length = (length + 2 * padding - 4)//stride + 1
            assert length > 0
            self.module_list.append(nn.ReLU())
        self.module_list.append(nn.Flatten())
        if linear:
            self.module_list.append(nn.Linear(3 * length * length, 256))
            self.module_list.append(nn.Linear(256, 10))
        self.model = nn.Sequential(*self.module_list)

    def forward(self, x):
        x = self.model(x)
        return x


class ConstantRootConvModel(nn.Module):
    """A convolutional model with an input normalization constant root."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            'input_mean',
            torch.tensor(
                [[[[0.1, -0.2, 0.3, -0.4]]]],
                dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE))
        self.conv1 = nn.Conv2d(
            1, 4, (1, 4), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
        self.conv2 = nn.Conv2d(
            4, 1, 1, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)

    def forward(self, x):
        x = x - self.input_mean
        x = torch.relu(self.conv1(x))
        return self.conv2(x).flatten(1)


class _ConvSigmoidReLU(nn.Module):
    """The downstream ReLU sparsifies specs propagated through Sigmoid."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 8 * 8, 3)

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.fc(x.flatten(1))


class TestConv(TestCase):
    def __init__(self, methodName='runTest', generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName,
            seed=1, ref_name=None,
            generate=generate,
            device=device, dtype=dtype)

    def test(self):
        models = [1, 2, 3]
        paddings = [1, 2]
        strides = [1, 3]

        N = 2
        n_classes = 10
        image = torch.randn(N, 1, 28, 28, dtype=self.default_dtype, device=self.default_device)
        image = image / 255.0

        for layer_num in models:
            for padding in paddings:
                for stride in strides:
                    for linear in [True, False]:
                        model_ori = cnn_model(layer_num, padding, stride, linear)
                        print('Model:', model_ori)
                        model_ori = model_ori.to(
                            device=self.default_device, dtype=self.default_dtype)

                        model = BoundedModule(model_ori, image, device=self.default_device, bound_opts={"conv_mode": "patches"})
                        eps = 0.3
                        ptb = PerturbationLpNorm(x_L=image-eps, x_U=image+eps)
                        image = BoundedTensor(image, ptb)
                        pred = model(image)
                        lb, ub = model.compute_bounds()

                        model = BoundedModule(model_ori, image, device=self.default_device, bound_opts={"conv_mode": "matrix"})
                        pred = model(image)
                        lb_ref, ub_ref = model.compute_bounds()

                        if linear:
                            assert lb.shape == ub.shape == torch.Size((N, n_classes))
                        self.assertEqual(lb, lb_ref)
                        self.assertEqual(ub, ub_ref)

                        if not linear and layer_num == 1:
                            pred = model(image)
                            lb_forward, ub_forward = model.compute_bounds(method='forward')
                            self.assertEqual(lb, lb_forward)
                            self.assertEqual(ub, ub_forward)
                            pred = model(image)
                            lb_forward, ub_forward = model.compute_bounds(method='dynamic-forward+backward')
                            self.assertEqual(lb, lb_forward)
                            self.assertEqual(ub, ub_forward)


def test_patches_constant_root_matches_matrix_mode():
    """CROWN handles a broadcast constant root before a patches-mode conv."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = ConstantRootConvModel()
    x_L = torch.full(
        (2, 1, 1, 4), -5.0, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    x_U = torch.full(
        (2, 1, 1, 4), 5.0, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    x = BoundedTensor(
        torch.zeros_like(x_L), PerturbationLpNorm(x_L=x_L, x_U=x_U))

    bounds = {}
    for conv_mode in ('patches', 'matrix'):
        bounded_model = BoundedModule(
            model, torch.zeros_like(x_L),
            bound_opts={
                'conv_mode': conv_mode,
                'sparse_intermediate_bounds': False,
            },
            device=DEFAULT_DEVICE)
        bounds[conv_mode] = bounded_model.compute_bounds(
            x=(x,), method='CROWN')

    patches_lb, patches_ub = bounds['patches']
    matrix_lb, matrix_ub = bounds['matrix']
    assert patches_lb.shape == patches_ub.shape == (2, 1)
    torch.testing.assert_close(patches_lb, matrix_lb, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(patches_ub, matrix_ub, rtol=1e-5, atol=1e-6)


def test_sparse_patches_activation_matches_dense_spec(monkeypatch):
    """Sparse Patches select the same activation coefficients as dense specs."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = _ConvSigmoidReLU().to(torch.float64)
        x = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    bounded_input = BoundedTensor(x, PerturbationLpNorm(eps=0.05))

    sparse_unfolds = []
    original_unfold = activation_base.maybe_unfold_patches

    # Record the actual sparse-spec selection through the optimized activation.
    # A tuple unstable_idx identifies Patches specs; output_shape gives their
    # full count before selection.
    def record_sparse_unfold(d_tensor, last_A, alpha_lookup_idx=None):
        if (d_tensor is not None and isinstance(last_A, Patches)
                and isinstance(last_A.unstable_idx, tuple)):
            sparse_unfolds.append((
                last_A.unstable_idx[0].numel(),
                math.prod(last_A.output_shape[1:4])))
        return original_unfold(d_tensor, last_A, alpha_lookup_idx)

    monkeypatch.setattr(
        activation_base, 'maybe_unfold_patches', record_sparse_unfold)

    bounds = {}
    for sparse in (True, False):
        bounded_model = BoundedModule(
            model, x, device='cpu',
            bound_opts={
                'conv_mode': 'patches',
                'sparse_intermediate_bounds': sparse,
            })
        bounds[sparse] = bounded_model.compute_bounds(
            x=(bounded_input,), method='CROWN-Optimized')
        if sparse:
            # Require a proper subset so the dense comparison exercises the
            # sparse unfolding path rather than two equivalent dense runs.
            assert any(0 < selected < full for selected, full in sparse_unfolds)

    for sparse_bound, dense_bound in zip(bounds[True], bounds[False]):
        torch.testing.assert_close(
            sparse_bound, dense_bound, rtol=0, atol=1e-9)


if __name__ == '__main__':
    testcase = TestConv()
    testcase.test()

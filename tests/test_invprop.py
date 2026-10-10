"""Test INVPROP."""
import copy
import sys
import pytest
sys.path.append('../complete_verifier')
from complete_verifier.load_model import unzip_and_optimize_onnx
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.utils import BatchedCrownC
from auto_LiRPA.perturbations import *
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


@pytest.mark.parametrize('best_of_oc_and_no_oc', [False, True])
@pytest.mark.parametrize('indices', [None, [2, 0]])
def test_batched_intermediate_bounds_preserve_output_constraints(best_of_oc_and_no_oc, indices):
    """Neuron batching retains INVPROP tightening and the unbatched bounds."""
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1., 0.], [2., 0.], [3., 0.]]))
        model[0].bias.zero_()
        model[1].weight.fill_(1.)
        model[1].bias.zero_()
    center = torch.zeros(1, 2, dtype=torch.float64)
    bounds = []
    for chunk_size in [100, 1]:
        net = BoundedModule(copy.deepcopy(model), center, device='cpu', bound_opts={
            'crown_batch_size': chunk_size,
            'batched_crown_max_vram_ratio': 0.9,
            'optimize_bound_args': {
                'apply_output_constraints_to': ['BoundLinear'],
                'best_of_oc_and_no_oc': best_of_oc_and_no_oc,
            },
        })
        # Hidden neurons equal x_0, 2*x_0, and 3*x_0. Conditioning on
        # 6*x_0 <= 0 makes their upper bounds zero. Distinct lower bounds
        # also check that sparse, reordered chunks retain neuron ordering.
        net.constraints = torch.ones(1, 1, 1, dtype=torch.float64)
        net.thresholds = torch.zeros(1, dtype=torch.float64)
        bounded_x = BoundedTensor(center, PerturbationLpNorm(
            x_L=-torch.ones_like(center), x_U=torch.ones_like(center)))
        net.compute_bounds(x=(bounded_x,), method='CROWN')
        hidden = next(node for node in net.nodes()
                      if node.__class__.__name__ == 'BoundLinear' and not node.is_final_node)
        hidden.gammas = torch.tensor([[[0., 0., 0.]], [[1/6, 2/6, 3/6]]],
                                     dtype=torch.float64)
        selected = torch.arange(3) if indices is None else torch.tensor(indices)
        specification = (torch.eye(3, dtype=torch.float64)[selected].unsqueeze(0)
                         if chunk_size == 100 else BatchedCrownC('eye'))
        lower, upper = net.backward_general(
            hidden, specification, unstable_idx=None if indices is None else selected,
            apply_output_constraints_to=['BoundLinear'])
        bounds.append((lower.detach().clone(), upper.detach().clone()))
    for batched, unbatched in zip(bounds[1], bounds[0]):
        torch.testing.assert_close(batched, unbatched)
    # Comparing two unconstrained runs would miss dropped INVPROP arguments.
    expected_lower = -torch.tensor([[1., 2., 3.]], dtype=torch.float64)[:, selected]
    torch.testing.assert_close(bounds[1][0], expected_lower)
    torch.testing.assert_close(bounds[1][1], torch.zeros(1, len(selected), dtype=torch.float64))


class SimpleExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Weights of linear layers.
        self.w1 = torch.tensor([[1., -1.], [2., -1.]])
        self.w2 = torch.tensor([[1., -1.]])

    def forward(self, x):
        # Linear layer.
        z1 = x.matmul(self.w1.t())
        # Relu layer.
        hz1 = torch.nn.functional.relu(z1)
        # Linear layer.
        z2 = hz1.matmul(self.w2.t())
        return z2

class TestInvpropSimpleExample(TestCase):
    def __init__(self, methodName='runTest', generate=False, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName,
            seed=1, ref_name=None,
                         generate=generate, device=device, dtype=dtype)

    def test(self):
        np.random.seed(123)

        model_ori = SimpleExampleModel().to(
            device=self.default_device, dtype=self.default_dtype)

        apply_output_constraints_to = ['BoundMatMul', 'BoundInput']
        x = torch.tensor([[1., 1.]], device=self.default_device,
                         dtype=self.default_dtype)
        model = BoundedModule(model_ori, torch.empty_like(x), bound_opts={
            'optimize_bound_args': {
                'apply_output_constraints_to': apply_output_constraints_to,
                'tighten_input_bounds': True,
                'best_of_oc_and_no_oc': False,
                'directly_optimize': [],
                'oc_lr': 0.1,
                'share_gammas': False,
                'iteration': 1000,
            }
        },
            device=self.default_device
        )
        model.constraints = torch.ones(
            1, 1, 1, device=self.default_device, dtype=self.default_dtype)
        model.thresholds = torch.tensor(
            [-1.], device=self.default_device, dtype=self.default_dtype)

        norm = float("inf")
        lower = torch.tensor(
            [[-1., -2.]], device=self.default_device, dtype=self.default_dtype)
        upper = torch.tensor(
            [[2., 1.]], device=self.default_device, dtype=self.default_dtype)
        ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
        bounded_x = BoundedTensor(x, ptb)

        lb, ub = model.compute_bounds(x=(bounded_x,), method='alpha-CROWN')

        if '/0' in model._modules:
            tightened_ptb = model['/0'].perturbation
        else:
            tightened_ptb = model['/x'].perturbation

        if self.default_dtype == torch.float64:
            data_path = 'data_64/'
        else:
            data_path = 'data/'

        if self.generate:
            torch.save({
                'lb': lb,
                'ub': ub,
                'x_L': tightened_ptb.x_L,
                'x_U': tightened_ptb.x_U
            }, data_path + 'invprop/simple_reference')
        else:
            data = torch.load(data_path + 'invprop/simple_reference')
            lb_ref = data['lb']
            ub_ref = data['ub']
            x_L_ref = data['x_L']
            x_U_ref = data['x_U']

            assert torch.allclose(lb, lb_ref, 1e-4)
            assert torch.allclose(ub, ub_ref, 1e-4)
            assert torch.allclose(tightened_ptb.x_L, x_L_ref, 1e-4)
            assert torch.allclose(tightened_ptb.x_U, x_U_ref, 1e-4)

class TestInvpropOODExample(TestCase):
    # Based on https://github.com/kothasuhas/verify-input/tree/main/examples/ood
    def __init__(self, methodName='runTest', generate=False, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName, seed=1, ref_name=None,
                         generate=generate, device=device, dtype=dtype)

    def test(self):
        np.random.seed(123)

        import onnx2pytorch

        model_ori = onnx2pytorch.ConvertModel(unzip_and_optimize_onnx('data/invprop/ood.onnx')).eval()
        model_ori = model_ori.to(
            device=self.default_device, dtype=self.default_dtype)

        x = torch.tensor([[1., 1.]], device=self.default_device,
                         dtype=self.default_dtype)
        model = BoundedModule(model_ori, torch.empty_like(x), bound_opts={
            'optimize_bound_args': {
                'apply_output_constraints_to': ['BoundInput', "/input", "/input-3", "/21"],
                'tighten_input_bounds': True,
                'best_of_oc_and_no_oc': True,
                'directly_optimize': ['/input'],
                'oc_lr': 0.01,
                'iteration': 1000,
                'share_gammas': False,
                'lr_decay': 0.99,
                'early_stop_patience': 1000,
                'init_alpha': False,
                'lr_alpha': 0.4,
                'start_save_best': -1,
            }
        },
            device=self.default_device
        )
        model.constraints = torch.tensor(
            [[[-1., 0., 1.]], [[0., -1., 1.]]], device=self.default_device, dtype=self.default_dtype)
        model.thresholds = torch.tensor(
            [0., 0.], device=self.default_device, dtype=self.default_dtype)

        norm = float("inf")
        lower = torch.tensor(
            [[-2., -2.], [-2., -2.]], device=self.default_device, dtype=self.default_dtype)
        upper = torch.tensor(
            [[0., 0.], [0., 0.]], device=self.default_device, dtype=self.default_dtype)
        ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
        x_expand = BoundedTensor(torch.tensor(
            [[-1., -1.], [-1., -1.]], device=self.default_device, dtype=self.default_dtype), ptb)
        c = torch.tensor([[[-1.,  0.,  1.]], [[0., -1.,  1.]]],
                         device=self.default_device, dtype=self.default_dtype)

        # Init manually, to set bound_upper=False
        model.init_alpha(
                (x_expand,), share_alphas=False, c=c, bound_upper=False)

        model.compute_bounds(x=(x_expand,), C=c, method='CROWN-Optimized')

        if self.default_dtype == torch.float64:
            data_path = 'data_64/'
        else:
            data_path = 'data/'

        if self.generate:
            torch.save({
                'lower': model['/input'].lower,
                'upper': model['/input'].upper,
            }, data_path + 'invprop/ood_reference')
        else:
            data = torch.load(data_path + 'invprop/ood_reference')
            lower_ref = data['lower']
            upper_ref = data['upper']

            lower_diff = model['/input'].lower[0] - lower_ref[0]
            assert torch.allclose(model['/input'].lower[0], lower_ref[0], atol=1e-3), (lower_diff, lower_diff.abs().max())
            assert torch.all(torch.isposinf(lower_ref[1]))
            assert torch.all(torch.isposinf(model['/input'].lower[1]))
            upper_diff = model['/input'].upper[0] - upper_ref[0]
            assert torch.allclose(model['/input'].upper[0], upper_ref[0], atol=1e-3), (upper_diff, upper_diff.abs().max())
            assert torch.all(torch.isneginf(upper_ref[1]))
            assert torch.all(torch.isneginf(model['/input'].upper[1]))


if __name__ == '__main__':
    testcase = TestInvpropSimpleExample(generate=False)
    testcase.test()
    testcase = TestInvpropOODExample(generate=False)
    testcase.test()

"""Test INVPROP."""

from complete_verifier.load_model import unzip_and_optimize_onnx
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase


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
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName,
            seed=1, ref_path=None,
            generate=generate)

    def test(self):
        np.random.seed(123)

        model_ori = SimpleExampleModel()

        apply_output_constraints_to = ['BoundMatMul', 'BoundInput']
        x = torch.tensor([[1., 1.]])
        model = BoundedModule(model_ori, torch.empty_like(x), device="cpu", bound_opts={
            'optimize_bound_args': {
                'apply_output_constraints_to': apply_output_constraints_to,
                'tighten_input_bounds': True,
                'best_of_oc_and_no_oc': False,
                'directly_optimize': [],
                'oc_lr': 0.1,
                'share_gammas': False,
                'iteration': 1000,
            }
        })
        model.constraints = torch.ones(1,1,1)
        model.thresholds = torch.tensor([-1.])

        norm = float("inf")
        lower = torch.tensor([[-1., -2.]])
        upper = torch.tensor([[2., 1.]])
        ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
        bounded_x = BoundedTensor(x, ptb)

        lb, ub = model.compute_bounds(x=(bounded_x,), method='alpha-CROWN')
        tightened_ptb = model['/0'].perturbation

        if self.generate:
            torch.save({
                'lb': lb,
                'ub': ub,
                'x_L': tightened_ptb.x_L,
                'x_U': tightened_ptb.x_U
            }, 'data/invprop/simple_reference')
        else:
            data = torch.load('data/invprop/simple_reference')
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
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName,
            seed=1, ref_path=None,
            generate=generate)

    def test(self):
        np.random.seed(123)

        import onnx2pytorch
        import onnx
        model_ori = onnx2pytorch.ConvertModel(unzip_and_optimize_onnx('data/invprop/ood.onnx')).eval()

        x = torch.tensor([[-1., -1.]])
        model = BoundedModule(model_ori, torch.empty_like(x), bound_opts={
            'optimize_bound_args': {
                'apply_output_constraints_to': ['BoundInput', "/input", "/input.3", "/21"],
                'tighten_input_bounds': True,
                'best_of_oc_and_no_oc': True,
                'directly_optimize': ['/21'],
                'oc_lr': 0.01,
                'iteration': 1000,
                'share_gammas': False,
                'lr_decay': 0.99,
                'early_stop_patience': 1000,
                'init_alpha': False,
                'lr_alpha': 0.4,
                'start_save_best': -1,
            }
        })
        model.constraints = torch.tensor([[[-1., 0., 1.]], [[0., -1., 1.]]])
        model.thresholds = torch.tensor([0., 0.])

        norm = float("inf")
        lower = torch.tensor([[-2., -2.], [-2., -2.]])
        upper = torch.tensor([[0., 0.], [0., 0.]])
        ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
        x_expand = BoundedTensor(torch.tensor([[-1., -1.], [-1., -1.]]), ptb)

        c = torch.tensor([[[-1.,  0.,  1.]], [[ 0., -1.,  1.]]])

        # Init manually, to set bound_upper=False
        model.init_alpha(
                (x_expand,), share_alphas=False, c=c, bound_upper=False)

        model.compute_bounds(x=(x_expand,), C=c, method='CROWN-Optimized')

        if self.generate:
            torch.save({
                'lower': model['/21'].lower,
                'upper': model['/21'].upper,
            }, 'data/invprop/ood_reference')
        else:
            data = torch.load('data/invprop/ood_reference')
            lower_ref = data['lower']
            upper_ref = data['upper']

            lower_diff = model['/21'].lower[0] - lower_ref[0]
            assert torch.allclose(model['/21'].lower[0], lower_ref[0], atol=1e-3), (lower_diff, lower_diff.abs().max())
            assert torch.all(torch.isposinf(lower_ref[1]))
            assert torch.all(torch.isposinf(model['/21'].lower[1]))
            upper_diff = model['/21'].upper[0] - upper_ref[0]
            assert torch.allclose(model['/21'].upper[0], upper_ref[0], atol=1e-3), (upper_diff, upper_diff.abs().max())
            assert torch.all(torch.isneginf(upper_ref[1]))
            assert torch.all(torch.isneginf(model['/21'].upper[1]))


if __name__ == '__main__':
    testcase = TestInvpropSimpleExample(generate=False)
    testcase.test()
    testcase = TestInvpropOODExample(generate=False)
    testcase.test()

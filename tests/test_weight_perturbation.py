import copy
import random
import argparse
import torch.nn.functional as F
import subprocess
import numpy as np
from testcase import TestCase
import sys
sys.path.append('../examples/vision')
import models
from auto_LiRPA import BoundedModule, BoundedParameter
from auto_LiRPA.perturbations import *


class TestWeightPerturbation(TestCase): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName, seed=1234, ref_path='data/weight_perturbation_test_data')
        self.result = {}

    def test_training(self):
        ret = subprocess.run(
            ['python', 'weight_perturbation_training.py', 
            '--device', 'cpu',
            '--scheduler_opts', 'start=1,length=100',
            '--num_epochs',  '1', 
            '--truncate_data', '5'],
            cwd='../examples/vision', capture_output=True)
        self.assertEqual(ret.returncode, 0, ret.stderr)
        res_test = ret.stdout.decode().split('\n')[-2].split(' ')
        assert abs(float(res_test[-3].split('=')[1]) - 2.246) < 0.01

    def verify_bounds(self, model, x, IBP, method, forward_ret, lb_name, ub_name):
        lb, ub = model(method_opt="compute_bounds", x=(x,), IBP=IBP, method=method)
        self.result[lb_name] = lb.detach().data.clone()
        self.result[ub_name] = ub.detach().data.clone()

        assert torch.allclose(self.reference[lb_name], self.result[lb_name], 1e-4, 1e-6)
        assert torch.allclose(self.reference[ub_name], self.result[ub_name], 1e-4, 1e-6)
        assert ((self.reference[lb_name] - self.result[lb_name]).pow(2).sum() < 1e-8)
        assert ((self.reference[ub_name] - self.result[ub_name]).pow(2).sum() < 1e-8)

        # test gradient backward propagation
        loss = (ub - lb).abs().sum()
        loss.backward()

        # gradient w.r.t input only
        grad = x.grad
        self.result[lb_name+'_grad'] = grad.detach().data.clone()
        assert torch.allclose(self.reference[lb_name+'_grad'], self.result[lb_name + '_grad'], 1e-4, 1e-6)
        assert ((self.reference[lb_name + '_grad'] - self.result[lb_name + '_grad']).pow(2).sum() < 1e-8)

    def test_perturbation(self):
        np.random.seed(123) # FIXME This seed is inconsistent with other seeds (1234)

        model_ori = models.Models['mlp_3layer_weight_perturb'](pert_weight=True, pert_bias=True).eval()
        self.result['model'] = model_ori.state_dict()
        self.result['data'] = torch.randn(8, 1, 28, 28)
        model_ori.load_state_dict(self.result['model'])
        state_dict = copy.deepcopy(model_ori.state_dict())  
        dummy_input = self.result['data'].requires_grad_()
        inputs = (dummy_input,)

        model = BoundedModule(model_ori, inputs)
        forward_ret = model(dummy_input)
        model_ori.eval()

        assert torch.isclose(model_ori(dummy_input), model_ori(dummy_input), 1e-8).all()

        def verify_model(pert_weight=True, pert_bias=True, norm=np.inf, lb_name='', ub_name=''):
            model_ori_ = models.Models['mlp_3layer_weight_perturb'](pert_weight=pert_weight, pert_bias=pert_bias, norm=norm).eval()
            model_ori_.load_state_dict(state_dict)
            model_ = BoundedModule(model_ori_, inputs)
            model_.ptb = model_ori.ptb

            self.verify_bounds(model_, dummy_input, IBP=True, method='backward', forward_ret=forward_ret,
                        lb_name=lb_name + '_CROWN-IBP', ub_name=ub_name + '_CROWN-IBP')  # CROWN-IBP
            self.verify_bounds(model_, dummy_input, IBP=False, method='backward', forward_ret=forward_ret,
                        lb_name=lb_name + '_CROWN', ub_name=ub_name + '_CROWN')  # CROWN

        # Linf
        verify_model(pert_weight=True, pert_bias=True, norm=np.inf, lb_name='l_inf_weights_bias_lb', ub_name='l_inf_weights_bias_ub')
        verify_model(pert_weight=True, pert_bias=False, norm=np.inf, lb_name='l_inf_weights_lb', ub_name='l_inf_weights_ub')
        verify_model(pert_weight=False, pert_bias=True, norm=np.inf, lb_name='l_inf_bias_lb', ub_name='l_inf_bias_ub')

        # L2
        verify_model(pert_weight=True, pert_bias=True, norm=2, lb_name='l_2_weights_bias_lb', ub_name='l_2_weights_bias_ub')
        verify_model(pert_weight=True, pert_bias=False, norm=2, lb_name='l_2_weights_lb', ub_name='l_2_weights_ub')
        verify_model(pert_weight=False, pert_bias=True, norm=2, lb_name='l_2_bias_lb', ub_name='l_2_bias_ub')

        if self.generate:
            self.save()

if __name__ == '__main__':
    testcase = TestWeightPerturbation()
    testcase.setUp()
    testcase.test_perturbation()
    testcase.test_training()

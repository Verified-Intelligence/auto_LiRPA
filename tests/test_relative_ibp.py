"""Test IBP with relative bounds"""
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_ops import *
from testcase import TestCase  
import sys
sys.path.append('../examples/vision')
from models import *

class TestRelativeIBP(TestCase): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName)

    def test(self):
        dummy_input = torch.randn(1, 3, 32, 32)

        model_ori = cnn_6layer(in_ch=3, in_dim=32)
        model = BoundedModule(model_ori, dummy_input, bound_opts={ 'ibp_relative': True })

        model_ori_ref = cnn_6layer(in_ch=3, in_dim=32)
        model_ori_ref.load_state_dict(model_ori.state_dict())
        model_ref = BoundedModule(model_ori_ref, dummy_input, bound_opts={ 'ibp_relative': False })

        eps = 1e-1
        data = torch.randn(8, 3, 32, 32)
        data_lb, data_ub = data - eps, data + eps
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=data_lb, x_U=data_ub, relative=True)
        x = (BoundedTensor(data, ptb),)

        fv = model(x)
        fv_ref = model_ref(x)
        lb, ub = model.compute_bounds(method='IBP')
        lb_ref, ub_ref = model_ref.compute_bounds(method='IBP')

        self.assertEqual(lb, lb_ref)
        self.assertEqual(ub, ub_ref)

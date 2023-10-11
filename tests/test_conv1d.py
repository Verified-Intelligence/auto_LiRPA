"""Test Conv1d."""

from collections import defaultdict
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import Flatten
from testcase import TestCase


class Model(nn.Module):
    def __init__(self, kernel_size=2, stride=1, padding=0, in_features=1,out_features=1):
        super(Model, self).__init__()        
        self.n_n_conv1d_1 = nn.Conv1d(**{'groups': 1, 'dilation': 1, 'out_channels': 1, 'padding': padding, 'kernel_size': kernel_size, 'stride': stride, 'in_channels': 1, 'bias': True})
        self.n_n_conv1d_2 = nn.Conv1d(**{'groups': 1, 'dilation': 1, 'out_channels': 1, 'padding': padding, 'kernel_size': kernel_size, 'stride': stride, 'in_channels': 1, 'bias': True})
        self.relu_2 = nn.ReLU()
        self.n_n_conv1d_3 = nn.Conv1d(**{'groups': 1, 'dilation': 1, 'out_channels': 1, 'padding': padding, 'kernel_size': kernel_size, 'stride': stride, 'in_channels': 1, 'bias': True})
        self.relu_3 = nn.ReLU()
        self.n_n_activation_Flatten = nn.Flatten(**{'start_dim': 1})
        L_in,dialation = in_features,1
        L_out_1 = math.floor((L_in+2*padding-dialation*(kernel_size-1)-1)/stride+1)
        L_out_2 = math.floor((L_out_1+2*padding-dialation*(kernel_size-1)-1)/stride+1)
        L_out_3 = math.floor((L_out_2+2*padding-dialation*(kernel_size-1)-1)/stride+1)
        self.n_n_linear = nn.Linear(**{'in_features':L_out_3, 'out_features':out_features,'bias':True})

    def forward(self, *inputs,debug=False):
        t_ImageInputLayer, = inputs
        t_conv1d_1 = self.n_n_conv1d_1(t_ImageInputLayer)
        if debug: print("t_ImageInputLayer",t_ImageInputLayer.shape)
        if debug: print("t_conv1d_1",t_conv1d_1.shape)
        t_conv1d_relu_1 = F.relu(t_conv1d_1)
        t_conv1d_2 = self.n_n_conv1d_2(t_conv1d_relu_1)
        if debug: print("t_conv1d_2",t_conv1d_2.shape)
        t_conv1d_relu_2 = F.relu(t_conv1d_2)
        t_conv1d_3 = self.n_n_conv1d_3(t_conv1d_relu_2)
        if debug: print("t_conv1d_3",t_conv1d_3.shape)
        t_conv1d_relu_3 = F.relu(t_conv1d_3)
        t_flatten = self.n_n_activation_Flatten(t_conv1d_relu_3)
        if debug: print("t_flatten",t_flatten.shape)
        t_linear = self.n_n_linear(t_flatten)        
        if debug: print("t_linear",t_linear.shape)
        return t_linear

class TestConv1D(TestCase):
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName,
            seed=1, ref_path=None,
            generate=generate)

    def test(self):
        np.random.seed(123)

        N = 3
        C = 1
        M = 173
        n_classes = 2
        for kernel_size in [3,4]:
            for padding in [0,1]:
                    for stride in [2,3]:
                        print(kernel_size, padding, stride)

                        model_ori = Model(kernel_size=kernel_size, padding=padding, stride=stride, in_features=M,out_features=n_classes)
                        if not self.generate:
                            data = torch.load('data/conv1d_test_data_{}-{}-{}'.format(kernel_size, padding, stride))
                            image = data['input']
                            model_ori(image)
                            model_ori.load_state_dict(data['model'])
                        else:
                            image = torch.rand([N, C, M])
                            model_ori(image)


                        conv_mode = "matrix"

                        model = BoundedModule(model_ori, image, device="cpu", bound_opts={"conv_mode": conv_mode})
                        eps = 0.3
                        norm = np.inf
                        ptb = PerturbationLpNorm(norm=norm, eps=eps)
                        image_clean = image.detach().clone().requires_grad_(requires_grad=True) 
                        output_clean = model_ori(image_clean)
                        image = BoundedTensor(image, ptb)
                        pred = model(image)
                        lb, ub,A = model.compute_bounds(return_A=True,needed_A_dict={model.output_name[0]:model.input_name[0]},)
                        '''
                        # 1. testing if lb == ub == pred when eps = 0
                        assert (lb == ub).all() and torch.allclose(lb,pred,rtol=1e-5) and torch.allclose(ub,pred,rtol=1e-5)
                        # 2. test if A matrix equals to gradient of the input
                        # get output's grad with respect to the input without iterating through torch.autograd.grad:
                        # https://stackoverflow.com/questions/64988010/getting-the-outputs-grad-with-respect-to-the-input
                        uA = A[model.output_name[0]][model.input_name[0]]['uA']
                        lA = A[model.output_name[0]][model.input_name[0]]['lA']
                        assert (uA==lA).all()
                        assert (torch.autograd.functional.jacobian(model_ori,image_clean).sum(dim=2)==uA).all()
                        assert (torch.autograd.functional.jacobian(model_ori,image_clean).sum(dim=2)==lA).all()
                        # double check
                        input_grads = torch.zeros(uA.shape)
                        for i in range(N):
                            for j in range(n_classes):
                                input_grads[i][j]=torch.autograd.grad(outputs=output_clean[i,j], inputs=image_clean, retain_graph=True)[0].sum(dim=0)
                        assert (input_grads==uA).all()
                        assert (input_grads==lA).all()
                        '''
                        # 3. test when eps = 0.3 (uncommented)
                        if self.generate:
                            torch.save(
                                {'model': model_ori.state_dict(),
                                'input': image,
                                'lb': lb,
                                'ub': ub}, 'data/conv1d_test_data_{}-{}-{}'.format(kernel_size, padding, stride)
                            )

                        if not self.generate:
                            lb_ref = data['lb']
                            ub_ref = data['ub']
                            assert torch.allclose(lb, lb_ref, 1e-3)
                            assert torch.allclose(ub, ub_ref, 1e-3)


if __name__ == '__main__':
    testcase = TestConv1D(generate=False)
    testcase.test()

"""Test max pooling."""

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
    def __init__(self, kernel_size=4, stride=4, padding=0, conv_padding=0):
        super(Model, self).__init__()
        self.n_n_conv2d = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 1, 'padding': conv_padding, 'kernel_size': (2, 2), 'stride': [1, 1], 'in_channels': 1, 'bias': True})
        self.n_n_maxpool = nn.MaxPool2d(**{'kernel_size': [kernel_size, kernel_size], 'ceil_mode': False, 'stride': [stride, stride], 'padding': [padding, padding]})
        self.n_n_conv2d_2 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 1, 'padding': [conv_padding, conv_padding], 'kernel_size': (2, 2), 'stride': [1, 1], 'in_channels': 1, 'bias': True})
        self.n_n_maxpool_2 = nn.MaxPool2d(**{'kernel_size': [kernel_size, kernel_size], 'ceil_mode': False, 'stride': [stride, stride], 'padding': [padding, padding]})
        self.n_n_flatten_Flatten = nn.Flatten(**{'start_dim': 1})

        self.n_n_dense = None

        self.n_n_activation_Flatten = nn.Flatten(**{'start_dim': 1})

    def forward(self, *inputs):
        t_ImageInputLayer, = inputs
        t_conv2d = self.n_n_conv2d(t_ImageInputLayer)
        t_conv2d_relu = F.relu(t_conv2d)
        t_maxpool = self.n_n_maxpool(t_conv2d_relu)[:, :, :, :]
        t_conv2d_max = self.n_n_conv2d_2(t_maxpool)
        t_conv2d_max = F.relu(t_conv2d_max)
        # t_maxpool_2 = self.n_n_maxpool_2(t_conv2d_max)
        t_flatten_Transpose = t_conv2d_max.permute(*[0, 2, 3, 1])
        t_flatten_Flatten = self.n_n_flatten_Flatten(t_flatten_Transpose)
        t_flatten_Unsqueeze = torch.unsqueeze(t_flatten_Flatten, 2)
        t_flatten_Unsqueeze = torch.unsqueeze(t_flatten_Unsqueeze, 3)

        if self.n_n_dense is None:
            self.n_n_dense = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 2, 'padding': [0, 0], 'kernel_size': (1, 1), 'stride': [1, 1], 'in_channels': t_flatten_Unsqueeze.shape[1], 'bias': True})
        t_dense = self.n_n_dense(t_flatten_Unsqueeze)
        t_activation_Flatten = self.n_n_activation_Flatten(t_dense)

        return t_activation_Flatten

class TestMaxPool(TestCase):
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName,
            seed=1, ref_path=None,
            generate=generate)

    def test(self):
        np.random.seed(123)

        N = 2

        for kernel_size in [3,4]:
            for padding in [0,1]:
                    for conv_padding in [0,1]:
                        print(kernel_size, padding, kernel_size, conv_padding)

                        model_ori = Model(kernel_size=kernel_size, padding=padding, stride=kernel_size, conv_padding=conv_padding)
                        if not self.generate:
                            data = torch.load('data/maxpool_test_data_{}-{}-{}-{}'.format(kernel_size, padding, kernel_size, conv_padding))
                            image = data['input']
                            model_ori(image)
                            model_ori.load_state_dict(data['model'])
                        else:
                            image = torch.rand([N, 1, 28, 28])
                            model_ori(image)


                        if self.generate:
                            conv_mode = "matrix"
                        else:
                            conv_mode = "patches"

                        model = BoundedModule(model_ori, image, device="cpu", bound_opts={"conv_mode": conv_mode})
                        eps = 0.3
                        norm = np.inf
                        ptb = PerturbationLpNorm(norm=norm, eps=eps)
                        image = BoundedTensor(image, ptb)
                        pred = model(image)

                        lb, ub = model.compute_bounds()


                        if self.generate:
                            torch.save(
                                {'model': model_ori.state_dict(),
                                'input': image,
                                'lb': lb,
                                'ub': ub}, 'data/maxpool_test_data_{}-{}-{}-{}'.format(kernel_size, padding, kernel_size, conv_padding)
                            )

                        if not self.generate:
                            lb_ref = data['lb']
                            ub_ref = data['ub']

                            assert torch.allclose(lb, lb_ref, 1e-4)
                            assert torch.allclose(ub, ub_ref, 1e-4)


if __name__ == '__main__':
    testcase = TestMaxPool(generate=False)
    testcase.test()

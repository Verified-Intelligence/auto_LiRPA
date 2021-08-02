import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *  
from testcase import TestCase

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        return x.view((x.shape[0], -1))

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.n_n_conv2d = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 32, 'padding': [0, 0], 'kernel_size': (2, 2), 'stride': [1, 1], 'in_channels': 1, 'bias': True})
    self.n_n_average_pooling2d = nn.MaxPool2d(**{'kernel_size': [4, 4], 'ceil_mode': False, 'stride': [4, 4], 'padding': [0, 0]})
    self.n_n_flatten_Flatten = nn.Flatten(**{'start_dim': 1})
    self.n_n_dense = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 10, 'padding': [0, 0], 'kernel_size': (1, 1), 'stride': [1, 1], 'in_channels': 1152, 'bias': True})
    self.n_n_activation_Flatten = nn.Flatten(**{'start_dim': 1})

  def forward(self, *inputs):
    t_ImageInputLayer, = inputs
    t_conv2d = self.n_n_conv2d(t_ImageInputLayer)
    t_conv2d_relu = F.relu(t_conv2d)
    t_average_pooling2d = self.n_n_average_pooling2d(t_conv2d_relu)[:, :, :, :]
    t_flatten_Transpose = t_average_pooling2d.permute(*[0, 2, 3, 1])
    t_flatten_Flatten = self.n_n_flatten_Flatten(t_flatten_Transpose)
    t_flatten_Unsqueeze = torch.unsqueeze(t_flatten_Flatten, 2)
    t_flatten_Unsqueeze = torch.unsqueeze(t_flatten_Unsqueeze, 3)
    t_dense = self.n_n_dense(t_flatten_Unsqueeze)
    t_activation_Flatten = self.n_n_activation_Flatten(t_dense)
    return t_activation_Flatten

class TestConv(TestCase): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName, 
            seed=1, ref_path=None,
            generate=generate)

    def test(self):
        np.random.seed(123)
        models = [2, 3]
        paddings = [1, 2]
        strides = [1, 3]

        model_ori = Model()
        data = torch.load('data/maxpool_test_data')
        model_ori.load_state_dict(data['model'])

        N = 2
        n_classes = 10
        image = data['input']
        # image = torch.rand([N,1,28,28])
        # image = image.to(torch.float32) / 255.0

        model = BoundedModule(model_ori, image, device="cpu", bound_opts={"conv_mode": "matrix"})
        eps = 0.3
        norm = np.inf
        ptb = PerturbationLpNorm(norm=norm, eps=eps)
        image = BoundedTensor(image, ptb)
        pred = model(image)
        lb, ub = model.compute_bounds()

        lb_ref = data['lb']
        ub_ref = data['ub']

        assert torch.allclose(lb, lb_ref, 1e-4)
        assert torch.allclose(ub, ub_ref, 1e-4)

        # lb, ub = model.compute_bounds(x=(image,), method="CROWN-Optimized")

        # torch.save({'input': image, 'model': model_ori.state_dict(), 'lb': lb, 'ub': ub}, 'data/maxpool_test_data')


if __name__ == '__main__':
    testcase = TestConv()
    testcase.test()

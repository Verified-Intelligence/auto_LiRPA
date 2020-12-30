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

class cnn_model(nn.Module):
    def __init__(self, layers, padding, stride):
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
        self.module_list.append(Flatten())
        self.module_list.append(nn.Linear(3 * length * length, 256))
        self.module_list.append(nn.Linear(256, 10))
        self.model = nn.Sequential(*self.module_list)

    def forward(self, x):
        x = self.model(x)
        return x

class TestConv(TestCase): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName, 
            seed=1, ref_path=None,
            generate=generate)

    def test(self):
        models = [2, 3]
        paddings = [1, 2]
        strides = [1, 3]

        N = 2
        n_classes = 10
        image = torch.randn(N, 1, 28, 28)
        image = image.to(torch.float32) / 255.0

        for layer_num in models:
            for padding in paddings:
                for stride in strides:
                    try:
                        model_ori = cnn_model(layer_num, padding, stride)
                    except:
                        continue

                    model = BoundedModule(model_ori, image, device="cpu", bound_opts={"conv_mode": "patches"})
                    eps = 0.3
                    norm = np.inf
                    ptb = PerturbationLpNorm(norm=norm, eps=eps)
                    image = BoundedTensor(image, ptb)
                    pred = model(image)
                    lb, ub = model.compute_bounds()

                    model = BoundedModule(model_ori, image, device="cpu", bound_opts={"conv_mode": "matrix"})
                    pred = model(image)
                    lb_ref, ub_ref = model.compute_bounds()

                    assert lb.shape == ub.shape == torch.Size((N, n_classes))    
                    self.assertEqual(lb, lb_ref)
                    self.assertEqual(ub, ub_ref)

if __name__ == '__main__':
    testcase = TestConv()
    testcase.test()

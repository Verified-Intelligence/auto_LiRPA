"""Test classes in auto_LiRPA/bound_ops.py"""
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase  

class cnn_MNIST(nn.Module):
    def __init__(self):
        super(cnn_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 784)
        x = 2.0 * x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return 0.5 * x

class TestConstant(TestCase): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName, 
            seed=1, ref_path='data/constant_test_data',
            generate=generate)

    def test(self):
        model = cnn_MNIST()
        checkpoint = torch.load("../examples/vision/pretrain/mnist_cnn_small.pth", map_location="cpu")
        model.load_state_dict(checkpoint)

        N = 2
        n_classes = 10
        image = torch.randn(N, 1, 28, 28)
        image = image.to(torch.float32) / 255.0

        model = BoundedModule(model, torch.empty_like(image), device="cpu")
        eps = 0.3
        norm = np.inf
        ptb = PerturbationLpNorm(norm=norm, eps=eps)
        image = BoundedTensor(image, ptb)
        pred = model(image)
        lb, ub = model.compute_bounds()

        assert lb.shape == ub.shape == torch.Size((2, 10))    

        self.result = (lb, ub)
        self.check()

if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestConstant(generate=False)
    testcase.test()        

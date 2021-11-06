import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import sys
sys.path.append('../examples/vision')
import models
from testcase import TestCase

class cnn_4layer_resnet(nn.Module):
    def __init__(self):
        super(cnn_4layer_resnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.shortcut = nn.Conv2d(3, 3, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(168, 10)

    def forward(self, x):
        x_ = x
        x = F.relu(self.conv1(self.bn(x)))
        x += self.shortcut(x_)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.fc1(x)

        return x

class TestResnetPatches(TestCase): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName, 
            seed=1234, ref_path='data/rectangle_patches_test_data',
            generate=generate)

    def test(self):
        model_oris = [
            cnn_4layer_resnet(),
        ]
        self.result = []
        if not self.generate:
            self.reference = torch.load(self.ref_path)

        for model_ori in model_oris:
            conv_mode = 'patches' # conv_mode can be set as 'matrix' or 'patches'        
                
            normalize = torchvision.transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
            test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, 
                            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize]))
            N = 1
            n_classes = 10

            image = torch.Tensor(test_data.data[:N]).reshape(N,3,32,32)
            image = image[:, :, :28, :]
            image = image.to(torch.float32) / 255.0

            model = BoundedModule(model_ori, image, bound_opts={"conv_mode": conv_mode})

            ptb = PerturbationLpNorm(norm = np.inf, eps = 0.03)
            image = BoundedTensor(image, ptb)
            pred = model(image)
            lb, ub = model.compute_bounds(IBP=False, C=None, method='backward')
            self.result += [lb, ub]

        self.check()

if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestResnetPatches(generate=False)
    testcase.test()
from numpy.core.numeric import allclose
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

class cnn_4layer_b(nn.Module):
    def __init__(self, paddingA, paddingB):
        super().__init__()
        self.paddingA = paddingA
        self.paddingB = paddingB

        self.padA = nn.ZeroPad2d(self.paddingA)
        self.padB = nn.ZeroPad2d(self.paddingB)

        self.conv1 = nn.Conv2d(3, 32, (5,5), stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 128, (4,4), stride=2, padding=1)

        self.linear = None
        self.fc = nn.Linear(250, 10)

    def forward(self, x):
        x = self.padA(x)
        x = self.conv1(x)
        x = self.conv2(self.padB(F.relu(x)))
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        if self.linear is None:
            self.linear = nn.Linear(x.size(1), 250)
        x = self.linear(x)
        return self.fc(F.relu(x))

class TestDistinctPatches(TestCase): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName, 
            seed=1234, ref_path='data/resnet_patches_test_data',
            generate=generate)

    def test(self):
        cases = [(2,1,2,1), (0,0,0,0), (1,3,3,1), (2,2,3,1)]
        for i in range(4):
            for j in range(4):
                paddingA = cases[i]
                paddingB = cases[j]


                print(paddingA, paddingB)

                model_ori = cnn_4layer_b(paddingA, paddingB)

                conv_mode = 'patches' # conv_mode can be set as 'matrix' or 'patches'        
                    
                normalize = torchvision.transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
                test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, 
                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize]))
                N = 1
                n_classes = 10

                seed = 1234
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                random.seed(seed)
                np.random.seed(seed)

                image = torch.Tensor(test_data.data[:N]).reshape(N,3,32,32)
                image = image.to(torch.float32) / 255.0
                pred = model_ori(image)
                for m in model_ori.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.running_mean.data.copy_(torch.randn_like(m.running_mean))
                        m.running_var.data.copy_(torch.abs(torch.randn_like(m.running_var)))

                model = BoundedModule(model_ori, image, bound_opts={"conv_mode": conv_mode})

                ptb = PerturbationLpNorm(norm = np.inf, eps = 0.03)
                image = BoundedTensor(image, ptb)
                lb, ub = model.compute_bounds(x=(image,), IBP=False, C=None, method='backward')

                # matrix mode
                conv_mode = 'matrix'

                seed = 1234
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                random.seed(seed)
                np.random.seed(seed)
                for m in model_ori.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.running_mean.data.copy_(torch.randn_like(m.running_mean))
                        m.running_var.data.copy_(torch.abs(torch.randn_like(m.running_var)))
                model = BoundedModule(model_ori, image, bound_opts={"conv_mode": conv_mode})

                ptb = PerturbationLpNorm(norm = np.inf, eps = 0.03)
                image = BoundedTensor(image, ptb)
                pred = model(image)

                lb_ref, ub_ref = model.compute_bounds(x=(image,), IBP=False, C=None, method='backward')

                assert torch.allclose(lb, lb_ref)
                assert torch.allclose(ub, ub_ref)
        

if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestDistinctPatches(generate=False)
    testcase.test()
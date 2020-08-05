"""Test classes in auto_LiRPA/bound_ops.py"""
import torch
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

parser = argparse.ArgumentParser()
parser.add_argument('--gen_ref', action='store_true', help='generate reference results')
args, unknown = parser.parse_known_args()   

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

def test():
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

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

    path = 'data/constant_test_data'
    if args.gen_ref:
        torch.save((lb, ub), path)
    else:
        lb_ref, ub_ref = torch.load(path)
        print(lb)
        print(lb_ref)
        assert torch.allclose(lb, lb_ref)
        assert torch.allclose(ub, ub_ref)

if __name__ == '__main__':
    test()

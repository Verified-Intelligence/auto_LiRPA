"""
A simple example for saving intermediate bounds.
"""
import os
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten

def mnist_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

model = mnist_model()
# Optionally, load the pretrained weights.
checkpoint = torch.load(
    os.path.join(os.path.dirname(__file__), 'pretrained/mnist_a_adv.pth'),
    map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

test_data = torchvision.datasets.MNIST(
    './data', train=False, download=True,
    transform=torchvision.transforms.ToTensor())
# For illustration we only use 2 image from dataset
N = 2
n_classes = 10
image = test_data.data[:N].view(N,1,28,28)
true_label = test_data.targets[:N]
# Convert to float
image = image.to(torch.float32) / 255.0
if torch.cuda.is_available():
    image = image.cuda()
    model = model.cuda()

lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
print('Running on', image.device)

eps = 0.3
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, eps = eps)
image = BoundedTensor(image, ptb)

lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})
lb, ub = lirpa_model.compute_bounds(x=(image,), method='CROWN-Optimized')
# Intermediate layer bounds are returned as a dictionary, and if an argument is given,
# a pytorch checkpoint will also be saved to disk.
save_dict = lirpa_model.save_intermediate('./mnist_a_adv_bounds.pt')
# To avoid saving the file and get just the bounds, call without any arguments:
# save_dict = lirpa_model.save_intermediate()

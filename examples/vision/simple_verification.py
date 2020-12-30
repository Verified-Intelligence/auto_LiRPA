import torch.nn as nn
import torch.nn.functional as F
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import models

## Step 1: Define computational graph by implementing forward()
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# This simple model comes from https://github.com/locuslab/convex_adversarial
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
# Load the pretrained weights
checkpoint = torch.load(os.path.join(os.path.dirname(__file__),"pretrain/kw_mnist.pth"))
model.load_state_dict(checkpoint)

## Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
# For illustration we only use 2 image from dataset
N = 2
n_classes = 10
image = test_data.data[:N].view(N,1,28,28)
true_label = test_data.targets[:N]
# Convert to float
image = image.to(torch.float32) / 255.0

## Step 3: wrap model with auto_LiRPA
# The second parameter is for constructing the trace of the computational graph, and its content is not important.
model = BoundedModule(model, torch.empty_like(image))
# For larger convolutional models, setting bound_opts={"conv_mode": "patches"} is more efficient.
# model = BoundedModule(model, torch.empty_like(image), bound_opts={"conv_mode": "patches"})

## Step 4: Compute bounds using LiRPA given a perturbation
eps = 0.1
norm = np.inf
# ptb = PerturbationL0Norm(eps=eps)
ptb = PerturbationLpNorm(norm = norm, eps = eps)
image = BoundedTensor(image, ptb)
# Get model prediction as usual
pred = model(image)
label = torch.argmax(pred, dim=1).cpu().numpy()

## Step 5: Compute bounds for final output
for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']:
    lb, ub = model.compute_bounds(x=(image,), method=method.split()[0])
    lb = lb.detach().cpu().numpy()
    ub = ub.detach().cpu().numpy()
    print("Bounding method:", method)
    for i in range(N):
        print("Image {} top-1 prediction {} ground-truth {}".format(i, label[i], true_label[i]))
        for j in range(n_classes):
            indicator = '(ground-truth)' if j == true_label[i] else ''
            print("f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}".format(
                j=j, l=lb[i][j], u=ub[i][j], ind=indicator))
    print()


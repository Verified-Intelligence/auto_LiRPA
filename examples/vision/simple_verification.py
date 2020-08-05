import torch.nn as nn
import torch.nn.functional as F
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import models

## Step 1: Define computational graph by implementing forward()
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
        # print(x.shape)
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class simple_MNIST(nn.Module):
    def __init__(self):
        super(simple_MNIST, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.fc3(x)
        return x

model = cnn_MNIST()
# Load the pretrained weights
checkpoint = torch.load(os.path.join(os.path.dirname(__file__),"pretrain/mnist_cnn_small.pth"))
model.load_state_dict(checkpoint)

## Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
# For illustration we only use 2 image from dataset
N = 2
n_classes = 10
image = test_data.data[:N].view(N,1,28,28).cuda()
# Convert to float
image = image.to(torch.float32) / 255.0

## Step 3: wrap model with auto_LiRPA
# The second parameter is for constructing the trace of the computational graph, and its content is not important.
model = BoundedModule(model, torch.empty_like(image), device="cuda")

## Step 4: Compute bounds using LiRPA given a perturbation
eps = 0.3
norm = np.inf
# ptb = PerturbationL0Norm(eps=eps)
ptb = PerturbationLpNorm(norm = norm, eps = eps)
image = BoundedTensor(image, ptb)
# Get model prediction as usual
pred = model(image)
# label = torch.argmax(pred, dim=1).cpu().numpy()
# Compute bounds
lb, ub = model.compute_bounds(IBP=False)

## Step 5: Final output
# pred = pred.detach().cpu().numpy()
lb = lb.detach().cpu().numpy()
ub = ub.detach().cpu().numpy()
for i in range(N):
    # print("Image {} top-1 prediction {}".format(i, label[i]))
    for j in range(n_classes):
        print("f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f}".format(j=j, l=lb[i][j], u=ub[i][j]))
    print()


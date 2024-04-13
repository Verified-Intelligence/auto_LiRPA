"""
A simple example for bounding neural network outputs using LP/MIP solvers.

Auto_LiRPA supports constructing LP/MIP optimization formulations (using
Gurobi).  This example serves as a skeleton for using the build_solver_module()
method to obtain LP/MIP formulations of neural networks.

Note that alpha-CROWN is used to calculate intermediate layer bounds for
constructing the convex relaxation of ReLU neurons. So we are actually using
"alpha-CROWN+MIP" or "alpha-CROWN+LP" here. Calculating intermediate layer
bounds using LP/MIP is often impractical due to the high cost.
"""
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import gurobipy as grb

## Step 1: Define computational graph by implementing forward()
# You can create your own model here.
class mnist_model(nn.Module):
    def __init__(
            self, input_size=28*28, hidden_size=128,
            hidden_size_2=64, output_size=10):
        super(mnist_model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 784)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

model = mnist_model()
# Optionally, load the pretrained weights.
checkpoint = torch.load('../vision/pretrained/mnist_fc_3layer.pth')
model.load_state_dict(checkpoint)

## Step 2: Prepare dataset.
test_data = torchvision.datasets.MNIST(
    './data', train=False, download=True,
    transform=torchvision.transforms.ToTensor())
# For illustration we only use 1 image from dataset.
N = 1
n_classes = 10
image = test_data.data[:N].view(N, 1, 28, 28)
true_label = test_data.targets[:N]
image = image.to(torch.float32) / 255.0

## Step 3: Define perturbation.
eps = 0.03
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, eps = eps)
# Here we only use one image as input
image = BoundedTensor(image[0], ptb)

## Step 4: Compute the bounds of different methods.
# For CROWN/alpha-CROWN, we use the compute_bounds() method.
# For LP and MIP, we use the build_solver_module() method.
result = {}
# Note that here 'lp' or 'mip' are essentially 'alpha-CROWN+lp' and 'alpha-CROWN+mip'.
# We use alpha-CROWN to calculate all the intermediate layer bounds for LP/MIP, because
# using MIP/LP for all intermediate neurons will be very slow.
for method in ['alpha-CROWN','lp','mip']:
    # To get clean results and avoid interference among methods, we create a
    # new BoundedModule object.  However, in your production code please pay
    # attention that BoundedModule() has high construction overhead.
    lirpa_model = BoundedModule(model, torch.empty_like(image[0]), device=image.device)
    # Call alpha-CROWN first, which gives all intermediate layer bounds.
    lb, ub = lirpa_model.compute_bounds(x=(image,), method='alpha-CROWN')

    if method != 'alpha-CROWN':
        lb = torch.full_like(lb, float('nan'))
        ub = torch.full_like(ub, float('nan'))
        # Obtain the optimizer (Gurobi) variables for the output layer.
        # Auto_LiRPA will construct the LP/MIP formulation based on computation graph.
        # Note that pre-activation bounds are required for using this function.
        # Preactivation bounds have been computed using alpha-CROWN above.
        solver_vars = lirpa_model.build_solver_module(model_type=method)
        # Set some parameters for Gurobi optimizer.
        lirpa_model.solver_model.setParam('OutputFlag', 0)
        for i in range(n_classes):
            print(f'Solving class {i} with method {method}')
            # Now you can define objectives based on the variables on the output layer.
            # And then solve them using gurobi. Here we just output the lower and upper
            # bounds for each output neuron.
            # Solve upper bound.
            lirpa_model.solver_model.setObjective(solver_vars[i], grb.GRB.MAXIMIZE)
            lirpa_model.solver_model.optimize()
            # If the solver does not terminate, you will get a NaN.
            if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                ub[0][i] = lirpa_model.solver_model.objVal
            # Solve lower bound.
            lirpa_model.solver_model.setObjective(solver_vars[i], grb.GRB.MINIMIZE)
            lirpa_model.solver_model.optimize()
            if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                lb[0][i] = lirpa_model.solver_model.objVal
    result[method] = (lb, ub)

# Step 5: output the final results of each method.
for method in result.keys():    
    print(f'Bounding method: {method}')
    lb, ub = result[method]
    for i in range(n_classes):
        print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f}'.format(
            j=i, l=lb[0][i].item(), u=ub[0][i].item()))

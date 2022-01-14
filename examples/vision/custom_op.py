""" A example for custom operators.

In this example, we create a custom operator called "PlusConstant", which can 
be written as "f(x) = x + c" for some constant "c" (an attribute of the operator). 
"""
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor, register_custom_op
from auto_LiRPA.operators import Bound
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten

""" Step 1: Define a `torch.autograd.Function` class to declare and implement the 
computation of the operator. """
class PlusConstantOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, const):
        """ In this function, define the arguments and attributes of the operator.
        "custom::PlusConstant" is the name of the new operator, "x" is an argument
        of the operator, "const_i" is an attribute which stands for "c" in the operator. 
        There can be multiple arguments and attributes. For attribute naming, 
        use a suffix such as "_i" to specify the data type, where "_i" stands for 
        integer, "_t" stands for tensor, "_f" stands for float, etc. """
        return g.op('custom::PlusConstant', x, const_i=const)

    @staticmethod
    def forward(ctx, x, const):
        """ In this function, implement the computation for the operator, i.e., 
        f(x) = x + c in this case. """
        return x + const

""" Step 2: Define a `torch.nn.Module` class to declare a module using the defined
custom operator. """
class PlusConstant(nn.Module):
    def __init__(self, const=1):
        super().__init__()
        self.const = const

    def forward(self, x):
        """ Use `PlusConstantOp.apply` to call the defined custom operator. """
        return PlusConstantOp.apply(x, self.const)

""" Step 3: Implement a Bound class to support bound computation for the new operator. """
class BoundPlusConstant(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        """ `const` is an attribute and can be obtained from the dict `attr` """
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.const = attr['const']

    def forward(self, x):
        return x + self.const

    def bound_backward(self, last_lA, last_uA, x):
        """ Backward mode bound propagation """
        print('Calling bound_backward for custom::PlusConstant')
        def _bound_oneside(last_A):
            # If last_lA or last_uA is None, it means lower or upper bound
            # is not required, so we simply return None.
            if last_A is None:
                return None, 0
            # The function f(x) = x + c is a linear function with coefficient 1.
            # Then A · f(x) = A · (x + c) = A · x + A · c.
            # Thus the new A matrix is the same as the last A matrix:
            A = last_A
            # For bias, compute A · c and reduce the dimensions by sum:
            bias = last_A.sum(dim=list(range(2, last_A.ndim))) * self.const
            return A, bias
        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)
        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v):
        """ IBP computation """
        print('Calling interval_propagate for custom::PlusConstant')
        # Interval bound of the input
        h_L, h_U = v[0]
        # Since this function is monotonic, we can get the lower bound and upper bound
        # by applying the function on h_L and h_U respectively.
        lower = h_L + self.const
        upper = h_U + self.const
        return lower, upper

""" Step 4: Register the custom operator """
register_custom_op("custom::PlusConstant", BoundPlusConstant)

# Use the `PlusConstant` module in model definition
model = nn.Sequential(
    Flatten(),
    nn.Linear(28 * 28, 256),
    PlusConstant(const=1),
    nn.Linear(256, 10),
)
print("Model:", model)

test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
N = 1
n_classes = 10
image = test_data.data[:N].view(N,1,28,28)
true_label = test_data.targets[:N]
image = image.to(torch.float32) / 255.0
if torch.cuda.is_available():
    image = image.cuda()
    model = model.cuda()

lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)

eps = 0.3
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, eps = eps)
image = BoundedTensor(image, ptb)
pred = lirpa_model(image)
label = torch.argmax(pred, dim=1).cpu().detach().numpy()

for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']:
    print("Bounding method:", method)
    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
    for i in range(N):
        print("Image {} top-1 prediction {} ground-truth {}".format(i, label[i], true_label[i]))
        for j in range(n_classes):
            indicator = '(ground-truth)' if j == true_label[i] else ''
            print("f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}".format(
                j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
    print()


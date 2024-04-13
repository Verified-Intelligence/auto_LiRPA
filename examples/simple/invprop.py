"""
A toy example for bounding neural network outputs under input perturbations using INVPROP

See https://arxiv.org/abs/2302.01404
"""
import torch
from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

class simple_model(torch.nn.Module):
    """
    A very simple 2-layer neural network for demonstration.
    """
    def __init__(self):
        super().__init__()
        # Weights of linear layers.
        self.w1 = torch.tensor([[1., -1.], [2., -1.]])
        self.w2 = torch.tensor([[1., -1.]])

    def forward(self, x):
        # Linear layer.
        z1 = x.matmul(self.w1.t())
        # Relu layer.
        hz1 = torch.nn.functional.relu(z1)
        # Linear layer.
        z2 = hz1.matmul(self.w2.t())
        return z2


model = simple_model()

# Input x.
x = torch.tensor([[1., 1.]])
# Lowe and upper bounds of x.
lower = torch.tensor([[-1., -2.]])
upper = torch.tensor([[2., 1.]])

# Compute bounds using LiRPA using the given lower and upper bounds.
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
bounded_x = BoundedTensor(x, ptb)

# INVPROP configuration
# apply_output_constraints_to: list of layer names or types to which the output
#     constraints should be applied. Here, they will be applied to all layers of type
#     'BoundMatMul' and 'BoundInput'. To only apply them to specific layers, use their
#     names, e.g. ['/0', '/z1']. The currently recommended way to get those names is
#     either to first construct an instance of BoundedModule with arbitrary bound_opts,
#     print it to stdout and inspect their names manually, or to access the layer names
#     as lirpa_model.final_node().inputs[0].inputs[0].name
# tighten_input_bounds: whether to tighten the input bounds. This will modify the
#     perturbation of the input. If set, apply_output_constraints_to should contain
#     'BoundInput' or the corresponding layer name. Otherwise, this will have no effect.
#     Similiar, adding 'BoundInput' to apply_output_constraints_to will have no effect
#     unless tighten_input_bounds is set.
# best_of_oc_and_no_oc: Using output constraints may sometimes lead to worse results,
#     because the optimization might find bad local minima. If this is set to True,
#     every optimization step will be run twice, once with and once without output
#     constraints, and the better result will be chosen.
# directly_optimize: Usually, only linear layers preceeding non-linear layers are
#     optimized using output constraints. If you want to optimize a specific layer that
#     would usually be skipped, add it's name to this list. This is most likely to be
#     used when preimages should be computed as they might use linear combinations of
#     the inputs. This requires the use of sequential linear layers. For detailed
#     examples, see https://github.com/kothasuhas/verify-input
# oc_lr: Learning rate for the optimization of output constraints.
# share_gammas: Whether neurons in each layer should share the same gamma

lirpa_model = BoundedModule(model, torch.empty_like(x), bound_opts={
    'optimize_bound_args': {
        'apply_output_constraints_to': ['BoundMatMul', 'BoundInput'],
        'tighten_input_bounds': True,
        'best_of_oc_and_no_oc': False,
        'directly_optimize': [],
        'oc_lr': 0.1,
        'share_gammas': False,
        'iteration': 1000,
    }
})
# To dynamically set the apply_output_constraints_to option, set it to `[]` in the
# above code, and then use the following:
# lirpa_model.set_bound_opts({
#   'optimize_bound_args': {
#     'apply_output_constraints_to': [
#       lirpa_model.final_node().inputs[0].inputs[0].inputs[0].name,
#       lirpa_model.final_node().inputs[0].inputs[0].name,
#     ]
#   }
# })

# The scalar output must be <= -1
# Constraints have the shape [1, num_constraints, num_output_neurons]
# They are treated as conjunctions, i.e., all constraints must be satisfied.
lirpa_model.constraints = torch.ones(1,1,1)
# Thresholds have the shape [num_constraints]
lirpa_model.thresholds = torch.tensor([-1.])

print(f"Original perturbation: x0: [{ptb.x_L[0][0]}, {ptb.x_U[0][0]}], x1: [{ptb.x_L[0][1]}, {ptb.x_U[0][1]}]")
lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='alpha-CROWN')
tightened_ptb = lirpa_model['/0'].perturbation
print(f"Tightened perturbation: x0: [{tightened_ptb.x_L[0][0]}, {tightened_ptb.x_U[0][0]}], x1: [{tightened_ptb.x_L[0][1]}, {tightened_ptb.x_U[0][1]}]")

# For the bounds without output constraints, refer to toy.py
print(f'alpha-CROWN bounds without output constraints: lower=-3, upper=2')
print(f'alpha-CROWN bounds with output constraints: lower={lb.item()}, upper={ub.item()}')
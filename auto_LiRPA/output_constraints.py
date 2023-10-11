
from .utils import *
from .bound_ops import *
from .operators import Bound

from typing import TYPE_CHECKING, Optional, List
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def backward_general_with_output_constraint(
    self: 'BoundedModule',
    bound_node,
    C,
    start_backporpagation_at_node = None,
    bound_lower=True,
    bound_upper=True,
    average_A=False,
    need_A_only=False,
    unstable_idx=None,
    update_mask=None,
    verbose=True,
):
    assert start_backporpagation_at_node is None
    assert not isinstance(C, str)

    neurons_in_layer = 1
    for d in bound_node.output_shape[1:]:
        neurons_in_layer *= d

    # backward_general uses C to compute batch_size, output_dim and output_shape, just like below.
    # When output constraints are applied, it will perform a different backpropagation,
    # but those variables need to be computed regardless. So we need to retain the original C
    # and pass it on to backward_general. If initial_As is set (which it is, if this code here
    # is executed), it will not use C for anything else.
    orig_C = C

    C, batch_size, output_dim, output_shape = self._preprocess_C(C, bound_node)
    device = bound_node.device

    num_constraints = self.constraints.size(0)

    # 1) Linear: Hx + d
    # Result is a tensor, <= 0 for all entries if output constraint is satisfied
    H = self.constraints.T  # (output_neurons, constraints)
    d = self.thresholds.squeeze(0)  # (constraints)
    assert H.ndim == 2
    assert H.size(1) == num_constraints
    assert d.ndim == 1
    assert d.size(0) == num_constraints 

    linear_Hxd_layer_weight_value = nn.Parameter(H.to(C))
    linear_Hxd_layer_weight = BoundParams(
        ori_name="/linear_Hxd_layer_weight",
        value=None,
        perturbation=None,
    )
    linear_Hxd_layer_weight.name = "linear_Hxd_layer_weight"
    linear_Hxd_layer_weight.lower = linear_Hxd_layer_weight_value
    linear_Hxd_layer_weight.upper = linear_Hxd_layer_weight_value

    linear_Hxd_layer_bias_value = nn.Parameter(d.float().to(device))
    linear_Hxd_layer_bias = BoundParams(
        ori_name="/linear_Hxd_layer_bias",
        value=None,
        perturbation=None,
    )
    linear_Hxd_layer_bias.name = "linear_Hxd_layer_bias"
    linear_Hxd_layer_bias.lower = linear_Hxd_layer_bias_value
    linear_Hxd_layer_bias.upper = linear_Hxd_layer_bias_value

    linear_Hxd_layer = BoundLinear(
        attr=None,
        inputs=[
            self.final_node(),
            linear_Hxd_layer_weight,
            linear_Hxd_layer_bias,
        ],
        output_index=0,
        options=self.bound_opts,
    )
    linear_Hxd_layer.name = "/linear_Hxd_layer"
    linear_Hxd_layer.device = device
    linear_Hxd_layer.perturbed = True
    linear_Hxd_layer.output_shape = torch.Size([1, num_constraints])
    linear_Hxd_layer.batch_dim = bound_node.batch_dim

    # 2) Gamma
    # A seperate gamma per output constraint. All gammas are always positive.
    # Note that we're not using a different gamma per neuron in the optimized layer.
    # That would be even more precise, but much slower and would require more memory.
    gamma_layer_weight = BoundParams(
        ori_name="/gamma_layer_weight",
        value=None,
        perturbation=None,
    )
    gamma_layer_weight.name = "gamma_layer_weight"
    assert bound_node.gammas.ndim == 2
    assert bound_node.gammas.size(0) == 2
    assert bound_node.gammas.size(1) == num_constraints
    gamma_layer_weight.lower = torch.diag(bound_node.gammas[0])  # (5, 5)
    gamma_layer_weight.upper = torch.diag(-bound_node.gammas[1])  # (5, 5)
    gamma_layer = BoundLinear(
        attr=None,
        inputs=[linear_Hxd_layer, gamma_layer_weight],
        output_index=0,
        options=self.bound_opts,
    )
    gamma_layer.name = "/gamma_layer"
    gamma_layer.device = device
    gamma_layer.perturbed = True
    gamma_layer.input_shape = linear_Hxd_layer.output_shape
    gamma_layer.output_shape = gamma_layer.input_shape
    gamma_layer.batch_dim = bound_node.batch_dim
    gamma_layer.use_seperate_weights_for_lower_and_upper_bounds = True

    # 3) Sum
    # Sum over all constraints.
    # In the dualization, if there are multiple output constraints, we have
    # min g(x) + gamma_1... + gamma_2...      or
    # max g(x) - gamma_1... - gamma_2...
    # Here, we only compute the sum over all gammas, the addition of g(x) is handled
    # further down.
    sum_weight_value = nn.Parameter(torch.ones((5,1), device=device))
    sum_weight = BoundParams(
        ori_name="/sum_weight",
        value=None,
        perturbation=None,
    )
    sum_weight.name = "sum_weight"
    sum_weight.lower = sum_weight_value
    sum_weight.upper = sum_weight_value
    sum_layer = BoundLinear(
        attr=None,
        inputs=[gamma_layer, sum_weight],
        output_index=0,
        options=self.bound_opts,
    )
    sum_layer.name = "/sum_layer"
    sum_layer.device = device
    sum_layer.perturbed = True
    sum_layer.input_shape = gamma_layer.output_shape
    sum_layer.output_shape = torch.Size([1, 1])
    sum_layer.batch_dim = bound_node.batch_dim
    
    # 4) Repeat
    # One copy per neuron in the layer that should be optimized.
    repeat_layer_weight_value = nn.Parameter(torch.ones((1, neurons_in_layer), device=device))
    repeat_layer_weight = BoundParams(
        ori_name="/repeat_layer_weight",
        value=repeat_layer_weight_value,
        perturbation=None,
    )
    repeat_layer_weight.name = "repeat_layer_weight"
    repeat_layer_weight.lower = repeat_layer_weight_value
    repeat_layer_weight.upper = repeat_layer_weight_value
    repeat_layer = BoundLinear(
        attr=None,
        inputs=[sum_layer, repeat_layer_weight],
        output_index=0,
        options=self.bound_opts,
    )
    repeat_layer.name = "/repeat_layer"
    repeat_layer.device = device
    repeat_layer.perturbed = True
    repeat_layer.input_shape = sum_layer.output_shape
    repeat_layer.output_shape = torch.Size([1, neurons_in_layer])
    repeat_layer.batch_dim = bound_node.batch_dim
    
    # 5) Reshape
    # To the same shape as the layer that's optimized.
    reshape_layer_output_shape = BoundBuffers(
        ori_name="/reshape_layer_output_shape",
        value = torch.tensor(bound_node.output_shape[1:]),
        perturbation=None,
        options=self.bound_opts,
    )
    reshape_layer_output_shape.name = "reshape_layer_output_shape"
    reshape_layer = BoundReshape(
        attr=None,
        inputs = [repeat_layer, reshape_layer_output_shape],
        output_index=0,
        options=self.bound_opts,
    )
    reshape_layer.name = "/reshape_layer"
    reshape_layer.device = device
    reshape_layer.perturbed = True
    reshape_layer.input_shape = repeat_layer.output_shape
    reshape_layer.output_shape = bound_node.output_shape
    reshape_layer.batch_dim = bound_node.batch_dim

    # The residual connection that connects the optimized layer and the reshape
    # layer from above is not explicitly coded, it's handled implicitly:
    # Here, we propagate backwards through 5->4->3->2->1->regular output layer and let
    # CROWN handle the propagation from there on backwards to the input layer.
    # The other half of the residual connection is implemented by explicitly setting
    # the .lA and .uA values of the optimized layer to C.
    # This is done via initial_As, initial_lb, initial_ub.
    
    if True or not isinstance(bound_node, BoundLinear):
        if isinstance(C, OneHotC):
            batch_size = C.shape[1]
            assert C.shape[0] <= C.shape[2]
            assert len(C.shape) == 3
            # This is expensive, but Reshape doesn't support OneHotC objects
            C = torch.eye(C.shape[2], device=C.device)[C.index].unsqueeze(1).repeat(1, batch_size, 1)

    start_shape = None
    lA = C if bound_lower else None
    uA = C if bound_upper else None

    # 5) Reshape
    A, lower_b, upper_b = reshape_layer.bound_backward(
        lA, uA, *reshape_layer.inputs,
        start_node=bound_node, unstable_idx=unstable_idx,
        start_shape=start_shape)
    assert lower_b == 0
    assert upper_b == 0
    lA = A[0][0]
    uA = A[0][1]

    # 4) Repeat
    A, lower_b, upper_b = repeat_layer.bound_backward(
        lA, uA, *repeat_layer.inputs,
        start_node=bound_node, unstable_idx=unstable_idx,
        start_shape=start_shape)
    assert lower_b == 0
    assert upper_b == 0
    lA = A[0][0]
    uA = A[0][1]

    # 3) Sum
    A, lower_b, upper_b = sum_layer.bound_backward(
        lA, uA, *sum_layer.inputs,
        start_node=bound_node, unstable_idx=unstable_idx,
        start_shape=start_shape)
    assert lower_b == 0
    assert upper_b == 0
    lA = A[0][0]
    uA = A[0][1]

    # 2) Gamma
    A, lower_b, upper_b = gamma_layer.bound_backward(
        lA, uA, *gamma_layer.inputs,
        start_node=bound_node, unstable_idx=unstable_idx,
        start_shape=start_shape)
    assert lower_b == 0
    assert upper_b == 0
    lA = A[0][0]
    uA = A[0][1]

    # 1) Hx + d
    A, lower_b, upper_b = linear_Hxd_layer.bound_backward(
        lA, uA, *linear_Hxd_layer.inputs,
        start_node=bound_node, unstable_idx=unstable_idx,
        start_shape=start_shape)
    # lower_b and upper_b are no longer 0, because d wasn't 0.
    lA = A[0][0]
    uA = A[0][1]

    # This encodes the residual connection.
    initial_As = {
        self.final_node().name: (lA, uA),
        bound_node.name: (C, C),
    }

    return self.backward_general(
        bound_node = bound_node,
        start_backpropagation_at_node = self.final_node(),
        C = orig_C,  #  only used for batch_size, output_dim, output_shape computation
        bound_lower = bound_lower,
        bound_upper = bound_upper,
        average_A = average_A,
        need_A_only = need_A_only,
        unstable_idx = unstable_idx,
        update_mask = update_mask,
        verbose = verbose,
        apply_output_constraints_to = [],  # no nested application
        initial_As = initial_As,
        initial_lb = lower_b,
        initial_ub = upper_b,
    )

#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################

from .utils import *
from .bound_ops import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def invprop_enabled(self: 'BoundedModule'):
    return self.bound_opts['optimize_bound_args']['apply_output_constraints_to']


def invprop_init_infeasible_bounds(self: 'BoundedModule', bound_node, C):
    # Infeasible bounds can result from unsatisfiable output constraints.
    # We track them to set the corresponding lower bounds to inf and upper bounds to
    # -inf.
    if self.infeasible_bounds is None:
        device = bound_node.attr['device']
        if isinstance(C, Patches):
            self.infeasible_bounds = torch.full((C.shape[1],), False, device=device)
        else:
            assert isinstance(C, (torch.Tensor, eyeC, OneHotC)), type(C)
            self.infeasible_bounds = torch.full((C.shape[0],), False, device=device)


def invprop_check_infeasible_bounds(self: 'BoundedModule', lb, ub):
    if torch.any(self.infeasible_bounds):
        if lb is not None:
            assert lb.size(0) == self.infeasible_bounds.size(0)
            lb = torch.where(self.infeasible_bounds.unsqueeze(1),
                             torch.tensor(float('inf'), device=lb.device), lb)
        if ub is not None:
            assert ub.size(0) == self.infeasible_bounds.size(0)
            ub = torch.where(self.infeasible_bounds.unsqueeze(1),
                             torch.tensor(float('-inf'), device=ub.device), ub)
    return lb, ub


def backward_general_invprop(
    self: 'BoundedModule',
    initial_As, initial_lb, initial_ub,
    bound_node,
    C,
    start_backpropagation_at_node = None,
    bound_lower=True,
    bound_upper=True,
    average_A=False,
    need_A_only=False,
    unstable_idx=None,
    update_mask=None,
    verbose=True,
):
    use_beta_crown = self.bound_opts['optimize_bound_args']['enable_beta_crown']
    # Sometimes, not using output constraints can give better results.
    # When this flag is set, the bounds are computed both with and without
    # output constraints, and the best of the two is returned.
    best_of_oc_and_no_oc = (
        self.bound_opts['optimize_bound_args']['best_of_oc_and_no_oc']
    )

    assert not use_beta_crown
    assert not self.cut_used
    assert initial_As is None
    assert initial_lb is None
    assert initial_ub is None
    if best_of_oc_and_no_oc:
        # Important: If input bounds are tightened, then this call must be done
        # *before* the use of output constraints.
        # At the end of backward_general, the bounds are concretized. For the input
        # bounds, those concrete bounds are used to overwrite the bounds in the
        # input perturbations, so they'll then be used by all other layers during
        # their concretization. These input bounds *must* have their gradients
        # w.r.t. the relaxations set up. The call to backward_general without
        # output constraints will overwrite these bounds with values that do not
        # have gradients. So it must come first.
        with torch.no_grad():
            o_res = self.backward_general(
                bound_node=bound_node,
                C=C,
                start_backpropagation_at_node=start_backpropagation_at_node,
                bound_lower=bound_lower,
                bound_upper=bound_upper,
                average_A=average_A,
                need_A_only=need_A_only,
                unstable_idx=unstable_idx,
                update_mask=update_mask,
                verbose=verbose,
                apply_output_constraints_to=[],
            )
    res = self.backward_general_with_output_constraint(
        bound_node=bound_node,
        C=C,
        start_backporpagation_at_node=start_backpropagation_at_node,
        bound_lower=bound_lower,
        bound_upper=bound_upper,
        average_A=average_A,
        need_A_only=need_A_only,
        unstable_idx=unstable_idx,
        update_mask=update_mask,
        verbose=verbose,
    )
    if best_of_oc_and_no_oc:
        # We use the best of both results. This would convert Infs to NaNs
        # (because inf - inf = nan), so those entries get masked.
        res0_inf_mask = torch.isinf(res[0])
        r0 = res[0] - res[0].detach() + torch.max(res[0].detach(), o_res[0].detach())
        r0 = torch.where(res0_inf_mask, res[0], r0)
        res1_inf_mask = torch.isinf(res[1])
        r1 = res[1] - res[1].detach() + torch.min(res[1].detach(), o_res[1].detach())
        r1 = torch.where(res1_inf_mask, res[1], r1)
        if self.return_A:
            if res[2] != {}:
                raise NotImplementedError(
                    "Merging of A not implemented yet. If set, try disabling --best_of_oc_and_no_oc"
                )
            res = (r0, r1, {})
        else:
            res = (r0, r1)
    batch_size = res[0].size(0)
    infeasible_bounds = torch.any(res[0].reshape((batch_size, -1)) > res[1].reshape((batch_size, -1)), dim=1)
    if torch.any(infeasible_bounds):
        self.infeasible_bounds = torch.logical_or(self.infeasible_bounds, infeasible_bounds)
    return res


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
    if device is None and hasattr(C, 'device'):
        device = C.device
    # self.constraints.shape == (batch_size, num_constraints, output_neurons)
    batch_size = self.constraints.size(0)
    num_constraints = self.constraints.size(1)

    # 1) Linear: Hx + d
    # Result is a tensor, <= 0 for all entries if output constraint is satisfied
    H = self.constraints.transpose(1,2)  # (batch_size, output_neurons, num_constraints)
    d = -self.thresholds  # (batch)
    assert H.ndim == 3
    assert H.size(0) == batch_size
    assert H.size(2) == num_constraints
    assert d.ndim == 1
    if batch_size > 1:
        assert num_constraints == 1
        assert d.size(0) == batch_size
    else:
        assert d.size(0) == num_constraints

    if hasattr(bound_node, 'gammas'):
        gammas = bound_node.gammas
    else:
        if hasattr(bound_node, 'opt_stage'):
            assert bound_node.opt_stage not in ['opt', 'reuse']
        if batch_size == 1:
            gammas = torch.zeros((2, num_constraints, neurons_in_layer), device=device)
        else:
            gammas = torch.zeros((2, batch_size, neurons_in_layer), device=device)

    # H.shape = (batch_size, output_neurons, num_constraints==1)
    # We need used_weight.shape = (batch_size, this_layer_neurons, prev_layer_neurons)
    # This is satisfied by H, because it will be transposed before being accessed and
    # output_neurons == prev_layer_neurons
    linear_Hxd_layer_weight_value = nn.Parameter(H.to(gammas))
    linear_Hxd_layer_weight = BoundParams(
        ori_name="/linear_Hxd_layer_weight",
        value=None,
        perturbation=None,
    )
    linear_Hxd_layer_weight.name = "linear_Hxd_layer_weight"
    linear_Hxd_layer_weight.lower = linear_Hxd_layer_weight_value
    linear_Hxd_layer_weight.upper = linear_Hxd_layer_weight_value

    if batch_size == 1:
        linear_Hxd_layer_bias_value = nn.Parameter(d.float().to(device))
    else:
        linear_Hxd_layer_bias_value = nn.Parameter(d.float().to(device).unsqueeze(1))
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
    linear_Hxd_layer.batched_weight_and_bias = (batch_size > 1)

    # 2) Gamma
    # A seperate gamma per output constraint. All gammas are always positive.
    # Depending on the configuration, gammas are shared across neurons in the
    # optimized layer.
    gamma_layer_weight = BoundParams(
        ori_name="/gamma_layer_weight",
        value=None,
        perturbation=None,
    )
    gamma_layer_weight.name = "gamma_layer_weight"
    assert gammas.ndim == 3
    assert gammas.size(0) == 2
    if batch_size == 1:
        # gammas.shape = (2, num_constraints, this_layer_neurons)
        assert gammas.ndim == 3
        assert gammas.size(0) == 2
        assert gammas.size(1) == num_constraints
        this_layer_neurons = gammas.size(2)

        # In linear.py, these weights will be used to compute next_A based on last_A:
        # last_A.shape = (unstable_neurons, batch_size==1, this_layer_neurons)
        # next_A.shape = (unstable_neurons, batch_size==1, prev_layer_neurons)
        # prev_layer_neurons == num_constraints
        # So we set the weights as
        # (num_constraints, this_layer_neurons)
        # This will be transposed and accessed by linear.py as
        # (this_layer_neurons, num_constraints)
        # Note that the shape will be further modified in linear.py
        gamma_layer_weight.lower = gammas[0].unsqueeze(0)
        gamma_layer_weight.upper = -gammas[1].unsqueeze(0)
    else:
        # ABCrown optimized the computation by transposing the query.
        # Instead of one batch entry with N constraints, we have N batch entries
        # with one contraint each. We do not support multiple batch entries
        # each with multiple constraints.
        # gammas.shape = (2, batch_size, this_layer_neurons)
        # Here, we can only check that the batch size is correct.
        assert gammas.size(1) == batch_size
        assert num_constraints == 1

        this_layer_neurons = gammas.size(2)

        # In linear.py, these weights will be used to compute next_A based on last_A:
        # last_A.shape = (unstable_neurons, batch_size, this_layer_neurons)
        # next_A.shape = (unstable_neurons, batch_size, prev_layer_neurons==1)
        # prev_layer_neurons == 1 because it's num_constraints
        # So we set the weights as
        # (batch_size, 1, this_layer_neurons)
        # This will be transposed and accessed by linear.py as
        # (batch_size, this_layer_neurons, 1)
        # Note that the shape will be further modified in linear.py
        gamma_layer_weight.lower = gammas[0].unsqueeze(1)
        gamma_layer_weight.upper = -gammas[1].unsqueeze(1)
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
    gamma_layer.output_shape = torch.Size([1, this_layer_neurons])
    gamma_layer.batch_dim = bound_node.batch_dim
    gamma_layer.use_seperate_weights_for_lower_and_upper_bounds = True
    gamma_layer.batched_weight_and_bias = (batch_size > 1)

    # 3) Reshape
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
        inputs = [gamma_layer, reshape_layer_output_shape],
        output_index=0,
        options=self.bound_opts,
    )
    reshape_layer.name = "/reshape_layer"
    reshape_layer.device = device
    reshape_layer.perturbed = True
    reshape_layer.input_shape = gamma_layer.output_shape
    reshape_layer.output_shape = bound_node.output_shape
    reshape_layer.batch_dim = bound_node.batch_dim

    # The residual connection that connects the optimized layer and the reshape
    # layer from above is not explicitly coded, it's handled implicitly:
    # Here, we propagate backwards through 5->4->3->2->1->regular output layer and let
    # CROWN handle the propagation from there on backwards to the input layer.
    # The other half of the residual connection is implemented by explicitly setting
    # the .lA and .uA values of the optimized layer to C.
    # This is done via initial_As, initial_lb, initial_ub.

    if isinstance(C, (OneHotC, eyeC)):
        batch_size = C.shape[1]
        assert C.shape[0] <= C.shape[2]
        assert len(C.shape) == 3
        # This is expensive, but Reshape doesn't support OneHotC objects
        if isinstance(C, OneHotC):
            C = torch.eye(C.shape[2], device=C.device)[C.index].unsqueeze(1).expand(-1, batch_size, -1)
        else:
            C = torch.eye(C.shape[2], device=C.device).unsqueeze(1).expand(-1, batch_size, -1)

    start_shape = None
    lA = C if bound_lower else None
    uA = C if bound_upper else None

    # 3) Reshape
    A, lower_b, upper_b = reshape_layer.bound_backward(
        lA, uA, *reshape_layer.inputs,
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

    assert lower_b.ndim == 2
    assert upper_b.ndim == 2

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

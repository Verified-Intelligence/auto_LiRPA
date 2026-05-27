#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import torch

from .utils import eyeC
from .bound_ops import *
from .patches import Patches
from .perturbations import PerturbationLpNorm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def concretize_bounds(
    self: 'BoundedModule',
    node,
    lower,
    upper,
    concretize_mode='backward',
    # for `backward_concretize`
    batch_size=None,
    output_dim=None,
    average_A=None,
    # for `forward_concretize`
    lw=None,
    uw=None,
    # common
    clip_neuron_selection_value=-1.0,
    clip_neuron_selection_type="ratio"
):
    """
    If neuron_selection_value >= 0, run an unconstrained/bounds-saving pass
    then a top-K constrained pass; otherwise just one pass.
    """
    # decide which underlying call to use
    def _call_concretize(use_constraints, save_bounds=False, heuristic_indices=None):
        if concretize_mode == 'backward':
            # backward concretize signature
            return backward_concretize(
                self, batch_size, output_dim, lower, upper,
                average_A=average_A,
                node_start=node,
                use_constraints=use_constraints,
                save_bounds=save_bounds,
                heuristic_indices=heuristic_indices,
            )
        elif concretize_mode == 'forward':
            # forward_concretize signature
            return forward_concretize(
                self, lower, upper, lw, uw,
                use_constraints=use_constraints,
                save_bounds=save_bounds,
                heuristic_indices=heuristic_indices,
            )
        else:
            raise ValueError(f"Unknown concretize mode: {concretize_mode}. "
                             "Please use 'backward' or 'forward'.")

    use_constraints = True
    save_bounds = False

    # If clip_neuron_selection_value >= 0, heuristic score-based topk selection is enabled.
    # And we will only apply constrained concretization on topk neurons based on their heuristics.
    # In this case, we'll need to 1) concretize all neurons without any constraints to get a looser bound 
    #                                           --> This is for computing the heuristics
    #                             2) concretize topk neurons with constraints.            
    #                                           --> This is for getting tighter bounds for topk neurons.                
    # In conclusion, if neuron_selection_value >= 0, use_consrtaints will be disabled first.
    # But for the output node in the computational graph we will directly concretize all neurons..
    if clip_neuron_selection_value >= 0 and node.name not in self.output_name:
        use_constraints = False
        # `output_activations` is the list of output activations from current pre-activation node.
        # This output_activations is manually assigned outside of auto_lirpa. Please check 
        #       complete_verifier/input_split/batch_branch_and_bound.py for more info.
        # If a node: 
        #       a) does not have any output_activation, and 
        #       b) heuristic topk selection is enabled, and
        #       c) is not the output node in the computational graph
        #  we will only compute naive bounds on it.
        # Otherwise, we'll need to do both step 1) and 2). And to accelarate step 2), we will save the bounds in 1).
        
        # If 1) this node has at least one output activation node
        #    2) at least one neuron will be selected
        # We will need to concretize with constraints, 
        if node.output_activations is not None and clip_neuron_selection_value > 0:
            save_bounds = True

    # If heuristic topk selection is enabled, this would be the step 1).
    new_lower, new_upper, has_constraints = _call_concretize(
        use_constraints=use_constraints,
        save_bounds=save_bounds,
    )

    # If heuristic topk selection is enabled, this if-branch would be the step 2).
    if (has_constraints
        and node.output_activations is not None
        and clip_neuron_selection_value > 0
        and node.name not in self.output_name):

        score = 0.0
        unstable_masks = False

        # loop through all the output activations to get a comprehensive unstable mask and heuristic score.
        # This output_activations is manually assigned outside of auto_lirpa.
        # Please check complete_verifier/input_split/batch_branch_and_bound.py
        for o_act_node in node.output_activations:
            score = score + o_act_node.compute_bound_improvement_heuristics(new_lower, new_upper)
            unstable_masks = unstable_masks | o_act_node.get_unstable_mask(new_lower, new_upper)
        score = score.flatten(1)                        # shape: (Batchsize, Hidden_dim)
        unstable_masks = unstable_masks.flatten(1)      # shape: (Batchsize, Hidden_dim)

        # Only do second concretize if there exists unstable neurons.
        if unstable_masks.any():
            max_unstable_size = unstable_masks.sum(dim=1).max()
            heuristic_indices = None
            # The K value in topk should be at least 1.
            if clip_neuron_selection_type == "ratio":
                K = max(int(max_unstable_size * clip_neuron_selection_value + 0.5), 1)
            else:
                K = min(clip_neuron_selection_value, max_unstable_size)
            _, heuristic_indices = torch.topk(score, k=K, dim=1, largest=True, sorted=False)
            new_lower, new_upper, _ = _call_concretize(
                use_constraints=True,
                heuristic_indices=heuristic_indices
            )
        else:
            # Previously we've stored to aux bounds, now it should be cleared to avoid any confusion.
            for root in self.roots():
                if (hasattr(root, 'perturbation')
                    and root.perturbation is not None
                    and isinstance(root.perturbation, PerturbationLpNorm)):
                    root.perturbation.clear_aux_bounds()

    return new_lower, new_upper


def concretize_root(self, root, batch_size, output_dim,
                    average_A=False, node_start=None, input_shape=None,
                    use_constraints=False, heuristic_indices=None, save_bounds=False): 
    # The last three optional argument are designed for heuristic-driven constrained concretization.
    # use_constraints:      A flag controling whether to enable constraints solving or not.
    # heuristic_indices:    A index tensor, it select EUQAL number of hidden neurons from each batch. 
    #                           Constrained solving will be further applied on these neurons. Shape (batchsize, n_h_neurons)
    # save_bounds:          A flag determining whether to save naive bounds (to avoid redundant computation)

    if average_A and isinstance(root, BoundParams):
        lA = root.lA.mean(
            node_start.batch_dim + 1, keepdim=True
        ).expand(root.lA.shape) if (root.lA is not None) else None
        uA = root.uA.mean(
            node_start.batch_dim + 1, keepdim=True
        ).expand(root.uA.shape) if (root.uA is not None) else None
    else:
        lA, uA = root.lA, root.uA
    if not isinstance(root.lA, eyeC) and not isinstance(root.lA, Patches):
        lA = root.lA.reshape(output_dim, batch_size, -1).transpose(0, 1) if (lA is not None) else None
    if not isinstance(root.uA, eyeC) and not isinstance(root.uA, Patches):
        uA = root.uA.reshape(output_dim, batch_size, -1).transpose(0, 1) if (uA is not None) else None
    
    has_constraints = False
    if hasattr(root, 'perturbation') and root.perturbation is not None:

        if isinstance(root.perturbation, PerturbationLpNorm):
            # Enable / Disable constraints solving according to `use_constraints`
            root.perturbation.constraints_enable = use_constraints
            if root.perturbation.constraints is not None:
                if self.infeasible_bounds_constraints is not None:
                    root.perturbation.add_infeasible_batches(self.infeasible_bounds_constraints)
                root.perturbation.add_objective_indices(heuristic_indices)
                has_constraints = True

        if isinstance(root, BoundParams):
            # add batch_size dim for weights node
            lb = root.perturbation.concretize(
                root.center.unsqueeze(0), lA, sign=-1, aux=root.aux
            ) if (lA is not None) else None
            ub = root.perturbation.concretize(
                root.center.unsqueeze(0), uA, sign=+1, aux=root.aux
            ) if (uA is not None) else None

        else:
            lb = root.perturbation.concretize(
                root.center, lA, sign=-1, aux=root.aux
            ) if lA is not None else None
            ub = root.perturbation.concretize(
                root.center, uA, sign=+1, aux=root.aux
            ) if uA is not None else None

        if (isinstance(root.perturbation, PerturbationLpNorm) 
            and root.perturbation.constraints is not None
            and root.perturbation.sorted_out_batches["infeasible_batches"] is not None):
            if self.infeasible_bounds_constraints is not None:
                self.infeasible_bounds_constraints = self.infeasible_bounds_constraints | root.perturbation.sorted_out_batches["infeasible_batches"]
            # else:
            #     self.infeasible_bounds_constraints = root.perturbation.sorted_out_batches["infeasible_batches"]

        # If required, save current (naive) bounds to prevent redundant computation next time concretize on the same node
        if isinstance(root.perturbation, PerturbationLpNorm) and root.perturbation.constraints is not None and save_bounds:
            root.perturbation.add_aux_bounds(lb, ub)
        elif isinstance(root.perturbation, PerturbationLpNorm):
        # Otherwise, always clear_aux_bounds to prevent confusion
            root.perturbation.clear_aux_bounds()

    else:
        fv = root.forward_value
        if type(root) == BoundInput:
            # Input node with a batch dimension
            batch_size_ = batch_size
        else:
            # Parameter node without a batch dimension
            batch_size_ = 1

        def concretize_constant(A):
            if isinstance(A, eyeC):
                return fv.view(batch_size_, -1)
            elif isinstance(A, Patches):
                return A.matmul(fv, input_shape=input_shape)
            elif type(root) == BoundInput:
                return A.matmul(fv.view(batch_size_, -1, 1)).squeeze(-1)
            else:
                return A.matmul(fv.view(-1, 1)).squeeze(-1)

        lb = concretize_constant(lA) if (lA is not None) else None
        ub = concretize_constant(uA) if (uA is not None) else None

    return lb, ub, has_constraints


def backward_concretize(self, batch_size, output_dim, lb=None, ub=None,
               average_A=False, node_start=None, 
               use_constraints=False, heuristic_indices=None, save_bounds=False):
    # The last three optional argument are designed for heuristic-driven constrained concretization.
    # use_constraints:      A flag controling whether to enable constraints solving or not.
    # heuristic_indices:    A index tensor, it select EUQAL number of hidden neurons from each batch. 
    #                           Constrained solving will be further applied on these neurons. Shape (batchsize, n_h_neurons)
    # save_bounds:          A flag determining whether to save naive bounds (to avoid redundant computation)
    roots = self.roots()
    if isinstance(lb, torch.Tensor) and lb.ndim > 2:
        lb = lb.reshape(lb.shape[0], -1)
    if isinstance(ub, torch.Tensor) and ub.ndim > 2:
        ub = ub.reshape(ub.shape[0], -1)

    def add_b(b1, b2):
        if b2 is None:
            return b1
        elif b1 is None:
            return b2
        # Check if b1 is a tensor and if all its elements are infinity
        if torch.is_tensor(b1) and torch.isinf(b1).all():
            return b1
        # Check if b2 is a tensor and if all its elements are infinity
        if torch.is_tensor(b2) and torch.isinf(b2).all():
            return b2
        else:
            return b1 + b2

    has_constraints = False
    for root in roots:
        root.lb = root.ub = None
        if root.lA is None and root.uA is None:
            continue
        root.lb, root.ub, has_constraints_this_root = self.concretize_root(
            root, batch_size, output_dim, average_A=average_A,
            node_start=node_start, input_shape=roots[0].center.shape,
            use_constraints=use_constraints, heuristic_indices=heuristic_indices, save_bounds=save_bounds)

        has_constraints = has_constraints | has_constraints_this_root

        lb = add_b(lb, root.lb)
        ub = add_b(ub, root.ub)

    return lb, ub, has_constraints


def forward_concretize(self, lower, upper, lw, uw, use_constraints=False, heuristic_indices=None, save_bounds=False):
    """
    Concretize function for forward bound. 

    :param lower:                   Tensor. Intermediate layer lower bounds.
    :param upper:                   Tensor. Intermediate layer upper bounds.
    :param lw:                      Tensor. Intermediate layer lower A matrix.
    :param uw:                      Tensor. Intermediate layer upper A matrix.
    :param use_constraints:         bool. A flag controling whether to enable constraints solving or not.
        If heuristic ratio is set, the first concretization run should disbale constraints solving.
    :param heuristic_indices:       Index Tensor. A index tensor, it select **equal** number of hidden neurons from each batch.
        Constrained solving will be further applied on these neurons. Shape (batchsize, n_h_neurons)
    :param save_bounds:             bool. A flag controling whether to save naive bounds.
    
    :return res_lower:              Tensor. The lower bound tensor.
    :return res_upper:              Tensor. The upper bound tensor.
    :return has_constraints:        bool. Whether constraints has been stored.
    """
    res_lower = 0.0
    res_upper = 0.0
    prev_dim_in = 0
    has_constraints = False
    roots = self.roots()
    assert (lw.ndim > 1)
    lA = lw.reshape(self.batch_size, self.dim_in, -1).transpose(1, 2)
    uA = uw.reshape(self.batch_size, self.dim_in, -1).transpose(1, 2)
    for root in roots:
        if hasattr(root, 'perturbation') and root.perturbation is not None:
            _lA = lA[:, :, prev_dim_in : (prev_dim_in + root.dim)]
            _uA = uA[:, :, prev_dim_in : (prev_dim_in + root.dim)]

            if isinstance(root.perturbation, PerturbationLpNorm):
                root.perturbation.constraints_enable = use_constraints
                if root.perturbation.constraints is not None:
                    if self.infeasible_bounds_constraints is not None:
                        root.perturbation.add_infeasible_batches(self.infeasible_bounds_constraints)
                    root.perturbation.add_objective_indices(heuristic_indices)
                    has_constraints = True                 

            # Previously added concretized bounds directly to lower/upper.
            # Now extract them first for reuse (e.g., in aux_bounds).
            temp_lower = root.perturbation.concretize(
                root.center, _lA, sign=-1, aux=root.aux
                ).view(lower.shape)
            temp_upper = root.perturbation.concretize(
                root.center, _uA, sign=+1, aux=root.aux
                ).view(upper.shape)
            
            # Update infeasible_batches
            if (isinstance(root.perturbation, PerturbationLpNorm)
                and root.perturbation.constraints is not None 
                and root.perturbation.sorted_out_batches["infeasible_batches"] is not None):
                if self.infeasible_bounds_constraints is not None:
                    self.infeasible_bounds_constraints = self.infeasible_bounds_constraints | root.perturbation.sorted_out_batches["infeasible_batches"]
                # else:
                #     self.infeasible_bounds_constraints = root.perturbation.sorted_out_batches["infeasible_batches"]

            # If required, save current (naive) bounds to prevent redundant computation next time concretize on the same node
            if isinstance(root.perturbation, PerturbationLpNorm) and root.perturbation.constraints is not None and save_bounds:
                root.perturbation.add_aux_bounds(temp_lower, temp_upper)
            elif isinstance(root.perturbation, PerturbationLpNorm):
            # Otherwise, always clear_aux_bounds to prevent confusion
                root.perturbation.clear_aux_bounds()

            # Now the concretization result from this root will be accumulated into final bounds.
            # Here we add temp_lower onto res_lower, instead of lower. 
            # It's because the lower value will be used twice, any modification to it should be avoided.
            res_lower = res_lower + temp_lower
            res_upper = res_upper + temp_upper                        
    
    res_lower = res_lower + lower
    res_upper = res_upper + upper
    return res_lower, res_upper, has_constraints

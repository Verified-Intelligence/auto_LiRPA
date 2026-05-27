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

from math import floor, ceil
from .utils import eyeC

# Declaration of the shape naming:

# B / batchsize  : The number of batches. In this `concretize_func.py`, if a tensor has batch dimension, we assume
#                    it will only be the first dimention of this tensor . That is: B = tensor.shape[0]
# 
# B_act          : The number of active batches. We will only apply constraints to a subset of batches, and these
#                   batches are called active batches. B_act <= B. There are two cases:
#                       -- When `no_return_inf` mode is disabled, we will keep B_act static throughout the entire 
#                           BaB iteration. In this case, B_act equals the number of batches not fully covered by 
#                           constraints, as determined by `sort_out_constr_batches` function.
#                       -- When `no_return_inf` mode is enabled, then B_act decreases over iterations, since more
#                           batches will be marked as infeasible. See `PerturbationLpNorm.add_infeasible_batches`.
# 
# X / x_dim      : The number of input neurons (batch dimension excluded). It stands for the input shape of the
#                   neural network. For tensors such as x0, epsilon, x_U, x_L, X = prod(*tensor.shape[1:])
# 
# H / hidden_dim : The number of hidden neurons (batch and input dimension excluded). It stands for the output
#                   shape of this hidden layer. For the objective A tensor, there are two cases:
#                       -- The tensor has batch dimention: H = tensor.view(B, -1, X).shape[1]
#                       -- The tensor does not have batch dimention: H = tensor.view(-1, X).shape[0]
# 
# H_act          : The number of active batches. We may only apply constraints to a subset of hidden neurons,
#                   and these neurons are called active neurons. H_act <= H.
#
# N_constr       : The number of constraints. For constraints_A matrix:
#                       -- In `sort_out_constr_batches` function, its shape is (B, N_constr, X)
#                       -- In `constraints_solving` function, its shape is (B_act, N_constt, X)

def construct_constraints(constr_A: torch.Tensor, constr_b: torch.Tensor, constr_rhs: torch.Tensor,
                            batchsize, x_dim, sign=1):
    r"""
    Construct the constraints tuple. This function provides a unified interface to generate this tuple.
    All the users should carefully read this function to fully understand the standard form of constraints.

    The first three argument expresses the non-standard form of the constraints:
                                    A @ x + b <= rhs
    We will first convert it into the standard form:
                                    A @ x + b' <= 0
    The the standard expression of constraints should be (constr_A, constr_b')

    Args:
        constr_A:   The coefficient A matrix of constraints.
                        It should be able to be reshaped into: (B, N_constr, X)
        constr_b:   The bias term of constraints.        
                        It should be able to be reshaped into: (B, N_constr)
        constr_rhs: The right-hand-side term of constraints.        
                        It should be able to be reshaped into: (B, N_constr)
        batchsize:  The batchsize B.
        x_dim:      The input dimension X (batchsize dimension excluded)
    """
    constr_A = sign * constr_A.reshape((batchsize, -1, x_dim))
    if constr_rhs is not None and not torch.all(constr_rhs == 0):
        constr_b = sign * (constr_b - constr_rhs).reshape((batchsize, -1))
    else:
        constr_b = sign * constr_b.reshape((batchsize, -1))
    return (constr_A, constr_b)

def _sort_out_constraints(A, b, x0, epsilon):
    r"""
    Filter out some batches with constraints not intersecting with input region

    Args:
        A (Tensor): A matrix of constraints with shape of (batchsize, n_constraints, x_dim)
        b (Tensor): Bias term of constraints with shape of (batchsize, n_constraints)
        x0 (Tensor): Centroid of the input space with shape of (batchsize, x_dim, 1)
        epsilon (Tensor): Offset from the centroid to the input space boundary with shape of (batchsize, x_dim, 1)
    Return:
        no_intersection (Tensor): A boolean tensor with shape (batchsize, ), indicating if certain batch is infeasible
            because a constraint does not intersect with input space
        fully_covered (Tensor): A boolean tensor with shape (batchsize, ), indicating if all the constraints in a certain 
            batch fully covers the corresponding input region. In this case, we can simply treat the batch as if it has no constraints
    """
    # minimal and maximal value of A*x + b
    x0_term = A.bmm(x0).squeeze(-1) + b        # shape: (B, N_constr)
    eps_term = A.abs().bmm(epsilon).squeeze(-1) # shape: (B, N_constr)
    minimal_val = x0_term - eps_term            # shape: (B, N_constr)
    maximal_val = x0_term + eps_term            # shape: (B, N_constr)

    # for any constrains: A * x + b <= 0,
    # if its min(A * x + b) > 0, it has no intersection with x0 +- epsilon
    # if its max(A * x + b) <= 0, it fully covers x0 +- epsilon 
    no_intersection = (minimal_val > 0).any(1)  # shape: (B, )
    if not no_intersection.any():
        no_intersection = None
    fully_covered = (maximal_val <= 0).all(1)   # shape: (B, )
    return no_intersection, fully_covered

@torch.jit.script
def _dist_rearrange(constraints_A, constraints_b, x0):
    r"""
    Reorder the constraints according to their distance to x_prime

    Args:
        constraints_A (Tensor): A matrix of constraints with shape of (batchsize, n_constraints, x_dim)
        constraints_b (Tensor): Bias term of constraints with shape of (batchsize, n_constraints)
        x0 (Tensor): x0 tensor with shape of (batchsize, x_dim, 1). Based on the heuristic,
        this can be the input space centroid x0, or the original optimal point x_prime
    Return:
        rearranged_A (Tensor): Rearranged matrix of constraints with shape of (batchsize, n_constraints, x_dim)
        rearranged_b (Tensor): Bias term of constraints with shape of (batchsize, n_constraints)
    """
    # Compute the normalized, directional distance from x_prime to constraints hyper-plane.
    distance = (constraints_A.bmm(x0).squeeze(-1) + constraints_b) # shape: (B, N_constr)
    l2_norm  = constraints_A.norm(p=2, dim=-1)                     # shape: (B, N_constr)
    normed_dist = distance / l2_norm                               # shape: (B, N_constr)

    # Sort the constraints according to this distance.
    order = torch.sort(normed_dist, descending=True, dim=1)[1]
    order_expand = order.unsqueeze(-1).expand(-1, -1, constraints_A.size(-1))
    rearranged_A = constraints_A.gather(index=order_expand, dim=1)
    rearranged_b = constraints_b.gather(index=order, dim=1)
    return rearranged_A, rearranged_b

@torch.jit.script
def _solve_dual_var(constr_a, object_a, constr_d, epsilon, a_mul_e=None):
    r"""
    Solve the following optimization problem:

    Primal:         min_x   object_a^T x
                    s.t.    constr_a^T x + constr_d <= 0,
                            x0-epsilon <= x <= x0+epsilon

    Dual:           min_x max_beta  object_a^T x + beta * (constr_a^T x + constr_d)
                    s.t.            x0 - epsilon <= x <= x0 + epsilon
                                    beta >= 0

    Strong duality:
                    max_{beta >= 0} min_{x \in X} object_a^T x + beta * (constr_a^T x + constr_d)

    Dual norm:
                    max_{beta >= 0} - |object_a + beta * constr_a|^T epsilon + beta * (constr_a^T x0 + constr_d) + object_a^T x0

    Now the sole optimize problem is piece-wise linear, we just have to check each 
    turning point and the end points of beta (0 and +inf)

    Args:
        constr_a (Tensor): Constraint A matrix with shape of (batchsize, x_dim)
        object_a (Tensor): Objective A matrix with shape of (batchsize, h_dim, x_dim)
        constr_d (Tensor): Pre-computed bias term of constraint with shape of (batchsize, )
                    constr_d = constr_a^T x0 + constr_b
        epsilon (Tensor): Offset from the centroid to the input space boundary with shape of (batchsize, x_dim, 1)
    Return:
        optimal_beta (Tensor): The optimal beta value with shape of (batchsize, h_dim)
    """

    B_act = constr_a.size(0)
    H_act = object_a.size(1)
    device = constr_a.device
    dtype = constr_a.dtype

    # --- prepare fill-in tensors 
    zeros = torch.zeros((1, 1, 1), device=device, dtype=dtype).expand(B_act, H_act, 1)
    infs = torch.full((1, 1, 1), fill_value=torch.inf, dtype=dtype, device=device).expand(B_act, H_act, 1)

    a_reshape = constr_a.unsqueeze(1)                   # shape: (B_act, 1, X)
    epsilon_reshape = epsilon.view((B_act, 1, -1))      # shape: (B_act, 1, X)
    b_reshape = constr_d.view((-1, 1, 1))               # shape: (B_act, 1, 1)

    # q is the turning points of the piece-wise linear function.
    q = - object_a/a_reshape                            # shape: (B_act, H_act, X)
    # idx indicates the ascending order of these turning points.
    q_sort, idx = q.sort(dim=-1)                        # shape: (B_act, H_act, X) 

    # --- calculating the gradient w.r.t. beta within each interval ---
    a_mul_e = (a_reshape * epsilon_reshape).expand(-1, H_act, -1)   # (B_act, H_act, X)
    # a_mul_e = a_mul_e.expand(-1, H_act, -1)

    #               (B_act, H_act, X)       (B_act, H_act, X)
    a_sort = torch.gather(a_mul_e, dim=-1, index=idx)               # (B_act, H_act, X)

    a_neg_cumsum = a_sort.abs().cumsum(dim=-1)              # shape: (B_act, H, x_dim)
    a_neg_cumsum = torch.cat((zeros, a_neg_cumsum), dim=-1) # shape: (B_act, H_act, 1+X)
    a_pos_cumsum = a_neg_cumsum - a_neg_cumsum[:, :, -1:]   # shape: (B_act, H_act, 1+X)
    grad_beta = a_pos_cumsum + a_neg_cumsum - b_reshape     # shape: (B_act, H_act, 1+X)

    # Due to the non-increasing trait of grad_beta, if there is a turning point
    # then the gradient must change from positive to negative, and this turning point is the optimal beta.
    sign_change = torch.searchsorted(grad_beta, zeros, right=False)

    # It might be the case that grad_beta is always positive when beta > 0. 
    # This means the maximization object is ever-increasing, hence it is unbounded.
    # For this case, a inf value would be returned.

    # Following comes a case of sign_change where all the turning points q are positive:
    # (g stands for grad_beta, q stands for turing points)
    #    g[0] = 2       g[1] = 1       g[2] = -1       g[3] = -3   
    # 0 --------- q[0] --------- q[1] ----------- q[2] ----------- ... --------> +inf
    #                             ^
    #                      sign_change=2
    #
    # q should represent the interval endpoints, hence, need to pad the left and right end with 0 and inf separately.

    # cat shape: (B_act, H_act, 1+X+1)                   
    q_new = torch.cat((zeros, q_sort, infs), dim=-1)                                       # shape: (B_act, H_act, X+2)
    optimal_beta = torch.gather(q_new, dim=-1, index=sign_change).clamp(min=0).squeeze(-1) # shape: (B_act, H_act)

    return optimal_beta

def sort_out_constr_batches(x_L, x_U, constraints, rearrange_constraints=False, no_return_inf=False):
    r"""
    Filter and preprocess input batches based on constraint feasibility.

    This function examines which input regions 
        1) has no intersection with one of the constraints.
        2) is fully covered by the all the constraints.

    It also optionally rearranges constraint order for better numerical behavior,
    and converts the constraint form from `(A, b)` to `(A, d)` where `d = A @ x0 + b`.
    Here x0 means the centroid of the input region, that is x0 = (x_L + x_U) / 2.
    
    Args:
        x_L (Tensor): Lower bound of input box, shape (B, *).
        x_U (Tensor): Upper bound of input box, shape (B, *).
        constraints (Tuple[Tensor, Tensor] or None): 
            A tuple `(A, b)` representing per-batch linear constraints.
            - `A`: shape (B, N_constr, X)
            - `b`: shape (B, N_constr)
            If None or empty, the function returns early.
        rearrange_constraints (bool): 
            Whether to rearrange constraints for better solver performance. Default: False.
        no_return_inf (bool): 
            If True, infeasible batches will be excluded from `active_indices`.
            Otherwise, infeasible batches are still marked active. Default: False.

    Returns:
        constraints (Optional[Tuple[Tensor, Tensor]]): 
            Filtered and reshaped constraint tuple `(A, d)` for active batches only.
            - `A`: shape (B_active, N_constr, X)
            - `d`: shape (B_active, N_constr)
            If all batches are fully covered, returns None.

        sorted_out_batches (dict): Diagnostic and filtering info with keys:
            - 'infeasible_batches' (BoolTensor): Shape (B,), True if batch has no feasible region.
                                                 If all the elements are False, it would be None. This would save space and time.
            - 'fully_covered' (BoolTensor): Shape (B,), True if batch is completely covered by constraints.
            - 'active_indices' (LongTensor): Indices of batches that are neither fully covered nor infeasible.
    """
    sorted_out_batches = None
    if constraints is None or constraints[0] is None or constraints[0].numel() == 0:
        return None, sorted_out_batches

    # Read argument and some necessary reshape
    assert x_L is not None and x_U is not None, "If constrained concretize is enabled, x_L and x_U cannot be None!"
    x0 = (x_L + x_U) / 2
    epsilon = (x_U - x_L) / 2
    constraints_A, constraints_b = constraints
    batch_size = x0.shape[0]
    x_dim = x0[0].numel()
    x0 = x0.view((batch_size, x_dim, 1))                        # shape: (B, X, 1)
    epsilon = epsilon.view((batch_size, x_dim, 1))              # shape: (B, X, 1)

    no_intersection, fully_covered = _sort_out_constraints(constraints_A, constraints_b, x0, epsilon)
    if fully_covered.all():
        print("All the added constraints fully cover the input space. No need to apply constraints .")
        return None, sorted_out_batches
    sorted_out_batches = {}
    sorted_out_batches["infeasible_batches"] = no_intersection
    # If there's no infeasible batch, simply set it to be None. 
    # This will provide a shortcut when update the infeasible_batches vector.
    # When batchsize is large and NN model has a lot of perturbed roots, this can save us some time.
    sorted_out_batches["fully_covered"] = fully_covered
    active_mask = ~fully_covered
    if no_intersection is not None and no_return_inf:
        active_mask = ~no_intersection & active_mask
    active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
    sorted_out_batches["active_indices"] = active_indices

    # Now constraints tuple only contains active constraints, shape change: (B, N_Constr, X) -> (B_act, N_constr, X)
    constraints_A = constraints_A[active_indices]   # shape: (B_act, N_Constr, X)
    constraints_b = constraints_b[active_indices]   # shape: (B_act, N_Constr)
    active_x0 = x0[active_indices]
    if rearrange_constraints:
        constraints_A, constraints_b = _dist_rearrange(constraints_A, constraints_b, active_x0)
    # Also, we will replace the constraint_b term with constraints_d term.
    # For the usage of constraints_d, please check _solve function and constraints_solving function.
    constraints_d = torch.einsum('bkx, bxo->bk', constraints_A, active_x0) + constraints_b    # shape: (B_act, N_Constr)
    # Only store the constraints for active batches.
    constraints = (constraints_A, constraints_d)

    return constraints, sorted_out_batches

def constraints_solving(
    x_L, x_U, objective, constraints, sign=-1.0,
    sorted_out_batches={}, objective_indices=None, 
    constraints_enable=True, no_return_inf=False,
    max_chunk_size=None, safety_factor=0.8, solver_memory_factor=2.0,
    timer=None, 
    aux_bounds=None, 
    x0=None, epsilon=None, 
    act_x0=None, act_eps=None,
    use_grad=True
    ):
    r"""
    Combined constraint solving function with conditional logic based on objective shape.

    - If objective is eyeC or broadcastable (shape[0]=1), uses a vectorized,
        auto-chunked approach.
    - If objective has batch dim matching input (shape[0]=N_batch), uses the
        original approach (repeating inputs, no chunking).

    Solves LP: max / min A_t * x, s.t. A_c * x + b_c <= 0, x_L <= x <= x_U

    Args:
        x_L, x_U (Tensor)               : Input bounds tensors.
        objective (Tensor)              : Target coefficients (Tensor or eyeC).
            - Tensor shape: (H, X), (1, H, X), or (N_batch, H, X).
            - eyeC: Represents identity matrix.W
        constraints (tuple, optional)   : Tuple (A_c, d_c) or None.
        sign (float, optional)          : -1.0 for lower bound, +1.0 for upper bound.
        sorted_out_batches (dict, optional): Dict with pre-filtered batch masks. Please check `sort_out_constr_batches` for more info.
        constraints_enable (bool, optional): Flag for enabling constraints solving, this is set for heuristic hybrid solving, should be True by default.
        no_return_inf (bool, optional)  :  Flag for returning inf value. If true, this function will return inf for all the infeasible subproblems.
                        Otherwise, return naive bounds for infeasible ones.
        max_chunk_size, safety_factor, solver_memory_factor: Params for chunking memory.
                max_chunk_size:
                        A hard upper limit on the number of problems to be processed in a single
                        chunk, regardless of available memory. If set to an integer, the
                        auto-calculated chunk size will not exceed this value.
                        Use Case: Prevents the solver from creating a single, massive chunk that
                        could cause system unresponsiveness, even if memory is technically
                        available. Set to None to allow the function to use its own dynamic
                        calculation.
                safety_factor:
                        A float between 0.0 and 1.0 that specifies what fraction of the free
                        GPU memory should be considered "usable" for the calculation. For example,
                        a value of 0.8 means the function will only use 80% of the available
                        free memory as its budget.
                        Use Case: This buffer helps prevent "Out of Memory" (OOM) errors by
                        accounting for memory fragmentation, memory used by other processes, or
                        overhead from the CUDA driver itself. A lower value is safer but may
                        result in smaller chunks and thus slower overall processing.
                solver_memory_factor:
                        A heuristic multiplier used to estimate the memory consumed by the
                        iterative solver loop. The theoretical memory usage is multiplied by
                        this factor to create a more realistic estimate.
                        Use Case: The exact memory allocated for intermediate tensors and
                        computations within the solver can be complex to predict perfectly. This
                        factor provides a "fudge factor" to pad the memory estimation, ensuring
                        that the dynamically created tensors inside the solver loop do not cause
                        an OOM error. Adjust this if you consistently face memory issues during
                        the solver phase.
        objective_indices (Tensor, optional): Indices tensor of shape (N_batch, H_active) indicating
                            which objectives to compute. If None, all are computed.
        timer: Optional Timer object.
        aux_bounds (Tensor, optional)   : When hybrid constraint solving is enbaled, constrains_solving function will be called twice.
                                       For its second run, we will load the result from the first run to save time computing naive results.
        x0, eps (Tensor, optional)      : x0 and epsilon to solve on. 
                                    Without these two, we can still compute x0 and eps out of x_L and x_U.
        act_x0, act_eps (Tensor, optional): Active x0 and epsilon to solve on.
        use_grad (bool, optional): If False, the main computation is wrapped in
                                    `torch.no_grad()` for better performance and lower
                                    memory usage. Set to True only when gradients are
                                    required (e.g., for clip during alpha crown). Defaults to True.

    Returns:
        bound (Tensor): Computed bounds (N_batch, H, 1).
        infeasible_batches (boolTensor, optional) : If no_return_inf is True, `infeasible_batches` will be returned.
                                                    It is a boolean tensor with shape (batch_size, ), with True indictating the batch is infeasible. 
    """
    if timer: timer.start('init')
    if timer: timer.start("concretize")

    device = x_L.device
    N_batch = x_L.size(0)

    epsilon = (x_U - x_L) / 2.0 if epsilon is None else epsilon
    x0 = (x_U + x_L) / 2.0 if x0 is None else x0
    epsilon = epsilon.reshape((N_batch, -1, 1))
    x0 = x0.reshape((N_batch, -1, 1))

    is_eyeC = isinstance(objective, eyeC)

    # --- Naive Case (No Constraints) ---
    no_constraints_condition = (constraints is None) or (constraints[0].numel() == 0)
    if no_constraints_condition or (not constraints_enable):
        if is_eyeC:
            solved_obj = x0 + sign * epsilon                                    # Shape: (N_batch, X, 1)
        else:
            base_term = torch.einsum('bhx,bxo->bho', objective, x0)             # Shape: (N_batch, X, 1)
            eps_term = torch.einsum('bhx,bxo->bho', objective.abs(), epsilon)   # Shape: (N_batch, X, 1)
            solved_obj = base_term + sign * eps_term # Shape: (N_batch, H, 1)
        if timer: timer.add("init")
        if timer: timer.add("concretize")
        if no_return_inf:
            return solved_obj, None
        else:
            return solved_obj

    with torch.set_grad_enabled(use_grad):
        is_broadcastable = False
        is_batch_specific = False
        H = -1 # Hidden dimension
        X = x0.size(1) # Input X dimension
        if is_eyeC:
            is_broadcastable = True
            H = X
            # Internally represent eyeC as identity matrix for broadcastable path.
            objective_tensor = torch.eye(X, device=device).unsqueeze(0) # Shape (1, X, X)
        else:
            if objective.shape[0] != N_batch:
                # objective comes in shape of (H, X) or (1, H, X).
                # It will be broadcasted to (B, H, X) later.
                # Currently, is_broadcastable is designed for relu-bab, which usually takes much gpu memory,
                # so is_broadcastable is also a control flag for objective chunking.
                is_broadcastable = True
            else:
                # objective comes in shape of (B, H, X).
                is_batch_specific = True
            H = objective.shape[1]
            objective_tensor = objective
            if objective.shape[2] != X: raise ValueError("Objective shape mismatch")

        # --- Constrained Case ---
        # --- Calculate Naive Bounds (used as default/fallback) ---
        naive_bounds = torch.zeros(N_batch, H, 1, device=device)
        if aux_bounds is not None:
            naive_bounds_all = aux_bounds.flatten(1).unsqueeze(-1)
        elif is_eyeC:
            naive_bounds_all = x0 + sign * epsilon # Shape (N_batch, X, 1) -> (N_batch, H, 1)
        elif is_broadcastable:
            # obj_tensor is (1, H, X)
            base_term_naive = torch.einsum('shx,bxo->bho', objective_tensor, x0)
            eps_term_naive = torch.einsum('shx,bxo->bho', objective_tensor.abs(), epsilon)
            naive_bounds_all = base_term_naive + sign * eps_term_naive # Shape (N_batch, H, 1)
        elif is_batch_specific:
            # obj_tensor is (N, H, X)
            base_term_naive = torch.einsum('bhx,bxo->bho', objective_tensor, x0)
            eps_term_naive = torch.einsum('bhx,bxo->bho', objective_tensor.abs(), epsilon)
            naive_bounds_all = base_term_naive + sign * eps_term_naive # Shape (N_batch, H, 1)
        else:
            raise RuntimeError("Internal logic error in naive bound calculation")
        naive_bounds = naive_bounds_all # Assign calculated bounds

        # Final bounds tensor initialized as naive bounds
        final_bounds = naive_bounds
        fill_value_inf = torch.tensor(torch.inf if sign == -1.0 else -torch.inf, device=device)

        # --- Initial Batch Filtering (Common Logic) ---
        active_indices = sorted_out_batches.get("active_indices", None)
        if active_indices is None:
            fully_covered = sorted_out_batches.get("fully_covered", torch.zeros(N_batch, dtype=torch.bool, device=device))
            active_batches_mask = ~fully_covered # Batches requiring solver
            if no_return_inf:
                infeasible_batches = sorted_out_batches.get("infeasible_batches", torch.zeros(N_batch, dtype=torch.bool, device=device))
                active_batches_mask = ~infeasible_batches & active_batches_mask
            active_indices = torch.nonzero(active_batches_mask, as_tuple=True)[0]
        B_act = active_indices.numel() # Number of batches needing the solver.
        if timer: timer.add('init') # Combined timing for setup.

        # --- Early Exit if No Active Batches ---
        if B_act == 0:
            print(f"Constrained concretize: No active batches after filtering.")
            # Ensure non-active parts have naive bounds before returning.
            # (already done above by initializing with naive/inf)
            if timer: timer.add("concretize")
            final_bounds = naive_bounds
            if no_return_inf:
                return final_bounds, None
            else:
                return final_bounds

        constraints_A, constraints_d = constraints
        n_constraints = constraints_A.size(1)

        # --- Dynamic Chunk Size Calculation ---
        if is_batch_specific:
            # If objective is batch-specific, we do not chunk it.
            num_chunks = 1
            final_chunk_size = B_act
        else:
            # This block dynamically estimates the optimal chunk size to maximize GPU
            # utilization while preventing out-of-memory (OOM) errors.
            calculated_chunk_size = B_act
            free_mem, total_mem = torch.cuda.mem_get_info()
            usable_mem = free_mem * safety_factor
            obj_dtype = objective.dtype
            dtype_size = torch.finfo(obj_dtype).bits // 8
            mem_constraints_per_item = (n_constraints * X + n_constraints) * dtype_size
            mem_x0eps_per_item = 2 * X * dtype_size
            mem_ori_c_per_item = H * X * dtype_size
            mem_dual_obj_per_item = H * dtype_size
            mem_solver_per_item_bh = H * (X + X + 1 + X + 1) * dtype_size * solver_memory_factor
            mem_masks_temps_per_item = H * 2 # approx
            mem_per_item_est = (mem_constraints_per_item + mem_x0eps_per_item +
                                mem_ori_c_per_item + mem_dual_obj_per_item +
                                mem_solver_per_item_bh + mem_masks_temps_per_item) * 5
            if mem_per_item_est > 0:
                estimated_max_chunk = max(1, floor(usable_mem / mem_per_item_est))
                calculated_chunk_size = min(B_act, estimated_max_chunk)
            if max_chunk_size is not None and max_chunk_size > 0:
                final_chunk_size = min(calculated_chunk_size, max_chunk_size)
            else:
                final_chunk_size = calculated_chunk_size
            final_chunk_size = max(1, final_chunk_size) # Ensure chunk size is at least 1.
            num_chunks = ceil(B_act / final_chunk_size)

        if no_return_inf:
            # Initialize infeasible_batches boolean mask to be None at first.
            # If an infeasible batch does occur later, we will then initialize it to be a actual vector.
            infeasible_batches = None

        for i_chunk in range(num_chunks):
            # --- Handle size and idx for this chunk ---
            chunk_start_idx_rel = i_chunk * final_chunk_size
            chunk_end_idx_rel = min(chunk_start_idx_rel + final_chunk_size, B_act)
            current_chunk_size = chunk_end_idx_rel - chunk_start_idx_rel
            if current_chunk_size == 0: continue
            chunk_indices_abs = active_indices[chunk_start_idx_rel:chunk_end_idx_rel]

            # --- Get matrices for this chunk ---
            constr_A_mat = constraints_A[chunk_start_idx_rel:chunk_end_idx_rel]             # shape (B_act, n_constraints, X)
            constr_d_mat = constraints_d[chunk_start_idx_rel:chunk_end_idx_rel]             # shape (B_act, n_constraints)
            if act_x0 is not None:
                x0_mat = act_x0[chunk_start_idx_rel:chunk_end_idx_rel]
            else:
                x0_mat = x0[chunk_indices_abs]                                              # shape (B_act, X, 1)
            if act_eps is not None:
                eps_mat = act_eps[chunk_start_idx_rel:chunk_end_idx_rel]
            else:
                eps_mat = epsilon[chunk_indices_abs]                                        # shape (B_act, X, 1)

            if is_broadcastable:
                ori_c_mat = objective_tensor.expand(current_chunk_size, H, X).clone()
            else:
                ori_c_mat = objective_tensor[chunk_indices_abs].clone()             # shape: (B_act, H, X)

            if objective_indices is not None:                                       # shape: (B, H_act) 
                # Select the mask rows corresponding to the active batches in this chunk
                current_objective_indices = objective_indices[chunk_indices_abs]    # shape: (B_act, H_act)
                idx_unsqueeze = current_objective_indices.unsqueeze(-1)             # shape: (B_act, H_act, 1)
                idx_expand = idx_unsqueeze.expand(-1, -1, X)                        # shape: (B_act, H_act, X)
                ori_c_mat = ori_c_mat.gather(index=idx_expand, dim=1)               # shape: (B_act, H_act, X)

            obj_mat = ori_c_mat                                                             # shape (B_act, H_act, X)
            # Initialize dual part and base part
            # Note that the final minimal value is:
            #    objective^T x0 +                                                                base_part
            #    constr_d_0 * beta_0 + constr_d_1 * beta_1 + ... +                               dual_part 1
            #    - ( objective+ constr_a_0 * beta_0 + constr_a_1 * beta_1)^T epsilon             dual_part 2
            base_objective_term = torch.einsum('bhx,bxo->bh', obj_mat, x0_mat) # shape: (B_act, H_act)
            dual_objective_part = torch.zeros_like(base_objective_term)        # shape: (B_act, H_act)

            # --- Initialize State for Vectorized Loop (Chunk) ---
            if sign == 1.0: # Adjust for minimization problem solved by _solve
                obj_mat *= -1.0                                                # shape (B_act, H_act, X)
                base_objective_term *= -1.0

            # --- Vectorized Constraint Loop (Operating on Chunk) ---
            for k in range(n_constraints):
                constr_a_solve = constr_A_mat[:, k, :] # constraint A matrix shape (B_act, X)
                constr_d_solve = constr_d_mat[:, k]    # related bias term   shape (B_act,)
                epsilon_solve = eps_mat                # epsilon             shape (B_act, X)
                object_a_solve = obj_mat               # objective matrix    shape (B_act, H_act, X)

                with torch.no_grad():   # Otherwise, the gradients will mess up the alpha crown optimization.
                    optimal_beta = _solve_dual_var(constr_a_solve, object_a_solve, constr_d_solve, epsilon_solve) # shape (B_act, H_act)

                # Accumulation for the parentheses term in dual part 2
                obj_mat += optimal_beta.unsqueeze(-1) * constr_a_solve.unsqueeze(1)    # shape (B_act, H_act, X)        
                #            (B_act, H_act, 1)          (B_act, 1, X)
                # Accumulation of dual part 1
                dual_objective_part += optimal_beta * constr_d_solve.unsqueeze(1)      # shape (B_act, H_act)
                #                     (B_act, H_act)    (B_act, 1)

            # --- End of k loop ---
            # --- Final Objective Calculation for Unfinished Items in Chunk ---
            final_obj_abs = obj_mat.abs() # shape: (B_act, H_act, X)
            final_eps_mat = eps_mat       # shape: (B_act, X, 1)
            final_eps_term = torch.einsum('nhx,nxo->nh', final_obj_abs, final_eps_mat) # shape: (B_act, H_act)
            dual_objective_part -= final_eps_term   

            # --- Combine terms and handle mask ---
            final_obj_minimized = base_objective_term + dual_objective_part # shape: (B_act, H_act)
            if sign == 1.0: final_obj_optimal = -final_obj_minimized        # Flip sign back if maximizing.
            else: final_obj_optimal = final_obj_minimized

            # Previously we will handle infeasible batches after running through all the chunks, during processing final_bounds.
            # But that would require to create a copy of naive bounds
            # To save space and time, we will process final_obj_optimal
            final_obj_optimal = torch.nan_to_num(final_obj_optimal, nan=fill_value_inf.item(), posinf=fill_value_inf.item(), neginf=-fill_value_inf.item())
            if no_return_inf:
                infeasible_batches_chunk = final_obj_optimal.isinf().any(1)
                if infeasible_batches_chunk.any():
                    # Note that infeasible_batches was initialized as None
                    infeasible_batches = torch.full((N_batch, ), fill_value=False, device=device, dtype=torch.bool) if infeasible_batches is None else infeasible_batches
                    infeasible_batches[chunk_indices_abs] = infeasible_batches_chunk
                    # Set the bounds of infeasible batches to be naive bounds
                    infeasible_batches_chunk_indices_abs = chunk_indices_abs[infeasible_batches_chunk]
                    if objective_indices is not None:
                        naive_bounds_chunk = naive_bounds[infeasible_batches_chunk_indices_abs].squeeze(-1)
                        # Get the infeasible objective indices for this chunk.
                        current_infeasible_objective_indices = current_objective_indices[infeasible_batches_chunk]
                        final_obj_optimal[infeasible_batches_chunk] = torch.gather(naive_bounds_chunk, dim=1, index=current_infeasible_objective_indices)
                    else:
                        final_obj_optimal[infeasible_batches_chunk] = naive_bounds[infeasible_batches_chunk_indices_abs].squeeze(-1)

            # Put the result of this chunk back into the overall final bounds
            if objective_indices is not None:
                final_bounds_active_chunk = final_bounds[chunk_indices_abs]  
                final_bounds_active_chunk.scatter_(dim=1, index=idx_unsqueeze, src=final_obj_optimal.unsqueeze(-1))
                final_bounds[chunk_indices_abs] = final_bounds_active_chunk
            else:
                final_bounds[chunk_indices_abs] = final_obj_optimal.unsqueeze(-1)

        if no_return_inf:
            if timer: timer.add("concretize")
            return final_bounds, infeasible_batches
        else:
            if timer: timer.add("concretize")
            return final_bounds


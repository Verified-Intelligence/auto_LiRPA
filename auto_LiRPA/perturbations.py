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
import json
import math
import os
import numpy as np
import torch
from .utils import logger, eyeC
from .patches import Patches, patches_to_matrix
from .linear_bound import LinearBound

from .concretize_func import constraints_solving, sort_out_constr_batches, construct_constraints

class Perturbation:
    r"""
    Base class for a perturbation specification. Please see examples
    at `auto_LiRPA/perturbations.py`.

    Examples:

    * `PerturbationLpNorm`: Lp-norm (p>=1) perturbation.

    * `PerturbationL0Norm`: L0-norm perturbation.

    * `PerturbationSynonym`: Synonym substitution perturbation for NLP.
    """

    def __init__(self):
        pass

    def set_eps(self, eps):
        self.eps = eps

    def concretize(self, x, A, sign=-1, aux=None):
        r"""
        Concretize bounds according to the perturbation specification.

        Args:
            x (Tensor): Input before perturbation.

            A (Tensor) : A matrix from LiRPA computation.

            sign (-1 or +1): If -1, concretize for lower bound; if +1, concretize for upper bound.

            aux (object, optional): Auxilary information for concretization.

        Returns:
            bound (Tensor): concretized bound with the shape equal to the clean output.
        """
        raise NotImplementedError

    def init(self, x, aux=None, forward=False):
        r"""
        Initialize bounds before LiRPA computation.

        Args:
            x (Tensor): Input before perturbation.

            aux (object, optional): Auxilary information.

            forward (bool): It indicates whether forward mode LiRPA is involved.

        Returns:
            bound (LinearBound): Initialized bounds.

            center (Tensor): Center of perturbation. It can simply be `x`, or some other value.

            aux (object, optional): Auxilary information. Bound initialization may modify or add auxilary information.
        """

        raise NotImplementedError


class PerturbationL0Norm(Perturbation):
    """Perturbation constrained by the L_0 norm.

    Assuming input data is in the range of 0-1.
    """

    def __init__(self, eps, x_L=None, x_U=None, ratio=1.0):
        self.eps = eps
        self.x_U = x_U
        self.x_L = x_L
        self.ratio = ratio

    def concretize(self, x, A, sign=-1, aux=None):
        if A is None:
            return None

        eps = math.ceil(self.eps)
        x = x.reshape(x.shape[0], -1, 1)
        center = A.matmul(x)

        x = x.reshape(x.shape[0], 1, -1)

        original = A * x.expand(x.shape[0], A.shape[-2], x.shape[2])
        neg_mask = A < 0
        pos_mask = A >= 0

        if sign == 1:
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = A[pos_mask] - original[pos_mask]# changes that one weight can contribute to the value
            A_diff[neg_mask] = - original[neg_mask]
        else:
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = original[pos_mask]
            A_diff[neg_mask] = original[neg_mask] - A[neg_mask]

        # FIXME: this assumes the input pixel range is between 0 and 1!
        A_diff, _= torch.sort(A_diff, dim = 2, descending=True)

        bound = center + sign * A_diff[:, :, :eps].sum(dim = 2).unsqueeze(2) * self.ratio

        return bound.squeeze(2)

    def init(self, x, aux=None, forward=False):
        # For other norms, we pass in the BoundedTensor objects directly.
        x_L = x
        x_U = x
        if not forward:
            return LinearBound(None, None, None, None, x_L, x_U), x, None
        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        eye = torch.eye(dim).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        lw = eye.reshape(batch_size, dim, *x.shape[1:])
        lb = torch.zeros_like(x).to(x.device)
        uw, ub = lw.clone(), lb.clone()
        return LinearBound(lw, lb, uw, ub, x_L, x_U), x, None

    def __repr__(self):
        return 'PerturbationLpNorm(norm=0, eps={})'.format(self.eps)


class PerturbationLpNorm(Perturbation):
    """Perturbation constrained by the L_p norm."""
    def __init__(self, eps=0, norm=np.inf, x_L=None, x_U=None, eps_min=0,
                 constraints=None, rearrange_constraints=False, no_return_inf=False, timer=None):
        r"""
        Initialize a p-norm perturbation instance.
        There are two ways to initialize it:
            -- x_L, x_U: (Higher priority)
            -- eps     : (Lower priority)
        If use eps to initialize it, the centroid x (or x0 as in the member attribute) will be
            passed into `init` and `concretize` function.  
        For the shape notations such as 'B' or 'X', please check the shape declaration 
            at the beginning of concretize_func.py

        Args:
            eps (Tensor): The epsilon tensor, it represents the pertubation added to a BoundedTensor.
            norm (int or torch.inf): The p in p-norm perturbation.
            x_L (Tensor): Lower bound of input box, shape (B, *input_shape[1:]).
            x_U (Tensor): Upper bound of input box, shape (B, *input_shape[1:]).
            eps_min ()
            constraints (Tuple[Tensor, Tensor] or None): 
                A tuple `(A, b)` representing per-batch linear constraints.
                - `A`: shape (B, N_constr, X)
                - `b`: shape (B, N_constr)
            rearrange_constraints (bool): 
                Whether to rearrange constraints for better solver performance. Default: False.
            no_return_inf (bool): 
                If True, infeasible batches will be excluded from `active_indices`.
                Otherwise, infeasible batches are still marked active. Default: False.
                Please check `constraints_solving` and `sort_out_constr_batches` for more details.
            timer (Timer):
                A timer recording the concretization time.
        """        
        self.eps = eps
        self.x0 = None

        # For p = inf, pre-compute x0 and eps would accerlerate the concretize function.
        if norm == np.inf and x_L is not None and x_U is not None:
            self.eps = (x_U - x_L) / 2
            self.x0 = (x_U + x_L) / 2
        
        # x0_act and eps_act stands for x0 and eps matrix for batches with active constraints
        self.x0_act = None          # shape (batchsize, *X_shape)
        self.eps_act = None         # shape (batchsize, *X_shape)
        # x0_sparse_act and eps_sparse_act are the active sparse x0 and eps matrix when sparse perturbation is enabled.
        # Check init_sparse_linf to see how sparse x0, eps, x0_act, eps_act are created.
        self.x0_sparse_act = None   # shape (batchsize, *X_sparse_shape)
        self.eps_sparse_act = None  # shape (batchsize, *X_sparse_shape)

        self.eps_min = eps_min
        self.norm = norm
        self.dual_norm = 1 if (norm == np.inf) else (np.float64(1.0) / (1 - 1.0 / self.norm))
        self.x_L = x_L
        self.x_U = x_U
        self.sparse = False

        self.timer = timer
        self.aux_lb = None
        self.aux_ub = None

        self.rearrange_constraints = rearrange_constraints

        # constraints is a tuple containing both the coefficient matrix and bias term
        # of the constraints. The constraints would appear in the form of:
        #                           A_c * x + b_c <= 0
        # Coefficient matrix will be reshaped into (batchsize, # of constraints,
        # input_dim). Bias term will be reshaped into (batchsize, # of constraints)
        # also see in `constraints_solving` in constraints_solver.py

        # Pre-process the constraints.
        self.constraints, self.sorted_out_batches = sort_out_constr_batches(x_L, x_U, constraints,
                                                                            rearrange_constraints=rearrange_constraints,
                                                                            no_return_inf=no_return_inf)
        # The indices of hidden neurons to apply constraints.
        self.objective_indices = None   # shape: (batchsize, num_of_neurons)
        if self.constraints is None or self.constraints[0].numel() == 0:
            self._constraints_enable = False
        else:
            self._constraints_enable = True
        self.no_return_inf = no_return_inf

        self._use_grad = False

    def get_input_bounds(self, x, A):
        if self.sparse:
            if self.x_L_sparse.shape[-1] == A.shape[-1]:
                x_L, x_U = self.x_L_sparse, self.x_U_sparse
                act_x0, act_eps = self.x0_sparse_act, self.eps_sparse_act
            else:
                # In backward mode, A is not sparse.
                x_L, x_U = self.x_L, self.x_U
                act_x0, act_eps = self.x0_act, self.eps_act
        else:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
            act_x0, act_eps = self.x0_act, self.eps_act
        return x_L, x_U, act_x0, act_eps

    def get_constraints(self, A):
        if self.constraints is None:
            return None
        if self.sparse and self.x_L_sparse.shape[-1] == A.shape[-1]:
            return self.constraints_sparse
        else:
            return self.constraints

    def concretize_matrix(self, x, A, sign, constraints=None):
        # If A is an identity matrix, we will handle specially.
        if not isinstance(A, eyeC):
            # A has (Batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
            A = A.reshape(A.shape[0], A.shape[1], -1)

        if self.norm == np.inf:
            x_L, x_U, act_x0, act_eps = self.get_input_bounds(x, A)
            if constraints is None:
                constraints = self.get_constraints(A)
            # The original code for matrix concretize has been merged into `constraints_solving`.
            # Pick out auxiliary bound based on the sign.
            aux_bounds = self.aux_lb if sign == -1.0 else self.aux_ub
            results = constraints_solving(x_L, x_U, A, constraints, sign,
                                        sorted_out_batches=self.sorted_out_batches, objective_indices=self.objective_indices, 
                                        constraints_enable=self._constraints_enable, no_return_inf=self.no_return_inf,
                                        timer=self.timer, 
                                        aux_bounds=aux_bounds, act_x0=act_x0, act_eps=act_eps,
                                        use_grad=self._use_grad)
            
            if self.no_return_inf:
                # return: (bound, infeasible_bounds)
                bound = results[0]
                infeasible_bounds = results[1]
                self.add_infeasible_batches(infeasible_bounds)
            else:
                # return: bound
                bound = results
        else:
            x = x.reshape(x.shape[0], -1, 1)
            if not isinstance(A, eyeC):
                # Find the upper and lower bounds via dual norm.
                deviation = A.norm(self.dual_norm, -1) * self.eps
                bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
            else:
                # A is an identity matrix. Its norm is all 1.
                bound = x + sign * self.eps
        bound = bound.squeeze(-1)
        return bound

    def concretize_patches(self, x, A, sign):
        if self.norm == np.inf:
            x_L, x_U, _, _,  = self.get_input_bounds(x, A)

            # Here we should not reshape
            # Find the uppwer and lower bound similarly to IBP.
            center = (x_U + x_L) / 2.0
            diff = (x_U - x_L) / 2.0

            if not A.identity == 1:
                bound = A.matmul(center)
                bound_diff = A.matmul(diff, patch_abs=True)
                if sign == 1:
                    bound += bound_diff
                elif sign == -1:
                    bound -= bound_diff
                else:
                    raise ValueError("Unsupported Sign")
            else:
                # A is an identity matrix. No need to do this matmul.
                bound = center + sign * diff
            return bound
        else:  # Lp norm
            input_shape = x.shape
            if not A.identity:
                # Find the upper and lower bounds via dual norm.
                # matrix has shape
                # (batch_size, out_c * out_h * out_w, input_c, input_h, input_w)
                # or (batch_size, unstable_size, input_c, input_h, input_w)
                matrix = patches_to_matrix(
                    A.patches, input_shape, A.stride, A.padding, A.output_shape,
                    A.unstable_idx)
                # Note that we should avoid reshape the matrix.
                # Due to padding, matrix cannot be reshaped without copying.
                deviation = matrix.norm(p=self.dual_norm, dim=(-3,-2,-1)) * self.eps
                # Bound has shape (batch, out_c * out_h * out_w) or (batch, unstable_size).
                bound = torch.einsum('bschw,bchw->bs', matrix, x) + sign * deviation
                if A.unstable_idx is None:
                    # Reshape to (batch, out_c, out_h, out_w).
                    bound = bound.view(matrix.size(0), A.patches.size(0),
                                       A.patches.size(2), A.patches.size(3))
            else:
                # A is an identity matrix. Its norm is all 1.
                bound = x + sign * self.eps
            return bound

    def concretize(self, x, A, sign=-1, constraints=None, aux=None):
        """Given an variable x and its bound matrix A, compute worst case bound according to Lp norm."""
        if A is None:
            return None
        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            ret = self.concretize_matrix(x, A, sign, constraints)
        elif isinstance(A, Patches):
            ret = self.concretize_patches(x, A, sign)
        else:
            raise NotImplementedError()
        if ret.ndim > 2:
            ret = ret.reshape(A.shape[1], -1)
        return ret

    def init_sparse_linf(self, x, x_L, x_U):
        """ Sparse Linf perturbation where only a few dimensions are actually perturbed"""
        self.sparse = True
        batch_size = x_L.shape[0]
        perturbed = (x_U > x_L).int()
        logger.debug(f'Perturbed: {perturbed.sum()}')
        lb = ub = x_L * (1 - perturbed) # x_L=x_U holds when perturbed=0
        perturbed = perturbed.view(batch_size, -1)
        index = torch.cumsum(perturbed, dim=-1)
        dim = max(perturbed.view(batch_size, -1).sum(dim=-1).max(), 1)
        self.x_L_sparse = torch.zeros(batch_size, dim + 1).to(x_L)
        self.x_L_sparse.scatter_(dim=-1, index=index, src=(x_L - lb).view(batch_size, -1), reduce='add')
        self.x_U_sparse = torch.zeros(batch_size, dim + 1).to(x_U)
        self.x_U_sparse.scatter_(dim=-1, index=index, src=(x_U - ub).view(batch_size, -1), reduce='add')
        self.x_L_sparse, self.x_U_sparse = self.x_L_sparse[:, 1:], self.x_U_sparse[:, 1:]
        
        # --- create x0 and eps for Lp Norm
        self.x0_sparse = (self.x_L_sparse + self.x_U_sparse) / 2
        self.eps_sparse = (self.x_U_sparse - self.x_L_sparse) / 2
        if self.sorted_out_batches is not None:
            active_indices = self.sorted_out_batches["active_indices"]
            self.x0_sparse_act = self.x0_sparse[active_indices].unsqueeze(-1)
            self.eps_sparse_act = self.eps_sparse[active_indices].unsqueeze(-1)

        lw = torch.zeros(batch_size, dim + 1, perturbed.shape[-1], device=x.device)
        perturbed = perturbed.to(torch.get_default_dtype())
        lw.scatter_(dim=1, index=index.unsqueeze(1), src=perturbed.unsqueeze(1))
        lw = uw = lw[:, 1:, :].view(batch_size, dim, *x.shape[1:])
        print(f'Using Linf sparse perturbation. Perturbed dimensions: {dim}.')
        print(f'Avg perturbation: {(self.x_U_sparse - self.x_L_sparse).mean()}')

        # When sparse linf is enabled, the input x perturbation would change its shape
        # Hence, the shape of constraints_A should change accordingly.
        # But for the final layer, we still need the dense linf, and use the original (dense) constraints
        if self.constraints is not None:
            # constraints_A: (batchsize, n_constraints, x_dim)
            constraints_A, constraints_b = self.constraints
            # reversed_lw: (batchsize, x_dim, sparse_dim)
            reversed_lw = lw.reshape((batch_size, dim, -1)).transpose(1, 2)
            lb_act = lb
            # When pre-processing the constraints, we only kept the active ones.
            # Hence, reversed_lw and lb_act should also be re-collected.
            active_indices = self.sorted_out_batches["active_indices"]
            reversed_lw = reversed_lw[active_indices]
            lb_act = lb_act[active_indices]
            # reversed lw will sort out the sparse dimensions out of all x dimension
            new_constraints_A = constraints_A.bmm(reversed_lw)
            # Besides original constraint_b, should also include the a*x terms where x is not perturbed
            # new_constraints_b = constraints_b + torch.einsum("bcx, bx -> bc", constraints_A, lb_act.flatten(1))
            new_constraints_b = constraints_b
            self.constraints_sparse = (new_constraints_A, new_constraints_b)
        return LinearBound(
            lw, lb, uw, ub, x_L, x_U), x, None

    def init(self, x, aux=None, forward=False):
        self.sparse = False
        if self.norm == np.inf:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        else:
            if int(os.environ.get('AUTOLIRPA_L2_DEBUG', 0)) == 1:
                # FIXME Experimental code. Need to change the IBP code also.
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U
            else:
                # FIXME This causes confusing lower bound and upper bound
                # For other norms, we pass in the BoundedTensor objects directly.
                x_L = x_U = x

        if self.x_L is not None and self.x_U is not None:
            self.x0 = (self.x_L + self.x_U) / 2
        else:
            self.x0 = x.data
        if self.sorted_out_batches is not None and self.sorted_out_batches.get("active_indices") is not None:
            active_indices = self.sorted_out_batches["active_indices"]
            self.x0_act = self.x0[active_indices].flatten(1).unsqueeze(-1)
            self.eps_act = self.eps[active_indices].flatten(1).unsqueeze(-1)

        if not forward:
            return LinearBound(
                None, None, None, None, x_L, x_U), x, None
        if (self.norm == np.inf and x_L.numel() > 1
                and (x_L == x_U).sum() > 0.5 * x_L.numel()):
            return self.init_sparse_linf(x, x_L, x_U)

        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        lb = ub = torch.zeros_like(x)
        eye = torch.eye(dim).to(x).expand(batch_size, dim, dim)
        lw = uw = eye.reshape(batch_size, dim, *x.shape[1:])
        return LinearBound(
            lw, lb, uw, ub, x_L, x_U), x, None

    def add_infeasible_batches(self, infeasible_batches):
        r"""
        Synchronize the `infeasible_batches` tensor between the global graph and the local perturbation node.

        If the computation graph includes multiple perturbed inputs, the BoundedModule (entire network) maintains a global
        `infeasible_batches` tensor, while each perturbed input (root) keeps its own local copy.

        - Before concretization: copy the global tensor to the local one.
        - After concretization: propagate updates from the local tensor back to the global tensor.

        Args:
            infeasible_batches: A boolean vector with shape (batchsize, ). A True value indicates that a batch is infeasible
                                given its constraints.
        """
        if self.constraints is not None and infeasible_batches is not None and infeasible_batches.any():
            if self.sorted_out_batches["infeasible_batches"] is None:
                self.sorted_out_batches["infeasible_batches"] = infeasible_batches
            else:
                infeasible_batches = infeasible_batches | self.sorted_out_batches["infeasible_batches"]
                self.sorted_out_batches["infeasible_batches"] = infeasible_batches
            
            active_indices = self.sorted_out_batches["active_indices"]
            B_act = active_indices.numel()
            active_feasible_mask = (~infeasible_batches)[active_indices]
            if active_feasible_mask.sum() < B_act:
                self.sorted_out_batches["active_indices"] = active_indices[active_feasible_mask]
                self.x0_act = self.x0_act[active_feasible_mask]
                self.eps_act = self.eps_act[active_feasible_mask]
                constraints_A, constraints_b = self.constraints
                constraints_A = constraints_A[active_feasible_mask]
                constraints_b = constraints_b[active_feasible_mask]
                self.constraints = (constraints_A, constraints_b)

    def add_objective_indices(self, objective_indices):
        if self.constraints is not None:
            self.objective_indices = objective_indices

    @property
    def constraints_enable(self):
        '''
        Enable / Disable the constrained concretize mode, regardless whether constraints is None or not. 
        '''
        return self._constraints_enable
    
    @constraints_enable.setter
    def constraints_enable(self, enable: bool):
        self._constraints_enable = enable

    @constraints_enable.deleter
    def constraints_enable(self):
        del self._constraints_enable  

    @property
    def use_grad(self):
        '''
        Enable / Disable the constrained concretize with gradient. 
        '''
        return self._use_grad
    
    @use_grad.setter
    def use_grad(self, use_grad: bool):
        self._use_grad = use_grad

    @use_grad.deleter
    def use_grad(self):
        del self._use_grad  

    def add_aux_bounds(self, aux_lb, aux_ub):
        self.aux_lb = aux_lb
        self.aux_ub = aux_ub

    def clear_aux_bounds(self):
        self.aux_lb = None
        self.aux_ub = None

    def reset_constraints(self, constraints, decision_thresh):
        r"""
        Reset the constraints of this perturbation. Also will call `sort_out_constr_batches` to preprocess the constraints.
        Be sure not to reset with the same constraints input repeatedly.
        """
        # We have to enable the gradient computation for the constraints
        # when using constraints_solving within alpha crown.
        self.use_grad = True
        constraints = construct_constraints(constraints[0], constraints[1], decision_thresh, self.x_L.shape[0], self.x_L.flatten(1).shape[1])
        self.constraints, self.sorted_out_batches = sort_out_constr_batches(self.x_L, self.x_U, constraints, 
                                                                            rearrange_constraints=self.rearrange_constraints,
                                                                            no_return_inf=self.no_return_inf)

    def __repr__(self):
        if self.norm == np.inf:
            if self.x_L is None and self.x_U is None:
                return f'PerturbationLpNorm(norm=inf, eps={self.eps})'
            else:
                return f'PerturbationLpNorm(norm=inf, eps={self.eps}, x_L={self.x_L}, x_U={self.x_U})'
        else:
            return f'PerturbationLpNorm(norm={self.norm}, eps={self.eps})'


class PerturbationLinear(Perturbation):
    """
    Perturbation defined by a Linear transformation.
    args:
        lower_A: Lower bound matrix of shape (B, output_dim, input_dim)
        upper_A: Upper bound matrix of shape (B, output_dim, input_dim)
        lower_b: Lower bound bias of shape (B, output_dim)
        upper_b: Upper bound bias of shape (B, output_dim)
        input_lb: Input lower bound of shape (B, input_dim)
        input_ub: Input upper bound of shape (B, input_dim)
        x_L: Output lower bound of shape (B, output_dim)
        x_U: Output upper bound of shape (B, output_dim)

        x_L and x_U can be None, in which case they will be computed from the other parameters.    
    """
    def __init__(self, lower_A, upper_A, lower_b, upper_b, input_lb, input_ub, x_L=None, x_U=None):
        super(PerturbationLinear, self).__init__()
        self.lower_A = lower_A
        self.upper_A = upper_A
        self.lower_b = lower_b.unsqueeze(-1) if lower_b is not None else None
        self.upper_b = upper_b.unsqueeze(-1) if upper_b is not None else None
        self.input_lb = input_lb.unsqueeze(-1) if input_lb is not None else None
        self.input_ub = input_ub.unsqueeze(-1) if input_ub is not None else None
        if x_L is None or x_U is None:
            mid = (self.input_lb + self.input_ub) / 2
            diff = (self.input_ub - self.input_lb) / 2
            self.x_U = (self.upper_A @ mid + torch.abs(self.upper_A) @ diff + self.upper_b).squeeze(-1)
            self.x_L = (self.lower_A @ mid - torch.abs(self.lower_A) @ diff + self.lower_b).squeeze(-1)
        else:
            self.x_L = x_L
            self.x_U = x_U

    def concretize(self, x, A, sign=-1, aux=None):
        if A is None:
            return None
        else:
            A_pos = torch.clamp(A, min=0)
            A_neg = torch.clamp(A, max=0)

            center = (self.input_lb + self.input_ub) / 2
            diff = (self.input_ub - self.input_lb) / 2

            if sign == 1:
                composite_A = A_pos @ self.upper_A + A_neg @ self.lower_A
                composite_b = A_pos @ self.upper_b + A_neg @ self.lower_b
                bound = composite_A @ center + torch.abs(composite_A) @ diff + composite_b
            else:
                composite_A = A_pos @ self.lower_A + A_neg @ self.upper_A
                composite_b = A_pos @ self.lower_b + A_neg @ self.upper_b
                bound = composite_A @ center - torch.abs(composite_A) @ diff + composite_b
            return bound.squeeze(-1)

    def init(self, x, aux=None, forward=False):
        if not forward:
            return LinearBound(None, None, None, None, self.x_L, self.x_U), x, None
        else:
            raise NotImplementedError("Linear perturbation does not support forward mode.")


class PerturbationSynonym(Perturbation):
    def __init__(self, budget, eps=1.0, use_simple=False):
        super(PerturbationSynonym, self).__init__()
        self._load_synonyms()
        self.budget = budget
        self.eps = eps
        self.use_simple = use_simple
        self.model = None
        self.train = False

    def __repr__(self):
        return (f'perturbation(Synonym-based word substitution '
                f'budget={self.budget}, eps={self.eps})')

    def _load_synonyms(self, path='data/synonyms.json'):
        with open(path) as file:
            self.synonym = json.loads(file.read())
        logger.info('Synonym list loaded for {} words'.format(len(self.synonym)))

    def set_train(self, train):
        self.train = train

    def concretize(self, x, A, sign, aux):
        assert(self.model is not None)

        x_rep, mask, can_be_replaced = aux
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]
        dim_out = A.shape[1]
        max_num_cand = x_rep.shape[2]

        mask_rep = torch.tensor(can_be_replaced, dtype=torch.get_default_dtype(), device=A.device)

        num_pos = int(np.max(np.sum(can_be_replaced, axis=-1)))
        update_A = A.shape[-1] > num_pos * dim_word
        if update_A:
            bias = torch.bmm(A, (x * (1 - mask_rep).unsqueeze(-1)).reshape(batch_size, -1, 1)).squeeze(-1)
        else:
            bias = 0.
        A = A.reshape(batch_size, dim_out, -1, dim_word)

        A_new, x_new, x_rep_new, mask_new = [], [], [], []
        zeros_A = torch.zeros(dim_out, dim_word, device=A.device)
        zeros_w = torch.zeros(dim_word, device=A.device)
        zeros_rep = torch.zeros(max_num_cand, dim_word, device=A.device)
        zeros_mask = torch.zeros(max_num_cand, device=A.device)
        for t in range(batch_size):
            cnt = 0
            for i in range(0, length):
                if can_be_replaced[t][i]:
                    if update_A:
                        A_new.append(A[t, :, i, :])
                    x_new.append(x[t][i])
                    x_rep_new.append(x_rep[t][i])
                    mask_new.append(mask[t][i])
                    cnt += 1
            if update_A:
                A_new += [zeros_A] * (num_pos - cnt)
            x_new += [zeros_w] * (num_pos - cnt)
            x_rep_new += [zeros_rep] * (num_pos - cnt)
            mask_new += [zeros_mask] * (num_pos - cnt)
        if update_A:
            A = torch.cat(A_new).reshape(batch_size, num_pos, dim_out, dim_word).transpose(1, 2)
        x = torch.cat(x_new).reshape(batch_size, num_pos, dim_word)
        x_rep = torch.cat(x_rep_new).reshape(batch_size, num_pos, max_num_cand, dim_word)
        mask = torch.cat(mask_new).reshape(batch_size, num_pos, max_num_cand)
        length = num_pos

        A = A.reshape(batch_size, A.shape[1], length, -1).transpose(1, 2)
        x = x.reshape(batch_size, length, -1, 1)

        if sign == 1:
            cmp, init = torch.max, -1e30
        else:
            cmp, init = torch.min, 1e30

        init_tensor = torch.ones(batch_size, dim_out).to(x.device) * init
        dp = [[init_tensor] * (self.budget + 1) for i in range(0, length + 1)]
        dp[0][0] = torch.zeros(batch_size, dim_out).to(x.device)

        A = A.reshape(batch_size * length, A.shape[2], A.shape[3])
        Ax = torch.bmm(
            A,
            x.reshape(batch_size * length, x.shape[2], x.shape[3])
        ).reshape(batch_size, length, A.shape[1])

        Ax_rep = torch.bmm(
            A,
            x_rep.reshape(batch_size * length, max_num_cand, x.shape[2]).transpose(-1, -2)
        ).reshape(batch_size, length, A.shape[1], max_num_cand)
        Ax_rep = Ax_rep * mask.unsqueeze(2) + init * (1 - mask).unsqueeze(2)
        Ax_rep_bound = cmp(Ax_rep, dim=-1).values

        if self.use_simple and self.train:
            return torch.sum(cmp(Ax, Ax_rep_bound), dim=1) + bias

        for i in range(1, length + 1):
            dp[i][0] = dp[i - 1][0] + Ax[:, i - 1]
            for j in range(1, self.budget + 1):
                dp[i][j] = cmp(
                    dp[i - 1][j] + Ax[:, i - 1],
                    dp[i - 1][j - 1] + Ax_rep_bound[:, i - 1]
                )
        dp = torch.cat(dp[length], dim=0).reshape(self.budget + 1, batch_size, dim_out)

        return cmp(dp, dim=0).values + bias

    def init(self, x, aux=None, forward=False):
        tokens, batch = aux
        self.tokens = tokens # DEBUG
        assert(len(x.shape) == 3)
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]

        max_pos = 1
        can_be_replaced = np.zeros((batch_size, length), dtype=bool)

        self._build_substitution(batch)

        for t in range(batch_size):
            cnt = 0
            candidates = batch[t]['candidates']
            # for transformers
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            for i in range(len(tokens[t])):
                if tokens[t][i] == '[UNK]' or \
                        len(candidates[i]) == 0 or tokens[t][i] != candidates[i][0]:
                    continue
                for w in candidates[i][1:]:
                    if w in self.model.vocab:
                        can_be_replaced[t][i] = True
                        cnt += 1
                        break
            max_pos = max(max_pos, cnt)

        dim = max_pos * dim_word
        if forward:
            eye = torch.eye(dim_word).to(x.device)
            lw = torch.zeros(batch_size, dim, length, dim_word).to(x.device)
            lb = torch.zeros_like(x).to(x.device)
        word_embeddings = self.model.word_embeddings.weight
        vocab = self.model.vocab
        x_rep = [[[] for i in range(length)] for t in range(batch_size)]
        max_num_cand = 1
        for t in range(batch_size):
            candidates = batch[t]['candidates']
            # for transformers
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            cnt = 0
            for i in range(length):
                if can_be_replaced[t][i]:
                    word_embed = word_embeddings[vocab[tokens[t][i]]]
                    # positional embedding and token type embedding
                    other_embed = x[t, i] - word_embed
                    if forward:
                        lw[t, (cnt * dim_word):((cnt + 1) * dim_word), i, :] = eye
                        lb[t, i, :] = torch.zeros_like(word_embed)
                    for w in candidates[i][1:]:
                        if w in self.model.vocab:
                            x_rep[t][i].append(
                                word_embeddings[self.model.vocab[w]] + other_embed)
                    max_num_cand = max(max_num_cand, len(x_rep[t][i]))
                    cnt += 1
                else:
                    if forward:
                        lb[t, i, :] = x[t, i, :]
        if forward:
            uw, ub = lw, lb
        else:
            lw = lb = uw = ub = None
        zeros = torch.zeros(dim_word, device=x.device)

        x_rep_, mask = [], []
        for t in range(batch_size):
            for i in range(length):
                x_rep_ += x_rep[t][i] + [zeros] * (max_num_cand - len(x_rep[t][i]))
                mask += [1] * len(x_rep[t][i]) + [0] * (max_num_cand - len(x_rep[t][i]))
        x_rep_ = torch.cat(x_rep_).reshape(batch_size, length, max_num_cand, dim_word)
        mask = torch.tensor(mask, dtype=torch.get_default_dtype(), device=x.device)\
            .reshape(batch_size, length, max_num_cand)
        x_rep_ = x_rep_ * self.eps + x.unsqueeze(2) * (1 - self.eps)

        inf = 1e20
        lower = torch.min(mask.unsqueeze(-1) * x_rep_ + (1 - mask).unsqueeze(-1) * inf, dim=2).values
        upper = torch.max(mask.unsqueeze(-1) * x_rep_ + (1 - mask).unsqueeze(-1) * (-inf), dim=2).values
        lower = torch.min(lower, x)
        upper = torch.max(upper, x)

        return LinearBound(lw, lb, uw, ub, lower, upper), x, (x_rep_, mask, can_be_replaced)

    def _build_substitution(self, batch):
        for example in batch:
            if not 'candidates' in example or example['candidates'] is None:
                candidates = []
                tokens = example['sentence'].strip().lower().split(' ')
                for i in range(len(tokens)):
                    _cand = []
                    if tokens[i] in self.synonym:
                        for w in self.synonym[tokens[i]]:
                            if w in self.model.vocab:
                                _cand.append(w)
                    if len(_cand) > 0:
                        _cand = [tokens[i]] + _cand
                    candidates.append(_cand)
                example['candidates'] = candidates

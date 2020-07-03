import os
import copy
import pdb
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, \
    AdaptiveAvgPool2d, ConstantPad2d, AvgPool2d, Tanh
from auto_LiRPA.utils import logger, recursive_map, eyeC, LinearBound
from auto_LiRPA.perturbations import Perturbation, PerturbationLpNorm, PerturbationSynonym


"""Interval object. Used for interval bound propagation."""
class Interval(tuple):
    # Subclassing tuple object so that all previous code can be reused.
    def __new__(self, lb=None, ub=None, ptb=None):
        # FIXME: for the issue in model loading
        if ub is None:
            assert(isinstance(lb, tuple))
            lb, ub = lb

        return tuple.__new__(Interval, (lb, ub))
    
    def __init__(self, lb, ub, ptb=None):
        if ptb is None:
            # We do not perturb this interval. It shall be treated as a constant and lb = ub.
            self.ptb = None
            # To avoid mistakes, in this case the caller must make sure lb and ub are the same object.
            assert lb is ub
        else:
            if not isinstance(ptb, Perturbation):
                raise ValueError("ptb must be a Perturbation object or None. Got type {}".format(type(ptb)))
            else:
                self.ptb = ptb

    def __str__(self):
        return "({}, {}) with ptb={}".format(self[0], self[1], self.ptb)

    def __repr__(self):
        return "Interval(lb={}, ub={}, ptb={})".format(self[0], self[1], self.ptb)

    """Checking if the other interval is tuple, keep the perturbation."""
    @staticmethod
    def make_interval(lb, ub, other):
        if isinstance(other, Interval):
            return Interval(lb, ub, other.ptb)
        else:
            return (lb, ub)

    """Given a tuple or Interval object, returns the norm and eps."""
    @staticmethod
    def get_perturbation(interval):
        if isinstance(interval, Interval):
            if isinstance(interval.ptb, PerturbationLpNorm):
                return interval.ptb.norm, interval.ptb.eps
            elif isinstance(interval.ptb, PerturbationSynonym):
                return np.inf, 1.0
            elif interval.ptb is None:
                raise RuntimeError("get_perturbation() encountered an interval that is not perturbed.")
            else:
                raise RuntimeError("get_perturbation() does not know how to handle {}".format(type(interval.ptb)))
        else:
            # Tuple object. Assuming L infinity norm lower and upper bounds.
            return np.inf, np.nan

    """Checking if a Interval or tuple object has perturbation enabled."""
    @staticmethod
    def is_perturbed(interval):
        if isinstance(interval, Interval) and interval.ptb is None:
            return False
        else:
            return True


class Bound(nn.Module):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(Bound, self).__init__()
        self.output_name = []
        self.input_name, self.name, self.ori_name, self.attr, self.inputs, self.output_index, self.device = \
            input_name, name, ori_name, attr, inputs, output_index, device
        self.forward_value = None
        self.from_input = False
        self.bounded = False
        self.IBP_rets = None
        # Determine if this node has a perturbed output or not. The function BoundedModule._mark_perturbed_nodes() will set this property.
        self.node_perturbed = False

    """Check if the i-th input is with perturbation or not."""
    def is_input_perturbed(self, i=0):
        return self.inputs[i].node_perturbed

    def forward(self, *input):
        raise NotImplementedError

    def interval_propagate(self, *v):
        assert (len(v) == 1)
        # unary monotonous functions only 
        h_L, h_U = v[0]
        return Interval.make_interval(self.forward(h_L), self.forward(h_U), v[0])

    def bound_forward(self, dim_in, last):
        raise NotImplementedError

    def bound_backward(self, last_lA, last_uA):
        raise NotImplementedError

    def _broadcast_backward(self, A, input):
        shape = input.default_shape
        if not input.from_input:
            shape = torch.Size([A.shape[0]] + list(shape))
        while len(A.shape[2:]) > len(shape[1:]):
            A = torch.sum(A, dim=2)
        for i in range(1, len(shape)):
            if shape[i] == 1:
                A = torch.sum(A, dim=(i + 1), keepdim=True)
        assert (A.shape[2:] == shape[1:])
        return A

    def _broadcast_forward(self, dim_in, x, shape_res):
        lw, lb, uw, ub = x.lw, x.lb, x.uw, x.ub
        shape_x, shape_res = list(x.lb.shape), list(shape_res)
        if lw is None:
            lw = uw = torch.zeros(dim_in, *shape_x, device=lb.device)
            has_batch_size = False
        else:
            has_batch_size = True
        while len(shape_x) < len(shape_res):
            if not has_batch_size:
                lw, uw = lw.unsqueeze(0), uw.unsqueeze(0)
                lb, ub = lb.unsqueeze(0), ub.unsqueeze(0)
                shape_x = [1] + shape_x
                has_batch_size = True
            else:
                lw, uw = lw.unsqueeze(2), uw.unsqueeze(2)
                lb, ub = lb.unsqueeze(1), ub.unsqueeze(1)
                shape_x = [shape_x[0], 1] + shape_x[1:]
        repeat = [(shape_res[i] // shape_x[i]) for i in range(len(shape_x))]
        lb, ub = lb.repeat(*repeat), ub.repeat(*repeat)
        repeat = repeat[:1] + [1] + repeat[1:]
        lw, uw = lw.repeat(*repeat), uw.repeat(*repeat)
        return lw, lb, uw, ub

class BoundReshape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundReshape, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

    def forward(self, x, shape):
        self.input_shape, self.shape = x.size()[1:], shape[1:]
        return x.reshape(list(shape))

    def bound_backward(self, last_lA, last_uA, x, shape):
        def _bound_oneside(A):
            if A is None:
                return None
            return A.reshape(A.shape[0], A.shape[1], *self.input_shape)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, shape):
        batch_size = x.lw.shape[0]
        lw = x.lw.reshape(batch_size, dim_in, *self.shape)
        uw = x.uw.reshape(batch_size, dim_in, *self.shape)
        lb = x.lb.reshape(batch_size, *self.shape)
        ub = x.ub.reshape(batch_size, *self.shape)
        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, *v, norm=None):
        return Interval.make_interval(v[0][0].reshape(*v[1][0]), v[0][1].reshape(*v[1][0]), v[0])


class BoundLinear(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        # Gemm:
        # A = A if transA == 0 else A.T
        # B = B if transB == 0 else B.T
        # C = C if C is not None else np.array(0)
        # Y = alpha * np.dot(A, B) + beta * C
        # return Y

        super(BoundLinear, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

        if attr is not None:
            # assumption: using it as a linear layer now
            assert (not ('transA' in attr))
            assert (attr['transB'] == 1)
            assert (attr['alpha'] == 1.0)
            assert (attr['beta'] == 1.0)

    def forward(self, x, w, b=None):
        self.input_shape = self.x_shape = x.shape
        self.y_shape = w.t().shape
        res = x.matmul(w.t())
        if b is not None:
            res += b
        return res

    def bound_backward(self, last_lA, last_uA, *x):
        assert len(x) == 2 or len(x) == 3
        has_bias = len(x) == 3
        # x[0]: input node, x[1]: weight, x[2]: bias
        # print(x[0], x[0].node_perturbed, x[1], x[1].node_perturbed)
        lA_y = uA_y = lA_bias = uA_bias = None
        lbias = ubias = 0

        # Case #1: No weight/bias perturbation, only perturbation on input.
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            # If last_lA and last_uA are indentity matrices.
            if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                # Use this layer's W as the next bound matrices. Duplicate the batch dimension. Other dimensions are kept 1.
                # Not perturbed, so we can use either lower or upper.
                lA_x = uA_x = x[1].lower.unsqueeze(0).repeat([last_lA.shape[0]] + [1] * len(x[1].lower.shape))
                # Bias will be directly added to output.
                if has_bias:
                    lbias = ubias = x[2].lower.unsqueeze(0).repeat(last_lA.shape[0], 1)
            else:
                def _bound_oneside(last_A):
                    if last_A is None:
                        return None, 0
                    # Just multiply this layer's weight into bound matrices, and produce biases.
                    next_A = last_A.matmul(x[1].lower)
                    sum_bias = last_A.matmul(x[2].lower) if has_bias else 0.0
                    return next_A, sum_bias
                lA_x, lbias = _bound_oneside(last_lA)
                uA_x, ubias = _bound_oneside(last_uA)

        # Case #2: weight is perturbed. bias may or may not be perturbed.
        elif self.is_input_perturbed(1):
            # Obtain relaxations for matrix multiplication.
            [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias = self.bound_backward_with_weight(last_lA, last_uA, x[0], x[1])
            if has_bias:
                if x[2].perturbation is not None:
                    # Bias is also perturbed. Since bias is directly added to the output, in backward mode it is treated
                    # as an input with last_lA and last_uA as associated bounds matrices.
                    # It's okay if last_lA or last_uA is eyeC, as it will be handled in the perturbation object.
                    lA_bias = last_lA
                    uA_bias = last_uA
                else:
                    # Bias not perturbed, so directly adding the bias of this layer to the final bound bias term.
                    if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                        # Bias will be directly added to output.
                        lbias += x[2].lower.unsqueeze(0).repeat(last_lA.shape[0], 1)
                        ubias += x[2].lower.unsqueeze(0).repeat(last_uA.shape[0], 1)
                    else:
                        if last_lA is not None:
                            lbias += last_lA.matmul(x[2].lower)
                        if last_uA is not None:
                            ubias += last_uA.matmul(x[2].lower)
            # If not has_bias, no need to compute lA_bias and uA_bias

        # Case 3: Only bias is perturbed, weight is not perturbed.
        elif not self.is_input_perturbed(1) and has_bias and self.is_input_perturbed(2):
            if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                # Use this layer's W as the next bound matrices. Duplicate the batch dimension. Other dimensions are kept 1.
                lA_x = uA_x = x[1].lower.unsqueeze(0).repeat([last_lA.shape[0]] + [1] * len(x[1].lower.shape))
            else:
                lA_x = last_lA.matmul(x[1].lower)
                uA_x = last_uA.matmul(x[1].lower)
            # It's okay if last_lA or last_uA is eyeC, as it will be handled in the perturbation object.
            lA_bias = last_lA
            uA_bias = last_uA

        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def _reshape(self, x_l, x_u, y_l, y_u):
        x_shape, y_shape = self.x_shape, self.y_shape

        # (x_1, x_2, ..., x_{n-1}, y_2, x_n)
        x_l = x_l.unsqueeze(-2)
        x_u = x_u.unsqueeze(-2)       

        if len(x_shape) == len(y_shape):
            # (x_1, x_2, ..., x_{n-1}, y_n, y_{n-1})
            shape = x_shape[:-1] + (y_shape[-1], y_shape[-2])
            y_l = y_l.unsqueeze(-3)
            y_u = y_u.unsqueeze(-3)
        elif len(y_shape) == 2:
            # (x_1, x_2, ..., x_{n-1}, y_2, y_1)
            shape = x_shape[:-1] + y_shape[1:] + y_shape[:1]
            y_l = y_l.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
            y_u = y_u.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
        return x_l, x_u, y_l, y_u

    def _relax(self, x, y):
        return BoundMul.get_bound_mul(*self._reshape(x.lower, x.upper, y.lower, y.upper))

    def bound_backward_with_weight(self, last_lA, last_uA, x, y):
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self._relax(x, y)

        alpha_l, alpha_u = alpha_l.unsqueeze(1), alpha_u.unsqueeze(1)
        beta_l, beta_u = beta_l.unsqueeze(1), beta_u.unsqueeze(1)
        self.x_shape = x.lower.size()
        self.y_shape = y.lower.size()
        gamma_l = torch.sum(gamma_l, dim=-1).reshape(self.x_shape[0], -1, 1)
        gamma_u = torch.sum(gamma_u, dim=-1).reshape(self.x_shape[0], -1, 1)

        if len(x.default_shape) != 2 and len(x.default_shape) == len(y.default_shape):
            # assert (self.x_shape[0] == self.y_shape[0] == 1)
            dim_y = [-3]
        elif len(y.default_shape) == 2:
            dim_y = list(range(2, 2 + len(self.x_shape) - 2))
        else:
            raise NotImplementedError

        uA_x = lA_x = uA_y = lA_y = None
        lbias = ubias = 0
        if last_uA is not None:
            if isinstance(last_uA, eyeC):
                # last_uA has size (batch, spec, output)
                # alpha_u has size (batch, spec, output, input)
                # However the batch and spec dimension might be 1 due to broadcast.
                uA_x = alpha_u.squeeze(1)
                # uA_x has size (batch, spec, input). Broadcast the batch if necessary.
                uA_x = uA_x.repeat(last_uA.shape[0] // uA_x.size(0), 1, 1)
                # uA_y has size (batch, spec, output, input), but spec and input dimension might be 1 due to broadcast.
                uA_y = beta_u * torch.eye(last_uA.shape[2], device=last_uA.device).view((1, last_uA.shape[2], last_uA.shape[2]) + (1,) * len(dim_y) + (1,))
                if len(dim_y) != 0:
                    uA_y = torch.sum(beta_u, dim=dim_y)
                ubias = gamma_u
            else:
                # last_uA has size (batch, spec, output)
                last_uA_pos = last_uA.clamp(min=0).unsqueeze(-1)
                last_uA_neg = last_uA.clamp(max=0).unsqueeze(-1)
                # alpha_u has size (batch, spec, output, input)
                # uA_x has size (batch, spec, input).
                # uA_x = torch.sum(last_uA_pos * alpha_u + last_uA_neg * alpha_l, dim=-2)
                uA_x = (alpha_u.transpose(-1, -2).matmul(last_uA_pos) + alpha_l.transpose(-1, -2).matmul(last_uA_neg)).squeeze(-1)
                # beta_u has size (batch, spec, output, input)
                # uA_y is for weight matrix, with parameter size (output, input)
                # uA_y has size (batch, spec, output, input). This is an element-wise multiplication.
                uA_y = last_uA_pos * beta_u + last_uA_neg * beta_l
                if len(dim_y) != 0:
                    uA_y = torch.sum(uA_y, dim=dim_y)
                # last_uA has size (batch, spec, output)
                _last_uA_pos = last_uA_pos.reshape(last_uA.shape[0], last_uA.shape[1], -1)
                _last_uA_neg = last_uA_neg.reshape(last_uA.shape[0], last_uA.shape[1], -1)
                # gamma_u has size (batch, output, 1)
                # ubias has size (batch, spec, 1)
                ubias = _last_uA_pos.matmul(gamma_u) + _last_uA_neg.matmul(gamma_l)
            ubias = ubias.squeeze(-1)

        if last_lA is not None:
            if isinstance(last_lA, eyeC):
                lA_x = alpha_l.squeeze(1)
                lA_x = lA_x.repeat(last_lA.shape[0] // lA_x.size(0), 1, 1)
                lA_y = beta_l * torch.eye(last_lA.shape[2], device=last_lA.device).view((1, last_lA.shape[2], last_lA.shape[2]) + (1,) * len(dim_y) + (1,))
                if len(dim_y) != 0:
                    lA_y = torch.sum(lA_y, dim=dim_y)
                lbias = gamma_l
            else:
                last_lA_pos = last_lA.clamp(min=0).unsqueeze(-1)
                last_lA_neg = last_lA.clamp(max=0).unsqueeze(-1)
                lA_x = (alpha_l.transpose(-1, -2).matmul(last_lA_pos) + alpha_u.transpose(-1, -2).matmul(last_lA_neg)).squeeze(-1)
                lA_y = last_lA_pos * beta_l + last_lA_neg * beta_u
                if len(dim_y) != 0:
                    lA_y = torch.sum(lA_y, dim=dim_y)
                _last_lA_pos = last_lA_pos.reshape(last_lA.shape[0], last_lA.shape[1], -1)
                _last_lA_neg = last_lA_neg.reshape(last_lA.shape[0], last_lA.shape[1], -1)
                lbias = _last_lA_pos.matmul(gamma_l) + _last_lA_neg.matmul(gamma_u)
            lbias = lbias.squeeze(-1)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    @staticmethod
    def _propogate_Linf(h_L, h_U, w):
        mid = (h_L + h_U) / 2
        diff = (h_U - h_L) / 2
        w_abs = w.abs()
        if len(mid.shape) == 2 and len(w.shape) == 3:
            center = torch.bmm(mid.unsqueeze(1), w.transpose(-1, -2)).squeeze(1)
            deviation = torch.bmm(diff.unsqueeze(1), w_abs.transpose(-1, -2)).squeeze(1)
        else:
            center = mid.matmul(w.transpose(-1, -2))
            deviation = diff.matmul(w_abs.transpose(-1, -2))
        return center, deviation

    def interval_propagate(self, *v, C=None, w=None):
        has_bias = len(v) == 3
        if w is None and self is None:
            # Use C as the weight, no bias.
            w, lb, ub = C, torch.tensor(0., device=C.device), torch.tensor(0., device=C.device)
        else:
            if w is None:
                # No specified weight, use this layer's weight.
                if self.is_input_perturbed(1): # input index 1 is weight.
                    # w is a perturbed tensor. Use IBP with weight perturbation.
                    # C matrix merging not supported.
                    assert C is None
                    l, u = self.interval_propagate_with_weight(*v)
                    if has_bias:
                        return l + v[2][0], u + v[2][1]
                    else:
                        return l, u
                else:
                    # Use weight 
                    w = v[1][0]
            if has_bias:
                lb, ub = v[2]
            else:
                lb = ub = 0.0
                
            if C is not None:
                w = C.matmul(w)
                lb = C.matmul(lb) if not isinstance(lb, float) else lb
                ub = C.matmul(ub) if not isinstance(ub, float) else ub

        # interval_propagate() of the Linear layer may encounter input with different norms.
        norm, eps = Interval.get_perturbation(v[0])

        if norm == np.inf:
            h_L, h_U = v[0]
            center, deviation = BoundLinear._propogate_Linf(h_L, h_U, w)
        else:
            # General Lp norm.
            mid = v[0][0]
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            if len(w.shape) == 3:
                # Extra batch dimension.
                # mid has dimension [batch, input], w has dimension [batch, output, input].
                center = w.matmul(mid.unsqueeze(-1)).squeeze(-1)
            else:
                # mid has dimension [batch, input], w has dimension [output, input].
                center = mid.matmul(w.t())
            deviation = w.norm(dual_norm, dim=-1) * eps

        lower, upper = center - deviation + lb, center + deviation + ub

        return lower, upper

    def interval_propagate_with_weight(self, *v):
        input_norm, input_eps = Interval.get_perturbation(v[0])
        weight_norm, weight_eps = Interval.get_perturbation(v[1])
        self.x_shape = v[0][0].shape
        self.y_shape = v[1][0].shape

        if input_norm == np.inf and weight_norm == np.inf:
            # Both input data and weight are Linf perturbed (with upper and lower bounds).
            # We need a x_l, x_u for each row of weight matrix.
            x_l, x_u = v[0][0].unsqueeze(-2), v[0][1].unsqueeze(-2)
            y_l, y_u = v[1][0].unsqueeze(-3), v[1][1].unsqueeze(-3)
            # Reuse the multiplication bounds and sum over results.
            lower, upper = BoundMul.interval_propagate(*[(x_l, x_u), (y_l, y_u)])
            lower, upper = torch.sum(lower, -1), torch.sum(upper, -1)
            return lower, upper
        elif input_norm == np.inf and weight_norm == 2:
            # This eps is actually the epsilon per row, as only one row is involved for each output element.
            eps = weight_eps
            # Input data and weight are Linf perturbed (with upper and lower bounds).
            h_L, h_U = v[0]
            # First, handle non-perturbed weight with Linf perturbed data.
            center, deviation = BoundLinear._propogate_Linf(h_L, h_U, v[1][0])
            # Compute the maximal L2 norm of data. Size is [batch, 1].
            max_l2 = torch.max(h_L.abs(), h_U.abs()).norm(2, dim=-1).unsqueeze(-1)
            # Add the L2 eps to bounds.
            lb, ub = center - deviation - max_l2 * eps, center + deviation + max_l2 * eps
            return lb, ub
        else:
            raise NotImplementedError("Unsupported perturbation combination: data={}, weight={}".format(input_norm, weight_norm))

    # w: an optional argument which can be utilized by BoundMatMul
    def bound_forward(self, dim_in, x, w=None, b=None):
        has_bias = b is not None

        # Case #1: No weight/bias perturbation, only perturbation on input.
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            if isinstance(w, LinearBound):
                w = w.lower
            if isinstance(b, LinearBound):
                b = b.lower
            w = w.t()
            w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
            lw = x.lw.matmul(w_pos) + x.uw.matmul(w_neg)
            lb = x.lb.matmul(w_pos) + x.ub.matmul(w_neg)
            uw = x.uw.matmul(w_pos) + x.lw.matmul(w_neg)
            ub = x.ub.matmul(w_pos) + x.lb.matmul(w_neg)
            if b is not None:
                lb += b
                ub += b
        # Case #2: weight is perturbed. bias may or may not be perturbed.
        elif self.is_input_perturbed(1):
            res = self.bound_forward_with_weight(dim_in, x, w)
            if has_bias:
                raise NotImplementedError
            lw, lb, uw, ub = res.lw, res.lb, res.uw, res.ub
        # Case 3: Only bias is perturbed, weight is not perturbed.
        elif not self.is_input_perturbed(1) and has_bias and self.is_input_perturbed(2):
            raise NotImplementedError

        return LinearBound(lw, lb, uw, ub, None, None)

    def bound_forward_with_weight(self, dim_in, x, y):
        x_unsqueeze = LinearBound(
            x.lw.unsqueeze(-2), 
            x.lb.unsqueeze(-2), 
            x.uw.unsqueeze(-2), 
            x.ub.unsqueeze(-2), 
            x.lower.unsqueeze(-2), 
            x.upper.unsqueeze(-2), 
        )
        y_unsqueeze = LinearBound(
            y.lw.unsqueeze(-3), 
            y.lb.unsqueeze(-3), 
            y.uw.unsqueeze(-3), 
            y.ub.unsqueeze(-3), 
            y.lower.unsqueeze(-3), 
            y.upper.unsqueeze(-3), 
        )   
        res_mul = BoundMul.bound_forward(dim_in, x_unsqueeze, y_unsqueeze)
        return LinearBound(
            res_mul.lw.sum(dim=-1) if res_mul.lw is not None else None,
            res_mul.lb.sum(dim=-1),
            res_mul.uw.sum(dim=-1) if res_mul.uw is not None else None,
            res_mul.ub.sum(dim=-1),
            None, None
        )

class BoundBatchNorm2d(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device, training):

        self.num_features = inputs[2].param.shape[0]
        self.eps = attr['epsilon']
        self.momentum = 1 - attr['momentum']  # take care!
        self.affine = True
        self.track_running_stats = True
        # self.num_batches_tracked = 0 # not support yet

        super(BoundBatchNorm2d, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

        self.input_name = input_name
        self.output_name = []
        self.name = name
        self.ori_name = ori_name
        self.bounded = False
        self.to(device)

        self.training = training
        self.IBP_rets = None

    def forward(self, x, w, b, m, v):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            exponential_average_factor = self.momentum

            self.current_mean = x.mean([0, 2, 3])  # m.data.clone()
            self.current_var = x.var([0, 2, 3], unbiased=False)  # v.data.clone()
        else:
            self.current_mean = m.data.clone()
            self.current_var = v.data.clone()

        # print(self.current_mean[0])

        output = F.batch_norm(x, m, v, w, b, self.training or not self.track_running_stats, exponential_average_factor, self.eps)


        return output

    def bound_backward(self, last_lA, last_uA, *x):
        # TODO: Weight perturbation not support yet
        # x[0]: input, x[1]: weight, x[2]: bias, x[3]: running_mean, x[4]: running_var
        weight, bias = x[1].param, x[2].param
        current_mean, current_var = self.current_mean.to(weight.device), self.current_var.to(weight.device)

        tmp_bias = bias - current_mean / torch.sqrt(current_var + self.eps) * weight
        tmp_weight = weight / torch.sqrt(current_var + self.eps)

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            next_A = last_A * tmp_weight.view(1, 1, -1, 1, 1)
            sum_bias = (last_A.sum((3, 4)) * tmp_bias).sum(2)
            return next_A, sum_bias

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)

        return [(lA, uA), (None, None), (None, None), (None, None), (None, None)], lbias, ubias

    def interval_propagate(self, *v, eps=None, norm=None, C=None):
        # TODO: Weight perturbation not support yet
        h_L, h_U = v[0]
        weight, bias = v[1][0], v[2][0]
        current_mean, current_var = self.current_mean.to(weight.device), self.current_var.to(weight.device)

        mid = (h_U + h_L) / 2.0
        diff = (h_U - h_L) / 2.0

        tmp_weight = weight / torch.sqrt(current_var + self.eps)
        tmp_weight_abs = tmp_weight.abs()
        tmp_bias = bias - current_mean * tmp_weight

        center = tmp_weight.view(1, -1, 1, 1) * mid + tmp_bias.view(1, -1, 1, 1)
        deviation = tmp_weight_abs.view(1, -1, 1, 1) * diff
        lower = center - deviation
        upper = center + deviation
        # print('bn', h_U.abs().max(), upper.abs().max(), current_var.abs().min())
        return lower, upper

class BoundConv2d(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        assert (attr['pads'][0] == attr['pads'][2])
        assert (attr['pads'][1] == attr['pads'][3])

        super(BoundConv2d, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.dilation = attr['dilations']
        self.groups = attr['group']
        if len(inputs) == 3:
            self.bias_ = True
        else:
            self.bias_ = False
        self.input_name = input_name
        self.output_name = []
        self.name = name
        self.ori_name = ori_name
        self.bounded = False
        self.IBP_rets = None
        self.to(device)

    def forward(self, *x):
        # x[0]: input, x[1]: weight, x[2]: bias if self.bias_
        if self.bias_:
            output = F.conv2d(x[0], x[1], x[2], self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            output = F.conv2d(x[0], x[1], None, self.stride,
                              self.padding, self.dilation, self.groups)
        self.output_shape = output.size()[1:]
        self.input_shape = x[0].size()[1:]
        return output

    def bound_backward(self, last_lA, last_uA, *x):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        lA_y = uA_y = lA_bias = uA_bias = None
        weight = x[1].param

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            shape = last_A.size()
            # when (W−F+2P)%S != 0, construct the output_padding
            output_padding0 = int(self.input_shape[1]) - (int(self.output_shape[1]) - 1) * self.stride[0] + 2 * \
                              self.padding[0] - int(weight.size()[2])
            output_padding1 = int(self.input_shape[2]) - (int(self.output_shape[2]) - 1) * self.stride[1] + 2 * \
                              self.padding[1] - int(weight.size()[3])
            next_A = F.conv_transpose2d(last_A.view(shape[0] * shape[1], *shape[2:]), weight, None,
                                        stride=self.stride, padding=self.padding, dilation=self.dilation,
                                        groups=self.groups, output_padding=(output_padding0, output_padding1))
            next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
            if self.bias_:
                sum_bias = (last_A.sum((3, 4)) * x[2].param).sum(2)
            else:
                sum_bias = 0
            return next_A, sum_bias

        lA_x, lbias = _bound_oneside(last_lA)
        uA_x, ubias = _bound_oneside(last_uA)
        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def interval_propagate(self, *v, C=None):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        norm, eps = Interval.get_perturbation(v[0])

        h_L, h_U = v[0]
        weight = v[1][0]
        if norm == np.inf:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = weight.abs()
            deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            # L2 norm, h_U and h_L are the same.
            mid = h_U
            # TODO: padding
            deviation = torch.mul(weight, weight).sum((1, 2, 3)).sqrt() * eps
            deviation = deviation.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        if self.bias_:
            center = F.conv2d(mid, weight, v[2][0], self.stride, self.padding, self.dilation, self.groups)
        else:
            center = F.conv2d(mid, weight, None, self.stride, self.padding, self.dilation, self.groups)

        upper = center + deviation
        lower = center - deviation
        return lower, upper  # , 0, 0, 0, 0

class BoundMaxPool2d(MaxPool2d):
    def __init__(self, input_name, name, ori_name,
                 prev_layer,
                 kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        raise NotImplementedError

        super(BoundMaxPool2d, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding,
                                             dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
        self.input_name = input_name
        self.output_name = []
        self.name = name
        self.ori_name = ori_name
        self.forward_value = None
        self.bounded = False
        self.IBP_rets = None

    @staticmethod
    def convert(act_layer, prev_layer, input_name, name):
        nl = BoundMaxPool2d(input_name, name, prev_layer, act_layer.kernel_size, act_layer.stride,
                            act_layer.padding, act_layer.dilation)
        return nl

    def bound_backward(self, last_lA, last_uA):
        raise NotImplementedError

    def interval_propagate(self, h_U, h_L, eps=None, norm=None, C=None):
        raise NotImplementedError

class BoundAvgPool2d(AvgPool2d):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        # assumptions: ceil_mode=False, count_include_pad=True
        assert (attr['pads'][0] == attr['pads'][2])
        assert (attr['pads'][1] == attr['pads'][3])
        kernel_size = attr['kernel_shape']
        stride = attr['strides']
        padding = [attr['pads'][0], attr['pads'][1]]
        ceil_mode = False
        count_include_pad = True
        super(BoundAvgPool2d, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding,
                                             ceil_mode=ceil_mode, count_include_pad=count_include_pad)
        self.input_name = input_name
        self.output_name = []
        self.name = name
        self.ori_name = ori_name
        self.forward_value = None
        self.bounded = False
        self.IBP_rets = None
        self.from_input = False


    def forward(self, x):
        self.input_shape = x.size()[1:]
        output = super(BoundAvgPool2d, self).forward(x)
        return output

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            shape = last_A.size()
            # propagate A to the next layer, with batch concatenated together
            next_A = F.interpolate(last_A.view(shape[0] * shape[1], *shape[2:]), scale_factor=self.kernel_size)/(np.prod(self.kernel_size))
            next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
            return next_A, 0

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)
        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v, eps=None, norm=None, C=None):
        h_L, h_U = v[0]
        # shape = h_U.size()
        h_L = super(BoundAvgPool2d, self).forward(h_L)
        h_U = super(BoundAvgPool2d, self).forward(h_U)

        # h_L = h_L.view(shape[0], *h_L.shape[1:])
        # h_U = h_U.view(shape[0], *h_U.shape[1:])
        return h_L, h_U

class BoundGlobalAveragePool(AdaptiveAvgPool2d):
    def __init__(self, input_name, name, ori_name, prev_layer, output_size, output_index):
        raise NotImplementedError

        super(BoundGlobalAveragePool, self).__init__(output_size=output_size)
        self.input_name = input_name
        self.output_name = []
        self.name = name
        self.ori_name = ori_name
        self.forward_value = None
        self.bounded = False
        self.IBP_rets = None

    @staticmethod
    def convert(act_layer, prev_layer, input_name, name):
        nl = BoundGlobalAveragePool(input_name, name, prev_layer, act_layer.output_size)
        return nl

    def bound_backward(self, last_lA, last_uA):
        raise NotImplementedError

    def interval_propagate(self, h_L, h_U, eps=None, norm=None, C=None):
        raise NotImplementedError

class BoundConcat(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundConcat, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.axis = attr['axis']
        self.IBP_rets = None

    def forward(self, *x):  # x is a list of tensors
        self.input_size = [item.shape[self.axis] for item in x]
        return torch.cat(list(x), dim=self.axis)

    def interval_propagate(self, *v, norm=None):
        norms = []
        eps = []
        # Collect perturbation information for all inputs.
        for i, _v in enumerate(v):
            if self.is_input_perturbed(i):
                n, e = Interval.get_perturbation(_v)
                norms.append(n)
                eps.append(e)
            else:
                norms.append(None)
                eps.append(0.0)
        eps = np.array(eps)
        # Supporting two cases: all inputs are Linf norm, or all inputs are L2 norm perturbed.
        # Some inputs can be constants without perturbations.
        all_inf = all(map(lambda x: x is None or x == np.inf, norms))
        all_2 = all(map(lambda x: x is None or x == 2, norms))
        
        h_L = [_v[0] for _v in v]
        h_U = [_v[1] for _v in v]
        if all_inf:
            # Simply returns a tuple. Every subtensor has its own lower and upper bounds.
            return self.forward(*h_L), self.forward(*h_U)
        elif all_2:
            # Sum the L2 norm over all subtensors, and use that value as the new L2 norm.
            # This will be an over-approximation of the original perturbation (we can prove it).
            max_eps = np.sqrt(np.sum(eps * eps))
            # For L2 norm perturbed inputs, lb=ub and for constants lb=ub. Just propagate one object.
            r = self.forward(*h_L)
            ptb = PerturbationLpNorm(norm=2, eps=max_eps)
            return Interval(r, r, ptb)
        else:
            raise RuntimeError("BoundConcat does not support inputs with norm {}".format(norms))

    def bound_backward(self, last_lA, last_uA, *x):
        if self.axis < 0:
            self.axis = len(self.default_shape) + self.axis
        assert (self.axis > 0)

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return torch.split(last_A, self.input_size, dim=self.axis + 1)

        uA = _bound_oneside(last_uA)
        lA = _bound_oneside(last_lA)
        if uA is None:
            return [(lA[i] if lA is not None else None, None) for i in range(len(lA))], 0, 0
        if lA is None:
            return [(None, uA[i] if uA is not None else None) for i in range(len(uA))], 0, 0
        return [(lA[i], uA[i]) for i in range(len(lA))], 0, 0

    def bound_forward(self, dim_in, *x):
        if self.axis < 0:
            self.axis = len(x[0].lb.shape) + self.axis
        assert (self.axis == 0 and not self.from_input or self.from_input)
        lw = torch.cat([item.lw for item in x], dim=self.axis + 1)
        lb = torch.cat([item.lb for item in x], dim=self.axis)
        uw = torch.cat([item.uw for item in x], dim=self.axis + 1)
        ub = torch.cat([item.ub for item in x], dim=self.axis)
        return LinearBound(lw, lb, uw, ub, None, None)

class BoundAdd(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundAdd, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x + y

    def bound_backward(self, last_lA, last_uA, x, y):
        def _bound_oneside(last_A, w):
            if last_A is None:
                return None
            return self._broadcast_backward(last_A, w)
        uA_x = _bound_oneside(last_uA, x)
        uA_y = _bound_oneside(last_uA, y)
        lA_x = _bound_oneside(last_lA, x)
        lA_y = _bound_oneside(last_lA, y)
        return [(lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def bound_forward(self, dim_in, x, y):
        x_lw, x_lb, x_uw, x_ub = self._broadcast_forward(dim_in, x, self.default_shape)
        y_lw, y_lb, y_uw, y_ub = self._broadcast_forward(dim_in, y, self.default_shape)
        lw, lb = x_lw + y_lw, x_lb + y_lb
        uw, ub = x_uw + y_uw, x_ub + y_ub
        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, x, y):
        assert (not isinstance(y, torch.Tensor))
        return x[0] + y[0], x[1] + y[1]

class BoundSub(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundSub, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x - y

    def bound_backward(self, last_lA, last_uA, x, y):
        def _bound_oneside(last_A, w, sign=-1):
            if last_A is None:
                return None
            return self._broadcast_backward(sign*last_A, w)
        uA_x = _bound_oneside(last_uA, x, sign=1)
        uA_y = _bound_oneside(last_uA, y, sign=-1)
        lA_x = _bound_oneside(last_lA, x, sign=1)
        lA_y = _bound_oneside(last_lA, y, sign=-1)
        return [(lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def bound_forward(self, dim_in, x, y):
        x_lw, x_lb, x_uw, x_ub = self._broadcast_forward(dim_in, x, self.default_shape)
        y_lw, y_lb, y_uw, y_ub = self._broadcast_forward(dim_in, y, self.default_shape)
        lw, lb = x_lw - y_uw, x_lb - y_ub
        uw, ub = x_uw - y_lw, x_ub - y_lb
        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, x, y):
        return x[0] - y[1], x[1] - y[0]

class BoundPad(ConstantPad2d):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        if len(attr) == 1:
            padding = [0, 0, 0, 0]
            value = 0.0
        else:
            padding = attr['pads'][2:4] + attr['pads'][6:8]
            value = attr['value']
        assert padding == [0, 0, 0, 0]
        super(BoundPad, self).__init__(padding=padding, value=value)
        self.input_name = input_name
        self.output_name = []
        self.name = name
        # FIXME: move all of those to the base class, avoid doing these once again!
        self.ori_name = ori_name
        self.forward_value = None
        self.bounded = False
        self.IBP_rets = None
        self.from_input = False

    def forward(self, *input):
        return super(BoundPad, self).forward(input[0])

    def bound_backward(self, last_lA, last_uA, x, pad=None):
        # TODO
        if pad:
            return [(last_lA, last_uA), (None, None)], 0, 0
        else:
            return [(last_lA, last_uA)], 0, 0

    def interval_propagate(self, *v, norm=None):
        h_L, h_U = v[0]
        return super(BoundPad, self).forward(h_L), super(BoundPad, self).forward(h_U)

class BoundActivation(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundActivation, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.nonlinear = True
        self.relaxed = False

    def _init_linear(self, input):
        self.mask_pos = torch.gt(input.lower, 0).to(torch.float)
        self.mask_neg = torch.lt(input.upper, 0).to(torch.float)
        self.mask_both = 1 - self.mask_pos - self.mask_neg

        self.lw = torch.zeros(input.lower.shape, device=self.device)
        self.lb = self.lw.clone()
        self.uw = self.lw.clone()
        self.ub = self.lw.clone()

    def _add_linear(self, mask, type, k, x0, y0):
        if mask is None:
            mask = 1
        if type == 'lower':
            w_out, b_out = self.lw, self.lb
        else:
            w_out, b_out = self.uw, self.ub

        w_out += mask * k
        b_out += mask * (-x0 * k + y0)

        # linear relaxation for nonlinear functions

    def bound_relax(self, input):
        raise NotImplementedError

    def bound_backward(self, last_lA, last_uA, input):
        if not self.relaxed:
            self._init_linear(input)
            self.bound_relax(input)

        def _bound_oneside(last_A, sign=-1):
            if last_A is None:
                return None, 0

            if input.from_input:
                if sign == -1:
                    _A = last_A.clamp(min=0) * self.lw.unsqueeze(1) + last_A.clamp(max=0) * self.uw.unsqueeze(1)
                    _bias = last_A.clamp(min=0) * self.lb.unsqueeze(1) + last_A.clamp(max=0) * self.ub.unsqueeze(1)
                elif sign == 1:
                    _A = last_A.clamp(min=0) * self.uw.unsqueeze(1) + last_A.clamp(max=0) * self.lw.unsqueeze(1)
                    _bias = last_A.clamp(min=0) * self.ub.unsqueeze(1) + last_A.clamp(max=0) * self.lb.unsqueeze(1)
                while len(_bias.shape) > 2:
                    _bias = torch.sum(_bias, dim=-1)
            else:
                mask = torch.gt(last_A, 0.).to(torch.float)
                if sign == -1:
                    _A = last_A * (mask * self.lw.unsqueeze(0).unsqueeze(1) +
                                   (1 - mask) * self.uw.unsqueeze(0).unsqueeze(1))
                    _bias = last_A * (mask * self.lb.unsqueeze(0).unsqueeze(1) +
                                       (1 - mask) * self.ub.unsqueeze(0).unsqueeze(1))
                elif sign == 1:
                    _A = last_A * (mask * self.uw.unsqueeze(0).unsqueeze(1) +
                                   (1 - mask) * self.lw.unsqueeze(0).unsqueeze(1))
                    _bias = last_A * (mask * self.ub.unsqueeze(0).unsqueeze(1) +
                                      (1 - mask) * self.lb.unsqueeze(0).unsqueeze(1))
                while len(_bias.shape) > 2:
                    _bias = torch.sum(_bias, dim=-1)

            return _A, _bias

        lA, lbias = _bound_oneside(last_lA, sign=-1)
        uA, ubias = _bound_oneside(last_uA, sign=+1)

        return [(lA, uA)], lbias, ubias

    def bound_forward(self, dim_in, input):
        if not self.relaxed:
            self._init_linear(input)
            self.bound_relax(input)

        if len(self.lw.shape) > 0:
            if input.lw is not None:
                lw = self.lw.unsqueeze(1).clamp(min=0) * input.lw + \
                    self.lw.unsqueeze(1).clamp(max=0) * input.uw
                uw = self.uw.unsqueeze(1).clamp(max=0) * input.lw + \
                    self.uw.unsqueeze(1).clamp(min=0) * input.uw
            else:
                lw = uw = None
        else:
            if input.lw is not None:
                lw = self.lw.unsqueeze(0).clamp(min=0) * input.lw + \
                    self.lw.unsqueeze(0).clamp(max=0) * input.uw
                uw = self.uw.unsqueeze(0).clamp(min=0) * input.lw + \
                    self.uw.unsqueeze(0).clamp(max=0) * input.uw 
            else:
                lw = uw = None
        lb = self.lw.clamp(min=0) * input.lb + self.lw.clamp(max=0) * input.ub + self.lb
        ub = self.uw.clamp(max=0) * input.lb + self.uw.clamp(min=0) * input.ub + self.ub            

        return LinearBound(lw, lb, uw, ub, None, None)

class BoundLeakyRelu(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device, bound_opts=None):
        super(BoundLeakyRelu, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.nonlinear = True

        if bound_opts is not None and 'relu' in bound_opts:
            self.bound_opts = bound_opts['relu']
        else:
            self.bound_opts = None

        self.alpha = bound_opts['leaky_relu'] if bound_opts is not None and 'leaky_relu' in bound_opts else 0.05
    
    def forward(self, input):
        return F.leaky_relu(input, negative_slope=self.alpha)
    
    def bound_relax(self, input):
        epsilon = 1e-12
        m = torch.min((input.lower + input.upper) / 2, input.lower + 0.99)
        self._add_linear(mask=self.mask_neg, type='lower',
                         k=self.alpha, x0=0, y0=0)
        self._add_linear(mask=self.mask_neg, type='upper',
                         k=self.alpha, x0=0, y0=0)
        self._add_linear(mask=self.mask_pos, type='lower',
                         k=torch.ones_like(input.lower), x0=0, y0=0)
        self._add_linear(mask=self.mask_pos, type='upper',
                         k=torch.ones_like(input.lower), x0=0, y0=0)
        self._add_linear(mask=self.mask_both, type='lower',
                         k=torch.gt(torch.abs(input.upper), torch.abs(input.lower)).to(torch.float), x0=0., y0=0.)
        self._add_linear(mask=self.mask_both, type='upper',
                         k=input.upper / (input.upper - input.lower + 1e-12), x0=input.lower, y0=0)

    def bound_backward(self, last_lA, last_uA, input=None):
        if input is not None:
            lb_r = input.lower.clamp(max=0)
            ub_r = input.upper.clamp(min=0)
        else:
            lb_r = self.lower.clamp(max=0)
            ub_r = self.upper.clamp(min=0)
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        upper_d = (ub_r - self.alpha * lb_r) / (ub_r - lb_r)
        upper_b = - lb_r * upper_d + self.alpha * lb_r
        # upper_d = upper_d.unsqueeze(1)

        # ub_r = torch.max(ub_r, lb_r + 1e-8)
        # upper_d = ub_r / (ub_r - lb_r)
        # # if upper_d.requires_grad:
        # #     upper_d = upper_d.clamp(min=1e-6)
        # upper_b = - lb_r * upper_d

        if self.bound_opts == "same-slope":
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.bound_opts == "zero-lb":
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float() + (upper_d < 1.0).float() * self.alpha
        elif self.bound_opts == "one-lb":
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float() + (upper_d <= 0.0).float() * self.alpha
        else:
            lower_d = (upper_d > 0.5).float() + (upper_d <= 0.5).float() * self.alpha

        upper_d = upper_d.unsqueeze(1)
        lower_d = lower_d.unsqueeze(1)
        # Choose upper or lower bounds based on the sign of last_A
        uA = lA = None
        ubias = lbias = 0
        if last_uA is not None:
            neg_uA = last_uA.clamp(max=0)
            pos_uA = last_uA.clamp(min=0)
            uA = upper_d * pos_uA + lower_d * neg_uA
            mult_uA = pos_uA.view(last_uA.shape[0], last_uA.shape[1], -1)
            ubias = mult_uA.matmul(upper_b.view(upper_b.shape[0], -1, 1)).squeeze(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            lA = upper_d * neg_lA + lower_d * pos_lA
            mult_lA = neg_lA.view(last_lA.shape[0], last_lA.shape[1], -1)
            lbias = mult_lA.matmul(upper_b.view(upper_b.shape[0], -1, 1)).squeeze(-1)
        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v, eps=None, norm=None):
        h_L, h_U = v[0][0], v[0][1]
        guard_eps = 1e-5
        self.upper = h_U
        self.lower = h_L
        return F.leaky_relu(h_L, self.alpha), F.leaky_relu(h_U, self.alpha)
    
class BoundReLU(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device, bound_opts=None):
        super(BoundReLU, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.nonlinear = True
        if bound_opts is not None and 'relu' in bound_opts:
            self.bound_opts = bound_opts['relu']
        else:
            self.bound_opts = None

    def forward(self, input):
        return F.relu(input)

    # linear relaxation for nonlinear functions
    def bound_relax(self, input):
        # FIXME bound_opts is not considered 
        epsilon = 1e-12
        m = torch.min((input.lower + input.upper) / 2, input.lower + 0.99)
        self._add_linear(mask=self.mask_neg, type='lower',
                         k=input.lower * 0, x0=0, y0=0)
        self._add_linear(mask=self.mask_neg, type='upper',
                         k=input.lower * 0, x0=0, y0=0)
        self._add_linear(mask=self.mask_pos, type='lower',
                         k=torch.ones_like(input.lower), x0=0, y0=0)
        self._add_linear(mask=self.mask_pos, type='upper',
                         k=torch.ones_like(input.lower), x0=0, y0=0)
        # adaptive
        self._add_linear(mask=self.mask_both, type='lower',
                         k=torch.gt(torch.abs(input.upper), torch.abs(input.lower)).to(torch.float), x0=0., y0=0.)
        self._add_linear(mask=self.mask_both, type='upper',
                         k=input.upper / (input.upper - input.lower + 1e-12), x0=input.lower, y0=0)

    def bound_backward(self, last_lA, last_uA, input=None):
        if input is not None:
            lb_r = input.lower.clamp(max=0)
            ub_r = input.upper.clamp(min=0)
        else:
            lb_r = self.lower.clamp(max=0)    
            ub_r = self.upper.clamp(min=0)
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        upper_d = ub_r / (ub_r - lb_r)
        upper_b = - lb_r * upper_d

        use_lower_b = False

        if self.bound_opts == "same-slope":
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.bound_opts == "zero-lb":
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float()
        elif self.bound_opts == "one-lb":
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float()
        elif self.bound_opts == "reversed-adaptive":
            lower_d = (upper_d < 0.5).float()
        else:
            lower_d = (upper_d > 0.5).float()

        upper_d = upper_d.unsqueeze(1)
        lower_d = lower_d.unsqueeze(1)

        # Choose upper or lower bounds based on the sign of last_A
        uA = lA = None
        ubias = lbias = 0
        if last_uA is not None:
            neg_uA = last_uA.clamp(max=0)
            pos_uA = last_uA.clamp(min=0)
            uA = upper_d * pos_uA + lower_d * neg_uA
            mult_uA = pos_uA.view(last_uA.shape[0], last_uA.shape[1], -1)
            ubias = mult_uA.matmul(upper_b.view(upper_b.shape[0], -1, 1)).squeeze(-1)
            if use_lower_b:
                mult_uA = neg_uA.view(last_uA.shape[0], last_uA.shape[1], -1)
                ubias = ubias + mult_uA.matmul(lower_b.view(lower_b.shape[0], -1, 1)).squeeze(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            lA = upper_d * neg_lA + lower_d * pos_lA
            mult_lA = neg_lA.view(last_lA.shape[0], last_lA.shape[1], -1)
            lbias = mult_lA.matmul(upper_b.view(upper_b.shape[0], -1, 1)).squeeze(-1)
            if use_lower_b:
                mult_lA = pos_lA.view(last_lA.shape[0], last_lA.shape[1], -1)
                lbias = lbias + mult_lA.matmul(lower_b.view(lower_b.shape[0], -1, 1)).squeeze(-1)

        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v, eps=None, norm=None):
        h_L, h_U = v[0][0], v[0][1]
        guard_eps = 1e-5
        self.upper = h_U
        self.lower = h_L
        return F.relu(h_L), F.relu(h_U)

class BoundTanh(Tanh, BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundTanh, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

        self._precompute()

    def dtanh(self, x):
        # to avoid bp error when cosh is too large
        x_limit = torch.tensor(10., device=x.device)
        mask = torch.lt(torch.abs(x), x_limit).float()
        cosh = torch.cosh(mask * x + 1 - mask)
        return mask * (1. / cosh.pow(2))        

    def _precompute(self):
        max_iter = 10
        epsilon = 1e-12
        filename = 'tmp/tanh.pkl'

        if not os.path.exists('tmp'):
            os.mkdir('tmp')

        if hasattr(self, 'd_lower'):
            return

        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.d_lower, self.d_upper = pickle.load(file)
                self.d_lower = torch.tensor(self.d_lower, device=self.device)
                self.d_upper = torch.tensor(self.d_upper, device=self.device)
            return

        self.d_lower, self.d_upper = [], []

        logger.info('Precomputing bounds for Tanh')

        # precompute lower bounds
        lower = torch.tensor(-1.)
        for _upper in range(0, 1005):
            upper = torch.tensor(_upper * 0.01)
            tanh_upper = torch.tanh(upper)
            diff = lambda d: (tanh_upper - torch.tanh(d)) / (upper - d + epsilon) - self.dtanh(d)
            d = lower
            _l = lower
            _u = lower * 0
            for t in range(max_iter):
                v = diff(d)
                mask_p = torch.gt(v, 0).to(torch.float)
                _l = d * mask_p + _l * (1 - mask_p)
                _u = d * (1 - mask_p) + _u * mask_p
                d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
            self.d_lower.append(d)

        upper = torch.tensor(1.)
        for _lower in range(0, 1005):
            lower = torch.tensor(_lower * -0.01)
            tanh_lower = torch.tanh(lower)
            diff = lambda d: (torch.tanh(d) - tanh_lower) / (d - lower + epsilon) - self.dtanh(d)
            d = upper
            _l = lower * 0
            _u = upper
            for t in range(max_iter):
                v = diff(d)
                mask_p = torch.gt(v, 0).to(torch.float)
                _l = d * (1 - mask_p) + _l * mask_p
                _u = d * mask_p + _u * (1 - mask_p)
                d = (d + _u) / 2 * (1 - mask_p) + (d + _l) / 2 * mask_p
            self.d_upper.append(d)

        with open(filename, 'wb') as file:
            pickle.dump((self.d_lower, self.d_upper), file)

        self.d_lower = torch.tensor(self.d_lower, device=self.device)
        self.d_upper = torch.tensor(self.d_upper, device=self.device)

    def forward(self, input):
        output = super(BoundTanh, self).forward(input)
        return output

    # linear relaxation for nonlinear functions
    def bound_relax(self, input):
        epsilon = 1e-12

        # lower bound for negative
        m = (input.lower + input.upper) / 2
        y_l, y_m, y_u = torch.tanh(input.lower), torch.tanh(m), torch.tanh(input.upper)
        k = self.dtanh(m)
        self._add_linear(mask=self.mask_neg, type='lower', k=k, x0=m, y0=y_m)
        # upper bound for positive
        self._add_linear(mask=self.mask_pos, type='upper', k=k, x0=m, y0=y_m)

        # upper bound for negative
        k = (y_u - y_l) / (input.upper - input.lower + epsilon)
        self._add_linear(mask=self.mask_neg, type='upper', k=k, x0=input.lower, y0=y_l)
        # lower bound for positive
        self._add_linear(mask=self.mask_pos, type='lower', k=k, x0=input.lower, y0=y_l)

        # bounds for both
        max_iter = 5

        # lower, upper = input.lower.detach(), input.upper.detach()
        lower, upper = input.lower, input.upper

        if torch.min(lower) > -10. and torch.max(upper) < 10.:
            d = torch.index_select(
                self.d_lower, 0,
                torch.max(
                    torch.zeros_like(upper, dtype=torch.long).reshape(-1),
                    (upper / 0.01).to(torch.long).reshape(-1)
                ) + 1
            ).reshape(upper.shape)
            k = (torch.tanh(d) - y_u) / (d - upper + epsilon)
            self._add_linear(mask=self.mask_both, type='lower', k=k, x0=d, y0=torch.tanh(d))

            d = torch.index_select(
                self.d_upper, 0,
                torch.max(
                    torch.zeros_like(lower, dtype=torch.long).reshape(-1),
                    (lower / -0.01).to(torch.long).reshape(-1)
                ) + 1
            ).reshape(upper.shape)
            k = (torch.tanh(d) - y_l) / (d - lower + epsilon)
            self._add_linear(mask=self.mask_both, type='upper', k=k, x0=d, y0=torch.tanh(d))
        else:
            with torch.no_grad():
                # lower bound for both
                tanh_upper = torch.tanh(upper)
                diff = lambda d: (tanh_upper - torch.tanh(d)) / (upper - d + epsilon) - self.dtanh(d)
                d = lower / 2
                _l = lower
                _u = torch.zeros(lower.shape, device=self.device)
                for t in range(max_iter):
                    v = diff(d)
                    mask_p = torch.gt(v, 0).to(torch.float)
                    _l = d * mask_p + _l * (1 - mask_p)
                    _u = d * (1 - mask_p) + _u * mask_p
                    d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
                k = (torch.tanh(d) - torch.tanh(upper)) / (d - upper + epsilon)
            self._add_linear(mask=self.mask_both, type='lower', k=k, x0=d, y0=torch.tanh(d))

            # upper bound for both
            with torch.no_grad():
                tanh_lower = torch.tanh(lower)
                diff = lambda d: (torch.tanh(d) - tanh_lower) / (d - lower + epsilon) - self.dtanh(d)
                d = upper / 2
                _l = torch.zeros(lower.shape, device=self.device)
                _u = upper
                for t in range(max_iter):
                    v = diff(d)
                    mask_p = torch.gt(v, 0).to(torch.float)
                    _l = d * (1 - mask_p) + _l * mask_p
                    _u = d * mask_p + _u * (1 - mask_p)
                    d = (d + _u) / 2 * (1 - mask_p) + (d + _l) / 2 * mask_p
                k = (torch.tanh(d) - torch.tanh(lower)) / (d - lower + epsilon)
            self._add_linear(mask=self.mask_both, type='upper', k=k, x0=d, y0=torch.tanh(d))

class BoundSigmoid(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundSigmoid, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

        self._precompute()

    def forward(self, input):
        return torch.sigmoid(input)

    def _precompute(self):
        dsigmoid = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        max_iter = 10
        epsilon = 1e-12
        filename = 'tmp/sigmoid.pkl'

        if not os.path.exists('tmp'):
            os.mkdir('tmp')

        if hasattr(self, 'd_lower'):
            return

        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.d_lower, self.d_upper = pickle.load(file)
                self.d_lower = torch.tensor(self.d_lower).to(self.device)
                self.d_upper = torch.tensor(self.d_upper).to(self.device)
            return

        self.d_lower, self.d_upper = [], []

        logger.info('Precomputing bounds for Sigmoid')

        # precompute lower bounds
        lower = torch.tensor(-10.)
        for _upper in range(0, 1005):
            upper = torch.tensor(_upper * 0.01)
            sigmoid_upper = torch.sigmoid(upper)
            diff = lambda d: (sigmoid_upper - torch.sigmoid(d)) / (upper - d + epsilon) - dsigmoid(d)
            d = lower
            _l = lower
            _u = lower * 0
            for t in range(max_iter):
                v = diff(d)
                mask_p = torch.gt(v, 0).to(torch.float)
                _l = d * mask_p + _l * (1 - mask_p)
                _u = d * (1 - mask_p) + _u * mask_p
                d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
            self.d_lower.append(d)

        upper = torch.tensor(10.)
        for _lower in range(0, 1005):
            lower = torch.tensor(_lower * -0.01)
            sigmoid_lower = torch.sigmoid(lower)
            diff = lambda d: (torch.sigmoid(d) - sigmoid_lower) / (d - lower + epsilon) - dsigmoid(d)
            d = upper / 2
            _l = lower * 0
            _u = upper
            for t in range(max_iter):
                v = diff(d)
                mask_p = torch.gt(v, 0).to(torch.float)
                _l = d * (1 - mask_p) + _l * mask_p
                _u = d * mask_p + _u * (1 - mask_p)
                d = (d + _u) / 2 * (1 - mask_p) + (d + _l) / 2 * mask_p
            self.d_upper.append(d)

        with open(filename, 'wb') as file:
            pickle.dump((self.d_lower, self.d_upper), file)

        self.d_lower = torch.tensor(self.d_lower, device=self.device)
        self.d_upper = torch.tensor(self.d_upper, device=self.device)

        # TODO: can merge this function with tanh's

    def bound_relax(self, input):
        def dsigmoid(x):
            return torch.sigmoid(x) * (1 - torch.sigmoid(x))

        epsilon = 1e-12

        # lower bound for negative
        m = (input.lower + input.upper) / 2
        k = dsigmoid(m)
        y_l, y_u, y_m = self.forward(input.lower), self.forward(input.upper), self.forward(m)
        self._add_linear(mask=self.mask_neg, type='lower', k=k, x0=m, y0=y_m)
        # upper bound for positive
        self._add_linear(mask=self.mask_pos, type='upper', k=k, x0=m, y0=y_m)

        # upper bound for negative
        k = (y_u - y_l) / (input.upper - input.lower + epsilon)
        self._add_linear(mask=self.mask_neg, type='upper', k=k, x0=input.lower, y0=y_l)
        # lower bound for positive
        self._add_linear(mask=self.mask_pos, type='lower', k=k, x0=input.lower, y0=y_l)

        # bounds for both
        max_iter = 5

        #lower, upper = input.lower.detach(), input.upper.detach()
        lower, upper = input.lower, input.upper

        if torch.min(lower) > -10. and torch.max(upper) < 10.:
            d = torch.index_select(
                self.d_lower, 0,
                torch.max(
                    torch.zeros_like(upper, dtype=torch.long).reshape(-1),
                    (upper / 0.01).to(torch.long).reshape(-1)
                ) + 1
            ).reshape(upper.shape)
            y_d = torch.sigmoid(d)
            k = (y_d - y_u) / (d - upper + epsilon)
            self._add_linear(mask=self.mask_both, type='lower', k=k, x0=d, y0=y_d)

            d = torch.index_select(
                self.d_upper, 0,
                torch.max(
                    torch.zeros_like(lower, dtype=torch.long).reshape(-1),
                    (lower / -0.01).to(torch.long).reshape(-1)
                ) + 1
            ).reshape(upper.shape)
            y_d = torch.sigmoid(d)
            k = (y_d - y_l) / (d - lower + epsilon)
            self._add_linear(mask=self.mask_both, type='upper', k=k, x0=d, y0=y_d)
        else:
            # lower bound for both
            with torch.no_grad():
                diff = lambda d: (y_u - self.forward(d)) / (upper - d + epsilon) - dsigmoid(d)
                d = lower / 2
                _l = lower
                _u = torch.zeros(lower.shape, device=self.device)
                for t in range(max_iter):
                    v = diff(d)
                    mask_p = torch.gt(v, 0).to(torch.float)
                    _l = d * mask_p + _l * (1 - mask_p)
                    _u = d * (1 - mask_p) + _u * mask_p
                    d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
                k = (self.forward(d) - y_u) / (d - upper + epsilon)
            self._add_linear(mask=self.mask_both, type='lower', k=k, x0=d, y0=self.forward(d))

            # upper bound for both
            with torch.no_grad():
                diff = lambda d: (self.forward(d) - y_l) / (d - lower + epsilon) - dsigmoid(d)
                d = upper / 2
                _l = torch.zeros(lower.shape, device=self.device)
                _u = upper
                for t in range(max_iter):
                    v = diff(d)
                    mask_p = torch.gt(v, 0).to(torch.float)
                    _l = d * (1 - mask_p) + _l * mask_p
                    _u = d * mask_p + _u * (1 - mask_p)
                    d = (d + _u) / 2 * (1 - mask_p) + (d + _l) / 2 * mask_p
                k = (self.forward(d) - y_l) / (d - lower + epsilon)
            self._add_linear(mask=self.mask_both, type='upper', k=k, x0=d, y0=self.forward(d))

class BoundExp(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device, bound_opts):
        super(BoundExp, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        if bound_opts is not None and 'exp' in bound_opts:
            self.bound_opts = bound_opts['exp']
        else:
            self.bound_opts = None
        if bound_opts is not None and 'loss_fusion' in bound_opts:
            self.loss_fusion = bound_opts['loss_fusion']
        else:
            self.loss_fusion = False
        self.max_input = 0

    def forward(self, input):
        return torch.exp(input)

    def interval_propagate(self, *v):
        assert (len(v) == 1)
        # unary monotonous functions only
        h_L, h_U = v[0]
        if self.bound_opts != 'no-max-input':
            self.max_input = torch.max(h_U, dim=1, keepdim=True)[0].detach()
            # print(self.max_input.shape, self.max_input.max())
            h_L, h_U = h_L - self.max_input, h_U - self.max_input
        else:
            self.max_input = 0
        return self.forward(h_L), self.forward(h_U)

    def bound_forward(self, dim_in, input):
        epsilon = 1e-12

        m = torch.min((input.lower + input.upper) / 2, input.lower + 0.99)

        exp_l, exp_m, exp_u = torch.exp(input.lower), torch.exp(m), torch.exp(input.upper)

        kl = exp_m
        lw = input.lw * kl.unsqueeze(1)
        lb = kl * (input.lb - m + 1)

        ku = (exp_u - exp_l) / (input.upper - input.lower + epsilon)
        uw = input.uw * ku.unsqueeze(1)
        ub = input.ub * ku - ku * input.lower + exp_l

        return LinearBound(lw, lb, uw, ub, None, None)

    def bound_backward(self, last_lA, last_uA, input):
        # Special case when computing log_softmax (FIXME: find a better solution, this trigger condition is not reliable).
        if self.loss_fusion and last_lA is None and last_uA is not None and torch.min(last_uA) >= 0 and input.from_input:
            # Adding an extra bias term to the input. This is equivalent to adding a constant and subtract layer before exp.
            # Note that we also need to adjust the bias term at end end.
            if self.bound_opts != 'no-max-input':
                self.max_input = torch.max(input.upper, dim=1, keepdim=True)[0].detach()
            else:
                self.max_input = 0
            adjusted_lower = input.lower - self.max_input
            adjusted_upper = input.upper - self.max_input
            # relaxation for upper bound only (used in loss fusion)
            exp_l, exp_u = torch.exp(adjusted_lower), torch.exp(adjusted_upper)
            k = (exp_u - exp_l) / (adjusted_upper - adjusted_lower + 1e-12)
            if k.requires_grad:
                k = k.clamp(min=1e-6)
            uA = last_uA * k.unsqueeze(1)
            ubias = last_uA * (-adjusted_lower * k + exp_l).unsqueeze(1)
            
            # can use tensor.ndim instead of len(tensor.shape) in newer Torch
            if len(ubias.shape) > 2:
                ubias = torch.sum(ubias, dim=tuple(range(2, len(ubias.shape))))
            # Also adjust the missing ubias term.
            if len(uA.shape) > 2:
                A = torch.sum(uA, dim=tuple(range(2, len(uA.shape))))
            else:
                A = uA

            ubias -= A * self.max_input
            return [(None, uA)], 0, ubias
        else:
            return super(BoundExp, self).bound_backward(last_lA, last_uA, input)   
    
    def bound_relax(self, input):
        m = torch.min((input.lower + input.upper) / 2, input.lower + 0.99)
        exp_l, exp_m, exp_u = torch.exp(input.lower), torch.exp(m), torch.exp(input.upper)
        k = exp_m
        self._add_linear(mask=None, type='lower', k=k, x0=m, y0=exp_m)
        k = (exp_u - exp_l) / (input.upper - input.lower + 1e-12)
        k = k.clamp(min=1e-6)
        self._add_linear(mask=None, type='upper', k=k, x0=input.lower, y0=exp_l)

class BoundReciprocal(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundReciprocal, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.nonlinear = True

    def forward(self, input):
        return torch.reciprocal(input)

    def bound_relax(self, input):
        m = (input.lower + input.upper) / 2
        kl = -1 / m.pow(2)
        self._add_linear(mask=None, type='lower', k=kl, x0=m, y0=1. / m)
        ku = -1. / (input.lower * input.upper)
        self._add_linear(mask=None, type='upper', k=ku, x0=input.lower, y0=1. / input.lower)

    def interval_propagate(self, *v):
        # support when h_L > 0
        h_L, h_U = v[0]
        return torch.reciprocal(h_U.float()), torch.reciprocal(h_L.float())

class BoundUnsqueeze(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundUnsqueeze, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.axes = attr['axes']
        assert (len(self.axes) == 1)
        self.axes = self.axes[0]

    def forward(self, x):
        self.input_shape = x.shape
        return x.unsqueeze(self.axes)

    def bound_backward(self, last_lA, last_uA, x):
        if self.axes == 0:
            return last_lA, 0, last_uA, 0
        else:
            return [(last_lA.squeeze(self.axes + 1) if last_lA is not None else None,
               last_uA.squeeze(self.axes + 1) if last_uA is not None else None)], 0, 0

    def bound_forward(self, dim_in, x):
        if len(self.input_shape) == 0:
            lw, lb = x.lw.unsqueeze(1), x.lb.unsqueeze(0)
            uw, ub = x.uw.unsqueeze(1), x.ub.unsqueeze(0)
        else:
            if self.axes < 0:
                self.axes = len(self.input_shape) + self.axes + 1
            assert(self.axes > 0)
            lw, lb = x.lw.unsqueeze(self.axes + 1), x.lb.unsqueeze(self.axes) 
            uw, ub = x.uw.unsqueeze(self.axes + 1), x.ub.unsqueeze(self.axes) 
        return LinearBound(lw, lb, uw, ub, None, None)

class BoundSqueeze(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundSqueeze, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.axes = attr['axes']
        assert (len(self.axes) == 1)
        self.axes = self.axes[0]

    def forward(self, x):
        return x.squeeze(self.axes)

    def bound_backward(self, last_lA, last_uA, x):
        assert(self.axes != 0)
        return [(last_lA.unsqueeze(self.axes + 1) if last_lA is not None else None,
            last_uA.unsqueeze(self.axes + 1) if last_uA is not None else None)], 0, 0

class BoundConstantOfShape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundConstantOfShape, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.device = device
        self.value = attr['value'].to(self.device)

    def forward(self, x):
        self.x = x
        self.from_input = True
        return torch.ones(list(x), device=self.device) * self.value.to(self.device)

    def bound_backward(self, last_lA, last_uA, x):
        if last_lA is not None:
            lower_sum_b = last_lA * self.value
            while len(lower_sum_b.shape) > 2:
                lower_sum_b = torch.sum(lower_sum_b, dim=-1)
        else:
            lower_sum_b = 0

        if last_uA is not None:
            upper_sum_b = last_uA * self.value
            while len(upper_sum_b.shape) > 2:
                upper_sum_b = torch.sum(upper_sum_b, dim=-1)
        else:
            upper_sum_b = 0

        return [(None, None)], lower_sum_b, upper_sum_b

    def bound_forward(self, dim_in, x):
        assert (len(self.x) >= 1)
        lb = ub = torch.ones(self.default_shape, device=self.device) * self.value
        lw = uw = torch.zeros(self.x[0], dim_in, *self.x[1:], device=self.device)
        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, *v):
        self.x = v[0][0]
        value = torch.ones(list(v[0][0]), device=self.device) * self.value
        return value, value

class BoundConstant(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundConstant, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.value = attr['value'].to(self.device)

    def forward(self):
        return self.value.to(self.device)

    def bound_backward(self, last_lA, last_uA):
        def _bound_oneside(A):
            if A is None:
                return 0.0
            while len(A.shape) > 2:
                A = torch.sum(A, dim=-1)
            return A * self.value.to(self.device)
        lbias = _bound_oneside(last_lA)
        ubias = _bound_oneside(last_uA)
        return [], lbias, ubias

    def bound_forward(self, dim_in):
        lw = uw = torch.zeros(dim_in, device=self.device)
        lb = ub = self.value
        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, *v, norm=None):
        return self.value.to(self.device), self.value.to(self.device)

class BoundShape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundShape, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

    def forward(self, x):
        self.from_input = False
        return x.shape

    def bound_backward(self, last_lA, last_uA, x):
        raise NotImplementedError

    def bound_forward(self, dim_in, x):
        return self.forward_value

class BoundGather(Bound):
    def __init__(self, input_name, name, ori_name, attr, input, output_index, device):
        super(BoundGather, self).__init__(input_name, name, ori_name, attr, input, output_index, device)
        self.axis = attr['axis']
        self.nonlinear = True # input shape required

    def forward(self, x, indices):
        if isinstance(x, torch.Size):
            x = torch.tensor(x, device=self.device)
            if indices != 0:
                self.from_input = False
        self.indices = indices = indices.to(x.device)
        self.input_shape = x.shape
        if len(indices.shape) == 0:
            selected = torch.index_select(x, dim=self.axis, index=indices).squeeze(self.axis)
            return selected
        raise NotImplementedError

    def bound_backward(self, last_lA, last_uA, x, indices):
        assert(self.from_input)

        def _bound_oneside(A):
            if A is None:
                return None
            assert (len(self.indices.shape) == 0)

            if self.from_input:
                d = 1
            else:
                d = 2 if self.axis == 0 else 1
            A = A.unsqueeze(self.axis + d)
            idx = int(self.indices)
            tensors = []
            if idx > 0:
                shape_pre = list(A.shape)
                shape_pre[self.axis + d] *= idx
                tensors.append(torch.zeros(shape_pre, device=self.device))
            tensors.append(A)
            if self.input_shape[self.axis] - idx - 1 > 0:
                shape_next = list(A.shape)
                shape_next[self.axis + d] *= self.input_shape[self.axis] - idx - 1
                tensors.append(torch.zeros(shape_next, device=self.device))
            return torch.cat(tensors, dim=self.axis + d)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, indices):
        if isinstance(x, torch.Size):
            lw = uw = torch.zeros(dim_in, device=self.device)
            lb = ub = torch.index_select(
                torch.tensor(x, device=self.device),
                dim=self.axis, index=self.indices).squeeze(self.axis)
        else:
            axis = self.axis + 1
            lw = torch.index_select(x.lw, dim=self.axis + 1, index=self.indices).squeeze(axis)
            uw = torch.index_select(x.uw, dim=self.axis + 1, index=self.indices).squeeze(axis)
            lb = torch.index_select(x.lb, dim=self.axis, index=self.indices).squeeze(self.axis)
            ub = torch.index_select(x.ub, dim=self.axis, index=self.indices).squeeze(self.axis)
        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, *v):
        if isinstance(v[0][0], torch.Tensor):
            h_L, h_U = v[0]
        else:
            h_L, h_U = torch.tensor(v[0][0]), torch.tensor(v[0][1])
        return torch.index_select(h_L.to(self.device),
                                  dim=self.axis, index=self.indices.to(self.device)).squeeze(self.axis), \
               torch.index_select(h_U.to(self.device),
                                  dim=self.axis, index=self.indices.to(self.device)).squeeze(self.axis)

class BoundGatherAten(Bound):
    def __init__(self, input_name, name, ori_name, attr, input, output_index, device):
        super(BoundGatherAten, self).__init__(input_name, name, ori_name, attr, input, output_index, device)
        # FIXME the value of input arguments are needed; but maybe rename 'nonlinear' into 'requires_input_bounds'
        self.nonlinear = True 

    def forward(self, x, dim, index, _):
        return torch.gather(x, dim=dim, index=index)

    def bound_backward(self, last_lA, last_uA, x, dim, index, _):
        assert(self.from_input)

        dim = dim.value
        if dim < 0:
            dim = len(self.default_shape) + dim

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = torch.zeros(
                last_A.shape[0], last_A.shape[1], *x.lower.shape[1:], device=last_A.device)
            A.scatter_(
                dim=dim + 1, 
                index=index.lower.unsqueeze(1).repeat(1, A.shape[1], *([1] * (len(A.shape) - 2))),
                src=last_A)
            return A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None), (None, None), (None, None)], 0, 0

    def interval_propagate(self, *v):
        return self.forward(v[0][0], v[1][0], v[2][0], v[3][0]), \
            self.forward(v[0][1], v[1][1], v[2][1], v[3][1])

    def bound_forward(self, dim_in, x, indices):
        raise NotImplementedError

class BoundGatherElements(Bound):
    def __init__(self, input_name, name, ori_name, attr, input, output_index, device):
        super(BoundGatherElements, self).__init__(input_name, name, ori_name, attr, input, output_index, device)
        self.axis = attr['axis']
        # FIXME the value of input arguments are needed; but maybe rename 'nonlinear' into 'requires_input_bounds'
        self.nonlinear = True 

    def forward(self, x, index):
        return torch.gather(x, dim=self.axis, index=index)

    def bound_backward(self, last_lA, last_uA, x, index):
        assert(self.from_input)

        dim = self.axis
        if dim < 0:
            dim = len(self.default_shape) + dim

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = torch.zeros(
                last_A.shape[0], last_A.shape[1], *x.lower.shape[1:], device=last_A.device)
            A.scatter_(
                dim=dim + 1, 
                index=index.lower.unsqueeze(1).repeat(1, A.shape[1], *([1] * (len(A.shape) - 2))),
                src=last_A)
            return A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def interval_propagate(self, *v):
        return self.forward(v[0][0], v[1][0]), \
            self.forward(v[0][1], v[1][1])

    def bound_forward(self, dim_in, x, indices):
        raise NotImplementedError

class BoundPrimConstant(Bound):
    def __init__(self, input_name, name, ori_name, attr, input, output_index, device):
        super(BoundPrimConstant, self).__init__(input_name, name, ori_name, attr, input, output_index, device)
        self.value = attr['value']

    def forward(self):
        return torch.tensor([], device=self.device)

    def bound_backward(self, last_lA, last_uA):
        raise NotImplementedError

    def interval_propagate(self, norm, h_U, h_L, eps, C=None):
        raise NotImplementedError

class BoundRNN(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundRNN, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.complex = True
        self.output_index = output_index

    def forward(self, x, weight_input, weight_recurrent, bias, sequence_length, initial_h):
        assert (torch.sum(torch.abs(initial_h)) == 0)

        self.input_size = x.shape[-1]
        self.hidden_size = weight_input.shape[-2]

        class BoundRNNImpl(nn.Module):
            def __init__(self, input_size, hidden_size,
                         weight_input, weight_recurrent, bias, output_index, device):
                super(BoundRNNImpl, self).__init__()

                self.input_size = input_size
                self.hidden_size = hidden_size

                self.cell = torch.nn.RNNCell(
                    input_size=input_size,
                    hidden_size=hidden_size
                )

                self.cell.weight_ih.data.copy_(weight_input.squeeze(0).data)
                self.cell.weight_hh.data.copy_(weight_recurrent.squeeze(0).data)
                self.cell.bias_ih.data.copy_((bias.squeeze(0))[:hidden_size].data)
                self.cell.bias_hh.data.copy_((bias.squeeze(0))[hidden_size:].data)

                self.output_index = output_index

            def forward(self, x):
                length = x.shape[0]
                outputs = []
                hidden = torch.zeros(x.shape[1], self.hidden_size, device=self.device)
                for i in range(length):
                    hidden = self.cell(x[i, :], hidden)
                    outputs.append(hidden.unsqueeze(0))
                outputs = torch.cat(outputs, dim=0)

                if self.output_index == 0:
                    return outputs
                else:
                    return hidden

        self.model = BoundRNNImpl(
            self.input_size, self.hidden_size,
            weight_input, weight_recurrent, bias,
            self.output_index, self.device)
        self.input = (x,)

        return self.model(self.input)

class BoundTranspose(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundTranspose, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.perm = attr['perm']
        self.perm_inv = [-1] * len(self.perm)
        for i in range(len(self.perm)):
            self.perm_inv[self.perm[i]] = i

    def forward(self, x):
        self.input_shape = x.shape
        return x.permute(*self.perm)

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(A):
            if A is None:
                return None
            if self.input_shape[0] == 1 and self.perm_inv[0] == 1 and self.perm_inv[1] == 0:
                return A

            perm = [0, 1]
            if self.from_input:
                assert (self.perm[0] == 0)
                for p in self.perm_inv[1:]:
                    perm.append(p + 1)
            else:
                for p in self.perm_inv:
                    perm.append(p + 2)

            return A.permute(perm)
        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        if self.input_shape[0] != 1:
            perm = [0] + [(p + 1) for p in self.perm]
        else:
            assert (self.perm[0] == 0)
            perm = [0, 1] + [(p + 1) for p in self.perm[1:]]
        lw, lb = x.lw.permute(*perm), x.lb.permute(self.perm)
        uw, ub = x.uw.permute(*perm), x.ub.permute(self.perm)

        return LinearBound(lw, lb, uw, ub, None, None)

class BoundMul(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundMul, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.nonlinear = True

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x * y

    @staticmethod
    def get_bound_mul(x_l, x_u, y_l, y_u):
        alpha_l = y_l
        beta_l = x_l
        gamma_l = -alpha_l * beta_l

        alpha_u = y_u
        beta_u = x_l
        gamma_u = -alpha_u * beta_u
        
        return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    # Special case when input is x * x.
    @staticmethod
    def get_bound_square(x_l, x_u):
        # Lower bound is a z=0 line if x_l and x_u have different signs.
        # Otherwise, the lower bound is a tangent line at x_l.
        # The lower bound should always be better than IBP.

        # If both x_l and x_u < 0, select x_u. If both > 0, select x_l.
        # If x_l < 0 and x_u > 0, we use the z=0 line as the lower bound.
        x_m = F.relu(x_l) - F.relu(-x_u)
        alpha_l = 2 * x_m
        gamma_l = - x_m * x_m

        # Upper bound: connect the two points (x_l, x_l^2) and (x_u, x_u^2).
        # The upper bound should always be better than IBP.
        alpha_u = x_l + x_u
        gamma_u = - x_l * x_u

        # Parameters before the second variable are all zeros, not used.
        beta_l = torch.zeros_like(x_l)
        beta_u = beta_l
        return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    @staticmethod
    def _relax(x, y):
        if x is y:
            # A shortcut for x * x.
            return BoundMul.get_bound_square(x.lower, x.upper)
        
        x_l, x_u = x.lower, x.upper
        y_l, y_u = y.lower, y.upper

        # broadcast
        for k in [1, -1]:
            x_l = x_l + k * y_l
            x_u = x_u + k * y_u
        for k in [1, -1]:
            y_l = y_l + k * x_l
            y_u = y_u + k * x_u

        return BoundMul.get_bound_mul(x_l, x_u, y_l, y_u)

    def bound_backward(self, last_lA, last_uA, x, y):
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = BoundMul._relax(x, y)

        alpha_l, alpha_u = alpha_l.unsqueeze(1), alpha_u.unsqueeze(1)
        beta_l, beta_u = beta_l.unsqueeze(1), beta_u.unsqueeze(1)
        batch_size = alpha_l.shape[0]

        gamma_l = gamma_l.reshape(batch_size, -1, 1)
        gamma_u = gamma_u.reshape(batch_size, -1, 1)     

        def _bound_oneside(last_A, 
                        alpha_pos, beta_pos, gamma_pos, 
                        alpha_neg, beta_neg, gamma_neg):
            if last_A is None:
                return None, None, 0
            A_x = last_A.clamp(min=0) * alpha_pos + last_A.clamp(max=0) * alpha_neg
            A_y = last_A.clamp(min=0) * beta_pos + last_A.clamp(max=0) * beta_neg
            last_A = last_A.reshape(last_A.shape[0], last_A.shape[1], -1)
            bias = torch.bmm(last_A.clamp(min=0), gamma_pos).squeeze(-1) + \
                    torch.bmm(last_A.clamp(max=0), gamma_neg).squeeze(-1)
            A_x = self._broadcast_backward(A_x, x)
            A_y = self._broadcast_backward(A_y, y)
            return A_x, A_y, bias
        
        lA_x, lA_y, lbias = _bound_oneside(
            last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
        uA_x, uA_y, ubias = _bound_oneside(
            last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    @staticmethod
    def bound_forward(dim_in, x, y):
        x_lw, x_lb, x_uw, x_ub = x.lw, x.lb, x.uw, x.ub
        y_lw, y_lb, y_uw, y_ub = y.lw, y.lb, y.uw, y.ub

        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = BoundMul._relax(x, y)

        if x_lw is None: x_lw = 0
        if y_lw is None: y_lw = 0
        if x_uw is None: x_uw = 0
        if y_uw is None: y_uw = 0

        lw = alpha_l.unsqueeze(1).clamp(min=0) * x_lw + alpha_l.unsqueeze(1).clamp(max=0) * x_uw
        lw = lw + beta_l.unsqueeze(1).clamp(min=0) * y_lw + beta_l.unsqueeze(1).clamp(max=0) * y_uw
        lb = alpha_l.clamp(min=0) * x_lb + alpha_l.clamp(max=0) * x_ub + \
            beta_l.clamp(min=0) * y_lb + beta_l.clamp(max=0) * y_ub + gamma_l
        uw = alpha_u.unsqueeze(1).clamp(max=0) * x_lw + alpha_u.unsqueeze(1).clamp(min=0) * x_uw
        uw = uw + beta_u.unsqueeze(1).clamp(max=0) * y_lw + beta_u.unsqueeze(1).clamp(min=0) * y_uw
        ub = alpha_u.clamp(max=0) * x_lb + alpha_u.clamp(min=0) * x_ub + \
            beta_u.clamp(max=0) * y_lb + beta_u.clamp(min=0) * y_ub + gamma_u

        return LinearBound(lw, lb, uw, ub, None, None)

    @staticmethod
    def interval_propagate(*v):
        x, y = v[0], v[1]
        if x is y:
            # A shortcut for x * x.
            h_L, h_U = v[0]
            r0 = h_L * h_L
            r1 = h_U * h_U
            # When h_L < 0, h_U > 0, lower bound is 0.
            # When h_L < 0, h_U < 0, lower bound is h_U * h_U.
            # When h_L > 0, h_U > 0, lower bound is h_L * h_L.
            l = F.relu(h_L) - F.relu(-h_U)
            return l * l, torch.max(r0, r1)
            
        r0, r1, r2, r3 = x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]
        lower = torch.min(torch.min(r0, r1), torch.min(r2, r3))
        upper = torch.max(torch.max(r0, r1), torch.max(r2, r3))
        return lower, upper

class BoundDiv(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundDiv, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.nonlinear = True

    def forward(self, x, y):
        self.x, self.y = x, y
        return x / y

    def bound_backward(self, last_lA, last_uA, x, y):
        reciprocal, mul, y_r = self._convert_to_mul(x, y)
        A, lower_b, upper_b = mul.bound_backward(last_lA, last_uA, x, y_r)

        A_y, lower_b_y, upper_b_y = reciprocal.bound_backward(A[1][0], A[1][1], y)
        upper_b = upper_b + upper_b_y
        lower_b = lower_b + lower_b_y

        return [A[0], A_y[0]], lower_b, upper_b

    def bound_forward(self, dim_in, x, y):
        reciprocal, mul, y_r = self._convert_to_mul(x, y)
        y_r_linear = reciprocal.bound_forward(dim_in, y)
        y_r_linear = y_r_linear._replace(lower=y_r.lower, upper=y_r.upper)
        return mul.bound_forward(dim_in, x, y_r_linear)

    def interval_propagate(self, *v):
        x, y = v[0], v[1]
        y_r = BoundReciprocal.interval_propagate(None, y)
        return BoundMul.interval_propagate(x, y_r)

    def _convert_to_mul(self, x, y):
        try:
            reciprocal = BoundReciprocal(self.input_name, self.name + '/reciprocal', self.ori_name, {}, [], 0, self.device)
            mul = BoundMul(self.input_name, self.name + '/mul', self.ori_name, {}, [], 0, self.device)
        except:
            # to make it compatible with previous code
            reciprocal = BoundReciprocal(self.input_name, self.name + '/reciprocal', None, {}, [], 0, self.device)
            mul = BoundMul(self.input_name, self.name + '/mul', None, {}, [], 0, self.device)
        mul.default_shape = self.default_shape

        y_r = copy.copy(y)
        if isinstance(y_r, LinearBound):
            y_r = y_r._replace(lower=1. / y.upper, upper=1. / y.lower)
        else:
            y_r.lower = 1. / y.upper
            y_r.upper = 1. / y.lower
        return reciprocal, mul, y_r

class BoundNeg(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundNeg, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

    def forward(self, x):
        return -x

    def bound_backward(self, last_lA, last_uA, x):
        return [(-last_lA if last_lA is not None else None, 
            -last_uA if last_uA is not None else None)], 0, 0

    def bound_forward(self, dim_in, x):
        return LinearBound(-x.lw, -x.lb, -x.uw, -x.ub, None, None)

    def interval_propagate(self, *v):
        return -v[0][1], -v[0][0]

class BoundMatMul(BoundLinear):
    # Reuse most functions from BoundLinear.
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundLinear, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.nonlinear = True

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        self.x = x
        self.y = y
        return x.matmul(y)

    def interval_propagate(self, *v):
        w_l = v[1][0].transpose(-1, -2)
        w_u = v[1][1].transpose(-1, -2)
        return super().interval_propagate(v[0], (w_l, w_u))

    def bound_backward(self, last_lA, last_uA, *x):
        assert len(x) == 2
        # BoundLinear has W transposed.
        x[1].lower = x[1].lower.transpose(-1, -2)
        x[1].upper = x[1].upper.transpose(-1, -2)
        results = super().bound_backward(last_lA, last_uA, *x)
        # Transpose input back.
        x[1].lower = x[1].lower.transpose(-1, -2)
        x[1].upper = x[1].upper.transpose(-1, -2)
        lA_y = results[0][1][0].transpose(-1, -2) if results[0][1][0] is not None else None
        uA_y = results[0][1][1].transpose(-1, -2) if results[0][1][1] is not None else None
        # Transpose result on A.
        return [results[0][0], (lA_y, uA_y), results[0][2]], results[1], results[2]
    
    def bound_forward(self, dim_in, x, y):
        return super().bound_forward(dim_in, x, LinearBound(
            y.lw.transpose(-1, -2) if y.lw is not None else None,
            y.lb.transpose(-1, -2) if y.lb is not None else None,
            y.uw.transpose(-1, -2) if y.uw is not None else None,  
            y.ub.transpose(-1, -2) if y.ub is not None else None,
            y.lower.transpose(-1, -2) if y.lower is not None else None,
            y.upper.transpose(-1, -2) if y.upper is not None else None
        ))
        
class BoundCast(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundCast, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.to = attr['to']
        assert (self.to == 1)  # to float

    def forward(self, x):
        assert (self.to == 1)
        return x.float()

    def bound_backward(self, last_lA, last_uA, x):
        return [(last_lA, last_uA)], 0, 0

    def bound_forward(self, dim_in, x):
        return LinearBound(
            x.lw.float(), x.lb.float(),
            x.uw.float(), x.ub.float(), None, None)

    def interval_propagate(self, *v):
        return v[0][0].float(), v[0][1].float()

class BoundSoftmaxImpl(nn.Module):
    def __init__(self, axis):
        super(BoundSoftmaxImpl, self).__init__()
        self.axis = axis

    def forward(self, x):
        x = torch.exp(x)
        s = torch.sum(x, dim=self.axis, keepdim=True)
        return x / s

class BoundSoftmax(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundSoftmax, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.axis = attr['axis']
        self.complex = True

    def forward(self, x):
        self.input = (x,)
        self.model = BoundSoftmaxImpl(self.axis)
        self.model.device = self.device
        return self.model(x)

    def bound_backward(self, last_lA, last_uA):
        raise NotImplementedError

    def interval_propagate(self, norm, h_L, h_U, eps, C=None):
        raise NotImplementedError

class BoundReduceMean(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundReduceMean, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.axis = attr['axes']
        self.keepdim = bool(attr['keepdims'])

    def forward(self, x):
        self.input_shape = x.shape
        return torch.mean(x, dim=self.axis, keepdim=self.keepdim)

    def bound_backward(self, last_lA, last_uA, x):
        for i in range(len(self.axis)):
            if self.axis[i] < 0:
                self.axis[i] = len(self.input_shape) + self.axis[i]

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if not self.keepdim:
                assert (self.from_input)
                for axis in self.axis:
                    if axis > 0:
                        last_A = last_A.unsqueeze(axis + 1)
            for axis in self.axis:
                repeat = [1] * len(last_A.shape)
                size = self.input_shape[axis]
                if axis > 0:
                    repeat[axis + 1] *= size
                last_A = last_A.repeat(*repeat) / size
            return last_A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert (self.keepdim)
        assert (len(self.axis) == 1)
        axis = self.axis[0]
        if axis < 0:
            axis = len(self.input_shape) + axis
        assert (axis > 0)
        size = self.input_shape[axis]
        lw = x.lw.sum(dim=axis + 1, keepdim=True) / size
        lb = x.lb.sum(dim=axis, keepdim=True) / size
        uw = x.uw.sum(dim=axis + 1, keepdim=True) / size
        ub = x.ub.sum(dim=axis, keepdim=True) / size
        return LinearBound(lw, lb, uw, ub, None, None)

class BoundReduceSum(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundReduceSum, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.axis = attr['axes']
        self.keepdim = bool(attr['keepdims'])

    def forward(self, x):
        self.input_shape = x.shape
        return torch.sum(x, dim=self.axis, keepdim=self.keepdim)

    def bound_backward(self, last_lA, last_uA, x):
        for i in range(len(self.axis)):
            if self.axis[i] < 0:
                self.axis[i] = len(self.input_shape) + self.axis[i]

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if not self.keepdim:
                assert(self.from_input)
                for axis in self.axis:
                    if axis > 0:
                        last_A = last_A.unsqueeze(axis + 1)
            for axis in self.axis:
                repeat = [1] * len(last_A.shape)
                size = self.input_shape[axis]
                if axis > 0:
                    repeat[axis + 1] *= size
                last_A = last_A.repeat(*repeat)
            return last_A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert (self.keepdim)
        assert (len(self.axis) == 1)
        axis = self.axis[0]
        if axis < 0:
            axis = len(self.input_shape) + axis
        assert (axis > 0)
        lw, lb = x.lw.sum(dim=axis + 1, keepdim=True), x.lb.sum(dim=axis, keepdim=True)
        uw, ub = x.uw.sum(dim=axis + 1, keepdim=True), x.ub.sum(dim=axis, keepdim=True)
        return LinearBound(lw, lb, uw, ub, None, None)

class BoundDropout(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundDropout, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.dropout = nn.Dropout(p=attr['ratio'])

    def forward(self, x):
        res = self.dropout(x)
        self.mask = (res / (x + 1e-12)).detach().requires_grad_(False)
        return res

    def bound_backward(self, last_lA, last_uA, x):
        lA = last_lA * self.mask.unsqueeze(1) if last_lA is not None else None
        uA = last_uA * self.mask.unsqueeze(1) if last_uA is not None else None
        return [(lA, uA)], 0, 0

    def bound_forward(self, dim_in, x):
        assert (torch.min(self.mask) >= 0)
        lw = x.lw * self.mask.unsqueeze(1)
        lb = x.lb * self.mask
        uw = x.uw * self.mask.unsqueeze(1)
        ub = x.ub * self.mask
        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        return h_L * self.mask, h_U * self.mask

class BoundSplit(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundSplit, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.axis = attr['axis']
        self.split = attr['split']

    def forward(self, x):
        self.input_shape = x.shape
        return torch.split(x, self.split, dim=self.axis)[self.output_index]

    def bound_backward(self, last_lA, last_uA, x):
        assert (self.axis > 0)
        pre = sum(self.split[:self.output_index])
        suc = sum(self.split[(self.output_index + 1):])

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = []
            if pre > 0:
                A.append(torch.zeros(
                    *last_A.shape[:(self.axis + 1)], pre, *last_A.shape[(self.axis + 2):],
                    device=last_A.device))
            A.append(last_A)
            if suc > 0:
                A.append(torch.zeros(
                    *last_A.shape[:(self.axis + 1)], suc, *last_A.shape[(self.axis + 2):],
                    device=last_A.device))
            return torch.cat(A, dim=self.axis + 1)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert (self.axis > 0 and self.from_input)
        lw = torch.split(x.lw, self.split, dim=self.axis + 1)[self.output_index]
        uw = torch.split(x.uw, self.split, dim=self.axis + 1)[self.output_index]
        lb = torch.split(x.lb, self.split, dim=self.axis)[self.output_index]
        ub = torch.split(x.ub, self.split, dim=self.axis)[self.output_index]
        return LinearBound(lw, lb, uw, ub, None, None)

class BoundInput(nn.Module):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super(BoundInput, self).__init__()
        self.input_name = input_name
        self.output_name = []
        self.name = name  # Name on converted computational graph
        self.ori_name = ori_name  # Name on original computational graph
        self.forward_value = None
        self.bounded = False
        self.value = value
        self.perturbation = perturbation

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        # Update node_perturbed property based on the perturbation set.
        if key == "perturbation":
            if self.perturbation is not None:
                self.node_perturbed = True
            else:
                self.node_perturbed = False

    def forward(self):
        return self.value

    def bound_forward(self, dim_in):
        assert (0)

    def bound_backward(self, last_lA, last_uA):
        assert (0)

    def interval_propagate(self, *v):
        assert (0)

class BoundParams(BoundInput):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super(BoundParams, self).__init__(input_name, name, ori_name, None, perturbation)
        self.from_input = False
        self.ori_name = ori_name
        self.register_parameter('param', value)

    """Override register_parameter() hook to register only needed parameters."""
    def register_parameter(self, name, param):
        if name == 'param':
            # self._parameters[name] = param  # cannot contain '.' in name, it will cause error when loading state_dict
            return super().register_parameter(name, param)
        else:
            # Just register it as a normal property of class.
            object.__setattr__(self, name, param)

    def forward(self):
        return self.param

class BoundBuffers(BoundInput):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super(BoundBuffers, self).__init__(input_name, name, ori_name, None, perturbation)
        self.register_buffer('buffer', value.clone().detach())

    def forward(self):
        return self.buffer

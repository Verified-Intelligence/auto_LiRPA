import copy, pdb, pickle, os, time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, \
    AdaptiveAvgPool2d, ConstantPad2d, AvgPool2d, Tanh
from auto_LiRPA.utils import logger

LinearBound = namedtuple('LinearBound', ('lw', 'lb', 'uw', 'ub', 'lower', 'upper'))
Hull = namedtuple('Hull', ('points', 'mask', 'eps', 'lower', 'upper'))

class Bound(nn.Module):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(Bound, self).__init__()
        self.output_name = []
        self.input_name, self.name, self.ori_name, self.attr, self.inputs, self.output_index, self.device = \
            input_name, name, ori_name, attr, inputs, output_index, device
        self.forward_value = None
        self.bounded = False
        self.IBP_rets = None

    def forward(self, *input):
        raise NotImplementedError

    def interval_propagate(self, *v):
        assert (len(v) == 1)
        # unary monotonous functions only 
        h_L, h_U = v[0]
        return self.forward(h_L), self.forward(h_U)

    def bound_forward(self, dim_in, last):
        raise NotImplementedError

    def bound_backward(self, last_lA, last_uA):
        raise NotImplementedError

    def _broadcast_backward(self, A, input):
        shape = input.forward_value.shape
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
        h_L, h_U = v[0]
        return h_L.reshape(self.forward_value.shape), h_U.reshape(self.forward_value.shape)


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

    # w: an optional argument which can be utilized by BoundMatMul
    def bound_forward(self, dim_in, x, w=None, b=None):
        if isinstance(w, LinearBound):
            w = w.lower
        if isinstance(b, LinearBound):
            b = b.lower
        w = w.t()
        mask = torch.gt(w, 0).to(torch.float32)
        w_pos = w * mask
        w_neg = w - w_pos

        lw = x.lw.matmul(w_pos) + x.uw.matmul(w_neg)
        lb = x.lb.matmul(w_pos) + x.ub.matmul(w_neg)
        uw = x.uw.matmul(w_pos) + x.lw.matmul(w_neg)
        ub = x.ub.matmul(w_pos) + x.lb.matmul(w_neg)
        if b is not None:
            lb += b
            ub += b

        return LinearBound(lw, lb, uw, ub, None, None)

    def bound_backward(self, last_lA, last_uA, *x):
        assert len(x) == 3
        # x[0]: input node, x[1]: weight, x[2]: bias
        lA_y = uA_y = lA_bias = uA_bias = None
        lbias = ubias = 0
        if x[1].perturbation is None and x[2].perturbation is None:
            if isinstance(last_lA, tuple) and isinstance(last_uA, tuple):
                lA_x = uA_x = x[1].param.unsqueeze(0).repeat([last_lA.shape[0]]+[1]*len(x[1].param.shape))
                lbias = ubias = x[2].param.unsqueeze(0).repeat(last_lA.shape[0], 1)
                return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

            def _bound_oneside(last_A):
                if last_A is None:
                    return None, 0
                next_A = last_A.matmul(x[1].param)
                sum_bias = last_A.matmul(x[2].param)
                return next_A, sum_bias

            lA_x, lbias = _bound_oneside(last_lA)
            uA_x, ubias = _bound_oneside(last_uA)
            return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

        if x[1].perturbation is not None:
            if isinstance(last_lA, tuple) and isinstance(last_uA, tuple):
                lA_x = x[1].lower.unsqueeze(0).repeat([last_lA.shape[0]] + [1] * len(x[1].param.shape))
                uA_x = x[1].upper.unsqueeze(0).repeat([last_uA.shape[0]] + [1] * len(x[1].param.shape))
                #  calculate Ay*y here directly and add them to bias to avoid redundant concretize
                lbias += x[0].lower.matmul(x[1].param.t())
                ubias += x[0].upper.matmul(x[1].param.t())
            else:
                [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias = self.boundMul_backward(last_lA, last_uA, x[0], x[1])
            if x[2].perturbation is not None:
                if isinstance(last_lA, tuple) and isinstance(last_uA, tuple):
                    #  calculate A_bias*A_bias here directly and add them to bias to avoid redundant concretize
                    lbias += x[2].lower.repeat(last_lA.shape[0], 1)
                    ubias += x[2].upper.repeat(last_lA.shape[0], 1)
                    return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias
                else:
                    lA_bias = last_lA
                    uA_bias = last_uA
            else:
                if isinstance(last_lA, tuple) and isinstance(last_uA, tuple):
                    lbias += x[2].param.unsqueeze(0).repeat(last_lA.shape[0], 1)
                    ubias += x[2].param.unsqueeze(0).repeat(last_uA.shape[0], 1)
                    return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias
                else:
                    lbias += last_lA.matmul(x[2].param)
                    ubias += last_uA.matmul(x[2].param)

            # 　lA_y.reshape(2,lA_y.shape[1],-1).bmm(x[1].param.t().unsqueeze(0).repeat(2,1,1).reshape(2,-1).unsqueeze(-1))
            #  transpose A_y, it will be multiplied with x[1].param in concretization.
            return [(lA_x, uA_x), (lA_y.transpose(-1, -2), uA_y.transpose(-1, -2)), (lA_bias, uA_bias)], lbias, ubias

        if x[1].perturbation is None:
            if isinstance(last_lA, tuple) and isinstance(last_uA, tuple):
                lA_x = uA_x = x[1].param.unsqueeze(0).repeat([last_lA.shape[0]]+[1]*len(x[1].param.shape))
                lbias += x[2].lower.repeat(last_lA.shape[0], 1)
                ubias += x[2].upper.repeat(last_lA.shape[0], 1)
                return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

            lA_x = last_lA.matmul(x[1].param)
            uA_x = last_uA.matmul(x[1].param)
            lA_bias = last_lA
            uA_bias = last_uA

            return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], 0, 0

    def _relax(self, x, y):
        x_shape, y_shape = x.lower.shape, y.lower.transpose(-1, -2).shape
        if len(y_shape) != 2 and len(x_shape) == len(y_shape):
            # (x_1, x_2, ..., x_{n-1}, y_{n}, x_n)
            repeat = (1,) * (len(x_shape) - 1) + y_shape[-1:] + (1,)
            x_l = x.lower.unsqueeze(-2).repeat(*repeat)
            x_u = x.upper.unsqueeze(-2).repeat(*repeat)

            # (x_1, x_2, ..., x_{n-1}, y_n, y_{n-1})
            shape = x_shape[:-1] + (y_shape[-1], y_shape[-2])
            repeat = (1,) * (len(x_shape) - 2) + (x_shape[-2], 1)
            y_l = y.lower.repeat(*repeat).reshape(*shape)
            y_u = y.upper.repeat(*repeat).reshape(*shape)
        elif len(y_shape) == 2:
            # (x_1, x_2, ..., x_{n-1}, y_2, x_n)
            repeat = (1,) * (len(x_shape) - 1) + y_shape[-1:] + (1,)
            x_l = x.lower.unsqueeze(-2).repeat(*repeat)
            x_u = x.upper.unsqueeze(-2).repeat(*repeat)

            # (x_1, x_2, ..., x_{n-1}, y_2, y_1)
            shape = x_shape[:-1] + y_shape[1:] + y_shape[:1]
            repeat = (np.prod(x_shape[:-1]),) + (1,)
            y_l = y.lower.repeat(*repeat).reshape(*shape)
            y_u = y.upper.repeat(*repeat).reshape(*shape)

        return BoundMul.get_bound_mul(x_l, x_u, y_l, y_u)

    def boundMul_backward(self, last_lA, last_uA, x, y):
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self._relax(x, y)

        alpha_l, alpha_u = alpha_l.unsqueeze(1), alpha_u.unsqueeze(1)
        beta_l, beta_u = beta_l.unsqueeze(1), beta_u.unsqueeze(1)
        gamma_l = torch.sum(gamma_l, dim=-1).reshape(self.x_shape[0], -1, 1)
        gamma_u = torch.sum(gamma_u, dim=-1).reshape(self.x_shape[0], -1, 1)

        x_shape, y_shape = self.x_shape, self.y_shape

        if len(x.forward_value.shape) != 2 and len(x.forward_value.shape) == len(y.forward_value.shape):
            assert (x_shape[0] == y_shape[0] == 1)
            dim_y = -3
        elif len(y.forward_value.shape) == 2:
            dim_y = list(range(2, 2 + len(x_shape) - 2))
        else:
            raise NotImplementedError

        mask = torch.gt(last_uA.unsqueeze(-1), 0.).to(torch.float)
        uA_x = torch.sum(last_uA.unsqueeze(-1) * (
                mask * alpha_u + (1 - mask) * alpha_l), dim=-2)
        if len(dim_y) == 0:
            uA_y = last_uA.unsqueeze(-1) * (mask * beta_u + (1 - mask) * beta_l)
        else:
            uA_y = torch.sum(last_uA.unsqueeze(-1) * (
                    mask * beta_u + (1 - mask) * beta_l), dim=dim_y)
        uA_y = uA_y.transpose(-1, -2)

        mask = torch.gt(last_lA.unsqueeze(-1), 0.).to(torch.float)
        lA_x = torch.sum(last_lA.unsqueeze(-1) * (
                mask * alpha_l + (1 - mask) * alpha_u), dim=-2)
        if len(dim_y) == 0:
            lA_y = last_lA.unsqueeze(-1) * (mask * beta_l + (1 - mask) * beta_u)
        else:
            lA_y = torch.sum(last_lA.unsqueeze(-1) * (
                    mask * beta_l + (1 - mask) * beta_u), dim=dim_y)
        lA_y = lA_y.transpose(-1, -2)

        _last_uA = last_uA.reshape(last_uA.shape[0], last_uA.shape[1], -1)
        _last_lA = last_lA.reshape(last_lA.shape[0], last_lA.shape[1], -1)
        mask = torch.gt(_last_uA, 0.).to(torch.float)
        ubias = (mask * _last_uA).matmul(gamma_u) + ((1 - mask) * _last_uA).matmul(gamma_l)
        mask = torch.gt(_last_lA, 0.).to(torch.float)
        lbias = (mask * _last_lA).matmul(gamma_l) + ((1 - mask) * _last_lA).matmul(gamma_u)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias.squeeze(), ubias.squeeze()

    def interval_propagate(self, *v, C=None, w=None):
        if w is None and self is None:
            # use C only
            w, lb, ub = C, torch.tensor(0., device=C.device), torch.tensor(0., device=C.device)
        else:
            if w is None:
                if not torch.isclose(v[1][1], v[1][0], 1e-8).all():
                    # w is perturbed tensor
                    l, u = self.bound_interval_propagate(*v)
                    return l + v[2][0], u + v[2][1]
                else:
                    w = v[1][0]
            if len(v) >= 3:
                lb = v[2][0]
                ub = v[2][1]
            else:
                lb = ub = torch.tensor(0., device=w.device)

            if C is not None:
                w = C.matmul(w)
                lb = C.matmul(lb)
                ub = C.matmul(ub)

        if isinstance(v[0], Hull):
            points = v[0].points.matmul(w.t())
            mask = v[0].mask.unsqueeze(-1)

            inf = 1e20
            lower = torch.min(points * mask + (inf) * (1 - mask), dim=2).values + lb
            upper = torch.max(points * mask + (-inf) * (1 - mask), dim=2).values + ub

            lower = self.forward_value - (self.forward_value - lower) * v[0].eps
            upper = self.forward_value + (upper - self.forward_value) * v[0].eps
        else:
            h_L, h_U = v[0]
            mid = (h_L + h_U) / 2
            diff = (h_U - h_L) / 2
            w_abs = w.abs()
            if C is not None:
                center = torch.bmm(mid.unsqueeze(1), w.transpose(-1, -2)).squeeze(1)
                deviation = torch.bmm(diff.unsqueeze(1), w_abs.transpose(-1, -2)).squeeze(1)
            else:
                center = mid.matmul(w.t())
                deviation = diff.matmul(w_abs.t())
            lower, upper = center - deviation + lb, center + deviation + ub

        return lower, upper

    def bound_interval_propagate(self, *v):
        # x, y = v[0], v[1]
        #
        # r0, r1, r2, r3 = x[0].matmul(y[0].transpose(-1,-2)), x[0].matmul(y[1].transpose(-1,-2)), \
        #                  x[1].matmul(y[0].transpose(-1,-2)), x[1].matmul(y[1].transpose(-1,-2))
        # lower = torch.min(torch.min(r0, r1), torch.min(r2, r3))
        # upper = torch.max(torch.max(r0, r1), torch.max(r2, r3))
        # return lower, upper

        x = LinearBound(None, None, None, None, v[0][0], v[0][1])
        y = LinearBound(None, None, None, None, v[1][0], v[1][1])

        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self._relax(x, y)

        mask = torch.gt(alpha_l, 0.).to(torch.float32)
        lower = torch.sum(alpha_l * \
                          (mask * x.lower.unsqueeze(-2) + (1 - mask) * x.upper.unsqueeze(-2)), dim=-1)
        mask = torch.lt(alpha_u, 0.).to(torch.float32)
        upper = torch.sum(alpha_u * \
                          (mask * x.lower.unsqueeze(-2) + (1 - mask) * x.upper.unsqueeze(-2)), dim=-1)

        y_lower = y.lower.unsqueeze(-3)
        y_upper = y.upper.unsqueeze(-3)

        mask = torch.gt(beta_l, 0.).to(torch.float32)
        lower += torch.sum(beta_l * (mask * y_lower + (1 - mask) * y_upper), dim=-1)
        lower += torch.sum(gamma_l, dim=-1)

        mask = torch.lt(beta_u, 0.).to(torch.float32)
        upper += torch.sum(beta_u * (mask * y_lower + (1 - mask) * y_upper), dim=-1)
        upper += torch.sum(gamma_u, dim=-1)

        return lower, upper


class BoundBatchNorm2d(BatchNorm2d):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device, training):
        input_name = input_name[:1]
        num_features = inputs[2].shape[0]
        eps = attr['epsilon']
        momentum = 1 - attr['momentum']  # take care!
        weight = inputs[1]
        bias = inputs[2]
        running_mean = inputs[3]
        running_var = inputs[4]
        affine = True
        track_running_stats = True

        super(BoundBatchNorm2d, self).__init__(num_features=num_features, eps=eps, momentum=momentum,
                                               affine=affine, track_running_stats=track_running_stats)
        self.input_name = input_name
        self.output_name = []
        self.name = name
        self.ori_name = ori_name
        self.forward_value = None
        self.bounded = False
        self.to(device)

        self.weight.data.copy_(weight.data)
        self.bias.data.copy_(bias.data)
        self.running_mean.data.copy_(running_mean.data)
        self.running_var.data.copy_(running_var.data)
        self.training = training
        self.IBP_rets = None

    def forward(self, x):
        output = super(BoundBatchNorm2d, self).forward(x)
        return output

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            next_A = last_A * (self.weight / torch.sqrt(self.running_var + self.eps)).view(1, 1, -1, 1, 1)
            tmp_bias = self.bias - self.running_mean / torch.sqrt(self.running_var + self.eps) * self.weight
            sum_bias = (last_A.sum((3, 4)) * tmp_bias).sum(2)
            return next_A, sum_bias

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)
        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v, eps=None, norm=None, C=None):
        h_U, h_L = v[0]
        mid = (h_U + h_L) / 2.0
        diff = (h_U - h_L) / 2.0

        tmp_weight = self.weight / torch.sqrt(self.running_var + self.eps)
        tmp_weight_abs = tmp_weight.abs()
        tmp_bias = self.bias - self.running_mean / torch.sqrt(self.running_var + self.eps) * self.weight

        center = tmp_weight.view(1, -1, 1, 1) * mid + tmp_bias.view(1, -1, 1, 1)
        deviation = tmp_weight_abs.view(1, -1, 1, 1) * diff
        lower = center - deviation
        upper = center + deviation
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
        self.forward_value = None
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
        assert torch.isclose(x[1].lower, x[1].upper, 1e-8).all()  # weight perturbation is not support so far
        lA_y = uA_y = lA_bias = uA_bias = None
        weight = x[1].forward_value

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
                sum_bias = (last_A.sum((3, 4)) * x[2].forward_value).sum(2)
            else:
                sum_bias = 0
            return next_A, sum_bias

        lA_x, lbias = _bound_oneside(last_lA)
        uA_x, ubias = _bound_oneside(last_uA)
        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def interval_propagate(self, *v, eps=None, norm=np.inf, C=None):
        assert torch.isclose(v[1][1], v[1][0], 1e-8).all()  # weight perturbation is not support so far
        h_L, h_U = v[0]
        weight = v[1][0]
        if norm == np.inf:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = weight.abs()
            deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            # L2 norm
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
            next_A = F.interpolate(last_A.view(shape[0] * shape[1], *shape[2:]), scale_factor=self.kernel_size)
            next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
            return next_A, 0

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)
        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v, eps=None, norm=None, C=None):
        h_L, h_U = v[0]
        shape = h_U.size()
        h_L = super(BoundAvgPool2d, self).forward(h_L)
        h_U = super(BoundAvgPool2d, self).forward(h_U)

        h_L = h_L.view(shape[0], *h_L.shape[0:])
        h_U = h_U.view(shape[0], *h_U.shape[0:])
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
        h_L = [_v[0] for _v in v]
        h_U = [_v[1] for _v in v]
        return self.forward(*h_L), self.forward(*h_U)

    def bound_backward(self, last_lA, last_uA, *x):
        if self.axis < 0:
            self.axis = len(self.forward_value.shape) + self.axis
        assert (self.axis > 0)

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return torch.split(last_A, self.input_size, dim=self.axis + 1)

        uA = _bound_oneside(last_uA)
        lA = _bound_oneside(last_lA)
        return [(lA[i], uA[i]) for i in range(len(uA))], 0, 0

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
        x_lw, x_lb, x_uw, x_ub = self._broadcast_forward(dim_in, x, self.forward_value.shape)
        y_lw, y_lb, y_uw, y_ub = self._broadcast_forward(dim_in, y, self.forward_value.shape)
        lw, lb = x_lw + y_lw, x_lb + y_lb
        uw, ub = x_uw + y_uw, x_ub + y_ub
        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, x, y):
        assert (not isinstance(y, torch.Tensor))

        if isinstance(y, Hull):
            eps = y.eps
            batch_size, length = self.x_shape[0], self.x_shape[1]
            mask = y.mask.unsqueeze(-1)

            # FIXME share this part with BoundLinear
            inf = 1e20
            lower = torch.min(y.points * mask + (inf) * (1 - mask), dim=2).values
            upper = torch.max(y.points * mask + (-inf) * (1 - mask), dim=2).values

            lower, upper = x[0] + lower, x[1] + upper

            lower = self.forward_value - (self.forward_value - lower) * eps
            upper = self.forward_value + (upper - self.forward_value) * eps

            return lower, upper
            
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
        x_lw, x_lb, x_uw, x_ub = self._broadcast_forward(dim_in, x, self.forward_value.shape)
        y_lw, y_lb, y_uw, y_ub = self._broadcast_forward(dim_in, y, self.forward_value.shape)
        lw, lb = x_lw - y_uw, x_lb - y_ub
        uw, ub = x_uw - y_lw, x_ub - y_lb
        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, x, y):
        return x[0] - y[1], x[1] - y[0]

class BoundPad(ConstantPad2d):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        padding = attr['pads'][2:4] + attr['pads'][6:8]
        value = attr['value'].to(device)
        super(BoundPad, self).__init__(padding=padding, value=value)
        self.input_name = input_name
        self.output_name = []
        self.name = name
        self.ori_name = ori_name
        self.forward_value = None
        self.bounded = False
        self.IBP_rets = None

    def bound_backward(self, last_lA, last_uA, x):
        return [(last_lA, last_uA)], 0, 0

    def interval_propagate(self, *v, norm=None):
        h_L, h_U = v[0]
        return h_L, h_U

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
                mask = torch.gt(last_A, 0.).to(torch.float)
                if sign == -1:
                    _A = last_A * (mask * self.lw.unsqueeze(1) + (1 - mask) * self.uw.unsqueeze(1))
                    _bias = last_A * (mask * self.lb.unsqueeze(1) + (1 - mask) * self.ub.unsqueeze(1))
                elif sign == 1:
                    _A = last_A * (mask * self.uw.unsqueeze(1) + (1 - mask) * self.lw.unsqueeze(1))
                    _bias = last_A * (mask * self.ub.unsqueeze(1) + (1 - mask) * self.lb.unsqueeze(1))

                while len(_bias.shape) > 2:
                    _bias = torch.sum(_bias, dim=-1)
            else:
                mask = torch.gt(last_A, 0.).to(torch.float)
                if sign == -1:
                    _A = last_A * (mask * self.lw.unsqueeze(0).unsqueeze(1) +
                                   (1 - mask) * self.uw.unsqueeze(0).unsqueeze(1))
                    _bias = last_lA * (mask * self.lb.unsqueeze(0).unsqueeze(1) +
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
                mask = torch.gt(self.lw.unsqueeze(1), 0.).to(torch.float)
                lw = self.lw.unsqueeze(1) * (mask * input.lw + (1 - mask) * input.uw)
                mask = torch.lt(self.uw.unsqueeze(1), 0.).to(torch.float)
                uw = self.uw.unsqueeze(1) * (mask * input.lw + (1 - mask) * input.uw)
            else:
                lw = uw = None
            mask = torch.gt(self.lw, 0.).to(torch.float)
            lb = self.lw * (mask * input.lb + (1 - mask) * input.ub) + self.lb
            mask = torch.lt(self.uw, 0.).to(torch.float)
            ub = self.uw * (mask * input.lb + (1 - mask) * input.ub) + self.ub            
        else:
            if input.lw is not None:
                mask = torch.gt(self.lw.unsqueeze(0), 0.).to(torch.float)
                lw = self.lw.unsqueeze(0) * (mask * input.lw + (1 - mask) * input.uw)
                mask = torch.lt(self.uw.unsqueeze(0), 0.).to(torch.float)
                uw = self.uw.unsqueeze(0) * (mask * input.lw + (1 - mask) * input.uw)
            else:
                lw = uw = None
            mask = torch.gt(self.lw, 0.).to(torch.float)
            lb = self.lw * (mask * input.lb + (1 - mask) * input.ub) + self.lb
            mask = torch.lt(self.uw, 0.).to(torch.float)
            ub = self.uw * (mask * input.lb + (1 - mask) * input.ub) + self.ub

        return LinearBound(lw, lb, uw, ub, None, None)

class BoundReLU(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device, bound_opts=None):
        super(BoundReLU, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.nonlinear = True
        self.bound_opts = bound_opts

    def forward(self, input):
        return F.relu(input)

    # linear relaxation for nonlinear functions
    def bound_relax(self, input):
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
        upper_d = upper_d.unsqueeze(1)

        if self.bound_opts == "same-slope":
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.bound_opts == "zero-lb":
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float()
        elif self.bound_opts == "one-lb":
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float()
        else:
            lower_d = (upper_d > 0.5).float()

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
        return F.relu(h_L), F.relu(h_U)

class BoundTanh(Tanh, BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundTanh, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

        self._precompute()

    def dtanh(self, x):
        # to avoid bp error when cosh is too large
        x_limit = torch.tensor(10., device=x.device)
        mask = torch.lt(torch.abs(x), x_limit).to(torch.float32)
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
        for _upper in range(0, 105):
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
        for _lower in range(0, 105):
            lower = torch.tensor(lower * -0.01)
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
        k = self.dtanh(m)
        self._add_linear(mask=self.mask_neg, type='lower', k=k, x0=m, y0=torch.tanh(m))
        # upper bound for positive
        self._add_linear(mask=self.mask_pos, type='upper', k=k, x0=m, y0=torch.tanh(m))

        # upper bound for negative
        k = (torch.tanh(input.upper) - torch.tanh(input.lower)) / \
            (input.upper - input.lower + epsilon)
        self._add_linear(mask=self.mask_neg, type='upper', k=k, x0=input.lower, y0=torch.tanh(input.lower))
        # lower bound for positive
        self._add_linear(mask=self.mask_pos, type='lower', k=k, x0=input.lower, y0=torch.tanh(input.lower))

        # bounds for both
        max_iter = 5

        lower, upper = input.lower.detach(), input.upper.detach()

        if torch.min(lower) > -1. and torch.max(upper) < 1.:
            d = torch.index_select(
                self.d_lower, 0,
                torch.max(
                    torch.zeros_like(upper, dtype=torch.long).reshape(-1),
                    (upper / 0.01).to(torch.long).reshape(-1)
                ) + 1
            ).reshape(upper.shape)
            k = (torch.tanh(d) - torch.tanh(upper)) / (d - upper + epsilon)
            self._add_linear(mask=self.mask_both, type='lower', k=k, x0=d, y0=torch.tanh(d))

            d = torch.index_select(
                self.d_upper, 0,
                torch.max(
                    torch.zeros_like(lower, dtype=torch.long).reshape(-1),
                    (lower / -0.01).to(torch.long).reshape(-1)
                ) + 1
            ).reshape(upper.shape)
            k = (torch.tanh(d) - torch.tanh(lower)) / (d - lower + epsilon)
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
            lower = torch.tensor(lower * -0.01)
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
        self._add_linear(mask=self.mask_neg, type='lower', k=k, x0=m, y0=self.forward(m))
        # upper bound for positive
        self._add_linear(mask=self.mask_pos, type='upper', k=k, x0=m, y0=self.forward(m))

        # upper bound for negative
        k = (self.forward(input.upper) - self.forward(input.lower)) / \
            (input.upper - input.lower + epsilon)
        self._add_linear(mask=self.mask_neg, type='upper', k=k, x0=input.lower, y0=self.forward(input.lower))
        # lower bound for positive
        self._add_linear(mask=self.mask_pos, type='lower', k=k, x0=input.lower, y0=self.forward(input.lower))

        # bounds for both
        max_iter = 5

        lower, upper = input.lower.detach(), input.upper.detach()

        if torch.min(lower) > -10. and torch.max(upper) < 10.:
            d = torch.index_select(
                self.d_lower, 0,
                torch.max(
                    torch.zeros_like(upper, dtype=torch.long).reshape(-1),
                    (upper / 0.01).to(torch.long).reshape(-1)
                ) + 1
            ).reshape(upper.shape)
            k = (torch.sigmoid(d) - torch.sigmoid(upper)) / (d - upper + epsilon)
            self._add_linear(mask=self.mask_both, type='lower', k=k, x0=d, y0=torch.sigmoid(d))

            d = torch.index_select(
                self.d_upper, 0,
                torch.max(
                    torch.zeros_like(lower, dtype=torch.long).reshape(-1),
                    (lower / -0.01).to(torch.long).reshape(-1)
                ) + 1
            ).reshape(upper.shape)
            k = (torch.sigmoid(d) - torch.sigmoid(lower)) / (d - lower + epsilon)
            self._add_linear(mask=self.mask_both, type='upper', k=k, x0=d, y0=torch.sigmoid(d))
        else:
            # lower bound for both
            with torch.no_grad():
                diff = lambda d: (self.forward(upper) - self.forward(d)) / (upper - d + epsilon) - dsigmoid(d)
                d = lower / 2
                _l = lower
                _u = torch.zeros(lower.shape, device=self.device)
                for t in range(max_iter):
                    v = diff(d)
                    mask_p = torch.gt(v, 0).to(torch.float)
                    _l = d * mask_p + _l * (1 - mask_p)
                    _u = d * (1 - mask_p) + _u * mask_p
                    d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
                k = (self.forward(d) - self.forward(upper)) / (d - upper + epsilon)
            self._add_linear(mask=self.mask_both, type='lower', k=k, x0=d, y0=self.forward(d))

            # upper bound for both
            with torch.no_grad():
                diff = lambda d: (self.forward(d) - self.forward(lower)) / (d - lower + epsilon) - dsigmoid(d)
                d = upper / 2
                _l = torch.zeros(lower.shape, device=self.device)
                _u = upper
                for t in range(max_iter):
                    v = diff(d)
                    mask_p = torch.gt(v, 0).to(torch.float)
                    _l = d * (1 - mask_p) + _l * mask_p
                    _u = d * mask_p + _u * (1 - mask_p)
                    d = (d + _u) / 2 * (1 - mask_p) + (d + _l) / 2 * mask_p
                k = (self.forward(d) - self.forward(lower)) / (d - lower + epsilon)
            self._add_linear(mask=self.mask_both, type='upper', k=k, x0=d, y0=self.forward(d))

class BoundExp(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundExp, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)

    def forward(self, input):
        return torch.exp(input)

    def bound_forward(self, dim_in, input):
        epsilon = 1e-12

        m = torch.min((input.lower + input.upper) / 2, input.lower + 0.99)

        kl = torch.exp(m)
        lw = input.lw * kl.unsqueeze(1)
        lb = kl * (input.lb - m + 1)

        ku = (torch.exp(input.upper) - torch.exp(input.lower)) / (input.upper - input.lower + epsilon)
        uw = input.uw * ku.unsqueeze(1)
        ub = input.ub * ku - ku * input.lower + torch.exp(input.lower)

        return LinearBound(lw, lb, uw, ub, None, None)

    def bound_relax(self, input):
        epsilon = 1e-12
        m = torch.min((input.lower + input.upper) / 2, input.lower + 0.99)
        k = torch.exp(m)
        self._add_linear(mask=None, type='lower', k=k, x0=m, y0=torch.exp(m))
        k = (torch.exp(input.upper) - torch.exp(input.lower)) / (input.upper - input.lower + epsilon)
        self._add_linear(mask=None, type='upper', k=k, x0=input.lower, y0=torch.exp(input.lower))

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
        return torch.reciprocal(h_U.to(torch.float32)), torch.reciprocal(h_L.to(torch.float32))

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
            lb = ub = self.forward(x)
            lw = uw = torch.zeros_like(lb.unsqueeze(1), device=x.device)
        return LinearBound(lw, lb, uw, ub, None, None)

class BoundSqueeze(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundSqueeze, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.axes = attr['axes']
        assert (len(self.axes) == 1)
        self.axes = self.axes[0]

    def forward(self, x):
        return x.squeeze(self.axes)

    def bound_backward(self, last_lA, last_uA):
        raise NotImplementedError

class BoundConstantOfShape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundConstantOfShape, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.value = attr['value'].to(device)

    def forward(self, x):
        self.x = x
        self.from_input = True
        return torch.ones(list(x), device=self.device) * self.value

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
        lb = ub = self.forward_value
        lw = uw = torch.zeros(self.x[0], dim_in, *self.x[1:], device=self.device)
        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, *v):
        return self.forward_value, self.forward_value

class BoundConstant(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundConstant, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.value = attr['value'].to(self.device)

    def forward(self):
        return self.value

    def bound_backward(self, last_lA, last_uA):
        def _bound_oneside(A):
            if A is None:
                return None
            while len(A.shape) > 2:
                A = torch.sum(A, dim=-1)
            return A * self.value
        lbias = _bound_oneside(last_lA)
        ubias = _bound_oneside(last_uA)
        return [], lbias, ubias

    def bound_forward(self, dim_in):
        lw = uw = torch.zeros(dim_in, device=self.device)
        lb = ub = self.value
        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, *v, norm=None):
        return self.value, self.value

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

    def forward(self, x, indices):
        if isinstance(x, torch.Size):
            x = torch.tensor(x, device=self.device)
            if indices != 0:
                self.from_input = False
        indices = indices.to(self.device)
        self.input_shape = x.shape
        self.indices = indices
        if len(indices.shape) == 0:
            selected = torch.index_select(x, dim=self.axis, index=indices).squeeze(self.axis)
            return selected
        raise NotImplementedError

    def bound_backward(self, last_lA, last_uA, x, indices):
        def _bound_oneside(A):
            if A is None:
                return None
            assert (len(self.indices.shape) == 0)

            if self.from_input:
                d = 1
            else:
                d = 2 if self.axis == 0 else 1
            A = A.unsqueeze(self.axis + d)
            tensors = []
            if self.indices > 0:
                shape_pre = list(A.shape)
                shape_pre[self.axis + d] *= self.indices
                tensors.append(torch.zeros(shape_pre, device=self.device))
            tensors.append(A)
            if self.input_shape[self.axis] - self.indices - 1 > 0:
                shape_next = list(A.shape)
                shape_next[self.axis + d] *= self.input_shape[self.axis] - self.indices - 1
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
        if isinstance(v[0], list):
            assert(0)
            batch_size = len(v[0])
            hull = []
            for t in range(batch_size):
                _hull = []
                for option in v[0][t]:
                    _hull.append(self.forward(option, self.indices))
                hull.append(_hull)
            return hull

        if isinstance(v[0][0], torch.Tensor):
            h_L, h_U = v[0]
        else:
            h_L, h_U = torch.tensor(v[0][0]), torch.tensor(v[0][1])
        return torch.index_select(h_L.to(self.device), \
                                  dim=self.axis, index=self.indices).squeeze(self.axis), \
               torch.index_select(h_U.to(self.device), \
                                  dim=self.axis, index=self.indices).squeeze(self.axis)


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

    def _relax(self, x, y):
        x_l, x_u = x.lower, x.upper
        y_l, y_u = y.lower, y.upper

        # broadcast
        for k in [1, -1]:
            x_l = x_l + k * y_l
            x_u = x_u + k * y_u
        for k in [1, -1]:
            y_l = y_l + k * x_l
            y_u = y_u + k * x_u

        return self.get_bound_mul(x_l, x_u, y_l, y_u)

    def bound_backward(self, last_lA, last_uA, x, y):
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self._relax(x, y)

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
            mask = torch.gt(last_A, 0.).to(torch.float)
            A_x = last_A * (mask * alpha_pos + (1 - mask) * alpha_neg)
            mask = torch.gt(last_A, 0.).to(torch.float)
            A_y = last_A * (mask * beta_pos + (1 - mask) * beta_neg)
            last_A = last_A.reshape(last_A.shape[0], last_A.shape[1], -1)
            mask = torch.gt(last_A, 0.).to(torch.float)
            bias = torch.bmm(mask * last_A, gamma_pos).squeeze(-1) + \
                    torch.bmm((1 - mask) * last_A, gamma_neg).squeeze(-1)
            A_x = self._broadcast_backward(A_x, x)
            A_y = self._broadcast_backward(A_y, y)
            return A_x, A_y, bias
        
        lA_x, lA_y, lbias = _bound_oneside(
            last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
        uA_x, uA_y, ubias = _bound_oneside(
            last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    def bound_forward(self, dim_in, x, y):
        x_lw, x_lb, x_uw, x_ub = self._broadcast_forward(dim_in, x, self.forward_value.shape)
        y_lw, y_lb, y_uw, y_ub = self._broadcast_forward(dim_in, y, self.forward_value.shape)

        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self._relax(x, y)

        mask = torch.gt(alpha_l.unsqueeze(1), 0.).to(torch.float)
        lw = alpha_l.unsqueeze(1) * (mask * x_lw + (1 - mask) * x_uw)
        mask = torch.gt(alpha_l, 0.).to(torch.float)
        lb = alpha_l * (mask * x_lb + (1 - mask) * x_ub)
        mask = torch.gt(beta_l.unsqueeze(1), 0.).to(torch.float)
        lw += beta_l.unsqueeze(1) * (mask * y_lw + (1 - mask) * y_uw)
        mask = torch.gt(beta_l, 0.).to(torch.float)
        lb += beta_l * (mask * y_lb + (1 - mask) * y_ub)
        lb += gamma_l

        mask = torch.lt(alpha_u.unsqueeze(1), 0.).to(torch.float)
        uw = alpha_u.unsqueeze(1) * (mask * x_lw + (1 - mask) * x_uw)
        mask = torch.lt(alpha_u, 0.).to(torch.float)
        ub = alpha_u * (mask * x_lb + (1 - mask) * x_ub)
        mask = torch.lt(beta_u.unsqueeze(1), 0.).to(torch.float)
        uw += beta_u.unsqueeze(1) * (mask * y_lw + (1 - mask) * y_uw)
        mask = torch.lt(beta_u, 0.).to(torch.float)
        ub += beta_u * (mask * y_lb + (1 - mask) * y_ub)
        ub += gamma_u

        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, *v):
        x, y = v[0], v[1]
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
        return BoundMul.interval_propagate(None, x, y_r)

    def _convert_to_mul(self, x, y):
        try:
            reciprocal = BoundReciprocal(self.input_name, self.name + '/reciprocal', self.ori_name, {}, [], 0, self.device)
            mul = BoundMul(self.input_name, self.name + '/mul', self.ori_name, {}, [], 0, self.device)
        except:
            # to make it compatible with previous code
            reciprocal = BoundReciprocal(self.input_name, self.name + '/reciprocal', None, {}, [], 0, self.device)
            mul = BoundMul(self.input_name, self.name + '/mul', None, {}, [], 0, self.device)

        mul.forward_value = mul(self.x, torch.reciprocal(self.y.to(torch.float32)))
        y_r = copy.copy(y)
        if isinstance(y_r, LinearBound):
            y_r = y_r._replace(lower=1. / y.upper, upper=1. / y.lower)
        else:
            y_r.lower = 1. / y.upper
            y_r.upper = 1. / y.lower
            y_r.forward_value = 1. / self.y
        return reciprocal, mul, y_r

class BoundMatMul(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundMatMul, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.nonlinear = True

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        self.x = x
        self.y = y
        return x.matmul(y)

    def _relax(self, x, y):
        x_shape, y_shape = self.x_shape, self.y_shape
        if len(x_shape) == len(y_shape):
            # (x_1, x_2, ..., x_{n-1}, y_{n}, x_n)
            repeat = (1,) * (len(x_shape) - 1) + y_shape[-1:] + (1,)
            x_l = x.lower.unsqueeze(-2).repeat(*repeat)
            x_u = x.upper.unsqueeze(-2).repeat(*repeat)

            # (x_1, x_2, ..., x_{n-1}, y_n, y_{n-1})
            shape = x_shape[:-1] + (y_shape[-1], y_shape[-2])
            repeat = (1,) * (len(x_shape) - 2) + (x_shape[-2], 1)
            y_l = y.lower.transpose(-1, -2).repeat(*repeat).reshape(*shape)
            y_u = y.upper.transpose(-1, -2).repeat(*repeat).reshape(*shape)
        elif len(y_shape) == 2:
            # (x_1, x_2, ..., x_{n-1}, y_2, x_n)
            repeat = (1,) * (len(x_shape) - 1) + y_shape[-1:] + (1,)
            x_l = x.lower.unsqueeze(-2).repeat(*repeat)
            x_u = x.upper.unsqueeze(-2).repeat(*repeat)

            # (x_1, x_2, ..., x_{n-1}, y_2, y_1)
            shape = x_shape[:-1] + y_shape[1:] + y_shape[:1]
            repeat = (np.prod(x_shape[:-1]),) + (1,)
            y_l = y.lower.transpose(0, 1).repeat(*repeat).reshape(*shape)
            y_u = y.upper.transpose(0, 1).repeat(*repeat).reshape(*shape)

        return BoundMul.get_bound_mul(x_l, x_u, y_l, y_u)

    def bound_backward(self, last_lA, last_uA, x, y):
        if len(self.y_shape) == 2:
            # y is a fixed weight
            if torch.sum(torch.abs(y.upper - y.lower)) < 1e-6:
                lA_x = last_lA.matmul(y.lower.t())
                uA_x = last_uA.matmul(y.lower.t())
                return [(lA_x, uA_x), (None, None)], 0., 0.

        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self._relax(x, y)

        batch_size = gamma_l.shape[0]
        alpha_l, alpha_u = alpha_l.unsqueeze(1), alpha_u.unsqueeze(1)
        beta_l, beta_u = beta_l.unsqueeze(1), beta_u.unsqueeze(1)
        gamma_l = torch.sum(gamma_l, dim=-1).reshape(batch_size, -1, 1)
        gamma_u = torch.sum(gamma_u, dim=-1).reshape(batch_size, -1, 1)

        x_shape, y_shape = self.x_shape, self.y_shape

        if len(x.forward_value.shape) == len(y.forward_value.shape):
            dim_y = -3
        elif len(y.forward_value.shape) == 2:
            dim_y = list(range(2, 2 + len(x_shape) - 2))
        else:
            raise NotImplementedError

        mask = torch.gt(last_uA.unsqueeze(-1), 0.).to(torch.float)
        uA_x = torch.sum(last_uA.unsqueeze(-1) * (
                mask * alpha_u + (1 - mask) * alpha_l), dim=-2)
        uA_y = torch.sum(last_uA.unsqueeze(-1) * (
                mask * beta_u + (1 - mask) * beta_l), dim=dim_y)
        uA_y = uA_y.transpose(-1, -2)

        mask = torch.gt(last_lA.unsqueeze(-1), 0.).to(torch.float)
        lA_x = torch.sum(last_lA.unsqueeze(-1) * (
                mask * alpha_l + (1 - mask) * alpha_u), dim=-2)
        lA_y = torch.sum(last_lA.unsqueeze(-1) * (
                mask * beta_l + (1 - mask) * beta_u), dim=dim_y)
        lA_y = lA_y.transpose(-1, -2)

        _last_uA = last_uA.reshape(last_uA.shape[0], last_uA.shape[1], -1)
        _last_lA = last_lA.reshape(last_lA.shape[0], last_lA.shape[1], -1)
        mask = torch.gt(_last_uA, 0.).to(torch.float)
        ubias = torch.bmm(mask * _last_uA, gamma_u) + torch.bmm((1 - mask) * _last_uA, gamma_l)
        mask = torch.gt(_last_lA, 0.).to(torch.float)
        lbias = torch.bmm(mask * _last_lA, gamma_l) + torch.bmm((1 - mask) * _last_lA, gamma_u)
        ubias, lbias = ubias.squeeze(-1), lbias.squeeze(-1)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    def bound_forward(self, dim_in, x, y):
        if len(self.y_shape) == 2:
            # y is a fixed weight
            if torch.sum(torch.abs(y.upper - y.lower)) < 1e-6:
                return BoundLinear.bound_forward(None, dim_in, x, y.lower.t())

        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self._relax(x, y)

        x_shape, y_shape = self.x_shape, self.y_shape

        lw = torch.zeros(x_shape[0], dim_in, *self.forward_value.shape[1:], device=self.device)
        uw = lw.clone()
        lb = torch.zeros(self.forward_value.shape, device=self.device)
        ub = lb.clone()

        mask = torch.gt(alpha_l.unsqueeze(1), 0.).to(torch.float32)
        lw += torch.sum(alpha_l.unsqueeze(1) * (
                mask * x.lw.unsqueeze(-2) + (1 - mask) * x.uw.unsqueeze(-2))
                        , dim=-1)
        mask = torch.gt(alpha_l, 0.).to(torch.float32)
        lb += torch.sum(alpha_l * (
                mask * x.lb.unsqueeze(-2) + (1 - mask) * x.ub.unsqueeze(-2))
                        , dim=-1)

        mask = torch.lt(alpha_u.unsqueeze(1), 0.).to(torch.float32)
        uw += torch.sum(alpha_u.unsqueeze(1) * (
                mask * x.lw.unsqueeze(-2) + (1 - mask) * x.uw.unsqueeze(-2))
                        , dim=-1)
        mask = torch.lt(alpha_u, 0.).to(torch.float32)
        ub += torch.sum(alpha_u * (
                mask * x.lb.unsqueeze(-2) + (1 - mask) * x.ub.unsqueeze(-2))
                        , dim=-1)

        if len(self.x_shape) == len(self.y_shape):
            y_lw = y.lw.unsqueeze(-3).transpose(-1, -2)
            y_uw = y.uw.unsqueeze(-3).transpose(-1, -2)
            y_lb = y.lb.unsqueeze(-3).transpose(-1, -2)
            y_ub = y.ub.unsqueeze(-3).transpose(-1, -2)
        else:
            y_lw = y.lw.transpose(1, 2)
            y_uw = y.uw.transpose(1, 2)
            y_lb = y.lb.transpose(0, 1)
            y_ub = y.ub.transpose(0, 1)

        mask = torch.gt(beta_l.unsqueeze(1), 0.).to(torch.float32)
        lw += torch.sum(beta_l.unsqueeze(1) * (
                mask * y_lw + (1 - mask) * y_uw
        ), dim=-1)
        mask = torch.gt(beta_l, 0.).to(torch.float32)
        lb += torch.sum(beta_l * (
                mask * y_lb + (1 - mask) * y_ub
        ), dim=-1)
        lb += torch.sum(gamma_l, dim=-1)

        mask = torch.lt(beta_u.unsqueeze(1), 0.).to(torch.float32)
        uw += torch.sum(beta_u.unsqueeze(1) * (
                mask * y_lw + (1 - mask) * y_uw
        ), dim=-1)
        mask = torch.lt(beta_u, 0.).to(torch.float32)
        ub += torch.sum(beta_u * (
                mask * y_lb + (1 - mask) * y_ub
        ), dim=-1)
        ub += torch.sum(gamma_u, dim=-1)

        return LinearBound(lw, lb, uw, ub, None, None)

    def interval_propagate(self, *v):
        if len(self.y_shape) == 2:
            # y is a fixed weight
            if not isinstance(v[1], list) and torch.sum(torch.abs(v[1][1] - v[1][0])) < 1e-12:
                bound = BoundLinear(self.input_name, None, None, None, None, None, self.device)
                bound.forward_value = self.forward_value
                w = v[1][0].t()
                return bound.interval_propagate(v[0], (w, w))

        x = LinearBound(None, None, None, None, v[0][0], v[0][1])
        y = LinearBound(None, None, None, None, v[1][0], v[1][1])

        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self._relax(x, y)

        mask = torch.gt(alpha_l, 0.).to(torch.float32)
        lower = torch.sum(alpha_l * \
                          (mask * x.lower.unsqueeze(-2) + (1 - mask) * x.upper.unsqueeze(-2)), dim=-1)
        mask = torch.lt(alpha_u, 0.).to(torch.float32)
        upper = torch.sum(alpha_u * \
                          (mask * x.lower.unsqueeze(-2) + (1 - mask) * x.upper.unsqueeze(-2)), dim=-1)

        if len(self.x_shape) == len(self.y_shape):
            y_lower = y.lower.unsqueeze(-3).transpose(-1, -2)
            y_upper = y.upper.unsqueeze(-3).transpose(-1, -2)
        else:
            y_lower = y.lower.transpose(0, 1)
            y_upper = y.upper.transpose(0, 1)

        mask = torch.gt(beta_l, 0.).to(torch.float32)
        lower += torch.sum(beta_l * (mask * y_lower + (1 - mask) * y_upper), dim=-1)
        lower += torch.sum(gamma_l, dim=-1)

        mask = torch.lt(beta_u, 0.).to(torch.float32)
        upper += torch.sum(beta_u * (mask * y_lower + (1 - mask) * y_upper), dim=-1)
        upper += torch.sum(gamma_u, dim=-1)

        return lower, upper

class BoundCast(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, device):
        super(BoundCast, self).__init__(input_name, name, ori_name, attr, inputs, output_index, device)
        self.to = attr['to']
        assert (self.to == 1)  # to float

    def forward(self, x):
        assert (self.to == 1)
        return x.to(torch.float32)

    def bound_backward(self, last_lA, last_uA, x):
        return [(last_lA, last_uA)], 0, 0

    def bound_forward(self, dim_in, x):
        return LinearBound(
            x.lw.to(torch.float32), x.lb.to(torch.float32),
            x.uw.to(torch.float32), x.ub.to(torch.float32), None, None)

    def interval_propagate(self, *v):
        return v[0][0].to(torch.float32), v[0][1].to(torch.float32)

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
        self.mask = res / (x + 1e-12)
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
        self.register_parameter('param', torch.nn.Parameter(value))

    def forward(self):
        return self.param

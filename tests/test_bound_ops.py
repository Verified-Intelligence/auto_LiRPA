"""Test classes for bound operators"""
import torch
import os
from auto_LiRPA.bound_ops import *
from auto_LiRPA.utils import LinearBound
from testcase import TestCase

"""Dummy node for testing"""
class Dummy:
    def __init__(self, lower, upper=None, perturbed=False):
        self.lower = lower
        self.upper = upper if upper is not None else lower
        self.perturbed = perturbed
        self.output_shape = lower.shape

class TestBoundOp(TestCase): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName, 
            seed=1, ref_path='data/bound_ops_data',
            generate=generate)

    def test(self):
        device = 'cpu'
        batch_size = 5
        dim_final = 7
        dim_output = 11
        dim_input = 11
        
        # multiplication of [batch_size, dim_input] and [dim_output, dim_input]^T
        weight = torch.randn(dim_output, dim_input, device=device)
        bias = torch.randn(dim_output, device=device)
        data_in = torch.randn(batch_size, dim_input, device=device)
        data_in_delta = torch.randn(batch_size, dim_input, device=device)
        dummy_in = Dummy(data_in - torch.abs(data_in_delta), data_in + torch.abs(data_in_delta), True)
        dummy_weight = Dummy(weight)
        dummy_bias = Dummy(bias)

        op = BoundLinear(
            input_name=[None, None, None], 
            name=None, ori_name=None, attr=None, 
            inputs=[dummy_in, dummy_weight, dummy_bias],
            output_index=0, options={}, device=device)

        # test `forward`
        data_out = op(data_in, weight, bias)
        self.assertEqual(data_out, data_in.matmul(weight.t()) + bias)

        # test `bound_backward`
        # The `transpose` here to make the randomization consistent with the previous reference.
        # It can be removed once a new reference is generated.
        last_lA = torch.randn(batch_size, dim_final, dim_output, device=device).transpose(0, 1)
        last_uA = torch.randn(batch_size, dim_final, dim_output, device=device).transpose(0, 1)
        A, lbias, ubias = op.bound_backward(last_lA, last_uA, *op.inputs)
        self.assertEqual(A[0][0], last_lA.matmul(weight))
        self.assertEqual(A[0][1], last_uA.matmul(weight))
        self.assertEqual(lbias, last_lA.matmul(bias))
        self.assertEqual(ubias, last_uA.matmul(bias))

        # test `bound_forward`
        # note that the upper bound may be actually smaller than the lower bound
        # in these dummy linear bounds
        bound_in = LinearBound(
            lw=torch.randn(batch_size, dim_final, dim_input, device=device),
            lb=torch.randn(batch_size, dim_input, device=device),
            uw=torch.randn(batch_size, dim_final, dim_input, device=device),
            ub=torch.randn(batch_size, dim_input, device=device),
            lower=None, upper=None)
        bound_weight = LinearBound(None, None, None, None, dummy_weight.lower, dummy_weight.upper)
        bound_bias = LinearBound(None, None, None, None, dummy_bias.lower, dummy_bias.upper)
        bound_out = op.bound_forward(dim_final, bound_in, bound_weight, bound_bias)
        self.assertEqual(bound_out.lw, 
            bound_in.lw.matmul(weight.t().clamp(min=0)) + bound_in.uw.matmul(weight.t().clamp(max=0)))
        self.assertEqual(bound_out.uw, 
            bound_in.uw.matmul(weight.t().clamp(min=0)) + bound_in.lw.matmul(weight.t().clamp(max=0)))
        self.assertEqual(bound_out.lb, 
            bound_in.lb.matmul(weight.t().clamp(min=0)) + bound_in.ub.matmul(weight.t().clamp(max=0)) + bias)
        self.assertEqual(bound_out.ub, 
            bound_in.ub.matmul(weight.t().clamp(min=0)) + bound_in.lb.matmul(weight.t().clamp(max=0)) + bias)

        # test `interval_propagate`
        bound_in = (
            torch.randn(*data_in.shape, device=device), 
            torch.randn(*data_in.shape, device=device))
        bound_weight = (bound_weight.lower, bound_weight.upper)
        bound_bias = (bound_bias.lower, bound_bias.upper)
        bound_out = op.interval_propagate(bound_in, bound_weight, bound_bias)
        self.assertEqual(bound_out[0], 
            bound_in[0].matmul(weight.t().clamp(min=0)) + bound_in[1].matmul(weight.t().clamp(max=0)) + bias)
        self.assertEqual(bound_out[1], 
            bound_in[1].matmul(weight.t().clamp(min=0)) + bound_in[0].matmul(weight.t().clamp(max=0)) + bias)

        # test weight perturbation
        # `bound_backward`
        ptb_weight = torch.randn(weight.shape)
        op.inputs[1].upper += ptb_weight
        op.inputs[1].perturbed = True
        op.inputs[2].perturbation = None # no perturbation on bias
        A, lbias, ubias = op.bound_backward(last_lA, last_uA, *op.inputs)
        # `interval_propagate`
        bound_weight = (op.inputs[1].lower, op.inputs[1].upper)
        bound_out = op.interval_propagate(bound_in, bound_weight, bound_bias)

        self.result = (A, lbias, ubias, bound_out)

        if self.generate:
            self.save()

        A_ref, lbias_ref, ubias_ref, bound_out_ref = self.reference
        for i in range(3):
            for j in range(2):
                if A_ref[i][j] is not None:
                    ref = A_ref[i][j]
                    # legacy reference
                    if ref.shape[0] == batch_size:
                        ref = ref.transpose(0, 1)
                    self.assertEqual(A[i][j], ref) 
        lbias, ubias = lbias.transpose(0, 1), ubias.transpose(0, 1)
        self.assertEqual(lbias, lbias_ref)
        self.assertEqual(ubias, ubias_ref)
        self.assertEqual(bound_out[0], bound_out_ref[0]) and equal(bound_out[1], bound_out_ref[1])

if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestBoundOp(generate=False)
    testcase.test()
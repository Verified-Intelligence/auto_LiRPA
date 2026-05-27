"""Tests for BoundSoftmax LSE bound_backward (Wei et al., AISTATS 2023)."""
import torch
import torch.nn as nn
import itertools
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.operators.softmax import BoundSoftmax
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class LinearSoftmax(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


class PureSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.softmax(x, dim=self.dim)


class TestSoftmaxBackward(TestCase):
    def __init__(self, methodName='runTest', generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName, seed=42, ref_name='softmax_backward_test_data',
            generate=generate, device=device, dtype=dtype)

    def _verify_bounds(self, model, input_lb, input_ub, lb, ub, atol=1e-5, n_samples=1000):
        """Empirically verify model's output bounds are correct given input bounds."""
        B = input_lb.shape[0]  # Original batch size
        rest_shape = input_lb.shape[1:]

        inputs = torch.rand(n_samples, *input_lb.shape, 
                            device=input_lb.device, 
                            dtype=input_lb.dtype) * (input_ub - input_lb) + input_lb
        inputs_fused = inputs.view(n_samples * B, *rest_shape)

        outputs_fused = model(inputs_fused)

        outputs = outputs_fused.view(n_samples, B, *outputs_fused.shape[1:])

        empirical_lb, empirical_ub = outputs.min(dim=0).values, outputs.max(dim=0).values
        
        if not (empirical_lb - lb >= -atol).all():
            raise AssertionError(f"Lower bound violated. Max: {(lb - empirical_lb).max().item()}")
        if not (empirical_ub - ub <= atol).all():
            raise AssertionError(f"Upper bound violated. Max: {(empirical_ub - ub).max().item()}")

    def _run_bound_test(self, model, input_lb, input_ub, methods=['CROWN'], n_samples=1000):
        """Compute bounds, verify empirically, and test batched vs individual consistency."""
        model = model.to(device=self.default_device, dtype=self.default_dtype)
        lirpa_model = BoundedModule(model, torch.empty_like(input_lb),
                                    bound_opts={'softmax': 'lse'},
                                    device=self.default_device)
        ptb_data = BoundedTensor(input_lb, PerturbationLpNorm(x_L=input_lb, x_U=input_ub))

        for method in methods:
            lb, ub = lirpa_model.compute_bounds(x=(ptb_data,), method=method)
            self._verify_bounds(model, input_lb, input_ub, lb, ub, n_samples=n_samples)
            
            # Cross-check: Verify batch broadcasting perfectly matches individual runs
            B = input_lb.shape[0]
            if B > 1:
                lirpa_single = BoundedModule(
                    model, torch.empty_like(input_lb[0:1]),
                    bound_opts={'softmax': 'lse'}, device=self.default_device)
                for i in range(B):
                    ptb_i = BoundedTensor(input_lb[i:i+1], PerturbationLpNorm(x_L=input_lb[i:i+1], x_U=input_ub[i:i+1]))
                    lb_i, ub_i = lirpa_single.compute_bounds(x=(ptb_i,), method=method)
                    assert torch.allclose(lb[i:i+1], lb_i, atol=1e-5), f"Batch broadcasting mismatch on lb!"
                    assert torch.allclose(ub[i:i+1], ub_i, atol=1e-5), f"Batch broadcasting mismatch on ub!"

            self.result.append((lb, ub))
        return lb, ub

    def _check_jacobian_and_fd(self, K, bound_width, eps=1e-8, fd_eps=1e-6, batch_size=2):
        """Consolidated method to test mathematical validity and finite differences."""
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)  # Math checks need float64
        
        try:
            lb = torch.randn(batch_size, K)
            ub = lb + torch.rand(batch_size, K) * bound_width
            x0 = (lb + ub) / 2.0
            sm_x0 = torch.softmax(x0, dim=-1)
            mx0 = x0.max(dim=-1, keepdim=True)[0]
            se0 = torch.exp(x0 - mx0).sum(dim=-1, keepdim=True)

            node = BoundSoftmax.__new__(BoundSoftmax)

            # 1. Evaluate Analytical Jacobians
            (D_L, u_L, v_L), b_L = node._softmax_lse_lower(lb, ub, x0)
            (D_U, u_U, v_U), b_U = node._softmax_lse_upper(lb, ub, x0, sm_x0, mx0, se0, eps)

            J_L = torch.diag_embed(D_L) - u_L.unsqueeze(-1) * v_L.unsqueeze(-2)
            J_U = torch.diag_embed(D_U) - u_U.unsqueeze(-1) * v_U.unsqueeze(-2)

            # 2. Tangent Validity Check
            L_x0 = (J_L * x0.unsqueeze(-2)).sum(dim=-1) + b_L
            U_x0 = (J_U * x0.unsqueeze(-2)).sum(dim=-1) + b_U
            assert (L_x0 <= sm_x0 + 1e-10).all()
            assert (U_x0 >= sm_x0 - 1e-10).all()
            self.result.extend([(J_L.float(), b_L.float()), (J_U.float(), b_U.float())])

            # 3. Finite Difference Check
            def _eval(x, mode='lower'):
                if mode == 'lower':
                    (D, u, v), b = node._softmax_lse_lower(lb, ub, x, eps)
                else:
                    sm = torch.softmax(x, dim=-1)
                    mx = x.max(dim=-1, keepdim=True)[0]
                    se = torch.exp(x - mx).sum(dim=-1, keepdim=True)
                    (D, u, v), b = node._softmax_lse_upper(lb, ub, x, sm, mx, se, eps)

                    J = torch.diag_embed(D) - u.unsqueeze(-1) * v.unsqueeze(-2)
                    return (J * x.unsqueeze(-2)).sum(dim=-1) + b

                J_L_fd, J_U_fd = torch.zeros(batch_size, K, K), torch.zeros(batch_size, K, K)
                for i in range(K):
                    e_i = torch.zeros(batch_size, K)
                    e_i[:, i] = fd_eps
                    J_L_fd[:, :, i] = ((_eval(x0 + e_i, 'lower') - _eval(x0 - e_i, 'lower')) / (2 * fd_eps))
                    J_U_fd[:, :, i] = ((_eval(x0 + e_i, 'upper') - _eval(x0 - e_i, 'upper')) / (2 * fd_eps))

                assert (J_L - J_L_fd).abs().max().item() < 1e-4
                assert (J_U - J_U_fd).abs().max().item() < 1e-4
                self.result.append((J_L.float(), J_U.float()))

        finally:
            # Guarantees the precision is reset even if assertions fail
            torch.set_default_dtype(prev_dtype)

    def test(self):
        self.result = []
        batch_size = 2

        # ----- 1. Graph Structure & Mode Routing -----
        model, dummy = LinearSoftmax(4, 3).eval(), torch.zeros(batch_size, 4)
        bm_lse = BoundedModule(model, dummy, bound_opts={'softmax': 'lse'})
        bm_cpx = BoundedModule(model, dummy, bound_opts={'softmax': 'complex', 'fixed_reducemax_index': True})

        assert 'BoundSoftmax' in [n.__class__.__name__ for n in bm_lse.nodes()]
        assert any(n in [n.__class__.__name__ for n in bm_cpx.nodes()] for n in ['BoundReduceMax', 'BoundExp'])

        x = torch.randn(batch_size, 4)
        bm_cpx.compute_bounds(x=(BoundedTensor(x, PerturbationLpNorm(x_L=x - 0.1, x_U=x + 0.1)),), method='CROWN')

        # ----- 2. Mathematical Validity & Jacobian Finite Differences -----
        self._check_jacobian_and_fd(K=4, bound_width=2.0, batch_size=batch_size)
        self._check_jacobian_and_fd(K=3, bound_width=1.0, batch_size=batch_size)

        # ----- 3. Soundness Across Configurations -----
        torch.manual_seed(123)
        for K, e in itertools.product([2, 3, 4, 8], [0.01, 0.1, 0.5]):
            # Standard 2D Input (batch, features), testing default dim=-1
            x_2d = torch.randn(batch_size, K + 1)
            self._run_bound_test(LinearSoftmax(K + 1, K).eval(), x_2d - e, x_2d + e)

            # 3D Input (batch, K, T), testing non-last axis dim=1
            x_3d = torch.randn(batch_size, K, 3) 
            self._run_bound_test(PureSoftmax(dim=1).eval(), x_3d - e, x_3d + e)

        # ----- 4. Corner Evaluation -----
        # Use higher sample count here as the primary soundness spot-check.
        K, eps, x = 4, 0.5, torch.randn(batch_size, 4)
        lb, ub = self._run_bound_test(PureSoftmax().eval(), x - eps, x + eps, n_samples=10000)
        for c in itertools.product([0, 1], repeat=K):
            c_t = torch.tensor(c, dtype=x.dtype).unsqueeze(0)
            y = PureSoftmax().eval()((x - eps) * (1 - c_t) + (x + eps) * c_t)
            assert (lb - 1e-6 <= y).all() and (y <= ub + 1e-6).all()

        # Pure softmax with large eps (higher sample count for extra confidence).
        self._run_bound_test(PureSoftmax().eval(), torch.randn(batch_size, 5) - 2.0, torch.randn(batch_size, 5) + 2.0, n_samples=10000)

        # ----- 5. Tightness Trend (Width decreases with eps) -----
        K, in_dim = 4, 5
        model, x = LinearSoftmax(in_dim, K).eval(), torch.randn(batch_size, in_dim)
        widths = [
            (ub - lb).detach().sum().item()
            for e in [2.0, 1.0, 0.5, 0.1, 0.05, 1e-6] 
            for lb, ub in [self._run_bound_test(model, x - e, x + e)]
        ]

        for i in range(len(widths) - 1):
            assert widths[i] >= widths[i + 1] - 1e-10
        assert widths[-1] < 1e-3

        self.check()


if __name__ == '__main__':
    # Change to generate=True when generating reference results
    testcase = TestSoftmaxBackward(generate=False)
    testcase.setUp()
    testcase.test()
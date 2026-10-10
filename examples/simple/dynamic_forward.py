"""Optimize and read a shared affine relaxation of a small ReLU network.

Run from the repository root: python examples/simple/dynamic_forward.py
The example uses CPU tensors and requires no model or dataset downloads.
"""
import torch
from torch import nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def main():
    dtype = torch.float64
    model = nn.Sequential(nn.Linear(2, 3, dtype=dtype), nn.ReLU(),
                          nn.Linear(3, 2, dtype=dtype)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(
            [[1., 1.], [1., -1.], [-1., .5]], dtype=dtype))
        model[0].bias.copy_(torch.tensor([.25, -.25, .5], dtype=dtype))
        model[2].weight.copy_(torch.tensor(
            [[1., -.5, .25], [-.25, 1., .5]], dtype=dtype))
        model[2].bias.copy_(torch.tensor([.1, -.2], dtype=dtype))

    x_L = torch.tensor([[-1., -.5]], dtype=dtype)
    x_U = torch.tensor([[1., 1.5]], dtype=dtype)
    x = (x_L + x_U) / 2
    bounded_x = BoundedTensor(x, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    # Two queries on the same box: y[0] - y[1] and y[0].
    C = torch.tensor([[[1., -1.], [1., 0.]]], dtype=dtype)
    bm = BoundedModule(model, x, device='cpu', bound_opts={
        'optimize_bound_args': {
            'iteration': 40, 'lr_alpha': .05, 'early_stop_patience': 40},
        'dynamic_forward': {'generators': 'auto'},
    })

    with torch.no_grad():
        # Fix the same pre-activation intervals for both relaxations.
        bm.compute_bounds(x=(bounded_x,), method='CROWN')
        pre = bm.relus[0].inputs[0]
        interm_bounds = {pre.name: (pre.lower.clone(), pre.upper.clone())}
        lo, hi = interm_bounds[pre.name]
        unstable = ((lo < 0) & (hi > 0)).flatten().nonzero(as_tuple=True)[0]
        requested = {bm.final_name: None, pre.name: unstable}

        initial_lb, initial_ub = bm.compute_bounds(
            x=(bounded_x,), C=C, method='dynamic-forward',
            interm_bounds=interm_bounds)
        lb, ub, forms = bm.compute_bounds(
            x=(bounded_x,), C=C, method='dynamic-forward-optimized',
            interm_bounds=interm_bounds, return_generators=True,
            needed_generators_dict=requested)

        # G describes y itself; C shapes the returned bounds, not these rows.
        # G: [batch, output elements, variables]; centre: [batch, elements].
        G, centre = forms[bm.final_name]
        query_G = C @ G
        query_centre = (C @ centre.unsqueeze(-1)).squeeze(-1)
        radius = query_G.abs().sum(-1)
        torch.testing.assert_close(lb, query_centre - radius)
        torch.testing.assert_close(ub, query_centre + radius)

        pre_G, pre_centre = forms[pre.name]
        ranges = bm[bm.input_name[0]].linear.dynamic_variable_ranges
        print('Initial query bounds:', initial_lb, initial_ub)
        print('Optimized query bounds:', lb, ub)
        print('Worst lower bound:', initial_lb.min().item(), '->', lb.min().item())
        print('Output generators G:', G)
        print('Output centre:', centre)
        print('Selected pre-activation rows:', unstable.tolist())
        print('Pre-activation generators:', pre_G)
        print('Pre-activation centre:', pre_centre)
        print('Variable column ranges (start, count):', ranges)
        print('Selected ReLU slopes:', bm.dynamic_forward_slopes)
        print('Checked query bounds = C centre +/- row sums of |C G|.')


if __name__ == '__main__':
    main()

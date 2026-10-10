"""Reverse mode and the slope search. Reading the generators one backward pass
per row (reverse mode) must give the same generators as a pass that carries
every variable (forward mode), under any ReLU slopes; and the bound the slope
search maximizes, and its gradient in the slopes, must be those of the
generators."""
import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA import dynamic_forward_bound
from auto_LiRPA.dynamic_forward_bound import (
    _dynamic_forward_pass, _same_slope_alpha)
from auto_LiRPA.operators.relu import BoundRelu
from auto_LiRPA.perturbations import PerturbationLpNorm

_DT = torch.float64


def _chain():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 6, dtype=_DT), nn.ReLU(),
                         nn.Linear(6, 6, dtype=_DT), nn.ReLU(),
                         nn.Linear(6, 2, dtype=_DT))


def _conv():
    torch.manual_seed(3)
    return nn.Sequential(nn.Conv2d(1, 3, 3, stride=2, padding=1, dtype=_DT),
                         nn.ReLU(), nn.Flatten(),
                         nn.Linear(3 * 4 * 4, 6, dtype=_DT), nn.ReLU(),
                         nn.Linear(6, 2, dtype=_DT))


class _Diamond(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(28)
        self.l = nn.Linear(4, 3, dtype=_DT)
        self.r = nn.Linear(4, 3, dtype=_DT)
        self.lo = nn.Linear(3, 2, dtype=_DT)
        self.ro = nn.Linear(3, 2, dtype=_DT)

    def forward(self, x):
        return self.lo(torch.relu(self.l(x))) - self.ro(torch.relu(self.r(x)))


class _ConvResidual(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        self.b = nn.ModuleList([nn.Conv2d(3, 3, 3, padding=1, dtype=_DT)
                                for _ in range(2)])
        self.s = nn.ModuleList([nn.Conv2d(3, 3, 1, dtype=_DT)
                                for _ in range(2)])
        self.o = nn.Linear(3 * 5 * 5, 2, dtype=_DT)

    def forward(self, x):
        for block, skip in zip(self.b, self.s):
            x = torch.relu(block(x)) + skip(x)
        return self.o(x.flatten(1))


_MODELS = {
    'chain': (_chain, (1, 3)),
    'conv': (_conv, (1, 1, 7, 7)),
    'diamond': (_Diamond, (1, 4)),
    'conv-residual': (_ConvResidual, (1, 3, 5, 5)),
}


def _setup(name, eps=0.6, fixed=None):
    """The module, the box (the coordinates in `fixed` not moving), and
    CROWN's pre-activation bounds."""
    make, shape = _MODELS[name]
    centre = torch.linspace(-0.5, 0.5, int(torch.Size(shape).numel()),
                            dtype=_DT).reshape(shape)
    radius = torch.full(shape, eps, dtype=_DT)
    if fixed is not None:
        radius.view(-1)[fixed] = 0.
    bm = BoundedModule(make(), centre, device='cpu')
    bx = BoundedTensor(centre, PerturbationLpNorm(
        norm=float('inf'), x_L=centre - radius, x_U=centre + radius))
    bm.compute_bounds(x=(bx,), method='backward')
    interm = {r.inputs[0].name: (r.inputs[0].lower.detach(),
                                 r.inputs[0].upper.detach()) for r in bm.relus}
    return bm, bx, interm


def _random_slopes(bm, interm):
    """Random slopes, with both ends of [0, 1] on some unstable neurons."""
    g = torch.Generator().manual_seed(0)
    slopes = {}
    for relu in bm.relus:
        lower, upper = interm[relu.inputs[0].name]
        slope = torch.rand(lower.shape, dtype=_DT, generator=g)
        unstable = ((lower < 0) & (upper > 0)).reshape(-1).nonzero().reshape(-1)
        slope.view(-1)[unstable[:2]] = 0.
        slope.view(-1)[unstable[-2:]] = 1.
        slopes[relu.name] = slope
    return slopes


def _read(bm, bx, interm, mode, chunk=None, monkeypatch=None, C=None):
    """The generators and centres of the output and of every pre-activation
    at its unstable neurons, read in `mode`, for the one box:
    `{name: (G, c)}`, and the numbering of the variables. With `C`, under the
    slopes that dynamic-forward-optimized finds for `C`."""
    if chunk is not None:
        real = dynamic_forward_bound._reverse_mode_generators
        monkeypatch.setattr(dynamic_forward_bound, '_reverse_mode_generators',
                            lambda *a: real(*a, chunk=chunk))
    bm.set_bound_opts({'dynamic_forward': {'generators': mode}})
    rows = {r.inputs[0].name: ((interm[r.inputs[0].name][0] < 0)
                               & (interm[r.inputs[0].name][1] > 0))
            .reshape(-1).nonzero().reshape(-1) for r in bm.relus}
    _, _, generators = bm.compute_bounds(
        x=(bx,), C=None if C is None else C.unsqueeze(0),
        method='dynamic-forward' if C is None else 'dynamic-forward-optimized',
        interm_bounds=interm, return_generators=True,
        needed_generators_dict={bm.final_name: None, **rows})
    ranges = dict(bm[bm.input_name[0]].linear.dynamic_variable_ranges)
    return {name: (G[0], c[0]) for name, (G, c) in generators.items()}, ranges


def _generators_under(bm, bx, interm, slopes):
    """The output's generators and centre under given ReLU slopes, from a pass
    run directly: compute_bounds itself only takes slopes from its search."""
    bm.compute_bounds(x=(bx,), method='dynamic-forward', interm_bounds=interm)
    ranges = bm[bm.input_name[0]].linear.dynamic_variable_ranges
    with _same_slope_alpha(bm, slopes):
        linear, = _dynamic_forward_pass(bm, [bm.final_node()], ranges, 0)
    return linear.lw[0].flatten(1).t(), linear.lb[0].flatten()


def _crown_under(bm, bx, interm, C, slopes):
    """Native same-slope CROWN under the given network-wide alpha."""
    with _same_slope_alpha(bm, slopes):
        lower, _ = bm.compute_bounds(
            x=(bx,), C=C[None], method='backward', reuse_alpha=True,
            bound_upper=False, interm_bounds=interm)
    return lower[0]


def _assert_same(got, want):
    for key, (G, c) in want.items():
        assert got[key][0].shape == G.shape, key
        torch.testing.assert_close(got[key][0], G, rtol=0, atol=1e-12)
        torch.testing.assert_close(got[key][1], c, rtol=0, atol=1e-12)


@pytest.mark.parametrize('chunk', [None, 1])
@pytest.mark.parametrize('name', list(_MODELS))
def test_reverse_mode_gives_the_forward_generators(name, chunk, monkeypatch):
    bm, bx, interm = _setup(name)
    want, want_ranges = _read(bm, bx, interm, 'forward')
    got, ranges = _read(bm, bx, interm, 'reverse', chunk, monkeypatch)
    assert ranges == want_ranges
    _assert_same(got, want)


@pytest.mark.parametrize('name', list(_MODELS))
def test_reverse_generators_include_errors_at_zero_and_one_slopes(name):
    bm, bx, interm = _setup(name)
    slopes = _random_slopes(bm, interm)
    found = []
    for mode in ('forward', 'reverse'):
        bm.set_bound_opts({'dynamic_forward': {'generators': mode}})
        with _same_slope_alpha(bm, slopes):
            _, _, generators = bm.compute_bounds(
                x=(bx,), method='dynamic-forward', interm_bounds=interm,
                reuse_alpha=True, return_generators=True,
                needed_generators_dict={bm.final_name: None})
            found.append(generators)
    _assert_same(found[0], found[1])


@pytest.mark.parametrize('name', list(_MODELS))
def test_and_under_the_slopes_the_search_finds(name):
    bm, bx, interm = _setup(name)
    C = torch.tensor([[1., -1.]], dtype=_DT)
    want, want_ranges = _read(bm, bx, interm, 'forward', C=C)
    assert any(((l > 0) & (l < 1)).any() for l in bm.dynamic_forward_slopes.values())
    got, ranges = _read(bm, bx, interm, 'reverse', C=C)
    assert ranges == want_ranges
    _assert_same(got, want)


def test_a_mostly_fixed_box_gets_the_same_input_variables():
    bm, bx, interm = _setup('chain', fixed=[0, 2])
    want, ranges = _read(bm, bx, interm, 'forward')
    assert ranges[bm.input_name[0]] == (0, 1), 'the box is not compressed'
    got, _ = _read(bm, bx, interm, 'reverse')
    _assert_same(got, want)


@pytest.mark.parametrize('name', ['chain', 'conv-residual'])
@pytest.mark.parametrize('method', ['dynamic-forward', 'dynamic-forward-optimized'])
@pytest.mark.parametrize('chunk', [256, 1])
@pytest.mark.parametrize('given', [False, True])
def test_generator_modes_preserve_parameter_gradients(name, method, chunk, given, monkeypatch):
    bm, bx, interm = _setup(name)
    bm.set_bound_opts({'optimize_bound_args': {'iteration': 5, 'lr_alpha': .03}})
    reverse = dynamic_forward_bound._reverse_mode_generators
    monkeypatch.setattr(dynamic_forward_bound, '_reverse_mode_generators',
                        lambda *args: reverse(*args, chunk=chunk))
    C = torch.tensor([[[1., -1.], [.3, .7]]], dtype=_DT)
    requested = {bm.final_name: None, bm.relus[-1].inputs[0].name: [0, 2]}
    parameters = tuple(bm.parameters())
    expected = expected_grads = expected_forms = expected_form_grads = None
    for mode, return_generators in [('forward', False), ('forward', True),
                                    ('reverse', True), ('auto', True)]:
        bm.set_bound_opts({'dynamic_forward': {'generators': mode}})
        result = bm.compute_bounds(
            x=(bx,), C=C, method=method, interm_bounds=interm if given else None,
            return_generators=return_generators, needed_generators_dict=requested)
        lower, upper = result[:2]
        grads = torch.autograd.grad(
            lower.sum() + .3 * upper.sum(), parameters,
            retain_graph=return_generators)
        if expected is None:
            expected = lower.detach(), upper.detach()
            expected_grads = grads
        else:
            for got, want in zip((lower, upper, *grads), (*expected, *expected_grads)):
                torch.testing.assert_close(got, want, rtol=0, atol=1e-10)
        if return_generators:
            forms = result[2]
            loss = sum(G.square().sum() + c.square().sum() for G, c in forms.values())
            form_grads = torch.autograd.grad(loss, parameters)
            if expected_forms is None:
                expected_forms, expected_form_grads = forms, form_grads
            else:
                _assert_same(forms, expected_forms)
                for got, want in zip(form_grads, expected_form_grads):
                    torch.testing.assert_close(got, want, rtol=0, atol=1e-10)


def test_reverse_gradients_include_input_box_and_slopes():
    bm, bx, interm = _setup('chain')
    bx.ptb.x_L.requires_grad_()
    bx.ptb.x_U.requires_grad_()
    slopes = {name: slope.requires_grad_()
              for name, slope in _random_slopes(bm, interm).items()}
    variables = (bx.ptb.x_L, bx.ptb.x_U, *slopes.values())
    gradients = []
    for mode in ('forward', 'reverse'):
        bm.set_bound_opts({'dynamic_forward': {'generators': mode}})
        with _same_slope_alpha(bm, slopes):
            lower, upper, forms = bm.compute_bounds(
                x=(bx,), method='dynamic-forward', interm_bounds=interm,
                reuse_alpha=True, return_generators=True)
        G, c = forms[bm.final_name]
        gradients.append(torch.autograd.grad(
            lower.sum() + .3 * upper.sum() + G.square().sum() + c.square().sum(),
            variables))
    for got, want in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-10)


@pytest.mark.parametrize('method', ['dynamic-forward', 'dynamic-forward-optimized'])
@pytest.mark.parametrize('mode', ['forward', 'reverse', 'auto'])
@pytest.mark.parametrize('return_generators', [False, True])
def test_dynamic_forward_respects_requested_bound_sides(method, mode, return_generators):
    bm, bx, interm = _setup('chain')
    bm.set_bound_opts({'dynamic_forward': {'generators': mode},
                       'optimize_bound_args': {'iteration': 5, 'lr_alpha': .03}})
    kwargs = dict(x=(bx,), method=method, interm_bounds=interm,
                  return_generators=return_generators)
    with torch.no_grad():
        expected = bm.compute_bounds(**kwargs)
        for bound_lower, bound_upper in [(True, False), (False, True)]:
            result = bm.compute_bounds(
                **kwargs, bound_lower=bound_lower, bound_upper=bound_upper)
            for side, requested in enumerate((bound_lower, bound_upper)):
                if requested:
                    torch.testing.assert_close(result[side], expected[side], rtol=0, atol=1e-12)
                    assert not result[side].requires_grad
                else:
                    assert result[side] is None
            if return_generators:
                _assert_same(result[2], expected[2])
                assert all(not t.requires_grad for form in result[2].values() for t in form)


def test_auto_reads_by_whichever_mode_has_fewer(monkeypatch):
    seen = []
    real = dynamic_forward_bound._reverse_mode_generators
    monkeypatch.setattr(dynamic_forward_bound, '_reverse_mode_generators',
                        lambda *a: seen.append('reverse') or real(*a))
    bm, bx, interm = _setup('conv')
    # 2 output rows against the 49 input variables and more
    bm.compute_bounds(x=(bx,), method='dynamic-forward', interm_bounds=interm,
                      return_generators=True,
                      needed_generators_dict={bm.final_name: None})
    assert seen == ['reverse']
    torch.manual_seed(0)
    wide = nn.Sequential(nn.Linear(1, 4, dtype=_DT), nn.ReLU(),
                         nn.Linear(4, 20, dtype=_DT))
    x0 = torch.zeros(1, 1, dtype=_DT)
    wbm = BoundedModule(wide, x0, device='cpu')
    wbx = BoundedTensor(x0, PerturbationLpNorm(norm=float('inf'),
                                               x_L=x0 - 1, x_U=x0 + 1))
    # 20 output rows against at most 5 variables
    wbm.compute_bounds(x=(wbx,), method='dynamic-forward',
                       return_generators=True,
                       needed_generators_dict={wbm.final_name: None})
    assert seen == ['reverse'], 'forward mode was not chosen'


@pytest.mark.parametrize('name', list(_MODELS))
def test_the_bound_and_its_gradient_are_those_of_the_generators(name):
    bm, bx, interm = _setup(name)
    slopes = {k: v.requires_grad_() for k, v in _random_slopes(bm, interm).items()}
    C = torch.randn(3, 2, dtype=_DT, generator=torch.Generator().manual_seed(1))
    G, c = _generators_under(bm, bx, interm, slopes)
    want = C @ c - (C @ G).abs().sum(1)
    got = _crown_under(bm, bx, interm, C, slopes)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-10)
    for a, b in zip(torch.autograd.grad(got.sum(), list(slopes.values())),
                    torch.autograd.grad(want.sum(), list(slopes.values()))):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-10)


@pytest.mark.parametrize('slope', [0.0, 0.5, 1.0])
def test_any_slope_in_the_unit_range_bounds_the_network(slope):
    """Every slope in [0, 1] is sound, the two ends included."""
    bm, bx, interm = _setup('conv-residual')
    slopes = {r.name: torch.full_like(interm[r.inputs[0].name][0], slope)
             for r in bm.relus}
    G, c = _generators_under(bm, bx, interm, slopes)
    radius = G.abs().sum(1)
    pts = bx.ptb.x_L + (bx.ptb.x_U - bx.ptb.x_L) * torch.rand(
        5000, *bx.shape[1:], dtype=_DT, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        values = bm(pts).reshape(5000, -1)
    assert (c - radius <= values.min(0).values + 1e-9).all()
    assert (c + radius >= values.max(0).values - 1e-9).all()


@pytest.mark.parametrize('slope', [-1.0, 2.0])
def test_a_slope_outside_the_unit_range_is_projected_onto_it(slope):
    """Outside [0, 1], relu(z) - s z no longer starts at 0 (slope -1 over
    [-1, 1] spans [-1, 2]), so the slope is projected onto [0, 1]."""
    lower = torch.tensor([[-1., -2., 0.5]], dtype=_DT)
    upper = torch.tensor([[1., -1., 2.]], dtype=_DT)
    got = BoundRelu.dynamic_forward_relaxation(
        lower, upper, torch.full_like(lower, slope))
    want = BoundRelu.dynamic_forward_relaxation(
        lower, upper, torch.full_like(lower, min(max(slope, 0.), 1.)))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_a_slope_that_is_not_a_number_does_not_become_one():
    """A NaN slope stays NaN on an unstable neuron instead of being clamped
    to 0 or 1, for the bound check to reject."""
    lower = torch.tensor([[-1., 0.5]], dtype=_DT)
    upper = torch.tensor([[1., 2.]], dtype=_DT)
    slope, half = BoundRelu.dynamic_forward_relaxation(
        lower, upper, torch.full_like(lower, float('nan')))
    assert slope[0, 0].isnan() and half[0, 0].isnan()
    assert slope[0, 1] == 1 and half[0, 1] == 0


def test_the_searched_slopes_raise_the_bound_and_stay_sound():
    bm, bx, interm = _setup('conv-residual')
    C = torch.tensor([[1., -1.]], dtype=_DT)
    bm.compute_bounds(x=(bx,), method='dynamic-forward', interm_bounds=interm)
    start = {r.name: r.dynamic_forward_relaxation(
        *interm[r.inputs[0].name])[0] for r in bm.relus}
    before = _crown_under(bm, bx, interm, C, start)
    bm.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': .03}})
    after, _ = bm.compute_bounds(
        x=(bx,), C=C.unsqueeze(0), method='dynamic-forward-optimized',
        interm_bounds=interm)
    slopes = bm.dynamic_forward_slopes
    assert float(after) > float(before) + 1e-6, (float(before), float(after))
    torch.testing.assert_close(
        after.reshape(-1), _crown_under(bm, bx, interm, C, slopes),
        rtol=0, atol=1e-12)
    assert all(((l >= 0) & (l <= 1)).all() for l in slopes.values())
    pts = bx.ptb.x_L + (bx.ptb.x_U - bx.ptb.x_L) * torch.rand(
        5000, *bx.shape[1:], dtype=_DT, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        values = bm(pts) @ C.t()
    assert float(after) <= float(values.min()) + 1e-9


def test_reverse_mode_refuses_a_relaxation_other_than_relu():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4, dtype=_DT), nn.Sigmoid(),
                          nn.Linear(4, 2, dtype=_DT))
    x0 = torch.zeros(1, 3, dtype=_DT)
    bm = BoundedModule(model, x0, device='cpu',
                       bound_opts={'dynamic_forward': {'generators': 'reverse'}})
    bx = BoundedTensor(x0, PerturbationLpNorm(norm=float('inf'),
                                              x_L=x0 - 1, x_U=x0 + 1))
    with pytest.raises(NotImplementedError, match='not implemented'):
        bm.compute_bounds(x=(bx,), method='dynamic-forward',
                          return_generators=True)


@pytest.mark.parametrize('mode', ['forward', 'reverse', 'auto'])
def test_empty_python_rows_have_the_same_contract(mode):
    bm, bx, interm = _setup('chain')
    bm.set_bound_opts({'dynamic_forward': {'generators': mode}})
    _, _, found = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', interm_bounds=interm,
        return_generators=True, needed_generators_dict={bm.final_name: []})
    G, c = found[bm.final_name]
    assert G.shape[:2] == c.shape == (1, 0)


@pytest.mark.parametrize('mode', ['forward', 'reverse'])
def test_optimized_generator_requests_follow_compute_bounds_contract(mode):
    bm, bx, interm = _setup('chain')
    bm.set_bound_opts({'dynamic_forward': {'generators': mode},
                       'optimize_bound_args': {'iteration': 5, 'lr_alpha': .03}})
    C = torch.tensor([[[1., -1.]]], dtype=_DT)
    lower, upper, forms = bm.compute_bounds(
        x=(bx,), C=C, method='dynamic-forward-optimized',
        interm_bounds=interm, return_generators=True)
    assert set(forms) == {bm.final_name}
    G, c = forms[bm.final_name]
    q, centre = C @ G, (C @ c.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(lower, centre - q.abs().sum(-1), rtol=0, atol=1e-12)
    torch.testing.assert_close(upper, centre + q.abs().sum(-1), rtol=0, atol=1e-12)
    ancestor = bm.relus[0].inputs[0].name
    for requested in [None, {}, {ancestor: []}]:
        result = bm.compute_bounds(
            x=(bx,), C=C, method='dynamic-forward-optimized', interm_bounds=interm,
            return_generators=requested is not None, needed_generators_dict=requested)
        torch.testing.assert_close(result[0], lower, rtol=0, atol=1e-12)
        torch.testing.assert_close(result[1], upper, rtol=0, atol=1e-12)
        if requested is not None:
            assert set(result[2]) == set(requested)


@pytest.mark.parametrize('iterations', [0, 5])
def test_optimized_affine_network_needs_no_parameters(iterations):
    x = torch.zeros(1, 3, dtype=_DT)
    bm = BoundedModule(nn.Linear(3, 2, dtype=_DT), x, device='cpu',
                       bound_opts={'optimize_bound_args': {'iteration': iterations}})
    bx = BoundedTensor(x, PerturbationLpNorm(x_L=x - .5, x_U=x + .5))
    want = bm.compute_bounds(x=(bx,), method='dynamic-forward',
                             return_generators=True)
    got = bm.compute_bounds(x=(bx,), method='dynamic-forward-optimized',
                            return_generators=True)
    torch.testing.assert_close(got[0], want[0])
    torch.testing.assert_close(got[1], want[1])
    torch.testing.assert_close(got[2][bm.final_name][0], want[2][bm.final_name][0])
    assert bm.dynamic_forward_slopes == {}


def test_search_without_given_intermediate_bounds_matches_both_directions():
    # Grad mode remains enabled. Intermediate bound graphs must not be reused
    # by the optimizer, nor affine caches reused under different alpha.
    results = []
    for mode in ('forward', 'reverse'):
        bm, bx, _ = _setup('chain')
        bm.set_bound_opts({'dynamic_forward': {'generators': mode}})
        results.append(bm.compute_bounds(
            x=(bx,), C=torch.tensor([[[1., -1.]]], dtype=_DT),
            method='dynamic-forward-optimized',
            return_generators=True))
    for side in (0, 1):
        torch.testing.assert_close(results[0][side], results[1][side], rtol=0, atol=1e-12)
    _assert_same(results[0][2], results[1][2])


def test_optimized_bounds_do_not_initialize_forward_generators(monkeypatch):
    bm, bx, _ = _setup('conv')

    def unused(*args, **kwargs):
        raise AssertionError('bounds-only alpha-CROWN allocated forward generators')

    monkeypatch.setattr(bm, 'init_forward', unused)
    lower, upper = bm.compute_bounds(x=(bx,), method='dynamic-forward-optimized')
    assert torch.isfinite(lower).all() and torch.isfinite(upper).all()


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('given', [False, True])
def test_search_preserves_existing_alpha_metadata_options_and_gradients(fail, given, monkeypatch):
    bm, bx, interm = _setup('conv')
    C = torch.tensor([[[1., -1.]]], dtype=_DT)
    bm.init_alpha((bx,), c=C, interm_bounds=interm)
    bm.opt_reuse()
    bm.bound_opts['clip_in_alpha_crown'] = True
    constraints = bm.constraints_optimized = object()
    for p in bm.parameters():
        p.grad = torch.full_like(p, 7.)
    options = bm.bound_opts
    fields = ('alpha', 'alpha_lookup_idx', 'alpha_indices', 'opt_stage',
              'relu_options', 'options')
    before = [{k: getattr(r, k) for k in fields} for r in bm.relus]
    alpha_values = [{k: a.clone() for k, a in r.alpha.items()} for r in bm.relus]
    grads = [(p, p.grad, p.grad.clone()) for p in bm.parameters()]
    if fail:
        def broken(*args, **kwargs):
            raise RuntimeError('optimizer failure')
        monkeypatch.setattr(bm, '_get_optimized_bounds', broken)
    try:
        if fail:
            with pytest.raises(RuntimeError, match='optimizer failure'):
                bm.compute_bounds(x=(bx,), C=C, method='dynamic-forward-optimized',
                                  interm_bounds=interm if given else None)
        else:
            bm.compute_bounds(x=(bx,), C=C, method='dynamic-forward-optimized',
                              interm_bounds=interm if given else None,
                              return_generators=True)
        assert bm.bound_opts is options
        assert bm.constraints_optimized is constraints
        assert bm[bm.input_name[0]].value is bx
        assert bm[bm.input_name[0]].perturbation is bx.ptb
        for r, saved, values in zip(bm.relus, before, alpha_values):
            for key in fields:
                assert getattr(r, key) is saved[key]
            for key, value in values.items():
                torch.testing.assert_close(r.alpha[key], value, rtol=0, atol=0)
        for p, grad, value in grads:
            assert p.grad is grad
            torch.testing.assert_close(p.grad, value, rtol=0, atol=0)
    finally:
        for p, _, _ in grads:
            p.grad = None


def test_native_optimizer_shares_across_queries_nodes_and_sides():
    bm, bx, interm = _setup('diamond')
    C = torch.tensor([[[1., -1.], [-1., 1.], [.2, .7]]], dtype=_DT)
    with _same_slope_alpha(bm) as options:
        options['optimize_bound_args'].update({'iteration': 20, 'lr_alpha': .03})
        best, _ = bm.compute_bounds(x=(bx,), C=C, method='CROWN-optimized',
                                     bound_upper=False, interm_bounds=interm)
        for r in bm.relus:
            assert r.shared_alpha().shape == r.inputs[0].lower.shape
        # A coherent best iteration, not per-query maxima from different alpha.
        lower, upper = bm.compute_bounds(
            x=(bx,), C=C, method='backward', interm_bounds=interm, reuse_alpha=True)
        torch.testing.assert_close(lower, best, rtol=0, atol=1e-12)
        reverse_lower, _ = bm.compute_bounds(
            x=(bx,), C=-C, method='backward', interm_bounds=interm,
            reuse_alpha=True, bound_upper=False)
        torch.testing.assert_close(upper, -reverse_lower, rtol=0, atol=1e-12)
        slopes = {r.name: r.shared_alpha().detach().clone() for r in bm.relus}
        # The same alpha is valid for a new start node, without allocating a
        # second parameter bank or re-running an optimization.
        target = bm.relus[0].inputs[0]
        bm.compute_bounds(x=(bx,), method='backward', final_node_name=target.name,
                          interm_bounds=interm, reuse_alpha=True)
        for r in bm.relus:
            torch.testing.assert_close(r.shared_alpha(), slopes[r.name], rtol=0, atol=0)
    got = _crown_under(bm, bx, interm, C[0], slopes)
    torch.testing.assert_close(got, best[0], rtol=0, atol=1e-12)


@pytest.mark.parametrize('name', list(_MODELS))
def test_search_is_the_existing_same_slope_alpha_crown(name):
    bm, bx, interm = _setup(name)
    bm.set_bound_opts({'optimize_bound_args': {
        'iteration': 20, 'lr_alpha': .03, 'early_stop_patience': 20}})
    C = torch.tensor([[[1., -1.], [.3, .7]]], dtype=_DT)
    with _same_slope_alpha(bm):
        expected, _ = bm.compute_bounds(
            x=(bx,), C=C, method='CROWN-optimized', bound_upper=False,
            interm_bounds=interm)
    found, _ = bm.compute_bounds(x=(bx,), C=C,
                                 method='dynamic-forward-optimized', interm_bounds=interm)
    torch.testing.assert_close(found, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize('loose_bounds', [False, True])
def test_fixed_box_keeps_zero_input_padding_in_every_generator_mode(loose_bounds):
    x = torch.zeros(1, 3, dtype=_DT)
    model = nn.Sequential(nn.Linear(3, 4, dtype=_DT), nn.ReLU(),
                          nn.Linear(4, 1, dtype=_DT))
    with torch.no_grad():
        model[0].weight.fill_(1.)
        model[0].bias.zero_()
        model[2].weight.copy_(torch.tensor([[1., -1., 2., -2.]], dtype=_DT))
        model[2].bias.zero_()
    bm = BoundedModule(model, x, device='cpu')
    bx = BoundedTensor(x, PerturbationLpNorm(x_L=x, x_U=x))
    interm = ({bm.relus[0].inputs[0].name:
               (-torch.ones(1, 4, dtype=_DT), torch.ones(1, 4, dtype=_DT))}
              if loose_bounds else None)
    expected_G = (torch.tensor([[[0., .25, -.25, .5, -.5]]], dtype=_DT)
                  if loose_bounds else torch.zeros(1, 1, 1, dtype=_DT))
    for mode in ('forward', 'reverse', 'auto'):
        bm.set_bound_opts({'dynamic_forward': {'generators': mode}})
        lower, upper, forms = bm.compute_bounds(
            x=(bx,), method='dynamic-forward', interm_bounds=interm,
            return_generators=True)
        G, centre = forms[bm.final_name]
        torch.testing.assert_close(G, expected_G, rtol=0, atol=0)
        torch.testing.assert_close(centre, torch.zeros(1, 1, dtype=_DT), rtol=0, atol=0)
        torch.testing.assert_close(lower, -expected_G.abs().sum(-1), rtol=0, atol=0)
        torch.testing.assert_close(upper, expected_G.abs().sum(-1), rtol=0, atol=0)


@pytest.mark.parametrize('return_generators', [False, True])
def test_slope_search_preserves_existing_box_and_specification_gradients(return_generators):
    bm, bx, interm = _setup('chain')
    bm.set_bound_opts({'optimize_bound_args': {'iteration': 3, 'lr_alpha': .03}})
    C = torch.tensor([[[1., -1.]]], dtype=_DT, requires_grad=True)
    variables = (bx.ptb.x_L.requires_grad_(), bx.ptb.x_U.requires_grad_(), C)
    for variable in variables:
        variable.grad = torch.full_like(variable, 7.)
    previous = [(v.grad, v.grad.clone()) for v in variables]
    result = bm.compute_bounds(
        x=(bx,), C=C, method='dynamic-forward-optimized', interm_bounds=interm,
        return_generators=return_generators)
    for variable, (grad, value) in zip(variables, previous):
        assert variable.grad is grad
        torch.testing.assert_close(variable.grad, value, rtol=0, atol=0)
    gradients = torch.autograd.grad(result[0].sum() + result[1].sum(), variables)
    assert all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize('mode', ['forward', 'reverse'])
@pytest.mark.parametrize('return_generators', [False, True])
@pytest.mark.parametrize('given', [False, True])
@pytest.mark.parametrize('iterations', [0, 3])
def test_search_leaves_nonleaf_domain_graph_for_evaluation_and_input_reuse(
        mode, return_generators, given, iterations):
    bm, original, interm = _setup('chain')
    bm.set_bound_opts({'dynamic_forward': {'generators': mode},
                       'optimize_bound_args': {'iteration': iterations, 'lr_alpha': .03}})
    theta = torch.tensor(.6, dtype=_DT, requires_grad=True)
    query_scale = torch.tensor(.7, dtype=_DT, requires_grad=True)
    centre = original.detach()
    ptb = PerturbationLpNorm(x_L=centre - theta.square(), x_U=centre + theta.square())
    bx = BoundedTensor(centre, ptb)
    C = query_scale.square() * torch.tensor([[[1., -1.]]], dtype=_DT)
    kwargs = dict(C=C, method='dynamic-forward-optimized',
                  interm_bounds=interm if given else None,
                  return_generators=return_generators)
    fresh = bm.compute_bounds(x=(bx,), **kwargs)
    reused = bm.compute_bounds(**kwargs)
    for got, want in zip(reused[:2], fresh[:2]):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
    assert bm[bm.input_name[0]].perturbation is ptb
    fixed = {r.inputs[0].name: (r.inputs[0].lower.detach(), r.inputs[0].upper.detach())
             for r in bm.relus}
    with _same_slope_alpha(bm, bm.dynamic_forward_slopes):
        expected = bm.compute_bounds(
            x=(bx,), C=C, method='dynamic-forward', interm_bounds=fixed,
            reuse_alpha=True, return_generators=return_generators)
    variables = (theta, query_scale, *bm.parameters())
    loss = reused[0].sum() + .3 * reused[1].sum()
    expected_loss = expected[0].sum() + .3 * expected[1].sum()
    if return_generators:
        G, c = reused[2][bm.final_name]
        loss = loss + G.square().sum() + c.square().sum()
        G, c = expected[2][bm.final_name]
        expected_loss = expected_loss + G.square().sum() + c.square().sum()
    gradients = torch.autograd.grad(loss, variables, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, variables)
    for got, want in zip(gradients, expected_gradients):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-10)


def test_input_reuse_after_failed_search_keeps_the_callers_domain_graph(monkeypatch):
    bm, original, interm = _setup('chain')
    bm.set_bound_opts({'optimize_bound_args': {'iteration': 3, 'lr_alpha': .03}})
    theta = torch.tensor(.6, dtype=_DT, requires_grad=True)
    centre = original.detach()
    bx = BoundedTensor(centre, PerturbationLpNorm(
        x_L=centre - theta.square(), x_U=centre + theta.square()))
    kwargs = dict(C=torch.tensor([[[1., -1.]]], dtype=_DT),
                  method='dynamic-forward-optimized', interm_bounds=interm)
    with monkeypatch.context() as patched:
        def broken(*args, **kwargs):
            raise RuntimeError('optimizer failure')
        patched.setattr(bm, '_get_optimized_bounds', broken)
        with pytest.raises(RuntimeError, match='optimizer failure'):
            bm.compute_bounds(x=(bx,), **kwargs)
    assert bm[bm.input_name[0]].value is bx
    assert bm[bm.input_name[0]].perturbation is bx.ptb
    lower, upper = bm.compute_bounds(**kwargs)
    gradient, = torch.autograd.grad(lower.sum() + .3 * upper.sum(), (theta,))
    assert torch.isfinite(gradient)

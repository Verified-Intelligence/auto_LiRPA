"""`compute_bounds(method='dynamic-forward', return_generators=True)` returns a
node as `c + G e`, its centre and generators over the dynamic-forward
variables. They must give the same bounds as compute_bounds, contain the
network, and not depend on how many variables a pass carries
(`forward_max_dim`) or on the order the nodes are asked for in."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators.relu import BoundRelu
from auto_LiRPA.perturbations import PerturbationLpNorm

_DT = torch.float64


def _chain(dtype=_DT):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 6, dtype=dtype), nn.ReLU(),
                         nn.Linear(6, 6, dtype=dtype), nn.ReLU(),
                         nn.Linear(6, 2, dtype=dtype))


def _conv(dtype=_DT):
    torch.manual_seed(3)
    return nn.Sequential(nn.Conv2d(1, 3, 3, stride=2, padding=1, dtype=dtype),
                         nn.ReLU(), nn.Flatten(),
                         nn.Linear(3 * 4 * 4, 6, dtype=dtype), nn.ReLU(),
                         nn.Linear(6, 2, dtype=dtype))


def _run(model, shape, max_dim=int(1e9), dtype=_DT, eps=1.0):  # noqa: E302
    x_L = torch.full(shape, -eps, dtype=dtype)
    x_U = torch.full(shape, eps, dtype=dtype)
    centre = (x_L + x_U) / 2
    bm = BoundedModule(model, centre, device='cpu',
                       bound_opts={'forward_max_dim': max_dim})
    bx = BoundedTensor(centre, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    lower, upper = bm.compute_bounds(x=(bx,), method='dynamic-forward')
    return bm, bx, lower, upper


def _nodes(bm):
    """Every pre-activation, then the output -- what a verifier asks for."""
    return [r.inputs[0] for r in bm.relus] + [bm.final_node()]


def _generators_of(bm, bx, node=None, **kwargs):
    """The generators `G [elements, n]` and centre `c [elements]` of `node`
    (the output by default), for a single input box."""
    name = bm.final_name if node is None else node.name
    _, _, generators = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', final_node_name=name,
        return_generators=True, needed_generators_dict={name: None}, **kwargs)
    G, c = generators[name]
    return G[0], c[0]


def _ranges(bm):
    """The variables each node introduces, as the last call numbered them."""
    return bm[bm.input_name[0]].linear.dynamic_variable_ranges


@pytest.mark.parametrize('max_dim', [1, int(1e9)])
def test_relu_generators_and_centre_match_the_relaxation(max_dim):
    """ReLU on [-1, 1] has input slope 1/2 and half-error 1/4."""
    x = torch.zeros(1, 1, dtype=_DT)
    bm = BoundedModule(nn.ReLU(), x, device='cpu',
                       bound_opts={'forward_max_dim': max_dim})
    bx = BoundedTensor(x, PerturbationLpNorm(
        norm=np.inf, x_L=x - 1, x_U=x + 1))
    lower, upper, forms = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', return_generators=True)
    G, c = forms[bm.final_name]
    torch.testing.assert_close(G, torch.tensor([[[0.5, 0.25]]], dtype=_DT))
    torch.testing.assert_close(c, torch.tensor([[0.25]], dtype=_DT))
    torch.testing.assert_close(lower, torch.tensor([[-0.5]], dtype=_DT))
    torch.testing.assert_close(upper, torch.tensor([[1.0]], dtype=_DT))


@pytest.mark.parametrize('radii, expected_G', [
    ([1., 1., 0.], [1., 2., 0.]),
    ([1., 0., 0.], [1.]),
    ([0., 0., 0.], [0.]),
], ids=['dense', 'sparse', 'fixed'])
def test_input_generator_columns_match_dense_and_sparse_boxes(radii, expected_G):
    """Dense and sparse input columns represent the same normalized box."""
    x = torch.tensor([[2., 3., 4.]], dtype=_DT)
    radius = torch.tensor([radii], dtype=_DT)
    model = nn.Linear(3, 1, dtype=_DT)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 2., 3.]], dtype=_DT))
        model.bias.fill_(0.25)
    bm = BoundedModule(model, x, device='cpu')
    bx = BoundedTensor(x, PerturbationLpNorm(
        norm=np.inf, x_L=x - radius, x_U=x + radius))
    _, _, forms = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', return_generators=True)
    G, c = forms[bm.final_name]
    torch.testing.assert_close(G, torch.tensor([[expected_G]], dtype=_DT))
    torch.testing.assert_close(c, torch.tensor([[20.25]], dtype=_DT))


@pytest.mark.parametrize('cached', [False, True])
def test_generator_requests_reject_nodes_outside_the_target_ancestors(cached):
    model = _chain()
    bx = BoundedTensor(torch.zeros(1, 3, dtype=_DT), PerturbationLpNorm(
        norm=np.inf, x_L=-torch.ones(1, 3, dtype=_DT),
        x_U=torch.ones(1, 3, dtype=_DT)))
    bm = BoundedModule(model, bx, device='cpu')
    if cached:
        bm.compute_bounds(x=(bx,), method='dynamic-forward')
    target = bm.relus[0].inputs[0]
    with pytest.raises(ValueError, match='current bound target'):
        bm.compute_bounds(x=None if cached else (bx,), method='dynamic-forward',
                          final_node_name=target.name, return_generators=True,
                          needed_generators_dict={bm.final_name: None})
    # A rejected request does not prevent a subsequent valid query.
    _, _, forms = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', final_node_name=target.name,
        return_generators=True, needed_generators_dict={target.name: None})
    assert set(forms) == {target.name}


@pytest.mark.parametrize('max_dim', [1, int(1e9)])
def test_C_bounds_agree_with_raw_generators_across_request_modes(max_dim):
    bm, bx, _, _ = _run(_chain(), (2, 3), max_dim=max_dim)
    C = torch.tensor([[[1., -2.]], [[-3., 1.]]], dtype=torch.float32)
    lo, hi, forms = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', C=C, return_generators=True)
    G, c = forms[bm.final_name]
    q = C.to(G).matmul(G)
    centre = C.to(c).matmul(c.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(lo, centre - q.abs().sum(-1))
    torch.testing.assert_close(hi, centre + q.abs().sum(-1))
    ancestor = bm.relus[0].inputs[0].name
    for requested in [None, {}, {ancestor: None}]:
        result = bm.compute_bounds(
            x=None, method='dynamic-forward', C=C,
            return_generators=requested is not None,
            needed_generators_dict=requested)
        torch.testing.assert_close(result[0], lo)
        torch.testing.assert_close(result[1], hi)
        if requested is not None:
            assert set(result[2]) == set(requested)


@pytest.mark.parametrize('name', ['chain', 'conv'])
def test_G_is_what_compute_bounds_concretized(name):
    """The generators are the pass's own bound: concretizing them gives the
    bounds compute_bounds returned."""
    model, shape = ((_chain(), (1, 3)) if name == 'chain'
                    else (_conv(), (1, 1, 7, 7)))
    bm, bm_x, _, _ = _run(model, shape)
    lower, upper, generators = bm.compute_bounds(
        x=(bm_x,), method='dynamic-forward',
        return_generators=True, needed_generators_dict={bm.final_name: None})
    G, c = (t[0] for t in generators[bm.final_name])
    radius = G.abs().sum(1)
    torch.testing.assert_close(c - radius, lower.reshape(-1),
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(c + radius, upper.reshape(-1),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize('name', ['chain', 'conv'])
def test_G_contains_the_network(name):
    """Whatever they describe still has to be an over-approximation."""
    model, shape = ((_chain(), (1, 3)) if name == 'chain'
                    else (_conv(), (1, 1, 7, 7)))
    bm, bm_x, _, _ = _run(model, shape)
    G, c = _generators_of(bm, bm_x)
    lower = c - G.abs().sum(1)
    upper = c + G.abs().sum(1)
    pts = torch.rand(2000, *shape[1:], dtype=_DT) * 2 - 1
    with torch.no_grad():
        v = model(pts).reshape(2000, -1)
    assert (lower <= v.min(0).values + 1e-9).all()
    assert (upper >= v.max(0).values - 1e-9).all()


@pytest.mark.parametrize('name', ['chain', 'conv'])
@pytest.mark.parametrize('split', ['one', 'half', 'all-but-one'])
def test_G_does_not_depend_on_forward_max_dim(name, split):
    """However `forward_max_dim` splits the variables into slices, the
    centre and generators are the same."""
    model, shape = ((_chain(), (1, 3)) if name == 'chain'
                    else (_conv(), (1, 1, 7, 7)))
    bm, bm_x, _, _ = _run(model, shape)
    want = {n.name: _generators_of(bm, bm_x, n) for n in _nodes(bm)}
    n_variables = max(r.shape[1] for r, _ in want.values())
    max_dim = {'one': 1, 'half': max(1, n_variables // 2),
               'all-but-one': n_variables - 1}[split]
    assert max_dim < n_variables, 'the split has to actually cut something'

    bm_w, bm_w_x, _, _ = _run(model, shape, max_dim=max_dim)
    for node in _nodes(bm_w):
        G, c = _generators_of(bm_w, bm_w_x, node)
        G_want, c_want = want[node.name]
        assert G.shape == G_want.shape, (node.name, G.shape)
        torch.testing.assert_close(G, G_want, rtol=0, atol=1e-12)
        torch.testing.assert_close(c, c_want, rtol=0, atol=1e-12)

    # the same through one call for every node, with rows picked out of
    # each, which is how a verifier asks
    rows = {n.name: torch.tensor([0, want[n.name][0].shape[0] - 1])
            for n in _nodes(bm_w)}
    _, _, generators = bm_w.compute_bounds(
        x=(bm_w_x,), method='dynamic-forward', return_generators=True,
        needed_generators_dict=rows)
    for name, (G, c) in generators.items():
        G_want, c_want = want[name]
        torch.testing.assert_close(G[0], G_want[rows[name]], rtol=0, atol=1e-12)
        torch.testing.assert_close(c[0], c_want[rows[name]], rtol=0, atol=1e-12)


def test_a_matmul_on_a_constant_on_the_left_is_refused():
    """W @ x is not covered, and says so like an unsupported operator."""
    class LeftMatMul(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.w = nn.Parameter(torch.randn(2, 3, dtype=_DT))

        def forward(self, x):
            return self.w @ x

    with pytest.raises(NotImplementedError, match='perturbed weight'):
        _run(LeftMatMul(), (1, 3, 1))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_it_keeps_the_model_dtype(dtype):
    """The centre and generators keep the model's dtype."""
    bm, bm_x, _, _ = _run(_chain(dtype), (1, 3), dtype=dtype)
    G, c = _generators_of(bm, bm_x)
    assert G.dtype is dtype and c.dtype is dtype


def test_every_pre_activation_has_one_too():
    """Every pre-activation has generators too, over the same variables."""
    model = _chain()
    bm, bm_x, _, _ = _run(model, (1, 3))
    n_variables = None
    for relu in bm.relus:
        node = relu.inputs[0]
        lower, upper, generators = bm.compute_bounds(
            x=(bm_x,), method='dynamic-forward', final_node_name=node.name,
            return_generators=True, needed_generators_dict={node.name: None})
        G, c = (t[0] for t in generators[node.name])
        numel = int(np.prod([int(d) for d in node.output_shape[1:]]))
        assert G.shape[0] == numel and c.shape[0] == numel
        # one set of variables, so a later node depends on at least as many
        assert n_variables is None or G.shape[1] >= n_variables
        n_variables = G.shape[1]
        radius = G.abs().sum(1)
        torch.testing.assert_close(c - radius, lower.reshape(-1),
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(c + radius, upper.reshape(-1),
                                   rtol=0, atol=1e-12)


def test_a_batch_of_boxes_gives_each_box_its_own():
    """Each box of a batch gets generators that give its own bounds."""
    model = _chain()
    x = torch.tensor([[0.1, 0.2, 0.3], [-0.3, 0.0, 0.4]], dtype=_DT)
    eps = 0.25
    bm = BoundedModule(model, x, device='cpu')
    bx = BoundedTensor(x, PerturbationLpNorm(
        norm=float('inf'), x_L=x - eps, x_U=x + eps))
    lower, upper, generators = bm.compute_bounds(
        x=(bx,), method='dynamic-forward',
        return_generators=True, needed_generators_dict={bm.final_name: None})
    G, c = generators[bm.final_name]
    radius = G.abs().sum(-1)
    torch.testing.assert_close(c - radius, lower, rtol=0, atol=1e-12)
    torch.testing.assert_close(c + radius, upper, rtol=0, atol=1e-12)


@pytest.mark.parametrize('max_dim', [1, int(1e9)])
def test_generator_requests_have_a_stable_return_contract_on_reused_input(max_dim):
    """The flag controls tuple arity; the query controls entries and rows."""
    x = torch.tensor([[0.1, 0.2, 0.3], [-0.3, 0.0, 0.4]], dtype=_DT)
    bm = BoundedModule(_chain(), x, device='cpu',
                       bound_opts={'forward_max_dim': max_dim})
    bx = BoundedTensor(x, PerturbationLpNorm(
        norm=float('inf'), x_L=x - 0.25, x_U=x + 0.25))
    lower, upper, full = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', return_generators=True)
    G, c = full[bm.final_name]
    rows = torch.tensor([1, 0, 1])
    selected = {bm.final_name: rows}
    # Each call reuses the input while replacing both parts of the request.
    requests = [(True, {}, {}), (False, selected, None),
                (True, selected, {bm.final_name: (G[:, rows], c[:, rows])}),
                (False, None, None), (True, None, full)]
    for enabled, query, expected in requests:
        result = bm.compute_bounds(
            x=None, method='dynamic-forward', return_generators=enabled,
            needed_generators_dict=query)
        assert len(result) == (3 if enabled else 2)
        torch.testing.assert_close(result[0], lower, rtol=0, atol=1e-12)
        torch.testing.assert_close(result[1], upper, rtol=0, atol=1e-12)
        if enabled:
            assert result[2].keys() == expected.keys()
            for name, tensors in expected.items():
                for got, want in zip(result[2][name], tensors):
                    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize('intermediate_target', [False, True])
def test_default_generators_describe_the_current_bound_target(intermediate_target):
    """An omitted query describes final_node_name, including intermediates."""
    bm, bx, _, _ = _run(_chain(), (1, 3))
    target = bm.relus[0].inputs[0] if intermediate_target else bm.final_node()
    lower, upper, forms = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', final_node_name=target.name,
        return_generators=True)
    assert set(forms) == {target.name}
    G, c = forms[target.name]
    reach = G.abs().sum(-1)
    torch.testing.assert_close(c - reach, lower, rtol=0, atol=1e-12)
    torch.testing.assert_close(c + reach, upper, rtol=0, atol=1e-12)


def _interm(bm):
    return {r.inputs[0].name: (r.inputs[0].lower.detach(),
                               r.inputs[0].upper.detach()) for r in bm.relus}


class _Diamond(nn.Module):
    """Both branches relax, so both introduce variables."""

    def __init__(self, sub=False):
        super().__init__()
        torch.manual_seed(28)
        self.l = nn.Linear(4, 2, dtype=_DT)
        self.r = nn.Linear(4, 2, dtype=_DT)
        self.lo = nn.Linear(2, 2, dtype=_DT)
        self.ro = nn.Linear(2, 2, dtype=_DT)
        self.sub = sub

    def forward(self, x):
        a = self.lo(torch.relu(self.l(x)))
        b = self.ro(torch.relu(self.r(x)))
        return a - b if self.sub else a + b


class _Residual(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(12)
        self.a = nn.Linear(4, 5, dtype=_DT)
        self.h = nn.Linear(5, 5, dtype=_DT)
        self.skip = nn.Linear(4, 5, dtype=_DT)
        self.out = nn.Linear(5, 2, dtype=_DT)

    def forward(self, x):
        return self.out(torch.relu(self.h(torch.relu(self.a(x))) + self.skip(x)))


@pytest.mark.parametrize('model', [_Diamond(), _Diamond(sub=True), _Residual()],
                         ids=['diamond-add', 'diamond-sub', 'residual'])
def test_a_dag_shares_one_set_of_variables(model):
    """Branches add up only if a variable is the same on both of them."""
    bm, bm_x, lower, upper = _run(model, (1, 4))
    G, c = _generators_of(bm, bm_x)
    radius = G.abs().sum(1)
    torch.testing.assert_close(c - radius, lower.reshape(-1),
                               rtol=0, atol=1e-12)
    pts = torch.rand(20000, 4, dtype=_DT) * 2 - 1
    with torch.no_grad():
        v = model(pts)
    assert (c - radius <= v.min(0).values + 1e-9).all()
    assert (c + radius >= v.max(0).values - 1e-9).all()
    # every variable is somebody's, and nobody's twice
    blocks = sorted(_ranges(bm).values())
    assert G.shape[1] == sum(count for _, count in blocks), (G.shape, blocks)
    assert [s for s, _ in blocks] == list(
        np.cumsum([0] + [c for _, c in blocks])[:-1])


class _Deep(nn.Module):
    """A residual chain: each block reaches its predecessor by two paths."""

    def __init__(self, blocks):
        super().__init__()
        torch.manual_seed(1)
        self.blocks = nn.ModuleList(
            [nn.Linear(4, 4, dtype=_DT) for _ in range(blocks)])
        self.skips = nn.ModuleList(
            [nn.Linear(4, 4, dtype=_DT) for _ in range(blocks)])
        self.out = nn.Linear(4, 2, dtype=_DT)

    def forward(self, x):
        for block, skip in zip(self.blocks, self.skips):
            x = torch.relu(block(x)) + skip(x)
        return self.out(x)


@pytest.mark.parametrize('final_first', [False, True])
def test_joint_generator_requests_agree_with_individual_requests(final_first):
    """Joint requests preserve each node's form in either request order."""
    model = _Deep(5)
    bm, bm_x, _, _ = _run(model, (1, 4))
    want = {n.name: _generators_of(bm, bm_x, n) for n in _nodes(bm)}

    bm, bm_x, _, _ = _run(model, (1, 4))
    interm = _interm(bm)
    nodes = _nodes(bm)
    if final_first:
        nodes = nodes[-1:] + nodes[:-1]
    _, _, generators = bm.compute_bounds(
        x=(bm_x,), method='dynamic-forward', interm_bounds=interm,
        return_generators=True,
        needed_generators_dict={n.name: None for n in nodes})
    assert set(generators) == set(want)
    for name, (G, c) in want.items():
        torch.testing.assert_close(generators[name][0][0], G, rtol=0, atol=1e-12)
        torch.testing.assert_close(generators[name][1][0], c, rtol=0, atol=1e-12)


def test_a_bound_is_released_after_its_last_reader():
    """Without autograd, a bound is freed after its last reader is computed."""
    import weakref
    from auto_LiRPA.operators.linear import BoundLinear

    model = _Residual()
    bm, bm_x, _, _ = _run(model, (1, 4))
    interm = _interm(bm)
    alive, seen_dead = {}, {}
    real = BoundLinear.bound_dynamic_forward

    def spy(self, *args, **kwargs):
        for name, ref in alive.items():
            seen_dead.setdefault(self.name, {})[name] = ref() is None
        out = real(self, *args, **kwargs)
        alive[self.name] = weakref.ref(out.lw)
        return out

    x_L = torch.full((1, 4), -1.0, dtype=_DT)
    x_U = torch.full((1, 4), 1.0, dtype=_DT)
    bx = BoundedTensor((x_L + x_U) / 2, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    BoundLinear.bound_dynamic_forward = spy
    try:
        with torch.no_grad():
            bm.compute_bounds(x=(bx,), method='dynamic-forward',
                              interm_bounds=interm)
    finally:
        BoundLinear.bound_dynamic_forward = real
    # _Residual: out(relu(h(relu(a(x))) + skip(x))). By the time `out` runs,
    # `a` (read by one ReLU) and `h` and `skip` (read by the Add) are done with.
    linears = [n.name for n in bm.nodes() if isinstance(n, BoundLinear)]
    final = bm.final_name
    assert all(seen_dead[final][name] for name in linears if name != final), (
        seen_dead[final])


@pytest.mark.parametrize('relu_layers', [1, 3])
def test_it_agrees_with_same_slope_crown(relu_layers):
    """Under the same ReLU slopes, the generators give the bound of
    same-slope CROWN (CROWN with one slope for both its lower and upper
    relaxation of every ReLU), whatever `C` the bounds are for."""
    torch.manual_seed(0)
    layers = [nn.Linear(4, 6, dtype=_DT), nn.ReLU()]
    for _ in range(relu_layers - 1):
        layers += [nn.Linear(6, 6, dtype=_DT), nn.ReLU()]
    layers += [nn.Linear(6, 3, dtype=_DT)]
    model = nn.Sequential(*layers)

    eps = 0.4
    x0 = torch.linspace(-1., 1., 4, dtype=_DT).reshape(1, 4)
    bm = BoundedModule(model, x0, device='cpu', bound_opts={
        'activation_bound_option': 'same-slope',
        'sparse_features_alpha': False, 'sparse_spec_alpha': False})
    bx = BoundedTensor(x0, PerturbationLpNorm(
        norm=float('inf'), x_L=x0 - eps, x_U=x0 + eps))
    bm.compute_bounds(x=(bx,), method='backward')
    interm = _interm(bm)
    assert any(int(((lo < 0) & (hi > 0)).sum()) for lo, hi in interm.values()), (
        'with nothing unstable there is no relaxation to agree about')
    C = torch.cat([
        torch.eye(3, dtype=_DT).unsqueeze(0),
        torch.tensor([[[1., -1., 0.], [0., 1., -1.]]], dtype=_DT),
    ], dim=1)

    # dynamic forward's slopes, u / (u - l)
    slopes = {r.name: BoundRelu.dynamic_forward_relaxation(
        *interm[r.inputs[0].name])[0] for r in bm.relus}

    _, _, generators = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', C=C, interm_bounds=interm,
        return_generators=True, needed_generators_dict={bm.final_name: None})
    # the output's own generators, whatever C the bounds were for
    G, c = (t[0] for t in generators[bm.final_name])
    q, c = C[0] @ G, C[0] @ c

    # CROWN with that same slope for both relaxations of every ReLU
    bm.init_alpha(x=(bx,), method='backward', c=C, interm_bounds=interm)
    for relu in bm.relus:
        shape = relu.alpha[bm.final_name].shape
        relu.alpha[bm.final_name] = slopes[relu.name].reshape(
            1, 1, *shape[2:]).expand(shape).clone().requires_grad_(True)
        relu.opt_start()
    lower, upper, A = bm.compute_bounds(
        x=(bx,), method='backward', C=C, interm_bounds=interm, return_A=True)

    torch.testing.assert_close(lower.reshape(-1), c - q.abs().sum(1),
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(upper.reshape(-1), c + q.abs().sum(1),
                               rtol=0, atol=1e-12)
    # and the input variables' generators are CROWN's coefficients scaled by
    # the box's half-widths
    lA = A[bm.final_name][bm.input_name[0]]['lA'].reshape(q.shape[0], -1)
    radii = ((bx.ptb.x_U - bx.ptb.x_L) / 2).reshape(-1)
    torch.testing.assert_close(q[:, :radii.numel()], lA * radii,
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize('subtract', [False, True])
def test_a_broadcasting_constant_widens_the_generators(subtract):
    """An operand with no generators (a constant) still sets the shape of
    the result."""

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                'constant', torch.tensor([1., 2., 3.], dtype=_DT))
            self.out = nn.Linear(3, 1, dtype=_DT)
            with torch.no_grad():
                self.out.weight.fill_(1.)
                self.out.bias.zero_()

        def forward(self, x):
            merged = self.constant - x if subtract else x + self.constant
            return self.out(merged)

    x_L = torch.tensor([[-1.]], dtype=_DT)
    x_U = torch.tensor([[1.]], dtype=_DT)
    bm = BoundedModule(Net(), (x_L + x_U) / 2, device='cpu')
    bx = BoundedTensor((x_L + x_U) / 2, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    lower, upper = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', C=torch.ones(1, 1, 1, dtype=_DT))
    torch.testing.assert_close(lower, torch.tensor([[3.]], dtype=_DT))
    torch.testing.assert_close(upper, torch.tensor([[9.]], dtype=_DT))


class _ConvResidual(nn.Module):
    """A residual net over 4-D tensors, which is the shape this is for."""

    def __init__(self, blocks=2):
        super().__init__()
        torch.manual_seed(5)
        self.b = nn.ModuleList([nn.Conv2d(3, 3, 3, padding=1, dtype=_DT)
                                for _ in range(blocks)])
        self.s = nn.ModuleList([nn.Conv2d(3, 3, 1, dtype=_DT)
                                for _ in range(blocks)])
        self.o = nn.Linear(3 * 5 * 5, 2, dtype=_DT)

    def forward(self, x):
        for block, skip in zip(self.b, self.s):
            x = torch.relu(block(x)) + skip(x)
        return self.o(x.flatten(1))


@pytest.mark.parametrize('max_dim', [3, 20, int(1e9)])
def test_a_conv_dag_does_not_depend_on_forward_max_dim(max_dim):
    """Padding and broadcasting hold for 4-D generators at any
    forward_max_dim."""
    model = _ConvResidual()
    eps = 0.15
    x0 = torch.zeros(1, 3, 5, 5, dtype=_DT)

    def read(width):
        bm = BoundedModule(model, x0, device='cpu',
                           bound_opts={'forward_max_dim': width})
        bx = BoundedTensor(x0, PerturbationLpNorm(
            norm=float('inf'), x_L=x0 - eps, x_U=x0 + eps))
        return bm, _generators_of(bm, bx)

    bm, (G, c) = read(max_dim)
    _, (G_want, c_want) = read(int(1e9))
    torch.testing.assert_close(G, G_want, rtol=0, atol=1e-12)
    torch.testing.assert_close(c, c_want, rtol=0, atol=1e-12)

    radius = G.abs().sum(1)
    pts = (torch.rand(20000, 3, 5, 5, dtype=_DT) * 2 - 1) * eps
    with torch.no_grad():
        value = model(pts)
    assert (c - radius <= value.min(0).values + 1e-9).all()
    assert (c + radius >= value.max(0).values - 1e-9).all()


def test_a_mostly_fixed_box_has_a_variable_per_moving_coordinate():
    """A mostly fixed box has one variable per coordinate it moves."""
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(8, 6, dtype=_DT), nn.ReLU(),
                          nn.Linear(6, 2, dtype=_DT))
    centre = torch.linspace(-0.5, 0.5, 8, dtype=_DT).reshape(1, 8)
    moving = torch.tensor([[1., 0., 0., 1., 0., 0., 0., 1.]], dtype=_DT)
    x_L, x_U = centre - 0.4 * moving, centre + 0.4 * moving
    bm = BoundedModule(model, centre, device='cpu')
    bx = BoundedTensor(centre, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    lower, upper, generators = bm.compute_bounds(
        x=(bx,), method='dynamic-forward',
        return_generators=True, needed_generators_dict={bm.final_name: None})

    G, c = (t[0] for t in generators[bm.final_name])
    assert _ranges(bm)[bm.input_name[0]] == (0, 3), _ranges(bm)
    radius = G.abs().sum(1).reshape_as(lower)
    torch.testing.assert_close(c.reshape_as(lower) - radius,
                               lower, rtol=0, atol=1e-12)
    torch.testing.assert_close(c.reshape_as(upper) + radius,
                               upper, rtol=0, atol=1e-12)


@pytest.mark.parametrize('ptb', ['l2', 'no_box'])
def test_a_perturbation_that_is_not_a_box_is_refused(ptb):
    """Without an explicit box the variables would describe a point: refuse."""
    model = _chain()
    x = torch.zeros(1, 3, dtype=_DT)
    bm = BoundedModule(model, x, device='cpu')
    bx = BoundedTensor(x, PerturbationLpNorm(norm=2, eps=1.0) if ptb == 'l2'
                       else PerturbationLpNorm(norm=float('inf'), eps=0.5))
    with pytest.raises(NotImplementedError, match='Linf'):
        bm.compute_bounds(x=(bx,), method='dynamic-forward',
                          return_generators=True)


@pytest.mark.parametrize('method', [
    'forward', 'CROWN', 'IBP', 'forward-optimized', 'dynamic-forward+backward'])
def test_return_generators_is_for_dynamic_forward_only(method):
    model = _chain()
    bm, bm_x, _, _ = _run(model, (1, 3))
    with pytest.raises(NotImplementedError, match='dynamic-forward'):
        bm.compute_bounds(x=(bm_x,), method=method,
                          return_generators=True)

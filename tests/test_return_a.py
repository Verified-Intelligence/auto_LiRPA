"""Tests for returning input-space affine bounds through ``return_A``."""

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators import BoundAdd
from auto_LiRPA.perturbations import PerturbationL0Norm, PerturbationLpNorm


class ShiftedRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('offset', torch.tensor([2.0, 3.0]))

    def forward(self, x):
        return torch.relu(x + self.offset)


class TwoInputAffine(nn.Module):
    def forward(self, first, second):
        return first.sum(dim=-1, keepdim=True) - 2 * second


class TwoInputBranches(nn.Module):
    def forward(self, first, second):
        first_branch = first + 1
        return first_branch + second


def _bounded_box(lower, upper):
    """Create an L-inf bounded input from explicit box endpoints."""
    return BoundedTensor(
        (lower + upper) / 2,
        PerturbationLpNorm(norm=torch.inf, x_L=lower, x_U=upper))


def _concretize_entry(entry, lower, upper):
    """Concretize one returned A entry over an input box."""
    lA, uA = entry['lA'], entry['uA']
    lower = lower.unsqueeze(1)
    upper = upper.unsqueeze(1)
    lb = (lA.clamp(min=0) * lower + lA.clamp(max=0) * upper)
    ub = (uA.clamp(min=0) * upper + uA.clamp(max=0) * lower)
    return (
        lb.flatten(2).sum(dim=2) + entry['lbias'],
        ub.flatten(2).sum(dim=2) + entry['ubias'],
    )


def _concretize_perturbation_entry(entry, center, perturbation):
    """Concretize one returned A entry with its input perturbation."""
    output_size = entry['lbias'].shape[1]
    lA = entry['lA'].reshape(center.shape[0], output_size, -1)
    uA = entry['uA'].reshape(center.shape[0], output_size, -1)
    return (
        perturbation.concretize(center, lA, sign=-1) + entry['lbias'],
        perturbation.concretize(center, uA, sign=1) + entry['ubias'],
    )


def _compute_a(model, inputs, C=None, target_name=None, method='forward'):
    """Return bounds and the target's default input-space A entries."""
    target_name = model.output_name[0] if target_name is None else target_name
    lower, upper, A = model.compute_bounds(
        x=inputs, C=C, method=method, return_A=True,
        final_node_name=target_name)
    return lower, upper, A[target_name]


def _model_input_names(model):
    """Return graph names for runtime model inputs, excluding parameters."""
    return [
        name for name in model.input_name
        if model[name].model_input_index is not None
    ]


@pytest.mark.parametrize('method', ['forward', 'CROWN'])
def test_return_a_for_an_exact_affine_specification(method):
    """The default query returns exact coefficients after applying C."""
    layer = nn.Linear(2, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -1.0], [-1.0, 3.0]]))
        layer.bias.copy_(torch.tensor([1.0, -2.0]))
    x_L = torch.tensor([[1.0, -1.0]])
    x_U = torch.tensor([[2.0, 1.0]])
    model = BoundedModule(layer, (x_L + x_U) / 2, device='cpu')
    C = torch.tensor([[[1.0, -1.0]]])

    lower, upper, entries = _compute_a(
        model, (_bounded_box(x_L, x_U),), C=C, method=method)
    entry = entries[next(iter(entries))]

    torch.testing.assert_close(entry['lA'], torch.tensor([[[3.0, -4.0]]]))
    torch.testing.assert_close(entry['uA'], torch.tensor([[[3.0, -4.0]]]))
    torch.testing.assert_close(entry['lbias'], torch.tensor([[3.0]]))
    torch.testing.assert_close(entry['ubias'], torch.tensor([[3.0]]))
    torch.testing.assert_close(lower, torch.tensor([[2.0]]))
    torch.testing.assert_close(upper, torch.tensor([[13.0]]))


def test_forward_a_rejects_an_explicit_node_query():
    """Pure forward mode accepts only its default target-to-input query."""
    x_L = torch.tensor([[1.0, -1.0]])
    x_U = torch.tensor([[2.0, 1.0]])
    model = BoundedModule(
        nn.Linear(2, 1), (x_L + x_U) / 2, device='cpu')

    with pytest.raises(ValueError, match='needed_A_dict'):
        model.compute_bounds(
            x=(_bounded_box(x_L, x_U),), method='forward', return_A=True,
            needed_A_dict={model.output_name[0]: model.input_name})


@pytest.mark.parametrize('method', ['forward', 'CROWN'])
def test_return_a_uses_the_specification_adjusted_bias(method):
    """A specification transforms both coefficients and accumulated bias."""
    x_L = torch.tensor([[1.0, 3.0]])
    x_U = torch.tensor([[2.0, 4.0]])
    model = BoundedModule(ShiftedRelu(), (x_L + x_U) / 2, device='cpu')
    C = torch.tensor([[[1.0, -2.0]]])

    lower, upper, entries = _compute_a(
        model, (_bounded_box(x_L, x_U),), C=C, method=method)
    entry = entries[next(iter(entries))]

    torch.testing.assert_close(entry['lA'], torch.tensor([[[1.0, -2.0]]]))
    torch.testing.assert_close(entry['uA'], torch.tensor([[[1.0, -2.0]]]))
    torch.testing.assert_close(entry['lbias'], torch.tensor([[-4.0]]))
    torch.testing.assert_close(entry['ubias'], torch.tensor([[-4.0]]))
    torch.testing.assert_close(lower, torch.tensor([[-11.0]]))
    torch.testing.assert_close(upper, torch.tensor([[-8.0]]))


@pytest.mark.parametrize('method', ['forward', 'CROWN', 'alpha-CROWN'])
def test_return_a_reconstructs_bounds_for_an_unstable_activation(method):
    """Returned affine forms reproduce bounds through a relaxation."""
    x_L = torch.tensor([[-1.0, -2.0]])
    x_U = torch.tensor([[2.0, 1.0]])
    model = BoundedModule(nn.ReLU(), (x_L + x_U) / 2, device='cpu')

    lower, upper, entries = _compute_a(
        model, (_bounded_box(x_L, x_U),), method=method)
    entry = entries[next(iter(entries))]
    reconstructed = _concretize_entry(entry, x_L, x_U)

    torch.testing.assert_close(reconstructed[0], lower)
    torch.testing.assert_close(reconstructed[1], upper)
    assert not torch.equal(entry['lA'], entry['uA'])


def test_optimized_return_a_rejects_an_unaligned_default_query():
    """Optimized bounds reject a default query containing multiple inputs."""
    first_lower = torch.tensor([[1.0, -1.0]])
    first_upper = torch.tensor([[2.0, 1.0]])
    second_lower = torch.tensor([[3.0]])
    second_upper = torch.tensor([[5.0]])
    model = BoundedModule(
        TwoInputAffine(),
        ((first_lower + first_upper) / 2,
         (second_lower + second_upper) / 2),
        device='cpu')

    with pytest.raises(NotImplementedError, match='primary model output'):
        model.compute_bounds(
            x=(_bounded_box(first_lower, first_upper),
               _bounded_box(second_lower, second_upper)),
            method='alpha-CROWN', return_A=True)


@pytest.mark.parametrize(
    'method', ['IBP', 'dynamic-forward', 'dynamic-forward-optimized', 'forward-optimized'])
def test_return_a_rejects_methods_without_input_space_coefficients(method):
    """Methods without stable input-space affine forms reject return_A."""
    x_L = torch.tensor([[-1.0, -2.0]])
    x_U = torch.tensor([[2.0, 1.0]])
    model = BoundedModule(nn.ReLU(), (x_L + x_U) / 2, device='cpu')

    with pytest.raises(NotImplementedError, match='return_A'):
        model.compute_bounds(
            x=(_bounded_box(x_L, x_U),), method=method, return_A=True)


@pytest.mark.parametrize('method', ['forward', 'CROWN'])
def test_return_a_includes_only_inputs_used_by_the_target(method):
    """An intermediate target returns A only for its input ancestors."""
    first_lower = torch.tensor([[1.0, 2.0]])
    first_upper = torch.tensor([[2.0, 3.0]])
    second_lower = torch.tensor([[4.0, 5.0]])
    second_upper = torch.tensor([[6.0, 7.0]])
    model = BoundedModule(
        TwoInputBranches(),
        ((first_lower + first_upper) / 2,
         (second_lower + second_upper) / 2),
        device='cpu')
    first_name, _ = _model_input_names(model)
    target = next(
        node for node in model.nodes()
        if isinstance(node, BoundAdd) and node.name != model.output_name[0])
    inputs = (
        _bounded_box(first_lower, first_upper),
        _bounded_box(second_lower, second_upper))

    # Populate traversal state for the full output before selecting one branch.
    model.compute_bounds(x=inputs, method='CROWN')

    lower, upper, entries = _compute_a(
        model, inputs,
        target_name=target.name, method=method)

    assert set(entries) == {first_name}
    entry = entries[first_name]
    reconstructed = _concretize_entry(
        entry, first_lower, first_upper)
    identity = torch.eye(2).unsqueeze(0)
    bias = torch.ones(1, 2)
    torch.testing.assert_close(entry['lA'], identity)
    torch.testing.assert_close(entry['uA'], identity)
    torch.testing.assert_close(entry['lbias'], bias)
    torch.testing.assert_close(entry['ubias'], bias)
    torch.testing.assert_close(reconstructed[0], lower)
    torch.testing.assert_close(reconstructed[1], upper)


def test_forward_a_maps_batch_specific_sparse_image_coordinates():
    """Sparse forward variables map to each batch item's input coordinates."""
    layer = nn.Sequential(nn.Flatten(), nn.Linear(4, 1))
    with torch.no_grad():
        layer[1].weight.copy_(torch.tensor([[2.0, -3.0, 4.0, 5.0]]))
        layer[1].bias.copy_(torch.tensor([1.0]))
    x_L = torch.tensor([
        [[[1.0, 2.0], [3.0, 4.0]]],
        [[[1.0, 2.0], [3.0, 4.0]]],
    ])
    x_U = x_L.clone()
    x_U[0, 0, 0, 1] = 4.0
    x_U[1, 0, 1, 0] = 5.0
    model = BoundedModule(layer, (x_L + x_U) / 2, device='cpu')

    lower, upper, entries = _compute_a(
        model, (_bounded_box(x_L, x_U),))
    entry = entries[next(iter(entries))]
    reconstructed = _concretize_entry(entry, x_L, x_U)
    expected_A = torch.tensor([
        [[[[0.0, -3.0], [0.0, 0.0]]]],
        [[[[0.0, 0.0], [4.0, 0.0]]]],
    ])

    torch.testing.assert_close(entry['lA'], expected_A)
    torch.testing.assert_close(entry['uA'], expected_A)
    torch.testing.assert_close(entry['lbias'], torch.tensor([[35.0], [17.0]]))
    torch.testing.assert_close(entry['ubias'], torch.tensor([[35.0], [17.0]]))
    torch.testing.assert_close(reconstructed[0], lower)
    torch.testing.assert_close(reconstructed[1], upper)
    torch.testing.assert_close(lower, torch.tensor([[23.0], [29.0]]))
    torch.testing.assert_close(upper, torch.tensor([[29.0], [37.0]]))


@pytest.mark.parametrize(
    'perturbation',
    [PerturbationLpNorm(norm=2, eps=0.5), PerturbationL0Norm(eps=1)],
    ids=['l2', 'l0'])
def test_forward_a_preserves_input_perturbation_concretization(perturbation):
    """Returned A retains finite-p/L0 input perturbation semantics."""
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([
            [2.0, -3.0, 4.0],
            [-1.0, 5.0, 2.0],
        ]))
        layer.bias.copy_(torch.tensor([1.0, -2.0]))
    center = torch.tensor([[0.2, 0.4, 0.8]])
    model = BoundedModule(layer, center, device='cpu')

    lower, upper, entries = _compute_a(
        model, (BoundedTensor(center, perturbation),))
    entry = entries[next(iter(entries))]
    reconstructed = _concretize_perturbation_entry(
        entry, center, perturbation)
    expected_A = layer.weight.detach().unsqueeze(0)

    torch.testing.assert_close(entry['lA'], expected_A)
    torch.testing.assert_close(entry['uA'], expected_A)
    torch.testing.assert_close(reconstructed[0], lower)
    torch.testing.assert_close(reconstructed[1], upper)


@pytest.mark.parametrize('cached_method', ['forward', 'CROWN'])
def test_forward_a_reuses_cached_input_and_recomputes_coefficients(
        cached_method):
    """return_A recomputes coefficients omitted by an existing bound cache."""
    x_L = torch.tensor([[-1.0, -2.0]])
    x_U = torch.tensor([[2.0, 1.0]])
    bounded_input = _bounded_box(x_L, x_U)
    model = BoundedModule(nn.ReLU(), (x_L + x_U) / 2, device='cpu')

    model.compute_bounds(x=(bounded_input,), method=cached_method)
    lower, upper, A = model.compute_bounds(
        method='forward', return_A=True)
    target_name = model.output_name[0]
    entry = A[target_name][next(iter(A[target_name]))]
    reconstructed = _concretize_entry(entry, x_L, x_U)

    torch.testing.assert_close(reconstructed[0], lower)
    torch.testing.assert_close(reconstructed[1], upper)


@pytest.mark.parametrize('method', ['forward', 'CROWN'])
def test_return_a_provides_independently_concretizable_input_entries(method):
    """Each multi-input entry folds other roots into its own bias."""
    first_lower = torch.tensor([[1.0, -1.0]])
    first_upper = torch.tensor([[2.0, 1.0]])
    second_lower = torch.tensor([[3.0]])
    second_upper = torch.tensor([[5.0]])
    model = BoundedModule(
        TwoInputAffine(),
        ((first_lower + first_upper) / 2,
         (second_lower + second_upper) / 2),
        device='cpu')
    first_name, second_name = _model_input_names(model)

    lower, upper, entries = _compute_a(
        model,
        (_bounded_box(first_lower, first_upper),
         _bounded_box(second_lower, second_upper)),
        method=method)
    first_entry = entries[first_name]
    second_entry = entries[second_name]
    first_reconstructed = _concretize_entry(
        first_entry, first_lower, first_upper)
    second_reconstructed = _concretize_entry(
        second_entry, second_lower, second_upper)

    assert set(entries) == {first_name, second_name}
    torch.testing.assert_close(first_entry['lA'], torch.tensor([[[1.0, 1.0]]]))
    torch.testing.assert_close(first_entry['uA'], torch.tensor([[[1.0, 1.0]]]))
    torch.testing.assert_close(first_entry['lbias'], torch.tensor([[-10.0]]))
    torch.testing.assert_close(first_entry['ubias'], torch.tensor([[-6.0]]))
    torch.testing.assert_close(second_entry['lA'], torch.tensor([[[-2.0]]]))
    torch.testing.assert_close(second_entry['uA'], torch.tensor([[[-2.0]]]))
    torch.testing.assert_close(second_entry['lbias'], torch.tensor([[0.0]]))
    torch.testing.assert_close(second_entry['ubias'], torch.tensor([[3.0]]))
    torch.testing.assert_close(first_reconstructed[0], lower)
    torch.testing.assert_close(first_reconstructed[1], upper)
    torch.testing.assert_close(second_reconstructed[0], lower)
    torch.testing.assert_close(second_reconstructed[1], upper)
    torch.testing.assert_close(lower, torch.tensor([[-10.0]]))
    torch.testing.assert_close(upper, torch.tensor([[-3.0]]))


@pytest.mark.parametrize('method', ['forward', 'CROWN'])
def test_return_a_preserves_a_fixed_batched_branch(method):
    """A fixed branch contributes each batch item's bias to the input bound."""
    # Two batch items and three input coordinates keep the batch and
    # forward-variable axes distinct when the two branches are combined.
    fixed = torch.tensor([[1.0, 2.0, 3.0], [-3.0, -2.0, -1.0]])
    x_L = torch.tensor([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]])
    x_U = x_L + 0.5
    model = BoundedModule(
        TwoInputBranches(), (fixed, (x_L + x_U) / 2), device='cpu')
    input_name = model.input_name[1]
    expected_A = torch.eye(3).expand(2, -1, -1)
    expected_bias = fixed + 1

    # Reusing the input also exercises the cached fixed branch.
    for inputs in ((fixed, _bounded_box(x_L, x_U)), None):
        lower, upper, entries = _compute_a(model, inputs, method=method)
        assert set(entries) == {input_name}
        entry = entries[input_name]
        torch.testing.assert_close(entry['lA'], expected_A)
        torch.testing.assert_close(entry['uA'], expected_A)
        torch.testing.assert_close(entry['lbias'], expected_bias)
        torch.testing.assert_close(entry['ubias'], expected_bias)
        torch.testing.assert_close(lower, x_L + expected_bias)
        torch.testing.assert_close(upper, x_U + expected_bias)


def test_forward_a_promotes_mixed_input_dtypes():
    """Mapping multiple input coordinate dtypes uses a common dtype."""
    first_lower = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    first_upper = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    second_lower = torch.tensor([[3.0]], dtype=torch.float64)
    second_upper = torch.tensor([[5.0]], dtype=torch.float64)
    model = BoundedModule(
        TwoInputAffine(),
        ((first_lower + first_upper) / 2,
         (second_lower + second_upper) / 2),
        device='cpu')

    lower, upper, entries = _compute_a(
        model,
        (_bounded_box(first_lower, first_upper),
         _bounded_box(second_lower, second_upper)))

    for name, input_lower, input_upper in zip(
            _model_input_names(model), (first_lower, second_lower),
            (first_upper, second_upper)):
        entry = entries[name]
        reconstructed = _concretize_entry(entry, input_lower, input_upper)
        assert entry['lA'].dtype == torch.float64
        assert entry['uA'].dtype == torch.float64
        torch.testing.assert_close(reconstructed[0], lower)
        torch.testing.assert_close(reconstructed[1], upper)


def test_return_a_rejects_bounds_tightened_without_affine_coefficients():
    """Elementwise CROWN/IBP selection has no single corresponding A."""
    x_L = torch.tensor([[-1.0, -2.0]])
    x_U = torch.tensor([[2.0, 1.0]])
    model = BoundedModule(
        nn.ReLU(), (x_L + x_U) / 2, device='cpu',
        bound_opts={'compare_crown_with_ibp': True})

    with pytest.raises(NotImplementedError, match='compare_crown_with_ibp'):
        model.compute_bounds(
            x=(_bounded_box(x_L, x_U),), method='CROWN', return_A=True)

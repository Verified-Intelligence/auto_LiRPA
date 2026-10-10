"""Regression coverage for graph-owned S-shaped relaxation state."""
import copy

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


class Atan(nn.Module):
    """Minimal arctangent module for exercising BoundAtan."""

    def forward(self, x):
        """Apply arctangent elementwise."""
        return torch.atan(x)


class MixedSShaped(nn.Module):
    """Two S-shaped nodes with distinct dtypes in one graph."""

    def forward(self, x_float, x_double):
        """Combine FP32 tanh and FP64 sigmoid outputs."""
        return torch.tanh(x_float).to(torch.float64) + torch.sigmoid(x_double)


ACTIVATIONS = (nn.Tanh, nn.Sigmoid, Atan)


def _model(activation, dtype):
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(2, 4), activation(), nn.Linear(4, 1)).to(dtype)


def _box(dtype):
    return (
        torch.tensor([[-0.7, -0.2]], dtype=dtype),
        torch.tensor([[0.4, 0.9]], dtype=dtype),
    )


def _s_shaped_node(bounded_model):
    nodes = [
        node for node in bounded_model.nodes()
        if node.__class__.__name__ in {
            'BoundTanh', 'BoundSigmoid', 'BoundAtan'}
    ]
    assert len(nodes) == 1
    return nodes[0]


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('activation', ACTIVATIONS)
def test_relaxation_tables_follow_graph_dtype(dtype, activation):
    """Tangent tables are lazy, graph-owned, and default-independent."""
    model = _model(activation, dtype)
    x_l, x_u = _box(dtype)
    results = []
    previous_dtype = torch.get_default_dtype()
    try:
        for default_dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(default_dtype)
            bounded_model = BoundedModule(
                copy.deepcopy(model), x_l)
            node = _s_shaped_node(bounded_model)
            assert node.d_lower is None
            assert node.d_upper is None

            bounded_input = BoundedTensor(
                x_l, PerturbationLpNorm(x_L=x_l, x_U=x_u))
            lower, upper = bounded_model.compute_bounds(
                x=(bounded_input,), method='CROWN')

            tables = (node.d_lower, node.d_upper)
            assert all(table.dtype == dtype for table in tables)
            assert all(table.device == x_l.device for table in tables)
            assert lower.dtype == dtype
            assert upper.dtype == dtype
            results.append((lower, upper))
    finally:
        torch.set_default_dtype(previous_dtype)

    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)

    generator = torch.Generator().manual_seed(1)
    samples = x_l + torch.rand(
        4096, 2, dtype=dtype, generator=generator) * (x_u - x_l)
    with torch.no_grad():
        outputs = model(samples)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    assert (results[0][0] <= outputs.amin(dim=0) + tolerance).all()
    assert (results[0][1] >= outputs.amax(dim=0) - tolerance).all()


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('activation', ACTIVATIONS)
def test_optimized_state_follows_graph_dtype(dtype, activation):
    """S-shaped alpha variables use the owning graph node's dtype."""
    model = _model(activation, dtype)
    x_l, x_u = _box(dtype)
    other_dtype = (
        torch.float64 if dtype == torch.float32 else torch.float32)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(other_dtype)
        bounded_model = BoundedModule(
            model,
            x_l,
            bound_opts={'optimize_bound_args': {'iteration': 1}},
        )
        bounded_input = BoundedTensor(
            x_l, PerturbationLpNorm(x_L=x_l, x_U=x_u))
        lower, upper = bounded_model.compute_bounds(
            x=(bounded_input,), method='alpha-CROWN')
    finally:
        torch.set_default_dtype(previous_dtype)

    node = _s_shaped_node(bounded_model)
    assert node.alpha
    assert all(alpha.dtype == dtype for alpha in node.alpha.values())
    assert lower.dtype == dtype
    assert upper.dtype == dtype


def test_each_node_owns_tables_in_mixed_dtype_graph():
    """Each S-shaped node follows its input rather than a graph-wide dtype."""
    inputs = (
        torch.tensor([[0.1, -0.2]], dtype=torch.float32),
        torch.tensor([[0.2, -0.1]], dtype=torch.float64),
    )
    results = []
    previous_dtype = torch.get_default_dtype()
    try:
        for default_dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(default_dtype)
            bounded_model = BoundedModule(MixedSShaped(), inputs)
            bounded_inputs = tuple(
                BoundedTensor(
                    value,
                    PerturbationLpNorm(
                        x_L=value - 0.1, x_U=value + 0.1),
                )
                for value in inputs
            )
            lower, upper = bounded_model.compute_bounds(
                x=bounded_inputs, method='CROWN')

            nodes = {
                node.__class__.__name__: node
                for node in bounded_model.nodes()
                if node.__class__.__name__ in {'BoundTanh', 'BoundSigmoid'}
            }
            assert set(nodes) == {'BoundTanh', 'BoundSigmoid'}
            assert nodes['BoundTanh'].d_lower.dtype == torch.float32
            assert nodes['BoundSigmoid'].d_lower.dtype == torch.float64
            assert lower.dtype == torch.float64
            assert upper.dtype == torch.float64
            results.append((lower, upper))
    finally:
        torch.set_default_dtype(previous_dtype)

    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)


@pytest.mark.parametrize('method', ('CROWN', 'alpha-CROWN'))
def test_repeated_bounds_reuse_precomputed_tables(method):
    """Repeated bound calls do not rebuild node-local tangent tables."""
    x_l, x_u = _box(torch.float32)
    bounded_model = BoundedModule(
        _model(nn.Tanh, torch.float32), x_l,
        bound_opts={'optimize_bound_args': {'iteration': 2}},
    )
    bounded_input = BoundedTensor(
        x_l, PerturbationLpNorm(x_L=x_l, x_U=x_u))
    node = _s_shaped_node(bounded_model)

    bounded_model.compute_bounds(x=(bounded_input,), method=method)
    tables = (node.d_lower, node.d_upper)
    bounded_model.compute_bounds(x=(bounded_input,), method=method)

    assert all(
        current is cached
        for current, cached in zip(
            (node.d_lower, node.d_upper), tables)
    )

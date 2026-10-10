"""Tests for the GeLU bound operator."""
import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators.gelu import BoundGelu, GELUOp
from auto_LiRPA.perturbations import PerturbationLpNorm


GELU_MINIMIZER = -0.7517915246935645


class GeluModel(nn.Module):
    """Apply the custom GeLU operation elementwise."""

    def forward(self, x):
        return GELUOp.apply(x)


class MixedGeluModel(nn.Module):
    """Combine GeLU nodes whose inputs have different dtypes."""

    def forward(self, x_float, x_double):
        return (GELUOp.apply(x_float).to(torch.float64)
                + GELUOp.apply(x_double))


def _gelu_interval_extrema(lower, upper):
    lower_value = torch.nn.functional.gelu(lower)
    upper_value = torch.nn.functional.gelu(upper)
    exact_lower = torch.minimum(lower_value, upper_value)
    exact_upper = torch.maximum(lower_value, upper_value)

    minimizer = lower.new_full(lower.shape, GELU_MINIMIZER)
    contains_minimizer = torch.logical_and(
        lower <= minimizer, minimizer <= upper)
    exact_lower = torch.where(
        contains_minimizer,
        torch.nn.functional.gelu(minimizer),
        exact_lower,
    )
    return exact_lower, exact_upper


def _compute_gelu_bounds(lower, upper, method, default_dtype):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        center = (lower + upper) / 2
        bounded_model = BoundedModule(GeluModel(), center)
        if method == 'alpha-CROWN':
            bounded_model.set_bound_opts({
                'optimize_bound_args': {'iteration': 2}})
        bounded_input = BoundedTensor(
            center, PerturbationLpNorm(x_L=lower, x_U=upper))
        return bounded_model.compute_bounds(
            x=(bounded_input,), method=method)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('method', ('CROWN', 'alpha-CROWN', 'forward'))
def test_gelu_bounds_follow_graph_dtype(dtype, method):
    """GeLU bounds are sound and independent of the global default dtype."""
    lower = torch.tensor([[-3.0, -0.7, 0.2]], dtype=dtype)
    upper = torch.tensor([[-0.5, 0.8, 3.0]], dtype=dtype)
    bounds = [
        _compute_gelu_bounds(lower, upper, method, default_dtype)
        for default_dtype in (torch.float32, torch.float64)
    ]

    for current, expected in zip(bounds[1], bounds[0]):
        assert current.dtype == dtype
        torch.testing.assert_close(current, expected, rtol=0, atol=0)

    exact_lower, exact_upper = _gelu_interval_extrema(lower, upper)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    computed_lower, computed_upper = bounds[0]
    assert torch.all(computed_lower <= exact_lower + tolerance)
    assert torch.all(computed_upper >= exact_upper - tolerance)


def _compute_mixed_gelu_bounds(inputs, method, default_dtype):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        bounded_model = BoundedModule(MixedGeluModel(), inputs)
        bounded_inputs = tuple(
            BoundedTensor(
                value,
                PerturbationLpNorm(
                    x_L=value - value.new_tensor(0.1),
                    x_U=value + value.new_tensor(0.1),
                ),
            )
            for value in inputs
        )
        return bounded_model.compute_bounds(
            x=bounded_inputs, method=method)
    finally:
        torch.set_default_dtype(previous_dtype)


def test_mixed_dtype_gelu_bounds_are_sound():
    """Each mixed-dtype GeLU contributes sound bounds in its own dtype."""
    inputs = (
        torch.tensor([[0.1, -0.2]], dtype=torch.float32),
        torch.tensor([[0.2, -0.1]], dtype=torch.float64),
    )
    bounds = [
        _compute_mixed_gelu_bounds(inputs, 'CROWN', default_dtype)
        for default_dtype in (torch.float32, torch.float64)
    ]

    for current, expected in zip(bounds[1], bounds[0]):
        assert current.dtype == torch.float64
        torch.testing.assert_close(current, expected, rtol=0, atol=0)

    extrema = [
        _gelu_interval_extrema(
            value - value.new_tensor(0.1),
            value + value.new_tensor(0.1),
        )
        for value in inputs
    ]
    exact_lower = extrema[0][0].to(torch.float64) + extrema[1][0]
    exact_upper = extrema[0][1].to(torch.float64) + extrema[1][1]
    computed_lower, computed_upper = bounds[0]
    assert torch.all(computed_lower <= exact_lower + 1e-6)
    assert torch.all(computed_upper >= exact_upper - 1e-6)


def _gelu_table_tensors(node):
    return (
        node.d_lower_right, node.d_upper_right,
        node.d_lower_left, node.d_upper_left,
    )


def test_gelu_table_lifecycle_and_ownership():
    """GeLU tangent tables are lazy, node-owned, and reusable."""
    inputs = (
        torch.tensor([[0.1, -0.2]], dtype=torch.float32),
        torch.tensor([[0.2, -0.1]], dtype=torch.float64),
    )
    bounded_model = BoundedModule(MixedGeluModel(), inputs)
    bounded_inputs = tuple(
        BoundedTensor(
            value,
            PerturbationLpNorm(
                x_L=value - value.new_tensor(0.1),
                x_U=value + value.new_tensor(0.1),
            ),
        )
        for value in inputs
    )
    nodes = [
        node for node in bounded_model.nodes()
        if isinstance(node, BoundGelu)
    ]
    assert len(nodes) == 2
    assert all(
        all(table is None for table in _gelu_table_tensors(node))
        for node in nodes
    )

    bounded_model.compute_bounds(x=bounded_inputs, method='CROWN')
    table_ids = []
    for node in nodes:
        tables = _gelu_table_tensors(node)
        assert all(table.dtype == node.inputs[0].lower.dtype
                   for table in tables)
        assert all(table.device == node.inputs[0].lower.device
                   for table in tables)
        table_ids.append(tuple(id(table) for table in tables))

    bounded_model.compute_bounds(x=bounded_inputs, method='CROWN')
    assert [
        tuple(id(table) for table in _gelu_table_tensors(node))
        for node in nodes
    ] == table_ids

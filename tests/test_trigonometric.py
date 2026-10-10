"""Tests for periodic trigonometric bound operators."""
import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators.base import OperatorCapabilityError
from auto_LiRPA.operators.trigonometric import BoundCos, BoundSin
from auto_LiRPA.perturbations import PerturbationLpNorm


class PeriodicModel(nn.Module):
    """Apply one periodic activation elementwise."""

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class MixedPeriodicModel(nn.Module):
    """Combine Sin and Cos nodes whose input bounds have different dtypes."""

    def forward(self, x_float, x_double):
        return torch.sin(x_float).to(torch.float64) + torch.cos(x_double)


def _periodic_table_state(node):
    """Return all periodic table fields, including uninitialized entries."""
    return (
        node.all_table_x, node.d_lower, node.d_upper,
        node.xl_lower_tb, node.xl_upper_tb,
        node.xu_lower_tb, node.xu_upper_tb,
    )


def _periodic_table_tensors(node):
    """Return every tensor owned by a Sin/Cos relaxation table."""
    tensors = [node.all_table_x, node.d_lower, node.d_upper]
    for table in (
            node.xl_lower_tb, node.xl_upper_tb,
            node.xu_lower_tb, node.xu_upper_tb):
        tensors.extend(table)
    return tensors


def _exact_periodic_bounds(function, lower, upper):
    """Return exact ranges for the fixed intervals used by these tests."""
    if function is torch.sin:
        exact_lower = torch.stack((
            torch.sin(lower[0, 0]),
            torch.sin(lower[0, 1]),
            lower.new_tensor(-1),
        )).unsqueeze(0)
        exact_upper = torch.stack((
            torch.sin(upper[0, 0]),
            upper.new_tensor(1),
            upper.new_tensor(1),
        )).unsqueeze(0)
    else:
        exact_lower = torch.stack((
            torch.cos(upper[0, 0]),
            torch.cos(upper[0, 1]),
            lower.new_tensor(-1),
        )).unsqueeze(0)
        exact_upper = torch.stack((
            upper.new_tensor(1),
            upper.new_tensor(1),
            torch.cos(lower[0, 2]),
        )).unsqueeze(0)
    return exact_lower, exact_upper


@pytest.mark.parametrize('function', [torch.sin, torch.cos])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('method', ['CROWN', 'alpha-CROWN', 'forward'])
def test_periodic_bounds_are_default_independent(function, dtype, method):
    """Periodic bounds are sound and independent of the global default."""
    lower = torch.tensor([[-7.0, -1.0, 0.2]], dtype=dtype)
    upper = torch.tensor([[-5.0, 2.0, 6.0]], dtype=dtype)
    exact_lower, exact_upper = _exact_periodic_bounds(
        function, lower, upper)
    options = {}
    if method == 'alpha-CROWN':
        options['optimize_bound_args'] = {'iteration': 2}

    results = []
    previous_dtype = torch.get_default_dtype()
    try:
        for default_dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(default_dtype)
            center = (lower + upper) / 2
            bounded_input = BoundedTensor(
                center, PerturbationLpNorm(x_L=lower, x_U=upper))
            bounded_model = BoundedModule(
                PeriodicModel(function), center, bound_opts=options)

            lb, ub = bounded_model.compute_bounds(
                x=(bounded_input,), method=method)
            assert lb.dtype == dtype
            assert ub.dtype == dtype
            tolerance = 8 * torch.finfo(dtype).eps
            assert (lb <= exact_lower + tolerance).all()
            assert (ub >= exact_upper - tolerance).all()
            results.append((lb, ub))
    finally:
        torch.set_default_dtype(previous_dtype)

    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)


@pytest.mark.parametrize('function,node_type', [
    (torch.sin, BoundSin),
    (torch.cos, BoundCos),
])
def test_bound_methods_share_the_same_periodic_tables(function, node_type):
    """CROWN, forward, and optimized bounds reuse one valid table set."""
    lower = torch.tensor([[-7.0, -1.0, 0.2]])
    upper = torch.tensor([[-5.0, 2.0, 6.0]])
    center = (lower + upper) / 2
    bounded_input = BoundedTensor(
        center, PerturbationLpNorm(x_L=lower, x_U=upper))
    bounded_model = BoundedModule(PeriodicModel(function), center)
    node = next(
        node for node in bounded_model.nodes()
        if isinstance(node, node_type))
    assert all(table is None for table in _periodic_table_state(node))

    bounded_model.compute_bounds(x=(bounded_input,), method='CROWN')
    cached_tables = _periodic_table_tensors(node)
    assert all(table.dtype == lower.dtype for table in cached_tables)
    assert all(table.device == lower.device for table in cached_tables)

    bounded_model.compute_bounds(x=(bounded_input,), method='forward')
    assert all(
        current is cached for current, cached in zip(
            _periodic_table_tensors(node), cached_tables))

    bounded_model.set_bound_opts({
        'optimize_bound_args': {'iteration': 2}})
    bounded_model.compute_bounds(x=(bounded_input,), method='alpha-CROWN')
    assert all(
        current is cached for current, cached in zip(
            _periodic_table_tensors(node), cached_tables))


@pytest.mark.parametrize('function,center', [
    (torch.sin, -4 * torch.pi),
    (torch.cos, 3.5 * torch.pi),
])
def test_periodic_same_slope_is_rejected(function, center):
    """Periodic operators fail closed for the unsupported same-slope mode."""
    lower = torch.tensor([[center - 0.01]])
    upper = torch.tensor([[center + 0.01]])
    value = (lower + upper) / 2
    bounded_model = BoundedModule(
        PeriodicModel(function), value,
        bound_opts={'activation_bound_option': 'same-slope'},
    )
    bounded_input = BoundedTensor(
        value, PerturbationLpNorm(x_L=lower, x_U=upper))

    with pytest.raises(OperatorCapabilityError):
        bounded_model.compute_bounds(x=(bounded_input,), method='CROWN')


def test_each_periodic_node_owns_tables_in_mixed_dtype_graph():
    """Sin and Cos tables follow their own inputs in a mixed-dtype graph."""
    inputs = (
        torch.tensor([[0.1, -0.2]], dtype=torch.float32),
        torch.tensor([[0.2, -0.1]], dtype=torch.float64),
    )
    bounded_model = BoundedModule(MixedPeriodicModel(), inputs)
    bounded_inputs = tuple(
        BoundedTensor(
            value,
            PerturbationLpNorm(x_L=value - 0.1, x_U=value + 0.1),
        )
        for value in inputs
    )

    lower, upper = bounded_model.compute_bounds(
        x=bounded_inputs, method='CROWN')
    nodes = {
        type(node): node for node in bounded_model.nodes()
        if isinstance(node, (BoundSin, BoundCos))
    }

    assert set(nodes) == {BoundSin, BoundCos}
    assert all(
        tensor.dtype == torch.float32
        for tensor in _periodic_table_tensors(nodes[BoundSin]))
    assert all(
        tensor.dtype == torch.float64
        for tensor in _periodic_table_tensors(nodes[BoundCos]))
    assert all(
        tensor.device.type == 'cpu'
        for node in nodes.values()
        for tensor in _periodic_table_tensors(node))
    assert lower.dtype == torch.float64
    assert upper.dtype == torch.float64

    sin_lower = bounded_inputs[0].ptb.x_L
    sin_upper = bounded_inputs[0].ptb.x_U
    cos_lower = bounded_inputs[1].ptb.x_L
    cos_upper = bounded_inputs[1].ptb.x_U
    exact_lower = (
        torch.sin(sin_lower).to(torch.float64)
        + torch.minimum(torch.cos(cos_lower), torch.cos(cos_upper)))
    exact_upper = (
        torch.sin(sin_upper).to(torch.float64)
        + torch.maximum(torch.cos(cos_lower), torch.cos(cos_upper)))
    tolerance = 8 * torch.finfo(torch.float32).eps
    assert (lower <= exact_lower + tolerance).all()
    assert (upper >= exact_upper - tolerance).all()

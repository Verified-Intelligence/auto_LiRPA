"""Regression tests for dtype handling across bound algorithms."""
import copy

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.beta_crown import SparseBeta
from auto_LiRPA.perturbations import PerturbationLpNorm


class MixedDtypeBinaryModel(nn.Module):
    """Binary operation whose two perturbed inputs have different dtypes."""

    def __init__(self, operation, reverse_inputs):
        super().__init__()
        self.operation = operation
        self.reverse_inputs = reverse_inputs

    def forward(self, x):
        first, second = x.float(), x.double()
        if self.reverse_inputs:
            first, second = second, first
        if self.operation == 'mul':
            return first * second
        if self.operation == 'min':
            return torch.minimum(first, second)
        if self.operation == 'max':
            return torch.maximum(first, second)
        raise ValueError(f'Unsupported operation {self.operation}')


class ExplicitMatMulModel(nn.Module):
    """Network whose weights remain parameter roots of MatMul operators."""

    def __init__(self):
        super().__init__()
        self.hidden_weight = nn.Parameter(torch.randn(3, 2))
        self.output_weight = nn.Parameter(torch.randn(1, 3))

    def forward(self, x):
        return torch.relu(x @ self.hidden_weight.t()) @ self.output_weight.t()


def _compute_box_bounds(model, dtype, method, default_dtype):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        center = torch.tensor([[0.2, -0.3]], dtype=dtype)
        bounded_model = BoundedModule(
            copy.deepcopy(model),
            center,
            bound_opts={'optimize_bound_args': {'iteration': 3}},
        )
        bounded_input = BoundedTensor(
            center,
            PerturbationLpNorm(norm=float('inf'), eps=0.1),
        )
        return bounded_model.compute_bounds(x=(bounded_input,), method=method)
    finally:
        torch.set_default_dtype(previous_dtype)


def test_sparse_beta_state_transforms_remain_optimizer_parameters():
    sparse_beta = SparseBeta((2, 1), dtype=torch.float64)
    converted = sparse_beta.to(device='cpu', dtype=torch.float32)

    assert converted.val.dtype == torch.float32
    assert converted.sign.dtype == torch.float32
    assert converted.loc.dtype == torch.int64
    assert converted.val.requires_grad
    assert converted.val.is_leaf

    converted.mask_batch_dim_inplace(torch.tensor([True, False]))
    assert converted.val.requires_grad
    assert converted.val.is_leaf

    optimizer = torch.optim.Adam([converted.val], lr=0.1)
    converted.val.sum().backward()
    optimizer.step()


def test_float64_batched_crown_preserves_dtype():
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    ).double()
    center = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    reference_model = BoundedModule(
        copy.deepcopy(model),
        center,
    )
    batched_model = BoundedModule(
        copy.deepcopy(model), center, bound_opts={'crown_batch_size': 1})
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(norm=float('inf'), eps=0.1),
    )

    reference_lower, reference_upper = reference_model.compute_bounds(
        x=(bounded_input,), method='CROWN')
    lower, upper = batched_model.compute_bounds(
        x=(bounded_input,), method='CROWN')

    assert lower.dtype == torch.float64
    assert upper.dtype == torch.float64
    torch.testing.assert_close(lower, reference_lower)
    torch.testing.assert_close(upper, reference_upper)


def test_float64_fresh_alpha_crown_preserves_dtype():
    torch.manual_seed(2)
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    ).double()
    center = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    bounded_model = BoundedModule(
        model,
        center,
        bound_opts={'optimize_bound_args': {'iteration': 1}},
    )
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(norm=float('inf'), eps=0.1),
    )

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='alpha-CROWN')

    alpha = [
        value for node in bounded_model.nodes()
        for value in getattr(node, 'alpha', {}).values()
    ]
    assert alpha
    assert all(value.dtype == torch.float64 for value in alpha)
    assert lower.dtype == torch.float64
    assert upper.dtype == torch.float64


def test_float64_output_constraint_gammas_preserve_dtype():
    torch.manual_seed(3)
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    ).double()
    center = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    bounded_model = BoundedModule(
        model,
        center,
        bound_opts={'optimize_bound_args': {
            'apply_output_constraints_to': ['BoundLinear'],
            'iteration': 1,
        }},
    )
    bounded_model.constraints = torch.ones(
        1, 1, 1, dtype=torch.float64)
    bounded_model.thresholds = torch.zeros(1, dtype=torch.float64)
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(norm=float('inf'), eps=0.1),
    )

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='alpha-CROWN')

    gammas = [
        node.gammas_underlying_tensor
        for node in bounded_model.nodes()
        if hasattr(node, 'gammas_underlying_tensor')
    ]
    assert gammas
    assert all(gamma.dtype == torch.float64 for gamma in gammas)
    assert lower.dtype == torch.float64
    assert upper.dtype == torch.float64


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_matmul_parameter_root_specification_uses_graph_dtype(dtype):
    torch.manual_seed(4)
    model = ExplicitMatMulModel().to(dtype)
    other_dtype = (
        torch.float64 if dtype == torch.float32 else torch.float32)

    lower, upper = _compute_box_bounds(
        model, dtype, method='CROWN', default_dtype=other_dtype)

    assert lower.dtype == dtype
    assert upper.dtype == dtype
    assert (lower <= upper).all()


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_optimized_bounds_do_not_depend_on_default_dtype(dtype):
    torch.manual_seed(5)
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    ).to(dtype)

    bounds = [
        _compute_box_bounds(
            model, dtype, method='CROWN-Optimized', default_dtype=default_dtype)
        for default_dtype in (torch.float32, torch.float64)
    ]

    for lower, upper in bounds:
        assert lower.dtype == dtype
        assert upper.dtype == dtype
    torch.testing.assert_close(bounds[0][0], bounds[1][0], rtol=0, atol=0)
    torch.testing.assert_close(bounds[0][1], bounds[1][1], rtol=0, atol=0)

    center = torch.tensor([[0.2, -0.3]], dtype=dtype)
    generator = torch.Generator().manual_seed(6)
    samples = center - 0.1 + 0.2 * torch.rand(
        4096, 2, dtype=dtype, generator=generator)
    with torch.no_grad():
        sample_outputs = model(samples)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    for lower, upper in bounds:
        assert (lower <= sample_outputs.amin(dim=0) + tolerance).all()
        assert (upper >= sample_outputs.amax(dim=0) - tolerance).all()


@pytest.mark.parametrize('operation', ('mul', 'min', 'max'))
@pytest.mark.parametrize('reverse_inputs', (False, True))
@pytest.mark.parametrize('method', ('CROWN', 'alpha-CROWN'))
def test_mixed_dtype_nonlinear_relaxation_uses_output_dtype(
        operation, reverse_inputs, method):
    center = torch.tensor([[0.2, -0.3]], dtype=torch.float32)
    model = MixedDtypeBinaryModel(operation, reverse_inputs)
    bounded_model = BoundedModule(
        model,
        center,
        bound_opts={'optimize_bound_args': {'iteration': 1}},
    )
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(norm=float('inf'), eps=0.1),
    )

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_input,), method=method)

    nonlinear_node = bounded_model.final_node()
    assert nonlinear_node.output_dtype == torch.float64
    assert torch.isfinite(lower).all()
    assert torch.isfinite(upper).all()
    assert (lower <= upper).all()
    samples = torch.cat((center - 0.1, center, center + 0.1), dim=0)
    sample_outputs = model(samples)
    assert (lower <= sample_outputs.amin(dim=0) + 1e-7).all()
    assert (upper >= sample_outputs.amax(dim=0) - 1e-7).all()
    if method == 'alpha-CROWN':
        assert nonlinear_node.alpha
        assert all(
            alpha.dtype == nonlinear_node.output_dtype
            for alpha in nonlinear_node.alpha.values()
        )

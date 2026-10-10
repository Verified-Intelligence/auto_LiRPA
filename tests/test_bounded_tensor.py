"""Tests for explicit BoundedTensor conversion."""

import pytest
import torch

from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def test_to_meta_converts_owned_perturbation_bounds():
    center = torch.tensor([[0.3, 0.4]], dtype=torch.float32)
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(
            norm=float('inf'), x_L=center - 0.1, x_U=center + 0.1),
    )

    converted = bounded_input.to(device='meta')

    assert isinstance(converted, BoundedTensor)
    assert converted.device.type == 'meta'
    assert converted.ptb.x_L.device.type == 'meta'
    assert converted.ptb.x_U.device.type == 'meta'


def test_to_preserves_source_and_nonleaf_gradients():
    source = torch.tensor(
        [[0.3, 0.4]], dtype=torch.float32, requires_grad=True)
    perturbation = PerturbationLpNorm(
        norm=float('inf'), x_L=source - 0.1, x_U=source + 0.1)
    bounded_input = BoundedTensor(source, perturbation)

    converted = bounded_input.double()

    assert converted.ptb is not perturbation
    assert converted.ptb.x_L.dtype == torch.float64
    assert perturbation.x_L.dtype == torch.float32
    converted.ptb.x_L.sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source))


@pytest.mark.parametrize(
    ('method', 'dtype'),
    (
        ('double', torch.float64),
        ('half', torch.float16),
        ('bfloat16', torch.bfloat16),
        ('float', torch.float32),
    ),
)
def test_dtype_helpers_convert_tensor_and_perturbation(method, dtype):
    center = torch.tensor([[0.3, 0.4]], dtype=torch.float32)
    perturbation = PerturbationLpNorm(
        norm=float('inf'), eps=torch.tensor(0.1, dtype=torch.float32))
    bounded_input = BoundedTensor(center, perturbation)

    converted = getattr(bounded_input, method)()

    assert isinstance(converted, BoundedTensor)
    assert converted.dtype == dtype
    assert converted.ptb is not perturbation
    assert converted.ptb.eps.dtype == dtype
    assert perturbation.eps.dtype == torch.float32

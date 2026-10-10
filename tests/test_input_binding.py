"""Tests for binding runtime model inputs to input graph nodes."""

import pytest
import torch
from torch import nn

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.operators.leaf import BoundInput


class UsesSecondInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([2.0]))
        self.register_buffer('offset', torch.tensor([1.0]))

    def forward(self, unused, used):
        return used * self.scale + self.offset


class AddsInputs(nn.Module):
    def forward(self, x, y):
        return x + y


class MultipliesInputs(nn.Module):
    def forward(self, x, scale):
        return x * scale


class UsesSecondReluInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1, dtype=torch.float64)
        with torch.no_grad():
            self.layer.weight.fill_(2.)
            self.layer.bias.fill_(1.)

    def forward(self, unused, used):
        return torch.relu(self.layer(used))


def test_unused_input_keeps_original_model_input_position():
    unused = torch.tensor([[1.0]])
    used = torch.tensor([[2.0]])
    bounded_model = BoundedModule(UsesSecondInput(), (unused, used))

    model_inputs = [
        bounded_model[name] for name in bounded_model.input_name
        if type(bounded_model[name]) is BoundInput]
    assert len(model_inputs) == 1
    assert model_inputs[0].model_input_index == 1
    non_model_inputs = [
        bounded_model[name] for name in bounded_model.input_name
        if type(bounded_model[name]) is not BoundInput]
    assert len(non_model_inputs) == 2
    assert all(node.model_input_index is None for node in non_model_inputs)
    assert all(not node.input_dependent for node in non_model_inputs)
    assert all(node.batch_dim == -1 for node in non_model_inputs)

    new_unused = torch.tensor([[3.0]])
    new_used = torch.tensor([[4.0]])
    torch.testing.assert_close(
        bounded_model(new_unused, new_used), new_used * 2 + 1)

    bounded_used = BoundedTensor(
        new_used,
        PerturbationLpNorm(
            norm=torch.inf, x_L=new_used - 0.5, x_U=new_used + 0.5))
    lower, upper = bounded_model.compute_bounds(
        x=(new_unused, bounded_used), method='IBP')
    torch.testing.assert_close(lower, (new_used - 0.5) * 2 + 1)
    torch.testing.assert_close(upper, (new_used + 0.5) * 2 + 1)


def test_nested_runtime_inputs_are_flattened_for_rebinding():
    x = torch.tensor([[1.0]])
    y = torch.tensor([[2.0]])
    bounded_model = BoundedModule(AddsInputs(), (x, y))

    new_x = torch.tensor([[3.0]])
    new_y = torch.tensor([[4.0]])
    torch.testing.assert_close(
        bounded_model((new_x, new_y)), new_x + new_y)


@pytest.mark.parametrize('return_generators', [False, True])
def test_optimized_relaxation_rebinds_the_original_model_input_positions(return_generators):
    unused = torch.tensor([[7.]], dtype=torch.float64)
    used = torch.zeros(1, 1, dtype=torch.float64)
    bm = BoundedModule(UsesSecondReluInput(), (unused, used), device='cpu',
                       bound_opts={'optimize_bound_args': {'iteration': 0}})
    bx = BoundedTensor(used, PerturbationLpNorm(x_L=used - 1, x_U=used + 1))
    kwargs = dict(method='dynamic-forward-optimized', return_generators=return_generators)
    fresh = bm.compute_bounds(x=(unused, bx), **kwargs)
    reused = bm.compute_bounds(**kwargs)
    for result in (fresh, reused):
        torch.testing.assert_close(result[0], torch.tensor([[-.75]], dtype=torch.float64))
        torch.testing.assert_close(result[1], torch.tensor([[3.]], dtype=torch.float64))
        if return_generators:
            G, c = result[2][bm.final_name]
            torch.testing.assert_close(G, torch.tensor([[[1.5, .375]]], dtype=torch.float64))
            torch.testing.assert_close(c, torch.tensor([[1.125]], dtype=torch.float64))


def test_unperturbed_model_input_keeps_batch_semantics_in_forward_bounds():
    x = torch.tensor([[1.0], [2.0]])
    scale = torch.tensor([[2.0], [3.0]])
    bounded_model = BoundedModule(MultipliesInputs(), (x, scale))
    bounded_x = BoundedTensor(
        x, PerturbationLpNorm(
            norm=torch.inf, x_L=x - 0.5, x_U=x + 0.5))

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x, scale), method='forward')

    torch.testing.assert_close(lower, (x - 0.5) * scale)
    torch.testing.assert_close(upper, (x + 0.5) * scale)


def test_graph_editing_updates_model_input_binding():
    unused = torch.tensor([[1.0]])
    used = torch.tensor([[2.0]])
    bounded_model = BoundedModule(UsesSecondInput(), (unused, used))
    extra = BoundInput(
        'extra', torch.zeros_like(used), options=bounded_model.bound_opts,
        attr={'device': bounded_model.device})
    extra.name = '/extra'

    bounded_model.add_input_node(extra, model_input_index=2)
    assert extra.model_input_index == 2

    extra_value = torch.tensor([[5.0]])
    bounded_model.set_input(unused, used, extra_value)
    torch.testing.assert_close(
        bounded_model.get_forward_value(extra), extra_value)

    bounded_model.delete_node(extra)
    assert extra.name not in bounded_model.input_name
    assert extra.name not in bounded_model.root_names

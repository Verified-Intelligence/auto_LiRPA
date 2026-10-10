import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedParameter, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.perturbations import PerturbationL0Norm, PerturbationLpNorm


class ConcatLinearModel(nn.Module):
    def __init__(self, second_input_op=None):
        super().__init__()
        self.second_input_op = (
            second_input_op if second_input_op is not None else nn.Identity())
        self.linear = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 2.0]]))

    def forward(self, x, y):
        return self.linear(torch.cat(
            [x, self.second_input_op(y)], dim=1))


def test_ibp_requires_interval_results():
    center = torch.zeros(1, 2)
    bounded_x = BoundedTensor(
        center, PerturbationLpNorm(norm=torch.inf, eps=0.5))
    bounded_model = BoundedModule(nn.ReLU(), center, device='cpu')
    relu = next(
        node for node in bounded_model.nodes()
        if isinstance(node, BoundRelu))
    relu.interval_propagate = lambda x: (x.lower, x.upper)

    with pytest.raises(TypeError, match='must return Interval'):
        bounded_model.compute_bounds(x=(bounded_x,), method='IBP')


@pytest.mark.parametrize('start_dim', [1, -2])
def test_flatten_preserves_finite_p_input_for_linear_ibp(start_dim):
    model = nn.Sequential(
        nn.Flatten(start_dim=start_dim), nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[1].weight.fill_(1.0)
    center = torch.zeros(1, 1, 2)
    perturbation = PerturbationLpNorm(norm=2, eps=1.0)
    bounded_x = BoundedTensor(center, perturbation)
    bounded_model = BoundedModule(model, center, device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), method='IBP')

    expected = torch.full_like(lower, 2 ** 0.5)
    torch.testing.assert_close(lower, -expected)
    torch.testing.assert_close(upper, expected)


@pytest.mark.parametrize(
    'eps',
    [torch.tensor([0.5, 1.0]), torch.tensor([[0.5], [1.0]])],
    ids=['vector', 'column'])
def test_batched_finite_p_radius_broadcasts_over_input_dimensions(eps):
    model = nn.Sequential(nn.Flatten(), nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[1].weight.fill_(1.0)
    center = torch.zeros(2, 1, 2)
    bounded_x = BoundedTensor(
        center, PerturbationLpNorm(norm=2, eps=eps))
    bounded_model = BoundedModule(model, center, device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), method='IBP')

    expected = torch.tensor([[0.5], [1.0]]) * (2 ** 0.5)
    torch.testing.assert_close(lower, -expected)
    torch.testing.assert_close(upper, expected)


def test_finite_p_input_uses_specification_norm_bound():
    center = torch.zeros(1, 1, 2)
    bounded_x = BoundedTensor(
        center, PerturbationLpNorm(norm=2, eps=1.0))
    bounded_model = BoundedModule(nn.Flatten(), center, device='cpu')
    specification = torch.ones(1, 1, 2)

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), C=specification, method='IBP')

    expected = torch.full_like(lower, 2 ** 0.5)
    torch.testing.assert_close(lower, -expected)
    torch.testing.assert_close(upper, expected)


def test_l0_input_uses_linear_weight_bound():
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([
            [1.0, -3.0, 2.0],
            [4.0, 0.5, -2.0],
        ]))
        model.bias.copy_(torch.tensor([0.5, -1.0]))
    center = torch.tensor([[0.2, 0.3, 0.4]])
    bounded_x = BoundedTensor(
        center, PerturbationL0Norm(eps=1, ratio=0.5))
    bounded_model = BoundedModule(model, center, device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), method='IBP')

    output_center = model(center)
    deviation = torch.tensor([[1.5, 2.0]])
    torch.testing.assert_close(lower, output_center - deviation)
    torch.testing.assert_close(upper, output_center + deviation)


def test_l2_input_uses_conv_filter_norm_bound():
    model = nn.Conv2d(1, 2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([
            [[[1.0, 2.0], [-1.0, 0.0]]],
            [[[0.5, -0.5], [0.5, -0.5]]],
        ]))
    center = torch.zeros(1, 1, 2, 2)
    bounded_x = BoundedTensor(
        center, PerturbationLpNorm(norm=2, eps=0.5))
    bounded_model = BoundedModule(model, center, device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), method='IBP')

    deviation = torch.tensor([0.5 * 6 ** 0.5, 0.5]).view(1, 2, 1, 1)
    torch.testing.assert_close(lower, -deviation)
    torch.testing.assert_close(upper, deviation)


def test_l0_input_uses_conv_weight_bound():
    model = nn.Conv2d(2, 1, 2, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[[
            [1.0, 4.0], [2.0, 0.0]],
            [[3.0, 1.0], [1.0, 2.0]],
        ]]))
        model.bias.fill_(0.1)
    center = torch.full((1, 2, 3, 3), 0.25)
    bounded_x = BoundedTensor(
        center, PerturbationL0Norm(eps=2, ratio=0.5))
    bounded_model = BoundedModule(model, center, device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), method='IBP')

    output_center = model(center)
    # Per-position channel sums are [4, 5, 3, 2]; the two largest contribute.
    deviation = torch.full_like(output_center, (5.0 + 4.0) * 0.5)
    torch.testing.assert_close(lower, output_center - deviation)
    torch.testing.assert_close(upper, output_center + deviation)


def test_non_l2_input_uses_safe_box_for_conv():
    model = nn.Conv2d(1, 1, (1, 2), bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    center = torch.zeros(1, 1, 1, 2)
    bounded_x = BoundedTensor(
        center, PerturbationLpNorm(norm=1, eps=1.0))
    bounded_model = BoundedModule(model, center, device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), method='IBP')

    # The enclosing input box is [-1, 1]^2, giving the conservative [-2, 2].
    expected = torch.full_like(lower, 2.0)
    torch.testing.assert_close(lower, -expected)
    torch.testing.assert_close(upper, expected)


def test_finite_p_input_before_nonlinearity_uses_safe_box():
    model = nn.Sequential(nn.ReLU(), nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[1].weight.fill_(1.0)
    center = torch.tensor([[-1.0, 0.5]])
    bounded_x = BoundedTensor(
        center, PerturbationLpNorm(norm=2, eps=2.0))
    bounded_model = BoundedModule(model, center, device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), method='IBP')

    torch.testing.assert_close(lower, torch.zeros_like(lower))
    torch.testing.assert_close(upper, torch.tensor([[3.5]]))


def test_concat_combines_l2_input_perturbations():
    model = ConcatLinearModel()
    center_x = torch.zeros(1, 1)
    center_y = torch.zeros(1, 1)
    eps_x, eps_y = 1.0, 1.0
    x = BoundedTensor(
        center_x, PerturbationLpNorm(norm=2, eps=eps_x))
    y = BoundedTensor(
        center_y, PerturbationLpNorm(norm=2, eps=eps_y))
    bounded_model = BoundedModule(
        model, (center_x, center_y), device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(x, y), method='IBP')

    combined_eps = (eps_x ** 2 + eps_y ** 2) ** 0.5
    expected = torch.full_like(lower, combined_eps * (5 ** 0.5))
    torch.testing.assert_close(lower, -expected)
    torch.testing.assert_close(upper, expected)


def test_concat_uses_box_after_a_branch_loses_l2_state():
    model = ConcatLinearModel(second_input_op=nn.ReLU())
    center_x = torch.zeros(1, 1)
    center_y = torch.zeros(1, 1)
    x = BoundedTensor(
        center_x, PerturbationLpNorm(norm=2, eps=1.0))
    y = BoundedTensor(
        center_y, PerturbationLpNorm(norm=2, eps=1.0))
    bounded_model = BoundedModule(
        model, (center_x, center_y), device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(x, y), method='IBP')

    torch.testing.assert_close(lower, torch.tensor([[-1.0]]))
    torch.testing.assert_close(upper, torch.tensor([[3.0]]))


@pytest.mark.parametrize('operation', ['unsqueeze', 'squeeze'])
def test_shape_operation_preserves_finite_p_input(operation):
    class ShapeLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, x):
            if operation == 'unsqueeze':
                x = x.unsqueeze(-1).flatten(1)
            else:
                x = x.squeeze(-1)
            return self.linear(x)

    shape = (1, 2) if operation == 'unsqueeze' else (1, 2, 1)
    center = torch.zeros(shape)
    bounded_x = BoundedTensor(
        center, PerturbationLpNorm(norm=2, eps=1.0))
    bounded_model = BoundedModule(
        ShapeLinearModel(), center, device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), method='IBP')

    expected = torch.full_like(lower, 2 ** 0.5)
    torch.testing.assert_close(lower, -expected)
    torch.testing.assert_close(upper, expected)


def test_padding_preserves_finite_p_input_state():
    model = nn.Sequential(
        nn.ZeroPad2d((1, 0, 0, 0)),
        nn.Flatten(),
        nn.Linear(3, 1, bias=False))
    with torch.no_grad():
        model[2].weight.copy_(torch.tensor([[0.0, 1.0, 1.0]]))
    center = torch.zeros(1, 1, 1, 2)
    bounded_x = BoundedTensor(
        center, PerturbationLpNorm(norm=2, eps=1.0))
    bounded_model = BoundedModule(model, center, device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), method='IBP')

    expected = torch.full_like(lower, 2 ** 0.5)
    torch.testing.assert_close(lower, -expected)
    torch.testing.assert_close(upper, expected)


def test_split_preserves_finite_p_input_state():
    class SplitLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, x):
            return self.linear(torch.split(x, 2, dim=1)[0])

    center = torch.zeros(1, 4)
    bounded_x = BoundedTensor(
        center, PerturbationLpNorm(norm=2, eps=1.0))
    bounded_model = BoundedModule(
        SplitLinearModel(), center, device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), method='IBP')

    expected = torch.full_like(lower, 2 ** 0.5)
    torch.testing.assert_close(lower, -expected)
    torch.testing.assert_close(upper, expected)


def test_batch_concat_drops_finite_p_input_state():
    class BatchConcatLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, x, y):
            return self.linear(torch.cat([x, y], dim=0))

    center_x = torch.zeros(2, 1)
    center_y = torch.zeros(2, 1)
    eps_x = torch.tensor([[0.5], [1.0]])
    eps_y = torch.tensor([[1.5], [2.0]])
    x = BoundedTensor(
        center_x, PerturbationLpNorm(norm=2, eps=eps_x))
    y = BoundedTensor(
        center_y, PerturbationLpNorm(norm=2, eps=eps_y))
    bounded_model = BoundedModule(
        BatchConcatLinearModel(), (center_x, center_y), device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(x, y), method='IBP')

    expected = torch.cat([eps_x, eps_y], dim=0)
    torch.testing.assert_close(lower, -expected)
    torch.testing.assert_close(upper, expected)


@pytest.mark.parametrize('operator', ['linear', 'conv'])
def test_finite_p_input_with_perturbed_affine_bias(operator):
    if operator == 'linear':
        model = nn.Linear(1, 1)
        center = torch.zeros(1, 1)
    else:
        model = nn.Conv2d(1, 1, 1)
        center = torch.zeros(1, 1, 1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    model.bias = BoundedParameter(
        model.bias.detach(),
        PerturbationLpNorm(norm=torch.inf, eps=0.25))
    bounded_x = BoundedTensor(
        center, PerturbationLpNorm(norm=2, eps=1.0))
    bounded_model = BoundedModule(model, center, device='cpu')

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_x,), method='IBP')

    expected = torch.full_like(lower, 0.25)
    torch.testing.assert_close(lower, -expected)
    torch.testing.assert_close(upper, expected)

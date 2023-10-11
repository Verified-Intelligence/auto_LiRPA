import torch
from .base import Bound


class JacobianOP(torch.autograd.Function):
    @staticmethod
    def symbolic(g, output, input):
        return g.op('grad::jacobian', output, input).setType(output.type())

    @staticmethod
    def forward(ctx, output, input):
        output_ = output.flatten(1)
        return torch.zeros(
            output.shape[0], output_.shape[-1], *input.shape[1:],
            device=output.device)


class BoundJacobianOP(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, output, input):
        return JacobianOP.apply(output, input)


class BoundJacobianInit(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.never_perturbed = True

    def forward(self, x):
        x = x.flatten(1)
        eye = torch.eye(x.shape[-1], device=x.device)
        return eye.unsqueeze(0).repeat(x.shape[0], 1, 1)

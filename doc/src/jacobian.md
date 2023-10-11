# APIs for Jacobian

## Specifying a Jacobian computation in the model

When defining a computational graph by creating a `torch.nn.Module`, a Jacobian computation can be introduced with `JacobianOP.apply(y, x)` which denotes computing the Jacobian between `y` and `x`.

For example, given a regular `model`, we may wrap it into a `JacobianWrapper` for computing the Jacobian between the output and the input of the model:
```python
import torch.nn as nn
from auto_LiRPA.bound_ops import JacobianOP

class JacobianWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        return JacobianOP.apply(y, x)
```
Note that `JacobianOP.apply` only returns dummy values if we directly run this PyTorch model.
The actual Jacobian computation will be parsed when the model is wrapped into a `BoundedModule`.
See more [examples](../../examples/jacobian_new.py) including computing local Lipschitz constants and Jacobian-Vector products using `JacobianOP`.

## Adding new operators

To support the Jacobian bounds for a new operator, we need to ensure that there are bound operators implemented for the forward computation (the computation of the operator itself) and the backward computation (the computation of gradient) respectively.
Builtin operators are implemented in [auto_LiRPA/operators](../../auto_LiRPA/operators).
For example, for ReLU, we have [`BoundRelu`](../../auto_LiRPA/operators/relu.py) for the forward computation and [`BoundReluGrad`](../../auto_LiRPA/operators/gradient_bounds.py) for the backward computation.
Follow the [document](custom_op.md) to add new custom operators if necessary.

Then for the forward operator, implement a `build_gradient_node` function.
This function tells the library how a gradient module should be created given the forward operator when building the computational graph with the Jacobian computation.
The function takes a single argument `grad_upstream` which is the upstream gradient during the gradient back-propagation.
The function should return three variables in a tuple, including `module_grad`, `grad_input` and `grad_extra_nodes`.
`module_grad` is a `torch.nn.Module` and the created module for the gradient computation.
`grad_input` contains a list of tensors denoting the gradients propagated to the input nodes.
`grad_extra_nodes` may contain a list of extra nodes if needed for gradient computation.
Note that for `grad_upstream` and `grad_input`, we only care about the shapes of the gradient tensors, and their values do not matter and can be dummy values.
See examples in [relu.py](../../auto_LiRPA/operators/relu.py) or [linear.py](../../auto_LiRPA/operators/linear.py).

## References

Please cite our paper for the Jacobian computation:
```
@article{shi2022efficiently,
  title={Efficiently computing local lipschitz constants of neural networks via bound propagation},
  author={Shi, Zhouxing and Wang, Yihan and Zhang, Huan and Kolter, J Zico and Hsieh, Cho-Jui},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={2350--2364},
  year={2022}
}
```

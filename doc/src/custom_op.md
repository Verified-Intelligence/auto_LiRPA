# Custom Operators

This page explains how users can register an application-specific operator
without modifying auto_LiRPA. Contributors adding or maintaining built-in
operators should also read the
[Operator Development and Maintenance](operator_development.md) guide.

## Write a Custom Operator

There are four steps to write an operator:

1. Define a `torch.autograd.Function` (or `Function` for short) class, wrap the computation of the operator into this `Function`, and also define a symbolic method so that the operator can be parsed in auto_LiRPA via ONNX. Please refer to [PyTorch documentation](https://pytorch.org/docs/stable/onnx.html?highlight=symbolic#static-symbolic-method) on defining a `Function` with a symbolic method. 

2. Create a `torch.nn.Module` which uses the defined operator. Call the operator via
`.apply()` of `Function`.

3. Implement a [Bound class](api.html#auto_LiRPA.bound_ops.Bound) to support bound propagation methods for this operator. 

4. Create a mapping from the operator name (defined in step 1) to the bound class (defined in step 3). Define a `dict` which each item is a mapping. Pass the `dict` to the `custom_ops` argument when calling `BoundedModule` (see the [documentation](api.html#auto_LiRPA.BoundedModule)). For example, if the operator name is `MyRelu`, and the bound class is `BoundMyRelu`, then add `"MyRelu": BoundMyRelu` to the `dict`.

## Example

We provide an [code example](../../examples/vision/custom_op.py) of using a custom operator called "PlusConstant". 

## Contributing to the Library

We encourage the community to contribute generally useful operators to the
auto_LiRPA library. Follow the
[Operator Development and Maintenance](operator_development.md) guide for
built-in ownership, registration, testing, and review requirements.

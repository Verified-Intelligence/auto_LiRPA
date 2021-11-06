# Custom Operators

In this documentation, we introduce how users can define custom operators (such as other activations) that are not currently supported in auto_LiRPA, with bound propagation methods. 

## Write an Operator

There are three steps to write an operator:

1. Define a `torch.autograd.Function` (or `Function` for short) class, wrap the computation of the operator into this `Function`, and also define a symbolic method so that the operator can be parsed in auto_LiRPA via ONNX. Please refer to [PyTorch documentation](https://pytorch.org/docs/stable/onnx.html?highlight=symbolic#static-symbolic-method) on defining a `Function` with a symbolic method. Call this `Function` via `.apply()` when using this operator in the model.

2. Implement a [Bound class](api.html#auto_LiRPA.bound_ops.Bound) to support bound propagation methods for this operator. 
3. Create a mapping from the operator name (defined in step 1) to the bound class (defined in step 2). Define a `dict` which each item is a mapping. Pass the `dict` to the `custom_ops` argument when calling `BoundedModule` (see the [documentation](api.html#auto_LiRPA.BoundedModule)). For example, if the operator name is `MyRelu`, and the bound class is `BoundMyRelu`, then add `"MyRelu": BoundMyRelu` to the `dict`.

## Contributing to the Library

We encourage the community to upload their new operators to the auto_LiRPA library so that the new operators can also be used by other users. To do this, please put the `Function` and the Bound class of the new operator at the `auto_LiRPA/operators`, add the mapping at `bound_op_map.py`, and submit a pull request.

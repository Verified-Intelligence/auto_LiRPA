from .bound_ops import Bound, BoundLinear, BoundPrimConstant
from .bound_ops import BoundReluGrad, BoundConv2dGrad, BoundSqr

bound_op_map = {
    'onnx::Gemm': BoundLinear,
    'prim::Constant': BoundPrimConstant,
    'grad::Relu': BoundReluGrad,
    'grad::Conv2d': BoundConv2dGrad,
    'grad::Sqr': BoundSqr,
}

def register_custom_op(op_name: str, bound_obj: Bound) -> None:
    bound_op_map[op_name] = bound_obj

def unregister_custom_op(op_name: str) -> None:
    bound_op_map.pop(op_name)

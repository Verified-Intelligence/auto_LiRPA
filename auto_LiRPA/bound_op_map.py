from .bound_ops import *

bound_op_map = {
    'onnx::Gemm': BoundLinear,
    'prim::Constant': BoundPrimConstant,
}

def register_custom_op(op_name: str, bound_obj: Bound) -> None:
    """ Register a custom operator.
   
    Args:
        op_name (str): Name of the custom operator
        
        bound_obj (Bound): The corresponding Bound class for the operator.
    """
    bound_op_map[op_name] = bound_obj

def unregister_custom_op(op_name: str) -> None:
    """ Unregister a custom operator.
   
    Args:
        op_name (str): Name of the custom operator
    """    
    bound_op_map.pop(op_name)
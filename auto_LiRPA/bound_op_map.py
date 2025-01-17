#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
from .bound_ops import *

bound_op_map = {
    'onnx::Gemm': BoundLinear,
    'prim::Constant': BoundPrimConstant,
    'grad::Concat': BoundConcatGrad,
    'grad::Relu': BoundReluGrad,
    'grad::Conv2d': BoundConv2dGrad,
    'grad::Slice': BoundSliceGrad,
    'grad::Sqr': BoundSqr,
    'grad::jacobian': BoundJacobianOP,
    'custom::Gelu': BoundGelu,
    'onnx::Clip': BoundHardTanh
}

def register_custom_op(op_name: str, bound_obj: Bound) -> None:
    bound_op_map[op_name] = bound_obj

def unregister_custom_op(op_name: str) -> None:
    bound_op_map.pop(op_name)

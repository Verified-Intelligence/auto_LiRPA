#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Optimize the graph to merge nodes and remove unnecessary ones.

Initial and experimental code only.
"""

from auto_LiRPA.bound_ops import *
from auto_LiRPA.utils import logger
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def _optimize_graph(self: 'BoundedModule'):
    """Optimize the graph to remove some unnecessary nodes."""
    merge_identical_act(self)
    convert_sqr(self)
    div_to_mul(self)
    merge_sec(self)
    minmax_to_relu(self)
    optimize_relu_relation(self)

    if self.bound_opts['optimize_graph']['optimizer'] is not None:
        # Use the custom graph optimizer
        self.bound_opts['optimize_graph']['optimizer'](self)

    for node in list(self.nodes()):
        if (not node.output_name
                and node.name != self.final_name
                and node.name not in self.root_names):
            self.delete_node(node)


def _copy_node_properties(new, ref):
    new.output_shape = ref.output_shape
    new.device = ref.device
    new.attr['device'] = ref.attr['device']
    new.batch_dim = ref.batch_dim
    new.from_complex_node = ref.from_complex_node


def merge_sec(model: 'BoundedModule'):
    nodes = list(model.nodes())
    for node in nodes:
        if type(node) == BoundReciprocal and type(node.inputs[0]) == BoundCos:
            node_new = BoundSec(inputs=[node.inputs[0].inputs[0]])
            node_new.name = f'{node.inputs[0].name}/sec'
            _copy_node_properties(node_new, node)
            if node_new.name in model._modules:
                node_existing = model._modules[node_new.name]
                assert isinstance(node_existing, BoundSec)
                assert node_existing.inputs[0] == node.inputs[0].inputs[0]
                model.replace_node(node, node_existing)
            else:
                model.add_nodes([node_new])
                model.replace_node(node, node_new)


def div_to_mul(model: 'BoundedModule'):
    nodes = list(model.nodes())
    for node in nodes:
        if type(node) == BoundDiv:
            logger.debug('Replacing BoundDiv node: %s', node)
            node_reciprocal = BoundReciprocal(inputs=[node.inputs[1]])
            node_reciprocal.name = f'{node.name}/reciprocal'
            # Properties of the reciprocal node only depend on inputs[1], i.e.
            # the node of denominator. They can be different from those of
            # the original BoundDiv node, due to possible broadcasting and
            # perturbed/unperturbed switching in multiplication.
            _copy_node_properties(node_reciprocal, node.inputs[1])
            model.add_nodes([node_reciprocal])
            node_mul = BoundMul(inputs=[node.inputs[0], node_reciprocal],
                                options=model.bound_opts)
            node_mul.name = f'{node.name}/mul'
            _copy_node_properties(node_mul, node)
            model.add_nodes([node_mul])
            model.replace_node(node, node_mul)


def convert_sqr(model: 'BoundedModule'):
    """Replace BoundMul or Bound Pow with BoundSqr if applicable.

    1. If the two inputs nodes of a BoundMul node are the same, use BoundSqr.
    2. Pow(x, 2) can be replaced with BoundSqr.
    """
    nodes = list(model.nodes())
    for node in nodes:
        replace = False
        if type(node) == BoundMul and node.inputs[0] == node.inputs[1]:
            replace = True
        elif type(node) == BoundPow:
            if ((isinstance(node.inputs[1], BoundBuffers) and node.inputs[1].buffer == 2) or
                (isinstance(node.inputs[1], BoundConstant) and node.inputs[1].value == 2)):
                replace = True
        if replace:
            node_new = BoundSqr(inputs=[node.inputs[0]])
            node_new.name = f'{node.name}/sqr'
            _copy_node_properties(node_new, node)
            model.add_nodes([node_new])
            logger.debug('Replaceing %s with %s', node, node_new)
            model.replace_node(node, node_new)


def merge_identical_act(model: 'BoundedModule'):
    """Merge identical BoundActivation"""
    nodes = list(model.nodes())
    merged = [False] * len(nodes)
    for i in range(len(nodes)):
        if (not merged[i]
                and isinstance(nodes[i], BoundActivation)
                and len(nodes[i].inputs) == 1):
            for j in range(i + 1, len(nodes)):
                if (not merged[j]
                        and type(nodes[j]) == type(nodes[i])
                        and len(nodes[i].inputs) == 1):
                    if nodes[i].inputs[0] == nodes[j].inputs[0]:
                        logger.debug('Merging node %s to %s', nodes[j], nodes[i])
                        model.replace_node(nodes[j], nodes[i])
                        merged[j] = True


def minmax_to_relu(model: 'BoundedModule'):
    """Replace BoundMinMax with BoundRelu if one of its inputs is constant"""
    nodes = list(model.nodes())
    for node in nodes:
        if type(node) == BoundMax:
            for i, input_node in enumerate(node.inputs):
                if not input_node.perturbed:
                    logger.debug('Replacing BoundMax node %s', node)
                    # max(x, c) = ReLU(x - c) + c
                    node_sub = BoundSub(inputs=[node.inputs[1-i], input_node],
                                        options=model.bound_opts)
                    node_sub.name = f'{node.name}/sub'
                    _copy_node_properties(node_sub, node)
                    node_relu = BoundRelu(inputs=[node_sub],
                                          options=model.bound_opts)
                    node_relu.name = f'{node.name}/relu'
                    _copy_node_properties(node_relu, node)
                    node_add = BoundAdd(inputs=[node_relu, input_node],
                                        options=model.bound_opts)
                    node_add.name = f'{node.name}/add'
                    _copy_node_properties(node_add, node)
                    model.add_nodes([node_sub, node_relu, node_add])
                    model.replace_node(node, node_add)
                    break
        elif type(node) == BoundMin:
            for i, input_node in enumerate(node.inputs):
                if not input_node.perturbed:
                    logger.debug('Replacing BoundMin node %s', node)
                    # min(x, c) = -ReLU(c - x) + c
                    node_sub_1 = BoundSub(inputs=[input_node, node.inputs[1-i]],
                                          options=model.bound_opts)
                    node_sub_1.name = f'{node.name}/sub/1'
                    _copy_node_properties(node_sub_1, node)
                    node_relu = BoundRelu(inputs=[node_sub_1],
                                          options=model.bound_opts)
                    node_relu.name = f'{node.name}/relu'
                    _copy_node_properties(node_relu, node)
                    node_sub_2 = BoundSub(inputs=[input_node, node_relu],
                                          options=model.bound_opts)
                    node_sub_2.name = f'{node.name}/sub/2'
                    _copy_node_properties(node_sub_2, node)
                    model.add_nodes([node_sub_1, node_relu, node_sub_2])
                    model.replace_node(node, node_sub_2)
                    break

def _check_merge(W_merge, skip, pairs, bias=None):
    """
    Check that the merge layer C has the exact structure required by the current fusion logic.

    W_merge: Tensor of shape (n_out, n_mid)
             or (n_out, n_mid, 1, 1)        
    skip: set[int] of skipped src indices
    pairs: dict[int,int] mapping src=j -> merge row r (from _pair_row), for adj pair (j,j+1)
    bias: optional Tensor of shape (n_out,) for merge bias. If provided, we enforce bias ~ 0
          for one-hot rows (since ReLU(h + b) == h only if b == 0 in general).

    Returns: (ok: bool)
    """
    if W_merge.dim() == 4:
        W = W_merge[..., 0, 0].detach()
    elif W_merge.dim() == 2:
        W = W_merge.detach()
    else:
        return False

    n_out, n_mid = W.shape
    dst2src = [s for s in range(n_mid) if s not in skip]
    if len(dst2src) != n_out:
        return False

    b = bias.detach() if bias is not None else None

    for r, src in enumerate(dst2src):
        nz = torch.nonzero(W[r].abs() > 1e-8, as_tuple=False).flatten()
        nnz = len(nz)

        # In this case, the rows should contain a +1 at the correct location.
        # And the same position in b should be 0.
        if nnz == 1:
            c = int(nz[0].item())
            if (c != src) or ((W[r, c] - 1.0).abs().item() > 1e-8) or \
               (b is not None and b[r].abs().item() > 1e-8):
                return False

        # In this case, the row should contain a pair of +1, -1 at the correct location.
        elif nnz == 2:
            c0 = int(nz[0].item())
            c1 = int(nz[1].item())
            if not (c0 == src and c1 == src + 1) or ((W[r, c0] - 1.0).abs().item() > 1e-8) \
                or ((W[r, c1] + 1.0).abs().item() > 1e-8):
                return False
            # Must correspond to a detected pair, and the mapping must hit this row.
            if pairs.get(src, None) != r:
                return False

    return True

def _pair_row(Ws, bs, Wm, j, atol=1e-8):
    """
    Checks the relation ReLU(x) - ReLU(-x) = x. Return
    the index at the merge weight if the relation exists,
    otherwise return None.
    """
    # Check whether this fits the pattern in docstring.
    if not (torch.allclose(Ws[j+1], -Ws[j], atol=atol)
            and abs(float(bs[j] + bs[j+1])) < atol):
        return None

    # Make merge weight 4D so Gemm and Conv share same indexing
    if Wm.dim() == 2:                 # Gemm path
        Wm4 = Wm.unsqueeze(-1).unsqueeze(-1)    
    else:                             # Conv path 
        Wm4 = Wm

    # Find corresponding columns of the merge weight
    # We check 1) The two nonzero element are in the same row
    #          2) The two entries are +1 and -1
    # If the check pass, we return the row index, otherwise it 
    # is not a valid pattern match and we return None.
    rows = torch.nonzero(Wm4[:, [j, j+1], 0, 0], as_tuple=False)
    if rows.size(0) != 2 or rows[0, 0] != rows[1, 0]:
        return None
    r = int(rows[0, 0])

    ok = (abs(float(Wm4[r, j, 0, 0] - 1)) < atol and
          abs(float(Wm4[r, j+1, 0, 0] + 1)) < atol and
          torch.count_nonzero(Wm4[r]) == 2)
    return r if ok else None
                
def optimize_relu_relation(model: 'BoundedModule'):
    """
    This graph optimization detects the optimizable path with
    the identity
        ReLU(ReLU(x + b) - ReLu(-x - b)) = ReLU(x + b)
    for both linear layer and convolution layer. Replace the 
    sequence of nodes with pattern
        Gemm -> ReLU -> Gemm -> ReLU or
        Conv -> ReLU -> Conv -> ReLU
    to one single Gemm -> ReLU or Conv -> ReLU.
    """
    nodes = list(model.nodes())
    i = 0
    while i + 3 < len(nodes):
        A, B, C, D = nodes[i:i+4]
        
        # In Conv layers, we detect whether the optimization can be done
        # for pairs of channels. If so, the optimization eliminates one
        # Conv layer and recover the original results with the identity 
        # in docstring.
        if (isinstance(A, BoundConv) and isinstance(B, BoundRelu) and
            isinstance(C, BoundConv) and isinstance(D, BoundRelu) and tuple(C.attr['kernel_shape'])==(1,1)):
            
            # Here use forward() to extract weights to handle BoundParam/BoundConstant, or any other node
            # that could represent weights a unified interface.
            Ws = C.inputs[1].forward()
            Wc = A.inputs[1].forward()
            
            # We only care about 2D conv
            if Ws.ndim != 4 or Wc.ndim != 4:
                i += 1
                continue
            
            bs = C.inputs[2].forward() if C.has_bias else torch.zeros_like(Ws[:, 0, 0, 0])
            bc = A.inputs[2].forward() if A.has_bias else torch.zeros_like(Wc[:, 0, 0, 0])
            
            # Detect whether and where the identity presents in the weight matrix.
            pairs, skip = {}, set()
            for j in range(0, Wc.size(0) - 1):
                r = _pair_row(Wc, bc, Ws, j)
                if r is not None:
                    pairs[j] = r
                    skip.add(j + 1)

            ok = _check_merge(Ws, skip, pairs, bias=bs)
            if not ok:
                i += 1
                continue
            
            if pairs:
                Cout, Cin, kH, kW = Ws.size(0), Wc.size(1), *Wc.shape[2:]
                W_new = torch.empty((Cout, Cin, kH, kW), dtype=Wc.dtype, device=Wc.device)
                b_new = torch.empty((Cout,), dtype=bc.dtype, device=bc.device)

                
                # Build fused weight and bias
                dst = 0
                for src in range(Wc.size(0)):
                    if src in skip:
                        continue
                    b_new[dst] = bs[pairs[src]] + bc[src] if src in pairs else bc[src]
                    W_new[dst] = Wc[src]
                    dst += 1
                
                # Modify the graph using the newly built weights and bias
                weight_node = BoundParams('fused_weight', torch.nn.Parameter(W_new))
                bias_node = BoundParams('fused_bias', torch.nn.Parameter(b_new))
                weight_node.name = f'{A.name}/optimized/weight' 
                bias_node.name = f'{A.name}/optimized/bias'
                
                fused = BoundConv(
                    attr=A.attr.copy(),
                    inputs=[A.inputs[0], weight_node, bias_node],
                    output_index=A.output_index,
                    options=model.bound_opts
                )
                fused.name = f'{A.name}/optimized'
                _copy_node_properties(fused, A)
                relu = BoundRelu(inputs=[fused], options=model.bound_opts)
                relu.name = f'{A.name}/optimized/relu'
                _copy_node_properties(relu, D)
                
                model.add_nodes([weight_node, bias_node, fused, relu])
                model.replace_node(D, relu)
                model.replace_node(A, fused)
                model.delete_node(B)
                model.delete_node(C) 
                
                # Skip the full sequence once the pattern is detected
                i += 4
                continue
        
        # In Linear layer, we detect whether the optimization can be 
        # done for pair of rows. The code structure is similar the 
        # one at Conv branch. 
        elif (isinstance(A, BoundLinear) and isinstance(B, BoundRelu) and
            isinstance(C, BoundLinear) and isinstance(D, BoundRelu)):
            
            Ws = A.inputs[1].forward()
            Wm = C.inputs[1].forward()
            bs = A.inputs[2].forward() if len(A.inputs) == 3 else torch.zeros_like(Ws[:, 0])
            bm = C.inputs[2].forward() if len(C.inputs) == 3 else torch.zeros_like(Wm[:, 0])
            
            pairs, skip = {}, set()
            for j in range(0, Ws.size(0) - 1):
                r = _pair_row(Ws, bs, Wm, j)
                if r is not None:
                    pairs[j] = r
                    skip.add(j + 1)

            ok = _check_merge(Wm, skip, pairs, bias=bm)
            if not ok:
                i += 1
                continue
                 
            if pairs:
                n_out = Wm.shape[0]
                W_new = torch.empty((n_out, Ws.shape[1]), dtype=Ws.dtype, device=A.attr['device'])
                b_new = torch.empty((n_out,), dtype=bs.dtype, device=A.attr['device'])

                dst = 0
                for src in range(Ws.size(0)):
                    if src in skip:
                        continue
                    b_new[dst] = bm[pairs[src]] + bs[src] if src in pairs else bs[src]
                    W_new[dst] = Ws[src]
                    dst += 1
                
                weight_node = BoundParams('fused_weight', torch.nn.Parameter(W_new), attr=dict(device=A.attr['device']))
                bias_node = BoundParams('fused_bias', torch.nn.Parameter(b_new), attr=dict(device=A.attr['device']))
                weight_node.name = f'{A.name}/optimized/weight'
                bias_node.name = f'{A.name}/optimized/bias'
                
                fused = BoundLinear(
                    attr=A.attr.copy(),
                    inputs=[A.inputs[0], weight_node, bias_node],
                    output_index=A.output_index,
                    options=model.bound_opts
                )
                fused.name = f'{A.name}/optimized'
                _copy_node_properties(fused, A)
                relu = BoundRelu(inputs=[fused], options=model.bound_opts)
                relu.name = f'{A.name}/optimized/relu'
                _copy_node_properties(relu, D)
                
                model.add_nodes([weight_node, bias_node, fused, relu])
                model.replace_node(D, relu)
                model.delete_node(A)
                model.delete_node(B)
                model.delete_node(C)
                
                i += 4
                continue
        i += 1

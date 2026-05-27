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
import torch
from graphviz import Digraph
import shutil
import re

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def visualize(self: 'BoundedModule', output_path, print_bounds=False):
    r"""A visualization tool for BoundedModule.
    If dot engine is available in the system enviornment, it renders the graph and output {output_path}.png.
    Otherwise, it output a {output_path}.dot.

    Args:
        output_path (str): The path to save the graph (without file extension).
        print_bounds (bool): Whether to display the mean width of the bounds for each node.
    """

    nodes = list(self.nodes())
    # Create a directed graph
    dot = Digraph(format='png', engine='dot')
    # Add nodes with optional attributes
    for node in nodes:
        # we name the Graphviz nodes with the sanitized node name,
        # while keeping the original name in the label which is displayed in the graph.
        export_node_name = sanitize_graphviz_name(node.name)
        label = f"""<
            <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="4">
                <TR><TD><FONT FACE="Arial" COLOR="black">{node.name}</FONT></TD></TR>
                <TR><TD><FONT FACE="Courier" COLOR="blue">{node.__class__.__name__}</FONT></TD></TR>
                <TR><TD><FONT FACE="Courier" COLOR="black">{
                    tuple(node.output_shape) if node.output_shape is not None else None}</FONT></TD></TR>
            </TABLE>
        >"""
        if print_bounds:
            # Display the mean width of the bounds)
            # (Both the empirical bound from forward value and the computed bound if available)
            label = f"""<
                <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="4">
                    <TR><TD><FONT FACE="Arial" COLOR="black">{node.name}</FONT></TD></TR>
                    <TR><TD><FONT FACE="Courier" COLOR="blue">{node.__class__.__name__}</FONT></TD></TR>
                    <TR><TD><FONT FACE="Courier" COLOR="black">{
                        tuple(node.output_shape) if node.output_shape is not None else None}</FONT></TD></TR>
                    <TR><TD><FONT FACE="Courier" COLOR="black">{
                        (node.forward_value.max(dim=0)[0] - node.forward_value.min(dim=0)[0]).to(dtype=torch.float).mean().item() if (
                            node.perturbed and
                            hasattr(node, "forward_value") and
                            isinstance(node.forward_value, torch.Tensor)) else None}</FONT></TD></TR>
                    <TR><TD><FONT FACE="Courier" COLOR="black">{
                        (node.upper - node.lower).to(dtype=torch.float).mean().item() if (
                            node.perturbed and
                            hasattr(node, "lower") and hasattr(node, "upper") and
                            node.lower is not None and node.upper is not None) else None}</FONT></TD></TR>
                </TABLE>
            >"""
        # perturbed nodes are highlighted in grey
        if getattr(node, "perturbed", False):
            style_attrs = {'style': 'filled', 'fillcolor': 'lightgrey'}
        else:
            style_attrs = {}
        if node.__class__.__name__ in ["BoundParams", "boundConstant", "BoundBuffers"]:
            dot.node(export_node_name, label=label, fontsize="8", width="0.5", height="0.2", shape="ellipse", **style_attrs)
        elif node.__class__.__name__ == "BoundInput":
            dot.node(export_node_name, label=label, shape="diamond", **style_attrs)
        else:
            dot.node(export_node_name, label=label, shape="square", **style_attrs)
        for inp in node.inputs:
            dot.edge(sanitize_graphviz_name(inp.name), export_node_name)
    # Render graph
    if shutil.which("dot") is None:
        print("Cannot render the graphviz file (dot not found).")
        print(f"Graph saved to {output_path}.dot")
        dot.save(output_path + ".dot")
    else:
        dot.render(output_path, cleanup=True)
        print(f"Graph saved to {output_path}.png")

def sanitize_graphviz_name(name):
    """
    Convert problematic characters (like `:`, `::`) in a Graphviz node name to a safe alternative character `_`.
    """
    unsafe_chars = r'[:;,\[\]{}()<>|#*@&=+`~^?"\\\s]'
    safe_name = re.sub(unsafe_chars, "_", name)
    
    return safe_name

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
##        contained in the LICENSE file in the repository root.       ##
##                                                                     ##
#########################################################################
"""Dynamic forward bounds.

Dynamic forward encloses each node's values in an affine relaxation over one
shared set of variables, each ranging over [-1, 1]:

    relaxed_node = c + G e,    e in [-1, 1]^n.

`c` is the relaxation's centre, its value when all input and relaxation
variables are zero. `G` has one column per variable (its generators).
Since every variable ranges over [-1, 1], the node's bounds are
`c - |G|_1` and `c + |G|_1`, with `|G|_1` the
sum of absolute values along each row. In a `LinearBound`, `lb` (= `ub`) holds
`c` and `lw` (= `uw`) holds `G`, shaped `[batch, n, *node shape]`.

The variables come in this order:

- input variables: dense initialization allocates one per input coordinate,
  with zero generators for fixed coordinates; sparse initialization allocates
  the batch's maximum number of moving coordinates (at least one column)
  and pads shorter rows with zero generators:
  `x = (x_L + x_U) / 2 + (x_U - x_L) / 2 * e`;
- then, for each ReLU, one per unstable neuron (`lower < 0 < upper`),
  padded to the batch's maximum unstable count:
  `relu(z) = s * z + h * (1 + e)`, with the slope `s` and
  half-width `h` of `BoundRelu.dynamic_forward_relaxation`.

A ReLU takes the next free columns the first time a pass reaches it, and
`dynamic_variable_ranges` records which columns each node took.

A pass computes the nodes for one slice of the variables, the columns
`[offset, offset + forward_max_dim)`, which bounds the memory a pass needs.
`compute_bounds(method='dynamic-forward')` runs one pass per slice and adds up
each slice's part of the bounds; with `return_generators=True` it also returns the
centre and generators of the nodes asked for.

`method='dynamic-forward-optimized'` first chooses the slopes of the unstable
ReLUs with the existing same-slope alpha-CROWN optimizer. Reverse mode reads
its generators with the existing CROWN backward propagation, recording the
coefficient of each ReLU error before propagating through that ReLU.
"""
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import copy
from functools import partial

import numpy as np
import torch
from .bounded_tensor import BoundedTensor
from .linear_bound import LinearBound
from .operators.linear import BoundLinear
from .operators.relu import BoundRelu
from .perturbations import PerturbationLpNorm
from .utils import logger, prod, reduction_min

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def _inputs_first(targets):
    """The nodes a pass computes for `targets`, each placed after its inputs.

    The order is depth first, inputs in order. ReLUs take their columns in the
    order a pass first reaches them, so every pass has to visit the nodes in
    this same order for a column to mean the same variable in all of them.
    The inputs of a node that `_is_given` are not visited.
    """
    order, seen = [], set()

    def visit(node):
        if node.name in seen:
            return
        seen.add(node.name)
        if not _is_given(node):
            for inp in node.inputs:
                visit(inp)
        order.append(node)

    for target in targets:
        visit(target)
    return order


def _is_given(node):
    """Whether a pass takes `node` as it is instead of computing it: a constant
    (it does not depend on the input), or a node that this compute_bounds call
    has already bounded and cached in `node.linear`."""
    return not node.perturbed or hasattr(node, 'linear')


def _dynamic_forward_pass(self: 'BoundedModule', targets, ranges, offset,
                          fused_node=None, C=None, *, max_dim=None):
    """One pass: the centre and generators of `targets`, as `LinearBound`s
    over the variables of one slice, the columns
    `[offset, offset + forward_max_dim)`.

    Each node is computed once, after its inputs. Its `LinearBound` is freed as
    soon as every node that reads it has been computed, so memory holds only
    the nodes still waiting for a reader. `ranges` is the numbering of the
    variables, filled in as ReLUs are reached.

    With `max_dim=0`, it computes only centres and variable numbering, for
    reading the generators in backward mode.
    """
    if max_dim is None:
        max_dim = self.bound_opts['forward_max_dim']
    order = _inputs_first(targets)
    # How many reads each node still has coming: one per node that takes it as
    # an input, and one for the caller per target.
    readers = Counter(t.name for t in targets)
    for node in order:
        if not _is_given(node):
            readers.update(inp.name for inp in node.inputs)
    bounds = {}
    for node in order:
        if hasattr(node, 'linear'):
            bounds[node.name] = _variable_slice(node.linear, offset, max_dim,
                                                ranges)
            continue
        if not node.perturbed:
            # A constant: no generators, the same centre in every slice.
            if not node.is_lower_bound_current():
                node.lower = node.upper = self.get_forward_value(node)
            bounds[node.name] = LinearBound(None, node.lower, None, node.lower,
                                            node.lower, node.upper)
            continue
        inputs = [bounds[inp.name] for inp in node.inputs]
        for inp, linear in zip(node.inputs, inputs):
            # The bounds of an input decide how a ReLU reading it relaxes.
            linear.lower, linear.upper = inp.lower, inp.upper
        node._start = '_forward'
        kwargs = {'C': C} if node is fused_node else {}
        linear = node.bound_dynamic_forward(*inputs, max_dim=max_dim,
                                            offset=offset, **kwargs)
        linear.dynamic_variable_ranges = ranges
        if offset > 0:
            # Only the first slice carries the centre.
            linear.lb = linear.ub = torch.zeros_like(linear.lb)
        bounds[node.name] = linear
        for inp in node.inputs:
            readers[inp.name] -= 1
            if readers[inp.name] == 0:
                del bounds[inp.name]
    return [bounds[t.name] for t in targets]


def _concretize(linear, C=None):
    """The lower and upper bounds of `linear`, or of `C` times it.

    Every variable ranges over [-1, 1], so each element's bounds are its centre
    minus and plus the sum of the absolute values of its generators.
    """
    generators, centre = linear.lw, linear.lb
    if C is not None:
        # C reads the node's elements flattened, whatever the node's shape.
        C = C.to(generators).transpose(-1, -2)
        generators = generators.flatten(2).matmul(C)
        centre = centre.flatten(1).unsqueeze(1).matmul(C).squeeze(1)
    radius = generators.flatten(2).abs().sum(1).view(centre.shape)
    return centre - radius, centre + radius


def forward_general_dynamic(self: 'BoundedModule', C=None, node:'Bound'=None,
                            generators=None):
    """The bounds of `node`, or of `C` times it, by dynamic forward: one pass
    per slice of `forward_max_dim` variables, each slice's part of the bounds
    added up.

    `generators` (`{name: rows}`, `compute_bounds`' `needed_generators_dict`) also
    asks for the centre and generators of those nodes, collected from the same
    passes; the bounds then come with `{name: (G, c)}`, including an empty
    dictionary when no nodes are requested. `None` returns only bounds.
    """
    root = next(r for r in self.roots()
                if getattr(r, 'perturbation', None) is not None)
    ranges = root.linear.dynamic_variable_ranges
    max_dim = self.bound_opts['forward_max_dim']
    logger.debug(f'forward_general_dynamic: node={node}')
    return_generators = generators is not None
    generators = generators or {}
    if generators and _generators_in_reverse(self, root, node, generators):
        return _reverse_mode_generators(self, node, C, generators)
    # A fused target represents C times the node, so it cannot provide the
    # node's own generators. Ancestor requests still use their original form.
    fuse_C = (C is not None and type(node) is BoundLinear
              and node.name not in generators and not hasattr(node, 'linear')
              and node.perturbed and not node.is_input_perturbed(1)
              and not node.is_input_perturbed(2))
    nodes = [node] + [self[name] for name in generators if name != node.name]
    pieces, centres = {name: [] for name in generators}, {}

    lower = upper = 0
    # each node's number of variables, to know when its last slice is done
    widths, offset = {}, 0
    while need := [n for n in nodes
                   if offset == 0 or offset < widths[n.name]]:
        for n, linear in zip(need, _dynamic_forward_pass(
                self, need, ranges, offset,
                fused_node=node if fuse_C else None, C=C)):
            widths[n.name] = linear.tot_dim
            if n.name in pieces:
                if linear.lw is None:
                    raise NotImplementedError(
                        f'{n.name} is not a function of the perturbed input')
                # [batch, variables, *shape] -> [batch, elements, variables].
                # The centre comes from the first slice, which holds it.
                G, c = linear.lw.flatten(2).transpose(1, 2), linear.lb.flatten(1)
                rows = generators[n.name]
                pieces[n.name].append(G if rows is None else G[:, rows])
                centres.setdefault(n.name, c if rows is None else c[:, rows])
            if n is node:
                if linear.lw is None:
                    # A constant: its bounds are its value. (Asking for the
                    # generators of a constant raises above.)
                    bounds = (linear.lower, linear.upper)
                    return (*bounds, {}) if return_generators else bounds
                lo, hi = _concretize(linear, None if fuse_C else C)
                lower, upper = lower + lo, upper + hi
                if C is None and linear.tot_dim <= max_dim:
                    # One slice held all the variables: cache the result, and
                    # later nodes in this call start from it.
                    node.linear = linear
            # Free this slice before the next pass computes its own.
            del linear
        offset += max_dim

    # Adding up the slices' parts can give NaN even when every part has
    # defined endpoints.
    node.check_bound_invariants(
        lower, upper, context='dynamic forward concretization')
    if C is None:
        node.lower, node.upper = lower, upper
    if not return_generators:
        return lower, upper
    return lower, upper, {name: (torch.cat(G, dim=-1), centres[name])
                          for name, G in pieces.items()}


def _variable_slice(linear, offset, max_dim, ranges):
    """`linear` restricted to the variables `[offset, offset + max_dim)`. Only
    the first slice carries the centre; later slices carry 0."""
    if linear.lw is not None and (offset > 0 or linear.lw.shape[1] > max_dim):
        lw = linear.lw[:, offset:offset + max_dim]
        lb = linear.lb if offset == 0 else torch.zeros_like(linear.lb)
        linear = LinearBound(lw, lb, lw, lb, tot_dim=linear.tot_dim)
    linear.dynamic_variable_ranges = ranges
    return linear


def _require_box_perturbation(ptb):
    """Dynamic forward needs an explicit box: it gives each input coordinate a
    variable over that coordinate's interval. A finite-p ball has no such
    intervals, and bounding it as its centre would be unsound."""
    if (type(ptb) is not PerturbationLpNorm or ptb.norm < np.inf
            or ptb.x_L is None or ptb.x_U is None):
        raise NotImplementedError(
            'For dynamic forward bounds, only Linf (box) perturbations are '
            'supported, and x_L and x_U must be explicitly provided.')


def _substitute_input_box(linear, ptb):
    """Rewrite the input's `LinearBound` from `init` as
    `x = (x_L + x_U) / 2 + (x_U - x_L) / 2 * e`, with every variable on
    [-1, 1]: each variable's generator is scaled by the half-width of the
    coordinate it moves, and the centre is the box's centre."""
    x_L = ptb.x_L_sparse if ptb.sparse else ptb.x_L
    x_U = ptb.x_U_sparse if ptb.sparse else ptb.x_U
    lw = linear.lw
    rad = ((x_U - x_L) / 2).reshape(*lw.shape[:2], *[1] * (lw.dim() - 2))
    linear.lw = linear.uw = lw * rad.to(lw)
    linear.lb = linear.ub = ((ptb.x_L + ptb.x_U) / 2).to(linear.lb)


def _unstable(relu):
    """The unstable neurons of `relu`, those whose input interval contains 0:
    each of them has a variable."""
    pre = relu.inputs[0]
    return torch.logical_and(pre.lower < 0, pre.upper > 0)


def _relus_of(self: 'BoundedModule', nodes):
    """The ReLUs that `nodes` depend on."""
    ancestors = set().union(*(self._ancestors_of(n) for n in nodes))
    return [r for r in self.relus if r.perturbed and r in ancestors]


def _generators_in_reverse(self: 'BoundedModule', root, node, generators):
    """Whether to read `generators` in reverse mode.

    Forward mode carries every variable through the pass, so it costs about
    one pass per variable; reverse mode costs about one backward pass per row
    of `G` asked for (`node`'s rows are all needed, for its bounds). So
    reverse mode is chosen when there are fewer rows than variables, for a
    single input box. `bound_opts['dynamic_forward']['generators']` can force
    'forward' or 'reverse' instead of this 'auto' choice.
    """
    mode = self.bound_opts['dynamic_forward']['generators']
    if mode != 'auto':
        return mode == 'reverse'
    if self.batch_size != 1:
        return False
    others = [self[name] for name in generators if name != node.name]
    n_rows = prod(node.output_shape[1:]) + sum(
        prod(n.output_shape[1:]) if generators[n.name] is None
        else len(generators[n.name]) for n in others)
    n_variables = root.linear.lw.shape[1] + sum(
        int(_unstable(r).sum()) for r in _relus_of(self, [node, *others]))
    return n_rows < n_variables

@contextmanager
def _same_slope_alpha(self: 'BoundedModule', slopes=None):
    """Use an isolated alpha bank on this graph, leaving beta-CROWN's intact.

    Bounds and inputs follow the normal compute_bounds lifecycle. Only the
    optimization parameters, their lookup metadata and options are scoped;
    no model or parameter tensors are copied.

    The relaxation uses fixed intermediate bounds and excludes beta, cuts,
    output-constraint refinement and mixed-precision optimization. Search
    settings such as iterations, learning rates and stopping criteria remain
    the caller's ``optimize_bound_args``.
    """
    options = self.bound_opts
    scoped = {**options, 'enable_opt_interm_bounds': False,
              'compare_crown_with_ibp': False,
              'clip_in_alpha_crown': False,
              'activation_bound_option': 'same-slope',
              'optimize_bound_args': {
                  **options['optimize_bound_args'],
                  'use_shared_alpha': 'network', 'enable_alpha_crown': True,
                  'enable_beta_crown': False, 'fix_interm_bounds': True,
                  'init_alpha': True, 'keep_best': True,
                  'reproduce_alpha_bounds': True, 'pruning_in_iteration': False,
                  'apply_output_constraints_to': [], 'tighten_input_bounds': False,
                  'directly_optimize': [], 'mixed_precision_dtype': None,
                  'loss_reduction_func': reduction_min}}
    activations = self.get_enabled_opt_act()
    if any(type(n) is not BoundRelu for n in activations):
        raise NotImplementedError('Network-wide alpha sharing supports ReLU only.')
    saved_options = [(n, n.options) for n in self.nodes()]
    cut_used = self.cut_used
    constraints_optimized = self.constraints_optimized
    # Native alpha-CROWN uses backward(); do not accumulate into the model's
    # gradients, or overwrite gradients the caller already has.
    grads = [(p, p.grad) for p in self.parameters()]
    with ExitStack() as stack:
        for n in activations:
            stack.enter_context(n.temporary_shared_alpha(
                None if slopes is None else slopes.get(n.name)))
        try:
            self.bound_opts, self.cut_used = scoped, False
            for p, _ in grads:
                p.grad = None
            for n, _ in saved_options:
                n.options = scoped
            yield scoped
        finally:
            self.bound_opts, self.cut_used = options, cut_used
            self.constraints_optimized = constraints_optimized
            for n, previous in saved_options:
                n.options = previous
            for p, grad in grads:
                p.grad = grad


def _reverse_mode_generators(self: 'BoundedModule', node, C, generators,
                             chunk=256):
    """Read affine rows with native CROWN, in chunks, under one shared alpha.

    The zero-width dynamic pass supplies centres and the existing variable
    numbering. CROWN supplies input coefficients and each ReLU's incoming A;
    its error generator is that incoming A times the relaxation half-width.
    """
    if self.batch_size != 1:
        raise NotImplementedError(
            'reverse mode reads the generators of one box at a time')
    root = next(r for r in self.roots()
                if getattr(r, 'perturbation', None) is not None)
    ranges = root.linear.dynamic_variable_ranges
    nodes = [node] + [self[name] for name in generators if name != node.name]
    linears = _dynamic_forward_pass(
        self, nodes, ranges, 0, max_dim=0)
    centres = {n.name: b.lb.flatten(1) for n, b in zip(nodes, linears)}
    widths = {n.name: b.tot_dim for n, b in zip(nodes, linears)}
    slopes = {r.name: r.dynamic_forward_relaxation(
        r.inputs[0].lower, r.inputs[0].upper,
        r.dynamic_forward_slope())[0]
        for r in _relus_of(self, nodes)}
    n_input = ranges[root.name][1]
    radius = ((root.perturbation.x_U - root.perturbation.x_L) / 2).flatten()
    coords = (torch.arange(radius.numel(), device=radius.device)
              if n_input == radius.numel()
              else radius.nonzero(as_tuple=True)[0])
    error_rows = {}
    for r in _relus_of(self, nodes):
        index = _unstable(r).flatten().nonzero(as_tuple=True)[0]
        half = r.dynamic_forward_relaxation(
            r.inputs[0].lower, r.inputs[0].upper, slopes[r.name])[1]
        error_rows[r.name] = (ranges[r.name][0], index, half.flatten()[index])

    def collect_generators(G, bound_node, lA, _uA):
        if lA is None:
            return
        if bound_node is root:
            # Sparse initialization keeps one zero column for a fixed box.
            # Real coordinates occupy the first columns; padding stays zero.
            G[:, :, :coords.numel()] = (
                lA.transpose(0, 1).flatten(2)[:, :, coords] * radius[coords])
        elif bound_node.name in error_rows:
            start, index, half = error_rows[bound_node.name]
            # Read incoming A before the ReLU multiplies it by its slope:
            # even a zero slope can have a nonzero error generator.
            G[:, :, start:start + index.numel()] = (
                lA.transpose(0, 1).flatten(2)[:, :, index] * half)

    rows = {**generators, node.name: None}
    read = {}
    with _same_slope_alpha(self, slopes):
        for n in nodes:
            value = centres[n.name]
            index = (torch.arange(value.shape[1], device=value.device)
                     if rows[n.name] is None else torch.as_tensor(
                         rows[n.name], dtype=torch.long, device=value.device))
            G = value.new_zeros(1, index.numel(), widths[n.name])
            for i in range(0, index.numel(), chunk):
                elements = index[i:i + chunk]
                select = value.new_zeros(1, elements.numel(), value.shape[1])
                select[0, torch.arange(elements.numel(), device=value.device), elements] = 1.
                self.backward_general(
                    n, select, bound_lower=True, bound_upper=False,
                    incoming_A_callback=partial(
                        collect_generators, G[:, i:i + elements.numel()]))
            read[n.name] = G, value[:, index]
    G, c = read[node.name]
    lower, upper = _concretize(LinearBound(G.transpose(1, 2), c), C)
    node.check_bound_invariants(
        lower, upper, context='dynamic forward concretization')
    if generators.get(node.name) is not None:
        pick = torch.as_tensor(generators[node.name], dtype=torch.long, device=c.device)
        read[node.name] = G[:, pick], c[:, pick]
    return lower, upper, {name: read[name] for name in generators}


def forward_general_dynamic_optimized(self: 'BoundedModule', C=None,
                                      node: 'Bound' = None, generators=None,
                                      bound_lower=True, bound_upper=True):
    """Same-slope alpha-CROWN, followed by reading that one affine relaxation.

    The existing optimizer maximizes the worst query with fixed intermediate
    bounds and a network-wide alpha bank. Both directions read the selected
    alpha. The caller's alpha bank and optimization options are restored.
    Evaluation follows the caller's grad mode, with alpha and intermediate
    bounds fixed; gradients do not pass through the slope search.
    Iterations and learning rates come from ``optimize_bound_args``. The
    search uses detached inputs and specifications; evaluation uses the
    original inputs and leaves its normal compute_bounds state on the module.
    """
    if self.batch_size != 1:
        raise NotImplementedError(
            'dynamic-forward-optimized searches the slopes of one box')
    root = next(r for r in self.roots()
                if getattr(r, 'perturbation', None) is not None)
    _require_box_perturbation(root.perturbation)
    input_nodes = [self[name] for name in self.input_name
                   if self[name].model_input_index is not None]
    # Root names include parameters; runtime inputs use their binding indices.
    inputs = [None] * (max(n.model_input_index for n in input_nodes) + 1)
    for n in input_nodes:
        inputs[n.model_input_index] = n.value
    inputs = tuple(inputs)
    # Search treats the input domain as data. Copy perturbation metadata so
    # its initialization cannot change the caller's box or consume its graph.
    search_inputs = []
    for value in inputs:
        detached = value.detach() if isinstance(value, torch.Tensor) else value
        if isinstance(value, BoundedTensor) and value.ptb is not None:
            ptb = copy(value.ptb)
            for attr in ('x_L', 'x_U', 'eps'):
                tensor = getattr(ptb, attr, None)
                if isinstance(tensor, torch.Tensor):
                    setattr(ptb, attr, tensor.detach())
            detached = BoundedTensor(detached, ptb)
        search_inputs.append(detached)
    relus = _relus_of(self, [node])
    try:
        with _same_slope_alpha(self) as scoped:
            with torch.no_grad():
                self.check_prior_bounds(node, C=C)
            interm = {r.inputs[0].name: (r.inputs[0].lower.detach(),
                                        r.inputs[0].upper.detach()) for r in relus}
            if relus and scoped['optimize_bound_args']['iteration'] > 0:
                with torch.enable_grad():
                    self.compute_bounds(
                        x=tuple(search_inputs), C=C.detach(),
                        method='CROWN-optimized', bound_upper=False,
                        final_node_name=node.name, interm_bounds=interm)
            else:
                self.init_alpha(tuple(search_inputs), share_alphas='network', c=C.detach(),
                                final_node_name=node.name, interm_bounds=interm)
            slopes = {}
            for r in relus:
                alpha = r.shared_alpha()
                if alpha is not None:
                    slopes[r.name] = r.dynamic_forward_relaxation(
                        r.inputs[0].lower, r.inputs[0].upper, alpha)[0].detach()
    finally:
        # Input reuse always refers to the caller's tensors, also after failure.
        for n in input_nodes:
            n.bind_input(inputs[n.model_input_index])
    with _same_slope_alpha(self, slopes):
        ret = self.compute_bounds(
            x=inputs, C=C,
            method='dynamic-forward' if generators is not None else 'backward',
            final_node_name=node.name, interm_bounds=interm,
            bound_lower=bound_lower, bound_upper=bound_upper,
            return_generators=generators is not None,
            needed_generators_dict=generators, reuse_alpha=True)
    self.dynamic_forward_slopes = slopes
    return ret

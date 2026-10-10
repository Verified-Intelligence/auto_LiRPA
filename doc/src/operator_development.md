# Operator Development and Maintenance

This guide is for contributors who add operators to auto_LiRPA or maintain
existing implementations in `auto_LiRPA/operators/`. It describes the common
operator lifecycle, implementation patterns, state ownership rules, and test
expectations used by the built-in operator library.

Users who only need to register an operator without changing auto_LiRPA should
start with [Custom Operators](custom_op.md). A custom operator and a built-in
operator use the same bound-method contracts, but registration and repository
tests differ.

- **Implement a new operator:** define its supported cases (Section 2), choose
  a reference implementation (Section 3), then implement and test the required
  methods (Sections 4-9).
- **Study or maintain an operator:** locate its class through the registration
  rules (Section 8), read its base class and the relevant bound-method contract
  (Section 5), then use the maintenance checklist (Section 10).

## 1. Understand the Operator Lifecycle

A bound operator connects a source PyTorch or ONNX operation to one or more
bound-propagation methods. The same bound node can participate in several
stages:

```text
PyTorch/ONNX operation
    -> graph conversion creates a BoundFoo node
    -> the framework calls forward() and records value, shape, dtype, and batch dimension
    -> the graph marks which nodes and inputs are perturbed
    -> required intermediate input bounds are computed
    -> IBP, forward LiRPA, CROWN, or alpha-CROWN calls the node's bound methods
```

A node is created once when `BoundedModule` converts the graph and can be reused
by many `compute_bounds()` calls. Constructor fields are therefore persistent
node state. Concrete values, input bounds, linear relaxations, and optimized
parameters belong to later computation stages and may change between calls.

Supporting an operator means more than making one example run. For every case
the implementation accepts, concrete evaluation must match the source
operation and every returned bound must be mathematically sound. It is valid
to support only a documented subset of methods or input domains.

## 2. Define the Contract Before Choosing Code

Write down the cases that can change the answer before selecting a base class
or copying an existing implementation:

- What is the exact source function, including attributes, broadcasting, and
  output shape?
- Which inputs may be perturbed? Which must remain fixed?
- Does the function have restricted, singular, periodic, or unbounded domains?
- Which methods must work: IBP, forward LiRPA, CROWN, alpha-CROWN, dynamic
  forward propagation, or Jacobian propagation?
- Does correctness depend on current input bounds, or only on immutable
  operator attributes?
- Are dense `Tensor` coefficients sufficient, or can the operator receive
  `Patches`?
- Which dtype, device, batch, and shape combinations are already supported by
  the source operation and neighboring bound operators?

## 3. Find the Owning Operator Family

Start in `auto_LiRPA/operators/` and extend the closest mathematical and
representation owner. Prefer an established family over a new abstraction.

| Implementation task | Starting point | What to read in the example |
| --- | --- | --- |
| Propagate an exact affine operation | `Bound` | [`BoundAdd`](../../auto_LiRPA/operators/add_sub.py): distribute backward coefficients and add input intervals |
| Handle fixed versus perturbed operands | An existing multi-input operator | [`BoundMul`](../../auto_LiRPA/operators/bivariate.py): select exact multiplication or a bilinear relaxation |
| Reshape or select tensor elements | `Bound` | [`BoundReshape`](../../auto_LiRPA/operators/reshape.py) and [`BoundGather`](../../auto_LiRPA/operators/indexing.py): transform values and coefficient axes |
| Define a unary relaxation without optimized parameters | `BoundActivation` | [`BoundLog.bound_relax()`](../../auto_LiRPA/operators/convex_concave.py): construct a secant lower line and tangent upper line |
| Optimize a relaxation parameter | `BoundOptimizableActivation` | [`BoundSqrt`](../../auto_LiRPA/operators/convex_concave.py): initialize tangent points and constrain them to the input interval |
| Handle changes in curvature | A matching nonlinear family | [`BoundSShaped`](../../auto_LiRPA/operators/s_shaped.py): split the interval into curvature cases; [`BoundSin`](../../auto_LiRPA/operators/trigonometric.py): handle periods and extrema |
| Propagate convolutional coefficients | A matching convolution/pooling operator | [`BoundConv.bound_backward()`](../../auto_LiRPA/operators/convolution.py): handle dense and patch layouts separately |

Derive directly from `Bound` when the operator has specialized affine,
multi-input, shape, or indexing behavior. Derive from `BoundActivation` when a
unary operator can be described by fixed affine lower and upper relaxations.
Use `BoundOptimizableActivation` only when alpha-CROWN should optimize explicit
relaxation parameters.

Inheritance shares a contract, not just code. Before enabling an inherited
algorithm, verify that its mathematical assumptions hold for the new
operator. For example, a helper written for a globally S-shaped function is
not automatically valid for a periodic function.

## 4. Implement Concrete Behavior and Declare State

The graph converter supplies these four standard constructor arguments:

```python
class BoundFoo(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
```

- `attr`: attributes produced by graph parsing, such as an axis or stride.
- `inputs`: the input bound nodes, in source-operand order.
- `output_index`: the source output represented by this node, usually zero.
- `options`: bound-computation options.

Store operator attributes and initialize persistent caches in the constructor.
Populate bound-dependent state during bound computation. Do not eagerly create
tensors whose graph dtype or device is not known yet.

Implement `forward()` first. It must match the source operation for every
accepted concrete input, including dtype, device, broadcasting, axes, and
shape. The framework also uses this pass to record output shape, dtype, and
batch information.

Common state flags are narrow contracts. Set them only when their documented
condition is true:

| Field or helper | Meaning |
| --- | --- |
| `requires_input_bounds` | Input indices whose lower and upper bounds are needed by relaxation methods |
| `use_default_ibp` | Evaluating the primary input at its two endpoints gives valid output bounds, and every other input is fixed |
| `ibp_intermediate` | Intermediate bounds for this unary node should use IBP instead of CROWN |
| `input_dependent` | Changing runtime model input values at fixed shapes can change the output; the graph forward pass propagates this state |
| `blocks_input_dependency` | The operator derives its output from input structure and yields a value-fixed graph node |
| `batch_dim` | The output's batch-axis index; `-1` identifies a batch-invariant layout |
| `perturbed` | The output depends on an active perturbation; the graph conversion propagates this state |
| `splittable` | The operator supports activation splitting on its perturbed operands |
| `blocks_perturbation` | The operator output remains fixed under active perturbations |
| `blocks_jacobian` | The operator acts as a leaf during Jacobian graph construction |
| `is_input_perturbed(i)` | Returns true when input `i` depends on an active perturbation |

The driver calls `update_input_perturbation_state()` after propagating input
perturbation flags. Operators use it to update relaxation requirements and
input-dependent splitting eligibility. For example,
[`BoundMul.update_input_perturbation_state()`](../../auto_LiRPA/operators/bivariate.py)
uses both input bounds for a bilinear relaxation. With one fixed operand,
multiplication uses exact affine propagation.

`splittable` records split support, while `requires_input_bounds` records the
inputs needed to construct a relaxation. Mul and MatMul become splittable when
both operands are perturbed; fixing either operand makes the operation affine.
Other supported activations set
`self.splittable = True` in their constructors. Additional parameters use
fixed values.
`BoundedModule.disabled_split_nodes` applies verifier policy.
The verifier calls `get_split_mask()` later to select splittable coordinates
for the current domain bounds.

## 5. Implement the Required Bound Methods

The user-facing method determines which operator hook is needed:

| Bound workflow | Primary operator method |
| --- | --- |
| IBP | `interval_propagate()` |
| Forward LiRPA | `bound_forward()` |
| CROWN / backward LiRPA | `bound_backward()` |
| alpha-CROWN | optimized activation state plus backward LiRPA |
| Dynamic forward LiRPA | `bound_dynamic_forward()` |

Different methods may support different input cases. Unsupported cases must
fail explicitly; do not return a placeholder bound.

### Interval Bound Propagation

`interval_propagate(*v)` receives one elementwise input box per operand and
returns the output box. Read each input through `v[i].lower` and
`v[i].upper`, compute sound output endpoints, and return
`Interval(lower, upper)`. These are the method's input objects; the IBP driver
stores the returned endpoints in the output node's `lower` and `upper` fields.

Set `self.use_default_ibp = True` only when
`forward(v[0].lower, ...)` and `forward(v[0].upper, ...)` are valid lower and
upper bounds and every other input is fixed. Otherwise, implement the required
endpoint, sign-split, periodic-extrema, or other interval arithmetic directly.

`BoundActivation.interval_propagate()` also evaluates the two endpoints in
this order. Override it for a decreasing or nonmonotone activation: decreasing
functions swap the endpoints, while nonmonotone functions may attain an
extremum inside the interval. See
[`BoundGelu.interval_propagate()`](../../auto_LiRPA/operators/gelu.py) for an
interior-minimum case.

#### Structured Input Perturbations

Most operators only need `interval_propagate()`. The optional hooks below are
for retaining tighter input-set information along selected graph paths.

For Lp (`1 <= p < infinity`) and L0 model inputs, the IBP driver passes an
enclosing `Interval` to each operator and stores the resulting endpoints in
the node's `lower` and `upper` bounds. It also tracks
`state=(center, perturbation)` while that representation remains valid along a
graph path.

An operator that can enclose its output in another structured set
can implement `_propagate_input_perturbation_state(v, states)`. The method
receives one `Interval` and one `(center, perturbation)` state or `None` per
input. It returns the output `(center, perturbation)` state, or `None` if the
representation is no longer valid. Shape transformations can preserve the set
exactly; BatchNorm and Concat instead construct enclosing L2 balls.

An affine operator that directly bounds the represented set can implement
`_bound_from_input_perturbation_state(*v, state, **kwargs)`. The method returns
the output `Interval` using the state associated with its first input, after
which propagation continues with that box. Linear supports finite-p and L0
formulas; Conv supports L2 and L0 formulas. Without this consuming hook, the
driver uses the enclosing box produced by `interval_propagate()`. The
propagation hook independently determines whether the structured state remains
available to downstream nodes. The IBP driver owns storage and invalidation of
the state; operators must not attach it to nodes or put it in an `Interval`.

Use these implementations as references:

| Task | Reference | Key condition or operation |
| --- | --- | --- |
| Preserve state through a shape change | [`BoundReshape._propagate_input_perturbation_state()`](../../auto_LiRPA/operators/reshape.py) | Transform the center without mixing batch samples |
| Combine graph branches | [`BoundConcat._propagate_input_perturbation_state()`](../../auto_LiRPA/operators/slice_concat.py) | Enclose the concatenated L2 sets using the square root of the sum of squared radii |
| Apply BatchNorm to L2 state | [`BoundBatchNormalization._propagate_input_perturbation_state()`](../../auto_LiRPA/operators/normalization.py) | Transform the center and scale the radius by the largest absolute scale factor |
| Bound an affine map directly | [`BoundLinear._bound_from_input_perturbation_state()`](../../auto_LiRPA/operators/linear.py) | Consume finite-p or L0 state and return a box |
| Bound a convolution directly | [`BoundConv._bound_from_input_perturbation_state()`](../../auto_LiRPA/operators/convolution.py) | Consume supported L2 or L0 state and return a box |
| Follow state lifetime and dispatch | [`IBP_general()`](../../auto_LiRPA/interval_bound.py) | Store state separately from the endpoint bounds |

### Forward LiRPA

`bound_forward(dim_in, *x)` receives `LinearBound` inputs and returns a
`LinearBound`. Its principal fields are:

- `lw`, `uw`: lower and upper coefficients, normally shaped
  `(batch, dim_in, *value_shape)`. `dim_in` indexes the shared forward input
  coordinates, including coordinates from different perturbed roots.
- `lb`, `ub`: lower and upper bias terms, normally shaped
  `(batch, *value_shape)`.
- `lower`, `upper`: concretized bounds when available.

The coefficient fields can be `None` for a coefficient-free input. Preserve
the extra coefficient axis when reshaping or broadcasting; it is not a value
axis. See [`BoundReshape.bound_forward()`](../../auto_LiRPA/operators/reshape.py)
for a shape transformation that preserves it. Use `zeros_like()`,
`new_zeros()`, and related constructors so newly created tensors follow their
owning inputs.

### Dynamic Forward LiRPA

Dynamic forward encloses each node's values in an affine relaxation over
shared **variables**, each ranging over `[-1, 1]`:

```
relaxed_node = c + G e,    e in [-1, 1]^n
```

`c` is the relaxation's **centre**, its value when all input and relaxation
variables are zero, and `G` its **generators**, one column per variable.
For example, ReLU on `[-1, 1]` has `c = 0.25` and `G = [0.5, 0.25]`:
the first variable describes the input and the second describes the ReLU error.

Input variables come first. Dense initialization allocates one per input
coordinate, with zero generators for fixed coordinates. Sparse initialization
allocates the batch's maximum number of moving coordinates (at least one
column), padding rows with fewer moving coordinates with zero generators.
Each ReLU then allocates the batch's maximum number
of unstable neurons (`lower < 0 < upper`), with zero generators for padding.

`bound_dynamic_forward(*x, max_dim=None, offset=0)` is a separate method,
implemented only when the operator supports dynamic forward. It receives its
inputs as `LinearBound`s and returns one: `lb` and `ub` are the same tensor,
the centre, shaped `[batch, *value_shape]`; `lw` and `uw` are the same tensor,
the generators, shaped `[batch, variables, *value_shape]`. The input box is
already folded in: its centre is in `lb`, and each input variable's generator
is scaled by its coordinate's half-width.

All nodes in one `compute_bounds` call share one **numbering of the
variables**: `LinearBound.dynamic_variable_ranges` maps a node name to the
`(start, count)` columns that node owns. Ranges never overlap and the input's
come first, so column `j` is the same variable in every node. A node's
generators cover the columns `[0, W)`, with `W` the end of the last range it
depends on (its own included); `tot_dim` records `W`. A **pass** computes one
**slice** of the columns, `[offset, offset + max_dim)`, and its tensors hold
only those columns.

Three rules follow for an operator:

- **Do not renumber columns.** A shape-changing or linear operator carries
  `x.tot_dim` through unchanged and only reshapes the generators.
- **Pad, never truncate, when operands disagree.** Two operands of a merge can
  reach different widths. The narrower one does not depend on the variables in
  the extra columns, so give it zero generators there; see
  [`BoundAdd.bound_dynamic_forward()`](../../auto_LiRPA/operators/add_sub.py).
  Aligning them any other way makes independent variables cancel and produces
  unsound bounds.
- **Take a range before introducing variables.** An operator that relaxes looks
  up its `(start, count)` in `dynamic_variable_ranges` under its own name,
  taking the next free columns the first time a pass reaches it, pads the
  columns between its inputs' width and `start` with zeros, and writes its own
  generators from `start`; see
  [`BoundRelu.bound_dynamic_forward()`](../../auto_LiRPA/operators/relu.py).
  Honor `max_dim` so a slice stays within its columns.

Operators that cannot honor these rules should simply not implement the
method; the base class raises for them.

`compute_bounds(method='dynamic-forward', return_generators=True,
needed_generators_dict={name: rows})`
returns the centre and generators of the named nodes, from the same
passes that bound the target, under the bounds in
`interm_bounds`. Omitting `needed_generators_dict` returns the current bound
target's full linear function; an empty dictionary returns no entries. The
return tuple has three elements whenever `return_generators` is `True`.
Requested nodes must be the current bound target or its ancestors; requests
outside that subgraph raise `ValueError`, regardless of previously cached bounds.

### Backward LiRPA

`bound_backward(last_lA, last_uA, *x, **kwargs)` receives input **bound nodes**
in `x`, with concrete endpoint bounds available as `x[i].lower` and
`x[i].upper` when requested by `requires_input_bounds`. It propagates objective
coefficients to each input and returns:

```python
[(input0_lA, input0_uA), ...], lower_bias, upper_bias
```

Either incoming coefficient can be `None` when that side is not requested.
Return one `(lA, uA)` pair for every input, using `None` for a coefficient that
does not propagate. Handle tensor broadcasting through helpers such as
`broadcast_backward()`.

For ordinary batch-first values, dense backward coefficients are shaped
`(spec, batch, *value_shape)`, and reduced bias contributions are
`(spec, batch)` tensors or scalar zero. This layout differs from forward
LiRPA's `(batch, dim_in, *value_shape)`.

An incoming coefficient may be a dense `Tensor` or `Patches`. Follow an
existing operator with the same layout and use the patch helpers; do not
unconditionally convert patches to dense matrices.
Some specialized operators also accept compact specifications such as `eyeC`
and `OneHotC`; see [`BoundLinear`](../../auto_LiRPA/operators/linear.py) if the
new operator can receive these representations.

### Nonlinear and Optimized Relaxations

For `BoundActivation` subclasses, `bound_relax()` initializes affine slopes
and intercepts through `init_linear_relaxation()` and
`add_linear_relaxation()`. Derive each lower line below the source function and
each upper line above it on the complete accepted interval. The
`BoundActivation` base class uses these lines to implement both forward and
backward LiRPA, so subclasses normally do not need to duplicate those methods.

Here `self.lw`, `self.lb`, `self.uw`, and `self.ub` describe local lines bounding
the activation as a function of its immediate input. They are not the
input-root coefficients stored in a forward `LinearBound`.

For optimized relaxations, read
[`BoundSqrt`](../../auto_LiRPA/operators/convex_concave.py) together with
[`BoundOptimizableActivation`](../../auto_LiRPA/operators/activation_base.py):

- `_init_opt_parameters_impl(size_spec, name_start)` returns the initial alpha
  tensor; the base class stores it in `self.alpha[name_start]`.
- `bound_relax()` uses fixed choices during ordinary CROWN and initialization.
  During `opt` or `reuse`, it uses the alpha entry for the current start node.
- Keep optimized parameters in their valid range. `BoundSqrt` clamps its tangent
  points to the input interval before forming slopes and intercepts.

Jacobian propagation is a separate capability. If needed, implement
`build_gradient_node()` as documented in `Bound`, and ensure the resulting
gradient graph also has supported bound operators. See
[`BoundTanh.build_gradient_node()`](../../auto_LiRPA/operators/s_shaped.py) for
an activation whose derivative uses its own bound operator.

## 6. Handle Domains and Unsupported Cases

Keep source semantics separate from backend capability:

- Raise `OperatorDomainError` when the input bounds include values outside the
  real domain of the source operation.
- Raise `OperatorCapabilityError` when the source input is valid but the
  requested bound method or relaxation does not support that case.
- Use assertions only for internal invariants whose failure indicates a bug,
  not for expected model input or capability checks.

Do not clamp an input, replace infinity with a finite sentinel, or substitute a
numerically convenient function when that changes source semantics. An
operator that cannot soundly handle a valid interval should request splitting
through an established contract or report a capability limitation.

Check boundary points, constants, zero-crossing intervals, unbounded
intervals, and singular points deliberately. A concrete sample inside the
domain does not prove that the later bound interval remains inside it.

## 7. Own Runtime Tensors Explicitly

Every tensor created by an operator needs an owner for dtype, device, shape,
and lifecycle. Prefer input-aware constructors. Do not fall back to
`torch.get_default_dtype()` or an implicit CPU tensor when a current bound,
coefficient, or parameter should own the result.

### Precomputed Relaxation Tables

Some nonlinear relaxations use sampled derivatives, tangent points, or lookup
tables. First establish that the associated relaxation is mathematically valid
for the operator on every accepted interval. An inherited table helper is not
itself evidence that the algorithm applies.

Precomputed tensors are persistent runtime state owned by each bound node. The
graph dtype and device may not be known in the constructor, and different
nodes in one graph may use different dtypes or devices. Initialize table fields
to `None`, and call `self._ensure_precomputed(x.lower)` from `bound_relax()`
before reading any tables. Call it on each use: the hook builds missing tables
or reuses a valid cache. Subclass attributes are already initialized when this
runs. The following excerpt shows the cache implementation;
`_build_tables()` is the operator-specific table builder:

```python
class BoundFoo(BoundActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self._precomputed_table_key = None
        self.tangent_x = None
        self.tangent_slope = None

    def _ensure_precomputed(self, reference):
        key = (reference.dtype, reference.device)
        tables = (self.tangent_x, self.tangent_slope)
        if (key == self._precomputed_table_key
                and all(table is not None for table in tables)):
            return

        self.tangent_x, self.tangent_slope = self._build_tables(
            dtype=reference.dtype, device=reference.device)
        self._precomputed_table_key = key
```

The cache validity check must account for every property that changes table
contents or placement. This always includes dtype and device and can also
include an exponent, approximation range, mode, or another operator parameter.
An explicit key is convenient, but checking every cached tensor and relevant
attribute directly is also valid.

Rebuild the complete table set when the validity criteria change; otherwise
reuse the same tensors across CROWN, forward LiRPA, and optimized bounds. Route
all consumers through one `_ensure_precomputed()` entry point. Do not derive
table ownership from a construction-time `forward_value`.

`reference` supplies only dtype and device; its values and shape do not define
the table. The hook updates node-owned fields, while builders such as
`precompute_relaxation()` implement the mathematical table generation.

Choose an implementation example according to the cache you need:

- **Fixed function parameters:**
  [`BoundSShaped._ensure_precomputed()`](../../auto_LiRPA/operators/s_shaped.py)
  checks whether all tables exist and match the input-bound dtype/device.
- **A parameter can change the table:**
  [`BoundPow._ensure_precomputed()`](../../auto_LiRPA/operators/s_shaped.py)
  also includes the exponent in cache validity. A different exponent requires
  a new table even when dtype/device are unchanged.
- **A subclass needs different tables:**
  [`BoundGelu._ensure_precomputed()`](../../auto_LiRPA/operators/gelu.py)
  builds its four GeLU tangent tables instead of the parent's S-shaped tables.
  Override the hook to manage the complete table set required by the subclass.

## 8. Register the Operator

Built-in ONNX operations named `onnx::Foo` resolve by convention to a class
named `BoundFoo`. Export the class through
`auto_LiRPA/operators/__init__.py` so graph conversion can find it.

Add an entry to `auto_LiRPA/bound_op_map.py` when the parsed name does not
follow that convention or maps to a differently named class:

```python
bound_op_map['custom::Foo'] = BoundFoo
```

External users can instead pass `custom_ops={'custom::Foo': BoundFoo}` to
`BoundedModule` or call `register_custom_op()`. Keep repository-level built-in
registration separate from application-specific custom registration.

If the source operation itself is new, its export or symbolic function must
preserve every input and attribute needed by the bound implementation. Inspect
the converted graph rather than assuming an eager PyTorch signature is
preserved unchanged.

## 9. Test the Operator Contract

Search `tests/` for the operator, source function, and closest operator family
before creating a file. Extend the existing owner when the new case exercises
the same contract:

- `test_1d_activation.py` covers broad unary activation behavior across IBP,
  CROWN, and optimized bounds.
- `test_operator_domains.py` covers domain, capability, and bound-invariant
  behavior shared by several operators.
- Family files such as `test_pow.py`, `test_reciprocal.py`, `test_gather.py`,
  and `test_softmax_backward.py` own focused family behavior.
- `test_bound_ops.py` is a reference for direct `bound_forward()`,
  `bound_backward()`, and interval checks. `test_general_shape.py` covers
  broadcasting and nontrivial shapes.
- `tests/testcase.py::TestCase` provides deterministic seeds, device/dtype
  setup, and optional stored-reference comparison.

Create `tests/test_<operator>.py` only when no existing file has coherent
ownership and the operator has a substantial independent test matrix.

Cover behavior rather than source-code structure:

1. Compare `forward()` with the source PyTorch operation.
2. Use explicit input boxes and test every supported bound method.
3. Check small deterministic coefficients or output bounds against hand-written
   expected tensors when practical.
4. Exercise meaningful broadcasting, axes, shapes, fixed inputs, and perturbed
   input combinations.
5. Test domain boundaries and unsupported cases as public contracts.
6. Build a `BoundedModule` in at least one test so graph conversion and
   registration are covered.

For nonlinear operations, verify representative concrete points inside each
box. Sampling supplements but does not replace a soundness argument. Include a
counterexample designed to invalidate the proposed strategy, not only the
original reported input.

For precomputed tables, keep numerical behavior tests separate from one
focused lifecycle test. Verify lazy creation, node-owned dtype/device, reuse on
compatible calls, and rebuilding when a validity property changes:

- **Creation and reuse:**
  [`test_gelu_table_lifecycle_and_ownership`](../../tests/test_gelu.py) checks
  that each node creates and reuses its own tables.
- **Parameter-dependent invalidation:**
  [`TestPowTableOwnership.test_exponent_change_invalidates_tables`](../../tests/test_pow.py)
  changes the exponent and checks the rebuilt tables and resulting bounds.

Keep library tests CPU-only; use mixed FP32/FP64 nodes to test per-node
ownership. Behavior tests should also confirm that changing the global default
dtype does not change a fixed graph's results.

Run focused tests first, then neighboring and full regressions:

```bash
source .venv/bin/activate
pytest -q tests/test_gather.py
pytest -q tests/test_operator_domains.py -k reciprocal
OMP_NUM_THREADS=1 pytest -n 10 --dist worksteal
```

GPU and solver tests require their documented hardware, data, and licenses.

## 10. Maintain an Existing Operator

Treat a maintenance change as a contract change even when the patch is small:

1. Identify the source semantics and currently supported bound methods.
2. Find the owning class, its base classes, all relevant consumers, and the
   nearest tests.
3. Enumerate fixed and perturbed operands, domain regions, broadcasting cases,
   and persistent state that can affect the result.
4. Add a focused behavior-level regression and confirm it fails on the
   pre-fix implementation for the expected reason.
5. Repair the earliest owner that creates the invalid bound or state. Do not
   hide the defect with a downstream fallback.
6. Re-run the reported case, a counterexample to the strategy, neighboring
   methods, and the relevant lifecycle tests.
7. Keep core operator fixes separate from verifier or interface changes unless
   one public contract genuinely requires both.

When modifying cached or optimized state, trace its complete lifecycle:
creation, invalidation, reuse, storage, restoration, and final consumption.
When modifying a shared base class, test representative child families rather
than assuming inheritance alone establishes compatibility.

## 11. Review Checklist

- Concrete behavior matches the source operation without hidden clamps or
  approximations.
- Every lower line is below the source function and every upper line is above
  it on the complete accepted interval.
- Supported and unsupported domain and perturbation cases are explicit.
- Shapes, batch dimensions, broadcasting, dtype, and device are preserved.
- Dense and `Patches` coefficient paths are handled when required.
- Persistent tensors have a clear node owner and complete invalidation rule.
- Optimized state is valid across initialization, optimization, and reuse.
- Built-in or custom registration works through an end-to-end
  `BoundedModule` conversion.
- Tests cover each supported method, realistic edge cases, and the relevant
  state lifecycle.
- The focused regression fails on the pre-fix implementation for the intended
  reason.

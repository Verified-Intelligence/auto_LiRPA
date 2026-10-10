Bound Options
====================

Bound options can be set by passing a dictionary to the `bound_opts` argument for `BoundedModule`.
This page lists available bound options.

## Arguments for Optimizing Bounds (`optimize_bound_args`)

Arguments for optimizing bounds with the `CROWN-Optimized` method can be provided as a dictionary. `dynamic-forward-optimized` uses the same iteration, optimizer, learning-rate, and stopping settings, and imposes the relaxation-specific settings described in the [shared affine relaxation example](examples.md#optimizing-and-reading-a-shared-affine-relaxation). Available arguments include:

* `enable_alpha_crown` (bool, default `True`): Enable α-CROWN (optimized CROWN/LiRPA).

* `enable_beta_crown` (bool, default `False`): Enable β-CROWN.

* `optimizer` (str, default `adam`): Optimizer: `adam` or `sgd`.

* `lr_alpha` (float, default 0.5), `lr_beta` (default 0.05): Learning rates for α and β parameters in α-CROWN and β-CROWN.

* `lr_decay` (float, default 0.98): Learning rate decay factor for the `ExponentialLR` scheduler.

* `iteration` (int, default 20): Number of optimization iterations.

* `loss_reduction_func` (function): Function for loss reduction over the specification dimension. By default, `auto_LiRPA.utils.reduction_sum` sums each batch element's specification bounds.

* `stop_criterion_func` (function): Early-stop criterion returning a boolean tensor of shape `(batch_size, 1)`, or a scalar boolean. The default returns `False`. Helpers include `auto_LiRPA.utils.stop_criterion_min`, `stop_criterion_max`, and `stop_criterion_all`. For example, `stop_criterion_min(threshold)` marks a batch element complete when its minimum specification bound exceeds the threshold.

* `keep_best` (bool, default `True`): If `True`, save α, β and bounds at the best iteration. Otherwise the last iteration result is used.

* `use_shared_alpha` (bool or `'network'`, default `False`): If `True`, all intermediate neurons from the same layer share the same set of α variables during bound optimization. For a very large model, enabling this option can save memory, at a cost of slightly looser bound. With `'network'`, each ReLU neuron has one α shared across specifications, backward start nodes, and lower and upper bounds. This mode supports ReLU only. Together with `activation_bound_option='same-slope'`, it defines one affine relaxation of the network, which can be read by either forward or backward propagation.

* `reproduce_alpha_bounds` (bool, default `False`): Save bounds and α from one best iteration, selected by `loss_reduction_func`, so replaying those α reproduces the returned bounds. With shared α and multiple specifications, enable this option to avoid combining bounds from different iterations.

* `fix_interm_bounds` (bool, default `True`): Optimize final bounds with fixed intermediate bounds during alpha/beta CROWN.

* `init_alpha` (bool, default `True`): Initial alpha variables by calling CROWN once.

* `early_stop_patience` (int, default, 10): Number of iterations that we will start considering early stop if tracking no improvement.

* `start_save_best` (float, default 0.5): Start to save optimized best bounds when current_iteration > int(iteration*start_save_best)


## Activation Initialization (`activation_bound_option`)

There are different choices for the lower bound relaxation of unstable ReLU activations (see the [CROWN paper](https://arxiv.org/pdf/1811.00866.pdf)):

* `adaptive` (default): For unstable ReLU neurons, use lower slope 1 when the upper slope is greater than 0.5, and 0 otherwise (CROWN-Ada). Optimized CROWN learns the lower slopes from this initialization.

* `same-slope`: Make the slope for lower bound the same as the upper bound.

* `zero-lb`: Always use 0 as the slope of lower bound for unstable neurons.

* `one-lb`: Always use 1 as the slope of lower bound for unstable neurons.

## Other Options

* `dynamic_forward` (dict): Dynamic-forward generator extraction options. `generators` selects `'auto'` (default), `'forward'`, or `'reverse'`. Reverse extraction supports one input box. `dynamic-forward-optimized` searches one input box using `optimize_bound_args`, including `iteration`, `lr_alpha`, `lr_decay`, and stopping criteria. It maximizes the worst lower bound using one shared same-slope ReLU relaxation and fixed intermediate bounds. Upper bounds use those same slopes. Alpha and beta parameters already on the module are preserved. Gradients of the result hold the selected slopes and intermediate bounds fixed; input-box and model gradients remain available. Search excludes beta, cuts, output-constraint refinement, and mixed-precision optimization. See the [shared affine relaxation example](examples.md#optimizing-and-reading-a-shared-affine-relaxation) for the API, supported scope, and a runnable CPU example.

* `loss_fusion`: If `True`, this bounded module has loss fusion, i.e., the loss function is also included in the module and the output of the model is the loss rather than logits.

* `deterministic`: If `True`, make PyTorch use deterministic algorithms.

* `clampmult_patches_padding_pos_masked` (bool, default `False`): If `True`, mark `Patches` returned by clamp multiplication as having padding positions already masked.

* `matmul`: If set to `economic`, use a memory-efficient IBP implementation for relaxing the `matmul` operation when both arguments of `matmul` are perturbed, which does not expand all the elementary multiplications to save memory.

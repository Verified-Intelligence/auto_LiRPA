API Usage
======================================

.. autoclass:: auto_LiRPA.BoundedModule

   .. autofunction:: auto_LiRPA.BoundedModule.forward
   .. autofunction:: auto_LiRPA.BoundedModule.compute_bounds

.. autoclass:: auto_LiRPA.bound_ops.Bound

   .. autofunction:: auto_LiRPA.bound_ops.Bound.forward
   .. autofunction:: auto_LiRPA.bound_ops.Bound.interval_propagate
   .. autofunction:: auto_LiRPA.bound_ops.Bound.bound_forward
   .. autofunction:: auto_LiRPA.bound_ops.Bound.bound_backward

.. autoclass:: auto_LiRPA.perturbations.Perturbation

   .. autofunction:: auto_LiRPA.perturbations.Perturbation.concretize
   .. autofunction:: auto_LiRPA.perturbations.Perturbation.init

.. autofunction:: auto_LiRPA.bound_op_map.register_custom_op
   
Indices and tables
-------------------

* :ref:`genindex`
* :ref:`search`

..
   * :ref:`modindex`
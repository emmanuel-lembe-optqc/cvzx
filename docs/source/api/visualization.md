# `cvzx.visualization`

`DiagramVisualizer` itself is assembled from several private per-diagram-type/concern
mixins (`cvzx.visualization.proper`, `.composition`, `.contracted`, `.swap_fourier`,
`.overflow`) plus a shared `cvzx.visualization.geometry` and a structural-typing
`cvzx.visualization.protocol` — not part of the public surface, so not separately
documented here; see this module's own docstring for how they fit together. Runnable
examples for every gate type live in `tests/visualization/` (`test_visualize_gates.py`,
`test_visualize_base_gates.py`).

## `cvzx.visualization.core`

```{eval-rst}
.. automodule:: cvzx.visualization.core
   :members:
   :undoc-members:
   :show-inheritance:
```

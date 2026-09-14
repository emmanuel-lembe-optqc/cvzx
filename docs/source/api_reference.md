# API reference

Full, auto-generated reference for every public class and function, grouped by module in
the order described in {doc}`dev_guide/architecture`.

## `cvzx.exceptions`

```{eval-rst}
.. automodule:: cvzx.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.config`

```{eval-rst}
.. automodule:: cvzx.config
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.backend`

```{eval-rst}
.. automodule:: cvzx.backend
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.ir.base`

```{eval-rst}
.. automodule:: cvzx.ir.base
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.ir.gates`

```{eval-rst}
.. automodule:: cvzx.ir.gates
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.backends.nx.graph`

```{eval-rst}
.. automodule:: cvzx.backends.nx.graph
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.backends.rx.graph`

```{eval-rst}
.. automodule:: cvzx.backends.rx.graph
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.passes.normalize`

```{eval-rst}
.. automodule:: cvzx.passes.normalize
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.backends.nx.rules`

```{eval-rst}
.. automodule:: cvzx.backends.nx.rules
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.backends.rx.rules`

```{eval-rst}
.. automodule:: cvzx.backends.rx.rules
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.passes.optimize`

```{eval-rst}
.. automodule:: cvzx.passes.optimize
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.passes.completion`

```{eval-rst}
.. automodule:: cvzx.passes.completion
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.lowering.bridges.mqc3`

```{eval-rst}
.. automodule:: cvzx.lowering.bridges.mqc3
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.lowering.dag`

```{eval-rst}
.. automodule:: cvzx.lowering.dag
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.lowering.lowering`

```{eval-rst}
.. automodule:: cvzx.lowering.lowering
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.logging_config`

```{eval-rst}
.. automodule:: cvzx.logging_config
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.utils.helpers`

```{eval-rst}
.. automodule:: cvzx.utils.helpers
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cvzx.visualization.core`

`DiagramVisualizer` itself is assembled from several private per-diagram-type/concern
mixins (`cvzx.visualization.proper`, `.composition`, `.contracted`, `.swap_fourier`,
`.overflow`) plus a shared `cvzx.visualization.geometry` and a structural-typing
`cvzx.visualization.protocol` -- not part of the public surface, so not separately
documented here; see this module's own docstring for how they fit together.

```{eval-rst}
.. automodule:: cvzx.visualization.core
   :members:
   :undoc-members:
   :show-inheritance:
```

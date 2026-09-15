# `cvzx.ir`

## `cvzx.ir.base`

```{eval-rst}
.. automodule:: cvzx.ir.base
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: ZxPoly
```

`ZxPoly` is documented separately, without `:undoc-members:`: it subclasses `sympy.Poly`, and
`:undoc-members:` on a class with an external base pulls in that base's own undocumented
internals (`rep`, `gens`, `default_assumptions`, `is_commutative`, ...) alongside cvzx's own
members -- noise that isn't part of this codebase's API.

```{eval-rst}
.. autoclass:: cvzx.ir.base.ZxPoly
   :members:
   :show-inheritance:
```

## `cvzx.ir.gates`

```{eval-rst}
.. automodule:: cvzx.ir.gates
   :members:
   :undoc-members:
   :show-inheritance:
```

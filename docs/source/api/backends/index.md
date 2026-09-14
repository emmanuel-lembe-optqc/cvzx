# `cvzx.backends`

`cvzx` can build its rewrite graph on top of either `networkx` or `rustworkx` (see
{doc}`../../user_guide/optimization`). The two backends mirror the same schema and expose
the same `CVZXGraph` class and rewrite rules, implemented independently once per graph
library rather than shared — so a bare `CVZXGraph` cross-reference elsewhere in the docs is
genuinely ambiguous between the two, by design.

```{toctree}
:maxdepth: 1

nx
rx
```

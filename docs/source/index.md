# cvzx

**cvzx** is a diagram-rewriting compiler for continuous-variable (CV) quantum circuits, built
around the CV ZX calculus of {cite}`nagayoshi2024zx`. It represents a circuit as a `Diagram`
of q-/p-spiders and gate leaves, converts it to a `networkx` graph, and simplifies it by
repeatedly applying a fixed set of graph-rewrite rules (fusion, chain reduction, terminal
absorption, Fourier normalization, the copy rule, ...) until nothing more matches.

The gate set and naming conventions follow MQC3's `graph`/`circuit` operations, so a `cvzx` diagram optimized with `cvzx.optimize.optimize` is meant to
compile down cleanly onto that machinery's measurement-angle model. `cvzx.circuit_to_diagram`
and `cvzx.diagram_to_circuit` convert to and from an actual mqc3 `CircuitRepr`, and
`cvzx.lowering` carries a diagram the rest of the way to mqc3's own `DependencyDAG` machinery
via a pluggable per-QPU backend.

This documentation has three parts:

- **{doc}`theory`** — the CV ZX calculus background this library implements, and a
  translation table between the paper's notation and `cvzx`'s classes/functions.
- **{doc}`user_guide/index`** — task-oriented guides for building, rewriting, optimizing,
  converting to/from mqc3 circuits, and visualizing diagrams, with runnable examples drawn
  from the test suite.
- **{doc}`dev_guide/index`** — the architectural decisions behind the codebase: why the
  rewrite engine operates on graphs rather than the `Diagram` tree directly, how the
  type-1/type-2 stage normalization works, and the module dependency structure.

```{toctree}
:maxdepth: 2
:hidden:

theory
user_guide/index
dev_guide/index
api_reference
references
```

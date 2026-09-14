# Architecture overview

## Module map

```{image} ../_static/module_map.png
:alt: cvzx module dependency graph
:width: 100%
```

This is a strict dependency order — nothing later in the list is imported by anything earlier
(the diagram draws each module's most architecturally relevant direct dependency/dependencies,
not the full import graph — see the module's own docstring for its complete import list).
`exceptions.py` and `config.py` have no `cvzx` imports at all and sit at the foundation;
`ir/base.py` (the `Diagram` hierarchy) depends only on `exceptions.py`.

`cvzx.backend.get_backend_modules()` is what makes the rest of the pipeline dual-backend:
`backends/nx/graph.py`/`backends/nx/rules.py` and `backends/rx/graph.py`/`backends/rx/rules.py` are parallel,
near-identical implementations over `networkx.DiGraph` and `rustworkx.PyDiGraph`
respectively, and `optimize(diagram, backend=...)` picks between them (defaulting to
`cvzx.config.DEFAULT_BACKEND`, which auto-detects whether `rustworkx` is installed).
`passes/normalize.py` itself is nx-only (it needs one concrete graph representation to
canonicalize into, and nx was it first); `optimize()` bridges that by converting through
whichever backend module `to_graph`/`to_diagram` it was given.

Three more modules sit downstream of `passes/normalize.py`, not pictured above:
`cvzx.lowering.bridges.mqc3` (both `from_circuit_repr` and `to_circuit_repr` — the two
directions of the mqc3 bridge live in one module, see below) depends on `ir/base.py` +
`ir/gates.py` + `passes/normalize.py` (nothing from `backends/nx/rules.py` or
`passes/optimize.py`); `cvzx.passes.completion` (boundary closing) depends only on
`cvzx.backend`/`ir/base.py`; and `cvzx.lowering.dag` + `cvzx.lowering.lowering` sit downstream
of both `lowering/bridges/mqc3.py` and `passes/completion.py` in turn. See the pipeline
diagram below for where they fit, and {doc}`../user_guide/circuit_conversion` for how to use
them.

## Two representations of a circuit, on purpose

`cvzx` deliberately keeps two representations of a circuit alive side by side, and converts
between them (`cvzx.backends.nx.graph.to_graph` / `to_diagram`) rather than picking one:

- **`Diagram`** (`ir/base.py`, `ir/gates.py`) — a nested tree: `TensorDiagram`/
  `CompositionDiagram`/`ContractedDiagram` containers of `Diagram` leaves. This is the
  representation you *build* a circuit in, and the one {doc}`visualization <../user_guide/visualization>`
  draws. It is a poor representation to *rewrite*: finding "does this leaf have a
  same-color spider two doors down, possibly through a nested sub-composition" requires
  walking the tree structure itself, and a rewrite that merges two leaves living in
  different containers has no natural place to attach the result.
- **`networkx.DiGraph` or `rustworkx.PyDiGraph`** (`backends/nx/graph.py` / `backends/rx/graph.py` — two
  near-identical implementations of the same schema, selected via `cvzx.backend`) — every
  leaf is a flat, randomly-addressable node (`kind="proper"` or `"compact"`), and every
  container is *also* a node (`kind="container"`, with `container_type` one of `"tensor"`,
  `"composition"`, or `"contracted"`) holding its children's node IDs and, for compositions, a
  `connectivity` dict. A `"composition"`/`"contracted_internal"` edge directly connects two
  **leaf** nodes regardless of how differently nested their containers are — see
  {doc}`rewrite_engine` for why that one property is what makes graph-based rewriting
  tractable at all, and {doc}`../user_guide/optimization` for how to pick a backend.

`GateRegister` (`backends/nx/graph.py` / `backends/rx/graph.py`) is the third piece: it indexes the graph's nodes by category
(squeezing gates, rotation gates, Fourier gates, identity spiders, input states, measurement
nodes, and the three container types) so a rule's `match()` can look up "every rotation
gate" in O(1) instead of scanning the whole graph. It has to be kept in sync with the graph
manually (`add_node`/`remove_node`, or rebuilt with `build_from_graph`) — nothing does this
for you automatically, so any code that mutates the graph directly (rather than going
through a `RewriteRule`) is responsible for updating the registry itself.

## The full pipeline: `CircuitRepr` to `DependencyDAG`

The end-to-end path from a user-authored mqc3 `CircuitRepr`, through `cvzx`'s own
canonicalization/optimization/boundary-completion, to an mqc3 `DependencyDAG` (every box below
is the actual function call that performs that step; the greyed-out tail is mqc3's own
downstream embedding/machinery pipeline, which `cvzx` hands off to but does not implement):

```{image} ../_static/pipeline_overview.png
:alt: Compiler pipeline overview, from CircuitRepr to DependencyDAG
:width: 100%
```

`cvzx.lowering.bridges.mqc3.from_circuit_repr` and `cvzx.lowering.bridges.mqc3.to_circuit_repr` are
the two halves of the round trip at the `Diagram` boundary. `cvzx.passes.completion.complete_boundaries`
closes an optimized diagram's open input/output ports (fresh ideal states in, fresh symbolic
measurement effects out) — the precondition both lowering backends below rely on: an mqc3
`CircuitRepr`/`DependencyDAG` has no concept of an externally-supplied mode, so every wire must
terminate at a real state/measurement node first. `cvzx.lowering.lowering.graph_to_dependency_dag`
is the pluggable dispatch point that turns that closed diagram into an mqc3 `DependencyDAG`.

`cvzx.lowering.lowering` exists as a plugin point (rather than hardcoding one fixed
`Diagram -> DependencyDAG` path) because a `DependencyDAG` can be built more than one way: the
bundled `"mqc3"` backend (`Mqc3ReferenceBackend`) is the simplest correct implementation —
`to_circuit_repr` then mqc3's own `DependencyDAG(circuit)` constructor — while `"cvzx-direct"`
(`CvzxDirectBackend`) skips the `CircuitRepr` round-trip entirely and discovers execution order
directly from the diagram's own `CVZXGraph` structure (dual-backend, via `cvzx.lowering.dag`),
reusing `cvzx.lowering.bridges.mqc3`'s per-leaf translators rather than duplicating them. Both are
verified to produce an isomorphic `DependencyDAG` for the same input. A QPU wanting a different
construction strategy registers its own `LoweringBackend` without touching `get_backend` or any
other registered backend. From there, `DependencyDAG` is QPU-agnostic and ready for mqc3's own
`GraphEmbedder` (`beamsearch.py`/`greedy.py`) to embed into a `GraphRepr`, and ultimately
`mqc3.machinery` to lower into a `MachineryRepr` — see {doc}`../user_guide/circuit_conversion`
for a worked example and how to register an alternative `LoweringBackend`.

### The `optimize()` step in more detail

```{image} ../_static/optimize_round_detail.png
:alt: optimize() round detail, from Diagram to OptimizeResult
:width: 100%
```

The two outputs are captured at different points *on purpose*: `remove_void_and_identity_nodes`
can restructure the graph in ways the nested `Diagram` tree can no longer losslessly
represent, so the `Diagram` snapshot used for visualization is taken immediately before that
final pass, not derived from the cleaned graph afterward. See
{doc}`../user_guide/optimization` for the user-facing version of this, and `optimize()`'s own
docstring for the full reasoning behind the round-repeat structure.

# Architecture overview

## Module map

```{image} ../_static/module_map.png
:alt: cvzx module dependency graph
:width: 100%
```

This is a strict dependency order — nothing later in the list is imported by anything earlier. `base_gates.py`
in particular has no `cvzx` imports at all: it is the one module every other module can
assume is self-contained.

Three more modules sit downstream of `normalize_diagram.py`, not pictured above: `cvzx.circuit_to_diagram`
and `cvzx.diagram_to_circuit` each depend on `base_gates.py` + `gates.py` +
`normalize_diagram.py` (nothing from `nx_rewrite_rules.py` or `optimize.py`), and
`cvzx.lowering` depends on `diagram_to_circuit.py` in turn. See the pipeline diagram below
for where they fit, and {doc}`../user_guide/circuit_conversion` for how to use them.

## Two representations of a circuit, on purpose

`cvzx` deliberately keeps two representations of a circuit alive side by side, and converts
between them (`cvzx.nx_graph.to_graph` / `to_diagram`) rather than picking one:

- **`Diagram`** (`base_gates.py`, `gates.py`) — a nested tree: `TensorDiagram`/
  `CompositionDiagram`/`ContractedDiagram` containers of `Diagram` leaves. This is the
  representation you *build* a circuit in, and the one {doc}`visualization <../user_guide/visualization>`
  draws. It is a poor representation to *rewrite*: finding "does this leaf have a
  same-color spider two doors down, possibly through a nested sub-composition" requires
  walking the tree structure itself, and a rewrite that merges two leaves living in
  different containers has no natural place to attach the result.
- **`networkx.DiGraph`** (`nx_graph.py`) — every leaf is a flat, randomly-addressable node
  (`kind="proper"` or `"compact"`), and every container is *also* a node
  (`kind="container"`, with `container_type` one of `"tensor"`, `"composition"`, or
  `"contracted"`) holding its children's node IDs and, for compositions, a `connectivity`
  dict. A `"composition"` edge directly connects two **leaf** nodes regardless of how
  differently nested their containers are — see {doc}`rewrite_engine` for why that one
  property is what makes graph-based rewriting tractable at all.

`cvzx.nx_graph.GateRegister` is the third piece: it indexes the graph's nodes by category
(squeezing gates, rotation gates, Fourier gates, identity spiders, input states, measurement
nodes, and the three container types) so a rule's `match()` can look up "every rotation
gate" in O(1) instead of scanning the whole graph. It has to be kept in sync with the graph
manually (`add_node`/`remove_node`, or rebuilt with `build_from_graph`) — nothing does this
for you automatically, so any code that mutates the graph directly (rather than going
through a `RewriteRule`) is responsible for updating the registry itself.

## The full pipeline: `CircuitRepr` to `MachineryRepr`

The end-to-end path from a user-authored mqc3 `CircuitRepr` down to a QPU's `MachineryRepr`,
through `cvzx`'s own canonicalization/optimization and mqc3's downstream embedding machinery
(every box below is the actual function call that performs that step):

```{image} ../_static/pipeline_overview.png
:alt: Compiler pipeline overview, from CircuitRepr to MachineryRepr
:width: 100%
```

`cvzx.circuit_to_diagram.from_circuit_repr` and `cvzx.diagram_to_circuit.to_circuit_repr` are
the two halves of the round trip at the `Diagram` boundary; `cvzx.lowering.graph_to_machinery_repr`
is the pluggable dispatch point that carries the canonicalized diagram the rest of the way to a
concrete `MachineryRepr`, through mqc3's own `DependencyDAG`/`GraphEmbedder`/`GraphRepr` chain.

`cvzx.lowering` exists as a plugin point (rather than hardcoding one fixed
`Diagram -> MachineryRepr` path) because mqc3 already treats the embedding step as one of its
own: `GraphEmbedder` is an abstract base class, with `beamsearch.py`/`greedy.py` as two concrete
strategies for embedding a `DependencyDAG` into a `GraphRepr` differently (trading off layout
quality against how expensive the search is). A `LoweringBackend` lets a QPU pick a different
embedding strategy, or skip the `CircuitRepr` round-trip entirely and build the `DependencyDAG`
directly from the diagram's own structure, without touching `get_backend`,
`graph_to_machinery_repr`, or any other registered backend. The bundled `"mqc3"` backend is
deliberately the simplest correct implementation: `to_circuit_repr`, mqc3's own
`DependencyDAG(circuit)` constructor, `GreedyEmbedder` (the simpler of mqc3's two embedders),
and `MachineryRepr.from_graph_repr` — see {doc}`../user_guide/circuit_conversion` for a worked
example and how to register an alternative.

### The `optimize()` step in more detail

```text
Diagram
  │
  ▼
normalize_diagram()  ── alternating type-1 (narrow-gate) / type-2 (one wide gate) stages
  │                     (see normalization.md — this MUST run before expansion, because it
  │                     only understands compact-form 2-mode gates)
  ▼
[assume_infinite_squeezing?] expand_two_mode_gates()  ── BS/CSUM -> ContractedDiagram
  │
  ▼
to_graph()
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ repeat to a fixed point:                                 │
│   IdentityRule → FusionRule → ChainReductionRule →       │
│   FourierNormalizationRule → TerminalAbsorptionRule →    │
│   [assume_infinite_squeezing?] CopyRule                  │
└─────────────────────────────────────────────────────────┘
  │
  ▼
to_diagram()  ── fed back into normalize_diagram() for another round,
  │              until a round changes nothing (or max_rounds is hit)
  ▼
to_diagram(graph)  ── captured as `.diagram` (pre-cleanup snapshot)
  │
  ▼
remove_void_and_identity_nodes()  ── run exactly once, on the graph only
  │
  ▼
OptimizeResult(graph=<cleaned graph>, diagram=<pre-cleanup snapshot>)
```

The two outputs are captured at different points *on purpose*: `remove_void_and_identity_nodes`
can restructure the graph in ways the nested `Diagram` tree can no longer losslessly
represent, so the `Diagram` snapshot used for visualization is taken immediately before that
final pass, not derived from the cleaned graph afterward. See
{doc}`../user_guide/optimization` for the user-facing version of this, and `optimize()`'s own
docstring for the full reasoning behind the round-repeat structure.

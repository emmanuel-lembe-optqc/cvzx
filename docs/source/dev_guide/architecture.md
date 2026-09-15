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

### `GateRegister`'s symbolic-parameter/feedforward indexes

Besides the category sets above, `GateRegister` also indexes symbolic-parameter and
feedforward provenance, populated by `_index_parameters` (and retracted by
`_retract_parameter_index`) every time `add_node`/`remove_node` runs:

- `parametric_nodes` / `symbol_registry`: which nodes carry a free `sympy.Symbol` in any of
  their parametric fields (a spider's `phase`, or a `CompactDiagram` gate's own parameter
  fields — `alpha`/`beta`/`lam`, `a`/`b`, etc.), and the reverse index from symbol to the set
  of nodes using it.
- `feedforward_nodes` / `measurement_to_feedforward_map`: which nodes have `feedforward=True`,
  and the reverse index from a measurement node's id to the set of nodes whose
  `measurement_ids` includes it — i.e. which downstream gates depend on that measurement's
  outcome.

`CVZXGraph.parameter_consistency_violations()` (and its raising counterpart
`validate_parameter_consistency()`) uses these two indexes to check the graph is internally
consistent: (1) every symbol used by 2+ nodes must be bound (via each node's own
`param_measurement_map[symbol]`) to the *same* measurement-id set everywhere it's declared —
a node that merely carries the symbol without declaring a binding (e.g. the measurement leaf
that originates it) is excluded from the comparison rather than treated as a conflict; (2)
every measurement id referenced by `measurement_to_feedforward_map` must still be a
registered measurement node. Symbol conflicts are reported (and raised, as
`ParameterConflictError`) before unbound-measurement ones (`UnboundMeasurementError`), since
an unresolved conflict makes any measurement-existence finding downstream of it suspect too.

## Diagram building blocks (`ir/base.py`)

### `ContractedDiagram`: partial trace over two diagrams

`ContractedDiagram` is the output of applying the contraction rule to a tensor of two diagrams
D1 and D2, from {cite}`nagayoshi2024zx`, Definition 11, Eq. (51):

```text
∫∫ ds̄ dȳ ⟨s_i| D1 |s_j⟩ ⊗ q⟨s_j| D2 |s_i⟩
```

The connections are `(I1, I2)` — outputs `I1` of the first diagram connect to inputs `I2` of
the second (forward) — and `(J1, J2)` — outputs `J2` of the second diagram connect to inputs
`J1` of the first (feedback). After connection, the integral over the connected variables is
implicit in the diagrammatic language; only wires not named in any of the four index sequences
remain external. `TensorDiagram.partial_trace` is the entry point that builds one: it requires
the two diagrams being contracted to be adjacent in the tensor product, which is purely a
layout restriction (to keep the resulting diagram easy to draw) rather than a fundamental one —
in principle partial trace could apply to any two diagrams in a `TensorDiagram`.

### `VoidDiagram`: keeping container arity stable under cross-container rewrites

The motivating case is `CopyRule`'s cross-container application: when a state/effect is copied
through a spider that lives in a *different* container, the state/effect's own original slot
has nothing left to put there (its content now lives as copies elsewhere) — but simply deleting
that slot would shrink its container's arity and force an arity-propagation cascade through
every parent container above it. Installing a `VoidDiagram` with the exact same arity instead
keeps that slot's shape identical to what it replaced, so nothing upstream ever needs to be
touched or recomputed. `VoidDiagram` is meant to be transient: the end-of-pipeline cleanup pass
(part of `optimize()`) removes every `VoidDiagram` for good, alongside any leftover identity
wires, actually shrinking the containers they sit in at that point once and for all, rather
than doing so eagerly on every application. Visually, a `VoidDiagram` reserves exactly the
layout space an identity wire of the same arity would take, but draws nothing — unlike an
identity spider, which draws as a straight wire.

### `ZxPoly`: phase polynomials as a wrapped `sympy.Poly`

`ZxPoly` wraps `sympy.Poly` (a single generator, `x`) so that CV-ZX phase functions can be
built from a plain `dict[degree, coeff]` as well as the usual sympy forms:

```python
>>> p = ZxPoly({0: 1.0, 2: -0.5})   # 1 - 0.5·x²
>>> q = ZxPoly({1: 2.0})             # 2·x
>>> (p + q).coeffs                   # 1 + 2·x - 0.5·x²
{0: 1.0, 1: 2.0, 2: -0.5}
>>> (p * q).coeffs                   # 2·x - x³
{1: 2.0, 3: -1.0}

>>> from sympy import symbols
>>> a, b = symbols('a b')
>>> (ZxPoly({0: a, 1: b}) + ZxPoly({0: 1, 1: 2})).coeffs   # (a+1) + (b+2)·x
{0: a + 1, 1: b + 2}

>>> from sympy import Poly
>>> x = symbols('x')
>>> ZxPoly(x**2 + 2*x + 1)           # from a sympy expression
>>> ZxPoly(Poly(x**2 + 1, x))        # from a sympy.Poly
>>> ZxPoly()                          # the zero polynomial
```

`.coeffs` always omits zero coefficients, converts numeric coefficients to Python `float` (or
`complex`, if the coefficient has a nonzero imaginary part), and leaves symbolic coefficients
as sympy expressions. `repr()` renders terms in increasing-degree order using ZX-calculus
notation (`·` for multiplication, `^` for exponents, e.g. `ZxPoly({0: 1.0, 1: 2.0, 2: 3.0})` →
`'1.0 + 2.0·x + 3.0·x^2'`, and the zero polynomial → `'0'`).

### `flatten_composition`: normalizing nested composition structure

`flatten_composition` recursively collapses nested `CompositionDiagram`s into one flat
sequence, adjusting connectivity indices to match:

```text
CompositionDiagram([A]) → A
CompositionDiagram([A, CompositionDiagram([B, C]), D])
    → CompositionDiagram([A, B, C, D])
TensorDiagram([CompositionDiagram([A, B]), C])
    → TensorDiagram([CompositionDiagram([A, B]), C])  # Composition inside Tensor is NOT flattened
CompositionDiagram([TensorDiagram([A, B]), C])
    → CompositionDiagram([TensorDiagram([A, B]), C])  # Tensor inside Composition is NOT flattened
```

Only composition-inside-composition is actually flattened; a composition nested inside a
`TensorDiagram` or `ContractedDiagram` is recursed into (in case *it* contains further nested
compositions) but left in place there.

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

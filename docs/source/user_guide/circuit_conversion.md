# Converting to and from mqc3 circuits

`cvzx.lowering.bridges.mqc3.from_circuit_repr` and `cvzx.lowering.bridges.mqc3.to_circuit_repr` are
the two halves of a round trip between an mqc3 `CircuitRepr` and a `cvzx` `Diagram`, and
`cvzx.lowering.lowering.graph_to_dependency_dag` carries a closed `Diagram` the rest of the way
to mqc3's own `DependencyDAG`. See {doc}`../dev_guide/architecture` for how these modules fit
into the full pipeline.

## mqc3 `CircuitRepr` to a `Diagram`

```python
from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic

from cvzx.lowering.bridges.mqc3 import from_circuit_repr

circuit = CircuitRepr("simple")
circuit.Q(0) | intrinsic.PhaseRotation(0.4)
circuit.Q(0) | intrinsic.Measurement(0.1)

diagram = from_circuit_repr(circuit)
```

`from_circuit_repr` naively translates each mqc3 operation and initial state into the
compact-form `cvzx.ir.gates` equivalent (see `cvzx.lowering.bridges.mqc3`'s module docstring for the
exact per-gate formulas), then canonicalizes the result via
`cvzx.passes.normalize.normalize_diagram` — pass `normalize=False` to get the raw,
non-canonical translation instead. A handful of mqc3 features have no `cvzx` counterpart yet
and raise `NotImplementedError`: the `Manual` gate (reached directly, or indirectly via a
`std.*` operation that lowers to it), a feedforward parameter, and any initial state outside
the idealized-squeezed-state / pure-Gaussian-`BosonicState` set the module docstring
documents.

## `Diagram` back to mqc3 `CircuitRepr`

```python
from cvzx.lowering.bridges.mqc3 import to_circuit_repr

circuit2 = to_circuit_repr(diagram)
```

`to_circuit_repr` canonicalizes `diagram` first (a no-op if it already is canonical), then
walks it stage by stage, translating each primitive leaf independently — it does **not**
attempt to pattern-match a canonicalized diagram's leaves back into the composite gate
(`BeamSplitter`, `Squeezing`) they may have come from, so a round-tripped circuit is
semantically equivalent but not necessarily op-for-op identical to the original: a
`BeamSplitter(sqrt_r, theta_rel)` round-trips as a `PhaseRotation` + `BeamSplitter(sqrt_r, 0)` +
two more `PhaseRotation`s, for instance, not as a single `BeamSplitter(sqrt_r, theta_rel)` call
(see `cvzx.lowering.bridges.mqc3`'s module docstring for the exact per-leaf formulas and which
cases — `ControlledSumGate`/`CubicPhaseGate`, a nonzero-phase state/effect leaf, a symbolic
parameter, external inputs, any `ContractedDiagram` — raise `NotImplementedError`/`ValueError`
instead of silently mistranslating).

Since `cvzx` has no numeric simulation backend of its own, the round trip is checked
structurally rather than against a reference statevector — that the translated circuit is
well-formed and that mqc3's own `DependencyDAG` accepts it:

```python
from mqc3.graph.embed.dep_dag import DependencyDAG

dag = DependencyDAG(circuit2)
assert dag.dag.number_of_nodes() > 0
```

## Lowering straight to a `DependencyDAG`

`cvzx.lowering.lowering.graph_to_dependency_dag` skips the `to_circuit_repr`/`DependencyDAG`
boilerplate at the call site — it dispatches to a registered `LoweringBackend` that performs
the `Diagram -> DependencyDAG` step however it sees fit, defaulting to `"mqc3"`. `diagram` must
already be closed first (`num_inputs == num_outputs == 0` — see
`cvzx.passes.completion.complete_boundaries`, {doc}`optimization`):

```python
from cvzx.lowering.lowering import graph_to_dependency_dag

dep_dag = graph_to_dependency_dag(diagram)  # backend="mqc3" by default
```

The bundled `"mqc3"` backend (`Mqc3ReferenceBackend`) is `to_circuit_repr` followed by mqc3's
own `DependencyDAG(circuit)` constructor — every limitation of `to_circuit_repr` above applies
transitively to it. A second bundled backend, `"cvzx-direct"` (`CvzxDirectBackend`), skips the
`CircuitRepr` round-trip entirely: it discovers execution order directly from the diagram's
own `CVZXGraph` structure (dual-backend, via `cvzx.lowering.dag`), reusing this module's own
per-leaf translators rather than duplicating them — both are verified to produce an isomorphic
`DependencyDAG` for the same input. A QPU that needs a different `Diagram -> DependencyDAG`
lowering registers its own `LoweringBackend` subclass with the `register_backend` class
decorator:

```python
from cvzx.lowering.lowering import LoweringBackend, register_backend


@register_backend("my_qpu")
class MyQpuBackend(LoweringBackend):
    def to_dependency_dag(self, diagram):
        ...
```

`list_backends()` lists every currently-registered backend name, and `get_backend(name)` looks
one up directly; `graph_to_dependency_dag(diagram, backend="my_qpu")` then dispatches to it —
nothing else in the pipeline needs to change. From there, `DependencyDAG` is ready for mqc3's
own `GraphEmbedder`/`GraphRepr`/`MachineryRepr` chain, which is outside `cvzx`'s own scope (see
{doc}`../dev_guide/architecture`).

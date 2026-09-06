# Converting to and from mqc3 circuits

`cvzx.circuit_to_diagram.from_circuit_repr` and `cvzx.diagram_to_circuit.to_circuit_repr` are
the two halves of a round trip between an mqc3 `CircuitRepr` and a `cvzx` `Diagram`, and
`cvzx.lowering.graph_to_dependency_dag` carries a `Diagram` the rest of the way to mqc3's own
`DependencyDAG` machinery. See {doc}`../dev_guide/architecture` for how these three modules
fit into the full pipeline.

## mqc3 `CircuitRepr` to a `Diagram`

```python
from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic

from cvzx.circuit_to_diagram import from_circuit_repr

circuit = CircuitRepr("simple")
circuit.Q(0) | intrinsic.PhaseRotation(0.4)
circuit.Q(0) | intrinsic.Measurement(0.1)

diagram = from_circuit_repr(circuit)
```

`from_circuit_repr` naively translates each mqc3 operation and initial state into the
compact-form `cvzx.gates` equivalent (see `cvzx.circuit_to_diagram`'s module docstring for the
exact per-gate formulas), then canonicalizes the result via
`cvzx.normalize_diagram.normalize_diagram` — pass `normalize=False` to get the raw,
non-canonical translation instead. A handful of mqc3 features have no `cvzx` counterpart yet
and raise `NotImplementedError`: the `Manual` gate (reached directly, or indirectly via a
`std.*` operation that lowers to it), a feedforward parameter, and any initial state outside
the idealized-squeezed-state / pure-Gaussian-`BosonicState` set the module docstring
documents.

## `Diagram` back to mqc3 `CircuitRepr`

```python
from cvzx.diagram_to_circuit import to_circuit_repr

circuit2 = to_circuit_repr(diagram)
```

`to_circuit_repr` canonicalizes `diagram` first (a no-op if it already is canonical), then
walks it stage by stage, translating each primitive leaf independently — it does **not**
attempt to pattern-match a canonicalized diagram's leaves back into the composite gate
(`BeamSplitter`, `Squeezing`) they may have come from, so a round-tripped circuit is
semantically equivalent but not necessarily op-for-op identical to the original (see
`cvzx.diagram_to_circuit`'s module docstring for the exact per-leaf formulas and which cases —
`ControlledSumGate`/`CubicPhaseGate`, a nonzero-phase state/effect leaf, a symbolic parameter,
external inputs, any `ContractedDiagram` — raise `NotImplementedError`/`ValueError` instead of
silently mistranslating).

Since `cvzx` has no numeric simulation backend of its own, the round trip is checked
structurally rather than against a reference statevector — that the translated circuit is
well-formed and that mqc3's own `DependencyDAG` accepts it:

```python
from mqc3.graph.embed.dep_dag import DependencyDAG

dag = DependencyDAG(circuit2)
assert dag.dag.number_of_nodes() > 0
```

## Lowering straight to a `DependencyDAG`

`cvzx.lowering.graph_to_dependency_dag` skips the intermediate `CircuitRepr` at the call site
— it dispatches to a registered `LoweringBackend` that performs the `Diagram -> DependencyDAG`
step however it sees fit:

```python
from cvzx.lowering import graph_to_dependency_dag

dag = graph_to_dependency_dag(diagram)  # backend="mqc3" by default
```

The bundled `"mqc3"` backend (`cvzx.lowering.Mqc3ReferenceBackend`) is exactly
`to_circuit_repr` followed by mqc3's own `DependencyDAG(circuit)` constructor — every
limitation of `to_circuit_repr` applies transitively to it. A QPU that needs a different
`Diagram -> DependencyDAG` lowering — for instance, one that wants to skip the `CircuitRepr`
round-trip and build the DAG directly from the diagram's own structure — registers its own
`LoweringBackend` subclass with the `register_backend` class decorator:

```python
from cvzx.lowering import LoweringBackend, register_backend


@register_backend("my_qpu")
class MyQpuBackend(LoweringBackend):
    def to_dependency_dag(self, diagram):
        ...
```

`list_backends()` lists every currently-registered backend name, and `get_backend(name)`
looks one up directly; `graph_to_dependency_dag(diagram, backend="my_qpu")` then dispatches to
it — nothing else in the pipeline needs to change.

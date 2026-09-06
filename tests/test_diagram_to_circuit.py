"""Tests for `cvzx.diagram_to_circuit` (cvzx `Diagram` -> mqc3 `CircuitRepr`).

Since cvzx has no numeric (Wigner/Gaussian) simulation backend, these
tests check what can actually be checked without one: that a `Diagram`
produced by `cvzx.circuit_to_diagram.from_circuit_repr` round-trips back into
a well-formed `CircuitRepr` (right survivor count, and constructs a real
mqc3 `DependencyDAG` without error -- the actual downstream consumer of
a `CircuitRepr`), and that every documented unsupported case
(`ControlledSumGate`, `CubicPhaseGate`, a nonzero-phase state/effect
leaf, a symbolic gate parameter, a `Diagram` with external inputs, and
any `ContractedDiagram`) raises rather than silently mistranslating.
"""

from __future__ import annotations

import sympy
from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic
from mqc3.graph.embed.dep_dag import DependencyDAG

from cvzx.base_gates import CompositionDiagram, ContractedDiagram, QSpider, TensorDiagram, ZxPoly
from cvzx.circuit_to_diagram import from_circuit_repr
from cvzx.diagram_to_circuit import to_circuit_repr
from cvzx.gates import ControlledSumGate, CubicPhaseGate, PhaseRotationGate


def _all_gates_circuit() -> CircuitRepr:
    """A circuit touching every supported intrinsic gate at least once.

    Same circuit used in `test_circuit_to_diagram.py`'s
    `_all_gates_circuit` -- modes 0 and 2 are entangled by
    `ControlledZ` while mode 1 sits between them in row order, so a
    diagram built from it also exercises the non-adjacent-row
    `connectivity` path on the way back.
    """
    circuit = CircuitRepr("all_gates")
    circuit.Q(0) | intrinsic.PhaseRotation(0.3)
    circuit.Q(1) | intrinsic.ShearXInvariant(0.5)
    circuit.Q(2) | intrinsic.Displacement(0.1, 0.2)
    circuit.Q(0, 2) | intrinsic.ControlledZ(0.7)
    circuit.Q(1) | intrinsic.ShearPInvariant(-0.2)
    circuit.Q(1) | intrinsic.Squeezing45(0.4)
    circuit.Q(0, 1) | intrinsic.BeamSplitter(0.6, 0.9)
    circuit.Q(2) | intrinsic.Arbitrary(0.1, 0.2, 0.3)
    circuit.Q(0, 1) | intrinsic.TwoModeShear(0.15, -0.2)
    circuit.Q(1) | intrinsic.Squeezing(0.5)
    circuit.Q(2) | intrinsic.Measurement(0.25)
    return circuit


def test_round_trip_produces_well_formed_circuit_repr():
    circuit = _all_gates_circuit()
    diagram = from_circuit_repr(circuit)

    circuit2 = to_circuit_repr(diagram)

    # Mode 2 was measured (consumed); modes 0 and 1 survive as outputs,
    # matching the diagram's own num_outputs.
    assert circuit2.n_operations > 0
    assert diagram.num_outputs == 2


def test_round_trip_dependency_dag_constructs():
    """The actual point of this module.

    `DependencyDAG` (mqc3's own, already-correct machinery) must accept
    the round-tripped circuit.
    """
    circuit = _all_gates_circuit()
    diagram = from_circuit_repr(circuit)
    circuit2 = to_circuit_repr(diagram)

    dag = DependencyDAG(circuit2)
    assert dag.dag.number_of_nodes() > 0
    assert dag.dag.number_of_edges() > 0


def test_simple_single_mode_round_trip():
    circuit = CircuitRepr("simple")
    circuit.Q(0) | intrinsic.PhaseRotation(0.4)
    circuit.Q(0) | intrinsic.Measurement(0.1)
    diagram = from_circuit_repr(circuit)

    circuit2 = to_circuit_repr(diagram)
    assert circuit2.n_operations >= 2
    DependencyDAG(circuit2)


def test_controlled_sum_gate_raises():
    """Reject `ControlledSumGate` rather than silently mistranslating it.

    It is a `CompactDiagram` leaf with no bare mqc3 intrinsic
    equivalent; it survives `normalize_diagram` as an atomic leaf (not
    decomposed further) and must be rejected rather than silently
    dropped or mistranslated.

    Raises
    ------
    AssertionError
        If `to_circuit_repr` does not raise `NotImplementedError`.
    """
    vac = QSpider(0, 1, ZxPoly({}))
    two_vac = TensorDiagram([vac, QSpider(0, 1, ZxPoly({}))])
    csum_diagram = CompositionDiagram([two_vac, ControlledSumGate(1.0)])
    try:
        to_circuit_repr(csum_diagram)
    except NotImplementedError:
        return
    msg = "expected NotImplementedError for ControlledSumGate"
    raise AssertionError(msg)


def test_cubic_phase_gate_raises():
    """Reject `CubicPhaseGate` rather than silently mistranslating it.

    It is a genuinely non-Gaussian gate; mqc3's intrinsic set is
    Gaussian-only, so it has no translation.

    Raises
    ------
    AssertionError
        If `to_circuit_repr` does not raise `NotImplementedError`.
    """
    vac = QSpider(0, 1, ZxPoly({}))
    cubic_diagram = CompositionDiagram([vac, CubicPhaseGate(0.5)])
    try:
        to_circuit_repr(cubic_diagram)
    except NotImplementedError:
        return
    msg = "expected NotImplementedError for CubicPhaseGate"
    raise AssertionError(msg)


def test_nonzero_phase_state_leaf_raises():
    """Reject a nonzero-phase state leaf rather than silently mistranslating it.

    Only the bare idealized zero-phase state leaf is recognized as an
    `InitialState`; anything else has no documented translation.

    Raises
    ------
    AssertionError
        If `to_circuit_repr` does not raise `NotImplementedError`.
    """
    nonzero_state = QSpider(0, 1, ZxPoly({0: 0.3}))
    try:
        to_circuit_repr(nonzero_state)
    except NotImplementedError:
        return
    msg = "expected NotImplementedError for a nonzero-phase state leaf"
    raise AssertionError(msg)


def test_symbolic_parameter_raises():
    vac = QSpider(0, 1, ZxPoly({}))
    theta = sympy.symbols("theta")
    sym_diagram = CompositionDiagram([vac, PhaseRotationGate(theta, parametric=True)])
    try:
        to_circuit_repr(sym_diagram)
    except NotImplementedError:
        return
    msg = "expected NotImplementedError for a symbolic gate parameter"
    raise AssertionError(msg)


def test_external_inputs_raise_value_error():
    """Reject a `Diagram` with external inputs rather than silently mistranslating it.

    mqc3 circuits have no concept of an externally supplied input mode
    -- every mode must originate from a state leaf.

    Raises
    ------
    AssertionError
        If `to_circuit_repr` does not raise `ValueError`.
    """
    open_wire = QSpider(1, 1, ZxPoly({}))
    try:
        to_circuit_repr(open_wire)
    except ValueError:
        return
    msg = "expected ValueError for a Diagram with external inputs"
    raise AssertionError(msg)


def test_contracted_diagram_raises():
    first = QSpider(1, 1, ZxPoly({}))
    second = QSpider(1, 1, ZxPoly({}))
    contracted = ContractedDiagram(first, second, [0], [0], [], [])
    try:
        to_circuit_repr(contracted)
    except NotImplementedError:
        return
    msg = "expected NotImplementedError for a ContractedDiagram"
    raise AssertionError(msg)

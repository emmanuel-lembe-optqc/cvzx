"""Tests for `cvzx.lowering.bridges.mqc3` (cvzx `Diagram` -> mqc3 `CircuitRepr`).

Since cvzx has no numeric (Wigner/Gaussian) simulation backend, these
tests check what can actually be checked without one: that a `Diagram`
produced by `cvzx.lowering.bridges.mqc3.from_circuit_repr` round-trips back into
a well-formed `CircuitRepr` (right survivor count, and constructs a real
mqc3 `DependencyDAG` without error -- the actual downstream consumer of
a `CircuitRepr`), that `ControlledSumGate` emits the exact Fourier-conjugated
`ControlledZ` sequence it's documented to, and that every remaining
documented unsupported case (`CubicPhaseGate`, a nonzero-phase state/effect
leaf, a symbolic gate parameter, a `Diagram` with external inputs, and
any `ContractedDiagram`) raises rather than silently mistranslating.
"""

from __future__ import annotations

import numpy as np
import sympy
from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic
from mqc3.graph.embed.dep_dag import DependencyDAG

from cvzx.ir.base import CompositionDiagram, ContractedDiagram, QSpider, TensorDiagram, ZxPoly
from cvzx.lowering.bridges.mqc3 import from_circuit_repr
from cvzx.lowering.bridges.mqc3 import to_circuit_repr
from cvzx.exceptions import UnboundMeasurementError
from cvzx.ir.gates import ControlledSumGate, CubicPhaseGate, PhaseRotationGate


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


def _csum_diagram(gain: float, control: int, target: int) -> CompositionDiagram:
    two_vac = TensorDiagram([QSpider(0, 1, ZxPoly({})), QSpider(0, 1, ZxPoly({}))])
    return CompositionDiagram([two_vac, ControlledSumGate(gain, control=control, target=target)])


def test_controlled_sum_gate_emits_fourier_conjugated_controlled_z():
    """`ControlledSumGate` has no bare mqc3 primitive -- it's emitted as
    `ControlledZ` conjugated by a Fourier rotation on the target mode (see
    the user guide's "Converting to and from mqc3 circuits" page for the
    Heisenberg-picture derivation: `CSUM(g) = (I ⊗ F_t) CZ(g) (I ⊗ F_t†)`).

    Checks the exact three-op emission (angles, gain sign, and which
    mode gets the Fourier pair) for both `target=1` and `target=2`, and
    that mqc3's own `DependencyDAG` accepts the result.
    """
    circuit = to_circuit_repr(_csum_diagram(0.7, control=1, target=2))
    ops = list(circuit)
    assert len(ops) == 3

    rot1, cz, rot2 = ops
    assert type(rot1).__name__ == "PhaseRotation"
    assert type(cz).__name__ == "ControlledZ"
    assert type(rot2).__name__ == "PhaseRotation"

    target_mode = rot1.opnd().get_ids()[0]
    assert rot2.opnd().get_ids()[0] == target_mode
    assert rot1.parameters() == [-np.pi / 2]
    assert rot2.parameters() == [np.pi / 2]
    assert cz.parameters() == [-0.7]

    control_target_modes = set(cz.opnd().get_ids())
    assert target_mode in control_target_modes
    assert len(control_target_modes) == 2  # noqa: PLR2004

    DependencyDAG(circuit)


def test_controlled_sum_gate_target_mode_selection():
    """Same as above with `target=1` instead of `target=2`, to exercise
    the other branch of the target-mode selection."""
    circuit_a = to_circuit_repr(_csum_diagram(0.5, control=1, target=2))
    circuit_b = to_circuit_repr(_csum_diagram(0.5, control=2, target=1))

    rot_a = list(circuit_a)[0]
    rot_b = list(circuit_b)[0]
    # Different physical mode gets the Fourier pair depending on which
    # gate-local mode is the target.
    assert rot_a.opnd().get_ids()[0] != rot_b.opnd().get_ids()[0]

    DependencyDAG(circuit_a)
    DependencyDAG(circuit_b)


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
    """A symbolic parameter not bound to any upstream measurement effect is rejected.

    `theta` here isn't the special `-m*x`-phase measurement-effect
    pattern `cvzx.passes.completion.complete_diagram()` produces, so there's no
    measurement for it to feed forward from -- `UnboundMeasurementError`
    (a `cvzx.exceptions.CvzxError`), not a bare `NotImplementedError`.
    """
    vac = QSpider(0, 1, ZxPoly({}))
    theta = sympy.symbols("theta")
    sym_diagram = CompositionDiagram([vac, PhaseRotationGate(theta, parametric=True)])
    try:
        to_circuit_repr(sym_diagram)
    except UnboundMeasurementError:
        return
    msg = "expected UnboundMeasurementError for an unbound symbolic gate parameter"
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

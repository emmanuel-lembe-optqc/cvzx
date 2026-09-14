"""Tests for `cvzx.lowering.bridges.mqc3` (mqc3 `CircuitRepr` -> cvzx `Diagram`).

Since cvzx has no numeric (Wigner/Gaussian) simulation backend, these
tests check what can actually be checked without one: that translation
succeeds and produces a well-formed `Diagram` (right input/output arity,
survives `to_graph`/`to_diagram`/`optimize`) for circuits exercising
every supported gate -- including a 2-mode gate on non-adjacent rows,
which exercises the `connectivity`-based layer-reordering path rather
than the simple adjacent-row case -- that an affine feedforward
parameter round-trips into a symbolic gate parameter bound to a
reconstructed measurement-effect leaf, and that every documented
unsupported case (`Manual`, unsupported initial states, an empty
circuit) raises rather than silently mistranslating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic, std
from mqc3.circuit.state import BosonicState, GaussianState, HardwareConstrainedSqueezedState
from mqc3.feedforward import FeedForward

from cvzx.lowering.bridges.mqc3 import from_circuit_repr
from cvzx.backends.nx.graph import to_diagram, to_graph
from cvzx.passes.optimize import optimize

if TYPE_CHECKING:
    from mqc3.circuit.ops._base import MeasuredVariable


def _all_gates_circuit() -> CircuitRepr:
    """A circuit touching every supported intrinsic gate at least once.

    Modes 0 and 2 are operated on by `ControlledZ` while mode 1 sits
    between them in row order, so this also exercises the non-adjacent-
    row `connectivity` path in `_naive_translate` (not just the simple
    same-row-neighbors case).
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


def test_all_gates_circuit_normalizes_to_well_formed_diagram():
    circuit = _all_gates_circuit()
    diagram = from_circuit_repr(circuit)

    # Mode 2 was measured (consumed); modes 0 and 1 survive as outputs.
    assert diagram.num_inputs == 0
    assert diagram.num_outputs == 2

    # The result must be a genuine, well-formed diagram: it must survive
    # a to_graph/to_diagram round trip and optimize() without error.
    graph = to_graph(diagram)
    to_diagram(graph)
    optimize(diagram)


def test_naive_translation_matches_normalized_arity():
    circuit = _all_gates_circuit()
    raw = from_circuit_repr(circuit, normalize=False)
    normalized = from_circuit_repr(circuit)

    assert raw.num_inputs == normalized.num_inputs == 0
    assert raw.num_outputs == normalized.num_outputs == 2


def test_odd_multiple_of_pi_over_2_phase_rotation_does_not_raise():
    """Route around `PhaseRotationGate`'s odd-multiple-of-pi/2 restriction.

    `intrinsic.Squeezing` and `PhaseRotation(+-pi/2)` both force cvzx's
    `PhaseRotationGate` to an angle it refuses directly (an odd multiple
    of pi/2); the converter must route around this via Fourier/FourierInv
    (see `_phase_rotation`) rather than propagating the ValueError.
    """
    for phi in (np.pi / 2, -np.pi / 2, 3 * np.pi / 2):
        circuit = CircuitRepr("edge")
        circuit.Q(0) | intrinsic.PhaseRotation(phi)
        diagram = from_circuit_repr(circuit)
        assert diagram.num_outputs == 1

    circuit = CircuitRepr("squeezing")
    circuit.Q(0) | intrinsic.Squeezing(0.4)
    diagram = from_circuit_repr(circuit)
    assert diagram.num_outputs == 1


def test_beam_splitter_edge_angles_do_not_raise():
    """Exercise the same odd-multiple-of-pi/2 restriction via `BeamSplitter`.

    `intrinsic.BeamSplitter`'s own translation introduces a fixed
    `-pi/2` rotation plus a `pi/2 - theta_rel` rotation, so `theta_rel`
    values that are themselves multiples of pi/2 are exactly the cases
    most likely to hit the same odd-multiple-of-pi/2 restriction.
    """
    for theta_rel in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        circuit = CircuitRepr("bs_edge")
        circuit.Q(0, 1) | intrinsic.BeamSplitter(0.5, theta_rel)
        diagram = from_circuit_repr(circuit)
        assert diagram.num_outputs == 2


def test_hardware_constrained_squeezed_state_nonzero_phi():
    circuit = CircuitRepr("squeezed_init")
    circuit.Q(0)
    circuit.set_initial_state(0, HardwareConstrainedSqueezedState(phi=0.37))
    circuit.Q(0) | intrinsic.PhaseRotation(0.2)
    circuit.Q(0) | intrinsic.Measurement(0.5)
    diagram = from_circuit_repr(circuit)
    assert diagram.num_outputs == 0


def test_bosonic_state_single_peak_squeezed_is_supported():
    circuit = CircuitRepr("bosonic_squeezed")
    circuit.Q(0)
    circuit.set_initial_state(0, BosonicState.squeezed(r=0.4, phi=0.6))
    circuit.Q(0) | intrinsic.Measurement(0.0)
    diagram = from_circuit_repr(circuit)
    assert diagram.num_outputs == 0


def test_bosonic_state_multi_peak_raises():
    circuit = CircuitRepr("bosonic_multi_peak")
    circuit.Q(0) | intrinsic.PhaseRotation(0.1)
    circuit.set_initial_state(
        0,
        BosonicState(np.array([0.5, 0.5]), [GaussianState.vacuum(), GaussianState.vacuum()]),
    )
    with pytest.raises(NotImplementedError, match="multi-peak"):
        from_circuit_repr(circuit)


def test_manual_gate_raises():
    circuit = CircuitRepr("manual")
    circuit.Q(0, 1) | intrinsic.Manual(0.0, 0.1, 0.2, 0.3)
    with pytest.raises(NotImplementedError, match=r"intrinsic\.manual"):
        from_circuit_repr(circuit)


def test_std_beam_splitter_lowers_to_manual_and_raises():
    """Reject `std.BeamSplitter` the same way a direct `Manual` op is rejected.

    `std.BeamSplitter` lowers to `intrinsic.manual` under
    `convert_std_ops_to_intrinsic()`, so it must raise the same way a
    direct `intrinsic.Manual` operation does, not silently mistranslate.
    """
    circuit = CircuitRepr("std_bs")
    circuit.Q(0, 1) | std.BeamSplitter(0.0, 1.0)
    with pytest.raises(NotImplementedError, match=r"intrinsic\.manual"):
        from_circuit_repr(circuit)


def test_feedforward_parameter_reconstructs_measurement_symbol():
    """A `FeedForward`-driven parameter reconstructs its source measurement symbolically.

    `intrinsic.Measurement(0.0)` (p-homodyne) feeding forward into a
    `PhaseRotation` is reconstructed as a `PSpider(1, 0, ZxPoly({1: -m}))`
    effect (not a plain `MeasurementGate`) for a fresh symbol `m`, and the
    downstream `PhaseRotationGate` carries that same symbol.
    """
    from cvzx.ir.base import PSpider  # ruff: ignore[import-outside-top-level]
    from cvzx.ir.gates import PhaseRotationGate  # ruff: ignore[import-outside-top-level]

    circuit = CircuitRepr("feedforward")
    measured: MeasuredVariable = circuit.Q(0) | intrinsic.Measurement(0.0)
    circuit.Q(1) | intrinsic.PhaseRotation(FeedForward(measured))

    diagram = from_circuit_repr(circuit)

    leaves = []

    def _collect(d):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
        if hasattr(d, "diagrams"):
            for child in d.diagrams:
                _collect(child)
        else:
            leaves.append(d)

    _collect(diagram)

    p_effects = [leaf for leaf in leaves if isinstance(leaf, PSpider) and leaf.num_inputs == 1 and leaf.num_outputs == 0]
    rotations = [leaf for leaf in leaves if isinstance(leaf, PhaseRotationGate)]
    assert len(p_effects) == 1
    assert len(rotations) == 1
    (symbol,) = p_effects[0].phase.get_parameters()
    assert rotations[0].parametric
    assert rotations[0].theta.free_symbols == {symbol}


def test_empty_circuit_raises():
    circuit = CircuitRepr("empty")
    with pytest.raises(ValueError, match="empty"):
        from_circuit_repr(circuit)


def test_circuit_is_not_mutated():
    """`from_circuit_repr` must not mutate the caller's circuit.

    It lowers `std.*` -> `intrinsic.*` on an internal deep copy.
    """
    circuit = CircuitRepr("not_mutated")
    circuit.Q(0, 1) | std.BeamSplitter(0.0, 1.0)
    n_ops_before = circuit.n_operations
    op_name_before = circuit.get_operation(0).name()

    with pytest.raises(NotImplementedError):
        from_circuit_repr(circuit)

    assert circuit.n_operations == n_ops_before
    assert circuit.get_operation(0).name() == op_name_before

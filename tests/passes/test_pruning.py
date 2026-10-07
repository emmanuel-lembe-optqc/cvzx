"""prune_small_gaussian_gates: near-identity Gaussian gates become identity wires, at the compact-gate level."""

import math

import pytest
from sympy import Symbol

from cvzx.ir.base import CompositionDiagram, QSpider, TensorDiagram, ZxPoly
from cvzx.ir.gates import (
    ArbitraryGate,
    BeamsplitterGate,
    ControlledZGate,
    DisplacementGate,
    MeasurementGate,
    PhaseRotationGate,
    SqueezingGate,
    TwoModeShearGate,
)
from cvzx.passes.optimize import optimize
from cvzx.passes.pruning import gate_distance, prune_small_gaussian_gates


def _nodes(diagram) -> int:
    """Node count of the fully cleaned graph `optimize` returns."""
    return optimize(diagram).graph.graph.num_nodes()


def _state() -> QSpider:
    return QSpider(0, 1, ZxPoly({}))


def _one_mode(*gates):
    return CompositionDiagram([_state(), *gates, MeasurementGate(0.3)])


def _two_mode(*gates):
    return CompositionDiagram([
        TensorDiagram([_state(), _state()]),
        *gates,
        TensorDiagram([MeasurementGate(0.3), MeasurementGate(1.1)]),
    ])


def _gates(diagram):
    out = []

    def walk(d):
        if isinstance(d, (CompositionDiagram, TensorDiagram)):
            for c in d.diagrams:
                walk(c)
        elif gate_distance(d) is not None:
            out.append(d)

    walk(diagram)
    return out


def test_small_rotation_is_pruned_and_fuses_away():
    d = _one_mode(SqueezingGate(1.5), PhaseRotationGate(1e-4), SqueezingGate(2.0))
    res = prune_small_gaussian_gates(d, 1e-3)
    assert [p.kind for p in res.pruned] == ["rotation"]
    assert [type(g).__name__ for g in _gates(res.diagram)] == ["SqueezingGate", "SqueezingGate"]
    assert _nodes(res.diagram) < _nodes(d)


@pytest.mark.parametrize("gate", [BeamsplitterGate(1e-4), ControlledZGate(1e-4), TwoModeShearGate(1e-4, -2e-4)])
def test_two_mode_gates_prune_without_dividing_by_zero(gate):
    res = prune_small_gaussian_gates(_two_mode(gate), 1e-3)
    assert len(res.pruned) == 1 and not _gates(res.diagram)
    optimize(res.diagram)  # expands and simplifies without a singular parameter


def test_squeezing_identity_is_tau_one():
    assert prune_small_gaussian_gates(_one_mode(SqueezingGate(1 + 1e-4)), 1e-3).modified
    assert not prune_small_gaussian_gates(_one_mode(SqueezingGate(1.5)), 1e-3).modified
    assert gate_distance(SqueezingGate(-1.0)) == math.inf


def test_rotation_distance_wraps_and_arbitrary_needs_both():
    assert gate_distance(PhaseRotationGate(2 * math.pi - 1e-5)) < 1e-4
    assert gate_distance(ArbitraryGate(0.4, -0.4, 1e-5)) < 1e-4
    assert gate_distance(ArbitraryGate(0.4, 0.4, 0.0)) > 0.5


def test_epsilon_zero_changes_nothing():
    d = _two_mode(BeamsplitterGate(1e-6), ControlledZGate(1e-6))
    res = prune_small_gaussian_gates(d, 0.0)
    assert not res.modified and res.diagram is d


def test_per_type_thresholds_prune_only_the_named_types():
    d = _two_mode(BeamsplitterGate(1e-4), ControlledZGate(1e-4))
    res = prune_small_gaussian_gates(d, {"beam_splitter": 1e-3})
    assert [p.kind for p in res.pruned] == ["beam_splitter"]
    assert [type(g).__name__ for g in _gates(res.diagram)] == ["ControlledZGate"]


def test_keep_vetoes_and_infinite_epsilon_lets_keep_decide():
    d = _two_mode(BeamsplitterGate(1e-4), ControlledZGate(0.8))
    kept = prune_small_gaussian_gates(d, 1e-3, keep=lambda g: True)
    assert not kept.modified
    noise_aware = prune_small_gaussian_gates(d, math.inf, keep=lambda g: not isinstance(g, ControlledZGate))
    assert [p.kind for p in noise_aware.pruned] == ["controlled_z"]


def test_feedforward_targets_and_symbolic_gates_are_never_pruned():
    m = Symbol("m", real=True)
    ff = DisplacementGate(1e-6 * m, parametric=True, param_measurement_map={m: {1}})
    assert gate_distance(ff) is None
    assert not prune_small_gaussian_gates(_one_mode(ff), math.inf).modified
    symbolic = PhaseRotationGate(Symbol("t", real=True), parametric=True)
    assert gate_distance(symbolic) is None


def test_optimize_option():
    d = _one_mode(SqueezingGate(1.5), PhaseRotationGate(1e-4), SqueezingGate(2.0))
    plain = optimize(d).graph.graph.num_nodes()
    pruned = optimize(d, prune_epsilon=1e-3).graph.graph.num_nodes()
    assert pruned < plain

"""Unit tests for `cvzx.utils.metrics`.

Several of these tests exist specifically because the naive version of
these metrics was wrong in ways only visible by actually running
`optimize()` and inspecting its output: `to_graph()` deletes every node's
`"diagram"` attribute before returning (see `count_non_clifford_phases`'s
docstring), and `OptimizeResult.diagram` still contains `normalize_diagram`'s
identity-wire filler and `CopyRule`'s `VoidDiagram` placeholders (see
`is_bookkeeping_leaf`'s docstring) -- both would otherwise make a genuinely
simplified diagram look unchanged or larger.
"""

import unittest

from cvzx.backend import get_backend_modules
from cvzx.ir.base import CompositionDiagram, ContractedDiagram, QSpider, TensorDiagram, ZxPoly
from cvzx.ir.gates import ArbitraryGate, ControlledSumGate, CubicPhaseGate, PhaseRotationGate
from cvzx.passes.optimize import optimize
from cvzx.utils.metrics import (
    compare_metrics,
    compute_metrics,
    count_edges,
    count_gates,
    count_generators,
    count_non_clifford_phases,
    count_nodes,
    count_spiders,
    diagram_depth,
)

_, _graph_mod, _ = get_backend_modules()
to_graph = _graph_mod.to_graph


def _build_four_mode_circuit():
    """A small 4-mode circuit: one ancilla-consuming CSUM, one bare CSUM, and two mergeable CSUMs.

    Deliberately mirrors `tests/passes/test_optimize.py::_build_four_mode_circuit`
    (same shape, reconstructed locally so this file doesn't depend on that
    module's test-collection layout): under `assume_infinite_squeezing=False`
    only `ChainReductionRule` fires (merging the two gain=0.5 CSUMs into one
    gain=1.0 CSUM), a small, precisely-known reduction to assert against.

    Returns
    -------
    CompositionDiagram
    """
    zero = ZxPoly({})
    identity = lambda: QSpider(1, 1, zero)  # ruff: ignore[lambda-assignment]
    ancilla = QSpider(0, 1, ZxPoly({1: 2}))
    csum_a = ControlledSumGate(control=2, target=1)
    csum_b = ControlledSumGate(control=2, target=1)
    csum_c1 = ControlledSumGate(gain=0.5, control=2, target=1)
    csum_c2 = ControlledSumGate(gain=0.5, control=2, target=1)

    seg_a = CompositionDiagram([TensorDiagram([ancilla, identity()]), csum_a])
    layer1 = TensorDiagram([seg_a, identity(), identity()])
    layer2 = TensorDiagram([identity(), csum_b, identity()])
    seg_c = CompositionDiagram([csum_c1, csum_c2])
    layer3 = TensorDiagram([identity(), identity(), seg_c])
    return CompositionDiagram([layer1, layer2, layer3])


class TestCountSpiders(unittest.TestCase):
    """`count_spiders` counts QSpider/PSpider leaves, excluding identity wires."""

    def test_excludes_identity_wires(self):
        """A bare identity spider (zero phase, 1-in-1-out) is not counted."""
        diagram = CompositionDiagram([QSpider(1, 1, ZxPoly({1: 1})), QSpider(1, 1, ZxPoly({}))])
        graph = to_graph(diagram)
        assert count_spiders(graph) == 1

    def test_counts_ancilla_states(self):
        """A 0-in/1-out state spider with nonzero phase is real content, not a wire."""
        diagram = QSpider(0, 1, ZxPoly({1: 2}))
        graph = to_graph(diagram)
        assert count_spiders(graph) == 1


class TestCountGatesAndGenerators(unittest.TestCase):
    """`count_gates`/`count_generators` on a mix of gates, spiders, and bookkeeping."""

    def test_count_gates_counts_compact_leaves(self):
        """Two un-expanded gates plus a spider: only the gates count as gates."""
        diagram = CompositionDiagram([PhaseRotationGate(0.3), CubicPhaseGate(0.1)])
        graph = to_graph(diagram)
        assert count_gates(graph) == 2

    def test_count_generators_excludes_void_diagram(self):
        """A `VoidDiagram` placeholder is bookkeeping, not a surviving generator."""
        from cvzx.ir.base import VoidDiagram  # ruff: ignore[import-outside-top-level]

        diagram = CompositionDiagram([PhaseRotationGate(0.3), VoidDiagram(1, 1)])
        graph = to_graph(diagram)
        assert count_generators(graph) == 1


class TestCountNonCliffordPhases(unittest.TestCase):
    """`count_non_clifford_phases`: the CV-ZX T-count analogue."""

    def test_cubic_phase_gate_counts(self):
        """CubicPhaseGate (degree-3, non-Gaussian) counts at the default threshold."""
        diagram = CompositionDiagram([CubicPhaseGate(0.3), PhaseRotationGate(0.5)])
        graph = to_graph(diagram)
        assert count_non_clifford_phases(graph) == 1

    def test_arbitrary_gate_does_not_count(self):
        """ArbitraryGate is Gaussian (rotation-squeeze-rotation): never non-Clifford."""
        diagram = ArbitraryGate(0.1, 0.2, 0.3)
        graph = to_graph(diagram)
        assert count_non_clifford_phases(graph) == 0

    def test_degree_3_spider_counts_directly(self):
        """A bare QSpider with an explicit degree-3 phase also counts."""
        diagram = QSpider(1, 1, ZxPoly({3: 1.0}))
        graph = to_graph(diagram)
        assert count_non_clifford_phases(graph) == 1

    def test_degree_threshold_is_respected(self):
        """Raising the threshold above CubicPhaseGate's degree excludes it."""
        diagram = CubicPhaseGate(0.3)
        graph = to_graph(diagram)
        assert count_non_clifford_phases(graph, degree_threshold=4) == 0
        assert count_non_clifford_phases(graph, degree_threshold=2) == 1


class TestDiagramDepth(unittest.TestCase):
    """`diagram_depth`: stage count via `normalize_diagram`."""

    def test_contracted_diagram_is_none(self):
        """A diagram containing a ContractedDiagram has no defined depth."""
        a = QSpider(1, 2, ZxPoly({1: 1}))
        b = QSpider(2, 1, ZxPoly({1: 1}))
        contracted = ContractedDiagram(a, b, I1=[1], I2=[0], J1=[], J2=[])
        assert diagram_depth(contracted) is None

    def test_two_stage_diagram(self):
        """One narrow (1-mode) stage followed by one wide (2-mode gate) stage."""
        diagram = CompositionDiagram(
            [TensorDiagram([PhaseRotationGate(0.3), QSpider(1, 1, ZxPoly({}))]), ControlledSumGate(control=2, target=1)]
        )
        assert diagram_depth(diagram) == 2


class TestCountNodesAndEdges(unittest.TestCase):
    """`count_nodes`/`count_edges`: raw totals, parity with `CVZXGraph.__repr__`."""

    def test_matches_repr(self):
        """The counts match the numbers `CVZXGraph.__repr__` reports."""
        diagram = CompositionDiagram([PhaseRotationGate(0.3), CubicPhaseGate(0.1)])
        graph = to_graph(diagram)
        assert repr(graph) == f"CVZXGraph(nodes={count_nodes(graph)}, edges={count_edges(graph)})"


class TestComputeAndCompareMetrics(unittest.TestCase):
    """`compute_metrics`/`compare_metrics`: end-to-end, against real `optimize()` output."""

    def test_exact_only_optimize_reduces_gate_count(self):
        """Under assume_infinite_squeezing=False, only the two mergeable CSUMs combine.

        Regression test for the bug this module was built to avoid: before
        `count_generators`/`count_spiders` excluded bookkeeping leaves, this
        diagram's post-`optimize()` generator count came out *higher* than
        before, even though exactly one real gate was eliminated.
        """
        diagram = _build_four_mode_circuit()
        result = optimize(diagram, assume_infinite_squeezing=False)
        comparison = compare_metrics(diagram, result.diagram)

        assert comparison.before.gates == 4
        assert comparison.after.gates == 3
        assert comparison.reduction("gates") == 0.25
        # The ancilla-consuming CSUM survives untouched under exact-only
        # simplification (CopyRule needs assume_infinite_squeezing=True).
        assert comparison.before.spiders == comparison.after.spiders == 1

    def test_ratio_and_reduction_guard_against_none_and_zero(self):
        """`ratio`/`reduction` return `None` rather than raising on edge cases."""
        a = QSpider(1, 2, ZxPoly({1: 1}))
        b = QSpider(2, 1, ZxPoly({1: 1}))
        contracted = ContractedDiagram(a, b, I1=[1], I2=[0], J1=[], J2=[])
        comparison = compare_metrics(contracted, contracted)
        assert comparison.ratio("depth") is None
        assert comparison.reduction("depth") is None

        empty_before = compute_metrics(QSpider(1, 1, ZxPoly({})))
        assert empty_before.non_clifford_phases == 0
        comparison_zero = compare_metrics(QSpider(1, 1, ZxPoly({})), CubicPhaseGate(0.1))
        assert comparison_zero.ratio("non_clifford_phases") is None

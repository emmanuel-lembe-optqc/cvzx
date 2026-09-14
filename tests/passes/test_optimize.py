"""Unit tests for the `cvzx.passes.optimize` pipeline.

These tests showcase the power of `optimize()`: what it can simplify with
`assume_infinite_squeezing=False` (only exact identities: chain reduction,
Fourier normalization, spider fusion, and rotation absorption) versus what
extra reductions `assume_infinite_squeezing=True` additionally unlocks
(squeezing absorption, cross-color discard, and BS/CSUM expansion followed
by `CopyRule`) -- contrasting the same diagrams under both settings wherever
that contrast is the point.

Most tests below inspect the result via `optimize(...).graph` (converted
back with `to_diagram()`) rather than `.diagram` out of habit -- the two are
equivalent (see `TestOptimizeReturnType`) -- and use `assume_infinite_squeezing=False`
inputs or otherwise avoid `CopyRule`/`TerminalAbsorptionRule` cross-container
substitutions where a leftover `VoidDiagram` residue would break the
`isinstance`/structural assertions these tests make.
"""

import math
import unittest
from typing import cast

from sympy import pi, simplify

from cvzx.backend import get_backend_modules
from cvzx.ir.base import (
    CompositionDiagram,
    ContractedDiagram,
    Diagram,
    Fourier,
    Fourier2,
    PSpider,
    QSpider,
    TensorDiagram,
    ZxPoly,
)
from cvzx.ir.gates import BeamsplitterGate, ControlledSumGate, PhaseRotationGate, SqueezingGate
from cvzx.passes.optimize import optimize

# `optimize()` picks its own backend (see `cvzx.config.DEFAULT_BACKEND`) when
# no `backend=` is passed, as every call in this file does -- so the graph
# it hands back, and the rule classes used to double-check it's a genuine
# fixed point, must come from that SAME backend's modules rather than being
# hardcoded to one, or `to_diagram`/`CVZXGraph` below would choke on a graph
# object from the other backend.
_, _graph_mod, _rules_mod = get_backend_modules()
CVZXGraph = _graph_mod.CVZXGraph
to_diagram = _graph_mod.to_diagram
ChainReductionRule = _rules_mod.ChainReductionRule
CopyRule = _rules_mod.CopyRule
FourierNormalizationRule = _rules_mod.FourierNormalizationRule
FusionRule = _rules_mod.FusionRule
IdentityRule = _rules_mod.IdentityRule
TerminalAbsorptionRule = _rules_mod.TerminalAbsorptionRule


def _no_matches_left(graph, *, assume_infinite_squeezing):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    """True if none of the pipeline's rules can still fire on `graph`.

    Used to confirm a result is a genuine fixed point, not just "some
    reduction happened".
    """
    rules = [
        IdentityRule(),
        FusionRule(),
        ChainReductionRule(),
        FourierNormalizationRule(),
        TerminalAbsorptionRule(assume_infinite_squeezing=assume_infinite_squeezing),
    ]
    if assume_infinite_squeezing:
        rules.append(CopyRule())
    graph.rebuild_registry()
    return all(len(rule.match(graph)) == 0 for rule in rules)


def _iter_node_attrs(cvzx_graph):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    """Yield every node's attribute payload, independent of graph backend.

    Returns
    -------
    Iterator[dict]
    """
    raw = cvzx_graph.graph
    if hasattr(raw, "node_indices"):  # rustworkx PyDiGraph
        return (raw[idx] for idx in raw.node_indices())
    return (attrs for _, attrs in raw.nodes(data=True))  # networkx DiGraph


def _count_node_type(graph, type_name):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    """Count graph nodes whose `type` attribute equals `type_name`."""
    return sum(1 for attrs in _iter_node_attrs(graph) if attrs.get("type") == type_name)


def _build_four_mode_circuit():  # ruff: ignore[missing-return-type-private-function]
    """A 4-qumode circuit with three ControlledSumGate instances, two reducible.

    Layer by layer (3 real circuit inputs; mode 0 is produced internally by
    an ancilla and only appears from layer 1 onward, so the circuit as a
    whole maps 3 inputs to 4 outputs):

    - Layer 1: an ancilla state feeds `csum_a` on modes (0, 1), tensored
      with identities on modes 2, 3. With `assume_infinite_squeezing=True`,
      the ancilla is an eigenstate CopyRule can push straight through
      `csum_a`'s control leg, eliminating it entirely -- the textbook
      "prepare an ancilla, consume it with a CSUM" pattern.
    - Layer 2: `csum_b` on modes (1, 2), with nothing feeding it but plain
      wires -- there's no state for CopyRule to act on, so this gate is
      never eliminated (it's the "not everything gets reduced" control).
    - Layer 3: two CSUM gates `csum_c1`, `csum_c2` (gain=0.5 each) back to
      back on the same modes (2, 3) -- ChainReductionRule combines these
      into one gain=1.0 CSUM regardless of `assume_infinite_squeezing`,
      since it operates on the still-compact gate nodes directly.
    - Layer 4: a PhaseRotationGate and a SqueezingGate on the two
      untouched modes, left alone by every rule here -- present just to
      keep the circuit from being trivially only-CSUM-gates.

    Returns
    -------
    CompositionDiagram
        A fresh 4-mode circuit (3 inputs, 4 outputs) built from new gate
        instances each call, safe to optimize repeatedly without aliasing.
    """
    zero = ZxPoly({})
    identity = lambda: QSpider(1, 1, zero)  # ruff: ignore[lambda-assignment]
    ancilla = PSpider(0, 1, ZxPoly({1: 2}))
    csum_a = ControlledSumGate(control=2, target=1)
    csum_b = ControlledSumGate(control=2, target=1)
    csum_c1 = ControlledSumGate(gain=0.5, control=2, target=1)
    csum_c2 = ControlledSumGate(gain=0.5, control=2, target=1)

    seg_a = CompositionDiagram([TensorDiagram([ancilla, identity()]), csum_a])
    layer1 = TensorDiagram([seg_a, identity(), identity()])
    layer2 = TensorDiagram([identity(), csum_b, identity()])
    seg_c = CompositionDiagram([csum_c1, csum_c2])
    layer3 = TensorDiagram([identity(), identity(), seg_c])
    layer4 = TensorDiagram([
        PhaseRotationGate(pi / 5),
        identity(),
        identity(),
        SqueezingGate(tau=1.5),
    ])
    return CompositionDiagram([layer1, layer2, layer3, layer4])


class TestOptimizeReturnType(unittest.TestCase):
    """optimize()'s return contract: always an `OptimizeResult(graph, diagram)`."""

    def setUp(self):
        """Create a simple reducible diagram shared by both tests."""
        self.comp = CompositionDiagram([Fourier(), Fourier(), Fourier(), Fourier()])

    def test_returns_graph_and_diagram(self):
        """optimize() always returns both the cleaned graph and a Diagram."""
        graph, diagram = optimize(self.comp)
        assert isinstance(graph, CVZXGraph)
        assert isinstance(diagram, Diagram)
        # Also reachable by attribute, not just tuple unpacking.
        result = optimize(self.comp)
        assert result.graph is not None
        assert isinstance(result.diagram, Diagram)

    def test_diagram_matches_graph_when_cleanup_is_a_no_op(self):
        """The two agree exactly when there's no VoidDiagram/identity residue to clean.

        This simple Fourier chain never invokes CopyRule, so the
        end-of-pipeline cleanup pass has nothing to do, and the pre-cleanup
        `.diagram` and `to_diagram(.graph)` describe the same state.
        """
        graph, diagram = optimize(self.comp)
        assert diagram == to_diagram(graph)

    def test_diagram_agrees_with_graph_when_void_nodes_are_present(self):
        """`.diagram` always agrees with `to_diagram(.graph)`, Void residue included.

        A control state tensor-composed with CSUM drives CopyRule, which
        leaves `VoidDiagram` placeholders behind (see `_install_void_placeholder`);
        `optimize()` no longer cleans these up, so `.diagram` is simply
        `to_diagram(.graph)` -- both still carry the same Void residue.
        """
        control_state = PSpider(0, 1, ZxPoly({1: 2}))
        target_state = QSpider(0, 1, ZxPoly({1: 3}))
        csum = ControlledSumGate(control=2, target=1)
        comp = CompositionDiagram([TensorDiagram([control_state, target_state]), csum])

        graph, diagram = optimize(comp, assume_infinite_squeezing=True)
        cleaned = to_diagram(graph)

        assert diagram == cleaned
        assert diagram.num_inputs == cleaned.num_inputs == 0
        assert diagram.num_outputs == cleaned.num_outputs == 2


class TestOptimizeExactOnly(unittest.TestCase):
    """Power of the pipeline with `assume_infinite_squeezing=False` (the default).

    Only rules that are exact for any physical state run: IdentityRule,
    FusionRule, ChainReductionRule, FourierNormalizationRule, and rotation
    absorption within TerminalAbsorptionRule.
    """

    def test_chain_reduction_combines_three_rotations(self):
        """R(t1) . R(t2) . R(t3) collapses into a single R(t1+t2+t3)."""
        theta1, theta2, theta3 = pi / 6, pi / 5, pi / 7
        comp = CompositionDiagram([
            PhaseRotationGate(theta1),
            PhaseRotationGate(theta2),
            PhaseRotationGate(theta3),
        ])
        graph, _ = optimize(comp)
        result = to_diagram(graph)
        assert isinstance(result, PhaseRotationGate)
        assert simplify(result.theta - (theta1 + theta2 + theta3)) == 0
        assert _no_matches_left(graph, assume_infinite_squeezing=False)

    def test_fourier_quadruple_collapses_to_identity(self):
        """F . F . F . F reduces all the way down to the identity spider.

        Regression test: this used to crash with `KeyError: None` in
        `IdentityRule` once the whole diagram collapsed to a single
        identity spider sitting at the graph's root (container_id=None).
        """
        comp = CompositionDiagram([Fourier(), Fourier(), Fourier(), Fourier()])
        graph, _ = optimize(comp)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.num_inputs == 1
        assert result.num_outputs == 1
        assert result.phase.is_zero

    def test_full_exact_pipeline_absorbs_fourier2_and_rotation_into_state(self):
        """A state . F2 . R(theta) chain fully collapses to a single terminal.

        Exercises FourierNormalizationRule (folding F2 into R), then
        TerminalAbsorptionRule (folding the combined rotation into the
        state) -- leaving no gates at all, only the terminal.
        """
        phase_state = ZxPoly({0: 1, 1: 2})
        comp = CompositionDiagram([QSpider(0, 1, phase_state), Fourier2(), PhaseRotationGate(pi / 6)])
        graph, _ = optimize(comp)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.num_inputs == 0
        assert result.num_outputs == 1
        assert _no_matches_left(graph, assume_infinite_squeezing=False)

    def test_fusion_combines_same_color_spiders_in_contracted_diagram(self):
        """Two same-color QSpiders wired together in a ContractedDiagram fuse into one."""
        phase_a = ZxPoly({1: 2})
        phase_b = ZxPoly({1: 5})
        contracted = ContractedDiagram(QSpider(1, 1, phase_a), QSpider(1, 1, phase_b), [0], [0], [], [])
        graph, _ = optimize(contracted)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.num_inputs == 1
        assert result.num_outputs == 1
        assert result.phase == phase_a + phase_b

    def test_chain_reduction_combines_two_adjacent_csum_gates(self):
        """Two adjacent same control/target CSUM gates combine their gains.

        This works even with `assume_infinite_squeezing=False`, since
        ChainReductionRule operates on the still-compact gate nodes
        directly -- no expansion needed.
        """
        csum_a = ControlledSumGate(gain=0.3, control=2, target=1)
        csum_b = ControlledSumGate(gain=0.7, control=2, target=1)
        comp = CompositionDiagram([csum_a, csum_b])
        graph, _ = optimize(comp)
        result = to_diagram(graph)
        assert isinstance(result, ControlledSumGate)
        assert math.isclose(cast("float", result.gain), 1.0)
        assert result.control == 2
        assert result.target == 1

    def test_beamsplitter_untouched_without_squeezing(self):
        """A bare BeamsplitterGate is never expanded when the flag is off."""
        bs = BeamsplitterGate(pi / 4)
        graph, _ = optimize(bs)
        result = to_diagram(graph)
        assert isinstance(result, BeamsplitterGate)

    def test_controlled_sum_gate_untouched_without_squeezing(self):
        """A CSUM+states diagram keeps its CSUM gate intact without the flag.

        Contrast with `TestOptimizeInfiniteSqueezing
        .test_csum_cross_container_full_reduction`, where the identical
        diagram fully reduces once `assume_infinite_squeezing=True`.
        """
        control_state = PSpider(0, 1, ZxPoly({1: 2}))
        target_state = QSpider(0, 1, ZxPoly({1: 3}))
        csum = ControlledSumGate(control=2, target=1)
        comp = CompositionDiagram([TensorDiagram([control_state, target_state]), csum])
        graph, _ = optimize(comp)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert isinstance(result.diagrams[1], ControlledSumGate)

    def test_squeezing_gate_untouched_without_squeezing(self):
        """A state . SqueezingGate pair is left alone without the flag.

        Contrast with `TestOptimizeInfiniteSqueezing
        .test_squeezing_absorption_folds_tau_into_terminal_phase`.
        """
        state = QSpider(0, 1, ZxPoly({0: 1, 1: 3, 2: 2}))
        comp = CompositionDiagram([state, SqueezingGate(tau=2.0)])
        graph, _ = optimize(comp)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert isinstance(result.diagrams[1], SqueezingGate)
        assert isinstance(result.diagrams[0], QSpider)
        assert result.diagrams[0].phase == state.phase

    def test_cross_color_spider_untouched_without_squeezing(self):
        """A state next to an opposite-color (1,1) spider isn't discarded.

        Contrast with `TestOptimizeInfiniteSqueezing
        .test_cross_color_discard_leaves_terminal_phase_unchanged`.
        """
        state = QSpider(0, 1, ZxPoly({1: 7}))
        comp = CompositionDiagram([state, PSpider(1, 1, ZxPoly({2: 4, 3: 9}))])
        graph, _ = optimize(comp)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2


class TestOptimizeInfiniteSqueezing(unittest.TestCase):
    """Power of the pipeline with `assume_infinite_squeezing=True`.

    Additionally runs squeezing absorption and cross-color discard within
    TerminalAbsorptionRule, expands BeamsplitterGate/ControlledSumGate, and
    runs CopyRule -- all only exact for idealized (infinite squeezing)
    eigenstates.
    """

    def test_squeezing_absorption_folds_tau_into_terminal_phase(self):
        """State . Sq(tau) folds into a single terminal with phase f(tau*x).

        Matches `TerminalAbsorptionRule`'s own established, independently
        tested convention for a `QSpider` state (see e.g.
        `test_apply_single_squeezing_state` in
        `test_nx_terminal_absorption_rule.py`, which folds `Sq(2.0)` into
        `QSpider(0, 1, 3.0*x)` and expects `6.0*x` -- multiplied by tau,
        not divided).
        """
        tau = 2.0
        state = QSpider(0, 1, ZxPoly({0: 1, 1: 3, 2: 2}))
        comp = CompositionDiagram([state, SqueezingGate(tau=tau)])
        graph, _ = optimize(comp, assume_infinite_squeezing=True)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        expected_phase = ZxPoly({degree: coeff * tau**degree for degree, coeff in state.phase.coeffs.items()})
        assert result.phase == expected_phase
        assert _no_matches_left(graph, assume_infinite_squeezing=True)

    def test_cross_color_discard_leaves_terminal_phase_unchanged(self):
        """State next to an opposite-color (1,1) spider: the spider vanishes untouched."""
        state = QSpider(0, 1, ZxPoly({1: 7}))
        comp = CompositionDiagram([state, PSpider(1, 1, ZxPoly({2: 4, 3: 9}))])
        graph, _ = optimize(comp, assume_infinite_squeezing=True)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.num_inputs == 0
        assert result.num_outputs == 1
        assert result.phase == state.phase

    def test_csum_cross_container_full_reduction(self):
        """A control state tensor-composed with CSUM gets copied inside it, fully.

        The opposite-color control state is duplicated onto both of the
        (now-expanded) CSUM's copy-spider legs; the same-color target state
        is left alone (that's ordinary fusion territory, not CopyRule's
        job). No ControlledSumGate, and no further matches, remain.
        """
        control_state = PSpider(0, 1, ZxPoly({1: 2}))
        target_state = QSpider(0, 1, ZxPoly({1: 3}))
        csum = ControlledSumGate(control=2, target=1)
        comp = CompositionDiagram([TensorDiagram([control_state, target_state]), csum])

        graph, _ = optimize(comp, assume_infinite_squeezing=True)
        result = to_diagram(graph)

        assert result.num_inputs == 0
        assert result.num_outputs == 2
        types_present = {attrs.get("type") for attrs in _iter_node_attrs(graph)}
        assert "ControlledSumGate" not in types_present
        assert _no_matches_left(graph, assume_infinite_squeezing=True)

    def test_beamsplitter_cross_container_full_reduction(self):
        """A control state tensor-composed with a balanced BS also fully reduces.

        BeamsplitterGate.expand() decomposes into TWO expanded CSUM gates
        with a tensor of squeezing gates in between; the opposite-color
        state feeding the first CSUM's copy spider still gets duplicated,
        and no BeamsplitterGate/ControlledSumGate compact node survives.
        """
        control_state = QSpider(0, 1, ZxPoly({1: 3}))
        other_state = QSpider(0, 1, ZxPoly({1: 2}))
        bs = BeamsplitterGate(pi / 4)
        comp = CompositionDiagram([TensorDiagram([control_state, other_state]), bs])

        graph, _ = optimize(comp, assume_infinite_squeezing=True)
        result = to_diagram(graph)

        assert result.num_inputs == 0
        assert result.num_outputs == 2
        types_present = {attrs.get("type") for attrs in _iter_node_attrs(graph)}
        assert "BeamsplitterGate" not in types_present
        assert "ControlledSumGate" not in types_present
        assert _no_matches_left(graph, assume_infinite_squeezing=True)

    def test_optimize_is_idempotent(self):
        """Re-optimizing an already-optimized diagram is a true no-op."""
        control_state = PSpider(0, 1, ZxPoly({1: 2}))
        target_state = QSpider(0, 1, ZxPoly({1: 3}))
        csum = ControlledSumGate(control=2, target=1)
        comp = CompositionDiagram([TensorDiagram([control_state, target_state]), csum])

        once_graph, _ = optimize(comp, assume_infinite_squeezing=True)
        once = to_diagram(once_graph)
        twice_graph, _ = optimize(once, assume_infinite_squeezing=True)
        twice = to_diagram(twice_graph)
        assert once == twice

    def test_expansion_not_committed_without_further_benefit(self):
        """A bare BS with nothing attached is never gratuitously expanded.

        Even with `assume_infinite_squeezing=True`, expanding BS/CSUM only
        happens if doing so unlocks at least one rule match; expanding a
        gate with no states/other structure to react with wouldn't
        simplify anything, so the round is discarded and the gate stays
        compact.
        """
        bs = BeamsplitterGate(pi / 4)
        graph, _ = optimize(bs, assume_infinite_squeezing=True)
        result = to_diagram(graph)
        assert isinstance(result, BeamsplitterGate)

        csum = ControlledSumGate(control=2, target=1)
        graph_csum, _ = optimize(csum, assume_infinite_squeezing=True)
        result_csum = to_diagram(graph_csum)
        assert isinstance(result_csum, ControlledSumGate)


class TestOptimizeRobustness(unittest.TestCase):
    """Safety-net behavior: optimize() never crashes or over-reduces."""

    def test_max_rounds_zero_returns_diagram_unchanged(self):
        """`max_rounds=0` runs no rounds at all -- the input comes back as-is."""
        control_state = PSpider(0, 1, ZxPoly({1: 2}))
        target_state = QSpider(0, 1, ZxPoly({1: 3}))
        csum = ControlledSumGate(control=2, target=1)
        comp = CompositionDiagram([TensorDiagram([control_state, target_state]), csum])

        _, diagram = optimize(comp, assume_infinite_squeezing=True, max_rounds=0)
        assert diagram == comp

    def test_unreducible_diagram_is_returned_unchanged(self):
        """A diagram with nothing any rule can act on comes back unchanged."""
        phi_any = ZxPoly({2: 2, 3: 1})
        untouched = CompositionDiagram([QSpider(1, 3, phi_any)])
        _, diagram = optimize(untouched)
        assert diagram == untouched

    def test_max_rounds_small_still_fully_reduces_simple_case(self):
        """A single expand-then-copy round suffices even with `max_rounds=1`.

        CopyRule runs in the SAME graph-level fixed-point pass as the
        expansion that exposes it, so a diagram that only needs one round
        of expansion reduces fully even under a tight round budget.
        """
        control_state = PSpider(0, 1, ZxPoly({1: 2}))
        target_state = QSpider(0, 1, ZxPoly({1: 3}))
        csum = ControlledSumGate(control=2, target=1)
        comp = CompositionDiagram([TensorDiagram([control_state, target_state]), csum])

        graph, _ = optimize(comp, assume_infinite_squeezing=True, max_rounds=1)
        result = to_diagram(graph)
        assert result.num_inputs == 0
        assert result.num_outputs == 2
        assert _no_matches_left(graph, assume_infinite_squeezing=True)

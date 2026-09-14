"""Unit tests for the terminal absorption rewrite rule using the graph formalism.

These tests verify the TerminalAbsorptionRule implementation on graphs: a gate
adjacent to a (1,0) effect or (0,1) state QSpider/PSpider terminal is folded
into the terminal's own phase. Three sub-cases: rotation (QSpider only,
degree <= 1 input), squeezing (either color, any degree), and cross-color
discard (opposite-color (1,1) raw spider vanishes, phase unchanged).

`TerminalAbsorptionRule.apply_single` updates the terminal's phase in place,
resets the absorbed gate to a zero-phase identity spider of its own color
and arity, and resets every identity/`Swap` passthrough the chase crossed
the same way. Nothing is contracted, spliced, or voided. Tests that assert
the fully-collapsed form drive `IdentityRule` themselves, via
`_absorb_then_identity`, after a `rebuild_registry()` -- `apply_single`
mutates node attrs without touching the registry, so `IdentityRule` would
otherwise read a stale index.
"""

import unittest
from math import cos, isclose, pi, tan

from sympy import Symbol, simplify
from sympy import cos as sym_cos
from sympy import tan as sym_tan

from cvzx.ir.base import (
    CompositionDiagram,
    ContractedDiagram,
    Fourier,
    Fourier2,
    FourierInv,
    PSpider,
    QSpider,
    Swap,
    TensorDiagram,
    ZxPoly,
)
from cvzx.ir.gates import BeamsplitterGate, DisplacementGate, PhaseRotationGate, SqueezingGate
from cvzx.backends.nx.graph import CVZXGraph, to_diagram, to_graph
from cvzx.backends.nx.rules import IdentityRule, TerminalAbsorptionRule


def _apply_absorption_and_cleanup(graph: CVZXGraph, rule: TerminalAbsorptionRule) -> None:
    """Apply `rule` then `IdentityRule`, in that order, in place.

    `TerminalAbsorptionRule.apply_single` resets the absorbed gate and any
    crossed passthroughs to zero-phase identity spiders; `IdentityRule`
    then removes the `(1,1)` ones (wide `(2,2)` ones stay, since they're
    not identities). The composition is done here, in the test helper,
    rather than inside `TerminalAbsorptionRule` -- a rule does not call
    another rule. `rebuild_registry()` between them is required because
    `apply_single` mutates node attrs without touching the registry.
    """
    rule.apply_rule(graph)
    graph.rebuild_registry()
    IdentityRule().apply_rule(graph)


def _absorb_single_then_cleanup(graph: CVZXGraph, rule: TerminalAbsorptionRule, match: dict) -> None:
    """Apply a single match with `rule`, rebuild the registry, then clean up."""
    rule.apply_single(graph, match)
    graph.rebuild_registry()
    IdentityRule().apply_rule(graph)


class TestTerminalAbsorptionRule(unittest.TestCase):
    """Test suite for TerminalAbsorptionRule."""

    def setUp(self):
        """Create common objects used in many tests."""
        # Rotation angle avoiding odd multiples of pi/2.
        self.theta1 = pi / 6
        self.theta2 = pi / 4

        # Terminals (effect = (1,0), state = (0,1)).
        self.q_effect = QSpider(1, 0, ZxPoly({1: -3.0}))
        self.q_state = QSpider(0, 1, ZxPoly({1: 3.0}))
        self.p_effect = PSpider(1, 0, ZxPoly({1: -3.0}))
        self.p_state = PSpider(0, 1, ZxPoly({1: 3.0}))

        # Gates.
        self.r1 = PhaseRotationGate(self.theta1)
        self.sq1 = SqueezingGate(2.0)
        self.sq2 = SqueezingGate(-3.0)

        # Inert filler / non-matching gates.
        self.swap = Swap()
        self.bs = BeamsplitterGate(pi / 4)
        self.q_filler = QSpider(1, 1, ZxPoly({1: 7.0}))
        self.p_filler = PSpider(1, 1, ZxPoly({2: 9.0, 1: 1.0}))

        # Default rule: exact-only (rotation absorption). Idealized rule: also
        # allows squeezing absorption and cross-color discard.
        self.rule = TerminalAbsorptionRule()
        self.rule_idealized = TerminalAbsorptionRule(assume_infinite_squeezing=True)

    def _expected_rotation(self, k: float, theta: float) -> tuple[float, float]:
        """Hand-computed rotation-fold coefficients: (linear, quadratic).

        Return:
        ------
        tuple[float, float]
        """
        return k / cos(theta), -tan(theta) / 2

    # -------------------------------------------------------------------------
    # 1. Testing match()  (unchanged)
    # -------------------------------------------------------------------------

    def test_match_rotation_effect(self):
        """R(theta) next to a QSpider effect matches, list order [R, effect]."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        lin, quad = self._expected_rotation(-3.0, self.theta1)
        assert isclose(matches[0]["result_phase"].coeffs[1], lin)
        assert isclose(matches[0]["result_phase"].coeffs[2], quad)
        assert matches[0]["result_type"] == "QSpider"
        assert matches[0]["result_num_inputs"] == 1
        assert matches[0]["result_num_outputs"] == 0

    def test_match_rotation_state(self):
        """R(theta) next to a QSpider state matches, list order [state, R]."""
        comp = CompositionDiagram([self.q_state, self.r1])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        lin, quad = self._expected_rotation(3.0, self.theta1)
        assert isclose(matches[0]["result_phase"].coeffs[1], lin)
        assert isclose(matches[0]["result_phase"].coeffs[2], quad)
        assert matches[0]["result_num_inputs"] == 0
        assert matches[0]["result_num_outputs"] == 1

    def test_match_rotation_pspider_no_match(self):
        """Rotation does not fold into a PSpider terminal -- QSpider only."""
        comp = CompositionDiagram([self.r1, self.p_effect])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_match_fourier2_effect(self):
        """Fourier2 next to a QSpider effect absorbs as R(pi), same formula."""
        comp = CompositionDiagram([Fourier2(), self.q_effect])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        lin, _ = self._expected_rotation(-3.0, pi)
        assert isclose(matches[0]["result_phase"].coeffs[1], lin)
        assert isclose(matches[0]["result_phase"].coeffs.get(2, 0.0), 0.0, abs_tol=1e-9)
        assert matches[0]["result_type"] == "QSpider"
        assert matches[0]["result_num_inputs"] == 1
        assert matches[0]["result_num_outputs"] == 0

    def test_match_fourier2_state(self):
        """Fourier2 next to a QSpider state absorbs as R(pi) too."""
        comp = CompositionDiagram([self.q_state, Fourier2()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        lin, _ = self._expected_rotation(3.0, pi)
        assert isclose(matches[0]["result_phase"].coeffs[1], lin)
        assert isclose(matches[0]["result_phase"].coeffs.get(2, 0.0), 0.0, abs_tol=1e-9)
        assert matches[0]["result_num_inputs"] == 0
        assert matches[0]["result_num_outputs"] == 1

    def test_match_fourier2_pspider_no_match(self):
        """Fourier2 does not fold into a PSpider terminal -- QSpider only."""
        comp = CompositionDiagram([Fourier2(), self.p_effect])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_match_fourier_no_match(self):
        """Fourier (theta=-pi/2 equivalent) is not absorbed."""
        comp = CompositionDiagram([Fourier(), self.q_effect])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_match_fourier_inv_no_match(self):
        """FourierInv (theta=+pi/2 equivalent) is not absorbed."""
        comp = CompositionDiagram([self.q_state, FourierInv()])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_match_squeezing_effect_qspider(self):
        """Sq(tau) next to a QSpider effect matches, x -> x/tau."""
        comp = CompositionDiagram([self.sq1, self.q_effect])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 1
        assert isclose(matches[0]["result_phase"].coeffs[1], -6.0)
        assert matches[0]["result_type"] == "QSpider"

    def test_match_squeezing_effect_pspider(self):
        """Sq(tau) next to a PSpider effect matches too -- either color."""
        comp = CompositionDiagram([self.sq1, self.p_effect])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "PSpider"

    def test_match_squeezing_state(self):
        """Sq(tau) next to a state matches, list order [state, Sq]."""
        comp = CompositionDiagram([self.q_state, self.sq1])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 1
        assert isclose(matches[0]["result_phase"].coeffs[1], 6.0)

    def test_match_cross_color_discard_effect_q_terminal(self):
        """A raw PSpider(1,1,f) folds into an adjacent QSpider effect, unchanged."""
        comp = CompositionDiagram([self.p_filler, self.q_effect])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "QSpider"
        assert matches[0]["result_phase"] == self.q_effect.phase

    def test_match_cross_color_discard_effect_p_terminal(self):
        """A raw QSpider(1,1,f) folds into an adjacent PSpider effect, unchanged."""
        comp = CompositionDiagram([self.q_filler, self.p_effect])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "PSpider"
        assert matches[0]["result_phase"] == self.p_effect.phase

    def test_match_cross_color_discard_state(self):
        """A raw PSpider(1,1,f) folds into an adjacent QSpider state, unchanged."""
        comp = CompositionDiagram([self.q_state, self.p_filler])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_phase"] == self.q_state.phase

    def test_match_displacement_effect_qspider(self):
        """A DisplacementGate collapses into an adjacent QSpider effect, unchanged."""
        comp = CompositionDiagram([DisplacementGate(0.5), self.q_effect])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "QSpider"
        assert matches[0]["result_phase"] == self.q_effect.phase

    def test_match_displacement_effect_pspider(self):
        """A DisplacementGate collapses into an adjacent PSpider effect too -- either color."""
        comp = CompositionDiagram([DisplacementGate(0.5), self.p_effect])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "PSpider"
        assert matches[0]["result_phase"] == self.p_effect.phase

    def test_match_displacement_state(self):
        """A DisplacementGate collapses into an adjacent state, list order [state, D]."""
        comp = CompositionDiagram([self.q_state, DisplacementGate(-1.0)])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_phase"] == self.q_state.phase
        assert matches[0]["result_num_inputs"] == 0
        assert matches[0]["result_num_outputs"] == 1

    def test_match_displacement_high_degree_no_match(self):
        """A DisplacementGate does not collapse into a degree > 1 (non-R1) terminal."""
        curved_effect = QSpider(1, 0, ZxPoly({2: 1.0, 1: -3.0}))
        comp = CompositionDiagram([DisplacementGate(0.5), curved_effect])
        graph = to_graph(comp)
        assert len(self.rule_idealized.match(graph)) == 0

    def test_match_same_color_no_match(self):
        """Same-color (1,1) spider next to the terminal is FusionRule's job."""
        comp = CompositionDiagram([self.q_filler, self.q_effect])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_match_two_terminals_adjacent_no_match(self):
        """A state directly followed by an effect isn't a gate/terminal pattern."""
        comp = CompositionDiagram([self.q_state, self.q_effect])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_match_no_unrelated_gate(self):
        """A terminal next to an unrelated gate does not match."""
        comp = CompositionDiagram([self.swap, self.swap])
        tensor = TensorDiagram([self.bs, self.q_effect])
        comp2 = CompositionDiagram([self.p_state, DisplacementGate(0.5)])
        for diagram in (comp, tensor, comp2):
            graph = to_graph(diagram)
            assert len(self.rule.match(graph)) == 0

    def test_match_no_rotation_rotation(self):
        """Two rotations do not match this rule -- ChainReductionRule's job."""
        comp = CompositionDiagram([self.r1, PhaseRotationGate(self.theta2)])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_match_indices_and_node_ids(self):
        """Match records correct container/indices/node_ids for a simple pair."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["indices"] == [0, 1]
        assert matches[0]["node_ids"] == [self.q_effect.id, self.r1.id]

    def test_match_multiple_independent(self):
        """State-end and effect-end absorptions both match in one call."""
        comp = CompositionDiagram([self.q_state, self.r1, self.sq1, self.q_effect])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 2
        assert matches[0]["indices"] == [0, 1]
        assert matches[1]["indices"] == [2, 3]

    def test_match_nested_in_tensor(self):
        """Match a pattern nested inside a tensor branch."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        tensor = TensorDiagram([self.swap, comp, self.bs])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] in graph.registry.composition_nodes

    def test_match_nested_in_contracted(self):
        """Match a pattern inside a composition inside a ContractedDiagram."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        contracted = ContractedDiagram(comp, self.swap, [], [], [0], [0])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id

    def test_match_excludes_gate_directly_in_contracted(self):
        """A gate directly in a ContractedDiagram is never matched."""
        filler_in_contracted = QSpider(1, 1, ZxPoly({1: 7.0}))
        contracted = ContractedDiagram(self.r1, filler_in_contracted, [], [], [0], [0])
        comp = CompositionDiagram([contracted, self.q_effect])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    # -------------------------------------------------------------------------
    # 2. Testing apply_single()
    # -------------------------------------------------------------------------

    def test_apply_single_rotation_effect(self):
        """R folded into a QSpider effect: terminal updated, gate reset to identity."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        _absorb_single_then_cleanup(graph, self.rule, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        lin, quad = self._expected_rotation(-3.0, self.theta1)
        assert isclose(result.phase.coeffs[1], lin)
        assert isclose(result.phase.coeffs[2], quad)
        assert result.num_inputs == 1
        assert result.num_outputs == 0

    def test_apply_single_fourier2_effect(self):
        """Fourier2 folded into a QSpider effect, same as R(pi)."""
        comp = CompositionDiagram([Fourier2(), self.q_effect])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        _absorb_single_then_cleanup(graph, self.rule, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        lin, _ = self._expected_rotation(-3.0, pi)
        assert isclose(result.phase.coeffs[1], lin)
        assert isclose(result.phase.coeffs.get(2, 0.0), 0.0, abs_tol=1e-9)
        assert result.num_inputs == 1
        assert result.num_outputs == 0

    def test_apply_single_squeezing_state(self):
        """Sq folded into a state: terminal updated, gate reset to identity."""
        comp = CompositionDiagram([self.q_state, self.sq1])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        _absorb_single_then_cleanup(graph, self.rule_idealized, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert isclose(result.phase.coeffs[1], 6.0)
        assert result.num_inputs == 0
        assert result.num_outputs == 1

    def test_apply_single_cross_color_discard(self):
        """Discarding a raw opposite-color spider leaves the terminal untouched."""
        comp = CompositionDiagram([self.p_filler, self.q_effect])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        _absorb_single_then_cleanup(graph, self.rule_idealized, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == self.q_effect.phase

    def test_apply_single_displacement_effect(self):
        """Collapsing a DisplacementGate leaves the terminal untouched."""
        comp = CompositionDiagram([DisplacementGate(0.5), self.q_effect])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        _absorb_single_then_cleanup(graph, self.rule_idealized, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == self.q_effect.phase

    def test_apply_single_with_trailing_content(self):
        """Folding leaves trailing content untouched."""
        comp = CompositionDiagram([self.q_state, self.r1, self.q_filler])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        _absorb_single_then_cleanup(graph, self.rule, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        lin, quad = self._expected_rotation(3.0, self.theta1)
        folded = result.diagrams[0]
        assert isinstance(folded, QSpider)
        assert isclose(folded.phase.coeffs[1], lin)
        assert isclose(folded.phase.coeffs[2], quad)
        assert result.diagrams[1] == self.q_filler
        assert result.connectivity == {0: {0: 0}}

    def test_apply_single_nested_in_tensor(self):
        """Apply the fold to a pattern nested inside a tensor branch."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        tensor = TensorDiagram([self.swap, comp, self.bs])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        _absorb_single_then_cleanup(graph, self.rule, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 3
        assert result.diagrams[0] == self.swap
        assert isinstance(result.diagrams[1], QSpider)
        assert result.diagrams[2] == self.bs

    def test_apply_single_nested_in_contracted(self):
        """Apply the fold to a pattern inside a composition inside a ContractedDiagram."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        contracted = ContractedDiagram(comp, self.swap, [], [], [0], [0])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        _absorb_single_then_cleanup(graph, self.rule, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, ContractedDiagram)
        assert isinstance(result.first, QSpider)
        assert result.second == self.swap

    # -------------------------------------------------------------------------
    # 3. Testing apply_rule() (full application)
    # -------------------------------------------------------------------------

    def test_apply_rule_rotation_effect(self):
        """Full rule application for R next to a QSpider effect."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        lin, quad = self._expected_rotation(-3.0, self.theta1)
        assert isclose(result.phase.coeffs[1], lin)
        assert isclose(result.phase.coeffs[2], quad)

    def test_apply_rule_rotation_state(self):
        """Full rule application for R next to a QSpider state."""
        comp = CompositionDiagram([self.q_state, self.r1])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        lin, quad = self._expected_rotation(3.0, self.theta1)
        assert isclose(result.phase.coeffs[1], lin)
        assert isclose(result.phase.coeffs[2], quad)

    def test_apply_rule_fourier2_effect(self):
        """Full rule application for Fourier2 next to a QSpider effect."""
        comp = CompositionDiagram([Fourier2(), self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        lin, _ = self._expected_rotation(-3.0, pi)
        assert isclose(result.phase.coeffs[1], lin)
        assert isclose(result.phase.coeffs.get(2, 0.0), 0.0, abs_tol=1e-9)

    def test_apply_rule_fourier2_state(self):
        """Full rule application for Fourier2 next to a QSpider state."""
        comp = CompositionDiagram([self.q_state, Fourier2()])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        lin, _ = self._expected_rotation(3.0, pi)
        assert isclose(result.phase.coeffs[1], lin)
        assert isclose(result.phase.coeffs.get(2, 0.0), 0.0, abs_tol=1e-9)

    def test_apply_rule_fourier_and_fourier_inv_leave_terminal_unreduced(self):
        """Fourier/FourierInv touching a terminal -- the rule leaves the diagram unchanged."""
        comp = CompositionDiagram([Fourier(), self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)

    def test_apply_rule_squeezing_effect(self):
        """Full rule application for Sq next to a QSpider effect."""
        comp = CompositionDiagram([self.sq1, self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert isclose(result.phase.coeffs[1], -6.0)

    def test_apply_rule_squeezing_pspider_state(self):
        """Full rule application for Sq next to a PSpider state."""
        comp = CompositionDiagram([self.p_state, self.sq1])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, PSpider)
        assert isclose(result.phase.coeffs[1], 3.0 / 2.0)

    def test_apply_rule_cross_color_discard_effect(self):
        """Full rule application discarding a raw opposite-color spider (effect)."""
        comp = CompositionDiagram([self.p_filler, self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == self.q_effect.phase

    def test_apply_rule_cross_color_discard_state(self):
        """Full rule application discarding a raw opposite-color spider (state)."""
        comp = CompositionDiagram([self.q_state, self.p_filler])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == self.q_state.phase

    def test_apply_rule_displacement_effect(self):
        """Full rule application collapsing a DisplacementGate into an effect."""
        comp = CompositionDiagram([DisplacementGate(0.5), self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == self.q_effect.phase

    def test_apply_rule_displacement_state(self):
        """Full rule application collapsing a DisplacementGate into a state."""
        comp = CompositionDiagram([self.q_state, DisplacementGate(-1.0)])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == self.q_state.phase

    def test_apply_rule_no_match_returns_same_diagram(self):
        """If no pattern is present, apply_rule should return the original diagram."""
        comp = CompositionDiagram([self.swap, self.bs])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert result == comp

    def test_apply_rule_multiple_independent_matches(self):
        """State-end and effect-end absorptions both fold in one pass."""
        comp = CompositionDiagram([self.q_state, self.r1, self.sq1, self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        lin_state, quad_state = self._expected_rotation(3.0, self.theta1)
        folded_state, folded_effect = result.diagrams
        assert isinstance(folded_state, QSpider)
        assert isinstance(folded_effect, QSpider)
        assert isclose(folded_state.phase.coeffs[1], lin_state)
        assert isclose(folded_state.phase.coeffs[2], quad_state)
        assert isclose(folded_effect.phase.coeffs[1], -6.0)

    def test_apply_rule_four_element_chain_one_pass(self):
        """[state, R1, R2, effect]: both boundary pairs fold in a single pass."""
        comp = CompositionDiagram([
            self.q_state,
            PhaseRotationGate(self.theta1),
            PhaseRotationGate(self.theta2),
            self.q_effect,
        ])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[0], QSpider)
        assert isinstance(result.diagrams[1], QSpider)
        lin_state, quad_state = self._expected_rotation(3.0, self.theta1)
        lin_effect, quad_effect = self._expected_rotation(-3.0, self.theta2)
        assert isclose(result.diagrams[0].phase.coeffs[1], lin_state)
        assert isclose(result.diagrams[0].phase.coeffs[2], quad_state)
        assert isclose(result.diagrams[1].phase.coeffs[1], lin_effect)
        assert isclose(result.diagrams[1].phase.coeffs[2], quad_effect)

    def test_apply_rule_nested_in_tensor(self):
        """Full rule application to a pattern nested inside a tensor branch."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        tensor = TensorDiagram([self.swap, comp, self.bs])
        graph = to_graph(tensor)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 3
        assert isinstance(result.diagrams[1], QSpider)

    # -------------------------------------------------------------------------
    # 4. Edge cases
    # -------------------------------------------------------------------------

    def test_default_rule_skips_squeezing(self):
        """The default (exact-only) rule does not match squeezing absorption."""
        comp = CompositionDiagram([self.sq1, self.q_effect])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_default_rule_skips_cross_color_discard(self):
        """The default (exact-only) rule does not match cross-color discard."""
        comp = CompositionDiagram([self.p_filler, self.q_effect])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_default_rule_skips_displacement(self):
        """The default (exact-only) rule does not match displacement absorption."""
        comp = CompositionDiagram([DisplacementGate(0.5), self.q_effect])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_default_rule_still_matches_rotation(self):
        """Rotation absorption is exact and matches regardless of the flag."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 1

    def test_apply_rule_default_leaves_squeezing_effect_unreduced(self):
        """Full rule application with the default flag leaves Sq/terminal untouched."""
        comp = CompositionDiagram([self.sq1, self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert result == comp

    def test_symbolic_theta_rotation(self):
        """Folding works with a symbolic rotation angle."""
        theta = Symbol("theta", real=True)
        comp = CompositionDiagram([PhaseRotationGate(theta, parametric=True), self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        expected = ZxPoly({1: -3.0 / sym_cos(theta), 2: -sym_tan(theta) / 2})
        assert simplify(result.phase.as_expr() - expected.as_expr()) == 0

    def test_degenerate_angle_no_match(self):
        """Theta an odd multiple of pi/2 does not match -- tan/1-over-cos undefined."""
        comp = CompositionDiagram([PhaseRotationGate(pi / 2, parametric=True), self.q_effect])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_rotation_degree_too_high_no_match(self):
        """A quadratic (or higher) terminal phase does not match rotation absorption."""
        quadratic_effect = QSpider(1, 0, ZxPoly({2: 1.0, 1: -3.0}))
        comp = CompositionDiagram([self.r1, quadratic_effect])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_constant_term_carries_through_rotation(self):
        """A constant term in the terminal's phase survives the rotation fold unchanged."""
        effect_with_const = QSpider(1, 0, ZxPoly({0: 2.0, 1: -3.0}))
        comp = CompositionDiagram([self.r1, effect_with_const])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert isclose(result.phase.coeffs[0], 2.0)

    def test_squeezing_any_degree(self):
        """Squeezing absorption isn't restricted to linear phase."""
        cubic_effect = QSpider(1, 0, ZxPoly({3: 1.0, 1: -2.0}))
        comp = CompositionDiagram([self.sq1, cubic_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert isclose(result.phase.coeffs[3], 8.0)
        assert isclose(result.phase.coeffs[1], -4.0)

    def test_squeezing_symbolic_tau(self):
        """Squeezing absorption works with a symbolic tau."""
        tau = Symbol("tau", real=True, nonzero=True)
        sq_sym = SqueezingGate(tau, parametric=True)
        comp = CompositionDiagram([sq_sym, self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        expected = ZxPoly({1: -3.0 * tau})
        assert simplify(result.phase.as_expr() - expected.as_expr()) == 0

    def test_squeezing_constant_term_unchanged(self):
        """A constant term is untouched by x -> x/tau (it doesn't depend on x)."""
        effect_with_const = QSpider(1, 0, ZxPoly({0: 5.0, 1: -3.0}))
        comp = CompositionDiagram([self.sq1, effect_with_const])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert isclose(result.phase.coeffs[0], 5.0)

    def test_cross_color_discard_arbitrary_degree(self):
        """Cross-color discard has no degree restriction on the vanishing gate."""
        high_degree_filler = PSpider(1, 1, ZxPoly({5: 2.0, 3: -1.0, 0: 4.0}))
        comp = CompositionDiagram([high_degree_filler, self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == self.q_effect.phase

    def test_displacement_constant_phase(self):
        """Displacement absorption works for a constant-only (degree 0) terminal phase."""
        constant_effect = QSpider(1, 0, ZxPoly({0: 4.0}))
        comp = CompositionDiagram([DisplacementGate(0.5), constant_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == constant_effect.phase

    def test_displacement_high_degree_left_untouched(self):
        """A DisplacementGate next to a degree > 1 (non-R1) terminal is left untouched."""
        curved_effect = QSpider(1, 0, ZxPoly({2: 1.0, 1: -3.0}))
        comp = CompositionDiagram([DisplacementGate(0.5), curved_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert result == comp

    def test_same_color_left_for_fusion_rule(self):
        """A same-color (1,1) filler next to a terminal is left untouched."""
        comp = CompositionDiagram([self.q_filler, self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert result == comp

    def test_squeezing_negative_tau(self):
        """Negative tau divides the phase's coefficients as-is (no sign special-casing)."""
        comp = CompositionDiagram([self.sq2, self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert isclose(result.phase.coeffs[1], 9.0)

    # -------------------------------------------------------------------------
    # 5. Chain-chasing through identity spiders AND `Swap`
    # -------------------------------------------------------------------------

    def test_match_through_identity_effect_direction(self):
        """An effect chases backward through one identity spider to its gate."""
        zero_phase = ZxPoly({})
        id1 = QSpider(1, 1, zero_phase)
        comp = CompositionDiagram([self.r1, id1, self.q_effect])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["identity_chain"] == [id1.id]
        assert matches[0]["node_ids"] == [self.q_effect.id, self.r1.id]

    def test_apply_rule_through_identity_effect_direction(self):
        """Applying the rule folds the rotation and leaves no match."""
        zero_phase = ZxPoly({})
        id1 = QSpider(1, 1, zero_phase)
        comp = CompositionDiagram([self.r1, id1, self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        lin, quad = self._expected_rotation(-3.0, self.theta1)
        qspiders = [
            d
            for d in (result.diagrams if hasattr(result, "diagrams") else [result])
            if isinstance(d, QSpider) and d.num_inputs == 1 and d.num_outputs == 0
        ]
        assert len(qspiders) == 1
        assert isclose(qspiders[0].phase.coeffs[1], lin)
        assert isclose(qspiders[0].phase.coeffs[2], quad)
        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

    def test_match_through_identity_state_direction(self):
        """A state chases forward through one identity spider to its gate."""
        zero_phase = ZxPoly({})
        id1 = PSpider(1, 1, zero_phase)
        comp = CompositionDiagram([self.q_state, id1, self.r1])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["identity_chain"] == [id1.id]
        assert matches[0]["node_ids"] == [self.q_state.id, self.r1.id]

    def test_match_swap_interleaved_with_identities(self):
        """Chase forward through a `Swap` interleaved with an identity."""
        zero_phase = ZxPoly({})
        term_state = QSpider(0, 1, ZxPoly({1: 3.0}))
        filler_state = PSpider(0, 1, ZxPoly({2: 9.0, 1: 1.0}))
        id_filler = PSpider(1, 1, zero_phase)
        id_term = QSpider(1, 1, zero_phase)
        filler_terminal = PSpider(1, 1, zero_phase)

        lhs = TensorDiagram([term_state, filler_state])
        swap = Swap()
        mid = TensorDiagram([id_filler, id_term])
        final_stage = TensorDiagram([filler_terminal, self.r1])
        comp = CompositionDiagram([lhs, swap, mid, final_stage])

        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["node_ids"] == [term_state.id, self.r1.id]
        assert set(matches[0]["identity_chain"]) == {swap.id, id_term.id}
        assert id_filler.id not in matches[0]["identity_chain"]

    def test_apply_rule_swap_interleaved_with_identities(self):
        """Applying the rule resolves the crossed pattern."""
        zero_phase = ZxPoly({})
        term_state = QSpider(0, 1, ZxPoly({1: 3.0}))
        filler_state = PSpider(0, 1, ZxPoly({2: 9.0, 1: 1.0}))
        id_filler = PSpider(1, 1, zero_phase)
        id_term = QSpider(1, 1, zero_phase)
        filler_terminal = PSpider(1, 1, zero_phase)

        lhs = TensorDiagram([term_state, filler_state])
        swap = Swap()
        mid = TensorDiagram([id_filler, id_term])
        final_stage = TensorDiagram([filler_terminal, self.r1])
        comp = CompositionDiagram([lhs, swap, mid, final_stage])

        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert result.num_inputs == 0
        assert result.num_outputs == 2
        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

    def test_match_displacement_through_identity_effect_direction(self):
        """An effect chases backward through one identity spider to a DisplacementGate."""
        zero_phase = ZxPoly({})
        id1 = QSpider(1, 1, zero_phase)
        d = DisplacementGate(0.5)
        comp = CompositionDiagram([d, id1, self.q_effect])
        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 1
        assert matches[0]["identity_chain"] == [id1.id]
        assert matches[0]["node_ids"] == [self.q_effect.id, d.id]
        assert matches[0]["result_phase"] == self.q_effect.phase

    def test_apply_rule_displacement_through_identity_effect_direction(self):
        """Applying the rule collapses the DisplacementGate, leaving the effect's phase unchanged."""
        zero_phase = ZxPoly({})
        id1 = QSpider(1, 1, zero_phase)
        comp = CompositionDiagram([DisplacementGate(0.5), id1, self.q_effect])
        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        qspiders = [
            d
            for d in (result.diagrams if hasattr(result, "diagrams") else [result])
            if isinstance(d, QSpider) and d.num_inputs == 1 and d.num_outputs == 0
        ]
        assert len(qspiders) == 1
        assert qspiders[0].phase == self.q_effect.phase
        graph = to_graph(result)
        assert len(self.rule_idealized.match(graph)) == 0

    def test_match_displacement_swap_interleaved_with_identities(self):
        """Chase forward through a `Swap` interleaved with an identity, to a DisplacementGate."""
        zero_phase = ZxPoly({})
        term_state = QSpider(0, 1, ZxPoly({1: 3.0}))
        filler_state = PSpider(0, 1, ZxPoly({2: 9.0, 1: 1.0}))
        id_filler = PSpider(1, 1, zero_phase)
        id_term = QSpider(1, 1, zero_phase)
        filler_terminal = PSpider(1, 1, zero_phase)
        d = DisplacementGate(0.5)

        lhs = TensorDiagram([term_state, filler_state])
        swap = Swap()
        mid = TensorDiagram([id_filler, id_term])
        final_stage = TensorDiagram([filler_terminal, d])
        comp = CompositionDiagram([lhs, swap, mid, final_stage])

        graph = to_graph(comp)
        matches = self.rule_idealized.match(graph)
        assert len(matches) == 1
        assert matches[0]["node_ids"] == [term_state.id, d.id]
        assert matches[0]["result_phase"] == term_state.phase
        assert set(matches[0]["identity_chain"]) == {swap.id, id_term.id}
        assert id_filler.id not in matches[0]["identity_chain"]

    def test_apply_rule_displacement_swap_interleaved_with_identities(self):
        """Applying the rule resolves the crossed displacement pattern."""
        zero_phase = ZxPoly({})
        term_state = QSpider(0, 1, ZxPoly({1: 3.0}))
        filler_state = PSpider(0, 1, ZxPoly({2: 9.0, 1: 1.0}))
        id_filler = PSpider(1, 1, zero_phase)
        id_term = QSpider(1, 1, zero_phase)
        filler_terminal = PSpider(1, 1, zero_phase)

        lhs = TensorDiagram([term_state, filler_state])
        swap = Swap()
        mid = TensorDiagram([id_filler, id_term])
        final_stage = TensorDiagram([filler_terminal, DisplacementGate(0.5)])
        comp = CompositionDiagram([lhs, swap, mid, final_stage])

        graph = to_graph(comp)
        _apply_absorption_and_cleanup(graph, self.rule_idealized)
        result = to_diagram(graph)
        assert result.num_inputs == 0
        assert result.num_outputs == 2
        graph = to_graph(result)
        assert len(self.rule_idealized.match(graph)) == 0


class TestTerminalAbsorptionResetToIdentityClearsParamMeasurementMap(unittest.TestCase):
    """Test that `_reset_to_identity` clears `param_measurement_map` (Phase 3 fix).

    A gate absorbed into a terminal (or an identity/Swap passthrough the
    chase crossed) must lose its feedforward provenance entirely, not just
    have its phase zeroed while a stale map lingers.
    """

    def test_reset_to_identity_clears_param_measurement_map(self):
        """A feedforward node with a param_measurement_map is fully cleared on reset."""
        m = Symbol("m", real=True)
        spider = QSpider(1, 1, ZxPoly({1: m}), parametric=True, param_measurement_map={m: {7}})
        graph = to_graph(spider)

        TerminalAbsorptionRule._reset_to_identity(graph.graph, spider.id)  # ruff: ignore[private-member-access]

        attrs = graph.graph.nodes[spider.id]
        assert attrs["param_measurement_map"] == {}
        assert attrs["feedforward"] is None
        assert attrs["measurement_ids"] is None
        assert attrs["phase"] == ZxPoly({})
        assert attrs["type"] == "QSpider"


if __name__ == "__main__":
    from cvzx.ir.base import Diagram
    from cvzx.backends.nx.rules import RewriteRule
    from cvzx.utils.visualization_base_gates import visualize_before_after

    rule_name = "Terminal Absorption Rule"
    rule = TerminalAbsorptionRule()
    rule_idealized = TerminalAbsorptionRule(assume_infinite_squeezing=True)
    theta = pi / 6

    swap = Swap()
    bs = BeamsplitterGate(pi / 4)
    q_effect = QSpider(1, 0, ZxPoly({1: -3.0}))
    q_state = QSpider(0, 1, ZxPoly({1: 3.0}))
    p_effect = PSpider(1, 0, ZxPoly({1: -3.0}))
    p_filler = PSpider(1, 1, ZxPoly({2: 4.0, 1: 1.0}))

    # Helper: apply rule then IdentityRule (driven here, not inside the rule)
    def _absorb_then_cleanup(diagram: Diagram, r: RewriteRule) -> Diagram:
        """Apply the Terminal Absorption Rule then the Identity Rule.

        Return:
        ------
        Diagram
        """
        g = to_graph(diagram)
        r.apply_rule(g)
        g.rebuild_registry()
        IdentityRule().apply_rule(g)
        return to_diagram(g)

    # Test 1: rotation folded into a QSpider effect
    comp1 = CompositionDiagram([PhaseRotationGate(theta), QSpider(1, 0, ZxPoly({1: -3.0}))])
    comp1_after = _absorb_then_cleanup(comp1, rule)
    visualize_before_after(comp1, comp1_after, "Rotation into Effect", rule_name)

    # Test 2: rotation folded into a QSpider state
    comp2 = CompositionDiagram([QSpider(0, 1, ZxPoly({1: 3.0})), PhaseRotationGate(theta)])
    comp2_after = _absorb_then_cleanup(comp2, rule)
    visualize_before_after(comp2, comp2_after, "Rotation into State", rule_name)

    # Test 3: squeezing folded into a QSpider effect
    comp3 = CompositionDiagram([SqueezingGate(2.0), QSpider(1, 0, ZxPoly({1: -3.0}))])
    comp3_after = _absorb_then_cleanup(comp3, rule_idealized)
    visualize_before_after(comp3, comp3_after, "Squeezing into Effect", rule_name)

    # Test 4: squeezing folded into a PSpider effect
    comp4 = CompositionDiagram([SqueezingGate(2.0), PSpider(1, 0, ZxPoly({1: -3.0}))])
    comp4_after = _absorb_then_cleanup(comp4, rule_idealized)
    visualize_before_after(comp4, comp4_after, "Squeezing into PSpider Effect", rule_name)

    # Test 5: cross-color discard into a QSpider effect
    comp5 = CompositionDiagram([PSpider(1, 1, ZxPoly({2: 4.0, 1: 1.0})), QSpider(1, 0, ZxPoly({1: -3.0}))])
    comp5_after = _absorb_then_cleanup(comp5, rule_idealized)
    visualize_before_after(comp5, comp5_after, "Cross-Color Discard", rule_name)

    # Test 6: both ends of a composition fold in one pass
    comp6 = CompositionDiagram([
        QSpider(0, 1, ZxPoly({1: 3.0})),
        PhaseRotationGate(theta),
        SqueezingGate(2.0),
        QSpider(1, 0, ZxPoly({1: -3.0})),
    ])
    comp6_after = _absorb_then_cleanup(comp6, rule_idealized)
    visualize_before_after(comp6, comp6_after, "Both Ends in One Pass", rule_name)

    # Test 7: pattern nested inside a tensor branch
    inner_comp = CompositionDiagram([PhaseRotationGate(theta), QSpider(1, 0, ZxPoly({1: -3.0}))])
    tensor = TensorDiagram([swap, inner_comp, bs])
    tensor_after = _absorb_then_cleanup(tensor, rule)
    visualize_before_after(tensor, tensor_after, "Nested in Tensor", rule_name)

    # Test 8: diagram unchanged because the rule doesn't apply
    diagram = CompositionDiagram([swap, bs])
    diagram_after = _absorb_then_cleanup(diagram, rule)
    visualize_before_after(diagram, diagram_after, "No reduction", rule_name)

    # Test 9: Fourier2 folds into a QSpider effect
    comp9 = CompositionDiagram([Fourier2(), QSpider(1, 0, ZxPoly({1: -3.0}))])
    comp9_after = _absorb_then_cleanup(comp9, rule)
    visualize_before_after(comp9, comp9_after, "Fourier2 into Effect", rule_name)

    # Test 10: Fourier/FourierInv (fixed rotations by -+pi/2) never absorb
    diagram_f = CompositionDiagram([Fourier(), QSpider(1, 0, ZxPoly({1: -3.0}))])
    diagram_f_after = _absorb_then_cleanup(diagram_f, rule)
    visualize_before_after(diagram_f, diagram_f_after, "Fourier into Effect - No reduction", rule_name)

    # Test 11: effect chases backward through an identity spider to its gate
    id1 = QSpider(1, 1, ZxPoly({}))
    comp11 = CompositionDiagram([PhaseRotationGate(theta), id1, QSpider(1, 0, ZxPoly({1: -3.0}))])
    comp11_after = _absorb_then_cleanup(comp11, rule)
    visualize_before_after(comp11, comp11_after, "Through Identity - Effect Direction", rule_name)

    # Test 12: state chases forward through an identity spider to its gate
    id2 = PSpider(1, 1, ZxPoly({}))
    comp12 = CompositionDiagram([QSpider(0, 1, ZxPoly({1: 3.0})), id2, PhaseRotationGate(theta)])
    comp12_after = _absorb_then_cleanup(comp12, rule)
    visualize_before_after(comp12, comp12_after, "Through Identity - State Direction", rule_name)

    # Test 13: a Swap interleaved with an identity spider on the terminal
    # state's own lane, with an unrelated wire sharing the Swap's other two ports.
    term_state13 = QSpider(0, 1, ZxPoly({1: 3.0}))
    filler_state13 = PSpider(0, 1, ZxPoly({2: 9.0, 1: 1.0}))
    id_filler13 = PSpider(1, 1, ZxPoly({}))
    id_term13 = QSpider(1, 1, ZxPoly({}))
    filler_terminal13 = PSpider(1, 1, ZxPoly({}))
    lhs13 = TensorDiagram([term_state13, filler_state13])
    swap13 = Swap()
    mid13 = TensorDiagram([id_filler13, id_term13])
    final_stage13 = TensorDiagram([filler_terminal13, PhaseRotationGate(theta)])
    comp13 = CompositionDiagram([lhs13, swap13, mid13, final_stage13])
    comp13_after = _absorb_then_cleanup(comp13, rule)
    visualize_before_after(comp13, comp13_after, "Swap Interleaved With Identities", rule_name)

    # Test 14: Complex Swap Network
    zero_phase = ZxPoly({})
    bloc1 = TensorDiagram([
        CompositionDiagram([DisplacementGate(2), QSpider(1, 1, zero_phase), QSpider(1, 1, zero_phase)]),
        CompositionDiagram([QSpider(0, 1, zero_phase), SqueezingGate(2.5), PSpider(1, 1, zero_phase)]),
        CompositionDiagram([PhaseRotationGate(4), PhaseRotationGate(8)]),
    ])
    bloc2 = TensorDiagram([Swap(), QSpider(1, 1, zero_phase)])
    bloc3 = TensorDiagram([SqueezingGate(2.5), DisplacementGate(-1), QSpider(1, 1, zero_phase)])
    bloc4 = TensorDiagram([QSpider(1, 1, zero_phase), Swap()])
    bloc5 = TensorDiagram([Swap(), QSpider(1, 1, zero_phase)])
    bloc6 = TensorDiagram([
        PhaseRotationGate(10),
        SqueezingGate(2.5),
        CompositionDiagram([DisplacementGate(-1), QSpider(1, 0, zero_phase)]),
    ])
    comp14 = CompositionDiagram([bloc1, bloc2, bloc3, bloc4, bloc5, bloc6])
    comp14_after = _absorb_then_cleanup(comp14, rule_idealized)
    visualize_before_after(
        comp14,
        comp14_after,
        "Complex Cross-Container Chain Through Swap And Identities",
        rule_name,
    )

    # Test 15: Complex Swap Network 2
    zero_phase = ZxPoly({})
    bloc1 = TensorDiagram([
        CompositionDiagram([
            QSpider(0, 1, zero_phase),
            DisplacementGate(2),
            QSpider(1, 1, zero_phase),
            QSpider(1, 1, zero_phase),
        ]),
        CompositionDiagram([QSpider(0, 1, zero_phase), SqueezingGate(2.5), PSpider(1, 1, zero_phase)]),
        CompositionDiagram([PhaseRotationGate(4), PhaseRotationGate(8)]),
    ])
    bloc2 = TensorDiagram([Swap(), QSpider(1, 1, zero_phase)])
    bloc3 = TensorDiagram([SqueezingGate(2.5), DisplacementGate(-1), QSpider(1, 1, zero_phase)])
    bloc4 = TensorDiagram([QSpider(1, 1, zero_phase), Swap()])
    bloc5 = TensorDiagram([Swap(), QSpider(1, 1, zero_phase)])
    bloc6 = TensorDiagram([
        PhaseRotationGate(10),
        SqueezingGate(2.5),
        CompositionDiagram([DisplacementGate(-1), QSpider(1, 0, ZxPoly({1: 1}))]),
    ])
    comp15 = CompositionDiagram([bloc1, bloc2, bloc3, bloc4, bloc5, bloc6])
    comp15_after = _absorb_then_cleanup(comp15, rule_idealized)
    visualize_before_after(
        comp15,
        comp15_after,
        "Complex Cross-Container Chain Through Swap And Identities 2",
        rule_name,
    )

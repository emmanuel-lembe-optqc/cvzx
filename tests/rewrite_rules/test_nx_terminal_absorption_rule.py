"""Unit tests for the terminal absorption rewrite rule using the graph formalism.

These tests verify the TerminalAbsorptionRule implementation on graphs: a gate
adjacent to a (1,0) effect or (0,1) state QSpider/PSpider terminal is folded
into the terminal's own phase. Three sub-cases: rotation (QSpider only,
degree <= 1 input), squeezing (either color, any degree), and cross-color
discard (opposite-color (1,1) raw spider vanishes, phase unchanged).
"""

import unittest
from math import cos, isclose, pi, tan

from sympy import Symbol, simplify
from sympy import cos as sym_cos
from sympy import tan as sym_tan

from cvzx.base_gates import (
    CompositionDiagram,
    ContractedDiagram,
    PSpider,
    QSpider,
    Swap,
    TensorDiagram,
    ZxPoly,
)
from cvzx.gates import BeamsplitterGate, DisplacementGate, PhaseRotationGate, SqueezingGate
from cvzx.nx_graph import GateRegister, to_diagram, to_graph
from cvzx.nx_rewrite_rules import TerminalAbsorptionRule, apply_rule_to_diagram


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

        self.rule = TerminalAbsorptionRule()

    def _expected_rotation(self, k: float, theta: float):  # noqa: ANN202
        """Hand-computed rotation-fold coefficients: (linear, quadratic)."""
        return k / cos(theta), -tan(theta) / 2

    # -------------------------------------------------------------------------
    # 1. Testing match()
    # -------------------------------------------------------------------------

    def test_match_rotation_effect(self):
        """R(theta) next to a QSpider effect matches, list order [R, effect]."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
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
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
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
        reg = GateRegister()
        reg.build_from_graph(graph)
        assert len(self.rule.match(graph, reg)) == 0

    def test_match_squeezing_effect_qspider(self):
        """Sq(tau) next to a QSpider effect matches, x -> x/tau."""
        comp = CompositionDiagram([self.sq1, self.q_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        assert len(matches) == 1
        assert isclose(matches[0]["result_phase"].coeffs[1], -3.0 / 2.0)
        assert matches[0]["result_type"] == "QSpider"

    def test_match_squeezing_effect_pspider(self):
        """Sq(tau) next to a PSpider effect matches too -- either color."""
        comp = CompositionDiagram([self.sq1, self.p_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "PSpider"

    def test_match_squeezing_state(self):
        """Sq(tau) next to a state matches, list order [state, Sq]."""
        comp = CompositionDiagram([self.q_state, self.sq1])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        assert len(matches) == 1
        assert isclose(matches[0]["result_phase"].coeffs[1], 3.0 / 2.0)

    def test_match_cross_color_discard_effect_q_terminal(self):
        """A raw PSpider(1,1,f) folds into an adjacent QSpider effect, unchanged."""
        comp = CompositionDiagram([self.p_filler, self.q_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "QSpider"
        assert matches[0]["result_phase"] == self.q_effect.phase

    def test_match_cross_color_discard_effect_p_terminal(self):
        """A raw QSpider(1,1,f) folds into an adjacent PSpider effect, unchanged."""
        comp = CompositionDiagram([self.q_filler, self.p_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "PSpider"
        assert matches[0]["result_phase"] == self.p_effect.phase

    def test_match_cross_color_discard_state(self):
        """A raw PSpider(1,1,f) folds into an adjacent QSpider state, unchanged."""
        comp = CompositionDiagram([self.q_state, self.p_filler])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        assert len(matches) == 1
        assert matches[0]["result_phase"] == self.q_state.phase

    def test_match_same_color_no_match(self):
        """Same-color (1,1) spider next to the terminal is FusionRule's job."""
        comp = CompositionDiagram([self.q_filler, self.q_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        assert len(self.rule.match(graph, reg)) == 0

    def test_match_two_terminals_adjacent_no_match(self):
        """A state directly followed by an effect isn't a gate/terminal pattern."""
        comp = CompositionDiagram([self.q_state, self.q_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        assert len(self.rule.match(graph, reg)) == 0

    def test_match_no_unrelated_gate(self):
        """A terminal next to an unrelated gate does not match."""
        comp = CompositionDiagram([self.swap, self.swap])
        tensor = TensorDiagram([self.bs, self.q_effect])
        comp2 = CompositionDiagram([self.p_state, DisplacementGate(0.5)])
        for diagram in (comp, tensor, comp2):
            graph = to_graph(diagram)
            reg = GateRegister()
            reg.build_from_graph(graph)
            assert len(self.rule.match(graph, reg)) == 0

    def test_match_no_rotation_rotation(self):
        """Two rotations do not match this rule -- ChainReductionRule's job."""
        comp = CompositionDiagram([self.r1, PhaseRotationGate(self.theta2)])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        assert len(self.rule.match(graph, reg)) == 0

    def test_match_indices_and_node_ids(self):
        """Match records correct container/indices/node_ids for a simple pair."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["indices"] == [0, 1]
        assert matches[0]["node_ids"] == [self.r1.id, self.q_effect.id]

    def test_match_multiple_independent(self):
        """State-end and effect-end absorptions both match in one call."""
        comp = CompositionDiagram([self.q_state, self.r1, self.sq1, self.q_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        assert len(matches) == 2
        assert matches[0]["indices"] == [0, 1]
        assert matches[1]["indices"] == [2, 3]

    def test_match_nested_in_tensor(self):
        """Match a pattern nested inside a tensor branch."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        tensor = TensorDiagram([self.swap, comp, self.bs])
        graph = to_graph(tensor)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        assert len(matches) == 1
        assert matches[0]["container_id"] in reg.composition_nodes

    def test_match_nested_in_contracted(self):
        """Match a pattern inside a composition inside a ContractedDiagram."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        contracted = ContractedDiagram(comp, self.swap, [], [], [0], [0])
        graph = to_graph(contracted)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id

    # -------------------------------------------------------------------------
    # 2. Testing apply_single()
    # -------------------------------------------------------------------------

    def test_apply_single_rotation_effect(self):
        """R folded into a QSpider effect leaves a single modified effect."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        lin, quad = self._expected_rotation(-3.0, self.theta1)
        assert isclose(result.phase.coeffs[1], lin)
        assert isclose(result.phase.coeffs[2], quad)
        assert result.num_inputs == 1
        assert result.num_outputs == 0

    def test_apply_single_squeezing_state(self):
        """Sq folded into a state leaves a single modified state."""
        comp = CompositionDiagram([self.q_state, self.sq1])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert isclose(result.phase.coeffs[1], 3.0 / 2.0)
        assert result.num_inputs == 0
        assert result.num_outputs == 1

    def test_apply_single_cross_color_discard(self):
        """Discarding a raw opposite-color spider leaves the terminal untouched."""
        comp = CompositionDiagram([self.p_filler, self.q_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == self.q_effect.phase

    def test_apply_single_with_trailing_content(self):
        """Folding leaves trailing content untouched."""
        comp = CompositionDiagram([self.q_state, self.r1, self.q_filler])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        lin, quad = self._expected_rotation(3.0, self.theta1)
        assert isclose(result.diagrams[0].phase.coeffs[1], lin)
        assert isclose(result.diagrams[0].phase.coeffs[2], quad)
        assert result.diagrams[1] == self.q_filler
        assert result.connectivity == {0: {0: 0}}

    def test_apply_single_nested_in_tensor(self):
        """Apply the fold to a pattern nested inside a tensor branch."""
        comp = CompositionDiagram([self.r1, self.q_effect])
        tensor = TensorDiagram([self.swap, comp, self.bs])
        graph = to_graph(tensor)
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        self.rule.apply_single(graph, matches[0])
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
        reg = GateRegister()
        reg.build_from_graph(graph)
        matches = self.rule.match(graph, reg)
        self.rule.apply_single(graph, matches[0])
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
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, QSpider)
        lin, quad = self._expected_rotation(-3.0, self.theta1)
        assert isclose(result.phase.coeffs[1], lin)
        assert isclose(result.phase.coeffs[2], quad)

    def test_apply_rule_rotation_state(self):
        """Full rule application for R next to a QSpider state."""
        comp = CompositionDiagram([self.q_state, self.r1])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, QSpider)
        lin, quad = self._expected_rotation(3.0, self.theta1)
        assert isclose(result.phase.coeffs[1], lin)
        assert isclose(result.phase.coeffs[2], quad)

    def test_apply_rule_squeezing_effect(self):
        """Full rule application for Sq next to a QSpider effect."""
        comp = CompositionDiagram([self.sq1, self.q_effect])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, QSpider)
        assert isclose(result.phase.coeffs[1], -3.0 / 2.0)

    def test_apply_rule_squeezing_pspider_state(self):
        """Full rule application for Sq next to a PSpider state."""
        comp = CompositionDiagram([self.p_state, self.sq1])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, PSpider)
        assert isclose(result.phase.coeffs[1], 3.0 / 2.0)

    def test_apply_rule_cross_color_discard_effect(self):
        """Full rule application discarding a raw opposite-color spider (effect)."""
        comp = CompositionDiagram([self.p_filler, self.q_effect])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, QSpider)
        assert result.phase == self.q_effect.phase

    def test_apply_rule_cross_color_discard_state(self):
        """Full rule application discarding a raw opposite-color spider (state)."""
        comp = CompositionDiagram([self.q_state, self.p_filler])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, QSpider)
        assert result.phase == self.q_state.phase

    def test_apply_rule_no_match_returns_same_diagram(self):
        """If no pattern is present, apply_rule should return the original diagram."""
        comp = CompositionDiagram([self.swap, self.bs])
        result = apply_rule_to_diagram(self.rule, comp)
        assert result == comp

    def test_apply_rule_multiple_independent_matches(self):
        """State-end and effect-end absorptions both fold in one pass."""
        comp = CompositionDiagram([self.q_state, self.r1, self.sq1, self.q_effect])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        lin_state, quad_state = self._expected_rotation(3.0, self.theta1)
        assert isclose(result.diagrams[0].phase.coeffs[1], lin_state)
        assert isclose(result.diagrams[0].phase.coeffs[2], quad_state)
        assert isclose(result.diagrams[1].phase.coeffs[1], -3.0 / 2.0)

    def test_apply_rule_four_element_chain_one_pass(self):
        """[state, R1, R2, effect]: both boundary pairs fold in a single pass.

        match()'s i += 2 skip-ahead lands exactly on indices 0 and 2 here,
        so (state, R1) and (R2, effect) both match in one call -- the
        middle (R1, R2) pair is never even examined, since it's consumed
        from both sides. No second pass is needed, unlike the Fourier
        rule's overlapping-chain case (there the overlap is on a shared
        node; here it isn't).
        """
        comp = CompositionDiagram([
            self.q_state,
            PhaseRotationGate(self.theta1),
            PhaseRotationGate(self.theta2),
            self.q_effect,
        ])
        result = apply_rule_to_diagram(self.rule, comp)
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
        result = apply_rule_to_diagram(self.rule, tensor)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 3
        assert isinstance(result.diagrams[1], QSpider)

    # -------------------------------------------------------------------------
    # 4. Edge cases
    # -------------------------------------------------------------------------

    def test_symbolic_theta_rotation(self):
        """Folding works with a symbolic rotation angle."""
        theta = Symbol("theta", real=True)
        comp = CompositionDiagram([PhaseRotationGate(theta, parametric=True), self.q_effect])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, QSpider)

        expected = ZxPoly({1: -3.0 / sym_cos(theta), 2: -sym_tan(theta) / 2})
        assert simplify(result.phase.as_expr() - expected.as_expr()) == 0

    def test_degenerate_angle_no_match(self):
        """Theta an odd multiple of pi/2 does not match -- tan/1-over-cos undefined."""
        comp = CompositionDiagram([PhaseRotationGate(pi / 2, parametric=True), self.q_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        assert len(self.rule.match(graph, reg)) == 0

    def test_rotation_degree_too_high_no_match(self):
        """A quadratic (or higher) terminal phase does not match rotation absorption."""
        quadratic_effect = QSpider(1, 0, ZxPoly({2: 1.0, 1: -3.0}))
        comp = CompositionDiagram([self.r1, quadratic_effect])
        graph = to_graph(comp)
        reg = GateRegister()
        reg.build_from_graph(graph)
        assert len(self.rule.match(graph, reg)) == 0

    def test_constant_term_carries_through_rotation(self):
        """A constant term in the terminal's phase survives the rotation fold unchanged."""
        effect_with_const = QSpider(1, 0, ZxPoly({0: 2.0, 1: -3.0}))
        comp = CompositionDiagram([self.r1, effect_with_const])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isclose(result.phase.coeffs[0], 2.0)

    def test_squeezing_any_degree(self):
        """Squeezing absorption isn't restricted to linear phase."""
        cubic_effect = QSpider(1, 0, ZxPoly({3: 1.0, 1: -2.0}))
        comp = CompositionDiagram([self.sq1, cubic_effect])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isclose(result.phase.coeffs[3], 1.0 / 8.0)
        assert isclose(result.phase.coeffs[1], -1.0)

    def test_squeezing_symbolic_tau(self):
        """Squeezing absorption works with a symbolic tau."""
        tau = Symbol("tau", real=True, nonzero=True)
        sq_sym = SqueezingGate(tau, parametric=True)
        comp = CompositionDiagram([sq_sym, self.q_effect])
        result = apply_rule_to_diagram(self.rule, comp)
        expected = ZxPoly({1: -3.0 / tau})
        assert simplify(result.phase.as_expr() - expected.as_expr()) == 0

    def test_squeezing_constant_term_unchanged(self):
        """A constant term is untouched by x -> x/tau (it doesn't depend on x)."""
        effect_with_const = QSpider(1, 0, ZxPoly({0: 5.0, 1: -3.0}))
        comp = CompositionDiagram([self.sq1, effect_with_const])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isclose(result.phase.coeffs[0], 5.0)

    def test_cross_color_discard_arbitrary_degree(self):
        """Cross-color discard has no degree restriction on the vanishing gate."""
        high_degree_filler = PSpider(1, 1, ZxPoly({5: 2.0, 3: -1.0, 0: 4.0}))
        comp = CompositionDiagram([high_degree_filler, self.q_effect])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, QSpider)
        assert result.phase == self.q_effect.phase

    def test_same_color_left_for_fusion_rule(self):
        """A same-color (1,1) filler next to a terminal is left untouched."""
        comp = CompositionDiagram([self.q_filler, self.q_effect])
        result = apply_rule_to_diagram(self.rule, comp)
        assert result == comp

    def test_squeezing_negative_tau(self):
        """Negative tau divides the phase's coefficients as-is (no sign special-casing)."""
        comp = CompositionDiagram([self.sq2, self.q_effect])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isclose(result.phase.coeffs[1], -3.0 / -3.0)


if __name__ == "__main__":
    from cvzx.visualize_base_gates import visualize_before_after

    rule_name = "Terminal Absorption Rule"
    rule = TerminalAbsorptionRule()
    theta = pi / 6

    swap = Swap()
    bs = BeamsplitterGate(pi / 4)
    q_effect = QSpider(1, 0, ZxPoly({1: -3.0}))
    q_state = QSpider(0, 1, ZxPoly({1: 3.0}))
    p_effect = PSpider(1, 0, ZxPoly({1: -3.0}))
    p_filler = PSpider(1, 1, ZxPoly({2: 4.0, 1: 1.0}))

    # Test 1: rotation folded into a QSpider effect
    comp1 = CompositionDiagram([PhaseRotationGate(theta), QSpider(1, 0, ZxPoly({1: -3.0}))])
    comp1_after = apply_rule_to_diagram(rule, comp1)
    visualize_before_after(comp1, comp1_after, "Rotation into Effect", rule_name)

    # Test 2: rotation folded into a QSpider state
    comp2 = CompositionDiagram([QSpider(0, 1, ZxPoly({1: 3.0})), PhaseRotationGate(theta)])
    comp2_after = apply_rule_to_diagram(rule, comp2)
    visualize_before_after(comp2, comp2_after, "Rotation into State", rule_name)

    # Test 3: squeezing folded into a QSpider effect
    comp3 = CompositionDiagram([SqueezingGate(2.0), QSpider(1, 0, ZxPoly({1: -3.0}))])
    comp3_after = apply_rule_to_diagram(rule, comp3)
    visualize_before_after(comp3, comp3_after, "Squeezing into Effect", rule_name)

    # Test 4: squeezing folded into a PSpider effect
    comp4 = CompositionDiagram([SqueezingGate(2.0), PSpider(1, 0, ZxPoly({1: -3.0}))])
    comp4_after = apply_rule_to_diagram(rule, comp4)
    visualize_before_after(comp4, comp4_after, "Squeezing into PSpider Effect", rule_name)

    # Test 5: cross-color discard into a QSpider effect
    comp5 = CompositionDiagram([PSpider(1, 1, ZxPoly({2: 4.0, 1: 1.0})), QSpider(1, 0, ZxPoly({1: -3.0}))])
    comp5_after = apply_rule_to_diagram(rule, comp5)
    visualize_before_after(comp5, comp5_after, "Cross-Color Discard", rule_name)

    # Test 6: both ends of a composition fold in one pass
    comp6 = CompositionDiagram([
        QSpider(0, 1, ZxPoly({1: 3.0})),
        PhaseRotationGate(theta),
        SqueezingGate(2.0),
        QSpider(1, 0, ZxPoly({1: -3.0})),
    ])
    comp6_after = apply_rule_to_diagram(rule, comp6)
    visualize_before_after(comp6, comp6_after, "Both Ends in One Pass", rule_name)

    # Test 7: pattern nested inside a tensor branch
    inner_comp = CompositionDiagram([PhaseRotationGate(theta), QSpider(1, 0, ZxPoly({1: -3.0}))])
    tensor = TensorDiagram([swap, inner_comp, bs])
    tensor_after = apply_rule_to_diagram(rule, tensor)
    visualize_before_after(tensor, tensor_after, "Nested in Tensor", rule_name)

    # Test 8: diagram unchanged because the rule doesn't apply
    diagram = CompositionDiagram([swap, bs])
    diagram_after = apply_rule_to_diagram(rule, diagram)
    visualize_before_after(diagram, diagram_after, "No reduction", rule_name)

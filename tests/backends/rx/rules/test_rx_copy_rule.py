"""Unit tests for the copy rewrite rule using the rustworkx graph formalism.

These tests verify the CopyRule implementation on rustworkx graphs, including matching,
application, flattening, and nested structures. The copy rule copies a spider
with arity (0,1) or (1,0) through a spider with arity (1,n) or (n,1),
producing n copies in a tensor diagram.
"""

import unittest
from math import pi

from sympy import I, symbols

from cvzx.ir.base import (
    CompositionDiagram,
    ContractedDiagram,
    Diagram,
    Fourier,
    PSpider,
    QSpider,
    Swap,
    TensorDiagram,
    VoidDiagram,
    ZxPoly,
)
from cvzx.ir.gates import BeamsplitterGate, ControlledSumGate, PhaseRotationGate, SqueezingGate
from cvzx.backends.rx.graph import to_diagram, to_graph
from cvzx.backends.rx.rules import CopyRule, apply_rule_to_diagram
from cvzx.utils.helpers import expand_two_mode_gates


class TestCopyRule(unittest.TestCase):
    """Test suite for CopyRule using rustworkx graphs."""

    def setUp(self):
        """Create common objects used in many tests."""
        # Phase polynomials
        self.phase_x2 = ZxPoly({1: 2})  # g ∈ R₁[X] (degree 1) -- coefficient 2 on the linear (x^1) term
        self.phase_x = ZxPoly({1: 3})  # g ∈ R₁[X] (degree 1)
        self.phase_x3 = ZxPoly({3: 2})  # NOT in R₁[X] (degree 3)
        self.phase_mixed = ZxPoly({1: 2, 2: 3})  # NOT in R₁[X] (degree 2)
        self.zero_phase = ZxPoly({})

        # Q-Spiders (copied spiders with phase in R₁[X])
        self.q_0_1 = QSpider(0, 1, self.phase_x2)  # Q(g, 0, 1) - input spider
        self.q_1_0 = QSpider(1, 0, self.phase_x)  # Q(g, 1, 0) - output spider
        self.q_0_1_zero = QSpider(0, 1, self.zero_phase)  # Q(0, 0, 1)

        # Q-Spiders with phase NOT in R₁[X]
        self.q_0_1_bad = QSpider(0, 1, self.phase_x3)  # Q(g, 0, 1) with g ∉ R₁[X]
        self.q_1_0_bad = QSpider(1, 0, self.phase_mixed)  # Q(g, 1, 0) with g ∉ R₁[X]

        # P-Spiders (copied spiders with phase in R₁[X])
        self.p_0_1 = PSpider(0, 1, self.phase_x2)  # P(g, 0, 1) - input spider
        self.p_1_0 = PSpider(1, 0, self.phase_x)  # P(g, 1, 0) - output spider
        self.p_0_1_zero = PSpider(0, 1, self.zero_phase)  # P(0, 0, 1)

        # P-Spiders with phase NOT in R₁[X]
        self.p_0_1_bad = PSpider(0, 1, self.phase_x3)  # P(g, 0, 1) with g ∉ R₁[X]
        self.p_1_0_bad = PSpider(1, 0, self.phase_mixed)  # P(g, 1, 0) with g ∉ R₁[X]

        # Disappearing spiders (any phase, arity 1→n or n→1)
        self.phi_any = ZxPoly({2: 2, 3: 1})  # Any polynomial

        # P(φ, 1, n) - disappears in case 1
        self.p_1_2 = PSpider(1, 2, self.phi_any)  # P(φ, 1, 2)
        self.p_1_3 = PSpider(1, 3, self.phi_any)  # P(φ, 1, 3)
        self.p_1_4 = PSpider(1, 4, self.phi_any)  # P(φ, 1, 4)

        # P(φ, n, 1) - disappears in case 2
        self.p_2_1 = PSpider(2, 1, self.phi_any)  # P(φ, 2, 1)
        self.p_3_1 = PSpider(3, 1, self.phi_any)  # P(φ, 3, 1)
        self.p_4_1 = PSpider(4, 1, self.phi_any)  # P(φ, 4, 1)

        # Q(φ, 1, n) - disappears in case 3
        self.q_1_2 = QSpider(1, 2, self.phi_any)  # Q(φ, 1, 2)
        self.q_1_3 = QSpider(1, 3, self.phi_any)  # Q(φ, 1, 3)

        # Q(φ, n, 1) - disappears in case 4
        self.q_2_1 = QSpider(2, 1, self.phi_any)  # Q(φ, 2, 1)
        self.q_3_1 = QSpider(3, 1, self.phi_any)  # Q(φ, 3, 1)

        # Other gates (not copy-able)
        self.fourier = Fourier()
        self.swap = Swap()
        self.ph_rot = PhaseRotationGate(pi / 4)
        self.bs = BeamsplitterGate(pi / 4)
        self.bs_c2 = BeamsplitterGate(pi / 4)
        self.bs_c3 = BeamsplitterGate(pi / 4)
        self.sq = SqueezingGate(2.0)

        # Complex nested fixtures
        # comp1: a copy pattern (case 1) followed by an inert 2-wire gate.
        self.copy_comp1_q = QSpider(0, 1, self.phase_x2)
        self.copy_comp1 = CompositionDiagram([self.copy_comp1_q, PSpider(1, 2, self.phi_any), self.bs])

        # comp2: a copy pattern (case 3) nested inside a tensor branch inside a composition.
        self.copy_comp2_p = PSpider(0, 1, self.phase_x2)
        self.copy_comp2 = CompositionDiagram([
            QSpider(2, 2, self.phi_any),
            TensorDiagram([
                self.bs_c2,
                CompositionDiagram([self.copy_comp2_p, QSpider(1, 2, self.phi_any)]),
            ]),
        ])

        # comp3: three copy patterns (case 3, case 1, case 2) at two nesting depths.
        self.copy_comp3_p_lead = PSpider(0, 1, self.phase_x2)
        self.copy_comp3_q_deep = QSpider(0, 1, self.phase_x2)
        self.copy_comp3_q_deep2 = QSpider(1, 0, self.phase_x)
        self.copy_comp3 = CompositionDiagram([
            CompositionDiagram([self.copy_comp3_p_lead, QSpider(1, 3, self.phi_any)]),
            QSpider(3, 6, self.phi_any),
            TensorDiagram([
                self.bs_c3,
                TensorDiagram([
                    CompositionDiagram([self.copy_comp3_q_deep, PSpider(1, 2, self.phi_any)]),
                    CompositionDiagram([PSpider(4, 1, self.phi_any), self.copy_comp3_q_deep2]),
                ]),
            ]),
        ])

        # tensor1/2/3: a growing tensor of independent copy-pattern branches
        self.copy_tensor1_q = QSpider(0, 1, self.phase_x2)
        self.copy_tensor1 = TensorDiagram([
            CompositionDiagram([self.copy_tensor1_q, PSpider(1, 2, self.phi_any)]),
            self.copy_comp1,
        ])

        self.copy_tensor2_q = QSpider(0, 1, self.phase_x2)
        self.copy_tensor2 = TensorDiagram([
            CompositionDiagram([self.copy_tensor2_q, PSpider(1, 2, self.phi_any)]),
            self.copy_comp1,
            QSpider(5, 3, self.phi_any),
            self.copy_comp2,
        ])

        self.copy_tensor3_q = QSpider(0, 1, self.phase_x2)
        self.copy_tensor3 = TensorDiagram([
            CompositionDiagram([self.copy_tensor3_q, PSpider(1, 2, self.phi_any)]),
            self.copy_comp1,
            QSpider(5, 3, self.phi_any),
            self.copy_comp2,
            PSpider(3, 2, self.phi_any),
            self.copy_comp3,
        ])

        # Rule instance
        self.rule = CopyRule()

    # -------------------------------------------------------------------------
    # 1. Testing _is_in_R1()
    # -------------------------------------------------------------------------

    def test_is_in_R1_degree_0(self):
        """Zero degree polynomial is in R₁[X]."""
        phase = ZxPoly({0: 2})
        assert self.rule.is_in_R1(phase) is True

    def test_is_in_R1_degree_1(self):
        """Degree 1 polynomial is in R₁[X]."""
        phase = ZxPoly({1: 3})
        assert self.rule.is_in_R1(phase) is True

    def test_is_in_R1_degree_2(self):
        """Degree 2 polynomial is NOT in R₁[X]."""
        phase = ZxPoly({2: 2})
        assert self.rule.is_in_R1(phase) is False

    def test_is_in_R1_degree_3(self):
        """Degree 3 polynomial is NOT in R₁[X]."""
        phase = ZxPoly({3: 2})
        assert self.rule.is_in_R1(phase) is False

    def test_is_in_R1_mixed(self):
        """Mixed polynomial with degree > 1 is NOT in R₁[X]."""
        phase = ZxPoly({1: 2, 2: 3})
        assert self.rule.is_in_R1(phase) is False

    def test_is_in_R1_zero(self):
        """Zero polynomial is in R₁[X]."""
        phase = ZxPoly({})
        assert self.rule.is_in_R1(phase)

    # -------------------------------------------------------------------------
    # 2. Testing _check_pair() and match()
    # -------------------------------------------------------------------------

    def test_match_case_1_p_1_n_q_0_1(self):
        """Case 1: P(φ, 1, n) ∘ Q(g, 0, 1) → Q(g, 0, 1) ⊗ ... (n times)."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["copy_spider_id"] == self.q_0_1.id
        assert matches[0]["disappearing_spider_id"] == self.p_1_2.id
        assert matches[0]["n_copies"] == 2
        assert matches[0]["copy_spider_type"] == "Q"
        assert matches[0]["copy_spider_phase"] == self.phase_x2
        assert matches[0]["indices"] == [0, 1]

    def test_match_case_1_p_1_3_q_0_1(self):
        """Case 1: P(φ, 1, 3) ∘ Q(g, 0, 1) → Q(g, 0, 1) ⊗ ... (3 times)."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_3])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["n_copies"] == 3

    def test_match_case_2_q_1_0_p_n_1(self):
        """Case 2: Q(g, 1, 0) ∘ P(φ, n, 1) → Q(g, 1, 0) ⊗ ... (n times)."""
        comp = CompositionDiagram([self.p_2_1, self.q_1_0])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["copy_spider_id"] == self.q_1_0.id
        assert matches[0]["disappearing_spider_id"] == self.p_2_1.id
        assert matches[0]["n_copies"] == 2
        assert matches[0]["copy_spider_type"] == "Q"
        assert matches[0]["copy_spider_phase"] == self.phase_x

    def test_match_case_3_q_1_n_p_0_1(self):
        """Case 3: Q(φ, 1, n) ∘ P(g, 0, 1) → P(g, 0, 1) ⊗ ... (n times)."""
        comp = CompositionDiagram([self.p_0_1, self.q_1_2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["copy_spider_id"] == self.p_0_1.id
        assert matches[0]["disappearing_spider_id"] == self.q_1_2.id
        assert matches[0]["n_copies"] == 2
        assert matches[0]["copy_spider_type"] == "P"
        assert matches[0]["copy_spider_phase"] == self.phase_x2

    def test_match_case_4_p_1_0_q_n_1(self):
        """Case 4: P(g, 1, 0) ∘ Q(φ, n, 1) → P(g, 1, 0) ⊗ ... (n times)."""
        comp = CompositionDiagram([self.q_2_1, self.p_1_0])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["copy_spider_id"] == self.p_1_0.id
        assert matches[0]["disappearing_spider_id"] == self.q_2_1.id
        assert matches[0]["n_copies"] == 2
        assert matches[0]["copy_spider_type"] == "P"
        assert matches[0]["copy_spider_phase"] == self.phase_x

    def test_match_no_copy_bad_phase(self):
        """No match when copied spider's phase is NOT in R₁[X]."""
        comp = CompositionDiagram([self.q_0_1_bad, self.p_1_2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_no_copy_bad_phase_case_2(self):
        """No match when copied spider's phase is NOT in R₁[X] (case 2)."""
        comp = CompositionDiagram([self.p_2_1, self.q_1_0_bad])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_no_copy_bad_phase_case_3(self):
        """No match when copied spider's phase is NOT in R₁[X] (case 3)."""
        comp = CompositionDiagram([self.p_0_1_bad, self.q_1_2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_no_copy_bad_phase_case_4(self):
        """No match when copied spider's phase is NOT in R₁[X] (case 4)."""
        comp = CompositionDiagram([self.q_2_1, self.p_1_0_bad])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_no_copy_not_spider(self):
        """No match when disappearing spider is not a spider."""
        comp = CompositionDiagram([self.q_0_1, self.fourier])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_no_copy_wrong_arity(self):
        """No match when the adjacent spider's arity isn't 1->n or n->1 (here 1->0)."""
        p_1_0_shape = PSpider(1, 0, self.phi_any)
        comp = CompositionDiagram([self.q_0_1, p_1_0_shape])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_copy_with_extra_elements(self):
        """Match when the composition has extra content trailing the pattern."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_2, self.swap])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["copy_spider_id"] == self.q_0_1.id
        assert matches[0]["disappearing_spider_id"] == self.p_1_2.id
        assert matches[0]["n_copies"] == 2
        assert matches[0]["indices"] == [0, 1]

    def test_match_nested_in_tensor(self):
        """Match copy pattern inside a tensor diagram."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_2])
        tensor = TensorDiagram([self.fourier, comp, self.swap])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] in graph.registry.composition_nodes

    def test_match_nested_in_composition(self):
        """Match copy pattern inside a nested composition (pattern must lead)."""
        inner = CompositionDiagram([self.q_0_1, self.p_1_2])
        outer = CompositionDiagram([inner, self.swap])
        graph = to_graph(outer)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == inner.id

    def test_match_nested_in_contracted(self):
        """Match copy pattern inside a composition inside a ContractedDiagram."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_2])
        contracted = ContractedDiagram(comp, self.swap, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id

    def test_match_multiple_copy_patterns(self):
        """Match multiple copy patterns chained in one composition."""
        comp = CompositionDiagram([
            self.q_0_1,
            self.p_1_2,  # Pattern 1
            self.p_2_1,
            self.q_1_0,  # Pattern 2
        ])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 2

    # -------------------------------------------------------------------------
    # 3. Testing apply_single()
    # -------------------------------------------------------------------------

    def test_apply_single_case_1_p_1_2_q_0_1(self):
        """Case 1: P(φ, 1, 2) ∘ Q(g, 0, 1) → Q(g, 0, 1) ⊗ Q(g, 0, 1)."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert all(isinstance(d, QSpider) and d.phase == self.phase_x2 for d in result.diagrams)
        assert all(d.num_inputs == 0 and d.num_outputs == 1 for d in result.diagrams)

    def test_apply_single_case_1_p_1_3_q_0_1(self):
        """Case 1: P(φ, 1, 3) ∘ Q(g, 0, 1) → Q(g, 0, 1) ⊗ Q(g, 0, 1) ⊗ Q(g, 0, 1)."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_3])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 3
        assert all(isinstance(d, QSpider) and d.phase == self.phase_x2 for d in result.diagrams)

    def test_apply_single_case_2_q_1_0_p_2_1(self):
        """Case 2: Q(g, 1, 0) ∘ P(φ, 2, 1) → Q(g, 1, 0) ⊗ Q(g, 1, 0)."""
        comp = CompositionDiagram([self.p_2_1, self.q_1_0])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert all(isinstance(d, QSpider) and d.phase == self.phase_x for d in result.diagrams)
        assert all(d.num_inputs == 1 and d.num_outputs == 0 for d in result.diagrams)

    def test_apply_single_case_3_q_1_2_p_0_1(self):
        """Case 3: Q(φ, 1, 2) ∘ P(g, 0, 1) → P(g, 0, 1) ⊗ P(g, 0, 1)."""
        comp = CompositionDiagram([self.p_0_1, self.q_1_2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert all(isinstance(d, PSpider) and d.phase == self.phase_x2 for d in result.diagrams)
        assert all(d.num_inputs == 0 and d.num_outputs == 1 for d in result.diagrams)

    def test_apply_single_case_4_p_1_0_q_2_1(self):
        """Case 4: P(g, 1, 0) ∘ Q(φ, 2, 1) → P(g, 1, 0) ⊗ P(g, 1, 0)."""
        comp = CompositionDiagram([self.q_2_1, self.p_1_0])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert all(isinstance(d, PSpider) and d.phase == self.phase_x for d in result.diagrams)
        assert all(d.num_inputs == 1 and d.num_outputs == 0 for d in result.diagrams)

    def test_apply_single_nested_in_tensor(self):
        """Apply copy rule on pattern inside a tensor."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_2])
        tensor = TensorDiagram([self.fourier, comp, self.swap])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 4
        assert result.diagrams[0] == self.fourier
        assert isinstance(result.diagrams[1], QSpider)
        assert isinstance(result.diagrams[2], QSpider)
        assert result.diagrams[3] == self.swap

    def test_apply_single_nested_in_composition(self):
        """Apply copy rule on pattern inside a nested composition (pattern leads)."""
        inner = CompositionDiagram([self.q_0_1, self.p_1_2])
        outer = CompositionDiagram([inner, self.swap])
        graph = to_graph(outer)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[0], TensorDiagram)
        assert len(result.diagrams[0].diagrams) == 2
        assert result.diagrams[1] == self.swap

    def test_apply_single_nested_in_contracted(self):
        """Apply copy rule on pattern inside a composition inside a ContractedDiagram."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_2])
        contracted = ContractedDiagram(comp, self.swap, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, ContractedDiagram)
        assert isinstance(result.first, TensorDiagram)
        assert len(result.first.diagrams) == 2
        assert result.second == self.swap

    def test_apply_single_copy_arity_0_1_preserved(self):
        """Verify a (0,1) copy spider's arity is preserved across all copies (n=4)."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_4])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 4
        for d in result.diagrams:
            assert d.num_inputs == 0
            assert d.num_outputs == 1

    def test_apply_single_copy_arity_1_0_preserved(self):
        """Verify a (1,0) copy spider's arity is preserved across all copies (n=4)."""
        comp = CompositionDiagram([self.p_4_1, self.q_1_0])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 4
        for d in result.diagrams:
            assert d.num_inputs == 1
            assert d.num_outputs == 0

    # -------------------------------------------------------------------------
    # 4. Testing apply_rule() (full application)
    # -------------------------------------------------------------------------

    def test_apply_rule_case_1(self):
        """Full rule application case 1."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_2])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert all(isinstance(d, QSpider) for d in result.diagrams)

    def test_apply_rule_case_2(self):
        """Full rule application case 2."""
        comp = CompositionDiagram([self.p_2_1, self.q_1_0])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert all(isinstance(d, QSpider) for d in result.diagrams)

    def test_apply_rule_case_3(self):
        """Full rule application case 3."""
        comp = CompositionDiagram([self.p_0_1, self.q_1_2])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert all(isinstance(d, PSpider) for d in result.diagrams)

    def test_apply_rule_case_4(self):
        """Full rule application case 4."""
        comp = CompositionDiagram([self.q_2_1, self.p_1_0])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert all(isinstance(d, PSpider) for d in result.diagrams)

    def test_apply_rule_multiple_patterns(self):
        """Full rule application with multiple chained copy patterns."""
        comp = CompositionDiagram([
            self.q_0_1,
            self.p_1_2,  # Pattern 1
            self.p_2_1,
            self.q_1_0,  # Pattern 2
        ])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)

        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[0], TensorDiagram)
        assert len(result.diagrams[0].diagrams) == 2
        assert isinstance(result.diagrams[1], TensorDiagram)
        assert len(result.diagrams[1].diagrams) == 2

    def test_apply_rule_no_match(self):
        """Full rule application when no match exists."""
        comp = CompositionDiagram([self.fourier, self.ph_rot])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
        assert result == comp

    def test_apply_rule_bad_phase_no_match(self):
        """Full rule application with bad phase - no match."""
        comp = CompositionDiagram([self.q_0_1_bad, self.p_1_2])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
        assert result == comp

    def test_apply_rule_deeply_nested(self):
        """Full rule application on a deeply nested structure."""
        inner = CompositionDiagram([self.q_0_1, self.p_1_2])
        middle = CompositionDiagram([inner, Swap()])
        outer = TensorDiagram([self.swap, middle, self.bs])
        graph = to_graph(outer)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 3
        assert result.diagrams[0] == self.swap
        assert isinstance(result.diagrams[1], CompositionDiagram)
        assert len(result.diagrams[1].diagrams) == 2
        assert isinstance(result.diagrams[1].diagrams[0], TensorDiagram)
        assert len(result.diagrams[1].diagrams[0].diagrams) == 2
        assert result.diagrams[1].diagrams[1] == Swap()
        assert result.diagrams[2] == self.bs

    def test_apply_rule_copy_with_zero_phase(self):
        """Copy rule with zero phase in copied spider."""
        comp = CompositionDiagram([self.q_0_1_zero, self.p_1_2])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert all(isinstance(d, QSpider) and d.phase == self.zero_phase for d in result.diagrams)

    def test_apply_rule_with_containers_only(self):
        """Copy rule where only the container remains."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_2])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2

    # -------------------------------------------------------------------------
    # 5. Edge Cases and Integration Tests
    # -------------------------------------------------------------------------

    def test_copy_spider_phase_in_R1_with_symbolic(self):
        """Copied spider with symbolic phase in R₁[X] (degree 1)."""
        a, b = symbols("a b", real=True)
        phase = ZxPoly({1: a, 0: b})  # a*x + b (degree 1)
        q = QSpider(0, 1, phase, True)
        comp = CompositionDiagram([q, self.p_1_2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["copy_spider_phase"] == phase

    def test_copy_spider_phase_not_in_R1_with_symbolic(self):
        """Copied spider with symbolic phase NOT in R₁[X] (degree 2)."""
        a = symbols("a", real=True)
        phase = ZxPoly({2: a})  # a*x² (degree 2)
        q = QSpider(0, 1, phase, True)
        comp = CompositionDiagram([q, self.p_1_2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_copy_preserves_phase(self):
        """Verify copied spiders have exactly the same phase."""
        comp = CompositionDiagram([self.q_0_1, self.p_1_3])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        for d in result.diagrams:
            assert isinstance(d, QSpider)
            assert d.phase == self.phase_x2

    def test_copy_disappearing_spider_phase_ignored(self):
        """The disappearing spider's phase can be anything."""
        phi1 = ZxPoly({2: 2, 3: 1})
        phi2 = ZxPoly({5: 7, 10: 3})
        p1 = PSpider(1, 2, phi1)
        p2 = PSpider(1, 2, phi2)

        comp1 = CompositionDiagram([self.q_0_1, p1])
        comp2 = CompositionDiagram([self.q_0_1, p2])

        graph1 = to_graph(comp1)
        matches1 = self.rule.match(graph1)

        graph2 = to_graph(comp2)
        matches2 = self.rule.match(graph2)

        assert len(matches1) == 1
        assert len(matches2) == 1

    def test_copy_rule_with_complex_phase(self):
        """Disappearing spider with complex phase."""
        phi_complex = ZxPoly({1: 1 + I, 2: 2 - I})
        p = PSpider(1, 2, phi_complex)
        comp = CompositionDiagram([self.q_0_1, p])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)

        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert all(isinstance(d, QSpider) for d in result.diagrams)

    # -------------------------------------------------------------------------
    # 6. Complex nested structures
    # -------------------------------------------------------------------------

    def test_match_complex_comp1(self):
        """comp1 has exactly one copy pattern, trailed by an inert gate."""
        graph = to_graph(self.copy_comp1)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["copy_spider_id"] == self.copy_comp1_q.id
        assert matches[0]["n_copies"] == 2

    def test_match_complex_comp2(self):
        """comp2 has exactly one copy pattern, nested inside a tensor branch."""
        graph = to_graph(self.copy_comp2)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["copy_spider_id"] == self.copy_comp2_p.id
        assert matches[0]["copy_spider_type"] == "P"

    def test_match_complex_comp3(self):
        """comp3 has three copy patterns split across two nesting depths."""
        graph = to_graph(self.copy_comp3)
        matches = self.rule.match(graph)
        assert len(matches) == 3
        copy_spider_ids = {m["copy_spider_id"] for m in matches}
        assert copy_spider_ids == {
            self.copy_comp3_p_lead.id,
            self.copy_comp3_q_deep.id,
            self.copy_comp3_q_deep2.id,
        }

    def test_match_complex_tensor1(self):
        """tensor1: the shallow branch's pattern plus comp1's pattern."""
        graph = to_graph(self.copy_tensor1)
        matches = self.rule.match(graph)
        assert len(matches) == 2
        copy_spider_ids = {m["copy_spider_id"] for m in matches}
        assert copy_spider_ids == {self.copy_tensor1_q.id, self.copy_comp1_q.id}

    def test_match_complex_tensor2(self):
        """tensor2: tensor1's two matches plus comp2's nested match."""
        graph = to_graph(self.copy_tensor2)
        matches = self.rule.match(graph)
        assert len(matches) == 3
        copy_spider_ids = {m["copy_spider_id"] for m in matches}
        assert copy_spider_ids == {self.copy_tensor2_q.id, self.copy_comp1_q.id, self.copy_comp2_p.id}

    def test_match_complex_tensor3(self):
        """tensor3: tensor2's three matches plus comp3's three matches."""
        graph = to_graph(self.copy_tensor3)
        matches = self.rule.match(graph)
        assert len(matches) == 6
        copy_spider_ids = {m["copy_spider_id"] for m in matches}
        assert copy_spider_ids == {
            self.copy_tensor3_q.id,
            self.copy_comp1_q.id,
            self.copy_comp2_p.id,
            self.copy_comp3_p_lead.id,
            self.copy_comp3_q_deep.id,
            self.copy_comp3_q_deep2.id,
        }

    def test_apply_rule_complex_comp1(self):
        """Full rule application to comp1 removes the pattern, keeps the trailing gate."""
        result = apply_rule_to_diagram(self.rule, self.copy_comp1)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[0], TensorDiagram)
        assert len(result.diagrams[0].diagrams) == 2
        assert all(isinstance(d, QSpider) and d.phase == self.phase_x2 for d in result.diagrams[0].diagrams)
        assert result.diagrams[1] == self.bs

    def test_apply_rule_complex_comp2(self):
        """Full rule application to comp2 resolves the nested tensor-branch pattern."""
        result = apply_rule_to_diagram(self.rule, self.copy_comp2)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        tensor_new = result.diagrams[1]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 3
        assert tensor_new.diagrams[0] == self.bs_c2
        assert all(isinstance(d, PSpider) and d.phase == self.phase_x2 for d in tensor_new.diagrams[1:])

    def test_apply_rule_complex_comp3(self):
        """Full rule application to comp3 resolves all three patterns in one pass."""
        result = apply_rule_to_diagram(self.rule, self.copy_comp3)
        assert isinstance(result, CompositionDiagram)
        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

    def test_apply_rule_complex_tensor1(self):
        """Full rule application to tensor1: tensors flatten, comp1's shape survives."""
        result = apply_rule_to_diagram(self.rule, self.copy_tensor1)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 3
        assert isinstance(result.diagrams[0], QSpider)
        assert isinstance(result.diagrams[1], QSpider)
        assert isinstance(result.diagrams[2], CompositionDiagram)
        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

    def test_apply_rule_complex_tensor2(self):
        """Full rule application to tensor2 resolves all three independent patterns."""
        result = apply_rule_to_diagram(self.rule, self.copy_tensor2)
        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

    def test_apply_rule_complex_tensor3(self):
        """Full rule application to tensor3 resolves all six independent patterns."""
        result = apply_rule_to_diagram(self.rule, self.copy_tensor3)
        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

    def test_apply_rule_complex_contracted(self):
        """Full rule application inside a ContractedDiagram wrapping tensor2."""
        q_spider_large = QSpider(10, 10, self.phi_any)
        contracted = ContractedDiagram(
            self.copy_tensor2, q_spider_large, [1, 2, 3], [4, 5, 6], [0, 1, 2, 6], [1, 3, 5, 7]
        )
        result = apply_rule_to_diagram(self.rule, contracted)
        assert isinstance(result, ContractedDiagram)
        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

    # -------------------------------------------------------------------------
    # 7. Cross-container matches
    # -------------------------------------------------------------------------

    def _expanded_csum_with_states(self, control: Diagram, target: Diagram, control_mode: int = 2) -> Diagram:
        """Build TensorDiagram([control, target]).compose(CSUM.expand())."""
        csum = ControlledSumGate(control=control_mode, target=3 - control_mode)
        comp = CompositionDiagram([TensorDiagram([control, target]), csum])
        return expand_two_mode_gates(comp)

    def test_match_cross_container_csum_control(self):
        """Both states form copy-able pairs into their respective ContractedDiagram half."""
        control_state = PSpider(0, 1, self.phase_x2)
        target_state = QSpider(0, 1, self.phase_x)
        comp = self._expanded_csum_with_states(control_state, target_state)
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 2
        by_copy_id = {m["copy_spider_id"]: m for m in matches}
        assert set(by_copy_id) == {control_state.id, target_state.id}
        for m in by_copy_id.values():
            assert m["same_parent"] is False
            assert m["copy_container_type"] == "tensor"
            assert m["disappear_container_type"] == "contracted"
            assert m["n_copies"] == 1

    def test_match_cross_container_same_color_no_match(self):
        """Same-color control/target states never match CopyRule -- that's fusion."""
        control_state = QSpider(0, 1, self.phase_x2)
        target_state = PSpider(0, 1, self.phase_x)
        comp = self._expanded_csum_with_states(control_state, target_state)
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_match_cross_container_bad_phase_no_match(self):
        """A control state with phase not in R1[X] still can't be copied."""
        control_state = PSpider(0, 1, self.phase_x3)
        target_state = QSpider(0, 1, self.phase_x)
        comp = self._expanded_csum_with_states(control_state, target_state)
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["copy_spider_id"] == target_state.id

    def test_apply_rule_cross_container_csum_control(self):
        """Applying the rule pulls BOTH states inside the ContractedDiagram."""
        control_state = PSpider(0, 1, self.phase_x2)
        target_state = QSpider(0, 1, self.phase_x)
        comp = self._expanded_csum_with_states(control_state, target_state)

        result = apply_rule_to_diagram(self.rule, comp)

        assert isinstance(result, CompositionDiagram)
        assert result.num_inputs == 0
        assert result.num_outputs == 2

        outer_tensor = result.diagrams[0]
        assert isinstance(outer_tensor, TensorDiagram)
        assert len(outer_tensor.diagrams) == 2
        assert all(
            isinstance(d, VoidDiagram) and d.num_inputs == 0 and d.num_outputs == 1 for d in outer_tensor.diagrams
        )

        contracted = result.diagrams[1]
        assert isinstance(contracted, ContractedDiagram)

        assert isinstance(contracted.first, TensorDiagram)
        assert len(contracted.first.diagrams) == 3
        void_first = [d for d in contracted.first.diagrams if isinstance(d, VoidDiagram)]
        p_spiders = [d for d in contracted.first.diagrams if isinstance(d, PSpider)]
        assert len(void_first) == 1
        assert void_first[0].num_inputs == 1
        assert void_first[0].num_outputs == 0
        assert len(p_spiders) == 2
        assert all(d.num_inputs == 0 and d.num_outputs == 1 and d.phase == self.phase_x2 for d in p_spiders)

        assert isinstance(contracted.second, TensorDiagram)
        assert len(contracted.second.diagrams) == 3
        void_second = [d for d in contracted.second.diagrams if isinstance(d, VoidDiagram)]
        q_spiders = [d for d in contracted.second.diagrams if isinstance(d, QSpider)]
        assert len(void_second) == 1
        assert void_second[0].num_inputs == 1
        assert void_second[0].num_outputs == 0
        assert len(q_spiders) == 2
        assert all(d.phase == self.phase_x for d in q_spiders)
        assert {(d.num_inputs, d.num_outputs) for d in q_spiders} == {(0, 1), (1, 0)}

        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

    def test_apply_rule_cross_container_control_mode_one(self):
        """Same cross-container reduction with the control on mode 1 instead of 2."""
        control_state = QSpider(0, 1, self.phase_x2)
        target_state = PSpider(0, 1, self.phase_x)
        comp = self._expanded_csum_with_states(control_state, target_state, control_mode=1)

        result = apply_rule_to_diagram(self.rule, comp)

        assert isinstance(result, CompositionDiagram)
        assert result.num_inputs == 0
        assert result.num_outputs == 2
        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

    def test_apply_rule_cross_container_no_match_leaves_diagram_unchanged(self):
        """The same-color scenario is left untouched by CopyRule."""
        control_state = QSpider(0, 1, self.phase_x2)
        target_state = PSpider(0, 1, self.phase_x)
        comp = self._expanded_csum_with_states(control_state, target_state)
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, CompositionDiagram)
        assert result.num_inputs == 0
        assert result.num_outputs == 2

    # -------------------------------------------------------------------------
    # 8. Chain-chasing through identity spiders AND `Swap`
    # -------------------------------------------------------------------------

    def test_match_cross_container_contracted_input_direction_through_identities(self):
        """Input-direction chase through a ContractedDiagram."""
        control_state = PSpider(0, 1, self.phase_x2)
        target_state = QSpider(0, 1, self.phase_x)
        id1 = QSpider(1, 1, self.zero_phase)
        id2 = PSpider(1, 1, self.zero_phase)
        csum = ControlledSumGate(control=2, target=1)
        comp = CompositionDiagram([
            TensorDiagram([
                CompositionDiagram([control_state, id1, id2]),
                target_state,
            ]),
            csum,
        ])
        expanded = expand_two_mode_gates(comp)
        graph = to_graph(expanded)
        matches = self.rule.match(graph)
        by_copy_id = {m["copy_spider_id"]: m for m in matches}
        assert control_state.id in by_copy_id
        control_match = by_copy_id[control_state.id]
        assert control_match["same_parent"] is False
        assert control_match["disappear_container_type"] == "contracted"
        assert set(control_match["identity_chain"]) == {id1.id, id2.id}
        target_match = by_copy_id[target_state.id]
        assert target_match["identity_chain"] == []

    def test_apply_rule_cross_container_contracted_input_direction_through_identities(self):
        """Applying the rule resolves both pairs."""
        control_state = PSpider(0, 1, self.phase_x2)
        target_state = QSpider(0, 1, self.phase_x)
        id1 = QSpider(1, 1, self.zero_phase)
        id2 = PSpider(1, 1, self.zero_phase)
        csum = ControlledSumGate(control=2, target=1)
        comp = CompositionDiagram([
            TensorDiagram([
                CompositionDiagram([control_state, id1, id2]),
                target_state,
            ]),
            csum,
        ])
        expanded = expand_two_mode_gates(comp)

        result = apply_rule_to_diagram(self.rule, expanded)

        assert isinstance(result, CompositionDiagram)
        assert result.num_inputs == 0
        assert result.num_outputs == 2
        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

    def test_match_cross_container_contracted_measurement_direction_through_identities(self):
        """Measurement-direction chase through a ContractedDiagram."""
        hub = QSpider(2, 1, self.phi_any)
        partner = PSpider(1, 1, self.zero_phase)
        contracted = ContractedDiagram(hub, partner, [], [], [0], [0])
        id1 = QSpider(1, 1, self.zero_phase)
        id2 = PSpider(1, 1, self.zero_phase)
        effect = PSpider(1, 0, self.phase_x)

        comp = CompositionDiagram([contracted, id1, id2, effect])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["copy_spider_id"] == effect.id
        assert matches[0]["disappearing_spider_id"] == hub.id
        assert matches[0]["same_parent"] is False
        assert matches[0]["disappear_container_type"] == "contracted"
        assert set(matches[0]["identity_chain"]) == {id1.id, id2.id}
        assert matches[0]["n_copies"] == 1

    def test_apply_rule_cross_container_contracted_measurement_direction_through_identities(self):
        """Applying the measurement-direction match."""
        hub = QSpider(2, 1, self.phi_any)
        partner = PSpider(1, 1, self.zero_phase)
        contracted = ContractedDiagram(hub, partner, [], [], [0], [0])
        id1 = QSpider(1, 1, self.zero_phase)
        id2 = PSpider(1, 1, self.zero_phase)
        effect = PSpider(1, 0, self.phase_x)
        comp = CompositionDiagram([contracted, id1, id2, effect])

        result = apply_rule_to_diagram(self.rule, comp)

        assert result.num_inputs == 2
        assert result.num_outputs == 0
        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

    def test_match_swap_interleaved_with_identities(self):
        """Chase forward through a `Swap` interleaved with an identity."""
        q_state = QSpider(0, 1, self.phase_x2)
        filler_state = PSpider(0, 1, self.phase_x3)
        id_filler = PSpider(1, 1, self.zero_phase)
        id_q = QSpider(1, 1, self.zero_phase)
        hub = PSpider(1, 3, self.phi_any)
        filler_terminal = PSpider(1, 1, self.zero_phase)

        lhs = TensorDiagram([q_state, filler_state])
        swap = Swap()
        mid = TensorDiagram([id_filler, id_q])
        final_stage = TensorDiagram([filler_terminal, hub])
        comp = CompositionDiagram([lhs, swap, mid, final_stage])

        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["copy_spider_id"] == q_state.id
        assert matches[0]["disappearing_spider_id"] == hub.id
        assert matches[0]["n_copies"] == 3
        assert set(matches[0]["identity_chain"]) == {swap.id, id_q.id}
        assert id_filler.id not in matches[0]["identity_chain"]

    def test_apply_rule_swap_interleaved_with_identities(self):
        """Applying the rule resolves the crossed pattern."""
        q_state = QSpider(0, 1, self.phase_x2)
        filler_state = PSpider(0, 1, self.phase_x3)
        id_filler = PSpider(1, 1, self.zero_phase)
        id_q = QSpider(1, 1, self.zero_phase)
        hub = PSpider(1, 3, self.phi_any)
        filler_terminal = PSpider(1, 1, self.zero_phase)

        lhs = TensorDiagram([q_state, filler_state])
        swap = Swap()
        mid = TensorDiagram([id_filler, id_q])
        final_stage = TensorDiagram([filler_terminal, hub])
        comp = CompositionDiagram([lhs, swap, mid, final_stage])

        result = apply_rule_to_diagram(self.rule, comp)

        assert result.num_inputs == 0
        assert result.num_outputs == 4
        graph = to_graph(result)
        assert len(self.rule.match(graph)) == 0

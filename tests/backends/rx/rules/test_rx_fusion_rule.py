"""Unit tests for the fusion rewrite rule using the rustworkx graph formalism.

These tests verify the FusionRule implementation on rustworkx graphs, including matching,
application, flattening, and nested structures. All diagram types from
`base_gates` are exercised, and both QSpider and PSpider fusion are covered.
"""

import math
import unittest

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
from cvzx.ir.gates import (
    BeamsplitterGate,
    CubicPhaseGate,
    DisplacementGate,
    PhaseRotationGate,
    SqueezingGate,
)
from cvzx.backends.rx.graph import to_diagram, to_graph
from cvzx.backends.rx.rules import FusionRule, apply_rule_to_diagram


class TestFusionRule(unittest.TestCase):
    """Test suite for FusionRule (rustworkx graph-based)."""

    def setUp(self):
        """Create common objects used in many tests."""
        # Phase polynomials
        self.phase_poly = ZxPoly({1: 2, 2: 4})
        self.phase_poly2 = ZxPoly({1: 3, 3: 5})
        self.phase_sum = self.phase_poly + self.phase_poly2
        self.phase_poly3 = ZxPoly({2: 1, 4: 3})

        # Non-identity spiders (1x1)
        self.q1 = QSpider(1, 1, self.phase_poly)
        self.q2 = QSpider(1, 1, self.phase_poly2)
        self.q_fused = QSpider(1, 1, self.phase_sum)

        self.p1 = PSpider(1, 1, self.phase_poly)
        self.p2 = PSpider(1, 1, self.phase_poly2)
        self.p_fused = PSpider(1, 1, self.phase_sum)

        # Spiders with different arities
        self.q_1x2 = QSpider(1, 2, self.phase_poly)
        self.q_2x1 = QSpider(2, 1, self.phase_poly2)
        self.q_1x3 = QSpider(1, 3, self.phase_poly)
        self.q_3x1 = QSpider(3, 1, self.phase_poly2)
        self.q_2x2 = QSpider(2, 2, self.phase_poly)
        self.q_3x3 = QSpider(3, 3, self.phase_poly2)

        self.p_1x2 = PSpider(1, 2, self.phase_poly)
        self.p_2x1 = PSpider(2, 1, self.phase_poly2)
        self.p_1x3 = PSpider(1, 3, self.phase_poly)
        self.p_3x1 = PSpider(3, 1, self.phase_poly2)
        self.p_2x2 = PSpider(2, 2, self.phase_poly)
        self.p_3x3 = PSpider(3, 3, self.phase_poly2)

        # Other Proper Diagrams
        self.fourier = Fourier()
        self.fourier2 = Fourier2()
        self.fourier_inv = FourierInv()
        self.swap = Swap()

        # Basic Gates
        self.disp = DisplacementGate(alpha=1.0 + 0.5j)
        self.ph_rot = PhaseRotationGate(theta=math.pi / 4)
        self.sq_gate = SqueezingGate(tau=0.5)
        self.beam_splitter = BeamsplitterGate(theta=math.pi / 4)
        self.cubic = CubicPhaseGate(gamma=0.1)

        # Rule instance
        self.rule = FusionRule()

    # -------------------------------------------------------------------------
    # 1. Testing the match() method
    # -------------------------------------------------------------------------

    def test_match_simple_q_spider_fusion(self):
        """Match two Q-spiders connected in a ContractedDiagram."""
        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["contracted_id"] == contracted.id
        assert matches[0]["first_id"] == self.q1.id
        assert matches[0]["second_id"] == self.q2.id
        assert matches[0]["first_type"] == "QSpider"

    def test_match_simple_p_spider_fusion(self):
        """Match two P-spiders connected in a ContractedDiagram."""
        contracted = ContractedDiagram(self.p1, self.p2, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["contracted_id"] == contracted.id
        assert matches[0]["first_id"] == self.p1.id
        assert matches[0]["second_id"] == self.p2.id
        assert matches[0]["first_type"] == "PSpider"

    def test_match_connected_both_directions(self):
        """Match spiders connected in both directions."""
        contracted = ContractedDiagram(self.q_2x2, self.q_2x2, [0], [0], [0], [0])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["contracted_id"] == contracted.id

    def test_match_different_spider_types(self):
        """No match when spiders are different types."""
        contracted = ContractedDiagram(self.q1, self.p2, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_non_spider_diagrams(self):
        """No match when diagrams are not spiders."""
        contracted = ContractedDiagram(self.fourier, self.swap, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_nested_in_composition(self):
        """Match ContractedDiagram inside a CompositionDiagram."""
        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        comp = CompositionDiagram([self.fourier, contracted, self.sq_gate])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["contracted_id"] == contracted.id

    def test_match_nested_in_tensor(self):
        """Match ContractedDiagram inside a TensorDiagram."""
        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        tensor = TensorDiagram([self.fourier, self.ph_rot, contracted])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["contracted_id"] == contracted.id

    def test_match_nested_in_contracted(self):
        """Match ContractedDiagram inside another ContractedDiagram."""
        inner_contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        outer_contracted = ContractedDiagram(inner_contracted, self.swap, [0], [0], [], [])
        graph = to_graph(outer_contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["contracted_id"] == inner_contracted.id

    def test_match_multiple_fusion_pairs(self):
        """Match multiple fusible pairs in a diagram."""
        contracted1 = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        contracted2 = ContractedDiagram(self.p1, self.p2, [0], [0], [], [])
        comp = CompositionDiagram([contracted1, contracted2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 2
        contracted_ids = [m["contracted_id"] for m in matches]
        assert contracted1.id in contracted_ids
        assert contracted2.id in contracted_ids

    def test_match_connected_through_second_to_first(self):
        """Match when connection from second to first (J1/J2)."""
        contracted = ContractedDiagram(self.q1, self.q2, [], [], [0], [0])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["contracted_id"] == contracted.id

    def test_match_deeply_nested(self):
        """Match deeply nested ContractedDiagram."""
        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        tensor = TensorDiagram([self.fourier, contracted])
        comp = CompositionDiagram([tensor, self.swap])
        tensor2 = TensorDiagram([comp, self.ph_rot])
        graph = to_graph(tensor2)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["contracted_id"] == contracted.id

    # -------------------------------------------------------------------------
    # 2. Testing apply_single()
    # -------------------------------------------------------------------------

    def test_apply_single_simple_q_spider_fusion(self):
        """Fuse two Q-spiders connected in a ContractedDiagram."""
        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.num_inputs == 1
        assert result.num_outputs == 1
        assert result.phase == self.phase_sum

    def test_apply_single_simple_p_spider_fusion(self):
        """Fuse two P-spiders connected in a ContractedDiagram."""
        contracted = ContractedDiagram(self.p1, self.p2, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, PSpider)
        assert result.num_inputs == 1
        assert result.num_outputs == 1
        assert result.phase == self.phase_sum

    def test_apply_single_with_diff_inputs_outputs(self):
        """Fuse spiders with different inputs/outputs."""
        q1 = QSpider(2, 1, self.phase_poly)
        q2 = QSpider(1, 2, self.phase_poly2)
        contracted = ContractedDiagram(q1, q2, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.num_inputs == 2
        assert result.num_outputs == 2
        assert result.phase == self.phase_sum

    def test_apply_single_with_connection_both_ways(self):
        """Fuse spiders connected in both directions."""
        q1 = QSpider(2, 2, self.phase_poly)
        q2 = QSpider(2, 2, self.phase_poly2)
        contracted = ContractedDiagram(q1, q2, [0], [0], [0], [0])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.num_inputs == 2
        assert result.num_outputs == 2
        assert result.phase == self.phase_sum

    def test_apply_single_nested_in_composition(self):
        """Fuse spiders in ContractedDiagram inside a CompositionDiagram."""
        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        comp = CompositionDiagram([self.fourier, contracted, self.sq_gate])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 3
        assert result.diagrams[0] == self.fourier
        assert isinstance(result.diagrams[1], TensorDiagram)
        assert isinstance(result.diagrams[1].diagrams[0], QSpider)
        assert result.diagrams[1].diagrams[0].phase == self.phase_sum
        assert result.diagrams[2] == self.sq_gate

    def test_apply_single_nested_in_tensor(self):
        """Fuse spiders in ContractedDiagram inside a TensorDiagram."""
        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        tensor = TensorDiagram([self.fourier, self.ph_rot, contracted])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 3
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.ph_rot
        assert isinstance(result.diagrams[2], QSpider)
        assert result.diagrams[2].phase == self.phase_sum

    def test_apply_single_nested_in_contracted(self):
        """Fuse spiders in inner ContractedDiagram inside outer ContractedDiagram."""
        inner_contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        outer_contracted = ContractedDiagram(inner_contracted, self.swap, [0], [0], [], [])
        graph = to_graph(outer_contracted)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, ContractedDiagram)
        assert isinstance(result.first, TensorDiagram)
        assert isinstance(result.first.diagrams[0], QSpider)
        assert result.first.diagrams[0].phase == self.phase_sum
        assert result.second == self.swap

    def test_apply_single_with_multiple_fusions(self):
        """Fuse spiders where there are multiple possibilities."""
        contracted1 = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        contracted2 = ContractedDiagram(self.p1, self.p2, [0], [0], [], [])
        comp = CompositionDiagram([contracted1, contracted2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        # Apply second fusion first (index 1)
        self.rule.apply_single(graph, matches[1])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[0], ContractedDiagram)
        assert isinstance(result.diagrams[1], TensorDiagram)
        assert isinstance(result.diagrams[1].diagrams[0], PSpider)
        assert result.diagrams[1].diagrams[0].phase == self.phase_sum

    def test_apply_single_with_deeply_nested(self):
        """Test fusion in deeply nested structure."""
        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        tensor = TensorDiagram([self.fourier, contracted])
        comp = CompositionDiagram([tensor, self.swap])
        tensor2 = TensorDiagram([comp, self.ph_rot])
        graph = to_graph(tensor2)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, TensorDiagram)
        assert isinstance(result.diagrams[0], CompositionDiagram)
        assert isinstance(result.diagrams[1], PhaseRotationGate)
        assert isinstance(result.diagrams[0].diagrams[0], TensorDiagram)
        assert isinstance(result.diagrams[0].diagrams[1], Swap)
        assert isinstance(result.diagrams[0].diagrams[0].diagrams[0], Fourier)
        reduced_diagram = result.diagrams[0].diagrams[0].diagrams[1]
        assert isinstance(reduced_diagram, QSpider)
        assert reduced_diagram.num_inputs == 1
        assert reduced_diagram.num_outputs == 1
        assert reduced_diagram.phase == self.phase_sum

    # -------------------------------------------------------------------------
    # 3. Testing apply_rule() (full application)
    # -------------------------------------------------------------------------

    def test_apply_rule_single_fusion(self):
        """Apply full rule with a single fusible pair."""
        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        result = apply_rule_to_diagram(self.rule, contracted)
        assert isinstance(result, QSpider)
        assert result.num_inputs == 1
        assert result.num_outputs == 1
        assert result.phase == self.phase_sum

    def test_apply_rule_multiple_fusions(self):
        """Apply full rule with multiple fusible pairs."""
        contracted1 = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        contracted2 = ContractedDiagram(self.p1, self.p2, [0], [0], [], [])
        comp = CompositionDiagram([contracted1, contracted2])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[0], TensorDiagram)
        assert isinstance(result.diagrams[0].diagrams[0], QSpider)
        assert result.diagrams[0].diagrams[0].phase == self.phase_sum
        assert isinstance(result.diagrams[1], TensorDiagram)
        assert isinstance(result.diagrams[1].diagrams[0], PSpider)
        assert result.diagrams[1].diagrams[0].phase == self.phase_sum

    def test_apply_rule_no_match(self):
        """Apply rule when there are no fusible pairs."""
        contracted = ContractedDiagram(self.q1, self.p2, [0], [0], [], [])
        result = apply_rule_to_diagram(self.rule, contracted)
        assert result == contracted

    def test_apply_rule_complex_nested(self):
        """Apply rule with complex nested structure."""
        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        tensor = TensorDiagram([self.fourier, contracted])
        comp = CompositionDiagram([tensor, self.swap])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[0], TensorDiagram)
        assert len(result.diagrams[0].diagrams) == 2
        assert result.diagrams[0].diagrams[0] == self.fourier
        assert isinstance(result.diagrams[0].diagrams[1], QSpider)
        assert result.diagrams[0].diagrams[1].phase == self.phase_sum
        assert result.diagrams[1] == self.swap

    # -------------------------------------------------------------------------
    # 4. Edge Cases and Integration Tests
    # -------------------------------------------------------------------------

    def test_fusion_preserves_phase_sum(self):
        """Verify that fusion correctly adds phases."""
        phase1 = ZxPoly({1: 2, 2: 3})
        phase2 = ZxPoly({3: 4, 5: 6})
        expected_phase = phase1 + phase2

        q1 = QSpider(1, 1, phase1)
        q2 = QSpider(1, 1, phase2)
        contracted = ContractedDiagram(q1, q2, [0], [0], [], [])
        result = apply_rule_to_diagram(self.rule, contracted)
        assert isinstance(result, QSpider)
        assert result.phase == expected_phase

    def test_fusion_with_non_fusible_contracted_diagram(self):
        """ContractedDiagram with non-spider diagrams should not be fused."""
        contracted = ContractedDiagram(self.fourier, self.swap, [0], [0], [], [])
        result = apply_rule_to_diagram(self.rule, contracted)
        assert result == contracted

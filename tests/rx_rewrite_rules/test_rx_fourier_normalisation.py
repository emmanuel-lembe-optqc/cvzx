"""Unit tests for the Fourier normalization rewrite rule using the rustworkx graph formalism.

These tests verify the FourierNormalizationRule implementation on rustworkx graphs, including
matching, application, flattening, and nested structures. The rule folds a
Fourier-type gate (F, Finv, or F2) into an adjacent rotation or squeezing gate --
a cross-type merge ChainReductionRule cannot do on its own.
"""

import unittest
from math import isclose, pi
from typing import cast

from sympy import Symbol, simplify
from sympy import pi as sympy_pi

from cvzx.base_gates import (
    CompositionDiagram,
    ContractedDiagram,
    Fourier,
    Fourier2,
    FourierInv,
    QSpider,
    Swap,
    TensorDiagram,
    ZxPoly,
)
from cvzx.gates import BeamsplitterGate, PhaseRotationGate, SqueezingGate
from cvzx.rx_graph import to_diagram, to_graph
from cvzx.rx_rewrite_rules import FourierNormalizationRule, apply_rule_to_diagram


class TestFourierNormalizationRule(unittest.TestCase):
    """Test suite for FourierNormalizationRule using rustworkx graphs."""

    def setUp(self):
        """Create common objects used in many tests."""
        # Rotation angles chosen so that theta +/- pi/2 and theta + pi never
        # land on an odd multiple of pi/2 (PhaseRotationGate rejects those).
        self.theta1 = pi / 6
        self.theta2 = pi / 4

        # Fourier-type gates
        self.f = Fourier()
        self.finv = FourierInv()
        self.f2 = Fourier2()

        # Rotation / squeezing gates
        self.r1 = PhaseRotationGate(self.theta1)
        self.r2 = PhaseRotationGate(self.theta2)
        self.sq1 = SqueezingGate(2.0)
        self.sq2 = SqueezingGate(-3.0)

        # Inert filler / non-matching gates
        self.swap = Swap()
        self.bs = BeamsplitterGate(pi / 4)
        self.filler = QSpider(1, 1, ZxPoly({1: 7}))

        # Rule instance
        self.rule = FourierNormalizationRule()

    # -------------------------------------------------------------------------
    # 1. Testing match()
    # -------------------------------------------------------------------------

    def test_match_fourier_rotation(self):
        """F next to R(theta) matches, list order [R, F]."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "PhaseRotationGate"
        assert isclose(float(matches[0]["result_value"]), self.theta1 - pi / 2)

    def test_match_fourier_rotation_other_order(self):
        """R(theta) next to F also matches, list order [F, R] (they commute)."""
        comp = CompositionDiagram([Fourier(), PhaseRotationGate(self.theta1)])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert isclose(float(matches[0]["result_value"]), self.theta1 - pi / 2)

    def test_match_finv_rotation(self):
        """Finv next to R(theta) matches, contributing +pi/2."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), FourierInv()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "PhaseRotationGate"
        assert isclose(float(matches[0]["result_value"]), self.theta1 + pi / 2)

    def test_match_f2_rotation(self):
        """F2 next to R(theta) matches, contributing +pi."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier2()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "PhaseRotationGate"
        assert isclose(float(matches[0]["result_value"]), self.theta1 + pi)

    def test_match_f2_squeezing(self):
        """F2 next to Sq(tau) matches, negating tau."""
        comp = CompositionDiagram([SqueezingGate(2.0), Fourier2()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_type"] == "SqueezingGate"
        assert matches[0]["result_value"] == -2.0

    def test_match_f2_squeezing_other_order(self):
        """Sq(tau) next to F2 also matches (they commute)."""
        comp = CompositionDiagram([Fourier2(), SqueezingGate(-3.0)])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["result_value"] == 3.0

    def test_match_no_fourier_squeezing(self):
        """F next to Sq does NOT match -- only F2 merges with squeezing."""
        comp = CompositionDiagram([SqueezingGate(2.0), Fourier()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_no_finv_squeezing(self):
        """Finv next to Sq does NOT match -- only F2 merges with squeezing."""
        comp = CompositionDiagram([SqueezingGate(2.0), FourierInv()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_no_unrelated_gate(self):
        """A Fourier-type gate next to an unrelated gate does not match."""
        comp = CompositionDiagram([Swap(), Swap()])
        tensor = TensorDiagram([Fourier(), self.filler])
        comp2 = CompositionDiagram([self.filler, Fourier()])
        for diagram in (comp, tensor, comp2):
            graph = to_graph(diagram)
            assert len(self.rule.match(graph)) == 0

    def test_match_no_rotation_rotation(self):
        """Two rotations do not match this rule -- that's ChainReductionRule's job."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), PhaseRotationGate(self.theta2)])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_match_no_squeezing_squeezing(self):
        """Two squeezing gates do not match this rule -- that's ChainReductionRule's job."""
        comp = CompositionDiagram([SqueezingGate(2.0), SqueezingGate(3.0)])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_match_no_fourier_fourier(self):
        """Two Fourier gates do not match this rule -- that's ChainReductionRule's job."""
        comp = CompositionDiagram([Fourier(), Fourier()])
        graph = to_graph(comp)
        assert len(self.rule.match(graph)) == 0

    def test_match_with_extra_trailing_element(self):
        """Extra trailing content after the pattern doesn't prevent matching."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier(), self.filler])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["indices"] == [0, 1]

    def test_match_nested_in_tensor(self):
        """Match a pattern nested inside a tensor branch."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier()])
        tensor = TensorDiagram([Swap(), comp, self.bs])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] in graph.registry.composition_nodes

    def test_match_nested_in_contracted(self):
        """Match a pattern inside a composition inside a ContractedDiagram."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier()])
        contracted = ContractedDiagram(comp, Swap(), [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id

    def test_match_overlapping_chain_returns_one(self):
        """F, R, Finv: only the first (non-overlapping) pair is matched in one call."""
        comp = CompositionDiagram([Fourier(), PhaseRotationGate(self.theta1), FourierInv()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["indices"] == [0, 1]

    # -------------------------------------------------------------------------
    # 2. Testing apply_single()
    # -------------------------------------------------------------------------

    def test_apply_single_fourier_rotation(self):
        """F folded into R(theta) leaves a single R(theta - pi/2)."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, PhaseRotationGate)
        assert isclose(cast("float", result.theta), self.theta1 - pi / 2)

    def test_apply_single_finv_rotation(self):
        """Finv folded into R(theta) leaves a single R(theta + pi/2)."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), FourierInv()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, PhaseRotationGate)
        assert isclose(cast("float", result.theta), self.theta1 + pi / 2)

    def test_apply_single_f2_rotation(self):
        """F2 folded into R(theta) leaves a single R(theta + pi)."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier2()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, PhaseRotationGate)
        assert isclose(cast("float", result.theta), self.theta1 + pi)

    def test_apply_single_f2_squeezing(self):
        """F2 folded into Sq(tau) leaves a single Sq(-tau)."""
        comp = CompositionDiagram([SqueezingGate(2.0), Fourier2()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, SqueezingGate)
        assert result.tau == -2.0

    def test_apply_single_with_trailing_content(self):
        """Folding leaves trailing content untouched."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier(), self.filler])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[0], PhaseRotationGate)
        assert isclose(cast("float", result.diagrams[0].theta), self.theta1 - pi / 2)
        assert result.diagrams[1] == self.filler
        assert result.connectivity == {0: {0: 0}}

    def test_apply_single_nested_in_tensor(self):
        """Apply the fold to a pattern nested inside a tensor branch."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier()])
        tensor = TensorDiagram([Swap(), comp, self.bs])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 3
        assert result.diagrams[0] == Swap()
        assert isinstance(result.diagrams[1], PhaseRotationGate)
        assert isclose(cast("float", result.diagrams[1].theta), self.theta1 - pi / 2)
        assert result.diagrams[2] == self.bs

    def test_apply_single_nested_in_contracted(self):
        """Apply the fold to a pattern inside a composition inside a ContractedDiagram."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier()])
        swap_node = Swap()
        contracted = ContractedDiagram(comp, swap_node, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, ContractedDiagram)
        assert isinstance(result.first, PhaseRotationGate)
        assert isclose(cast("float", result.first.theta), self.theta1 - pi / 2)
        assert result.second == swap_node

    # -------------------------------------------------------------------------
    # 3. Testing apply_rule() (full application)
    # -------------------------------------------------------------------------

    def test_apply_rule_fourier_rotation(self):
        """Full rule application for F next to R(theta)."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier()])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, PhaseRotationGate)
        assert isclose(cast("float", result.theta), self.theta1 - pi / 2)

    def test_apply_rule_finv_rotation(self):
        """Full rule application for Finv next to R(theta)."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), FourierInv()])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, PhaseRotationGate)
        assert isclose(cast("float", result.theta), self.theta1 + pi / 2)

    def test_apply_rule_f2_rotation(self):
        """Full rule application for F2 next to R(theta)."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier2()])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, PhaseRotationGate)
        assert isclose(cast("float", result.theta), self.theta1 + pi)

    def test_apply_rule_f2_squeezing(self):
        """Full rule application for F2 next to Sq(tau)."""
        comp = CompositionDiagram([SqueezingGate(2.0), Fourier2()])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, SqueezingGate)
        assert result.tau == -2.0

    def test_apply_rule_no_match_returns_same_diagram(self):
        """If no pattern is present, apply_rule should return the original diagram."""
        comp = CompositionDiagram([Swap(), BeamsplitterGate(pi / 4)])
        result = apply_rule_to_diagram(self.rule, comp)
        assert result == comp

    def test_apply_rule_multiple_independent_matches(self):
        """Two independent, non-adjacent patterns both fold in one pass."""
        comp = CompositionDiagram([
            PhaseRotationGate(self.theta1),
            Fourier(),
            SqueezingGate(2.0),
            Fourier2(),
        ])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[0], PhaseRotationGate)
        assert isclose(cast("float", result.diagrams[0].theta), self.theta1 - pi / 2)
        assert isinstance(result.diagrams[1], SqueezingGate)
        assert result.diagrams[1].tau == -2.0

    def test_apply_rule_overlapping_chain_needs_two_passes(self):
        """F, R, Finv: one `apply_rule()` call reaches the full fixed point."""
        comp = CompositionDiagram([Fourier(), PhaseRotationGate(self.theta1), FourierInv()])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        final = to_diagram(graph)
        assert isinstance(final, PhaseRotationGate)
        assert isclose(cast("float", final.theta), self.theta1)

    def test_apply_rule_nested_in_tensor(self):
        """Full rule application to a pattern nested inside a tensor branch."""
        comp = CompositionDiagram([PhaseRotationGate(self.theta1), Fourier()])
        tensor = TensorDiagram([Swap(), comp, self.bs])
        result = apply_rule_to_diagram(self.rule, tensor)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 3
        assert isinstance(result.diagrams[1], PhaseRotationGate)
        assert isclose(cast("float", result.diagrams[1].theta), self.theta1 - pi / 2)

    # -------------------------------------------------------------------------
    # 4. Edge cases
    # -------------------------------------------------------------------------

    def test_symbolic_theta(self):
        """Folding works with a symbolic rotation angle."""
        theta = Symbol("theta", real=True)
        comp = CompositionDiagram([PhaseRotationGate(theta, parametric=True), Fourier()])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1

        assert simplify(matches[0]["result_value"] - (theta - sympy_pi / 2)) == 0

    def test_degenerate_angle_does_not_raise(self):
        """Folding to an odd multiple of pi/2 does NOT raise, unlike direct construction."""
        comp = CompositionDiagram([PhaseRotationGate(0), Fourier()])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, PhaseRotationGate)
        assert result.parametric is True
        assert isclose(float(cast("float | int", result.theta)), -pi / 2)

    def test_squeezing_fold_stays_non_parametric(self):
        """Unlike the rotation folds, the F2/squeezing fold uses plain negation."""
        comp = CompositionDiagram([SqueezingGate(2.0), Fourier2()])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, SqueezingGate)
        assert result.parametric is False
        assert result.tau == -2.0

    def test_near_zero_squeezing_fold(self):
        """A small-magnitude tau still folds correctly (sign flip, no special-casing)."""
        comp = CompositionDiagram([SqueezingGate(1e-6), Fourier2()])
        result = apply_rule_to_diagram(self.rule, comp)
        assert isinstance(result, SqueezingGate)
        assert result.tau == -1e-6

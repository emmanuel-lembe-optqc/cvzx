"""Unit tests for the identity rewrite rule.

These tests verify the IdentityRule implementation, including matching,
application, flattening, and nested structures.  All diagram types from
`base_gates` are exercised, and both QSpider and PSpider identities are
covered.
"""

import unittest
from math import pi

from mqc3.zx.base_gates import (
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
from mqc3.zx.gates import (
    BeamsplitterGate,
    ControlledSumGate,
    ControlledZGate,
    CubicPhaseGate,
    DisplacementGate,
    PhaseRotationGate,
    SqueezingGate,
)
from mqc3.zx.rewrite_rules import IdentityRule


class TestIdentityRule(unittest.TestCase):
    """Test suite for IdentityRule."""

    def setUp(self):
        """Create common objects used in many tests."""
        # Zero phase polynomial
        self.zero_phase = ZxPoly({})
        self.phase_poly = ZxPoly({1: 2, 2: 4})
        # Identity spiders
        self.id_q = QSpider(1, 1, self.zero_phase)
        self.id_p = PSpider(1, 1, self.zero_phase)
        self.id_q3x3 = QSpider(3, 3, self.zero_phase)
        self.id_p4x4 = PSpider(4, 4, self.zero_phase)
        # Non identity spiders
        self.non_id_q = QSpider(1, 1, self.phase_poly)
        self.non_id_p = PSpider(1, 1, self.phase_poly)
        self.non_id_q_3x2 = QSpider(3, 2, self.phase_poly)
        self.non_id_q_3x3 = QSpider(3, 3, self.phase_poly)
        self.non_id_q_4x4 = QSpider(4, 4, self.phase_poly)
        self.non_id_p_3x2 = PSpider(3, 2, self.phase_poly)
        self.non_id_p_3x3 = PSpider(3, 3, self.phase_poly)
        # Other Proper Diagrams
        self.fourier = Fourier()
        self.fourier2 = Fourier2()
        self.fourier_inv = FourierInv()
        self.swap = Swap()
        # Basic Gates
        self.disp = DisplacementGate(alpha=1.0 + 0.5j)
        self.ph_rot = PhaseRotationGate(theta=pi / 4)
        self.sq_gate = SqueezingGate(tau=0.5)
        self.ctrl_sum_gate1 = ControlledSumGate(gain=1.0, control=1, target=2)
        self.ctrl_sum_gate2 = ControlledSumGate(gain=2.0, control=2, target=1)
        self.ctrl_z_gate1 = ControlledZGate(gain=1.0)
        self.ctrl_sum_gate2 = ControlledZGate(gain=2.0)
        self.beam_splitter1 = BeamsplitterGate(theta=pi / 4)
        self.beam_splitter2 = BeamsplitterGate(theta=pi / 6)
        self.cubic = CubicPhaseGate(gamma=0.1)

        # Compositions
        self.comp1 = CompositionDiagram([self.non_id_p, self.sq_gate, self.id_q, self.id_p])
        self.comp2 = CompositionDiagram([
            self.non_id_p_3x3,
            TensorDiagram([
                self.beam_splitter1,
                CompositionDiagram([self.fourier, self.id_p, self.id_p]),
            ]),
        ])
        self.comp3 = CompositionDiagram([
            self.non_id_q_4x4,
            self.non_id_q_4x4,
            TensorDiagram([
                self.beam_splitter1,
                TensorDiagram([CompositionDiagram([self.id_p, self.fourier_inv]), self.ph_rot]),
            ]),
        ])
        # Nested Diagrams
        self.tensor1 = TensorDiagram([self.id_q, self.comp1])
        self.tensor2 = TensorDiagram([self.id_q, self.comp1, self.non_id_q_3x3, self.comp2])
        self.tensor3 = TensorDiagram([
            self.id_q,
            self.comp1,
            self.non_id_q_3x3,
            self.comp2,
            self.non_id_p_3x2,
            self.comp3,
        ])
        # Rule instance
        self.rule = IdentityRule()

    # -------------------------------------------------------------------------
    # 1. Testing the match() method
    # -------------------------------------------------------------------------

    def test_match_simple_composition1(self):
        """Identity inside a CompositionDiagram should be matched."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        matches = self.rule.match(comp)
        assert matches == [[0]]

    def test_match_simple_composition2(self):
        """Identity inside a CompositionDiagram should be matched."""
        comp = CompositionDiagram([self.fourier, self.id_p, self.fourier_inv])
        matches = self.rule.match(comp)
        assert matches == [[1]]

    def test_match_simple_composition3(self):
        """Identity inside a CompositionDiagram should be matched."""
        comp = CompositionDiagram([self.fourier, self.id_p, self.fourier_inv, self.id_q])
        matches = self.rule.match(comp)
        assert matches == [[1], [3]]

    def test_match_nested_composition(self):
        """Identity inside a nested CompositionDiagram should be matched."""
        inner = CompositionDiagram([self.id_q, self.fourier])
        outer = CompositionDiagram([inner, self.fourier2])
        matches = self.rule.match(outer)
        assert matches == [[0, 0]]

    def test_match_tensor(self):
        """Identity inside a TensorDiagram should NOT be matched."""
        tensor = TensorDiagram([self.id_q, self.fourier, self.id_q3x3])
        matches = self.rule.match(tensor)
        assert matches == []

    def test_match_tensor2(self):
        """Identity inside a TensorDiagram should NOT be matched."""
        comp = CompositionDiagram([self.non_id_p_3x3, TensorDiagram([self.beam_splitter1, self.id_q])])
        tensor = TensorDiagram([self.id_q, comp, self.non_id_q_3x3])
        matches = self.rule.match(tensor)
        assert matches == []

    def test_match_compos_in_tensor(self):
        """Identity inside a Composition inside a Tensor should be matched."""
        matches1 = self.rule.match(self.tensor1)
        matches2 = self.rule.match(self.tensor2)
        matches3 = self.rule.match(self.tensor3)
        assert matches1 == [[1, 2], [1, 3]]
        assert matches2 == [[1, 2], [1, 3], [3, 1, 1, 1], [3, 1, 1, 2]]
        assert matches3 == [
            [1, 2],
            [1, 3],
            [3, 1, 1, 1],
            [3, 1, 1, 2],
            [5, 2, 1, 0],
        ]

    def test_match_contracted_direct(self):
        """Identity as first or second of ContractedDiagram should NOT be matched."""
        contracted = ContractedDiagram(self.id_q, self.fourier, [0], [0], [], [])
        matches = self.rule.match(contracted)
        assert matches == []

    def test_match_contracted_with_composition1(self):
        """Identity inside a composition that is inside a ContractedDiagram should match."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(comp, self.swap, [0], [0], [], [])
        matches = self.rule.match(contracted)
        assert matches == [[0, 0]]

    def test_match_contracted_with_composition2(self):
        """Identity inside a composition that is inside a ContractedDiagram should match."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(self.swap, comp, [0], [0], [], [])
        matches = self.rule.match(contracted)
        assert matches == [[1, 0]]

    def test_match_contracted_nested(self):
        """Identity inside a composition that is inside a ContractedDiagram should match."""
        q_spider = QSpider(10, 10, self.phase_poly)
        contracted1 = ContractedDiagram(self.tensor2, q_spider, [1, 2, 3], [4, 5, 6], [0, 1, 2, 6], [1, 3, 5, 7])
        contracted2 = ContractedDiagram(q_spider, self.tensor3, [1, 2, 3], [4, 5, 6], [0, 1, 2, 6], [1, 3, 5, 7])
        matches1 = self.rule.match(contracted1)
        matches2 = self.rule.match(contracted2)
        assert matches1 == [[0, 1, 2], [0, 1, 3], [0, 3, 1, 1, 1], [0, 3, 1, 1, 2]]
        assert matches2 == [[1, 1, 2], [1, 1, 3], [1, 3, 1, 1, 1], [1, 3, 1, 1, 2], [1, 5, 2, 1, 0]]

    def test_match_does_not_match_non_identity(self):
        """Non identity spiders should not be matched."""
        comp = CompositionDiagram([self.non_id_q, self.fourier])
        matches = self.rule.match(comp)
        assert matches == []

    # -------------------------------------------------------------------------
    # 2. Testing apply_single()
    # -------------------------------------------------------------------------

    def test_apply_single_wrong_match(self):
        """Make sure the diagram is unchanged when the match path is wrong."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        new = self.rule.apply_single(comp, [1])
        assert new == comp

    def test_apply_single_remove_identity(self):
        """Removing identity from a two element composition leaves the other."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        new = self.rule.apply_single(comp, [0])
        assert isinstance(new, Fourier)
        assert new == self.fourier

    def test_apply_single_remove_identity_from_middle(self):
        """Remove identity that is not at index 0."""
        comp = CompositionDiagram([self.fourier, self.id_q, self.sq_gate])
        new = self.rule.apply_single(comp, [1])
        assert isinstance(new, CompositionDiagram)
        assert len(new.diagrams) == 2
        assert new.diagrams[0] == self.fourier
        assert new.diagrams[1] == self.sq_gate

    def test_apply_single_single_element_composition(self):
        """If composition has only the identity, it should NOT be removed."""
        comp = CompositionDiagram([self.id_q])
        new = self.rule.apply_single(comp, [0])
        assert new == comp

    def test_apply_single_nested_composition(self):
        """Remove identity from inner composition; flatten inner if needed."""
        inner = CompositionDiagram([self.id_q, self.fourier])
        outer = CompositionDiagram([inner, self.ph_rot])
        new = self.rule.apply_single(outer, [0, 0])
        assert isinstance(new, CompositionDiagram)
        assert len(new.diagrams) == 2
        assert new.diagrams[0] == self.fourier
        assert new.diagrams[1] == self.ph_rot

    def test_apply_single_identity_in_tensor_does_nothing(self):
        """apply_single on a path that is not under a CompositionDiagram returns original."""
        tensor = TensorDiagram([self.id_q, self.fourier])
        new = self.rule.apply_single(tensor, [0])
        assert new == tensor

    def test_apply_single_identity_in_contracted_does_nothing(self):
        """apply_single on a path that is directly in ContractedDiagram returns original."""
        contracted = ContractedDiagram(self.id_q, self.fourier, [0], [0], [], [])
        new = self.rule.apply_single(contracted, [0])
        assert new == contracted

    def test_apply_single_contracted_with_composition1(self):
        """Apply single to identity inside composition that is first child of ContractedDiagram."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(comp, self.swap, [0], [0], [], [])
        new = self.rule.apply_single(contracted, [0, 0])
        assert isinstance(new, ContractedDiagram)
        assert new.first == self.fourier
        assert new.second == self.swap

    def test_apply_single_contracted_with_composition2(self):
        """Apply single to identity inside composition that is second child of ContractedDiagram."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(self.swap, comp, [0], [0], [], [])
        new = self.rule.apply_single(contracted, [1, 0])
        assert isinstance(new, ContractedDiagram)
        assert new.first == self.swap
        assert new.second == self.fourier

    def test_apply_single_contracted_nested(self):
        """Apply single to deeply nested identity inside ContractedDiagram."""
        q_spider = QSpider(10, 10, self.phase_poly)
        contracted = ContractedDiagram(self.tensor2, q_spider, [1, 2, 3], [4, 5, 6], [0, 1, 2, 6], [1, 3, 5, 7])
        new = self.rule.apply_single(contracted, [0, 1, 2])
        assert isinstance(new, ContractedDiagram)
        tensor_new = new.first
        assert isinstance(tensor_new, TensorDiagram)
        comp1_new = tensor_new.diagrams[1]
        assert isinstance(comp1_new, CompositionDiagram)
        assert len(comp1_new.diagrams) == 3
        assert comp1_new.diagrams[0] == self.non_id_p
        assert comp1_new.diagrams[1] == self.sq_gate
        assert comp1_new.diagrams[2] == self.id_p

    def test_apply_single_complex_composition1(self):
        """Apply single to complex composition from self.comp1."""
        new = self.rule.apply_single(self.comp1, [2])
        assert isinstance(new, CompositionDiagram)
        assert len(new.diagrams) == 3
        assert new.diagrams[0] == self.non_id_p
        assert new.diagrams[1] == self.sq_gate
        assert new.diagrams[2] == self.id_p

    def test_apply_single_complex_composition2(self):
        """Apply single to complex composition from self.comp1."""
        new = self.rule.apply_single(self.comp1, [3])
        assert isinstance(new, CompositionDiagram)
        assert len(new.diagrams) == 3
        assert new.diagrams[0] == self.non_id_p
        assert new.diagrams[1] == self.sq_gate
        assert new.diagrams[2] == self.id_q

    def test_apply_single_complex_composition3(self):
        """Apply single to complex composition from self.comp1."""
        new1 = self.rule.apply_single(self.comp1, [2])
        new2 = self.rule.apply_single(new1, [2])
        assert isinstance(new2, CompositionDiagram)
        assert len(new2.diagrams) == 2
        assert new2.diagrams[0] == self.non_id_p
        assert new2.diagrams[1] == self.sq_gate

    def test_apply_single_nested_composition_deep(self):
        """Apply single to deeply nested composition."""
        new = self.rule.apply_single(self.tensor1, [1, 2])
        assert isinstance(new, TensorDiagram)
        assert len(new.diagrams) == 2
        assert new.diagrams[0] == self.id_q
        comp1_new = new.diagrams[1]
        assert isinstance(comp1_new, CompositionDiagram)
        assert len(comp1_new.diagrams) == 3
        assert comp1_new.diagrams[0] == self.non_id_p
        assert comp1_new.diagrams[1] == self.sq_gate
        assert comp1_new.diagrams[2] == self.id_p

    def test_apply_single_nested_tensor(self):
        """Apply single to nested tensor structure."""
        new = self.rule.apply_single(self.comp2, [1, 1, 1])
        assert isinstance(new, CompositionDiagram)
        assert len(new.diagrams) == 2
        assert new.diagrams[0] == self.non_id_p_3x3
        tensor_new = new.diagrams[1]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 2
        assert tensor_new.diagrams[0] == self.beam_splitter1
        inner_comp_new = tensor_new.diagrams[1]
        assert isinstance(inner_comp_new, CompositionDiagram)
        assert len(inner_comp_new.diagrams) == 2
        assert inner_comp_new.diagrams[0] == self.fourier
        assert inner_comp_new.diagrams[1] == self.id_p

    def test_apply_single_nested_tensor_multiple(self):
        """Apply single to nested tensor structure removing multiple identities."""
        new1 = self.rule.apply_single(self.comp2, [1, 1, 1])
        new2 = self.rule.apply_single(new1, [1, 1, 1])
        assert isinstance(new2, CompositionDiagram)
        assert len(new2.diagrams) == 2
        assert new2.diagrams[0] == self.non_id_p_3x3
        tensor_new = new2.diagrams[1]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 2
        assert tensor_new.diagrams[0] == self.beam_splitter1
        assert tensor_new.diagrams[1] == self.fourier

    # -------------------------------------------------------------------------
    # 3. Testing apply_rule() (full application)
    # -------------------------------------------------------------------------

    def test_apply_rule_removes_all_identities(self):
        """apply_rule should remove all matched identities."""
        comp1 = CompositionDiagram([self.id_q, self.fourier])
        comp2 = CompositionDiagram([self.id_p, self.disp])
        outer = CompositionDiagram([comp1, comp2])
        new = self.rule.apply_rule(outer)
        assert isinstance(new, CompositionDiagram)
        assert len(new.diagrams) == 2
        assert new.diagrams[0] == self.fourier
        assert new.diagrams[1] == self.disp

    def test_apply_rule_no_identity_returns_same_diagram(self):
        """If no identity present, apply_rule should return the original object."""
        comp = CompositionDiagram([self.fourier, self.ph_rot])
        new = self.rule.apply_rule(comp)
        assert new == comp

    def test_apply_rule_with_nested_composition_and_tensor(self):
        """Complex nesting: identities in tensor ignored."""
        tensor = TensorDiagram([self.id_q, self.fourier])
        outer = CompositionDiagram([self.swap, tensor, self.swap])
        new = self.rule.apply_rule(outer)
        assert new == outer

    def test_apply_rule_complex_comp1(self):
        """Apply full rule to self.comp1."""
        # comp1 = [non_id_p, sq_gate, id_q, id_p]
        new = self.rule.apply_rule(self.comp1)
        # Should remove both identities: [non_id_p, sq_gate]
        assert isinstance(new, CompositionDiagram)
        assert len(new.diagrams) == 2
        assert new.diagrams[0] == self.non_id_p
        assert new.diagrams[1] == self.sq_gate

    def test_apply_rule_complex_comp2(self):
        """Apply full rule to self.comp2."""
        new = self.rule.apply_rule(self.comp2)
        assert isinstance(new, CompositionDiagram)
        assert len(new.diagrams) == 2
        assert new.diagrams[0] == self.non_id_p_3x3
        tensor_new = new.diagrams[1]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 2
        assert tensor_new.diagrams[0] == self.beam_splitter1
        assert tensor_new.diagrams[1] == self.fourier

    def test_apply_rule_complex_comp3(self):
        """Apply full rule to self.comp3."""
        new = self.rule.apply_rule(self.comp3)
        assert isinstance(new, CompositionDiagram)
        assert len(new.diagrams) == 3
        assert new.diagrams[0] == self.non_id_q_4x4
        assert new.diagrams[1] == self.non_id_q_4x4
        tensor_new = new.diagrams[2]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 3
        assert tensor_new.diagrams[0] == self.beam_splitter1
        assert tensor_new.diagrams[1] == self.fourier_inv
        assert tensor_new.diagrams[2] == self.ph_rot

    def test_apply_rule_tensor1(self):
        """Apply full rule to self.tensor1."""
        new = self.rule.apply_rule(self.tensor1)
        assert isinstance(new, TensorDiagram)
        assert len(new.diagrams) == 2
        assert new.diagrams[0] == self.id_q
        comp_new = new.diagrams[1]
        assert isinstance(comp_new, CompositionDiagram)
        assert len(comp_new.diagrams) == 2
        assert comp_new.diagrams[0] == self.non_id_p
        assert comp_new.diagrams[1] == self.sq_gate

    def test_apply_rule_tensor2(self):
        """Apply full rule to self.tensor2."""
        # tensor2 = TensorDiagram([id_q, comp1, non_id_q_3x3, comp2])
        # id_q at index 0 ignored
        # comp1 loses both identities -> [non_id_p, sq_gate]
        # comp2 loses both identities -> [non_id_p_3x3, TensorDiagram([beam_splitter1, fourier])]
        new = self.rule.apply_rule(self.tensor2)
        assert isinstance(new, TensorDiagram)
        assert len(new.diagrams) == 4
        assert new.diagrams[0] == self.id_q
        # comp1_new
        comp1_new = new.diagrams[1]
        assert isinstance(comp1_new, CompositionDiagram)
        assert len(comp1_new.diagrams) == 2
        assert comp1_new.diagrams[0] == self.non_id_p
        assert comp1_new.diagrams[1] == self.sq_gate
        # non_id_q_3x3 unchanged
        assert new.diagrams[2] == self.non_id_q_3x3
        # comp2_new
        comp2_new = new.diagrams[3]
        assert isinstance(comp2_new, CompositionDiagram)
        assert len(comp2_new.diagrams) == 2
        assert comp2_new.diagrams[0] == self.non_id_p_3x3
        tensor_new = comp2_new.diagrams[1]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 2
        assert tensor_new.diagrams[0] == self.beam_splitter1
        assert tensor_new.diagrams[1] == self.fourier

    def test_apply_rule_tensor3(self):
        """Apply full rule to self.tensor3."""
        # tensor3 = TensorDiagram([id_q, comp1, non_id_q_3x3, comp2, non_id_p_3x2, comp3])
        new = self.rule.apply_rule(self.tensor3)
        assert isinstance(new, TensorDiagram)
        assert len(new.diagrams) == 6
        assert new.diagrams[0] == self.id_q
        # comp1 -> [non_id_p, sq_gate]
        comp1_new = new.diagrams[1]
        assert isinstance(comp1_new, CompositionDiagram)
        assert len(comp1_new.diagrams) == 2
        assert comp1_new.diagrams[0] == self.non_id_p
        assert comp1_new.diagrams[1] == self.sq_gate
        # non_id_q_3x3 unchanged
        assert new.diagrams[2] == self.non_id_q_3x3
        # comp2 -> [non_id_p_3x3, TensorDiagram([beam_splitter1, fourier])]
        comp2_new = new.diagrams[3]
        assert isinstance(comp2_new, CompositionDiagram)
        assert len(comp2_new.diagrams) == 2
        assert comp2_new.diagrams[0] == self.non_id_p_3x3
        tensor_new = comp2_new.diagrams[1]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 2
        assert tensor_new.diagrams[0] == self.beam_splitter1
        assert tensor_new.diagrams[1] == self.fourier
        # non_id_p_3x2 unchanged
        assert new.diagrams[4] == self.non_id_p_3x2
        # comp3 -> [non_id_q_4x4, non_id_q_4x4, TensorDiagram([beam_splitter1, TensorDiagram([fourier_inv, ph_rot])])]
        comp3_new = new.diagrams[5]
        assert isinstance(comp3_new, CompositionDiagram)
        assert len(comp3_new.diagrams) == 3
        assert comp3_new.diagrams[0] == self.non_id_q_4x4
        assert comp3_new.diagrams[1] == self.non_id_q_4x4
        tensor_new = comp3_new.diagrams[2]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 3
        assert tensor_new.diagrams[0] == self.beam_splitter1
        assert tensor_new.diagrams[1] == self.fourier_inv
        assert tensor_new.diagrams[2] == self.ph_rot

    def test_apply_rule_contracted_with_composition1(self):
        """Apply full rule to ContractedDiagram with composition containing identity as first child."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(comp, self.swap, [0], [0], [], [])
        new = self.rule.apply_rule(contracted)
        # comp loses id_q -> just fourier, contracted becomes ContractedDiagram(fourier, swap)
        assert isinstance(new, ContractedDiagram)
        assert new.first == self.fourier
        assert new.second == self.swap

    def test_apply_rule_contracted_with_composition2(self):
        """Apply full rule to ContractedDiagram with composition containing identity as second child."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(self.swap, comp, [0], [0], [], [])
        new = self.rule.apply_rule(contracted)
        # comp loses id_q -> just fourier, contracted becomes ContractedDiagram(swap, fourier)
        assert isinstance(new, ContractedDiagram)
        assert new.first == self.swap
        assert new.second == self.fourier

    def test_apply_rule_contracted_nested(self):
        """Apply full rule to ContractedDiagram with deeply nested identities."""
        q_spider = QSpider(10, 10, self.phase_poly)
        contracted = ContractedDiagram(self.tensor2, q_spider, [1, 2, 3], [4, 5, 6], [0, 1, 2, 6], [1, 3, 5, 7])
        new = self.rule.apply_rule(contracted)
        # Should remove all identities in tensor2[1] (comp1) -> [non_id_p, sq_gate]
        # And in tensor2[3] (comp2) -> [non_id_p_3x3, TensorDiagram([beam_splitter1, fourier])]
        assert isinstance(new, ContractedDiagram)
        tensor_new = new.first
        assert isinstance(tensor_new, TensorDiagram)
        # comp1_new
        comp1_new = tensor_new.diagrams[1]
        assert isinstance(comp1_new, CompositionDiagram)
        assert len(comp1_new.diagrams) == 2
        assert comp1_new.diagrams[0] == self.non_id_p
        assert comp1_new.diagrams[1] == self.sq_gate
        # comp2_new
        comp2_new = tensor_new.diagrams[3]
        assert isinstance(comp2_new, CompositionDiagram)
        assert len(comp2_new.diagrams) == 2
        assert comp2_new.diagrams[0] == self.non_id_p_3x3
        tensor_comp2 = comp2_new.diagrams[1]
        assert isinstance(tensor_comp2, TensorDiagram)
        assert len(tensor_comp2.diagrams) == 2
        assert tensor_comp2.diagrams[0] == self.beam_splitter1
        assert tensor_comp2.diagrams[1] == self.fourier


if __name__ == "__main__":
    # Create output directory for visualizations
    from mqc3.zx.visualize_base_gates import visualize_before_after

    # Create rule instance and test objects
    rule_name = "Identity Rule"
    rule = IdentityRule()
    zero_phase = ZxPoly({})
    phase_poly = ZxPoly({1: 2, 2: 4})

    id_q = QSpider(1, 1, zero_phase)
    id_p = PSpider(1, 1, zero_phase)
    fourier = Fourier()
    fourier2 = Fourier2()
    fourier_inv = FourierInv()
    swap = Swap()
    non_id_p = PSpider(1, 1, phase_poly)
    non_id_q_3x3 = QSpider(3, 3, phase_poly)
    non_id_p_3x3 = PSpider(3, 3, phase_poly)
    beam_splitter1 = BeamsplitterGate(theta=pi / 4)
    sq_gate = SqueezingGate(tau=0.5)
    ph_rot = PhaseRotationGate(theta=pi / 4)

    # Test 1: Simple composition with identity in middle
    comp1 = CompositionDiagram([fourier, id_q, sq_gate, id_p, fourier2])
    comp1_after = rule.apply_rule(comp1)
    visualize_before_after(comp1, comp1_after, "Simple Composition", rule_name)

    # Test 2: Nested composition inside tensor
    inner_comp = CompositionDiagram([id_q, fourier, id_p])
    tensor = TensorDiagram([inner_comp, non_id_p, ph_rot])
    tensor_after = rule.apply_rule(tensor)
    visualize_before_after(tensor, tensor_after, "Nested Composition in Tensor", rule_name)

    # Test 3: Complex composition with multiple identities at different depths
    comp_deep = CompositionDiagram([
        non_id_q_3x3,
        TensorDiagram([beam_splitter1, CompositionDiagram([id_p, fourier, id_p])]),
        TensorDiagram([CompositionDiagram([id_q, sq_gate, id_p]), ph_rot, fourier]),
    ])
    comp_deep_after = rule.apply_rule(comp_deep)
    visualize_before_after(comp_deep, comp_deep_after, "Complex Multiple Identities", rule_name)

    # Test 4: ContractedDiagram with composition containing identities
    comp_for_contracted = CompositionDiagram([id_q, fourier, id_p, ph_rot])
    tensor = TensorDiagram([comp_for_contracted, swap])
    q_spider_large = QSpider(3, 3, phase_poly)
    contracted = ContractedDiagram(tensor, q_spider_large, [0, 1], [1, 2], [0], [1])
    contracted_after = rule.apply_rule(contracted)
    visualize_before_after(contracted, contracted_after, "Contracted with Composition", rule_name)

    # Test 5: Diagram unchanged because the rule doesn't apply
    diagram = CompositionDiagram([non_id_p, fourier, ph_rot])
    diagram_after = rule.apply_rule(diagram)
    visualize_before_after(diagram, diagram_after, "No reduction", rule_name)

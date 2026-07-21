"""Unit tests for the bialgebra rewrite rule.

These tests verify the BialgebraRule implementation, including matching,
application, and nested structures.
"""

import math
import unittest
import pytest

from mqc3.zx.base_gates import (
    CompositionDiagram,
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
    ControlledSumGate,
    DisplacementGate,
    PhaseRotationGate,
    SqueezingGate,
)
from mqc3.zx.rewrite_rules import BialgebraRule


class TestBialgebraRule(unittest.TestCase):
    """Test suite for BialgebraRule."""

    def trivial_connect(self, n: int) -> dict:
        """Return trivial connectivity dictionnary.

        Parameters:
        ----------
        n : int
            number of arrows(inputs/outputs).

        Returns:
        -------
            dict
        """
        return {i: i for i in range(n)}

    def setUp(self):
        """Create common objects used in many tests."""
        self.zero = ZxPoly({})
        self.phase_x2 = ZxPoly({2: 2})
        self.rule = BialgebraRule()

        # Q-spiders (1 input, 2 outputs)
        self.q1x2 = QSpider(1, 2, self.zero)
        self.q2x1 = QSpider(2, 1, self.zero)

        # P-spiders (2 inputs, 1 output)
        self.p2x1 = PSpider(2, 1, self.zero)
        self.p1x2 = PSpider(1, 2, self.zero)

        # Tensor diagrams
        # Form 1: Q-P pattern (Q(1x2) ⊗ Q(1x2))
        self.q_tensor1 = TensorDiagram([self.q1x2, self.q1x2])
        self.p_tensor1 = TensorDiagram([self.p2x1, self.p2x1])
        # Form 2: P-Q pattern (P(2x1) ⊗ P(2x1))
        self.q_tensor2 = TensorDiagram([self.q2x1, self.q2x1])
        self.p_tensor2 = TensorDiagram([self.p1x2, self.p1x2])

        # Reduced forms
        self.reduced_q_first = CompositionDiagram([self.p2x1, self.q1x2])
        self.reduced_p_first = CompositionDiagram([self.q2x1, self.p1x2])

        # Connectivity for bialgebra rule
        self.conn = {0: 0, 1: 2, 2: 1, 3: 3}

        # Other diagrams
        self.q1x1 = QSpider(1, 1, self.phase_x2)
        self.p1x1 = QSpider(1, 1, self.phase_x2)
        self.q3x3 = QSpider(3, 3, self.phase_x2)
        self.p3x3 = QSpider(3, 3, self.phase_x2)
        self.fourier = Fourier()
        self.fourier_inv = FourierInv()
        self.fourier2 = Fourier2()
        self.swap = Swap()

        self.ph_rot1 = PhaseRotationGate(math.pi / 4)
        self.ph_rot2 = PhaseRotationGate(math.pi / 3)

        self.cs1 = ControlledSumGate(1)
        self.cs2 = ControlledSumGate(2)

        self.sq1 = SqueezingGate(2.0)
        self.sq2 = SqueezingGate(3.0)

        self.disp1 = DisplacementGate(1.0 + 0.5j)
        self.disp2 = DisplacementGate(2.0 + 1.0j)

        # Compositions
        self.comp1 = CompositionDiagram(
            [self.cs1, self.q_tensor1, self.p_tensor1, self.p_tensor2, self.q_tensor2, self.q_tensor1, self.p_tensor1],
            {
                0: self.trivial_connect(2),
                1: self.conn,
                2: self.trivial_connect(2),
                3: self.conn,
                4: self.trivial_connect(2),
                5: self.trivial_connect(4),
            },
        )
        self.comp1_red = CompositionDiagram([
            self.cs1,
            self.reduced_q_first,
            self.reduced_p_first,
            self.q_tensor1,
            self.p_tensor1,
        ])
        self.comp2 = CompositionDiagram([
            self.q3x3,
            TensorDiagram([
                self.ph_rot1,
                CompositionDiagram(
                    [self.cs2, self.q_tensor1, self.p_tensor1, self.p_tensor2, self.q_tensor2],
                    {
                        0: self.trivial_connect(2),
                        1: self.conn,
                        2: self.trivial_connect(2),
                        3: self.trivial_connect(4),
                    },
                ),
            ]),
        ])
        self.comp2_red = CompositionDiagram([
            self.q3x3,
            TensorDiagram([
                self.ph_rot1,
                CompositionDiagram([self.cs2, self.reduced_q_first, self.p_tensor2, self.q_tensor2]),
            ]),
        ])
        self.comp3 = CompositionDiagram([
            TensorDiagram([self.q3x3, self.p3x3]),
            TensorDiagram([
                CompositionDiagram([self.cs1, self.swap]),
                TensorDiagram([
                    CompositionDiagram(
                        [self.cs1, self.p_tensor2, self.q_tensor2, self.q_tensor1, self.p_tensor1],
                        {
                            0: self.trivial_connect(2),
                            1: self.conn,
                            2: self.trivial_connect(2),
                            3: self.conn,
                        },
                    ),
                    self.swap,
                ]),
            ]),
        ])
        self.comp3_red = CompositionDiagram([
            TensorDiagram([self.q3x3, self.p3x3]),
            TensorDiagram([
                CompositionDiagram([self.cs1, self.swap]),
                TensorDiagram([
                    CompositionDiagram([self.cs1, self.reduced_p_first, self.reduced_q_first]),
                    self.swap,
                ]),
            ]),
        ])
        # Nested Diagrams
        self.tensor1 = TensorDiagram([self.q1x1, self.comp1])
        self.tensor2 = TensorDiagram([self.q1x1, self.comp1, self.p1x1, self.comp2])
        self.tensor3 = TensorDiagram([
            self.q1x1,
            self.comp1,
            self.p1x1,
            self.comp2,
            self.p1x2,
            self.comp3,
        ])

        # Matches
        self.match1 = [[1, 1], [1, 3]]
        self.match2 = [[3, 1, 1, 1]]
        self.match3 = [[5, 1, 1, 1], [5, 1, 1, 3]]

    # -------------------------------------------------------------------------
    # 1. Testing is_bialgebra_pattern()
    # -------------------------------------------------------------------------

    # def test_is_bialgebra_pattern_q_first(self):
    #     """Test Q-P pattern is recognized."""
    #     comp = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     assert self.rule.is_bialgebra_pattern(comp) is True

    # def test_is_bialgebra_pattern_p_first(self):
    #     """Test P-Q pattern is recognized."""
    #     comp = CompositionDiagram([self.p_tensor2, self.q_tensor2], {0: self.conn})
    #     assert self.rule.is_bialgebra_pattern(comp) is True

    # def test_is_bialgebra_pattern_wrong_connectivity(self):
    #     """Test wrong connectivity fails."""
    #     wrong_conn = {0: 0, 1: 1, 2: 2, 3: 3}
    #     comp = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: wrong_conn})
    #     assert self.rule.is_bialgebra_pattern(comp) is False

    # def test_is_bialgebra_pattern_not_composition(self):
    #     """Test non-CompositionDiagram returns False."""
    #     assert self.rule.is_bialgebra_pattern(self.q_tensor1) is False

    # def test_is_bialgebra_pattern_wrong_length(self):
    #     """Test composition with wrong number of diagrams."""
    #     comp = CompositionDiagram([self.q_tensor1, self.p_tensor1, self.swap])
    #     assert self.rule.is_bialgebra_pattern(comp) is False

    # def test_is_bialgebra_pattern_not_tensor(self):
    #     """Test composition with non-TensorDiagram children."""
    #     comp = CompositionDiagram([self.q1x2, self.p2x1])
    #     assert self.rule.is_bialgebra_pattern(comp) is False

    # def test_is_bialgebra_pattern_wrong_spider_arities(self):
    #     """Test wrong spider arities."""
    #     q_wrong = QSpider(2, 2, self.zero)
    #     q_tensor_wrong = TensorDiagram([q_wrong, q_wrong])
    #     comp = CompositionDiagram([q_tensor_wrong, self.p_tensor1], {0: self.conn})
    #     assert self.rule.is_bialgebra_pattern(comp) is False

    # def test_is_bialgebra_pattern_non_zero_phase(self):
    #     """Test spiders with non-zero phase fail."""
    #     phase = ZxPoly({1: 2})
    #     q_with_phase = QSpider(1, 2, phase)
    #     q_tensor_phase = TensorDiagram([q_with_phase, self.q1x2])
    #     comp = CompositionDiagram([q_tensor_phase, self.p_tensor1], {0: self.conn})
    #     assert self.rule.is_bialgebra_pattern(comp) is False

    # -------------------------------------------------------------------------
    # 2. Testing match()
    # -------------------------------------------------------------------------

    # def test_match_q_first(self):
    #     """Match Q-P pattern at top level."""
    #     comp = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     matches = self.rule.match(comp)
    #     assert len(matches) == 1
    #     assert matches[0] == [0]

    # def test_match_p_first(self):
    #     """Match P-Q pattern at top level."""
    #     comp = CompositionDiagram([self.p_tensor2, self.q_tensor2], {0: self.conn})
    #     matches = self.rule.match(comp)
    #     assert len(matches) == 1
    #     assert matches[0] == [0]

    # def test_match_no_pattern(self):
    #     """Test diagrams without pattern don't match."""
    #     comp = CompositionDiagram([self.q_tensor1, TensorDiagram([self.swap, self.swap])])
    #     matches = self.rule.match(comp)
    #     assert len(matches) == 0

    # def test_match_wrong_connectivity(self):
    #     """Test wrong connectivity doesn't match."""
    #     wrong_conn = {0: 0, 1: 1, 2: 2, 3: 3}
    #     comp = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: wrong_conn})
    #     matches = self.rule.match(comp)
    #     assert len(matches) == 0

    # def test_match_nested_composition(self):
    #     """Match pattern inside a CompositionDiagram."""
    #     pattern = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     outer = CompositionDiagram([self.cs1, pattern])
    #     matches = self.rule.match(outer)
    #     assert len(matches) == 1
    #     assert matches[0] == [1, 0]

    # def test_match_tensor(self):
    #     """Match pattern inside a TensorDiagram."""
    #     pattern = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     tensor = TensorDiagram([pattern, self.cs2])
    #     matches = self.rule.match(tensor)
    #     assert len(matches) == 1
    #     assert matches[0] == [0, 0]

    # def test_match_compos_in_tensor(self):
    #     """Bialgebra pattern inside a Composition inside a Tensor should be matched."""
    #     matches1 = self.rule.match(self.tensor1)
    #     assert len(matches1) == 2
    #     assert matches1 == self.match1

    #     matches2 = self.rule.match(self.tensor2)
    #     assert len(matches2) == 3
    #     assert matches2[0:2] == self.match1
    #     assert matches2[2] == self.match2[0]

    #     matches3 = self.rule.match(self.tensor3)
    #     assert len(matches3) == 5
    #     assert matches3[0:2] == self.match1
    #     assert matches3[2] == self.match2[0]
    #     assert matches3[3:] == self.match3

    # def test_match_contracted(self):
    #     """Match pattern inside a ContractedDiagram."""
    #     pattern = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     contracted = ContractedDiagram(pattern, self.fourier, [0], [0], [], [])
    #     matches = self.rule.match(contracted)
    #     assert len(matches) == 1
    #     assert matches[0] == [0, 0]

    # def test_match_contracted_nested(self):
    #     """Pattern inside a composition that is inside a ContractedDiagram should match."""
    #     q_spider = QSpider(8, 8, self.phase_x2)
    #     contracted1 = ContractedDiagram(self.tensor2, q_spider, [1, 2, 3], [4, 5, 6], [0, 1, 2, 5], [1, 3, 5, 7])
    #     contracted2 = ContractedDiagram(q_spider, self.tensor3, [1, 2, 3], [4, 5, 6], [0, 1, 2, 6], [1, 3, 5, 7])
    #     matches1 = self.rule.match(contracted1)
    #     matches2 = self.rule.match(contracted2)
    #     match_list1 = [
    #         self.match1.copy(),
    #         self.match2.copy(),
    #     ]
    #     for match in match_list1:
    #         for i, path in enumerate(match):
    #             match[i] = [0, *path]
    #     match_list2 = [
    #         self.match1.copy(),
    #         self.match2.copy(),
    #         self.match3.copy(),
    #     ]
    #     for match in match_list2:
    #         for i, path in enumerate(match):
    #             match[i] = [1, *path]

    #     assert len(matches1) == 3
    #     assert matches1[0:2] == match_list1[0]
    #     assert matches1[2] == match_list1[1][0]

    #     assert len(matches2) == 5
    #     assert matches2[0:2] == match_list2[0]
    #     assert matches2[2] == match_list2[1][0]
    #     assert matches2[3:] == match_list2[2]

    # -------------------------------------------------------------------------
    # 3. Testing apply_single()
    # -------------------------------------------------------------------------

    # def test_apply_single_no_match(self):
    #     """Applying to non-matching diagram returns original."""
    #     comp = CompositionDiagram([self.q_tensor1, self.p_tensor1])
    #     result = self.rule.apply_single(comp, [])
    #     assert result == comp

    # def test_apply_single_q_first(self):
    #     """Apply Q-P → P-Q rule."""
    #     comp = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     result = self.rule.apply_single(comp, [0])
    #     assert isinstance(result, CompositionDiagram)
    #     assert len(result.diagrams) == 2
    #     assert result.diagrams[0] == self.p2x1
    #     assert result.diagrams[1] == self.q1x2

    # def test_apply_single_p_first(self):
    #     """Apply P-Q → Q-P rule."""
    #     comp = CompositionDiagram([self.p_tensor2, self.q_tensor2], {0: self.conn})
    #     result = self.rule.apply_single(comp, [0])
    #     assert isinstance(result, CompositionDiagram)
    #     assert len(result.diagrams) == 2
    #     assert result.diagrams[0] == self.q2x1
    #     assert result.diagrams[1] == self.p1x2

    # def test_apply_single_nested_in_composition(self):
    #     """Apply rule to nested pattern in composition."""
    #     pattern = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     outer = CompositionDiagram([self.cs1, pattern])
    #     result = self.rule.apply_single(outer, [1, 0])
    #     assert isinstance(result, CompositionDiagram)
    #     assert len(result.diagrams) == 2
    #     assert result.diagrams[0] == self.cs1
    #     assert isinstance(result.diagrams[1], CompositionDiagram)
    #     assert len(result.diagrams[1].diagrams) == 2
    #     assert result.diagrams[1].diagrams[0] == self.p2x1
    #     assert result.diagrams[1].diagrams[1] == self.q1x2

    # def test_apply_single_in_tensor(self):
    #     """Apply rule to a pattern in tensor."""
    #     pattern = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     tensor = TensorDiagram([self.cs2, pattern])
    #     result = self.rule.apply_single(tensor, [1, 0])
    #     assert isinstance(result, TensorDiagram)
    #     assert len(result.diagrams) == 2
    #     assert result.diagrams[0] == self.cs2
    #     assert isinstance(result.diagrams[1], CompositionDiagram)
    #     assert len(result.diagrams[1].diagrams) == 2
    #     assert result.diagrams[1].diagrams[0] == self.p2x1
    #     assert result.diagrams[1].diagrams[1] == self.q1x2

    # def test_apply_single_in_contracted(self):
    #     """Apply rule to a pattern in contracted."""
    #     pattern = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     contracted = ContractedDiagram(pattern, self.fourier, [0], [0], [], [])
    #     result = self.rule.apply_single(contracted, [0, 0])
    #     assert isinstance(result, ContractedDiagram)
    #     assert isinstance(result.first, CompositionDiagram)
    #     assert len(result.first.diagrams) == 2
    #     assert result.first.diagrams[0] == self.p2x1
    #     assert result.first.diagrams[1] == self.q1x2
    #     assert result.second == self.fourier

    # def test_apply_single_wrong_path(self):
    #     """Applying with wrong path returns original."""
    #     comp = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     with pytest.raises(IndexError, match="list index out of range"):
    #         self.rule.apply_single(comp, [5])

    # def test_apply_single_with_non_zero_phase_preserved(self):
    #     """Test that non-zero phases in surrounding context are preserved."""
    #     phase = ZxPoly({1: 2})
    #     q_with_phase = QSpider(2, 2, phase)
    #     comp = CompositionDiagram(
    #         [q_with_phase, self.q_tensor1, self.p_tensor1], {0: self.trivial_connect(2), 1: self.conn}
    #     )
    #     result = self.rule.apply_single(comp, [1])
    #     assert isinstance(result, CompositionDiagram)
    #     assert len(result.diagrams) == 3
    #     assert result.diagrams[0] == q_with_phase
    #     assert result.diagrams[1] == self.p2x1
    #     assert result.diagrams[2] == self.q1x2

    # def test_apply_single_nested(self):
    #     """Apply single to nested composition and tensor."""
    #     result = self.rule.apply_single(self.tensor1, [1, 1])
    #     expected_comp = CompositionDiagram(
    #         [
    #             self.cs1,
    #             self.reduced_q_first,
    #             self.p_tensor2,
    #             self.q_tensor2,
    #             self.q_tensor1,
    #             self.p_tensor1,
    #         ],
    #         {
    #             0: self.trivial_connect(2),
    #             1: self.trivial_connect(2),
    #             2: self.conn,
    #             3: self.trivial_connect(2),
    #             4: self.trivial_connect(4),
    #         },
    #     )
    #     expected_comp = self.rule.flatten_composition(expected_comp)
    #     expected = TensorDiagram([self.q1x1, expected_comp])
    #     assert result == expected

    #     # Apply single a single a second time
    #     result2 = self.rule.apply_single(result, [1, 3])
    #     expected2 = TensorDiagram([self.q1x1, self.rule.flatten_composition(self.comp1_red)])
    #     assert result2 == expected2

    # -------------------------------------------------------------------------
    # 4. Testing apply_rule() (full application)
    # -------------------------------------------------------------------------

    # def test_apply_rule_q_first(self):
    #     """Full application of Q-P → P-Q rule."""
    #     comp = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     result = self.rule.apply_rule(comp)
    #     assert isinstance(result, CompositionDiagram)
    #     assert len(result.diagrams) == 2
    #     assert result.diagrams[0] == self.p2x1
    #     assert result.diagrams[1] == self.q1x2

    # def test_apply_rule_p_first(self):
    #     """Full application of P-Q → Q-P rule."""
    #     comp = CompositionDiagram([self.p_tensor2, self.q_tensor2], {0: self.conn})
    #     result = self.rule.apply_rule(comp)
    #     assert isinstance(result, CompositionDiagram)
    #     assert len(result.diagrams) == 2
    #     assert isinstance(result.diagrams[0], QSpider)
    #     assert isinstance(result.diagrams[1], PSpider)

    # def test_apply_rule_no_match(self):
    #     """Applying rule with no matches returns original."""
    #     comp = CompositionDiagram([self.q_tensor1, self.swap.tensor(self.cs1)])
    #     result = self.rule.apply_rule(comp)
    #     assert result == comp

    # def test_apply_rule_multiple_patterns(self):
    #     """Apply rule to multiple patterns."""
    #     pattern1 = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #     pattern2 = CompositionDiagram([self.p_tensor2, self.q_tensor2], {0: self.conn})
    #     comp = CompositionDiagram([pattern1, pattern2])
    #     result = self.rule.apply_rule(comp)
    #     assert isinstance(result, CompositionDiagram)
    #     assert len(result.diagrams) == 4
    #     assert result.diagrams[0] == self.p2x1
    #     assert result.diagrams[1] == self.q1x2
    #     assert result.diagrams[2] == self.q2x1
    #     assert result.diagrams[3] == self.p1x2

    #     def test_apply_rule_nested(self):
    #         """Apply rule to nested patterns."""
    #         pattern = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #         outer = CompositionDiagram([self.fourier, pattern])
    #         result = self.rule.apply_rule(outer)
    #         assert isinstance(result, CompositionDiagram)
    #         assert len(result.diagrams) == 2
    #         assert result.diagrams[0] == self.fourier
    #         assert isinstance(result.diagrams[1], CompositionDiagram)
    #         assert len(result.diagrams[1].diagrams) == 2
    #         assert isinstance(result.diagrams[1].diagrams[0], PSpider)
    #         assert isinstance(result.diagrams[1].diagrams[1], QSpider)

    #     def test_apply_rule_preserves_connectivity(self):
    #         """Test that connectivity is preserved after application."""
    #         comp = CompositionDiagram([self.q_tensor1, self.p_tensor1], {0: self.conn})
    #         result = self.rule.apply_rule(comp)
    #         # The reduced composition should have default connectivity
    #         # since the pattern was replaced with a composition of 2 spiders
    #         assert result.connectivity == {0: {0: 0, 1: 1}}

    #     def test_apply_rule_with_non_zero_phase_preserved(self):
    #         """Test that non-zero phases in surrounding context are preserved."""
    #         phase = ZxPoly({1: 2})
    #         q_with_phase = QSpider(1, 1, phase)
    #         comp = CompositionDiagram([q_with_phase, self.q_tensor1, self.p_tensor1], {
    #             0: {0: 0},
    #             1: self.conn
    #         })
    #         result = self.rule.apply_rule(comp)
    #         assert isinstance(result, CompositionDiagram)
    #         assert len(result.diagrams) == 2
    #         assert result.diagrams[0] == q_with_phase
    #         assert isinstance(result.diagrams[1], CompositionDiagram)
    #         assert len(result.diagrams[1].diagrams) == 2
    #         assert isinstance(result.diagrams[1].diagrams[0], PSpider)
    #         assert isinstance(result.diagrams[1].diagrams[1], QSpider)

    def test_apply_rule_compos_in_tensor(self):
        """Apply rule on nested composition/tensor with chains at multiple levels."""
        result = self.rule.apply_rule(self.tensor3)
        expected = self.rule.flatten_composition(
            TensorDiagram([
                self.q1x1,
                self.comp1_red,
                self.p1x1,
                self.comp2_red,
                self.p1x2,
                self.comp3_red,
            ])
        )
        assert result == expected


if __name__ == "__main__":
    import math

    from mqc3.zx.visualize_base_gates import visualize_before_after

    test = TestBialgebraRule()
    test.setUp()
    rule_name = "Bialgebra Rule"
    rule = BialgebraRule()
    zero = ZxPoly({})

    q1x2 = QSpider(1, 2, zero)
    q2x1 = QSpider(2, 1, zero)
    p2x1 = PSpider(2, 1, zero)
    p1x2 = PSpider(1, 2, zero)

    q_tensor1 = TensorDiagram([q1x2, q1x2])
    p_tensor1 = TensorDiagram([p2x1, p2x1])
    q_tensor2 = TensorDiagram([q2x1, q2x1])
    p_tensor2 = TensorDiagram([p1x2, p1x2])

    conn = {0: 0, 1: 2, 2: 1, 3: 3}

    fourier = Fourier()

    # Test 1: Q-P pattern (q_tensor1, p_tensor1)
    comp1 = CompositionDiagram([q_tensor1, p_tensor1], {0: conn})
    comp1_after = rule.apply_rule(comp1)
    visualize_before_after(comp1, comp1_after, "Q-P Pattern", rule_name)

    # Test 2: P-Q pattern (p_tensor2, q_tensor2)
    comp2 = CompositionDiagram([p_tensor2, q_tensor2], {0: conn})
    comp2_after = rule.apply_rule(comp2)
    visualize_before_after(comp2, comp2_after, "P-Q Pattern", rule_name)

    # Test 3: Nested in composition
    comp3_after = rule.apply_rule(test.comp1)
    visualize_before_after(test.comp1, comp3_after, "Nested in Composition", rule_name)

    # Test 4: Nested in tensor
    tensor_after = rule.apply_rule(test.tensor2)
    visualize_before_after(test.tensor2, tensor_after, "Nested in Tensor", rule_name)

    # Test 5: No pattern (wrong connectivity)
    wrong_conn = {0: 0, 1: 1, 2: 2, 3: 3}
    comp5 = CompositionDiagram([q_tensor1, p_tensor1], {0: wrong_conn})
    comp5_after = rule.apply_rule(comp5)
    visualize_before_after(comp5, comp5_after, "No Pattern (Wrong Connectivity)", rule_name)

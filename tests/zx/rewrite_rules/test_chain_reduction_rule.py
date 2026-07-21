"""Unit tests for the chain reduction rule.

These tests verify the ChainReductionRule implementation, including matching,
application, and nested structures. All gate types are covered with special
attention to the physical constraint that only monomial phases of the same
degree can be chained.
"""

import math
import unittest

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
    DisplacementGate,
    PhaseRotationGate,
    SqueezingGate,
)
from mqc3.zx.rewrite_rules import ChainReductionRule


class TestChainReductionRule(unittest.TestCase):
    """Test suite for ChainReductionRule."""

    def setUp(self):  # noqa: PLR0915
        """Create common objects used in many tests."""
        # Phase polynomials
        self.phase_x2 = ZxPoly({2: 2})
        self.phase_x2_3 = ZxPoly({2: 3})
        self.phase_x2_sum = ZxPoly({2: 5})
        self.phase_x3 = ZxPoly({3: 2})
        self.phase_x3_4 = ZxPoly({3: 4})
        self.phase_x3_sum = ZxPoly({3: 6})
        self.phase_x4 = ZxPoly({4: 1})
        self.phase_x4_2 = ZxPoly({4: 2})
        self.phase_x4_sum = ZxPoly({4: 3})

        # Mixed polynomial (should NOT chain)
        self.phase_mixed = ZxPoly({2: 2, 3: 3})

        # Identity
        self.zero_phase = ZxPoly({})

        # Q-Spiders (monomial, same degree)
        self.q_x2_2 = QSpider(1, 1, self.phase_x2)
        self.q_x2_3 = QSpider(1, 1, self.phase_x2_3)
        self.q_x2_2_n32 = QSpider(3, 2, self.phase_x2)
        self.q_x2_3_n23 = QSpider(2, 3, self.phase_x2_3)
        self.q_x2_3_n34 = QSpider(3, 4, self.phase_x2_3)
        self.q_x2_sum = QSpider(1, 1, self.phase_x2_sum)

        self.q_x3_2 = QSpider(1, 1, self.phase_x3)
        self.q_x3_4 = QSpider(1, 1, self.phase_x3_4)
        self.q_x3_sum = QSpider(1, 1, self.phase_x3_sum)

        self.q_x4_1 = QSpider(1, 1, self.phase_x4)
        self.q_x4_2 = QSpider(1, 1, self.phase_x4_2)
        self.q_x4_sum = QSpider(1, 1, self.phase_x4_sum)

        # Q-Spiders (mixed polynomial - should NOT chain)
        self.q_mixed = QSpider(1, 1, self.phase_mixed)

        # P-Spiders (monomial, same degree)
        self.p_x2_2 = PSpider(1, 1, self.phase_x2)
        self.p_x2_3 = PSpider(1, 1, self.phase_x2_3)
        self.p_x2_2_n32 = PSpider(3, 2, self.phase_x2)
        self.p_x2_3_n23 = PSpider(2, 3, self.phase_x2_3)
        self.p_x2_3_n34 = PSpider(3, 4, self.phase_x2_3)
        self.p_x2_sum = PSpider(1, 1, self.phase_x2_sum)

        self.p_x3_2 = PSpider(1, 1, self.phase_x3)
        self.p_x3_4 = PSpider(1, 1, self.phase_x3_4)
        self.p_x3_sum = PSpider(1, 1, self.phase_x3_sum)

        # P-Spiders (mixed polynomial - should NOT chain)
        self.p_mixed = PSpider(1, 1, self.phase_mixed)

        # Identity spider
        self.id_q = QSpider(1, 1, self.zero_phase)

        # Other gates
        self.fourier = Fourier()
        self.fourier_inv = FourierInv()
        self.fourier2 = Fourier2()
        self.swap = Swap()

        self.ph_rot1 = PhaseRotationGate(math.pi / 4)
        self.ph_rot2 = PhaseRotationGate(math.pi / 3)
        self.ph_rot_sum = PhaseRotationGate(7 * math.pi / 12)

        self.bs1 = BeamsplitterGate(math.pi / 4)
        self.bs2 = BeamsplitterGate(math.pi / 6)
        self.bs_sum = BeamsplitterGate(5 * math.pi / 12)

        self.sq1 = SqueezingGate(2.0)
        self.sq2 = SqueezingGate(3.0)
        self.sq_prod = SqueezingGate(6.0)

        self.disp1 = DisplacementGate(1.0 + 0.5j)
        self.disp2 = DisplacementGate(2.0 + 1.0j)
        self.disp_sum = DisplacementGate(3.0 + 1.5j)

        # Compositions
        self.comp1 = CompositionDiagram([self.p_x2_2, self.sq1, self.sq2, self.q_x2_2, self.q_x2_3])
        self.comp1_red = CompositionDiagram([self.p_x2_2, self.sq_prod, self.q_x2_sum])
        self.comp2 = CompositionDiagram([
            self.q_x2_3_n23,
            TensorDiagram([
                self.bs1,
                CompositionDiagram([self.ph_rot1, self.ph_rot2, self.fourier, self.fourier_inv]),
            ]),
        ])
        self.comp2_red = CompositionDiagram([
            self.q_x2_3_n23,
            TensorDiagram([
                self.bs1,
                CompositionDiagram([self.ph_rot_sum, self.id_q]),
            ]),
        ])
        self.comp3 = CompositionDiagram([
            self.q_x2_3_n23,
            self.p_x2_3_n34,
            TensorDiagram([
                CompositionDiagram([self.bs1, self.bs2]),
                TensorDiagram([
                    CompositionDiagram([self.p_x3_2, self.disp2, self.disp1, self.disp2, self.p_x3_2]),
                    self.ph_rot1,
                ]),
            ]),
        ])
        # Nested Diagrams
        self.tensor1 = TensorDiagram([self.q_x4_1, self.comp1])
        self.tensor2 = TensorDiagram([self.q_x4_1, self.comp1, self.p_x2_3_n23, self.comp2])
        self.tensor3 = TensorDiagram([
            self.q_x4_1,
            self.comp1,
            self.p_x2_3_n23,
            self.comp2,
            self.p_x2_3,
            self.comp3,
        ])

        # Matches
        self.match1 = {
            "path": [1],
            "start": 1,
            "end": 3,
            "gate_type": "Sq",
            "values": [2.0, 3.0],
        }
        self.match2 = {
            "path": [1],
            "start": 3,
            "end": 5,
            "gate_type": "Q",
            "values": [(self.phase_x2, 2, 1, 1), (self.phase_x2_3, 2, 1, 1)],
        }
        self.match3 = {
            "path": [3, 1, 1],
            "start": 0,
            "end": 2,
            "gate_type": "R",
            "values": [math.pi / 4, math.pi / 3],
        }
        self.match4 = {
            "path": [3, 1, 1],
            "start": 2,
            "end": 4,
            "gate_type": "F_pair",
            "values": ["F", "Finv"],
        }
        self.match5 = {
            "path": [5, 2, 0],
            "start": 0,
            "end": 2,
            "gate_type": "BS",
            "values": [math.pi / 4, math.pi / 6],
        }
        self.match6 = {
            "path": [5, 2, 1],
            "start": 1,
            "end": 4,
            "gate_type": "D",
            "values": [2.0 + 1.0j, 1.0 + 0.5j, 2.0 + 1.0j],
        }
        # Rule instance
        self.rule = ChainReductionRule()

    # -------------------------------------------------------------------------
    # 1. Testing _get_gate_info()
    # -------------------------------------------------------------------------

    def test_get_gate_info_q_spider_monomial(self):
        """Q-spider with monomial phase returns ('Q', (phase, degree, num_inputs, num_outputs))."""
        gate_type, value = self.rule.get_gate_info(self.q_x2_2)
        assert gate_type == "Q"
        assert value[0] == self.phase_x2
        assert value[1] == 2
        assert value[2] == 1
        assert value[3] == 1

        gate_type, value = self.rule.get_gate_info(self.q_x2_3_n23)
        assert gate_type == "Q"
        assert value[0] == self.phase_x2_3
        assert value[1] == 2
        assert value[2] == 2
        assert value[3] == 3

    def test_get_gate_info_q_spider_mixed(self):
        """Q-spider with mixed phase returns ('Q', (phase, None))."""
        gate_type, value = self.rule.get_gate_info(self.q_mixed)
        assert gate_type == "Q"
        assert value[0] == self.phase_mixed
        assert value[1] == None

    def test_get_gate_info_p_spider_monomial(self):
        """P-spider with monomial phase returns ('P', (phase, degree, num_inputs, num_outputs))."""
        gate_type, value = self.rule.get_gate_info(self.p_x2_2)
        assert gate_type == "P"
        assert value[0] == self.phase_x2
        assert value[1] == 2
        assert value[2] == 1
        assert value[3] == 1

        gate_type, value = self.rule.get_gate_info(self.p_x2_3_n34)
        assert gate_type == "P"
        assert value[0] == self.phase_x2_3
        assert value[1] == 2
        assert value[2] == 3
        assert value[3] == 4

    def test_get_gate_info_p_spider_mixed(self):
        """P-spider with mixed phase returns ('P', (phase, None))."""
        gate_type, value = self.rule.get_gate_info(self.p_mixed)
        assert gate_type == "P"
        assert value[0] == self.phase_mixed
        assert value[1] == None

    def test_get_gate_info_phase_rotation(self):
        """PhaseRotationGate returns ('R', theta)."""
        gate_type, value = self.rule.get_gate_info(self.ph_rot1)
        assert gate_type == "R"
        assert value == math.pi / 4  # noqa: RUF069

    def test_get_gate_info_beamsplitter(self):
        """BeamsplitterGate returns ('BS', theta)."""
        gate_type, value = self.rule.get_gate_info(self.bs1)
        assert gate_type == "BS"
        assert value == math.pi / 4  # noqa: RUF069

    def test_get_gate_info_squeezing(self):
        """SqueezingGate returns ('Sq', tau)."""
        gate_type, value = self.rule.get_gate_info(self.sq1)
        assert gate_type == "Sq"
        assert value == 2.0  # noqa: RUF069

    def test_get_gate_info_displacement(self):
        """DisplacementGate returns ('D', alpha)."""
        gate_type, value = self.rule.get_gate_info(self.disp1)
        assert gate_type == "D"
        assert value == 1.0 + 0.5j  # noqa: RUF069

    def test_get_gate_info_fourier(self):
        """Fourier returns ('F', 'F')."""
        gate_type, value = self.rule.get_gate_info(self.fourier)
        assert gate_type == "F"
        assert value == "F"

    def test_get_gate_info_fourier_inv(self):
        """FourierInv returns ('F', 'Finv')."""
        gate_type, value = self.rule.get_gate_info(self.fourier_inv)
        assert gate_type == "F"
        assert value == "Finv"

    def test_get_gate_info_fourier2(self):
        """Fourier2 returns ('F2', None)."""
        gate_type, value = self.rule.get_gate_info(self.fourier2)
        assert gate_type == "F2"
        assert value == "F2"

    def test_get_gate_info_swap_returns_none(self):
        """Swap returns None (not reducible)."""
        gate_type, value = self.rule.get_gate_info(self.swap)
        assert gate_type == None
        assert value == None

    # -------------------------------------------------------------------------
    # 2. Testing match() and find_chains_in_composition()
    # -------------------------------------------------------------------------

    def test_match_q_spider_same_degree(self):
        """Q-spiders with same degree chain together."""
        comp = CompositionDiagram([self.q_x2_2, self.q_x2_3])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["start"] == 0
        assert matches[0]["end"] == 2
        assert matches[0]["gate_type"] == "Q"
        assert len(matches[0]["values"]) == 2

    def test_match_q_spider_different_degree(self):
        """Q-spiders with different degrees do NOT chain."""
        comp = CompositionDiagram([self.q_x2_2, self.q_x3_2])
        matches = self.rule.match(comp)
        assert len(matches) == 0

    def test_match_q_spider_different_compatible_arities(self):
        """Q-spiders with same degree and different compatible arities chain together."""
        comp = CompositionDiagram([TensorDiagram([self.bs1, self.fourier]), self.q_x2_2_n32, self.q_x2_3_n23])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["start"] == 1
        assert matches[0]["end"] == 3
        assert matches[0]["gate_type"] == "Q"
        assert len(matches[0]["values"]) == 2

    def test_match_q_spider_mixed_polynomial(self):
        """Q-spider with mixed polynomial does NOT chain."""
        comp = CompositionDiagram([self.q_x2_2, self.q_mixed])
        matches = self.rule.match(comp)
        assert len(matches) == 0

    def test_match_p_spider_same_degree(self):
        """P-spiders with same degree chain together."""
        comp = CompositionDiagram([self.p_x2_2, self.p_x2_3])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "P"

    def test_match_p_spider_different_compatible_arities(self):
        """P-spiders with same degree and different compatible arities chain together."""
        comp = CompositionDiagram([TensorDiagram([self.bs2, self.fourier2]), self.p_x2_2_n32, self.p_x2_3_n23])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["start"] == 1
        assert matches[0]["end"] == 3
        assert matches[0]["gate_type"] == "P"
        assert len(matches[0]["values"]) == 2

    def test_match_p_spider_different_degree(self):
        """P-spiders with different degrees do NOT chain."""
        comp = CompositionDiagram([self.p_x2_2, self.p_x3_2])
        matches = self.rule.match(comp)
        assert len(matches) == 0

    def test_match_phase_rotation_chain(self):
        """PhaseRotation gates chain together."""
        comp = CompositionDiagram([self.ph_rot1, self.ph_rot2])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "R"

    def test_match_beamsplitter_chain(self):
        """Beamsplitter gates chain together."""
        comp = CompositionDiagram([self.bs1, self.bs2])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "BS"

    def test_match_squeezing_chain(self):
        """Squeezing gates chain together."""
        comp = CompositionDiagram([self.sq1, self.sq2])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "Sq"

    def test_match_displacement_chain(self):
        """Displacement gates chain together."""
        comp = CompositionDiagram([self.disp1, self.disp2])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "D"

    def test_match_fourier_chain(self):
        """Fourier gates chain together."""
        comp = CompositionDiagram([self.fourier, self.fourier])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "F"

    def test_match_fourier_inv_chain(self):
        """FourierInv gates chain together."""
        comp = CompositionDiagram([self.fourier_inv, self.fourier_inv])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "F"

    def test_match_fourier_pair(self):
        """F ∘ Finv pairs are detected."""
        comp = CompositionDiagram([self.fourier, self.fourier_inv])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "F_pair"

    def test_match_fourier_inv_pair(self):
        """Finv ∘ F pairs are detected."""
        comp = CompositionDiagram([self.fourier_inv, self.fourier])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "F_pair"

    def test_match_fourier2_chain(self):
        """Fourier2 gates chain together."""
        comp = CompositionDiagram([self.fourier2, self.fourier2])
        matches = self.rule.match(comp)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "F2"

    def test_match_nested_composition(self):
        """Chains in nested compositions are found."""
        inner = CompositionDiagram([self.q_x2_2, self.q_x2_3])
        outer = CompositionDiagram([self.ph_rot1, inner, self.ph_rot2])
        matches = self.rule.match(outer)
        assert len(matches) == 1
        assert matches[0]["path"] == [1]
        assert matches[0]["gate_type"] == "Q"

    def test_match_tensor(self):
        """TensorDiagram does not create chains at its level."""
        tensor = TensorDiagram([self.q_x2_2, self.q_x2_3])
        matches = self.rule.match(tensor)
        assert len(matches) == 0

    def test_match_tensor2(self):
        """Identity inside a TensorDiagram should NOT be matched."""
        comp = CompositionDiagram([self.p_x2_3_n23, TensorDiagram([self.bs1, self.p_x2_2])])
        tensor = TensorDiagram([self.q_x2_2, comp, self.q_x2_3_n34])
        matches = self.rule.match(tensor)
        assert matches == []

    def test_match_compos_in_tensor(self):
        """Identity inside a Composition inside a Tensor should be matched."""
        matches1 = self.rule.match(self.tensor1)
        assert len(matches1) == 2
        assert matches1[0] == self.match1
        assert matches1[1] == self.match2

        matches2 = self.rule.match(self.tensor2)
        assert len(matches2) == 4
        assert matches2[0] == self.match1
        assert matches2[1] == self.match2
        assert matches2[2] == self.match3
        assert matches2[3] == self.match4

        matches3 = self.rule.match(self.tensor3)
        assert len(matches3) == 6
        assert matches3[0] == self.match1
        assert matches3[1] == self.match2
        assert matches3[2] == self.match3
        assert matches3[3] == self.match4
        assert matches3[4] == self.match5
        assert matches3[5] == self.match6

    def test_match_contracted(self):
        """ContractedDiagram does not create chains at its level."""
        contracted = ContractedDiagram(self.q_x2_2, self.q_x2_3, [], [], [], [])
        matches = self.rule.match(contracted)
        assert len(matches) == 0

    def test_match_contracted_with_composition1(self):
        """Chain inside a composition that is inside a ContractedDiagram should match."""
        comp = CompositionDiagram([self.fourier, self.p_x2_2, self.p_x2_3])
        contracted = ContractedDiagram(comp, self.swap, [], [], [], [])
        matches = self.rule.match(contracted)
        m = {
            "path": [0],
            "start": 1,
            "end": 3,
            "gate_type": "P",
            "values": [(self.phase_x2, 2, 1, 1), (self.phase_x2_3, 2, 1, 1)],
        }
        assert len(matches) == 1
        assert matches[0] == m

    def test_match_contracted_with_composition2(self):
        """Chain inside a composition that is inside a ContractedDiagram should match."""
        comp = CompositionDiagram([self.fourier, self.p_x2_2, self.p_x2_3])
        contracted = ContractedDiagram(self.swap, comp, [], [], [], [])
        matches = self.rule.match(contracted)
        m = {
            "path": [1],
            "start": 1,
            "end": 3,
            "gate_type": "P",
            "values": [(self.phase_x2, 2, 1, 1), (self.phase_x2_3, 2, 1, 1)],
        }
        assert len(matches) == 1
        assert matches[0] == m

    def test_match_contracted_nested(self):
        """Chain inside a composition that is inside a ContractedDiagram should match."""
        q_spider = QSpider(10, 10, self.phase_x2)
        contracted1 = ContractedDiagram(self.tensor2, q_spider, [1, 2, 3], [4, 5, 6], [0, 1, 2, 5], [1, 3, 5, 7])
        contracted2 = ContractedDiagram(q_spider, self.tensor3, [1, 2, 3], [4, 5, 6], [0, 1, 2, 6], [1, 3, 5, 7])
        matches1 = self.rule.match(contracted1)
        matches2 = self.rule.match(contracted2)
        match_list1 = [self.match1.copy(), self.match2.copy(), self.match3.copy(), self.match4.copy()]
        for m in match_list1:
            m["path"] = [0] + m["path"]
        match_list2 = [
            self.match1.copy(),
            self.match2.copy(),
            self.match3.copy(),
            self.match4.copy(),
            self.match5.copy(),
            self.match6.copy(),
        ]
        for m in match_list2:
            m["path"] = [1] + m["path"]

        assert len(matches1) == 4
        assert matches1[0] == match_list1[0]
        assert matches1[1] == match_list1[1]
        assert matches1[2] == match_list1[2]
        assert matches1[3] == match_list1[3]

        assert len(matches2) == 6
        assert matches2[0] == match_list2[0]
        assert matches2[1] == match_list2[1]
        assert matches2[2] == match_list2[2]
        assert matches2[3] == match_list2[3]
        assert matches2[4] == match_list2[4]
        assert matches2[5] == match_list2[5]

    # -------------------------------------------------------------------------
    # 3. Testing reduce_chain()
    # -------------------------------------------------------------------------

    def test_reduce_q_chain(self):
        """Reduce Q(a) ∘ Q(b) → Q(a+b)."""
        values = [(self.phase_x2, 2, 1, 1), (self.phase_x2_3, 2, 1, 1)]
        result = self.rule.reduce_chain("Q", values)
        assert isinstance(result, QSpider)
        assert result.phase == self.phase_x2_sum
        assert result.num_inputs == 1
        assert result.num_outputs == 1

        values = [(self.phase_x2, 2, 3, 2), (self.phase_x2_3, 2, 2, 3), (self.phase_x2_3, 2, 3, 4)]
        result = self.rule.reduce_chain("Q", values)
        assert isinstance(result, QSpider)
        assert result.phase == self.phase_x2_sum + self.phase_x2_3
        assert result.num_inputs == 3
        assert result.num_outputs == 4

    def test_reduce_q_chain_to_identity(self):
        """Q(a) ∘ Q(-a) → identity."""
        phase_pos = ZxPoly({2: 2})
        phase_neg = ZxPoly({2: -2})
        # We need to create values from these
        values = [(phase_pos, 2, 4, 5), (phase_neg, 2, 5, 6)]
        result = self.rule.reduce_chain("Q", values)
        assert isinstance(result, QSpider)
        assert result.phase == ZxPoly({})
        assert result.num_inputs == 4
        assert result.num_outputs == 6

    def test_reduce_p_chain(self):
        """Reduce P(a) ∘ P(b) → P(a+b)."""
        values = [(self.phase_x2, 2, 1, 1), (self.phase_x2_3, 2, 1, 1)]
        result = self.rule.reduce_chain("P", values)
        assert isinstance(result, PSpider)
        assert result.phase == self.phase_x2_sum
        assert result.num_inputs == 1
        assert result.num_outputs == 1

        values = [(self.phase_x2, 2, 3, 2), (self.phase_x2_3, 2, 2, 3), (self.phase_x2_3, 2, 3, 4)]
        result = self.rule.reduce_chain("P", values)
        assert isinstance(result, PSpider)
        assert result.phase == self.phase_x2_sum + self.phase_x2_3
        assert result.num_inputs == 3
        assert result.num_outputs == 4

    def test_reduce_r_chain(self):
        """Reduce R(θ) ∘ R(φ) → R(θ+φ)."""
        values = [math.pi / 4, math.pi / 3]
        result = self.rule.reduce_chain("R", values)
        assert result == self.ph_rot_sum

    def test_reduce_bs_chain(self):
        """Reduce BS(θ) ∘ BS(φ) → BS(θ+φ)."""
        values = [math.pi / 4, math.pi / 6]
        result = self.rule.reduce_chain("BS", values)
        assert isinstance(result, BeamsplitterGate)
        assert math.isclose(result.theta, 5 * math.pi / 12)

    def test_reduce_sq_chain(self):
        """Reduce Sq(τ) ∘ Sq(κ) → Sq(τ·κ)."""
        values = [2.0, 3.0]
        result = self.rule.reduce_chain("Sq", values)
        assert isinstance(result, SqueezingGate)
        assert result.tau == 6.0

    def test_reduce_sq_chain_to_identity(self):
        """Sq(τ) ∘ Sq(1/τ) → identity."""
        values = [2.0, 0.5]
        result = self.rule.reduce_chain("Sq", values)
        assert result == self.id_q

    def test_reduce_d_chain(self):
        """Reduce D(α) ∘ D(β) → D(α+β)."""
        values = [1.0 + 0.5j, 2.0 + 1.0j]
        result = self.rule.reduce_chain("D", values)
        assert isinstance(result, DisplacementGate)
        assert result.alpha == 3.0 + 1.5j

    def test_reduce_f_chain_two(self):
        """F ∘ F → F²."""
        values = ["F", "F"]
        result = self.rule.reduce_chain("F", values)
        assert isinstance(result, Fourier2)

    def test_reduce_f_chain_four(self):
        """F ∘ F ∘ F ∘ F → identity."""
        values = ["F", "F", "F", "F"]
        result = self.rule.reduce_chain("F", values)
        assert result == self.id_q

    def test_reduce_f_chain_three(self):
        """F ∘ F ∘ F → Finv."""
        values = ["F", "F", "F"]
        result = self.rule.reduce_chain("F", values)
        assert isinstance(result, FourierInv)

    def test_reduce_f_pair(self):
        """F ∘ Finv → identity."""
        values = ["F", "Finv"]
        result = self.rule.reduce_chain("F_pair", values)
        assert result == self.id_q

    def test_reduce_f2_chain_even(self):
        """F² ∘ F² → identity."""
        values = ["F2", "F2"]
        result = self.rule.reduce_chain("F2", values)
        assert result == self.id_q

    def test_reduce_f2_chain_odd(self):
        """F² ∘ F² ∘ F² → F²."""
        values = ["F2", "F2", "F2"]
        result = self.rule.reduce_chain("F2", values)
        assert isinstance(result, Fourier2)

    # -------------------------------------------------------------------------
    # 4. Testing apply_single()
    # -------------------------------------------------------------------------

    def test_apply_single_q_chain(self):
        """Apply single Q chain reduction."""
        comp = CompositionDiagram([self.fourier, self.q_x2_2, self.q_x2_3])
        match = {
            "path": [],
            "start": 1,
            "end": 3,
            "gate_type": "Q",
            "values": [(self.phase_x2, 2, 1, 1), (self.phase_x2_3, 2, 1, 1)],
        }
        result = self.rule.apply_single(comp, match)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], QSpider)
        assert result.diagrams[1].phase == self.phase_x2_sum
        assert result.diagrams[1].num_inputs == 1
        assert result.diagrams[1].num_outputs == 1

        comp = CompositionDiagram([self.q_x2_2, self.q_x2_3])
        match = {
            "path": [],
            "start": 0,
            "end": 2,
            "gate_type": "Q",
            "values": [(self.phase_x2, 2, 1, 1), (self.phase_x2_3, 2, 1, 1)],
        }
        result = self.rule.apply_single(comp, match)
        assert isinstance(result, QSpider)
        assert result.phase == self.phase_x2_sum
        assert result.num_inputs == 1
        assert result.num_outputs == 1

    def test_apply_single_q_chain_to_identity(self):
        """Apply single Q chain that reduces to identity."""
        phase_pos = ZxPoly({2: 2})
        phase_neg = ZxPoly({2: -2})
        q_pos = QSpider(2, 3, phase_pos)
        q_neg = QSpider(3, 2, phase_neg)
        comp = CompositionDiagram([self.swap, self.bs1, q_pos, q_neg, self.bs2])
        match = {
            "path": [],
            "start": 2,
            "end": 4,
            "gate_type": "Q",
            "values": [(phase_pos, 2, 2, 3), (phase_neg, 2, 3, 2)],
        }
        result = self.rule.apply_single(comp, match)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 4
        assert isinstance(result.diagrams[2], QSpider)
        assert result.diagrams[2].phase == ZxPoly({})
        assert result.num_inputs == 2
        assert result.num_outputs == 2

    def test_apply_single_nested(self):
        """Apply single to nested composition and tensor."""
        result = self.rule.apply_single(self.tensor1, self.match2)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], CompositionDiagram)
        assert len(result.diagrams[1].diagrams) == 4
        assert result.diagrams[1].diagrams[0] == self.p_x2_2
        assert result.diagrams[1].diagrams[1] == self.sq1
        assert result.diagrams[1].diagrams[2] == self.sq2
        assert result.diagrams[1].diagrams[3] == self.q_x2_sum

        # Execute apply single a second time on another match
        result = self.rule.apply_single(result, self.match1)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], CompositionDiagram)
        assert len(result.diagrams[1].diagrams) == 3
        assert result.diagrams[1].diagrams[0] == self.p_x2_2
        assert result.diagrams[1].diagrams[1] == self.sq_prod
        assert result.diagrams[1].diagrams[2] == self.q_x2_sum

    # -------------------------------------------------------------------------
    # 5. Testing apply_rule() (full application)
    # -------------------------------------------------------------------------

    def test_apply_rule_simple_q_chain(self):
        """Full rule application on simple Q chain."""
        comp = CompositionDiagram([self.q_x2_2, self.q_x2_3])
        result = self.rule.apply_rule(comp)
        assert isinstance(result, QSpider)
        assert result.phase == self.phase_x2_sum

    def test_apply_rule_multiple_chains(self):
        """Full rule application on multiple chains."""
        comp = CompositionDiagram([
            self.q_x2_2,
            self.q_x2_3,
            self.ph_rot1,
            self.ph_rot2,
            self.sq1,
            self.sq2,
        ])
        result = self.rule.apply_rule(comp)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 3
        assert result.diagrams[0] == self.q_x2_sum
        assert result.diagrams[1] == self.ph_rot_sum
        assert result.diagrams[2] == SqueezingGate(6)

    def test_apply_rule_no_match(self):
        """Apply rule when no chains exist."""
        comp = CompositionDiagram([self.q_x2_2, self.q_x3_2])
        result = self.rule.apply_rule(comp)
        assert result == comp

    def test_apply_rule_mixed_polynomial_no_chain(self):
        """Mixed polynomial spiders do NOT chain."""
        q1 = QSpider(1, 1, ZxPoly({2: 2}))
        q2 = QSpider(1, 1, ZxPoly({2: 2, 3: 3}))  # mixed
        comp = CompositionDiagram([q1, q2])
        result = self.rule.apply_rule(comp)
        assert result == comp

    def test_apply_rule_different_degree_no_chain(self):
        """Different degree spiders do NOT chain."""
        q1 = QSpider(1, 1, ZxPoly({2: 2}))
        q2 = QSpider(1, 1, ZxPoly({3: 3}))
        comp = CompositionDiagram([q1, q2])
        result = self.rule.apply_rule(comp)
        assert result == comp

    def test_apply_rule_with_identity_removal(self):
        """Apply rule that removes identity."""
        phase_pos = ZxPoly({2: 2})
        phase_neg = ZxPoly({2: -2})
        q_pos = QSpider(1, 1, phase_pos)
        q_neg = QSpider(1, 1, phase_neg)
        comp = CompositionDiagram([q_pos, q_neg, self.ph_rot1])
        result = self.rule.apply_rule(comp)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.id_q
        assert result.diagrams[1] == self.ph_rot1

    def test_apply_rule_fourier_chain(self):
        """Apply rule on Fourier chain."""
        comp = CompositionDiagram([self.fourier, self.fourier])
        result = self.rule.apply_rule(comp)
        assert isinstance(result, Fourier2)

    def test_apply_rule_fourier_pair(self):
        """Apply rule on F ∘ Finv pair."""
        comp = CompositionDiagram([self.fourier, self.fourier_inv])
        result = self.rule.apply_rule(comp)
        assert result == self.id_q

    def test_apply_rule_fourier2_chain(self):
        """Apply rule on F² ∘ F² chain."""
        comp = CompositionDiagram([self.fourier2, self.fourier2])
        result = self.rule.apply_rule(comp)
        assert result == self.id_q

    def test_apply_rule_three_r_gates(self):
        """Three rotation gates chain together."""
        r1 = PhaseRotationGate(math.pi / 6)
        r2 = PhaseRotationGate(math.pi / 4)
        r3 = PhaseRotationGate(math.pi / 3)
        comp = CompositionDiagram([r1, r2, r3])
        result = self.rule.apply_rule(comp)
        assert result == PhaseRotationGate(3 * math.pi / 4)

    def test_apply_rule_compos_in_tensor(self):
        """Apply rule on nested composition/tensor with chains at multiple levels."""
        result = self.rule.apply_rule(self.tensor3)
        assert isinstance(result, TensorDiagram)
        assert result.diagrams[0] == self.q_x4_1
        # Check reduction of self.comp1
        assert result.diagrams[1] == self.comp1_red
        assert result.diagrams[2] == self.p_x2_3_n23
        # Check reduction of self.comp2
        assert result.diagrams[3] == self.comp2_red
        assert result.diagrams[4] == self.p_x2_3
        # Check reduction of self.comp3
        theta = math.pi / 4 + math.pi / 6
        assert result.diagrams[5].diagrams[:2] == [self.q_x2_3_n23, self.p_x2_3_n34]
        assert isinstance(result.diagrams[5].diagrams[2].diagrams[0], BeamsplitterGate)
        assert math.isclose(result.diagrams[5].diagrams[2].diagrams[0].theta, theta)
        assert result.diagrams[5].diagrams[2].diagrams[1] == CompositionDiagram([
            self.p_x3_2,
            DisplacementGate(5.0 + 2.5j),
            self.p_x3_2,
        ])
        assert result.diagrams[5].diagrams[2].diagrams[2] == self.ph_rot1

    def test_apply_rule_compos_in_contracted(self):
        """Chain inside a composition that is inside a ContractedDiagram should match."""
        q_spider = QSpider(10, 10, self.phase_x2)
        contracted1 = ContractedDiagram(self.tensor1, q_spider, [1], [0], [0], [1])
        contracted2 = ContractedDiagram(q_spider, self.tensor2, [4, 5, 6], [1, 2, 3], [1, 3, 5, 7], [0, 1, 2, 5])
        result1 = self.rule.apply_rule(contracted1)
        result2 = self.rule.apply_rule(contracted2)
        tensor1_red = TensorDiagram([self.q_x4_1, self.comp1_red])
        assert result1 == ContractedDiagram(tensor1_red, q_spider, [1], [0], [0], [1])
        assert result2 == ContractedDiagram(
            q_spider,
            TensorDiagram([self.q_x4_1, self.comp1_red, self.p_x2_3_n23, self.comp2_red]),
            [4, 5, 6],
            [1, 2, 3],
            [1, 3, 5, 7],
            [0, 1, 2, 5],
        )


if __name__ == "__main__":
    # Create output directory for visualizations
    from pathlib import Path

    from mqc3.zx.visualize_base_gates import visualize_before_after

    VIS_OUTPUT_DIR = Path("test_images_chain_reduction_rule")
    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create rule instance and test objects
    rule_name = "Chain Reduction Rule"
    rule = ChainReductionRule()
    zero_phase = ZxPoly({})
    phase_x2 = ZxPoly({2: 2})
    phase_x2_3 = ZxPoly({2: 3})
    phase_x2_sum = ZxPoly({2: 5})
    phase_x3 = ZxPoly({3: 2})
    phase_x4 = ZxPoly({4: 1})

    q1 = QSpider(1, 1, phase_x2)
    q2 = QSpider(1, 1, phase_x2_3)
    q3 = QSpider(1, 1, phase_x3)
    q_x4 = QSpider(1, 1, phase_x4)

    p1 = PSpider(1, 1, phase_x2)
    p2 = PSpider(1, 1, phase_x2_3)

    fourier = Fourier()
    fourier2 = Fourier2()
    fourier_inv = FourierInv()
    swap = Swap()

    ph_rot1 = PhaseRotationGate(math.pi / 4)
    ph_rot2 = PhaseRotationGate(math.pi / 3)
    ph_rot_sum = PhaseRotationGate(7 * math.pi / 12)

    bs1 = BeamsplitterGate(math.pi / 4)
    bs2 = BeamsplitterGate(math.pi / 6)

    sq1 = SqueezingGate(2.0)
    sq2 = SqueezingGate(3.0)

    disp1 = DisplacementGate(1.0 + 0.5j)
    disp2 = DisplacementGate(2.0 + 1.0j)

    # Test 1: Simple Q-spider chain
    comp1 = CompositionDiagram([q1, q2])
    comp1_after = rule.apply_rule(comp1)
    visualize_before_after(comp1, comp1_after, "Simple Q Spider Chain", rule_name)

    # Test 2: Simple P-spider chain
    comp2 = CompositionDiagram([p1, p2])
    comp2_after = rule.apply_rule(comp2)
    visualize_before_after(comp2, comp2_after, "Simple P Spider Chain", rule_name)

    # Test 3: Rotation chain
    comp3 = CompositionDiagram([ph_rot1, ph_rot2])
    comp3_after = rule.apply_rule(comp3)
    visualize_before_after(comp3, comp3_after, "Rotation Chain", rule_name)

    # Test 4: Squeezing chain
    comp4 = CompositionDiagram([sq1, sq2])
    comp4_after = rule.apply_rule(comp4)
    visualize_before_after(comp4, comp4_after, "Squeezing Chain", rule_name)

    # Test 5: Displacement chain
    comp5 = CompositionDiagram([disp1, disp2])
    comp5_after = rule.apply_rule(comp5)
    visualize_before_after(comp5, comp5_after, "Displacement Chain", rule_name)

    # Test 6: Beamsplitter chain
    comp6 = CompositionDiagram([bs1, bs2])
    comp6_after = rule.apply_rule(comp6)
    visualize_before_after(comp6, comp6_after, "Beamsplitter Chain", rule_name)

    # Test 7: Fourier chain (F ∘ F → F²)
    comp7 = CompositionDiagram([fourier, fourier])
    comp7_after = rule.apply_rule(comp7)
    visualize_before_after(comp7, comp7_after, "Fourier Chain (F ∘ F)", rule_name)

    # Test 8: Fourier pair (F ∘ Finv → Identity)
    comp8 = CompositionDiagram([fourier, fourier_inv])
    comp8_after = rule.apply_rule(comp8)
    visualize_before_after(comp8, comp8_after, "Fourier Pair (F ∘ Finv)", rule_name)

    # Test 9: Multiple chains in one composition
    comp9 = CompositionDiagram([q1, q2, ph_rot1, ph_rot2, sq1, sq2])
    comp9_after = rule.apply_rule(comp9)
    visualize_before_after(comp9, comp9_after, "Multiple Chains", rule_name)

    # Test 10: Mixed Q and P spiders (different types - no chain)
    comp10 = CompositionDiagram([q1, p1])
    comp10_after = rule.apply_rule(comp10)
    visualize_before_after(comp10, comp10_after, "Mixed Q and P (No Chain)", rule_name)

    # Test 11: Different degrees (no chain)
    comp11 = CompositionDiagram([q1, q3])
    comp11_after = rule.apply_rule(comp11)
    visualize_before_after(comp11, comp11_after, "Different Degrees (No Chain)", rule_name)

    # Test 12: Mixed polynomial (no chain)
    mixed_phase = ZxPoly({2: 2, 3: 3})
    q_mixed = QSpider(1, 1, mixed_phase)
    comp12 = CompositionDiagram([q1, q_mixed])
    comp12_after = rule.apply_rule(comp12)
    visualize_before_after(comp12, comp12_after, "Mixed Polynomial (No Chain)", rule_name)

    # Test 13: Nested composition inside tensor
    inner_comp = CompositionDiagram([q1, q2])
    tensor = TensorDiagram([inner_comp, sq1, ph_rot1])
    tensor_after = rule.apply_rule(tensor)
    visualize_before_after(tensor, tensor_after, "Nested Composition in Tensor", rule_name)

    # Test 14: ContractedDiagram with composition containing chain
    comp_for_contracted = CompositionDiagram([q1, q2, ph_rot1, ph_rot2])
    tensor = TensorDiagram([comp_for_contracted, swap])
    q_spider_large = QSpider(3, 3, phase_x2)
    contracted = ContractedDiagram(tensor, q_spider_large, [0, 1], [1, 2], [0], [1])
    contracted_after = rule.apply_rule(contracted)
    visualize_before_after(contracted, contracted_after, "Contracted with Composition", rule_name)

    # Test 15: Chain with different arities (compatible)
    q_3x2 = QSpider(3, 2, phase_x2)
    q_2x3 = QSpider(2, 3, phase_x2_3)
    comp15 = CompositionDiagram([q_3x2, q_2x3])
    comp15_after = rule.apply_rule(comp15)
    visualize_before_after(comp15, comp15_after, "Different Arities (Compatible)", rule_name)

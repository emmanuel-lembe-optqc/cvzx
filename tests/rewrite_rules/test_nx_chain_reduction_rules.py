"""Unit tests for the chain reduction rewrite rule using the graph formalism.

These tests verify the ChainReductionRule implementation on graphs, including matching,
application, flattening, and nested structures. All diagram types from
`base_gates` are exercised, and both QSpider and PSpider identities are
covered.
"""

import unittest
from math import isclose, pi
from typing import cast

from sympy import I, cos, exp, symbols

from cvzx.base_gates import (
    CompositionDiagram,
    ContractedDiagram,
    Diagram,
    Fourier,
    Fourier2,
    FourierInv,
    PSpider,
    QSpider,
    Swap,
    TensorDiagram,
    ZxPoly,
)
from cvzx.gates import (
    BeamsplitterGate,
    ControlledSumGate,
    ControlledZGate,
    DisplacementGate,
    PhaseRotationGate,
    SqueezingGate,
)
from cvzx.nx_graph import to_diagram, to_graph
from cvzx.nx_rewrite_rules import ChainReductionRule


def _sub(diagram: Diagram, index: int) -> Diagram:
    """Index into a container `Diagram`'s children, for navigating a fixture's known shape."""
    assert isinstance(diagram, (CompositionDiagram, TensorDiagram, ContractedDiagram))
    return diagram.diagrams[index]


class TestChainReductionRule(unittest.TestCase):
    """Test suite for ChainReductionRule."""

    def setUp(self):  # ruff: ignore[too-many-statements]
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

        self.ph_rot1 = PhaseRotationGate(pi / 4)
        self.ph_rot2 = PhaseRotationGate(pi / 3)
        self.ph_rot_sum = PhaseRotationGate(7 * pi / 12)

        self.bs1 = BeamsplitterGate(pi / 4)
        self.bs2 = BeamsplitterGate(pi / 6)
        self.bs3 = BeamsplitterGate(pi / 4)
        self.bs_sum = BeamsplitterGate(5 * pi / 12)

        self.sq1 = SqueezingGate(2.0)
        self.sq2 = SqueezingGate(3.0)
        self.sq_prod = SqueezingGate(6.0)

        self.disp1 = DisplacementGate(1.0 + 0.5j)
        self.disp2 = DisplacementGate(2.0 + 1.0j)
        self.disp3 = DisplacementGate(5.0 + 1.0j)
        self.disp_sum = DisplacementGate(8.0 + 2.5j)
        m = symbols("m", real=True)
        self.meas = QSpider(1, 0, ZxPoly({1: m}))

        self.cs = ControlledSumGate(gain=1.0, control=1, target=2)
        self.cs_sum = ControlledSumGate(gain=2.0, control=1, target=2)
        self.cs_bar = ControlledSumGate(gain=2.0, control=2, target=1)

        self.cz1 = ControlledZGate(gain=1.0)
        self.cz2 = ControlledZGate(gain=2.0)
        self.cz_sum = ControlledZGate(gain=3.0)

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
                CompositionDiagram([self.bs3, self.bs2]),
                TensorDiagram([
                    CompositionDiagram([self.p_x3_2, self.disp3, self.disp1, self.disp2, self.p_x3_2]),
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
            "container_id": self.tensor1.diagrams[
                1
            ].id,  # Due to container id modifications in TensorDiagram __post_init__ function
            "gate_type": "Sq",
            "values": [2.0, 3.0],
            "node_ids": [self.sq1.id, self.sq2.id],
            "gate_info": None,
        }
        self.match2 = {
            "container_id": self.tensor1.diagrams[1].id,
            "gate_type": "Q",
            "values": [self.phase_x2, self.phase_x2_3],
            "node_ids": [self.q_x2_2.id, self.q_x2_3.id],
            "gate_info": {"num_inputs": 1, "num_outputs": 1},
        }
        self.match3 = {
            "container_id": _sub(_sub(_sub(self.tensor2, 3), 1), 1).id,
            "gate_type": "R",
            "values": [self.ph_rot1.theta, self.ph_rot2.theta],
            "node_ids": [self.ph_rot1.id, self.ph_rot2.id],
            "gate_info": None,
        }
        self.match4 = {
            "container_id": _sub(_sub(_sub(self.tensor2, 3), 1), 1).id,
            "gate_type": "F_pair",
            "values": ["F", "Finv"],
            "node_ids": [self.fourier.id, self.fourier_inv.id],
            "gate_info": None,
        }
        self.match5 = {
            "container_id": _sub(_sub(_sub(self.tensor3, 5), 2), 0).id,
            "gate_type": "BS",
            "values": [self.bs3.theta, self.bs2.theta],
            "node_ids": [self.bs3.id, self.bs2.id],
            "gate_info": None,
        }
        self.match6 = {
            "container_id": _sub(_sub(_sub(self.tensor3, 5), 2), 1).id,
            "gate_type": "D",
            "values": [self.disp3.alpha, self.disp1.alpha, self.disp2.alpha],
            "node_ids": [self.disp3.id, self.disp1.id, self.disp2.id],
            "gate_info": None,
        }
        # Others
        self.id_q_dict = {
            "type": "QSpider",
            "phase": self.zero_phase,
            "num_inputs": 1,
            "num_outputs": 1,
        }
        self.id_q2 = {
            "type": "QSpider",
            "phase": self.zero_phase,
            "num_inputs": 2,
            "num_outputs": 2,
        }
        # Rule instance
        self.rule = ChainReductionRule()

    # -------------------------------------------------------------------------
    # 1. Testing _get_gate_info()
    # -------------------------------------------------------------------------

    def test_get_gate_info_q_spider_monomial(self):
        """Q-spider with monomial phase returns ('Q', (phase, degree, num_inputs, num_outputs))."""
        graph = to_graph(self.q_x2_2)
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, self.q_x2_2.id)
        assert gate_info is not None
        assert gate_type == "Q"
        assert value == self.phase_x2
        assert gate_info["num_inputs"] == 1
        assert gate_info["num_outputs"] == 1

        graph = to_graph(self.q_x2_3_n23)
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, self.q_x2_3_n23.id)
        assert gate_info is not None
        assert gate_type == "Q"
        assert value == self.phase_x2_3
        assert gate_info["num_inputs"] == 2
        assert gate_info["num_outputs"] == 3

    def test_get_gate_info_q_spider_mixed(self):
        """Q-spider with mixed phase returns ('Q', (phase, None))."""
        graph = to_graph(self.q_mixed)
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, self.q_mixed.id)
        assert gate_info is not None
        assert gate_type == "Q"
        assert value == self.phase_mixed
        assert gate_info["num_inputs"] == 1
        assert gate_info["num_outputs"] == 1

    def test_get_gate_info_p_spider_monomial(self):
        """P-spider with monomial phase returns ('P', (phase, degree, num_inputs, num_outputs))."""
        graph = to_graph(self.p_x2_2)
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, self.p_x2_2.id)
        assert gate_info is not None
        assert gate_type == "P"
        assert value == self.phase_x2
        assert gate_info["num_inputs"] == 1
        assert gate_info["num_outputs"] == 1

    def test_get_gate_info_p_spider_mixed(self):
        """P-spider with mixed phase returns ('P', (phase, None))."""
        graph = to_graph(self.p_mixed)
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, self.p_mixed.id)
        assert gate_info is not None
        assert gate_type == "P"
        assert value == self.phase_mixed
        assert gate_info["num_inputs"] == 1
        assert gate_info["num_outputs"] == 1

    def test_get_gate_info_phase_rotation(self):
        """PhaseRotationGate returns ('R', theta)."""
        graph = to_graph(self.ph_rot1)
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, self.ph_rot1.id)
        assert gate_type == "R"
        assert isclose(value, pi / 4)

    def test_get_gate_info_beamsplitter(self):
        """BeamsplitterGate returns ('BS', theta)."""
        graph = to_graph(self.bs1)
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, self.bs1.id)
        assert gate_type == "BS"
        assert isclose(value, pi / 4)

    def test_get_gate_info_squeezing(self):
        """SqueezingGate returns ('Sq', tau)."""
        graph = to_graph(self.sq1)
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, self.sq1.id)
        assert gate_type == "Sq"
        assert isclose(value, 2.0)

    def test_get_gate_info_displacement(self):
        """DisplacementGate returns ('D', alpha)."""
        graph = to_graph(self.disp1)
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, self.disp1.id)
        assert gate_type == "D"
        assert value == 1.0 + 0.5j  # ruff: ignore[float-equality-comparison]

    def test_get_gate_info_fourier(self):
        """Fourier returns ('F', 'F')."""
        graph = to_graph(self.fourier)
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, self.fourier.id)
        assert gate_type == "F"
        assert value == "F"

    def test_get_gate_info_fourier_inv(self):
        """FourierInv returns ('F', 'Finv')."""
        graph = to_graph(self.fourier_inv)
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, self.fourier_inv.id)
        assert gate_type == "F"
        assert value == "Finv"

    def test_get_gate_info_fourier2(self):
        """Fourier2 returns ('F2', None)."""
        graph = to_graph(self.fourier2)
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, self.fourier2.id)
        assert gate_type == "F2"
        assert value == "F2"

    def test_get_gate_info_swap_returns_none(self):
        """Swap returns None (not reducible)."""
        graph = to_graph(self.swap)
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, self.swap.id)
        assert gate_type is None
        assert value is None

    def test_get_gate_info_csum(self):
        """CSum gate."""
        graph = to_graph(self.cs)
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, self.cs.id)
        assert gate_info is not None
        assert gate_type == "CSUM"
        assert value == self.cs.gain
        assert gate_info["control"] == 1
        assert gate_info["target"] == 2

    def test_get_gate_info_cz(self):
        """CZ gate."""
        graph = to_graph(self.cz1)
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, self.cz1.id)
        assert gate_type == "CZ"
        assert value == self.cz1.gain

    # -------------------------------------------------------------------------
    # 2. Testing match() and find_chains_in_composition()
    # -------------------------------------------------------------------------

    def test_match_q_spider_same_degree(self):
        """Q-spiders with same degree chain together."""
        comp = CompositionDiagram([self.q_x2_2, self.q_x2_3])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "Q"
        assert matches[0]["values"] == [self.q_x2_2.phase, self.q_x2_3.phase]
        assert matches[0]["node_ids"] == [self.q_x2_2.id, self.q_x2_3.id]
        assert matches[0]["gate_info"]["num_inputs"] == 1
        assert matches[0]["gate_info"]["num_outputs"] == 1

    def test_match_q_spider_different_degree(self):
        """Q-spiders with different degrees."""
        comp = CompositionDiagram([self.q_x2_2, self.q_x3_2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "Q"
        assert matches[0]["values"] == [self.q_x2_2.phase, self.q_x3_2.phase]
        assert matches[0]["node_ids"] == [self.q_x2_2.id, self.q_x3_2.id]
        assert matches[0]["gate_info"]["num_inputs"] == 1
        assert matches[0]["gate_info"]["num_outputs"] == 1

    def test_match_q_spider_different_compatible_arities(self):
        """Q-spiders with same degree and different compatible arities chain together."""
        comp = CompositionDiagram([TensorDiagram([self.bs1, self.fourier]), self.q_x2_2_n32, self.q_x2_3_n23])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "Q"
        assert matches[0]["values"] == [self.q_x2_2_n32.phase, self.q_x2_3_n23.phase]
        assert matches[0]["node_ids"] == [self.q_x2_2_n32.id, self.q_x2_3_n23.id]
        assert matches[0]["gate_info"]["num_inputs"] == 3
        assert matches[0]["gate_info"]["num_outputs"] == 3

    def test_match_q_spider_mixed_polynomial(self):
        """Q-spider with mixed polynomial does NOT chain."""
        comp = CompositionDiagram([self.q_x2_2, self.q_mixed])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "Q"
        assert matches[0]["values"] == [self.q_x2_2.phase, self.q_mixed.phase]
        assert matches[0]["node_ids"] == [self.q_x2_2.id, self.q_mixed.id]
        assert matches[0]["gate_info"]["num_inputs"] == 1
        assert matches[0]["gate_info"]["num_outputs"] == 1

    def test_match_p_spider_same_degree(self):
        """P-spiders with same degree chain together."""
        comp = CompositionDiagram([self.p_x2_2, self.p_x2_3])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "P"
        assert matches[0]["values"] == [self.p_x2_2.phase, self.p_x2_3.phase]
        assert matches[0]["node_ids"] == [self.p_x2_2.id, self.p_x2_3.id]
        assert matches[0]["gate_info"]["num_inputs"] == 1
        assert matches[0]["gate_info"]["num_outputs"] == 1

    def test_match_p_spider_different_degree(self):
        """P-spiders with different degrees."""
        comp = CompositionDiagram([self.p_x2_2, self.p_x3_2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "P"
        assert matches[0]["values"] == [self.p_x2_2.phase, self.p_x3_2.phase]
        assert matches[0]["node_ids"] == [self.p_x2_2.id, self.p_x3_2.id]
        assert matches[0]["gate_info"]["num_inputs"] == 1
        assert matches[0]["gate_info"]["num_outputs"] == 1

    def test_match_p_spider_different_compatible_arities(self):
        """P-spiders with same degree and different compatible arities chain together."""
        comp = CompositionDiagram([TensorDiagram([self.bs1, self.fourier]), self.p_x2_2_n32, self.p_x2_3_n23])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "P"
        assert matches[0]["values"] == [self.p_x2_2_n32.phase, self.p_x2_3_n23.phase]
        assert matches[0]["node_ids"] == [self.p_x2_2_n32.id, self.p_x2_3_n23.id]
        assert matches[0]["gate_info"]["num_inputs"] == 3
        assert matches[0]["gate_info"]["num_outputs"] == 3

    def test_match_p_spider_mixed_polynomial(self):
        """P-spider with mixed polynomial does NOT chain."""
        comp = CompositionDiagram([self.p_x2_2, self.p_mixed])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "P"
        assert matches[0]["values"] == [self.p_x2_2.phase, self.p_mixed.phase]
        assert matches[0]["node_ids"] == [self.p_x2_2.id, self.p_mixed.id]
        assert matches[0]["gate_info"]["num_inputs"] == 1
        assert matches[0]["gate_info"]["num_outputs"] == 1

    def test_match_phase_rotation_chain(self):
        """PhaseRotation gates chain together."""
        comp = CompositionDiagram([self.ph_rot1, self.ph_rot2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "R"
        assert matches[0]["values"] == [self.ph_rot1.theta, self.ph_rot2.theta]
        assert matches[0]["node_ids"] == [self.ph_rot1.id, self.ph_rot2.id]
        assert matches[0]["gate_info"] is None

    def test_match_beamsplitter_chain(self):
        """Beamsplitter gates chain together."""
        comp = CompositionDiagram([self.bs1, self.bs2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "BS"
        assert matches[0]["values"] == [self.bs1.theta, self.bs2.theta]
        assert matches[0]["node_ids"] == [self.bs1.id, self.bs2.id]
        assert matches[0]["gate_info"] is None

    def test_match_squeezing_chain(self):
        """Squeezing gates chain together."""
        comp = CompositionDiagram([self.sq1, self.sq2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "Sq"
        assert matches[0]["values"] == [self.sq1.tau, self.sq2.tau]
        assert matches[0]["node_ids"] == [self.sq1.id, self.sq2.id]
        assert matches[0]["gate_info"] is None

    def test_match_displacement_chain(self):
        """Displacement gates chain together."""
        comp = CompositionDiagram([self.disp1, self.disp2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "D"
        assert matches[0]["values"] == [self.disp1.alpha, self.disp2.alpha]
        assert matches[0]["node_ids"] == [self.disp1.id, self.disp2.id]
        assert matches[0]["gate_info"] is None

    def test_match_fourier_chain(self):
        """Fourier gates chain together."""
        fourier = Fourier()
        comp = CompositionDiagram([self.fourier, fourier])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "F"
        assert matches[0]["values"] == ["F", "F"]
        assert matches[0]["node_ids"] == [self.fourier.id, fourier.id]
        assert matches[0]["gate_info"] is None

    def test_match_fourier_inv_chain(self):
        """FourierInv gates chain together."""
        fourier_inv = FourierInv()
        comp = CompositionDiagram([self.fourier_inv, fourier_inv])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "F"
        assert matches[0]["values"] == ["Finv", "Finv"]
        assert matches[0]["node_ids"] == [self.fourier_inv.id, fourier_inv.id]
        assert matches[0]["gate_info"] is None

    def test_match_fourier_pair(self):
        """F ∘ Finv pairs are detected."""
        comp = CompositionDiagram([self.fourier, self.fourier_inv])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "F_pair"
        assert matches[0]["values"] == ["F", "Finv"]
        assert matches[0]["node_ids"] == [self.fourier.id, self.fourier_inv.id]
        assert matches[0]["gate_info"] is None

    def test_match_fourier_inv_pair(self):
        """Finv ∘ F pairs are detected."""
        comp = CompositionDiagram([self.fourier_inv, self.fourier])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "F_pair"
        assert matches[0]["values"] == ["Finv", "F"]
        assert matches[0]["node_ids"] == [self.fourier_inv.id, self.fourier.id]
        assert matches[0]["gate_info"] is None

    def test_match_fourier2_chain(self):
        """Fourier2 gates chain together."""
        fourier2 = Fourier2()
        comp = CompositionDiagram([self.fourier2, fourier2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "F2"
        assert matches[0]["values"] == ["F2", "F2"]
        assert matches[0]["node_ids"] == [self.fourier2.id, fourier2.id]
        assert matches[0]["gate_info"] is None

    def test_match_csum_chain(self):
        """ControlSum gates chain together."""
        cs = ControlledSumGate(gain=5, control=1, target=2)
        comp = CompositionDiagram([self.cs, cs])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "CSUM"
        assert matches[0]["values"] == [self.cs.gain, cs.gain]
        assert matches[0]["node_ids"] == [self.cs.id, cs.id]
        assert matches[0]["gate_info"] == {"control": 1, "target": 2}

        cs_bar = ControlledSumGate(gain=5)
        comp = CompositionDiagram([self.cs_bar, cs_bar])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "CSUM"
        assert matches[0]["values"] == [self.cs_bar.gain, cs_bar.gain]
        assert matches[0]["node_ids"] == [self.cs_bar.id, cs_bar.id]
        assert matches[0]["gate_info"] == {"control": 2, "target": 1}

        # ControlSumGates with different control and/or target do not chain
        comp = CompositionDiagram([self.cs, self.cs_bar])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

        comp = CompositionDiagram([self.cs_bar, self.cs])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_cz_chain(self):
        """ControlZ gates chain together."""
        comp = CompositionDiagram([self.cz1, self.cz2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "CZ"
        assert matches[0]["values"] == [self.cz1.gain, self.cz2.gain]
        assert matches[0]["node_ids"] == [self.cz1.id, self.cz2.id]
        assert matches[0]["gate_info"] is None

    def test_match_nested_composition(self):
        """Chains in nested compositions are found."""
        inner = CompositionDiagram([self.q_x2_2, self.q_x2_3])
        outer = CompositionDiagram([self.ph_rot1, inner, self.ph_rot2])
        graph = to_graph(outer)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "Q"
        assert matches[0]["values"] == [self.q_x2_2.phase, self.q_x2_3.phase]
        assert matches[0]["node_ids"] == [self.q_x2_2.id, self.q_x2_3.id]
        assert matches[0]["gate_info"]["num_inputs"] == 1
        assert matches[0]["gate_info"]["num_outputs"] == 1

    def test_match_tensor(self):
        """TensorDiagram does not create chains at its level."""
        tensor = TensorDiagram([self.q_x2_2, self.q_x2_3])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_tensor2(self):
        """Identity inside a TensorDiagram should NOT be matched."""
        comp = CompositionDiagram([self.p_x2_3_n23, TensorDiagram([self.bs1, self.p_x2_2])])
        tensor = TensorDiagram([self.q_x2_2, comp, self.q_x2_3_n34])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        assert matches == []

    def test_match_compos_in_tensor(self):  # ruff: ignore[complex-structure, too-many-branches]
        """Identity inside a Composition inside a Tensor should be matched."""
        graph = to_graph(self.tensor1)
        matches1 = self.rule.match(graph)
        assert len(matches1) == 2
        for match in matches1:
            if match["gate_type"] == "Sq":
                assert match == self.match1
            if match["gate_type"] == "Q":
                assert match == self.match2

        graph = to_graph(self.tensor2)
        matches2 = self.rule.match(graph)
        # We need to update the container id which is modified each time a
        # tensor is created
        match1 = self.match1.copy()
        match2 = self.match2.copy()
        match1["container_id"] = self.tensor2.diagrams[1].id
        match2["container_id"] = self.tensor2.diagrams[1].id
        assert len(matches2) == 4
        for match in matches2:
            if match["gate_type"] == "Sq":
                assert match == match1
            if match["gate_type"] == "Q":
                assert match == match2
            if match["gate_type"] == "R":
                assert match == self.match3
            if match["gate_type"] == "F_pair":
                assert match == self.match4

        graph = to_graph(self.tensor3)
        matches3 = self.rule.match(graph)
        # We need to update the container id which is modified each time a
        # tensor is created
        match3 = self.match3.copy()
        match4 = self.match4.copy()
        match1["container_id"] = self.tensor3.diagrams[1].id
        match2["container_id"] = self.tensor3.diagrams[1].id
        match3["container_id"] = _sub(_sub(_sub(self.tensor3, 3), 1), 1).id
        match4["container_id"] = _sub(_sub(_sub(self.tensor3, 3), 1), 1).id
        assert len(matches3) == 6
        for match in matches3:
            if match["gate_type"] == "Sq":
                assert match == match1
            if match["gate_type"] == "Q":
                assert match == match2
            if match["gate_type"] == "R":
                assert match == match3
            if match["gate_type"] == "F_pair":
                assert match == match4
            if match["gate_type"] == "BS":
                assert match == self.match5
            if match["gate_type"] == "D":
                assert match == self.match6

    def test_match_contracted(self):
        """ContractedDiagram does not create chains at its level."""
        contracted = ContractedDiagram(self.q_x2_2, self.q_x2_3, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_contracted_with_composition1(self):
        """Chain inside a composition that is inside a ContractedDiagram should match."""
        comp = CompositionDiagram([self.fourier, self.p_x2_2, self.p_x2_3])
        contracted = ContractedDiagram(comp, self.swap, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "P"
        assert matches[0]["values"] == [self.p_x2_2.phase, self.p_x2_3.phase]
        assert matches[0]["node_ids"] == [self.p_x2_2.id, self.p_x2_3.id]
        assert matches[0]["gate_info"]["num_inputs"] == 1
        assert matches[0]["gate_info"]["num_outputs"] == 1

    def test_match_contracted_with_composition2(self):
        """Chain inside a composition that is inside a ContractedDiagram should match."""
        comp = CompositionDiagram([self.fourier, self.p_x2_2, self.p_x2_3])
        contracted = ContractedDiagram(self.swap, comp, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["container_id"] == comp.id
        assert matches[0]["gate_type"] == "P"
        assert matches[0]["values"] == [self.p_x2_2.phase, self.p_x2_3.phase]
        assert matches[0]["node_ids"] == [self.p_x2_2.id, self.p_x2_3.id]
        assert matches[0]["gate_info"]["num_outputs"] == 1

    def test_match_contracted_nested(self):  # ruff: ignore[complex-structure]
        """Chain inside a composition that is inside a ContractedDiagram should match."""
        q_spider = QSpider(10, 10, self.phase_x2)
        contracted1 = ContractedDiagram(self.tensor2, q_spider, [1, 2, 3], [4, 5, 6], [0, 1, 2, 5], [1, 3, 5, 7])
        contracted2 = ContractedDiagram(q_spider, self.tensor3, [1, 2, 3], [4, 5, 6], [0, 1, 2, 6], [1, 3, 5, 7])
        graph1 = to_graph(contracted1)
        matches1 = self.rule.match(graph1)

        match1 = self.match1.copy()
        match2 = self.match2.copy()
        match1["container_id"] = self.tensor2.diagrams[1].id
        match2["container_id"] = self.tensor2.diagrams[1].id
        assert len(matches1) == 4
        for match in matches1:
            if match["gate_type"] == "Sq":
                assert match == match1
            if match["gate_type"] == "Q":
                assert match == match2
            if match["gate_type"] == "R":
                assert match == self.match3
            if match["gate_type"] == "F_pair":
                assert match == self.match4

        graph2 = to_graph(contracted2)
        matches2 = self.rule.match(graph2)
        match3 = self.match3.copy()
        match4 = self.match4.copy()
        match1["container_id"] = self.tensor3.diagrams[1].id
        match2["container_id"] = self.tensor3.diagrams[1].id
        match3["container_id"] = _sub(_sub(_sub(self.tensor3, 3), 1), 1).id
        match4["container_id"] = _sub(_sub(_sub(self.tensor3, 3), 1), 1).id
        assert len(matches2) == 6
        for match in matches2:
            if match["gate_type"] == "Sq":
                assert match == match1
            if match["gate_type"] == "Q":
                assert match == match2
            if match["gate_type"] == "R":
                assert match == match3
            if match["gate_type"] == "F_pair":
                assert match == match4
            if match["gate_type"] == "BS":
                assert match == self.match5
            if match["gate_type"] == "D":
                assert match == self.match6

    # -------------------------------------------------------------------------
    # 3. Testing reduce_chain()
    # -------------------------------------------------------------------------

    def test_reduce_q_chain(self):
        """Reduce Q(a) ∘ Q(b) → Q(a+b)."""
        values = [self.phase_x2, self.phase_x2_3]
        result = self.rule.reduce_chain("Q", values, {"num_inputs": 1, "num_outputs": 1})
        assert result is not None
        assert result["type"] == "QSpider"
        assert result["phase"] == self.phase_x2_sum
        assert result["num_inputs"] == 1
        assert result["num_outputs"] == 1

        values = [self.phase_x2, self.phase_x2_3, self.phase_x2_3]
        result = self.rule.reduce_chain("Q", values, {"num_inputs": 3, "num_outputs": 4})
        assert result is not None
        assert result["type"] == "QSpider"
        assert result["phase"] == self.phase_x2_sum + self.phase_x2_3
        assert result["num_inputs"] == 3
        assert result["num_outputs"] == 4

    def test_reduce_q_chain_to_identity(self):
        """Q(a) ∘ Q(-a) → identity."""
        phase_pos = ZxPoly({2: 2})
        phase_neg = ZxPoly({2: -2})
        # We need to create values from these
        values = [phase_pos, phase_neg]
        result = self.rule.reduce_chain("Q", values, {"num_inputs": 4, "num_outputs": 6})
        assert result is not None
        assert result["type"] == "QSpider"
        assert result["phase"] == ZxPoly({})
        assert result["num_inputs"] == 4
        assert result["num_outputs"] == 6

    def test_reduce_p_chain(self):
        """Reduce P(a) ∘ P(b) → P(a+b)."""
        values = [self.phase_x2, self.phase_x2_3]
        result = self.rule.reduce_chain("P", values, {"num_inputs": 1, "num_outputs": 1})
        assert result is not None
        assert result["type"] == "PSpider"
        assert result["phase"] == self.phase_x2_sum
        assert result["num_inputs"] == 1
        assert result["num_outputs"] == 1

        values = [self.phase_x2, self.phase_x2_3, self.phase_x2_3]
        result = self.rule.reduce_chain("P", values, {"num_inputs": 3, "num_outputs": 4})
        assert result is not None
        assert result["type"] == "PSpider"
        assert result["phase"] == self.phase_x2_sum + self.phase_x2_3
        assert result["num_inputs"] == 3
        assert result["num_outputs"] == 4

    def test_reduce_r_chain(self):
        """Reduce R(θ) ∘ R(φ) → R(θ+φ)."""
        values = [pi / 4, pi / 3]
        result = self.rule.reduce_chain("R", values)
        assert result is not None
        assert result["type"] == "PhaseRotationGate"
        assert isclose(result["theta"], cast("float", self.ph_rot_sum.theta))

    def test_reduce_bs_chain(self):
        """Reduce BS(θ) ∘ BS(φ) → BS(θ+φ)."""
        values = [pi / 4, pi / 6]
        result = self.rule.reduce_chain("BS", values)
        assert result is not None
        assert result["type"] == "BeamsplitterGate"
        assert isclose(result["theta"], cast("float", self.bs_sum.theta))

    def test_reduce_sq_chain(self):
        """Reduce Sq(τ) ∘ Sq(κ) → Sq(τ·κ)."""
        values = [2.0, 3.0]
        result = self.rule.reduce_chain("Sq", values)
        assert result is not None
        assert result["type"] == "SqueezingGate"
        assert isclose(result["tau"], cast("float", self.sq_prod.tau))

    def test_reduce_sq_chain_to_identity(self):
        """Sq(τ) ∘ Sq(1/τ) → identity."""
        values = [2.0, 0.5]
        result = self.rule.reduce_chain("Sq", values)
        assert result == self.id_q_dict

    def test_reduce_d_chain(self):
        """Reduce D(a) ∘ D(β) → D(a+β)."""
        values = [1.0 + 0.5j, 2.0 + 1.0j]
        result = self.rule.reduce_chain("D", values)
        assert result is not None
        assert result["type"] == "DisplacementGate"
        assert result["alpha"] == 3.0 + 1.5j  # ruff: ignore[float-equality-comparison]

    def test_reduce_f_chain_two(self):
        """F ∘ F → F²."""
        values = ["F", "F"]
        result = self.rule.reduce_chain("F", values)
        assert result is not None
        assert result["type"] == "Fourier2"

    def test_reduce_f_chain_four(self):
        """F ∘ F ∘ F ∘ F → identity."""
        values = ["F", "F", "F", "F"]
        result = self.rule.reduce_chain("F", values)
        assert result == self.id_q_dict

    def test_reduce_f_chain_three(self):
        """F ∘ F ∘ F → Finv."""
        values = ["F", "F", "F"]
        result = self.rule.reduce_chain("F", values)
        assert result is not None
        assert result["type"] == "FourierInv"

    def test_reduce_f_pair(self):
        """F ∘ Finv → identity."""
        values = ["F", "Finv"]
        result = self.rule.reduce_chain("F_pair", values)
        assert result == self.id_q_dict

    def test_reduce_f2_chain_even(self):
        """F² ∘ F² → identity."""
        values = ["F2", "F2"]
        result = self.rule.reduce_chain("F2", values)
        assert result == self.id_q_dict

    def test_reduce_f2_chain_odd(self):
        """F² ∘ F² ∘ F² → F²."""
        values = ["F2", "F2", "F2"]
        result = self.rule.reduce_chain("F2", values)
        assert result is not None
        assert result["type"] == "Fourier2"

    def test_reduce_cs_chain(self):
        """Reduce CSum12(g1) ∘ CSum12(g2) → CSum12(g1+g2)."""
        values = [4, 1]
        result = self.rule.reduce_chain("CSUM", values, {"control": 1, "target": 2})
        assert result is not None
        assert result["type"] == "ControlledSumGate"
        assert isclose(result["gain"], 5)

    def test_reduce_cz_chain(self):
        """Reduce CZ(g1) ∘ CZ(g2) → CZ(g1+g2)."""
        values = [4, 1]
        result = self.rule.reduce_chain("CZ", values)
        assert result is not None
        assert result["type"] == "ControlledZGate"
        assert isclose(result["gain"], 5)

    # -------------------------------------------------------------------------
    # 4. Testing apply_single()
    # -------------------------------------------------------------------------

    def test_apply_single_q_chain(self):
        """Apply single Q chain reduction."""
        comp = CompositionDiagram([self.fourier, self.q_x2_2, self.q_x2_3])
        match = {
            "container_id": comp.id,
            "gate_type": "Q",
            "values": [self.phase_x2, self.phase_x2_3],
            "node_ids": [self.q_x2_2.id, self.q_x2_3.id],
            "gate_info": {"num_inputs": 1, "num_outputs": 1},
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], QSpider)
        assert result.diagrams[1].phase == self.phase_x2_sum
        assert result.diagrams[1].num_inputs == 1
        assert result.diagrams[1].num_outputs == 1

        comp = CompositionDiagram([self.q_x2_2, self.q_x2_3])
        match = {
            "container_id": comp.id,
            "gate_type": "Q",
            "values": [self.phase_x2, self.phase_x2_3],
            "node_ids": [self.q_x2_2.id, self.q_x2_3.id],
            "gate_info": {"num_inputs": 1, "num_outputs": 1},
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
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
            "container_id": comp.id,
            "gate_type": "Q",
            "values": [phase_pos, phase_neg],
            "node_ids": [q_pos.id, q_neg.id],
            "gate_info": {"num_inputs": 2, "num_outputs": 2},
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 4
        assert isinstance(result.diagrams[2], QSpider)
        assert result.diagrams[2].phase == ZxPoly({})
        assert result.num_inputs == 2
        assert result.num_outputs == 2

    def test_apply_single_p_chain(self):
        """Apply single P chain reduction."""
        comp = CompositionDiagram([self.ph_rot1, self.p_x2_2, self.p_x2_3])
        match = {
            "container_id": comp.id,
            "gate_type": "P",
            "values": [self.phase_x2, self.phase_x2_3],
            "node_ids": [self.p_x2_2.id, self.p_x2_3.id],
            "gate_info": {"num_inputs": 1, "num_outputs": 1},
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], PSpider)
        assert result.diagrams[1].phase == self.phase_x2_sum
        assert result.diagrams[1].num_inputs == 1
        assert result.diagrams[1].num_outputs == 1

    def test_apply_single_rot(self):
        """Apply single PhaseRotation chain reduction."""
        comp = CompositionDiagram([self.disp1, self.ph_rot1, self.ph_rot2])
        match = {
            "container_id": comp.id,
            "gate_type": "R",
            "values": [self.ph_rot1.theta, self.ph_rot2.theta],
            "node_ids": [self.ph_rot1.id, self.ph_rot2.id],
            "gate_info": None,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], PhaseRotationGate)
        assert isclose(cast("float", result.diagrams[1].theta), cast("float", self.ph_rot_sum.theta))

        theta = symbols("theta", real=True)
        comp = CompositionDiagram([self.disp1, self.ph_rot1, self.ph_rot2])
        match = {
            "container_id": comp.id,
            "gate_type": "R",
            "values": [theta, -2 * theta],
            "node_ids": [self.ph_rot1.id, self.ph_rot2.id],
            "gate_info": None,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], PhaseRotationGate)
        assert result.diagrams[1].theta == -theta

    def test_apply_single_bs(self):
        """Apply single Beamsplitter chain reduction."""
        comp = CompositionDiagram([self.swap, self.bs1, self.bs2])
        match = {
            "container_id": comp.id,
            "gate_type": "BS",
            "values": [self.bs1.theta, self.bs2.theta],
            "node_ids": [self.bs1.id, self.bs2.id],
            "gate_info": None,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], BeamsplitterGate)
        assert isclose(cast("float", result.diagrams[1].theta), cast("float", self.bs_sum.theta))

        theta = symbols("theta", real=True)
        comp = CompositionDiagram([self.swap, self.bs1, self.bs2])
        match = {
            "container_id": comp.id,
            "gate_type": "BS",
            "values": [theta, 2 * theta],
            "node_ids": [self.bs1.id, self.bs2.id],
            "gate_info": None,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], BeamsplitterGate)
        assert result.diagrams[1].theta == 3 * theta

        comp = CompositionDiagram([self.bs1, self.bs2])
        match = {
            "container_id": comp.id,
            "gate_type": "BS",
            "values": [theta, -theta],
            "node_ids": [self.bs1.id, self.bs2.id],
            "gate_info": None,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == ZxPoly({})
        assert result.num_inputs == 2
        assert result.num_outputs == 2

    def test_apply_single_sq(self):
        """Apply single Squeezing chain reduction."""
        comp = CompositionDiagram([self.fourier, self.sq1, self.sq2])
        match = {
            "container_id": comp.id,
            "gate_type": "Sq",
            "values": [self.sq1.tau, self.sq2.tau],
            "node_ids": [self.sq1.id, self.sq2.id],
            "gate_info": None,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], SqueezingGate)
        assert isclose(cast("float", result.diagrams[1].tau), cast("float", self.sq_prod.tau))

        tau = symbols("tau", real=True)
        comp = CompositionDiagram([self.fourier, self.sq1, self.sq2])
        match = {
            "container_id": comp.id,
            "gate_type": "Sq",
            "values": [tau, self.sq2.tau],
            "node_ids": [self.sq1.id, self.sq2.id],
            "gate_info": None,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], SqueezingGate)
        assert result.diagrams[1].tau == tau * self.sq2.tau

        comp = CompositionDiagram([self.sq1, self.sq2])
        match = {
            "container_id": comp.id,
            "gate_type": "Sq",
            "values": [tau, 1 / tau],
            "node_ids": [self.sq1.id, self.sq2.id],
            "gate_info": None,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == ZxPoly({})
        assert result.num_inputs == 1
        assert result.num_outputs == 1

    def test_apply_single_disp(self):
        """Apply single Displacement chain reduction."""
        a, b = symbols("a b", real=True)
        alpha_sym = cos(a) + exp(I * b)
        disp1 = DisplacementGate(alpha_sym, parametric=True, feedforward=True, measurement_ids={self.meas.id})
        disp2 = DisplacementGate(2)
        comp = CompositionDiagram([self.fourier2, disp1, disp2, self.meas])
        match = {
            "container_id": comp.id,
            "gate_type": "D",
            "values": [disp1.alpha, disp2.alpha],
            "node_ids": [disp1.id, disp2.id],
            "gate_info": None,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 3
        assert isinstance(result.diagrams[1], DisplacementGate)
        assert result.diagrams[1].alpha == alpha_sym + 2
        assert result.diagrams[1].feedforward
        assert result.diagrams[1].measurement_ids == {self.meas.id}

        comp = CompositionDiagram([self.disp1, self.disp2])
        match = {
            "container_id": comp.id,
            "gate_type": "D",
            "values": [disp1.alpha, -disp1.alpha],
            "node_ids": [self.disp1.id, self.disp2.id],
            "gate_info": None,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == ZxPoly({})
        assert result.num_inputs == 1
        assert result.num_outputs == 1

    def test_apply_single_nested(self):
        """Apply single to nested composition and tensor."""
        graph = to_graph(self.tensor1)
        self.rule.apply_single(graph, self.match2)
        result = to_diagram(graph)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], CompositionDiagram)
        assert len(result.diagrams[1].diagrams) == 4
        assert result.diagrams[1].diagrams[0] == self.p_x2_2
        assert result.diagrams[1].diagrams[1] == self.sq1
        assert result.diagrams[1].diagrams[2] == self.sq2
        assert result.diagrams[1].diagrams[3] == self.q_x2_sum

        # Execute apply single a second time on another match
        self.rule.apply_single(graph, self.match1)
        result = to_diagram(graph)
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
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
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
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 3
        assert result.diagrams[0] == self.q_x2_sum
        assert result.diagrams[1] == self.ph_rot_sum
        assert result.diagrams[2] == SqueezingGate(6)

    def test_apply_rule_with_identity_removal(self):
        """Apply rule that removes identity."""
        phase_pos = ZxPoly({2: 2})
        phase_neg = ZxPoly({2: -2})
        q_pos = QSpider(1, 1, phase_pos)
        q_neg = QSpider(1, 1, phase_neg)
        comp = CompositionDiagram([q_pos, q_neg, self.ph_rot1])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == QSpider(1, 1, ZxPoly({}))
        assert result.diagrams[1] == self.ph_rot1

    def test_apply_rule_fourier_chain(self):
        """Apply rule on Fourier chain."""
        fourier = Fourier()
        comp = CompositionDiagram([self.fourier, fourier])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, Fourier2)

    def test_apply_rule_fourier_pair(self):
        """Apply rule on F ∘ Finv pair."""
        comp = CompositionDiagram([self.fourier, self.fourier_inv])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
        assert result == QSpider(1, 1, ZxPoly({}))

    def test_apply_rule_fourier2_chain(self):
        """Apply rule on F² ∘ F² chain."""
        fourier2 = Fourier2()
        comp = CompositionDiagram([self.fourier2, fourier2])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
        assert result == QSpider(1, 1, ZxPoly({}))

    def test_apply_rule_three_r_gates(self):
        """Three rotation gates chain together."""
        r1 = PhaseRotationGate(pi / 6)
        r2 = PhaseRotationGate(pi / 4)
        r3 = PhaseRotationGate(pi / 3)
        comp = CompositionDiagram([r1, r2, r3])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
        assert result == PhaseRotationGate(3 * pi / 4)

    def test_apply_rule_compos_in_tensor(self):
        """Apply rule on nested composition/tensor with chains at multiple levels."""
        graph = to_graph(self.tensor3)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, TensorDiagram)
        assert result.diagrams[0] == self.q_x4_1
        # Check reduction of self.comp1
        assert result.diagrams[1] == self.comp1_red
        assert result.diagrams[2] == self.p_x2_3_n23
        # Check reduction of self.comp2
        assert result.diagrams[3] == self.comp2_red
        assert result.diagrams[4] == self.p_x2_3
        # Check reduction of self.comp3
        theta = pi / 4 + pi / 6
        comp3_red = _sub(result, 5)
        assert isinstance(comp3_red, CompositionDiagram)
        assert comp3_red.diagrams[:2] == [self.q_x2_3_n23, self.p_x2_3_n34]
        bs_tensor = _sub(comp3_red, 2)
        assert isinstance(bs_tensor, TensorDiagram)
        assert isinstance(bs_tensor.diagrams[0], BeamsplitterGate)
        assert isclose(cast("float", bs_tensor.diagrams[0].theta), theta)
        assert bs_tensor.diagrams[1] == CompositionDiagram([
            self.p_x3_2,
            DisplacementGate(8.0 + 2.5j),
            self.p_x3_2,
        ])
        assert bs_tensor.diagrams[2] == self.ph_rot_sum

    def test_apply_rule_compos_in_contracted(self):
        """Chain inside a composition that is inside a ContractedDiagram should match."""
        q_spider = QSpider(10, 10, self.phase_x2)
        contracted1 = ContractedDiagram(self.tensor1, q_spider, [1], [0], [0], [1])
        contracted2 = ContractedDiagram(q_spider, self.tensor2, [4, 5, 6], [1, 2, 3], [1, 3, 5, 7], [0, 1, 2, 5])
        graph1 = to_graph(contracted1)
        self.rule.apply_rule(graph1)
        result1 = to_diagram(graph1)
        graph2 = to_graph(contracted2)
        self.rule.apply_rule(graph2)
        result2 = to_diagram(graph2)
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

    from cvzx.base_gates import Diagram
    from cvzx.nx_graph import to_diagram, to_graph
    from cvzx.visualize_base_gates import visualize_before_after

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
    q_3x2 = QSpider(3, 2, phase_x2)
    q_2x3 = QSpider(2, 3, phase_x2_3)

    p1 = PSpider(1, 1, phase_x2)
    p2 = PSpider(1, 1, phase_x2_3)

    fourier = Fourier()
    fourier2 = Fourier2()
    fourier_inv = FourierInv()
    swap = Swap()

    ph_rot1 = PhaseRotationGate(pi / 4)
    ph_rot2 = PhaseRotationGate(pi / 3)
    ph_rot_sum = PhaseRotationGate(7 * pi / 12)

    bs1 = BeamsplitterGate(pi / 4)
    bs2 = BeamsplitterGate(pi / 6)

    sq1 = SqueezingGate(2.0)
    sq2 = SqueezingGate(3.0)

    disp1 = DisplacementGate(1.0 + 0.5j)
    disp2 = DisplacementGate(2.0 + 1.0j)

    cs1 = ControlledSumGate(gain=1.0, control=1, target=2)
    cs2 = ControlledSumGate(gain=2.0, control=1, target=2)
    cs3 = ControlledSumGate(gain=2.0, control=2, target=1)

    cz1 = ControlledZGate(gain=1.0)
    cz2 = ControlledZGate(gain=2.0)

    # Helper function to apply rule and visualize
    def apply_and_visualize(diagram: Diagram, test_name: str, title_prefix: str = "") -> Diagram:
        """Apply chain reduction rule using graph-based approach."""
        graph = to_graph(diagram)
        rule.apply_rule(graph)
        result = to_diagram(graph)
        visualize_before_after(diagram, result, f"{title_prefix}{test_name}", rule_name)
        return result

    # Test 1: Simple Q-spider chain
    comp1 = CompositionDiagram([q1, q2])
    apply_and_visualize(comp1, "Simple Q Spider Chain")

    # Test 2: Simple P-spider chain
    comp2 = CompositionDiagram([p1, p2])
    apply_and_visualize(comp2, "Simple P Spider Chain")

    # Test 3: Rotation chain
    comp3 = CompositionDiagram([ph_rot1, ph_rot2])
    apply_and_visualize(comp3, "Rotation Chain")

    # Test 4: Squeezing chain
    comp4 = CompositionDiagram([sq1, sq2])
    apply_and_visualize(comp4, "Squeezing Chain")

    # Test 5: Displacement chain
    comp5 = CompositionDiagram([disp1, disp2])
    apply_and_visualize(comp5, "Displacement Chain")

    # Test 6: Beamsplitter chain
    comp6 = CompositionDiagram([bs1, bs2])
    apply_and_visualize(comp6, "Beamsplitter Chain")

    # Test 7: Fourier chain (F ∘ F → F²)
    comp7 = CompositionDiagram([fourier, Fourier()])
    apply_and_visualize(comp7, "Fourier Chain (F ∘ F)")

    # Test 8: Fourier pair (F ∘ Finv → Identity)
    comp8 = CompositionDiagram([fourier, fourier_inv])
    apply_and_visualize(comp8, "Fourier Pair (F ∘ Finv)")

    # Test 9: Multiple chains in one composition
    comp9 = CompositionDiagram([q1, q2, ph_rot1, ph_rot2, sq1, sq2])
    apply_and_visualize(comp9, "Multiple Chains")

    # Test 10: CSUM pair with compatible target and control parameters
    comp10 = CompositionDiagram([swap, cs1, cs2])
    apply_and_visualize(comp10, "CSUM chain")

    # Test 11: CSUM pair with compatible target and control parameters
    comp11 = CompositionDiagram([cs1, cs3, swap])
    apply_and_visualize(comp11, "CSUM chain-No reduction")

    # Test 12: CSUM pair with compatible target and control parameters
    comp12 = CompositionDiagram([swap, cz1, cz2])
    apply_and_visualize(comp12, "CZ chain")

    # Test 13: Mixed Q and P spiders (different types - no chain)
    comp13 = CompositionDiagram([q1, p1])
    apply_and_visualize(comp13, "Mixed Q and P (No Chain)")

    # Test 14: Nested composition inside tensor
    inner_comp = CompositionDiagram([q1, q2])
    tensor = TensorDiagram([inner_comp, sq1, ph_rot1])
    apply_and_visualize(tensor, "Nested Composition in Tensor")

    # Test 15: ContractedDiagram with composition containing chain
    comp_for_contracted = CompositionDiagram([q1, q2, ph_rot1, ph_rot2])
    tensor = TensorDiagram([comp_for_contracted, swap])
    q_spider_large = QSpider(3, 3, phase_x2)
    contracted = ContractedDiagram(tensor, q_spider_large, [0, 1], [1, 2], [0], [1])
    apply_and_visualize(contracted, "Contracted with Composition")

    # Test 16: Chain with different arities (compatible)
    comp15 = CompositionDiagram([q_3x2, q_2x3])
    apply_and_visualize(comp15, "Different Arities (Compatible)")

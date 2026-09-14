"""Unit tests for the chain reduction rewrite rule using the rustworkx graph formalism.

These tests verify the ChainReductionRule implementation on rustworkx graphs, including matching,
application, flattening, and nested structures. All diagram types from
`base_gates` are exercised, and both QSpider and PSpider identities are
covered.

`ChainReductionRule.apply_single` reduces a chain in place: the first
member keeps its own slot and takes the reduced value; every other member
is overwritten with a zero-phase identity spider of its own color and
arity. The rule never invokes another rule. Tests that assert the fully
collapsed form therefore drive `IdentityRule` themselves, after
`ChainReductionRule`, via `_apply_chain_then_identity`.
"""

import unittest
from math import isclose, pi
from typing import cast

from sympy import I, cos, exp, symbols

from cvzx.ir.base import (
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
from cvzx.ir.gates import (
    BeamsplitterGate,
    ControlledSumGate,
    ControlledZGate,
    DisplacementGate,
    PhaseRotationGate,
    SqueezingGate,
)
from cvzx.backends.rx.graph import CVZXGraph, to_diagram, to_graph
from cvzx.backends.rx.rules import ChainReductionRule, IdentityRule, _get_id_map  # ruff: ignore[import-private-name]


def _sub(diagram: Diagram, index: int) -> Diagram:
    """Index into a container `Diagram`'s children, for navigating a fixture's known shape."""
    assert isinstance(diagram, (CompositionDiagram, TensorDiagram, ContractedDiagram))
    return diagram.diagrams[index]


def _apply_chain_then_identity(graph: CVZXGraph) -> None:
    """Apply `ChainReductionRule`, then `IdentityRule`, in that order, in place."""
    ChainReductionRule().apply_rule(graph)
    IdentityRule().apply_rule(graph)


class TestChainReductionRule(unittest.TestCase):
    """Test suite for ChainReductionRule using rustworkx graphs."""

    def setUp(self):
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
        self.q_x2_3_n23_c = QSpider(2, 3, self.phase_x2_3)
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
        self.p_x3_2_b = PSpider(1, 1, self.phase_x3)
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
        self.ph_rot1_c = PhaseRotationGate(pi / 4)
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
        self.meas = QSpider(1, 0, ZxPoly({1: m}), True)

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
                self.ph_rot_sum,
            ]),
        ])
        self.comp3 = CompositionDiagram([
            self.q_x2_3_n23_c,
            self.p_x2_3_n34,
            TensorDiagram([
                CompositionDiagram([self.bs3, self.bs2]),
                TensorDiagram([
                    CompositionDiagram([self.p_x3_2, self.disp3, self.disp1, self.disp2, self.p_x3_2_b]),
                    self.ph_rot1_c,
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
            "container_id": self.tensor1.diagrams[1].id,
            "gate_type": "Sq",
            "values": [2.0, 3.0],
            "node_ids": [self.sq1.id, self.sq2.id],
            "gate_info": None,
            "identity_chain": [],
            "same_parent": True,
        }
        self.match2 = {
            "container_id": self.tensor1.diagrams[1].id,
            "gate_type": "Q",
            "values": [self.phase_x2, self.phase_x2_3],
            "node_ids": [self.q_x2_2.id, self.q_x2_3.id],
            "gate_info": {"num_inputs": 1, "num_outputs": 1},
            "identity_chain": [],
            "same_parent": True,
        }
        self.match3 = {
            "container_id": _sub(_sub(_sub(self.tensor2, 3), 1), 1).id,
            "gate_type": "R",
            "values": [self.ph_rot1.theta, self.ph_rot2.theta],
            "node_ids": [self.ph_rot1.id, self.ph_rot2.id],
            "gate_info": None,
            "identity_chain": [],
            "same_parent": True,
        }
        self.match4 = {
            "container_id": _sub(_sub(_sub(self.tensor2, 3), 1), 1).id,
            "gate_type": "F_pair",
            "values": ["F", "Finv"],
            "node_ids": [self.fourier.id, self.fourier_inv.id],
            "gate_info": None,
            "identity_chain": [],
            "same_parent": True,
        }
        self.match5 = {
            "container_id": _sub(_sub(_sub(self.tensor3, 5), 2), 0).id,
            "gate_type": "BS",
            "values": [self.bs3.theta, self.bs2.theta],
            "node_ids": [self.bs3.id, self.bs2.id],
            "gate_info": None,
            "identity_chain": [],
            "same_parent": True,
        }
        self.match6 = {
            "container_id": _sub(_sub(_sub(self.tensor3, 5), 2), 1).id,
            "gate_type": "D",
            "values": [self.disp3.alpha, self.disp1.alpha, self.disp2.alpha],
            "node_ids": [self.disp3.id, self.disp1.id, self.disp2.id],
            "gate_info": None,
            "identity_chain": [],
            "same_parent": True,
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
        """Q-spider with monomial phase returns ('Q', phase, arities)."""
        graph = to_graph(self.q_x2_2)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, id_map, self.q_x2_2.id)
        assert gate_info is not None
        assert gate_type == "Q"
        assert value == self.phase_x2
        assert gate_info["num_inputs"] == 1
        assert gate_info["num_outputs"] == 1

        graph = to_graph(self.q_x2_3_n23)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, id_map, self.q_x2_3_n23.id)
        assert gate_info is not None
        assert gate_type == "Q"
        assert value == self.phase_x2_3
        assert gate_info["num_inputs"] == 2
        assert gate_info["num_outputs"] == 3

    def test_get_gate_info_q_spider_mixed(self):
        """Q-spider with mixed phase returns ('Q', phase, arities)."""
        graph = to_graph(self.q_mixed)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, id_map, self.q_mixed.id)
        assert gate_info is not None
        assert gate_type == "Q"
        assert value == self.phase_mixed
        assert gate_info["num_inputs"] == 1
        assert gate_info["num_outputs"] == 1

    def test_get_gate_info_p_spider_monomial(self):
        """P-spider with monomial phase returns ('P', phase, arities)."""
        graph = to_graph(self.p_x2_2)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, id_map, self.p_x2_2.id)
        assert gate_info is not None
        assert gate_type == "P"
        assert value == self.phase_x2
        assert gate_info["num_inputs"] == 1
        assert gate_info["num_outputs"] == 1

    def test_get_gate_info_p_spider_mixed(self):
        """P-spider with mixed phase returns ('P', phase, arities)."""
        graph = to_graph(self.p_mixed)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, id_map, self.p_mixed.id)
        assert gate_info is not None
        assert gate_type == "P"
        assert value == self.phase_mixed
        assert gate_info["num_inputs"] == 1
        assert gate_info["num_outputs"] == 1

    def test_get_gate_info_phase_rotation(self):
        """PhaseRotationGate returns ('R', theta)."""
        graph = to_graph(self.ph_rot1)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, id_map, self.ph_rot1.id)
        assert gate_type == "R"
        assert isclose(value, pi / 4)

    def test_get_gate_info_beamsplitter(self):
        """BeamsplitterGate returns ('BS', theta)."""
        graph = to_graph(self.bs1)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, id_map, self.bs1.id)
        assert gate_type == "BS"
        assert isclose(value, pi / 4)

    def test_get_gate_info_squeezing(self):
        """SqueezingGate returns ('Sq', tau)."""
        graph = to_graph(self.sq1)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, id_map, self.sq1.id)
        assert gate_type == "Sq"
        assert isclose(value, 2.0)

    def test_get_gate_info_displacement(self):
        """DisplacementGate returns ('D', alpha)."""
        graph = to_graph(self.disp1)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, id_map, self.disp1.id)
        assert gate_type == "D"
        assert value == 1.0 + 0.5j

    def test_get_gate_info_fourier(self):
        """Fourier returns ('F', 'F')."""
        graph = to_graph(self.fourier)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, id_map, self.fourier.id)
        assert gate_type == "F"
        assert value == "F"

    def test_get_gate_info_fourier_inv(self):
        """FourierInv returns ('F', 'Finv')."""
        graph = to_graph(self.fourier_inv)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, id_map, self.fourier_inv.id)
        assert gate_type == "F"
        assert value == "Finv"

    def test_get_gate_info_fourier2(self):
        """Fourier2 returns ('F2', 'F2')."""
        graph = to_graph(self.fourier2)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, id_map, self.fourier2.id)
        assert gate_type == "F2"
        assert value == "F2"

    def test_get_gate_info_swap_returns_none(self):
        """Swap returns None (not reducible)."""
        graph = to_graph(self.swap)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, id_map, self.swap.id)
        assert gate_type is None
        assert value is None

    def test_get_gate_info_csum(self):
        """CSum gate returns ('CSUM', gain, control/target)."""
        graph = to_graph(self.cs)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, gate_info = self.rule.get_gate_info(graph.graph, id_map, self.cs.id)
        assert gate_info is not None
        assert gate_type == "CSUM"
        assert value == self.cs.gain
        assert gate_info["control"] == 1
        assert gate_info["target"] == 2

    def test_get_gate_info_cz(self):
        """CZ gate returns ('CZ', gain)."""
        graph = to_graph(self.cz1)
        id_map = {graph.graph[i]["id"]: i for i in graph.graph.node_indices()}
        gate_type, value, _ = self.rule.get_gate_info(graph.graph, id_map, self.cz1.id)
        assert gate_type == "CZ"
        assert value == self.cz1.gain

    # -------------------------------------------------------------------------
    # 2. Testing match()
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
        """Q-spiders with different degrees chain together."""
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
        """Q-spiders with different compatible arities chain together."""
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
        """Q-spider with mixed polynomial still chains."""
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
        """P-spiders with different degrees chain together."""
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
        """P-spiders with different compatible arities chain together."""
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
        """P-spider with mixed polynomial still chains."""
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
        """Chains do not form across tensor lanes."""
        comp = CompositionDiagram([self.p_x2_3_n23, TensorDiagram([self.bs1, self.p_x2_2])])
        tensor = TensorDiagram([self.q_x2_2, comp, self.q_x2_3_n34])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        assert matches == []

    def test_match_compos_in_tensor(self):
        """Chains inside compositions inside a tensor are matched."""
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
        """Chain inside a composition inside a ContractedDiagram is found."""
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
        """Chain inside a composition inside a ContractedDiagram is found."""
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

    def test_match_contracted_nested(self):
        """Chain inside a composition inside a ContractedDiagram is found (nested)."""
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
        assert result["alpha"] == 3.0 + 1.5j

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
        """Q chain: survivor + zero-phase identity, then IdentityRule collapses."""
        comp = CompositionDiagram([self.fourier, self.q_x2_2, self.q_x2_3])
        match = {
            "container_id": comp.id,
            "gate_type": "Q",
            "values": [self.phase_x2, self.phase_x2_3],
            "node_ids": [self.q_x2_2.id, self.q_x2_3.id],
            "gate_info": {"num_inputs": 1, "num_outputs": 1},
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        mid = to_diagram(graph)
        assert isinstance(mid, CompositionDiagram)
        assert len(mid.diagrams) == 3
        assert isinstance(mid.diagrams[1], QSpider)
        assert mid.diagrams[1].phase == self.phase_x2_sum
        assert isinstance(mid.diagrams[2], QSpider)
        assert mid.diagrams[2].phase == ZxPoly({})

        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
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
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == self.phase_x2_sum
        assert result.num_inputs == 1
        assert result.num_outputs == 1

    def test_apply_single_q_chain_to_identity(self):
        """Q chain that reduces to identity leaves a zero-phase QSpider."""
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
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 4
        assert isinstance(result.diagrams[2], QSpider)
        assert result.diagrams[2].phase == ZxPoly({})
        assert result.num_inputs == 2
        assert result.num_outputs == 2

    def test_apply_single_p_chain(self):
        """P chain: survivor + identity, then IdentityRule collapses."""
        comp = CompositionDiagram([self.ph_rot1, self.p_x2_2, self.p_x2_3])
        match = {
            "container_id": comp.id,
            "gate_type": "P",
            "values": [self.phase_x2, self.phase_x2_3],
            "node_ids": [self.p_x2_2.id, self.p_x2_3.id],
            "gate_info": {"num_inputs": 1, "num_outputs": 1},
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], PSpider)
        assert result.diagrams[1].phase == self.phase_x2_sum
        assert result.diagrams[1].num_inputs == 1
        assert result.diagrams[1].num_outputs == 1

    def test_apply_single_rot(self):
        """Rotation chain: survivor + identity, then IdentityRule collapses."""
        comp = CompositionDiagram([self.disp1, self.ph_rot1, self.ph_rot2])
        match = {
            "container_id": comp.id,
            "gate_type": "R",
            "values": [self.ph_rot1.theta, self.ph_rot2.theta],
            "node_ids": [self.ph_rot1.id, self.ph_rot2.id],
            "gate_info": None,
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
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
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], PhaseRotationGate)
        assert result.diagrams[1].theta == -theta

    def test_apply_single_bs(self):
        """Beamsplitter chain: survivor keeps the slot, second member is spliced away entirely.

        Unlike a `(1, 1)` `QSpider`/`PSpider` chain, a wide gate's "extra"
        chain member has no same-arity zero-phase-spider identity
        representation `IdentityRule` knows how to prune (only a bare
        `(1, 1)` spider does -- see `apply_single`'s own docstring), so
        it's removed from the graph outright via the general splice-and-
        propagate path. This holds even when the reduced gate itself
        collapses to a `(2, 2)` zero-phase `QSpider` (a fully-cancelled
        `theta + (-theta) = 0` chain): that canonical zero-BS-as-identity
        representation still isn't something `IdentityRule` prunes, but it
        no longer needs to be pruned -- there's nothing else left to
        splice away.
        """
        comp = CompositionDiagram([self.swap, self.bs1, self.bs2])
        match = {
            "container_id": comp.id,
            "gate_type": "BS",
            "values": [self.bs1.theta, self.bs2.theta],
            "node_ids": [self.bs1.id, self.bs2.id],
            "gate_info": None,
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.swap
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
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
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
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
        result = to_diagram(graph)
        # Only the (now-canonical) zero-phase (2, 2) survivor is left --
        # the composition itself flattens away since a single element
        # doesn't need one.
        assert isinstance(result, QSpider)
        assert result.num_inputs == 2
        assert result.num_outputs == 2
        assert result.phase == ZxPoly({})

    def test_apply_single_sq(self):
        """Squeezing chain: survivor + identity, then IdentityRule collapses."""
        comp = CompositionDiagram([self.fourier, self.sq1, self.sq2])
        match = {
            "container_id": comp.id,
            "gate_type": "Sq",
            "values": [self.sq1.tau, self.sq2.tau],
            "node_ids": [self.sq1.id, self.sq2.id],
            "gate_info": None,
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
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
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
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
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == ZxPoly({})
        assert result.num_inputs == 1
        assert result.num_outputs == 1

    def test_apply_single_disp(self):
        """Displacement chain: survivor + identity, then IdentityRule collapses."""
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
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
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
            "identity_chain": [],
            "same_parent": True,
        }
        graph = to_graph(comp)
        self.rule.apply_single(graph, match)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, QSpider)
        assert result.phase == ZxPoly({})
        assert result.num_inputs == 1
        assert result.num_outputs == 1

    def test_apply_single_nested(self):
        """Apply single to nested composition and tensor: then IdentityRule collapses."""
        graph = to_graph(self.tensor1)
        self.rule.apply_single(graph, self.match2)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert isinstance(result.diagrams[1], CompositionDiagram)
        assert len(result.diagrams[1].diagrams) == 4
        assert result.diagrams[1].diagrams[0] == self.p_x2_2
        assert result.diagrams[1].diagrams[1] == self.sq1
        assert result.diagrams[1].diagrams[2] == self.sq2
        assert result.diagrams[1].diagrams[3] == self.q_x2_sum

        self.rule.apply_single(graph, self.match1)
        graph.rebuild_registry()
        IdentityRule().apply_rule(graph)
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
        _apply_chain_then_identity(graph)
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
        _apply_chain_then_identity(graph)
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
        _apply_chain_then_identity(graph)
        result = to_diagram(graph)
        assert result == self.ph_rot1

    def test_apply_rule_fourier_chain(self):
        """Apply rule on Fourier chain."""
        fourier = Fourier()
        comp = CompositionDiagram([self.fourier, fourier])
        graph = to_graph(comp)
        _apply_chain_then_identity(graph)
        result = to_diagram(graph)
        assert isinstance(result, Fourier2)

    def test_apply_rule_fourier_pair(self):
        """Apply rule on F ∘ Finv pair."""
        comp = CompositionDiagram([self.fourier, self.fourier_inv])
        graph = to_graph(comp)
        _apply_chain_then_identity(graph)
        result = to_diagram(graph)
        assert result == QSpider(1, 1, ZxPoly({}))

    def test_apply_rule_fourier2_chain(self):
        """Apply rule on F² ∘ F² chain."""
        fourier2 = Fourier2()
        comp = CompositionDiagram([self.fourier2, fourier2])
        graph = to_graph(comp)
        _apply_chain_then_identity(graph)
        result = to_diagram(graph)
        assert result == QSpider(1, 1, ZxPoly({}))

    def test_apply_rule_three_r_gates(self):
        """Three rotation gates chain together."""
        r1 = PhaseRotationGate(pi / 6)
        r2 = PhaseRotationGate(pi / 4)
        r3 = PhaseRotationGate(pi / 3)
        comp = CompositionDiagram([r1, r2, r3])
        graph = to_graph(comp)
        _apply_chain_then_identity(graph)
        result = to_diagram(graph)
        assert result == PhaseRotationGate(3 * pi / 4)

    def test_apply_rule_compos_in_tensor(self):
        """Apply rule on nested composition/tensor with chains at multiple levels."""
        graph = to_graph(self.tensor3)
        _apply_chain_then_identity(graph)
        result = to_diagram(graph)
        assert isinstance(result, TensorDiagram)
        assert result.diagrams[0] == self.q_x4_1
        assert result.diagrams[1] == self.comp1_red
        assert result.diagrams[2] == self.p_x2_3_n23
        assert result.diagrams[3] == self.comp2_red
        assert result.diagrams[4] == self.p_x2_3
        theta = pi / 4 + pi / 6
        comp3_red = _sub(result, 5)
        assert isinstance(comp3_red, CompositionDiagram)
        assert comp3_red.diagrams[:2] == [self.q_x2_3_n23_c, self.p_x2_3_n34]
        bs_tensor = _sub(comp3_red, 2)
        assert isinstance(bs_tensor, TensorDiagram)
        # The BS chain's "extra" member has no same-arity zero-phase-spider
        # identity representation (only a bare `(1, 1)` spider does -- see
        # `apply_single`'s own docstring), so it's spliced out of the graph
        # entirely rather than left for `IdentityRule` to prune: the slot
        # holds just the surviving, reduced BeamsplitterGate.
        bs_slot = bs_tensor.diagrams[0]
        assert isinstance(bs_slot, BeamsplitterGate)
        assert isclose(cast("float", bs_slot.theta), theta)

        assert bs_tensor.diagrams[1] == CompositionDiagram([
            self.p_x3_2,
            DisplacementGate(8.0 + 2.5j),
            self.p_x3_2_b,
        ])
        assert bs_tensor.diagrams[2] == self.ph_rot1_c

    def test_apply_rule_compos_in_contracted(self):
        """Chain inside a composition that is inside a ContractedDiagram."""
        q_spider = QSpider(10, 10, self.phase_x2)
        contracted1 = ContractedDiagram(self.tensor1, q_spider, [1], [0], [0], [1])
        contracted2 = ContractedDiagram(q_spider, self.tensor3, [4, 5, 6], [1, 2, 3], [1, 3, 5, 7], [0, 1, 2, 5])
        graph1 = to_graph(contracted1)
        _apply_chain_then_identity(graph1)
        result1 = to_diagram(graph1)
        graph2 = to_graph(contracted2)
        _apply_chain_then_identity(graph2)
        result2 = to_diagram(graph2)

        tensor1_red = TensorDiagram([self.q_x4_1, self.comp1_red])
        assert result1 == ContractedDiagram(tensor1_red, q_spider, [1], [0], [0], [1])

        assert result2.num_inputs == contracted2.num_inputs
        assert result2.num_outputs == contracted2.num_outputs

        inner_tensor = result2.second
        assert isinstance(inner_tensor, TensorDiagram)
        assert inner_tensor.diagrams[0] == self.q_x4_1
        assert inner_tensor.diagrams[1] == self.comp1_red
        assert inner_tensor.diagrams[2] == self.p_x2_3_n23
        comp2_red = inner_tensor.diagrams[3]
        assert isinstance(comp2_red, CompositionDiagram)
        assert comp2_red.diagrams[0] == self.q_x2_3_n23
        inner_tensor_lane = comp2_red.diagrams[1]
        assert isinstance(inner_tensor_lane, TensorDiagram)
        assert inner_tensor_lane.diagrams[0] == self.bs1
        inner_ph = inner_tensor_lane.diagrams[1]
        assert isinstance(inner_ph, PhaseRotationGate)
        assert isclose(cast("float", inner_ph.theta), cast("float", self.ph_rot_sum.theta))

    # -------------------------------------------------------------------------
    # 6. Chain-chasing through identity spiders AND `Swap`
    # -------------------------------------------------------------------------

    def test_match_through_identity_same_container(self):
        """Two chainable rotations separated by one identity still chain."""
        r1 = PhaseRotationGate(pi / 6)
        r2 = PhaseRotationGate(pi / 4)
        id1 = QSpider(1, 1, ZxPoly({}))
        comp = CompositionDiagram([r1, id1, r2])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "R"
        assert matches[0]["node_ids"] == [r1.id, r2.id]
        assert matches[0]["identity_chain"] == [id1.id]
        assert matches[0]["same_parent"] is False

    def test_apply_rule_through_identity_same_container(self):
        """Applying the rule folds both rotations; IdentityRule cleans up."""
        r1 = PhaseRotationGate(pi / 6)
        r2 = PhaseRotationGate(pi / 4)
        id1 = QSpider(1, 1, ZxPoly({}))
        comp = CompositionDiagram([r1, id1, r2])
        graph = to_graph(comp)
        _apply_chain_then_identity(graph)
        result = to_diagram(graph)
        assert isinstance(result, PhaseRotationGate)
        assert isclose(cast("float", result.theta), pi / 6 + pi / 4)

    def test_match_cross_container_chain(self):
        """Two chainable rotations in DIFFERENT immediate containers."""
        r1 = PhaseRotationGate(pi / 6)
        r2 = PhaseRotationGate(pi / 4)
        filler1 = Fourier()
        filler3 = QSpider(1, 1, ZxPoly({2: 5.0}))

        inner_comp = CompositionDiagram([filler1, r1])
        lane1 = TensorDiagram([inner_comp, Fourier()])
        lane2 = TensorDiagram([r2, filler3])
        comp = CompositionDiagram([lane1, lane2])

        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "R"
        assert matches[0]["node_ids"] == [r1.id, r2.id]
        assert matches[0]["identity_chain"] == []
        assert matches[0]["same_parent"] is False
        assert matches[0]["container_id"] == ("cross", r1.id, r2.id)

    def test_apply_rule_cross_container_chain(self):
        """Applying the cross-container chain folds it in place."""
        r1 = PhaseRotationGate(pi / 6)
        r2 = PhaseRotationGate(pi / 4)
        filler1 = Fourier()
        filler3 = QSpider(1, 1, ZxPoly({2: 5.0}))

        inner_comp = CompositionDiagram([filler1, r1])
        lane1 = TensorDiagram([inner_comp, Fourier()])
        lane2 = TensorDiagram([r2, filler3])
        comp = CompositionDiagram([lane1, lane2])

        graph = to_graph(comp)
        _apply_chain_then_identity(graph)
        result = to_diagram(graph)
        graph2 = to_graph(result)
        assert len(self.rule.match(graph2)) == 0

        rotations = [d for d in result.diagrams[0].diagrams[0].diagrams if isinstance(d, PhaseRotationGate)]
        assert len(rotations) == 1
        assert isclose(cast("float", rotations[0].theta), pi / 6 + pi / 4)
        assert result.diagrams[1].diagrams[1] == filler3

    def test_match_cross_container_chain_with_swap_and_identities(self):
        """Cross-container chain with identities and a Swap."""
        r1 = PhaseRotationGate(pi / 6)
        r2 = PhaseRotationGate(pi / 4)
        filler1 = Fourier()
        id_after_r1 = QSpider(1, 1, ZxPoly({}))
        filler_state = Fourier()
        id_a = PSpider(1, 1, ZxPoly({}))
        id_b = QSpider(1, 1, ZxPoly({}))
        filler_final = Fourier()

        inner_comp = CompositionDiagram([filler1, r1, id_after_r1])
        lane1 = TensorDiagram([inner_comp, filler_state])
        swap = Swap()
        mid = TensorDiagram([id_a, id_b])
        lane2 = TensorDiagram([filler_final, r2])
        comp = CompositionDiagram([lane1, swap, mid, lane2])

        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["gate_type"] == "R"
        assert matches[0]["node_ids"] == [r1.id, r2.id]
        assert set(matches[0]["identity_chain"]) == {id_after_r1.id, swap.id, id_b.id}
        assert id_a.id not in matches[0]["identity_chain"]
        assert matches[0]["same_parent"] is False

    def test_apply_rule_cross_container_chain_with_swap_and_identities(self):
        """A single `apply_rule` call folds both chains crossing the `Swap`."""
        r1 = PhaseRotationGate(pi / 6)
        r2 = PhaseRotationGate(pi / 4)
        filler1 = Fourier()
        id_after_r1 = QSpider(1, 1, ZxPoly({}))
        filler_state = Fourier()
        id_a = PSpider(1, 1, ZxPoly({}))
        id_b = QSpider(1, 1, ZxPoly({}))
        filler_final = Fourier()

        inner_comp = CompositionDiagram([filler1, r1, id_after_r1])
        lane1 = TensorDiagram([inner_comp, filler_state])
        swap = Swap()
        mid = TensorDiagram([id_a, id_b])
        lane2 = TensorDiagram([filler_final, r2])
        comp = CompositionDiagram([lane1, swap, mid, lane2])

        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        IdentityRule().apply_rule(graph)
        result = to_diagram(graph)
        assert len(self.rule.match(to_graph(result))) == 0

        lane1_result = result.diagrams[0]
        inner_result = lane1_result.diagrams[0]
        rotations = [d for d in inner_result.diagrams if isinstance(d, PhaseRotationGate)]
        assert len(rotations) == 1
        assert isclose(cast("float", rotations[0].theta), pi / 6 + pi / 4)
        assert isinstance(lane1_result.diagrams[1], Fourier2)

    def test_apply_rule_chain_through_swap_nested_in_tensor_lane(self):
        """The chase compensates correctly when the `Swap` sits inside a `TensorDiagram`."""
        r1 = PhaseRotationGate(pi / 6)
        r2 = PhaseRotationGate(pi / 4)
        id_after_r1 = QSpider(1, 1, ZxPoly({}))
        id_a = PSpider(1, 1, ZxPoly({}))
        id_b = QSpider(1, 1, ZxPoly({}))

        pad1 = PhaseRotationGate(0.111)
        pad2 = PhaseRotationGate(0.222)
        pad3 = PhaseRotationGate(0.333)
        pad4 = PhaseRotationGate(0.444)

        inner_comp = CompositionDiagram([Fourier(), r1, id_after_r1])
        lane1 = TensorDiagram([pad1, inner_comp, Fourier()])
        swap_stage = TensorDiagram([pad2, Swap()])
        mid = TensorDiagram([pad3, id_a, id_b])
        lane2 = TensorDiagram([pad4, Fourier(), r2])
        comp = CompositionDiagram([lane1, swap_stage, mid, lane2])

        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        IdentityRule().apply_rule(graph)
        result = to_diagram(graph)
        assert len(self.rule.match(to_graph(result))) == 0

        pad_lane = result.diagrams[0].diagrams[0]
        assert isinstance(pad_lane, PhaseRotationGate)
        assert isclose(cast("float", pad_lane.theta), 0.111 + 0.222 + 0.333 + 0.444)

        inner_result = result.diagrams[0].diagrams[1]
        rotations = [d for d in inner_result.diagrams if isinstance(d, PhaseRotationGate)]
        assert len(rotations) == 1
        assert isclose(cast("float", rotations[0].theta), pi / 6 + pi / 4)

        assert isinstance(result.diagrams[0].diagrams[2], Fourier2)


class TestChainReductionResetToIdentityClearsParamMeasurementMap(unittest.TestCase):
    """Test that `_reset_to_identity` clears `param_measurement_map` (Phase 3 fix).

    Mirrors `tests/rewrite_rules/test_nx_chain_reduction_rules.py::
    TestChainReductionResetToIdentityClearsParamMeasurementMap`. A node
    reset to identity by chain reduction (every chain member other than
    the survivor) must lose its feedforward provenance entirely, not just
    have its phase zeroed while a stale map lingers.
    """

    def test_reset_to_identity_clears_param_measurement_map(self):
        """A feedforward node with a param_measurement_map is fully cleared on reset."""
        m = symbols("m", real=True)
        spider = QSpider(1, 1, ZxPoly({1: m}), parametric=True, param_measurement_map={m: {7}})
        cvzx_graph = to_graph(spider)
        id_map = _get_id_map(cvzx_graph.graph)

        ChainReductionRule._reset_to_identity(cvzx_graph.graph, id_map, spider.id)  # ruff: ignore[private-member-access]

        attrs = cvzx_graph.graph[id_map[spider.id]]
        assert attrs["param_measurement_map"] == {}
        assert attrs["feedforward"] is None
        assert attrs["measurement_ids"] is None
        assert attrs["phase"] == ZxPoly({})
        assert attrs["type"] == "QSpider"

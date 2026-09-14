"""Unit tests for the identity rewrite rule using the rustworkx graph formalism.

These tests verify the IdentityRule implementation on rustworkx graphs,
including matching, application, flattening, and nested structures. All
diagram types from `base_gates` are exercised, and both QSpider and PSpider
identities are covered.
"""

import unittest
from math import pi

import rustworkx as rx

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
from cvzx.rx_graph import to_diagram, to_graph
from cvzx.rx_rewrite_rules import IdentityRule, apply_rule_to_diagram


def _sub(diagram: Diagram, index: int) -> Diagram:
    """Index into a container `Diagram`'s children, for navigating a fixture's known shape."""
    assert isinstance(diagram, (CompositionDiagram, TensorDiagram, ContractedDiagram))
    return diagram.diagrams[index]


def _get_node_attrs(graph: rx.PyDiGraph, node_id: int) -> dict:
    """Helper to extract payload attributes for a specific diagram node ID."""
    for idx in graph.node_indices():
        if graph[idx].get("id") == node_id:
            return graph[idx]
    raise KeyError(f"Node ID {node_id} not found in PyDiGraph.")


class TestIdentityRule(unittest.TestCase):
    """Test suite for IdentityRule (rustworkx graph-based)."""

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
        # Basic gates
        self.disp = DisplacementGate(alpha=1.0 + 0.5j)
        self.ph_rot = PhaseRotationGate(theta=pi / 4)
        self.sq_gate = SqueezingGate(tau=0.5)
        self.ctrl_sum_gate1 = ControlledSumGate(gain=1.0, control=1, target=2)
        self.ctrl_sum_gate2 = ControlledSumGate(gain=2.0, control=2, target=1)
        self.ctrl_z_gate1 = ControlledZGate(gain=1.0)
        self.ctrl_z_gate2 = ControlledZGate(gain=2.0)
        self.beam_splitter1 = BeamsplitterGate(theta=pi / 4)
        self.beam_splitter2 = BeamsplitterGate(theta=pi / 6)

        # Compositions
        self.comp1 = CompositionDiagram([self.non_id_p, self.sq_gate, self.id_q, self.id_p])
        self.comp2 = CompositionDiagram([
            self.non_id_p_3x3,
            TensorDiagram([
                self.beam_splitter1,
                CompositionDiagram([self.fourier, PSpider(1, 1, self.zero_phase), PSpider(1, 1, self.zero_phase)]),
            ]),
        ])
        self.comp3 = CompositionDiagram([
            self.non_id_q_4x4,
            self.non_id_q_4x4,
            TensorDiagram([
                self.beam_splitter1,
                TensorDiagram([CompositionDiagram([PSpider(1, 1, self.zero_phase), self.fourier_inv]), self.ph_rot]),
            ]),
        ])
        # Nested Diagrams
        self.tensor1 = TensorDiagram([QSpider(1, 1, self.zero_phase), self.comp1])
        self.tensor2 = TensorDiagram([QSpider(1, 1, self.zero_phase), self.comp1, self.non_id_q_3x3, self.comp2])
        self.tensor3 = TensorDiagram([
            QSpider(1, 1, self.zero_phase),
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
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["node_id"] == self.id_q.id
        assert matches[0]["container_id"] == comp.id

    def test_match_simple_composition2(self):
        """Identity inside a CompositionDiagram should be matched."""
        comp = CompositionDiagram([self.fourier, self.id_p, self.fourier_inv])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["node_id"] == self.id_p.id

    def test_match_simple_composition3(self):
        """Multiple identities should be matched."""
        comp = CompositionDiagram([self.fourier, self.id_p, self.fourier_inv, self.id_q])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 2
        node_ids = [m["node_id"] for m in matches]
        assert self.id_p.id in node_ids
        assert self.id_q.id in node_ids

    def test_match_nested_composition(self):
        """Identity inside a nested CompositionDiagram should be matched."""
        inner = CompositionDiagram([self.id_q, self.fourier])
        outer = CompositionDiagram([inner, self.fourier2])
        graph = to_graph(outer)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["node_id"] == self.id_q.id
        container_id = matches[0]["container_id"]
        assert container_id == inner.id

    def test_match_tensor(self):
        """Identity inside a TensorDiagram should not be matched."""
        tensor = TensorDiagram([self.id_q, self.fourier, self.id_q3x3])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_tensor2(self):
        """An identity nested inside a Tensor lane is NOT matched."""
        id_q_inner = QSpider(1, 1, self.zero_phase)
        comp = CompositionDiagram([self.non_id_p_3x3, TensorDiagram([self.beam_splitter1, id_q_inner])])
        id_q_outer = QSpider(1, 1, self.zero_phase)
        tensor = TensorDiagram([id_q_outer, comp, self.non_id_q_3x3])
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_compos_in_tensor(self):
        """Complex nested structure should find all identities."""
        graph1 = to_graph(self.tensor1)
        graph2 = to_graph(self.tensor2)
        graph3 = to_graph(self.tensor3)

        matches1 = self.rule.match(graph1)
        matches2 = self.rule.match(graph2)
        matches3 = self.rule.match(graph3)

        # tensor1: id_q and id_p in comp1
        assert len(matches1) == 2

        # tensor2: id_q at top level + id_q and id_p in comp1 + id_p and id_p in comp2
        assert len(matches2) == 4

        # tensor3: all identities from tensor2 + id_p in comp3
        assert len(matches3) == 5

    def test_match_contracted_direct(self):
        """Identity as first or second of ContractedDiagram should not be matched."""
        contracted = ContractedDiagram(self.id_q, self.fourier, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    def test_match_contracted_with_composition1(self):
        """Identity inside a composition inside a ContractedDiagram should be matched."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(comp, self.swap, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["node_id"] == self.id_q.id

    def test_match_contracted_with_composition2(self):
        """Identity inside a composition inside a ContractedDiagram should be matched."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(self.swap, comp, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert len(matches) == 1
        assert matches[0]["node_id"] == self.id_q.id

    def test_match_does_not_match_non_identity(self):
        """Non identity spiders should not be matched."""
        comp = CompositionDiagram([self.non_id_q, self.fourier])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        assert len(matches) == 0

    # -------------------------------------------------------------------------
    # 2. Testing apply_single()
    # -------------------------------------------------------------------------

    def test_apply_single_wrong_match(self):
        """Make sure the diagram is unchanged when the match path is wrong."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        graph = to_graph(comp)
        fourier_container = _get_node_attrs(graph.graph, self.fourier.id)["container_id"]
        self.rule.apply_single(graph, {"node_id": self.fourier.id, "container_id": fourier_container})
        assert to_diagram(graph) == comp

    def test_apply_single_identity_in_tensor_does_nothing(self):
        """apply_single on a path that is not under a CompositionDiagram returns original."""
        tensor = TensorDiagram([self.id_q, self.fourier])
        graph = to_graph(tensor)
        id_q_container = _get_node_attrs(graph.graph, self.id_q.id)["container_id"]
        self.rule.apply_single(graph, {"node_id": self.id_q.id, "container_id": id_q_container})
        assert to_diagram(graph) == tensor

    def test_apply_single_identity_in_contracted_does_nothing(self):
        """apply_single on a path that is directly in ContractedDiagram returns original."""
        contracted = ContractedDiagram(self.id_q, self.fourier, [0], [0], [], [])
        graph = to_graph(contracted)
        id_q_container = _get_node_attrs(graph.graph, self.id_q.id)["container_id"]
        self.rule.apply_single(graph, {"node_id": self.id_q.id, "container_id": id_q_container})
        assert to_diagram(graph) == contracted

    def test_apply_single_remove_identity(self):
        """Removing identity from a two element composition leaves the other."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, Fourier)
        assert result == self.fourier

    def test_apply_single_remove_identity_from_middle(self):
        """Remove identity that is not at index 0."""
        comp = CompositionDiagram([self.fourier, self.id_q, self.sq_gate])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.sq_gate
        assert result.connectivity == {0: {0: 0}}

    def test_apply_single_remove_identity_from_edge(self):
        """Remove identity that is not at index 0."""
        comp = CompositionDiagram([self.fourier, self.ph_rot, self.id_q])
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.ph_rot
        assert result.connectivity == {0: {0: 0}}

    def test_apply_single_nested_composition(self):
        """Remove identity from inner composition; flatten inner if needed."""
        inner = CompositionDiagram([self.id_q, self.fourier])
        outer = CompositionDiagram([inner, self.ph_rot])
        graph = to_graph(outer)
        matches = self.rule.match(graph)
        self.rule.apply_single(graph, matches[0])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.ph_rot
        assert result.connectivity == {0: {0: 0}}

    def test_apply_single_composition1(self):
        """Apply single to complex composition from self.comp1."""
        comp = self.comp1
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        matches_by_id = {m["node_id"]: m for m in matches}
        self.rule.apply_single(graph, matches_by_id[self.id_q.id])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 3
        assert result.diagrams[0] == self.non_id_p
        assert result.diagrams[1] == self.sq_gate
        assert result.diagrams[2] == self.id_p
        assert result.connectivity == {0: {0: 0}, 1: {0: 0}}

    def test_apply_single_composition2(self):
        """Remove second identity from comp1."""
        comp = self.comp1
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        matches_by_id = {m["node_id"]: m for m in matches}
        self.rule.apply_single(graph, matches_by_id[self.id_p.id])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 3
        assert result.diagrams[0] == self.non_id_p
        assert result.diagrams[1] == self.sq_gate
        assert result.diagrams[2] == self.id_q
        assert result.connectivity == {0: {0: 0}, 1: {0: 0}}

    def test_apply_single_complex_composition3(self):
        """Apply single to nested tensor structure."""
        comp = self.comp2
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        matches_by_id = {m["node_id"]: m for m in matches}
        idx = _sub(_sub(_sub(comp, 1), 1), 1).id
        self.rule.apply_single(graph, matches_by_id[idx])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.non_id_p_3x3
        tensor_new = result.diagrams[1]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 2
        assert tensor_new.diagrams[0] == self.beam_splitter1
        inner_comp_new = tensor_new.diagrams[1]
        assert isinstance(inner_comp_new, CompositionDiagram)
        assert len(inner_comp_new.diagrams) == 2
        assert inner_comp_new.diagrams[0] == self.fourier
        assert inner_comp_new.diagrams[1] == self.id_p
        assert inner_comp_new.connectivity == {0: {0: 0}}

    def test_apply_single_composition_multiple(self):
        """Apply single to nested tensor structure removing multiple identities."""
        comp = self.comp2
        graph = to_graph(comp)
        matches = self.rule.match(graph)
        matches_by_id = {m["node_id"]: m for m in matches}
        idx = _sub(_sub(_sub(comp, 1), 1), 1).id
        self.rule.apply_single(graph, matches_by_id[idx])
        idx = _sub(_sub(_sub(comp, 1), 1), 2).id
        self.rule.apply_single(graph, matches_by_id[idx])
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.non_id_p_3x3
        tensor_new = result.diagrams[1]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 2
        assert tensor_new.diagrams[0] == self.beam_splitter1
        assert tensor_new.diagrams[1] == self.fourier

    def test_apply_single_nested_composition_deep(self):
        """Apply single to deeply nested composition."""
        tensor = self.tensor1
        graph = to_graph(tensor)
        matches = self.rule.match(graph)
        matches_by_id = {m["node_id"]: m for m in matches}
        idx = _sub(_sub(tensor, 1), 2).id
        self.rule.apply_single(graph, matches_by_id[idx])
        result = to_diagram(graph)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.id_q
        comp1_new = result.diagrams[1]
        assert isinstance(comp1_new, CompositionDiagram)
        assert len(comp1_new.diagrams) == 3
        assert comp1_new.diagrams[0] == self.non_id_p
        assert comp1_new.diagrams[1] == self.sq_gate
        assert comp1_new.diagrams[2] == self.id_p
        assert comp1_new.connectivity == {0: {0: 0}, 1: {0: 0}}

    def test_apply_single_contracted_with_composition1(self):
        """Apply single to identity inside composition that is first child of ContractedDiagram."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(comp, self.swap, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        matches_by_id = {m["node_id"]: m for m in matches}
        idx = _sub(_sub(contracted, 0), 0).id
        self.rule.apply_single(graph, matches_by_id[idx])
        result = to_diagram(graph)
        assert isinstance(result, ContractedDiagram)
        assert result.first == self.fourier
        assert result.second == self.swap

    def test_apply_single_contracted_with_composition2(self):
        """Apply single to identity inside composition that is second child of ContractedDiagram."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(self.swap, comp, [0], [0], [], [])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        matches_by_id = {m["node_id"]: m for m in matches}
        idx = _sub(_sub(contracted, 1), 0).id
        self.rule.apply_single(graph, matches_by_id[idx])
        result = to_diagram(graph)
        assert isinstance(result, ContractedDiagram)
        assert result.first == self.swap
        assert result.second == self.fourier

    def test_apply_single_contracted_nested(self):
        """Apply single to deeply nested identity inside ContractedDiagram."""
        q_spider = QSpider(10, 10, self.phase_poly)
        contracted = ContractedDiagram(self.tensor2, q_spider, [1, 2, 3], [4, 5, 6], [0, 1, 2, 6], [1, 3, 5, 7])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        matches_by_id = {m["node_id"]: m for m in matches}
        idx = _sub(_sub(_sub(contracted, 0), 1), 2).id
        self.rule.apply_single(graph, matches_by_id[idx])
        result = to_diagram(graph)
        assert isinstance(result, ContractedDiagram)
        tensor_new = result.first
        assert isinstance(tensor_new, TensorDiagram)
        comp1_new = tensor_new.diagrams[1]
        assert isinstance(comp1_new, CompositionDiagram)
        assert len(comp1_new.diagrams) == 3
        assert comp1_new.diagrams[0] == self.non_id_p
        assert comp1_new.diagrams[1] == self.sq_gate
        assert comp1_new.diagrams[2] == self.id_p
        assert comp1_new.connectivity == {0: {0: 0}, 1: {0: 0}}

    # -------------------------------------------------------------------------
    # 3. Testing apply_rule() (full application)
    # -------------------------------------------------------------------------

    def test_apply_rule_removes_all_identities(self):
        """apply_rule should remove all matched identities."""
        comp1 = CompositionDiagram([self.id_q, self.fourier])
        comp2 = CompositionDiagram([self.id_p, self.disp])
        outer = CompositionDiagram([comp1, comp2])
        graph = to_graph(outer)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.disp

    def test_apply_rule_no_identity_returns_same_diagram(self):
        """If no identity present, apply_rule should return the original diagram."""
        comp = CompositionDiagram([self.fourier, self.ph_rot])
        graph = to_graph(comp)
        self.rule.apply_rule(graph)
        result = to_diagram(graph)
        assert result == comp

    def test_apply_rule_complex_comp1(self):
        """Apply full rule to self.comp1."""
        result = apply_rule_to_diagram(self.rule, self.comp1)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.non_id_p
        assert result.diagrams[1] == self.sq_gate

    def test_apply_rule_complex_comp2(self):
        """Apply full rule to self.comp2."""
        result = apply_rule_to_diagram(self.rule, self.comp2)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.non_id_p_3x3
        tensor_new = result.diagrams[1]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 2
        assert tensor_new.diagrams[0] == self.beam_splitter1
        assert tensor_new.diagrams[1] == self.fourier

    def test_apply_rule_complex_comp3(self):
        """Apply full rule to self.comp3."""
        result = apply_rule_to_diagram(self.rule, self.comp3)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 3
        assert result.diagrams[0] == self.non_id_q_4x4
        assert result.diagrams[1] == self.non_id_q_4x4
        tensor_new = result.diagrams[2]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 3
        assert tensor_new.diagrams[0] == self.beam_splitter1
        assert tensor_new.diagrams[1] == self.fourier_inv
        assert tensor_new.diagrams[2] == self.ph_rot

    def test_apply_rule_tensor1(self):
        """Apply full rule to self.tensor1."""
        result = apply_rule_to_diagram(self.rule, self.tensor1)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.id_q
        comp_new = result.diagrams[1]
        assert isinstance(comp_new, CompositionDiagram)
        assert len(comp_new.diagrams) == 2
        assert comp_new.diagrams[0] == self.non_id_p
        assert comp_new.diagrams[1] == self.sq_gate

    def test_apply_rule_tensor2(self):
        """Apply full rule to self.tensor2."""
        result = apply_rule_to_diagram(self.rule, self.tensor2)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 4
        assert result.diagrams[0] == self.id_q
        comp1_new = result.diagrams[1]
        assert isinstance(comp1_new, CompositionDiagram)
        assert len(comp1_new.diagrams) == 2
        assert comp1_new.diagrams[0] == self.non_id_p
        assert comp1_new.diagrams[1] == self.sq_gate
        assert result.diagrams[2] == self.non_id_q_3x3
        comp2_new = result.diagrams[3]
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
        result = apply_rule_to_diagram(self.rule, self.tensor3)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 6
        assert result.diagrams[0] == self.id_q
        comp1_new = result.diagrams[1]
        assert isinstance(comp1_new, CompositionDiagram)
        assert len(comp1_new.diagrams) == 2
        assert comp1_new.diagrams[0] == self.non_id_p
        assert comp1_new.diagrams[1] == self.sq_gate
        assert result.diagrams[2] == self.non_id_q_3x3
        comp2_new = result.diagrams[3]
        assert isinstance(comp2_new, CompositionDiagram)
        assert len(comp2_new.diagrams) == 2
        assert comp2_new.diagrams[0] == self.non_id_p_3x3
        tensor_new = comp2_new.diagrams[1]
        assert isinstance(tensor_new, TensorDiagram)
        assert len(tensor_new.diagrams) == 2
        assert tensor_new.diagrams[0] == self.beam_splitter1
        assert tensor_new.diagrams[1] == self.fourier
        assert result.diagrams[4] == self.non_id_p_3x2
        comp3_new = result.diagrams[5]
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
        """Apply full rule to ContractedDiagram with composition containing identity."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(comp, self.swap, [0], [0], [], [])
        result = apply_rule_to_diagram(self.rule, contracted)
        assert isinstance(result, ContractedDiagram)
        assert result.first == self.fourier
        assert result.second == self.swap

    def test_apply_rule_contracted_with_composition2(self):
        """Apply full rule to ContractedDiagram with composition containing identity as second."""
        comp = CompositionDiagram([self.id_q, self.fourier])
        contracted = ContractedDiagram(self.swap, comp, [0], [0], [], [])
        result = apply_rule_to_diagram(self.rule, contracted)
        assert isinstance(result, ContractedDiagram)
        assert result.first == self.swap
        assert result.second == self.fourier

    def test_apply_rule_contracted_nested(self):
        """Apply full rule to ContractedDiagram with deeply nested identities."""
        q_spider = QSpider(10, 10, self.phase_poly)
        contracted = ContractedDiagram(self.tensor2, q_spider, [1, 2, 3], [4, 5, 6], [0, 1, 2, 6], [1, 3, 5, 7])
        result = apply_rule_to_diagram(self.rule, contracted)
        assert isinstance(result, ContractedDiagram)
        tensor_new = result.first
        assert isinstance(tensor_new, TensorDiagram)
        comp1_new = tensor_new.diagrams[1]
        assert isinstance(comp1_new, CompositionDiagram)
        assert len(comp1_new.diagrams) == 2
        assert comp1_new.diagrams[0] == self.non_id_p
        assert comp1_new.diagrams[1] == self.sq_gate
        comp2_new = tensor_new.diagrams[3]
        assert isinstance(comp2_new, CompositionDiagram)
        assert len(comp2_new.diagrams) == 2
        assert comp2_new.diagrams[0] == self.non_id_p_3x3
        tensor_comp2 = comp2_new.diagrams[1]
        assert isinstance(tensor_comp2, TensorDiagram)
        assert len(tensor_comp2.diagrams) == 2
        assert tensor_comp2.diagrams[0] == self.beam_splitter1
        assert tensor_comp2.diagrams[1] == self.fourier

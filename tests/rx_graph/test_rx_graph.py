"""Tests for the CVZXGraph class and graph extraction utilities using rustworkx.

These tests verify that to_graph correctly converts CV ZX diagrams to rustworkx directed graphs
and that to_diagram correctly reconstructs the original diagrams.
"""

from math import pi
from typing import cast

import pytest
import rustworkx as rx
from sympy import Expr, I, symbols

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
    flatten_composition,
)
from cvzx.gates import (
    BeamsplitterGate,
    ControlledSumGate,
    ControlledZGate,
    DisplacementGate,
    PhaseRotationGate,
    SqueezingGate,
)
from cvzx.rx_graph import (
    CVZXGraph,
    GateRegister,
    get_connectivity,
    get_contracted_connections,
    get_immediate_container,
    get_nodes_by_container,
    get_root_node,
    to_diagram,
    to_graph,
)


def _make_diagram() -> Diagram:
    """Build a small representative diagram for the tests below."""
    zero = ZxPoly({})
    state = QSpider(0, 1, zero)
    sq = SqueezingGate(tau=0.5)
    effect = QSpider(1, 0, zero)
    return CompositionDiagram([state, sq, effect])


def _get_node_attrs(graph: rx.PyDiGraph, node_id: int) -> dict:
    """Helper to extract payload attrs for a specific diagram node ID."""
    for idx in graph.node_indices():
        if graph[idx].get("id") == node_id:
            return graph[idx]
    raise KeyError(f"Node ID {node_id} not found in PyDiGraph.")


def _has_node(graph: rx.PyDiGraph, node_id: int) -> bool:
    """Helper to check if a diagram node ID exists in the PyDiGraph."""
    return any(graph[idx].get("id") == node_id for idx in graph.node_indices())


def _has_edge(graph: rx.PyDiGraph, src_id: int, tgt_id: int) -> bool:
    """Helper to check if an edge exists between two diagram node IDs."""
    try:
        src_idx = next(idx for idx in graph.node_indices() if graph[idx].get("id") == src_id)
        tgt_idx = next(idx for idx in graph.node_indices() if graph[idx].get("id") == tgt_id)
        return graph.has_edge(src_idx, tgt_idx)
    except StopIteration:
        return False


def _get_edge_data(graph: rx.PyDiGraph, src_id: int, tgt_id: int) -> dict:
    """Helper to get edge payload between two diagram node IDs."""
    src_idx = next(idx for idx in graph.node_indices() if graph[idx].get("id") == src_id)
    tgt_idx = next(idx for idx in graph.node_indices() if graph[idx].get("id") == tgt_id)
    return graph.get_edge_data(src_idx, tgt_idx)


class TestCVZXGraphConstruction:
    """Test suite for CVZXGraph construction using rustworkx."""

    def test_from_diagram_round_trips(self):
        """from_diagram builds a graph whose to_diagram reproduces the input."""
        diagram = _make_diagram()
        cvzx_graph = CVZXGraph.from_diagram(diagram)
        assert cvzx_graph.to_diagram() == diagram

    def test_from_diagram_matches_to_graph(self):
        """from_diagram's underlying graph matches a plain to_graph call."""
        diagram = _make_diagram()
        cvzx_graph = CVZXGraph.from_diagram(diagram)
        expected_graph = to_graph(diagram)

        nodes1 = set(cvzx_graph.graph[i]["id"] for i in cvzx_graph.graph.node_indices())
        nodes2 = set(expected_graph.graph[i]["id"] for i in expected_graph.graph.node_indices())
        assert nodes1 == nodes2
        assert len(cvzx_graph.graph.edge_list()) == len(expected_graph.graph.edge_list())

    def test_auto_builds_registry_when_none_given(self):
        """Constructing without a registry builds one that matches the graph."""
        diagram = _make_diagram()
        graph = to_graph(diagram).graph
        cvzx_graph = CVZXGraph(graph)

        expected_registry = GateRegister()
        expected_registry.build_from_graph(graph)

        assert cvzx_graph.registry.squeezing_gates == expected_registry.squeezing_gates
        assert cvzx_graph.registry.input_states == expected_registry.input_states
        assert cvzx_graph.registry.measurement_nodes == expected_registry.measurement_nodes

    def test_uses_explicit_registry_when_given(self):
        """An explicitly passed registry is stored as-is, not rebuilt."""
        diagram = _make_diagram()
        graph = to_graph(diagram).graph
        registry = GateRegister()
        registry.build_from_graph(graph)

        cvzx_graph = CVZXGraph(graph, registry)
        assert cvzx_graph.registry is registry


class TestCVZXGraphToDiagram:
    """Test suite for CVZXGraph.to_diagram."""

    def test_to_diagram_matches_module_level_function(self):
        """to_diagram delegates to the module-level to_diagram function."""
        diagram = _make_diagram()
        cvzx_graph = CVZXGraph.from_diagram(diagram)
        assert cvzx_graph.to_diagram() == to_diagram(cvzx_graph)

    def test_to_diagram_raises_when_no_root(self):
        """to_diagram raises ValueError if the graph has no root node."""
        diagram = _make_diagram()
        graph = to_graph(diagram).graph
        for idx in graph.node_indices():
            graph[idx].pop("is_root", None)

        cvzx_graph = CVZXGraph(graph)
        with pytest.raises(ValueError, match="No root node"):
            cvzx_graph.to_diagram()


class TestCVZXGraphRebuildRegistry:
    """Test suite for CVZXGraph.rebuild_registry."""

    def test_rebuild_registry_picks_up_direct_graph_mutation(self):
        """rebuild_registry re-syncs the registry after a direct graph edit."""
        diagram = _make_diagram()
        cvzx_graph = CVZXGraph.from_diagram(diagram)

        squeezing_nodes_before = set(cvzx_graph.registry.squeezing_gates)
        assert squeezing_nodes_before

        node_id = next(iter(squeezing_nodes_before))
        for idx in cvzx_graph.graph.node_indices():
            if cvzx_graph.graph[idx].get("id") == node_id:
                cvzx_graph.graph[idx]["type"] = "Identity"

        assert node_id in cvzx_graph.registry.squeezing_gates

        cvzx_graph.rebuild_registry()

        assert node_id not in cvzx_graph.registry.squeezing_gates


class TestCVZXGraphCopy:
    """Test suite for CVZXGraph.copy."""

    def test_copy_graph_is_independent(self):
        """Mutating the copy's graph does not affect the original."""
        diagram = _make_diagram()
        original = CVZXGraph.from_diagram(diagram)
        duplicate = original.copy()

        duplicate.graph.add_node({"id": -1, "kind": "proper", "type": "Identity"})

        assert not _has_node(original.graph, -1)
        assert _has_node(duplicate.graph, -1)

    def test_copy_registry_is_independent(self):
        """Mutating the copy's registry does not affect the original."""
        diagram = _make_diagram()
        original = CVZXGraph.from_diagram(diagram)
        duplicate = original.copy()

        duplicate.registry.squeezing_gates.add(-1)

        assert -1 not in original.registry.squeezing_gates
        assert -1 in duplicate.registry.squeezing_gates

    def test_copy_reconstructs_an_equal_diagram(self):
        """A fresh copy still reconstructs the same diagram as the original."""
        diagram = _make_diagram()
        original = CVZXGraph.from_diagram(diagram)
        duplicate = original.copy()

        assert duplicate.to_diagram() == diagram
        assert duplicate == original


class TestCVZXGraphParameterConsistency:
    """Test suite for CVZXGraph.parameter_consistency_violations/validate_parameter_consistency.

    Mirrors `tests/nx_graph/test_nx_graph.py::TestCVZXGraphParameterConsistency`.
    """

    def test_clean_graph_has_no_violations(self):
        """A graph with no shared symbols and no stale measurement ids is consistent."""
        diagram = _make_diagram()
        cvzx_graph = CVZXGraph.from_diagram(diagram)

        assert cvzx_graph.parameter_consistency_violations() == []
        cvzx_graph.validate_parameter_consistency()  # must not raise

    def test_symbol_conflict_violation(self):
        """Two nodes sharing a symbol but disagreeing on its measurement binding conflict."""
        m = symbols("m")
        graph = rx.PyDiGraph()
        graph.add_node({
            "id": 1,
            "kind": "proper",
            "type": "QSpider",
            "num_inputs": 1,
            "num_outputs": 1,
            "phase": ZxPoly({1: m}),
            "feedforward": True,
            "measurement_ids": {100},
            "param_measurement_map": {m: {100}},
        })
        graph.add_node({
            "id": 2,
            "kind": "proper",
            "type": "QSpider",
            "num_inputs": 1,
            "num_outputs": 1,
            "phase": ZxPoly({1: m}),
            "feedforward": True,
            "measurement_ids": {200},
            "param_measurement_map": {m: {200}},
        })
        graph.add_node({"id": 100, "kind": "proper", "num_inputs": 1, "num_outputs": 0})
        graph.add_node({"id": 200, "kind": "proper", "num_inputs": 1, "num_outputs": 0})
        cvzx_graph = CVZXGraph(graph)

        violations = cvzx_graph.parameter_consistency_violations()

        assert len(violations) == 1
        assert "Symbol" in violations[0]
        assert "conflicting measurement sets" in violations[0]
        with pytest.raises(ValueError, match="conflicting measurement sets"):
            cvzx_graph.validate_parameter_consistency()

    def test_measurement_existence_violation(self):
        """A feedforward node referencing a measurement id with no such node is a violation."""
        graph = rx.PyDiGraph()
        graph.add_node({
            "id": 1,
            "kind": "compact",
            "type": "DisplacementGate",
            "num_inputs": 1,
            "num_outputs": 1,
            "feedforward": True,
            "measurement_ids": {999},
        })
        cvzx_graph = CVZXGraph(graph)

        violations = cvzx_graph.parameter_consistency_violations()

        assert len(violations) == 1
        assert "measurement id 999" in violations[0]
        with pytest.raises(ValueError, match="measurement id 999"):
            cvzx_graph.validate_parameter_consistency()

    def test_no_conflict_when_bindings_agree(self):
        """Two nodes sharing a symbol are fine if they agree on its measurement binding."""
        m = symbols("m")
        graph = rx.PyDiGraph()
        graph.add_node({
            "id": 1,
            "kind": "proper",
            "type": "QSpider",
            "num_inputs": 1,
            "num_outputs": 1,
            "phase": ZxPoly({1: m}),
            "feedforward": True,
            "measurement_ids": {100},
            "param_measurement_map": {m: {100}},
        })
        graph.add_node({
            "id": 2,
            "kind": "proper",
            "type": "QSpider",
            "num_inputs": 1,
            "num_outputs": 1,
            "phase": ZxPoly({1: m}),
            "feedforward": True,
            "measurement_ids": {100},
            "param_measurement_map": {m: {100}},
        })
        graph.add_node({"id": 100, "kind": "proper", "num_inputs": 1, "num_outputs": 0})
        cvzx_graph = CVZXGraph(graph)

        assert cvzx_graph.parameter_consistency_violations() == []


class TestCVZXGraphShapeProperties:
    """Test suite for CVZXGraph.root_id, num_inputs, and num_outputs."""

    def test_root_id_matches_get_root_node(self):
        """root_id agrees with the root node found via get_root_node."""
        diagram = _make_diagram()
        cvzx_graph = CVZXGraph.from_diagram(diagram)
        assert cvzx_graph.root_id == get_root_node(cvzx_graph)
        assert cvzx_graph.root_id is not None

    def test_root_id_is_none_when_no_root(self):
        """root_id is None when the graph has no root node."""
        diagram = _make_diagram()
        graph = to_graph(diagram).graph
        for idx in graph.node_indices():
            graph[idx].pop("is_root", None)

        cvzx_graph = CVZXGraph(graph)
        assert cvzx_graph.root_id is None

    def test_num_inputs_and_num_outputs_for_closed_diagram(self):
        """A fully-terminated diagram has zero external inputs and outputs."""
        diagram = _make_diagram()
        cvzx_graph = CVZXGraph.from_diagram(diagram)
        assert cvzx_graph.num_inputs == 0
        assert cvzx_graph.num_outputs == 0

    def test_num_inputs_and_num_outputs_for_open_diagram(self):
        """An open diagram reports the correct external input/output counts."""
        zero = ZxPoly({})
        sq = SqueezingGate(tau=0.5)
        open_diagram = TensorDiagram([sq, QSpider(0, 1, zero)])
        cvzx_graph = CVZXGraph.from_diagram(open_diagram)
        assert cvzx_graph.num_inputs == 1
        assert cvzx_graph.num_outputs == 2

    def test_num_inputs_raises_when_no_root(self):
        """num_inputs raises ValueError if the graph has no root node."""
        diagram = _make_diagram()
        graph = to_graph(diagram).graph
        for idx in graph.node_indices():
            graph[idx].pop("is_root", None)

        cvzx_graph = CVZXGraph(graph)
        with pytest.raises(ValueError, match="No root node"):
            _ = cvzx_graph.num_inputs


class TestCVZXGraphDunderMethods:
    """Test suite for CVZXGraph.__len__, __repr__, and __eq__."""

    def test_len_matches_node_count(self):
        """len() reports the number of nodes in the graph."""
        diagram = _make_diagram()
        cvzx_graph = CVZXGraph.from_diagram(diagram)
        assert len(cvzx_graph) == len(cvzx_graph.graph)
        assert len(cvzx_graph) > 0

    def test_repr_contains_node_and_edge_counts(self):
        """repr() reports node and edge counts."""
        diagram = _make_diagram()
        cvzx_graph = CVZXGraph.from_diagram(diagram)
        text = repr(cvzx_graph)
        assert "CVZXGraph" in text
        assert str(len(cvzx_graph.graph)) in text
        assert str(len(cvzx_graph.graph.edge_list())) in text

    def test_eq_true_for_equal_diagrams(self):
        """Two CVZXGraph instances built from equal diagrams compare equal."""
        diagram = _make_diagram()
        first = CVZXGraph.from_diagram(diagram)
        second = CVZXGraph.from_diagram(diagram)
        assert first == second

    def test_eq_false_for_different_diagrams(self):
        """Two CVZXGraph instances built from different diagrams compare unequal."""
        first = CVZXGraph.from_diagram(_make_diagram())
        zero = ZxPoly({})
        other_diagram = TensorDiagram([QSpider(0, 1, zero), QSpider(0, 1, zero)])
        second = CVZXGraph.from_diagram(other_diagram)
        assert first != second

    def test_eq_false_when_reconstruction_fails(self):
        """A graph that cannot reconstruct a diagram is never equal to another."""
        diagram = _make_diagram()
        graph = to_graph(diagram).graph
        for idx in graph.node_indices():
            graph[idx].pop("is_root", None)

        broken = CVZXGraph(graph)
        other = CVZXGraph.from_diagram(diagram)
        assert broken != other

    def test_eq_not_implemented_for_other_types(self):
        """__eq__ returns NotImplemented for objects that are not CVZXGraph."""
        diagram = _make_diagram()
        cvzx_graph = CVZXGraph.from_diagram(diagram)
        assert cvzx_graph.__eq__(object()) is NotImplemented
        assert cvzx_graph != object()


class TestGraphConversion:
    """Test suite for to_graph and to_diagram functions with rustworkx."""

    def setup_method(self):
        """Set up common objects for testing."""
        self.zero = ZxPoly({})
        self.phase_q = ZxPoly({2: 2.0})
        self.phase_p = ZxPoly({3: 3.0})

        self.q1 = QSpider(1, 1, self.phase_q)
        self.q2 = QSpider(1, 1, self.phase_p)
        self.q3 = QSpider(1, 1, self.zero)
        self.q4 = QSpider(1, 1, self.zero)

        self.p1 = PSpider(1, 1, self.phase_q)
        self.p2 = PSpider(1, 1, self.phase_p)

        self.fourier = Fourier()
        self.fourier_inv = FourierInv()
        self.fourier2 = Fourier2()
        self.swap = Swap()

        self.disp = DisplacementGate(alpha=1.0 + 0.5j)
        self.ph_rot = PhaseRotationGate(theta=pi / 4)
        self.sq_gate = SqueezingGate(tau=0.5)
        self.ctrl_sum_gate = ControlledSumGate(gain=1.0, control=1, target=2)
        self.ctrl_z_gate = ControlledZGate(gain=1.0)
        self.beam_splitter = BeamsplitterGate(theta=pi / 4)

        a, b, c, g1, g2, m, tau, theta = symbols("a b c g1 g2 m tau theta", real=True)
        self.q_param = QSpider(1, 1, ZxPoly({1: a, 2: b}), True)
        self.p_param = PSpider(1, 1, ZxPoly({1: a, 2: b}), True)
        self.disp_param = DisplacementGate(alpha=1.0 + c * I, parametric=True)
        self.meas = QSpider(1, 0, ZxPoly({1: m}), True)
        self.disp_ff = DisplacementGate(alpha=m**2, parametric=True, feedforward=True, measurement_ids={self.meas.id})
        self.ph_rot_param = PhaseRotationGate(theta, parametric=True)
        self.sq_gate_param = SqueezingGate(tau, parametric=True)
        self.ctrl_sum_gate_param = ControlledSumGate(gain=g1, control=1, target=2, parametric=True)
        self.ctrl_z_gate_param = ControlledZGate(gain=g2, parametric=True)
        self.beam_splitter_param = BeamsplitterGate(theta / 2, parametric=True)

        self.q_1x2 = QSpider(1, 2, self.phase_q)
        self.q_2x1 = QSpider(2, 1, self.phase_p)
        self.p_2x1 = PSpider(2, 1, self.phase_q)

        self.conn_swap = {0: 0, 1: 1}
        self.conn_cross = {0: 1, 1: 0}

    def _assert_diagram_equality(self, original: Diagram, reconstructed: Diagram) -> None:
        """Helper to assert that two diagrams are equal."""
        if isinstance(original, (QSpider, PSpider, Swap, Fourier, FourierInv, Fourier2)):
            assert original == reconstructed
            return

        if isinstance(original, CompositionDiagram):
            assert isinstance(reconstructed, CompositionDiagram)
            assert len(original.diagrams) == len(reconstructed.diagrams)
            assert original.connectivity == reconstructed.connectivity
            for orig_sub, recon_sub in zip(original.diagrams, reconstructed.diagrams, strict=False):
                self._assert_diagram_equality(orig_sub, recon_sub)
            return

        if isinstance(original, TensorDiagram):
            assert isinstance(reconstructed, TensorDiagram)
            assert len(original.diagrams) == len(reconstructed.diagrams)
            for orig_sub, recon_sub in zip(original.diagrams, reconstructed.diagrams, strict=False):
                self._assert_diagram_equality(orig_sub, recon_sub)
            return

        if isinstance(original, ContractedDiagram):
            assert isinstance(reconstructed, ContractedDiagram)
            assert original.I1 == reconstructed.I1
            assert original.I2 == reconstructed.I2
            assert original.J1 == reconstructed.J1
            assert original.J2 == reconstructed.J2
            self._assert_diagram_equality(original.first, reconstructed.first)
            self._assert_diagram_equality(original.second, reconstructed.second)
            return

    def _assert_roundtrip(self, diagram: Diagram) -> None:
        """Test roundtrip conversion: diagram → graph → diagram."""
        cvzx_graph = to_graph(diagram)
        reconstructed = to_diagram(cvzx_graph)
        self._assert_diagram_equality(diagram, reconstructed)

    def test_proper_qspider(self):
        """Test conversion of a single QSpider."""
        cvzx_graph = to_graph(self.q1)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        assert len(graph.edge_list()) == 0

        attrs = _get_node_attrs(graph, self.q1.id)
        assert attrs["id"] == self.q1.id
        assert attrs["type"] == "QSpider"
        assert attrs["kind"] == "proper"
        assert attrs["phase"] == self.phase_q
        assert attrs["num_inputs"] == 1
        assert attrs["num_outputs"] == 1
        assert attrs["container_id"] is None
        assert attrs["is_root"]

        self._assert_roundtrip(self.q1)

    def test_proper_qspider_param(self):
        """Test conversion of a parametrized QSpider."""
        cvzx_graph = to_graph(self.q_param)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        assert len(graph.edge_list()) == 0

        attrs = _get_node_attrs(graph, self.q_param.id)
        assert attrs["id"] == self.q_param.id
        assert attrs["type"] == "QSpider"
        assert attrs["kind"] == "proper"
        assert attrs["phase"] == self.q_param.phase
        assert attrs["num_inputs"] == 1
        assert attrs["num_outputs"] == 1
        assert attrs["container_id"] is None
        assert attrs["is_root"]

        self._assert_roundtrip(self.q_param)

    def test_proper_pspider(self):
        """Test conversion of a single PSpider."""
        cvzx_graph = to_graph(self.p1)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.p1.id)
        assert attrs["type"] == "PSpider"
        assert attrs["phase"] == self.phase_q

        self._assert_roundtrip(self.p1)

    def test_proper_pspider_param(self):
        """Test conversion of a parametrized PSpider."""
        cvzx_graph = to_graph(self.p_param)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        assert len(graph.edge_list()) == 0

        attrs = _get_node_attrs(graph, self.p_param.id)
        assert attrs["id"] == self.p_param.id
        assert attrs["type"] == "PSpider"
        assert attrs["kind"] == "proper"
        assert attrs["phase"] == self.p_param.phase
        assert attrs["num_inputs"] == 1
        assert attrs["num_outputs"] == 1
        assert attrs["container_id"] is None
        assert attrs["is_root"]

        self._assert_roundtrip(self.p_param)

    def test_proper_fourier(self):
        """Test conversion of a single Fourier gate."""
        cvzx_graph = to_graph(self.fourier)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.fourier.id)
        assert attrs["type"] == "Fourier"
        assert attrs["phase"] is None
        assert attrs["num_inputs"] == 1
        assert attrs["num_outputs"] == 1

        self._assert_roundtrip(self.fourier)

    def test_proper_swap(self):
        """Test conversion of a single Swap gate."""
        cvzx_graph = to_graph(self.swap)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.swap.id)
        assert attrs["type"] == "Swap"
        assert attrs["num_inputs"] == 2
        assert attrs["num_outputs"] == 2

        self._assert_roundtrip(self.swap)

    def test_proper_multi_arity_spider(self):
        """Test conversion of a spider with multiple inputs/outputs."""
        cvzx_graph = to_graph(self.q_1x2)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.q_1x2.id)
        assert attrs["num_inputs"] == 1
        assert attrs["num_outputs"] == 2
        assert attrs["external_inputs"] == [0]
        assert attrs["external_outputs"] == [0, 1]

        self._assert_roundtrip(self.q_1x2)

    def test_displacement(self):
        """Test conversion of a DisplacementGate."""
        cvzx_graph = to_graph(self.disp)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.disp.id)
        assert attrs["type"] == "DisplacementGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.disp.alpha

        self._assert_roundtrip(self.disp)

    def test_displacement_param(self):
        """Test conversion of a parametrized DisplacementGate."""
        cvzx_graph = to_graph(self.disp_param)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.disp_param.id)
        assert attrs["type"] == "DisplacementGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.disp_param.alpha
        assert isinstance(attrs["phase"], Expr)
        assert attrs["feedforward"] == self.disp_param.feedforward
        assert attrs["measurement_ids"] == self.disp_param.measurement_ids

        self._assert_roundtrip(self.disp_param)

    def test_feedforward(self):
        """Test conversion of a feedforward DisplacementGate."""
        cvzx_graph = to_graph(self.disp_ff)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.disp_ff.id)
        assert attrs["type"] == "DisplacementGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.disp_ff.alpha
        assert isinstance(attrs["phase"], Expr)
        assert attrs["feedforward"] == self.disp_ff.feedforward
        assert attrs["measurement_ids"] == self.disp_ff.measurement_ids

        self._assert_roundtrip(self.disp_ff)

    def test_rotation(self):
        """Test conversion of a PhaseRotationGate."""
        cvzx_graph = to_graph(self.ph_rot)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.ph_rot.id)
        assert attrs["type"] == "PhaseRotationGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.ph_rot.theta

        self._assert_roundtrip(self.ph_rot)

    def test_rotation_param(self):
        """Test conversion of a parametrized PhaseRotationGate."""
        cvzx_graph = to_graph(self.ph_rot_param)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.ph_rot_param.id)
        assert attrs["type"] == "PhaseRotationGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.ph_rot_param.theta
        assert isinstance(attrs["phase"], Expr)

        self._assert_roundtrip(self.ph_rot_param)

    def test_squeezing(self):
        """Test conversion of a SqueezingGate."""
        cvzx_graph = to_graph(self.sq_gate)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.sq_gate.id)
        assert attrs["type"] == "SqueezingGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.sq_gate.tau

        self._assert_roundtrip(self.sq_gate)

    def test_squeezing_param(self):
        """Test conversion of a parametrized SqueezingGate."""
        cvzx_graph = to_graph(self.sq_gate_param)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.sq_gate_param.id)
        assert attrs["type"] == "SqueezingGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.sq_gate_param.tau
        assert isinstance(attrs["phase"], Expr)

        self._assert_roundtrip(self.sq_gate_param)

    def test_beamsplitter(self):
        """Test conversion of a BeamsplitterGate."""
        cvzx_graph = to_graph(self.beam_splitter)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.beam_splitter.id)
        assert attrs["type"] == "BeamsplitterGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.beam_splitter.theta

        self._assert_roundtrip(self.beam_splitter)

    def test_beamsplitter_param(self):
        """Test conversion of a parametrized BeamsplitterGate."""
        cvzx_graph = to_graph(self.beam_splitter_param)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.beam_splitter_param.id)
        assert attrs["type"] == "BeamsplitterGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.beam_splitter_param.theta
        assert isinstance(attrs["phase"], Expr)

        self._assert_roundtrip(self.beam_splitter_param)

    def test_controlledsumgate(self):
        """Test conversion of a ControlledSumGate."""
        cvzx_graph = to_graph(self.ctrl_sum_gate)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.ctrl_sum_gate.id)
        assert attrs["type"] == "ControlledSumGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.ctrl_sum_gate.gain

        self._assert_roundtrip(self.ctrl_sum_gate)

    def test_controlledsumgate_param(self):
        """Test conversion of a parametrized ControlledSumGate."""
        cvzx_graph = to_graph(self.ctrl_sum_gate_param)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.ctrl_sum_gate_param.id)
        assert attrs["type"] == "ControlledSumGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.ctrl_sum_gate_param.gain
        assert isinstance(attrs["phase"], Expr)

        self._assert_roundtrip(self.ctrl_sum_gate_param)

    def test_controlledzgate(self):
        """Test conversion of a ControlledZGate."""
        cvzx_graph = to_graph(self.ctrl_z_gate)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.ctrl_z_gate.id)
        assert attrs["type"] == "ControlledZGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.ctrl_z_gate.gain

        self._assert_roundtrip(self.ctrl_z_gate)

    def test_controlledzgate_param(self):
        """Test conversion of a parametrized ControlledZGate."""
        cvzx_graph = to_graph(self.ctrl_z_gate_param)
        graph = cvzx_graph.graph

        assert len(graph) == 1
        attrs = _get_node_attrs(graph, self.ctrl_z_gate_param.id)
        assert attrs["type"] == "ControlledZGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.ctrl_z_gate_param.gain
        assert isinstance(attrs["phase"], Expr)

        self._assert_roundtrip(self.ctrl_z_gate_param)

    def test_composition_two_proper_diagrams(self):
        """Test conversion of CompositionDiagram with two proper diagrams."""
        comp = CompositionDiagram([self.q1, self.q2])
        cvzx_graph = to_graph(comp)
        graph = cvzx_graph.graph

        assert len(graph) == 3
        assert _has_node(graph, self.q1.id)
        assert _has_node(graph, self.q2.id)
        assert _has_node(graph, comp.id)

        assert len(graph.edge_list()) == 1
        edge_data = _get_edge_data(graph, self.q1.id, self.q2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]
        assert edge_data["edge_type"] == "composition"
        assert not edge_data["internal"]
        assert edge_data["left_idx"] == 0
        assert edge_data["right_idx"] == 1

        attrs = _get_node_attrs(graph, comp.id)
        assert attrs["kind"] == "container"
        assert attrs["container_type"] == "composition"
        assert attrs["sub_diagram_ids"] == [self.q1.id, self.q2.id]
        assert attrs["connectivity"] == comp.connectivity
        assert attrs["is_root"]

        assert _get_node_attrs(graph, self.q1.id)["container_id"] == comp.id
        assert _get_node_attrs(graph, self.q2.id)["container_id"] == comp.id

        self._assert_roundtrip(comp)

    def test_composition_three_proper_diagrams(self):
        """Test conversion of CompositionDiagram with three proper diagrams."""
        comp = CompositionDiagram([self.q1, self.q2, self.q3])
        cvzx_graph = to_graph(comp)
        graph = cvzx_graph.graph

        assert len(graph) == 4
        assert len(graph.edge_list()) == 2

        edge_data = _get_edge_data(graph, self.q1.id, self.q2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        edge_data = _get_edge_data(graph, self.q2.id, self.q3.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        self._assert_roundtrip(comp)

    def test_composition_proper_with_connectivity(self):
        """Test CompositionDiagram with non-trivial connectivity."""
        conn = {0: 1, 1: 0}
        comp = CompositionDiagram([self.q_1x2, self.q_2x1], {0: conn})
        cvzx_graph = to_graph(comp)
        graph = cvzx_graph.graph

        assert len(graph) == 3
        assert len(graph.edge_list()) == 1

        edge_data = _get_edge_data(graph, self.q_1x2.id, self.q_2x1.id)
        for i, key in enumerate(edge_data["source_ports"]):
            assert conn[key] == edge_data["target_ports"][i]

        self._assert_roundtrip(comp)

    def test_composition_mixed_proper_types(self):
        """Test composition of different proper diagram types."""
        comp = CompositionDiagram([self.q1, self.fourier, self.q2])
        cvzx_graph = to_graph(comp)
        graph = cvzx_graph.graph

        assert len(graph) == 4
        assert len(graph.edge_list()) == 2
        assert _has_edge(graph, self.q1.id, self.fourier.id)
        assert _has_edge(graph, self.fourier.id, self.q2.id)

        self._assert_roundtrip(comp)

    def test_tensor_proper_diagrams(self):
        """Test conversion of TensorDiagram of proper diagrams."""
        tensor = TensorDiagram([self.q1, self.q2, self.q3])
        cvzx_graph = to_graph(tensor)
        graph = cvzx_graph.graph

        assert len(graph) == 4
        assert _has_node(graph, self.q1.id)
        assert _has_node(graph, self.q2.id)
        assert _has_node(graph, self.q3.id)
        assert _has_node(graph, tensor.id)

        assert len(graph.edge_list()) == 0

        attrs = _get_node_attrs(graph, tensor.id)
        assert attrs["kind"] == "container"
        assert attrs["container_type"] == "tensor"
        assert attrs["sub_diagram_ids"] == [self.q1.id, self.q2.id, self.q3.id]
        assert attrs["is_root"] is True

        assert attrs["external_input_mapping"] == {0: (0, 0), 1: (1, 0), 2: (2, 0)}
        assert attrs["external_output_mapping"] == {0: (0, 0), 1: (1, 0), 2: (2, 0)}

        self._assert_roundtrip(tensor)

    def test_tensor_proper_diagrams_with_arity(self):
        """Test TensorDiagram with different arities."""
        tensor = TensorDiagram([self.q_1x2, self.q_2x1])
        cvzx_graph = to_graph(tensor)
        graph = cvzx_graph.graph

        assert len(graph) == 3
        attrs = _get_node_attrs(graph, tensor.id)
        assert attrs["external_input_mapping"] == {0: (0, 0), 1: (1, 0), 2: (1, 1)}
        assert attrs["external_output_mapping"] == {0: (0, 0), 1: (0, 1), 2: (1, 0)}

        self._assert_roundtrip(tensor)

    def test_composition_of_tensors1(self):
        """Test conversion of CompositionDiagram containing TensorDiagrams."""
        tensor1 = TensorDiagram([self.q1, self.q2])
        tensor2 = TensorDiagram([self.q3, self.q4])
        comp = CompositionDiagram([tensor1, tensor2])
        cvzx_graph = to_graph(comp)
        graph = cvzx_graph.graph

        assert len(graph) == 7

        comp_attrs = _get_node_attrs(graph, comp.id)
        assert comp_attrs["sub_diagram_ids"] == [tensor1.id, tensor2.id]

        tensor1_attrs = _get_node_attrs(graph, tensor1.id)
        assert tensor1_attrs["container_id"] == comp.id
        assert tensor1_attrs["sub_diagram_ids"] == [self.q1.id, self.q2.id]

        tensor2_attrs = _get_node_attrs(graph, tensor2.id)
        assert tensor2_attrs["container_id"] == comp.id
        assert tensor2_attrs["sub_diagram_ids"] == [self.q3.id, self.q4.id]

        assert len(graph.edge_list()) == 2
        edge_data = _get_edge_data(graph, self.q1.id, self.q3.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]
        edge_data = _get_edge_data(graph, self.q2.id, self.q4.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        self._assert_roundtrip(comp)

    def test_composition_of_tensors2(self):
        """Test conversion of CompositionDiagram containing TensorDiagrams."""
        two_fouriers_tensor = self.fourier.tensor(self.fourier_inv)
        swap_conn = two_fouriers_tensor.compose(self.swap, connectivity={0: 1, 1: 0})

        cvzx_graph = to_graph(swap_conn)
        graph = cvzx_graph.graph

        assert len(graph) == 5

        tensor_node = None
        for idx in graph.node_indices():
            if graph[idx].get("container_type") == "tensor":
                tensor_node = graph[idx]["id"]
                break
        assert tensor_node is not None

        comp_node = None
        for idx in graph.node_indices():
            if graph[idx].get("container_type") == "composition":
                comp_node = graph[idx]["id"]
                break
        assert comp_node is not None

        comp_attrs = _get_node_attrs(graph, comp_node)
        assert comp_attrs["is_root"] is True
        assert comp_attrs["sub_diagram_ids"] == [self.swap.id, tensor_node]

        tensor_attrs = _get_node_attrs(graph, tensor_node)
        assert tensor_attrs["container_id"] == comp_node
        assert tensor_attrs["sub_diagram_ids"] == [self.fourier.id, self.fourier_inv.id]

        for f_node in [self.fourier.id, self.fourier_inv.id]:
            assert _get_node_attrs(graph, f_node)["container_id"] == tensor_node

        assert _get_node_attrs(graph, self.swap.id)["container_id"] == comp_node

        fourier_nodes = [
            graph[idx]["id"]
            for idx in graph.node_indices()
            if "Fourier" in str(graph[idx].get("type")) and graph[idx].get("kind") == "proper"
        ]
        assert len(fourier_nodes) == 2

        assert len(graph.edge_list()) == 2

        edge_data = _get_edge_data(graph, self.swap.id, fourier_nodes[0])
        assert edge_data["source_ports"] == [1]
        assert edge_data["target_ports"] == [0]

        edge_data = _get_edge_data(graph, self.swap.id, fourier_nodes[1])
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        self._assert_roundtrip(swap_conn)

    def test_tensor_of_compositions(self):
        """Test conversion of TensorDiagram containing CompositionDiagrams."""
        comp1: Diagram = CompositionDiagram([self.q1, self.q2])
        comp2: Diagram = CompositionDiagram([self.q3, self.q4])
        tensor = TensorDiagram([comp1, comp2])
        comp1 = tensor.diagrams[0]
        comp2 = tensor.diagrams[1]
        cvzx_graph = to_graph(tensor)
        graph = cvzx_graph.graph

        assert len(graph) == 7

        tensor_attrs = _get_node_attrs(graph, tensor.id)
        assert tensor_attrs["sub_diagram_ids"] == [comp1.id, comp2.id]

        comp1_attrs = _get_node_attrs(graph, comp1.id)
        assert comp1_attrs["container_id"] == tensor.id
        assert comp1_attrs["sub_diagram_ids"] == [self.q1.id, self.q2.id]

        comp2_attrs = _get_node_attrs(graph, comp2.id)
        assert comp2_attrs["container_id"] == tensor.id
        assert comp2_attrs["sub_diagram_ids"] == [self.q3.id, self.q4.id]

        assert len(graph.edge_list()) == 2

        self._assert_roundtrip(tensor)

    def test_contracted_proper_diagrams(self):
        """Test conversion of ContractedDiagram with proper diagrams."""
        contracted = ContractedDiagram(self.q_1x2, self.p_2x1, [0], [0], [0], [0])
        cvzx_graph = to_graph(contracted)
        graph = cvzx_graph.graph

        assert len(graph) == 3

        attrs = _get_node_attrs(graph, contracted.id)
        assert attrs["kind"] == "container"
        assert attrs["container_type"] == "contracted"
        assert attrs["first_id"] == self.q_1x2.id
        assert attrs["second_id"] == self.p_2x1.id
        assert attrs["I1"] == [0]
        assert attrs["I2"] == [0]
        assert attrs["J1"] == [0]
        assert attrs["J2"] == [0]

        assert _has_edge(graph, self.q_1x2.id, self.p_2x1.id)
        edge_data = _get_edge_data(graph, self.q_1x2.id, self.p_2x1.id)
        assert edge_data["edge_type"] == "contracted_internal"
        assert edge_data["internal"] is True
        assert edge_data["connection_type"] == "I1_I2"
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        assert _has_edge(graph, self.p_2x1.id, self.q_1x2.id)
        edge_data = _get_edge_data(graph, self.p_2x1.id, self.q_1x2.id)
        assert edge_data["connection_type"] == "J2_J1"

        assert attrs["external_input_mapping"][0] == ("second", 1)
        assert attrs["external_output_mapping"][0] == ("first", 1)

        self._assert_roundtrip(contracted)

    def test_composition_of_contracted_diagrams(self):
        """Test conversion of CompositionDiagram containing ContractedDiagrams."""
        contracted1 = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        contracted2 = ContractedDiagram(self.q3, self.q4, [0], [0], [], [])
        comp = CompositionDiagram([contracted1, contracted2])
        cvzx_graph = to_graph(comp)
        graph = cvzx_graph.graph

        assert len(graph) == 7
        assert _has_edge(graph, self.q2.id, self.q3.id)

        comp_attrs = _get_node_attrs(graph, comp.id)
        assert comp_attrs["sub_diagram_ids"] == [contracted1.id, contracted2.id]

        contracted1_attrs = _get_node_attrs(graph, contracted1.id)
        assert contracted1_attrs["container_id"] == comp.id
        assert contracted1_attrs["first_id"] == self.q1.id
        assert contracted1_attrs["second_id"] == self.q2.id

        contracted2_attrs = _get_node_attrs(graph, contracted2.id)
        assert contracted2_attrs["container_id"] == comp.id
        assert contracted2_attrs["first_id"] == self.q3.id
        assert contracted2_attrs["second_id"] == self.q4.id

        self._assert_roundtrip(comp)

    def test_tensor_of_contracted_diagrams(self):
        """Test conversion of TensorDiagram containing ContractedDiagrams."""
        contracted1 = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        contracted2 = ContractedDiagram(self.q3, self.q4, [0], [0], [], [])
        tensor = TensorDiagram([contracted1, contracted2])
        cvzx_graph = to_graph(tensor)
        graph = cvzx_graph.graph

        assert len(graph) == 7
        assert len(graph.edge_list()) == 2

        self._assert_roundtrip(tensor)

    def test_complex_nested_structure1(self):
        """Test conversion of complex nested structures."""
        comp: Diagram = CompositionDiagram([self.q1, self.q2])
        contracted: Diagram = ContractedDiagram(self.q3, self.q4, [0], [0], [], [])
        tensor = TensorDiagram([comp, contracted])
        comp = tensor.diagrams[0]
        contracted = tensor.diagrams[1]
        cvzx_graph = to_graph(tensor)
        graph = cvzx_graph.graph

        assert len(graph) == 7

        tensor_attrs = _get_node_attrs(graph, tensor.id)
        assert tensor_attrs["sub_diagram_ids"] == [comp.id, contracted.id]

        comp_attrs = _get_node_attrs(graph, comp.id)
        assert comp_attrs["container_id"] == tensor.id
        assert comp_attrs["sub_diagram_ids"] == [self.q1.id, self.q2.id]

        contracted_attrs = _get_node_attrs(graph, contracted.id)
        assert contracted_attrs["container_id"] == tensor.id

        assert _has_edge(graph, self.q1.id, self.q2.id)
        edge_data = _get_edge_data(graph, self.q1.id, self.q2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        assert _has_edge(graph, self.q3.id, self.q4.id)
        edge_data = _get_edge_data(graph, self.q3.id, self.q4.id)
        assert edge_data["edge_type"] == "contracted_internal"
        assert edge_data["internal"] is True
        assert edge_data["connection_type"] == "I1_I2"

        self._assert_roundtrip(tensor)

    def test_complex_nested_structure2(self):
        """Test conversion of complex nested structures."""
        fourier = self.fourier
        fourier2 = Fourier()
        fourier_outer = Fourier()  # <--- Add distinct Fourier instance for outer tensor
        fourier_squared = self.fourier2
        fourier_squared2 = Fourier2()
        fourier_inv = self.fourier_inv
        fourier_inv2 = FourierInv()
        swap_node1 = Swap()
        swap_node2 = Swap()
        swap_node3 = Swap()
        q_spider1 = QSpider(3, 3, self.phase_q)
        q_spider2 = QSpider(3, 3, cast("ZxPoly", self.phase_q * 4))
        q_spider3 = QSpider(3, 3, cast("ZxPoly", self.phase_q * 10))
        q_spider_5x5 = QSpider(5, 5, self.phase_q)
        p_spider_4x4 = PSpider(4, 4, self.phase_p)

        contracted_diagram_1 = ContractedDiagram(
            q_spider_5x5, p_spider_4x4, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 3]
        )

        large_tensor_conn = fourier.tensor(fourier_squared)
        large_tensor_conn = large_tensor_conn.tensor(fourier_inv)
        tensor1 = swap_node1.tensor(fourier2)
        tensor2 = swap_node2.tensor(fourier_squared2)
        tensor3 = swap_node3.tensor(fourier_inv2)
        large_comp_conn = large_tensor_conn.compose(tensor1, connectivity={0: 2, 1: 0, 2: 1})
        large_comp_conn = tensor2.compose(large_comp_conn, connectivity={0: 1, 1: 0, 2: 2})
        large_comp_conn = tensor3.compose(large_comp_conn)
        nested_block_conn: Diagram = CompositionDiagram([q_spider1, q_spider2])
        nested_block_conn = nested_block_conn.compose(q_spider3)
        nested_block_conn = nested_block_conn.compose(contracted_diagram_1)

        large_comp_conn = large_comp_conn.compose(nested_block_conn, connectivity={0: 1, 1: 2, 2: 0})
        large_comp_conn = flatten_composition(large_comp_conn)

        # Use fourier_outer instead of fourier here
        final_large_conn = fourier_outer.tensor(large_comp_conn)
        assert isinstance(final_large_conn, TensorDiagram)
        large_comp_conn = final_large_conn.diagrams[1]

        cvzx_graph = to_graph(final_large_conn)
        graph = cvzx_graph.graph

        # Count nodes:
        assert len(graph) == 22

        # Check that the root is the outer tensor
        root = get_root_node(cvzx_graph)
        assert root is not None
        root_attrs = _get_node_attrs(graph, root)
        assert root_attrs["type"] == "TensorDiagram"
        assert root_attrs["is_root"] is True
        assert root_attrs["container_type"] == "tensor"
        # Update assertion to check fourier_outer.id
        assert root_attrs["sub_diagram_ids"] == [fourier_outer.id, large_comp_conn.id]

        comp_node = None
        for idx in graph.node_indices():
            if graph[idx].get("id") == large_comp_conn.id:
                comp_node = graph[idx]["id"]
                break
        assert comp_node is not None

        comp_attrs = _get_node_attrs(graph, comp_node)
        assert comp_attrs["container_type"] == "composition"
        assert len(comp_attrs["sub_diagram_ids"]) == 8
        assert comp_attrs["container_id"] == root

        connectivity = get_connectivity(cvzx_graph, comp_node)
        assert connectivity is not None
        assert isinstance(large_comp_conn, CompositionDiagram)
        assert connectivity == large_comp_conn.connectivity
        assert len(_get_node_attrs(graph, comp_node)["sub_diagram_ids"]) == 8

        tensor_nodes = [
            graph[idx]["id"]
            for idx in graph.node_indices()
            if graph[idx].get("container_type") == "tensor" and graph[idx].get("id") != root
        ]
        assert len(tensor_nodes) == 4

        large_tensor_node = None
        for n in tensor_nodes:
            if _get_node_attrs(graph, n).get("sub_diagram_ids") == [fourier.id, fourier_squared.id, fourier_inv.id]:
                large_tensor_node = n
                break
        assert large_tensor_node is not None
        assert _get_node_attrs(graph, large_tensor_node)["container_id"] == comp_node

        swap_tensor_nodes = [n for n in tensor_nodes if len(_get_node_attrs(graph, n).get("sub_diagram_ids")) == 2]
        assert len(swap_tensor_nodes) == 3
        for n in swap_tensor_nodes:
            assert _get_node_attrs(graph, n)["container_id"] == comp_node

        contracted_node = None
        for idx in graph.node_indices():
            if graph[idx].get("container_type") == "contracted":
                contracted_node = graph[idx]["id"]
                break
        assert contracted_node is not None
        assert _get_node_attrs(graph, contracted_node)["container_id"] == comp_node
        assert _get_node_attrs(graph, contracted_node)["first_id"] == q_spider_5x5.id
        assert _get_node_attrs(graph, contracted_node)["second_id"] == p_spider_4x4.id

        contracted_connections = get_contracted_connections(cvzx_graph, contracted_node)
        assert contracted_connections is not None
        assert contracted_connections["I1"] == [0, 1, 4]
        assert contracted_connections["I2"] == [1, 2, 3]
        assert contracted_connections["J1"] == [0, 1, 2]
        assert contracted_connections["J2"] == [1, 2, 3]

        contracted_attrs = _get_node_attrs(graph, contracted_node)
        assert contracted_attrs["external_input_mapping"] == {
            0: ("first", 3),
            1: ("first", 4),
            2: ("second", 0),
        }
        assert contracted_attrs["external_output_mapping"] == {
            0: ("first", 2),
            1: ("first", 3),
            2: ("second", 0),
        }

        assert _has_edge(graph, q_spider2.id, swap_node1.id)
        edge_data = _get_edge_data(graph, q_spider2.id, swap_node1.id)
        assert edge_data["source_ports"] == [0, 2]
        assert edge_data["target_ports"] == [1, 0]

        assert _has_edge(graph, q_spider2.id, fourier2.id)
        edge_data = _get_edge_data(graph, q_spider2.id, fourier2.id)
        assert edge_data["source_ports"] == [1]
        assert edge_data["target_ports"] == [0]

        assert _has_edge(graph, swap_node1.id, fourier.id)
        edge_data = _get_edge_data(graph, swap_node1.id, fourier.id)
        assert edge_data["source_ports"] == [1]
        assert edge_data["target_ports"] == [0]

        assert _has_edge(graph, swap_node1.id, fourier_inv.id)
        edge_data = _get_edge_data(graph, swap_node1.id, fourier_inv.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        assert _has_edge(graph, fourier2.id, fourier_squared.id)
        edge_data = _get_edge_data(graph, fourier2.id, fourier_squared.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        assert _has_edge(graph, fourier_squared.id, swap_node2.id)
        edge_data = _get_edge_data(graph, fourier_squared.id, swap_node2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        assert _has_edge(graph, fourier.id, swap_node2.id)
        edge_data = _get_edge_data(graph, fourier.id, swap_node2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [1]

        assert _has_edge(graph, swap_node2.id, swap_node3.id)
        edge_data = _get_edge_data(graph, swap_node2.id, swap_node3.id)
        assert edge_data["source_ports"] == [0, 1]
        assert edge_data["target_ports"] == [0, 1]

        assert _has_edge(graph, fourier_squared2.id, fourier_inv2.id)
        edge_data = _get_edge_data(graph, fourier_squared2.id, fourier_inv2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        self._assert_roundtrip(final_large_conn)

    def test_root_node_identification(self):
        """Test that the root node is correctly identified."""
        comp = CompositionDiagram([self.q1, self.q2])
        cvzx_graph = to_graph(comp)
        assert get_root_node(cvzx_graph) == comp.id

        tensor = TensorDiagram([self.q1, self.q2])
        cvzx_graph = to_graph(tensor)
        assert get_root_node(cvzx_graph) == tensor.id

        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        cvzx_graph = to_graph(contracted)
        assert get_root_node(cvzx_graph) == contracted.id

        comp = CompositionDiagram([TensorDiagram([self.q1, self.q2]), self.swap])
        cvzx_graph = to_graph(comp)
        assert get_root_node(cvzx_graph) == comp.id

    def test_container_id_tracking(self):
        """Test that container_id is correctly tracked for all nodes."""
        comp = CompositionDiagram([self.swap, TensorDiagram([self.q2, self.q3])])
        cvzx_graph = to_graph(comp)
        graph = cvzx_graph.graph

        assert get_immediate_container(cvzx_graph, self.swap.id) == comp.id

        tensor_id = None
        for idx in graph.node_indices():
            if graph[idx].get("container_type") == "tensor":
                tensor_id = graph[idx]["id"]
                break
        assert tensor_id is not None
        assert get_immediate_container(cvzx_graph, self.q2.id) == tensor_id
        assert get_immediate_container(cvzx_graph, self.q3.id) == tensor_id

        assert get_immediate_container(cvzx_graph, tensor_id) == comp.id
        assert get_immediate_container(cvzx_graph, comp.id) is None

    def test_get_nodes_by_container(self):
        """Test getting all nodes belonging to a container."""
        comp = CompositionDiagram([self.swap, TensorDiagram([self.q2, self.q3])])
        cvzx_graph = to_graph(comp)
        graph = cvzx_graph.graph

        tensor_id = None
        for idx in graph.node_indices():
            if graph[idx].get("container_type") == "tensor":
                tensor_id = graph[idx]["id"]
                break

        assert tensor_id is not None
        nodes_in_tensor = get_nodes_by_container(cvzx_graph, tensor_id)
        assert self.q2.id in nodes_in_tensor
        assert self.q3.id in nodes_in_tensor
        assert self.swap.id not in nodes_in_tensor

        nodes_in_comp = get_nodes_by_container(cvzx_graph, comp.id)
        assert self.swap.id in nodes_in_comp
        assert tensor_id in nodes_in_comp

    def test_get_connectivity(self):
        """Test getting connectivity from composition container."""
        conn = {0: 1, 1: 0}
        comp = CompositionDiagram([self.q_1x2, self.q_2x1], {0: conn})
        cvzx_graph = to_graph(comp)

        retrieved_conn = get_connectivity(cvzx_graph, comp.id)
        assert retrieved_conn is not None
        assert retrieved_conn[0] == conn

        tensor = TensorDiagram([self.q1, self.q2])
        cvzx_graph = to_graph(tensor)
        assert get_connectivity(cvzx_graph, tensor.id) is None


class TestToGraphDoesNotMutateInput:
    """`to_graph` must be a pure read of the `diagram` it is given."""

    def _snapshot(self, diagram: Diagram) -> str:
        """A recursive snapshot of every object."""
        if isinstance(diagram, (TensorDiagram, CompositionDiagram)):
            own = {k: v for k, v in vars(diagram).items() if k != "diagrams"}
            children = tuple(self._snapshot(d) for d in diagram.diagrams)
            return repr((type(diagram).__name__, own, children))
        if isinstance(diagram, ContractedDiagram):
            own = {k: v for k, v in vars(diagram).items() if k not in {"first", "second"}}
            return repr((type(diagram).__name__, own, self._snapshot(diagram.first), self._snapshot(diagram.second)))
        return repr((type(diagram).__name__, dict(vars(diagram))))

    def test_to_graph_does_not_mutate_a_flat_diagram(self):
        """A single composed gate, including a feedforward/measurement_ids leaf."""
        meas = QSpider(1, 0, ZxPoly({1: 2.0}))
        gate = DisplacementGate(alpha=1.0 + 0.5j, feedforward=True, measurement_ids={meas.id})
        diagram = CompositionDiagram([QSpider(1, 1, ZxPoly({2: 2.0})), gate])

        before = self._snapshot(diagram)
        to_graph(diagram)
        after = self._snapshot(diagram)

        assert before == after

    def test_to_graph_does_not_mutate_a_deeply_nested_diagram(self):
        """Tensor(Contracted(...), Composition(...)) -- exercises every container kind."""
        contracted = ContractedDiagram(
            QSpider(1, 2, ZxPoly({2: 2.0})),
            PSpider(2, 1, ZxPoly({3: 3.0})),
            [0],
            [0],
            [0],
            [0],
        )
        composition = CompositionDiagram([QSpider(1, 1, ZxPoly({})), PhaseRotationGate(theta=pi / 4)])
        diagram = TensorDiagram([contracted, composition])

        before = self._snapshot(diagram)
        to_graph(diagram)
        after = self._snapshot(diagram)

        assert before == after

    def test_to_graph_called_twice_gives_identical_results(self):
        """A second call on an already-converted-once diagram must match the first."""
        diagram = CompositionDiagram([QSpider(1, 1, ZxPoly({2: 2.0})), PSpider(1, 1, ZxPoly({3: 3.0}))])

        cvzx_first = to_graph(diagram)
        cvzx_second = to_graph(diagram)
        first = cvzx_first.graph
        second = cvzx_second.graph

        assert len(first) == len(second)
        assert len(first.edge_list()) == len(second.edge_list())
        assert repr(to_diagram(cvzx_first)) == repr(to_diagram(cvzx_second))

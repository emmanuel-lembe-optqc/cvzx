"""Tests for graph extraction utilities for CV ZX diagrams.

These tests verify that to_graph correctly converts CV ZX diagrams to directed graphs
and that to_diagram correctly reconstructs the original diagrams.
"""

from math import pi

from mqc3.zx.base_gates import (
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
from mqc3.zx.gates import (
    BeamsplitterGate,
    ControlledSumGate,
    ControlledZGate,
    DisplacementGate,
    PhaseRotationGate,
    SqueezingGate,
)
from mqc3.zx.nx_graph import (
    get_connectivity,
    get_contracted_connections,
    get_immediate_container,
    get_nodes_by_container,
    get_root_node,
    to_diagram,
    to_graph,
)


class TestgraphraphConversion:
    """Test suite for to_graph and to_diagram functions."""

    def setup_method(self):
        """Set up common objects for testing."""
        self.zero = ZxPoly({})
        self.phase_q = ZxPoly({2: 2.0})
        self.phase_p = ZxPoly({3: 3.0})

        # Proper diagrams
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

        # Other diagrams
        self.disp = DisplacementGate(alpha=1.0 + 0.5j)
        self.ph_rot = PhaseRotationGate(theta=pi / 4)
        self.sq_gate = SqueezingGate(tau=0.5)
        self.ctrl_sum_gate = ControlledSumGate(gain=1.0, control=1, target=2)
        self.ctrl_z_gate = ControlledZGate(gain=1.0)
        self.beam_splitter = BeamsplitterGate(theta=pi / 4)

        # Multiple arity spiders
        self.q_1x2 = QSpider(1, 2, self.phase_q)
        self.q_2x1 = QSpider(2, 1, self.phase_p)
        self.p_2x1 = PSpider(2, 1, self.phase_q)

        # Composition connectivity
        self.conn_swap = {0: 0, 1: 1}
        self.conn_cross = {0: 1, 1: 0}

    def _assert_diagram_equality(self, original: Diagram, reconstructed: Diagram) -> None:
        """Helper to assert that two diagrams are equal by comparing their string representations."""
        # For proper diagrams, compare directly
        if isinstance(original, (QSpider, PSpider, Swap, Fourier, FourierInv, Fourier2)):
            assert original == reconstructed
            return

        # For containers, compare structure
        if isinstance(original, CompositionDiagram):
            assert isinstance(reconstructed, CompositionDiagram)
            assert len(original.diagrams) == len(reconstructed.diagrams)
            assert original.connectivity == reconstructed.connectivity
            for _, (orig_sub, recon_sub) in enumerate(zip(original.diagrams, reconstructed.diagrams, strict=False)):
                self._assert_diagram_equality(orig_sub, recon_sub)
            return

        if isinstance(original, TensorDiagram):
            assert isinstance(reconstructed, TensorDiagram)
            assert len(original.diagrams) == len(reconstructed.diagrams)
            for _, (orig_sub, recon_sub) in enumerate(zip(original.diagrams, reconstructed.diagrams, strict=False)):
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
        graph = to_graph(diagram)
        reconstructed = to_diagram(graph)
        self._assert_diagram_equality(diagram, reconstructed)

    # =========================================================================
    # 1. Proper Diagram Tests
    # =========================================================================

    def test_proper_qspider(self):
        """Test conversion of a single QSpider."""
        graph = to_graph(self.q1)

        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0

        attrs = graph.nodes[self.q1.id]
        assert attrs["id"] == self.q1.id
        assert attrs["type"] == "QSpider"
        assert attrs["kind"] == "proper"
        assert attrs["phase"] == self.phase_q
        assert attrs["n_inputs"] == 1
        assert attrs["n_outputs"] == 1
        assert attrs["diagram"] == self.q1
        assert attrs["container_id"] is None
        assert attrs["is_root"]

        self._assert_roundtrip(self.q1)

    def test_proper_pspider(self):
        """Test conversion of a single PSpider."""
        graph = to_graph(self.p1)

        assert graph.number_of_nodes() == 1
        attrs = graph.nodes[self.p1.id]
        assert attrs["type"] == "PSpider"
        assert attrs["phase"] == self.phase_q

        self._assert_roundtrip(self.p1)

    def test_proper_fourier(self):
        """Test conversion of a single Fourier gate."""
        graph = to_graph(self.fourier)

        assert graph.number_of_nodes() == 1
        attrs = graph.nodes[self.fourier.id]
        assert attrs["type"] == "Fourier"
        assert attrs["phase"] is None
        assert attrs["n_inputs"] == 1
        assert attrs["n_outputs"] == 1

        self._assert_roundtrip(self.fourier)

    def test_proper_swap(self):
        """Test conversion of a single Swap gate."""
        graph = to_graph(self.swap)

        assert graph.number_of_nodes() == 1
        attrs = graph.nodes[self.swap.id]
        assert attrs["type"] == "Swap"
        assert attrs["n_inputs"] == 2
        assert attrs["n_outputs"] == 2

        self._assert_roundtrip(self.swap)

    def test_proper_multi_arity_spider(self):
        """Test conversion of a spider with multiple inputs/outputs."""
        graph = to_graph(self.q_1x2)

        assert graph.number_of_nodes() == 1
        attrs = graph.nodes[self.q_1x2.id]
        assert attrs["n_inputs"] == 1
        assert attrs["n_outputs"] == 2
        assert attrs["external_inputs"] == [0]
        assert attrs["external_outputs"] == [0, 1]

        self._assert_roundtrip(self.q_1x2)

    def test_displacement(self):
        """Test conversion of a DisplacementGate."""
        graph = to_graph(self.disp)

        assert graph.number_of_nodes() == 1
        attrs = graph.nodes[self.disp.id]
        assert attrs["type"] == "DisplacementGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.disp.alpha

    def test_rotation(self):
        """Test conversion of a PhaseRotationGate."""
        graph = to_graph(self.ph_rot)

        assert graph.number_of_nodes() == 1
        attrs = graph.nodes[self.ph_rot.id]
        assert attrs["type"] == "PhaseRotationGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.ph_rot.theta

    def test_squeezing(self):
        """Test conversion of a SqueezingGate."""
        graph = to_graph(self.sq_gate)

        assert graph.number_of_nodes() == 1
        attrs = graph.nodes[self.sq_gate.id]
        assert attrs["type"] == "SqueezingGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.sq_gate.tau

    def test_beamsplitter(self):
        """Test conversion of a BeamsplitterGate."""
        graph = to_graph(self.beam_splitter)

        assert graph.number_of_nodes() == 1
        attrs = graph.nodes[self.beam_splitter.id]
        assert attrs["type"] == "BeamsplitterGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.beam_splitter.theta

    def test_controlledsumgate(self):
        """Test conversion of a ControlledSumGate."""
        graph = to_graph(self.ctrl_sum_gate)

        assert graph.number_of_nodes() == 1
        attrs = graph.nodes[self.ctrl_sum_gate.id]
        assert attrs["type"] == "ControlledSumGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.ctrl_sum_gate.gain

    def test_controlledzgate(self):
        """Test conversion of a ControlledSumGate."""
        graph = to_graph(self.ctrl_z_gate)

        assert graph.number_of_nodes() == 1
        attrs = graph.nodes[self.ctrl_z_gate.id]
        assert attrs["type"] == "ControlledZGate"
        assert attrs["kind"] == "compact"
        assert attrs["phase"] == self.ctrl_z_gate.gain

    # =========================================================================
    # 2. Composition of Proper Diagrams Tests
    # =========================================================================

    def test_composition_two_proper_diagrams(self):
        """Test conversion of CompositionDiagram with two proper diagrams."""
        comp = CompositionDiagram([self.q1, self.q2])
        graph = to_graph(comp)

        # Check nodes: q1, q2, and composition container
        assert graph.number_of_nodes() == 3
        assert graph.has_node(self.q1.id)
        assert graph.has_node(self.q2.id)
        assert graph.has_node(comp.id)

        # Check edges: q1 → q2
        assert graph.number_of_edges() == 1
        edge_data = graph.get_edge_data(self.q1.id, self.q2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]
        assert edge_data["edge_type"] == "composition"
        assert not edge_data["internal"]
        assert edge_data["left_idx"] == 0
        assert edge_data["right_idx"] == 1

        # Check container attributes
        attrs = graph.nodes[comp.id]
        assert attrs["kind"] == "container"
        assert attrs["container_type"] == "composition"
        assert attrs["sub_diagram_ids"] == [self.q1.id, self.q2.id]
        assert attrs["connectivity"] == comp.connectivity
        assert attrs["is_root"]

        # Check container_id of proper nodes
        assert graph.nodes[self.q1.id]["container_id"] == comp.id
        assert graph.nodes[self.q2.id]["container_id"] == comp.id

        self._assert_roundtrip(comp)

    def test_composition_three_proper_diagrams(self):
        """Test conversion of CompositionDiagram with three proper diagrams."""
        comp = CompositionDiagram([self.q1, self.q2, self.q3])
        graph = to_graph(comp)

        assert graph.number_of_nodes() == 4  # 3 proper + 1 container
        assert graph.number_of_edges() == 2

        # q1 → q2
        edge_data = graph.get_edge_data(self.q1.id, self.q2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        # q2 → q3
        edge_data = graph.get_edge_data(self.q2.id, self.q3.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        self._assert_roundtrip(comp)

    def test_composition_proper_with_connectivity(self):
        """Test CompositionDiagram with non-trivial connectivity."""
        conn = {0: 1, 1: 0}
        comp = CompositionDiagram([self.q_1x2, self.q_2x1], {0: conn})
        graph = to_graph(comp)

        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 1

        edge_data = graph.get_edge_data(self.q_1x2.id, self.q_2x1.id)
        for i, key in enumerate(edge_data["source_ports"]):
            assert conn[key] == edge_data["target_ports"][i]

        self._assert_roundtrip(comp)

    def test_composition_mixed_proper_types(self):
        """Test composition of different proper diagram types."""
        comp = CompositionDiagram([self.q1, self.fourier, self.q2])
        graph = to_graph(comp)

        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 2
        assert graph.has_edge(self.q1.id, self.fourier.id)
        assert graph.has_edge(self.fourier.id, self.q2.id)

        self._assert_roundtrip(comp)

    # =========================================================================
    # 3. Tensor of Proper Diagrams Tests
    # =========================================================================

    def test_tensor_proper_diagrams(self):
        """Test conversion of TensorDiagram of proper diagrams."""
        tensor = TensorDiagram([self.q1, self.q2, self.q3])
        graph = to_graph(tensor)

        # Nodes: 3 proper + 1 tensor container
        assert graph.number_of_nodes() == 4
        assert graph.has_node(self.q1.id)
        assert graph.has_node(self.q2.id)
        assert graph.has_node(self.q3.id)
        assert graph.has_node(tensor.id)

        # No edges between tensor components
        assert graph.number_of_edges() == 0

        # Check tensor container attributes
        attrs = graph.nodes[tensor.id]
        assert attrs["kind"] == "container"
        assert attrs["container_type"] == "tensor"
        assert attrs["sub_diagram_ids"] == [self.q1.id, self.q2.id, self.q3.id]
        assert attrs["is_root"] is True

        # Check external port mappings
        assert attrs["external_input_mapping"] == {0: (0, 0), 1: (1, 0), 2: (2, 0)}
        assert attrs["external_output_mapping"] == {0: (0, 0), 1: (1, 0), 2: (2, 0)}

        self._assert_roundtrip(tensor)

    def test_tensor_proper_diagrams_with_arity(self):
        """Test TensorDiagram with different arities."""
        tensor = TensorDiagram([self.q_1x2, self.q_2x1])
        graph = to_graph(tensor)

        assert graph.number_of_nodes() == 3
        attrs = graph.nodes[tensor.id]
        assert attrs["external_input_mapping"] == {0: (0, 0), 1: (1, 0), 2: (1, 1)}
        assert attrs["external_output_mapping"] == {0: (0, 0), 1: (0, 1), 2: (1, 0)}

        self._assert_roundtrip(tensor)

    # =========================================================================
    # 4. Composition containing Tensors Tests
    # =========================================================================

    def test_composition_of_tensors1(self):
        """Test conversion of CompositionDiagram containing TensorDiagrams."""
        tensor1 = TensorDiagram([self.q1, self.q2])
        tensor2 = TensorDiagram([self.q3, self.q4])
        comp = CompositionDiagram([tensor1, tensor2])
        graph = to_graph(comp)

        # Nodes: 4 proper + 2 tensor containers + 1 composition container
        assert graph.number_of_nodes() == 7

        # Check hierarchy
        comp_attrs = graph.nodes[comp.id]
        assert comp_attrs["sub_diagram_ids"] == [tensor1.id, tensor2.id]

        tensor1_attrs = graph.nodes[tensor1.id]
        assert tensor1_attrs["container_id"] == comp.id
        assert tensor1_attrs["sub_diagram_ids"] == [self.q1.id, self.q2.id]

        tensor2_attrs = graph.nodes[tensor2.id]
        assert tensor2_attrs["container_id"] == comp.id
        assert tensor2_attrs["sub_diagram_ids"] == [self.q3.id, self.q4.id]

        # Edges: q1 → q3, q2 -> q3 (outputs of tensor1 to inputs of tensor2)
        assert graph.number_of_edges() == 2
        edge_data = graph.get_edge_data(self.q1.id, self.q3.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]
        edge_data = graph.get_edge_data(self.q2.id, self.q4.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        self._assert_roundtrip(comp)

    def test_composition_of_tensors2(self):
        """Test conversion of CompositionDiagram containing TensorDiagrams."""
        two_fouriers_tensor = self.fourier.tensor(self.fourier_inv)
        swap_conn = two_fouriers_tensor.compose(self.swap, connectivity={0: 1, 1: 0})

        graph = to_graph(swap_conn)

        # Nodes: 2 Fourier proper + 1 Swap proper + 1 tensor container + 1 composition container
        # = 5 nodes total
        assert graph.number_of_nodes() == 5

        # Check hierarchy
        # Find the tensor container node
        tensor_node = None
        for n, attrs in graph.nodes(data=True):
            if attrs.get("container_type") == "tensor":
                tensor_node = n
                break
        assert tensor_node is not None

        # Find the composition container node
        comp_node = None
        for n, attrs in graph.nodes(data=True):
            if attrs.get("container_type") == "composition":
                comp_node = n
                break
        assert comp_node is not None

        # Composition container should be the root
        comp_attrs = graph.nodes[comp_node]
        assert comp_attrs["is_root"] is True
        assert comp_attrs["sub_diagram_ids"] == [self.swap.id, tensor_node]

        # Tensor container should be inside the composition
        tensor_attrs = graph.nodes[tensor_node]
        assert tensor_attrs["container_id"] == comp_node
        assert tensor_attrs["sub_diagram_ids"] == [self.fourier.id, self.fourier_inv.id]

        # Fourier nodes should be inside the tensor
        for f_node in [self.fourier.id, self.fourier.id]:
            assert graph.nodes[f_node]["container_id"] == tensor_node

        # Swap node should be inside the composition
        assert graph.nodes[self.swap.id]["container_id"] == comp_node

        # Find the Fourier nodes in the graph
        fourier_nodes = [
            n
            for n, attrs in graph.nodes(data=True)
            if "Fourier" in attrs.get("type") and attrs.get("kind") == "proper"
        ]
        assert len(fourier_nodes) == 2

        # Check edges
        assert graph.number_of_edges() == 2

        # Check first Fourier → Swap connection
        edge_data = graph.get_edge_data(self.swap.id, fourier_nodes[0])
        assert edge_data is not None
        assert "source_ports" in edge_data
        assert "target_ports" in edge_data
        assert edge_data["source_ports"] == [1]
        assert edge_data["target_ports"] == [0]  # Swapped: input 1

        # Check second Fourier → Swap connection
        edge_data = graph.get_edge_data(self.swap.id, fourier_nodes[1])
        assert edge_data is not None
        assert "source_ports" in edge_data
        assert "target_ports" in edge_data
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]  # Swapped: input 0

        self._assert_roundtrip(swap_conn)

    # =========================================================================
    # 5. Tensor containing Compositions Tests
    # =========================================================================

    def test_tensor_of_compositions(self):
        """Test conversion of TensorDiagram containing CompositionDiagrams."""
        comp1 = CompositionDiagram([self.q1, self.q2])
        comp2 = CompositionDiagram([self.q3, self.q4])
        tensor = TensorDiagram([comp1, comp2])
        comp1 = tensor.diagrams[0]
        comp2 = tensor.diagrams[1]
        graph = to_graph(tensor)

        # Nodes: 4 proper + 2 composition containers + 1 tensor container
        assert graph.number_of_nodes() == 7
        # Check hierarchy
        tensor_attrs = graph.nodes[tensor.id]
        assert tensor_attrs["sub_diagram_ids"] == [comp1.id, comp2.id]

        comp1_attrs = graph.nodes[comp1.id]
        assert comp1_attrs["container_id"] == tensor.id
        assert comp1_attrs["sub_diagram_ids"] == [self.q1.id, self.q2.id]

        comp2_attrs = graph.nodes[comp2.id]
        assert comp2_attrs["container_id"] == tensor.id
        assert comp2_attrs["sub_diagram_ids"] == [self.q3.id, self.q4.id]

        # No edges between compositions (tensor product)
        assert graph.number_of_edges() == 2  # q1→q2 and q3→q4

        self._assert_roundtrip(tensor)

    # =========================================================================
    # 6. Contracted of Proper Diagrams Tests
    # =========================================================================

    def test_contracted_proper_diagrams(self):
        """Test conversion of ContractedDiagram with proper diagrams."""
        contracted = ContractedDiagram(self.q_1x2, self.p_2x1, [0], [0], [0], [0])
        graph = to_graph(contracted)

        # Nodes: 2 proper + 1 contracted container
        assert graph.number_of_nodes() == 3

        # Check contracted container attributes
        attrs = graph.nodes[contracted.id]
        assert attrs["kind"] == "container"
        assert attrs["container_type"] == "contracted"
        assert attrs["first_id"] == self.q_1x2.id
        assert attrs["second_id"] == self.p_2x1.id
        assert attrs["I1"] == [0]
        assert attrs["I2"] == [0]
        assert attrs["J1"] == [0]
        assert attrs["J2"] == [0]

        # Check internal edges: I1→I2 (q_1x2 → p_2x1)
        assert graph.has_edge(self.q_1x2.id, self.p_2x1.id)
        edge_data = graph.get_edge_data(self.q_1x2.id, self.p_2x1.id)
        assert edge_data["edge_type"] == "contracted_internal"
        assert edge_data["internal"] is True
        assert edge_data["connection_type"] == "I1_I2"
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        # Check internal edges: J2→J1 (p_2x1 → q_1x2)
        assert graph.has_edge(self.p_2x1.id, self.q_1x2.id)
        edge_data = graph.get_edge_data(self.p_2x1.id, self.q_1x2.id)
        assert edge_data["connection_type"] == "J2_J1"

        # Check external input mapping
        assert attrs["external_input_mapping"][0] == ("second", 1)

        # Check external output mapping
        assert attrs["external_output_mapping"][0] == ("first", 1)

        self._assert_roundtrip(contracted)

    # =========================================================================
    # 7. Composition containing Contracted Diagrams Tests
    # =========================================================================

    def test_composition_of_contracted_diagrams(self):
        """Test conversion of CompositionDiagram containing ContractedDiagrams."""
        contracted1 = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        contracted2 = ContractedDiagram(self.q3, self.q4, [0], [0], [], [])
        comp = CompositionDiagram([contracted1, contracted2])
        graph = to_graph(comp)

        # Nodes: 4 proper + 2 contracted containers + 1 composition container
        assert graph.number_of_nodes() == 7

        # Edges: contracted1 → contracted2 (q2 → q3)
        assert graph.has_edge(self.q2.id, self.q3.id)

        # Check hierarchy
        comp_attrs = graph.nodes[comp.id]
        assert comp_attrs["sub_diagram_ids"] == [contracted1.id, contracted2.id]

        contracted1_attrs = graph.nodes[contracted1.id]
        assert contracted1_attrs["container_id"] == comp.id
        assert contracted1_attrs["first_id"] == self.q1.id
        assert contracted1_attrs["second_id"] == self.q2.id

        contracted2_attrs = graph.nodes[contracted2.id]
        assert contracted2_attrs["container_id"] == comp.id
        assert contracted2_attrs["first_id"] == self.q3.id
        assert contracted2_attrs["second_id"] == self.q4.id

        self._assert_roundtrip(comp)

    # =========================================================================
    # 8. Tensor containing Contracted Diagrams Tests
    # =========================================================================

    def test_tensor_of_contracted_diagrams(self):
        """Test conversion of TensorDiagram containing ContractedDiagrams."""
        contracted1 = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        contracted2 = ContractedDiagram(self.q3, self.q4, [0], [0], [], [])
        tensor = TensorDiagram([contracted1, contracted2])
        graph = to_graph(tensor)

        # Nodes: 4 proper + 2 contracted containers + 1 tensor container
        assert graph.number_of_nodes() == 7

        # No edges between contracted diagrams (tensor product)
        assert graph.number_of_edges() == 2  # internal edges only

        self._assert_roundtrip(tensor)

    # =========================================================================
    # 9. Complex Structures Tests
    # =========================================================================

    def test_complex_nested_structure1(self):
        """Test conversion of complex nested structures."""
        # Create: Tensor([Composition([q1, q2]), Contracted(q3, q4)])
        comp = CompositionDiagram([self.q1, self.q2])
        contracted = ContractedDiagram(self.q3, self.q4, [0], [0], [], [])
        tensor = TensorDiagram([comp, contracted])
        comp = tensor.diagrams[0]
        contracted = tensor.diagrams[1]
        graph = to_graph(tensor)

        # Nodes: 4 proper + 2 containers (comp, contracted) + 1 tensor container
        assert graph.number_of_nodes() == 7

        # Check hierarchy
        tensor_attrs = graph.nodes[tensor.id]
        assert tensor_attrs["sub_diagram_ids"] == [comp.id, contracted.id]

        comp_attrs = graph.nodes[comp.id]
        assert comp_attrs["container_id"] == tensor.id
        assert comp_attrs["sub_diagram_ids"] == [self.q1.id, self.q2.id]

        contracted_attrs = graph.nodes[contracted.id]
        assert contracted_attrs["container_id"] == tensor.id

        # Edges: q1→q2 (inside comp)
        assert graph.has_edge(self.q1.id, self.q2.id)
        edge_data = graph.get_edge_data(self.q1.id, self.q2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        # Internal edges inside contracted
        assert graph.has_edge(self.q3.id, self.q4.id)
        edge_data = graph.get_edge_data(self.q3.id, self.q4.id)
        assert edge_data["edge_type"] == "contracted_internal"
        assert edge_data["internal"] is True
        assert edge_data["connection_type"] == "I1_I2"

        # Roundtrip test
        self._assert_roundtrip(tensor)

    def test_complex_nested_structure2(self):  # noqa: PLR0914, PLR0915
        """Test conversion of complex nested structures."""
        fourier = self.fourier
        fourier2 = Fourier()
        fourier_squared = self.fourier2
        fourier_squared2 = Fourier2()
        fourier_inv = self.fourier_inv
        fourier_inv2 = FourierInv()
        swap_node1 = Swap()
        swap_node2 = Swap()
        swap_node3 = Swap()
        q_spider1 = QSpider(3, 3, self.phase_q)
        q_spider2 = QSpider(3, 3, 4 * self.phase_q)
        q_spider3 = QSpider(3, 3, 10 * self.phase_q)
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
        nested_block_conn = CompositionDiagram([q_spider1, q_spider2])
        nested_block_conn = nested_block_conn.compose(q_spider3)
        nested_block_conn = nested_block_conn.compose(contracted_diagram_1)

        large_comp_conn = large_comp_conn.compose(nested_block_conn, connectivity={0: 1, 1: 2, 2: 0})
        large_comp_conn = flatten_composition(large_comp_conn)
        final_large_conn = fourier.tensor(large_comp_conn)
        large_comp_conn = final_large_conn.diagrams[1]

        graph = to_graph(final_large_conn)

        # Count nodes:
        assert graph.number_of_nodes() == 21

        # Check that the root is the outer tensor
        root = get_root_node(graph)
        assert root is not None
        root_attrs = graph.nodes[root]
        assert root_attrs["type"] == "TensorDiagram"
        assert root_attrs["is_root"] is True
        assert root_attrs["container_type"] == "tensor"
        assert root_attrs["sub_diagram_ids"] == [fourier.id, large_comp_conn.id]

        # Check outer tensor port mappings
        outer_tensor_attrs = graph.nodes[root]
        assert outer_tensor_attrs["n_inputs"] == final_large_conn.num_inputs
        assert outer_tensor_attrs["n_outputs"] == final_large_conn.num_outputs

        # Find the composition container
        comp_node = None
        for n, attrs in graph.nodes(data=True):
            if attrs.get("id") == large_comp_conn.id:
                comp_node = n
                break
        assert comp_node is not None

        # Check composition structure
        comp_attrs = graph.nodes[comp_node]
        assert comp_attrs["container_type"] == "composition"
        assert len(comp_attrs["sub_diagram_ids"]) == 8
        assert comp_attrs["container_id"] == root

        # Check composition
        connectivity = get_connectivity(graph, comp_node)
        assert connectivity is not None
        assert connectivity == large_comp_conn.connectivity
        assert len(graph.nodes[comp_node]["sub_diagram_ids"]) == 8

        # Find tensor containers
        tensor_nodes = [
            n for n, attrs in graph.nodes(data=True) if attrs.get("container_type") == "tensor" and n != root
        ]
        assert len(tensor_nodes) == 4

        # Check large_tensor (first sub-diagram of composition)
        large_tensor_node = None
        for n in tensor_nodes:
            if graph.nodes[n].get("sub_diagram_ids") == [fourier.id, fourier_squared.id, fourier_inv.id]:
                large_tensor_node = n
                break
        assert large_tensor_node is not None
        assert graph.nodes[large_tensor_node]["container_id"] == comp_node

        # Check swap_tensor (second and third sub-diagrams of composition)
        swap_tensor_nodes = [n for n in tensor_nodes if len(graph.nodes[n].get("sub_diagram_ids")) == 2]
        assert len(swap_tensor_nodes) == 3
        for n in swap_tensor_nodes:
            assert graph.nodes[n]["container_id"] == comp_node

        # Check contracted diagram inside nested composition
        contracted_node = None
        for n, attrs in graph.nodes(data=True):
            if attrs.get("container_type") == "contracted":
                contracted_node = n
                break
        assert contracted_node is not None
        assert graph.nodes[contracted_node]["container_id"] == comp_node
        assert graph.nodes[contracted_node]["first_id"] == q_spider_5x5.id
        assert graph.nodes[contracted_node]["second_id"] == p_spider_4x4.id

        # Check contracted connections
        contracted_connections = get_contracted_connections(graph, contracted_node)
        assert contracted_connections is not None
        assert contracted_connections["I1"] == [0, 1, 4]
        assert contracted_connections["I2"] == [1, 2, 3]
        assert contracted_connections["J1"] == [0, 1, 2]
        assert contracted_connections["J2"] == [1, 2, 3]

        # Check contracted port mappings
        contracted_attrs = graph.nodes[contracted_node]
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

        # Check Connection nested_block → tensor1
        assert graph.has_edge(q_spider2.id, swap_node1.id)
        edge_data = graph.get_edge_data(q_spider2.id, swap_node1.id)
        assert edge_data["source_ports"] == [0, 2]
        assert edge_data["target_ports"] == [1, 0]

        assert graph.has_edge(q_spider2.id, fourier2.id)
        edge_data = graph.get_edge_data(q_spider2.id, fourier2.id)
        assert edge_data["source_ports"] == [1]
        assert edge_data["target_ports"] == [0]

        # Check Connexion tensor 1 -> large_tensor_conn
        assert graph.has_edge(swap_node1.id, fourier.id)
        edge_data = graph.get_edge_data(swap_node1.id, fourier.id)
        assert edge_data["source_ports"] == [1]
        assert edge_data["target_ports"] == [0]

        assert graph.has_edge(swap_node1.id, fourier_inv.id)
        edge_data = graph.get_edge_data(swap_node1.id, fourier_inv.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        assert graph.has_edge(fourier2.id, fourier_squared.id)
        edge_data = graph.get_edge_data(fourier2.id, fourier_squared.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        # Check Connexion large_tensor_conn -> tensor2
        assert graph.has_edge(fourier_squared.id, swap_node2.id)
        edge_data = graph.get_edge_data(fourier_squared.id, swap_node2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        assert graph.has_edge(fourier_inv.id, fourier_squared2.id)
        edge_data = graph.get_edge_data(fourier_squared.id, swap_node2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        assert graph.has_edge(fourier.id, swap_node2.id)
        edge_data = graph.get_edge_data(fourier.id, swap_node2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [1]

        # Check Connection tensor2 → tensor3
        assert graph.has_edge(swap_node2.id, swap_node3.id)
        edge_data = graph.get_edge_data(swap_node2.id, swap_node3.id)
        assert edge_data["source_ports"] == [0, 1]
        assert edge_data["target_ports"] == [0, 1]

        assert graph.has_edge(fourier_squared2.id, fourier_inv2.id)
        edge_data = graph.get_edge_data(fourier_squared2.id, fourier_inv2.id)
        assert edge_data["source_ports"] == [0]
        assert edge_data["target_ports"] == [0]

        # Roundtrip test
        self._assert_roundtrip(final_large_conn)

    # =========================================================================
    # 10. Miscellaneous Tests
    # =========================================================================

    def test_root_node_identification(self):
        """Test that the root node is correctly identified."""
        # Root is composition
        comp = CompositionDiagram([self.q1, self.q2])
        graph = to_graph(comp)
        assert get_root_node(graph) == comp.id

        # Root is tensor
        tensor = TensorDiagram([self.q1, self.q2])
        graph = to_graph(tensor)
        assert get_root_node(graph) == tensor.id

        # Root is contracted
        contracted = ContractedDiagram(self.q1, self.q2, [0], [0], [], [])
        graph = to_graph(contracted)
        assert get_root_node(graph) == contracted.id

        # Nested: root is composition
        comp = CompositionDiagram([TensorDiagram([self.q1, self.q2]), self.swap])
        graph = to_graph(comp)
        assert get_root_node(graph) == comp.id

    def test_container_id_tracking(self):
        """Test that container_id is correctly tracked for all nodes."""
        comp = CompositionDiagram([self.swap, TensorDiagram([self.q2, self.q3])])
        graph = to_graph(comp)

        # q1 is directly in comp
        assert get_immediate_container(graph, self.swap.id) == comp.id

        # q2 and q3 are in the tensor
        tensor_id = None
        for node, attrs in graph.nodes(data=True):
            if attrs.get("container_type") == "tensor":
                tensor_id = node
                break
        assert tensor_id is not None
        assert get_immediate_container(graph, self.q2.id) == tensor_id
        assert get_immediate_container(graph, self.q3.id) == tensor_id

        # Tensor is in comp
        assert get_immediate_container(graph, tensor_id) == comp.id

        # Comp has no container
        assert get_immediate_container(graph, comp.id) is None

    def test_get_nodes_by_container(self):
        """Test getting all nodes belonging to a container."""
        comp = CompositionDiagram([self.swap, TensorDiagram([self.q2, self.q3])])
        graph = to_graph(comp)

        # Find tensor node
        tensor_id = None
        for node, attrs in graph.nodes(data=True):
            if attrs.get("container_type") == "tensor":
                tensor_id = node
                break

        # Nodes in tensor: q2, q3, and the tensor itself
        nodes_in_tensor = get_nodes_by_container(graph, tensor_id)
        assert self.q2.id in nodes_in_tensor
        assert self.q3.id in nodes_in_tensor
        assert self.swap.id not in nodes_in_tensor

        # Nodes in comp: all nodes
        nodes_in_comp = get_nodes_by_container(graph, comp.id)
        print(nodes_in_comp)
        assert self.swap.id in nodes_in_comp
        assert tensor_id in nodes_in_comp

    def test_get_connectivity(self):
        """Test getting connectivity from composition container."""
        conn = {0: 1, 1: 0}
        comp = CompositionDiagram([self.q_1x2, self.q_2x1], {0: conn})
        graph = to_graph(comp)

        retrieved_conn = get_connectivity(graph, comp.id)
        assert retrieved_conn[0] == conn

        # Non-composition container returns None
        tensor = TensorDiagram([self.q1, self.q2])
        graph = to_graph(tensor)
        assert get_connectivity(graph, tensor.id) is None

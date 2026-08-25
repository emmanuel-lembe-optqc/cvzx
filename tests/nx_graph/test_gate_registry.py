"""Tests for the GateRegister class.

This module contains comprehensive unit tests for the GateRegister class,
which tracks specific gate types and nodes in a CV ZX graph.

"""

from math import pi

import networkx as nx
import pytest

from cvzx.base_gates import ZxPoly
from cvzx.gates import CubicPhaseGate, DisplacementGate, PhaseRotationGate, SqueezingGate
from cvzx.nx_graph import GateRegister


class TestGateRegister:
    """Test suite for GateRegister class."""

    def test_initialization(self):
        """Test that GateRegister initializes with empty sets."""
        reg = GateRegister()

        assert isinstance(reg.squeezing_gates, set)
        assert isinstance(reg.displacement_gates, set)
        assert isinstance(reg.rotation_gates, set)
        assert isinstance(reg.fourier_gates, set)
        assert isinstance(reg.identity_spiders, set)
        assert isinstance(reg.input_states, set)
        assert isinstance(reg.measurement_nodes, set)
        assert isinstance(reg.contracted_diagrams, set)
        assert isinstance(reg.tensor_nodes, set)
        assert isinstance(reg.composition_nodes, set)

        assert len(reg.squeezing_gates) == 0
        assert len(reg.displacement_gates) == 0
        assert len(reg.rotation_gates) == 0
        assert len(reg.fourier_gates) == 0
        assert len(reg.identity_spiders) == 0
        assert len(reg.input_states) == 0
        assert len(reg.measurement_nodes) == 0
        assert len(reg.contracted_diagrams) == 0
        assert len(reg.tensor_nodes) == 0
        assert len(reg.composition_nodes) == 0

    def test_add_node_squeezing_gate(self):
        """Test adding a squeezing gate node."""
        reg = GateRegister()
        node_id = 1
        attrs = {"kind": "compact", "type": "SqueezingGate", "num_inputs": 1, "num_outputs": 1}

        reg.add_node(node_id, attrs)
        assert node_id in reg.squeezing_gates
        assert node_id not in reg.displacement_gates
        assert node_id not in reg.rotation_gates
        assert node_id not in reg.fourier_gates

    def test_add_node_displacement_gate(self):
        """Test adding a displacement gate node."""
        reg = GateRegister()
        node_id = 2
        attrs = {"kind": "compact", "type": "DisplacementGate", "num_inputs": 1, "num_outputs": 1}

        reg.add_node(node_id, attrs)
        assert node_id in reg.displacement_gates
        assert node_id not in reg.squeezing_gates
        assert node_id not in reg.rotation_gates
        assert node_id not in reg.fourier_gates

    def test_add_node_rotation_gate(self):
        """Test adding a phase rotation gate node."""
        reg = GateRegister()
        node_id = 3
        attrs = {"kind": "compact", "type": "PhaseRotationGate", "num_inputs": 1, "num_outputs": 1}

        reg.add_node(node_id, attrs)
        assert node_id in reg.rotation_gates
        assert node_id not in reg.squeezing_gates
        assert node_id not in reg.displacement_gates
        assert node_id not in reg.fourier_gates

    def test_add_node_fourier_gates(self):
        """Test adding Fourier gate nodes."""
        reg = GateRegister()

        # Test Fourier
        node_id_1 = 4
        attrs_1 = {"kind": "proper", "type": "Fourier"}
        reg.add_node(node_id_1, attrs_1)
        assert node_id_1 in reg.fourier_gates

        # Test FourierInv
        node_id_2 = 5
        attrs_2 = {"kind": "proper", "type": "FourierInv"}
        reg.add_node(node_id_2, attrs_2)
        assert node_id_2 in reg.fourier_gates

        # Test Fourier2
        node_id_3 = 6
        attrs_3 = {"kind": "proper", "type": "Fourier2"}
        reg.add_node(node_id_3, attrs_3)
        assert node_id_3 in reg.fourier_gates

    def test_add_node_identity_spider(self):
        """Test adding identity spider nodes."""
        reg = GateRegister()

        # QSpider identity
        node_id_1 = 7
        attrs_1 = {
            "kind": "proper",
            "type": "QSpider",
            "num_inputs": 1,
            "num_outputs": 1,
            "phase": ZxPoly({0: 0}),
        }
        reg.add_node(node_id_1, attrs_1)
        assert node_id_1 in reg.identity_spiders

        # PSpider identity
        node_id_2 = 8
        attrs_2 = {
            "kind": "proper",
            "type": "PSpider",
            "num_inputs": 1,
            "num_outputs": 1,
            "phase": ZxPoly({}),
        }
        reg.add_node(node_id_2, attrs_2)
        assert node_id_2 in reg.identity_spiders

    def test_add_node_non_identity_spider(self):
        """Test adding a non-identity spider (should not be in identity_spiders)."""
        reg = GateRegister()
        node_id = 9
        attrs = {
            "kind": "proper",
            "type": "QSpider",
            "num_inputs": 1,
            "num_outputs": 1,
            "phase": ZxPoly({1: 2.0}),
        }

        reg.add_node(node_id, attrs)
        assert node_id not in reg.identity_spiders

    def test_add_node_identity_spider_wrong_arity(self):
        """Test that spiders with wrong arity are not identity spiders."""
        reg = GateRegister()
        node_id = 10
        attrs = {
            "kind": "proper",
            "type": "QSpider",
            "num_inputs": 2,
            "num_outputs": 1,
            "phase": ZxPoly({0: 0}),
        }

        reg.add_node(node_id, attrs)
        assert node_id not in reg.identity_spiders

    def test_add_node_input_state(self):
        """Test adding an input state node."""
        reg = GateRegister()
        node_id = 11
        attrs = {"kind": "proper", "num_inputs": 0, "num_outputs": 1}

        reg.add_node(node_id, attrs)
        assert node_id in reg.input_states

    def test_add_node_measurement(self):
        """Test adding a measurement node."""
        reg = GateRegister()
        node_id = 12
        attrs = {"kind": "proper", "num_inputs": 1, "num_outputs": 0}

        reg.add_node(node_id, attrs)
        assert node_id in reg.measurement_nodes

    def test_add_node_container_tensor(self):
        """Test adding a tensor diagram container node."""
        reg = GateRegister()
        node_id = 13
        attrs = {"kind": "container", "container_type": "tensor"}

        reg.add_node(node_id, attrs)
        assert node_id in reg.tensor_nodes
        assert node_id not in reg.composition_nodes
        assert node_id not in reg.contracted_diagrams

    def test_add_node_container_composition(self):
        """Test adding a composition diagram container node."""
        reg = GateRegister()
        node_id = 14
        attrs = {"kind": "container", "container_type": "composition"}

        reg.add_node(node_id, attrs)
        assert node_id in reg.composition_nodes
        assert node_id not in reg.tensor_nodes
        assert node_id not in reg.contracted_diagrams

    def test_add_node_container_contracted(self):
        """Test adding a contracted diagram container node."""
        reg = GateRegister()
        node_id = 15
        attrs = {"kind": "container", "container_type": "contracted"}

        reg.add_node(node_id, attrs)
        assert node_id in reg.contracted_diagrams
        assert node_id not in reg.tensor_nodes
        assert node_id not in reg.composition_nodes

    def test_add_node_ignores_other_kinds(self):
        """Test that nodes with non-proper/non-container kinds are ignored."""
        reg = GateRegister()
        node_id = 16
        attrs = {
            "kind": "wire",
            "type": "SqueezingGate",
        }

        reg.add_node(node_id, attrs)
        assert len(reg.squeezing_gates) == 0
        assert len(reg.displacement_gates) == 0
        assert len(reg.rotation_gates) == 0
        assert len(reg.fourier_gates) == 0
        assert len(reg.identity_spiders) == 0
        assert len(reg.input_states) == 0
        assert len(reg.measurement_nodes) == 0
        assert len(reg.contracted_diagrams) == 0
        assert len(reg.tensor_nodes) == 0
        assert len(reg.composition_nodes) == 0

    def test_remove_node(self):
        """Test removing a node from all sets."""
        reg = GateRegister()
        node_id = 17

        # Add node to multiple sets
        reg.squeezing_gates.add(node_id)
        reg.displacement_gates.add(node_id)
        reg.rotation_gates.add(node_id)
        reg.fourier_gates.add(node_id)
        reg.identity_spiders.add(node_id)
        reg.input_states.add(node_id)
        reg.measurement_nodes.add(node_id)
        reg.contracted_diagrams.add(node_id)
        reg.tensor_nodes.add(node_id)
        reg.composition_nodes.add(node_id)

        # Verify it's in all sets
        assert node_id in reg.squeezing_gates
        assert node_id in reg.displacement_gates
        assert node_id in reg.rotation_gates
        assert node_id in reg.fourier_gates
        assert node_id in reg.identity_spiders
        assert node_id in reg.input_states
        assert node_id in reg.measurement_nodes
        assert node_id in reg.contracted_diagrams
        assert node_id in reg.tensor_nodes
        assert node_id in reg.composition_nodes

        # Remove it
        reg.remove_node(node_id)

        # Verify it's removed from all sets
        assert node_id not in reg.squeezing_gates
        assert node_id not in reg.displacement_gates
        assert node_id not in reg.rotation_gates
        assert node_id not in reg.fourier_gates
        assert node_id not in reg.identity_spiders
        assert node_id not in reg.input_states
        assert node_id not in reg.measurement_nodes
        assert node_id not in reg.contracted_diagrams
        assert node_id not in reg.tensor_nodes
        assert node_id not in reg.composition_nodes

    def test_remove_node_idempotent(self):
        """Test that removing a node that doesn't exist is safe."""
        reg = GateRegister()
        node_id = 18

        # Should not raise an error
        reg.remove_node(node_id)

        # All sets should still be empty
        assert len(reg.squeezing_gates) == 0
        assert len(reg.displacement_gates) == 0
        assert len(reg.rotation_gates) == 0
        assert len(reg.fourier_gates) == 0
        assert len(reg.identity_spiders) == 0
        assert len(reg.input_states) == 0
        assert len(reg.measurement_nodes) == 0
        assert len(reg.contracted_diagrams) == 0
        assert len(reg.tensor_nodes) == 0
        assert len(reg.composition_nodes) == 0

    def test_copy(self):
        """Test creating a copy of the register."""
        reg = GateRegister()

        # Add some nodes
        reg.squeezing_gates.add(1)
        reg.displacement_gates.add(2)
        reg.rotation_gates.add(3)
        reg.fourier_gates.add(4)
        reg.identity_spiders.add(5)
        reg.input_states.add(6)
        reg.measurement_nodes.add(7)
        reg.contracted_diagrams.add(8)
        reg.tensor_nodes.add(9)
        reg.composition_nodes.add(10)

        # Create copy
        copy_reg = reg.copy()

        # Verify copies are independent
        assert copy_reg is not reg
        assert copy_reg.squeezing_gates == reg.squeezing_gates
        assert copy_reg.displacement_gates == reg.displacement_gates
        assert copy_reg.rotation_gates == reg.rotation_gates
        assert copy_reg.fourier_gates == reg.fourier_gates
        assert copy_reg.identity_spiders == reg.identity_spiders
        assert copy_reg.input_states == reg.input_states
        assert copy_reg.measurement_nodes == reg.measurement_nodes
        assert copy_reg.contracted_diagrams == reg.contracted_diagrams
        assert copy_reg.tensor_nodes == reg.tensor_nodes
        assert copy_reg.composition_nodes == reg.composition_nodes

        # Modify original and verify copy unchanged
        reg.squeezing_gates.add(100)
        assert 100 not in copy_reg.squeezing_gates

    def test_clear(self):
        """Test clearing the register."""
        reg = GateRegister()

        # Add some nodes
        reg.squeezing_gates.add(1)
        reg.displacement_gates.add(2)
        reg.rotation_gates.add(3)
        reg.fourier_gates.add(4)
        reg.identity_spiders.add(5)
        reg.input_states.add(6)
        reg.measurement_nodes.add(7)
        reg.contracted_diagrams.add(8)
        reg.tensor_nodes.add(9)
        reg.composition_nodes.add(10)

        # Clear all
        reg.clear()

        # Verify all sets are empty
        assert len(reg.squeezing_gates) == 0
        assert len(reg.displacement_gates) == 0
        assert len(reg.rotation_gates) == 0
        assert len(reg.fourier_gates) == 0
        assert len(reg.identity_spiders) == 0
        assert len(reg.input_states) == 0
        assert len(reg.measurement_nodes) == 0
        assert len(reg.contracted_diagrams) == 0
        assert len(reg.tensor_nodes) == 0
        assert len(reg.composition_nodes) == 0

    def test_build_from_graph(self):
        """Test building the registry from a graph."""
        reg = GateRegister()
        # Create a graph with various nodes
        graph = nx.DiGraph()

        # Add nodes with attributes
        graph.add_node(1, kind="compact", type="SqueezingGate", num_inputs=1, num_outputs=1)
        graph.add_node(2, kind="compact", type="DisplacementGate", num_inputs=1, num_outputs=1)
        graph.add_node(3, kind="compact", type="PhaseRotationGate", num_inputs=1, num_outputs=1)
        graph.add_node(4, kind="proper", type="Fourier", num_inputs=1, num_outputs=1)
        graph.add_node(5, kind="proper", type="QSpider", num_inputs=1, num_outputs=1, phase=ZxPoly({0: 0}))
        graph.add_node(6, kind="proper", num_inputs=0, num_outputs=1)
        graph.add_node(7, kind="proper", num_inputs=1, num_outputs=0)
        graph.add_node(8, kind="container", container_type="tensor")
        graph.add_node(9, kind="container", container_type="composition")
        graph.add_node(10, kind="container", container_type="contracted")

        # Rebuild registry
        reg.build_from_graph(graph)

        # Verify all nodes were added correctly
        assert 1 in reg.squeezing_gates
        assert 2 in reg.displacement_gates
        assert 3 in reg.rotation_gates
        assert 4 in reg.fourier_gates
        assert 5 in reg.identity_spiders
        assert 6 in reg.input_states
        assert 7 in reg.measurement_nodes
        assert 8 in reg.tensor_nodes
        assert 9 in reg.composition_nodes
        assert 10 in reg.contracted_diagrams

        # Verify sizes match
        assert len(reg.squeezing_gates) == 1
        assert len(reg.displacement_gates) == 1
        assert len(reg.rotation_gates) == 1
        assert len(reg.fourier_gates) == 1
        assert len(reg.identity_spiders) == 1
        assert len(reg.input_states) == 1
        assert len(reg.measurement_nodes) == 1
        assert len(reg.tensor_nodes) == 1
        assert len(reg.composition_nodes) == 1
        assert len(reg.contracted_diagrams) == 1

    def test_build_from_graph_clears_existing(self):
        """Test that build_from_graph clears existing entries."""
        reg = GateRegister()

        # Add some nodes manually
        reg.squeezing_gates.add(100)
        reg.displacement_gates.add(200)

        # Create a new graph
        graph = nx.DiGraph()
        graph.add_node(1, kind="compact", type="PhaseRotationGate", num_inputs=1, num_outputs=1)
        graph.add_node(2, kind="proper", type="Fourier2", num_inputs=1, num_outputs=1)

        # Rebuild
        reg.build_from_graph(graph)

        # Old entries should be gone
        assert 100 not in reg.squeezing_gates
        assert 200 not in reg.displacement_gates

        # New entries should be there
        assert 1 in reg.rotation_gates
        assert 2 in reg.fourier_gates

    def test_add_node_with_feedforward_displacement(self):
        """Test adding a displacement gate with feedforward information."""
        reg = GateRegister()
        node_id = 101
        attrs = {
            "kind": "compact",
            "type": "DisplacementGate",
            "num_inputs": 1,
            "num_outputs": 1,
            "feedforward": True,
            "measurement_id": 42,
        }

        reg.add_node(node_id, attrs)
        assert node_id in reg.displacement_gates
        # Note: feedforward info is stored in node attributes, not in the registry
        # The registry only tracks node IDs by type

    def test_identity_spider_detection_with_complex_phase(self):
        """Test that identity spider detection works with various zero phase representations."""
        reg = GateRegister()

        test_cases = [
            # Zero polynomial (empty dict)
            {"kind": "proper", "type": "QSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({})},
            # Zero constant
            {"kind": "proper", "type": "PSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({0: 0})},
            # Zero polynomial after simplification
            {"kind": "proper", "type": "QSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({1: 0, 2: 0})},
        ]

        for i, attrs in enumerate(test_cases):
            node_id = 1000 + i
            reg.add_node(node_id, attrs)
            assert node_id in reg.identity_spiders, f"Failed for test case {i}"

    def test_non_identity_spider_with_various_phases(self):
        """Test that various non-zero phases are not detected as identity spiders."""
        reg = GateRegister()

        test_cases = [
            {"kind": "proper", "type": "QSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({1: 2.0})},
            {"kind": "proper", "type": "PSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({2: 0.5})},
            {
                "kind": "proper",
                "type": "QSpider",
                "num_inputs": 1,
                "num_outputs": 1,
                "phase": ZxPoly({0: 1.0, 1: 2.0}),
            },
        ]

        for i, attrs in enumerate(test_cases):
            node_id = 2000 + i
            reg.add_node(node_id, attrs)
            assert node_id not in reg.identity_spiders, f"Failed for test case {i}"

    def test_input_state_with_additional_attributes(self):
        """Test that input state detection works with extra attributes."""
        reg = GateRegister()
        node_id = 300
        attrs = {
            "kind": "proper",
            "num_inputs": 0,
            "num_outputs": 1,
            "type": "InputState",  # Extra attribute
            "label": "|0>",
        }

        reg.add_node(node_id, attrs)
        assert node_id in reg.input_states

    def test_measurement_with_additional_attributes(self):
        """Test that measurement detection works with extra attributes."""
        reg = GateRegister()
        node_id = 301
        attrs = {
            "kind": "proper",
            "num_inputs": 1,
            "num_outputs": 0,
            "type": "Measurement",  # Extra attribute
            "basis": "homodyne",
        }

        reg.add_node(node_id, attrs)
        assert node_id in reg.measurement_nodes

    def test_performance_large_graph(self):
        """Test performance with a large graph."""
        reg = GateRegister()
        graph = nx.DiGraph()

        # Create a large graph with 1000 nodes
        kind = "proper"
        for i in range(1000):
            if i % 5 == 0:
                gate_type = "SqueezingGate"
                kind = "compact"
            elif i % 5 == 1:
                gate_type = "DisplacementGate"
                kind = "compact"
            elif i % 5 == 2:
                gate_type = "PhaseRotationGate"
                kind = "compact"
            elif i % 5 == 3:
                gate_type = "Fourier"
            else:
                gate_type = "QSpider"

            graph.add_node(
                i,
                kind=kind,
                type=gate_type,
                num_inputs=1,
                num_outputs=1,
                phase=ZxPoly({0: 0}) if gate_type == "QSpider" else None,
            )

        # Rebuild registry
        reg.build_from_graph(graph)

        # Check counts
        assert len(reg.squeezing_gates) == 200  # 1000 / 5
        assert len(reg.displacement_gates) == 200
        assert len(reg.rotation_gates) == 200
        assert len(reg.fourier_gates) == 200
        assert len(reg.identity_spiders) == 200  # All QSpider are identity
        assert len(reg.input_states) == 0
        assert len(reg.measurement_nodes) == 0
        assert len(reg.contracted_diagrams) == 0
        assert len(reg.tensor_nodes) == 0
        assert len(reg.composition_nodes) == 0

    def test_registry_lookup_usage(self):
        """Test that the registry can be used for O(1) lookups."""
        reg = GateRegister()

        # Add nodes
        reg.squeezing_gates.add(1)
        reg.squeezing_gates.add(2)
        reg.displacement_gates.add(3)

        # Test lookups
        assert 1 in reg.squeezing_gates
        assert 2 in reg.squeezing_gates
        assert 3 in reg.displacement_gates
        assert 4 not in reg.squeezing_gates
        assert 1 not in reg.displacement_gates

        # This would be O(1) instead of scanning all nodes

    @pytest.mark.parametrize(
        ("gate_type", "expected_set"),
        [
            ("SqueezingGate", "squeezing_gates"),
            ("DisplacementGate", "displacement_gates"),
            ("PhaseRotationGate", "rotation_gates"),
            ("Fourier", "fourier_gates"),
            ("FourierInv", "fourier_gates"),
            ("Fourier2", "fourier_gates"),
        ],
    )
    def test_add_node_parametrized(self, gate_type: str, expected_set: str):
        """Parametrized test for adding different gate types."""
        reg = GateRegister()
        node_id = 400
        attrs = {"kind": "proper", "type": gate_type}

        reg.add_node(node_id, attrs)

        # Get the set by name
        reg_set = getattr(reg, expected_set)
        assert node_id in reg_set

        # Verify it's not in other sets
        all_sets = [
            "squeezing_gates",
            "displacement_gates",
            "rotation_gates",
            "fourier_gates",
            "identity_spiders",
            "input_states",
            "measurement_nodes",
            "contracted_diagrams",
            "tensor_nodes",
            "composition_nodes",
        ]
        for set_name in all_sets:
            if set_name != expected_set:
                other_set = getattr(reg, set_name)
                assert node_id not in other_set

    @pytest.mark.parametrize(
        ("container_type", "expected_set"),
        [
            ("tensor", "tensor_nodes"),
            ("composition", "composition_nodes"),
            ("contracted", "contracted_diagrams"),
        ],
    )
    def test_add_container_parametrized(self, container_type: str, expected_set: str):
        """Parametrized test for adding different container types."""
        reg = GateRegister()
        node_id = 500
        attrs = {"kind": "container", "container_type": container_type}

        reg.add_node(node_id, attrs)

        reg_set = getattr(reg, expected_set)
        assert node_id in reg_set


# Integration test with actual gates
class TestGateRegisterIntegration:
    """Integration tests with actual gate classes."""

    def test_register_with_real_gates(self):
        """Test registering actual gate instances."""
        reg = GateRegister()

        # Create real gates
        sq = SqueezingGate(0.5)
        d = DisplacementGate(1.0 + 0.5j)
        r = PhaseRotationGate(pi / 4)
        cpg = CubicPhaseGate(0.1)

        # In a real implementation, these would be added to a graph
        # and the registry would be updated. This test verifies the
        # registry structure works with real gate types.

        # Simulate adding them to the registry
        sq_id = id(sq)
        d_id = id(d)
        r_id = id(r)
        cpg_id = id(cpg)

        reg.squeezing_gates.add(sq_id)
        reg.displacement_gates.add(d_id)
        reg.rotation_gates.add(r_id)
        # CubicPhaseGate is not in the registry (not tracked)

        assert sq_id in reg.squeezing_gates
        assert d_id in reg.displacement_gates
        assert r_id in reg.rotation_gates
        assert cpg_id not in reg.squeezing_gates
        assert cpg_id not in reg.displacement_gates
        assert cpg_id not in reg.rotation_gates

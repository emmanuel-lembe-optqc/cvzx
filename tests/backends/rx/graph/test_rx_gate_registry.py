"""Tests for the GateRegister class using rustworkx.

This module contains comprehensive unit tests for the GateRegister class,
which tracks specific gate types and nodes in a CV ZX rustworkx graph.
"""

from math import pi

import pytest
import rustworkx as rx
from sympy import symbols

from cvzx.ir.base import ZxPoly
from cvzx.ir.gates import CubicPhaseGate, DisplacementGate, PhaseRotationGate, SqueezingGate
from cvzx.backends.rx.graph import GateRegister


class TestGateRegister:
    """Test suite for GateRegister class with rustworkx."""

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

        reg.remove_node(node_id)

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

        copy_reg = reg.copy()

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

        reg.squeezing_gates.add(100)
        assert 100 not in copy_reg.squeezing_gates

    def test_clear(self):
        """Test clearing the register."""
        reg = GateRegister()

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

        reg.clear()

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
        """Test building the registry from a rustworkx PyDiGraph."""
        reg = GateRegister()
        graph = rx.PyDiGraph()

        graph.add_node({"id": 1, "kind": "compact", "type": "SqueezingGate", "num_inputs": 1, "num_outputs": 1})
        graph.add_node({"id": 2, "kind": "compact", "type": "DisplacementGate", "num_inputs": 1, "num_outputs": 1})
        graph.add_node({"id": 3, "kind": "compact", "type": "PhaseRotationGate", "num_inputs": 1, "num_outputs": 1})
        graph.add_node({"id": 4, "kind": "proper", "type": "Fourier", "num_inputs": 1, "num_outputs": 1})
        graph.add_node({
            "id": 5,
            "kind": "proper",
            "type": "QSpider",
            "num_inputs": 1,
            "num_outputs": 1,
            "phase": ZxPoly({0: 0}),
        })
        graph.add_node({"id": 6, "kind": "proper", "num_inputs": 0, "num_outputs": 1})
        graph.add_node({"id": 7, "kind": "proper", "num_inputs": 1, "num_outputs": 0})
        graph.add_node({"id": 8, "kind": "container", "container_type": "tensor"})
        graph.add_node({"id": 9, "kind": "container", "container_type": "composition"})
        graph.add_node({"id": 10, "kind": "container", "container_type": "contracted"})

        reg.build_from_graph(graph)

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

        reg.squeezing_gates.add(100)
        reg.displacement_gates.add(200)

        graph = rx.PyDiGraph()
        graph.add_node({"id": 1, "kind": "compact", "type": "PhaseRotationGate", "num_inputs": 1, "num_outputs": 1})
        graph.add_node({"id": 2, "kind": "proper", "type": "Fourier2", "num_inputs": 1, "num_outputs": 1})

        reg.build_from_graph(graph)

        assert 100 not in reg.squeezing_gates
        assert 200 not in reg.displacement_gates

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

    def test_identity_spider_detection_with_complex_phase(self):
        """Test that identity spider detection works with various zero phase representations."""
        reg = GateRegister()

        test_cases = [
            {"kind": "proper", "type": "QSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({})},
            {"kind": "proper", "type": "PSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({0: 0})},
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
            "type": "InputState",
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
            "type": "Measurement",
            "basis": "homodyne",
        }

        reg.add_node(node_id, attrs)
        assert node_id in reg.measurement_nodes

    def test_performance_large_graph(self):
        """Test performance with a large rustworkx graph."""
        reg = GateRegister()
        graph = rx.PyDiGraph()

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

            graph.add_node({
                "id": i,
                "kind": kind,
                "type": gate_type,
                "num_inputs": 1,
                "num_outputs": 1,
                "phase": ZxPoly({0: 0}) if gate_type == "QSpider" else None,
            })

        reg.build_from_graph(graph)

        assert len(reg.squeezing_gates) == 200
        assert len(reg.displacement_gates) == 200
        assert len(reg.rotation_gates) == 200
        assert len(reg.fourier_gates) == 200
        assert len(reg.identity_spiders) == 200
        assert len(reg.input_states) == 0
        assert len(reg.measurement_nodes) == 0
        assert len(reg.contracted_diagrams) == 0
        assert len(reg.tensor_nodes) == 0
        assert len(reg.composition_nodes) == 0

    def test_registry_lookup_usage(self):
        """Test that the registry can be used for O(1) lookups."""
        reg = GateRegister()

        reg.squeezing_gates.add(1)
        reg.squeezing_gates.add(2)
        reg.displacement_gates.add(3)

        assert 1 in reg.squeezing_gates
        assert 2 in reg.squeezing_gates
        assert 3 in reg.displacement_gates
        assert 4 not in reg.squeezing_gates
        assert 1 not in reg.displacement_gates

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

        reg_set = getattr(reg, expected_set)
        assert node_id in reg_set

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


class TestGateRegisterIntegration:
    """Integration tests with actual gate classes."""

    def test_register_with_real_gates(self):
        """Test registering actual gate instances."""
        reg = GateRegister()

        sq = SqueezingGate(0.5)
        d = DisplacementGate(1.0 + 0.5j)
        r = PhaseRotationGate(pi / 4)
        cpg = CubicPhaseGate(0.1)

        sq_id = id(sq)
        d_id = id(d)
        r_id = id(r)
        cpg_id = id(cpg)

        reg.squeezing_gates.add(sq_id)
        reg.displacement_gates.add(d_id)
        reg.rotation_gates.add(r_id)

        assert sq_id in reg.squeezing_gates
        assert d_id in reg.displacement_gates
        assert r_id in reg.rotation_gates
        assert cpg_id not in reg.squeezing_gates
        assert cpg_id not in reg.displacement_gates
        assert cpg_id not in reg.rotation_gates


class TestGateRegisterParameterTracking:
    """Test suite for the parametric/feedforward-provenance registry fields.

    Covers `parametric_nodes`, `feedforward_nodes`, `symbol_registry`, and
    `measurement_to_feedforward_map`, added alongside `param_measurement_map`.
    Mirrors `tests/nx_graph/test_gate_registry.py::TestGateRegisterParameterTracking`.
    """

    def test_symbol_from_zxpoly_phase_excludes_generator(self):
        """A symbol in a ZxPoly phase is indexed; the polynomial's own generator is not.

        This is the highest-value regression test here: `sympy.Poly.free_symbols`
        incorrectly includes the polynomial's own generator variable, which is
        why `_symbols_in`/`_collect_symbols` deliberately use `.coeffs` instead.
        """
        reg = GateRegister()
        m = symbols("m")
        generator = symbols("x", real=True)  # ZxPoly's own generator variable.
        node_id = 5000
        attrs = {
            "kind": "proper",
            "type": "QSpider",
            "num_inputs": 1,
            "num_outputs": 1,
            "phase": ZxPoly({1: m}),
        }

        reg.add_node(node_id, attrs)

        assert node_id in reg.parametric_nodes
        assert reg.symbol_registry == {m: {node_id}}
        assert generator not in reg.symbol_registry

    def test_symbol_from_multi_parameter_gate(self):
        """All three symbols of a multi-parameter gate (e.g. ArbitraryGate) are indexed."""
        reg = GateRegister()
        alpha, beta, lam = symbols("alpha beta lam")
        node_id = 5001
        attrs = {
            "kind": "compact",
            "type": "ArbitraryGate",
            "num_inputs": 1,
            "num_outputs": 1,
            "alpha": alpha,
            "beta": beta,
            "lam": lam,
        }

        reg.add_node(node_id, attrs)

        assert node_id in reg.parametric_nodes
        assert reg.symbol_registry[alpha] == {node_id}
        assert reg.symbol_registry[beta] == {node_id}
        assert reg.symbol_registry[lam] == {node_id}

    def test_last_user_removal_deletes_symbol_key(self):
        """When a symbol's last user node is removed, its key is deleted, not left empty."""
        reg = GateRegister()
        m = symbols("m")
        attrs = {"kind": "proper", "type": "QSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({1: m})}
        reg.add_node(5010, attrs)

        reg.remove_node(5010)

        assert m not in reg.symbol_registry
        assert 5010 not in reg.parametric_nodes

    def test_symbol_key_survives_while_another_user_remains(self):
        """Removing one of two nodes sharing a symbol must keep the key, minus that node."""
        reg = GateRegister()
        m = symbols("m")
        attrs = {"kind": "proper", "type": "QSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({1: m})}
        reg.add_node(5020, attrs)
        reg.add_node(5021, attrs)

        reg.remove_node(5020)

        assert reg.symbol_registry[m] == {5021}

    def test_readd_refreshes_not_duplicates(self):
        """Re-adding a node with the same symbol must refresh, not duplicate, the entry."""
        reg = GateRegister()
        m = symbols("m")
        attrs = {"kind": "proper", "type": "QSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({1: m})}
        reg.add_node(5030, attrs)
        reg.add_node(5030, attrs)

        assert reg.symbol_registry[m] == {5030}
        assert len(reg._indexed_symbols[5030]) == 1  # ruff: ignore[private-member-access]

    def test_feedforward_nodes_and_measurement_to_feedforward_map(self):
        """A feedforward node is indexed by measurement_to_feedforward_map."""
        reg = GateRegister()
        node_id = 5040
        attrs = {
            "kind": "compact",
            "type": "DisplacementGate",
            "num_inputs": 1,
            "num_outputs": 1,
            "feedforward": True,
            "measurement_ids": {42, 43},
        }

        reg.add_node(node_id, attrs)

        assert node_id in reg.feedforward_nodes
        assert reg.measurement_to_feedforward_map[42] == {node_id}
        assert reg.measurement_to_feedforward_map[43] == {node_id}

    def test_measurement_key_deleted_when_last_feedforward_node_removed(self):
        """Removing the last feedforward node referencing a measurement id deletes that key."""
        reg = GateRegister()
        attrs = {
            "kind": "compact",
            "type": "DisplacementGate",
            "num_inputs": 1,
            "num_outputs": 1,
            "feedforward": True,
            "measurement_ids": {99},
        }
        reg.add_node(5050, attrs)

        reg.remove_node(5050)

        assert 99 not in reg.measurement_to_feedforward_map
        assert 5050 not in reg.feedforward_nodes

    def test_clear_resets_all_new_fields_and_void_nodes(self):
        """clear() must reset parametric/feedforward/symbol fields, and void_nodes."""
        reg = GateRegister()
        m = symbols("m")
        reg.add_node(
            5060,
            {"kind": "proper", "type": "QSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({1: m})},
        )
        reg.add_node(
            5061,
            {
                "kind": "compact",
                "type": "DisplacementGate",
                "num_inputs": 1,
                "num_outputs": 1,
                "feedforward": True,
                "measurement_ids": {7},
            },
        )
        reg.add_node(5062, {"kind": "proper", "type": "VoidDiagram", "num_inputs": 1, "num_outputs": 1})
        assert 5062 in reg.void_nodes

        reg.clear()

        assert reg.parametric_nodes == set()
        assert reg.feedforward_nodes == set()
        assert reg.symbol_registry == {}
        assert reg.measurement_to_feedforward_map == {}
        assert reg.void_nodes == set()
        assert reg._indexed_symbols == {}  # ruff: ignore[private-member-access]
        assert reg._indexed_measurements == {}  # ruff: ignore[private-member-access]

    def test_copy_round_trips_new_fields_and_does_not_alias(self):
        """copy() must reproduce the new fields and not alias their inner sets/dicts."""
        reg = GateRegister()
        m = symbols("m")
        reg.add_node(
            5070,
            {"kind": "proper", "type": "QSpider", "num_inputs": 1, "num_outputs": 1, "phase": ZxPoly({1: m})},
        )
        reg.add_node(
            5071,
            {
                "kind": "compact",
                "type": "DisplacementGate",
                "num_inputs": 1,
                "num_outputs": 1,
                "feedforward": True,
                "measurement_ids": {7},
            },
        )

        copy_reg = reg.copy()

        assert copy_reg.parametric_nodes == reg.parametric_nodes
        assert copy_reg.feedforward_nodes == reg.feedforward_nodes
        assert copy_reg.symbol_registry == reg.symbol_registry
        assert copy_reg.measurement_to_feedforward_map == reg.measurement_to_feedforward_map

        # Mutate the copy's inner sets; the original must be unaffected.
        copy_reg.symbol_registry[m].add(9999)
        copy_reg.measurement_to_feedforward_map[7].add(9999)
        copy_reg.parametric_nodes.add(9999)
        copy_reg.feedforward_nodes.add(9999)

        assert 9999 not in reg.symbol_registry[m]
        assert 9999 not in reg.measurement_to_feedforward_map[7]
        assert 9999 not in reg.parametric_nodes
        assert 9999 not in reg.feedforward_nodes

    def test_build_from_graph_round_trips_new_fields(self):
        """build_from_graph() must populate the new fields exactly like add_node() does."""
        reg = GateRegister()
        graph = rx.PyDiGraph()
        m = symbols("m")
        graph.add_node({
            "id": 1,
            "kind": "proper",
            "type": "QSpider",
            "num_inputs": 1,
            "num_outputs": 1,
            "phase": ZxPoly({1: m}),
        })
        graph.add_node({
            "id": 2,
            "kind": "compact",
            "type": "DisplacementGate",
            "num_inputs": 1,
            "num_outputs": 1,
            "feedforward": True,
            "measurement_ids": {7},
        })

        reg.build_from_graph(graph)

        assert 1 in reg.parametric_nodes
        assert reg.symbol_registry[m] == {1}
        assert 2 in reg.feedforward_nodes
        assert reg.measurement_to_feedforward_map[7] == {2}

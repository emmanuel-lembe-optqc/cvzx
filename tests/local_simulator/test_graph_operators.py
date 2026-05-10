"""
Test suite for all graph operations defined in ops.py.

This test suite validates:
1. Wiring (swap and through)
2. Measurement (various theta angles, readout, feedforward displacements)
3. Initialization (creating modes, different theta angles)
4. PhaseRotation (with and without swap, displacements)
5. ShearXInvariant (kappa parameter)
6. ShearPInvariant (eta parameter)
7. Squeezing (theta parameter, validation)
8. Squeezing45 (45-degree squeezing)
9. ArbitraryFirst and ArbitrarySecond (alpha, beta, lam parameters)
10. ControlledZ (g parameter)
11. BeamSplitter (sqrt_r and theta_rel parameters)
12. TwoModeShear (a and b parameters)
13. Manual (theta_a, theta_b, theta_c, theta_d parameters)

Run with: pytest test_graph_operations.py -v
"""

from math import pi

import pytest

from mqc3.client._local_simulator import (
    local_run_graph,
)
from mqc3.feedforward import feedforward
from mqc3.graph import GraphRepr
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import (
    ArbitraryFirst,
    ArbitrarySecond,
    BeamSplitter,
    ControlledZ,
    Initialization,
    Manual,
    Measurement,
    PhaseRotation,
    ShearPInvariant,
    ShearXInvariant,
    Squeezing,
    Squeezing45,
    TwoModeShear,
    Wiring,
)

# =============================================================================
# Helper function to create a simple graph with a mode
# =============================================================================


def create_single_mode_graph() -> GraphRepr:
    """Create a 1x3 graph for single-mode operation tests."""
    graph = GraphRepr(n_local_macronodes=1, n_steps=3)
    # Initialize mode 0 at step 0
    init = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
    graph.place_operation(init)
    return graph


def add_measurement(graph: GraphRepr, mode: int = 0, theta: float = 0.0) -> None:
    """Add a measurement at the last step."""
    last_step = graph.n_steps - 1
    meas = Measurement((0, last_step), theta=theta, readout=True)
    graph.place_operation(meas)


# =============================================================================
# Test 1: Wiring (Through and Swap)
# =============================================================================


class TestWiring:
    """Tests for Wiring operation."""

    def test_wiring_through(self):
        """Test through wiring (swap=False)."""
        print("\n=== Test: Wiring Through ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)

        wire0 = Wiring((0, 1), swap=False)
        wire1 = Wiring((1, 1), swap=False)
        graph.place_operation(wire0)
        graph.place_operation(wire1)

        meas0 = Measurement((0, 1), theta=0.0, readout=True)
        meas1 = Measurement((1, 1), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=100, state_save_policy="none")
        assert len(result.graph_result) == 100
        print("✓ Through wiring works")

    def test_wiring_swap(self):
        """Test swap wiring (swap=True)."""
        print("\n=== Test: Wiring Swap ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)

        swap = Wiring((0, 1), swap=True)
        graph.place_operation(swap)

        meas0 = Measurement((0, 1), theta=0.0, readout=True)
        meas1 = Measurement((1, 1), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=100, state_save_policy="none")
        assert len(result.graph_result) == 100
        print("✓ Swap wiring works")


# =============================================================================
# Test 2: Measurement
# =============================================================================


class TestMeasurement:
    """Tests for Measurement operation."""

    def test_measurement_basic(self):
        """Test basic measurement."""
        print("\n=== Test: Measurement Basic ===")
        graph = create_single_mode_graph()
        meas = Measurement((0, 1), theta=0.0, readout=True)
        graph.place_operation(meas)

        result = local_run_graph(graph, n_shots=100, state_save_policy="first_only")
        assert len(result.graph_result) == 100
        print("✓ Basic measurement works")

    def test_measurement_various_angles(self):
        """Test measurement with various theta angles."""
        print("\n=== Test: Measurement Various Angles ===")
        angles = [0, pi / 4, pi / 2, 3 * pi / 4, pi]

        for theta in angles:
            graph = create_single_mode_graph()
            meas = Measurement((0, 1), theta=theta, readout=True)
            graph.place_operation(meas)

            result = local_run_graph(graph, n_shots=50, state_save_policy="none")
            assert len(result.graph_result) == 50
            print(f"  ✓ Measurement with theta={theta:.2f} works")

    def test_measurement_with_feedforward(self):
        """Test measurement with feedforward displacements."""
        print("\n=== Test: Measurement with Feedforward ===")

        @feedforward
        def ff(x):
            return 2 * x

        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        graph.place_operation(init0)

        meas1 = Measurement((0, 1), theta=0.0, readout=True)
        graph.place_operation(meas1)

        x0 = graph.get_mode_measured_value(mode=0)

        meas2 = Measurement((1, 1), theta=pi / 2, displacement_k_minus_n=(ff(x0), 0.0), readout=True)
        graph.place_operation(meas2)

        result = local_run_graph(graph, n_shots=100, state_save_policy="none")
        assert len(result.graph_result) == 100
        print("✓ Measurement with feedforward works")

    def test_measurement_displacements(self):
        """Test measurement with displacement parameters."""
        print("\n=== Test: Measurement with Displacements ===")
        graph = GraphRepr(n_local_macronodes=1, n_steps=2)

        init = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        graph.place_operation(init)

        # Measurement with displacement
        meas = Measurement((0, 1), theta=0.0, displacement_k_minus_n=(0.5, 0.0), readout=True)
        graph.place_operation(meas)

        result = local_run_graph(graph, n_shots=100, state_save_policy="first_only")
        assert len(result.graph_result) == 100
        print("✓ Measurement with displacement works")


# =============================================================================
# Test 3: Initialization
# =============================================================================


class TestInitialization:
    """Tests for Initialization operation."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        print("\n=== Test: Initialization Basic ===")
        graph = GraphRepr(n_local_macronodes=1, n_steps=2)

        init = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        graph.place_operation(init)

        # Add a measurement to make the graph valid
        meas = Measurement((0, 1), theta=0.0, readout=True)
        graph.place_operation(meas)

        result = local_run_graph(graph, n_shots=50, state_save_policy="first_only")
        assert len(result.graph_result) == 50
        print("✓ Basic initialization works")

    def test_initialization_various_theta(self):
        """Test initialization with various theta angles."""
        print("\n=== Test: Initialization Various Theta ===")
        thetas = [0, pi / 4, pi / 2, 3 * pi / 4, pi]

        for theta in thetas:
            graph = GraphRepr(n_local_macronodes=1, n_steps=2)
            init = Initialization((0, 0), theta=theta, initialized_modes=(0, BLANK_MODE))
            graph.place_operation(init)
            meas = Measurement((0, 1), theta=0.0, readout=True)
            graph.place_operation(meas)

            result = local_run_graph(graph, n_shots=30, state_save_policy="none")
            assert len(result.graph_result) == 30
            print(f"  ✓ Initialization with theta={theta:.2f} works")

    def test_initialization_two_modes(self):
        """Test initialization of two modes."""
        print("\n=== Test: Initialization Two Modes ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, 1))
        graph.place_operation(init)

        meas0 = Measurement((0, 1), theta=0.0, readout=True)
        meas1 = Measurement((1, 1), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ Two-mode initialization works")


# =============================================================================
# Test 4: PhaseRotation
# =============================================================================


class TestPhaseRotation:
    """Tests for PhaseRotation operation."""

    def test_phase_rotation_basic(self):
        """Test basic phase rotation."""
        print("\n=== Test: PhaseRotation Basic ===")
        graph = create_single_mode_graph()
        phase = PhaseRotation((0, 1), phi=pi / 4, swap=False)
        graph.place_operation(phase)
        add_measurement(graph)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ Basic phase rotation works")

    def test_phase_rotation_with_swap(self):
        """Test phase rotation with swap."""
        print("\n=== Test: PhaseRotation with Swap ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)

        phase = PhaseRotation((0, 1), phi=pi / 4, swap=True)
        graph.place_operation(phase)

        meas0 = Measurement((0, 1), theta=0.0, readout=True)
        meas1 = Measurement((1, 1), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ Phase rotation with swap works")

    def test_phase_rotation_with_displacements(self):
        """Test phase rotation with displacement parameters."""
        print("\n=== Test: PhaseRotation with Displacements ===")
        graph = create_single_mode_graph()
        phase = PhaseRotation(
            (0, 1), phi=pi / 4, swap=False, displacement_k_minus_n=(0.5, 0.0), displacement_k_minus_1=(0.0, 0.5)
        )
        graph.place_operation(phase)
        add_measurement(graph)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ Phase rotation with displacements works")


# =============================================================================
# Test 5: Shear Gates
# =============================================================================


class TestShearGates:
    """Tests for ShearXInvariant and ShearPInvariant operations."""

    def test_shear_x_invariant(self):
        """Test X-invariant shear gate."""
        print("\n=== Test: ShearXInvariant ===")
        graph = create_single_mode_graph()
        shear = ShearXInvariant((0, 1), kappa=0.5, swap=False)
        graph.place_operation(shear)
        add_measurement(graph)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ ShearXInvariant works")

    def test_shear_p_invariant(self):
        """Test P-invariant shear gate."""
        print("\n=== Test: ShearPInvariant ===")
        graph = create_single_mode_graph()
        shear = ShearPInvariant((0, 1), eta=0.5, swap=False)
        graph.place_operation(shear)
        add_measurement(graph)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ ShearPInvariant works")

    def test_shear_with_swap(self):
        """Test shear gates with swap."""
        print("\n=== Test: Shear Gates with Swap ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)

        shear = ShearXInvariant((0, 1), kappa=0.5, swap=True)
        graph.place_operation(shear)

        meas0 = Measurement((0, 1), theta=0.0, readout=True)
        meas1 = Measurement((1, 1), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ Shear gate with swap works")


# =============================================================================
# Test 6: Squeezing Gates
# =============================================================================


class TestSqueezingGates:
    """Tests for Squeezing and Squeezing45 operations."""

    def test_squeezing_basic(self):
        """Test basic squeezing gate."""
        print("\n=== Test: Squeezing Basic ===")
        graph = create_single_mode_graph()
        squeeze = Squeezing((0, 1), theta=0.5, swap=False)
        graph.place_operation(squeeze)
        add_measurement(graph)

        result = local_run_graph(graph, n_shots=50, state_save_policy="first_only")
        assert len(result.graph_result) == 50
        print("✓ Basic squeezing works")

    def test_squeezing_invalid_theta(self):
        """Test that invalid theta raises error."""
        print("\n=== Test: Squeezing Invalid Theta ===")
        graph = create_single_mode_graph()

        with pytest.raises(ValueError):
            squeeze = Squeezing((0, 1), theta=0.0, swap=False)  # theta=0 is invalid
            graph.place_operation(squeeze)
        print("✓ Invalid theta correctly rejected")

    def test_squeezing45_basic(self):
        """Test 45-degree squeezing gate."""
        print("\n=== Test: Squeezing45 Basic ===")
        graph = create_single_mode_graph()
        squeeze45 = Squeezing45((0, 1), theta=0.5, swap=False)
        graph.place_operation(squeeze45)
        add_measurement(graph)

        result = local_run_graph(graph, n_shots=50, state_save_policy="first_only")
        assert len(result.graph_result) == 50
        print("✓ Squeezing45 works")

    def test_squeezing_with_swap(self):
        """Test squeezing with swap."""
        print("\n=== Test: Squeezing with Swap ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)

        squeeze = Squeezing((0, 1), theta=0.5, swap=True)
        graph.place_operation(squeeze)

        meas0 = Measurement((0, 1), theta=0.0, readout=True)
        meas1 = Measurement((1, 1), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ Squeezing with swap works")


# =============================================================================
# Test 7: Arbitrary Gates
# =============================================================================


class TestArbitraryGates:
    """Tests for ArbitraryFirst and ArbitrarySecond operations."""

    def test_arbitrary_first_basic(self):
        """Test ArbitraryFirst gate."""
        print("\n=== Test: ArbitraryFirst Basic ===")
        graph = create_single_mode_graph()
        arb = ArbitraryFirst((0, 1), alpha=0.2, beta=0.3, lam=0.4, swap=False)
        graph.place_operation(arb)
        add_measurement(graph)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ ArbitraryFirst works")

    def test_arbitrary_second_basic(self):
        """Test ArbitrarySecond gate."""
        print("\n=== Test: ArbitrarySecond Basic ===")
        graph = create_single_mode_graph()
        arb = ArbitrarySecond((0, 1), alpha=0.2, beta=0.3, lam=0.4, swap=False)
        graph.place_operation(arb)
        add_measurement(graph)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ ArbitrarySecond works")

    def test_arbitrary_with_swap(self):
        """Test arbitrary gates with swap."""
        print("\n=== Test: Arbitrary Gates with Swap ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)

        arb = ArbitraryFirst((0, 1), alpha=0.2, beta=0.3, lam=0.4, swap=True)
        graph.place_operation(arb)

        meas0 = Measurement((0, 1), theta=0.0, readout=True)
        meas1 = Measurement((1, 1), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ Arbitrary gate with swap works")


# =============================================================================
# Test 8: ControlledZ
# =============================================================================


class TestControlledZ:
    """Tests for ControlledZ operation."""

    def test_controlled_z_basic(self):
        """Test basic ControlledZ gate."""
        print("\n=== Test: ControlledZ Basic ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)

        cz = ControlledZ((0, 1), g=1.0, swap=False)
        graph.place_operation(cz)

        meas0 = Measurement((0, 1), theta=0.0, readout=True)
        meas1 = Measurement((1, 1), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ ControlledZ works")

    def test_controlled_z_various_g(self):
        """Test ControlledZ with various g parameters."""
        print("\n=== Test: ControlledZ Various g ===")
        g_values = [0.5, 1.0, 1.5, 2.0]

        for g in g_values:
            graph = GraphRepr(n_local_macronodes=2, n_steps=2)

            init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
            init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
            graph.place_operation(init0)
            graph.place_operation(init1)

            cz = ControlledZ((0, 1), g=g, swap=False)
            graph.place_operation(cz)

            meas0 = Measurement((0, 1), theta=0.0, readout=True)
            meas1 = Measurement((1, 1), theta=0.0, readout=True)
            graph.place_operation(meas0)
            graph.place_operation(meas1)

            result = local_run_graph(graph, n_shots=30, state_save_policy="none")
            assert len(result.graph_result) == 30
            print(f"  ✓ ControlledZ with g={g} works")


# =============================================================================
# Test 9: BeamSplitter
# =============================================================================


class TestBeamSplitter:
    """Tests for BeamSplitter operation."""

    def test_beam_splitter_basic(self):
        """Test basic BeamSplitter."""
        print("\n=== Test: BeamSplitter Basic ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)

        bs = BeamSplitter((0, 1), sqrt_r=0.5, theta_rel=0.1, swap=False)
        graph.place_operation(bs)

        meas0 = Measurement((0, 1), theta=0.0, readout=True)
        meas1 = Measurement((1, 1), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ BeamSplitter works")

    def test_beam_splitter_invalid_sqrt_r(self):
        """Test that invalid sqrt_r raises error."""
        print("\n=== Test: BeamSplitter Invalid sqrt_r ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        with pytest.raises(ValueError):
            bs = BeamSplitter((0, 1), sqrt_r=1.5, theta_rel=0.1, swap=False)  # sqrt_r > 1
            graph.place_operation(bs)
        print("✓ Invalid sqrt_r correctly rejected")

    def test_beam_splitter_various_parameters(self):
        """Test BeamSplitter with various parameters."""
        print("\n=== Test: BeamSplitter Various Parameters ===")
        params = [
            (0.3, 0.1),
            (0.5, 0.5),
            (0.7, 0.8),
            (0.9, 1.2),
        ]

        for sqrt_r, theta_rel in params:
            graph = GraphRepr(n_local_macronodes=2, n_steps=2)

            init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
            init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
            graph.place_operation(init0)
            graph.place_operation(init1)

            bs = BeamSplitter((0, 1), sqrt_r=sqrt_r, theta_rel=theta_rel, swap=False)
            graph.place_operation(bs)

            meas0 = Measurement((0, 1), theta=0.0, readout=True)
            meas1 = Measurement((1, 1), theta=0.0, readout=True)
            graph.place_operation(meas0)
            graph.place_operation(meas1)

            result = local_run_graph(graph, n_shots=30, state_save_policy="none")
            assert len(result.graph_result) == 30
            print(f"  ✓ BeamSplitter with sqrt_r={sqrt_r}, theta_rel={theta_rel} works")


# =============================================================================
# Test 10: TwoModeShear
# =============================================================================


class TestTwoModeShear:
    """Tests for TwoModeShear operation."""

    def test_two_mode_shear_basic(self):
        """Test basic TwoModeShear."""
        print("\n=== Test: TwoModeShear Basic ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)

        shear = TwoModeShear((0, 1), a=0.5, b=0.3, swap=False)
        graph.place_operation(shear)

        meas0 = Measurement((0, 1), theta=0.0, readout=True)
        meas1 = Measurement((1, 1), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ TwoModeShear works")

    def test_two_mode_shear_various_parameters(self):
        """Test TwoModeShear with various a and b parameters."""
        print("\n=== Test: TwoModeShear Various Parameters ===")
        params = [
            (0.2, 0.1),
            (0.5, 0.5),
            (0.8, 0.3),
            (1.0, 0.0),
        ]

        for a, b in params:
            graph = GraphRepr(n_local_macronodes=2, n_steps=2)

            init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
            init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
            graph.place_operation(init0)
            graph.place_operation(init1)

            shear = TwoModeShear((0, 1), a=a, b=b, swap=False)
            graph.place_operation(shear)

            meas0 = Measurement((0, 1), theta=0.0, readout=True)
            meas1 = Measurement((1, 1), theta=0.0, readout=True)
            graph.place_operation(meas0)
            graph.place_operation(meas1)

            result = local_run_graph(graph, n_shots=30, state_save_policy="none")
            assert len(result.graph_result) == 30
            print(f"  ✓ TwoModeShear with a={a}, b={b} works")


# =============================================================================
# Test 11: Manual Operation
# =============================================================================


class TestManual:
    """Tests for Manual operation."""

    def test_manual_basic(self):
        """Test basic Manual operation."""
        print("\n=== Test: Manual Basic ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)

        manual = Manual((0, 1), theta_a=0.0, theta_b=pi / 2, theta_c=pi / 4, theta_d=3 * pi / 4, swap=False)
        graph.place_operation(manual)

        meas0 = Measurement((0, 1), theta=0.0, readout=True)
        meas1 = Measurement((1, 1), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ Manual operation works")

    def test_manual_invalid_angles(self):
        """Test that invalid angle combinations raise error."""
        print("\n=== Test: Manual Invalid Angles ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=2)

        # theta_a == theta_b modulo pi should raise error
        with pytest.raises(ValueError):
            manual = Manual((0, 1), theta_a=0.0, theta_b=pi, theta_c=pi / 4, theta_d=3 * pi / 4, swap=False)
            graph.place_operation(manual)
        print("✓ Invalid angle combination correctly rejected")


# =============================================================================
# Test 12: Operation Combinations
# =============================================================================


class TestOperationCombinations:
    """Tests for combinations of multiple operation types."""

    def test_squeeze_and_phase(self):
        """Test combination of Squeezing and PhaseRotation."""
        print("\n=== Test: Squeeze and Phase Combination ===")
        graph = create_single_mode_graph()
        squeeze = Squeezing((0, 1), theta=0.5, swap=False)
        graph.place_operation(squeeze)
        phase = PhaseRotation((0, 2), phi=pi / 4, swap=False)
        graph.place_operation(phase)
        add_measurement(graph, theta=0.0)

        result = local_run_graph(graph, n_shots=50, state_save_policy="first_only")
        assert len(result.graph_result) == 50
        print("✓ Squeeze and Phase combination works")

    def test_cz_and_bs(self):
        """Test combination of ControlledZ and BeamSplitter."""
        print("\n=== Test: CZ and BeamSplitter Combination ===")
        graph = GraphRepr(n_local_macronodes=2, n_steps=3)

        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)

        cz = ControlledZ((0, 1), g=1.0, swap=False)
        graph.place_operation(cz)

        bs = BeamSplitter((0, 2), sqrt_r=0.5, theta_rel=0.2, swap=False)
        graph.place_operation(bs)

        meas0 = Measurement((0, 2), theta=0.0, readout=True)
        meas1 = Measurement((1, 2), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas1)

        result = local_run_graph(graph, n_shots=50, state_save_policy="none")
        assert len(result.graph_result) == 50
        print("✓ CZ and BeamSplitter combination works")

    def test_full_teleportation_pattern(self):
        """Test a simplified teleportation pattern with multiple operations."""
        print("\n=== Test: Simplified Teleportation Pattern ===")
        graph = GraphRepr(n_local_macronodes=3, n_steps=4)

        # Initialize modes
        init0 = Initialization((0, 0), theta=pi / 2, initialized_modes=(0, BLANK_MODE))
        init1 = Initialization((1, 0), theta=pi / 2, initialized_modes=(1, BLANK_MODE))
        init2 = Initialization((2, 0), theta=pi / 2, initialized_modes=(2, BLANK_MODE))
        graph.place_operation(init0)
        graph.place_operation(init1)
        graph.place_operation(init2)

        # Entangling operations
        cz01 = ControlledZ((0, 1), g=1.0, swap=False)
        cz12 = ControlledZ((1, 2), g=1.0, swap=False)
        graph.place_operation(cz01)
        graph.place_operation(cz12)

        # Phase rotations
        phase0 = PhaseRotation((0, 2), phi=pi / 4, swap=False)
        phase2 = PhaseRotation((2, 2), phi=pi / 4, swap=False)
        graph.place_operation(phase0)
        graph.place_operation(phase2)

        # Measurements
        meas0 = Measurement((0, 3), theta=0.0, readout=True)
        meas2 = Measurement((2, 3), theta=0.0, readout=True)
        graph.place_operation(meas0)
        graph.place_operation(meas2)

        result = local_run_graph(graph, n_shots=100, state_save_policy="none")
        assert len(result.graph_result) == 100
        print("✓ Full teleportation pattern works")

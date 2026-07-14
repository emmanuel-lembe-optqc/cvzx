"""Test visualization for CV quantum gates using CompactDiagram formalism.

This module provides test functions to visualize all gate types and verify
that the compact representation works correctly.

Run this script directly to generate test images.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    ZxPoly,
)
from mqc3.zx.gates import (
    BeamsplitterGate,
    CompactDiagram,
    ControlledSumGate,
    ControlledZGate,
    CubicPhaseGate,
    DisplacementGate,
    PhaseRotationGate,
    SqueezingGate,
    create_compact_diagram,
    expand_all,
)
from mqc3.zx.visualize_base_gates import DiagramVisualizer, VisualizerConfig, visualize

# Create output directory
OUTPUT_DIR = Path("test_images_gates")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
saved_ids = []


def save_and_close(diagram: Diagram, filename: str, title: str = ""):
    """Save a diagram visualization to a file and close the figure."""
    # Check diagram id
    assert isinstance(diagram.id, int)
    assert diagram.id not in saved_ids
    saved_ids.append(diagram.id)
    filepath = OUTPUT_DIR / filename
    fig = visualize(diagram, title=title)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def compact_gates_test():  # noqa: PLR0914
    """Test visualization of all gates in compact form."""
    print("Testing compact gate visualization...")

    # 1. Single gates (compact form)
    gates = [
        DisplacementGate(alpha=1.0 + 0.5j),
        PhaseRotationGate(theta=np.pi / 4),
        SqueezingGate(tau=0.5),
        ControlledSumGate(gain=1.0, control=1, target=2),
        ControlledSumGate(gain=2.0, control=2, target=1),
        ControlledZGate(gain=1.0),
        ControlledZGate(gain=2.0),
        BeamsplitterGate(theta=np.pi / 4),
        BeamsplitterGate(theta=np.pi / 6),
        CubicPhaseGate(gamma=0.1),
    ]

    for gate in gates:
        filename = f"{gate.__class__.__name__}_compact.png"
        save_and_close(gate, filename, f"{gate!r} (compact)")

    # 2. Test CompactDiagram.compose with connectivity
    print("\n" + "=" * 60)
    print("Testing CompactDiagram.compose with connectivity")
    print("=" * 60)

    # Create some compact gates
    D = DisplacementGate(alpha=1.0 + 0.5j)  # noqa: N806
    R = PhaseRotationGate(theta=np.pi / 4)  # noqa: N806
    Sq = SqueezingGate(tau=0.5)  # noqa: N806
    BS = BeamsplitterGate(theta=np.pi / 4)  # noqa: N806
    CS = ControlledSumGate(gain=2.0, control=2, target=1)  # noqa: N806
    CZ = ControlledZGate(gain=1.0)  # noqa: N806

    # Test 1: Simple composition with non-trivial connectivity (1 input, 1 output)
    # D has 1 input, 1 output; R has 1 input, 1 output
    # Connectivity: map input 0 of D to output 0 of R (only possible)
    comp1 = D.compose(R, connectivity={0: 0})
    save_and_close(comp1, "Compact_compose_D_R_conn.png", "D ∘ R with connectivity (0→0)")

    # Test 2: Composition with swapped connectivity (if arities allow)
    # Create a diagram with 2 inputs and 2 outputs
    # We can use a CompactDiagram from a tensor product
    D_tensor = D.tensor(D)  # noqa: N806
    R_tensor = R.tensor(R)  # noqa: N806

    # Connectivity: swap the wires (0→1, 1→0)
    comp2 = D_tensor.compose(R_tensor, connectivity={0: 1, 1: 0})
    save_and_close(
        comp2, "Compact_compose_tensor_swapped_conn.png", "(D⊗D) ∘ (R⊗R) with swapped connectivity (0→1, 1→0)"
    )

    # Test 3: Connectivity with identity (0→0, 1→1) for comparison
    comp3 = D_tensor.compose(R_tensor, connectivity={0: 0, 1: 1})
    save_and_close(
        comp3, "Compact_compose_tensor_identity_conn.png", "(D⊗D) ∘ (R⊗R) with identity connectivity (0→0, 1→1)"
    )

    # Test 4: CompactDiagram with CompositionDiagram connectivity
    comp_diag = CS.compose(CZ)  # CompositionDiagram
    BS_compact = BS  # CompactDiagram  # noqa: N806

    comp4 = comp_diag.compose(BS_compact)
    save_and_close(comp4, "Compact_compose_CompDiagram_BS_conn.png", "(D ∘ R) ∘ BS with connectivity (0→0)")

    # Test 5: Non-trivial connectivity with 2-input, 2-output diagrams
    # Create two 2-mode gates
    R_tensor = R.tensor(R)  # noqa: N806
    Sq_tensor = Sq.tensor(Sq)  # noqa: N806

    comp5 = R_tensor.compose(Sq_tensor, connectivity={0: 1, 1: 0})
    save_and_close(
        comp5,
        "Compact_compose_R_tensor_Sq_tensor_reverse_conn.png",
        "(R⊗R) ∘ (Sq⊗Sq) with reverse connectivity (0→1, 1→0)",
    )

    # Test 6: Connectivity with partial mapping

    phase = ZxPoly({2: 2})
    q_spider = QSpider(2, 1, phase)
    compact_q = CompactDiagram("Q2x1", 2, 1, q_spider)
    q_spider_1x2 = QSpider(1, 2, phase)
    compact_q_1x2 = CompactDiagram("Q1x2", 1, 2, q_spider_1x2)

    comp6 = compact_q.compose(compact_q_1x2, connectivity={0: 0, 1: 1})
    save_and_close(comp6, "Compact_compose_different_arities_conn.png", "Q(2→1) ∘ Q(1→2) with connectivity (0→0, 1→1)")

    # Test 7: Connectivity with expansion
    # Compose with expand_self=True
    comp7 = compact_q.compose(compact_q_1x2, connectivity={0: 1, 1: 0}, expand_self=True)
    save_and_close(
        comp7,
        "Compact_compose_expand_self_conn.png",
        "Q(2→1) ∘ Q(1→2) with expand_self=True and swapped connectivity (0→1, 1→0)",
    )

    print("CompactDiagram.compose connectivity tests completed.\n")


def expanded_gates_test():
    """Test visualization of gates after expansion."""
    print("Testing expanded gate visualization...")

    gates = [
        DisplacementGate(alpha=1.0 + 0.5j),
        PhaseRotationGate(theta=np.pi / 4),
        SqueezingGate(tau=0.5),
        ControlledSumGate(gain=1.0, control=1, target=2),
        ControlledZGate(gain=1.0),
        BeamsplitterGate(theta=np.pi / 8),
        CubicPhaseGate(gamma=0.1),
    ]

    for gate in gates:
        expanded = gate.expand()
        filename = f"{gate.__class__.__name__}_expanded.png"
        save_and_close(expanded, filename, f"{gate!r} (expanded)")


def composed_gates_test():
    """Test visualization of composed gates (compact and expanded)."""
    print("Testing composed gate visualization...")

    R = PhaseRotationGate(theta=np.pi / 4)  # noqa: N806
    Sq = SqueezingGate(tau=0.5)  # noqa: N806
    BS = BeamsplitterGate(theta=np.pi / 4)  # noqa: N806

    # Compact composition
    circuit = R.compose(Sq)
    save_and_close(circuit, "Compact_composition.png", "Compact: Sq ∘ R")

    # Expanded composition
    expanded = expand_all(circuit)
    save_and_close(expanded, "Expanded_composition.png", "Expanded: Sq ∘ R")

    # More complex circuit
    circuit2 = R.compose(Sq).tensor(BS)
    save_and_close(circuit2, "Compact_complex_circuit.png", "Compact: (Sq ∘ R) ⊗ BS")

    expanded2 = expand_all(circuit2)
    save_and_close(expanded2, "Expanded_complex_circuit.png", "Expanded: (Sq ∘ R) ⊗ BS")


def with_custom_config_test():
    """Test visualization with custom visualizer configuration."""
    print("Testing custom configuration...")

    config = VisualizerConfig(
        node_radius=4,
        vertical_factor=3.5,
        wire_width=2.0,
        fontsize=12,
    )

    R = PhaseRotationGate(theta=np.pi / 4)  # noqa: N806
    Sq = SqueezingGate(tau=0.5)  # noqa: N806
    circuit = R.compose(Sq)

    visualizer = DiagramVisualizer(config)
    fig = visualizer.visualize(circuit, title="Custom Config: Sq ∘ R")

    filepath = OUTPUT_DIR / "Custom_config.png"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved: {filepath}")


def compact_diagram_label_validation_test():
    """Test label validation for CompactDiagram."""
    print("Testing label validation...")
    # Empty label
    try:
        d = CompactDiagram("", 1, 1, None)  # noqa: F841
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")


def create_compact_diagram_test():  # noqa: PLR0914, PLR0915
    """Compress a diagram into a compact diagram."""
    print("\n" + "=" * 60)
    print("Testing create_compact_diagram")
    print("=" * 60)

    # ------------------------------------------------------------------------
    # Setup: Build complex diagrams from test_visualize_base_gates
    # ------------------------------------------------------------------------

    phase_poly_simple = ZxPoly({1: 2, 2: 4})
    phase_poly_complex = ZxPoly({1: 2, 3: 4, 5: 7})

    fourier = Fourier()
    inverse_fourier = FourierInv()
    fourier_squared = Fourier2()
    swap = Swap()
    q_spider_3x3 = QSpider(3, 3, phase_poly_simple)
    q_spider_5x5 = QSpider(5, 5, phase_poly_simple)
    p_spider_3x2 = PSpider(3, 2, phase_poly_simple)
    p_spider_5x5 = PSpider(5, 5, phase_poly_simple)
    q_spider_4x3 = QSpider(4, 3, phase_poly_simple)

    # ------------------------------------------------------------------------
    # 1. Big complex diagram
    # ------------------------------------------------------------------------

    large_tensor = fourier.tensor(fourier_squared)
    large_tensor = large_tensor.tensor(fourier_squared)
    swap_tensor_fourier = swap.tensor(fourier)

    large_composition = large_tensor.compose(swap_tensor_fourier)
    large_composition = swap_tensor_fourier.compose(large_composition)
    large_composition = swap_tensor_fourier.compose(large_composition)

    nested_composition_block = CompositionDiagram([q_spider_3x3, q_spider_3x3])
    nested_composition_block = nested_composition_block.compose(q_spider_3x3)
    nested_composition_block = nested_composition_block.compose(q_spider_3x3)

    large_composition = large_composition.compose(nested_composition_block)

    final_large_diagram = fourier.tensor(large_composition)

    # Compact the big complex diagram
    compact_large = create_compact_diagram("Big", 4, 4, final_large_diagram)
    save_and_close(compact_large, "compact_big_complex.png", "Compact Big Complex Diagram")

    # Test expansion: expand() should return the original diagram
    expanded_large = compact_large.expand()
    # Compare by checking if they are equivalent (or the same object)
    # Since expand() returns the decomposition stored in compact_large,
    # it should be the same as final_large_diagram if we stored it correctly
    assert expanded_large == final_large_diagram

    save_and_close(compact_large.expand(), "original_big_complex.png", "Original Big Complex Diagram")

    # ------------------------------------------------------------------------
    # 2. Full circuit: (F⊗F) ∘ Swap ∘ (F⊗F)
    # ------------------------------------------------------------------------

    left_tensor = fourier.tensor(fourier)
    middle_swap = left_tensor.compose(swap)
    right_tensor = fourier.tensor(fourier)
    full_circuit = middle_swap.compose(right_tensor)

    # Compact the full circuit
    compact_full = create_compact_diagram("Full", 2, 2, full_circuit)
    save_and_close(compact_full, "compact_full_circuit.png", "Compact Full Circuit")

    # Test expansion
    expanded_full = compact_full.expand()
    assert expanded_full == full_circuit
    save_and_close(full_circuit, "original_full_circuit.png", "Original Full Circuit")

    # ------------------------------------------------------------------------
    # 3. Contracted diagrams inside Composition Diagrams
    # ------------------------------------------------------------------------

    # Create contracted diagrams from visualize_base_gates
    # Contract QSpider_5x5 with PSpider_5x5
    contracted_diagram_1 = ContractedDiagram(q_spider_5x5, p_spider_5x5, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 4])
    # Contract Swap with InverseFourier (no feedback connections)
    contracted_diagram_2 = ContractedDiagram(swap, inverse_fourier, [0], [0], [], [])
    # Contract QSpider_4x3 with Swap
    contracted_diagram_3 = ContractedDiagram(q_spider_4x3, swap, [1], [1], [0, 2], [0, 1])
    # Contract Fourier2 with PSpider_3x2
    contracted_diagram_4 = ContractedDiagram(fourier_squared, p_spider_3x2, [0], [1], [0], [0])

    # Build composition with contracted diagrams
    tensor_block_1 = swap.tensor(fourier)
    tensor_block_1 = tensor_block_1.tensor(fourier_squared)

    composition_with_contracted_1 = contracted_diagram_1.compose(tensor_block_1)
    composition_with_contracted_1 = tensor_block_1.compose(composition_with_contracted_1)
    tensor_block_2 = fourier_squared.tensor(inverse_fourier)

    # Test different compositions with contracted diagrams
    test_cases = [
        ("contracted_comp_1", composition_with_contracted_1, 4, 4),
        ("contracted_comp_2", contracted_diagram_2.compose(tensor_block_2), 2, 2),
        ("contracted_comp_3", tensor_block_2.compose(contracted_diagram_3), 3, 2),
        ("contracted_comp_4", fourier.compose(contracted_diagram_4.compose(tensor_block_2)), 2, 1),
    ]

    for name, diagram, num_in, num_out in test_cases:
        compact = create_compact_diagram(name[:5], num_in, num_out, diagram)
        save_and_close(compact, f"compact_{name}.png", f"Compact {name}")

        # Test expansion
        expanded = compact.expand()
        assert expanded == diagram, f"Expanded {name} does not match original"
        save_and_close(expanded, f"original_{name}.png", f"Original {name}")

    # ------------------------------------------------------------------------
    # 4. Contracted diagrams inside Tensor Diagrams
    # ------------------------------------------------------------------------

    q_spider_5x5_large = QSpider(5, 5, 20 * (phase_poly_simple + phase_poly_complex))

    large_tensor_with_contracted = contracted_diagram_1.tensor(q_spider_5x5_large)
    large_tensor_with_contracted = p_spider_5x5.tensor(large_tensor_with_contracted)
    large_tensor_with_contracted = contracted_diagram_2.tensor(large_tensor_with_contracted)

    compact_tensor_contracted = create_compact_diagram("Tc", 11, 11, large_tensor_with_contracted)
    save_and_close(compact_tensor_contracted, "compact_tensor_contracted.png", "Compact Tensor with Contracted")

    expanded_tensor = compact_tensor_contracted.expand()
    assert expanded_tensor == large_tensor_with_contracted
    save_and_close(expanded_tensor, "original_tensor_contracted.png", "Original Tensor with Contracted")


def conjugate_gates_test():
    """Test conjugation of gates."""
    print("Testing gate conjugation...")

    gates = [
        DisplacementGate(alpha=1.0 + 0.5j),
        PhaseRotationGate(theta=np.pi / 4),
        SqueezingGate(tau=0.5),
        ControlledSumGate(gain=1.0),
        ControlledZGate(gain=1.0),
        BeamsplitterGate(theta=np.pi / 4),
        CubicPhaseGate(gamma=0.1),
    ]

    for gate in gates:
        print()
        conjugated = gate.conjugate()
        filename = f"{gate.__class__.__name__}_conjugate.png"
        save_and_close(conjugated, filename, f"{gate!r}†")

        # Verify conjugation consistency: (gate†)† = gate
        double_conj = conjugated.conjugate()
        if isinstance(double_conj, type(gate)):
            print(f"✅ {gate.__class__.__name__} conjugation consistent")
        else:
            print(f"❌ {gate.__class__.__name__} conjugation failed")


def run_all_tests():
    """Run all test functions."""
    compact_gates_test()
    expanded_gates_test()
    composed_gates_test()
    with_custom_config_test()
    compact_diagram_label_validation_test()
    conjugate_gates_test()
    create_compact_diagram_test()


if __name__ == "__main__":
    run_all_tests()

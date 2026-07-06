"""Graphical tests for visualize_cv_zx.py - human verification required.

Run this script directly to generate all test images for visual inspection.
Images are saved in the 'test_images' directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt

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
from mqc3.zx.visualize_cv_zx import visualize

# Create output directory using Path
OUTPUT_DIR = Path("test_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_and_close(diagram: Diagram, filename: str, title: str = ""):
    """Save a diagram visualization to a file in test_images/ and close the figure."""
    filepath = OUTPUT_DIR / filename  # Path with / operator
    fig = visualize(diagram, title=title)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def run_graphical_tests():  # noqa: PLR0914, PLR0915
    """Run all graphical tests - requires human verification."""
    print("Generating graphical test images...")
    print(f"Output directory: {OUTPUT_DIR}/")

    # ========================================================================
    # 1. Single proper diagrams
    # ========================================================================

    phase_poly_simple = ZxPoly({1: 2, 2: 4})
    phase_poly_complex = ZxPoly({1: 2, 3: 4, 5: 7})

    fourier = Fourier()
    fourier_squared = Fourier2()
    inverse_fourier = FourierInv()
    swap = Swap()

    # Q-Spiders with different arities and phases
    q_spider_1x1 = QSpider(1, 1, phase_poly_simple)
    q_spider_3x2 = QSpider(3, 2, phase_poly_simple)
    q_spider_4x3 = QSpider(4, 3, phase_poly_simple)
    q_spider_5x5 = QSpider(5, 5, phase_poly_simple)

    # P-Spiders with different arities and phases
    p_spider_1x1 = PSpider(1, 1, phase_poly_complex)
    p_spider_3x2 = PSpider(3, 2, phase_poly_simple)
    p_spider_3x4 = PSpider(3, 4, phase_poly_simple)
    p_spider_5x5 = PSpider(5, 5, phase_poly_simple)

    save_and_close(fourier, "Fourier.png", "Fourier")
    save_and_close(fourier_squared, "Fourier2.png", "Fourier2")
    save_and_close(inverse_fourier, "Inverse_Fourier.png", "Inverse Fourier")
    save_and_close(q_spider_1x1, "Single_QSpider_1.png", "Single QSpider (1*1)")
    save_and_close(q_spider_3x2, "Single_QSpider_2.png", "Single QSpider (3*2)")
    save_and_close(q_spider_4x3, "Single_QSpider_3.png", "Single QSpider (4*3)")
    save_and_close(q_spider_5x5, "Single_QSpider_4.png", "Single QSpider (5*5)")
    save_and_close(p_spider_1x1, "Single_PSpider_1.png", "Single PSpider (1*1)")
    save_and_close(p_spider_3x2, "Single_PSpider_2.png", "Single PSpider (3*2)")
    save_and_close(p_spider_3x4, "Single_PSpider_3.png", "Single PSpider (3*4)")
    save_and_close(p_spider_5x5, "Single_PSpider_4.png", "Single PSpider (5*5)")
    save_and_close(swap, "Swap.png", "Swap")

    # ========================================================================
    # 2. Simple compositions
    # ========================================================================

    # QSpider (1*1) followed by Fourier (1*1)
    q_then_fourier = q_spider_1x1.compose(fourier)
    save_and_close(q_then_fourier, "Composition_Fourier_then_QSpider.png", "Composition: Fourier then QSpider")

    # Complex composition chain: Fourier2 → (QSpider→Fourier) → InverseFourier → PSpider
    complex_composition = fourier_squared.compose(q_then_fourier)
    complex_composition = inverse_fourier.compose(complex_composition)
    complex_composition = p_spider_1x1.compose(complex_composition)
    save_and_close(complex_composition, "Complex_Composition.png", "Complex Composition")

    # ========================================================================
    # 3. Tensor diagrams
    # ========================================================================

    # Simple tensor: Fourier ⊗ Fourier2
    tensor_fourier_pair = fourier.tensor(fourier_squared)
    save_and_close(tensor_fourier_pair, "Tensor_Fourier_tensor_Fourier2.png", "Tensor: Fourier ⊗ Fourier2")

    # Complex tensor: Swap ⊗ (Fourier⊗Fourier2) ⊗ PSpider_3x4 ⊗ QSpider_5x5
    complex_tensor = swap.tensor(tensor_fourier_pair)
    complex_tensor = p_spider_3x4.tensor(complex_tensor)
    complex_tensor = q_spider_5x5.tensor(complex_tensor)
    save_and_close(complex_tensor, "Complex_Tensor_Diagram.png", "Complex Tensor Diagram")

    # ========================================================================
    # 4. Composition of tensor with swap (2 outputs → 2 inputs)
    # ========================================================================

    # Fourier ⊗ Fourier has 2 outputs, Swap has 2 inputs
    two_fouriers_tensor = fourier.tensor(fourier)
    swap_after_two_fouriers = two_fouriers_tensor.compose(swap)
    save_and_close(swap_after_two_fouriers, "Composition_Swap_then_F_tensor_F.png", "Composition: Swap then (F ⊗ F)")

    # ========================================================================
    # 5. Nested composition
    # ========================================================================

    nested_composition = q_spider_1x1.compose(fourier).compose(fourier_squared)
    save_and_close(
        nested_composition,
        "Nested_composition_c_then_a_then_b.png",
        "Nested composition: (QSpider ∘ Fourier) ∘ Fourier2",
    )

    # ========================================================================
    # 6. Tensor containing composition
    # ========================================================================

    # Fourier ⊗ (QSpider→Fourier)
    tensor_with_composition = fourier.tensor(q_then_fourier)
    save_and_close(tensor_with_composition, "Tensor_containing_composition.png", "Tensor containing composition")

    # ========================================================================
    # 7. Complex full circuit: (F⊗F) ∘ Swap ∘ (F⊗F)
    # ========================================================================

    left_tensor = fourier.tensor(fourier)  # 2 outputs
    middle_swap = left_tensor.compose(swap)  # 2 outputs after swap
    right_tensor = fourier.tensor(fourier)  # 2 inputs
    full_circuit = middle_swap.compose(right_tensor)
    save_and_close(full_circuit, "Full_circuit_F_tensor_F_swap_F_tensor_F.png", "Full circuit: (F⊗F) ∘ Swap ∘ (F⊗F)")

    # ========================================================================
    # 8. Big complex diagram
    # ========================================================================

    # Build up a large diagram incrementally
    large_tensor = fourier.tensor(fourier_squared)
    large_tensor = large_tensor.tensor(fourier_squared)
    swap_tensor_fourier = swap.tensor(fourier)

    large_composition = large_tensor.compose(swap_tensor_fourier)
    large_composition = swap_tensor_fourier.compose(large_composition)
    large_composition = swap_tensor_fourier.compose(large_composition)

    q_spider_3x3 = QSpider(3, 3, phase_poly_simple)
    nested_composition_block = CompositionDiagram([q_spider_3x3, q_spider_3x3])
    nested_composition_block = nested_composition_block.compose(q_spider_3x3)
    nested_composition_block = nested_composition_block.compose(q_spider_3x3)

    large_composition = large_composition.compose(nested_composition_block)

    final_large_diagram = fourier.tensor(large_composition)
    save_and_close(final_large_diagram, "Complex_Diagram.png", "Complex Diagram")

    # ========================================================================
    # 9. Proper contracted diagrams
    # ========================================================================

    # Contract QSpider_5x5 with PSpider_5x5
    contracted_diagram_1 = ContractedDiagram(q_spider_5x5, p_spider_5x5, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 4])
    # Contract Swap with InverseFourier (no feedback connections)
    contracted_diagram_2 = ContractedDiagram(swap, inverse_fourier, [0], [0], [], [])
    # Contract QSpider_4x3 with Swap
    contracted_diagram_3 = ContractedDiagram(q_spider_4x3, swap, [1], [1], [0, 2], [0, 1])
    # Contract Fourier2 with PSpider_3x2
    contracted_diagram_4 = ContractedDiagram(fourier_squared, p_spider_3x2, [0], [1], [0], [0])

    save_and_close(contracted_diagram_1, "Proper_Contracted_Diagram.png", "Proper Contracted Diagram")
    save_and_close(
        contracted_diagram_2,
        "Proper_Contracted_Diagram_Swap_and_Inv_Fourier.png",
        "Proper Contracted Diagram: Swap and Inv Fourier",
    )
    save_and_close(
        contracted_diagram_3,
        "Proper_Contracted_Diagram_QSpider_and_Swap.png",
        "Proper Contracted Diagram: QSpider and Swap",
    )
    save_and_close(
        contracted_diagram_4,
        "Proper_Contracted_Diagram_Fourier2_and_PSpider.png",
        "Proper Contracted Diagram: Fourier2 and PSpider",
    )

    # ========================================================================
    # 10. Contracted diagrams inside Composition Diagrams
    # ========================================================================

    # Build some tensor blocks for composition
    tensor_block_1 = swap.tensor(fourier)
    tensor_block_1 = tensor_block_1.tensor(fourier_squared)

    composition_with_contracted_1 = contracted_diagram_1.compose(tensor_block_1)
    composition_with_contracted_1 = tensor_block_1.compose(composition_with_contracted_1)
    tensor_block_2 = fourier_squared.tensor(inverse_fourier)

    save_and_close(
        composition_with_contracted_1,
        "Proper_Contracted_Diagram_inside_Composition_1.png",
        "Proper Contracted Diagram inside a Composition Diagram 1",
    )
    save_and_close(
        contracted_diagram_2.compose(tensor_block_2),
        "Proper_Contracted_Diagram_inside_Composition_2.png",
        "Proper Contracted Diagram inside a Composition Diagram 2",
    )
    save_and_close(
        tensor_block_2.compose(contracted_diagram_3),
        "Proper_Contracted_Diagram_inside_Composition_3.png",
        "Proper Contracted Diagram inside a Composition Diagram 3",
    )
    save_and_close(
        fourier.compose(contracted_diagram_4.compose(tensor_block_2)),
        "Proper_Contracted_Diagram_inside_Composition_4.png",
        "Proper Contracted Diagram inside a Composition Diagram 4",
    )

    # ========================================================================
    # 11. Contracted diagrams inside Tensor Diagrams
    # ========================================================================

    q_spider_5x5_large = QSpider(5, 5, 20 * (phase_poly_simple + phase_poly_complex))

    large_tensor_with_contracted = contracted_diagram_1.tensor(q_spider_5x5_large)
    large_tensor_with_contracted = p_spider_5x5.tensor(large_tensor_with_contracted)
    large_tensor_with_contracted = contracted_diagram_2.tensor(large_tensor_with_contracted)
    save_and_close(
        large_tensor_with_contracted,
        "Proper_Contracted_Diagram_inside_Tensor.png",
        "Proper Contracted Diagram inside a Tensor Diagram",
    )

    # ========================================================================
    # 12. Contracted diagrams composed of Composition Diagrams
    # ========================================================================

    # Build composition chains and contract them
    composition_chain_1 = q_spider_5x5.compose(p_spider_5x5)
    composition_chain_1 = q_spider_5x5_large.compose(composition_chain_1)

    contracted_from_composition_1 = ContractedDiagram(
        composition_chain_1, q_spider_5x5, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 4]
    )
    contracted_from_composition_2 = ContractedDiagram(
        q_spider_5x5, composition_chain_1, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 4]
    )

    save_and_close(
        contracted_from_composition_1,
        "Contracted_diagram_composed_of_Composition_1.png",
        "Contracted diagram composed of Composition Diagrams 1",
    )
    save_and_close(
        contracted_from_composition_2,
        "Contracted_diagram_composed_of_Composition_2.png",
        "Contracted diagram composed of Composition Diagrams 2",
    )

    # ========================================================================
    # 13. Contracted diagrams composed of Tensor Diagrams
    # ========================================================================

    # Build tensor blocks and contract them
    tensor_block_3 = q_spider_5x5.tensor(swap)
    tensor_block_4 = p_spider_3x4.tensor(fourier)
    tensor_block_4 = tensor_block_4.tensor(fourier_squared)
    tensor_block_4 = tensor_block_4.tensor(inverse_fourier)
    tensor_block_5 = q_spider_4x3.tensor(fourier)

    contracted_from_tensor_1 = ContractedDiagram(
        tensor_block_3, tensor_block_4, [0, 1, 4, 6], [1, 2, 3, 4], [0, 1, 2, 5], [0, 1, 2, 5]
    )
    contracted_from_tensor_2 = ContractedDiagram(
        q_spider_5x5, tensor_block_3, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 4]
    )
    contracted_from_tensor_3 = ContractedDiagram(
        tensor_block_4, tensor_block_5, [0, 1, 2, 6], [0, 1, 2, 4], [4, 5], [1, 3]
    )

    save_and_close(
        contracted_from_tensor_1,
        "Contracted_diagram_composed_of_Tensor_1.png",
        "Contracted diagram composed of Tensor Diagrams 1",
    )
    save_and_close(
        contracted_from_tensor_2,
        "Contracted_diagram_composed_of_Tensor_2.png",
        "Contracted diagram composed of Tensor Diagrams 2",
    )
    save_and_close(
        contracted_from_tensor_3,
        "Contracted_diagram_composed_of_Tensor_3.png",
        "Contracted diagram composed of Tensor Diagrams 3",
    )

    print(f"\nAll graphical test images generated successfully in '{OUTPUT_DIR}/'!")
    print("Please inspect the generated images manually.")


if __name__ == "__main__":
    run_graphical_tests()

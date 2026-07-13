"""Graphical tests for visualize_base_gates.py - human verification required.

Run this script directly to generate all test images for visual inspection.
Images are saved in the 'test_images' directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

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
)
from mqc3.zx.visualize_base_gates import visualize

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


def test_base_gates_exceptions():
    """Test that exceptions are raised correctly base_gates classes."""
    phase = ZxPoly({1: 2, 2: 4})
    q1 = QSpider(1, 1, phase)
    q2 = QSpider(2, 2, phase)
    fourier = Fourier()

    # --- CompositionDiagram ---
    with pytest.raises(ValueError, match="Cannot compose diagram"):
        CompositionDiagram([q1, q2])  # q1 outputs=1, q2 inputs=2 -> should fail

    with pytest.raises(ValueError, match="The keys of the connectivity dictionary do not correspond"):
        CompositionDiagram([fourier, fourier], connectivity={0: {1: 0}})  # fourier has 1 input, key must be 0 only

    with pytest.raises(ValueError, match="The values of the connectivity dictionary do not correspond"):
        CompositionDiagram([fourier, fourier], connectivity={0: {0: 1}})  # fourier has 1 output, value must be 0

    # --- ContractedDiagram ---
    with pytest.raises(ValueError, match="contain duplicate indices"):
        ContractedDiagram(q1, q2, [0, 0], [0, 1], [], [])

    with pytest.raises(ValueError, match="must equal"):
        ContractedDiagram(q1, q2, [0], [0, 1], [], [])

    with pytest.raises(ValueError, match="out of range"):
        ContractedDiagram(q1, q2, [2], [0], [], [])

    with pytest.raises(ValueError, match="out of range"):
        ContractedDiagram(q1, q2, [], [], [2], [0])

    # --- ProperDiagram.compose ---
    with pytest.raises(ValueError, match="The keys of the connectivity dictionary do not correspond"):
        fourier.compose(fourier, connectivity={1: 0})  # fourier has 1 input, key must be 0

    with pytest.raises(ValueError, match="The values of the connectivity dictionary do not correspond"):
        fourier.compose(fourier, connectivity={0: 1})  # other.fourier has 1 output, value must be 0

    # --- TensorDiagram.compose ---
    tensor = TensorDiagram([q1, q2])  # num_inputs = 1+2 = 3
    with pytest.raises(ValueError, match="The keys of the connectivity dictionary do not correspond"):
        tensor.compose(fourier, connectivity={0: 0, 1: 1})  # keys must be 0,1,2

    with pytest.raises(ValueError, match="The values of the connectivity dictionary do not correspond"):
        tensor.compose(fourier, connectivity={0: 0, 1: 0, 2: 1})  # fourier has 1 output, values must be 0


def run_graphical_tests():  # noqa: PLR0914, PLR0915
    """Run all graphical tests - requires human verification."""
    print("Generating graphical test images...")
    print(f"Output directory: {OUTPUT_DIR}/")

    # ========================================================================
    # 1. Single proper diagrams (unchanged)
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
    # save_and_close(q_then_fourier, "Composition_Fourier_then_QSpider.png", "Composition: Fourier then QSpider")

    # Complex composition chain: Fourier2 → (QSpider→Fourier) → InverseFourier → PSpider
    complex_composition = fourier_squared.compose(q_then_fourier)
    complex_composition = inverse_fourier.compose(complex_composition)
    complex_composition = p_spider_1x1.compose(complex_composition)
    # save_and_close(complex_composition, "Complex_Composition.png", "Complex Composition")

    # ========================================================================
    # 3. Tensor diagrams
    # ========================================================================

    tensor_fourier_pair = fourier.tensor(fourier_squared)
    save_and_close(tensor_fourier_pair, "Tensor_Fourier_tensor_Fourier2.png", "Tensor: Fourier ⊗ Fourier2")

    complex_tensor = swap.tensor(tensor_fourier_pair)
    complex_tensor = p_spider_3x4.tensor(complex_tensor)
    complex_tensor = q_spider_5x5.tensor(complex_tensor)
    save_and_close(complex_tensor, "Complex_Tensor_Diagram.png", "Complex Tensor Diagram")

    # ========================================================================
    # 4. Composition of tensor with swap (2 outputs → 2 inputs)
    # ========================================================================

    two_fouriers_tensor = fourier.tensor(fourier)
    swap_after_two_fouriers = two_fouriers_tensor.compose(swap)
    save_and_close(swap_after_two_fouriers, "Composition_Swap_then_F_tensor_F.png", "Composition: Swap then (F ⊗ F)")

    # Connectivity variant
    swap_conn = two_fouriers_tensor.compose(swap, connectivity={0: 1, 1: 0})
    save_and_close(
        swap_conn, "Composition_Swap_then_F_tensor_F_conn.png", "Composition: Swap then (F ⊗ F) (explicit conn)"
    )

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

    tensor_with_composition = fourier.tensor(q_then_fourier)
    save_and_close(tensor_with_composition, "Tensor_containing_composition.png", "Tensor containing composition")

    # ========================================================================
    # 7. Complex full circuit: (F⊗F) ∘ Swap ∘ (F⊗F)
    # ========================================================================

    left_tensor = fourier.tensor(fourier)
    middle_swap = left_tensor.compose(swap)
    right_tensor = fourier.tensor(fourier)
    full_circuit = middle_swap.compose(right_tensor)
    save_and_close(full_circuit, "Full_circuit_F_tensor_F_swap_F_tensor_F.png", "Full circuit: (F⊗F) ∘ Swap ∘ (F⊗F)")

    # Connectivity variant
    left_conn = fourier.tensor(fourier)
    mid_conn = left_conn.compose(swap, connectivity={0: 1, 1: 0})
    right_conn = fourier.tensor(fourier)
    full_conn = mid_conn.compose(right_conn, connectivity={0: 1, 1: 0})
    save_and_close(full_conn, "Full_circuit_conn.png", "Full circuit (explicit conn): (F⊗F) ∘ Swap ∘ (F⊗F)")

    # ========================================================================
    # 8. Big complex diagram
    # ========================================================================

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

    # Connectivity variant for big diagram (identity connections)
    large_tensor_conn = fourier.tensor(fourier_squared)
    large_tensor_conn = large_tensor_conn.tensor(fourier_squared)
    swap_tensor_fourier_conn = swap.tensor(fourier)

    large_comp_conn = large_tensor_conn.compose(swap_tensor_fourier_conn, connectivity={0: 2, 1: 0, 2: 1})
    large_comp_conn = swap_tensor_fourier_conn.compose(large_comp_conn, connectivity={0: 1, 1: 0, 2: 2})
    large_comp_conn = swap_tensor_fourier_conn.compose(large_comp_conn)

    nested_block_conn = CompositionDiagram([q_spider_3x3, q_spider_3x3])
    nested_block_conn = nested_block_conn.compose(q_spider_3x3)
    nested_block_conn = nested_block_conn.compose(q_spider_3x3)

    large_comp_conn = large_comp_conn.compose(nested_block_conn, connectivity={0: 1, 1: 2, 2: 0})
    final_large_conn = fourier.tensor(large_comp_conn)
    save_and_close(final_large_conn, "Complex_Diagram_conn.png", "Complex Diagram (explicit conn)")

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

    # Connectivity variant 1
    n_inputs_cd1 = contracted_diagram_1.num_inputs
    n_outputs_tb1 = tensor_block_1.num_outputs

    if n_inputs_cd1 == n_outputs_tb1 and n_inputs_cd1 >= 2:
        # Create a cyclic permutation: 0->1, 1->2, ..., n-1->0
        conn_cd1_tb1 = {i: (i + 1) % n_inputs_cd1 for i in range(n_inputs_cd1)}
    else:
        # Fallback to identity
        conn_cd1_tb1 = {i: i for i in range(min(n_inputs_cd1, n_outputs_tb1))}

    # Also need connectivity for tensor_block_1.compose(composition_with_contracted_1)
    # This connects tensor_block_1 (outputs) to composition_with_contracted_1 (inputs)
    # We'll use a similar permutation
    n_outputs_tb1_2 = tensor_block_1.num_outputs
    n_inputs_cwc1 = composition_with_contracted_1.num_inputs

    if n_outputs_tb1_2 == n_inputs_cwc1 and n_outputs_tb1_2 >= 2:
        conn_tb1_cwc1 = {i: (i + 1) % n_outputs_tb1_2 for i in range(n_outputs_tb1_2)}
    else:
        conn_tb1_cwc1 = {i: i for i in range(min(n_outputs_tb1_2, n_inputs_cwc1))}

    # Build the composition with explicit connectivity
    composition_with_contracted_1_conn = contracted_diagram_1.compose(tensor_block_1, connectivity=conn_cd1_tb1)
    composition_with_contracted_1_conn = tensor_block_1.compose(
        composition_with_contracted_1_conn, connectivity=conn_tb1_cwc1
    )

    save_and_close(
        composition_with_contracted_1_conn,
        "Proper_Contracted_Diagram_inside_Composition_1_conn.png",
        "Proper Contracted Diagram inside a Composition Diagram 1 (non-trivial conn)",
    )

    # Connectivity variant 2
    n_inputs_cd2 = contracted_diagram_2.num_inputs
    n_outputs_tb2 = tensor_block_2.num_outputs

    if n_inputs_cd2 == n_outputs_tb2 and n_inputs_cd2 >= 2:
        # Reverse order: 0->n-1, 1->n-2, ...
        conn_cd2_tb2 = {i: n_inputs_cd2 - 1 - i for i in range(n_inputs_cd2)}
    else:
        conn_cd2_tb2 = {i: i for i in range(min(n_inputs_cd2, n_outputs_tb2))}

    comp2_conn = contracted_diagram_2.compose(tensor_block_2, connectivity=conn_cd2_tb2)

    save_and_close(
        comp2_conn,
        "Proper_Contracted_Diagram_inside_Composition_2_conn.png",
        "Proper Contracted Diagram inside a Composition Diagram 2 (non-trivial conn)",
    )

    # Connectivity variant 3
    n_outputs_tb2_2 = tensor_block_2.num_outputs
    n_inputs_cd3 = contracted_diagram_3.num_inputs

    if n_outputs_tb2_2 == n_inputs_cd3 and n_outputs_tb2_2 >= 2:
        # Swap pairs: 0->1, 1->0, 2->3, 3->2, ...
        conn_tb2_cd3 = {}
        for i in range(n_outputs_tb2_2):
            if i % 2 == 0 and i + 1 < n_outputs_tb2_2:
                conn_tb2_cd3[i] = i + 1
            elif i % 2 == 1:
                conn_tb2_cd3[i] = i - 1
            else:
                conn_tb2_cd3[i] = i
    else:
        conn_tb2_cd3 = {i: i for i in range(min(n_outputs_tb2_2, n_inputs_cd3))}

    comp3_conn = tensor_block_2.compose(contracted_diagram_3, connectivity=conn_tb2_cd3)

    save_and_close(
        comp3_conn,
        "Proper_Contracted_Diagram_inside_Composition_3_conn.png",
        "Proper Contracted Diagram inside a Composition Diagram 3 (non-trivial conn)",
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

"""Test visualization for CV quantum gates using CompactDiagram formalism.

This module provides test functions to visualize all gate types and verify
that the compact representation works correctly.

Run this script directly to generate test images.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sympy import I, cos, exp, sin, symbols
from sympy import pi as sym_pi

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
from cvzx.visualize_base_gates import DiagramVisualizer, VisualizerConfig, visualize

OUTPUT_DIR = Path("test_images_gates")


def save_and_close(diagram: Diagram, filename: str, title: str = ""):
    """Save a diagram visualization to a file and close the figure."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_ids = []
    # Check diagram id
    assert isinstance(diagram.id, int)
    assert diagram.id not in saved_ids
    saved_ids.append(diagram.id)
    filepath = OUTPUT_DIR / filename
    fig = visualize(diagram, title=title)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def compact_gates_test():  # noqa: PLR0914, PLR0915
    """Test visualization of all gates in compact form."""
    print("Testing compact gate visualization...")

    # Define symbolic variables
    a, b, c, d, e, theta, phi, gamma_sym = symbols("a b c d e theta phi gamma", real=True)
    alpha_sym = a + I * b  # Complex symbolic parameter for displacement

    # 1. Single gates - Numeric versions (compact form)
    gates_numeric = [
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

    for gate in gates_numeric:
        filename = f"{gate.__class__.__name__}_compact.png"
        # save_and_close(gate, filename, f"{gate!r} (compact)")

    # 2. Single gates - Parametric versions
    gates_parametric = [
        DisplacementGate(alpha=alpha_sym, parametric=True),
        DisplacementGate(alpha=exp(I * theta), parametric=True),
        PhaseRotationGate(theta=theta, parametric=True),
        PhaseRotationGate(theta=sin(theta) + cos(theta), parametric=True),
        SqueezingGate(tau=a, parametric=True),
        SqueezingGate(tau=exp(-b), parametric=True),
        ControlledSumGate(gain=c, control=1, target=2, parametric=True),
        ControlledSumGate(gain=d, control=2, target=1, parametric=True),
        ControlledZGate(gain=e, parametric=True),
        ControlledZGate(gain=sin(theta) + 1, parametric=True),
        BeamsplitterGate(theta=phi, parametric=True),
        BeamsplitterGate(theta=sym_pi / 4 + theta, parametric=True),
        CubicPhaseGate(gamma=gamma_sym, parametric=True),
        CubicPhaseGate(gamma=sin(theta) * 0.5, parametric=True),
    ]

    for gate in gates_parametric:
        filename = f"{gate.__class__.__name__}_parametric_compact.png"
        save_and_close(gate, filename, f"{gate!r} (parametric compact)")

    # 3. Test CompactDiagram.compose with connectivity
    print("\n" + "=" * 60)
    print("Testing CompactDiagram.compose with connectivity")
    print("=" * 60)

    # Create some compact gates (numeric and parametric)
    D = DisplacementGate(alpha=1.0 + 0.5j)  # noqa: N806
    D_sym = DisplacementGate(alpha=alpha_sym, parametric=True)  # noqa: N806
    R = PhaseRotationGate(theta=np.pi / 4)  # noqa: N806
    R_sym = PhaseRotationGate(theta=theta, parametric=True)  # noqa: N806
    Sq = SqueezingGate(tau=0.5)  # noqa: N806
    Sq_sym = SqueezingGate(tau=a, parametric=True)  # noqa: N806
    BS = BeamsplitterGate(theta=np.pi / 4)  # noqa: N806
    BS_sym = BeamsplitterGate(theta=phi, parametric=True)  # noqa: N806
    CS = ControlledSumGate(gain=2.0, control=2, target=1)  # noqa: N806
    CS_sym = ControlledSumGate(gain=c, control=2, target=1, parametric=True)  # noqa: N806
    CZ = ControlledZGate(gain=1.0)  # noqa: N806
    CZ_sym = ControlledZGate(gain=e, parametric=True)  # noqa: N806

    # Test 1: Simple composition with non-trivial connectivity (1 input, 1 output)
    comp1 = D.compose(R, connectivity={0: 0})
    save_and_close(comp1, "Compact_compose_D_R_conn.png", "D ∘ R with connectivity (0→0)")

    # Test 1b: Parametric version
    comp1_sym = D_sym.compose(R_sym, connectivity={0: 0})
    save_and_close(comp1_sym, "Compact_compose_D_sym_R_sym_conn.png", "D(a) ∘ R(θ) with connectivity (0→0)")

    # Test 2: Composition with swapped connectivity (if arities allow)
    D_tensor = D.tensor(D)  # noqa: N806
    R_tensor = R.tensor(R)  # noqa: N806

    comp2 = D_tensor.compose(R_tensor, connectivity={0: 1, 1: 0})
    save_and_close(
        comp2, "Compact_compose_tensor_swapped_conn.png", "(D⊗D) ∘ (R⊗R) with swapped connectivity (0→1, 1→0)"
    )

    # Test 2b: Parametric version
    D_sym_tensor = D_sym.tensor(D_sym)  # noqa: N806
    R_sym_tensor = R_sym.tensor(R_sym)  # noqa: N806
    comp2_sym = D_sym_tensor.compose(R_sym_tensor, connectivity={0: 1, 1: 0})
    save_and_close(
        comp2_sym,
        "Compact_compose_tensor_swapped_conn_sym.png",
        "(D(a)⊗D(a)) ∘ (R(θ)⊗R(θ)) with swapped connectivity (0→1, 1→0)",
    )

    # Test 3: Connectivity with identity (0→0, 1→1) for comparison
    comp3 = D_tensor.compose(R_tensor, connectivity={0: 0, 1: 1})
    save_and_close(
        comp3, "Compact_compose_tensor_identity_conn.png", "(D⊗D) ∘ (R⊗R) with identity connectivity (0→0, 1→1)"
    )

    # Test 4: CompactDiagram with CompositionDiagram connectivity
    comp_diag = CS.compose(CZ)
    BS_compact = BS  # noqa: N806

    comp4 = comp_diag.compose(BS_compact)
    save_and_close(comp4, "Compact_compose_CompDiagram_BS_conn.png", "(CS ∘ CZ) ∘ BS with connectivity (0→0)")

    # Test 4b: Parametric version
    comp_diag_sym = CS_sym.compose(CZ_sym)
    comp4_sym = comp_diag_sym.compose(BS_sym)
    save_and_close(
        comp4_sym,
        "Compact_compose_CompDiagram_BS_conn_sym.png",
        "(CS(c) ∘ CZ(e)) ∘ BS(φ) with connectivity (0→0)",
    )

    # Test 5: Non-trivial connectivity with 2-input, 2-output diagrams
    R_tensor = R.tensor(R)  # noqa: N806
    Sq_tensor = Sq.tensor(Sq)  # noqa: N806

    comp5 = R_tensor.compose(Sq_tensor, connectivity={0: 1, 1: 0})
    save_and_close(
        comp5,
        "Compact_compose_R_tensor_Sq_tensor_reverse_conn.png",
        "(R⊗R) ∘ (Sq⊗Sq) with reverse connectivity (0→1, 1→0)",
    )

    # Test 5b: Parametric version
    R_sym_tensor = R_sym.tensor(R_sym)  # noqa: N806
    Sq_sym_tensor = Sq_sym.tensor(Sq_sym)  # noqa: N806
    comp5_sym = R_sym_tensor.compose(Sq_sym_tensor, connectivity={0: 1, 1: 0})
    save_and_close(
        comp5_sym,
        "Compact_compose_R_tensor_Sq_tensor_reverse_conn_sym.png",
        "(R(θ)⊗R(θ)) ∘ (Sq(a)⊗Sq(a)) with reverse connectivity (0→1, 1→0)",
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

    # Define symbolic variables
    a, b, theta, phi, gamma_sym = symbols("a b theta phi gamma", real=True)
    alpha_sym = a + I * b

    # Numeric gates
    gates_numeric = [
        DisplacementGate(alpha=1.0 + 0.5j),
        PhaseRotationGate(theta=np.pi / 4),
        SqueezingGate(tau=0.5),
        ControlledSumGate(gain=1.0, control=1, target=2),
        ControlledZGate(gain=1.0),
        BeamsplitterGate(theta=np.pi / 8),
        CubicPhaseGate(gamma=0.1),
    ]

    for gate in gates_numeric:
        expanded = gate.expand()
        filename = f"{gate.__class__.__name__}_expanded.png"
        save_and_close(expanded, filename, f"{gate!r} (expanded)")

    # Parametric gates
    gates_parametric = [
        DisplacementGate(alpha=alpha_sym, parametric=True),
        PhaseRotationGate(theta=theta, parametric=True),
        SqueezingGate(tau=a, parametric=True),
        ControlledSumGate(gain=b, control=1, target=2, parametric=True),
        ControlledZGate(gain=phi, parametric=True),
        BeamsplitterGate(theta=theta, parametric=True),
        CubicPhaseGate(gamma=gamma_sym, parametric=True),
        # Complex parametric gates
        DisplacementGate(alpha=exp(I * theta), parametric=True),
        PhaseRotationGate(theta=sin(theta) + cos(theta), parametric=True),
        SqueezingGate(tau=exp(-a), parametric=True),
        ControlledZGate(gain=sin(phi) + 1, parametric=True),
        BeamsplitterGate(theta=sym_pi / 4 + theta, parametric=True),
        CubicPhaseGate(gamma=sin(theta) * 0.5, parametric=True),
    ]

    for gate in gates_parametric:
        expanded = gate.expand()
        filename = f"{gate.__class__.__name__}_parametric_expanded.png"
        save_and_close(expanded, filename, f"{gate!r} (parametric expanded)")


def with_custom_config_test():
    """Test visualization with custom visualizer configuration."""
    print("Testing custom configuration...")

    config = VisualizerConfig(
        node_radius=4,
        vertical_factor=3.5,
        wire_width=2.0,
        fontsize=12,
    )

    # Define symbolic variables
    a, theta = symbols("a theta", real=True)

    # Numeric version
    R = PhaseRotationGate(theta=np.pi / 4)  # noqa: N806
    Sq = SqueezingGate(tau=0.5)  # noqa: N806
    circuit = R.compose(Sq)

    visualizer = DiagramVisualizer(config)
    fig = visualizer.visualize(circuit, title="Custom Config: Sq ∘ R")

    filepath = OUTPUT_DIR / "Custom_config.png"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved: {filepath}")

    # Parametric version
    R_sym = PhaseRotationGate(theta=theta, parametric=True)  # noqa: N806
    Sq_sym = SqueezingGate(tau=a, parametric=True)  # noqa: N806
    circuit_sym = R_sym.compose(Sq_sym)

    fig_sym = visualizer.visualize(circuit_sym, title="Custom Config: Sq(a) ∘ R(θ)")
    filepath_sym = OUTPUT_DIR / "Custom_config_parametric.png"
    fig_sym.savefig(filepath_sym, dpi=150)
    plt.close(fig_sym)
    print(f"Saved: {filepath_sym}")


def compact_diagram_label_validation_test():
    """Test label validation for CompactDiagram."""
    print("Testing label validation...")
    # Empty label
    try:
        CompactDiagram("", 1, 1, None)
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")


def create_compact_diagram_test():  # noqa: PLR0914, PLR0915
    """Compress a diagram into a compact diagram."""
    print("\n" + "=" * 60)
    print("Testing create_compact_diagram")
    print("=" * 60)

    # Define symbolic variables
    a, b, c, theta, phi = symbols("a b c theta phi", real=True)

    # ------------------------------------------------------------------------
    # Setup: Build complex diagrams from test_visualize_base_gates
    # ------------------------------------------------------------------------

    phase_poly_simple = ZxPoly({1: 2, 2: 4})
    phase_poly_complex = ZxPoly({1: 2, 3: 4, 5: 7})
    # Symbolic phase polynomials
    phase_poly_sym_simple = ZxPoly({1: a, 2: b})
    phase_poly_sym_complex = ZxPoly({1: c, 3: theta, 5: phi})

    fourier = Fourier()
    inverse_fourier = FourierInv()
    fourier_squared = Fourier2()
    swap = Swap()

    # Numeric spiders
    q_spider_3x3 = QSpider(3, 3, phase_poly_simple)
    q_spider_5x5 = QSpider(5, 5, phase_poly_simple)
    p_spider_3x2 = PSpider(3, 2, phase_poly_simple)
    p_spider_5x5 = PSpider(5, 5, phase_poly_simple)
    q_spider_4x3 = QSpider(4, 3, phase_poly_simple)

    # Parametric spiders
    q_spider_3x3_sym = QSpider(3, 3, phase_poly_sym_simple)
    q_spider_5x5_sym = QSpider(5, 5, phase_poly_sym_complex)
    p_spider_5x5_sym = PSpider(5, 5, phase_poly_sym_complex)

    # ------------------------------------------------------------------------
    # 1. Big complex diagram (numeric)
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
    expanded_large = expand_all(compact_large)
    assert expanded_large == final_large_diagram

    save_and_close(final_large_diagram, "original_big_complex.png", "Original Big Complex Diagram")

    # ------------------------------------------------------------------------
    # 1b. Big complex diagram (parametric)
    # ------------------------------------------------------------------------

    large_tensor_sym = fourier.tensor(fourier_squared)
    large_tensor_sym = large_tensor_sym.tensor(fourier_squared)

    # Use parametric spiders
    nested_composition_block_sym = CompositionDiagram([q_spider_3x3_sym, q_spider_3x3_sym])
    nested_composition_block_sym = nested_composition_block_sym.compose(q_spider_3x3_sym)
    nested_composition_block_sym = nested_composition_block_sym.compose(q_spider_3x3_sym)

    large_composition_sym = swap_tensor_fourier.compose(nested_composition_block_sym)
    final_large_diagram_sym = fourier.tensor(large_composition_sym)

    compact_large_sym = create_compact_diagram("BigSym", 4, 4, final_large_diagram_sym)
    save_and_close(compact_large_sym, "compact_big_complex_sym.png", "Compact Big Complex Diagram (Parametric)")

    expanded_large_sym = expand_all(compact_large_sym)
    assert expanded_large_sym == final_large_diagram_sym

    save_and_close(
        final_large_diagram_sym, "original_big_complex_sym.png", "Original Big Complex Diagram (Parametric)"
    )

    # ------------------------------------------------------------------------
    # 2. Full circuit: (F⊗F) ∘ Swap ∘ (F⊗F)
    # ------------------------------------------------------------------------

    left_tensor = fourier.tensor(fourier)
    middle_swap = left_tensor.compose(swap)
    right_tensor = fourier.tensor(fourier)
    full_circuit = middle_swap.compose(right_tensor)

    compact_full = create_compact_diagram("Full", 2, 2, full_circuit)
    save_and_close(compact_full, "compact_full_circuit.png", "Compact Full Circuit")

    expanded_full = expand_all(compact_full)
    assert expanded_full == full_circuit
    save_and_close(full_circuit, "original_full_circuit.png", "Original Full Circuit")

    # ------------------------------------------------------------------------
    # 3. Contracted diagrams inside Composition Diagrams
    # ------------------------------------------------------------------------

    # Numeric contracted diagrams
    contracted_diagram_1 = ContractedDiagram(q_spider_5x5, p_spider_5x5, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 4])
    contracted_diagram_2 = ContractedDiagram(swap, inverse_fourier, [0], [0], [], [])
    contracted_diagram_3 = ContractedDiagram(q_spider_4x3, swap, [1], [1], [0, 2], [0, 1])
    contracted_diagram_4 = ContractedDiagram(fourier_squared, p_spider_3x2, [0], [1], [0], [0])

    # Parametric contracted diagrams
    contracted_diagram_1_sym = ContractedDiagram(
        q_spider_5x5_sym, p_spider_5x5_sym, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 4]
    )
    contracted_diagram_2_sym = ContractedDiagram(swap, inverse_fourier, [0], [0], [], [])
    contracted_diagram_3_sym = ContractedDiagram(q_spider_4x3, swap, [1], [1], [0, 2], [0, 1])

    # Build composition with contracted diagrams (numeric)
    tensor_block_1 = swap.tensor(fourier)
    tensor_block_1 = tensor_block_1.tensor(fourier_squared)

    composition_with_contracted_1 = contracted_diagram_1.compose(tensor_block_1)
    composition_with_contracted_1 = tensor_block_1.compose(composition_with_contracted_1)
    tensor_block_2 = fourier_squared.tensor(inverse_fourier)

    # Build composition with contracted diagrams (parametric)
    tensor_block_1_sym = swap.tensor(fourier)
    tensor_block_1_sym = tensor_block_1_sym.tensor(fourier_squared)

    composition_with_contracted_1_sym = contracted_diagram_1_sym.compose(tensor_block_1_sym)
    composition_with_contracted_1_sym = tensor_block_1_sym.compose(composition_with_contracted_1_sym)

    test_cases = [
        ("contracted_comp_1", composition_with_contracted_1, 4, 4),
        ("contracted_comp_2", contracted_diagram_2.compose(tensor_block_2), 2, 2),
        ("contracted_comp_3", tensor_block_2.compose(contracted_diagram_3), 3, 2),
        ("contracted_comp_4", fourier.compose(contracted_diagram_4.compose(tensor_block_2)), 2, 1),
    ]

    for name, diagram, num_in, num_out in test_cases:
        compact = create_compact_diagram(name[:5], num_in, num_out, diagram)
        save_and_close(compact, f"compact_{name}.png", f"Compact {name}")

        expanded = expand_all(compact)
        assert expanded == diagram, f"Expanded {name} does not match original"
        save_and_close(expanded, f"original_{name}.png", f"Original {name}")

    # Parametric test cases
    test_cases_sym = [
        ("contracted_comp_1_sym", composition_with_contracted_1_sym, 4, 4),
        ("contracted_comp_2_sym", contracted_diagram_2_sym.compose(tensor_block_2), 2, 2),
        ("contracted_comp_3_sym", tensor_block_2.compose(contracted_diagram_3_sym), 3, 2),
    ]

    for name, diagram, num_in, num_out in test_cases_sym:
        compact = create_compact_diagram(name[:5], num_in, num_out, diagram)
        save_and_close(compact, f"compact_{name}.png", f"Compact {name}")

        expanded = expand_all(compact)
        assert expanded == diagram, f"Expanded {name} does not match original"
        save_and_close(expanded, f"original_{name}.png", f"Original {name}")

    # ------------------------------------------------------------------------
    # 4. Contracted diagrams inside Tensor Diagrams
    # ------------------------------------------------------------------------

    # Numeric
    q_spider_5x5_large = QSpider(5, 5, 20 * (phase_poly_simple + phase_poly_complex))

    large_tensor_with_contracted = contracted_diagram_1.tensor(q_spider_5x5_large)
    large_tensor_with_contracted = p_spider_5x5.tensor(large_tensor_with_contracted)
    large_tensor_with_contracted = contracted_diagram_2.tensor(large_tensor_with_contracted)

    compact_tensor_contracted = create_compact_diagram("Tc", 11, 11, large_tensor_with_contracted)
    save_and_close(compact_tensor_contracted, "compact_tensor_contracted.png", "Compact Tensor with Contracted")

    expanded_tensor = expand_all(compact_tensor_contracted)
    assert expanded_tensor == large_tensor_with_contracted
    save_and_close(expanded_tensor, "original_tensor_contracted.png", "Original Tensor with Contracted")

    # Parametric
    q_spider_5x5_large_sym = QSpider(5, 5, 20 * (phase_poly_sym_simple + phase_poly_sym_complex))

    large_tensor_with_contracted_sym = contracted_diagram_1_sym.tensor(q_spider_5x5_large_sym)
    large_tensor_with_contracted_sym = p_spider_5x5_sym.tensor(large_tensor_with_contracted_sym)
    large_tensor_with_contracted_sym = contracted_diagram_2_sym.tensor(large_tensor_with_contracted_sym)

    compact_tensor_contracted_sym = create_compact_diagram("TcSym", 11, 11, large_tensor_with_contracted_sym)
    save_and_close(
        compact_tensor_contracted_sym,
        "compact_tensor_contracted_sym.png",
        "Compact Tensor with Contracted (Parametric)",
    )

    expanded_tensor_sym = expand_all(compact_tensor_contracted_sym)
    assert expanded_tensor_sym == large_tensor_with_contracted_sym
    save_and_close(
        expanded_tensor_sym, "original_tensor_contracted_sym.png", "Original Tensor with Contracted (Parametric)"
    )


def conjugate_gates_test():
    """Test conjugation of gates."""
    print("Testing gate conjugation...")

    # Define symbolic variables
    a, b, theta, phi, gamma_sym = symbols("a b theta phi gamma", real=True)
    alpha_sym = a + I * b

    # Numeric gates
    gates_numeric = [
        DisplacementGate(alpha=1.0 + 0.5j),
        PhaseRotationGate(theta=np.pi / 4),
        SqueezingGate(tau=0.5),
        ControlledSumGate(gain=1.0),
        ControlledZGate(gain=1.0),
        BeamsplitterGate(theta=np.pi / 4),
        CubicPhaseGate(gamma=0.1),
    ]

    for gate in gates_numeric:
        print()
        conjugated = gate.conjugate()
        filename = f"{gate.__class__.__name__}_conjugate.png"
        save_and_close(conjugated, filename, f"{gate!r}†")

        double_conj = conjugated.conjugate()
        if isinstance(double_conj, type(gate)):
            print(f"✅ {gate.__class__.__name__} conjugation consistent")
        else:
            print(f"❌ {gate.__class__.__name__} conjugation failed")

    # Parametric gates
    gates_parametric = [
        DisplacementGate(alpha=alpha_sym, parametric=True),
        PhaseRotationGate(theta=theta, parametric=True),
        SqueezingGate(tau=a, parametric=True),
        ControlledSumGate(gain=b, parametric=True),
        ControlledZGate(gain=phi, parametric=True),
        BeamsplitterGate(theta=theta, parametric=True),
        CubicPhaseGate(gamma=gamma_sym, parametric=True),
    ]

    for gate in gates_parametric:
        print()
        conjugated = gate.conjugate()
        filename = f"{gate.__class__.__name__}_parametric_conjugate.png"
        save_and_close(conjugated, filename, f"{gate!r}† (parametric)")

        double_conj = conjugated.conjugate()
        if isinstance(double_conj, type(gate)):
            print(f"✅ {gate.__class__.__name__} conjugation consistent (parametric)")
        else:
            print(f"❌ {gate.__class__.__name__} conjugation failed (parametric)")


def feedforward_test():  # noqa: PLR0914
    """Test feedforward display."""
    zero_phase = ZxPoly({})
    id_q = QSpider(1, 1, zero_phase)
    tensor1 = TensorDiagram([id_q, QSpider(0, 2, zero_phase)])
    contract = ContractedDiagram(PSpider(1, 2, zero_phase), QSpider(2, 1, zero_phase), [1], [0], [], [])
    tensor2 = TensorDiagram([contract, id_q])
    m1, m2 = symbols("m1 m2", real=True)
    meas_diag1 = PSpider(1, 0, ZxPoly({1: -m1}))
    meas_diag2 = QSpider(1, 0, ZxPoly({1: -m2}))
    tensor3 = TensorDiagram([
        DisplacementGate(
            m1 + I * m2, parametric=True, feedforward=True, measurement_ids={meas_diag1.id, meas_diag2.id}
        ),
        SqueezingGate(0.5),
        SqueezingGate(0.5),
    ])
    tensor4 = TensorDiagram([
        id_q,
        meas_diag1,
        meas_diag2,
    ])
    tensor5 = TensorDiagram([
        SqueezingGate(0.5),
        SqueezingGate(0.5),
        DisplacementGate(
            m1 + I * m2, parametric=True, feedforward=True, measurement_ids={meas_diag1.id, meas_diag2.id}
        ),
    ])
    tensor6 = TensorDiagram([
        meas_diag1,
        meas_diag2,
        id_q,
    ])
    diagram1 = CompositionDiagram([tensor1, tensor2, tensor3, tensor4])
    diagram2 = CompositionDiagram([tensor1, tensor2, tensor5, tensor6])
    filename1 = "Teleportation circuit 1.png"
    filename2 = "Teleportation circuit 2.png"
    save_and_close(diagram1, filename1, "Teleportation circuit 1")
    save_and_close(diagram2, filename2, "Teleportation circuit 2")


def run_all_tests():
    """Run all test functions."""
    # compact_gates_test()
    # expanded_gates_test()
    # with_custom_config_test()
    # compact_diagram_label_validation_test()
    # conjugate_gates_test()
    # create_compact_diagram_test()
    feedforward_test()


if __name__ == "__main__":
    run_all_tests()

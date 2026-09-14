"""Test visualization for CV quantum gates using CompactDiagram formalism.

This module provides test functions to visualize all gate types and verify
that the compact representation works correctly.

Run this script directly to generate test images.
"""

import random
from pathlib import Path
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
from sympy import I, cos, exp, sin, symbols
from sympy import pi as sym_pi

from cvzx.ir.base import (
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
from cvzx.ir.gates import (
    ArbitraryGate,
    BeamsplitterGate,
    CompactDiagram,
    ControlledSumGate,
    ControlledZGate,
    CubicPhaseGate,
    DisplacementGate,
    MeasurementGate,
    PhaseRotationGate,
    ShearPInvariantGate,
    ShearXInvariantGate,
    Squeezing45Gate,
    SqueezingGate,
    TwoModeShearGate,
    create_compact_diagram,
    expand_all,
)
from cvzx.backends.nx.graph import to_diagram, to_graph
from cvzx.visualization.core import DiagramVisualizer, VisualizerConfig, visualize

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


def compact_gates_test():  # ruff: ignore[too-many-locals, too-many-statements]
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
        ShearXInvariantGate(kappa=0.3),
        ShearPInvariantGate(eta=-0.3),
        Squeezing45Gate(theta=0.9),
        ArbitraryGate(alpha=0.3, beta=-0.6, lam=0.2),
        TwoModeShearGate(a=0.4, b=-0.2),
        MeasurementGate(theta=0.7),
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
        ShearXInvariantGate(kappa=a, parametric=True),
        ShearPInvariantGate(eta=b, parametric=True),
        Squeezing45Gate(theta=theta, parametric=True),
        ArbitraryGate(alpha=phi, beta=theta, lam=a, parametric=True),
        TwoModeShearGate(a=c, b=d, parametric=True),
        MeasurementGate(theta=phi, parametric=True),
    ]

    for gate in gates_parametric:
        filename = f"{gate.__class__.__name__}_parametric_compact.png"
        save_and_close(gate, filename, f"{gate!r} (parametric compact)")

    # 3. Test CompactDiagram.compose with connectivity
    print("\n" + "=" * 60)
    print("Testing CompactDiagram.compose with connectivity")
    print("=" * 60)

    # Create some compact gates (numeric and parametric)
    D = DisplacementGate(alpha=1.0 + 0.5j)  # ruff: ignore[non-lowercase-variable-in-function]
    D_sym = DisplacementGate(alpha=alpha_sym, parametric=True)  # ruff: ignore[non-lowercase-variable-in-function]
    R = PhaseRotationGate(theta=np.pi / 4)  # ruff: ignore[non-lowercase-variable-in-function]
    R_sym = PhaseRotationGate(theta=theta, parametric=True)  # ruff: ignore[non-lowercase-variable-in-function]
    Sq = SqueezingGate(tau=0.5)  # ruff: ignore[non-lowercase-variable-in-function]
    Sq_sym = SqueezingGate(tau=a, parametric=True)  # ruff: ignore[non-lowercase-variable-in-function]
    BS = BeamsplitterGate(theta=np.pi / 4)  # ruff: ignore[non-lowercase-variable-in-function]
    BS_sym = BeamsplitterGate(theta=phi, parametric=True)  # ruff: ignore[non-lowercase-variable-in-function]
    CS = ControlledSumGate(gain=2.0, control=2, target=1)  # ruff: ignore[non-lowercase-variable-in-function]
    CS_sym = ControlledSumGate(gain=c, control=2, target=1, parametric=True)  # ruff: ignore[non-lowercase-variable-in-function]
    CZ = ControlledZGate(gain=1.0)  # ruff: ignore[non-lowercase-variable-in-function]
    CZ_sym = ControlledZGate(gain=e, parametric=True)  # ruff: ignore[non-lowercase-variable-in-function]

    # Test 1: Simple composition with non-trivial connectivity (1 input, 1 output)
    comp1 = D.compose(R, connectivity={0: 0})
    save_and_close(comp1, "Compact_compose_D_R_conn.png", "D ∘ R with connectivity (0→0)")

    # Test 1b: Parametric version
    comp1_sym = D_sym.compose(R_sym, connectivity={0: 0})
    save_and_close(comp1_sym, "Compact_compose_D_sym_R_sym_conn.png", "D(a) ∘ R(θ) with connectivity (0→0)")

    # Test 2: Composition with swapped connectivity (if arities allow)
    D_tensor = D.tensor(D)  # ruff: ignore[non-lowercase-variable-in-function]
    R_tensor = R.tensor(R)  # ruff: ignore[non-lowercase-variable-in-function]

    comp2 = D_tensor.compose(R_tensor, connectivity={0: 1, 1: 0})
    save_and_close(
        comp2, "Compact_compose_tensor_swapped_conn.png", "(D⊗D) ∘ (R⊗R) with swapped connectivity (0→1, 1→0)"
    )

    # Test 2b: Parametric version
    D_sym_tensor = D_sym.tensor(D_sym)  # ruff: ignore[non-lowercase-variable-in-function]
    R_sym_tensor = R_sym.tensor(R_sym)  # ruff: ignore[non-lowercase-variable-in-function]
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
    BS_compact = BS  # ruff: ignore[non-lowercase-variable-in-function]

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
    R_tensor = R.tensor(R)  # ruff: ignore[non-lowercase-variable-in-function]
    Sq_tensor = Sq.tensor(Sq)  # ruff: ignore[non-lowercase-variable-in-function]

    comp5 = R_tensor.compose(Sq_tensor, connectivity={0: 1, 1: 0})
    save_and_close(
        comp5,
        "Compact_compose_R_tensor_Sq_tensor_reverse_conn.png",
        "(R⊗R) ∘ (Sq⊗Sq) with reverse connectivity (0→1, 1→0)",
    )

    # Test 5b: Parametric version
    R_sym_tensor = R_sym.tensor(R_sym)  # ruff: ignore[non-lowercase-variable-in-function]
    Sq_sym_tensor = Sq_sym.tensor(Sq_sym)  # ruff: ignore[non-lowercase-variable-in-function]
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
        ShearXInvariantGate(kappa=0.3),
        ShearPInvariantGate(eta=-0.3),
        Squeezing45Gate(theta=0.9),
        ArbitraryGate(alpha=0.3, beta=-0.6, lam=0.2),
        TwoModeShearGate(a=0.4, b=-0.2),
        MeasurementGate(theta=0.7),
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
        ShearXInvariantGate(kappa=a, parametric=True),
        ShearPInvariantGate(eta=b, parametric=True),
        Squeezing45Gate(theta=theta, parametric=True),
        ArbitraryGate(alpha=phi, beta=theta, lam=a, parametric=True),
        TwoModeShearGate(a=a, b=b, parametric=True),
        MeasurementGate(theta=phi, parametric=True),
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
    R = PhaseRotationGate(theta=np.pi / 4)  # ruff: ignore[non-lowercase-variable-in-function]
    Sq = SqueezingGate(tau=0.5)  # ruff: ignore[non-lowercase-variable-in-function]
    circuit = R.compose(Sq)

    visualizer = DiagramVisualizer(config)
    fig = visualizer.visualize(circuit, title="Custom Config: Sq ∘ R")

    filepath = OUTPUT_DIR / "Custom_config.png"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved: {filepath}")

    # Parametric version
    R_sym = PhaseRotationGate(theta=theta, parametric=True)  # ruff: ignore[non-lowercase-variable-in-function]
    Sq_sym = SqueezingGate(tau=a, parametric=True)  # ruff: ignore[non-lowercase-variable-in-function]
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


def create_compact_diagram_test():  # ruff: ignore[too-many-locals, too-many-statements]
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

    nested_composition_block: Diagram = CompositionDiagram([q_spider_3x3, q_spider_3x3])
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
    nested_composition_block_sym: Diagram = CompositionDiagram([q_spider_3x3_sym, q_spider_3x3_sym])
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
        ShearXInvariantGate(kappa=0.3),
        ShearPInvariantGate(eta=-0.3),
        Squeezing45Gate(theta=0.9),
        ArbitraryGate(alpha=0.3, beta=-0.6, lam=0.2),
        TwoModeShearGate(a=0.4, b=-0.2),
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

    # MeasurementGate is an effect (1-in-0-out), not a unitary gate: its
    # conjugate is a state (0-in-1-out) rather than another MeasurementGate,
    # so it does not fit the "same type after double conjugate" pattern
    # above and is checked separately in mqc3_gates_numeric_verification_test.
    print()
    meas = MeasurementGate(theta=0.7)
    meas_conj = meas.conjugate()
    save_and_close(meas_conj, "MeasurementGate_conjugate.png", f"{meas!r}†")
    print(f"MeasurementGate conjugate (state, not effect): {meas_conj}")

    # Parametric gates
    gates_parametric = [
        DisplacementGate(alpha=alpha_sym, parametric=True),
        PhaseRotationGate(theta=theta, parametric=True),
        SqueezingGate(tau=a, parametric=True),
        ControlledSumGate(gain=b, parametric=True),
        ControlledZGate(gain=phi, parametric=True),
        BeamsplitterGate(theta=theta, parametric=True),
        CubicPhaseGate(gamma=gamma_sym, parametric=True),
        ShearXInvariantGate(kappa=a, parametric=True),
        ShearPInvariantGate(eta=b, parametric=True),
        Squeezing45Gate(theta=theta, parametric=True),
        ArbitraryGate(alpha=phi, beta=theta, lam=a, parametric=True),
        TwoModeShearGate(a=a, b=b, parametric=True),
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


def feedforward_test():  # ruff: ignore[too-many-locals]
    """Test feedforward display."""
    zero_phase = ZxPoly({})
    id_q = QSpider(1, 1, zero_phase)
    tensor1 = TensorDiagram([id_q, QSpider(0, 2, zero_phase)])
    contract = ContractedDiagram(PSpider(1, 2, zero_phase), QSpider(2, 1, zero_phase), [1], [0], [], [])
    tensor2 = TensorDiagram([contract, id_q])
    m1, m2 = symbols("m1 m2", real=True)
    meas_diag1 = PSpider(1, 0, ZxPoly({1: -m1}), True)
    meas_diag2 = QSpider(1, 0, ZxPoly({1: -m2}), True)
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


def _spider_quadratic_coef(spider: QSpider | PSpider) -> float:
    """Extract the coefficient of x^2 from a QSpider/PSpider's phase (0 if absent)."""
    return spider.phase.coeffs.get(2, 0.0)


def _apply_1mode_cvzx(diagram: Diagram, x: float, p: float) -> tuple[float, float]:
    """Walk a 1-in-1-out diagram and return its Heisenberg-transformed (x, p).

    The diagram is built from QSpider/PSpider (quadratic phase),
    CompositionDiagram, and the 1-mode compact gates defined in this
    module. Returns (x, p) in cvzx's own convention (i.e. following
    each spider's matrix directly, with no mqc3 sign correction).
    """
    if isinstance(diagram, QSpider):
        c = _spider_quadratic_coef(diagram)
        return x, p + 2 * c * x
    if isinstance(diagram, PSpider):
        c = _spider_quadratic_coef(diagram)
        return x + 2 * c * p, p
    if isinstance(diagram, CompositionDiagram):
        for d in diagram.diagrams:
            x, p = _apply_1mode_cvzx(d, x, p)
        return x, p
    assert isinstance(diagram, CompactDiagram)
    return _apply_1mode_cvzx(diagram.expand(), x, p)


def _apply_1mode_mqc3(diagram: Diagram, x: float, p: float) -> tuple[float, float]:  # ruff: ignore[too-many-return-statements]
    """Like `_apply_1mode_cvzx`, but in mqc3's own rotation convention.

    Interprets `PhaseRotationGate`, `Fourier`, and `FourierInv` nodes
    using mqc3's OWN rotation convention (`cos/sin` matrix), for
    directly checking what an expanded gate does to mqc3-convention
    (x, p).
    """
    if isinstance(diagram, QSpider):
        c = _spider_quadratic_coef(diagram)
        return x, p + 2 * c * x
    if isinstance(diagram, PSpider):
        c = _spider_quadratic_coef(diagram)
        return x + 2 * c * p, p
    if isinstance(diagram, CompositionDiagram):
        for d in diagram.diagrams:
            x, p = _apply_1mode_mqc3(d, x, p)
        return x, p
    if isinstance(diagram, PhaseRotationGate):
        phi = -diagram.theta  # cvzx PhaseRotationGate(theta) == mqc3 R(-theta)
        return np.cos(phi) * x - np.sin(phi) * p, np.sin(phi) * x + np.cos(phi) * p
    if isinstance(diagram, Fourier):
        return -p, x  # mqc3 R(pi/2)
    if isinstance(diagram, FourierInv):
        return p, -x  # mqc3 R(-pi/2)
    assert isinstance(diagram, CompactDiagram)
    return _apply_1mode_mqc3(diagram.expand(), x, p)


def mqc3_gates_numeric_verification_test() -> None:  # ruff: ignore[complex-structure, too-many-locals, too-many-statements]
    """Numerically verify each mqc3-derived gate's `expand()` against mqc3's own matrix definitions.

    Checks each gate's `expand()` against the exact Heisenberg matrix
    mqc3's own docstrings define for it. Unlike the rest of this file,
    this uses real `assert`s rather than printed checkmarks, since it
    is checking mathematical correctness rather than just "did this
    render."
    """
    print("Testing mqc3-derived gates against mqc3's own matrix definitions...")
    rng = random.Random(12345)

    def rand() -> float:
        return rng.uniform(-2.0, 2.0)

    def rand_angle() -> float:
        return rng.uniform(-1.4, 1.4)  # stay clear of PhaseRotationGate's pi/2 exclusion

    def mqc3_r(phi: float, x: float, p: float) -> tuple[float, float]:
        return np.cos(phi) * x - np.sin(phi) * p, np.sin(phi) * x + np.cos(phi) * p

    def mqc3_s(lam: float, x: float, p: float) -> tuple[float, float]:
        return np.exp(lam) * x, np.exp(-lam) * p

    # ShearXInvariant(kappa): x invariant, p -> p + 2*kappa*x
    for _ in range(10):
        kappa = rand()
        xv, pv = rand(), rand()
        xo, po = _apply_1mode_cvzx(ShearXInvariantGate(kappa).expand(), xv, pv)
        assert abs(xo - xv) < 1e-9
        assert abs(po - (pv + 2 * kappa * xv)) < 1e-9

    # ShearPInvariant(eta): p invariant, x -> x + 2*eta*p
    for _ in range(10):
        eta = rand()
        xv, pv = rand(), rand()
        xo, po = _apply_1mode_cvzx(ShearPInvariantGate(eta).expand(), xv, pv)
        assert abs(po - pv) < 1e-9
        assert abs(xo - (xv + 2 * eta * pv)) < 1e-9

    # Arbitrary(alpha, beta, lam) = R(alpha) . S(lam) . R(beta), rightmost applied first
    for _ in range(20):
        alpha, beta, lam = rand_angle(), rand_angle(), rand()
        xv, pv = rand(), rand()
        xo, po = _apply_1mode_mqc3(ArbitraryGate(alpha, beta, lam).expand(), xv, pv)
        xe, pe = mqc3_r(beta, xv, pv)
        xe, pe = mqc3_s(lam, xe, pe)
        xe, pe = mqc3_r(alpha, xe, pe)
        assert abs(xo - xe) < 1e-7
        assert abs(po - pe) < 1e-7

    # Squeezing45(theta) = R(-pi/4) . S_V(cot theta) . R(pi/4), rightmost applied first
    for _ in range(20):
        theta = rng.uniform(0.2, 1.3)
        xv, pv = rand(), rand()
        xo, po = _apply_1mode_mqc3(Squeezing45Gate(theta).expand(), xv, pv)
        c = 1 / np.tan(theta)
        xe, pe = mqc3_r(np.pi / 4, xv, pv)
        xe, pe = xe / c, c * pe
        xe, pe = mqc3_r(-np.pi / 4, xe, pe)
        assert abs(xo - xe) < 1e-7
        assert abs(po - pe) < 1e-7

    # TwoModeShear(a, b): p1' = 2a*x1 + b*x2 + p1, p2' = b*x1 + 2a*x2 + p2, x invariant
    for _ in range(10):
        a, b = rand(), rand()
        x1, p1, x2, p2 = rand(), rand(), rand(), rand()
        expanded = TwoModeShearGate(a, b).expand()
        tensor, cz = expanded.diagrams
        assert isinstance(tensor, TensorDiagram)
        shear1, shear2 = tensor.diagrams
        x1o, p1o = _apply_1mode_cvzx(shear1, x1, p1)
        x2o, p2o = _apply_1mode_cvzx(shear2, x2, p2)
        assert isinstance(cz, ControlledZGate)
        g = cast("float", cz.gain)
        p1o, p2o = p1o - g * x2o, p2o - g * x1o
        assert abs(x1o - x1) < 1e-9
        assert abs(x2o - x2) < 1e-9
        assert abs(p1o - (2 * a * x1 + b * x2 + p1)) < 1e-9
        assert abs(p2o - (b * x1 + 2 * a * x2 + p2)) < 1e-9

    # Measurement(theta) measures x*sin(theta) + p*cos(theta), including the
    # theta = 0 (measure p) and theta = pi (measure -p) cases that force the
    # Fourier/FourierInv fallback inside MeasurementGate._rotation_diagram.
    test_thetas = [0.0, np.pi, -np.pi, 2 * np.pi, *[rand_angle() * 3 for _ in range(15)]]
    for theta in test_thetas:
        rot, effect = MeasurementGate(theta).expand().diagrams
        assert effect.num_inputs == 1
        assert effect.num_outputs == 0
        xv, pv = rand(), rand()
        x_after, _p_after = _apply_1mode_mqc3(rot, xv, pv)
        expected = np.sin(theta) * xv + np.cos(theta) * pv
        assert abs(x_after - expected) < 1e-7, (theta, x_after, expected)

    # MeasurementGate.conjugate(): arity flips from effect (1-in-0-out) to
    # state (0-in-1-out), including at the Fourier/FourierInv angles.
    for theta in [0.0, np.pi / 3, np.pi]:
        conj = MeasurementGate(theta).conjugate()
        assert conj.num_inputs == 0
        assert conj.num_outputs == 1

    print("✅ All mqc3-derived gates verified against mqc3's own matrix definitions")


def feedforward_params_all_gates_test():
    """Every parametrized gate accepts `feedforward`/`measurement_ids` with the same validation.

    Follows the same pattern `DisplacementGate` established:
    `feedforward=True` requires a non-empty `measurement_ids` set.

    Raises
    ------
    AssertionError
        If any gate class fails to reject `feedforward=True` with no
        (or empty) `measurement_ids`.
    """
    print("Testing feedforward/measurement_ids on all parametrized gates...")
    m = symbols("m", real=True)
    meas_leaf = QSpider(1, 0, ZxPoly({1: m}), True)

    gate_specs = [
        (DisplacementGate, {"alpha": m, "parametric": True}),
        (PhaseRotationGate, {"theta": m, "parametric": True}),
        (SqueezingGate, {"tau": m, "parametric": True}),
        (ControlledSumGate, {"gain": m, "control": 1, "target": 2, "parametric": True}),
        (ControlledZGate, {"gain": m, "parametric": True}),
        (BeamsplitterGate, {"theta": m, "parametric": True}),
        (CubicPhaseGate, {"gamma": m, "parametric": True}),
        (ShearXInvariantGate, {"kappa": m, "parametric": True}),
        (ShearPInvariantGate, {"eta": m, "parametric": True}),
        (ArbitraryGate, {"alpha": m, "beta": 0.1, "lam": 0.2, "parametric": True}),
        (Squeezing45Gate, {"theta": m, "parametric": True}),
        (TwoModeShearGate, {"a": m, "b": 0.1, "parametric": True}),
        (MeasurementGate, {"theta": m, "parametric": True}),
    ]

    for cls, kwargs in gate_specs:
        gate = cls(feedforward=True, measurement_ids={meas_leaf.id}, **kwargs)
        assert gate.feedforward is True
        assert gate.measurement_ids == {meas_leaf.id}

        default_gate = cls(**kwargs)
        assert default_gate.feedforward is False
        assert default_gate.measurement_ids is None

        try:
            cls(feedforward=True, **kwargs)
        except ValueError:
            pass
        else:
            msg = f"{cls.__name__} should reject feedforward=True with no measurement_ids"
            raise AssertionError(msg)

        try:
            cls(feedforward=True, measurement_ids=set(), **kwargs)
        except ValueError:
            pass
        else:
            msg = f"{cls.__name__} should reject feedforward=True with empty measurement_ids"
            raise AssertionError(msg)

    print(f"✅ feedforward/measurement_ids verified on all {len(gate_specs)} gates")


class _Feedforwardable(Protocol):
    """Structural type for the `feedforward`/`measurement_ids` fields every concrete gate class declares."""

    feedforward: bool
    measurement_ids: set[int] | None


def nx_graph_roundtrip_test():
    """Every gate class must survive `to_graph()` then `to_diagram()` unchanged.

    Same class, same parameters, same feedforward/measurement_ids --
    since `optimize()` and friends round-trip diagrams through the
    graph representation routinely. This caught a real pre-existing
    gap (`CubicPhaseGate` was entirely unhandled by
    `_reconstruct_proper_node` and would raise) as well as the
    mqc3-derived gates needing the same wiring.
    """
    print("Testing nx_graph.py to_graph()/to_diagram() round-trip for all gates...")
    m = symbols("m", real=True)
    meas_leaf = QSpider(1, 0, ZxPoly({1: m}), True)

    gates = [
        DisplacementGate(alpha=0.5 + 0.2j),
        PhaseRotationGate(theta=0.4, feedforward=True, measurement_ids={meas_leaf.id}),
        SqueezingGate(tau=0.6),
        ControlledSumGate(gain=1.5, control=1, target=2),
        ControlledZGate(gain=0.8),
        BeamsplitterGate(theta=np.pi / 6),
        CubicPhaseGate(gamma=0.1),
        ShearXInvariantGate(kappa=0.3),
        ShearPInvariantGate(eta=-0.4),
        Squeezing45Gate(theta=0.9),
        ArbitraryGate(alpha=0.3, beta=-0.6, lam=0.2),
        TwoModeShearGate(a=0.5, b=-0.2),
        MeasurementGate(theta=0.7),
    ]

    for gate in gates:
        graph = to_graph(gate)
        rebuilt = to_diagram(graph)
        assert type(rebuilt) is type(gate), f"{gate!r} round-tripped to {rebuilt!r}"
        if hasattr(gate, "feedforward"):
            ff_gate = cast("_Feedforwardable", gate)
            ff_rebuilt = cast("_Feedforwardable", rebuilt)
            assert ff_rebuilt.feedforward == ff_gate.feedforward, gate
            assert ff_rebuilt.measurement_ids == ff_gate.measurement_ids, gate

    print(f"✅ nx_graph.py round-trip verified for all {len(gates)} gates")


def run_all_tests():
    """Run all test functions."""
    compact_gates_test()
    expanded_gates_test()
    with_custom_config_test()
    compact_diagram_label_validation_test()
    conjugate_gates_test()
    create_compact_diagram_test()
    feedforward_test()
    mqc3_gates_numeric_verification_test()
    feedforward_params_all_gates_test()
    nx_graph_roundtrip_test()


if __name__ == "__main__":
    run_all_tests()

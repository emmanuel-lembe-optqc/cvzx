"""Partial trace tests."""

import pytest

from mqc3.zx.base_gates import (
    CompositionDiagram,
    ContractedDiagram,
    Fourier,
    Fourier2,
    FourierInv,
    PSpider,
    QSpider,
    Swap,
    TensorDiagram,
    ZxPoly,
)


class TestTensorDiagramPartialTrace:
    """Test suite for TensorDiagram.partial_trace method."""

    def test_basic_contraction(self):
        """Test basic partial trace of two diagrams."""
        d1 = QSpider(1, 1, phase=ZxPoly({}))
        d2 = PSpider(1, 1, phase=ZxPoly({}))

        tensor = TensorDiagram([d1, d2])
        tensor.partial_trace([(0, [0], [0]), (1, [0], [0])])

        assert len(tensor.diagrams) == 1
        assert isinstance(tensor.diagrams[0], ContractedDiagram)

        contracted = tensor.diagrams[0]
        assert contracted.num_inputs == 0
        assert contracted.num_outputs == 0

    def test_simple_connection(self):
        """Test partial trace where only some wires are connected."""
        d1 = QSpider(2, 2, phase=ZxPoly({}))
        d2 = Swap()

        tensor = TensorDiagram([d1, d2])
        tensor.partial_trace([(0, [0], [1]), (1, [1], [0])])

        assert len(tensor.diagrams) == 1
        contracted = tensor.diagrams[0]
        assert isinstance(contracted, ContractedDiagram)
        assert contracted.num_inputs == 2
        assert contracted.num_outputs == 2

    def test_multiple_diagrams(self):
        """Test partial trace with multiple diagrams in a tensor diagram."""
        d1 = QSpider(4, 3, phase=ZxPoly({}))
        d2 = PSpider(3, 4, phase=ZxPoly({}))
        d3 = Swap()

        tensor = TensorDiagram([d1, d2, d3])
        tensor.partial_trace([(0, [0, 1], [1, 2]), (1, [2, 3], [0, 2])])

        assert len(tensor.diagrams) == 2
        assert isinstance(tensor.diagrams[0], ContractedDiagram)
        assert isinstance(tensor.diagrams[1], Swap)

    def test_with_composition_diagrams(self):
        """Test partial trace with nested composition diagrams."""
        q1 = QSpider(3, 3, phase=ZxPoly({}))
        q2 = TensorDiagram([Fourier(), Fourier2(), FourierInv()])
        p1 = PSpider(4, 4, phase=ZxPoly({}))
        p2 = TensorDiagram([Swap(), PSpider(2, 2, phase=ZxPoly({}))])

        comp1 = CompositionDiagram([q1, q2])
        comp2 = CompositionDiagram([p1, p2])

        tensor = TensorDiagram([comp1, comp2])
        tensor.partial_trace([(0, [0, 1, 2], [0, 2]), (1, [0, 2], [1, 2, 3])])

        assert len(tensor.diagrams) == 1
        contracted = tensor.diagrams[0]
        assert contracted.num_inputs == 2
        assert contracted.num_outputs == 2
        assert isinstance(tensor.diagrams[0], ContractedDiagram)

    def test_preserves_diagram_order(self):
        """Test that partial trace preserves remaining diagram order."""
        d1 = QSpider(2, 2, phase=ZxPoly({}))
        d2 = PSpider(2, 2, phase=ZxPoly({}))
        d3 = Swap()
        d4 = Fourier()

        tensor = TensorDiagram([d1, d2, d3, d4])

        tensor.partial_trace([(1, [0], [0]), (2, [1], [1])])

        assert len(tensor.diagrams) == 3
        assert isinstance(tensor.diagrams[0], QSpider)
        assert isinstance(tensor.diagrams[1], ContractedDiagram)
        assert isinstance(tensor.diagrams[2], Fourier)


class TestTensorDiagramPartialTraceExceptions:
    """Test exception handling in TensorDiagram.partial_trace."""

    def test_empty_diagram_pairs(self):
        """Test that empty diagram_pairs raises ValueError."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="diagram_pairs cannot be empty"):
            tensor.partial_trace([])

    def test_first_index_out_of_range(self):
        """Test that first diagram index out of range raises ValueError."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="First diagram index 2 out of range"):
            tensor.partial_trace([(2, [0], [0]), (3, [0], [0])])

    def test_second_index_out_of_range(self):
        """Test that second diagram index out of range raises ValueError."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="Second diagram index 5 out of range"):
            tensor.partial_trace([(0, [0], [0]), (5, [0], [0])])

    def test_non_consecutive_diagrams(self):
        """Test that non-consecutive diagrams raise ValueError."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        d3 = Swap()
        tensor = TensorDiagram([d1, d2, d3])

        with pytest.raises(ValueError, match="are not consecutive"):
            tensor.partial_trace([(0, [0], [0]), (2, [0], [0])])

    def test_duplicate_output_wires_first(self):
        """Test that duplicate output wires in first diagram raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="Output wires of the first diagram.*contain duplicate indices"):  # noqa: RUF043
            tensor.partial_trace([(0, [0, 0], [0, 1]), (1, [0, 1], [0, 1])])

    def test_duplicate_input_wires_first(self):
        """Test that duplicate input wires in first diagram raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="Input wires of the first diagram.*contain duplicate indices"):  # noqa: RUF043
            tensor.partial_trace([(0, [0, 1], [0, 0]), (1, [0, 1], [0, 1])])

    def test_duplicate_output_wires_second(self):
        """Test that duplicate output wires in second diagram raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="Output wires of the second diagram.*contain duplicate indices"):  # noqa: RUF043
            tensor.partial_trace([(0, [0, 1], [0, 1]), (1, [0, 0], [0, 1])])

    def test_duplicate_input_wires_second(self):
        """Test that duplicate input wires in second diagram raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="Input wires of the second diagram.*contain duplicate indices"):  # noqa: RUF043
            tensor.partial_trace([(0, [0, 1], [0, 1]), (1, [0, 1], [0, 0])])

    def test_mismatched_i_lengths(self):
        """Test that mismatched I1 and I2 lengths raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="I1 length.*must equal I2 length"):  # noqa: RUF043
            tensor.partial_trace([
                (0, [0, 1], [0]),
                (1, [0], [1]),
            ])

    def test_mismatched_j_lengths(self):
        """Test that mismatched J1 and J2 lengths raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="J1 length.*must equal J2 length"):  # noqa: RUF043
            tensor.partial_trace([
                (0, [0], [0, 1]),
                (1, [0], [0]),
            ])

    def test_output_wire_out_of_range_first(self):
        """Test output wire index out of range for first diagram."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="Output wire 5 out of range"):
            tensor.partial_trace([(0, [5], [0]), (1, [0], [0])])

    def test_input_wire_out_of_range_first(self):
        """Test input wire index out of range for first diagram."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="Input wire 5 out of range"):
            tensor.partial_trace([(0, [0], [5]), (1, [0], [0])])

    def test_output_wire_out_of_range_second(self):
        """Test output wire index out of range for second diagram."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="Input wire 5 out of range"):
            tensor.partial_trace([(0, [0], [0]), (1, [5], [0])])

    def test_input_wire_out_of_range_second(self):
        """Test input wire index out of range for second diagram."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ValueError, match="Output wire 5 out of range"):
            tensor.partial_trace([(0, [0], [0]), (1, [0], [5])])

    def test_first_and_second_indices_swapped(self):
        """Test that swapping indices correctly handles order."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        # Swap the order of indices - should still work
        tensor.partial_trace([(1, [0], [0]), (0, [0], [0])])

        # The implementation sorts by first index, so indices 0 and 1
        # will be reordered to (0, [0], [0]) and (1, [0], [0])
        assert len(tensor.diagrams) == 1
        assert isinstance(tensor.diagrams[0], ContractedDiagram)


class TestTensorDiagramPartialTraceEdgeCases:
    """Test edge cases for TensorDiagram.partial_trace."""

    def test_contract_all_wires(self):
        """Test contracting all wires of two diagrams."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))

        tensor = TensorDiagram([d1, d2])
        tensor.partial_trace([(0, [0], [0]), (1, [0], [0])])

        assert len(tensor.diagrams) == 1
        contracted = tensor.diagrams[0]
        assert contracted.num_inputs == 0
        assert contracted.num_outputs == 0

    def test_complex_nested_structure(self):
        """Test partial trace with complex nested structure."""
        # Create a complex diagram with multiple levels
        q1 = QSpider(2, 2, ZxPoly({}))
        q2 = QSpider(2, 2, ZxPoly({}))
        p1 = PSpider(2, 2, ZxPoly({}))

        comp = CompositionDiagram([q1, p1])
        tensor = TensorDiagram([comp, q2])

        # Contract comp with q2
        tensor.partial_trace([(0, [0, 1], [0, 1]), (1, [0, 1], [0, 1])])

        assert len(tensor.diagrams) == 1
        contracted = tensor.diagrams[0]
        assert isinstance(contracted, ContractedDiagram)
        # All wires connected, so no external ports
        assert contracted.num_inputs == 0
        assert contracted.num_outputs == 0

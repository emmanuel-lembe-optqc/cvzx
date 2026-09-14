"""Unit test for base_gates."""

import unittest

import pytest
from sympy import symbols

from cvzx.base_gates import (
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
    flatten_composition,
)
from cvzx.exceptions import ArityMismatchError, InvalidSymbolError


class TestFlattenComposition(unittest.TestCase):
    """Test suite for RewriteRule.flatten_composition."""

    def setUp(self):
        """Create common objects used in many tests."""
        self.zero = ZxPoly({})
        self.phase = ZxPoly({2: 2})

        self.q1 = QSpider(1, 1, self.zero)
        self.q2 = QSpider(1, 1, self.zero)
        self.q3 = QSpider(1, 1, self.zero)
        self.p1 = PSpider(1, 1, self.zero)
        self.p2 = PSpider(1, 1, self.zero)
        self.fourier = Fourier()
        self.fourier2 = Fourier2()
        self.swap = Swap()

    def test_single_element_returns_element(self):
        """Test that a single-element composition returns the element."""
        comp = CompositionDiagram([self.fourier])
        result = flatten_composition(comp)
        assert result == self.fourier

    def test_two_elements_unchanged(self):
        """Test that a two-element composition remains unchanged."""
        comp = CompositionDiagram([self.fourier, self.q1])
        result = flatten_composition(comp)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.q1

    def test_flatten_single_nested(self):
        """Test flattening a single nested composition."""
        inner = CompositionDiagram([self.fourier, self.q1])
        outer = CompositionDiagram([inner])
        result = flatten_composition(outer)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.q1

    def test_flatten_multiple_nested(self):
        """Test flattening multiple nested compositions."""
        inner1 = CompositionDiagram([self.fourier, self.q1])
        inner2 = CompositionDiagram([self.p1, self.q2])
        outer = CompositionDiagram([inner1, inner2])
        result = flatten_composition(outer)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 4
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.q1
        assert result.diagrams[2] == self.p1
        assert result.diagrams[1] == self.q2

    def test_flatten_deeply_nested(self):
        """Test flattening deeply nested compositions."""
        inner_inner = CompositionDiagram([self.fourier, self.q1])
        inner = CompositionDiagram([inner_inner, self.p1])
        outer = CompositionDiagram([inner, self.q2])
        result = flatten_composition(outer)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 4
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.q1
        assert result.diagrams[2] == self.p1
        assert result.diagrams[1] == self.q2

    def test_flatten_with_non_composition_elements(self):
        """Test flattening with non-composition elements inside."""
        inner = CompositionDiagram([self.q1, self.p1])
        outer = CompositionDiagram([self.fourier, inner, self.q2])
        result = flatten_composition(outer)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 4
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.q1
        assert result.diagrams[2] == self.p1
        assert result.diagrams[1] == self.q2

    def test_does_not_flatten_tensor_diagram(self):
        """Test that TensorDiagram == NOT flattened."""
        inner = CompositionDiagram([self.fourier, self.q1])
        tensor = TensorDiagram([inner, self.p1])
        result = flatten_composition(tensor)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        # The composition inside the tensor should NOT be flattened
        assert isinstance(result.diagrams[0], CompositionDiagram)
        assert len(result.diagrams[0].diagrams) == 2
        assert result.diagrams[0].diagrams[0] == self.fourier
        assert result.diagrams[0].diagrams[1] == self.q1
        assert result.diagrams[1] == self.p1

    def test_flattens_composition_inside_tensor(self):
        """Test that composition inside TensorDiagram == flattened only if it's a single element."""
        inner = CompositionDiagram([self.fourier])
        tensor = TensorDiagram([inner, self.q1])
        result = flatten_composition(tensor)
        assert isinstance(result, TensorDiagram)
        assert len(result.diagrams) == 2
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.q1

    def test_flattens_composition_inside_contracted(self):
        """Test that compositions inside ContractedDiagram are flattened if they are single-element."""
        comp1 = CompositionDiagram([self.fourier])
        comp2 = CompositionDiagram([self.q1])
        contracted = ContractedDiagram(comp1, comp2, [0], [0], [], [])
        result = flatten_composition(contracted)
        assert isinstance(result, ContractedDiagram)
        assert result.first == self.fourier
        assert result.second == self.q1

    def test_does_not_flatten_multi_element_contracted(self):
        """Test that multi-element compositions inside ContractedDiagram are NOT flattened."""
        comp1 = CompositionDiagram([self.fourier, self.q1])
        comp2 = CompositionDiagram([self.p1, self.q2])
        contracted = ContractedDiagram(comp1, comp2, [0], [0], [], [])
        result = flatten_composition(contracted)
        assert isinstance(result, ContractedDiagram)
        assert result.first == comp1
        assert result.second == comp2

    def test_flattens_deep_nested_with_tensor(self):
        """Test flattening compositions inside a diagram that contains both Composition and Tensor."""
        inner_comp = CompositionDiagram([self.fourier, self.q1])
        tensor = TensorDiagram([inner_comp, self.p1])
        outer_comp = CompositionDiagram([tensor, CompositionDiagram([self.swap, self.swap])])
        result = flatten_composition(outer_comp)

        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 3

        # First element should be the Tensor (with inner composition still nested)
        assert isinstance(result.diagrams[0], TensorDiagram)
        assert len(result.diagrams[0].diagrams) == 2
        assert isinstance(result.diagrams[0].diagrams[0], CompositionDiagram)
        assert len(result.diagrams[0].diagrams[0].diagrams) == 2
        assert result.diagrams[0].diagrams[0].diagrams[0] == self.fourier
        assert result.diagrams[0].diagrams[0].diagrams[1] == self.q1
        assert result.diagrams[0].diagrams[1] == self.p1

        # Second element should be the flattened
        assert result.diagrams[1] == self.swap
        assert result.diagrams[2] == self.swap

    def test_flatten_composition_with_single_nested_composition_inside(self):
        """Test flattening a composition with a single-element nested composition."""
        inner = CompositionDiagram([self.q1])
        outer = CompositionDiagram([self.fourier, inner, self.p1])
        result = flatten_composition(outer)
        assert isinstance(result, CompositionDiagram)
        assert len(result.diagrams) == 3
        assert result.diagrams[0] == self.fourier
        assert result.diagrams[1] == self.q1
        assert result.diagrams[2] == self.p1

    def test_identity_diagram_not_flattened(self):
        """Test that proper diagrams are returned as-is."""
        result = flatten_composition(self.fourier)
        assert result == self.fourier

        result = flatten_composition(self.q1)
        assert result == self.q1

        result = flatten_composition(self.swap)
        assert result == self.swap


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

        with pytest.raises(ArityMismatchError, match="diagram_pairs cannot be empty"):
            tensor.partial_trace([])

    def test_first_index_out_of_range(self):
        """Test that first diagram index out of range raises ValueError."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="First diagram index 2 out of range"):
            tensor.partial_trace([(2, [0], [0]), (3, [0], [0])])

    def test_second_index_out_of_range(self):
        """Test that second diagram index out of range raises ValueError."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="Second diagram index 5 out of range"):
            tensor.partial_trace([(0, [0], [0]), (5, [0], [0])])

    def test_non_consecutive_diagrams(self):
        """Test that non-consecutive diagrams raise ValueError."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        d3 = Swap()
        tensor = TensorDiagram([d1, d2, d3])

        with pytest.raises(ArityMismatchError, match="are not consecutive"):
            tensor.partial_trace([(0, [0], [0]), (2, [0], [0])])

    def test_duplicate_output_wires_first(self):
        """Test that duplicate output wires in first diagram raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="Output wires of the first diagram.*contain duplicate indices"):  # ruff: ignore[pytest-raises-ambiguous-pattern]
            tensor.partial_trace([(0, [0, 0], [0, 1]), (1, [0, 1], [0, 1])])

    def test_duplicate_input_wires_first(self):
        """Test that duplicate input wires in first diagram raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="Input wires of the first diagram.*contain duplicate indices"):  # ruff: ignore[pytest-raises-ambiguous-pattern]
            tensor.partial_trace([(0, [0, 1], [0, 0]), (1, [0, 1], [0, 1])])

    def test_duplicate_output_wires_second(self):
        """Test that duplicate output wires in second diagram raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="Output wires of the second diagram.*contain duplicate indices"):  # ruff: ignore[pytest-raises-ambiguous-pattern]
            tensor.partial_trace([(0, [0, 1], [0, 1]), (1, [0, 0], [0, 1])])

    def test_duplicate_input_wires_second(self):
        """Test that duplicate input wires in second diagram raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="Input wires of the second diagram.*contain duplicate indices"):  # ruff: ignore[pytest-raises-ambiguous-pattern]
            tensor.partial_trace([(0, [0, 1], [0, 1]), (1, [0, 1], [0, 0])])

    def test_mismatched_i_lengths(self):
        """Test that mismatched I1 and I2 lengths raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="I1 length.*must equal I2 length"):  # ruff: ignore[pytest-raises-ambiguous-pattern]
            tensor.partial_trace([
                (0, [0, 1], [0]),
                (1, [0], [1]),
            ])

    def test_mismatched_j_lengths(self):
        """Test that mismatched J1 and J2 lengths raise ValueError."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="J1 length.*must equal J2 length"):  # ruff: ignore[pytest-raises-ambiguous-pattern]
            tensor.partial_trace([
                (0, [0], [0, 1]),
                (1, [0], [0]),
            ])

    def test_output_wire_out_of_range_first(self):
        """Test output wire index out of range for first diagram."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="Output wire 5 out of range"):
            tensor.partial_trace([(0, [5], [0]), (1, [0], [0])])

    def test_input_wire_out_of_range_first(self):
        """Test input wire index out of range for first diagram."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="Input wire 5 out of range"):
            tensor.partial_trace([(0, [0], [5]), (1, [0], [0])])

    def test_output_wire_out_of_range_second(self):
        """Test output wire index out of range for second diagram."""
        d1 = QSpider(2, 2, ZxPoly({}))
        d2 = PSpider(1, 1, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="Input wire 5 out of range"):
            tensor.partial_trace([(0, [0], [0]), (1, [5], [0])])

    def test_input_wire_out_of_range_second(self):
        """Test input wire index out of range for second diagram."""
        d1 = QSpider(1, 1, ZxPoly({}))
        d2 = PSpider(2, 2, ZxPoly({}))
        tensor = TensorDiagram([d1, d2])

        with pytest.raises(ArityMismatchError, match="Output wire 5 out of range"):
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


class TestZxPolyGetParameters:
    """Test suite for ZxPoly.get_parameters()."""

    def test_excludes_own_generator_variable(self):
        """The polynomial's own generator variable must never be a parameter.

        This is the highest-value regression test flagged by the
        param_measurement_map plan: `sympy.Poly.free_symbols` incorrectly
        includes the generator, which is why `get_parameters()` is built
        from `.coeffs` instead.
        """
        m = symbols("m")
        poly = ZxPoly({1: m})
        x = symbols("x", real=True)  # ZxPoly's own generator variable.

        assert poly.get_parameters() == {m}
        assert x not in poly.get_parameters()

    def test_numeric_poly_has_no_parameters(self):
        """A purely numeric polynomial carries no free symbols."""
        poly = ZxPoly({0: 1.0, 2: -0.5})
        assert poly.get_parameters() == set()

    def test_multi_symbol_poly(self):
        """Multiple coefficients each contribute their own symbol."""
        a, b = symbols("a b")
        poly = ZxPoly({1: a, 2: b})
        assert poly.get_parameters() == {a, b}


class TestParametrizedMixin:
    """Test suite for the `Parametrized` mixin, exercised via QSpider/PSpider.

    Covers `param_measurement_map` subset validation, filtering during
    `substitute_parameters`/`evaluate`, `slice_param_map`, and `conjugate()`
    threading -- shared logic used identically by every `CompactDiagram`
    gate subclass in `gates.py`.
    """

    def test_subset_param_measurement_map_is_legal(self):
        """A param_measurement_map that is a strict subset of the parameters is legal."""
        a, b = symbols("a b", real=True)
        phase = ZxPoly({1: a, 2: b})
        spider = QSpider(1, 1, phase, True, param_measurement_map={a: {10}})

        assert spider.param_measurement_map == {a: {10}}
        assert spider.get_parameters() == {a, b}
        assert spider.is_parametric
        # Non-empty map derives feedforward/measurement_ids automatically.
        assert spider.feedforward
        assert spider.measurement_ids == {10}

    def test_param_measurement_map_invalid_symbol_raises(self):
        """A map keyed on a symbol that is not a parameter of the object is illegal."""
        a, c = symbols("a c", real=True)
        phase = ZxPoly({1: a})

        with pytest.raises(InvalidSymbolError, match="not among its own parameters"):
            QSpider(1, 1, phase, True, param_measurement_map={c: {1}})

    def test_substitute_parameters_filters_out_substituted_symbol(self):
        """Substituting a symbol away must drop its entry from the resulting map."""
        a = symbols("a", real=True)
        phase = ZxPoly({1: a})
        spider = QSpider(1, 1, phase, True, param_measurement_map={a: {5}})

        result = spider.substitute_parameters({a: 2.0})

        assert result.param_measurement_map == {}
        assert not result.is_parametric
        assert result.phase == ZxPoly({1: 2.0})

    def test_evaluate_partial_substitution_narrows_map(self):
        """Substituting one of two symbols must keep the other's map entry."""
        a, b = symbols("a b")
        phase = ZxPoly({1: a, 2: b})
        spider = QSpider(1, 1, phase, True, param_measurement_map={a: {1}, b: {2}})

        result = spider.evaluate(a=5.0)

        assert result.param_measurement_map == {b: {2}}
        assert result.get_parameters() == {b}

    def test_conjugate_preserves_param_measurement_map(self):
        """conjugate() must carry param_measurement_map through unchanged."""
        a = symbols("a", real=True)
        phase = ZxPoly({1: a})
        spider = QSpider(1, 1, phase, True, param_measurement_map={a: {7}})

        conjugated = spider.conjugate()

        assert conjugated.param_measurement_map == {a: {7}}
        assert conjugated.param_measurement_map is not spider.param_measurement_map
        assert conjugated.phase == -phase

    def test_slice_param_map_copies_inner_sets(self):
        """slice_param_map must restrict to the given symbols and copy, not alias, sets."""
        a, b = symbols("a b", real=True)
        phase = ZxPoly({1: a, 2: b})
        spider = QSpider(1, 1, phase, True, param_measurement_map={a: {1}, b: {2}})

        sliced = spider.slice_param_map({a})

        assert sliced == {a: {1}}
        sliced[a].add(999)
        assert spider.param_measurement_map[a] == {1}

    def test_pspider_also_supports_param_measurement_map(self):
        """The mixin applies identically to PSpider."""
        a = symbols("a", real=True)
        phase = ZxPoly({1: a})
        spider = PSpider(1, 1, phase, True, param_measurement_map={a: {3}})

        assert spider.param_measurement_map == {a: {3}}
        assert spider.feedforward
        assert spider.measurement_ids == {3}

        conjugated = spider.conjugate()
        assert conjugated.param_measurement_map == {a: {3}}

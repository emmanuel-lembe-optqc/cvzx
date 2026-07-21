"""Unit tests for the core RewriteRule class and its flatten_composition method.

These tests verify that nested compositions are properly flattened
to expose patterns that could otherwise be hidden.
"""

import unittest

from mqc3.zx.base_gates import (
    CompositionDiagram,
    ContractedDiagram,
    Fourier,
    Fourier2,
    PSpider,
    QSpider,
    Swap,
    TensorDiagram,
    ZxPoly,
    flatten_composition,
)
from mqc3.zx.rewrite_rules import RewriteRule


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

        self.rule = RewriteRule()

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
        assert result.diagrams[3] == self.q2

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
        assert result.diagrams[3] == self.q2

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
        assert result.diagrams[3] == self.q2

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

    # TODO Test the preservation of the connectivity in complex settings

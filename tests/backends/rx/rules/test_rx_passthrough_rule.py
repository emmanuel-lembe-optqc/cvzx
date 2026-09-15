"""Unit tests for `PassthroughRule` using the rustworkx graph formalism.

`PassthroughRule` rewrites a `ContractedDiagram` whose two halves are bare,
different-color spiders linked by a single one-way wire (forward `I1`/`I2`
or feedback `J1`/`J2`), each with kept arity exactly `(1, 1)`, into
`Compose`/`Tensor`/`Swap`/`VoidDiagram` -- but only once the two "outer"
connections it would repurpose are already proven dead (absent, or a
genuine `VoidDiagram`). These tests exercise both link directions, the
eligibility gate (using the exact counter-examples and worked examples
that motivated it), and the phase-based case dispatch (A/B/C/D).
"""

import unittest

from cvzx.backends.rx.graph import to_diagram, to_graph
from cvzx.backends.rx.rules import PassthroughRule, _get_id_map, apply_rule_to_diagram  # ruff: ignore[import-private-name]
from cvzx.ir.base import (
    CompositionDiagram,
    ContractedDiagram,
    Fourier,
    FourierInv,
    PSpider,
    QSpider,
    Swap,
    TensorDiagram,
    VoidDiagram,
    ZxPoly,
    flatten_composition,
    flatten_tensor,
)
from cvzx.ir.gates import DisplacementGate, PhaseRotationGate
from cvzx.visualization.core import visualize


class TestPassthroughRuleEligibility(unittest.TestCase):
    """Not-yet-eligible vs. eligible surrounding contexts, both link directions."""

    def setUp(self):
        """Create the shared phases and rule instance used by every test."""
        self.rule = PassthroughRule()
        self.nonzero_phase = ZxPoly({1: 2, 2: 4})
        self.zero_phase = ZxPoly({})
        self.identity_phase = ZxPoly({})

    # -- Downward (I1/I2) -----------------------------------------------

    def test_not_eligible_downward_successor_of_first_is_identity(self):
        """Predecessor(second) is void, but successor(first) is an identity, not void."""
        contracted = ContractedDiagram(
            QSpider(1, 2, self.nonzero_phase), PSpider(2, 1, self.zero_phase), [0], [0], [], []
        )
        diagram = CompositionDiagram([
            TensorDiagram([QSpider(1, 1, self.nonzero_phase), VoidDiagram(1, 1)]),
            contracted,
            TensorDiagram([QSpider(1, 1, self.identity_phase), QSpider(1, 1, self.identity_phase)]),
        ])
        result = apply_rule_to_diagram(self.rule, diagram)
        assert result == diagram

    def test_not_eligible_downward_predecessor_of_second_is_real(self):
        """Successor(first) is void, but predecessor(second) is a real gate, not void."""
        contracted = ContractedDiagram(
            QSpider(1, 2, self.nonzero_phase), PSpider(2, 1, self.zero_phase), [0], [0], [], []
        )
        diagram = CompositionDiagram([
            TensorDiagram([QSpider(1, 1, self.nonzero_phase), PhaseRotationGate(1)]),
            contracted,
            TensorDiagram([VoidDiagram(1, 1), QSpider(1, 1, self.identity_phase)]),
        ])
        result = apply_rule_to_diagram(self.rule, diagram)
        assert result == diagram

    def test_eligible_downward(self):
        """Both successor(first) and predecessor(second) are void -- rewrite fires."""
        contracted = ContractedDiagram(
            QSpider(1, 2, self.nonzero_phase), PSpider(2, 1, self.zero_phase), [0], [0], [], []
        )
        diagram = CompositionDiagram([
            TensorDiagram([QSpider(1, 1, self.nonzero_phase), VoidDiagram(1, 1)]),
            contracted,
            TensorDiagram([VoidDiagram(1, 1), PhaseRotationGate(1)]),
        ])
        result = apply_rule_to_diagram(self.rule, diagram)
        assert "Contract(" not in repr(result)
        inner = result.diagrams[1]
        assert isinstance(inner, CompositionDiagram)
        tensor, swap = inner.diagrams
        assert isinstance(tensor, TensorDiagram)
        assert isinstance(swap, Swap)
        survivor, void = tensor.diagrams
        assert isinstance(survivor, QSpider)
        assert survivor.phase == self.nonzero_phase
        assert (survivor.num_inputs, survivor.num_outputs) == (1, 1)
        assert (void.num_inputs, void.num_outputs) == (1, 1)
        assert swap.void_input_port == 0

    # -- Upward (J1/J2) ---------------------------------------------------

    def test_not_eligible_upward_predecessor_of_first_is_real(self):
        """Successor(second) is void, but predecessor(first) is a real spider, not void."""
        contracted = ContractedDiagram(
            QSpider(2, 1, self.nonzero_phase), PSpider(1, 2, self.zero_phase), [], [], [0], [0]
        )
        diagram = CompositionDiagram([
            TensorDiagram([QSpider(1, 1, self.nonzero_phase), PhaseRotationGate(1)]),
            contracted,
            TensorDiagram([DisplacementGate(4), VoidDiagram(1, 1)]),
        ])
        result = apply_rule_to_diagram(self.rule, diagram)
        assert result == diagram

    def test_not_eligible_upward_successor_of_second_is_identity(self):
        """Predecessor(first) is void, but successor(second) is an identity, not void."""
        contracted = ContractedDiagram(
            QSpider(2, 1, self.nonzero_phase), PSpider(1, 2, self.zero_phase), [], [], [0], [0]
        )
        diagram = CompositionDiagram([
            TensorDiagram([VoidDiagram(1, 1), PhaseRotationGate(1)]),
            contracted,
            TensorDiagram([DisplacementGate(4), QSpider(1, 1, self.identity_phase)]),
        ])
        result = apply_rule_to_diagram(self.rule, diagram)
        assert result == diagram

    def test_eligible_upward(self):
        """Both predecessor(first) and successor(second) are void -- rewrite fires."""
        contracted = ContractedDiagram(
            QSpider(2, 1, self.nonzero_phase), PSpider(1, 2, self.zero_phase), [], [], [0], [0]
        )
        diagram = CompositionDiagram([
            TensorDiagram([VoidDiagram(1, 1), PhaseRotationGate(1)]),
            contracted,
            TensorDiagram([DisplacementGate(4), VoidDiagram(1, 1)]),
        ])
        result = apply_rule_to_diagram(self.rule, diagram)
        assert "Contract(" not in repr(result)
        inner = result.diagrams[1]
        assert isinstance(inner, CompositionDiagram)
        swap, tensor = inner.diagrams
        assert isinstance(swap, Swap)
        assert isinstance(tensor, TensorDiagram)
        survivor, void = tensor.diagrams
        assert isinstance(survivor, QSpider)
        assert survivor.phase == self.nonzero_phase
        assert (survivor.num_inputs, survivor.num_outputs) == (1, 1)
        assert (void.num_inputs, void.num_outputs) == (1, 1)
        assert swap.void_input_port == 1

    # -- Root-level (no neighbor at all) ----------------------------------

    def test_root_level_is_eligible(self):
        """A root-level ContractedDiagram (no neighbors at all) is trivially eligible."""
        contracted = ContractedDiagram(
            QSpider(1, 2, self.nonzero_phase), PSpider(2, 1, self.zero_phase), [1], [0], [], []
        )
        result = apply_rule_to_diagram(self.rule, contracted)
        expected = CompositionDiagram([
            TensorDiagram([QSpider(1, 1, self.nonzero_phase), VoidDiagram(1, 1)]),
            Swap(void_input_port=0),
        ])
        assert result == expected


class TestPassthroughRuleCases(unittest.TestCase):
    """Phase-based case dispatch (A/B/C/D), all wrapped in an eligible context."""

    def setUp(self):
        """Create the shared phases and rule instance used by every test."""
        self.rule = PassthroughRule()
        self.nonzero_phase = ZxPoly({1: 2, 2: 4})
        self.zero_phase = ZxPoly({})

    def _wrap_eligible(self, contracted):
        return CompositionDiagram([
            TensorDiagram([QSpider(1, 1, self.zero_phase), VoidDiagram(1, 1)]),
            contracted,
            TensorDiagram([VoidDiagram(1, 1), PhaseRotationGate(1)]),
        ])

    def test_case_a_first_nonzero_second_zero(self):
        """`a` non-zero, `b` zero -- `a` survives at kept arity (1, 1)."""
        contracted = ContractedDiagram(
            QSpider(1, 2, self.nonzero_phase), PSpider(2, 1, self.zero_phase), [0], [0], [], []
        )
        result = apply_rule_to_diagram(self.rule, self._wrap_eligible(contracted))
        inner = result.diagrams[1]
        tensor, swap = inner.diagrams
        survivor, _void = tensor.diagrams
        assert isinstance(survivor, QSpider)
        assert survivor.phase == self.nonzero_phase
        assert swap.void_input_port == 0

    def test_case_b_first_zero_second_nonzero(self):
        """`a` zero, `b` non-zero -- `b` survives at kept arity (1, 1)."""
        contracted = ContractedDiagram(
            QSpider(1, 2, self.zero_phase), PSpider(2, 1, self.nonzero_phase), [0], [0], [], []
        )
        result = apply_rule_to_diagram(self.rule, self._wrap_eligible(contracted))
        inner = result.diagrams[1]
        swap, tensor = inner.diagrams
        survivor, _void = tensor.diagrams
        assert isinstance(survivor, PSpider)
        assert survivor.phase == self.nonzero_phase
        assert swap.void_input_port == 1

    def test_case_c_both_zero(self):
        """Both zero -- collapses to a bare Swap."""
        contracted = ContractedDiagram(
            QSpider(1, 2, self.zero_phase), PSpider(2, 1, self.zero_phase), [0], [0], [], []
        )
        result = apply_rule_to_diagram(self.rule, self._wrap_eligible(contracted))
        inner = result.diagrams[1]
        assert isinstance(inner, Swap)
        assert inner.void_input_port is None

    def test_case_d_both_nonzero_not_yet_implemented(self):
        """Both non-zero -- left untouched even when eligible (no verified routing yet)."""
        contracted = ContractedDiagram(
            QSpider(1, 2, self.nonzero_phase), PSpider(2, 1, ZxPoly({1: 5})), [0], [0], [], []
        )
        diagram = self._wrap_eligible(contracted)
        result = apply_rule_to_diagram(self.rule, diagram)
        assert result == diagram


class TestPassthroughRuleWiring(unittest.TestCase):
    """Graph-level regression tests: which neighbor actually feeds the survivor.

    `apply_rule_to_diagram`'s structural equality checks (used throughout
    `TestPassthroughRuleCases`) can't tell a correctly-wired rewrite from
    one that silently swapped in the dead (void) branch instead of the
    real one -- both produce an isomorphic-looking `Compose([Tensor([survivor,
    Void]), Swap()])` shape. `Fourier`/`FourierInv` are used as
    distinguishable, easy-to-spot-in-a-traceback stand-ins for "the real
    neighbor" on each side, and the raw graph edges are inspected directly
    (`to_diagram` doesn't expose this: it reconstructs purely from each
    container's own `connectivity`/mapping dicts, never the raw edges these
    tests check) since that's what a later rule's own graph traversal
    (e.g. `ChainReductionRule`'s `_composition_step`-based chasing) relies on.
    """

    def setUp(self):
        """Create the shared phases and rule instance used by every test."""
        self.rule = PassthroughRule()
        self.nonzero_phase = ZxPoly({1: 2, 2: 4})
        self.zero_phase = ZxPoly({})

    def test_case_a_swap_routes_real_value_to_successor_b(self):
        """`a` survives: `successor(b)` must be fed from the real (Fourier)
        side of the swap, not the void side."""
        contracted = ContractedDiagram(
            QSpider(1, 2, self.nonzero_phase), PSpider(2, 1, self.zero_phase), [0], [0], [], []
        )
        diagram = CompositionDiagram([
            TensorDiagram([Fourier(), VoidDiagram(1, 1)]),
            contracted,
            TensorDiagram([VoidDiagram(1, 1), FourierInv()]),
        ])
        cvzx_graph = to_graph(diagram)
        self.rule.apply_rule(cvzx_graph)
        graph = cvzx_graph.graph
        id_map = _get_id_map(graph)

        swap_idx = next(idx for idx in id_map.values() if graph[idx].get("type") == "Swap")
        survivor_idx = next(
            idx
            for idx in id_map.values()
            if graph[idx].get("type") == "QSpider" and graph[idx].get("kind") == "proper"
        )
        fourier_inv_idx = next(idx for idx in id_map.values() if graph[idx].get("type") == "FourierInv")

        survivor_to_swap = graph.get_edge_data(survivor_idx, swap_idx)
        swap_to_fourier_inv = graph.get_edge_data(swap_idx, fourier_inv_idx)

        assert survivor_to_swap["source_ports"] == [0]
        assert survivor_to_swap["target_ports"] == [0]
        assert swap_to_fourier_inv["source_ports"] == [0]
        assert graph[swap_idx]["void_input_port"] == 0

    def test_case_b_swap_routes_real_value_to_survivor(self):
        """`b` survives: it must be fed by the swap output crossing back to
        `predecessor(a)` (the real, Fourier-derived value forced equal by
        `a`'s zero phase) -- not `predecessor(b)` (void by eligibility).

        Regression test for a bug where the crossing was backwards: the
        survivor silently received the dead (void) branch's value while
        the real one was routed into the `Void` node and discarded.
        """
        contracted = ContractedDiagram(
            QSpider(1, 2, self.zero_phase), PSpider(2, 1, self.nonzero_phase), [0], [0], [], []
        )
        diagram = CompositionDiagram([
            TensorDiagram([Fourier(), VoidDiagram(1, 1)]),
            contracted,
            TensorDiagram([VoidDiagram(1, 1), FourierInv()]),
        ])
        cvzx_graph = to_graph(diagram)
        self.rule.apply_rule(cvzx_graph)
        graph = cvzx_graph.graph
        id_map = _get_id_map(graph)

        swap_idx = next(idx for idx in id_map.values() if graph[idx].get("type") == "Swap")
        survivor_idx = next(
            idx
            for idx in id_map.values()
            if graph[idx].get("type") == "PSpider" and graph[idx].get("kind") == "proper"
        )
        fourier_idx = next(idx for idx in id_map.values() if graph[idx].get("type") == "Fourier")

        fourier_to_swap = graph.get_edge_data(fourier_idx, swap_idx)
        swap_to_survivor = graph.get_edge_data(swap_idx, survivor_idx)

        assert fourier_to_swap["target_ports"] == [0]
        assert swap_to_survivor["source_ports"] == [1]
        assert graph[swap_idx]["void_input_port"] == 1

    def test_visualize_after_flatten_does_not_crash(self):
        """Regression test for the reported `IndexError` in `_draw_swap`.

        Reproduces the exact usage pattern that crashed: apply the rule
        directly to a graph, convert back to a diagram, run it through
        `flatten_tensor`/`flatten_composition` (which exposes a `Swap`
        directly adjacent to a 2-slot `Tensor` in a flat `Compose`
        sequence -- the shape that triggered the visualization-index bug),
        then `visualize()` it.
        """
        for contracted in (
            ContractedDiagram(QSpider(1, 2, self.nonzero_phase), PSpider(2, 1, self.zero_phase), [0], [0], [], []),
            ContractedDiagram(QSpider(1, 2, self.zero_phase), PSpider(2, 1, self.nonzero_phase), [0], [0], [], []),
        ):
            diagram = CompositionDiagram([
                TensorDiagram([Fourier(), VoidDiagram(1, 1)]),
                contracted,
                TensorDiagram([VoidDiagram(1, 1), FourierInv()]),
            ])
            cvzx_graph = to_graph(diagram)
            self.rule.apply_rule(cvzx_graph)
            result = to_diagram(cvzx_graph)
            flattened = CompositionDiagram([flatten_tensor(d) for d in result.diagrams])
            fig = visualize(flatten_composition(flattened), "regression test")
            fig.clf()


class TestPassthroughRuleMatchGuards(unittest.TestCase):
    """Structural guards: wrong shape, same color, root context."""

    def setUp(self):
        """Create the shared phases and rule instance used by every test."""
        self.rule = PassthroughRule()
        self.nonzero_phase = ZxPoly({1: 2, 2: 4})
        self.zero_phase = ZxPoly({})

    def test_no_match_when_kept_arity_not_one_one(self):
        """A structurally different-shaped contraction is left untouched."""
        a = QSpider(2, 3, self.nonzero_phase)
        b = PSpider(2, 1, self.zero_phase)
        contracted = ContractedDiagram(a, b, I1=[1], I2=[0], J1=[], J2=[])
        result = apply_rule_to_diagram(self.rule, contracted)
        assert result == contracted

    def test_same_color_pair_never_matches(self):
        """Two same-color spiders in this exact shape are FusionRule's job, not this rule's."""
        a = QSpider(1, 2, self.nonzero_phase)
        b = QSpider(2, 1, self.zero_phase)
        contracted = ContractedDiagram(a, b, I1=[1], I2=[0], J1=[], J2=[])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert matches == []

    def test_no_match_when_both_i1_and_j1_present(self):
        """A genuine two-way partial trace (both I1 and J1 nonempty) isn't this shape."""
        a = QSpider(2, 2, self.nonzero_phase)
        b = PSpider(2, 2, self.zero_phase)
        contracted = ContractedDiagram(a, b, I1=[0], I2=[0], J1=[1], J2=[1])
        graph = to_graph(contracted)
        matches = self.rule.match(graph)
        assert matches == []


if __name__ == "__main__":
    unittest.main()

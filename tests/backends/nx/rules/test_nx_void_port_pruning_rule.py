"""Unit tests for VoidPortPruningRule using the graph formalism.

`VoidPortPruningRule` trims a proper node's last input/output port when it
is wired, by a single "composition" edge, directly to a `VoidDiagram` --
shrinking both the live node and the void by exactly that one port, and
propagating the change up through each side's own parent. Restricted to
the *last* port on each side so neither endpoint needs its other ports
renumbered.
"""

import unittest

from sympy import symbols

from cvzx.backends.nx.graph import to_diagram, to_graph
from cvzx.backends.nx.rules import VoidPortPruningRule
from cvzx.ir.base import (
    CompositionDiagram,
    PSpider,
    QSpider,
    TensorDiagram,
    VoidDiagram,
    ZxPoly,
)


class TestVoidPortPruningRule(unittest.TestCase):
    """Test suite for VoidPortPruningRule."""

    def setUp(self):
        self.rule = VoidPortPruningRule()
        self.gamma = symbols("gamma", real=True)

    def test_match_output_dead_ends_at_void(self):
        """A (1,1) gate whose output feeds a Void is a prunable match."""
        gate = QSpider(1, 1, ZxPoly({1: self.gamma}), True)
        comp = CompositionDiagram([gate, VoidDiagram(1, 1)])
        graph = to_graph(comp)

        matches = self.rule.match(graph)

        assert len(matches) == 1
        assert matches[0]["is_input"] is False

    def test_match_input_fed_by_void(self):
        """A (1,1) gate whose input is fed by a Void is a prunable match."""
        gate = QSpider(1, 1, ZxPoly({1: self.gamma}), True)
        comp = CompositionDiagram([VoidDiagram(1, 1), gate])
        graph = to_graph(comp)

        matches = self.rule.match(graph)

        assert len(matches) == 1
        assert matches[0]["is_input"] is True

    def test_no_match_between_two_real_gates(self):
        """Two genuine (1,1) gates composed together never match."""
        comp = CompositionDiagram([
            QSpider(1, 1, ZxPoly({1: self.gamma}), True),
            PSpider(1, 1, ZxPoly({})),
        ])
        graph = to_graph(comp)

        assert self.rule.match(graph) == []

    def test_apply_shrinks_node_and_void_to_terminal_shapes(self):
        """Pruning a (1,1) gate's output against a (1,1) Void yields a (1,0) effect and a (0,1) void."""
        gate = QSpider(1, 1, ZxPoly({1: self.gamma}), True)
        comp = CompositionDiagram([gate, VoidDiagram(1, 1)])
        graph = to_graph(comp)

        self.rule.apply_rule(graph)
        graph.rebuild_registry()
        result = to_diagram(graph)

        assert isinstance(result, CompositionDiagram)
        survivor, void = result.diagrams
        assert isinstance(survivor, QSpider)
        assert survivor.num_inputs == 1
        assert survivor.num_outputs == 0
        assert survivor.phase == ZxPoly({1: self.gamma})
        assert isinstance(void, VoidDiagram)
        assert void.num_inputs == 0
        assert void.num_outputs == 1

    def test_no_prune_when_result_would_orphan_non_zero_phase(self):
        """A node already at (1,0)/(0,1) with a non-zero phase is never pruned further.

        Pruning its last remaining port would leave a (0, 0) node whose
        phase polynomial has no port left for its variable to bind to --
        silently discarding whatever physical contribution that phase
        represented. Only a zero phase is safe to erase entirely this way.
        """
        effect = QSpider(1, 0, ZxPoly({1: self.gamma}), True)
        comp = CompositionDiagram([VoidDiagram(1, 1), effect])
        graph = to_graph(comp)

        assert self.rule.match(graph) == []

    def test_prune_allowed_when_result_phase_is_zero(self):
        """A zero-phase (1,0)/(0,1) node CAN still be pruned down to (0, 0) -- it's a genuine identity."""
        effect = QSpider(1, 0, ZxPoly({}))
        comp = CompositionDiagram([VoidDiagram(1, 1), effect])
        graph = to_graph(comp)

        matches = self.rule.match(graph)

        assert len(matches) == 1

    def test_void_sibling_auto_shrunk_by_arity_propagation_is_not_double_shrunk(self):
        """When shrinking the live node's own parent auto-shrinks the void sibling, apply_single doesn't double-shrink it.

        This mirrors the composition-boundary case documented in
        `VoidPortPruningRule.apply_single`: `_propagate_arity_to_parent`'s
        own composition-connectivity fixup can shrink an adjacent sibling
        through `_remove_external_port`'s "if it traces to a Void leaf"
        fallback, as a side effect of shrinking the matched node -- before
        `apply_single` gets a chance to shrink the matched void itself.
        Reproduces the shape that originally surfaced this: a wide gate's
        LAST output feeding a `Tensor` row whose corresponding slot is a
        bare `VoidDiagram`, with another real row alongside it.
        """
        gate = QSpider(1, 2, ZxPoly({1: self.gamma}), True)
        next_tensor = TensorDiagram([QSpider(1, 1, ZxPoly({})), VoidDiagram(1, 1)])
        comp = CompositionDiagram([gate, next_tensor])
        graph = to_graph(comp)

        matches = self.rule.match(graph)
        assert len(matches) == 1

        # Should not raise (e.g. from a negative port count) and should
        # still converge to a valid, arity-consistent diagram.
        self.rule.apply_rule(graph)
        graph.rebuild_registry()
        result = to_diagram(graph)
        assert result.num_inputs == 1
        assert result.num_outputs == 2
        assert isinstance(result, CompositionDiagram)
        survivor, tensor = result.diagrams
        assert isinstance(survivor, QSpider)
        assert survivor.num_inputs == 1
        assert survivor.num_outputs == 1
        void_row = tensor.diagrams[1]
        assert isinstance(void_row, VoidDiagram)
        assert void_row.num_inputs == 0
        assert void_row.num_outputs == 1


if __name__ == "__main__":
    unittest.main()

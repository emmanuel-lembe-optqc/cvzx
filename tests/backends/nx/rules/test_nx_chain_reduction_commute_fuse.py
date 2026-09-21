"""Unit tests for ChainReductionRule's commute-and-fuse match kind.

Covers the four "move a Sq/Disp across a spider, then fuse" pattern
shapes `ChainReductionRule._find_commute_matches` recognizes:

- `Sq(tau) -- Spider -- Sq(kappa)`: cross `tau` into Spider, fuse the two
  squeezings (tau*kappa).
- `Disp(a) -- Spider -- Disp(b)` (Disp opposite-color to Spider): cross
  `a` into Spider, fuse the two Disps (a+b).
- `Q -- Sq/Disp -- Q` / `P -- Sq/Disp -- P`: cross the mover into one
  flank, fuse the two same-color flanks.

Both are exact identities (no `assume_infinite_squeezing` gate), matching
`ChainReductionRule`'s own "exact identities only" scope -- unlike the
analogous absorption of a genuine TERMINAL state/effect
(`TerminalAbsorptionRule`), these move a GATE (arity (1, 1)) across a
spider, which is exact regardless of squeezing.
"""

import unittest
from math import isclose

from sympy import sqrt, symbols

from cvzx.backends.nx.graph import to_diagram, to_graph
from cvzx.backends.nx.rules import ChainReductionRule, IdentityRule
from cvzx.ir.base import CompositionDiagram, PSpider, QSpider, ZxPoly
from cvzx.ir.gates import SqueezingGate


def _apply_and_cleanup(graph, rule):
    """Apply `rule` to a fixed point then sweep leftover identities."""
    rule.apply_rule(graph)
    graph.rebuild_registry()
    IdentityRule().apply_rule(graph)


class TestChainReductionCommuteFuse(unittest.TestCase):
    """Test suite for ChainReductionRule's commute-and-fuse logic."""

    def setUp(self):
        self.rule = ChainReductionRule()

    def test_sq_spider_sq_crosses_and_fuses(self):
        """Sq(tau) -- Q(f(x)) -- Sq(kappa): f becomes f(tau*x), squeezings fuse to tau*kappa."""
        comp = CompositionDiagram([
            SqueezingGate(2.0),
            QSpider(1, 1, ZxPoly({1: 3.0, 2: 1.0})),
            SqueezingGate(5.0),
        ])
        graph = to_graph(comp)

        matches = self.rule.match(graph)
        assert any(m.get("kind") == "commute" for m in matches)

        _apply_and_cleanup(graph, self.rule)
        result = to_diagram(graph)

        assert isinstance(result, CompositionDiagram)
        spider, sq = result.diagrams
        assert isinstance(spider, QSpider)
        assert isclose(spider.phase.coeffs[1], 3.0 * 2.0)
        assert isclose(spider.phase.coeffs[2], 1.0 * 2.0**2)
        assert isinstance(sq, SqueezingGate)
        assert isclose(sq.tau, 2.0 * 5.0)

    def test_disp_spider_disp_opposite_color_crosses_and_fuses(self):
        """P(a) -- Q(f(x)) -- P(b): f becomes f(x+a), the two P's fuse to P(a+b)."""
        a, b = 1.5, 2.5
        comp = CompositionDiagram([
            PSpider(1, 1, ZxPoly({1: a})),
            QSpider(1, 1, ZxPoly({1: 1.0, 2: 1.0})),
            PSpider(1, 1, ZxPoly({1: b})),
        ])
        graph = to_graph(comp)

        _apply_and_cleanup(graph, self.rule)
        result = to_diagram(graph)

        assert isinstance(result, CompositionDiagram)
        spider, p_fused = result.diagrams
        assert isinstance(spider, QSpider)
        # (x+a) + (x+a)**2 = x + a + x**2 + 2ax + a**2 -- constant term
        # (a + a**2) is dropped by QSpider's own global-phase normalization.
        assert isclose(spider.phase.coeffs[1], 1.0 + 2 * a)
        assert isclose(spider.phase.coeffs[2], 1.0)
        assert isinstance(p_fused, PSpider)
        assert isclose(p_fused.phase.coeffs[1], a + b)

    def test_disp_does_not_cross_same_color_spider(self):
        """A linear-phase Disp may only cross its OPPOSITE color -- same color never matches."""
        comp = CompositionDiagram([
            QSpider(1, 1, ZxPoly({1: 1.5})),
            QSpider(1, 1, ZxPoly({2: 1.0})),
            QSpider(1, 1, ZxPoly({1: 2.5})),
        ])
        graph = to_graph(comp)

        # No commute match for this triple -- same-type chain fusion
        # (Q + Q + Q, unrelated to commuting) may still apply, but never
        # a "commute" kind spanning all three via crossing.
        matches = self.rule.match(graph)
        assert not any(m.get("kind") == "commute" for m in matches)

    def test_q_sq_q_crosses_into_right_flank_and_fuses(self):
        """Q(f) -- Sq(tau) -- Q(g): Sq crosses into g, then f and g(tau*x) fuse."""
        comp = CompositionDiagram([
            QSpider(1, 1, ZxPoly({1: 2.0, 2: 1.0})),
            SqueezingGate(3.0),
            QSpider(1, 1, ZxPoly({1: 5.0})),
        ])
        graph = to_graph(comp)

        _apply_and_cleanup(graph, self.rule)
        result = to_diagram(graph)

        # The mover (Sq) is left in its own slot, unmoved -- only the two
        # flanking Q's fuse; the drained flank is what IdentityRule sweeps.
        assert isinstance(result, CompositionDiagram)
        spider, sq = result.diagrams
        assert isinstance(spider, QSpider)
        # crossed: g(tau*x) = 5*(3x) = 15x; fused: f + crossed = 2x + x**2 + 15x = x**2 + 17x
        assert isclose(spider.phase.coeffs[1], 17.0)
        assert isclose(spider.phase.coeffs[2], 1.0)
        assert isinstance(sq, SqueezingGate)
        assert isclose(sq.tau, 3.0)

    def test_p_disp_p_with_opposite_color_disp_in_middle(self):
        """P(f) -- Q(a) -- P(g) (Q is Disp, opposite of P): Q crosses into g, fuses with f."""
        a = 2.0
        comp = CompositionDiagram([
            PSpider(1, 1, ZxPoly({3: 1.0})),
            QSpider(1, 1, ZxPoly({1: a})),
            PSpider(1, 1, ZxPoly({1: 1.0})),
        ])
        graph = to_graph(comp)

        _apply_and_cleanup(graph, self.rule)
        result = to_diagram(graph)

        assert isinstance(result, CompositionDiagram)
        spider, mover = result.diagrams
        assert isinstance(spider, PSpider)
        # crossed: g(x+a) = (x+a) -- constant dropped, coeff of x stays 1;
        # fused: f + crossed = x**3 + x.
        assert isclose(spider.phase.coeffs[3], 1.0)
        assert isclose(spider.phase.coeffs[1], 1.0)
        assert isinstance(mover, QSpider)
        assert isclose(mover.phase.coeffs[1], a)

    def test_no_commute_when_spider_is_a_terminal(self):
        """A terminal-shaped 'Spider' (arity (0, 1) or (1, 0)) is TerminalAbsorptionRule's job, not this rule's."""
        comp = CompositionDiagram([
            SqueezingGate(2.0),
            QSpider(1, 0, ZxPoly({1: 1.0})),
        ])
        graph = to_graph(comp)

        matches = self.rule.match(graph)
        assert not any(m.get("kind") == "commute" for m in matches)

    def test_commute_chases_through_zero_phase_identity_chain(self):
        """The commute scan crosses an intervening zero-phase (1,1) identity spider."""
        comp = CompositionDiagram([
            QSpider(1, 1, ZxPoly({1: 2.0, 2: 1.0})),
            SqueezingGate(3.0),
            PSpider(1, 1, ZxPoly({})),  # zero-phase identity in between
            QSpider(1, 1, ZxPoly({1: 5.0})),
        ])
        graph = to_graph(comp)

        matches = self.rule.match(graph)
        commute_matches = [m for m in matches if m.get("kind") == "commute"]
        assert len(commute_matches) == 1
        assert commute_matches[0]["identity_chain"]

        _apply_and_cleanup(graph, self.rule)
        result = to_diagram(graph)
        assert isinstance(result, CompositionDiagram)
        spider, sq = result.diagrams
        assert isinstance(spider, QSpider)
        assert isclose(spider.phase.coeffs[1], 17.0)
        assert isclose(spider.phase.coeffs[2], 1.0)
        assert isinstance(sq, SqueezingGate)
        assert isclose(sq.tau, 3.0)


if __name__ == "__main__":
    unittest.main()

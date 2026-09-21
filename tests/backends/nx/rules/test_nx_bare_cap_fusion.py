"""Dedicated unit tests for TerminalAbsorptionRule's `bare_cap_fusion` match kind.

Collapses a `ContractedDiagram` whose one half is already a fully-internal
bare terminal (a genuine `(0, 1)`/`(1, 0)` `QSpider`/`PSpider` directly
occupying `first_id`/`second_id`, with its single port entirely consumed
by the contraction -- nothing external connects to it) into just the
surviving other half. Same-color folds via addition (exact, unconditional);
opposite-color folds via the shift formula (gated on
`assume_infinite_squeezing`, since that treats the terminal as an
idealized eigenstate).

This is exercised only incidentally elsewhere (as a second-round side
effect inside `test_apply_rule_state_absorbs_output_role_state1`) -- these
tests hit `_try_fuse_bare_contracted_terminal`/`_apply_bare_cap_fusion`
directly and in isolation.
"""

import unittest
from math import isclose

from sympy import symbols

from cvzx.backends.nx.graph import to_diagram, to_graph
from cvzx.backends.nx.rules import TerminalAbsorptionRule
from cvzx.ir.base import ContractedDiagram, PSpider, QSpider, ZxPoly


class TestBareCapFusion(unittest.TestCase):
    """Test suite for TerminalAbsorptionRule's bare_cap_fusion match kind."""

    def setUp(self):
        self.gamma, self.m = symbols("gamma m", real=True)

    def test_same_color_fold_is_unconditional(self):
        """A same-color bare terminal cap folds via addition even without assume_infinite_squeezing."""
        diag = ContractedDiagram(
            QSpider(1, 2, ZxPoly({2: 1.0})),
            QSpider(1, 0, ZxPoly({1: self.gamma}), True),
            I1=[1], I2=[0], J1=[], J2=[],
        )
        graph = to_graph(diag)
        rule = TerminalAbsorptionRule()  # default: assume_infinite_squeezing=False

        matches = rule.match(graph)
        assert any(m.get("kind") == "bare_cap_fusion" for m in matches)

        rule.apply_rule(graph)
        graph.rebuild_registry()
        result = to_diagram(graph)

        assert isinstance(result, QSpider)
        assert result.num_inputs == 1
        assert result.num_outputs == 1
        assert result.phase == ZxPoly({2: 1.0, 1: self.gamma})

    def test_opposite_color_fold_requires_assume_infinite_squeezing(self):
        """An opposite-color bare terminal cap only folds when assume_infinite_squeezing=True."""
        diag = ContractedDiagram(
            QSpider(1, 2, ZxPoly({2: 1.0, 1: 3.0})),
            PSpider(1, 0, ZxPoly({1: -self.m}), True),
            I1=[1], I2=[0], J1=[], J2=[],
        )

        graph_default = to_graph(diag)
        assert not any(
            m.get("kind") == "bare_cap_fusion" for m in TerminalAbsorptionRule().match(graph_default)
        )

        graph_idealized = to_graph(diag)
        rule = TerminalAbsorptionRule(assume_infinite_squeezing=True)
        matches = rule.match(graph_idealized)
        assert any(m.get("kind") == "bare_cap_fusion" for m in matches)

        rule.apply_rule(graph_idealized)
        graph_idealized.rebuild_registry()
        result = to_diagram(graph_idealized)

        assert isinstance(result, QSpider)
        # f(x) = x**2 + 3x, shifted by a = -m: (x-m)**2 + 3(x-m)
        #      = x**2 - 2mx + m**2 + 3x - 3m = x**2 + (3 - 2m)x + (m**2 - 3m)
        # -- the constant term is dropped by QSpider's own global-phase
        # normalization.
        assert isclose(float(result.phase.coeffs[2]), 1.0)
        coeff_x = result.phase.coeffs[1]
        assert coeff_x.equals(3 - 2 * self.m)

    def test_degenerate_both_sides_fully_consumed_is_not_folded(self):
        """If the survivor would ALSO collapse to (0, 0), it's a closed scalar -- left alone."""
        diag = ContractedDiagram(
            QSpider(0, 1, ZxPoly({3: self.gamma}), True),
            PSpider(1, 0, ZxPoly({1: -self.m}), True),
            I1=[0], I2=[0], J1=[], J2=[],
        )
        graph = to_graph(diag)
        rule = TerminalAbsorptionRule(assume_infinite_squeezing=True)

        matches = rule.match(graph)
        assert not any(m.get("kind") == "bare_cap_fusion" for m in matches)

    def test_no_fold_when_neither_half_is_a_bare_terminal(self):
        """A ContractedDiagram whose halves both have real external structure never matches."""
        diag = ContractedDiagram(
            QSpider(1, 2, ZxPoly({})),
            PSpider(2, 1, ZxPoly({})),
            I1=[1], I2=[0], J1=[], J2=[],
        )
        graph = to_graph(diag)
        rule = TerminalAbsorptionRule(assume_infinite_squeezing=True)

        matches = rule.match(graph)
        assert not any(m.get("kind") == "bare_cap_fusion" for m in matches)


if __name__ == "__main__":
    unittest.main()

"""Regression tests: `optimize()` reaches byte-identical results on both backends.

Both bugs these tests guard against were found the same way: run a
realistic, multi-`ContractedDiagram` circuit through `optimize()` on `nx`
and `rx` and diff the results. Each was a rewrite-rule fix (already
present and tested on `nx`) that had simply never been ported to `rx`:

- `FusionRule._apply_contracted`'s `special_case` branch didn't account
  for which of `first`/`second` was actually the `TensorDiagram` side
  (the `down`/no-`down` distinction), so it spliced the fused spider into
  the wrong end of the survivor's `sub_diagram_ids` whenever `second`
  wasn't the `TensorDiagram`.
- `TerminalAbsorptionRule._apply_contracted_child` reset a crossed
  identity-chain passthrough to a same-arity identity spider instead of
  voiding it and propagating the shrink -- correct only when the
  terminal keeps its own slot (ordinary absorption), not when it moves
  into the contraction (this case), where the chain becomes fully dead.

Neither bug showed up in any single-backend test, since both require a
specific multi-contraction circuit shape (a real identity crossed en
route to a contracted child, or a `first`/`second` in the "wrong" order)
that the existing per-rule unit tests didn't happen to construct.
"""

import unittest

from sympy import I, pi, symbols

from cvzx.backends.nx.graph import to_diagram as nx_to_diagram
from cvzx.backends.nx.graph import to_graph as nx_to_graph
from cvzx.backends.rx.graph import to_diagram as rx_to_diagram
from cvzx.backends.rx.graph import to_graph as rx_to_graph
from cvzx.ir.base import CompositionDiagram, PSpider, QSpider, TensorDiagram, ZxPoly
from cvzx.ir.gates import BeamsplitterGate, DisplacementGate, PhaseRotationGate
from cvzx.passes.optimize import _build_rules, optimize


def _nonunit_gain_gadget() -> CompositionDiagram:
    """`examples/example_4_cubic_phase_injection_nonunit_gain.ipynb`'s initial diagram.

    Two beamsplitter-derived `ContractedDiagram` pairs plus a
    `CopyRule`-driven measurement fan-out -- the shape that exercises
    both bugs this module guards against.

    Returns
    -------
    CompositionDiagram
    """
    zero_phase = ZxPoly({})
    m1, m2, theta1, lambda1, gamma = symbols("m1 m2 theta1 lambda1 gamma", real=True)
    meas1 = PSpider(1, 0, ZxPoly({1: -m1}), True)
    meas2 = QSpider(1, 0, ZxPoly({1: -m2}), True)

    stage0 = TensorDiagram([
        QSpider(1, 1, zero_phase),
        PSpider(0, 1, zero_phase),
        QSpider(0, 1, ZxPoly({3: gamma}), True),
    ])
    stage1 = TensorDiagram([BeamsplitterGate(pi / 4).expand(), QSpider(1, 1, zero_phase)])
    stage2 = TensorDiagram([QSpider(1, 1, zero_phase), BeamsplitterGate(pi / 4).expand()])
    stage3 = TensorDiagram([
        DisplacementGate(I * lambda1, True, True, {meas2.id}, {lambda1: {meas2.id}}),
        CompositionDiagram([PhaseRotationGate(theta1, True, True, {meas1.id}, {theta1: {meas1.id}}), meas2]),
        meas1,
    ])
    return CompositionDiagram(
        [stage0, stage1, stage2, stage3],
        {0: {0: 0, 1: 1, 2: 2}, 1: {0: 0, 1: 2, 2: 1}, 2: {0: 0, 1: 2, 2: 1}},
    )


class TestOptimizeBackendParity(unittest.TestCase):
    """`optimize()` must reach the same result regardless of backend."""

    def test_nonunit_gain_gadget_optimizes_identically_on_both_backends(self):
        """`optimize()` under `assume_infinite_squeezing=True` agrees between `nx` and `rx`."""
        gadget = _nonunit_gain_gadget()
        result_nx = optimize(gadget, backend="networkx", assume_infinite_squeezing=True)
        result_rx = optimize(gadget, backend="rustworkx", assume_infinite_squeezing=True)
        assert repr(result_nx.diagram) == repr(result_rx.diagram)

    def test_nonunit_gain_gadget_single_pass_agrees_between_backends(self):
        """A single round-tripped rule pass (not just `optimize()`) also agrees between backends.

        Runs the shared rule list to a fixed point on each backend
        separately, round-tripping through `to_diagram()`/`to_graph()`
        after every individual rule application -- this is what actually
        caught both bugs (diffing this trace step by step is how each
        was isolated to `FusionRule` and
        `TerminalAbsorptionRule._apply_contracted_child` respectively).
        """
        from cvzx.backends.nx import rules as nx_rules_mod
        from cvzx.backends.rx import rules as rx_rules_mod

        gadget = _nonunit_gain_gadget()

        def run(rules_mod, to_graph, to_diagram):
            graph = to_graph(gadget)
            graph.rebuild_registry()
            rules = _build_rules(rules_mod, assume_infinite_squeezing=True)
            for _ in range(30):
                fired = False
                for rule in rules:
                    if rule.match(graph):
                        rule.apply_rule(graph)
                        fired = True
                        graph.rebuild_registry()
                        graph = to_graph(to_diagram(graph))
                        graph.rebuild_registry()
                if not fired:
                    break
            return to_diagram(graph)

        result_nx = run(nx_rules_mod, nx_to_graph, nx_to_diagram)
        result_rx = run(rx_rules_mod, rx_to_graph, rx_to_diagram)
        assert repr(result_nx) == repr(result_rx)


if __name__ == "__main__":
    unittest.main()

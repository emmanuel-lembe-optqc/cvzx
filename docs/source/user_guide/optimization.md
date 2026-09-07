# Optimizing a diagram

`cvzx.optimize.optimize(diagram, *, assume_infinite_squeezing=False, max_rounds=100)` is the
main entry point: it repeatedly applies every rewrite rule (see {doc}`rewrite_rules`) until
none of them match, then runs the end-of-pipeline cleanup once.

```python
from sympy import pi, simplify

from cvzx.base_gates import CompositionDiagram
from cvzx.gates import PhaseRotationGate
from cvzx.nx_graph import to_diagram
from cvzx.optimize import optimize

theta1, theta2, theta3 = pi / 6, pi / 5, pi / 7
comp = CompositionDiagram([
    PhaseRotationGate(theta1),
    PhaseRotationGate(theta2),
    PhaseRotationGate(theta3),
])

graph, diagram = optimize(comp)
result = to_diagram(graph)
assert isinstance(result, PhaseRotationGate)
assert simplify(result.theta - (theta1 + theta2 + theta3)) == 0
```

(adapted from `tests/test_optimize.py::test_chain_reduction_combines_three_rotations`.)

## What you get back

`optimize()` returns an `OptimizeResult` — a `NamedTuple` with two fields, so both
`graph, diagram = optimize(...)` unpacking and `result.graph`/`result.diagram` attribute
access work:

- **`.graph`** — the simplified `networkx.DiGraph`, *after* the final cleanup pass. This is
  what you want to keep computing with (feed to another `RewriteRule`, inspect with
  `cvzx.nx_graph`'s `get_*` helpers, or convert with `to_diagram` for a fresh snapshot).
- **`.diagram`** — the simplified `Diagram`, converted from the graph *right before* that
  cleanup pass ran. Use this one for visualization: the cleanup pass can restructure the
  graph in ways a nested `Diagram` tree can no longer losslessly represent, so `.diagram` is
  captured beforehand rather than re-derived from the cleaned graph afterward.

```python
from cvzx.visualize_base_gates import visualize

fig = visualize(result.diagram, title="optimized circuit")
```

## `assume_infinite_squeezing`

This flag threads through to `TerminalAbsorptionRule` and to whether `optimize()` expands
two-mode gates (`expand_two_mode_gates`) and runs `CopyRule` at all — see {doc}`rewrite_rules`
and {doc}`../theory` for exactly which identities each setting corresponds to.

- **`False` (default)** — only rewrites that are exact for *any* physical state run:
  `IdentityRule`, `FusionRule`, `ChainReductionRule`, `FourierNormalizationRule`, and
  rotation absorption within `TerminalAbsorptionRule`. Two-mode gates are left compact and
  `CopyRule` never fires.
- **`True`** — additionally treats terminal states/effects as idealized, infinitely-squeezed
  eigenstates, unlocking squeezing absorption, cross-color discard, two-mode gate expansion,
  and `CopyRule`. This is what lets, e.g., an ancilla state consumed by a
  `ControlledSumGate` disappear entirely instead of staying as an explicit gate:

```python
from cvzx.base_gates import CompositionDiagram, PSpider, QSpider, TensorDiagram, ZxPoly
from cvzx.gates import ControlledSumGate
from cvzx.nx_graph import to_diagram
from cvzx.optimize import optimize

control_state = PSpider(0, 1, ZxPoly({1: 2}))
target_state = QSpider(0, 1, ZxPoly({1: 3}))
csum = ControlledSumGate(control=2, target=1)
comp = CompositionDiagram([TensorDiagram([control_state, target_state]), csum])

graph, _ = optimize(comp, assume_infinite_squeezing=True)
```

Without `assume_infinite_squeezing=True`, this same diagram is left with the `CSUM` gate
intact (see `tests/test_optimize.py::TestOptimizeExact::test_controlled_sum_gate_untouched_without_squeezing`
for the contrasting case).

## Convergence

Each round: normalize (`cvzx.normalize_diagram.normalize_diagram` — see
{doc}`../dev_guide/normalization`), optionally expand two-mode gates, then run every rule to
a fixed point over the graph, then convert back to a `Diagram` to feed the next round.
Rounds repeat because folding can combine adjacent two-mode gates that themselves need
re-expanding to expose further copy patterns (e.g. `ChainReductionRule` merging two
`ControlledSumGate`s on the same modes). A round that only expanded gates without any rule
actually matching is discarded rather than committed, and `max_rounds` is a safety cap
against a diagram that never reaches a fixed point (it should not be hit in practice —
raise an issue if you find a circuit that does; `optimize()` logs a warning if it is).

## Performance

`optimize()`'s round loop does `O(n)` work per round in diagram size, not
`O(rounds × rules × n)`: `_simplify_to_fixed_point` only rebuilds its internal `GateRegister`
(the O(1)-lookup index `match()` uses so it doesn't have to scan the whole graph for, say,
every rotation gate) right after a rule actually applies a change — never merely because a
pass moves on to the next rule. A rule whose `match()` finds nothing leaves the graph, and
therefore every category the registry indexes, untouched, so rebuilding in that case would be
pure wasted work; as a round approaches the fixed point and fewer rules still match, this
saves more and more rebuilds. `to_graph`/`to_diagram` also no longer carry a defensive
`deepcopy` of the diagram they convert — nothing in that conversion walk mutates its input
(pinned down by `tests/nx_graph/test_nx_graph.py::TestToGraphDoesNotMutateInput`), so the copy
was pure overhead, on the order of a third of `to_graph`'s own cost on a realistic circuit.
Neither change affects what `optimize()` computes, only how much work it takes to get there —
see {doc}`../dev_guide/rewrite_engine` for the registry-sync rationale in more detail.

## Debugging with logs

`cvzx.normalize_diagram`, `cvzx.nx_rewrite_rules`, and `cvzx.optimize` each log through the
standard `logging` module (which rule matched, how many rounds/passes ran, and a warning if
`max_rounds` is hit) — useful when a diagram isn't simplifying the way you expect. They only
ever emit records; nothing is written anywhere until you configure logging yourself. The
easiest way to do that is `cvzx.logging_config.setup_file_logging()`, called once early in
your script:

```python
from cvzx.logging_config import setup_file_logging

setup_file_logging()  # writes to ./logs/ by default; pass log_dir=... to change that
```

This gives each of the three loggers above its own file under the log directory
(`normalize_diagram.log`, `nx_rewrite_rules.log`, `optimize.log`) instead of one mixed-together
file, so you can tell, e.g., "did `FusionRule` ever match" apart from "did the round loop
converge" at a glance.

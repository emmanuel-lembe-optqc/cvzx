# Rewrite rules

Every rewrite rule in `cvzx.backends.nx.rules` implements the same `RewriteRule` interface
(see {doc}`../dev_guide/rewrite_engine` for why the contract looks the way it does):

- `rule.match(graph, registry) -> list[dict]` finds every independent match in the graph and
  returns one dict per match with whatever bookkeeping `apply_single` needs.
- `rule.apply_single(graph, match) -> None` applies one match in place.
- `rule.apply_rule(graph, registry) -> nx.DiGraph` (inherited, not overridden) applies every
  match `match()` found, once, and returns the graph for chaining.

You will normally just call `cvzx.passes.optimize.optimize` (see {doc}`optimization`), which drives
every rule to a fixed point for you. This page is for when you want to apply one rule
directly — e.g. in a test, or to inspect what a single rule does in isolation.

## Applying one rule to a whole diagram

`cvzx.backends.nx.rules.apply_rule_to_diagram(rule, diagram)` is the simplest entry point: it
converts to a graph, calls `rule.apply_rule`, and converts back.

```python
from sympy import pi

from cvzx.ir.base import CompositionDiagram, Fourier2
from cvzx.ir.gates import PhaseRotationGate
from cvzx.backends.nx.rules import FourierNormalizationRule, apply_rule_to_diagram

comp = CompositionDiagram([PhaseRotationGate(pi / 6), Fourier2()])
result = apply_rule_to_diagram(FourierNormalizationRule(), comp)
# result is a single PhaseRotationGate(pi/6 + pi)
```

## Matching and applying one match manually

For finer control — e.g. to apply only *one* of several matches, or to inspect a match's
fields before deciding whether to apply it — build the graph and registry yourself:

```python
from cvzx.ir.base import CompositionDiagram, Fourier, QSpider, ZxPoly
from cvzx.backends.nx.graph import to_diagram, to_graph
from cvzx.backends.nx.rules import IdentityRule

id_wire = QSpider(1, 1, ZxPoly({}))  # a blank spider is an identity wire
comp = CompositionDiagram([id_wire, Fourier()])

graph = to_graph(comp)  # a CVZXGraph, with its GateRegister already built

rule = IdentityRule()
matches = rule.match(graph)
assert len(matches) == 1
assert matches[0]["node_id"] == id_wire.id

rule.apply_single(graph, matches[0])
result = to_diagram(graph)   # -> Fourier()
```

This pattern — build the `CVZXGraph`, call `match`, inspect/apply `matches[0]` — is used
throughout `tests/backends/nx/rules/`; every rule's test file is a good
source of further worked examples for that specific rule's match fields.

## Rule-by-rule notes

- **`IdentityRule`** — matches a bare identity spider ($QSpider(1,1,0)$ or $PSpider(1,1,0)$)
  sitting inside a `CompositionDiagram` and splices it out.
- **`FusionRule`** — matches two directly-adjacent same-color spiders (any number of wires
  between them) and merges them, summing phases.
- **`ChainReductionRule`** — matches a run of same-*type* gates (`PhaseRotationGate`,
  `BeamsplitterGate`, `SqueezingGate`, `DisplacementGate`, `Fourier`/`FourierInv`/`Fourier2`,
  `ControlledSumGate`, `ControlledZGate`) directly adjacent in one `CompositionDiagram`, and
  collapses the whole chain into one gate (or the identity) using the composition law for
  that gate type — see `ChainReductionRule.reduce_chain` for the closed-form per type.
- **`FourierNormalizationRule`** — matches a `Fourier`/`FourierInv`/`Fourier2` adjacent to a
  `PhaseRotationGate`, `BeamsplitterGate`, or `SqueezingGate` and folds it into that gate's
  parameter.
- **`TerminalAbsorptionRule`** — matches a gate adjacent to a $(1,0)$-effect or
  $(0,1)$-state `QSpider`/`PSpider` terminal, and folds the gate into the terminal's phase.
  Constructed as `TerminalAbsorptionRule(assume_infinite_squeezing=False)` by default, which
  restricts it to rotation absorption (exact for any state); passing `True` additionally
  enables squeezing absorption and "cross-color discard" (an opposite-color raw spider
  vanishing next to a terminal), both of which are only exact for idealized,
  infinitely-squeezed terminal eigenstates.
- **`CopyRule`** — matches a $(0,1)$/$(1,0)$ spider adjacent to a wide ($n$-input or
  $n$-output) opposite-color spider whose phase is in $\mathbb{R}_1[X]$
  (`CopyRule.is_in_R1`), and copies the narrow spider through, producing $n$ copies in a
  `TensorDiagram`. Unlike the other rules, its `match()` also finds pairs whose immediate
  parents differ (e.g. a control state tensor-composed alongside a `ContractedDiagram`
  produced by `expand_two_mode_gates`) — see its docstring for exactly which container
  shapes it knows how to restructure.

## Helper functions

- **`expand_two_mode_gates(diagram)`** expands every `BeamsplitterGate`/`ControlledSumGate`
  leaf into its `ContractedDiagram` form (never `ControlledZGate` — see the function's
  docstring for why). `optimize()` calls this once per round when
  `assume_infinite_squeezing=True`, since `CopyRule` can only see the copy-spider pattern
  once a two-mode gate is expanded.
- **`remove_void_and_identity_nodes(graph)`** is the end-of-pipeline cleanup: it strips the
  transient `VoidDiagram` placeholders `CopyRule` leaves behind and gives `IdentityRule` one
  more pass. `optimize()` runs this exactly once, after the simplification loop reaches a
  fixed point (see {doc}`optimization`).

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
  between them) and merges them, summing phases. Two shapes: the two halves of a
  `ContractedDiagram` coupled through its `I1`/`I2`/`J1`/`J2` wiring (`match_contracted`), or
  two same-color spiders joined through a composition, possibly across identities/`Swap`
  nodes (`match_terminal`). `first_id` always survives in `apply_single`. Guards that keep it
  from overlapping other rules: a bare $(1,1)$ zero-phase identity spider is never fused
  (that's `IdentityRule`'s job); a direct half of a `ContractedDiagram` is only fused via
  `match_contracted` (restructuring its I/J coupling is a distinct operation from ordinary
  composition fusion — a contract half whose partner isn't a fusible same-color spider, e.g. a
  Q/P CSUM-style contract, is left alone); and a composition match whose endpoints are already
  claimed by a contracted match is filtered out, so the more specific contracted path wins.
- **`PassthroughRule`** — rewrites a `ContractedDiagram` that is secretly just a composition
  into an explicit `Compose`/`Tensor`/`Swap`. This applies whenever the contraction's two
  halves are bare, different-color spiders (or one buried as the last slot of a
  `TensorDiagram` — `CopyRule` debris) linked by a *single* one-way wire, with each side's
  kept arity exactly $(1,1)$: a phase-0 spider with kept arity $(1,1)$ is a ZX-calculus
  identity (a phase-0 spider forces all of its legs equal), so the "trace" the
  `ContractedDiagram` represents is really nothing more than plugging one wire into the
  other. Calling the output-owning half $a$ and the input-owning half $b$, the rewrite only
  fires once both "outer" connections it would repurpose (whatever receives $a$'s kept
  output, whatever feeds $b$'s kept input) are already proven dead (absent or a
  `VoidDiagram`) — a not-yet-simplified identity spider there is *not* enough, since it might
  still matter; this eligibility check is why the rewrite is its own rule rather than folded
  into `FusionRule`, and why it must be re-verified every pass. It then dispatches on each
  spider's own phase:

  | `a` phase | `b` phase | Result |
  | --- | --- | --- |
  | nonzero | nonzero | not yet implemented — left as a genuine `ContractedDiagram` |
  | nonzero | zero | `a` survives: `Compose([Tensor([a(1,1), Void(1,1)]), Swap()])` |
  | zero | nonzero | `b` survives: `Compose([Swap(), Tensor([b(1,1), Void(1,1)])])` |
  | zero | zero | bare, possibly-marked `Swap()` |

  Whichever spider survives is placed at arity $(1,1)$ (not its raw $(1,2)$/$(2,1)$ shape)
  next to a same-shaped `(1,1)` `Void` filling the dead slot, so `void_input_port` is always
  known deterministically rather than checked against any pre-existing wiring.
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
  Four sub-cases, each a different closed-form fold (the gate may sit on either side of the
  terminal, whichever its own arity allows — an effect's gate is upstream, a state's gate is
  downstream):
  - *Rotation* (QSpider terminal, input phase degree $\le 1$): a terminal with phase
    $c + kx$ folds $R(\theta)$ into $c - \tfrac{\tan\theta}{2}x^2 + \tfrac{k}{\cos\theta}x$
    ({cite}`nagayoshi2024zx`, Eq. (239a)-(239e)); doesn't match at odd
    multiples of $\pi/2$ (`Fourier`/`FourierInv` are never absorbed this way; `Fourier2`, a
    fixed rotation by $\pi$, is fine since $\pi$ isn't an odd multiple of $\pi/2$). This is
    the only sub-case exact for any physical state.
  - *Squeezing* (QSpider or PSpider terminal, any phase degree): a QSpider terminal with
    phase $f(x)$ folds $\mathrm{Sq}(\tau)$ into $f(x\tau)$; a PSpider terminal folds it into
    $f(x/\tau)$.
  - *Cross-color discard*: an opposite-color raw $(1,1)$ spider next to a terminal simply
    vanishes, phase unchanged (same-color $(1,1)$ pairs are `FusionRule`'s job instead).
  - *Displacement* (input phase degree $\le 1$, same $\mathbb{R}_1[X]$ restriction as
    `CopyRule`): a `DisplacementGate` next to such a terminal collapses, phase unchanged, like
    cross-color discard; a higher-degree terminal phase describes a genuinely squeezed state,
    for which this doesn't apply.

  Squeezing, cross-color discard, and displacement absorption are all only exact for an
  idealized (infinitely squeezed) terminal eigenstate, and only run when constructed as
  `TerminalAbsorptionRule(assume_infinite_squeezing=True)`; the default `False` restricts
  matching to rotation absorption. Neither the terminal nor the gate may be directly one of a
  `ContractedDiagram`'s own two halves — that shape belongs to `PassthroughRule` instead.
- **`CopyRule`** — matches a $(0,1)$/$(1,0)$ spider adjacent to a wide ($n$-input or
  $n$-output) opposite-color spider whose phase is in $\mathbb{R}_1[X]$
  (`CopyRule.is_in_R1`), and copies the narrow spider through, producing $n$ copies in a
  `TensorDiagram`:

  | Match | Result |
  | --- | --- |
  | $P(\varphi,1,n) \circ Q(g,0,1)$ | $Q(g,0,1) \otimes \cdots \otimes Q(g,0,1)$ ($n$ copies) |
  | $Q(g,1,0) \circ P(\varphi,n,1)$ | $Q(g,1,0) \otimes \cdots \otimes Q(g,1,0)$ ($n$ copies) |
  | $Q(\varphi,1,n) \circ P(g,0,1)$ | $P(g,0,1) \otimes \cdots \otimes P(g,0,1)$ ($n$ copies) |
  | $P(g,1,0) \circ Q(\varphi,n,1)$ | $P(g,1,0) \otimes \cdots \otimes P(g,1,0)$ ($n$ copies) |

  Only the *copied* spider's phase $g$ needs to be in $\mathbb{R}_1[X]$ (degree $\le 1$); the
  disappearing spider's phase $\varphi$ can be any polynomial. Unlike the other rules, its
  `match()` also finds pairs whose immediate parents differ (e.g. a control state
  tensor-composed alongside a `ContractedDiagram` produced by `expand_two_mode_gates`) — see
  {doc}`../dev_guide/rewrite_engine` for exactly which container shapes it knows how to
  restructure.

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

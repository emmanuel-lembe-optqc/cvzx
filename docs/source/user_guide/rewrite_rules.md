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

  It also has a second, unrelated match kind: **commute-and-fuse**, which moves a
  `SqueezingGate` or a linear-phase ("Disp") spider across an adjacent spider so that two
  matching elements end up next to each other and fuse. Both directions are exact identities
  (no `assume_infinite_squeezing` gate needed — unlike absorbing a genuine terminal
  state/effect, this moves a $(1,1)$ *gate* across a spider, which is exact regardless of
  squeezing) and cover four shapes:

  | Pattern | Crossing formula | Result |
  | --- | --- | --- |
  | $\mathrm{Sq}(\tau) \circ \mathrm{Spider}(f) \circ \mathrm{Sq}(\kappa)$ | $f(x) \to f(\tau x)$ (QSpider) or $f(x/\tau)$ (PSpider) | $\mathrm{Spider}(f') \circ \mathrm{Sq}(\tau\kappa)$ |
  | $\mathrm{Disp}(a) \circ \mathrm{Spider}(f) \circ \mathrm{Disp}(b)$ (opposite color) | $f(x) \to f(x+a)$ | $\mathrm{Spider}(f') \circ \mathrm{Disp}(a+b)$ |
  | $Q(f) \circ [\mathrm{Sq}\vert\mathrm{Disp}] \circ Q(g)$ | crosses into $g$ | $Q(f+g')$, mover unchanged, relocated |
  | $P(f) \circ [\mathrm{Sq}\vert\mathrm{Disp}] \circ P(g)$ | crosses into $g$ | $P(f+g')$, mover unchanged, relocated |

  Here "Spider"/$Q$/$P$ carry no phase restriction (any degree); a "Disp" is specifically a
  bare $(1,1)$ `QSpider`/`PSpider` with phase in $\mathbb{R}_1[X]$ (not a `DisplacementGate`
  node, which decomposes into exactly this shape elsewhere) and may only cross its *opposite*
  color. "Spider"/$Q$/$P$ here is never a terminal — that's `TerminalAbsorptionRule`'s
  territory, which starts at $(0,1)$/$(1,0)$, one port narrower than this rule's $(1,1)$
  scope. Implemented by leaving the moved-across element in its own slot, reset to a
  same-arity zero-phase identity (an identity commutes with everything, so this is
  diagrammatically equivalent to actually relocating it) — see `IdentityRule` for how that
  leftover identity is swept on a later pass.
- **`VoidPortPruningRule`** — drops a bare `QSpider`/`PSpider`'s last port when it's wired, by
  a single composition edge (or a chase through zero-phase identities), directly to a
  `VoidDiagram` — a real gate stranded with one port feeding, or fed by, nothing, typically
  left behind by a cross-container splice (e.g. `TerminalAbsorptionRule`'s contracted-child
  absorption, below). Restricted to bare spiders (every other gate type has fixed, paired port
  semantics — e.g. a `Swap`'s two inputs/outputs are cross-wired to each other, not
  independently prunable) and to a port already last on its side, so neither it nor its
  remaining siblings ever need renumbering. Refuses to prune a node already down to its last
  port if its phase is non-zero — that would leave a $(0,0)$ node with a phase nothing binds
  to, silently discarding whatever physical contribution it represented.
- **`FourierNormalizationRule`** — matches a `Fourier`/`FourierInv`/`Fourier2` adjacent to a
  `PhaseRotationGate`, `BeamsplitterGate`, or `SqueezingGate` and folds it into that gate's
  parameter.
- **`TerminalAbsorptionRule`** — matches a gate adjacent to a $(1,0)$-effect or
  $(0,1)$-state `QSpider`/`PSpider` terminal, and folds the gate into the terminal's phase.
  Six sub-cases, each a different closed-form fold (the gate may sit on either side of the
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
  - *Contracted-child absorption*: a terminal composed directly into a node (`state1`) that is
    itself one of a `ContractedDiagram`'s own two children. `state1` is eligible only if
    exactly one of its raw ports is used internally by the contraction and every other raw
    port (besides the one connecting to the terminal) already traces to a `VoidDiagram` —
    otherwise a live wire would be silently dropped, so the match doesn't fire. Same color as
    the terminal: phases simply add. Opposite color: `state1`'s phase must be in
    $\mathbb{R}_1[X]$, $ax$ say; if the terminal's phase is $f(x)$ it becomes $f(x+a)$. Either
    way `state1` disappears, the terminal takes over its exact slot in the contraction (with
    the contraction's arity re-derived from the terminal's own $(1,0)$/$(0,1)$ shape), and the
    terminal's own vacated slot becomes a `VoidDiagram`.
  - *Bare-cap fusion* (`bare_cap_fusion`): the mirror image of contracted-child absorption —
    a `ContractedDiagram` where one half is **already** a genuine $(0,1)$/$(1,0)$ terminal
    directly occupying `first_id`/`second_id`, with its single port entirely consumed by the
    contraction itself (nothing external connects to it at all, unlike contracted-child
    absorption's `state1`, which still has one live connecting port). No terminal needs to
    chase in from outside — the shape is self-contained. Same color: phases add (exact,
    unconditional, like ordinary same-color `FusionRule`). Opposite color: the shift formula,
    gated on `assume_infinite_squeezing`. Either way the whole `ContractedDiagram` collapses
    into just the surviving partner. Refuses to fold when the survivor's own remaining role
    would *also* collapse to $(0,0)$ — that's a genuine closed scalar (a state composed
    directly into its own opposite effect), which this library has no scalar bookkeeping to
    represent.

  Squeezing, cross-color discard, displacement, contracted-child absorption, and the
  opposite-color case of bare-cap fusion are all only exact for an idealized (infinitely
  squeezed) terminal eigenstate, and only run when constructed as
  `TerminalAbsorptionRule(assume_infinite_squeezing=True)`; the default `False` restricts
  matching to rotation absorption plus bare-cap fusion's same-color (exact, unconditional)
  case. The terminal itself may never be directly one of a `ContractedDiagram`'s own two
  halves (it must stay a genuine standalone leaf); `state1` being one of those two halves is
  precisely the contracted-child/bare-cap-fusion sub-cases above rather than
  `PassthroughRule`'s territory, since a real terminal or phase-bearing spider — not a bare
  identity — is being folded in.
- **`CopyRule`** — matches a $(0,1)$/$(1,0)$ spider adjacent to a wide ($n$-input or
  $n$-output) opposite-color spider whose phase is in $\mathbb{R}_1[X]$
  (`CopyRule.is_in_R1`), and copies the narrow spider through, producing $n$ copies in a
  `TensorDiagram`:

  | Match | Result |
  | --- | --- |
  | $P(1,n, \varphi) \circ Q(0,1, g)$ | $Q(0,1,g) \otimes \cdots \otimes Q(0,1,g)$ ($n$ copies) |
  | $Q(1,0,g) \circ P(n,1,\varphi)$ | $Q(1,0,g) \otimes \cdots \otimes Q(1,0,g)$ ($n$ copies) |
  | $Q(1,n,\varphi) \circ P(0,1,g)$ | $P(0,1,g) \otimes \cdots \otimes P(0,1,g)$ ($n$ copies) |
  | $P(1,0,g) \circ Q(n,1,\varphi)$ | $P(1,0,g) \otimes \cdots \otimes P(1,0,g)$ ($n$ copies) |

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

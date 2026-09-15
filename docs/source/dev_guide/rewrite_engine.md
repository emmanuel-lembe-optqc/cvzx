# The rewrite rule engine

## Why rules operate on the graph, not the `Diagram` tree

An earlier version of this codebase implemented rewrite rules by walking the `Diagram` tree
directly (see the `Remove legacy class-based rewrite rules` commit). That approach has a
structural problem: two leaves that are logically adjacent in the circuit — one feeds
straight into the other, nothing in between — are not necessarily *siblings* of one flat
container. Building a circuit the natural way (e.g. an ancilla prepared mid-circuit via a
width-changing sub-`CompositionDiagram` nested inside one row of a wider `TensorDiagram`)
easily produces two same-row, back-to-back 1-mode gates that live in different immediate
parents purely because of how the tree happens to be nested. A tree-walking rule that only
ever compares siblings never sees this pair at all.

`to_graph()` sidesteps the problem by resolving every wire down to the two **leaf** nodes it
actually connects, regardless of container nesting, and recording that as a `"composition"`
edge directly between them. A rule's `match()` can then simply scan
`graph.edges(data=True)` for `edge_type == "composition"` and check the two endpoint leaves
— no tree traversal, no notion of "sibling" required. `CopyRule` is the rule that leans on
this hardest — see "Which container shapes cross-container matching supports" below for
exactly which container shapes this lets it restructure that a tree-walking version could not.

`ChainReductionRule` and `TerminalAbsorptionRule` lean on the same edge-scanning approach —
both chase across container boundaries and past runs of identity spiders/`Swap` nodes to find
their real neighboring leaf, exactly like `CopyRule` — so cross-container matching is the norm
across the rule set, not a `CopyRule`-only special case.

The corollary is that **normalization is still useful, just not for matching correctness** —
{doc}`normalization` exists so that every diagram settles into the same canonical shape
(same micro-layers, same `connectivity`-dict convention between stages) for rules — and
non-rule consumers like `cvzx.lowering.bridges.mqc3.to_circuit_repr` — to reason about, rather
than because any rule would otherwise fail to find a match.

### Which container shapes cross-container matching supports

`CopyRule.match()` and `TerminalAbsorptionRule.match()` both scan candidate terminal
(state/effect) nodes and chase outward past identities/`Swap` nodes via `_chase_identity_chain`
rather than walking one container's flat `sub_diagram_ids` list — see above for why. Two
shapes come out of that scan:

- **Same immediate parent**: both nodes are direct, adjacent entries of one flat
  `CompositionDiagram` — resolved via that container's own `sub_diagram_ids`/`connectivity`.
- **Cross-container** (`CopyRule` only): the two nodes have different immediate parents. Only
  combinations `CopyRule` knows how to restructure are matched: the disappearing spider's
  parent must be a `TensorDiagram` (direct list substitution) or a `ContractedDiagram`
  (`first_id`/`second_id` substitution); the copy spider's parent must be a `TensorDiagram` or a
  flat `CompositionDiagram` where the copy spider sits at one of the two ends (a state must be
  first, an effect must be last, so removing it never requires splicing two neighbors
  together).

`TerminalAbsorptionRule` uses the identical candidate-and-chase scan (any terminal, chase its
one wire past identities/`Swap` nodes to the real neighboring gate) but never restructures a
container itself — the gate and every passthrough crossed are simply reset in place to identity
spiders, so no cross-container substitution logic is needed there. Neither rule ever matches a
pair where either endpoint is directly one of a `ContractedDiagram`'s own two halves — that
shape belongs to `PassthroughRule` instead (see below).

## The `RewriteRule` contract

```python
class RewriteRule(ABC):
    @abstractmethod
    def match(self, graph: nx.DiGraph, registry: GateRegister) -> list[dict]: ...

    @abstractmethod
    def apply_single(self, graph: nx.DiGraph, match: dict) -> None: ...

    def apply_rule(self, graph: nx.DiGraph, registry: GateRegister) -> nx.DiGraph:
        for match in self.match(graph, registry):
            self.apply_single(graph, match)
        return graph
```

`match()` returns a list of independent matches — each one a `dict` carrying whatever
`apply_single` needs to know (node IDs, container IDs, computed replacement values, ...);
there is no fixed schema across rules, each rule documents its own match dict's keys.
`apply_single()` mutates `graph` in place for exactly one match. `apply_rule()` (defined
once, on the base class, not overridden) is the only place that ties the two together for a
whole pass; `cvzx.passes.optimize.optimize` calls `match()` + `apply_rule()` per rule, per round,
until nothing matches.

```{note}
Until recently, the abstract base class declared `apply_single(self, graph, match: list)`
— a `list`, not a `dict` — despite every real subclass, and the base class's own
docstring, always treating `match` as a single `dict` record. It was a copy-paste
artifact that happened to be harmless at runtime (Python doesn't enforce parameter type
annotations), but it meant `mypy` couldn't verify any subclass's `apply_single` against
the contract it was actually supposed to satisfy. If you add a new rule, match the
*docstring*, not just whatever the abstract signature currently says — and if you find
another place where they disagree, that is a bug in the abstract signature, not license
to make your subclass match it.
```

### `apply_rule`'s round loop

`apply_rule` (defined once on `RewriteRule`, never overridden) runs up to
`_APPLY_RULE_MAX_ROUNDS` rounds. Each round: find every match via `match()`, group the
matches by `container_id`, apply each group's matches in *reverse* position order (so an
earlier removal can't shift a later match's own index), then rebuild `graph.registry` so the
next round's `match()` sees the graph as it now stands. A round that applied at least one
match is followed by another `match()` call, and another round runs if that finds anything.

More than one round is sometimes needed because applying a match can turn a node that was
blocking another, still-unapplied match into a pass-through: two independent chains that both
need to chase through the same shared node (a `Swap`, say) can only have ONE of them claim
that node per `match()` call — `match()`'s own bookkeeping treats a node consumed by one match
as unavailable to any other match found in the same call, even when the pattern would allow
folding both, one after the other, once the first is voided. A single round would leave the
second chain sitting there unreduced even though nothing about the graph actually prevents
folding it too.

`_APPLY_RULE_MAX_ROUNDS` is deliberately small rather than large enough to always reach a
rule's own local fixed point in one call: `optimize()`'s own outer `_simplify_to_fixed_point`
loop already re-invokes every rule's `apply_rule` again on the very next pass whenever the
current pass changed anything, and keeps doing so until a whole pass changes nothing anywhere
— so whatever one `apply_rule` call doesn't finish is picked up there at no extra total cost.
Reaching the cap is routine (logged at debug level, not warned), and it exists to bound how
much internal work one call can do chasing a single pathological match, e.g. a
`ChainReductionRule` closure that would otherwise try to resolve its way through an arbitrarily
long run of chained `Swap` nodes in one round (`_trace_closure_leftovers` caps that at one
`Swap`-like bundle per match for the same reason).

## Keeping `GateRegister` in sync

`GateRegister` indexes graph nodes by category purely as a performance cache; it is not the
source of truth (the graph is), and nothing updates it automatically. A rule's
`apply_single` that adds or removes nodes is responsible for calling
`registry.add_node()`/`remove_node()` itself, or the caller must rebuild the registry with
`build_from_graph()` before the next `match()` call. `cvzx.passes.optimize._simplify_to_fixed_point`
rebuilds the registry fresh right after a rule actually applies a change to the graph —
never merely because a pass moves on to the next rule, since a rule whose `match()` returns
nothing leaves the graph (and therefore every category the registry indexes) untouched, so a
rebuild in that case would be pure wasted work for no change in behavior. It still rebuilds
from scratch, rather than patching incrementally, whenever a rule *did* just mutate the
graph: that's simpler and cheap enough at the circuit sizes this library targets than
auditing every rule's node bookkeeping for registry-sync correctness.

## `VoidDiagram`: a transient bookkeeping leaf, not a calculus generator

`VoidDiagram` (`ir/base.py`) has no counterpart in the CV ZX calculus paper — it exists
purely so that `CopyRule` (and, more generally, any rule that needs to shrink a
container's *content* without shrinking its *arity* mid-pass) can leave a same-shaped
placeholder in a slot instead of forcing an immediate, potentially cascading re-derivation of
every ancestor container's port count. `cvzx.backends.nx.rules.remove_void_and_identity_nodes`
is the only thing that ever removes these, and `optimize()` runs it exactly once, at the very
end of the pipeline (see {doc}`architecture`) — a `Diagram` produced mid-pipeline (e.g. the
`.diagram` field on `OptimizeResult`, or any partial round) may still contain them.

## Keeping `Diagram.id` unique across reconstruction

`Diagram.__init__` (`ir/base.py`) assigns every freshly-constructed object the next value of a
single, class-level `Diagram._next_id` counter, guaranteeing every `Diagram` in a process gets
a distinct id — *except* that `reconstruct_proper_node` (`backends/nx/graph.py`,
`backends/rx/graph.py`) deliberately overrides this for a node in `registry.measurement_nodes`,
stamping `.id` to that node's original graph id instead (needed so a measurement leaf's id
survives a `to_diagram()`/`to_graph()` round-trip, since other code refers back to it by that
exact value — e.g. `param_measurement_map`). Despite the name, `registry.measurement_nodes` is
purely shape-based (any `(1, 0)`-arity node — see `_is_measurement`), so this override fires
far more often than the name suggests.

Because that override never advances `_next_id`, without care it can silently let `_next_id`
independently reach a value already claimed this way, handing that same integer to a second,
unrelated object — `to_graph()` then keys graph nodes by `diagram.id`, so the two colliding
objects merge into a single malformed graph node (a `nx.Graph.add_node`/`rx` node-weight
update on an id that already exists updates that node in place rather than creating a second
one). Every call site that force-sets `.id` from a specific value **must** call
`Diagram._reserve_id(that_value)` right after, and `to_diagram()` additionally reserves past
the whole graph's max node id once up front, before any reconstruction begins — covering
override order regardless of which node the recursive walk visits first.

## Chase, expansion, and algebra helpers (`cvzx.utils.helpers`)

### `is_chase_passthrough`: why a square `VoidDiagram` counts as transparent

A passthrough is an identity/wiring diagram, a `Swap`, or a same-arity (square) `VoidDiagram` —
all three are pure wire-routing with no bearing on whatever pattern `_chase_identity_chain` is
chasing through them. A square `VoidDiagram` belongs alongside a bare identity spider and a
`Swap` because it *is* one of them, mid-chase: every rule that installs one in place of a chain
member it has already decided is a pure pass-through does so via a bare type relabel, at the
same node, with the same arity — nothing about what the node physically does changes, only its
label does. Treating it as opaque instead (which is what happened before this check was added)
makes a voided `Swap` a permanent one-way wall: since `_simplify_to_fixed_point` re-runs every
rule to a fixed point, a chain that shares a `Swap` with another, already-reduced chain would
otherwise never become reachable on any later pass, even though the wire it needs to cross is
exactly as pass-through as it always was. A `VoidDiagram`'s arity must still be checked rather
than assumed, since one installed by some other mechanism entirely (a vanished state/effect,
arity `(0, 1)` or `(1, 0)`) is a genuine dead end, not a wire to chase through, and must stay
opaque.

### `expand_two_mode_gates`: the Squeezing rule and why CZ stays compact

Only ever called under `assume_infinite_squeezing=True` (see {doc}`../user_guide/optimization`),
which licenses two things. `ControlledZGate` is deliberately left unexpanded, since its own
decomposition sandwiches a Fourier gate between its copy spider and the target mode, so
`CopyRule` can't reach across it — expanding CZ would unlock no new reduction, unlike BS/CSUM,
whose expansions expose a bare copy spider `CopyRule` can act on directly. And a biased
`ControlledSumGate(gain=g != 1)` is normalized to `gain=1` before expanding (the "Squeezing
rule") rather than expanded via its own squeeze-CSUM-unsqueeze decomposition: under infinite
squeezing the modes involved are idealized eigenstates, for which CSUM(g) and CSUM(1) act
identically, so the bias can simply be dropped instead of carried through as an explicit pair of
`SqueezingGate` instances. This also sidesteps a real limitation of `.expand()`'s biased
decomposition — the squeeze gates it introduces land in a different container than whatever
state feeds the CSUM, so `TerminalAbsorptionRule`/`CopyRule` (which only match within a single
container) can't reach through them, and a biased CSUM fed by a reducible ancilla would
otherwise fail to reduce at all.

### `flatten_expanded_composition`: keeping expansion inside one flat container

`to_graph()`'s composition-edge resolution (`find_node_by_external_output`/
`find_node_by_external_input` in `cvzx.backends.nx.graph`) recurses into a composition's
TENSOR/CONTRACTED children (both store an `external_*_mapping`), but never into a
`CompositionDiagram` nested directly inside another `CompositionDiagram`, since composition
containers don't store one — a flat composition's own boundary edges are resolved from
`sub_diagram_ids`/`connectivity` directly. Left un-flattened, any wire crossing such a nested
boundary is silently dropped — exactly what `BeamsplitterGate.expand()`'s balanced case
produces (a `CompositionDiagram` of two expanded `ControlledSumGate` instances and a
`TensorDiagram` of squeezing gates). `flatten_expanded_composition` splices any child that
expanded into its own `CompositionDiagram` directly into the parent's flat list, re-indexing
`connectivity` to match the new, flat positions.

### `simplify_reduced_value`: keeping `ChainReductionRule`'s combined values exact

`ChainReductionRule` combines a chain's phases and gate parameters with plain `+`/`*`, which
never algebraically reduces the result — e.g. `sin(x)**2 + cos(x)**2` stays exactly that rather
than collapsing to `1`, and two chained gates whose parameters are exact negatives of each other
may not compare equal to the identity's `0` by structural equality alone. Running the combined
value through `sympy.simplify()` first catches both. The function deliberately does *not*
coerce a simplified `Expr` back to a plain Python number even when it collapses to one with no
free symbols left (e.g. a chain of exact multiples of `pi`) — `Expr.is_number` is true for any
such exact irrational constant, not just literal numbers, so that coercion would silently lose
exactness for phases like `pi/6 + pi/5 + pi/7`.

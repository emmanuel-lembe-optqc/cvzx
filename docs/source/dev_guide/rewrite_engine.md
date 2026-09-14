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
this hardest: its docstring is the most detailed explanation in the codebase of exactly
which container shapes this lets it restructure that a tree-walking version could not.

`ChainReductionRule` and `TerminalAbsorptionRule` lean on the same edge-scanning approach —
both chase across container boundaries and past runs of identity spiders/`Swap` nodes to find
their real neighboring leaf, exactly like `CopyRule` — so cross-container matching is the norm
across the rule set, not a `CopyRule`-only special case.

The corollary is that **normalization is still useful, just not for matching correctness** —
{doc}`normalization` exists so that every diagram settles into the same canonical shape
(same micro-layers, same `connectivity`-dict convention between stages) for rules — and
non-rule consumers like `cvzx.lowering.bridges.mqc3.to_circuit_repr` — to reason about, rather
than because any rule would otherwise fail to find a match.

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

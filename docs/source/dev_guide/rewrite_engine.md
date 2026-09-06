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

The corollary is that **normalizing same-container matching is still useful** —
{doc}`normalization` exists precisely so that, after normalization, the common case (two
directly causally-adjacent leaves) *is* also a same-container, same-`sub_diagram_ids` case,
which keeps most rules' `match()` logic simple; only `CopyRule` needs the general
cross-container path.

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
whole pass; `cvzx.optimize.optimize` calls `match()` + `apply_rule()` per rule, per round,
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
`build_from_graph()` before the next `match()` call. `cvzx.optimize._simplify_to_fixed_point`
rebuilds the registry fresh before every rule in every pass for exactly this reason — it is
simpler and cheap enough at the circuit sizes this library targets to rebuild than to audit
every rule's node bookkeeping for registry-sync correctness.

## `VoidDiagram`: a transient bookkeeping leaf, not a calculus generator

`VoidDiagram` (`base_gates.py`) has no counterpart in the CV ZX calculus paper — it exists
purely so that `CopyRule` (and, more generally, any rule that needs to shrink a
container's *content* without shrinking its *arity* mid-pass) can leave a same-shaped
placeholder in a slot instead of forcing an immediate, potentially cascading re-derivation of
every ancestor container's port count. `cvzx.nx_rewrite_rules.remove_void_and_identity_nodes`
is the only thing that ever removes these, and `optimize()` runs it exactly once, at the very
end of the pipeline (see {doc}`architecture`) — a `Diagram` produced mid-pipeline (e.g. the
`.diagram` field on `OptimizeResult`, or any partial round) may still contain them.

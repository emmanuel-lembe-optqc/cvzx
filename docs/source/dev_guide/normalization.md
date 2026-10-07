# Type-1/type-2 stage normalization

`cvzx.passes.normalize.normalize_diagram(diagram)` rewrites an arbitrary compact-form
`Diagram` into a canonical `CompositionDiagram` of alternating stages:

- a **type-2** stage is a `TensorDiagram` containing exactly one "wide" leaf (any generator
  touching more than one mode — a 2-mode gate, `Swap`, ...) plus one identity wire per every
  other currently-open mode;
- a **type-1** stage is a `TensorDiagram` where every row is a `CompositionDiagram` chaining
  together whatever 1-mode gates (or state/effect/`VoidDiagram` leaves) act on that row
  between the type-2 touches on either side, or a bare identity wire on an untouched row.

```text
input:  a 3-mode circuit with one 2-mode gate touching modes 1 and 2,
        with a 1-mode gate before it on mode 1 and one after it on mode 2

  row0 ──────────────────────────────────────────────────────────
  row1 ──[ 1-mode gate ]──┬──────────────────┬───────────────────
  row2 ───────────────────┤   2-mode gate    ├──[ 1-mode gate ]──
  row3 ──────────────────────────────────────────────────────────

normalized: (type-1) ⊕ (type-2) ⊕ (type-1)

  stage 1 (type-1)        stage 2 (type-2)        stage 3 (type-1)
  ┌──────────────┐        ┌──────────────────┐    ┌──────────────┐
  │ identity     │        │ identity         │    │ identity     │
  │ [1-mode gate]│──────▶│  2-mode gate     │───▶│ identity     │
  │ identity     │        │                  │    │ [1-mode gate]│
  │ identity     │        │ identity         │    │ identity     │
  └──────────────┘        └──────────────────┘    └──────────────┘
```

This is entirely an internal normal form — it changes nothing observable about the circuit
(the result is semantically equal to the input), it just guarantees a structural property
several rewrite rules rely on.

## Why this exists

`ChainReductionRule` and `TerminalAbsorptionRule` — like `CopyRule` — scan the graph's own
`"composition"` edges directly rather than walking `sub_diagram_ids` siblings, so none of the
three is actually limited to same-immediate-parent matches: all three chase across
`TensorDiagram`/`ContractedDiagram` container boundaries, and past any run of identity
spiders or `Swap` nodes sitting directly in the path, to find the real neighboring leaf to
check (see {doc}`rewrite_engine`).

So normalization isn't load-bearing for *finding* matches the way it once was. What it still
buys is a predictable, canonical shape to reason about in the first place: instead of an
arbitrarily-nested tree where the same causal relationship (e.g. an ancilla prepared
mid-circuit via a width-changing sub-`CompositionDiagram` nested inside one row of a wider
`TensorDiagram`) can show up in any number of container shapes, every diagram normalizes to
the same alternating type-1/type-2 form, with the same micro-layer ordering and the same
`connectivity`-dict convention between stages — one shape every rule, present and future, can
assume rather than re-derive, and the same shape `cvzx.lowering.bridges.claveles.to_circuit_repr`
relies on for its own stage-by-stage walk (see {doc}`circuit_conversion
<../user_guide/circuit_conversion>`). This is also why `cvzx.passes.optimize.optimize` calls
`normalize_diagram` at the start of *every* round, not just once at the start of the whole
pipeline: each round's rewriting can re-nest the diagram in ways worth flattening back out
before the next round, even though the rules themselves no longer strictly need that to see
across the result.

## How it works, briefly

The algorithm operates on the leaf-level wire graph from `to_graph()` (every `"composition"`
edge already connects two fully-resolved leaves regardless of container nesting — see
{doc}`rewrite_engine`). Leaves are classified *wide* (`max(num_inputs, num_outputs) > 1`) or
*narrow* (everything else):

1. Each leaf gets a greedy topological "micro-layer" index: a wide leaf claims an entire
   layer to itself; narrow leaves pack into the earliest layer not claimed by a wide leaf,
   without violating causal (wire) order.
2. Wide leaves, in layer order, become the sequence of type-2 stages. Narrow leaves are
   partitioned into the (possibly empty) runs strictly between consecutive type-2 stages
   (or before the first / after the last); each run becomes one type-1 stage.
3. Each stage is built by tracking, per currently-open row, an opaque *token* identifying
   which leaf/port (or external input) last produced that wire — never mutated by identity
   filler insertion, so a leaf's true predecessor is always found correctly regardless of how
   many filler wires end up structurally in between.
4. Consecutive stages are joined by an explicit `connectivity` dict (stages are free to
   reorder rows internally — e.g. a 2-mode gate whose two touched modes weren't adjacent —
   so this is never assumed to be the identity map). The first stage's row order matches the
   diagram's external input order directly (nothing precedes it to permute against); if the
   last stage's natural output order doesn't already match the diagram's real external
   output order, one small trailing all-identity stage is appended to fix that up.

## Scope

`normalize_diagram` is primarily designed for **compact-form** diagrams — before
`expand_two_mode_gates` turns a two-mode gate into its `ContractedDiagram` form — which is
exactly why `optimize()` always normalizes *before* the (optional) expansion step in each
round, not after (see {doc}`architecture`). But a `ContractedDiagram` can still legitimately
show up by the time a *later* round's normalization runs — a contraction one round didn't
finish reducing survives into the diagram `optimize()` hands to the next round's
`normalize_diagram()` call — so it isn't simply out of scope.

A **top-level** `ContractedDiagram` (one not itself nested inside another one's `first`/
`second`) is treated as a single opaque leaf: its own kept (external) `num_inputs`/
`num_outputs` — already tracked on the graph node itself — are what the leaf-level dependency
graph and stage-building machinery above see, while `first`/`second` (and whatever they
contain) are reconstructed as one atomic unit via `reconstruct_contracted_node`, never
decomposed into separate rows. This isn't just a simplification: a contraction's two halves
generally *can't* be represented as independent leaves at all, since plain `Tensor`/`Compose`
nesting has no way to express "these two, though far apart in the normalized stages, still
share an internal wire" — that shape is precisely what `ContractedDiagram` exists to name.
Since a raw `"composition"` edge from `to_graph()` lands directly on `first`/`second`
themselves (never on the `ContractedDiagram` node — see {doc}`rewrite_engine`), a small
reverse-mapping step (`_build_contracted_port_remaps`) redirects such an edge to the
contraction's own external port before the leaf-level predecessor map is built.

This still bails out — leaving the input diagram completely unchanged, the same fallback used
unconditionally before this capability existed — in the one shape not handled: a top-level
contraction whose own kept port resolves down through *another*, nested `ContractedDiagram`
instead of a genuine leaf.

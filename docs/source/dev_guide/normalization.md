# Type-1/type-2 stage normalization

`cvzx.normalize_diagram.normalize_diagram(diagram)` rewrites an arbitrary compact-form
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

`ChainReductionRule` and `TerminalAbsorptionRule` only ever look for two matchable leaves
that are *directly adjacent elements of one flat `CompositionDiagram`* (same immediate
parent, consecutive `sub_diagram_ids`) — the simple case, deliberately, so that adding a new
rule of this kind never has to reimplement cross-container matching. But a diagram built the
natural way (e.g. an ancilla prepared mid-circuit via a width-changing sub-`CompositionDiagram`
nested inside one row of a wider `TensorDiagram`) can easily produce two same-row,
back-to-back 1-mode gates that are *not* siblings of one flat composition purely because of
how the tree happens to be nested — so those two rules would never see them. Normalizing
first makes "same immediate parent" matching sufficient everywhere it's used, instead of
requiring every rule to grow the general cross-container matching logic `CopyRule` needed
(see {doc}`rewrite_engine`). This is also why `cvzx.optimize.optimize` calls
`normalize_diagram` at the start of *every* round, not just once at the start of the whole
pipeline: each round's rewriting can re-nest the diagram in ways that need renormalizing
before the next round's rules can see across them again.

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

`normalize_diagram` only understands **compact-form** two-mode gates (a single leaf), not
the `ContractedDiagram` form `expand_two_mode_gates` produces. If the graph contains any
`ContractedDiagram`, it conservatively returns the input diagram unchanged rather than risk
mis-normalizing internal wire-bending it doesn't attempt to trace through — which is exactly
why `optimize()` always normalizes *before* the (optional) expansion step in each round, not
after (see {doc}`architecture`).

# Benchmarking optimization quality

`optimize()` is meant to shrink a diagram, but until now there was no way to
measure by how much: no benchmark suite, no metrics, nothing beyond the
prose complexity note in {doc}`../user_guide/optimization`. This page
introduces `cvzx.utils.metrics` and the `benchmarks/` suite built on it,
which measure **optimization quality** specifically -- how much smaller
`optimize()` makes a diagram -- as distinct from compiler wall-clock
performance, `networkx`-vs-`rustworkx` backend comparison, or simulation/
visualization cost, none of which this page covers.

## Metric definitions

Every metric is computed on a `CVZXGraph` (the same representation
`optimize()` itself works on), via one entry point,
`cvzx.utils.metrics.compute_metrics(diagram)`, returning a `DiagramMetrics`
snapshot:

| Metric | Meaning |
|---|---|
| `spiders` | `QSpider`/`PSpider` leaves, excluding identity wires |
| `gates` | Surviving un-expanded `CompactDiagram` gate leaves |
| `generators` | `spiders` + `gates` + other real proper leaves (`Swap`, `Fourier*`) |
| `nodes` / `edges` | Raw graph totals (also counts container bookkeeping) |
| `non_clifford_phases` | Leaves with phase-polynomial degree >= 3 (default) |
| `depth` | Stage count via `normalize_diagram()`, or `None` |

`compare_metrics(before, after)` returns a `MetricsComparison` with
`.ratio(field)` (`after / before`) and `.reduction(field)` (`1 - ratio`,
i.e. the fraction a metric shrank by), both `None` when undefined (either
side `None`, or `before == 0`).

### Why exclude bookkeeping leaves

`count_spiders`/`count_generators` deliberately exclude two kinds of leaf
that look like real content but aren't:

- **Identity/wiring spiders** -- `normalize_diagram()` inserts a filler
  identity spider on every otherwise-untouched row of every stage it
  builds, as part of its own canonical form. This is bookkeeping, not
  circuit content the input diagram or `optimize()` "added".
- **`VoidDiagram` placeholders** -- a same-arity marker a rewrite rule
  leaves behind when it eliminates a state/effect elsewhere (see
  `VoidDiagram`'s own docstring). Its docstring says it "should never
  survive past `optimize()`'s return value" -- true of the *cleaned graph*
  (`OptimizeResult.graph`), but **not** of `OptimizeResult.diagram`, which
  is captured right *before* that cleanup pass runs (see `optimize()`'s
  own docstring: the cleaned graph isn't guaranteed losslessly
  representable as a `Diagram`, so `.diagram` is the pre-cleanup
  snapshot).

This was not a theoretical concern: building this module without the
exclusion, a diagram where `optimize()` genuinely eliminated an ancilla and
merged two gates came out showing *higher* spider and generator counts
after optimization than before, purely from this filler. See the worked
example below.

`count_nodes`/`count_edges` are **not** filtered this way -- they're
included mainly for parity with `CVZXGraph.__repr__` and documented as a
weaker signal, since they also move with container-flattening cleanup, not
just genuine circuit simplification.

### The non-Clifford threshold

`CubicPhaseGate` is the only gate in `cvzx.ir.gates` tagged
`spider_type = "non_gaussian"`; every other gate (including `ArbitraryGate`,
which decomposes into rotation/squeezing only) expands to phase degree
<= 2. `count_non_clifford_phases` defaults to `degree_threshold=3`,
matching that boundary -- the CV-ZX analogue of qubit ZX calculus's
T-count.

```{note}
This is a **different** boundary from the degree <= 1 (affine) check
`TerminalAbsorptionRule`/`CopyRule` use internally for their own exact-
identity applicability (`is_in_R1`). That threshold exists because those
rules' specific algebra breaks past degree 1, not because a degree-2 phase
is somehow "non-Clifford". Pass `degree_threshold=2` to
`count_non_clifford_phases`/`compute_metrics` for that stricter reading
instead.
```

A `QSpider`/`PSpider` leaf's degree is read directly off its `ZxPoly`
phase. A compact gate's degree is looked up by its graph `type` name
(`_NON_GAUSSIAN_COMPACT_DEGREE`), **not** via the gate's own `phase`
attribute or `spider_type` field: `to_graph()` deletes every node's
`"diagram"` attribute before returning (so the original gate object isn't
reachable from the graph at all), and even where a compact gate's `phase`
graph attribute exists, it holds the gate's raw parameter (`gamma`,
`gain`, `theta`, ...), not a `ZxPoly`. Both of these were verified by
reading `to_graph()`/`_add_proper_node` directly, not assumed.

### Depth

`diagram_depth()` reuses `normalize_diagram()`'s own public return value
rather than reaching into its private `stage_records`: each element of the
`CompositionDiagram` it returns is, by construction, exactly one stage, so
depth is `len(result.diagrams)`. Returns `None` if the diagram contains a
`ContractedDiagram` (which `normalize_diagram` conservatively leaves
unchanged -- no stage count is defined there), and `0` for a leafless
diagram.

## Worked example

`four_mode_csum` (`benchmarks/fixtures.py`): 4 modes, one ancilla feeding a
CSUM, one bare CSUM, and two mergeable CSUM(0.5) gates.

Under `assume_infinite_squeezing=False` (only exact identities apply):

| metric | before | after | reduction |
|---|---|---|---|
| gates | 6 | 5 | 0.17 |
| generators | 7 | 6 | 0.14 |
| spiders | 1 | 1 | 0.00 |

Exactly the two mergeable CSUM(0.5) gates combine into one -- the only
reduction available without the infinite-squeezing assumption.

Under `assume_infinite_squeezing=True`, `gates` drops further (6 -> 2) and
`depth` becomes `None` (the ancilla-consuming `CopyRule` reduction leaves a
`ContractedDiagram` behind), but `generators` and `spiders` **increase**
(7 -> 7, roughly unchanged, and 1 -> 5 respectively) despite the ancilla
and one CSUM being genuinely eliminated. This is not a metric bug: under
this setting, `optimize()` unconditionally expands every surviving CSUM/BS
gate into its own multi-spider decomposition (`expand_two_mode_gates`) so
`CopyRule` can see the copy-spider structure it needs -- trading a single
compact gate for several elementary spiders. Real simplification (ancilla
+ redundant gate eliminated) and a representation-format change (compact
gate -> spider expansion) happen in the same pass, and the raw counts
conflate them. Read `gates` (compact-gate survival) as the cleaner signal
for this setting; a wash or even an increase in `spiders`/`generators`
does not mean nothing was optimized.

## How to run

```console
$ python -m benchmarks.run_benchmarks
$ python -m benchmarks.run_benchmarks --format json
```

Runs every fixture in `benchmarks/fixtures.py` under both
`assume_infinite_squeezing` settings, printing a before/after/ratio/
reduction table per fixture/setting plus an arithmetic-mean-reduction
summary across fixtures (mean of `reduction`, not a geometric mean of
`ratio`, for interpretability as a "% shrank by" figure).

Add a fixture by adding a zero-argument factory function to
`benchmarks/fixtures.py`'s `FIXTURES` dict; each call must return a fresh
diagram (`Diagram.id_counter` is a shared global counter, so reusing one
instance across runs would corrupt node identities).

## Relationship to CI

The suite is **dev-invoked only** -- not wired into CI. There's no existing
perf/benchmark job to match (the CI workflow runs lint, type-check, tests,
and a docs build only), and no established baseline yet to gate a
regression check against. `tests/utils/test_metrics.py` (ordinary unit
tests for the metrics module itself) does run in CI as part of the normal
test suite -- only the benchmark *report* is dev-invoked.

## Caveats

- `diagram_depth()` returns `None` for a diagram containing a
  `ContractedDiagram` -- common after an ancilla-eliminating `optimize()`
  run under `assume_infinite_squeezing=True`.
- Metrics are always computed from `OptimizeResult.diagram`, never
  `OptimizeResult.graph` -- see `optimize()`'s own docstring for why the
  cleaned graph isn't guaranteed losslessly representable as a `Diagram`.
- A diagram built entirely from bare identity wires but already expressed
  as a `CompositionDiagram` can misreport `diagram_depth` as its stage
  count instead of `0` -- `normalize_diagram` elides such wiring
  internally and returns the (still-composed) input unchanged, which
  can't be distinguished from a genuine single-stage result from the
  outside. Not a concern for real circuits, which always have substantive
  leaves.

# Changelog

All notable changes to this project are documented in this file.

This project has not yet made a tagged release; the entries below summarize
development on `main` to date, grouped by area rather than by commit.

## [Unreleased]

### Added

- **Pruning of near-identity Gaussian gates** (`cvzx.passes.pruning.prune_small_gaussian_gates`, and
  `optimize(..., prune_epsilon=, prune_keep=)`): per-type distances from identity, thresholds per gate type,
  a `keep` callback for outside cost models, feedforward targets and symbolic gates protected. Pruned gates become
  identity wires that are dropped from compositions with their wiring maps composed. User guide page
  `user_guide/pruning.md`.

### Changed

- **claveles replaces mqc3** as the circuit SDK: `cvzx.lowering.bridges.claveles` (`from_circuit_repr`,
  `to_circuit_repr`, `placed_operations`), `graph_to_dependency_dag(backend="claveles")` on claveles' `DependencyDAG`,
  and the `[claveles]` / `[dev]` extras install `claveles-core` from OptQC's `sdk-optqc`. claveles keeps an operation's
  modes on its placement and places input states as `StatePreparation` operations; the bridge follows both. The
  unused `grpcio`, `protobuf` and `requests` dependencies (mqc3's) are dropped.

### Fixed

- `intrinsic.Arbitrary` crosses the bridge with `lam -> -lam` (and `SqueezingGate(tau)` is emitted as
  `Arbitrary(0, 0, -ln tau)`): claveles' compiler and the machine squeeze with the opposite sign to its docstring.
- `BeamsplitterGate(theta)` is emitted as `R(-pi/2)` on mode 2, `BeamSplitter(cos eta, 0)`, `R(pi/2)` on mode 2, with
  `theta` folded so that `sqrt_r` stays in [0, 1]; the bare `BeamSplitter(cos theta, 0)` mixed x1 with p2.
- The importer wrote each layer's `connectivity` inverted ({next input: previous output}) while `to_graph`,
  `normalize_diagram` and the exporter read {previous output: next input}: any circuit whose gates permute the open
  modes (3+ modes) came back rewired. Fixed; random 3-5 mode circuits now round-trip exactly with `normalize=False`.
- New `tests/lowering/bridges/test_claveles_conventions.py` pins every intrinsic gate's conventions numerically.

### Known issues

- `normalize_diagram` rewires multi-mode circuits whose two-mode gates act on non-adjacent modes (an `xfail` test
  records it); `from_circuit_repr(..., normalize=False)` is exact.

## [0.1.0] - Unreleased

### Added

- **Core diagram model** (`base_gates.py`): `Diagram` hierarchy with spiders
  (`QSpider`/`PSpider`), `TensorDiagram`, `CompositionDiagram` (with
  non-trivial connectivity), and `ContractedDiagram`, plus a `VoidDiagram`
  placeholder type; unique IDs assigned to every diagram; symbolic
  (sympy-backed) parameters via `ZxPoly`; a `flatten_composition` helper.
- **Gate library** (`gates.py`): a complete set of compact CV gates
  (displacement, phase rotation, squeezing, beamsplitter, controlled-sum,
  controlled-Z, cubic phase, shear, arbitrary, two-mode shear, measurement,
  etc.) with symbolic-parameter and feedforward/measurement-id support.
- **Rewrite rules** (`nx_rewrite_rules.py`), each implemented and tested:
  identity, fusion, bialgebra reduction, chain reduction, terminal
  absorption (including finite-squeezing support), Fourier normalization,
  and the copy rule (later extended to apply across container boundaries).
  A `GateRegistry` tracks gate types/nodes across a graph.
- **Diagram-normalization pipeline** (`normalize_diagram.py`,
  `optimize.py`): a full reduce/optimize pipeline built on top of the graph
  representation and rewrite rules.
- **Graph conversion** (`nx_graph.py`): lossless conversion between
  `Diagram` objects and `networkx` graphs in both directions.
- **Visualization** (`visualize_base_gates.py`, originally
  `visualize_cv_zx.py`): drawing support for spiders, tensor and
  composition diagrams (including nested/resized cases), contracted
  diagrams (including contractions nested inside contractions), with
  wrapped labels and documented, typed drawing functions.
- **Documentation**: a quick-start notebook and a citation for the
  foundational CV-ZX formalism this project implements.
- **Tooling**: `.gitignore`; a `ruff` configuration and pre-commit hook;
  a `mypy` configuration (`pyproject.toml`) and pre-commit hook, with the
  package marked `py.typed`; a GitHub Actions CI workflow running `ruff
  check`, `ruff format --check`, `mypy`, and the `pytest` suite.
- **mqc3 circuit conversion**: `circuit_to_diagram.py` (mqc3 `CircuitRepr`
  -> canonical `Diagram`) and `diagram_to_circuit.py` (the reverse
  direction), each with documented per-gate/per-leaf conversion formulas
  and explicit `NotImplementedError`/`ValueError` on unsupported input
  (the `Manual` gate, feedforward, non-idealized initial states, symbolic
  parameters, `ContractedDiagram`, external inputs) instead of silently
  mistranslating.
- **Pluggable lowering** (`lowering.py`): a `LoweringBackend` plugin
  registry (`register_backend`/`get_backend`/`list_backends`) dispatching
  `Diagram -> MachineryRepr` via `graph_to_machinery_repr`, with a bundled
  `"mqc3"` reference backend built on `diagram_to_circuit.to_circuit_repr`,
  mqc3's own `DependencyDAG` constructor, `GreedyEmbedder`, and
  `MachineryRepr.from_graph_repr`.
- **Logging** (`logging_config.py`): the rewriting pipeline
  (`normalize_diagram`, `nx_rewrite_rules`, `optimize`) now logs through
  the standard `logging` module (rule matches, round/pass counts, a
  warning if `max_rounds` is hit without converging); `setup_file_logging()`
  is an opt-in helper that gives each of those three loggers its own file
  under a log directory.
- **Meaningful type errors**: `optimize()`, `normalize_diagram()`, and
  `RewriteRule.apply_rule()` now raise a `TypeError` with a clear message
  on a malformed argument (a non-`Diagram`, or a rule's `match()`
  returning something other than a `dict`) instead of failing later with
  a harder-to-diagnose `AttributeError`/`KeyError`.
- **Documentation**: a Sphinx site (`docs/`) with a theory page mapping
  the CV ZX calculus paper onto this codebase, task-oriented user guides
  (quickstart, gates, rewrite rules, optimization, mqc3 circuit
  conversion, visualization, debugging with logs), dev-guide pages
  (architecture, the rewrite engine, type-1/type-2 normalization) with
  two diagrams (module dependency graph, full `CircuitRepr` ->
  `MachineryRepr` pipeline), and an auto-generated API reference for
  every module; built and published via GitHub Pages, and built (without
  publishing) on every CI run.
- A `docs` extras group (`pyproject.toml`) for just the documentation
  build dependencies, pulled in by `dev` as well.
- **Feedforward provenance** (`base_gates.py`, `gates.py`): a universal
  `param_measurement_map: dict[Symbol, set[int]]` attribute on `QSpider`,
  `PSpider`, and every `CompactDiagram` gate subclass, linking a symbolic
  parameter to the specific measurement node ID(s) whose outcome it
  depends on. Backed by a shared `Parametrized` mixin providing
  `get_parameters()`, an `is_parametric` property, `substitute_parameters()`/
  `evaluate()`, and `slice_param_map()` (used by `expand()` to hand each
  spawned sub-spider/sub-gate exactly its own share of the map); validated
  with subset semantics (`param_measurement_map.keys() <= get_parameters()`)
  so a symbolic-but-not-feedforward object stays legal. `conjugate()` and
  `expand()` across all 13 gates now thread/slice this map correctly
  instead of silently dropping it.
- **`rustworkx` as a core dependency** (`pyproject.toml`, and the `mypy`
  pre-commit hook's `additional_dependencies`), alongside `networkx`.
- **Dual-backend parameter registry & validation** (`nx_graph.py`,
  `rx_graph.py`): `GateRegister` gained `parametric_nodes`,
  `feedforward_nodes`, `symbol_registry` (symbol -> node IDs), and
  `measurement_to_feedforward_map` (measurement node ID -> dependent node
  IDs), kept in O(1)-lookup sync with the graph via `add_node`/
  `remove_node`/`clear`/`copy` on both backends; and
  `CVZXGraph.validate_parameter_consistency()` (backed by a pure
  `parameter_consistency_violations()`), checking that a symbol shared by
  multiple nodes binds to the same measurement IDs everywhere, and that
  every referenced measurement ID is a real registered measurement node.
  `param_measurement_map` round-trips losslessly through graph
  conversion and is cleared alongside `feedforward`/`measurement_ids`
  wherever a rewrite rule resets a node to identity.
- A shared/mirrored parity test suite for all of the above: AST-level
  tests (`test_base_gates.py`, `test_gates.py`) plus matching nx/rx
  backend tests (`tests/nx_graph`, `tests/rx_graph`,
  `tests/rewrite_rules`, `tests/rx_rewrite_rules`).
- **Backend dispatcher** (`config.py`, `backend.py`): a `Backend` enum
  (`NETWORKX`/`RUSTWORKX`) with an auto-detecting `DEFAULT_BACKEND`, and
  `get_backend_modules()` resolving a `Backend` selector to its
  `*_graph`/`*_rewrite_rules` module pair. `optimize()` now takes an
  optional `backend=` keyword, so callers can pick either graph backend
  explicitly instead of `optimize.py` being hardwired to `networkx`.
- **Domain-specific exception hierarchy** (`exceptions.py`): every error
  the compiler itself raises now derives from `CvzxError`, letting
  callers catch any internal compiler failure with a single
  `except CvzxError:` while still narrowing to a specific failure mode --
  `ParameterError` (`ParameterConflictError`, `UnboundMeasurementError`,
  `InvalidSymbolError`) for symbolic-parameter/feedforward problems,
  `DiagramError` (`ArityMismatchError`, `ExpansionError`) for malformed
  diagram structure, `RewriteError` (`RuleApplicationError`) for a
  rewrite rule hitting a corrupted graph state, and `BackendError`
  (`UnsupportedBackendError`) for an unavailable or invalid graph
  backend. Wired into `CVZXGraph.validate_parameter_consistency()`,
  `Parametrized._sync_feedforward_state()`, `Diagram.compose()`/
  `TensorDiagram` contraction/`ContractedDiagram` construction,
  `ChainReductionRule`/`FusionRule`'s internal "should not occur"
  invariant checks, and `get_backend_modules()`, replacing the generic
  `ValueError`s these previously raised.
- **mqc3 `FeedForward` bridge** (`circuit_to_diagram.py`,
  `diagram_to_circuit.py`): a `CircuitRepr` operation parameter driven by
  `FeedForward[MeasuredVariable]` now round-trips through `Diagram` in
  both directions -- `from_circuit_repr` reconstructs the source
  measurement as a symbolic effect (recovering an affine
  `FeedForwardFunction`'s slope/intercept by evaluating it at `0.0`/`1.0`)
  and threads the resulting `Symbol` onto the downstream gate, while
  `to_circuit_repr` re-emits an equivalent `FeedForward` from a gate's
  `param_measurement_map` provenance. Unsupported cases (a `FeedForward`
  depending on more than one measurement, a non-affine
  `FeedForwardFunction`) raise `NotImplementedError` instead of
  mistranslating.
- **Boundary completion** (`completion.py`): `complete_diagram()` closes
  every open *output* port of a `Diagram` with a fresh symbolic
  measurement effect (`QSpider`/`PSpider` in the chosen basis), returning
  the completed graph, diagram, and a `dict[Symbol, int]` of the fresh
  symbols' measurement-node bindings; `complete_boundaries()` additionally
  closes open *input* ports first (fresh ideal states in the chosen
  basis), so a `Diagram` with any combination of open inputs/outputs can
  be turned into a fully-bounded one in one call. `relink_measurement_symbol()`
  repoints an already-bound symbol's `param_measurement_map` entries at a
  different measurement node (re-validating via
  `validate_parameter_consistency()` afterward), for callers that need to
  correct or reassign provenance post-completion.
- **Direct `CVZXGraph` -> `DependencyDAG` extraction** (`dag_extraction.py`,
  registered in `lowering.py` as the `"cvzx-direct"` `LoweringBackend`):
  an alternative to the `"mqc3"` reference backend's `to_circuit_repr`
  round-trip. `extract_dependency_dag()` discovers execution order with a
  single dual-backend (`networkx`/`rustworkx`) forward sweep directly over
  a `CVZXGraph`'s own `"composition"`/`"contracted_internal"` wire edges,
  anchored at `GateRegister.input_states` and gated on both wire-readiness
  (every input port fed) and classical-readiness (every measurement a
  node's `param_measurement_map` cites already visited), so a feedforward
  edge is never added pointing at an unvisited measurement. Automatically
  closes open ports first via `complete_boundaries()`, then delegates
  actual per-leaf op translation to `diagram_to_circuit`'s existing
  `_apply_1mode_leaf`/`_apply_2mode_leaf` translators rather than
  duplicating that logic, before handing the result to mqc3's own
  `DependencyDAG` constructor. Verified to produce a `DependencyDAG`
  isomorphic to the `"mqc3"` reference backend's output across both graph
  backends.
- **Phase-text overflow handling** (`cvzx.visualization.overflow`): nothing
  previously checked whether a node's wrapped phase text still fit its own
  box — a long/complex symbolic phase could visually spill out of its
  square. `DiagramVisualizer` now measures each phase `Text` artist's
  actual rendered pixel width *and* height against its own box's pixel
  width/height (all four via real matplotlib measurement —
  `Text.get_window_extent()` and an affine `ax.transData.transform()` —
  once the figure's layout is final, not a guessed character-to-pixel
  heuristic) and replaces an overflowing one with a generic `"D<n>"`
  label, relocating the real (still length-capped) phase into a legend
  appended below the diagram instead of losing it. Overflowing the box
  horizontally, not vertically, turns out to be the common case in
  practice: `textwrap.fill`'s wrap width comes from the box's data-unit
  radius, which has no fixed relationship to how many pixels a line
  actually renders to once a busy diagram autoscales every box down. The
  legend itself stays a small, bounded addition regardless of diagram
  size: entries wrap into multiple columns, cap at 20 shown, and any
  remainder collapses into one final "... and N more" line rather than
  growing the figure unboundedly.
- **Optimization-quality metrics** (`cvzx/utils/metrics.py`):
  `compute_metrics`/`compare_metrics` measure how much `optimize()` shrinks
  a diagram -- spider/gate/generator counts (excluding `normalize_diagram`'s
  identity-wire filler and `CopyRule`'s `VoidDiagram` placeholders, both of
  which otherwise inflate "after" counts enough to make a genuinely
  simplified diagram look larger), a non-Clifford ("T-count"-analogue)
  phase count via `ZxPoly` degree, and stage depth via `normalize_diagram`.
  Backed by a `benchmarks/` suite (`python -m benchmarks.run_benchmarks`,
  dev-invoked only, not wired into CI) and a new dev-guide page
  (`docs/source/dev_guide/benchmarks.md`) walking through the metric
  definitions and a worked before/after example.
- **`ChainReductionRule` now runs combined phases/parameters through
  `sympy.simplify`** (`cvzx.utils.helpers.simplify_reduced_value`, used by
  both `nx.rules`/`rx.rules`): the chain's values were previously combined
  with plain `+`/`*` and left exactly as produced, so an algebraic identity
  spanning the chain (e.g. `sin(θ)² + cos(θ)²` folding to `1`, or a
  trig-identity combination that's exactly the identity's `0` without being
  *structurally* `0`) was never recognized — the chain stayed unreduced, or
  worse, wasn't collapsed to the identity spider it actually was. Applies to
  `QSpider`/`PSpider` phases (each `ZxPoly` coefficient) and every
  arithmetic-combination gate parameter (`PhaseRotationGate`,
  `BeamsplitterGate`, `SqueezingGate`, `DisplacementGate`,
  `ControlledZGate`, `ControlledSumGate`'s gain) — not the `Fourier`
  family's count-based table lookups, which have nothing to simplify.
  Deliberately does *not* coerce a simplified value back to a plain Python
  number just because it has no free symbols left: `Expr.is_number` is true
  for any exact irrational constant too (a `pi` multiple included), so doing
  that would have silently turned an exact phase like `π/6 + π/5 + π/7` into
  a lossy decimal approximation. Symmetrically, a `ZxPoly` is only
  round-tripped through this simplification at all when
  `is_parametric()` — for an already fully-numeric phase there is nothing
  to simplify, and the round-trip risks changing the underlying
  `sympy.Poly`'s domain (exact integers becoming floats) for zero benefit.
- **`ControlledSumGate` now converts to mqc3** (`cvzx.lowering.bridges.mqc3`,
  `_apply_2mode_leaf`): mqc3's intrinsic set has no CSUM/CNOT-style
  primitive, only `ControlledZ`, so a `Diagram` containing a CSUM leaf
  previously raised `NotImplementedError` in `to_circuit_repr` (cvzx ->
  mqc3) unconditionally. Since `ControlledSumGate(g)` is
  `exp(-i g q̂_c p̂_t)` and `ControlledZGate(g)` is `exp(-i g q̂₁ q̂₂)`,
  conjugating `ControlledZ` by a Fourier rotation on CSUM's *target* mode
  converts CZ's q-q coupling into CSUM's q-p coupling:
  `CSUM_{c→t}(g) = (I ⊗ F_t) CZ(g) (I ⊗ F_t†)`, emitted as
  `intrinsic.PhaseRotation(-π/2)` (target mode) ->
  `intrinsic.ControlledZ(-g)` (both modes) -> `intrinsic.PhaseRotation(π/2)`
  (target mode). Verified via the Heisenberg-picture (classical
  Hamiltonian-flow) transform of `(q_c, p_c, q_t, p_t)` under each side
  independently -- both give `q_c' = q_c`, `p_c' = p_c - g p_t`,
  `q_t' = q_t + g q_c`, `p_t' = p_t` -- rather than by inspection, since
  this codebase has no numeric Gaussian simulation backend to check
  against directly. See `docs/source/user_guide/circuit_conversion.md`
  for the full derivation. `CubicPhaseGate` (genuinely non-Gaussian)
  remains the only unsupported compact gate.

- **`VoidPortPruningRule`** (both backends): drops a bare `QSpider`/
  `PSpider`'s last port when it's wired, by a single "composition" edge
  (or a chase through zero-phase identities), directly to a
  `VoidDiagram` — a real gate stranded with one port feeding, or fed by,
  nothing, typically left behind by a cross-container splice (e.g.
  `TerminalAbsorptionRule._apply_contracted_child`). Restricted to bare
  spiders and to a port already last on its side, so neither the port
  nor its remaining siblings ever need renumbering, and refuses to prune
  a node already down to one port if its phase is non-zero (that would
  leave a `(0, 0)` node with a phase nothing binds to). Wired into
  `optimize()`'s rule list for both backends.
- **`ChainReductionRule`'s commute-and-fuse match kind** (both backends):
  a `SqueezingGate`/linear-phase ("Disp") spider directly adjacent to
  another spider — with a matching mover or same-color target on its
  far side — commutes across it (`Sq(tau)` scales the crossed phase's
  variable by `tau`/`1/tau`; a Disp shifts it by its own linear
  coefficient) and fuses with whatever it lands next to. Covers all four
  shapes: `Sq/Spider/Sq`, `Disp/Spider/Disp` (opposite color), and
  `Q/[Sq|Disp]/Q`, `P/[Sq|Disp]/P`. Exact, unconditional identities (no
  `assume_infinite_squeezing` gate) — unlike absorbing a genuine
  terminal state/effect, this moves a `(1, 1)` gate across a spider,
  which is exact regardless of squeezing.
- **`TerminalAbsorptionRule`'s `bare_cap_fusion` match kind** (both
  backends): collapses a `ContractedDiagram` whose one half is *already*
  a genuine `(0, 1)`/`(1, 0)` terminal directly occupying `first_id`/
  `second_id`, with its single port entirely consumed by the contraction
  itself (nothing external connects to it) — a shape neither ordinary
  absorption nor `PassthroughRule` covers. Same-color folds via addition
  (exact); opposite-color via the shift formula (gated on
  `assume_infinite_squeezing`, an idealized-eigenstate assumption).
  Refuses to fold when the survivor's own remaining role would also
  collapse to `(0, 0)` — that's a genuine closed scalar (a state composed
  directly into its own opposite effect), which this codebase has no
  scalar bookkeeping to represent correctly.
- **`normalize_diagram` now handles a `ContractedDiagram`** instead of
  unconditionally leaving the whole input untouched the moment one
  appears anywhere in the tree. A top-level `ContractedDiagram` (one not
  itself nested inside another's `first`/`second`) is treated as a
  single opaque leaf: its own kept (external) `num_inputs`/`num_outputs`
  — already tracked on the graph node — are what the leaf-level
  dependency graph and row bookkeeping see, while `first`/`second`
  (and whatever they contain) are reconstructed as one atomic unit via
  `reconstruct_contracted_node`, never decomposed into separate rows —
  a contraction's two halves generally can't be represented as
  independent leaves, since plain `Tensor`/`Compose` nesting has no way
  to express "these two, though far apart, still share an internal
  wire." A raw composition edge landing directly on `first`/`second`
  (never on the `ContractedDiagram` node itself) is redirected to the
  contraction's own external port via a new reverse-mapping step
  (`_build_contracted_port_remaps`) before `_build_pred_map` reads it.
  Falls back to the old unconditional "leave it unchanged" behavior only
  if a top-level contraction's own kept port resolves down through
  *another*, nested `ContractedDiagram` instead of a genuine leaf — a
  shape not yet handled. Unlocks real reductions previously blocked by
  `optimize()`'s outer loop silently no-op'ing its own per-round
  `normalize_diagram()` call the moment any contraction survived into a
  later round (common under `assume_infinite_squeezing=True`).

### Changed

- Migrated the rewrite-rule engine from a class-based, `Diagram`-walking
  implementation to one operating directly on `networkx` graphs
  (identity, fusion, and chain-reduction rules ported first), and later
  removed the legacy class-based rules entirely once the graph-based ones
  covered the same ground.
- Moved the CV-ZX compiler out of the `mqc3` monorepo into this
  independent `cvzx` repository.
- Brought the codebase into full compliance with a strict `ruff` (all
  rules) and `mypy` (`check_untyped_defs`, no implicit `Any`) configuration
  across `src/` and `tests/`, fixing several latent bugs surfaced along the
  way (e.g. an unreachable-but-broken `isinstance` call, a mismatched
  helper signature, an always-true truthy-function check).
- Switched `ruff`'s `pydocstyle` convention from `google` to `numpy` to
  match the docstring style actually used throughout the codebase (the
  previous mismatch silently disabled several `pydoclint` checks —
  correcting it surfaced and fixed a handful of stale/incorrect
  documented parameters and return types); reformatted every affected
  docstring (numpy-style section headers, exception names) accordingly.
  Also trimmed the longest module/function docstrings
  (`normalize_diagram`, `optimize()`, `lowering`, `diagram_to_circuit`)
  down to a summary plus a pointer, moving the narrative/design-rationale
  content they duplicated into the corresponding docs page instead.
- `ci.yaml` now installs the package via the `dev` extra (pulling in
  `mqc3`, needed by `circuit_to_diagram`/`diagram_to_circuit`/`lowering`
  and their tests) instead of a bare install, and builds the
  documentation with warnings treated as errors.
- **Reorganized `src/cvzx/` into subpackages by role**, and mirrored the same
  layout under `tests/`: `ir/` (`base.py`, `gates.py` — was `base_gates.py`,
  `gates.py`), `backends/{nx,rx}/` (`graph.py`, `rules.py` — was
  `{nx,rx}_graph.py`, `{nx,rx}_rewrite_rules.py`), `passes/` (`optimize.py`,
  `completion.py`, `normalize.py` — was `normalize_diagram.py`),
  `lowering/` (`dag.py` — was `dag_extraction.py`; `lowering.py`, unchanged
  in content; `bridges/mqc3.py` — merges the former `circuit_to_diagram.py`
  and `diagram_to_circuit.py`, the two directions of the same mqc3 bridge,
  into one module), `utils/` (`helpers.py` — was `utils.py`), and a
  dedicated `visualization/` package (`core.py` — `VisualizerConfig`,
  `DiagramVisualizer`, and the module-level `visualize()`; `proper.py`,
  `composition.py`, `contracted.py`, `swap_fourier.py`, `overflow.py` — one
  private mixin each, split by diagram-type/concern out of the former
  single ~2100-line `visualize_base_gates.py`; `geometry.py` for the
  diagram-agnostic helpers all of them share; `protocol.py` for the
  structural `Visualizer` type every mixin method types `self` as, since
  mypy requires an explicit `self` type to be a *nominal* supertype of a
  bare mixin's own class, which a `Protocol` sidesteps via structural
  typing instead; `debug.py` for the `visualize_before_after` test helper).
  `config.py`/`exceptions.py`/`logging_config.py`/`backend.py` stay at the
  package root. A clean rename with no compatibility shims (nothing has
  been tagged/released yet, so the old flat `cvzx.<module>` import paths
  are gone rather than kept working); every import across `src/`, `tests/`,
  and the documentation was updated to match, and `docs/source/_static/
  generate_diagrams.py`'s module-map and pipeline diagrams were rebuilt
  for the new layout.

### Fixed

- Index handling when composing a contracted diagram.
- Vertical-shift handling when drawing nested contraction diagrams.
- `_reorder_positions` in the visualization module.
- `normalize_diagram`'s final-stage reordering (`_absorb_final_permutation`)
  crashing with an `IndexError` whenever the last stage's final element had
  no outputs (e.g. a measurement) — it now falls back to the existing
  trailing-permutation-stage path instead, the same as its other
  can't-reorder-in-place case.
- Broken `cvzx.circ_to_diag`/`cvzx.diag_to_circ` imports in `lowering.py`
  and the new modules' tests (the modules were renamed to
  `circuit_to_diagram`/`diagram_to_circuit` without updating every
  reference), which made those test files fail to collect at all.
- Docstring citation markers in `base_gates.py`/`gates.py`/
  `nx_rewrite_rules.py` incorrectly numbered `[1]` throughout, corrected
  to `[1]` to match the single paper reference these docstrings actually
  cite.
- A handful of latent bugs surfaced while building the feedforward-provenance
  work above: `QSpider`/`PSpider` reconstruction (`nx_graph.py`,
  `rx_graph.py`) returning `feedforward=None` instead of `False` when a
  node never had the attribute explicitly set; `ZxPoly.coeffs` crashing
  with `TypeError` on a genuinely complex coefficient; `QSpider`/
  `PSpider.conjugate()` dropping the `parametric` flag; the `rustworkx`
  backend's node reconstruction not forwarding `parametric`/`feedforward`/
  `measurement_ids` at all; three gates' `expand()`
  (`CubicPhaseGate`/`ShearXInvariantGate`/`ShearPInvariantGate`) not
  threading `is_parametric` into their spawned sub-spider; several other
  gates' `expand()` blindly copying the parent's `parametric` flag onto
  sub-objects instead of each sub-value's own state;
  `DisplacementGate.substitute_parameters` crashing on a complex result;
  `ControlledSumGate.substitute_parameters` silently dropping `control`/
  `target`; and `nx_graph.py`'s `GateRegister.clear()` omitting
  `void_nodes.clear()`.
- A dropped `node_map` parameter on `rx_graph.py`'s `_add_proper_node` and
  an undefined `_has_node` reference in `rx_rewrite_rules.py`, both
  `NameError`/`TypeError` crashes that silently broke the majority of the
  `rustworkx`-backend test suite.
- The `mypy` pre-commit hook's isolated environment missing `rustworkx`
  from its `additional_dependencies`, which made it unable to resolve
  `rustworkx`'s types (surfacing as `import-not-found` and a downstream
  `no-any-return` in `rx_graph.py`/`rx_rewrite_rules.py`) even though
  `pyproject.toml` already listed it as a real dependency.
- `optimize()`'s inner simplification loop
  (`_simplify_to_fixed_point`) had no round cap, unlike every other loop
  in the pipeline -- a rule interaction that never reaches a true fixed
  point could hang effectively forever instead of returning a
  not-fully-simplified result. Root cause of one such hang:
  `FusionRule._apply_contracted`'s `special_case` branch swaps
  `first_id`/`second_id` but kept reading the stale `first_attrs`
  captured before the swap, always seeing `phase=None` and silently
  no-op'ing the same match forever (fixed in both backends).
- `ChainReductionRule.apply_single` reset every non-surviving chain
  member to a same-arity zero-phase spider, which is only a genuine,
  prunable identity at arity `(1, 1)`; for wider gates
  (`ControlledSumGate`, `BeamsplitterGate`, ...) the leftover node was
  never pruned and stayed stranded in the diagram. Now falls back to the
  general splice-and-flatten path for any non-`(1, 1)` chain, in both
  backends.
- `tests/test_optimize.py::test_squeezing_absorption_folds_tau_into_terminal_phase`
  asserted the wrong direction for `TerminalAbsorptionRule`'s squeezing
  absorption (dividing by `tau**degree` instead of multiplying),
  contradicting the codebase's own established, independently tested
  convention for a `QSpider` state.
- Nine `sphinx -W` doc-build failures: missing blank lines before RST
  bullet lists in five rewrite-rule `match()` docstrings, and a
  `circuit_to_diagram`/`diagram_to_circuit` module-docstring heading
  collision (`"Feedforward"` used by both, now `"Feedforward on
  ingestion"`/`"Feedforward on emission"`) once both were documented on
  the same API-reference page. A repo-wide sweep also fixed the same
  underlying RST bug pattern (a single-backtick reference immediately
  followed by a plural "s", e.g. `` `Swap`s ``, which breaks docutils'
  inline-markup end-string rule) everywhere else it occurred, before it
  could cause the same failures elsewhere.
- `api_reference.md` was missing half the package
  (`backend`/`config`/`exceptions`/`completion`/`dag_extraction`/
  `utils`/both `rx_*` modules) from its `automodule` listing; extending it
  surfaced the same missing-blank-line docstring bug in
  `rx_rewrite_rules.py`'s `match()` methods (mirroring the `nx` fix above)
  and a genuine bug in `exceptions.py`'s own docstring (an unmarked ASCII
  tree diagram parsed as a malformed paragraph instead of a literal
  block), plus an inherent, permanent ambiguity from documenting both
  `CVZXGraph` classes (`nx`/`rx`) on one page — resolved by suppressing
  Sphinx's `ref.python` warning class in `conf.py` rather than qualifying
  every bare `` `CVZXGraph` `` mention project-wide.
- `docs/source/user_guide/circuit_conversion.md` and the architecture/
  pipeline docs described a `graph_to_machinery_repr`/`to_machinery_repr`/
  `MachineryRepr.from_graph_repr` API that was never actually implemented
  in `lowering.py` (whose real, current surface ends at
  `graph_to_dependency_dag`/`DependencyDAG`) — rewritten to match the real
  API, including a second worked example for the `"cvzx-direct"` backend.
- `docs/source/user_guide/rewrite_rules.md`'s and `quickstart.md`'s worked
  examples called the pre-`CVZXGraph` `RewriteRule.match(graph, registry)`/
  `GateRegister().build_from_graph(graph)` API against a `to_graph()` call
  that has returned a `CVZXGraph` (a single object, registry included) for
  some time -- both raised `AttributeError`/`TypeError` if actually run;
  fixed to use the current `match(cvzx_graph)` signature and
  `graph.registry` directly.
- `Diagram` ids could collide across a `to_diagram()`/`to_graph()`
  round-trip: `reconstruct_proper_node` (`backends/nx/graph.py`,
  `backends/rx/graph.py`) preserves a node's original graph id for anything
  in `registry.measurement_nodes` -- which, despite the name, is any
  `(1, 0)`-arity node, not just literal measurements -- by force-setting
  `.id` directly, without ever advancing the shared `Diagram` id counter
  past that value. The counter could later independently reach the same
  integer and hand it to an unrelated object, so two different `Diagram`s
  ended up with the same `.id`; since `to_graph()` keys graph nodes by
  `diagram.id`, the two silently merged into one malformed graph node.
  Fixed by replacing the bare counter with a `Diagram._reserve_id()`
  classmethod that every force-set call site (including a new upfront
  reservation in both backends' `to_diagram()`, covering every node the
  graph could hand out regardless of visit order) now calls.
- `optimize()` could converge to a different, less-reduced fixed point than
  applying the same rewrite rules by hand: `_simplify_to_fixed_point`
  mutated the graph continuously across a whole pass, while the
  hand-driven workflow (and the example notebooks) round-trips through
  `to_diagram()`/`to_graph()` between every individual rule application --
  some rules (e.g. `FusionRule`'s handling of a `CopyRule`-produced
  fan-out) leave bookkeeping that a later rule's `match()` can misread
  without that round-trip. Concretely, the measurement-induced-squeezer
  example (`examples/measurement_induced_squeezer.ipynb`)
  reduced to `SqueezingGate(sin(theta)**2/cos(theta))` via `optimize()` but
  correctly to `SqueezingGate(sin(theta))` (matching the paper) when the
  same rules were applied one at a time. `_simplify_to_fixed_point` now
  round-trips after each rule application, matching the manual workflow;
  this alone would have surfaced the `Diagram`-id-collision bug above as
  outright crashes, which is how that bug was actually found and fixed.
- `ChainReductionRule`'s commute-and-fuse `_try_target_mover_target`
  (both backends) hardcoded `(1, 1)` as the fused survivor's `gate_info`
  when checking whether a same-color flank could fuse across a crossed
  `Sq`/`Disp`, without confirming the flank actually *was* `(1, 1)` —
  only that it wasn't a terminal. When a flank was really a wider spider
  (e.g. a `ContractedDiagram` child with further ports of its own,
  reached because one of its ports happened to chase-connect to the
  crossed gate), `_update_node_for_reduced_gate` force-overwrote its true
  arity down to `(1, 1)`, leaving any `I1`/`I2`/`J1`/`J2` list that still
  referenced the now-missing port pointing at a nonexistent index —
  surfacing as an `ArityMismatchError` on the next `to_diagram()` call.
  Now requires both flanks to already be bare `(1, 1)` spiders, matching
  the restriction `_get_squeeze_param`/`_get_disp_param` already enforce
  on the mover side of the other commute shape. Found via `optimize()`
  crashing on `examples/example_4_cubic_phase_injection_nonunit_gain.ipynb`'s
  diagram under the default `rustworkx` backend.
- `VoidPortPruningRule` (both backends) matched *any* "proper" node type,
  not just bare spiders, and its port-shrink is a blind decrement --
  correct for a `QSpider`/`PSpider` (whose shape is just "however many
  legs its phase function is applied to"), but wrong for a gate with
  fixed, paired port semantics, such as `Swap` (whose two inputs/outputs
  are cross-wired to each other, not independently meaningful). Pruning
  a `Swap`'s port produced a malformed node (a `Swap` that wasn't
  `(2, 2)`), corrupting a containing `CompositionDiagram`'s connectivity
  and crashing on the next `to_diagram()` call. Restricted to
  `QSpider`/`PSpider` in both backends. Found via `optimize()` crashing
  on `examples/measurement_induced_squeezer.ipynb`'s diagram -- with
  both fixes in place, that example now reduces to exactly
  `SqueezingGate(sin(theta))` (matching the paper) on both backends,
  where it previously reduced incorrectly or crashed depending on
  backend and rule ordering.
- `ChainReductionRule`'s entire commute-and-fuse match kind existed only
  in the `nx` backend -- never ported to `rx` -- despite being wired into
  `optimize()`'s shared rule list for both. Since `rx` is
  `DEFAULT_BACKEND` whenever `rustworkx` is installed, `optimize()`'s
  default path was silently missing this capability entirely. Ported.
- `BeamsplitterGate.expand()`'s general (non-balanced-angle) branch
  reused the *same* `SqueezingGate` object instance (`sq1`) at two
  different positions in the returned `CompositionDiagram`, instead of
  building two separate instances with the same `tau`. `to_graph()` keys
  its nodes by `Diagram.id`, so the two physically distinct wires this
  decomposition actually has collapsed onto one graph node carrying
  edges from both positions. Harmless for anything that only walks
  container structure (`to_diagram()` never noticed), but a genuine
  wire-level corruption for anything reading predecessor/successor
  edges -- surfaced as `normalize_diagram: leaf-level wire graph is not
  a DAG (cycle detected)` once `normalize_diagram` gained the ability to
  see through a `ContractedDiagram` (see `Added`, above) and tried to
  build a real dependency graph through this shape for the first time.
  Fixed by constructing a second, independent `SqueezingGate` instance.
- `FusionRule._apply_contracted`'s `special_case` branch had its
  `down`/no-`down` fix (correctly swapping which side a fused spider
  gets spliced into, and whether it's prepended or appended to the
  survivor's `sub_diagram_ids`, depending on which of `first`/`second`
  was actually the `TensorDiagram`) applied only to the `nx` backend --
  `rx` still unconditionally prepended, exactly reproducing the bug the
  `nx` fix addressed. Same root cause for the *other* half of the
  `TerminalAbsorptionRule._apply_contracted_child` identity-chain fix
  from earlier in this changelog (reset-to-identity leaving a dead
  passthrough at its old, nonzero arity instead of voiding it and
  propagating the shrink): that fix, too, had only ever been applied to
  `nx`. Together these two `rx`-only gaps were the actual cause of
  `optimize()` reaching a visibly less-reduced fixed point under `rx`
  than under `nx` on realistic multi-contraction circuits (e.g.
  `examples/example_4_cubic_phase_injection_nonunit_gain.ipynb`) --
  traced by diffing a step-by-step rule trace between backends on
  identical starting states until the first rule (`FusionRule`) that
  produced different output from structurally identical matches. Both
  ported to `rx`; all three example notebooks' diagrams now reduce to
  byte-identical results on both backends, in both a single manual pass
  and through `optimize()`.

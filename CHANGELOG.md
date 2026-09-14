# Changelog

All notable changes to this project are documented in this file.

This project has not yet made a tagged release; the entries below summarize
development on `main` to date, grouped by area rather than by commit.

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

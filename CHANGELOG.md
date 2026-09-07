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
  `nx_rewrite_rules.py` incorrectly numbered `[3]` throughout, corrected
  to `[1]` to match the single paper reference these docstrings actually
  cite.

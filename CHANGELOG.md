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

### Fixed

- Index handling when composing a contracted diagram.
- Vertical-shift handling when drawing nested contraction diagrams.
- `_reorder_positions` in the visualization module.

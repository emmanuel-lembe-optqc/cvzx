# CVZX Compiler

A graph-based compiler for continuous-variable quantum circuits and CV-ZX diagrams, with a
full rewrite/optimization pipeline and round-trip conversion to/from mqc3 circuits.

> **Status:** Research codebase. The graph representation, the full set of main CV-ZX rewrite
> rules, the optimization pipeline, and conversion to/from mqc3's `CircuitRepr`/`MachineryRepr`
> are all implemented. Completeness beyond the currently-implemented rules/gates (finite
> squeezing, non-Gaussian resources beyond the cubic phase gate, GKP/mixed states) is out of
> scope for now — see the [documentation](#documentation)'s theory page for the exact list.

## Overview

This project compiles continuous-variable (CV) quantum circuits using a CV-ZX-inspired
diagrammatic representation, built around the CV ZX calculus of Nagayoshi et al. (see
[Acknowledgements](#acknowledgements)).

The current implementation covers:

- A structured `Diagram` model (spiders, gates, tensor/composition/contraction containers)
  and a lossless conversion to/from a `networkx` graph representation for rewriting.
- The full set of CV-ZX rewrite rules (identity, fusion, chain reduction, Fourier
  normalization, terminal absorption, and the copy rule), applied to a fixed point by
  `cvzx.optimize.optimize`.
- Round-trip conversion between an mqc3 `CircuitRepr` and a canonical `cvzx.Diagram`
  (`cvzx.circuit_to_diagram`, `cvzx.diagram_to_circuit`).
- A pluggable lowering step from a `Diagram` to a concrete mqc3 `MachineryRepr`
  (`cvzx.lowering`), so a different QPU backend can be added without touching the rest of
  the pipeline.
- Visualization of CV-ZX diagrams (`cvzx.visualize_base_gates`).
- Opt-in structured file logging for the rewriting pipeline (`cvzx.logging_config`).

## Documentation

Full documentation — a theory page mapping the CV ZX calculus paper onto this codebase,
task-oriented user guides, architecture/design-rationale pages for contributors, and the
auto-generated API reference — is built with Sphinx from [`docs/`](docs/):

```bash
python -m pip install -e ".[docs]"
cd docs
make html
```

Open `docs/build/html/index.html` in a browser. The same build runs in CI on every push and
pull request.

## Project layout

```text
cvzx/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── LICENSE
├── docs/                          # Sphinx documentation (theory, guides, API reference)
├── src/
│   └── cvzx/
│       ├── __init__.py
│       ├── base_gates.py          # Diagram hierarchy, ZxPoly, container types
│       ├── gates.py               # CompactDiagram + every gate class
│       ├── nx_graph.py            # Diagram <-> networkx.DiGraph conversion, GateRegister
│       ├── normalize_diagram.py   # Type-1/type-2 stage canonicalization
│       ├── nx_rewrite_rules.py    # The CV-ZX rewrite rules
│       ├── optimize.py            # optimize(): runs the rules to a fixed point
│       ├── circuit_to_diagram.py  # mqc3 CircuitRepr -> cvzx Diagram
│       ├── diagram_to_circuit.py  # cvzx Diagram -> mqc3 CircuitRepr
│       ├── lowering.py            # Diagram -> mqc3 MachineryRepr (pluggable backends)
│       ├── logging_config.py      # opt-in file logging for the rewriting pipeline
│       └── visualize_base_gates.py
└── tests/
    ├── nx_graph/
    ├── rewrite_rules/
    └── test_*.py
```

Generated directories such as `build/`, `dist/`, `*.egg-info/`, and `__pycache__/` should not
be committed.

## Installation

Create and activate a Python environment, then install the package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For development (tests, mqc3 integration, docs, notebooks):

```bash
python -m pip install -e ".[dev]"
```

Just the documentation build:

```bash
python -m pip install -e ".[docs]"
```

If you use the optional Strawberry Fields integration:

```bash
python -m pip install -e ".[sf]"
```

The package currently targets Python 3.10 and newer.

## Quick start

A guided introduction is available in [`quick_start.ipynb`](quick_start.ipynb), and the
Sphinx docs' user guide has task-oriented pages for building/rewriting/optimizing diagrams,
converting to and from mqc3 circuits, and visualization — see [Documentation](#documentation).

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open `quick_start.ipynb` and run the cells from top to bottom.

## Running tests

Run the test suite with:

```bash
pytest
```

For coverage:

```bash
pytest --cov=cvzx
```

## Minimal example

```python
from sympy import pi

from cvzx.base_gates import CompositionDiagram
from cvzx.gates import PhaseRotationGate
from cvzx.optimize import optimize

comp = CompositionDiagram([
    PhaseRotationGate(pi / 6),
    PhaseRotationGate(pi / 5),
    PhaseRotationGate(pi / 7),
])

graph, diagram = optimize(comp)  # the three rotations fuse into one
```

See the user guide's "Converting to and from mqc3 circuits" page for a walkthrough that covers
converting to/from an mqc3 `CircuitRepr` and lowering to a `MachineryRepr`.

## Debugging

The rewriting pipeline (`normalize_diagram`, `nx_rewrite_rules`, `optimize`) logs through the
standard `logging` module. Call `cvzx.logging_config.setup_file_logging()` once, early in your
script, to get one log file per module under `./logs/` instead of nothing — see the user
guide's "Debugging with logs" section.

## Research roadmap

Planned extensions, not yet implemented:

1. Finite-squeezing parameters and covariance/noise propagation (the current pipeline treats
   terminal states as idealized, infinitely-squeezed eigenstates).
2. Non-Gaussian resource handling beyond the cubic-phase gate (e.g. Fock-space interfaces).
3. GKP diagrams and mixed states.
4. Additional paper rules not yet implemented (quadratic, inversion, antipode, rotation
   self-loop) — deferred because the currently-implemented set was sufficient for the
   circuits this project targets so far.

## Scientific scope

The project is motivated by continuous-variable quantum computation, CV measurement-based
quantum computation, CV-ZX diagrammatic reasoning, Gaussian and non-Gaussian gate compilation,
and graph-based optimization.

## Acknowledgements

This project is a that builds on the continuous-variable ZX formalism
introduced in:

> Hironari Nagayoshi, Warit Asavanant, Ryuhoh Ide, Kosuke Fukui,
> Atsushi Sakaguchi, Jun-ichi Yoshikawa, Nicolas C. Menicucci, and
> Akira Furusawa, “ZX graphical calculus for continuous-variable quantum
> processes,” *Physical Review Research* **7**, 033141 (2025).
> [arXiv:2405.07246](https://arxiv.org/abs/2405.07246)

The gate set and naming conventions follow MQC3's `graph`/`circuit` operations, and
`cvzx.circuit_to_diagram`/`cvzx.diagram_to_circuit`/`cvzx.lowering` convert to/from and lower
onto that machinery directly.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

## Contact and contributions

Issues, discussions, and pull requests are
welcome as the project develops.

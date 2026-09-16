# CVZX Compiler

A graph-based compiler for continuous-variable quantum circuits and CV-ZX diagrams, with a
full rewrite/optimization pipeline, round-trip conversion to/from mqc3 circuits, and a dual
`networkx`/`rustworkx` graph backend.

## Overview

This project compiles continuous-variable (CV) quantum circuits using a CV-ZX-inspired
diagrammatic representation, built around the CV ZX calculus of Nagayoshi et al. (see
[Acknowledgements](#acknowledgements)).

The current implementation covers:

- A structured `Diagram` model (spiders, gates, tensor/composition/contraction containers,
  `cvzx.ir.base`/`cvzx.ir.gates`) and a lossless conversion to/from a graph representation
  for rewriting, on either a `networkx` or `rustworkx` backend
  (`cvzx.backends.nx`/`cvzx.backends.rx`, dispatched via `cvzx.config.Backend`) — both produce
  identical results, `rustworkx` is the faster default when installed.
- The full set of CV-ZX rewrite rules (identity, fusion, chain reduction, Fourier
  normalization, terminal absorption, and the copy rule), applied to a fixed point by
  `cvzx.passes.optimize.optimize`.
- Round-trip conversion between an mqc3 `CircuitRepr` and a canonical `cvzx.ir.base.Diagram`
  (`cvzx.lowering.bridges.mqc3`), and boundary completion for open diagrams
  (`cvzx.passes.completion`).
- A pluggable lowering step from a `Diagram` to a concrete mqc3 `DependencyDAG`
  (`cvzx.lowering.lowering`), with a bundled `"mqc3"` reference backend and a `"cvzx-direct"`
  backend that skips the `CircuitRepr` round trip, so a different QPU backend can be added
  without touching the rest of the pipeline. From there, mqc3's own `GraphEmbedder`/
  `GraphRepr`/`MachineryRepr` chain is outside this project's scope.
- Visualization of CV-ZX diagrams (`cvzx.visualization`), and optimization-quality metrics
  measuring how much `optimize()` shrinks a diagram (`cvzx.utils.metrics`).
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
├── examples/                      # Runnable Jupyter notebooks (see "Quick start")
├── src/
│   └── cvzx/
│       ├── __init__.py
│       ├── backend.py             # Backend dispatcher (get_backend_modules)
│       ├── config.py              # Backend enum, DEFAULT_BACKEND
│       ├── exceptions.py          # CvzxError hierarchy
│       ├── logging_config.py      # opt-in file logging for the rewriting pipeline
│       ├── ir/
│       │   ├── base.py            # Diagram hierarchy, ZxPoly, container types
│       │   └── gates.py           # CompactDiagram + every gate class
│       ├── backends/
│       │   ├── nx/                # networkx: graph.py (Diagram <-> DiGraph, GateRegister),
│       │   │                      # rules.py (the CV-ZX rewrite rules)
│       │   └── rx/                # rustworkx mirror of the above
│       ├── passes/
│       │   ├── normalize.py       # Type-1/type-2 stage canonicalization
│       │   ├── optimize.py        # optimize(): runs the rules to a fixed point
│       │   └── completion.py      # Closing a Diagram's open input/output ports
│       ├── lowering/
│       │   ├── bridges/mqc3.py    # mqc3 CircuitRepr <-> cvzx Diagram
│       │   ├── dag.py             # Direct CVZXGraph -> DependencyDAG extraction
│       │   └── lowering.py        # Diagram -> mqc3 DependencyDAG (pluggable backends)
│       ├── utils/
│       │   ├── helpers.py         # Rewrite-engine-adjacent utilities
│       │   └── metrics.py         # Optimization-quality metrics (compute_metrics, ...)
│       └── visualization/         # Drawing CV-ZX diagrams
└── tests/                         # Mirrors the src/cvzx/ layout above
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

A guided introduction is available in
[`examples/quickstart.ipynb`](examples/quickstart.ipynb) — building a diagram, converting it
to the graph representation, and simplifying it by hand and via `optimize()`. See
[`examples/example_1_measurement_induced_squeezer.ipynb`](examples/example_1_measurement_induced_squeezer.ipynb)
for a deeper, physically-motivated worked example. The Sphinx docs' user guide also has
task-oriented pages for building/rewriting/optimizing diagrams, converting to and from mqc3
circuits, and visualization — see [Documentation](#documentation).

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open a notebook under `examples/` and run the cells from top to bottom.

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
from cvzx.ir.base import CompositionDiagram
from cvzx.ir.gates import PhaseRotationGate
from cvzx.passes.optimize import optimize

comp = CompositionDiagram([
    PhaseRotationGate(0.3),
    PhaseRotationGate(0.4),
    PhaseRotationGate(0.5),
])

result = optimize(comp)  # the three rotations fuse into one; result.diagram, result.graph
```

See the user guide's "Converting to and from mqc3 circuits" page for a walkthrough that covers
converting to/from an mqc3 `CircuitRepr` and lowering to a `DependencyDAG`.

## Debugging

The rewriting pipeline (`cvzx.passes.normalize`, `cvzx.backends.nx.rules`/
`cvzx.backends.rx.rules`, `cvzx.passes.optimize`) logs through the standard `logging` module.
Call `cvzx.logging_config.setup_file_logging()` once, early in your script, to get one log
file per module under `./logs/` instead of nothing — see the user guide's "Debugging with
logs" section.

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

This project builds on the continuous-variable ZX formalism introduced in:

> Hironari Nagayoshi, Warit Asavanant, Ryuhoh Ide, Kosuke Fukui,
> Atsushi Sakaguchi, Jun-ichi Yoshikawa, Nicolas C. Menicucci, and
> Akira Furusawa, “ZX graphical calculus for continuous-variable quantum
> processes,” *Physical Review Research* **7**, 033141 (2025).
> [arXiv:2405.07246](https://arxiv.org/abs/2405.07246)

The gate set and naming conventions follow MQC3's `graph`/`circuit` operations, and
`cvzx.lowering.bridges.mqc3`/`cvzx.lowering.lowering` convert to/from and lower onto that
machinery directly.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

## Contact and contributions

Issues, discussions, and pull requests are
welcome as the project develops.

# CVZX Compiler

A graph-based compiler prototype for continuous-variable quantum circuits and CV-ZX diagrams.

> **Status:** Early research prototype. The graph representation and initial rewrite rules are implemented; the complete optimization pipeline is still under active development.

## Overview

This project explores compilation and optimization of continuous-variable (CV) quantum computations using a CV-ZX-inspired diagrammatic representation.

The current implementation focuses on:

- Graph representations of CV-ZX diagrams using NetworkX.
- Conversion between structured diagram objects and graph representations.
- Identity-spider removal.
- Chain reduction of compatible graph structures.
- Fusion of compatible spiders and gates.
- Explicit port/connectivity metadata on graph edges.
- Visualization of CV-ZX diagrams.
- A planned optimization pipeline for finite-squeezing and hardware-aware CV compilation.

The project is intended as a research codebase and a foundation for compiler development rather than as a finished production compiler.

## Current state

### Implemented foundations

The repository currently contains:

- Core diagram classes:
  - `QSpider`
  - `PSpider`
  - `CompositionDiagram`
  - `TensorDiagram`
  - `ContractedDiagram`
  - Fourier-related diagrams: `Fourier`, `Fourier2` and `FourierInv`
  - `ZxPoly` phase expressions
- CV gates including:
  - displacement
  - phase rotation
  - squeezing
  - controlled-sum / CSUM
  - controlled-Z / CZ
  - beamsplitter
  - cubic phase
- NetworkX conversion utilities in `cvzx.nx_graph`.
- NetworkX-based graph rewrite infrastructure in `cvzx.nx_rewrite_rules`.
- Initial NetworkX implementations of the identity, fusion, and chain-reduction rules.
- Visualization utilities in `cvzx.visualize_base_gates`.
- Tests for base gates, graph conversion, rewrite rules, and visualization.

### In progress

The next development stage is to complete the full optimizer, including:

- Fourier normalization;
- terminal-state and measurement absorption using displacement and squeezing rules;
- copy and quadrature-copy rules;
- finite-squeezing and covariance/noise tracking;
- robust reconstruction of optimized graphs as structured CV-ZX diagrams;
- architecture-aware lowering for CV measurement-based implementations.

## Project layout

```text
cvzx-main/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   └── cvzx/
│       ├── __init__.py
│       ├── base_gates.py
│       ├── gates.py
│       ├── nx_graph.py
│       ├── nx_rewrite_rules.py
│       ├── visualize_base_gates.py
└── tests/
    ├── nx_graph/
    │   ├── test_gate_registry.py
    │   └── test_nx_graph.py
    ├── rewrite_rules/
    │   ├── test_nx_chain_reduction_rules.py
    │   ├── test_nx_fusion_rule.py
    │   └── test_nx_identity_rule.py
    ├── test_base_gates.py
    ├── test_visualize_base_gates.py
    └── test_visualize_gates.py
```

Generated directories such as `build/`, `dist/`, `*.egg-info/`, and `__pycache__/` should not be committed.

## Installation

Create and activate a Python environment, then install the package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For development dependencies:

```bash
python -m pip install -e ".[dev]"
```

If you use the optional Strawberry Fields integration:

```bash
python -m pip install -e ".[sf]"
```

The package currently targets Python 3.10 and newer.

## Running tests

Run the test suite with:

```bash
pytest
```

For coverage:

```bash
pytest --cov=cvzx
```

To visualize diagrams or inspect the effect of rewriting rules, run the relevant visualization test module directly, including its `main` entry point. The current test modules generate before/after visualizations for diagram structures and rewrite-rule applications.

For example:

```bash
python tests/test_visualize_base_gates.py
python tests/test_visualize_gates.py
```

In the future, these visualization workflows will be moved to dedicated Jupyter notebooks.

## Minimal example

```python
from cvzx.base_gates import CompositionDiagram, QSpider, ZxPoly
from cvzx.gates import ControlledSumGate
from cvzx.nx_graph import to_diagram, to_graph

phase = ZxPoly({})
state = QSpider(0, 1, phase)

# Convert a structured diagram to a graph and back.
graph = to_graph(state)
restored = to_diagram(graph)

print(graph)
print(restored)
```

The exact constructor signatures are defined in `src/cvzx/base_gates.py` and `src/cvzx/gates.py`.

## Optimization direction

The intended optimizer follows this high-level pipeline:

```text
CV circuit
    │
    ▼
Structured CV-ZX diagram
    │
    ▼
NetworkX graph representation
    │
    ▼
Identity and chain reduction
    │
    ▼
Spider and gate fusion
    │
    ▼
Fourier / displacement normalization
    │
    ▼
Terminal and copy-rule passes
    │
    ▼
Optimized CV-ZX graph
    │
    ▼
Optimized structured CV-ZX diagram
    │
    ▼
Target-specific lowering
    │
    ├──► CV simulator
    │
    └──► Real CV quantum computer
```

The graph representation stores operation types, phase labels, input/output arities, and port-level connectivity. This is intended to support both rewrite matching and later hardware-aware lowering.

## Research roadmap

Planned extensions include:

1. Complete the currently stubbed graph rewrite functions.
2. Add finite-squeezing parameters and covariance propagation.
3. Track displacement transformations through squeezing and Fourier operations.
4. Add weighted CSUM arithmetic transformations and CZ interaction fusion.
5. Add non-Gaussian resource handling, including cubic-phase and Fock-space interfaces.
6. Benchmark graph reduction on increasingly complex CV arithmetic and interaction diagrams.
7. Add architecture-aware output for measurement-based CV implementations.

## Scientific scope

The project is motivated by continuous-variable quantum computation, CV measurement-based quantum computation, CV-ZX diagrammatic reasoning, Gaussian and non-Gaussian gate compilation, and graph-based optimization.

The current repository should be read as a snapshot of ongoing research. It documents the implemented foundations and preserves the planned interfaces for future work.

## Acknowledgements

This project is an independent compiler prototype that builds on the
continuous-variable ZX formalism introduced in:

> Hironari Nagayoshi, Warit Asavanant, Ryuhoh Ide, Kosuke Fukui,
> Atsushi Sakaguchi, Jun-ichi Yoshikawa, Nicolas C. Menicucci, and
> Akira Furusawa, “ZX graphical calculus for continuous-variable quantum
> processes,” *Physical Review Research* **7**, 033141 (2025).
> [arXiv:2405.07246](https://arxiv.org/abs/2405.07246)

The CVZX Compiler is an independent software implementation and research
prototype. It is not an official implementation of, nor affiliated with, the
authors or institutions of the original work.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

## Contact and contributions

This is currently a personal research repository. Issues, discussions, and pull requests are welcome as the project develops.

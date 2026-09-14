"""Plugin registry for lowering a cvzx `Diagram` into an mqc3 `DependencyDAG`.

Different QPUs can require different lowering strategies once a circuit
leaves cvzx's diagrammatic representation. mqc3 already has exactly
this kind of plugin point *downstream* of `DependencyDAG`:
`mqc3.graph.embed.embed.GraphEmbedder` is an abstract base class with
concrete per-strategy subclasses (`beamsearch.py`, `greedy.py`) that
each embed a `DependencyDAG` into a concrete `GraphRepr` differently,
and `GraphRepr` is in turn lowered toward a machinery representation
via `mqc3.machinery`. `DependencyDAG` itself is QPU-agnostic -- it only
encodes per-mode operation dependencies and feedforward edges, not
anything hardware-specific -- so it is the natural, stable interface
for cvzx to hand off to mqc3's own machinery.

This module provides the analogous plugin point for the one step still
missing: cvzx `Diagram` -> mqc3 `DependencyDAG`. A `LoweringBackend` is
any strategy for performing that step; backends register themselves
under a name via the `register_backend` class decorator, and
`graph_to_dependency_dag(diagram, backend=...)` dispatches to whichever
one is requested (`"mqc3"`, the reference backend, by default). Adding
support for a QPU that needs a different `Diagram -> DependencyDAG`
lowering (for instance, one that wants to skip the `CircuitRepr`
round-trip and build the DAG directly from the diagram's own structure)
means writing and registering one more `LoweringBackend` subclass --
`get_backend`/`graph_to_dependency_dag` and every existing backend are
untouched.

The bundled reference backend (registered as `"mqc3"`) is deliberately
the simplest correct implementation: it converts the diagram to an
mqc3 `CircuitRepr` (via `cvzx.lowering.bridges.mqc3.to_circuit_repr` -- see that
module for the exact per-gate translation and its documented
limitations) and then builds the `DependencyDAG` from it using mqc3's
own, already-tested `_DependencyBuilder.from_circuit()`. Every
limitation of `to_circuit_repr` (no `ContractedDiagram`, no symbolic
parameters, no `ControlledSumGate`/`CubicPhaseGate`, and so on) applies
transitively to this backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from mqc3.graph.embed.dep_dag import DependencyDAG

    from cvzx.ir.base import Diagram

__all__ = [
    "CvzxDirectBackend",
    "LoweringBackend",
    "Mqc3ReferenceBackend",
    "get_backend",
    "graph_to_dependency_dag",
    "list_backends",
    "register_backend",
]


class LoweringBackend(ABC):
    """A strategy for lowering a cvzx `Diagram` into an mqc3 `DependencyDAG`.

    Subclasses must be constructible with no arguments (`register_backend`
    instantiates them immediately at registration time).
    """

    @abstractmethod
    def to_dependency_dag(self, diagram: Diagram) -> DependencyDAG:
        """Lower `diagram` into an mqc3 `DependencyDAG`.

        Parameters
        ----------
        diagram : Diagram
            The cvzx diagram to lower. Implementations are free to
            impose their own restrictions on what `diagram` may
            contain (documented on the implementation itself).

        Returns
        -------
        DependencyDAG
            The resulting dependency DAG, ready for mqc3's own
            `GraphEmbedder` machinery to embed into a concrete
            `GraphRepr`.
        """


_REGISTRY: dict[str, LoweringBackend] = {}


def register_backend(name: str) -> Callable[[type[LoweringBackend]], type[LoweringBackend]]:
    """Class decorator registering a `LoweringBackend` subclass under `name`.

    The decorated class is instantiated immediately (with no arguments)
    and the instance is stored in the module-level registry; the class
    itself is returned unchanged, so it remains usable directly.

    Returns
    -------
    Callable[[type[LoweringBackend]], type[LoweringBackend]]
        A decorator that registers its `LoweringBackend` subclass
        argument under `name` and returns it unchanged.

    Examples
    --------
    >>> @register_backend("my_qpu")
    ... class MyQpuBackend(LoweringBackend):
    ...     def to_dependency_dag(self, diagram):
    ...         ...
    """

    def _decorator(cls: type[LoweringBackend]) -> type[LoweringBackend]:
        _REGISTRY[name] = cls()
        return cls

    return _decorator


def get_backend(name: str) -> LoweringBackend:
    """Look up a registered `LoweringBackend` by name.

    Returns
    -------
    LoweringBackend
        The backend instance registered under `name`.

    Raises
    ------
    KeyError
        If no backend is registered under `name`. The message lists
        every currently-registered backend name.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        msg = f"No lowering backend registered under {name!r}. Available backends: {available}."
        raise KeyError(msg) from None


def list_backends() -> list[str]:
    """Return the names of every currently-registered backend, sorted.

    Returns
    -------
    list[str]
        The registered backend names, sorted.
    """
    return sorted(_REGISTRY)


def graph_to_dependency_dag(diagram: Diagram, backend: str = "mqc3") -> DependencyDAG:
    """Lower a cvzx `Diagram` into an mqc3 `DependencyDAG` via a named backend.

    Parameters
    ----------
    diagram : Diagram
        The cvzx diagram to lower.
    backend : str
        Name of the registered `LoweringBackend` to use. Defaults to
        `"mqc3"`, the bundled reference backend (see this module's
        docstring). Must name a backend `list_backends()` lists, or
        `get_backend` raises `KeyError`.

    Returns
    -------
    DependencyDAG
        The resulting dependency DAG.
    """
    return get_backend(backend).to_dependency_dag(diagram)


@register_backend("mqc3")
class Mqc3ReferenceBackend(LoweringBackend):
    """Reference lowering backend: `Diagram` -> `CircuitRepr` -> `DependencyDAG`.

    Converts via `cvzx.lowering.bridges.mqc3.to_circuit_repr` and then builds the
    `DependencyDAG` using mqc3's own `_DependencyBuilder.from_circuit()`
    (through `DependencyDAG`'s own constructor) -- no cvzx-specific
    dependency-graph logic is reimplemented here. See
    `cvzx.lowering.bridges.mqc3` for the exact per-gate translation performed
    and its documented limitations, which apply transitively to this
    backend.
    """

    def to_dependency_dag(self, diagram: Diagram) -> DependencyDAG:
        """Lower `diagram` via `to_circuit_repr` then mqc3's own `DependencyDAG`.

        Returns
        -------
        DependencyDAG
            The resulting dependency DAG.
        """
        from mqc3.graph.embed.dep_dag import DependencyDAG  # ruff: ignore[import-outside-top-level]

        from cvzx.lowering.bridges.mqc3 import to_circuit_repr  # ruff: ignore[import-outside-top-level]

        circuit = to_circuit_repr(diagram)
        return DependencyDAG(circuit)


@register_backend("cvzx-direct")
class CvzxDirectBackend(LoweringBackend):
    """Direct lowering backend: `Diagram` -> `CVZXGraph` -> `DependencyDAG`.

    Skips `normalize_diagram`'s canonical-stage requirement: execution
    order is discovered by a single forward sweep over the diagram's own
    `CVZXGraph` structure instead, anchored at `GateRegister.input_states`
    and dual-backend (uses whichever of `networkx`/`rustworkx` is
    selected, or the default -- see `cvzx.config.Backend`). See
    `cvzx.lowering.dag` for the full algorithm; per-leaf op translation
    is still delegated to `cvzx.lowering.bridges.mqc3`'s own translators (no
    duplicated translation logic), so the same `FeedForward` support and
    documented per-gate limitations apply. Automatically closes any open
    input/output ports first via `cvzx.passes.completion.complete_boundaries()`.
    """

    def to_dependency_dag(self, diagram: Diagram) -> DependencyDAG:
        """Lower `diagram` via a direct `CVZXGraph` forward sweep.

        Returns
        -------
        DependencyDAG
            The resulting dependency DAG.
        """
        from cvzx.lowering.dag import extract_dependency_dag  # ruff: ignore[import-outside-top-level]

        return extract_dependency_dag(diagram)

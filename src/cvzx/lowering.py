"""Plugin registry for lowering a cvzx `Diagram` into an mqc3 `MachineryRepr`.

A `LoweringBackend` is any strategy for that step; backends register under a
name via the `register_backend` class decorator, and
`graph_to_machinery_repr(diagram, n_local_macronodes=..., backend=...)`
dispatches to whichever one is requested (`"mqc3"`, the bundled reference
backend, by default). Adding a QPU-specific backend means writing and
registering one more `LoweringBackend` subclass -- `get_backend`/
`graph_to_machinery_repr` and every existing backend are untouched. See the
docs' dev guide ("Architecture overview") for why `MachineryRepr` is the
right handoff point, and the user guide ("Converting to and from mqc3
circuits") for a worked example.

The bundled `"mqc3"` backend converts the diagram to an mqc3 `CircuitRepr`
(via `cvzx.diagram_to_circuit.to_circuit_repr` -- see that module for the
exact per-gate translation and its documented limitations, which apply
transitively here), builds a `DependencyDAG` from it, embeds that DAG into a
`GraphRepr` with mqc3's `GreedyEmbedder`, and converts the result to a
`MachineryRepr` via mqc3's own `MachineryRepr.from_graph_repr`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from mqc3.machinery import MachineryRepr

    from cvzx.base_gates import Diagram

__all__ = [
    "LoweringBackend",
    "Mqc3ReferenceBackend",
    "get_backend",
    "graph_to_machinery_repr",
    "list_backends",
    "register_backend",
]


class LoweringBackend(ABC):
    """A strategy for lowering a cvzx `Diagram` into an mqc3 `MachineryRepr`.

    Subclasses must be constructible with no arguments (`register_backend`
    instantiates them immediately at registration time).
    """

    @abstractmethod
    def to_machinery_repr(self, diagram: Diagram, *, n_local_macronodes: int) -> MachineryRepr:
        """Lower `diagram` into an mqc3 `MachineryRepr`.

        Parameters
        ----------
        diagram : Diagram
            The cvzx diagram to lower. Implementations are free to impose
            their own restrictions on what `diagram` may contain
            (documented on the implementation itself).
        n_local_macronodes : int
            Number of macronodes per column in the target `GraphRepr`/
            `MachineryRepr` (mqc3's `GraphEmbedSettings.n_local_macronodes`).
            Must be large enough to embed `diagram`'s dependency structure --
            implementations should let mqc3's own embedder raise if it isn't,
            rather than silently guessing a value of their own.

        Returns
        -------
        MachineryRepr
            The resulting machinery representation.
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
    ...     def to_machinery_repr(self, diagram, *, n_local_macronodes):
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


def graph_to_machinery_repr(diagram: Diagram, *, n_local_macronodes: int, backend: str = "mqc3") -> MachineryRepr:
    """Lower a cvzx `Diagram` into an mqc3 `MachineryRepr` via a named backend.

    Parameters
    ----------
    diagram : Diagram
        The cvzx diagram to lower.
    n_local_macronodes : int
        Number of macronodes per column in the target representation --
        see `LoweringBackend.to_machinery_repr`.
    backend : str
        Name of the registered `LoweringBackend` to use. Defaults to
        `"mqc3"`, the bundled reference backend (see this module's
        docstring). Must name a backend `list_backends()` lists, or
        `get_backend` raises `KeyError`.

    Returns
    -------
    MachineryRepr
        The resulting machinery representation.
    """
    return get_backend(backend).to_machinery_repr(diagram, n_local_macronodes=n_local_macronodes)


@register_backend("mqc3")
class Mqc3ReferenceBackend(LoweringBackend):
    """Reference lowering backend: `Diagram` -> `CircuitRepr` -> `DependencyDAG` -> `GraphRepr` -> `MachineryRepr`.

    Converts via `cvzx.diagram_to_circuit.to_circuit_repr`, builds a
    `DependencyDAG` from the result, embeds it into a `GraphRepr` with
    mqc3's `GreedyEmbedder` (the simplest embedding strategy mqc3 provides --
    a QPU wanting a different layout strategy, e.g. `BeamSearchEmbedder`,
    should register its own backend instead), and converts to `MachineryRepr`
    via mqc3's own `MachineryRepr.from_graph_repr`. Every limitation of
    `to_circuit_repr` (no `ContractedDiagram`, no symbolic parameters, no
    `ControlledSumGate`/`CubicPhaseGate`, and so on) applies transitively to
    this backend, plus mqc3's own embedding requirements (every mode must
    originate at an initialization and terminate at a measurement).
    """

    def to_machinery_repr(self, diagram: Diagram, *, n_local_macronodes: int) -> MachineryRepr:
        """Lower `diagram` via `to_circuit_repr`, `DependencyDAG`, and `GreedyEmbedder`.

        Returns
        -------
        MachineryRepr
            The resulting machinery representation.
        """
        from mqc3.graph.embed.dep_dag import DependencyDAG  # ruff: ignore[import-outside-top-level]
        from mqc3.graph.embed.greedy import GreedyEmbedder, GreedyEmbedSettings  # ruff: ignore[import-outside-top-level]
        from mqc3.machinery import MachineryRepr  # ruff: ignore[import-outside-top-level]

        from cvzx.diagram_to_circuit import to_circuit_repr  # ruff: ignore[import-outside-top-level]

        circuit = to_circuit_repr(diagram)
        dep_dag = DependencyDAG(circuit)
        settings = GreedyEmbedSettings(n_local_macronodes=n_local_macronodes)
        graph_repr = GreedyEmbedder(settings).embed(dep_dag)
        return MachineryRepr.from_graph_repr(graph_repr)

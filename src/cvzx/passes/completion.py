"""Close a diagram's open output boundary with symbolic measurement effects.

`optimize()`'s rewrite rules deliberately never terminate a wire on their
own, so a diagram destined for `cvzx.lowering.bridges.mqc3.to_circuit_repr`
generally still has open output ports. `complete_diagram()` closes each
one with a fresh, symbolically-labeled `(1, 0)` `QSpider`/`PSpider`
measurement effect (`ZxPoly({1: -m})` for a fresh symbol `m`);
`complete_boundaries()` additionally closes any open input ports with a
fresh idealized state first. See :doc:`../user_guide/circuit_conversion`
for the effect-leaf convention, why a straddling 2-mode gate needs no
special-case handling, and the recommended feedforward-relinking flow via
`relink_measurement_symbol()`.
"""

from itertools import count
from typing import TYPE_CHECKING, NamedTuple

from sympy import Symbol

from cvzx.backend import get_backend_modules
from cvzx.config import Backend
from cvzx.exceptions import UnboundMeasurementError
from cvzx.ir.base import Diagram, PSpider, QSpider, TensorDiagram, ZxPoly

if TYPE_CHECKING:
    from cvzx.backends.nx.graph import CVZXGraph as NxCVZXGraph
    from cvzx.backends.rx.graph import CVZXGraph as RxCVZXGraph

__all__ = [
    "BoundaryCompletionResult",
    "CompletionResult",
    "complete_boundaries",
    "complete_diagram",
    "relink_measurement_symbol",
]

_symbol_counter = count(1)

_BASIS_TO_CLS = {"Q": QSpider, "P": PSpider}


class CompletionResult(NamedTuple):
    """`complete_diagram()`'s return value.

    Attributes
    ----------
    graph
        The completed `CVZXGraph` (`networkx`- or `rustworkx`-backed
        depending on `backend`), with `num_outputs == 0` and its registry
        already rebuilt to include the newly appended measurement leaves.
    diagram : Diagram
        The completed compact-form `Diagram` (`diagram` itself, unchanged,
        if it had no open outputs to begin with).
    bindings : dict[Symbol, int]
        One entry per newly appended measurement leaf: the fresh symbol
        bound to that leaf's own `Diagram.id` (which is also its node id
        once converted to a graph). Empty if `diagram` had no open outputs.
    """

    graph: "NxCVZXGraph | RxCVZXGraph"
    diagram: Diagram
    bindings: dict[Symbol, int]


def _fresh_symbol() -> Symbol:
    """A fresh, globally-unique (within this process) measurement-outcome symbol.

    Returns
    -------
    Symbol
    """
    return Symbol(f"m_{next(_symbol_counter)}", real=True)


def complete_diagram(
    diagram: Diagram,
    *,
    backend: Backend | str | None = None,
    basis: str | list[str] = "Q",
) -> CompletionResult:
    """Close every open output port of `diagram` with a fresh symbolic measurement effect.

    Parameters
    ----------
    diagram : Diagram
        The (typically already-`optimize()`d) diagram to complete. Not
        mutated -- a new, composed `Diagram` is returned.
    backend : Backend | str | None
        Which `CVZXGraph` backend to build the completed graph with -- see
        `cvzx.config.Backend`. `None` (default) uses
        `cvzx.config.DEFAULT_BACKEND`.
    basis : str | list[str]
        Which quadrature each newly closed port is measured in: `"Q"`
        (x-homodyne) or `"P"` (p-homodyne). A single string applies that
        basis to every open port; a list must have exactly
        `diagram.num_outputs` entries, one per port in port order.

    Returns
    -------
    CompletionResult
        The completed graph, diagram, and the fresh symbol -> node id
        bindings for the newly appended measurement leaves.

    Raises
    ------
    ValueError
        If `basis` (or any entry of it) is not `"Q"`/`"P"`, or if `basis`
        is a list whose length doesn't match `diagram.num_outputs`.
    """
    n_open = diagram.num_outputs

    bases = [basis] * n_open if isinstance(basis, str) else list(basis)
    if len(bases) != n_open:
        msg = f"basis has {len(bases)} entries but diagram has {n_open} open output port(s)."
        raise ValueError(msg)
    for b in bases:
        if b not in _BASIS_TO_CLS:
            msg = f"Unknown basis {b!r}; expected one of {sorted(_BASIS_TO_CLS)}."
            raise ValueError(msg)

    bindings: dict[Symbol, int] = {}
    if n_open == 0:
        completed_diagram = diagram
    else:
        effects = []
        for b in bases:
            m = _fresh_symbol()
            cls = _BASIS_TO_CLS[b]
            effect = cls(1, 0, ZxPoly({1: -m}), parametric=True)
            effects.append(effect)
            bindings[m] = effect.id
        closing_layer = TensorDiagram(effects) if len(effects) > 1 else effects[0]
        completed_diagram = closing_layer.compose(diagram)

    _, graph_mod, _ = get_backend_modules(backend)
    graph = graph_mod.to_graph(completed_diagram)
    return CompletionResult(graph=graph, diagram=completed_diagram, bindings=bindings)


class BoundaryCompletionResult(NamedTuple):
    """`complete_boundaries()`'s return value.

    Attributes
    ----------
    graph
        The fully closed `CVZXGraph` (`num_inputs == num_outputs == 0`),
        registry already rebuilt.
    diagram : Diagram
        The fully closed compact-form `Diagram`.
    measurement_bindings : dict[Symbol, int]
        One entry per synthesized output-side measurement effect: the
        fresh symbol bound to that leaf's node id (see
        `CompletionResult.bindings` -- input-side states get no symbol,
        since a state has no "outcome" to feed forward from).
    n_synthesized_inputs : int
        How many open input ports were closed with a fresh state.
    n_synthesized_outputs : int
        How many open output ports were closed with a fresh measurement
        effect (`len(measurement_bindings)`).
    """

    graph: "NxCVZXGraph | RxCVZXGraph"
    diagram: Diagram
    measurement_bindings: dict[Symbol, int]
    n_synthesized_inputs: int
    n_synthesized_outputs: int


def complete_boundaries(
    diagram: Diagram,
    *,
    backend: Backend | str | None = None,
    input_basis: str | list[str] = "Q",
    output_basis: str | list[str] = "Q",
) -> BoundaryCompletionResult:
    """Close both open input and output ports of `diagram`.

    Every open input port gets a fresh idealized (zero-phase) state leaf
    (no symbol minted, since a state has no outcome); every open output
    port gets a fresh symbolic measurement effect exactly as
    `complete_diagram()` produces. The result is what
    `cvzx.lowering.dag.extract_dependency_dag()`'s forward sweep requires
    -- see :doc:`../user_guide/circuit_conversion` for the full mechanics.

    Parameters
    ----------
    diagram : Diagram
        The diagram to close. Not mutated.
    backend : Backend | str | None
        Which `CVZXGraph` backend to build the completed graph with.
        `None` (default) uses `cvzx.config.DEFAULT_BACKEND`.
    input_basis : str | list[str]
        Which quadrature each newly opened input port's state is
        idealized along: `"Q"` (the x-eigenstate) or `"P"` (the
        p-eigenstate). Same broadcasting rule as `output_basis`.
    output_basis : str | list[str]
        Forwarded to `complete_diagram()` as `basis` -- see there.

    Returns
    -------
    BoundaryCompletionResult

    Raises
    ------
    ValueError
        If `input_basis`/`output_basis` (or any entry) isn't `"Q"`/`"P"`,
        or a list of the wrong length.
    """
    n_open_in = diagram.num_inputs

    in_bases = [input_basis] * n_open_in if isinstance(input_basis, str) else list(input_basis)
    if len(in_bases) != n_open_in:
        msg = f"input_basis has {len(in_bases)} entries but diagram has {n_open_in} open input port(s)."
        raise ValueError(msg)
    for b in in_bases:
        if b not in _BASIS_TO_CLS:
            msg = f"Unknown basis {b!r}; expected one of {sorted(_BASIS_TO_CLS)}."
            raise ValueError(msg)

    if n_open_in:
        states = [_BASIS_TO_CLS[b](0, 1, ZxPoly({})) for b in in_bases]
        opening_layer = TensorDiagram(states) if len(states) > 1 else states[0]
        diagram = diagram.compose(opening_layer)

    output_result = complete_diagram(diagram, backend=backend, basis=output_basis)
    return BoundaryCompletionResult(
        graph=output_result.graph,
        diagram=output_result.diagram,
        measurement_bindings=output_result.bindings,
        n_synthesized_inputs=n_open_in,
        n_synthesized_outputs=len(output_result.bindings),
    )


def _node_attrs(cvzx_graph: "NxCVZXGraph | RxCVZXGraph", node_id: int) -> dict:
    """Return the raw attribute payload for `node_id`, independent of graph backend.

    Returns
    -------
    dict
    """
    raw = cvzx_graph.graph
    if hasattr(raw, "node_indices"):  # rustworkx PyDiGraph
        id_map = {raw[idx]["id"]: idx for idx in raw.node_indices()}
        return raw[id_map[node_id]]  # type: ignore[no-any-return]
    return raw.nodes[node_id]  # type: ignore[index, no-any-return]


def relink_measurement_symbol(cvzx_graph: "NxCVZXGraph | RxCVZXGraph", symbol: Symbol, node_id: int) -> None:
    """Repoint every node whose `param_measurement_map` cites `symbol` to `node_id`.

    For every node indexed under `symbol` in
    `cvzx_graph.registry.symbol_registry` that also has a
    `param_measurement_map` entry for it, rewrites that entry to
    `{node_id}`, re-derives `feedforward`/`measurement_ids`, and rebuilds
    the registry. See :doc:`../user_guide/circuit_conversion` for the
    recommended relinking flow and an aliasing note worth knowing about.

    Parameters
    ----------
    cvzx_graph : CVZXGraph
        The graph to update in place (either backend).
    symbol : Symbol
        The feedforward symbol to repoint.
    node_id : int
        The (now-correct) measurement node id `symbol` should be bound to.

    Raises
    ------
    UnboundMeasurementError
        If `node_id` is not a registered measurement node in
        `cvzx_graph.registry.measurement_nodes`.
    """
    if node_id not in cvzx_graph.registry.measurement_nodes:
        msg = f"Cannot relink symbol {symbol!r} to node {node_id}: it is not a registered measurement node."
        raise UnboundMeasurementError(msg)

    referencing = set(cvzx_graph.registry.symbol_registry.get(symbol, ()))
    changed = False
    for ref_id in referencing:
        attrs = _node_attrs(cvzx_graph, ref_id)
        param_measurement_map = attrs.get("param_measurement_map")
        if not param_measurement_map or symbol not in param_measurement_map:
            continue
        param_measurement_map[symbol] = {node_id}
        attrs["measurement_ids"] = set().union(*param_measurement_map.values())
        attrs["feedforward"] = bool(attrs["measurement_ids"])
        changed = True

    if changed:
        cvzx_graph.rebuild_registry()

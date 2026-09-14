"""Close a diagram's open output boundary with symbolic measurement effects.

An `optimize()`d diagram destined for physical mqc3 execution generally
still has open output ports (unmeasured modes) -- `optimize()` and the
rewrite rules it runs deliberately never terminate a wire on their own,
since a boundary state/effect sitting in the graph would otherwise block
spider fusion, chain reduction, and identity removal during the rewrite
loop itself. `complete_diagram()` is the dedicated post-`optimize()` pass
that closes those open ports: it appends one fresh, symbolically-labeled
measurement effect per open output port, so the result is ready to feed
into `cvzx.diagram_to_circuit.to_circuit_repr`.

Each appended effect is a `(1, 0)` `QSpider`/`PSpider` with phase
`ZxPoly({1: -m})` for a fresh `sympy.Symbol` `m` -- the standard CV-ZX
notation for "the idealized homodyne effect whose own outcome is `m`"
(see e.g. `MeasurementGate`'s own docstring for the same `QSpider(1, 0, 0)`
convention at a *fixed* outcome). `QSpider` measures the x-quadrature
(mqc3 `intrinsic.Measurement(theta=pi/2)`); `PSpider` measures the
p-quadrature (`theta=0`) -- see `cvzx.diagram_to_circuit`'s own
`_apply_1mode_leaf` for that exact mapping.

Why this doesn't need special-case handling for a straddling 2-mode gate
------------------------------------------------------------------------
A boundary-completing layer is built the same way any other closing layer
in this codebase is: a `TensorDiagram` of one 1-mode effect per output
port, composed onto `diagram`. `Diagram.compose()`/`TensorDiagram`
construction already validate that the closing layer's total input arity
matches `diagram`'s total output arity port-for-port (see
`cvzx.exceptions.ArityMismatchError`) -- so whether a given output port
came from a 1-mode gate or is one of a 2-mode gate's two outputs makes no
difference here: each port still gets exactly one effect, and the
existing arity system rejects any mismatch outright rather than silently
misattributing a port.

Downstream feedforward relinking
---------------------------------
Each output port's fresh symbol `m` is known (via the returned
`CompletionResult.bindings`) only *after* `complete_diagram()` runs, since
that's when the measurement leaf (and its node id) is actually created.
The recommended flow is to build any feedforward-dependent gate (e.g.
`DisplacementGate(m, param_measurement_map={m: {node_id}})`) using
`bindings` *after* calling `complete_diagram()`. If a gate was already
built earlier referencing a symbol that turns out to need re-pointing at
a different measurement node id (e.g. because the diagram was assembled
in pieces), use `relink_measurement_symbol()` to repoint every graph node
that cites that symbol and resynchronize the registry (both backends) in
one step, rather than mutating node attributes by hand.
"""

from itertools import count
from typing import TYPE_CHECKING, NamedTuple

from sympy import Symbol

from cvzx.backend import get_backend_modules
from cvzx.base_gates import Diagram, PSpider, QSpider, TensorDiagram, ZxPoly
from cvzx.config import Backend
from cvzx.exceptions import UnboundMeasurementError

if TYPE_CHECKING:
    from cvzx.nx_graph import CVZXGraph as NxCVZXGraph
    from cvzx.rx_graph import CVZXGraph as RxCVZXGraph

__all__ = ["CompletionResult", "complete_diagram", "relink_measurement_symbol"]

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

    For every node currently indexed under `symbol` in
    `cvzx_graph.registry.symbol_registry` (i.e. every node whose own
    parameters mention `symbol`) that also has a `param_measurement_map`
    entry for `symbol`, rewrites that entry to `{node_id}` and re-derives
    `feedforward`/`measurement_ids` accordingly, then rebuilds the
    registry (both backends use the same `rebuild_registry()` API) so
    `symbol_registry`/`measurement_to_feedforward_map` reflect the change.

    A node's `param_measurement_map` is the SAME dict object as the
    underlying `Diagram` instance's own attribute (not a copy) -- this
    intentionally also updates that `Diagram` object in place, so a
    diagram built from the same objects stays consistent with the graph.

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

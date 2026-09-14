"""Extract an mqc3 `DependencyDAG` directly from a `CVZXGraph`.

`cvzx.lowering.Mqc3ReferenceBackend` builds the `DependencyDAG` by first
canonicalizing the diagram into `normalize_diagram`'s alternating
type-1/type-2 stages and walking those. This module builds the exact
same kind of `DependencyDAG` a different way: a single deterministic
forward sweep over the `CVZXGraph`'s own node/edge structure, anchored at
`GateRegister.input_states` and following the graph's own `"composition"`/
`"contracted_internal"` wire edges (the same edges `nx_rewrite_rules`/
`rx_rewrite_rules` already use for rule-matching, so no new edge
convention is introduced) until each mode reaches a
`GateRegister.measurement_nodes` node -- without requiring the diagram to
already be in `normalize_diagram`'s canonical form first.

`mqc3.graph.embed.dep_dag.DependencyDAG` is not itself backend-specific
(its `.dag` is always a plain `networkx.DiGraph` internally, and its
constructor only accepts an mqc3 `CircuitRepr`/`GraphRepr`) -- there is no
rustworkx-backed variant of it to build instead. What *is* backend-aware
here is the traversal: `extract_dependency_dag()` walks whichever
`CVZXGraph` it's given (`networkx`- or `rustworkx`-backed, dispatched via
`cvzx.backend`), then feeds the resulting mode-ordered leaf sequence
through the same per-leaf translators `cvzx.lowering.bridges.mqc3` already
uses (`_apply_1mode_leaf`/`_apply_2mode_leaf`, including their existing
`FeedForward` support) to build a `CircuitRepr`, and hands that to
`DependencyDAG` -- no cvzx-specific dependency-graph logic is
reimplemented for the actual op translation, only the traversal order.

Algorithm
---------
1. `complete_boundaries()` (unless `complete=False`) closes every open
   input/output port, so every wire is bounded by a real
   `GateRegister.input_states`/`measurement_nodes` node -- the
   precondition the sweep below relies on.
2. Every `input_states` node starts a fresh mode (a monotonically
   increasing integer, exactly as `cvzx.lowering.bridges.mqc3`'s own
   `_ModeCounter` assigns them).
3. A node is visited once **both** of its dependencies are satisfied:
   every one of its input ports has a mode threaded into it from an
   already-visited node (a "wire-ready" condition, propagated forward
   along `"composition"`/`"contracted_internal"` edges port-for-port),
   and every measurement id any of its parameters' `param_measurement_map`
   cites has itself already been visited (a "classical-ready" condition
   -- this is what guarantees a feedforward edge never points at an
   unvisited measurement). A 1-mode leaf's single output port inherits
   its input's mode; a 2-mode leaf's two output ports inherit its two
   inputs' modes unchanged (matching `to_circuit_repr`'s own convention
   that a wide gate never advances the mode counter); a measurement
   effect (or any other 1-in-0-out leaf) ends its mode's thread.
4. Container nodes (`kind == "container"`) are skipped entirely --
   `V_leaves` is exactly the proper/compact node set.
"""

from collections import deque
from typing import TYPE_CHECKING

from cvzx.backend import get_backend_modules
from cvzx.config import Backend
from cvzx.ir.base import Diagram, VoidDiagram
from cvzx.lowering.bridges.mqc3 import MeasurementOps, _apply_1mode_leaf, _apply_2mode_leaf, _open_mode_state
from cvzx.passes.completion import complete_boundaries

if TYPE_CHECKING:
    from mqc3.circuit import CircuitRepr
    from mqc3.graph.embed.dep_dag import DependencyDAG

    from cvzx.backends.nx.graph import CVZXGraph as NxCVZXGraph
    from cvzx.backends.rx.graph import CVZXGraph as RxCVZXGraph

__all__ = ["extract_dependency_dag"]

_WIRE_EDGE_TYPES = {"composition", "contracted_internal"}
_LEAF_KINDS = {"proper", "compact"}


class _GraphAccess:
    """Backend-independent read access to a `CVZXGraph`'s nodes and wire edges.

    Built once per extraction (not per node) so the `rustworkx` id ->
    index map is only ever constructed a single time.
    """

    def __init__(self, cvzx_graph: "NxCVZXGraph | RxCVZXGraph", *, is_rx: bool) -> None:
        self.cvzx_graph = cvzx_graph
        self.is_rx = is_rx
        self.raw = cvzx_graph.graph
        self.id_to_idx = {self.raw[idx]["id"]: idx for idx in self.raw.node_indices()} if is_rx else None

    def leaf_node_ids(self) -> list[int]:
        """Every `kind in {"proper", "compact"}` node id, in id order.

        Returns
        -------
        list[int]
        """
        if self.is_rx:
            ids = [self.raw[idx]["id"] for idx in self.raw.node_indices() if self.raw[idx].get("kind") in _LEAF_KINDS]
        else:
            ids = [n for n, attrs in self.raw.nodes(data=True) if attrs.get("kind") in _LEAF_KINDS]  # type: ignore[call-arg]
        return sorted(ids)

    def attrs(self, node_id: int) -> dict:
        """The raw attribute payload for `node_id`.

        Returns
        -------
        dict
        """
        if self.is_rx:
            return self.raw[self.id_to_idx[node_id]]  # type: ignore[no-any-return, index]
        return self.raw.nodes[node_id]  # type: ignore[no-any-return, index]

    def out_wire_edges(self, node_id: int) -> list[tuple[int, dict]]:
        """`(target_id, edge_data)` for every outgoing composition/contracted-internal edge.

        Returns
        -------
        list[tuple[int, dict]]
        """
        if self.is_rx:
            idx = self.id_to_idx[node_id]  # type: ignore[index]
            edges = [
                (self.raw[t]["id"], data)
                for _, t, data in self.raw.out_edges(idx)
                if data.get("edge_type") in _WIRE_EDGE_TYPES
            ]
        else:
            edges = [
                (t, data)
                for _, t, data in self.raw.out_edges(node_id, data=True)  # type: ignore[call-arg]
                if data.get("edge_type") in _WIRE_EDGE_TYPES
            ]
        return edges

    def reconstruct(self, node_id: int, graph_mod: object) -> Diagram:
        """Reconstruct the `Diagram` leaf sitting at `node_id`.

        Returns
        -------
        Diagram
        """
        if self.is_rx:
            return graph_mod.reconstruct_from_node(  # type: ignore[attr-defined, no-any-return]
                self.raw, node_id, self.cvzx_graph.registry, self.id_to_idx
            )
        return graph_mod.reconstruct_from_node(self.raw, node_id, self.cvzx_graph.registry)  # type: ignore[attr-defined, no-any-return]


def _measurement_ids_of(attrs: dict) -> set[int]:
    """Every measurement id any of `attrs["param_measurement_map"]`'s bindings cites.

    Returns
    -------
    set[int]
    """
    param_measurement_map = attrs.get("param_measurement_map") or {}
    ids: set[int] = set()
    for binding in param_measurement_map.values():
        ids |= set(binding)
    return ids


def _mode_ordered_leaves(  # ruff: ignore[complex-structure, too-many-branches, too-many-locals]
    access: _GraphAccess,
) -> list[tuple[int, list[int]]]:
    """Forward-sweep the graph, returning each leaf's node id and its mode(s), in visit order.

    Implements this module's own "Algorithm" section (steps 2-4).

    Returns
    -------
    list[tuple[int, list[int]]]
        `(node_id, modes)` per visited leaf, in the order they became
        ready -- `modes` has one entry per leaf, except a 2-mode leaf
        (two entries, input order) and a fresh state (one entry, its own
        newly minted mode).
    """
    leaf_ids = access.leaf_node_ids()
    attrs_by_id = {node_id: access.attrs(node_id) for node_id in leaf_ids}

    wire_pending: dict[int, dict[int, int]] = {node_id: {} for node_id in leaf_ids}
    wire_ready: set[int] = set()
    classical_ready: set[int] = set()
    remaining_classical: dict[int, set[int]] = {}
    pending_on: dict[int, list[int]] = {}

    for node_id, attrs in attrs_by_id.items():
        needed = _measurement_ids_of(attrs)
        if needed:
            remaining_classical[node_id] = set(needed)
            for measurement_id in needed:
                pending_on.setdefault(measurement_id, []).append(node_id)
        else:
            classical_ready.add(node_id)

    queue: deque[int] = deque()
    queued: set[int] = set()

    def _try_enqueue(node_id: int) -> None:
        if node_id not in queued and node_id in wire_ready and node_id in classical_ready:
            queue.append(node_id)
            queued.add(node_id)

    for node_id in access.cvzx_graph.registry.input_states:
        wire_ready.add(node_id)
        _try_enqueue(node_id)

    mode_counter = 0
    visit_order: list[tuple[int, list[int]]] = []

    while queue:
        node_id = queue.popleft()
        attrs = attrs_by_id[node_id]
        num_in = attrs.get("num_inputs", 0)
        num_out = attrs.get("num_outputs", 0)

        if num_in == 0:
            modes = [mode_counter]
            mode_counter += 1
        else:
            modes = [wire_pending[node_id][port] for port in range(num_in)]

        visit_order.append((node_id, modes))

        # Any node id could be cited as a measurement id in some other
        # node's `param_measurement_map`, regardless of its own type --
        # unblock those waiters unconditionally now that this node (and
        # therefore whatever measurement it may represent) is visited.
        for waiting_id in pending_on.get(node_id, []):
            remaining_classical[waiting_id].discard(node_id)
            if not remaining_classical[waiting_id]:
                classical_ready.add(waiting_id)
                _try_enqueue(waiting_id)

        if num_out == 0:
            continue

        out_modes = modes if num_out == num_in else [modes[0]]
        for out_port, mode_id in enumerate(out_modes):
            for target_id, edge_data in access.out_wire_edges(node_id):
                src_ports = edge_data.get("source_ports", [])
                if out_port not in src_ports:
                    continue
                target_port = edge_data.get("target_ports", [])[src_ports.index(out_port)]
                wire_pending[target_id][target_port] = mode_id
                if len(wire_pending[target_id]) == attrs_by_id[target_id].get("num_inputs", 0):
                    wire_ready.add(target_id)
                    _try_enqueue(target_id)

    return visit_order


def _build_circuit_repr(diagram_name: str, access: _GraphAccess, graph_mod: object) -> "CircuitRepr":
    """Build the mqc3 `CircuitRepr` for `access`'s graph via a direct forward sweep.

    Returns
    -------
    CircuitRepr
    """
    from mqc3.circuit import CircuitRepr  # ruff: ignore[import-outside-top-level]

    circuit = CircuitRepr(diagram_name)
    measurement_ops: MeasurementOps = {}

    for node_id, modes in _mode_ordered_leaves(access):
        leaf = access.reconstruct(node_id, graph_mod)

        if len(modes) == 1 and access.attrs(node_id).get("num_inputs", 0) == 0:
            if isinstance(leaf, VoidDiagram):
                continue
            (mode_id,) = modes
            circuit.Q(mode_id)
            circuit.set_initial_state(mode_id, _open_mode_state(leaf, mode_id))
            continue

        if len(modes) == 2:  # ruff: ignore[magic-value-comparison]
            _apply_2mode_leaf(circuit, modes[0], modes[1], leaf, measurement_ops)
            continue

        (mode_id,) = modes
        _apply_1mode_leaf(circuit, mode_id, leaf, measurement_ops)

    return circuit


def extract_dependency_dag(
    diagram: Diagram,
    *,
    backend: Backend | str | None = None,
    complete: bool = True,
    input_basis: str | list[str] = "Q",
    output_basis: str | list[str] = "Q",
) -> "DependencyDAG":
    """Extract an mqc3 `DependencyDAG` directly from `diagram`'s `CVZXGraph`.

    See the module docstring for the full algorithm. This is an
    alternative to `cvzx.lowering.Mqc3ReferenceBackend` (registered as
    `"cvzx-direct"` in `cvzx.lowering.lowering`'s backend registry) that discovers
    execution order from the graph's own wire/classical edges instead of
    `normalize_diagram`'s canonical stage form, so it tolerates diagram
    shapes that form isn't required to (e.g. an explicit `Swap`, or a
    `ContractedDiagram` `complete_boundaries()`/the translators below can
    still resolve).

    Parameters
    ----------
    diagram : Diagram
        The diagram to lower. Not mutated.
    backend : Backend | str | None
        Which `CVZXGraph` backend to walk with -- see `cvzx.config.Backend`.
        `None` (default) uses `cvzx.config.DEFAULT_BACKEND`.
    complete : bool
        If True (default), run `complete_boundaries()` first so every
        wire is bounded. If False, `diagram` must already have
        `num_inputs == num_outputs == 0` (its own `CVZXGraph`'s
        `GateRegister.input_states`/`measurement_nodes` must already
        bound every wire) -- set this only when that's already guaranteed
        upstream, to skip the (cheap, but non-zero) completion pass.

    Returns
    -------
    DependencyDAG
        The resulting dependency DAG, ready for mqc3's own
        `GraphEmbedder` machinery to embed into a concrete `GraphRepr`.

    Raises
    ------
    ValueError
        If `complete=False` and `diagram` still has open ports.
    """
    from mqc3.graph.embed.dep_dag import DependencyDAG  # ruff: ignore[import-outside-top-level]

    if complete:
        result = complete_boundaries(diagram, backend=backend, input_basis=input_basis, output_basis=output_basis)
        cvzx_graph = result.graph
        chosen, graph_mod, _ = get_backend_modules(backend)
    else:
        if diagram.num_inputs != 0 or diagram.num_outputs != 0:
            msg = (
                f"extract_dependency_dag(complete=False) requires a fully closed diagram "
                f"(num_inputs={diagram.num_inputs}, num_outputs={diagram.num_outputs}); "
                "pass complete=True (the default) or close it yourself first."
            )
            raise ValueError(msg)
        chosen, graph_mod, _ = get_backend_modules(backend)
        cvzx_graph = graph_mod.to_graph(diagram)
        cvzx_graph.rebuild_registry()

    access = _GraphAccess(cvzx_graph, is_rx=chosen == Backend.RUSTWORKX)
    circuit = _build_circuit_repr("extracted", access, graph_mod)
    return DependencyDAG(circuit)

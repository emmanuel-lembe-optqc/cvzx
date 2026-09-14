"""Graph extraction utilities for CV ZX diagrams using rustworkx.

This module provides high-performance functions to convert CV ZX diagrams
to rustworkx directed graphs (PyDiGraph) for large-scale graph optimization.

Key design decisions:
    - Uses rustworkx.PyDiGraph for 10x-100x traversal and embedding speedups
    - Each node payload is a dictionary storing metadata, type, and original diagram ID
    - node_map dictionary maintains O(1) bi-directional mapping between original
      diagram.id and rustworkx node indices (0, 1, 2...)
    - Containers (CompositionDiagram, TensorDiagram, ContractedDiagram) are
      preserved as nodes with kind='container'
    - Proper diagrams are nodes with kind='proper'
    - The hierarchy is preserved through sub_diagram_ids
    - The root container is identified by is_root=True
"""

from copy import deepcopy
from typing import cast

import rustworkx as rx
from sympy import Expr, Symbol

from cvzx.exceptions import ParameterConflictError, UnboundMeasurementError
from cvzx.ir.base import (
    CompositionDiagram,
    ContractedDiagram,
    Diagram,
    Fourier,
    Fourier2,
    FourierInv,
    ProperDiagram,
    PSpider,
    QSpider,
    Swap,
    TensorDiagram,
    VoidDiagram,
    ZxPoly,
)
from cvzx.ir.gates import (
    ArbitraryGate,
    BeamsplitterGate,
    CompactDiagram,
    ControlledSumGate,
    ControlledZGate,
    CubicPhaseGate,
    DisplacementGate,
    MeasurementGate,
    PhaseRotationGate,
    ShearPInvariantGate,
    ShearXInvariantGate,
    Squeezing45Gate,
    SqueezingGate,
    TwoModeShearGate,
)


def _symbols_in(value: object) -> set[Symbol]:
    """Extract the free symbols carried by a single node-attribute value.

    Handles `ZxPoly` specially: it deliberately reads `.coeffs` and pulls
    `free_symbols` off each symbolic coefficient, rather than using
    `ZxPoly.free_symbols` (inherited from `sympy.Poly`), since the latter
    incorrectly includes the polynomial's own generator variable.

    Parameters
    ----------
    value : object
        A node-attribute value: a `ZxPoly`, a plain sympy `Expr`, or
        anything else (numeric, `None`, ...), which contributes no symbols.

    Returns
    -------
    set[Symbol]
    """
    if isinstance(value, ZxPoly):
        return {symbol for coef in value.coeffs.values() if isinstance(coef, Expr) for symbol in coef.free_symbols}
    if isinstance(value, Expr):
        return cast("set[Symbol]", value.free_symbols)
    return set()


def _collect_symbols(attrs: dict) -> set[Symbol]:
    """Union the free symbols across a node payload's parametric-carrying keys.

    Looks at every key `add_node`/`build_from_graph` may populate with a
    symbolic value (`phase`, and the multi-parameter gate keys `alpha`,
    `beta`, `lam`, `a`, `b`), plus the keys of `param_measurement_map` if
    present on the payload.

    Parameters
    ----------
    attrs : dict
        The node's attribute payload.

    Returns
    -------
    set[Symbol]
    """
    symbols: set[Symbol] = set()
    for key in ("phase", "alpha", "beta", "lam", "a", "b"):
        if key in attrs:
            symbols |= _symbols_in(attrs[key])
    symbols |= set(attrs.get("param_measurement_map", {}))
    return symbols


class GateRegister:
    """Registry for tracking specific gate types and nodes in a CV ZX graph.

    The GateRegister maintains sets of node IDs (original diagram IDs) for
    different gate types, enabling O(1) lookups instead of scanning the graph.

    Parameters
    ----------
        squeezing_gates set[int]: Node IDs of squeezing gates (Sq)
        displacement_gates set[int]: Node IDs of displacement gates (D)
        rotation_gates set[int]: Node IDs of phase rotation gates (R)
        fourier_gates set[int]: Node IDs of Fourier gates (F, F†, F²)
        identity_spiders set[int]: Node IDs of identity spiders (zero phase, 1→1)
        input_states set[int]: Node IDs of input states (0→1)
        measurement_nodes set[int]: Node IDs of measurements (1→0)
        contracted_diagrams set[int]: Node IDs of ContractedDiagram containers
        tensor_nodes set[int]: Node IDs of TensorDiagram containers
        composition_nodes set[int]: Node IDs of CompositionDiagram containers
        void_nodes set[int]: Node IDs of VoidDiagram placeholders
        parametric_nodes set[int]: Node IDs carrying at least one free symbol
        feedforward_nodes set[int]: Node IDs with feedforward=True
        symbol_registry dict[Symbol, set[int]]: Symbol -> node IDs using it
        measurement_to_feedforward_map dict[int, set[int]]: Measurement node ID ->
            node IDs whose measurement_ids includes it
    """

    def __init__(self) -> None:
        """Initialize an empty GateRegister with all sets empty."""
        self.squeezing_gates: set[int] = set()
        self.displacement_gates: set[int] = set()
        self.rotation_gates: set[int] = set()
        self.fourier_gates: set[int] = set()
        self.identity_spiders: set[int] = set()
        self.input_states: set[int] = set()
        self.measurement_nodes: set[int] = set()
        self.contracted_diagrams: set[int] = set()
        self.tensor_nodes: set[int] = set()
        self.composition_nodes: set[int] = set()
        self.void_nodes: set[int] = set()
        self.parametric_nodes: set[int] = set()
        self.feedforward_nodes: set[int] = set()
        self.symbol_registry: dict[Symbol, set[int]] = {}
        self.measurement_to_feedforward_map: dict[int, set[int]] = {}
        self._indexed_symbols: dict[int, set[Symbol]] = {}
        self._indexed_measurements: dict[int, set[int]] = {}

    def add_node(self, node_id: int, attrs: dict) -> None:  # ruff: ignore[complex-structure, too-many-branches]
        """Add a node to the appropriate sets based on its attributes.

        Parameters
        ----------
        node_id : int
            The original diagram ID of the node to add.
        attrs : dict
            The node's attributes from the graph payload.
        """
        gate_type = attrs.get("type")
        kind = attrs.get("kind")
        container_type = attrs.get("container_type")

        if kind not in {"proper", "container", "compact"}:
            return

        if kind == "container":
            if container_type == "tensor":
                self.tensor_nodes.add(node_id)
            elif container_type == "composition":
                self.composition_nodes.add(node_id)
            elif container_type == "contracted":
                self.contracted_diagrams.add(node_id)
            return

        if gate_type == "SqueezingGate":
            self.squeezing_gates.add(node_id)
        elif gate_type == "DisplacementGate":
            self.displacement_gates.add(node_id)
        elif gate_type == "PhaseRotationGate":
            self.rotation_gates.add(node_id)
        elif gate_type in {"Fourier", "FourierInv", "Fourier2"}:
            self.fourier_gates.add(node_id)

        if self._is_identity_spider(attrs):
            self.identity_spiders.add(node_id)

        if gate_type == "VoidDiagram":
            self.void_nodes.add(node_id)

        if self._is_input_state(attrs):
            self.input_states.add(node_id)
        elif self._is_measurement(attrs):
            self.measurement_nodes.add(node_id)

        self._index_parameters(node_id, attrs)

    def remove_node(self, node_id: int) -> None:
        """Remove a node ID from all sets in the registry.

        Parameters
        ----------
        node_id : int
            The ID of the node to remove.
        """
        self.squeezing_gates.discard(node_id)
        self.displacement_gates.discard(node_id)
        self.rotation_gates.discard(node_id)
        self.fourier_gates.discard(node_id)
        self.identity_spiders.discard(node_id)
        self.input_states.discard(node_id)
        self.measurement_nodes.discard(node_id)
        self.contracted_diagrams.discard(node_id)
        self.tensor_nodes.discard(node_id)
        self.composition_nodes.discard(node_id)
        self.void_nodes.discard(node_id)
        self._retract_parameter_index(node_id)

    def copy(self) -> "GateRegister":
        """Create a shallow copy of the register.

        Returns
        -------
        GateRegister
            A new GateRegister with copied sets.
        """
        new_reg = GateRegister()
        new_reg.squeezing_gates = self.squeezing_gates.copy()
        new_reg.displacement_gates = self.displacement_gates.copy()
        new_reg.rotation_gates = self.rotation_gates.copy()
        new_reg.fourier_gates = self.fourier_gates.copy()
        new_reg.identity_spiders = self.identity_spiders.copy()
        new_reg.input_states = self.input_states.copy()
        new_reg.measurement_nodes = self.measurement_nodes.copy()
        new_reg.contracted_diagrams = self.contracted_diagrams.copy()
        new_reg.tensor_nodes = self.tensor_nodes.copy()
        new_reg.composition_nodes = self.composition_nodes.copy()
        new_reg.void_nodes = self.void_nodes.copy()
        new_reg.parametric_nodes = self.parametric_nodes.copy()
        new_reg.feedforward_nodes = self.feedforward_nodes.copy()
        new_reg.symbol_registry = {symbol: nodes.copy() for symbol, nodes in self.symbol_registry.items()}
        new_reg.measurement_to_feedforward_map = {
            measurement_id: nodes.copy() for measurement_id, nodes in self.measurement_to_feedforward_map.items()
        }
        new_reg._indexed_symbols = {node_id: symbols.copy() for node_id, symbols in self._indexed_symbols.items()}
        new_reg._indexed_measurements = {node_id: ids.copy() for node_id, ids in self._indexed_measurements.items()}
        return new_reg

    def clear(self) -> None:
        """Clear all sets in the registry."""
        self.squeezing_gates.clear()
        self.displacement_gates.clear()
        self.rotation_gates.clear()
        self.fourier_gates.clear()
        self.identity_spiders.clear()
        self.input_states.clear()
        self.measurement_nodes.clear()
        self.contracted_diagrams.clear()
        self.tensor_nodes.clear()
        self.composition_nodes.clear()
        self.void_nodes.clear()
        self.parametric_nodes.clear()
        self.feedforward_nodes.clear()
        self.symbol_registry.clear()
        self.measurement_to_feedforward_map.clear()
        self._indexed_symbols.clear()
        self._indexed_measurements.clear()

    def build_from_graph(self, graph: rx.PyDiGraph) -> None:
        """Build the entire registry from a rustworkx PyDiGraph.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to rebuild the registry from.
        """
        self.clear()
        for idx in graph.node_indices():
            attrs = graph[idx]
            node_id = attrs.get("id", idx)
            self.add_node(node_id, attrs)

    def _is_identity_spider(self, attrs: dict) -> bool:
        """Check if node attributes represent an identity spider.

        Returns
        -------
        bool
        """
        node_type = attrs.get("type")
        if node_type not in {"QSpider", "PSpider"}:
            return False

        if attrs.get("num_inputs", 0) != 1 or attrs.get("num_outputs", 0) != 1:
            return False

        phase = attrs.get("phase")
        return bool(phase is not None and phase.is_zero)

    def _is_input_state(self, attrs: dict) -> bool:
        """Check if a node is an input state.

        Returns
        -------
        bool
        """
        return attrs.get("num_inputs") == 0 and attrs.get("num_outputs") == 1

    def _is_measurement(self, attrs: dict) -> bool:
        """Check if a node is a measurement.

        Returns
        -------
        bool
        """
        return attrs.get("num_inputs") == 1 and attrs.get("num_outputs") == 0

    def _index_parameters(self, node_id: int, attrs: dict) -> None:
        """Index a node's symbolic parameters and feedforward provenance.

        Populates `parametric_nodes`/`symbol_registry` from the free
        symbols found in `attrs` (see `_collect_symbols`), and
        `feedforward_nodes`/`measurement_to_feedforward_map` from
        `attrs["feedforward"]`/`attrs["measurement_ids"]`. Also records
        what was indexed in `_indexed_symbols`/`_indexed_measurements` so
        `_retract_parameter_index` can undo it later without re-deriving
        it from (possibly already-gone) attrs.

        Parameters
        ----------
        node_id : int
            The ID of the node being added.
        attrs : dict
            The node's attributes.
        """
        symbols = _collect_symbols(attrs)
        if symbols:
            self.parametric_nodes.add(node_id)
            self._indexed_symbols[node_id] = symbols
            for symbol in symbols:
                self.symbol_registry.setdefault(symbol, set()).add(node_id)

        if attrs.get("feedforward"):
            self.feedforward_nodes.add(node_id)
            measurement_ids = set(attrs.get("measurement_ids") or ())
            self._indexed_measurements[node_id] = measurement_ids
            for measurement_id in measurement_ids:
                self.measurement_to_feedforward_map.setdefault(measurement_id, set()).add(node_id)

    def _retract_parameter_index(self, node_id: int) -> None:
        """Undo `_index_parameters` for a node being removed.

        Uses the cached sets recorded in `_indexed_symbols`/
        `_indexed_measurements` at index time (not attrs re-derivation),
        since the node may already be gone from the graph by the time
        this runs. When a symbol's/measurement's node set becomes empty,
        the key itself is deleted (not left as an empty set), so
        `symbol_registry`/`measurement_to_feedforward_map` stay an exact
        "what's still live" index.

        Parameters
        ----------
        node_id : int
            The ID of the node being removed.
        """
        symbols = self._indexed_symbols.pop(node_id, None)
        if symbols is not None:
            for symbol in symbols:
                nodes = self.symbol_registry.get(symbol)
                if nodes is not None:
                    nodes.discard(node_id)
                    if not nodes:
                        del self.symbol_registry[symbol]
            self.parametric_nodes.discard(node_id)

        measurement_ids = self._indexed_measurements.pop(node_id, None)
        if measurement_ids is not None:
            for measurement_id in measurement_ids:
                nodes = self.measurement_to_feedforward_map.get(measurement_id)
                if nodes is not None:
                    nodes.discard(node_id)
                    if not nodes:
                        del self.measurement_to_feedforward_map[measurement_id]
            self.feedforward_nodes.discard(node_id)


class CVZXGraph:
    """A CV ZX diagram represented as a rustworkx PyDiGraph, paired with its registry.

    Parameters
    ----------
    graph : rx.PyDiGraph
        The rustworkx graph representation of a CV ZX diagram.
    registry : GateRegister | None, optional
        A registry already built from ``graph``. If ``None``, a new registry
        is built from ``graph``.

    Attributes
    ----------
    graph : rx.PyDiGraph
        The underlying rustworkx directed graph.
    registry : GateRegister
        The registry indexing ``graph``.
    """

    def __init__(self, graph: rx.PyDiGraph, registry: "GateRegister | None" = None) -> None:
        self.graph = graph
        if registry is None:
            registry = GateRegister()
            registry.build_from_graph(graph)
        self.registry = registry

    @classmethod
    def from_diagram(cls, diagram: Diagram) -> "CVZXGraph":
        """Build a ``CVZXGraph`` from a CV ZX diagram.

        Parameters
        ----------
        diagram : Diagram
            The diagram to convert.

        Returns
        -------
        CVZXGraph
            The rustworkx graph representation of ``diagram``.
        """
        return to_graph(diagram)

    def to_diagram(self) -> Diagram:
        """Reconstruct the CV ZX diagram from the current graph.

        Returns
        -------
        Diagram
            The reconstructed diagram.
        """
        return to_diagram(self)

    def rebuild_registry(self) -> None:
        """Rebuild ``self.registry`` from the current state of ``self.graph``."""
        self.registry.build_from_graph(self.graph)

    def parameter_consistency_violations(self) -> list[str]:
        """Find symbolic-parameter/feedforward inconsistencies, without raising.

        Returns
        -------
        list[str]
            One message per violation found; empty if the graph is consistent.
        """
        return [*self._symbol_conflict_violations(), *self._unbound_measurement_violations()]

    def _symbol_conflict_violations(self) -> list[str]:
        """Find symbols bound to conflicting measurement-id sets across nodes.

        Only nodes that actually *declare* a binding for the symbol (a
        non-empty `param_measurement_map[symbol]`) participate in the
        comparison -- a node that merely carries the symbol in its own
        parameters without binding it (e.g. the measurement leaf that
        *originates* a feedforward symbol, or any other node with a
        legitimately unbound symbolic parameter) has nothing to agree or
        disagree with, so it's excluded rather than treated as an
        implicit "bound to nothing" conflict against every real binding.

        Returns
        -------
        list[str]
        """
        violations: list[str] = []
        id_map = {self.graph[idx]["id"]: idx for idx in self.graph.node_indices()}
        for symbol, node_ids in self.registry.symbol_registry.items():
            bindings = {
                node_id: frozenset(binding)
                for node_id in node_ids
                if (binding := self.graph[id_map[node_id]].get("param_measurement_map", {}).get(symbol))
            }
            if len(set(bindings.values())) > 1:
                violations.append(
                    f"Symbol {symbol!r} is bound to conflicting measurement sets across nodes "
                    f"{sorted(bindings)}: {bindings}."
                )
        return violations

    def _unbound_measurement_violations(self) -> list[str]:
        """Find measurement IDs referenced by no registered measurement node.

        Returns
        -------
        list[str]
        """
        return [
            f"measurement_to_feedforward_map references measurement id {measurement_id}, "
            "which is not a registered measurement node."
            for measurement_id in self.registry.measurement_to_feedforward_map
            if measurement_id not in self.registry.measurement_nodes
        ]

    def validate_parameter_consistency(self) -> None:
        """Raise if `parameter_consistency_violations()` finds anything.

        Symbol conflicts are checked (and raised) first, since an
        unresolved conflict makes any measurement-existence finding
        downstream of it suspect too.

        Raises
        ------
        ParameterConflictError
            Listing every symbol bound to conflicting measurement-id sets.
        UnboundMeasurementError
            Listing every measurement id referenced by
            `measurement_to_feedforward_map` that isn't a registered
            measurement node.
        """
        conflicts = self._symbol_conflict_violations()
        if conflicts:
            msg = "Parameter consistency violations found:\n" + "\n".join(conflicts)
            raise ParameterConflictError(msg)
        unbound = self._unbound_measurement_violations()
        if unbound:
            msg = "Parameter consistency violations found:\n" + "\n".join(unbound)
            raise UnboundMeasurementError(msg)

    def copy(self) -> "CVZXGraph":
        """Return an independent copy of this ``CVZXGraph``.

        Returns
        -------
        CVZXGraph
            An independent copy of this graph and its registry.
        """
        return CVZXGraph(self.graph.copy(), self.registry.copy())

    def _root_attrs(self) -> dict:
        root_id = get_root_node(self)
        if root_id is None:
            msg = "No root node found in the graph."
            raise ValueError(msg)
        for idx in self.graph.node_indices():
            if self.graph[idx].get("id") == root_id:
                return cast("dict", self.graph[idx])
        msg = "Root node payload missing from graph."
        raise ValueError(msg)

    @property
    def root_id(self) -> int | None:
        """The node ID of the root container, or ``None`` if there is none."""
        return get_root_node(self)

    @property
    def num_inputs(self) -> int:
        """The number of external inputs of the diagram."""
        return cast("int", self._root_attrs()["num_inputs"])

    @property
    def num_outputs(self) -> int:
        """The number of external outputs of the diagram."""
        return cast("int", self._root_attrs()["num_outputs"])

    def __len__(self) -> int:
        """Return the number of nodes in the graph.

        Returns
        -------
        int
        """
        return len(self.graph)

    def __repr__(self) -> str:
        """Return a debug representation of this ``CVZXGraph``.

        Returns
        -------
        str
        """
        return f"CVZXGraph(nodes={len(self.graph)}, edges={len(self.graph.edge_list())})"

    def __eq__(self, other: object) -> bool:
        """Check equality by comparing the reconstructed diagrams.

        Two ``CVZXGraph`` instances are equal if the diagrams they
        reconstruct via :meth:`to_diagram` are equal. A graph that fails to
        reconstruct (for example, because it has no root node) is never
        equal to anything.

        Parameters
        ----------
        other : object
            The object to compare against.

        Returns
        -------
        bool
        """
        if not isinstance(other, CVZXGraph):
            return NotImplemented
        try:
            self_diagram = self.to_diagram()
            other_diagram = other.to_diagram()
        except ValueError:
            return False
        return bool(self_diagram == other_diagram)

    __hash__ = None  # type: ignore[assignment]


def to_graph(diagram: Diagram) -> CVZXGraph:
    """Convert a CV ZX diagram to a rustworkx graph representation.

    Parameters
    ----------
    diagram : Diagram
        The CV ZX diagram to convert.

    Returns
    -------
    CVZXGraph
        Graph representation backed by rustworkx PyDiGraph.
    """
    graph = rx.PyDiGraph(multigraph=False)
    node_map: dict[int, int] = {}

    root_id = _convert_diagram_to_graph(deepcopy(diagram), graph, node_map, container_id=None, is_root=True)

    if root_id is not None and root_id != -1 and root_id in node_map:
        root_idx = node_map[root_id]
        graph[root_idx]["is_root"] = True

    for idx in graph.node_indices():
        attrs = graph[idx]
        if "diagram" in attrs:
            del attrs["diagram"]

    return CVZXGraph(graph)


def to_diagram(cvzx_graph: CVZXGraph) -> Diagram:
    """Reconstruct a CV ZX diagram from a rustworkx graph representation.

    Parameters
    ----------
    cvzx_graph : CVZXGraph
        The graph representation of the diagram.

    Returns
    -------
    Diagram
        The reconstructed CV ZX diagram.

    Raises
    ------
    ValueError
        If the graph is malformed or missing required attributes.
    """
    graph = cvzx_graph.graph

    root = get_root_node(cvzx_graph)
    if root is None:
        msg = "No root node found in the graph."
        raise ValueError(msg)

    id_to_idx = {graph[idx]["id"]: idx for idx in graph.node_indices()}
    return reconstruct_from_node(graph, root, cvzx_graph.registry, id_to_idx)


def find_node_by_external_output(
    diagram: Diagram,
    ext_port: int,
    G: rx.PyDiGraph,  # ruff: ignore[invalid-argument-name]
    node_map: dict[int, int],
) -> tuple[int | None, int | None]:
    """Find the proper node and its internal port for a given external output port.

    This function recursively traverses the diagram structure:
        1. If diagram is proper: returns (node_id, ext_port) (direct mapping)
        2. If diagram is a container: uses its output_mapping to find the sub-diagram
           and internal port, then recurses into that sub-diagram

    Parameters
    ----------
    diagram : Diagram
        The diagram to search within.
    ext_port : int
        The external output port index.
    G : nx.DiGraph
        The graph containing the nodes.

    Returns
    -------
    tuple[int | None, int | None]
        (node_id, internal_port) of the proper node handling this external port,
        or (None, None) if not found.
    """
    """Find proper node ID and internal port for an external output port."""
    if isinstance(diagram, (ProperDiagram, CompactDiagram)):
        return diagram.id, ext_port

    node_idx = node_map.get(diagram.id)
    if node_idx is None:
        return None, None

    attrs = G[node_idx]
    output_mapping = attrs.get("external_output_mapping", {})

    if ext_port not in output_mapping:
        return None, None

    sub_ref, internal_port = output_mapping[ext_port]

    if attrs.get("container_type") in {"tensor", "composition"}:
        sub_node_id = attrs["sub_diagram_ids"][sub_ref]
        sub_node_idx = node_map[sub_node_id]
        sub_diagram = G[sub_node_idx]["diagram"]
    elif attrs.get("container_type") == "contracted" and isinstance(diagram, ContractedDiagram):
        sub_diagram = diagram.first if sub_ref == "first" else diagram.second
    else:
        return None, None

    return find_node_by_external_output(sub_diagram, internal_port, G, node_map)


def find_node_by_external_input(
    diagram: Diagram,
    ext_port: int,
    G: rx.PyDiGraph,  # ruff: ignore[invalid-argument-name]
    node_map: dict[int, int],
) -> tuple[int | None, int | None]:
    """Find the proper node and its internal port for a given external input port.

    This function recursively traverses the diagram structure:
        1. If diagram is proper: returns (node_id, ext_port) (direct mapping)
        2. If diagram is a container: uses its input_mapping to find the sub-diagram
           and internal port, then recurses into that sub-diagram

    Parameters
    ----------
    diagram : Diagram
        The diagram to search within.
    ext_port : int
        The external input port index.
    G : nx.DiGraph
        The graph containing the nodes.

    Returns
    -------
    tuple[int | None, int | None]
        (node_id, internal_port) of the proper node handling this external input port,
        or (None, None) if not found.
    """
    if isinstance(diagram, (ProperDiagram, CompactDiagram)):
        return diagram.id, ext_port

    node_idx = node_map.get(diagram.id)
    if node_idx is None:
        return None, None

    attrs = G[node_idx]
    input_mapping = attrs.get("external_input_mapping", {})

    if ext_port not in input_mapping:
        return None, None

    sub_ref, internal_port = input_mapping[ext_port]

    if attrs.get("container_type") in {"tensor", "composition"}:
        sub_node_id = attrs["sub_diagram_ids"][sub_ref]
        sub_node_idx = node_map[sub_node_id]
        sub_diagram = G[sub_node_idx]["diagram"]
    elif attrs.get("container_type") == "contracted" and isinstance(diagram, ContractedDiagram):
        sub_diagram = diagram.first if sub_ref == "first" else diagram.second
    else:
        return None, None

    return find_node_by_external_input(sub_diagram, internal_port, G, node_map)


def get_proper_nodes(cvzx_graph: CVZXGraph) -> list[int]:
    """Get all proper nodes (leaf operations) from the graph.

    Proper nodes represent actual operations (spiders, gates, etc.)
    and have kind='proper'.

    Parameters
    ----------
    cvzx_graph : CVZXGraph
        The graph to search for proper nodes.

    Returns
    -------
    list[int]
        List of node IDs for all proper nodes in the graph.
    """
    graph = cvzx_graph.graph
    return [graph[idx]["id"] for idx in graph.node_indices() if graph[idx].get("kind") in {"proper", "compact"}]


def get_container_nodes(cvzx_graph: CVZXGraph) -> list[int]:
    """Get all container nodes from the graph.

    Container nodes represent structural elements (CompositionDiagram,
    TensorDiagram, ContractedDiagram) and have kind='container'.

    Parameters
    ----------
    cvzx_graph : CVZXGraph
        The graph to search for container nodes.

    Returns
    -------
    list[int]
        List of node IDs for all container nodes in the graph.
    """
    graph = cvzx_graph.graph
    return [graph[idx]["id"] for idx in graph.node_indices() if graph[idx].get("kind") == "container"]


def get_root_node(cvzx_graph: CVZXGraph) -> int | None:
    """Get the root node (the outermost diagram) from the graph.

    The root node is the container node with is_root=True.

    Parameters
    ----------
    cvzx_graph : CVZXGraph
        The graph to search for the root node.

    Returns
    -------
    int | None
        The node ID of the root container, or None if no root is found.
    """
    graph = cvzx_graph.graph
    for idx in graph.node_indices():
        if graph[idx].get("is_root", False):
            return cast("int", graph[idx]["id"])
    return None


def get_immediate_container(cvzx_graph: CVZXGraph, node_id: int) -> int | None:
    """Get the immediate container of a node.

    Parameters
    ----------
    cvzx_graph : CVZXGraph
        The graph containing the node.
    node_id : int
        The node ID to find the container for.

    Returns
    -------
    int | None
        The node ID of the immediate container, or None if the node is the root.
    """
    graph = cvzx_graph.graph
    for idx in graph.node_indices():
        if graph[idx].get("id") == node_id:
            return cast("int | None", graph[idx].get("container_id"))
    return None


def get_sub_diagrams(cvzx_graph: CVZXGraph, container_node: int) -> list[int]:
    """Get the sub-diagram node IDs of a container node.

    Parameters
    ----------
    cvzx_graph : CVZXGraph
        The graph containing the container node.
    container_node : int
        The node ID of the container node.

    Returns
    -------
    list[int]
        List of sub-diagram node IDs, or an empty list if the node is not
        a container or has no sub-diagrams.
    """
    graph = cvzx_graph.graph
    for idx in graph.node_indices():
        if graph[idx].get("id") == container_node:
            attrs = graph[idx]
            if attrs.get("kind") != "container":
                return []
            return cast("list[int]", attrs.get("sub_diagram_ids", []))
    return []


def get_connectivity(cvzx_graph: CVZXGraph, container_node: int) -> dict[int, dict[int, int]] | None:
    """Get the connectivity dictionary of a CompositionDiagram container node.

    Parameters
    ----------
    cvzx_graph : CVZXGraph
        The graph containing the container node.
    container_node : int
        The node ID of the composition container node.

    Returns
    -------
    dict[int, dict[int, int]] | None
        The connectivity dictionary if the node is a CompositionDiagram,
        otherwise None.
    """
    graph = cvzx_graph.graph
    for idx in graph.node_indices():
        if graph[idx].get("id") == container_node:
            attrs = graph[idx]
            if attrs.get("container_type") != "composition":
                return None
            return cast("dict[int, dict[int, int]] | None", attrs.get("connectivity"))
    return None


def get_contracted_connections(cvzx_graph: CVZXGraph, container_node: int) -> dict | None:
    """Get the contracted connections of a ContractedDiagram container node.

    Parameters
    ----------
    cvzx_graph : CVZXGraph
        The graph containing the container node.
    container_node : int
        The node ID of the contracted container node.

    Returns
    -------
    dict | None
        A dictionary with keys 'I1', 'I2', 'J1', 'J2' if the node is a
        ContractedDiagram, otherwise None.
    """
    graph = cvzx_graph.graph
    for idx in graph.node_indices():
        if graph[idx].get("id") == container_node:
            attrs = graph[idx]
            if attrs.get("container_type") != "contracted":
                return None
            return {
                "I1": attrs.get("I1", []),
                "I2": attrs.get("I2", []),
                "J1": attrs.get("J1", []),
                "J2": attrs.get("J2", []),
            }
    return None


def get_nodes_by_container(cvzx_graph: CVZXGraph, container_node: int) -> list[int]:
    """Get all nodes that belong to a specific container.

    This includes both proper nodes and nested container nodes.

    Parameters
    ----------
    cvzx_graph : CVZXGraph
        The graph containing the nodes.
    container_node : int
        The node ID of the container.

    Returns
    -------
    list[int]
        List of node IDs belonging to the container.
    """
    graph = cvzx_graph.graph
    return [graph[idx]["id"] for idx in graph.node_indices() if graph[idx].get("container_id") == container_node]


def _convert_diagram_to_graph(
    diagram: Diagram,
    G: rx.PyDiGraph,  # ruff: ignore[invalid-argument-name]
    node_map: dict[int, int],
    container_id: int | None,
    is_root: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
) -> int:
    """Convert a diagram to graph nodes and return the root node ID.

    This function recursively converts a diagram and its sub-diagrams
    to graph nodes. It returns the node ID of the root diagram.

    Parameters
    ----------
    diagram : Diagram
        The diagram to convert.
    G : nx.DiGraph
        The graph to add nodes to (modified in place).
    container_id : int | None
        The node ID of the immediate container of this diagram.
        None if this is the root diagram.
    is_root : bool, default=False
        Whether this diagram is the root (outermost) diagram.

    Returns
    -------
    int
        The node ID of the root diagram in the graph, or -1 if the diagram
        has no representation.
    """
    if isinstance(diagram, (ProperDiagram, CompactDiagram)):
        return _add_proper_node(diagram, G, node_map, container_id)
    if isinstance(diagram, CompositionDiagram):
        return _add_composition_node(diagram, G, node_map, container_id, is_root)
    if isinstance(diagram, TensorDiagram):
        return _add_tensor_node(diagram, G, node_map, container_id, is_root)
    if isinstance(diagram, ContractedDiagram):
        return _add_contracted_node(diagram, G, node_map, container_id, is_root)
    return -1


def _add_proper_node(  # ruff: ignore[complex-structure]
    diagram: ProperDiagram | CompactDiagram,
    G: rx.PyDiGraph,  # ruff: ignore[invalid-argument-name]
    node_map: dict[int, int],
    container_id: int | None,
) -> int:
    """Add a proper diagram as a node in the graph.

    Proper nodes have kind='proper' and store all relevant attributes
    including type, phase, port information, and its container ID.

    Parameters
    ----------
    diagram : ProperDiagram | CompactDiagram
        The proper diagram to add as a node.
    G : nx.DiGraph
        The graph to add the node to (modified in place).
    node_map : dict[int, int]
        Mapping from cvzx diagram id to rustworkx node index (modified in place).
    container_id : int | None
        The node ID of the immediate container of this proper diagram.

    Returns
    -------
    int
        The node ID of the added proper node.
    """
    node_id = diagram.id
    phase = getattr(diagram, "phase", None)
    node_type = diagram.__class__.__name__
    kind = "proper" if isinstance(diagram, ProperDiagram) else "compact"

    feedforward = getattr(diagram, "feedforward", None)
    measurement_ids = getattr(diagram, "measurement_ids", None)
    param_measurement_map = getattr(diagram, "param_measurement_map", {})

    if isinstance(diagram, PhaseRotationGate):
        phase = getattr(diagram, "theta", None)
    elif isinstance(diagram, SqueezingGate):
        phase = getattr(diagram, "tau", None)
    elif isinstance(diagram, BeamsplitterGate):
        phase = getattr(diagram, "theta", None)
    elif isinstance(diagram, (ControlledSumGate, ControlledZGate)):
        phase = getattr(diagram, "gain", None)
    elif isinstance(diagram, CubicPhaseGate):
        phase = getattr(diagram, "gamma", None)
    elif isinstance(diagram, ShearXInvariantGate):
        phase = getattr(diagram, "kappa", None)
    elif isinstance(diagram, ShearPInvariantGate):
        phase = getattr(diagram, "eta", None)
    elif isinstance(diagram, (MeasurementGate, Squeezing45Gate)):
        phase = getattr(diagram, "theta", None)

    attrs = {
        "id": node_id,
        "type": node_type,
        "kind": kind,
        "phase": phase,
        "feedforward": feedforward,
        "measurement_ids": measurement_ids,
        "param_measurement_map": param_measurement_map,
        "num_inputs": diagram.num_inputs,
        "num_outputs": diagram.num_outputs,
        "diagram": diagram,
        "container_id": container_id,
        "external_inputs": list(range(diagram.num_inputs)),
        "external_outputs": list(range(diagram.num_outputs)),
    }

    if isinstance(diagram, DisplacementGate):
        attrs["phase"] = getattr(diagram, "alpha", None)
    elif isinstance(diagram, ControlledSumGate):
        attrs["control"] = diagram.control
        attrs["target"] = diagram.target
    elif isinstance(diagram, ArbitraryGate):
        attrs.update({
            "alpha": diagram.alpha,
            "beta": diagram.beta,
            "lam": diagram.lam,
        })
    elif isinstance(diagram, TwoModeShearGate):
        attrs.update({
            "a": diagram.a,
            "b": diagram.b,
        })

    node_idx = G.add_node(attrs)
    node_map[node_id] = node_idx
    return node_id


def _add_composition_node(  # ruff: ignore[too-many-locals]
    diagram: CompositionDiagram,
    G: rx.PyDiGraph,  # ruff: ignore[invalid-argument-name]
    node_map: dict[int, int],
    container_id: int | None,
    is_root: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
) -> int:
    """Add a CompositionDiagram as a container node in the graph.

    Parameters
    ----------
    diagram : CompositionDiagram
        The composition diagram to add as a container node.
    G : nx.DiGraph
        The graph to add nodes and edges to (modified in place).
    container_id : int | None
        The node ID of the immediate container of this composition.
        None if this composition is the root diagram.
    is_root : bool, default=False
        Whether this is the root (outermost) diagram.

    Returns
    -------
    int
        The node ID of the added composition container node.

    """
    node_id = diagram.id

    sub_node_ids = []
    for sub_diagram in diagram.diagrams:
        sub_node_id = _convert_diagram_to_graph(sub_diagram, G, node_map, container_id=node_id)
        sub_node_ids.append(sub_node_id)

    external_input_mapping = {j: (0, j) for j in range(diagram.diagrams[0].num_inputs)}
    last_idx = len(diagram.diagrams) - 1
    external_output_mapping = {j: (last_idx, j) for j in range(diagram.diagrams[-1].num_outputs)}

    attrs = {
        "id": node_id,
        "type": "CompositionDiagram",
        "kind": "container",
        "container_type": "composition",
        "phase": None,
        "num_inputs": diagram.num_inputs,
        "num_outputs": diagram.num_outputs,
        "diagram": diagram,
        "container_id": container_id,
        "is_root": is_root,
        "sub_diagram_ids": sub_node_ids,
        "connectivity": diagram.connectivity,
        "external_inputs": list(range(diagram.num_inputs)),
        "external_outputs": list(range(diagram.num_outputs)),
        "external_input_mapping": external_input_mapping,
        "external_output_mapping": external_output_mapping,
    }

    node_idx = G.add_node(attrs)
    node_map[node_id] = node_idx

    for left_idx, conn in diagram.connectivity.items():
        left_diagram = diagram.diagrams[left_idx]
        right_diagram = diagram.diagrams[left_idx + 1]

        connections_by_node: dict[tuple, list] = {}

        for out_port, in_port in conn.items():
            src_node, src_internal_port = find_node_by_external_output(left_diagram, out_port, G, node_map)
            tgt_node, tgt_internal_port = find_node_by_external_input(right_diagram, in_port, G, node_map)
            if src_node is not None and tgt_node is not None:
                key = (src_node, tgt_node)
                if key not in connections_by_node:
                    connections_by_node[key] = []
                connections_by_node[key].append((src_internal_port, tgt_internal_port))

        for (src_node, tgt_node), port_pairs in connections_by_node.items():
            src_ports = [p[0] for p in port_pairs]
            tgt_ports = [p[1] for p in port_pairs]

            edge_data = {
                "source_ports": src_ports,
                "target_ports": tgt_ports,
                "edge_type": "composition",
                "internal": False,
                "connection_type": None,
                "left_idx": left_idx,
                "right_idx": left_idx + 1,
            }

            src_idx = node_map[src_node]
            tgt_idx = node_map[tgt_node]
            G.add_edge(src_idx, tgt_idx, edge_data)

    return node_id


def _add_tensor_node(
    diagram: TensorDiagram,
    G: rx.PyDiGraph,  # ruff: ignore[invalid-argument-name]
    node_map: dict[int, int],
    container_id: int | None,
    is_root: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
) -> int:
    """Add a TensorDiagram as a container node in the graph.

    Parameters
    ----------
    diagram : TensorDiagram
        The tensor diagram to add as a container node.
    G : nx.DiGraph
        The graph to add nodes to (modified in place).
    container_id : int | None
        The node ID of the immediate container of this tensor.
    is_root : bool, default=False
        Whether this is the root (outermost) diagram.

    Returns
    -------
    int
        The node ID of the added tensor container node.
    """
    node_id = diagram.id
    sub_node_ids = []
    for sub_diagram in diagram.diagrams:
        sub_node_id = _convert_diagram_to_graph(sub_diagram, G, node_map, container_id=node_id)
        sub_node_ids.append(sub_node_id)

    external_input_mapping = {}
    input_offset = 0
    for idx, sub_diagram in enumerate(diagram.diagrams):
        for internal_port in range(sub_diagram.num_inputs):
            external_input_mapping[input_offset + internal_port] = (idx, internal_port)
        input_offset += sub_diagram.num_inputs

    external_output_mapping = {}
    output_offset = 0
    for idx, sub_diagram in enumerate(diagram.diagrams):
        for internal_port in range(sub_diagram.num_outputs):
            external_output_mapping[output_offset + internal_port] = (idx, internal_port)
        output_offset += sub_diagram.num_outputs

    attrs = {
        "id": node_id,
        "type": "TensorDiagram",
        "kind": "container",
        "container_type": "tensor",
        "phase": None,
        "num_inputs": diagram.num_inputs,
        "num_outputs": diagram.num_outputs,
        "diagram": diagram,
        "container_id": container_id,
        "is_root": is_root,
        "sub_diagram_ids": sub_node_ids,
        "external_inputs": list(range(diagram.num_inputs)),
        "external_outputs": list(range(diagram.num_outputs)),
        "external_input_mapping": external_input_mapping,
        "external_output_mapping": external_output_mapping,
    }

    node_idx = G.add_node(attrs)
    node_map[node_id] = node_idx
    return node_id


def _add_contracted_node(  # ruff: ignore[complex-structure, too-many-locals]
    diagram: ContractedDiagram,
    G: rx.PyDiGraph,  # ruff: ignore[invalid-argument-name]
    node_map: dict[int, int],
    container_id: int | None,
    is_root: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
) -> int:
    """Add a ContractedDiagram as a container node in the graph.

    Container nodes have kind='container' and store:
        - container_type: 'contracted'
        - first_id: node ID of the first diagram
        - second_id: node ID of the second diagram
        - I1, I2, J1, J2: connection indices
        - container_id: the immediate container of this contraction
        - is_root: whether this is the root container
        - external_input_mapping: maps external input ports to ('first'/'second', internal_port)
        - external_output_mapping: maps external output ports to ('first'/'second', internal_port)

    Internal edges are added for both directions of contraction.

    Parameters
    ----------
    diagram : ContractedDiagram
        The contracted diagram to add as a container node.
    G : nx.DiGraph
        The graph to add nodes and edges to (modified in place).
    container_id : int | None
        The node ID of the immediate container of this contraction.
    is_root : bool, default=False
        Whether this is the root (outermost) diagram.

    Returns
    -------
    int
        The node ID of the added contracted container node.
    """
    node_id = diagram.id

    first_node_id = _convert_diagram_to_graph(diagram.first, G, node_map, container_id=node_id)
    second_node_id = _convert_diagram_to_graph(diagram.second, G, node_map, container_id=node_id)

    external_input_mapping = {}
    input_offset = 0

    for internal_port in diagram.kept_first_inputs:
        external_input_mapping[input_offset] = ("first", internal_port)
        input_offset += 1

    for internal_port in diagram.kept_second_inputs:
        external_input_mapping[input_offset] = ("second", internal_port)
        input_offset += 1

    external_output_mapping = {}
    output_offset = 0

    for internal_port in diagram.kept_first_outputs:
        external_output_mapping[output_offset] = ("first", internal_port)
        output_offset += 1

    for internal_port in diagram.kept_second_outputs:
        external_output_mapping[output_offset] = ("second", internal_port)
        output_offset += 1

    attrs = {
        "id": node_id,
        "type": "ContractedDiagram",
        "kind": "container",
        "container_type": "contracted",
        "phase": None,
        "num_inputs": diagram.num_inputs,
        "num_outputs": diagram.num_outputs,
        "diagram": diagram,
        "container_id": container_id,
        "is_root": is_root,
        "first_id": first_node_id,
        "second_id": second_node_id,
        "I1": list(diagram.I1),
        "I2": list(diagram.I2),
        "J1": list(diagram.J1),
        "J2": list(diagram.J2),
        "kept_first_inputs": diagram.kept_first_inputs,
        "kept_second_inputs": diagram.kept_second_inputs,
        "kept_first_outputs": diagram.kept_first_outputs,
        "kept_second_outputs": diagram.kept_second_outputs,
        "external_inputs": list(range(diagram.num_inputs)),
        "external_outputs": list(range(diagram.num_outputs)),
        "external_input_mapping": external_input_mapping,
        "external_output_mapping": external_output_mapping,
    }

    node_idx = G.add_node(attrs)
    node_map[node_id] = node_idx

    i1_i2_connections: dict[tuple, list] = {}
    for out_idx, in_idx in zip(diagram.I1, diagram.I2, strict=False):
        src_node, src_internal_port = find_node_by_external_output(diagram.first, out_idx, G, node_map)
        tgt_node, tgt_internal_port = find_node_by_external_input(diagram.second, in_idx, G, node_map)

        if src_node is not None and tgt_node is not None:
            key = (src_node, tgt_node)
            if key not in i1_i2_connections:
                i1_i2_connections[key] = []
            i1_i2_connections[key].append((src_internal_port, tgt_internal_port))

    for (src_node, tgt_node), port_pairs in i1_i2_connections.items():
        src_ports = [p[0] for p in port_pairs]
        tgt_ports = [p[1] for p in port_pairs]
        edge_data = {
            "source_ports": src_ports,
            "target_ports": tgt_ports,
            "edge_type": "contracted_internal",
            "internal": True,
            "connection_type": "I1_I2",
        }
        G.add_edge(node_map[src_node], node_map[tgt_node], edge_data)

    j2_j1_connections: dict[tuple, list] = {}
    for out_idx, in_idx in zip(diagram.J2, diagram.J1, strict=False):
        src_node, src_internal_port = find_node_by_external_output(diagram.second, out_idx, G, node_map)
        tgt_node, tgt_internal_port = find_node_by_external_input(diagram.first, in_idx, G, node_map)

        if src_node is not None and tgt_node is not None:
            key = (src_node, tgt_node)
            if key not in j2_j1_connections:
                j2_j1_connections[key] = []
            j2_j1_connections[key].append((src_internal_port, tgt_internal_port))

    for (src_node, tgt_node), port_pairs in j2_j1_connections.items():
        src_ports = [p[0] for p in port_pairs]
        tgt_ports = [p[1] for p in port_pairs]
        edge_data = {
            "source_ports": src_ports,
            "target_ports": tgt_ports,
            "edge_type": "contracted_internal",
            "internal": True,
            "connection_type": "J2_J1",
        }
        G.add_edge(node_map[src_node], node_map[tgt_node], edge_data)

    return node_id


def reconstruct_from_node(G: rx.PyDiGraph, node_id: int, reg: GateRegister, id_to_idx: dict[int, int]) -> Diagram:  # ruff: ignore[invalid-argument-name]
    """Reconstruct a diagram from a graph node.

    Parameters
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to reconstruct.
    reg : GateRegister
        Gate register of the input graph.

    Returns
    -------
    Diagram
        The reconstructed diagram.

    Raises
    ------
    ValueError
        If the node type is unknown.
    """
    node_idx = id_to_idx[node_id]
    attrs = G[node_idx]
    kind = attrs.get("kind")

    if kind in {"proper", "compact"}:
        return reconstruct_proper_node(G, node_idx, reg)
    if kind == "container":
        container_type = attrs.get("container_type")
        if container_type == "composition":
            return reconstruct_composition_node(G, node_idx, reg, id_to_idx)
        if container_type == "tensor":
            return reconstruct_tensor_node(G, node_idx, reg, id_to_idx)
        if container_type == "contracted":
            return reconstruct_contracted_node(G, node_idx, reg, id_to_idx)
        msg = f"Unknown container type: {container_type}"
        raise ValueError(msg)
    msg = f"Unknown node kind: {kind}"
    raise ValueError(msg)


def reconstruct_proper_node(G: rx.PyDiGraph, node_idx: int, reg: GateRegister) -> Diagram:  # ruff: ignore[complex-structure, invalid-argument-name, too-many-return-statements, too-many-branches, too-many-locals]
    """Reconstruct a proper diagram from a graph node.

    Parameters
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_idx : int
        The rustworkx node index to reconstruct.
    reg : GateRegister
        Gate register of the input graph.

    Returns
    -------
    Diagram
        The reconstructed diagram.

    Raises
    ------
    ValueError
        If the node type is not among proper diagram types.
    """
    attrs = G[node_idx]
    node_id = attrs["id"]
    node_type = attrs.get("type")
    phase = attrs.get("phase")
    num_inputs = attrs.get("num_inputs", 0)
    num_outputs = attrs.get("num_outputs", 0)
    is_parametric = phase.is_parametric() if isinstance(phase, ZxPoly) else isinstance(phase, Expr)
    feedforward = bool(attrs.get("feedforward"))
    measurement_ids = attrs.get("measurement_ids")
    param_measurement_map = attrs.get("param_measurement_map") or {}

    if node_type == "QSpider":
        result = QSpider(
            num_inputs,
            num_outputs,
            phase if phase is not None else ZxPoly({}),
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
        if node_id in reg.measurement_nodes:
            result.id = node_id
        return result
    if node_type == "PSpider":
        result = PSpider(
            num_inputs,
            num_outputs,
            phase if phase is not None else ZxPoly({}),
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )  # type: ignore[assignment]
        if node_id in reg.measurement_nodes:
            result.id = node_id
        return result
    if node_type == "Swap":
        return Swap()
    if node_type == "VoidDiagram":
        return VoidDiagram(num_inputs, num_outputs)
    if node_type == "Fourier":
        return Fourier()
    if node_type == "FourierInv":
        return FourierInv()
    if node_type == "Fourier2":
        return Fourier2()
    if node_type == "DisplacementGate":
        return DisplacementGate(
            phase,
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "PhaseRotationGate":
        return PhaseRotationGate(
            phase,
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "SqueezingGate":
        return SqueezingGate(
            phase,
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "BeamsplitterGate":
        return BeamsplitterGate(
            phase,
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "ControlledSumGate":
        control = attrs.get("control")
        target = attrs.get("target")
        return ControlledSumGate(
            phase,
            control,
            target,
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "ControlledZGate":
        return ControlledZGate(
            phase,
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "CubicPhaseGate":
        return CubicPhaseGate(
            phase,
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "ShearXInvariantGate":
        return ShearXInvariantGate(
            phase,
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "ShearPInvariantGate":
        return ShearPInvariantGate(
            phase,
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "Squeezing45Gate":
        return Squeezing45Gate(
            phase,
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "MeasurementGate":
        return MeasurementGate(
            phase,
            is_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "ArbitraryGate":
        alpha = attrs.get("alpha")
        beta = attrs.get("beta")
        lam = attrs.get("lam")
        arb_parametric = isinstance(alpha, Expr) or isinstance(beta, Expr) or isinstance(lam, Expr)
        return ArbitraryGate(
            alpha,
            beta,
            lam,
            arb_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )
    if node_type == "TwoModeShearGate":
        a = attrs.get("a")
        b = attrs.get("b")
        shear2_parametric = isinstance(a, Expr) or isinstance(b, Expr)
        return TwoModeShearGate(
            a,
            b,
            shear2_parametric,
            feedforward,
            measurement_ids,
            param_measurement_map=param_measurement_map,
        )

    msg = f"Unknown proper node type: {node_type}"
    raise ValueError(msg)


def reconstruct_composition_node(
    G: rx.PyDiGraph,  # ruff: ignore[invalid-argument-name]
    node_idx: int,
    reg: GateRegister,
    id_to_idx: dict[int, int],
) -> Diagram:
    """Reconstruct a CompositionDiagram from a graph node.

    Parameters
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_idx : int
        The rustworkx node index to reconstruct.
    reg : GateRegister
        Gate register of the input graph.
    id_to_idx : dict[int, int]
        Mapping from cvzx diagram id to rustworkx node index.

    Returns
    -------
    Diagram
        The reconstructed CompositionDiagram.

    Raises
    ------
    ValueError
        If the connectivity is malformed.
    """
    attrs = G[node_idx]
    sub_diagram_ids = attrs.get("sub_diagram_ids", [])
    connectivity = dict(attrs.get("connectivity", {}))

    sub_diagrams = [reconstruct_from_node(G, sub_id, reg, id_to_idx) for sub_id in sub_diagram_ids]

    if connectivity:
        for left_idx, conn in connectivity.items():
            if left_idx >= len(sub_diagrams) - 1:
                msg = f"Invalid connectivity key {left_idx}: exceeds number of sub-diagrams"
                raise ValueError(msg)

            left_diagram = sub_diagrams[left_idx]
            right_diagram = sub_diagrams[left_idx + 1]

            if set(conn.keys()) != set(range(right_diagram.num_inputs)):
                msg = f"Connectivity for index {left_idx} does not match input ports of sub_diagram {left_idx + 1}"
                raise ValueError(msg)

            for out_port in conn.values():
                if out_port < 0 or out_port >= left_diagram.num_outputs:
                    msg = (
                        f"Connectivity for index {left_idx} has invalid output port "
                        f"{out_port} for sub_diagram {left_idx}"
                    )
                    raise ValueError(msg)

    return CompositionDiagram(sub_diagrams, connectivity)


def reconstruct_tensor_node(G: rx.PyDiGraph, node_idx: int, reg: GateRegister, id_to_idx: dict[int, int]) -> Diagram:  # ruff: ignore[invalid-argument-name]
    """Reconstruct a TensorDiagram from a graph node.

    Parameters
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_idx : int
        The rustworkx node index to reconstruct.
    reg : GateRegister
        Gate register of the input graph.
    id_to_idx : dict[int, int]
        Mapping from cvzx diagram id to rustworkx node index.

    Returns
    -------
    Diagram
        The reconstructed diagram.
    """
    attrs = G[node_idx]
    sub_diagram_ids = attrs.get("sub_diagram_ids", [])

    sub_diagrams = [reconstruct_from_node(G, sub_id, reg, id_to_idx) for sub_id in sub_diagram_ids]

    return TensorDiagram(sub_diagrams)


def reconstruct_contracted_node(
    G: rx.PyDiGraph,  # ruff: ignore[invalid-argument-name]
    node_idx: int,
    reg: GateRegister,
    id_to_idx: dict[int, int],
) -> Diagram:
    """Reconstruct a ContractedDiagram from a graph node.

    Parameters
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_idx : int
        The rustworkx node index to reconstruct.
    reg : GateRegister
        Gate register of the input graph.
    id_to_idx : dict[int, int]
        Mapping from cvzx diagram id to rustworkx node index.

    Returns
    -------
    Diagram
        The reconstructed diagram.

    Raises
    ------
    ValueError
        If the ContractedDiagram node misses first_id or second_id.
    """
    attrs = G[node_idx]
    first_id = attrs.get("first_id")
    second_id = attrs.get("second_id")
    I1 = attrs.get("I1", [])  # ruff: ignore[non-lowercase-variable-in-function]
    I2 = attrs.get("I2", [])  # ruff: ignore[non-lowercase-variable-in-function]
    J1 = attrs.get("J1", [])  # ruff: ignore[non-lowercase-variable-in-function]
    J2 = attrs.get("J2", [])  # ruff: ignore[non-lowercase-variable-in-function]

    if first_id is None or second_id is None:
        msg = f"ContractedDiagram node {attrs.get('id')} missing first_id or second_id"
        raise ValueError(msg)

    first = reconstruct_from_node(G, first_id, reg, id_to_idx)
    second = reconstruct_from_node(G, second_id, reg, id_to_idx)

    return ContractedDiagram(first, second, I1, I2, J1, J2)

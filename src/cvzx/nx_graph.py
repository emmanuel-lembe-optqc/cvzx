"""Graph extraction utilities for CV ZX diagrams.

This module provides functions to convert CV ZX diagrams to directed graphs
for optimization purposes. The graph preserves all information needed for
faithful reconstruction.

Key design decisions:
    - Containers (CompositionDiagram, TensorDiagram, ContractedDiagram) are
      preserved as nodes with kind='container'
    - Proper diagrams are nodes with kind='proper'
    - The hierarchy is preserved through sub_diagram_ids
    - Each node stores its immediate container ID (container_id)
    - The root container is identified by is_root=True
    - All port connections are stored as edges with port information
"""

from typing import cast

import networkx as nx
from sympy import Expr

from cvzx.base_gates import (
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
from cvzx.gates import (
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


class GateRegister:
    """Registry for tracking specific gate types and nodes in a CV ZX graph.

    The GateRegister maintains sets of node IDs for different gate types,
    enabling O(1) lookups instead of O(N) scans of the entire graph.
    This is essential for performance in large circuits.

    The registry tracks:
        - Proper nodes: Q/P spiders, squeezing, displacement, rotation, Fourier gates
        - Terminals: input states (0→1) and measurements (1→0)
        - Containers: tensor, composition, and contracted diagrams

    The registry must be kept in sync with the graph. Whenever the graph is
    modified (nodes added, removed, or changed), the registry must be updated
    accordingly using `add_node()`, `remove_node()`, or rebuilding from scratch.

    Parameters
    ----------
        squeezing_gates set[int]: Node IDs of squeezing gates (Sq)
        displacement_gates set[int]: Node IDs of displacement gates (D)
        rotation_gates set[int]: Node IDs of phase rotation gates (R)
        fourier_gates set[int]: Node IDs of Fourier gates (F, F†, F²)
        identity_spiders set[int]: Node IDs of identity spiders (zero phase, 1→1)
        input_states set[int]: Node IDs of input states (0 inputs, 1 output)
        measurement_nodes set[int]: Node IDs of measurements (1 input, 0 outputs)
        contracted_diagrams set[int]: Node IDs of ContractedDiagram containers
        tensor_nodes set[int]: Node IDs of TensorDiagram containers
        composition_nodes set[int]: Node IDs of CompositionDiagram containers
        void_nodes set[int]: Node IDs of VoidDiagram placeholders
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

    def add_node(self, node_id: int, attrs: dict) -> None:  # ruff: ignore[complex-structure, too-many-branches]
        """Add a node to the appropriate sets based on its attributes.

        This method inspects the node's attributes and adds its ID to the
        corresponding sets based on its type, kind, and container type.

        Parameters
        ----------
        node_id : int
            The ID of the node to add.
        attrs : dict
            The node's attributes from the graph, containing at least 'kind',
            and optionally 'type', 'container_type', 'num_inputs', 'num_outputs',
            and 'phase'.

        Notes
        -----
        - Container nodes (kind='container') are added to container-specific sets.
        - Proper nodes (kind='proper') are added to gate-type-specific sets.
        - Identity spiders are detected using `_is_identity_spider()`.
        - Input states have num_inputs=0, num_outputs=1.
        - Measurements have num_inputs=1, num_outputs=0.
        """
        gate_type = attrs.get("type")
        kind = attrs.get("kind")
        container_type = attrs.get("container_type")

        # Ignore nodes that are neither proper nor container
        if kind not in {"proper", "container", "compact"}:
            return

        # Handle container nodes
        if kind == "container":
            if container_type == "tensor":
                self.tensor_nodes.add(node_id)
            elif container_type == "composition":
                self.composition_nodes.add(node_id)
            elif container_type == "contracted":
                self.contracted_diagrams.add(node_id)
            return

        # Handle proper nodes
        # Gate types
        if gate_type == "SqueezingGate":
            self.squeezing_gates.add(node_id)
        elif gate_type == "DisplacementGate":
            self.displacement_gates.add(node_id)
        elif gate_type == "PhaseRotationGate":
            self.rotation_gates.add(node_id)
        elif gate_type in {"Fourier", "FourierInv", "Fourier2"}:
            self.fourier_gates.add(node_id)

        # Identity spiders
        if self._is_identity_spider(attrs):
            self.identity_spiders.add(node_id)

        # Void placeholders (see `VoidDiagram`)
        if gate_type == "VoidDiagram":
            self.void_nodes.add(node_id)

        # Terminals
        if self._is_input_state(attrs):
            self.input_states.add(node_id)
        elif self._is_measurement(attrs):
            self.measurement_nodes.add(node_id)

    def remove_node(self, node_id: int) -> None:
        """Remove a node from all sets.

        This method removes the given node ID from every set in the registry.
        It uses `discard()` to safely handle cases where the node is not present.

        Parameters
        ----------
        node_id : int
            The ID of the node to remove from the registry.
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

    def copy(self) -> "GateRegister":
        """Create a shallow copy of the register.

        Returns
        -------
        GateRegister
            A new GateRegister instance with copies of all sets.

        Notes
        -----
        This is useful for parallelization where each partition needs its
        own independent registry that can be modified without affecting others.
        The copy is shallow (sets are copied, but the contained integers are immutable).
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
        return new_reg

    def clear(self) -> None:
        """Clear all sets in the registry.

        This removes all node IDs from every set, effectively resetting
        the registry to an empty state.
        """
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

    def build_from_graph(self, graph: nx.DiGraph) -> None:
        """Build the entire registry from a graph.

        This clears all sets and then adds every node in the graph
        using `add_node()`. This is useful when the graph has been
        extensively modified and the registry may be out of sync.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to rebuild the registry from.

        Notes
        -----
        This operation is O(N) where N is the number of nodes in the graph.
        It should be used sparingly; incremental updates are preferred.
        """
        self.clear()
        for node_id, attrs in graph.nodes(data=True):
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

        num_inputs = attrs.get("num_inputs", 0)
        num_outputs = attrs.get("num_outputs", 0)
        if num_inputs != 1 or num_outputs != 1:
            return False

        phase = attrs.get("phase")
        if phase is None:
            return False

        return bool(phase.is_zero)

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


class CVZXGraph:
    """A CV ZX diagram represented as a directed graph, paired with its registry.

    ``CVZXGraph`` bundles the ``networkx.DiGraph`` produced by :func:`to_graph`
    together with the :class:`GateRegister` that indexes it, so that code
    working on a diagram's graph representation does not have to thread the
    graph and registry through separately. It is a thin, mutable wrapper:
    it does not change how the graph or registry are built, read, or kept
    in sync, it only gives them a single home.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph representation of a CV ZX diagram, as produced by
        :func:`to_graph`.
    registry : GateRegister | None, optional
        A registry already built from ``graph``. If ``None`` (the default),
        a new registry is built from ``graph`` via
        :meth:`GateRegister.build_from_graph`.

    Attributes
    ----------
    graph : nx.DiGraph
        The underlying graph.
    registry : GateRegister
        The registry indexing ``graph``.
    """

    def __init__(self, graph: nx.DiGraph, registry: "GateRegister | None" = None) -> None:
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
            The graph representation of ``diagram``, with a freshly built
            registry.
        """
        return to_graph(diagram)

    def to_diagram(self) -> Diagram:
        """Reconstruct the CV ZX diagram from the current graph.

        See :func:`to_diagram`; this raises ``ValueError`` under the same
        condition, since it delegates to that function.

        Returns
        -------
        Diagram
            The reconstructed diagram.
        """
        return to_diagram(self)

    def rebuild_registry(self) -> None:
        """Rebuild ``self.registry`` from the current state of ``self.graph``.

        This performs a full O(N) rescan of the graph, discarding whatever
        the registry previously held. Call this after mutating ``self.graph``
        directly (for example, after applying a rewrite rule) if the
        registry has not already been kept in sync incrementally.
        """
        self.registry.build_from_graph(self.graph)

    def copy(self) -> "CVZXGraph":
        """Return an independent copy of this ``CVZXGraph``.

        The graph is copied via ``networkx.DiGraph.copy()`` (a new graph
        with its own node/edge attribute dicts) and the registry via
        :meth:`GateRegister.copy`, so mutating the copy's graph or registry
        does not affect this one.

        Returns
        -------
        CVZXGraph
            An independent copy of this graph and its registry.
        """
        return CVZXGraph(self.graph.copy(), self.registry.copy())

    def _root_attrs(self) -> dict:
        """Return the attribute dict of the graph's root node.

        Returns
        -------
        dict
            The root node's attributes.

        Raises
        ------
        ValueError
            If the graph has no root node.
        """
        root = get_root_node(self)
        if root is None:
            msg = "No root node found in the graph."
            raise ValueError(msg)
        return cast("dict", self.graph.nodes[root])

    @property
    def root_id(self) -> int | None:
        """The node ID of the root container, or ``None`` if there is none.

        Returns
        -------
        int | None
        """
        return get_root_node(self)

    @property
    def num_inputs(self) -> int:
        """The number of external inputs of the diagram.

        See :meth:`_root_attrs`; this raises ``ValueError`` under the same
        condition, since it delegates to that method.

        Returns
        -------
        int
        """
        return cast("int", self._root_attrs()["num_inputs"])

    @property
    def num_outputs(self) -> int:
        """The number of external outputs of the diagram.

        See :meth:`_root_attrs`; this raises ``ValueError`` under the same
        condition, since it delegates to that method.

        Returns
        -------
        int
        """
        return cast("int", self._root_attrs()["num_outputs"])

    def __len__(self) -> int:
        """Return the number of nodes in the graph.

        Returns
        -------
        int
        """
        return cast("int", self.graph.number_of_nodes())

    def __repr__(self) -> str:
        """Return a debug representation of this ``CVZXGraph``.

        Returns
        -------
        str
        """
        return f"CVZXGraph(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"

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
    """Convert a CV ZX diagram to a graph representation.

    The graph preserves the full hierarchy of the original diagram:
        - Container nodes (CompositionDiagram, TensorDiagram, ContractedDiagram)
          have kind='container' and store their sub-diagram IDs
        - Proper nodes have kind='proper' and store their attributes
        - Each node stores its immediate container ID (container_id)
        - The root container is marked with is_root=True
        - Edges represent connections between nodes

    Parameters
    ----------
    diagram : Diagram
        The CV ZX diagram to convert.

    Returns
    -------
    CVZXGraph
        Graph representation of the diagram, with all nodes and edges, and
        a freshly built registry.
    """
    graph = nx.DiGraph()
    root_id = _convert_diagram_to_graph(diagram, graph, container_id=None, is_root=True)

    # Mark the root node
    if root_id is not None and root_id != -1:
        graph.nodes[root_id]["is_root"] = True

    # Remove diagram references
    for _, attrs in graph.nodes(data=True):
        if "diagram" in attrs:
            del attrs["diagram"]

    return CVZXGraph(graph)


def to_diagram(cvzx_graph: CVZXGraph) -> Diagram:
    """Reconstruct a CV ZX diagram from a graph representation.

    This function reconstructs the original diagram from the graph, preserving
    the full hierarchy, container nodes, and all connections.

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

    # Find the root node
    root = get_root_node(cvzx_graph)
    if root is None:
        msg = "No root node found in the graph."
        raise ValueError(msg)

    # Reconstruct the diagram from the root
    return _reconstruct_from_node(graph, root)


# =========================================================================
# Diagram to graph helper functions
# =========================================================================


def find_node_by_external_output(
    diagram: Diagram,
    ext_port: int,
    G: nx.DiGraph,  # ruff: ignore[invalid-argument-name]
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
    # If it's a proper diagram, return its node ID and the port
    if isinstance(diagram, (ProperDiagram, CompactDiagram)):
        return diagram.id, ext_port

    # If it's a container diagram, get its node attributes
    attrs = G.nodes[diagram.id]
    output_mapping = attrs.get("external_output_mapping", {})

    if ext_port not in output_mapping:
        return None, None

    # output_mapping[ext_port] = (sub_ref, internal_port)
    # where sub_ref is either:
    #   - an index (for TensorDiagram)
    #   - 'first' or 'second' (for ContractedDiagram)
    sub_ref, internal_port = output_mapping[ext_port]

    # Get the sub-diagram based on the reference type
    if attrs.get("container_type") == "tensor" or attrs.get("container_type") == "composition":
        sub_node_id = attrs["sub_diagram_ids"][sub_ref]
        sub_diagram = G.nodes[sub_node_id]["diagram"]
    elif attrs.get("container_type") == "contracted" and isinstance(diagram, ContractedDiagram):
        # For ContractedDiagram: sub_ref is 'first' or 'second'
        sub_diagram = diagram.first if sub_ref == "first" else diagram.second
    else:
        return None, None

    # Recursively traverse into the sub-diagram with the internal port
    return find_node_by_external_output(sub_diagram, internal_port, G)


def find_node_by_external_input(
    diagram: Diagram,
    ext_port: int,
    G: nx.DiGraph,  # ruff: ignore[invalid-argument-name]
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
    # If it's a proper diagram, return its node ID and the port
    if isinstance(diagram, (ProperDiagram, CompactDiagram)):
        return diagram.id, ext_port

    # If it's a container diagram, get its node attributes
    attrs = G.nodes[diagram.id]
    input_mapping = attrs.get("external_input_mapping", {})

    if ext_port not in input_mapping:
        return None, None

    # input_mapping[ext_port] = (sub_ref, internal_port)
    sub_ref, internal_port = input_mapping[ext_port]

    # Get the sub-diagram based on the reference type
    if attrs.get("container_type") == "tensor" or attrs.get("container_type") == "composition":
        sub_node_id = attrs["sub_diagram_ids"][sub_ref]
        sub_diagram = G.nodes[sub_node_id]["diagram"]
    elif attrs.get("container_type") == "contracted" and isinstance(diagram, ContractedDiagram):
        sub_diagram = diagram.first if sub_ref == "first" else diagram.second
    else:
        return None, None

    # Recursively traverse into the sub-diagram with the internal port
    return find_node_by_external_input(sub_diagram, internal_port, G)


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
    return [n for n, attrs in graph.nodes(data=True) if attrs.get("kind") == "proper"]


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
    return [n for n, attrs in graph.nodes(data=True) if attrs.get("kind") == "container"]


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
    for node, attrs in graph.nodes(data=True):
        if attrs.get("is_root", False):
            return cast("int", node)
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
    attrs = graph.nodes[node_id]
    return cast("int | None", attrs.get("container_id"))


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
    attrs = graph.nodes[container_node]
    if attrs.get("kind") != "container":
        return []
    return cast("list[int]", attrs.get("sub_diagram_ids", []))


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
    attrs = graph.nodes[container_node]
    if attrs.get("container_type") != "composition":
        return None
    return cast("dict[int, dict[int, int]] | None", attrs.get("connectivity"))


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
    attrs = graph.nodes[container_node]
    if attrs.get("container_type") != "contracted":
        return None
    return {
        "I1": attrs.get("I1", []),
        "I2": attrs.get("I2", []),
        "J1": attrs.get("J1", []),
        "J2": attrs.get("J2", []),
    }


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
    return [n for n, attrs in graph.nodes(data=True) if attrs.get("container_id") == container_node]


def _convert_diagram_to_graph(
    diagram: Diagram,
    G: nx.DiGraph,  # ruff: ignore[invalid-argument-name]
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
        return _add_proper_node(diagram, G, container_id)
    if isinstance(diagram, CompositionDiagram):
        return _add_composition_node(diagram, G, container_id, is_root)
    if isinstance(diagram, TensorDiagram):
        return _add_tensor_node(diagram, G, container_id, is_root)
    if isinstance(diagram, ContractedDiagram):
        return _add_contracted_node(diagram, G, container_id, is_root)
    return -1


def _add_proper_node(  # ruff: ignore[complex-structure, too-many-branches]
    diagram: ProperDiagram | CompactDiagram,
    G: nx.DiGraph,  # ruff: ignore[invalid-argument-name]
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

    # Every CompactDiagram gate (see cvzx.gates) now carries feedforward /
    # measurement_ids (mirroring DisplacementGate's original pattern), so
    # this is pulled once here and attached to every node below rather
    # than being recomputed per gate-type branch.
    feedforward = getattr(diagram, "feedforward", None)
    measurement_ids = getattr(diagram, "measurement_ids", None)

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

    if isinstance(diagram, DisplacementGate):
        phase = getattr(diagram, "alpha", None)
        G.add_node(
            node_id,
            id=node_id,
            type=node_type,
            kind=kind,
            phase=phase,
            feedforward=feedforward,
            measurement_ids=measurement_ids,
            num_inputs=diagram.num_inputs,
            num_outputs=diagram.num_outputs,
            diagram=diagram,
            container_id=container_id,
            # Store external port mappings
            external_inputs=list(range(diagram.num_inputs)),
            external_outputs=list(range(diagram.num_outputs)),
        )
    elif isinstance(diagram, ControlledSumGate):
        control = diagram.control
        target = diagram.target
        G.add_node(
            node_id,
            id=node_id,
            type=node_type,
            kind=kind,
            phase=phase,
            control=control,
            target=target,
            feedforward=feedforward,
            measurement_ids=measurement_ids,
            num_inputs=diagram.num_inputs,
            num_outputs=diagram.num_outputs,
            diagram=diagram,
            container_id=container_id,
            # Store external port mappings
            external_inputs=list(range(diagram.num_inputs)),
            external_outputs=list(range(diagram.num_outputs)),
        )
    elif isinstance(diagram, ArbitraryGate):
        G.add_node(
            node_id,
            id=node_id,
            type=node_type,
            kind=kind,
            alpha=diagram.alpha,
            beta=diagram.beta,
            lam=diagram.lam,
            feedforward=feedforward,
            measurement_ids=measurement_ids,
            num_inputs=diagram.num_inputs,
            num_outputs=diagram.num_outputs,
            diagram=diagram,
            container_id=container_id,
            # Store external port mappings
            external_inputs=list(range(diagram.num_inputs)),
            external_outputs=list(range(diagram.num_outputs)),
        )
    elif isinstance(diagram, TwoModeShearGate):
        G.add_node(
            node_id,
            id=node_id,
            type=node_type,
            kind=kind,
            a=diagram.a,
            b=diagram.b,
            feedforward=feedforward,
            measurement_ids=measurement_ids,
            num_inputs=diagram.num_inputs,
            num_outputs=diagram.num_outputs,
            diagram=diagram,
            container_id=container_id,
            # Store external port mappings
            external_inputs=list(range(diagram.num_inputs)),
            external_outputs=list(range(diagram.num_outputs)),
        )
    else:
        G.add_node(
            node_id,
            id=node_id,
            type=node_type,
            kind=kind,
            phase=phase,
            feedforward=feedforward,
            measurement_ids=measurement_ids,
            num_inputs=diagram.num_inputs,
            num_outputs=diagram.num_outputs,
            diagram=diagram,
            container_id=container_id,
            # Store external port mappings
            external_inputs=list(range(diagram.num_inputs)),
            external_outputs=list(range(diagram.num_outputs)),
        )
    return node_id


def _add_composition_node(
    diagram: CompositionDiagram,
    G: nx.DiGraph,  # ruff: ignore[invalid-argument-name]
    container_id: int | None,
    is_root: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
) -> int:
    """Add a CompositionDiagram as a container node in the graph.

    Container nodes have kind='container' and store:
        - container_type: 'composition'
        - sub_diagram_ids: ordered list of child node IDs
        - connectivity: the composition connectivity dictionary
        - container_id: the immediate container of this composition
        - is_root: whether this is the root container

    Edges are added between sub-diagrams based on the connectivity dictionary.
    The connectivity maps output ports of the left diagram to input ports of
    the right diagram. For each connection, the function traverses through
    any nested containers to find the actual proper nodes that handle the ports.

    If multiple connections exist between the same pair of proper nodes, they
    are stored as lists of source_port and target_port on a single edge.

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

    Notes
    -----
    The connectivity dictionary follows the convention:
        connectivity[i] connects diagrams[i] → diagrams[i+1]
        where each entry maps: output port of left → input port of right
    """
    node_id = diagram.id

    # First, recursively convert all sub-diagrams
    sub_node_ids = []
    for sub_diagram in diagram.diagrams:
        sub_node_id = _convert_diagram_to_graph(sub_diagram, G, container_id=node_id)
        sub_node_ids.append(sub_node_id)

    # `find_node_by_external_output`/`find_node_by_external_input` resolve
    # THROUGH a composition container exactly like they do a tensor
    # container: via an `external_output_mapping`/`external_input_mapping`
    # dict keyed by external port, valued (sub_diagram_index, internal_port)
    # (their own container_type check already branches on "tensor" OR
    # "composition" and does `sub_diagram_ids[sub_ref]` -- that branch was
    # simply unreachable for composition nodes before this, since this
    # dict was never populated here, so `ext_port not in output_mapping`
    # always failed and any composition edge whose endpoint resolved
    # through a *nested* CompositionDiagram (a CompositionDiagram sitting
    # as one row of an enclosing TensorDiagram, itself then composed with
    # a neighbor -- e.g. an ancilla-preparation sub-composition placed at
    # one row of a wider layer) was silently dropped instead of being
    # added). A composition's external inputs are exactly its first
    # element's inputs, in order; its external outputs are exactly its
    # last element's outputs, in order (see `__post_init__`:
    # `_num_inputs = self.diagrams[0].num_inputs`,
    # `_num_outputs = self.diagrams[-1].num_outputs`).
    external_input_mapping = {j: (0, j) for j in range(diagram.diagrams[0].num_inputs)}
    last_idx = len(diagram.diagrams) - 1
    external_output_mapping = {j: (last_idx, j) for j in range(diagram.diagrams[-1].num_outputs)}

    G.add_node(
        node_id,
        id=node_id,
        type="CompositionDiagram",
        kind="container",
        container_type="composition",
        phase=None,
        num_inputs=diagram.num_inputs,
        num_outputs=diagram.num_outputs,
        diagram=diagram,
        container_id=container_id,
        is_root=is_root,
        sub_diagram_ids=sub_node_ids,
        connectivity=diagram.connectivity,
        external_inputs=list(range(diagram.num_inputs)),
        external_outputs=list(range(diagram.num_outputs)),
        external_input_mapping=external_input_mapping,
        external_output_mapping=external_output_mapping,
    )

    # Add edges between sub-diagrams based on connectivity
    # connectivity[i] connects diagrams[i] → diagrams[i+1]
    # conn maps: output port of left diagram → input port of right diagram
    for left_idx, conn in diagram.connectivity.items():
        left_diagram = diagram.diagrams[left_idx]
        right_diagram = diagram.diagrams[left_idx + 1]

        # Group connections by (src_node, tgt_node) to combine them
        connections_by_node: dict[tuple, list] = {}  # (src_node, tgt_node) -> list of (src_port, tgt_port)

        # conn maps: output port of left → input port of right
        for out_port, in_port in conn.items():
            # Find the proper node in left diagram for this output port
            src_node, src_internal_port = find_node_by_external_output(left_diagram, out_port, G)
            # Find the proper node in right diagram for this input port
            tgt_node, tgt_internal_port = find_node_by_external_input(right_diagram, in_port, G)
            if src_node is not None and tgt_node is not None:
                key = (src_node, tgt_node)
                if key not in connections_by_node:
                    connections_by_node[key] = []
                connections_by_node[key].append((src_internal_port, tgt_internal_port))

        # Add one edge per unique (src_node, tgt_node) pair with combined port lists
        for (src_node, tgt_node), port_pairs in connections_by_node.items():
            # Separate the port pairs into two lists
            src_ports = [p[0] for p in port_pairs]
            tgt_ports = [p[1] for p in port_pairs]

            G.add_edge(
                src_node,
                tgt_node,
                source_ports=src_ports,
                target_ports=tgt_ports,
                edge_type="composition",
                internal=False,
                connection_type=None,
                left_idx=left_idx,
                right_idx=left_idx + 1,
            )

    return node_id


def _add_tensor_node(
    diagram: TensorDiagram,
    G: nx.DiGraph,  # ruff: ignore[invalid-argument-name]
    container_id: int | None,
    is_root: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
) -> int:
    """Add a TensorDiagram as a container node in the graph.

    Container nodes have kind='container' and store:
        - container_type: 'tensor'
        - sub_diagram_ids: ordered list of child node IDs
        - container_id: the immediate container of this tensor
        - is_root: whether this is the root container
        - external_input_mapping: maps external input ports to (sub_diagram_idx, internal_port)
        - external_output_mapping: maps external output ports to (sub_diagram_idx, internal_port)

    No edges are added between tensor components since they are parallel.

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
    # First, recursively convert all sub-diagrams
    sub_node_ids = []
    for sub_diagram in diagram.diagrams:
        sub_node_id = _convert_diagram_to_graph(sub_diagram, G, container_id=node_id)
        sub_node_ids.append(sub_node_id)

    # Build external input mapping
    # Tensor inputs are concatenated: inputs of sub_diagram 0, then sub_diagram 1, etc.
    external_input_mapping = {}
    input_offset = 0
    for idx, sub_diagram in enumerate(diagram.diagrams):
        for internal_port in range(sub_diagram.num_inputs):
            external_input_mapping[input_offset + internal_port] = (idx, internal_port)
        input_offset += sub_diagram.num_inputs

    # Build external output mapping
    external_output_mapping = {}
    output_offset = 0
    for idx, sub_diagram in enumerate(diagram.diagrams):
        for internal_port in range(sub_diagram.num_outputs):
            external_output_mapping[output_offset + internal_port] = (idx, internal_port)
        output_offset += sub_diagram.num_outputs

    G.add_node(
        node_id,
        id=node_id,
        type="TensorDiagram",
        kind="container",
        container_type="tensor",
        phase=None,
        num_inputs=diagram.num_inputs,
        num_outputs=diagram.num_outputs,
        diagram=diagram,
        container_id=container_id,
        is_root=is_root,
        sub_diagram_ids=sub_node_ids,
        external_inputs=list(range(diagram.num_inputs)),
        external_outputs=list(range(diagram.num_outputs)),
        external_input_mapping=external_input_mapping,
        external_output_mapping=external_output_mapping,
    )

    # No edges between tensor components (they are parallel)
    return node_id


def _add_contracted_node(  # ruff: ignore[complex-structure]
    diagram: ContractedDiagram,
    G: nx.DiGraph,  # ruff: ignore[invalid-argument-name]
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

    # Convert first and second diagrams
    first_node_id = _convert_diagram_to_graph(diagram.first, G, container_id=node_id)
    second_node_id = _convert_diagram_to_graph(diagram.second, G, container_id=node_id)

    # Build external input mapping
    # Inputs come from: first inputs (kept: not in J1) + second inputs (kept: not in I2)
    external_input_mapping = {}
    input_offset = 0

    # First diagram inputs (kept: those NOT in J1)
    for internal_port in diagram.kept_first_inputs:
        external_input_mapping[input_offset] = ("first", internal_port)
        input_offset += 1

    # Second diagram inputs (kept: those NOT in I2)
    for internal_port in diagram.kept_second_inputs:
        external_input_mapping[input_offset] = ("second", internal_port)
        input_offset += 1

    # Build external output mapping
    # Outputs come from: first outputs (kept: not in I1) + second outputs (kept: not in J2)
    external_output_mapping = {}
    output_offset = 0

    # First diagram outputs (kept: those NOT in I1)
    for internal_port in diagram.kept_first_outputs:
        external_output_mapping[output_offset] = ("first", internal_port)
        output_offset += 1

    # Second diagram outputs (kept: those NOT in J2)
    for internal_port in diagram.kept_second_outputs:
        external_output_mapping[output_offset] = ("second", internal_port)
        output_offset += 1

    G.add_node(
        node_id,
        id=node_id,
        type="ContractedDiagram",
        kind="container",
        container_type="contracted",
        phase=None,
        num_inputs=diagram.num_inputs,
        num_outputs=diagram.num_outputs,
        diagram=diagram,
        container_id=container_id,
        is_root=is_root,
        first_id=first_node_id,
        second_id=second_node_id,
        I1=list(diagram.I1),
        I2=list(diagram.I2),
        J1=list(diagram.J1),
        J2=list(diagram.J2),
        kept_first_inputs=diagram.kept_first_inputs,
        kept_second_inputs=diagram.kept_second_inputs,
        kept_first_outputs=diagram.kept_first_outputs,
        kept_second_outputs=diagram.kept_second_outputs,
        external_inputs=list(range(diagram.num_inputs)),
        external_outputs=list(range(diagram.num_outputs)),
        external_input_mapping=external_input_mapping,
        external_output_mapping=external_output_mapping,
    )

    # Group I1→I2 connections by (src_node, tgt_node)
    i1_i2_connections: dict[tuple, list] = {}  # (src_node, tgt_node) -> list of (src_port, tgt_port)

    # Add internal edges: I1 (outputs of first) → I2 (inputs of second)
    for out_idx, in_idx in zip(diagram.I1, diagram.I2, strict=False):
        src_node, src_internal_port = find_node_by_external_output(diagram.first, out_idx, G)
        tgt_node, tgt_internal_port = find_node_by_external_input(diagram.second, in_idx, G)

        if src_node is not None and tgt_node is not None:
            key = (src_node, tgt_node)
            if key not in i1_i2_connections:
                i1_i2_connections[key] = []
            i1_i2_connections[key].append((src_internal_port, tgt_internal_port))

    # Add edges for I1→I2 connections
    for (src_node, tgt_node), port_pairs in i1_i2_connections.items():
        src_ports = [p[0] for p in port_pairs]
        tgt_ports = [p[1] for p in port_pairs]
        G.add_edge(
            src_node,
            tgt_node,
            source_ports=src_ports,
            target_ports=tgt_ports,
            edge_type="contracted_internal",
            internal=True,
            connection_type="I1_I2",
        )

    # Group J2→J1 connections by (src_node, tgt_node)
    j2_j1_connections: dict[tuple, list] = {}  # (src_node, tgt_node) -> list of (src_port, tgt_port)

    # Add internal edges: J2 (outputs of second) → J1 (inputs of first)
    for out_idx, in_idx in zip(diagram.J2, diagram.J1, strict=False):
        src_node, src_internal_port = find_node_by_external_output(diagram.second, out_idx, G)
        tgt_node, tgt_internal_port = find_node_by_external_input(diagram.first, in_idx, G)

        if src_node is not None and tgt_node is not None:
            key = (src_node, tgt_node)
            if key not in j2_j1_connections:
                j2_j1_connections[key] = []
            j2_j1_connections[key].append((src_internal_port, tgt_internal_port))

    # Add edges for J2→J1 connections
    for (src_node, tgt_node), port_pairs in j2_j1_connections.items():
        src_ports = [p[0] for p in port_pairs]
        tgt_ports = [p[1] for p in port_pairs]
        G.add_edge(
            src_node,
            tgt_node,
            source_ports=src_ports,
            target_ports=tgt_ports,
            edge_type="contracted_internal",
            internal=True,
            connection_type="J2_J1",
        )

    return node_id


# =========================================================================
# Diagram to graph helper functions
# =========================================================================


def _reconstruct_from_node(G: nx.DiGraph, node_id: int) -> Diagram:  # ruff: ignore[invalid-argument-name]
    """Reconstruct a diagram from a graph node.

    Parameters
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to reconstruct.

    Returns
    -------
    Diagram
        The reconstructed diagram.

    Raises
    ------
    ValueError
        If the node type is unknown.
    """
    attrs = G.nodes[node_id]
    kind = attrs.get("kind")

    if kind in {"proper", "compact"}:
        return _reconstruct_proper_node(G, node_id)
    if kind == "container":
        container_type = attrs.get("container_type")
        if container_type == "composition":
            return _reconstruct_composition_node(G, node_id)
        if container_type == "tensor":
            return _reconstruct_tensor_node(G, node_id)
        if container_type == "contracted":
            return _reconstruct_contracted_node(G, node_id)
        msg = f"Unknown container type: {container_type}"
        raise ValueError(msg)
    msg = f"Unknown node kind: {kind}"
    raise ValueError(msg)


def _reconstruct_proper_node(G: nx.DiGraph, node_id: int) -> Diagram:  # ruff: ignore[complex-structure, invalid-argument-name, too-many-return-statements, too-many-branches, too-many-locals]
    """Reconstruct a proper diagram from a graph node.

    Parameters
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to reconstruct.

    Returns
    -------
    Diagram
        The reconstructed diagram.

    Raises
    ------
    ValueError
        If the node type is not among proper diagram types.
    """
    attrs = G.nodes[node_id]
    node_type = attrs.get("type")
    phase = attrs.get("phase")
    num_inputs = attrs.get("num_inputs", 0)
    num_outputs = attrs.get("num_outputs", 0)
    is_parametric = isinstance(phase, Expr)
    feedforward = attrs.get("feedforward")
    measurement_ids = attrs.get("measurement_ids")

    if node_type == "QSpider":
        return QSpider(num_inputs, num_outputs, phase if phase is not None else ZxPoly({}))
    if node_type == "PSpider":
        return PSpider(num_inputs, num_outputs, phase if phase is not None else ZxPoly({}))
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
        return DisplacementGate(phase, is_parametric, feedforward, measurement_ids)
    if node_type == "PhaseRotationGate":
        return PhaseRotationGate(phase, is_parametric, feedforward, measurement_ids)
    if node_type == "SqueezingGate":
        return SqueezingGate(phase, is_parametric, feedforward, measurement_ids)
    if node_type == "BeamsplitterGate":
        return BeamsplitterGate(phase, is_parametric, feedforward, measurement_ids)
    if node_type == "ControlledSumGate":
        control = attrs.get("control")
        target = attrs.get("target")
        return ControlledSumGate(phase, control, target, is_parametric, feedforward, measurement_ids)
    if node_type == "ControlledZGate":
        return ControlledZGate(phase, is_parametric, feedforward, measurement_ids)
    if node_type == "CubicPhaseGate":
        return CubicPhaseGate(phase, is_parametric, feedforward, measurement_ids)
    if node_type == "ShearXInvariantGate":
        return ShearXInvariantGate(phase, is_parametric, feedforward, measurement_ids)
    if node_type == "ShearPInvariantGate":
        return ShearPInvariantGate(phase, is_parametric, feedforward, measurement_ids)
    if node_type == "Squeezing45Gate":
        return Squeezing45Gate(phase, is_parametric, feedforward, measurement_ids)
    if node_type == "MeasurementGate":
        return MeasurementGate(phase, is_parametric, feedforward, measurement_ids)
    if node_type == "ArbitraryGate":
        alpha = attrs.get("alpha")
        beta = attrs.get("beta")
        lam = attrs.get("lam")
        arb_parametric = isinstance(alpha, Expr) or isinstance(beta, Expr) or isinstance(lam, Expr)
        return ArbitraryGate(alpha, beta, lam, arb_parametric, feedforward, measurement_ids)
    if node_type == "TwoModeShearGate":
        a = attrs.get("a")
        b = attrs.get("b")
        shear2_parametric = isinstance(a, Expr) or isinstance(b, Expr)
        return TwoModeShearGate(a, b, shear2_parametric, feedforward, measurement_ids)
    msg = f"Unknown proper node type: {node_type}"
    raise ValueError(msg)


def _reconstruct_composition_node(G: nx.DiGraph, node_id: int) -> Diagram:  # ruff: ignore[invalid-argument-name]
    """Reconstruct a CompositionDiagram from a graph node.

    Parameters
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to reconstruct.

    Returns
    -------
    Diagram
        The reconstructed CompositionDiagram.

    Raises
    ------
    ValueError
        If the connectivity is malformed.
    """
    attrs = G.nodes[node_id]
    sub_diagram_ids = attrs.get("sub_diagram_ids", [])
    # Shallow-copy the connectivity dict rather than handing out the
    # graph-owned object directly: several rewrite rules (e.g.
    # `RewriteRule._propagate_arity_to_parent`, `IdentityRule.apply_single`)
    # mutate a container's "connectivity" dict IN PLACE (`connectivity[key]
    # = ...`, `del connectivity[key]`) rather than always replacing it
    # wholesale. A `Diagram` built from this node must stay a genuine,
    # independent snapshot even if the source graph keeps being mutated
    # afterward (e.g. `optimize()` calls `to_diagram(graph)` to capture
    # its pre-cleanup result, then keeps mutating `graph` in
    # `remove_void_and_identity_nodes`) -- without this copy, the earlier
    # snapshot's `connectivity` would silently change underneath it. A
    # shallow copy is enough: no rule mutates one of the per-index INNER
    # dicts in place, only ever the outer one.
    connectivity = dict(attrs.get("connectivity", {}))
    # Recursively reconstruct all sub-diagrams
    sub_diagrams = [_reconstruct_from_node(G, sub_id) for sub_id in sub_diagram_ids]

    # The connectivity in the graph should already be in the correct format:
    # connectivity[left_idx] = {in_port: out_port}
    # This matches the CompositionDiagram constructor format

    # Validate connectivity if present
    if connectivity:
        for left_idx, conn in connectivity.items():
            # Check that left_idx is valid
            if left_idx >= len(sub_diagrams) - 1:
                msg = f"Invalid connectivity key {left_idx}: exceeds number of sub-diagrams"
                raise ValueError(msg)
            # Validate that the connectivity matches the arities
            left_diagram = sub_diagrams[left_idx]
            right_diagram = sub_diagrams[left_idx + 1]

            # Check that the number of connections matches the input ports of the right diagram
            # The keys of conn are input ports of the right diagram
            if set(conn.keys()) != set(range(right_diagram.num_inputs)):
                msg = f"Connectivity for index {left_idx} does not match input ports of sub_diagram {left_idx + 1}"
                raise ValueError(msg)

            # Check that the values are within the output ports of the left diagram
            for out_port in conn.values():
                if out_port < 0 or out_port >= left_diagram.num_outputs:
                    msg = (
                        f"Connectivity for index {left_idx} has invalid output port "
                        f"{out_port} for sub_diagram {left_idx}"
                    )
                    raise ValueError(msg)

    return CompositionDiagram(sub_diagrams, connectivity)


def _reconstruct_tensor_node(G: nx.DiGraph, node_id: int) -> Diagram:  # ruff: ignore[invalid-argument-name]
    """Reconstruct a TensorDiagram from a graph node.

    Parameters
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to reconstruct.

    Returns
    -------
    Diagram
        The reconstructed diagram.
    """
    attrs = G.nodes[node_id]
    sub_diagram_ids = attrs.get("sub_diagram_ids", [])

    # Recursively reconstruct all sub-diagrams
    sub_diagrams = [_reconstruct_from_node(G, sub_id) for sub_id in sub_diagram_ids]

    return TensorDiagram(sub_diagrams)


def _reconstruct_contracted_node(G: nx.DiGraph, node_id: int) -> Diagram:  # ruff: ignore[invalid-argument-name]
    """Reconstruct a ContractedDiagram from a graph node.

    Parameters
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to reconstruct.

    Returns
    -------
    Diagram
        The reconstructed diagram.

    Raises
    ------
    ValueError
        If the ContractedDiagram node misses first_id or second_id.
    """
    attrs = G.nodes[node_id]
    first_id = attrs.get("first_id")
    second_id = attrs.get("second_id")
    I1 = attrs.get("I1", [])  # ruff: ignore[non-lowercase-variable-in-function]
    I2 = attrs.get("I2", [])  # ruff: ignore[non-lowercase-variable-in-function]
    J1 = attrs.get("J1", [])  # ruff: ignore[non-lowercase-variable-in-function]
    J2 = attrs.get("J2", [])  # ruff: ignore[non-lowercase-variable-in-function]

    if first_id is None or second_id is None:
        msg = f"ContractedDiagram node {node_id} missing first_id or second_id"
        raise ValueError(msg)

    first = _reconstruct_from_node(G, first_id)
    second = _reconstruct_from_node(G, second_id)

    return ContractedDiagram(first, second, I1, I2, J1, J2)

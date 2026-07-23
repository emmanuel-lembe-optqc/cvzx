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

import networkx as nx

from mqc3.zx.base_gates import (
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
    ZxPoly,
)


def to_graph(diagram: Diagram) -> nx.DiGraph:
    """Convert a CV ZX diagram to a directed graph representation.

    The graph preserves the full hierarchy of the original diagram:
        - Container nodes (CompositionDiagram, TensorDiagram, ContractedDiagram)
          have kind='container' and store their sub-diagram IDs
        - Proper nodes have kind='proper' and store their attributes
        - Each node stores its immediate container ID (container_id)
        - The root container is marked with is_root=True
        - Edges represent connections between nodes

    Parameters:
    ----------
    diagram : Diagram
        The CV ZX diagram to convert.

    Returns:
    -------
    nx.DiGraph
        Directed graph representation of the diagram with all nodes and edges.
    """
    G = nx.DiGraph()  # noqa: N806
    root_id = _convert_diagram_to_graph(diagram, G, container_id=None, is_root=True)

    # Mark the root node
    if root_id is not None and root_id != -1:
        G.nodes[root_id]["is_root"] = True

    return G


def to_diagram(G: nx.DiGraph) -> Diagram:  # noqa: N803
    """Reconstruct a CV ZX diagram from a directed graph representation.

    This function reconstructs the original diagram from the graph, preserving
    the full hierarchy, container nodes, and all connections.

    Parameters:
    ----------
    G : nx.DiGraph
        The directed graph representation of the diagram.

    Returns:
    -------
    Diagram
        The reconstructed CV ZX diagram.

    Raises:
    ------
    ValueError:
        If the graph is malformed or missing required attributes.
    """
    # Find the root node
    root = get_root_node(G)
    if root is None:
        msg = "No root node found in the graph."
        raise ValueError(msg)

    # Reconstruct the diagram from the root
    return _reconstruct_from_node(G, root)


# =========================================================================
# Diagram to graph helper functions
# =========================================================================


def find_node_by_external_output(
    diagram: Diagram,
    ext_port: int,
    G: nx.DiGraph,  # noqa: N803
) -> tuple[int | None, int | None]:
    """Find the proper node and its internal port for a given external output port.

    This function recursively traverses the diagram structure:
        1. If diagram is proper: returns (node_id, ext_port) (direct mapping)
        2. If diagram is a container: uses its output_mapping to find the sub-diagram
           and internal port, then recurses into that sub-diagram

    Parameters:
    ----------
    diagram : Diagram
        The diagram to search within.
    ext_port : int
        The external output port index.
    G : nx.DiGraph
        The graph containing the nodes.

    Returns:
    -------
    tuple[int | None, int | None]
        (node_id, internal_port) of the proper node handling this external port,
        or (None, None) if not found.
    """
    # If it's a proper diagram, return its node ID and the port
    if isinstance(diagram, ProperDiagram):
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
    elif attrs.get("container_type") == "contracted":
        # For ContractedDiagram: sub_ref is 'first' or 'second'
        sub_diagram = diagram.first if sub_ref == "first" else diagram.second
    else:
        return None, None

    # Recursively traverse into the sub-diagram with the internal port
    return find_node_by_external_output(sub_diagram, internal_port, G)


def find_node_by_external_input(
    diagram: Diagram,
    ext_port: int,
    G: nx.DiGraph,  # noqa: N803
) -> tuple[int | None, int | None]:
    """Find the proper node and its internal port for a given external input port.

    This function recursively traverses the diagram structure:
        1. If diagram is proper: returns (node_id, ext_port) (direct mapping)
        2. If diagram is a container: uses its input_mapping to find the sub-diagram
           and internal port, then recurses into that sub-diagram

    Parameters:
    ----------
    diagram : Diagram
        The diagram to search within.
    ext_port : int
        The external input port index.
    G : nx.DiGraph
        The graph containing the nodes.

    Returns:
    -------
    tuple[int | None, int | None]
        (node_id, internal_port) of the proper node handling this external input port,
        or (None, None) if not found.
    """
    # If it's a proper diagram, return its node ID and the port
    if isinstance(diagram, ProperDiagram):
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
    elif attrs.get("container_type") == "contracted":
        sub_diagram = diagram.first if sub_ref == "first" else diagram.second
    else:
        return None, None

    # Recursively traverse into the sub-diagram with the internal port
    return find_node_by_external_input(sub_diagram, internal_port, G)


def get_proper_nodes(G: nx.DiGraph) -> list[int]:  # noqa: N803
    """Get all proper nodes (leaf operations) from the graph.

    Proper nodes represent actual operations (spiders, gates, etc.)
    and have kind='proper'.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph to search for proper nodes.

    Returns:
    -------
    list[int]
        List of node IDs for all proper nodes in the graph.
    """
    return [n for n, attrs in G.nodes(data=True) if attrs.get("kind") == "proper"]


def get_container_nodes(G: nx.DiGraph) -> list[int]:  # noqa: N803
    """Get all container nodes from the graph.

    Container nodes represent structural elements (CompositionDiagram,
    TensorDiagram, ContractedDiagram) and have kind='container'.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph to search for container nodes.

    Returns:
    -------
    list[int]
        List of node IDs for all container nodes in the graph.
    """
    return [n for n, attrs in G.nodes(data=True) if attrs.get("kind") == "container"]


def get_root_node(G: nx.DiGraph) -> int | None:  # noqa: N803
    """Get the root node (the outermost diagram) from the graph.

    The root node is the container node with is_root=True.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph to search for the root node.

    Returns:
    -------
    int | None
        The node ID of the root container, or None if no root is found.
    """
    for node, attrs in G.nodes(data=True):
        if attrs.get("is_root", False):
            return node
    return None


def get_immediate_container(G: nx.DiGraph, node_id: int) -> int | None:  # noqa: N803
    """Get the immediate container of a node.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to find the container for.

    Returns:
    -------
    int | None
        The node ID of the immediate container, or None if the node is the root.
    """
    attrs = G.nodes[node_id]
    return attrs.get("container_id")


def get_sub_diagrams(G: nx.DiGraph, container_node: int) -> list[int]:  # noqa: N803
    """Get the sub-diagram node IDs of a container node.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph containing the container node.
    container_node : int
        The node ID of the container node.

    Returns:
    -------
    list[int]
        List of sub-diagram node IDs, or an empty list if the node is not
        a container or has no sub-diagrams.
    """
    attrs = G.nodes[container_node]
    if attrs.get("kind") != "container":
        return []
    return attrs.get("sub_diagram_ids", [])


def get_connectivity(G: nx.DiGraph, container_node: int) -> dict | None:  # noqa: N803
    """Get the connectivity dictionary of a CompositionDiagram container node.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph containing the container node.
    container_node : int
        The node ID of the composition container node.

    Returns:
    -------
    dict | None
        The connectivity dictionary if the node is a CompositionDiagram,
        otherwise None.
    """
    attrs = G.nodes[container_node]
    if attrs.get("container_type") != "composition":
        return None
    return attrs.get("connectivity")


def get_contracted_connections(G: nx.DiGraph, container_node: int) -> dict | None:  # noqa: N803
    """Get the contracted connections of a ContractedDiagram container node.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph containing the container node.
    container_node : int
        The node ID of the contracted container node.

    Returns:
    -------
    dict | None
        A dictionary with keys 'I1', 'I2', 'J1', 'J2' if the node is a
        ContractedDiagram, otherwise None.
    """
    attrs = G.nodes[container_node]
    if attrs.get("container_type") != "contracted":
        return None
    return {
        "I1": attrs.get("I1", []),
        "I2": attrs.get("I2", []),
        "J1": attrs.get("J1", []),
        "J2": attrs.get("J2", []),
    }


def get_nodes_by_container(G: nx.DiGraph, container_node: int) -> list[int]:  # noqa: N803
    """Get all nodes that belong to a specific container.

    This includes both proper nodes and nested container nodes.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph containing the nodes.
    container_node : int
        The node ID of the container.

    Returns:
    -------
    list[int]
        List of node IDs belonging to the container.
    """
    return [n for n, attrs in G.nodes(data=True) if attrs.get("container_id") == container_node]


def _convert_diagram_to_graph(
    diagram: Diagram,
    G: nx.DiGraph,  # noqa: N803
    container_id: int | None,
    is_root: bool = False,  # noqa: FBT001, FBT002
) -> int:
    """Convert a diagram to graph nodes and return the root node ID.

    This function recursively converts a diagram and its sub-diagrams
    to graph nodes. It returns the node ID of the root diagram.

    Parameters:
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

    Returns:
    -------
    int
        The node ID of the root diagram in the graph, or -1 if the diagram
        has no representation.
    """
    if isinstance(diagram, ProperDiagram):
        return _add_proper_node(diagram, G, container_id)
    if isinstance(diagram, CompositionDiagram):
        return _add_composition_node(diagram, G, container_id, is_root)
    if isinstance(diagram, TensorDiagram):
        return _add_tensor_node(diagram, G, container_id, is_root)
    if isinstance(diagram, ContractedDiagram):
        return _add_contracted_node(diagram, G, container_id, is_root)
    return -1


def _add_proper_node(
    diagram: ProperDiagram,
    G: nx.DiGraph,  # noqa: N803
    container_id: int | None,
) -> int:
    """Add a proper diagram as a node in the graph.

    Proper nodes have kind='proper' and store all relevant attributes
    including type, phase, port information, and its container ID.

    Parameters:
    ----------
    diagram : ProperDiagram
        The proper diagram to add as a node.
    G : nx.DiGraph
        The graph to add the node to (modified in place).
    container_id : int | None
        The node ID of the immediate container of this proper diagram.

    Returns:
    -------
    int
        The node ID of the added proper node.
    """
    node_id = diagram.id

    phase = getattr(diagram, "phase", None)
    node_type = diagram.__class__.__name__

    G.add_node(
        node_id,
        id=node_id,
        type=node_type,
        kind="proper",
        phase=phase,
        n_inputs=diagram.num_inputs,
        n_outputs=diagram.num_outputs,
        diagram=diagram,
        container_id=container_id,
        # Store external port mappings
        external_inputs=list(range(diagram.num_inputs)),
        external_outputs=list(range(diagram.num_outputs)),
    )
    return node_id


def _add_composition_node(
    diagram: CompositionDiagram,
    G: nx.DiGraph,  # noqa: N803
    container_id: int | None,
    is_root: bool = False,  # noqa: FBT001, FBT002
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

    Parameters:
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

    Returns:
    -------
    int
        The node ID of the added composition container node.

    Notes:
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

    G.add_node(
        node_id,
        id=node_id,
        type="CompositionDiagram",
        kind="container",
        container_type="composition",
        phase=None,
        n_inputs=diagram.num_inputs,
        n_outputs=diagram.num_outputs,
        diagram=diagram,
        container_id=container_id,
        is_root=is_root,
        sub_diagram_ids=sub_node_ids,
        connectivity=diagram.connectivity,
        external_inputs=list(range(diagram.num_inputs)),
        external_outputs=list(range(diagram.num_outputs)),
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
    G: nx.DiGraph,  # noqa: N803
    container_id: int | None,
    is_root: bool = False,  # noqa: FBT001, FBT002
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

    Parameters:
    ----------
    diagram : TensorDiagram
        The tensor diagram to add as a container node.
    G : nx.DiGraph
        The graph to add nodes to (modified in place).
    container_id : int | None
        The node ID of the immediate container of this tensor.
    is_root : bool, default=False
        Whether this is the root (outermost) diagram.

    Returns:
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
        n_inputs=diagram.num_inputs,
        n_outputs=diagram.num_outputs,
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


def _add_contracted_node(  # noqa: C901
    diagram: ContractedDiagram,
    G: nx.DiGraph,  # noqa: N803
    container_id: int | None,
    is_root: bool = False,  # noqa: FBT001, FBT002
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

    Parameters:
    ----------
    diagram : ContractedDiagram
        The contracted diagram to add as a container node.
    G : nx.DiGraph
        The graph to add nodes and edges to (modified in place).
    container_id : int | None
        The node ID of the immediate container of this contraction.
    is_root : bool, default=False
        Whether this is the root (outermost) diagram.

    Returns:
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
        n_inputs=diagram.num_inputs,
        n_outputs=diagram.num_outputs,
        diagram=diagram,
        container_id=container_id,
        is_root=is_root,
        first_id=first_node_id,
        second_id=second_node_id,
        I1=list(diagram.I1),
        I2=list(diagram.I2),
        J1=list(diagram.J1),
        J2=list(diagram.J2),
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


def _reconstruct_from_node(G: nx.DiGraph, node_id: int) -> Diagram:  # noqa: N803
    """Reconstruct a diagram from a graph node.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to reconstruct.

    Returns:
    -------
    Diagram
        The reconstructed diagram.

    Raises:
    ------
    ValueError:
        If the node type is unknown.
    """
    attrs = G.nodes[node_id]
    kind = attrs.get("kind")

    if kind == "proper":
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


def _reconstruct_proper_node(G: nx.DiGraph, node_id: int) -> Diagram:  # noqa: N803
    """Reconstruct a proper diagram from a graph node.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to reconstruct.

    Returns:
    -------
    Diagram
        The reconstructed diagram.

    Raises:
    ------
    ValueError:
        If the node type is not among proper diagram types.
    """
    attrs = G.nodes[node_id]
    node_type = attrs.get("type")
    phase = attrs.get("phase")
    n_inputs = attrs.get("n_inputs", 0)
    n_outputs = attrs.get("n_outputs", 0)

    if node_type == "QSpider":
        return QSpider(n_inputs, n_outputs, phase if phase is not None else ZxPoly({}))
    if node_type == "PSpider":
        return PSpider(n_inputs, n_outputs, phase if phase is not None else ZxPoly({}))
    if node_type == "Swap":
        return Swap()
    if node_type == "Fourier":
        return Fourier()
    if node_type == "FourierInv":
        return FourierInv()
    if node_type == "Fourier2":
        return Fourier2()
    msg = f"Unknown proper node type: {node_type}"
    raise ValueError(msg)


def _reconstruct_composition_node(G: nx.DiGraph, node_id: int) -> Diagram:  # noqa: N803
    """Reconstruct a CompositionDiagram from a graph node.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to reconstruct.

    Returns:
    -------
    Diagram
        The reconstructed CompositionDiagram.

    Raises:
    ------
    ValueError:
        If the connectivity is malformed.
    """
    attrs = G.nodes[node_id]
    sub_diagram_ids = attrs.get("sub_diagram_ids", [])
    connectivity = attrs.get("connectivity", {})

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


def _reconstruct_tensor_node(G: nx.DiGraph, node_id: int) -> Diagram:  # noqa: N803
    """Reconstruct a TensorDiagram from a graph node.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to reconstruct.

    Returns:
    -------
    Diagram
        The reconstructed diagram.
    """
    attrs = G.nodes[node_id]
    sub_diagram_ids = attrs.get("sub_diagram_ids", [])

    # Recursively reconstruct all sub-diagrams
    sub_diagrams = [_reconstruct_from_node(G, sub_id) for sub_id in sub_diagram_ids]

    return TensorDiagram(sub_diagrams)


def _reconstruct_contracted_node(G: nx.DiGraph, node_id: int) -> Diagram:  # noqa: N803
    """Reconstruct a ContractedDiagram from a graph node.

    Parameters:
    ----------
    G : nx.DiGraph
        The graph containing the node.
    node_id : int
        The node ID to reconstruct.

    Returns:
    -------
    Diagram
        The reconstructed diagram.

    Raises:
    ------
    ValueError:
        If the ContractedDiagram node misses first_id or second_id.
    """
    attrs = G.nodes[node_id]
    first_id = attrs.get("first_id")
    second_id = attrs.get("second_id")
    I1 = attrs.get("I1", [])  # noqa: N806
    I2 = attrs.get("I2", [])  # noqa: N806
    J1 = attrs.get("J1", [])  # noqa: N806
    J2 = attrs.get("J2", [])  # noqa: N806

    if first_id is None or second_id is None:
        msg = f"ContractedDiagram node {node_id} missing first_id or second_id"
        raise ValueError(msg)

    first = _reconstruct_from_node(G, first_id)
    second = _reconstruct_from_node(G, second_id)

    return ContractedDiagram(first, second, I1, I2, J1, J2)

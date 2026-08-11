"""CV ZX calculus rewrite rules and applications.

This module implements the 10 basic rewrite rules from [3] Sec. IV.A,
and the derived rules from Sec. IV.B; using a graph structure.

References:
-----------
[3] Nagayoshi et al., CV ZX calculus, 2024
"""

from abc import ABC, abstractmethod

import networkx as nx

from mqc3.zx.base_gates import Diagram
from mqc3.zx.nx_graph import (
    to_diagram,
    to_graph,
)


class RewriteRule(ABC):
    """Base class for rewrite rules operating on graphs.

    All rewrite rules operate on `nx.DiGraph` in-place. This avoids repeated
    conversions between diagram and graph representations.

    The typical workflow is:
        1. Convert diagram to graph: `G = to_graph(diagram)`
        2. Apply rules: `G = rule1.apply_rule(G); G = rule2.apply_rule(G)`
        3. Convert back: `diagram = to_diagram(G)`

    Subclasses must implement:
        - `match(G)`: Find all matches in the graph
        - `apply_single(G, match)`: Apply a single match in-place

    The `apply_rule` method provides a default implementation that:
        1. Finds all matches
        2. Processes them in order (deepest to shallowest)
        3. Returns the modified graph

    Parameters:
    ----------
    G : nx.DiGraph
        The graph to modify.

    Returns:
    -------
    nx.DiGraph
        The modified graph (same object, for chaining).
    """

    @abstractmethod
    def match(self, G: nx.DiGraph) -> list:  # noqa: N803
        """Find all matches of the rule pattern in the graph.

        Parameters:
        ----------
        G : nx.DiGraph
            The graph to search.

        Returns:
        -------
        list
            List of matches. Each match must be a dictionary or object
            that can be passed to `apply_single`.
        """

    @abstractmethod
    def apply_single(self, G: nx.DiGraph, match: any) -> None:  # noqa: N803
        """Apply the rule to a specific match in-place.

        Parameters:
        ----------
        G : nx.DiGraph
            The graph to modify.
        match : any
            A match returned by `match()`.
        """

    def apply_rule(self, G: nx.DiGraph) -> nx.DiGraph:  # noqa: N803
        """Apply a rule to the entire graph.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.

        Returns:
        -------
        Diagram
            The modified diagram.
        """
        # Find all matches
        matches = self.match(G)

        if not matches:
            return G

        # Process matches
        for match in matches:
            result = self.apply_single(G, match)

        return result

    def _flatten_container(self, G: nx.DiGraph, container_id: int) -> None:  # noqa: N803
        """Flatten a container if it has only one element.

        Parameters:
        ----------
        G : nx.DiGraph
            The graph to modify.
        container_id : int
            The container node ID.
        """
        if container_id not in G.nodes:
            return

        attrs = G.nodes[container_id]
        sub_ids = attrs.get("sub_diagram_ids", [])

        # If the container has exactly one element, flatten it
        if len(sub_ids) == 1:
            child_id = sub_ids[0]

            # Get the container's container
            parent_container_id = attrs.get("container_id")

            # Replace the container with its child in the parent
            if parent_container_id is not None and parent_container_id in G.nodes:
                parent_attrs = G.nodes[parent_container_id]
                if parent_attrs.get("container_type") in {"composition", "tensor"}:
                    parent_sub_ids = parent_attrs.get("sub_diagram_ids", [])
                    # Replace container_id with child_id in parent's sub_diagram_ids
                    idx = parent_sub_ids.index(container_id)
                    parent_sub_ids[idx] = child_id
                # Replace container_id with child_id
                elif parent_attrs.get("first_id") == container_id:
                    parent_attrs["first_id"] = child_id
                else:
                    parent_attrs["second_id"] = child_id

                # Update container_id of the child
                if child_id in G.nodes:
                    G.nodes[child_id]["container_id"] = parent_container_id
                # Remove the container node
                G.remove_node(container_id)

            elif child_id in G.nodes:
                # The Container was the root
                G.nodes[child_id]["container_id"] = None
                G.nodes[child_id]["is_root"] = True


class IdentityRule(RewriteRule):
    r"""Identity rule (id) from [3] Eq. (69) - Graph-based version.

    A q-spider with no phase and 1 input/1 output is the identity.
    A p-spider with no phase and 1 input/1 output is also the identity.

    The rule applies to identity spiders anywhere in the graph.
    After removal, containers with a single element are flattened.
    """

    def match(self, G: nx.DiGraph) -> list[dict]:  # noqa: N803
        """Find all identity spiders in the graph.

        Parameters:
        ----------
        G : nx.DiGraph
            The graph to search.

        Returns:
        -------
        list[dict]
            List of matches, each containing:
            - 'node_id': the node ID of the identity spider
            - 'container_id': the node ID of its immediate container
        """
        matches = []

        for node, attrs in G.nodes(data=True):
            if (
                attrs.get("kind") == "proper"
                and is_wiring_node_from_attrs(attrs)
                and G.nodes[attrs["container_id"]]["container_type"] == "composition"
            ):
                matches.append({
                    "node_id": node,
                    "container_id": attrs.get("container_id"),
                })

        return matches

    def apply_single(self, graph: nx.DiGraph, match: dict) -> None:
        """Remove an identity spider from the graph in-place.

        Parameters:
        ----------
        G : nx.DiGraph
            The graph to modify.
        match : dict
            Match containing 'node_id' and 'container_id'.
        """
        node_id = match["node_id"]
        attrs = graph.nodes[node_id]
        container_id = match["container_id"]

        # Check that the node is an identity spider which can be reduced
        if not (
            attrs.get("kind") == "proper"
            and is_wiring_node_from_attrs(attrs)
            and graph.nodes[container_id]["container_type"] == "composition"
        ):
            return
        # Remove the identity node from the graph
        attrs = graph.nodes[container_id]
        sub_ids = attrs.get("sub_diagram_ids", [])
        if len(sub_ids) > 1:
            idx = sub_ids.index(node_id)
            connectivity = graph.nodes[container_id]["connectivity"]
            if idx < len(sub_ids) - 1:
                graph = nx.contracted_nodes(graph, u=sub_ids[idx + 1], v=node_id, self_loops=False, copy=False)
                del graph.nodes[sub_ids[idx + 1]]["contraction"]
                new_connectivity = {}
                idx_to_del = idx - 1 if idx > 0 else idx
                del connectivity[idx_to_del]
                # Update the keys after deleting the obsolete key
                for key in connectivity:
                    if key > idx_to_del:
                        new_connectivity[key - 1] = connectivity[key]
                    else:
                        new_connectivity[key] = connectivity[key]
                graph.nodes[container_id]["connectivity"] = new_connectivity
            else:
                graph = nx.contracted_nodes(graph, u=sub_ids[idx - 1], v=node_id, self_loops=False, copy=False)
                del graph.nodes[sub_ids[idx - 1]]["contraction"]
                # We must update the connectivity
                del connectivity[idx - 1]
            # We must update sub_diagram_ids
            graph.nodes[container_id]["sub_diagram_ids"].pop(idx)
            if len(sub_ids) == 1:
                self._flatten_container(graph, container_id)


def is_wiring_node_from_attrs(attrs: dict) -> bool:
    """Check if a node's attributes represent a wiring (identity) diagram.

    A wiring diagram is:
        - A QSpider or PSpider
        - With 1 input and 1 output
        - With zero phase

    Parameters:
    ----------
    attrs : dict
        Node attributes from the graph.

    Returns:
    -------
    bool
        True if the node represents an identity spider.
    """
    node_type = attrs.get("type")
    if node_type not in {"QSpider", "PSpider"}:
        return False

    n_inputs = attrs.get("n_inputs", 0)
    n_outputs = attrs.get("n_outputs", 0)
    if n_inputs != 1 or n_outputs != 1:
        return False

    phase = attrs.get("phase")
    if phase is None:
        return False

    return phase.is_zero()


def apply_rule_to_diagram(rule: RewriteRule, diagram: Diagram) -> Diagram:
    """Apply a rewrite rule to a diagram, converting to graph and back.

    Parameters:
    ----------
    rule : RewriteRule
        The rule to apply.
    diagram : Diagram
        The diagram to modify.

    Returns:
    -------
    Diagram
        The modified diagram.
    """
    graph = to_graph(diagram)
    rule.apply_rule(graph)
    return to_diagram(graph)

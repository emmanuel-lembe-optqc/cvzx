"""CV ZX calculus rewrite rules and applications.

This module implements the 10 basic rewrite rules from [3] Sec. IV.A,
and the derived rules from Sec. IV.B; using a graph structure.

References:
-----------
[3] Nagayoshi et al., CV ZX calculus, 2024
"""

from abc import ABC, abstractmethod

import networkx as nx

from cvzx.base_gates import Diagram, ZxPoly
from cvzx.nx_graph import (
    GateRegister,
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

    def apply_rule(self, G: nx.DiGraph, registry: GateRegister) -> nx.DiGraph:  # noqa: N803
        """Apply a rule to the entire graph.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.

        registry : GateRegister
            Registry for tracking specific gate types and nodes in a CV ZX graph.

        Returns:
        -------
        Diagram
            The modified diagram.
        """
        # Find all matches
        matches = self.match(G, registry)

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

    def match(self, graph: nx.DiGraph, registry: GateRegister) -> list[dict]:
        """Find all identity spiders in the graph.

        Parameters:
        ----------
        graph : nx.DiGraph
            The graph to search.

        registry : GateRegister
            Registry for tracking specific gate types and nodes in a CV ZX graph.

        Returns:
        -------
        list[dict]
            List of matches, each containing:
            - 'node_id': the node ID of the identity spider
            - 'container_id': the node ID of its immediate container
        """
        matches = []

        for node in registry.identity_spiders:
            attrs = graph.nodes[node]
            if (
                attrs.get("kind") == "proper"
                and is_wiring_node_from_attrs(attrs)
                and graph.nodes[attrs["container_id"]]["container_type"] == "composition"
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
        graph : nx.DiGraph
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


class FusionRule(RewriteRule):
    r"""Fusion rule (f) from [3] Eq. (70) & (71) - Graph-based version.

    Two same-type spiders connected by wires can be fused into a single spider.
    For Q-spiders: two q-spiders connected by wires can be fused with phase addition.
    For P-spiders: two p-spiders connected by wires can be fused with phase addition.

    The rule applies when:
    - Both spiders are the same type (both Q or both P)
    - They are in a ContractedDiagram container
    - They are connected via some wires (I1 and I2 matching J1 and J2)
    - The resulting spider has:
        inputs = inputs of first + inputs of second (minus connected wires)
        outputs = outputs of first + outputs of second (minus connected wires)
    - Phase is the sum of the two phases

    The rule applies to ContractedDiagram containers anywhere in the graph.
    After fusion, containers with a single element are flattened.
    """

    def match(self, graph: nx.DiGraph, registry: GateRegister) -> list[dict]:  # noqa: PLR0914
        """Find all ContractedDiagram containers containing fusible spiders.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to search.

        registry : GateRegister
            Registry for tracking specific gate types and nodes in a CV ZX graph.

        Returns:
        -------
        list[dict]
            List of matches, each containing:
            - 'contracted_id': the node ID of the ContractedDiagram
            - 'first_id': the node ID of the first spider
            - 'second_id': the node ID of the second spider
            - 'kept_first_inputs': list of input indices kept from first spider
            - 'kept_second_inputs': list of input indices kept from second spider
            - 'kept_first_outputs': list of output indices kept from first spider
            - 'kept_second_outputs': list of output indices kept from second spider
            - 'J1': connections from first spider outputs to second spider inputs
            - 'I1': connections from second spider outputs to first spider inputs
            - 'J2': connections from first spider inputs to second spider outputs
            - 'I2': connections from second spider inputs to first spider outputs
        """
        matches = []

        for node in registry.contracted_diagrams:
            # Check if this is a ContractedDiagram container
            # Get the first and second diagrams
            attrs = graph.nodes[node]
            first_id = attrs.get("first_id")
            second_id = attrs.get("second_id")

            if first_id is None or second_id is None:
                continue

            # Check if both are spiders (QSpider or PSpider)
            first_attrs = graph.nodes[first_id]
            second_attrs = graph.nodes[second_id]

            first_type = first_attrs.get("type")
            second_type = second_attrs.get("type")

            # Both must be spiders of the same type
            if not (first_type in {"QSpider", "PSpider"} and second_type in {"QSpider", "PSpider"}):
                continue

            if first_type != second_type:
                continue

            # Get the connectivity information
            J1 = attrs.get("J1", [])  # first outputs to second inputs  # noqa: N806
            I1 = attrs.get("I1", [])  # second outputs to first inputs  # noqa: N806
            J2 = attrs.get("J2", [])  # first inputs to second outputs  # noqa: N806
            I2 = attrs.get("I2", [])  # second inputs to first outputs  # noqa: N806

            # Check if there is at least one connection between the spiders
            has_connection = (len(J1) > 0 and len(J2) > 0) or (len(I1) > 0 and len(I2) > 0)

            if not has_connection:
                continue

            # Get kept inputs and outputs
            kept_first_inputs = attrs.get("kept_first_inputs", [])
            kept_second_inputs = attrs.get("kept_second_inputs", [])
            kept_first_outputs = attrs.get("kept_first_outputs", [])
            kept_second_outputs = attrs.get("kept_second_outputs", [])

            # Store the match
            matches.append({
                "contracted_id": node,
                "first_id": first_id,
                "second_id": second_id,
                "first_type": first_type,
                "kept_first_inputs": kept_first_inputs,
                "kept_second_inputs": kept_second_inputs,
                "kept_first_outputs": kept_first_outputs,
                "kept_second_outputs": kept_second_outputs,
                "J1": J1,
                "I1": I1,
                "J2": J2,
                "I2": I2,
            })

        return matches

    def apply_single(self, graph: nx.DiGraph, match: dict) -> None:  # noqa: PLR0914
        """Fuse two same-type spiders in a ContractedDiagram in-place.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        match : dict
            Match containing the ContractedDiagram and spider information.
        """
        contracted_id = match["contracted_id"]
        first_id = match["first_id"]
        second_id = match["second_id"]
        first_type = match["first_type"]

        # Get attributes
        contracted_attrs = graph.nodes[contracted_id]
        first_attrs = graph.nodes[first_id]
        second_attrs = graph.nodes[second_id]

        # Get phases and sum them
        first_phase = first_attrs.get("phase")
        second_phase = second_attrs.get("phase")

        if first_phase is None or second_phase is None:
            return  # Cannot fuse without phases

        # Compute new phase (sum of both phases)
        new_phase = first_phase + second_phase

        # Compute new arities
        # Inputs = kept inputs from first + kept inputs from second
        # (minus connected wires which are removed)
        new_num_inputs = len(match["kept_first_inputs"]) + len(match["kept_second_inputs"])
        new_num_outputs = len(match["kept_first_outputs"]) + len(match["kept_second_outputs"])

        # Determine spider type
        fused_type = "QSpider" if first_type == "QSpider" else "PSpider"

        # Get parent container info before modifying
        parent_container_id = contracted_attrs.get("container_id")
        is_root = contracted_attrs.get("is_root", False)

        # Contract the two spiders into one
        # This preserves all connections from both nodes
        nx.contracted_nodes(graph, first_id, second_id, self_loops=False, copy=False)
        del graph.nodes[first_id]["contraction"]

        # Now update the contracted node (first_id) with fused attributes
        graph.nodes[first_id].update({
            "type": fused_type,
            "num_inputs": new_num_inputs,
            "num_outputs": new_num_outputs,
            "phase": new_phase,
            "container_id": parent_container_id,
            "is_root": is_root,
        })

        # Now remove the ContractedDiagram container
        # First, update the parent container to point to first_id instead of contracted_id
        if parent_container_id is not None and parent_container_id in graph.nodes:
            parent_attrs = graph.nodes[parent_container_id]
            parent_type = parent_attrs.get("container_type")

            if parent_type in {"composition", "tensor"}:
                # Replace in sub_diagram_ids
                sub_ids = parent_attrs.get("sub_diagram_ids", [])
                if contracted_id in sub_ids:
                    idx = sub_ids.index(contracted_id)
                    sub_ids[idx] = first_id
                    graph.nodes[parent_container_id]["sub_diagram_ids"] = sub_ids
                    # Update container_id of the fused spider
                    graph.nodes[first_id]["container_id"] = parent_container_id

            elif parent_type == "contracted":
                # Replace in first_id or second_id
                if parent_attrs.get("first_id") == contracted_id:
                    parent_attrs["first_id"] = first_id
                elif parent_attrs.get("second_id") == contracted_id:
                    parent_attrs["second_id"] = first_id
                # Update container_id of the fused spider
                graph.nodes[first_id]["container_id"] = parent_container_id

        else:
            # ContractedDiagram was root
            graph.nodes[first_id]["is_root"] = True
            graph.nodes[first_id]["container_id"] = None

        # Remove the ContractedDiagram container node
        if contracted_id in graph.nodes:
            graph.remove_node(contracted_id)


class ChainReductionRule(RewriteRule):
    """Simplifies chains of identical gates using algebraic identities - Graph-based version.

    Reduces adjacent same-type gates in CompositionDiagram:
    - Q(u(x)) ∘ Q(v(x)) → Q((u+v)(x))  (any polynomials, no degree restriction)
    - P(u(x)) ∘ P(v(x)) → P((u+v)(x))  (any polynomials, no degree restriction)
    - R(θ) ∘ R(φ) → R(θ+φ)
    - BS(θ) ∘ BS(φ) → BS(θ+φ)
    - Sq(τ) ∘ Sq(κ) → Sq(τ·κ)
    - D(a) ∘ D(β) → D(a+β)
    - F ∘ F → F², F ∘ F ∘ F ∘ F → I, etc.
    - Finv ∘ Finv → F², Finv ∘ Finv ∘ Finv ∘ Finv → I, etc.
    - F ∘ Finv → I, Finv ∘ F → I
    - F² ∘ F² → I, F² ∘ F² ∘ F² → F², etc.
    - ControlledZGate(g1) ∘ ControlledZGate(g2) → ControlledZGate(g1+g2)
    - ControlledSumGate(i,j,g1) ∘ ControlledSumGate(i,j,g2) → ControlledSumGate(i,j,g1+g2)
    """

    def match(self, graph: nx.DiGraph, registry: GateRegister) -> list[dict]:
        """Find all chains of reducible gates in CompositionDiagram containers.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to search.
        registry : GateRegister
            Registry for tracking specific gate types and nodes.

        Returns:
        -------
        list[dict]
            List of matches, each containing:
            - 'container_id': the node ID of the CompositionDiagram
            - 'gate_type': type of gates in the chain
            - 'values': list of values for each gate in the chain
            - 'node_ids': list of node IDs in the chain (in order)
        """
        matches = []

        # Iterate over all CompositionDiagram containers
        for container_id in registry.composition_nodes:
            attrs = graph.nodes[container_id]
            sub_ids = attrs.get("sub_diagram_ids", [])

            if not sub_ids:
                continue

            # Find chains in this CompositionDiagram
            chains = self.find_chains_in_composition(graph, sub_ids)

            matches.extend(
                {
                    "container_id": container_id,
                    "gate_type": chain["gate_type"],
                    "values": chain["values"],
                    "node_ids": chain["node_ids"],
                    "gate_info": chain["gate_info"],
                }
                for chain in chains
            )

        return matches

    def apply_single(self, graph: nx.DiGraph, match: dict) -> None:
        """Apply chain reduction to a specific match in-place.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        match : dict
            Match containing chain information.
        """
        container_id = match["container_id"]
        gate_type = match["gate_type"]
        values = match["values"]
        node_ids = match["node_ids"]
        gate_info = match["gate_info"]

        # Get container attributes
        container_attrs = graph.nodes[container_id]
        sub_ids = container_attrs.get("sub_diagram_ids", [])

        # Reduce the chain based on gate type
        reduced_gate = self.reduce_chain(gate_type, values, gate_info)

        # Get the first node in the chain (will absorb the others)
        first_node = node_ids[0]

        # Contract all nodes in the chain into the first node
        for i in range(1, len(node_ids), 1):
            nx.contracted_nodes(graph, first_node, node_ids[i], self_loops=False, copy=False)
            if "contraction" in graph.nodes[first_node]:
                del graph.nodes[first_node]["contraction"]

        # Update the first node with reduced gate attributes
        self._update_node_for_reduced_gate(graph, first_node, reduced_gate, container_attrs)

        # Update sub_diagram_ids: keep only the first node, remove the rest
        new_sub_ids = []
        removed_connectivity = []
        for i, sub_id in enumerate(sub_ids):
            if sub_id in node_ids:
                if sub_id == first_node:
                    new_sub_ids.append(first_node)
                else:
                    removed_connectivity.append(i - 1)
                # Skip other nodes in chain
            else:
                new_sub_ids.append(sub_id)

        graph.nodes[container_id]["sub_diagram_ids"] = new_sub_ids
        graph.nodes[container_id]["sub_diagram_indices"] = list(range(len(new_sub_ids)))
        # Update connectivity
        self._update_connectivity_after_reduction(graph, container_id, removed_connectivity)

        # Flatten if container has only one element
        if len(new_sub_ids) == 1:
            self._flatten_container(graph, container_id)

    def find_chains_in_composition(self, graph: nx.DiGraph, sub_ids: list[int]) -> list[dict]:
        """Find all maximal chains of reducible gates in a CompositionDiagram.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph containing the nodes.
        sub_ids : list[int]
            List of sub-diagram IDs in the composition.
        container_id : int
            The container node ID.

        Returns:
        -------
        list[dict]
            List of chains, each containing:
            - 'start_node': first node in the chain
            - 'gate_type': type of gates
            - 'values': list of values for each gate
            - 'node_ids': list of node IDs in order
        """
        chains = []
        i = 0

        while i < len(sub_ids):
            node_id = sub_ids[i]
            gate_type, value, gate_info = self.get_gate_info(graph, node_id)
            if gate_type is not None:
                values = [value]
                node_ids = [node_id]
                j = i + 1

                # Check if this is a reducible chain
                while j < len(sub_ids):
                    next_node_id = sub_ids[j]
                    next_type, next_value, next_gate_info = self.get_gate_info(graph, next_node_id)
                    if self.can_chain(gate_type, value, gate_info, next_type, next_value, next_gate_info):
                        values.append(next_value)
                        node_ids.append(next_node_id)
                        j += 1
                        # {F, Finv} chains are only of size 2
                        if {value, next_value} == {"F", "Finv"}:
                            gate_type = "F_pair"
                            break
                    else:
                        next_node_id = sub_ids[j - 1]
                        next_type, next_value, next_gate_info = self.get_gate_info(graph, next_node_id)
                        break

                # Record the chain if it has at least 2 gates
                if len(values) >= 2:  # noqa: PLR2004
                    if gate_type in {"Q", "P"}:
                        gate_info["num_outputs"] = next_gate_info["num_outputs"]
                    chains.append({
                        "gate_type": gate_type,
                        "values": values,
                        "node_ids": node_ids,
                        "gate_info": gate_info,
                    })

                i = max(i + 1, j)
            else:
                i += 1

        return chains

    def get_gate_info(self, graph: nx.DiGraph, node_id: int) -> tuple[str | None, any, dict | None]:  # noqa: C901, PLR0911
        """Extract gate type and value from a node.

        Returns:
        -------
        tuple[str | None, any, dict | None]
            (gate_type, value, gate_info)
            gate_type: 'Q', 'P', 'R', 'BS', 'Sq', 'D', 'F', 'F2', 'ControlledZGate', 'ControlledSumGate', or None
            value: the parameter value (phase polynomial, angle, etc.)
            gate_info: additional info (arities, control/target, etc.)
        """
        attrs = graph.nodes[node_id]
        node_type = attrs.get("type")

        # Q-Spider
        if node_type == "QSpider":
            phase = attrs.get("phase")
            num_inputs = attrs.get("num_inputs")
            num_outputs = attrs.get("num_outputs")
            return ("Q", phase, {"num_inputs": num_inputs, "num_outputs": num_outputs})

        # P-Spider
        if node_type == "PSpider":
            phase = attrs.get("phase")
            num_inputs = attrs.get("num_inputs")
            num_outputs = attrs.get("num_outputs")
            return ("P", phase, {"num_inputs": num_inputs, "num_outputs": num_outputs})

        # Phase Rotation Gate
        if node_type == "PhaseRotationGate":
            theta = attrs.get("phase")
            return ("R", theta, None)

        # Beamsplitter Gate
        if node_type == "BeamsplitterGate":
            theta = attrs.get("phase")
            return ("BS", theta, None)

        # Squeezing Gate
        if node_type == "SqueezingGate":
            tau = attrs.get("phase")
            return ("Sq", tau, None)

        # Displacement Gate
        if node_type == "DisplacementGate":
            alpha = attrs.get("phase")
            return ("D", alpha, None)

        # Fourier Gates
        if node_type == "Fourier":
            return ("F", "F", None)
        if node_type == "FourierInv":
            return ("F", "Finv", None)
        if node_type == "Fourier2":
            return ("F2", "F2", None)

        # Controlled-Z Gate
        if node_type == "ControlledZGate":
            gain = attrs.get("phase")
            return ("CZ", gain, None)

        # Controlled-Sum Gate
        if node_type == "ControlledSumGate":
            gain = attrs.get("phase")
            control = attrs.get("control")
            target = attrs.get("target", 1)
            return ("CSUM", gain, {"control": control, "target": target})

        return (None, None, None)

    def can_chain(  # noqa: PLR0911, PLR0913, PLR0917
        self,
        gate_type: str,
        value: any,
        gate_info: dict | None,
        next_type: str | None,
        next_value: any,
        next_gate_info: dict | None,
    ) -> bool:
        """Check if two gates can be chained.

        Returns:
        -------
        bool
            True if the gates can be chained (reduced together).
        """
        if gate_type != next_type:
            return False
        # Q/P spiders: always chain (any polynomials)
        if gate_type in {"Q", "P"}:
            return True

        # R, BS, D: always chain
        if gate_type in {"R", "BS", "D", "Sq"}:
            return True

        # Fourier: only same type (F with F, Finv with Finv) or (F with Finv or Finv and F)
        if gate_type == "F":
            return value == next_value or {value, next_value} == {"F", "Finv"}

        # F2: always chain
        if gate_type == "F2":
            return True

        # ControlledZGate: always chain
        if gate_type == "CZ":
            return True

        # ControlledSumGate: chain only if control and target are the same
        if gate_type == "CSUM":
            return (gate_info["control"] == next_gate_info["control"]) and (
                gate_info["target"] == next_gate_info["target"]
            )

        return False

    def reduce_chain(self, gate_type: str, values: list, gate_info: dict | None = None) -> dict | None:  # noqa: C901, PLR0911, PLR0912
        """Reduce a chain of gates to a single gate.

        Parameters:
        ----------
        gate_type : str
            Type of gates in the chain
        values : list
            List of values for each gate in the chain
        gate_info : dict | None
            Useful information about the gate to reduce.

        Returns:
        -------
        dict | None
            Dictionary with reduced gate attributes, or None if identity.
            Contains: 'type', and type-specific fields.
        """
        zero_phase = ZxPoly({})
        id_q = {
            "type": "QSpider",
            "phase": zero_phase,
            "num_inputs": 1,
            "num_outputs": 1,
        }
        id_q2 = {
            "type": "QSpider",
            "phase": zero_phase,
            "num_inputs": 2,
            "num_outputs": 2,
        }

        if gate_type == "Q":
            total_phase = zero_phase
            for v in values:
                total_phase += v
            return {
                "type": "QSpider",
                "phase": total_phase,
                "num_inputs": gate_info["num_inputs"],
                "num_outputs": gate_info["num_outputs"],
            }

        # P-Spider: sum phases
        if gate_type == "P":
            total_phase = ZxPoly({})
            for v in values:
                total_phase += v
            return {
                "type": "PSpider",
                "phase": total_phase,
                "num_inputs": gate_info["num_inputs"],
                "num_outputs": gate_info["num_outputs"],
            }

        # R: sum angles
        if gate_type == "R":
            total = sum(values)
            if total == 0:
                return id_q
            return {"type": "PhaseRotationGate", "theta": total}

        # BS: sum angles
        if gate_type == "BS":
            total = sum(values)
            if total == 0:
                return id_q2
            return {"type": "BeamsplitterGate", "theta": total}

        # Sq: multiply
        if gate_type == "Sq":
            total = 1.0
            for v in values:
                total *= v
            if total == 1.0:  # noqa: RUF069
                return id_q
            return {"type": "SqueezingGate", "tau": total}

        # D: sum
        if gate_type == "D":
            total = sum(values)
            if total == 0:
                return id_q
            return {"type": "DisplacementGate", "alpha": total}

        # F: Fourier rules
        if gate_type == "F":
            n = len(values)
            remainder = n % 4
            if remainder == 0:
                return id_q
            if remainder == 1:
                return {"type": "Fourier" if values[0] == "F" else "FourierInv"}
            if remainder == 2:  # noqa: PLR2004
                return {"type": "Fourier2"}
            # Remainder == 3
            return {"type": "FourierInv" if values[0] == "F" else "Fourier"}

        # F2: parity
        if gate_type == "F2":
            if len(values) % 2 == 0:
                return id_q
            return {"type": "Fourier2"}

        # F_pair
        if gate_type == "F_pair":
            return id_q

        # ControlledZGate: sum gains
        if gate_type == "CZ":
            total = sum(values)
            if total == 0:
                return id_q2
            return {"type": "ControlledZGate", "gain": total}

        # ControlledSumGate: sum gains (only if control/target same)
        if gate_type == "CSUM":
            # Check that all ControlledSumGate gates have same control and target
            # We need to get this from the graph
            total = sum(values)
            if total == 0:
                return id_q2
            # Control/target info will be preserved from the first gate
            return {
                "type": "ControlledSumGate",
                "gain": total,
                "control": gate_info["control"],
                "target": gate_info["target"],
            }

        return None

    def _update_node_for_reduced_gate(
        self, graph: nx.DiGraph, node_id: int, reduced_gate: dict, container_attrs: dict
    ) -> None:
        """Update a node with reduced gate attributes."""
        gate_type = reduced_gate["type"]

        # Base attributes
        updates = {
            "type": gate_type,
            "container_id": container_attrs.get("container_id"),
        }

        # Type-specific attributes
        if gate_type in {"QSpider", "PSpider"}:
            updates["phase"] = reduced_gate["phase"]
            updates["num_inputs"] = reduced_gate["num_inputs"]
            updates["num_outputs"] = reduced_gate["num_outputs"]

        elif gate_type in {"PhaseRotationGate", "BeamsplitterGate"}:
            updates["phase"] = reduced_gate["theta"]

        elif gate_type == "SqueezingGate":
            updates["phase"] = reduced_gate["tau"]

        elif gate_type == "DisplacementGate":
            updates["phase"] = reduced_gate["alpha"]

        elif gate_type in {"Fourier", "FourierInv", "Fourier2"}:
            pass  # No additional attributes

        elif gate_type == "ControlledZGate":
            updates["phase"] = reduced_gate["gain"]
            updates["num_inputs"] = 2
            updates["num_outputs"] = 2

        elif gate_type == "ControlledSumGate":
            updates["phase"] = reduced_gate["gain"]
            updates["num_inputs"] = 2
            updates["num_outputs"] = 2
            updates["control"] = reduced_gate["control"]
            updates["target"] = reduced_gate["target"]

        graph.nodes[node_id].update(updates)

    def _update_connectivity_after_reduction(
        self, graph: nx.DiGraph, container_id: int, removed_connectivity: list[int]
    ) -> None:
        """Update connectivity after removing nodes from a composition."""
        attrs = graph.nodes[container_id]
        if "connectivity" not in attrs:
            return
        connectivity = attrs["connectivity"]
        new_connectivity = {}
        # Build mapping from old indices to new indices
        removed_set = set(removed_connectivity)
        index_map = {}
        new_idx = 0
        for old_idx in range(len(connectivity)):
            if old_idx not in removed_set:
                index_map[old_idx] = new_idx
                new_idx += 1

        # Remap connectivity
        for old_idx, targets in connectivity.items():
            if old_idx not in removed_set:
                new_idx = index_map[old_idx]
                new_connectivity[new_idx] = targets

        attrs["connectivity"] = new_connectivity


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

    num_inputs = attrs.get("num_inputs", 0)
    num_outputs = attrs.get("num_outputs", 0)
    if num_inputs != 1 or num_outputs != 1:
        return False

    phase = attrs.get("phase")
    if phase is None:
        return False

    return phase.is_zero


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
    register = GateRegister()
    register.build_from_graph(graph)
    rule.apply_rule(graph, register)
    return to_diagram(graph)

"""CV ZX calculus rewrite rules and applications.

This module implements the 10 basic rewrite rules from [1] Sec. IV.A,
and the derived rules from Sec. IV.B; using a graph structure.

References:
-----------
[1] Nagayoshi et al., CV ZX calculus, 2024
"""

import math
from abc import ABC, abstractmethod
from typing import Any

import networkx as nx
from sympy import Expr, cos, pi, tan

from cvzx.base_gates import CompositionDiagram, ContractedDiagram, Diagram, TensorDiagram, ZxPoly
from cvzx.gates import BeamsplitterGate, ControlledSumGate
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
        2. Returns the modified graph

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
    def match(self, graph: nx.DiGraph, registry: GateRegister) -> list:
        """Find all matches of the rule pattern in the graph.

        Parameters:
        ----------
        graph : nx.DiGraph
            The graph to search.
        registry : GateRegister
                    Registry for tracking specific gate types and nodes in a CV ZX graph.

        Returns:
        -------
        list
            List of matches. Each match must be a dictionary or object
            that can be passed to `apply_single`.
        """

    @abstractmethod
    def apply_single(self, graph: nx.DiGraph, match: list) -> None:
        """Apply the rule to a specific match in-place.

        Parameters:
        ----------
        graph : nx.DiGraph
            The graph to modify.
        match : list
            A match returned by `match()`.
        """

    def apply_rule(self, graph: nx.DiGraph, registry: GateRegister) -> nx.DiGraph:
        """Apply a rule to the entire graph.

        Parameters:
        ----------
        graph : nx.DiGraph
                    The graph to modify.
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
        matches = self.match(graph, registry)

        if not matches:
            return graph

        # Group by container and apply each group's matches in reverse
        # position order, so an earlier removal can't shift a later match's index.
        groups: dict[object, list] = {}
        group_order: list[object] = []
        for match in matches:
            key = match.get("container_id") if isinstance(match, dict) else None
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(match)

        for key in group_order:
            for match in reversed(groups[key]):
                self.apply_single(graph, match)

        return graph

    def _flatten_container(self, graph: nx.DiGraph, container_id: int) -> None:
        """Flatten a container if it has only one element.

        Parameters:
        ----------
        graph : nx.DiGraph
            The graph to modify.
        container_id : int
            The container node ID.
        """
        if container_id not in graph.nodes:
            return

        attrs = graph.nodes[container_id]
        sub_ids = attrs.get("sub_diagram_ids", [])

        # If the container has exactly one element, flatten it
        if len(sub_ids) == 1:
            child_id = sub_ids[0]

            # Get the container's container
            parent_container_id = attrs.get("container_id")

            # Replace the container with its child in the parent
            if parent_container_id is not None and parent_container_id in graph.nodes:
                parent_attrs = graph.nodes[parent_container_id]
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
                if child_id in graph.nodes:
                    graph.nodes[child_id]["container_id"] = parent_container_id
                # Remove the container node
                graph.remove_node(container_id)

            elif child_id in graph.nodes:
                # The Container was the root
                graph.nodes[child_id]["container_id"] = None
                graph.nodes[child_id]["is_root"] = True

    @staticmethod
    def _remap_by_identity(old_mapping: dict, new_mapping: dict) -> dict:
        """Match old external ports to new ones by identical target.

        Used when a container's ports were recomputed WITHOUT any sibling
        being removed (a contracted side's arity changing in place, or a
        tensor child being substituted at a fixed position) -- in that
        case the (side-or-index, internal_port) target a port maps to is
        stable, so old ports are matched to new ports by that identity.

        Parameters
        ----------
        old_mapping : dict
            The container's `external_input_mapping` (or output) before
            the recompute, port -> (side_or_idx, internal_port).
        new_mapping : dict
            The same, after the recompute.

        Returns:
        -------
        dict
            old_port -> new_port for every old port whose target still
            exists in the new mapping.
        """
        reverse_new = {tuple(target): new_port for new_port, target in new_mapping.items()}
        remap = {}
        for old_port, target in old_mapping.items():
            key = tuple(target)
            if key in reverse_new:
                remap[old_port] = reverse_new[key]
        return remap

    def _recompute_contracted_arity(self, graph: nx.DiGraph, container_id: int) -> tuple:  # noqa: PLR0914
        """Recompute a ContractedDiagram container's ports after a side's arity changed in place.

        `first_id`/`second_id` are assumed already updated to point at the
        (possibly different-arity) current children; `I1`/`I2`/`J1`/`J2`
        are unchanged. Mirrors `_add_contracted_node`'s own bookkeeping.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        container_id : int
            The ContractedDiagram container node ID.

        Returns:
        -------
        tuple
            (old_num_inputs, new_num_inputs, old_num_outputs,
            new_num_outputs, input_remap, output_remap) for
            `_propagate_arity_to_parent`.
        """
        attrs = graph.nodes[container_id]
        first_id = attrs["first_id"]
        second_id = attrs["second_id"]
        j1 = attrs.get("J1", [])
        i2 = attrs.get("I2", [])
        i1 = attrs.get("I1", [])
        j2 = attrs.get("J2", [])

        first_num_inputs = graph.nodes[first_id].get("num_inputs", 0)
        first_num_outputs = graph.nodes[first_id].get("num_outputs", 0)
        second_num_inputs = graph.nodes[second_id].get("num_inputs", 0)
        second_num_outputs = graph.nodes[second_id].get("num_outputs", 0)

        old_num_inputs = attrs.get("num_inputs", 0)
        old_num_outputs = attrs.get("num_outputs", 0)
        old_input_mapping = attrs.get("external_input_mapping", {})
        old_output_mapping = attrs.get("external_output_mapping", {})

        kept_first_inputs = [p for p in range(first_num_inputs) if p not in j1]
        kept_second_inputs = [p for p in range(second_num_inputs) if p not in i2]
        kept_first_outputs = [p for p in range(first_num_outputs) if p not in i1]
        kept_second_outputs = [p for p in range(second_num_outputs) if p not in j2]

        input_mapping = {}
        offset = 0
        for p in kept_first_inputs:
            input_mapping[offset] = ("first", p)
            offset += 1
        for p in kept_second_inputs:
            input_mapping[offset] = ("second", p)
            offset += 1
        new_num_inputs = offset

        output_mapping = {}
        offset = 0
        for p in kept_first_outputs:
            output_mapping[offset] = ("first", p)
            offset += 1
        for p in kept_second_outputs:
            output_mapping[offset] = ("second", p)
            offset += 1
        new_num_outputs = offset

        attrs.update({
            "kept_first_inputs": kept_first_inputs,
            "kept_second_inputs": kept_second_inputs,
            "kept_first_outputs": kept_first_outputs,
            "kept_second_outputs": kept_second_outputs,
            "external_input_mapping": input_mapping,
            "external_output_mapping": output_mapping,
            "num_inputs": new_num_inputs,
            "num_outputs": new_num_outputs,
            "external_inputs": list(range(new_num_inputs)),
            "external_outputs": list(range(new_num_outputs)),
        })

        input_remap = self._remap_by_identity(old_input_mapping, input_mapping)
        output_remap = self._remap_by_identity(old_output_mapping, output_mapping)
        return old_num_inputs, new_num_inputs, old_num_outputs, new_num_outputs, input_remap, output_remap

    def _recompute_tensor_arity(self, graph: nx.DiGraph, container_id: int) -> tuple:
        """Recompute a TensorDiagram container's ports after a child was substituted in place.

        The child at each index is assumed unchanged in position (only its
        own arity may have changed) -- no sibling was added or removed, so
        old ports are safely matched to new ports by (index, internal_port)
        identity even though numeric port offsets may shift.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        container_id : int
            The TensorDiagram container node ID.

        Returns:
        -------
        tuple
            (old_num_inputs, new_num_inputs, old_num_outputs,
            new_num_outputs, input_remap, output_remap) for
            `_propagate_arity_to_parent`.
        """
        attrs = graph.nodes[container_id]
        sub_ids = attrs.get("sub_diagram_ids", [])

        old_num_inputs = attrs.get("num_inputs", 0)
        old_num_outputs = attrs.get("num_outputs", 0)
        old_input_mapping = attrs.get("external_input_mapping", {})
        old_output_mapping = attrs.get("external_output_mapping", {})

        input_mapping = {}
        offset = 0
        for idx, sub_id in enumerate(sub_ids):
            for p in range(graph.nodes[sub_id].get("num_inputs", 0)):
                input_mapping[offset] = (idx, p)
                offset += 1
        new_num_inputs = offset

        output_mapping = {}
        offset = 0
        for idx, sub_id in enumerate(sub_ids):
            for p in range(graph.nodes[sub_id].get("num_outputs", 0)):
                output_mapping[offset] = (idx, p)
                offset += 1
        new_num_outputs = offset

        attrs.update({
            "external_input_mapping": input_mapping,
            "external_output_mapping": output_mapping,
            "num_inputs": new_num_inputs,
            "num_outputs": new_num_outputs,
            "external_inputs": list(range(new_num_inputs)),
            "external_outputs": list(range(new_num_outputs)),
        })

        input_remap = self._remap_by_identity(old_input_mapping, input_mapping)
        output_remap = self._remap_by_identity(old_output_mapping, output_mapping)
        return old_num_inputs, new_num_inputs, old_num_outputs, new_num_outputs, input_remap, output_remap

    @staticmethod
    def _recompute_tensor_arity_from_child_remap(  # noqa: C901, PLR0913, PLR0917
        graph: nx.DiGraph,
        container_id: int,
        changed_child_id: int,
        child_old_num_inputs: int,
        child_input_remap: dict,
        child_old_num_outputs: int,
        child_output_remap: dict,
    ) -> tuple:
        """Recompute a TensorDiagram's ports using a child's own KNOWN remap.

        `_recompute_tensor_arity` re-derives port correspondence by
        comparing old and new (index, port) pairs and trusting a match
        whenever one happens to line up -- correct when a child's own
        arity change is a plain grow/shrink at the tail, but WRONG when
        the child is itself a container whose ports were just reordered by
        dropping one from the *middle* (exactly what happens when this
        same upward propagation recomputes an intermediate ancestor):
        the child's remaining ports are always freshly numbered from 0,
        so "port 0 before" and "port 0 after" get matched even when they
        are, semantically, two different wires.

        This instead composes directly from `changed_child_id`'s own
        already-known `input_remap`/`output_remap` -- correct regardless
        of how its ports moved around internally, since every other
        sibling's ports are provably untouched (only `changed_child_id`
        itself changed) and only need a uniform offset shift.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        container_id : int
            The TensorDiagram container node ID.
        changed_child_id : int
            The one child whose own arity just changed.
        child_old_num_inputs, child_old_num_outputs : int
            That child's arity before its own change.
        child_input_remap, child_output_remap : dict
            That child's own old port -> new port maps, for ports that
            still exist.

        Returns:
        -------
        tuple
            (old_num_inputs, new_num_inputs, old_num_outputs,
            new_num_outputs, input_remap, output_remap) for
            `_propagate_arity_to_parent`.
        """
        attrs = graph.nodes[container_id]
        sub_ids = attrs.get("sub_diagram_ids", [])
        pos = sub_ids.index(changed_child_id)

        old_num_inputs = attrs.get("num_inputs", 0)
        old_num_outputs = attrs.get("num_outputs", 0)

        def build(old_child_count: int, child_remap: dict, count_attr: str) -> tuple:
            old_offset = 0
            new_offset = 0
            remap = {}
            for idx, sid in enumerate(sub_ids):
                if idx == pos:
                    for p in range(old_child_count):
                        if p in child_remap:
                            remap[old_offset] = new_offset + child_remap[p]
                        old_offset += 1
                    new_offset += graph.nodes[sid].get(count_attr, 0)
                else:
                    count = graph.nodes[sid].get(count_attr, 0)
                    for _ in range(count):
                        remap[old_offset] = new_offset
                        old_offset += 1
                        new_offset += 1
            return remap, new_offset

        input_remap, new_num_inputs = build(child_old_num_inputs, child_input_remap, "num_inputs")
        output_remap, new_num_outputs = build(child_old_num_outputs, child_output_remap, "num_outputs")

        input_mapping = {}
        offset = 0
        for idx, sid in enumerate(sub_ids):
            for p in range(graph.nodes[sid].get("num_inputs", 0)):
                input_mapping[offset] = (idx, p)
                offset += 1
        output_mapping = {}
        offset = 0
        for idx, sid in enumerate(sub_ids):
            for p in range(graph.nodes[sid].get("num_outputs", 0)):
                output_mapping[offset] = (idx, p)
                offset += 1

        attrs.update({
            "external_input_mapping": input_mapping,
            "external_output_mapping": output_mapping,
            "num_inputs": new_num_inputs,
            "num_outputs": new_num_outputs,
            "external_inputs": list(range(new_num_inputs)),
            "external_outputs": list(range(new_num_outputs)),
        })

        return old_num_inputs, new_num_inputs, old_num_outputs, new_num_outputs, input_remap, output_remap

    @staticmethod
    def _remove_tensor_child(  # noqa: C901
        graph: nx.DiGraph,
        container_id: int,
        removed_id: int,
        removed_num_inputs: int,
        removed_num_outputs: int,
    ) -> tuple:
        """Remove one child from a TensorDiagram container in place.

        Unlike `_recompute_tensor_arity`, a child is actually leaving the
        list here, so every later sibling's index shifts down by one --
        identity-based matching would silently misplace them, so old ports
        are mapped to new ports by direct position arithmetic instead.

        `removed_num_inputs`/`removed_num_outputs` must be the arity the
        removed child had WHILE it was still a member of this container --
        the caller may have already overwritten `removed_id`'s own node
        attrs in place to describe an unrelated new role (e.g. transformed
        into a TensorDiagram of copies) before calling this, so those
        current attrs cannot be trusted for what it used to contribute
        here.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        container_id : int
            The TensorDiagram container node ID.
        removed_id : int
            The child node ID being removed.
        removed_num_inputs : int
            The number of inputs the removed child contributed here.
        removed_num_outputs : int
            The number of outputs the removed child contributed here.

        Returns:
        -------
        tuple
            (old_num_inputs, new_num_inputs, old_num_outputs,
            new_num_outputs, input_remap, output_remap) for
            `_propagate_arity_to_parent`.
        """
        attrs = graph.nodes[container_id]
        sub_ids = attrs["sub_diagram_ids"]
        pos = sub_ids.index(removed_id)

        old_num_inputs = attrs.get("num_inputs", 0)
        old_num_outputs = attrs.get("num_outputs", 0)

        input_offset_before = sum(graph.nodes[sid].get("num_inputs", 0) for sid in sub_ids[:pos])
        input_remap = {}
        for old_port in range(old_num_inputs):
            if old_port < input_offset_before:
                input_remap[old_port] = old_port
            elif old_port >= input_offset_before + removed_num_inputs:
                input_remap[old_port] = old_port - removed_num_inputs

        output_offset_before = sum(graph.nodes[sid].get("num_outputs", 0) for sid in sub_ids[:pos])
        output_remap = {}
        for old_port in range(old_num_outputs):
            if old_port < output_offset_before:
                output_remap[old_port] = old_port
            elif old_port >= output_offset_before + removed_num_outputs:
                output_remap[old_port] = old_port - removed_num_outputs

        sub_ids.remove(removed_id)
        new_num_inputs = old_num_inputs - removed_num_inputs
        new_num_outputs = old_num_outputs - removed_num_outputs

        input_mapping = {}
        offset = 0
        for idx, sid in enumerate(sub_ids):
            for p in range(graph.nodes[sid].get("num_inputs", 0)):
                input_mapping[offset] = (idx, p)
                offset += 1
        output_mapping = {}
        offset = 0
        for idx, sid in enumerate(sub_ids):
            for p in range(graph.nodes[sid].get("num_outputs", 0)):
                output_mapping[offset] = (idx, p)
                offset += 1

        attrs.update({
            "num_inputs": new_num_inputs,
            "num_outputs": new_num_outputs,
            "external_inputs": list(range(new_num_inputs)),
            "external_outputs": list(range(new_num_outputs)),
            "external_input_mapping": input_mapping,
            "external_output_mapping": output_mapping,
        })

        return old_num_inputs, new_num_inputs, old_num_outputs, new_num_outputs, input_remap, output_remap

    def _propagate_arity_to_parent(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917
        self,
        graph: nx.DiGraph,
        container_id: int,
        old_num_inputs: int,
        new_num_inputs: int,
        old_num_outputs: int,
        new_num_outputs: int,
        input_remap: dict,
        output_remap: dict,
    ) -> None:
        """Propagate a child's arity change up through every ancestor that needs it.

        A single cross-container copy only directly touches two containers
        (the copy spider's old parent and the disappearing spider's old
        parent), each one level removed from the match itself. But either
        one can sit arbitrarily deep inside further TensorDiagram/
        ContractedDiagram/CompositionDiagram nesting -- e.g. a two-mode
        segment that itself lives inside a TensorDiagram layer of a bigger
        circuit -- and an arity change has to keep climbing until it either
        dies out (an interior CompositionDiagram slot whose neighbors don't
        care) or reaches the root.

        At each step: a TensorDiagram or ContractedDiagram parent's own
        external arity is entirely determined by its children's, so it's
        recomputed via `_recompute_tensor_arity`/`_recompute_contracted_arity`
        (which read the already-updated child in place) and the change
        keeps climbing regardless of whether that recompute actually
        changed anything -- if it didn't, the next iteration's `while`
        check simply stops. A flat CompositionDiagram parent instead fixes
        the internal `connectivity` link(s) next to `container_id`'s slot;
        its own external arity only changes when that slot is the first
        (own inputs) or last (own outputs) child, in which case that
        boundary change is folded in and propagation continues upward too.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        container_id : int
            The node whose external arity just changed.
        old_num_inputs, new_num_inputs, old_num_outputs, new_num_outputs : int
            Its external arity before and after the change.
        input_remap : dict
            old input port -> new input port, for ports that still exist.
        output_remap : dict
            old output port -> new output port, for ports that still exist.
        """
        while old_num_inputs != new_num_inputs or old_num_outputs != new_num_outputs:
            parent_id = graph.nodes[container_id].get("container_id")
            if parent_id is None or parent_id not in graph.nodes:
                return
            parent_attrs = graph.nodes[parent_id]
            parent_type = parent_attrs.get("container_type")

            if parent_type == "tensor":
                arity_change = self._recompute_tensor_arity_from_child_remap(
                    graph,
                    parent_id,
                    container_id,
                    old_num_inputs,
                    input_remap,
                    old_num_outputs,
                    output_remap,
                )
            elif parent_type == "contracted":
                # `container_id`'s own ports may just have been renumbered
                # (not merely recounted) by whatever produced this arity
                # change -- e.g. a sibling was spliced out of a TensorDiagram
                # it sits in, shifting every later port down. The trace
                # indices I1/J1 (when `container_id` is "first") or I2/J2
                # (when it's "second") were recorded against its OLD
                # numbering, so they have to be translated through the same
                # `input_remap`/`output_remap` before recomputing -- unlike
                # a fresh CopyRule "install" substitution, where the
                # replacement has the identical port count/order as what it
                # replaced and the stored indices are still valid as-is.
                if parent_attrs.get("first_id") == container_id:
                    if old_num_inputs != new_num_inputs:
                        parent_attrs["J1"] = [input_remap[p] for p in parent_attrs.get("J1", []) if p in input_remap]
                    if old_num_outputs != new_num_outputs:
                        parent_attrs["I1"] = [output_remap[p] for p in parent_attrs.get("I1", []) if p in output_remap]
                else:
                    if old_num_inputs != new_num_inputs:
                        parent_attrs["I2"] = [input_remap[p] for p in parent_attrs.get("I2", []) if p in input_remap]
                    if old_num_outputs != new_num_outputs:
                        parent_attrs["J2"] = [output_remap[p] for p in parent_attrs.get("J2", []) if p in output_remap]
                arity_change = self._recompute_contracted_arity(graph, parent_id)
            elif parent_type == "composition":
                sub_ids = parent_attrs.get("sub_diagram_ids", [])
                if container_id not in sub_ids:
                    return
                pos = sub_ids.index(container_id)
                connectivity = parent_attrs.get("connectivity", {})

                if old_num_inputs != new_num_inputs and pos > 0:
                    key = pos - 1
                    old_conn = connectivity.get(key, {})
                    new_conn = {}
                    for in_port, out_port in old_conn.items():
                        if in_port in input_remap:
                            new_conn[input_remap[in_port]] = out_port
                    connectivity[key] = new_conn

                if old_num_outputs != new_num_outputs and pos < len(sub_ids) - 1:
                    key = pos
                    old_conn = connectivity.get(key, {})
                    new_conn = {}
                    for in_port, out_port in old_conn.items():
                        if out_port in output_remap:
                            new_conn[in_port] = output_remap[out_port]
                    connectivity[key] = new_conn

                parent_attrs["connectivity"] = connectivity

                # A flat composition's own external arity is exactly its
                # first child's num_inputs and its last child's
                # num_outputs -- so it changes, and needs propagating
                # further up in turn, precisely when the affected slot is
                # that first/last position.
                parent_old_in = parent_attrs.get("num_inputs", 0)
                parent_old_out = parent_attrs.get("num_outputs", 0)
                is_first = pos == 0
                is_last = pos == len(sub_ids) - 1
                parent_new_in = new_num_inputs if is_first else parent_old_in
                parent_new_out = new_num_outputs if is_last else parent_old_out
                parent_input_remap = input_remap if is_first else {p: p for p in range(parent_old_in)}
                parent_output_remap = output_remap if is_last else {p: p for p in range(parent_old_out)}

                parent_attrs["num_inputs"] = parent_new_in
                parent_attrs["num_outputs"] = parent_new_out
                parent_attrs["external_inputs"] = list(range(parent_new_in))
                parent_attrs["external_outputs"] = list(range(parent_new_out))

                arity_change = (
                    parent_old_in,
                    parent_new_in,
                    parent_old_out,
                    parent_new_out,
                    parent_input_remap,
                    parent_output_remap,
                )
            else:
                return

            (
                old_num_inputs,
                new_num_inputs,
                old_num_outputs,
                new_num_outputs,
                input_remap,
                output_remap,
            ) = arity_change
            container_id = parent_id


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

        # Sort for deterministic match order (set iteration order is not stable).
        for node in sorted(registry.identity_spiders):
            attrs = graph.nodes[node]
            container_id = attrs.get("container_id")
            # A root identity spider (container_id is None -- the whole
            # diagram simplified down to a single wire) has no composition
            # to be removed from; there's nothing left for this rule to do.
            if (
                attrs.get("kind") == "proper"
                and is_wiring_node_from_attrs(attrs)
                and container_id is not None
                and graph.nodes[container_id]["container_type"] == "composition"
            ):
                matches.append({
                    "node_id": node,
                    "container_id": container_id,
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
        container_id = match["container_id"]

        # Matches are computed once up front and then grouped/applied by
        # `RewriteRule.apply_rule` (reverse position order *within* one
        # container's own group of matches, so an earlier removal there
        # can't shift a later index) -- but nothing stops a match in one
        # group from being invalidated by a *different* group's removal
        # applied first (e.g. `_flatten_container`, triggered while
        # collapsing a neighboring container, can delete this match's
        # own `container_id` node entirely). Re-check both node ids are
        # still actually in the graph before touching their attrs: a
        # missing one means an earlier match in this same batch already
        # resolved this node one way or another, so there's nothing left
        # for this stale match to do.
        if node_id not in graph or container_id not in graph:
            return
        attrs = graph.nodes[node_id]

        # Check that the node is an identity spider which can be reduced
        if not (
            attrs.get("kind") == "proper"
            and is_wiring_node_from_attrs(attrs)
            and container_id is not None
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

        # Sort for deterministic match order (set iteration order is not stable).
        for node in sorted(registry.contracted_diagrams):
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

        # Sort for deterministic match order (set iteration order is not stable).
        for container_id in sorted(registry.composition_nodes):
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

    def get_gate_info(self, graph: nx.DiGraph, node_id: int) -> tuple[str | None, Any, dict | None]:  # noqa: C901, PLR0911
        """Extract gate type and value from a node.

        Returns:
        -------
        tuple[str | None, Any, dict | None]
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
        value: Any,  # noqa: ANN401
        gate_info: dict | None,
        next_type: str | None,
        next_value: Any,  # noqa: ANN401
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
            if total == 1.0:
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


class FourierNormalizationRule(RewriteRule):
    r"""Fourier normalization rule - Graph-based version.

    Folds a Fourier-type gate (F, Finv, or F2) into an adjacent rotation or
    squeezing gate -- a cross-type merge `ChainReductionRule` cannot do on
    its own, since its chaining is strictly same-type. Treats the
    Fourier-type gate as an equivalent rotation/squeezing value and combines
    it the same way ChainReductionRule combines same-type chains (sum for
    rotations, product for squeezing). Prepares the ground for terminal
    absorption.

    Cases (either list order -- F, Finv and F2 all commute with rotations,
    and F2 also commutes with squeezing, since they act as fixed
    rotations/parity flips):
    - F ∘ R(θ) → R(θ - π/2)
    - Finv ∘ R(θ) → R(θ + π/2)
    - F2 ∘ R(θ) → R(θ + π)
    - F2 ∘ Sq(τ) → Sq(-τ)

    F and Finv never merge with a squeezing gate.
    """

    # Equivalent rotation angle contributed by each Fourier-type gate.
    _ROTATION_DELTA = {  # noqa: RUF012
        "Fourier": -pi / 2,
        "FourierInv": pi / 2,
        "Fourier2": pi,
    }

    def match(self, graph: nx.DiGraph, registry: GateRegister) -> list[dict]:
        """Find all Fourier-type/rotation or F2/squeezing pairs.

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
            - 'node_ids': [keep_id, absorb_id] in list order (keep_id survives)
            - 'result_type': 'PhaseRotationGate' or 'SqueezingGate'
            - 'result_value': the combined angle or squeezing parameter
        """
        matches = []

        # Sort for deterministic match order (set iteration order is not stable).
        for container_id in sorted(registry.composition_nodes):
            attrs = graph.nodes[container_id]
            sub_ids = attrs.get("sub_diagram_ids", [])

            if len(sub_ids) < 2:  # noqa: PLR2004
                continue

            # Skip past a matched pair (i += 2) rather than sliding one at a
            # time (i += 1): a Fourier-type node's "result_value" is computed
            # once, from the neighbor's *current* phase. If two overlapping
            # pairs were both matched here (e.g. F, R, Finv matching both
            # (F,R) and (R,Finv)), applying the second one would overwrite
            # the shared R node with a value computed before the first
            # match's fold -- silently wrong, not even a crash. Consuming
            # each matched pair before continuing avoids that; a 3+ chain
            # like F ∘ R ∘ Finv still fully resolves, just over two
            # apply_rule() passes instead of one (the fixed-point loop this
            # rule is meant to run in already repeats until nothing changes).
            i = 0
            while i < len(sub_ids) - 1:
                first_id = sub_ids[i]
                second_id = sub_ids[i + 1]

                match = self._check_pair(graph, first_id, second_id)
                if match:
                    match["container_id"] = container_id
                    match["indices"] = [i, i + 1]
                    matches.append(match)
                    i += 2
                else:
                    i += 1

        return matches

    def _check_pair(self, graph: nx.DiGraph, first_id: int, second_id: int) -> dict | None:
        """Check if a pair of adjacent nodes can be folded together.

        Returns:
        -------
        dict | None
            Match dictionary if the pair can be folded, None otherwise.
        """
        first_type = graph.nodes[first_id].get("type")
        second_type = graph.nodes[second_id].get("type")

        # Fourier-type gate next to a rotation, in either order.
        if first_type in self._ROTATION_DELTA and second_type == "PhaseRotationGate":
            return self._rotation_match(
                graph, fourier_id=first_id, rotation_id=second_id, first_id=first_id, second_id=second_id
            )
        if second_type in self._ROTATION_DELTA and first_type == "PhaseRotationGate":
            return self._rotation_match(
                graph, fourier_id=second_id, rotation_id=first_id, first_id=first_id, second_id=second_id
            )

        # F2 next to a squeezing gate, in either order. F and Finv never merge with squeezing.
        if first_type == "Fourier2" and second_type == "SqueezingGate":
            return self._squeezing_match(graph, squeezing_id=second_id, first_id=first_id, second_id=second_id)
        if second_type == "Fourier2" and first_type == "SqueezingGate":
            return self._squeezing_match(graph, squeezing_id=first_id, first_id=first_id, second_id=second_id)

        return None

    def _rotation_match(
        self, graph: nx.DiGraph, *, fourier_id: int, rotation_id: int, first_id: int, second_id: int
    ) -> dict:
        """Build the match dict for a Fourier-type/rotation pair."""  # noqa: DOC201
        fourier_type = graph.nodes[fourier_id]["type"]
        theta = graph.nodes[rotation_id]["phase"]
        return {
            "node_ids": [first_id, second_id],
            "result_type": "PhaseRotationGate",
            "result_value": theta + self._ROTATION_DELTA[fourier_type],
        }

    def _squeezing_match(self, graph: nx.DiGraph, *, squeezing_id: int, first_id: int, second_id: int) -> dict:
        """Build the match dict for an F2/squeezing pair."""  # noqa: DOC201
        tau = graph.nodes[squeezing_id]["phase"]
        return {
            "node_ids": [first_id, second_id],
            "result_type": "SqueezingGate",
            "result_value": -tau,
        }

    def apply_single(self, graph: nx.DiGraph, match: dict) -> None:
        """Apply a Fourier-normalization fold to a specific match in-place.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        match : dict
            Match containing fold information.
        """
        container_id = match["container_id"]
        keep_id, absorb_id = match["node_ids"]
        i, _ = match["indices"]

        # Contract the two nodes via nx; keep_id survives with its attrs
        # fully overwritten below, so which original node's id survives
        # doesn't matter.
        nx.contracted_nodes(graph, keep_id, absorb_id, self_loops=False, copy=False)
        if "contraction" in graph.nodes[keep_id]:
            del graph.nodes[keep_id]["contraction"]

        graph.nodes[keep_id].update({
            "type": match["result_type"],
            "phase": match["result_value"],
            "container_id": container_id,
        })

        container_attrs = graph.nodes[container_id]
        sub_ids = container_attrs.get("sub_diagram_ids", [])
        new_sub_ids = [sub_id for sub_id in sub_ids if sub_id != absorb_id]
        container_attrs["sub_diagram_ids"] = new_sub_ids

        # Drop the now-internal link at i, move i+1's outgoing link to i,
        # and shift every index past i+1 down by one.
        if "connectivity" in container_attrs:
            connectivity = container_attrs["connectivity"]
            new_connectivity = {}
            for old_idx, targets in connectivity.items():
                if old_idx == i:
                    continue
                if old_idx == i + 1:
                    new_connectivity[i] = targets
                elif old_idx < i:
                    new_connectivity[old_idx] = targets
                else:  # old_idx > i + 1
                    new_connectivity[old_idx - 1] = targets
            container_attrs["connectivity"] = new_connectivity

        if len(new_sub_ids) == 1:
            self._flatten_container(graph, container_id)


class TerminalAbsorptionRule(RewriteRule):
    r"""Terminal absorption rule - Graph-based version.

    Absorbs a gate adjacent to a QSpider/PSpider terminal (arity (1,0)
    effect or (0,1) state) into the terminal's own phase, eliminating the
    gate. Three sub-cases (the gate may sit on either side of the
    terminal, whichever its own arity allows -- an effect's gate is
    upstream, a state's gate is downstream):

    - Rotation (QSpider terminal only, input phase degree <= 1): a
      terminal with phase `c + k*x` folds R(theta) into
      `c - tan(theta)/2*x**2 + k/cos(theta)*x`. Doesn't match when theta
      is an odd multiple of pi/2 (tan/1-over-cos undefined). This is the
      QSpider/PSpider-absorbs-a-rotation identity worked out in [1] Eq.
      (239a)-(239e). `Fourier2` (a fixed rotation by pi) is absorbed the
      same way, using theta = pi in the formula above -- pi isn't an odd
      multiple of pi/2, so it's never degenerate. `Fourier`/`FourierInv`
      (fixed rotations by -+pi/2) are deliberately NOT recognized here:
      that is exactly the degenerate angle the formula excludes, so they
      can never absorb this way regardless of how they're spelled.
    - Squeezing (QSpider or PSpider terminal, any phase degree): a
      terminal with phase f(x) folds Sq(tau) into f(x/tau).
    - Cross-color discard (opposite-color raw (1,1) spider): a QSpider
      terminal absorbing an adjacent PSpider(1,1,f(x)), or vice versa,
      leaves the terminal's phase unchanged -- the gate simply vanishes.
      Same-color (1,1) spiders are ordinary spider fusion, FusionRule's
      job, not this rule's.

    Only the rotation sub-case is an exact identity for any physical
    state. Squeezing absorption and cross-color discard both treat the
    terminal's arbitrary-polynomial phase as an idealized (infinite
    squeezing) eigenstate -- a real finite-squeezed state would carry
    extra terms these two sub-cases drop. `assume_infinite_squeezing`
    (default False) gates whether those two sub-cases are allowed to
    match at all; when False, only rotation absorption runs.

    References:
    ----------
    [1] Nagayoshi et al., CV ZX calculus, 2024, Eq. (239a)-(239e).
    """

    def __init__(self, assume_infinite_squeezing: bool = False) -> None:  # noqa: FBT001 FBT002
        """Create the rule.

        Parameters
        ----------
        assume_infinite_squeezing : bool
            If True, also allow squeezing absorption and cross-color
            discard (both idealized). If False (default), only the
            exact rotation sub-case matches.
        """
        self.assume_infinite_squeezing = assume_infinite_squeezing

    def match(self, graph: nx.DiGraph, registry: GateRegister) -> list[dict]:
        """Find all gate/terminal pairs that can be absorbed.

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
            - 'node_ids': [keep_id, absorb_id] in list order (keep_id survives)
            - 'result_type': 'QSpider' or 'PSpider'
            - 'result_num_inputs' / 'result_num_outputs': the terminal's own arity
            - 'result_phase': the folded phase
        """
        matches = []

        # Sort for deterministic match order (set iteration order is not stable).
        for container_id in sorted(registry.composition_nodes):
            attrs = graph.nodes[container_id]
            sub_ids = attrs.get("sub_diagram_ids", [])

            if len(sub_ids) < 2:  # noqa: PLR2004
                continue

            # Skip past a matched pair (i += 2), same reasoning as
            # FourierNormalizationRule: avoids reusing a stale
            # precomputed result on a node a prior match already consumed.
            i = 0
            while i < len(sub_ids) - 1:
                first_id = sub_ids[i]
                second_id = sub_ids[i + 1]

                match = self._check_pair(graph, first_id, second_id)
                if match:
                    match["container_id"] = container_id
                    match["indices"] = [i, i + 1]
                    matches.append(match)
                    i += 2
                else:
                    i += 1

        return matches

    def _check_pair(self, graph: nx.DiGraph, first_id: int, second_id: int) -> dict | None:
        """Check if a pair of adjacent nodes forms an absorbable gate/terminal pattern.

        Returns:
        -------
        dict | None
            Match dictionary if the pair can be folded, None otherwise.
        """
        first_attrs = graph.nodes[first_id]
        second_attrs = graph.nodes[second_id]

        # State case: terminal (0,1) is upstream, gate is downstream.
        if (
            first_attrs.get("num_inputs") == 0
            and first_attrs.get("num_outputs") == 1
            and first_attrs.get("type") in {"QSpider", "PSpider"}
        ):
            result = self._try_absorb(terminal_attrs=first_attrs, gate_attrs=second_attrs)
            if result:
                result["node_ids"] = [first_id, second_id]
                return result

        # Effect case: terminal (1,0) is downstream, gate is upstream.
        if (
            second_attrs.get("num_inputs") == 1
            and second_attrs.get("num_outputs") == 0
            and second_attrs.get("type") in {"QSpider", "PSpider"}
        ):
            result = self._try_absorb(terminal_attrs=second_attrs, gate_attrs=first_attrs)
            if result:
                result["node_ids"] = [first_id, second_id]
                return result

        return None

    def _try_absorb(self, *, terminal_attrs: dict, gate_attrs: dict) -> dict | None:
        """Fold `gate_attrs` into `terminal_attrs`, if the pattern matches.

        Returns:
        -------
        dict | None
            Partial match dict (result type/arity/phase), or None.
        """
        terminal_type = terminal_attrs["type"]
        phase = terminal_attrs.get("phase")
        gate_type = gate_attrs.get("type")

        new_phase = None
        if gate_type == "PhaseRotationGate" and terminal_type == "QSpider":
            new_phase = self._rotation_absorb(phase, gate_attrs.get("phase"))
        elif gate_type == "Fourier2" and terminal_type == "QSpider":
            # F2 is a fixed rotation by pi -- exact, and pi isn't a
            # degenerate angle (only F/Finv's +-pi/2 would be, so those
            # can never absorb this way: nothing else recognizes them).
            new_phase = self._rotation_absorb(phase, pi)
        elif gate_type == "SqueezingGate" and self.assume_infinite_squeezing:
            new_phase = self._squeeze_absorb(phase, gate_attrs.get("phase"))
        elif (
            gate_type in {"QSpider", "PSpider"}
            and gate_type != terminal_type
            and gate_attrs.get("num_inputs") == 1
            and gate_attrs.get("num_outputs") == 1
            and self.assume_infinite_squeezing
        ):
            new_phase = phase  # Cross-color discard: unchanged.

        if new_phase is None:
            return None

        return {
            "result_type": terminal_type,
            "result_num_inputs": terminal_attrs["num_inputs"],
            "result_num_outputs": terminal_attrs["num_outputs"],
            "result_phase": new_phase,
        }

    @staticmethod
    def _rotation_absorb(phase: ZxPoly, theta: float | Expr) -> ZxPoly | None:
        """Fold a rotation R(theta) into a terminal's phase (degree <= 1 only).

        Returns:
        -------
        ZxPoly | None
            Updated phase after absorption.
        """
        if phase.degree() > 1:
            return None
        if TerminalAbsorptionRule._is_degenerate_angle(theta):
            return None
        coeffs = phase.coeffs
        c = coeffs.get(0, 0)
        k = coeffs.get(1, 0)
        return ZxPoly({0: c, 1: k / cos(theta), 2: -tan(theta) / 2})

    @staticmethod
    def _squeeze_absorb(phase: ZxPoly, tau: float | Expr) -> ZxPoly:
        """Fold a squeezing Sq(tau) into a terminal's phase (any degree).

        x -> x/tau leaves each x**d term's degree unchanged and divides
        its coefficient by tau**d (built via the dict form -- ZxPoly's
        expr+gen constructor path has a latent bug, see base_gates.py).

        Returns:
        -------
        ZxPoly
            Update phase after applying the Squeezing rule.

        References:
        ----------
        [1] Nagayoshi et al., CV ZX calculus, 2024, Eq. (82)-(83).
        """
        return ZxPoly({degree: coeff / tau**degree for degree, coeff in phase.coeffs.items()})

    @staticmethod
    def _is_degenerate_angle(theta: float | Expr) -> bool:
        """True if theta is an odd multiple of pi/2 (tan/1-over-cos undefined).

        Returns:
        -------
        bool
        """
        try:
            theta_val = float(theta)
        except (TypeError, ValueError):
            return False  # Symbolic theta: can't determine, assume safe.
        return math.isclose(abs(theta_val) % math.pi, math.pi / 2)

    def apply_single(self, graph: nx.DiGraph, match: dict) -> None:
        """Apply a terminal absorption to a specific match in-place.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        match : dict
            Match containing fold information.
        """
        container_id = match["container_id"]
        keep_id, absorb_id = match["node_ids"]
        i, _ = match["indices"]

        nx.contracted_nodes(graph, keep_id, absorb_id, self_loops=False, copy=False)
        if "contraction" in graph.nodes[keep_id]:
            del graph.nodes[keep_id]["contraction"]

        graph.nodes[keep_id].update({
            "type": match["result_type"],
            "phase": match["result_phase"],
            "num_inputs": match["result_num_inputs"],
            "num_outputs": match["result_num_outputs"],
            "container_id": container_id,
        })

        container_attrs = graph.nodes[container_id]
        sub_ids = container_attrs.get("sub_diagram_ids", [])
        new_sub_ids = [sub_id for sub_id in sub_ids if sub_id != absorb_id]
        container_attrs["sub_diagram_ids"] = new_sub_ids

        # Drop the now-internal link at i, move i+1's outgoing link to i,
        # and shift every index past i+1 down by one.
        if "connectivity" in container_attrs:
            connectivity = container_attrs["connectivity"]
            new_connectivity = {}
            for old_idx, targets in connectivity.items():
                if old_idx == i:
                    continue
                if old_idx == i + 1:
                    new_connectivity[i] = targets
                elif old_idx < i:
                    new_connectivity[old_idx] = targets
                else:  # old_idx > i + 1
                    new_connectivity[old_idx - 1] = targets
            container_attrs["connectivity"] = new_connectivity

        if len(new_sub_ids) == 1:
            self._flatten_container(graph, container_id)


class CopyRule(RewriteRule):
    r"""Copy rule - Graph-based version.

    The copy rule copies a spider with arity (0,1) or (1,0) through a spider
    with arity (1,n) or (n,1), producing n copies in a tensor diagram.

    Cases:
    1. P(φ, 1, n) ∘ Q(g, 0, 1) → Q(g, 0, 1) ⊗ ... ⊗ Q(g, 0, 1) (n copies)
    2. Q(g, 1, 0) ∘ P(φ, n, 1) → Q(g, 1, 0) ⊗ ... ⊗ Q(g, 1, 0) (n copies)
    3. Q(φ, 1, n) ∘ P(g, 0, 1) → P(g, 0, 1) ⊗ ... ⊗ P(g, 0, 1) (n copies)
    4. P(g, 1, 0) ∘ Q(φ, n, 1) → P(g, 1, 0) ⊗ ... ⊗ P(g, 1, 0) (n copies)

    Constraints:
    - The copied spider's phase g MUST be in R₁[X] (degree ≤ 1)
    - The disappearing spider's phase φ can be ANY polynomial
    - The result is a TensorDiagram of n copies of the copied spider
    - The composition is removed and replaced by a tensor diagram

    The two spiders need not be direct siblings of one flat
    CompositionDiagram. `match()` finds them via the graph's own
    "composition" edges, which already connect fully-resolved leaf nodes
    regardless of how deeply either sits inside a TensorDiagram or
    ContractedDiagram.
    """

    def match(self, graph: nx.DiGraph, registry: GateRegister) -> list[dict]:
        """Find all composition-adjacent copy-able patterns in the graph.

        Copy-able pairs are found by scanning the graph's own "composition"
        edges rather than one CompositionDiagram's flat `sub_diagram_ids`
        list. Every composition edge already connects fully-resolved proper
        (leaf) nodes regardless of how deeply either endpoint is nested
        inside TensorDiagram/ContractedDiagram containers (that resolution
        is exactly what `external_input_mapping`/`external_output_mapping`
        are used for when the graph is first built) -- so this also finds
        pairs that cross a TensorDiagram or ContractedDiagram boundary.

        Two shapes are recognized:

        - Same immediate parent (both nodes are direct, necessarily
          adjacent, entries of one flat CompositionDiagram): resolved via
          that container's own `sub_diagram_ids`/`connectivity`, exactly as
          before.
        - Cross-container: the two nodes have different immediate parents.
          Only combinations this rule knows how to restructure are matched:
          the disappearing spider's parent must be a TensorDiagram (direct
          list substitution) or a ContractedDiagram (`first_id`/`second_id`
          substitution); the copy spider's parent must be a TensorDiagram
          or a flat CompositionDiagram where the copy spider sits at one of
          the two ends (a state must be first, an effect must be last, so
          removing it never requires splicing two neighbors together).

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
            - 'copy_spider_id': the node ID of the spider to copy (Q or P with 0→1 or 1→0)
            - 'disappearing_spider_id': the node ID of the spider that disappears (P or Q with 1→n or n→1)
            - 'copy_spider_phase': the phase of the copied spider (g ∈ R₁[X])
            - 'copy_spider_type': 'Q' or 'P'
            - 'n_copies': number of copies to create
            - 'same_parent': whether both nodes share one flat CompositionDiagram parent
            - 'container_id': grouping key for `RewriteRule.apply_rule()`
            - additional container bookkeeping fields used by `apply_single`
        """
        del registry
        matches = []

        # Sort for deterministic match order (dict/edge iteration order
        # isn't guaranteed to reflect any particular scan order).
        composition_edges = sorted(
            (u, v) for u, v, edge_attrs in graph.edges(data=True) if edge_attrs.get("edge_type") == "composition"
        )

        for first_id, second_id in composition_edges:
            base_match = self._check_pair(graph, first_id, second_id)
            if base_match is None:
                continue

            resolved = self._resolve_containers(graph, base_match)
            if resolved is None:
                continue

            base_match.update(resolved)
            matches.append(base_match)

        return matches

    def _resolve_containers(self, graph: nx.DiGraph, base_match: dict) -> dict | None:
        """Work out how to restructure the containers around a candidate match.

        Returns:
        -------
        dict | None
            Extra fields to merge into the match, or None if this pair's
            container arrangement isn't one this rule can restructure.
        """
        copy_id = base_match["copy_spider_id"]
        disappear_id = base_match["disappearing_spider_id"]
        copy_parent_id = graph.nodes[copy_id].get("container_id")
        disappear_parent_id = graph.nodes[disappear_id].get("container_id")
        copy_parent_type = graph.nodes[copy_parent_id].get("container_type") if copy_parent_id is not None else None
        disappear_parent_type = (
            graph.nodes[disappear_parent_id].get("container_type") if disappear_parent_id is not None else None
        )

        if copy_parent_id == disappear_parent_id:
            # Classic case
            if copy_parent_type != "composition":
                return None
            sub_ids = graph.nodes[copy_parent_id].get("sub_diagram_ids", [])
            if copy_id not in sub_ids or disappear_id not in sub_ids:
                return None
            indices = sorted([sub_ids.index(copy_id), sub_ids.index(disappear_id)])
            return {
                "same_parent": True,
                "container_id": copy_parent_id,
                "indices": indices,
            }

        # Cross-container case:
        if copy_parent_type != "tensor":
            return None
        if disappear_parent_type not in {"tensor", "contracted"}:
            return None

        return {
            "same_parent": False,
            "copy_container_id": copy_parent_id,
            "copy_container_type": copy_parent_type,
            "disappear_container_id": disappear_parent_id,
            "disappear_container_type": disappear_parent_type,
            "container_id": ("cross", copy_id, disappear_id),
        }

    def _check_pair(self, graph: nx.DiGraph, first_id: int, second_id: int) -> dict | None:
        """Check if a pair of nodes forms a copy-able pattern.

        Returns:
        -------
        dict | None
            Match dictionary if the pair is copy-able, None otherwise.
        """
        first_attrs = graph.nodes[first_id]
        second_attrs = graph.nodes[second_id]

        first_type = first_attrs.get("type")
        second_type = second_attrs.get("type")

        first_num_inputs = first_attrs.get("num_inputs")
        first_num_outputs = first_attrs.get("num_outputs")
        second_num_inputs = second_attrs.get("num_inputs")
        second_num_outputs = second_attrs.get("num_outputs")

        # Pattern: P(φ, 1, n) ∘ Q(g, 0, 1)
        # `not is_wiring_node_from_attrs(second_attrs)` excludes the
        # degenerate n=1, zero-phase case -- a state feeding a literal
        # identity wire is nothing to COPY through, it's IdentityRule's
        # job (splice the wire, leave the state as-is); without this
        # guard the pattern below still matches it (num_outputs>=1 is
        # satisfied at n=1), and folding it through `_create_match`'s
        # general n-copy machinery corrupts the surrounding structure
        # instead of just eliding a no-op wire, since that machinery
        # assumes the "disappearing" spider is a genuine terminal
        # absorption point, not a mid-chain pass-through with its own
        # unrelated incoming/outgoing wiring on either side.
        if (
            first_type == "QSpider"  # noqa: PLR0916
            and first_num_inputs == 0
            and first_num_outputs == 1
            and second_type == "PSpider"
            and second_num_inputs == 1
            and second_num_outputs >= 1
            and not is_wiring_node_from_attrs(second_attrs)
        ):
            return self._create_match(
                graph,
                copy_spider_id=first_id,
                disappearing_spider_id=second_id,
                n_copies=second_num_outputs,
                copy_spider_type="Q",
            )

        # Pattern: Q(g, 1, 0) ∘ P(φ, n, 1)
        if (
            first_type == "PSpider"  # noqa: PLR0916
            and first_num_inputs >= 1
            and first_num_outputs == 1
            and second_type == "QSpider"
            and second_num_inputs == 1
            and second_num_outputs == 0
            and not is_wiring_node_from_attrs(first_attrs)
        ):
            return self._create_match(
                graph,
                copy_spider_id=second_id,
                disappearing_spider_id=first_id,
                n_copies=first_num_inputs,
                copy_spider_type="Q",
            )

        # Pattern: Q(φ, 1, n) ∘ P(g, 0, 1)
        if (
            first_type == "PSpider"  # noqa: PLR0916
            and first_num_inputs == 0
            and first_num_outputs == 1
            and second_type == "QSpider"
            and second_num_inputs == 1
            and second_num_outputs >= 1
            and not is_wiring_node_from_attrs(second_attrs)
        ):
            return self._create_match(
                graph,
                copy_spider_id=first_id,
                disappearing_spider_id=second_id,
                n_copies=second_num_outputs,
                copy_spider_type="P",
            )

        # Pattern: P(g, 1, 0) ∘ Q(φ, n, 1)
        if (
            first_type == "QSpider"  # noqa: PLR0916
            and first_num_inputs >= 1
            and first_num_outputs == 1
            and second_type == "PSpider"
            and second_num_inputs == 1
            and second_num_outputs == 0
            and not is_wiring_node_from_attrs(first_attrs)
        ):
            return self._create_match(
                graph,
                copy_spider_id=second_id,
                disappearing_spider_id=first_id,
                n_copies=first_num_inputs,
                copy_spider_type="P",
            )

        return None

    def _create_match(
        self,
        graph: nx.DiGraph,
        copy_spider_id: int,
        disappearing_spider_id: int,
        n_copies: int,
        copy_spider_type: str,
    ) -> dict | None:
        """Create a match dictionary if the copied spider's phase is in R₁[X]."""  # noqa: DOC201
        copy_attrs = graph.nodes[copy_spider_id]
        copy_phase = copy_attrs.get("phase")

        # Check that the copied spider's phase is in R₁[X] (degree ≤ 1)
        if not self.is_in_R1(copy_phase):
            return None

        return {
            "copy_spider_id": copy_spider_id,
            "disappearing_spider_id": disappearing_spider_id,
            "n_copies": n_copies,
            "copy_spider_type": copy_spider_type,
            "copy_spider_phase": copy_phase,
            "copy_spider_num_inputs": copy_attrs.get("num_inputs"),
            "copy_spider_num_outputs": copy_attrs.get("num_outputs"),
        }

    def is_in_R1(self, phase: ZxPoly) -> bool:
        """Check if a phase polynomial is in R₁[X] (degree ≤ 1).

        Parameters:
        ----------
        phase : ZxPoly
            The phase polynomial to check.

        Returns:
        -------
        bool
            True if the polynomial has degree ≤ 1, False otherwise.
        """
        if phase is None:
            return False
        return phase.degree() <= 1

    def apply_single(self, graph: nx.DiGraph, match: dict) -> None:  # noqa: PLR0912, PLR0914, PLR0915, C901
        """Apply the copy rule to a specific match in-place.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        match : dict
            Match containing the copy/disappearing spider information plus
            whatever container bookkeeping `match()` resolved for this pair.
        """
        copy_spider_id = match["copy_spider_id"]
        disappearing_spider_id = match["disappearing_spider_id"]
        n_copies = match["n_copies"]
        copy_spider_type = match["copy_spider_type"]
        copy_spider_phase = match["copy_spider_phase"]
        copy_num_inputs = match["copy_spider_num_inputs"]
        copy_num_outputs = match["copy_spider_num_outputs"]
        same_parent = match.get("same_parent", True)

        if same_parent:
            container_id = match["container_id"]
            idx1, idx2 = match["indices"]
            container_attrs = graph.nodes[container_id]
            sub_ids = container_attrs.get("sub_diagram_ids", [])

        # Contract the disappearing spider into the copy spider.
        # This preserves all connections automatically.
        nx.contracted_nodes(graph, copy_spider_id, disappearing_spider_id, self_loops=False, copy=False)

        # Remove the contraction metadata that NetworkX adds
        if "contraction" in graph.nodes[copy_spider_id]:
            del graph.nodes[copy_spider_id]["contraction"]

        # Now transform the copy spider into a TensorDiagram container
        # Create n copies of the spider to place inside the tensor
        copy_ids = []
        for _ in range(n_copies):
            new_id = max(graph.nodes) + 1 if graph.nodes else 0
            graph.add_node(
                new_id,
                id=new_id,
                type=f"{copy_spider_type}Spider",
                kind="proper",
                phase=copy_spider_phase,
                num_inputs=copy_num_inputs,
                num_outputs=copy_num_outputs,
                container_id=copy_spider_id,
                external_inputs=list(range(copy_num_inputs)),
                external_outputs=list(range(copy_num_outputs)),
            )
            copy_ids.append(new_id)

        # Transform the copy_spider_id node from a proper spider into a TensorDiagram container
        graph.nodes[copy_spider_id].update({
            "type": "TensorDiagram",
            "kind": "container",
            "container_type": "tensor",
            "sub_diagram_ids": copy_ids,
            "num_inputs": n_copies * copy_num_inputs,
            "num_outputs": n_copies * copy_num_outputs,
            # Keep the container_id and is_root from the original
            # Remove spider-specific attributes
            "phase": None,
        })
        # Build a proper port-mapping dict (port -> (child_index,
        # internal_port)), not a bare list -- `_recompute_tensor_arity`
        # (called below in the cross-container branch, folding in the
        # pad) reads this as a dict to compute its old->new port remap.
        input_mapping, offset = {}, 0
        for idx, sub_id in enumerate(copy_ids):
            for p in range(graph.nodes[sub_id].get("num_inputs", 0)):
                input_mapping[offset] = (idx, p)
                offset += 1
        output_mapping, offset = {}, 0
        for idx, sub_id in enumerate(copy_ids):
            for p in range(graph.nodes[sub_id].get("num_outputs", 0)):
                output_mapping[offset] = (idx, p)
                offset += 1
        graph.nodes[copy_spider_id]["external_inputs"] = list(range(len(input_mapping)))
        graph.nodes[copy_spider_id]["external_outputs"] = list(range(len(output_mapping)))
        graph.nodes[copy_spider_id]["external_input_mapping"] = input_mapping
        graph.nodes[copy_spider_id]["external_output_mapping"] = output_mapping

        if same_parent:
            # Replace the two elements with the tensor in the CompositionDiagram
            new_sub_ids = [*sub_ids[:idx1], copy_spider_id, *sub_ids[idx2 + 1 :]]
            graph.nodes[container_id]["sub_diagram_ids"] = new_sub_ids

            # Drop the now-internal idx1 link, move idx2's outgoing link to idx1,
            # and shift every index past idx2 down by one.
            if "connectivity" in container_attrs:
                connectivity = container_attrs["connectivity"]
                new_connectivity = {}

                for old_idx, targets in connectivity.items():
                    if old_idx == idx1:
                        continue  # internal link between the merged pair
                    if old_idx == idx2:
                        new_connectivity[idx1] = targets  # fused node's outgoing link
                    elif old_idx < idx1:
                        new_connectivity[old_idx] = targets
                    else:  # old_idx > idx2
                        new_connectivity[old_idx - 1] = targets

                graph.nodes[container_id]["connectivity"] = new_connectivity
            if len(sub_ids) == 2:  # noqa: PLR2004
                self._flatten_container(graph, container_id)
            return

        # Cross-container case
        disappear_container_id = match["disappear_container_id"]
        disappear_container_type = match["disappear_container_type"]
        copy_container_id = match["copy_container_id"]

        # 1. Install the transformed copy spider at the disappearing
        #    spider's old slot, padded with a VoidDiagram sized to
        #    exactly the port(s) the contraction consumed: the single
        #    wire that used to connect copy_spider to disappearing_spider
        #    fed one of disappearing_spider's ports directly, and that
        #    port's count is exactly (copy_num_outputs, copy_num_inputs)
        #    -- an input if copy_spider was a state (0,1) feeding
        #    disappearing_spider's input, an output if copy_spider was an
        #    effect (1,0) fed by disappearing_spider's output. Padding
        #    with a same-shaped VoidDiagram makes the installed
        #    replacement's TOTAL arity come out identical to
        #    disappearing_spider's own ORIGINAL arity (the n copies
        #    supply exactly its other, unconsumed ports), so
        #    disappear_container_id's own shape never changes either --
        #    no recompute or `_propagate_arity_to_parent` needed on this
        #    side at all, only a straight substitution of node IDs.
        pad_id = max(graph.nodes) + 1
        graph.add_node(
            pad_id,
            id=pad_id,
            type="VoidDiagram",
            kind="proper",
            phase=None,
            num_inputs=copy_num_outputs,
            num_outputs=copy_num_inputs,
            container_id=copy_spider_id,
            external_inputs=list(range(copy_num_outputs)),
            external_outputs=list(range(copy_num_inputs)),
        )
        graph.nodes[copy_spider_id]["sub_diagram_ids"].append(pad_id)
        # Rebuild copy_spider_id's own port bookkeeping fresh from its
        # (now n_copies + 1) children -- `_recompute_tensor_arity` is
        # generic over the full child list, not just a substitution, so
        # this correctly folds the pad in. The returned remap is unused:
        # by construction the new total matches disappearing_spider's
        # old arity exactly, so nothing above `copy_spider_id` needs it.
        self._recompute_tensor_arity(graph, copy_spider_id)

        if disappear_container_type == "contracted":
            disappear_parent_attrs = graph.nodes[disappear_container_id]
            if disappear_parent_attrs.get("first_id") == disappearing_spider_id:
                disappear_parent_attrs["first_id"] = copy_spider_id
            else:
                disappear_parent_attrs["second_id"] = copy_spider_id
        else:  # "tensor"
            disappear_sub_ids = graph.nodes[disappear_container_id]["sub_diagram_ids"]
            disappear_sub_ids[disappear_sub_ids.index(disappearing_spider_id)] = copy_spider_id
        graph.nodes[copy_spider_id]["container_id"] = disappear_container_id

        # 2. The copy spider's own original slot has nothing left to put
        #    there -- its content now lives as the n copies installed in
        #    step 1. Rather than deleting the slot (which would shrink
        #    copy_container_id's arity and force `_propagate_arity_to_parent`
        #    to cascade the change through every parent container above
        #    it), install a `VoidDiagram` of the exact same arity in its
        #    place. copy_container_id is always a TensorDiagram here (see
        #    `_resolve_containers`), so this is a same-index list swap:
        #    the container's own shape never changes -- no propagation,
        #    no flattening, nothing upstream needs to be touched at all.
        #    `VoidDiagram` is meant to be transient: the end-of-pipeline
        #    cleanup pass removes it for good once optimization is done.
        void_id = max(graph.nodes) + 1 if graph.nodes else 0
        graph.add_node(
            void_id,
            id=void_id,
            type="VoidDiagram",
            kind="proper",
            phase=None,
            num_inputs=copy_num_inputs,
            num_outputs=copy_num_outputs,
            container_id=copy_container_id,
            external_inputs=list(range(copy_num_inputs)),
            external_outputs=list(range(copy_num_outputs)),
        )
        copy_sub_ids = graph.nodes[copy_container_id]["sub_diagram_ids"]
        copy_sub_ids[copy_sub_ids.index(copy_spider_id)] = void_id


class IdentityRemovalRule(RewriteRule):
    """End-of-pipeline cleanup: strip transient `VoidDiagram` placeholders.

    Unlike every other rule in this module, this is NOT meant to run
    inside `optimize()`'s per-round rewrite loop -- see
    `remove_void_and_identity_nodes`, which runs it exactly ONCE, after
    that loop has already reached a fixed point, right before lowering to
    a machine representation.

    A `VoidDiagram` is a bookkeeping artifact `CopyRule` leaves behind so
    it never has to shrink a container's arity mid-optimization (see
    `VoidDiagram`'s docstring in `base_gates.py`): it reserves the exact
    layout space the vanished leg used to occupy, but represents no real
    wire or state any more. This rule performs the REAL removal that
    `CopyRule` deferred: for each `VoidDiagram`, shrink its TensorDiagram
    parent's arity by exactly the Void's own arity (the leg it stood in
    for really is gone for good) and ripple that change upward, exactly
    like removing any other TensorDiagram child.

    `VoidDiagram` is only ever installed as a TensorDiagram child (see
    both `CopyRule.apply_single` call sites); a Void found anywhere else
    is left untouched rather than guessed at.
    """

    def match(self, graph: nx.DiGraph, registry: GateRegister) -> list[dict]:
        """Find every remaining `VoidDiagram` node in the graph.

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
            - 'node_id': the node ID of the VoidDiagram
            - 'container_id': the node ID of its immediate container
        """
        return [
            {"node_id": node, "container_id": graph.nodes[node].get("container_id")}
            for node in sorted(registry.void_nodes)
            if node in graph.nodes
        ]

    def apply_single(self, graph: nx.DiGraph, match: dict) -> None:
        """Remove one VoidDiagram from the graph in-place, shrinking its parent.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        match : dict
            Match containing 'node_id' (the container is re-resolved fresh
            here rather than trusted from `match`, since an earlier
            `apply_single` call in the same pass -- e.g. removing a
            sibling Void down to a single remaining child -- can have
            flattened this Void's container away in the meantime).
        """
        node_id = match["node_id"]
        if node_id not in graph.nodes:
            return
        attrs = graph.nodes[node_id]
        if attrs.get("type") != "VoidDiagram":
            return

        container_id = attrs.get("container_id")
        if container_id is None or container_id not in graph.nodes:
            # A root-level Void (the whole diagram reduced to a vanished
            # leg) has no container to shrink into -- just drop it.
            graph.remove_node(node_id)
            return

        container_attrs = graph.nodes[container_id]
        if container_attrs.get("container_type") != "tensor":
            # CopyRule only ever installs a VoidDiagram as a TensorDiagram
            # child; anything else is unexpected, so leave it alone rather
            # than guess at how to shrink an unfamiliar container type.
            return

        arity_change = self._remove_tensor_child(
            graph,
            container_id,
            node_id,
            attrs.get("num_inputs", 0),
            attrs.get("num_outputs", 0),
        )
        graph.remove_node(node_id)
        self._propagate_arity_to_parent(graph, container_id, *arity_change)
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

    num_inputs = attrs.get("num_inputs", 0)
    num_outputs = attrs.get("num_outputs", 0)
    if num_inputs != 1 or num_outputs != 1:
        return False

    phase = attrs.get("phase")
    if phase is None:
        return False

    return phase.is_zero


def remove_void_and_identity_nodes(graph: nx.DiGraph) -> nx.DiGraph:
    """End-of-pipeline cleanup: strip `VoidDiagram` placeholders and stray identities.

    Meant to run exactly ONCE, after `optimize()`'s round loop has already
    reached a fixed point -- never as one of the per-round rules -- since
    removing a `VoidDiagram` genuinely shrinks its parent's arity (no rule
    that runs mid-optimization does that; `CopyRule` deliberately avoids
    it, padding with a `VoidDiagram` instead precisely so container shapes
    stay stable while more rules may still need to match against them).

    Two passes, in order:

    1. `IdentityRemovalRule` removes every remaining `VoidDiagram`,
       shrinking its TensorDiagram parent's arity by the Void's own arity
       and rippling that change upward -- the real removal `CopyRule`
       deferred.
    2. `IdentityRule` is then run to a fresh fixed point: flattening a
       container down to a single child (step 1's `_flatten_container`
       calls, triggered when removing a Void leaves only one sibling
       behind) can expose an identity spider that was previously nested
       too deep for `IdentityRule` to reach on its own -- it only matches
       an identity whose OWN immediate parent is a flat "composition"
       container.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to clean up in-place (already at a rewrite fixed point).

    Returns:
    -------
    nx.DiGraph
        The same graph object, for chaining.
    """
    void_registry = GateRegister()
    void_registry.build_from_graph(graph)
    IdentityRemovalRule().apply_rule(graph, void_registry)

    identity_rule = IdentityRule()
    while True:
        registry = GateRegister()
        registry.build_from_graph(graph)
        if not identity_rule.match(graph, registry):
            break
        identity_rule.apply_rule(graph, registry)

    return graph


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


def expand_two_mode_gates(diagram: Diagram) -> Diagram:
    """Recursively expand BeamsplitterGate/ControlledSumGate, but not CZ.

    Like `cvzx.gates.expand_all`, but deliberately excludes
    `ControlledZGate`: CZ's own decomposition sandwiches a Fourier gate
    between its copy spider and the target mode, so `CopyRule` can't reach
    across it -- expanding CZ wouldn't unlock any new reduction, unlike
    BS/CSUM, whose expansions expose a bare copy spider `CopyRule` can act
    on directly.

    This function is only ever called under `assume_infinite_squeezing=True`
    (see `optimize()`), which is also what licenses a second simplification
    applied here: a biased `ControlledSumGate(gain=g != 1)` is normalized to
    `gain=1` before expanding, rather than expanded via its own squeeze-CSUM-
    unsqueeze decomposition. This is the Squeezing rule: under the infinite-
    squeezing assumption the modes involved are idealized eigenstates, for
    which CSUM(g) and CSUM(1) act identically, so the bias can simply be
    dropped instead of being carried through as an explicit pair of
    `SqueezingGate`s. Doing it this way also sidesteps a real limitation of
    `.expand()`'s biased decomposition: the squeeze gates it introduces land
    in a different container than whatever state feeds the CSUM, so
    `TerminalAbsorptionRule`/`CopyRule` (which only match within a single
    container) can't reach through them, and a biased CSUM fed by a
    reducible ancilla would otherwise fail to reduce at all. `CZ` is left
    completely unexpanded (and hence unnormalized), per the docstring above.

    Parameters:
    ----------
    diagram : Diagram
        The diagram to modify.

    Returns:
    -------
    Diagram
        Expanded diagram.
    """
    if isinstance(diagram, ControlledSumGate):
        if diagram.gain != 1:
            diagram = ControlledSumGate(control=diagram.control, target=diagram.target)
        return diagram.expand()
    if isinstance(diagram, BeamsplitterGate):
        return diagram.expand()
    if isinstance(diagram, CompositionDiagram):
        return _flatten_expanded_composition(
            [expand_two_mode_gates(d) for d in diagram.diagrams], diagram.connectivity
        )
    if isinstance(diagram, TensorDiagram):
        return TensorDiagram([expand_two_mode_gates(d) for d in diagram.diagrams])
    if isinstance(diagram, ContractedDiagram):
        return ContractedDiagram(
            expand_two_mode_gates(diagram.first),
            expand_two_mode_gates(diagram.second),
            diagram.I1,
            diagram.I2,
            diagram.J1,
            diagram.J2,
        )
    return diagram


def _flatten_expanded_composition(
    expanded_children: list[Diagram], connectivity: dict[int, dict[int, int]]
) -> CompositionDiagram:
    """Splice a child that expanded into its own CompositionDiagram into the parent's flat list.

    `to_graph()`'s composition-edge resolution (`find_node_by_external_output`/
    `find_node_by_external_input` in `cvzx.nx_graph`) recurses into a
    composition's TENSOR/CONTRACTED children -- both store an
    `external_*_mapping` -- but never into a CompositionDiagram nested
    directly inside another CompositionDiagram, since composition containers
    don't store one (a flat composition's own boundary edges are resolved
    from `sub_diagram_ids`/`connectivity` directly, see `_add_composition_node`).
    Left un-flattened, any wire crossing such a nested boundary is silently
    dropped -- exactly what `BeamsplitterGate.expand()`'s balanced case
    produces (a `CompositionDiagram` of two expanded `ControlledSumGate`s and
    a `TensorDiagram` of squeezing gates), so expanding a two-mode gate that
    sits alongside other elements in a composition must flatten the result
    back into one flat list rather than nesting it.

    Parameters:
    ----------
    expanded_children : list[Diagram]
        The already-`expand_two_mode_gates`-processed children, in order.
    connectivity : dict[int, dict[int, int]]
        The ORIGINAL (pre-expansion) parent composition's connectivity dict.
        Expansion never changes a diagram's external arity, so this still
        correctly describes each boundary `i` -> `i + 1` in terms of the
        un-flattened children's own port numbering.

    Returns:
    -------
    CompositionDiagram
        A flat composition with every formerly-nested CompositionDiagram
        child spliced directly into the list, and connectivity re-indexed
        to match the new, flat positions.
    """
    flat_diagrams: list[Diagram] = []
    flat_connectivity: dict[int, dict[int, int]] = {}

    for i, child in enumerate(expanded_children):
        start_pos = len(flat_diagrams)

        if isinstance(child, CompositionDiagram):
            flat_diagrams.extend(child.diagrams)
            for key, value in child.connectivity.items():
                flat_connectivity[start_pos + key] = value
        else:
            flat_diagrams.append(child)

        if i > 0:
            # The boundary between child i-1 and child i is, in the flat
            # list, the boundary between whatever position child i-1's own
            # contribution ended on (start_pos - 1) and whatever position
            # child i's contribution begins on (start_pos) -- regardless of
            # whether either side was itself just spliced in.
            flat_connectivity[start_pos - 1] = connectivity[i - 1]

    return CompositionDiagram(flat_diagrams, flat_connectivity)

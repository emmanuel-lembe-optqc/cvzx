"""CV ZX calculus rewrite rules and applications.

This module implements the 10 basic rewrite rules from [1] Sec. IV.A,
and the derived rules from Sec. IV.B; using a graph structure.

References
----------
[1] Nagayoshi et al., CV ZX calculus, 2024
"""

import logging
import math
import operator
from abc import ABC, abstractmethod
from typing import Any

import networkx as nx
from sympy import Expr, cos, pi, tan

from cvzx.base_gates import CompositionDiagram, ContractedDiagram, Diagram, TensorDiagram, ZxPoly
from cvzx.gates import BeamsplitterGate, ControlledSumGate
from cvzx.nx_graph import (
    CVZXGraph,
    to_diagram,
    to_graph,
)

logger = logging.getLogger(__name__)


class RewriteRule(ABC):
    """Base class for rewrite rules operating on graphs.

    All rewrite rules operate on a `CVZXGraph` in-place (mutating its
    underlying `.graph`). This avoids repeated conversions between diagram
    and graph representations.

    The typical workflow is:
        1. Convert diagram to graph: `G = to_graph(diagram)`
        2. Apply rules: `rule1.apply_rule(G); rule2.apply_rule(G)`
        3. Convert back: `diagram = to_diagram(G)`

    Subclasses must implement:
        - `match(G)`: Find all matches in the graph
        - `apply_single(G, match)`: Apply a single match in-place

    The `apply_rule` method provides a default implementation that:
        1. Finds all matches
        2. Returns the modified graph

    Parameters
    ----------
    graph : CVZXGraph
        The graph to modify.

    Returns
    -------
    CVZXGraph
        The modified graph (same object, for chaining).
    """

    @abstractmethod
    def match(self, graph: CVZXGraph) -> list[dict]:
        """Find all matches of the rule pattern in the graph.

        Parameters
        ----------
        graph : CVZXGraph
            The graph to search, together with its registry.

        Returns
        -------
        list[dict]
            List of matches. Each match is a dictionary that can be
            passed to `apply_single`.
        """

    @abstractmethod
    def apply_single(self, graph: CVZXGraph, match: dict) -> None:
        """Apply the rule to a specific match in-place.

        Parameters
        ----------
        graph : CVZXGraph
            The graph to modify.
        match : dict
            A single match returned by `match()`.
        """

    def apply_rule(self, graph: CVZXGraph) -> CVZXGraph:
        """Apply a rule to the entire graph.

        Parameters
        ----------
        graph : CVZXGraph
            The graph to modify, together with its registry.

        Returns
        -------
        CVZXGraph
            The modified graph.

        Raises
        ------
        TypeError
            If `self.match()` returns a match that is not a dict.
        """
        # Find all matches
        matches = self.match(graph)

        if not matches:
            logger.debug("%s: no matches", type(self).__name__)
            return graph

        # Group by container and apply each group's matches in reverse
        # position order, so an earlier removal can't shift a later match's index.
        groups: dict[object, list] = {}
        group_order: list[object] = []
        for match in matches:
            if not isinstance(match, dict):
                msg = (
                    f"{type(self).__name__}.match() returned a non-dict match "
                    f"({match!r}, type {type(match).__name__}); every match must be a dict."
                )
                raise TypeError(msg)
            key = match.get("container_id")
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(match)

        logger.debug("%s: %d match(es), applying", type(self).__name__, len(matches))
        for key in group_order:
            for match in reversed(groups[key]):
                self.apply_single(graph, match)

        return graph

    def _flatten_container(self, graph: nx.DiGraph, container_id: int) -> None:
        """Flatten a container if it has only one element.

        Parameters
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

    def _replace_in_parent(self, graph: nx.DiGraph, parent_id: int, old_id: int, new_id: int) -> None:
        """Substitute `new_id` for `old_id` in `parent_id`'s own bookkeeping.

        Used whenever a node is removed or merged away and something else
        (a survivor of a merge, or a same-shaped placeholder) needs to
        take over its exact slot -- a `sub_diagram_ids` list entry for a
        "composition"/"tensor" parent, or `first_id`/`second_id` for a
        "contracted" one. Never touches port counts or any other
        bookkeeping (`connectivity`, `I1`/`I2`/`J1`/`J2`, ...): the caller
        is responsible for only using this when `new_id`'s own arity
        matches whatever shape `old_id` occupied, so nothing downstream
        needs to change.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        parent_id : int
            The immediate parent container of `old_id`.
        old_id : int
            The node ID currently occupying the slot.
        new_id : int
            The node ID that should occupy it instead.
        """
        parent_attrs = graph.nodes[parent_id]
        parent_type = parent_attrs.get("container_type")

        if parent_type in {"composition", "tensor"}:
            sub_ids = parent_attrs.get("sub_diagram_ids", [])
            if old_id in sub_ids:
                sub_ids[sub_ids.index(old_id)] = new_id
        elif parent_type == "contracted":
            if parent_attrs.get("first_id") == old_id:
                parent_attrs["first_id"] = new_id
            elif parent_attrs.get("second_id") == old_id:
                parent_attrs["second_id"] = new_id

        if new_id in graph.nodes:
            graph.nodes[new_id]["container_id"] = parent_id

    def _install_void_placeholder(
        self, graph: nx.DiGraph, parent_id: int, old_id: int, num_inputs: int, num_outputs: int
    ) -> int:
        """Replace `old_id`'s slot in `parent_id` with a same-shaped `VoidDiagram`.

        Used when a node disappears (merged into something elsewhere in
        the graph) but its own parent's shape must stay exactly as it
        was -- no `connectivity`/arity recompute, no propagation to
        grandparents -- because every port count on both sides of the old
        slot is preserved exactly. `VoidDiagram` is a transient
        placeholdere -- the same trade-off `CopyRule`'s own
        cross-container substitution already makes.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        parent_id : int
            The immediate parent container `old_id` used to occupy.
        old_id : int
            The node ID that no longer exists (already merged away).
        num_inputs : int
            The placeholder's input arity (== `old_id`'s own, before it
            disappeared).
        num_outputs : int
            The placeholder's output arity (== `old_id`'s own).

        Returns
        -------
        int
            The new placeholder's node ID.
        """
        void_id = max(graph.nodes) + 1 if graph.nodes else 0
        graph.add_node(
            void_id,
            id=void_id,
            type="VoidDiagram",
            kind="proper",
            phase=None,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            container_id=parent_id,
            external_inputs=list(range(num_inputs)),
            external_outputs=list(range(num_outputs)),
        )
        self._replace_in_parent(graph, parent_id, old_id, void_id)
        return void_id

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

        Returns
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

    def _recompute_contracted_arity(self, graph: nx.DiGraph, container_id: int) -> tuple:  # ruff: ignore[too-many-locals]
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

        Returns
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

        Returns
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
    def _recompute_tensor_arity_from_child_remap(  # ruff: ignore[complex-structure, too-many-arguments, too-many-positional-arguments]
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

        Returns
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
    def _remove_tensor_child(  # ruff: ignore[complex-structure]
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

        Returns
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

    def _remove_composition_child(self, graph: nx.DiGraph, container_id: int, removed_id: int) -> tuple:
        """Remove one child from a flat CompositionDiagram container in place.

        Only valid for a child that carries nothing any neighbor still
        depends on -- concretely, a `VoidDiagram`, which by construction
        (every call site in this module) always has zero inputs XOR zero
        outputs, with its one nonzero side never wired to anything by
        `connectivity` (see `IdentityRemovalRule`, the only caller). This
        does not attempt to merge a live wire across the gap; it is not a
        general "splice out any composition child" operation.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        container_id : int
            The CompositionDiagram container node ID.
        removed_id : int
            The child node ID being removed.

        Returns
        -------
        tuple
            (old_num_inputs, new_num_inputs, old_num_outputs,
            new_num_outputs, input_remap, output_remap) for
            `_propagate_arity_to_parent`.
        """
        attrs = graph.nodes[container_id]
        sub_ids = attrs.get("sub_diagram_ids", [])
        pos = sub_ids.index(removed_id)
        n = len(sub_ids)

        old_connectivity = attrs.get("connectivity", {})
        new_connectivity = {}
        for key, targets in old_connectivity.items():
            if key in {pos - 1, pos}:
                continue  # either side of the removed slot -- nothing to merge, see docstring
            new_connectivity[key - 1 if key > pos else key] = targets

        sub_ids.pop(pos)
        attrs["connectivity"] = new_connectivity

        old_num_inputs = attrs.get("num_inputs", 0)
        old_num_outputs = attrs.get("num_outputs", 0)
        new_num_inputs = old_num_inputs
        new_num_outputs = old_num_outputs
        input_remap: dict = {}
        output_remap: dict = {}

        if pos == 0:
            new_num_inputs = graph.nodes[sub_ids[0]].get("num_inputs", 0) if sub_ids else 0
            if new_num_inputs == old_num_inputs:
                input_remap = {p: p for p in range(old_num_inputs)}

        if pos == n - 1:
            new_num_outputs = graph.nodes[sub_ids[-1]].get("num_outputs", 0) if sub_ids else 0
            if new_num_outputs == old_num_outputs:
                output_remap = {p: p for p in range(old_num_outputs)}

        attrs.update({
            "num_inputs": new_num_inputs,
            "num_outputs": new_num_outputs,
            "external_inputs": list(range(new_num_inputs)),
            "external_outputs": list(range(new_num_outputs)),
        })

        return old_num_inputs, new_num_inputs, old_num_outputs, new_num_outputs, input_remap, output_remap

    def _remove_external_port(self, graph: nx.DiGraph, node_id: int, is_input: bool, port: int) -> bool:  # ruff: ignore[complex-structure, too-many-return-statements, too-many-locals, boolean-type-hint-positional-argument]
        """Try to eliminate one external port of `node_id`, recursing through wrappers.

        Succeeds (returns True, graph mutated) only when `port` traces
        back through any nesting of TensorDiagram/ContractedDiagram/
        CompositionDiagram wrappers to an existing `VoidDiagram` leaf --
        shrinking a Void is always safe, since it carries no physical
        content or connectivity that must be preserved. Refuses (returns
        False, graph untouched) the instant the port traces back to a
        real physical leaf, since removing a genuine port would silently
        drop real connectivity.

        Used by `_propagate_arity_to_parent`'s "composition" branch to
        attempt a *consistent* shrink of an adjacent sibling before
        falling back to today's silent `connectivity`-entry drop, which
        otherwise leaves that sibling's own declared arity stale and
        eventually trips `CompositionDiagram.__post_init__`'s strict
        adjacent-size check.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        node_id : int
            The node whose external port should be removed.
        is_input : bool
            True to remove one of `node_id`'s input ports, False for one
            of its output ports.
        port : int
            The external port index (in the `is_input` side's own
            numbering) to remove.

        Returns
        -------
        bool
            Whether the removal succeeded.
        """
        attrs = graph.nodes[node_id]
        container_type = attrs.get("container_type")

        if container_type is None:
            if attrs.get("type") != "VoidDiagram":
                return False
            count_key = "num_inputs" if is_input else "num_outputs"
            list_key = "external_inputs" if is_input else "external_outputs"
            count = attrs.get(count_key, 0)
            if port >= count:
                return False
            attrs[count_key] = count - 1
            attrs[list_key] = list(range(count - 1))
            return True

        if container_type == "tensor":
            mapping_key = "external_input_mapping" if is_input else "external_output_mapping"
            target = attrs.get(mapping_key, {}).get(port)
            if target is None:
                return False
            child_idx, child_port = target
            child_id = attrs.get("sub_diagram_ids", [])[child_idx]
            if not self._remove_external_port(graph, child_id, is_input, child_port):
                return False
            self._recompute_tensor_arity(graph, node_id)
            return True

        if container_type == "contracted":
            mapping_key = "external_input_mapping" if is_input else "external_output_mapping"
            target = attrs.get(mapping_key, {}).get(port)
            if target is None:
                return False
            side, local_port = target
            child_id = attrs["first_id"] if side == "first" else attrs["second_id"]
            if not self._remove_external_port(graph, child_id, is_input, local_port):
                return False
            # `child_id`'s own raw port numbering above `local_port` just
            # shifted down by one; translate whichever consumed-index
            # list tracks this (side, is_input) combination the same way.
            index_key = self._contracted_index_key(side, is_input)
            shifted = [p if p < local_port else p - 1 for p in attrs.get(index_key, []) if p != local_port]
            attrs[index_key] = shifted
            self._recompute_contracted_arity(graph, node_id)
            return True

        if container_type == "composition":
            sub_ids = attrs.get("sub_diagram_ids", [])
            if not sub_ids:
                return False
            target_id = sub_ids[0] if is_input else sub_ids[-1]
            if not self._remove_external_port(graph, target_id, is_input, port):
                return False
            count_key = "num_inputs" if is_input else "num_outputs"
            list_key = "external_inputs" if is_input else "external_outputs"
            count = attrs.get(count_key, 0)
            attrs[count_key] = count - 1
            attrs[list_key] = list(range(count - 1))
            return True

        return False

    @staticmethod
    def _contracted_index_key(side: str, is_input: bool) -> str:  # ruff: ignore[boolean-type-hint-positional-argument]
        """Which of I1/J1/I2/J2 tracks consumed ports for (side, is_input).

        Mirrors `_recompute_contracted_arity`'s own convention: J1 =
        first's consumed inputs, I1 = first's consumed outputs, I2 =
        second's consumed inputs, J2 = second's consumed outputs.

        Returns
        -------
        str
            The attribute name ("I1", "J1", "I2", or "J2") to read/write.
        """
        if side == "first":
            return "J1" if is_input else "I1"
        return "I2" if is_input else "J2"

    def _propagate_arity_to_parent(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments, too-many-locals, too-many-statements, too-many-positional-arguments]
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
                    prev_sibling_id = sub_ids[pos - 1]
                    # An `in_port` (one of `container_id`'s own inputs)
                    # that vanished needs the sibling's matching `out_port`
                    # to vanish too, or the strict adjacent-size check in
                    # `CompositionDiagram.__post_init__` breaks later.
                    # Only attempted (never required) -- when it fails,
                    # this falls back to today's pre-existing silent drop.
                    dropped_out_ports = sorted(
                        {out_port for in_port, out_port in old_conn.items() if in_port not in input_remap},
                        reverse=True,
                    )
                    removed_out_ports = [
                        out_port
                        for out_port in dropped_out_ports
                        if self._remove_external_port(graph, prev_sibling_id, is_input=False, port=out_port)
                    ]
                    new_conn = {}
                    for in_port, out_port in old_conn.items():
                        if in_port in input_remap:
                            shift = sum(1 for removed in removed_out_ports if removed < out_port)
                            new_conn[input_remap[in_port]] = out_port - shift
                    connectivity[key] = new_conn

                if old_num_outputs != new_num_outputs and pos < len(sub_ids) - 1:
                    key = pos
                    old_conn = connectivity.get(key, {})
                    next_sibling_id = sub_ids[pos + 1]
                    # Symmetric case: an `out_port` (one of `container_id`'s
                    # own outputs) that vanished needs the next sibling's
                    # matching `in_port` to vanish too.
                    dropped_in_ports = sorted(
                        {in_port for in_port, out_port in old_conn.items() if out_port not in output_remap},
                        reverse=True,
                    )
                    removed_in_ports = [
                        in_port
                        for in_port in dropped_in_ports
                        if self._remove_external_port(graph, next_sibling_id, is_input=True, port=in_port)
                    ]
                    new_conn = {}
                    for in_port, out_port in old_conn.items():
                        if out_port in output_remap:
                            shift = sum(1 for removed in removed_in_ports if removed < in_port)
                            new_conn[in_port - shift] = output_remap[out_port]
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

    The rule only applies when the identity's own immediate parent is a
    flat CompositionDiagram it shares with the neighbor it would be
    spliced into. In that case the identity's slot is removed from
    `sub_diagram_ids`/`connectivity` entirely and its neighbors are
    re-linked directly, with no leftover placeholder.

    An identity spider that instead sits in a Tensor lane (or as one of a
    ContractedDiagram's two halves) is NOT touched by this rule: removing
    it there would require shrinking that parent's own port count and
    cascading the change upward through the diagram (including into any
    non-adjacent sibling in an enclosing flat composition), which is not
    a shape-preserving operation and would corrupt connectivity. A stray
    identity spider is valid, harmless diagram content -- unlike a
    `VoidDiagram`, it carries no "must never survive" invariant -- so
    simply leaving it in place is safe.
    """

    def match(self, cvzx_graph: CVZXGraph) -> list[dict]:
        """Find all identity spiders in the graph with a splice-able neighbor.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to search, together with its registry.

        Returns
        -------
        list[dict]
            List of matches, each containing:
            - 'node_id': the node ID of the identity spider
            - 'container_id': its immediate container (grouping key for
              `RewriteRule.apply_rule`)
            - 'survivor_id': the neighbor node ID the identity is
              contracted into
        """
        graph = cvzx_graph.graph
        registry = cvzx_graph.registry
        matches = []

        # Sort for deterministic match order (set iteration order is not stable).
        for node in sorted(registry.identity_spiders):
            attrs = graph.nodes[node]
            if attrs.get("kind") != "proper" or not is_wiring_node_from_attrs(attrs):
                continue

            preds = [
                u
                for u, _, edge_attrs in graph.in_edges(node, data=True)
                if edge_attrs.get("edge_type") == "composition"
            ]
            succs = [
                v
                for _, v, edge_attrs in graph.out_edges(node, data=True)
                if edge_attrs.get("edge_type") == "composition"
            ]
            # A root identity spider (no composition edge on either side
            # -- the whole diagram simplified down to a single wire) has
            # nothing to splice into; there's nothing left for this rule
            # to do.
            if not preds and not succs:
                continue

            container_id = attrs.get("container_id")
            if container_id is None or graph.nodes[container_id].get("container_type") != "composition":
                continue

            # Only a neighbor sharing this identity's own immediate
            # composition parent is a safe splice target -- see class
            # docstring for why a cross-container neighbor is not.
            survivor_id = None
            if succs and graph.nodes[succs[0]].get("container_id") == container_id:
                survivor_id = succs[0]
            elif preds and graph.nodes[preds[0]].get("container_id") == container_id:
                survivor_id = preds[0]
            if survivor_id is None:
                continue

            matches.append({
                "node_id": node,
                "container_id": container_id,
                "survivor_id": survivor_id,
            })

        return matches

    def apply_single(  # ruff: ignore[complex-structure, too-many-branches]
        self, cvzx_graph: CVZXGraph, match: dict
    ) -> None:
        """Remove an identity spider from the graph in-place.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match containing 'node_id' and 'survivor_id'.
        """
        graph = cvzx_graph.graph
        node_id = match["node_id"]

        # Matches are computed once up front; nothing stops a match in
        # one group from being invalidated by a different one applied
        # first in the same batch (e.g. two identities back-to-back,
        # where removing the first changes what's actually adjacent to
        # the second). So `match["survivor_id"]` is NOT trusted here --
        # the identity's current neighbor(s) are re-derived fresh from
        # the graph, exactly as `match()` itself does, which self-heals
        # against exactly that kind of staleness.
        if node_id not in graph:
            return
        attrs = graph.nodes[node_id]
        if not (attrs.get("kind") == "proper" and is_wiring_node_from_attrs(attrs)):
            return

        container_id = attrs.get("container_id")
        if container_id is None or graph.nodes[container_id].get("container_type") != "composition":
            return

        preds = [
            u
            for u, _, edge_attrs in graph.in_edges(node_id, data=True)
            if edge_attrs.get("edge_type") == "composition"
        ]
        succs = [
            v
            for _, v, edge_attrs in graph.out_edges(node_id, data=True)
            if edge_attrs.get("edge_type") == "composition"
        ]
        survivor_id = None
        if succs and graph.nodes[succs[0]].get("container_id") == container_id:
            survivor_id = succs[0]
        elif preds and graph.nodes[preds[0]].get("container_id") == container_id:
            survivor_id = preds[0]
        if survivor_id is None:
            return

        # Same flat CompositionDiagram as its survivor: splice the
        # identity's slot out of `sub_diagram_ids`/`connectivity`
        # entirely and re-link its neighbors directly.
        container_attrs = graph.nodes[container_id]
        sub_ids = container_attrs.get("sub_diagram_ids", [])
        if node_id not in sub_ids or survivor_id not in sub_ids:
            return
        idx = sub_ids.index(node_id)
        connectivity = container_attrs.get("connectivity", {})

        nx.contracted_nodes(graph, survivor_id, node_id, self_loops=False, copy=False)
        if "contraction" in graph.nodes[survivor_id]:
            del graph.nodes[survivor_id]["contraction"]

        new_connectivity = {}
        idx_to_del = idx - 1 if idx > 0 else idx
        if idx_to_del in connectivity:
            del connectivity[idx_to_del]
        for key, targets in connectivity.items():
            if key > idx_to_del:
                new_connectivity[key - 1] = targets
            else:
                new_connectivity[key] = targets
        container_attrs["connectivity"] = new_connectivity
        sub_ids.pop(idx)
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
    - The resulting spider's inputs are the inputs of first + inputs of second
      (minus connected wires), and likewise for outputs
    - Phase is the sum of the two phases

    The rule applies to ContractedDiagram containers anywhere in the graph.
    After fusion, containers with a single element are flattened.
    """

    def match(self, cvzx_graph: CVZXGraph) -> list[dict]:  # ruff: ignore[too-many-locals]
        """Find all ContractedDiagram containers containing fusible spiders.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to search, together with its registry.

        Returns
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
        graph = cvzx_graph.graph
        registry = cvzx_graph.registry
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
            J1 = attrs.get("J1", [])  # first outputs to second inputs  # ruff: ignore[non-lowercase-variable-in-function]
            I1 = attrs.get("I1", [])  # second outputs to first inputs  # ruff: ignore[non-lowercase-variable-in-function]
            J2 = attrs.get("J2", [])  # first inputs to second outputs  # ruff: ignore[non-lowercase-variable-in-function]
            I2 = attrs.get("I2", [])  # second inputs to first outputs  # ruff: ignore[non-lowercase-variable-in-function]

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

    def apply_single(self, cvzx_graph: CVZXGraph, match: dict) -> None:  # ruff: ignore[too-many-locals]
        """Fuse two same-type spiders in a ContractedDiagram in-place.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match containing the ContractedDiagram and spider information.
        """
        graph = cvzx_graph.graph
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

    def match(self, cvzx_graph: CVZXGraph) -> list[dict]:
        """Find all chains of reducible gates in CompositionDiagram containers.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to search, together with its registry.

        Returns
        -------
        list[dict]
            List of matches, each containing:
            - 'container_id': the node ID of the CompositionDiagram
            - 'gate_type': type of gates in the chain
            - 'values': list of values for each gate in the chain
            - 'node_ids': list of node IDs in the chain (in order)
        """
        graph = cvzx_graph.graph
        registry = cvzx_graph.registry
        matches: list[dict] = []

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

    def apply_single(self, cvzx_graph: CVZXGraph, match: dict) -> None:
        """Apply chain reduction to a specific match in-place.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match containing chain information.

        Raises
        ------
        ValueError
            If `match["gate_type"]` is not a type `reduce_chain` recognizes
            (should not occur for a match produced by `match()`).
        """
        graph = cvzx_graph.graph
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
        if reduced_gate is None:
            msg = f"reduce_chain: unrecognized gate_type {gate_type!r}"
            raise ValueError(msg)

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

        Returns
        -------
        list[dict]
            List of chains, each containing:
            - 'start_node': first node in the chain
            - 'gate_type': type of gates
            - 'values': list of values for each gate
            - 'node_ids': list of node IDs in order

        Raises
        ------
        ValueError
            If `get_gate_info` returns no `gate_info` for a 'Q'/'P' node
            (should not occur: those kinds always carry gate_info).
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
                if len(values) >= 2:  # ruff: ignore[magic-value-comparison]
                    if gate_type in {"Q", "P"}:
                        if gate_info is None or next_gate_info is None:
                            msg = f"get_gate_info returned no gate_info for a {gate_type!r} node."
                            raise ValueError(msg)
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

    def get_gate_info(self, graph: nx.DiGraph, node_id: int) -> tuple[str | None, Any, dict | None]:  # ruff: ignore[complex-structure, too-many-return-statements]
        """Extract gate type and value from a node.

        Returns
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

    def can_chain(  # ruff: ignore[too-many-return-statements, too-many-arguments, too-many-positional-arguments]
        self,
        gate_type: str,
        value: Any,  # ruff: ignore[any-type]
        gate_info: dict | None,
        next_type: str | None,
        next_value: Any,  # ruff: ignore[any-type]
        next_gate_info: dict | None,
    ) -> bool:
        """Check if two gates can be chained.

        Returns
        -------
        bool
            True if the gates can be chained (reduced together).

        Raises
        ------
        ValueError
            If `gate_type == "CSUM"` but `gate_info`/`next_gate_info` is
            None (should not occur: CSUM nodes always carry gate_info).
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
            if gate_info is None or next_gate_info is None:
                msg = "get_gate_info returned no gate_info for a 'CSUM' node."
                raise ValueError(msg)
            return bool(
                (gate_info["control"] == next_gate_info["control"])
                and (gate_info["target"] == next_gate_info["target"])
            )

        return False

    def reduce_chain(self, gate_type: str, values: list, gate_info: dict | None = None) -> dict | None:  # ruff: ignore[complex-structure, too-many-return-statements, too-many-branches]
        """Reduce a chain of gates to a single gate.

        Parameters
        ----------
        gate_type : str
            Type of gates in the chain
        values : list
            List of values for each gate in the chain
        gate_info : dict | None
            Useful information about the gate to reduce.

        Returns
        -------
        dict | None
            Dictionary with reduced gate attributes (contains 'type' and
            type-specific fields), or None if `gate_type` is not recognized.

        Raises
        ------
        ValueError
            If `gate_type` is 'Q', 'P', or 'CSUM' but `gate_info` is None
            (should not occur: those kinds always carry gate_info).
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
            if gate_info is None:
                msg = "get_gate_info returned no gate_info for a 'Q' node."
                raise ValueError(msg)
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
            if gate_info is None:
                msg = "get_gate_info returned no gate_info for a 'P' node."
                raise ValueError(msg)
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
            if total == 1.0:  # ruff: ignore[float-equality-comparison]
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
            if remainder == 2:  # ruff: ignore[magic-value-comparison]
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
            if gate_info is None:
                msg = "get_gate_info returned no gate_info for a 'CSUM' node."
                raise ValueError(msg)
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
    _ROTATION_DELTA = {  # ruff: ignore[mutable-class-default]
        "Fourier": -pi / 2,
        "FourierInv": pi / 2,
        "Fourier2": pi,
    }

    def match(self, cvzx_graph: CVZXGraph) -> list[dict]:
        """Find all Fourier-type/rotation or F2/squeezing pairs.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to search, together with its registry.

        Returns
        -------
        list[dict]
            List of matches, each containing:
            - 'container_id': the node ID of the CompositionDiagram
            - 'node_ids': [keep_id, absorb_id] in list order (keep_id survives)
            - 'result_type': 'PhaseRotationGate' or 'SqueezingGate'
            - 'result_value': the combined angle or squeezing parameter
        """
        graph = cvzx_graph.graph
        registry = cvzx_graph.registry
        matches = []

        # Sort for deterministic match order (set iteration order is not stable).
        for container_id in sorted(registry.composition_nodes):
            attrs = graph.nodes[container_id]
            sub_ids = attrs.get("sub_diagram_ids", [])

            if len(sub_ids) < 2:  # ruff: ignore[magic-value-comparison]
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

        Returns
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
        """Build the match dict for a Fourier-type/rotation pair."""  # ruff: ignore[docstring-missing-returns]
        fourier_type = graph.nodes[fourier_id]["type"]
        theta = graph.nodes[rotation_id]["phase"]
        return {
            "node_ids": [first_id, second_id],
            "result_type": "PhaseRotationGate",
            "result_value": theta + self._ROTATION_DELTA[fourier_type],
        }

    def _squeezing_match(self, graph: nx.DiGraph, *, squeezing_id: int, first_id: int, second_id: int) -> dict:
        """Build the match dict for an F2/squeezing pair."""  # ruff: ignore[docstring-missing-returns]
        tau = graph.nodes[squeezing_id]["phase"]
        return {
            "node_ids": [first_id, second_id],
            "result_type": "SqueezingGate",
            "result_value": -tau,
        }

    def apply_single(self, cvzx_graph: CVZXGraph, match: dict) -> None:
        """Apply a Fourier-normalization fold to a specific match in-place.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match containing fold information.
        """
        graph = cvzx_graph.graph
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

    Neither the terminal nor the gate may be directly one of a
    ContractedDiagram's own two halves (its `first_id`/`second_id`) --
    `match()` excludes any such pair outright.

    References
    ----------
    [1] Nagayoshi et al., CV ZX calculus, 2024, Eq. (239a)-(239e).
    """

    def __init__(self, assume_infinite_squeezing: bool = False) -> None:  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        """Create the rule.

        Parameters
        ----------
        assume_infinite_squeezing : bool
            If True, also allow squeezing absorption and cross-color
            discard (both idealized). If False (default), only the
            exact rotation sub-case matches.
        """
        self.assume_infinite_squeezing = assume_infinite_squeezing

    def match(self, cvzx_graph: CVZXGraph) -> list[dict]:
        """Find all gate/terminal pairs that can be absorbed.

        Candidate pairs are found by scanning the graph's own
        "composition" edges directly (mirroring `CopyRule`), rather than
        only checking adjacent `sub_diagram_ids` entries of one
        CompositionDiagram. Every composition edge already connects
        fully-resolved leaf nodes regardless of how deeply either
        endpoint sits inside a TensorDiagram/ContractedDiagram wrapper,
        so this also finds pairs that cross a container boundary -- e.g.
        a gate sitting in one Tensor lane feeding a measurement terminal
        that lives in a completely different Tensor two containers away.
        `_check_pair` itself is unchanged.

        A pair is skipped outright, before `_check_pair` even runs, when
        either endpoint's own immediate parent is a ContractedDiagram --
        see the class docstring for why this rule leaves that case to
        `CopyRule` entirely.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to search, together with its registry.

        Returns
        -------
        list[dict]
            List of matches, each containing:
            - 'container_id': grouping key for `RewriteRule.apply_rule`
              (the pair's shared immediate parent when they have one,
              else a synthetic per-match key)
            - 'node_ids': [keep_id, absorb_id] in list order (keep_id survives)
            - 'result_type': 'QSpider' or 'PSpider'
            - 'result_num_inputs' / 'result_num_outputs': the terminal's own arity
            - 'result_phase': the folded phase
        """
        graph = cvzx_graph.graph
        matches = []
        used_nodes: set[int] = set()

        # Sort for deterministic match order (dict/edge iteration order
        # isn't guaranteed to reflect any particular scan order).
        composition_edges = sorted(
            (u, v) for u, v, edge_attrs in graph.edges(data=True) if edge_attrs.get("edge_type") == "composition"
        )

        for first_id, second_id in composition_edges:
            # Avoid reusing a node a prior match in this same batch
            # already consumed (same reasoning as the old index-stepping
            # loop this replaces).
            if first_id in used_nodes or second_id in used_nodes:
                continue

            # Neither endpoint may be directly one of a ContractedDiagram's
            # own two halves (its `first_id`/`second_id`).
            first_container_id = graph.nodes[first_id].get("container_id")
            second_container_id = graph.nodes[second_id].get("container_id")
            if (
                first_container_id is not None
                and graph.nodes[first_container_id].get("container_type") == "contracted"
            ) or (
                second_container_id is not None
                and graph.nodes[second_container_id].get("container_type") == "contracted"
            ):
                continue

            match = self._check_pair(graph, first_id, second_id)
            if match is None:
                continue

            first_parent = graph.nodes[first_id].get("container_id")
            second_parent = graph.nodes[second_id].get("container_id")
            if first_parent == second_parent and first_parent is not None:
                match["container_id"] = first_parent
                sub_ids = graph.nodes[first_parent].get("sub_diagram_ids", [])
                if first_id in sub_ids and second_id in sub_ids:
                    match["indices"] = sorted([sub_ids.index(first_id), sub_ids.index(second_id)])
            else:
                match["container_id"] = ("cross", first_id, second_id)

            matches.append(match)
            used_nodes.add(first_id)
            used_nodes.add(second_id)

        return matches

    def _check_pair(self, graph: nx.DiGraph, first_id: int, second_id: int) -> dict | None:
        """Check if a pair of adjacent nodes forms an absorbable gate/terminal pattern.

        Returns
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

        Returns
        -------
        dict | None
            Partial match dict (result type/arity/phase), or None.

        Raises
        ------
        TypeError
            If `terminal_attrs["phase"]` is present but not a `ZxPoly`.
        ValueError
            If a phase is required to absorb `gate_attrs` but
            `terminal_attrs` has none (should not occur for a real
            QSpider/PSpider terminal).
        """
        terminal_type = terminal_attrs["type"]
        phase = terminal_attrs.get("phase")
        gate_type = gate_attrs.get("type")

        if phase is not None and not isinstance(phase, ZxPoly):
            msg = f"terminal_attrs['phase'] must be a ZxPoly, got {type(phase)}."
            raise TypeError(msg)

        new_phase = None
        if gate_type == "PhaseRotationGate" and terminal_type == "QSpider":
            if phase is None:
                msg = f"terminal_attrs has no 'phase' to absorb {gate_type!r} into."
                raise ValueError(msg)
            new_phase = self._rotation_absorb(phase, gate_attrs.get("phase"))
        elif gate_type == "Fourier2" and terminal_type == "QSpider":
            # F2 is a fixed rotation by pi -- exact, and pi isn't a
            # degenerate angle (only F/Finv's +-pi/2 would be, so those
            # can never absorb this way: nothing else recognizes them).
            if phase is None:
                msg = f"terminal_attrs has no 'phase' to absorb {gate_type!r} into."
                raise ValueError(msg)
            new_phase = self._rotation_absorb(phase, pi)
        elif gate_type == "SqueezingGate" and self.assume_infinite_squeezing:
            if phase is None:
                msg = f"terminal_attrs has no 'phase' to absorb {gate_type!r} into."
                raise ValueError(msg)
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

        Returns
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

        Returns
        -------
        ZxPoly
            Update phase after applying the Squeezing rule.

        References
        ----------
        [1] Nagayoshi et al., CV ZX calculus, 2024, Eq. (82)-(83).
        """
        return ZxPoly({degree: coeff / tau**degree for degree, coeff in phase.coeffs.items()})

    @staticmethod
    def _is_degenerate_angle(theta: float | Expr) -> bool:
        """True if theta is an odd multiple of pi/2 (tan/1-over-cos undefined).

        Returns
        -------
        bool
        """
        try:
            theta_val = float(theta)
        except (TypeError, ValueError):
            return False  # Symbolic theta: can't determine, assume safe.
        return math.isclose(abs(theta_val) % math.pi, math.pi / 2)

    def apply_single(self, cvzx_graph: CVZXGraph, match: dict) -> None:  # ruff: ignore[complex-structure]
        """Apply a terminal absorption to a specific match in-place.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match containing fold information.
        """
        graph = cvzx_graph.graph
        keep_id, absorb_id = match["node_ids"]
        if keep_id not in graph or absorb_id not in graph:
            return

        # Capture the absorbed gate's own parent before the contraction
        # below removes `absorb_id` from the graph entirely.
        absorb_parent_id = graph.nodes[absorb_id].get("container_id")
        keep_parent_id = graph.nodes[keep_id].get("container_id")

        # Same flat CompositionDiagram: splice the absorbed gate's slot
        # out of `sub_diagram_ids`/`connectivity` entirely, exactly as
        # before this rule could also look past a single container -- the
        # common case, and it keeps producing a fully collapsed result
        # (no leftover placeholder) for it.
        same_parent = (
            absorb_parent_id is not None
            and absorb_parent_id == keep_parent_id
            and graph.nodes[absorb_parent_id].get("container_type") == "composition"
        )
        container_attrs = graph.nodes[absorb_parent_id] if same_parent else None
        sub_ids = container_attrs.get("sub_diagram_ids", []) if container_attrs is not None else []
        same_parent = same_parent and keep_id in sub_ids and absorb_id in sub_ids

        nx.contracted_nodes(graph, keep_id, absorb_id, self_loops=False, copy=False)
        if "contraction" in graph.nodes[keep_id]:
            del graph.nodes[keep_id]["contraction"]

        # `keep_id` doesn't move -- only its own type/phase/arity change;
        # its own immediate parent is untouched.
        graph.nodes[keep_id].update({
            "type": match["result_type"],
            "phase": match["result_phase"],
            "num_inputs": match["result_num_inputs"],
            "num_outputs": match["result_num_outputs"],
        })

        if same_parent:
            assert container_attrs is not None  # ruff: ignore[assert] -- see same_parent's definition above
            i = min(sub_ids.index(keep_id), sub_ids.index(absorb_id))
            new_sub_ids = [sub_id for sub_id in sub_ids if sub_id != absorb_id]
            container_attrs["sub_diagram_ids"] = new_sub_ids

            # Drop the now-internal link at i, move i+1's outgoing link
            # to i, and shift every index past i+1 down by one.
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
                self._flatten_container(graph, absorb_parent_id)
            return

        if absorb_parent_id is not None and absorb_parent_id in graph.nodes:
            self._install_void_placeholder(graph, absorb_parent_id, absorb_id, num_inputs=1, num_outputs=1)


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

    `match()` also transparently sees through any run of identity
    spiders (zero-phase, raw arity (1,1)) sitting directly in the
    composition path between the copy spider and its target hub: an
    identity there is a pure pass-through with no bearing on the copy
    law, and must not block a match that is otherwise exactly this
    pattern. Every identity crossed this way is recorded in the match
    and, in `apply_single`, converted into a same-shape, in-place
    `VoidDiagram` -- a bare type/phase relabeling at its own exact
    existing position, touching no container's shape or bookkeeping, so
    it trivially preserves connectivity.
    """

    def match(self, cvzx_graph: CVZXGraph) -> list[dict]:
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
        cvzx_graph : CVZXGraph
            The graph to search, together with its registry.

        Returns
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
        graph = cvzx_graph.graph
        registry = cvzx_graph.registry
        ordered_matches: list[tuple[int, int, dict]] = []

        # Every real match's copy_spider is a bare state (0,1) or effect
        # (1,0) leaf -- never the "n"-sided hub -- so scanning candidates
        # by this shape, rather than raw composition edges directly,
        # finds exactly the same adjacent pairs the old edge-driven scan
        # did, while ALSO transparently chasing through any run of
        # identity spiders sitting directly in the path (see
        # `_chase_identity_chain`).
        candidates = sorted(
            node
            for node in registry.input_states | registry.measurement_nodes
            if node in graph.nodes and graph.nodes[node].get("type") in {"QSpider", "PSpider"}
        )

        for copy_id in candidates:
            forward = graph.nodes[copy_id].get("num_outputs") == 1
            neighbor_id, identity_chain = self._chase_identity_chain(graph, copy_id, forward=forward)
            if neighbor_id is None:
                continue

            first_id, second_id = (copy_id, neighbor_id) if forward else (neighbor_id, copy_id)
            base_match = self._check_pair(graph, first_id, second_id)
            if base_match is None:
                continue

            resolved = self._resolve_containers(graph, base_match, identity_chain)
            if resolved is None:
                continue

            base_match.update(resolved)
            base_match["identity_chain"] = identity_chain
            ordered_matches.append((first_id, second_id, base_match))

        # Sort for deterministic match order (dict/node iteration order
        # isn't guaranteed to reflect any particular scan order) -- by
        # the same (predecessor, successor) key the old edge-driven scan
        # produced, so match order is unaffected for every pair that
        # doesn't cross an identity chain.
        ordered_matches.sort(key=operator.itemgetter(0, 1))
        return [entry[2] for entry in ordered_matches]

    def _chase_identity_chain(
        self, graph: nx.DiGraph, start_id: int, *, forward: bool
    ) -> tuple[int | None, list[int]]:
        """Follow composition edges from `start_id`, stepping over identities.

        Walks from `start_id` in the given direction, one composition
        edge at a time, continuing past every node that is itself an
        identity spider (zero-phase, raw arity (1,1)) -- a pure
        pass-through with no bearing on the copy law -- until it reaches
        the first node that isn't. That node is the real neighbor to
        check as a copy-law hub; the identities skipped along the way
        are returned too, so `apply_single` can convert each into an
        in-place `VoidDiagram` once the match fires.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to search.
        start_id : int
            The node to start from -- a copy_spider candidate.
        forward : bool
            True to follow outgoing composition edges (`start_id` is a
            state, arity (0,1)); False to follow incoming ones
            (`start_id` is an effect, arity (1,0)).

        Returns
        -------
        tuple[int | None, list[int]]
            `(neighbor_id, identity_ids_crossed)` -- the identity ids in
            traversal order, nearest `start_id` first -- or `(None, [])`
            if there is no composition edge at all in that direction.
        """
        crossed: list[int] = []
        current = start_id
        while True:
            if forward:
                neighbors = [
                    v
                    for _, v, edge_attrs in graph.out_edges(current, data=True)
                    if edge_attrs.get("edge_type") == "composition"
                ]
            else:
                neighbors = [
                    u
                    for u, _, edge_attrs in graph.in_edges(current, data=True)
                    if edge_attrs.get("edge_type") == "composition"
                ]
            if not neighbors:
                return None, crossed
            nxt = neighbors[0]
            nxt_attrs = graph.nodes[nxt]
            if nxt_attrs.get("kind") == "proper" and is_wiring_node_from_attrs(nxt_attrs):
                crossed.append(nxt)
                current = nxt
                continue
            return nxt, crossed

    def _resolve_containers(
        self, graph: nx.DiGraph, base_match: dict, identity_chain: list[int] | None = None
    ) -> dict | None:
        """Work out how to restructure the containers around a candidate match.

        Parameters
        ----------
        identity_chain : list[int] | None
            The identity spiders (if any) crossed by `_chase_identity_chain`
            to find this pair. When non-empty, the "same flat
            CompositionDiagram" splice below is never used -- it assumes
            `copy_id` and `disappear_id` are directly, contiguously
            adjacent entries of one container, which isn't true once
            something (even just one identity) sits between them -- so
            resolution always falls through to the general
            cross-container substitution instead, which only touches
            `copy_id`'s and `disappear_id`'s own individual slots (each
            with a same-shape swap) and never needs to know what sits
            between them.

        Returns
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

        if not identity_chain and copy_parent_id == disappear_parent_id:
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

        Returns
        -------
        dict | None
            Match dictionary if the pair is copy-able, None otherwise.
        """
        first_attrs = graph.nodes[first_id]
        second_attrs = graph.nodes[second_id]

        first_type = first_attrs.get("type")
        second_type = second_attrs.get("type")

        # A node's RAW declared num_inputs/num_outputs is its true
        # external arity UNLESS it's one of a ContractedDiagram's two
        # halves, in which case some of those ports are already wired
        # internally to its sibling and only the "kept" remainder is
        # ever reachable via a composition edge -- see `_external_arity`.
        # `is_wiring_node_from_attrs` below deliberately still reads the
        # RAW attrs dicts (`first_attrs`/`second_attrs`), not this: a
        # node that only *looks* like a trivial (1,1) zero-phase
        # identity from the outside -- because its other port is hidden
        # inside such an internal contraction -- is not a real no-op
        # wire, and must not be excluded as one.
        first_num_inputs, first_num_outputs = _external_arity(graph, first_id)
        second_num_inputs, second_num_outputs = _external_arity(graph, second_id)

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
            first_type == "QSpider"  # ruff: ignore[too-many-boolean-expressions]
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
            first_type == "PSpider"  # ruff: ignore[too-many-boolean-expressions]
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
            first_type == "PSpider"  # ruff: ignore[too-many-boolean-expressions]
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
            first_type == "QSpider"  # ruff: ignore[too-many-boolean-expressions]
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
        """Create a match dictionary if the copied spider's phase is in R₁[X]."""  # ruff: ignore[docstring-missing-returns]
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

        Parameters
        ----------
        phase : ZxPoly
            The phase polynomial to check.

        Returns
        -------
        bool
            True if the polynomial has degree ≤ 1, False otherwise.
        """
        if phase is None:
            return False
        return bool(phase.degree() <= 1)

    def apply_single(self, cvzx_graph: CVZXGraph, match: dict) -> None:  # ruff: ignore[too-many-branches, too-many-locals, too-many-statements, complex-structure]
        """Apply the copy rule to a specific match in-place.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match containing the copy/disappearing spider information plus
            whatever container bookkeeping `match()` resolved for this pair.
        """
        graph = cvzx_graph.graph
        copy_spider_id = match["copy_spider_id"]
        disappearing_spider_id = match["disappearing_spider_id"]
        n_copies = match["n_copies"]
        copy_spider_type = match["copy_spider_type"]
        copy_spider_phase = match["copy_spider_phase"]
        copy_num_inputs = match["copy_spider_num_inputs"]
        copy_num_outputs = match["copy_spider_num_outputs"]
        same_parent = match.get("same_parent", True)

        # Any identity spider crossed by `match()` while finding this
        # pair is a pure pass-through with no bearing on the copy law
        # itself: convert each into a same-shape, in-place VoidDiagram --
        # a bare type/phase relabeling at its own exact existing
        # position, touching no container's shape or bookkeeping at all,
        # so it trivially preserves connectivity.
        for identity_id in match.get("identity_chain", []):
            identity_attrs = graph.nodes[identity_id]
            identity_attrs["type"] = "VoidDiagram"
            identity_attrs["phase"] = None

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
            if len(sub_ids) == 2:  # ruff: ignore[magic-value-comparison]
                self._flatten_container(graph, container_id)
            return

        # Cross-container case
        disappear_container_id = match["disappear_container_id"]
        disappear_container_type = match["disappear_container_type"]
        copy_container_id = match["copy_container_id"]

        # If the disappearing spider is one of a ContractedDiagram's two
        # halves, some of its OWN raw ports may already be wired
        # internally to its sibling (an `edge_type="contracted_internal"`
        # edge, tracked by that ContractedDiagram's own `I1`/`I2`/`J1`/
        # `J2`) -- ports that are neither the one fed by `copy_spider`
        # (padded below) nor among the ones becoming the `n` copies
        # (`_check_pair`'s pattern only ever counts the *external*, kept
        # ports for those two roles -- see `_external_arity`). Capture
        # which raw indices those are now, before `nx.contracted_nodes`
        # below removes `disappearing_spider_id`'s own attrs for good.
        disappear_parent_attrs = (
            graph.nodes[disappear_container_id] if disappear_container_type == "contracted" else None
        )
        disappearing_is_first = (
            disappear_parent_attrs is not None and disappear_parent_attrs.get("first_id") == disappearing_spider_id
        )
        consumed_output_indices: list[int] = []
        consumed_input_indices: list[int] = []
        if disappear_parent_attrs is not None:
            if disappearing_is_first:
                consumed_output_indices = list(disappear_parent_attrs.get("I1", []))
                consumed_input_indices = list(disappear_parent_attrs.get("J1", []))
            else:
                consumed_input_indices = list(disappear_parent_attrs.get("I2", []))
                consumed_output_indices = list(disappear_parent_attrs.get("J2", []))

        # 1. Install the transformed copy spider at the disappearing
        #    spider's old slot, padded with a VoidDiagram sized to
        #    exactly the port the contraction consumed: the single wire
        #    that used to connect copy_spider to disappearing_spider fed
        #    one of disappearing_spider's ports directly, and that
        #    port's count is exactly (copy_num_outputs, copy_num_inputs)
        #    -- an input if copy_spider was a state (0,1) feeding
        #    disappearing_spider's input, an output if copy_spider was an
        #    effect (1,0) fed by disappearing_spider's output.
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

        # 1b. Any port ALSO internally consumed by a ContractedDiagram
        #     contraction (not accounted for by the pad above, nor by the
        #     `n` copies) is still a leg of the disappearing spider that
        #     the copy law applies to -- it needs its OWN fresh copy of
        #     `copy_spider`, exactly like every one of the `n_copies`
        #     instances, not a placeholder standing in for nothing (a
        #     `VoidDiagram` there is silently dropping a real copied
        #     state/effect, and the end-of-pipeline cleanup pass sweeps
        #     `VoidDiagram`s away regardless of whether their slot is
        #     still load-bearing -- corrupting `I1`/`I2`/`J1`/`J2`'s
        #     pairing once it does).
        consumed_output_placeholders = []
        for _ in consumed_output_indices:
            ph_id = max(graph.nodes) + 1
            graph.add_node(
                ph_id,
                id=ph_id,
                type=f"{copy_spider_type}Spider",
                kind="proper",
                phase=copy_spider_phase,
                num_inputs=0,
                num_outputs=1,
                container_id=copy_spider_id,
                external_inputs=[],
                external_outputs=[0],
            )
            graph.nodes[copy_spider_id]["sub_diagram_ids"].append(ph_id)
            consumed_output_placeholders.append(ph_id)
        consumed_input_placeholders = []
        for _ in consumed_input_indices:
            ph_id = max(graph.nodes) + 1
            graph.add_node(
                ph_id,
                id=ph_id,
                type=f"{copy_spider_type}Spider",
                kind="proper",
                phase=copy_spider_phase,
                num_inputs=1,
                num_outputs=0,
                container_id=copy_spider_id,
                external_inputs=[0],
                external_outputs=[],
            )
            graph.nodes[copy_spider_id]["sub_diagram_ids"].append(ph_id)
            consumed_input_placeholders.append(ph_id)

        # Rebuild copy_spider_id's own port bookkeeping fresh from its
        # full child list -- `_recompute_tensor_arity` is generic over
        # the full child list, not just a substitution, so this folds in
        # every pad/placeholder added above.
        self._recompute_tensor_arity(graph, copy_spider_id)

        if disappear_container_type == "contracted":
            assert disappear_parent_attrs is not None  # ruff: ignore[assert] -- see its definition above
            sub_ids = graph.nodes[copy_spider_id]["sub_diagram_ids"]
            output_mapping = graph.nodes[copy_spider_id]["external_output_mapping"]
            input_mapping = graph.nodes[copy_spider_id]["external_input_mapping"]
            reverse_output = {tuple(target): port for port, target in output_mapping.items()}
            reverse_input = {tuple(target): port for port, target in input_mapping.items()}

            new_i1_or_i2 = [reverse_input[sub_ids.index(ph_id), 0] for ph_id in consumed_input_placeholders]
            new_j1_or_j2 = [reverse_output[sub_ids.index(ph_id), 0] for ph_id in consumed_output_placeholders]

            if disappearing_is_first:
                disappear_parent_attrs["J1"] = new_i1_or_i2  # first's inputs consumed by second's outputs
                disappear_parent_attrs["I1"] = new_j1_or_j2  # first's outputs consumed by second's inputs
                disappear_parent_attrs["first_id"] = copy_spider_id
            else:
                disappear_parent_attrs["I2"] = new_i1_or_i2  # second's inputs consumed by first's outputs
                disappear_parent_attrs["J2"] = new_j1_or_j2  # second's outputs consumed by first's inputs
                disappear_parent_attrs["second_id"] = copy_spider_id

            # `kept_first/second_inputs/outputs` and this ContractedDiagram's
            # own external_input_mapping/external_output_mapping were
            # computed from the OLD child's raw port layout and are now
            # stale (the new child's ports are laid out in a different
            # order: copies, then the copy-facing pad, then these
            # placeholders) -- recompute them fresh from the just-updated
            # first_id/second_id and I1/I2/J1/J2, mirroring
            # `_add_contracted_node`'s own bookkeeping, and propagate
            # upward like any other in-place arity change.
            arity_change = self._recompute_contracted_arity(graph, disappear_container_id)
            self._propagate_arity_to_parent(graph, disappear_container_id, *arity_change)
        else:  # "tensor"
            disappear_sub_ids = graph.nodes[disappear_container_id]["sub_diagram_ids"]
            disappear_sub_ids[disappear_sub_ids.index(disappearing_spider_id)] = copy_spider_id
        graph.nodes[copy_spider_id]["container_id"] = disappear_container_id

        # 2. The copy spider's own original slot has nothing left to put
        #    there -- its content now lives as the n copies installed in
        #    step 1. Rather than deleting the slot install a `VoidDiagram`
        #    of the exact same arity in its place.
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


def _external_arity(graph: nx.DiGraph, node_id: int) -> tuple:
    """A leaf's true externally-reachable (input, output) port counts.

    Ordinarily this is just the node's own declared `num_inputs`/
    `num_outputs`. But a node that is one of a `ContractedDiagram`'s two
    halves (`first_id`/`second_id`) may have some of those ports already
    wired to its sibling by an internal `"contracted_internal"` edge --
    those ports are never reachable via a `"composition"` edge from
    outside, so a rule matching against composition-edge neighbors must
    not count them. The `ContractedDiagram` container already computes
    exactly which ports survive externally (`kept_first_inputs`/
    `kept_second_inputs`/`kept_first_outputs`/`kept_second_outputs`,
    built by `_add_contracted_node`); this reads those instead of the
    leaf's own raw declared arity whenever they apply.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to read from.
    node_id : int
        The leaf node to compute the external arity of.

    Returns
    -------
    tuple
        (external_num_inputs, external_num_outputs).
    """
    attrs = graph.nodes[node_id]
    parent_id = attrs.get("container_id")
    parent_attrs = graph.nodes.get(parent_id) if parent_id is not None else None

    if parent_attrs is not None and parent_attrs.get("container_type") == "contracted":
        if parent_attrs.get("first_id") == node_id:
            return len(parent_attrs.get("kept_first_inputs", [])), len(parent_attrs.get("kept_first_outputs", []))
        if parent_attrs.get("second_id") == node_id:
            return len(parent_attrs.get("kept_second_inputs", [])), len(parent_attrs.get("kept_second_outputs", []))

    return attrs.get("num_inputs"), attrs.get("num_outputs")


def is_wiring_node_from_attrs(attrs: dict) -> bool:
    """Check if a node's attributes represent a wiring (identity) diagram.

    A wiring diagram is:
        - A QSpider or PSpider
        - With 1 input and 1 output
        - With zero phase

    Parameters
    ----------
    attrs : dict
        Node attributes from the graph.

    Returns
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

    return bool(phase.is_zero)


def apply_rule_to_diagram(rule: RewriteRule, diagram: Diagram) -> Diagram:
    """Apply a rewrite rule to a diagram, converting to graph and back.

    Parameters
    ----------
    rule : RewriteRule
        The rule to apply.
    diagram : Diagram
        The diagram to modify.

    Returns
    -------
    Diagram
        The modified diagram.
    """
    graph = to_graph(diagram)
    rule.apply_rule(graph)
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

    Parameters
    ----------
    diagram : Diagram
        The diagram to modify.

    Returns
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

    Parameters
    ----------
    expanded_children : list[Diagram]
        The already-`expand_two_mode_gates`-processed children, in order.
    connectivity : dict[int, dict[int, int]]
        The ORIGINAL (pre-expansion) parent composition's connectivity dict.
        Expansion never changes a diagram's external arity, so this still
        correctly describes each boundary `i` -> `i + 1` in terms of the
        un-flattened children's own port numbering.

    Returns
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

"""CV ZX calculus rewrite rules and applications using rustworkx.

This module implements the 10 basic rewrite rules from [1] Sec. IV.A,
and the derived rules from Sec. IV.B; operating on rustworkx PyDiGraph structures.

References
----------
[1] Nagayoshi et al., CV ZX calculus, 2024
"""

import logging
import math
import operator
from abc import ABC, abstractmethod
from typing import Any

import rustworkx as rx
from sympy import Expr, cos, pi, tan

from cvzx.backends.rx.graph import (
    CVZXGraph,
    to_diagram,
    to_graph,
)
from cvzx.exceptions import RuleApplicationError
from cvzx.ir.base import Diagram, ZxPoly
from cvzx.utils.helpers import (
    is_chase_passthrough,
    is_wiring_node_from_attrs,
    passthrough_exit_port,
    simplify_reduced_value,
)

logger = logging.getLogger(__name__)

_APPLY_RULE_MAX_ROUNDS = 100


# =========================================================================
# Base Rewrite Rule
# =========================================================================


class RewriteRule(ABC):
    """Base class for rewrite rules operating on rustworkx CVZXGraph."""

    @abstractmethod
    def match(self, cvzx_graph: CVZXGraph) -> list[dict]:
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
    def apply_single(self, cvzx_graph: CVZXGraph, match: dict) -> None:
        """Apply the rule to a specific match in-place.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            A single match returned by `match()`.
        """

    def apply_rule(self, cvzx_graph: CVZXGraph) -> CVZXGraph:
        """Apply a rule to the graph for up to `_APPLY_RULE_MAX_ROUNDS` rounds.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
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
        for round_index in range(_APPLY_RULE_MAX_ROUNDS):
            matches = self.match(cvzx_graph)

            if not matches:
                logger.debug("%s: no matches, done after %d round(s)", type(self).__name__, round_index)
                return cvzx_graph

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

            logger.debug("%s: round %d, %d match(es), applying", type(self).__name__, round_index, len(matches))
            for key in group_order:
                for match in reversed(groups[key]):
                    self.apply_single(cvzx_graph, match)

            cvzx_graph.rebuild_registry()

        logger.debug(
            "%s.apply_rule: reached _APPLY_RULE_MAX_ROUNDS=%d rounds this call -- "
            "any remaining matches will be picked up by optimize()'s next pass",
            type(self).__name__,
            _APPLY_RULE_MAX_ROUNDS,
        )
        return cvzx_graph

    def _flatten_container(self, graph: rx.PyDiGraph, id_map: dict[int, int], container_id: int) -> None:
        """Flatten a container if it has only one element.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to modify.
        container_id : int
            The container node ID.
        """
        if container_id not in id_map:
            return

        attrs = graph[id_map[container_id]]
        sub_ids = attrs.get("sub_diagram_ids", [])

        if len(sub_ids) == 1:
            child_id = sub_ids[0]
            parent_container_id = attrs.get("container_id")

            if parent_container_id is not None and parent_container_id in id_map:
                parent_attrs = graph[id_map[parent_container_id]]
                if parent_attrs.get("container_type") in {"composition", "tensor"}:
                    parent_sub_ids = parent_attrs.get("sub_diagram_ids", [])
                    idx = parent_sub_ids.index(container_id)
                    parent_sub_ids[idx] = child_id
                elif parent_attrs.get("first_id") == container_id:
                    parent_attrs["first_id"] = child_id
                else:
                    parent_attrs["second_id"] = child_id

                if child_id in id_map:
                    graph[id_map[child_id]]["container_id"] = parent_container_id
                _remove_node(graph, id_map, container_id)

            elif child_id in id_map:
                graph[id_map[child_id]]["container_id"] = None
                graph[id_map[child_id]]["is_root"] = True

    def _replace_in_parent(
        self, graph: rx.PyDiGraph, id_map: dict[int, int], parent_id: int, old_id: int, new_id: int
    ) -> None:
        """Substitute `new_id` for `old_id` in `parent_id`'s own bookkeeping.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to modify.
        parent_id : int
            The immediate parent container of `old_id`.
        old_id : int
            The node ID currently occupying the slot.
        new_id : int
            The node ID that should occupy it instead.
        """
        if parent_id not in id_map:
            return
        parent_attrs = graph[id_map[parent_id]]
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

        if new_id in id_map:
            graph[id_map[new_id]]["container_id"] = parent_id

    def _install_void_placeholder(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        graph: rx.PyDiGraph,
        id_map: dict[int, int],
        parent_id: int,
        old_id: int,
        num_inputs: int,
        num_outputs: int,
    ) -> int:
        """Replace `old_id`'s slot in `parent_id` with a same-shaped `VoidDiagram`.

        Parameters
        ----------
        graph : rx.PyDiGraph
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
        void_id = max(id_map.keys()) + 1 if id_map else 0
        attrs = {
            "id": void_id,
            "type": "VoidDiagram",
            "kind": "proper",
            "phase": None,
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
            "container_id": parent_id,
            "external_inputs": list(range(num_inputs)),
            "external_outputs": list(range(num_outputs)),
        }
        idx = graph.add_node(attrs)
        id_map[void_id] = idx
        self._replace_in_parent(graph, id_map, parent_id, old_id, void_id)
        return void_id

    def _refresh_composition_external_mappings(
        self, graph: rx.PyDiGraph, id_map: dict[int, int], container_id: int
    ) -> None:
        """Recompute a `CompositionDiagram` container's own external port mappings.

        Call this right after any such splice, whenever more than one
        sub-diagram remains (a single-element container is flattened
        away entirely instead, via `_flatten_container`, making its own
        mapping moot).

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to modify.
        container_id : int
            The composition container whose `sub_diagram_ids` was just
            spliced.
        """
        if container_id not in id_map:
            return
        attrs = graph[id_map[container_id]]
        if attrs.get("container_type") != "composition":
            return
        sub_ids = attrs.get("sub_diagram_ids", [])
        if not sub_ids:
            return
        first_attrs = graph[id_map[sub_ids[0]]]
        last_attrs = graph[id_map[sub_ids[-1]]]
        attrs["external_input_mapping"] = {j: (0, j) for j in range(first_attrs.get("num_inputs", 0))}
        last_idx = len(sub_ids) - 1
        attrs["external_output_mapping"] = {j: (last_idx, j) for j in range(last_attrs.get("num_outputs", 0))}

    @staticmethod
    def _composition_step(
        graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int, port: int, *, forward: bool
    ) -> tuple[int | None, int | None]:
        """Follow the single composition edge carrying one specific port onward.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to search.
        node_id : int
            The node to step from.
        port : int
            The port on `node_id` to follow -- an OUTPUT port if
            `forward` (we are walking downstream), an INPUT port
            otherwise (we are walking upstream).
        forward : bool
            True to follow `node_id`'s out-edges, False its in-edges.

        Returns
        -------
        tuple[int | None, int | None]
            `(neighbor_id, neighbor_port)` -- `neighbor_port` is an INPUT
            port of `neighbor_id` if `forward`, an OUTPUT port
            otherwise -- or `(None, None)` if no composition edge on
            `node_id` carries that port.
        """
        if forward:
            for _u, v, edge_attrs in _get_out_edges(graph, id_map, node_id):
                if edge_attrs.get("edge_type") != "composition":
                    continue
                source_ports = edge_attrs.get("source_ports", [])
                if port in source_ports:
                    return v, edge_attrs["target_ports"][source_ports.index(port)]
        else:
            for u, _v, edge_attrs in _get_in_edges(graph, id_map, node_id):
                if edge_attrs.get("edge_type") != "composition":
                    continue
                target_ports = edge_attrs.get("target_ports", [])
                if port in target_ports:
                    return u, edge_attrs["source_ports"][target_ports.index(port)]
        return None, None

    def _rebuild_composition_connectivity(
        self, graph: rx.PyDiGraph, id_map: dict[int, int], container_id: int
    ) -> None:
        """Rebuild a CompositionDiagram container's `connectivity` from its children.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to modify.
        container_id : int
            The CompositionDiagram container node ID.
        """
        if container_id not in id_map:
            return
        attrs = graph[id_map[container_id]]
        if attrs.get("container_type") != "composition":
            return

        sub_ids = attrs.get("sub_diagram_ids", [])

        new_conn: dict[int, dict[int, int]] = {}
        for i in range(len(sub_ids) - 1):
            left = graph[id_map[sub_ids[i]]]
            right = graph[id_map[sub_ids[i + 1]]]
            n = min(left.get("num_outputs", 0), right.get("num_inputs", 0))
            new_conn[i] = {j: j for j in range(n)}
        attrs["connectivity"] = new_conn

        if sub_ids:
            attrs["num_inputs"] = graph[id_map[sub_ids[0]]].get("num_inputs", 0)
            attrs["num_outputs"] = graph[id_map[sub_ids[-1]]].get("num_outputs", 0)
        else:
            attrs["num_inputs"] = 0
            attrs["num_outputs"] = 0

        new_num_inputs = attrs["num_inputs"]
        new_num_outputs = attrs["num_outputs"]
        attrs["external_inputs"] = list(range(new_num_inputs))
        attrs["external_outputs"] = list(range(new_num_outputs))

    def _chase_identity_chain(
        self, graph: rx.PyDiGraph, id_map: dict[int, int], start_id: int, *, forward: bool, start_port: int = 0
    ) -> tuple[int | None, int | None, list[int]]:
        """Follow one wire from `start_id`, stepping over identity/`Swap` passthroughs.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to search.
        start_id : int
            The node to start from.
        forward : bool
            True to follow outgoing composition edges (`start_port` is
            one of `start_id`'s OUTPUT ports); False to follow incoming
            ones (`start_port` is one of `start_id`'s INPUT ports).
        start_port : int
            The port of `start_id` to chase from. Defaults to 0, the
            only possible value for a single-wire state/effect endpoint
            (`CopyRule`/`TerminalAbsorptionRule`'s use case);
            `ChainReductionRule` chases every port of a wider gate
            separately and passes each one explicitly.

        Returns
        -------
        tuple[int | None, int | None, list[int]]
            `(neighbor_id, neighbor_port, passthrough_ids_crossed)` --
            `neighbor_port` is an INPUT port of `neighbor_id` if
            `forward`, an OUTPUT port otherwise; the crossed ids are in
            traversal order, nearest `start_id` first -- or
            `(None, None, crossed_so_far)` if the chase runs off the end
            of the graph (no composition edge at all in that direction)
            before reaching a non-passthrough node.
        """
        crossed: list[int] = []
        current, port = start_id, start_port
        while True:
            next_id, next_port = self._composition_step(graph, id_map, current, port, forward=forward)
            if next_id is None or next_port is None or next_id not in id_map:
                return None, None, crossed
            next_attrs = graph[id_map[next_id]]
            next_container_id = next_attrs.get("container_id")
            next_container = graph[id_map[next_container_id]] if next_container_id in id_map else {}
            if (
                next_attrs.get("kind") == "proper"
                and next_container.get("type") != "ContractedDiagram"
                and is_chase_passthrough(next_attrs)
            ):
                crossed.append(next_id)
                current = next_id
                port = passthrough_exit_port(next_attrs, next_port)
                continue
            return next_id, next_port, crossed

    @staticmethod
    def _remap_by_identity(old_mapping: dict, new_mapping: dict) -> dict:
        """Match old external ports to new ones by identical target.

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

    def _recompute_contracted_arity(self, graph: rx.PyDiGraph, id_map: dict[int, int], container_id: int) -> tuple:  # ruff: ignore[too-many-locals]
        """Recompute a ContractedDiagram container's ports after a side's arity changed in place.

        `first_id`/`second_id` are assumed already updated to point at the
        (possibly different-arity) current children; `I1`/`I2`/`J1`/`J2`
        are unchanged. Mirrors `_add_contracted_node`'s own bookkeeping.

        Parameters
        ----------
        graph : rx.PyDiGraph
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
        attrs = graph[id_map[container_id]]
        first_id = attrs["first_id"]
        second_id = attrs["second_id"]
        j1 = attrs.get("J1", [])
        i2 = attrs.get("I2", [])
        i1 = attrs.get("I1", [])
        j2 = attrs.get("J2", [])

        first_num_inputs = graph[id_map[first_id]].get("num_inputs", 0)
        first_num_outputs = graph[id_map[first_id]].get("num_outputs", 0)
        second_num_inputs = graph[id_map[second_id]].get("num_inputs", 0)
        second_num_outputs = graph[id_map[second_id]].get("num_outputs", 0)

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

    def _recompute_tensor_arity(self, graph: rx.PyDiGraph, id_map: dict[int, int], container_id: int) -> tuple:
        """Recompute a TensorDiagram container's ports after a child was substituted in place.

        The child at each index is assumed unchanged in position (only its
        own arity may have changed) -- no sibling was added or removed, so
        old ports are safely matched to new ports by (index, internal_port)
        identity even though numeric port offsets may shift.

        Parameters
        ----------
        graph : rx.PyDiGraph
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
        attrs = graph[id_map[container_id]]
        sub_ids = attrs.get("sub_diagram_ids", [])

        old_num_inputs = attrs.get("num_inputs", 0)
        old_num_outputs = attrs.get("num_outputs", 0)
        old_input_mapping = attrs.get("external_input_mapping", {})
        old_output_mapping = attrs.get("external_output_mapping", {})

        input_mapping = {}
        offset = 0
        for idx, sub_id in enumerate(sub_ids):
            for p in range(graph[id_map[sub_id]].get("num_inputs", 0)):
                input_mapping[offset] = (idx, p)
                offset += 1
        new_num_inputs = offset

        output_mapping = {}
        offset = 0
        for idx, sub_id in enumerate(sub_ids):
            for p in range(graph[id_map[sub_id]].get("num_outputs", 0)):
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
        graph: rx.PyDiGraph,
        id_map: dict[int, int],
        container_id: int,
        changed_child_id: int,
        child_old_num_inputs: int,
        child_input_remap: dict,
        child_old_num_outputs: int,
        child_output_remap: dict,
    ) -> tuple:
        """Recompute a TensorDiagram's ports using a child's own KNOWN remap.

        Parameters
        ----------
        graph : rx.PyDiGraph
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
        attrs = graph[id_map[container_id]]
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
                    new_offset += graph[id_map[sid]].get(count_attr, 0)
                else:
                    count = graph[id_map[sid]].get(count_attr, 0)
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
            for p in range(graph[id_map[sid]].get("num_inputs", 0)):
                input_mapping[offset] = (idx, p)
                offset += 1
        output_mapping = {}
        offset = 0
        for idx, sid in enumerate(sub_ids):
            for p in range(graph[id_map[sid]].get("num_outputs", 0)):
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
    def _remove_tensor_child(  # ruff: ignore[complex-structure, too-many-arguments, too-many-positional-arguments]
        graph: rx.PyDiGraph,
        id_map: dict[int, int],
        container_id: int,
        removed_id: int,
        removed_num_inputs: int,
        removed_num_outputs: int,
    ) -> tuple:
        """Remove one child from a TensorDiagram container in place.

        Parameters
        ----------
        graph : rx.PyDiGraph
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
        attrs = graph[id_map[container_id]]
        sub_ids = attrs["sub_diagram_ids"]
        pos = sub_ids.index(removed_id)

        old_num_inputs = attrs.get("num_inputs", 0)
        old_num_outputs = attrs.get("num_outputs", 0)

        input_offset_before = sum(graph[id_map[sid]].get("num_inputs", 0) for sid in sub_ids[:pos])
        input_remap = {}
        for old_port in range(old_num_inputs):
            if old_port < input_offset_before:
                input_remap[old_port] = old_port
            elif old_port >= input_offset_before + removed_num_inputs:
                input_remap[old_port] = old_port - removed_num_inputs

        output_offset_before = sum(graph[id_map[sid]].get("num_outputs", 0) for sid in sub_ids[:pos])
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
            for p in range(graph[id_map[sid]].get("num_inputs", 0)):
                input_mapping[offset] = (idx, p)
                offset += 1
        output_mapping = {}
        offset = 0
        for idx, sid in enumerate(sub_ids):
            for p in range(graph[id_map[sid]].get("num_outputs", 0)):
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

    def _remove_composition_child(
        self, graph: rx.PyDiGraph, id_map: dict[int, int], container_id: int, removed_id: int
    ) -> tuple:
        """Remove one child from a flat CompositionDiagram container in place.

        Parameters
        ----------
        graph : rx.PyDiGraph
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
        attrs = graph[id_map[container_id]]
        sub_ids = attrs.get("sub_diagram_ids", [])
        pos = sub_ids.index(removed_id)
        n = len(sub_ids)

        old_connectivity = attrs.get("connectivity", {})
        new_connectivity = {}
        for key, targets in old_connectivity.items():
            if key in {pos - 1, pos}:
                continue
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
            new_num_inputs = graph[id_map[sub_ids[0]]].get("num_inputs", 0) if sub_ids else 0
            if new_num_inputs == old_num_inputs:
                input_remap = {p: p for p in range(old_num_inputs)}

        if pos == n - 1:
            new_num_outputs = graph[id_map[sub_ids[-1]]].get("num_outputs", 0) if sub_ids else 0
            if new_num_outputs == old_num_outputs:
                output_remap = {p: p for p in range(old_num_outputs)}

        attrs.update({
            "num_inputs": new_num_inputs,
            "num_outputs": new_num_outputs,
            "external_inputs": list(range(new_num_inputs)),
            "external_outputs": list(range(new_num_outputs)),
        })

        return old_num_inputs, new_num_inputs, old_num_outputs, new_num_outputs, input_remap, output_remap

    def _remove_external_port(  # ruff: ignore[complex-structure, too-many-return-statements, too-many-locals, too-many-branches]
        self,
        graph: rx.PyDiGraph,
        id_map: dict[int, int],
        node_id: int,
        is_input: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
        port: int,
    ) -> bool:
        """Try to eliminate one external port of `node_id`, recursing through wrappers.

        Parameters
        ----------
        graph : rx.PyDiGraph
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
        if node_id not in id_map:
            return False
        attrs = graph[id_map[node_id]]
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
            if not self._remove_external_port(graph, id_map, child_id, is_input, child_port):
                return False
            self._recompute_tensor_arity(graph, id_map, node_id)
            return True

        if container_type == "contracted":
            mapping_key = "external_input_mapping" if is_input else "external_output_mapping"
            target = attrs.get(mapping_key, {}).get(port)
            if target is None:
                return False
            side, local_port = target
            child_id = attrs["first_id"] if side == "first" else attrs["second_id"]
            if not self._remove_external_port(graph, id_map, child_id, is_input, local_port):
                return False
            index_key = self._contracted_index_key(side, is_input)
            shifted = [p if p < local_port else p - 1 for p in attrs.get(index_key, []) if p != local_port]
            attrs[index_key] = shifted
            self._recompute_contracted_arity(graph, id_map, node_id)
            return True

        if container_type == "composition":
            sub_ids = attrs.get("sub_diagram_ids", [])
            if not sub_ids:
                return False
            target_id = sub_ids[0] if is_input else sub_ids[-1]
            if not self._remove_external_port(graph, id_map, target_id, is_input, port):
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
        graph: rx.PyDiGraph,
        id_map: dict[int, int],
        container_id: int,
        old_num_inputs: int,
        new_num_inputs: int,
        old_num_outputs: int,
        new_num_outputs: int,
        input_remap: dict,
        output_remap: dict,
    ) -> None:
        """Propagate a child's arity change up through every ancestor that needs it.

        Parameters
        ----------
        graph : rx.PyDiGraph
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
            if container_id not in id_map:
                return
            parent_id = graph[id_map[container_id]].get("container_id")
            if parent_id is None or parent_id not in id_map:
                return
            parent_attrs = graph[id_map[parent_id]]
            parent_type = parent_attrs.get("container_type")

            if parent_type == "tensor":
                arity_change = self._recompute_tensor_arity_from_child_remap(
                    graph,
                    id_map,
                    parent_id,
                    container_id,
                    old_num_inputs,
                    input_remap,
                    old_num_outputs,
                    output_remap,
                )
            elif parent_type == "contracted":
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
                arity_change = self._recompute_contracted_arity(graph, id_map, parent_id)
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
                    dropped_out_ports = sorted(
                        {out_port for in_port, out_port in old_conn.items() if in_port not in input_remap},
                        reverse=True,
                    )
                    removed_out_ports = [
                        out_port
                        for out_port in dropped_out_ports
                        if self._remove_external_port(graph, id_map, prev_sibling_id, is_input=False, port=out_port)
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
                    dropped_in_ports = sorted(
                        {in_port for in_port, out_port in old_conn.items() if out_port not in output_remap},
                        reverse=True,
                    )
                    removed_in_ports = [
                        in_port
                        for in_port in dropped_in_ports
                        if self._remove_external_port(graph, id_map, next_sibling_id, is_input=True, port=in_port)
                    ]
                    new_conn = {}
                    for in_port, out_port in old_conn.items():
                        if out_port in output_remap:
                            shift = sum(1 for removed in removed_in_ports if removed < in_port)
                            new_conn[in_port - shift] = output_remap[out_port]
                    connectivity[key] = new_conn

                parent_attrs["connectivity"] = connectivity

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

    def _structural_step(  # ruff: ignore[complex-structure, too-many-branches, too-many-return-statements, too-many-statements, too-many-locals]
        self, graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int, port: int, *, forward: bool
    ) -> tuple[int | None, int | None]:
        """Advance one true wire-hop from `(node_id, port)`, using diagram STRUCTURE only.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to read (never mutated).
        node_id : int
            The node to step from.
        port : int
            The port on `node_id` to follow.
        forward : bool
            Direction to walk.

        Returns
        -------
        tuple[int | None, int | None]
            `(next_node_id, next_port)` -- `next_port` is an INPUT port
            of `next_node_id` if `forward`, an OUTPUT port otherwise --
            or `(None, None)` if the wire runs off the whole diagram (a
            genuine external boundary) or through unsupported structure
            before reaching a leaf.
        """
        while True:  # ruff: ignore[too-many-nested-blocks]
            if node_id not in id_map:
                return None, None
            parent_id = graph[id_map[node_id]].get("container_id")
            if parent_id is None or parent_id not in id_map:
                return None, None
            parent_attrs = graph[id_map[parent_id]]
            parent_type = parent_attrs.get("container_type")
            sub_ids = parent_attrs.get("sub_diagram_ids", [])

            if parent_type == "composition" and node_id in sub_ids:
                pos = sub_ids.index(node_id)
                connectivity = parent_attrs.get("connectivity", {})
                if forward:
                    if pos < len(sub_ids) - 1:
                        landed = None
                        for in_port, out_port in connectivity.get(pos, {}).items():
                            if out_port == port:
                                landed = (sub_ids[pos + 1], in_port)
                                break
                        if landed is None:
                            return None, None
                        node_id, port = landed
                        break
                    node_id = parent_id
                    continue
                if pos > 0:
                    conn = connectivity.get(pos - 1, {})
                    if port not in conn:
                        return None, None
                    node_id, port = sub_ids[pos - 1], conn[port]
                    break
                node_id = parent_id
                continue

            if parent_type == "tensor" and node_id in sub_ids:
                child_idx = sub_ids.index(node_id)
                mapping_key = "external_output_mapping" if forward else "external_input_mapping"
                mapping = parent_attrs.get(mapping_key, {})
                external_port = next((ext for ext, tgt in mapping.items() if tgt == (child_idx, port)), None)
                if external_port is None:
                    return None, None
                node_id, port = parent_id, external_port
                continue

            return None, None

        while True:
            if node_id not in id_map:
                return None, None
            attrs = graph[id_map[node_id]]
            container_type = attrs.get("container_type")
            if container_type is None:
                return node_id, port
            sub_ids = attrs.get("sub_diagram_ids", [])
            if container_type == "composition":
                if not sub_ids:
                    return None, None
                node_id = sub_ids[0 if forward else len(sub_ids) - 1]
                continue
            if container_type == "tensor":
                mapping_key = "external_input_mapping" if forward else "external_output_mapping"
                target = attrs.get(mapping_key, {}).get(port)
                if target is None:
                    return None, None
                child_idx, child_port = target
                node_id, port = sub_ids[child_idx], child_port
                continue
            return None, None

    def _trace_closure_leftovers(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        graph: rx.PyDiGraph,
        id_map: dict[int, int],
        first_id: int,
        last_id: int,
        port: int,
        known_absorbed: set[int],
    ) -> list[tuple[int, int]] | None:
        """Walk one vanishing output port of `first_id` to `last_id`, purely structurally.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to read (never mutated).
        first_id : int
            The chain's own first member (the wire's origin).
        last_id : int
            The chain's own last member (the wire's true destination).
        port : int
            Which of `first_id`'s own output ports to trace.
        known_absorbed : set[int]
            Node IDs belonging to the current match itself -- safe to
            pass through unconditionally, and never inspected directly.

        Returns
        -------
        list[tuple[int, int]] | None
            `(node_id, port)` for every leftover (non-`known_absorbed`)
            `VoidDiagram` strictly between `first_id` and `last_id`, in
            structural order nearest-first, `port` being the specific
            port index this wire passes through it at -- or `None` if
            the wire doesn't reach `last_id` through nothing but those
            two kinds of node.
        """
        leftovers: list[tuple[int, int]] = []
        bundles_crossed = 0
        current_id, current_port = first_id, port
        for _ in range(10_000):
            next_id, next_port = self._structural_step(graph, id_map, current_id, current_port, forward=True)
            if next_id is None or next_port is None or next_id not in id_map:
                return None
            if next_id == last_id:
                return leftovers
            if next_id in known_absorbed:
                current_id, current_port = next_id, next_port
                continue
            next_attrs = graph[id_map[next_id]]
            if next_attrs.get("type") != "VoidDiagram":
                return None
            next_in = next_attrs.get("num_inputs", 0)
            if next_in != next_attrs.get("num_outputs", 0):
                return None
            if next_in > 1:
                if bundles_crossed >= 1:
                    return None
                bundles_crossed += 1
            leftovers.append((next_id, next_port))
            current_id, current_port = next_id, next_port
        return None


# =========================================================================
# Rule Implementations
# =========================================================================


class IdentityRule(RewriteRule):
    r"""Identity rule (id) from [1] Eq. (69) - Graph-based version.

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
        id_map = _get_id_map(graph)
        matches = []

        for node in sorted(registry.identity_spiders):
            if node not in id_map:
                continue
            attrs = graph[id_map[node]]
            if attrs.get("kind") != "proper" or not is_wiring_node_from_attrs(attrs):
                continue

            preds = [
                u
                for u, _, edge_attrs in _get_in_edges(graph, id_map, node)
                if edge_attrs.get("edge_type") == "composition"
            ]
            succs = [
                v
                for _, v, edge_attrs in _get_out_edges(graph, id_map, node)
                if edge_attrs.get("edge_type") == "composition"
            ]

            if not preds and not succs:
                continue

            container_id = attrs.get("container_id")
            if (
                container_id is None
                or container_id not in id_map
                or graph[id_map[container_id]].get("container_type") != "composition"
            ):
                continue

            survivor_id = None
            if succs and graph[id_map[succs[0]]].get("container_id") == container_id:
                survivor_id = succs[0]
            elif preds and graph[id_map[preds[0]]].get("container_id") == container_id:
                survivor_id = preds[0]
            if survivor_id is None:
                continue

            matches.append({
                "node_id": node,
                "container_id": container_id,
                "survivor_id": survivor_id,
            })

        return matches

    def apply_single(  # ruff: ignore[complex-structure]
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
        id_map = _get_id_map(graph)
        node_id = match["node_id"]

        if node_id not in id_map:
            return
        attrs = graph[id_map[node_id]]
        if not (attrs.get("kind") == "proper" and is_wiring_node_from_attrs(attrs)):
            return

        container_id = attrs.get("container_id")
        if (
            container_id is None
            or container_id not in id_map
            or graph[id_map[container_id]].get("container_type") != "composition"
        ):
            return

        preds = [
            u
            for u, _, edge_attrs in _get_in_edges(graph, id_map, node_id)
            if edge_attrs.get("edge_type") == "composition"
        ]
        succs = [
            v
            for _, v, edge_attrs in _get_out_edges(graph, id_map, node_id)
            if edge_attrs.get("edge_type") == "composition"
        ]

        survivor_id = None
        if succs and graph[id_map[succs[0]]].get("container_id") == container_id:
            survivor_id = succs[0]
        elif preds and graph[id_map[preds[0]]].get("container_id") == container_id:
            survivor_id = preds[0]
        if survivor_id is None:
            return

        container_attrs = graph[id_map[container_id]]
        sub_ids = container_attrs.get("sub_diagram_ids", [])
        if node_id not in sub_ids or survivor_id not in sub_ids:
            return
        idx = sub_ids.index(node_id)
        connectivity = container_attrs.get("connectivity", {})

        _contract_nodes(graph, id_map, survivor_id, node_id)

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
            self._flatten_container(graph, id_map, container_id)


class FusionRule(RewriteRule):
    r"""Fusion rule (f) from [1] Eq. (70) & (71) - Graph-based version.

    Two same-color spiders connected by a wire fuse into a single spider
    with the summed phase and the union of their unconnected ports. Two
    shapes are recognized:

    - **Contracted**: the two halves of a `ContractedDiagram`, coupled
        through its `I1`/`I2`/`J1`/`J2` wiring. Handled by
        `match_contracted` / `_apply_contracted`.
    - **Composition**: two same-color spiders joined by a wire through a
        composition, possibly with identities or `Swap` nodes in between.
        Handled by `match_terminal` / `_apply_terminal`.

    Match-level guards:

    - A bare `(1, 1)` zero-phase identity spider is never fused into a
        real spider by `match_terminal`. It is a wire, and
        `IdentityRule` owns its removal.
    - A direct half of a `ContractedDiagram` is never touched by
        `match_terminal`. Restructuring a contract half requires
        rewriting the contract's own I/J coupling and is a distinct
        operation, not a same-color two-spider fusion. Contract halves
        whose partner is a fusible same-color spider are handled by
        `match_contracted`; contract halves whose partner is not (e.g. a
        Q/P CSUM-style contract) are left alone for now.
    - Composition matches whose endpoints are already claimed by a
        contracted match are filtered out, so the more specific contracted
        path wins when both apply.

    `first_id` is always the survivor in `apply_single`.
    """

    def match(self, cvzx_graph: CVZXGraph) -> list[dict]:
        """Find all fusible same-color spider pairs.

        Returns
        -------
        list[dict]
            Union of `match_contracted` and `match_terminal`, with
            composition matches whose endpoints are already claimed by
            a contracted match filtered out.
        """
        contracted = self.match_contracted(cvzx_graph)

        contracted_nodes: set[int] = set()
        for m in contracted:
            contracted_nodes.add(m["first_id"])
            contracted_nodes.add(m["second_id"])

        composition = [
            m
            for m in self.match_terminal(cvzx_graph)
            if m["first_id"] not in contracted_nodes and m["second_id"] not in contracted_nodes
        ]

        return [*contracted, *composition]

    def match_contracted(self, cvzx_graph: CVZXGraph) -> list[dict]:  # ruff: ignore[too-many-locals]
        """Find ContractedDiagram containers whose halves are fusible.

        A contract is fusible when both halves are same-color
        `QSpider` nodes or same-color `PSpider` nodes, at least one I/J coupling
        exists between them, and either the two are plain spiders or the
        contract has the tensor-shaped form `CopyRule` leaves behind
        (the `special_case`).

        Returns
        -------
        list[dict]
            Matches for `_apply_contracted`.
        """
        graph = cvzx_graph.graph
        registry = cvzx_graph.registry
        id_map = _get_id_map(graph)
        matches = []

        for node in sorted(registry.contracted_diagrams):
            if node not in id_map:
                continue
            attrs = graph[id_map[node]]
            first_id = attrs.get("first_id")
            second_id = attrs.get("second_id")

            if first_id is None or second_id is None or first_id not in id_map or second_id not in id_map:
                continue

            first_attrs = graph[id_map[first_id]]
            second_attrs = graph[id_map[second_id]]

            first_type = first_attrs.get("type")
            second_type = second_attrs.get("type")

            J1 = attrs.get("J1", [])  # ruff: ignore[non-lowercase-variable-in-function]
            I1 = attrs.get("I1", [])  # ruff: ignore[non-lowercase-variable-in-function]
            J2 = attrs.get("J2", [])  # ruff: ignore[non-lowercase-variable-in-function]
            I2 = attrs.get("I2", [])  # ruff: ignore[non-lowercase-variable-in-function]

            special_case = (
                (first_type in {"QSpider", "PSpider"} and second_type == "TensorDiagram")
                or (second_type in {"QSpider", "PSpider"} and first_type == "TensorDiagram")
            ) and (len(I1) == 1 and len(J1) == 0)

            if (
                not (first_type in {"QSpider", "PSpider"} and second_type in {"QSpider", "PSpider"})
                and not special_case
            ):
                continue

            if first_type != second_type and not special_case:
                continue

            has_connection = (len(J1) > 0 and len(J2) > 0) or (len(I1) > 0 and len(I2) > 0)
            if not has_connection:
                continue

            kept_first_inputs = attrs.get("kept_first_inputs", [])
            kept_second_inputs = attrs.get("kept_second_inputs", [])
            kept_first_outputs = attrs.get("kept_first_outputs", [])
            kept_second_outputs = attrs.get("kept_second_outputs", [])

            matches.append({
                "shape": "contracted",
                "contracted_id": node,
                "first_id": first_id,
                "second_id": second_id,
                "first_type": first_type,
                "second_type": second_type,
                "kept_first_inputs": kept_first_inputs,
                "kept_second_inputs": kept_second_inputs,
                "kept_first_outputs": kept_first_outputs,
                "kept_second_outputs": kept_second_outputs,
                "J1": J1,
                "I1": I1,
                "J2": J2,
                "I2": I2,
                "special_case": special_case,
            })

        return matches

    def match_terminal(self, cvzx_graph: CVZXGraph) -> list[dict]:
        """Find a pair of same-color spiders, one in a contraction and the other a terminal node.

        Any identity chain between the terminal node and the same color
        spider in the contraction will be crossed.

        Returns
        -------
        list[dict]
            Matches for `_apply_terminal`.
        """
        graph = cvzx_graph.graph
        registry = cvzx_graph.registry
        id_map = _get_id_map(graph)
        matches = []
        used: set[int] = set()

        candidates = sorted(
            node
            for node in registry.input_states | registry.measurement_nodes
            if node in id_map and graph[id_map[node]]["type"] != "VoidDiagram"
        )
        for terminal_id in candidates:
            if terminal_id in used:
                continue
            forward = terminal_id in registry.input_states
            neighbor_id, neighbor_port, chain = self._chase_identity_chain(graph, id_map, terminal_id, forward=forward)
            if neighbor_id is None or neighbor_id in used:
                continue

            spider_attrs = graph[id_map[terminal_id]]
            neighbor_attrs = graph[id_map[neighbor_id]]

            if neighbor_attrs.get("type") != spider_attrs.get("type"):
                continue

            if self._is_identity_leaf(neighbor_attrs):
                continue

            matches.append({
                "shape": "composition",
                "first_id": terminal_id,
                "second_id": neighbor_id,
                "spider_type": spider_attrs.get("type"),
                "identity_chain": chain,
                "consumed_second_input": neighbor_port,
            })
            used.add(terminal_id)
            used.add(neighbor_id)
            used.update(chain)

        return matches

    @staticmethod
    def _is_directly_in_contracted(graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int) -> bool:
        """Whether `node_id` is directly one of a ContractedDiagram's own halves.

        Returns
        -------
        bool
        """
        if node_id not in id_map:
            return False
        parent_id = graph[id_map[node_id]].get("container_id")
        return (
            parent_id is not None
            and parent_id in id_map
            and graph[id_map[parent_id]].get("container_type") == "contracted"
        )

    @staticmethod
    def _is_identity_leaf(attrs: dict) -> bool:
        """True if a node is a bare (1, 1) zero-phase spider.

        Returns
        -------
        bool
        """
        if attrs.get("type") not in {"QSpider", "PSpider"}:
            return False
        if attrs.get("num_inputs") != 1 or attrs.get("num_outputs") != 1:
            return False
        phase = attrs.get("phase")
        return phase is not None and bool(phase.is_zero)

    def apply_single(self, cvzx_graph: CVZXGraph, match: dict) -> None:
        """Apply the fusion rule to a specific match in-place.

        Dispatches on the match's `"shape"` key. Defaults to
        `"contracted"` when absent, so hand-built matches predating the
        split keep working.

        Raises
        ------
        RuleApplicationError
            If `"shape"` is neither "contracted" nor "composition".
        """
        shape = match.get("shape", "contracted")
        if shape == "contracted":
            self._apply_contracted(cvzx_graph, match)
        elif shape == "composition":
            self._apply_terminal(cvzx_graph, match)
        else:
            msg = f"FusionRule.apply_single: unknown match shape {shape!r}"
            raise RuleApplicationError(msg)

    def _apply_contracted(self, cvzx_graph: CVZXGraph, match: dict) -> None:  # ruff: ignore[too-many-locals, complex-structure, too-many-branches, too-many-statements]
        """Fuse two same-color spiders in a ContractedDiagram in-place.

        `first_id` survives; the fused spider is wrapped in a fresh
        `TensorDiagram` (or, in the special case, kept inside the
        contract's existing tensor) to preserve canonicity. The contract
        container is dropped from its parent's slot and the fused result
        takes its place.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match from `match_contracted`.
        """
        graph = cvzx_graph.graph
        id_map = _get_id_map(graph)
        contracted_id = match["contracted_id"]
        first_id = match["first_id"]
        second_id = match["second_id"]
        first_type = match["first_type"]
        second_type = match["second_type"]
        special_case = match["special_case"]

        if contracted_id not in id_map or first_id not in id_map or second_id not in id_map:
            return

        contracted_attrs = graph[id_map[contracted_id]]
        first_attrs = graph[id_map[first_id]]
        second_attrs = graph[id_map[second_id]]

        new_num_inputs = len(match["kept_first_inputs"]) + len(match["kept_second_inputs"])
        new_num_outputs = len(match["kept_first_outputs"]) + len(match["kept_second_outputs"])
        if special_case:
            new_num_inputs -= 1
            new_num_outputs -= 1
        fused_type = "QSpider" if first_type == "QSpider" else "PSpider"

        parent_container_id = contracted_attrs.get("container_id")
        is_root = contracted_attrs.get("is_root", False)

        if special_case:
            if second_type != "TensorDiagram":
                first_id, second_id = second_id, first_id
                # `first_attrs` was captured against the pre-swap
                # `first_id` (the `TensorDiagram`, whose own `phase` is
                # always None) -- refresh it against the real spider now
                # bound to `first_id`, or `first_phase` below is always
                # None and this match silently no-ops forever.
                first_attrs = graph[id_map[first_id]]

            node_to_contract_id = graph[id_map[second_id]]["sub_diagram_ids"][-1]
            first_phase = first_attrs.get("phase")
            second_phase = graph[id_map[node_to_contract_id]].get("phase")

            if first_phase is None or second_phase is None:
                return

            new_phase = first_phase + second_phase
            _contract_nodes(graph, id_map, first_id, node_to_contract_id)

            graph[id_map[first_id]].update({
                "type": fused_type,
                "num_inputs": new_num_inputs,
                "num_outputs": new_num_outputs,
                "phase": new_phase,
                "container_id": parent_container_id,
                "is_root": is_root,
            })

            graph[id_map[second_id]]["sub_diagram_ids"].pop(-1)
            graph[id_map[second_id]]["sub_diagram_ids"] = [first_id] + graph[id_map[second_id]]["sub_diagram_ids"]
            num_output = graph[id_map[second_id]]["num_outputs"]
            graph[id_map[second_id]]["external_output_mapping"][num_output] = (0, 0)
            graph[id_map[second_id]]["external_outputs"].append(num_output)
            graph[id_map[second_id]].update({
                "num_inputs": 1,
                "num_outputs": num_output + 1,
                "external_inputs": [0],
                "external_input_mapping": {0: (0, 0)},
            })

            fusion_id = second_id
        else:
            _contract_nodes(graph, id_map, first_id, second_id)

            new_id = max(id_map.keys()) + 1 if id_map else 0
            attrs = {
                "id": new_id,
                "type": "TensorDiagram",
                "kind": "container",
                "container_type": "tensor",
                "phase": None,
                "num_inputs": new_num_inputs,
                "num_outputs": new_num_outputs,
                "container_id": None,
                "is_root": is_root,
                "sub_diagram_ids": [first_id],
                "external_inputs": list(range(new_num_inputs)),
                "external_outputs": list(range(new_num_outputs)),
                "external_input_mapping": {0: (0, 0)},
                "external_output_mapping": {0: (0, 0)},
            }
            idx = graph.add_node(attrs)
            id_map[new_id] = idx

            first_phase = first_attrs.get("phase")
            second_phase = second_attrs.get("phase")

            if first_phase is None or second_phase is None:
                return

            new_phase = first_phase + second_phase
            graph[id_map[first_id]].update({
                "type": fused_type,
                "num_inputs": new_num_inputs,
                "num_outputs": new_num_outputs,
                "phase": new_phase,
                "container_id": new_id,
                "is_root": is_root,
            })
            fusion_id = new_id

        if parent_container_id is not None and parent_container_id in id_map:
            parent_attrs = graph[id_map[parent_container_id]]
            parent_type = parent_attrs.get("container_type")

            if parent_type in {"composition", "tensor"}:
                sub_ids = parent_attrs.get("sub_diagram_ids", [])
                if contracted_id in sub_ids:
                    idx_val = sub_ids.index(contracted_id)
                    sub_ids[idx_val] = fusion_id
                    graph[id_map[parent_container_id]]["sub_diagram_ids"] = sub_ids
                    graph[id_map[fusion_id]]["container_id"] = parent_container_id

            elif parent_type == "contracted":
                if parent_attrs.get("first_id") == contracted_id:
                    parent_attrs["first_id"] = fusion_id
                elif parent_attrs.get("second_id") == contracted_id:
                    parent_attrs["second_id"] = fusion_id
                graph[id_map[fusion_id]]["container_id"] = parent_container_id

        else:
            graph[id_map[fusion_id]]["is_root"] = True
            graph[id_map[fusion_id]]["container_id"] = None

        if contracted_id in id_map:
            _remove_node(graph, id_map, contracted_id)

    def _apply_terminal(self, cvzx_graph: CVZXGraph, match: dict) -> None:
        """Fuse same-color spiders, one in a contraction and the other a terminal node.

        Any identity chain between the terminal node and the same color
        spider in the contraction will be crossed.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match from `match_terminal`.
        """
        graph = cvzx_graph.graph
        id_map = _get_id_map(graph)
        first_id = match["first_id"]
        second_id = match["second_id"]
        identity_chain = match.get("identity_chain", [])

        if first_id not in id_map or second_id not in id_map:
            return

        first_attrs = graph[id_map[first_id]]
        second_attrs = graph[id_map[second_id]]

        for pid in identity_chain:
            if pid in id_map:
                pid_attrs = graph[id_map[pid]]
                pid_attrs["type"] = "VoidDiagram"
                pid_attrs["phase"] = None

        first_phase = first_attrs.get("phase") or ZxPoly({})
        second_phase = second_attrs.get("phase") or ZxPoly({})
        fused_phase = first_phase + second_phase

        spider_type = first_attrs.get("type")

        second_attrs.update({
            "type": spider_type,
            "phase": fused_phase,
        })
        first_attrs.update({
            "type": "VoidDiagram",
            "phase": None,
        })


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

    def match(  # ruff: ignore[too-many-locals, too-many-branches, too-many-statements, complex-structure]
        self, cvzx_graph: CVZXGraph
    ) -> list[dict]:
        """Find all chains of reducible gates in the graph.

        Candidates are found by scanning the graph's own "composition"
        edges (mirroring `CopyRule`/`TerminalAbsorptionRule`), rather than
        only checking adjacent `sub_diagram_ids` entries of one
        CompositionDiagram -- so this also finds chains that cross a
        TensorDiagram/ContractedDiagram boundary. `_find_full_neighbor`
        additionally chases through any run of identity spiders or
        `Swap` nodes sitting directly in the path, so a passthrough there
        never blocks an otherwise-reducible chain either.

        Each maximal chain is discovered exactly once, starting from its
        true first member (`_is_chain_start`), then extended forward one
        `can_chain`-compatible neighbor at a time.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to search, together with its registry.

        Returns
        -------
        list[dict]
            List of matches, each containing:

            - 'container_id': grouping key for `RewriteRule.apply_rule`
              (the chain's shared immediate parent when it's a
              contiguous run in one flat CompositionDiagram, else a
              synthetic per-match key)
            - 'same_parent': whether the chain is such a contiguous run
            - 'gate_type': type of gates in the chain
            - 'values': list of values for each gate in the chain
            - 'node_ids': list of node IDs in the chain (in order)
            - 'gate_info': gate_info for `reduce_chain`
            - 'identity_chain': passthrough node ids crossed while
              chasing between chain members, in traversal order

        Raises
        ------
        ValueError
            If `get_gate_info` returns no `gate_info` for a 'Q'/'P' node
            (should not occur: those kinds always carry gate_info).
        """
        graph = cvzx_graph.graph
        id_map = _get_id_map(graph)
        matches: list[dict] = []
        used_nodes: set[int] = set()

        candidates = sorted(
            node
            for node in id_map
            if graph[id_map[node]].get("kind") in {"proper", "compact"}
            and self.get_gate_info(graph, id_map, node)[0] is not None
            and not self._is_directly_in_contracted(graph, id_map, node)
        )

        for start_id in candidates:
            if start_id in used_nodes:
                continue

            gate_type, value, gate_info = self.get_gate_info(graph, id_map, start_id)
            if gate_type is None or not self._is_chain_start(graph, id_map, start_id, gate_type, value, gate_info):
                continue

            values = [value]
            node_ids = [start_id]
            identity_chain: list[int] = []
            identity_chain_checkpoints = [0]
            gate_info_history = [gate_info]
            last_gate_info = gate_info
            cur_id, cur_type, cur_value, cur_gate_info = start_id, gate_type, value, gate_info

            while True:
                neighbor_id, chain = self._find_full_neighbor(graph, id_map, cur_id, forward=True)
                if (
                    neighbor_id is None
                    or neighbor_id in used_nodes
                    or neighbor_id in node_ids
                    or any(cid in used_nodes for cid in chain)
                ):
                    break
                next_type, next_value, next_gate_info = self.get_gate_info(graph, id_map, neighbor_id)
                if not self.can_chain(cur_type, cur_value, cur_gate_info, next_type, next_value, next_gate_info):
                    break
                assert next_type is not None  # ruff: ignore[assert]

                for cid in chain:
                    if cid not in identity_chain:
                        identity_chain.append(cid)
                values.append(next_value)
                node_ids.append(neighbor_id)
                identity_chain_checkpoints.append(len(identity_chain))
                gate_info_history.append(next_gate_info)
                last_gate_info = next_gate_info

                if {cur_value, next_value} == {"F", "Finv"}:
                    gate_type = "F_pair"
                    break

                cur_id, cur_type, cur_value, cur_gate_info = neighbor_id, next_type, next_value, next_gate_info

            if len(values) < 2:  # ruff: ignore[magic-value-comparison]
                continue

            if gate_type in {"Q", "P"}:
                if gate_info is None or last_gate_info is None:
                    msg = f"get_gate_info returned no gate_info for a {gate_type!r} node."
                    raise ValueError(msg)
                gate_info["num_outputs"] = last_gate_info["num_outputs"]

                while len(node_ids) >= 2:  # ruff: ignore[magic-value-comparison]
                    first_id, candidate_last_id = node_ids[0], node_ids[-1]
                    first_out = graph[id_map[first_id]].get("num_outputs", 0)
                    last_out = gate_info_history[-1].get("num_outputs", 0) if gate_info_history[-1] else 0
                    if last_out != 0 or first_out == last_out:
                        break
                    known_absorbed = set(node_ids)
                    if all(
                        self._trace_closure_leftovers(graph, id_map, first_id, candidate_last_id, port, known_absorbed)
                        is not None
                        for port in range(first_out)
                    ):
                        break
                    node_ids = node_ids[:-1]
                    values = values[:-1]
                    gate_info_history = gate_info_history[:-1]
                if len(node_ids) < 2:  # ruff: ignore[magic-value-comparison]
                    continue
                identity_chain = identity_chain[: identity_chain_checkpoints[len(node_ids) - 1]]
                last_gate_info = gate_info_history[-1]
                gate_info["num_outputs"] = last_gate_info["num_outputs"] if last_gate_info else 0

            first_id, last_id = node_ids[0], node_ids[-1]

            match: dict = {
                "gate_type": gate_type,
                "values": values,
                "node_ids": node_ids,
                "gate_info": gate_info,
                "identity_chain": identity_chain,
            }

            same_parent = False
            resolved_parent: int | None = None
            if not identity_chain:
                candidate_parents = []
                seen_candidates = set()
                for nid in node_ids:
                    candidate = graph[id_map[nid]].get("container_id")
                    if candidate is not None and candidate not in seen_candidates:
                        seen_candidates.add(candidate)
                        candidate_parents.append(candidate)
                for candidate in candidate_parents:
                    if candidate not in id_map:
                        continue
                    if graph[id_map[candidate]].get("container_type") != "composition":
                        continue
                    sub_ids = graph[id_map[candidate]].get("sub_diagram_ids", [])
                    if not all(nid in sub_ids for nid in node_ids):
                        continue
                    indices = [sub_ids.index(nid) for nid in node_ids]
                    if indices == list(range(indices[0], indices[0] + len(node_ids))):
                        same_parent = True
                        resolved_parent = candidate
                        break

            match["same_parent"] = same_parent
            match["container_id"] = resolved_parent if same_parent else ("cross", first_id, last_id)

            matches.append(match)
            used_nodes.update(node_ids)
            used_nodes.update(identity_chain)

        matches.sort(key=lambda m: m["node_ids"][0])
        return matches

    def _is_chain_start(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        graph: rx.PyDiGraph,
        id_map: dict[int, int],
        node_id: int,
        gate_type: str,
        value: Any,  # ruff: ignore[any-type]
        gate_info: dict | None,
    ) -> bool:
        """Check that nothing chains INTO `node_id` -- i.e. it's a chain's true first member.

        Prevents re-discovering the same maximal chain more than once
        from an internal member: `match()` only ever starts building a
        chain from a node whose own predecessor either doesn't exist or
        doesn't `can_chain` into it.

        Returns
        -------
        bool
            True if `node_id` has no `can_chain`-compatible predecessor.
        """
        prev_id, _ = self._find_full_neighbor(graph, id_map, node_id, forward=False)
        if prev_id is None:
            return True
        prev_type, prev_value, prev_gate_info = self.get_gate_info(graph, id_map, prev_id)
        if prev_type is None:
            return True
        return not self.can_chain(prev_type, prev_value, prev_gate_info, gate_type, value, gate_info)

    def _find_full_neighbor(  # ruff: ignore[complex-structure, too-many-return-statements]
        self, graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int, *, forward: bool
    ) -> tuple[int | None, list[int]]:
        """Find `node_id`'s single full composition neighbor, chasing every port.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to search.
        node_id : int
            The node to look for a full neighbor from.
        forward : bool
            True to look downstream (chase every OUTPUT port), False to
            look upstream (chase every INPUT port).

        Returns
        -------
        tuple[int | None, list[int]]
            `(neighbor_id, passthrough_ids_crossed)`, deduplicated in
            first-seen order, or `(None, [])` if there's no such single,
            fully-saturating neighbor.
        """
        if node_id not in id_map:
            return None, []
        attrs = graph[id_map[node_id]]
        num_ports = attrs.get("num_outputs" if forward else "num_inputs", 0)
        if not num_ports:
            return None, []

        neighbor_id: int | None = None
        seen_neighbor_ports: set[int] = set()
        combined_chain: list[int] = []
        seen_chain: set[int] = set()

        for port in range(num_ports):
            nxt, nxt_port, chain = self._chase_identity_chain(graph, id_map, node_id, forward=forward, start_port=port)
            if nxt is None or nxt_port is None:
                return None, []
            if neighbor_id is None:
                neighbor_id = nxt
            elif nxt != neighbor_id:
                return None, []
            if nxt_port in seen_neighbor_ports:
                return None, []
            seen_neighbor_ports.add(nxt_port)
            for cid in chain:
                if cid not in seen_chain:
                    seen_chain.add(cid)
                    combined_chain.append(cid)

        if neighbor_id is None or neighbor_id not in id_map:
            return None, []
        if self._is_directly_in_contracted(graph, id_map, neighbor_id):
            return None, []
        neighbor_attrs = graph[id_map[neighbor_id]]
        neighbor_arity = neighbor_attrs.get("num_inputs" if forward else "num_outputs", 0)
        if neighbor_arity != num_ports:
            return None, []
        return neighbor_id, combined_chain

    @staticmethod
    def _is_directly_in_contracted(graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int) -> bool:
        """Check whether `node_id` is directly a ContractedDiagram's own half.

        Returns
        -------
        bool
            True if `node_id`'s own immediate parent is a `ContractedDiagram`
            (i.e. `node_id` is literally that container's `first_id` or
            `second_id`), False otherwise.
        """
        if node_id not in id_map:
            return False
        parent_id = graph[id_map[node_id]].get("container_id")
        return (
            parent_id is not None
            and parent_id in id_map
            and graph[id_map[parent_id]].get("container_type") == "contracted"
        )

    def _overwrite_node_for_reduced_gate(
        self, graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int, reduced_gate: dict
    ) -> None:
        """Overwrite `node_id`'s attrs with the reduced gate's, in place.

        Same shape as `_update_node_for_reduced_gate` but without the
        `container_id` bookkeeping: this method is only called when the
        reduced arity matches the node's own, so nothing about the node's
        position or port counts changes.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to modify.
        node_id : int
            The node to overwrite.
        reduced_gate : dict
            The dict returned by `reduce_chain`.
        """
        self._update_node_for_reduced_gate(graph, id_map, node_id, reduced_gate)

    @staticmethod
    def _reset_to_identity(graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int) -> None:
        """Reset a node in place to a zero-phase identity spider.

        The node keeps its own slot, container, and port counts. Its type
        becomes the spider color that identity should have:

        - A raw `QSpider`/`PSpider` keeps its own color.
        - Every other gate type becomes `QSpider`.

        Any `feedforward`/`measurement_ids`/`param_measurement_map`
        attributes are cleared.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to modify.
        node_id : int
            The node to reset.
        """
        if node_id not in id_map:
            return
        attrs = graph[id_map[node_id]]
        gate_type = attrs.get("type")

        identity_type = gate_type if gate_type in {"QSpider", "PSpider"} else "QSpider"

        attrs.update({
            "type": identity_type,
            "kind": "proper",
            "phase": ZxPoly({}),
            "feedforward": None,
            "measurement_ids": None,
            "param_measurement_map": {},
        })

    def _apply_single_splice(self, cvzx_graph: CVZXGraph, match: dict, reduced_gate: dict) -> None:  # ruff: ignore[too-many-locals, too-many-statements, too-many-branches, complex-structure]
        """Splice-and-propagate path for arity-changing chain reductions.

        Used only for `QSpider`/`PSpider` chains whose reduced arity differs
        from the first member's own arity. This is the previous
        `apply_single` body, renamed; it contracts the chain members into
        the survivor and updates the surrounding container's bookkeeping.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match containing chain information.
        reduced_gate : dict
            The dict returned by `reduce_chain`.
        """
        graph = cvzx_graph.graph
        id_map = _get_id_map(graph)
        node_ids = match["node_ids"]

        for passthrough_id in match.get("identity_chain", []):
            if passthrough_id in id_map:
                passthrough_attrs = graph[id_map[passthrough_id]]
                passthrough_attrs["type"] = "VoidDiagram"
                passthrough_attrs["phase"] = None

        first_node = node_ids[0]

        if match.get("same_parent", True):
            container_id = match["container_id"]
            container_attrs = graph[id_map[container_id]]
            sub_ids = container_attrs.get("sub_diagram_ids", [])

            for i in range(1, len(node_ids), 1):
                _contract_nodes(graph, id_map, first_node, node_ids[i])

            self._update_node_for_reduced_gate(graph, id_map, first_node, reduced_gate)

            new_sub_ids = []
            removed_connectivity = []
            for i, sub_id in enumerate(sub_ids):
                if sub_id in node_ids:
                    if sub_id == first_node:
                        new_sub_ids.append(first_node)
                    else:
                        removed_connectivity.append(i - 1)
                else:
                    new_sub_ids.append(sub_id)

            graph[id_map[container_id]]["sub_diagram_ids"] = new_sub_ids
            graph[id_map[container_id]]["sub_diagram_indices"] = list(range(len(new_sub_ids)))
            self._update_connectivity_after_reduction(graph, id_map, container_id, removed_connectivity)

            if len(new_sub_ids) == 1:
                self._flatten_container(graph, id_map, container_id)
            return

        last_node = node_ids[-1]
        absorbed_info = [
            (
                nid,
                graph[id_map[nid]].get("container_id"),
                graph[id_map[nid]].get("num_inputs", 0),
                graph[id_map[nid]].get("num_outputs", 0),
            )
            for nid in node_ids[1:]
            if nid in id_map
        ]
        first_parent_id = graph[id_map[first_node]].get("container_id")
        first_old_shape = (
            graph[id_map[first_node]].get("num_inputs", 0),
            graph[id_map[first_node]].get("num_outputs", 0),
        )

        is_closure = (
            graph[id_map[first_node]].get("num_outputs", 0) != 0
            and graph[id_map[last_node]].get("num_outputs", 0) == 0
        )
        closure_leftovers: list[tuple[int, int]] = []
        if is_closure:
            known_absorbed = set(node_ids)
            seen_leftovers: set[tuple[int, int]] = set()
            for port in range(graph[id_map[first_node]].get("num_outputs", 0)):
                found = self._trace_closure_leftovers(graph, id_map, first_node, last_node, port, known_absorbed)
                for entry in found or []:
                    if entry not in seen_leftovers:
                        seen_leftovers.add(entry)
                        closure_leftovers.append(entry)

        leftover_ports_by_id: dict[int, list[int]] = {}
        leftover_order: list[int] = []
        for nid, original_port in closure_leftovers:
            if nid not in leftover_ports_by_id:
                leftover_ports_by_id[nid] = []
                leftover_order.append(nid)
            leftover_ports_by_id[nid].append(original_port)

        for leftover_id in leftover_order:
            if leftover_id not in id_map:
                continue
            for port in sorted(set(leftover_ports_by_id[leftover_id]), reverse=True):
                leftover_attrs = graph[id_map[leftover_id]]
                old_in = leftover_attrs.get("num_inputs", 0)
                old_out = leftover_attrs.get("num_outputs", 0)
                if port >= old_out:
                    continue
                if not self._remove_external_port(graph, id_map, leftover_id, is_input=False, port=port):
                    continue
                new_out = old_out - 1
                output_remap = {i: (i if i < port else i - 1) for i in range(old_out) if i != port}
                self._propagate_arity_to_parent(
                    graph, id_map, leftover_id, old_in, old_in, old_out, new_out, {}, output_remap
                )

        first_stale_out = [
            target
            for _, target, edge_attrs in _get_out_edges(graph, id_map, first_node)
            if edge_attrs.get("edge_type") == "composition"
        ]
        far_edges = [
            (target, dict(edge_attrs))
            for _, target, edge_attrs in _get_out_edges(graph, id_map, last_node)
            if edge_attrs.get("edge_type") == "composition"
        ]

        for nid in node_ids[1:]:
            _remove_node(graph, id_map, nid)

        for target in first_stale_out:
            if _has_edge(graph, id_map, first_node, target):
                _remove_edge(graph, id_map, first_node, target)
        for far_target, edge_attrs in far_edges:
            if far_target in id_map:
                u_idx = id_map[first_node]
                v_idx = id_map[far_target]
                graph.add_edge(u_idx, v_idx, edge_attrs)

        self._update_node_for_reduced_gate(graph, id_map, first_node, reduced_gate)

        for nid, parent_id, n_in, n_out in absorbed_info:
            if parent_id is None or parent_id not in id_map:
                continue
            if is_closure:
                void_id = self._install_void_placeholder(graph, id_map, parent_id, nid, num_inputs=0, num_outputs=0)
                if n_in != 0 or n_out != 0:
                    self._propagate_arity_to_parent(graph, id_map, void_id, n_in, 0, n_out, 0, {}, {})
            else:
                self._install_void_placeholder(graph, id_map, parent_id, nid, num_inputs=n_in, num_outputs=n_out)

        new_shape = (graph[id_map[first_node]].get("num_inputs", 0), graph[id_map[first_node]].get("num_outputs", 0))
        if new_shape != first_old_shape and first_parent_id is not None and first_parent_id in id_map:
            first_parent_type = graph[id_map[first_parent_id]].get("container_type")
            if first_parent_type == "tensor":
                arity_change = self._recompute_tensor_arity(graph, id_map, first_parent_id)
                self._propagate_arity_to_parent(graph, id_map, first_parent_id, *arity_change)
            else:
                old_in, old_out = first_old_shape
                new_in, new_out = new_shape
                in_remap = {p: p for p in range(old_in)} if new_in == old_in else {}
                out_remap = {p: p for p in range(old_out)} if new_out == old_out else {}
                self._propagate_arity_to_parent(
                    graph, id_map, first_node, old_in, new_in, old_out, new_out, in_remap, out_remap
                )

    def apply_single(self, cvzx_graph: CVZXGraph, match: dict) -> None:
        """Reduce a chain of same-type gates in place.

        The chain's first member keeps its own node ID and slot; its type and
        phase are overwritten with the reduced result. When the chain is a
        run of bare `(1, 1)` `QSpider`/`PSpider` nodes, every other member is
        cheaply overwritten in place with a zero-phase `(1, 1)` identity
        spider instead -- a bare zero-phase spider genuinely IS the identity
        only at that one arity (see `IdentityRule`'s own docstring), so this
        shortcut relies on `IdentityRule` to prune it on a later pass, and
        nothing here is contracted, spliced, or voided.

        Every other case -- multi-wire gates (`ControlledSumGate`,
        `ControlledZGate`, `BeamsplitterGate`, ...) whose "extra" chain
        members have no such prunable identity representation, and any
        `QSpider`/`PSpider` chain whose reduced arity differs from its
        first member's own -- falls back to the general splice-and-
        propagate path, which actually removes the extra nodes from the
        graph and rewires the container around them (flattening it if only
        the survivor remains), so no leftover node is ever left standing in
        for "nothing happens here".

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match containing chain information.

        Raises
        ------
        RuleApplicationError
            If `match["gate_type"]` is not a type `reduce_chain` recognizes
            (should not occur for a match produced by `match()`).
        """
        graph = cvzx_graph.graph
        id_map = _get_id_map(graph)
        gate_type = match["gate_type"]
        values = match["values"]
        node_ids = match["node_ids"]
        gate_info = match["gate_info"]

        reduced_gate = self.reduce_chain(gate_type, values, gate_info)
        if reduced_gate is None:
            msg = f"reduce_chain: unrecognized gate_type {gate_type!r}"
            raise RuleApplicationError(msg)

        first_id = node_ids[0]
        first_attrs = graph[id_map[first_id]]
        first_num_inputs = first_attrs.get("num_inputs", 0)
        first_num_outputs = first_attrs.get("num_outputs", 0)
        reduced_num_inputs = reduced_gate.get("num_inputs", first_num_inputs)
        reduced_num_outputs = reduced_gate.get("num_outputs", first_num_outputs)

        # Arity-changing chains, and any chain whose members aren't bare
        # (1, 1) spiders (so "reset the extras to a same-arity zero-phase
        # spider" wouldn't actually be a prunable identity), fall back to
        # the general splice-and-propagate path.
        stays_1_1 = first_num_inputs == 1 and first_num_outputs == 1
        if reduced_num_inputs != first_num_inputs or reduced_num_outputs != first_num_outputs or not stays_1_1:
            self._apply_single_splice(cvzx_graph, match, reduced_gate)
            return

        # Common case: overwrite the survivor, replace the rest with
        # zero-phase (1, 1) identity spiders -- a genuine identity at that
        # one arity, safe to leave for IdentityRule to prune later.
        self._overwrite_node_for_reduced_gate(graph, id_map, first_id, reduced_gate)
        for extra_id in node_ids[1:]:
            self._reset_to_identity(graph, id_map, extra_id)

    def get_gate_info(  # ruff: ignore[complex-structure, too-many-return-statements]
        self, graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int
    ) -> tuple[str | None, Any, dict | None]:
        """Extract gate type and value from a node.

        Returns
        -------
        tuple[str | None, Any, dict | None]
            (gate_type, value, gate_info)
            gate_type: 'Q', 'P', 'R', 'BS', 'Sq', 'D', 'F', 'F2', 'ControlledZGate', 'ControlledSumGate', or None
            value: the parameter value (phase polynomial, angle, etc.)
            gate_info: additional info (arities, control/target, etc.)
        """
        if node_id not in id_map:
            return None, None, None
        attrs = graph[id_map[node_id]]
        node_type = attrs.get("type")

        if node_type == "QSpider":
            phase = attrs.get("phase")
            num_inputs = attrs.get("num_inputs")
            num_outputs = attrs.get("num_outputs")
            return ("Q", phase, {"num_inputs": num_inputs, "num_outputs": num_outputs})

        if node_type == "PSpider":
            phase = attrs.get("phase")
            num_inputs = attrs.get("num_inputs")
            num_outputs = attrs.get("num_outputs")
            return ("P", phase, {"num_inputs": num_inputs, "num_outputs": num_outputs})

        if node_type == "PhaseRotationGate":
            theta = attrs.get("phase")
            return ("R", theta, None)

        if node_type == "BeamsplitterGate":
            theta = attrs.get("phase")
            return ("BS", theta, None)

        if node_type == "SqueezingGate":
            tau = attrs.get("phase")
            return ("Sq", tau, None)

        if node_type == "DisplacementGate":
            alpha = attrs.get("phase")
            return ("D", alpha, None)

        if node_type == "Fourier":
            return ("F", "F", None)
        if node_type == "FourierInv":
            return ("F", "Finv", None)
        if node_type == "Fourier2":
            return ("F2", "F2", None)

        if node_type == "ControlledZGate":
            gain = attrs.get("phase")
            return ("CZ", gain, None)

        if node_type == "ControlledSumGate":
            gain = attrs.get("phase")
            control = attrs.get("control")
            target = attrs.get("target", 1)
            return ("CSUM", gain, {"control": control, "target": target})

        return (None, None, None)

    def can_chain(  # ruff: ignore[complex-structure, too-many-return-statements, too-many-arguments, too-many-positional-arguments]
        self,
        gate_type: str,
        value: Any,  # ruff: ignore[any-type]
        gate_info: dict | None,
        next_type: str | None,
        next_value: Any,  # ruff: ignore[any-type]
        next_gate_info: dict | None,
    ) -> bool:
        """Check if two gates can be chained.

        A terminal (a `QSpider`/`PSpider` with arity (0,1) or (1,0)) is not
        a gate that can be chained with: it is a state or an effect, and
        absorbing it into another gate is `TerminalAbsorptionRule`'s job, not
        this rule's. So this returns False outright whenever either side
        looks like a terminal, regardless of the gate-type logic below.

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
        if (gate_type in {"Q", "P"} and gate_info is not None) and (
            gate_info.get("num_inputs") == 0 or gate_info.get("num_outputs") == 0
        ):
            return False
        if (next_type in {"Q", "P"} and next_gate_info is not None) and (
            next_gate_info.get("num_inputs") == 0 or next_gate_info.get("num_outputs") == 0
        ):
            return False

        if gate_type != next_type:
            return False
        if gate_type in {"Q", "P"}:
            return True

        if gate_type in {"R", "BS", "D", "Sq"}:
            return True

        if gate_type == "F":
            return value == next_value or {value, next_value} == {"F", "Finv"}

        if gate_type == "F2":
            return True

        if gate_type == "CZ":
            return True

        if gate_type == "CSUM":
            if gate_info is None or next_gate_info is None:
                msg = "get_gate_info returned no gate_info for a 'CSUM' node."
                raise ValueError(msg)
            return bool(
                (gate_info["control"] == next_gate_info["control"])
                and (gate_info["target"] == next_gate_info["target"])
            )

        return False

    def reduce_chain(self, gate_type: str, values: list, gate_info: dict | None = None) -> dict | None:  # ruff: ignore[complex-structure, too-many-return-statements, too-many-branches, too-many-statements]
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
        RuleApplicationError
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
                raise RuleApplicationError(msg)
            total_phase = zero_phase
            for v in values:
                total_phase += v
            # Only round-trip through .coeffs (which normalizes numeric
            # coefficients to plain float/complex) when something is actually
            # symbolic -- for an already fully-numeric phase, that round-trip
            # is pure downside: it can change the underlying sympy.Poly's
            # domain (e.g. exact integers -> floats), which breaks equality
            # against a phase built directly, for zero simplification benefit.
            if total_phase.is_parametric():
                total_phase = ZxPoly({d: simplify_reduced_value(c) for d, c in total_phase.coeffs.items()})
            return {
                "type": "QSpider",
                "phase": total_phase,
                "num_inputs": gate_info["num_inputs"],
                "num_outputs": gate_info["num_outputs"],
            }

        if gate_type == "P":
            if gate_info is None:
                msg = "get_gate_info returned no gate_info for a 'P' node."
                raise RuleApplicationError(msg)
            total_phase = ZxPoly({})
            for v in values:
                total_phase += v
            # Only round-trip through .coeffs (which normalizes numeric
            # coefficients to plain float/complex) when something is actually
            # symbolic -- for an already fully-numeric phase, that round-trip
            # is pure downside: it can change the underlying sympy.Poly's
            # domain (e.g. exact integers -> floats), which breaks equality
            # against a phase built directly, for zero simplification benefit.
            if total_phase.is_parametric():
                total_phase = ZxPoly({d: simplify_reduced_value(c) for d, c in total_phase.coeffs.items()})
            return {
                "type": "PSpider",
                "phase": total_phase,
                "num_inputs": gate_info["num_inputs"],
                "num_outputs": gate_info["num_outputs"],
            }

        if gate_type == "R":
            total = simplify_reduced_value(sum(values))
            if total == 0:
                return id_q
            return {"type": "PhaseRotationGate", "theta": total}

        if gate_type == "BS":
            total = simplify_reduced_value(sum(values))
            if total == 0:
                return id_q2
            return {"type": "BeamsplitterGate", "theta": total}

        if gate_type == "Sq":
            total = 1.0
            for v in values:
                total *= v
            total = simplify_reduced_value(total)
            if total == 1.0:  # ruff: ignore[float-equality-comparison]
                return id_q
            return {"type": "SqueezingGate", "tau": total}

        if gate_type == "D":
            total = simplify_reduced_value(sum(values))
            if total == 0:
                return id_q
            return {"type": "DisplacementGate", "alpha": total}

        if gate_type == "F":
            n = len(values)
            remainder = n % 4
            if remainder == 0:
                return id_q
            if remainder == 1:
                return {"type": "Fourier" if values[0] == "F" else "FourierInv"}
            if remainder == 2:  # ruff: ignore[magic-value-comparison]
                return {"type": "Fourier2"}
            return {"type": "FourierInv" if values[0] == "F" else "Fourier"}

        if gate_type == "F2":
            if len(values) % 2 == 0:
                return id_q
            return {"type": "Fourier2"}

        if gate_type == "F_pair":
            return id_q

        if gate_type == "CZ":
            total = simplify_reduced_value(sum(values))
            if total == 0:
                return id_q2
            return {"type": "ControlledZGate", "gain": total}

        if gate_type == "CSUM":
            if gate_info is None:
                msg = "get_gate_info returned no gate_info for a 'CSUM' node."
                raise RuleApplicationError(msg)
            total = simplify_reduced_value(sum(values))
            if total == 0:
                return id_q2
            return {
                "type": "ControlledSumGate",
                "gain": total,
                "control": gate_info["control"],
                "target": gate_info["target"],
            }

        return None

    def _update_node_for_reduced_gate(
        self, graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int, reduced_gate: dict
    ) -> None:
        """Update a node with reduced gate attributes.

        `node_id`'s own `container_id` is deliberately left untouched --
        it doesn't move, only its type/phase/arity change (mirroring
        `TerminalAbsorptionRule`/`CopyRule`'s own in-place survivors).
        `_flatten_container`, called separately once the chain's other
        members are spliced away, is what updates it on the rare
        occasion `node_id` ends up as its container's sole remaining
        child.
        """
        if node_id not in id_map:
            return
        attrs = graph[id_map[node_id]]
        gate_type = reduced_gate["type"]

        updates = {
            "type": gate_type,
        }

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
            pass

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

        attrs.update(updates)

    def _update_connectivity_after_reduction(
        self, graph: rx.PyDiGraph, id_map: dict[int, int], container_id: int, removed_connectivity: list[int]
    ) -> None:
        """Update connectivity after removing nodes from a composition."""
        if container_id not in id_map:
            return
        attrs = graph[id_map[container_id]]
        if "connectivity" not in attrs:
            return
        connectivity = attrs["connectivity"]
        new_connectivity = {}
        removed_set = set(removed_connectivity)
        index_map = {}
        new_idx = 0
        for old_idx in range(len(connectivity)):
            if old_idx not in removed_set:
                index_map[old_idx] = new_idx
                new_idx += 1

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
        id_map = _get_id_map(graph)
        matches = []

        for container_id in sorted(registry.composition_nodes):
            if container_id not in id_map:
                continue
            attrs = graph[id_map[container_id]]
            sub_ids = attrs.get("sub_diagram_ids", [])

            if len(sub_ids) < 2:  # ruff: ignore[magic-value-comparison]
                continue

            i = 0
            while i < len(sub_ids) - 1:
                first_id = sub_ids[i]
                second_id = sub_ids[i + 1]

                match = self._check_pair(graph, id_map, first_id, second_id)
                if match:
                    match["container_id"] = container_id
                    match["indices"] = [i, i + 1]
                    matches.append(match)
                    i += 2
                else:
                    i += 1

        return matches

    def _check_pair(self, graph: rx.PyDiGraph, id_map: dict[int, int], first_id: int, second_id: int) -> dict | None:
        """Check if a pair of adjacent nodes can be folded together.

        Returns
        -------
        dict | None
            Match dictionary if the pair can be folded, None otherwise.
        """
        if first_id not in id_map or second_id not in id_map:
            return None
        first_type = graph[id_map[first_id]].get("type")
        second_type = graph[id_map[second_id]].get("type")

        if first_type in self._ROTATION_DELTA and second_type == "PhaseRotationGate":
            return self._rotation_match(
                graph, id_map, fourier_id=first_id, rotation_id=second_id, first_id=first_id, second_id=second_id
            )
        if second_type in self._ROTATION_DELTA and first_type == "PhaseRotationGate":
            return self._rotation_match(
                graph, id_map, fourier_id=second_id, rotation_id=first_id, first_id=first_id, second_id=second_id
            )

        if first_type == "Fourier2" and second_type == "SqueezingGate":
            return self._squeezing_match(graph, id_map, squeezing_id=second_id, first_id=first_id, second_id=second_id)
        if second_type == "Fourier2" and first_type == "SqueezingGate":
            return self._squeezing_match(graph, id_map, squeezing_id=first_id, first_id=first_id, second_id=second_id)

        return None

    def _rotation_match(  # ruff: ignore[too-many-arguments]
        self,
        graph: rx.PyDiGraph,
        id_map: dict[int, int],
        *,
        fourier_id: int,
        rotation_id: int,
        first_id: int,
        second_id: int,
    ) -> dict:
        """Build the match dict for a Fourier-type/rotation pair."""  # ruff: ignore[docstring-missing-returns]
        fourier_type = graph[id_map[fourier_id]]["type"]
        theta = graph[id_map[rotation_id]]["phase"]
        return {
            "node_ids": [first_id, second_id],
            "result_type": "PhaseRotationGate",
            "result_value": theta + self._ROTATION_DELTA[fourier_type],
        }

    def _squeezing_match(
        self, graph: rx.PyDiGraph, id_map: dict[int, int], *, squeezing_id: int, first_id: int, second_id: int
    ) -> dict:
        """Build the match dict for an F2/squeezing pair."""  # ruff: ignore[docstring-missing-returns]
        tau = graph[id_map[squeezing_id]]["phase"]
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
        id_map = _get_id_map(graph)
        container_id = match["container_id"]
        keep_id, absorb_id = match["node_ids"]
        i, _ = match["indices"]

        if keep_id not in id_map or absorb_id not in id_map:
            return

        _contract_nodes(graph, id_map, keep_id, absorb_id)

        graph[id_map[keep_id]].update({
            "type": match["result_type"],
            "phase": match["result_value"],
            "container_id": container_id,
        })

        container_attrs = graph[id_map[container_id]]
        sub_ids = container_attrs.get("sub_diagram_ids", [])
        new_sub_ids = [sub_id for sub_id in sub_ids if sub_id != absorb_id]
        container_attrs["sub_diagram_ids"] = new_sub_ids

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
                else:
                    new_connectivity[old_idx - 1] = targets
            container_attrs["connectivity"] = new_connectivity

        if len(new_sub_ids) == 1:
            self._flatten_container(graph, id_map, container_id)
        else:
            self._refresh_composition_external_mappings(graph, id_map, container_id)


class TerminalAbsorptionRule(RewriteRule):
    r"""Terminal absorption rule - Graph-based version.

    Absorbs a gate adjacent to a QSpider/PSpider terminal (arity (1,0)
    effect or (0,1) state) into the terminal's own phase, eliminating the
    gate. Four sub-cases (the gate may sit on either side of the
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
        QSpider terminal with phase f(x) folds Sq(tau) into f(x * tau)
        and a PSpider terminal with phase f(x) folds Sq(tau) into f(x / tau).
    - Cross-color discard (opposite-color raw (1,1) spider): a QSpider
        terminal absorbing an adjacent PSpider(1,1,f(x)), or vice versa,
        leaves the terminal's phase unchanged -- the gate simply vanishes.
        Same-color (1,1) spiders are ordinary spider fusion, FusionRule's
        job, not this rule's.
    - Displacement (QSpider or PSpider terminal, input phase degree <=
        1, i.e. in R1[X] -- the same restriction `CopyRule.is_in_R1`
        enforces for its own copy pattern): a `DisplacementGate` adjacent
        to such a terminal simply collapses, phase unchanged, exactly like
        cross-color discard. A higher-degree terminal phase describes a
        genuinely squeezed (curved) eigenstate, for which an adjacent
        displacement is not simply irrelevant, so this sub-case doesn't
        match there (`_check_pair` returns no match rather than guessing).

    Only the rotation sub-case is an exact identity for any physical
    state. Squeezing absorption, cross-color discard, and displacement
    absorption all treat the terminal's phase as an idealized (infinite
    squeezing) eigenstate -- a real finite-squeezed state would carry
    extra terms these sub-cases drop. `assume_infinite_squeezing`
    (default False) gates whether those sub-cases are allowed to match
    at all; when False, only rotation absorption runs.

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

    def match(self, cvzx_graph: CVZXGraph) -> list[dict]:  # ruff: ignore[too-many-locals]
        """Find all gate/terminal pairs that can be absorbed.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to search, together with its registry.

        Returns
        -------
        list[dict]
            List of matches, each containing:

            - 'container_id': grouping key for `RewriteRule.apply_rule`
              (the pair's shared immediate parent when they have one and
              no identity/`Swap` sits between them, else a synthetic
              per-match key)
            - 'node_ids': [keep_id, absorb_id] in list order (keep_id survives)
            - 'identity_chain': passthrough node ids crossed to find this
              pair, nearest the terminal first
            - 'result_type': 'QSpider' or 'PSpider'
            - 'result_num_inputs' / 'result_num_outputs': the terminal's own arity
            - 'result_phase': the folded phase
        """
        graph = cvzx_graph.graph
        registry = cvzx_graph.registry
        id_map = _get_id_map(graph)
        ordered_matches: list[tuple[int, int, dict]] = []
        used_nodes: set[int] = set()

        candidates = sorted(node for node in registry.input_states | registry.measurement_nodes if node in id_map)

        for terminal_id in candidates:
            if terminal_id in used_nodes:
                continue

            forward = graph[id_map[terminal_id]].get("num_outputs") == 1
            gate_id, _, identity_chain = self._chase_identity_chain(graph, id_map, terminal_id, forward=forward)
            if gate_id is None or gate_id in used_nodes or any(cid in used_nodes for cid in identity_chain):
                continue
            if sum(1 for cid in identity_chain if graph[id_map[cid]].get("type") == "Swap") > 1:
                continue

            first_id, second_id = (terminal_id, gate_id) if forward else (gate_id, terminal_id)

            first_container_id = graph[id_map[first_id]].get("container_id")
            second_container_id = graph[id_map[second_id]].get("container_id")
            if (  # ruff: ignore[too-many-boolean-expressions]
                first_container_id is not None
                and first_container_id in id_map
                and graph[id_map[first_container_id]].get("container_type") == "contracted"
            ) or (
                second_container_id is not None
                and second_container_id in id_map
                and graph[id_map[second_container_id]].get("container_type") == "contracted"
            ):
                continue

            match = self._check_pair(graph, id_map, first_id, second_id)
            if match is None:
                continue
            match["identity_chain"] = identity_chain

            first_parent = graph[id_map[first_id]].get("container_id")
            second_parent = graph[id_map[second_id]].get("container_id")
            if (
                not identity_chain
                and first_parent == second_parent
                and first_parent is not None
                and first_parent in id_map
            ):
                match["container_id"] = first_parent
                sub_ids = graph[id_map[first_parent]].get("sub_diagram_ids", [])
                if first_id in sub_ids and second_id in sub_ids:
                    match["indices"] = sorted([sub_ids.index(first_id), sub_ids.index(second_id)])
            else:
                match["container_id"] = ("cross", first_id, second_id)

            ordered_matches.append((first_id, second_id, match))
            used_nodes.add(first_id)
            used_nodes.add(second_id)
            used_nodes.update(identity_chain)

        ordered_matches.sort(key=operator.itemgetter(0, 1))
        return [entry[2] for entry in ordered_matches]

    def _check_pair(self, graph: rx.PyDiGraph, id_map: dict[int, int], first_id: int, second_id: int) -> dict | None:
        """Check if a pair of adjacent nodes forms an absorbable gate/terminal pattern.

        Returns
        -------
        dict | None
            Match dictionary if the pair can be folded, None otherwise.
        """
        if first_id not in id_map or second_id not in id_map:
            return None
        first_attrs = graph[id_map[first_id]]
        second_attrs = graph[id_map[second_id]]

        if (
            first_attrs.get("num_inputs") == 0
            and first_attrs.get("num_outputs") == 1
            and first_attrs.get("type") in {"QSpider", "PSpider"}
        ):
            result = self._try_absorb(terminal_attrs=first_attrs, gate_attrs=second_attrs)
            if result:
                result["node_ids"] = [first_id, second_id]
                return result

        if (
            second_attrs.get("num_inputs") == 1
            and second_attrs.get("num_outputs") == 0
            and second_attrs.get("type") in {"QSpider", "PSpider"}
        ):
            result = self._try_absorb(terminal_attrs=second_attrs, gate_attrs=first_attrs)
            if result:
                result["node_ids"] = [second_id, first_id]
                return result

        return None

    def _try_absorb(self, *, terminal_attrs: dict, gate_attrs: dict) -> dict | None:  # ruff: ignore[complex-structure]
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
            if phase is None:
                msg = f"terminal_attrs has no 'phase' to absorb {gate_type!r} into."
                raise ValueError(msg)
            new_phase = self._rotation_absorb(phase, pi)
        elif gate_type == "SqueezingGate" and self.assume_infinite_squeezing:
            if phase is None:
                msg = f"terminal_attrs has no 'phase' to absorb {gate_type!r} into."
                raise ValueError(msg)
            is_q_spider = terminal_type == "QSpider"
            new_phase = self._squeeze_absorb(phase, gate_attrs.get("phase"), is_q_spider)
        elif (
            gate_type in {"QSpider", "PSpider"}
            and gate_type != terminal_type
            and gate_attrs.get("num_inputs") == 1
            and gate_attrs.get("num_outputs") == 1
            and self.assume_infinite_squeezing
        ):
            new_phase = phase
        elif gate_type == "DisplacementGate" and self.assume_infinite_squeezing:
            if phase is None:
                msg = f"terminal_attrs has no 'phase' to absorb {gate_type!r} into."
                raise ValueError(msg)
            new_phase = self._displacement_absorb(phase)

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
    def _squeeze_absorb(phase: ZxPoly, tau: float | Expr, is_q_spider: bool) -> ZxPoly:  # ruff: ignore[boolean-type-hint-positional-argument]
        """Fold a squeezing Sq(tau) into a terminal's phase (any degree).

        Substituting x -> factor*x leaves each x**d term's degree
        unchanged and multiplies its coefficient by factor**d (built via
        the dict form -- ZxPoly's expr+gen constructor path has a latent
        bug, see base_gates.py). `factor` is `tau` for a `QSpider`
        terminal and `1/tau` for a `PSpider` one -- squeezing scales the
        conjugate Q/P quadratures reciprocally, per the cited reference.

        Returns
        -------
        ZxPoly
            Update phase after applying the Squeezing rule.

        References
        ----------
        [1] Nagayoshi et al., CV ZX calculus, 2024, Eq. (82)-(83).
        """
        factor = tau if is_q_spider else 1 / tau
        return ZxPoly({degree: coeff * factor**degree for degree, coeff in phase.coeffs.items()})

    @staticmethod
    def _displacement_absorb(phase: ZxPoly) -> ZxPoly | None:
        """Fold a `DisplacementGate` into a terminal's phase, if the phase allows it.

        Like squeezing absorption, this idealizes the terminal as an
        infinite-squeezing eigenstate -- but unlike squeezing (any
        degree) or rotation (its own, separate degree <= 1 restriction),
        the eigenstate stays exact under an adjacent displacement only
        when its own phase is already in R1[X] (degree <= 1, the same
        restriction `CopyRule.is_in_R1` enforces for its own copy
        pattern): a higher-degree phase describes a genuinely squeezed
        (curved) state, for which an adjacent displacement is not simply
        irrelevant. When the phase does qualify, the gate contributes
        nothing to it at all -- it just collapses, phase unchanged,
        exactly like cross-color discard.

        Returns
        -------
        ZxPoly | None
            `phase` unchanged if its degree is <= 1, else None (no
            match).
        """
        if phase.degree() > 1:
            return None
        return phase

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
            return False
        return math.isclose(abs(theta_val) % math.pi, math.pi / 2)

    @staticmethod
    def _reset_to_identity(graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int) -> None:
        """Reset a node in place to a zero-phase identity spider.

        The node keeps its own slot, container, and port counts. Its type
        becomes the spider color that identity should have:

        - A raw `QSpider`/`PSpider` keeps its own color.
        - Every other gate type (`PhaseRotationGate`, `SqueezingGate`,
        `DisplacementGate`, `BeamsplitterGate`, `ControlledSumGate`,
        `ControlledZGate`, `Fourier`, `FourierInv`, `Fourier2`, `Swap`)
        becomes `QSpider`. A `Swap` reset to a `QSpider(2, 2, 0)` is a
        zero-phase wide spider, not a permutation -- the chase's crossing
        has been compensated for by the terminal absorbing the phase, so
        the crossing is no longer needed.

        `kind` is set to `"proper"` so downstream rules (`IdentityRule` in
        particular) recognize the result as a genuine proper node.
        `feedforward`/`measurement_ids`/`param_measurement_map` are cleared.

        Parameters
        ----------
        graph : rx.PyDiGraph
            The graph to modify.
        node_id : int
            The node to reset.
        """
        if node_id not in id_map:
            return
        attrs = graph[id_map[node_id]]
        gate_type = attrs.get("type")

        identity_type = gate_type if gate_type in {"QSpider", "PSpider"} else "QSpider"

        attrs.update({
            "type": identity_type,
            "kind": "proper",
            "phase": ZxPoly({}),
            "feedforward": None,
            "measurement_ids": None,
            "param_measurement_map": {},
        })

    def apply_single(self, cvzx_graph: CVZXGraph, match: dict) -> None:
        """Absorb a gate's phase into an adjacent terminal's own phase.

        The terminal (`keep_id`) keeps its own node ID, slot, arity, and
        container -- only its phase is updated to the fold result. The gate
        (`absorb_id`) is reset in place to a zero-phase identity spider of
        its own color and arity. Every identity/`Swap` passthrough the
        chase crossed is also reset in place to a zero-phase identity
        spider of its own color and arity. Nothing is contracted, spliced,
        or voided, and no other rule is invoked.

        The composition hierarchy, connectivity, and every container's
        shape stay exactly as they were.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match containing fold information. `match["node_ids"]` is
            `[terminal_id, gate_id]` -- the terminal is `node_ids[0]`, the
            gate is `node_ids[1]`.
        """
        graph = cvzx_graph.graph
        id_map = _get_id_map(graph)
        terminal_id, gate_id = match["node_ids"]
        if terminal_id not in id_map or gate_id not in id_map:
            return

        keep_id = terminal_id
        absorb_id = gate_id

        graph[id_map[keep_id]].update({
            "type": match["result_type"],
            "phase": match["result_phase"],
        })

        self._reset_to_identity(graph, id_map, absorb_id)

        for passthrough_id in match.get("identity_chain", []):
            if passthrough_id not in id_map or graph[id_map[passthrough_id]]["type"] == "Swap":
                continue
            self._reset_to_identity(graph, id_map, passthrough_id)


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
        id_map = _get_id_map(graph)
        ordered_matches: list[tuple[int, int, dict]] = []

        candidates = sorted(
            node
            for node in registry.input_states | registry.measurement_nodes
            if node in id_map and graph[id_map[node]].get("type") in {"QSpider", "PSpider"}
        )

        for copy_id in candidates:
            forward = graph[id_map[copy_id]].get("num_outputs") == 1
            neighbor_id, _, identity_chain = self._chase_identity_chain(graph, id_map, copy_id, forward=forward)
            if neighbor_id is None:
                continue

            first_id, second_id = (copy_id, neighbor_id) if forward else (neighbor_id, copy_id)
            base_match = self._check_pair(graph, id_map, first_id, second_id)
            if base_match is None:
                continue
            resolved = self._resolve_containers(graph, id_map, base_match, identity_chain)
            if resolved is None:
                continue

            base_match.update(resolved)
            base_match["identity_chain"] = identity_chain
            ordered_matches.append((first_id, second_id, base_match))

        ordered_matches.sort(key=operator.itemgetter(0, 1))
        return [entry[2] for entry in ordered_matches]

    def _resolve_containers(
        self, graph: rx.PyDiGraph, id_map: dict[int, int], base_match: dict, identity_chain: list[int] | None = None
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
        copy_parent_id = graph[id_map[copy_id]].get("container_id")
        disappear_parent_id = graph[id_map[disappear_id]].get("container_id")
        copy_parent_type = graph[id_map[copy_parent_id]].get("container_type") if copy_parent_id in id_map else None
        disappear_parent_type = (
            graph[id_map[disappear_parent_id]].get("container_type") if disappear_parent_id in id_map else None
        )

        if not identity_chain and copy_parent_id == disappear_parent_id:
            if copy_parent_type != "composition":
                return None
            sub_ids = graph[id_map[copy_parent_id]].get("sub_diagram_ids", [])
            if copy_id not in sub_ids or disappear_id not in sub_ids:
                return None
            indices = sorted([sub_ids.index(copy_id), sub_ids.index(disappear_id)])
            return {
                "same_parent": True,
                "container_id": copy_parent_id,
                "indices": indices,
            }

        if copy_parent_type not in {"tensor", "composition"}:
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

    def _check_pair(self, graph: rx.PyDiGraph, id_map: dict[int, int], first_id: int, second_id: int) -> dict | None:
        """Check if a pair of nodes forms a copy-able pattern.

        Returns
        -------
        dict | None
            Match dictionary if the pair is copy-able, None otherwise.
        """
        if first_id not in id_map or second_id not in id_map:
            return None
        first_attrs = graph[id_map[first_id]]
        second_attrs = graph[id_map[second_id]]

        first_type = first_attrs.get("type")
        second_type = second_attrs.get("type")

        first_num_inputs, first_num_outputs = _external_arity(graph, id_map, first_id)
        second_num_inputs, second_num_outputs = _external_arity(graph, id_map, second_id)

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
                id_map,
                copy_spider_id=first_id,
                disappearing_spider_id=second_id,
                n_copies=second_num_outputs,
                copy_spider_type="Q",
            )

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
                id_map,
                copy_spider_id=second_id,
                disappearing_spider_id=first_id,
                n_copies=first_num_inputs,
                copy_spider_type="Q",
            )

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
                id_map,
                copy_spider_id=first_id,
                disappearing_spider_id=second_id,
                n_copies=second_num_outputs,
                copy_spider_type="P",
            )

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
                id_map,
                copy_spider_id=second_id,
                disappearing_spider_id=first_id,
                n_copies=first_num_inputs,
                copy_spider_type="P",
            )

        return None

    def _create_match(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        graph: rx.PyDiGraph,
        id_map: dict[int, int],
        copy_spider_id: int,
        disappearing_spider_id: int,
        n_copies: int,
        copy_spider_type: str,
    ) -> dict | None:
        """Create a match dictionary if the copied spider's phase is in R₁[X]."""  # ruff: ignore[docstring-missing-returns]
        copy_attrs = graph[id_map[copy_spider_id]]
        copy_phase = copy_attrs.get("phase")

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
        id_map = _get_id_map(graph)
        copy_spider_id = match["copy_spider_id"]
        disappearing_spider_id = match["disappearing_spider_id"]
        n_copies = match["n_copies"]
        copy_spider_type = match["copy_spider_type"]
        copy_spider_phase = match["copy_spider_phase"]
        copy_num_inputs = match["copy_spider_num_inputs"]
        copy_num_outputs = match["copy_spider_num_outputs"]
        same_parent = match.get("same_parent", True)

        for identity_id in match.get("identity_chain", []):
            if identity_id in id_map:
                identity_attrs = graph[id_map[identity_id]]
                identity_attrs["type"] = "VoidDiagram"
                identity_attrs["phase"] = None

        if same_parent:
            container_id = match["container_id"]
            idx1, idx2 = match["indices"]
            container_attrs = graph[id_map[container_id]]
            sub_ids = container_attrs.get("sub_diagram_ids", [])

        _contract_nodes(graph, id_map, copy_spider_id, disappearing_spider_id)

        copy_ids = []
        for _ in range(n_copies):
            new_id = max(id_map.keys()) + 1 if id_map else 0
            attrs = {
                "id": new_id,
                "type": f"{copy_spider_type}Spider",
                "kind": "proper",
                "phase": copy_spider_phase,
                "num_inputs": copy_num_inputs,
                "num_outputs": copy_num_outputs,
                "container_id": copy_spider_id,
                "external_inputs": list(range(copy_num_inputs)),
                "external_outputs": list(range(copy_num_outputs)),
            }
            idx = graph.add_node(attrs)
            id_map[new_id] = idx
            copy_ids.append(new_id)

        graph[id_map[copy_spider_id]].update({
            "type": "TensorDiagram",
            "kind": "container",
            "container_type": "tensor",
            "sub_diagram_ids": copy_ids,
            "num_inputs": n_copies * copy_num_inputs,
            "num_outputs": n_copies * copy_num_outputs,
            "phase": None,
        })

        input_mapping, offset = {}, 0
        for idx, sub_id in enumerate(copy_ids):
            for p in range(graph[id_map[sub_id]].get("num_inputs", 0)):
                input_mapping[offset] = (idx, p)
                offset += 1
        output_mapping, offset = {}, 0
        for idx, sub_id in enumerate(copy_ids):
            for p in range(graph[id_map[sub_id]].get("num_outputs", 0)):
                output_mapping[offset] = (idx, p)
                offset += 1
        graph[id_map[copy_spider_id]]["external_inputs"] = list(range(len(input_mapping)))
        graph[id_map[copy_spider_id]]["external_outputs"] = list(range(len(output_mapping)))
        graph[id_map[copy_spider_id]]["external_input_mapping"] = input_mapping
        graph[id_map[copy_spider_id]]["external_output_mapping"] = output_mapping

        if same_parent:
            new_sub_ids = [*sub_ids[:idx1], copy_spider_id, *sub_ids[idx2 + 1 :]]
            graph[id_map[container_id]]["sub_diagram_ids"] = new_sub_ids

            if "connectivity" in container_attrs:
                connectivity = container_attrs["connectivity"]
                new_connectivity = {}

                for old_idx, targets in connectivity.items():
                    if old_idx == idx1:
                        continue
                    if old_idx == idx2:
                        new_connectivity[idx1] = targets
                    elif old_idx < idx1:
                        new_connectivity[old_idx] = targets
                    else:
                        new_connectivity[old_idx - 1] = targets

                graph[id_map[container_id]]["connectivity"] = new_connectivity
            if len(sub_ids) == 2:  # ruff: ignore[magic-value-comparison]
                self._flatten_container(graph, id_map, container_id)
            else:
                self._refresh_composition_external_mappings(graph, id_map, container_id)
            return

        disappear_container_id = match["disappear_container_id"]
        disappear_container_type = match["disappear_container_type"]
        copy_container_id = match["copy_container_id"]

        disappear_parent_attrs = (
            graph[id_map[disappear_container_id]]
            if disappear_container_type == "contracted" and disappear_container_id in id_map
            else None
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

        pad_id = max(id_map.keys()) + 1 if id_map else 0
        pad_attrs = {
            "id": pad_id,
            "type": "VoidDiagram",
            "kind": "proper",
            "phase": None,
            "num_inputs": copy_num_outputs,
            "num_outputs": copy_num_inputs,
            "container_id": copy_spider_id,
            "external_inputs": list(range(copy_num_outputs)),
            "external_outputs": list(range(copy_num_inputs)),
        }
        pad_idx = graph.add_node(pad_attrs)
        id_map[pad_id] = pad_idx
        graph[id_map[copy_spider_id]]["sub_diagram_ids"].append(pad_id)

        consumed_output_placeholders = []
        for _ in consumed_output_indices:
            ph_id = max(id_map.keys()) + 1 if id_map else 0
            ph_attrs = {
                "id": ph_id,
                "type": f"{copy_spider_type}Spider",
                "kind": "proper",
                "phase": copy_spider_phase,
                "num_inputs": 0,
                "num_outputs": 1,
                "container_id": copy_spider_id,
                "external_inputs": [],
                "external_outputs": [0],
            }
            ph_idx = graph.add_node(ph_attrs)
            id_map[ph_id] = ph_idx
            graph[id_map[copy_spider_id]]["sub_diagram_ids"].append(ph_id)
            consumed_output_placeholders.append(ph_id)

        consumed_input_placeholders = []
        for _ in consumed_input_indices:
            ph_id = max(id_map.keys()) + 1 if id_map else 0
            ph_attrs = {
                "id": ph_id,
                "type": f"{copy_spider_type}Spider",
                "kind": "proper",
                "phase": copy_spider_phase,
                "num_inputs": 1,
                "num_outputs": 0,
                "container_id": copy_spider_id,
                "external_inputs": [0],
                "external_outputs": [],
            }
            ph_idx = graph.add_node(ph_attrs)
            id_map[ph_id] = ph_idx
            graph[id_map[copy_spider_id]]["sub_diagram_ids"].append(ph_id)
            consumed_input_placeholders.append(ph_id)

        self._recompute_tensor_arity(graph, id_map, copy_spider_id)

        if disappear_container_type == "contracted":
            assert disappear_parent_attrs is not None  # ruff: ignore[assert] -- see its definition above
            sub_ids = graph[id_map[copy_spider_id]]["sub_diagram_ids"]
            output_mapping = graph[id_map[copy_spider_id]]["external_output_mapping"]
            input_mapping = graph[id_map[copy_spider_id]]["external_input_mapping"]
            reverse_output = {tuple(target): port for port, target in output_mapping.items()}
            reverse_input = {tuple(target): port for port, target in input_mapping.items()}

            new_i1_or_i2 = [reverse_input[sub_ids.index(ph_id), 0] for ph_id in consumed_input_placeholders]
            new_j1_or_j2 = [reverse_output[sub_ids.index(ph_id), 0] for ph_id in consumed_output_placeholders]

            if disappearing_is_first:
                disappear_parent_attrs["J1"] = new_i1_or_i2
                disappear_parent_attrs["I1"] = new_j1_or_j2
                disappear_parent_attrs["first_id"] = copy_spider_id
            else:
                disappear_parent_attrs["I2"] = new_i1_or_i2
                disappear_parent_attrs["J2"] = new_j1_or_j2
                disappear_parent_attrs["second_id"] = copy_spider_id

            arity_change = self._recompute_contracted_arity(graph, id_map, disappear_container_id)
            self._propagate_arity_to_parent(graph, id_map, disappear_container_id, *arity_change)
        else:
            disappear_sub_ids = graph[id_map[disappear_container_id]]["sub_diagram_ids"]
            disappear_sub_ids[disappear_sub_ids.index(disappearing_spider_id)] = copy_spider_id
        graph[id_map[copy_spider_id]]["container_id"] = disappear_container_id

        void_id = max(id_map.keys()) + 1 if id_map else 0
        void_attrs = {
            "id": void_id,
            "type": "VoidDiagram",
            "kind": "proper",
            "phase": None,
            "num_inputs": copy_num_inputs,
            "num_outputs": copy_num_outputs,
            "container_id": copy_container_id,
            "external_inputs": list(range(copy_num_inputs)),
            "external_outputs": list(range(copy_num_outputs)),
        }
        void_idx = graph.add_node(void_attrs)
        id_map[void_id] = void_idx

        copy_sub_ids = graph[id_map[copy_container_id]]["sub_diagram_ids"]
        copy_sub_ids[copy_sub_ids.index(copy_spider_id)] = void_id


# =========================================================================
# Utility Functions
# =========================================================================


def _external_arity(graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int) -> tuple:
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
    graph : rx.PyDiGraph
        The graph to read from.
    node_id : int
        The leaf node to compute the external arity of.

    Returns
    -------
    tuple
        (external_num_inputs, external_num_outputs).
    """
    attrs = graph[id_map[node_id]]
    parent_id = attrs.get("container_id")
    parent_attrs = graph[id_map[parent_id]] if parent_id is not None and parent_id in id_map else None

    if parent_attrs is not None and parent_attrs.get("container_type") == "contracted":
        if parent_attrs.get("first_id") == node_id:
            return len(parent_attrs.get("kept_first_inputs", [])), len(parent_attrs.get("kept_first_outputs", []))
        if parent_attrs.get("second_id") == node_id:
            return len(parent_attrs.get("kept_second_inputs", [])), len(parent_attrs.get("kept_second_outputs", []))

    return attrs.get("num_inputs"), attrs.get("num_outputs")


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


# =========================================================================
# Graph Mutation & Lookup Helpers for rustworkx
# =========================================================================


def _get_id_map(graph: rx.PyDiGraph) -> dict[int, int]:
    """Return a dictionary mapping diagram node IDs to PyDiGraph node indices."""  # ruff: ignore[docstring-missing-returns]
    return {graph[idx]["id"]: idx for idx in graph.node_indices()}


def _remove_node(graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int) -> None:
    """Safely remove a node by diagram node ID."""
    idx = id_map.get(node_id)
    if idx is not None:
        graph.remove_node(idx)
        del id_map[node_id]


def _get_out_edges(graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int) -> list[tuple[int, int, dict]]:
    """Return outgoing edges as (src_id, target_id, edge_data) tuples."""  # ruff: ignore[docstring-missing-returns]
    idx = id_map.get(node_id)
    if idx is None:
        return []
    res = []
    for _src_idx, tgt_idx, edge_data in graph.out_edges(idx):
        tgt_id = graph[tgt_idx]["id"]
        res.append((node_id, tgt_id, edge_data))
    return res


def _get_in_edges(graph: rx.PyDiGraph, id_map: dict[int, int], node_id: int) -> list[tuple[int, int, dict]]:
    """Return incoming edges as (src_id, target_id, edge_data) tuples."""  # ruff: ignore[docstring-missing-returns]
    idx = id_map.get(node_id)
    if idx is None:
        return []
    res = []
    for src_idx, _tgt_idx, edge_data in graph.in_edges(idx):
        src_id = graph[src_idx]["id"]
        res.append((src_id, node_id, edge_data))
    return res


def _has_edge(graph: rx.PyDiGraph, id_map: dict[int, int], u_id: int, v_id: int) -> bool:
    u_idx = id_map.get(u_id)
    v_idx = id_map.get(v_id)
    if u_idx is None or v_idx is None:
        return False
    return graph.has_edge(u_idx, v_idx)


def _remove_edge(graph: rx.PyDiGraph, id_map: dict[int, int], u_id: int, v_id: int) -> None:
    u_idx = id_map.get(u_id)
    v_idx = id_map.get(v_id)
    if u_idx is not None and v_idx is not None and graph.has_edge(u_idx, v_idx):
        graph.remove_edge(u_idx, v_idx)


def _contract_nodes(graph: rx.PyDiGraph, id_map: dict[int, int], keep_id: int, remove_id: int) -> None:
    """Contract remove_id into keep_id in rustworkx graph."""
    keep_idx = id_map.get(keep_id)
    remove_idx = id_map.get(remove_id)
    if keep_idx is None or remove_idx is None:
        return

    for src_idx, _, edge_data in list(graph.in_edges(remove_idx)):
        if src_idx != keep_idx:
            graph.add_edge(src_idx, keep_idx, edge_data)

    for _, tgt_idx, edge_data in list(graph.out_edges(remove_idx)):
        if tgt_idx != keep_idx:
            graph.add_edge(keep_idx, tgt_idx, edge_data)

    graph.remove_node(remove_idx)
    del id_map[remove_id]

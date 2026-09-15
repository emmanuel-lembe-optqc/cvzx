"""CV ZX calculus rewrite rules and applications.

This module implements the 10 basic rewrite rules from :cite:`nagayoshi2024zx` Sec. IV.A,
and the derived rules from Sec. IV.B; using a graph structure.
"""

import logging
import math
import operator
from abc import ABC, abstractmethod
from typing import Any

import networkx as nx
from sympy import Expr, cos, pi, tan

from cvzx.backends.nx.graph import (
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

# Cap on how many internal match/apply rounds `RewriteRule.apply_rule` runs
# per call before returning control to its caller. Kept small -- rather than
# looping until this rule reaches its own full local fixed point -- because
# `optimize()`'s own outer `_simplify_to_fixed_point` loop already re-invokes
# every rule's `apply_rule` again on the very next pass whenever that pass
# changed anything, and keeps doing so until a whole pass changes nothing.
# So whatever this call doesn't finish is picked up there at no extra total
# cost, while a rule whose own match can chase through an arbitrarily long
# run of passthrough structure (a permutation of `Swap` nodes, say) never has to
# pay for verifying/resolving all of it inside one `apply_rule` call. See
# `RewriteRule.apply_rule` for why more than one round is ever useful.
_APPLY_RULE_MAX_ROUNDS = 100


class RewriteRule(ABC):
    """Base class for rewrite rules operating on a `CVZXGraph` in-place.

    Subclasses implement `match()` (find every independent match in the
    graph) and `apply_single()` (apply exactly one match). `apply_rule()`
    (defined once here, never overridden) drives both through a whole
    round of matches. See :doc:`../../dev_guide/rewrite_engine` for the full
    contract, the match-dict convention, and why rules operate on the
    graph rather than walking the `Diagram` tree directly.
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
        """Apply the rule to `graph` for up to `_APPLY_RULE_MAX_ROUNDS` rounds.

        Each round: find every match, group by `container_id`, apply each
        group in reverse position order (so an earlier removal can't shift
        a later match's index), then rebuild `graph.registry`. Repeats
        while a round still finds matches, up to `_APPLY_RULE_MAX_ROUNDS`
        rounds -- see :doc:`../../dev_guide/rewrite_engine` for why more than
        one round is sometimes needed and why the cap is kept small.

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
        for round_index in range(_APPLY_RULE_MAX_ROUNDS):
            matches = self.match(graph)

            if not matches:
                logger.debug("%s: no matches, done after %d round(s)", type(self).__name__, round_index)
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

            logger.debug("%s: round %d, %d match(es), applying", type(self).__name__, round_index, len(matches))
            for key in group_order:
                for match in reversed(groups[key]):
                    self.apply_single(graph, match)

            # `match()` implementations read `graph.registry` rather than
            # rescanning `graph.graph` from scratch, and nothing here keeps
            # that registry in sync incrementally as `apply_single` mutates
            # the graph -- so it must be rebuilt before the next round's
            # `match()` call, exactly as `_simplify_to_fixed_point` already
            # does between rules. Skipping this is not just stale data: a
            # match naming a node `apply_single` removed makes the next
            # `match()` call crash outright.
            graph.rebuild_registry()

        # Reaching the cap is routine, not exceptional: `optimize()`'s own
        # outer loop (see the docstring above) will call `apply_rule` again
        # on its next pass if anything here still needs resolving, so this
        # is logged at debug level, not warned as if this call needed to
        # finish everything itself.
        logger.debug(
            "%s.apply_rule: reached _APPLY_RULE_MAX_ROUNDS=%d rounds this call -- "
            "any remaining matches will be picked up by optimize()'s next pass",
            type(self).__name__,
            _APPLY_RULE_MAX_ROUNDS,
        )
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

    def _refresh_composition_external_mappings(self, graph: nx.DiGraph, container_id: int) -> None:
        """Recompute a `CompositionDiagram` container's own external port mappings.

        A composition's external inputs are exactly its first
        sub-diagram's inputs, and its external outputs exactly its last
        sub-diagram's outputs (see `to_graph`'s own construction of
        `external_input_mapping`/`external_output_mapping` for a
        `CompositionDiagram`) -- so both go stale the moment
        `sub_diagram_ids` is spliced (an element removed, shrinking the
        list) without recomputing them. In particular,
        `external_output_mapping`'s `{j: (last_idx, j)}` keeps pointing
        at whatever `last_idx` used to be: once the list is shorter, that
        index either points at the wrong sub-diagram or is out of range
        entirely -- silently corrupting anything that later resolves a
        port through it (`find_node_by_external_output`/`_input`, used
        by several rules' own `match()` to chase a chain of identities/
        `Swap` nodes across container boundaries), even though `to_diagram`
        itself never notices, since reconstruction only reads
        `sub_diagram_ids`/`connectivity`, not this mapping.

        Call this right after any such splice, whenever more than one
        sub-diagram remains (a single-element container is flattened
        away entirely instead, via `_flatten_container`, making its own
        mapping moot).

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        container_id : int
            The composition container whose `sub_diagram_ids` was just
            spliced.
        """
        attrs = graph.nodes[container_id]
        if attrs.get("container_type") != "composition":
            return
        sub_ids = attrs.get("sub_diagram_ids", [])
        if not sub_ids:
            return
        first_attrs = graph.nodes[sub_ids[0]]
        last_attrs = graph.nodes[sub_ids[-1]]
        attrs["external_input_mapping"] = {j: (0, j) for j in range(first_attrs.get("num_inputs", 0))}
        last_idx = len(sub_ids) - 1
        attrs["external_output_mapping"] = {j: (last_idx, j) for j in range(last_attrs.get("num_outputs", 0))}

    @staticmethod
    def _composition_step(
        graph: nx.DiGraph, node_id: int, port: int, *, forward: bool
    ) -> tuple[int | None, int | None]:
        """Follow the single composition edge carrying one specific port onward.

        Parameters
        ----------
        graph : nx.DiGraph
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
            for _, v, edge_attrs in graph.out_edges(node_id, data=True):
                if edge_attrs.get("edge_type") != "composition":
                    continue
                source_ports = edge_attrs.get("source_ports", [])
                if port in source_ports:
                    return v, edge_attrs["target_ports"][source_ports.index(port)]
        else:
            for u, _, edge_attrs in graph.in_edges(node_id, data=True):
                if edge_attrs.get("edge_type") != "composition":
                    continue
                target_ports = edge_attrs.get("target_ports", [])
                if port in target_ports:
                    return u, edge_attrs["source_ports"][target_ports.index(port)]
        return None, None

    def _rebuild_composition_connectivity(self, graph: nx.DiGraph, container_id: int) -> None:
        """Rebuild a CompositionDiagram container's `connectivity` from its children.

        Every adjacent pair `(i, i+1)` in the composition's `sub_diagram_ids`
        is wired by the identity mapping over the intersection of slot `i`'s
        outputs and slot `i+1`'s inputs: `{in_port: out_port for port in
        range(min(left.num_outputs, right.num_inputs))}`. If either side has
        zero ports on the relevant side (e.g. a freshly-inserted
        `VoidDiagram(0, 0)`, or a terminal with no outputs), the pair's entry
        is empty.

        Also refreshes the container's own `num_inputs` / `num_outputs` and
        its `external_inputs` / `external_outputs` lists, since a flat
        composition's external arity is exactly its first child's inputs and
        its last child's outputs. (`external_input_mapping` and
        `external_output_mapping` are refreshed by
        `_refresh_composition_external_mappings`, which the caller is
        expected to invoke after this method.)

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        container_id : int
            The CompositionDiagram container node ID.
        """
        attrs = graph.nodes[container_id]
        if attrs.get("container_type") != "composition":
            return

        sub_ids = attrs.get("sub_diagram_ids", [])

        # Pairs of adjacent sub-diagrams, wired by the identity mapping over
        # the intersection of their arities on the relevant sides.
        new_conn: dict[int, dict[int, int]] = {}
        for i in range(len(sub_ids) - 1):
            left = graph.nodes[sub_ids[i]]
            right = graph.nodes[sub_ids[i + 1]]
            n = min(left.get("num_outputs", 0), right.get("num_inputs", 0))
            new_conn[i] = {j: j for j in range(n)}
        attrs["connectivity"] = new_conn

        # External arity of a flat composition is its first child's inputs
        # and its last child's outputs.
        if sub_ids:
            attrs["num_inputs"] = graph.nodes[sub_ids[0]].get("num_inputs", 0)
            attrs["num_outputs"] = graph.nodes[sub_ids[-1]].get("num_outputs", 0)
        else:
            attrs["num_inputs"] = 0
            attrs["num_outputs"] = 0

        new_num_inputs = attrs["num_inputs"]
        new_num_outputs = attrs["num_outputs"]
        attrs["external_inputs"] = list(range(new_num_inputs))
        attrs["external_outputs"] = list(range(new_num_outputs))

    def _chase_identity_chain(
        self, graph: nx.DiGraph, start_id: int, *, forward: bool, start_port: int = 0
    ) -> tuple[int | None, int | None, list[int]]:
        """Follow one wire from `start_id`, stepping over identity/`Swap` passthroughs.

        Walks the graph's own "composition" edges one hop at a time,
        starting from a specific port of `start_id`, continuing past
        every node that is itself a pure passthrough -- an identity
        spider (zero-phase, raw arity (1,1)) or a `Swap` -- until it
        reaches the first node that isn't. Both kinds of passthrough have
        no bearing on whatever pattern is being matched around them: an
        identity is a bare wire, and a `Swap` only re-labels which port
        the wire arrives/leaves on, tracked here via `port`. That node is
        the real neighbor to check; every passthrough skipped along the
        way is returned too, so `apply_single` can convert each into an
        in-place `VoidDiagram` once the match fires.

        Parameters
        ----------
        graph : nx.DiGraph
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
            next_id, next_port = self._composition_step(graph, current, port, forward=forward)
            if next_id is None or next_port is None:
                return None, None, crossed
            next_attrs = graph.nodes[next_id]
            # We must make sure that no node involved in a contraction
            # is crossed. For simplicity we will bloc the crossing of the first
            # and second diagram of a contraction which should not be an
            # identity. The only contracted diagrams dealt in a real setting
            # have the first and second diagrams Spiders. We will not treat the
            # case it could be a container itself.
            next_container = graph.nodes[next_attrs.get("container_id")]
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

    def _structural_step(  # ruff: ignore[complex-structure, too-many-branches, too-many-return-statements, too-many-statements, too-many-locals]
        self, graph: nx.DiGraph, node_id: int, port: int, *, forward: bool
    ) -> tuple[int | None, int | None]:
        """Advance one true wire-hop from `(node_id, port)`, using diagram STRUCTURE only.

        `forward=True`: `port` is one of `node_id`'s own OUTPUT ports,
        walking downstream to whoever's input port it truly feeds.
        `forward=False`: the mirror image, walking upstream from one of
        `node_id`'s own INPUT ports.

        Unlike `_chase_identity_chain`/`_composition_step`, this never
        consults graph EDGES -- only `sub_diagram_ids`/`connectivity`
        (composition parents) and `external_input_mapping`/
        `external_output_mapping` (tensor parents), climbing out through
        however many container boundaries `node_id` itself sits at the
        edge of, then descending back down through whatever container
        the landed sibling itself is, until reaching an actual leaf (no
        `container_type` of its own). That makes it accurate even when
        an earlier round of some rule's own `apply_single` has already
        rewired a node's graph edge to skip past structure that is,
        physically, still there -- exactly the situation
        `ChainReductionRule` finds itself in one round after its own
        cross-container fix bypasses whatever it just voided.

        A `ContractedDiagram` parent, or any other structure this can't
        resolve, ends the walk with `(None, None)` -- treated by every
        caller as "not safely traceable", never as "safe to assume
        nothing's there".

        Parameters
        ----------
        graph : nx.DiGraph
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
        # Phase 1: climb out from (node_id, port) until landing on a
        # genuine sibling (still possibly itself a container) or running
        # off the root.
        while True:  # ruff: ignore[too-many-nested-blocks]
            parent_id = graph.nodes[node_id].get("container_id")
            if parent_id is None or parent_id not in graph.nodes:
                return None, None
            parent_attrs = graph.nodes[parent_id]
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

        # Phase 2: descend into whatever container the landed sibling
        # itself is, until reaching an actual leaf.
        while True:
            attrs = graph.nodes[node_id]
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

    def _trace_closure_leftovers(
        self, graph: nx.DiGraph, first_id: int, last_id: int, port: int, known_absorbed: set[int]
    ) -> list[tuple[int, int]] | None:
        """Walk one vanishing output port of `first_id` to `last_id`, purely structurally.

        Used to decide (and, once decided safe, to actually locate) which
        nodes lie on the physical wire a `ChainReductionRule` closure is
        about to erase -- both this match's own `node_ids`/`identity_chain`
        members (already known, passed in via `known_absorbed`, skipped
        without inspection since some may already be gone from `graph` by
        the time this is called for the second, mutating pass) and any
        OTHER node still structurally in the way: an earlier round's own
        leftover `VoidDiagram` (safe -- add it to the result and keep
        going) or anything else (a real, un-absorbed gate -- e.g. a
        `Swap` whose other port carries unrelated live content -- unsafe,
        since cutting this one wire out of it would need shrinking a
        REAL gate, which nothing here does).

        A leftover `VoidDiagram` isn't necessarily single-wire: `CopyRule`
        pads a copied wide leg with a same-shape `VoidDiagram` rather than
        shrinking its container mid-optimization, so a multi-port leftover
        can have an OTHER port pair still carrying an entirely unrelated,
        live wire -- so the port actually walked through is reported
        alongside each leftover, letting the caller remove exactly that
        one port pair rather than the whole node.

        By design, this crosses at most ONE such multi-port (`num_inputs
        > 1`) leftover -- one Swap-like bundle -- per call. A second one
        refuses the whole trace (`None`), exactly like hitting a real,
        un-absorbed gate, rather than being crossed too: a *succession*
        of `Swap` nodes chained together (a small permutation network) would
        otherwise make both this walk and the caller's own bookkeeping
        scale with however many are chained, and nothing else in this
        codebase currently shrinks a run of `Swap` nodes on its own between
        rounds -- so once `match()` backs a chain off past the point
        where a second bundle would need crossing, there's no later
        round that ever gets a *shorter* span to retry. The chosen
        behavior is therefore to leave a state/effect pair connected by
        two or more `Swap` nodes exactly as it is -- unreduced, indefinitely
        -- rather than forcing the full closure through every bundle in
        one match. A single-wire (`1, 1`) leftover carries no other wire
        to protect and costs only one hop to cross, so it's never
        counted against this cap.

        Parameters
        ----------
        graph : nx.DiGraph
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
            next_id, next_port = self._structural_step(graph, current_id, current_port, forward=True)
            if next_id is None or next_port is None:
                return None
            if next_id == last_id:
                return leftovers
            if next_id in known_absorbed:
                current_id, current_port = next_id, next_port
                continue
            next_attrs = graph.nodes[next_id]
            if next_attrs.get("type") != "VoidDiagram":
                return None
            # A leftover only has an established port-for-port identity
            # correspondence (needed for the *next* hop to reinterpret
            # `next_port` as one of its OUTPUT ports) when it's square --
            # e.g. an untouched voided `Swap`/spider bundle. A leftover
            # some earlier round already shrunk asymmetrically no longer
            # has any reliable input-port-to-output-port mapping, so
            # walking "through" it here would just be guessing; refuse.
            next_in = next_attrs.get("num_inputs", 0)
            if next_in != next_attrs.get("num_outputs", 0):
                return None
            # At most one Swap-like bundle (more than one wire) is
            # crossed per call; a second one means this whole span is
            # left unreduced this round (see the docstring above).
            if next_in > 1:
                if bundles_crossed >= 1:
                    return None
                bundles_crossed += 1
            leftovers.append((next_id, next_port))
            current_id, current_port = next_id, next_port
        return None


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
    r"""Fusion rule (f) from [1] Eq. (70) & (71) - Graph-based version.

    Two same-color spiders connected by a wire fuse into a single spider
    with the summed phase and the union of their unconnected ports. Two
    shapes are recognized: the two halves of a `ContractedDiagram`
    (`match_contracted`/`_apply_contracted`), and two same-color spiders
    joined through a composition, possibly across identities/`Swap`
    nodes (`match_terminal`/`_apply_terminal`). `first_id` is always the
    survivor in `apply_single`.

    See :doc:`../../user_guide/rewrite_rules` for the match-level guards
    that keep this from overlapping `IdentityRule`/`CopyRule`.
    """

    # -------------------------------------------------------------------------
    # Match
    # -------------------------------------------------------------------------

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
        matches = []

        for node in sorted(registry.contracted_diagrams):
            attrs = graph.nodes[node]
            first_id = attrs.get("first_id")
            second_id = attrs.get("second_id")

            if first_id is None or second_id is None:
                continue

            first_attrs = graph.nodes[first_id]
            second_attrs = graph.nodes[second_id]

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
        matches = []
        used: set[int] = set()

        candidates = sorted(
            node
            for node in registry.input_states | registry.measurement_nodes
            if graph.nodes[node]["type"] != "VoidDiagram"
        )
        for terminal_id in candidates:
            if terminal_id in used:
                continue
            forward = terminal_id in registry.input_states
            neighbor_id, neighbor_port, chain = self._chase_identity_chain(graph, terminal_id, forward=forward)
            if neighbor_id is None or neighbor_id in used:
                continue

            spider_attrs = graph.nodes[terminal_id]
            neighbor_attrs = graph.nodes[neighbor_id]

            if neighbor_attrs.get("type") != spider_attrs.get("type"):
                continue

            # Skip pairs involving a bare (1, 1) zero-phase identity wire.
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
    def _is_directly_in_contracted(graph: nx.DiGraph, node_id: int) -> bool:
        """Whether `node_id` is directly one of a ContractedDiagram's own halves.

        Returns
        -------
        bool
        """
        parent_id = graph.nodes[node_id].get("container_id")
        return parent_id is not None and graph.nodes[parent_id].get("container_type") == "contracted"

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

    # -------------------------------------------------------------------------
    # Apply
    # -------------------------------------------------------------------------

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

    def _apply_contracted(self, cvzx_graph: CVZXGraph, match: dict) -> None:  # ruff: ignore[too-many-locals, too-many-statements, complex-structure, too-many-branches]
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
        contracted_id = match["contracted_id"]
        first_id = match["first_id"]
        second_id = match["second_id"]
        first_type = match["first_type"]
        second_type = match["second_type"]
        special_case = match["special_case"]

        contracted_attrs = graph.nodes[contracted_id]
        first_attrs = graph.nodes[first_id]
        second_attrs = graph.nodes[second_id]

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
                first_attrs = graph.nodes[first_id]

            node_to_contract_id = graph.nodes[second_id]["sub_diagram_ids"][-1]
            first_phase = first_attrs.get("phase")
            second_phase = graph.nodes[node_to_contract_id].get("phase")

            if first_phase is None or second_phase is None:
                return

            new_phase = first_phase + second_phase
            nx.contracted_nodes(graph, first_id, node_to_contract_id, self_loops=False, copy=False)
            del graph.nodes[first_id]["contraction"]
            graph.nodes[first_id].update({
                "type": fused_type,
                "num_inputs": new_num_inputs,
                "num_outputs": new_num_outputs,
                "phase": new_phase,
                "container_id": parent_container_id,
                "is_root": is_root,
            })

            graph.nodes[second_id]["sub_diagram_ids"].pop(-1)
            graph.nodes[second_id]["sub_diagram_ids"] = [first_id] + graph.nodes[second_id]["sub_diagram_ids"]
            num_output = graph.nodes[second_id]["num_outputs"]
            graph.nodes[second_id]["external_output_mapping"][num_output] = (0, 0)
            graph.nodes[second_id]["external_outputs"].append(num_output)
            graph.nodes[second_id].update({
                "num_inputs": 1,
                "num_outputs": num_output + 1,
                "external_inputs": [0],
                "external_input_mapping": {0: (0, 0)},
            })

            fusion_id = second_id
        else:
            nx.contracted_nodes(graph, first_id, second_id, self_loops=False, copy=False)
            del graph.nodes[first_id]["contraction"]

            new_id = max(graph.nodes) + 1 if graph.nodes else 0
            graph.add_node(
                new_id,
                id=new_id,
                type="TensorDiagram",
                kind="container",
                container_type="tensor",
                phase=None,
                num_inputs=new_num_inputs,
                num_outputs=new_num_outputs,
                container_id=None,
                is_root=is_root,
                sub_diagram_ids=[first_id],
                external_inputs=list(range(new_num_inputs)),
                external_outputs=list(range(new_num_outputs)),
                external_input_mapping={0: (0, 0)},
                external_output_mapping={0: (0, 0)},
            )

            first_phase = first_attrs.get("phase")
            second_phase = second_attrs.get("phase")

            if first_phase is None or second_phase is None:
                return

            new_phase = first_phase + second_phase
            graph.nodes[first_id].update({
                "type": fused_type,
                "num_inputs": new_num_inputs,
                "num_outputs": new_num_outputs,
                "phase": new_phase,
                "container_id": new_id,
                "is_root": is_root,
            })
            fusion_id = new_id

        if parent_container_id is not None and parent_container_id in graph.nodes:
            parent_attrs = graph.nodes[parent_container_id]
            parent_type = parent_attrs.get("container_type")

            if parent_type in {"composition", "tensor"}:
                sub_ids = parent_attrs.get("sub_diagram_ids", [])
                if contracted_id in sub_ids:
                    idx = sub_ids.index(contracted_id)
                    sub_ids[idx] = fusion_id
                    graph.nodes[parent_container_id]["sub_diagram_ids"] = sub_ids
                    graph.nodes[fusion_id]["container_id"] = parent_container_id

            elif parent_type == "contracted":
                if parent_attrs.get("first_id") == contracted_id:
                    parent_attrs["first_id"] = fusion_id
                elif parent_attrs.get("second_id") == contracted_id:
                    parent_attrs["second_id"] = fusion_id
                graph.nodes[fusion_id]["container_id"] = parent_container_id

        else:
            graph.nodes[fusion_id]["is_root"] = True
            graph.nodes[fusion_id]["container_id"] = None

        if contracted_id in graph.nodes:
            graph.remove_node(contracted_id)

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
        first_id = match["first_id"]
        second_id = match["second_id"]
        identity_chain = match.get("identity_chain", [])

        first_attrs = graph.nodes[first_id]
        second_attrs = graph.nodes[second_id]

        # 1. Void every passthrough the chase crossed, in place.
        for pid in identity_chain:
            pid_attrs = graph.nodes[pid]
            pid_attrs["type"] = "VoidDiagram"
            pid_attrs["phase"] = None

        # 2. Sum phases.
        first_phase = first_attrs.get("phase") or ZxPoly({})
        second_phase = second_attrs.get("phase") or ZxPoly({})
        fused_phase = first_phase + second_phase

        spider_type = first_attrs.get("type")

        # 4. Survivor takes the fused value; absorbed becomes a
        #    same-shaped void.
        second_attrs.update({
            "type": spider_type,
            "phase": fused_phase,
        })
        first_attrs.update({
            "type": "VoidDiagram",
            "phase": None,
        })

        # When a contracted diagram absorbs a terminal node
        # the internal arrow is now a regular one
        parent_id = graph.nodes[second_id].get("container_id")
        graph.nodes[parent_id]["passthrough"] = True


class PassthroughRule(RewriteRule):
    """Rewrites a disguised-composition `ContractedDiagram` into `Compose`/`Tensor`/`Swap`.

    A `ContractedDiagram` whose two halves are bare, different-color
    spiders (or one buried as the last slot of a `TensorDiagram` --
    `CopyRule` debris), linked by a single one-way wire with each side's
    kept arity exactly `(1, 1)`, is mathematically just a composition,
    not a genuine partial trace. This only fires once both "outer"
    connections it would repurpose are already proven dead (absent, or a
    `VoidDiagram`) -- see :doc:`../../user_guide/rewrite_rules` for the full
    eligibility argument and the phase-dispatch table.

    Whichever spider survives is placed at arity `(1, 1)` next to a
    same-shaped `Void` filling the dead slot, so `void_input_port` is
    always known deterministically rather than checked against any
    pre-existing wiring.
    """

    @staticmethod
    def _resolve_spider(graph: nx.DiGraph, node_id: int, node_type: str | None) -> tuple[int | None, int | None]:
        """Resolve a `ContractedDiagram` half down to its bare spider, if possible.

        Returns
        -------
        tuple[int | None, int | None]
            `(spider_id, tensor_id)` -- `tensor_id` is `None` unless
            `node_id` was itself a `TensorDiagram` and the spider was
            extracted from its last slot (`CopyRule` debris); either is
            `None` if `node_id` isn't (or doesn't resolve to) a bare
            `QSpider`/`PSpider`.
        """
        if node_type in {"QSpider", "PSpider"}:
            return node_id, None
        if node_type == "TensorDiagram":
            sub_ids = graph.nodes[node_id].get("sub_diagram_ids", [])
            if not sub_ids:
                return None, None
            candidate = sub_ids[-1]
            if graph.nodes[candidate].get("type") not in {"QSpider", "PSpider"}:
                return None, None
            return candidate, node_id
        return None, None

    def _match_shape(  # ruff: ignore[too-many-arguments, too-many-positional-arguments, too-many-return-statements, complex-structure, too-many-locals]
        self,
        graph: nx.DiGraph,
        first_id: int,
        second_id: int,
        first_type: str | None,
        second_type: str | None,
        I1: list[int],  # ruff: ignore[invalid-argument-name]
        J1: list[int],  # ruff: ignore[invalid-argument-name]
    ) -> dict | None:
        """Detect the eligible disguised-composition shape, either link direction.

        Returns
        -------
        dict | None
            `{"a_id", "b_id", "a_tensor_id", "b_tensor_id"}` if eligible
            (see class docstring), else `None`.
        """
        if len(I1) == 1 and len(J1) == 0:
            a_raw_id, a_raw_type = first_id, first_type
            b_raw_id, b_raw_type = second_id, second_type
            expected_connection_type = "I1_I2"
        elif len(J1) == 1 and len(I1) == 0:
            a_raw_id, a_raw_type = second_id, second_type
            b_raw_id, b_raw_type = first_id, first_type
            expected_connection_type = "J2_J1"
        else:
            return None

        a_id, a_tensor_id = self._resolve_spider(graph, a_raw_id, a_raw_type)
        b_id, b_tensor_id = self._resolve_spider(graph, b_raw_id, b_raw_type)
        if a_id is None or b_id is None:
            return None
        # Both sides buried in TensorDiagrams isn't this shape at all --
        # neither side is directly a bare spider, so there is no plain
        # composition link to expose here.
        if a_tensor_id is not None and b_tensor_id is not None:
            return None

        a_type = graph.nodes[a_id].get("type")
        b_type = graph.nodes[b_id].get("type")
        if a_type == b_type:
            # Same-color pairs fuse correctly via ordinary phase addition
            # (FusionRule) -- this shape is only needed for different colors.
            return None

        if not graph.has_edge(a_id, b_id):
            return None
        edge_attrs = graph.get_edge_data(a_id, b_id)
        if edge_attrs.get("edge_type") != "contracted_internal" or edge_attrs.get("connection_type") != (
            expected_connection_type
        ):
            return None

        a_attrs = graph.nodes[a_id]
        b_attrs = graph.nodes[b_id]
        if a_attrs.get("num_inputs") != 1 or a_attrs.get("num_outputs") != 2:  # ruff: ignore[magic-value-comparison]
            return None
        if b_attrs.get("num_inputs") != 2 or b_attrs.get("num_outputs") != 1:  # ruff: ignore[magic-value-comparison]
            return None

        # Eligibility: successor(a) and predecessor(b) -- the two "outer"
        # connections this rewrite would repurpose -- must already be
        # proven dead: absent, or a genuine VoidDiagram. See class docstring.
        consumed_a_port = edge_attrs["source_ports"][0]
        consumed_b_port = edge_attrs["target_ports"][0]
        kept_a_port = 1 - consumed_a_port
        kept_b_port = 1 - consumed_b_port

        def is_void_or_absent(node_id: int | None) -> bool:
            return node_id is None or graph.nodes[node_id].get("type") == "VoidDiagram"

        a_output_neighbor, _ = self._composition_step(graph, a_id, kept_a_port, forward=True)
        b_input_neighbor, _ = self._composition_step(graph, b_id, kept_b_port, forward=False)
        if not (is_void_or_absent(a_output_neighbor) and is_void_or_absent(b_input_neighbor)):
            return None

        return {"a_id": a_id, "b_id": b_id, "a_tensor_id": a_tensor_id, "b_tensor_id": b_tensor_id}

    def match(self, cvzx_graph: CVZXGraph) -> list[dict]:
        """Find `ContractedDiagram` nodes eligible for the disguised-composition rewrite.

        Returns
        -------
        list[dict]
            Matches for `apply_single`.
        """
        graph = cvzx_graph.graph
        registry = cvzx_graph.registry
        matches = []

        for node in sorted(registry.contracted_diagrams):
            attrs = graph.nodes[node]
            first_id = attrs.get("first_id")
            second_id = attrs.get("second_id")
            if first_id is None or second_id is None:
                continue

            first_type = graph.nodes[first_id].get("type")
            second_type = graph.nodes[second_id].get("type")
            J1 = attrs.get("J1", [])  # ruff: ignore[non-lowercase-variable-in-function]
            I1 = attrs.get("I1", [])  # ruff: ignore[non-lowercase-variable-in-function]

            shape_info = self._match_shape(graph, first_id, second_id, first_type, second_type, I1, J1)
            if shape_info is None:
                continue

            match = {"contracted_id": node}
            match.update(shape_info)
            matches.append(match)

        return matches

    @staticmethod
    def _recompute_tensor_bookkeeping(graph: nx.DiGraph, tensor_id: int) -> None:
        """Recompute a `TensorDiagram` container's own arity/port mappings.

        Call after splicing a new entry into `sub_diagram_ids` whose own
        arity differs from what it replaced -- mirrors `_add_tensor_node`'s
        own concatenation formula (each slot's ports, in order), just
        reading each slot's *current* graph attrs instead of a live
        `Diagram` object.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        tensor_id : int
            The `TensorDiagram` container node whose `sub_diagram_ids` was
            just spliced.
        """
        attrs = graph.nodes[tensor_id]
        sub_ids = attrs.get("sub_diagram_ids", [])
        external_input_mapping = {}
        input_offset = 0
        for idx, sub_id in enumerate(sub_ids):
            sub_num_inputs = graph.nodes[sub_id].get("num_inputs", 0)
            for internal_port in range(sub_num_inputs):
                external_input_mapping[input_offset + internal_port] = (idx, internal_port)
            input_offset += sub_num_inputs

        external_output_mapping = {}
        output_offset = 0
        for idx, sub_id in enumerate(sub_ids):
            sub_num_outputs = graph.nodes[sub_id].get("num_outputs", 0)
            for internal_port in range(sub_num_outputs):
                external_output_mapping[output_offset + internal_port] = (idx, internal_port)
            output_offset += sub_num_outputs

        attrs.update({
            "num_inputs": input_offset,
            "num_outputs": output_offset,
            "external_inputs": list(range(input_offset)),
            "external_outputs": list(range(output_offset)),
            "external_input_mapping": external_input_mapping,
            "external_output_mapping": external_output_mapping,
        })

    def apply_single(self, cvzx_graph: CVZXGraph, match: dict) -> None:  # ruff: ignore[too-many-locals, complex-structure, too-many-statements]
        """Replace an eligible disguised-composition `ContractedDiagram` with `Compose`/`Tensor`/`Swap`.

        Parameters
        ----------
        cvzx_graph : CVZXGraph
            The graph to modify.
        match : dict
            Match from `match`.
        """
        graph = cvzx_graph.graph
        contracted_id = match["contracted_id"]
        a_id = match["a_id"]
        b_id = match["b_id"]
        a_tensor_id = match["a_tensor_id"]
        b_tensor_id = match["b_tensor_id"]

        a_phase = graph.nodes[a_id].get("phase")
        b_phase = graph.nodes[b_id].get("phase")
        if a_phase is None or b_phase is None:
            return

        a_zero = bool(a_phase.is_zero)
        b_zero = bool(b_phase.is_zero)

        if not a_zero and not b_zero:
            # Neither spider is droppable. The literal `Tensor`/`Swap`/`Tensor`
            # construction this shape would otherwise dispatch to needs a
            # verified, non-lossy port routing that isn't in place yet --
            # leave the ContractedDiagram untouched rather than risk
            # silently discarding information.
            return

        contracted_attrs = graph.nodes[contracted_id]
        parent_container_id = contracted_attrs.get("container_id")
        is_root = contracted_attrs.get("is_root", False)

        # Shape 2 (one side extracted from a `TensorDiagram`): the freshly
        # built top-level node lands *inside* that tensor, not in
        # `contracted_id`'s own outer slot -- `tensor_container_id` itself
        # is what later takes over `contracted_id`'s slot (see the
        # placement step at the end of this method).
        tensor_container_id = a_tensor_id if a_tensor_id is not None else b_tensor_id
        top_container_id = tensor_container_id if tensor_container_id is not None else parent_container_id
        top_is_root = False if tensor_container_id is not None else is_root

        edge_attrs = graph.get_edge_data(a_id, b_id)
        consumed_a_port = edge_attrs["source_ports"][0]
        consumed_b_port = edge_attrs["target_ports"][0]
        kept_a_port = 1 - consumed_a_port
        kept_b_port = 1 - consumed_b_port

        # Capture every pre-existing outer connection on a kept port before
        # anything is removed/deleted -- `a`'s own input (its only one,
        # always kept since J1 is empty) and `b`'s own output (its only
        # one, always kept since J2 is empty) never move; `a`'s kept output
        # and `b`'s kept input are the ones a new outer edge gets pointed
        # at below.
        a_input_neighbor, a_input_neighbor_port = self._composition_step(graph, a_id, 0, forward=False)
        a_output_neighbor, a_output_neighbor_port = self._composition_step(graph, a_id, kept_a_port, forward=True)
        b_input_neighbor, b_input_neighbor_port = self._composition_step(graph, b_id, kept_b_port, forward=False)
        b_output_neighbor, b_output_neighbor_port = self._composition_step(graph, b_id, 0, forward=True)

        graph.remove_edge(a_id, b_id)

        def redirect(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
            old_u: int | None,
            old_v: int | None,
            new_u: int | None,
            new_v: int | None,
            source_port: int | None,
            target_port: int | None,
        ) -> None:
            if old_u is None or old_v is None or new_u is None or new_v is None:
                return
            if source_port is None or target_port is None:
                return
            graph.remove_edge(old_u, old_v)
            graph.add_edge(
                new_u,
                new_v,
                source_ports=[source_port],
                target_ports=[target_port],
                edge_type="composition",
                internal=False,
                connection_type=None,
            )

        if a_zero and b_zero:
            # Case: both vanish -- a bare (unmarked) Swap.
            top_id = max(graph.nodes) + 1 if graph.nodes else 0
            graph.add_node(
                top_id,
                id=top_id,
                type="Swap",
                kind="proper",
                phase=None,
                feedforward=None,
                measurement_ids=None,
                param_measurement_map={},
                num_inputs=2,
                num_outputs=2,
                container_id=top_container_id,
                is_root=top_is_root,
                external_inputs=[0, 1],
                external_outputs=[0, 1],
                void_input_port=None,
            )
            redirect(a_input_neighbor, a_id, a_input_neighbor, top_id, a_input_neighbor_port, 0)
            redirect(b_input_neighbor, b_id, b_input_neighbor, top_id, b_input_neighbor_port, 1)
            redirect(a_id, a_output_neighbor, top_id, a_output_neighbor, 1, a_output_neighbor_port)
            redirect(b_id, b_output_neighbor, top_id, b_output_neighbor, 0, b_output_neighbor_port)
            graph.remove_node(a_id)
            graph.remove_node(b_id)

        elif b_zero:
            # Case: `a` survives -- Compose([Tensor([a(1,1), Void(1,1)]), Swap()]).
            a_type = graph.nodes[a_id].get("type")
            a_phase_value = graph.nodes[a_id].get("phase")
            a_reduced_id = max(graph.nodes) + 1 if graph.nodes else 0
            void_id = a_reduced_id + 1
            swap_id = void_id + 1
            tensor_id = swap_id + 1
            compose_id = tensor_id + 1

            graph.add_node(
                a_reduced_id,
                id=a_reduced_id,
                type=a_type,
                kind="proper",
                phase=a_phase_value,
                num_inputs=1,
                num_outputs=1,
                container_id=tensor_id,
                external_inputs=[0],
                external_outputs=[0],
            )
            graph.add_node(
                void_id,
                id=void_id,
                type="VoidDiagram",
                kind="proper",
                phase=None,
                num_inputs=1,
                num_outputs=1,
                container_id=tensor_id,
                external_inputs=[0],
                external_outputs=[0],
            )
            graph.add_node(
                swap_id,
                id=swap_id,
                type="Swap",
                kind="proper",
                phase=None,
                feedforward=None,
                measurement_ids=None,
                param_measurement_map={},
                num_inputs=2,
                num_outputs=2,
                container_id=compose_id,
                external_inputs=[0, 1],
                external_outputs=[0, 1],
                # Visualization-only: `_draw_tensor` draws its
                # `sub_diagram_ids` top-to-bottom but numbers the returned
                # `output_positions` bottom-to-top (it prepends each
                # sub-diagram's position onto the accumulator), so as seen
                # by this `Swap`, drawing-index 0 is actually
                # `sub_diagram_ids[-1]` (void) and index 1 is
                # `sub_diagram_ids[0]` (a_reduced). The void slot is
                # therefore at input position 0 here, not 1 -- the
                # underlying graph wiring (`compose_id`'s
                # `external_output_mapping`) is unaffected and already
                # routes the real value correctly.
                void_input_port=0,
            )
            graph.add_node(
                tensor_id,
                id=tensor_id,
                type="TensorDiagram",
                kind="container",
                container_type="tensor",
                phase=None,
                num_inputs=2,
                num_outputs=2,
                container_id=compose_id,
                is_root=False,
                sub_diagram_ids=[a_reduced_id, void_id],
                external_inputs=[0, 1],
                external_outputs=[0, 1],
                external_input_mapping={0: (0, 0), 1: (1, 0)},
                external_output_mapping={0: (0, 0), 1: (1, 0)},
            )
            graph.add_node(
                compose_id,
                id=compose_id,
                type="CompositionDiagram",
                kind="container",
                container_type="composition",
                phase=None,
                num_inputs=2,
                num_outputs=2,
                container_id=top_container_id,
                is_root=top_is_root,
                sub_diagram_ids=[tensor_id, swap_id],
                connectivity={0: {0: 0, 1: 1}},
                external_inputs=[0, 1],
                external_outputs=[0, 1],
                external_input_mapping={0: (0, 0), 1: (0, 1)},
                external_output_mapping={0: (1, 0), 1: (1, 1)},
            )

            redirect(a_input_neighbor, a_id, a_input_neighbor, a_reduced_id, a_input_neighbor_port, 0)
            graph.add_edge(
                a_reduced_id,
                swap_id,
                source_ports=[0],
                target_ports=[0],
                edge_type="composition",
                internal=False,
                connection_type=None,
            )
            graph.add_edge(
                void_id,
                swap_id,
                source_ports=[0],
                target_ports=[1],
                edge_type="composition",
                internal=False,
                connection_type=None,
            )
            redirect(b_input_neighbor, b_id, b_input_neighbor, void_id, b_input_neighbor_port, 0)
            redirect(a_id, a_output_neighbor, swap_id, a_output_neighbor, 1, a_output_neighbor_port)
            redirect(b_id, b_output_neighbor, swap_id, b_output_neighbor, 0, b_output_neighbor_port)

            graph.remove_node(a_id)
            graph.remove_node(b_id)
            top_id = compose_id

        else:
            # Case: `b` survives -- Compose([Swap(), Tensor([b(1,1), Void(1,1)])]).
            b_type = graph.nodes[b_id].get("type")
            b_phase_value = graph.nodes[b_id].get("phase")
            swap_id = max(graph.nodes) + 1 if graph.nodes else 0
            b_reduced_id = swap_id + 1
            void_id = b_reduced_id + 1
            tensor_id = void_id + 1
            compose_id = tensor_id + 1

            graph.add_node(
                swap_id,
                id=swap_id,
                type="Swap",
                kind="proper",
                phase=None,
                feedforward=None,
                measurement_ids=None,
                param_measurement_map={},
                num_inputs=2,
                num_outputs=2,
                container_id=compose_id,
                external_inputs=[0, 1],
                external_outputs=[0, 1],
                # Logically, swap input 1 (predecessor(b), proven void by
                # eligibility) is always the dead leg here -- deterministic,
                # not an outward check. Visualization-only caveat: unlike
                # the "`a` survives" branch (where the Swap's predecessor is
                # this rule's own freshly-built Tensor, so which drawn
                # *position* is void is fully known), here the Swap is
                # first in its Compose, so its predecessor is whatever
                # externally precedes the original ContractedDiagram --
                # `_draw_tensor`'s bottom-to-top-numbered/top-to-bottom-drawn
                # convention means the drawn position of the void slot
                # depends on that external structure's own layout, which
                # isn't knowable here. `1` is this method's best-effort
                # guess (the logically-void port); `_draw_swap`/`_draw_fourier`
                # guard with `if input_positions[i]:` so a wrong guess never
                # crashes, only occasionally mis-hides the wrong diagonal.
                void_input_port=1,
            )
            graph.add_node(
                b_reduced_id,
                id=b_reduced_id,
                type=b_type,
                kind="proper",
                phase=b_phase_value,
                num_inputs=1,
                num_outputs=1,
                container_id=tensor_id,
                external_inputs=[0],
                external_outputs=[0],
            )
            graph.add_node(
                void_id,
                id=void_id,
                type="VoidDiagram",
                kind="proper",
                phase=None,
                num_inputs=1,
                num_outputs=1,
                container_id=tensor_id,
                external_inputs=[0],
                external_outputs=[0],
            )
            graph.add_node(
                tensor_id,
                id=tensor_id,
                type="TensorDiagram",
                kind="container",
                container_type="tensor",
                phase=None,
                num_inputs=2,
                num_outputs=2,
                container_id=compose_id,
                is_root=False,
                sub_diagram_ids=[b_reduced_id, void_id],
                external_inputs=[0, 1],
                external_outputs=[0, 1],
                external_input_mapping={0: (0, 0), 1: (1, 0)},
                external_output_mapping={0: (0, 0), 1: (1, 0)},
            )
            graph.add_node(
                compose_id,
                id=compose_id,
                type="CompositionDiagram",
                kind="container",
                container_type="composition",
                phase=None,
                num_inputs=2,
                num_outputs=2,
                container_id=top_container_id,
                is_root=top_is_root,
                sub_diagram_ids=[swap_id, tensor_id],
                connectivity={0: {0: 0, 1: 1}},
                external_inputs=[0, 1],
                external_outputs=[0, 1],
                external_input_mapping={0: (0, 0), 1: (0, 1)},
                external_output_mapping={0: (1, 0), 1: (1, 1)},
            )

            # Crossing: swap input0 (predecessor(a), real -- the internal
            # link's value, since a is being eliminated and forces its own
            # legs equal) produces swap output1; swap input1
            # (predecessor(b), void by eligibility) produces swap output0.
            # `b_reduced` -- which needs the real internal-link value, not
            # its own (void) predecessor(b) -- must take swap output1, and
            # `void_id` takes the already-dead swap output0.
            graph.add_edge(
                swap_id,
                b_reduced_id,
                source_ports=[1],
                target_ports=[0],
                edge_type="composition",
                internal=False,
                connection_type=None,
            )
            graph.add_edge(
                swap_id,
                void_id,
                source_ports=[0],
                target_ports=[0],
                edge_type="composition",
                internal=False,
                connection_type=None,
            )
            redirect(a_input_neighbor, a_id, a_input_neighbor, swap_id, a_input_neighbor_port, 0)
            redirect(b_input_neighbor, b_id, b_input_neighbor, swap_id, b_input_neighbor_port, 1)
            redirect(a_id, a_output_neighbor, void_id, a_output_neighbor, 0, a_output_neighbor_port)
            redirect(b_id, b_output_neighbor, b_reduced_id, b_output_neighbor, 0, b_output_neighbor_port)

            graph.remove_node(a_id)
            graph.remove_node(b_id)
            top_id = compose_id

        # Placement: either splice into the enclosing TensorDiagram that
        # `node_to_contract` (the surviving/absorbed half not passed in
        # directly) came from, or replace `contracted_id` directly in its
        # own parent -- never both.
        tensor_container_id = a_tensor_id if a_tensor_id is not None else b_tensor_id
        if tensor_container_id is not None:
            sub_ids = graph.nodes[tensor_container_id]["sub_diagram_ids"]
            node_to_contract_id = b_id if a_tensor_id is not None else a_id
            sub_ids[sub_ids.index(node_to_contract_id)] = top_id
            graph.nodes[top_id]["container_id"] = tensor_container_id
            self._recompute_tensor_bookkeeping(graph, tensor_container_id)
            top_id = tensor_container_id

        if parent_container_id is not None and parent_container_id in graph.nodes:
            self._replace_in_parent(graph, parent_container_id, contracted_id, top_id)
        else:
            graph.nodes[top_id]["is_root"] = True
            graph.nodes[top_id]["container_id"] = None

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
        matches: list[dict] = []
        used_nodes: set[int] = set()

        # Sort for deterministic match order (dict/node iteration order
        # isn't guaranteed to reflect any particular scan order).
        candidates = sorted(
            node
            for node in graph.nodes
            if graph.nodes[node].get("kind") in {"proper", "compact"}
            and self.get_gate_info(graph, node)[0] is not None
            and not self._is_directly_in_contracted(graph, node)
        )

        for start_id in candidates:
            if start_id in used_nodes:
                continue

            gate_type, value, gate_info = self.get_gate_info(graph, start_id)
            if gate_type is None or not self._is_chain_start(graph, start_id, gate_type, value, gate_info):
                continue

            values = [value]
            node_ids = [start_id]
            identity_chain: list[int] = []
            identity_chain_checkpoints = [0]
            gate_info_history = [gate_info]
            last_gate_info = gate_info
            cur_id, cur_type, cur_value, cur_gate_info = start_id, gate_type, value, gate_info

            while True:
                neighbor_id, chain = self._find_full_neighbor(graph, cur_id, forward=True)
                if (
                    neighbor_id is None
                    or neighbor_id in used_nodes
                    or neighbor_id in node_ids
                    or any(cid in used_nodes for cid in chain)
                ):
                    break
                next_type, next_value, next_gate_info = self.get_gate_info(graph, neighbor_id)
                if not self.can_chain(cur_type, cur_value, cur_gate_info, next_type, next_value, next_gate_info):
                    break
                # `can_chain` only returns True when `next_type == cur_type`
                # (its very first check), and `cur_type` is never None here.
                assert next_type is not None  # ruff: ignore[assert]

                for cid in chain:
                    if cid not in identity_chain:
                        identity_chain.append(cid)
                values.append(next_value)
                node_ids.append(neighbor_id)
                identity_chain_checkpoints.append(len(identity_chain))
                gate_info_history.append(next_gate_info)
                last_gate_info = next_gate_info

                # {F, Finv} chains are only of size 2.
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

                # A chain whose first and last members' own OUTPUT counts
                # differ folds down to a `first_node` with a genuinely
                # different arity -- `apply_single` only knows how to
                # cascade that safely through the rest of the diagram when
                # it's a FULL closure to zero outputs (a state fusing all
                # the way through to an effect, becoming a bare scalar):
                # every node structurally sitting on the wire that's about
                # to vanish -- this chain's own members, once absorbed,
                # included -- has to be nothing but `VoidDiagram`
                # placeholders, since cutting that wire out of a REAL
                # multi-wire gate (a `Swap` whose other port carries
                # unrelated, still-live content) isn't something anything
                # here does. `_trace_closure_leftovers` checks that by
                # walking the true diagram structure (not graph edges,
                # which an earlier round may already have short-circuited
                # past exactly this kind of leftover) -- and when the full
                # chain isn't safe, this backs off one member at a time
                # until it finds a length that's either shape-preserving
                # or fully safe, rather than either forcing an unsafe
                # collapse or discarding a perfectly good shorter
                # reduction outright. Any OTHER first/last mismatch
                # (shrinking to some other nonzero count, or growing) is
                # left exactly as it was before this check existed.
                while len(node_ids) >= 2:  # ruff: ignore[magic-value-comparison]
                    first_id, candidate_last_id = node_ids[0], node_ids[-1]
                    first_out = graph.nodes[first_id].get("num_outputs", 0)
                    last_out = gate_info_history[-1].get("num_outputs", 0) if gate_info_history[-1] else 0
                    if last_out != 0 or first_out == last_out:
                        break
                    known_absorbed = set(node_ids)
                    if all(
                        self._trace_closure_leftovers(graph, first_id, candidate_last_id, port, known_absorbed)
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

            # Classic case: every member of the chain sits, contiguously
            # and in order, in ONE flat CompositionDiagram's own
            # `sub_diagram_ids` -- exactly what the old same-container
            # scan required. Fully collapsible, no leftover placeholder.
            # A crossed identity/Swap means members aren't actually
            # adjacent there even when they share an immediate parent --
            # see `CopyRule._resolve_containers` for the same reasoning.
            #
            # A node's own `container_id` attribute can't be trusted in
            # isolation: `to_graph` assigns a single shared graph node to
            # a diagram object reused at two different tree positions,
            # and that node's `container_id` reflects only whichever
            # position was processed last -- even though the OTHER
            # position's container still legitimately lists it in its own
            # `sub_diagram_ids`. So instead of anchoring on
            # `first_id`'s own `container_id`, try every distinct
            # `container_id` seen among the chain's members as a
            # candidate parent, and accept the first candidate whose OWN
            # `sub_diagram_ids` contiguously contains every node_id --
            # regardless of what individual members' possibly-aliased
            # `container_id` attributes say.
            same_parent = False
            resolved_parent: int | None = None
            if not identity_chain:
                candidate_parents = []
                seen_candidates = set()
                for nid in node_ids:
                    candidate = graph.nodes[nid].get("container_id")
                    if candidate is not None and candidate not in seen_candidates:
                        seen_candidates.add(candidate)
                        candidate_parents.append(candidate)
                for candidate in candidate_parents:
                    if candidate not in graph.nodes:
                        continue
                    if graph.nodes[candidate].get("container_type") != "composition":
                        continue
                    sub_ids = graph.nodes[candidate].get("sub_diagram_ids", [])
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

    def _is_chain_start(
        self,
        graph: nx.DiGraph,
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
        prev_id, _ = self._find_full_neighbor(graph, node_id, forward=False)
        if prev_id is None:
            return True
        prev_type, prev_value, prev_gate_info = self.get_gate_info(graph, prev_id)
        if prev_type is None:
            return True
        return not self.can_chain(prev_type, prev_value, prev_gate_info, gate_type, value, gate_info)

    def _find_full_neighbor(  # ruff: ignore[complex-structure, too-many-return-statements]
        self, graph: nx.DiGraph, node_id: int, *, forward: bool
    ) -> tuple[int | None, list[int]]:
        """Find `node_id`'s single full composition neighbor, chasing every port.

        Generalizes `_chase_identity_chain` (single-wire) to a
        potentially-multi-port gate: every one of `node_id`'s own ports
        on the relevant side is chased independently -- each stepping
        over any identity/`Swap` passthrough directly in its own path --
        and the result only counts as a match when all of them agree on
        landing at the SAME neighbor, at DISTINCT ports of its own,
        together covering that neighbor's entire arity on the other
        side. That's exactly what "two gates are fully, serially
        composed with nothing else feeding either of them" means at the
        port level, and it's what a `sub_diagram_ids`-adjacent pair in a
        flat CompositionDiagram is always guaranteed to satisfy, so this
        reduces to the old same-container adjacency check as a special
        case while also working across container boundaries.

        Parameters
        ----------
        graph : nx.DiGraph
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
        attrs = graph.nodes[node_id]
        num_ports = attrs.get("num_outputs" if forward else "num_inputs", 0)
        if not num_ports:
            return None, []

        neighbor_id: int | None = None
        seen_neighbor_ports: set[int] = set()
        combined_chain: list[int] = []
        seen_chain: set[int] = set()

        for port in range(num_ports):
            nxt, nxt_port, chain = self._chase_identity_chain(graph, node_id, forward=forward, start_port=port)
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

        if neighbor_id is None:
            return None, []
        # A node that is directly one of a ContractedDiagram's own two
        # halves (its `first_id`/`second_id`) can have SOME of its ports
        # wired internally by the contraction itself (I1/I2/J1/J2) rather
        # than to this diagram's own external interface -- merging such a
        # node into a chain would silently change the ContractedDiagram's
        # own external arity in a way nothing here accounts for. Mirrors
        # `TerminalAbsorptionRule.match`'s identical exclusion.
        if self._is_directly_in_contracted(graph, neighbor_id):
            return None, []
        neighbor_attrs = graph.nodes[neighbor_id]
        neighbor_arity = neighbor_attrs.get("num_inputs" if forward else "num_outputs", 0)
        if neighbor_arity != num_ports:
            return None, []
        return neighbor_id, combined_chain

    @staticmethod
    def _is_directly_in_contracted(graph: nx.DiGraph, node_id: int) -> bool:
        """Check whether `node_id` is directly a ContractedDiagram's own half.

        Returns
        -------
        bool
            True if `node_id`'s own immediate parent is a `ContractedDiagram`
            (i.e. `node_id` is literally that container's `first_id` or
            `second_id`), False otherwise.
        """
        parent_id = graph.nodes[node_id].get("container_id")
        return parent_id is not None and graph.nodes[parent_id].get("container_type") == "contracted"

    def _overwrite_node_for_reduced_gate(self, graph: nx.DiGraph, node_id: int, reduced_gate: dict) -> None:
        """Overwrite `node_id`'s attrs with the reduced gate's, in place.

        Same shape as `_update_node_for_reduced_gate` but without the
        `container_id` bookkeeping: this method is only called when the
        reduced arity matches the node's own, so nothing about the node's
        position or port counts changes.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        node_id : int
            The node to overwrite.
        reduced_gate : dict
            The dict returned by `reduce_chain`.
        """
        self._update_node_for_reduced_gate(graph, node_id, reduced_gate)

    @staticmethod
    def _reset_to_identity(graph: nx.DiGraph, node_id: int) -> None:
        """Reset a node in place to a zero-phase identity spider.

        The node keeps its own slot, container, and port counts. Its type
        becomes the spider color that identity should have:

        - A raw `QSpider`/`PSpider` keeps its own color.
        - Every other gate type becomes `QSpider`.

        Any `feedforward`/`measurement_ids`/`param_measurement_map`
        attributes are cleared.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        node_id : int
            The node to reset.
        """
        attrs = graph.nodes[node_id]
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

    def _apply_single_splice(self, graph: nx.DiGraph, match: dict, reduced_gate: dict) -> None:  # ruff: ignore[too-many-locals, too-many-statements, too-many-branches, complex-structure]
        """Splice-and-propagate path for arity-changing chain reductions.

        Used only for `QSpider`/`PSpider` chains whose reduced arity differs
        from the first member's own arity. This is the previous
        `apply_single` body, renamed; it contracts the chain members into
        the survivor and updates the surrounding container's bookkeeping.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to modify.
        match : dict
            Match containing chain information.
        reduced_gate : dict
            The dict returned by `reduce_chain`.
        """
        node_ids = match["node_ids"]

        for passthrough_id in match.get("identity_chain", []):
            passthrough_attrs = graph.nodes[passthrough_id]
            passthrough_attrs["type"] = "VoidDiagram"
            passthrough_attrs["phase"] = None

        first_node = node_ids[0]

        if match.get("same_parent", True):
            container_id = match["container_id"]
            container_attrs = graph.nodes[container_id]
            sub_ids = container_attrs.get("sub_diagram_ids", [])

            for i in range(1, len(node_ids), 1):
                nx.contracted_nodes(graph, first_node, node_ids[i], self_loops=False, copy=False)
                if "contraction" in graph.nodes[first_node]:
                    del graph.nodes[first_node]["contraction"]

            self._update_node_for_reduced_gate(graph, first_node, reduced_gate)

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

            graph.nodes[container_id]["sub_diagram_ids"] = new_sub_ids
            graph.nodes[container_id]["sub_diagram_indices"] = list(range(len(new_sub_ids)))
            self._update_connectivity_after_reduction(graph, container_id, removed_connectivity)

            if len(new_sub_ids) == 1:
                self._flatten_container(graph, container_id)
            return

        # Cross-container / gapped arity-changing Q/P case -- preserved from
        # the previous implementation, since the arity change has to cascade.
        last_node = node_ids[-1]
        absorbed_info = [
            (
                nid,
                graph.nodes[nid].get("container_id"),
                graph.nodes[nid].get("num_inputs", 0),
                graph.nodes[nid].get("num_outputs", 0),
            )
            for nid in node_ids[1:]
        ]
        first_parent_id = graph.nodes[first_node].get("container_id")
        first_old_shape = (
            graph.nodes[first_node].get("num_inputs", 0),
            graph.nodes[first_node].get("num_outputs", 0),
        )

        is_closure = (
            graph.nodes[first_node].get("num_outputs", 0) != 0 and graph.nodes[last_node].get("num_outputs", 0) == 0
        )
        closure_leftovers: list[tuple[int, int]] = []
        if is_closure:
            known_absorbed = set(node_ids)
            seen_leftovers: set[tuple[int, int]] = set()
            for port in range(graph.nodes[first_node].get("num_outputs", 0)):
                found = self._trace_closure_leftovers(graph, first_node, last_node, port, known_absorbed)
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
            if leftover_id not in graph.nodes:
                continue
            for port in sorted(set(leftover_ports_by_id[leftover_id]), reverse=True):
                leftover_attrs = graph.nodes[leftover_id]
                old_in = leftover_attrs.get("num_inputs", 0)
                old_out = leftover_attrs.get("num_outputs", 0)
                if port >= old_out:
                    continue
                if not self._remove_external_port(graph, leftover_id, is_input=False, port=port):
                    continue
                new_out = old_out - 1
                output_remap = {i: (i if i < port else i - 1) for i in range(old_out) if i != port}
                self._propagate_arity_to_parent(graph, leftover_id, old_in, old_in, old_out, new_out, {}, output_remap)

        first_stale_out = [
            target
            for _, target, edge_attrs in graph.out_edges(first_node, data=True)
            if edge_attrs.get("edge_type") == "composition"
        ]
        far_edges = [
            (target, dict(edge_attrs))
            for _, target, edge_attrs in graph.out_edges(last_node, data=True)
            if edge_attrs.get("edge_type") == "composition"
        ]

        graph.remove_nodes_from(node_ids[1:])

        for target in first_stale_out:
            if graph.has_edge(first_node, target):
                graph.remove_edge(first_node, target)
        for far_target, edge_attrs in far_edges:
            if far_target in graph.nodes:
                graph.add_edge(first_node, far_target, **edge_attrs)

        self._update_node_for_reduced_gate(graph, first_node, reduced_gate)

        for nid, parent_id, n_in, n_out in absorbed_info:
            if parent_id is None or parent_id not in graph.nodes:
                continue
            if is_closure:
                void_id = self._install_void_placeholder(graph, parent_id, nid, num_inputs=0, num_outputs=0)
                if n_in != 0 or n_out != 0:
                    self._propagate_arity_to_parent(graph, void_id, n_in, 0, n_out, 0, {}, {})
            else:
                self._install_void_placeholder(graph, parent_id, nid, num_inputs=n_in, num_outputs=n_out)

        new_shape = (graph.nodes[first_node].get("num_inputs", 0), graph.nodes[first_node].get("num_outputs", 0))
        if new_shape != first_old_shape and first_parent_id is not None and first_parent_id in graph.nodes:
            first_parent_type = graph.nodes[first_parent_id].get("container_type")
            if first_parent_type == "tensor":
                arity_change = self._recompute_tensor_arity(graph, first_parent_id)
                self._propagate_arity_to_parent(graph, first_parent_id, *arity_change)
            else:
                old_in, old_out = first_old_shape
                new_in, new_out = new_shape
                in_remap = {p: p for p in range(old_in)} if new_in == old_in else {}
                out_remap = {p: p for p in range(old_out)} if new_out == old_out else {}
                self._propagate_arity_to_parent(
                    graph, first_node, old_in, new_in, old_out, new_out, in_remap, out_remap
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
        gate_type = match["gate_type"]
        values = match["values"]
        node_ids = match["node_ids"]
        gate_info = match["gate_info"]

        reduced_gate = self.reduce_chain(gate_type, values, gate_info)
        if reduced_gate is None:
            msg = f"reduce_chain: unrecognized gate_type {gate_type!r}"
            raise RuleApplicationError(msg)

        first_id = node_ids[0]
        first_attrs = graph.nodes[first_id]
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
            self._apply_single_splice(graph, match, reduced_gate)
            return

        # Common case: overwrite the survivor, replace the rest with
        # zero-phase (1, 1) identity spiders -- a genuine identity at that
        # one arity, safe to leave for IdentityRule to prune later.
        self._overwrite_node_for_reduced_gate(graph, first_id, reduced_gate)
        for extra_id in node_ids[1:]:
            self._reset_to_identity(graph, extra_id)

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
        # Terminal guard: neither side of a chain may be a state or effect.
        # `Q`/`P` carry their own arity in `gate_info`; every other gate type
        # this rule recognizes is intrinsically a (1,1) or (2,2) gate, never
        # a terminal -- so the guard only needs to fire for `Q`/`P`.
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

        # P-Spider: sum phases
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

        # R: sum angles
        if gate_type == "R":
            total = simplify_reduced_value(sum(values))
            if total == 0:
                return id_q
            return {"type": "PhaseRotationGate", "theta": total}

        # BS: sum angles
        if gate_type == "BS":
            total = simplify_reduced_value(sum(values))
            if total == 0:
                return id_q2
            return {"type": "BeamsplitterGate", "theta": total}

        # Sq: multiply
        if gate_type == "Sq":
            total = 1.0
            for v in values:
                total *= v
            total = simplify_reduced_value(total)
            if total == 1.0:  # ruff: ignore[float-equality-comparison]
                return id_q
            return {"type": "SqueezingGate", "tau": total}

        # D: sum
        if gate_type == "D":
            total = simplify_reduced_value(sum(values))
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
            total = simplify_reduced_value(sum(values))
            if total == 0:
                return id_q2
            return {"type": "ControlledZGate", "gain": total}

        # ControlledSumGate: sum gains (only if control/target same)
        if gate_type == "CSUM":
            if gate_info is None:
                msg = "get_gate_info returned no gate_info for a 'CSUM' node."
                raise RuleApplicationError(msg)
            # Check that all ControlledSumGate gates have same control and target
            # We need to get this from the graph
            total = simplify_reduced_value(sum(values))
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

    def _update_node_for_reduced_gate(self, graph: nx.DiGraph, node_id: int, reduced_gate: dict) -> None:
        """Update a node with reduced gate attributes.

        `node_id`'s own `container_id` is deliberately left untouched --
        it doesn't move, only its type/phase/arity change (mirroring
        `TerminalAbsorptionRule`/`CopyRule`'s own in-place survivors).
        `_flatten_container`, called separately once the chain's other
        members are spliced away, is what updates it on the rare
        occasion `node_id` ends up as its container's sole remaining
        child.
        """
        gate_type = reduced_gate["type"]

        # Base attributes
        updates = {
            "type": gate_type,
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
        else:
            self._refresh_composition_external_mappings(graph, container_id)


class TerminalAbsorptionRule(RewriteRule):
    r"""Terminal absorption rule - Graph-based version.

    Absorbs a gate adjacent to a QSpider/PSpider terminal (arity (1,0)
    effect or (0,1) state) into the terminal's own phase, eliminating the
    gate. Four sub-cases -- rotation, squeezing, cross-color discard,
    displacement -- each fold the gate's parameter into the terminal's
    phase by a different closed-form rule; see
    :doc:`../../user_guide/rewrite_rules` for each case's exact formula and
    validity condition.

    Only rotation absorption is exact for any physical state; the other
    three treat the terminal as an idealized (infinite squeezing)
    eigenstate and only run when `assume_infinite_squeezing=True` (the
    default, False, restricts matching to rotation absorption alone).

    Neither the terminal nor the gate may be directly one of a
    `ContractedDiagram`'s own two halves -- that shape belongs to
    `PassthroughRule` instead.

    References
    ----------
    :cite:`nagayoshi2024zx`, Eq. (239a)-(239e).
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

        Uses the same candidate-and-chase scan as `CopyRule.match` (see
        :doc:`../../dev_guide/rewrite_engine`) -- terminal nodes, chased past
        identities/`Swap` nodes to the real neighboring gate -- and, like
        `PassthroughRule`, skips a pair outright when either endpoint's
        immediate parent is a `ContractedDiagram`.

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
        ordered_matches: list[tuple[int, int, dict]] = []
        used_nodes: set[int] = set()

        candidates = sorted(node for node in registry.input_states | registry.measurement_nodes)

        for terminal_id in candidates:
            if terminal_id in used_nodes:
                continue

            forward = graph.nodes[terminal_id].get("num_outputs") == 1
            gate_id, _, identity_chain = self._chase_identity_chain(graph, terminal_id, forward=forward)
            if gate_id is None or gate_id in used_nodes or any(cid in used_nodes for cid in identity_chain):
                continue
            # At most one `Swap` is voided per match -- crossing a second
            # one is left for a later round (same "resolve one swap per
            # run" policy as `ChainReductionRule`'s closure cap), so a
            # single absorption never sweeps through an entire run of
            # chained swaps at once.
            if sum(1 for cid in identity_chain if graph.nodes[cid].get("type") == "Swap") > 1:
                continue

            first_id, second_id = (terminal_id, gate_id) if forward else (gate_id, terminal_id)

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
            match["identity_chain"] = identity_chain

            first_parent = graph.nodes[first_id].get("container_id")
            second_parent = graph.nodes[second_id].get("container_id")
            # A crossed identity/Swap means the pair isn't actually
            # adjacent in any flat CompositionDiagram's own list, even
            # when they happen to share an immediate parent -- see
            # `CopyRule._resolve_containers` for the same reasoning --
            # so the adjacent-index bookkeeping below is only used when
            # the chase found nothing in between.
            if not identity_chain and first_parent == second_parent and first_parent is not None:
                match["container_id"] = first_parent
                sub_ids = graph.nodes[first_parent].get("sub_diagram_ids", [])
                if first_id in sub_ids and second_id in sub_ids:
                    match["indices"] = sorted([sub_ids.index(first_id), sub_ids.index(second_id)])
            else:
                match["container_id"] = ("cross", first_id, second_id)

            ordered_matches.append((first_id, second_id, match))
            used_nodes.add(first_id)
            used_nodes.add(second_id)
            used_nodes.update(identity_chain)

        # Sort by the (first_id, second_id) pair the chase discovered --
        # not by `match["node_ids"]`, whose order reflects `keep_id`
        # (always the terminal) rather than graph scan order -- for
        # deterministic match order (dict/node iteration order isn't
        # guaranteed to reflect any particular scan order), mirroring
        # `CopyRule`'s own sort.
        ordered_matches.sort(key=operator.itemgetter(0, 1))
        return [entry[2] for entry in ordered_matches]

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
                # `node_ids` is always [terminal_id, gate_id] -- the
                # terminal is `apply_single`'s `keep_id` in both cases,
                # never the gate: only the terminal's arity is
                # guaranteed unchanged by folding (its phase changes, its
                # shape doesn't), so it's the only one of the two whose
                # own immediate parent can safely stay untouched when the
                # pair turns out to be cross-container.
                result["node_ids"] = [second_id, first_id]
                return result

        return None

    def _try_absorb(  # ruff: ignore[complex-structure]
        self, *, terminal_attrs: dict, gate_attrs: dict
    ) -> dict | None:
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
            is_q_spider = terminal_type == "QSpider"
            new_phase = self._squeeze_absorb(phase, gate_attrs.get("phase"), is_q_spider)
        elif (
            gate_type in {"QSpider", "PSpider"}
            and gate_type != terminal_type
            and gate_attrs.get("num_inputs") == 1
            and gate_attrs.get("num_outputs") == 1
            and self.assume_infinite_squeezing
        ):
            new_phase = phase  # Cross-color discard: unchanged.
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
        :cite:`nagayoshi2024zx`, Eq. (82)-(83).
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
            return False  # Symbolic theta: can't determine, assume safe.
        return math.isclose(abs(theta_val) % math.pi, math.pi / 2)

    @staticmethod
    def _reset_to_identity(graph: nx.DiGraph, node_id: int) -> None:
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
        graph : nx.DiGraph
            The graph to modify.
        node_id : int
            The node to reset.
        """
        attrs = graph.nodes[node_id]
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
        terminal_id, gate_id = match["node_ids"]
        if terminal_id not in graph or gate_id not in graph:
            return

        keep_id = terminal_id
        absorb_id = gate_id

        # 1. Update the terminal in place: its phase becomes the fold result.
        # Its type, arity, container, and slot are all unchanged.
        graph.nodes[keep_id].update({
            "type": match["result_type"],
            "phase": match["result_phase"],
        })

        # 2. Reset the gate in place to a zero-phase identity spider of its
        # own color and arity.
        self._reset_to_identity(graph, absorb_id)

        # 3. Reset every identity passthrough the chase crossed, in
        # place, to a zero-phase identity spider of its own color and arity.
        for passthrough_id in match.get("identity_chain", []):
            if passthrough_id not in graph or graph.nodes[passthrough_id]["type"] == "Swap":
                continue
            self._reset_to_identity(graph, passthrough_id)


class CopyRule(RewriteRule):
    r"""Copy rule - Graph-based version.

    Copies a spider with arity (0,1) or (1,0) through an adjacent,
    opposite-color spider with arity (1,n) or (n,1), producing n copies
    of the narrow spider in a `TensorDiagram` in place of the
    composition -- valid only when the *copied* spider's phase is in
    R1[X] (degree <= 1); the disappearing spider's own phase can be any
    polynomial. See :doc:`../../user_guide/rewrite_rules` for the four
    color/arity cases and their exact result shapes.

    The two spiders need not be direct siblings of one flat
    `CompositionDiagram` -- `match()` finds them via the graph's own
    "composition" edges, which already resolve to fully-resolved leaf
    nodes regardless of container nesting, and transparently sees
    through any run of identity spiders in the path (recorded in the
    match and voided in `apply_single`).
    """

    def match(self, cvzx_graph: CVZXGraph) -> list[dict]:
        """Find all composition-adjacent copy-able patterns in the graph.

        Recognizes both same-parent pairs (direct entries of one flat
        `CompositionDiagram`) and the cross-container shapes this rule
        knows how to restructure -- see :doc:`../../dev_guide/rewrite_engine`
        for exactly which container combinations are supported. Also
        transparently chases through any run of identity spiders directly
        in the path (`_chase_identity_chain`), recording them so
        `apply_single` can void them.

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
            neighbor_id, _, identity_chain = self._chase_identity_chain(graph, copy_id, forward=forward)
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
            else:
                self._refresh_composition_external_mappings(graph, container_id)
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
        #     `VoidDiagram` leaves away regardless of whether their slot is
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

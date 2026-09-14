"""Type-1/type-2 tensor normalization for CV ZX diagrams.

This module implements a single public entry point, `normalize_diagram`,
that rewrites an arbitrary (compact-form) `Diagram` into a canonical
`CompositionDiagram` of alternating "stages":

    - a *type-2* stage is a `TensorDiagram` containing exactly one
      "wide" leaf (any generator touching more than one mode -- a
      2-mode gate, `Swap`, etc.) plus one identity wire (`QSpider(1,1,0)`)
      per every other currently-open mode;
    - a *type-1* stage is a `TensorDiagram` where every row is a
      `CompositionDiagram` chaining together whatever 1-mode gates (or
      `VoidDiagram`/state/effect leaves) act on that row between the
      type-2 touches on either side, or a bare identity wire on a row
      nothing happens on.

Several rewrite rules (`ChainReductionRule`, `TerminalAbsorptionRule`) only
match two leaves that are directly adjacent elements of one flat
`CompositionDiagram`; normalizing first makes that check sufficient
everywhere, instead of requiring every such rule to grow `CopyRule`-style
cross-container matching logic. This is only designed to run on
*compact*-form diagrams (before `expand_two_mode_gates`) -- if the graph
contains any `ContractedDiagram`, it conservatively returns the input
diagram unchanged. See the docs' dev guide ("Type-1/type-2 stage
normalization") for the full algorithm walkthrough and design rationale.
"""

from __future__ import annotations

import heapq
import logging
import math
from typing import TYPE_CHECKING

from cvzx.backends.nx.graph import GateRegister, get_root_node, reconstruct_proper_node, to_graph
from cvzx.ir.base import CompositionDiagram, Diagram, QSpider, TensorDiagram, ZxPoly

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

__all__ = ["normalize_diagram"]

logger = logging.getLogger(__name__)

_ZERO = ZxPoly({})

# A token identifying a specific open wire: either the diagram's j-th
# external input (`("EXT_IN", j)`), or the p-th output port of leaf
# `leaf_id` (`(leaf_id, p)`). Never mutated for a row that's merely
# passed through a filler identity wire -- see module docstring, step 3.
_Token = tuple


def _make_identity_wire() -> Diagram:
    """Build a fresh 1-mode identity spider to pad an untouched row.

    Returns
    -------
    Diagram
        A brand-new `QSpider(1, 1, 0)` (each call mints its own id, so
        callers never need to worry about accidentally sharing one
        identity leaf across two different rows/stages).
    """
    return QSpider(1, 1, _ZERO)


def _leaf_diagram(graph: nx.DiGraph, leaf_id: int, reg: GateRegister) -> Diagram:
    """Rebuild a leaf's `Diagram` object from its graph attributes.

    `to_graph()` deliberately strips the cached `"diagram"` object off
    every node before returning (so the graph doesn't hold stale
    references once rewrite rules start mutating node attributes in
    place) -- exactly the same situation `to_diagram()` itself is
    always in, so this reuses its own leaf-reconstruction helper rather
    than inventing a second way to rebuild a leaf from its attributes.

    Returns
    -------
    Diagram
        The rebuilt leaf.
    """
    return reconstruct_proper_node(graph, leaf_id, reg)


def _resolve_input(graph: nx.DiGraph, node_id: int, port: int) -> tuple[int, int]:
    """Resolve a (possibly nested-container) input port down to its leaf.

    Mirrors `nx_graph.find_node_by_external_input`, but works directly
    off already-built graph node attributes (so it also correctly
    resolves *through* a `CompositionDiagram` nested inside a
    `TensorDiagram` row, exactly like the fixed `external_input_mapping`
    now populated for composition containers by `_add_composition_node`).

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to resolve within.
    node_id : int
        Container (or leaf) node id to resolve the port against.
    port : int
        The port index, local to `node_id`.

    Returns
    -------
    tuple[int, int]
        `(leaf_id, leaf_port)` of the leaf that ultimately owns this
        input port.
    """
    attrs = graph.nodes[node_id]
    if attrs.get("kind") in {"proper", "compact"}:
        return node_id, port

    container_type = attrs.get("container_type")
    if container_type == "tensor":
        sub_idx, internal_port = attrs["external_input_mapping"][port]
        sub_id = attrs["sub_diagram_ids"][sub_idx]
        return _resolve_input(graph, sub_id, internal_port)
    if container_type == "composition":
        sub_id = attrs["sub_diagram_ids"][0]
        return _resolve_input(graph, sub_id, port)
    # "contracted" (or anything else unexpected): treated as opaque --
    # callers guard against a ContractedDiagram ever reaching here at
    # all (see `normalize_diagram`'s pre-flight check), so this branch
    # is defensive only.
    return node_id, port


def _resolve_output(graph: nx.DiGraph, node_id: int, port: int) -> tuple[int, int]:
    """Resolve a (possibly nested-container) output port down to its leaf.

    See `_resolve_input` -- this is its output-side mirror.

    Returns
    -------
    tuple[int, int]
        `(leaf_id, leaf_port)` of the leaf that ultimately owns this
        output port.
    """
    attrs = graph.nodes[node_id]
    if attrs.get("kind") in {"proper", "compact"}:
        return node_id, port

    container_type = attrs.get("container_type")
    if container_type == "tensor":
        sub_idx, internal_port = attrs["external_output_mapping"][port]
        sub_id = attrs["sub_diagram_ids"][sub_idx]
        return _resolve_output(graph, sub_id, internal_port)
    if container_type == "composition":
        sub_id = attrs["sub_diagram_ids"][-1]
        return _resolve_output(graph, sub_id, port)
    return node_id, port


def _natural_key(graph: nx.DiGraph, leaf_id: int) -> tuple[int, ...]:
    """A leaf's row position among its original `TensorDiagram` siblings, root-to-leaf.

    Walks up the `container_id` chain from `leaf_id`, and at every
    `TensorDiagram` container crossed, records this leaf's (or its
    ancestor's) index among its immediate siblings there --
    `CompositionDiagram` containers are skipped entirely, since a
    stage index there encodes *when* something happens, not *where*
    (which row) it lives. The resulting tuple, read root-to-leaf, is
    directly comparable (plain tuple order) across any two leaves of
    the same diagram and reproduces the input's own left-to-right,
    row-by-row reading order.

    Used to decide where a newly-born state (a leaf with no input --
    e.g. an ancilla `PSpider`) belongs among the rows already active,
    instead of always appending it after every other row regardless of
    where it actually sits in the source diagram.

    Returns
    -------
    tuple[int, ...]
        The leaf's row-index key, root-to-leaf.
    """
    keys: list[int] = []
    current = leaf_id
    while True:
        attrs = graph.nodes[current]
        parent_id = attrs.get("container_id")
        if parent_id is None:
            break
        parent_attrs = graph.nodes[parent_id]
        if parent_attrs.get("container_type") == "tensor":
            keys.append(parent_attrs["sub_diagram_ids"].index(current))
        current = parent_id
    keys.reverse()
    return tuple(keys)


def _token_natural_key(graph: nx.DiGraph, token: _Token, root_id: int) -> tuple[int, ...]:
    """`_natural_key` for a `_Token` (an `EXT_IN` marker or a `(leaf, port)` pair).

    An `EXT_IN` token's key is that of the leaf that actually owns it
    (via `_resolve_input`) -- the same leaf `_build_pred_map` would
    resolve it to -- so `EXT_IN` tokens and leaf tokens compare
    correctly against each other in one shared key space.

    Returns
    -------
    tuple[int, ...]
        The owning leaf's `_natural_key`.
    """
    if token[0] == "EXT_IN":
        leaf_id, _port = _resolve_input(graph, root_id, token[1])
        return _natural_key(graph, leaf_id)
    leaf_id, _port = token
    return _natural_key(graph, leaf_id)


def _natural_insertion_index(
    graph: nx.DiGraph,
    root_id: int,
    active: list[_Token],
    new_leaf_id: int,
) -> int:
    """Where a newly-born leaf (no inputs) belongs among `active`'s current tokens.

    `active` is maintained in natural-reading-order throughout
    `normalize_diagram` (see `_natural_key`), so this is a first-match
    linear scan for the first existing token whose key exceeds the new
    leaf's -- i.e. an insertion-sort position, not a fresh sort.

    Returns
    -------
    int
        The index into `active` the new leaf's row should be inserted
        at.
    """
    new_key = _natural_key(graph, new_leaf_id)
    for i, token in enumerate(active):
        if new_key < _token_natural_key(graph, token, root_id):
            return i
    return len(active)


def _is_wide(attrs: dict) -> bool:
    """Classify a leaf as "wide" (touches more than one mode).

    Returns
    -------
    bool
        True if the leaf touches more than one mode.
    """
    return bool(max(attrs.get("num_inputs", 0), attrs.get("num_outputs", 0)) > 1)


def _build_pred_map(
    graph: nx.DiGraph,
    leaves: list[int],
    root_id: int,
    num_inputs: int,
) -> dict[tuple[int, int], _Token]:
    """Build the leaf-level predecessor wire map.

    Returns
    -------
    dict
        `pred_of[(leaf, in_port)]` is the token feeding that input:
        either `("EXT_IN", j)` for the diagram's j-th external input, or
        `(other_leaf, other_out_port)` for an internal wire.
    """
    leaf_set = set(leaves)
    pred_of: dict[tuple[int, int], _Token] = {}

    for u, v, data in graph.edges(data=True):
        if data.get("edge_type") != "composition":
            continue
        if u not in leaf_set or v not in leaf_set:
            continue
        for src_port, tgt_port in zip(data["source_ports"], data["target_ports"], strict=True):
            pred_of[v, tgt_port] = (u, src_port)

    for j in range(num_inputs):
        leaf_id, leaf_port = _resolve_input(graph, root_id, j)
        pred_of[leaf_id, leaf_port] = ("EXT_IN", j)

    return pred_of


def _is_bare_identity(attrs: dict) -> bool:
    """True for a graph node representing a plain 1-mode identity wire.

    `QSpider(1, 1, ZxPoly({}))` specifically -- the exact shape
    `_make_identity_wire` mints for filler wires. An identity leaf that
    already sits in the *input* diagram (as opposed to one this module
    inserts itself) carries zero information: it is semantically
    indistinguishable from "nothing touched this wire here", and is
    elided the same way (see `_strip_identity_leaves`) so that a run of
    nothing-but-wiring never has to be represented as its own type-1
    stage.

    Returns
    -------
    bool
        True if `attrs` describes a plain 1-mode identity wire.
    """
    return (
        attrs.get("type") == "QSpider"
        and attrs.get("num_inputs") == 1
        and attrs.get("num_outputs") == 1
        and attrs.get("phase") == _ZERO
    )


def _strip_identity_leaves(
    graph: nx.DiGraph,
    leaves: list[int],
    pred_of: dict[tuple[int, int], _Token],
) -> tuple[list[int], dict[tuple[int, int], _Token], Callable[[_Token], _Token]]:
    """Elide every bare identity leaf already present in the input diagram.

    Without this, an input diagram that spells "nothing happens on this
    wire" with an explicit `QSpider(1, 1, ZxPoly({}))` (rather than
    simply not touching that mode) can produce a whole type-1 stage
    whose every row is such a leaf -- structurally indistinguishable
    from pure filler, but *not* caught by the empty-run elision in
    `normalize_diagram` (that check only fires when a run has literally
    no leaves at all, not when its leaves all happen to be no-ops).

    Every elided leaf's output is rewritten to point straight at
    whatever *actually* produced the wire, chasing back through any run
    of such leaves (and leaving `EXT_IN` untouched, since it can never
    itself be a leaf id).

    Parameters
    ----------
    graph : nx.DiGraph
        The graph built by `to_graph()`.
    leaves : list[int]
        Every proper/compact leaf id, wide or narrow, identity or not.
    pred_of : dict
        As returned by `_build_pred_map`, before any elision.

    Returns
    -------
    tuple[list[int], dict, Callable]
        The filtered `leaves` list (identity ids removed), the
        rewritten `pred_of` map (no longer keyed on, or valued with, an
        elided leaf), and a `flatten` function applying the same chase
        to any other token -- used so the diagram's own external-output
        resolution agrees with `pred_of` on what counts as a leaf's
        true predecessor (see `normalize_diagram`'s use for
        `desired_final_tokens`).
    """
    identity_ids = {leaf for leaf in leaves if _is_bare_identity(graph.nodes[leaf])}

    def flatten(token: _Token) -> _Token:
        while token[0] in identity_ids:
            token = pred_of[token[0], 0]
        return token

    new_pred_of = {k: flatten(v) for k, v in pred_of.items() if k[0] not in identity_ids}
    new_leaves = [leaf for leaf in leaves if leaf not in identity_ids]
    return new_leaves, new_pred_of, flatten


def _assign_stages(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements, too-many-locals]
    graph: nx.DiGraph,
    leaves: list[int],
    pred_of: dict[tuple[int, int], _Token],
) -> dict[int, int]:
    """Greedily assign each leaf a "micro-layer" index (module docstring, step 1).

    Returns
    -------
    dict[int, int]
        `stage_of[leaf_id]`: the micro-layer this leaf lands in.

    Raises
    ------
    ValueError
        If the leaf-level wire graph is not a DAG (a cycle was detected).
    """
    leaf_attrs = {leaf: graph.nodes[leaf] for leaf in leaves}
    leaf_predecessors: dict[int, set[int]] = {}
    in_degree: dict[int, int] = {}
    for leaf in leaves:
        preds = set()
        for port in range(leaf_attrs[leaf].get("num_inputs", 0)):
            src = pred_of.get((leaf, port))
            if src is not None and src[0] != "EXT_IN":
                preds.add(src[0])
        leaf_predecessors[leaf] = preds
        in_degree[leaf] = len(preds)

    successors: dict[int, list[int]] = {leaf: [] for leaf in leaves}
    for leaf, preds in leaf_predecessors.items():
        for p in preds:
            successors[p].append(leaf)

    # A min-heap keeps this an O(L log L) topological sort. The earlier
    # version re-sorted the whole `ready` list (and did an O(k) `pop(0)`
    # shift) on every iteration, which made a circuit of L leaves cost
    # O(L^2 log L) instead -- invisible at the dozens-of-gates scale this
    # library currently targets, but needlessly quadratic all the same.
    ready = [leaf for leaf in leaves if in_degree[leaf] == 0]
    heapq.heapify(ready)
    topo_order: list[int] = []
    remaining_in_degree = dict(in_degree)
    while ready:
        leaf = heapq.heappop(ready)
        topo_order.append(leaf)
        for nxt in successors[leaf]:
            remaining_in_degree[nxt] -= 1
            if remaining_in_degree[nxt] == 0:
                heapq.heappush(ready, nxt)
    if len(topo_order) != len(leaves):
        msg = "normalize_diagram: leaf-level wire graph is not a DAG (cycle detected)."
        raise ValueError(msg)

    stage_of: dict[int, int] = {}
    stage_kind: list[str | None] = []  # None (unclaimed), "wide", or "narrow"
    # `wide_frontier` is the smallest stage index that *might* still be
    # free for a wide leaf to claim exclusively -- narrow leaves never
    # move it, since they can never block or unblock a wide claim except
    # at the one exact index they land on. Without it, a layer of many
    # mutually-independent wide leaves (e.g. a row of parallel 2-mode
    # gates across disjoint mode pairs -- a perfectly ordinary circuit
    # shape, not a contrived one) would each re-scan from their own
    # `min_stage` through every already-claimed index before finding a
    # free one, i.e. O(L) work per leaf and O(L^2) overall. The frontier
    # is only ever advanced past indices this loop has already resolved
    # one way or another, so across the whole call it moves at most
    # `len(stage_kind)` steps in total -- amortized O(1) per leaf.
    wide_frontier = 0
    for leaf in topo_order:
        preds = leaf_predecessors[leaf]
        min_stage = 0 if not preds else 1 + max(stage_of[p] for p in preds)
        wide = _is_wide(leaf_attrs[leaf])
        if wide:
            s = max(min_stage, wide_frontier)
            while True:
                while s >= len(stage_kind):
                    stage_kind.append(None)
                if stage_kind[s] is None:
                    stage_kind[s] = "wide"
                    break
                s += 1
            stage_of[leaf] = s
            if s == wide_frontier:
                while wide_frontier < len(stage_kind) and stage_kind[wide_frontier] is not None:
                    wide_frontier += 1
        else:
            s = min_stage
            while True:
                while s >= len(stage_kind):
                    stage_kind.append(None)
                if stage_kind[s] in {None, "narrow"}:
                    stage_kind[s] = "narrow"
                    break
                s += 1
            stage_of[leaf] = s

    return stage_of


class _Row:
    """One in-progress row (mode) of a stage under construction.

    Attributes
    ----------
    content : list[int]
        Leaf ids chained onto this row so far, in causal order.
    token_in : _Token | None
        The token this row started with (None if the row was born mid-
        stage, i.e. its first leaf is a state).
    token : _Token | None
        The token this row currently ends with (None once the row has
        been closed by an effect).
    """

    __slots__ = ("content", "token", "token_in")

    def __init__(self, token: _Token | None) -> None:
        self.content: list[int] = []
        self.token_in = token
        self.token = token


def _finalize_row(graph: nx.DiGraph, row: _Row, reg: GateRegister) -> Diagram:
    """Turn one row's accumulated leaf chain into a single `Diagram`.

    Returns
    -------
    Diagram
        A bare identity wire if `row` is empty, the row's sole leaf if
        it has exactly one, or a `CompositionDiagram` chaining all of
        them otherwise.
    """
    if not row.content:
        return _make_identity_wire()
    if len(row.content) == 1:
        return _leaf_diagram(graph, row.content[0], reg)
    return CompositionDiagram([_leaf_diagram(graph, leaf, reg) for leaf in row.content])


def _build_narrow_rows(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    graph: nx.DiGraph,
    root_id: int,
    active: list[_Token],
    run_leaves: list[int],
    stage_of: dict[int, int],
    pred_of: dict[tuple[int, int], _Token],
) -> list[_Row]:
    """Build one type-1 (narrow) stage's rows (before final ordering/output).

    Parameters
    ----------
    graph : nx.DiGraph
        The graph built by `to_graph()`.
    root_id : int
        The graph's root node id, needed to resolve `EXT_IN` tokens'
        natural row position (see `_token_natural_key`) when placing a
        newly-born row.
    active : list[_Token]
        The tokens open at the start of this stage, in order.
    run_leaves : list[int]
        The narrow leaves assigned to this run (any order).
    stage_of : dict[int, int]
        Micro-layer index per leaf, used only to order leaves that
        land on the same row.
    pred_of : dict
        As returned by `_build_pred_map`.

    Returns
    -------
    list[_Row]
        One `_Row` per position `active` started with, plus one per
        state-leaf that opened a brand new row, in construction order
        (surviving rows have `token is not None`; closed rows have
        `token is None`).
    """
    rows = [_Row(token) for token in active]
    token_pos = {token: i for i, token in enumerate(active)}

    for leaf in sorted(run_leaves, key=lambda leaf: (stage_of[leaf], leaf)):
        attrs = graph.nodes[leaf]
        n_in = attrs.get("num_inputs", 0)
        n_out = attrs.get("num_outputs", 0)
        if n_in == 0 and n_out == 1:
            # A newly-born state (e.g. an ancilla) has no predecessor
            # token to anchor its position to, so insert its row at the
            # natural reading-order position among the rows already
            # open -- not always at the end -- so a state born
            # mid-circuit stays where the source diagram actually
            # places it, both now and in every later stage that
            # inherits this row's position.
            open_tokens = [row.token for row in rows if row.token is not None]
            insert_among_open = _natural_insertion_index(graph, root_id, open_tokens, leaf)
            actual_at = len(rows)
            open_seen = 0
            for i, row in enumerate(rows):
                if row.token is None:
                    continue
                if open_seen == insert_among_open:
                    actual_at = i
                    break
                open_seen += 1

            new_row = _Row(None)
            new_row.content.append(leaf)
            new_row.token = (leaf, 0)
            rows.insert(actual_at, new_row)
            for token, pos in list(token_pos.items()):
                if pos >= actual_at:
                    token_pos[token] = pos + 1
            token_pos[leaf, 0] = actual_at
        elif n_in == 1 and n_out == 0:
            src = pred_of[leaf, 0]
            idx = token_pos.pop(src)
            rows[idx].content.append(leaf)
            rows[idx].token = None
        elif n_in == 1 and n_out == 1:
            src = pred_of[leaf, 0]
            idx = token_pos.pop(src)
            rows[idx].content.append(leaf)
            new_token = (leaf, 0)
            rows[idx].token = new_token
            token_pos[new_token] = idx
        # n_in == 0 and n_out == 0: a genuinely portless leaf. Nothing to
        # wire up; it contributes no row (defensive -- should not occur
        # for any real generator).

    return rows


def _absorb_final_permutation(
    stage_records: list[tuple[list[Diagram], list[_Token], list[_Token]]],
    desired_final_tokens: list[_Token],
) -> bool:
    """Reorder the last stage's own elements to already match `desired_final_tokens`.

    Every element `normalize_diagram` ever builds contributes an equal,
    fixed number of input and output ports (a filler or one run's
    chained row is 1-in-1-out; the one wide leaf of a type-2 stage is
    whatever fixed shape it is) -- reordering *which* element sits where
    within one stage is therefore always semantically free (it changes
    nothing but port numbering), so the diagram's own external-output
    order can be met by reordering the last stage directly instead of
    appending a whole new trailing stage whose sole job is that
    permutation. Appending such a stage is exactly what this function
    exists to avoid: it would always be a type-1 stage whose every row
    is a bare identity wire, which must never exist as its own stage
    (see module docstring).

    Mutates `stage_records[-1]` in place on success.

    Parameters
    ----------
    stage_records : list
        The stages built so far, as `(elements, input_tokens,
        new_active)` triples; only the last entry is touched.
    desired_final_tokens : list[_Token]
        The diagram's true external-output tokens, in its own declared
        output order.

    Returns
    -------
    bool
        True if the last stage's own elements were successfully
        reordered (or already matched). False only if some element's
        own multi-port output block would have to be split apart to
        match `desired_final_tokens` -- a leaf's own ports are born
        adjacent and stay adjacent through every row-position-
        preserving step this module performs, so this should not occur
        for any realistic circuit; the caller falls back to a genuine
        trailing permutation stage in that case.
    """
    elements, input_tokens, new_active = stage_records[-1]
    if sorted(new_active, key=repr) != sorted(desired_final_tokens, key=repr):
        return False  # different token set entirely -- nothing to reorder here

    # Re-derive each element's own contiguous slice of `input_tokens`/
    # `new_active`: every stage-building path appends both lists in the
    # same element order, each element contributing exactly its own
    # `num_inputs`/`num_outputs` worth of consecutive entries.
    blocks: list[tuple[Diagram, list[_Token], list[_Token]]] = []
    in_pos = 0
    out_pos = 0
    for element in elements:
        n_in, n_out = element.num_inputs, element.num_outputs
        blocks.append((element, input_tokens[in_pos : in_pos + n_in], new_active[out_pos : out_pos + n_out]))
        in_pos += n_in
        out_pos += n_out

    target_pos = {token: i for i, token in enumerate(desired_final_tokens)}
    for _element, _in_toks, out_toks in blocks:
        if not out_toks:
            # A 0-output (effect) element contributes nothing to
            # `desired_final_tokens`, so its position can't be verified
            # against it -- fall back to a genuine trailing permutation
            # stage instead, same as the multi-port-splitting case below.
            return False
        positions = [target_pos[t] for t in out_toks]
        if positions != list(range(positions[0], positions[0] + len(positions))):
            return False

    blocks.sort(key=lambda block: target_pos[block[2][0]])
    stage_records[-1] = (
        [block[0] for block in blocks],
        [token for block in blocks for token in block[1]],
        [token for block in blocks for token in block[2]],
    )
    return True


def normalize_diagram(  # ruff: ignore[complex-structure, too-many-branches, too-many-locals, too-many-statements, too-many-return-statements]
    diagram: Diagram,
) -> Diagram:
    """Rewrite `diagram` into alternating type-1/type-2 stages.

    See the module docstring for the full algorithm description.

    Parameters
    ----------
    diagram : Diagram
        A compact-form diagram (i.e. one that has not yet been passed
        through `expand_two_mode_gates`).

    Returns
    -------
    Diagram
        A `CompositionDiagram` of alternating type-1 (`TensorDiagram`
        of per-row `CompositionDiagram` rows of 1-mode gates/wiring) and
        type-2 (`TensorDiagram` of exactly one multi-mode gate plus
        identity wiring) stages, semantically equal to `diagram`. If
        `diagram` contains no leaves at all, or contains a
        `ContractedDiagram` (out of scope -- see module docstring),
        `diagram` is returned unchanged.

    Raises
    ------
    TypeError
        If `diagram` is not a `Diagram` instance.
    """
    if not isinstance(diagram, Diagram):
        msg = f"normalize_diagram() expects a Diagram, got {type(diagram).__name__}."
        raise TypeError(msg)

    cvzx_graph = to_graph(diagram)
    graph = cvzx_graph.graph
    reg = cvzx_graph.registry
    root_id = get_root_node(cvzx_graph)
    if root_id is None:
        logger.debug("normalize_diagram: empty diagram, nothing to do")
        return diagram

    for _n, attrs in graph.nodes(data=True):
        if attrs.get("container_type") == "contracted":
            logger.debug("normalize_diagram: contains a ContractedDiagram, left unchanged")
            return diagram

    leaves = [n for n, attrs in graph.nodes(data=True) if attrs.get("kind") in {"proper", "compact"}]
    if not leaves:
        return diagram

    num_inputs = diagram.num_inputs
    num_outputs = diagram.num_outputs

    pred_of = _build_pred_map(graph, leaves, root_id, num_inputs)

    # Elide any bare identity leaf the *input* diagram already contains
    # (as opposed to filler this module inserts itself) -- see
    # `_strip_identity_leaves`. A diagram that turns out to be nothing
    # but such wiring, once elided, is a no-op the same way an already-
    # leafless diagram is.
    leaves, pred_of, flatten_output_token = _strip_identity_leaves(graph, leaves, pred_of)
    if not leaves:
        return diagram

    stage_of = _assign_stages(graph, leaves, pred_of)
    leaf_attrs = {leaf: graph.nodes[leaf] for leaf in leaves}

    wide_leaves_sorted = sorted(
        (leaf for leaf in leaves if _is_wide(leaf_attrs[leaf])), key=lambda leaf: stage_of[leaf]
    )
    narrow_leaves_sorted = sorted(
        (leaf for leaf in leaves if not _is_wide(leaf_attrs[leaf])), key=lambda leaf: stage_of[leaf]
    )

    boundaries = [-1, *[stage_of[w] for w in wide_leaves_sorted], math.inf]
    runs: list[list[int]] = [[] for _ in range(len(wide_leaves_sorted) + 1)]
    for leaf in narrow_leaves_sorted:
        s = stage_of[leaf]
        for k in range(len(boundaries) - 1):
            if boundaries[k] < s < boundaries[k + 1]:
                runs[k].append(leaf)
                break

    # Each entry: (elements, input_tokens, new_active) for one stage,
    # in final left-to-right order.
    stage_records: list[tuple[list[Diagram], list[_Token], list[_Token]]] = []
    active: list[_Token] = [("EXT_IN", j) for j in range(num_inputs)]

    def emit_narrow(run_leaves: list[int]) -> None:
        nonlocal active
        if not run_leaves:
            # A run with nothing in it is pure pass-through on every row
            # it spans (every row would finalize to a bare identity wire)
            # -- exactly the "elided entirely when empty on every row"
            # case from the module docstring. Skip it rather than
            # emitting an inert all-identity `TensorDiagram` stage: with
            # nothing to consume, `_build_narrow_rows` would otherwise
            # hand back one filler per currently-open row even when
            # `active` is empty-of-content-but-not-empty-of-rows (e.g. a
            # diagram that's nothing but one wide leaf), which is both
            # unnecessary structure and breaks the invariant that
            # normalizing an already-unreducible diagram is a true no-op
            # (several tests assert `optimize()` leaves such a diagram
            # byte-for-byte untouched).
            return
        rows = _build_narrow_rows(graph, root_id, active, run_leaves, stage_of, pred_of)

        elements = [_finalize_row(graph, row, reg) for row in rows]
        input_tokens = [row.token_in for row in rows if row.token_in is not None]
        new_active = [row.token for row in rows if row.token is not None]

        stage_records.append((elements, input_tokens, new_active))
        active = new_active

    def emit_wide(leaf: int) -> None:  # ruff: ignore[complex-structure]
        nonlocal active
        attrs = leaf_attrs[leaf]
        n_in = attrs.get("num_inputs", 0)
        n_out = attrs.get("num_outputs", 0)
        wide_input_tokens = [pred_of[leaf, p] for p in range(n_in)]

        elements: list[Diagram] = []
        input_tokens: list[_Token] = []
        new_active: list[_Token] = []

        if wide_input_tokens:
            # Place the wide leaf's own output block *in* the position
            # its inputs occupied, leaving every other row exactly
            # where it already was -- rather than always moving the
            # wide leaf's outputs to the front and compacting
            # everything else after it. The latter needlessly
            # scrambled row order (and so stage `connectivity`) even
            # for circuits with nothing to actually reorder -- e.g. a
            # layer of mutually-independent 2-mode gates on disjoint
            # mode pairs, where every row should simply stay put
            # across every stage. A wide leaf's own inputs are always
            # contiguous in `active`: `active` is kept in natural
            # reading-order throughout (see `_natural_key`), and a
            # leaf's own ports are siblings born adjacent in the
            # source diagram, so nothing before this point could have
            # split them apart.
            token_pos = {token: i for i, token in enumerate(active)}
            consumed_positions = sorted(token_pos[t] for t in wide_input_tokens)
            first, last = consumed_positions[0], consumed_positions[-1]
            contiguous = consumed_positions == list(range(first, last + 1))
            consumed_set = set(consumed_positions)
            if contiguous:
                for i, token in enumerate(active):
                    if i == first:
                        elements.append(_leaf_diagram(graph, leaf, reg))
                        input_tokens.extend(wide_input_tokens)
                        new_active.extend((leaf, p) for p in range(n_out))
                    elif i not in consumed_set:
                        elements.append(_make_identity_wire())
                        input_tokens.append(token)
                        new_active.append(token)
            else:
                # Defensive fallback (not expected to be reachable for
                # any circuit this module's row-position invariants
                # hold for): consumed inputs aren't contiguous, so an
                # in-place block replacement can't express this stage.
                # Fall back to the simple "wide leaf first, then every
                # untouched row" layout rather than producing a
                # malformed one.
                elements.append(_leaf_diagram(graph, leaf, reg))
                input_tokens.extend(wide_input_tokens)
                new_active.extend((leaf, p) for p in range(n_out))
                for i, token in enumerate(active):
                    if i not in consumed_set:
                        elements.append(_make_identity_wire())
                        input_tokens.append(token)
                        new_active.append(token)
        else:
            # A wide leaf with no inputs at all (a multi-mode state)
            # has no consumed position to anchor to -- place it at its
            # natural reading-order slot among the currently-open
            # rows, same as a narrow state birth in `_build_narrow_rows`.
            insert_at = _natural_insertion_index(graph, root_id, active, leaf)
            for i, token in enumerate(active):
                if i == insert_at:
                    elements.append(_leaf_diagram(graph, leaf, reg))
                    new_active.extend((leaf, p) for p in range(n_out))
                elements.append(_make_identity_wire())
                input_tokens.append(token)
                new_active.append(token)
            if insert_at == len(active):
                elements.append(_leaf_diagram(graph, leaf, reg))
                new_active.extend((leaf, p) for p in range(n_out))

        stage_records.append((elements, input_tokens, new_active))
        active = new_active

    for k in range(len(wide_leaves_sorted)):
        emit_narrow(runs[k])
        emit_wide(wide_leaves_sorted[k])
    emit_narrow(runs[-1])

    if not stage_records:
        return diagram

    # Nothing follows the last stage to permute against (the outer
    # composition's own `_num_outputs` is literally `diagrams[-1]
    # .num_outputs`, port for port), so its *native* output order must
    # already match the diagram's real external output order exactly.
    # Reconstruction is free to reorder every *internal* boundary (via
    # an explicit `connectivity` dict, as `emit_wide` already does), so
    # rather than special-case the last content stage's own row order,
    # append one small trailing stage of bare identity wires built
    # directly in the required final order and connect it to whatever
    # order the content naturally ended in -- exactly the same
    # `token position -> port` connectivity mechanism used everywhere
    # else. When the natural order already matches, this is a no-op.
    final_active = stage_records[-1][2]
    if num_outputs:
        # Flattened through `flatten_output_token` so this comparison
        # uses the same notion of "true producer" as `pred_of` does --
        # otherwise a diagram whose real last output is produced by an
        # elided identity leaf would never match `final_active` (which
        # only ever contains flattened tokens), and a spurious
        # permutation stage would be appended even when none is needed.
        desired_final_tokens = [flatten_output_token(_resolve_output(graph, root_id, j)) for j in range(num_outputs)]
        if desired_final_tokens != final_active and not _absorb_final_permutation(stage_records, desired_final_tokens):
            # Only reached if some element's own multi-port output block
            # would need splitting to match -- see
            # `_absorb_final_permutation`'s docstring for why this is a
            # defensive fallback rather than the common path.
            perm_elements = [_make_identity_wire() for _ in desired_final_tokens]
            stage_records.append((perm_elements, desired_final_tokens, [(leaf.id, 0) for leaf in perm_elements]))

    logger.debug("normalize_diagram: built %d stage(s)", len(stage_records))
    stages = [elements[0] if len(elements) == 1 else TensorDiagram(elements) for elements, _, _ in stage_records]

    if len(stages) == 1:
        return stages[0]

    connectivity: dict[int, dict[int, int]] = {}
    prev_new_active = [("EXT_IN", j) for j in range(num_inputs)]
    for stage_idx in range(len(stage_records)):
        _elements, input_tokens, new_active = stage_records[stage_idx]
        if stage_idx > 0:
            prev_pos = {token: i for i, token in enumerate(prev_new_active)}
            # `CompositionDiagram.connectivity[i]` is keyed by *input*
            # port of `diagrams[i+1]` (the later stage) and valued by
            # *output* port of `diagrams[i]` (the earlier stage) -- see
            # `CompositionDiagram.__post_init__`, which validates
            # `connectivity[i]`'s keys against
            # `range(diagrams[i+1].num_inputs)` and its values against
            # `range(diagrams[i].num_outputs)`. (A prior version of this
            # comment claimed the opposite convention based on a
            # misleading docstring on `_add_composition_node` in
            # nx_graph.py -- that function's *own* edge materialization
            # is backwards relative to `__post_init__`'s validated,
            # authoritative contract, which is what actually governs a
            # `CompositionDiagram`'s semantics. Getting this backwards
            # here produces a `connectivity` dict that is still a
            # well-formed bijection -- so `__post_init__` never raises --
            # but wires every boundary to the *inverse* permutation of
            # the one actually needed.)
            connectivity[stage_idx - 1] = {b_port: prev_pos[token] for b_port, token in enumerate(input_tokens)}
        prev_new_active = new_active

    return CompositionDiagram(stages, connectivity)

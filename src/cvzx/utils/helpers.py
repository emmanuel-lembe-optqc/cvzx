"""Utility functions for CVZX graph traversal, structural checking, and diagram expansion.

This module provides common helper logic for identifying wiring nodes,
traversing passthrough chains, applying rewrite rules, and safely
flattening and expanding two-mode gates. These utilities are designed
to be backend-agnostic, supporting both NetworkX and Rustworkx execution flows.
"""

from __future__ import annotations

from sympy import Expr, simplify

from cvzx.ir.base import (
    CompositionDiagram,
    ContractedDiagram,
    Diagram,
    TensorDiagram,
)
from cvzx.ir.gates import BeamsplitterGate, ControlledSumGate


def simplify_reduced_value(value: Expr | complex) -> Expr | complex:
    """Run a chain-reduction-combined algebraic value through `sympy.simplify`.

    `ChainReductionRule` (`nx.rules`/`rx.rules`) combines a chain's phases
    and gate parameters with plain `+`/`*`, which never algebraically
    reduces the result -- e.g. `sin(x)**2 + cos(x)**2` stays exactly that,
    rather than collapsing to `1`, and two chained gates whose parameters
    are exact negatives of each other (`x` then `-x`) may not compare
    equal to the identity's `0` by structural equality alone. Running the
    combined value through `simplify()` first catches both.

    Parameters
    ----------
    value : Expr | complex
        The chain-reduction-combined value to simplify.

    Returns
    -------
    Expr | complex
        The simplified value. A plain Python number is returned unchanged
        (nothing to simplify -- and nothing to lose precision on: this
        function only ever calls `simplify()` on a value that was already
        a sympy `Expr`, since a value built purely from plain Python
        numbers never becomes one via `+`/`*` alone). Deliberately does
        *not* coerce a simplified `Expr` back to a plain Python number
        even when it collapses to one with no free symbols left (e.g. a
        chain of exact multiples of `pi` staying an exact `Expr` rather
        than an approximate `float` -- `Expr.is_number` is true for any
        such exact irrational constant, not just literal numbers, so
        doing that coercion would silently lose exactness for phases like
        `pi/6 + pi/5 + pi/7`).
    """
    if not isinstance(value, Expr):
        return value
    return simplify(value)


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


def is_chase_passthrough(attrs: dict) -> bool:
    """Check if a node's attributes represent a `_chase_identity_chain` passthrough.

    A passthrough is an identity/wiring diagram (see
    `is_wiring_node_from_attrs`), a `Swap`, or a same-arity (square)
    `VoidDiagram` -- all three are pure wire-routing with no bearing on
    whatever pattern is being chased through them.

    A square `VoidDiagram` belongs here alongside a bare identity spider
    and a `Swap` because it is one of them, mid-chase: every rule that
    installs one in place of a chain member it has already decided is a
    pure pass-through does so via a bare type relabel, at the *same* node,
    with the *same* arity. Nothing about what the node physically does changes;
    only its label does.
    Treating it as opaque instead -- which is what happened before this
    was added -- makes a voided `Swap` a permanent one-way wall: since
    `_simplify_to_fixed_point` re-runs every rule to a fixed point,
    a chain that shares a `Swap` with another, already-reduced chain
    would otherwise never become reachable on any later pass, not just
    the current one, even though the wire it needs to cross is exactly
    as pass-through as it always was.

    A `VoidDiagram`'s arity must still be checked here rather than
    assumed: this rule's own `identity_chain` mechanism (see
    `_is_chase_passthrough`'s callers) only ever installs one at 1-in/
    1-out (a voided identity spider) or 2-in/2-out (a voided `Swap`), so
    in practice this only ever matches those two shapes -- but a
    `VoidDiagram` installed by some other mechanism entirely (a vanished
    state/effect, arity (0, 1) or (1, 0)) is a genuine dead end, not a
    wire to chase through, and must stay opaque.

    Parameters
    ----------
    attrs : dict
        Node attributes from the graph.

    Returns
    -------
    bool
        True if the node is transparent to `_chase_identity_chain`.
    """
    if is_wiring_node_from_attrs(attrs) or attrs.get("type") == "Swap":
        return True
    if attrs.get("type") == "VoidDiagram":
        num_inputs = attrs.get("num_inputs", 0)
        return bool(num_inputs > 0 and num_inputs == attrs.get("num_outputs", 0))
    return False


def passthrough_exit_port(attrs: dict, entry_port: int) -> int:
    """The port to continue a chase from, on the far side of a passthrough node.

    An identity spider has exactly one port on each side (always index
    0), so the exit port is just the entry port unchanged. A `Swap`
    exchanges its two modes (`in0<->out1`, `in1<->out0` -- self-inverse,
    so the same rule works whether `entry_port` names an input port
    (chasing forward) or an output port (chasing backward)): the exit
    port is the other one.

    Parameters
    ----------
    attrs : dict
        The passthrough node's attributes.
    entry_port : int
        The port the chase arrived on.

    Returns
    -------
    int
        The port to look for the next composition edge from.
    """
    if attrs.get("type") == "Swap":
        return 1 - entry_port
    return entry_port


def expand_two_mode_gates(diagram: Diagram) -> Diagram:
    """Recursively expand BeamsplitterGate/ControlledSumGate, but not CZ.

    Like `cvzx.ir.gates.expand_all`, but deliberately excludes
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
    `SqueezingGate` instances. Doing it this way also sidesteps a real limitation of
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
        return flatten_expanded_composition([expand_two_mode_gates(d) for d in diagram.diagrams], diagram.connectivity)
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


def flatten_expanded_composition(
    expanded_children: list[Diagram], connectivity: dict[int, dict[int, int]]
) -> CompositionDiagram:
    """Splice a child that expanded into its own CompositionDiagram into the parent's flat list.

    `to_graph()`'s composition-edge resolution (`find_node_by_external_output`/
    `find_node_by_external_input` in `cvzx.backends.nx.graph`) recurses into a
    composition's TENSOR/CONTRACTED children -- both store an
    `external_*_mapping` -- but never into a CompositionDiagram nested
    directly inside another CompositionDiagram, since composition containers
    don't store one (a flat composition's own boundary edges are resolved
    from `sub_diagram_ids`/`connectivity` directly, see `_add_composition_node`).
    Left un-flattened, any wire crossing such a nested boundary is silently
    dropped -- exactly what `BeamsplitterGate.expand()`'s balanced case
    produces (a `CompositionDiagram` of two expanded `ControlledSumGate` instances and
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

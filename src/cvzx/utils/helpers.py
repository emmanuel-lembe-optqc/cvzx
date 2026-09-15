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

    `ChainReductionRule` combines a chain's phases/parameters with plain
    `+`/`*`, which never algebraically reduces the result (e.g.
    `sin(x)**2 + cos(x)**2` stays that way rather than collapsing to `1`);
    running it through `simplify()` first catches that. See
    :doc:`../dev_guide/rewrite_engine` for why a simplified `Expr` is
    deliberately never coerced back to a plain number.

    Parameters
    ----------
    value : Expr | complex
        The chain-reduction-combined value to simplify.

    Returns
    -------
    Expr | complex
        The simplified value, or `value` unchanged if it wasn't a sympy
        `Expr` to begin with.
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
    whatever pattern is being chased through them. See
    :doc:`../dev_guide/rewrite_engine` for why a square `VoidDiagram`
    must count as transparent here too, not just a bare identity/`Swap`.

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
    `ControlledZGate` (its decomposition sandwiches a Fourier gate that
    would block `CopyRule` anyway). Only called under
    `assume_infinite_squeezing=True` (see `optimize()`), which also
    licenses normalizing a biased `ControlledSumGate(gain != 1)` to
    `gain=1` before expanding rather than expanding its own biased
    decomposition -- see :doc:`../dev_guide/rewrite_engine` for why both
    choices are needed for `CopyRule`/`TerminalAbsorptionRule` to reach
    through the result.

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

    `to_graph()`'s composition-edge resolution never recurses into a
    `CompositionDiagram` nested directly inside another one, so a wire
    crossing such a nested boundary would otherwise be silently dropped --
    exactly what `BeamsplitterGate.expand()`'s balanced case produces. See
    :doc:`../dev_guide/rewrite_engine` for why this only affects a directly
    nested `CompositionDiagram` (not the TENSOR/CONTRACTED case, which
    `to_graph()` already handles).

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

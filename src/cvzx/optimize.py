"""Optimization pipeline for CV ZX calculus diagrams.

Ties the individual rewrite rules in `cvzx.nx_rewrite_rules` together into a
single `optimize()` entry point: repeatedly apply the rules to a diagram
until a full round makes no further changes, then run the end-of-pipeline
`remove_void_and_identity_nodes` cleanup pass exactly once. Returns BOTH the
cleaned `nx.DiGraph` (the representation to keep computing with) and the
`Diagram` form captured right before that final cleanup ran -- once cleaned,
the graph is no longer guaranteed to be losslessly representable as a nested
`Diagram` tree, so the pre-cleanup snapshot is what `optimize()` hands back
for visualization instead of re-deriving one from the cleaned graph.
"""

from typing import NamedTuple

import networkx as nx

from cvzx.base_gates import Diagram
from cvzx.normalize_diagram import normalize_diagram
from cvzx.nx_graph import GateRegister, to_diagram, to_graph
from cvzx.nx_rewrite_rules import (
    ChainReductionRule,
    CopyRule,
    FourierNormalizationRule,
    FusionRule,
    IdentityRule,
    RewriteRule,
    TerminalAbsorptionRule,
    expand_two_mode_gates,
    remove_void_and_identity_nodes,
)

# Upper bound on the number of expand/simplify rounds `optimize()` runs, as a
# safety net against a pathological diagram that never reaches a fixed point.
_DEFAULT_MAX_ROUNDS = 100


class OptimizeResult(NamedTuple):
    """`optimize()`'s return value: the cleaned graph plus its pre-cleanup diagram.

    A plain `NamedTuple` so existing `graph, diagram = optimize(...)`-style
    unpacking works, while also allowing `.graph`/`.diagram` attribute
    access at call sites where that reads more clearly.

    Attributes:
    ----------
    graph : nx.DiGraph
        The simplified graph, AFTER the end-of-pipeline
        `remove_void_and_identity_nodes` cleanup pass -- ready to feed into
        more rewriting, analysis, or another `RewriteRule`.
    diagram : Diagram
        The simplified diagram, converted from the graph right BEFORE that
        cleanup pass ran. The cleaned graph is not guaranteed to still be
        losslessly representable as a nested `Diagram` tree, so this is
        the diagram to use for visualization -- not `to_diagram(graph)`.
    """

    graph: nx.DiGraph
    diagram: Diagram


def _build_rules(*, assume_infinite_squeezing: bool) -> list[RewriteRule]:
    """Build the ordered list of rewrite rules `optimize()` applies each round.

    The order matters: structural cleanup (dropping identities, fusing
    same-color spiders, collapsing chains of the same gate) runs before the
    cross-type algebraic folding (`FourierNormalizationRule`) it feeds, which
    in turn prepares the ground for `TerminalAbsorptionRule`. `CopyRule` runs
    last since it can only fire on the copy spiders that BS/CSUM expansion
    exposes, and both are only sound under the idealized-eigenstate
    assumption `assume_infinite_squeezing` names.

    Parameters:
    ----------
    assume_infinite_squeezing : bool
        Whether to include the rules that are only exact for idealized
        (infinite squeezing) eigenstates: `CopyRule`, and
        `TerminalAbsorptionRule`'s squeezing/cross-color-discard sub-cases.

    Returns:
    -------
    list[RewriteRule]
        The rules to run, in the order they should be tried each round.
    """
    rules: list[RewriteRule] = [
        IdentityRule(),
        FusionRule(),
        ChainReductionRule(),
        FourierNormalizationRule(),
        TerminalAbsorptionRule(assume_infinite_squeezing=assume_infinite_squeezing),
    ]
    if assume_infinite_squeezing:
        rules.append(CopyRule())
    return rules


def _simplify_to_fixed_point(graph: nx.DiGraph, rules: list[RewriteRule]) -> bool:
    """Apply `rules` to `graph` in-place, repeatedly, until none of them match.

    Parameters:
    ----------
    graph : nx.DiGraph
        The graph to simplify in-place.
    rules : list[RewriteRule]
        The rules to try each pass, in order.

    Returns:
    -------
    bool
        True if at least one rule matched (and so `graph` was modified) at
        any point during this call.
    """
    changed = False
    while True:
        pass_changed = False
        for rule in rules:
            registry = GateRegister()
            registry.build_from_graph(graph)
            if rule.match(graph, registry):
                rule.apply_rule(graph, registry)
                pass_changed = True
        if not pass_changed:
            return changed
        changed = True


def optimize(
    diagram: Diagram,
    *,
    assume_infinite_squeezing: bool = False,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
) -> OptimizeResult:
    """Simplify a diagram by repeatedly applying the CV ZX rewrite rules.

    Each round: (1) if `assume_infinite_squeezing`, expand every
    `BeamsplitterGate`/`ControlledSumGate` into its `ContractedDiagram` form
    via `expand_two_mode_gates` (never `ControlledZGate` -- see that
    function's docstring for why); (2) convert to the graph representation
    once; (3) run `IdentityRule`, `FusionRule`, `ChainReductionRule`,
    `FourierNormalizationRule`, `TerminalAbsorptionRule` and, when
    `assume_infinite_squeezing`, `CopyRule` to a fixed point over that graph;
    (4) convert back to a diagram to feed the next round. Rounds repeat
    because folding/absorbing gates can combine adjacent two-mode gates
    (e.g. two `ControlledSumGate`s on the same modes into one, via
    `ChainReductionRule`) that themselves need expanding again to expose
    further copy patterns.

    A round that expands two-mode gates but has no rule actually match is
    discarded rather than committed: expansion alone is not a simplification,
    so the state from before that round is kept instead of a
    gratuitously-expanded one.

    With `assume_infinite_squeezing=False` (the default), only the rules
    that are exact for any physical state run: `IdentityRule`, `FusionRule`,
    `ChainReductionRule`, `FourierNormalizationRule`, and rotation absorption
    within `TerminalAbsorptionRule`. Two-mode gates are never expanded and
    `CopyRule` never runs, since both are only sound when terminal states
    are treated as idealized (infinite squeezing) eigenstates.

    Once the round loop reaches a fixed point, `to_diagram(graph)` is
    captured -- this is the value returned as `.diagram` -- and THEN
    `remove_void_and_identity_nodes` runs once on the graph: it strips
    every transient `VoidDiagram` `CopyRule` left behind (genuinely
    shrinking whichever container held it) and gives `IdentityRule` one
    more pass to remove any identity spider that removal exposed. That
    final step can restructure the graph in ways a nested `Diagram` tree
    can no longer losslessly express, which is exactly why the diagram
    snapshot is taken beforehand rather than derived from the cleaned
    graph afterward.

    The `Diagram` tree (`CompositionDiagram`/`TensorDiagram`/`ContractedDiagram`
    nesting `Diagram` leaves) is a good representation for visualizing a
    circuit, but it's a poor one to compute with further: matching and
    rewriting need the flat, randomly-addressable `nx.DiGraph` form (that's
    what every rule in `cvzx.nx_rewrite_rules` operates on) instead of walking
    a nested tree. `optimize()` returns both: the cleaned graph, ready to
    feed into more rewriting, analysis, or another `RewriteRule`, and the
    pre-cleanup diagram, ready for visualization.

    Parameters:
    ----------
    diagram : Diagram
        The diagram to simplify.
    assume_infinite_squeezing : bool
        If True, also run the rules and expansions that are only exact
        for idealized (infinite squeezing) eigenstates: two-mode gate
        expansion, `CopyRule`, and `TerminalAbsorptionRule`'s
        squeezing/cross-color-discard sub-cases. If False (default), only
        exact identities are applied.
    max_rounds : int
        Safety cap on the number of expand/simplify rounds, in case a
        diagram never reaches a fixed point. Defaults to
        `_DEFAULT_MAX_ROUNDS`.

    Returns:
    -------
    OptimizeResult
        A `(graph, diagram)` named tuple: the cleaned `nx.DiGraph`, and the
        `Diagram` as it stood right before the final cleanup pass.
    """
    rules = _build_rules(assume_infinite_squeezing=assume_infinite_squeezing)
    graph = to_graph(diagram)

    for _ in range(max_rounds):
        # Normalize into alternating type-1/type-2 stages *before* any
        # expansion: `normalize_diagram` only understands compact-form
        # 2-mode gates (a single leaf) and bails out as a no-op the
        # moment it sees a `ContractedDiagram` container, which is
        # exactly what `expand_two_mode_gates` produces -- so it has to
        # run on `diagram` first. This is what makes same-container
        # matching enough for `ChainReductionRule`/`TerminalAbsorptionRule`
        # even when the diagram was built with a gate nested inside one
        # row of a wider `TensorDiagram` (e.g. a mid-circuit ancilla
        # preparation) -- see `cvzx.normalize_diagram` for why that otherwise
        # leaves same-row neighbors that aren't direct composition
        # siblings.
        normalized = normalize_diagram(diagram)
        candidate = expand_two_mode_gates(normalized) if assume_infinite_squeezing else normalized

        candidate_graph = to_graph(candidate)
        if not _simplify_to_fixed_point(candidate_graph, rules):
            # Nothing matched this round -- expanding (if we did) didn't
            # unlock anything, so keep the state from before this round.
            break

        graph = candidate_graph
        diagram = to_diagram(graph)

    pre_cleanup_diagram = to_diagram(graph)
    remove_void_and_identity_nodes(graph)
    return OptimizeResult(graph=graph, diagram=pre_cleanup_diagram)

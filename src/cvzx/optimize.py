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

import logging
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

logger = logging.getLogger(__name__)

# Upper bound on the number of expand/simplify rounds `optimize()` runs, as a
# safety net against a pathological diagram that never reaches a fixed point.
_DEFAULT_MAX_ROUNDS = 100


class OptimizeResult(NamedTuple):
    """`optimize()`'s return value: the cleaned graph plus its pre-cleanup diagram.

    A plain `NamedTuple` so existing `graph, diagram = optimize(...)`-style
    unpacking works, while also allowing `.graph`/`.diagram` attribute
    access at call sites where that reads more clearly.

    Attributes
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

    Parameters
    ----------
    assume_infinite_squeezing : bool
        Whether to include the rules that are only exact for idealized
        (infinite squeezing) eigenstates: `CopyRule`, and
        `TerminalAbsorptionRule`'s squeezing/cross-color-discard sub-cases.

    Returns
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

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to simplify in-place.
    rules : list[RewriteRule]
        The rules to try each pass, in order.

    Returns
    -------
    bool
        True if at least one rule matched (and so `graph` was modified) at
        any point during this call.

    Notes
    -----
    `GateRegister` is rebuilt from scratch only right after a rule actually
    applies a change to `graph` -- never merely because a pass moves on to
    the next rule. A rule whose `match()` returns nothing leaves `graph`
    (and therefore every category the registry indexes) untouched, so the
    registry already on hand is still exactly correct for whichever rule is
    tried next; rebuilding in that case would be pure wasted `O(N)` work for
    no change in behavior. This relies on nothing about how any individual
    rule mutates the graph -- only on `RewriteRule.apply_rule`'s own
    guarantee that "no match" means "no mutation" (see the dev guide's
    "Keeping GateRegister in sync" for why rules themselves don't maintain
    the registry incrementally).
    """
    changed = False
    n_passes = 0
    registry = GateRegister()
    registry.build_from_graph(graph)
    while True:
        n_passes += 1
        pass_changed = False
        for rule in rules:
            if rule.match(graph, registry):
                rule.apply_rule(graph, registry)
                pass_changed = True
                registry.build_from_graph(graph)
        if not pass_changed:
            logger.debug("_simplify_to_fixed_point: reached fixed point after %d pass(es)", n_passes)
            return changed
        changed = True


def optimize(
    diagram: Diagram,
    *,
    assume_infinite_squeezing: bool = False,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
) -> OptimizeResult:
    """Simplify a diagram by repeatedly applying the CV ZX rewrite rules.

    Each round: optionally expand two-mode gates (`assume_infinite_squeezing`),
    convert to the graph representation, run every rule to a fixed point,
    then convert back to a diagram for the next round -- rounds repeat
    because folding/absorbing gates can expose further reductions (e.g. two
    `ControlledSumGate`s merging into one that itself needs re-expanding).
    See the user guide ("Optimizing a diagram") for what
    `assume_infinite_squeezing` unlocks and worked examples of each, and
    the dev guide ("Architecture overview") for the full round-by-round
    breakdown.

    Parameters
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

    Returns
    -------
    OptimizeResult
        A `(graph, diagram)` named tuple: the cleaned `nx.DiGraph`, and the
        `Diagram` as it stood right before the final cleanup pass.

    Raises
    ------
    TypeError
        If `diagram` is not a `Diagram` instance.
    """
    if not isinstance(diagram, Diagram):
        msg = f"optimize() expects a Diagram, got {type(diagram).__name__}."
        raise TypeError(msg)

    rules = _build_rules(assume_infinite_squeezing=assume_infinite_squeezing)
    graph = to_graph(diagram)

    for round_index in range(max_rounds):
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
            logger.debug("optimize: converged after %d round(s)", round_index)
            break

        graph = candidate_graph
        diagram = to_diagram(graph)
    else:
        logger.warning(
            "optimize: reached max_rounds=%d without converging -- result may not be fully simplified",
            max_rounds,
        )

    pre_cleanup_diagram = to_diagram(graph)
    remove_void_and_identity_nodes(graph)
    return OptimizeResult(graph=graph, diagram=pre_cleanup_diagram)

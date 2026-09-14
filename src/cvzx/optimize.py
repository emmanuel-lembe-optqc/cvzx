"""Optimization pipeline for CV ZX calculus diagrams.

Ties the individual rewrite rules together into a single `optimize()` entry
point: repeatedly apply the rules to a diagram until a full round makes no
further changes. Backend-agnostic: `backend` (see `cvzx.config.Backend`)
picks whether the `networkx` or `rustworkx` `CVZXGraph`/rewrite-rule
modules do the work, via `cvzx.backend.get_backend_modules`. Returns BOTH
the cleaned `CVZXGraph` (the representation to keep computing with) and the
`Diagram` form captured right before that final cleanup ran -- once cleaned,
the graph is no longer guaranteed to be losslessly representable as a nested
`Diagram` tree, so the pre-cleanup snapshot is what `optimize()` hands back
for visualization instead of re-deriving one from the cleaned graph.
"""

import logging
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple

from cvzx.backend import get_backend_modules
from cvzx.base_gates import Diagram
from cvzx.config import Backend
from cvzx.normalize_diagram import normalize_diagram
from cvzx.utils import expand_two_mode_gates

if TYPE_CHECKING:
    from cvzx.nx_graph import CVZXGraph as NxCVZXGraph
    from cvzx.nx_rewrite_rules import RewriteRule as NxRewriteRule
    from cvzx.rx_graph import CVZXGraph as RxCVZXGraph
    from cvzx.rx_rewrite_rules import RewriteRule as RxRewriteRule

logger = logging.getLogger(__name__)

# Upper bound on the number of expand/simplify rounds `optimize()` runs, as a
# safety net against a pathological diagram that never reaches a fixed point.
_DEFAULT_MAX_ROUNDS = 100

# Upper bound on the number of per-pass rule-application passes
# `_simplify_to_fixed_point` runs within a single round, as a safety net
# against a pathological rule interaction that never reaches a fixed point.
# Matches `RewriteRule.apply_rule`'s own internal `_APPLY_RULE_MAX_ROUNDS`
# cap, so one capped-out rule doesn't multiply against this cap too.
_MAX_SIMPLIFY_PASSES = 100


class OptimizeResult(NamedTuple):
    """`optimize()`'s return value: the cleaned graph plus its pre-cleanup diagram.

    A plain `NamedTuple` so existing `graph, diagram = optimize(...)`-style
    unpacking works, while also allowing `.graph`/`.diagram` attribute
    access at call sites where that reads more clearly.

    Attributes
    ----------
    graph : CVZXGraph
        The simplified graph ready to feed into
        more rewriting, analysis, or another `RewriteRule`.
    diagram : Diagram
        The simplified diagram, converted from the graph right BEFORE that
        cleanup pass ran. The cleaned graph is not guaranteed to still be
        losslessly representable as a nested `Diagram` tree, so this is
        the diagram to use for visualization -- not `to_diagram(graph)`.
    """

    graph: "NxCVZXGraph | RxCVZXGraph"
    diagram: Diagram


def _build_rules(rules_mod: ModuleType, *, assume_infinite_squeezing: bool) -> "list[NxRewriteRule | RxRewriteRule]":
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
    rules_mod : ModuleType
        The backend's rewrite-rule module (`nx_rewrite_rules` or
        `rx_rewrite_rules`), as resolved by `cvzx.backend.get_backend_modules`.
    assume_infinite_squeezing : bool
        Whether to include the rules that are only exact for idealized
        (infinite squeezing) eigenstates: `CopyRule`, and
        `TerminalAbsorptionRule`'s squeezing/cross-color-discard sub-cases.

    Returns
    -------
    list[NxRewriteRule | RxRewriteRule]
        The rules to run, in the order they should be tried each round.
    """
    rules = [
        rules_mod.IdentityRule(),
        rules_mod.FusionRule(),
        rules_mod.ChainReductionRule(),
        rules_mod.FourierNormalizationRule(),
        rules_mod.TerminalAbsorptionRule(assume_infinite_squeezing=assume_infinite_squeezing),
    ]
    if assume_infinite_squeezing:
        rules.append(rules_mod.CopyRule())
    return rules


def _simplify_to_fixed_point(graph: "NxCVZXGraph | RxCVZXGraph", rules: "list[NxRewriteRule | RxRewriteRule]") -> bool:
    """Apply `rules` to `graph` in-place, repeatedly, until none of them match.

    Parameters
    ----------
    graph : CVZXGraph
        The graph to simplify in-place.
    rules : list[NxRewriteRule | RxRewriteRule]
        The rules to try each pass, in order.

    Returns
    -------
    bool
        True if a genuine fixed point was reached AND at least one rule
        matched (so `graph` was modified) along the way. False both for
        "nothing matched at all" and for "hit `_MAX_SIMPLIFY_PASSES`
        without reaching a fixed point" -- `optimize()`'s caller treats
        both the same way (stop, don't commit this round), since a
        not-fully-simplified `graph` isn't safe to build the next round's
        `normalize_diagram`/`expand_two_mode_gates` on top of.

    Notes
    -----
    `graph`'s registry is rebuilt from scratch only right after a rule
    actually applies a change to `graph` -- never merely because a pass
    moves on to the next rule. A rule whose `match()` returns nothing leaves
    `graph` (and therefore every category the registry indexes) untouched,
    so the registry already on hand is still exactly correct for whichever
    rule is tried next; rebuilding in that case would be pure wasted `O(N)`
    work for no change in behavior. This relies on nothing about how any
    individual rule mutates the graph -- only on `RewriteRule.apply_rule`'s
    own guarantee that "no match" means "no mutation" (see the dev guide's
    "Keeping GateRegister in sync" for why rules themselves don't maintain
    the registry incrementally).

    Bounded by `_MAX_SIMPLIFY_PASSES` as a safety net, mirroring
    `RewriteRule.apply_rule`'s own internal `_APPLY_RULE_MAX_ROUNDS` cap:
    some rule interactions (e.g. `CopyRule`-driven duplication feeding back
    into `FusionRule`) are not currently guaranteed to reach a true fixed
    point, so without a cap here a pathological case can loop effectively
    forever instead of returning a not-fully-simplified result.
    """
    changed = False
    n_passes = 0
    graph.rebuild_registry()
    while True:
        n_passes += 1
        if n_passes > _MAX_SIMPLIFY_PASSES:
            logger.warning(
                "_simplify_to_fixed_point: reached _MAX_SIMPLIFY_PASSES=%d without a fixed "
                "point -- discarding this round's (incomplete) simplification",
                _MAX_SIMPLIFY_PASSES,
            )
            return False
        pass_changed = False
        for rule in rules:
            # `rule` and `graph` are always resolved from the same backend
            # module pair by `get_backend_modules` (see `optimize()`), so
            # this is always the matching nx/nx or rx/rx combination even
            # though the static union type can't express that correlation.
            if rule.match(graph):  # type: ignore[arg-type]
                rule.apply_rule(graph)  # type: ignore[arg-type]
                pass_changed = True
                graph.rebuild_registry()
        if not pass_changed:
            logger.debug("_simplify_to_fixed_point: reached fixed point after %d pass(es)", n_passes)
            return changed
        changed = True


def optimize(
    diagram: Diagram,
    *,
    backend: Backend | str | None = None,
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
    backend : Backend | str | None
        Which `CVZXGraph` backend to run the rewrite rules on -- see
        `cvzx.config.Backend`. `None` (default) uses
        `cvzx.config.DEFAULT_BACKEND`.
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
        A `(graph, diagram)` named tuple: the cleaned `CVZXGraph`, and the
        `Diagram` as it stood right before the final cleanup pass.

    Raises
    ------
    TypeError
        If `diagram` is not a `Diagram` instance.
    """
    if not isinstance(diagram, Diagram):
        msg = f"optimize() expects a Diagram, got {type(diagram).__name__}."
        raise TypeError(msg)

    _, graph_mod, rules_mod = get_backend_modules(backend)
    rules = _build_rules(rules_mod, assume_infinite_squeezing=assume_infinite_squeezing)
    graph = graph_mod.to_graph(diagram)

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

        candidate_graph = graph_mod.to_graph(candidate)
        if not _simplify_to_fixed_point(candidate_graph, rules):
            # Nothing matched this round -- expanding (if we did) didn't
            # unlock anything, so keep the state from before this round.
            logger.debug("optimize: converged after %d round(s)", round_index)
            break

        graph = candidate_graph
        diagram = graph_mod.to_diagram(graph)
    else:
        logger.warning(
            "optimize: reached max_rounds=%d without converging -- result may not be fully simplified",
            max_rounds,
        )

    pre_cleanup_diagram = graph_mod.to_diagram(graph)
    return OptimizeResult(graph=graph, diagram=pre_cleanup_diagram)

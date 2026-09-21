"""Rule-order experiment harness.

A small, repeatable tool for comparing how a `Diagram` reduces under
different simplification strategies, so rule-interaction gaps can be
found by diffing results instead of guessing. Not shipped as library
code -- a dev/investigation tool only.

Three run modes:

- "manual": loop the rule list once per round, round-tripping through
  `to_diagram()`/`to_graph()` after every SINGLE rule application --
  matches the hand-driven notebook workflow.
- "fixed_point": `cvzx.passes.optimize._simplify_to_fixed_point` alone,
  with no outer `normalize_diagram`/`expand_two_mode_gates` re-invocation.
- "optimize": the real `cvzx.passes.optimize.optimize()` entry point,
  unmodified.

Usage (as a library, from a REPL or another script)::

    from scripts.rule_order_experiment import run, diff_report
    from cvzx.backend import Backend

    r1 = run(diagram, mode="manual", assume_infinite_squeezing=True)
    r2 = run(diagram, mode="optimize", assume_infinite_squeezing=True)
    print(diff_report(r1, r2))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cvzx.backend import get_backend_modules
from cvzx.passes.normalize import normalize_diagram
from cvzx.passes.optimize import _build_rules, _simplify_to_fixed_point
from cvzx.utils.helpers import expand_two_mode_gates

if TYPE_CHECKING:
    from cvzx.config import Backend
    from cvzx.ir.base import Diagram

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """One harness run's outcome."""

    mode: str
    backend: Backend
    diagram: Diagram
    repr_str: str
    rounds: int
    trace: list[str] = field(default_factory=list)


def run(
    diagram: Diagram,
    *,
    mode: str,
    assume_infinite_squeezing: bool = False,
    backend: Backend | str | None = None,
    max_rounds: int = 100,
) -> RunResult:
    """Run `diagram` through `mode` and report the final result plus a trace.

    Parameters
    ----------
    diagram : Diagram
        The diagram to simplify.
    mode : str
        One of "manual", "fixed_point", "optimize".
    assume_infinite_squeezing : bool
        Forwarded to whichever rules/expansion this mode uses.
    backend : Backend | str | None
        Which CVZXGraph backend to run on. `None` uses the configured default.
    max_rounds : int
        Safety cap on rounds/passes.

    Returns
    -------
    RunResult

    Raises
    ------
    ValueError
        If `mode` isn't recognized.
    """
    resolved_backend, graph_mod, rules_mod = get_backend_modules(backend)
    rules = _build_rules(rules_mod, assume_infinite_squeezing=assume_infinite_squeezing)

    if mode == "optimize":
        from cvzx.passes.optimize import optimize

        result = optimize(
            diagram, backend=backend, assume_infinite_squeezing=assume_infinite_squeezing, max_rounds=max_rounds
        )
        final = result.diagram
        return RunResult(mode=mode, backend=resolved_backend, diagram=final, repr_str=repr(final), rounds=-1)

    if mode == "fixed_point":
        normalized = normalize_diagram(diagram)
        candidate = expand_two_mode_gates(normalized) if assume_infinite_squeezing else normalized
        graph = graph_mod.to_graph(candidate)
        graph.rebuild_registry()
        graph, _ = _simplify_to_fixed_point(graph, rules, graph_mod)
        final = graph_mod.to_diagram(graph)
        return RunResult(mode=mode, backend=resolved_backend, diagram=final, repr_str=repr(final), rounds=-1)

    if mode == "manual":
        trace: list[str] = []
        normalized = normalize_diagram(diagram)
        candidate = expand_two_mode_gates(normalized) if assume_infinite_squeezing else normalized
        graph = graph_mod.to_graph(candidate)
        graph.rebuild_registry()

        rounds = 0
        for _ in range(max_rounds):
            rounds += 1
            fired = False
            for rule in rules:
                if rule.match(graph):
                    trace.append(f"round {rounds}: {type(rule).__name__} fires")
                    rule.apply_rule(graph)
                    fired = True
                    graph.rebuild_registry()
                    graph = graph_mod.to_graph(graph_mod.to_diagram(graph))
                    graph.rebuild_registry()
            if not fired:
                break
        final = graph_mod.to_diagram(graph)
        return RunResult(
            mode=mode, backend=resolved_backend, diagram=final, repr_str=repr(final), rounds=rounds, trace=trace
        )

    msg = f"Unknown mode {mode!r}; expected 'manual', 'fixed_point', or 'optimize'."
    raise ValueError(msg)


def diff_report(a: RunResult, b: RunResult) -> str:
    """Human-readable comparison of two `RunResult`s.

    Returns
    -------
    str
    """
    if a.repr_str == b.repr_str:
        return f"IDENTICAL ({a.mode} == {b.mode})"
    lines = [
        f"DIVERGENT ({a.mode} vs {b.mode})",
        f"  [{a.mode}] {a.repr_str}",
        f"  [{b.mode}] {b.repr_str}",
    ]
    return "\n".join(lines)


def run_all_modes(
    diagram: Diagram, *, assume_infinite_squeezing: bool = False, backend: Backend | str | None = None
) -> dict[str, RunResult]:
    """Run all three modes on `diagram` and return them keyed by mode name.

    Returns
    -------
    dict[str, RunResult]
    """
    return {
        mode: run(diagram, mode=mode, assume_infinite_squeezing=assume_infinite_squeezing, backend=backend)
        for mode in ("manual", "fixed_point", "optimize")
    }

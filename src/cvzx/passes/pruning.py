"""Pruning of near-identity Gaussian gates.

A gate whose parameter is close to the identity value does almost nothing to the state, but on hardware it may still
cost noise (on MoQuren, every macronode step it adds teleports every live mode once). `prune_small_gaussian_gates`
replaces such gates by identity wires, at the compact-gate level and before expansion: snapping a parameter to its
identity value is not always expandable (`BeamsplitterGate(0)` divides by tan 0, `ControlledZGate(0)` builds
Sq(1/sqrt(0)), and the squeezing identity is tau = 1, not 0), whereas identity wires are exact and are then fused away
by `optimize`'s `IdentityRule`.

Each gate type has its own distance from identity, in the units its parameter acts on (see `gate_distance`), and
`epsilon` can be one number or a threshold per type. A gate is never pruned if its parameter is symbolic or is a
feedforward target (a non-empty `param_measurement_map`): removing it would orphan the measurement it depends on. A
`keep(gate)` callback lets an outside cost model, such as moquren-emu's, veto or drive the decision ("noise-aware"
pruning: pass ``epsilon=math.inf`` and let `keep` decide).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from sympy import Expr

from cvzx.ir.base import CompositionDiagram, Diagram, QSpider, TensorDiagram, ZxPoly
from cvzx.ir.gates import (
    ArbitraryGate,
    BeamsplitterGate,
    ControlledSumGate,
    ControlledZGate,
    DisplacementGate,
    PhaseRotationGate,
    ShearPInvariantGate,
    ShearXInvariantGate,
    Squeezing45Gate,
    SqueezingGate,
    TwoModeShearGate,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

GATE_TYPES: dict[type, str] = {
    PhaseRotationGate: "rotation",
    SqueezingGate: "squeezing",
    Squeezing45Gate: "squeezing45",
    ArbitraryGate: "arbitrary",
    DisplacementGate: "displacement",
    ShearXInvariantGate: "shear",
    ShearPInvariantGate: "shear",
    BeamsplitterGate: "beam_splitter",
    ControlledZGate: "controlled_z",
    ControlledSumGate: "controlled_sum",
    TwoModeShearGate: "two_mode_shear",
}
"""Gate class -> the type name used for per-type thresholds."""


def _wrap(angle: float) -> float:
    """Fold an angle into (-pi, pi].

    Returns
    -------
    float
        The folded angle.
    """
    return math.pi - (math.pi - angle) % (2 * math.pi)


def gate_distance(gate: Diagram) -> float | None:  # ruff: ignore[too-many-return-statements]
    """How far a Gaussian gate is from the identity, in its own parameter's units; None if it cannot be pruned.

    rotation |theta| (wrapped), squeezing |ln tau| (tau <= 0 is never the identity), squeezing45 |ln cot theta|,
    arbitrary max(|lam|, |alpha + beta|), displacement |alpha|, shears |kappa| or |eta|, beam splitter |theta|
    (wrapped), CZ and CSUM |gain|, two-mode shear max(|a|, |b|).

    Returns
    -------
    float | None
        The distance, or None for gates that are not prunable here (non-Gaussian, measurements, states, spiders,
        symbolic parameters).
    """
    kind = GATE_TYPES.get(type(gate))
    if kind is None:
        return None
    values = [getattr(gate, f) for f in getattr(type(gate), "_param_fields", ())]
    if getattr(gate, "parametric", False) or any(isinstance(v, Expr) for v in values):
        return None
    if kind in {"rotation", "beam_splitter"}:
        return abs(_wrap(float(values[0])))
    if kind == "squeezing":
        tau = float(values[0])
        return abs(math.log(tau)) if tau > 0 else math.inf
    if kind == "squeezing45":
        t = math.tan(float(values[0]))
        return abs(math.log(1 / t)) if t > 0 else math.inf
    if kind == "arbitrary":
        alpha, beta, lam = (float(v) for v in values)
        return max(abs(lam), abs(_wrap(alpha + beta)))
    return max(abs(complex(v)) for v in values)


def is_feedforward_target(gate: Diagram) -> bool:
    """Tell whether one of the gate's parameters is fed forward from a measurement.

    Returns
    -------
    bool
        True for a gate with a non-empty ``param_measurement_map``.
    """
    return bool(getattr(gate, "param_measurement_map", None))


@dataclass(frozen=True)
class PrunedGate:
    """One gate replaced by identity wires."""

    kind: str
    distance: float
    gate: Diagram


@dataclass
class PruneResult:
    """The pruned diagram and what was removed."""

    diagram: Diagram
    pruned: list[PrunedGate] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        """Whether any gate was pruned."""
        return bool(self.pruned)


def _threshold(epsilon: float | Mapping[str, float], kind: str) -> float:
    if isinstance(epsilon, (int, float)):
        return float(epsilon)
    return float(epsilon.get(kind, 0.0))


def _drop_identities(d: CompositionDiagram, children: list[Diagram], made: set[int]) -> Diagram:
    """Rebuild a composition without the identities pruning created, composing the wiring maps around each one.

    ``connectivity[i]`` maps each input port of child i+1 to an output port of child i. Dropping an identity child j
    in the middle gives ``new[x] = c[j-1][c[j][x]]``; at either end it is dropped only if its map is the identity,
    since otherwise the composition's own port order would change.

    Returns
    -------
    Diagram
        The rebuilt composition, or its only remaining child.
    """
    kids = list(children)
    conn = {i: dict(m) for i, m in d.connectivity.items()}
    j = 0
    while j < len(kids) and len(kids) > 1:
        if id(kids[j]) not in made:
            j += 1
            continue
        if 0 < j < len(kids) - 1:
            before, after = conn[j - 1], conn[j]
            merged = {x: before[after[x]] for x in after}
            conn = {(i if i < j - 1 else i - 1): m for i, m in conn.items() if i not in {j - 1, j}}
            conn[j - 1] = merged
        elif j == 0 and all(k == v for k, v in conn[0].items()):
            conn = {i - 1: m for i, m in conn.items() if i != 0}
        elif j == len(kids) - 1 and all(k == v for k, v in conn[j - 1].items()):
            conn = {i: m for i, m in conn.items() if i != j - 1}
        else:
            j += 1
            continue
        del kids[j]
    if len(kids) == 1:
        return kids[0]
    return replace(d, diagrams=kids, connectivity=dict(sorted(conn.items())))


def prune_small_gaussian_gates(
    diagram: Diagram,
    epsilon: float | Mapping[str, float] = 1e-3,
    *,
    keep: Callable[[Diagram], bool] | None = None,
) -> PruneResult:
    """Replace every Gaussian gate closer to the identity than its threshold by identity wires.

    Parameters
    ----------
    diagram : Diagram
        The diagram, in compact-gate form (before `optimize` expands or fuses gates).
    epsilon : float | Mapping[str, float]
        One threshold for every gate type, or a threshold per type name of `GATE_TYPES` (types not named are kept).
        A gate is a candidate when ``gate_distance(gate) < epsilon``; 0 prunes nothing.
    keep : Callable[[Diagram], bool] | None
        Called on every candidate; returning True keeps it. Feedforward targets are kept regardless.

    Returns
    -------
    PruneResult
        The new diagram (the input is not mutated) and the pruned gates.
    """
    result = PruneResult(diagram)
    made: set[int] = set()  # ids of the identities this pass created (only these are dropped)

    def visit(d: Diagram) -> Diagram:  # ruff: ignore[too-many-return-statements]
        if isinstance(d, CompositionDiagram):
            new = [visit(c) for c in d.diagrams]
            if all(a is b for a, b in zip(new, d.diagrams, strict=True)):
                return d
            return _drop_identities(d, new, made)
        if isinstance(d, TensorDiagram):
            new = [visit(c) for c in d.diagrams]
            if all(a is b for a, b in zip(new, d.diagrams, strict=True)):
                return d
            out = replace(d, diagrams=new)
            if all(id(c) in made for c in new):
                made.add(id(out))  # a layer of pruned gates only: an identity on all its wires
            return out
        distance = gate_distance(d)
        if distance is None or is_feedforward_target(d):
            return d
        kind = GATE_TYPES[type(d)]
        if distance >= _threshold(epsilon, kind) or (keep is not None and keep(d)):
            return d
        result.pruned.append(PrunedGate(kind, distance, d))
        wires = [QSpider(1, 1, ZxPoly({})) for _ in range(d.num_inputs)]
        made.update(id(w) for w in wires)
        if len(wires) == 1:
            return wires[0]
        out = TensorDiagram(wires)
        made.add(id(out))
        return out

    result.diagram = visit(diagram)
    return result

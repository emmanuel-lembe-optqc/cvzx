"""Convert a canonical cvzx `Diagram` into an mqc3 `CircuitRepr`.

This is the reverse direction of `cvzx.circuit_to_diagram`: where that module
naively translates an mqc3 `CircuitRepr` into a cvzx `Diagram` and then
canonicalizes it, this module walks an already-canonical (or
canonicalizable) `Diagram` -- the alternating type-1/type-2 stages
`normalize_diagram` produces -- and emits the equivalent sequence of
mqc3 intrinsic operations.

Why walk the *canonical* form and not an arbitrary one
-------------------------------------------------------
`normalize_diagram` decomposes every diagram to the same flat set of
primitive leaves regardless of how they were originally grouped, so this
module translates each primitive leaf independently rather than pattern-
matching compositions back into, e.g., a single `BeamSplitter` op -- the
result is semantically equivalent to the original circuit but not
necessarily *op-for-op* identical (see the user guide, "Converting to and
from mqc3 circuits", for a worked example). The input `Diagram` must also
not already be the target of `optimize()`'s algebraic spider-fusion
rewrites (as opposed to `normalize_diagram`'s purely structural ones): a
fused spider's phase polynomial may no longer match any primitive shape
this module recognizes, in which case conversion raises
`NotImplementedError` rather than silently producing something else.

Per-leaf conversion formulas
-----------------------------
Each of these is the algebraic inverse of the corresponding formula in
`cvzx.circuit_to_diagram` (see that module's docstring for the derivations):

- `PhaseRotationGate(theta)` -> `intrinsic.PhaseRotation(-theta)`.
- `Fourier()` -> `intrinsic.PhaseRotation(pi/2)`; `FourierInv()` ->
  `intrinsic.PhaseRotation(-pi/2)`; `Fourier2()` ->
  `intrinsic.PhaseRotation(pi)`.
- `ShearXInvariantGate(kappa)`, `ShearPInvariantGate(eta)`,
  `ArbitraryGate(alpha, beta, lam)`, `Squeezing45Gate(theta)`,
  `TwoModeShearGate(a, b)`, `MeasurementGate(theta)` all map 1:1 onto
  their correspondingly-named mqc3 op with the *same* parameter(s).
- `SqueezingGate(tau)` has no bare mqc3 primitive (mqc3's `Squeezing`
  op has a different, fixed-rotation definition -- see
  `circuit_to_diagram`); it is instead expressed via `ArbitraryGate`'s own
  `alpha = beta = 0` special case: `intrinsic.Arbitrary(0, 0, ln(tau))`.
- `DisplacementGate(alpha)` -> `intrinsic.Displacement(x, p)` with
  `x = sqrt(2)*Re(alpha)`, `p = sqrt(2)*Im(alpha)`.
- `ControlledZGate(gain=g)` -> `intrinsic.ControlledZ(-g)`.
- `BeamsplitterGate(theta)` -> `intrinsic.BeamSplitter(sqrt_r, 0)` with
  `sqrt_r = cos(theta)` -- the `theta_rel = 0` special case of the
  general formula derived in `circuit_to_diagram`.
- A bare zero-phase state leaf (`QSpider(0, 1, 0)`) -> a mode with
  `HardwareConstrainedSqueezedState(phi=0)`; the `PSpider` counterpart
  -> `phi = pi/2` (mirroring the `QSpider`-is-x-type/`PSpider`-is-p-
  type convention used throughout this codebase). A bare zero-phase
  effect leaf (`QSpider(1, 0, 0)` / `PSpider(1, 0, 0)`) -> a plain
  `intrinsic.Measurement(pi/2)` / `Measurement(0)` (measuring x or p
  directly) -- these arise if `diagram` contains an already-`.expand()`-
  ed `MeasurementGate`/state rather than the compact form, which is
  otherwise handled directly.

Feedforward
-----------
A `(1, 0)` `QSpider`/`PSpider` effect with phase *exactly* `ZxPoly({1: -m})`
for a single symbol `m` -- the leaf shape `cvzx.completion.
complete_diagram()` produces to close an open output port -- is
recognized specially: it becomes a plain `intrinsic.Measurement`
(`pi/2`/`0` for `QSpider`/`PSpider`, same as the bare zero-phase case
above), and the resulting mqc3 `Operation` is tracked against `m` for the
rest of the walk. Any later leaf whose own parameter is affine
(`slope*m + intercept`, for that same single symbol `m`) in a tracked
symbol has that parameter translated into an mqc3
`FeedForward[MeasuredVariable]` (via `mqc3.feedforward.
ff_to_mul_constant`/`ff_to_add_constant`, composed to match the affine
relationship) instead of a plain float -- the exact algebraic inverse of
`cvzx.circuit_to_diagram`'s own feedforward reconstruction. A parameter
depending on more than one symbol, on a symbol with no tracked
measurement, or nonlinearly on its symbol, raises (`NotImplementedError`
for the first and third cases; `cvzx.exceptions.UnboundMeasurementError`
for the second).

Not (yet) supported -- raises `NotImplementedError`
-----------------------------------------------------
- `ControlledSumGate` and `CubicPhaseGate` (the latter is a genuinely
  non-Gaussian gate; mqc3's intrinsic set is Gaussian-only).
- Any state/effect leaf with a nonzero phase polynomial, other than the
  single measurement-effect shape described above.
- A symbolic (parametric, unresolved) gate parameter that isn't a
  tracked feedforward symbol as described above (a genuinely free,
  measurement-unrelated symbol; more than one symbol at once; a
  nonlinear function of one symbol).
- A `Diagram` with `num_inputs != 0`: mqc3 `CircuitRepr` has no concept
  of an externally supplied input mode -- every mode must originate
  from a state leaf.
- `ContractedDiagram` anywhere in `diagram` (same limitation
  `normalize_diagram` itself documents).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from sympy import Expr, Symbol, im, re, sqrt

from cvzx.base_gates import (
    CompositionDiagram,
    ContractedDiagram,
    Diagram,
    Fourier,
    Fourier2,
    FourierInv,
    PSpider,
    QSpider,
    TensorDiagram,
    VoidDiagram,
)
from cvzx.exceptions import UnboundMeasurementError
from cvzx.gates import (
    ArbitraryGate,
    BeamsplitterGate,
    ControlledZGate,
    DisplacementGate,
    MeasurementGate,
    PhaseRotationGate,
    ShearPInvariantGate,
    ShearXInvariantGate,
    Squeezing45Gate,
    SqueezingGate,
    TwoModeShearGate,
)
from cvzx.normalize_diagram import normalize_diagram

if TYPE_CHECKING:
    from mqc3.circuit import CircuitRepr
    from mqc3.circuit.ops._base import MeasuredVariable, Operation
    from mqc3.circuit.state import InitialState
    from mqc3.feedforward import FeedForward

__all__ = ["to_circuit_repr"]

# Every open output port `completion.complete_diagram()` closes is a
# (1, 0) Q/P-spider effect with phase exactly `-m*x` for a fresh symbol
# `m`: a `MeasurementOps` dict tracks, while walking a diagram, which
# mqc3 `Operation` each such symbol's measurement turned into, so a later
# leaf whose own parameter mentions that same symbol can be translated
# into an mqc3 `FeedForward` referencing it instead of being rejected.
MeasurementOps = dict[Symbol, "Operation"]


# --------------------------------------------------------------------------
# Numeric coercion
# --------------------------------------------------------------------------


def _as_real(value: complex | Expr, leaf_name: str) -> float:
    """Coerce a gate parameter to `float`, rejecting unresolved symbols.

    Returns
    -------
    float
        `value`, coerced.

    Raises
    ------
    NotImplementedError
        If `value` is a symbolic (unresolved) `Expr`.
    """
    if isinstance(value, Expr) and value.free_symbols:
        msg = (
            f"Cannot convert `{leaf_name}`: symbolic (unresolved parametric) gate "
            "parameters are not supported by the Diagram -> CircuitRepr converter "
            "-- substitute concrete values first (see `substitute_parameters`)."
        )
        raise NotImplementedError(msg)
    if isinstance(value, complex):
        return value.real
    return float(value)


def _as_complex(value: complex | Expr, leaf_name: str) -> complex:
    """Coerce a gate parameter to `complex`, rejecting unresolved symbols.

    Returns
    -------
    complex
        `value`, coerced.

    Raises
    ------
    NotImplementedError
        If `value` is a symbolic (unresolved) `Expr`.
    """
    if isinstance(value, Expr) and value.free_symbols:
        msg = (
            f"Cannot convert `{leaf_name}`: symbolic (unresolved parametric) gate "
            "parameters are not supported by the Diagram -> CircuitRepr converter "
            "-- substitute concrete values first (see `substitute_parameters`)."
        )
        raise NotImplementedError(msg)
    return complex(value)


def _affine_coeffs(expr: Expr, symbol: Symbol) -> tuple[float, float] | None:
    """Return `(slope, intercept)` if `expr` is affine (degree <= 1) in `symbol` alone.

    Returns
    -------
    tuple[float, float] | None
        `None` if `expr` isn't affine in exactly `symbol` (it mentions
        other free symbols, or has degree > 1 in `symbol`).
    """
    if expr.free_symbols != {symbol}:
        return None
    poly = expr.as_poly(symbol)
    if poly is None or poly.degree() > 1:
        return None
    intercept = float(poly.eval(0))
    slope = float(poly.eval(1)) - intercept
    return slope, intercept


def _resolve_scalar(
    value: float | Expr, leaf_name: str, measurement_ops: MeasurementOps
) -> float | FeedForward[MeasuredVariable]:
    """Coerce a gate parameter to `float`, or to an mqc3 `FeedForward` if measurement-dependent.

    A plain numeric value (or a symbol-free `Expr`) is coerced exactly as
    `_as_real` does. A value that's affine in exactly one symbol bound in
    `measurement_ops` (see `MeasurementOps`) becomes
    `FeedForward(MeasuredVariable(op))`, scaled/shifted via
    `mqc3.feedforward.ff_to_mul_constant`/`ff_to_add_constant` to match
    that affine relationship -- mqc3 has no arithmetic operators on
    `FeedForward` itself, so the scaling has to be baked in this way
    rather than applied to the returned value afterward.

    Returns
    -------
    float | FeedForward[MeasuredVariable]

    Raises
    ------
    NotImplementedError
        If `value` depends on more than one symbol, or isn't affine in
        its single symbol.
    UnboundMeasurementError
        If `value`'s symbol isn't bound to any measurement encountered so
        far while walking this diagram (no matching completion effect --
        a `QSpider`/`PSpider(1, 0, ZxPoly({1: -symbol}))` leaf -- was
        found upstream of this one).
    """
    if not (isinstance(value, Expr) and value.free_symbols):
        return _as_real(value, leaf_name)

    free = value.free_symbols
    if len(free) != 1:
        msg = (
            f"Cannot convert `{leaf_name}`: feedforward depending on more than one "
            "measurement symbol at once is not supported."
        )
        raise NotImplementedError(msg)
    (symbol,) = free
    if symbol not in measurement_ops:
        msg = (
            f"Cannot convert `{leaf_name}`: symbol {symbol!r} is not bound to any "
            "measurement encountered so far in this diagram (no matching completion "
            "effect -- QSpider/PSpider(1, 0, ZxPoly({1: -symbol})) -- found upstream)."
        )
        raise UnboundMeasurementError(msg)
    affine = _affine_coeffs(value, symbol)
    if affine is None:
        msg = (
            f"Cannot convert `{leaf_name}`: feedforward parameter {value} is not affine "
            f"(degree <= 1) in its measurement symbol {symbol!r}."
        )
        raise NotImplementedError(msg)
    slope, intercept = affine

    from mqc3.circuit.ops._base import MeasuredVariable  # ruff: ignore[import-outside-top-level, import-private-name]
    from mqc3.feedforward import ff_to_add_constant, ff_to_mul_constant  # ruff: ignore[import-outside-top-level]

    result = ff_to_mul_constant(slope)(MeasuredVariable(measurement_ops[symbol]))
    if intercept != 0:
        result = ff_to_add_constant(intercept)(result)
    return result


# --------------------------------------------------------------------------
# Leaf recognition
# --------------------------------------------------------------------------


def _is_identity(elt: Diagram) -> bool:
    """A bare 1-in-1-out zero-phase Q/P-spider, or a same-shaped Void: a no-op pass-through wire.

    A `VoidDiagram(1, 1)` arises whenever `_install_void_placeholder`
    same-shape-swaps a plain (1,1) gate (see `TerminalAbsorptionRule`'s
    and `CopyRule`'s cross-container substitutions); functionally it is
    exactly an identity wire, since `optimize()` no longer strips these
    out itself.

    Returns
    -------
    bool
        True if `elt` is such an identity leaf.
    """
    if isinstance(elt, VoidDiagram):
        return elt.num_inputs == 1 and elt.num_outputs == 1
    return bool(
        isinstance(elt, (QSpider, PSpider)) and elt.num_inputs == 1 and elt.num_outputs == 1 and elt.phase.is_zero
    )


def _is_zero_phase_leaf(elt: Diagram, num_inputs: int, num_outputs: int) -> bool:
    return (
        isinstance(elt, (QSpider, PSpider))
        and elt.num_inputs == num_inputs
        and elt.num_outputs == num_outputs
        and elt.phase.is_zero
    )


def _measurement_symbol(elt: Diagram) -> Symbol | None:
    """Return `m` if `elt` is a `(1, 0)` Q/P-spider effect with phase exactly `-m*x`.

    This is the exact leaf shape `cvzx.completion.complete_diagram()`
    appends to close an open output port: the standard CV-ZX notation for
    "the idealized homodyne effect whose own outcome is `m`" (see that
    module's docstring). Recognized purely structurally -- no
    `param_measurement_map` entry is expected on this leaf itself, since
    it's the symbol's *origin*, not something depending on it.

    Returns
    -------
    Symbol | None
        `m`, or `None` if `elt` isn't exactly this shape.
    """
    if not (isinstance(elt, (QSpider, PSpider)) and elt.num_inputs == 1 and elt.num_outputs == 0):
        return None
    coeffs = elt.phase.coeffs
    if set(coeffs) != {1}:
        return None
    coeff = coeffs[1]
    if not (isinstance(coeff, Expr) and len(coeff.free_symbols) == 1):
        return None
    (symbol,) = coeff.free_symbols
    return symbol if coeff == -symbol else None


# --------------------------------------------------------------------------
# Single-leaf translators
# --------------------------------------------------------------------------


def _open_mode_state(elt: Diagram, mode_index: int) -> InitialState:
    """Translate a 0-in-1-out state leaf into an mqc3 `InitialState`.

    Returns
    -------
    InitialState
        The translated state.

    Raises
    ------
    NotImplementedError
        If `elt` is not the idealized zero-phase Q/P-spider state.
    """
    from mqc3.circuit.state import HardwareConstrainedSqueezedState  # ruff: ignore[import-outside-top-level]

    if _is_zero_phase_leaf(elt, 0, 1):
        phi = 0.0 if isinstance(elt, QSpider) else np.pi / 2
        return HardwareConstrainedSqueezedState(phi=phi)
    msg = (
        f"Cannot convert the state leaf opening mode {mode_index} "
        f"({type(elt).__name__}): only the idealized zero-phase Q/P-spider "
        "state is supported by the Diagram -> CircuitRepr converter."
    )
    raise NotImplementedError(msg)


def _apply_1mode_leaf(  # ruff: ignore[complex-structure, too-many-branches, too-many-return-statements]
    circuit: CircuitRepr,
    mode_id: int,
    elt: Diagram,
    measurement_ops: MeasurementOps,
) -> bool:
    """Apply the mqc3 op for one already-open mode's 1-in leaf.

    `measurement_ops` is updated in place whenever `elt` is a
    `_measurement_symbol()`-shaped effect (see `MeasurementOps`), so any
    later leaf mentioning that same symbol can be translated as a
    `FeedForward` instead of being rejected.

    Returns
    -------
    bool
        True if the mode survives (the leaf has an output -- a gate or
        an identity wire); False if the leaf consumes it (an effect).
    """
    # ruff: ignore[import-outside-top-level]
    from mqc3.circuit.ops import intrinsic

    if _is_identity(elt):
        return True

    if isinstance(elt, VoidDiagram):
        # A (1, 0)-shaped Void: installed in place of a vanished effect
        # (see `_install_void_placeholder`) -- there is nothing there to
        # measure, so the mode is silently discarded rather than
        # emitting any op.
        return False

    if isinstance(elt, MeasurementGate):
        circuit.Q(mode_id) | intrinsic.Measurement(_resolve_scalar(elt.theta, "MeasurementGate", measurement_ops))
        return False
    if _is_zero_phase_leaf(elt, 1, 0):
        theta = np.pi / 2 if isinstance(elt, QSpider) else 0.0
        circuit.Q(mode_id) | intrinsic.Measurement(theta)
        return False
    symbol = _measurement_symbol(elt)
    if symbol is not None:
        theta = np.pi / 2 if isinstance(elt, QSpider) else 0.0
        op = intrinsic.Measurement(theta)
        circuit.Q(mode_id) | op
        measurement_ops[symbol] = op
        return False

    if isinstance(elt, PhaseRotationGate):
        raw_theta = _resolve_scalar(elt.theta, "PhaseRotationGate", measurement_ops)
        neg_theta = -raw_theta if isinstance(raw_theta, float) else _negate_feedforward(raw_theta)
        circuit.Q(mode_id) | intrinsic.PhaseRotation(neg_theta)
        return True
    if isinstance(elt, Fourier):
        circuit.Q(mode_id) | intrinsic.PhaseRotation(np.pi / 2)
        return True
    if isinstance(elt, FourierInv):
        circuit.Q(mode_id) | intrinsic.PhaseRotation(-np.pi / 2)
        return True
    if isinstance(elt, Fourier2):
        circuit.Q(mode_id) | intrinsic.PhaseRotation(np.pi)
        return True
    if isinstance(elt, ShearXInvariantGate):
        kappa = _resolve_scalar(elt.kappa, "ShearXInvariantGate", measurement_ops)
        circuit.Q(mode_id) | intrinsic.ShearXInvariant(kappa)
        return True
    if isinstance(elt, ShearPInvariantGate):
        eta = _resolve_scalar(elt.eta, "ShearPInvariantGate", measurement_ops)
        circuit.Q(mode_id) | intrinsic.ShearPInvariant(eta)
        return True
    if isinstance(elt, SqueezingGate):
        tau = _as_real(elt.tau, "SqueezingGate")
        circuit.Q(mode_id) | intrinsic.Arbitrary(0.0, 0.0, float(np.log(tau)))
        return True
    if isinstance(elt, ArbitraryGate):
        circuit.Q(mode_id) | intrinsic.Arbitrary(
            _resolve_scalar(elt.alpha, "ArbitraryGate", measurement_ops),
            _resolve_scalar(elt.beta, "ArbitraryGate", measurement_ops),
            _resolve_scalar(elt.lam, "ArbitraryGate", measurement_ops),
        )
        return True
    if isinstance(elt, Squeezing45Gate):
        circuit.Q(mode_id) | intrinsic.Squeezing45(_resolve_scalar(elt.theta, "Squeezing45Gate", measurement_ops))
        return True
    if isinstance(elt, DisplacementGate):
        x_expr = sqrt(2) * re(elt.alpha) if isinstance(elt.alpha, Expr) else sqrt(2) * elt.alpha.real
        p_expr = sqrt(2) * im(elt.alpha) if isinstance(elt.alpha, Expr) else sqrt(2) * elt.alpha.imag
        circuit.Q(mode_id) | intrinsic.Displacement(
            _resolve_scalar(x_expr, "DisplacementGate", measurement_ops),
            _resolve_scalar(p_expr, "DisplacementGate", measurement_ops),
        )
        return True

    msg = f"Cannot convert 1-mode leaf `{type(elt).__name__}`: no CircuitRepr translation is registered for it."
    raise NotImplementedError(msg)


def _negate_feedforward(value: FeedForward[MeasuredVariable]) -> FeedForward[MeasuredVariable]:
    """Negate a `FeedForward` value (no arithmetic operators exist on it directly).

    Returns
    -------
    FeedForward[MeasuredVariable]
    """
    from mqc3.feedforward import ff_to_mul_constant  # ruff: ignore[import-outside-top-level]

    return cast("FeedForward[MeasuredVariable]", ff_to_mul_constant(-1.0)(value))


def _apply_2mode_leaf(
    circuit: CircuitRepr, mode_a: int, mode_b: int, elt: Diagram, measurement_ops: MeasurementOps
) -> None:
    """Apply the mqc3 op for a wide (2-mode) leaf touching `mode_a`, `mode_b`."""
    # ruff: ignore[import-outside-top-level]
    from mqc3.circuit.ops import intrinsic

    if isinstance(elt, ControlledZGate):
        gain = _resolve_scalar(elt.gain, "ControlledZGate", measurement_ops)
        gain = -gain if isinstance(gain, float) else _negate_feedforward(gain)
        circuit.Q(mode_a, mode_b) | intrinsic.ControlledZ(gain)
        return
    if isinstance(elt, BeamsplitterGate):
        theta = _as_real(elt.theta, "BeamsplitterGate")
        circuit.Q(mode_a, mode_b) | intrinsic.BeamSplitter(float(np.cos(theta)), 0.0)
        return
    if isinstance(elt, TwoModeShearGate):
        circuit.Q(mode_a, mode_b) | intrinsic.TwoModeShear(
            _resolve_scalar(elt.a, "TwoModeShearGate", measurement_ops),
            _resolve_scalar(elt.b, "TwoModeShearGate", measurement_ops),
        )
        return

    msg = f"Cannot convert 2-mode leaf `{type(elt).__name__}`: no CircuitRepr translation is registered for it."
    raise NotImplementedError(msg)


# --------------------------------------------------------------------------
# Row / stage walking
# --------------------------------------------------------------------------


class _ModeCounter:
    """Hands out fresh, strictly increasing mqc3 mode ids."""

    def __init__(self) -> None:
        self._next = 0

    def fresh(self) -> int:
        mode_id = self._next
        self._next += 1
        return mode_id


def _walk_row(
    circuit: CircuitRepr,
    row: Diagram,
    input_modes: list[int],
    mode_counter: _ModeCounter,
    measurement_ops: MeasurementOps,
) -> list[int]:
    """Apply one row's content and return its surviving output mode id(s).

    A row is either a single leaf, or (for a 1-mode row only --
    `normalize_diagram` never nests a wide leaf inside a per-row
    `CompositionDiagram`, see the module docstring) a chain of 1-mode
    leaves threaded together on the same mode.

    Returns
    -------
    list[int]
        `[mode_a, mode_b]` for a wide leaf; the single surviving mode
        id for a 1-mode chain that did not end in an effect; `[]` if
        the chain's last leaf consumed its mode.
    """
    if row.num_inputs > 1 or row.num_outputs > 1:
        # A standalone wide (2-mode) leaf.
        mode_a, mode_b = input_modes
        _apply_2mode_leaf(circuit, mode_a, mode_b, row, measurement_ops)
        return [mode_a, mode_b]

    elements = row.diagrams if isinstance(row, CompositionDiagram) else [row]
    mode_id = input_modes[0] if input_modes else None

    for elt in elements:
        if isinstance(elt, VoidDiagram) and elt.num_inputs == 0:
            # A (0, 1)-shaped Void: installed in place of a vanished
            # state (see `_install_void_placeholder`) -- there is
            # nothing there to originate a mode from, so none is opened;
            # this row simply contributes no output.
            mode_id = None
        elif elt.num_inputs == 0:
            mode_id = mode_counter.fresh()
            circuit.Q(mode_id)
            circuit.set_initial_state(mode_id, _open_mode_state(elt, mode_id))
        else:
            # A row's non-first element always has an input port -- only
            # the first element of a chain can be a 0-in state leaf -- so
            # `mode_id` was already set either from `input_modes` or by a
            # prior iteration of this same loop.
            assert mode_id is not None  # ruff: ignore[assert]
            survives = _apply_1mode_leaf(circuit, mode_id, elt, measurement_ops)
            if not survives:
                mode_id = None

    return [mode_id] if mode_id is not None else []


def to_circuit_repr(diagram: Diagram, *, name: str = "converted") -> CircuitRepr:
    """Translate a cvzx `Diagram` into an mqc3 `CircuitRepr`.

    `diagram` is canonicalized via `normalize_diagram` first (a no-op if
    it already is canonical), then walked stage by stage -- each stage's
    rows are processed left to right, each row's own leaf (or chain of
    1-mode leaves) is translated independently (see the module docstring
    for the exact per-leaf formulas), and mode ids are threaded across
    stage boundaries via the diagram's own `connectivity` dicts.

    Parameters
    ----------
    diagram : Diagram
        The cvzx diagram to convert. Any compact-form or already-
        canonical `Diagram` (not mutated).
    name : str
        Name for the resulting `CircuitRepr`.

    Returns
    -------
    CircuitRepr
        The translated circuit. Raises `NotImplementedError` if
        `diagram` contains a leaf, state, or effect this module does
        not recognize (see the module docstring), or a `ContractedDiagram`
        anywhere.

    Raises
    ------
    ValueError
        If `diagram` has any external inputs (`num_inputs != 0`): mqc3
        circuits have no notion of an externally supplied mode.
    """
    # ruff: ignore[import-outside-top-level]
    from mqc3.circuit import CircuitRepr

    def _reject_contracted(d: Diagram) -> None:
        if isinstance(d, ContractedDiagram):
            msg = (
                "Cannot convert a Diagram containing a ContractedDiagram "
                "(out of scope, same as normalize_diagram's own limitation)."
            )
            raise NotImplementedError(msg)
        for child in getattr(d, "diagrams", []):
            _reject_contracted(child)

    _reject_contracted(diagram)
    diagram = normalize_diagram(diagram)

    if diagram.num_inputs != 0:
        msg = (
            f"Cannot convert a Diagram with external inputs (num_inputs={diagram.num_inputs}) "
            "to a CircuitRepr: mqc3 circuits have no concept of an externally supplied "
            "input mode -- every mode must originate from a state leaf (an ancilla/InitialState)."
        )
        raise ValueError(msg)

    circuit = CircuitRepr(name)
    stages = diagram.diagrams if isinstance(diagram, CompositionDiagram) else [diagram]
    connectivity = diagram.connectivity if isinstance(diagram, CompositionDiagram) else {}

    mode_counter = _ModeCounter()
    measurement_ops: MeasurementOps = {}
    prev_output_modes: list[int] = []

    for stage_index, stage in enumerate(stages):
        if stage_index == 0:
            active_modes: list[int] = []
        else:
            conn = connectivity.get(stage_index - 1, {})
            active_modes = [prev_output_modes[conn[k]] for k in range(len(conn))]

        rows = stage.diagrams if isinstance(stage, TensorDiagram) else [stage]

        new_outputs: list[int] = []
        pos = 0
        for row in rows:
            n_in = row.num_inputs
            row_inputs = active_modes[pos : pos + n_in]
            pos += n_in
            new_outputs.extend(_walk_row(circuit, row, row_inputs, mode_counter, measurement_ops))

        prev_output_modes = new_outputs

    return circuit

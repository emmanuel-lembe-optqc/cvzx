"""Bidirectional bridge between mqc3 `CircuitRepr` and cvzx `Diagram`.

`from_circuit_repr` converts an mqc3 `CircuitRepr` into a canonical cvzx `Diagram`;
`to_circuit_repr` is the reverse direction. The two are algebraic inverses of each
other throughout. See :doc:`../user_guide/circuit_conversion` for the exact
per-gate/per-leaf formulas, feedforward handling, and the full
not-yet-supported list -- only the key steps are kept here.

`CircuitRepr` -> `Diagram`
---------------------------
`from_circuit_repr` implements a "naive translate, then normalize" architecture:

1. `_naive_translate` walks the mqc3 `CircuitRepr` in time order (via
   `CircuitRepr.__iter__`, after `convert_std_ops_to_intrinsic()` has
   lowered every `std.*` operation down to `intrinsic.*` ones) and builds
   *some* compact-form cvzx `Diagram` that is semantically equivalent to
   the circuit -- one flat `CompositionDiagram` of per-operation
   `TensorDiagram` layers, threaded together with explicit `connectivity`
   dicts so that a 2-mode operation touching two non-adjacent rows never
   needs an explicit `Swap`. This is exactly the kind of
   "natural"/arbitrarily-shaped diagram `normalize_diagram` (see
   `cvzx.passes.normalize`) is designed to consume.
2. `from_circuit_repr` (the public entry point) calls `_naive_translate`
   and then `normalize_diagram` on the result, returning the canonical
   alternating type-1/type-2 form.

Every mode's `InitialState` becomes an idealized (infinitely squeezed) CV-ZX
state leaf, optionally rotated to set the squeezing axis. A measurement some
later operation's `FeedForward` depends on is reconstructed as a
`QSpider`/`PSpider` effect carrying that dependency in its own
`param_measurement_map`, and the referencing operation's parameter is
recovered from the feedforward function's affine coefficients -- the
algebraic inverse of `to_circuit_repr`'s own feedforward resolution below.

`Diagram` -> `CircuitRepr`
---------------------------
`to_circuit_repr` is the reverse direction: it walks an already-canonical
(or canonicalizable) `Diagram` -- the alternating type-1/type-2 stages
`normalize_diagram` produces -- and emits the equivalent sequence of mqc3
intrinsic operations, translating each primitive leaf independently rather
than pattern-matching compositions back into a single composite op (e.g. a
round-tripped `BeamSplitter` is not necessarily *op-for-op* identical to the
original, just semantically equivalent -- see the user guide for a worked
example). The input `Diagram` must not already be the target of
`optimize()`'s algebraic spider-fusion rewrites (as opposed to
`normalize_diagram`'s purely structural ones): a fused spider's phase
polynomial may no longer match any primitive shape this module recognizes.

A measurement-effect leaf with the feedforward shape above becomes a plain
`intrinsic.Measurement` and is tracked against its symbol for the rest of
the walk; any later leaf whose parameter is affine in a tracked symbol is
re-emitted as an mqc3 `FeedForward[MeasuredVariable]` instead of a plain
float -- the algebraic inverse of `from_circuit_repr`'s reconstruction above.

Not (yet) supported -- raises `NotImplementedError`
-----------------------------------------------------
- `CubicPhaseGate`: a genuinely non-Gaussian gate; mqc3's intrinsic set
  is Gaussian-only.
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
- `intrinsic.Manual` (reached directly, or indirectly via a `std.*`
  operation that lowers to it) has no known CV-ZX decomposition yet.
"""

from __future__ import annotations

import copy
from itertools import count
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
from sympy import Expr, I, Symbol, acos, im, re, sqrt, tan

from cvzx.exceptions import UnboundMeasurementError
from cvzx.ir.base import (
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
    ZxPoly,
)
from cvzx.ir.gates import (
    ArbitraryGate,
    BeamsplitterGate,
    ControlledSumGate,
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
from cvzx.passes.normalize import normalize_diagram

if TYPE_CHECKING:
    from collections.abc import Callable

    from mqc3.circuit import CircuitRepr
    from mqc3.circuit.ops._base import MeasuredVariable, Operation
    from mqc3.circuit.program import CircOpParam
    from mqc3.circuit.state import InitialState
    from mqc3.feedforward import FeedForward

__all__ = ["from_circuit_repr", "to_circuit_repr"]

_symbol_counter = count(1)


# --------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------


def _identity_wire() -> Diagram:
    """A bare 1-in-1-out wire (see `normalize_diagram`'s `IdentityRule`).

    Returns
    -------
    Diagram
        A fresh zero-phase `QSpider(1, 1)`.
    """
    return QSpider(1, 1, ZxPoly({}))


def _is_symbolic(value: float | Expr) -> bool:
    """True if `value` is a genuinely symbolic (unresolved) sympy `Expr`.

    Returns
    -------
    bool
    """
    return isinstance(value, Expr) and bool(value.free_symbols)


def _pmm_for(*exprs: float | Expr, measurement_leaf_ids: dict[Symbol, int]) -> dict[Symbol, set[int]]:
    """Build a `param_measurement_map` entry set from one or more parameter expressions.

    Each `expr`'s free symbols that are keys of `measurement_leaf_ids` (i.e.
    trace back to a reconstructed measurement leaf -- see
    `_measurements_needing_symbols`/`_translate_measurement`) contribute
    `{symbol: {measurement_leaf_ids[symbol]}}`, unioned across every `expr`
    passed in -- so a gate with more than one of its own parameters (e.g.
    `ArbitraryGate`, `TwoModeShearGate`) gets one call covering all of them.
    A symbol not in `measurement_leaf_ids` is a plain (non-feedforward)
    parameter and contributes nothing.

    Returns
    -------
    dict[Symbol, set[int]]
    """
    merged: dict[Symbol, set[int]] = {}
    for expr in exprs:
        if not isinstance(expr, Expr):
            continue
        for symbol in expr.free_symbols:
            if symbol in measurement_leaf_ids:
                merged.setdefault(symbol, set()).add(measurement_leaf_ids[symbol])
    return merged


def _phase_rotation(theta: float | Expr, measurement_leaf_ids: dict[Symbol, int] | None = None) -> Diagram:
    """`PhaseRotationGate(theta)`, routing around its odd-multiple-of-pi/2 restriction.

    `PhaseRotationGate` refuses any angle that is an odd multiple of
    `pi/2` (it asks callers to use `Fourier`/`FourierInv` instead). Every
    translator in this module that builds a `PhaseRotationGate` from an
    mqc3 angle can land on exactly such an angle (e.g. `intrinsic.
    Squeezing`'s fixed `pi/2` factor, or `intrinsic.PhaseRotation(pi/2)`
    itself), so this helper folds `theta` into `(-pi, pi]` and
    substitutes the equivalent `Fourier`/`FourierInv` proper diagram at
    the two problem angles -- the exact same routing already used by
    `MeasurementGate._rotation_diagram` in `cvzx.ir.gates` (see its
    docstring for the `Fourier() = mqc3 R(pi/2)`, `FourierInv() = mqc3
    R(-pi/2)` derivation this relies on).

    A symbolic (feedforward-derived) `theta` skips the fold entirely --
    it's a numeric-only optimization for landing exactly on the two
    problem angles, not a correctness requirement, and there's no way to
    tell whether an unresolved symbolic angle will land there.

    Parameters
    ----------
    theta : float | Expr
        The rotation angle, possibly a feedforward-derived symbolic `Expr`.
    measurement_leaf_ids : dict[Symbol, int] | None
        Passed through to `_pmm_for` when `theta` is symbolic, to populate
        the resulting `PhaseRotationGate`'s `param_measurement_map`.

    Returns
    -------
    Diagram
        `PhaseRotationGate(-theta, parametric=True)` if `theta` is
        symbolic; otherwise `Fourier()`/`FourierInv()` at the two problem
        angles, else `PhaseRotationGate(folded)`.
    """
    if _is_symbolic(theta):
        pmm = _pmm_for(theta, measurement_leaf_ids=measurement_leaf_ids or {})
        return PhaseRotationGate(theta, parametric=True, param_measurement_map=pmm)
    folded = ((theta + np.pi) % (2 * np.pi)) - np.pi
    if np.isclose(folded, np.pi / 2):
        return FourierInv()
    if np.isclose(folded, -np.pi / 2):
        return Fourier()
    return PhaseRotationGate(folded)


def _rotated_ideal_state(phi: float) -> Diagram:
    """Idealized (infinitely squeezed) state, rotated to squeezing axis `phi`.

    `phi = 0` is the bare x-eigenstate spider `QSpider(0, 1, 0)`; any other
    angle applies `PhaseRotationGate(-phi)` afterward (`state` is a ket,
    not a Heisenberg-conjugated operator, so the diagram's own
    `PhaseRotationGate(theta) = mqc3 R(-theta)` convention rotates the
    state's phase-space picture by mqc3's `R(-theta)` acting directly on
    the ket, i.e. by `-theta`; to rotate the state to angle `phi` we
    therefore need mqc3's `R(phi)`, i.e. `PhaseRotationGate(-phi)`).

    Returns
    -------
    Diagram
        The bare state spider at `phi = 0`, otherwise the state
        composed with the rotation that sets its squeezing axis.
    """
    base = QSpider(0, 1, ZxPoly({}))
    if phi == 0:
        return base
    return CompositionDiagram([base, _phase_rotation(-phi)])


def _phi_from_gaussian_cov(cov: npt.ArrayLike) -> float:
    """Recover the squeezing axis angle from a pure single-mode covariance matrix.

    Returns
    -------
    float
        The squeezing axis angle.

    Raises
    ------
    NotImplementedError
        If `cov` is not (numerically) a valid pure-state covariance
        matrix (symmetric, `det(cov) == 0.25` for `hbar = 1`).
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2) or not np.allclose(cov, cov.T, atol=1e-8):
        msg = "Cannot convert initial GaussianState: covariance must be a symmetric 2x2 matrix."
        raise NotImplementedError(msg)
    if not np.isclose(np.linalg.det(cov), 0.25, atol=1e-6):
        msg = (
            "Cannot convert initial GaussianState: only pure (minimum-uncertainty) "
            "states are supported by the CircuitRepr -> Diagram converter."
        )
        raise NotImplementedError(msg)
    eigvals, eigvecs = np.linalg.eigh(cov)
    squeezed_axis = eigvecs[:, int(np.argmin(eigvals))]
    return float(np.arctan2(squeezed_axis[1], squeezed_axis[0]))


def _translate_initial_state(state: InitialState, mode_index: int) -> Diagram:
    """Translate one mqc3 `InitialState` into a cvzx state leaf.

    Returns
    -------
    Diagram
        The translated state leaf.

    Raises
    ------
    NotImplementedError
        If `state` is outside the supported set (see the module
        docstring's "Initial states" section).
    """
    # ruff: ignore[import-outside-top-level]
    from mqc3.circuit.state import BosonicState, HardwareConstrainedSqueezedState

    if isinstance(state, HardwareConstrainedSqueezedState):
        return _rotated_ideal_state(state.phi)

    if isinstance(state, BosonicState):
        if len(state.coeffs) != 1 or len(state.gaussian_states) != 1:
            msg = (
                f"Cannot convert the initial state of mode {mode_index}: multi-peak "
                "(non-Gaussian superposition) BosonicState initial states are not "
                "yet supported by the CircuitRepr -> Diagram converter."
            )
            raise NotImplementedError(msg)
        gaussian_state = state.gaussian_states[0]
        mean = np.asarray(gaussian_state.mean, dtype=complex)
        if not np.allclose(mean, 0, atol=1e-8):
            msg = (
                f"Cannot convert the initial state of mode {mode_index}: displaced "
                "(nonzero-mean) initial states are not yet supported by the "
                "CircuitRepr -> Diagram converter -- use an `intrinsic.Displacement` "
                "operation instead."
            )
            raise NotImplementedError(msg)
        phi = _phi_from_gaussian_cov(gaussian_state.cov)
        return _rotated_ideal_state(phi)

    msg = f"Cannot convert the initial state of mode {mode_index}: unsupported state type {type(state)!r}."
    raise NotImplementedError(msg)


def _as_float(value: CircOpParam, op_name: str, param_index: int) -> float:
    """Reject feedforward parameters (not resolvable here) and coerce to `float`.

    Returns
    -------
    float
        `value`, coerced.

    Raises
    ------
    NotImplementedError
        If `value` is a `FeedForward`.
    """
    # ruff: ignore[import-outside-top-level]
    from mqc3.feedforward import FeedForward

    if isinstance(value, FeedForward):
        msg = f"Cannot convert `{op_name}`: feedforward on parameter #{param_index} is not a plain float."
        raise NotImplementedError(msg)
    return float(value)


def _resolve_param(
    value: CircOpParam, op_name: str, param_index: int, measurement_symbols: dict[int, Symbol]
) -> float | Expr:
    """Coerce a `CircOpParam` to `float`, or to a symbolic `slope*symbol + intercept` `Expr`.

    A plain float coerces exactly as `_as_float` does. A `FeedForward`
    referencing a measurement operation that `_measurements_needing_symbols`
    assigned a symbol to resolves to that affine relationship (the
    `FeedForwardFunction`'s slope/intercept are recovered by evaluating it
    numerically at `0.0` and `1.0`, mirroring `cvzx.lowering.bridges.mqc3`'s
    own `_resolve_scalar` in reverse); the resulting `Expr` is meant to be
    threaded into `param_measurement_map={symbol: {measurement_leaf.id}}`
    on the cvzx gate that ends up carrying it (see `_naive_translate`).

    Returns
    -------
    float | Expr

    Raises
    ------
    NotImplementedError
        If `value` is a `FeedForward` referencing an operation that
        wasn't recognized as a measurement needing a symbol (should not
        occur for a `value` produced by iterating `circuit`, since
        `_measurements_needing_symbols` scans every `FeedForward` in the
        same circuit up front).
    """
    # ruff: ignore[import-outside-top-level]
    from mqc3.feedforward import FeedForward

    if not isinstance(value, FeedForward):
        return _as_float(value, op_name, param_index)

    source_id = id(value.variable.get_from_operation())
    symbol = measurement_symbols.get(source_id)
    if symbol is None:
        msg = (
            f"Cannot convert `{op_name}`: feedforward on parameter #{param_index} references "
            "an operation that wasn't recognized as a measurement (should not occur)."
        )
        raise NotImplementedError(msg)

    # `FeedForwardFunction.__call__` returns a plain `float` when called
    # with a `float` (its own documented behavior); the `float | FeedForward`
    # return annotation only matters for its other (Variable/FeedForward)
    # input cases.
    intercept = cast("float", value.func(0.0))
    slope = cast("float", value.func(1.0)) - intercept
    return slope * symbol + intercept


# --------------------------------------------------------------------------
# Per-operation translators
# --------------------------------------------------------------------------
# Each translator takes the operation's already-float-coerced parameters
# (in the exact order `Operation.parameters()` returns them -- see
# `mqc3.circuit.ops.intrinsic`) and returns the compact-form cvzx `Diagram`
# for that operation alone (a 1-mode or 2-mode gate/effect, in the same
# mode order as `Operation.opnd().get_ids()`).


def _translate_measurement(params: list[float], symbol: Symbol | None = None) -> Diagram:
    """Translate `intrinsic.Measurement(theta)`, or the measurement-effect leaf a symbol needs.

    If `symbol` is given (a later operation's `FeedForward` references
    this measurement -- see `_naive_translate`), and `theta` is (close
    to) the canonical x- or p-homodyne angle, reconstructs the exact leaf
    shape `cvzx.passes.completion.complete_diagram()` produces --
    `QSpider`/`PSpider(1, 0, ZxPoly({1: -symbol}))` -- instead of a plain
    `MeasurementGate(theta)`, so the symbol is available for
    `param_measurement_map` bindings on whatever downstream gate feeds
    forward from it. Any other angle falls back to `MeasurementGate`
    unconditionally: only the two canonical angles have a Q/P-spider
    representation to reconstruct.

    Returns
    -------
    Diagram
    """
    (theta,) = params
    if symbol is not None:
        if np.isclose(theta, np.pi / 2):
            return QSpider(1, 0, ZxPoly({1: -symbol}), parametric=True)
        if np.isclose(theta, 0.0):
            return PSpider(1, 0, ZxPoly({1: -symbol}), parametric=True)
    return MeasurementGate(theta)


def _translate_displacement(params: list[float | Expr], measurement_leaf_ids: dict[Symbol, int]) -> Diagram:
    x, p = params
    if _is_symbolic(x) or _is_symbolic(p):
        alpha_expr = (x + I * p) / sqrt(2)
        pmm = _pmm_for(alpha_expr, measurement_leaf_ids=measurement_leaf_ids)
        return DisplacementGate(alpha_expr, parametric=True, param_measurement_map=pmm)
    alpha = complex(x, p) / np.sqrt(2)
    return DisplacementGate(alpha)


def _translate_phase_rotation(params: list[float | Expr], measurement_leaf_ids: dict[Symbol, int]) -> Diagram:
    (phi,) = params
    return _phase_rotation(-phi, measurement_leaf_ids)


def _translate_shear_x_invariant(params: list[float | Expr], measurement_leaf_ids: dict[Symbol, int]) -> Diagram:
    (kappa,) = params
    pmm = _pmm_for(kappa, measurement_leaf_ids=measurement_leaf_ids)
    return ShearXInvariantGate(kappa, parametric=_is_symbolic(kappa), param_measurement_map=pmm)


def _translate_shear_p_invariant(params: list[float | Expr], measurement_leaf_ids: dict[Symbol, int]) -> Diagram:
    (eta,) = params
    pmm = _pmm_for(eta, measurement_leaf_ids=measurement_leaf_ids)
    return ShearPInvariantGate(eta, parametric=_is_symbolic(eta), param_measurement_map=pmm)


def _translate_squeezing(params: list[float | Expr], measurement_leaf_ids: dict[Symbol, int]) -> Diagram:
    (theta,) = params
    tau = tan(theta) if _is_symbolic(theta) else np.tan(theta)
    pmm = _pmm_for(tau, measurement_leaf_ids=measurement_leaf_ids)
    return CompositionDiagram([
        SqueezingGate(tau, parametric=_is_symbolic(theta), param_measurement_map=pmm),
        _phase_rotation(np.pi / 2),
    ])


def _translate_squeezing45(params: list[float | Expr], measurement_leaf_ids: dict[Symbol, int]) -> Diagram:
    (theta,) = params
    pmm = _pmm_for(theta, measurement_leaf_ids=measurement_leaf_ids)
    return Squeezing45Gate(theta, parametric=_is_symbolic(theta), param_measurement_map=pmm)


def _translate_arbitrary(params: list[float | Expr], measurement_leaf_ids: dict[Symbol, int]) -> Diagram:
    alpha, beta, lam = params
    parametric = _is_symbolic(alpha) or _is_symbolic(beta) or _is_symbolic(lam)
    pmm = _pmm_for(alpha, beta, lam, measurement_leaf_ids=measurement_leaf_ids)
    return ArbitraryGate(alpha, beta, lam, parametric=parametric, param_measurement_map=pmm)


def _translate_controlled_z(params: list[float | Expr], measurement_leaf_ids: dict[Symbol, int]) -> Diagram:
    (g,) = params
    pmm = _pmm_for(g, measurement_leaf_ids=measurement_leaf_ids)
    return ControlledZGate(gain=-g, parametric=_is_symbolic(g), param_measurement_map=pmm)


def _translate_beam_splitter(params: list[float | Expr], measurement_leaf_ids: dict[Symbol, int]) -> Diagram:
    sqrt_r, theta_rel = params
    eta = acos(sqrt_r) if _is_symbolic(sqrt_r) else np.arccos(sqrt_r)
    beam_pmm = _pmm_for(eta, measurement_leaf_ids=measurement_leaf_ids)
    return CompositionDiagram([
        TensorDiagram([_identity_wire(), _phase_rotation(-np.pi / 2)]),
        BeamsplitterGate(eta, parametric=_is_symbolic(eta), param_measurement_map=beam_pmm),
        TensorDiagram([
            _phase_rotation(-theta_rel, measurement_leaf_ids),
            _phase_rotation(np.pi / 2 - theta_rel, measurement_leaf_ids),
        ]),
    ])


def _translate_two_mode_shear(params: list[float | Expr], measurement_leaf_ids: dict[Symbol, int]) -> Diagram:
    a, b = params
    parametric = _is_symbolic(a) or _is_symbolic(b)
    pmm = _pmm_for(a, b, measurement_leaf_ids=measurement_leaf_ids)
    return TwoModeShearGate(a, b, parametric=parametric, param_measurement_map=pmm)


def _translate_manual(_params: list[float], _measurement_leaf_ids: dict[Symbol, int]) -> Diagram:
    msg = (
        "Cannot convert `intrinsic.manual`: the Manual gate's CV-ZX decomposition "
        "is not yet implemented (intentionally deferred). This can be reached "
        "either directly or via `convert_std_ops_to_intrinsic()` lowering a "
        "`std.*` operation -- e.g. `std.BeamSplitter` -- through `Manual`."
    )
    raise NotImplementedError(msg)


_TRANSLATORS: dict[str, Callable[..., Diagram]] = {
    "intrinsic.measurement": _translate_measurement,
    "intrinsic.displacement": _translate_displacement,
    "intrinsic.phase_rotation": _translate_phase_rotation,
    "intrinsic.shear_x_invariant": _translate_shear_x_invariant,
    "intrinsic.shear_p_invariant": _translate_shear_p_invariant,
    "intrinsic.squeezing": _translate_squeezing,
    "intrinsic.squeezing45": _translate_squeezing45,
    "intrinsic.arbitrary": _translate_arbitrary,
    "intrinsic.controlled_z": _translate_controlled_z,
    "intrinsic.beam_splitter": _translate_beam_splitter,
    "intrinsic.two_mode_shear": _translate_two_mode_shear,
    "intrinsic.manual": _translate_manual,
}


# --------------------------------------------------------------------------
# Naive translation
# --------------------------------------------------------------------------


def _measurements_needing_symbols(circuit: CircuitRepr) -> dict[int, Symbol]:
    """Find every `intrinsic.measurement` operation some later operation feeds forward from.

    A fresh `Symbol` is assigned to each such measurement's `id(operation)`
    (identity, not equality -- mqc3 `Operation` objects aren't meaningfully
    comparable by value) -- see `_translate_measurement`, which uses it to
    reconstruct the measurement as a `QSpider`/`PSpider(1, 0,
    ZxPoly({1: -symbol}))` leaf instead of a plain `MeasurementGate`.

    Returns
    -------
    dict[int, Symbol]
    """
    # ruff: ignore[import-outside-top-level]
    from mqc3.feedforward import FeedForward

    referenced_ids: set[int] = set()
    for op in circuit:
        for param in op.parameters():
            if isinstance(param, FeedForward):
                referenced_ids.add(id(param.variable.get_from_operation()))

    symbols: dict[int, Symbol] = {}
    for op in circuit:
        if op.name() == "intrinsic.measurement" and id(op) in referenced_ids:
            symbols[id(op)] = Symbol(f"m_{next(_symbol_counter)}", real=True)
    return symbols


def _record_measurement_leaf_id(
    op: Operation,
    gate_diagram: Diagram,
    measurement_symbols: dict[int, Symbol],
    measurement_leaf_ids: dict[Symbol, int],
) -> None:
    """Record a translated measurement's cvzx leaf id under its assigned symbol, if any.

    Mutates `measurement_leaf_ids` in place. A no-op unless `op` is an
    `intrinsic.measurement` that `_measurements_needing_symbols` assigned a
    symbol to AND `_translate_measurement` actually reconstructed it as a
    `QSpider`/`PSpider` leaf carrying that symbol (the `MeasurementGate`
    fallback, for a non-canonical angle, carries no symbol at all).
    """
    if op.name() != "intrinsic.measurement":
        return
    symbol = measurement_symbols.get(id(op))
    if symbol is not None and isinstance(gate_diagram, (QSpider, PSpider)):
        measurement_leaf_ids[symbol] = gate_diagram.id


def _translate_operation(
    op: Operation, measurement_symbols: dict[int, Symbol], measurement_leaf_ids: dict[Symbol, int]
) -> Diagram:
    name = op.name()
    translator = _TRANSLATORS.get(name)
    if translator is None:
        msg = f"Cannot convert operation `{name}`: no CircuitRepr -> Diagram translation is registered for it."
        raise NotImplementedError(msg)
    params = [_resolve_param(p, name, i, measurement_symbols) for i, p in enumerate(op.parameters())]
    if name == "intrinsic.measurement":
        return _translate_measurement(params, measurement_symbols.get(id(op)))
    return translator(params, measurement_leaf_ids)


def _naive_translate(circuit: CircuitRepr) -> Diagram:  # ruff: ignore[too-many-locals]
    """Build *some* compact-form `Diagram` semantically equal to `circuit`.

    See the module docstring for the overall strategy. `circuit` is not
    mutated -- a deep copy is lowered to intrinsic operations instead.

    Returns
    -------
    Diagram
        The naively translated (not yet canonicalized) diagram.

    Raises
    ------
    ValueError
        If `circuit` has no modes.
    """
    circuit = copy.deepcopy(circuit)
    circuit.convert_std_ops_to_intrinsic()

    n_modes = circuit.n_modes
    if n_modes == 0:
        msg = "Cannot convert an empty CircuitRepr (no modes)."
        raise ValueError(msg)

    state_leaves = [_translate_initial_state(circuit.get_initial_state(i), i) for i in range(n_modes)]
    diagrams: list[Diagram] = [TensorDiagram(state_leaves) if n_modes > 1 else state_leaves[0]]
    connectivity: dict[int, dict[int, int]] = {}
    open_modes: list[int] = list(range(n_modes))
    measurement_symbols = _measurements_needing_symbols(circuit)
    measurement_leaf_ids: dict[Symbol, int] = {}

    for op in circuit:
        gate_diagram = _translate_operation(op, measurement_symbols, measurement_leaf_ids)
        _record_measurement_leaf_id(op, gate_diagram, measurement_symbols, measurement_leaf_ids)
        touched = list(op.opnd().get_ids())
        other = [m for m in open_modes if m not in touched]
        new_row_order = [*touched, *other]

        rows: list[Diagram] = [gate_diagram, *(_identity_wire() for _ in other)]
        layer = TensorDiagram(rows) if len(rows) > 1 else rows[0]

        prev_position = {m: idx for idx, m in enumerate(open_modes)}
        connectivity[len(diagrams) - 1] = {idx: prev_position[m] for idx, m in enumerate(new_row_order)}
        diagrams.append(layer)

        open_modes = other if op.name() == "intrinsic.measurement" else new_row_order

    if len(diagrams) == 1:
        return diagrams[0]
    return CompositionDiagram(diagrams, connectivity)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def from_circuit_repr(circuit: CircuitRepr, *, normalize: bool = True) -> Diagram:
    """Translate an mqc3 `CircuitRepr` into a cvzx `Diagram`.

    Naively translates every operation and initial state (see
    :doc:`../user_guide/circuit_conversion` for the exact per-gate/per-state
    conversion formulas) into a compact-form `Diagram`, then -- unless
    `normalize=False` -- rewrites it into canonical alternating type-1/type-2
    stages via `cvzx.passes.normalize.normalize_diagram`.

    Parameters
    ----------
    circuit : CircuitRepr
        The mqc3 circuit to convert. Not mutated (a deep copy is used
        internally for the `std.* -> intrinsic.*` lowering step).
    normalize : bool
        If True (default), canonicalize the result via
        `normalize_diagram`. If False, return the raw naive translation
        (still a semantically-correct, but non-canonical, `Diagram`).

    Returns
    -------
    Diagram
        The translated (and, by default, canonicalized) diagram.

        `_naive_translate` raises `NotImplementedError` if `circuit`
        uses the `Manual` gate (directly, or indirectly via a `std.*`
        operation that lowers to it), a feedforward parameter, or an
        initial state outside the supported set (see
        :doc:`../user_guide/circuit_conversion`), and `ValueError` if
        `circuit` has no modes.
    """
    diagram = _naive_translate(circuit)
    if normalize:
        diagram = normalize_diagram(diagram)
    return diagram


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

    This is the exact leaf shape `cvzx.passes.completion.complete_diagram()`
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
    if isinstance(elt, ControlledSumGate):
        # No direct mqc3 primitive: conjugating ControlledZ by a Fourier
        # rotation on the target mode converts CZ's q-q coupling into
        # CSUM's q-p coupling -- see :doc:`../user_guide/circuit_conversion`
        # for the derivation and its Heisenberg-picture verification.
        gain = _resolve_scalar(elt.gain, "ControlledSumGate", measurement_ops)
        gain = -gain if isinstance(gain, float) else _negate_feedforward(gain)
        target_mode = mode_a if elt.target == 1 else mode_b
        circuit.Q(target_mode) | intrinsic.PhaseRotation(-np.pi / 2)
        circuit.Q(mode_a, mode_b) | intrinsic.ControlledZ(gain)
        circuit.Q(target_mode) | intrinsic.PhaseRotation(np.pi / 2)
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
    1-mode leaves) is translated independently (see
    :doc:`../user_guide/circuit_conversion` for the exact per-leaf
    formulas), and mode ids are threaded across stage boundaries via the
    diagram's own `connectivity` dicts.

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
        not recognize (see :doc:`../user_guide/circuit_conversion`), or a
        `ContractedDiagram` anywhere.

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

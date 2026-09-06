"""Convert an mqc3 `CircuitRepr` into a canonical cvzx `Diagram`.

This module implements the "naive translate, then normalize" architecture:

1. `_naive_translate` walks the mqc3 `CircuitRepr` in time order (via
   `CircuitRepr.__iter__`, after `convert_std_ops_to_intrinsic()` has
   lowered every `std.*` operation down to `intrinsic.*` ones) and builds
   *some* compact-form cvzx `Diagram` that is semantically equivalent to
   the circuit -- one flat `CompositionDiagram` of per-operation
   `TensorDiagram` layers, threaded together with explicit `connectivity`
   dicts so that a 2-mode operation touching two non-adjacent rows never
   needs an explicit `Swap`: each layer is free to place its rows in
   whatever order is convenient (touched modes first), and the
   connectivity dict maps that layer's inputs back to wherever the
   previous layer actually produced them. This is exactly the kind of
   "natural"/arbitrarily-shaped diagram `normalize_diagram` (see
   `cvzx.normalize_diagram`) is designed to consume.
2. `from_circuit_repr` (the public entry point) calls `_naive_translate`
   and then `normalize_diagram` on the result, returning the canonical
   alternating type-1/type-2 form.

Parameter-convention notes
---------------------------
Every gate conversion below was derived from mqc3's own docstrings (see
`mqc3.circuit.ops.intrinsic`) and numerically verified against cvzx's
own gate matrices. A short summary:

- `intrinsic.PhaseRotation(phi)` -> `PhaseRotationGate(-phi)`.
- `intrinsic.ControlledZ(g)` -> `ControlledZGate(gain=-g)`.
- `intrinsic.ShearXInvariant`, `ShearPInvariant`, `Squeezing45`,
  `Arbitrary`, `TwoModeShear`, `Measurement` all map 1:1 onto the
  correspondingly-named cvzx gate with the *same* parameter(s) -- the
  sign corrections are already baked into each gate's own `expand()`.
- `intrinsic.Displacement(x, p)` -> `DisplacementGate(alpha)` with
  `alpha = (x + 1j*p) / sqrt(2)`. Derived from `D(a) = exp(a a^dag - a*
  a)` with `a = (x_hat + i p_hat) / sqrt(2)`, giving the standard result
  `D^dagger(a) (x_hat + i p_hat) D(a) = x_hat + i p_hat + sqrt(2) a`.
- `intrinsic.Squeezing(theta)` (mqc3: `R(-pi/2) . S_V(cot theta)`) has no
  direct 1:1 cvzx gate; it is translated as the 2-step composition
  `[SqueezingGate(tan(theta)), PhaseRotationGate(pi/2)]`.
- `intrinsic.BeamSplitter(sqrt_r, theta_rel)` has no direct 1:1 cvzx
  gate either. Its 4x4 matrix, worked out via the complex mode
  `z = x + i*p`, is `exp(i*theta_rel) * [[sqrt_r, -i*sqrt(1-sqrt_r**2)],
  [-i*sqrt(1-sqrt_r**2), sqrt_r]]` -- an overall phase times a
  sigma_x-generated beamsplitter, whereas cvzx's own `BeamsplitterGate`
  is sigma_y-generated. Conjugating by a +-pi/2 rotation on mode 2
  converts between the two, giving (with `eta = arccos(sqrt_r)`):

      [TensorDiagram([identity, PhaseRotationGate(-pi/2)]),   # mode2 only
       BeamsplitterGate(eta),
       TensorDiagram([PhaseRotationGate(-theta_rel),
                      PhaseRotationGate(pi/2 - theta_rel)])]

  Both formulas were checked numerically against mqc3's exact docstring
  matrices to machine precision across many random parameter values.
- `intrinsic.Manual` has no known CV-ZX decomposition yet and is
  intentionally out of scope -- converting a circuit that uses it (or a
  `std.*` operation, such as `std.BeamSplitter`, that lowers to it via
  `convert_std_ops_to_intrinsic()`) raises `NotImplementedError`.

Initial states
--------------
Every mode's `InitialState` is translated into an idealized (infinitely
squeezed) CV-ZX state leaf, `QSpider(0, 1, ZxPoly({}))`, optionally
rotated by a `PhaseRotationGate` to set the squeezing axis -- this
matches how the rest of this codebase already represents resource
states (see e.g. `MeasurementGate.conjugate()`), and how CV-ZX calculus
treats such states in the idealized/infinite-squeezing limit generally
(finite squeezing has no representation anywhere in this codebase).
Supported initial states:

- `HardwareConstrainedSqueezedState(phi)`: rotated ideal state, using
  the same `R(phi)` sign convention as everywhere else in this module.
- `BosonicState` with exactly one peak and zero mean, provided its
  Gaussian component's covariance matrix is that of a pure
  (minimum-uncertainty) squeezed/vacuum state -- the squeezing axis
  `phi` is recovered from the covariance matrix's eigenvectors.

Any other initial state (a genuine multi-peak/non-Gaussian
superposition, a displaced state, or a mixed covariance) raises
`NotImplementedError` rather than being silently approximated.

Feedforward
-----------
mqc3 feedforward (`FeedForward[MeasuredVariable]`, an arbitrary Python
function of a measurement outcome) is not yet wired through to cvzx's
`measurement_ids`/symbolic-parameter mechanism -- converting an
operation with a feedforward parameter raises `NotImplementedError`.
Both sides already have the structural support needed for this
(`feedforward`/`measurement_ids` on every parametrized cvzx gate; a
resolvable `from_operation` on every mqc3 `FeedForward`), so this is a
natural follow-up, not a dead end.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from cvzx.base_gates import CompositionDiagram, Diagram, Fourier, FourierInv, QSpider, TensorDiagram, ZxPoly
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
    from mqc3.circuit.ops._base import Operation
    from mqc3.circuit.program import CircOpParam
    from mqc3.circuit.state import InitialState

__all__ = ["from_circuit_repr"]


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


def _phase_rotation(theta: float) -> Diagram:
    """`PhaseRotationGate(theta)`, routing around its odd-multiple-of-pi/2 restriction.

    `PhaseRotationGate` refuses any angle that is an odd multiple of
    `pi/2` (it asks callers to use `Fourier`/`FourierInv` instead). Every
    translator in this module that builds a `PhaseRotationGate` from an
    mqc3 angle can land on exactly such an angle (e.g. `intrinsic.
    Squeezing`'s fixed `pi/2` factor, or `intrinsic.PhaseRotation(pi/2)`
    itself), so this helper folds `theta` into `(-pi, pi]` and
    substitutes the equivalent `Fourier`/`FourierInv` proper diagram at
    the two problem angles -- the exact same routing already used by
    `MeasurementGate._rotation_diagram` in `cvzx.gates` (see its
    docstring for the `Fourier() = mqc3 R(pi/2)`, `FourierInv() = mqc3
    R(-pi/2)` derivation this relies on).

    Returns
    -------
    Diagram
        `Fourier()`/`FourierInv()` at the two problem angles, otherwise
        `PhaseRotationGate(folded)`.
    """
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
    """Reject feedforward parameters (not yet supported) and coerce to `float`.

    Returns
    -------
    float
        `value`, coerced.

    Raises
    ------
    NotImplementedError
        If `value` is a `FeedForward` (not yet supported).
    """
    # ruff: ignore[import-outside-top-level]
    from mqc3.feedforward import FeedForward

    if isinstance(value, FeedForward):
        msg = (
            f"Cannot convert `{op_name}`: feedforward on parameter #{param_index} is "
            "not yet supported by the CircuitRepr -> Diagram converter (both sides "
            "already have the structural support for it -- see this module's "
            "docstring -- but the wiring itself is not implemented yet)."
        )
        raise NotImplementedError(msg)
    return float(value)


# --------------------------------------------------------------------------
# Per-operation translators
# --------------------------------------------------------------------------
# Each translator takes the operation's already-float-coerced parameters
# (in the exact order `Operation.parameters()` returns them -- see
# `mqc3.circuit.ops.intrinsic`) and returns the compact-form cvzx `Diagram`
# for that operation alone (a 1-mode or 2-mode gate/effect, in the same
# mode order as `Operation.opnd().get_ids()`).


def _translate_measurement(params: list[float]) -> Diagram:
    (theta,) = params
    return MeasurementGate(theta)


def _translate_displacement(params: list[float]) -> Diagram:
    x, p = params
    alpha = complex(x, p) / np.sqrt(2)
    return DisplacementGate(alpha)


def _translate_phase_rotation(params: list[float]) -> Diagram:
    (phi,) = params
    return _phase_rotation(-phi)


def _translate_shear_x_invariant(params: list[float]) -> Diagram:
    (kappa,) = params
    return ShearXInvariantGate(kappa)


def _translate_shear_p_invariant(params: list[float]) -> Diagram:
    (eta,) = params
    return ShearPInvariantGate(eta)


def _translate_squeezing(params: list[float]) -> Diagram:
    (theta,) = params
    return CompositionDiagram([SqueezingGate(np.tan(theta)), _phase_rotation(np.pi / 2)])


def _translate_squeezing45(params: list[float]) -> Diagram:
    (theta,) = params
    return Squeezing45Gate(theta)


def _translate_arbitrary(params: list[float]) -> Diagram:
    alpha, beta, lam = params
    return ArbitraryGate(alpha, beta, lam)


def _translate_controlled_z(params: list[float]) -> Diagram:
    (g,) = params
    return ControlledZGate(gain=-g)


def _translate_beam_splitter(params: list[float]) -> Diagram:
    sqrt_r, theta_rel = params
    eta = np.arccos(sqrt_r)
    return CompositionDiagram([
        TensorDiagram([_identity_wire(), _phase_rotation(-np.pi / 2)]),
        BeamsplitterGate(eta),
        TensorDiagram([_phase_rotation(-theta_rel), _phase_rotation(np.pi / 2 - theta_rel)]),
    ])


def _translate_two_mode_shear(params: list[float]) -> Diagram:
    a, b = params
    return TwoModeShearGate(a, b)


def _translate_manual(_params: list[float]) -> Diagram:
    msg = (
        "Cannot convert `intrinsic.manual`: the Manual gate's CV-ZX decomposition "
        "is not yet implemented (intentionally deferred). This can be reached "
        "either directly or via `convert_std_ops_to_intrinsic()` lowering a "
        "`std.*` operation -- e.g. `std.BeamSplitter` -- through `Manual`."
    )
    raise NotImplementedError(msg)


_TRANSLATORS = {
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


def _translate_operation(op: Operation) -> Diagram:
    name = op.name()
    translator = _TRANSLATORS.get(name)
    if translator is None:
        msg = f"Cannot convert operation `{name}`: no CircuitRepr -> Diagram translation is registered for it."
        raise NotImplementedError(msg)
    params = [_as_float(p, name, i) for i, p in enumerate(op.parameters())]
    return translator(params)


def _naive_translate(circuit: CircuitRepr) -> Diagram:
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

    for op in circuit:
        gate_diagram = _translate_operation(op)
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

    Naively translates every operation and initial state (see the module
    docstring for the exact per-gate/per-state conversion formulas) into
    a compact-form `Diagram`, then -- unless `normalize=False` -- rewrites
    it into canonical alternating type-1/type-2 stages via
    `cvzx.normalize_diagram.normalize_diagram`.

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
        initial state outside the supported set (see the module
        docstring), and `ValueError` if `circuit` has no modes.
    """
    diagram = _naive_translate(circuit)
    if normalize:
        diagram = normalize_diagram(diagram)
    return diagram

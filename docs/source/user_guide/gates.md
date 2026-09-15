# Gate catalog

Every gate lives in `cvzx.ir.gates` and is a `CompactDiagram` subclass: it behaves as an
opaque, 1- or 2-mode leaf until something calls `.expand()` on it (or
`cvzx.ir.gates.expand_all`, or `cvzx.backends.nx.rules.expand_two_mode_gates` for the two-mode
gates specifically), at which point it produces its spider decomposition (see
{doc}`../theory`). All gates share the same four extra fields:

- `parametric: bool = False` — if `True`, the gate's parameter(s) may be (and are expected
  to be) `sympy` expressions rather than plain numbers.
- `feedforward: bool = False` and `measurement_ids: set[int] | None = None` — mark a gate as
  depending on the outcome of earlier measurements; `feedforward=True` requires a non-empty
  `measurement_ids` (the `.id`s of the measurement leaves it depends on), or construction
  raises `ValueError`.

```python
from sympy import symbols
from cvzx.ir.gates import DisplacementGate, PhaseRotationGate

D = DisplacementGate(alpha=1.0 + 0.5j)                       # numeric
theta = symbols("theta", real=True)
R_sym = PhaseRotationGate(theta=theta, parametric=True)        # symbolic
```

## One-mode gates

| Class | Parameter(s) | Action |
|---|---|---|
| `DisplacementGate(alpha)` | complex $\alpha$ | $\hat q \to \hat q + \sqrt2\,\mathrm{Re}(\alpha)$, $\hat p \to \hat p + \sqrt2\,\mathrm{Im}(\alpha)$ |
| `PhaseRotationGate(theta)` | real $\theta$ | rotates $(\hat q, \hat p)$ by $\theta$ |
| `SqueezingGate(tau)` | real $\tau \ne 0$ | $\hat q \to \tau \hat q$, $\hat p \to \hat p / \tau$ |
| `Squeezing45Gate(theta)` | real $\theta$ | squeezing at a $45°$ angle, parametrized like `SqueezingGate` but pre/post-rotated by $\pi/4$ |
| `ShearXInvariantGate(kappa)` | real $\kappa$ | $\hat q$ invariant, $\hat p \to \hat p + 2\kappa \hat q$ |
| `ShearPInvariantGate(eta)` | real $\eta$ | $\hat p$ invariant, $\hat q \to \hat q + 2\eta \hat p$ |
| `CubicPhaseGate(gamma)` | real $\gamma$ | non-Gaussian: $\hat p \to \hat p + 3\gamma \hat q^2$ |
| `ArbitraryGate(alpha, beta, lam)` | real $\alpha, \beta, \lambda$ | the general one-mode Gaussian unitary $R(\alpha)\,S(\lambda)\,R(\beta)$ |
| `MeasurementGate(theta)` | real $\theta$ | a homodyne **effect** (1 input, 0 outputs) — not a unitary gate |

## Two-mode gates

| Class | Parameter(s) | Generator $\hat H$ (unitary is $e^{-i\hat H}$) |
|---|---|---|
| `ControlledSumGate(gain, control, target)` | real gain $g$, `control`/`target` $\in \{1, 2\}$, $\ne$ each other | $g\,\hat q_{\text{control}}\hat p_{\text{target}}$ (the CV analogue of CNOT at $g=1$) |
| `ControlledZGate(gain)` | real gain $g$ | $-g\,\hat q_1\hat q_2$ |
| `BeamsplitterGate(theta)` | real $\theta$ | $\theta\,(\hat q_1\hat p_2 - \hat p_1\hat q_2)$ |
| `TwoModeShearGate(a, b)` | real $a, b$ | (not a single exponential) $\hat p_1 \mathrel{+}= 2a\hat q_1 + b\hat q_2,\ \hat p_2 \mathrel{+}= b\hat q_1 + 2a\hat q_2$, $\hat q$'s invariant |

The tables above give each gate's exact Heisenberg-picture action; each class's own
docstring keeps only the key step its `expand()` performs. The derivations behind the
non-obvious steps — mqc3 sign-convention corrections, special-case reductions to other
gates, and why certain `conjugate()` formulas hold — are collected in the "Gate
decompositions" section below.

## Gate decompositions

### `ArbitraryGate` — general one-mode Gaussian gate

mqc3 defines `intrinsic.Arbitrary(alpha, beta, lam)` as the operator product
`R(alpha) . S(lam) . R(beta)` (rightmost applied first, i.e. the mode meets `R(beta)`
first, then `S(lam)`, then `R(alpha)` last), where

$$R^\dagger(\phi)\,(\hat q, \hat p)\,R(\phi) = \begin{pmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{pmatrix}(\hat q, \hat p), \qquad S^\dagger(r)\,(\hat q, \hat p)\,S(r) = \begin{pmatrix}e^r & 0\\ 0 & e^{-r}\end{pmatrix}(\hat q, \hat p).$$

Two conversions are needed to express this with cvzx's existing gates:

- cvzx's `PhaseRotationGate(theta)` implements mqc3's `R(-theta)`, not `R(theta)`
  (verified directly from its own three-spider decomposition) — so `R(phi)` here becomes
  `PhaseRotationGate(-phi)`.
- cvzx's `SqueezingGate(tau)` implements $\mathrm{diag}(\tau, 1/\tau)$, exactly `S(lam)`
  under $\tau = e^{\lambda}$ (no sign correction needed there).

Composed in signal-flow order (first-applied-first, matching `CompositionDiagram`'s own
convention):

```text
ArbitraryGate(alpha, beta, lam).expand()
    = PhaseRotationGate(-beta) . SqueezingGate(e^lam) . PhaseRotationGate(-alpha)
```

Verified numerically against the raw mqc3 matrix product for random `(alpha, beta, lam)`.

`conjugate()`: `(R(alpha) S(lam) R(beta))^dagger = R(-beta) S(-lam) R(-alpha)`, which is
again of the form `R(alpha') S(lam') R(beta')` with `alpha' = -beta`, `lam' = -lam`,
`beta' = -alpha` — so `ArbitraryGate(alpha, beta, lam).conjugate()` is
`ArbitraryGate(-beta, -alpha, -lam)`.

### `Squeezing45Gate` — special case of `ArbitraryGate`

mqc3 defines `intrinsic.Squeezing45(theta)` as `R(-pi/4) S_V(cot theta) R(pi/4)`, where
`S_V(c)` has matrix $\mathrm{diag}(1/c, c)$. This is exactly `ArbitraryGate` with
`alpha = -pi/4`, `beta = pi/4`, and `lam` chosen so that `S(lam) = S_V(cot theta)`: since
$e^{\lambda} = 1/\cot\theta = \tan\theta$, cvzx's `SqueezingGate(tau)` can be used directly
with `tau = tan(theta)`, no logarithm required.

`conjugate()`: since $S_V(c)^\dagger = S_V(1/c)$ and $1/\cot\theta = \cot(\pi/2 - \theta)$,
`Squeezing45Gate(theta).conjugate()` is `Squeezing45Gate(pi/2 - theta)`.

### `TwoModeShearGate` — combined diagonal shear and `ControlledZGate`

mqc3's `intrinsic.TwoModeShear(a, b)` Heisenberg transformation is

$$P_2^\dagger(a,b)\,(\hat q_1,\hat q_2,\hat p_1,\hat p_2)\,P_2(a,b) = \begin{pmatrix}1&0&0&0\\0&1&0&0\\2a&b&1&0\\b&2a&0&1\end{pmatrix}(\hat q_1,\hat q_2,\hat p_1,\hat p_2).$$

The diagonal `2a` terms are exactly `ShearXInvariantGate(a)` applied to each mode; the
cross `b` term is exactly cvzx's `ControlledZGate` generator $\exp(-ig\hat q_1\hat q_2)$
evaluated at `g = -b` (cvzx's `ControlledZGate(g)` produces `p1 -= g*q2, p2 -= g*q1`, the
negative of mqc3's `ControlledZ(g)` convention — verified directly from its own
decomposition's generator). Both pieces are shears of the same abelian family (they only
ever add a linear function of the $\hat q$'s to the $\hat p$'s, leaving the $\hat q$'s
invariant), so they commute and `expand()` needs no new primitive — just
`(ShearXInvariantGate(a) ⊗ ShearXInvariantGate(a))` composed with `ControlledZGate(gain=-b)`.

`conjugate()` negates both `a` and `b` directly, rather than delegating to
`ControlledZGate.conjugate()`: both parameters parametrize the same real quadratic-form
generator $\hat H = a\hat q_1^2 + a\hat q_2^2 + b\hat q_1\hat q_2$, so
$e^{-i\hat H\dagger} = e^{+i\hat H} = e^{-i(-\hat H)}$ is the same gate with both
coefficients negated.

### `MeasurementGate` — rotate into alignment, then measure x

mqc3's `intrinsic.Measurement(theta)` measures the quadrature
$\hat q\sin\theta + \hat p\cos\theta$. Measuring $\hat q$ directly is a plain q-spider
effect, `QSpider(1, 0, 0)`; to measure the rotated quadrature, `expand()` rotates the mode
into alignment first. Solving `R(phi)`'s Heisenberg matrix (top row) for
$\cos\phi\,\hat q - \sin\phi\,\hat p = \sin\theta\,\hat q + \cos\theta\,\hat p$ gives
`phi = theta - pi/2`; composed in signal-flow order (rotate first, measure second) and
converted to cvzx's rotation convention (`PhaseRotationGate(psi)` = mqc3's `R(-psi)`, see
`ArbitraryGate` above):

```text
MeasurementGate(theta).expand() = PhaseRotationGate(pi/2 - theta) . QSpider(1, 0, 0)
```

Checked against `theta=0` (measuring $\hat p$ directly): `phi=-pi/2` correctly rotates
$\hat q$ onto $\hat p$.

`conjugate()` cannot return another `MeasurementGate`: an effect's adjoint is a *state*
(0 inputs, 1 output), a different shape than a `CompactDiagram` effect can represent, so
it is built directly as a zero-phase state followed by the inverse rotation, rather than
delegating to `expand().conjugate()` (which fails arity validation for this non-square
leaf, since `QSpider.conjugate()` does not flip input/output arity for state/effect leaves).

`expand()`'s internal rotation-building step routes around `PhaseRotationGate`'s refusal
of an odd multiple of pi/2 — exactly the angles a plain x or p measurement needs (`theta`
an integer multiple of pi). It folds the rotation angle into `(-pi, pi]` and substitutes
`Fourier`/`FourierInv` at the two problem angles instead: `Fourier` implements `R(pi/2)`
($\hat q' = -\hat p, \hat p' = \hat q$), i.e. `PhaseRotationGate(-pi/2)`; `FourierInv` is
its adjoint, `PhaseRotationGate(pi/2)`. This only applies in numeric mode — a parametric
angle skips `PhaseRotationGate`'s own validation the same way already.

## Composing gates directly

Every `Diagram` (gates included) supports `.compose()` and `.tensor()` for building a
circuit without going through `CompositionDiagram`/`TensorDiagram` list literals, and
`.conjugate()` for the adjoint:

```python
D = DisplacementGate(alpha=1.0 + 0.5j)
R = PhaseRotationGate(theta=0.7)

# D ∘ R, i.e. R applied first, then D
comp = D.compose(R)

# D ⊗ R, side by side
tensor = D.tensor(R)

D_dagger = D.conjugate()
```

## Symbolic parameters and substitution

A `parametric=True` gate accepts `sympy` symbols and expressions for its parameter(s), and
exposes `.get_parameters()` (the free symbols involved) and `.substitute_parameters(mapping)`
/`.evaluate(**kwargs)` to specialize them:

```python
from sympy import symbols
from cvzx.ir.gates import DisplacementGate

a, b = symbols("a b")
D_sym = DisplacementGate(alpha=a + 1j * b, parametric=True)
D_sym.get_parameters()                       # {a, b}
D_num = D_sym.substitute_parameters({a: 0.5, b: 0.3})   # -> non-parametric DisplacementGate
```

## Feedforward

A gate whose parameter depends on an earlier measurement outcome carries that
provenance in `param_measurement_map`: a `dict` mapping each symbol to the set of
measuring leaves' `.id`s it depends on. `feedforward`/`measurement_ids` are derived
from it automatically — pass `param_measurement_map`, not `feedforward`/
`measurement_ids` directly:

```python
from sympy import symbols
from cvzx.ir.base import QSpider, ZxPoly
from cvzx.ir.gates import PhaseRotationGate

m = symbols("m", real=True)
meas_leaf = QSpider(1, 0, ZxPoly({1: m}))  # a measurement effect producing outcome m
R = PhaseRotationGate(
    theta=m,
    parametric=True,
    param_measurement_map={m: {meas_leaf.id}},
)
R.feedforward       # True -- derived
R.measurement_ids   # {meas_leaf.id} -- derived
```

Constructing a gate with `feedforward=True` or a non-empty `measurement_ids` but an
empty `param_measurement_map` raises `ValueError` — provenance is required, not
optional. See `tests/visualization/test_visualize_gates.py::feedforward_params_all_gates_test`
for a sweep over every gate class checking exactly this.

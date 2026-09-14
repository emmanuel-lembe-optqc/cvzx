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

See each class's docstring for its exact Heisenberg-picture transformation and the
derivation of its spider decomposition — they are written out in full and cross-referenced
to the paper equation they implement.

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

A gate whose parameter depends on an earlier measurement outcome is marked
`feedforward=True` with the measuring leaf's `.id` recorded in `measurement_ids`:

```python
from sympy import symbols
from cvzx.ir.base import QSpider, ZxPoly
from cvzx.ir.gates import PhaseRotationGate

m = symbols("m", real=True)
meas_leaf = QSpider(1, 0, ZxPoly({1: m}))  # a measurement effect producing outcome m
R = PhaseRotationGate(
    theta=m,
    parametric=True,
    feedforward=True,
    measurement_ids={meas_leaf.id},
)
```

Constructing a `feedforward=True` gate with no `measurement_ids` (or an empty set) raises
`ValueError` — see `tests/utils/test_visualize_gates.py::feedforward_params_all_gates_test` for a
sweep over every gate class checking exactly this.

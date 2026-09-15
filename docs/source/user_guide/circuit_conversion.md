# Converting to and from mqc3 circuits

`cvzx.lowering.bridges.mqc3.from_circuit_repr` and `cvzx.lowering.bridges.mqc3.to_circuit_repr` are
the two halves of a round trip between an mqc3 `CircuitRepr` and a `cvzx` `Diagram`, and
`cvzx.lowering.lowering.graph_to_dependency_dag` carries a closed `Diagram` the rest of the way
to mqc3's own `DependencyDAG`. See {doc}`../dev_guide/architecture` for how these modules fit
into the full pipeline.

## mqc3 `CircuitRepr` to a `Diagram`

```python
from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic

from cvzx.lowering.bridges.mqc3 import from_circuit_repr

circuit = CircuitRepr("simple")
circuit.Q(0) | intrinsic.PhaseRotation(0.4)
circuit.Q(0) | intrinsic.Measurement(0.1)

diagram = from_circuit_repr(circuit)
```

`from_circuit_repr` naively translates each mqc3 operation and initial state into the
compact-form `cvzx.ir.gates` equivalent (exact per-gate formulas below), then canonicalizes
the result via `cvzx.passes.normalize.normalize_diagram` — pass `normalize=False` to get the
raw, non-canonical translation instead. A handful of mqc3 features have no `cvzx` counterpart
yet and raise `NotImplementedError`: the `Manual` gate (reached directly, or indirectly via a
`std.*` operation that lowers to it), a feedforward parameter outside the affine/single-symbol
case described below, and any initial state outside the idealized-squeezed-state /
pure-Gaussian-`BosonicState` set described below.

### Per-gate conversion formulas (mqc3 -> cvzx)

Every formula below was derived from mqc3's own docstrings (see `mqc3.circuit.ops.intrinsic`)
and numerically verified against cvzx's own gate matrices:

- `intrinsic.PhaseRotation(phi)` -> `PhaseRotationGate(-phi)`.
- `intrinsic.ControlledZ(g)` -> `ControlledZGate(gain=-g)`.
- `intrinsic.ShearXInvariant`, `ShearPInvariant`, `Squeezing45`, `Arbitrary`, `TwoModeShear`,
  `Measurement` all map 1:1 onto the correspondingly-named cvzx gate with the *same*
  parameter(s) — the sign corrections are already baked into each gate's own `expand()`.
- `intrinsic.Displacement(x, p)` -> `DisplacementGate(alpha)` with
  `alpha = (x + 1j*p) / sqrt(2)`. Derived from `D(a) = exp(a a^dag - a* a)` with
  `a = (x_hat + i p_hat) / sqrt(2)`, giving the standard result
  `D^dagger(a) (x_hat + i p_hat) D(a) = x_hat + i p_hat + sqrt(2) a`.
- `intrinsic.Squeezing(theta)` (mqc3: `R(-pi/2) . S_V(cot theta)`) has no direct 1:1 cvzx gate;
  it is translated as the 2-step composition
  `[SqueezingGate(tan(theta)), PhaseRotationGate(pi/2)]`.
- `intrinsic.BeamSplitter(sqrt_r, theta_rel)` has no direct 1:1 cvzx gate either. Its 4x4
  matrix, worked out via the complex mode `z = x + i*p`, is
  `exp(i*theta_rel) * [[sqrt_r, -i*sqrt(1-sqrt_r**2)], [-i*sqrt(1-sqrt_r**2), sqrt_r]]` — an
  overall phase times a sigma_x-generated beamsplitter, whereas cvzx's own `BeamsplitterGate`
  is sigma_y-generated. Conjugating by a +-pi/2 rotation on mode 2 converts between the two,
  giving (with `eta = arccos(sqrt_r)`):

  ```python
  [TensorDiagram([identity, PhaseRotationGate(-pi/2)]),   # mode2 only
   BeamsplitterGate(eta),
   TensorDiagram([PhaseRotationGate(-theta_rel),
                  PhaseRotationGate(pi/2 - theta_rel)])]
  ```

  Both formulas were checked numerically against mqc3's exact docstring matrices to machine
  precision across many random parameter values.

### Initial states

Every mode's `InitialState` is translated into an idealized (infinitely squeezed) CV-ZX state
leaf, `QSpider(0, 1, ZxPoly({}))`, optionally rotated by a `PhaseRotationGate` to set the
squeezing axis — this matches how the rest of `cvzx` already represents resource states (see
e.g. `MeasurementGate.conjugate()`), and how CV-ZX calculus treats such states in the
idealized/infinite-squeezing limit generally (finite squeezing has no representation anywhere
in this codebase). Supported initial states:

- `HardwareConstrainedSqueezedState(phi)`: rotated ideal state, using the same `R(phi)` sign
  convention as everywhere else in this module.
- `BosonicState` with exactly one peak and zero mean, provided its Gaussian component's
  covariance matrix is that of a pure (minimum-uncertainty) squeezed/vacuum state — the
  squeezing axis `phi` is recovered from the covariance matrix's eigenvectors.

Any other initial state (a genuine multi-peak/non-Gaussian superposition, a displaced state,
or a mixed covariance) raises `NotImplementedError` rather than being silently approximated.

### Feedforward on ingestion

An `intrinsic.Measurement` operation that some later operation's `FeedForward[MeasuredVariable]`
parameter references is reconstructed as `QSpider`/`PSpider(1, 0, ZxPoly({1: -m}))` (the same
leaf shape `cvzx.passes.completion.complete_diagram()` produces) for a fresh symbol `m`, chosen
by the measured quadrature (`theta` close to `pi/2` -> `QSpider`, close to `0` -> `PSpider`);
any other angle falls back to a plain `MeasurementGate(theta)` — only the two canonical angles
have a Q/P-spider representation to reconstruct into.

The referencing operation's own parameter is recovered as `slope*m + intercept` (evaluating
the `FeedForwardFunction` numerically at `0.0` and `1.0` to recover that affine relationship)
and threaded into the corresponding cvzx gate as a symbolic (`parametric=True`) parameter — the
exact algebraic inverse of `to_circuit_repr`'s own feedforward resolution (below). A
`FeedForward` depending on more than one measurement symbol, or a nonlinear function of one
(mqc3's `FeedForwardFunction` supports arbitrary Python callables; only the affine case is
inverted here), is not supported and raises `NotImplementedError`.

The reconstructed cvzx gate also carries `param_measurement_map={m: {measurement_leaf.id}}` —
the `GateRegister`-traceable binding from the symbol back to the measurement leaf that produces
it — so `feedforward`/`measurement_ids` are always correctly derived rather than left unset. A
downstream gate with more than one feedforward-derived parameter (e.g. `ArbitraryGate`,
`TwoModeShearGate`) gets one entry per measurement symbol its own parameters actually
reference.

Note: a measurement's own `theta` parameter could itself be feedforward-dependent on an
*earlier* measurement (nested/adaptive feedforward), but the canonical-angle check that
selects `QSpider`/`PSpider` above does not handle a symbolic `theta` — this is a pre-existing,
separate limitation, out of scope here.

## `Diagram` back to mqc3 `CircuitRepr`

```python
from cvzx.lowering.bridges.mqc3 import to_circuit_repr

circuit2 = to_circuit_repr(diagram)
```

`to_circuit_repr` canonicalizes `diagram` first (a no-op if it already is canonical), then
walks it stage by stage, translating each primitive leaf independently — it does **not**
attempt to pattern-match a canonicalized diagram's leaves back into the composite gate
(`BeamSplitter`, `Squeezing`) they may have come from, so a round-tripped circuit is
semantically equivalent but not necessarily op-for-op identical to the original: a
`BeamSplitter(sqrt_r, theta_rel)` round-trips as a `PhaseRotation` + `BeamSplitter(sqrt_r, 0)` +
two more `PhaseRotation`s, for instance, not as a single `BeamSplitter(sqrt_r, theta_rel)` call.
`normalize_diagram` decomposes every diagram to the same flat set of primitive leaves
regardless of how they were originally grouped, which is why per-leaf (not per-composite-gate)
translation is the only option here — and why the input `Diagram` must not already be the
target of `optimize()`'s algebraic spider-fusion rewrites (as opposed to `normalize_diagram`'s
purely structural ones): a fused spider's phase polynomial may no longer match any primitive
shape this module recognizes, in which case conversion raises `NotImplementedError` rather than
silently producing something else.

### Per-leaf conversion formulas (cvzx -> mqc3)

Each of these is the algebraic inverse of the corresponding formula above:

- `PhaseRotationGate(theta)` -> `intrinsic.PhaseRotation(-theta)`.
- `Fourier()` -> `intrinsic.PhaseRotation(pi/2)`; `FourierInv()` -> `intrinsic.PhaseRotation(-pi/2)`;
  `Fourier2()` -> `intrinsic.PhaseRotation(pi)`.
- `ShearXInvariantGate(kappa)`, `ShearPInvariantGate(eta)`, `ArbitraryGate(alpha, beta, lam)`,
  `Squeezing45Gate(theta)`, `TwoModeShearGate(a, b)`, `MeasurementGate(theta)` all map 1:1 onto
  their correspondingly-named mqc3 op with the *same* parameter(s).
- `SqueezingGate(tau)` has no bare mqc3 primitive (mqc3's `Squeezing` op has a different,
  fixed-rotation definition — see the formulas above); it is instead expressed via
  `ArbitraryGate`'s own `alpha = beta = 0` special case: `intrinsic.Arbitrary(0, 0, ln(tau))`.
- `DisplacementGate(alpha)` -> `intrinsic.Displacement(x, p)` with `x = sqrt(2)*Re(alpha)`,
  `p = sqrt(2)*Im(alpha)`.
- `ControlledZGate(gain=g)` -> `intrinsic.ControlledZ(-g)`.
- `BeamsplitterGate(theta)` -> `intrinsic.BeamSplitter(sqrt_r, 0)` with `sqrt_r = cos(theta)` —
  the `theta_rel = 0` special case of the general formula above.
- `ControlledSumGate(gain=g, control=c, target=t)` has no bare mqc3 primitive (mqc3's intrinsic
  set has `ControlledZ` but no CSUM/CNOT-style analogue). cvzx's `ControlledSumGate(g)` is
  `exp(-i g q̂_c p̂_t)` and `ControlledZGate(g)` is `exp(-i g q̂₁ q̂₂)` — conjugating `ControlledZ`
  by a Fourier rotation on the *target* mode converts CZ's q-q coupling into CSUM's q-p
  coupling: `CZ(g) = (I ⊗ F_t†) CSUM_{c→t}(g) (I ⊗ F_t)`, i.e.
  `CSUM_{c→t}(g) = (I ⊗ F_t) CZ(g) (I ⊗ F_t†)`. Emitted as three ops on `(mode_a, mode_b)`:
  `intrinsic.PhaseRotation(-pi/2)` on the target mode (= `FourierInv`), then
  `intrinsic.ControlledZ(-g)` on both modes, then `intrinsic.PhaseRotation(pi/2)` on the target
  mode (= `Fourier`) — verified via the Heisenberg-picture (classical Hamiltonian flow)
  transform of `(q_c, p_c, q_t, p_t)` under each side independently; both sides give
  `q_c' = q_c`, `p_c' = p_c - g p_t`, `q_t' = q_t + g q_c`, `p_t' = p_t`.
- A bare zero-phase state leaf (`QSpider(0, 1, 0)`) -> a mode with
  `HardwareConstrainedSqueezedState(phi=0)`; the `PSpider` counterpart -> `phi = pi/2`
  (mirroring the `QSpider`-is-x-type/`PSpider`-is-p-type convention used throughout this
  codebase). A bare zero-phase effect leaf (`QSpider(1, 0, 0)` / `PSpider(1, 0, 0)`) -> a plain
  `intrinsic.Measurement(pi/2)` / `Measurement(0)` (measuring x or p directly) — these arise if
  `diagram` contains an already-`.expand()`-ed `MeasurementGate`/state rather than the compact
  form, which is otherwise handled directly.

### Feedforward on emission

A `(1, 0)` `QSpider`/`PSpider` effect with phase *exactly* `ZxPoly({1: -m})` for a single symbol
`m` — the leaf shape `cvzx.passes.completion.complete_diagram()` produces to close an open
output port — is recognized specially: it becomes a plain `intrinsic.Measurement` (`pi/2`/`0`
for `QSpider`/`PSpider`, same as the bare zero-phase case above), and the resulting mqc3
`Operation` is tracked against `m` for the rest of the walk.

Any later leaf whose own parameter is affine (`slope*m + intercept`, for that same single
symbol `m`) in a tracked symbol has that parameter translated into an mqc3
`FeedForward[MeasuredVariable]` (via `mqc3.feedforward.ff_to_mul_constant`/`ff_to_add_constant`,
composed to match the affine relationship) instead of a plain float — the exact algebraic
inverse of `from_circuit_repr`'s own feedforward reconstruction above.

A parameter depending on more than one symbol, on a symbol with no tracked measurement, or
nonlinearly on its symbol, raises `NotImplementedError` (first and third cases) or
`cvzx.exceptions.UnboundMeasurementError` (second case).

### Not (yet) supported -- raises `NotImplementedError`

- `CubicPhaseGate`: a genuinely non-Gaussian gate; mqc3's intrinsic set is Gaussian-only.
- Any state/effect leaf with a nonzero phase polynomial, other than the single
  measurement-effect shape described above.
- A symbolic (parametric, unresolved) gate parameter that isn't a tracked feedforward symbol as
  described above (a genuinely free, measurement-unrelated symbol; more than one symbol at
  once; a nonlinear function of one symbol).
- A `Diagram` with `num_inputs != 0`: mqc3 `CircuitRepr` has no concept of an externally
  supplied input mode — every mode must originate from a state leaf.
- `ContractedDiagram` anywhere in `diagram` (same limitation `normalize_diagram` itself
  documents).

Since `cvzx` has no numeric simulation backend of its own, the round trip is checked
structurally rather than against a reference statevector — that the translated circuit is
well-formed and that mqc3's own `DependencyDAG` accepts it:

```python
from mqc3.graph.embed.dep_dag import DependencyDAG

dag = DependencyDAG(circuit2)
assert dag.dag.number_of_nodes() > 0
```

## Lowering straight to a `DependencyDAG`

`cvzx.lowering.lowering.graph_to_dependency_dag` skips the `to_circuit_repr`/`DependencyDAG`
boilerplate at the call site — it dispatches to a registered `LoweringBackend` that performs
the `Diagram -> DependencyDAG` step however it sees fit, defaulting to `"mqc3"`. `diagram` must
already be closed first (`num_inputs == num_outputs == 0` — see below, and {doc}`optimization`):

### Closing open boundaries first

`optimize()`'s rewrite rules deliberately never terminate a wire on their own — a boundary
state/effect sitting in the graph would otherwise block spider fusion, chain reduction, and
identity removal during the rewrite loop itself — so a diagram destined for lowering generally
still has open ports. `cvzx.passes.completion.complete_diagram`/`complete_boundaries` are the
dedicated post-`optimize()` pass that closes them.

`complete_diagram(diagram)` appends one fresh, symbolically-labeled measurement effect per open
output port: a `(1, 0)` `QSpider`/`PSpider` with phase `ZxPoly({1: -m})` for a fresh
`sympy.Symbol` `m` — the standard CV-ZX notation for "the idealized homodyne effect whose own
outcome is `m`" (see `MeasurementGate`'s own docstring for the same `QSpider(1, 0, 0)`
convention at a *fixed* outcome). `QSpider` measures the x-quadrature (mqc3
`intrinsic.Measurement(theta=pi/2)`); `PSpider` measures the p-quadrature (`theta=0`).
`complete_boundaries(diagram)` does the same for output ports and additionally closes any open
input ports first with a fresh idealized (zero-phase) state leaf — there's no outcome to name
on the input side, so no symbol is minted for it. The result is a diagram every
`GateRegister.input_states`/`measurement_nodes` entry bounds a complete wire for, the
precondition `extract_dependency_dag()` (below) relies on.

A boundary-completing layer is built the same way any other closing layer in this codebase is:
a `TensorDiagram` of one 1-mode effect per output port, composed onto `diagram`.
`Diagram.compose()`/`TensorDiagram` construction already validate that the closing layer's
total input arity matches `diagram`'s total output arity port-for-port (see
`cvzx.exceptions.ArityMismatchError`), so whether a given output port came from a 1-mode gate or
is one of a 2-mode gate's two outputs makes no difference — no special-case handling is needed
for a straddling 2-mode gate.

Each output port's fresh symbol `m` is known (via the returned `bindings`) only *after*
completion runs, since that's when the measurement leaf (and its node id) is actually created.
The recommended flow is to build any feedforward-dependent gate (e.g.
`DisplacementGate(m, param_measurement_map={m: {node_id}})`) using `bindings` *after* calling
`complete_diagram()`/`complete_boundaries()`. If a gate was already built earlier referencing a
symbol that turns out to need re-pointing at a different measurement node id (e.g. because the
diagram was assembled in pieces), use `cvzx.passes.completion.relink_measurement_symbol()` to
repoint every graph node that cites that symbol and resynchronize the registry (both backends)
in one step, rather than mutating node attributes by hand — note that a node's
`param_measurement_map` is the *same* dict object as the underlying `Diagram` instance's own
attribute (not a copy), so this intentionally also updates that `Diagram` object in place.

```python
from cvzx.lowering.lowering import graph_to_dependency_dag

dep_dag = graph_to_dependency_dag(diagram)  # backend="mqc3" by default
```

### The bundled backends

The bundled `"mqc3"` backend (`Mqc3ReferenceBackend`) is `to_circuit_repr` followed by mqc3's
own `DependencyDAG(circuit)` constructor — every limitation of `to_circuit_repr` above applies
transitively to it. A second bundled backend, `"cvzx-direct"` (`CvzxDirectBackend`), skips the
`CircuitRepr` round-trip entirely: it discovers execution order directly from the diagram's
own `CVZXGraph` structure (dual-backend, via `cvzx.lowering.dag`), reusing this module's own
per-leaf translators rather than duplicating them — both are verified to produce an isomorphic
`DependencyDAG` for the same input. A QPU that needs a different `Diagram -> DependencyDAG`
lowering registers its own `LoweringBackend` subclass with the `register_backend` class
decorator:

```python
from cvzx.lowering.lowering import LoweringBackend, register_backend


@register_backend("my_qpu")
class MyQpuBackend(LoweringBackend):
    def to_dependency_dag(self, diagram):
        ...
```

`list_backends()` lists every currently-registered backend name, and `get_backend(name)` looks
one up directly; `graph_to_dependency_dag(diagram, backend="my_qpu")` then dispatches to it —
nothing else in the pipeline needs to change. From there, `DependencyDAG` is ready for mqc3's
own `GraphEmbedder`/`GraphRepr`/`MachineryRepr` chain, which is outside `cvzx`'s own scope (see
{doc}`../dev_guide/architecture`).

#### Why a `LoweringBackend` plugin registry

Different QPUs can require different lowering strategies once a circuit leaves cvzx's
diagrammatic representation. mqc3 already has exactly this kind of plugin point *downstream* of
`DependencyDAG`: `mqc3.graph.embed.embed.GraphEmbedder` is an abstract base class with concrete
per-strategy subclasses (`beamsearch.py`, `greedy.py`) that each embed a `DependencyDAG` into a
concrete `GraphRepr` differently, and `GraphRepr` is in turn lowered toward a machinery
representation via `mqc3.machinery`. `DependencyDAG` itself is QPU-agnostic — it only encodes
per-mode operation dependencies and feedforward edges, not anything hardware-specific — so it is
the natural, stable interface for cvzx to hand off to mqc3's own machinery. `cvzx.lowering.lowering`
provides the analogous plugin point for the one step still missing, `Diagram -> DependencyDAG`:
a `LoweringBackend` is any strategy for performing that step, backends register themselves under
a name via `register_backend`, and adding support for a QPU that needs a different lowering
means writing and registering one more `LoweringBackend` subclass — `get_backend`/
`graph_to_dependency_dag` and every existing backend are untouched. The bundled `"mqc3"` backend
is deliberately the simplest correct implementation, built on mqc3's own, already-tested
`_DependencyBuilder.from_circuit()` rather than reimplementing any dependency-graph logic.

### How `extract_dependency_dag` discovers execution order

`cvzx.lowering.dag.extract_dependency_dag` (the engine behind the `"cvzx-direct"` backend)
builds the exact same kind of `DependencyDAG` the `"mqc3"` backend does, but via a single
deterministic forward sweep over the `CVZXGraph`'s own node/edge structure instead of first
canonicalizing into `normalize_diagram`'s alternating stages — so it tolerates diagram shapes
that form isn't required to (e.g. an explicit `Swap`, or a `ContractedDiagram`
`complete_boundaries()`/the translators below can still resolve). The sweep:

1. `complete_boundaries()` (unless `complete=False`) closes every open input/output port first,
   so every wire is bounded by a real `input_states`/`measurement_nodes` node.
2. Every `input_states` node starts a fresh mode (a monotonically increasing integer, exactly as
   `cvzx.lowering.bridges.mqc3`'s own `_ModeCounter` assigns them).
3. A node is visited once both of its dependencies are satisfied: every input port has a mode
   threaded into it from an already-visited node (the "wire-ready" condition, propagated forward
   along `"composition"`/`"contracted_internal"` edges port-for-port), and every measurement id
   any of its parameters' `param_measurement_map` cites has itself already been visited (the
   "classical-ready" condition — this is what guarantees a feedforward edge never points at an
   unvisited measurement). A 1-mode leaf's single output port inherits its input's mode; a
   2-mode leaf's two output ports inherit its two inputs' modes unchanged (matching
   `to_circuit_repr`'s own convention that a wide gate never advances the mode counter); a
   measurement effect (or any other 1-in-0-out leaf) ends its mode's thread.
4. Container nodes (`kind == "container"`) are skipped entirely — only `"proper"`/`"compact"`
   leaves are visited.

The resulting mode-ordered leaf sequence is fed through the same per-leaf translators
`cvzx.lowering.bridges.mqc3` already uses (`_apply_1mode_leaf`/`_apply_2mode_leaf`, including
their existing `FeedForward` support) to build a `CircuitRepr`, which is then handed to
`DependencyDAG` — no cvzx-specific dependency-graph logic is reimplemented for the actual op
translation, only the traversal order. `mqc3.graph.embed.dep_dag.DependencyDAG` is not itself
backend-specific (its `.dag` is always a plain `networkx.DiGraph`, and its constructor only
accepts a `CircuitRepr`/`GraphRepr`) — there is no rustworkx-backed variant of it to build
instead. What *is* backend-aware is the traversal: `extract_dependency_dag()` walks whichever
`CVZXGraph` it's given (`networkx`- or `rustworkx`-backed, dispatched via `cvzx.backend`).

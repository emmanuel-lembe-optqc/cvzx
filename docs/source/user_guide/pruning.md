# Pruning near-identity gates

A Gaussian gate whose parameter is close to its identity value barely changes the state, but on hardware it can still
cost noise. On a time-domain cluster such as MoQuren, every macronode step a gate adds teleports every live mode once.
`cvzx.passes.pruning.prune_small_gaussian_gates` removes such gates before they reach the hardware.

```python
from cvzx.lowering.bridges.claveles import from_circuit_repr, to_circuit_repr
from cvzx.passes.pruning import prune_small_gaussian_gates

result = prune_small_gaussian_gates(from_circuit_repr(circuit), {"beam_splitter": 1e-3, "rotation": 1e-4})
pruned_circuit = to_circuit_repr(result.diagram)
print([(p.kind, p.distance) for p in result.pruned])
```

## How a gate is judged

Each gate type has its own **distance from the identity**, in the units its parameter acts on
(`cvzx.passes.pruning.gate_distance`):

| type (`GATE_TYPES`) | gate | distance |
|---|---|---|
| `rotation` | `PhaseRotationGate(theta)` | \|theta\| folded into (-pi, pi] |
| `squeezing` | `SqueezingGate(tau)` | \|ln tau\| (the identity is tau = 1; tau <= 0 is never pruned) |
| `squeezing45` | `Squeezing45Gate(theta)` | \|ln cot theta\| |
| `arbitrary` | `ArbitraryGate(alpha, beta, lam)` | max(\|lam\|, \|alpha + beta\|) |
| `displacement` | `DisplacementGate(alpha)` | \|alpha\| |
| `shear` | `ShearXInvariantGate(kappa)`, `ShearPInvariantGate(eta)` | \|kappa\|, \|eta\| |
| `beam_splitter` | `BeamsplitterGate(theta)` | \|theta\| folded into (-pi, pi] |
| `controlled_z`, `controlled_sum` | `ControlledZGate(g)`, `ControlledSumGate(g)` | \|g\| |
| `two_mode_shear` | `TwoModeShearGate(a, b)` | max(\|a\|, \|b\|) |

`epsilon` is one threshold for every type, or a dict per type name (types not named are kept), because the types act
on different scales: 10⁻³ of displacement and 10⁻³ of log-squeezing do not move a state by the same amount.

A gate is **never** pruned when its parameter is symbolic, which includes every feedforward target (a gate with a
`param_measurement_map`): removing it would orphan the measurement its parameter depends on. Non-Gaussian gates,
measurements and states are not touched.

## Why identity wires, and when they disappear

Pruning works on compact gates, before expansion. Snapping a parameter to its identity value is not always
expandable: `BeamsplitterGate(0)` divides by tan 0, `ControlledZGate(0)` builds `Sq(1/sqrt(0))`, and the squeezing
identity is tau = 1. A pruned gate is therefore replaced by identity wires. Inside a composition the wires are dropped
outright and the wiring maps on either side are composed, so the neighbouring gates meet and `optimize` can fuse them
(two squeezers that a tiny rotation kept apart become one). Inside a tensor product the wire stays, as cvzx always
writes "nothing on this mode".

## Noise-aware pruning

`keep(gate) -> bool` lets an outside cost model veto a candidate. Pass `epsilon=math.inf` to make every Gaussian gate a
candidate and let `keep` decide alone. moquren-emu (`moquren_emu.io.pruning_cost`) prices each gate on the emulated
MoQuren: the macronode steps it costs, split into short-delay and fiber teleportations, against the error of removing
it.

`optimize(diagram, prune_epsilon=..., prune_keep=...)` runs the pass first and then the usual rewrite rules.

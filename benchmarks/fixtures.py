"""Diagram fixtures for the optimization-quality benchmark suite.

Each fixture is a zero-argument factory returning a **fresh** diagram (safe
to call repeatedly, including once per `assume_infinite_squeezing` setting,
without aliasing) -- `Diagram.id_counter` (`cvzx.ir.base`) is a shared
global counter, so reusing one diagram instance across multiple `optimize()`
calls would corrupt node identities between runs.

Built from the same construction idioms as `tests/passes/test_optimize.py`
(`_build_four_mode_circuit`, reused here as `four_mode_csum`) and
`tests/ir/test_gates.py`, rather than depending on the `tests/` package
(which isn't installed/importable outside a test run) or on the
`examples/` notebooks (whose API is still being migrated -- see the dev
guide).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import pi

from cvzx.ir.base import CompositionDiagram, PSpider, QSpider, TensorDiagram, ZxPoly
from cvzx.ir.gates import ControlledSumGate, CubicPhaseGate, PhaseRotationGate, SqueezingGate

if TYPE_CHECKING:
    from collections.abc import Callable


def _identity() -> QSpider:
    """A single 1-in/1-out zero-phase identity wire (fresh instance each call).

    Returns
    -------
    QSpider
    """
    return QSpider(1, 1, ZxPoly({}))


def small_ancilla_csum() -> CompositionDiagram:
    """2 modes: one ancilla feeding one CSUM -- CopyRule's textbook eliminable case.

    Under `assume_infinite_squeezing=True`, the ancilla is an idealized
    eigenstate `CopyRule` can push straight through the CSUM's control leg,
    eliminating both leaves entirely. Under `assume_infinite_squeezing=False`,
    neither rule applies and the diagram is a fixed point already -- the
    smallest example of "not everything reduces without the assumption".

    Returns
    -------
    CompositionDiagram
    """
    ancilla = PSpider(0, 1, ZxPoly({1: 2}))
    csum = ControlledSumGate(control=2, target=1)
    return CompositionDiagram([TensorDiagram([ancilla, _identity()]), csum])


def four_mode_csum() -> CompositionDiagram:
    """4 modes, three CSUM gates: one ancilla-eliminable, one bare, two mergeable.

    Identical in shape to `tests/passes/test_optimize.py::_build_four_mode_circuit`,
    reconstructed here so the benchmark suite doesn't depend on importing
    the `tests/` package. See that function's docstring for the full
    layer-by-layer reduction story under each `assume_infinite_squeezing`
    setting.

    Returns
    -------
    CompositionDiagram
    """
    ancilla = PSpider(0, 1, ZxPoly({1: 2}))
    csum_a = ControlledSumGate(control=2, target=1)
    csum_b = ControlledSumGate(control=2, target=1)
    csum_c1 = ControlledSumGate(gain=0.5, control=2, target=1)
    csum_c2 = ControlledSumGate(gain=0.5, control=2, target=1)

    seg_a = CompositionDiagram([TensorDiagram([ancilla, _identity()]), csum_a])
    layer1 = TensorDiagram([seg_a, _identity(), _identity()])
    layer2 = TensorDiagram([_identity(), csum_b, _identity()])
    seg_c = CompositionDiagram([csum_c1, csum_c2])
    layer3 = TensorDiagram([_identity(), _identity(), seg_c])
    layer4 = TensorDiagram([
        PhaseRotationGate(pi / 5),
        _identity(),
        _identity(),
        SqueezingGate(tau=1.5),
    ])
    return CompositionDiagram([layer1, layer2, layer3, layer4])


def six_mode_with_cubic_phase() -> CompositionDiagram:
    """6 modes: two ancilla/CSUM pairs, a mergeable CSUM chain, and a CubicPhaseGate.

    A scaled-up `four_mode_csum` with a second ancilla-eliminable CSUM pair
    (modes 4-5) and a `CubicPhaseGate` on mode 3, so `non_clifford_phases`
    has something nonzero to report -- `four_mode_csum`/`small_ancilla_csum`
    are both entirely Gaussian.

    Returns
    -------
    CompositionDiagram
    """
    ancilla_1 = PSpider(0, 1, ZxPoly({1: 2}))
    csum_a1 = ControlledSumGate(control=2, target=1)
    ancilla_2 = PSpider(0, 1, ZxPoly({1: 2}))
    csum_a2 = ControlledSumGate(control=2, target=1)
    csum_b = ControlledSumGate(control=2, target=1)
    csum_c1 = ControlledSumGate(gain=0.5, control=2, target=1)
    csum_c2 = ControlledSumGate(gain=0.5, control=2, target=1)

    # 4 real inputs (feeding modes 1, 2, 4, 5) plus the two ancillas
    # (producing modes 0 and 3) give 6 modes onward. seg_a1/seg_a2 each
    # take 1 real input and emit 2 outputs (their ancilla's new mode plus
    # the real mode passed through the CSUM), so layer1's 4 inputs become
    # 6 outputs, in mode order [0, 1, 2, 3, 4, 5].
    seg_a1 = CompositionDiagram([TensorDiagram([ancilla_1, _identity()]), csum_a1])
    seg_a2 = CompositionDiagram([TensorDiagram([ancilla_2, _identity()]), csum_a2])
    layer1 = TensorDiagram([seg_a1, _identity(), seg_a2, _identity()])
    layer2 = TensorDiagram([_identity(), csum_b, _identity(), _identity(), _identity()])
    seg_c = CompositionDiagram([csum_c1, csum_c2])
    layer3 = TensorDiagram([_identity(), _identity(), _identity(), _identity(), seg_c])
    layer4 = TensorDiagram([
        PhaseRotationGate(pi / 5),
        _identity(),
        _identity(),
        CubicPhaseGate(0.2),
        SqueezingGate(tau=1.5),
        _identity(),
    ])
    return CompositionDiagram([layer1, layer2, layer3, layer4])


FIXTURES: dict[str, Callable[[], CompositionDiagram]] = {
    "small_ancilla_csum": small_ancilla_csum,
    "four_mode_csum": four_mode_csum,
    "six_mode_with_cubic_phase": six_mode_with_cubic_phase,
}

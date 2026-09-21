"""The three example notebooks' initial diagrams, extracted for the harness.

Re-declares the pre-optimization `Diagram` construction from each
notebook's early cells directly (no notebook execution) so
`rule_order_experiment.py` can run against them repeatedly. Each function
returns `(gadget, target_diagram_or_None)` -- `target_diagram` is the
paper's claimed closed form, where the notebook states one, for a hard
correctness check beyond just "did the shape change."
"""

from __future__ import annotations

from sympy import I, sin, sqrt, symbols, tan
from sympy import pi as sympy_pi

from cvzx.ir.base import CompositionDiagram, ContractedDiagram, PSpider, QSpider, TensorDiagram, ZxPoly
from cvzx.ir.gates import BeamsplitterGate, DisplacementGate, PhaseRotationGate, SqueezingGate

zero_phase = ZxPoly({})


def cubic_phase_injection() -> tuple[CompositionDiagram, QSpider]:
    """`examples/cubic_phase_injection.ipynb`, cell 3 -- the simplest of the three.

    Returns
    -------
    tuple[CompositionDiagram, QSpider]
        (gadget, target_diagram) -- target is `QSpider(1,1,gamma*x**3)`.
    """
    m, gamma = symbols("m gamma", real=True)

    meas = PSpider(1, 0, ZxPoly({1: -m}), True)

    stage0 = TensorDiagram([
        QSpider(1, 1, zero_phase),
        QSpider(0, 1, ZxPoly({3: gamma}), True),
    ])
    stage1 = ContractedDiagram(QSpider(2, 1, zero_phase), PSpider(1, 2, zero_phase), [], [], [1], [0])
    stage2 = TensorDiagram([
        QSpider(
            1,
            1,
            ZxPoly({1: -3 * gamma * m**2, 2: 3 * gamma * m}),
            True,
            True,
            {meas.id},
            {m: {meas.id}},
        ),
        meas,
    ])

    gadget = CompositionDiagram([stage0, stage1, stage2])
    target = QSpider(1, 1, ZxPoly({3: gamma}), True)
    return gadget, target


def measurement_induced_squeezer() -> tuple[CompositionDiagram, SqueezingGate]:
    """`examples/measurement_induced_squeezer.ipynb`, cell 3 -- the historical regression canary.

    Returns
    -------
    tuple[CompositionDiagram, SqueezingGate]
        (gadget, target_diagram) -- target is `SqueezingGate(sin(theta))`,
        matching the paper. `optimize()` regressing to
        `SqueezingGate(sin(theta)**2/cos(theta))` was the bug documented
        in `CHANGELOG.md`'s "optimize() could converge to a different..."
        entry.
    """
    m, theta = symbols("m theta", real=True)

    beamsplitter_expanded = BeamsplitterGate(theta, True).expand()
    measurement = QSpider(1, 0, ZxPoly({1: -m}), parametric=True)

    stage0 = TensorDiagram([
        QSpider(1, 1, zero_phase),
        CompositionDiagram([PSpider(0, 1, zero_phase), QSpider(1, 1, zero_phase)]),
    ])
    stage1 = TensorDiagram([beamsplitter_expanded])
    stage2 = TensorDiagram([
        measurement,
        DisplacementGate(
            I * m / (sqrt(2) * tan(theta)),
            True,
            True,
            {measurement.id},
            {theta: {measurement.id}},
        ).expand(),
    ])

    gadget = CompositionDiagram([stage0, stage1, stage2])
    target = SqueezingGate(sin(theta), True)
    return gadget, target


def example_4_nonunit_gain() -> tuple[CompositionDiagram, QSpider]:
    """`examples/example_4_cubic_phase_injection_nonunit_gain.ipynb`, cell 3 -- the complex, known-hard case.

    Returns
    -------
    tuple[CompositionDiagram, QSpider]
        (gadget, target_diagram) -- target is `QSpider(1,1,gamma*x**3)`;
        per the notebook's own markdown this case is NOT expected to
        fully reduce to it because of the scalar-cap limitation
        (`TerminalAbsorptionRule`/`ChainReductionRule` both explicitly
        refuse to fold a state directly composed into its own opposite
        effect with nothing left over -- this codebase has no scalar
        bookkeeping).
    """
    m1, m2, theta1, lambda1, gamma = symbols("m1 m2 theta1 lambda1 gamma", real=True)
    meas1 = PSpider(1, 0, ZxPoly({1: -m1}), True)
    meas2 = QSpider(1, 0, ZxPoly({1: -m2}), True)

    stage0 = TensorDiagram([
        QSpider(1, 1, zero_phase),
        PSpider(0, 1, zero_phase),
        QSpider(0, 1, ZxPoly({3: gamma}), True),
    ])
    stage1 = TensorDiagram([BeamsplitterGate(sympy_pi / 4).expand(), QSpider(1, 1, zero_phase)])
    stage2 = TensorDiagram([QSpider(1, 1, zero_phase), BeamsplitterGate(sympy_pi / 4).expand()])
    stage3 = TensorDiagram([
        DisplacementGate(I * lambda1, True, True, {meas2.id}, {lambda1: {meas2.id}}),
        CompositionDiagram([PhaseRotationGate(theta1, True, True, {meas1.id}, {theta1: {meas1.id}}), meas2]),
        meas1,
    ])
    gadget = CompositionDiagram(
        [stage0, stage1, stage2, stage3],
        {0: {0: 0, 1: 1, 2: 2}, 1: {0: 0, 1: 2, 2: 1}, 2: {0: 0, 1: 2, 2: 1}},
    )
    target = QSpider(1, 1, ZxPoly({3: gamma}), True)
    return gadget, target


ALL = {
    "cubic_phase_injection": cubic_phase_injection,
    "measurement_induced_squeezer": measurement_induced_squeezer,
    "example_4_nonunit_gain": example_4_nonunit_gain,
}

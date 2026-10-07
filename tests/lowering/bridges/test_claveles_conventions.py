"""The claveles bridge's conventions, pinned numerically.

Every claveles intrinsic gate is applied to squeezed input states and read out by homodyne measurements; the circuit is
converted to a cvzx `Diagram` and back. The exact moments of the measured values must be unchanged (the round trip may
rewrite a rotated input state as a rotation after an unrotated one, which is the same state).
Gate matrices are transcribed from claveles' own docstrings (`claveles.circuit.ops.intrinsic`), in (x1, p1, x2, p2)
order with hbar = 1, so the test does not trust either side of the bridge.
"""

import copy
import math

import numpy as np
import pytest

from claveles.circuit import CircuitRepr, HardwareConstrainedSqueezedState
from claveles.circuit.ops import intrinsic

from cvzx.lowering.bridges.claveles import from_circuit_repr, placed_operations, to_circuit_repr


def _rot(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [s, c]])


def _xxpp_to_xpxp(m: np.ndarray) -> np.ndarray:
    p = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return p @ m @ p.T


def _beam_splitter(alpha: float, beta: float) -> np.ndarray:
    s, d = alpha + beta, alpha - beta
    cs, ss, cd, sd = math.cos(s), math.sin(s), math.cos(d), math.sin(d)
    return np.array([
        [cs * cd, ss * sd, -ss * cd, sd * cs],
        [ss * sd, cs * cd, sd * cs, -ss * cd],
        [ss * cd, -sd * cs, cs * cd, ss * sd],
        [-sd * cs, ss * cd, ss * sd, cs * cd],
    ])


def _matrix(op) -> np.ndarray:  # noqa: ANN001
    """Heisenberg matrix of a claveles intrinsic gate, from its docstring."""
    name = type(op).__name__
    p = [float(v) for v in op.parameters()]
    if name == "PhaseRotation":
        return _rot(p[0])
    if name == "ShearXInvariant":
        return np.array([[1.0, 0.0], [2 * p[0], 1.0]])
    if name == "ShearPInvariant":
        return np.array([[1.0, 2 * p[0]], [0.0, 1.0]])
    if name == "Squeezing":
        return _rot(-math.pi / 2) @ np.diag([math.tan(p[0]), 1 / math.tan(p[0])])
    if name == "Squeezing45":
        return _rot(-math.pi / 4) @ np.diag([math.tan(p[0]), 1 / math.tan(p[0])]) @ _rot(math.pi / 4)
    if name == "Arbitrary":
        return _rot(p[0]) @ np.diag([math.exp(-p[2]), math.exp(p[2])]) @ _rot(p[1])
    if name == "ControlledZ":
        g = p[0]
        return _xxpp_to_xpxp(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, g, 1, 0], [g, 0, 0, 1]], dtype=float))
    if name == "TwoModeShear":
        a, b = p
        return _xxpp_to_xpxp(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [2 * a, b, 1, 0], [b, 2 * a, 0, 1]], dtype=float))
    if name == "BeamSplitter":
        sqrt_r, theta_rel = p
        h = math.acos(sqrt_r) / 2
        return _xxpp_to_xpxp(_beam_splitter(theta_rel / 2 + h, theta_rel / 2 - h))
    msg = f"no reference matrix for {name}"
    raise NotImplementedError(msg)


R_INPUT = 0.6  # squeezing of every input state (the comparison only needs both sides to use the same one)


def _input_moments(state) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    """HardwareConstrainedSqueezedState(phi): the x-squeezed vacuum rotated by R(phi); an open mode is phi = 0."""
    phi = 0.0 if state is None else float(state.phi)
    rot = _rot(phi)
    return np.zeros(2), rot @ np.diag([math.exp(-2 * R_INPUT), math.exp(2 * R_INPUT)]) / 2 @ rot.T


def _outcome_moments(circuit: CircuitRepr) -> tuple[np.ndarray, np.ndarray]:
    """Exact (mean, covariance) of the homodyne outcomes, in measurement order of (mode, angle)."""
    c = copy.deepcopy(circuit)
    c.convert_std_ops_to_intrinsic()
    n = c.n_modes
    mu0, v0 = np.zeros(2 * n), np.zeros((2 * n, 2 * n))
    for m, state in c.placed_states.items():
        mean, cov = _input_moments(state)
        mu0[2 * m : 2 * m + 2], v0[2 * m : 2 * m + 2, 2 * m : 2 * m + 2] = mean, cov
    s, d = np.eye(2 * n), np.zeros(2 * n)
    rows, offsets, keys = [], [], []
    for op, modes in placed_operations(c):
        idx = [2 * m + k for m in modes for k in (0, 1)]
        name = type(op).__name__
        if name == "Displacement":
            d[idx] += [float(v) for v in op.parameters()]
        elif name == "Measurement":
            theta = float(op.parameters()[0])
            v = np.zeros(2 * n)
            v[idx] = [math.sin(theta), math.cos(theta)]
            rows.append(v @ s)
            offsets.append(v @ d)
            keys.append(modes[0])
        else:
            u = np.eye(2 * n)
            u[np.ix_(idx, idx)] = _matrix(op)
            s, d = u @ s, u @ d
    order = np.argsort(keys, kind="stable")
    lin, offs = np.array(rows)[order], np.array(offsets)[order]
    return lin @ mu0 + offs, lin @ v0 @ lin.T


def _round_trip(circuit: CircuitRepr) -> CircuitRepr:
    return to_circuit_repr(from_circuit_repr(circuit))


def _circuit(op, n_modes: int, *, read: tuple[float, ...] = (0.0, math.pi / 2)) -> list[CircuitRepr]:  # noqa: ANN001
    """The gate on n_modes squeezed inputs, then every mode read at each angle of ``read`` (one circuit per angle)."""
    out = []
    for theta in read:
        c = CircuitRepr("convention")
        for m in range(n_modes):
            c.Q(m) | HardwareConstrainedSqueezedState(phi=0.3 * m)
        c.Q(*range(n_modes)) | op
        for m in range(n_modes):
            c.Q(m) | intrinsic.Measurement(theta + 0.2 * m)
        out.append(c)
    return out


RNG = np.random.default_rng(7)
GATES = [
    ("Displacement", lambda: intrinsic.Displacement(*RNG.normal(size=2)), 1),
    ("PhaseRotation", lambda: intrinsic.PhaseRotation(RNG.uniform(-3, 3)), 1),
    ("ShearXInvariant", lambda: intrinsic.ShearXInvariant(RNG.normal()), 1),
    ("ShearPInvariant", lambda: intrinsic.ShearPInvariant(RNG.normal()), 1),
    ("Squeezing", lambda: intrinsic.Squeezing(RNG.uniform(0.3, 1.2)), 1),
    ("Squeezing45", lambda: intrinsic.Squeezing45(RNG.uniform(0.3, 1.2)), 1),
    ("Arbitrary", lambda: intrinsic.Arbitrary(*RNG.uniform(-1, 1, size=3)), 1),
    ("ControlledZ", lambda: intrinsic.ControlledZ(RNG.normal()), 2),
    ("BeamSplitter", lambda: intrinsic.BeamSplitter(RNG.uniform(0.1, 0.9), RNG.uniform(-2, 2)), 2),
    ("TwoModeShear", lambda: intrinsic.TwoModeShear(*RNG.normal(size=2)), 2),
]


@pytest.mark.parametrize(("name", "make", "n_modes"), GATES, ids=[g[0] for g in GATES])
def test_round_trip_keeps_every_readout(name, make, n_modes):
    for circuit in _circuit(make(), n_modes):
        mean, cov = _outcome_moments(circuit)
        mean2, cov2 = _outcome_moments(_round_trip(circuit))
        np.testing.assert_allclose(mean2, mean, atol=1e-9, err_msg=name)
        np.testing.assert_allclose(cov2, cov, atol=1e-9, err_msg=name)


def test_reference_matrices_are_symplectic():
    omega1 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    for name, make, n_modes in GATES:
        if name == "Displacement":
            continue
        m = _matrix(make())
        omega = np.kron(np.eye(n_modes), omega1)
        np.testing.assert_allclose(m @ omega @ m.T, omega, atol=1e-12, err_msg=name)


def _bs_cvzx(theta: float) -> np.ndarray:
    """cvzx BeamsplitterGate(theta) = exp(-i theta (q1 p2 - p1 q2)): q1 -> c q1 - s q2, q2 -> s q1 + c q2 (same for p)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, -s, 0], [0, c, 0, -s], [s, 0, c, 0], [0, s, 0, c]])


def _unitary_matrix(circuit: CircuitRepr) -> np.ndarray:
    m = np.eye(2 * circuit.n_modes)
    for op, modes in placed_operations(circuit):
        idx = [2 * k + j for k in modes for j in (0, 1)]
        u = np.eye(2 * circuit.n_modes)
        u[np.ix_(idx, idx)] = _matrix(op)
        m = u @ m
    return m


@pytest.mark.parametrize("theta", [0.3, math.pi / 4, 1.2, -0.4, 2.0, -2.5, math.pi])
def test_beam_splitter_export_matches_cvzx_semantics(theta):
    from cvzx.ir.gates import BeamsplitterGate
    from cvzx.ir.base import QSpider, TensorDiagram, CompositionDiagram, ZxPoly

    state = TensorDiagram([QSpider(0, 1, ZxPoly({})), QSpider(0, 1, ZxPoly({}))])
    circuit = to_circuit_repr(CompositionDiagram([state, BeamsplitterGate(theta)]))
    for op, _ in placed_operations(circuit):
        if type(op).__name__ == "BeamSplitter":
            assert 0.0 <= float(op.parameters()[0]) <= 1.0
    np.testing.assert_allclose(_unitary_matrix(circuit), _bs_cvzx(theta), atol=1e-12)


@pytest.mark.parametrize(("gain", "control", "target"), [(0.7, 1, 2), (0.7, 2, 1), (-1.3, 1, 2), (1.0, 2, 1)])
def test_controlled_sum_export_matches_cvzx_semantics(gain, control, target):
    """cvzx CSUM(g) = exp(-i g q_c p_t): q_t -> q_t + g q_c and p_c -> p_c - g p_t, nothing else."""
    from cvzx.ir.base import CompositionDiagram, QSpider, TensorDiagram, ZxPoly
    from cvzx.ir.gates import ControlledSumGate

    state = TensorDiagram([QSpider(0, 1, ZxPoly({})), QSpider(0, 1, ZxPoly({}))])
    circuit = to_circuit_repr(CompositionDiagram([state, ControlledSumGate(gain, control=control, target=target)]))
    c, t = control - 1, target - 1
    expected = np.eye(4)
    expected[2 * t, 2 * c] = gain  # x_t += g x_c
    expected[2 * c + 1, 2 * t + 1] = -gain  # p_c -= g p_t
    np.testing.assert_allclose(_unitary_matrix(circuit), expected, atol=1e-12)


def test_feedforward_survives_the_round_trip():
    """A displacement fed forward from a measurement keeps pointing at that measurement's placement."""
    from claveles.circuit.ops._base import MeasuredVariable
    from claveles.feedforward import FeedForward
    from claveles.graph.embed.dep_dag import DependencyDAG

    c = CircuitRepr("ff")
    c.Q(0) | HardwareConstrainedSqueezedState(phi=0.0)
    c.Q(1) | HardwareConstrainedSqueezedState(phi=0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(1.0)
    m = c.Q(0) | intrinsic.Measurement(math.pi / 2)
    c.Q(1) | intrinsic.Displacement(m, 0.0)
    c.Q(1) | intrinsic.Measurement(0.0)
    r = _round_trip(c)
    sources = []
    for op, _ in placed_operations(r):
        for param in op.parameters():
            if isinstance(param, FeedForward):
                var = param.variable
                assert isinstance(var, MeasuredVariable)
                placement = var.get_from_placement()
                assert type(placement.op).__name__ == "Measurement"
                sources.append(placement)
    assert sources, "the feedforward was lost"
    assert all(any(p is pl for pl in r.placements) for p in sources)
    DependencyDAG(r)


def test_pruned_circuit_stays_valid_and_close():
    """Import, prune a near-identity beam splitter and a tiny rotation, export: moments move by O(epsilon) only."""
    from claveles.graph.embed.dep_dag import DependencyDAG

    from cvzx.passes.pruning import prune_small_gaussian_gates

    c = CircuitRepr("prune")
    c.Q(0) | HardwareConstrainedSqueezedState(phi=0.0)
    c.Q(1) | HardwareConstrainedSqueezedState(phi=0.4)
    c.Q(0, 1) | intrinsic.ControlledZ(0.8)
    c.Q(0, 1) | intrinsic.BeamSplitter(math.cos(1e-4), 0.0)
    c.Q(1) | intrinsic.PhaseRotation(2e-4)
    c.Q(0) | intrinsic.Measurement(0.3)
    c.Q(1) | intrinsic.Measurement(1.0)
    res = prune_small_gaussian_gates(from_circuit_repr(c), 1e-3)
    assert {p.kind for p in res.pruned} >= {"beam_splitter"}
    pruned = to_circuit_repr(res.diagram)
    DependencyDAG(pruned)
    assert len(placed_operations(pruned)) < len(placed_operations(_round_trip(c)))
    mean, cov = _outcome_moments(c)
    mean2, cov2 = _outcome_moments(pruned)
    np.testing.assert_allclose(mean2, mean, atol=1e-3)
    np.testing.assert_allclose(cov2, cov, atol=5e-3)


def _random_circuit(n_modes: int, n_gates: int, seed: int) -> CircuitRepr:
    rng = np.random.default_rng(seed)
    c = CircuitRepr(f"random{seed}")
    for m in range(n_modes):
        c.Q(m) | HardwareConstrainedSqueezedState(phi=float(rng.uniform(0, math.pi)))
    for _ in range(n_gates):
        kind = rng.integers(5)
        if kind == 0:
            c.Q(int(rng.integers(n_modes))) | intrinsic.PhaseRotation(float(rng.uniform(-3, 3)))
        elif kind == 1:
            c.Q(int(rng.integers(n_modes))) | intrinsic.Displacement(*(float(v) for v in rng.normal(size=2)))
        else:
            a, b = (int(v) for v in rng.choice(n_modes, 2, replace=False))
            op = (
                intrinsic.ControlledZ(float(rng.normal()))
                if kind == 2
                else intrinsic.BeamSplitter(float(rng.uniform(0.1, 0.9)), float(rng.uniform(-2, 2)))
                if kind == 3
                else intrinsic.TwoModeShear(*(float(v) for v in rng.normal(size=2)))
            )
            c.Q(a, b) | op
    for m in range(n_modes):
        c.Q(m) | intrinsic.Measurement(float(rng.uniform(0, math.pi)))
    return c


@pytest.mark.parametrize("seed", range(20))
def test_random_multi_mode_circuits_round_trip(seed):
    """Gates on arbitrary modes of 3-5 mode circuits: the wiring between layers must survive the round trip."""
    circuit = _random_circuit(3 + seed % 3, 10, seed)
    back = to_circuit_repr(from_circuit_repr(circuit, normalize=False))
    assert _same_up_to_relabelling(_outcome_moments(circuit), _outcome_moments(back))


@pytest.mark.parametrize("seed", range(20))
def test_random_multi_mode_circuits_round_trip_normalized(seed):
    """The same through normalize_diagram (on import, and again inside to_circuit_repr): its stage wiring must use
    the {output: next input} convention it reads, or normalizing twice scrambles the circuit."""
    circuit = _random_circuit(3 + seed % 3, 10, seed)
    assert _same_up_to_relabelling(_outcome_moments(circuit), _outcome_moments(_round_trip(circuit)))


def _same_up_to_relabelling(a, b, atol: float = 1e-9) -> bool:
    """Equal outcome moments after some permutation of the measured modes (the exporter numbers modes afresh)."""
    from itertools import permutations

    (m1, c1), (m2, c2) = a, b
    return any(
        np.allclose(m2[list(p)], m1, atol=atol) and np.allclose(c2[np.ix_(p, p)], c1, atol=atol)
        for p in permutations(range(len(m1)))
    )

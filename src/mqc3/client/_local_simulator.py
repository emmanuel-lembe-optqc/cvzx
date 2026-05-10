"""Locally run circuit and graph representations with dependency tracking.

This module provides:
- Local simulation of CircuitRepr using StrawberryFields
- Local simulation of GraphRepr (MBQC patterns) with dependency tracking
- Shared execution logic for StrawberryFields programs
- Mode flow tracking inspired by convert.py's SearchState
- Dependency graph validation for graphs
- Parallel execution layer extraction
- Mode lifecycle tracking (creation to measurement)
"""


# pyright: reportUnusedExpression=false

from __future__ import annotations

try:
    import strawberryfields as sf
    from strawberryfields import ops as sf_ops
except ImportError as e:
    msg = """StrawberryFields is not installed.
Please install mqc3 with the sf option using:

pip install '<path/to/mqc3>[sf]'

or install StrawberryFields separately with:

pip install strawberryfields

"""
    raise ImportError(msg) from e


import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import acos, atan, log, pi, tan
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

import mqc3.circuit.ops.intrinsic as intrinsic_ops
import mqc3.circuit.ops.std as std_ops
from mqc3.circuit.result import CircuitOperationMeasuredValue, CircuitResult, CircuitShotMeasuredValue
from mqc3.circuit.state import BosonicState, GaussianState
from mqc3.constant import hbar
from mqc3.feedforward import FeedForward
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import (
    ArbitraryFirst as GraphArbitraryFirst,
)
from mqc3.graph.ops import (
    ArbitrarySecond as GraphArbitrarySecond,
)
from mqc3.graph.ops import (
    BeamSplitter as GraphBeamSplitter,
)
from mqc3.graph.ops import (
    ControlledZ as GraphControlledZ,
)
from mqc3.graph.ops import (
    Initialization as GraphInitialization,
)
from mqc3.graph.ops import Manual as GraphManual
from mqc3.graph.ops import (
    Measurement as GraphMeasurement,
)
from mqc3.graph.ops import (
    PhaseRotation as GraphPhaseRotation,
)
from mqc3.graph.ops import (
    ShearPInvariant as GraphShearPInvariant,
)
from mqc3.graph.ops import (
    ShearXInvariant as GraphShearXInvariant,
)
from mqc3.graph.ops import (
    Squeezing as GraphSqueezing,
)
from mqc3.graph.ops import (
    Squeezing45 as GraphSqueezing45,
)
from mqc3.graph.ops import TwoModeShear as GraphTwoModeShear
from mqc3.graph.ops import (
    Wiring as GraphWiring,
)
from mqc3.graph.result import GraphMacronodeMeasuredValue, GraphResult, GraphShotMeasuredValue

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqc3.circuit.program import CircOpParam, CircuitRepr
    from mqc3.graph import GraphRepr

logger = logging.getLogger(__name__)


# =============================================================================
# Helper functions for graph analysis
# =============================================================================


def find_measurement_for_mode(graph: GraphRepr, mode: int) -> int | None:
    """Find the macronode index where a mode is measured."""
    io_modes = graph.io_modes_dict()
    for (h, w), (left, up, right, down) in io_modes.items():
        op = graph.get_operation(h, w)
        if isinstance(op, GraphMeasurement):
            measured_mode = left if left != BLANK_MODE else up
            if measured_mode == mode:
                return graph.get_index(h, w)
    return None


# =============================================================================
# Mode Flow Tracker (inspired by SearchState from convert.py)
# =============================================================================


@dataclass
class ModeFlowState:
    """Tracks the flow of modes through the graph, inspired by SearchState."""

    left_modes: list[int] = field(default_factory=list)
    up_mode: int = BLANK_MODE
    measurement_outcomes: dict[int, float] = field(default_factory=dict)
    step: int = 0

    def __init__(self, n_local_macronodes: int):
        self.left_modes = [BLANK_MODE] * n_local_macronodes
        self.up_mode = BLANK_MODE
        self.measurement_outcomes = {}
        self.step = 0


class ModeFlowTracker:
    """
    Tracks the flow of modes through the graph, inspired by SearchState from convert.py.

    This class simulates the propagation of modes through macronodes, handling
    swaps, through operations, initialisations, and measurements.
    """

    def __init__(self, graph: GraphRepr):
        self.graph = graph
        self.n_local_macronodes = graph.n_local_macronodes
        self.n_steps = graph.n_steps

        # Current state (updated as we process)
        self._left_modes = [BLANK_MODE] * self.n_local_macronodes
        self._up_mode = BLANK_MODE

        # Store measurement outcomes for feedforward
        self._measurement_outcomes: dict[int | tuple[int, int, int], float] = {}

        # Track the path of each mode for debugging
        self._mode_path: dict[int, list[tuple[int, int]]] = defaultdict(list)

    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self._left_modes = [BLANK_MODE] * self.n_local_macronodes
        self._up_mode = BLANK_MODE
        self._measurement_outcomes.clear()
        self._mode_path.clear()

    def get_left_mode(self, h: int) -> int:
        """Get the mode currently at the left input of macronode (h, current step)."""
        return self._left_modes[h]

    def get_up_mode(self) -> int:
        """Get the mode currently at the up input of the current macronode."""
        return self._up_mode

    def process_macronode(self, h: int, w: int) -> dict[str, Any]:
        """
        Process a single macronode, updating mode flow.
        """
        op = self.graph.get_operation(h, w)
        left_in = self._left_modes[h]
        up_in = self._up_mode

        result = {
            "left_in": left_in,
            "up_in": up_in,
            "left_out": BLANK_MODE,
            "up_out": BLANK_MODE,
            "measurement": None,
            "measurement_angle": None,
            "operation_type": type(op).__name__,
        }

        # Track mode path
        if left_in != BLANK_MODE:
            self._mode_path[left_in].append((h, w))
        if up_in != BLANK_MODE:
            self._mode_path[up_in].append((h, w))

        if isinstance(op, GraphMeasurement):
            measured_mode = left_in if left_in != BLANK_MODE else up_in
            if measured_mode != BLANK_MODE:
                result["measurement"] = measured_mode
                result["measurement_angle"] = op.parameters[0] if op.parameters else 0.0
            result["left_out"] = BLANK_MODE
            result["up_out"] = BLANK_MODE

        elif isinstance(op, GraphInitialization):
            mode0, mode1 = op.initialized_modes
            if mode0 != BLANK_MODE:
                self._left_modes[h] = mode0
                self._mode_path[mode0].append((h, w))
            result["left_out"] = mode0 if mode0 != BLANK_MODE else BLANK_MODE
            result["up_out"] = mode1 if mode1 != BLANK_MODE else BLANK_MODE

        elif isinstance(op, GraphWiring):
            if op.swap:
                result["left_out"] = up_in
                result["up_out"] = left_in
            else:
                result["left_out"] = left_in
                result["up_out"] = up_in

        elif isinstance(
            op,
            (
                GraphPhaseRotation,
                GraphSqueezing,
                GraphSqueezing45,
                GraphShearXInvariant,
                GraphShearPInvariant,
                GraphArbitraryFirst,
                GraphArbitrarySecond,
            ),
        ):
            # Single-mode operations - preserve mode routing
            if op.swap:
                result["left_out"] = up_in
                result["up_out"] = left_in
            else:
                result["left_out"] = left_in
                result["up_out"] = up_in

        elif isinstance(op, (GraphControlledZ, GraphBeamSplitter, GraphTwoModeShear, GraphManual)):
            # Two-mode operations - preserve mode routing (typical for CV)
            # For these operations, modes pass through
            result["left_out"] = left_in
            result["up_out"] = up_in

        else:
            # Default: through behaviour
            result["left_out"] = left_in
            result["up_out"] = up_in

        # Update state for next macronodes
        self._left_modes[h] = result["left_out"]
        self._up_mode = result["up_out"]

        return result

    def set_measurement_outcome(self, mode: int, value: float) -> None:
        """Set the measurement outcome for a mode (for feedforward)."""
        self._measurement_outcomes[mode] = value

    def get_measurement_outcome(self, mode: int) -> float:
        """Get the measurement outcome for a mode."""
        return self._measurement_outcomes.get(mode, 0.0)

    def evaluate_parameter(self, param, default: float = 0.0) -> float:
        """Evaluate a parameter that may be a feedforward expression."""
        if param is None:
            return default
        if isinstance(param, (float, int)):
            return float(param)
        if isinstance(param, FeedForward):
            var = param.variable
            if hasattr(var, "mode"):
                outcome = self.get_measurement_outcome(var.mode)
                return param.func(outcome)
            elif hasattr(var, "get_from_operation"):
                h, w, bd = var.get_from_operation()
                return param.func(self.get_measurement_outcome(h))
        return default

    def get_mode_path(self, mode: int) -> list[tuple[int, int]]:
        """Get the path of macronodes a mode passed through."""
        return self._mode_path.get(mode, [])

    def get_mode_lifetime(self, mode: int) -> tuple[int, int]:
        """Get the step range (start, end) for a mode."""
        path = self._mode_path.get(mode, [])
        if not path:
            return (-1, -1)
        start_w = min(w for _, w in path)
        end_w = max(w for _, w in path)
        return (start_w, end_w)


# =============================================================================
# Dependency Graph Builder (inspired by DependencyDAG from convert.py)
# =============================================================================


class GraphDependencyAnalyzer:
    """Builds and analyses dependency graphs for GraphRepr."""

    def __init__(self, graph: GraphRepr):
        self.graph = graph
        self.dep_graph: nx.DiGraph | None = None
        self.last_op_per_mode: dict[int, int] = {}
        self.measurement_to_mode: dict[int, int] = {}
        self.mode_to_measurement: dict[int, int] = {}
        self._build()

    def _build(self) -> None:
        """Build the dependency graph."""
        self.dep_graph = nx.DiGraph()

        for idx in range(self.graph.n_total_macronodes):
            self.dep_graph.add_node(idx)

        for w in range(self.graph.n_steps):
            for h in range(self.graph.n_local_macronodes):
                idx = self.graph.get_index(h, w)
                op = self.graph.get_operation(h, w)

                io_modes = self.graph.io_modes_dict()
                left, up, right, down = io_modes.get((h, w), (BLANK_MODE, BLANK_MODE, BLANK_MODE, BLANK_MODE))
                modes = {m for m in (left, up, right, down) if m != BLANK_MODE}

                for mode in modes:
                    if mode in self.last_op_per_mode:
                        self.dep_graph.add_edge(self.last_op_per_mode[mode], idx)
                    self.last_op_per_mode[mode] = idx

                if isinstance(op, GraphMeasurement):
                    measured_mode = left if left != BLANK_MODE else up
                    if measured_mode != BLANK_MODE:
                        self.measurement_to_mode[idx] = measured_mode
                        self.mode_to_measurement[measured_mode] = idx

                self._add_feedforward_dependencies(idx, op, io_modes, (h, w))

    def _add_feedforward_dependencies(self, idx: int, op, io_modes: dict, coord: tuple[int, int]) -> None:
        """Add dependencies from feedforward parameters."""
        h, w = coord
        left, up, _, _ = io_modes.get(coord, (BLANK_MODE, BLANK_MODE, BLANK_MODE, BLANK_MODE))

        for param in op.parameters:
            if isinstance(param, FeedForward):
                var = param.variable
                if hasattr(var, "mode"):
                    meas_idx = self.mode_to_measurement.get(var.mode)
                    if meas_idx is not None:
                        self.dep_graph.add_edge(meas_idx, idx)

        for disp in [op.displacement_k_minus_1, op.displacement_k_minus_n]:
            for d in disp:
                if isinstance(d, FeedForward):
                    var = d.variable
                    if hasattr(var, "mode"):
                        meas_idx = self.mode_to_measurement.get(var.mode)
                        if meas_idx is not None:
                            self.dep_graph.add_edge(meas_idx, idx)

    def get_dependency_graph(self) -> nx.DiGraph:
        return self.dep_graph

    def get_execution_layers(self) -> list[list[int]]:
        if self.dep_graph is None:
            return []

        layers = []
        remaining = set(self.dep_graph.nodes)
        in_degree = dict(self.dep_graph.in_degree())

        while remaining:
            current_layer = [node for node in remaining if in_degree.get(node, 0) == 0]
            if not current_layer:
                logger.warning("Cycle detected in dependency graph")
                break
            layers.append(current_layer)
            for node in current_layer:
                remaining.remove(node)
                for succ in self.dep_graph.successors(node):
                    in_degree[succ] = in_degree.get(succ, 1) - 1

        return layers

    def is_valid(self) -> bool:
        if self.dep_graph is None:
            return False
        return nx.is_directed_acyclic_graph(self.dep_graph)

    def get_critical_path_length(self) -> int:
        if self.dep_graph is None:
            return 0
        try:
            return nx.dag_longest_path_length(self.dep_graph)
        except nx.NetworkXUnfeasible:
            return 0


# =============================================================================
# Mode Lifecycle Tracker
# =============================================================================


@dataclass
class ModeLifecycle:
    creation_op: tuple[int, int] | None = None
    measurement_op: tuple[int, int] | None = None
    usage_ops: list[tuple[int, int]] = field(default_factory=list)
    is_initialized: bool = False
    is_measured: bool = False


class ModeLifecycleTracker:
    """Tracks the lifecycle of each mode: creation, usage, and measurement."""

    def __init__(self, graph: GraphRepr):
        self.graph = graph
        self.lifecycles: dict[int, ModeLifecycle] = {}
        self._analyze()

    def _analyze(self) -> None:
        io_modes = self.graph.io_modes_dict()

        for (h, w), (left, up, right, down) in io_modes.items():
            op = self.graph.get_operation(h, w)

            if isinstance(op, GraphInitialization):
                for mode in op.initialized_modes:
                    if mode != BLANK_MODE:
                        if mode not in self.lifecycles:
                            self.lifecycles[mode] = ModeLifecycle()
                        self.lifecycles[mode].creation_op = (h, w)
                        self.lifecycles[mode].is_initialized = True

            if isinstance(op, GraphMeasurement):
                mode = left if left != BLANK_MODE else up
                if mode != BLANK_MODE:
                    if mode not in self.lifecycles:
                        self.lifecycles[mode] = ModeLifecycle()
                    self.lifecycles[mode].measurement_op = (h, w)
                    self.lifecycles[mode].is_measured = True

            for mode in (left, up, right, down):
                if mode != BLANK_MODE:
                    if mode not in self.lifecycles:
                        self.lifecycles[mode] = ModeLifecycle()
                    self.lifecycles[mode].usage_ops.append((h, w))

    def get_lifecycle(self, mode: int) -> ModeLifecycle | None:
        return self.lifecycles.get(mode)

    def get_lifetime_steps(self, mode: int) -> tuple[int, int]:
        lifecycle = self.lifecycles.get(mode)
        if lifecycle is None:
            return (-1, -1)
        start_w = lifecycle.creation_op[1] if lifecycle.creation_op else -1
        end_w = lifecycle.measurement_op[1] if lifecycle.measurement_op else -1
        return (start_w, end_w)

    def validate(self) -> bool:
        valid = True
        for mode, lifecycle in self.lifecycles.items():
            if not lifecycle.is_initialized:
                logger.warning(f"Mode {mode} used but never created")
                valid = False
            if not lifecycle.is_measured:
                logger.warning(f"Mode {mode} created but never measured")
                valid = False
        return valid

    def get_uninitialized_modes(self) -> list[int]:
        return [mode for mode, lc in self.lifecycles.items() if not lc.is_initialized]

    def get_unmeasured_modes(self) -> list[int]:
        return [mode for mode, lc in self.lifecycles.items() if not lc.is_measured]


# =============================================================================
# Core conversion functions (Circuit → SF Program, Graph → SF Program)
# =============================================================================

# -----------------------------------------------------------------------------
# Circuit conversion (existing)
# -----------------------------------------------------------------------------


def __get_value(q, param: CircOpParam) -> float:
    if isinstance(param, float | int):
        return param
    if isinstance(param, FeedForward):
        func = param.func
        mode = param.variable.get_from_operation().opnd().get_ids()[0]
        return func(q[mode].par)
    msg = f"Invalid parameter: {param}"
    raise ValueError(msg)


def __add_std_squeezing(program: sf.Program, squeezing: std_ops.Squeezing) -> None:
    with program.context as q:
        inds = squeezing.opnd().get_ids()
        r = __get_value(q, squeezing.r)
        sf_ops.Sgate(r) | q[inds[0]]


def __add_std_bs(program: sf.Program, bs: std_ops.BeamSplitter) -> None:
    with program.context as q:
        inds = bs.opnd().get_ids()
        theta = __get_value(q, bs.theta)
        phi = __get_value(q, bs.phi)
        sf_ops.BSgate(theta, phi) | (q[inds[0]], q[inds[1]])


def __add_intrinsic_op(program: sf.Program, mqc3_op: intrinsic_ops.Intrinsic) -> None:
    inds = mqc3_op.opnd().get_ids()

    if isinstance(mqc3_op, intrinsic_ops.Measurement):
        with program.context as q:
            theta = __get_value(q, mqc3_op.theta)
            sf_ops.MeasureHomodyne(pi / 2.0 - theta) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.Displacement):
        with program.context as q:
            x = __get_value(q, mqc3_op.x)
            p = __get_value(q, mqc3_op.p)
            sf_ops.Xgate(x) | q[inds[0]]
            sf_ops.Zgate(p) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.PhaseRotation):
        with program.context as q:
            phi = __get_value(q, mqc3_op.phi)
            sf_ops.Rgate(phi) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.ShearXInvariant):
        with program.context as q:
            kappa = __get_value(q, mqc3_op.kappa)
            sf_ops.Pgate(kappa * 2.0) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.ShearPInvariant):
        with program.context as q:
            eta = __get_value(q, mqc3_op.eta)
            if eta == 0:
                acot_eta = pi / 2.0
            elif eta > 0:
                acot_eta = atan(1 / eta)
            elif eta < 0:
                acot_eta = atan(1 / eta) + pi
            half_pi = pi / 2.0
            half_theta = acot_eta / 2.0
            sign = sf.math.sign(tan(half_theta))
            sf_ops.Rgate(half_theta) | q[inds[0]]
            sf_ops.Sgate(r=-log(sign * tan(half_theta))) | q[inds[0]]
            sf_ops.Rgate(half_theta - sign * half_pi) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.Squeezing):
        with program.context as q:
            theta = __get_value(q, mqc3_op.theta)
            sign = sf.math.sign(tan(theta))
            sf_ops.Sgate(r=log(sign * tan(theta))) | q[inds[0]]
            sf_ops.Rgate(-sign * pi / 2.0) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.Squeezing45):
        with program.context as q:
            theta = __get_value(q, mqc3_op.theta)
            sign = sf.math.sign(tan(theta))
            sf_ops.Rgate(pi / 4.0) | q[inds[0]]
            sf_ops.Sgate(r=log(sign * tan(theta))) | q[inds[0]]
            if sign > 0:
                sf_ops.Rgate(-pi / 4.0) | q[inds[0]]
            else:
                sf_ops.Rgate(3 * pi / 4.0) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.Arbitrary):
        with program.context as q:
            alpha = __get_value(q, mqc3_op.alpha)
            beta = __get_value(q, mqc3_op.beta)
            lam = __get_value(q, mqc3_op.lam)
            sf_ops.Rgate(beta) | q[inds[0]]
            sf_ops.Sgate(lam) | q[inds[0]]
            sf_ops.Rgate(alpha) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.ControlledZ):
        with program.context as q:
            g = __get_value(q, mqc3_op.g)
            sf_ops.CZgate(g) | (q[inds[0]], q[inds[1]])
    elif isinstance(mqc3_op, intrinsic_ops.BeamSplitter):
        with program.context as q:
            sqrt_r = __get_value(q, mqc3_op.sqrt_r)
            theta_rel = __get_value(q, mqc3_op.theta_rel)
            h_theta = theta_rel / 2.0
            h_acos = acos(sqrt_r) / 2.0
            alpha = h_theta + h_acos
            beta = h_theta - h_acos
            sf_ops.BSgate() | (q[inds[0]], q[inds[1]])
            sf_ops.Rgate(alpha * 2) | q[inds[0]]
            sf_ops.Rgate(beta * 2) | q[inds[1]]
            sf_ops.BSgate() | (q[inds[1]], q[inds[0]])
    elif isinstance(mqc3_op, intrinsic_ops.TwoModeShear):
        with program.context as q:
            a = __get_value(q, mqc3_op.a)
            b = __get_value(q, mqc3_op.b)
            sf_ops.CZgate(b) | (q[inds[0]], q[inds[1]])
            sf_ops.Pgate(a * 2.0) | q[inds[0]]
            sf_ops.Pgate(a * 2.0) | q[inds[1]]
    else:
        msg = f"Unsupported operation type: {mqc3_op.name()}."
        raise TypeError(msg)


def _circuit_to_program(circuit: CircuitRepr) -> sf.Program:
    """Convert a CircuitRepr to a StrawberryFields program."""
    program = sf.Program(circuit.n_modes)

    # Set initial state
    for index in range(circuit.n_modes):
        state = circuit.get_initial_state(index)
        if not isinstance(state, BosonicState):
            raise TypeError(f"Initial state must be 'BosonicState' instance not {type(state)}.")
        if state.n_peaks != 1:
            raise ValueError("Only single-peak initial states are supported.")
        gaussian = state.get_gaussian_state(0)
        with program.context as q:
            sf_ops.Gaussian(gaussian.cov, gaussian.mean) | q[index]

    # Set operations
    for mqc3_op in circuit:
        if isinstance(mqc3_op, std_ops.Squeezing):
            __add_std_squeezing(program, mqc3_op)
        elif isinstance(mqc3_op, std_ops.BeamSplitter):
            __add_std_bs(program, mqc3_op)
        else:
            for intrinsic in mqc3_op.to_intrinsic_ops():
                if not isinstance(intrinsic, intrinsic_ops.Intrinsic):
                    raise TypeError("Intrinsic must be an instance of Intrinsic.")
                __add_intrinsic_op(program, intrinsic)
    return program


# -----------------------------------------------------------------------------
# Graph conversion with all operations
# -----------------------------------------------------------------------------


def _graph_to_program(graph: GraphRepr) -> sf.Program:
    """Convert a GraphRepr (MBQC pattern) into a StrawberryFields program."""
    io_modes = graph.io_modes_dict()

    # Determine number of physical modes
    all_modes = set()
    for left, up, right, down in io_modes.values():
        for m in (left, up, right, down):
            if m >= 0:
                all_modes.add(m)
    n_modes = max(all_modes) + 1 if all_modes else 1

    program = sf.Program(n_modes)

    # First pass: Initializations (create squeezed states)
    with program.context as q:
        for op in graph.operations.values():
            if isinstance(op, GraphInitialization):
                r = 1.0  # squeezing parameter
                for mode_idx in op.initialized_modes:
                    if mode_idx != -1:
                        sf_ops.Sgate(r) | q[mode_idx]
                        logger.debug(f"Initialized mode {mode_idx} with Sgate({r})")

    # Second pass: Two-mode gates (CZ, BeamSplitter, TwoModeShear)
    with program.context as q:
        for (h, w), op in graph.operations.items():
            left, up, _, _ = io_modes[h, w]

            if isinstance(op, GraphControlledZ):
                g = op.parameters[0] if op.parameters else 1.0
                if left != -1 and up != -1:
                    sf_ops.CZgate(g) | (q[left], q[up])
                    logger.debug(f"CZ gate between modes {left} and {up} with g={g}")

            elif isinstance(op, GraphBeamSplitter):
                sqrt_r = op.parameters[0] if len(op.parameters) > 0 else 0.7071
                theta_rel = op.parameters[1] if len(op.parameters) > 1 else 0.0
                if left != -1 and up != -1:
                    sf_ops.BSgate(theta=theta_rel) | (q[left], q[up])
                    logger.debug(f"Beam splitter between modes {left} and {up} with theta_rel={theta_rel}")

            elif isinstance(op, GraphTwoModeShear):
                a = op.parameters[0] if len(op.parameters) > 0 else 0.0
                b = op.parameters[1] if len(op.parameters) > 1 else 0.0
                if left != -1 and up != -1:
                    # Two-mode shear: CZ(b) followed by P(2a) on both modes
                    sf_ops.CZgate(b) | (q[left], q[up])
                    sf_ops.Pgate(2 * a) | q[left]
                    sf_ops.Pgate(2 * a) | q[up]
                    logger.debug(f"Two-mode shear on modes {left} and {up} with a={a}, b={b}")

    # Third pass: Single-mode gates (PhaseRotation, Squeezing, Squeezing45,
    # ShearXInvariant, ShearPInvariant, Arbitrary gates)
    with program.context as q:
        for (h, w), op in graph.operations.items():
            left, up, _, _ = io_modes[h, w]
            mode = left if left != -1 else up

            if mode == -1:
                continue

            if isinstance(op, GraphPhaseRotation):
                phi = op.parameters[0] if op.parameters else 0.0
                sf_ops.Rgate(phi) | q[mode]
                logger.debug(f"Phase rotation on mode {mode} with phi={phi}")

            elif isinstance(op, GraphSqueezing):
                theta = op.parameters[0] if op.parameters else 0.0
                # Convert theta to squeezing parameter r
                # The graph's Squeezing gate uses a non-standard definition
                # We approximate with standard Sgate
                r = 1.0
                sf_ops.Sgate(r) | q[mode]
                logger.debug(f"Squeezing on mode {mode} with theta={theta}")

            elif isinstance(op, GraphSqueezing45):
                theta = op.parameters[0] if op.parameters else 0.0
                # 45-degree squeezing: R(-π/4) S(θ) R(π/4)
                sf_ops.Rgate(-np.pi / 4) | q[mode]
                r = 1.0
                sf_ops.Sgate(r) | q[mode]
                sf_ops.Rgate(np.pi / 4) | q[mode]
                logger.debug(f"45-degree squeezing on mode {mode} with theta={theta}")

            elif isinstance(op, GraphShearXInvariant):
                kappa = op.parameters[0] if op.parameters else 0.0
                # P(kappa) gate: shears in x-quadrature
                sf_ops.Pgate(2 * kappa) | q[mode]
                logger.debug(f"Shear X-invariant on mode {mode} with kappa={kappa}")

            elif isinstance(op, GraphShearPInvariant):
                eta = op.parameters[0] if op.parameters else 0.0
                # Q(eta) gate: shears in p-quadrature
                # In StrawberryFields, this is implemented via rotation and Pgate
                # For now, use a combination of gates
                if eta != 0:
                    # R(π/2) P(2η) R(-π/2)
                    sf_ops.Rgate(np.pi / 2) | q[mode]
                    sf_ops.Pgate(2 * eta) | q[mode]
                    sf_ops.Rgate(-np.pi / 2) | q[mode]
                logger.debug(f"Shear P-invariant on mode {mode} with eta={eta}")

            elif isinstance(op, GraphArbitraryFirst):
                alpha = op.parameters[0] if len(op.parameters) > 0 else 0.0
                beta = op.parameters[1] if len(op.parameters) > 1 else 0.0
                lam = op.parameters[2] if len(op.parameters) > 2 else 0.0
                # Arbitrary Gaussian gate: R(beta) S(lam) R(alpha) for first part
                sf_ops.Rgate(beta) | q[mode]
                r = lam  # Use lam as squeezing parameter
                sf_ops.Sgate(r) | q[mode]
                sf_ops.Rgate(alpha) | q[mode]
                logger.debug(f"ArbitraryFirst on mode {mode} with alpha={alpha}, beta={beta}, lam={lam}")

            elif isinstance(op, GraphArbitrarySecond):
                alpha = op.parameters[0] if len(op.parameters) > 0 else 0.0
                beta = op.parameters[1] if len(op.parameters) > 1 else 0.0
                lam = op.parameters[2] if len(op.parameters) > 2 else 0.0
                # Arbitrary Gaussian gate: R(beta) S(lam) R(alpha) for second part
                # This is similar to first but may have different routing
                sf_ops.Rgate(beta) | q[mode]
                r = lam
                sf_ops.Sgate(r) | q[mode]
                sf_ops.Rgate(alpha) | q[mode]
                logger.debug(f"ArbitrarySecond on mode {mode} with alpha={alpha}, beta={beta}, lam={lam}")

            elif isinstance(op, GraphManual):
                theta_a = op.parameters[0] if len(op.parameters) > 0 else 0.0
                theta_b = op.parameters[1] if len(op.parameters) > 1 else 0.0
                theta_c = op.parameters[2] if len(op.parameters) > 2 else 0.0
                theta_d = op.parameters[3] if len(op.parameters) > 3 else 0.0
                # Manual operation: four specified homodyne angles
                # This is a complex operation that may require multiple measurements
                # For simulation, we approximate by a beam splitter-like interaction
                logger.debug(
                    f"Manual operation on modes {left} and {up} with angles {theta_a}, {theta_b}, {theta_c}, {theta_d}"
                )
                # Placeholder: use a beam splitter with average angles
                if left != -1 and up != -1:
                    avg_theta = (theta_a + theta_b + theta_c + theta_d) / 4
                    sf_ops.BSgate(theta=avg_theta) | (q[left], q[up])

    # Fourth pass: Wiring (mode routing) - handled by the mode flow tracking
    # Wiring operations don't need explicit gate application; they just route modes
    # The mode flow is already handled by the io_modes_dict

    # Fifth pass: Measurements (homodyne)
    with program.context as q:
        for (h, w), op in graph.operations.items():
            if isinstance(op, GraphMeasurement):
                left, up, _, _ = io_modes[h, w]
                mode = left if left != -1 else up
                if mode != -1:
                    theta = op.parameters[0] if op.parameters else 0.0
                    sf_ops.MeasureHomodyne(np.pi / 2 - theta) | q[mode]
                    logger.debug(f"Homodyne measurement on mode {mode} with theta={theta}, angle={np.pi / 2 - theta}")

    return program


# =============================================================================
# Shared execution logic
# =============================================================================


def _run_program(
    program: sf.Program,
    n_shots: int,
    state_save_policy: str,
) -> tuple[timedelta, list[NDArray[np.float64] | None], list[BosonicState]]:
    """
    Execute a StrawberryFields program and return results.

    This is the shared execution logic used by both local_run and local_run_graph.

    Returns:
        tuple of (execution_time, results_list, states_list)
    """
    sf.hbar = hbar

    started_at = datetime.now()
    results: list[NDArray[np.float64] | None] = []
    states: list[BosonicState] = []

    for i in range(n_shots):
        engine = sf.Engine(backend="bosonic")
        result = engine.run(program)
        if result.samples is not None and len(result.samples) > 0:
            results.append(result.samples[0])
        else:
            results.append(None)
        if state_save_policy == "all" or (i == 0 and state_save_policy == "first_only"):
            states.append(_convert_to_bosonic_state(result.state))

    finished_at = datetime.now(started_at.tzinfo)
    return (finished_at - started_at, results, states)


def _convert_to_circuit_result(sf_results: list[NDArray[np.float64] | None]) -> CircuitResult:
    """Convert StrawberryFields results to CircuitResult."""
    result = CircuitResult(shot_measured_values=[])
    for sf_shot_result in sf_results:
        shot_measured_value: list[CircuitOperationMeasuredValue] = []
        if sf_shot_result is not None:
            for index, value in enumerate(sf_shot_result):
                if value is not None:
                    shot_measured_value.append(CircuitOperationMeasuredValue(index, value))
        result.measured_vals.append(CircuitShotMeasuredValue(shot_measured_value))
    return result


def _convert_to_bosonic_state(sf_state: sf.backends.states.BaseBosonicState) -> BosonicState:
    """Convert StrawberryFields bosonic state to MQC3 BosonicState."""
    n = sf_state.num_modes
    return BosonicState(
        np.array(sf_state.weights(), dtype=np.complex128),
        [
            GaussianState(
                np.array(sf_state.means()[i], dtype=np.complex128).reshape(n, 2).T.flatten(),
                np
                .array(sf_state.covs()[i], dtype=np.float64)
                .reshape(n, 2, n, 2)
                .transpose(1, 0, 3, 2)
                .reshape(n * 2, n * 2),
            )
            for i in range(sf_state.num_weights)
        ],
    )


# =============================================================================
# Public API: local_run (for circuits)
# =============================================================================


@dataclass(frozen=True)
class LocalResult:
    execution_time: timedelta
    circuit_result: CircuitResult
    states: list[BosonicState]


def local_run(
    n_shots: int,
    state_save_policy: str,
    circuit: CircuitRepr,
    transpile: bool = True,
) -> LocalResult:
    """Run a quantum circuit locally using StrawberryFields."""
    if transpile:
        circuit = deepcopy(circuit)  # Simplified: actual transpilation would be here

    program = _circuit_to_program(circuit)
    exec_time, results, states = _run_program(program, n_shots, state_save_policy)
    circuit_result = _convert_to_circuit_result(results)

    return LocalResult(
        execution_time=exec_time,
        circuit_result=circuit_result,
        states=states,
    )


# =============================================================================
# Public API: local_run_graph (for graphs) with standard validation
# =============================================================================
@dataclass(frozen=True)
class LocalGraphResult:
    """Result of executing a GraphRepr locally."""

    execution_time: timedelta
    graph_result: GraphResult
    states: list[BosonicState]
    causal_structure: dict[int, list[int]] | None = None
    execution_layers: list[list[int]] | None = None


@dataclass(frozen=True)
class GraphSimulatorConfig:
    """Configuration for GraphSimulator validation."""

    enable_dependency_validation: bool = True
    enable_lifecycle_validation: bool = True
    fixed_squeezing_r: float = 1.0


class GraphSimulator:
    """
    Simulator for graph-based MBQC patterns on CV cluster states with validation.

    This class provides:
    - Graph validation (dependency and lifecycle)
    - Causal structure extraction
    - Execution layer scheduling
    - Mode path tracking
    - Simulation via StrawberryFields
    """

    def __init__(self, graph: GraphRepr, config: GraphSimulatorConfig | None = None):
        self.graph = graph
        self.config = config or GraphSimulatorConfig()

        # Validation components
        self.dependency_analyzer = GraphDependencyAnalyzer(graph)
        self.lifecycle_tracker = ModeLifecycleTracker(graph)
        self.flow_tracker = ModeFlowTracker(graph)

        # Mapping for result extraction
        self._mode_to_macronode: dict[int, int] = {}
        self._macronode_to_mode: dict[int, int] = {}
        self._measurement_order: list[int] = []
        self._build_mode_mapping()

        self._validated = False

    def _build_mode_mapping(self) -> None:
        """Build mapping between physical modes and graph macronodes."""
        io_modes = self.graph.io_modes_dict()
        for (h, w), (left, up, _, _) in io_modes.items():
            op = self.graph.get_operation(h, w)
            if isinstance(op, GraphMeasurement):
                mode = left if left != BLANK_MODE else up
                if mode != -1:
                    macronode_idx = self.graph.get_index(h, w)
                    self._mode_to_macronode[mode] = macronode_idx
                    self._macronode_to_mode[macronode_idx] = mode
                    if op.readout:
                        self._measurement_order.append(macronode_idx)

    def validate(self) -> bool:
        """Validate the graph (dependency and lifecycle)."""
        valid = True

        if self.config.enable_lifecycle_validation:
            if not self.lifecycle_tracker.validate():
                valid = False

        if self.config.enable_dependency_validation:
            if not self.dependency_analyzer.is_valid():
                logger.warning("Dependency graph contains cycles")
                valid = False

        self._validated = valid
        return valid

    def get_causal_structure(self) -> dict[int, list[int]]:
        """Extract causal dependencies between measurements."""
        dependencies = defaultdict(list)
        io_modes = self.graph.io_modes_dict()

        for (h, w), (left, up, _, _) in io_modes.items():
            op = self.graph.get_operation(h, w)
            if isinstance(op, GraphMeasurement):
                current_idx = self.graph.get_index(h, w)
                mode = left if left != BLANK_MODE else up
                if mode != BLANK_MODE and mode in self._mode_to_macronode:
                    for other_idx, other_mode in self._macronode_to_mode.items():
                        if other_idx != current_idx and other_mode == mode:
                            dependencies[current_idx].append(other_idx)

        return dict(dependencies)

    def get_execution_layers(self) -> list[list[int]]:
        """Compute parallel execution layers from the graph structure."""
        return self.dependency_analyzer.get_execution_layers()

    def get_critical_path_length(self) -> int:
        """Get the length of the critical path."""
        return self.dependency_analyzer.get_critical_path_length()

    def get_mode_path(self, mode: int) -> list[tuple[int, int]]:
        """Get the path of macronodes a mode passed through."""
        return self.flow_tracker.get_mode_path(mode)

    def get_mode_lifetime(self, mode: int) -> tuple[int, int]:
        """Get the step range (start, end) for a mode."""
        return self.lifecycle_tracker.get_lifetime_steps(mode)

    def validate_gaussian_transform(self, symplectic_matrix: np.ndarray, tol: float = 1e-8) -> bool:
        """
        Validate that a matrix is symplectic.

        A matrix S is symplectic if S^T Ω S = Ω, where Ω is the symplectic form.

        Args:
            symplectic_matrix: 2n x 2n matrix to validate
            tol: Tolerance for numerical comparison

        Returns:
            True if the matrix is symplectic within tolerance
        """
        n = len(symplectic_matrix) // 2
        omega = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])

        # Compute S^T Ω S
        result = symplectic_matrix.T @ omega @ symplectic_matrix

        # Check if equal to omega within tolerance
        return np.allclose(result, omega, atol=tol)

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information about the graph."""
        return {
            "n_modes": len(self.lifecycle_tracker.lifecycles),
            "n_measurements": len([op for op in self.graph.operations.values() if isinstance(op, GraphMeasurement)]),
            "n_initializations": len([
                op for op in self.graph.operations.values() if isinstance(op, GraphInitialization)
            ]),
            "valid": self._validated,
            "acyclic": self.dependency_analyzer.is_valid(),
            "lifecycle_valid": self.lifecycle_tracker.validate(),
            "critical_path_length": self.get_critical_path_length(),
            "execution_layers": len(self.get_execution_layers()),
        }

    def simulate(self, n_shots: int, state_save_policy: str = "none") -> LocalGraphResult:
        """Run the graph simulation with validation."""
        if self.config.enable_dependency_validation or self.config.enable_lifecycle_validation:
            if not self.validate():
                raise ValueError("Graph validation failed. Check diagnostics for details.")

        return local_run_graph_impl(self.graph, n_shots, state_save_policy)


def _get_causal_info(graph: GraphRepr) -> tuple[dict[int, list[int]], list[list[int]]]:
    """Helper to get causal structure and execution layers."""
    simulator = GraphSimulator(
        graph, GraphSimulatorConfig(enable_dependency_validation=False, enable_lifecycle_validation=False)
    )
    return simulator.get_causal_structure(), simulator.get_execution_layers()


def local_run_graph_impl(
    graph: GraphRepr,
    n_shots: int,
    state_save_policy: str = "none",
    return_causal_info: bool = False,
    debug: bool = False,
) -> LocalGraphResult:
    """
    Internal implementation of graph execution.
    """
    if debug:
        print("\n[DEBUG] local_run_graph called with:")
        print(f"  graph: {graph.n_local_macronodes}x{graph.n_steps}")
        print(f"  n_shots: {n_shots}")
        print(f"  state_save_policy: {state_save_policy}")

    # Build mode-to-macronode mapping for result extraction
    io_modes = graph.io_modes_dict()
    mode_to_macronode_for_result = {}
    for (h, w), (left, up, right, down) in io_modes.items():
        op = graph.get_operation(h, w)
        if isinstance(op, GraphMeasurement):
            measured_mode = left if left != BLANK_MODE else up
            if measured_mode != BLANK_MODE:
                mode_to_macronode_for_result[measured_mode] = graph.get_index(h, w)

    if debug:
        print(f"[DEBUG] Mode to macronode for result extraction: {mode_to_macronode_for_result}")

    # Convert graph to StrawberryFields program
    program = _graph_to_program(graph)

    if debug:
        print(f"\n[DEBUG] StrawberryFields program created with {len(program.circuit)} operations")
        print(f"[DEBUG] Program has {program.num_subsystems} modes")

    # Run the program using shared execution logic
    exec_time, sf_results, states = _run_program(program, n_shots, state_save_policy)

    if debug:
        print(f"\n[DEBUG] Execution completed in {exec_time.total_seconds() * 1000:.2f} ms")
        print(f"[DEBUG] Number of SF results: {len(sf_results)}")

    # Convert results to GraphResult
    results_per_shot = []

    for sf_shot_result in sf_results:
        measured_vals = []

        if sf_shot_result is not None:
            for mode_idx, value in enumerate(sf_shot_result):
                if mode_idx in mode_to_macronode_for_result:
                    macronode_idx = mode_to_macronode_for_result[mode_idx]
                    h, w = graph.get_coord(macronode_idx)
                    measured_vals.append(
                        GraphMacronodeMeasuredValue(index=macronode_idx, h=h, w=w, m_b=value, m_d=0.0)
                    )

        results_per_shot.append(GraphShotMeasuredValue(measured_vals, n_local_macronodes=graph.n_local_macronodes))

    graph_result = GraphResult(
        n_local_macronodes=graph.n_local_macronodes,
        shot_measured_values=results_per_shot,
    )

    # Compute causal info if requested
    causal_structure = None
    exec_layers = None
    if return_causal_info:
        causal_structure, exec_layers = _get_causal_info(graph)
        if debug:
            print(f"[DEBUG] Causal structure: {causal_structure}")
            print(f"[DEBUG] Execution layers: {exec_layers}")

    return LocalGraphResult(
        execution_time=exec_time,
        graph_result=graph_result,
        states=states,
        causal_structure=causal_structure,
        execution_layers=exec_layers,
    )


# Public API
def local_run_graph(
    graph: GraphRepr,
    n_shots: int,
    state_save_policy: str = "none",
    return_causal_info: bool = False,
    validate: bool = True,
    debug: bool = False,
) -> LocalGraphResult:
    """
    Run a GraphRepr locally using StrawberryFields with optional validation.

    Args:
        graph: Graph representation to execute
        n_shots: Number of measurement shots
        state_save_policy: Policy for saving final states
        return_causal_info: If True, attach causal structure to result
        validate: If True, perform graph validation before execution (default: True)
        debug: If True, print debug information during execution

    Returns:
        LocalGraphResult with execution time, graph result, and saved states
    """
    if validate:
        simulator = GraphSimulator(graph)
        if not simulator.validate():
            logger.warning("Graph validation failed. Proceeding anyway.")
            if debug:
                print("[DEBUG] Graph validation failed but continuing")
                diag = simulator.get_diagnostics()
                print(f"[DEBUG] Diagnostics: {diag}")
        elif debug:
            print("[DEBUG] Graph validation passed")

    return local_run_graph_impl(graph, n_shots, state_save_policy, return_causal_info, debug)

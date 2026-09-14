"""Domain-specific exception hierarchy for the `cvzx` compiler.

Every error the compiler itself raises (as opposed to a caller's mistake
like passing the wrong Python type, which stays a plain `TypeError`)
derives from `CvzxError`, so callers can catch any internal compiler
failure with a single ``except CvzxError:`` -- while still being able to
narrow to a specific failure mode (a bad symbolic parameter, a malformed
diagram, a rewrite rule hitting a corrupted graph, an unavailable graph
backend) when that's useful.

Hierarchy
---------
CvzxError
├── ParameterError
│   ├── ParameterConflictError
│   ├── UnboundMeasurementError
│   └── InvalidSymbolError
├── DiagramError
│   ├── ArityMismatchError
│   └── ExpansionError
├── RewriteError
│   └── RuleApplicationError
└── BackendError
    └── UnsupportedBackendError
"""


class CvzxError(Exception):
    """Base class for every error the `cvzx` compiler itself raises."""


class ParameterError(CvzxError):
    """A symbolic parameter or its feedforward provenance is invalid."""


class ParameterConflictError(ParameterError):
    """A symbol is bound to conflicting measurement IDs on different nodes.

    Raised by `CVZXGraph.validate_parameter_consistency()` when the same
    `sympy.Symbol` appears in more than one node's `param_measurement_map`
    with different bound measurement-ID sets.
    """


class UnboundMeasurementError(ParameterError):
    """A `param_measurement_map` references a measurement ID that doesn't exist.

    Raised by `CVZXGraph.validate_parameter_consistency()` when a
    feedforward gate's `param_measurement_map` names a measurement ID that
    has no corresponding registered `MeasurementGate` node in
    `GateRegister.measurement_nodes`.
    """


class InvalidSymbolError(ParameterError):
    """A `param_measurement_map` references a symbol that isn't a real parameter.

    Raised by `Parametrized._sync_feedforward_state()` when
    `param_measurement_map.keys()` isn't a subset of the object's own
    `get_parameters()`.
    """


class DiagramError(CvzxError):
    """A diagram's structure (ports, connectivity, expansion) is invalid."""


class ArityMismatchError(DiagramError):
    """Port counts or connectivity indices don't align.

    Raised by `Diagram.compose()`, `Diagram.tensor()`, or
    `ContractedDiagram` construction when the number of inputs/outputs or
    the connectivity/coupling indices on either side don't match up.
    """


class ExpansionError(DiagramError):
    """A `CompactDiagram.expand()` call failed to produce a valid decomposition.

    Raised when sub-spider symbol slicing (`Parametrized.slice_param_map`)
    or phase-polynomial expansion fails while expanding a gate into its
    constituent spiders/gates.
    """


class RewriteError(CvzxError):
    """A rewrite rule failed to apply cleanly to a graph."""


class RuleApplicationError(RewriteError):
    """A `RewriteRule.apply_single()` call hit an unexpected or corrupted graph state.

    Raised when a match found by `RewriteRule.match()` no longer applies
    cleanly by the time `apply_single()` runs against it -- e.g. a node
    the match referenced is missing, or an invariant the rule relies on
    (arity, port count, container shape) doesn't hold. Distinct from a
    plain "no match" (which isn't an error): this signals the graph
    itself is in a state the rule did not expect.
    """


class BackendError(CvzxError):
    """A requested `CVZXGraph` backend is unavailable or invalid."""


class UnsupportedBackendError(BackendError):
    """The requested backend isn't installed, or isn't a recognized `Backend` value.

    Raised by `cvzx.backend.get_backend_modules()`.
    """

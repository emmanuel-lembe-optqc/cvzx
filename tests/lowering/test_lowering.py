"""Tests for `cvzx.lowering.lowering` (the `Diagram -> DependencyDAG` backend registry).

Checks the registry mechanics (a fresh backend can be registered and
looked up, an unknown name raises with a helpful message, dispatch
actually calls the requested backend) and that the bundled `"mqc3"`
reference backend produces a real, well-formed mqc3 `DependencyDAG`
end to end from a cvzx `Diagram`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic
from mqc3.graph.embed.dep_dag import DependencyDAG

from cvzx.lowering.bridges.mqc3 import from_circuit_repr
from cvzx.lowering.lowering import (
    LoweringBackend,
    Mqc3ReferenceBackend,
    get_backend,
    graph_to_dependency_dag,
    list_backends,
    register_backend,
)

if TYPE_CHECKING:
    from cvzx.ir.base import Diagram


def _simple_diagram() -> Diagram:
    circuit = CircuitRepr("c")
    circuit.Q(0) | intrinsic.PhaseRotation(0.3)
    circuit.Q(0) | intrinsic.Measurement(0.1)
    return from_circuit_repr(circuit)


def test_mqc3_backend_is_registered_by_default():
    assert "mqc3" in list_backends()
    assert isinstance(get_backend("mqc3"), Mqc3ReferenceBackend)


def test_graph_to_dependency_dag_default_backend_constructs_real_dag():
    diagram = _simple_diagram()
    dag = graph_to_dependency_dag(diagram)
    assert isinstance(dag, DependencyDAG)
    assert dag.dag.number_of_nodes() > 0


def test_graph_to_dependency_dag_explicit_mqc3_backend_matches_default():
    diagram = _simple_diagram()
    default_dag = graph_to_dependency_dag(diagram)
    explicit_dag = graph_to_dependency_dag(diagram, backend="mqc3")
    assert default_dag.dag.number_of_nodes() == explicit_dag.dag.number_of_nodes()
    assert default_dag.dag.number_of_edges() == explicit_dag.dag.number_of_edges()


def test_unknown_backend_raises_key_error_listing_available_backends():
    with pytest.raises(KeyError, match="mqc3"):
        graph_to_dependency_dag(_simple_diagram(), backend="nonexistent_qpu")


def test_register_backend_adds_a_new_dispatchable_backend():
    """Check that a new QPU-specific backend can be added via the plugin design.

    Without touching the registry, `get_backend`, or
    `graph_to_dependency_dag` at all -- this is the whole point of the
    plugin design.
    """

    @register_backend("_test_dummy_backend")
    class _DummyBackend(LoweringBackend):
        def to_dependency_dag(self, diagram: Diagram) -> DependencyDAG:
            # Delegate to the reference backend; this test only checks
            # that dispatch reaches a freshly-registered backend, not
            # that it does anything different.
            return Mqc3ReferenceBackend().to_dependency_dag(diagram)

    try:
        assert "_test_dummy_backend" in list_backends()
        diagram = _simple_diagram()
        dag = graph_to_dependency_dag(diagram, backend="_test_dummy_backend")
        assert isinstance(dag, DependencyDAG)
        assert dag.dag.number_of_nodes() > 0
    finally:
        # Registration is a module-level side effect; clean up so this
        # test doesn't leak state into other tests in the same run.
        # ruff: ignore[import-outside-top-level, import-private-name]
        from cvzx.lowering.lowering import _REGISTRY

        _REGISTRY.pop("_test_dummy_backend", None)

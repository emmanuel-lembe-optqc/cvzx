"""Tests for `cvzx.lowering` (the `Diagram -> MachineryRepr` backend registry).

Checks the registry mechanics (a fresh backend can be registered and
looked up, an unknown name raises with a helpful message, dispatch
actually calls the requested backend) and that the bundled `"mqc3"`
reference backend produces a real, well-formed mqc3 `MachineryRepr`
end to end from a cvzx `Diagram`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic
from mqc3.machinery import MachineryRepr

from cvzx.circuit_to_diagram import from_circuit_repr
from cvzx.lowering import (
    LoweringBackend,
    Mqc3ReferenceBackend,
    get_backend,
    graph_to_machinery_repr,
    list_backends,
    register_backend,
)

if TYPE_CHECKING:
    from cvzx.base_gates import Diagram

_N_LOCAL_MACRONODES = 5


def _simple_diagram() -> Diagram:
    circuit = CircuitRepr("c")
    circuit.Q(0) | intrinsic.PhaseRotation(0.3)
    circuit.Q(0) | intrinsic.Measurement(0.1)
    return from_circuit_repr(circuit)


def test_mqc3_backend_is_registered_by_default():
    assert "mqc3" in list_backends()
    assert isinstance(get_backend("mqc3"), Mqc3ReferenceBackend)


def test_graph_to_machinery_repr_default_backend_constructs_real_machinery_repr():
    diagram = _simple_diagram()
    machinery = graph_to_machinery_repr(diagram, n_local_macronodes=_N_LOCAL_MACRONODES)
    assert isinstance(machinery, MachineryRepr)
    assert machinery.n_local_macronodes == _N_LOCAL_MACRONODES
    assert machinery.n_steps > 0


def test_graph_to_machinery_repr_explicit_mqc3_backend_matches_default():
    diagram = _simple_diagram()
    default = graph_to_machinery_repr(diagram, n_local_macronodes=_N_LOCAL_MACRONODES)
    explicit = graph_to_machinery_repr(diagram, n_local_macronodes=_N_LOCAL_MACRONODES, backend="mqc3")
    assert default.n_local_macronodes == explicit.n_local_macronodes
    assert default.n_steps == explicit.n_steps


def test_unknown_backend_raises_key_error_listing_available_backends():
    with pytest.raises(KeyError, match="mqc3"):
        graph_to_machinery_repr(_simple_diagram(), n_local_macronodes=_N_LOCAL_MACRONODES, backend="nonexistent_qpu")


def test_register_backend_adds_a_new_dispatchable_backend():
    """Check that a new QPU-specific backend can be added via the plugin design.

    Without touching the registry, `get_backend`, or
    `graph_to_machinery_repr` at all -- this is the whole point of the
    plugin design.
    """

    @register_backend("_test_dummy_backend")
    class _DummyBackend(LoweringBackend):
        def to_machinery_repr(self, diagram: Diagram, *, n_local_macronodes: int) -> MachineryRepr:
            # Delegate to the reference backend; this test only checks
            # that dispatch reaches a freshly-registered backend, not
            # that it does anything different.
            return Mqc3ReferenceBackend().to_machinery_repr(diagram, n_local_macronodes=n_local_macronodes)

    try:
        assert "_test_dummy_backend" in list_backends()
        diagram = _simple_diagram()
        machinery = graph_to_machinery_repr(
            diagram, n_local_macronodes=_N_LOCAL_MACRONODES, backend="_test_dummy_backend"
        )
        assert isinstance(machinery, MachineryRepr)
        assert machinery.n_local_macronodes == _N_LOCAL_MACRONODES
    finally:
        # Registration is a module-level side effect; clean up so this
        # test doesn't leak state into other tests in the same run.
        # ruff: ignore[import-outside-top-level, import-private-name]
        from cvzx.lowering import _REGISTRY

        _REGISTRY.pop("_test_dummy_backend", None)

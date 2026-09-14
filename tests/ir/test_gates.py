"""Unit tests for gates.py's CompactDiagram gate subclasses.

Focused on the `Parametrized` mixin behaviour threaded through
`DisplacementGate`, `ControlledSumGate`, and `ArbitraryGate`: parameter
substitution, `param_measurement_map` slicing/narrowing through `expand()`
and `substitute_parameters()`, and the two bugs fixed as part of wiring
`param_measurement_map` through every gate: `DisplacementGate.substitute_parameters`
crashing on a complex-valued result, and `ControlledSumGate.substitute_parameters`
dropping `control`/`target`.
"""

import pytest
from sympy import I, symbols

from cvzx.exceptions import InvalidSymbolError
from cvzx.ir.gates import ArbitraryGate, ControlledSumGate, DisplacementGate, PhaseRotationGate, SqueezingGate


class TestDisplacementGateSubstituteParameters:
    """Regression tests for DisplacementGate.substitute_parameters."""

    def test_complex_valued_result_does_not_crash(self):
        """Substituting into alpha = a + I*b with a complex result must not crash.

        This reproduces `DisplacementGate`'s own docstring example, which
        used to crash: the old `_rebuild` path called `float()` on a
        substituted value with a nonzero imaginary part.
        """
        a, b = symbols("a b", real=True)
        gate = DisplacementGate(a + I * b, parametric=True)

        result = gate.substitute_parameters({a: 0.5, b: 0.3})

        assert not result.parametric
        assert result.alpha == complex(0.5, 0.3)

    def test_invalid_param_measurement_map_symbol_raises(self):
        """A param_measurement_map keyed on a non-parameter symbol is illegal."""
        a, c = symbols("a c", real=True)

        with pytest.raises(InvalidSymbolError, match="not among its own parameters"):
            DisplacementGate(a, parametric=True, param_measurement_map={c: {1}})


class TestControlledSumGateSubstituteParameters:
    """Regression test for ControlledSumGate.substitute_parameters."""

    def test_preserves_control_and_target(self):
        """Substituting the gain must not silently reset control/target to defaults."""
        g = symbols("g", real=True)
        gate = ControlledSumGate(gain=g, control=1, target=2, parametric=True)

        result = gate.substitute_parameters({g: 2.0})

        assert result.control == 1
        assert result.target == 2
        assert result.gain == 2.0
        assert not result.parametric


class TestArbitraryGateParameterMap:
    """Multi-symbol `param_measurement_map` slicing/narrowing on ArbitraryGate.

    `ArbitraryGate` is the most illustrative multi-parameter case (3
    independent symbolic fields: alpha, beta, lam), used here to exercise
    the sub-map slicing that `expand()` performs for each spawned sub-gate.
    """

    def setup_method(self):
        """Build a 3-parameter ArbitraryGate with a partial param_measurement_map."""
        self.alpha, self.beta, self.lam = symbols("alpha beta lam", real=True)
        self.gate = ArbitraryGate(
            self.alpha,
            self.beta,
            self.lam,
            parametric=True,
            param_measurement_map={self.alpha: {1}, self.beta: {2}},
        )

    def test_get_parameters_includes_all_three(self):
        """get_parameters() must return alpha, beta, and lam."""
        assert self.gate.get_parameters() == {self.alpha, self.beta, self.lam}
        assert self.gate.is_parametric

    def test_subset_map_is_legal(self):
        """Lam has no entry in param_measurement_map -- a legal subset."""
        assert self.gate.param_measurement_map == {self.alpha: {1}, self.beta: {2}}
        assert self.lam not in self.gate.param_measurement_map

    def test_expand_slices_map_per_sub_gate(self):
        """expand() must hand each spawned rotation/squeeze gate only its own symbol's entry."""
        decomp = self.gate.expand()
        rot_beta, squeeze, rot_alpha = decomp.diagrams

        assert isinstance(rot_beta, PhaseRotationGate)
        assert isinstance(squeeze, SqueezingGate)
        assert isinstance(rot_alpha, PhaseRotationGate)

        assert rot_beta.param_measurement_map == {self.beta: {2}}
        assert rot_alpha.param_measurement_map == {self.alpha: {1}}
        # lam carries no provenance entry, so the squeeze gets an empty map.
        assert squeeze.param_measurement_map == {}

    def test_partial_substitution_narrows_map(self):
        """Substituting alpha away must drop only alpha's entry, keeping beta's."""
        result = self.gate.substitute_parameters({self.alpha: 0.1})

        assert result.param_measurement_map == {self.beta: {2}}
        assert self.alpha not in result.get_parameters()
        assert self.beta in result.get_parameters()

    def test_conjugate_preserves_map_unchanged(self):
        """conjugate() must carry param_measurement_map through unchanged (same symbols)."""
        conjugated = self.gate.conjugate()

        assert conjugated.param_measurement_map == self.gate.param_measurement_map
        assert conjugated.param_measurement_map is not self.gate.param_measurement_map
        # Parameters permute (alpha' = -beta, beta' = -alpha) but the map is
        # still keyed on the same original symbols, which remain the gate's
        # free parameters either way.
        assert conjugated.get_parameters() == {self.alpha, self.beta, self.lam}

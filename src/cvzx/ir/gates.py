"""CV-ZX representation of quantum gates from [1] Nagayoshi et al. (2024), Sec. II.C.

This module implements the standard CV quantum gates as compact diagrams
built from proper diagrams (spiders, Fourier, Swap). Each gate is a subclass
of CompactDiagram and provides a expand() method that returns the
equivalent CompositionDiagram or TensorDiagram of basic CV ZX elements.

References
----------
[1] Nagayoshi et al., CV ZX calculus, Sec. II.C, Table I
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from sympy import Expr, Symbol, cos, exp, im, pi, re, simplify, sin, sqrt, sympify, tan

from cvzx.exceptions import ExpansionError
from cvzx.ir.base import (
    CompositionDiagram,
    ContractedDiagram,
    Diagram,
    Fourier,
    FourierInv,
    Parametrized,
    PSpider,
    QSpider,
    TensorDiagram,
    ZxPoly,
)


@dataclass
class CompactDiagram(Diagram, Parametrized):
    """Compact representation of a diagram with an optional decomposition.

    This class allows representing a complex diagram (composed of tensors,
    compositions, contractions) in a compact form, with an optional
    decomposition that can be expanded when needed.

    This is useful for:
        - Representing gates in a compact form (e.g., R(θ) instead of three spiders)
        - Keeping the diagram structure simple during early compilation stages
        - Deferring expansion until necessary (e.g., for optimization or visualization)
        - Creating reusable composite blocks

    Attributes
    ----------
    label : str
        A label to display on the diagram (e.g., "R(θ)", "CZ", "BS(π/4)").
    _num_inputs : int
        Number of input wires.
    _num_outputs : int
        Number of output wires.
    decomposition : Diagram | None
        The expanded form of this diagram (optional).
    """

    label: str
    _num_inputs: int
    _num_outputs: int
    decomposition: Diagram | None

    def expand(self) -> Diagram:
        """Expand the compact diagram to its full decomposition.

        Returns
        -------
        Diagram
            The full decomposition. If no decomposition is set, returns self.
        """
        if self.decomposition is not None:
            return self.decomposition
        return self

    def can_expand(self) -> bool:
        """Check if the diagram has a decomposition set.

        Returns
        -------
        bool
            True if decomposition is not None.
        """
        return self.decomposition is not None

    def with_decomposition(self, decomp: Diagram) -> "CompactDiagram":
        """Return a new CompactDiagram with the given decomposition.

        Parameters
        ----------
        decomp : Diagram
            The decomposition to attach.

        Returns
        -------
        CompactDiagram
            A new CompactDiagram with the same label but with decomposition set.
        """
        return CompactDiagram(
            label=self.label,
            _num_inputs=self.num_inputs,
            _num_outputs=self.num_outputs,
            decomposition=decomp,
        )

    def conjugate(self) -> "Diagram":
        """Conjugate the compact diagram.

        For the label, this adds a '†' suffix if not already present.
        The decomposition is also conjugated if present.

        Overridden by most subclasses to narrow the return type to
        `CompactDiagram`; declared as `Diagram` here since at least one
        subclass (`MeasurementGate`) has a true adjoint that isn't
        compact-representable (see its own `conjugate()` docstring).

        Returns
        -------
        Diagram
            A new CompactDiagram with the conjugated label and decomposition.
        """
        # Conjugate the label
        new_label = self.label + "†" if not self.label.endswith("†") else self.label[:-1]

        # Conjugate the decomposition if present
        new_decomp = None
        if self.decomposition is not None:
            new_decomp = self.decomposition.conjugate()

        return CompactDiagram(
            label=new_label,
            _num_inputs=self.num_inputs,
            _num_outputs=self.num_outputs,
            decomposition=new_decomp,
        )

    def tensor(self, other: Diagram, expand_self: bool = False) -> Diagram:  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        """Tensor product of this compact diagram with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to tensor with.
        expand_self : bool, default=False
            If True, expand this diagram before tensoring.

        Returns
        -------
        Diagram
            Tensor product diagram.

        Notes
        -----
        - If both are CompactDiagram and self is not expanded, keep compact.
        - If expand_self is True, self is expanded first.
        - The other diagram is only expanded if it is not a CompactDiagram.
        """
        # Expand self if requested
        if expand_self:
            if self.can_expand():
                return self.expand().tensor(other)
            if isinstance(other, TensorDiagram):
                diagrams = list(other.diagrams)
                return TensorDiagram([self, *diagrams])
            return TensorDiagram([self, other])
        if isinstance(other, CompactDiagram):
            return TensorDiagram([self, other])
        if isinstance(other, TensorDiagram):
            diagrams = list(other.diagrams)
            return TensorDiagram([self, *diagrams])
        return TensorDiagram([self, other])

    def compose(self, other: Diagram, connectivity: dict | None = None, expand_self: bool = False) -> Diagram:  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        """Compose this compact diagram with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self (other ∘ self).
        connectivity : dict | None
            Dictionary mapping input indices of self to output indices of other.
            If None, uses identity mapping.
        expand_self : bool, default=False
            If True, expand this diagram before composing.

        Returns
        -------
        Diagram
            Composition diagram (other ∘ self).

        Notes
        -----
        - If both are CompactDiagram and self is not expanded, keep compact.
        - If expand_self is True, self is expanded first.
        - The other diagram is only expanded if it is not a CompactDiagram.
        """
        # Default connectivity if None
        if connectivity is None:
            connectivity = {i: i for i in range(self.num_inputs)}
        # Expand self if requested
        if expand_self:
            if self.can_expand():
                return self.expand().compose(other, connectivity=connectivity)
            if isinstance(other, CompositionDiagram):
                diagrams = list(other.diagrams)
                old_connectivity = other.connectivity
                last_idx = len(diagrams) - 1
                old_connectivity[last_idx] = connectivity
                return CompositionDiagram([*diagrams, self], old_connectivity)
            return CompositionDiagram([other, self], {0: connectivity})
        if isinstance(other, CompactDiagram):
            # Both are CompactDiagrams
            # Keep compact: compute new label and decomposition
            new_label = f"{other.label} ∘ {self.label}"
            # For decomposition, we need to handle connectivity
            # Expand both and compose with the given connectivity
            expanded_self = self.expand()
            expanded_other = other.expand()
            new_decomp = expanded_self.compose(expanded_other, connectivity)

            return CompactDiagram(
                label=new_label,
                _num_inputs=self.num_inputs,
                _num_outputs=other.num_outputs,
                decomposition=new_decomp,
            )
        if isinstance(other, CompositionDiagram):
            diagrams = list(other.diagrams)
            old_connectivity = other.connectivity
            # Add this diagram as the last element
            old_connectivity[len(diagrams) - 1] = connectivity
            return CompositionDiagram([*diagrams, self], old_connectivity)
        return CompositionDiagram([other, self], {0: connectivity})

    def is_proper(self) -> bool:
        """Compact diagrams are not proper..

        Returns
        -------
            bool
        """
        return False

    @property
    def num_inputs(self) -> int:
        """Number of input wires.

        Returns
        -------
        int
            Number of input ports.
        """
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of output wires.

        Returns
        -------
        int
            Number of output ports.
        """
        return self._num_outputs

    def __post_init__(self) -> None:
        """Initialize the compact diagram.

        Raises
        ------
        ValueError
            If label is empty or None.
            If label exceeds 5 characters.
        """
        super().__init__()
        if self.label is None or not self.label:
            msg = "CompactDiagram requires a non-empty label"
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Return string representation of the compact diagram.

        Returns
        -------
        str
            String showing the label and input/output counts if available.
        """
        decomp_str = " (with decomposition)" if self.decomposition is not None else ""
        return f"CompactDiagram({self.label}, {self.num_inputs}→{self.num_outputs}{decomp_str})"


@dataclass
class DisplacementGate(CompactDiagram):
    r"""Displacement gate D(a).

    Represents the displacement operator D(a) = exp(a â† - a* â).
    Decomposes into a q-spider and a p-spider as shown in [1] Eq. (57).

    Parameters
    ----------
    alpha : float | int | complex | Expr
        Displacement amplitude. Can be numeric or symbolic.
    parametric : bool
        If True, treat alpha as symbolic parameter. Default False.
    feedforward : bool
        If True, means the displacement is linked to a measurement.
    measurement_ids : set[int] | None
        Measurement diagrams linked to this displacement gate when used as
        feedforward.
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 1).
    _num_outputs : int
        Number of output wires (always 1).
    decomposition : Diagram | None
        Cached decomposition of the gate.

    References
    ----------
    [1] Nagayoshi et al., CV ZX calculus, Sec. II.C.1, Eq. (57)

    Examples
    --------
    >>> # Numeric displacement
    >>> D = DisplacementGate(0.5 + 0.3j)
    >>> D.expand()

    >>> # Symbolic displacement with real and imaginary parts
    >>> from sympy import symbols, I
    >>> a, b = symbols('a b', real=True)
    >>> D = DisplacementGate(a + I*b, parametric=True)
    >>> decomp = D.expand()
    >>> # Decomposition: Q(√2*b*x) ∘ P(√2*a*x)

    >>> # Substitute parameters
    >>> D_sub = D.substitute_parameters({a: 0.5, b: 0.3})

    >>> from base_gates import QSpider, ZxPoly
    >>> m = symbols('m', real=True)
    >>> meas = QSpider(1, 0, ZxPoly({1: m}), True)
    >>> D = Displacement(m + I*2, parametric=True, param_measurement_map={m: {meas.id}})

    """

    alpha: float | int | complex | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("alpha",)

    def expand(self) -> CompositionDiagram:
        """Decompose displacement gate into q-spider and p-spider.

        D(a) = Q(√2 Im(a) x) ∘ P(√2 Re(a) x)

        For symbolic alpha:
            D(a) = Q(√2 Im(a) x) ∘ P(√2 Re(a) x)
            where Re(a) and Im(a) are symbolic expressions.

        Returns
        -------
        CompositionDiagram
            Composition of p-spider then q-spider.

        Examples
        --------
        >>> # Numeric displacement
        >>> D = DisplacementGate(0.5 + 0.3j)
        >>> decomp = D.expand()

        >>> # Symbolic displacement with parameters
        >>> from sympy import symbols, I
        >>> a, b = symbols('a b', real=True)
        >>> D = DisplacementGate(a + I*b, parametric=True)
        >>> decomp = D.expand()
        """
        # Get real and imaginary parts
        real_part, imag_part = self._get_re_im()

        # Calculate coefficients with √2 factor
        sqrt2 = np.sqrt(2)
        sympy_sqrt2 = sqrt(2)
        q_coeff = sympy_sqrt2 * imag_part if isinstance(imag_part, Expr) else sqrt2 * imag_part
        p_coeff = sympy_sqrt2 * real_part if isinstance(imag_part, Expr) else sqrt2 * real_part

        # Create phase polynomials
        q_phase = self._create_phase_poly(q_coeff, degree=1)
        p_phase = self._create_phase_poly(p_coeff, degree=1)

        # Create spiders
        q_spider = _build_spider(self, QSpider, 1, 1, q_phase)
        p_spider = _build_spider(self, PSpider, 1, 1, p_phase)

        # D(a) = Q ∘ P (P applied first, then Q)
        return CompositionDiagram([p_spider, q_spider])

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "DisplacementGate":
        new_alpha, parametric, _ = _resolve_parameter(values["alpha"])
        return DisplacementGate(
            new_alpha,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of displacement gate is displacement with negated alpha.

        Returns
        -------
        DisplacementGate
            D(-a)
        """
        if self.parametric:
            return DisplacementGate(
                -self.alpha, parametric=True, param_measurement_map=dict(self.param_measurement_map)
            )
        return DisplacementGate(alpha=-self.alpha, param_measurement_map=dict(self.param_measurement_map))

    def __repr__(self) -> str:
        """Return string representation of the displacement gate.

        Returns
        -------
        str
            String showing the displacement amplitude alpha.
        """
        if self.parametric:
            return f"DisplacementGate(alpha={self.alpha}, parametric=True)"
        if isinstance(self.alpha, complex):
            return f"DisplacementGate(alpha={self.alpha.real:.2f}{self.alpha.imag:+.2f}j)"
        return f"DisplacementGate(alpha={self.alpha:.2f})"

    def _validate_alpha(self) -> None:
        """Validate that alpha is a valid complex number or sympy expression."""  # ruff: ignore[docstring-missing-exception]
        if self.parametric and not isinstance(self.alpha, Expr):
            msg = f"Expected sympy expression for parametric mode, got {type(self.alpha)}"
            raise ValueError(msg)
        if not isinstance(self.alpha, (complex, float, int)) and not self.parametric:
            msg_0 = f"Expected numeric type, got {type(self.alpha)}"
            raise TypeError(msg_0)

    def __post_init__(self) -> None:
        """Initialize the displacement gate and validate parameters."""
        # Convert numeric alpha to sympy if parametric
        if self.parametric and not isinstance(self.alpha, Expr):
            self.alpha = sympify(self.alpha)

        # Validate alpha
        self._validate_alpha()

        # Set label
        if self.parametric:
            self.label = f"D({self.alpha})"
        # Format numeric label
        elif isinstance(self.alpha, complex):
            self.label = f"D({self.alpha.real:.2f}{self.alpha.imag:+.2f}j)"
        else:
            self.label = f"D({self.alpha:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()

    def _get_re_im(self) -> tuple[float, float] | tuple[Expr, Expr]:
        """Extract real and imaginary parts of alpha.

        Returns
        -------
        tuple
            (real_part, imag_part) as either numeric or symbolic values.
        """
        if self.parametric:
            # Symbolic case
            alpha_expr = self.alpha if isinstance(self.alpha, Expr) else sympify(self.alpha)
            real_part = simplify(re(alpha_expr))
            imag_part = simplify(im(alpha_expr))
            return real_part, imag_part
        # Numeric case
        if isinstance(self.alpha, complex):
            return self.alpha.real, self.alpha.imag
        return float(self.alpha.real), float(self.alpha.imag)

    def _create_phase_poly(self, coeff: float | Expr, degree: int = 1) -> ZxPoly:
        """Create a phase polynomial for the spider.

        Parameters
        ----------
        coeff : float | Expr
            Coefficient for the phase polynomial.
        degree : int
            Degree of the monomial. Default 1 for linear phase.

        Returns
        -------
        ZxPoly
            Phase polynomial with the given coefficient.

        Raises
        ------
        ExpansionError
            If `coeff` can't be converted to a `float` in the non-parametric
            case (e.g. a stray non-real symbolic remainder).
        """
        if self.parametric:
            return ZxPoly({degree: coeff})
        # Convert to float for numeric case
        try:
            return ZxPoly({degree: float(coeff)})
        except (TypeError, ValueError) as exc:
            msg = f"DisplacementGate.expand(): could not convert phase coefficient {coeff!r} to float."
            raise ExpansionError(msg) from exc


@dataclass
class PhaseRotationGate(CompactDiagram):
    r"""Phase rotation gate R(θ).

    Represents the phase rotation operator R(θ) = exp(iθ â† â).
    Decomposes into three quadratic q-spiders as shown in [1] Eq. (58).

    Parameters
    ----------
    theta : float | int | Expr
        Rotation angle in radians. Can be numeric or symbolic.
    parametric : bool
        If True, treat theta as symbolic parameter. Default False.
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 1).
    _num_outputs : int
        Number of output wires (always 1).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider.

    Raises
    ------
    ValueError
        If theta is an odd multiple of π/2 (where tan is infinite).
        For these cases, use Fourier2 or composition of Fourier gates.

    References
    ----------
    [1] Nagayoshi et al., CV ZX calculus, Sec. II.C.2, Eq. (58)

    Examples
    --------
    >>> # Numeric phase rotation
    >>> R = PhaseRotationGate(np.pi/4)
    >>> R.expand()

    >>> # Symbolic phase rotation
    >>> from sympy import symbols
    >>> theta = symbols('theta', real=True)
    >>> R = PhaseRotationGate(theta, parametric=True)
    >>> decomp = R.expand()

    """

    theta: float | int | complex | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("theta",)

    def expand(self) -> CompositionDiagram:
        """Decompose phase rotation into three quadratic q-spiders.

        R(θ) = P(tan(θ/2)/2) ∘ Q(-sinθ/2) ∘ P(tan(θ/2)/2)

        Returns
        -------
        CompositionDiagram
            Composition of three q-spiders.

        Examples
        --------
        >>> R = PhaseRotationGate(np.pi/4)
        >>> decomp = R.expand()
        """
        tan_half = self._get_tan_half()
        sin_theta = self._get_sin()
        phase1 = ZxPoly({2: tan_half / 2})
        phase2 = ZxPoly({2: -sin_theta / 2})

        spider1 = _build_spider(self, PSpider, 1, 1, phase1)
        spider2 = _build_spider(self, QSpider, 1, 1, phase2)
        spider3 = _build_spider(self, PSpider, 1, 1, phase1)

        return CompositionDiagram([spider1, spider2, spider3])

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "PhaseRotationGate":
        new_theta, parametric, _ = _resolve_parameter(values["theta"])
        return PhaseRotationGate(
            new_theta,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of phase rotation is rotation by negative angle.

        Returns
        -------
        PhaseRotationGate
            R(-θ)
        """
        if self.parametric:
            return PhaseRotationGate(
                -self.theta, parametric=True, param_measurement_map=dict(self.param_measurement_map)
            )
        return PhaseRotationGate(theta=-self.theta, param_measurement_map=dict(self.param_measurement_map))

    def __repr__(self) -> str:
        """Return string representation of the phase rotation gate.

        Returns
        -------
        str
            String showing the rotation angle theta.
        """
        if self.parametric:
            return f"PhaseRotationGate(theta={self.theta}, parametric=True)"
        return f"PhaseRotationGate(theta={self.theta:.2f})"

    def _validate_theta(self) -> None:
        """Validate theta for invalid angles (odd multiples of π/2)."""  # ruff: ignore[docstring-missing-exception]
        if self.parametric:
            return  # Skip validation for symbolic

        if np.isclose(np.abs(self.theta) % np.pi, np.pi / 2):
            msg = (
                f"θ = {self.theta} is an odd multiple of π/2. "
                f"For π/2 rotation, use Fourier gate. For 3π/2, use FourierInv."
            )
            raise ValueError(msg)

    def __post_init__(self) -> None:
        """Initialize the phase rotation gate and validate parameters."""
        if self.parametric and not isinstance(self.theta, Expr):
            self.theta = sympify(self.theta)

        self._validate_theta()

        if self.parametric:
            self.label = f"R({self.theta})"
        # Check for special angles
        elif np.isclose(self.theta, np.pi / 4):
            self.label = "R(π/4)"
        elif np.isclose(self.theta, np.pi / 2):
            self.label = "R(π/2)"
        elif np.isclose(self.theta, np.pi):
            self.label = "R(π)"
        else:
            self.label = f"R({self.theta:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()

    def _get_tan_half(self) -> float | int | complex | Expr:
        """Get tan(θ/2) for the decomposition.

        Returns
        -------
            float | int | complex | Expr
        """
        if self.parametric:
            return tan(self.theta / 2)
        return np.tan(self.theta / 2)

    def _get_sin(self) -> float | int | complex | Expr:
        """Get sin(θ) for the decomposition.

        Returns
        -------
            float | int | complex | Expr
        """
        if self.parametric:
            return sin(self.theta)
        return np.sin(self.theta)


@dataclass
class SqueezingGate(CompactDiagram):
    r"""1-mode squeezing gate Sq(τ).

    Represents the squeezing operator with parameter τ (where τ = e^{-r} for
    standard squeezing). Decomposes into four quadratic spiders as shown in
    [1] Eq. (59).

    Parameters
    ----------
    tau : float | int | Expr
        Squeezing parameter. τ > 0 for squeezing, τ < 0 for anti-squeezing.
        τ = e^{-r} corresponds to squeezing in p̂ (x̂ anti-squeezed).
    parametric : bool
        If True, treat tau as symbolic parameter. Default False.
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 1).
    _num_outputs : int
        Number of output wires (always 1).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider.

    References
    ----------
    [1] Nagayoshi et al., CV ZX calculus, Sec. II.C.3, Eq. (59)

    Examples
    --------
    >>> # Numeric squeezing
    >>> S = SqueezingGate(0.5)
    >>> S.expand()

    >>> # Symbolic squeezing
    >>> from sympy import symbols
    >>> r = symbols('r', real=True)
    >>> S = SqueezingGate(exp(-r), parametric=True)
    >>> decomp = S.expand()

    """

    tau: float | int | complex | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("tau",)

    def expand(self) -> CompositionDiagram:
        """Decompose squeezing gate into four quadratic spiders.

        Sq(τ) = Q(a) ∘ P(b) ∘ Q(c) ∘ P(d)
        where a = τ(1-τ)/4, b = -1/τ, c = (τ-1)/4, d = 1

        Returns
        -------
        CompositionDiagram
            Composition of Q, P, Q, P spiders in sequence.

        Examples
        --------
        >>> S = SqueezingGate(0.5)
        >>> decomp = S.expand()
        """
        a, b, c, d = self._get_coefficients()
        phase1 = ZxPoly({2: a})
        phase2 = ZxPoly({2: b})
        phase3 = ZxPoly({2: c})
        phase4 = ZxPoly({2: d})

        spider1 = _build_spider(self, QSpider, 1, 1, phase1)
        spider2 = _build_spider(self, PSpider, 1, 1, phase2)
        spider3 = _build_spider(self, QSpider, 1, 1, phase3)
        spider4 = _build_spider(self, PSpider, 1, 1, phase4)

        return CompositionDiagram([spider1, spider2, spider3, spider4])

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "SqueezingGate":
        new_tau, parametric, _ = _resolve_parameter(values["tau"])
        return SqueezingGate(
            new_tau,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of squeezing gate is squeezing with reciprocal parameter.

        Returns
        -------
        SqueezingGate
            Sq(1/τ)
        """
        if self.parametric:
            return SqueezingGate(1 / self.tau, parametric=True, param_measurement_map=dict(self.param_measurement_map))
        return SqueezingGate(tau=1 / self.tau, param_measurement_map=dict(self.param_measurement_map))

    def __repr__(self) -> str:
        """Return string representation of the squeezing gate.

        Returns
        -------
        str
            String showing the squeezing parameter tau.
        """
        if self.parametric:
            return f"SqueezingGate(tau={self.tau}, parametric=True)"
        return f"SqueezingGate(tau={self.tau:.2f})"

    def _validate_tau(self) -> None:
        """Validate tau parameter."""  # ruff: ignore[docstring-missing-exception]
        if self.parametric:
            return
        if abs(self.tau) < 1e-10:  # ruff: ignore[magic-value-comparison]
            msg = f"tau={self.tau} is too close to zero for numerical stability"
            raise ValueError(msg)

    def __post_init__(self) -> None:
        """Initialize the squeezing gate."""
        if self.parametric and not isinstance(self.tau, Expr):
            self.tau = sympify(self.tau)

        self._validate_tau()

        if self.parametric:
            self.label = f"Sq({self.tau})"
        else:
            self.label = f"Sq({self.tau:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()

    def _get_coefficients(
        self,
    ) -> tuple[
        float | int | complex | Expr,
        float | int | complex | Expr,
        float | int | complex | Expr,
        float | int | complex | Expr,
    ]:
        """Get the four coefficients for the spider decomposition.

        Returns
        -------
        tuple
            (a, b, c, d) where:
            a = τ(1-τ)/4
            b = -1/τ
            c = (τ-1)/4
            d = 1
        """
        if self.parametric:
            a = self.tau * (1 - self.tau) / 4
            b = -1 / self.tau
            c = (self.tau - 1) / 4
            d = 1.0
        else:
            a = self.tau * (1 - self.tau) / 4
            b = -1 / self.tau if abs(self.tau) > 1e-10 else 0  # ruff: ignore[magic-value-comparison]
            c = (self.tau - 1) / 4
            d = 1.0
        return a, b, c, d


@dataclass
class ControlledSumGate(CompactDiagram):
    r"""Controlled-sum (CSUM) gate with gain g and specified control/target modes.

    Represents the operation exp(-i g q̂_c p̂_t) where c is the control mode
    and t is the target mode. For g=1, this is the unbiased CSUM gate
    (CV analogue of CNOT). Decomposes into q-spider and p-spider
    with a contraction as shown in [1] Eq. (61)-(62).

    Parameters
    ----------
    gain : float | int | Expr
        Gain parameter g. Default is 1 (unbiased CSUM).
    control : int
        Index of the control mode (1 or 2). Default is 2.
    target : int
        Index of the target mode (1 or 2). Default is 1.
    parametric : bool
        If True, treat gain as symbolic parameter. Default False.
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 2).
    _num_outputs : int
        Number of output wires (always 2).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider.

    Raises
    ------
    ValueError
        If control == target (must be different modes).

    References
    ----------
    [1] Nagayoshi et al., CV ZX calculus, Sec. II.C.4, Eq. (61)-(62)
    [4] Yoshikawa et al., QRL configuration, Sec. IV.C.3

    Examples
    --------
    >>> # Unbiased CSUM
    >>> C = ControlledSumGate(control=2, target=1)
    >>> C.expand()

    >>> # Biased CSUM with symbolic gain
    >>> from sympy import symbols
    >>> g = symbols('g', real=True)
    >>> C = ControlledSumGate(gain=g, control=2, target=1, parametric=True)
    >>> decomp = C.expand()

    """

    gain: float | int | complex | Expr = 1.0
    control: int = 2
    target: int = 1
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=2, init=False)
    _num_outputs: int = field(default=2, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("gain",)

    def expand(self) -> Diagram:
        """Decompose CSUM gate into spiders with contraction.

        For unbiased (g=1) gate [1] Eq. (61):
            CSUM2,1 = ContractedDiagram of q-spider and p-spider
            CSUM1,2 = ContractedDiagram of p-spider and q-spider

        For biased gate [1] Eq. (62):
            CSUM1,2(g) = (Sq(g) ⊗ Id) ∘ CSUM1,2(1) ∘ (Sq(g⁻¹) ⊗ Id)
            CSUM2,1(g) = (Sq(g) ⊗ Id) ∘ CSUM2,1(1) ∘ (Sq(g⁻¹) ⊗ Id)

        The squeezing is applied to the control mode.

        Returns
        -------
        Diagram
            ContractedDiagram for unbiased, CompositionDiagram for biased.

        Examples
        --------
        >>> C = ControlledSumGate(control=2, target=1)
        >>> decomp = C.expand()
        """
        # q-spider copies position from control mode
        # p-spider adds the copied position to the momentum of target mode
        control_spider: QSpider | PSpider
        target_spider: QSpider | PSpider
        if self.control == 2:  # ruff: ignore[magic-value-comparison]
            control_spider = QSpider(1, 2, ZxPoly({}))  # 1 input, 2 outputs (copy)
            target_spider = PSpider(2, 1, ZxPoly({}))  # 2 inputs, 1 output (add)
        else:
            control_spider = PSpider(1, 2, ZxPoly({}))  # 1 input, 2 outputs (copy)
            target_spider = QSpider(2, 1, ZxPoly({}))  # 2 inputs, 1 output (add)

        if self._is_unbiased():
            # Unbiased CSUM
            tensor = TensorDiagram([control_spider, target_spider])

            tensor.partial_trace([
                (0, [1], []),  # q-spider: outputs 0,1 → input 0 (feedback)
                (1, [], [0]),  # p-spider: output 0 → inputs 0,1 (forward)
            ])

            return tensor.diagrams[0]

        # Biased CSUM: squeeze then unbiased then unsqueeze
        if self.parametric:
            sqrt_gain = sqrt(self.gain)
            inv_sqrt = 1 / sqrt_gain
        else:
            sqrt_gain = np.sqrt(self.gain)
            inv_sqrt = 1 / sqrt_gain

        identity = QSpider(1, 1, ZxPoly({}))
        squeeze1 = _build_gate(self, SqueezingGate, "tau", sqrt_gain)
        upper_diagram = control_spider.compose(squeeze1)
        squeeze2 = _build_gate(self, SqueezingGate, "tau", inv_sqrt)
        squeeze2_id = squeeze2.tensor(identity)
        upper_diagram = squeeze2_id.compose(upper_diagram)

        tensor = TensorDiagram([upper_diagram, target_spider])
        tensor.partial_trace([
            (0, [1], []),
            (1, [], [0]),
        ])

        return tensor.diagrams[0]

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "ControlledSumGate":
        new_gain, parametric, _ = _resolve_parameter(values["gain"])
        return ControlledSumGate(
            gain=new_gain,
            control=self.control,
            target=self.target,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of CSUM is CSUM with negated gain (inverse).

        (e^{-i g q_c p_t})† = e^{+i g q_c p_t} = CSUM(-g)

        Returns
        -------
        ControlledSumGate
            CSUM with gain = -g (same control/target).
        """
        if self.parametric:
            return ControlledSumGate(
                gain=-self.gain,
                control=self.control,
                target=self.target,
                parametric=True,
                param_measurement_map=dict(self.param_measurement_map),
            )
        return ControlledSumGate(
            gain=-self.gain,
            control=self.control,
            target=self.target,
            param_measurement_map=dict(self.param_measurement_map),
        )

    def __repr__(self) -> str:
        """Return string representation of the controlled-sum gate.

        Returns
        -------
        str
            String showing the control/target modes and gain (if not 1).
        """
        if self.parametric:
            return (
                f"ControlledSumGate(gain={self.gain}, control={self.control}, target={self.target}, parametric=True)"
            )

        if self.gain == 1:
            return f"ControlledSumGate(CS{self.target},{self.control})"
        return f"ControlledSumGate(CS{self.target},{self.control}, gain={self.gain:.2f})"

    def __post_init__(self) -> None:
        """Initialize the controlled-sum gate and validate parameters."""  # ruff: ignore[docstring-missing-exception]
        if self.control == self.target:
            msg = f"Control mode {self.control} and target mode {self.target} must be different"
            raise ValueError(msg)

        if self.control not in {1, 2}:
            msg = f"Control mode {self.control} must be either 1 or 2"
            raise ValueError(msg)

        if self.target not in {1, 2}:
            msg = f"Target mode {self.target} must be either 1 or 2"
            raise ValueError(msg)

        if self.parametric and not isinstance(self.gain, Expr):
            self.gain = sympify(self.gain)

        # Set label
        if self.parametric:
            self.label = f"CS{self.target},{self.control}({self.gain})"
        elif self.gain == 1:
            self.label = f"CS{self.target},{self.control}"
        else:
            self.label = f"CS{self.target},{self.control}({self.gain:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()

    def _is_unbiased(self) -> bool:
        """Check if this is an unbiased CSUM gate (gain = 1).

        Returns
        -------
        bool
        """
        return self.gain == 1


@dataclass
class ControlledZGate(CompactDiagram):
    r"""Controlled-Z (CZ) gate with gain g.

    Represents the operation exp(-i g q̂₁ q̂₂).

    Parameters
    ----------
    gain : float | int | Expr
        Gain parameter g. Default is 1 (unbiased CZ).
    parametric : bool
        If True, treat gain as symbolic parameter. Default False.

    Attributes
    ----------
    gain : float | int | Expr
        Gain parameter (symbolic or numeric).
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 2).
    _num_outputs : int
        Number of output wires (always 2).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider.

    References
    ----------
    [1] Nagayoshi et al., CV ZX calculus, Sec. II.C.5, Eq. (63)-(64)

    Examples
    --------
    >>> # Unbiased CZ
    >>> CZ = ControlledZGate()
    >>> CZ.expand()

    >>> # Symbolic CZ
    >>> from sympy import symbols
    >>> g = symbols('g', real=True)
    >>> CZ = ControlledZGate(gain=g, parametric=True)
    >>> decomp = CZ.expand()

    """

    gain: float | int | complex | Expr = 1.0
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=2, init=False)
    _num_outputs: int = field(default=2, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("gain",)

    def expand(self) -> Diagram:
        """Decompose CZ gate using Fourier gates and CSUM.

        Unbiased CZ [1] Eq. (63):
            CZ = ContractedDiagram of q-spider and p-spider with
            a fourier diagram in between

        Biased CZ [1] Eq. (64):
            CZ(g) = (Sq(g⁻¹) ⊗ Id) ∘ CZ(1) ∘ (Sq(g) ⊗ Id)

        Returns
        -------
        Diagram
            Composition of Fourier, CSUM, Fourier.

        Examples
        --------
        >>> CZ = ControlledZGate()
        >>> decomp = CZ.expand()
        """
        # Fourier diagram
        fourier_inv = FourierInv()

        q_spider1 = QSpider(2, 1, ZxPoly({}))
        q_spider2 = QSpider(1, 2, ZxPoly({}))
        i_tensor_f = TensorDiagram([fourier_inv, QSpider(1, 1, ZxPoly({}))])

        if self._is_unbiased():
            # Identity on mode 1: a q-spider with zero phase
            tensor = TensorDiagram([q_spider1, i_tensor_f.compose(q_spider2)])
            # Contract: I1 → I2 (q output 0 to p input 0, q output 1 to p input 1)
            tensor.partial_trace([
                (0, [], [1]),
                (1, [0], []),
            ])

            return tensor.diagrams[0]

        # Biased CSUM: squeeze then unbiased then unsqueeze
        if self.parametric:
            sqrt_gain = sqrt(self.gain)
            inv_sqrt = 1 / sqrt_gain
        else:
            sqrt_gain = np.sqrt(self.gain)
            inv_sqrt = 1 / sqrt_gain

        squeeze1 = _build_gate(self, SqueezingGate, "tau", sqrt_gain)
        squeeze1_id = squeeze1.tensor(QSpider(1, 1, ZxPoly({})))
        upper_diagram = q_spider1.compose(squeeze1_id)
        squeeze2 = _build_gate(self, SqueezingGate, "tau", inv_sqrt)
        upper_diagram = squeeze2.compose(upper_diagram)

        tensor = TensorDiagram([upper_diagram, i_tensor_f.compose(q_spider2)])
        tensor.partial_trace([
            (0, [], [1]),
            (1, [0], []),
        ])

        return tensor.diagrams[0]

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "ControlledZGate":
        new_gain, parametric, _ = _resolve_parameter(values["gain"])
        return ControlledZGate(
            gain=new_gain,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of CZ is CZ with same gain (self-adjoint).

        Returns
        -------
        ControlledZGate
            CZ(g)
        """
        if self.parametric:
            return ControlledZGate(
                gain=self.gain, parametric=True, param_measurement_map=dict(self.param_measurement_map)
            )
        return ControlledZGate(gain=self.gain, param_measurement_map=dict(self.param_measurement_map))

    def __repr__(self) -> str:
        """Return string representation of the controlled-Z gate.

        Returns
        -------
        str
            String showing the gain if not 1.
        """
        if self.parametric:
            return f"ControlledZGate(gain={self.gain}, parametric=True)"
        if self.gain == 1:
            return "ControlledZGate()"
        return f"ControlledZGate(gain={self.gain:.2f})"

    def __post_init__(self) -> None:
        """Initialize the controlled-Z gate."""
        if self.parametric and not isinstance(self.gain, Expr):
            self.gain = sympify(self.gain)

        if self.parametric:
            self.label = f"CZ({self.gain})"
        elif self.gain == 1:
            self.label = "CZ"
        else:
            self.label = f"CZ({self.gain:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()

    def _is_unbiased(self) -> bool:
        """Check if this is an unbiased CSUM gate (gain = 1).

        Returns
        -------
        bool
        """
        return self.gain == 1


@dataclass
class BeamsplitterGate(CompactDiagram):
    r"""Beamsplitter gate BS(θ).

    Represents the operation exp(-iθ (q̂₁ p̂₂ - p̂₁ q̂₂)). Decomposes into
    squeezing gates and CSUM gates as shown in [1] Eq. (66).

    Parameters
    ----------
    theta : float | int | Expr
        Beamsplitter angle. θ = π/4 gives a 50:50 beamsplitter.
    parametric : bool
        If True, treat theta as symbolic parameter. Default False.

    Attributes
    ----------
    theta : float | int | Expr
        Beamsplitter angle (symbolic or numeric).
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 2).
    _num_outputs : int
        Number of output wires (always 2).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider.

    References
    ----------
    [1] Nagayoshi et al., CV ZX calculus, Sec. II.C.6, Eq. (66)-(67)

    Examples
    --------
    >>> # 50:50 beamsplitter
    >>> BS = BeamsplitterGate(np.pi/4)
    >>> BS.expand()

    >>> # Symbolic beamsplitter
    >>> from sympy import symbols
    >>> theta = symbols('theta', real=True)
    >>> BS = BeamsplitterGate(theta, parametric=True)
    >>> decomp = BS.expand()

    """

    theta: float | int | complex | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=2, init=False)
    _num_outputs: int = field(default=2, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("theta",)

    def expand(self) -> Diagram:
        """Decompose beamsplitter using squeezing and CSUM gates.

        Returns
        -------
        Diagram
            Composition following [1] Eq. (66) or simplified Eq. (67).

        Examples
        --------
        >>> BS = BeamsplitterGate(np.pi/4)
        >>> decomp = BS.expand()
        """
        if self._is_balanced():
            # Balanced beamsplitter [1] Eq. (67)
            if self.parametric:
                sqrt2 = sqrt(2)
                inv_sqrt2 = 1 / sqrt2
            else:
                sqrt2 = np.sqrt(2)
                inv_sqrt2 = 1 / sqrt2

            csum12 = ControlledSumGate(gain=1, control=1, target=2)
            csum21 = ControlledSumGate(gain=1, control=2, target=1)

            tensor = TensorDiagram([
                _build_gate(self, SqueezingGate, "tau", sqrt2),
                _build_gate(self, SqueezingGate, "tau", inv_sqrt2),
            ])

            return CompositionDiagram([csum12.expand(), tensor, csum21.expand()])

        # General beamsplitter - simplified representation
        # Full decomposition from [1] Appendix A.1.f
        if self.parametric:
            tan_theta = tan(self.theta)
            sin2_theta = sin(self.theta) ** 2
            cos_theta = cos(self.theta)
        else:
            tan_theta = np.tan(self.theta)
            sin2_theta = np.sin(self.theta) ** 2
            cos_theta = np.cos(self.theta)

        sq1 = _build_gate(self, SqueezingGate, "tau", 1 / tan_theta)
        sq2 = _build_gate(self, SqueezingGate, "tau", sin2_theta / cos_theta)
        sq3 = _build_gate(self, SqueezingGate, "tau", 1 / cos_theta)
        tensor3 = sq2.tensor(sq3)

        csum12 = ControlledSumGate(gain=1, control=1, target=2)
        csum21 = ControlledSumGate(gain=1, control=2, target=1)

        return CompositionDiagram([
            sq1.tensor(QSpider(1, 1, ZxPoly({}))),
            csum12.expand(),
            tensor3,
            csum21.expand(),
            sq1.tensor(QSpider(1, 1, ZxPoly({}))),
        ])

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "BeamsplitterGate":
        new_theta, parametric, _ = _resolve_parameter(values["theta"])
        return BeamsplitterGate(
            new_theta,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of beamsplitter is beamsplitter with negated angle.

        Returns
        -------
        BeamsplitterGate
            BS(-θ)
        """
        if self.parametric:
            return BeamsplitterGate(
                theta=-self.theta, parametric=True, param_measurement_map=dict(self.param_measurement_map)
            )
        return BeamsplitterGate(theta=-self.theta, param_measurement_map=dict(self.param_measurement_map))

    def __repr__(self) -> str:
        """Return string representation of the beamsplitter gate.

        Returns
        -------
        str
            String showing the beamsplitter angle theta.
        """
        if self.parametric:
            return f"BeamsplitterGate(theta={self.theta}, parametric=True)"
        if np.isclose(self.theta, np.pi / 4):
            return "BeamsplitterGate(π/4)"
        return f"BeamsplitterGate(theta={self.theta:.2f})"

    def __post_init__(self) -> None:
        """Initialize the beamsplitter gate."""
        if self.parametric and not isinstance(self.theta, Expr):
            self.theta = sympify(self.theta)

        if self.parametric:
            self.label = f"BS({self.theta})"
        elif np.isclose(self.theta, np.pi / 4):
            self.label = "BS(π/4)"
        elif np.isclose(self.theta, -np.pi / 4):
            self.label = "BS(-π/4)"
        else:
            self.label = f"BS({self.theta:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()

    def _is_balanced(self) -> bool:
        """Check if this is a balanced 50:50 beamsplitter.

        Returns
        -------
        bool
        """
        if self.parametric:
            return bool(self.theta == pi / 4)  # ruff: ignore[float-equality-comparison]
        return bool(np.isclose(self.theta, np.pi / 4))


@dataclass
class CubicPhaseGate(CompactDiagram):
    r"""Cubic phase gate CPG(y).

    Represents the non-Gaussian operation exp(iy x̂³). This is a native
    non-Gaussian gate represented by a single q-spider with cubic phase.

    Parameters
    ----------
    gamma : float | int | Expr
        Cubic phase strength parameter.
    parametric : bool
        If True, treat gamma as symbolic parameter. Default False.
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 1).
    _num_outputs : int
        Number of output wires (always 1).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider (non_gaussian).

    References
    ----------
    [1] Nagayoshi et al., CV ZX calculus, Sec. II.C.7, Eq. (68)

    Examples
    --------
    >>> # Numeric cubic phase gate
    >>> CPG = CubicPhaseGate(0.5)
    >>> CPG.expand()

    >>> # Symbolic cubic phase gate
    >>> from sympy import symbols
    >>> gamma = symbols('gamma', real=True)
    >>> CPG = CubicPhaseGate(gamma, parametric=True)
    >>> decomp = CPG.expand()

    """

    gamma: float | int | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="non_gaussian", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("gamma",)

    def expand(self) -> QSpider:
        """Decompose cubic phase gate into a single q-spider with cubic phase.

        Returns
        -------
        QSpider
            q-spider with phase function f(x) = y x³.

        Examples
        --------
        >>> CPG = CubicPhaseGate(0.5)
        >>> spider = CPG.expand()
        """
        phase = ZxPoly({3: self.gamma})
        return _build_spider(self, QSpider, 1, 1, phase)

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "CubicPhaseGate":
        new_gamma, parametric, _ = _resolve_parameter(values["gamma"])
        return CubicPhaseGate(
            new_gamma,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of cubic phase gate is cubic phase with negated gamma.

        Returns
        -------
        CubicPhaseGate
            CPG(-y)
        """
        if self.parametric:
            return CubicPhaseGate(
                gamma=-self.gamma, parametric=True, param_measurement_map=dict(self.param_measurement_map)
            )
        return CubicPhaseGate(gamma=-self.gamma, param_measurement_map=dict(self.param_measurement_map))

    def __repr__(self) -> str:
        """Return string representation of the cubic phase gate.

        Returns
        -------
        str
            String showing the cubic phase strength gamma.
        """
        if self.parametric:
            return f"CubicPhaseGate(gamma={self.gamma}, parametric=True)"
        return f"CubicPhaseGate(gamma={self.gamma:.2f})"

    def __post_init__(self) -> None:
        """Initialize the cubic phase gate."""
        if self.parametric and not isinstance(self.gamma, Expr):
            self.gamma = sympify(self.gamma)

        if self.parametric:
            self.label = f"CPG({self.gamma})"
        else:
            self.label = f"CPG({self.gamma:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()


@dataclass
class ShearXInvariantGate(CompactDiagram):
    r"""X-invariant shear gate P(kappa) (mqc3 `intrinsic.ShearXInvariant`).

    Represents the shear operator that leaves x-hat invariant and shifts
    p-hat:

        P^dagger(kappa) (x, p) P(kappa) = [[1, 0], [2*kappa, 1]] (x, p)

    A quadratic-phase q-spider *is* this shear directly -- the same
    building block already used inside `PhaseRotationGate`/`SqueezingGate`
    -- so no composition is needed.

    Parameters
    ----------
    kappa : float | int | Expr
        Shear strength parameter.
    parametric : bool
        If True, treat kappa as symbolic parameter. Default False.
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 1).
    _num_outputs : int
        Number of output wires (always 1).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider.

    References
    ----------
    mqc3 `intrinsic.ShearXInvariant`; a quadratic-phase q-spider implements
    this shear directly, the same pattern used by `PhaseRotationGate` and
    `SqueezingGate` for [1] Eq. (58)-(59).

    Examples
    --------
    >>> P = ShearXInvariantGate(0.3)
    >>> P.expand()

    """

    kappa: float | int | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("kappa",)

    def expand(self) -> QSpider:
        """Decompose the shear gate into a single q-spider.

        P(kappa) = Q(kappa)

        Returns
        -------
        QSpider
            q-spider with phase function f(x) = kappa * x^2.
        """
        phase = ZxPoly({2: self.kappa})
        return _build_spider(self, QSpider, 1, 1, phase)

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "ShearXInvariantGate":
        new_kappa, parametric, _ = _resolve_parameter(values["kappa"])
        return ShearXInvariantGate(
            new_kappa,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of the shear gate is the shear with negated kappa.

        Returns
        -------
        ShearXInvariantGate
            P(-kappa)
        """
        if self.parametric:
            return ShearXInvariantGate(
                -self.kappa, parametric=True, param_measurement_map=dict(self.param_measurement_map)
            )
        return ShearXInvariantGate(kappa=-self.kappa, param_measurement_map=dict(self.param_measurement_map))

    def __repr__(self) -> str:
        """Return string representation of the shear gate.

        Returns
        -------
        str
            String showing the shear strength kappa.
        """
        if self.parametric:
            return f"ShearXInvariantGate(kappa={self.kappa}, parametric=True)"
        return f"ShearXInvariantGate(kappa={self.kappa:.2f})"

    def __post_init__(self) -> None:
        """Initialize the shear gate."""
        if self.parametric and not isinstance(self.kappa, Expr):
            self.kappa = sympify(self.kappa)

        if self.parametric:
            self.label = f"P({self.kappa})"
        else:
            self.label = f"P({self.kappa:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()


@dataclass
class ShearPInvariantGate(CompactDiagram):
    r"""P-invariant shear gate Q(eta) (mqc3 `intrinsic.ShearPInvariant`).

    Represents the shear operator that leaves p-hat invariant and shifts
    x-hat:

        Q^dagger(eta) (x, p) Q(eta) = [[1, 2*eta], [0, 1]] (x, p)

    A quadratic-phase p-spider *is* this shear directly, symmetric to
    `ShearXInvariantGate`.

    Parameters
    ----------
    eta : float | int | Expr
        Shear strength parameter.
    parametric : bool
        If True, treat eta as symbolic parameter. Default False.
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 1).
    _num_outputs : int
        Number of output wires (always 1).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider.

    References
    ----------
    mqc3 `intrinsic.ShearPInvariant`; a quadratic-phase p-spider implements
    this shear directly, the same pattern used by `PhaseRotationGate` and
    `SqueezingGate` for [1] Eq. (58)-(59).

    Examples
    --------
    >>> Q = ShearPInvariantGate(0.3)
    >>> Q.expand()

    """

    eta: float | int | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("eta",)

    def expand(self) -> PSpider:
        """Decompose the shear gate into a single p-spider.

        Q(eta) = P(eta)

        Returns
        -------
        PSpider
            p-spider with phase function f(x) = eta * x^2.
        """
        phase = ZxPoly({2: self.eta})
        return _build_spider(self, PSpider, 1, 1, phase)

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "ShearPInvariantGate":
        new_eta, parametric, _ = _resolve_parameter(values["eta"])
        return ShearPInvariantGate(
            new_eta,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of the shear gate is the shear with negated eta.

        Returns
        -------
        ShearPInvariantGate
            Q(-eta)
        """
        if self.parametric:
            return ShearPInvariantGate(
                -self.eta, parametric=True, param_measurement_map=dict(self.param_measurement_map)
            )
        return ShearPInvariantGate(eta=-self.eta, param_measurement_map=dict(self.param_measurement_map))

    def __repr__(self) -> str:
        """Return string representation of the shear gate.

        Returns
        -------
        str
            String showing the shear strength eta.
        """
        if self.parametric:
            return f"ShearPInvariantGate(eta={self.eta}, parametric=True)"
        return f"ShearPInvariantGate(eta={self.eta:.2f})"

    def __post_init__(self) -> None:
        """Initialize the shear gate."""
        if self.parametric and not isinstance(self.eta, Expr):
            self.eta = sympify(self.eta)

        if self.parametric:
            self.label = f"Q({self.eta})"
        else:
            self.label = f"Q({self.eta:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()


@dataclass
class ArbitraryGate(CompactDiagram):
    r"""Arbitrary single-mode Gaussian gate R(alpha) S(lam) R(beta) (mqc3 `intrinsic.Arbitrary`).

    mqc3 defines this as the operator product `R(alpha) . S(lam) . R(beta)`
    (rightmost applied first, i.e. the mode meets `R(beta)` first, then
    `S(lam)`, then `R(alpha)` last), where:

        R^dagger(phi) (x, p) R(phi) = [[cos(phi), -sin(phi)], [sin(phi), cos(phi)]] (x, p)
        S^dagger(r) (x, p) S(r) = [[e^r, 0], [0, e^-r]] (x, p)

    Two conversions are needed to express this with cvzx's existing gates:

    - cvzx's `PhaseRotationGate(theta)` implements mqc3's `R(-theta)`, not
      `R(theta)` (verified directly from its own three-spider
      decomposition) -- so `R(phi)` here becomes `PhaseRotationGate(-phi)`.
    - cvzx's `SqueezingGate(tau)` implements `diag(tau, 1/tau)`, exactly
      `S(lam)` under `tau = e^lam` (no sign correction needed there).

    Composed in signal-flow order (first-applied-first, matching
    `CompositionDiagram`'s own convention):

        Arbitrary(alpha, beta, lam)
            = PhaseRotationGate(-beta) . SqueezingGate(e^lam) . PhaseRotationGate(-alpha)

    Verified numerically against the raw mqc3 matrix product for random
    (alpha, beta, lam).

    Parameters
    ----------
    alpha : float | int | Expr
        Final rotation angle.
    beta : float | int | Expr
        Initial rotation angle.
    lam : float | int | Expr
        Squeezing strength (natural-log convention, matching mqc3's `S(lam)`).
    parametric : bool
        If True, treat alpha, beta, lam as symbolic parameters. Default False.
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 1).
    _num_outputs : int
        Number of output wires (always 1).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider.

    References
    ----------
    mqc3 `intrinsic.Arbitrary`; decomposed here via cvzx's existing
    `PhaseRotationGate` and `SqueezingGate` ([1] Eq. (58)-(59)) with the
    sign correction noted above.
    """

    alpha: float | int | Expr
    beta: float | int | Expr
    lam: float | int | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("alpha", "beta", "lam")

    def expand(self) -> CompositionDiagram:
        """Decompose into rotation-squeeze-rotation.

        Returns
        -------
        CompositionDiagram
            Composition of two phase rotations around a squeezing gate.
        """
        tau = self._get_tau()

        rot_beta = _build_gate(self, PhaseRotationGate, "theta", -self.beta)
        squeeze = _build_gate(self, SqueezingGate, "tau", tau)
        rot_alpha = _build_gate(self, PhaseRotationGate, "theta", -self.alpha)

        return CompositionDiagram([rot_beta, squeeze, rot_alpha])

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "ArbitraryGate":
        new_alpha, alpha_parametric, _ = _resolve_parameter(values["alpha"])
        new_beta, beta_parametric, _ = _resolve_parameter(values["beta"])
        new_lam, lam_parametric, _ = _resolve_parameter(values["lam"])
        parametric = alpha_parametric or beta_parametric or lam_parametric
        return ArbitraryGate(
            new_alpha,
            new_beta,
            new_lam,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of Arbitrary(alpha, beta, lam) is Arbitrary(-beta, -alpha, -lam).

        `(R(alpha)S(lam)R(beta))^dagger = R(-beta)S(-lam)R(-alpha)`, which
        is again of the form `R(alpha')S(lam')R(beta')` with
        `alpha' = -beta`, `lam' = -lam`, `beta' = -alpha`.

        Returns
        -------
        ArbitraryGate
            Arbitrary(-beta, -alpha, -lam)
        """
        if self.parametric:
            return ArbitraryGate(
                -self.beta,
                -self.alpha,
                -self.lam,
                parametric=True,
                param_measurement_map=dict(self.param_measurement_map),
            )
        return ArbitraryGate(
            alpha=-self.beta,
            beta=-self.alpha,
            lam=-self.lam,
            param_measurement_map=dict(self.param_measurement_map),
        )

    def __repr__(self) -> str:
        """Return string representation of the arbitrary gate.

        Returns
        -------
        str
            String showing alpha, beta, lam.
        """
        if self.parametric:
            return f"ArbitraryGate(alpha={self.alpha}, beta={self.beta}, lam={self.lam}, parametric=True)"
        return f"ArbitraryGate(alpha={self.alpha:.2f}, beta={self.beta:.2f}, lam={self.lam:.2f})"

    def __post_init__(self) -> None:
        """Initialize the arbitrary gate."""
        if self.parametric:
            if not isinstance(self.alpha, Expr):
                self.alpha = sympify(self.alpha)
            if not isinstance(self.beta, Expr):
                self.beta = sympify(self.beta)
            if not isinstance(self.lam, Expr):
                self.lam = sympify(self.lam)

        if self.parametric:
            self.label = f"Arb({self.alpha},{self.beta},{self.lam})"
        else:
            self.label = f"Arb({self.alpha:.2f},{self.beta:.2f},{self.lam:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()

    def _get_tau(self) -> float | int | complex | Expr:
        """Get tau = e^lam for the squeezing gate.

        Returns
        -------
        float | int | complex | Expr
        """
        if self.parametric:
            return exp(self.lam)
        return float(np.exp(self.lam))


@dataclass
class Squeezing45Gate(CompactDiagram):
    r"""45-degree squeezing gate (mqc3 `intrinsic.Squeezing45`).

    Defined by mqc3 as `R(-pi/4) S_V(cot theta) R(pi/4)`, where `S_V(c)`
    has matrix `diag(1/c, c)`. This is exactly `ArbitraryGate` with
    `alpha = -pi/4`, `beta = pi/4`, and `lam` chosen so that
    `S(lam) = S_V(cot theta)`: since `e^lam = 1/cot(theta) = tan(theta)`,
    cvzx's `SqueezingGate(tau)` (which implements `diag(tau, 1/tau)`) can
    be used directly with `tau = tan(theta)`, no logarithm required.

    Composed in signal-flow order:

        Squeezing45(theta)
            = PhaseRotationGate(-pi/4) . SqueezingGate(tan theta) . PhaseRotationGate(pi/4)

    Parameters
    ----------
    theta : float | int | Expr
        Squeezing angle parameter (mqc3 convention).
    parametric : bool
        If True, treat theta as symbolic parameter. Default False.
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 1).
    _num_outputs : int
        Number of output wires (always 1).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider.

    References
    ----------
    mqc3 `intrinsic.Squeezing45`; a special case of `ArbitraryGate`.
    """

    theta: float | int | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("theta",)

    def expand(self) -> CompositionDiagram:
        """Decompose into rotation-squeeze-rotation at fixed +/- pi/4 angles.

        Returns
        -------
        CompositionDiagram
            Composition of two fixed phase rotations around a squeezing gate.
        """
        tau = self._get_tau()

        rot_in = PhaseRotationGate(-np.pi / 4)
        squeeze = _build_gate(self, SqueezingGate, "tau", tau)
        rot_out = PhaseRotationGate(np.pi / 4)

        return CompositionDiagram([rot_in, squeeze, rot_out])

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "Squeezing45Gate":
        new_theta, parametric, _ = _resolve_parameter(values["theta"])
        return Squeezing45Gate(
            new_theta,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of Squeezing45(theta) is Squeezing45(pi/2 - theta).

        Since `S_V(c)^dagger = S_V(1/c)` and `1/cot(theta) = cot(pi/2-theta)`.

        Returns
        -------
        Squeezing45Gate
            Squeezing45(pi/2 - theta)
        """
        if self.parametric:
            return Squeezing45Gate(
                pi / 2 - self.theta, parametric=True, param_measurement_map=dict(self.param_measurement_map)
            )
        return Squeezing45Gate(theta=np.pi / 2 - self.theta, param_measurement_map=dict(self.param_measurement_map))

    def __repr__(self) -> str:
        """Return string representation of the 45-degree squeezing gate.

        Returns
        -------
        str
            String showing theta.
        """
        if self.parametric:
            return f"Squeezing45Gate(theta={self.theta}, parametric=True)"
        return f"Squeezing45Gate(theta={self.theta:.2f})"

    def __post_init__(self) -> None:
        """Initialize the 45-degree squeezing gate."""
        if self.parametric and not isinstance(self.theta, Expr):
            self.theta = sympify(self.theta)

        if self.parametric:
            self.label = f"Sq45({self.theta})"
        else:
            self.label = f"Sq45({self.theta:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()

    def _get_tau(self) -> float | int | complex | Expr:
        """Get tau = tan(theta) for the squeezing gate.

        Returns
        -------
        float | int | complex | Expr
        """
        if self.parametric:
            return tan(self.theta)
        return float(np.tan(self.theta))


@dataclass
class TwoModeShearGate(CompactDiagram):
    r"""Two-mode shear gate P2(a, b) (mqc3 `intrinsic.TwoModeShear`).

        P2^dagger(a, b) (x1, x2, p1, p2) P2(a, b)
            = [[1, 0, 0, 0], [0, 1, 0, 0], [2a, b, 1, 0], [b, 2a, 0, 1]] (x1, x2, p1, p2)

    The diagonal `2a` terms are exactly `ShearXInvariantGate(a)` applied to
    each mode; the cross `b` term is exactly cvzx's `ControlledZGate`
    generator `exp(-ig q1 q2)` evaluated at `g = -b` (cvzx's
    `ControlledZGate(g)` produces `p1 -= g*x2, p2 -= g*x1`, the negative of
    mqc3's `ControlledZ(g)` convention -- verified directly from its own
    decomposition's generator). Both pieces are shears of the same abelian
    family (they only ever add a linear function of the x's to the p's,
    leaving the x's invariant), so they commute and no new primitive is
    required.

    Parameters
    ----------
    a : float | int | Expr
        Diagonal (single-mode) shear strength, applied to both modes.
    b : float | int | Expr
        Cross-mode coupling strength.
    parametric : bool
        If True, treat a and b as symbolic parameters. Default False.
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 2).
    _num_outputs : int
        Number of output wires (always 2).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider.

    References
    ----------
    mqc3 `intrinsic.TwoModeShear`; decomposed here as a diagonal shear on
    each mode (see `ShearXInvariantGate`) composed with `ControlledZGate`.
    """

    a: float | int | Expr
    b: float | int | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=2, init=False)
    _num_outputs: int = field(default=2, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("a", "b")

    def expand(self) -> CompositionDiagram:
        """Decompose into per-mode shears and a controlled-Z coupling.

        P2(a, b) = CZ(g=-b) after (P(a) tensor P(a))

        Returns
        -------
        CompositionDiagram
            Tensor of two shears, followed by a controlled-Z gate.
        """
        shear1 = _build_gate(self, ShearXInvariantGate, "kappa", self.a)
        shear2 = _build_gate(self, ShearXInvariantGate, "kappa", self.a)
        cz = _build_gate(self, ControlledZGate, "gain", -self.b)

        return CompositionDiagram([TensorDiagram([shear1, shear2]), cz])

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "TwoModeShearGate":
        new_a, a_parametric, _ = _resolve_parameter(values["a"])
        new_b, b_parametric, _ = _resolve_parameter(values["b"])
        parametric = a_parametric or b_parametric
        return TwoModeShearGate(
            new_a,
            new_b,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> CompactDiagram:
        """Conjugate of the two-mode shear gate negates both parameters.

        Both `a` and `b` parametrize the same real quadratic-form generator
        (`H = a*x1^2 + a*x2^2 + b*x1*x2`), so
        `exp(-iH)^dagger = exp(+iH) = exp(-i(-H))` is the same gate with
        both coefficients negated -- computed directly rather than
        delegating to `ControlledZGate.conjugate()`, since the two-mode
        shear's adjoint is defined independently of whatever convention
        that method follows.

        Returns
        -------
        TwoModeShearGate
            P2(-a, -b)
        """
        if self.parametric:
            return TwoModeShearGate(
                -self.a, -self.b, parametric=True, param_measurement_map=dict(self.param_measurement_map)
            )
        return TwoModeShearGate(a=-self.a, b=-self.b, param_measurement_map=dict(self.param_measurement_map))

    def __repr__(self) -> str:
        """Return string representation of the two-mode shear gate.

        Returns
        -------
        str
            String showing a and b.
        """
        if self.parametric:
            return f"TwoModeShearGate(a={self.a}, b={self.b}, parametric=True)"
        return f"TwoModeShearGate(a={self.a:.2f}, b={self.b:.2f})"

    def __post_init__(self) -> None:
        """Initialize the two-mode shear gate."""
        if self.parametric:
            if not isinstance(self.a, Expr):
                self.a = sympify(self.a)
            if not isinstance(self.b, Expr):
                self.b = sympify(self.b)

        if self.parametric:
            self.label = f"P2({self.a},{self.b})"
        else:
            self.label = f"P2({self.a:.2f},{self.b:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()


@dataclass
class MeasurementGate(CompactDiagram):
    r"""Homodyne measurement effect (mqc3 `intrinsic.Measurement`).

    Measures the quadrature `x-hat sin(theta) + p-hat cos(theta)`. This is
    not a unitary gate but an *effect* -- a diagram leaf with an output
    arity of zero, the dual of how e.g. `PSpider(0, 1, ...)` is already
    used elsewhere in this codebase (and in
    `tests/test_visualize_gates.py`'s `feedforward_test`) to represent
    ancilla/measurement leaves.

    Measuring `x-hat` directly is a plain q-spider effect, `QSpider(1, 0, 0)`.
    To measure the rotated quadrature, rotate the mode into alignment
    first: solving `R(phi)`'s Heisenberg matrix (top row) for
    `cos(phi)*x - sin(phi)*p = sin(theta)*x + cos(theta)*p` gives
    `phi = theta - pi/2`. Composed in signal-flow order (rotate first, then
    measure), and converting to cvzx's rotation-gate convention
    (`PhaseRotationGate(psi)` = mqc3's `R(-psi)`):

        Measurement(theta) = PhaseRotationGate(pi/2 - theta) . QSpider(1, 0, 0)

    Checked against the theta=0 case (measuring p-hat directly):
    `phi = -pi/2` correctly rotates x-hat onto p-hat.

    Note: `conjugate()` returns a bare `Diagram` (a 0-in-1-out state), not
    another `MeasurementGate` -- an effect's adjoint is a state, a
    different shape, so it cannot be wrapped back into this same compact
    gate class the way every other gate in this module does.

    Parameters
    ----------
    theta : float | int | Expr
        Measured quadrature angle.
    parametric : bool
        If True, treat theta as symbolic parameter. Default False.
    label : str
        String label for the gate.
    _num_inputs : int
        Number of input wires (always 1).
    _num_outputs : int
        Number of output wires (always 0 -- this is an effect, not a gate).
    decomposition : Diagram | None
        Cached decomposition of the gate.
    spider_type : str
        Type identifier for the spider.

    References
    ----------
    mqc3 `intrinsic.Measurement`.
    """

    theta: float | int | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=0, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="effect", init=False)

    _param_fields: ClassVar[tuple[str, ...]] = ("theta",)

    def expand(self) -> CompositionDiagram:
        """Decompose into a rotation followed by an x-basis effect.

        Returns
        -------
        CompositionDiagram
            Rotation into alignment, followed by a q-spider effect.
        """
        rot = self._rotation_diagram(self._get_phi())
        effect = QSpider(1, 0, ZxPoly({}))

        return CompositionDiagram([rot, effect])

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "MeasurementGate":
        new_theta, parametric, _ = _resolve_parameter(values["theta"])
        return MeasurementGate(
            new_theta,
            parametric=parametric,
            param_measurement_map=new_map,
        )

    def conjugate(self) -> Diagram:
        """Conjugate of the measurement effect.

        Built directly rather than delegating to `self.expand().conjugate()`:
        `QSpider.conjugate()` does not currently flip the input/output
        arity for non-square (state/effect) leaves, so composing the
        reversed, per-piece-conjugated pieces (as `CompositionDiagram`'s
        own `conjugate()` does generically) fails arity validation. The
        true adjoint of a 1-in-0-out effect is a 0-in-1-out state, so it
        is constructed here explicitly: a zero-phase state followed by the
        inverse rotation.

        Returns
        -------
        Diagram
            A 0-in-1-out state diagram, the adjoint of this effect.
        """
        phi = self._get_phi()
        state = QSpider(0, 1, ZxPoly({}))
        rot_inv = self._rotation_diagram(-phi)
        return CompositionDiagram([state, rot_inv])

    def __repr__(self) -> str:
        """Return string representation of the measurement effect.

        Returns
        -------
        str
            String showing theta.
        """
        if self.parametric:
            return f"MeasurementGate(theta={self.theta}, parametric=True)"
        return f"MeasurementGate(theta={self.theta:.2f})"

    def __post_init__(self) -> None:
        """Initialize the measurement effect."""
        if self.parametric and not isinstance(self.theta, Expr):
            self.theta = sympify(self.theta)

        if self.parametric:
            self.label = f"M({self.theta})"
        else:
            self.label = f"M({self.theta:.2f})"

        self._sync_feedforward_state()
        super().__post_init__()

    def _get_phi(self) -> float | Expr:
        """Get the cvzx rotation angle pi/2 - theta.

        Returns
        -------
        float | Expr
        """
        if self.parametric:
            return pi / 2 - self.theta
        return np.pi / 2 - self.theta

    def _rotation_diagram(self, phi: float | Expr) -> Diagram:
        """Build the rotation diagram for angle `phi`.

        Routes around `PhaseRotationGate`'s restriction on odd
        multiples of pi/2: it refuses angles that are an odd multiple of
        pi/2 -- exactly the angles this gate needs for a plain x or p
        measurement (`theta` an integer multiple of pi). Folding `phi`
        into `(-pi, pi]` and substituting the equivalent `Fourier`/
        `FourierInv` proper diagram at the two problem angles sidesteps
        this without touching `PhaseRotationGate` itself. Derived (and
        numerically checked) from `Fourier`'s own docstring definition:
        `Fourier` implements the standard rotation matrix R(pi/2)
        (x' = -p, p' = x), which is cvzx's `PhaseRotationGate(-pi/2)`
        under the sign convention noted on `ArbitraryGate`; `FourierInv`
        is its adjoint, R(-pi/2), i.e. `PhaseRotationGate(pi/2)`.
        Only applies in numeric mode -- parametric angles skip
        `PhaseRotationGate`'s own validation the same way already.

        Returns
        -------
        Diagram
            `PhaseRotationGate`, `Fourier`, or `FourierInv`, as appropriate.
        """
        if self.parametric:
            return _build_gate(self, PhaseRotationGate, "theta", phi)

        folded = ((float(phi) + np.pi) % (2 * np.pi)) - np.pi
        if np.isclose(folded, np.pi / 2):
            return FourierInv()
        if np.isclose(folded, -np.pi / 2):
            return Fourier()
        return PhaseRotationGate(folded)


# =============================================================================
# Helper functions
# =============================================================================


def is_numeric(expr: Expr) -> bool:
    """Check if a sympy expression is purely numeric.

    Parameters
    ----------
    expr : Expr
        Symbolic parameter.

    Returns
    -------
    bool
        True if `expr` has no free symbols.
    """
    return len(expr.free_symbols) == 0


def _resolve_parameter(value: Any) -> tuple[Any, bool, set[Symbol]]:  # ruff: ignore[any-type]
    """Resolve a (possibly substituted or derived) field value.

    Used both by `_rebuild` hooks (after substitution) and by `expand()`
    methods (when deriving a sub-gate's own parameter from one of this
    gate's fields): a plain, non-`Expr` value is never parametric; a
    symbolic `Expr` with no remaining free symbols is folded down to a
    concrete Python numeric (`complex` only if genuinely non-real,
    otherwise `float`) -- avoiding the crash from blindly `float()`-ing a
    complex result; a genuinely symbolic `Expr` is passed through as-is.

    Parameters
    ----------
    value : Any
        A field value, or an expression derived from one.

    Returns
    -------
    tuple[Any, bool, set[Symbol]]
        `(resolved_value, is_parametric, free_symbols)`.
    """
    if not isinstance(value, Expr):
        return value, False, set()
    if is_numeric(value):
        as_complex = complex(value)
        resolved = as_complex.real if as_complex.imag == 0 else as_complex
        return resolved, False, set()
    return value, True, set(value.free_symbols)


def _build_spider[T: (QSpider, PSpider)](
    gate: "CompactDiagram", cls: type[T], num_inputs: int, num_outputs: int, phase: ZxPoly
) -> T:
    """Construct a `QSpider`/`PSpider` sub-object carrying `gate`'s provenance.

    Shared by every `expand()` that spawns a spider from one of its own
    `ZxPoly` phases: threads that phase's own `parametric`/`is_parametric`
    state (not a blanket copy of `gate.parametric`) and the relevant slice
    of `gate.param_measurement_map`, keeping each `expand()` body to one
    local per spider instead of three.

    Parameters
    ----------
    gate : CompactDiagram
        The gate whose `param_measurement_map` provenance to slice from.
    cls : type[T]
        Which spider class to build (`QSpider` or `PSpider`).
    num_inputs : int
    num_outputs : int
    phase : ZxPoly
        The phase polynomial for the new spider.

    Returns
    -------
    T
    """
    return cls(
        num_inputs,
        num_outputs,
        phase,
        phase.is_parametric(),
        param_measurement_map=gate.slice_param_map(phase.get_parameters()),
    )


def _build_gate[T: "CompactDiagram"](
    gate: "CompactDiagram",
    cls: type[T],
    field_name: str,
    value: Any,  # ruff: ignore[any-type]
) -> T:
    """Construct a `CompactDiagram` sub-gate carrying `gate`'s provenance.

    Shared by every `expand()` that derives a value (e.g. a squeeze ratio,
    a rotation angle) from one of `gate`'s own fields and hands it to a
    single-parameter sub-gate class: resolves the derived value (see
    `_resolve_parameter`), computes that sub-value's own `parametric`
    state, and slices `gate.param_measurement_map` down to the symbols it
    actually carries -- instead of blindly copying `gate.parametric`.

    Parameters
    ----------
    gate : CompactDiagram
        The gate whose `param_measurement_map` provenance to slice from.
    cls : type[T]
        The sub-gate class to construct (e.g. `SqueezingGate`).
    field_name : str
        The constructor keyword for the derived value (e.g. `"tau"`).
    value : Any
        The derived (possibly symbolic) value for that field.

    Returns
    -------
    T
        A new instance of `cls`.
    """
    resolved, parametric, symbols_in_value = _resolve_parameter(value)
    kwargs: dict[str, Any] = {
        field_name: resolved,
        "parametric": parametric,
        "param_measurement_map": gate.slice_param_map(symbols_in_value),
    }
    return cls(**kwargs)


def create_compact_diagram(label: str, num_inputs: int, num_outputs: int, decomp: Diagram) -> CompactDiagram:
    """Create a compact diagram for a given diagram.

    Parameters
    ----------
    label : str
        Label for the block.
    num_inputs : int
        Number of input wires.
    num_outputs : int
        Number of output wires.
    decomp : Diagram
        Diagram to compact.

    Returns
    -------
    CompactDiagram
        A compact diagram representing the block.
    """
    return CompactDiagram(
        label=label,
        _num_inputs=num_inputs,
        _num_outputs=num_outputs,
        decomposition=decomp,
    )


def expand_all(diagram: Diagram) -> Diagram:
    """Recursively expand all two mode diagrams instances in a diagram.

    Parameters
    ----------
    diagram : Diagram
        The diagram to expand.

    Returns
    -------
    Diagram
        The expanded diagram with all CompactDiagram expanded.
    """
    if isinstance(diagram, (BeamsplitterGate, ControlledSumGate, ControlledZGate, TwoModeShearGate)):
        return diagram.expand()

    if type(diagram) is CompactDiagram:
        # This block allows to expand sub-circuit diagrams
        return diagram.expand()

    if isinstance(diagram, CompositionDiagram):
        expanded = [expand_all(d) for d in diagram.diagrams]
        return CompositionDiagram(expanded)

    if isinstance(diagram, TensorDiagram):
        expanded = [expand_all(d) for d in diagram.diagrams]
        return TensorDiagram(expanded)

    if isinstance(diagram, ContractedDiagram):
        first_expanded = expand_all(diagram.first)
        second_expanded = expand_all(diagram.second)
        return ContractedDiagram(
            first=first_expanded,
            second=second_expanded,
            I1=diagram.I1,
            I2=diagram.I2,
            J1=diagram.J1,
            J2=diagram.J2,
        )

    return diagram

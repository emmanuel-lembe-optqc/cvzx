"""CV-ZX representation of quantum gates from [1] Nagayoshi et al. (2024), Sec. II.C.

This module implements the standard CV quantum gates as compact diagrams
built from proper diagrams (spiders, Fourier, Swap). Each gate is a subclass
of CompactDiagram and provides a expand() method that returns the
equivalent CompositionDiagram or TensorDiagram of basic CV ZX elements.

References:
-----------
[1] Nagayoshi et al., CV ZX calculus, Sec. II.C, Table I
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sympy import Expr, Symbol, cos, im, pi, re, simplify, sin, sqrt, symbols, sympify, tan

from cvzx.base_gates import (
    CompositionDiagram,
    ContractedDiagram,
    Diagram,
    FourierInv,
    PSpider,
    QSpider,
    TensorDiagram,
    ZxPoly,
)


@dataclass
class CompactDiagram(Diagram):
    """Compact representation of a diagram with an optional decomposition.

    This class allows representing a complex diagram (composed of tensors,
    compositions, contractions) in a compact form, with an optional
    decomposition that can be expanded when needed.

    This is useful for:
        - Representing gates in a compact form (e.g., R(θ) instead of three spiders)
        - Keeping the diagram structure simple during early compilation stages
        - Deferring expansion until necessary (e.g., for optimization or visualization)
        - Creating reusable composite blocks

    Attributes:
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
    decomposition: Diagram

    def expand(self) -> Diagram:
        """Expand the compact diagram to its full decomposition.

        Returns:
        -------
        Diagram
            The full decomposition. If no decomposition is set, returns self.
        """
        if self.decomposition is not None:
            return self.decomposition
        return self

    def can_expand(self) -> bool:
        """Check if the diagram has a decomposition set.

        Returns:
        -------
        bool
            True if decomposition is not None.
        """
        return self.decomposition is not None

    def with_decomposition(self, decomp: Diagram) -> "CompactDiagram":
        """Return a new CompactDiagram with the given decomposition.

        Parameters:
        ----------
        decomp : Diagram
            The decomposition to attach.

        Returns:
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

    def conjugate(self) -> "CompactDiagram":
        """Conjugate the compact diagram.

        For the label, this adds a '†' suffix if not already present.
        The decomposition is also conjugated if present.

        Returns:
        -------
        CompactDiagram
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

    def tensor(self, other: Diagram, expand_self: bool = False) -> Diagram:  # noqa: FBT001, FBT002
        """Tensor product of this compact diagram with another diagram.

        Parameters:
        ----------
        other : Diagram
            Diagram to tensor with.
        expand_self : bool, default=False
            If True, expand this diagram before tensoring.

        Returns:
        -------
        Diagram
            Tensor product diagram.

        Notes:
        -----
        - If both are CompactDiagram and self is not expanded, keep compact.
        - If expand_self is True, self is expanded first.
        - The other diagram is only expanded if it is not a CompactDiagram.
        """
        # Expand self if requested
        if expand_self:
            if self.can_expand:
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

    def compose(self, other: Diagram, connectivity: dict | None = None, expand_self: bool = False) -> Diagram:  # noqa: FBT001, FBT002
        """Compose this compact diagram with another diagram.

        Parameters:
        ----------
        other : Diagram
            Diagram to apply after self (other ∘ self).
        connectivity : dict | None
            Dictionary mapping input indices of self to output indices of other.
            If None, uses identity mapping.
        expand_self : bool, default=False
            If True, expand this diagram before composing.

        Returns:
        -------
        Diagram
            Composition diagram (other ∘ self).

        Notes:
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
            if self.can_expand:
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

        Returns:
        -------
            bool
        """
        return False

    @property
    def num_inputs(self) -> int:
        """Number of input wires.

        Returns:
        -------
        int
            Number of input ports.
        """
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of output wires.

        Returns:
        -------
        int
            Number of output ports.
        """
        return self._num_outputs

    def __post_init__(self) -> None:
        """Initialize the compact diagram.

        Raises:
        ------
        ValueError:
            If label is empty or None.
            If label exceeds 5 characters.
        """
        super().__init__()
        if self.label is None or not self.label:
            msg = "CompactDiagram requires a non-empty label"
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Return string representation of the compact diagram.

        Returns:
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
    Decomposes into a q-spider and a p-spider as shown in [3] Eq. (57).

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

    Examples:
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
    >>> meas = QSpider(1, 0, ZxPoly({1: m}))
    >>> D = Displacement(m + I*2, parametric=True, feedforward=True, measurement_ids = {meas.id})

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.1, Eq. (57)
    """

    alpha: float | int | complex | Expr
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)

    def expand(self) -> CompositionDiagram:
        """Decompose displacement gate into q-spider and p-spider.

        D(a) = Q(√2 Im(a) x) ∘ P(√2 Re(a) x)

        For symbolic alpha:
            D(a) = Q(√2 Im(a) x) ∘ P(√2 Re(a) x)
            where Re(a) and Im(a) are symbolic expressions.

        Returns:
        -------
        CompositionDiagram
            Composition of p-spider then q-spider.

        Examples:
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
        q_coeff = sqrt2 * imag_part
        p_coeff = sqrt2 * real_part

        # Create phase polynomials
        q_phase = self._create_phase_poly(q_coeff, degree=1)
        p_phase = self._create_phase_poly(p_coeff, degree=1)

        # Create spiders
        q_spider = QSpider(1, 1, q_phase)
        p_spider = PSpider(1, 1, p_phase)

        # D(a) = Q ∘ P (P applied first, then Q)
        return CompositionDiagram([p_spider, q_spider])

    def substitute_parameters(self, mapping: dict[Symbol | str, Any]) -> CompactDiagram:
        """Substitute symbolic parameters in the displacement gate.

        Parameters
        ----------
        mapping : dict
            Dictionary mapping symbols to values.
            Keys can be Symbol objects or strings.

        Returns:
        -------
        DisplacementGate
            A new DisplacementGate with substituted parameters.

        Examples:
        --------
        >>> from sympy import symbols
        >>> a, b = symbols('a b')
        >>> D = DisplacementGate(a + I*b, parametric=True)
        >>> D_sub = D.substitute_parameters({a: 0.5, b: 0.3})
        """
        if not self.parametric:
            return self

        # Convert string keys to symbols if needed
        new_mapping = {}
        for key, value in mapping.items():
            n_key = key
            if isinstance(key, str):
                n_key = symbols(key)
            new_mapping[n_key] = value

        # Substitute in alpha
        new_alpha = self.alpha.subs(new_mapping)
        new_alpha = simplify(new_alpha)

        if is_numeric(new_alpha):
            return DisplacementGate(float(new_alpha), parametric=False)
        return DisplacementGate(new_alpha, parametric=True)

    def evaluate(self, **kwargs: Any) -> CompactDiagram:  # noqa: ANN401
        """Evaluate the displacement gate with given parameter values.

        Parameters
        ----------
        **kwargs : dict
            Parameter values as keyword arguments.
            Keys are parameter names (strings), values are numeric.

        Returns:
        -------
        DisplacementGate
            A new DisplacementGate with evaluated parameters.

        Examples:
        --------
        >>> from sympy import symbols
        >>> a, b = symbols('a b')
        >>> D = DisplacementGate(a + I*b, parametric=True)
        >>> D_eval = D.evaluate(a=0.5, b=0.3)
        """
        if not self.parametric:
            return self

        mapping = {}
        for name, value in kwargs.items():
            mapping[symbols(name)] = value

        return self.substitute_parameters(mapping)

    def get_parameters(self) -> set[Symbol]:
        """Get all symbolic parameters in the displacement gate.

        Returns:
        -------
        set[Symbol]
            Set of symbols used in the displacement gate.

        Examples:
        --------
        >>> from sympy import symbols
        >>> a, b = symbols('a b')
        >>> D = DisplacementGate(a + I*b, parametric=True)
        >>> D.get_parameters()
        {a, b}
        """
        if not self.parametric or not isinstance(self.alpha, Expr):
            return set()
        return set(self.alpha.free_symbols)

    def conjugate(self) -> CompactDiagram:
        """Conjugate of displacement gate is displacement with negated alpha.

        Returns:
        -------
        DisplacementGate
            D(-a)
        """
        if self.parametric:
            return DisplacementGate(-self.alpha, parametric=True)
        return DisplacementGate(alpha=-self.alpha)

    def __repr__(self) -> str:
        """Return string representation of the displacement gate.

        Returns:
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
        """Validate that alpha is a valid complex number or sympy expression."""  # noqa: DOC501
        if self.parametric and not isinstance(self.alpha, Expr):
            msg = f"Expected sympy expression for parametric mode, got {type(self.alpha)}"
            raise ValueError(msg)
        if not isinstance(self.alpha, (complex, float, int)) and not self.parametric:
            msg_0 = f"Expected numeric type, got {type(self.alpha)}"
            raise TypeError(msg_0)

    def __post_init__(self) -> None:
        """Initialize the displacement gate and validate parameters."""  # noqa: DOC501
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

        # Check feedforward
        if self.feedforward:
            if not isinstance(self.measurement_ids, set):
                msg = f"The measurement_ids attribute must be a set, got {type(self.measurement_ids)}."
                raise ValueError(msg)
            if not self.measurement_ids:
                msg = "The measurement_ids attribute can not be empty."
                raise ValueError(msg)
        super().__post_init__()

    def _get_re_im(self) -> tuple[float | int | complex | Expr, float | int | complex | Expr]:
        """Extract real and imaginary parts of alpha.

        Returns:
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

    def _create_phase_poly(self, coeff: complex | Expr, degree: int = 1) -> ZxPoly:
        """Create a phase polynomial for the spider.

        Parameters
        ----------
        coeff : float | int | complex | Expr
            Coefficient for the phase polynomial.
        degree : int
            Degree of the monomial. Default 1 for linear phase.

        Returns:
        -------
        ZxPoly
            Phase polynomial with the given coefficient.
        """
        if self.parametric:
            return ZxPoly({degree: coeff})
        # Convert to float for numeric case
        if isinstance(coeff, complex):
            return ZxPoly({degree: float(coeff)})
        return ZxPoly({degree: float(coeff)})


@dataclass
class PhaseRotationGate(CompactDiagram):
    r"""Phase rotation gate R(θ).

    Represents the phase rotation operator R(θ) = exp(iθ â† â).
    Decomposes into three quadratic q-spiders as shown in [3] Eq. (58).

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

    Raises:
    ------
    ValueError
        If theta is an odd multiple of π/2 (where tan is infinite).
        For these cases, use Fourier2 or composition of Fourier gates.

    Examples:
    --------
    >>> # Numeric phase rotation
    >>> R = PhaseRotationGate(np.pi/4)
    >>> R.expand()

    >>> # Symbolic phase rotation
    >>> from sympy import symbols
    >>> theta = symbols('theta', real=True)
    >>> R = PhaseRotationGate(theta, parametric=True)
    >>> decomp = R.expand()

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.2, Eq. (58)
    """

    theta: float | int | complex | Expr
    parametric: bool = False
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    def expand(self) -> CompositionDiagram:
        """Decompose phase rotation into three quadratic q-spiders.

        R(θ) = P(tan(θ/2)/2) ∘ Q(-sinθ/2) ∘ P(tan(θ/2)/2)

        Returns:
        -------
        CompositionDiagram
            Composition of three q-spiders.

        Examples:
        --------
        >>> R = PhaseRotationGate(np.pi/4)
        >>> decomp = R.expand()
        """
        tan_half = self._get_tan_half()
        sin_theta = self._get_sin()

        spider1 = PSpider(1, 1, ZxPoly({2: tan_half / 2}))
        spider2 = QSpider(1, 1, ZxPoly({2: -sin_theta / 2}))
        spider3 = PSpider(1, 1, ZxPoly({2: tan_half / 2}))

        return CompositionDiagram([spider1, spider2, spider3])

    def substitute_parameters(self, mapping: dict[Symbol | str, Any]) -> CompactDiagram:
        """Substitute symbolic parameters in the phase rotation gate.

        Parameters
        ----------
        mapping : dict
            Dictionary mapping symbols to values.

        Returns:
        -------
        PhaseRotationGate
            A new PhaseRotationGate with substituted parameters.
        """
        if not self.parametric:
            return self

        new_mapping = {}
        for key, value in mapping.items():
            n_key = key
            if isinstance(key, str):
                n_key = symbols(key)
            new_mapping[n_key] = value

        new_theta = simplify(self.theta.subs(new_mapping))

        if is_numeric(new_theta):
            return PhaseRotationGate(float(new_theta), parametric=False)
        return PhaseRotationGate(new_theta, parametric=True)

    def evaluate(self, **kwargs) -> CompactDiagram:  # noqa: ANN003
        """Evaluate the phase rotation gate with given parameter values.

        Parameters
        ----------
        **kwargs : dict
            Parameter values as keyword arguments.

        Returns:
        -------
        PhaseRotationGate
            A new PhaseRotationGate with evaluated parameters.
        """
        if not self.parametric:
            return self

        mapping = {}
        for name, value in kwargs.items():
            mapping[symbols(name)] = value

        return self.substitute_parameters(mapping)

    def get_parameters(self) -> set[Symbol]:
        """Get all symbolic parameters in the phase rotation gate.

        Returns:
        -------
        set[Symbol]
            Set of symbols used in the gate.
        """
        if not self.parametric or not isinstance(self.theta, Expr):
            return set()
        return set(self.theta.free_symbols)

    def conjugate(self) -> CompactDiagram:
        """Conjugate of phase rotation is rotation by negative angle.

        Returns:
        -------
        PhaseRotationGate
            R(-θ)
        """
        if self.parametric:
            return PhaseRotationGate(-self.theta, parametric=True)
        return PhaseRotationGate(theta=-self.theta)

    def __repr__(self) -> str:
        """Return string representation of the phase rotation gate.

        Returns:
        -------
        str
            String showing the rotation angle theta.
        """
        if self.parametric:
            return f"PhaseRotationGate(theta={self.theta}, parametric=True)"
        return f"PhaseRotationGate(theta={self.theta:.2f})"

    def _validate_theta(self) -> None:
        """Validate theta for invalid angles (odd multiples of π/2)."""  # noqa: DOC501
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

        super().__post_init__()

    def _get_tan_half(self) -> float | int | complex | Expr:
        """Get tan(θ/2) for the decomposition.

        Returns:
        -------
            float | int | complex | Expr
        """
        if self.parametric:
            return tan(self.theta / 2)
        return np.tan(self.theta / 2)

    def _get_sin(self) -> float | int | complex | Expr:
        """Get sin(θ) for the decomposition.

        Returns:
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
    [3] Eq. (59).

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

    Examples:
    --------
    >>> # Numeric squeezing
    >>> S = SqueezingGate(0.5)
    >>> S.expand()

    >>> # Symbolic squeezing
    >>> from sympy import symbols
    >>> r = symbols('r', real=True)
    >>> S = SqueezingGate(exp(-r), parametric=True)
    >>> decomp = S.expand()

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.3, Eq. (59)
    """

    tau: float | int | complex | Expr
    parametric: bool = False
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    def expand(self) -> CompositionDiagram:
        """Decompose squeezing gate into four quadratic spiders.

        Sq(τ) = Q(a) ∘ P(b) ∘ Q(c) ∘ P(d)
        where a = τ(1-τ)/4, b = -1/τ, c = (τ-1)/4, d = 1

        Returns:
        -------
        CompositionDiagram
            Composition of Q, P, Q, P spiders in sequence.

        Examples:
        --------
        >>> S = SqueezingGate(0.5)
        >>> decomp = S.expand()
        """
        a, b, c, d = self._get_coefficients()

        spider1 = QSpider(1, 1, ZxPoly({2: a}))
        spider2 = PSpider(1, 1, ZxPoly({2: b}))
        spider3 = QSpider(1, 1, ZxPoly({2: c}))
        spider4 = PSpider(1, 1, ZxPoly({2: d}))

        return CompositionDiagram([spider1, spider2, spider3, spider4])

    def substitute_parameters(self, mapping: dict[Symbol | str, Any]) -> CompactDiagram:
        """Substitute symbolic parameters in the squeezing gate.

        Parameters
        ----------
        mapping : dict
            Dictionary mapping symbols to values.

        Returns:
        -------
        SqueezingGate
            A new SqueezingGate with substituted parameters.
        """
        if not self.parametric:
            return self

        new_mapping = {}
        for key, value in mapping.items():
            n_key = key
            if isinstance(key, str):
                n_key = symbols(key)
            new_mapping[n_key] = value

        new_tau = simplify(self.tau.subs(new_mapping))

        if is_numeric(new_tau):
            return SqueezingGate(float(new_tau), parametric=False)
        return SqueezingGate(new_tau, parametric=True)

    def evaluate(self, **kwargs) -> CompactDiagram:  # noqa: ANN003
        """Evaluate the squeezing gate with given parameter values.

        Parameters
        ----------
        **kwargs : dict
            Parameter values as keyword arguments.

        Returns:
        -------
        SqueezingGate
            A new SqueezingGate with evaluated parameters.
        """
        if not self.parametric:
            return self

        mapping = {}
        for name, value in kwargs.items():
            mapping[symbols(name)] = value

        return self.substitute_parameters(mapping)

    def get_parameters(self) -> set[Symbol]:
        """Get all symbolic parameters in the squeezing gate.

        Returns:
        -------
        set[Symbol]
            Set of symbols used in the gate.
        """
        if not self.parametric or not isinstance(self.tau, Expr):
            return set()
        return set(self.tau.free_symbols)

    def conjugate(self) -> CompactDiagram:
        """Conjugate of squeezing gate is squeezing with reciprocal parameter.

        Returns:
        -------
        SqueezingGate
            Sq(1/τ)
        """
        if self.parametric:
            return SqueezingGate(1 / self.tau, parametric=True)
        return SqueezingGate(tau=1 / self.tau)

    def __repr__(self) -> str:
        """Return string representation of the squeezing gate.

        Returns:
        -------
        str
            String showing the squeezing parameter tau.
        """
        if self.parametric:
            return f"SqueezingGate(tau={self.tau}, parametric=True)"
        return f"SqueezingGate(tau={self.tau:.2f})"

    def _validate_tau(self) -> None:
        """Validate tau parameter."""  # noqa: DOC501
        if self.parametric:
            return
        if abs(self.tau) < 1e-10:  # noqa: PLR2004
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

        Returns:
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
            d = 1
        else:
            a = self.tau * (1 - self.tau) / 4
            b = -1 / self.tau if abs(self.tau) > 1e-10 else 0  # noqa: PLR2004
            c = (self.tau - 1) / 4
            d = 1.0
        return a, b, c, d


@dataclass
class ControlledSumGate(CompactDiagram):
    r"""Controlled-sum (CSUM) gate with gain g and specified control/target modes.

    Represents the operation exp(-i g q̂_c p̂_t) where c is the control mode
    and t is the target mode. For g=1, this is the unbiased CSUM gate
    (CV analogue of CNOT). Decomposes into q-spider and p-spider
    with a contraction as shown in [3] Eq. (61)-(62).

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

    Raises:
    ------
    ValueError
        If control == target (must be different modes).

    Examples:
    --------
    >>> # Unbiased CSUM
    >>> C = ControlledSumGate(control=2, target=1)
    >>> C.expand()

    >>> # Biased CSUM with symbolic gain
    >>> from sympy import symbols
    >>> g = symbols('g', real=True)
    >>> C = ControlledSumGate(gain=g, control=2, target=1, parametric=True)
    >>> decomp = C.expand()

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.4, Eq. (61)-(62)
    [4] Yoshikawa et al., QRL configuration, Sec. IV.C.3
    """

    gain: float | int | complex | Expr = 1.0
    control: int = 2
    target: int = 1
    parametric: bool = False
    label: str = field(init=False)
    _num_inputs: int = field(default=2, init=False)
    _num_outputs: int = field(default=2, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    def expand(self) -> Diagram:
        """Decompose CSUM gate into spiders with contraction.

        For unbiased (g=1) gate [3] Eq. (61):
            CSUM2,1 = ContractedDiagram of q-spider and p-spider
            CSUM1,2 = ContractedDiagram of p-spider and q-spider

        For biased gate [3] Eq. (62):
            CSUM1,2(g) = (Sq(g) ⊗ Id) ∘ CSUM1,2(1) ∘ (Sq(g⁻¹) ⊗ Id)
            CSUM2,1(g) = (Sq(g) ⊗ Id) ∘ CSUM2,1(1) ∘ (Sq(g⁻¹) ⊗ Id)

        The squeezing is applied to the control mode.

        Returns:
        -------
        Diagram
            ContractedDiagram for unbiased, CompositionDiagram for biased.

        Examples:
        --------
        >>> C = ControlledSumGate(control=2, target=1)
        >>> decomp = C.expand()
        """
        # q-spider copies position from control mode
        # p-spider adds the copied position to the momentum of target mode
        if self.control == 2:  # noqa: PLR2004
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
        squeeze1 = SqueezingGate(tau=sqrt_gain, parametric=self.parametric)
        upper_diagram = control_spider.compose(squeeze1)
        squeeze2 = SqueezingGate(tau=inv_sqrt, parametric=self.parametric)
        squeeze2_id = squeeze2.tensor(identity)
        upper_diagram = squeeze2_id.compose(upper_diagram)

        tensor = TensorDiagram([upper_diagram, target_spider])
        tensor.partial_trace([
            (0, [1], []),
            (1, [], [0]),
        ])

        return tensor.diagrams[0]

    def substitute_parameters(self, mapping: dict[Symbol | str, Any]) -> CompactDiagram:
        """Substitute symbolic parameters in the CSUM gate.

        Parameters
        ----------
        mapping : dict
            Dictionary mapping symbols to values.

        Returns:
        -------
        ControlledSumGate
            A new ControlledSumGate with substituted parameters.
        """
        if not self.parametric:
            return self

        new_mapping = {}
        for key, value in mapping.items():
            n_key = key
            if isinstance(key, str):
                n_key = symbols(key)
            new_mapping[n_key] = value

        new_gain = simplify(self.gain.subs(new_mapping))

        if is_numeric(new_gain):
            return ControlledSumGate(float(new_gain), parametric=False)
        return ControlledSumGate(new_gain, parametric=True)

    def evaluate(self, **kwargs: Any) -> CompactDiagram:  # noqa: ANN401
        """Evaluate the CSUM gate with given parameter values.

        Parameters
        ----------
        **kwargs : dict
            Parameter values as keyword arguments.

        Returns:
        -------
        ControlledSumGate
            A new ControlledSumGate with evaluated parameters.
        """
        if not self.parametric:
            return self

        mapping = {}
        for name, value in kwargs.items():
            mapping[symbols(name)] = value

        return self.substitute_parameters(mapping)

    def get_parameters(self) -> set[Symbol]:
        """Get all symbolic parameters in the CSUM gate.

        Returns:
        -------
        set[Symbol]
            Set of symbols used in the gate.
        """
        if not self.parametric or not isinstance(self.gain, Expr):
            return set()
        return set(self.gain.free_symbols)

    def conjugate(self) -> CompactDiagram:
        """Conjugate of CSUM is CSUM with negated gain (inverse).

        (e^{-i g q_c p_t})† = e^{+i g q_c p_t} = CSUM(-g)

        Returns:
        -------
        ControlledSumGate
            CSUM with gain = -g (same control/target).
        """
        if self.parametric:
            return ControlledSumGate(gain=-self.gain, control=self.control, target=self.target, parametric=True)
        return ControlledSumGate(gain=-self.gain, control=self.control, target=self.target)

    def __repr__(self) -> str:
        """Return string representation of the controlled-sum gate.

        Returns:
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
        """Initialize the controlled-sum gate and validate parameters."""  # noqa: DOC501
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

        super().__post_init__()

    def _is_unbiased(self) -> bool:
        """Check if this is an unbiased CSUM gate (gain = 1).

        Returns:
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

    Attributes:
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

    Examples:
    --------
    >>> # Unbiased CZ
    >>> CZ = ControlledZGate()
    >>> CZ.expand()

    >>> # Symbolic CZ
    >>> from sympy import symbols
    >>> g = symbols('g', real=True)
    >>> CZ = ControlledZGate(gain=g, parametric=True)
    >>> decomp = CZ.expand()

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.5, Eq. (63)-(64)
    """

    gain: float | int | complex | Expr = 1.0
    parametric: bool = False
    label: str = field(init=False)
    _num_inputs: int = field(default=2, init=False)
    _num_outputs: int = field(default=2, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    def expand(self) -> Diagram:
        """Decompose CZ gate using Fourier gates and CSUM.

        Unbiased CZ [3] Eq. (63):
            CZ = ContractedDiagram of q-spider and p-spider with
            a fourier diagram in between

        Biased CZ [3] Eq. (64):
            CZ(g) = (Sq(g⁻¹) ⊗ Id) ∘ CZ(1) ∘ (Sq(g) ⊗ Id)

        Returns:
        -------
        Diagram
            Composition of Fourier, CSUM, Fourier.

        Examples:
        --------
        >>> CZ = ControlledZGate()
        >>> decomp = CZ.expand()
        """
        # Fourier diagram
        fourier_inv = FourierInv()

        q_spider1 = QSpider(2, 1, ZxPoly({}))
        q_spider2 = QSpider(1, 2, ZxPoly({}))
        identity = QSpider(1, 1, ZxPoly({}))
        i_tensor_f = TensorDiagram([fourier_inv, identity])

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

        squeeze1 = SqueezingGate(tau=sqrt_gain, parametric=self.parametric)
        squeeze1_id = squeeze1.tensor(identity)
        upper_diagram = q_spider1.compose(squeeze1_id)
        squeeze2 = SqueezingGate(tau=inv_sqrt, parametric=self.parametric)
        upper_diagram = squeeze2.compose(upper_diagram)

        tensor = TensorDiagram([upper_diagram, i_tensor_f.compose(q_spider2)])
        tensor.partial_trace([
            (0, [], [1]),
            (1, [0], []),
        ])

        return tensor.diagrams[0]

    def substitute_parameters(self, mapping: dict[Symbol | str, Any]) -> CompactDiagram:
        """Substitute symbolic parameters in the CZ gate.

        Parameters
        ----------
        mapping : dict
            Dictionary mapping symbols to values.

        Returns:
        -------
        ControlledZGate
            A new ControlledZGate with substituted parameters.
        """
        if not self.parametric:
            return self

        new_mapping = {}
        for key, value in mapping.items():
            n_key = key
            if isinstance(key, str):
                n_key = symbols(key)
            new_mapping[n_key] = value

        new_gain = simplify(self.gain.subs(new_mapping))

        if is_numeric(new_gain):
            return ControlledZGate(float(new_gain), parametric=False)
        return ControlledZGate(new_gain, parametric=True)

    def evaluate(self, **kwargs) -> CompactDiagram:  # noqa: ANN003
        """Evaluate the CZ gate with given parameter values.

        Parameters
        ----------
        **kwargs : dict
            Parameter values as keyword arguments.

        Returns:
        -------
        ControlledZGate
            A new ControlledZGate with evaluated parameters.
        """
        if not self.parametric:
            return self

        mapping = {}
        for name, value in kwargs.items():
            mapping[symbols(name)] = value

        return self.substitute_parameters(mapping)

    def get_parameters(self) -> set[Symbol]:
        """Get all symbolic parameters in the CZ gate.

        Returns:
        -------
        set[Symbol]
            Set of symbols used in the gate.
        """
        if not self.parametric or not isinstance(self.gain, Expr):
            return set()
        return set(self.gain.free_symbols)

    def conjugate(self) -> CompactDiagram:
        """Conjugate of CZ is CZ with same gain (self-adjoint).

        Returns:
        -------
        ControlledZGate
            CZ(g)
        """
        if self.parametric:
            return ControlledZGate(gain=self.gain, parametric=True)
        return ControlledZGate(gain=self.gain)

    def __repr__(self) -> str:
        """Return string representation of the controlled-Z gate.

        Returns:
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

        super().__post_init__()

    def _is_unbiased(self) -> bool:
        """Check if this is an unbiased CSUM gate (gain = 1).

        Returns:
        -------
        bool
        """
        return self.gain == 1


@dataclass
class BeamsplitterGate(CompactDiagram):
    r"""Beamsplitter gate BS(θ).

    Represents the operation exp(-iθ (q̂₁ p̂₂ - p̂₁ q̂₂)). Decomposes into
    squeezing gates and CSUM gates as shown in [3] Eq. (66).

    Parameters
    ----------
    theta : float | int | Expr
        Beamsplitter angle. θ = π/4 gives a 50:50 beamsplitter.
    parametric : bool
        If True, treat theta as symbolic parameter. Default False.

    Attributes:
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

    Examples:
    --------
    >>> # 50:50 beamsplitter
    >>> BS = BeamsplitterGate(np.pi/4)
    >>> BS.expand()

    >>> # Symbolic beamsplitter
    >>> from sympy import symbols
    >>> theta = symbols('theta', real=True)
    >>> BS = BeamsplitterGate(theta, parametric=True)
    >>> decomp = BS.expand()

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.6, Eq. (66)-(67)
    """

    theta: float | int | complex | Expr
    parametric: bool = False
    label: str = field(init=False)
    _num_inputs: int = field(default=2, init=False)
    _num_outputs: int = field(default=2, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="gate", init=False)

    def expand(self) -> Diagram:
        """Decompose beamsplitter using squeezing and CSUM gates.

        Returns:
        -------
        Diagram
            Composition following [3] Eq. (66) or simplified Eq. (67).

        Examples:
        --------
        >>> BS = BeamsplitterGate(np.pi/4)
        >>> decomp = BS.expand()
        """
        if self._is_balanced():
            # Balanced beamsplitter [3] Eq. (67)
            if self.parametric:
                sqrt2 = sqrt(2)
                inv_sqrt2 = 1 / sqrt2
            else:
                sqrt2 = np.sqrt(2)
                inv_sqrt2 = 1 / sqrt2

            csum12 = ControlledSumGate(gain=1, control=1, target=2)
            csum21 = ControlledSumGate(gain=1, control=2, target=1)

            tensor = TensorDiagram([
                SqueezingGate(tau=sqrt2, parametric=self.parametric),
                SqueezingGate(tau=inv_sqrt2, parametric=self.parametric),
            ])

            return CompositionDiagram([expand_all(csum12), tensor, expand_all(csum21)])

        # General beamsplitter - simplified representation
        # Full decomposition from [3] Appendix A.1.f
        if self.parametric:
            tan_theta = tan(self.theta)
            sin2_theta = sin(2 * self.theta)
            cos_theta = cos(self.theta)
        else:
            tan_theta = np.tan(self.theta)
            sin2_theta = np.sin(2 * self.theta)
            cos_theta = np.cos(self.theta)

        sq1 = SqueezingGate(tau=1 / tan_theta, parametric=self.parametric)
        sq2 = SqueezingGate(tau=sin2_theta / cos_theta, parametric=self.parametric)
        sq3 = SqueezingGate(tau=1 / cos_theta, parametric=self.parametric)
        identity = QSpider(1, 1, ZxPoly({}))
        tensor1 = sq1.tensor(identity)
        tensor2 = sq2.tensor(sq3)

        csum12 = ControlledSumGate(gain=1, control=1, target=2)
        csum21 = ControlledSumGate(gain=1, control=2, target=1)

        return CompositionDiagram([tensor1, expand_all(csum12), tensor2, expand_all(csum21), tensor1])

    def substitute_parameters(self, mapping: dict[Symbol | str, Any]) -> CompactDiagram:
        """Substitute symbolic parameters in the beamsplitter gate.

        Parameters
        ----------
        mapping : dict
            Dictionary mapping symbols to values.

        Returns:
        -------
        BeamsplitterGate
            A new BeamsplitterGate with substituted parameters.
        """
        if not self.parametric:
            return self

        new_mapping = {}
        for key, value in mapping.items():
            n_key = key
            if isinstance(key, str):
                n_key = symbols(key)
            new_mapping[n_key] = value

        new_theta = simplify(self.theta.subs(new_mapping))

        if is_numeric(new_theta):
            return BeamsplitterGate(float(new_theta), parametric=False)
        return BeamsplitterGate(new_theta, parametric=True)

    def evaluate(self, **kwargs) -> CompactDiagram:  # noqa: ANN003
        """Evaluate the beamsplitter gate with given parameter values.

        Parameters
        ----------
        **kwargs : dict
            Parameter values as keyword arguments.

        Returns:
        -------
        BeamsplitterGate
            A new BeamsplitterGate with evaluated parameters.
        """
        if not self.parametric:
            return self

        mapping = {}
        for name, value in kwargs.items():
            mapping[symbols(name)] = value

        return self.substitute_parameters(mapping)

    def get_parameters(self) -> set[Symbol]:
        """Get all symbolic parameters in the beamsplitter gate.

        Returns:
        -------
        set[Symbol]
            Set of symbols used in the gate.
        """
        if not self.parametric or not isinstance(self.theta, Expr):
            return set()
        return set(self.theta.free_symbols)

    def conjugate(self) -> CompactDiagram:
        """Conjugate of beamsplitter is beamsplitter with negated angle.

        Returns:
        -------
        BeamsplitterGate
            BS(-θ)
        """
        if self.parametric:
            return BeamsplitterGate(theta=-self.theta, parametric=True)
        return BeamsplitterGate(theta=-self.theta)

    def __repr__(self) -> str:
        """Return string representation of the beamsplitter gate.

        Returns:
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

        super().__post_init__()

    def _is_balanced(self) -> bool:
        """Check if this is a balanced 50:50 beamsplitter.

        Returns:
        -------
        bool
        """
        if self.parametric:
            return self.theta == pi / 4  # noqa: RUF069
        return np.isclose(self.theta, np.pi / 4)


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

    Examples:
    --------
    >>> # Numeric cubic phase gate
    >>> CPG = CubicPhaseGate(0.5)
    >>> CPG.expand()

    >>> # Symbolic cubic phase gate
    >>> from sympy import symbols
    >>> gamma = symbols('gamma', real=True)
    >>> CPG = CubicPhaseGate(gamma, parametric=True)
    >>> decomp = CPG.expand()

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.7, Eq. (68)
    """

    gamma: float | int | Expr
    parametric: bool = False
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)
    spider_type: str = field(default="non_gaussian", init=False)

    def expand(self) -> QSpider:
        """Decompose cubic phase gate into a single q-spider with cubic phase.

        Returns:
        -------
        QSpider
            q-spider with phase function f(x) = y x³.

        Examples:
        --------
        >>> CPG = CubicPhaseGate(0.5)
        >>> spider = CPG.expand()
        """
        return QSpider(1, 1, ZxPoly({3: self.gamma}))

    def substitute_parameters(self, mapping: dict[Symbol | str, Any]) -> CompactDiagram:
        """Substitute symbolic parameters in the cubic phase gate.

        Parameters
        ----------
        mapping : dict
            Dictionary mapping symbols to values.

        Returns:
        -------
        CubicPhaseGate
            A new CubicPhaseGate with substituted parameters.
        """
        if not self.parametric:
            return self

        new_mapping = {}
        for key, value in mapping.items():
            n_key = key
            if isinstance(key, str):
                n_key = symbols(key)
            new_mapping[n_key] = value

        new_gamma = simplify(self.gamma.subs(new_mapping))

        if is_numeric(new_gamma):
            return CubicPhaseGate(float(new_gamma), parametric=False)
        return CubicPhaseGate(new_gamma, parametric=True)

    def evaluate(self, **kwargs) -> CompactDiagram:  # noqa: ANN003
        """Evaluate the cubic phase gate with given parameter values.

        Parameters
        ----------
        **kwargs : dict
            Parameter values as keyword arguments.

        Returns:
        -------
        CubicPhaseGate
            A new CubicPhaseGate with evaluated parameters.
        """
        if not self.parametric:
            return self

        mapping = {}
        for name, value in kwargs.items():
            mapping[symbols(name)] = value

        return self.substitute_parameters(mapping)

    def get_parameters(self) -> set[Symbol]:
        """Get all symbolic parameters in the cubic phase gate.

        Returns:
        -------
        set[Symbol]
            Set of symbols used in the gate.
        """
        if not self.parametric or not isinstance(self.gamma, Expr):
            return set()
        return set(self.gamma.free_symbols)

    def conjugate(self) -> CompactDiagram:
        """Conjugate of cubic phase gate is cubic phase with negated gamma.

        Returns:
        -------
        CubicPhaseGate
            CPG(-y)
        """
        if self.parametric:
            return CubicPhaseGate(gamma=-self.gamma, parametric=True)
        return CubicPhaseGate(gamma=-self.gamma)

    def __repr__(self) -> str:
        """Return string representation of the cubic phase gate.

        Returns:
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

        super().__post_init__()


# =============================================================================
# Helper functions
# =============================================================================


def is_numeric(expr: Expr) -> bool:
    """Check if a sympy expression is purely numeric.

    Parameters:
    ----------
    exp : Expr
        Symbolic parameter.

    Returns:
    -------
    bool

    """
    return len(expr.free_symbols) == 0


def create_compact_diagram(label: str, num_inputs: int, num_outputs: int, decomp: Diagram) -> CompactDiagram:
    """Create a compact diagram for a given diagram.

    Parameters:
    ----------
    label : str
        Label for the block.
    num_inputs : int
        Number of input wires.
    num_outputs : int
        Number of output wires.
    decomp : Diagram
        Diagram to compact.

    Returns:
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

    Parameters:
    ----------
    diagram : Diagram
        The diagram to expand.

    Returns:
    -------
    Diagram
        The expanded diagram with all CompactDiagram expanded.
    """
    if isinstance(diagram, (BeamsplitterGate, ControlledSumGate, ControlledZGate)):
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

"""CV-ZX representation of usual gates."""

import operator
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field


class ZxPoly:
    """Real polynomial in one variable for CV ZX calculus phase functions.

    This class represents real polynomials of the form:
        f(x) = c_0 + c_1·x + c_2·x² + ... + c_n·xⁿ

    The polynomial is stored as a dictionary mapping degree → coefficient.
    Zero coefficients are omitted from the dictionary.

    Attributes:
    ----------
    coeffs : dict[int, float]
        Dictionary where keys are monomial degrees (non-negative integers)
        and values are the corresponding real coefficients.

    Examples:
    --------
    >>> p = ZxPoly({0: 1.0, 2: -0.5})  # 1 - 0.5·x²
    >>> q = ZxPoly({1: 2.0})            # 2·x
    >>> r = p + q                       # 1 + 2·x - 0.5·x²
    >>> r.degree()
    2
    >>> s = p * q                       # (1 - 0.5·x²)·(2·x) = 2·x - x³
    """

    def __init__(self, coeffs: dict[int, float]) -> None:
        """Initialize a polynomial from coefficient dictionary.

        Parameters
        ----------
        coeffs : dict[int, float]
            Dictionary mapping degree → coefficient. Zero coefficients
            may be omitted. An empty dictionary represents the zero polynomial.

        Examples:
        --------
        >>> p = ZxPoly({0: 1.0, 2: -0.5})  # 1 - 0.5·x²
        >>> q = ZxPoly({})                  # zero polynomial
        """
        # Remove zero coefficients for clean representation
        self.coeffs = {k: v for k, v in coeffs.items() if v != 0}

    def __add__(self, other: "ZxPoly") -> "ZxPoly":
        """Add two polynomials.

        Parameters
        ----------
        other : ZxPoly
            Polynomial to add to this one.

        Returns:
        -------
        ZxPoly
            New polynomial representing (self + other).

        Examples:
        --------
        >>> p = ZxPoly({0: 1.0, 2: 0.5})
        >>> q = ZxPoly({2: -0.5, 3: 1.0})
        >>> r = p + q
        >>> r.coeffs
        {0: 1.0, 3: 1.0}
        """
        result = self.coeffs.copy()
        for degree, coeff in other.coeffs.items():
            if degree in result:
                result[degree] += coeff
                if result[degree] == 0:
                    del result[degree]
            else:
                result[degree] = coeff
        return ZxPoly(result)

    def __sub__(self, other: "ZxPoly") -> "ZxPoly":
        """Subtract another polynomial.

        Parameters
        ----------
        other : ZxPoly
            Polynomial to subtract from this one.

        Returns:
        -------
        ZxPoly
            New polynomial representing (self - other).

        Examples:
        --------
        >>> p = ZxPoly({0: 1.0, 2: 0.5})
        >>> q = ZxPoly({2: 0.5, 3: 1.0})
        >>> r = p - q
        >>> r.coeffs
        {0: 1.0, 3: -1.0}
        """
        result = self.coeffs.copy()
        for degree, coeff in other.coeffs.items():
            if degree in result:
                result[degree] -= coeff
                if result[degree] == 0:
                    del result[degree]
            else:
                result[degree] = -coeff
        return ZxPoly(result)

    def __mul__(self, other: "ZxPoly") -> "ZxPoly":
        """Multiply two polynomials.

        Parameters
        ----------
        other : ZxPoly
            Polynomial to multiply with this one.

        Returns:
        -------
        ZxPoly
            New polynomial representing (self * other).

        Examples:
        --------
        >>> p = ZxPoly({0: 2.0, 1: 1.0})  # 2 + x
        >>> q = ZxPoly({1: 1.0, 2: 3.0})  # x + 3x²
        >>> r = p * q
        >>> r.coeffs
        {1: 2.0, 2: 7.0, 3: 3.0}  # 2x + 7x² + 3x³
        """
        result: dict[int, float] = {}
        for deg1, coeff1 in self.coeffs.items():
            for deg2, coeff2 in other.coeffs.items():
                new_deg = deg1 + deg2
                new_coeff = coeff1 * coeff2
                if new_deg in result:
                    result[new_deg] += new_coeff
                    if result[new_deg] == 0:
                        del result[new_deg]
                else:
                    result[new_deg] = new_coeff
        return ZxPoly(result)

    def __rmul__(self, scalar: float) -> "ZxPoly":
        """Multiply polynomial by a scalar (left multiplication).

        Parameters
        ----------
        scalar : int | float
            Scalar multiplier.

        Returns:
        -------
        ZxPoly
            New polynomial scaled by the scalar.

        Examples:
        --------
        >>> p = ZxPoly({0: 1.0, 2: 0.5})
        >>> q = 2 * p
        >>> q.coeffs
        {0: 2.0, 2: 1.0}
        """
        if scalar == 0:
            return ZxPoly({})
        result = {deg: coeff * scalar for deg, coeff in self.coeffs.items()}
        return ZxPoly(result)

    def __neg__(self) -> "ZxPoly":
        """Negate the polynomial.

        Returns:
        -------
        ZxPoly
            New polynomial representing (-self).

        Examples:
        --------
        >>> p = ZxPoly({0: 1.0, 2: 0.5})
        >>> q = -p
        >>> q.coeffs
        {0: -1.0, 2: -0.5}
        """
        return ZxPoly({deg: -coeff for deg, coeff in self.coeffs.items()})

    def __hash__(self) -> int:
        """Compute hash value for the polynomial.

        Returns:
        -------
        int
            Hash value based on the coefficient dictionary.

        Notes:
        -----
        The class must be immutable for consistent hashing.
        All methods that modify coeffs should return new instances
        rather than mutating self.coeffs.

        Examples:
        --------
        >>> p = ZxPoly({0: 1.0, 2: -0.5})
        >>> q = ZxPoly({0: 1.0, 2: -0.5})
        >>> hash(p) == hash(q)
        True
        >>> d = {p: "polynomial"}
        >>> d[q]
        'polynomial'
        """
        # Hash based on sorted items for consistency
        # Convert float coefficients to a hashable representation
        items = tuple(sorted((deg, coeff) for deg, coeff in self.coeffs.items()))
        return hash(items)

    def degree(self) -> int:
        """Return the degree of the polynomial.

        The degree is the highest exponent with non-zero coefficient.
        The zero polynomial has degree -1 by convention.

        Returns:
        -------
        int
            Degree of the polynomial, or -1 if polynomial is zero.

        Examples:
        --------
        >>> ZxPoly({0: 1.0, 3: -2.0}).degree()
        3
        >>> ZxPoly({}).degree()
        -1
        """
        if not self.coeffs:
            return -1
        return max(self.coeffs.keys())

    def evaluate(self, x: float) -> float:
        """Evaluate the polynomial at a given point.

        Uses Horner's method for numerical stability.

        Parameters
        ----------
        x : int | float
            Point at which to evaluate the polynomial.

        Returns:
        -------
        float
            Value of f(x).

        Examples:
        --------
        >>> p = ZxPoly({0: 1.0, 1: 2.0, 2: 3.0})  # 1 + 2x + 3x²
        >>> p.evaluate(2.0)
        17.0  # 1 + 4 + 12 = 17
        """
        if not self.coeffs:
            return 0.0

        # Horner's method: evaluate from highest degree down
        deg = self.degree()
        result = self.coeffs.get(deg, 0.0)
        for d in range(deg - 1, -1, -1):
            result = result * x + self.coeffs.get(d, 0.0)
        return result

    def compose(self, other: "ZxPoly") -> "ZxPoly":
        """Compose this polynomial with another: self(other(x)).

        Parameters
        ----------
        other : ZxPoly
            Polynomial to substitute into this one.

        Returns:
        -------
        ZxPoly
            New polynomial representing f(g(x)) where f = self, g = other.

        Examples:
        --------
        >>> f = ZxPoly({0: 1.0, 1: 2.0})  # 1 + 2x
        >>> g = ZxPoly({0: 3.0, 1: 4.0})  # 3 + 4x
        >>> h = f.compose(g)              # 1 + 2(3 + 4x) = 7 + 8x
        >>> h.coeffs
        {0: 7.0, 1: 8.0}
        """
        if not self.coeffs:
            return ZxPoly({})

        result = ZxPoly({0: 0.0})
        # Build from constant term upward using repeated multiplication
        power = ZxPoly({0: 1.0})  # g(x)⁰ = 1
        for deg in range(self.degree() + 1):
            coeff = self.coeffs.get(deg, 0.0)
            if coeff != 0:
                term = power * ZxPoly({0: coeff})
                result += term
            # Update power = power * other for next degree
            if deg < self.degree():
                power *= other
        return result

    def derivative(self) -> "ZxPoly":
        """Return the derivative of the polynomial.

        Returns:
        -------
        ZxPoly
            New polynomial representing f'(x).

        Examples:
        --------
        >>> p = ZxPoly({0: 1.0, 1: 2.0, 2: 3.0})  # 1 + 2x + 3x²
        >>> q = p.derivative()                     # 2 + 6x
        >>> q.coeffs
        {0: 2.0, 1: 6.0}
        """
        if not self.coeffs or self.degree() <= 0:
            return ZxPoly({})

        result: dict[int, float] = {}
        for deg, coeff in self.coeffs.items():
            if deg > 0:
                result[deg - 1] = coeff * deg
        return ZxPoly(result)

    def shift(self, a: float) -> "ZxPoly":
        """Return the polynomial shifted by a: f(x + a).

        Uses binomial expansion: (x + a)ⁿ = Σ C(n,k) xᵏ aⁿ⁻ᵏ.

        Parameters
        ----------
        a : int | float
            Amount to shift by.

        Returns:
        -------
        ZxPoly
            New polynomial representing f(x + a).

        Examples:
        --------
        >>> p = ZxPoly({0: 1.0, 1: 1.0})  # 1 + x
        >>> q = p.shift(2.0)              # 1 + (x+2) = 3 + x
        >>> q.coeffs
        {0: 3.0, 1: 1.0}
        """
        if not self.coeffs:
            return ZxPoly({})

        result = ZxPoly({0: 0.0})
        # Pre-compute powers of a
        a_powers: list[float] = [1.0]
        for _ in range(self.degree()):
            a_powers.append(a_powers[-1] * a)

        for deg, coeff in self.coeffs.items():
            # Expand coeff * (x + a)ᵈᵉᵍ using binomial theorem
            for k in range(deg + 1):
                binom = self._binomial(deg, k)
                term_coeff = coeff * binom * a_powers[deg - k]
                if term_coeff != 0:
                    # Add to existing coefficient for xᵏ
                    existing = result.coeffs.get(k, 0.0)
                    result.coeffs[k] = existing + term_coeff

        # Clean up zero coefficients
        result.coeffs = {k: v for k, v in result.coeffs.items() if v != 0}
        return result

    @staticmethod
    def _binomial(n: int, k: int) -> int:
        """Compute binomial coefficient C(n, k).

        Returns:
        -------
        int
            Binomial coefficient.
        """
        if k < 0 or k > n:
            return 0
        if k in {0, n}:
            return 1
        # Use multiplicative formula
        result = 1
        for i in range(1, k + 1):
            result = result * (n - k + i) // i
        return result

    def is_zero(self) -> bool:
        """Check if polynomial is identically zero.

        Returns:
        -------
        bool
            True if polynomial has no non-zero coefficients.
        """
        return len(self.coeffs) == 0

    def is_constant(self) -> bool:
        """Check if polynomial is constant (degree ≤ 0).

        Returns:
        -------
        bool
            True if polynomial has no terms or only constant term.
        """
        if self.is_zero():
            return True
        return all(deg == 0 for deg in self.coeffs)

    def is_quadratic(self) -> bool:
        """Check if polynomial is at most quadratic (degree ≤ 2).

        Returns:
        -------
        bool
            True if polynomial degree ≤ 2 (Gaussian operation).

        Notes:
        -----
        In CV ZX calculus [3], quadratic phase functions correspond
        to Gaussian operations. Higher-degree polynomials indicate
        non-Gaussian resources [2].
        """
        return self.degree() <= 2  # noqa: PLR2004

    def __repr__(self) -> str:
        """Return string representation of the polynomial.

        Returns:
        -------
        str
            Human-readable representation like "1.0 + 2.0·x + 3.0·x²"
        """
        if self.is_zero():
            return "0"

        terms = []
        for deg in sorted(self.coeffs.keys()):
            coeff = self.coeffs[deg]
            if deg == 0:
                terms.append(f"{coeff}")
            elif deg == 1:
                terms.append(f"{coeff}·x")
            else:
                terms.append(f"{coeff}·x^{deg}")
        return " + ".join(terms)

    def __eq__(self, other: object) -> bool:
        """Check equality with another polynomial.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns:
        -------
        bool
            True if other is ZxPoly with identical coefficients.
        """
        if not isinstance(other, ZxPoly):
            return False
        # Compare after stripping zeros
        return self.coeffs == other.coeffs


class Diagram(ABC):
    """Abstract base class for all CV ZX diagrams.

    A diagram is an open, directed graph with labeled nodes.
    Inputs are open edges pointing inward. Outputs are open edges
    pointing outward. Diagrams are read from right to left so that
    input-output relations match bra-ket notation.

    Diagram operations include:
        - Parallelisation (tensor product)
        - Composition (sequential connection)
        - Contraction (partial trace over connected modes)
    """

    @abstractmethod
    def tensor(self, other: "Diagram") -> "Diagram":
        """Parallelisation: tensor product of two diagrams.

        Parameters
        ----------
        other : Diagram
            Diagram to place in parallel (vertically).

        Returns:
        -------
        Diagram
            New diagram representing self ⊗ other.
        """

    @abstractmethod
    def compose(self, other: "Diagram") -> "Diagram":
        """Composition: sequential connection of diagrams.

        Connects outputs of self to inputs of other.
        Requires number of outputs in self equals number of inputs in other.

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self (other ∘ self).

        Returns:
        -------
        Diagram
            New diagram representing other ∘ self.
        """

    @abstractmethod
    def conjugate(self) -> "Diagram":
        """Return the conjugate diagram.

        Conjugation inverts all arrow directions and multiplies all
        phase functions by -1. For Fourier diagrams, conjugation
        interchanges Fourier and inverse Fourier.

        Returns:
        -------
        Diagram
            New diagram representing D†.
        """

    @abstractmethod
    def is_proper(self) -> bool:
        """Check if diagram is a proper diagram generator.

        Proper diagrams are exactly the generators:
            - q-spider
            - p-spider
            - swap
            - fourier
            - inverse fourier
            - squared fourier

        Returns:
        -------
        bool
            True if diagram is one of the basic generators.
        """

    @property
    @abstractmethod
    def num_inputs(self) -> int:
        """Number of input wires (open edges pointing inward).

        Returns:
        -------
        int
            Number of input ports.
        """

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        """Number of output wires (open edges pointing outward).

        Returns:
        -------
        int
            Number of output ports.
        """


@dataclass
class ProperDiagram(Diagram):
    """Base class for proper diagram generators.

    Proper diagrams are atomic generators that cannot be decomposed
    into smaller diagrams. They include:
        - QSpider: position-basis spider with polynomial phase
        - PSpider: momentum-basis spider with polynomial phase
        - Swap: swaps two modes
        - Fourier: Fourier transform between q and p bases
        - FourierInv: inverse Fourier transform
        - Fourier2: squared Fourier transform (π/2 rotation)

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    _num_inputs: int
    _num_outputs: int

    def tensor(self, other: Diagram) -> Diagram:
        """Parallelize with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to place in parallel.

        Returns:
        -------
        TensorDiagram
            Tensor product of self and other.
        """
        if isinstance(other, TensorDiagram):
            diagrams = list(other.diagrams)
            return TensorDiagram([self, *diagrams])
        return TensorDiagram([self, other])

    def compose(self, other: Diagram) -> Diagram:
        """Compose with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self.

        Returns:
        -------
        CompositionDiagram
            Composition other ∘ self.
        """
        if isinstance(other, CompositionDiagram):
            diagrams = list(other.diagrams)
            return CompositionDiagram([*diagrams, self])
        return CompositionDiagram([other, self])

    def is_proper(self) -> bool:
        """Proper diagrams are always proper by definition.

        Returns:
        -------
        bool
            Always True for ProperDiagram instances.
        """
        return True

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

    def __repr__(self) -> str:
        """Return string representation of the proper diagram.

        Returns:
        -------
        str
            String representation of the diagram.
        """
        return f"{self.__class__.__name__}(inputs={self._num_inputs}, outputs={self._num_outputs})"


@dataclass
class ContractedDiagram(Diagram):
    """Diagram resulting from contracting (tracing) outputs to inputs in both directions.

    This is the output of the contraction rule apply to a tensor diagram of two diagrams D1 and D2
    from [3] Eq. (51):
        ∫∫ ds̄ dȳ ⟨s_i| D1 |s_j⟩ ⊗ q⟨s_j| D2 |s_i⟩

    The connections are:
        - (I1, I2): outputs I1 of first diagram connect to inputs I2 of second diagram (forward)
        - (J1, J2): outputs J2 of second diagram connect to inputs J1 of first diagram (feedback)

    After connection, the integral over the connected variables is implicit in the
    diagrammatic language. Only unconnected wires remain as external inputs/outputs.

    Attributes:
    ----------
    diagrams : list[Diagram]
        List containing [first, second] (D1, D2)
    first : Diagram
        First diagram (D1)
    second : Diagram
        Second diagram (D2)
    I1 : Sequence[int]
        Indices of outputs from first diagram that connect to second diagram
    I2 : Sequence[int]
        Indices of inputs from second diagram that receive connections from first diagram
    J1 : Sequence[int]
        Indices of inputs from first diagram that receive connections from second diagram
    J2 : Sequence[int]
        Indices of outputs from second diagram that connect to first diagram

    Notes:
    -----
    The lengths must satisfy: |I1| = |I2| and |J1| = |J2|
    The connection is made in order: I1[0] connects to I2[0], I1[1] to I2[1], etc.

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Definition 11, Eq. (51)
    """

    diagrams: list[Diagram]
    I1: Sequence[int]
    I2: Sequence[int]
    J1: Sequence[int]
    J2: Sequence[int]

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0917
        self,
        first: Diagram,
        second: Diagram,
        I1: Sequence[int],  # noqa: N803
        I2: Sequence[int],  # noqa: N803
        J1: Sequence[int],  # noqa: N803
        J2: Sequence[int],  # noqa: N803
    ) -> None:
        """Initialize ContractedDiagram with two diagrams and connection indices.

        Parameters:
        ----------
        first : Diagram
            First diagram (D1)
        second : Diagram
            Second diagram (D2)
        I1 : Sequence[int]
            Indices of outputs from first diagram that connect to second diagram
        I2 : Sequence[int]
            Indices of inputs from second diagram that receive connections from first diagram
        J1 : Sequence[int]
            Indices of inputs from first diagram that receive connections from second diagram
        J2 : Sequence[int]
            Indices of outputs from second diagram that connect to first diagram

        Raises:
        ------
        ValueError: If I1, J1, I2, or J2 contains duplicate indices
                    If |I1| != |I2| or |J1| != |J2|
                    If I1 indices are out of range for first diagram outputs
                    If I2 indices are out of range for second diagram inputs
                    If J1 indices are out of range for first diagram inputs
                    If J2 indices are out of range for second diagram outputs
        """
        self.diagrams = [first, second]
        self.first = first
        self.second = second
        self.I1 = I1
        self.I2 = I2
        self.J1 = J1
        self.J2 = J2

        # Check uniqueness of indices of the first diagram
        if len(set(I1)) != len(I1):
            msg = f"Output wires of the first diagram {I1} contain duplicate indices"
            raise ValueError(msg)
        if len(set(J1)) != len(J1):
            msg = f"Input wires of the first diagram {J1} contain duplicate indices"
            raise ValueError(msg)

        # Check uniqueness of indices of the second diagram
        if len(set(I2)) != len(I2):
            msg = f"Output wires of the second diagram {I2} contain duplicate indices"
            raise ValueError(msg)
        if len(set(J2)) != len(J2):
            msg = f"Input wires of the second diagram {J2} contain duplicate indices"
            raise ValueError(msg)

        # Validate lengths match
        if len(I1) != len(I2):
            msg = f"I1 length ({len(I1)}) must equal I2 length ({len(I2)})"
            raise ValueError(msg)
        if len(J1) != len(J2):
            msg_0 = f"J1 length ({len(J1)}) must equal J2 length ({len(J2)})"
            raise ValueError(msg_0)

        # Validate I1 indices are within first's outputs
        for idx in self.I1:
            if idx < 0 or idx >= self.first.num_outputs:
                msg_1 = f"I1 index {idx} out of range for first diagram outputs [0, {self.first.num_outputs})"
                raise ValueError(msg_1)

        # Validate I2 indices are within second's inputs
        for idx in self.I2:
            if idx < 0 or idx >= self.second.num_inputs:
                msg_2 = f"I2 index {idx} out of range for second diagram inputs [0, {self.second.num_inputs})"
                raise ValueError(msg_2)

        # Validate J1 indices are within first's inputs
        for idx in self.J1:
            if idx < 0 or idx >= self.first.num_inputs:
                msg_3 = f"J1 index {idx} out of range for first diagram inputs [0, {self.first.num_inputs})"
                raise ValueError(msg_3)

        # Validate J2 indices are within second's outputs
        for idx in self.J2:
            if idx < 0 or idx >= self.second.num_outputs:
                msg_4 = f"J2 index {idx} out of range for second diagram outputs [0, {self.second.num_outputs})"
                raise ValueError(msg_4)

        # Determine which inputs/outputs remain external
        # Inputs: all inputs from first diagram EXCEPT those in J1
        #         plus all inputs from second diagram EXCEPT those in I2
        self._kept_first_inputs = [i for i in range(self.first.num_inputs) if i not in self.J1]
        self._kept_second_inputs = [i for i in range(self.second.num_inputs) if i not in self.I2]

        # Outputs: all outputs from first diagram EXCEPT those in I1
        #          plus all outputs from second diagram EXCEPT those in J2
        self._kept_first_outputs = [j for j in range(self.first.num_outputs) if j not in self.I1]
        self._kept_second_outputs = [j for j in range(self.second.num_outputs) if j not in self.J2]

        # External interface
        self._num_inputs = len(self._kept_first_inputs) + len(self._kept_second_inputs)
        self._num_outputs = len(self._kept_first_outputs) + len(self._kept_second_outputs)

    def tensor(self, other: Diagram) -> Diagram:
        """Associative tensor product.

        Parameters
        ----------
        other : Diagram
            Diagram to place in parallel with self.

        Returns:
        -------
        TensorDiagram
            New tensor diagram with self.diagrams + [other].
        """
        if isinstance(other, TensorDiagram):
            diagrams = list(other.diagrams)
            return TensorDiagram([self, *diagrams])
        return TensorDiagram([self, other])

    def compose(self, other: Diagram) -> Diagram:
        """Compose tensor diagram with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self.

        Returns:
        -------
        CompositionDiagram
            Composition other ∘ self.
        """
        if isinstance(other, CompositionDiagram):
            diagrams = list(other.diagrams)
            return CompositionDiagram([*diagrams, self])
        return CompositionDiagram([other, self])

    def conjugate(self) -> Diagram:
        """Conjugate reverses order and swaps connection sets.

        (trace_{I1,I2,J1,J2}(D2 ∘ D1))† = trace_{J1,J2,I1,I2}(D1† ∘ D2†)

        Returns:
        -------
        ContractedDiagram
            New contracted diagram with order reversed and connection sets swapped.
        """
        return ContractedDiagram(
            first=self.second.conjugate(),
            second=self.first.conjugate(),
            I1=self.J1,  # J1 becomes I1 in the conjugated diagram
            I2=self.J2,  # J2 becomes I2
            J1=self.I1,  # I1 becomes J1
            J2=self.I2,  # I2 becomes J2
        )

    def is_proper(self) -> bool:
        """Tensor of proper diagrams is not a basic generator.

        Returns:
        -------
        bool
            Always False for composite diagrams.
        """
        return False

    @property
    def num_inputs(self) -> int:
        """Number of external input wires."""
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of external output wires."""
        return self._num_outputs

    @property
    def kept_first_inputs(self) -> list:
        """Number of remaining external input wires of the first diagram."""
        return self._kept_first_inputs

    @property
    def kept_first_outputs(self) -> list:
        """Number of remaining external output wires of the first diagram."""
        return self._kept_first_outputs

    @property
    def kept_second_inputs(self) -> list:
        """Number of remaining external input wires of the second diagram."""
        return self._kept_second_inputs

    @property
    def kept_second_outputs(self) -> list:
        """Number of remaining external output wires of the second diagram."""
        return self._kept_second_outputs

    def __repr__(self) -> str:
        """Return string representation of the contracted diagram."""
        return f"Contract({self.first}, {self.second}, I1={self.I1}, I2={self.I2}, J1={self.J1}, J2={self.J2})"


@dataclass
class TensorDiagram(Diagram):
    """Tensor product (parallelization) of multiple diagrams.

    Places diagrams in parallel, typically drawn vertically stacked.
    The number of inputs/outputs is the sum of the components' counts.

    Example:
        For diagrams [D1, D2, D3] with respective inputs (2, 1, 3)
        and outputs (1, 3, 2), the tensor has inputs = 6, outputs = 6.

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Definition 9
    """

    diagrams: Sequence[Diagram]

    def __post_init__(self) -> None:
        """Initialise tensor diagram by summing input/output counts."""
        self._num_inputs = sum(d.num_inputs for d in self.diagrams)
        self._num_outputs = sum(d.num_outputs for d in self.diagrams)

    def partial_trace(  # noqa: C901, PLR0912, PLR0915
        self,
        diagram_pairs: Sequence[tuple[int, Sequence[int], Sequence[int]]],
    ) -> None:
        """Perform partial trace by connecting outputs of some diagrams to inputs of others.

        Contract the diagram by connecting specified output wires of certain diagrams
        to specified input wires of other diagrams. The contracted diagrams must be
        consecutive in the tensor product.

        This implements diagram contraction as defined in [3] Eq. (51):
            ∫∫ ds̄ dȳ ⟨s_i| D1 |s_j⟩ ⊗ q⟨s_j| D2 |s_i⟩

        Parameters
        ----------
        diagram_pairs : Sequence[tuple[int, Sequence[int], Sequence[int]]]
            Each tuple contains:
                - diagram index of the first diagram
                - output wire indices from that diagram to contract
                - diagram index of the second diagram
                - input wire indices from that diagram to contract
            The output wires from the first diagram are connected to the input wires
            of the second diagram. And the input wires of the first diagram are
            connected to the output wires of the second diagram.

        Raises:
        ------
        ValueError: If diagram_pairs is empty.
                    If first diagram index is out of range.
                    If second diagram index is out of range.
                    If contracted diagrams are not consecutive (|first - second| != 1).
                    If input wire index is out of range for the second diagram.
                    If output wires contain duplicate indices.
                    If input wires contain duplicate indices.

        Notes:
        -----
        The contraction operation is only valid when the contracted diagrams
        are adjacent in the tensor product. This ensures the contraction corresponds
        to a valid partial trace over connected modes.

        After contraction, the two diagrams are replaced by a single diagram
        representing their composition with the contracted wires traced out.
        Only the unconsumed inputs and outputs wire from the first diagram and
        from the second diagram remain as external wires.

        References:
        ----------
        [3] Nagayoshi et al., CV ZX calculus, Definition 11 (contraction rule)
        """
        if not diagram_pairs:
            msg = "diagram_pairs cannot be empty"
            raise ValueError(msg)

        # Make a mutable copy of the diagrams list
        diagrams = list(self.diagrams)

        # Sort pairs by first diagram index (to process from higher to lower)
        sorted_pairs = sorted(diagram_pairs, key=operator.itemgetter(0), reverse=True)
        first_idx, f_output_wires, f_input_wires = sorted_pairs[0]
        second_idx, s_output_wires, s_input_wires = sorted_pairs[1]
        # Validate diagram indices in the tensor product
        if first_idx < 0 or first_idx >= len(diagrams):
            msg = f"First diagram index {first_idx} out of range [0, {len(diagrams) - 1}]"
            raise ValueError(msg)
        if second_idx < 0 or second_idx >= len(diagrams):
            msg = f"Second diagram index {second_idx} out of range [0, {len(diagrams) - 1}]"
            raise ValueError(msg)

        # Check that diagrams are consecutive
        if abs(first_idx - second_idx) != 1:
            msg = (
                f"Diagrams {first_idx} and {second_idx} are not consecutive. "
                f"Contraction requires consecutive diagrams."
            )
            raise ValueError(msg)

        first = diagrams[first_idx]
        second = diagrams[second_idx]

        # Validate wire indices for the first diagram
        for w in f_output_wires:
            if w < 0 or w >= first.num_outputs:
                msg = f"Output wire {w} out of range [0, {first.num_outputs - 1}] for diagram {first_idx}"
                raise ValueError(msg)
        for w in f_input_wires:
            if w < 0 or w >= first.num_inputs:
                msg = f"Input wire {w} out of range [0, {second.num_inputs - 1}] for diagram {first_idx}"
                raise ValueError(msg)
        # Validate wire indices for the second diagram
        for w in s_output_wires:
            if w < 0 or w >= second.num_outputs:
                msg = f"Input wire {w} out of range [0, {first.num_inputs - 1}] for diagram {second_idx}"
                raise ValueError(msg)
        for w in s_input_wires:
            if w < 0 or w >= second.num_inputs:
                msg = f"Output wire {w} out of range [0, {second.num_outputs - 1}] for diagram {second_idx}"
                raise ValueError(msg)

        # Validate lengths match
        if len(f_output_wires) != len(s_input_wires):
            msg_0 = f"I1 length ({len(f_output_wires)}) must equal I2 length ({len(s_input_wires)})"
            raise ValueError(msg_0)
        if len(f_input_wires) != len(s_output_wires):
            msg_1 = f"J1 length ({len(f_input_wires)}) must equal J2 length ({len(s_output_wires)})"
            raise ValueError(msg_1)

        # Check uniqueness of indices of the first diagram
        if len(set(f_output_wires)) != len(f_output_wires):
            msg = f"Output wires of the first diagram {f_output_wires} contain duplicate indices"
            raise ValueError(msg)
        if len(set(f_input_wires)) != len(f_input_wires):
            msg = f"Input wires of the first diagram {f_input_wires} contain duplicate indices"
            raise ValueError(msg)

        # Check uniqueness of indices of the second diagram
        if len(set(s_output_wires)) != len(s_output_wires):
            msg = f"Output wires of the second diagram {s_output_wires} contain duplicate indices"
            raise ValueError(msg)
        if len(set(s_input_wires)) != len(s_input_wires):
            msg = f"Input wires of the second diagram {s_input_wires} contain duplicate indices"
            raise ValueError(msg)

        # Replace the two diagrams with the contracted one
        diagrams[first_idx : second_idx + 1] = [
            ContractedDiagram(
                diagrams[first_idx], diagrams[second_idx], f_output_wires, f_input_wires, s_output_wires, s_input_wires
            )
        ]

        self.diagrams = diagrams

    def tensor(self, other: Diagram) -> Diagram:
        """Associative tensor product.

        Parameters
        ----------
        other : Diagram
            Diagram to place in parallel with self.

        Returns:
        -------
        TensorDiagram
            New tensor diagram with self.diagrams + [other].
        """
        return TensorDiagram([*list(self.diagrams), other])

    def compose(self, other: Diagram) -> Diagram:
        """Compose tensor diagram with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self.

        Returns:
        -------
        CompositionDiagram
            Composition other ∘ self.
        """
        return CompositionDiagram([other, self])

    def conjugate(self) -> Diagram:
        """Conjugate distributes over tensor product.

        (A ⊗ B ⊗ ...)† = A† ⊗ B† ⊗ ...

        Returns:
        -------
        TensorDiagram
            New tensor diagram with conjugated components.
        """
        return TensorDiagram([d.conjugate() for d in self.diagrams])

    def is_proper(self) -> bool:
        """Tensor of proper diagrams is not a basic generator.

        Returns:
        -------
        bool
            Always False for composite diagrams.
        """
        return False

    @property
    def num_inputs(self) -> int:
        """Number of input wires (sum of components' inputs).

        Returns:
        -------
        int
            Total number of input ports.
        """
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of output wires (sum of components' outputs).

        Returns:
        -------
        int
            Total number of output ports.
        """
        return self._num_outputs

    def __repr__(self) -> str:
        """Return string representation of the tensor diagram.

        Returns:
        -------
        str
            String representation showing all components.
        """
        return f"Tensor({self.diagrams})"


@dataclass
class CompositionDiagram(Diagram):
    """Composition (sequential connection) of multiple diagrams.

    Connects outputs of each diagram to inputs of the next.
    Requires for each consecutive pair: first.num_outputs == second.num_inputs.

    Example:
        For diagrams [D1, D2, D3], the composition is D3 ∘ D2 ∘ D1.
        Inputs come from D1, outputs from D3.

    Notes:
    -----
    Composition is associative, so we store a flat sequence.

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Definition 10
    """

    diagrams: Sequence[Diagram]

    def __post_init__(self) -> None:
        """Initialise composition diagram by validating consecutive composition.

        Raises:
        ------
        ValueError: If any consecutive diagrams have mismatched input/output counts.
        """
        for i in range(len(self.diagrams) - 1):
            first = self.diagrams[i]
            second = self.diagrams[i + 1]
            if first.num_outputs != second.num_inputs:
                msg = (
                    f"Cannot compose diagram {i} (outputs={first.num_outputs}) "
                    f"with diagram {i + 1} (inputs={second.num_inputs})"
                )
                raise ValueError(msg)
        self._num_inputs = self.diagrams[0].num_inputs
        self._num_outputs = self.diagrams[-1].num_outputs

    def tensor(self, other: Diagram) -> Diagram:
        """Parallelize composition with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to place in parallel.

        Returns:
        -------
        TensorDiagram
            Tensor product of self and other.
        """
        return TensorDiagram([self, other])

    def compose(self, other: Diagram) -> Diagram:
        """Associative composition: (other ∘ self).

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self.

        Returns:
        -------
        CompositionDiagram
            New composition diagram with self.diagrams + [other].
        """
        if isinstance(other, CompositionDiagram):
            diagrams = list(other.diagrams)
            return CompositionDiagram([*diagrams, *list(self.diagrams)])
        return CompositionDiagram([other, *list(self.diagrams)])

    def conjugate(self) -> Diagram:
        """Conjugate reverses composition order.

        (A ∘ B ∘ ...)† = ...† ∘ B† ∘ A†

        Returns:
        -------
        CompositionDiagram
            New composition with order reversed and components conjugated.
        """
        return CompositionDiagram([d.conjugate() for d in reversed(self.diagrams)])

    def is_proper(self) -> bool:
        """Composition of proper diagrams is not a basic generator.

        Returns:
        -------
        bool
            Always False for composite diagrams.
        """
        return False

    @property
    def num_inputs(self) -> int:
        """Number of input wires (from first diagram).

        Returns:
        -------
        int
            Number of input ports.
        """
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of output wires (from last diagram).

        Returns:
        -------
        int
            Number of output ports.
        """
        return self._num_outputs

    def __repr__(self) -> str:
        """Return string representation of the composition diagram.

        Returns:
        -------
        str
            String representation showing all components in order.
        """
        return f"Compose({self.diagrams})"


@dataclass
class ScalarDiagram(Diagram):
    """Closed diagram representing a scalar (no inputs/outputs).

    This represents the result of contracting all modes, which yields
    a complex scalar (potentially infinite in idealised CV theory).
    """

    _num_inputs: int = field(default=0, init=False)
    _num_outputs: int = field(default=0, init=False)

    def tensor(self, other: Diagram) -> Diagram:
        """Tensor product with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to place in parallel.

        Returns:
        -------
        Diagram
            Other diagram unchanged (since scalar ⊗ X = X).
        """
        return other

    def compose(self, other: Diagram) -> Diagram:
        """Composition with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self.

        Returns:
        -------
        Diagram
            Other diagram (since identity composition doesn't change).
        """
        return other

    def conjugate(self) -> Diagram:
        """Scalar diagram is self-conjugate.

        Returns:
        -------
        ScalarDiagram
            New scalar diagram instance.
        """
        return ScalarDiagram()

    def is_proper(self) -> bool:
        """Scalar is not a proper diagram generator.

        Returns:
        -------
        bool
            Always False.
        """
        return False

    @property
    def num_inputs(self) -> int:
        """Number of input wires (always 0).

        Returns:
        -------
        int
            0 for closed diagrams.
        """
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of output wires (always 0).

        Returns:
        -------
        int
            0 for closed diagrams.
        """
        return self._num_outputs

    def __repr__(self) -> str:
        """Return string representation of the scalar diagram.

        Returns:
        -------
        str
            "Scalar()"
        """
        return "Scalar()"


@dataclass
class QSpider(ProperDiagram):
    """q-spider: position-basis spider with polynomial phase.

    Represents the diagram:
        ┌─────┐
    ────┤ f(x)├────
        └─────┘

    where f(x) is a real polynomial phase function.

    Matrix elements:
        ∫ ds e^{i f(s)} |s...s⟩_qm ⟨s...s|_qn

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    phase: ZxPoly
    _num_inputs: int
    _num_outputs: int

    def conjugate(self) -> Diagram:
        """Conjugate: negate phase, keep q-spider type.

        Returns:
        -------
        QSpider
            New QSpider with negated phase.
        """
        return QSpider(phase=-self.phase, _num_inputs=self.num_inputs, _num_outputs=self.num_outputs)

    def __repr__(self) -> str:
        """Return string representation of the q-spider.

        Returns:
        -------
        str
            String showing phase function and number of wires.
        """
        if self.phase.is_zero():
            return f"QSpider(num_imputs={self.num_inputs}, num_outputs={self.num_outputs})"
        return f"QSpider(f(x)={self.phase}, num_imputs={self.num_inputs}, num_outputs={self.num_outputs})"


@dataclass
class PSpider(ProperDiagram):
    """p-spider: momentum-basis spider with polynomial phase.

    Represents the diagram:
        ┌─────┐
    ────┤ f(x)├────
        └─────┘

    where the spider type is p (momentum basis) and f(x) is a
    real polynomial phase function.

    Matrix elements:
        ∫ dt e^{-i f(t)} |t...t⟩_pm ⟨t...t|_pn

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    phase: ZxPoly
    _num_inputs: int
    _num_outputs: int

    def conjugate(self) -> Diagram:
        """Conjugate: negate phase, keep p-spider type.

        Returns:
        -------
        PSpider
            New PSpider with negated phase.
        """
        return PSpider(phase=-self.phase, _num_inputs=self.num_inputs, _num_outputs=self.num_outputs)

    def __repr__(self) -> str:
        """Return string representation of the p-spider.

        Returns:
        -------
        str
            String showing phase function and number of wires.
        """
        if self.phase.is_zero():
            return f"PSpider(num_imputs={self.num_inputs}, num_outputs={self.num_outputs})"
        return f"PSpider(f(x)={self.phase}, num_imputs={self.num_inputs}, num_outputs={self.num_outputs})"


@dataclass
class Swap(ProperDiagram):
    """Swap diagram: exchanges two modes.

    Represents the diagram:
        ┌───┐
    ────┤ X ├────
    ────┤   ├────
        └───┘

    Matrix elements:
        ∫ ds ds' |s', s⟩⟨s, s'|

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    _num_inputs: int = field(default=2, init=False)
    _num_outputs: int = field(default=2, init=False)

    def conjugate(self) -> Diagram:
        """Swap is self-conjugate.

        Returns:
        -------
        Swap
            New Swap instance.
        """
        return Swap()

    def __repr__(self) -> str:
        """Return string representation of the swap diagram.

        Returns:
        -------
        str
            "Swap()"
        """
        return "Swap()"


@dataclass
class Fourier(ProperDiagram):
    """Fourier transform diagram.

    Represents the diagram:
        ┌───┐
    ────┤ F ├────
        └───┘

    Transforms between position and momentum bases.

    Matrix elements:
        ∫ du |u⟩_p ⟨u|_q

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)

    def conjugate(self) -> Diagram:
        """Conjugate of Fourier is inverse Fourier.

        Returns:
        -------
        FourierInv
            Inverse Fourier diagram.
        """
        return FourierInv()

    def __repr__(self) -> str:
        """Return string representation of the Fourier diagram.

        Returns:
        -------
        str
            "Fourier()"
        """
        return "Fourier()"


@dataclass
class FourierInv(ProperDiagram):
    """Inverse Fourier transform diagram.

    Represents the diagram:
        ┌────┐
    ────┤ F† ├────
        └────┘

    Matrix elements:
        ∫ du |u⟩_q ⟨u|_p

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)

    def conjugate(self) -> Diagram:
        """Conjugate of inverse Fourier is Fourier.

        Returns:
        -------
        Fourier
            Fourier diagram.
        """
        return Fourier()

    def __repr__(self) -> str:
        """Return string representation of the inverse Fourier diagram.

        Returns:
        -------
        str
            "FourierInv()"
        """
        return "FourierInv()"


@dataclass
class Fourier2(ProperDiagram):
    """Squared Fourier transform diagram.

    Represents the diagram:
        ┌────┐
    ────┤ F² ├────
        └────┘

    This is equivalent to a π rotation in phase space.

    Matrix elements:
        ∫ ds |-s⟩_q ⟨s|_q

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)

    def conjugate(self) -> Diagram:
        """F² is self-conjugate.

        Returns:
        -------
        Fourier2
            New Fourier2 instance.
        """
        return Fourier2()

    def __repr__(self) -> str:
        """Return string representation of the squared Fourier diagram.

        Returns:
        -------
        str
            "Fourier2()"
        """
        return "Fourier2()"

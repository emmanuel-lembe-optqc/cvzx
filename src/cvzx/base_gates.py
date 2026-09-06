"""CV-ZX representation of usual gates."""

import operator
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import count

from sympy import Expr, Poly, S, symbols, sympify


class ZxPoly(Poly):
    """Real polynomial in one variable for CV ZX calculus phase functions.

    This class inherits from sympy.Poly and adds ZX-specific functionality
    while maintaining backward compatibility with the original ZxPoly API.

    The polynomial is stored as a sympy.Poly object internally, supporting both
    numeric and symbolic coefficients. Zero coefficients are omitted from the
    dictionary representation.

    Parameters
    ----------
    coeffs : dict[int, float | Expr]
        Dictionary mapping degree → coefficient. For numeric coefficients,
        returns Python floats; for symbolic coefficients, returns sympy expressions.

    Examples
    --------
    >>> p = ZxPoly({0: 1.0, 2: -0.5})  # 1 - 0.5·x²
    >>> q = ZxPoly({1: 2.0})            # 2·x
    >>> r = p + q                       # 1 + 2·x - 0.5·x²
    >>> r.degree()
    2
    >>> s = p * q                       # 2·x - x³
    >>> s.coeffs                        # Dict for compatibility
    {1: 2.0, 3: -1.0}

    >>> from sympy import symbols
    >>> a, b = symbols('a b')
    >>> p = ZxPoly({0: a, 1: b})        # a + b·x
    >>> q = ZxPoly({0: 1, 1: 2})        # 1 + 2·x
    >>> r = p + q                       # (a+1) + (b+2)·x
    >>> r.coeffs
    {0: a + 1, 1: b + 2}
    """

    _var = symbols("x", real=True)

    def __new__(cls, coeffs_or_poly: dict[int, float | int | Expr] | Poly | Expr | None = None, *args, **kwargs):  # ruff: ignore[missing-type-args, missing-type-kwargs, missing-return-type-special-method]
        """Create a new ZxPoly instance.

        This method intercepts instance creation to handle the special case
        where the user provides a coefficient dictionary instead of a sympy
        expression or Poly object.

        Parameters
        ----------
        coeffs_or_poly : dict[int, float | int | Expr] | Poly | Expr | None
            Either a coefficient dictionary mapping degree → coefficient,
            a sympy.Poly object, a sympy expression, or None.
            If None, creates the zero polynomial.
        *args, **kwargs
            Additional arguments passed to sympy.Poly constructor.

        Returns
        -------
        ZxPoly
            A new ZxPoly instance.

        Examples
        --------
        >>> # From coefficient dictionary
        >>> p = ZxPoly({0: 1.0, 2: -0.5})

        >>> # From sympy expression
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> p = ZxPoly(x**2 + 2*x + 1)

        >>> # From sympy.Poly
        >>> from sympy import Poly
        >>> q = ZxPoly(Poly(x**2 + 1, x))

        >>> # Zero polynomial
        >>> z = ZxPoly()
        """
        if isinstance(coeffs_or_poly, dict):
            # Build expression from coefficient dictionary
            var = kwargs.get("gen", cls._var)
            expr = S.Zero
            for degree, coeff in coeffs_or_poly.items():
                if coeff != 0:
                    expr += sympify(coeff) * var**degree
            return super().__new__(cls, expr, var)
        if coeffs_or_poly is None:
            return super().__new__(cls, 0, cls._var)
        return super().__new__(cls, coeffs_or_poly, *args, **kwargs)

    def __init__(
        self,
        coeffs_or_poly: dict[int, float | int | Expr] | Poly | Expr | None = None,
        *args,  # ruff: ignore[missing-type-args]
        **kwargs,  # ruff: ignore[missing-type-kwargs]
    ) -> None:
        """Initialize the ZxPoly instance.

        This method handles initialization for cases where the instance
        wasn't fully initialized in __new__ (e.g., when passing through
        to Poly.__init__).

        Parameters
        ----------
        coeffs_or_poly : dict[int, float | int | Expr] | Poly | Expr | None
            Either a coefficient dictionary, sympy.Poly, sympy expression, or None.
        *args, **kwargs
            Additional arguments passed to sympy.Poly constructor.

        Notes
        -----
        When initialized from a coefficient dictionary, the instance is
        already fully created in __new__, so this method does nothing.
        When initialized from other types, it delegates to Poly.__init__.
        """
        if not isinstance(coeffs_or_poly, dict) and coeffs_or_poly is not None:
            super().__init__(*args, **kwargs)

    @property
    def coeffs(self) -> dict[int, float | Expr]:
        """Coefficients as a dictionary, for backward compatibility.

        Returns a dictionary mapping degree → coefficient. Zero coefficients
        are omitted. Numeric coefficients are converted to Python floats
        for compatibility, while symbolic coefficients remain as sympy expressions.

        Returns
        -------
        dict[int, float | Expr]
            Dictionary mapping degree to coefficient. Returns empty dict
            for the zero polynomial.

        Examples
        --------
        >>> p = ZxPoly({0: 1.0, 1: 2.0, 2: 3.0})
        >>> p.coeffs
        {0: 1.0, 1: 2.0, 2: 3.0}

        >>> from sympy import symbols
        >>> a = symbols('a')
        >>> p = ZxPoly({0: a, 1: 2.0})
        >>> p.coeffs
        {0: a, 1: 2.0}
        """
        if self.is_zero:
            return {}

        coeff_dict = {}
        for monom, coeff in self.terms():
            degree = monom[0]  # For univariate
            # Convert numeric coefficients to Python floats for compatibility
            if coeff.is_number and not coeff.is_symbol:
                coeff_dict[degree] = float(coeff)
            else:
                coeff_dict[degree] = coeff
        return coeff_dict

    def __repr__(self) -> str:
        """Return a string representation of the polynomial.

        Returns a human-readable string in ZX calculus notation:
        - Terms are ordered by increasing degree
        - Uses '·' for multiplication
        - Uses '^' for exponents
        - Constant term is shown as just the coefficient
        - Linear term is shown as 'c·x'
        - Higher-degree terms are shown as 'c·x^n'

        Returns
        -------
        str
            String representation of the polynomial.

        Examples
        --------
        >>> p = ZxPoly({0: 1.0, 1: 2.0, 2: 3.0})
        >>> repr(p)
        '1.0 + 2.0·x + 3.0·x^2'

        >>> from sympy import symbols
        >>> a = symbols('a')
        >>> p = ZxPoly({0: a, 1: 2.0, 2: -0.5})
        >>> repr(p)
        'a + 2.0·x - 0.5·x^2'

        >>> z = ZxPoly({})
        >>> repr(z)
        '0'
        """
        if self.is_zero:
            return "0"

        coeffs = self.coeffs
        terms = []
        for deg in sorted(coeffs.keys()):
            coeff = coeffs[deg]
            if deg == 0:
                terms.append(f"{coeff}")
            elif deg == 1:
                terms.append(f"{coeff}·x")
            else:
                terms.append(f"{coeff}·x^{deg}")
        return " + ".join(terms)

    def __eq__(self, other: object) -> bool:
        """Check equality with another polynomial or Poly object.

        Two polynomials are considered equal if they have the same
        coefficients for all degrees. The comparison works with both
        ZxPoly instances and sympy.Poly instances.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        bool
            True if other is a ZxPoly or Poly with identical coefficients,
            False otherwise.
        """
        if isinstance(other, ZxPoly | Poly):
            return bool(super().__eq__(other))
        return False

    def __hash__(self) -> int:
        """Compute hash value for the polynomial.

        Returns a hash based on the polynomial representation, allowing
        ZxPoly instances to be used as dictionary keys.

        Returns
        -------
        int
            Hash value for the polynomial.
        """
        return int(super().__hash__())


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

    _id_counter = count(1)

    def __init__(self) -> None:
        self._id = next(Diagram._id_counter)

    @abstractmethod
    def tensor(self, other: "Diagram") -> "Diagram":
        """Parallelisation: tensor product of two diagrams.

        Parameters
        ----------
        other : Diagram
            Diagram to place in parallel (vertically).

        Returns
        -------
        Diagram
            New diagram representing self ⊗ other.
        """

    @abstractmethod
    def compose(self, other: "Diagram", connectivity: dict | None = None) -> "Diagram":
        """Composition: sequential connection of diagrams.

        Connects outputs of self to inputs of other.
        Requires number of outputs in self equals number of inputs in other.

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self (other ∘ self).
        connectivity: dict
            Dictionary indicating how the diagrams are connected.

        Returns
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

        Returns
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

        Returns
        -------
        bool
            True if diagram is one of the basic generators.
        """

    @property
    def id(self) -> int:
        """ID of the diagram.

        Returns
        -------
            int
                ID of the diagram.
        """
        return self._id

    @property
    @abstractmethod
    def num_inputs(self) -> int:
        """Number of input wires (open edges pointing inward).

        Returns
        -------
        int
            Number of input ports.
        """

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        """Number of output wires (open edges pointing outward).

        Returns
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

    References
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

        Returns
        -------
        TensorDiagram
            Tensor product of self and other.
        """
        if isinstance(other, TensorDiagram):
            diagrams = list(other.diagrams)
            return TensorDiagram([self, *diagrams])
        return TensorDiagram([self, other])

    def compose(self, other: Diagram, connectivity: dict | None = None) -> Diagram:
        """Compose with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self.
        connectivity: dict
            Dictionary indicating how the diagrams are connected.

        Returns
        -------
        CompositionDiagram
            Composition other ∘ self.

        Raises
        ------
        ValueError
            If the connectivity dictionary coherent with the inputs of
            self and/or the outputs of the input diagram.
        """
        if connectivity is None:
            connectivity = {k: k for k in range(self.num_inputs)}
        # Check the validity of the connectivity
        keys = list(connectivity.keys())
        keys.sort()
        if keys != list(range(self.num_inputs)):
            msg = "The keys of the connectivity dictionary do not correspond the input indices of the current diagram."
            raise ValueError(msg)
        values = list(connectivity.values())
        values.sort()
        if values != list(range(other.num_outputs)):
            msg = (
                "The values of the connectivity dictionary do not correspond the output indices of the input diagram."
            )
            raise ValueError(msg)
        if isinstance(other, CompositionDiagram):
            diagrams = list(other.diagrams)
            old_connectivity = dict(other.connectivity)
            old_connectivity[len(diagrams) - 1] = connectivity
            return CompositionDiagram([*diagrams, self], old_connectivity)
        return CompositionDiagram([other, self], {0: connectivity})

    def is_proper(self) -> bool:
        """Proper diagrams are always proper by definition.

        Returns
        -------
        bool
            Always True for ProperDiagram instances.
        """
        return True

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

    def __post_init__(self) -> None:  # ruff: ignore[undocumented-magic-method]
        super().__init__()

    def __repr__(self) -> str:
        """Return string representation of the proper diagram.

        Returns
        -------
        str
            String representation of the diagram.
        """
        return f"{self.__class__.__name__}(inputs={self._num_inputs}, outputs={self._num_outputs})"


@dataclass
class ContractedDiagram(Diagram):
    r"""Diagram resulting from contracting (tracing) outputs to inputs in both directions.

    This is the output of the contraction rule apply to a tensor diagram of two diagrams D1 and D2
    from [3] Eq. (51)::

        ∫∫ ds̄ dȳ ⟨s_i\| D1 \|s_j⟩ ⊗ q⟨s_j\| D2 \|s_i⟩

    The connections are:

    - (I1, I2): outputs I1 of first diagram connect to inputs I2 of second diagram (forward)
    - (J1, J2): outputs J2 of second diagram connect to inputs J1 of first diagram (feedback)

    After connection, the integral over the connected variables is implicit in the
    diagrammatic language. Only unconnected wires remain as external inputs/outputs.

    Attributes
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

    Notes
    -----
    The lengths must satisfy: \|I1\| = \|I2\| and \|J1\| = \|J2\|
    The connection is made in order: I1[0] connects to I2[0], I1[1] to I2[1], etc.

    References
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Definition 11, Eq. (51)
    """

    diagrams: list[Diagram]
    I1: Sequence[int]
    I2: Sequence[int]
    J1: Sequence[int]
    J2: Sequence[int]

    def __init__(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments, too-many-positional-arguments]
        self,
        first: Diagram,
        second: Diagram,
        I1: Sequence[int],  # ruff: ignore[invalid-argument-name]
        I2: Sequence[int],  # ruff: ignore[invalid-argument-name]
        J1: Sequence[int],  # ruff: ignore[invalid-argument-name]
        J2: Sequence[int],  # ruff: ignore[invalid-argument-name]
    ) -> None:
        r"""Initialize ContractedDiagram with two diagrams and connection indices.

        Parameters
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

        Raises
        ------
        ValueError
            If I1, J1, I2, or J2 contains duplicate indices.
            If \|I1\| != \|I2\| or \|J1\| != \|J2\|.
            If \|I1\| = \|I2\| = \|J1\| = \|J2\| = 0.
            If I1 indices are out of range for first diagram outputs.
            If I2 indices are out of range for second diagram inputs.
            If J1 indices are out of range for first diagram inputs.
            If J2 indices are out of range for second diagram outputs.
        """
        super().__init__()
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

        # Validate I1, I2, J1 and J2 are not all empty
        if not I1 and not I2 and not J1 and not J2:
            msg = "A contraction diagam must have at least one contraction link."
            raise ValueError(msg)

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

        Returns
        -------
        TensorDiagram
            New tensor diagram with self.diagrams + [other].
        """
        if isinstance(other, TensorDiagram):
            diagrams = list(other.diagrams)
            return TensorDiagram([self, *diagrams])
        return TensorDiagram([self, other])

    def compose(self, other: Diagram, connectivity: dict | None = None) -> Diagram:
        """Compose with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self.
        connectivity: dict
            Dictionary indicating how the diagrams are connected.

        Returns
        -------
        CompositionDiagram
            Composition other ∘ self.

        Raises
        ------
        ValueError
            If the connectivity dictionary coherent with the inputs of
            self and/or the outputs of the input diagram.
        """
        if connectivity is None:
            connectivity = {k: k for k in range(self.num_inputs)}
        # Check the validity of the connectivity
        keys = list(connectivity.keys())
        keys.sort()
        if keys != list(range(self.num_inputs)):
            msg = "The keys of the connectivity dictionary do not correspond the input indices of the current diagram."
            raise ValueError(msg)
        values = list(connectivity.values())
        values.sort()
        if values != list(range(other.num_outputs)):
            msg = (
                "The values of the connectivity dictionary do not correspond the output indices of the input diagram."
            )
            raise ValueError(msg)
        if isinstance(other, CompositionDiagram):
            diagrams = list(other.diagrams)
            old_connectivity = other.connectivity
            old_connectivity[len(diagrams) - 1] = connectivity
            return CompositionDiagram([*diagrams, self], old_connectivity)
        return CompositionDiagram([other, self], {0: connectivity})

    def conjugate(self) -> Diagram:
        """Conjugate reverses order and swaps connection sets.

        (trace_{I1,I2,J1,J2}(D2 ∘ D1))† = trace_{J1,J2,I1,I2}(D1† ∘ D2†)

        Returns
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

        Returns
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
        """Return string representation of the contracted diagram.

        Returns
        -------
        str
            String showing the two diagrams and their connections.
        """
        return f"Contract({self.first}, {self.second}, I1={self.I1}, I2={self.I2}, J1={self.J1}, J2={self.J2})"


@dataclass
class TensorDiagram(Diagram):
    """Tensor product (parallelization) of multiple diagrams.

    Places diagrams in parallel, typically drawn vertically stacked.
    The number of inputs/outputs is the sum of the components' counts.

    Example:
        For diagrams [D1, D2, D3] with respective inputs (2, 1, 3)
        and outputs (1, 3, 2), the tensor has inputs = 6, outputs = 6.

    References
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Definition 9
    """

    diagrams: Sequence[Diagram]

    def partial_trace(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements]
        self,
        diagram_pairs: Sequence[tuple[int, Sequence[int], Sequence[int]]],
    ) -> None:
        r"""Perform partial trace by connecting outputs of some diagrams to inputs of others.

        Contract the diagram by connecting specified output wires of certain diagrams
        to specified input wires of other diagrams. The contracted diagrams must be
        consecutive in the tensor product.

        This implements diagram contraction as defined in [3] Eq. (51)::

            ∫∫ ds̄ dȳ ⟨s_i\| D1 \|s_j⟩ ⊗ q⟨s_j\| D2 \|s_i⟩

        Parameters
        ----------
        diagram_pairs : Sequence[tuple[int, Sequence[int], Sequence[int]]]
            Each tuple contains:

            - diagram index of the first/second diagram
            - output wire indices from that diagram to contract
            - input wire indices from that diagram to contract

            The output wires from the first diagram are connected to the input wires
            of the second diagram. And the input wires of the first diagram are
            connected to the output wires of the second diagram.

        Raises
        ------
        ValueError
            If diagram_pairs is empty.
            If first diagram index is out of range.
            If second diagram index is out of range.
            If contracted diagrams are not consecutive (\|first - second\| != 1).
            If input wire index is out of range for the second diagram.
            If output wires contain duplicate indices.
            If input wires contain duplicate indices.

        Notes
        -----
        The contraction operation is only valid when the contracted diagrams
        are adjacent in the tensor product. This restriction is to draw the
        resulting contracted diagram easily. But in theory the partial trace
        can be applied to any two diagrams of a tensor diagram.

        After contraction, the two diagrams are replaced by a single diagram
        representing their composition with the contracted wires traced out.
        Only the unconsumed inputs and outputs wire from the first diagram and
        from the second diagram remain as external wires.

        References
        ----------
        [3] Nagayoshi et al., CV ZX calculus, Definition 11
        """
        if not diagram_pairs:
            msg = "diagram_pairs cannot be empty"
            raise ValueError(msg)

        # Make a mutable copy of the diagrams list
        diagrams = list(self.diagrams)

        # Sort pairs by first diagram index (to process from higher to lower)
        sorted_pairs = sorted(diagram_pairs, key=operator.itemgetter(0))
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
                diagrams[first_idx], diagrams[second_idx], f_output_wires, s_input_wires, f_input_wires, s_output_wires
            )
        ]

        self.diagrams = diagrams

    def tensor(self, other: Diagram) -> Diagram:
        """Associative tensor product.

        Parameters
        ----------
        other : Diagram
            Diagram to place in parallel with self.

        Returns
        -------
        TensorDiagram
            New tensor diagram with self.diagrams + [other].
        """
        if isinstance(other, TensorDiagram):
            diagrams = list(other.diagrams)
            return TensorDiagram([*list(self.diagrams), *diagrams])
        return TensorDiagram([*list(self.diagrams), other])

    def compose(self, other: Diagram, connectivity: dict | None = None) -> Diagram:
        """Compose with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self.
        connectivity: dict
            Dictionary indicating how the diagrams are connected.

        Returns
        -------
        CompositionDiagram
            Composition other ∘ self.

        Raises
        ------
        ValueError
            If the connectivity dictionary coherent with the inputs of
            self and/or the outputs of the input diagram.
        """
        if connectivity is None:
            connectivity = {k: k for k in range(self.num_inputs)}
        # Check the validity of the connectivity
        keys = list(connectivity.keys())
        keys.sort()
        if keys != list(range(self.num_inputs)):
            msg = "The keys of the connectivity dictionary do not correspond the input indices of the current diagram."
            raise ValueError(msg)
        values = list(connectivity.values())
        values.sort()
        if values != list(range(other.num_outputs)):
            msg = (
                "The values of the connectivity dictionary do not correspond the output indices of the input diagram."
            )
            raise ValueError(msg)
        if isinstance(other, CompositionDiagram):
            diagrams = list(other.diagrams)
            old_connectivity = other.connectivity
            old_connectivity[len(diagrams) - 1] = connectivity
            return CompositionDiagram([*diagrams, self], old_connectivity)
        return CompositionDiagram([other, self], {0: connectivity})

    def conjugate(self) -> Diagram:
        """Conjugate distributes over tensor product.

        (A ⊗ B ⊗ ...)† = A† ⊗ B† ⊗ ...

        Returns
        -------
        TensorDiagram
            New tensor diagram with conjugated components.
        """
        return TensorDiagram([d.conjugate() for d in self.diagrams])

    def is_proper(self) -> bool:
        """Tensor of proper diagrams is not a basic generator.

        Returns
        -------
        bool
            Always False for composite diagrams.
        """
        return False

    @property
    def num_inputs(self) -> int:
        """Number of input wires (sum of components' inputs).

        Returns
        -------
        int
            Total number of input ports.
        """
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of output wires (sum of components' outputs).

        Returns
        -------
        int
            Total number of output ports.
        """
        return self._num_outputs

    def __post_init__(self) -> None:
        """Initialise tensor diagram by summing input/output counts."""
        super().__init__()
        flattened: list = []
        for sub_diagram in self.diagrams:
            # Recursively flatten the sub-diagram first
            flattened_sub = self._flatten_tensor(sub_diagram)
            if isinstance(flattened_sub, TensorDiagram):
                flattened.extend(flattened_sub.diagrams)
            else:
                flattened.append(flattened_sub)

        if not flattened:
            self.diagrams = []
            self._num_inputs = 0
            self._num_outputs = 0
            return

        if len(flattened) == 1:
            self._single_element = flattened[0]
            self.diagrams = [flattened[0]]
            self._num_inputs = flattened[0].num_inputs
            self._num_outputs = flattened[0].num_outputs
            return

        self.diagrams = flattened
        self._num_inputs = sum(d.num_inputs for d in self.diagrams)
        self._num_outputs = sum(d.num_outputs for d in self.diagrams)

    def _flatten_tensor(self, diagram: Diagram) -> Diagram:
        """Recursively flatten a diagram, specifically flattening TensorDiagrams.

        This only flattens TensorDiagrams, not CompositionDiagrams.

        Parameters
        ----------
        diagram : Diagram
            The diagram to modify.

        Returns
        -------
        Diagram
        """
        if isinstance(diagram, TensorDiagram):
            flattened: list[Diagram] = []
            for sub in diagram.diagrams:
                flattened_sub = self._flatten_tensor(sub)
                if isinstance(flattened_sub, TensorDiagram):
                    flattened.extend(flattened_sub.diagrams)
                else:
                    flattened.append(flattened_sub)
            if len(flattened) == 1:
                return flattened[0]
            return TensorDiagram(flattened)

        if isinstance(diagram, CompositionDiagram):
            # Don't flatten compositions, but recursively flatten tensors inside
            flattened_diagrams = [self._flatten_tensor(sub) for sub in diagram.diagrams]
            return CompositionDiagram(flattened_diagrams, diagram.connectivity)

        if isinstance(diagram, ContractedDiagram):
            first = self._flatten_tensor(diagram.first)
            second = self._flatten_tensor(diagram.second)
            if first is not diagram.first or second is not diagram.second:
                return ContractedDiagram(
                    first=first,
                    second=second,
                    I1=diagram.I1,
                    I2=diagram.I2,
                    J1=diagram.J1,
                    J2=diagram.J2,
                )

        return diagram

    def __repr__(self) -> str:
        """Return string representation of the tensor diagram.

        Returns
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

    Notes
    -----
    Composition is associative, so we store a flat sequence.

    References
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Definition 10
    """

    diagrams: Sequence[Diagram]
    connectivity: dict[int, dict[int, int]] = field(default_factory=dict)

    def tensor(self, other: Diagram) -> Diagram:
        """Parallelize composition with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to place in parallel.

        Returns
        -------
        TensorDiagram
            Tensor product of self and other.
        """
        if isinstance(other, TensorDiagram):
            diagrams = list(other.diagrams)
            return TensorDiagram([self, *diagrams])
        return TensorDiagram([self, other])

    def compose(self, other: Diagram, connectivity: dict | None = None) -> Diagram:
        """Compose with another diagram.

        Parameters
        ----------
        other : Diagram
            Diagram to apply after self.
        connectivity: dict
            Dictionary indicating how the diagrams are connected.

        Returns
        -------
        CompositionDiagram
            Composition other ∘ self.

        Raises
        ------
        ValueError
            If the connectivity dictionary coherent with the inputs of
            self and/or the outputs of the input diagram.
        """
        if connectivity is None:
            connectivity = {k: k for k in range(self.num_inputs)}
        # Check the validity of the connectivity
        keys = list(connectivity.keys())
        keys.sort()
        if keys != list(range(self.num_inputs)):
            msg = "The keys of the connectivity dictionary do not correspond the input indices of the current diagram."
            raise ValueError(msg)
        values = list(connectivity.values())
        values.sort()
        if values != list(range(other.num_outputs)):
            msg = (
                "The values of the connectivity dictionary do not correspond the output indices of the input diagram."
            )
            raise ValueError(msg)
        # Composition
        diagrams = list(self.diagrams)
        if isinstance(other, CompositionDiagram):
            new_diagrams = list(other.diagrams)
            new_connectivity = deepcopy(other.connectivity)
            size = len(new_diagrams)
            new_connectivity[size - 1] = connectivity
            for key, value in self.connectivity.items():
                new_connectivity[size + key] = value
            return CompositionDiagram([*new_diagrams, *diagrams], new_connectivity)
        new_connectivity = {key + 1: value for key, value in self.connectivity.items()}
        new_connectivity[0] = connectivity
        return CompositionDiagram([other, *diagrams], new_connectivity)

    def conjugate(self) -> Diagram:
        """Conjugate reverses composition order.

        (A ∘ B ∘ ...)† = ...† ∘ B† ∘ A†

        Returns
        -------
        CompositionDiagram
            New composition with order reversed and components conjugated.
        """
        return CompositionDiagram([d.conjugate() for d in reversed(self.diagrams)])

    def is_proper(self) -> bool:
        """Composition of proper diagrams is not a basic generator.

        Returns
        -------
        bool
            Always False for composite diagrams.
        """
        return False

    @property
    def num_inputs(self) -> int:
        """Number of input wires (from first diagram).

        Returns
        -------
        int
            Number of input ports.
        """
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of output wires (from last diagram).

        Returns
        -------
        int
            Number of output ports.
        """
        return self._num_outputs

    def __post_init__(self) -> None:
        """Initialise composition diagram by validating consecutive composition.

        Raises
        ------
        ValueError
            If any consecutive diagrams have mismatched input/output counts.
            If the connectivity dictionary is coherent with the input/output
                of diagrams.
        """
        super().__init__()
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
        if not self.connectivity:
            for i in range(len(self.diagrams) - 1):
                self.connectivity[i] = {k: k for k in range(self.diagrams[i + 1].num_inputs)}
        # Check connectivity
        for i in range(len(self.diagrams) - 1):
            # Check keys
            keys = list(self.connectivity[i].keys())
            keys.sort()
            if keys != list(range(self.diagrams[i + 1].num_inputs)):
                msg = (
                    "The keys of the connectivity dictionary do not correspond "
                    f"the input indices of the sub_diagram {i + 1}."
                )
                raise ValueError(msg)
            # Check values
            values = list(self.connectivity[i].values())
            values.sort()
            if values != list(range(self.diagrams[i].num_outputs)):
                msg = (
                    "The values of the connectivity dictionary do not "
                    f"correspond the output indices of the sub_diagram {i}."
                )
                raise ValueError(msg)

    def __repr__(self) -> str:
        """Return string representation of the composition diagram.

        Returns
        -------
        str
            String representation showing all components in order.
        """
        return f"Compose({self.diagrams})"


@dataclass
class QSpider(ProperDiagram):
    r"""q-spider: position-basis spider with polynomial phase.

    Represents the diagram:
        ┌─────┐
    ────┤ f(x)├────
        └─────┘

    where f(x) is a real polynomial phase function.

    Matrix elements:
        ∫ ds e^{i f(s)} \|s...s⟩_qm ⟨s...s\|_qn

    References
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    phase: ZxPoly
    _num_inputs: int
    _num_outputs: int

    def conjugate(self) -> Diagram:
        """Conjugate: negate phase, keep q-spider type.

        Returns
        -------
        QSpider
            New QSpider with negated phase.
        """
        return QSpider(phase=-self.phase, _num_inputs=self.num_inputs, _num_outputs=self.num_outputs)

    def __repr__(self) -> str:
        """Return string representation of the q-spider.

        Returns
        -------
        str
            String showing phase function and number of wires.
        """
        if self.phase.is_zero:
            return f"QSpider(num_imputs={self.num_inputs}, num_outputs={self.num_outputs})"
        return f"QSpider(f(x)={self.phase}, num_imputs={self.num_inputs}, num_outputs={self.num_outputs})"


@dataclass
class PSpider(ProperDiagram):
    r"""p-spider: momentum-basis spider with polynomial phase.

    Represents the diagram:
        ┌─────┐
    ────┤ f(x)├────
        └─────┘

    where the spider type is p (momentum basis) and f(x) is a
    real polynomial phase function.

    Matrix elements:
        ∫ dt e^{-i f(t)} \|t...t⟩_pm ⟨t...t\|_pn

    References
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    phase: ZxPoly
    _num_inputs: int
    _num_outputs: int

    def conjugate(self) -> Diagram:
        """Conjugate: negate phase, keep p-spider type.

        Returns
        -------
        PSpider
            New PSpider with negated phase.
        """
        return PSpider(phase=-self.phase, _num_inputs=self.num_inputs, _num_outputs=self.num_outputs)

    def __repr__(self) -> str:
        """Return string representation of the p-spider.

        Returns
        -------
        str
            String showing phase function and number of wires.
        """
        if self.phase.is_zero:
            return f"PSpider(num_imputs={self.num_inputs}, num_outputs={self.num_outputs})"
        return f"PSpider(f(x)={self.phase}, num_imputs={self.num_inputs}, num_outputs={self.num_outputs})"


@dataclass
class Swap(ProperDiagram):
    r"""Swap diagram: exchanges two modes.

    Represents the diagram::

            ┌───┐
        ────┤ X ├────
        ────┤   ├────
            └───┘

    Matrix elements::

        ∫ ds ds' \|s', s⟩⟨s, s'\|

    References
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    _num_inputs: int = field(default=2, init=False)
    _num_outputs: int = field(default=2, init=False)

    def conjugate(self) -> Diagram:
        """Swap is self-conjugate.

        Returns
        -------
        Swap
            New Swap instance.
        """
        return Swap()

    def __repr__(self) -> str:
        """Return string representation of the swap diagram.

        Returns
        -------
        str
            "Swap()"
        """
        return "Swap()"


@dataclass
class VoidDiagram(ProperDiagram):
    r"""Void diagram: a transient, undrawn placeholder of arbitrary arity.

    Represents no wire and no physical content at all -- it is pure
    bookkeeping, not a state, effect, gate, or identity wire. It exists
    solely so that a container's shape (its `num_inputs`/`num_outputs`)
    never has to change when one of its slots is fully consumed elsewhere
    in the diagram.

    The motivating case is `CopyRule`'s cross-container application: when
    a state/effect is copied through a spider that lives in a different
    container, the state/effect's own original slot has nothing left to
    put there (its content now lives as copies elsewhere) -- but simply
    deleting that slot would shrink its container's arity and force an
    arity-propagation cascade through every parent container above it.
    Installing a `VoidDiagram` with the exact same arity instead keeps
    that slot's shape identical to what it replaced, so nothing upstream
    ever needs to be touched or recomputed.

    A `VoidDiagram` is meant to be transient: it should never survive past
    `optimize()`'s return value. The end-of-pipeline cleanup pass removes
    every `VoidDiagram` for good (alongside any leftover identity wires),
    actually shrinking the containers they sit in at that point, once and
    for all, rather than doing so eagerly on every application.

    Visually, a `VoidDiagram` reserves exactly the layout space an
    identity wire of the same arity would take, but draws nothing --
    unlike an identity spider, which draws as a straight wire.

    Parameters
    ----------
    _num_inputs : int
        Number of input wires, matching whatever this slot replaced.
    _num_outputs : int
        Number of output wires, matching whatever this slot replaced.
    """

    def conjugate(self) -> Diagram:
        """Void is self-conjugate (there is no phase to negate).

        Returns
        -------
        VoidDiagram
            New VoidDiagram with the same arity.
        """
        return VoidDiagram(self.num_inputs, self.num_outputs)

    def __repr__(self) -> str:
        """Return string representation of the void diagram.

        Returns
        -------
        str
            "VoidDiagram(num_inputs, num_outputs)"
        """
        return f"VoidDiagram({self.num_inputs}, {self.num_outputs})"


@dataclass
class Fourier(ProperDiagram):
    r"""Fourier transform diagram.

    Represents the diagram:
        ┌───┐
    ────┤ F ├────
        └───┘

    Transforms between position and momentum bases.

    Matrix elements:
        ∫ du \|u⟩_p ⟨u\|_q

    References
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)

    def conjugate(self) -> Diagram:
        """Conjugate of Fourier is inverse Fourier.

        Returns
        -------
        FourierInv
            Inverse Fourier diagram.
        """
        return FourierInv()

    def __repr__(self) -> str:
        """Return string representation of the Fourier diagram.

        Returns
        -------
        str
            "Fourier()"
        """
        return "Fourier()"


@dataclass
class FourierInv(ProperDiagram):
    r"""Inverse Fourier transform diagram.

    Represents the diagram:
        ┌────┐
    ────┤ F† ├────
        └────┘

    Matrix elements:
        ∫ du \|u⟩_q ⟨u\|_p

    References
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)

    def conjugate(self) -> Diagram:
        """Conjugate of inverse Fourier is Fourier.

        Returns
        -------
        Fourier
            Fourier diagram.
        """
        return Fourier()

    def __repr__(self) -> str:
        """Return string representation of the inverse Fourier diagram.

        Returns
        -------
        str
            "FourierInv()"
        """
        return "FourierInv()"


@dataclass
class Fourier2(ProperDiagram):
    r"""Squared Fourier transform diagram.

    Represents the diagram:
        ┌────┐
    ────┤ F² ├────
        └────┘

    This is equivalent to a π rotation in phase space.

    Matrix elements:
        ∫ ds \|-s⟩_q ⟨s\|_q

    References
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. III.B, Table II
    """

    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)

    def conjugate(self) -> Diagram:
        """F² is self-conjugate.

        Returns
        -------
        Fourier2
            New Fourier2 instance.
        """
        return Fourier2()

    def __repr__(self) -> str:
        """Return string representation of the squared Fourier diagram.

        Returns
        -------
        str
            "Fourier2()"
        """
        return "Fourier2()"


def flatten_composition(diagram: Diagram) -> Diagram:  # ruff: ignore[complex-structure, too-many-branches]
    """Recursively flatten any CompositionDiagram found in the diagram.

    This method recursively traverses the diagram and flattens:

    1. Single-element compositions → return the element directly
    2. Nested compositions → extract and merge their diagrams, preserving connectivity
    3. Compositions inside TensorDiagram → flatten the composition
    4. Compositions inside ContractedDiagram → flatten the composition

    When flattening nested compositions, the connectivity is adjusted to reflect
    the flattened structure.

    Examples::

        CompositionDiagram([A]) → A
        CompositionDiagram([A, CompositionDiagram([B, C]), D])
            → CompositionDiagram([A, B, C, D])
        TensorDiagram([CompositionDiagram([A, B]), C])
            → TensorDiagram([CompositionDiagram([A, B]), C])  # Composition inside Tensor is NOT flattened
        CompositionDiagram([TensorDiagram([A, B]), C])
            → CompositionDiagram([TensorDiagram([A, B]), C])  # Tensor inside Composition is NOT flattened

    Parameters
    ----------
    diagram : Diagram
        The diagram to flatten.

    Returns
    -------
    Diagram
        Flattened diagram with no nested compositions.
    """
    # Base case: if it's a CompositionDiagram, flatten it
    if isinstance(diagram, CompositionDiagram):
        # If it's a single-element composition, return the element directly
        if len(diagram.diagrams) == 1:
            return flatten_composition(diagram.diagrams[0])

        # Build the flattened list of diagrams and accumulate connectivity
        flattened_diagrams: list[Diagram] = []
        # Maps from original diagram index to the range of flattened indices
        index_mapping = {}  # original_index -> (start_idx, end_idx)
        all_connectivity = {}

        current_idx = 0
        for i, sub_diagram in enumerate(diagram.diagrams):
            flattened_sub = flatten_composition(sub_diagram)

            if isinstance(flattened_sub, CompositionDiagram):
                # If the sub-diagram is a CompositionDiagram, merge its elements
                sub_start = current_idx
                sub_end = current_idx + len(flattened_sub.diagrams)
                flattened_diagrams.extend(flattened_sub.diagrams)
                # Merge the sub-composition's connectivity
                for key, conn in flattened_sub.connectivity.items():
                    # Adjust indices: key + sub_start gives the new position
                    new_key = key + sub_start
                    all_connectivity[new_key] = conn
                if i < len(diagram.diagrams) - 1:
                    all_connectivity[sub_end - 1] = diagram.connectivity[i]
                current_idx = sub_end
            else:
                # Single element: just append it
                index_mapping[i] = (current_idx, current_idx + 1)
                if i < len(diagram.diagrams) - 1:
                    all_connectivity[current_idx] = diagram.connectivity[i]
                flattened_diagrams.append(flattened_sub)
                current_idx += 1

        # If after flattening we have a single element, return it directly
        if len(flattened_diagrams) == 1:
            return flattened_diagrams[0]

        # Return the flattened composition with adjusted connectivity
        return CompositionDiagram(flattened_diagrams, all_connectivity)

    # If it's a TensorDiagram, flatten each of its sub-diagrams
    if isinstance(diagram, TensorDiagram):
        flattened_diagrams = []
        for sub_diagram in diagram.diagrams:
            flattened_diagrams.append(flatten_composition(sub_diagram))
        return TensorDiagram(flattened_diagrams)

    # If it's a ContractedDiagram, flatten its first and second diagrams
    if isinstance(diagram, ContractedDiagram):
        first = flatten_composition(diagram.first)
        second = flatten_composition(diagram.second)
        # If either changed, create a new ContractedDiagram
        if first is not diagram.first or second is not diagram.second:
            return ContractedDiagram(
                first=first,
                second=second,
                I1=diagram.I1,
                I2=diagram.I2,
                J1=diagram.J1,
                J2=diagram.J2,
            )

    return diagram

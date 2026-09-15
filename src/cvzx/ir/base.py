"""CV-ZX representation of usual gates."""

import operator
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar

from sympy import Expr, Poly, S, Symbol, symbols, sympify

from cvzx.exceptions import ArityMismatchError, InvalidSymbolError


class ZxPoly(Poly):
    """Real polynomial in one variable for CV ZX calculus phase functions.

    Wraps `sympy.Poly` (single generator `x`) with a `dict[degree, coeff]`-based convenience
    API -- build from a coefficient dict, a plain sympy expression, or another `Poly`; zero
    coefficients are omitted from `.coeffs`. Coefficients may be numeric or symbolic. See
    :doc:`../dev_guide/architecture` for the representation choices and worked examples.

    Attributes
    ----------
    coeffs : dict[int, float | Expr]
        Dictionary mapping degree → coefficient. For numeric coefficients,
        returns Python floats; for symbolic coefficients, returns sympy expressions.
    """

    _var = symbols("x", real=True)

    def __new__(cls, coeffs_or_poly: dict[int, float | int | Expr] | Poly | Expr | None = None, *args, **kwargs):  # ruff: ignore[missing-type-args, missing-type-kwargs, missing-return-type-special-method]
        """Create a new ZxPoly instance.

        Intercepts instance creation to handle the special case where the user provides a
        coefficient dictionary instead of a sympy expression or Poly object -- a `Poly`, `Expr`,
        or `None` argument is passed straight through to `sympy.Poly`. See
        :doc:`../dev_guide/architecture` for an example of each accepted input form.

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
    def coeffs(self) -> dict[int, float | complex | Expr]:
        """Coefficients as a dictionary, for backward compatibility.

        Returns a dictionary mapping degree → coefficient. Zero coefficients
        are omitted. Numeric coefficients are converted to Python floats
        (or `complex`, if the coefficient has a nonzero imaginary part) for
        compatibility, while symbolic coefficients remain as sympy expressions.

        Returns
        -------
        dict[int, float | complex | Expr]
            Dictionary mapping degree to coefficient. Returns empty dict
            for the zero polynomial.
        """
        if self.is_zero:
            return {}

        coeff_dict = {}
        for monom, coeff in self.terms():
            degree = monom[0]  # For univariate
            # Convert numeric coefficients to Python floats for compatibility
            # -- unless the coefficient is genuinely complex (nonzero
            # imaginary part), in which case `float()` would raise.
            if coeff.is_number and not coeff.is_symbol:
                as_complex = complex(coeff)
                coeff_dict[degree] = as_complex.real if as_complex.imag == 0 else as_complex
            else:
                coeff_dict[degree] = coeff
        return coeff_dict

    def is_parametric(self) -> bool:
        """Check if the phase is parametric or not.

        Returns
        -------
        bool
        """
        return any(isinstance(coef, Expr) for coef in self.coeffs.values())

    def get_parameters(self) -> set[Symbol]:
        """Return the free symbols used in the polynomial's coefficients.

        Deliberately not `self.free_symbols`: sympy's `Poly.free_symbols`
        also includes the polynomial's own generator variable, which is
        never a gate parameter.

        Returns
        -------
        set[Symbol]
        """
        return {symbol for coef in self.coeffs.values() if isinstance(coef, Expr) for symbol in coef.free_symbols}

    def __repr__(self) -> str:
        """Return a human-readable ZX-calculus-notation string, terms ordered by increasing degree.

        See :doc:`../dev_guide/architecture` for example output.

        Returns
        -------
        str
            String representation of the polynomial.
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

    _next_id: ClassVar[int] = 1

    def __init__(self) -> None:
        self.id = Diagram._next_id
        Diagram._next_id += 1

    @classmethod
    def _reserve_id(cls, node_id: int) -> None:
        """Ensure future auto-assigned ids stay past `node_id`.

        Called wherever a `Diagram`'s `.id` is force-set to a value not
        drawn from `_next_id` (e.g. graph reconstruction preserving a
        specific node's original id) -- without this, `_next_id` can
        later independently reach that same value and hand it to an
        unrelated object, producing two different `Diagram` instances
        with the same `.id`.
        """
        cls._next_id = max(cls._next_id, node_id + 1)

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
    :cite:`nagayoshi2024zx`, Sec. III.B, Table II
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
        ArityMismatchError
            If the connectivity dictionary isn't coherent with the inputs
            of self and/or the outputs of the input diagram.
        """
        if connectivity is None:
            connectivity = {k: k for k in range(self.num_inputs)}
        # Check the validity of the connectivity
        keys = list(connectivity.keys())
        keys.sort()
        if keys != list(range(self.num_inputs)):
            msg = "The keys of the connectivity dictionary do not correspond the input indices of the current diagram."
            raise ArityMismatchError(msg)
        values = list(connectivity.values())
        values.sort()
        if values != list(range(other.num_outputs)):
            msg = (
                "The values of the connectivity dictionary do not correspond the output indices of the input diagram."
            )
            raise ArityMismatchError(msg)
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
    r"""Diagram from contracting (tracing) two diagrams' outputs into each other's inputs.

    `I1`/`J1` index into `first`'s own output/input port numbering and `I2`/`J2` into
    `second`'s -- never the container's external port numbering. `I1` (first's outputs)
    connects, in order, to `I2` (second's inputs); `J2` (second's outputs) connects to `J1`
    (first's inputs). `len(I1) == len(I2)` and `len(J1) == len(J2)` are enforced by `__init__`.
    Only wires not named in any of the four sequences remain external. See
    :doc:`../dev_guide/architecture` for the full Eq. (51) derivation this implements.

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

    References
    ----------
    :cite:`nagayoshi2024zx`, Definition 11, Eq. (51)
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
        ArityMismatchError
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
            raise ArityMismatchError(msg)
        if len(set(J1)) != len(J1):
            msg = f"Input wires of the first diagram {J1} contain duplicate indices"
            raise ArityMismatchError(msg)

        # Check uniqueness of indices of the second diagram
        if len(set(I2)) != len(I2):
            msg = f"Output wires of the second diagram {I2} contain duplicate indices"
            raise ArityMismatchError(msg)
        if len(set(J2)) != len(J2):
            msg = f"Input wires of the second diagram {J2} contain duplicate indices"
            raise ArityMismatchError(msg)

        # Validate lengths match
        if len(I1) != len(I2):
            msg = f"I1 length ({len(I1)}) must equal I2 length ({len(I2)})"
            raise ArityMismatchError(msg)
        if len(J1) != len(J2):
            msg_0 = f"J1 length ({len(J1)}) must equal J2 length ({len(J2)})"
            raise ArityMismatchError(msg_0)

        # Validate I1, I2, J1 and J2 are not all empty
        if not I1 and not I2 and not J1 and not J2:
            msg = "A contraction diagam must have at least one contraction link."
            raise ArityMismatchError(msg)

        # Validate I1 indices are within first's outputs
        for idx in self.I1:
            if idx < 0 or idx >= self.first.num_outputs:
                msg_1 = f"I1 index {idx} out of range for first diagram outputs [0, {self.first.num_outputs})"
                raise ArityMismatchError(msg_1)

        # Validate I2 indices are within second's inputs
        for idx in self.I2:
            if idx < 0 or idx >= self.second.num_inputs:
                msg_2 = f"I2 index {idx} out of range for second diagram inputs [0, {self.second.num_inputs})"
                raise ArityMismatchError(msg_2)

        # Validate J1 indices are within first's inputs
        for idx in self.J1:
            if idx < 0 or idx >= self.first.num_inputs:
                msg_3 = f"J1 index {idx} out of range for first diagram inputs [0, {self.first.num_inputs})"
                raise ArityMismatchError(msg_3)

        # Validate J2 indices are within second's outputs
        for idx in self.J2:
            if idx < 0 or idx >= self.second.num_outputs:
                msg_4 = f"J2 index {idx} out of range for second diagram outputs [0, {self.second.num_outputs})"
                raise ArityMismatchError(msg_4)

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
        ArityMismatchError
            If the connectivity dictionary isn't coherent with the inputs
            of self and/or the outputs of the input diagram.
        """
        if connectivity is None:
            connectivity = {k: k for k in range(self.num_inputs)}
        # Check the validity of the connectivity
        keys = list(connectivity.keys())
        keys.sort()
        if keys != list(range(self.num_inputs)):
            msg = "The keys of the connectivity dictionary do not correspond the input indices of the current diagram."
            raise ArityMismatchError(msg)
        values = list(connectivity.values())
        values.sort()
        if values != list(range(other.num_outputs)):
            msg = (
                "The values of the connectivity dictionary do not correspond the output indices of the input diagram."
            )
            raise ArityMismatchError(msg)
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
    :cite:`nagayoshi2024zx`, Definition 9
    """

    diagrams: Sequence[Diagram]

    def partial_trace(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements]
        self,
        diagram_pairs: Sequence[tuple[int, Sequence[int], Sequence[int]]],
    ) -> None:
        r"""Contract two consecutive diagrams in the tensor product into one `ContractedDiagram`.

        Connects the given output wires of one diagram to the given input wires of the other,
        and vice versa; unconsumed wires remain external. The two diagrams must be consecutive
        in the tensor product -- a layout restriction to keep the result easy to draw, not a
        fundamental one. Implements diagram contraction as defined in [1] Eq. (51); see
        :doc:`../dev_guide/architecture` for the full derivation.

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
        ArityMismatchError
            If diagram_pairs is empty.
            If first diagram index is out of range.
            If second diagram index is out of range.
            If contracted diagrams are not consecutive (\|first - second\| != 1).
            If input wire index is out of range for the second diagram.
            If output wires contain duplicate indices.
            If input wires contain duplicate indices.

        References
        ----------
        :cite:`nagayoshi2024zx`, Definition 11
        """
        if not diagram_pairs:
            msg = "diagram_pairs cannot be empty"
            raise ArityMismatchError(msg)

        # Make a mutable copy of the diagrams list
        diagrams = list(self.diagrams)

        # Sort pairs by first diagram index (to process from higher to lower)
        sorted_pairs = sorted(diagram_pairs, key=operator.itemgetter(0))
        first_idx, f_output_wires, f_input_wires = sorted_pairs[0]
        second_idx, s_output_wires, s_input_wires = sorted_pairs[1]
        # Validate diagram indices in the tensor product
        if first_idx < 0 or first_idx >= len(diagrams):
            msg = f"First diagram index {first_idx} out of range [0, {len(diagrams) - 1}]"
            raise ArityMismatchError(msg)
        if second_idx < 0 or second_idx >= len(diagrams):
            msg = f"Second diagram index {second_idx} out of range [0, {len(diagrams) - 1}]"
            raise ArityMismatchError(msg)

        # Check that diagrams are consecutive
        if abs(first_idx - second_idx) != 1:
            msg = (
                f"Diagrams {first_idx} and {second_idx} are not consecutive. "
                f"Contraction requires consecutive diagrams."
            )
            raise ArityMismatchError(msg)

        first = diagrams[first_idx]
        second = diagrams[second_idx]

        # Validate wire indices for the first diagram
        for w in f_output_wires:
            if w < 0 or w >= first.num_outputs:
                msg = f"Output wire {w} out of range [0, {first.num_outputs - 1}] for diagram {first_idx}"
                raise ArityMismatchError(msg)
        for w in f_input_wires:
            if w < 0 or w >= first.num_inputs:
                msg = f"Input wire {w} out of range [0, {second.num_inputs - 1}] for diagram {first_idx}"
                raise ArityMismatchError(msg)
        # Validate wire indices for the second diagram
        for w in s_output_wires:
            if w < 0 or w >= second.num_outputs:
                msg = f"Input wire {w} out of range [0, {first.num_inputs - 1}] for diagram {second_idx}"
                raise ArityMismatchError(msg)
        for w in s_input_wires:
            if w < 0 or w >= second.num_inputs:
                msg = f"Output wire {w} out of range [0, {second.num_outputs - 1}] for diagram {second_idx}"
                raise ArityMismatchError(msg)

        # Validate lengths match
        if len(f_output_wires) != len(s_input_wires):
            msg_0 = f"I1 length ({len(f_output_wires)}) must equal I2 length ({len(s_input_wires)})"
            raise ArityMismatchError(msg_0)
        if len(f_input_wires) != len(s_output_wires):
            msg_1 = f"J1 length ({len(f_input_wires)}) must equal J2 length ({len(s_output_wires)})"
            raise ArityMismatchError(msg_1)

        # Check uniqueness of indices of the first diagram
        if len(set(f_output_wires)) != len(f_output_wires):
            msg = f"Output wires of the first diagram {f_output_wires} contain duplicate indices"
            raise ArityMismatchError(msg)
        if len(set(f_input_wires)) != len(f_input_wires):
            msg = f"Input wires of the first diagram {f_input_wires} contain duplicate indices"
            raise ArityMismatchError(msg)

        # Check uniqueness of indices of the second diagram
        if len(set(s_output_wires)) != len(s_output_wires):
            msg = f"Output wires of the second diagram {s_output_wires} contain duplicate indices"
            raise ArityMismatchError(msg)
        if len(set(s_input_wires)) != len(s_input_wires):
            msg = f"Input wires of the second diagram {s_input_wires} contain duplicate indices"
            raise ArityMismatchError(msg)

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
        ArityMismatchError
            If the connectivity dictionary isn't coherent with the inputs
            of self and/or the outputs of the input diagram.
        """
        if connectivity is None:
            connectivity = {k: k for k in range(self.num_inputs)}
        # Check the validity of the connectivity
        keys = list(connectivity.keys())
        keys.sort()
        if keys != list(range(self.num_inputs)):
            msg = "The keys of the connectivity dictionary do not correspond the input indices of the current diagram."
            raise ArityMismatchError(msg)
        values = list(connectivity.values())
        values.sort()
        if values != list(range(other.num_outputs)):
            msg = (
                "The values of the connectivity dictionary do not correspond the output indices of the input diagram."
            )
            raise ArityMismatchError(msg)
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
            flattened_sub = flatten_tensor(sub_diagram)
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
    :cite:`nagayoshi2024zx`, Definition 10
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
        ArityMismatchError
            If the connectivity dictionary isn't coherent with the inputs
            of self and/or the outputs of the input diagram.
        """
        if connectivity is None:
            connectivity = {k: k for k in range(self.num_inputs)}
        # Check the validity of the connectivity
        keys = list(connectivity.keys())
        keys.sort()
        if keys != list(range(self.num_inputs)):
            msg = "The keys of the connectivity dictionary do not correspond the input indices of the current diagram."
            raise ArityMismatchError(msg)
        values = list(connectivity.values())
        values.sort()
        if values != list(range(other.num_outputs)):
            msg = (
                "The values of the connectivity dictionary do not correspond the output indices of the input diagram."
            )
            raise ArityMismatchError(msg)
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
        ArityMismatchError
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
                raise ArityMismatchError(msg)
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
                raise ArityMismatchError(msg)
            # Check values
            values = list(self.connectivity[i].values())
            values.sort()
            if values != list(range(self.diagrams[i].num_outputs)):
                msg = (
                    "The values of the connectivity dictionary do not "
                    f"correspond the output indices of the sub_diagram {i}."
                )
                raise ArityMismatchError(msg)

    def __repr__(self) -> str:
        """Return string representation of the composition diagram.

        Returns
        -------
        str
            String representation showing all components in order.
        """
        return f"Compose({self.diagrams})"


def _substitute_value(
    value: Any,  # ruff: ignore[any-type]
    mapping: dict[Symbol, Any],
) -> Any:  # ruff: ignore[any-type]
    """Apply a symbol substitution to a phase/parameter value.

    Handles `ZxPoly` specially since `sympy.Poly.subs` returns a plain
    `Add`, not a `Poly` -- substitution is done coefficient-wise instead.

    Parameters
    ----------
    value : Any
        The current field value (a `ZxPoly`, a plain `Expr`, or a
        non-symbolic value passed through unchanged).
    mapping : dict[Symbol, Any]
        Mapping from symbol to replacement value.

    Returns
    -------
    Any
    """
    if isinstance(value, ZxPoly):
        return ZxPoly({
            degree: sympify(coeff).subs(mapping) if isinstance(coeff, Expr) else coeff
            for degree, coeff in value.coeffs.items()
        })
    if isinstance(value, Expr):
        return value.subs(mapping)
    return value


class Parametrized:
    """Mixin providing shared symbolic-parameter bookkeeping.

    Adds `param_measurement_map` handling (feedforward provenance: which
    measurement outcomes a symbolic parameter depends on) uniformly across
    `QSpider`, `PSpider`, and every `CompactDiagram` gate subclass. This is
    a plain method-only mixin, not a dataclass itself: dataclass field
    ordering means fields can't be hoisted into a shared base once a
    subclass adds a non-default field, so each concrete class keeps
    declaring its own `parametric`/`feedforward`/`measurement_ids`/
    `param_measurement_map` fields, but every method touching them lives
    here.

    Every concrete class using this mixin must declare a class-level
    `_param_fields: ClassVar[tuple[str, ...]]` naming its symbolic-value
    attributes (e.g. `("phase",)` for a spider, `("alpha", "beta", "lam")`
    for `ArbitraryGate`), and implement `_rebuild(values, new_map)` to
    construct a new instance from substituted field values.
    """

    _param_fields: ClassVar[tuple[str, ...]] = ()

    param_measurement_map: dict[Symbol, set[int]]
    parametric: bool
    feedforward: bool
    measurement_ids: set[int] | None

    def get_parameters(self) -> set[Symbol]:
        """Return every symbolic parameter used by this object's fields.

        Returns
        -------
        set[Symbol]
        """
        parameters: set[Symbol] = set()
        for field_name in self._param_fields:
            value = getattr(self, field_name)
            if isinstance(value, ZxPoly):
                parameters |= value.get_parameters()
            elif isinstance(value, Expr):
                parameters |= value.free_symbols
        return parameters

    @property
    def is_parametric(self) -> bool:
        """Whether this object carries any free symbolic parameter.

        Not to be confused with `ZxPoly.is_parametric()`, a method on a
        different class checking the same notion for a bare polynomial.

        Returns
        -------
        bool
        """
        return bool(self.get_parameters())

    def _sync_feedforward_state(self) -> None:
        """Validate `param_measurement_map` and derive dependent fields.

        Raises
        ------
        TypeError
            If `param_measurement_map` is not a `dict`.
        InvalidSymbolError
            If `param_measurement_map` references a symbol that is not
            one of this object's own parameters.
        ValueError
            If any of `param_measurement_map`'s values is not a non-empty
            `set`, or if `feedforward` is True (or `measurement_ids` is
            set) while `param_measurement_map` is empty -- provenance is
            required, not optional: `feedforward`/`measurement_ids` are
            derived from `param_measurement_map`, never set independently
            of it.
        """
        if not isinstance(self.param_measurement_map, dict):
            msg = f"The param_measurement_map attribute must be a dict, got {type(self.param_measurement_map)}."
            raise TypeError(msg)
        if not self.param_measurement_map.keys() <= self.get_parameters():
            extra = self.param_measurement_map.keys() - self.get_parameters()
            msg = (
                f"param_measurement_map on {type(self).__name__} references symbol(s) {sorted(extra, key=str)} "
                f"that are not among its own parameters {sorted(self.get_parameters(), key=str)}."
            )
            raise InvalidSymbolError(msg)
        for symbol, ids in self.param_measurement_map.items():
            if not isinstance(ids, set) or not ids:
                msg = f"The param_measurement_map value for {symbol} must be a non-empty set, got {ids!r}."
                raise ValueError(msg)

        if self.param_measurement_map:
            self.measurement_ids = set().union(*self.param_measurement_map.values())
            self.feedforward = bool(self.measurement_ids)
            return

        if self.feedforward or self.measurement_ids:
            msg = (
                f"{type(self).__name__} has feedforward={self.feedforward!r} / "
                f"measurement_ids={self.measurement_ids!r} but an empty param_measurement_map -- "
                "pass a non-empty param_measurement_map (symbol -> measurement ids) instead; "
                "feedforward/measurement_ids are derived from it, not set independently."
            )
            raise ValueError(msg)

    def slice_param_map(self, params: set[Symbol]) -> dict[Symbol, set[int]]:
        """Restrict `param_measurement_map` to a subset of symbols.

        Used by `expand()` to hand each spawned sub-object exactly the
        provenance entries relevant to its own parameters.

        Parameters
        ----------
        params : set[Symbol]
            Symbols relevant to the sub-object being constructed.

        Returns
        -------
        dict[Symbol, set[int]]
            A fresh dict (values copied, not aliased) restricted to `params`.
        """
        return {symbol: set(ids) for symbol, ids in self.param_measurement_map.items() if symbol in params}

    def substitute_parameters(self, mapping: dict[Symbol, Any]) -> "Parametrized":
        """Substitute symbolic parameters with concrete or other symbolic values.

        Parameters
        ----------
        mapping : dict[Symbol, Any]
            Mapping from symbol to replacement value (numeric or symbolic).

        Returns
        -------
        Parametrized
            A new instance with the substitution applied.
        """
        values = {
            field_name: _substitute_value(getattr(self, field_name), mapping) for field_name in self._param_fields
        }
        new_map = {symbol: ids for symbol, ids in self.param_measurement_map.items() if symbol not in mapping}
        return self._rebuild(values, new_map)

    def evaluate(self, **kwargs: Any) -> "Parametrized":  # ruff: ignore[any-type]
        """Substitute symbolic parameters by name.

        Parameters
        ----------
        **kwargs : Any
            Replacement values keyed by symbol name.

        Returns
        -------
        Parametrized
            A new instance with the substitution applied.
        """
        mapping = {Symbol(name): value for name, value in kwargs.items()}
        return self.substitute_parameters(mapping)

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "Parametrized":
        """Construct a new instance from substituted field values.

        Must be implemented by every concrete class using this mixin.

        Parameters
        ----------
        values : dict[str, Any]
            New values for each name in `_param_fields`.
        new_map : dict[Symbol, set[int]]
            `param_measurement_map` filtered to still-symbolic parameters.

        Returns
        -------
        Parametrized
        """
        raise NotImplementedError


@dataclass
class QSpider(ProperDiagram, Parametrized):
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
    :cite:`nagayoshi2024zx`, Sec. III.B, Table II
    """

    phase: ZxPoly
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    _num_inputs: int
    _num_outputs: int
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)

    _param_fields: ClassVar[tuple[str, ...]] = ("phase",)

    def conjugate(self) -> Diagram:
        """Conjugate: negate phase, keep q-spider type.

        Returns
        -------
        QSpider
            New QSpider with negated phase.
        """
        return QSpider(
            phase=-self.phase,
            parametric=self.parametric,
            feedforward=self.feedforward,
            measurement_ids=self.measurement_ids,
            _num_inputs=self.num_inputs,
            _num_outputs=self.num_outputs,
            param_measurement_map=dict(self.param_measurement_map),
        )

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

    def __post_init__(self) -> None:
        """Initialize the q-spider and validate parameters.

        Raises
        ------
        ValueError
            If `parametric` doesn't match whether `phase` is actually
            parametric, or if `_sync_feedforward_state` finds an
            inconsistency (see its own docstring).
        """
        if self.parametric != self.phase.is_parametric():
            msg = "The parametric attribute is not accurate."
            raise ValueError(msg)
        self._sync_feedforward_state()
        super().__post_init__()

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "QSpider":
        phase = values["phase"]
        return QSpider(
            phase=phase,
            parametric=phase.is_parametric(),
            _num_inputs=self.num_inputs,
            _num_outputs=self.num_outputs,
            param_measurement_map=new_map,
        )


@dataclass
class PSpider(ProperDiagram, Parametrized):
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
    :cite:`nagayoshi2024zx`, Sec. III.B, Table II
    """

    phase: ZxPoly
    parametric: bool = False
    feedforward: bool = False
    measurement_ids: set[int] | None = None
    _num_inputs: int
    _num_outputs: int
    param_measurement_map: dict[Symbol, set[int]] = field(default_factory=dict)

    _param_fields: ClassVar[tuple[str, ...]] = ("phase",)

    def conjugate(self) -> Diagram:
        """Conjugate: negate phase, keep p-spider type.

        Returns
        -------
        PSpider
            New PSpider with negated phase.
        """
        return PSpider(
            phase=-self.phase,
            parametric=self.parametric,
            feedforward=self.feedforward,
            measurement_ids=self.measurement_ids,
            _num_inputs=self.num_inputs,
            _num_outputs=self.num_outputs,
            param_measurement_map=dict(self.param_measurement_map),
        )

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

    def __post_init__(self) -> None:
        """Initialize the p-spider and validate parameters.

        Raises
        ------
        ValueError
            If `parametric` doesn't match whether `phase` is actually
            parametric, or if `_sync_feedforward_state` finds an
            inconsistency (see its own docstring).
        """
        if self.parametric != self.phase.is_parametric():
            msg = "The parametric attribute is not accurate."
            raise ValueError(msg)
        self._sync_feedforward_state()
        super().__post_init__()

    def _rebuild(self, values: dict[str, Any], new_map: dict[Symbol, set[int]]) -> "PSpider":
        phase = values["phase"]
        return PSpider(
            phase=phase,
            parametric=phase.is_parametric(),
            _num_inputs=self.num_inputs,
            _num_outputs=self.num_outputs,
            param_measurement_map=new_map,
        )


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
    :cite:`nagayoshi2024zx`, Sec. III.B, Table II
    """

    _num_inputs: int = field(default=2, init=False)
    _num_outputs: int = field(default=2, init=False)
    void_input_port: int | None = None
    """Visualization-only marker: which input port (0 or 1) is the one whose
    diagonal, once followed outward through the rest of the diagram, dead-ends
    in `VoidDiagram` filler. `None` for an ordinary/unmarked `Swap`. Does not
    affect the diagram's semantics -- only `_draw_swap` reads it, to skip
    drawing that uninteresting leg."""

    def conjugate(self) -> Diagram:
        """Swap is self-conjugate.

        Returns
        -------
        Swap
            New Swap instance, preserving `void_input_port`.
        """
        return Swap(void_input_port=self.void_input_port)

    def __repr__(self) -> str:
        """Return string representation of the swap diagram.

        Returns
        -------
        str
            "Swap()", or "Swap(void_input_port=<n>)" when marked.
        """
        if self.void_input_port is not None:
            return f"Swap(void_input_port={self.void_input_port})"
        return "Swap()"


@dataclass
class VoidDiagram(ProperDiagram):
    r"""Void diagram: a transient, undrawn placeholder of arbitrary arity.

    Represents no wire and no physical content at all -- pure bookkeeping, not a state, effect,
    gate, or identity wire. It exists so a container's shape (`num_inputs`/`num_outputs`) never
    has to change when one of its slots is fully consumed elsewhere in the diagram, avoiding an
    arity-propagation cascade through every parent container above it. A `VoidDiagram` must
    never survive past `optimize()`'s return value -- the end-of-pipeline cleanup pass removes
    every one (and any leftover identity wires) for good. See :doc:`../dev_guide/architecture`
    for the motivating cross-container rewrite case and its visualization behavior.

    Attributes
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
    :cite:`nagayoshi2024zx`, Sec. III.B, Table II
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
    :cite:`nagayoshi2024zx`, Sec. III.B, Table II
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
    :cite:`nagayoshi2024zx`, Sec. III.B, Table II
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
    """Recursively flatten any CompositionDiagram found in the diagram into one flat sequence.

    A single-element composition collapses to its element; a `CompositionDiagram` nested inside
    another `CompositionDiagram` is merged in, with connectivity indices shifted to match. A
    composition nested inside a `TensorDiagram`/`ContractedDiagram` is recursed into but left in
    place there -- only composition-inside-composition actually gets flattened. See
    :doc:`../dev_guide/architecture` for worked examples of each case.

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


def flatten_tensor(diagram: Diagram) -> Diagram:
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
            flattened_sub = flatten_tensor(sub)
            if isinstance(flattened_sub, TensorDiagram):
                flattened.extend(flattened_sub.diagrams)
            else:
                flattened.append(flattened_sub)
        if len(flattened) == 1:
            return flattened[0]
        return TensorDiagram(flattened)

    if isinstance(diagram, CompositionDiagram):
        # Don't flatten compositions, but recursively flatten tensors inside
        flattened_diagrams = [flatten_tensor(sub) for sub in diagram.diagrams]
        return CompositionDiagram(flattened_diagrams, diagram.connectivity)

    if isinstance(diagram, ContractedDiagram):
        first = flatten_tensor(diagram.first)
        second = flatten_tensor(diagram.second)
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

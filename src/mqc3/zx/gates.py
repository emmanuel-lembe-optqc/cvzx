"""CV-ZX representation of quantum gates from Nagayoshi et al. (2024), Sec. II.C.

This module implements the standard CV quantum gates as compact diagrams
built from proper diagrams (spiders, Fourier, Swap). Each gate is a subclass
of CompactDiagram and provides a expand() method that returns the
equivalent CompositionDiagram or TensorDiagram of basic CV ZX elements.

References:
-----------
[3] Nagayoshi et al., CV ZX calculus, Sec. II.C, Table I
"""

from dataclasses import dataclass, field

import numpy as np

from mqc3.zx.base_gates import (
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
            last_idx = len(diagrams)
            old_connectivity[last_idx] = connectivity
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
    r"""Displacement gate D(α).

    Represents the displacement operator D(α) = exp(α â† - α* â).
    Decomposes into a q-spider and a p-spider as shown in [3] Eq. (57).

    Parameters
    ----------
    alpha : complex
        Displacement amplitude.

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.1, Eq. (57)
    """  # noqa: RUF002

    alpha: complex
    label: str = field(init=False)
    _num_inputs: int = field(default=1, init=False)
    _num_outputs: int = field(default=1, init=False)
    decomposition: Diagram | None = field(default=None, init=False)

    def expand(self) -> CompositionDiagram:
        """Decompose displacement gate into q-spider and p-spider.

        D(α) = Q(√2 Im(α) x) ∘ P(√2 Re(α) x)

        Returns:
        -------
        CompositionDiagram
            Composition of p-spider then q-spider.
        """  # noqa: RUF002
        re = np.sqrt(2) * self.alpha.real
        im = np.sqrt(2) * self.alpha.imag

        q_spider = QSpider(1, 1, ZxPoly({1: im}))
        p_spider = PSpider(1, 1, ZxPoly({1: re}))

        # D(α) = Q ∘ P  (P applied first, then Q)  # noqa: RUF003
        return CompositionDiagram([p_spider, q_spider])

    def conjugate(self) -> "DisplacementGate":
        """Conjugate of displacement gate is displacement with negated alpha.

        Returns:
        -------
        DisplacementGate
            D(-α)
        """  # noqa: RUF002
        return DisplacementGate(alpha=-self.alpha)

    def __post_init__(self) -> None:
        """Initialize the Displacement Gate."""
        self.label = f"D({self.alpha:.2f})"
        super().__post_init__()

    def __repr__(self) -> str:
        """Return string representation of the displacement gate.

        Returns:
        -------
        str
            String showing the displacement amplitude alpha.
        """
        return f"DisplacementGate(α={self.alpha:.2f})"  # noqa: RUF001


@dataclass
class PhaseRotationGate(CompactDiagram):
    r"""Phase rotation gate R(θ).

    Represents the phase rotation operator R(θ) = exp(iθ â† â).
    Decomposes into three quadratic q-spiders as shown in [3] Eq. (58).

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Raises:
    ------
    ValueError:
            If theta is an odd multiple of π/2 (where tan is infinite).
            For these cases, use Fourier2 or composition of Fourier gates.

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.2, Eq. (58)
    """

    theta: float
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
        """
        tan_half = np.tan(self.theta / 2)
        sin_theta = np.sin(self.theta)

        spider1 = PSpider(1, 1, ZxPoly({2: tan_half / 2}))
        spider2 = QSpider(1, 1, ZxPoly({2: -sin_theta / 2}))
        spider3 = PSpider(1, 1, ZxPoly({2: tan_half / 2}))

        return CompositionDiagram([spider1, spider2, spider3])

    def conjugate(self) -> "PhaseRotationGate":
        """Conjugate of phase rotation is rotation by negative angle.

        Returns:
        -------
        PhaseRotationGate
            R(-θ)
        """
        return PhaseRotationGate(theta=-self.theta)

    def __post_init__(self) -> None:
        """Initialise the Phase Rotation Gate.

        Raises:
        ------
        ValueError: If the angle is an odd multiple of π/2.
        """
        self.label = f"R({self.theta:.2f})"
        super().__post_init__()
        # Check for invalid angles where tan is infinite
        if np.isclose(np.abs(self.theta) % np.pi, np.pi / 2):
            msg = (
                f"θ = {self.theta} is an odd multiple of π/2. "
                f"For π/2 rotation, use Fourier gate. For 3π/2, use FourierInv."
            )
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Return string representation of the phase rotation gate.

        Returns:
        -------
        str
            String showing the rotation angle theta.
        """
        return f"PhaseRotationGate(θ={self.theta:.2f})"


@dataclass
class SqueezingGate(CompactDiagram):
    r"""1-mode squeezing gate Sq(τ).

    Represents the squeezing operator with parameter τ (where τ = e^{-r} for
    standard squeezing). Decomposes into four quadratic spiders as shown in
    [3] Eq. (59).

    Parameters
    ----------
    tau : float
        Squeezing parameter. τ > 0 for squeezing, τ < 0 for anti-squeezing.
        τ = e^{-r} corresponds to squeezing in p̂ (x̂ anti-squeezed).

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.3, Eq. (59)
    """

    tau: float
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
        """
        a = self.tau * (1 - self.tau) / 4
        b = -1 / self.tau if abs(self.tau) > 1e-10 else 0  # noqa: PLR2004
        c = (self.tau - 1) / 4
        d = 1.0

        spider1 = QSpider(1, 1, ZxPoly({2: a}))
        spider2 = PSpider(1, 1, ZxPoly({2: b}))
        spider3 = QSpider(1, 1, ZxPoly({2: c}))
        spider4 = PSpider(1, 1, ZxPoly({2: d}))

        return CompositionDiagram([spider1, spider2, spider3, spider4])

    def conjugate(self) -> "SqueezingGate":
        """Conjugate of squeezing gate is squeezing with reciprocal parameter.

        Returns:
        -------
        SqueezingGate
            Sq(1/τ)
        """
        return SqueezingGate(tau=1 / self.tau)

    def __post_init__(self) -> None:
        """Initialize the Squeezing Gate."""
        self.label = f"Sq({self.tau:.2f})"
        super().__post_init__()

    def __repr__(self) -> str:
        """Return string representation of the squeezing gate.

        Returns:
        -------
        str
            String showing the squeezing parameter tau.
        """
        return f"SqueezingGate(τ={self.tau:.2f})"


@dataclass
class ControlledSumGate(CompactDiagram):
    r"""Controlled-sum (CSUM) gate with gain g and specified control/target modes.

    Represents the operation exp(-i g q̂_c p̂_t) where c is the control mode
    and t is the target mode. For g=1, this is the unbiased CSUM gate
    (CV analogue of CNOT). Decomposes into q-spider and p-spider
    with a contraction as shown in [3] Eq. (61)-(62).

    Parameters
    ----------
    gain : float
        Gain parameter g. Default is 1 (unbiased CSUM).
    control : int
        Index of the control mode (1 or 2). Default is 2.
    target : int
        Index of the target mode (1 or 2). Default is 1.

    Raises:
    ------
    ValueError
            If control == target (must be different modes).

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.4, Eq. (61)-(62)
    [4] Yoshikawa et al., QRL configuration, Sec. IV.C.3
    """

    gain: float = 1.0
    control: int = 2
    target: int = 1
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
        """
        # q-spider copies position from control mode
        # p-spider adds the copied position to the momentum of target mode
        if self.control == 2:  # noqa: PLR2004
            control_spider = QSpider(1, 2, ZxPoly({}))  # 1 input, 2 outputs (copy)
            target_spider = PSpider(2, 1, ZxPoly({}))  # 2 inputs, 1 output (add)
        else:
            control_spider = PSpider(1, 2, ZxPoly({}))  # 1 input, 2 outputs (copy)
            target_spider = QSpider(2, 1, ZxPoly({}))  # 2 inputs, 1 output (add)
        if self.gain == 1:
            # Unbiased CSUM
            tensor = TensorDiagram([control_spider, target_spider])

            tensor.partial_trace([
                (0, [1], []),  # q-spider: outputs 0,1 → input 0 (feedback)
                (1, [], [0]),  # p-spider: output 0 → inputs 0,1 (forward)
            ])

            return tensor.diagrams[0]

        # Biased CSUM: squeeze then unbiased then unsqueeze
        sqrt_gain = np.sqrt(self.gain)
        inv_sqrt = 1 / sqrt_gain

        identity = QSpider(1, 1, ZxPoly({}))
        squeeze1 = SqueezingGate(tau=sqrt_gain)
        upper_diagram = control_spider.compose(squeeze1)
        squeeze2 = SqueezingGate(tau=inv_sqrt)
        squeeze2_id = squeeze2.tensor(identity)
        upper_diagram = squeeze2_id.compose(upper_diagram)

        tensor = TensorDiagram([upper_diagram, target_spider])
        tensor.partial_trace([
            (0, [1], []),
            (1, [], [0]),
        ])

        return tensor.diagrams[0]

    def conjugate(self) -> "ControlledSumGate":
        """Conjugate of CSUM is CSUM with negated gain (inverse).

        (e^{-i g q_c p_t})† = e^{+i g q_c p_t} = CSUM(-g)

        Returns:
        -------
        ControlledSumGate
            CSUM with gain = -g (same control/target).
        """
        return ControlledSumGate(gain=-self.gain, control=self.control, target=self.target)

    def __post_init__(self) -> None:
        """Initializes the ControlledSum Gate.

        Raises:
        ------
        ValueError: If the control is equal to the target.
        """
        if self.control == self.target:
            msg = f"Control mode {self.control} and target mode {self.target} must be different"
            raise ValueError(msg)

        if self.gain == 1:
            self.label = f"CS{self.target},{self.control}"
        else:
            self.label = f"CS{self.target},{self.control}({self.gain:.2f})"
        super().__post_init__()

    def __repr__(self) -> str:
        """Return string representation of the controlled-sum gate.

        Returns:
        -------
        str
            String showing the control/target modes and gain (if not 1).
        """
        if self.gain == 1:
            return f"ControlledSumGate(CS{self.target},{self.control})"
        return f"ControlledSumGate(CS{self.target},{self.control}, g={self.gain:.2f})"


@dataclass
class ControlledZGate(CompactDiagram):
    r"""Controlled-Z (CZ) gate with gain g.

    Represents the operation exp(-i g q̂₁ q̂₂).

    Parameters
    ----------
    gain : float
        Gain parameter g. Default is 1 (unbiased CZ).

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.5, Eq. (63)-(64)
    """

    gain: float = 1.0
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
        """
        # Fourier diagram
        fourier_inv = FourierInv()

        q_spider1 = QSpider(2, 1, ZxPoly({}))
        q_spider2 = QSpider(1, 2, ZxPoly({}))
        identity = QSpider(1, 1, ZxPoly({}))
        i_tensor_f = TensorDiagram([fourier_inv, identity])
        if self.gain == 1:
            # Identity on mode 1: a q-spider with zero phase
            tensor = TensorDiagram([q_spider1, i_tensor_f.compose(q_spider2)])
            # Contract: I1 → I2 (q output 0 to p input 0, q output 1 to p input 1)
            tensor.partial_trace([
                (0, [], [1]),
                (1, [0], []),
            ])

            return tensor.diagrams[0]
        # Biased CSUM: squeeze then unbiased then unsqueeze
        sqrt_gain = np.sqrt(self.gain)
        inv_sqrt = 1 / sqrt_gain

        squeeze1 = SqueezingGate(tau=sqrt_gain)
        squeeze1_id = squeeze1.tensor(identity)
        upper_diagram = q_spider1.compose(squeeze1_id)
        squeeze2 = SqueezingGate(tau=inv_sqrt)
        upper_diagram = squeeze2.compose(upper_diagram)

        tensor = TensorDiagram([upper_diagram, i_tensor_f.compose(q_spider2)])
        tensor.partial_trace([
            (0, [], [1]),
            (1, [0], []),
        ])

        return tensor.diagrams[0]

    def conjugate(self) -> "ControlledZGate":
        """Conjugate of CZ is CZ with same gain (self-adjoint).

        Returns:
        -------
        ControlledZGate
            CZ(g)
        """
        return ControlledZGate(gain=self.gain)

    def __post_init__(self) -> None:
        """Initialize the ControlZ Gate."""
        if self.gain == 1:
            self.label = "CZ"
        else:
            self.label = f"CZ({self.gain:.2f})"
        super().__post_init__()

    def __repr__(self) -> str:
        """Return string representation of the controlled-Z gate.

        Returns:
        -------
        str
            String showing the gain if not 1.
        """
        if self.gain == 1:
            return "ControlledZGate()"
        return f"ControlledZGate(g={self.gain:.2f})"


@dataclass
class BeamsplitterGate(CompactDiagram):
    r"""Beamsplitter gate BS(θ).

    Represents the operation exp(-iθ (q̂₁ p̂₂ - p̂₁ q̂₂)). Decomposes into
    squeezing gates and CSUM gates as shown in [3] Eq. (66).

    Parameters
    ----------
    theta : float
        Beamsplitter angle. θ = π/4 gives a 50:50 beamsplitter.

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.6, Eq. (66)-(67)
    """

    theta: float
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
        """
        if np.isclose(self.theta, np.pi / 4):
            # Balanced beamsplitter [3] Eq. (67)
            sqrt2 = np.sqrt(2)
            inv_sqrt2 = 1 / sqrt2

            csum12 = ControlledSumGate(gain=1, control=1, target=2)
            csum21 = ControlledSumGate(gain=1)

            tensor = TensorDiagram([SqueezingGate(tau=sqrt2), SqueezingGate(tau=inv_sqrt2)])

            return CompositionDiagram([expand_all(csum12), tensor, expand_all(csum21)])
        # General beamsplitter - simplified representation
        # Full decomposition from [3] Appendix A.1.f
        tan_theta = np.tan(self.theta)
        sin2_theta = np.sin(2 * self.theta)
        cos_theta = np.cos(self.theta)

        sq1 = SqueezingGate(tau=1 / tan_theta)
        sq2 = SqueezingGate(tau=sin2_theta / cos_theta)
        sq3 = SqueezingGate(tau=1 / cos_theta)
        identity = QSpider(1, 1, ZxPoly({}))
        tensor1 = sq1.tensor(identity)
        tensor2 = sq2.tensor(sq3)

        csum12 = ControlledSumGate(gain=1, control=1, target=2)
        csum21 = ControlledSumGate(gain=1)

        return CompositionDiagram([tensor1, expand_all(csum12), tensor2, expand_all(csum21), tensor1])

    def conjugate(self) -> "BeamsplitterGate":
        """Conjugate of beamsplitter is beamsplitter with negated angle.

        Returns:
        -------
        BeamsplitterGate
            BS(-θ)
        """
        return BeamsplitterGate(theta=-self.theta)

    def __post_init__(self) -> None:
        """Initialize the Beamsplitter Gate."""
        if np.isclose(self.theta, np.pi / 4):
            self.label = "BS(π/4)"
        else:
            self.label = f"BS({self.theta:.2f})"
        super().__post_init__()

    def __repr__(self) -> str:
        """Return string representation of the beamsplitter gate.

        Returns:
        -------
        str
            String showing the beamsplitter angle theta.
        """
        if np.isclose(self.theta, np.pi / 4):
            return "BeamsplitterGate(π/4)"
        return f"BeamsplitterGate(θ={self.theta:.2f})"


@dataclass
class CubicPhaseGate(CompactDiagram):
    r"""Cubic phase gate CPG(γ).

    Represents the non-Gaussian operation exp(iγ x̂³). This is a native
    non-Gaussian gate represented by a single q-spider with cubic phase.

    Parameters
    ----------
    gamma : float
        Cubic phase strength parameter.

    References:
    ----------
    [3] Nagayoshi et al., CV ZX calculus, Sec. II.C.7, Eq. (68)
    """  # noqa: RUF002

    gamma: float
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
            q-spider with phase function f(x) = γ x³.
        """  # noqa: RUF002
        return QSpider(1, 1, ZxPoly({3: self.gamma}))

    def conjugate(self) -> "CubicPhaseGate":
        """Conjugate of cubic phase gate is cubic phase with negated gamma.

        Returns:
        -------
        CubicPhaseGate
            CPG(-γ)
        """  # noqa: RUF002
        return CubicPhaseGate(gamma=-self.gamma)

    def __post_init__(self) -> None:
        """Initialize the CubicPhase Gate."""
        self.label = f"CPG({self.gamma:.2f})"
        super().__post_init__()

    def __repr__(self) -> str:
        """Return string representation of the cubic phase gate.

        Returns:
        -------
        str
            String showing the cubic phase strength gamma.
        """
        return f"CubicPhaseGate(γ={self.gamma:.2f})"  # noqa: RUF001


# =============================================================================
# Helper functions
# =============================================================================


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
    """Recursively expand all CompactDiagram instances in a diagram.

    Parameters:
    ----------
    diagram : Diagram
        The diagram to expand.

    Returns:
    -------
    Diagram
        The expanded diagram with all CompactDiagram expanded.
    """
    if isinstance(diagram, CompactDiagram):
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

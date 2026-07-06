"""CV-ZX representation of quantum gates from Nagayoshi et al. (2024), Sec. II.C.

This module implements the standard CV quantum gates as composite diagrams
built from proper diagrams (spiders, Fourier, Swap). Each gate is a subclass
of ProperDiagram and provides a decompose() method that returns the
equivalent CompositionDiagram or TensorDiagram of basic CV ZX elements.

References:
-----------
[3] Nagayoshi et al., CV ZX calculus, Sec. II.C, Table I
"""

from dataclasses import dataclass

import numpy as np

from mqc3.zx.base_gates import (
    CompositionDiagram,
    Diagram,
    Fourier,
    ProperDiagram,
    PSpider,
    QSpider,
    TensorDiagram,
    ZxPoly,
)


@dataclass
class DisplacementGate(ProperDiagram):
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
    """

    alpha: complex

    def __post_init__(self) -> None:
        """Initialize the Displacement Gate."""
        self._num_inputs = 1
        self._num_outputs = 1

    def decompose(self) -> CompositionDiagram:
        """Decompose displacement gate into q-spider and p-spider.

        D(α) = Q(√2 Im(α) x) ∘ P(√2 Re(α) x)

        Returns:
        -------
        CompositionDiagram
            Composition of p-spider then q-spider.
        """
        re = np.sqrt(2) * self.alpha.real
        im = np.sqrt(2) * self.alpha.imag

        q_spider = QSpider(1, 1, ZxPoly({1: im}))
        p_spider = PSpider(1, 1, ZxPoly({1: re}))

        # D(α) = Q ∘ P  (P applied first, then Q)
        return CompositionDiagram([p_spider, q_spider])

    def conjugate(self) -> "DisplacementGate":
        """Conjugate of displacement gate is displacement with negated alpha.

        Returns:
        -------
        DisplacementGate
            D(-α)
        """
        return DisplacementGate(alpha=-self.alpha)


@dataclass
class PhaseRotationGate(ProperDiagram):
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

    def __post_init__(self) -> None:
        """Initialise the Phase Rotation Gate.

        Raises:
        ------
        ValueError: If the angle is a multiple of π or of π/2.
        """
        self._num_inputs = 1
        self._num_outputs = 1

        # Check for invalid angles where tan is infinite
        if np.isclose(np.abs(self.theta) % np.pi, np.pi / 2):
            msg = (
                f"θ = {self.theta} is an odd multiple of π/2. "
                f"For π/2 rotation, use Fourier gate. For 3π/2, use FourierInv."
            )
            raise ValueError(msg)

    def decompose(self) -> CompositionDiagram:
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


@dataclass
class SqueezingGate(ProperDiagram):
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

    def __post_init__(self) -> None:
        """Initialize the Squeezing Gate."""
        self._num_inputs = 1
        self._num_outputs = 1

    def decompose(self) -> CompositionDiagram:
        """Decompose squeezing gate into four quadratic spiders.

        Sq(τ) = Q(a) ∘ P(b) ∘ Q(c) ∘ P(d)
        where a = τ(1-τ)/4, b = -1/τ, c = (τ-1)/4, d = 1

        Returns:
        -------
        CompositionDiagram
            Composition of Q, P, Q, P spiders in sequence.
        """
        a = self.tau * (1 - self.tau) / 4
        b = -1 / self.tau if abs(self.tau) > 1e-10 else 0
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


@dataclass
class ControlledSumGate(ProperDiagram):
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

    def __post_init__(self) -> None:
        """Initializes the ControlledSum Gate.

        Raises:
        ------
        ValueError: If the control is equal to the target.
        """
        self._num_inputs = 2
        self._num_outputs = 2

        if self.control == self.target:
            msg = f"Control mode {self.control} and target mode {self.target} must be different"
            raise ValueError(msg)

    def decompose(self) -> Diagram:
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
            # q-spider (1→2) connected to p-spider (2→1)
            tensor = TensorDiagram([control_spider, target_spider])

            # Contract: q output 0 → p input 0 (forward)
            #          q output 1 → p input 1 (forward)
            #          p output 0 → q input 0 (feedback)
            tensor.partial_trace([
                (0, [0], []),  # q-spider: outputs 0,1 → input 0 (feedback)
                (1, [], [1]),  # p-spider: output 0 → inputs 0,1 (forward)
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
            (0, [0], []),
            (1, [], [1]),
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


@dataclass
class ControlledZGate(ProperDiagram):
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

    def __post_init__(self) -> None:
        """Initialize the ControlZ Gate."""
        self._num_inputs = 2
        self._num_outputs = 2

    def decompose(self) -> CompositionDiagram:
        """Decompose CZ gate using Fourier gates and CSUM.

        Unbiased CZ [3] Eq. (63):
            CZ = ContractedDiagram of q-spider and p-spider with
            a fourier diagram in between

        Biased CZ [3] Eq. (64):
            CZ(g) = (Sq(g⁻¹) ⊗ Id) ∘ CZ(1) ∘ (Sq(g) ⊗ Id)

        Returns:
        -------
        CompositionDiagram
            Composition of Fourier, CSUM, Fourier.
        """
        # Fourier diagram
        fourier = Fourier()

        q_spider1 = QSpider(1, 2, ZxPoly({}))
        q_spider2 = QSpider(2, 1, ZxPoly({}))
        if self.gain == 1:
            # Identity on mode 1: a q-spider with zero phase
            identity = QSpider(1, 1, ZxPoly({}))
            i_tensor_f = TensorDiagram([identity, fourier])
            tensor = TensorDiagram([i_tensor_f.compose(q_spider1), q_spider2])

            # Contract: I1 → I2 (q output 0 to p input 0, q output 1 to p input 1)
            tensor.partial_trace([
                (0, [0], []),
                (1, [], [1]),
            ])

            return tensor.diagrams[0]
        # Biased CSUM: squeeze then unbiased then unsqueeze
        sqrt_gain = np.sqrt(self.gain)
        inv_sqrt = 1 / sqrt_gain

        identity = QSpider(1, 1, ZxPoly({}))
        squeeze1 = SqueezingGate(tau=sqrt_gain)
        upper_diagram = q_spider1.compose(squeeze1)
        squeeze2 = SqueezingGate(tau=inv_sqrt)
        squeeze2_id = squeeze2.tensor(fourier)
        upper_diagram = squeeze2_id.compose(upper_diagram)

        tensor = TensorDiagram([upper_diagram, q_spider2])
        tensor.partial_trace([
            (0, [0], []),
            (1, [], [1]),
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


@dataclass
class BeamsplitterGate(ProperDiagram):
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

    def __post_init__(self) -> None:
        """Initialize the Beamsplitter Gate."""
        self._num_inputs = 2
        self._num_outputs = 2

    def decompose(self) -> Diagram:
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

            return CompositionDiagram([csum12, tensor, csum21])
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

        return CompositionDiagram([tensor1, csum12, tensor2, csum21, tensor1])

    def conjugate(self) -> "BeamsplitterGate":
        """Conjugate of beamsplitter is beamsplitter with negated angle.

        Returns:
        -------
        BeamsplitterGate
            BS(-θ)
        """
        return BeamsplitterGate(theta=-self.theta)


@dataclass
class CubicPhaseGate(ProperDiagram):
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
    """

    gamma: float

    def __post_init__(self) -> None:
        """Initialize the CubicPhase Gate."""
        self._num_inputs = 1
        self._num_outputs = 1

    def decompose(self) -> QSpider:
        """Decompose cubic phase gate into a single q-spider with cubic phase.

        Returns:
        -------
        QSpider
            q-spider with phase function f(x) = γ x³.
        """
        return QSpider(1, 1, ZxPoly({3: self.gamma}))

    def conjugate(self) -> "CubicPhaseGate":
        """Conjugate of cubic phase gate is cubic phase with negated gamma.

        Returns:
        -------
        CubicPhaseGate
            CPG(-γ)
        """
        return CubicPhaseGate(gamma=-self.gamma)

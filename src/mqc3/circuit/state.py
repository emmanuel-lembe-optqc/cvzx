"""State class for modes in quantum circuit."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from mqc3.constant import hbar
from mqc3.pb.mqc3_cloud.common.v1.math_pb2 import Complex as PbComplex
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import BosonicState as PbBosonicState
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import GaussianState as PbGaussianState
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import (
    HardwareConstrainedSqueezedState as PbHardwareConstrainedSqueezedState,
)
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import InitialState

PbInitialState = InitialState  # Alias for better readability

if TYPE_CHECKING:
    from numpy.typing import NDArray


class State(ABC):
    """Abstract State class."""

    @abstractmethod
    def n_modes(self) -> int:
        """Get the number of modes in the State.

        Returns:
            int: The number of modes.
        """
        raise NotImplementedError


class GaussianState(State):
    """Gaussian state class."""

    _mean: NDArray[np.complex128]
    """Mean vector."""

    _cov: NDArray[np.float64]
    """Covariance matrix."""

    def __init__(
        self,
        mean: NDArray[np.complex128],
        cov: NDArray[np.float64],
    ) -> None:
        """Construct a GaussianState object.

        The mean is 2 x N complex vector, where N is the number of modes.
        The mean contains the following values in order.

        - The x-mean of modes 0, 1, ..., N-1.
        - The p-mean of modes 0, 1, ..., N-1.

        The covariance matrix (cov) is a 2N x 2N real-valued matrix.
        The covariance matrix (cov) must be symmetric and positive definite.

        Args:
            mean (NDArray[np.complex128]): Mean vector.
            cov (NDArray[np.float64]): Covariance matrix.

        Raises:
            ValueError: If the shapes of mean and cov are not compatible.
        """
        if mean.shape[0] % 2 != 0:
            msg = "The argument `mean` must be a vector with an even number of elements."
            raise ValueError(msg)
        if not (cov.shape[0] == cov.shape[1] == mean.shape[0]):
            msg = "The dimensions of `cov` must match those of `mean`."
            raise ValueError(msg)
        self._mean = np.fromiter(mean, dtype=np.complex128)
        self._cov = np.array(cov, dtype=np.float64)

    @classmethod
    def vacuum(cls) -> GaussianState:
        r"""Construct a vacuum state.

        A vacuum state is assumed to be a single-mode Gaussian state
        with a zero mean vector and a covariance matrix given by:

        .. math::

            \frac{\hbar}{2} I_2

        where :math:`I_2` is the 2x2 identity matrix.

        In MQC3, we assume :math:`\hbar = 1` (see :data:`mqc3.constant.hbar`).

        Returns:
            GaussianState: The vacuum state as a single-mode GaussianState.
        """
        return GaussianState(
            mean=np.zeros(2, dtype=np.complex128),
            cov=0.5 * hbar * np.eye(2, dtype=np.float64),
        )

    @classmethod
    def squeezed(cls, r: float, phi: float) -> GaussianState:
        r"""Construct a single-mode squeezed state.

        A squeezed state is a Gaussian state characterized by the following mean vector and covariance matrix:

        - Mean vector: Zero vector.
        - Covariance matrix:

          .. math::

              \frac{\hbar}{2} \, R(\phi) \, \mathrm{diag}\left( e^{-2r}, e^{2r} \right) \, R(\phi)^\text{T}

        where :math:`R(\phi)` is the rotation matrix defined as:

        .. math::

            R(\phi) = \begin{bmatrix}
            \cos{\phi} & -\sin{\phi} \\
            \sin{\phi} & \cos{\phi}
            \end{bmatrix}

        and :math:`R(\phi)^\text{T}` denotes its transpose.

        In MQC3, we assume :math:`\hbar = 1` (see :data:`mqc3.constant.hbar`).

        Args:
            r (float): Squeezing parameter.
            phi (float): Squeezing angle in radians.

        Returns:
            GaussianState: A single-mode squeezed state represented as a GaussianState object.
        """
        mean = np.zeros(2, dtype=np.complex128)
        # Calculate the covariance matrix of the squeezed state.
        rotation_matrix = np.array(
            [
                [np.cos(phi), -np.sin(phi)],
                [np.sin(phi), np.cos(phi)],
            ],
            dtype=np.float64,
        )
        cov = 0.5 * hbar * rotation_matrix @ np.diag([np.exp(-2 * r), np.exp(2 * r)]) @ rotation_matrix.T

        return GaussianState(mean=mean, cov=cov)

    def is_vacuum(self) -> bool:
        """Check if the state is a vacuum state.

        Returns:
           bool: True if the GaussianState is a vacuum state, False otherwise.
        """
        return np.allclose(self.mean, np.zeros(2, dtype=np.complex128)) and np.allclose(
            self.cov,
            0.5 * hbar * np.eye(2, dtype=np.float64),
        )

    @property
    def n_modes(self) -> int:
        """Get the number of modes in the GaussianState.

        Returns:
            int: The number of modes.

        Examples:
            >>> from mqc3.circuit.state import GaussianState
            >>> state = GaussianState.vacuum()
            >>> state.n_modes
            1
        """
        return self.mean.shape[0] // 2

    @property
    def mean(self) -> NDArray[np.complex128]:
        """Get the mean vector of the GaussianState.

        Returns:
            NDArray[np.complex128]: Mean vector.
        """
        return self._mean

    @property
    def cov(self) -> NDArray[np.float64]:
        """Get the covariance matrix of the GaussianState.

        Returns:
            NDArray[np.float64]: Covariance matrix.
        """
        return self._cov

    def extract_mode(self, mode: int) -> GaussianState:
        """Extract the specified mode from the GaussianState.

        This method extracts the elements corresponding to the specified modes from the mean vector and
        covariance matrix of a GaussianState, and create a new GaussianState. However, it is not guaranteed that
        the resulting parameters will satisfy the conditions required for GaussianState.

        Args:
            mode (int): The index of the mode to extract.

        Returns:
            GaussianState: The extracted mode.

        Raises:
            ValueError: If the mode index is out of range.
        """
        if mode >= self.n_modes:
            msg = "Mode index out of range."
            raise ValueError(msg)

        mean = np.array((self._mean[mode], self._mean[mode + self.n_modes]))
        cov = np.array(
            [
                [self._cov[mode, mode], self._cov[mode, mode + self.n_modes]],
                [self._cov[mode + self.n_modes, mode], self._cov[mode + self.n_modes, mode + self.n_modes]],
            ],
        )
        return GaussianState(mean, cov)

    def __str__(self) -> str:
        """Get the parameters of the GaussianState as a string.

        Returns:
            str: Parameters of the GaussianState.

        Examples:
        >>> from mqc3.circuit.state import GaussianState
        >>> state = GaussianState.vacuum()
        >>> print(str(state))
        GaussianState(n_modes=1)
        """
        return f"GaussianState(n_modes={self.n_modes})"

    def __repr__(self) -> str:
        """Get the string representation of the GaussianState.

        Returns:
            str: String representation.

        Examples:
        >>> from mqc3.circuit.state import GaussianState
        >>> state = GaussianState.vacuum()
        >>> print(repr(state))
        GaussianState(
                mean=array([0.+0.j, 0.+0.j]),
                cov=array([[0.5, 0.],
               [0., 0.5]]))
        """
        repr_mean = np.array_repr(self.mean)
        repr_cov = np.array_repr(self.cov)
        return f"GaussianState(\n\tmean={repr_mean},\n\tcov={repr_cov})"

    def proto(self) -> PbGaussianState:  # noqa: D102
        # PbGaussianState has covariance matrix as 1 dimensional float array.
        cov = self.cov.ravel().tolist()
        mean = [PbComplex(real=mean_val.real, imag=mean_val.imag) for mean_val in self.mean]

        return PbGaussianState(mean=mean, cov=cov)

    @staticmethod
    def construct_from_proto(proto: PbGaussianState) -> GaussianState:  # noqa: D102
        # PbGaussianState has covariance matrix as 1 dimensional float array.
        # We need to reshape it to 2 dimensional array.
        dims = len(proto.mean)
        cov = np.array(proto.cov, dtype=np.float64).reshape(dims, dims)
        mean = np.fromiter([mean_val.real + mean_val.imag * 1j for mean_val in proto.mean], dtype=np.complex128)

        return GaussianState(mean=mean, cov=cov)


class BosonicState(State):
    """Bosonic state class."""

    coeffs: NDArray[np.complex128]
    """Coefficients of gaussian states."""

    gaussian_states: list[GaussianState]
    """Gaussian states."""

    @classmethod
    def vacuum(cls) -> BosonicState:
        """Construct a vacuum state.

        We assume a vacuum state is a single peak bosonic state constructed from a single-mode Gaussian vacuum state.

        Returns:
            BosonicState: Vacuum state.
        """
        return BosonicState(
            np.fromiter([1], dtype=np.complex128),
            [GaussianState.vacuum()],
        )

    @classmethod
    def squeezed(cls, r: float, phi: float) -> BosonicState:
        """Construct a squeezed state.

        We assume a squeezed state is a single peak bosonic state
        constructed from a single-mode Gaussian squeezed state.

        Args:
            r (float): Squeezing parameter.
            phi (float): Squeezing angle in radians.

        Returns:
            BosonicState: Vacuum state.
        """
        return BosonicState(
            np.fromiter([1], dtype=np.complex128),
            [GaussianState.squeezed(r=r, phi=phi)],
        )

    @classmethod
    def cat(cls, x: float, p: float, parity: float) -> BosonicState:
        r"""Construct a cat state.

        :math:`\ket{\text{cat}(\alpha)} = \frac{1}{N} (\ket{\alpha} +e^{i\phi} \ket{-\alpha})`,

        where :math:`\alpha = x + ip`, :math:`N` is a normalization constant
        , and :math:`\phi = \pi \times \text{parity}` .

        Returns:
            BosonicState: Cat state.
        """
        if np.isclose(x, 0) and np.isclose(p, 0):
            return BosonicState.vacuum()

        phase = pi * parity
        n = 0.5 / (1 + np.exp(-2 * (x**2 + p**2)) * np.cos(phase))
        alpha = x + 1j * p
        # Mean and coeff of |alpha><alpha| term.
        mu_plus = np.sqrt(2 * hbar) * np.array([alpha.real, alpha.imag], dtype=np.complex128)
        c_plus = 1
        # Mean and coeff of |alpha><-alpha| term.
        mu_minus = np.sqrt(2 * hbar) * np.array([1j * alpha.imag, -1j * alpha.real], dtype=np.complex128)
        c_minus = np.exp(-2 * (x**2 + p**2) - 1j * phase)
        cov = GaussianState.vacuum().cov

        return BosonicState(
            n * np.array([c_plus, c_plus, c_minus, c_minus.conjugate()], dtype=np.complex128),
            [
                GaussianState(mu_plus, cov),
                GaussianState(-mu_plus, cov),
                GaussianState(mu_minus, cov),
                GaussianState(np.conjugate(mu_minus), cov),
            ],
        )

    def is_vacuum(self) -> bool:
        """Check if the BosonicState is a vacuum state.

        Returns:
            bool: True if the BosonicState is a vacuum state, False otherwise.
        """
        return len(self.gaussian_states) == 1 and self.gaussian_states[0].is_vacuum()

    def __init__(self, coeffs: NDArray[np.complex128], gaussian_states: list[GaussianState]) -> None:
        """Construct a BosonicState object.

        We assume the bosonic state is a superposition of Gaussian states.
        BosonicState must satisfy the following constraints:

        - All GaussianState must have the same number of modes.
        - Sum of coefficients must be 1.

        Args:
            coeffs (NDArray[np.complex128]): Coefficients of Gaussian states.
            gaussian_states (list[GaussianState]): Gaussian states.

        Raises:
            ValueError : If one of the following 3 cases occurs:

                1. The shape of coeffs, means, and covs are not compatible.
                2. The number of modes of all GaussianState is not the same.
                3. The sum of coefficients is not 1.
        """
        # Check if the shape of coeffs, means, and covs are compatible.
        if coeffs.shape[0] != len(gaussian_states):
            msg = "The lengths of the arguments `coeffs` and `gaussian_states` must match."
            raise ValueError(msg)

        # Sum of coefficients must be 1.
        if not np.isclose(np.sum(coeffs), 1.0):
            msg = "The sum of the `coeffs` must be 1."
            raise ValueError(msg)

        # All GaussianState must have the same number of modes.
        state_modes = gaussian_states[0].n_modes
        if any(peak.n_modes != state_modes for peak in gaussian_states):
            msg = "All `GaussianState` instances must have the same number of modes."
            raise ValueError(msg)

        self.coeffs = np.fromiter(coeffs, dtype=np.complex128)
        self.gaussian_states = deepcopy(gaussian_states)

    @property
    def n_modes(self) -> int:
        """Get the number of modes in the BosonicState.

        Returns:
            int: The number of modes.

        Examples:
            >>> from mqc3.circuit.state import BosonicState
            >>> state = BosonicState.vacuum()
            >>> state.n_modes
            1
        """
        return self.gaussian_states[0].n_modes

    @property
    def n_peaks(self) -> int:
        """Get the number of peaks of the BosonicState.

        Returns:
            int: The number of peaks.
        """
        return self.coeffs.shape[0]

    def get_coeff(self, i: int) -> np.complex128:
        """Get the coefficient of the i-th GaussianState.

        Args:
            i (int): Index of the GaussianState.

        Returns:
            np.complex128: The coefficient of the i-th GaussianState.
        """
        return self.coeffs[i]

    def get_gaussian_state(self, i: int) -> GaussianState:
        """Get the i-th GaussianState.

        Args:
            i (int): Index of the GaussianState.

        Returns:
            GaussianState: The i-th GaussianState.
        """
        return self.gaussian_states[i]

    def extract_mode(self, mode: int) -> BosonicState:
        """Extract a single mode from the BosonicState.

        This method extracts the elements corresponding to the specified modes from the mean vector and
        covariance matrix of a BosonicState, and create a new BosonicState. However, it is not guaranteed that
        the resulting parameters will satisfy the conditions required for BosonicState.

        Args:
            mode (int): The index of the mode to extract.

        Returns:
            BosonicState: A new BosonicState containing only the specified mode.
        """
        return BosonicState(self.coeffs, [gauss.extract_mode(mode) for gauss in self.gaussian_states])

    def __str__(self) -> str:
        """Get the parameters of the BosonicState as a string.

        Returns:
            str: String representation.

        Examples:
            >>> from mqc3.circuit.state import BosonicState
            >>> gaussian_state = GaussianState.vacuum()
            >>> state = BosonicState([1.0], [gaussian_state])
            >>> repr(state)
            BosonicState(n_modes=1)
        """
        return f"BosonicState(n_modes={self.n_modes})"

    def __repr__(self) -> str:
        r"""Get the string representation of the BosonicState.

        Returns:
            str: String representation.

        Examples:
            >>> from mqc3.circuit.state import BosonicState
            >>> gaussian_state = GaussianState.vacuum()
            >>> state = BosonicState([1.0], [gaussian_state])
            >>> repr(state)
            BosonicState(
                    coeffs=array([1.+0.j]),
                    gaussian_states=[GaussianState(
                    mean=array([0.+0.j, 0.+0.j]),
                    cov=array([[0.5, 0.],
                   [0., 0.5]]))])
        """
        repr_coeffs = np.array_repr(self.coeffs)
        repr_gaussian_states = "[{repr_states}]".format(
            repr_states=",\n".join(map(repr, self.gaussian_states)),
        )
        return f"BosonicState(\n\tcoeffs={repr_coeffs},\n\tgaussian_states={repr_gaussian_states})"

    def proto(self) -> PbBosonicState:  # noqa: D102
        gaussian_states = [gaussian_state.proto() for gaussian_state in self.gaussian_states]
        coeffs = [PbComplex(real=coeff.real, imag=coeff.imag) for coeff in self.coeffs]

        return PbBosonicState(gaussian_states=gaussian_states, coeffs=coeffs)

    @staticmethod
    def construct_from_proto(proto: PbBosonicState) -> BosonicState:  # noqa: D102
        coeffs = np.fromiter([coeff.real + coeff.imag * 1j for coeff in proto.coeffs], dtype=np.complex128)
        gaussian_states = []
        for pb_gaussian_state in proto.gaussian_states:
            gaussian_state = GaussianState.construct_from_proto(pb_gaussian_state)
            gaussian_states.append(gaussian_state)

        return BosonicState(coeffs, gaussian_states)


@dataclass
class HardwareConstrainedSqueezedState:
    """A squeezed state for hardware execution with a fixed squeezing level.

    Only the squeezing angle 'phi' is user-controllable due to hardware constraints.

    The squeezing level (magnitude) is determined by the physical hardware
    (e.g., pump power, OPO design) and cannot be changed at runtime.
    This class represents the states that your actual photonic hardware can produce.

    Attributes:
        phi: Squeezing angle in radians. Determines the squeezing direction.
             x-squeezed states correspond to phi=0 (amplitude squeezing).
             p-squeezed states correspond to phi=π/2 (phase squeezing).
        fixed_squeezing_db: The hardware's fixed squeezing level in decibels.
                            This can be set per instance or use the class default.
    """

    phi: float = 0.0
    r"""Squeezing angle :math:`\phi` [radians].

    Determines the squeezing direction in phase space.
    Assuming that the squeezing level is positive,
    x-squeezed states correspond to :math:`\phi=0`, while p-squeezed states correspond to :math:`\phi=\pi/2`.
    """

    fixed_squeezing_db: float = 10.0  # 10 dB fixed squeezing (configurable per instance)

    # Class-level hardware constant for default squeezing when not specified
    DEFAULT_SQUEEZING_DB: ClassVar[float] = 10.0

    def __post_init__(self) -> None:
        """Validate the squeezing angle."""  # noqa: DOC501
        if not (0.0 <= self.phi <= pi):
            msg = f"Phi must be in [0, π], got {self.phi}"
            raise ValueError(msg)
        if self.fixed_squeezing_db < 0:
            msg_0 = f"Squeezing level cannot be negative: {self.fixed_squeezing_db} dB"
            raise ValueError(msg_0)

    def _get_r_parameter(self) -> float:
        """Convert squeezing level in dB to the squeezing parameter r.

        Formula: squeezing (dB) = -10 * log10(e^{-2r})
        Therefore: r = (squeezing_db / 10) * (ln(10) / 2)

        Returns:
            Squeezing parameter r (dimensionless)
        """
        return (self.fixed_squeezing_db / 10.0) * (np.log(10.0) / 2.0)

    def to_bosonic_state(self) -> BosonicState:
        """Convert this hardware-constrained squeezed state to MQC3 BosonicState.

        Uses the hardware's fixed squeezing level and the user-provided angle.
        Delegates to MQC3's native BosonicState.squeezed() factory method.

        Returns:
            BosonicState with single peak (pure Gaussian state) ready for MQC3.
        """
        r = self._get_r_parameter()
        # MQC3's BosonicState.squeezed() expects (r, phi) where:
        #   r is the squeezing parameter
        #   phi is the squeezing angle in radians
        return BosonicState.squeezed(r=r, phi=self.phi)

    @classmethod
    def from_gaussian_state(
        cls, gaussian_state: GaussianState, fixed_squeezing_db: float | None = None
    ) -> HardwareConstrainedSqueezedState:
        """Convert a GaussianState back to HardwareConstrainedSqueezedState.

        This is useful for extracting hardware parameters from simulation results.

        Args:
            gaussian_state: MQC3 GaussianState (must be a squeezed state)
            fixed_squeezing_db: Hardware's fixed squeezing level (if None, inferred)

        Returns:
            HardwareConstrainedSqueezedState with extracted phi angle.
        """
        # Extract the covariance matrix
        cov = gaussian_state.cov

        # Find the eigenvector with smaller eigenvalue (squeezed quadrature)
        eigvals, eigvecs = np.linalg.eigh(cov)
        squeezed_idx = np.argmin(eigvals)
        phi = np.arctan2(eigvecs[1, squeezed_idx], eigvecs[0, squeezed_idx])

        # If fixed_squeezing_db not provided, infer from covariance
        if fixed_squeezing_db is None:
            # For a pure squeezed state: variance_min = (ħ/2) * e^{-2r}
            variance_min = eigvals[squeezed_idx]
            r = 0.5 * np.log((hbar / 2.0) / variance_min)
            fixed_squeezing_db = 20.0 * r * np.log10(np.e)
            # Clamp to reasonable values
            fixed_squeezing_db = max(0.0, min(20.0, fixed_squeezing_db))

        return cls(phi=float(phi), fixed_squeezing_db=fixed_squeezing_db)

    @classmethod
    def from_bosonic_state(
        cls, bosonic_state: BosonicState, fixed_squeezing_db: float | None = None, peak_index: int = 0
    ) -> HardwareConstrainedSqueezedState:
        """Convert a BosonicState back to HardwareConstrainedSqueezedState.

        This is the inverse of to_bosonic_state().

        Args:
            bosonic_state: MQC3 BosonicState (must have at least one peak)
            fixed_squeezing_db: Hardware's fixed squeezing level (if None, inferred)
            peak_index: Which peak to extract (default 0)

        Returns:
            HardwareConstrainedSqueezedState with extracted phi angle.

        Raises:
            ValueError: If peak_index is out of range.
        """
        if bosonic_state.n_peaks <= peak_index:
            msg = f"Peak index {peak_index} out of range (n_peaks={bosonic_state.n_peaks})"
            raise ValueError(msg)

        gaussian_state = bosonic_state.get_gaussian_state(peak_index)
        return cls.from_gaussian_state(gaussian_state, fixed_squeezing_db)

    def proto(self) -> PbHardwareConstrainedSqueezedState:
        """Convert to Protocol Buffer representation.

        Returns:
            PbHardwareConstrainedSqueezedState: The protocol buffer representation.
        """
        return PbHardwareConstrainedSqueezedState(theta=pi / 2 - self.phi)

    @staticmethod
    def construct_from_proto(proto: PbHardwareConstrainedSqueezedState) -> HardwareConstrainedSqueezedState:
        """Construct from Protocol Buffer representation.

        Returns:
            HardwareConstrainedSqueezedState: The constructed squeezed state.
        """
        return HardwareConstrainedSqueezedState(phi=pi / 2 - proto.theta)

    def get_squeezing_parameters(self) -> tuple[float, float]:
        """Get the physical squeezing parameters.

        Returns:
            Tuple of (squeezing_level_dB, squeezing_angle_radians)
        """
        return (self.fixed_squeezing_db, self.phi)

    def get_quadrature_variances(self) -> tuple[float, float]:
        """Calculate the variances of the squeezed and anti-squeezed quadratures.

        Returns:
            Tuple of (min_variance, max_variance) with respect to hbar.
            For a vacuum state, variance = hbar/2.
        """
        r = self._get_r_parameter()
        var_min = (hbar / 2.0) * np.exp(-2 * r)
        var_max = (hbar / 2.0) * np.exp(2 * r)
        return (var_min, var_max)

    def __repr__(self) -> str:
        """Return a string representation of the squeezed state."""
        return f"HardwareConstrainedSqueezedState(phi={self.phi:.3f} rad, squeezing={self.fixed_squeezing_db:.1f} dB)"


def construct_proto_from_initial_state(initial_state: InitialState) -> PbInitialState:
    """Construct a protobuf message from a InitialState.

    Args:
        initial_state (InitialState): InitialState object.

    Returns:
        PbInitialState: InitialState in proto format.
    """
    bosonic_state = None
    hcss_state = None
    if isinstance(initial_state, BosonicState):
        bosonic_state = initial_state.proto()
    if isinstance(initial_state, HardwareConstrainedSqueezedState):
        hcss_state = initial_state.proto()
    return PbInitialState(squeezed=hcss_state, bosonic=bosonic_state)


def construct_initial_state_from_proto(proto: PbInitialState) -> InitialState | None:
    """Construct a InitialState object from a protobuf message.

    Args:
        proto (PbInitialState): Protobuf message.

    Returns:
        InitialState | None: InitialState object or None if empty BosonicState in proto format.
    """
    if proto.HasField("squeezed"):
        return HardwareConstrainedSqueezedState.construct_from_proto(proto.squeezed)

    return BosonicState.construct_from_proto(proto.bosonic)

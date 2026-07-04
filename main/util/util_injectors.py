# util_injectors.py
"""
Contains particle conversion, delivery definitions, and more
"""

import abc
import numpy as np
import scipy.constants as _sc

def torr_l_to_particles(V_torr_l: float, T_K: float = 293.15) -> float:
    """Convert a gas amount from Torr-L to number of particles.

    Applies the ideal gas law: PV = NkT

    NOTE: Multiply by 2 for deuterium

    Parameters
    ----------
    quant   : float, Pressure * volume [torr-L]
    T_K     : float, optional. Gas temperature [K]. Default: 293.15
    """

    PV = V_torr_l * _sc.torr * 1.0e-3 # [Torr -> Pa] * [L -> m^3] = Pa m^3
    return PV / (_sc.k * T_K)


# ---------------------------------------------------------------------------
# ---- Delivery Profiles
# ---------------------------------------------------------------------------
class DeliveryProfile(abc.ABC):
    """Normalized time-delivery profile, multiplying by shape(t)
    by a total particle count gives a rate
    """

    @abc.abstractmethod
    def shape(self, t: float) -> float:
        """Normalized rate at a time t [ms-1]"""
        ...

    @abc.abstractmethod
    def first_light_time(self, threshold_frac: float = 0.01) -> float:
        """Time when cumulative delivery first exceeds threshold_frac"""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class GaussianProfile(DeliveryProfile):
    """Gaussian delivery profile"""

    def __init__(self, t_peak: float, dt_pulse: float) -> None:
        self.t_peak = float(t_peak)
        self.dt = float(dt_pulse)
        self._norm = 1.0 / (np.sqrt(np.pi) * self.dt)

    def shape(self, t: float) -> float:
        return self._norm * np.exp(-(((t - self.t_peak) / self.dt) ** 2))

    def first_light_time(self, threshold_frac: float = 0.01) -> float:
        return self.t_peak - self.dt * np.sqrt(-np.log(threshold_frac))

    def __repr__(self) -> str:
        return f"GaussianProfile(t_peak={self.t_peak}, dt={self.dt})"
    

class SquareProfile(DeliveryProfile):
    """Top-hat: constant delivery on [t_start, t_end]."""

    def __init__(self, t_start: float, t_end: float) -> None:
        if t_end <= t_start:
            raise ValueError("SquareProfile requires t_end > t_start")
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self._norm = 1.0 / (self.t_end - self.t_start)

    def shape(self, t: float) -> float:
        return self._norm if self.t_start <= t <= self.t_end else 0.0

    def first_light_time(self, threshold_frac: float = 0.01) -> float:
        return self.t_start

    def __repr__(self) -> str:
        return f"SquareProfile(t_start={self.t_start}, t_end={self.t_end})"


class ExponentialProfile(DeliveryProfile):
    """Instant onset at t_start, then exponential decay with time-constant tau."""

    def __init__(self, t_start: float, tau: float) -> None:
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.t_start = float(t_start)
        self.tau = float(tau)

    def shape(self, t: float) -> float:
        return np.exp(-(t - self.t_start) / self.tau) / self.tau if t >= self.t_start else 0.0

    def first_light_time(self, threshold_frac: float = 0.01) -> float:
        return self.t_start

    def __repr__(self) -> str:
        return f"ExponentialProfile(t_start={self.t_start}, tau={self.tau})"

_PROFILE_REGISTRY: dict[str, type] = {
    "gaussian":    GaussianProfile,
    "square":      SquareProfile,
    "exponential": ExponentialProfile,
}


def make_profile(name: str, **kwargs) -> DeliveryProfile:
    """Build a DeliveryProfile by name.

    >>> make_profile("gaussian",    t_peak=5.0, dt_pulse=1.8)
    >>> make_profile("square",      t_start=4.0, t_end=7.0)
    >>> make_profile("exponential", t_start=4.0, tau=1.0)
    """
    key = name.lower()
    if key not in _PROFILE_REGISTRY:
        raise ValueError(f"Unknown profile {name!r}. Options: {sorted(_PROFILE_REGISTRY)}")
    return _PROFILE_REGISTRY[key](**kwargs)



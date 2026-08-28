"""Analytic smootherstep-windowed Gaussian deformation primitives."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import erf

from eos_generation.bsk24.reconstruction import (
    BSk24ConsistentBaseline,
    _mass_density_from_energy_density,
)

if TYPE_CHECKING:
    from eos_generation.bsk24.deformation import BSk24WindowedDeformation


PURE_GAUSSIAN_GENERATOR_ID = "pure_gaussian_v1"
WINDOWED_GAUSSIAN_GENERATOR_ID = "windowed_gaussian_v1"
PRIMARY_EPSILON0_MEV_FM3 = 200.0
PRIMARY_SIGMA_MEV_FM3 = 50.0
PRIMARY_DELTA_MEV_FM3 = 40.0
CONTROL_DELTAS_MEV_FM3 = (30.0, 45.0)
PRIMARY_AMPLITUDES = (0.0, 0.01, -0.01, 0.025, -0.025, 0.05, -0.05)
BSK24_RETAINED_EPSILON_MATCH_MEV_FM3 = 152.4912472062717
BSK24_RETAINED_EPSILON_MAX_MEV_FM3 = 1508.9793344234


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def smootherstep_window(
    energy_density_mev_fm3: Any,
    *,
    epsilon_t_mev_fm3: float,
    delta_mev_fm3: float,
) -> float | np.ndarray:
    """Return the exact piecewise quintic smootherstep window."""
    if not np.isfinite([epsilon_t_mev_fm3, delta_mev_fm3]).all():
        raise ValueError("window parameters must be finite")
    if delta_mev_fm3 <= 0.0:
        raise ValueError("Delta must be positive")
    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    if not np.all(np.isfinite(epsilon)):
        raise ValueError("energy density must be finite")
    result = np.zeros_like(epsilon)
    upper = epsilon_t_mev_fm3 + delta_mev_fm3
    ramp = (epsilon > epsilon_t_mev_fm3) & (epsilon < upper)
    x = (epsilon[ramp] - epsilon_t_mev_fm3) / delta_mev_fm3
    result[ramp] = x**3 * (10.0 + x * (-15.0 + 6.0 * x))
    result[epsilon >= upper] = 1.0
    return _scalar_or_array(result)


def smootherstep_window_first_derivative(
    energy_density_mev_fm3: Any,
    *,
    epsilon_t_mev_fm3: float,
    delta_mev_fm3: float,
) -> float | np.ndarray:
    """Return dW/d-epsilon for the exact piecewise smootherstep."""
    if not np.isfinite([epsilon_t_mev_fm3, delta_mev_fm3]).all():
        raise ValueError("window parameters must be finite")
    if delta_mev_fm3 <= 0.0:
        raise ValueError("Delta must be positive")
    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    if not np.all(np.isfinite(epsilon)):
        raise ValueError("energy density must be finite")
    result = np.zeros_like(epsilon)
    upper = epsilon_t_mev_fm3 + delta_mev_fm3
    ramp = (epsilon > epsilon_t_mev_fm3) & (epsilon < upper)
    x = (epsilon[ramp] - epsilon_t_mev_fm3) / delta_mev_fm3
    result[ramp] = 30.0 * x**2 * (x - 1.0) ** 2 / delta_mev_fm3
    return _scalar_or_array(result)


def smootherstep_window_second_derivative(
    energy_density_mev_fm3: Any,
    *,
    epsilon_t_mev_fm3: float,
    delta_mev_fm3: float,
) -> float | np.ndarray:
    """Return d2W/d-epsilon2 for the exact piecewise smootherstep."""
    if not np.isfinite([epsilon_t_mev_fm3, delta_mev_fm3]).all():
        raise ValueError("window parameters must be finite")
    if delta_mev_fm3 <= 0.0:
        raise ValueError("Delta must be positive")
    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    if not np.all(np.isfinite(epsilon)):
        raise ValueError("energy density must be finite")
    result = np.zeros_like(epsilon)
    upper = epsilon_t_mev_fm3 + delta_mev_fm3
    ramp = (epsilon > epsilon_t_mev_fm3) & (epsilon < upper)
    x = (epsilon[ramp] - epsilon_t_mev_fm3) / delta_mev_fm3
    result[ramp] = (
        60.0 * x * (2.0 * x**2 - 3.0 * x + 1.0) / delta_mev_fm3**2
    )
    return _scalar_or_array(result)


def gaussian_profile(
    energy_density_mev_fm3: Any,
    deformation: BSk24WindowedDeformation,
) -> float | np.ndarray:
    """Return the nominal unit-amplitude Gaussian G."""
    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    if not np.all(np.isfinite(epsilon)):
        raise ValueError("energy density must be finite")
    result = np.exp(
        -0.5
        * (
            (epsilon - deformation.epsilon0_mev_fm3)
            / deformation.sigma_mev_fm3
        )
        ** 2
    )
    return _scalar_or_array(result)


def windowed_gaussian_shape(
    energy_density_mev_fm3: Any,
    deformation: BSk24WindowedDeformation,
    *,
    epsilon_t_mev_fm3: float,
) -> float | np.ndarray:
    """Return G*W without the nominal amplitude."""
    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    gaussian = np.asarray(gaussian_profile(epsilon, deformation), dtype=float)
    window = np.asarray(
        smootherstep_window(
            epsilon,
            epsilon_t_mev_fm3=epsilon_t_mev_fm3,
            delta_mev_fm3=deformation.delta_mev_fm3,
        ),
        dtype=float,
    )
    return _scalar_or_array(gaussian * window)


def windowed_gaussian_delta_cs2(
    energy_density_mev_fm3: Any,
    deformation: BSk24WindowedDeformation,
    *,
    epsilon_t_mev_fm3: float,
) -> float | np.ndarray:
    """Return the raw additive deformation A*G*W."""
    shape = np.asarray(
        windowed_gaussian_shape(
            energy_density_mev_fm3,
            deformation,
            epsilon_t_mev_fm3=epsilon_t_mev_fm3,
        ),
        dtype=float,
    )
    return _scalar_or_array(deformation.amplitude * shape)


def _normal_moment_integrals(lower: float, upper: np.ndarray) -> list[np.ndarray]:
    """Return definite integral of z**k exp(-z**2/2), k=0..5."""
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    exp_lower = np.exp(-0.5 * lower_array**2)
    exp_upper = np.exp(-0.5 * upper_array**2)
    moments: list[np.ndarray] = [
        math.sqrt(math.pi / 2.0)
        * (erf(upper_array / math.sqrt(2.0)) - erf(lower_array / math.sqrt(2.0))),
        exp_lower - exp_upper,
    ]
    for order in range(2, 6):
        moments.append(
            lower_array ** (order - 1) * exp_lower
            - upper_array ** (order - 1) * exp_upper
            + (order - 1) * moments[order - 2]
        )
    return moments


def _shifted_gaussian_moment(
    lower: float,
    upper: np.ndarray,
    *,
    order: int,
    center: float,
    sigma: float,
    origin: float,
) -> np.ndarray:
    """Integrate (epsilon-origin)**order times a Gaussian."""
    z_lower = (lower - center) / sigma
    z_upper = (upper - center) / sigma
    normal_moments = _normal_moment_integrals(z_lower, z_upper)
    offset = center - origin
    result = np.zeros_like(z_upper, dtype=float)
    for power in range(order + 1):
        result += (
            math.comb(order, power)
            * offset ** (order - power)
            * sigma ** (power + 1)
            * normal_moments[power]
        )
    return result


def _ramp_gaussian_integral(
    upper: np.ndarray,
    deformation: BSk24WindowedDeformation,
    *,
    epsilon_t_mev_fm3: float,
) -> np.ndarray:
    """Analytically integrate G times the quintic ramp from epsilon_t."""
    delta = deformation.delta_mev_fm3
    center = deformation.epsilon0_mev_fm3
    sigma = deformation.sigma_mev_fm3
    m3 = _shifted_gaussian_moment(
        epsilon_t_mev_fm3,
        upper,
        order=3,
        center=center,
        sigma=sigma,
        origin=epsilon_t_mev_fm3,
    )
    m4 = _shifted_gaussian_moment(
        epsilon_t_mev_fm3,
        upper,
        order=4,
        center=center,
        sigma=sigma,
        origin=epsilon_t_mev_fm3,
    )
    m5 = _shifted_gaussian_moment(
        epsilon_t_mev_fm3,
        upper,
        order=5,
        center=center,
        sigma=sigma,
        origin=epsilon_t_mev_fm3,
    )
    return 10.0 * m3 / delta**3 - 15.0 * m4 / delta**4 + 6.0 * m5 / delta**5


def windowed_gaussian_pressure_primitive(
    energy_density_mev_fm3: Any,
    deformation: BSk24WindowedDeformation,
    *,
    epsilon_t_mev_fm3: float,
) -> float | np.ndarray:
    """Return A times the analytic integral of G*W from the anchor."""
    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    if not np.all(np.isfinite(epsilon)):
        raise ValueError("energy density must be finite")
    if deformation.amplitude == 0.0:
        return _scalar_or_array(np.zeros_like(epsilon))
    flat = epsilon.reshape(-1)
    result = np.zeros_like(flat)
    ramp_end = epsilon_t_mev_fm3 + deformation.delta_mev_fm3
    ramp = (flat > epsilon_t_mev_fm3) & (flat < ramp_end)
    if np.any(ramp):
        result[ramp] = _ramp_gaussian_integral(
            flat[ramp],
            deformation,
            epsilon_t_mev_fm3=epsilon_t_mev_fm3,
        )
    above = flat >= ramp_end
    if np.any(above):
        ramp_area = float(
            _ramp_gaussian_integral(
                np.asarray(ramp_end),
                deformation,
                epsilon_t_mev_fm3=epsilon_t_mev_fm3,
            )
        )
        center = deformation.epsilon0_mev_fm3
        sigma = deformation.sigma_mev_fm3
        z_start = (ramp_end - center) / sigma
        z_upper = (flat[above] - center) / sigma
        gaussian_area = math.sqrt(math.pi / 2.0) * sigma * (
            erf(z_upper / math.sqrt(2.0)) - erf(z_start / math.sqrt(2.0))
        )
        result[above] = ramp_area + gaussian_area
    result *= deformation.amplitude
    return _scalar_or_array(result.reshape(epsilon.shape))


def _windowed_pressure(
    epsilon: np.ndarray,
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24WindowedDeformation,
) -> np.ndarray:
    rho = _mass_density_from_energy_density(epsilon)
    pressure = np.asarray(
        baseline.eos.published_fit_pressure_from_mass_density(rho),
        dtype=float,
    )
    if deformation.amplitude == 0.0:
        return pressure
    pressure += np.asarray(
        windowed_gaussian_pressure_primitive(
            epsilon,
            deformation,
            epsilon_t_mev_fm3=baseline.anchor.energy_density_mev_fm3,
        ),
        dtype=float,
    )
    return pressure


def _windowed_cs2(
    epsilon: np.ndarray,
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24WindowedDeformation,
) -> np.ndarray:
    rho = _mass_density_from_energy_density(epsilon)
    cs2 = np.asarray(
        baseline.eos.published_fit_sound_speed_squared_from_mass_density(rho),
        dtype=float,
    )
    if deformation.amplitude == 0.0:
        return cs2
    cs2 += np.asarray(
        windowed_gaussian_delta_cs2(
            epsilon,
            deformation,
            epsilon_t_mev_fm3=baseline.anchor.energy_density_mev_fm3,
        ),
        dtype=float,
    )
    return cs2

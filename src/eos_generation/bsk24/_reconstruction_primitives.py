"""Low-level numerical primitives for BSk24 thermodynamic reconstruction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.special import erf

from eos_generation.bsk24.baseline import (
    CAUSAL_MASS_DENSITY_MAX_G_CM3,
    FIT_MASS_DENSITY_MAX_G_CM3,
    FIT_MASS_DENSITY_MIN_G_CM3,
    MEV_FM3_TO_MASS_DENSITY_G_CM3,
    NEUTRON_REST_ENERGY_MEV,
)

if TYPE_CHECKING:
    from eos_generation.bsk24.reconstruction import (
        BSk24AnchorState,
        BSk24GridSettings,
    )


ANCHOR_BARYON_DENSITY_FM3 = 0.16
APPROVED_EPSILON0_MEV_FM3 = 400.0
APPROVED_SIGMA_MEV_FM3 = 40.0
APPROVED_CASE_PARAMETERS = {
    "a0": 0.0,
    "positive": 0.05,
    "negative": -0.05,
}
COMPOSE_CORE_ENTRY_BARYON_DENSITY_FM3 = 0.0807555
COMPOSE_MUON_ONSET_BARYON_DENSITY_FM3 = 0.12577337
COMPOSE_OUTER_INNER_TRANSITION_EPSILON_MEV_FM3 = 0.253215574967
COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3 = 76.5591451931


def _finite_numeric_array(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain only finite numeric values"
        ) from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def gaussian_pressure_primitive(
    epsilon: Any,
    *,
    amplitude: float,
    epsilon0: float,
    sigma: float,
    epsilon_ref: float,
) -> float | np.ndarray:
    """Return the unclipped Gaussian pressure primitive."""
    eps = _finite_numeric_array("epsilon", epsilon)
    _finite_numeric_array(
        "Gaussian parameters",
        [amplitude, epsilon0, sigma, epsilon_ref],
    )
    width = float(sigma)
    if width <= 0.0:
        raise ValueError("sigma must be positive")
    root_two_sigma = np.sqrt(2.0) * width
    result = float(amplitude) * width * np.sqrt(np.pi / 2.0) * (
        erf((eps - float(epsilon0)) / root_two_sigma)
        - erf(
            (float(epsilon_ref) - float(epsilon0)) / root_two_sigma
        )
    )
    return (
        float(result)
        if np.ndim(result) == 0
        else np.asarray(result, dtype=float)
    )


def gaussian_sound_speed_bump(
    epsilon: np.ndarray | float,
    *,
    amplitude: float,
    epsilon0: float,
    sigma: float,
) -> np.ndarray | float:
    """Return the unclipped Gaussian sound-speed perturbation."""
    values = np.asarray(epsilon, dtype=float)
    result = float(amplitude) * np.exp(
        -0.5 * ((values - float(epsilon0)) / float(sigma)) ** 2
    )
    return float(result) if result.ndim == 0 else result


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def _mass_density_from_energy_density(epsilon: np.ndarray) -> np.ndarray:
    """Convert units while snapping only roundoff-adjacent declared endpoints."""
    rho = np.asarray(epsilon, dtype=float) * MEV_FM3_TO_MASS_DENSITY_G_CM3
    guard = 16.0 * np.finfo(float).eps
    rho = np.where(
        np.isclose(rho, FIT_MASS_DENSITY_MIN_G_CM3, rtol=guard, atol=0.0),
        FIT_MASS_DENSITY_MIN_G_CM3,
        rho,
    )
    rho = np.where(
        np.isclose(rho, CAUSAL_MASS_DENSITY_MAX_G_CM3, rtol=guard, atol=0.0),
        CAUSAL_MASS_DENSITY_MAX_G_CM3,
        rho,
    )
    rho = np.where(
        np.isclose(rho, FIT_MASS_DENSITY_MAX_G_CM3, rtol=guard, atol=0.0),
        FIT_MASS_DENSITY_MAX_G_CM3,
        rho,
    )
    return rho


def _profile_grid(
    anchor: BSk24AnchorState,
    settings: BSk24GridSettings,
) -> tuple[np.ndarray, int]:
    epsilon_min = FIT_MASS_DENSITY_MIN_G_CM3 / MEV_FM3_TO_MASS_DENSITY_G_CM3
    epsilon_max = CAUSAL_MASS_DENSITY_MAX_G_CM3 / MEV_FM3_TO_MASS_DENSITY_G_CM3
    lower = np.geomspace(epsilon_min, anchor.energy_density_mev_fm3, settings.lower_points)
    upper = np.linspace(anchor.energy_density_mev_fm3, epsilon_max, settings.upper_points)
    epsilon = np.concatenate((lower[:-1], upper))
    anchor_index = settings.lower_points - 1
    if epsilon[anchor_index] != anchor.energy_density_mev_fm3:
        raise RuntimeError("profile grid does not retain the exact selected anchor")
    return epsilon, anchor_index


def _bidirectional_baryon_reconstruction(
    epsilon: np.ndarray,
    pressure: np.ndarray,
    *,
    anchor_index: int,
    anchor_density_fm3: float,
) -> np.ndarray:
    """Integrate dln(n)=epsilon/(epsilon+P) dln(epsilon) around the anchor."""
    if not np.all(np.diff(epsilon) > 0.0) or not np.all(pressure > 0.0):
        raise ValueError("thermodynamic reconstruction requires positive monotone inputs")
    logarithmic_integrand = epsilon / (epsilon + pressure)
    cumulative = cumulative_simpson(
        logarithmic_integrand,
        x=np.log(epsilon),
        initial=0.0,
    )
    log_n = math.log(anchor_density_fm3) + cumulative - cumulative[anchor_index]
    baryon_density = np.exp(log_n)
    baryon_density[anchor_index] = anchor_density_fm3
    if not np.all(np.diff(baryon_density) > 0.0):
        raise ValueError("reconstructed baryon density is not strictly increasing")
    return baryon_density


def _derived_state(
    epsilon: np.ndarray,
    pressure: np.ndarray,
    cs2: np.ndarray,
    baryon_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = (epsilon + pressure) / baryon_density
    gamma = ((epsilon + pressure) / pressure) * cs2
    energy_per_baryon = epsilon / baryon_density - NEUTRON_REST_ENERGY_MEV
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(gamma)):
        raise ValueError("derived thermodynamic state contains nonfinite values")
    return mu, gamma, energy_per_baryon


def _max_residual(values: np.ndarray, epsilon: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    index = int(np.argmax(absolute))
    return {
        "maximum_absolute": float(absolute[index]),
        "epsilon_at_maximum_mev_fm3": float(epsilon[index]),
        "p50_absolute": float(np.percentile(absolute, 50.0)),
        "p95_absolute": float(np.percentile(absolute, 95.0)),
        "p99_absolute": float(np.percentile(absolute, 99.0)),
    }

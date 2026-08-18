"""Sampled baseline and generated-profile calculations for reconstruction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator

from eos_generation.bsk24._reconstruction_primitives import (
    _mass_density_from_energy_density,
    gaussian_pressure_primitive,
    gaussian_sound_speed_bump,
)

if TYPE_CHECKING:
    from eos_generation.bsk24.reconstruction import (
        BSk24ConsistentBaseline,
        BSk24Deformation,
    )


def _generated_pressure(
    epsilon: np.ndarray,
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24Deformation,
) -> np.ndarray:
    rho = _mass_density_from_energy_density(epsilon)
    pressure = np.asarray(baseline.eos.pressure_from_mass_density(rho), dtype=float)
    if deformation.amplitude == 0.0:
        return pressure
    mask = epsilon >= baseline.anchor.energy_density_mev_fm3
    pressure[mask] += np.asarray(
        gaussian_pressure_primitive(
            epsilon[mask],
            amplitude=deformation.amplitude,
            epsilon0=deformation.epsilon0_mev_fm3,
            sigma=deformation.sigma_mev_fm3,
            epsilon_ref=baseline.anchor.energy_density_mev_fm3,
        ),
        dtype=float,
    )
    return pressure


def _generated_cs2(
    epsilon: np.ndarray,
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24Deformation,
) -> np.ndarray:
    rho = _mass_density_from_energy_density(epsilon)
    cs2 = np.asarray(baseline.eos.sound_speed_squared_from_mass_density(rho), dtype=float)
    if deformation.amplitude == 0.0:
        return cs2
    mask = epsilon >= baseline.anchor.energy_density_mev_fm3
    cs2[mask] += np.asarray(
        gaussian_sound_speed_bump(
            epsilon[mask],
            amplitude=deformation.amplitude,
            epsilon0=deformation.epsilon0_mev_fm3,
            sigma=deformation.sigma_mev_fm3,
        ),
        dtype=float,
    )
    return cs2


def _residual_arrays(
    epsilon: np.ndarray,
    pressure: np.ndarray,
    cs2: np.ndarray,
    baryon_density: np.ndarray,
    chemical_potential: np.ndarray,
) -> dict[str, np.ndarray]:
    pressure_derivative = PchipInterpolator(epsilon, pressure, extrapolate=False).derivative()(epsilon)
    density_derivative = PchipInterpolator(
        epsilon, baryon_density, extrapolate=False
    ).derivative()(epsilon)
    independent_mu = 1.0 / density_derivative
    algebraic_r_p = pressure - (baryon_density * chemical_potential - epsilon)
    algebraic_r_mu = chemical_potential - (epsilon + pressure) / baryon_density
    independent_r_p = pressure - (baryon_density * independent_mu - epsilon)
    independent_r_mu = chemical_potential - independent_mu
    derivative_r_c = cs2 - pressure_derivative
    first_law = chemical_potential * density_derivative - 1.0
    pressure_scale = np.maximum.reduce(
        [np.abs(pressure), np.abs(baryon_density * independent_mu), np.abs(epsilon)]
    )
    mu_scale = np.maximum(np.abs(chemical_potential), np.abs(independent_mu))
    return {
        "r_p_algebraic": algebraic_r_p,
        "r_mu_algebraic": algebraic_r_mu,
        "r_p_independent": independent_r_p,
        "r_p_independent_normalized": independent_r_p / pressure_scale,
        "r_mu_independent": independent_r_mu,
        "r_mu_independent_normalized": independent_r_mu / mu_scale,
        "r_c": derivative_r_c,
        "first_law_normalized": first_law,
        "dP_dEpsilon_independent": pressure_derivative,
        "mu_from_dEpsilon_dn_independent": independent_mu,
    }

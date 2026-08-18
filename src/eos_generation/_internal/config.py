"""Frozen numerical constants used by the BSk24 method.

This is intentionally smaller than the historical project-wide configuration
module: only values used by the analytical BSk24, TOV, and tidal workflows are
retained here.  The values are unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PhysicsUnits:
    hbar_c_mev_fm: float = 197.33
    neutron_mass_mev: float = 939.0
    gravity_conversion: float = 1.124e-5
    solar_mass_length_km: float = 1.4766


@dataclass(frozen=True)
class ThermodynamicGridConfig:
    absolute_pressure_max_fallback: float = 10**4.2


@dataclass(frozen=True)
class TovConfig:
    radius_min_km: float = 1e-4
    radius_max_km: float = 25.0
    pressure_min_safe: float = 1e-14
    grid_pressure_min_log: float = 1e-14
    grid_pressure_transition: float = 1.0
    grid_pressure_max_linear: float = 4000.0
    grid_crust_points: int = 300
    grid_core_points: int = 1200
    small_step_mass: float = 1e-5
    ode_rtol: float = 1e-10
    ode_atol: float = 1e-12
    sequence_rtol: float = 1e-8
    sequence_atol: float = 1e-10
    sequence_points: int = 200
    sequence_low_ratio: float = 0.3
    dense_profile_points: int = 300
    surface_pressure_cutoff: float = 1e-13
    singularity_limit: float = 1e-5
    center_radius_limit: float = 1e-4


@dataclass(frozen=True)
class FilterPolicy:
    buchdahl_limit: float = 4.0 / 9.0
    minimum_mass_cutoff: float = 0.05
    minimum_radius_cutoff_km: float = 3.0


@dataclass(frozen=True)
class HadronicRunConfig:
    causal_root_xtol: float = 1.0e-12
    causal_root_rtol: float = 1.0e-12


@dataclass(frozen=True)
class MethodConfig:
    units: PhysicsUnits = PhysicsUnits()
    thermodynamics: ThermodynamicGridConfig = ThermodynamicGridConfig()
    tov: TovConfig = TovConfig()
    filters: FilterPolicy = FilterPolicy()
    hadronic: HadronicRunConfig = HadronicRunConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = MethodConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "FilterPolicy",
    "HadronicRunConfig",
    "MethodConfig",
    "PhysicsUnits",
    "ThermodynamicGridConfig",
    "TovConfig",
]

"""Frozen zero-temperature CFL thermodynamics for bare self-bound stars.

The governed model is the full finite-``m_s`` common-Fermi-momentum bag
formulation, with the leading ``Delta**2`` condensation contribution.  It is
parametric in the quark chemical potential ``mu``; all public energy and
pressure values are total densities in MeV fm^-3 and ``c = 1``.

No perturbative ``a4`` term, leptons, Goldstone modes, crust, or hadronic
matching is present.  The finite source domain is deliberate and all public
evaluators reject extrapolation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq

from eos_generation.stellar.discontinuities import (
    BARE_SELF_BOUND_SEQUENCE_POLICY,
    SEED_PRESERVING_LOCAL_REFINEMENT_POLICY,
    EosDiscontinuity,
)


CFL_FORMULATION_ID = "cfl_bag_full_ms_delta2_v1"
CFL_FORMULATION_VERSION = "1"
CFL_FORMULATION_DESCRIPTION_ID = "cfl_full_finite_ms_common_fermi_bag_delta2"
FROZEN_PARAMETER_SET_ID = (
    "cfl_full_finite_ms_bag_delta2_b57p5_ms100_delta100_v1"
)
FROZEN_PARAMETER_SCHEMA_VERSION = "cfl_frozen_parameter_set_v1"
CFL_DOMAIN_ID = "cfl_mu_q_249p31780807778472_to_600_mev_v1"
CFL_DEFORMATION_PROFILE_ID = "cfl_surface_anchored_windowed_gaussian_v1"
CFL_DEFORMATION_PROFILE_VERSION = "1"

HBAR_C_MEV_FM = 197.3269804
HBAR_C_CUBED_MEV3_FM3 = HBAR_C_MEV_FM**3
TEMPERATURE_MEV = 0.0
UP_QUARK_MASS_MEV = 0.0
DOWN_QUARK_MASS_MEV = 0.0
STRANGE_QUARK_MASS_MEV = 100.0
PAIRING_GAP_MEV = 100.0
# The literature profile specifies B itself in MeV fm^-3.  Preserve that
# decimal as the authority and derive the natural-unit value and fourth root;
# reversing this order would define a different parameter set after rounding.
BAG_CONSTANT_MEV_FM3 = 57.5
BAG_CONSTANT_NATURAL_MEV4 = (
    BAG_CONSTANT_MEV_FM3 * HBAR_C_CUBED_MEV3_FM3
)
BAG_CONSTANT_FOURTH_ROOT_MEV = BAG_CONSTANT_NATURAL_MEV4**0.25

QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV = 249.31780807778472
QUARK_CHEMICAL_POTENTIAL_MAX_MEV = 600.0


def _common_fermi_momentum_formula(mu: np.ndarray) -> np.ndarray:
    return 2.0 * mu - np.sqrt(
        mu * mu + STRANGE_QUARK_MASS_MEV**2 / 3.0
    )


def _raw_thermodynamic_potential_formula(mu: np.ndarray) -> np.ndarray:
    """Return the published finite-ms Omega before FP surface normalization."""

    nu = _common_fermi_momentum_formula(mu)
    ms = STRANGE_QUARK_MASS_MEV
    strange_energy = np.sqrt(nu * nu + ms * ms)
    massless_integral = nu**4 / 4.0 - mu * nu**3 / 3.0
    strange_integral = (
        (
            nu * strange_energy * (2.0 * nu**2 + ms**2)
            - ms**4 * np.arcsinh(nu / ms)
        )
        / 8.0
        - mu * nu**3 / 3.0
    )
    omega_natural = (
        6.0 * massless_integral / math.pi**2
        + 3.0 * strange_integral / math.pi**2
        - 3.0 * PAIRING_GAP_MEV**2 * mu**2 / math.pi**2
        + BAG_CONSTANT_NATURAL_MEV4
    )
    return omega_natural / HBAR_C_CUBED_MEV3_FM3


# The approved decimal mu_surface evaluates the published expression at a
# sub-pico-MeV fm^-3 residual in binary64.  Subtract that one deterministic
# constant globally so P(mu_surface) is exactly zero without replacing any
# endpoint n_B, mu_B, epsilon, or c_s^2 value by rounded packet numbers.
_RAW_PRESSURE_AT_SURFACE_MEV_FM3 = float.fromhex("0x1.a33d317f41022p-46")


def _governed_state_formula(mu_value: float) -> tuple[float, float, float, float, float]:
    mu = np.asarray(float(mu_value))
    nu = _common_fermi_momentum_formula(mu)
    pressure = float(
        -_raw_thermodynamic_potential_formula(mu)
        - _RAW_PRESSURE_AT_SURFACE_MEV_FM3
    )
    density = float(
        (nu**3 + 2.0 * PAIRING_GAP_MEV**2 * mu)
        / (math.pi**2 * HBAR_C_CUBED_MEV3_FM3)
    )
    mu_b = 3.0 * float(mu)
    epsilon = -pressure + mu_b * density
    nu_prime = 2.0 - mu / np.sqrt(
        mu * mu + STRANGE_QUARK_MASS_MEV**2 / 3.0
    )
    cs2 = float(
        (nu**3 + 2.0 * PAIRING_GAP_MEV**2 * mu)
        / (
            mu
            * (3.0 * nu**2 * nu_prime + 2.0 * PAIRING_GAP_MEV**2)
        )
    )
    return float(nu), pressure, density, epsilon, cs2


_FROZEN_SURFACE_STATE = (
    float.fromhex("0x1.e570bd67efc6ap+7"),
    0.0,
    float.fromhex("0x1.046bfe2aba766p-2"),
    float.fromhex("0x1.7c6fb4c3f9a60p+7"),
    float.fromhex("0x1.8980f0250edbcp-2"),
)
_FROZEN_MAXIMUM_STATE = (
    float.fromhex("0x1.2a9d4381ec6e3p+9"),
    float.fromhex("0x1.4d1624e58ac50p+10"),
    float.fromhex("0x1.7bd0e8da4ac62p+1"),
    float.fromhex("0x1.f51a26dcf20dcp+11"),
    float.fromhex("0x1.5eba6851ce031p-2"),
)
(
    COMMON_FERMI_MOMENTUM_SURFACE_MEV,
    PRESSURE_SURFACE_MEV_FM3,
    BARYON_DENSITY_SURFACE_FM3,
    ENERGY_DENSITY_SURFACE_MEV_FM3,
    SOUND_SPEED_SQUARED_SURFACE,
) = _FROZEN_SURFACE_STATE
(
    _COMMON_FERMI_MOMENTUM_MAX_MEV,
    PRESSURE_MAX_MEV_FM3,
    BARYON_DENSITY_MAX_FM3,
    ENERGY_DENSITY_MAX_MEV_FM3,
    SOUND_SPEED_SQUARED_MAX_ENDPOINT,
) = _FROZEN_MAXIMUM_STATE
BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV = (
    3.0 * QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV
)
BARYON_CHEMICAL_POTENTIAL_MAX_MEV = 3.0 * QUARK_CHEMICAL_POTENTIAL_MAX_MEV

_BINARY64_REFERENCE_VERIFIED = False


def _verify_binary64_reference() -> None:
    """Fail closed if the runtime cannot reproduce the frozen formula state."""

    global _BINARY64_REFERENCE_VERIFIED
    if _BINARY64_REFERENCE_VERIFIED:
        return
    raw_surface_pressure = float(
        -_raw_thermodynamic_potential_formula(
            np.asarray(QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV)
        )
    )
    if raw_surface_pressure != _RAW_PRESSURE_AT_SURFACE_MEV_FM3:
        raise RuntimeError(
            "this runtime does not reproduce the governed CFL binary64 surface residual"
        )
    if (
        _governed_state_formula(QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV)
        != _FROZEN_SURFACE_STATE
    ):
        raise RuntimeError(
            "this runtime does not reproduce the governed CFL binary64 surface state"
        )
    if (
        _governed_state_formula(QUARK_CHEMICAL_POTENTIAL_MAX_MEV)
        != _FROZEN_MAXIMUM_STATE
    ):
        raise RuntimeError(
            "this runtime does not reproduce the governed CFL binary64 endpoint state"
        )
    _BINARY64_REFERENCE_VERIFIED = True

# Phase-1 decision-packet values were intentionally rounded for human display.
# They are retained only for traceability and never participate in domains,
# hashes, equality checks, or evaluator endpoint replacement.
PHASE1_DISPLAY_REFERENCE_ENERGY_DENSITY_SURFACE_MEV_FM3 = 190.218176006531
PHASE1_DISPLAY_REFERENCE_ENERGY_DENSITY_MAX_MEV_FM3 = 4008.8172440269


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class CFLFrozenParameters:
    """Immutable, hash-addressed record of every governed CFL convention."""

    hbar_c_mev_fm: float = HBAR_C_MEV_FM
    temperature_mev: float = TEMPERATURE_MEV
    up_quark_mass_mev: float = UP_QUARK_MASS_MEV
    down_quark_mass_mev: float = DOWN_QUARK_MASS_MEV
    strange_quark_mass_mev: float = STRANGE_QUARK_MASS_MEV
    pairing_gap_mev: float = PAIRING_GAP_MEV
    bag_constant_fourth_root_mev: float = BAG_CONSTANT_FOURTH_ROOT_MEV
    bag_constant_natural_mev4: float = BAG_CONSTANT_NATURAL_MEV4
    bag_constant_mev_fm3: float = BAG_CONSTANT_MEV_FM3

    def __post_init__(self) -> None:
        expected = (
            HBAR_C_MEV_FM,
            TEMPERATURE_MEV,
            UP_QUARK_MASS_MEV,
            DOWN_QUARK_MASS_MEV,
            STRANGE_QUARK_MASS_MEV,
            PAIRING_GAP_MEV,
            BAG_CONSTANT_FOURTH_ROOT_MEV,
            BAG_CONSTANT_NATURAL_MEV4,
            BAG_CONSTANT_MEV_FM3,
        )
        actual = (
            self.hbar_c_mev_fm,
            self.temperature_mev,
            self.up_quark_mass_mev,
            self.down_quark_mass_mev,
            self.strange_quark_mass_mev,
            self.pairing_gap_mev,
            self.bag_constant_fourth_root_mev,
            self.bag_constant_natural_mev4,
            self.bag_constant_mev_fm3,
        )
        if actual != expected:
            raise ValueError(
                "CFLFrozenParameters is a single non-configurable governed record"
            )

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FROZEN_PARAMETER_SCHEMA_VERSION,
            "parameter_set_id": FROZEN_PARAMETER_SET_ID,
            "formulation_id": CFL_FORMULATION_ID,
            "formulation_version": CFL_FORMULATION_VERSION,
            "constants": {
                "hbar_c_mev_fm": self.hbar_c_mev_fm,
                "temperature_mev": self.temperature_mev,
                "up_quark_mass_mev": self.up_quark_mass_mev,
                "down_quark_mass_mev": self.down_quark_mass_mev,
                "strange_quark_mass_mev": self.strange_quark_mass_mev,
                "pairing_gap_mev": self.pairing_gap_mev,
                "bag_constant_fourth_root_mev": (
                    self.bag_constant_fourth_root_mev
                ),
                "bag_constant_natural_mev4": self.bag_constant_natural_mev4,
                "bag_constant_mev_fm3": self.bag_constant_mev_fm3,
            },
            "conventions": {
                "quark_chemical_potential": "common_flavor_mu_mev",
                "baryon_chemical_potential": "mu_B_equals_3_mu",
                "strange_mass_treatment": "full_finite_ms_free_integral",
                "pairing_treatment": "leading_minus_3_Delta_squared_mu_squared_over_pi_squared",
                "perturbative_correction": "absent_a4_equals_1",
                "renormalization_scale": "not_applicable_no_perturbative_term",
                "electrons": "absent_mu_e_equals_0_neutral_CFL",
                "muons": "absent",
                "goldstone_and_kaon_terms": "omitted_uncondensed_CFL",
                "energy_density": "total_including_rest_energy",
                "surface": "bare_self_bound_vacuum_no_crust",
                "descriptive_formulation_alias": CFL_FORMULATION_DESCRIPTION_ID,
                "binary64_surface_pressure_normalization_mev_fm3": (
                    _RAW_PRESSURE_AT_SURFACE_MEV_FM3
                ),
            },
            "domain": {
                "domain_id": CFL_DOMAIN_ID,
                "quark_chemical_potential_mev": [
                    QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV,
                    QUARK_CHEMICAL_POTENTIAL_MAX_MEV,
                ],
                "baryon_chemical_potential_mev": [
                    BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV,
                    BARYON_CHEMICAL_POTENTIAL_MAX_MEV,
                ],
                "energy_density_mev_fm3": [
                    ENERGY_DENSITY_SURFACE_MEV_FM3,
                    ENERGY_DENSITY_MAX_MEV_FM3,
                ],
                "pressure_mev_fm3": [
                    PRESSURE_SURFACE_MEV_FM3,
                    PRESSURE_MAX_MEV_FM3,
                ],
            },
            "surface": {
                "energy_density_mev_fm3": ENERGY_DENSITY_SURFACE_MEV_FM3,
                "pressure_mev_fm3": PRESSURE_SURFACE_MEV_FM3,
                "baryon_density_fm3": BARYON_DENSITY_SURFACE_FM3,
                "quark_chemical_potential_mev": (
                    QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV
                ),
                "baryon_chemical_potential_mev": (
                    BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV
                ),
                "common_fermi_momentum_mev": (
                    COMMON_FERMI_MOMENTUM_SURFACE_MEV
                ),
                "sound_speed_squared": SOUND_SPEED_SQUARED_SURFACE,
                "energy_per_baryon_mev": (
                    BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV
                ),
            },
            "stability_and_validity": {
                "absolute_stability_condition": "mu_B_surface_mev <= 930",
                "absolute_stability_limit_mev": 930.0,
                "absolute_stability_passed": bool(
                    BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV <= 930.0
                ),
                "fully_gapped_CFL_condition": "m_s_squared_over_mu < 2_Delta",
                "surface_stress_mev": (
                    STRANGE_QUARK_MASS_MEV**2
                    / QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV
                ),
                "surface_pairing_margin_mev": (
                    2.0 * PAIRING_GAP_MEV
                    - STRANGE_QUARK_MASS_MEV**2
                    / QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV
                ),
                "fully_gapped_CFL_passed": bool(
                    STRANGE_QUARK_MASS_MEV**2
                    / QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV
                    < 2.0 * PAIRING_GAP_MEV
                ),
                "ordinary_nuclei_two_flavor_condition": (
                    "two_flavor_energy_per_baryon_mev >= 934"
                ),
                "ordinary_nuclei_two_flavor_limit_mev": 934.0,
                "ordinary_nuclei_two_flavor_status": (
                    "external_assumption_not_demonstrable_from_the_CFL_phase_only"
                ),
            },
        }

    @property
    def parameter_set_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self._hash_payload())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        return {
            "schema_version": payload["schema_version"],
            "parameter_set_id": payload["parameter_set_id"],
            "parameter_set_sha256": self.parameter_set_sha256,
            "formulation_id": payload["formulation_id"],
            "formulation_version": payload["formulation_version"],
            "constants": payload["constants"],
            "conventions": payload["conventions"],
            "domain": payload["domain"],
            "surface": payload["surface"],
            "stability_and_validity": payload["stability_and_validity"],
        }


FROZEN_CFL_PARAMETERS = CFLFrozenParameters()
FROZEN_PARAMETER_SET_SHA256 = (
    "3991cb8615d2d29617ccb90c6dc54b23aae64bcc752856d07f17f99abc048307"
)
if FROZEN_CFL_PARAMETERS.parameter_set_sha256 != FROZEN_PARAMETER_SET_SHA256:
    raise RuntimeError("the frozen CFL parameter record hash has drifted")


class CFLDomainError(ValueError):
    """Raised when a governed CFL evaluator would extrapolate."""


class CFLInversionError(RuntimeError):
    """Raised when a monotone CFL thermodynamic inversion fails."""


@dataclass(frozen=True, slots=True)
class CFLGridSettings:
    """Governed baseline table resolution in quark chemical potential."""

    points: int = 8193
    source_lower_points: int | None = None
    source_upper_points: int | None = None
    source_contract: str = "native_cfl_points"

    def __post_init__(self) -> None:
        if (
            isinstance(self.points, bool)
            or not isinstance(self.points, int)
            or self.points < 33
            or self.points % 2 == 0
        ):
            raise ValueError("CFL grid points must be an odd integer of at least 33")

    @classmethod
    def resolve(cls, value: Any = None) -> "CFLGridSettings":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(points=value)
        lower = getattr(value, "lower_points", None)
        upper = getattr(value, "upper_points", None)
        if lower is not None and upper is not None:
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 17
                for item in (lower, upper)
            ):
                raise ValueError("stage lower_points and upper_points are invalid")
            return cls(
                points=int(lower + upper - 1),
                source_lower_points=int(lower),
                source_upper_points=int(upper),
                source_contract=(
                    "shared_stage_grid_settings_mapped_to_one_complete_CFL_mu_grid"
                ),
            )
        if isinstance(value, dict) and "points" in value:
            return cls(points=int(value["points"]))
        raise TypeError(
            "CFL grid settings must be CFLGridSettings, an odd point count, or "
            "a governed stage grid with lower_points and upper_points"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cfl_grid_settings_v1",
            "points": self.points,
            "coordinate": "quark_chemical_potential_mev",
            "source_lower_points": self.source_lower_points,
            "source_upper_points": self.source_upper_points,
            "source_contract": self.source_contract,
            "complete_domain_includes_both_endpoints": True,
        }


@dataclass(frozen=True, slots=True)
class CFLSurfaceState:
    baryon_density_fm3: float = BARYON_DENSITY_SURFACE_FM3
    energy_density_mev_fm3: float = ENERGY_DENSITY_SURFACE_MEV_FM3
    pressure_mev_fm3: float = 0.0
    chemical_potential_mev: float = BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV
    quark_chemical_potential_mev: float = QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV
    common_fermi_momentum_mev: float = COMMON_FERMI_MOMENTUM_SURFACE_MEV

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_type": "bare_self_bound_surface",
            "baryon_density_fm3": self.baryon_density_fm3,
            "energy_density_mev_fm3": self.energy_density_mev_fm3,
            "pressure_mev_fm3": self.pressure_mev_fm3,
            "chemical_potential_mev": self.chemical_potential_mev,
            "quark_chemical_potential_mev": self.quark_chemical_potential_mev,
            "common_fermi_momentum_mev": self.common_fermi_momentum_mev,
        }


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def _require_finite_domain(
    value: Any,
    *,
    lower: float,
    upper: float,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise CFLDomainError(f"{name} must be finite")
    if np.any(array < lower) or np.any(array > upper):
        raise CFLDomainError(
            f"{name} is outside the governed non-extrapolating interval "
            f"[{lower!r}, {upper!r}]"
        )
    return array


class CFLAnalyticEos:
    """Analytic, non-extrapolating frozen CFL EoS.

    ``mu`` in the internal formulae is the common *quark* chemical potential;
    the baryon chemical potential is exactly ``mu_B = 3 mu``.
    """

    model_name = "CFL_FULL_FINITE_MS_BAG_DELTA2"
    parameter_set_id = FROZEN_PARAMETER_SET_ID
    parameter_set_sha256 = FROZEN_PARAMETER_SET_SHA256
    formulation_id = CFL_FORMULATION_ID
    formulation_version = CFL_FORMULATION_VERSION
    domain_id = CFL_DOMAIN_ID
    pressure_min_mev_fm3 = PRESSURE_SURFACE_MEV_FM3
    pressure_max_mev_fm3 = PRESSURE_MAX_MEV_FM3
    energy_density_min_mev_fm3 = ENERGY_DENSITY_SURFACE_MEV_FM3
    energy_density_max_mev_fm3 = ENERGY_DENSITY_MAX_MEV_FM3
    eps_surf = ENERGY_DENSITY_SURFACE_MEV_FM3
    requires_discontinuity_metadata = True
    allow_parallel_tov_sequence = True
    # Construction certifies the complete immutable branch.  Background-only
    # TOV integration may therefore request epsilon(P) without redundantly
    # evaluating c_s^2, which the background equations do not consume.
    _background_energy_only_is_certified = True
    stellar_sequence_policy = BARE_SELF_BOUND_SEQUENCE_POLICY
    stellar_local_refinement_policy = SEED_PRESERVING_LOCAL_REFINEMENT_POLICY

    def __init__(self, grid_settings: Any = None) -> None:
        _verify_binary64_reference()
        self.settings = CFLGridSettings.resolve(grid_settings)
        self.anchor = CFLSurfaceState()
        provenance = (
            f"{CFL_FORMULATION_ID}:parameter_set_sha256="
            f"{FROZEN_PARAMETER_SET_SHA256}"
        )
        self.discontinuities = (
            EosDiscontinuity.from_sides(
                identifier="cfl_bare_self_bound_surface_v1",
                kind="surface",
                pressure=0.0,
                inner_energy_density=ENERGY_DENSITY_SURFACE_MEV_FM3,
                outer_energy_density=0.0,
                provenance=provenance,
            ),
        )
        quark_mu = np.linspace(
            QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV,
            QUARK_CHEMICAL_POTENTIAL_MAX_MEV,
            self.settings.points,
        )
        self.quark_chemical_potential = quark_mu
        self.epsilon = np.asarray(
            self.energy_density_from_quark_chemical_potential(quark_mu), dtype=float
        )
        self.pressure = np.asarray(
            self.pressure_from_quark_chemical_potential(quark_mu), dtype=float
        )
        self.cs2 = np.asarray(
            self.sound_speed_squared_from_quark_chemical_potential(quark_mu),
            dtype=float,
        )
        self.baryon_density = np.asarray(
            self.baryon_density_from_quark_chemical_potential(quark_mu), dtype=float
        )
        self.chemical_potential = np.asarray(
            self.baryon_chemical_potential_from_quark_chemical_potential(quark_mu),
            dtype=float,
        )
        self.baryon_chemical_potential = self.chemical_potential
        self.energy_per_baryon = self.epsilon / self.baryon_density
        self.energy_per_baryon_minus_neutron_rest = (
            self.energy_per_baryon - 939.5654
        )
        self.adiabatic_index = np.empty_like(self.epsilon)
        self.adiabatic_index[0] = math.inf
        self.adiabatic_index[1:] = (
            (self.epsilon[1:] + self.pressure[1:])
            * self.cs2[1:]
            / self.pressure[1:]
        )
        for array in (
            self.quark_chemical_potential,
            self.epsilon,
            self.pressure,
            self.cs2,
            self.baryon_density,
            self.chemical_potential,
            self.energy_per_baryon,
            self.energy_per_baryon_minus_neutron_rest,
            self.adiabatic_index,
        ):
            array.setflags(write=False)
        # Compatibility with the existing baseline/reporting callback shape.
        self.eos = self

    @staticmethod
    def _quark_mu(value: Any) -> np.ndarray:
        return _require_finite_domain(
            value,
            lower=QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV,
            upper=QUARK_CHEMICAL_POTENTIAL_MAX_MEV,
            name="quark_chemical_potential_mev",
        )

    def common_fermi_momentum_from_quark_chemical_potential(
        self, quark_chemical_potential_mev: Any
    ) -> float | np.ndarray:
        mu = self._quark_mu(quark_chemical_potential_mev)
        nu = _common_fermi_momentum_formula(mu)
        return _scalar_or_array(nu)

    def thermodynamic_potential_from_quark_chemical_potential(
        self, quark_chemical_potential_mev: Any
    ) -> float | np.ndarray:
        mu = self._quark_mu(quark_chemical_potential_mev)
        omega = (
            _raw_thermodynamic_potential_formula(mu)
            + _RAW_PRESSURE_AT_SURFACE_MEV_FM3
        )
        return _scalar_or_array(omega)

    def pressure_from_quark_chemical_potential(
        self, quark_chemical_potential_mev: Any
    ) -> float | np.ndarray:
        pressure = -np.asarray(
            self.thermodynamic_potential_from_quark_chemical_potential(
                quark_chemical_potential_mev
            ),
            dtype=float,
        )
        pressure = np.where(
            np.asarray(quark_chemical_potential_mev, dtype=float)
            == QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV,
            0.0,
            pressure,
        )
        return _scalar_or_array(np.asarray(pressure, dtype=float))

    def baryon_density_from_quark_chemical_potential(
        self, quark_chemical_potential_mev: Any
    ) -> float | np.ndarray:
        mu = self._quark_mu(quark_chemical_potential_mev)
        nu = np.asarray(
            self.common_fermi_momentum_from_quark_chemical_potential(mu),
            dtype=float,
        )
        density = (
            nu**3 + 2.0 * PAIRING_GAP_MEV**2 * mu
        ) / (math.pi**2 * HBAR_C_CUBED_MEV3_FM3)
        return _scalar_or_array(density)

    def baryon_chemical_potential_from_quark_chemical_potential(
        self, quark_chemical_potential_mev: Any
    ) -> float | np.ndarray:
        mu = self._quark_mu(quark_chemical_potential_mev)
        mu_b = 3.0 * mu
        return _scalar_or_array(mu_b)

    def energy_density_from_quark_chemical_potential(
        self, quark_chemical_potential_mev: Any
    ) -> float | np.ndarray:
        mu = self._quark_mu(quark_chemical_potential_mev)
        pressure = np.asarray(
            self.pressure_from_quark_chemical_potential(mu), dtype=float
        )
        density = np.asarray(
            self.baryon_density_from_quark_chemical_potential(mu), dtype=float
        )
        mu_b = np.asarray(
            self.baryon_chemical_potential_from_quark_chemical_potential(mu),
            dtype=float,
        )
        epsilon = -pressure + mu_b * density
        return _scalar_or_array(epsilon)

    def common_fermi_momentum_derivative_from_quark_chemical_potential(
        self, quark_chemical_potential_mev: Any
    ) -> float | np.ndarray:
        mu = self._quark_mu(quark_chemical_potential_mev)
        derivative = 2.0 - mu / np.sqrt(
            mu * mu + STRANGE_QUARK_MASS_MEV**2 / 3.0
        )
        return _scalar_or_array(derivative)

    def baryon_density_derivative_from_quark_chemical_potential(
        self, quark_chemical_potential_mev: Any
    ) -> float | np.ndarray:
        mu = self._quark_mu(quark_chemical_potential_mev)
        nu = np.asarray(
            self.common_fermi_momentum_from_quark_chemical_potential(mu),
            dtype=float,
        )
        nu_prime = np.asarray(
            self.common_fermi_momentum_derivative_from_quark_chemical_potential(
                mu
            ),
            dtype=float,
        )
        derivative = (
            3.0 * nu**2 * nu_prime + 2.0 * PAIRING_GAP_MEV**2
        ) / (math.pi**2 * HBAR_C_CUBED_MEV3_FM3)
        return _scalar_or_array(derivative)

    def pressure_derivative_from_quark_chemical_potential(
        self, quark_chemical_potential_mev: Any
    ) -> float | np.ndarray:
        result = 3.0 * np.asarray(
            self.baryon_density_from_quark_chemical_potential(
                quark_chemical_potential_mev
            ),
            dtype=float,
        )
        return _scalar_or_array(result)

    def energy_density_derivative_from_quark_chemical_potential(
        self, quark_chemical_potential_mev: Any
    ) -> float | np.ndarray:
        mu = self._quark_mu(quark_chemical_potential_mev)
        dn_dmu = np.asarray(
            self.baryon_density_derivative_from_quark_chemical_potential(mu),
            dtype=float,
        )
        result = 3.0 * mu * dn_dmu
        return _scalar_or_array(result)

    def sound_speed_squared_from_quark_chemical_potential(
        self, quark_chemical_potential_mev: Any
    ) -> float | np.ndarray:
        mu = self._quark_mu(quark_chemical_potential_mev)
        nu = np.asarray(
            self.common_fermi_momentum_from_quark_chemical_potential(mu),
            dtype=float,
        )
        nu_prime = np.asarray(
            self.common_fermi_momentum_derivative_from_quark_chemical_potential(
                mu
            ),
            dtype=float,
        )
        numerator = nu**3 + 2.0 * PAIRING_GAP_MEV**2 * mu
        denominator = mu * (
            3.0 * nu**2 * nu_prime + 2.0 * PAIRING_GAP_MEV**2
        )
        result = numerator / denominator
        return _scalar_or_array(result)

    def _invert_monotone(
        self,
        value: Any,
        *,
        lower: float,
        upper: float,
        name: str,
        forward: Callable[[float], float],
    ) -> float | np.ndarray:
        targets = _require_finite_domain(
            value,
            lower=lower,
            upper=upper,
            name=name,
        )
        result = np.empty_like(targets, dtype=float)
        target_flat = targets.reshape(-1)
        result_flat = result.reshape(-1)
        for index, target_value in enumerate(target_flat):
            target = float(target_value)
            if target == lower:
                result_flat[index] = QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV
                continue
            if target == upper:
                result_flat[index] = QUARK_CHEMICAL_POTENTIAL_MAX_MEV
                continue
            try:
                result_flat[index] = brentq(
                    lambda mu: forward(mu) - target,
                    QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV,
                    QUARK_CHEMICAL_POTENTIAL_MAX_MEV,
                    xtol=5.0e-13,
                    rtol=1.0e-14,
                    maxiter=100,
                )
            except (RuntimeError, ValueError) as exc:
                raise CFLInversionError(
                    f"failed to invert governed {name} at {target!r}"
                ) from exc
        return _scalar_or_array(result)

    def quark_chemical_potential_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        return self._invert_monotone(
            energy_density_mev_fm3,
            lower=ENERGY_DENSITY_SURFACE_MEV_FM3,
            upper=ENERGY_DENSITY_MAX_MEV_FM3,
            name="energy_density_mev_fm3",
            forward=lambda mu: float(
                self.energy_density_from_quark_chemical_potential(mu)
            ),
        )

    def quark_chemical_potential_from_pressure(
        self, pressure_mev_fm3: Any
    ) -> float | np.ndarray:
        return self._invert_monotone(
            pressure_mev_fm3,
            lower=PRESSURE_SURFACE_MEV_FM3,
            upper=PRESSURE_MAX_MEV_FM3,
            name="pressure_mev_fm3",
            forward=lambda mu: float(
                self.pressure_from_quark_chemical_potential(mu)
            ),
        )

    def pressure_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        mu = self.quark_chemical_potential_from_energy_density(
            energy_density_mev_fm3
        )
        return self.pressure_from_quark_chemical_potential(mu)

    def energy_density_from_pressure(
        self, pressure_mev_fm3: Any
    ) -> float | np.ndarray:
        mu = self.quark_chemical_potential_from_pressure(pressure_mev_fm3)
        return self.energy_density_from_quark_chemical_potential(mu)

    def baryon_density_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        mu = self.quark_chemical_potential_from_energy_density(
            energy_density_mev_fm3
        )
        return self.baryon_density_from_quark_chemical_potential(mu)

    consistent_baryon_density_from_energy_density = (
        baryon_density_from_energy_density
    )

    def baryon_density_from_pressure(
        self, pressure_mev_fm3: Any
    ) -> float | np.ndarray:
        mu = self.quark_chemical_potential_from_pressure(pressure_mev_fm3)
        return self.baryon_density_from_quark_chemical_potential(mu)

    def baryon_chemical_potential_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        mu = self.quark_chemical_potential_from_energy_density(
            energy_density_mev_fm3
        )
        return self.baryon_chemical_potential_from_quark_chemical_potential(mu)

    def baryon_chemical_potential_from_pressure(
        self, pressure_mev_fm3: Any
    ) -> float | np.ndarray:
        mu = self.quark_chemical_potential_from_pressure(pressure_mev_fm3)
        return self.baryon_chemical_potential_from_quark_chemical_potential(mu)

    def sound_speed_squared_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        mu = self.quark_chemical_potential_from_energy_density(
            energy_density_mev_fm3
        )
        return self.sound_speed_squared_from_quark_chemical_potential(mu)

    def sound_speed_squared_from_pressure(
        self, pressure_mev_fm3: Any
    ) -> float | np.ndarray:
        mu = self.quark_chemical_potential_from_pressure(pressure_mev_fm3)
        return self.sound_speed_squared_from_quark_chemical_potential(mu)

    # Shared stellar duck-API spellings.
    cs2_from_energy_density = sound_speed_squared_from_energy_density
    cs2_from_pressure = sound_speed_squared_from_pressure

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]:
        # Both quantities have the same governed monotone pressure inversion.
        # Reusing its deterministic root preserves the formulas and avoids a
        # second identical Brent solve at every stellar EoS evaluation.
        mu = self.quark_chemical_potential_from_pressure(pressure_mev_fm3)
        epsilon = float(self.energy_density_from_quark_chemical_potential(mu))
        cs2 = float(
            self.sound_speed_squared_from_quark_chemical_potential(mu)
        )
        return epsilon, cs2

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cfl_analytic_eos_v1",
            "model_name": self.model_name,
            "parameter_set": FROZEN_CFL_PARAMETERS.to_dict(),
            "grid": self.settings.to_dict(),
            "anchor": self.anchor.to_dict(),
            "non_extrapolating": True,
            "requires_discontinuity_metadata": True,
            "discontinuities": [item.to_dict() for item in self.discontinuities],
        }

    def provenance(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "source_manifest": "eos_generation/cfl/source_manifest.json",
            "thermodynamic_potential": (
                "full_finite_ms_common_Fermi_momentum_free_integrals_plus_"
                "leading_Delta_squared_condensation_plus_bag"
            ),
            "derivatives": "analytic",
            "inversions": "Brent_monotone_no_extrapolation",
            "surface_exterior": "vacuum",
            "binary64_reference_policy": (
                "hexadecimal_formula_derived_endpoints_hash_authoritative_and_"
                "recomputed_exactly_at_EoS_construction; no_endpoint_replacement"
            ),
            "phase1_display_reference_values": {
                "energy_density_surface_mev_fm3": (
                    PHASE1_DISPLAY_REFERENCE_ENERGY_DENSITY_SURFACE_MEV_FM3
                ),
                "energy_density_max_mev_fm3": (
                    PHASE1_DISPLAY_REFERENCE_ENERGY_DENSITY_MAX_MEV_FM3
                ),
                "role": "rounded_human_reference_only_not_identity_authority",
                "authoritative_binary64_values": {
                    "energy_density_surface_mev_fm3": (
                        ENERGY_DENSITY_SURFACE_MEV_FM3
                    ),
                    "energy_density_max_mev_fm3": ENERGY_DENSITY_MAX_MEV_FM3,
                },
            },
        }


def build_cfl_baseline(grid_settings: Any = None) -> CFLAnalyticEos:
    """Build the frozen analytic baseline on one governed reporting grid.

    A shared stage object with ``lower_points`` and ``upper_points`` maps to
    ``lower_points + upper_points - 1`` points across the one physical CFL
    interval.  This adapter keeps quick/strict resolution changes explicit
    without inventing a nonexistent below-surface branch.
    """

    return CFLAnalyticEos(grid_settings)


def make_cfl_eos(grid_settings: Any = None) -> CFLAnalyticEos:
    """Return one frozen governed CFL EoS instance and reporting grid."""

    return build_cfl_baseline(grid_settings)


__all__ = [
    "BAG_CONSTANT_FOURTH_ROOT_MEV",
    "BAG_CONSTANT_MEV_FM3",
    "BAG_CONSTANT_NATURAL_MEV4",
    "BARYON_CHEMICAL_POTENTIAL_MAX_MEV",
    "BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV",
    "BARYON_DENSITY_MAX_FM3",
    "BARYON_DENSITY_SURFACE_FM3",
    "CFLAnalyticEos",
    "CFLDomainError",
    "CFLFrozenParameters",
    "CFLGridSettings",
    "CFLInversionError",
    "CFLSurfaceState",
    "CFL_DOMAIN_ID",
    "CFL_DEFORMATION_PROFILE_ID",
    "CFL_DEFORMATION_PROFILE_VERSION",
    "CFL_FORMULATION_ID",
    "CFL_FORMULATION_DESCRIPTION_ID",
    "CFL_FORMULATION_VERSION",
    "COMMON_FERMI_MOMENTUM_SURFACE_MEV",
    "ENERGY_DENSITY_MAX_MEV_FM3",
    "ENERGY_DENSITY_SURFACE_MEV_FM3",
    "FROZEN_CFL_PARAMETERS",
    "FROZEN_PARAMETER_SCHEMA_VERSION",
    "FROZEN_PARAMETER_SET_ID",
    "FROZEN_PARAMETER_SET_SHA256",
    "HBAR_C_MEV_FM",
    "PAIRING_GAP_MEV",
    "PHASE1_DISPLAY_REFERENCE_ENERGY_DENSITY_MAX_MEV_FM3",
    "PHASE1_DISPLAY_REFERENCE_ENERGY_DENSITY_SURFACE_MEV_FM3",
    "PRESSURE_MAX_MEV_FM3",
    "PRESSURE_SURFACE_MEV_FM3",
    "QUARK_CHEMICAL_POTENTIAL_MAX_MEV",
    "QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV",
    "SOUND_SPEED_SQUARED_MAX_ENDPOINT",
    "SOUND_SPEED_SQUARED_SURFACE",
    "STRANGE_QUARK_MASS_MEV",
    "build_cfl_baseline",
    "make_cfl_eos",
]

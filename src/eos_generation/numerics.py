"""Numerics for controlled BSk24 / BSk25 experiments."""

from __future__ import annotations

import hashlib
import math
import pandas as pd
from .assessment import DEFORMATION_SUPPORT_SIGMAS, _meaningful_support_interval
from .baseline import COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3, MODEL_VERSION, baseline_definition
from .deformation import (
    BSK24_RETAINED_EPSILON_MATCH_MEV_FM3,
    BSK24_RETAINED_EPSILON_MAX_MEV_FM3,
    PRIMARY_AMPLITUDES,
    PRIMARY_EPSILON0_MEV_FM3,
    PRIMARY_SIGMA_MEV_FM3,
)
from .storage import canonical_json, json_clean
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any, Iterable


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


EXTENDED_STELLAR_DIAGNOSTICS_CASE_POLICIES = ("endpoints", "all-accepted")


DEFAULT_MAXIMUM_MASS_THRESHOLD_MSUN = 1.95


def _canonical_zero(value: float) -> float:
    """Return positive zero for either IEEE-754 signed-zero representation."""
    return 0.0 if value == 0.0 else value


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number, not a coerced value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_number(name: str, value: Any) -> float:
    result = _finite_number(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _unique_floats(
    name: str, values: Iterable[Any], *, positive: bool = False
) -> tuple[float, ...]:
    normalized: list[float] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        number = (
            _positive_number(f"{name}[{index}]", value)
            if positive
            else _finite_number(f"{name}[{index}]", value)
        )
        number = _canonical_zero(number)
        key = number.hex()
        if key not in seen:
            normalized.append(number)
            seen.add(key)
    if not normalized:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(normalized)


def _safe_identifier(value: float) -> str:
    if value == 0.0:
        return "0"
    sign = "p" if value > 0.0 else "m"
    magnitude = (
        format(abs(value), ".12g").replace(".", "p").replace("-", "m").replace("+", "")
    )
    return f"{sign}{magnitude}"


def deterministic_case_id(
    *,
    amplitude: float,
    delta_mev_fm3: float,
    epsilon0_mev_fm3: float,
    sigma_mev_fm3: float,
    epsilon_match_mev_fm3: float | None = None,
    matter_model: str = "bsk24",
) -> str:
    """Return a readable, collision-resistant identifier for one proposal."""
    amplitude = _canonical_zero(_finite_number("amplitude", amplitude))
    delta_mev_fm3 = _canonical_zero(_finite_number("delta_mev_fm3", delta_mev_fm3))
    epsilon0_mev_fm3 = _canonical_zero(
        _finite_number("epsilon0_mev_fm3", epsilon0_mev_fm3)
    )
    sigma_mev_fm3 = _canonical_zero(_finite_number("sigma_mev_fm3", sigma_mev_fm3))
    payload = {
        "amplitude": amplitude.hex(),
        "delta_mev_fm3": delta_mev_fm3.hex(),
        "epsilon0_mev_fm3": epsilon0_mev_fm3.hex(),
        "sigma_mev_fm3": sigma_mev_fm3.hex(),
    }
    if epsilon_match_mev_fm3 is not None:
        epsilon_match_mev_fm3 = _canonical_zero(
            _finite_number("epsilon_match_mev_fm3", epsilon_match_mev_fm3)
        )
        payload["epsilon_match_mev_fm3"] = epsilon_match_mev_fm3.hex()
    baseline_definition(matter_model)
    if matter_model != "bsk24":
        payload["matter_model"] = matter_model
    suffix = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:10]
    match_prefix = (
        ""
        if epsilon_match_mev_fm3 is None
        else f"m{_safe_identifier(epsilon_match_mev_fm3)}_"
    )
    return (
        f"{'bsk25_' if matter_model == 'bsk25' else ''}{match_prefix}d{_safe_identifier(delta_mev_fm3)}"
        f"_a{_safe_identifier(amplitude)}_{suffix}"
    )


def bsk24_physical_baseline_id(
    epsilon_match_mev_fm3: float | None = None,
    *, matter_model: str = "bsk24",
) -> str:
    """Return the physical identity shared by logical BSk24 A=0 controls.

    The exploratory matching anchor changes the reconstructed baryon-density
    normalization, so it remains part of the physical baseline identity even
    though the undeformed pressure curve is the same.
    """

    definition = baseline_definition(matter_model)
    effective_match = (
        definition.standard_anchor_epsilon_mev_fm3
        if epsilon_match_mev_fm3 is None
        else _positive_number("epsilon_match_mev_fm3", epsilon_match_mev_fm3)
    )
    payload = {
        "matter_model": matter_model,
        "baseline_model_version": MODEL_VERSION,
        "epsilon_match_mev_fm3": effective_match.hex(),
        "retained_epsilon_max_mev_fm3": (definition.retained_epsilon_max_mev_fm3.hex()),
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{matter_model}_baseline_{digest[:16]}"


@dataclass(frozen=True)
class BSk24ThermodynamicStage:
    """One named C4-consistent thermodynamic grid."""

    name: str
    lower_points: int
    upper_points: int
    causal_root_xtol_mev_fm3: float = DEFAULT_CONFIG.hadronic.causal_root_xtol
    causal_root_rtol: float = DEFAULT_CONFIG.hadronic.causal_root_rtol

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("thermodynamic stage name must not be empty")
        for key in ("lower_points", "upper_points"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            if value < 17 or value % 2 == 0:
                raise ValueError(f"{key} must be an odd integer of at least 17")
        _positive_number("causal_root_xtol_mev_fm3", self.causal_root_xtol_mev_fm3)
        _positive_number("causal_root_rtol", self.causal_root_rtol)

    def grid_settings(self) -> BSk24GridSettings:
        from .thermodynamics import BSk24GridSettings

        return BSk24GridSettings(
            lower_points=self.lower_points,
            upper_points=self.upper_points,
            causal_root_xtol_mev_fm3=self.causal_root_xtol_mev_fm3,
            causal_root_rtol=self.causal_root_rtol,
        )


@dataclass(frozen=True)
class BSk24TOVStage:
    """One named shared-solver sequence and radial-profile configuration."""

    name: str
    sequence_points: int
    rtol: float
    atol: float
    radial_profile_points: int = DEFAULT_CONFIG.tov.dense_profile_points

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("TOV stage name must not be empty")
        for key, minimum in (("sequence_points", 5), ("radial_profile_points", 3)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer of at least {minimum}")
        _positive_number("rtol", self.rtol)
        _positive_number("atol", self.atol)


def _default_thermodynamic_stages() -> tuple[BSk24ThermodynamicStage, ...]:
    return (
        BSk24ThermodynamicStage("coarse", 1025, 2049),
        BSk24ThermodynamicStage("standard", 2049, 4097),
        BSk24ThermodynamicStage("refined", 4097, 8193),
    )


def _default_tov_stages() -> tuple[BSk24TOVStage, ...]:
    return (
        BSk24TOVStage("current", 61, 1.0e-8, 1.0e-10, 601),
        BSk24TOVStage("finer_grid", 121, 1.0e-8, 1.0e-10, 601),
        BSk24TOVStage("tighter_ode", 121, 1.0e-10, 1.0e-12, 1201),
    )


@dataclass(frozen=True)
class BSk24TrialConfig:
    """Complete governed configuration for one deterministic BSk24 trial.

    Defaults reproduce the parameter meanings and numerical stages of the
    accepted windowed reference experiment. Stellar work is disabled by
    default so notebook execution cannot accidentally launch an expensive run.
    """

    amplitudes: tuple[float, ...] = PRIMARY_AMPLITUDES
    epsilon_match_mev_fm3: float | None = None
    epsilon0_mev_fm3: float = PRIMARY_EPSILON0_MEV_FM3
    sigma_mev_fm3: float = PRIMARY_SIGMA_MEV_FM3
    deltas_mev_fm3: tuple[float, ...] = (30.0, 40.0, 45.0)
    fixed_masses_msun: tuple[float, ...] = (1.4,)
    thermodynamic_stages: tuple[BSk24ThermodynamicStage, ...] = field(
        default_factory=_default_thermodynamic_stages
    )
    tov_stages: tuple[BSk24TOVStage, ...] = field(default_factory=_default_tov_stages)
    raw_gate_lower_points: int = 4097
    raw_gate_upper_points: int = 16385
    central_pressure_min_mev_fm3: float = 2.0
    fixed_mass_root_xtol_mev_fm3: float = 1.0e-7
    stellar_enabled: bool = False
    requested_observables: tuple[str, ...] | None = None
    maximum_mass_threshold_msun: float = DEFAULT_MAXIMUM_MASS_THRESHOLD_MSUN
    maximum_mass_initial_points: int = 17
    extended_stellar_diagnostics_enabled: bool = False
    extended_stellar_diagnostics_case_policy: str = "endpoints"
    diagnostic_delta_mev_fm3: float = 40.0
    zero_amplitude_control_owner: bool | None = None
    matter_model: str = "bsk24"

    def __post_init__(self) -> None:
        baseline_definition(self.matter_model)
        amplitudes = _unique_floats("amplitudes", self.amplitudes)
        epsilon_match = self.epsilon_match_mev_fm3
        if epsilon_match is not None:
            epsilon_match = _positive_number("epsilon_match_mev_fm3", epsilon_match)
            if epsilon_match == self.model_definition.standard_anchor_epsilon_mev_fm3:
                epsilon_match = None
            elif not (
                self.model_definition.core_entry_epsilon_mev_fm3
                < epsilon_match
                < self.model_definition.retained_epsilon_max_mev_fm3
            ):
                raise ValueError(
                    "exploratory epsilon_match_mev_fm3 must lie strictly "
                    "above the retained homogeneous-core entry "
                    f"({self.model_definition.core_entry_epsilon_mev_fm3:.12g}) and below "
                    "the retained causal endpoint "
                    f"({self.model_definition.retained_epsilon_max_mev_fm3:.12g}) for "
                    f"{self.matter_model.upper()}. Use epsilon_match='standard' for the standard anchor."
                )
        deltas = _unique_floats("deltas_mev_fm3", self.deltas_mev_fm3, positive=True)
        masses = _unique_floats(
            "fixed_masses_msun", self.fixed_masses_msun, positive=True
        )
        if any(mass >= 10.0 for mass in masses):
            raise ValueError("fixed_masses_msun must be below 10 solar masses")
        epsilon0 = _positive_number("epsilon0_mev_fm3", self.epsilon0_mev_fm3)
        sigma = _positive_number("sigma_mev_fm3", self.sigma_mev_fm3)
        effective_epsilon_match = (
            self.model_definition.standard_anchor_epsilon_mev_fm3
            if epsilon_match is None
            else epsilon_match
        )
        if (
            _meaningful_support_interval(
                epsilon0_mev_fm3=epsilon0,
                sigma_mev_fm3=sigma,
                epsilon_match_mev_fm3=effective_epsilon_match,
                epsilon_max_mev_fm3=self.model_definition.retained_epsilon_max_mev_fm3,
            )
            is None
        ):
            raise ValueError(
                "deformation center and width have no meaningful in-domain "
                f"support: the {DEFORMATION_SUPPORT_SIGMAS:g}-sigma interval "
                "must overlap the deformable domain strictly above "
                f"epsilon_match and below the retained {self.matter_model.upper()} endpoint"
            )
        central_pressure = _positive_number(
            "central_pressure_min_mev_fm3", self.central_pressure_min_mev_fm3
        )
        root_xtol = _positive_number(
            "fixed_mass_root_xtol_mev_fm3", self.fixed_mass_root_xtol_mev_fm3
        )
        diagnostic_delta = _positive_number(
            "diagnostic_delta_mev_fm3", self.diagnostic_delta_mev_fm3
        )
        maximum_mass_threshold = _positive_number(
            "maximum_mass_threshold_msun", self.maximum_mass_threshold_msun
        )
        if maximum_mass_threshold >= 10.0:
            raise ValueError("maximum_mass_threshold_msun must be below 10")
        if (
            isinstance(self.maximum_mass_initial_points, bool)
            or not isinstance(self.maximum_mass_initial_points, int)
            or self.maximum_mass_initial_points < 9
            or self.maximum_mass_initial_points % 2 == 0
        ):
            raise ValueError(
                "maximum_mass_initial_points must be an odd integer of at least 9"
            )
        for key in (
            "stellar_enabled",
            "extended_stellar_diagnostics_enabled",
        ):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be boolean")
        if self.zero_amplitude_control_owner is not None and not isinstance(
            self.zero_amplitude_control_owner, bool
        ):
            raise ValueError("zero_amplitude_control_owner must be boolean or None")
        if (
            not isinstance(self.extended_stellar_diagnostics_case_policy, str)
            or self.extended_stellar_diagnostics_case_policy
            not in EXTENDED_STELLAR_DIAGNOSTICS_CASE_POLICIES
        ):
            raise ValueError(
                "extended_stellar_diagnostics_case_policy must be one of "
                f"{EXTENDED_STELLAR_DIAGNOSTICS_CASE_POLICIES}"
            )
        for key in ("raw_gate_lower_points", "raw_gate_upper_points"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 17:
                raise ValueError(f"{key} must be an integer of at least 17")
        thermo = tuple(self.thermodynamic_stages)
        tov = tuple(self.tov_stages)
        if not thermo:
            raise ValueError("at least one thermodynamic stage is required")
        if self.stellar_enabled and not tov:
            raise ValueError("background TOV screening requires at least one TOV stage")
        if self.extended_stellar_diagnostics_enabled and not self.stellar_enabled:
            raise ValueError(
                "extended stellar diagnostics require background TOV screening"
            )
        if any(not isinstance(item, BSk24ThermodynamicStage) for item in thermo):
            raise TypeError(
                "thermodynamic_stages must contain BSk24ThermodynamicStage values"
            )
        if any(not isinstance(item, BSk24TOVStage) for item in tov):
            raise TypeError("tov_stages must contain BSk24TOVStage values")
        for label, items in (("thermodynamic", thermo), ("TOV", tov)):
            names = [item.name for item in items]
            if len(names) != len(set(names)):
                raise ValueError(f"{label} stage names must be unique")
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "epsilon_match_mev_fm3", epsilon_match)
        object.__setattr__(self, "deltas_mev_fm3", deltas)
        object.__setattr__(self, "fixed_masses_msun", masses)
        object.__setattr__(self, "epsilon0_mev_fm3", epsilon0)
        object.__setattr__(self, "sigma_mev_fm3", sigma)
        object.__setattr__(self, "central_pressure_min_mev_fm3", central_pressure)
        object.__setattr__(self, "fixed_mass_root_xtol_mev_fm3", root_xtol)
        object.__setattr__(self, "diagnostic_delta_mev_fm3", diagnostic_delta)
        object.__setattr__(self, "maximum_mass_threshold_msun", maximum_mass_threshold)
        object.__setattr__(self, "thermodynamic_stages", thermo)
        object.__setattr__(self, "tov_stages", tov)

    @property
    def model_definition(self):
        return baseline_definition(self.matter_model)

    @property
    def a0_was_injected(self) -> bool:
        if not self.a0_deduplication_enabled:
            return self.logical_a0_was_injected
        return bool(self.zero_amplitude_control_owner and self.logical_a0_was_injected)

    @property
    def logical_a0_was_injected(self) -> bool:
        return not any(value == 0.0 for value in self.amplitudes)

    @property
    def a0_deduplication_enabled(self) -> bool:
        return self.zero_amplitude_control_owner is not None

    @property
    def logical_amplitudes(self) -> tuple[float, ...]:
        if not self.a0_deduplication_enabled:
            if not self.logical_a0_was_injected:
                return self.amplitudes
            return (0.0, *self.amplitudes)
        return (0.0, *(value for value in self.amplitudes if value != 0.0))

    @property
    def effective_amplitudes(self) -> tuple[float, ...]:
        if not self.a0_deduplication_enabled:
            return self.logical_amplitudes
        if self.zero_amplitude_control_owner is False:
            return tuple(value for value in self.logical_amplitudes if value != 0.0)
        return self.logical_amplitudes

    @property
    def zero_amplitude_physical_case_id(self) -> str | None:
        if not self.a0_deduplication_enabled:
            return None
        return bsk24_physical_baseline_id(self.epsilon_match_mev_fm3, matter_model=self.matter_model)

    @property
    def exploratory_anchor_requested(self) -> bool:
        return self.epsilon_match_mev_fm3 is not None

    @property
    def effective_epsilon_match_mev_fm3(self) -> float:
        return (
            self.model_definition.standard_anchor_epsilon_mev_fm3
            if self.epsilon_match_mev_fm3 is None
            else self.epsilon_match_mev_fm3
        )

    @property
    def background_tov_requested(self) -> bool:
        return self.stellar_enabled

    @property
    def fixed_mass_background_requested(self) -> bool:
        return self.stellar_enabled and (
            "fixed_mass" in self.requested_observables
            if self.requested_observables is not None
            else True
        )

    @property
    def maximum_mass_requested(self) -> bool:
        return self.stellar_enabled and (
            "maximum_mass" in self.requested_observables
            if self.requested_observables is not None
            else True
        )

    @property
    def retained_stellar_profiles_requested(self) -> bool:
        return self.stellar_enabled and self.extended_stellar_diagnostics_enabled

    @property
    def thermodynamic_residual_reporting_requested(self) -> bool:
        return True

    @property
    def tidal_requested(self) -> bool:
        return self.stellar_enabled


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return strict-JSON records with semantic missing values as ``None``."""

    return json_clean(frame.to_dict(orient="records"))


PROFILE_VERSION = "governed_profiles_v1"
# Values are retained from v1.2.0. Products and plotting are separate choices.
_STRICT_THERMO = (
    ("coarse", 1025, 2049),
    ("standard", 2049, 4097),
    ("refined", 4097, 8193),
)
_PROFILE_ROWS = {
    ("quick", "thermodynamics"): (
        (("smoke", 129, 257), ("smoke_refined", 257, 513)),
        (),
        257,
        1025,
        17,
    ),
    ("quick", "stellar"): (
        (("pilot", 257, 513),),
        (("pilot_background", 17, 1e-8, 1e-10, 301),),
        1025,
        4097,
        9,
    ),
    ("strict", "any"): (
        _STRICT_THERMO,
        (
            ("current", 61, 1e-8, 1e-10, 601),
            ("finer_grid", 121, 1e-8, 1e-10, 601),
            ("tighter_ode", 121, 1e-10, 1e-12, 1201),
        ),
        4097,
        16385,
        17,
    ),
    ("dataset", "any"): (
        _STRICT_THERMO,
        (("dataset", 61, 1e-10, 1e-12, 1201),),
        4097,
        16385,
        17,
    ),
    ("dataset_10_tighter", "any"): (
        _STRICT_THERMO,
        (("dataset_10_tighter", 10, 1e-11, 1e-13, 1201),),
        4097,
        16385,
        17,
    ),
    ("dataset_20", "any"): (
        _STRICT_THERMO,
        (("dataset_20", 20, 1e-10, 1e-12, 1201),),
        4097,
        16385,
        17,
    ),
    ("dataset_40", "any"): (
        _STRICT_THERMO,
        (("dataset_40", 40, 1e-10, 1e-12, 1201),),
        4097,
        16385,
        17,
    ),
    ("dataset_40_curves", "any"): (
        (_STRICT_THERMO[-1],),
        (("dataset_40", 40, 1e-10, 1e-12, 1201),),
        4097,
        16385,
        17,
    ),
    ("dataset_relaxed", "any"): (
        _STRICT_THERMO,
        (("dataset_relaxed", 61, 1e-8, 1e-10, 1201),),
        4097,
        16385,
        17,
    ),
    ("dataset_relaxed_80", "any"): (
        _STRICT_THERMO,
        (("dataset_relaxed_80", 80, 1e-8, 1e-10, 1201),),
        4097,
        16385,
        17,
    ),
}


def precision_profile(name: str, calculation: str) -> dict[str, Any]:
    try:
        thermo, stellar, lower, upper, initial = _PROFILE_ROWS.get(
            (name, calculation), _PROFILE_ROWS.get((name, "any"))
        )
    except TypeError as exc:
        raise ValueError(f"unknown precision {name!r}") from exc
    return {
        "version": PROFILE_VERSION,
        "name": name,
        "thermodynamic_stages": tuple(BSk24ThermodynamicStage(*row) for row in thermo),
        "tov_stages": (
            tuple(BSk24TOVStage(*row) for row in stellar)
            if calculation == "stellar"
            else ()
        ),
        "raw_gate_lower_points": lower,
        "raw_gate_upper_points": upper,
        "maximum_mass_initial_points": initial,
    }

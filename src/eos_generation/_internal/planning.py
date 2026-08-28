"""Immutable configuration and passive planning for BSk24 trials.

This internal module contains no scientific execution, packet writing,
validation, or rendering entry points.  The established
``eos_generation.experiment`` re-exports its stable public names.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Real
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from eos_generation._internal.config import DEFAULT_CONFIG
from eos_generation._internal.artifacts import (
    canonical_json,
    ensure_within_runs,
    json_clean,
    resolve_runs_path,
    runs_root,
)
from eos_generation.bsk24.reconstruction import (
    COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3,
    BSk24GridSettings,
)
from eos_generation.bsk24.baseline import MODEL_VERSION
from eos_generation.bsk24._deformation_bounds import (
    DEFORMATION_SUPPORT_SIGMAS,
    _meaningful_support_interval,
)
from eos_generation.bsk24.deformation import (
    BSK24_RETAINED_EPSILON_MATCH_MEV_FM3,
    BSK24_RETAINED_EPSILON_MAX_MEV_FM3,
    PRIMARY_AMPLITUDES,
    PRIMARY_EPSILON0_MEV_FM3,
    PRIMARY_SIGMA_MEV_FM3,
    PURE_GAUSSIAN_GENERATOR_ID,
    WINDOWED_GAUSSIAN_GENERATOR_ID,
)


TRIAL_PLAN_SCHEMA = "eos_generation_trial_plan_v1"
DEFAULT_OUTPUT_STEM = "bsk24_trial"
PLOT_GROUPS = (
    "none",
    "thermodynamics",
    "stellar",
    "stellar-diagnostics",
    "all-applicable",
)
RESUME_POLICIES = ("error", "resume-completed")
EXTENDED_STELLAR_DIAGNOSTICS_CASE_POLICIES = ("endpoints", "all-accepted")
DEFAULT_MAXIMUM_MASS_THRESHOLD_MSUN = 1.95

_PROTECTED_PACKET_NAMES: frozenset[str] = frozenset()
_WINDOWS_RESERVED_PACKET_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_WINDOWS_INVALID_PACKET_CHARACTERS = frozenset('<>:"/\\|?*')


def _canonical_zero(value: float) -> float:
    """Return positive zero for either IEEE-754 signed-zero representation."""
    return 0.0 if value == 0.0 else value


def _protected_packet_roots() -> tuple[Path, ...]:
    root = runs_root().resolve(strict=False)
    return tuple((root / name).resolve(strict=False) for name in sorted(_PROTECTED_PACKET_NAMES))


def _assert_writable_packet_path(path: str | Path) -> Path:
    """Resolve a packet write target below the repository ``runs`` root."""
    resolved = ensure_within_runs(path)
    for protected in _protected_packet_roots():
        if resolved == protected or protected in resolved.parents:
            raise ValueError(
                "protected packets and their descendants are read-only: "
                f"{resolved}"
            )
    return resolved


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


def _unique_floats(name: str, values: Iterable[Any], *, positive: bool = False) -> tuple[float, ...]:
    normalized: list[float] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        number = _positive_number(f"{name}[{index}]", value) if positive else _finite_number(
            f"{name}[{index}]", value
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
    magnitude = format(abs(value), ".12g").replace(".", "p").replace("-", "m").replace("+", "")
    return f"{sign}{magnitude}"


def deterministic_case_id(
    *,
    amplitude: float,
    delta_mev_fm3: float,
    epsilon0_mev_fm3: float,
    sigma_mev_fm3: float,
    epsilon_match_mev_fm3: float | None = None,
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
            _finite_number(
                "epsilon_match_mev_fm3", epsilon_match_mev_fm3
            )
        )
        payload["epsilon_match_mev_fm3"] = epsilon_match_mev_fm3.hex()
    suffix = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:10]
    match_prefix = (
        ""
        if epsilon_match_mev_fm3 is None
        else f"m{_safe_identifier(epsilon_match_mev_fm3)}_"
    )
    return (
        f"{match_prefix}d{_safe_identifier(delta_mev_fm3)}"
        f"_a{_safe_identifier(amplitude)}_{suffix}"
    )


def bsk24_physical_baseline_id(
    epsilon_match_mev_fm3: float | None = None,
) -> str:
    """Return the physical identity shared by logical BSk24 A=0 controls.

    The exploratory matching anchor changes the reconstructed baryon-density
    normalization, so it remains part of the physical baseline identity even
    though the undeformed pressure curve is the same.
    """

    effective_match = (
        BSK24_RETAINED_EPSILON_MATCH_MEV_FM3
        if epsilon_match_mev_fm3 is None
        else _positive_number(
            "epsilon_match_mev_fm3", epsilon_match_mev_fm3
        )
    )
    payload = {
        "matter_model": "bsk24",
        "baseline_model_version": MODEL_VERSION,
        "epsilon_match_mev_fm3": effective_match.hex(),
        "retained_epsilon_max_mev_fm3": (
            BSK24_RETAINED_EPSILON_MAX_MEV_FM3.hex()
        ),
    }
    digest = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return f"bsk24_baseline_{digest[:16]}"


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
    maximum_mass_threshold_msun: float = DEFAULT_MAXIMUM_MASS_THRESHOLD_MSUN
    maximum_mass_initial_points: int = 17
    extended_stellar_diagnostics_enabled: bool = False
    extended_stellar_diagnostics_case_policy: str = "endpoints"
    diagnostic_delta_mev_fm3: float = 40.0
    requested_plot_groups: tuple[str, ...] = ("all-applicable",)
    output_packet_name: str | None = None
    output_path: str | Path | None = None
    resume_policy: str = "error"
    zero_amplitude_control_owner: bool | None = None

    def __post_init__(self) -> None:
        amplitudes = _unique_floats("amplitudes", self.amplitudes)
        epsilon_match = self.epsilon_match_mev_fm3
        if epsilon_match is not None:
            epsilon_match = _positive_number(
                "epsilon_match_mev_fm3", epsilon_match
            )
            if epsilon_match == BSK24_RETAINED_EPSILON_MATCH_MEV_FM3:
                epsilon_match = None
            elif not (
                COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3
                < epsilon_match
                < BSK24_RETAINED_EPSILON_MAX_MEV_FM3
            ):
                raise ValueError(
                    "exploratory epsilon_match_mev_fm3 must lie strictly "
                    "above the retained homogeneous-core entry "
                    f"({COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3:.12g}) and below "
                    "the retained causal endpoint "
                    f"({BSK24_RETAINED_EPSILON_MAX_MEV_FM3:.12g})"
                )
        deltas = _unique_floats("deltas_mev_fm3", self.deltas_mev_fm3, positive=True)
        masses = _unique_floats("fixed_masses_msun", self.fixed_masses_msun, positive=True)
        if any(mass >= 10.0 for mass in masses):
            raise ValueError("fixed_masses_msun must be below 10 solar masses")
        epsilon0 = _positive_number("epsilon0_mev_fm3", self.epsilon0_mev_fm3)
        sigma = _positive_number("sigma_mev_fm3", self.sigma_mev_fm3)
        effective_epsilon_match = (
            BSK24_RETAINED_EPSILON_MATCH_MEV_FM3
            if epsilon_match is None
            else epsilon_match
        )
        if _meaningful_support_interval(
            epsilon0_mev_fm3=epsilon0,
            sigma_mev_fm3=sigma,
            epsilon_match_mev_fm3=effective_epsilon_match,
            epsilon_max_mev_fm3=BSK24_RETAINED_EPSILON_MAX_MEV_FM3,
        ) is None:
            raise ValueError(
                "deformation center and width have no meaningful in-domain "
                f"support: the {DEFORMATION_SUPPORT_SIGMAS:g}-sigma interval "
                "must overlap the deformable domain strictly above "
                "epsilon_match and below the retained BSk24 endpoint"
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
        if (
            self.zero_amplitude_control_owner is not None
            and not isinstance(self.zero_amplitude_control_owner, bool)
        ):
            raise ValueError(
                "zero_amplitude_control_owner must be boolean or None"
            )
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
            raise ValueError("extended stellar diagnostics require background TOV screening")
        if any(not isinstance(item, BSk24ThermodynamicStage) for item in thermo):
            raise TypeError("thermodynamic_stages must contain BSk24ThermodynamicStage values")
        if any(not isinstance(item, BSk24TOVStage) for item in tov):
            raise TypeError("tov_stages must contain BSk24TOVStage values")
        for label, items in (("thermodynamic", thermo), ("TOV", tov)):
            names = [item.name for item in items]
            if len(names) != len(set(names)):
                raise ValueError(f"{label} stage names must be unique")
        groups = tuple(dict.fromkeys(str(value) for value in self.requested_plot_groups))
        invalid_groups = sorted(set(groups) - set(PLOT_GROUPS))
        if not groups or invalid_groups:
            raise ValueError(f"requested_plot_groups must use {PLOT_GROUPS}; invalid={invalid_groups}")
        if "none" in groups and len(groups) != 1:
            raise ValueError("plot group 'none' must be used alone")
        if "all-applicable" in groups and len(groups) > 1:
            groups = ("all-applicable",)
        if self.resume_policy not in RESUME_POLICIES:
            raise ValueError(f"resume_policy must be one of {RESUME_POLICIES}")
        if self.output_packet_name is not None:
            name_path = Path(self.output_packet_name)
            name = self.output_packet_name
            reserved_stem = name.split(".", 1)[0].upper()
            if (
                not name
                or name_path.is_absolute()
                or bool(name_path.drive)
                or len(name_path.parts) != 1
                or name in {".", ".."}
                or name[-1] in {".", " "}
                or any(ord(character) < 32 for character in name)
                or any(
                    character in _WINDOWS_INVALID_PACKET_CHARACTERS
                    for character in name
                )
                or reserved_stem in _WINDOWS_RESERVED_PACKET_STEMS
            ):
                raise ValueError(
                    "output_packet_name must be one portable, non-reserved "
                    "directory name without path separators, Windows-invalid "
                    "characters, or a trailing dot/space"
                )
        resolved_output_path: Path | None = None
        if self.output_path is not None:
            # Scope is validated here and again when planning or execution
            # resolves the destination.
            resolved_output_path = resolve_runs_path(self.output_path)
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
        object.__setattr__(self, "requested_plot_groups", groups)
        object.__setattr__(self, "output_path", resolved_output_path)

    @property
    def a0_was_injected(self) -> bool:
        if not self.a0_deduplication_enabled:
            return self.logical_a0_was_injected
        return bool(
            self.zero_amplitude_control_owner
            and self.logical_a0_was_injected
        )

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
            return tuple(
                value for value in self.logical_amplitudes if value != 0.0
            )
        return self.logical_amplitudes

    @property
    def zero_amplitude_physical_case_id(self) -> str | None:
        if not self.a0_deduplication_enabled:
            return None
        return bsk24_physical_baseline_id(self.epsilon_match_mev_fm3)

    @property
    def exploratory_anchor_requested(self) -> bool:
        return self.epsilon_match_mev_fm3 is not None

    @property
    def effective_epsilon_match_mev_fm3(self) -> float:
        return (
            BSK24_RETAINED_EPSILON_MATCH_MEV_FM3
            if self.epsilon_match_mev_fm3 is None
            else self.epsilon_match_mev_fm3
        )

    @property
    def background_tov_requested(self) -> bool:
        return self.stellar_enabled

    @property
    def fixed_mass_background_requested(self) -> bool:
        return self.stellar_enabled

    @property
    def tidal_requested(self) -> bool:
        return self.stellar_enabled

    @property
    def reference_compatible(self) -> bool:
        return (
            self.epsilon_match_mev_fm3 is None
            and self.maximum_mass_threshold_msun
            == DEFAULT_MAXIMUM_MASS_THRESHOLD_MSUN
            and self.maximum_mass_initial_points == 17
            and self.epsilon0_mev_fm3 == PRIMARY_EPSILON0_MEV_FM3
            and self.sigma_mev_fm3 == PRIMARY_SIGMA_MEV_FM3
            and self.deltas_mev_fm3 == (30.0, 40.0, 45.0)
            and self.effective_amplitudes == PRIMARY_AMPLITUDES
            and self.thermodynamic_stages == _default_thermodynamic_stages()
            and self.tov_stages == _default_tov_stages()
            and self.central_pressure_min_mev_fm3 == 2.0
            and self.fixed_mass_root_xtol_mev_fm3 == 1.0e-7
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not self.a0_deduplication_enabled:
            data.pop("zero_amplitude_control_owner", None)
        if self.epsilon_match_mev_fm3 is None:
            data.pop("epsilon_match_mev_fm3", None)
        if self.extended_stellar_diagnostics_case_policy == "endpoints":
            # Keep established default configuration hashes stable. The
            # opt-in reporting policy is serialized explicitly.
            data.pop("extended_stellar_diagnostics_case_policy", None)
        new_execution_controls_active = (
            self.maximum_mass_threshold_msun
            != DEFAULT_MAXIMUM_MASS_THRESHOLD_MSUN
        ) or self.maximum_mass_initial_points != 17
        if not new_execution_controls_active:
            for key in (
                "maximum_mass_threshold_msun",
                "maximum_mass_initial_points",
            ):
                data.pop(key, None)
        data["effective_amplitudes"] = list(self.effective_amplitudes)
        data["a0_identity_control_injected"] = self.a0_was_injected
        if self.a0_deduplication_enabled:
            data["logical_amplitudes"] = list(self.logical_amplitudes)
            data["logical_a0_identity_control_injected"] = (
                self.logical_a0_was_injected
            )
            data["zero_amplitude_physical_case_id"] = (
                self.zero_amplitude_physical_case_id
            )
        data["generator_id"] = WINDOWED_GAUSSIAN_GENERATOR_ID
        data["preserved_generator_id"] = PURE_GAUSSIAN_GENERATOR_ID
        data["reference_compatible"] = self.reference_compatible
        return json_clean(data)

    def expanded_dict(self) -> dict[str, Any]:
        """Return every resolved calculation and operational setting."""

        data = asdict(self)
        if not self.a0_deduplication_enabled:
            data.pop("zero_amplitude_control_owner", None)
        data["effective_amplitudes"] = list(self.effective_amplitudes)
        data["a0_identity_control_injected"] = self.a0_was_injected
        if self.a0_deduplication_enabled:
            data["logical_amplitudes"] = list(self.logical_amplitudes)
            data["logical_a0_identity_control_injected"] = (
                self.logical_a0_was_injected
            )
            data["zero_amplitude_physical_case_id"] = (
                self.zero_amplitude_physical_case_id
            )
        data["effective_epsilon_match_mev_fm3"] = (
            self.effective_epsilon_match_mev_fm3
        )
        data["generator_id"] = WINDOWED_GAUSSIAN_GENERATOR_ID
        data["preserved_generator_id"] = PURE_GAUSSIAN_GENERATOR_ID
        data["reference_compatible"] = self.reference_compatible
        return json_clean(data)

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def deterministic_hash(self) -> str:
        # Artifact destination and resume permission are operational controls,
        # not part of the scientific/numerical calculation definition.
        definition = dict(self.to_dict())
        for key in ("output_packet_name", "output_path", "resume_policy"):
            definition.pop(key, None)
        return hashlib.sha256(canonical_json(definition).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BSk24TrialConfig":
        values = dict(data)
        supplied = dict(values)
        if "$schema" in values:
            schema_annotation = values.pop("$schema")
            if not isinstance(schema_annotation, str):
                raise ValueError("top-level $schema must be a string")
        for extra in (
            "effective_amplitudes",
            "logical_amplitudes",
            "a0_identity_control_injected",
            "logical_a0_identity_control_injected",
            "zero_amplitude_physical_case_id",
            "generator_id",
            "preserved_generator_id",
            "reference_compatible",
        ):
            values.pop(extra, None)
        if "thermodynamic_stages" in values:
            values["thermodynamic_stages"] = tuple(
                item
                if isinstance(item, BSk24ThermodynamicStage)
                else BSk24ThermodynamicStage(**item)
                for item in values["thermodynamic_stages"]
            )
        if "tov_stages" in values:
            values["tov_stages"] = tuple(
                item if isinstance(item, BSk24TOVStage) else BSk24TOVStage(**item)
                for item in values["tov_stages"]
            )
        for key in (
            "amplitudes",
            "deltas_mev_fm3",
            "fixed_masses_msun",
            "requested_plot_groups",
        ):
            if key in values and values[key] is not None:
                values[key] = tuple(values[key])
        result = cls(**values)
        canonical = result.to_dict()
        for name in (
            "effective_amplitudes",
            "logical_amplitudes",
            "a0_identity_control_injected",
            "logical_a0_identity_control_injected",
            "zero_amplitude_physical_case_id",
        ):
            if name in supplied and supplied[name] != canonical.get(name):
                raise ValueError(
                    f"saved BSk24 configuration {name} disagrees with its inputs"
                )
        return result


@dataclass(frozen=True)
class BSk24TrialPlan:
    """Read-only, calculation-free trial preview."""

    config: BSk24TrialConfig
    case_table: pd.DataFrame
    logical_alias_table: pd.DataFrame
    a0_identity_control_injected: bool
    anticipated_plot_inventory: pd.DataFrame
    estimates: Mapping[str, int]
    output_path: Path

    @property
    def logical_case_table(self) -> pd.DataFrame:
        if not self.config.a0_deduplication_enabled:
            return self.case_table
        frames = [
            frame
            for frame in (self.case_table, self.logical_alias_table)
            if not frame.empty
        ]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values(
            ["logical_delta_index", "logical_amplitude_index"],
            kind="stable",
            ignore_index=True,
        )

    def to_dict(self) -> dict[str, Any]:
        expanded = self.config.expanded_dict()
        operational = {
            name: expanded.pop(name)
            for name in ("output_packet_name", "output_path", "resume_policy")
        }
        result = {
            "schema_id": TRIAL_PLAN_SCHEMA,
            "configuration_hash": self.config.deterministic_hash(),
            "expanded_configuration": expanded,
            "operational_destination": operational,
            "a0_identity_control_injected": self.a0_identity_control_injected,
            "case_table": _json_records(self.case_table),
            "anticipated_plot_inventory": _json_records(
                self.anticipated_plot_inventory
            ),
            "estimates": dict(self.estimates),
            "output_path": json_clean(self.output_path),
            "planning_is_passive": True,
            "scientific_solver_calls": 0,
            "filesystem_writes": 0,
        }
        if self.config.a0_deduplication_enabled:
            result["logical_alias_table"] = _json_records(
                self.logical_alias_table
            )
        return result

    def _repr_html_(self) -> str:
        return (
            "<h3>BSk24 trial plan</h3>"
            + self.logical_case_table.to_html(index=False)
            + "<h4>Anticipated figure applicability</h4>"
            + self.anticipated_plot_inventory.to_html(index=False)
        )


@dataclass(frozen=True)
class PlotSpec:
    filename: str
    group: str
    required_tables: tuple[str, ...]
    prerequisite: str


PLOT_REGISTRY: tuple[PlotSpec, ...] = (
    PlotSpec("window_profiles.png", "thermodynamics", ("raw_gate_profiles.csv",), "one accepted proposal"),
    PlotSpec("gaussian_realization.png", "thermodynamics", ("raw_gate_profiles.csv",), "one accepted proposal"),
    PlotSpec("raw_cs2_full_domain.png", "thermodynamics", ("raw_gate_profiles.csv",), "one accepted proposal"),
    PlotSpec("raw_cs2_anchor_core_zoom.png", "thermodynamics", ("raw_gate_profiles.csv",), "one accepted proposal"),
    PlotSpec("delta_cs2.png", "thermodynamics", ("raw_gate_profiles.csv",), "one accepted proposal"),
    PlotSpec("pressure_response.png", "thermodynamics", ("thermodynamic_profiles.csv",), "one accepted reconstructed proposal"),
    PlotSpec("baryon_density_response.png", "thermodynamics", ("thermodynamic_profiles.csv",), "one accepted reconstructed proposal"),
    PlotSpec("effective_baryon_enthalpy_response.png", "thermodynamics", ("thermodynamic_profiles.csv",), "one accepted reconstructed proposal"),
    PlotSpec("gamma_eff_response.png", "thermodynamics", ("thermodynamic_profiles.csv",), "one accepted reconstructed proposal"),
    PlotSpec("thermodynamic_residuals.png", "thermodynamics", ("thermodynamic_residuals.csv",), "sampled PCHIP residual profiles"),
    PlotSpec("stellar_mr_k2_lambda.png", "stellar", ("stellar_sequences.csv",), "stellar execution with successful configurations"),
    PlotSpec("observable_response_vs_amplitude.png", "stellar", ("fixed_mass_observables.csv",), "at least two amplitudes at one Delta"),
    PlotSpec("observable_response_vs_delta.png", "stellar", ("fixed_mass_observables.csv",), "at least two Delta values at one amplitude"),
    PlotSpec("a0_identity.png", "thermodynamics", ("a0_identity_table.csv",), "A=0 identity control"),
    PlotSpec("radial_structure_profiles.png", "stellar-diagnostics", ("radial_profiles.csv",), "fixed-mass radial profiles"),
    PlotSpec("deformation_support_fractions.png", "stellar-diagnostics", ("deformation_support_fractions.csv",), "realized-deformation radial support"),
    PlotSpec("outside_support_control.png", "stellar-diagnostics", ("outside_support_control.csv",), "specifically constructed outside-support controls"),
    PlotSpec("turning_point_sequences.png", "stellar-diagnostics", ("turning_point_sequences.csv",), "explicit turning-point refinement"),
    PlotSpec("turning_point_derivatives.png", "stellar-diagnostics", ("turning_point_sequences.csv",), "explicit turning-point derivative refinement"),
    PlotSpec("baryonic_mass_vs_mass.png", "stellar-diagnostics", ("baryonic_observables.csv",), "baryon-number integration at two or more masses"),
    PlotSpec("binding_energy_vs_mass.png", "stellar-diagnostics", ("baryonic_observables.csv",), "baryon-number integration at two or more masses"),
    PlotSpec("stellar_response_across_mass.png", "stellar-diagnostics", ("fixed_mass_observables.csv", "radial_profiles.csv"), "exact fixed-mass endpoint response at two or more requested masses"),
    PlotSpec("baryonic_response_across_mass.png", "stellar-diagnostics", ("baryonic_response_across_mass.csv",), "common baryonic-mass support"),
    PlotSpec("odd_even_response.png", "stellar-diagnostics", ("odd_even_response.csv",), "matched positive/negative amplitudes plus A=0"),
    PlotSpec("matched_area_comparison.png", "stellar-diagnostics", ("matched_area_comparison.csv",), "specifically constructed matched-integrated-strength cases"),
    PlotSpec("numerical_error_summary.png", "stellar-diagnostics", ("numerical_error_summary.csv",), "multiple numerical stages"),
)


def _selected_groups(groups: Sequence[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(str(value) for value in groups))
    invalid = sorted(set(values) - set(PLOT_GROUPS))
    if not values or invalid:
        raise ValueError(f"plot groups must use {PLOT_GROUPS}; invalid={invalid}")
    if "none" in values:
        if len(values) != 1:
            raise ValueError("plot group 'none' must be used alone")
        return ()
    if "all-applicable" in values:
        return ("thermodynamics", "stellar", "stellar-diagnostics")
    return values


def _output_path(config: BSk24TrialConfig) -> Path:
    if config.output_path is not None:
        resolved = _assert_writable_packet_path(
            resolve_runs_path(config.output_path)
        )
        if resolved == runs_root().resolve(strict=False):
            raise ValueError(
                "output_path must name a packet below runs, "
                "not the output root itself"
            )
        return resolved
    name = config.output_packet_name
    if name is None:
        name = f"{DEFAULT_OUTPUT_STEM}_{config.deterministic_hash()[:12]}"
    resolved = _assert_writable_packet_path(runs_root() / name)
    if resolved == runs_root().resolve(strict=False):
        raise ValueError(
            "output_packet_name must resolve below runs"
        )
    return resolved


def _case_row(
    config: BSk24TrialConfig,
    *,
    amplitude: float,
    delta: float,
    delta_index: int,
    amplitude_index: int,
    planned_for_execution: bool,
) -> dict[str, Any]:
    case_id = deterministic_case_id(
        amplitude=amplitude,
        delta_mev_fm3=delta,
        epsilon0_mev_fm3=config.epsilon0_mev_fm3,
        sigma_mev_fm3=config.sigma_mev_fm3,
        epsilon_match_mev_fm3=config.epsilon_match_mev_fm3,
    )
    physical_case_id = (
        config.zero_amplitude_physical_case_id
        if amplitude == 0.0 and config.a0_deduplication_enabled
        else case_id
    )
    row = {
        "case_id": case_id,
        "amplitude": amplitude,
        "epsilon_match_mev_fm3": config.effective_epsilon_match_mev_fm3,
        "anchor_mode": (
            "exploratory" if config.exploratory_anchor_requested else "standard"
        ),
        "epsilon0_mev_fm3": config.epsilon0_mev_fm3,
        "sigma_mev_fm3": config.sigma_mev_fm3,
        "delta_mev_fm3": delta,
        "is_a0_identity_control": amplitude == 0.0,
        "identity_control_injected": (
            amplitude == 0.0 and config.logical_a0_was_injected
        ),
        "planned_thermodynamic_stages": ",".join(
            stage.name for stage in config.thermodynamic_stages
        ),
        "planned_stellar_stages": (
            ",".join(stage.name for stage in config.tov_stages)
            if config.background_tov_requested
            else ""
        ),
        "anticipated_plot_groups": ",".join(
            _selected_groups(config.requested_plot_groups)
        ),
    }
    if config.a0_deduplication_enabled:
        row.update(
            {
                "physical_case_id": physical_case_id,
                "matter_model": "bsk24",
                "logical_delta_index": delta_index,
                "logical_amplitude_index": amplitude_index,
                "is_physical_case_alias": physical_case_id != case_id,
                "planned_for_execution": planned_for_execution,
                "physical_case_owner": planned_for_execution,
            }
        )
    return row


def _case_rows(config: BSk24TrialConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    owner_delta = min(config.deltas_mev_fm3)
    for delta_index, delta in enumerate(config.deltas_mev_fm3):
        for amplitude_index, amplitude in enumerate(config.logical_amplitudes):
            if (
                config.a0_deduplication_enabled
                and amplitude == 0.0
                and (
                    config.zero_amplitude_control_owner is False
                    or delta != owner_delta
                )
            ):
                continue
            rows.append(
                _case_row(
                    config,
                    amplitude=amplitude,
                    delta=delta,
                    delta_index=delta_index,
                    amplitude_index=amplitude_index,
                    planned_for_execution=True,
                )
            )
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("deterministic case-ID collision")
    return rows


def _logical_alias_rows(config: BSk24TrialConfig) -> list[dict[str, Any]]:
    if not config.a0_deduplication_enabled:
        return []
    aliases: list[dict[str, Any]] = []
    owner_delta = min(config.deltas_mev_fm3)
    for delta_index, delta in enumerate(config.deltas_mev_fm3):
        if config.zero_amplitude_control_owner is True and delta == owner_delta:
            continue
        aliases.append(
            _case_row(
                config,
                amplitude=0.0,
                delta=delta,
                delta_index=delta_index,
                amplitude_index=0,
                planned_for_execution=False,
            )
        )
    return aliases


def _anticipated_plot_inventory(config: BSk24TrialConfig) -> pd.DataFrame:
    groups = set(_selected_groups(config.requested_plot_groups))
    amplitudes = config.effective_amplitudes
    has_pair = any(value > 0.0 and -value in amplitudes for value in amplitudes)
    rows = []
    for spec in PLOT_REGISTRY:
        requested = spec.group in groups
        applicable = requested
        reason = "anticipated applicable after successful execution"
        if not requested:
            applicable = False
            reason = f"plot group {spec.group!r} was not requested"
        elif spec.group == "stellar" and not config.background_tov_requested:
            applicable = False
            reason = "stellar calculations are disabled"
        elif spec.group == "stellar-diagnostics" and not config.extended_stellar_diagnostics_enabled:
            applicable = False
            reason = "extended stellar diagnostics are disabled"
        elif spec.filename == "observable_response_vs_amplitude.png" and len(amplitudes) < 2:
            applicable = False
            reason = "requires at least two amplitudes"
        elif spec.filename == "observable_response_vs_delta.png" and len(config.deltas_mev_fm3) < 2:
            applicable = False
            reason = "requires at least two Delta values"
        elif spec.filename == "odd_even_response.png" and not has_pair:
            applicable = False
            reason = "requires a matched positive/negative amplitude pair"
        elif spec.filename in {
            "outside_support_control.png",
            "turning_point_sequences.png",
            "turning_point_derivatives.png",
            "matched_area_comparison.png",
        }:
            applicable = False
            reason = (
                "requires a separately constructed diagnostic control/refinement; "
                "the trial engine does not infer a new scientific definition"
            )
        rows.append(
            {
                "figure": spec.filename,
                "group": spec.group,
                "anticipated_applicable": applicable,
                "reason": reason,
                "prerequisite": spec.prerequisite,
            }
        )
    return pd.DataFrame(rows)


def prepare_bsk24_trial(config: BSk24TrialConfig) -> BSk24TrialPlan:
    """Validate and preview a trial without scientific calculations or writes."""
    if not isinstance(config, BSk24TrialConfig):
        raise TypeError("config must be a BSk24TrialConfig")
    cases = pd.DataFrame(_case_rows(config))
    aliases = pd.DataFrame(_logical_alias_rows(config))
    if cases.empty and not len(cases.columns) and not aliases.empty:
        cases = aliases.iloc[0:0].copy()
    if config.a0_deduplication_enabled:
        physical_cases = (
            int(cases["physical_case_id"].nunique()) if not cases.empty else 0
        )
        thermodynamic_cases = physical_cases
        stellar_cases = physical_cases if config.background_tov_requested else 0
    else:
        physical_cases = len(cases)
        thermodynamic_cases = 1 + physical_cases
        stellar_cases = (
            thermodynamic_cases if config.background_tov_requested else 0
        )
    estimates = {
        "proposed_deformation_cases": int(len(cases)),
        "direct_baseline_cases": (
            int(bool(config.zero_amplitude_control_owner))
            if config.a0_deduplication_enabled
            else 1
        ),
        "thermodynamic_case_stage_evaluations": int(
            thermodynamic_cases * len(config.thermodynamic_stages)
        ),
        "stellar_case_stage_evaluations": int(
            stellar_cases * len(config.tov_stages)
        ),
        "fixed_mass_root_targets": int(
            stellar_cases
            * len(config.tov_stages)
            * len(config.fixed_masses_msun)
            if config.fixed_mass_background_requested
            else 0
        ),
    }
    if config.a0_deduplication_enabled:
        estimates.update(
            {
                "logical_deformation_cases": int(len(cases) + len(aliases)),
                "physical_deformation_cases": int(physical_cases),
                "deduplicated_logical_case_aliases": int(len(aliases)),
                "baseline_construction_stage_evaluations": len(
                    config.thermodynamic_stages
                ),
            }
        )
    if config.stellar_enabled:
        estimates["sampled_sequence_tidal_targets"] = int(
            stellar_cases
            * sum(stage.sequence_points for stage in config.tov_stages)
        )
        estimates["fixed_mass_tidal_targets"] = int(
            stellar_cases
            * len(config.tov_stages)
            * len(config.fixed_masses_msun)
        )
        estimates["maximum_mass_local_screen_targets"] = int(
            stellar_cases
            * len(config.tov_stages)
            * config.maximum_mass_initial_points
        )
    return BSk24TrialPlan(
        config=config,
        case_table=cases,
        logical_alias_table=aliases,
        a0_identity_control_injected=config.a0_was_injected,
        anticipated_plot_inventory=_anticipated_plot_inventory(config),
        estimates=estimates,
        output_path=_output_path(config),
    )


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return strict-JSON records with semantic missing values as ``None``."""

    return json_clean(frame.to_dict(orient="records"))


__all__ = [
    "BSk24TOVStage",
    "BSk24ThermodynamicStage",
    "BSk24TrialConfig",
    "BSk24TrialPlan",
    "DEFAULT_OUTPUT_STEM",
    "DEFAULT_MAXIMUM_MASS_THRESHOLD_MSUN",
    "EXTENDED_STELLAR_DIAGNOSTICS_CASE_POLICIES",
    "PLOT_GROUPS",
    "PLOT_REGISTRY",
    "PlotSpec",
    "RESUME_POLICIES",
    "TRIAL_PLAN_SCHEMA",
    "bsk24_physical_baseline_id",
    "deterministic_case_id",
    "prepare_bsk24_trial",
]

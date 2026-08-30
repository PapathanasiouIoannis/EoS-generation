"""Immutable, calculation-free planning objects for frozen-baseline CFL trials.

This module records scientific intent and deterministic identities only.  It
does not construct an equation of state, reconstruct a deformation, call a
stellar solver, or write a result packet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from eos_generation._internal.artifacts import (
    canonical_json,
    json_clean,
    resolve_runs_path,
    runs_root,
)
# The established stage dataclasses are intentionally reused only as immutable
# containers for model-neutral grid/ODE profile fields. Their serialized form
# carries no BSk24 equations, anchors, composition, or surface assumptions.
from eos_generation._internal.planning import (
    BSk24TOVStage,
    BSk24ThermodynamicStage,
    DEFAULT_MAXIMUM_MASS_THRESHOLD_MSUN,
    EXTENDED_STELLAR_DIAGNOSTICS_CASE_POLICIES,
    PLOT_GROUPS,
    RESUME_POLICIES,
    _WINDOWS_INVALID_PACKET_CHARACTERS,
    _WINDOWS_RESERVED_PACKET_STEMS,
    _assert_writable_packet_path,
    _anticipated_plot_inventory,
    _finite_number,
    _positive_number,
    _safe_identifier,
    _selected_groups,
    _unique_floats,
)


CFL_TRIAL_PLAN_SCHEMA = "eos_generation_cfl_trial_plan_v1"
CFL_DEFAULT_OUTPUT_STEM = "cfl_trial"


def _frozen_baseline_contract() -> dict[str, Any]:
    """Return and structurally verify the canonical immutable baseline record."""

    from .baseline import (
        CFLAnalyticEos,
        CFL_DEFORMATION_PROFILE_ID,
        CFL_DEFORMATION_PROFILE_VERSION,
        CFL_DOMAIN_ID,
        CFL_FORMULATION_ID,
        CFL_FORMULATION_VERSION,
        ENERGY_DENSITY_MAX_MEV_FM3,
        ENERGY_DENSITY_SURFACE_MEV_FM3,
        FROZEN_CFL_PARAMETERS,
        FROZEN_PARAMETER_SET_ID,
        FROZEN_PARAMETER_SET_SHA256,
    )
    from .deformation import CFL_PRESSURE_PRIMITIVE_POLICY
    from .reconstruction import (
        CFL_RECONSTRUCTION_PROFILE_ID,
        CFL_WINDOWED_EOS_SCHEMA_VERSION,
    )

    payload = FROZEN_CFL_PARAMETERS.to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError("FROZEN_CFL_PARAMETERS.to_dict() must return a mapping")
    record = json_clean(dict(payload))
    required = {
        "schema_version",
        "parameter_set_id",
        "parameter_set_sha256",
        "formulation_id",
        "formulation_version",
        "constants",
        "conventions",
        "domain",
        "surface",
    }
    missing = sorted(required - set(record))
    if missing:
        raise RuntimeError(
            f"frozen CFL baseline record is missing required field {missing[0]!r}"
        )
    expected_scalars = {
        "parameter_set_id": FROZEN_PARAMETER_SET_ID,
        "parameter_set_sha256": FROZEN_PARAMETER_SET_SHA256,
        "formulation_id": CFL_FORMULATION_ID,
        "formulation_version": CFL_FORMULATION_VERSION,
    }
    for name, expected in expected_scalars.items():
        if record.get(name) != expected:
            raise RuntimeError(f"frozen CFL baseline {name} disagrees with its export")
    domain = record.get("domain")
    surface = record.get("surface")
    if not isinstance(domain, Mapping) or domain.get("domain_id") != CFL_DOMAIN_ID:
        raise RuntimeError("frozen CFL baseline domain identity is inconsistent")
    if not isinstance(surface, Mapping):
        raise RuntimeError("frozen CFL baseline surface record is malformed")
    if surface.get("energy_density_mev_fm3") != ENERGY_DENSITY_SURFACE_MEV_FM3:
        raise RuntimeError("frozen CFL surface energy density is inconsistent")
    epsilon_domain = domain.get("energy_density_mev_fm3")
    if (
        not isinstance(epsilon_domain, list)
        or len(epsilon_domain) != 2
        or epsilon_domain[0] != ENERGY_DENSITY_SURFACE_MEV_FM3
        or epsilon_domain[1] != ENERGY_DENSITY_MAX_MEV_FM3
    ):
        raise RuntimeError("frozen CFL energy-density domain is inconsistent")
    if surface.get("pressure_mev_fm3") != 0.0:
        raise RuntimeError("frozen CFL self-bound surface pressure must be exactly zero")
    # Fail closed if any record member cannot be represented as strict JSON.
    canonical_json(record)
    return {
        "parameter_set": record,
        "parameter_set_id": FROZEN_PARAMETER_SET_ID,
        "parameter_set_sha256": FROZEN_PARAMETER_SET_SHA256,
        "formulation_id": CFL_FORMULATION_ID,
        "formulation_version": CFL_FORMULATION_VERSION,
        "deformation_profile_id": CFL_DEFORMATION_PROFILE_ID,
        "deformation_profile_version": CFL_DEFORMATION_PROFILE_VERSION,
        "reconstruction_profile_id": CFL_RECONSTRUCTION_PROFILE_ID,
        "reconstruction_schema_version": CFL_WINDOWED_EOS_SCHEMA_VERSION,
        "pressure_primitive_policy": CFL_PRESSURE_PRIMITIVE_POLICY,
        "stellar_sequence_policy": CFLAnalyticEos.stellar_sequence_policy,
        "stellar_local_refinement_policy": CFLAnalyticEos.stellar_local_refinement_policy,
        "domain_id": CFL_DOMAIN_ID,
        "epsilon_surface_mev_fm3": ENERGY_DENSITY_SURFACE_MEV_FM3,
        "epsilon_max_mev_fm3": ENERGY_DENSITY_MAX_MEV_FM3,
    }


def deterministic_cfl_case_id(
    *,
    amplitude: float,
    delta_mev_fm3: float,
    epsilon0_mev_fm3: float,
    sigma_mev_fm3: float,
) -> str:
    """Return the stable logical identity of one CFL deformation proposal."""

    contract = _frozen_baseline_contract()
    values = {
        "amplitude": _finite_number("amplitude", amplitude),
        "delta_mev_fm3": _positive_number("delta_mev_fm3", delta_mev_fm3),
        "epsilon0_mev_fm3": _positive_number(
            "epsilon0_mev_fm3", epsilon0_mev_fm3
        ),
        "sigma_mev_fm3": _positive_number("sigma_mev_fm3", sigma_mev_fm3),
    }
    values = {name: (0.0 if value == 0.0 else value) for name, value in values.items()}
    payload = {
        "matter_model": "cfl",
        "baseline_parameter_set_sha256": contract["parameter_set_sha256"],
        "deformation_profile_id": contract["deformation_profile_id"],
        "domain_id": contract["domain_id"],
        "epsilon_match_mev_fm3": float(
            contract["epsilon_surface_mev_fm3"]
        ).hex(),
        **{name: value.hex() for name, value in values.items()},
    }
    suffix = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return (
        f"cfl_d{_safe_identifier(values['delta_mev_fm3'])}"
        f"_a{_safe_identifier(values['amplitude'])}_{suffix}"
    )


def frozen_cfl_physical_baseline_id() -> str:
    """Return the one physical identity shared by all logical A=0 controls."""

    digest = str(_frozen_baseline_contract()["parameter_set_sha256"])
    return f"cfl_baseline_{digest[:16]}"


@dataclass(frozen=True)
class CFLTrialConfig:
    """Complete governed passive configuration for one CFL geometry."""

    amplitudes: tuple[float, ...]
    epsilon0_mev_fm3: float
    sigma_mev_fm3: float
    deltas_mev_fm3: tuple[float, ...]
    zero_amplitude_control_owner: bool = True
    fixed_masses_msun: tuple[float, ...] = (1.4,)
    thermodynamic_stages: tuple[BSk24ThermodynamicStage, ...] = field(
        default_factory=tuple
    )
    tov_stages: tuple[BSk24TOVStage, ...] = field(default_factory=tuple)
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

    def __post_init__(self) -> None:
        contract = _frozen_baseline_contract()
        epsilon_surface = float(contract["epsilon_surface_mev_fm3"])
        epsilon_max = float(contract["epsilon_max_mev_fm3"])
        amplitudes = _unique_floats("amplitudes", self.amplitudes)
        deltas = _unique_floats(
            "deltas_mev_fm3", self.deltas_mev_fm3, positive=True
        )
        masses = _unique_floats(
            "fixed_masses_msun", self.fixed_masses_msun, positive=True
        )
        if any(mass >= 10.0 for mass in masses):
            raise ValueError("fixed_masses_msun must be below 10 solar masses")
        if not isinstance(self.zero_amplitude_control_owner, bool):
            raise ValueError("zero_amplitude_control_owner must be boolean")
        epsilon0 = _positive_number("epsilon0_mev_fm3", self.epsilon0_mev_fm3)
        sigma = _positive_number("sigma_mev_fm3", self.sigma_mev_fm3)
        if not epsilon_surface < epsilon0 < epsilon_max:
            raise ValueError(
                "CFL epsilon0_mev_fm3 must lie strictly inside the complete "
                f"baseline domain ({epsilon_surface:.12g}, {epsilon_max:.12g})"
            )
        domain_span = epsilon_max - epsilon_surface
        if not epsilon0 - sigma < epsilon0 < epsilon0 + sigma:
            raise ValueError(
                "CFL sigma_mev_fm3 must move epsilon0 in both directions "
                "in binary64"
            )
        if any(epsilon_surface + delta <= epsilon_surface for delta in deltas):
            raise ValueError(
                "each CFL ramp width must produce a representably distinct "
                "binary64 endpoint above the self-bound surface"
            )
        if any(delta > domain_span for delta in deltas):
            raise ValueError(
                "each CFL ramp width must end at or below the complete "
                f"baseline endpoint; require delta <= {domain_span:.12g} MeV fm^-3"
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
        for name in (
            "stellar_enabled",
            "extended_stellar_diagnostics_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.extended_stellar_diagnostics_enabled:
            raise ValueError(
                "CFL extended diagnostics are unavailable because their bare "
                "self-bound radial, baryonic, and support semantics have not "
                "been established"
            )
        if (
            self.extended_stellar_diagnostics_case_policy
            not in EXTENDED_STELLAR_DIAGNOSTICS_CASE_POLICIES
        ):
            raise ValueError(
                "extended_stellar_diagnostics_case_policy must be one of "
                f"{EXTENDED_STELLAR_DIAGNOSTICS_CASE_POLICIES}"
            )
        for name in ("raw_gate_lower_points", "raw_gate_upper_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 17:
                raise ValueError(f"{name} must be an integer of at least 17")
        thermo = tuple(self.thermodynamic_stages)
        tov = tuple(self.tov_stages)
        if not thermo:
            raise ValueError("at least one thermodynamic stage is required")
        if any(not isinstance(item, BSk24ThermodynamicStage) for item in thermo):
            raise TypeError(
                "thermodynamic_stages must contain governed thermodynamic stage values"
            )
        if any(not isinstance(item, BSk24TOVStage) for item in tov):
            raise TypeError("tov_stages must contain governed TOV stage values")
        if self.stellar_enabled and not tov:
            raise ValueError("background TOV screening requires at least one TOV stage")
        for label, stages in (("thermodynamic", thermo), ("TOV", tov)):
            names = [stage.name for stage in stages]
            if len(names) != len(set(names)):
                raise ValueError(f"{label} stage names must be unique")
        groups = tuple(dict.fromkeys(str(value) for value in self.requested_plot_groups))
        invalid_groups = sorted(set(groups) - set(PLOT_GROUPS))
        if not groups or invalid_groups:
            raise ValueError(
                f"requested_plot_groups must use {PLOT_GROUPS}; invalid={invalid_groups}"
            )
        if "all-applicable" in groups and len(groups) > 1:
            groups = ("all-applicable",)
        if self.resume_policy not in RESUME_POLICIES:
            raise ValueError(f"resume_policy must be one of {RESUME_POLICIES}")
        if self.output_packet_name is not None:
            name = self.output_packet_name
            name_path = Path(name)
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
        resolved_output_path = (
            None
            if self.output_path is None
            else resolve_runs_path(self.output_path)
        )
        object.__setattr__(self, "amplitudes", amplitudes)
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
    def matter_model(self) -> str:
        return "cfl"

    @property
    def a0_was_injected(self) -> bool:
        return self.zero_amplitude_control_owner and self.logical_a0_was_injected

    @property
    def logical_a0_was_injected(self) -> bool:
        return not any(value == 0.0 for value in self.amplitudes)

    @property
    def logical_amplitudes(self) -> tuple[float, ...]:
        return (
            0.0,
            *(value for value in self.amplitudes if value != 0.0),
        )

    @property
    def effective_amplitudes(self) -> tuple[float, ...]:
        if not self.zero_amplitude_control_owner:
            return self.logical_amplitudes[1:]
        return self.logical_amplitudes

    @property
    def epsilon_match_mev_fm3(self) -> float:
        return float(_frozen_baseline_contract()["epsilon_surface_mev_fm3"])

    @property
    def effective_epsilon_match_mev_fm3(self) -> float:
        return self.epsilon_match_mev_fm3

    @property
    def exploratory_anchor_requested(self) -> bool:
        return False

    @property
    def complete_domain_mev_fm3(self) -> tuple[float, float]:
        contract = _frozen_baseline_contract()
        return (
            float(contract["epsilon_surface_mev_fm3"]),
            float(contract["epsilon_max_mev_fm3"]),
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

    def _serialized(self) -> dict[str, Any]:
        contract = _frozen_baseline_contract()
        data = asdict(self)
        data.update(
            {
                "matter_model": "cfl",
                "epsilon_match": "surface",
                "epsilon_match_mev_fm3": self.epsilon_match_mev_fm3,
                "complete_domain_mev_fm3": list(self.complete_domain_mev_fm3),
                "baseline_parameter_set_id": contract["parameter_set_id"],
                "baseline_parameter_set_sha256": contract[
                    "parameter_set_sha256"
                ],
                "baseline_profile": contract["parameter_set"],
                "formulation_id": contract["formulation_id"],
                "formulation_version": contract["formulation_version"],
                "deformation_profile_id": contract["deformation_profile_id"],
                "deformation_profile_version": contract[
                    "deformation_profile_version"
                ],
                "reconstruction_profile_id": contract[
                    "reconstruction_profile_id"
                ],
                "reconstruction_schema_version": contract[
                    "reconstruction_schema_version"
                ],
                "pressure_primitive_policy": contract[
                    "pressure_primitive_policy"
                ],
                "stellar_sequence_policy": contract["stellar_sequence_policy"],
                "stellar_local_refinement_policy": contract["stellar_local_refinement_policy"],
                "domain_id": contract["domain_id"],
                "effective_amplitudes": list(self.effective_amplitudes),
                "logical_amplitudes": list(self.logical_amplitudes),
                "a0_identity_control_injected": self.a0_was_injected,
                "logical_a0_identity_control_injected": (
                    self.logical_a0_was_injected
                ),
                "zero_amplitude_physical_case_id": (
                    frozen_cfl_physical_baseline_id()
                ),
            }
        )
        return json_clean(data)

    def to_dict(self) -> dict[str, Any]:
        return self._serialized()

    def expanded_dict(self) -> dict[str, Any]:
        return self._serialized()

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def deterministic_hash(self) -> str:
        definition = dict(self.to_dict())
        for name in ("output_packet_name", "output_path", "resume_policy"):
            definition.pop(name, None)
        return hashlib.sha256(canonical_json(definition).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CFLTrialConfig":
        values = dict(data)
        supplied = dict(values)
        if "$schema" in values and not isinstance(values.pop("$schema"), str):
            raise ValueError("top-level $schema must be a string")
        contract = _frozen_baseline_contract()
        expected_derived: dict[str, Any] = {
            "matter_model": "cfl",
            "epsilon_match": "surface",
            "epsilon_match_mev_fm3": contract["epsilon_surface_mev_fm3"],
            "complete_domain_mev_fm3": [
                contract["epsilon_surface_mev_fm3"],
                contract["epsilon_max_mev_fm3"],
            ],
            "baseline_parameter_set_id": contract["parameter_set_id"],
            "baseline_parameter_set_sha256": contract[
                "parameter_set_sha256"
            ],
            "baseline_profile": contract["parameter_set"],
            "formulation_id": contract["formulation_id"],
            "formulation_version": contract["formulation_version"],
            "deformation_profile_id": contract["deformation_profile_id"],
            "deformation_profile_version": contract[
                "deformation_profile_version"
            ],
            "reconstruction_profile_id": contract[
                "reconstruction_profile_id"
            ],
            "reconstruction_schema_version": contract[
                "reconstruction_schema_version"
            ],
            "pressure_primitive_policy": contract[
                "pressure_primitive_policy"
            ],
            "stellar_sequence_policy": contract["stellar_sequence_policy"],
            "stellar_local_refinement_policy": contract["stellar_local_refinement_policy"],
            "domain_id": contract["domain_id"],
            "zero_amplitude_physical_case_id": (
                frozen_cfl_physical_baseline_id()
            ),
        }
        for name, expected in expected_derived.items():
            if name in values and values[name] != expected:
                raise ValueError(
                    f"saved CFL configuration {name} disagrees with the frozen contract"
                )
        for name in (
            "matter_model",
            "epsilon_match",
            "epsilon_match_mev_fm3",
            "complete_domain_mev_fm3",
            "baseline_parameter_set_id",
            "baseline_parameter_set_sha256",
            "baseline_profile",
            "formulation_id",
            "formulation_version",
            "deformation_profile_id",
            "deformation_profile_version",
            "reconstruction_profile_id",
            "reconstruction_schema_version",
            "pressure_primitive_policy",
            "stellar_sequence_policy",
            "stellar_local_refinement_policy",
            "domain_id",
            "effective_amplitudes",
            "logical_amplitudes",
            "a0_identity_control_injected",
            "logical_a0_identity_control_injected",
            "zero_amplitude_physical_case_id",
        ):
            values.pop(name, None)
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
        for name in (
            "amplitudes",
            "deltas_mev_fm3",
            "fixed_masses_msun",
            "requested_plot_groups",
        ):
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        result = cls(**values)
        canonical = result.to_dict()
        for name in (
            "effective_amplitudes",
            "logical_amplitudes",
            "a0_identity_control_injected",
            "logical_a0_identity_control_injected",
        ):
            if name in supplied and supplied[name] != canonical[name]:
                raise ValueError(
                    f"saved CFL configuration {name} disagrees with its inputs"
                )
        return result


@dataclass(frozen=True)
class CFLTrialPlan:
    """Read-only, calculation-free preview of one CFL geometry."""

    config: CFLTrialConfig
    case_table: pd.DataFrame
    logical_alias_table: pd.DataFrame
    a0_identity_control_injected: bool
    anticipated_plot_inventory: pd.DataFrame
    estimates: Mapping[str, int]
    output_path: Path

    @property
    def logical_case_table(self) -> pd.DataFrame:
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
        return {
            "schema_id": CFL_TRIAL_PLAN_SCHEMA,
            "matter_model": "cfl",
            "configuration_hash": self.config.deterministic_hash(),
            "expanded_configuration": expanded,
            "operational_destination": operational,
            "a0_identity_control_injected": self.a0_identity_control_injected,
            "case_table": json_clean(self.case_table.to_dict(orient="records")),
            "logical_alias_table": json_clean(
                self.logical_alias_table.to_dict(orient="records")
            ),
            "anticipated_plot_inventory": json_clean(
                self.anticipated_plot_inventory.to_dict(orient="records")
            ),
            "estimates": dict(self.estimates),
            "output_path": json_clean(self.output_path),
            "planning_is_passive": True,
            "scientific_solver_calls": 0,
            "filesystem_writes": 0,
        }

    def _repr_html_(self) -> str:
        return (
            "<h3>CFL trial plan</h3>"
            + self.logical_case_table.to_html(index=False)
            + "<h4>Anticipated figure applicability</h4>"
            + self.anticipated_plot_inventory.to_html(index=False)
        )


def _output_path(config: CFLTrialConfig) -> Path:
    if config.output_path is not None:
        return _assert_writable_packet_path(resolve_runs_path(config.output_path))
    name = config.output_packet_name
    if name is None:
        name = f"{CFL_DEFAULT_OUTPUT_STEM}_{config.deterministic_hash()[:12]}"
    return _assert_writable_packet_path(runs_root() / name)


def _case_row(
    config: CFLTrialConfig,
    *,
    amplitude: float,
    delta: float,
    delta_index: int,
    amplitude_index: int,
    planned_for_execution: bool,
) -> dict[str, Any]:
    contract = _frozen_baseline_contract()
    case_id = deterministic_cfl_case_id(
        amplitude=amplitude,
        delta_mev_fm3=delta,
        epsilon0_mev_fm3=config.epsilon0_mev_fm3,
        sigma_mev_fm3=config.sigma_mev_fm3,
    )
    physical_case_id = (
        frozen_cfl_physical_baseline_id() if amplitude == 0.0 else case_id
    )
    return {
        "case_id": case_id,
        "physical_case_id": physical_case_id,
        "matter_model": "cfl",
        "baseline_parameter_set_id": contract["parameter_set_id"],
        "baseline_parameter_set_sha256": contract["parameter_set_sha256"],
        "deformation_profile_id": contract["deformation_profile_id"],
        "domain_id": contract["domain_id"],
        "amplitude": amplitude,
        "epsilon_match_mev_fm3": config.epsilon_match_mev_fm3,
        "anchor_mode": "self_bound_surface",
        "epsilon0_mev_fm3": config.epsilon0_mev_fm3,
        "sigma_mev_fm3": config.sigma_mev_fm3,
        "delta_mev_fm3": delta,
        "logical_delta_index": delta_index,
        "logical_amplitude_index": amplitude_index,
        "is_a0_identity_control": amplitude == 0.0,
        "identity_control_injected": (
            amplitude == 0.0 and config.logical_a0_was_injected
        ),
        "is_physical_case_alias": physical_case_id != case_id,
        "planned_for_execution": planned_for_execution,
        "physical_case_owner": planned_for_execution,
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


def _case_rows(config: CFLTrialConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    owner_delta = min(config.deltas_mev_fm3)
    for delta_index, delta in enumerate(config.deltas_mev_fm3):
        for amplitude_index, amplitude in enumerate(config.logical_amplitudes):
            if amplitude == 0.0 and (
                not config.zero_amplitude_control_owner or delta != owner_delta
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
    logical_ids = [row["case_id"] for row in rows]
    if len(logical_ids) != len(set(logical_ids)):
        raise RuntimeError("deterministic CFL execution case-ID collision")
    return rows


def _logical_alias_rows(config: CFLTrialConfig) -> list[dict[str, Any]]:
    aliases: list[dict[str, Any]] = []
    owner_delta = min(config.deltas_mev_fm3)
    for delta_index, delta in enumerate(config.deltas_mev_fm3):
        if config.zero_amplitude_control_owner and delta == owner_delta:
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


def prepare_cfl_trial(config: CFLTrialConfig) -> CFLTrialPlan:
    """Validate and preview one CFL geometry without calculations or writes."""

    if not isinstance(config, CFLTrialConfig):
        raise TypeError("config must be a CFLTrialConfig")
    cases = pd.DataFrame(_case_rows(config))
    aliases = pd.DataFrame(_logical_alias_rows(config))
    if cases.empty and not len(cases.columns) and not aliases.empty:
        cases = aliases.iloc[0:0].copy()
    physical_cases = (
        int(cases["physical_case_id"].nunique()) if not cases.empty else 0
    )
    # Runtime solves A=0 once as "direct" and retains the physical zero case
    # as its explicit alias; it does not solve both independently.
    stellar_cases = physical_cases if config.background_tov_requested else 0
    estimates = {
        "proposed_deformation_cases": int(len(cases)),
        "logical_deformation_cases": int(len(cases) + len(aliases)),
        "physical_deformation_cases": physical_cases,
        "deduplicated_logical_case_aliases": int(len(aliases)),
        "direct_baseline_cases": int(config.zero_amplitude_control_owner),
        "baseline_construction_stage_evaluations": len(config.thermodynamic_stages),
        "thermodynamic_case_stage_evaluations": int(
            physical_cases * len(config.thermodynamic_stages)
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
    if config.stellar_enabled:
        estimates.update(
            {
                "sampled_sequence_tidal_targets": int(
                    stellar_cases
                    * sum(stage.sequence_points for stage in config.tov_stages)
                ),
                "fixed_mass_tidal_targets": int(
                    stellar_cases
                    * len(config.tov_stages)
                    * len(config.fixed_masses_msun)
                ),
                "maximum_mass_local_screen_targets": int(
                    stellar_cases
                    * len(config.tov_stages)
                    * config.maximum_mass_initial_points
                ),
            }
        )
    return CFLTrialPlan(
        config=config,
        case_table=cases,
        logical_alias_table=aliases,
        a0_identity_control_injected=config.a0_was_injected,
        anticipated_plot_inventory=_anticipated_plot_inventory(config),
        estimates=estimates,
        output_path=_output_path(config),
    )


__all__ = [
    "CFL_DEFAULT_OUTPUT_STEM",
    "CFL_TRIAL_PLAN_SCHEMA",
    "CFLTrialConfig",
    "CFLTrialPlan",
    "deterministic_cfl_case_id",
    "frozen_cfl_physical_baseline_id",
    "prepare_cfl_trial",
]

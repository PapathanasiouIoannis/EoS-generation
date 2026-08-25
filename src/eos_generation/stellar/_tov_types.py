"""Immutable constants and evidence/result types for the TOV workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from eos_generation._internal.config import DEFAULT_CONFIG
from eos_generation.stellar.discontinuities import (
    EOS_DISCONTINUITY_CONTRACT_VERSION,
)

DENOMINATOR_FLOOR = 1.0e-25
TOV_SEQUENCE_EVIDENCE_VERSION = "tov_sequence_evidence_v2"
TIDAL_CORRECTION_VERSION = "hinderer_takatsy_postnikov_v1"
LAMBDA_FRAMEWORK_CAPABILITY = "validated_lambda_validation_v1"
LAMBDA_SCIENTIFIC_STATUS = LAMBDA_FRAMEWORK_CAPABILITY
TIDAL_CORRECTION_STATUS = "validated_conditional_per_calculation"
TIDAL_NOT_REQUESTED_STATUS = "not_requested_background_only"
TIDAL_JUMP_FORMULA = (
    "delta_y=-G_CONV*r^3*(epsilon_inner-epsilon_outer)/"
    "(m+G_CONV*r^3*P)"
)
TIDAL_JUMP_SOURCES = (
    "Hinderer 2008 erratum, arXiv:0711.2420v4",
    "Postnikov-Prakash-Lattimer 2010, doi:10.1103/PhysRevD.82.024016",
    "Takatsy-Kovacs 2020, arXiv:2007.01139v3, Eq. 11",
)
TOV_SEQUENCE_FIELDS = (
    "Mass",
    "Radius",
    "Lambda",
    "P_Central",
    "Eps_Central",
    "CS2_Central",
    "eps_surf",
)
TOV_TIDAL_DIAGNOSTIC_FIELDS = (
    "Mass",
    "Radius",
    "P_Central",
    "Eps_Central",
    "CS2_Central",
    "Compactness",
    "y_R",
    "k2",
    "Lambda",
    "eps_surf",
)
_MAXIMUM_AUTOMATIC_SEQUENCE_WORKERS = 4
_OUTER_PARALLEL_WORKER_ENV = "BSK24_NOTEBOOK_OUTER_WORKER"

_DEFAULT_TOV = DEFAULT_CONFIG.tov
_R_MIN = _DEFAULT_TOV.radius_min_km
_R_MAX = _DEFAULT_TOV.radius_max_km
_P_MIN_SAFE = _DEFAULT_TOV.pressure_min_safe
_G_CONV = DEFAULT_CONFIG.units.gravity_conversion
_A_CONV = DEFAULT_CONFIG.units.solar_mass_length_km
_BUCHDAHL_LIMIT = DEFAULT_CONFIG.filters.buchdahl_limit
_MIN_RADIUS_CUTOFF = DEFAULT_CONFIG.filters.minimum_radius_cutoff_km
_MIN_MASS_CUTOFF = DEFAULT_CONFIG.filters.minimum_mass_cutoff
_ABSOLUTE_P_MAX_FALLBACK = DEFAULT_CONFIG.thermodynamics.absolute_pressure_max_fallback
_TOV_SINGULARITY_LIMIT = _DEFAULT_TOV.singularity_limit
_TIDAL_SEGMENT_BOUNDARY_ULPS = 8.0


class TovConvergenceError(Exception):
    """Raised internally when one TOV integration encounters a domain error."""

    def __init__(self, pc, reason, message=None):
        self.pc = pc
        self.reason = reason
        self.message = message or f"TOV solver failed to converge at central pressure {pc}. Reason: {reason}"
        super().__init__(self.message)


def _evidence_finite_float(name: str, value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _freeze_rows(
    rows: Any,
    *,
    fields: tuple[str, ...],
    name: str,
    allow_nan_fields: tuple[str, ...] = (),
) -> tuple[tuple[float, ...], ...]:
    frozen_rows = []
    for row_index, row in enumerate(rows):
        values = tuple(row)
        if len(values) != len(fields):
            raise ValueError(f"{name} row {row_index} must contain {len(fields)} values")
        frozen_values = []
        for field, value in zip(fields, values):
            resolved = float(value)
            if field in allow_nan_fields and math.isnan(resolved):
                frozen_values.append(resolved)
            else:
                frozen_values.append(
                    _evidence_finite_float(f"{name}[{row_index}].{field}", resolved)
                )
        frozen_rows.append(tuple(frozen_values))
    return tuple(frozen_rows)


def _freeze_profiles(
    profiles: Any,
    *,
    name: str,
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    frozen_profiles = []
    for profile_index, profile in enumerate(profiles):
        values = tuple(profile)
        if len(values) != 2:
            raise ValueError(f"{name} profile {profile_index} must contain radius and mass arrays")
        radius = tuple(
            _evidence_finite_float(f"{name}[{profile_index}].radius", value)
            for value in values[0]
        )
        mass = tuple(
            _evidence_finite_float(f"{name}[{profile_index}].mass", value)
            for value in values[1]
        )
        if len(radius) != len(mass):
            raise ValueError(f"{name} profile {profile_index} radius and mass lengths differ")
        frozen_profiles.append((radius, mass))
    return tuple(frozen_profiles)


def _rows_to_dicts(
    rows: tuple[tuple[float, ...], ...],
    fields: tuple[str, ...],
) -> list[dict[str, float | None]]:
    return [
        {
            field: (None if math.isnan(value) else value)
            for field, value in zip(fields, row)
        }
        for row in rows
    ]


def _rows_equal(left: Any, right: Any) -> bool:
    left_rows = tuple(left)
    right_rows = tuple(right)
    if len(left_rows) != len(right_rows):
        return False
    return all(
        np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True)
        for a, b in zip(left_rows, right_rows)
    )


@dataclass(frozen=True)
class TovFailureDetail:
    """One central-pressure sample skipped by the existing solver policy."""

    central_pressure: float
    category: str
    reason: str
    solver_status: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "central_pressure",
            _evidence_finite_float("central_pressure", self.central_pressure),
        )
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("failure category must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("failure reason must be a non-empty string")
        if self.solver_status is not None:
            if isinstance(self.solver_status, bool):
                raise ValueError("solver_status must be an integer or None")
            object.__setattr__(self, "solver_status", int(self.solver_status))

    def to_dict(self) -> dict[str, Any]:
        return {
            "central_pressure": self.central_pressure,
            "category": self.category,
            "reason": self.reason,
            "solver_status": self.solver_status,
        }


def _optional_finite(name: str, value: Any | None) -> float | None:
    if value is None:
        return None
    return _evidence_finite_float(name, value)


@dataclass(frozen=True)
class AppliedTidalJump:
    """One analytically applied outward matching condition."""

    identifier: str
    kind: str
    pressure: float
    radius: float
    mass: float
    inner_energy_density: float
    outer_energy_density: float
    delta_energy_density: float
    correction_denominator: float
    y_before: float
    delta_y: float
    y_after: float
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str) or not self.identifier:
            raise ValueError("applied jump identifier must be non-empty")
        if self.kind not in ("internal", "surface"):
            raise ValueError("applied jump kind must be internal or surface")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("applied jump provenance must be non-empty")
        for name in (
            "pressure",
            "radius",
            "mass",
            "inner_energy_density",
            "outer_energy_density",
            "delta_energy_density",
            "correction_denominator",
            "y_before",
            "delta_y",
            "y_after",
        ):
            object.__setattr__(self, name, _evidence_finite_float(name, getattr(self, name)))
        if self.radius <= 0.0 or self.mass <= 0.0:
            raise ValueError("applied jump radius and mass must be positive")
        if self.correction_denominator <= 0.0:
            raise ValueError("applied jump correction denominator must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "type": self.kind,
            "pressure_MeV_fm3": self.pressure,
            "radius_km": self.radius,
            "mass_Msun": self.mass,
            "inner_energy_density_MeV_fm3": self.inner_energy_density,
            "outer_energy_density_MeV_fm3": self.outer_energy_density,
            "signed_outward_delta_energy_density_MeV_fm3": self.delta_energy_density,
            "correction_denominator_Msun": self.correction_denominator,
            "y_before": self.y_before,
            "delta_y": self.delta_y,
            "y_after": self.y_after,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class TovLambdaDiagnostic:
    """Per-star tidal matching evidence, including fail-closed outcomes."""

    central_pressure: float
    mass: float
    radius: float
    compactness: float
    expected_jump_count: int
    applied_jumps: tuple[AppliedTidalJump, ...]
    skipped_discontinuity_ids: tuple[str, ...]
    surface_event_pressure: float
    y_surface_interior: float | None
    y_surface_vacuum: float | None
    y_supplied_to_k2: float | None
    k2: float | None
    lambda_dimensionless: float | None
    scientific_status: str
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("central_pressure", "mass", "radius", "compactness", "surface_event_pressure"):
            object.__setattr__(self, name, _evidence_finite_float(name, getattr(self, name)))
        if self.mass <= 0.0 or self.radius <= 0.0 or self.compactness <= 0.0:
            raise ValueError("tidal diagnostic background values must be positive")
        if isinstance(self.expected_jump_count, bool) or int(self.expected_jump_count) < 0:
            raise ValueError("expected_jump_count must be a nonnegative integer")
        object.__setattr__(self, "expected_jump_count", int(self.expected_jump_count))
        jumps = tuple(self.applied_jumps)
        if any(not isinstance(item, AppliedTidalJump) for item in jumps):
            raise TypeError("applied_jumps must contain AppliedTidalJump values")
        object.__setattr__(self, "applied_jumps", jumps)
        skipped = tuple(str(value) for value in self.skipped_discontinuity_ids)
        object.__setattr__(self, "skipped_discontinuity_ids", skipped)
        for name in (
            "y_surface_interior",
            "y_surface_vacuum",
            "y_supplied_to_k2",
            "k2",
            "lambda_dimensionless",
        ):
            object.__setattr__(self, name, _optional_finite(name, getattr(self, name)))
        if not isinstance(self.scientific_status, str) or not self.scientific_status:
            raise ValueError("scientific_status must be non-empty")
        if self.failure_reason is not None and (
            not isinstance(self.failure_reason, str) or not self.failure_reason
        ):
            raise ValueError("failure_reason must be None or a non-empty string")
        if self.scientific_status == LAMBDA_FRAMEWORK_CAPABILITY:
            if len(jumps) != self.expected_jump_count:
                raise ValueError("completed tidal diagnostics require every expected jump")
            if any(
                value is None
                for value in (self.y_surface_interior, self.y_supplied_to_k2, self.k2, self.lambda_dimensionless)
            ):
                raise ValueError("completed tidal diagnostics require finite tidal outputs")
            if self.failure_reason is not None:
                raise ValueError("completed tidal diagnostics cannot contain a failure reason")
        elif self.scientific_status not in {
            "failed_closed",
            TIDAL_NOT_REQUESTED_STATUS,
        }:
            raise ValueError(
                "scientific_status must be validated_lambda_validation_v1, "
                "failed_closed, or not_requested_background_only"
            )
        elif self.scientific_status == TIDAL_NOT_REQUESTED_STATUS:
            if any(
                value is not None
                for value in (
                    self.y_surface_interior,
                    self.y_surface_vacuum,
                    self.y_supplied_to_k2,
                    self.k2,
                    self.lambda_dimensionless,
                    self.failure_reason,
                )
            ) or jumps:
                raise ValueError(
                    "background-only tidal diagnostics cannot contain tidal results"
                )

    @property
    def applied_jump_count(self) -> int:
        return len(self.applied_jumps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "tov_lambda_diagnostic_v1",
            "central_pressure_MeV_fm3": self.central_pressure,
            "Mass": self.mass,
            "Radius": self.radius,
            "Compactness": self.compactness,
            "expected_jump_count": self.expected_jump_count,
            "applied_jump_count": self.applied_jump_count,
            "applied_jumps": [item.to_dict() for item in self.applied_jumps],
            "skipped_discontinuity_ids": list(self.skipped_discontinuity_ids),
            "surface_event_pressure_MeV_fm3": self.surface_event_pressure,
            "y_surface_interior": self.y_surface_interior,
            "y_surface_vacuum": self.y_surface_vacuum,
            "y_supplied_to_k2": self.y_supplied_to_k2,
            "y_R": self.y_supplied_to_k2,
            "k2": self.k2,
            "Lambda": self.lambda_dimensionless,
            "correction_formula": TIDAL_JUMP_FORMULA,
            "correction_version": TIDAL_CORRECTION_VERSION,
            "correction_status": TIDAL_CORRECTION_STATUS,
            "correction_sources": list(TIDAL_JUMP_SOURCES),
            "discontinuity_contract_version": EOS_DISCONTINUITY_CONTRACT_VERSION,
            "framework_lambda_capability": LAMBDA_FRAMEWORK_CAPABILITY,
            "calculation_lambda_validated": self.scientific_status == LAMBDA_FRAMEWORK_CAPABILITY,
            "scientific_status": self.scientific_status,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class TovMassSecantEvidence:
    """Raw sampled mass slope; it is not convergence-resolved evidence."""

    lower_index: int
    upper_index: int
    lower_central_pressure: float
    upper_central_pressure: float
    lower_mass: float
    upper_mass: float
    delta_mass: float
    slope: float
    sign: str

    def __post_init__(self) -> None:
        if self.lower_index < 0 or self.upper_index != self.lower_index + 1:
            raise ValueError("TOV mass-secant indices must be adjacent and nonnegative")
        for name in (
            "lower_central_pressure",
            "upper_central_pressure",
            "lower_mass",
            "upper_mass",
            "delta_mass",
            "slope",
        ):
            object.__setattr__(self, name, _evidence_finite_float(name, getattr(self, name)))
        if self.upper_central_pressure <= self.lower_central_pressure:
            raise ValueError("TOV mass-secant central pressures must be strictly increasing")
        if self.sign not in ("positive", "negative", "zero"):
            raise ValueError("TOV mass-secant sign must be positive, negative, or zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower_index": self.lower_index,
            "upper_index": self.upper_index,
            "lower_central_pressure": self.lower_central_pressure,
            "upper_central_pressure": self.upper_central_pressure,
            "lower_mass": self.lower_mass,
            "upper_mass": self.upper_mass,
            "delta_mass": self.delta_mass,
            "slope": self.slope,
            "sign": self.sign,
            "interpretation": "raw_sampled_slope_not_convergence_resolved",
        }


@dataclass(frozen=True)
class TovSequenceEvidence:
    """Immutable full-sequence evidence plus the historical stable-prefix view."""

    full_sequence: tuple[tuple[float, ...], ...]
    stable_sequence: tuple[tuple[float, ...], ...]
    full_dense_profiles: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]
    stable_dense_profiles: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]
    full_tidal_diagnostics: tuple[tuple[float, ...], ...] | None
    stable_tidal_diagnostics: tuple[tuple[float, ...], ...] | None
    attempted_central_pressures: tuple[float, ...]
    successful_central_pressures: tuple[float, ...]
    central_pressure_ordering: str
    failed_central_pressures: tuple[TovFailureDetail, ...]
    sampled_peak_index: int | None
    sampled_peak_row: tuple[float, ...] | None
    domain_end_row: tuple[float, ...] | None
    sampled_peak_is_interior: bool
    pre_peak_slopes: tuple[TovMassSecantEvidence, ...]
    post_peak_slopes: tuple[TovMassSecantEvidence, ...]
    eos_endpoint_pressure: float | None
    eos_endpoint_margin: float | None
    final_available_model_contacts_eos_endpoint: bool | None
    max_mass_stable: float
    full_lambda_diagnostics: tuple[TovLambdaDiagnostic, ...] | None = None
    stable_lambda_diagnostics: tuple[TovLambdaDiagnostic, ...] | None = None

    schema_version: ClassVar[str] = TOV_SEQUENCE_EVIDENCE_VERSION
    scientific_scope: ClassVar[str] = (
        "sampled cold, nonrotating, one-parameter barotropic TOV sequence; "
        "sampled slopes and argmax are not radial-mode stability evidence"
    )
    lambda_caveat: ClassVar[str] = (
        "Declared Hadronic joins and bare self-bound surfaces use analytic matching "
        "conditions under validated_lambda_validation_v1, but each calculation remains "
        "valid only when its tidal solve and all required matching diagnostics succeed."
    )

    def __post_init__(self) -> None:
        full = _freeze_rows(
            self.full_sequence,
            fields=TOV_SEQUENCE_FIELDS,
            name="full_sequence",
            allow_nan_fields=("Lambda",),
        )
        stable = _freeze_rows(
            self.stable_sequence,
            fields=TOV_SEQUENCE_FIELDS,
            name="stable_sequence",
            allow_nan_fields=("Lambda",),
        )
        full_profiles = _freeze_profiles(self.full_dense_profiles, name="full_dense_profiles")
        stable_profiles = _freeze_profiles(
            self.stable_dense_profiles,
            name="stable_dense_profiles",
        )
        full_tidal = None
        stable_tidal = None
        if self.full_tidal_diagnostics is not None:
            full_tidal = _freeze_rows(
                self.full_tidal_diagnostics,
                fields=TOV_TIDAL_DIAGNOSTIC_FIELDS,
                name="full_tidal_diagnostics",
                allow_nan_fields=("y_R", "k2", "Lambda"),
            )
        if self.stable_tidal_diagnostics is not None:
            stable_tidal = _freeze_rows(
                self.stable_tidal_diagnostics,
                fields=TOV_TIDAL_DIAGNOSTIC_FIELDS,
                name="stable_tidal_diagnostics",
                allow_nan_fields=("y_R", "k2", "Lambda"),
            )
        full_lambda = None if self.full_lambda_diagnostics is None else tuple(self.full_lambda_diagnostics)
        stable_lambda = (
            None if self.stable_lambda_diagnostics is None else tuple(self.stable_lambda_diagnostics)
        )
        if full_lambda is not None and any(
            not isinstance(item, TovLambdaDiagnostic) for item in full_lambda
        ):
            raise TypeError("full_lambda_diagnostics must contain TovLambdaDiagnostic values")
        if stable_lambda is not None and any(
            not isinstance(item, TovLambdaDiagnostic) for item in stable_lambda
        ):
            raise TypeError("stable_lambda_diagnostics must contain TovLambdaDiagnostic values")
        attempted = tuple(
            _evidence_finite_float("attempted_central_pressure", value)
            for value in self.attempted_central_pressures
        )
        successful = tuple(
            _evidence_finite_float("successful_central_pressure", value)
            for value in self.successful_central_pressures
        )
        failures = tuple(self.failed_central_pressures)
        pre_slopes = tuple(self.pre_peak_slopes)
        post_slopes = tuple(self.post_peak_slopes)
        if any(not isinstance(item, TovFailureDetail) for item in failures):
            raise TypeError("failed_central_pressures must contain TovFailureDetail values")
        if any(not isinstance(item, TovMassSecantEvidence) for item in pre_slopes + post_slopes):
            raise TypeError("TOV slope evidence must contain TovMassSecantEvidence values")

        object.__setattr__(self, "full_sequence", full)
        object.__setattr__(self, "stable_sequence", stable)
        object.__setattr__(self, "full_dense_profiles", full_profiles)
        object.__setattr__(self, "stable_dense_profiles", stable_profiles)
        object.__setattr__(self, "full_tidal_diagnostics", full_tidal)
        object.__setattr__(self, "stable_tidal_diagnostics", stable_tidal)
        object.__setattr__(self, "full_lambda_diagnostics", full_lambda)
        object.__setattr__(self, "stable_lambda_diagnostics", stable_lambda)
        object.__setattr__(self, "attempted_central_pressures", attempted)
        object.__setattr__(self, "successful_central_pressures", successful)
        object.__setattr__(self, "failed_central_pressures", failures)
        object.__setattr__(self, "pre_peak_slopes", pre_slopes)
        object.__setattr__(self, "post_peak_slopes", post_slopes)
        object.__setattr__(
            self,
            "max_mass_stable",
            _evidence_finite_float("max_mass_stable", self.max_mass_stable),
        )

        if len(full) != len(full_profiles):
            raise ValueError("full TOV sequence and dense-profile counts differ")
        if len(stable) != len(stable_profiles):
            raise ValueError("stable TOV sequence and dense-profile counts differ")
        if full_tidal is not None and len(full_tidal) != len(full):
            raise ValueError("full tidal-diagnostic and sequence counts differ")
        if stable_tidal is not None and len(stable_tidal) != len(stable):
            raise ValueError("stable tidal-diagnostic and sequence counts differ")
        if (full_tidal is None) != (stable_tidal is None):
            raise ValueError("full and stable tidal diagnostics must be requested together")
        if (full_lambda is None) != (stable_lambda is None):
            raise ValueError("full and stable Lambda diagnostics must be requested together")
        if (full_tidal is None) != (full_lambda is None):
            raise ValueError("numeric and correction tidal diagnostics must be requested together")
        if full_lambda is not None and len(full_lambda) != len(full):
            raise ValueError("full Lambda-diagnostic and sequence counts differ")
        if stable_lambda is not None and len(stable_lambda) != len(stable):
            raise ValueError("stable Lambda-diagnostic and sequence counts differ")
        if not _rows_equal(stable, full[: len(stable)]):
            raise ValueError("stable TOV sequence must be a prefix of the full sequence")
        if stable_profiles != full_profiles[: len(stable_profiles)]:
            raise ValueError("stable dense profiles must be a prefix of full dense profiles")
        if full_tidal is not None and not _rows_equal(
            stable_tidal, full_tidal[: len(stable_tidal)]
        ):
            raise ValueError("stable tidal diagnostics must be a prefix of full diagnostics")
        if full_lambda is not None and stable_lambda != full_lambda[: len(stable_lambda)]:
            raise ValueError("stable Lambda diagnostics must be a prefix of full diagnostics")
        if successful != tuple(row[3] for row in full):
            raise ValueError("successful central pressures must match the full sequence")
        if len(attempted) != len(full) + len(failures):
            raise ValueError("each attempted central pressure must be successful or have failure detail")
        if len(attempted) > 1 and np.any(np.diff(attempted) <= 0.0):
            raise ValueError("attempted central pressures must be strictly increasing")
        expected_ordering = (
            "unavailable"
            if not successful
            else "single_sample"
            if len(successful) == 1
            else "strictly_increasing"
            if np.all(np.diff(successful) > 0.0)
            else "invalid"
        )
        if self.central_pressure_ordering != expected_ordering:
            raise ValueError("central_pressure_ordering does not match successful sequence")

        if not full:
            if any(
                value is not None
                for value in (self.sampled_peak_index, self.sampled_peak_row, self.domain_end_row)
            ):
                raise ValueError("empty TOV evidence cannot contain sampled peak or domain-end rows")
            if stable or pre_slopes or post_slopes or self.sampled_peak_is_interior:
                raise ValueError("empty TOV evidence cannot contain stable rows or slope evidence")
            if self.max_mass_stable != 0.0:
                raise ValueError("empty TOV evidence must report max_mass_stable=0.0")
        else:
            if self.sampled_peak_index is None:
                raise ValueError("nonempty TOV evidence requires a sampled peak index")
            peak_index = int(self.sampled_peak_index)
            object.__setattr__(self, "sampled_peak_index", peak_index)
            if peak_index < 0 or peak_index >= len(full):
                raise ValueError("sampled peak index is outside the full sequence")
            peak_row = _freeze_rows(
                (self.sampled_peak_row,),
                fields=TOV_SEQUENCE_FIELDS,
                name="sampled_peak_row",
                allow_nan_fields=("Lambda",),
            )[0]
            end_row = _freeze_rows(
                (self.domain_end_row,),
                fields=TOV_SEQUENCE_FIELDS,
                name="domain_end_row",
                allow_nan_fields=("Lambda",),
            )[0]
            object.__setattr__(self, "sampled_peak_row", peak_row)
            object.__setattr__(self, "domain_end_row", end_row)
            if not _rows_equal((peak_row,), (full[peak_index],)) or not _rows_equal(
                (end_row,), (full[-1],)
            ):
                raise ValueError("sampled peak or domain-end row does not match full sequence")
            if not _rows_equal(stable, full[: peak_index + 1]):
                raise ValueError("stable TOV view must end at the sampled argmax")
            if self.sampled_peak_is_interior != (0 < peak_index < len(full) - 1):
                raise ValueError("sampled_peak_is_interior does not match sampled peak index")
            if self.max_mass_stable != peak_row[0]:
                raise ValueError("max_mass_stable must equal the sampled peak mass")

        endpoint = self.eos_endpoint_pressure
        margin = self.eos_endpoint_margin
        contact = self.final_available_model_contacts_eos_endpoint
        if endpoint is None:
            if margin is not None or contact is not None:
                raise ValueError("endpoint margin/contact require an explicit EoS endpoint")
        else:
            endpoint = _evidence_finite_float("eos_endpoint_pressure", endpoint)
            object.__setattr__(self, "eos_endpoint_pressure", endpoint)
            if not full:
                if margin is not None or contact is not None:
                    raise ValueError("empty TOV evidence has no final model for endpoint comparison")
            else:
                margin = _evidence_finite_float("eos_endpoint_margin", margin)
                object.__setattr__(self, "eos_endpoint_margin", margin)
                expected_margin = endpoint - full[-1][3]
                if margin != expected_margin or contact != (expected_margin == 0.0):
                    raise ValueError("EoS endpoint margin/contact does not match the domain-end row")

    @property
    def successful_row_count(self) -> int:
        return len(self.full_sequence)

    @property
    def failed_central_pressure_count(self) -> int:
        return len(self.failed_central_pressures)

    @property
    def attempted_central_pressure_count(self) -> int:
        return len(self.attempted_central_pressures)

    @property
    def sampled_peak_values(self) -> dict[str, float | None] | None:
        if self.sampled_peak_row is None:
            return None
        return _rows_to_dicts((self.sampled_peak_row,), TOV_SEQUENCE_FIELDS)[0]

    @property
    def domain_end_values(self) -> dict[str, float | None] | None:
        if self.domain_end_row is None:
            return None
        return _rows_to_dicts((self.domain_end_row,), TOV_SEQUENCE_FIELDS)[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence_fields": list(TOV_SEQUENCE_FIELDS),
            "tidal_diagnostic_fields": list(TOV_TIDAL_DIAGNOSTIC_FIELDS),
            "full_sequence": _rows_to_dicts(self.full_sequence, TOV_SEQUENCE_FIELDS),
            "stable_sequence": _rows_to_dicts(self.stable_sequence, TOV_SEQUENCE_FIELDS),
            "full_dense_profiles": [
                {"radius_km": list(radius), "mass_solar": list(mass)}
                for radius, mass in self.full_dense_profiles
            ],
            "stable_dense_profiles": [
                {"radius_km": list(radius), "mass_solar": list(mass)}
                for radius, mass in self.stable_dense_profiles
            ],
            "full_tidal_diagnostics": None
            if self.full_tidal_diagnostics is None
            else _rows_to_dicts(self.full_tidal_diagnostics, TOV_TIDAL_DIAGNOSTIC_FIELDS),
            "stable_tidal_diagnostics": None
            if self.stable_tidal_diagnostics is None
            else _rows_to_dicts(self.stable_tidal_diagnostics, TOV_TIDAL_DIAGNOSTIC_FIELDS),
            "full_lambda_diagnostics": None
            if self.full_lambda_diagnostics is None
            else [item.to_dict() for item in self.full_lambda_diagnostics],
            "stable_lambda_diagnostics": None
            if self.stable_lambda_diagnostics is None
            else [item.to_dict() for item in self.stable_lambda_diagnostics],
            "attempted_central_pressures": list(self.attempted_central_pressures),
            "successful_central_pressures": list(self.successful_central_pressures),
            "central_pressure_ordering": self.central_pressure_ordering,
            "attempted_central_pressure_count": self.attempted_central_pressure_count,
            "successful_row_count": self.successful_row_count,
            "failed_central_pressure_count": self.failed_central_pressure_count,
            "failed_central_pressures": [item.to_dict() for item in self.failed_central_pressures],
            "sampled_peak_index": self.sampled_peak_index,
            "sampled_peak_values": self.sampled_peak_values,
            "domain_end_values": self.domain_end_values,
            "sampled_peak_is_interior": self.sampled_peak_is_interior,
            "pre_peak_slopes": [item.to_dict() for item in self.pre_peak_slopes],
            "post_peak_slopes": [item.to_dict() for item in self.post_peak_slopes],
            "eos_endpoint_pressure": self.eos_endpoint_pressure,
            "eos_endpoint_margin": self.eos_endpoint_margin,
            "final_available_model_contacts_eos_endpoint": (
                self.final_available_model_contacts_eos_endpoint
            ),
            "max_mass_stable": self.max_mass_stable,
            "scientific_scope": self.scientific_scope,
            "lambda_caveat": self.lambda_caveat,
        }


@dataclass(frozen=True)
class TidalAlgebraResult:
    """Compact result for one algebraic tidal calculation."""

    compactness: float
    y_r: float
    k2: float
    lambda_dimensionless: float


@dataclass(frozen=True)
class TovStarResult:
    """One background star plus an independently fallible tidal result."""

    central_pressure: float
    central_energy_density: float
    central_sound_speed_squared: float
    mass: float
    radius: float
    lambda_dimensionless: float | None
    surface_energy_density: float
    radius_profile: tuple[float, ...]
    mass_profile: tuple[float, ...]
    lambda_diagnostic: TovLambdaDiagnostic

    @property
    def curve_row(self) -> list[float]:
        return [
            self.mass,
            self.radius,
            (
                float(self.lambda_dimensionless)
                if self.lambda_dimensionless is not None
                else float("nan")
            ),
            self.central_pressure,
            self.central_energy_density,
            self.central_sound_speed_squared,
            self.surface_energy_density,
        ]


@dataclass(frozen=True)
class TovMaximumMassResult:
    """Resolved or explicitly unresolved nonrotating maximum-mass evidence."""

    status: str
    maximum_mass_resolved: bool
    maximum_mass_threshold_msun: float
    passes_maximum_mass_threshold: bool | None
    maximum_mass_msun: float | None
    central_pressure_mev_fm3: float | None
    central_energy_density_mev_fm3: float | None
    central_sound_speed_squared: float | None
    radius_km: float | None
    turning_point_brackets: tuple[tuple[float, ...], ...]
    selected_bracket: tuple[float, ...] | None
    stable_branch_models: tuple[tuple[float, ...], ...]
    sampled_models: tuple[tuple[float, ...], ...]
    positive_left_secant: float | None
    negative_right_secant: float | None
    eos_endpoint_pressure_mev_fm3: float
    endpoint_reached: bool
    endpoint_limitation: str | None
    refinement_status: str
    refinement_iterations: int
    global_refinement_rounds: int
    solver_call_count: int
    solver_failure_count: int
    solver_failures: tuple[tuple[float, str], ...]

    def to_dict(self) -> dict[str, Any]:
        def model(row: tuple[float, ...]) -> dict[str, float]:
            return {
                "central_pressure_mev_fm3": row[0],
                "mass_msun": row[1],
                "radius_km": row[2],
                "central_energy_density_mev_fm3": row[3],
                "central_sound_speed_squared": row[4],
            }

        def bracket(row: tuple[float, ...]) -> dict[str, float]:
            return {
                "lower_pressure_mev_fm3": row[0],
                "middle_pressure_mev_fm3": row[1],
                "upper_pressure_mev_fm3": row[2],
                "lower_mass_msun": row[3],
                "middle_mass_msun": row[4],
                "upper_mass_msun": row[5],
                "left_dM_dPc_secant": row[6],
                "right_dM_dPc_secant": row[7],
            }

        return {
            "schema_id": "tov_resolved_maximum_mass_v2",
            "status": self.status,
            "maximum_mass_resolved": self.maximum_mass_resolved,
            "decision_basis": (
                "refined_positive_to_negative_dM_dPc_turning_point"
                if self.maximum_mass_resolved
                else "fail_closed_no_resolved_turning_point"
            ),
            "sampled_argmax_is_maximum_mass": False,
            "maximum_mass_threshold_msun": self.maximum_mass_threshold_msun,
            "passes_maximum_mass_threshold": self.passes_maximum_mass_threshold,
            "maximum_mass_msun": self.maximum_mass_msun,
            "central_pressure_mev_fm3": self.central_pressure_mev_fm3,
            "central_energy_density_mev_fm3": self.central_energy_density_mev_fm3,
            "central_sound_speed_squared": self.central_sound_speed_squared,
            "radius_km": self.radius_km,
            "turning_point_count": len(self.turning_point_brackets),
            "turning_point_brackets": [
                bracket(row) for row in self.turning_point_brackets
            ],
            "selected_bracket": (
                None if self.selected_bracket is None else bracket(self.selected_bracket)
            ),
            "positive_left_secant": self.positive_left_secant,
            "negative_right_secant": self.negative_right_secant,
            "stable_branch_extent": {
                "model_count": len(self.stable_branch_models),
                "maximum_central_pressure_mev_fm3": (
                    self.stable_branch_models[-1][0]
                    if self.stable_branch_models
                    else None
                ),
                "models": [model(row) for row in self.stable_branch_models],
            },
            "sampled_models": [model(row) for row in self.sampled_models],
            "eos_endpoint": {
                "pressure_mev_fm3": self.eos_endpoint_pressure_mev_fm3,
                "reached_by_search": self.endpoint_reached,
                "limitation": self.endpoint_limitation,
            },
            "convergence": {
                "refinement_status": self.refinement_status,
                "refinement_iterations": self.refinement_iterations,
                "global_refinement_rounds": self.global_refinement_rounds,
                "solver_call_count": self.solver_call_count,
                "solver_failure_count": self.solver_failure_count,
                "solver_failures": [
                    {
                        "central_pressure_mev_fm3": pressure,
                        "reason": reason,
                    }
                    for pressure, reason in self.solver_failures
                ],
            },
            "tidal_calculations_performed": 0,
        }

_PUBLIC_MODULE = "eos_generation.stellar.tov"
for _compatibility_object in (
    TovConvergenceError,
    _evidence_finite_float,
    _freeze_rows,
    _freeze_profiles,
    _rows_to_dicts,
    _rows_equal,
    TovFailureDetail,
    _optional_finite,
    AppliedTidalJump,
    TovLambdaDiagnostic,
    TovMassSecantEvidence,
    TovSequenceEvidence,
    TidalAlgebraResult,
    TovStarResult,
    TovMaximumMassResult,
):
    _compatibility_object.__module__ = _PUBLIC_MODULE
del _compatibility_object

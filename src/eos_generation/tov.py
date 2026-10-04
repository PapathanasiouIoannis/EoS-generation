"""Tov for controlled BSk24 / BSk25 experiments."""

from __future__ import annotations

import math
import numba
import numpy as np
from .numerics import DEFAULT_CONFIG, TovConfig
from dataclasses import dataclass
from decimal import Decimal, localcontext
from scipy.integrate import solve_ivp
from typing import Any, Callable, ClassVar, Iterable, Literal


EOS_DISCONTINUITY_CONTRACT_VERSION = "eos_discontinuity_v1"


DISCONTINUITY_KINDS = ("internal", "surface")


BARE_SELF_BOUND_SEQUENCE_POLICY = "bare_self_bound_positive_mass_radius_v1"


SEED_PRESERVING_LOCAL_REFINEMENT_POLICY = "seed_preserving_split_log_pressure_v1"


def _finite(name: str, value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class EosDiscontinuity:
    """One explicitly declared outward energy-density discontinuity.

    ``inner_energy_density - outer_energy_density`` is the signed outward
    ``delta_energy_density``. Negative internal seams are valid and retained.
    """

    identifier: str
    kind: Literal["internal", "surface"]
    pressure: float
    inner_energy_density: float
    outer_energy_density: float
    delta_energy_density: float
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise ValueError("discontinuity identifier must be a non-empty string")
        if self.kind not in DISCONTINUITY_KINDS:
            raise ValueError(f"discontinuity kind must be one of {DISCONTINUITY_KINDS}")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("discontinuity provenance must be a non-empty string")

        pressure = _finite("discontinuity pressure", self.pressure)
        inner = _finite("inner energy density", self.inner_energy_density)
        outer = _finite("outer energy density", self.outer_energy_density)
        delta = _finite(
            "signed outward delta energy density", self.delta_energy_density
        )
        if pressure < 0.0:
            raise ValueError("discontinuity pressure must be nonnegative")
        if inner < 0.0 or outer < 0.0:
            raise ValueError("one-sided energy densities must be nonnegative")
        if self.kind == "internal" and pressure <= 0.0:
            raise ValueError("internal discontinuities require positive pressure")
        if self.kind == "surface":
            if pressure != 0.0:
                raise ValueError("surface discontinuities must be declared at P=0")
            if inner <= 0.0 or outer != 0.0:
                raise ValueError(
                    "a bare surface requires positive inner density and vacuum outside"
                )

        expected = inner - outer
        tolerance = 8.0 * math.ulp(max(abs(inner), abs(outer), 1.0))
        if not math.isclose(delta, expected, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                "delta_energy_density must equal inner_energy_density - outer_energy_density"
            )
        object.__setattr__(self, "pressure", pressure)
        object.__setattr__(self, "inner_energy_density", inner)
        object.__setattr__(self, "outer_energy_density", outer)
        object.__setattr__(self, "delta_energy_density", delta)

    @classmethod
    def from_sides(
        cls,
        *,
        identifier: str,
        kind: Literal["internal", "surface"],
        pressure: float,
        inner_energy_density: float,
        outer_energy_density: float,
        provenance: str,
    ) -> "EosDiscontinuity":
        inner = float(inner_energy_density)
        outer = float(outer_energy_density)
        return cls(
            identifier=identifier,
            kind=kind,
            pressure=pressure,
            inner_energy_density=inner,
            outer_energy_density=outer,
            delta_energy_density=inner - outer,
            provenance=provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EOS_DISCONTINUITY_CONTRACT_VERSION,
            "identifier": self.identifier,
            "type": self.kind,
            "pressure_MeV_fm3": self.pressure,
            "inner_energy_density_MeV_fm3": self.inner_energy_density,
            "outer_energy_density_MeV_fm3": self.outer_energy_density,
            "signed_outward_delta_energy_density_MeV_fm3": self.delta_energy_density,
            "provenance": self.provenance,
        }


def validate_discontinuity_sequence(
    values: Iterable[EosDiscontinuity],
) -> tuple[EosDiscontinuity, ...]:
    """Return a validated descending-pressure discontinuity tuple."""
    resolved = tuple(values)
    if any(not isinstance(item, EosDiscontinuity) for item in resolved):
        raise TypeError("discontinuities must contain only EosDiscontinuity values")
    identifiers = [item.identifier for item in resolved]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("discontinuity identifiers must be unique")
    pressures = [item.pressure for item in resolved]
    if any(upper <= lower for upper, lower in zip(pressures, pressures[1:])):
        raise ValueError(
            "discontinuities must be in strictly descending pressure order"
        )
    surface_indices = [
        index for index, item in enumerate(resolved) if item.kind == "surface"
    ]
    if len(surface_indices) > 1:
        raise ValueError("at most one bare surface discontinuity may be declared")
    if surface_indices and surface_indices[0] != len(resolved) - 1:
        raise ValueError("a surface discontinuity must be the final ordered entry")
    return resolved


DENOMINATOR_FLOOR = 1.0e-25


TOV_SEQUENCE_EVIDENCE_VERSION = "tov_sequence_evidence_v2"


TIDAL_CORRECTION_VERSION = "hinderer_takatsy_postnikov_v1"


LAMBDA_FRAMEWORK_CAPABILITY = "validated_lambda_validation_v1"


LAMBDA_SCIENTIFIC_STATUS = LAMBDA_FRAMEWORK_CAPABILITY


TIDAL_CORRECTION_STATUS = "validated_conditional_per_calculation"


TIDAL_NOT_REQUESTED_STATUS = "not_requested_background_only"


TIDAL_JUMP_FORMULA = (
    "delta_y=-G_CONV*r^3*(epsilon_inner-epsilon_outer)/" "(m+G_CONV*r^3*P)"
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
        self.message = (
            message
            or f"TOV solver failed to converge at central pressure {pc}. Reason: {reason}"
        )
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
            raise ValueError(
                f"{name} row {row_index} must contain {len(fields)} values"
            )
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
            raise ValueError(
                f"{name} profile {profile_index} must contain radius and mass arrays"
            )
        radius = tuple(
            _evidence_finite_float(f"{name}[{profile_index}].radius", value)
            for value in values[0]
        )
        mass = tuple(
            _evidence_finite_float(f"{name}[{profile_index}].mass", value)
            for value in values[1]
        )
        if len(radius) != len(mass):
            raise ValueError(
                f"{name} profile {profile_index} radius and mass lengths differ"
            )
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
        np.array_equal(
            np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True
        )
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
            object.__setattr__(
                self, name, _evidence_finite_float(name, getattr(self, name))
            )
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
        for name in (
            "central_pressure",
            "mass",
            "radius",
            "compactness",
            "surface_event_pressure",
        ):
            object.__setattr__(
                self, name, _evidence_finite_float(name, getattr(self, name))
            )
        if self.mass <= 0.0 or self.radius <= 0.0 or self.compactness <= 0.0:
            raise ValueError("tidal diagnostic background values must be positive")
        if (
            isinstance(self.expected_jump_count, bool)
            or int(self.expected_jump_count) < 0
        ):
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
                raise ValueError(
                    "completed tidal diagnostics require every expected jump"
                )
            if any(
                value is None
                for value in (
                    self.y_surface_interior,
                    self.y_supplied_to_k2,
                    self.k2,
                    self.lambda_dimensionless,
                )
            ):
                raise ValueError(
                    "completed tidal diagnostics require finite tidal outputs"
                )
            if self.failure_reason is not None:
                raise ValueError(
                    "completed tidal diagnostics cannot contain a failure reason"
                )
        elif self.scientific_status not in {
            "failed_closed",
            TIDAL_NOT_REQUESTED_STATUS,
        }:
            raise ValueError(
                "scientific_status must be validated_lambda_validation_v1, "
                "failed_closed, or not_requested_background_only"
            )
        elif self.scientific_status == TIDAL_NOT_REQUESTED_STATUS:
            if (
                any(
                    value is not None
                    for value in (
                        self.y_surface_interior,
                        self.y_surface_vacuum,
                        self.y_supplied_to_k2,
                        self.k2,
                        self.lambda_dimensionless,
                        self.failure_reason,
                    )
                )
                or jumps
            ):
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
            "calculation_lambda_validated": self.scientific_status
            == LAMBDA_FRAMEWORK_CAPABILITY,
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
            object.__setattr__(
                self, name, _evidence_finite_float(name, getattr(self, name))
            )
        if self.upper_central_pressure <= self.lower_central_pressure:
            raise ValueError(
                "TOV mass-secant central pressures must be strictly increasing"
            )
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
        full_profiles = _freeze_profiles(
            self.full_dense_profiles, name="full_dense_profiles"
        )
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
        full_lambda = (
            None
            if self.full_lambda_diagnostics is None
            else tuple(self.full_lambda_diagnostics)
        )
        stable_lambda = (
            None
            if self.stable_lambda_diagnostics is None
            else tuple(self.stable_lambda_diagnostics)
        )
        if full_lambda is not None and any(
            not isinstance(item, TovLambdaDiagnostic) for item in full_lambda
        ):
            raise TypeError(
                "full_lambda_diagnostics must contain TovLambdaDiagnostic values"
            )
        if stable_lambda is not None and any(
            not isinstance(item, TovLambdaDiagnostic) for item in stable_lambda
        ):
            raise TypeError(
                "stable_lambda_diagnostics must contain TovLambdaDiagnostic values"
            )
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
            raise TypeError(
                "failed_central_pressures must contain TovFailureDetail values"
            )
        if any(
            not isinstance(item, TovMassSecantEvidence)
            for item in pre_slopes + post_slopes
        ):
            raise TypeError(
                "TOV slope evidence must contain TovMassSecantEvidence values"
            )

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
            raise ValueError(
                "full and stable tidal diagnostics must be requested together"
            )
        if (full_lambda is None) != (stable_lambda is None):
            raise ValueError(
                "full and stable Lambda diagnostics must be requested together"
            )
        if (full_tidal is None) != (full_lambda is None):
            raise ValueError(
                "numeric and correction tidal diagnostics must be requested together"
            )
        if full_lambda is not None and len(full_lambda) != len(full):
            raise ValueError("full Lambda-diagnostic and sequence counts differ")
        if stable_lambda is not None and len(stable_lambda) != len(stable):
            raise ValueError("stable Lambda-diagnostic and sequence counts differ")
        if not _rows_equal(stable, full[: len(stable)]):
            raise ValueError(
                "stable TOV sequence must be a prefix of the full sequence"
            )
        if stable_profiles != full_profiles[: len(stable_profiles)]:
            raise ValueError(
                "stable dense profiles must be a prefix of full dense profiles"
            )
        if full_tidal is not None and not _rows_equal(
            stable_tidal, full_tidal[: len(stable_tidal)]
        ):
            raise ValueError(
                "stable tidal diagnostics must be a prefix of full diagnostics"
            )
        if (
            full_lambda is not None
            and stable_lambda != full_lambda[: len(stable_lambda)]
        ):
            raise ValueError(
                "stable Lambda diagnostics must be a prefix of full diagnostics"
            )
        if successful != tuple(row[3] for row in full):
            raise ValueError(
                "successful central pressures must match the full sequence"
            )
        if len(attempted) != len(full) + len(failures):
            raise ValueError(
                "each attempted central pressure must be successful or have failure detail"
            )
        if len(attempted) > 1 and np.any(np.diff(attempted) <= 0.0):
            raise ValueError("attempted central pressures must be strictly increasing")
        expected_ordering = (
            "unavailable"
            if not successful
            else (
                "single_sample"
                if len(successful) == 1
                else (
                    "strictly_increasing"
                    if np.all(np.diff(successful) > 0.0)
                    else "invalid"
                )
            )
        )
        if self.central_pressure_ordering != expected_ordering:
            raise ValueError(
                "central_pressure_ordering does not match successful sequence"
            )

        if not full:
            if any(
                value is not None
                for value in (
                    self.sampled_peak_index,
                    self.sampled_peak_row,
                    self.domain_end_row,
                )
            ):
                raise ValueError(
                    "empty TOV evidence cannot contain sampled peak or domain-end rows"
                )
            if stable or pre_slopes or post_slopes or self.sampled_peak_is_interior:
                raise ValueError(
                    "empty TOV evidence cannot contain stable rows or slope evidence"
                )
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
                raise ValueError(
                    "sampled peak or domain-end row does not match full sequence"
                )
            if not _rows_equal(stable, full[: peak_index + 1]):
                raise ValueError("stable TOV view must end at the sampled argmax")
            if self.sampled_peak_is_interior != (0 < peak_index < len(full) - 1):
                raise ValueError(
                    "sampled_peak_is_interior does not match sampled peak index"
                )
            if self.max_mass_stable != peak_row[0]:
                raise ValueError("max_mass_stable must equal the sampled peak mass")

        endpoint = self.eos_endpoint_pressure
        margin = self.eos_endpoint_margin
        contact = self.final_available_model_contacts_eos_endpoint
        if endpoint is None:
            if margin is not None or contact is not None:
                raise ValueError(
                    "endpoint margin/contact require an explicit EoS endpoint"
                )
        else:
            endpoint = _evidence_finite_float("eos_endpoint_pressure", endpoint)
            object.__setattr__(self, "eos_endpoint_pressure", endpoint)
            if not full:
                if margin is not None or contact is not None:
                    raise ValueError(
                        "empty TOV evidence has no final model for endpoint comparison"
                    )
            else:
                margin = _evidence_finite_float("eos_endpoint_margin", margin)
                object.__setattr__(self, "eos_endpoint_margin", margin)
                expected_margin = endpoint - full[-1][3]
                if margin != expected_margin or contact != (expected_margin == 0.0):
                    raise ValueError(
                        "EoS endpoint margin/contact does not match the domain-end row"
                    )

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
            "stable_sequence": _rows_to_dicts(
                self.stable_sequence, TOV_SEQUENCE_FIELDS
            ),
            "full_dense_profiles": [
                {"radius_km": list(radius), "mass_solar": list(mass)}
                for radius, mass in self.full_dense_profiles
            ],
            "stable_dense_profiles": [
                {"radius_km": list(radius), "mass_solar": list(mass)}
                for radius, mass in self.stable_dense_profiles
            ],
            "full_tidal_diagnostics": (
                None
                if self.full_tidal_diagnostics is None
                else _rows_to_dicts(
                    self.full_tidal_diagnostics, TOV_TIDAL_DIAGNOSTIC_FIELDS
                )
            ),
            "stable_tidal_diagnostics": (
                None
                if self.stable_tidal_diagnostics is None
                else _rows_to_dicts(
                    self.stable_tidal_diagnostics, TOV_TIDAL_DIAGNOSTIC_FIELDS
                )
            ),
            "full_lambda_diagnostics": (
                None
                if self.full_lambda_diagnostics is None
                else [item.to_dict() for item in self.full_lambda_diagnostics]
            ),
            "stable_lambda_diagnostics": (
                None
                if self.stable_lambda_diagnostics is None
                else [item.to_dict() for item in self.stable_lambda_diagnostics]
            ),
            "attempted_central_pressures": list(self.attempted_central_pressures),
            "successful_central_pressures": list(self.successful_central_pressures),
            "central_pressure_ordering": self.central_pressure_ordering,
            "attempted_central_pressure_count": self.attempted_central_pressure_count,
            "successful_row_count": self.successful_row_count,
            "failed_central_pressure_count": self.failed_central_pressure_count,
            "failed_central_pressures": [
                item.to_dict() for item in self.failed_central_pressures
            ],
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
                None
                if self.selected_bracket is None
                else bracket(self.selected_bracket)
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


@numba.njit
def taylor_expansion(
    r: float,
    P_safe: float,
    epsilon: float,
    cs2_local: float,
    G_CONV: float,
    A_CONV: float,
) -> list:
    dm_dr = (r**2) * epsilon * G_CONV
    dP_dr = -A_CONV * G_CONV * (epsilon + P_safe) * (epsilon / 3.0 + P_safe) * r
    dy_dr = (
        -(2.0 / 7.0)
        * A_CONV
        * G_CONV
        * r
        * (11.0 * P_safe + epsilon / 3.0 + (epsilon + P_safe) / cs2_local)
    )
    return [dm_dr, dP_dr, dy_dr]


@numba.njit
def _tov_equations_with_limit(
    r: float,
    m: float,
    P_safe: float,
    y_tidal: float,
    epsilon: float,
    cs2_local: float,
    G_CONV: float,
    A_CONV: float,
    singularity_limit: float,
) -> list:
    term_1 = epsilon + P_safe
    term_2 = m + (r**3 * P_safe * G_CONV)
    term_3 = r * (r - 2.0 * m * A_CONV)

    if abs(term_3) < singularity_limit:
        return [0.0, 0.0, 0.0]

    dP_dr = -A_CONV * (term_1 * term_2) / term_3
    dm_dr = (r**2) * epsilon * G_CONV

    exp_lambda = 1.0 / (1.0 - 2.0 * A_CONV * m / r)
    F = (1.0 - A_CONV * G_CONV * (r**2) * (epsilon - P_safe)) * exp_lambda
    matter_source = (
        A_CONV
        * G_CONV
        * (5.0 * epsilon + 9.0 * P_safe + (epsilon + P_safe) / cs2_local)
        * (r**2)
    )
    pressure_mass_source = A_CONV * (m + (r**3 * P_safe * G_CONV))
    nu_prime_squared_source = (
        4.0 * (pressure_mass_source / (r * (1.0 - 2.0 * A_CONV * m / r))) ** 2
    )
    Q = (matter_source - 6.0) * exp_lambda - nu_prime_squared_source
    dy_dr = -(y_tidal**2 + y_tidal * F + Q) / r
    return [dm_dr, dP_dr, dy_dr]


@numba.njit
def tov_equations(
    r: float,
    m: float,
    P_safe: float,
    y_tidal: float,
    epsilon: float,
    cs2_local: float,
    G_CONV: float,
    A_CONV: float,
) -> list:
    return _tov_equations_with_limit(
        r,
        m,
        P_safe,
        y_tidal,
        epsilon,
        cs2_local,
        G_CONV,
        A_CONV,
        _TOV_SINGULARITY_LIMIT,
    )


def tov_rhs(
    r: float,
    y_state: list,
    eos_callable: Callable,
    settings: TovConfig | None = None,
) -> list:
    """Return historical TOV/tidal derivatives for one radius and state."""
    resolved = _DEFAULT_TOV if settings is None else settings
    r = max(r, 1.0e-10)
    m, pressure, y_tidal = y_state
    if pressure < resolved.surface_pressure_cutoff:
        return [0.0, 0.0, 0.0]

    pressure_safe = max(pressure, resolved.pressure_min_safe)
    epsilon, cs2_local = eos_callable(pressure_safe)
    if epsilon <= 0:
        return [0.0, 0.0, 0.0]
    if cs2_local < 1.0e-10:
        cs2_local = 1.0e-10
    if r <= resolved.center_radius_limit:
        return taylor_expansion(r, pressure_safe, epsilon, cs2_local, _G_CONV, _A_CONV)
    if resolved.singularity_limit == _TOV_SINGULARITY_LIMIT:
        return tov_equations(
            r, m, pressure_safe, y_tidal, epsilon, cs2_local, _G_CONV, _A_CONV
        )
    return _tov_equations_with_limit(
        r,
        m,
        pressure_safe,
        y_tidal,
        epsilon,
        cs2_local,
        _G_CONV,
        _A_CONV,
        resolved.singularity_limit,
    )


def surface_event(_radius, state, *args):
    """Return pressure minus the configured historical surface cutoff."""
    settings = args[-1] if args and isinstance(args[-1], TovConfig) else _DEFAULT_TOV
    return state[1] - settings.surface_pressure_cutoff


surface_event.terminal = True


surface_event.direction = -1


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _love_number_k2_decimal(compactness: float, y_r: float) -> float:
    """Evaluate the corrected Hinderer expression with decimal guard digits."""
    with localcontext() as context:
        context.prec = 100
        c = Decimal(str(compactness))
        y = Decimal(str(y_r))
        one = Decimal(1)
        two = Decimal(2)
        three = Decimal(3)
        four = Decimal(4)
        five = Decimal(5)
        eight = Decimal(8)
        factor = one - two * c
        numerator = (eight / five) * factor**2 * c**5 * (two * c * (y - one) - y + two)
        denominator = (
            two * c * (Decimal(6) - three * y + three * c * (five * y - eight))
            + four
            * c**3
            * (
                Decimal(13)
                - Decimal(11) * y
                + c * (three * y - two)
                + two * c**2 * (one + y)
            )
            + three * factor**2 * (two - y + two * c * (y - one)) * factor.ln()
        )
        if denominator == 0:
            raise ValueError("tidal Love-number denominator is zero")
        value = numerator / denominator
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("tidal Love number is nonfinite")
    return result


def love_number_k2(
    compactness: float,
    y_r: float,
    *,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> float:
    """Return the corrected Hinderer dimensionless tidal Love number ``k2``.

    Postnikov et al. explicitly identify severe cancellation for ``C < 0.1``.
    That range is evaluated with standard-library decimal guard digits rather
    than imposing an unsupported compactness floor.
    """
    c = _require_finite("compactness", compactness)
    y = _require_finite("y_r", y_r)
    floor = abs(_require_finite("denominator_floor", denominator_floor))
    if c <= 0.0 or c >= 0.5:
        raise ValueError("compactness must satisfy 0 < C < 0.5 for the log term")
    if c < 0.1:
        return _love_number_k2_decimal(c, y)

    numerator = (
        (8.0 / 5.0) * (1.0 - 2.0 * c) ** 2 * c**5 * (2.0 * c * (y - 1.0) - y + 2.0)
    )
    denominator_term1 = 2.0 * c * (6.0 - 3.0 * y + 3.0 * c * (5.0 * y - 8.0))
    denominator_term2 = (
        4.0 * c**3 * (13.0 - 11.0 * y + c * (3.0 * y - 2.0) + 2.0 * c**2 * (1.0 + y))
    )
    denominator_term3 = (
        3.0
        * (1.0 - 2.0 * c) ** 2
        * (2.0 - y + 2.0 * c * (y - 1.0))
        * np.log1p(-2.0 * c)
    )
    denominator = denominator_term1 + denominator_term2 + denominator_term3
    if abs(denominator) < floor:
        raise ValueError("tidal Love-number denominator is too close to zero")
    return float(numerator / denominator)


def dimensionless_lambda(compactness: float, k2: float) -> float:
    """Return historical dimensionless tidal deformability from ``C`` and ``k2``."""
    c = _require_finite("compactness", compactness)
    k2_value = _require_finite("k2", k2)
    if c <= 0.0:
        raise ValueError("compactness must be positive")
    return float((2.0 / 3.0) * k2_value * c**-5)


def tidal_algebra(
    compactness: float,
    y_r: float,
    *,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> TidalAlgebraResult:
    """Return historical ``k2`` and ``Lambda`` for one surface state."""
    k2_value = love_number_k2(compactness, y_r, denominator_floor=denominator_floor)
    return TidalAlgebraResult(
        compactness=float(compactness),
        y_r=float(y_r),
        k2=k2_value,
        lambda_dimensionless=dimensionless_lambda(compactness, k2_value),
    )


def tidal_jump_delta_y(
    *,
    radius_km: float,
    mass_msun: float,
    pressure_mev_fm3: float,
    delta_energy_density_mev_fm3: float,
) -> tuple[float, float]:
    """Return ``(delta_y, denominator)`` in repository solver units.

    The geometric matching condition of Takatsy and Kovacs (2020), Eq. 11,
    reduces here because ``dm/dr = G_CONV * epsilon * r**2`` and mass is in
    solar masses. The finite-pressure term is retained for internal joins.
    """
    radius = _require_finite("radius_km", radius_km)
    mass = _require_finite("mass_msun", mass_msun)
    pressure = _require_finite("pressure_mev_fm3", pressure_mev_fm3)
    delta_epsilon = _require_finite(
        "delta_energy_density_mev_fm3", delta_energy_density_mev_fm3
    )
    if radius <= 0.0 or mass <= 0.0 or pressure < 0.0:
        raise ValueError("jump radius/mass must be positive and pressure nonnegative")
    pressure_mass = _G_CONV * radius**3 * pressure
    denominator = mass + pressure_mass
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(
            "tidal jump correction denominator must be finite and positive"
        )
    delta_y = -(_G_CONV * radius**3 * delta_epsilon) / denominator
    if not math.isfinite(delta_y):
        raise ValueError("tidal jump correction must be finite")
    return float(delta_y), float(denominator)


@dataclass
class _BackgroundSegment:
    solution: Any
    upper_discontinuity: EosDiscontinuity | None
    lower_discontinuity: EosDiscontinuity | None

    @property
    def radius_start(self) -> float:
        return float(self.solution.t[0])

    @property
    def radius_end(self) -> float:
        return float(self.solution.t_events[0][0])

    @property
    def event_state(self) -> np.ndarray:
        return np.asarray(self.solution.y_events[0][0], dtype=float)


def _tidal_segment_bounds(
    segment: _BackgroundSegment,
) -> tuple[float, float, float]:
    """Return validated radial bounds and their floating-point allowance."""
    radius_start = segment.radius_start
    radius_end = segment.radius_end
    if not math.isfinite(radius_start) or not math.isfinite(radius_end):
        raise ValueError("tidal background segment bounds must be finite")
    if radius_end <= radius_start:
        raise ValueError("tidal background segment must have positive radial span")
    allowance = _TIDAL_SEGMENT_BOUNDARY_ULPS * math.ulp(
        max(abs(radius_start), abs(radius_end), 1.0)
    )
    return radius_start, radius_end, allowance


def _bounded_tidal_first_step(
    segment: _BackgroundSegment,
    settings: TovConfig,
    *,
    scale: float = 1.0,
) -> float:
    """Return a positive RK45 first step bounded by the background segment.

    The nominal scale is the existing center/start radius (``radius_min_km``).
    The private ``scale`` argument exists only for numerical sensitivity tests.
    """
    radius_start, radius_end, _allowance = _tidal_segment_bounds(segment)
    multiplier = float(scale)
    nominal = float(settings.radius_min_km) * multiplier
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("tidal first-step scale must be finite and positive")
    if not math.isfinite(nominal) or nominal <= 0.0:
        raise ValueError("tidal first-step nominal radius must be finite and positive")
    first_step = min(radius_end - radius_start, nominal)
    if not math.isfinite(first_step) or first_step <= 0.0:
        raise ValueError("tidal first step must be finite and positive")
    return float(first_step)


def _assert_tidal_radius_on_segment(
    radius: float,
    segment: _BackgroundSegment,
) -> float:
    """Fail before dense-background evaluation when a radius leaves its segment."""
    requested = float(radius)
    if not math.isfinite(requested):
        raise ValueError("tidal RHS radius must be finite")
    radius_start, radius_end, allowance = _tidal_segment_bounds(segment)
    if requested < radius_start - allowance or requested > radius_end + allowance:
        raise ValueError(
            "tidal RHS radius is outside its background segment: "
            f"requested={requested!r}, bounds=({radius_start!r}, {radius_end!r}), "
            f"allowance={allowance!r}"
        )
    return requested


def _discontinuity_metadata_is_required(eos_callable: Callable) -> bool:
    """Return whether this EoS must use its declared discontinuity path."""
    try:
        surface_density = float(getattr(eos_callable, "eps_surf", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("eps_surf metadata must be finite and nonnegative") from exc
    if not math.isfinite(surface_density) or surface_density < 0.0:
        raise ValueError("eps_surf metadata must be finite and nonnegative")
    return (
        bool(getattr(eos_callable, "requires_discontinuity_metadata", False))
        or surface_density > 0.0
    )


def _resolved_discontinuities(eos_callable: Callable) -> tuple[EosDiscontinuity, ...]:
    required = _discontinuity_metadata_is_required(eos_callable)
    surface_density = float(getattr(eos_callable, "eps_surf", 0.0))
    declared = getattr(eos_callable, "discontinuities", None)
    if declared is None:
        if required:
            raise ValueError("required EoS discontinuity metadata are absent")
        return ()
    resolved = validate_discontinuity_sequence(declared)
    if required and not resolved:
        raise ValueError("required EoS discontinuity metadata are empty")
    surfaces = [item for item in resolved if item.kind == "surface"]
    if surface_density > 0.0:
        if len(surfaces) != 1:
            raise ValueError("a finite eps_surf requires one declared bare surface")
        if not math.isclose(
            surfaces[0].inner_energy_density,
            surface_density,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("declared surface density disagrees with eps_surf")
    elif surfaces:
        raise ValueError("a declared bare surface requires positive eps_surf metadata")
    return resolved


def _segment_pressure_floor(
    *,
    lower_discontinuity: EosDiscontinuity | None,
    settings: TovConfig,
) -> float:
    """Use the physical vacuum boundary on an explicit bare-surface segment."""
    if lower_discontinuity is not None and lower_discontinuity.kind == "surface":
        return 0.0
    return float(settings.pressure_min_safe)


def _applicable_discontinuities(
    discontinuities: tuple[EosDiscontinuity, ...],
    central_pressure: float,
) -> tuple[tuple[EosDiscontinuity, ...], tuple[str, ...]]:
    applicable = []
    skipped = []
    for item in discontinuities:
        if item.kind == "surface" or item.pressure < central_pressure:
            applicable.append(item)
        else:
            skipped.append(item.identifier)
    return tuple(applicable), tuple(skipped)


def _branch_pressure(
    pressure: float,
    *,
    upper_discontinuity: EosDiscontinuity | None,
    lower_discontinuity: EosDiscontinuity | None,
    settings: TovConfig,
) -> float:
    pressure_floor = _segment_pressure_floor(
        lower_discontinuity=lower_discontinuity,
        settings=settings,
    )
    candidate = max(float(pressure), pressure_floor)
    if upper_discontinuity is not None and candidate >= upper_discontinuity.pressure:
        candidate = float(np.nextafter(upper_discontinuity.pressure, -math.inf))
    if (
        lower_discontinuity is not None
        and lower_discontinuity.kind == "internal"
        and candidate <= lower_discontinuity.pressure
    ):
        candidate = float(np.nextafter(lower_discontinuity.pressure, math.inf))
    return max(candidate, pressure_floor)


def _evaluate_branch(
    eos_callable: Callable,
    pressure: float,
    *,
    upper_discontinuity: EosDiscontinuity | None,
    lower_discontinuity: EosDiscontinuity | None,
    settings: TovConfig,
) -> tuple[float, float]:
    evaluation_pressure = _branch_pressure(
        pressure,
        upper_discontinuity=upper_discontinuity,
        lower_discontinuity=lower_discontinuity,
        settings=settings,
    )
    epsilon, cs2_local = eos_callable(evaluation_pressure)
    epsilon = float(epsilon)
    cs2_local = float(cs2_local)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("EoS branch returned nonfinite or nonpositive energy density")
    if not math.isfinite(cs2_local):
        raise ValueError("EoS branch returned nonfinite sound speed")
    if cs2_local <= 0.0:
        raise ValueError("EoS branch returned nonpositive sound speed")
    return epsilon, cs2_local


def _evaluate_background_branch(
    eos_callable: Callable,
    pressure: float,
    *,
    upper_discontinuity: EosDiscontinuity | None,
    lower_discontinuity: EosDiscontinuity | None,
    settings: TovConfig,
) -> float:
    """Evaluate only epsilon(P) for an explicitly certified immutable EoS.

    Arbitrary callables retain the established full ``(epsilon, c_s^2)``
    validation path. Accepted reconstructed objects opt in only after their
    complete branches have passed construction-time checks.
    """

    if getattr(eos_callable, "_background_energy_only_is_certified", False) is True:
        evaluator = getattr(eos_callable, "energy_density_from_pressure", None)
        if not callable(evaluator):
            raise ValueError("certified background EoS has no energy-density evaluator")
        evaluation_pressure = _branch_pressure(
            pressure,
            upper_discontinuity=upper_discontinuity,
            lower_discontinuity=lower_discontinuity,
            settings=settings,
        )
        epsilon = float(evaluator(evaluation_pressure))
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(
                "EoS branch returned nonfinite or nonpositive energy density"
            )
        return epsilon
    epsilon, _cs2_local = _evaluate_branch(
        eos_callable,
        pressure,
        upper_discontinuity=upper_discontinuity,
        lower_discontinuity=lower_discontinuity,
        settings=settings,
    )
    return epsilon


def _background_rhs(
    radius: float,
    state: np.ndarray,
    eos_callable: Callable,
    *,
    upper_discontinuity: EosDiscontinuity | None,
    lower_discontinuity: EosDiscontinuity | None,
    settings: TovConfig,
) -> list[float]:
    radius = max(float(radius), 1.0e-10)
    mass, pressure = map(float, state)
    epsilon = _evaluate_background_branch(
        eos_callable,
        pressure,
        upper_discontinuity=upper_discontinuity,
        lower_discontinuity=lower_discontinuity,
        settings=settings,
    )
    pressure_safe = max(
        pressure,
        _segment_pressure_floor(
            lower_discontinuity=lower_discontinuity,
            settings=settings,
        ),
    )
    dm_dr = radius**2 * epsilon * _G_CONV
    if radius <= settings.center_radius_limit:
        dpressure_dr = (
            -_A_CONV
            * _G_CONV
            * (epsilon + pressure_safe)
            * (epsilon / 3.0 + pressure_safe)
            * radius
        )
        return [dm_dr, dpressure_dr]
    denominator = radius * (radius - 2.0 * mass * _A_CONV)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(
            "background TOV state reached the Schwarzschild-radius boundary"
        )
    dpressure_dr = (
        -_A_CONV
        * (epsilon + pressure_safe)
        * (mass + radius**3 * pressure_safe * _G_CONV)
        / denominator
    )
    if not math.isfinite(dm_dr) or not math.isfinite(dpressure_dr):
        raise ValueError("background TOV derivative is nonfinite")
    return [dm_dr, dpressure_dr]


def _pressure_event(target_pressure: float):
    def event(_radius: float, state: np.ndarray) -> float:
        return float(state[1] - target_pressure)

    event.terminal = True
    event.direction = -1
    return event


def _integrate_background(
    eos_callable: Callable,
    central_pressure: float,
    central_energy_density: float,
    discontinuities: tuple[EosDiscontinuity, ...],
    *,
    settings: TovConfig,
    rtol: float,
    atol: float,
    dense_output: bool = True,
) -> tuple[list[_BackgroundSegment], tuple[str, ...]]:
    applicable, skipped = _applicable_discontinuities(discontinuities, central_pressure)
    internal = [item for item in applicable if item.kind == "internal"]
    surface = next((item for item in applicable if item.kind == "surface"), None)
    targets: list[EosDiscontinuity | None] = [*internal, surface]

    radius_start = float(settings.radius_min_km)
    mass_start = radius_start**3 * central_energy_density * (_G_CONV / 3.0)
    state_start = np.asarray([mass_start, central_pressure], dtype=float)
    upper: EosDiscontinuity | None = None
    segments: list[_BackgroundSegment] = []
    for lower in targets:
        target_pressure = (
            settings.surface_pressure_cutoff if lower is None else lower.pressure
        )
        event = _pressure_event(target_pressure)

        def rhs(radius: float, state: np.ndarray) -> list[float]:
            return _background_rhs(
                radius,
                state,
                eos_callable,
                upper_discontinuity=upper,
                lower_discontinuity=lower,
                settings=settings,
            )

        solution = solve_ivp(
            rhs,
            (radius_start, settings.radius_max_km),
            state_start,
            events=event,
            method="RK45",
            dense_output=dense_output,
            rtol=rtol,
            atol=atol,
        )
        event_count = len(solution.t_events[0]) if solution.t_events else 0
        if solution.status != 1 or event_count != 1:
            identifier = "surface" if lower is None else lower.identifier
            raise RuntimeError(
                f"background event {identifier!r} was not reached exactly once "
                f"(status={solution.status}, count={event_count})"
            )
        solution.y_events[0][0][1] = target_pressure
        segment = _BackgroundSegment(solution, upper, lower)
        event_state = segment.event_state
        if not np.all(np.isfinite(event_state)):
            raise ValueError("background event state is nonfinite")
        if segments and segment.radius_start < segments[-1].radius_end:
            raise ValueError("background events are out of radial order")
        segments.append(segment)
        radius_start = segment.radius_end
        state_start = event_state.copy()
        state_start[1] = target_pressure
        upper = lower if lower is not None and lower.kind == "internal" else upper
    return segments, skipped


def _profile_from_segments(
    segments: list[_BackgroundSegment],
    *,
    points: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    radius_start = segments[0].radius_start
    radius_end = segments[-1].radius_end
    radii = np.linspace(radius_start, radius_end, int(points))
    masses = np.empty_like(radii)
    segment_index = 0
    for index, radius in enumerate(radii):
        while (
            segment_index < len(segments) - 1
            and radius > segments[segment_index].radius_end
        ):
            segment_index += 1
        masses[index] = float(segments[segment_index].solution.sol(radius)[0])
    return tuple(float(value) for value in radii), tuple(
        float(value) for value in masses
    )


def _validate_declared_branch_values(
    eos_callable: Callable,
    discontinuities: tuple[EosDiscontinuity, ...],
) -> None:
    for item in discontinuities:
        if item.kind == "internal":
            inner_pressure = float(np.nextafter(item.pressure, math.inf))
            outer_pressure = float(np.nextafter(item.pressure, -math.inf))
            evaluated_inner = float(eos_callable(inner_pressure)[0])
            evaluated_outer = float(eos_callable(outer_pressure)[0])
        else:
            evaluated_inner = float(eos_callable(0.0)[0])
            evaluated_outer = 0.0
        if not math.isfinite(evaluated_inner) or not math.isfinite(evaluated_outer):
            raise ValueError(
                f"declared jump {item.identifier} has nonfinite evaluated sides"
            )
        for side, declared, evaluated in (
            ("inner", item.inner_energy_density, evaluated_inner),
            ("outer", item.outer_energy_density, evaluated_outer),
        ):
            if not math.isclose(declared, evaluated, rel_tol=1.0e-8, abs_tol=1.0e-10):
                raise ValueError(
                    f"declared jump {item.identifier} {side} density disagrees with EoS branch: "
                    f"declared={declared!r}, evaluated={evaluated!r}"
                )


def _tidal_rhs_on_segment(
    radius: float,
    y_state: np.ndarray,
    segment: _BackgroundSegment,
    eos_callable: Callable,
    settings: TovConfig,
) -> list[float]:
    radius = _assert_tidal_radius_on_segment(radius, segment)
    mass, pressure = map(float, segment.solution.sol(radius))
    epsilon, cs2_local = _evaluate_branch(
        eos_callable,
        pressure,
        upper_discontinuity=segment.upper_discontinuity,
        lower_discontinuity=segment.lower_discontinuity,
        settings=settings,
    )
    pressure_safe = max(
        pressure,
        _segment_pressure_floor(
            lower_discontinuity=segment.lower_discontinuity,
            settings=settings,
        ),
    )
    y_tidal = float(y_state[0])
    if radius <= settings.center_radius_limit:
        derivative = taylor_expansion(
            radius,
            pressure_safe,
            epsilon,
            cs2_local,
            _G_CONV,
            _A_CONV,
        )[2]
    else:
        derivative = _tov_equations_with_limit(
            radius,
            mass,
            pressure_safe,
            y_tidal,
            epsilon,
            cs2_local,
            _G_CONV,
            _A_CONV,
            0.0,
        )[2]
    if not math.isfinite(derivative):
        raise ValueError("tidal derivative is nonfinite")
    return [float(derivative)]


def _integrate_tidal(
    eos_callable: Callable,
    central_pressure: float,
    segments: list[_BackgroundSegment],
    discontinuities: tuple[EosDiscontinuity, ...],
    skipped_ids: tuple[str, ...],
    *,
    settings: TovConfig,
    rtol: float,
    atol: float,
) -> TovLambdaDiagnostic:
    applicable, expected_skipped = _applicable_discontinuities(
        discontinuities, central_pressure
    )
    if expected_skipped != skipped_ids:
        raise ValueError("background and tidal phase selection disagree")
    expected = len(applicable)
    y_value = 2.0
    applied: list[AppliedTidalJump] = []
    y_surface_interior = None
    for segment in segments:
        first_step = _bounded_tidal_first_step(segment, settings)
        solution = solve_ivp(
            lambda radius, state: _tidal_rhs_on_segment(
                radius, state, segment, eos_callable, settings
            ),
            (segment.radius_start, segment.radius_end),
            [y_value],
            method="RK45",
            rtol=rtol,
            atol=atol,
            first_step=first_step,
        )
        if not solution.success or len(solution.t) == 0:
            raise RuntimeError(f"tidal segment integration failed: {solution.message}")
        y_value = float(solution.y[0, -1])
        if not math.isfinite(y_value):
            raise ValueError("tidal segment endpoint is nonfinite")
        lower = segment.lower_discontinuity
        if lower is None:
            y_surface_interior = y_value
            continue
        event_mass = float(segment.event_state[0])
        event_radius = segment.radius_end
        if lower.kind == "surface":
            y_surface_interior = y_value
        delta_y, denominator = tidal_jump_delta_y(
            radius_km=event_radius,
            mass_msun=event_mass,
            pressure_mev_fm3=lower.pressure,
            delta_energy_density_mev_fm3=lower.delta_energy_density,
        )
        if lower.kind == "surface" and delta_y >= 0.0:
            raise ValueError("a bare-surface tidal correction must be negative")
        y_after = y_value + delta_y
        if not math.isfinite(y_after):
            raise ValueError("post-jump tidal state is nonfinite")
        applied.append(
            AppliedTidalJump(
                identifier=lower.identifier,
                kind=lower.kind,
                pressure=lower.pressure,
                radius=event_radius,
                mass=event_mass,
                inner_energy_density=lower.inner_energy_density,
                outer_energy_density=lower.outer_energy_density,
                delta_energy_density=lower.delta_energy_density,
                correction_denominator=denominator,
                y_before=y_value,
                delta_y=delta_y,
                y_after=y_after,
                provenance=lower.provenance,
            )
        )
        y_value = y_after

    if y_surface_interior is None:
        raise ValueError("surface interior tidal value is unavailable")
    if len(applied) != expected:
        raise ValueError(
            f"expected {expected} discontinuity corrections but applied {len(applied)}"
        )
    expected_identifiers = tuple(item.identifier for item in applicable)
    applied_identifiers = tuple(item.identifier for item in applied)
    if applied_identifiers != expected_identifiers:
        raise ValueError(
            "applied discontinuity identities do not match the required sequence: "
            f"expected={expected_identifiers!r}, applied={applied_identifiers!r}"
        )
    final_state = segments[-1].event_state
    mass = float(final_state[0])
    radius = segments[-1].radius_end
    surface_pressure = float(final_state[1])
    explicit_bare_surface = any(item.kind == "surface" for item in applicable)
    expected_surface_pressure = (
        0.0 if explicit_bare_surface else settings.surface_pressure_cutoff
    )
    surface_tolerance = max(atol, abs(expected_surface_pressure) * 1.0e-6, 1.0e-14)
    if not math.isfinite(surface_pressure) or not math.isclose(
        surface_pressure,
        expected_surface_pressure,
        rel_tol=0.0,
        abs_tol=surface_tolerance,
    ):
        raise ValueError(
            "surface event was not localized at the declared policy pressure: "
            f"expected={expected_surface_pressure!r}, observed={surface_pressure!r}"
        )
    compactness = mass * _A_CONV / radius
    algebra = tidal_algebra(compactness, y_value)
    return TovLambdaDiagnostic(
        central_pressure=central_pressure,
        mass=mass,
        radius=radius,
        compactness=compactness,
        expected_jump_count=expected,
        applied_jumps=tuple(applied),
        skipped_discontinuity_ids=skipped_ids,
        surface_event_pressure=surface_pressure,
        y_surface_interior=y_surface_interior,
        y_surface_vacuum=y_value,
        y_supplied_to_k2=y_value,
        k2=algebra.k2,
        lambda_dimensionless=algebra.lambda_dimensionless,
        scientific_status=LAMBDA_SCIENTIFIC_STATUS,
    )


def _failed_lambda_diagnostic(
    *,
    central_pressure: float,
    mass: float,
    radius: float,
    surface_pressure: float,
    expected_jump_count: int,
    skipped_ids: tuple[str, ...],
    reason: str,
) -> TovLambdaDiagnostic:
    return TovLambdaDiagnostic(
        central_pressure=central_pressure,
        mass=mass,
        radius=radius,
        compactness=mass * _A_CONV / radius,
        expected_jump_count=expected_jump_count,
        applied_jumps=(),
        skipped_discontinuity_ids=skipped_ids,
        surface_event_pressure=surface_pressure,
        y_surface_interior=None,
        y_surface_vacuum=None,
        y_supplied_to_k2=None,
        k2=None,
        lambda_dimensionless=None,
        scientific_status="failed_closed",
        failure_reason=reason,
    )


def _background_only_lambda_diagnostic(
    *,
    central_pressure: float,
    mass: float,
    radius: float,
    surface_pressure: float,
    skipped_ids: tuple[str, ...],
) -> TovLambdaDiagnostic:
    """Record that no tidal ODE was requested for a valid background star."""

    return TovLambdaDiagnostic(
        central_pressure=central_pressure,
        mass=mass,
        radius=radius,
        compactness=mass * _A_CONV / radius,
        expected_jump_count=0,
        applied_jumps=(),
        skipped_discontinuity_ids=skipped_ids,
        surface_event_pressure=surface_pressure,
        y_surface_interior=None,
        y_surface_vacuum=None,
        y_supplied_to_k2=None,
        k2=None,
        lambda_dimensionless=None,
        scientific_status=TIDAL_NOT_REQUESTED_STATUS,
        failure_reason=None,
    )


def solve_star(
    eos_callable: Callable,
    central_pressure: float,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    settings: TovConfig | None = None,
    calculate_tidal: bool = True,
    retain_profile: bool = True,
) -> TovStarResult:
    """Solve one star, preserving a valid background when only Lambda fails.

    ``retain_profile=False`` is an exact-work optimization for internal
    searches that consume only the surface and central state.  It does not
    change the ODE, tolerances, events, or returned scalar observables.  The
    default remains ``True`` for public compatibility.

    Required discontinuity metadata and its segmented background are part of
    background validity.  Their failure therefore returns no star result.
    """
    resolved = _DEFAULT_TOV if settings is None else settings
    pc = _require_finite("central_pressure", central_pressure)
    if pc <= 0.0:
        raise ValueError("central_pressure must be positive")
    effective_rtol = (
        resolved.ode_rtol if rtol is None else _require_finite("rtol", rtol)
    )
    effective_atol = (
        resolved.ode_atol if atol is None else _require_finite("atol", atol)
    )
    if effective_rtol <= 0.0 or effective_atol <= 0.0:
        raise ValueError("TOV tolerances must be positive")
    if not isinstance(calculate_tidal, bool):
        raise ValueError("calculate_tidal must be boolean")
    if not isinstance(retain_profile, bool):
        raise ValueError("retain_profile must be boolean")

    metadata_required = _discontinuity_metadata_is_required(eos_callable)
    metadata_error = None
    try:
        discontinuities = _resolved_discontinuities(eos_callable)
    except (TypeError, ValueError) as exc:
        if metadata_required:
            raise ValueError(f"required_discontinuity_metadata:{exc}") from exc
        discontinuities = ()
        metadata_error = f"metadata_validation:{exc}"

    if metadata_required:
        try:
            _validate_declared_branch_values(eos_callable, discontinuities)
        except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
            raise ValueError(f"required_discontinuity_metadata:{exc}") from exc

    eps_init, cs2_init = map(float, eos_callable(pc))
    if not math.isfinite(eps_init) or eps_init <= 0.0:
        raise ValueError("initial energy density must be finite and positive")
    if not math.isfinite(cs2_init):
        raise ValueError("initial sound speed must be finite")

    try:
        segments, skipped = _integrate_background(
            eos_callable,
            pc,
            eps_init,
            discontinuities,
            settings=resolved,
            rtol=effective_rtol,
            atol=effective_atol,
            dense_output=bool(calculate_tidal or retain_profile),
        )
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        if metadata_required:
            raise RuntimeError(f"segmented_background:{exc}") from exc
        if not discontinuities:
            raise
        metadata_error = f"segmented_background:{exc}"
        segments, skipped = _integrate_background(
            eos_callable,
            pc,
            eps_init,
            (),
            settings=resolved,
            rtol=effective_rtol,
            atol=effective_atol,
            dense_output=bool(calculate_tidal or retain_profile),
        )

    final_state = segments[-1].event_state
    mass = float(final_state[0])
    radius = segments[-1].radius_end
    surface_pressure = float(final_state[1])
    if (
        not math.isfinite(mass)
        or not math.isfinite(radius)
        or mass <= 0.0
        or radius <= 0.0
    ):
        raise ValueError("background surface mass/radius is invalid")
    if (
        any(item.kind == "surface" for item in discontinuities)
        and surface_pressure != 0.0
    ):
        raise ValueError("a declared bare surface must terminate exactly at P=0")
    if retain_profile:
        radius_profile, mass_profile = _profile_from_segments(
            segments, points=resolved.dense_profile_points
        )
    else:
        radius_profile, mass_profile = (), ()

    if not calculate_tidal:
        _applicable, skipped_for_record = _applicable_discontinuities(
            discontinuities, pc
        )
        lambda_diagnostic = _background_only_lambda_diagnostic(
            central_pressure=pc,
            mass=mass,
            radius=radius,
            surface_pressure=surface_pressure,
            skipped_ids=skipped_for_record,
        )
    elif metadata_error is None:
        try:
            if not metadata_required:
                _validate_declared_branch_values(eos_callable, discontinuities)
            lambda_diagnostic = _integrate_tidal(
                eos_callable,
                pc,
                segments,
                discontinuities,
                skipped,
                settings=resolved,
                rtol=effective_rtol,
                atol=effective_atol,
            )
        except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
            applicable, skipped = _applicable_discontinuities(discontinuities, pc)
            lambda_diagnostic = _failed_lambda_diagnostic(
                central_pressure=pc,
                mass=mass,
                radius=radius,
                surface_pressure=surface_pressure,
                expected_jump_count=len(applicable),
                skipped_ids=skipped,
                reason=f"tidal_integration:{exc}",
            )
    else:
        lambda_diagnostic = _failed_lambda_diagnostic(
            central_pressure=pc,
            mass=mass,
            radius=radius,
            surface_pressure=surface_pressure,
            expected_jump_count=0,
            skipped_ids=(),
            reason=metadata_error,
        )

    return TovStarResult(
        central_pressure=pc,
        central_energy_density=eps_init,
        central_sound_speed_squared=cs2_init,
        mass=mass,
        radius=radius,
        lambda_dimensionless=lambda_diagnostic.lambda_dimensionless,
        surface_energy_density=float(getattr(eos_callable, "eps_surf", 0.0)),
        radius_profile=radius_profile,
        mass_profile=mass_profile,
        lambda_diagnostic=lambda_diagnostic,
    )

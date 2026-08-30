"""First-law reconstruction of accepted CFL sound-speed deformations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import PchipInterpolator

from eos_generation.cfl.baseline import (
    BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV,
    BARYON_DENSITY_SURFACE_FM3,
    CFLAnalyticEos,
    CFLDomainError,
    ENERGY_DENSITY_MAX_MEV_FM3,
    ENERGY_DENSITY_SURFACE_MEV_FM3,
    FROZEN_PARAMETER_SET_ID,
    FROZEN_PARAMETER_SET_SHA256,
    PRESSURE_MAX_MEV_FM3,
    QUARK_CHEMICAL_POTENTIAL_MAX_MEV,
    QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV,
    make_cfl_eos,
)
from eos_generation.cfl.deformation import (
    CFL_DEFORMATION_PROFILE_ID,
    CFL_PRESSURE_PRIMITIVE_POLICY,
    CFL_RAW_GATE_SCHEMA_VERSION,
    CFLWindowedDeformation,
    _canonical_sha256,
    windowed_gaussian_delta_cs2,
    windowed_gaussian_pressure_primitive,
)
from eos_generation.stellar.discontinuities import EosDiscontinuity


CFL_WINDOWED_EOS_SCHEMA_VERSION = "cfl_windowed_eos_v1"
CFL_RECONSTRUCTION_PROFILE_ID = "cfl_surface_anchored_first_law_v1"
DEFAULT_CFL_RECONSTRUCTION_POINTS = 8193


class CFLGeneratedDomainError(CFLDomainError):
    """Raised when a generated CFL evaluator would extrapolate."""


class CFLMechanicalStabilityError(ValueError):
    """Raised when reconstruction is requested for a rejected raw proposal."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        reason = self.report.get("first_failure") or self.report.get("status")
        super().__init__(f"CFL raw proposal is not reconstructable: {reason!r}")


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def _require_domain(
    value: Any,
    *,
    lower: float,
    upper: float,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise CFLGeneratedDomainError(f"{name} must be finite")
    if np.any(array < lower) or np.any(array > upper):
        raise CFLGeneratedDomainError(
            f"{name} is outside the generated non-extrapolating interval "
            f"[{lower!r}, {upper!r}]"
        )
    return array


def _validate_authoritative_raw_report(
    raw_gate_report: Mapping[str, Any] | None,
    deformation: CFLWindowedDeformation,
) -> dict[str, Any]:
    if raw_gate_report is None:
        raise ValueError(
            "build_windowed_eos requires an authoritative accepted full-domain "
            "raw gate report"
        )
    if not isinstance(raw_gate_report, Mapping):
        raise TypeError("raw_gate_report must be a mapping")
    report = dict(raw_gate_report)
    claimed_hash = report.get("report_sha256")
    if not isinstance(claimed_hash, str) or len(claimed_hash) != 64:
        raise ValueError("raw_gate_report has no valid report_sha256")
    hash_payload = dict(report)
    hash_payload.pop("report_sha256", None)
    if _canonical_sha256(hash_payload) != claimed_hash:
        raise ValueError("raw_gate_report deterministic hash does not match its content")
    exact_requirements = {
        "schema_version": CFL_RAW_GATE_SCHEMA_VERSION,
        "profile_id": CFL_DEFORMATION_PROFILE_ID,
        "case_id": deformation.case_id,
        "case_sha256": deformation.case_sha256,
        "baseline_parameter_set_id": FROZEN_PARAMETER_SET_ID,
        "baseline_parameter_set_sha256": FROZEN_PARAMETER_SET_SHA256,
        "parameters": deformation.to_dict(),
        "complete_declared_domain_mev_fm3": [
            ENERGY_DENSITY_SURFACE_MEV_FM3,
            ENERGY_DENSITY_MAX_MEV_FM3,
        ],
        "pressure_primitive_policy": CFL_PRESSURE_PRIMITIVE_POLICY,
    }
    for key, expected in exact_requirements.items():
        if report.get(key) != expected:
            raise ValueError(
                f"raw_gate_report {key!r} does not match the requested CFL case"
            )
    accepted = bool(
        report.get("status") == "accepted_raw_local_physics_gate"
        and report.get("full_declared_domain_passed") is True
        and report.get("amplitude_interval_passed") is True
        and report.get("clipping_clamping_smoothing_posthoc_repair") == "none"
    )
    if not accepted:
        raise CFLMechanicalStabilityError(report)
    surface = report.get("surface")
    if not isinstance(surface, Mapping) or not bool(surface.get("preserved_exactly")):
        raise ValueError("raw_gate_report does not certify the exact CFL surface anchor")
    if (
        surface.get("epsilon_mev_fm3") != ENERGY_DENSITY_SURFACE_MEV_FM3
        or surface.get("pressure_mev_fm3") != 0.0
        or surface.get("deformation_delta_cs2") != 0.0
    ):
        raise ValueError("raw_gate_report surface values disagree with the frozen anchor")
    return report


def _reconstruction_grid_and_baseline(
    baseline: CFLAnalyticEos,
    deformation: CFLWindowedDeformation,
    *,
    grid_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.linspace(
        QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV,
        QUARK_CHEMICAL_POTENTIAL_MAX_MEV,
        grid_points,
    )
    epsilon_mapping = np.asarray(
        baseline.energy_density_from_quark_chemical_potential(mu), dtype=float
    )
    requested = np.asarray(
        [
            deformation.center_mev_fm3,
            ENERGY_DENSITY_SURFACE_MEV_FM3
            + deformation.ramp_width_mev_fm3,
        ],
        dtype=float,
    )
    extra_mu = np.asarray(
        baseline.quark_chemical_potential_from_energy_density(requested),
        dtype=float,
    )
    mu = np.unique(np.concatenate((mu, extra_mu)))
    epsilon = np.asarray(
        baseline.energy_density_from_quark_chemical_potential(mu), dtype=float
    )
    pressure = np.asarray(
        baseline.pressure_from_quark_chemical_potential(mu), dtype=float
    )
    cs2 = np.asarray(
        baseline.sound_speed_squared_from_quark_chemical_potential(mu),
        dtype=float,
    )
    density = np.asarray(
        baseline.baryon_density_from_quark_chemical_potential(mu), dtype=float
    )
    mu_b = np.asarray(
        baseline.baryon_chemical_potential_from_quark_chemical_potential(mu),
        dtype=float,
    )
    # These exact assignments only normalize governed analytic endpoints; no
    # failed raw value is altered.
    epsilon[0] = ENERGY_DENSITY_SURFACE_MEV_FM3
    epsilon[-1] = ENERGY_DENSITY_MAX_MEV_FM3
    pressure[0] = 0.0
    density[0] = BARYON_DENSITY_SURFACE_FM3
    mu_b[0] = BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV
    return epsilon, pressure, cs2, density, mu_b


@dataclass(slots=True)
class CFLWindowedEos:
    """One accepted effective cold CFL barotrope on the governed domain."""

    stellar_sequence_policy = CFLAnalyticEos.stellar_sequence_policy
    stellar_local_refinement_policy = CFLAnalyticEos.stellar_local_refinement_policy
    allow_parallel_tov_sequence = True
    _background_energy_only_is_certified = True

    baseline: CFLAnalyticEos
    deformation: CFLWindowedDeformation
    epsilon: np.ndarray
    pressure: np.ndarray
    cs2: np.ndarray
    baryon_density: np.ndarray
    baryon_chemical_potential: np.ndarray
    raw_gate_report: dict[str, Any]
    diagnostics: dict[str, Any]
    _pressure_inverse: PchipInterpolator = field(init=False, repr=False)
    _log_density_interpolator: PchipInterpolator = field(init=False, repr=False)
    chemical_potential: np.ndarray = field(init=False)
    energy_per_baryon: np.ndarray = field(init=False)
    energy_per_baryon_minus_neutron_rest: np.ndarray = field(init=False)
    adiabatic_index: np.ndarray = field(init=False)
    residuals: dict[str, np.ndarray] = field(init=False)
    discontinuities: tuple[EosDiscontinuity, ...] = field(init=False)
    pressure_min_mev_fm3: float = field(init=False)
    pressure_max_mev_fm3: float = field(init=False)
    energy_density_min_mev_fm3: float = field(init=False)
    energy_density_max_mev_fm3: float = field(init=False)
    eps_surf: float = field(init=False)
    requires_discontinuity_metadata: bool = field(init=False)

    def __post_init__(self) -> None:
        arrays = (
            self.epsilon,
            self.pressure,
            self.cs2,
            self.baryon_density,
            self.baryon_chemical_potential,
        )
        lengths = {len(np.asarray(value)) for value in arrays}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 3:
            raise ValueError("generated CFL state arrays must have matching lengths")
        copied: list[np.ndarray] = []
        for value in arrays:
            array = np.asarray(value, dtype=float).copy()
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError("generated CFL state arrays must be finite and one-dimensional")
            array.setflags(write=False)
            copied.append(array)
        (
            self.epsilon,
            self.pressure,
            self.cs2,
            self.baryon_density,
            self.baryon_chemical_potential,
        ) = copied
        if self.epsilon[0] != ENERGY_DENSITY_SURFACE_MEV_FM3:
            raise ValueError("generated CFL grid does not begin at the frozen surface")
        if self.epsilon[-1] != ENERGY_DENSITY_MAX_MEV_FM3:
            raise ValueError("generated CFL grid does not end at the governed endpoint")
        if self.pressure[0] != 0.0:
            raise ValueError("generated CFL pressure is not exactly zero at the surface")
        if not np.all(np.diff(self.epsilon) > 0.0):
            raise ValueError("generated CFL energy density is not strictly increasing")
        if not np.all(np.diff(self.pressure) > 0.0):
            raise ValueError("generated CFL pressure is not strictly increasing")
        if np.any(self.cs2 <= 0.0) or np.any(self.cs2 > 1.0):
            raise ValueError("generated CFL sound speed violates 0 < c_s^2 <= 1")
        if np.any(self.baryon_density <= 0.0):
            raise ValueError("generated CFL baryon density must be positive")
        if np.any(self.baryon_chemical_potential <= 0.0):
            raise ValueError("generated CFL baryon chemical potential must be positive")
        self.chemical_potential = self.baryon_chemical_potential
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
        enthalpy = (self.epsilon + self.pressure) / self.baryon_density
        dpressure_depsilon = np.gradient(
            self.pressure, self.epsilon, edge_order=2
        )
        dn_depsilon = np.gradient(
            self.baryon_density, self.epsilon, edge_order=2
        )
        pressure_scale = np.maximum.reduce(
            (
                np.abs(self.pressure),
                np.abs(
                    self.baryon_density
                    * self.baryon_chemical_potential
                ),
                np.abs(self.epsilon),
            )
        )
        self.residuals = {
            "r_p_independent_normalized": (
                self.pressure
                - (
                    self.baryon_density
                    * self.baryon_chemical_potential
                    - self.epsilon
                )
            )
            / pressure_scale,
            "r_mu_independent_normalized": (
                self.baryon_chemical_potential - enthalpy
            )
            / self.baryon_chemical_potential,
            "first_law_normalized": (
                1.0
                - self.baryon_chemical_potential * dn_depsilon
            ),
            "r_c": self.cs2 - dpressure_depsilon,
        }
        for array in (
            self.energy_per_baryon,
            self.energy_per_baryon_minus_neutron_rest,
            self.adiabatic_index,
            *self.residuals.values(),
        ):
            array.setflags(write=False)
        self._pressure_inverse = PchipInterpolator(
            self.pressure, self.epsilon, extrapolate=False
        )
        self._log_density_interpolator = PchipInterpolator(
            self.epsilon, np.log(self.baryon_density), extrapolate=False
        )
        self.pressure_min_mev_fm3 = 0.0
        self.pressure_max_mev_fm3 = float(self.pressure[-1])
        self.energy_density_min_mev_fm3 = ENERGY_DENSITY_SURFACE_MEV_FM3
        self.energy_density_max_mev_fm3 = ENERGY_DENSITY_MAX_MEV_FM3
        self.eps_surf = ENERGY_DENSITY_SURFACE_MEV_FM3
        self.requires_discontinuity_metadata = True
        provenance = (
            f"{CFL_DEFORMATION_PROFILE_ID}:case_id={self.deformation.case_id}:"
            f"case_sha256={self.deformation.case_sha256}:"
            f"baseline_sha256={FROZEN_PARAMETER_SET_SHA256}"
        )
        self.discontinuities = (
            EosDiscontinuity.from_sides(
                identifier=(
                    "cfl_bare_self_bound_surface_"
                    f"{self.deformation.case_sha256[:16]}"
                ),
                kind="surface",
                pressure=0.0,
                inner_energy_density=ENERGY_DENSITY_SURFACE_MEV_FM3,
                outer_energy_density=0.0,
                provenance=provenance,
            ),
        )

    @property
    def parameter_set_id(self) -> str:
        return FROZEN_PARAMETER_SET_ID

    @property
    def parameter_set_sha256(self) -> str:
        return FROZEN_PARAMETER_SET_SHA256

    def pressure_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        epsilon = _require_domain(
            energy_density_mev_fm3,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        if self.deformation.amplitude == 0.0:
            return self.baseline.pressure_from_energy_density(epsilon)
        result = np.asarray(
            self.baseline.pressure_from_energy_density(epsilon), dtype=float
        ) + np.asarray(
            windowed_gaussian_pressure_primitive(epsilon, self.deformation),
            dtype=float,
        )
        result = np.where(epsilon == ENERGY_DENSITY_SURFACE_MEV_FM3, 0.0, result)
        return _scalar_or_array(result)

    def energy_density_from_pressure(
        self, pressure_mev_fm3: Any
    ) -> float | np.ndarray:
        pressure = _require_domain(
            pressure_mev_fm3,
            lower=self.pressure_min_mev_fm3,
            upper=self.pressure_max_mev_fm3,
            name="pressure_mev_fm3",
        )
        if self.deformation.amplitude == 0.0:
            return self.baseline.energy_density_from_pressure(pressure)
        result = np.asarray(self._pressure_inverse(pressure), dtype=float)
        if not np.all(np.isfinite(result)):
            raise CFLGeneratedDomainError("generated pressure inverse failed")
        result = np.where(
            pressure == self.pressure_min_mev_fm3,
            self.energy_density_min_mev_fm3,
            result,
        )
        result = np.where(
            pressure == self.pressure_max_mev_fm3,
            self.energy_density_max_mev_fm3,
            result,
        )
        return _scalar_or_array(result)

    def sound_speed_squared_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        epsilon = _require_domain(
            energy_density_mev_fm3,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        if self.deformation.amplitude == 0.0:
            return self.baseline.sound_speed_squared_from_energy_density(epsilon)
        result = np.asarray(
            self.baseline.sound_speed_squared_from_energy_density(epsilon),
            dtype=float,
        ) + np.asarray(
            windowed_gaussian_delta_cs2(epsilon, self.deformation), dtype=float
        )
        return _scalar_or_array(result)

    def sound_speed_squared_from_pressure(
        self, pressure_mev_fm3: Any
    ) -> float | np.ndarray:
        epsilon = self.energy_density_from_pressure(pressure_mev_fm3)
        return self.sound_speed_squared_from_energy_density(epsilon)

    cs2_from_energy_density = sound_speed_squared_from_energy_density
    cs2_from_pressure = sound_speed_squared_from_pressure

    def baryon_density_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        epsilon = _require_domain(
            energy_density_mev_fm3,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        if self.deformation.amplitude == 0.0:
            return self.baseline.baryon_density_from_energy_density(epsilon)
        result = np.exp(self._log_density_interpolator(epsilon))
        result = np.where(
            epsilon == ENERGY_DENSITY_SURFACE_MEV_FM3,
            BARYON_DENSITY_SURFACE_FM3,
            result,
        )
        return _scalar_or_array(np.asarray(result, dtype=float))

    def baryon_density_from_pressure(
        self, pressure_mev_fm3: Any
    ) -> float | np.ndarray:
        return self.baryon_density_from_energy_density(
            self.energy_density_from_pressure(pressure_mev_fm3)
        )

    consistent_baryon_density_from_energy_density = (
        baryon_density_from_energy_density
    )

    def baryon_chemical_potential_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        epsilon = _require_domain(
            energy_density_mev_fm3,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        if self.deformation.amplitude == 0.0:
            return self.baseline.baryon_chemical_potential_from_energy_density(
                epsilon
            )
        pressure = np.asarray(
            self.pressure_from_energy_density(epsilon), dtype=float
        )
        density = np.asarray(
            self.baryon_density_from_energy_density(epsilon), dtype=float
        )
        result = (epsilon + pressure) / density
        result = np.where(
            epsilon == ENERGY_DENSITY_SURFACE_MEV_FM3,
            BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV,
            result,
        )
        return _scalar_or_array(result)

    def baryon_chemical_potential_from_pressure(
        self, pressure_mev_fm3: Any
    ) -> float | np.ndarray:
        return self.baryon_chemical_potential_from_energy_density(
            self.energy_density_from_pressure(pressure_mev_fm3)
        )

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]:
        epsilon = float(self.energy_density_from_pressure(pressure_mev_fm3))
        cs2 = float(self.sound_speed_squared_from_energy_density(epsilon))
        return epsilon, cs2

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CFL_WINDOWED_EOS_SCHEMA_VERSION,
            "reconstruction_profile_id": CFL_RECONSTRUCTION_PROFILE_ID,
            "pressure_primitive_policy": CFL_PRESSURE_PRIMITIVE_POLICY,
            "deformation": self.deformation.to_dict(),
            "baseline_parameter_set_id": FROZEN_PARAMETER_SET_ID,
            "baseline_parameter_set_sha256": FROZEN_PARAMETER_SET_SHA256,
            "raw_gate_report_sha256": self.raw_gate_report["report_sha256"],
            "domain": {
                "energy_density_mev_fm3": [
                    self.energy_density_min_mev_fm3,
                    self.energy_density_max_mev_fm3,
                ],
                "pressure_mev_fm3": [
                    self.pressure_min_mev_fm3,
                    self.pressure_max_mev_fm3,
                ],
                "grid_points": int(len(self.epsilon)),
                "extrapolation": "forbidden",
            },
            "surface": {
                "energy_density_mev_fm3": self.eps_surf,
                "pressure_mev_fm3": 0.0,
                "baryon_density_fm3": BARYON_DENSITY_SURFACE_FM3,
                "baryon_chemical_potential_mev": (
                    BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV
                ),
                "preserved_from_undeformed_baseline": True,
            },
            "requires_discontinuity_metadata": True,
            "discontinuities": [item.to_dict() for item in self.discontinuities],
            "zero_amplitude_analytic_identity": (
                self.deformation.amplitude == 0.0
            ),
            "diagnostics": self.diagnostics,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "baseline": self.baseline.provenance(),
            "effective_one_fluid_barotrope": True,
            "microscopic_composition_status": "not_reconstructed",
            "species_chemical_potential_status": "unavailable",
            "beta_equilibrium_status": "inherited_only_from_frozen_CFL_baseline",
            "sound_speed_clipping": False,
            "posthoc_repair": False,
        }


def build_windowed_eos(
    deformation: CFLWindowedDeformation,
    *,
    raw_gate_report: Mapping[str, Any] | None = None,
    baseline: CFLAnalyticEos | None = None,
    grid_points: int | None = None,
) -> CFLWindowedEos:
    """Reconstruct one already accepted full-domain raw CFL proposal.

    The raw report is intentionally required even at ``A=0`` so no proposal
    can bypass the same assessment checkpoint used by nonzero cases.
    """

    if not isinstance(deformation, CFLWindowedDeformation):
        raise TypeError("deformation must be CFLWindowedDeformation")
    model = baseline or make_cfl_eos()
    resolved_grid_points = (
        int(model.settings.points) if grid_points is None else grid_points
    )
    if (
        not isinstance(resolved_grid_points, int)
        or isinstance(resolved_grid_points, bool)
        or resolved_grid_points < 33
        or resolved_grid_points % 2 == 0
    ):
        raise ValueError("grid_points must be an odd integer of at least 33")
    report = _validate_authoritative_raw_report(raw_gate_report, deformation)
    if deformation.amplitude == 0.0:
        if resolved_grid_points != model.settings.points:
            raise ValueError(
                "A=0 reconstruction grid_points must equal the baseline reporting "
                "grid so exact array identity cannot be weakened"
            )
        epsilon = model.epsilon.copy()
        pressure = model.pressure.copy()
        cs2 = model.cs2.copy()
        density = model.baryon_density.copy()
        mu_b = model.chemical_potential.copy()
        identity_policy = (
            "byte-identical copies of baseline reporting arrays and delegated evaluators"
        )
    else:
        (
            epsilon,
            baseline_pressure,
            baseline_cs2,
            baseline_density,
            baseline_mu_b,
        ) = _reconstruction_grid_and_baseline(
            model, deformation, grid_points=resolved_grid_points
        )
        pressure = baseline_pressure + np.asarray(
            windowed_gaussian_pressure_primitive(epsilon, deformation),
            dtype=float,
        )
        cs2 = baseline_cs2 + np.asarray(
            windowed_gaussian_delta_cs2(epsilon, deformation), dtype=float
        )
        pressure[0] = 0.0
        cs2[0] = baseline_cs2[0]
        integrand = 1.0 / (epsilon + pressure)
        logarithmic_density_change = cumulative_simpson(
            integrand, x=epsilon, initial=0.0
        )
        density = BARYON_DENSITY_SURFACE_FM3 * np.exp(
            logarithmic_density_change
        )
        density[0] = BARYON_DENSITY_SURFACE_FM3
        mu_b = (epsilon + pressure) / density
        mu_b[0] = BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV
        identity_policy = "not_applicable_nonzero_amplitude"
    finite = bool(
        np.all(np.isfinite(pressure))
        and np.all(np.isfinite(cs2))
        and np.all(np.isfinite(density))
        and np.all(np.isfinite(mu_b))
    )
    if not finite:
        raise ValueError("CFL reconstruction produced a nonfinite state")
    if np.any(pressure < 0.0):
        raise ValueError("CFL reconstruction produced negative pressure")
    if not np.all(np.diff(pressure) > 0.0):
        raise ValueError("CFL reconstruction pressure is not strictly increasing")
    if np.any(cs2 <= 0.0) or np.any(cs2 > 1.0):
        raise ValueError("CFL reconstruction contradicts its accepted raw gate")
    if np.any(density <= 0.0) or np.any(mu_b <= 0.0):
        raise ValueError("CFL first-law reconstruction produced nonpositive state")
    euler_residual = pressure - (density * mu_b - epsilon)
    diagnostics = {
        "schema_version": "cfl_reconstruction_diagnostics_v1",
        "case_id": deformation.case_id,
        "case_sha256": deformation.case_sha256,
        "raw_gate_report_sha256": report["report_sha256"],
        "full_domain_raw_gate_preceded_reconstruction": True,
        "surface_anchor": {
            "energy_density_mev_fm3": float(epsilon[0]),
            "pressure_mev_fm3": float(pressure[0]),
            "baryon_density_fm3": float(density[0]),
            "baryon_chemical_potential_mev": float(mu_b[0]),
            "preserved_exactly": bool(
                epsilon[0] == ENERGY_DENSITY_SURFACE_MEV_FM3
                and pressure[0] == 0.0
                and density[0] == BARYON_DENSITY_SURFACE_FM3
                and mu_b[0] == BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV
            ),
        },
        "grid": {
            "requested_quark_chemical_potential_points": resolved_grid_points,
            "actual_points": int(len(epsilon)),
            "coordinate": (
                "exact_baseline_quark_chemical_potential_grid"
                if deformation.amplitude == 0.0
                else "quark_chemical_potential_with_exact_geometry_nodes"
            ),
            "geometry_nodes_inserted": deformation.amplitude != 0.0,
            "first_law_quadrature": (
                "scipy_cumulative_simpson_of_d_epsilon_over_epsilon_plus_P"
            ),
        },
        "pressure_reconstruction": {
            "formula": "P=P_CFL+integral_surface_to_epsilon_A_G_W_d_epsilon",
            "primitive": CFL_PRESSURE_PRIMITIVE_POLICY,
            "clipping_or_repair": "none",
        },
        "baryon_reconstruction": {
            "formula": "n_B=n_surface_exp(integral_d_epsilon_over_epsilon_plus_P)",
            "normalization": "frozen_undeformed_CFL_surface",
        },
        "zero_amplitude_identity_policy": identity_policy,
        "maximum_absolute_euler_residual_mev_fm3": float(
            np.max(np.abs(euler_residual))
        ),
        "endpoint_policy": (
            "fixed_governed_CFL_endpoint; any raw causal violation is rejected, "
            "never truncated or repaired"
        ),
        "below_anchor_support": (
            "W=0 and pressure primitive=0 below the surface anchor; physical "
            "stellar exterior is vacuum, not an EoS continuation"
        ),
    }
    return CFLWindowedEos(
        baseline=model,
        deformation=deformation,
        epsilon=epsilon,
        pressure=pressure,
        cs2=cs2,
        baryon_density=density,
        baryon_chemical_potential=mu_b,
        raw_gate_report=report,
        diagnostics=diagnostics,
    )


__all__ = [
    "CFLGeneratedDomainError",
    "CFLMechanicalStabilityError",
    "CFLWindowedEos",
    "CFL_RECONSTRUCTION_PROFILE_ID",
    "CFL_WINDOWED_EOS_SCHEMA_VERSION",
    "DEFAULT_CFL_RECONSTRUCTION_POINTS",
    "build_windowed_eos",
]

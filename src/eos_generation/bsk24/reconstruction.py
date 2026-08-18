"""Additive BSk24 sound-speed generation with physical cold-barotrope state.

The approved Pearson-et-al. Appendix-C equation (C4) remains authoritative
for pressure.  Appendix C1 supplies only the physical baryon-density
normalization at the approved homogeneous-core anchor.  No legacy hadronic
model, TOV equation, tidal equation, or numerical tolerance is changed here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from eos_generation._internal.config import DEFAULT_CONFIG
from eos_generation.bsk24._reconstruction_diagnostics import (
    local_identity_report as _local_identity_report_impl,
    round_trip_diagnostics as _round_trip_diagnostics_impl,
    summarize_residuals as _summarize_residuals_impl,
)
from eos_generation.bsk24._reconstruction_primitives import (
    ANCHOR_BARYON_DENSITY_FM3,
    APPROVED_CASE_PARAMETERS,
    APPROVED_EPSILON0_MEV_FM3,
    APPROVED_SIGMA_MEV_FM3,
    COMPOSE_CORE_ENTRY_BARYON_DENSITY_FM3,
    COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3,
    COMPOSE_MUON_ONSET_BARYON_DENSITY_FM3,
    COMPOSE_OUTER_INNER_TRANSITION_EPSILON_MEV_FM3,
    _bidirectional_baryon_reconstruction,
    _derived_state,
    _finite_numeric_array,
    _mass_density_from_energy_density,
    _max_residual,
    _profile_grid,
    _scalar_or_array,
    gaussian_pressure_primitive as _gaussian_pressure_primitive_impl,
    gaussian_sound_speed_bump as _gaussian_sound_speed_bump_impl,
)
from eos_generation.bsk24._reconstruction_profiles import (
    _generated_cs2,
    _generated_pressure,
    _residual_arrays,
)
from eos_generation.bsk24.baseline import (
    CAUSAL_MASS_DENSITY_MAX_G_CM3,
    FIT_MASS_DENSITY_MIN_G_CM3,
    MEV_FM3_TO_MASS_DENSITY_G_CM3,
    MODEL_NAME,
    BSk24AnalyticEos,
    make_bsk24_eos,
)


def gaussian_pressure_primitive(
    epsilon: Any,
    *,
    amplitude: float,
    epsilon0: float,
    sigma: float,
    epsilon_ref: float,
) -> float | np.ndarray:
    """Return the unclipped Gaussian pressure primitive."""
    return _gaussian_pressure_primitive_impl(
        epsilon,
        amplitude=amplitude,
        epsilon0=epsilon0,
        sigma=sigma,
        epsilon_ref=epsilon_ref,
    )


def gaussian_sound_speed_bump(
    epsilon: np.ndarray | float,
    *,
    amplitude: float,
    epsilon0: float,
    sigma: float,
) -> np.ndarray | float:
    """Return the unclipped Gaussian sound-speed perturbation."""
    return _gaussian_sound_speed_bump_impl(
        epsilon,
        amplitude=amplitude,
        epsilon0=epsilon0,
        sigma=sigma,
    )


class BSk24GeneratedDomainError(ValueError):
    """Raised when a generated BSk24 evaluation would extrapolate."""


class BSk24MechanicalStabilityError(ValueError):
    """Raised when the unclipped raw proposal contains nonpositive sound speed."""

    def __init__(self, diagnostics: Mapping[str, Any]):
        self.diagnostics = dict(diagnostics)
        super().__init__("raw BSk24 sound-speed proposal is nonpositive")


@dataclass(frozen=True)
class BSk24AnchorState:
    """Approved physical anchor using C1 for normalization and C4 for pressure."""

    baryon_density_fm3: float
    mass_density_g_cm3: float
    energy_density_mev_fm3: float
    pressure_mev_fm3: float
    chemical_potential_mev: float
    source_energy_path: str
    source_pressure_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BSk24GridSettings:
    """Nested profile controls; odd point counts preserve the exact anchor."""

    lower_points: int = 2049
    upper_points: int = 4097
    causal_root_xtol_mev_fm3: float = DEFAULT_CONFIG.hadronic.causal_root_xtol
    causal_root_rtol: float = DEFAULT_CONFIG.hadronic.causal_root_rtol

    def __post_init__(self) -> None:
        for name in ("lower_points", "upper_points"):
            value = int(getattr(self, name))
            if value < 17 or value % 2 == 0:
                raise ValueError(f"{name} must be an odd integer of at least 17")
        if not np.isfinite(
            [self.causal_root_xtol_mev_fm3, self.causal_root_rtol]
        ).all():
            raise ValueError("causal-root tolerances must be finite")
        if self.causal_root_xtol_mev_fm3 <= 0.0 or self.causal_root_rtol <= 0.0:
            raise ValueError("causal-root tolerances must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BSk24Deformation:
    """One deterministic single-Gaussian BSk24 proposal."""

    case_id: str
    amplitude: float
    epsilon0_mev_fm3: float = APPROVED_EPSILON0_MEV_FM3
    sigma_mev_fm3: float = APPROVED_SIGMA_MEV_FM3

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not np.isfinite(
            [self.amplitude, self.epsilon0_mev_fm3, self.sigma_mev_fm3]
        ).all():
            raise ValueError("deformation parameters must be finite")
        if self.sigma_mev_fm3 <= 0.0:
            raise ValueError("sigma must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BSk24ConsistentBaseline:
    """C4 pressure plus bidirectionally reconstructed physical baryon state."""

    eos: BSk24AnalyticEos
    anchor: BSk24AnchorState
    settings: BSk24GridSettings
    epsilon: np.ndarray
    pressure: np.ndarray
    cs2: np.ndarray
    baryon_density: np.ndarray
    chemical_potential: np.ndarray
    adiabatic_index: np.ndarray
    energy_per_baryon_minus_neutron_rest: np.ndarray
    c1_baryon_density: np.ndarray
    c1_relative_discrepancy: np.ndarray
    anchor_index: int
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        self._n_interpolator = PchipInterpolator(
            np.log(self.epsilon), np.log(self.baryon_density), extrapolate=False
        )

    @property
    def energy_density_min_mev_fm3(self) -> float:
        return float(self.epsilon[0])

    @property
    def energy_density_max_mev_fm3(self) -> float:
        return float(self.epsilon[-1])

    def consistent_baryon_density_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        values = _require_domain(
            energy_density_mev_fm3,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        result = np.exp(self._n_interpolator(np.log(values)))
        return _scalar_or_array(result)


@dataclass
class BSk24GeneratedEos:
    """Non-extrapolating generated callable and retained thermodynamic profile."""

    baseline: BSk24ConsistentBaseline
    deformation: BSk24Deformation
    epsilon: np.ndarray
    pressure: np.ndarray
    cs2: np.ndarray
    baryon_density: np.ndarray
    chemical_potential: np.ndarray
    adiabatic_index: np.ndarray
    energy_per_baryon_minus_neutron_rest: np.ndarray
    raw_epsilon: np.ndarray
    raw_pressure: np.ndarray
    raw_cs2: np.ndarray
    residuals: dict[str, np.ndarray]
    diagnostics: dict[str, Any]

    eps_surf: float = 0.0
    requires_discontinuity_metadata: bool = False
    discontinuities: tuple = ()

    def __post_init__(self) -> None:
        self._inverse = PchipInterpolator(
            np.log(self.pressure), np.log(self.epsilon), extrapolate=False
        )
        self._n_interpolator = PchipInterpolator(
            np.log(self.epsilon), np.log(self.baryon_density), extrapolate=False
        )

    @property
    def pressure_min_mev_fm3(self) -> float:
        return float(self.pressure[0])

    @property
    def pressure_max_mev_fm3(self) -> float:
        return float(self.pressure[-1])

    @property
    def energy_density_min_mev_fm3(self) -> float:
        return float(self.epsilon[0])

    @property
    def energy_density_max_mev_fm3(self) -> float:
        return float(self.epsilon[-1])

    @property
    def p_max_causal(self) -> float:
        return self.pressure_max_mev_fm3

    def pressure_from_energy_density(self, energy_density_mev_fm3: Any) -> float | np.ndarray:
        values = _require_domain(
            energy_density_mev_fm3,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        result = _generated_pressure(values, self.baseline, self.deformation)
        return _scalar_or_array(result)

    def sound_speed_squared_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        values = _require_domain(
            energy_density_mev_fm3,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        result = _generated_cs2(values, self.baseline, self.deformation)
        return _scalar_or_array(result)

    def energy_density_from_pressure(self, pressure_mev_fm3: Any) -> float | np.ndarray:
        values = _require_domain(
            pressure_mev_fm3,
            lower=self.pressure_min_mev_fm3,
            upper=self.pressure_max_mev_fm3,
            name="pressure_mev_fm3",
        )
        if self.deformation.amplitude == 0.0:
            # Exact zero deformation is the authoritative C4 barotrope, not a
            # new interpolated representation.  The dormant generated-PCHIP
            # residual is still measured separately by round_trip_diagnostics.
            result = np.asarray(
                self.baseline.eos.energy_density_from_pressure(values), dtype=float
            )
        else:
            result = self._interpolated_energy_density_from_pressure(values)
        # PCHIP is evaluated exactly at declared endpoint knots, but the
        # logarithm/exponential round trip can move the floating result by one
        # ulp.  Snap only those already in-domain endpoint requests; this is
        # not extrapolation and does not alter an interior state.
        result = np.where(values == self.pressure_min_mev_fm3, self.energy_density_min_mev_fm3, result)
        result = np.where(values == self.pressure_max_mev_fm3, self.energy_density_max_mev_fm3, result)
        return _scalar_or_array(result)

    def _interpolated_energy_density_from_pressure(self, pressure_mev_fm3: Any) -> np.ndarray:
        values = np.asarray(pressure_mev_fm3, dtype=float)
        return np.exp(self._inverse(np.log(values)))

    def baryon_density_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
        values = _require_domain(
            energy_density_mev_fm3,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        result = np.exp(self._n_interpolator(np.log(values)))
        return _scalar_or_array(result)

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]:
        pressure = float(pressure_mev_fm3)
        epsilon = float(self.energy_density_from_pressure(pressure))
        cs2 = float(self.sound_speed_squared_from_energy_density(epsilon))
        return epsilon, cs2

    def provenance(self) -> dict[str, Any]:
        return {
            "model_name": f"{MODEL_NAME}_SINGLE_GAUSSIAN_EFFECTIVE_BAROTROPE",
            "source_baseline": self.baseline.eos.provenance(),
            "anchor": self.baseline.anchor.to_dict(),
            "deformation": self.deformation.to_dict(),
            "pressure_authority": "Pearson_2018_Appendix_C4",
            "baryon_normalization_authority": "Pearson_2018_Appendix_C1_at_anchor_only",
            "thermodynamic_state": (
                "C4-consistent bidirectional first-law reconstruction from physical anchor"
            ),
            "microscopic_composition_status": "unavailable",
            "species_chemical_potential_status": "unavailable",
            "beta_equilibrium_status": "unassessed",
            "description": (
                "thermodynamically consistent effective cold barotrope anchored to BSk24"
            ),
            "no_extrapolation": True,
            "sound_speed_clipping": False,
            "diagnostics": self.diagnostics,
        }


def _require_domain(value: Any, *, lower: float, upper: float, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise BSk24GeneratedDomainError(f"{name} must be finite")
    if np.any(array < lower) or np.any(array > upper):
        raise BSk24GeneratedDomainError(
            f"{name} is outside the retained non-extrapolating interval "
            f"[{lower!r}, {upper!r}]"
        )
    return array


def approved_anchor_state(eos: BSk24AnalyticEos | None = None) -> BSk24AnchorState:
    """Return the owner-approved C1/C4 anchor at n_B=0.16 fm^-3."""
    model = eos or make_bsk24_eos()
    n_t = ANCHOR_BARYON_DENSITY_FM3
    # This is the direct Appendix-C1/BSkEofN direction retained by the
    # approved adapter; C4 is evaluated only after epsilon_t is fixed.
    rho_t = float(model._mass_density_from_baryon_density(n_t))
    epsilon_t = rho_t / MEV_FM3_TO_MASS_DENSITY_G_CM3
    pressure_t = float(model.pressure_from_mass_density(rho_t))
    mu_t = (epsilon_t + pressure_t) / n_t
    return BSk24AnchorState(
        baryon_density_fm3=n_t,
        mass_density_g_cm3=rho_t,
        energy_density_mev_fm3=epsilon_t,
        pressure_mev_fm3=pressure_t,
        chemical_potential_mev=mu_t,
        source_energy_path="Pearson_2018_Appendix_C1_BSkEofN",
        source_pressure_path="Pearson_2018_Appendix_C4_at_C1_epsilon_t",
    )


def exploratory_anchor_state_from_energy_density(
    energy_density_mev_fm3: float,
    eos: BSk24AnalyticEos | None = None,
) -> BSk24AnchorState:
    """Derive one C1/C4-consistent exploratory homogeneous-core anchor.

    The caller selects only total energy density.  The remaining state is
    derived from the published BSk24 analytical representations so pressure,
    baryon density, and chemical potential cannot be configured
    independently.
    """

    epsilon_t = float(energy_density_mev_fm3)
    if not math.isfinite(epsilon_t):
        raise ValueError("exploratory anchor energy density must be finite")
    if not (
        COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3
        < epsilon_t
        < CAUSAL_MASS_DENSITY_MAX_G_CM3
        / MEV_FM3_TO_MASS_DENSITY_G_CM3
    ):
        raise ValueError(
            "exploratory anchor energy density must lie strictly inside the "
            "retained homogeneous-core interval"
        )
    model = eos or make_bsk24_eos()
    rho_t = epsilon_t * MEV_FM3_TO_MASS_DENSITY_G_CM3
    n_t = float(model.baryon_density_from_mass_density(rho_t))
    pressure_t = float(model.pressure_from_mass_density(rho_t))
    mu_t = (epsilon_t + pressure_t) / n_t
    if not all(math.isfinite(value) for value in (rho_t, n_t, pressure_t, mu_t)):
        raise ValueError("exploratory anchor derivation produced nonfinite state")
    return BSk24AnchorState(
        baryon_density_fm3=n_t,
        mass_density_g_cm3=rho_t,
        energy_density_mev_fm3=epsilon_t,
        pressure_mev_fm3=pressure_t,
        chemical_potential_mev=mu_t,
        source_energy_path="user_selected_total_energy_density_in_BSk24_C4_domain",
        source_pressure_path="Pearson_2018_Appendix_C4_at_selected_epsilon_match",
    )


def build_consistent_baseline(
    settings: BSk24GridSettings | None = None,
    *,
    eos: BSk24AnalyticEos | None = None,
    anchor_energy_density_mev_fm3: float | None = None,
) -> BSk24ConsistentBaseline:
    """Build the bidirectional C4-consistent baseline state.

    ``None`` preserves the approved n_B=0.16 fm^-3 anchor exactly.  A numeric
    value requests an explicitly exploratory homogeneous-core anchor whose
    complete thermodynamic state is derived by
    :func:`exploratory_anchor_state_from_energy_density`.
    """
    resolved = settings or BSk24GridSettings()
    model = eos or make_bsk24_eos()
    anchor = (
        approved_anchor_state(model)
        if anchor_energy_density_mev_fm3 is None
        else exploratory_anchor_state_from_energy_density(
            anchor_energy_density_mev_fm3,
            model,
        )
    )
    epsilon, anchor_index = _profile_grid(anchor, resolved)
    rho = _mass_density_from_energy_density(epsilon)
    pressure = np.asarray(model.pressure_from_mass_density(rho), dtype=float)
    cs2 = np.asarray(model.sound_speed_squared_from_mass_density(rho), dtype=float)
    n_consistent = _bidirectional_baryon_reconstruction(
        epsilon,
        pressure,
        anchor_index=anchor_index,
        anchor_density_fm3=anchor.baryon_density_fm3,
    )
    c1_density = np.asarray(model.baryon_density_from_mass_density(rho), dtype=float)
    representation = (n_consistent - c1_density) / c1_density
    mu, gamma, energy_per_baryon = _derived_state(
        epsilon, pressure, cs2, n_consistent
    )
    maximum = _max_residual(representation, epsilon)
    diagnostics = {
        "model_identifier": MODEL_NAME,
        "anchor": anchor.to_dict(),
        "anchor_selection": {
            "mode": (
                "standard_n_b_0p16_fm3"
                if anchor_energy_density_mev_fm3 is None
                else "exploratory_selected_epsilon_match"
            ),
            "exploratory": anchor_energy_density_mev_fm3 is not None,
            "independently_configurable_anchor_fields": [
                "energy_density_mev_fm3"
            ],
            "derived_anchor_fields": [
                "baryon_density_fm3",
                "mass_density_g_cm3",
                "pressure_mev_fm3",
                "chemical_potential_mev",
            ],
        },
        "grid": {
            **resolved.to_dict(),
            "total_points": int(len(epsilon)),
            "integration_coordinate": "ln(epsilon)",
            "integrand": "epsilon/(epsilon+P)",
            "quadrature": "scipy.integrate.cumulative_simpson_nonuniform",
            "bidirectional": True,
        },
        "c1_c4_representation_discrepancy": {
            "definition": "(n_B_C4_consistent - n_B_C1) / n_B_C1",
            "classification": "independent_analytical_fit_representation_difference",
            "not_numerical_integration_error": True,
            **maximum,
            "signed_value_at_maximum": float(
                representation[int(np.argmax(np.abs(representation)))]
            ),
            "previous_7p4031e_minus_4_definition": (
                "maximum absolute relative residual above the former n_t=0.12 fm^-3 "
                "anchor through the retained causal endpoint"
            ),
        },
        "phase_and_composition_separation": {
            "compose_core_entry_n_fm3": COMPOSE_CORE_ENTRY_BARYON_DENSITY_FM3,
            "compose_muon_onset_n_fm3": COMPOSE_MUON_ONSET_BARYON_DENSITY_FM3,
            "anchor_minus_core_entry_n_fm3": (
                anchor.baryon_density_fm3 - COMPOSE_CORE_ENTRY_BARYON_DENSITY_FM3
            ),
            "anchor_minus_muon_onset_n_fm3": (
                anchor.baryon_density_fm3 - COMPOSE_MUON_ONSET_BARYON_DENSITY_FM3
            ),
            "anchor_phase": "homogeneous_core_phase_code_0",
        },
    }
    return BSk24ConsistentBaseline(
        eos=model,
        anchor=anchor,
        settings=resolved,
        epsilon=epsilon,
        pressure=pressure,
        cs2=cs2,
        baryon_density=n_consistent,
        chemical_potential=mu,
        adiabatic_index=gamma,
        energy_per_baryon_minus_neutron_rest=energy_per_baryon,
        c1_baryon_density=c1_density,
        c1_relative_discrepancy=representation,
        anchor_index=anchor_index,
        diagnostics=diagnostics,
    )


def summarize_residuals(
    eos: BSk24GeneratedEos,
    *,
    exclude_boundary_points: int = 4,
) -> dict[str, Any]:
    """Summarize algebraic and independent residuals with sensitive regions split."""
    return _summarize_residuals_impl(
        eos,
        exclude_boundary_points=exclude_boundary_points,
    )


def build_generated_eos(
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24Deformation,
) -> BSk24GeneratedEos:
    """Build one raw proposal, reject instability, and retain its causal/source domain."""
    raw_epsilon = baseline.epsilon.copy()
    raw_pressure = _generated_pressure(raw_epsilon, baseline, deformation)
    raw_cs2 = _generated_cs2(raw_epsilon, baseline, deformation)
    minimum_index = int(np.argmin(raw_cs2))
    maximum_index = int(np.argmax(raw_cs2))
    raw_delta = raw_cs2 - baseline.cs2
    support_mask = (
        raw_epsilon >= baseline.anchor.energy_density_mev_fm3
    ) & (
        np.abs(raw_epsilon - deformation.epsilon0_mev_fm3)
        <= 4.0 * deformation.sigma_mev_fm3
    )
    support_indices = np.flatnonzero(support_mask)
    support_minimum = int(support_indices[np.argmin(raw_cs2[support_indices])])
    support_maximum = int(support_indices[np.argmax(raw_cs2[support_indices])])
    center_cs2 = float(
        _generated_cs2(
            np.asarray(deformation.epsilon0_mev_fm3), baseline, deformation
        )
    )
    center_baseline_cs2 = float(
        _generated_cs2(
            np.asarray(deformation.epsilon0_mev_fm3),
            baseline,
            BSk24Deformation(
                case_id="diagnostic_a0",
                amplitude=0.0,
                epsilon0_mev_fm3=deformation.epsilon0_mev_fm3,
                sigma_mev_fm3=deformation.sigma_mev_fm3,
            ),
        )
    )
    mechanical = {
        "raw_minimum_cs2": float(raw_cs2[minimum_index]),
        "raw_minimum_epsilon_mev_fm3": float(raw_epsilon[minimum_index]),
        "raw_maximum_cs2": float(raw_cs2[maximum_index]),
        "raw_maximum_epsilon_mev_fm3": float(raw_epsilon[maximum_index]),
        "support_definition": "anchor-clipped interval epsilon0 +/- 4*sigma",
        "support_minimum_cs2": float(raw_cs2[support_minimum]),
        "support_minimum_epsilon_mev_fm3": float(raw_epsilon[support_minimum]),
        "support_maximum_cs2": float(raw_cs2[support_maximum]),
        "support_maximum_epsilon_mev_fm3": float(raw_epsilon[support_maximum]),
        "raw_cs2_at_deformation_center": center_cs2,
        "baseline_cs2_at_deformation_center": center_baseline_cs2,
        "delta_cs2_at_deformation_center": center_cs2 - center_baseline_cs2,
        "raw_delta_cs2_minimum": float(np.min(raw_delta)),
        "raw_delta_cs2_maximum": float(np.max(raw_delta)),
        "clipping_applied": False,
        "raw_profile_retained": True,
    }
    if not np.all(np.isfinite(raw_cs2)) or np.any(raw_cs2 <= 0.0):
        mechanical["status"] = "rejected_nonpositive_raw_sound_speed"
        raise BSk24MechanicalStabilityError(mechanical)
    mechanical["status"] = "pass_strictly_positive"

    first_superluminal = np.flatnonzero(raw_cs2 > 1.0)
    causal_root: dict[str, Any]
    if len(first_superluminal):
        upper_index = int(first_superluminal[0])
        if upper_index == 0:
            raise ValueError("raw proposal is superluminal at the lower source boundary")
        lower_epsilon = float(raw_epsilon[upper_index - 1])
        upper_epsilon = float(raw_epsilon[upper_index])

        def target(value: float) -> float:
            values = np.asarray(value, dtype=float)
            return float(_generated_cs2(values, baseline, deformation)) - 1.0

        root = brentq(
            target,
            lower_epsilon,
            upper_epsilon,
            xtol=baseline.settings.causal_root_xtol_mev_fm3,
            rtol=baseline.settings.causal_root_rtol,
        )
        retained_epsilon = np.concatenate((raw_epsilon[:upper_index], [root]))
        causal_root = {
            "observed": True,
            "method": "Brent_first_raw_cs2_equals_one_crossing",
            "bracket_epsilon_mev_fm3": [lower_epsilon, upper_epsilon],
            "refined_epsilon_mev_fm3": float(root),
            "residual_from_unity": float(target(root)),
            "raw_points_beyond_retained_endpoint": int(
                np.count_nonzero(raw_epsilon > root)
            ),
            "endpoint_reason": "first_upper_causal_crossing",
        }
    else:
        retained_epsilon = raw_epsilon.copy()
        causal_root = {
            "observed": False,
            "method": "not_applicable",
            "bracket_epsilon_mev_fm3": None,
            "refined_epsilon_mev_fm3": None,
            "residual_from_unity": None,
            "raw_points_beyond_retained_endpoint": 0,
            "endpoint_reason": "approved_BSk24_source_causal_domain_endpoint",
        }

    pressure = _generated_pressure(retained_epsilon, baseline, deformation)
    cs2 = _generated_cs2(retained_epsilon, baseline, deformation)
    if not np.all(np.diff(pressure) > 0.0):
        raise ValueError("generated pressure is not strictly increasing")
    anchor_index = int(np.flatnonzero(retained_epsilon == baseline.anchor.energy_density_mev_fm3)[0])
    if deformation.amplitude == 0.0:
        # The zero Gaussian contributes exactly nothing.  Retain the already
        # approved bidirectional C4-consistent baseline nodes byte-for-byte;
        # independent quadrature refinement is reported separately.
        baryon_density = baseline.baryon_density[: len(retained_epsilon)].copy()
    else:
        baryon_density = baseline.baryon_density[: anchor_index + 1].copy()
        upper_density = _bidirectional_baryon_reconstruction(
            retained_epsilon[anchor_index:],
            pressure[anchor_index:],
            anchor_index=0,
            anchor_density_fm3=baseline.anchor.baryon_density_fm3,
        )
        baryon_density = np.concatenate((baryon_density[:-1], upper_density))
    mu, gamma, energy_per_baryon = _derived_state(
        retained_epsilon, pressure, cs2, baryon_density
    )
    residuals = _residual_arrays(
        retained_epsilon, pressure, cs2, baryon_density, mu
    )

    anchor_pressure_residual = float(pressure[anchor_index] - baseline.anchor.pressure_mev_fm3)
    anchor_density_residual = float(
        baryon_density[anchor_index] - baseline.anchor.baryon_density_fm3
    )
    below = slice(0, anchor_index)
    exact_below = {
        "pressure_array_equal": bool(np.array_equal(pressure[below], baseline.pressure[below])),
        "cs2_array_equal": bool(np.array_equal(cs2[below], baseline.cs2[below])),
        "baryon_density_array_equal": bool(
            np.array_equal(baryon_density[below], baseline.baryon_density[below])
        ),
    }
    diagnostics = {
        "case_id": deformation.case_id,
        "deformation": deformation.to_dict(),
        "anchor": baseline.anchor.to_dict(),
        "anchor_continuity": {
            "pressure_residual_mev_fm3": anchor_pressure_residual,
            "baryon_density_residual_fm3": anchor_density_residual,
            "pressure_exact": bool(anchor_pressure_residual == 0.0),
            "baryon_density_exact": bool(anchor_density_residual == 0.0),
            "gated_raw_cs2_tail_at_anchor": float(
                cs2[anchor_index] - baseline.cs2[anchor_index]
            ),
        },
        "unchanged_below_anchor": exact_below,
        "mechanical_stability": mechanical,
        "causal_domain": {
            **causal_root,
            "retained_epsilon_max_mev_fm3": float(retained_epsilon[-1]),
            "retained_pressure_max_mev_fm3": float(pressure[-1]),
            "retained_cs2_endpoint": float(cs2[-1]),
            "repair_or_clipping": "none",
            "extrapolation": "forbidden",
        },
        "pressure_reconstruction": {
            "formula": "P_theta=P_C4+integral_anchor^epsilon Gaussian d_epsilon",
            "gaussian_primitive": "analytic_error_function_primitive",
            "anchor_offset": "none",
            "strictly_monotone": True,
        },
        "baryon_reconstruction": {
            "below_anchor": "exact_C4_consistent_baseline_state",
            "above_anchor": "cumulative_Simpson_in_ln_epsilon_with_physical_n_t",
            "direct_C1_splice": False,
        },
        "inverse": {
            "method": (
                "direct_authoritative_C4_inverse_for_exact_A0; "
                "PCHIP(log_pressure,log_energy_density)_for_nonzero_cases"
            ),
            "extrapolate": False,
        },
        "microscopic_composition_status": "unavailable",
        "species_chemical_potential_status": "unavailable",
        "beta_equilibrium_status": "unassessed",
    }
    generated = BSk24GeneratedEos(
        baseline=baseline,
        deformation=deformation,
        epsilon=retained_epsilon,
        pressure=pressure,
        cs2=cs2,
        baryon_density=baryon_density,
        chemical_potential=mu,
        adiabatic_index=gamma,
        energy_per_baryon_minus_neutron_rest=energy_per_baryon,
        raw_epsilon=raw_epsilon,
        raw_pressure=raw_pressure,
        raw_cs2=raw_cs2,
        residuals=residuals,
        diagnostics=diagnostics,
    )
    generated.diagnostics["residual_summary"] = summarize_residuals(generated)
    generated.diagnostics["round_trip"] = round_trip_diagnostics(generated)
    return generated


def round_trip_diagnostics(eos: BSk24GeneratedEos) -> dict[str, Any]:
    """Measure forward and inverse residuals at non-node midpoints."""
    return _round_trip_diagnostics_impl(eos)


def local_identity_report(
    baseline: BSk24ConsistentBaseline,
    a0: BSk24GeneratedEos,
) -> dict[str, Any]:
    """Compare direct C4/C4-consistent state with the A=0 generator path."""
    return _local_identity_report_impl(baseline, a0)


def approved_deformations() -> tuple[BSk24Deformation, ...]:
    """Return the exact owner-approved deterministic generator case tuple."""
    return tuple(
        BSk24Deformation(case_id=case_id, amplitude=amplitude)
        for case_id, amplitude in APPROVED_CASE_PARAMETERS.items()
    )


__all__ = [
    "ANCHOR_BARYON_DENSITY_FM3",
    "APPROVED_CASE_PARAMETERS",
    "APPROVED_EPSILON0_MEV_FM3",
    "APPROVED_SIGMA_MEV_FM3",
    "BSk24AnchorState",
    "BSk24ConsistentBaseline",
    "BSk24Deformation",
    "BSk24GeneratedDomainError",
    "BSk24GeneratedEos",
    "BSk24GridSettings",
    "BSk24MechanicalStabilityError",
    "approved_anchor_state",
    "exploratory_anchor_state_from_energy_density",
    "approved_deformations",
    "build_consistent_baseline",
    "build_generated_eos",
    "local_identity_report",
    "round_trip_diagnostics",
    "summarize_residuals",
]

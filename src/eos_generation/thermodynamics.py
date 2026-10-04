"""Thermodynamics for controlled BSk24 / BSk25 experiments."""

from __future__ import annotations

import math
import numpy as np
from .assessment import (
    RETAINED_INTERVALS_PER_SCALE,
    _analytical_pressure_derivative_certificate,
    _retained_geometry_grid,
    raw_local_physics_gate,
)
from .baseline import (
    BSk24AnalyticEos,
    CAUSAL_MASS_DENSITY_MAX_G_CM3,
    COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3,
    FIT_MASS_DENSITY_MIN_G_CM3,
    MEV_FM3_TO_MASS_DENSITY_G_CM3,
    MODEL_NAME,
    NEUTRON_REST_ENERGY_MEV,
    _mass_density_from_energy_density,
    make_bsk24_eos,
)
from .deformation import (
    BSk24WindowedDeformation,
    PURE_GAUSSIAN_GENERATOR_ID,
    WINDOWED_GAUSSIAN_GENERATOR_ID,
    _scalar_or_array,
    _windowed_cs2,
    _windowed_pressure,
    smootherstep_window,
    windowed_gaussian_delta_cs2,
    windowed_gaussian_pressure_primitive,
)
from .numerics import DEFAULT_CONFIG
from dataclasses import asdict, dataclass
from scipy.integrate import cumulative_simpson
from scipy.interpolate import PchipInterpolator
from typing import Any, Mapping


ANCHOR_BARYON_DENSITY_FM3 = 0.16


COMPOSE_CORE_ENTRY_BARYON_DENSITY_FM3 = 0.0807555


COMPOSE_MUON_ONSET_BARYON_DENSITY_FM3 = 0.12577337


COMPOSE_OUTER_INNER_TRANSITION_EPSILON_MEV_FM3 = 0.253215574967


def _finite_numeric_array(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite numeric values") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def _profile_grid(
    anchor: BSk24AnchorState,
    settings: BSk24GridSettings,
    eos: BSk24AnalyticEos | None = None,
) -> tuple[np.ndarray, int]:
    epsilon_min = FIT_MASS_DENSITY_MIN_G_CM3 / MEV_FM3_TO_MASS_DENSITY_G_CM3
    epsilon_max = (eos or make_bsk24_eos()).energy_density_max_causal_mev_fm3
    lower = np.geomspace(
        epsilon_min, anchor.energy_density_mev_fm3, settings.lower_points
    )
    upper = np.linspace(
        anchor.energy_density_mev_fm3, epsilon_max, settings.upper_points
    )
    epsilon = np.concatenate((lower[:-1], upper))
    anchor_index = settings.lower_points - 1
    if epsilon[anchor_index] != anchor.energy_density_mev_fm3:
        raise RuntimeError("profile grid does not retain the exact selected anchor")
    return epsilon, anchor_index


def _bidirectional_baryon_reconstruction(
    epsilon: np.ndarray,
    pressure: np.ndarray,
    *,
    anchor_index: int,
    anchor_density_fm3: float,
) -> np.ndarray:
    """Integrate dln(n)=epsilon/(epsilon+P) dln(epsilon) around the anchor."""
    if not np.all(np.diff(epsilon) > 0.0) or not np.all(pressure > 0.0):
        raise ValueError(
            "thermodynamic reconstruction requires positive monotone inputs"
        )
    logarithmic_integrand = epsilon / (epsilon + pressure)
    cumulative = cumulative_simpson(
        logarithmic_integrand,
        x=np.log(epsilon),
        initial=0.0,
    )
    log_n = math.log(anchor_density_fm3) + cumulative - cumulative[anchor_index]
    baryon_density = np.exp(log_n)
    baryon_density[anchor_index] = anchor_density_fm3
    if not np.all(np.diff(baryon_density) > 0.0):
        raise ValueError("reconstructed baryon density is not strictly increasing")
    return baryon_density


def _derived_state(
    epsilon: np.ndarray,
    pressure: np.ndarray,
    cs2: np.ndarray,
    baryon_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = (epsilon + pressure) / baryon_density
    gamma = ((epsilon + pressure) / pressure) * cs2
    energy_per_baryon = epsilon / baryon_density - NEUTRON_REST_ENERGY_MEV
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(gamma)):
        raise ValueError("derived thermodynamic state contains nonfinite values")
    return mu, gamma, energy_per_baryon


def _max_residual(values: np.ndarray, epsilon: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    index = int(np.argmax(absolute))
    return {
        "maximum_absolute": float(absolute[index]),
        "epsilon_at_maximum_mev_fm3": float(epsilon[index]),
        "p50_absolute": float(np.percentile(absolute, 50.0)),
        "p95_absolute": float(np.percentile(absolute, 95.0)),
        "p99_absolute": float(np.percentile(absolute, 99.0)),
    }


def _residual_arrays(
    epsilon: np.ndarray,
    pressure: np.ndarray,
    cs2: np.ndarray,
    baryon_density: np.ndarray,
    chemical_potential: np.ndarray,
) -> dict[str, np.ndarray]:
    pressure_derivative = PchipInterpolator(
        epsilon, pressure, extrapolate=False
    ).derivative()(epsilon)
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


class BSk24GeneratedDomainError(ValueError):
    """Raised when a generated BSk24 evaluation would extrapolate."""


class BSk24MechanicalStabilityError(ValueError):
    """Raised when the unclipped raw proposal contains nonpositive sound speed."""

    def __init__(self, diagnostics: Mapping[str, Any]):
        self.diagnostics = dict(diagnostics)
        super().__init__("raw sound-speed proposal is nonpositive")


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

    model = eos or make_bsk24_eos()
    epsilon_t = float(energy_density_mev_fm3)
    if not math.isfinite(epsilon_t):
        raise ValueError("exploratory anchor energy density must be finite")
    if not (
        model.definition.core_entry_epsilon_mev_fm3
        < epsilon_t
        < model.energy_density_max_causal_mev_fm3
    ):
        raise ValueError(
            "exploratory anchor energy density must lie strictly inside the "
            "retained homogeneous-core interval"
        )
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
        source_energy_path=f"user_selected_total_energy_density_in_BSk{model.matter_model[-2:]}_C4_domain",
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
    epsilon, anchor_index = _profile_grid(anchor, resolved, model)
    rho = model.mass_density_from_energy_density(epsilon)
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
    mu, gamma, energy_per_baryon = _derived_state(epsilon, pressure, cs2, n_consistent)
    maximum = _max_residual(representation, epsilon)
    diagnostics = {
        "model_identifier": model.model_name,
        "anchor": anchor.to_dict(),
        "anchor_selection": {
            "mode": (
                "standard_n_b_0p16_fm3"
                if anchor_energy_density_mev_fm3 is None
                else "exploratory_selected_epsilon_match"
            ),
            "exploratory": anchor_energy_density_mev_fm3 is not None,
            "independently_configurable_anchor_fields": ["energy_density_mev_fm3"],
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
            "compose_core_entry_n_fm3": model.definition.core_entry_baryon_density_fm3,
            "compose_muon_onset_n_fm3": model.definition.muon_onset_baryon_density_fm3,
            "anchor_minus_core_entry_n_fm3": (
                anchor.baryon_density_fm3 - model.definition.core_entry_baryon_density_fm3
            ),
            "anchor_minus_muon_onset_n_fm3": (
                anchor.baryon_density_fm3 - model.definition.muon_onset_baryon_density_fm3
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


def round_trip_diagnostics(eos: BSk24WindowedEos) -> dict[str, Any]:
    """Measure forward and inverse residuals at non-node midpoints."""
    epsilon_probe = np.sqrt(eos.epsilon[:-1] * eos.epsilon[1:])
    pressure_probe = np.sqrt(eos.pressure[:-1] * eos.pressure[1:])
    forward_pressure = np.asarray(
        eos.pressure_from_energy_density(epsilon_probe), dtype=float
    )
    forward_back = np.asarray(
        eos.energy_density_from_pressure(forward_pressure), dtype=float
    )
    pchip_forward_back = eos._interpolated_energy_density_from_pressure(
        forward_pressure
    )
    inverse_epsilon = np.asarray(
        eos.energy_density_from_pressure(pressure_probe), dtype=float
    )
    inverse_back = np.asarray(
        eos.pressure_from_energy_density(inverse_epsilon), dtype=float
    )
    pchip_inverse_epsilon = eos._interpolated_energy_density_from_pressure(
        pressure_probe
    )
    pchip_inverse_back = np.asarray(
        eos.pressure_from_energy_density(pchip_inverse_epsilon), dtype=float
    )
    forward_abs = forward_back - epsilon_probe
    inverse_abs = inverse_back - pressure_probe
    pchip_forward_abs = pchip_forward_back - epsilon_probe
    pchip_inverse_abs = pchip_inverse_back - pressure_probe
    forward_rel = forward_abs / epsilon_probe
    inverse_rel = inverse_abs / pressure_probe

    def maximum(values: np.ndarray, coordinate: np.ndarray) -> dict[str, float]:
        index = int(np.argmax(np.abs(values)))
        return {
            "maximum_absolute": float(abs(values[index])),
            "signed_value_at_maximum": float(values[index]),
            "coordinate_at_maximum": float(coordinate[index]),
        }

    return {
        "probe_policy": "geometric_midpoints_between_retained_interpolation_nodes",
        "forward_epsilon_to_pressure_to_epsilon_absolute": maximum(
            forward_abs, epsilon_probe
        ),
        "forward_epsilon_to_pressure_to_epsilon_relative": maximum(
            forward_rel, epsilon_probe
        ),
        "inverse_pressure_to_epsilon_to_pressure_absolute": maximum(
            inverse_abs, pressure_probe
        ),
        "inverse_pressure_to_epsilon_to_pressure_relative": maximum(
            inverse_rel, pressure_probe
        ),
        "generated_pchip_forward_relative": maximum(
            pchip_forward_abs / epsilon_probe, epsilon_probe
        ),
        "generated_pchip_inverse_relative": maximum(
            pchip_inverse_abs / pressure_probe, pressure_probe
        ),
        "active_inverse_policy": (
            "authoritative_direct_C4_for_A0"
            if eos.deformation.amplitude == 0.0
            else "nonextrapolating_generated_PCHIP"
        ),
    }


@dataclass
class BSk24WindowedEos:
    """Published-fit-bounded windowed EoS on its first causal branch."""

    baseline: BSk24ConsistentBaseline
    deformation: BSk24WindowedDeformation
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

    def pressure_from_energy_density(self, value: Any) -> float | np.ndarray:
        epsilon = _require_domain(
            value,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        return _scalar_or_array(
            _windowed_pressure(epsilon, self.baseline, self.deformation)
        )

    def sound_speed_squared_from_energy_density(self, value: Any) -> float | np.ndarray:
        epsilon = _require_domain(
            value,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        return _scalar_or_array(_windowed_cs2(epsilon, self.baseline, self.deformation))

    def energy_density_from_pressure(self, value: Any) -> float | np.ndarray:
        pressure = _require_domain(
            value,
            lower=self.pressure_min_mev_fm3,
            upper=self.pressure_max_mev_fm3,
            name="pressure_mev_fm3",
        )
        if self.deformation.amplitude == 0.0:
            result = np.asarray(
                self.baseline.eos.energy_density_from_pressure(pressure),
                dtype=float,
            )
        else:
            result = np.exp(self._inverse(np.log(pressure)))
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

    def baryon_density_from_energy_density(self, value: Any) -> float | np.ndarray:
        epsilon = _require_domain(
            value,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        return _scalar_or_array(np.exp(self._n_interpolator(np.log(epsilon))))

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]:
        epsilon = float(self.energy_density_from_pressure(pressure_mev_fm3))
        return epsilon, float(self.sound_speed_squared_from_energy_density(epsilon))

    def provenance(self) -> dict[str, Any]:
        return {
            "generator_id": WINDOWED_GAUSSIAN_GENERATOR_ID,
            "preserved_existing_generator_id": PURE_GAUSSIAN_GENERATOR_ID,
            "model_name": f"{self.baseline.eos.model_name}_WINDOWED_GAUSSIAN_EFFECTIVE_BAROTROPE",
            "source_baseline": self.baseline.eos.provenance(),
            "anchor": self.baseline.anchor.to_dict(),
            "deformation": self.deformation.to_dict(),
            "window": "quintic_smootherstep_6x5_minus_15x4_plus_10x3",
            "pressure_authority": "Pearson_2018_Appendix_C4_plus_analytic_integral_A_G_W",
            "baryon_normalization_authority": (
                "Pearson_2018_Appendix_C1_at_anchor_only"
            ),
            "thermodynamic_state": (
                "C4-consistent bidirectional first-law reconstruction from physical anchor"
            ),
            "microscopic_composition_status": "unavailable",
            "species_chemical_potential_status": "unavailable",
            "beta_equilibrium_status": "unassessed",
            "no_extrapolation_outside_published_fit_domain": True,
            "sound_speed_clipping": False,
            "diagnostics": self.diagnostics,
        }


def _authoritative_retained_endpoint(
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24WindowedDeformation,
    raw_gate_report: Mapping[str, Any],
) -> tuple[float, bool]:
    """Validate and return one raw-gate-selected retained endpoint."""

    expected_domain = [
        float(baseline.epsilon[0]),
        (
            float(baseline.epsilon[-1])
            if deformation.amplitude == 0.0
            else float(baseline.eos.energy_density_max_published_fit_mev_fm3)
        ),
    ]
    retained = raw_gate_report.get("retained_domain")
    if (
        raw_gate_report.get("status") != "accepted_raw_local_physics_gate"
        or raw_gate_report.get("selected_retained_domain_authoritative") is not True
        or raw_gate_report.get("selected_retained_domain_passed") is not True
        or raw_gate_report.get("case_id") != deformation.case_id
        or raw_gate_report.get("parameters") != deformation.to_dict()
        or raw_gate_report.get("complete_proposed_retained_domain_mev_fm3")
        != expected_domain
        or not isinstance(retained, Mapping)
        or retained.get("policy") != "prefix_through_first_continuous_cs2_equals_one"
        or retained.get("passed") is not True
        or retained.get("resolution_certified") is not True
    ):
        raise ValueError(
            "reconstruction requires matching authoritative first-causal-branch evidence"
        )
    try:
        endpoint = float(retained["epsilon_max_mev_fm3"])
        retained_minimum = float(retained["epsilon_min_mev_fm3"])
        endpoint_pressure = float(retained["pressure_max_mev_fm3"])
        endpoint_cs2 = float(retained["cs2_at_endpoint"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("raw-gate retained endpoint is malformed") from exc
    if (
        not math.isfinite(endpoint)
        or not math.isfinite(retained_minimum)
        or not math.isfinite(endpoint_pressure)
        or not math.isfinite(endpoint_cs2)
        or retained_minimum != expected_domain[0]
        or not baseline.anchor.energy_density_mev_fm3 < endpoint
        or endpoint > expected_domain[1]
        or endpoint_pressure <= 0.0
        or not 0.0 < endpoint_cs2 <= 1.0
    ):
        raise ValueError("raw-gate retained endpoint is outside the deformable domain")
    expected_pressure = float(
        _windowed_pressure(np.asarray([endpoint], dtype=float), baseline, deformation)[
            0
        ]
    )
    expected_cs2 = float(
        _windowed_cs2(np.asarray([endpoint], dtype=float), baseline, deformation)[0]
    )
    comparison_rtol = 64.0 * np.finfo(float).eps
    if not math.isclose(
        endpoint_pressure,
        expected_pressure,
        rel_tol=comparison_rtol,
        abs_tol=comparison_rtol * max(1.0, abs(expected_pressure)),
    ) or not math.isclose(
        endpoint_cs2,
        expected_cs2,
        rel_tol=comparison_rtol,
        abs_tol=comparison_rtol,
    ):
        raise ValueError(
            "raw-gate retained endpoint state disagrees with the analytical proposal"
        )
    crossing = retained.get("first_causal_crossing")
    reason = retained.get("endpoint_reason")
    if reason == f"direct_{baseline.eos.matter_model}_causal_endpoint":
        if (
            deformation.amplitude != 0.0
            or endpoint != expected_domain[1]
            or crossing is not None
            or raw_gate_report.get("full_retained_domain_passed") is not True
            or raw_gate_report.get(
                "complete_raw_proposal_causal_through_direct_endpoint"
            )
            is not True
        ):
            raise ValueError(
                "direct raw-gate endpoint must be the complete selected baseline causal endpoint"
            )
        return endpoint, False
    if reason == f"published_{baseline.eos.matter_model}_fit_endpoint":
        if (
            deformation.amplitude == 0.0
            or endpoint != expected_domain[1]
            or crossing is not None
            or raw_gate_report.get("full_retained_domain_passed") is not True
            or raw_gate_report.get(
                "complete_raw_proposal_causal_through_declared_assessment_endpoint"
            )
            is not True
        ):
            raise ValueError(
                "published-fit raw-gate endpoint must be the complete fit endpoint"
            )
        return endpoint, False
    if reason != "first_continuous_causal_crossing":
        raise ValueError("raw-gate endpoint reason is not recognized")
    if not isinstance(crossing, Mapping):
        raise ValueError("raw-gate causal crossing evidence is missing")
    try:
        crossing_epsilon = float(crossing["epsilon_mev_fm3"])
        crossing_cs2 = float(crossing["cs2_at_endpoint"])
        bracket = tuple(float(value) for value in crossing["bracket_mev_fm3"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("raw-gate causal crossing evidence is malformed") from exc
    representable_width = crossing.get("representable_bracket_width_mev_fm3")
    governed_width = crossing.get("governed_root_tolerance_mev_fm3")
    if representable_width is None and governed_width is None:
        width_evidence_valid = bool(
            crossing.get("endpoint_selection") == "exact_representable_contact"
            and crossing_cs2 == 1.0
            and len(bracket) == 2
            and bracket[1] == endpoint
        )
    else:
        try:
            bracket_width = float(representable_width)
            governed_tolerance = float(governed_width)
        except (TypeError, ValueError):
            width_evidence_valid = False
        else:
            common_width_evidence_valid = bool(
                len(bracket) == 2
                and math.isfinite(bracket_width)
                and math.isfinite(governed_tolerance)
                and bracket_width >= 0.0
                and bracket_width <= governed_tolerance
                and math.isclose(
                    bracket_width,
                    bracket[1] - bracket[0],
                    rel_tol=comparison_rtol,
                    abs_tol=comparison_rtol * max(1.0, abs(endpoint)),
                )
            )
            if common_width_evidence_valid and bracket_width == 0.0:
                width_evidence_valid = bool(
                    bracket[0] == endpoint
                    and bracket[1] == endpoint
                    and expected_cs2 == 1.0
                    and crossing.get("first_noncausal_epsilon_mev_fm3") is None
                    and crossing.get("first_noncausal_cs2") is None
                )
            elif common_width_evidence_valid:
                noncausal_cs2_values = _windowed_cs2(
                    np.asarray([bracket[1]], dtype=float),
                    baseline,
                    deformation,
                )
                analytical_noncausal_cs2 = float(noncausal_cs2_values[0])
                reported_noncausal_epsilon = crossing.get(
                    "first_noncausal_epsilon_mev_fm3"
                )
                reported_noncausal_cs2 = crossing.get("first_noncausal_cs2")
                try:
                    reported_noncausal_epsilon = float(reported_noncausal_epsilon)
                    reported_noncausal_cs2 = float(reported_noncausal_cs2)
                except (TypeError, ValueError):
                    width_evidence_valid = False
                else:
                    width_evidence_valid = bool(
                        bracket[0] == endpoint
                        and bracket[1] > endpoint
                        and math.nextafter(endpoint, bracket[1]) == bracket[1]
                        and expected_cs2 < 1.0
                        and analytical_noncausal_cs2 > 1.0
                        and reported_noncausal_epsilon == bracket[1]
                        and reported_noncausal_cs2 == analytical_noncausal_cs2
                    )
            else:
                width_evidence_valid = False
    if (
        crossing.get("status") != "resolved_first_continuous_causal_crossing"
        or crossing.get("continuous_crossing_bracketed") is not True
        or crossing.get("crossing_included_to_governed_tolerance") is not True
        or crossing.get("cs2_values_modified") is not False
        or raw_gate_report.get("full_retained_domain_passed") is not False
        or raw_gate_report.get(
            "complete_raw_proposal_causal_through_declared_assessment_endpoint"
        )
        is not False
        or crossing_epsilon != endpoint
        or crossing_cs2 != endpoint_cs2
        or len(bracket) != 2
        or not all(math.isfinite(value) for value in bracket)
        or not bracket[0] <= endpoint <= bracket[1]
        or not width_evidence_valid
    ):
        raise ValueError("raw-gate endpoint and first-crossing evidence disagree")
    return endpoint, True


def _retained_resolution_grid(
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24WindowedDeformation,
    *,
    endpoint: float,
    has_causal_crossing: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct and certify a bounded geometry-aware retained grid."""
    return _retained_geometry_grid(
        baseline.epsilon,
        amplitude=deformation.amplitude,
        endpoint_mev_fm3=endpoint,
        has_causal_crossing=has_causal_crossing,
        epsilon0_mev_fm3=deformation.epsilon0_mev_fm3,
        sigma_mev_fm3=deformation.sigma_mev_fm3,
        delta_mev_fm3=deformation.delta_mev_fm3,
        epsilon_match_mev_fm3=(baseline.anchor.energy_density_mev_fm3),
    )


def _analytical_tabulation_certificate(
    epsilon: np.ndarray,
    pressure: np.ndarray,
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24WindowedDeformation,
    spacing_certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the tabulated pressure derivative with analytical ``c_s^2``.

    Finite diagnostic residual magnitudes remain nonblocking elsewhere.  This
    is the narrower hard resolution check: on the declared ramp/support/
    endpoint sections, a midpoint PCHIP derivative must reproduce the
    analytical deformation to the deterministic second-order scale implied by
    the 16-interval-per-feature rule.
    """

    if deformation.amplitude == 0.0:
        return {
            "status": "resolved_exact_baseline_identity_grid",
            "probe_count": 0,
            "criterion": "not_applicable_exact_zero_amplitude",
        }

    def analytical_cs2(value: Any) -> float | np.ndarray:
        result = np.asarray(
            _windowed_cs2(np.asarray(value), baseline, deformation),
            dtype=float,
        )
        return float(result) if result.ndim == 0 else result

    return _analytical_pressure_derivative_certificate(
        epsilon,
        pressure,
        analytical_cs2,
        spacing_certificate,
        intervals_per_scale=RETAINED_INTERVALS_PER_SCALE,
    )


def build_windowed_eos(
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24WindowedDeformation,
    *,
    raw_gate_report: Mapping[str, Any] | None = None,
    require_full_domain: bool = False,
) -> BSk24WindowedEos:
    """Build an accepted windowed EoS without clipping, repair, or extrapolation."""
    from .diagnostics import (
        summarize_windowed_residuals,
        full_domain_thermodynamic_admissibility,
    )

    if raw_gate_report is None:
        raw_gate_report, _gate_epsilon, _gate_cs2 = raw_local_physics_gate(
            baseline, deformation
        )
    if raw_gate_report.get("status") != "accepted_raw_local_physics_gate":
        raise BSk24MechanicalStabilityError(raw_gate_report)
    endpoint_epsilon, has_causal_crossing = _authoritative_retained_endpoint(
        baseline,
        deformation,
        raw_gate_report,
    )
    if require_full_domain and (
        raw_gate_report.get("full_retained_domain_passed") is not True
        or endpoint_epsilon != float(baseline.epsilon[-1])
    ):
        raise ValueError(
            "require_full_domain received a valid but case-truncated raw proposal"
        )

    raw_domain = raw_gate_report["complete_proposed_retained_domain_mev_fm3"]
    raw_upper = float(raw_domain[1])
    if raw_upper == float(baseline.epsilon[-1]):
        raw_epsilon = baseline.epsilon.copy()
    else:
        terminal_spacing = float(baseline.epsilon[-1] - baseline.epsilon[-2])
        extension_intervals = int(
            math.ceil((raw_upper - float(baseline.epsilon[-1])) / terminal_spacing)
        )
        raw_epsilon = np.concatenate(
            (
                baseline.epsilon,
                np.linspace(
                    float(baseline.epsilon[-1]),
                    raw_upper,
                    extension_intervals + 1,
                    dtype=float,
                )[1:],
            )
        )
    raw_pressure = _windowed_pressure(raw_epsilon, baseline, deformation)
    raw_cs2 = _windowed_cs2(raw_epsilon, baseline, deformation)
    if (
        not np.all(np.isfinite(raw_pressure))
        or not np.all(np.isfinite(raw_cs2))
        or np.any(raw_pressure <= 0.0)
        or np.any(raw_cs2 <= 0.0)
    ):
        diagnostics = {
            "case_id": deformation.case_id,
            "status": "rejected_nonfinite_or_nonpositive_raw_core_state",
            "raw_minimum_cs2": float(np.nanmin(raw_cs2)),
            "clipping_applied": False,
            "raw_profile_retained": True,
        }
        raise BSk24MechanicalStabilityError(diagnostics)

    retained_epsilon, tabulation_resolution = _retained_resolution_grid(
        baseline,
        deformation,
        endpoint=endpoint_epsilon,
        has_causal_crossing=has_causal_crossing,
    )
    if tabulation_resolution["status"] not in {
        "resolved_tabulation_resolution",
        "resolved_exact_baseline_identity_grid",
    }:
        raise BSk24MechanicalStabilityError(
            {
                "case_id": deformation.case_id,
                "status": "unresolved_tabulation_resolution",
                "tabulation_resolution": tabulation_resolution,
                "raw_profile_retained": True,
                "reconstruction_available": False,
                "stellar_work_permitted": False,
            }
        )

    pressure = _windowed_pressure(retained_epsilon, baseline, deformation)
    cs2 = _windowed_cs2(retained_epsilon, baseline, deformation)
    if (
        not np.all(np.isfinite(pressure))
        or not np.all(np.isfinite(cs2))
        or np.any(pressure <= 0.0)
        or np.any(cs2 <= 0.0)
        or not np.all(np.diff(pressure) > 0.0)
        or (has_causal_crossing and (np.any(cs2[:-1] >= 1.0) or cs2[-1] > 1.0))
        or (not has_causal_crossing and np.any(cs2 > 1.0))
    ):
        raise BSk24MechanicalStabilityError(
            {
                "case_id": deformation.case_id,
                "status": "rejected_invalid_retained_core_state",
                "tabulation_resolution": tabulation_resolution,
                "clipping_applied": False,
            }
        )
    analytical_resolution = _analytical_tabulation_certificate(
        retained_epsilon,
        pressure,
        baseline,
        deformation,
        tabulation_resolution,
    )
    tabulation_resolution["analytical_comparison"] = analytical_resolution
    if analytical_resolution["status"] not in {
        "resolved_analytical_tabulation",
        "resolved_exact_baseline_identity_grid",
    }:
        tabulation_resolution["status"] = "unresolved_tabulation_resolution"
        tabulation_resolution["failure_reason"] = analytical_resolution.get(
            "failure_reason"
        )
        raise BSk24MechanicalStabilityError(
            {
                "case_id": deformation.case_id,
                "status": "unresolved_tabulation_resolution",
                "tabulation_resolution": tabulation_resolution,
                "raw_profile_retained": True,
                "reconstruction_available": False,
                "stellar_work_permitted": False,
            }
        )
    anchor_index = int(
        np.flatnonzero(retained_epsilon == baseline.anchor.energy_density_mev_fm3)[0]
    )
    if deformation.amplitude == 0.0:
        baryon_density = baseline.baryon_density[: len(retained_epsilon)].copy()
    else:
        upper_density = _bidirectional_baryon_reconstruction(
            retained_epsilon[anchor_index:],
            pressure[anchor_index:],
            anchor_index=0,
            anchor_density_fm3=baseline.anchor.baryon_density_fm3,
        )
        baryon_density = np.concatenate(
            (baseline.baryon_density[:anchor_index], upper_density)
        )
    mu, gamma, energy_per_baryon = _derived_state(
        retained_epsilon, pressure, cs2, baryon_density
    )
    residuals = _residual_arrays(retained_epsilon, pressure, cs2, baryon_density, mu)
    if not residuals or any(
        np.asarray(values).shape != retained_epsilon.shape
        or not np.all(np.isfinite(values))
        for values in residuals.values()
    ):
        raise BSk24MechanicalStabilityError(
            {
                "case_id": deformation.case_id,
                "status": "rejected_nonfinite_reconstruction_or_inversion",
                "tabulation_resolution": tabulation_resolution,
                "finite_diagnostic_magnitudes_are_nonblocking": True,
            }
        )
    below = slice(0, anchor_index)
    ramp_end = baseline.anchor.energy_density_mev_fm3 + deformation.delta_mev_fm3
    diagnostics = {
        "generator_id": WINDOWED_GAUSSIAN_GENERATOR_ID,
        "preserved_existing_generator_id": PURE_GAUSSIAN_GENERATOR_ID,
        "deformation": deformation.to_dict(),
        "raw_gate_report": (
            dict(raw_gate_report) if raw_gate_report is not None else None
        ),
        "tabulation_resolution": tabulation_resolution,
        "unchanged_below_anchor": {
            "pressure_array_equal": bool(
                np.array_equal(pressure[below], baseline.pressure[below])
            ),
            "cs2_array_equal": bool(np.array_equal(cs2[below], baseline.cs2[below])),
            "baryon_density_array_equal": bool(
                np.array_equal(baryon_density[below], baseline.baryon_density[below])
            ),
        },
        "anchor_continuity": {
            "delta_cs2_exact_zero": bool(
                float(
                    windowed_gaussian_delta_cs2(
                        baseline.anchor.energy_density_mev_fm3,
                        deformation,
                        epsilon_t_mev_fm3=baseline.anchor.energy_density_mev_fm3,
                    )
                )
                == 0.0
            ),
            "pressure_primitive_exact_zero": bool(
                float(
                    windowed_gaussian_pressure_primitive(
                        baseline.anchor.energy_density_mev_fm3,
                        deformation,
                        epsilon_t_mev_fm3=baseline.anchor.energy_density_mev_fm3,
                    )
                )
                == 0.0
            ),
            "pressure_residual_mev_fm3": float(
                pressure[anchor_index] - baseline.anchor.pressure_mev_fm3
            ),
            "baryon_density_residual_fm3": float(
                baryon_density[anchor_index] - baseline.anchor.baryon_density_fm3
            ),
            "chemical_potential_residual_mev": float(
                mu[anchor_index] - baseline.anchor.chemical_potential_mev
            ),
        },
        "ramp_endpoint": {
            "epsilon_mev_fm3": ramp_end,
            "inside_retained_domain": bool(ramp_end <= retained_epsilon[-1]),
            "window_exact_one": bool(
                float(
                    smootherstep_window(
                        ramp_end,
                        epsilon_t_mev_fm3=baseline.anchor.energy_density_mev_fm3,
                        delta_mev_fm3=deformation.delta_mev_fm3,
                    )
                )
                == 1.0
            ),
            "above_ramp_equals_ordinary_gaussian": True,
        },
        "pressure_reconstruction": {
            "formula": "P_A=P_C4+integral_anchor^epsilon A*G*W d_epsilon",
            "primitive": (
                "analytic Gaussian moments through polynomial ramp plus error-function Gaussian tail"
            ),
            "cumulative_sum_used": False,
        },
        "baryon_reconstruction": {
            "below_anchor": "exact C4-consistent reconstruction, C1-normalized at anchor",
            "above_anchor": "cumulative Simpson in ln(epsilon) with same physical anchor",
            "direct_C1_splice": False,
        },
        "causal_domain": {
            "observed_upper_causal_crossing": has_causal_crossing,
            "endpoint_reason": raw_gate_report["retained_domain"]["endpoint_reason"],
            "refined_epsilon_mev_fm3": (
                endpoint_epsilon if has_causal_crossing else None
            ),
            "raw_gate_endpoint_consumed_without_rediscovery": True,
            "retained_epsilon_max_mev_fm3": float(retained_epsilon[-1]),
            "retained_pressure_max_mev_fm3": float(pressure[-1]),
            "retained_cs2_endpoint": float(cs2[-1]),
            "repair_or_clipping": "none",
            "extrapolation": "forbidden_outside_published_fit_domain",
        },
        "microscopic_composition_status": "unavailable",
        "species_chemical_potential_status": "unavailable",
        "beta_equilibrium_status": "unassessed",
    }
    try:
        eos = BSk24WindowedEos(
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
        pressure_probe = np.sqrt(pressure[:-1] * pressure[1:])
        recovered = np.asarray(
            eos.energy_density_from_pressure(pressure_probe), dtype=float
        )
        forward = np.asarray(eos.pressure_from_energy_density(recovered), dtype=float)
        inversion_usable = bool(
            np.all(np.isfinite(recovered))
            and np.all(np.isfinite(forward))
            and np.all(recovered > retained_epsilon[:-1])
            and np.all(recovered < retained_epsilon[1:])
            and np.all(np.diff(recovered) > 0.0)
        )
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise BSk24MechanicalStabilityError(
            {
                "case_id": deformation.case_id,
                "status": "rejected_unusable_reconstruction_interpolation",
                "reason": f"{type(exc).__name__}:{exc}",
                "tabulation_resolution": tabulation_resolution,
            }
        ) from exc
    if not inversion_usable:
        raise BSk24MechanicalStabilityError(
            {
                "case_id": deformation.case_id,
                "status": "rejected_unusable_reconstruction_inversion",
                "tabulation_resolution": tabulation_resolution,
            }
        )
    eos.diagnostics["tabulation_resolution"][
        "interpolation_inversion_status"
    ] = "resolved_finite_monotone_nonextrapolating"
    eos.diagnostics["residual_summary"] = summarize_windowed_residuals(eos)
    admissibility = full_domain_thermodynamic_admissibility(
        baseline,
        eos,
        raw_gate_report=raw_gate_report,
    )
    eos.diagnostics["retained_domain_thermodynamic_admissibility"] = admissibility
    accepted_admissibility_statuses = {
        "accepted_full_domain_thermodynamic_gate",
        "accepted_selected_domain_thermodynamic_gate",
    }
    if admissibility["status"] not in accepted_admissibility_statuses:
        raise BSk24MechanicalStabilityError(admissibility)
    if require_full_domain:
        # Preserve the established diagnostic key and status for callers that
        # explicitly request direct-endpoint compatibility.
        eos.diagnostics["full_domain_thermodynamic_admissibility"] = admissibility
        if admissibility["status"] != "accepted_full_domain_thermodynamic_gate":
            raise BSk24MechanicalStabilityError(admissibility)
    return eos

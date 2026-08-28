"""Smooth-window BSk24 sound-speed deformation.

``windowed_gaussian_v1`` multiplies the Gaussian deformation by the approved
quintic smootherstep before integrating it into pressure.

The authoritative BSk24 C4 pressure, C1-normalized physical anchor, causal
production domain, and first-law reconstruction are reused without changing
their scientific conventions.  No sound-speed proposal is clipped or repaired.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator

from eos_generation.bsk24.baseline import MODEL_NAME
from eos_generation.bsk24._deformation_bounds import (
    RAW_DISCOVERY_INTERVALS_PER_SCALE,
    RETAINED_INTERVALS_PER_SCALE,
    _analytical_pressure_derivative_certificate,
    _continuous_local_minima,
    _geometry_aware_grid,
    _log_windowed_gaussian_shape_scalar,
    _meaningful_support_interval,
    _retained_geometry_grid,
)
from eos_generation.bsk24._deformation_core import (
    BSK24_RETAINED_EPSILON_MATCH_MEV_FM3,
    BSK24_RETAINED_EPSILON_MAX_MEV_FM3,
    CONTROL_DELTAS_MEV_FM3,
    PRIMARY_AMPLITUDES,
    PRIMARY_DELTA_MEV_FM3,
    PRIMARY_EPSILON0_MEV_FM3,
    PRIMARY_SIGMA_MEV_FM3,
    PURE_GAUSSIAN_GENERATOR_ID,
    WINDOWED_GAUSSIAN_GENERATOR_ID,
    _normal_moment_integrals,
    _ramp_gaussian_integral,
    _scalar_or_array,
    _shifted_gaussian_moment,
    _windowed_cs2,
    _windowed_pressure,
    gaussian_profile as _gaussian_profile_impl,
    smootherstep_window as _smootherstep_window_impl,
    smootherstep_window_first_derivative as _smootherstep_window_first_derivative_impl,
    smootherstep_window_second_derivative as _smootherstep_window_second_derivative_impl,
    windowed_gaussian_delta_cs2 as _windowed_gaussian_delta_cs2_impl,
    windowed_gaussian_pressure_primitive as _windowed_gaussian_pressure_primitive_impl,
    windowed_gaussian_shape as _windowed_gaussian_shape_impl,
)
from eos_generation.bsk24._deformation_diagnostics import (
    _WINDOW_CHARACTERIZATION_CACHE,
    _window_characterization_uncached,
    full_domain_thermodynamic_admissibility as _full_domain_thermodynamic_admissibility_impl,
    summarize_windowed_residuals as _summarize_windowed_residuals_impl,
    window_characterization as _window_characterization_impl,
    windowed_a0_identity_report as _windowed_a0_identity_report_impl,
)
from eos_generation.bsk24._deformation_gate import (
    _RAW_GATE_BASELINE_CACHE,
    _cached_raw_gate_baseline_arrays,
    _dense_gate_grid,
    _failure_region,
    _refined_extremum,
    raw_local_physics_gate as _raw_local_physics_gate_impl,
)
from eos_generation.bsk24.reconstruction import (
    COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3,
    BSk24ConsistentBaseline,
    BSk24GeneratedDomainError,
    BSk24MechanicalStabilityError,
    _bidirectional_baryon_reconstruction,
    _derived_state,
    _mass_density_from_energy_density,
    _require_domain,
    _residual_arrays,
)


@dataclass(frozen=True)
class BSk24WindowedDeformation:
    """One deterministic smootherstep-windowed Gaussian proposal."""

    case_id: str
    amplitude: float
    epsilon0_mev_fm3: float = PRIMARY_EPSILON0_MEV_FM3
    sigma_mev_fm3: float = PRIMARY_SIGMA_MEV_FM3
    delta_mev_fm3: float = PRIMARY_DELTA_MEV_FM3

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        values = (
            self.amplitude,
            self.epsilon0_mev_fm3,
            self.sigma_mev_fm3,
            self.delta_mev_fm3,
        )
        if not np.isfinite(values).all():
            raise ValueError("windowed deformation parameters must be finite")
        if self.sigma_mev_fm3 <= 0.0:
            raise ValueError("sigma must be positive")
        if self.delta_mev_fm3 <= 0.0:
            raise ValueError("Delta must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "generator_id": WINDOWED_GAUSSIAN_GENERATOR_ID,
        }


@dataclass(frozen=True)
class BSk24AmplitudeBounds:
    """Exact full-direct-domain amplitude interval for one geometry.

    The lower endpoint is open because the smooth deformation requires
    ``c_s^2 > 0``.  The upper endpoint is closed because ``c_s^2 = 1`` is
    causal through the direct endpoint.  A larger positive amplitude is not
    automatically invalid: it requires a case-specific first causal endpoint.
    Locations and candidate extrema are retained in deterministic increasing-
    energy order for scientific provenance.
    """

    epsilon0_mev_fm3: float
    sigma_mev_fm3: float
    delta_mev_fm3: float
    epsilon_match_mev_fm3: float
    epsilon_max_mev_fm3: float
    amplitude_min: float
    amplitude_max: float
    lower_limiting_epsilon_mev_fm3: float
    upper_limiting_epsilon_mev_fm3: float
    lower_limiting_baseline_cs2: float
    upper_limiting_baseline_cs2: float
    lower_limiting_shape: float
    upper_limiting_shape: float
    baseline_minimum_cs2: float
    baseline_minimum_epsilon_mev_fm3: float
    baseline_maximum_cs2: float
    baseline_maximum_epsilon_mev_fm3: float
    lower_candidate_extrema_mev_fm3: tuple[float, ...]
    upper_candidate_extrema_mev_fm3: tuple[float, ...]
    discovery_grid_points: int

    @property
    def lower_endpoint_open(self) -> bool:
        return True

    @property
    def upper_endpoint_closed(self) -> bool:
        return True

    def contains(self, amplitude: float) -> bool:
        """Return the exact full-direct-domain interval predicate."""

        try:
            value = float(amplitude)
        except (TypeError, ValueError):
            return False
        return bool(
            math.isfinite(value)
            and value > self.amplitude_min
            and value <= self.amplitude_max
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": "bsk24_windowed_amplitude_bounds_v1",
            "geometry": {
                "epsilon0_mev_fm3": self.epsilon0_mev_fm3,
                "sigma_mev_fm3": self.sigma_mev_fm3,
                "delta_mev_fm3": self.delta_mev_fm3,
            },
            "retained_domain_mev_fm3": [
                self.epsilon_match_mev_fm3,
                self.epsilon_max_mev_fm3,
            ],
            "policy": "full_direct_domain_diagnostic_not_case_acceptance",
            "amplitude_interval": {
                "A_min": self.amplitude_min,
                "A_max": self.amplitude_max,
                "notation": "(A_min, A_max]",
                "lower_endpoint_open": True,
                "upper_endpoint_closed": True,
                "width": self.amplitude_max - self.amplitude_min,
                "zero_to_lower_endpoint_margin": -self.amplitude_min,
                "zero_to_upper_endpoint_margin": self.amplitude_max,
            },
            "lower_limit": {
                "condition": "c_s_squared=0",
                "epsilon_mev_fm3": self.lower_limiting_epsilon_mev_fm3,
                "baseline_cs2": self.lower_limiting_baseline_cs2,
                "windowed_gaussian_shape": self.lower_limiting_shape,
            },
            "upper_limit": {
                "condition": "c_s_squared=1",
                "epsilon_mev_fm3": self.upper_limiting_epsilon_mev_fm3,
                "baseline_cs2": self.upper_limiting_baseline_cs2,
                "windowed_gaussian_shape": self.upper_limiting_shape,
            },
            "baseline_margins": {
                "minimum_cs2": self.baseline_minimum_cs2,
                "minimum_epsilon_mev_fm3": (
                    self.baseline_minimum_epsilon_mev_fm3
                ),
                "mechanical_stability_margin": self.baseline_minimum_cs2,
                "maximum_cs2": self.baseline_maximum_cs2,
                "maximum_epsilon_mev_fm3": (
                    self.baseline_maximum_epsilon_mev_fm3
                ),
                "causality_margin": 1.0 - self.baseline_maximum_cs2,
            },
            "continuous_extremum_search": {
                "policy": (
                    "complete retained-domain discovery including declared "
                    "boundaries followed by bounded continuous refinement"
                ),
                "discovery_grid_points": self.discovery_grid_points,
                "lower_candidate_extrema_mev_fm3": list(
                    self.lower_candidate_extrema_mev_fm3
                ),
                "upper_candidate_extrema_mev_fm3": list(
                    self.upper_candidate_extrema_mev_fm3
                ),
                "gaussian_tail_evaluation": "log_domain",
                "f_equals_zero_policy": "no_amplitude_bound",
            },
        }


def smootherstep_window(
    energy_density_mev_fm3: Any,
    *,
    epsilon_t_mev_fm3: float,
    delta_mev_fm3: float,
) -> float | np.ndarray:
    """Return the exact piecewise quintic smootherstep window."""
    return _smootherstep_window_impl(
        energy_density_mev_fm3,
        epsilon_t_mev_fm3=epsilon_t_mev_fm3,
        delta_mev_fm3=delta_mev_fm3,
    )


def smootherstep_window_first_derivative(
    energy_density_mev_fm3: Any,
    *,
    epsilon_t_mev_fm3: float,
    delta_mev_fm3: float,
) -> float | np.ndarray:
    """Return dW/d-epsilon for the exact piecewise smootherstep."""
    return _smootherstep_window_first_derivative_impl(
        energy_density_mev_fm3,
        epsilon_t_mev_fm3=epsilon_t_mev_fm3,
        delta_mev_fm3=delta_mev_fm3,
    )


def smootherstep_window_second_derivative(
    energy_density_mev_fm3: Any,
    *,
    epsilon_t_mev_fm3: float,
    delta_mev_fm3: float,
) -> float | np.ndarray:
    """Return d2W/d-epsilon2 for the exact piecewise smootherstep."""
    return _smootherstep_window_second_derivative_impl(
        energy_density_mev_fm3,
        epsilon_t_mev_fm3=epsilon_t_mev_fm3,
        delta_mev_fm3=delta_mev_fm3,
    )


def gaussian_profile(
    energy_density_mev_fm3: Any,
    deformation: BSk24WindowedDeformation,
) -> float | np.ndarray:
    """Return the nominal unit-amplitude Gaussian G."""
    return _gaussian_profile_impl(energy_density_mev_fm3, deformation)


def windowed_gaussian_shape(
    energy_density_mev_fm3: Any,
    deformation: BSk24WindowedDeformation,
    *,
    epsilon_t_mev_fm3: float,
) -> float | np.ndarray:
    """Return G*W without the nominal amplitude."""
    return _windowed_gaussian_shape_impl(
        energy_density_mev_fm3,
        deformation,
        epsilon_t_mev_fm3=epsilon_t_mev_fm3,
    )


def windowed_gaussian_delta_cs2(
    energy_density_mev_fm3: Any,
    deformation: BSk24WindowedDeformation,
    *,
    epsilon_t_mev_fm3: float,
) -> float | np.ndarray:
    """Return the raw additive deformation A*G*W."""
    return _windowed_gaussian_delta_cs2_impl(
        energy_density_mev_fm3,
        deformation,
        epsilon_t_mev_fm3=epsilon_t_mev_fm3,
    )


def calculate_windowed_amplitude_bounds(
    baseline: BSk24ConsistentBaseline,
    *,
    epsilon0_mev_fm3: float,
    sigma_mev_fm3: float,
    delta_mev_fm3: float,
    discovery_points: int = 32769,
) -> BSk24AmplitudeBounds:
    """Return full-direct-domain bounds for one raw-amplitude geometry.

    The match must lie in the retained homogeneous-core domain, while
    ``epsilon0``, ``sigma``, and ``Delta`` must be finite and positive.  The
    returned open-lower/closed-upper interval is set by the continuous raw
    conditions ``0 < c_s^2 <= 1`` through the *direct* BSk24 endpoint.  Its
    lower limit remains a hard mechanical-stability bound.  Exceeding its
    upper limit instead requires a case-specific first causal endpoint and is
    not, by itself, a rejection under the retained-branch policy.

    Pressure introduces no additional bound here: the deformation is
    integrated from a positive-pressure match and every admitted proposal has
    a strictly positive pressure derivative over the affected domain.
    """

    return _calculate_windowed_amplitude_bounds(
        baseline,
        epsilon0_mev_fm3=epsilon0_mev_fm3,
        sigma_mev_fm3=sigma_mev_fm3,
        delta_mev_fm3=delta_mev_fm3,
        discovery_points=discovery_points,
    )


def _calculate_windowed_amplitude_bounds(
    baseline: BSk24ConsistentBaseline,
    *,
    epsilon0_mev_fm3: float,
    sigma_mev_fm3: float,
    delta_mev_fm3: float,
    discovery_points: int,
) -> BSk24AmplitudeBounds:
    """Return continuous raw-amplitude bounds ``(A_min, A_max]``.

    Ratios are minimized in log space, so very small nonzero Gaussian tails
    cannot overflow a division.  The exact ``f=0`` region at and below the
    match contributes no amplitude constraint, while the baseline remains
    subject to its own complete-domain physical gate.
    """

    if not isinstance(discovery_points, int) or isinstance(discovery_points, bool):
        raise ValueError("discovery_points must be an odd integer of at least 257")
    if discovery_points < 257 or discovery_points % 2 == 0:
        raise ValueError("discovery_points must be an odd integer of at least 257")
    epsilon_match = float(baseline.anchor.energy_density_mev_fm3)
    epsilon_max = float(baseline.epsilon[-1])
    retained_endpoint_matches = math.isclose(
        epsilon_max,
        BSK24_RETAINED_EPSILON_MAX_MEV_FM3,
        rel_tol=0.0,
        abs_tol=5.0e-12,
    )
    valid_anchor = bool(
        COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3
        < epsilon_match
        < BSK24_RETAINED_EPSILON_MAX_MEV_FM3
    )
    if not valid_anchor or not retained_endpoint_matches:
        raise ValueError(
            "amplitude bounds require the declared authoritative retained BSk24 domain"
        )
    raw_geometry = np.asarray(
        [epsilon0_mev_fm3, sigma_mev_fm3, delta_mev_fm3], dtype=float
    )
    if not np.all(np.isfinite(raw_geometry)) or np.any(raw_geometry <= 0.0):
        raise ValueError(
            "ordinary raw-amplitude geometry requires finite positive "
            "epsilon0, sigma, and Delta"
        )
    if _meaningful_support_interval(
        epsilon0_mev_fm3=float(epsilon0_mev_fm3),
        sigma_mev_fm3=float(sigma_mev_fm3),
        epsilon_match_mev_fm3=epsilon_match,
        epsilon_max_mev_fm3=epsilon_max,
    ) is None:
        raise ValueError(
            "ordinary raw-amplitude geometry has no meaningful in-domain "
            "four-sigma support"
        )
    arrays = tuple(
        np.asarray(getattr(baseline, name), dtype=float)
        for name in (
            "epsilon",
            "pressure",
            "cs2",
            "baryon_density",
            "chemical_potential",
        )
    )
    if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
        raise ValueError("baseline state arrays must be aligned one-dimensional profiles")
    (
        epsilon_nodes,
        pressure_nodes,
        cs2_nodes,
        baryon_density_nodes,
        chemical_potential_nodes,
    ) = arrays
    if (
        not np.all(np.isfinite(epsilon_nodes))
        or not np.all(np.isfinite(pressure_nodes))
        or not np.all(np.isfinite(cs2_nodes))
        or np.any(epsilon_nodes <= 0.0)
        or np.any(pressure_nodes <= 0.0)
        or np.any(cs2_nodes <= 0.0)
        or np.any(cs2_nodes > 1.0)
        or not np.all(np.diff(epsilon_nodes) > 0.0)
        or not np.all(np.diff(pressure_nodes) > 0.0)
        or np.any(baryon_density_nodes <= 0.0)
        or not np.all(np.diff(baryon_density_nodes) > 0.0)
        or np.any(chemical_potential_nodes <= 0.0)
    ):
        raise ValueError("baseline state fails the retained-domain physical gate")

    full_grid = _dense_gate_grid(
        baseline,
        lower_points=max(257, (discovery_points + 1) // 4 * 2 + 1),
        upper_points=discovery_points,
    )

    def baseline_cs2(value: float) -> float:
        rho = _mass_density_from_energy_density(np.asarray(value, dtype=float))
        return float(baseline.eos.sound_speed_squared_from_mass_density(rho))

    baseline_values = np.asarray(
        baseline.eos.sound_speed_squared_from_mass_density(
            _mass_density_from_energy_density(full_grid)
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(baseline_values)):
        raise ValueError("baseline continuous sound speed is non-finite")
    baseline_min, baseline_min_epsilon = _refined_extremum(
        full_grid,
        baseline_values,
        baseline_cs2,
        maximize=False,
    )
    baseline_max, baseline_max_epsilon = _refined_extremum(
        full_grid,
        baseline_values,
        baseline_cs2,
        maximize=True,
    )
    if baseline_min <= 0.0 or baseline_max > 1.0:
        raise ValueError("baseline continuous sound speed fails 0 < c_s^2 <= 1")

    ramp_end = epsilon_match + float(delta_mev_fm3)
    upper_grid = np.linspace(epsilon_match, epsilon_max, discovery_points)
    upper_grid, geometry_certificate = _geometry_aware_grid(
        upper_grid,
        epsilon0_mev_fm3=float(epsilon0_mev_fm3),
        sigma_mev_fm3=float(sigma_mev_fm3),
        delta_mev_fm3=float(delta_mev_fm3),
        epsilon_match_mev_fm3=epsilon_match,
        epsilon_max_mev_fm3=epsilon_max,
        intervals_per_scale=RAW_DISCOVERY_INTERVALS_PER_SCALE,
    )
    if geometry_certificate.get("status") != "resolved_geometry_aware_sampling":
        raise ValueError(
            "amplitude-bound geometry resolution is unresolved: "
            f"{geometry_certificate.get('failure_reason')}"
        )
    upper_grid = np.unique(
        np.concatenate(
            (
                upper_grid,
                np.asarray(
                    [
                        np.nextafter(epsilon_match, epsilon_max),
                        ramp_end,
                        float(epsilon0_mev_fm3),
                        epsilon_max,
                    ]
                ),
            )
        )
    )
    upper_grid = upper_grid[
        (upper_grid > epsilon_match) & (upper_grid <= epsilon_max)
    ]

    def log_shape(value: float) -> float:
        return _log_windowed_gaussian_shape_scalar(
            value,
            epsilon0_mev_fm3=float(epsilon0_mev_fm3),
            sigma_mev_fm3=float(sigma_mev_fm3),
            delta_mev_fm3=float(delta_mev_fm3),
            epsilon_match_mev_fm3=epsilon_match,
        )

    def lower_log_ratio(value: float) -> float:
        cs2 = baseline_cs2(value)
        shape_log = log_shape(value)
        if not 0.0 < cs2 <= 1.0 or not math.isfinite(shape_log):
            return math.inf
        return math.log(cs2) - shape_log

    def upper_log_ratio(value: float) -> float:
        cs2 = baseline_cs2(value)
        shape_log = log_shape(value)
        if not 0.0 < cs2 < 1.0 or not math.isfinite(shape_log):
            if cs2 == 1.0 and math.isfinite(shape_log):
                return -math.inf
            return math.inf
        return math.log1p(-cs2) - shape_log

    upper_cs2_values = np.asarray(
        baseline.eos.sound_speed_squared_from_mass_density(
            _mass_density_from_energy_density(upper_grid)
        ),
        dtype=float,
    )
    upper_log_shape = np.asarray([log_shape(value) for value in upper_grid])
    lower_sampled = np.log(upper_cs2_values) - upper_log_shape
    with np.errstate(divide="ignore", invalid="ignore"):
        upper_sampled = np.log1p(-upper_cs2_values) - upper_log_shape
    if not np.all(np.isfinite(lower_sampled)):
        raise ValueError("lower amplitude-bound objective is non-finite")
    if np.any(np.isnan(upper_sampled)) or np.any(np.isposinf(upper_sampled)):
        raise ValueError("upper amplitude-bound objective is invalid")
    mandatory = (ramp_end, float(epsilon0_mev_fm3), epsilon_max)
    lower_candidates = _continuous_local_minima(
        upper_grid,
        lower_sampled,
        lower_log_ratio,
        mandatory_points=mandatory,
    )
    if np.any(np.isneginf(upper_sampled)):
        equality_indices = np.flatnonzero(np.isneginf(upper_sampled))
        upper_candidates = tuple(
            (-math.inf, float(upper_grid[index])) for index in equality_indices
        )
    else:
        upper_candidates = _continuous_local_minima(
            upper_grid,
            upper_sampled,
            upper_log_ratio,
            mandatory_points=mandatory,
        )
    lower_log, lower_epsilon = min(lower_candidates, key=lambda item: item[0])
    upper_log, upper_epsilon = min(upper_candidates, key=lambda item: item[0])
    if lower_log > math.log(np.finfo(float).max):
        raise ValueError("lower amplitude bound is not representable")
    amplitude_min = -math.exp(lower_log)
    amplitude_max = 0.0 if upper_log == -math.inf else math.exp(upper_log)
    if (
        not math.isfinite(amplitude_min)
        or not math.isfinite(amplitude_max)
        or not amplitude_min < amplitude_max
        or not amplitude_min < 0.0
        or amplitude_max < 0.0
    ):
        raise ValueError("geometry has an empty or invalid admissible amplitude interval")

    lower_cs2 = baseline_cs2(lower_epsilon)
    upper_cs2 = baseline_cs2(upper_epsilon)
    lower_shape = math.exp(log_shape(lower_epsilon))
    upper_shape = math.exp(log_shape(upper_epsilon))
    return BSk24AmplitudeBounds(
        epsilon0_mev_fm3=float(epsilon0_mev_fm3),
        sigma_mev_fm3=float(sigma_mev_fm3),
        delta_mev_fm3=float(delta_mev_fm3),
        epsilon_match_mev_fm3=epsilon_match,
        epsilon_max_mev_fm3=epsilon_max,
        amplitude_min=amplitude_min,
        amplitude_max=amplitude_max,
        lower_limiting_epsilon_mev_fm3=lower_epsilon,
        upper_limiting_epsilon_mev_fm3=upper_epsilon,
        lower_limiting_baseline_cs2=lower_cs2,
        upper_limiting_baseline_cs2=upper_cs2,
        lower_limiting_shape=lower_shape,
        upper_limiting_shape=upper_shape,
        baseline_minimum_cs2=baseline_min,
        baseline_minimum_epsilon_mev_fm3=baseline_min_epsilon,
        baseline_maximum_cs2=baseline_max,
        baseline_maximum_epsilon_mev_fm3=baseline_max_epsilon,
        lower_candidate_extrema_mev_fm3=tuple(
            item[1] for item in lower_candidates
        ),
        upper_candidate_extrema_mev_fm3=tuple(
            item[1] for item in upper_candidates
        ),
        discovery_grid_points=int(len(upper_grid)),
    )


def windowed_gaussian_pressure_primitive(
    energy_density_mev_fm3: Any,
    deformation: BSk24WindowedDeformation,
    *,
    epsilon_t_mev_fm3: float,
) -> float | np.ndarray:
    """Return A times the analytic integral of G*W from the anchor."""
    return _windowed_gaussian_pressure_primitive_impl(
        energy_density_mev_fm3,
        deformation,
        epsilon_t_mev_fm3=epsilon_t_mev_fm3,
    )


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

    def sound_speed_squared_from_energy_density(
        self, value: Any
    ) -> float | np.ndarray:
        epsilon = _require_domain(
            value,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
            name="energy_density_mev_fm3",
        )
        return _scalar_or_array(
            _windowed_cs2(epsilon, self.baseline, self.deformation)
        )

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
        return _scalar_or_array(
            np.exp(self._n_interpolator(np.log(epsilon)))
        )

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]:
        epsilon = float(self.energy_density_from_pressure(pressure_mev_fm3))
        return epsilon, float(
            self.sound_speed_squared_from_energy_density(epsilon)
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "generator_id": WINDOWED_GAUSSIAN_GENERATOR_ID,
            "preserved_existing_generator_id": PURE_GAUSSIAN_GENERATOR_ID,
            "model_name": f"{MODEL_NAME}_WINDOWED_GAUSSIAN_EFFECTIVE_BAROTROPE",
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


def raw_local_physics_gate(
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24WindowedDeformation,
    *,
    dense_lower_points: int = 16385,
    dense_upper_points: int = 65537,
    amplitude_bounds: BSk24AmplitudeBounds | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Assess the complete raw proposal and select its first causal branch.

    Optional ``amplitude_bounds`` supply an independently calculated
    full-direct-domain interval.  Its mechanical lower bound remains hard;
    its upper bound is diagnostic when a first causal endpoint is resolved.
    """
    return _raw_local_physics_gate_impl(
        baseline,
        deformation,
        dense_lower_points=dense_lower_points,
        dense_upper_points=dense_upper_points,
        amplitude_bounds=amplitude_bounds,
    )


def window_characterization(
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24WindowedDeformation,
) -> dict[str, Any]:
    """Measure one deformation while reusing exact unit-geometry integrals.

    Gaussian/window geometry is independent of ``A``.  The adaptive
    quadratures, 131073-point extremum discovery, and FWHM roots are therefore
    evaluated once for consecutive amplitudes with the same
    ``(epsilon0, sigma, Delta)``.  Amplitude-dependent quantities are then
    formed with the same scalar operations used by the uncached path.
    """
    return _window_characterization_impl(baseline, deformation)


def summarize_windowed_residuals(
    eos: BSk24WindowedEos,
    *,
    exclude_boundary_points: int = 4,
) -> dict[str, Any]:
    """Separate global, interior, ramp, transition, and boundary residuals."""
    return _summarize_windowed_residuals_impl(
        eos,
        exclude_boundary_points=exclude_boundary_points,
    )


def full_domain_thermodynamic_admissibility(
    baseline: BSk24ConsistentBaseline,
    eos: BSk24WindowedEos,
    *,
    raw_gate_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the authoritative thermodynamic gate without truncation.

    The independent constraints are finiteness, positive retained energy and
    pressure, ``0 < c_s^2 <= 1``, exact below-match identity, continuous
    matching, successful effective first-law reconstruction, positive and
    monotone effective baryon density, and positive effective chemical
    potential.  ``Gamma_eff``, ``P <= epsilon``, and ``dmu/dn`` are reported
    only as diagnostics and are not extra parameter constraints.
    """
    return _full_domain_thermodynamic_admissibility_impl(
        baseline,
        eos,
        raw_gate_report=raw_gate_report,
    )


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
            else float(
                baseline.eos.energy_density_max_published_fit_mev_fm3
            )
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
        or retained.get("policy")
        != "prefix_through_first_continuous_cs2_equals_one"
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
        _windowed_pressure(
            np.asarray([endpoint], dtype=float), baseline, deformation
        )[0]
    )
    expected_cs2 = float(
        _windowed_cs2(
            np.asarray([endpoint], dtype=float), baseline, deformation
        )[0]
    )
    comparison_rtol = 64.0 * np.finfo(float).eps
    if (
        not math.isclose(
            endpoint_pressure,
            expected_pressure,
            rel_tol=comparison_rtol,
            abs_tol=comparison_rtol * max(1.0, abs(expected_pressure)),
        )
        or not math.isclose(
            endpoint_cs2,
            expected_cs2,
            rel_tol=comparison_rtol,
            abs_tol=comparison_rtol,
        )
    ):
        raise ValueError(
            "raw-gate retained endpoint state disagrees with the analytical proposal"
        )
    crossing = retained.get("first_causal_crossing")
    reason = retained.get("endpoint_reason")
    if reason == "direct_bsk24_causal_endpoint":
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
                "direct raw-gate endpoint must be the complete BSk24 causal endpoint"
            )
        return endpoint, False
    if reason == "published_bsk24_fit_endpoint":
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
    representable_width = crossing.get(
        "representable_bracket_width_mev_fm3"
    )
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
                    and crossing.get("first_noncausal_epsilon_mev_fm3")
                    is None
                    and crossing.get("first_noncausal_cs2") is None
                )
            elif common_width_evidence_valid:
                noncausal_cs2_values = _windowed_cs2(
                    np.asarray([bracket[1]], dtype=float),
                    baseline,
                    deformation,
                )
                analytical_noncausal_cs2 = float(
                    noncausal_cs2_values[0]
                )
                reported_noncausal_epsilon = crossing.get(
                    "first_noncausal_epsilon_mev_fm3"
                )
                reported_noncausal_cs2 = crossing.get(
                    "first_noncausal_cs2"
                )
                try:
                    reported_noncausal_epsilon = float(
                        reported_noncausal_epsilon
                    )
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
                        and reported_noncausal_cs2
                        == analytical_noncausal_cs2
                    )
            else:
                width_evidence_valid = False
    if (
        crossing.get("status")
        != "resolved_first_continuous_causal_crossing"
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
        epsilon_match_mev_fm3=(
            baseline.anchor.energy_density_mev_fm3
        ),
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
    return _analytical_pressure_derivative_certificate(
        epsilon,
        pressure,
        lambda value: float(
            _windowed_cs2(np.asarray(value), baseline, deformation)
        ),
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
    if raw_gate_report is None:
        raw_gate_report, _gate_epsilon, _gate_cs2 = (
            _raw_local_physics_gate_impl(baseline, deformation)
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

    raw_domain = raw_gate_report[
        "complete_proposed_retained_domain_mev_fm3"
    ]
    raw_upper = float(raw_domain[1])
    if raw_upper == float(baseline.epsilon[-1]):
        raw_epsilon = baseline.epsilon.copy()
    else:
        terminal_spacing = float(
            baseline.epsilon[-1] - baseline.epsilon[-2]
        )
        extension_intervals = int(
            math.ceil(
                (raw_upper - float(baseline.epsilon[-1]))
                / terminal_spacing
            )
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
        or (
            has_causal_crossing
            and (np.any(cs2[:-1] >= 1.0) or cs2[-1] > 1.0)
        )
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
        np.flatnonzero(
            retained_epsilon == baseline.anchor.energy_density_mev_fm3
        )[0]
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
    residuals = _residual_arrays(
        retained_epsilon, pressure, cs2, baryon_density, mu
    )
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
    ramp_end = (
        baseline.anchor.energy_density_mev_fm3 + deformation.delta_mev_fm3
    )
    diagnostics = {
        "generator_id": WINDOWED_GAUSSIAN_GENERATOR_ID,
        "preserved_existing_generator_id": PURE_GAUSSIAN_GENERATOR_ID,
        "deformation": deformation.to_dict(),
        "raw_gate_report": dict(raw_gate_report) if raw_gate_report is not None else None,
        "tabulation_resolution": tabulation_resolution,
        "unchanged_below_anchor": {
            "pressure_array_equal": bool(
                np.array_equal(pressure[below], baseline.pressure[below])
            ),
            "cs2_array_equal": bool(
                np.array_equal(cs2[below], baseline.cs2[below])
            ),
            "baryon_density_array_equal": bool(
                np.array_equal(
                    baryon_density[below], baseline.baryon_density[below]
                )
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
                baryon_density[anchor_index]
                - baseline.anchor.baryon_density_fm3
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
            "endpoint_reason": raw_gate_report["retained_domain"][
                "endpoint_reason"
            ],
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
        forward = np.asarray(
            eos.pressure_from_energy_density(recovered), dtype=float
        )
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
    eos.diagnostics["retained_domain_thermodynamic_admissibility"] = (
        admissibility
    )
    accepted_admissibility_statuses = {
        "accepted_full_domain_thermodynamic_gate",
        "accepted_selected_domain_thermodynamic_gate",
    }
    if admissibility["status"] not in accepted_admissibility_statuses:
        raise BSk24MechanicalStabilityError(admissibility)
    if require_full_domain:
        # Preserve the established diagnostic key and status for callers that
        # explicitly request direct-endpoint compatibility.
        eos.diagnostics["full_domain_thermodynamic_admissibility"] = (
            admissibility
        )
        if admissibility["status"] != "accepted_full_domain_thermodynamic_gate":
            raise BSk24MechanicalStabilityError(admissibility)
    return eos


def windowed_a0_identity_report(
    baseline: BSk24ConsistentBaseline,
    cases: Mapping[float, BSk24WindowedEos],
) -> dict[str, Any]:
    """Check exact local A=0 identity independently for every Delta."""
    return _windowed_a0_identity_report_impl(baseline, cases)


__all__ = [
    "BSK24_RETAINED_EPSILON_MATCH_MEV_FM3",
    "BSK24_RETAINED_EPSILON_MAX_MEV_FM3",
    "BSk24AmplitudeBounds",
    "CONTROL_DELTAS_MEV_FM3",
    "PRIMARY_AMPLITUDES",
    "PRIMARY_DELTA_MEV_FM3",
    "PRIMARY_EPSILON0_MEV_FM3",
    "PRIMARY_SIGMA_MEV_FM3",
    "PURE_GAUSSIAN_GENERATOR_ID",
    "WINDOWED_GAUSSIAN_GENERATOR_ID",
    "BSk24WindowedDeformation",
    "BSk24WindowedEos",
    "build_windowed_eos",
    "calculate_windowed_amplitude_bounds",
    "full_domain_thermodynamic_admissibility",
    "gaussian_profile",
    "raw_local_physics_gate",
    "smootherstep_window",
    "smootherstep_window_first_derivative",
    "smootherstep_window_second_derivative",
    "summarize_windowed_residuals",
    "window_characterization",
    "windowed_a0_identity_report",
    "windowed_gaussian_delta_cs2",
    "windowed_gaussian_pressure_primitive",
    "windowed_gaussian_shape",
]

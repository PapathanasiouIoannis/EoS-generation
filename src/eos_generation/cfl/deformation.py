"""CFL-specific surface-anchored sound-speed deformation and raw gate.

The independent variable is total energy density on the complete governed CFL
domain.  A quintic smootherstep preserves the undeformed finite-density
surface exactly; no crust or BSk24 assumption enters this module.  Raw
proposals are assessed without clipping, smoothing, or post-hoc repair.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, ndtr

from eos_generation.cfl.baseline import (
    CFLAnalyticEos,
    CFL_DEFORMATION_PROFILE_ID,
    CFL_DEFORMATION_PROFILE_VERSION,
    ENERGY_DENSITY_MAX_MEV_FM3,
    ENERGY_DENSITY_SURFACE_MEV_FM3,
    FROZEN_PARAMETER_SET_ID,
    FROZEN_PARAMETER_SET_SHA256,
    QUARK_CHEMICAL_POTENTIAL_MAX_MEV,
    QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV,
    make_cfl_eos,
)


CFL_DEFORMATION_SCHEMA_VERSION = "cfl_windowed_deformation_v1"
CFL_AMPLITUDE_BOUNDS_SCHEMA_VERSION = "cfl_windowed_amplitude_bounds_v1"
CFL_RAW_GATE_SCHEMA_VERSION = "cfl_raw_local_physics_gate_v1"
CFL_PRESSURE_PRIMITIVE_POLICY = (
    "normalized_segmented_gauss_legendre_64_with_stable_normal_tail_v1"
)
_RAMP_GAUSS_LEGENDRE_ORDER = 64
_RAMP_GAUSSIAN_BREAKPOINTS = (
    -12.0,
    -8.0,
    -4.0,
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    4.0,
    8.0,
    12.0,
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


@dataclass(frozen=True, slots=True)
class CFLWindowedDeformation:
    """One deterministic surface-anchored Gaussian sound-speed proposal."""

    case_id: str
    amplitude: float
    center_mev_fm3: float
    width_mev_fm3: float
    ramp_width_mev_fm3: float

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        values = (
            self.amplitude,
            self.center_mev_fm3,
            self.width_mev_fm3,
            self.ramp_width_mev_fm3,
        )
        if not np.isfinite(values).all():
            raise ValueError("CFL deformation parameters must be finite")
        if not (
            ENERGY_DENSITY_SURFACE_MEV_FM3
            < self.center_mev_fm3
            < ENERGY_DENSITY_MAX_MEV_FM3
        ):
            raise ValueError(
                "center must lie strictly between the self-bound surface and "
                "the governed CFL energy-density endpoint"
            )
        if self.width_mev_fm3 <= 0.0:
            raise ValueError("width must be positive")
        if not (
            self.center_mev_fm3 - self.width_mev_fm3
            < self.center_mev_fm3
            < self.center_mev_fm3 + self.width_mev_fm3
        ):
            raise ValueError(
                "width must move the center in both directions in binary64"
            )
        if self.ramp_width_mev_fm3 <= 0.0:
            raise ValueError("ramp_width must be positive")
        ramp_end = (
            ENERGY_DENSITY_SURFACE_MEV_FM3
            + self.ramp_width_mev_fm3
        )
        if ramp_end <= ENERGY_DENSITY_SURFACE_MEV_FM3:
            raise ValueError(
                "ramp_width must produce a representably distinct binary64 "
                "endpoint above the CFL surface"
            )
        if ramp_end > ENERGY_DENSITY_MAX_MEV_FM3:
            raise ValueError(
                "surface plus ramp_width must not exceed the governed CFL endpoint"
            )

    @property
    def epsilon0_mev_fm3(self) -> float:
        """Compatibility spelling for the Gaussian center."""

        return self.center_mev_fm3

    @property
    def sigma_mev_fm3(self) -> float:
        """Compatibility spelling for the Gaussian standard deviation."""

        return self.width_mev_fm3

    @property
    def delta_mev_fm3(self) -> float:
        """Compatibility spelling for the smootherstep ramp width."""

        return self.ramp_width_mev_fm3

    @property
    def epsilon_match_mev_fm3(self) -> float:
        return ENERGY_DENSITY_SURFACE_MEV_FM3

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CFL_DEFORMATION_SCHEMA_VERSION,
            "profile_id": CFL_DEFORMATION_PROFILE_ID,
            "profile_version": CFL_DEFORMATION_PROFILE_VERSION,
            "case_id": self.case_id,
            "amplitude": self.amplitude,
            "center_mev_fm3": self.center_mev_fm3,
            "width_mev_fm3": self.width_mev_fm3,
            "ramp_width_mev_fm3": self.ramp_width_mev_fm3,
            "epsilon_match_mev_fm3": ENERGY_DENSITY_SURFACE_MEV_FM3,
            "complete_domain_mev_fm3": [
                ENERGY_DENSITY_SURFACE_MEV_FM3,
                ENERGY_DENSITY_MAX_MEV_FM3,
            ],
            "baseline_parameter_set_id": FROZEN_PARAMETER_SET_ID,
            "baseline_parameter_set_sha256": FROZEN_PARAMETER_SET_SHA256,
        }

    @property
    def case_sha256(self) -> str:
        return _canonical_sha256(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "case_sha256": self.case_sha256}


@dataclass(frozen=True, slots=True)
class CFLAmplitudeBounds:
    """Continuous raw interval ``(A_min, A_max]`` for one geometry."""

    center_mev_fm3: float
    width_mev_fm3: float
    ramp_width_mev_fm3: float
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

    def contains(self, amplitude: Any) -> bool:
        try:
            value = float(amplitude)
        except (TypeError, ValueError):
            return False
        return bool(
            math.isfinite(value)
            and value > self.amplitude_min
            and value <= self.amplitude_max
        )

    def matches(self, deformation: CFLWindowedDeformation) -> bool:
        return bool(
            self.center_mev_fm3 == deformation.center_mev_fm3
            and self.width_mev_fm3 == deformation.width_mev_fm3
            and self.ramp_width_mev_fm3 == deformation.ramp_width_mev_fm3
            and self.epsilon_match_mev_fm3
            == ENERGY_DENSITY_SURFACE_MEV_FM3
            and self.epsilon_max_mev_fm3 == ENERGY_DENSITY_MAX_MEV_FM3
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CFL_AMPLITUDE_BOUNDS_SCHEMA_VERSION,
            "profile_id": CFL_DEFORMATION_PROFILE_ID,
            "baseline_parameter_set_id": FROZEN_PARAMETER_SET_ID,
            "baseline_parameter_set_sha256": FROZEN_PARAMETER_SET_SHA256,
            "geometry": {
                "center_mev_fm3": self.center_mev_fm3,
                "width_mev_fm3": self.width_mev_fm3,
                "ramp_width_mev_fm3": self.ramp_width_mev_fm3,
            },
            "complete_domain_mev_fm3": [
                self.epsilon_match_mev_fm3,
                self.epsilon_max_mev_fm3,
            ],
            "amplitude_interval": {
                "A_min": self.amplitude_min,
                "A_max": self.amplitude_max,
                "continuous_ratio_A_max_before_binary64_inward_rounding": (
                    math.nextafter(self.amplitude_max, math.inf)
                ),
                "notation": "(A_min, A_max]",
                "lower_endpoint_open": True,
                "upper_endpoint_closed": True,
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
                "maximum_cs2": self.baseline_maximum_cs2,
                "maximum_epsilon_mev_fm3": (
                    self.baseline_maximum_epsilon_mev_fm3
                ),
                "mechanical_stability_margin": self.baseline_minimum_cs2,
                "causality_margin": 1.0 - self.baseline_maximum_cs2,
            },
            "continuous_extremum_search": {
                "policy": (
                    "complete-domain discovery plus bounded continuous refinement"
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
                "upper_endpoint_binary64_policy": (
                    "one_ulp_inward_from_the_continuous_ratio_so_the_closed_"
                    "representable_endpoint_itself_retains_raw_cs2<=1"
                ),
            },
        }


def smootherstep_window(
    energy_density_mev_fm3: Any,
    *,
    epsilon_match_mev_fm3: float = ENERGY_DENSITY_SURFACE_MEV_FM3,
    ramp_width_mev_fm3: float,
) -> float | np.ndarray:
    """Return ``0``, the exact quintic ramp, or ``1`` piecewise."""

    if not np.isfinite([epsilon_match_mev_fm3, ramp_width_mev_fm3]).all():
        raise ValueError("window geometry must be finite")
    if ramp_width_mev_fm3 <= 0.0:
        raise ValueError("ramp_width must be positive")
    ramp_end = epsilon_match_mev_fm3 + ramp_width_mev_fm3
    if not math.isfinite(ramp_end) or ramp_end <= epsilon_match_mev_fm3:
        raise ValueError(
            "ramp_width must produce a finite, representably distinct "
            "binary64 endpoint above epsilon_match"
        )
    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    if not np.all(np.isfinite(epsilon)):
        raise ValueError("energy density must be finite")
    result = np.zeros_like(epsilon)
    ramp = (epsilon > epsilon_match_mev_fm3) & (epsilon < ramp_end)
    t = (epsilon[ramp] - epsilon_match_mev_fm3) / ramp_width_mev_fm3
    result[ramp] = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
    result[epsilon >= ramp_end] = 1.0
    return _scalar_or_array(result)


def gaussian_profile(
    energy_density_mev_fm3: Any,
    deformation: CFLWindowedDeformation,
) -> float | np.ndarray:
    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    if not np.all(np.isfinite(epsilon)):
        raise ValueError("energy density must be finite")
    z = (epsilon - deformation.center_mev_fm3) / deformation.width_mev_fm3
    return _scalar_or_array(np.exp(-0.5 * z * z))


def windowed_gaussian_shape(
    energy_density_mev_fm3: Any,
    deformation: CFLWindowedDeformation,
) -> float | np.ndarray:
    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    gaussian = np.asarray(gaussian_profile(epsilon, deformation), dtype=float)
    window = np.asarray(
        smootherstep_window(
            epsilon,
            ramp_width_mev_fm3=deformation.ramp_width_mev_fm3,
        ),
        dtype=float,
    )
    return _scalar_or_array(gaussian * window)


def windowed_gaussian_delta_cs2(
    energy_density_mev_fm3: Any,
    deformation: CFLWindowedDeformation,
) -> float | np.ndarray:
    shape = np.asarray(
        windowed_gaussian_shape(energy_density_mev_fm3, deformation),
        dtype=float,
    )
    return _scalar_or_array(deformation.amplitude * shape)


@lru_cache(maxsize=1)
def _ramp_gauss_legendre_rule() -> tuple[np.ndarray, np.ndarray]:
    """Create the governed rule lazily so passive planning stays calculation-free."""

    nodes, weights = np.polynomial.legendre.leggauss(
        _RAMP_GAUSS_LEGENDRE_ORDER
    )
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _ramp_gaussian_integral_to_fraction(
    upper_fraction: float,
    deformation: CFLWindowedDeformation,
) -> float:
    """Integrate the ramp in its dimensionless coordinate without moments."""

    upper = float(upper_fraction)
    if not math.isfinite(upper) or upper < 0.0 or upper > 1.0:
        raise ValueError("ramp integration fraction must lie in [0, 1]")
    if upper == 0.0:
        return 0.0
    anchor = ENERGY_DENSITY_SURFACE_MEV_FM3
    width = deformation.width_mev_fm3
    delta = deformation.ramp_width_mev_fm3
    center_fraction = (
        deformation.center_mev_fm3 - anchor
    ) / delta
    sigma_fraction = width / delta
    boundaries = [0.0, upper]
    boundaries.extend(
        value
        for multiplier in _RAMP_GAUSSIAN_BREAKPOINTS
        if 0.0
        < (value := center_fraction + multiplier * sigma_fraction)
        < upper
    )
    boundaries = sorted(set(boundaries))
    nodes, weights = _ramp_gauss_legendre_rule()
    z_at_surface = (anchor - deformation.center_mev_fm3) / width
    delta_over_width = delta / width
    integral = 0.0
    for lower, upper_boundary in zip(boundaries[:-1], boundaries[1:]):
        half_width = 0.5 * (upper_boundary - lower)
        midpoint = 0.5 * (upper_boundary + lower)
        coordinates = midpoint + half_width * nodes
        window = coordinates**3 * (
            10.0 + coordinates * (-15.0 + 6.0 * coordinates)
        )
        z = z_at_surface + delta_over_width * coordinates
        with np.errstate(over="ignore", under="ignore"):
            gaussian = np.exp(-0.5 * z * z)
        integral += half_width * float(np.dot(weights, window * gaussian))
    return delta * integral


def _standard_normal_probability_between(
    lower: float,
    upper: np.ndarray,
) -> np.ndarray:
    """Return Phi(upper)-Phi(lower) without same-tail cancellation."""

    resolved_upper = np.asarray(upper, dtype=float)
    if np.any(resolved_upper < lower):
        raise ValueError("normal integral upper bound precedes lower bound")
    result = np.empty_like(resolved_upper)
    negative = resolved_upper <= 0.0
    positive = lower >= 0.0
    if positive:
        log_survival_lower = float(log_ndtr(-lower))
        log_survival_upper = log_ndtr(-resolved_upper)
        result[...] = math.exp(log_survival_lower) * (
            -np.expm1(log_survival_upper - log_survival_lower)
        )
        return result
    if np.any(negative):
        log_cdf_upper = log_ndtr(resolved_upper[negative])
        log_cdf_lower = float(log_ndtr(lower))
        result[negative] = np.exp(log_cdf_upper) * (
            -np.expm1(log_cdf_lower - log_cdf_upper)
        )
    crossing = ~negative
    if np.any(crossing):
        result[crossing] = ndtr(resolved_upper[crossing]) - ndtr(lower)
    return result


def windowed_gaussian_pressure_primitive(
    energy_density_mev_fm3: Any,
    deformation: CFLWindowedDeformation,
) -> float | np.ndarray:
    """Return the governed stable ``A integral_surface^epsilon G W de``."""

    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    if not np.all(np.isfinite(epsilon)):
        raise ValueError("energy density must be finite")
    if deformation.amplitude == 0.0:
        return _scalar_or_array(np.zeros_like(epsilon))
    flat = epsilon.reshape(-1)
    result = np.zeros_like(flat)
    anchor = ENERGY_DENSITY_SURFACE_MEV_FM3
    ramp_end = anchor + deformation.ramp_width_mev_fm3
    ramp = (flat > anchor) & (flat < ramp_end)
    if np.any(ramp):
        fractions = (flat[ramp] - anchor) / deformation.ramp_width_mev_fm3
        result[ramp] = np.asarray(
            [
                _ramp_gaussian_integral_to_fraction(value, deformation)
                for value in fractions
            ],
            dtype=float,
        )
    above = flat >= ramp_end
    if np.any(above):
        ramp_area = _ramp_gaussian_integral_to_fraction(1.0, deformation)
        center = deformation.center_mev_fm3
        width = deformation.width_mev_fm3
        z_start = (ramp_end - center) / width
        z_upper = (flat[above] - center) / width
        gaussian_area = math.sqrt(2.0 * math.pi) * width * (
            _standard_normal_probability_between(z_start, z_upper)
        )
        result[above] = ramp_area + gaussian_area
    result *= deformation.amplitude
    return _scalar_or_array(result.reshape(epsilon.shape))


def _log_windowed_shape_scalar(
    epsilon_mev_fm3: float,
    *,
    center_mev_fm3: float,
    width_mev_fm3: float,
    ramp_width_mev_fm3: float,
) -> float:
    epsilon = float(epsilon_mev_fm3)
    anchor = ENERGY_DENSITY_SURFACE_MEV_FM3
    if epsilon <= anchor:
        return -math.inf
    ramp_end = anchor + ramp_width_mev_fm3
    if epsilon < ramp_end:
        t = (epsilon - anchor) / ramp_width_mev_fm3
        window = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
        if window <= 0.0:
            return -math.inf
        log_window = math.log(window)
    else:
        log_window = 0.0
    z = (epsilon - center_mev_fm3) / width_mev_fm3
    if not math.isfinite(z):
        return -math.inf
    z_squared = z * z
    if not math.isfinite(z_squared):
        return -math.inf
    return -0.5 * z_squared + log_window


def _geometry_discovery_grid(
    baseline: CFLAnalyticEos,
    deformation: CFLWindowedDeformation,
    *,
    discovery_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.linspace(
        QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV,
        QUARK_CHEMICAL_POTENTIAL_MAX_MEV,
        discovery_points,
    )
    epsilon_for_mapping = np.asarray(
        baseline.energy_density_from_quark_chemical_potential(mu), dtype=float
    )
    anchor = ENERGY_DENSITY_SURFACE_MEV_FM3
    ramp_end = anchor + deformation.ramp_width_mev_fm3
    local_lower = max(anchor, deformation.center_mev_fm3 - 12.0 * deformation.width_mev_fm3)
    local_upper = min(
        ENERGY_DENSITY_MAX_MEV_FM3,
        deformation.center_mev_fm3 + 12.0 * deformation.width_mev_fm3,
    )
    extra_parts = [
        np.asarray([deformation.center_mev_fm3, ramp_end], dtype=float),
        np.linspace(anchor, ramp_end, 2049)[1:],
    ]
    if local_upper > local_lower:
        extra_parts.append(np.linspace(local_lower, local_upper, 4097))
    extras = np.unique(np.concatenate(extra_parts))
    extras = extras[
        (extras >= ENERGY_DENSITY_SURFACE_MEV_FM3)
        & (extras <= ENERGY_DENSITY_MAX_MEV_FM3)
    ]
    if extras.size:
        # The interpolation supplies only discovery coordinates.  Re-evaluate
        # every thermodynamic quantity at those mu values so the returned raw
        # state is internally exact rather than mixing an approximate inverse
        # with the requested epsilon coordinate.
        extra_mu = np.interp(extras, epsilon_for_mapping, mu)
        mu = np.unique(np.concatenate((mu, extra_mu)))
    epsilon = np.asarray(
        baseline.energy_density_from_quark_chemical_potential(mu), dtype=float
    )
    cs2 = np.asarray(
        baseline.sound_speed_squared_from_quark_chemical_potential(mu),
        dtype=float,
    )
    pressure = np.asarray(
        baseline.pressure_from_quark_chemical_potential(mu), dtype=float
    )
    return epsilon, cs2, pressure


def _continuous_candidates(
    grid: np.ndarray,
    sampled_values: np.ndarray,
    function: Callable[[float], float],
    *,
    mandatory_points: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    finite = np.isfinite(sampled_values)
    grid = grid[finite]
    sampled_values = sampled_values[finite]
    if len(grid) < 3:
        raise ValueError("geometry is not numerically resolvable on the governed domain")
    local = np.flatnonzero(
        (sampled_values[1:-1] <= sampled_values[:-2])
        & (sampled_values[1:-1] <= sampled_values[2:])
    ) + 1
    candidates: list[tuple[float, float]] = [
        (float(sampled_values[0]), float(grid[0])),
        (float(sampled_values[-1]), float(grid[-1])),
    ]
    for index in local:
        lower = float(grid[index - 1])
        upper = float(grid[index + 1])
        if not upper > lower:
            continue
        result = minimize_scalar(
            function,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        if result.success and math.isfinite(float(result.fun)):
            candidates.append((float(result.fun), float(result.x)))
    for point in mandatory_points:
        if grid[0] <= point <= grid[-1]:
            value = float(function(point))
            if math.isfinite(value):
                candidates.append((value, float(point)))
    candidates.sort(key=lambda item: (item[1], item[0]))
    deduplicated: list[tuple[float, float]] = []
    for value, location in candidates:
        if deduplicated and math.isclose(
            location, deduplicated[-1][1], rel_tol=0.0, abs_tol=2.0e-9
        ):
            if value < deduplicated[-1][0]:
                deduplicated[-1] = (value, location)
        else:
            deduplicated.append((value, location))
    return tuple(deduplicated)


def _refined_extremum(
    grid: np.ndarray,
    sampled_values: np.ndarray,
    function: Callable[[float], float],
    *,
    maximize: bool,
) -> tuple[float, float]:
    transformed = -sampled_values if maximize else sampled_values
    index = int(np.argmin(transformed))
    candidates = [(float(sampled_values[index]), float(grid[index]))]
    if 0 < index < len(grid) - 1:
        result = minimize_scalar(
            (lambda value: -function(value)) if maximize else function,
            bounds=(float(grid[index - 1]), float(grid[index + 1])),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        if result.success:
            candidates.append((float(function(float(result.x))), float(result.x)))
    return (max(candidates) if maximize else min(candidates))


def calculate_windowed_amplitude_bounds(
    *,
    center_mev_fm3: float,
    width_mev_fm3: float,
    ramp_width_mev_fm3: float,
    baseline: CFLAnalyticEos | None = None,
    discovery_points: int = 32769,
) -> CFLAmplitudeBounds:
    """Calculate the continuous no-repair interval ``0 < c_s^2 <= 1``."""

    if (
        not isinstance(discovery_points, int)
        or isinstance(discovery_points, bool)
        or discovery_points < 257
        or discovery_points % 2 == 0
    ):
        raise ValueError("discovery_points must be an odd integer of at least 257")
    geometry = CFLWindowedDeformation(
        case_id="amplitude_bounds_geometry",
        amplitude=0.0,
        center_mev_fm3=float(center_mev_fm3),
        width_mev_fm3=float(width_mev_fm3),
        ramp_width_mev_fm3=float(ramp_width_mev_fm3),
    )
    model = baseline or make_cfl_eos()
    epsilon, baseline_cs2, _baseline_pressure = _geometry_discovery_grid(
        model, geometry, discovery_points=discovery_points
    )
    log_shape = np.asarray(
        [
            _log_windowed_shape_scalar(
                value,
                center_mev_fm3=geometry.center_mev_fm3,
                width_mev_fm3=geometry.width_mev_fm3,
                ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
            )
            for value in epsilon
        ],
        dtype=float,
    )
    lower_samples = np.log(baseline_cs2) - log_shape
    upper_samples = np.log1p(-baseline_cs2) - log_shape

    def baseline_cs2_scalar(value: float) -> float:
        return float(model.sound_speed_squared_from_energy_density(value))

    def lower_log_ratio(value: float) -> float:
        log_f = _log_windowed_shape_scalar(
            value,
            center_mev_fm3=geometry.center_mev_fm3,
            width_mev_fm3=geometry.width_mev_fm3,
            ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
        )
        if not math.isfinite(log_f):
            return math.inf
        return math.log(baseline_cs2_scalar(value)) - log_f

    def upper_log_ratio(value: float) -> float:
        log_f = _log_windowed_shape_scalar(
            value,
            center_mev_fm3=geometry.center_mev_fm3,
            width_mev_fm3=geometry.width_mev_fm3,
            ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
        )
        if not math.isfinite(log_f):
            return math.inf
        return math.log1p(-baseline_cs2_scalar(value)) - log_f

    mandatory = (
        geometry.center_mev_fm3,
        ENERGY_DENSITY_SURFACE_MEV_FM3 + geometry.ramp_width_mev_fm3,
        ENERGY_DENSITY_MAX_MEV_FM3,
    )
    lower_candidates = _continuous_candidates(
        epsilon,
        lower_samples,
        lower_log_ratio,
        mandatory_points=mandatory,
    )
    upper_candidates = _continuous_candidates(
        epsilon,
        upper_samples,
        upper_log_ratio,
        mandatory_points=mandatory,
    )
    lower_log, lower_location = min(lower_candidates)
    upper_log, upper_location = min(upper_candidates)
    lower_ratio = math.exp(lower_log)
    upper_ratio = math.exp(upper_log)
    lower_shape = math.exp(
        _log_windowed_shape_scalar(
            lower_location,
            center_mev_fm3=geometry.center_mev_fm3,
            width_mev_fm3=geometry.width_mev_fm3,
            ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
        )
    )
    upper_shape = math.exp(
        _log_windowed_shape_scalar(
            upper_location,
            center_mev_fm3=geometry.center_mev_fm3,
            width_mev_fm3=geometry.width_mev_fm3,
            ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
        )
    )
    baseline_min, baseline_min_location = _refined_extremum(
        epsilon,
        baseline_cs2,
        baseline_cs2_scalar,
        maximize=False,
    )
    baseline_max, baseline_max_location = _refined_extremum(
        epsilon,
        baseline_cs2,
        baseline_cs2_scalar,
        maximize=True,
    )
    return CFLAmplitudeBounds(
        center_mev_fm3=geometry.center_mev_fm3,
        width_mev_fm3=geometry.width_mev_fm3,
        ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
        epsilon_match_mev_fm3=ENERGY_DENSITY_SURFACE_MEV_FM3,
        epsilon_max_mev_fm3=ENERGY_DENSITY_MAX_MEV_FM3,
        amplitude_min=-lower_ratio,
        amplitude_max=math.nextafter(upper_ratio, -math.inf),
        lower_limiting_epsilon_mev_fm3=lower_location,
        upper_limiting_epsilon_mev_fm3=upper_location,
        lower_limiting_baseline_cs2=baseline_cs2_scalar(lower_location),
        upper_limiting_baseline_cs2=baseline_cs2_scalar(upper_location),
        lower_limiting_shape=lower_shape,
        upper_limiting_shape=upper_shape,
        baseline_minimum_cs2=baseline_min,
        baseline_minimum_epsilon_mev_fm3=baseline_min_location,
        baseline_maximum_cs2=baseline_max,
        baseline_maximum_epsilon_mev_fm3=baseline_max_location,
        lower_candidate_extrema_mev_fm3=tuple(
            location for _, location in lower_candidates
        ),
        upper_candidate_extrema_mev_fm3=tuple(
            location for _, location in upper_candidates
        ),
        discovery_grid_points=int(len(epsilon)),
    )


def window_characterization(
    deformation: CFLWindowedDeformation,
) -> dict[str, Any]:
    """Return geometry diagnostics, separate from the hard raw gate."""

    anchor = ENERGY_DENSITY_SURFACE_MEV_FM3
    ramp_end = anchor + deformation.ramp_width_mev_fm3
    result = minimize_scalar(
        lambda value: -float(windowed_gaussian_shape(value, deformation)),
        bounds=(anchor, ENERGY_DENSITY_MAX_MEV_FM3),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    peak_location = float(result.x)
    peak_shape = float(windowed_gaussian_shape(peak_location, deformation))
    return {
        "schema_version": "cfl_window_characterization_v1",
        "profile_id": CFL_DEFORMATION_PROFILE_ID,
        "deformation": deformation.to_dict(),
        "independent_variable": "total_energy_density_mev_fm3",
        "complete_domain_mev_fm3": [
            anchor,
            ENERGY_DENSITY_MAX_MEV_FM3,
        ],
        "surface_anchor_mev_fm3": anchor,
        "ramp_end_mev_fm3": ramp_end,
        "window_at_surface": 0.0,
        "window_at_ramp_end": 1.0,
        "window_above_ramp_end": 1.0,
        "gaussian_standard_deviation_mev_fm3": deformation.width_mev_fm3,
        "gaussian_fwhm_mev_fm3": (
            2.0 * math.sqrt(2.0 * math.log(2.0)) * deformation.width_mev_fm3
        ),
        "effective_shape_peak_epsilon_mev_fm3": peak_location,
        "effective_shape_peak": peak_shape,
        "diagnostic_only": True,
    }


def raw_local_physics_gate(
    deformation: CFLWindowedDeformation,
    *,
    baseline: CFLAnalyticEos | None = None,
    amplitude_bounds: CFLAmplitudeBounds | None = None,
    dense_points: int = 65537,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Assess the untouched raw proposal over the full governed CFL domain."""

    if (
        not isinstance(dense_points, int)
        or isinstance(dense_points, bool)
        or dense_points < 257
        or dense_points % 2 == 0
    ):
        raise ValueError("dense_points must be an odd integer of at least 257")
    model = baseline or make_cfl_eos()
    bounds = amplitude_bounds or calculate_windowed_amplitude_bounds(
        center_mev_fm3=deformation.center_mev_fm3,
        width_mev_fm3=deformation.width_mev_fm3,
        ramp_width_mev_fm3=deformation.ramp_width_mev_fm3,
        baseline=model,
    )
    if not bounds.matches(deformation):
        raise ValueError("amplitude_bounds do not match the deformation geometry")
    epsilon, baseline_cs2, baseline_pressure = _geometry_discovery_grid(
        model, deformation, discovery_points=dense_points
    )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        raw_cs2 = baseline_cs2 + np.asarray(
            windowed_gaussian_delta_cs2(epsilon, deformation), dtype=float
        )
        raw_pressure = baseline_pressure + np.asarray(
            windowed_gaussian_pressure_primitive(epsilon, deformation), dtype=float
        )

    def raw_cs2_scalar(value: float) -> float:
        return float(model.sound_speed_squared_from_energy_density(value)) + float(
            windowed_gaussian_delta_cs2(value, deformation)
        )

    sampled_finite = bool(
        np.all(np.isfinite(epsilon))
        and np.all(np.isfinite(raw_pressure))
        and np.all(np.isfinite(raw_cs2))
    )
    if sampled_finite:
        minimum, minimum_location = _refined_extremum(
            epsilon, raw_cs2, raw_cs2_scalar, maximize=False
        )
        maximum, maximum_location = _refined_extremum(
            epsilon, raw_cs2, raw_cs2_scalar, maximize=True
        )
    else:
        minimum = math.nan
        minimum_location = math.nan
        maximum = math.nan
        maximum_location = math.nan
    finite = bool(
        sampled_finite and math.isfinite(minimum) and math.isfinite(maximum)
    )
    positive_energy = bool(finite and np.all(epsilon > 0.0))
    nonnegative_pressure = bool(finite and np.all(raw_pressure >= 0.0))
    amplitude_passed = bounds.contains(deformation.amplitude)
    stable = bool(finite and minimum > 0.0)
    causal = bool(finite and maximum <= 1.0)
    surface_window = float(
        smootherstep_window(
            ENERGY_DENSITY_SURFACE_MEV_FM3,
            ramp_width_mev_fm3=deformation.ramp_width_mev_fm3,
        )
    )
    surface_delta_cs2 = float(
        windowed_gaussian_delta_cs2(
            ENERGY_DENSITY_SURFACE_MEV_FM3,
            deformation,
        )
    )
    surface_baseline_cs2 = float(
        model.sound_speed_squared_from_energy_density(
            ENERGY_DENSITY_SURFACE_MEV_FM3
        )
    )
    surface_preserved = bool(
        raw_pressure[0] == 0.0
        and raw_cs2[0] == surface_baseline_cs2
        and surface_window == 0.0
        and surface_delta_cs2 == 0.0
    )
    passed = bool(
        finite
        and positive_energy
        and nonnegative_pressure
        and amplitude_passed
        and stable
        and causal
        and surface_preserved
    )

    failure: dict[str, Any] | None = None
    if not finite:
        invalid = (
            ~np.isfinite(epsilon)
            | ~np.isfinite(raw_pressure)
            | ~np.isfinite(raw_cs2)
        )
        index = int(np.flatnonzero(invalid)[0]) if np.any(invalid) else 0

        def retained(value: float) -> float | None:
            return float(value) if math.isfinite(float(value)) else None

        def classification(value: float) -> str:
            resolved = float(value)
            if math.isnan(resolved):
                return "nan"
            if resolved == math.inf:
                return "positive_infinity"
            if resolved == -math.inf:
                return "negative_infinity"
            return "finite"

        failure = {
            "reason": "nonfinite_raw_state",
            "epsilon_mev_fm3": retained(epsilon[index]),
            "pressure_mev_fm3": retained(raw_pressure[index]),
            "pressure_classification": classification(raw_pressure[index]),
            "raw_cs2": retained(raw_cs2[index]),
            "raw_cs2_classification": classification(raw_cs2[index]),
        }
    elif not positive_energy:
        index = int(np.flatnonzero(epsilon <= 0.0)[0])
        failure = {
            "reason": "nonpositive_energy_density",
            "epsilon_mev_fm3": float(epsilon[index]),
            "pressure_mev_fm3": float(raw_pressure[index]),
            "raw_cs2": float(raw_cs2[index]),
        }
    elif not nonnegative_pressure:
        index = int(np.flatnonzero(raw_pressure < 0.0)[0])
        failure = {
            "reason": "negative_raw_pressure",
            "epsilon_mev_fm3": float(epsilon[index]),
            "pressure_mev_fm3": float(raw_pressure[index]),
            "raw_cs2": float(raw_cs2[index]),
        }
    elif not surface_preserved:
        failure = {
            "reason": "self_bound_surface_anchor_not_preserved_exactly",
            "epsilon_mev_fm3": ENERGY_DENSITY_SURFACE_MEV_FM3,
            "pressure_mev_fm3": float(raw_pressure[0]),
            "raw_cs2": float(raw_cs2[0]),
            "baseline_cs2": surface_baseline_cs2,
            "window": surface_window,
            "deformation_delta_cs2": surface_delta_cs2,
        }
    elif deformation.amplitude <= bounds.amplitude_min:
        location = bounds.lower_limiting_epsilon_mev_fm3
        failure = {
            "reason": "mechanical_stability_nonpositive_cs2",
            "amplitude_interval_violation": "at_or_below_open_lower_bound",
            "epsilon_mev_fm3": location,
            "pressure_mev_fm3": float(
                model.pressure_from_energy_density(location)
                + windowed_gaussian_pressure_primitive(location, deformation)
            ),
            "raw_cs2": raw_cs2_scalar(location),
        }
    elif deformation.amplitude > bounds.amplitude_max:
        location = bounds.upper_limiting_epsilon_mev_fm3
        failure = {
            "reason": "causality_superluminal_cs2",
            "amplitude_interval_violation": "above_closed_upper_bound",
            "epsilon_mev_fm3": location,
            "pressure_mev_fm3": float(
                model.pressure_from_energy_density(location)
                + windowed_gaussian_pressure_primitive(location, deformation)
            ),
            "raw_cs2": raw_cs2_scalar(location),
        }
    elif not stable:
        failure = {
            "reason": "mechanical_stability_nonpositive_cs2",
            "epsilon_mev_fm3": minimum_location,
            "pressure_mev_fm3": float(
                model.pressure_from_energy_density(minimum_location)
                + windowed_gaussian_pressure_primitive(
                    minimum_location, deformation
                )
            ),
            "raw_cs2": minimum,
        }
    elif not causal:
        failure = {
            "reason": "causality_superluminal_cs2",
            "epsilon_mev_fm3": maximum_location,
            "pressure_mev_fm3": float(
                model.pressure_from_energy_density(maximum_location)
                + windowed_gaussian_pressure_primitive(
                    maximum_location, deformation
                )
            ),
            "raw_cs2": maximum,
        }

    finite_pressure_values = raw_pressure[np.isfinite(raw_pressure)]
    report: dict[str, Any] = {
        "schema_version": CFL_RAW_GATE_SCHEMA_VERSION,
        "profile_id": CFL_DEFORMATION_PROFILE_ID,
        "profile_version": CFL_DEFORMATION_PROFILE_VERSION,
        "case_id": deformation.case_id,
        "case_sha256": deformation.case_sha256,
        "parameters": deformation.to_dict(),
        "baseline_parameter_set_id": FROZEN_PARAMETER_SET_ID,
        "baseline_parameter_set_sha256": FROZEN_PARAMETER_SET_SHA256,
        "evaluation_precedes_reconstruction_and_stellar_work": True,
        "independent_variable": "total_energy_density_mev_fm3",
        "complete_declared_domain_mev_fm3": [
            ENERGY_DENSITY_SURFACE_MEV_FM3,
            ENERGY_DENSITY_MAX_MEV_FM3,
        ],
        "dense_grid_points": int(len(epsilon)),
        "continuous_extremum_policy": (
            "geometry-aware complete-domain discovery plus bounded refinement; "
            "continuous amplitude interval authoritative"
        ),
        "pressure_primitive_policy": CFL_PRESSURE_PRIMITIVE_POLICY,
        "amplitude_bounds": bounds.to_dict(),
        "amplitude_interval_passed": amplitude_passed,
        "finite_values": finite,
        "positive_energy_density": positive_energy,
        "nonnegative_pressure_including_zero_surface": nonnegative_pressure,
        "raw_minimum_pressure_mev_fm3": (
            None
            if not finite_pressure_values.size
            else float(np.min(finite_pressure_values))
        ),
        "raw_maximum_pressure_mev_fm3": (
            None
            if not finite_pressure_values.size
            else float(np.max(finite_pressure_values))
        ),
        "raw_minimum_cs2": None if not finite else minimum,
        "raw_minimum_epsilon_mev_fm3": (
            None if not finite else minimum_location
        ),
        "raw_maximum_cs2": None if not finite else maximum,
        "raw_maximum_epsilon_mev_fm3": (
            None if not finite else maximum_location
        ),
        "mechanical_stability_margin": None if not finite else minimum,
        "causality_margin": None if not finite else 1.0 - maximum,
        "closed_upper_endpoint_one_ulp_roundoff_accepted": False,
        "surface": {
            "epsilon_mev_fm3": ENERGY_DENSITY_SURFACE_MEV_FM3,
            "pressure_mev_fm3": float(raw_pressure[0]),
            "raw_cs2": float(raw_cs2[0]),
            "window": surface_window,
            "deformation_delta_cs2": surface_delta_cs2,
            "preserved_exactly": surface_preserved,
        },
        "clipping_clamping_smoothing_posthoc_repair": "none",
        "extrapolation": "forbidden",
        "full_declared_domain_authoritative": True,
        "full_declared_domain_passed": passed,
        "first_failure": failure,
        "status": (
            "accepted_raw_local_physics_gate"
            if passed
            else "rejected_raw_local_physics_gate"
        ),
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report, epsilon.copy(), raw_cs2.copy()


__all__ = [
    "CFLAmplitudeBounds",
    "CFLWindowedDeformation",
    "CFL_AMPLITUDE_BOUNDS_SCHEMA_VERSION",
    "CFL_DEFORMATION_PROFILE_ID",
    "CFL_DEFORMATION_PROFILE_VERSION",
    "CFL_PRESSURE_PRIMITIVE_POLICY",
    "CFL_DEFORMATION_SCHEMA_VERSION",
    "CFL_RAW_GATE_SCHEMA_VERSION",
    "calculate_windowed_amplitude_bounds",
    "gaussian_profile",
    "raw_local_physics_gate",
    "smootherstep_window",
    "window_characterization",
    "windowed_gaussian_delta_cs2",
    "windowed_gaussian_pressure_primitive",
    "windowed_gaussian_shape",
]

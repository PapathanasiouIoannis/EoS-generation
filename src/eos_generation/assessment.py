"""Assessment for controlled BSk24 / BSk25 experiments."""

from __future__ import annotations

import math
import numpy as np
from .deformation import (
    BSk24WindowedDeformation,
    WINDOWED_GAUSSIAN_GENERATOR_ID,
    _mass_density_from_energy_density,
    _windowed_cs2,
    _windowed_pressure,
    _windowed_retained_state,
    windowed_gaussian_delta_cs2,
)
from dataclasses import dataclass
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize_scalar
from typing import Any, Callable, Mapping


DEFORMATION_SUPPORT_SIGMAS = 4.0


RAW_DISCOVERY_INTERVALS_PER_SCALE = 32


RETAINED_INTERVALS_PER_SCALE = 16


MAX_GEOMETRY_REFINEMENT_POINTS = 4096


def _analytical_pressure_derivative_certificate(
    epsilon: np.ndarray,
    pressure: np.ndarray,
    analytical_cs2: Callable[[float], float],
    spacing_certificate: Mapping[str, Any],
    *,
    intervals_per_scale: int,
) -> dict[str, Any]:
    """Certify tabulated pressure against the analytical deformation.

    Independent interval midpoints in every declared geometry section compare
    ``dP/d-epsilon`` from the pressure PCHIP with the analytical sound speed.
    The deterministic second-order rule is shared by raw assessment and the
    retained reconstruction so neither path can silently smooth a feature.
    """

    sections = spacing_certificate.get("sections")
    if not isinstance(sections, Mapping) or not sections:
        return {
            "status": "unresolved_analytical_tabulation",
            "failure_reason": "missing_resolution_sections",
        }
    epsilon_values = np.asarray(epsilon, dtype=float)
    pressure_values = np.asarray(pressure, dtype=float)
    if (
        epsilon_values.ndim != 1
        or pressure_values.shape != epsilon_values.shape
        or len(epsilon_values) < 2
        or not np.all(np.isfinite(epsilon_values))
        or not np.all(np.isfinite(pressure_values))
        or not np.all(np.diff(epsilon_values) > 0.0)
    ):
        return {
            "status": "unresolved_analytical_tabulation",
            "failure_reason": "invalid_pressure_tabulation",
        }
    probes = 0.5 * (epsilon_values[:-1] + epsilon_values[1:])
    mask = np.zeros(len(probes), dtype=bool)
    for record in sections.values():
        if not isinstance(record, Mapping):
            continue
        domain = record.get("domain_mev_fm3")
        if (
            isinstance(domain, list)
            and len(domain) == 2
            and all(isinstance(value, (int, float)) for value in domain)
        ):
            mask |= (probes >= float(domain[0])) & (probes <= float(domain[1]))
    selected = probes[mask]
    if not len(selected):
        return {
            "status": "unresolved_analytical_tabulation",
            "failure_reason": "no_independent_midpoint_probes",
        }
    try:
        interpolated_derivative = np.asarray(
            PchipInterpolator(
                epsilon_values, pressure_values, extrapolate=False
            ).derivative()(selected),
            dtype=float,
        )
        # Production analytical sound-speed callables accept ndarrays.  Use
        # that vectorized path for the thousands of independent midpoint
        # probes, while retaining the scalar fallback for tests and third-
        # party callables that implement only the historical scalar contract.
        try:
            analytical = np.asarray(analytical_cs2(selected), dtype=float)
        except (TypeError, ValueError, ArithmeticError):
            analytical = np.asarray(
                [float(analytical_cs2(value)) for value in selected],
                dtype=float,
            )
        if analytical.shape != selected.shape:
            analytical = np.asarray(
                [float(analytical_cs2(value)) for value in selected],
                dtype=float,
            )
    except (TypeError, ValueError, ArithmeticError) as exc:
        return {
            "status": "unresolved_analytical_tabulation",
            "failure_reason": f"{type(exc).__name__}:{exc}",
        }
    error = interpolated_derivative - analytical
    finite = bool(
        np.all(np.isfinite(interpolated_derivative))
        and np.all(np.isfinite(analytical))
        and np.all(np.isfinite(error))
    )
    scale = float(np.max(np.abs(analytical))) if finite else math.nan
    allowed = (
        max(
            512.0 * np.finfo(float).eps * max(1.0, scale),
            scale / float(intervals_per_scale**2),
        )
        if finite
        else math.nan
    )
    maximum_error = float(np.max(np.abs(error))) if finite else math.nan
    index = int(np.argmax(np.abs(error))) if finite else 0
    passed = bool(finite and maximum_error <= allowed)
    return {
        "status": (
            "resolved_analytical_tabulation"
            if passed
            else "unresolved_analytical_tabulation"
        ),
        "failure_reason": (
            None if passed else "analytical_midpoint_error_exceeds_resolution_rule"
        ),
        "probe_count": int(len(selected)),
        "comparison": ("analytical_cs2_vs_derivative_of_tabulated_pressure_PCHIP"),
        "maximum_absolute_error": maximum_error if finite else None,
        "epsilon_at_maximum_error_mev_fm3": (
            float(selected[index]) if finite else None
        ),
        "maximum_allowed_absolute_error": allowed if finite else None,
        "criterion": (
            "max(512*machine_epsilon*scale, analytical_cs2_scale/"
            "intervals_per_scale^2)"
        ),
        "intervals_per_scale": intervals_per_scale,
        "pressure_or_cs2_values_modified": False,
    }


def _retained_geometry_grid(
    base_grid: np.ndarray,
    *,
    amplitude: float,
    endpoint_mev_fm3: float,
    has_causal_crossing: bool,
    epsilon0_mev_fm3: float,
    sigma_mev_fm3: float,
    delta_mev_fm3: float,
    epsilon_match_mev_fm3: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct the shared case-specific production tabulation grid."""

    base = np.asarray(base_grid, dtype=float)
    endpoint = float(endpoint_mev_fm3)
    if amplitude == 0.0:
        if base.ndim != 1 or not len(base) or endpoint != float(base[-1]):
            return base.copy(), {
                "status": "unresolved_tabulation_resolution",
                "failure_reason": (
                    "zero_amplitude_endpoint_would_break_exact_identity"
                ),
            }
        return base.copy(), {
            "status": "resolved_exact_baseline_identity_grid",
            "failure_reason": None,
            "support_definition": "not_applicable_exact_zero_amplitude",
            "base_point_count": int(len(base)),
            "resolved_point_count": int(len(base)),
            "added_point_count": 0,
        }

    prefix = base[base < endpoint]
    if endpoint > float(base[-1]):
        terminal_spacing = float(base[-1] - base[-2])
        if not math.isfinite(terminal_spacing) or terminal_spacing <= 0.0:
            return base.copy(), {
                "status": "unresolved_tabulation_resolution",
                "failure_reason": "invalid_base_terminal_spacing",
            }
        extension_intervals = int(
            math.ceil((endpoint - float(base[-1])) / terminal_spacing)
        )
        extension = np.linspace(
            float(base[-1]), endpoint, extension_intervals + 1, dtype=float
        )[1:]
        candidate = np.concatenate((base, extension))
    else:
        candidate = np.concatenate((prefix, np.asarray((endpoint,), dtype=float)))
    grid, certificate = _geometry_aware_grid(
        candidate,
        epsilon0_mev_fm3=epsilon0_mev_fm3,
        sigma_mev_fm3=sigma_mev_fm3,
        delta_mev_fm3=delta_mev_fm3,
        epsilon_match_mev_fm3=epsilon_match_mev_fm3,
        epsilon_max_mev_fm3=endpoint,
        intervals_per_scale=RETAINED_INTERVALS_PER_SCALE,
        causal_endpoint_mev_fm3=(endpoint if has_causal_crossing else None),
    )
    result = dict(certificate)
    result["status"] = (
        "resolved_tabulation_resolution"
        if certificate.get("status") == "resolved_geometry_aware_sampling"
        else "unresolved_tabulation_resolution"
    )
    result["case_specific_endpoint_included_exactly"] = bool(
        len(grid) and grid[-1] == endpoint
    )
    if (
        result["status"] == "resolved_tabulation_resolution"
        and not result["case_specific_endpoint_included_exactly"]
    ):
        result["status"] = "unresolved_tabulation_resolution"
        result["failure_reason"] = "case_specific_endpoint_not_represented_exactly"
    return grid, result


def _meaningful_support_interval(
    *,
    epsilon0_mev_fm3: float,
    sigma_mev_fm3: float,
    epsilon_match_mev_fm3: float,
    epsilon_max_mev_fm3: float,
) -> tuple[float, float] | None:
    """Return the strict in-domain four-sigma support intersection.

    A point contact has zero measure and is therefore not meaningful support.
    The function is total for finite inputs and is shared by passive planning,
    raw assessment, and retained-grid certification.
    """

    values = np.asarray(
        (
            epsilon0_mev_fm3,
            sigma_mev_fm3,
            epsilon_match_mev_fm3,
            epsilon_max_mev_fm3,
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or sigma_mev_fm3 <= 0.0:
        return None
    if not epsilon_match_mev_fm3 < epsilon_max_mev_fm3:
        return None
    lower = max(
        float(epsilon_match_mev_fm3),
        float(epsilon0_mev_fm3) - DEFORMATION_SUPPORT_SIGMAS * float(sigma_mev_fm3),
    )
    upper = min(
        float(epsilon_max_mev_fm3),
        float(epsilon0_mev_fm3) + DEFORMATION_SUPPORT_SIGMAS * float(sigma_mev_fm3),
    )
    return (lower, upper) if lower < upper else None


def _section_grid(
    lower: float,
    upper: float,
    *,
    scale: float,
    intervals_per_scale: int,
) -> np.ndarray:
    """Return a bounded deterministic grid resolving one physical section."""

    if not lower < upper:
        return np.asarray((lower,), dtype=float)
    interval_count = max(
        1,
        int(math.ceil((upper - lower) / scale * intervals_per_scale)),
    )
    if interval_count + 1 > MAX_GEOMETRY_REFINEMENT_POINTS:
        return np.asarray((), dtype=float)
    return np.linspace(lower, upper, interval_count + 1, dtype=float)


def _geometry_aware_grid(
    base_grid: np.ndarray,
    *,
    epsilon0_mev_fm3: float,
    sigma_mev_fm3: float,
    delta_mev_fm3: float,
    epsilon_match_mev_fm3: float,
    epsilon_max_mev_fm3: float,
    intervals_per_scale: int,
    causal_endpoint_mev_fm3: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Augment a governed grid without changing its declared point counts.

    The added nodes resolve the four-sigma support, the smootherstep ramp and,
    when present, one width immediately below a case-specific causal endpoint.
    The returned certificate fails closed when floating-point representation or
    the bounded node budget cannot realize the rule.
    """

    base = np.asarray(base_grid, dtype=float)
    values = np.asarray(
        (
            epsilon0_mev_fm3,
            sigma_mev_fm3,
            delta_mev_fm3,
            epsilon_match_mev_fm3,
            epsilon_max_mev_fm3,
        ),
        dtype=float,
    )
    failure: str | None = None
    if (
        base.ndim != 1
        or len(base) < 2
        or not np.all(np.isfinite(base))
        or not np.all(np.diff(base) > 0.0)
    ):
        failure = "invalid_base_grid"
    elif (
        not np.all(np.isfinite(values))
        or sigma_mev_fm3 <= 0.0
        or delta_mev_fm3 <= 0.0
        or not epsilon_match_mev_fm3 < epsilon_max_mev_fm3
        or isinstance(intervals_per_scale, bool)
        or not isinstance(intervals_per_scale, int)
        or intervals_per_scale < 4
    ):
        failure = "invalid_geometry_or_resolution_rule"

    support = _meaningful_support_interval(
        epsilon0_mev_fm3=epsilon0_mev_fm3,
        sigma_mev_fm3=sigma_mev_fm3,
        epsilon_match_mev_fm3=epsilon_match_mev_fm3,
        epsilon_max_mev_fm3=epsilon_max_mev_fm3,
    )
    endpoint = (
        None if causal_endpoint_mev_fm3 is None else float(causal_endpoint_mev_fm3)
    )
    if (
        failure is None
        and endpoint is not None
        and (
            not math.isfinite(endpoint)
            or not epsilon_match_mev_fm3 < endpoint <= epsilon_max_mev_fm3
        )
    ):
        failure = "invalid_case_specific_causal_endpoint"
    if failure is None and support is None and endpoint is None:
        failure = "no_meaningful_four_sigma_support"

    sections: list[tuple[str, float, float, float]] = []
    if support is not None:
        sections.append(("four_sigma_support", *support, sigma_mev_fm3))
    ramp_upper = min(
        float(epsilon_max_mev_fm3),
        float(epsilon_match_mev_fm3) + float(delta_mev_fm3),
    )
    if epsilon_match_mev_fm3 < ramp_upper:
        sections.append(
            (
                "smootherstep_ramp",
                float(epsilon_match_mev_fm3),
                ramp_upper,
                float(delta_mev_fm3),
            )
        )
    if endpoint is not None:
        endpoint_lower = max(
            float(epsilon_match_mev_fm3),
            endpoint - float(sigma_mev_fm3),
        )
        if endpoint_lower < endpoint:
            sections.append(
                (
                    "causal_endpoint_band",
                    endpoint_lower,
                    endpoint,
                    float(sigma_mev_fm3),
                )
            )

    added: list[np.ndarray] = []
    section_reports: dict[str, dict[str, object]] = {}
    if failure is None:
        for name, lower, upper, scale in sections:
            allowed = float(scale / intervals_per_scale)
            existing_cover = np.unique(
                np.concatenate(
                    (
                        np.asarray((lower,), dtype=float),
                        base[(base > lower) & (base < upper)],
                        np.asarray((upper,), dtype=float),
                    )
                )
            )
            representation_allowance = 64.0 * math.ulp(max(abs(lower), abs(upper), 1.0))
            existing_sufficient = bool(
                len(existing_cover) >= 2
                and float(np.max(np.diff(existing_cover)))
                <= allowed + representation_allowance
            )
            if existing_sufficient:
                section = np.asarray((), dtype=float)
            else:
                section = _section_grid(
                    lower,
                    upper,
                    scale=scale,
                    intervals_per_scale=intervals_per_scale,
                )
                if not len(section):
                    failure = f"bounded_node_budget_exceeded:{name}"
                    break
                added.append(section)
            section_reports[name] = {
                "domain_mev_fm3": [float(lower), float(upper)],
                "scale_mev_fm3": float(scale),
                "requested_maximum_spacing_mev_fm3": float(scale / intervals_per_scale),
                "generated_point_count": int(len(section)),
                "governed_base_grid_already_sufficient": existing_sufficient,
            }

    if failure is None:
        declared = np.asarray(
            (
                epsilon_match_mev_fm3,
                epsilon_max_mev_fm3,
                *(() if endpoint is None else (endpoint,)),
            ),
            dtype=float,
        )
        declared = declared[
            (declared >= float(base[0])) & (declared <= float(base[-1]))
        ]
        candidates = np.concatenate((base, declared, *added))
        grid = np.unique(candidates)
        if (
            len(grid) < len(base)
            or not np.all(np.isfinite(grid))
            or not np.all(np.diff(grid) > 0.0)
        ):
            failure = "geometry_grid_not_representable"
    else:
        grid = base.copy()

    if failure is None:
        for name, lower, upper, scale in sections:
            region = np.unique(
                np.concatenate(
                    (
                        np.asarray((lower,), dtype=float),
                        grid[(grid > lower) & (grid < upper)],
                        np.asarray((upper,), dtype=float),
                    )
                )
            )
            if len(region) < 2:
                failure = f"section_not_representable:{name}"
                break
            maximum_spacing = float(np.max(np.diff(region)))
            allowed = float(scale / intervals_per_scale)
            representation_allowance = 64.0 * math.ulp(max(abs(lower), abs(upper), 1.0))
            if maximum_spacing > allowed + representation_allowance:
                failure = f"section_spacing_unresolved:{name}"
                break
            section_reports[name]["realized_maximum_spacing_mev_fm3"] = maximum_spacing
            section_reports[name]["resolved"] = True

    status = (
        "resolved_geometry_aware_sampling"
        if failure is None
        else "unresolved_geometry_aware_sampling"
    )
    return grid, {
        "status": status,
        "failure_reason": failure,
        "support_definition": "four_sigma_intersection_with_deformable_domain",
        "four_sigma_support_inside_selected_domain": support is not None,
        "support_sigmas": DEFORMATION_SUPPORT_SIGMAS,
        "intervals_per_scale": intervals_per_scale,
        "maximum_added_point_budget_per_section": (MAX_GEOMETRY_REFINEMENT_POINTS),
        "base_point_count": int(len(base)),
        "resolved_point_count": int(len(grid)),
        "added_point_count": int(len(grid) - len(base)),
        "sections": section_reports,
    }


def _log_windowed_gaussian_shape_scalar(
    epsilon_mev_fm3: float,
    *,
    epsilon0_mev_fm3: float,
    sigma_mev_fm3: float,
    delta_mev_fm3: float,
    epsilon_match_mev_fm3: float,
) -> float:
    """Return log(G W), with ``-inf`` representing the exact ``f=0`` set."""

    epsilon = float(epsilon_mev_fm3)
    if epsilon <= epsilon_match_mev_fm3:
        return -math.inf
    ramp_end = epsilon_match_mev_fm3 + delta_mev_fm3
    if epsilon < ramp_end:
        x = (epsilon - epsilon_match_mev_fm3) / delta_mev_fm3
        window = x**3 * (10.0 + x * (-15.0 + 6.0 * x))
        if window <= 0.0:
            return -math.inf
        log_window = math.log(window)
    else:
        log_window = 0.0
    z = (epsilon - epsilon0_mev_fm3) / sigma_mev_fm3
    return -0.5 * z * z + log_window


def _continuous_local_minima(
    grid: np.ndarray,
    sampled_values: np.ndarray,
    function,
    *,
    mandatory_points: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    """Discover sampled basins and refine each one with bounded minimization."""

    if len(grid) < 3 or len(grid) != len(sampled_values):
        raise ValueError("continuous-extremum discovery requires matching grids")
    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(sampled_values)):
        raise ValueError("continuous-extremum discovery values must be finite")
    indices = (
        np.flatnonzero(
            (sampled_values[1:-1] <= sampled_values[:-2])
            & (sampled_values[1:-1] <= sampled_values[2:])
        )
        + 1
    )
    candidates: list[tuple[float, float]] = [
        (float(sampled_values[0]), float(grid[0])),
        (float(sampled_values[-1]), float(grid[-1])),
    ]
    for index in indices:
        lower = float(grid[index - 1])
        upper = float(grid[index + 1])
        width = upper - lower

        def normalized_objective(coordinate: float) -> float:
            return float(function(lower + float(coordinate) * width))

        result = minimize_scalar(
            normalized_objective,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1.0e-13},
        )
        if not result.success or not math.isfinite(float(result.fun)):
            raise ValueError("bounded continuous-extremum refinement failed")
        location = lower + float(result.x) * width
        candidates.append((float(function(location)), location))
    for point in mandatory_points:
        if float(grid[0]) <= point <= float(grid[-1]):
            value = float(function(point))
            if math.isfinite(value):
                candidates.append((value, float(point)))
    candidates.sort(key=lambda item: (item[1], item[0]))
    deduplicated: list[tuple[float, float]] = []
    for value, location in candidates:
        if deduplicated and math.isclose(
            location,
            deduplicated[-1][1],
            rel_tol=0.0,
            abs_tol=2.0e-9,
        ):
            if value < deduplicated[-1][0]:
                deduplicated[-1] = (value, location)
        else:
            deduplicated.append((value, location))
    return tuple(deduplicated)


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
                "minimum_epsilon_mev_fm3": (self.baseline_minimum_epsilon_mev_fm3),
                "mechanical_stability_margin": self.baseline_minimum_cs2,
                "maximum_cs2": self.baseline_maximum_cs2,
                "maximum_epsilon_mev_fm3": (self.baseline_maximum_epsilon_mev_fm3),
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
    conditions ``0 < c_s^2 <= 1`` through the selected *direct* baseline endpoint.  Its
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
        baseline.eos.definition.retained_epsilon_max_mev_fm3,
        rel_tol=0.0,
        abs_tol=5.0e-12,
    )
    valid_anchor = bool(
        baseline.eos.definition.core_entry_epsilon_mev_fm3
        < epsilon_match
        < baseline.eos.definition.retained_epsilon_max_mev_fm3
    )
    if not valid_anchor or not retained_endpoint_matches:
        raise ValueError(
            f"amplitude bounds require the declared authoritative retained {baseline.eos.model_name} domain"
        )
    raw_geometry = np.asarray(
        [epsilon0_mev_fm3, sigma_mev_fm3, delta_mev_fm3], dtype=float
    )
    if not np.all(np.isfinite(raw_geometry)) or np.any(raw_geometry <= 0.0):
        raise ValueError(
            "ordinary raw-amplitude geometry requires finite positive "
            "epsilon0, sigma, and Delta"
        )
    if (
        _meaningful_support_interval(
            epsilon0_mev_fm3=float(epsilon0_mev_fm3),
            sigma_mev_fm3=float(sigma_mev_fm3),
            epsilon_match_mev_fm3=epsilon_match,
            epsilon_max_mev_fm3=epsilon_max,
        )
        is None
    ):
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
    if (
        any(array.ndim != 1 for array in arrays)
        or len({len(array) for array in arrays}) != 1
    ):
        raise ValueError(
            "baseline state arrays must be aligned one-dimensional profiles"
        )
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
        rho = baseline.eos.mass_density_from_energy_density(np.asarray(value, dtype=float))
        return float(baseline.eos.sound_speed_squared_from_mass_density(rho))

    baseline_values = np.asarray(
        baseline.eos.sound_speed_squared_from_mass_density(
            baseline.eos.mass_density_from_energy_density(full_grid)
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
    upper_grid = upper_grid[(upper_grid > epsilon_match) & (upper_grid <= epsilon_max)]

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
            baseline.eos.mass_density_from_energy_density(upper_grid)
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
        raise ValueError(
            "geometry has an empty or invalid admissible amplitude interval"
        )

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
        lower_candidate_extrema_mev_fm3=tuple(item[1] for item in lower_candidates),
        upper_candidate_extrema_mev_fm3=tuple(item[1] for item in upper_candidates),
        discovery_grid_points=int(len(upper_grid)),
    )


def _dense_gate_grid(
    baseline: BSk24ConsistentBaseline,
    *,
    epsilon_max_mev_fm3: float | None = None,
    lower_points: int = 16385,
    upper_points: int = 65537,
) -> np.ndarray:
    anchor = baseline.anchor.energy_density_mev_fm3
    upper_endpoint = (
        float(baseline.epsilon[-1])
        if epsilon_max_mev_fm3 is None
        else float(epsilon_max_mev_fm3)
    )
    lower = np.geomspace(baseline.epsilon[0], anchor, lower_points)
    upper = np.linspace(anchor, upper_endpoint, upper_points)
    return np.concatenate((lower[:-1], upper))


_RAW_GATE_BASELINE_CACHE: (
    tuple[
        BSk24ConsistentBaseline,
        int,
        int,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
    | None
) = None


def _cached_raw_gate_baseline_arrays(
    baseline: BSk24ConsistentBaseline,
    *,
    lower_points: int,
    upper_points: int,
    epsilon_max_mev_fm3: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _RAW_GATE_BASELINE_CACHE
    cached = _RAW_GATE_BASELINE_CACHE
    if (
        cached is not None
        and cached[0] is baseline
        and cached[1] == lower_points
        and cached[2] == upper_points
        and cached[3] == epsilon_max_mev_fm3
    ):
        return cached[4].copy(), cached[5].copy(), cached[6].copy()
    epsilon = _dense_gate_grid(
        baseline,
        lower_points=lower_points,
        upper_points=upper_points,
        epsilon_max_mev_fm3=epsilon_max_mev_fm3,
    )
    baseline_cs2 = np.asarray(
        baseline.eos.published_fit_sound_speed_squared_from_mass_density(
            baseline.eos.mass_density_from_energy_density(epsilon)
        ),
        dtype=float,
    )
    baseline_pressure = np.asarray(
        baseline.eos.published_fit_pressure_from_energy_density(epsilon),
        dtype=float,
    )
    _RAW_GATE_BASELINE_CACHE = (
        baseline,
        lower_points,
        upper_points,
        epsilon_max_mev_fm3,
        epsilon.copy(),
        baseline_cs2.copy(),
        baseline_pressure.copy(),
    )
    return epsilon, baseline_cs2, baseline_pressure


def _refined_extremum(
    grid: np.ndarray,
    values: np.ndarray,
    function,
    *,
    maximize: bool,
) -> tuple[float, float]:
    transformed = -values if maximize else values
    index = int(np.argmin(transformed))
    if index in (0, len(grid) - 1):
        return float(values[index]), float(grid[index])
    lower = float(grid[index - 1])
    upper = float(grid[index + 1])
    width = upper - lower

    def normalized_objective(coordinate: float) -> float:
        value = float(function(lower + float(coordinate) * width))
        return -value if maximize else value

    result = minimize_scalar(
        normalized_objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    if not result.success:
        raise ValueError("bounded continuous-extremum refinement failed")
    location = lower + float(result.x) * width
    value = float(function(location))
    return value, location


def _continuous_extrema(
    grid: np.ndarray,
    values: np.ndarray,
    function,
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
]:
    """Return every discovered continuous minimum and maximum candidate."""

    minima = _continuous_local_minima(
        grid,
        values,
        function,
        mandatory_points=(),
    )
    negated = _continuous_local_minima(
        grid,
        -values,
        lambda value: -float(function(value)),
        mandatory_points=(),
    )
    maxima = tuple((-value, location) for value, location in negated)
    return minima, maxima


def _first_causal_crossing(
    grid: np.ndarray,
    function,
    *,
    extrema_locations: tuple[float, ...],
    xtol: float,
    rtol: float,
    sampled_values: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Locate the first continuous contact with ``c_s^2 = 1``.

    Refined extrema are inserted before the ordered scan.  This exposes a
    narrow island even when both ordinary-grid endpoints are subluminal and
    also preserves an isolated tangential contact as an endpoint.
    """

    scan = np.unique(np.concatenate((grid, np.asarray(extrema_locations, dtype=float))))
    if sampled_values is None:
        values = np.asarray([float(function(value)) for value in scan], dtype=float)
    else:
        sampled = np.asarray(sampled_values, dtype=float)
        if sampled.shape != grid.shape:
            raise ValueError("sampled causal-scan values must match the grid")
        values = np.empty(scan.shape, dtype=float)
        grid_positions = np.searchsorted(scan, grid)
        if np.any(grid_positions >= len(scan)) or not np.array_equal(
            scan[grid_positions], grid
        ):
            raise ValueError("causal-scan grid was not preserved exactly")
        values[grid_positions] = sampled
        supplied = np.zeros(scan.shape, dtype=bool)
        supplied[grid_positions] = True
        missing = np.flatnonzero(~supplied)
        if len(missing):
            values[missing] = np.asarray(
                [float(function(scan[index])) for index in missing], dtype=float
            )
    contacts = np.flatnonzero(values >= 1.0)
    # A near-one refined maximum below one is not proof that an earlier
    # tangential contact is absent.  This ambiguity remains authoritative
    # even when a definite crossing exists later in an extended fit domain.
    contact_allowance = 512.0 * np.finfo(float).eps
    near_contacts = sorted(
        (
            (float(location), float(function(location)))
            for location in extrema_locations
            if 1.0 - contact_allowance <= float(function(location)) < 1.0
        ),
        key=lambda item: item[0],
    )
    first_definite_contact = (
        float(scan[int(contacts[0])]) if len(contacts) else math.inf
    )
    if near_contacts and near_contacts[0][0] < first_definite_contact:
        candidate, candidate_value = near_contacts[0]
        return {
            "status": "unresolved_near_tangential_causal_contact",
            "bracket_mev_fm3": None,
            "epsilon_mev_fm3": None,
            "cs2_at_endpoint": None,
            "candidate_extremum_epsilon_mev_fm3": candidate,
            "candidate_extremum_cs2": candidate_value,
            "refinement_method": "normalized_bounded_extremum_ambiguity",
            "contact_ulp_allowance": 512,
            "continuous_crossing_bracketed": False,
            "crossing_included_to_governed_tolerance": False,
            "cs2_values_modified": False,
            "root_xtol_mev_fm3": xtol,
            "root_rtol": rtol,
        }
    if not len(contacts):
        return None
    index = int(contacts[0])
    upper = float(scan[index])
    upper_value = float(values[index])
    if upper_value == 1.0:
        root = upper
        lower = float(scan[index - 1]) if index else upper
        method = "refined_extremum_or_grid_exact_contact"
    else:
        if index == 0:
            return {
                "status": "unresolved_causal_at_lower_boundary",
                "bracket_mev_fm3": None,
                "epsilon_mev_fm3": None,
                "cs2_at_endpoint": upper_value,
                "refinement_method": "unavailable",
            }
        lower = float(scan[index - 1])
        lower_value = float(values[index - 1])
        if not lower_value < 1.0 < upper_value:
            return {
                "status": "unresolved_first_causal_bracket",
                "bracket_mev_fm3": [lower, upper],
                "epsilon_mev_fm3": None,
                "cs2_at_endpoint": None,
                "refinement_method": "unavailable",
            }
        root_estimate = float(
            brentq(
                lambda value: float(function(value)) - 1.0,
                lower,
                upper,
                xtol=xtol,
                rtol=rtol,
            )
        )
        root_estimate_value = float(function(root_estimate))
        if root_estimate_value == 1.0:
            root = root_estimate
            representable_bracket = [root, root]
        else:
            causal_side = lower
            noncausal_side = upper
            if root_estimate_value < 1.0:
                causal_side = root_estimate
            else:
                noncausal_side = root_estimate
            # Brent's tolerance controls the continuous root estimate, but
            # the returned binary64 value can lie a few ulps above one.  Keep
            # a sign-preserving bracket and choose its nearest representable
            # causal-side value.  This selects the usable domain; it does not
            # alter, clip, or repair c_s^2.
            for _ in range(256):
                adjacent = math.nextafter(causal_side, noncausal_side)
                if adjacent >= noncausal_side:
                    break
                midpoint = causal_side + 0.5 * (noncausal_side - causal_side)
                if midpoint <= causal_side or midpoint >= noncausal_side:
                    break
                midpoint_value = float(function(midpoint))
                if midpoint_value == 1.0:
                    causal_side = midpoint
                    noncausal_side = midpoint
                    break
                if midpoint_value < 1.0:
                    causal_side = midpoint
                else:
                    noncausal_side = midpoint
            root = causal_side
            representable_bracket = [causal_side, noncausal_side]
        method = "brentq_estimate_plus_causal_side_float_refinement"
        bracket_width = representable_bracket[1] - representable_bracket[0]
        governed_root_tolerance = max(float(xtol), float(rtol) * abs(root_estimate))
        return {
            "status": "resolved_first_continuous_causal_crossing",
            "bracket_mev_fm3": representable_bracket,
            "epsilon_mev_fm3": root,
            "cs2_at_endpoint": float(function(root)),
            "first_noncausal_epsilon_mev_fm3": (
                representable_bracket[1]
                if representable_bracket[1] > representable_bracket[0]
                else None
            ),
            "first_noncausal_cs2": (
                float(function(representable_bracket[1]))
                if representable_bracket[1] > representable_bracket[0]
                else None
            ),
            "continuous_root_estimate_mev_fm3": root_estimate,
            "continuous_root_estimate_cs2": root_estimate_value,
            "refinement_method": method,
            "endpoint_selection": (
                "nearest_representable_causal_side_of_first_crossing"
            ),
            "continuous_crossing_bracketed": True,
            "representable_bracket_width_mev_fm3": bracket_width,
            "governed_root_tolerance_mev_fm3": governed_root_tolerance,
            "crossing_included_to_governed_tolerance": bool(
                bracket_width <= governed_root_tolerance
            ),
            "cs2_values_modified": False,
            "root_xtol_mev_fm3": xtol,
            "root_rtol": rtol,
        }
    return {
        "status": "resolved_first_continuous_causal_crossing",
        "bracket_mev_fm3": [lower, upper],
        "epsilon_mev_fm3": root,
        "cs2_at_endpoint": float(function(root)),
        "refinement_method": method,
        "endpoint_selection": "exact_representable_contact",
        "continuous_crossing_bracketed": True,
        "crossing_included_to_governed_tolerance": True,
        "cs2_values_modified": False,
        "root_xtol_mev_fm3": xtol,
        "root_rtol": rtol,
    }


def _failure_region(
    epsilon: float,
    *,
    epsilon_t: float,
    delta: float,
    epsilon0: float,
    sigma: float,
) -> str:
    if epsilon_t <= epsilon <= epsilon_t + delta:
        return "smootherstep_ramp"
    if abs(epsilon - epsilon0) <= sigma:
        return "Gaussian_center_region"
    if epsilon > epsilon0 + sigma:
        return "high_density_baseline_region"
    return "below_anchor_or_low_density_baseline_region"


def raw_local_physics_gate(
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24WindowedDeformation,
    *,
    dense_lower_points: int = 16385,
    dense_upper_points: int = 65537,
    amplitude_bounds: BSk24AmplitudeBounds | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Assess the complete raw proposal and select its first causal branch.

    Mechanical stability, finiteness, and positive pressure remain complete-
    proposal requirements.  A first continuous ``c_s^2 = 1`` contact is a
    valid case-specific endpoint rather than a reason to reject the proposal.
    No later return below one can extend the retained branch.
    """
    proposed_upper = (
        float(baseline.epsilon[-1])
        if deformation.amplitude == 0.0
        else float(baseline.eos.energy_density_max_published_fit_mev_fm3)
    )
    base_epsilon, _base_cs2, _base_pressure = _cached_raw_gate_baseline_arrays(
        baseline,
        lower_points=dense_lower_points,
        upper_points=dense_upper_points,
        epsilon_max_mev_fm3=proposed_upper,
    )
    epsilon_t = float(baseline.anchor.energy_density_mev_fm3)
    epsilon_max = float(base_epsilon[-1])
    if deformation.amplitude == 0.0:
        epsilon = base_epsilon
        resolution = {
            "status": "resolved_exact_zero_amplitude_identity_sampling",
            "failure_reason": None,
            "support_definition": "not_applicable_exact_zero_amplitude",
            "intervals_per_scale": None,
            "base_point_count": int(len(base_epsilon)),
            "resolved_point_count": int(len(base_epsilon)),
            "added_point_count": 0,
            "sections": {},
        }
    else:
        epsilon, resolution = _geometry_aware_grid(
            base_epsilon,
            epsilon0_mev_fm3=deformation.epsilon0_mev_fm3,
            sigma_mev_fm3=deformation.sigma_mev_fm3,
            delta_mev_fm3=deformation.delta_mev_fm3,
            epsilon_match_mev_fm3=epsilon_t,
            epsilon_max_mev_fm3=epsilon_max,
            intervals_per_scale=RAW_DISCOVERY_INTERVALS_PER_SCALE,
        )
    raw_resolution_resolved = bool(
        resolution["status"]
        in {
            "resolved_geometry_aware_sampling",
            "resolved_exact_zero_amplitude_identity_sampling",
        }
    )
    raw = np.asarray(_windowed_cs2(epsilon, baseline, deformation), dtype=float)
    raw_pressure = np.asarray(
        _windowed_pressure(epsilon, baseline, deformation), dtype=float
    )

    def raw_values(value: Any) -> float | np.ndarray:
        result = np.asarray(
            _windowed_cs2(np.asarray(value), baseline, deformation),
            dtype=float,
        )
        return float(result) if result.ndim == 0 else result

    def raw_scalar(value: float) -> float:
        return float(raw_values(value))

    raw_analytical_resolution = (
        {
            "status": "resolved_exact_baseline_identity_grid",
            "failure_reason": None,
            "probe_count": 0,
            "criterion": "not_applicable_exact_zero_amplitude",
            "pressure_or_cs2_values_modified": False,
        }
        if deformation.amplitude == 0.0
        else _analytical_pressure_derivative_certificate(
            epsilon,
            raw_pressure,
            raw_values,
            resolution,
            # Use the retained-table rule on the more finely sampled raw
            # geometry grid.  This tests the production accuracy contract
            # without making compact raw-gate test grids an undocumented
            # stricter profile.
            intervals_per_scale=RETAINED_INTERVALS_PER_SCALE,
        )
    )
    raw_pressure_cs2_consistent = bool(
        raw_analytical_resolution["status"]
        in {
            "resolved_analytical_tabulation",
            "resolved_exact_baseline_identity_grid",
        }
    )

    sampled_finite = bool(
        np.all(np.isfinite(epsilon))
        and np.all(np.isfinite(raw_pressure))
        and np.all(np.isfinite(raw))
    )
    finite = sampled_finite
    minima: tuple[tuple[float, float], ...] = ()
    maxima: tuple[tuple[float, float], ...] = ()
    if finite and raw_resolution_resolved:
        minima, maxima = _continuous_extrema(epsilon, raw, raw_scalar)
    minimum, minimum_epsilon = (
        min(minima, key=lambda item: item[0]) if minima else (math.nan, math.nan)
    )
    maximum, maximum_epsilon = (
        max(maxima, key=lambda item: item[0]) if maxima else (math.nan, math.nan)
    )
    finite = bool(
        finite
        and minima
        and maxima
        and math.isfinite(minimum)
        and math.isfinite(maximum)
    )
    support = _meaningful_support_interval(
        epsilon0_mev_fm3=deformation.epsilon0_mev_fm3,
        sigma_mev_fm3=deformation.sigma_mev_fm3,
        epsilon_match_mev_fm3=epsilon_t,
        epsilon_max_mev_fm3=epsilon_max,
    )
    relevant_minimum = math.nan
    relevant_minimum_epsilon = math.nan
    relevant_maximum = math.nan
    relevant_maximum_epsilon = math.nan
    if finite and support is not None:
        relevant_mask = (epsilon >= support[0]) & (epsilon <= support[1])
        relevant_epsilon = epsilon[relevant_mask]
        relevant_raw = raw[relevant_mask]
        if len(relevant_epsilon) >= 3:
            relevant_minima, relevant_maxima = _continuous_extrema(
                relevant_epsilon,
                relevant_raw,
                raw_scalar,
            )
            relevant_minimum, relevant_minimum_epsilon = min(
                relevant_minima, key=lambda item: item[0]
            )
            relevant_maximum, relevant_maximum_epsilon = max(
                relevant_maxima, key=lambda item: item[0]
            )
        elif deformation.amplitude != 0.0:
            finite = False

    positive_domain = bool(sampled_finite and np.all(epsilon > 0.0))
    positive_pressure = bool(sampled_finite and np.all(raw_pressure > 0.0))
    raw_pressure_differences = (
        np.diff(raw_pressure)
        if sampled_finite and len(raw_pressure) > 1
        else np.asarray([], dtype=float)
    )
    raw_pressure_monotone = bool(
        len(raw_pressure_differences) and np.all(raw_pressure_differences > 0.0)
    )
    first_nonmonotone_pressure_index = (
        int(np.flatnonzero(raw_pressure_differences <= 0.0)[0])
        if len(raw_pressure_differences) and np.any(raw_pressure_differences <= 0.0)
        else None
    )
    stable = bool(finite and minimum > 0.0)
    full_domain_causal = bool(finite and maximum <= 1.0)
    full_amplitude_interval_passed = bool(
        amplitude_bounds is None or amplitude_bounds.contains(deformation.amplitude)
    )
    lower_amplitude_bound_passed = bool(
        amplitude_bounds is None
        or deformation.amplitude > amplitude_bounds.amplitude_min
    )
    crossing: dict[str, Any] | None = None
    if finite and deformation.amplitude != 0.0:
        crossing = _first_causal_crossing(
            epsilon,
            raw_scalar,
            extrema_locations=tuple(item[1] for item in maxima),
            xtol=baseline.settings.causal_root_xtol_mev_fm3,
            rtol=baseline.settings.causal_root_rtol,
            sampled_values=raw,
        )
        if (
            crossing is not None
            and crossing.get("epsilon_mev_fm3") is not None
            and math.isclose(
                float(crossing["epsilon_mev_fm3"]),
                epsilon_max,
                rel_tol=0.0,
                abs_tol=baseline.settings.causal_root_xtol_mev_fm3,
            )
        ):
            crossing = None

    crossing_resolved = bool(
        crossing is not None
        and crossing.get("status") == "resolved_first_continuous_causal_crossing"
        and crossing.get("epsilon_mev_fm3") is not None
        and crossing.get("crossing_included_to_governed_tolerance") is True
        and crossing.get("cs2_at_endpoint") is not None
        and 0.0 < float(crossing["cs2_at_endpoint"]) <= 1.0
    )
    crossing_ambiguous = bool(
        crossing is not None
        and crossing.get("status") == "unresolved_near_tangential_causal_contact"
    )
    causal_endpoint_available = bool(
        crossing_resolved or (full_domain_causal and not crossing_ambiguous)
    )
    resolution_passed = bool(raw_resolution_resolved and causal_endpoint_available)
    retained_endpoint = (
        float(crossing["epsilon_mev_fm3"]) if crossing_resolved else epsilon_max
    )
    retained_endpoint_pressure = (
        (
            float(raw_pressure[-1])
            if deformation.amplitude == 0.0
            else float(
                _windowed_pressure(np.asarray(retained_endpoint), baseline, deformation)
            )
        )
        if finite and causal_endpoint_available
        else None
    )
    retained_endpoint_cs2 = (
        (
            float(raw[-1])
            if deformation.amplitude == 0.0
            else raw_scalar(retained_endpoint)
        )
        if finite and causal_endpoint_available
        else None
    )
    later_return_below_one = bool(
        crossing_resolved
        and np.any(raw[epsilon > float(crossing["epsilon_mev_fm3"])] < 1.0)
    )
    preproduction_hard_passed = bool(
        finite
        and positive_domain
        and positive_pressure
        and raw_pressure_monotone
        and raw_pressure_cs2_consistent
        and stable
        and lower_amplitude_bound_passed
    )
    retained_tabulation_resolution: dict[str, Any] = {
        "status": "not_evaluated_before_raw_gate_resolution",
        "failure_reason": "raw_gate_or_causal_endpoint_not_resolved",
        "preconstruction_only": True,
        "reconstruction_performed": False,
        "stellar_work_performed": False,
    }
    if preproduction_hard_passed and causal_endpoint_available:
        retained_grid, retained_tabulation_resolution = _retained_geometry_grid(
            baseline.epsilon,
            amplitude=deformation.amplitude,
            endpoint_mev_fm3=retained_endpoint,
            has_causal_crossing=crossing_resolved,
            epsilon0_mev_fm3=deformation.epsilon0_mev_fm3,
            sigma_mev_fm3=deformation.sigma_mev_fm3,
            delta_mev_fm3=deformation.delta_mev_fm3,
            epsilon_match_mev_fm3=epsilon_t,
        )
        retained_tabulation_resolution["preconstruction_only"] = True
        retained_tabulation_resolution["reconstruction_performed"] = False
        retained_tabulation_resolution["stellar_work_performed"] = False
        if retained_tabulation_resolution["status"] in {
            "resolved_tabulation_resolution",
            "resolved_exact_baseline_identity_grid",
        }:
            retained_pressure, retained_cs2 = _windowed_retained_state(
                retained_grid, baseline, deformation
            )
            retained_core_usable = bool(
                np.all(np.isfinite(retained_pressure))
                and np.all(np.isfinite(retained_cs2))
                and np.all(retained_pressure > 0.0)
                and np.all(retained_cs2 > 0.0)
                and np.all(np.diff(retained_pressure) > 0.0)
                and (
                    (
                        crossing_resolved
                        and np.all(retained_cs2[:-1] < 1.0)
                        and retained_cs2[-1] <= 1.0
                    )
                    or (not crossing_resolved and np.all(retained_cs2 <= 1.0))
                )
            )
            retained_analytical_resolution = (
                {
                    "status": "resolved_exact_baseline_identity_grid",
                    "failure_reason": None,
                    "probe_count": 0,
                    "criterion": "not_applicable_exact_zero_amplitude",
                    "pressure_or_cs2_values_modified": False,
                }
                if deformation.amplitude == 0.0
                else _analytical_pressure_derivative_certificate(
                    retained_grid,
                    retained_pressure,
                    raw_values,
                    retained_tabulation_resolution,
                    intervals_per_scale=RETAINED_INTERVALS_PER_SCALE,
                )
            )
            retained_tabulation_resolution["analytical_comparison"] = (
                retained_analytical_resolution
            )
            retained_tabulation_resolution["retained_core_state_usable"] = (
                retained_core_usable
            )
            if not retained_core_usable:
                retained_tabulation_resolution["status"] = (
                    "unresolved_tabulation_resolution"
                )
                retained_tabulation_resolution["failure_reason"] = (
                    "invalid_retained_analytical_core_state"
                )
            elif retained_analytical_resolution["status"] not in {
                "resolved_analytical_tabulation",
                "resolved_exact_baseline_identity_grid",
            }:
                retained_tabulation_resolution["status"] = (
                    "unresolved_tabulation_resolution"
                )
                retained_tabulation_resolution["failure_reason"] = (
                    retained_analytical_resolution.get("failure_reason")
                )
    production_resolution_passed = bool(
        retained_tabulation_resolution["status"]
        in {
            "resolved_tabulation_resolution",
            "resolved_exact_baseline_identity_grid",
        }
    )
    hard_proposal_passed = bool(
        preproduction_hard_passed and production_resolution_passed
    )
    selected_resolution_certified = bool(
        resolution_passed
        and raw_pressure_monotone
        and raw_pressure_cs2_consistent
        and production_resolution_passed
    )
    selected_domain_passed = bool(
        hard_proposal_passed
        and causal_endpoint_available
        and selected_resolution_certified
    )
    full_domain_passed = bool(
        hard_proposal_passed
        and full_domain_causal
        and full_amplitude_interval_passed
        and resolution_passed
    )
    failure: dict[str, Any] | None = None
    unresolved = False
    if not raw_resolution_resolved:
        unresolved = True
        failure = {
            "reason": "unresolved_geometry_aware_continuous_assessment",
            "detail": resolution.get("failure_reason"),
            "first_failing_epsilon_mev_fm3": None,
            "first_failing_cs2": None,
        }
    elif not finite:
        invalid = ~np.isfinite(epsilon) | ~np.isfinite(raw_pressure) | ~np.isfinite(raw)
        invalid_indices = np.flatnonzero(invalid)
        first_index = int(invalid_indices[0]) if len(invalid_indices) else None
        first = None if first_index is None else float(epsilon[first_index])
        failure = {
            "reason": "nonfinite_or_unresolved_raw_continuous_state",
            "first_failing_epsilon_mev_fm3": first,
            "first_failing_pressure_mev_fm3": (
                float(raw_pressure[first_index])
                if first_index is not None
                and math.isfinite(float(raw_pressure[first_index]))
                else None
            ),
            "first_failing_cs2": (
                float(raw[first_index])
                if first_index is not None and math.isfinite(float(raw[first_index]))
                else None
            ),
        }
    elif not positive_domain:
        failure = {
            "reason": "nonpositive_retained_energy_density",
            "first_failing_epsilon_mev_fm3": float(
                epsilon[np.flatnonzero(epsilon <= 0.0)[0]]
            ),
            "first_failing_cs2": None,
        }
    elif not positive_pressure:
        index = int(np.flatnonzero(raw_pressure <= 0.0)[0])
        failure = {
            "reason": "nonpositive_raw_pressure",
            "first_failing_epsilon_mev_fm3": float(epsilon[index]),
            "first_failing_pressure_mev_fm3": float(raw_pressure[index]),
            "first_failing_cs2": float(raw[index]),
        }
    elif not lower_amplitude_bound_passed:
        failure = {
            "reason": "amplitude_at_or_below_mechanical_stability_lower_bound",
            "first_failing_epsilon_mev_fm3": amplitude_bounds.lower_limiting_epsilon_mev_fm3,
            "first_failing_cs2": None,
        }
    elif not stable:
        invalid = raw <= 0.0
        target = 0.0
        reason = "mechanical_stability_nonpositive_cs2"
        sampled_invalid = np.flatnonzero(invalid)
        if sampled_invalid.size:
            index = int(sampled_invalid[0])
            first: float | None = float(epsilon[index])
            refined = first
            if index > 0:
                lo = float(epsilon[index - 1])
                hi = float(epsilon[index])
                if (raw[index - 1] - target) * (raw[index] - target) <= 0.0:
                    refined = float(
                        brentq(
                            lambda value: raw_scalar(value) - target,
                            lo,
                            hi,
                            xtol=1.0e-12,
                            rtol=1.0e-12,
                        )
                    )
        else:
            # The bounded continuous-extremum refinement can find a strict
            # violation between dense grid samples.  That is authoritative
            # rejection evidence even though there is no failing sampled
            # index to report.  Preserve the refined extremum location and
            # mark the sampled location unavailable instead of indexing an
            # empty array or weakening the strict gate.
            first = None
            refined = float(minimum_epsilon)
        failure = {
            "reason": reason,
            "first_failing_epsilon_mev_fm3": refined,
            "first_failing_sample_epsilon_mev_fm3": first,
            "first_failing_cs2": raw_scalar(refined),
        }
    elif not raw_pressure_monotone:
        unresolved = True
        index = first_nonmonotone_pressure_index
        failure = {
            "reason": "unresolved_raw_pressure_cs2_consistency",
            "detail": "analytical_pressure_not_strictly_increasing_despite_positive_cs2",
            "first_failing_epsilon_mev_fm3": (
                float(epsilon[index + 1]) if index is not None else None
            ),
            "first_failing_pressure_mev_fm3": (
                float(raw_pressure[index + 1]) if index is not None else None
            ),
            "first_failing_cs2": (float(raw[index + 1]) if index is not None else None),
            "first_nonpositive_pressure_difference_mev_fm3": (
                float(raw_pressure_differences[index]) if index is not None else None
            ),
        }
    elif not raw_pressure_cs2_consistent:
        unresolved = True
        failure = {
            "reason": "unresolved_raw_pressure_cs2_consistency",
            "detail": raw_analytical_resolution.get("failure_reason"),
            "first_failing_epsilon_mev_fm3": (
                raw_analytical_resolution.get("epsilon_at_maximum_error_mev_fm3")
            ),
            "first_failing_pressure_mev_fm3": None,
            "first_failing_cs2": None,
            "maximum_pressure_derivative_cs2_error": (
                raw_analytical_resolution.get("maximum_absolute_error")
            ),
        }
    elif not causal_endpoint_available or not resolution_passed:
        unresolved = True
        failure = {
            "reason": "unresolved_first_continuous_causal_crossing",
            "detail": None if crossing is None else crossing.get("status"),
            "first_failing_epsilon_mev_fm3": None,
            "first_failing_cs2": None,
        }
    elif not production_resolution_passed:
        unresolved = True
        failure = {
            "reason": "unresolved_retained_tabulation_resolution",
            "detail": retained_tabulation_resolution.get("failure_reason"),
            "first_failing_epsilon_mev_fm3": (
                retained_tabulation_resolution.get("analytical_comparison", {}).get(
                    "epsilon_at_maximum_error_mev_fm3"
                )
            ),
            "first_failing_cs2": None,
        }
    if failure is not None and failure.get("first_failing_epsilon_mev_fm3") is not None:
        failure["region"] = _failure_region(
            failure["first_failing_epsilon_mev_fm3"],
            epsilon_t=epsilon_t,
            delta=deformation.delta_mev_fm3,
            epsilon0=deformation.epsilon0_mev_fm3,
            sigma=deformation.sigma_mev_fm3,
        )
    center_in_declared_domain = bool(
        float(epsilon[0]) <= deformation.epsilon0_mev_fm3 <= float(epsilon[-1])
    )
    status = (
        "accepted_raw_local_physics_gate"
        if selected_domain_passed
        else (
            "unresolved_raw_local_physics_gate"
            if unresolved
            else "rejected_raw_local_physics_gate"
        )
    )

    def finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(float(value)) else None

    report = {
        "case_id": deformation.case_id,
        "generator_id": WINDOWED_GAUSSIAN_GENERATOR_ID,
        "parameters": deformation.to_dict(),
        "evaluation_precedes_pressure_reconstruction_and_TOV": True,
        "continuous_extremum_policy": (
            "governed dense grid plus deterministic geometry-scale nodes "
            "followed by all-basin bounded refinement"
        ),
        "dense_grid_points": int(len(epsilon)),
        "production_profile_points": int(len(baseline.epsilon)),
        "continuous_resolution_certificate": resolution,
        "retained_tabulation_resolution_certificate": (retained_tabulation_resolution),
        "complete_proposed_retained_domain_mev_fm3": [
            float(epsilon[0]),
            float(epsilon[-1]),
        ],
        "finite_values": sampled_finite,
        "positive_energy_density": positive_domain,
        "positive_pressure": positive_pressure,
        "raw_pressure_reconstruction_certificate": {
            "status": (
                "resolved_strictly_increasing_raw_pressure"
                if raw_pressure_monotone and raw_pressure_cs2_consistent
                else "unresolved_raw_pressure_cs2_consistency"
            ),
            "complete_declared_domain_assessed": True,
            "minimum_forward_pressure_difference_mev_fm3": (
                float(np.min(raw_pressure_differences))
                if len(raw_pressure_differences)
                else None
            ),
            "first_nonmonotone_interval_index": (first_nonmonotone_pressure_index),
            "pressure_values_modified": False,
            "analytical_derivative_comparison": raw_analytical_resolution,
        },
        "raw_minimum_cs2": finite_or_none(minimum),
        "raw_minimum_epsilon_mev_fm3": finite_or_none(minimum_epsilon),
        "raw_maximum_cs2": finite_or_none(maximum),
        "raw_maximum_epsilon_mev_fm3": finite_or_none(maximum_epsilon),
        "mechanical_stability_margin": finite_or_none(minimum),
        "causality_margin": finite_or_none(1.0 - maximum),
        "amplitude_bound_semantics": (
            None
            if amplitude_bounds is None
            else {
                "A_min": amplitude_bounds.amplitude_min,
                "A_max": amplitude_bounds.amplitude_max,
                "interval": "(A_min, A_max]",
                "lower_endpoint_open": True,
                "upper_endpoint_closed": True,
                "amplitude": deformation.amplitude,
                "full_direct_domain_passed": full_amplitude_interval_passed,
                "mechanical_stability_lower_bound_passed": (
                    lower_amplitude_bound_passed
                ),
                "upper_bound_is_nonblocking_when_a_first_causal_endpoint_is_resolved": True,
            }
        ),
        "deformation_relevant_domain_definition": (
            "strict four-sigma intersection with the deformable domain"
        ),
        "deformation_relevant_domain_mev_fm3": (
            None if support is None else [support[0], support[1]]
        ),
        "deformation_region_minimum_cs2": finite_or_none(relevant_minimum),
        "deformation_region_minimum_epsilon_mev_fm3": (
            finite_or_none(relevant_minimum_epsilon)
        ),
        "deformation_region_maximum_cs2": finite_or_none(relevant_maximum),
        "deformation_region_maximum_epsilon_mev_fm3": (
            finite_or_none(relevant_maximum_epsilon)
        ),
        "raw_cs2_at_epsilon0": (
            raw_scalar(deformation.epsilon0_mev_fm3)
            if center_in_declared_domain
            else None
        ),
        "raw_cs2_at_epsilon0_status": (
            "evaluated"
            if center_in_declared_domain
            else "center_outside_declared_raw_domain"
        ),
        "delta_cs2_at_epsilon0": float(
            windowed_gaussian_delta_cs2(
                deformation.epsilon0_mev_fm3,
                deformation,
                epsilon_t_mev_fm3=epsilon_t,
            )
        ),
        "strictly_monotone_pressure_implied": bool(stable and raw_pressure_monotone),
        "anchor_tail_magnitude": abs(
            float(
                windowed_gaussian_delta_cs2(
                    epsilon_t,
                    deformation,
                    epsilon_t_mev_fm3=epsilon_t,
                )
            )
        ),
        "anchor_tail_status": "exactly_zero",
        "clipping_clamping_smoothing_repair": "none",
        "extrapolation": "forbidden",
        "complete_raw_proposal_assessed": True,
        "complete_raw_proposal_mechanically_stable": stable,
        "complete_raw_pressure_numerically_usable": bool(
            raw_pressure_monotone and raw_pressure_cs2_consistent
        ),
        "complete_raw_proposal_causal_through_direct_endpoint": (full_domain_causal),
        "complete_raw_proposal_causal_through_declared_assessment_endpoint": (
            full_domain_causal
        ),
        "declared_assessment_endpoint": (
            f"direct_{baseline.eos.matter_model}_causal_endpoint"
            if deformation.amplitude == 0.0
            else f"published_{baseline.eos.matter_model}_fit_endpoint"
        ),
        "retained_domain": {
            "policy": "prefix_through_first_continuous_cs2_equals_one",
            "endpoint_reason": (
                "first_continuous_causal_crossing"
                if crossing_resolved
                else (
                    (
                        f"direct_{baseline.eos.matter_model}_causal_endpoint"
                        if deformation.amplitude == 0.0
                        else f"published_{baseline.eos.matter_model}_fit_endpoint"
                    )
                    if causal_endpoint_available
                    else "unavailable_unresolved_continuous_assessment"
                )
            ),
            "epsilon_min_mev_fm3": float(epsilon[0]),
            "epsilon_max_mev_fm3": retained_endpoint,
            "pressure_max_mev_fm3": retained_endpoint_pressure,
            "cs2_at_endpoint": retained_endpoint_cs2,
            "first_causal_crossing": crossing,
            "later_return_below_one_outside_usable_branch": (later_return_below_one),
            "resolution_certified": selected_resolution_certified,
            "passed": selected_domain_passed,
        },
        "full_retained_domain_authoritative": True,
        "full_retained_domain_passed": full_domain_passed,
        "selected_retained_domain_authoritative": True,
        "selected_retained_domain_passed": selected_domain_passed,
        "first_failure": failure,
        "status": status,
    }
    return report, epsilon, raw

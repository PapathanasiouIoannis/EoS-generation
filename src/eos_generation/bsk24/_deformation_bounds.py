"""Continuous-extremum helpers for physical deformation bounds."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar


# The existing deformation diagnostics define the physically relevant
# Gaussian tail through four nominal widths on either side of the center.
# Reuse that convention for passive overlap checks and deterministic
# geometry-aware sampling rather than introducing a second support policy.
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
            mask |= (probes >= float(domain[0])) & (
                probes <= float(domain[1])
            )
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
            None
            if passed
            else "analytical_midpoint_error_exceeds_resolution_rule"
        ),
        "probe_count": int(len(selected)),
        "comparison": (
            "analytical_cs2_vs_derivative_of_tabulated_pressure_PCHIP"
        ),
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
        if (
            base.ndim != 1
            or not len(base)
            or endpoint != float(base[-1])
        ):
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
        candidate = np.concatenate(
            (prefix, np.asarray((endpoint,), dtype=float))
        )
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
    if result["status"] == "resolved_tabulation_resolution" and not result[
        "case_specific_endpoint_included_exactly"
    ]:
        result["status"] = "unresolved_tabulation_resolution"
        result["failure_reason"] = (
            "case_specific_endpoint_not_represented_exactly"
        )
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
        float(epsilon0_mev_fm3)
        - DEFORMATION_SUPPORT_SIGMAS * float(sigma_mev_fm3),
    )
    upper = min(
        float(epsilon_max_mev_fm3),
        float(epsilon0_mev_fm3)
        + DEFORMATION_SUPPORT_SIGMAS * float(sigma_mev_fm3),
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
        None
        if causal_endpoint_mev_fm3 is None
        else float(causal_endpoint_mev_fm3)
    )
    if failure is None and endpoint is not None and (
        not math.isfinite(endpoint)
        or not epsilon_match_mev_fm3 < endpoint <= epsilon_max_mev_fm3
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
            representation_allowance = 64.0 * math.ulp(
                max(abs(lower), abs(upper), 1.0)
            )
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
                "requested_maximum_spacing_mev_fm3": float(
                    scale / intervals_per_scale
                ),
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
            representation_allowance = 64.0 * math.ulp(
                max(abs(lower), abs(upper), 1.0)
            )
            if maximum_spacing > allowed + representation_allowance:
                failure = f"section_spacing_unresolved:{name}"
                break
            section_reports[name]["realized_maximum_spacing_mev_fm3"] = (
                maximum_spacing
            )
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
        "maximum_added_point_budget_per_section": (
            MAX_GEOMETRY_REFINEMENT_POINTS
        ),
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
    indices = np.flatnonzero(
        (sampled_values[1:-1] <= sampled_values[:-2])
        & (sampled_values[1:-1] <= sampled_values[2:])
    ) + 1
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

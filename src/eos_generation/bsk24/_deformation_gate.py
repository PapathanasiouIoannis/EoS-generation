"""Complete-domain raw physical gating for windowed BSk24 proposals."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from eos_generation.bsk24._deformation_bounds import (
    RAW_DISCOVERY_INTERVALS_PER_SCALE,
    RETAINED_INTERVALS_PER_SCALE,
    _analytical_pressure_derivative_certificate,
    _continuous_local_minima,
    _geometry_aware_grid,
    _meaningful_support_interval,
    _retained_geometry_grid,
)
from eos_generation.bsk24._deformation_core import (
    WINDOWED_GAUSSIAN_GENERATOR_ID,
    _windowed_cs2,
    _windowed_pressure,
    windowed_gaussian_delta_cs2,
    windowed_gaussian_pressure_primitive,
)
from eos_generation.bsk24.reconstruction import (
    BSk24ConsistentBaseline,
    _mass_density_from_energy_density,
)

if TYPE_CHECKING:
    from eos_generation.bsk24.deformation import (
        BSk24AmplitudeBounds,
        BSk24WindowedDeformation,
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


# One packet evaluates many amplitudes on the same retained baseline and
# governed dense grid.  Keep only the most recent baseline/grid tuple so the
# analytical C4 pressure and sound speed are evaluated once, while every
# amplitude still receives independent arrays and its own strict continuous
# extrema/refinement.  The single-entry design bounds memory and cannot leak
# results between distinct baseline objects.
_RAW_GATE_BASELINE_CACHE: tuple[
    BSk24ConsistentBaseline,
    int,
    int,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
] | None = None


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
            _mass_density_from_energy_density(epsilon)
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

    scan = np.unique(
        np.concatenate((grid, np.asarray(extrema_locations, dtype=float)))
    )
    if sampled_values is None:
        values = np.asarray([float(function(value)) for value in scan], dtype=float)
    else:
        sampled = np.asarray(sampled_values, dtype=float)
        if sampled.shape != grid.shape:
            raise ValueError("sampled causal-scan values must match the grid")
        values = np.empty(scan.shape, dtype=float)
        grid_positions = np.searchsorted(scan, grid)
        if (
            np.any(grid_positions >= len(scan))
            or not np.array_equal(scan[grid_positions], grid)
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
            if 1.0 - contact_allowance
            <= float(function(location))
            < 1.0
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
                midpoint = causal_side + 0.5 * (
                    noncausal_side - causal_side
                )
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
        governed_root_tolerance = max(
            float(xtol), float(rtol) * abs(root_estimate)
        )
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
    raw = np.asarray(
        _windowed_cs2(epsilon, baseline, deformation), dtype=float
    )
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
        min(minima, key=lambda item: item[0])
        if minima
        else (math.nan, math.nan)
    )
    maximum, maximum_epsilon = (
        max(maxima, key=lambda item: item[0])
        if maxima
        else (math.nan, math.nan)
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
        len(raw_pressure_differences)
        and np.all(raw_pressure_differences > 0.0)
    )
    first_nonmonotone_pressure_index = (
        int(np.flatnonzero(raw_pressure_differences <= 0.0)[0])
        if len(raw_pressure_differences)
        and np.any(raw_pressure_differences <= 0.0)
        else None
    )
    stable = bool(finite and minimum > 0.0)
    full_domain_causal = bool(finite and maximum <= 1.0)
    full_amplitude_interval_passed = bool(
        amplitude_bounds is None
        or amplitude_bounds.contains(deformation.amplitude)
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
        and crossing.get("status")
        == "resolved_first_continuous_causal_crossing"
        and crossing.get("epsilon_mev_fm3") is not None
        and crossing.get("crossing_included_to_governed_tolerance") is True
        and crossing.get("cs2_at_endpoint") is not None
        and 0.0 < float(crossing["cs2_at_endpoint"]) <= 1.0
    )
    crossing_ambiguous = bool(
        crossing is not None
        and crossing.get("status")
        == "unresolved_near_tangential_causal_contact"
    )
    causal_endpoint_available = bool(
        crossing_resolved
        or (full_domain_causal and not crossing_ambiguous)
    )
    resolution_passed = bool(
        raw_resolution_resolved and causal_endpoint_available
    )
    retained_endpoint = (
        float(crossing["epsilon_mev_fm3"])
        if crossing_resolved
        else epsilon_max
    )
    retained_endpoint_pressure = (
        float(
            _windowed_pressure(
                np.asarray(retained_endpoint), baseline, deformation
            )
        )
        if finite and causal_endpoint_available
        else None
    )
    retained_endpoint_cs2 = (
        raw_scalar(retained_endpoint)
        if finite and causal_endpoint_available
        else None
    )
    later_return_below_one = bool(
        crossing_resolved
        and np.any(
            raw[
                epsilon
                > float(crossing["epsilon_mev_fm3"])
            ]
            < 1.0
        )
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
        retained_grid, retained_tabulation_resolution = (
            _retained_geometry_grid(
                baseline.epsilon,
                amplitude=deformation.amplitude,
                endpoint_mev_fm3=retained_endpoint,
                has_causal_crossing=crossing_resolved,
                epsilon0_mev_fm3=deformation.epsilon0_mev_fm3,
                sigma_mev_fm3=deformation.sigma_mev_fm3,
                delta_mev_fm3=deformation.delta_mev_fm3,
                epsilon_match_mev_fm3=epsilon_t,
            )
        )
        retained_tabulation_resolution["preconstruction_only"] = True
        retained_tabulation_resolution["reconstruction_performed"] = False
        retained_tabulation_resolution["stellar_work_performed"] = False
        if retained_tabulation_resolution["status"] in {
            "resolved_tabulation_resolution",
            "resolved_exact_baseline_identity_grid",
        }:
            retained_pressure = np.asarray(
                _windowed_pressure(
                    retained_grid, baseline, deformation
                ),
                dtype=float,
            )
            retained_cs2 = np.asarray(
                _windowed_cs2(retained_grid, baseline, deformation),
                dtype=float,
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
                    or (
                        not crossing_resolved
                        and np.all(retained_cs2 <= 1.0)
                    )
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
        invalid = (
            ~np.isfinite(epsilon)
            | ~np.isfinite(raw_pressure)
            | ~np.isfinite(raw)
        )
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
                if first_index is not None
                and math.isfinite(float(raw[first_index]))
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
            "first_failing_cs2": (
                float(raw[index + 1]) if index is not None else None
            ),
            "first_nonpositive_pressure_difference_mev_fm3": (
                float(raw_pressure_differences[index])
                if index is not None
                else None
            ),
        }
    elif not raw_pressure_cs2_consistent:
        unresolved = True
        failure = {
            "reason": "unresolved_raw_pressure_cs2_consistency",
            "detail": raw_analytical_resolution.get("failure_reason"),
            "first_failing_epsilon_mev_fm3": (
                raw_analytical_resolution.get(
                    "epsilon_at_maximum_error_mev_fm3"
                )
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
                retained_tabulation_resolution.get(
                    "analytical_comparison", {}
                ).get("epsilon_at_maximum_error_mev_fm3")
            ),
            "first_failing_cs2": None,
        }
    if (
        failure is not None
        and failure.get("first_failing_epsilon_mev_fm3") is not None
    ):
        failure["region"] = _failure_region(
            failure["first_failing_epsilon_mev_fm3"],
            epsilon_t=epsilon_t,
            delta=deformation.delta_mev_fm3,
            epsilon0=deformation.epsilon0_mev_fm3,
            sigma=deformation.sigma_mev_fm3,
        )
    center_in_declared_domain = bool(
        float(epsilon[0])
        <= deformation.epsilon0_mev_fm3
        <= float(epsilon[-1])
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
        "retained_tabulation_resolution_certificate": (
            retained_tabulation_resolution
        ),
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
            "first_nonmonotone_interval_index": (
                first_nonmonotone_pressure_index
            ),
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
        "strictly_monotone_pressure_implied": bool(
            stable and raw_pressure_monotone
        ),
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
        "complete_raw_proposal_causal_through_direct_endpoint": (
            full_domain_causal
        ),
        "complete_raw_proposal_causal_through_declared_assessment_endpoint": (
            full_domain_causal
        ),
        "declared_assessment_endpoint": (
            "direct_bsk24_causal_endpoint"
            if deformation.amplitude == 0.0
            else "published_bsk24_fit_endpoint"
        ),
        "retained_domain": {
            "policy": "prefix_through_first_continuous_cs2_equals_one",
            "endpoint_reason": (
                "first_continuous_causal_crossing"
                if crossing_resolved
                else (
                    (
                        "direct_bsk24_causal_endpoint"
                        if deformation.amplitude == 0.0
                        else "published_bsk24_fit_endpoint"
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
            "later_return_below_one_outside_usable_branch": (
                later_return_below_one
            ),
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

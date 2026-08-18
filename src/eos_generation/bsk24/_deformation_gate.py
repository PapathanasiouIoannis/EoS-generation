"""Complete-domain raw physical gating for windowed BSk24 proposals."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from eos_generation.bsk24._deformation_core import (
    WINDOWED_GAUSSIAN_GENERATOR_ID,
    _windowed_cs2,
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
    lower_points: int = 16385,
    upper_points: int = 65537,
) -> np.ndarray:
    anchor = baseline.anchor.energy_density_mev_fm3
    lower = np.geomspace(baseline.epsilon[0], anchor, lower_points)
    upper = np.linspace(anchor, baseline.epsilon[-1], upper_points)
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
    np.ndarray,
    np.ndarray,
    np.ndarray,
] | None = None


def _cached_raw_gate_baseline_arrays(
    baseline: BSk24ConsistentBaseline,
    *,
    lower_points: int,
    upper_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _RAW_GATE_BASELINE_CACHE
    cached = _RAW_GATE_BASELINE_CACHE
    if (
        cached is not None
        and cached[0] is baseline
        and cached[1] == lower_points
        and cached[2] == upper_points
    ):
        return cached[3].copy(), cached[4].copy(), cached[5].copy()
    epsilon = _dense_gate_grid(
        baseline,
        lower_points=lower_points,
        upper_points=upper_points,
    )
    rho = _mass_density_from_energy_density(epsilon)
    baseline_cs2 = np.asarray(
        baseline.eos.sound_speed_squared_from_mass_density(rho), dtype=float
    )
    baseline_pressure = np.asarray(
        baseline.eos.pressure_from_mass_density(rho), dtype=float
    )
    _RAW_GATE_BASELINE_CACHE = (
        baseline,
        lower_points,
        upper_points,
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
    result = minimize_scalar(
        (lambda x: -function(x)) if maximize else function,
        bounds=(float(grid[index - 1]), float(grid[index + 1])),
        method="bounded",
        options={"xatol": 1.0e-11},
    )
    value = float(function(float(result.x)))
    return value, float(result.x)


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
    """Evaluate the raw proposal on the complete retained domain.

    Optional ``amplitude_bounds`` supply an independently calculated
    continuous physical interval.  The dense profile and local refinement are
    retained as diagnostics and failure-location evidence.
    """
    epsilon, raw, raw_pressure = _cached_raw_gate_baseline_arrays(
        baseline,
        lower_points=dense_lower_points,
        upper_points=dense_upper_points,
    )
    if deformation.amplitude != 0.0:
        raw += np.asarray(
            windowed_gaussian_delta_cs2(
                epsilon,
                deformation,
                epsilon_t_mev_fm3=(
                    baseline.anchor.energy_density_mev_fm3
                ),
            ),
            dtype=float,
        )
        raw_pressure += np.asarray(
            windowed_gaussian_pressure_primitive(
                epsilon,
                deformation,
                epsilon_t_mev_fm3=(
                    baseline.anchor.energy_density_mev_fm3
                ),
            ),
            dtype=float,
        )

    def raw_scalar(value: float) -> float:
        return float(
            _windowed_cs2(np.asarray(value), baseline, deformation)
        )

    minimum, minimum_epsilon = _refined_extremum(
        epsilon, raw, raw_scalar, maximize=False
    )
    maximum, maximum_epsilon = _refined_extremum(
        epsilon, raw, raw_scalar, maximize=True
    )
    epsilon_t = baseline.anchor.energy_density_mev_fm3
    relevant_upper = min(
        float(epsilon[-1]),
        deformation.epsilon0_mev_fm3 + 4.0 * deformation.sigma_mev_fm3,
    )
    relevant_mask = (epsilon >= epsilon_t) & (epsilon <= relevant_upper)
    relevant_epsilon = epsilon[relevant_mask]
    relevant_raw = raw[relevant_mask]
    relevant_minimum, relevant_minimum_epsilon = _refined_extremum(
        relevant_epsilon, relevant_raw, raw_scalar, maximize=False
    )
    relevant_maximum, relevant_maximum_epsilon = _refined_extremum(
        relevant_epsilon, relevant_raw, raw_scalar, maximize=True
    )
    finite = bool(
        np.all(np.isfinite(epsilon))
        and np.all(np.isfinite(raw_pressure))
        and np.all(np.isfinite(raw))
        and math.isfinite(minimum)
        and math.isfinite(maximum)
    )
    positive_domain = bool(finite and np.all(epsilon > 0.0))
    positive_pressure = bool(finite and np.all(raw_pressure > 0.0))
    stable = bool(finite and minimum > 0.0)
    causal = bool(finite and maximum <= 1.0)
    amplitude_interval_passed = bool(
        amplitude_bounds is None
        or amplitude_bounds.contains(deformation.amplitude)
    )
    full_domain_passed = bool(
        finite
        and positive_domain
        and positive_pressure
        and stable
        and causal
        and amplitude_interval_passed
    )
    failure: dict[str, Any] | None = None
    if not finite:
        invalid = (
            ~np.isfinite(epsilon)
            | ~np.isfinite(raw_pressure)
            | ~np.isfinite(raw)
        )
        first_index = int(np.flatnonzero(invalid)[0])
        first = float(epsilon[first_index])
        failure = {
            "reason": "nonfinite_raw_sound_speed",
            "first_failing_epsilon_mev_fm3": first,
            "first_failing_pressure_mev_fm3": (
                float(raw_pressure[first_index])
                if math.isfinite(float(raw_pressure[first_index]))
                else None
            ),
            "first_failing_cs2": (
                float(raw[first_index])
                if math.isfinite(float(raw[first_index]))
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
    elif not amplitude_interval_passed:
        failure = {
            "reason": "amplitude_outside_open_lower_closed_upper_bounds",
            "first_failing_epsilon_mev_fm3": (
                amplitude_bounds.lower_limiting_epsilon_mev_fm3
                if deformation.amplitude <= amplitude_bounds.amplitude_min
                else amplitude_bounds.upper_limiting_epsilon_mev_fm3
            ),
            "first_failing_cs2": None,
        }
    elif not stable or not causal:
        invalid = (raw <= 0.0) if not stable else (raw > 1.0)
        target = 0.0 if not stable else 1.0
        reason = (
            "mechanical_stability_nonpositive_cs2"
            if not stable
            else "causality_superluminal_cs2"
        )
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
            refined = float(minimum_epsilon if not stable else maximum_epsilon)
        failure = {
            "reason": reason,
            "first_failing_epsilon_mev_fm3": refined,
            "first_failing_sample_epsilon_mev_fm3": first,
            "first_failing_cs2": raw_scalar(refined),
        }
    if failure is not None:
        failure["region"] = _failure_region(
            failure["first_failing_epsilon_mev_fm3"],
            epsilon_t=epsilon_t,
            delta=deformation.delta_mev_fm3,
            epsilon0=deformation.epsilon0_mev_fm3,
            sigma=deformation.sigma_mev_fm3,
        )
    report = {
        "case_id": deformation.case_id,
        "generator_id": WINDOWED_GAUSSIAN_GENERATOR_ID,
        "parameters": deformation.to_dict(),
        "evaluation_precedes_pressure_reconstruction_and_TOV": True,
        "continuous_extremum_policy": (
            "independent dense grid exceeding production nodes followed by bounded local refinement"
        ),
        "dense_grid_points": int(len(epsilon)),
        "production_profile_points": int(len(baseline.epsilon)),
        "complete_proposed_retained_domain_mev_fm3": [
            float(epsilon[0]),
            float(epsilon[-1]),
        ],
        "finite_values": finite,
        "positive_energy_density": positive_domain,
        "positive_pressure": positive_pressure,
        "raw_minimum_cs2": minimum,
        "raw_minimum_epsilon_mev_fm3": minimum_epsilon,
        "raw_maximum_cs2": maximum,
        "raw_maximum_epsilon_mev_fm3": maximum_epsilon,
        "mechanical_stability_margin": minimum,
        "causality_margin": 1.0 - maximum,
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
                "passed": amplitude_interval_passed,
            }
        ),
        "deformation_relevant_domain_definition": (
            "anchor through min(retained endpoint, epsilon0+4*sigma)"
        ),
        "deformation_relevant_domain_mev_fm3": [
            epsilon_t,
            relevant_upper,
        ],
        "deformation_region_minimum_cs2": relevant_minimum,
        "deformation_region_minimum_epsilon_mev_fm3": (
            relevant_minimum_epsilon
        ),
        "deformation_region_maximum_cs2": relevant_maximum,
        "deformation_region_maximum_epsilon_mev_fm3": (
            relevant_maximum_epsilon
        ),
        "raw_cs2_at_epsilon0": raw_scalar(
            deformation.epsilon0_mev_fm3
        ),
        "delta_cs2_at_epsilon0": float(
            windowed_gaussian_delta_cs2(
                deformation.epsilon0_mev_fm3,
                deformation,
                epsilon_t_mev_fm3=epsilon_t,
            )
        ),
        "strictly_monotone_pressure_implied": stable,
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
        "full_retained_domain_authoritative": True,
        "full_retained_domain_passed": full_domain_passed,
        "first_failure": failure,
        "status": (
            "accepted_raw_local_physics_gate"
            if full_domain_passed
            else "rejected_raw_local_physics_gate"
        ),
    }
    return report, epsilon, raw

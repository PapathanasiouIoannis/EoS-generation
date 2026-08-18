"""Bounded stellar diagnostics for the analytical BSk24 fit family.

The helpers in this module retain the shared TOV background and tidal
equations.  They add post-processing for dense radial profiles, total baryon
number, support-aware interpolation, paired-amplitude response, and
turning-point refinement evidence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from eos_generation._internal.config import DEFAULT_CONFIG
from eos_generation.stellar import tov as tov_core
from eos_generation.bsk24.baseline import (
    C_LIGHT_CM_S,
    MEV_FM3_TO_ERG_CM3,
    NEUTRON_REST_ENERGY_MEV,
)
from eos_generation.bsk24.deformation import (
    BSk24WindowedDeformation,
    smootherstep_window,
    windowed_gaussian_delta_cs2,
)


FM3_TO_KM3 = 1.0e54
G_CGS = 6.67430e-8
KM_TO_CM = 1.0e5
MEV_TO_ERG = MEV_FM3_TO_ERG_CM3 * 1.0e-39
SOLAR_MASS_G_FROM_PROJECT_LENGTH = (
    DEFAULT_CONFIG.units.solar_mass_length_km
    * KM_TO_CM
    * C_LIGHT_CM_S**2
    / G_CGS
)
NEUTRON_REST_MASS_G = NEUTRON_REST_ENERGY_MEV * MEV_TO_ERG / C_LIGHT_CM_S**2


def _validate_pressure_profile_monotonicity(
    pressures: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Reject radial pressure reversals larger than the local ODE error scale.

    The TOV equation makes pressure nonincreasing, but the RK dense-output
    polynomial is not itself monotonicity preserving.  Close to the finite
    surface-pressure cutoff it can therefore produce a small upward step even
    when every accepted integration node is decreasing.  Keep those raw
    samples unchanged when the step is bounded by the combined error scale of
    the adjacent values; larger reversals remain a hard diagnostic failure.
    """
    differences = np.diff(pressures)
    if not np.any(differences > 0.0):
        return
    local_scale = float(atol) + float(rtol) * np.maximum(
        np.abs(pressures[:-1]), np.abs(pressures[1:])
    )
    # Each difference contains the numerical error of two dense-output
    # evaluations, hence the sum of their local error scales.
    allowance = 2.0 * local_scale
    violations = differences > allowance
    if np.any(violations):
        maximum_increase = float(np.max(differences[violations]))
        maximum_allowance = float(np.max(allowance[violations]))
        raise ValueError(
            "diagnostic pressure profile has a nonincreasing violation larger "
            "than the local ODE error scale: "
            f"increase={maximum_increase:.17g}, "
            f"allowance={maximum_allowance:.17g}"
        )


def _validate_nondecreasing_profile(
    values: np.ndarray,
    *,
    rtol: float,
    atol: float,
    quantity: str,
) -> tuple[float, float, int, float]:
    """Validate raw solver samples without repairing bounded interpolation noise."""
    effective_rtol = float(rtol)
    effective_atol = float(atol)
    if (
        not math.isfinite(effective_rtol)
        or not math.isfinite(effective_atol)
        or effective_rtol < 0.0
        or effective_atol < 0.0
    ):
        raise ValueError("profile solver tolerances must be finite and nonnegative")
    differences = np.diff(values)
    reversals = differences < 0.0
    allowance = 2.0 * (
        effective_atol
        + effective_rtol
        * np.maximum(np.abs(values[:-1]), np.abs(values[1:]))
    )
    violations = reversals & (-differences > allowance)
    if np.any(violations):
        maximum_drop = float(np.max(-differences[violations]))
        maximum_allowance = float(np.max(allowance[violations]))
        raise ValueError(
            f"{quantity} has a nondecreasing violation larger than the local "
            "ODE error scale: "
            f"drop={maximum_drop:.17g}, "
            f"allowance={maximum_allowance:.17g}"
        )
    return (
        effective_rtol,
        effective_atol,
        int(np.count_nonzero(reversals)),
        (
            float(np.max(-differences[reversals]))
            if np.any(reversals)
            else 0.0
        ),
    )


def pressure_profile_from_solved_star(
    eos_callable: Any,
    star: Any,
    *,
    settings: Any,
    rtol: float,
    atol: float,
) -> tuple[float, ...]:
    """Recover P(r) with the shared segmented background integrator.

    ``solve_star`` intentionally exposes only the long-standing radius and
    mass profile interface.  This diagnostic-only sampler repeats the same
    background integration and evaluates its dense solution at those exact
    radii.  It changes neither the TOV equations nor their tolerances.  Exact
    initial/event states are used at the endpoints to prevent dense-output
    roundoff from evaluating one ulp outside the declared EoS domain.
    """
    effective_rtol = float(rtol)
    effective_atol = float(atol)
    if (
        not math.isfinite(effective_rtol)
        or not math.isfinite(effective_atol)
        or effective_rtol <= 0.0
        or effective_atol <= 0.0
    ):
        raise ValueError("TOV tolerances must be finite and positive")
    central_pressure = float(star.central_pressure)
    central_energy_density = float(star.central_energy_density)
    try:
        discontinuities = tov_core._resolved_discontinuities(eos_callable)
    except (TypeError, ValueError):
        discontinuities = ()
    try:
        segments, _ = tov_core._integrate_background(
            eos_callable,
            central_pressure,
            central_energy_density,
            discontinuities,
            settings=settings,
            rtol=effective_rtol,
            atol=effective_atol,
        )
    except (ValueError, RuntimeError, ArithmeticError):
        if not discontinuities:
            raise
        segments, _ = tov_core._integrate_background(
            eos_callable,
            central_pressure,
            central_energy_density,
            (),
            settings=settings,
            rtol=effective_rtol,
            atol=effective_atol,
        )

    radii = np.asarray(star.radius_profile, dtype=float)
    pressures = np.empty_like(radii)
    segment_index = 0
    for index, radius in enumerate(radii):
        while (
            segment_index < len(segments) - 1
            and radius > segments[segment_index].radius_end
        ):
            segment_index += 1
        pressures[index] = float(segments[segment_index].solution.sol(radius)[1])
    pressures[0] = float(segments[0].solution.y[1, 0])
    pressures[-1] = float(segments[-1].event_state[1])
    if not np.all(np.isfinite(pressures)):
        raise ValueError("diagnostic pressure profile is nonfinite")
    _validate_pressure_profile_monotonicity(
        pressures,
        rtol=effective_rtol,
        atol=effective_atol,
    )
    return tuple(float(value) for value in pressures)


@dataclass(frozen=True)
class BaryonIntegralResult:
    """Total baryon number and binding measures for one stellar profile."""

    baryon_number: float
    baryonic_mass_msun: float
    gravitational_mass_msun: float
    mass_excess_msun: float
    binding_energy_erg: float
    fractional_binding: float
    center_correction_baryons: float
    radial_integral_baryons: float
    minimum_metric_factor: float
    integration_method: str
    profile_points: int
    profile_solver_rtol: float
    profile_solver_atol: float
    bounded_mass_reversal_count: int
    maximum_bounded_mass_reversal_msun: float
    raw_mass_profile_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TurningPointEstimate:
    """Derivative-based turning-point estimate from one pressure grid."""

    status: str
    pchip_pressure_mev_fm3: float | None
    pchip_mass_msun: float | None
    pchip_energy_density_mev_fm3: float | None
    quadratic_pressure_mev_fm3: float | None
    quadratic_mass_msun: float | None
    method_pressure_difference_mev_fm3: float | None
    method_mass_difference_msun: float | None
    positive_secants_before: int
    negative_secants_after: int
    derivative_sign_change_bracket_mev_fm3: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.derivative_sign_change_bracket_mev_fm3 is not None:
            value["derivative_sign_change_bracket_mev_fm3"] = list(
                self.derivative_sign_change_bracket_mev_fm3
            )
        return value


def interpolate_within_common_support(
    masses_msun: Any,
    values: Any,
    target_masses_msun: Any,
) -> np.ndarray:
    """Interpolate a single increasing branch without extrapolation."""
    masses = np.asarray(masses_msun, dtype=float)
    observable = np.asarray(values, dtype=float)
    targets = np.asarray(target_masses_msun, dtype=float)
    if masses.ndim != 1 or observable.ndim != 1 or masses.size != observable.size:
        raise ValueError("mass and observable arrays must be aligned one-dimensional data")
    if masses.size < 2 or not np.all(np.isfinite(masses)) or not np.all(
        np.isfinite(observable)
    ):
        raise ValueError("interpolation inputs must contain at least two finite rows")
    if not np.all(np.diff(masses) > 0.0):
        raise ValueError("mass support must be strictly increasing on one branch")
    if not np.all(np.isfinite(targets)):
        raise ValueError("target masses must be finite")
    if np.any(targets < masses[0]) or np.any(targets > masses[-1]):
        raise ValueError("target mass is outside the common stable support")
    return np.asarray(
        PchipInterpolator(masses, observable, extrapolate=False)(targets),
        dtype=float,
    )


def baryon_number_from_profile(
    radius_km: Any,
    mass_msun: Any,
    baryon_density_fm3: Any,
    *,
    central_baryon_density_fm3: float,
    gravitational_mass_msun: float,
    solar_mass_length_km: float = DEFAULT_CONFIG.units.solar_mass_length_km,
    solver_rtol: float = 0.0,
    solver_atol: float = 0.0,
) -> BaryonIntegralResult:
    """Integrate total baryon number with the relativistic proper-volume factor.

    The solver begins at a finite Taylor radius.  The omitted central ball is
    included analytically with its supplied central density.  Every sampled
    surface value is evaluated at the configured in-domain stellar boundary;
    no surface extrapolation is performed.
    """
    radius = np.asarray(radius_km, dtype=float)
    mass = np.asarray(mass_msun, dtype=float)
    density = np.asarray(baryon_density_fm3, dtype=float)
    if (
        radius.ndim != 1
        or mass.ndim != 1
        or density.ndim != 1
        or not (radius.size == mass.size == density.size)
        or radius.size < 3
    ):
        raise ValueError("radial baryon integral requires aligned profiles of at least 3 points")
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(mass)) or not np.all(
        np.isfinite(density)
    ):
        raise ValueError("radial baryon integral inputs must be finite")
    if radius[0] <= 0.0 or not np.all(np.diff(radius) > 0.0):
        raise ValueError("radius profile must be positive and strictly increasing")
    if np.any(mass <= 0.0):
        raise ValueError("enclosed mass must be positive")
    (
        effective_rtol,
        effective_atol,
        bounded_mass_reversal_count,
        maximum_bounded_mass_reversal,
    ) = _validate_nondecreasing_profile(
        mass,
        rtol=solver_rtol,
        atol=solver_atol,
        quantity="enclosed mass",
    )
    if np.any(density <= 0.0):
        raise ValueError("baryon number density must be positive")
    central_density = float(central_baryon_density_fm3)
    stellar_mass = float(gravitational_mass_msun)
    if not math.isfinite(central_density) or central_density <= 0.0:
        raise ValueError("central baryon density must be finite and positive")
    if not math.isfinite(stellar_mass) or stellar_mass <= 0.0:
        raise ValueError("gravitational mass must be finite and positive")
    metric = 1.0 - 2.0 * float(solar_mass_length_km) * mass / radius
    if np.any(metric <= 0.0) or not np.all(np.isfinite(metric)):
        raise ValueError("proper-volume metric factor must remain finite and positive")
    density_km3 = density * FM3_TO_KM3
    integrand = density_km3 * radius**2 / np.sqrt(metric)
    radial = 4.0 * math.pi * float(simpson(integrand, x=radius))
    center = (
        4.0
        * math.pi
        * central_density
        * FM3_TO_KM3
        * radius[0] ** 3
        / 3.0
    )
    baryon_number = center + radial
    baryonic_mass = baryon_number * NEUTRON_REST_MASS_G / SOLAR_MASS_G_FROM_PROJECT_LENGTH
    excess = baryonic_mass - stellar_mass
    binding_erg = excess * SOLAR_MASS_G_FROM_PROJECT_LENGTH * C_LIGHT_CM_S**2
    fractional = excess / baryonic_mass
    return BaryonIntegralResult(
        baryon_number=float(baryon_number),
        baryonic_mass_msun=float(baryonic_mass),
        gravitational_mass_msun=stellar_mass,
        mass_excess_msun=float(excess),
        binding_energy_erg=float(binding_erg),
        fractional_binding=float(fractional),
        center_correction_baryons=float(center),
        radial_integral_baryons=float(radial),
        minimum_metric_factor=float(np.min(metric)),
        integration_method="analytic_center_ball_plus_scipy_simpson_sampled_profile",
        profile_points=int(radius.size),
        profile_solver_rtol=effective_rtol,
        profile_solver_atol=effective_atol,
        bounded_mass_reversal_count=bounded_mass_reversal_count,
        maximum_bounded_mass_reversal_msun=maximum_bounded_mass_reversal,
        raw_mass_profile_preserved=True,
    )


def odd_even_response(
    positive: float,
    negative: float,
    zero: float,
    *,
    amplitude: float,
    numerical_envelope: float,
) -> dict[str, float | bool]:
    """Return paired odd/even response and the central local slope."""
    values = np.asarray(
        [positive, negative, zero, amplitude, numerical_envelope], dtype=float
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("odd/even response inputs must be finite")
    if amplitude <= 0.0 or numerical_envelope < 0.0:
        raise ValueError("amplitude must be positive and envelope nonnegative")
    odd = 0.5 * (positive - negative)
    even = 0.5 * (positive + negative - 2.0 * zero)
    denominator = max(numerical_envelope, np.finfo(float).eps * max(1.0, abs(zero)))
    return {
        "amplitude": float(amplitude),
        "odd_response": float(odd),
        "even_response": float(even),
        "local_central_slope_per_unit_A": float(odd / amplitude),
        "odd_over_numerical_envelope": float(abs(odd) / denominator),
        "even_over_numerical_envelope": float(abs(even) / denominator),
        "odd_resolved": bool(abs(odd) > denominator),
        "even_resolved": bool(abs(even) > denominator),
    }


def support_tolerance_from_observable_envelopes(
    local_slopes: Mapping[str, float],
    numerical_envelopes: Mapping[str, float],
) -> dict[str, Any]:
    """Convert measured observable envelopes into an amplitude-like tail limit."""
    equivalent: dict[str, float] = {}
    for name, slope_value in local_slopes.items():
        if name not in numerical_envelopes:
            raise KeyError(f"missing numerical envelope for {name}")
        slope = abs(float(slope_value))
        envelope = float(numerical_envelopes[name])
        if not math.isfinite(slope) or not math.isfinite(envelope):
            raise ValueError("support-tolerance inputs must be finite")
        if slope <= 0.0 or envelope < 0.0:
            raise ValueError("support-tolerance slopes must be nonzero and envelopes nonnegative")
        equivalent[name] = envelope / slope
    controlling = min(equivalent, key=equivalent.get)
    return {
        "tau_support_delta_cs2": float(equivalent[controlling]),
        "controlling_observable": controlling,
        "amplitude_equivalent_limits": equivalent,
        "definition": (
            "minimum measured observable envelope divided by the paired local "
            "slope near A=0 for R1.4, k2_1.4, and Lambda1.4"
        ),
    }


def select_outside_support_center(
    *,
    baseline,
    amplitude: float,
    sigma_mev_fm3: float,
    delta_mev_fm3: float,
    epsilon_c_1p4_mev_fm3: float,
    epsilon_c_massive_mev_fm3: float,
    tau_support: float,
    nonnegligible_fraction: float = 0.1,
) -> dict[str, Any]:
    """Select the smallest center satisfying the null-tail and massive-star gates."""
    amplitude_abs = abs(float(amplitude))
    sigma = float(sigma_mev_fm3)
    epsilon_c_1p4 = float(epsilon_c_1p4_mev_fm3)
    epsilon_c_massive = float(epsilon_c_massive_mev_fm3)
    tau = float(tau_support)
    if amplitude_abs <= 0.0 or sigma <= 0.0 or tau < 0.0:
        raise ValueError("outside-support controls require positive amplitude/sigma and nonnegative tau")
    epsilon_t = float(baseline.anchor.energy_density_mev_fm3)
    upper = float(baseline.energy_density_max_mev_fm3)

    def tail(center: float, epsilon: float) -> float:
        deformation = BSk24WindowedDeformation(
            "outside_support_selection",
            amplitude_abs,
            center,
            sigma,
            delta_mev_fm3,
        )
        return abs(
            float(
                windowed_gaussian_delta_cs2(
                    epsilon,
                    deformation,
                    epsilon_t_mev_fm3=epsilon_t,
                )
            )
        )

    lower_center = max(epsilon_c_1p4, epsilon_t + delta_mev_fm3)
    if tail(lower_center, epsilon_c_1p4) <= tau:
        selected = lower_center
    else:
        if tail(upper, epsilon_c_1p4) > tau:
            return {
                "status": "unavailable_no_center_meets_1p4_tail_gate",
                "selected_center_mev_fm3": None,
                "tau_support": tau,
            }
        selected = brentq(
            lambda center: tail(center, epsilon_c_1p4) - tau,
            lower_center,
            upper,
            xtol=1.0e-10,
            rtol=4.0 * np.finfo(float).eps,
        )
    realized_peak = amplitude_abs
    # A star samples every material energy density from its lower boundary to
    # its center.  If the selected Gaussian center lies below epsilon_c, the
    # star reaches the realized peak even though the tail at epsilon_c itself
    # can already be small.
    massive_tail = (
        realized_peak
        if selected <= epsilon_c_massive
        else tail(selected, epsilon_c_massive)
    )
    reaches = bool(massive_tail >= nonnegligible_fraction * realized_peak)
    if not reaches:
        return {
            "status": "unavailable_no_clean_control_with_massive_overlap",
            "selected_center_mev_fm3": float(selected),
            "tail_at_1p4": float(tail(selected, epsilon_c_1p4)),
            "tail_at_massive_star": float(massive_tail),
            "nonnegligible_threshold": float(nonnegligible_fraction * realized_peak),
            "tau_support": tau,
        }
    return {
        "status": "selected",
        "selected_center_mev_fm3": float(selected),
        "tail_at_1p4": float(tail(selected, epsilon_c_1p4)),
        "tail_at_massive_star": float(massive_tail),
        "nonnegligible_threshold": float(nonnegligible_fraction * realized_peak),
        "tau_support": tau,
        "smallest_center_rule": True,
        "retained_domain_upper_mev_fm3": upper,
    }


def matched_area_amplitude(
    target_amplitude: float,
    target_unit_shape_area: float,
    comparison_unit_shape_area: float,
) -> float:
    """Return the amplitude matching one signed integrated deformation."""
    values = np.asarray(
        [target_amplitude, target_unit_shape_area, comparison_unit_shape_area],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or target_unit_shape_area <= 0.0 or comparison_unit_shape_area <= 0.0:
        raise ValueError("matched-area inputs must be finite with positive unit areas")
    return float(target_amplitude * target_unit_shape_area / comparison_unit_shape_area)


def matched_area_radius_resolution(
    radius_by_case: Mapping[str, float],
    same_case_envelope_by_case: Mapping[str, float],
) -> dict[str, Any]:
    """Classify a matched-area radius spread with same-case uncertainty.

    The response is the maximum-minus-minimum radius across the matched-width
    family.  Its paired numerical envelope is the conservative sum of the
    independently measured same-case envelopes for the cases defining that
    spread.  The packet's existing rule is retained: a response is resolved
    only when it exceeds its numerical envelope.
    """
    if set(radius_by_case) != set(same_case_envelope_by_case):
        raise ValueError("radius values and same-case envelopes must share exact keys")
    if len(radius_by_case) < 2:
        raise ValueError("matched-area resolution requires at least two cases")
    radius = {key: float(value) for key, value in radius_by_case.items()}
    envelope = {
        key: float(value) for key, value in same_case_envelope_by_case.items()
    }
    if not all(math.isfinite(value) for value in radius.values()):
        raise ValueError("matched-area radii must be finite")
    if not all(math.isfinite(value) and value >= 0.0 for value in envelope.values()):
        raise ValueError("same-case numerical envelopes must be finite and nonnegative")
    minimum_case = min(radius, key=radius.get)
    maximum_case = max(radius, key=radius.get)
    variation = float(radius[maximum_case] - radius[minimum_case])
    paired_envelope = float(envelope[maximum_case] + envelope[minimum_case])
    resolved = bool(variation > paired_envelope)
    ratio = (
        float(variation / paired_envelope)
        if paired_envelope > 0.0
        else (float("inf") if variation > 0.0 else 0.0)
    )
    return {
        "minimum_radius_case": minimum_case,
        "maximum_radius_case": maximum_case,
        "matched_area_radius_variation_km": variation,
        "paired_same_case_numerical_envelope_km": paired_envelope,
        "response_to_envelope_ratio": ratio,
        "resolved": resolved,
        "status": (
            "matched_area_radius_shape_dependence_resolved"
            if resolved
            else "matched_area_radius_shape_dependence_unresolved"
        ),
        "resolution_rule": (
            "max_minus_min_central_radius_exceeds_sum_of_same_case_envelopes"
        ),
    }


def turning_point_from_samples(
    central_pressure_mev_fm3: Any,
    mass_msun: Any,
    central_energy_density_mev_fm3: Any,
) -> TurningPointEstimate:
    """Estimate a turning point with PCHIP derivative and local quadratic fits."""
    pressure = np.asarray(central_pressure_mev_fm3, dtype=float)
    mass = np.asarray(mass_msun, dtype=float)
    epsilon = np.asarray(central_energy_density_mev_fm3, dtype=float)
    if (
        pressure.ndim != 1
        or mass.ndim != 1
        or epsilon.ndim != 1
        or not (pressure.size == mass.size == epsilon.size)
        or pressure.size < 7
    ):
        raise ValueError("turning-point estimation requires at least seven aligned samples")
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(mass)) or not np.all(
        np.isfinite(epsilon)
    ):
        raise ValueError("turning-point samples must be finite")
    if not np.all(np.diff(pressure) > 0.0) or not np.all(np.diff(epsilon) > 0.0):
        raise ValueError("turning-point coordinates must be strictly increasing")
    peak = int(np.argmax(mass))
    if peak < 2 or peak > len(mass) - 3:
        return TurningPointEstimate(
            "unresolved_not_interior",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            None,
        )
    interpolator = PchipInterpolator(pressure, mass, extrapolate=False)
    derivative = interpolator.derivative()
    derivative_at_nodes = np.asarray(derivative(pressure), dtype=float)
    bracket: tuple[float, float] | None = None
    if (
        derivative_at_nodes[peak - 1] > 0.0
        and derivative_at_nodes[peak + 1] < 0.0
    ):
        bracket = (float(pressure[peak - 1]), float(pressure[peak + 1]))
    else:
        changes = np.flatnonzero(
            (derivative_at_nodes[:-1] >= 0.0)
            & (derivative_at_nodes[1:] <= 0.0)
            & (
                (derivative_at_nodes[:-1] > 0.0)
                | (derivative_at_nodes[1:] < 0.0)
            )
        )
        if len(changes):
            change = int(changes[np.argmin(np.abs(changes - peak))])
            bracket = (float(pressure[change]), float(pressure[change + 1]))
    if bracket is None:
        return TurningPointEstimate(
            "unresolved_no_derivative_sign_change",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            int(np.count_nonzero(np.diff(mass[: peak + 1]) > 0.0)),
            int(np.count_nonzero(np.diff(mass[peak:]) < 0.0)),
            None,
        )
    lower, upper = bracket
    root = brentq(
        lambda value: float(derivative(value)),
        lower,
        upper,
        xtol=1.0e-11,
        rtol=4.0 * np.finfo(float).eps,
    )
    pchip_mass = float(interpolator(root))
    pchip_epsilon = float(
        PchipInterpolator(pressure, epsilon, extrapolate=False)(root)
    )
    local = slice(peak - 2, peak + 3)
    coefficients = np.polyfit(pressure[local], mass[local], 2)
    quadratic_pressure = float(-coefficients[1] / (2.0 * coefficients[0]))
    quadratic_mass = float(np.polyval(coefficients, quadratic_pressure))
    return TurningPointEstimate(
        status="bracketed_derivative_sign_change",
        pchip_pressure_mev_fm3=root,
        pchip_mass_msun=pchip_mass,
        pchip_energy_density_mev_fm3=pchip_epsilon,
        quadratic_pressure_mev_fm3=quadratic_pressure,
        quadratic_mass_msun=quadratic_mass,
        method_pressure_difference_mev_fm3=float(abs(root - quadratic_pressure)),
        method_mass_difference_msun=float(abs(pchip_mass - quadratic_mass)),
        positive_secants_before=int(np.count_nonzero(np.diff(mass[: peak + 1]) > 0.0)),
        negative_secants_after=int(np.count_nonzero(np.diff(mass[peak:]) < 0.0)),
        derivative_sign_change_bracket_mev_fm3=(lower, upper),
    )


def radial_deformation_support(
    radius_km: Any,
    mass_msun: Any,
    delta_cs2: Any,
    *,
    total_radius_km: float,
    total_mass_msun: float,
    fractions: tuple[float, ...] = (0.01, 0.10, 0.50),
    realized_peak_absolute_delta_cs2: float | None = None,
    solver_rtol: float = 0.0,
    solver_atol: float = 0.0,
) -> dict[str, Any]:
    """Locate continuous radial support intervals of the realized deformation.

    On every adjacent radial-node pair, the threshold function

    ``q_f = abs(delta_cs2) - f * realized_peak_absolute_delta_cs2``

    is represented by its piecewise-linear interpolant.  Each sign-changing
    boundary is its exact linear root, and enclosed mass is evaluated with the
    same interpolation fraction on the same stellar segment.  No boundary is
    snapped to a node and no value is extrapolated beyond the solved profile.
    Multiple disjoint support intervals are retained explicitly.
    """
    radius = np.asarray(radius_km, dtype=float)
    mass = np.asarray(mass_msun, dtype=float)
    delta = np.asarray(delta_cs2, dtype=float)
    if (
        not (radius.shape == mass.shape == delta.shape)
        or radius.ndim != 1
        or radius.size < 2
    ):
        raise ValueError("support profiles must be aligned one-dimensional arrays")
    if not (
        np.all(np.isfinite(radius))
        and np.all(np.isfinite(mass))
        and np.all(np.isfinite(delta))
    ):
        raise ValueError("support profiles must be finite")
    if not np.all(np.diff(radius) > 0.0):
        raise ValueError("support radius must be strictly increasing")
    if np.any(mass < 0.0):
        raise ValueError("support enclosed mass must be nonnegative")
    (
        effective_rtol,
        effective_atol,
        bounded_mass_reversal_count,
        maximum_bounded_mass_reversal,
    ) = _validate_nondecreasing_profile(
        mass,
        rtol=solver_rtol,
        atol=solver_atol,
        quantity="support enclosed mass",
    )
    stellar_radius = float(total_radius_km)
    stellar_mass = float(total_mass_msun)
    if (
        not math.isfinite(stellar_radius)
        or not math.isfinite(stellar_mass)
        or stellar_radius <= 0.0
        or stellar_mass <= 0.0
    ):
        raise ValueError("total stellar radius and mass must be finite and positive")
    profile_peak = float(np.max(np.abs(delta)))
    peak = (
        profile_peak
        if realized_peak_absolute_delta_cs2 is None
        else float(realized_peak_absolute_delta_cs2)
    )
    if not math.isfinite(peak) or peak < 0.0:
        raise ValueError("realized deformation peak must be finite and nonnegative")
    peak_guard = 32.0 * np.finfo(float).eps * max(1.0, peak, profile_peak)
    if profile_peak > peak + peak_guard:
        raise ValueError("declared realized peak is below the stored profile maximum")
    report: dict[str, Any] = {
        "realized_peak_absolute_delta_cs2": peak,
        "stored_profile_peak_absolute_delta_cs2": profile_peak,
        "profile_solver_rtol": effective_rtol,
        "profile_solver_atol": effective_atol,
        "bounded_mass_reversal_count": bounded_mass_reversal_count,
        "maximum_bounded_mass_reversal_msun": maximum_bounded_mass_reversal,
        "raw_mass_profile_preserved": True,
        "boundary_method": (
            "piecewise_linear_root_of_abs_delta_minus_threshold;"
            "mass_interpolated_with_same_radial_segment_fraction;"
            "no_extrapolation"
        ),
        "thresholds": {},
    }
    for fraction in fractions:
        fraction_value = float(fraction)
        if not math.isfinite(fraction_value) or fraction_value <= 0.0:
            raise ValueError("support fractions must be finite and positive")
        key = f"{fraction:.2f}"
        threshold = fraction_value * peak
        if peak == 0.0:
            report["thresholds"][key] = {
                "status": "not_reached",
                "threshold_absolute_delta_cs2": threshold,
                "intervals": [],
                "crossing_count": 0,
            }
            continue
        q = np.abs(delta) - threshold
        pieces: list[tuple[float, float, float, float]] = []
        crossings: list[dict[str, float | int]] = []
        for index in range(radius.size - 1):
            q_left = float(q[index])
            q_right = float(q[index + 1])
            left_inside = q_left >= 0.0
            right_inside = q_right >= 0.0
            if left_inside and right_inside:
                piece = (
                    float(radius[index]),
                    float(mass[index]),
                    float(radius[index + 1]),
                    float(mass[index + 1]),
                )
            elif not left_inside and not right_inside:
                continue
            else:
                interpolation_fraction = float(-q_left / (q_right - q_left))
                interpolation_fraction = min(1.0, max(0.0, interpolation_fraction))
                crossing_radius = float(
                    radius[index]
                    + interpolation_fraction * (radius[index + 1] - radius[index])
                )
                crossing_mass = float(
                    mass[index]
                    + interpolation_fraction * (mass[index + 1] - mass[index])
                )
                crossings.append(
                    {
                        "left_node_index": int(index),
                        "right_node_index": int(index + 1),
                        "interpolation_fraction": interpolation_fraction,
                        "radius_km": crossing_radius,
                        "enclosed_mass_msun": crossing_mass,
                    }
                )
                if left_inside:
                    piece = (
                        float(radius[index]),
                        float(mass[index]),
                        crossing_radius,
                        crossing_mass,
                    )
                else:
                    piece = (
                        crossing_radius,
                        crossing_mass,
                        float(radius[index + 1]),
                        float(mass[index + 1]),
                    )
            if pieces and math.isclose(
                pieces[-1][2],
                piece[0],
                rel_tol=0.0,
                abs_tol=64.0 * np.finfo(float).eps * max(1.0, stellar_radius),
            ):
                previous = pieces[-1]
                pieces[-1] = (previous[0], previous[1], piece[2], piece[3])
            else:
                pieces.append(piece)
        if not pieces:
            report["thresholds"][key] = {
                "status": "not_reached",
                "threshold_absolute_delta_cs2": threshold,
                "intervals": [],
                "crossings": crossings,
                "crossing_count": len(crossings),
            }
            continue
        intervals = []
        for inner_radius, inner_mass, outer_radius, outer_mass in pieces:
            intervals.append(
                {
                    "radius_interval_km": [inner_radius, outer_radius],
                    "radius_interval_r_over_R": [
                        inner_radius / stellar_radius,
                        outer_radius / stellar_radius,
                    ],
                    "enclosed_mass_interval_msun": [inner_mass, outer_mass],
                    "enclosed_mass_interval_over_M": [
                        inner_mass / stellar_mass,
                        outer_mass / stellar_mass,
                    ],
                    "radial_span_fraction": (
                        outer_radius - inner_radius
                    )
                    / stellar_radius,
                    "enclosed_mass_span_fraction": (
                        outer_mass - inner_mass
                    )
                    / stellar_mass,
                }
            )
        radial_span = float(
            sum(item["radial_span_fraction"] for item in intervals)
        )
        mass_span = float(
            sum(item["enclosed_mass_span_fraction"] for item in intervals)
        )
        inner_radius = float(intervals[0]["radius_interval_km"][0])
        outer_radius = float(intervals[-1]["radius_interval_km"][1])
        inner_mass = float(intervals[0]["enclosed_mass_interval_msun"][0])
        outer_mass = float(intervals[-1]["enclosed_mass_interval_msun"][1])
        report["thresholds"][key] = {
            "status": "reached",
            "threshold_absolute_delta_cs2": threshold,
            "intervals": intervals,
            "interval_count": len(intervals),
            "crossings": crossings,
            "crossing_count": len(crossings),
            "reaches_profile_inner_boundary": bool(q[0] >= 0.0),
            "reaches_profile_outer_boundary": bool(q[-1] >= 0.0),
            "radius_interval_km": [inner_radius, outer_radius],
            "radius_interval_r_over_R": [
                inner_radius / stellar_radius,
                outer_radius / stellar_radius,
            ],
            "enclosed_mass_interval_msun": [inner_mass, outer_mass],
            "enclosed_mass_interval_over_M": [
                inner_mass / stellar_mass,
                outer_mass / stellar_mass,
            ],
            "inner_radius_fraction": inner_radius / stellar_radius,
            "outer_radius_fraction": outer_radius / stellar_radius,
            "inner_enclosed_mass_fraction": inner_mass / stellar_mass,
            "outer_enclosed_mass_fraction": outer_mass / stellar_mass,
            "radial_span_fraction": radial_span,
            "enclosed_mass_span_fraction": mass_span,
        }
    return report


__all__ = [
    "BaryonIntegralResult",
    "FM3_TO_KM3",
    "G_CGS",
    "KM_TO_CM",
    "MEV_TO_ERG",
    "NEUTRON_REST_MASS_G",
    "SOLAR_MASS_G_FROM_PROJECT_LENGTH",
    "TurningPointEstimate",
    "baryon_number_from_profile",
    "interpolate_within_common_support",
    "matched_area_amplitude",
    "matched_area_radius_resolution",
    "odd_even_response",
    "radial_deformation_support",
    "select_outside_support_center",
    "support_tolerance_from_observable_envelopes",
    "turning_point_from_samples",
]

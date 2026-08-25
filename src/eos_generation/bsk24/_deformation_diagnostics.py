"""Saved diagnostics and final admissibility for windowed deformations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from eos_generation.bsk24._deformation_bounds import (
    RAW_DISCOVERY_INTERVALS_PER_SCALE,
    _geometry_aware_grid,
)
from eos_generation.bsk24._deformation_core import (
    PURE_GAUSSIAN_GENERATOR_ID,
    WINDOWED_GAUSSIAN_GENERATOR_ID,
    gaussian_profile,
    smootherstep_window,
    windowed_gaussian_shape,
)
from eos_generation.bsk24._deformation_gate import _refined_extremum
from eos_generation.bsk24.reconstruction import (
    COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3,
    COMPOSE_OUTER_INNER_TRANSITION_EPSILON_MEV_FM3,
    BSk24ConsistentBaseline,
)

if TYPE_CHECKING:
    from eos_generation.bsk24.deformation import (
        BSk24WindowedDeformation,
        BSk24WindowedEos,
    )


def _window_characterization_uncached(
    baseline: BSk24ConsistentBaseline,
    deformation: BSk24WindowedDeformation,
) -> dict[str, Any]:
    """Measure realized windowed-deformation shape and area."""
    lower = float(baseline.epsilon[0])
    upper = float(baseline.epsilon[-1])
    epsilon_t = baseline.anchor.energy_density_mev_fm3

    def gaussian(value: float) -> float:
        return float(gaussian_profile(value, deformation))

    def shape(value: float) -> float:
        return float(
            windowed_gaussian_shape(
                value,
                deformation,
                epsilon_t_mev_fm3=epsilon_t,
            )
        )

    quadrature_points = sorted(
        {
            float(point)
            for point in (
                epsilon_t,
                epsilon_t + deformation.delta_mev_fm3,
                deformation.epsilon0_mev_fm3
                - 4.0 * deformation.sigma_mev_fm3,
                deformation.epsilon0_mev_fm3,
                deformation.epsilon0_mev_fm3
                + 4.0 * deformation.sigma_mev_fm3,
            )
            if lower < point < upper
        }
    )

    shape_area, shape_error = quad(
        shape,
        lower,
        upper,
        points=quadrature_points,
        epsabs=1.0e-11,
        epsrel=1.0e-12,
        limit=400,
    )
    gaussian_area, gaussian_error = quad(
        gaussian,
        lower,
        upper,
        points=quadrature_points,
        epsabs=1.0e-11,
        epsrel=1.0e-12,
        limit=400,
    )
    removed_area, removed_error = quad(
        lambda value: gaussian(value) - shape(value),
        lower,
        upper,
        points=quadrature_points,
        epsabs=1.0e-11,
        epsrel=1.0e-12,
        limit=400,
    )
    centroid_numerator, centroid_error = quad(
        lambda value: value * shape(value),
        lower,
        upper,
        points=quadrature_points,
        epsabs=1.0e-9,
        epsrel=1.0e-12,
        limit=400,
    )
    base_grid = np.linspace(epsilon_t, upper, 131073)
    grid, geometry_resolution = _geometry_aware_grid(
        base_grid,
        epsilon0_mev_fm3=deformation.epsilon0_mev_fm3,
        sigma_mev_fm3=deformation.sigma_mev_fm3,
        delta_mev_fm3=deformation.delta_mev_fm3,
        epsilon_match_mev_fm3=epsilon_t,
        epsilon_max_mev_fm3=upper,
        intervals_per_scale=RAW_DISCOVERY_INTERVALS_PER_SCALE,
    )
    usable_area = bool(
        geometry_resolution["status"] == "resolved_geometry_aware_sampling"
        and np.isfinite(
            (
                shape_area,
                gaussian_area,
                removed_area,
                centroid_numerator,
            )
        ).all()
        and shape_area > 0.0
        and gaussian_area > 0.0
    )
    if not usable_area:
        return {
            "case_id": deformation.case_id,
            "parameters": deformation.to_dict(),
            "status": "unavailable_no_resolved_in_domain_support",
            "nominal_amplitude": deformation.amplitude,
            "window_at_epsilon0": float(
                smootherstep_window(
                    deformation.epsilon0_mev_fm3,
                    epsilon_t_mev_fm3=epsilon_t,
                    delta_mev_fm3=deformation.delta_mev_fm3,
                )
            ),
            "realized_delta_cs2_minimum": 0.0,
            "realized_delta_cs2_maximum": 0.0,
            "realized_extremum_epsilon_mev_fm3": None,
            "maximum_unit_shape_G_times_W": 0.0,
            "integrated_signed_deformation_mev_fm3": 0.0,
            "integrated_absolute_deformation_mev_fm3": 0.0,
            "unwindowed_gaussian_area_same_domain_mev_fm3": (
                float(gaussian_area) if np.isfinite(gaussian_area) else None
            ),
            "windowed_unit_shape_area_mev_fm3": (
                float(shape_area) if np.isfinite(shape_area) else None
            ),
            "window_suppressed_area_mev_fm3": (
                float(removed_area) if np.isfinite(removed_area) else None
            ),
            "suppressed_area_fraction": None,
            "centroid_definition": (
                "first moment of nonnegative realized unit shape G*W"
            ),
            "numerical_centroid_mev_fm3": None,
            "numerical_fwhm_mev_fm3": None,
            "fwhm_bounds_mev_fm3": None,
            "geometry_resolution": geometry_resolution,
            "quadrature": {
                "method": "adaptive Gauss-Kronrod scipy.integrate.quad",
                "epsabs": 1.0e-11,
                "epsrel": 1.0e-12,
                "shape_area_error_estimate": shape_error,
                "gaussian_area_error_estimate": gaussian_error,
                "removed_area_error_estimate": removed_error,
                "centroid_numerator_error_estimate": centroid_error,
            },
            "nominal_and_realized_parameters_distinguished": True,
        }
    shape_values = np.asarray(
        windowed_gaussian_shape(
            grid, deformation, epsilon_t_mev_fm3=epsilon_t
        ),
        dtype=float,
    )
    maximum_shape, extremum_epsilon = _refined_extremum(
        grid, shape_values, shape, maximize=True
    )
    half = 0.5 * maximum_shape
    peak_index = int(np.argmax(shape_values))
    left_candidates = np.flatnonzero(shape_values[: peak_index + 1] <= half)
    right_candidates = np.flatnonzero(shape_values[peak_index:] <= half)
    fwhm = None
    fwhm_bounds = None
    if len(left_candidates) and len(right_candidates):
        left_index = int(left_candidates[-1])
        right_index = int(peak_index + right_candidates[0])
        if left_index + 1 < len(grid) and right_index > 0:
            left_root = brentq(
                lambda value: shape(value) - half,
                float(grid[left_index]),
                float(grid[left_index + 1]),
            )
            right_root = brentq(
                lambda value: shape(value) - half,
                float(grid[right_index - 1]),
                float(grid[right_index]),
            )
            fwhm = float(right_root - left_root)
            fwhm_bounds = [float(left_root), float(right_root)]
    amplitude = deformation.amplitude
    actual_minimum = min(0.0, amplitude * maximum_shape)
    actual_maximum = max(0.0, amplitude * maximum_shape)
    return {
        "case_id": deformation.case_id,
        "parameters": deformation.to_dict(),
        "status": "computed_resolved_in_domain_support",
        "nominal_amplitude": amplitude,
        "window_at_epsilon0": float(
            smootherstep_window(
                deformation.epsilon0_mev_fm3,
                epsilon_t_mev_fm3=epsilon_t,
                delta_mev_fm3=deformation.delta_mev_fm3,
            )
        ),
        "realized_delta_cs2_minimum": actual_minimum,
        "realized_delta_cs2_maximum": actual_maximum,
        "realized_extremum_epsilon_mev_fm3": extremum_epsilon,
        "maximum_unit_shape_G_times_W": maximum_shape,
        "integrated_signed_deformation_mev_fm3": amplitude * shape_area,
        "integrated_absolute_deformation_mev_fm3": abs(amplitude) * shape_area,
        "unwindowed_gaussian_area_same_domain_mev_fm3": gaussian_area,
        "windowed_unit_shape_area_mev_fm3": shape_area,
        "window_suppressed_area_mev_fm3": removed_area,
        "suppressed_area_fraction": removed_area / gaussian_area,
        "centroid_definition": "first moment of nonnegative realized unit shape G*W",
        "numerical_centroid_mev_fm3": centroid_numerator / shape_area,
        "numerical_fwhm_mev_fm3": fwhm,
        "fwhm_bounds_mev_fm3": fwhm_bounds,
        "geometry_resolution": geometry_resolution,
        "quadrature": {
            "method": "adaptive Gauss-Kronrod scipy.integrate.quad",
            "epsabs": 1.0e-11,
            "epsrel": 1.0e-12,
            "shape_area_error_estimate": shape_error,
            "gaussian_area_error_estimate": gaussian_error,
            "removed_area_error_estimate": removed_error,
            "centroid_numerator_error_estimate": centroid_error,
        },
        "nominal_and_realized_parameters_distinguished": True,
    }


_WINDOW_CHARACTERIZATION_CACHE: tuple[
    BSk24ConsistentBaseline,
    tuple[float, float, float],
    dict[str, Any],
] | None = None


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

    global _WINDOW_CHARACTERIZATION_CACHE
    geometry = (
        float(deformation.epsilon0_mev_fm3),
        float(deformation.sigma_mev_fm3),
        float(deformation.delta_mev_fm3),
    )
    cached = _WINDOW_CHARACTERIZATION_CACHE
    if cached is None or cached[0] is not baseline or cached[1] != geometry:
        template = _window_characterization_uncached(baseline, deformation)
        _WINDOW_CHARACTERIZATION_CACHE = (baseline, geometry, dict(template))
    else:
        template = cached[2]

    result = dict(template)
    result["quadrature"] = dict(template["quadrature"])
    amplitude = float(deformation.amplitude)
    maximum_shape = float(result["maximum_unit_shape_G_times_W"])
    shape_area = float(result["windowed_unit_shape_area_mev_fm3"])
    result.update(
        {
            "case_id": deformation.case_id,
            "parameters": deformation.to_dict(),
            "nominal_amplitude": amplitude,
            "realized_delta_cs2_minimum": min(
                0.0, amplitude * maximum_shape
            ),
            "realized_delta_cs2_maximum": max(
                0.0, amplitude * maximum_shape
            ),
            "integrated_signed_deformation_mev_fm3": (
                amplitude * shape_area
            ),
            "integrated_absolute_deformation_mev_fm3": (
                abs(amplitude) * shape_area
            ),
        }
    )
    return result


def summarize_windowed_residuals(
    eos: BSk24WindowedEos,
    *,
    exclude_boundary_points: int = 4,
) -> dict[str, Any]:
    """Separate global, interior, ramp, transition, and boundary residuals."""
    epsilon = eos.epsilon
    anchor = eos.baseline.anchor.energy_density_mev_fm3
    ramp_end = anchor + eos.deformation.delta_mev_fm3
    base = np.ones(len(epsilon), dtype=bool)
    boundary = np.zeros(len(epsilon), dtype=bool)
    boundary[:exclude_boundary_points] = True
    boundary[-exclude_boundary_points:] = True
    base &= ~boundary
    upper_spacing = float(
        np.median(np.diff(epsilon[eos.baseline.anchor_index :]))
    )
    anchor_ramp = (epsilon >= anchor - 3.0 * upper_spacing) & (
        epsilon <= ramp_end + 3.0 * upper_spacing
    )
    transition = np.zeros(len(epsilon), dtype=bool)
    for value in (
        COMPOSE_OUTER_INNER_TRANSITION_EPSILON_MEV_FM3,
        COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3,
    ):
        index = int(np.argmin(np.abs(epsilon - value)))
        transition[max(0, index - 3) : min(len(epsilon), index + 4)] = True
    interior = base & ~anchor_ramp & ~transition

    def region(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        indices = np.flatnonzero(mask & np.isfinite(values))
        if not len(indices):
            return {"status": "unavailable", "point_count": 0}
        selected = np.abs(values[indices])
        index = int(indices[int(np.argmax(selected))])
        return {
            "status": "computed",
            "maximum_absolute": float(abs(values[index])),
            "signed_value_at_maximum": float(values[index]),
            "epsilon_at_maximum_mev_fm3": float(epsilon[index]),
            "p95_absolute": float(np.percentile(selected, 95.0)),
            "p99_absolute": float(np.percentile(selected, 99.0)),
            "point_count": int(len(indices)),
        }

    summaries: dict[str, Any] = {}
    for name in (
        "r_p_independent_normalized",
        "r_mu_independent_normalized",
        "first_law_normalized",
        "r_c",
    ):
        values = eos.residuals[name]
        summaries[name] = {
            "global_all_nodes": region(values, np.ones(len(epsilon), dtype=bool)),
            "global_excluding_boundaries": region(values, base),
            "interior_excluding_sensitive_bands": region(values, interior),
            "anchor_and_ramp_sensitive_band": region(values, anchor_ramp),
            "phase_transition_sensitive_bands": region(values, transition),
            "boundary_exclusion": region(values, boundary),
        }
    return {
        "case_id": eos.deformation.case_id,
        "definitions": {
            "related_PCHIP_closure_family": [
                "r_p_independent_normalized",
                "r_mu_independent_normalized",
                "first_law_normalized",
            ],
            "interpretation": (
                "algebraically related forms of one PCHIP derivative-consistency discrepancy"
            ),
            "distinct_check": (
                "r_c = raw continuous cs2 - dP/d-epsilon from independently differentiated PCHIP"
            ),
        },
        "derivative_method": "PCHIP derivatives of sampled pressure and baryon-density profiles",
        "anchor_and_ramp_band_mev_fm3": [
            anchor - 3.0 * upper_spacing,
            ramp_end + 3.0 * upper_spacing,
        ],
        "phase_transition_centers_mev_fm3": [
            COMPOSE_OUTER_INNER_TRANSITION_EPSILON_MEV_FM3,
            COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3,
        ],
        "boundary_excluded_points_per_side": exclude_boundary_points,
        "summaries": summaries,
    }


def full_domain_thermodynamic_admissibility(
    baseline: BSk24ConsistentBaseline,
    eos: BSk24WindowedEos,
    *,
    raw_gate_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the authoritative gate on the selected first-causal prefix.

    The independent constraints are finiteness, positive retained energy and
    pressure, causal/stable sound speed through the raw-gate endpoint, exact
    below-match identity, continuous matching, usable effective first-law
    reconstruction, positive monotone effective baryon density, and positive
    effective chemical potential.  Complete raw evidence remains separate and
    may contain the superluminal continuation after the selected endpoint.
    ``Gamma_eff``, ``P <= epsilon``, and ``dmu/dn`` remain diagnostics only.
    """

    epsilon = np.asarray(eos.epsilon, dtype=float)
    pressure = np.asarray(eos.pressure, dtype=float)
    cs2 = np.asarray(eos.cs2, dtype=float)
    baryon_density = np.asarray(eos.baryon_density, dtype=float)
    chemical_potential = np.asarray(eos.chemical_potential, dtype=float)
    gamma_eff = np.asarray(eos.adiabatic_index, dtype=float)
    raw_epsilon = np.asarray(eos.raw_epsilon, dtype=float)
    raw_pressure = np.asarray(eos.raw_pressure, dtype=float)
    raw_cs2 = np.asarray(eos.raw_cs2, dtype=float)
    retained_gate = (
        raw_gate_report.get("retained_domain")
        if isinstance(raw_gate_report, Mapping)
        else None
    )
    gate_selected = bool(
        isinstance(raw_gate_report, Mapping)
        and raw_gate_report.get("status")
        == "accepted_raw_local_physics_gate"
        and raw_gate_report.get("selected_retained_domain_authoritative")
        is True
        and raw_gate_report.get("selected_retained_domain_passed") is True
        and isinstance(retained_gate, Mapping)
        and retained_gate.get("passed") is True
        and retained_gate.get("resolution_certified") is True
    )
    try:
        selected_endpoint = (
            float(retained_gate["epsilon_max_mev_fm3"])
            if isinstance(retained_gate, Mapping)
            else float(epsilon[-1])
        )
    except (KeyError, TypeError, ValueError):
        selected_endpoint = math.nan
    crossing = (
        retained_gate.get("first_causal_crossing")
        if isinstance(retained_gate, Mapping)
        else None
    )
    selected_at_crossing = bool(
        isinstance(crossing, Mapping)
        and crossing.get("status")
        == "resolved_first_continuous_causal_crossing"
        and crossing.get("epsilon_mev_fm3") == selected_endpoint
    )
    retained_state_arrays = (
        epsilon,
        pressure,
        cs2,
        baryon_density,
        chemical_potential,
    )
    raw_state_arrays = (
        raw_epsilon,
        raw_pressure,
        raw_cs2,
    )
    retained_aligned = bool(
        all(array.ndim == 1 for array in (*retained_state_arrays, gamma_eff))
        and len(
            {len(array) for array in (*retained_state_arrays, gamma_eff)}
        )
        == 1
        and len(epsilon) > 1
    )
    raw_aligned = bool(
        all(array.ndim == 1 for array in raw_state_arrays)
        and len({len(array) for array in raw_state_arrays}) == 1
        and len(raw_epsilon) > 1
    )
    finite_state = bool(
        retained_aligned
        and raw_aligned
        and all(
            np.all(np.isfinite(array))
            for array in (*retained_state_arrays, *raw_state_arrays)
        )
    )
    complete_raw_evidence_retained = bool(
        raw_aligned
        and raw_epsilon[0] == baseline.epsilon[0]
        and raw_epsilon[-1] == baseline.epsilon[-1]
        and np.all(np.diff(raw_epsilon) > 0.0)
    )
    selected_domain_retained = bool(
        retained_aligned
        and epsilon[0] == baseline.epsilon[0]
        and math.isfinite(selected_endpoint)
        and epsilon[-1] == selected_endpoint
        and np.all(np.diff(epsilon) > 0.0)
    )
    positive_epsilon = bool(finite_state and np.all(epsilon > 0.0))
    positive_pressure = bool(finite_state and np.all(pressure > 0.0))
    mechanically_stable = bool(finite_state and np.all(cs2 > 0.0))
    causal = bool(
        finite_state
        and (
            (
                selected_at_crossing
                and np.all(cs2[:-1] < 1.0)
                and cs2[-1] <= 1.0
                and isinstance(retained_gate, Mapping)
                and cs2[-1] == retained_gate.get("cs2_at_endpoint")
            )
            or (not selected_at_crossing and np.all(cs2 <= 1.0))
        )
    )
    anchor_matches = np.flatnonzero(
        epsilon == baseline.anchor.energy_density_mev_fm3
    )
    anchor_index = int(anchor_matches[0]) if len(anchor_matches) == 1 else -1
    below = slice(0, anchor_index)
    exact_below_match = bool(
        retained_aligned
        and anchor_index == baseline.anchor_index
        and np.array_equal(epsilon[below], baseline.epsilon[below])
        and np.array_equal(pressure[below], baseline.pressure[below])
        and np.array_equal(cs2[below], baseline.cs2[below])
        and np.array_equal(
            baryon_density[below], baseline.baryon_density[below]
        )
        and np.array_equal(
            chemical_potential[below], baseline.chemical_potential[below]
        )
    )
    matching_residuals = {
        "pressure_mev_fm3": float(
            pressure[anchor_index] - baseline.pressure[anchor_index]
        ) if retained_aligned and anchor_index >= 0 else None,
        "cs2": float(cs2[anchor_index] - baseline.cs2[anchor_index])
        if retained_aligned and anchor_index >= 0 else None,
        "baryon_density_fm3": float(
            baryon_density[anchor_index] - baseline.baryon_density[anchor_index]
        ) if retained_aligned and anchor_index >= 0 else None,
        "chemical_potential_mev": float(
            chemical_potential[anchor_index]
            - baseline.chemical_potential[anchor_index]
        ) if retained_aligned and anchor_index >= 0 else None,
    }
    continuous_matching = bool(
        retained_aligned
        and anchor_index >= 0
        and all(value == 0.0 for value in matching_residuals.values())
    )
    residual_arrays = tuple(
        np.asarray(values, dtype=float) for values in eos.residuals.values()
    )
    reconstruction_residuals_finite = bool(
        residual_arrays
        and all(
            array.shape == epsilon.shape and np.all(np.isfinite(array))
            for array in residual_arrays
        )
    )
    baryon_density_positive = bool(
        finite_state and np.all(baryon_density > 0.0)
    )
    baryon_density_monotone = bool(
        finite_state and np.all(np.diff(baryon_density) > 0.0)
    )
    chemical_potential_positive = bool(
        finite_state and np.all(chemical_potential > 0.0)
    )
    raw_gate_passed = bool(raw_gate_report is None or gate_selected)
    independent_checks = {
        "raw_selected_domain_gate_passed": raw_gate_passed,
        "aligned_finite_state": finite_state,
        "complete_raw_evidence_retained": complete_raw_evidence_retained,
        "selected_retained_domain_matches_raw_gate": selected_domain_retained,
        "epsilon_positive": positive_epsilon,
        "pressure_positive": positive_pressure,
        "sound_speed_strictly_positive": mechanically_stable,
        "sound_speed_causal_on_selected_prefix": causal,
        "exact_preservation_below_epsilon_match": exact_below_match,
        "continuous_matching": continuous_matching,
        "effective_first_law_reconstruction_succeeded": (
            reconstruction_residuals_finite
        ),
        "effective_baryon_density_positive": baryon_density_positive,
        "effective_baryon_density_strictly_monotone": (
            baryon_density_monotone
        ),
        "effective_chemical_potential_positive": (
            chemical_potential_positive
        ),
    }
    passed = bool(all(independent_checks.values()))
    failed_checks = [
        name for name, value in independent_checks.items() if not value
    ]
    if retained_aligned and len(baryon_density) > 1:
        dmu_dn = np.diff(chemical_potential) / np.diff(baryon_density)
        finite_dmu_dn = dmu_dn[np.isfinite(dmu_dn)]
    else:
        finite_dmu_dn = np.asarray([], dtype=float)
    direct_endpoint_retained = bool(
        math.isfinite(selected_endpoint)
        and selected_endpoint == float(baseline.epsilon[-1])
    )
    status = (
        (
            "accepted_full_domain_thermodynamic_gate"
            if direct_endpoint_retained
            else "accepted_selected_domain_thermodynamic_gate"
        )
        if passed
        else (
            "rejected_full_domain_thermodynamic_gate"
            if direct_endpoint_retained
            else "rejected_selected_domain_thermodynamic_gate"
        )
    )
    return {
        "schema_id": "bsk24_selected_domain_thermodynamic_gate_v2",
        "case_id": eos.deformation.case_id,
        "authoritative_for_trial_acceptance": True,
        "domain_policy": "prefix_through_first_continuous_cs2_equals_one",
        "direct_endpoint_retained": direct_endpoint_retained,
        "retained_domain_mev_fm3": [
            float(epsilon[0]) if len(epsilon) else None,
            float(epsilon[-1]) if len(epsilon) else None,
        ],
        "complete_raw_domain_mev_fm3": [
            float(raw_epsilon[0]) if len(raw_epsilon) else None,
            float(raw_epsilon[-1]) if len(raw_epsilon) else None,
        ],
        "independent_checks": independent_checks,
        "matching_residuals": matching_residuals,
        "physical_margins": {
            "minimum_pressure_mev_fm3": (
                float(np.min(pressure)) if finite_state else None
            ),
            "minimum_cs2": float(np.min(cs2)) if finite_state else None,
            "causality_margin": (
                float(1.0 - np.max(cs2)) if finite_state else None
            ),
            "minimum_effective_baryon_density_fm3": (
                float(np.min(baryon_density)) if finite_state else None
            ),
            "minimum_effective_chemical_potential_mev": (
                float(np.min(chemical_potential)) if finite_state else None
            ),
        },
        "diagnostics_not_independent_parameter_constraints": {
            "gamma_eff_finite": bool(
                retained_aligned and np.all(np.isfinite(gamma_eff))
            ),
            "pressure_leq_energy_density": bool(
                retained_aligned and np.all(pressure <= epsilon)
            ),
            "dmu_eff_dn_positive": bool(
                len(finite_dmu_dn)
                and len(finite_dmu_dn) == len(epsilon) - 1
                and np.all(finite_dmu_dn > 0.0)
            ),
            "microscopic_composition": "unavailable",
            "species_chemical_potentials": "unavailable",
            "microscopic_beta_equilibrium": "unassessed",
        },
        "failed_checks": failed_checks,
        "rejection_reason": None if passed else failed_checks[0],
        "status": status,
    }


def windowed_a0_identity_report(
    baseline: BSk24ConsistentBaseline,
    cases: Mapping[float, BSk24WindowedEos],
) -> dict[str, Any]:
    """Check exact local A=0 identity independently for every Delta."""
    quantities = {
        "pressure": "pressure",
        "sound_speed_squared": "cs2",
        "baryon_density": "baryon_density",
        "effective_chemical_potential": "chemical_potential",
        "adiabatic_index": "adiabatic_index",
        "energy_per_baryon": "energy_per_baryon_minus_neutron_rest",
    }
    report: dict[str, Any] = {
        "generator_id": WINDOWED_GAUSSIAN_GENERATOR_ID,
        "identity_target": (
            "direct C4 pressure/sound speed plus C4-consistent reconstruction, C1-normalized at anchor"
        ),
        "pure_gaussian_method_unchanged": PURE_GAUSSIAN_GENERATOR_ID,
        "deltas": {},
    }
    for delta, eos in cases.items():
        if eos.deformation.amplitude != 0.0:
            raise ValueError("A=0 identity report received nonzero amplitude")
        items = {}
        for name, attribute in quantities.items():
            observed = np.asarray(getattr(eos, attribute))
            expected = np.asarray(getattr(baseline, attribute))
            residual = observed - expected
            items[name] = {
                "maximum_absolute_residual": float(np.max(np.abs(residual))),
                "array_equal": bool(np.array_equal(observed, expected)),
                "status": (
                    "pass" if np.array_equal(observed, expected) else "fail"
                ),
            }
        report["deltas"][str(delta)] = items
    report["status"] = (
        "pass"
        if all(
            item["status"] == "pass"
            for delta_report in report["deltas"].values()
            for item in delta_report.values()
        )
        else "fail"
    )
    return report

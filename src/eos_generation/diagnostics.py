"""Diagnostics for controlled BSk24 / BSk25 experiments."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from . import tov as tov_core
from .assessment import (
    RAW_DISCOVERY_INTERVALS_PER_SCALE,
    _geometry_aware_grid,
    _refined_extremum,
)
from .baseline import C_LIGHT_CM_S, MEV_FM3_TO_ERG_CM3, NEUTRON_REST_ENERGY_MEV
from .deformation import (
    BSk24WindowedDeformation,
    PURE_GAUSSIAN_GENERATOR_ID,
    WINDOWED_GAUSSIAN_GENERATOR_ID,
    gaussian_profile,
    smootherstep_window,
    windowed_gaussian_delta_cs2,
    windowed_gaussian_pressure_primitive,
    windowed_gaussian_shape,
)
from .numerics import BSk24TrialConfig, DEFAULT_CONFIG
from .stellar import _tov_settings
from .storage import write_csv_atomic
from .thermodynamics import (
    BSk24ConsistentBaseline,
    BSk24WindowedEos,
    COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3,
    COMPOSE_OUTER_INNER_TRANSITION_EPSILON_MEV_FM3,
    _bidirectional_baryon_reconstruction,
)
from dataclasses import asdict, dataclass
from pathlib import Path
from scipy.integrate import quad, simpson
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from typing import Any, Mapping, Sequence


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
                deformation.epsilon0_mev_fm3 - 4.0 * deformation.sigma_mev_fm3,
                deformation.epsilon0_mev_fm3,
                deformation.epsilon0_mev_fm3 + 4.0 * deformation.sigma_mev_fm3,
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
        windowed_gaussian_shape(grid, deformation, epsilon_t_mev_fm3=epsilon_t),
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


_WINDOW_CHARACTERIZATION_CACHE: (
    tuple[
        BSk24ConsistentBaseline,
        tuple[float, float, float],
        dict[str, Any],
    ]
    | None
) = None


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
            "realized_delta_cs2_minimum": min(0.0, amplitude * maximum_shape),
            "realized_delta_cs2_maximum": max(0.0, amplitude * maximum_shape),
            "integrated_signed_deformation_mev_fm3": (amplitude * shape_area),
            "integrated_absolute_deformation_mev_fm3": (abs(amplitude) * shape_area),
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
    upper_spacing = float(np.median(np.diff(epsilon[eos.baseline.anchor_index :])))
    anchor_ramp = (epsilon >= anchor - 3.0 * upper_spacing) & (
        epsilon <= ramp_end + 3.0 * upper_spacing
    )
    transition = np.zeros(len(epsilon), dtype=bool)
    for value in (
        eos.baseline.eos.definition.outer_inner_transition_epsilon_mev_fm3,
        eos.baseline.eos.definition.core_entry_epsilon_mev_fm3,
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
            eos.baseline.eos.definition.outer_inner_transition_epsilon_mev_fm3,
            eos.baseline.eos.definition.core_entry_epsilon_mev_fm3,
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
        and raw_gate_report.get("status") == "accepted_raw_local_physics_gate"
        and raw_gate_report.get("selected_retained_domain_authoritative") is True
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
        and crossing.get("status") == "resolved_first_continuous_causal_crossing"
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
        and len({len(array) for array in (*retained_state_arrays, gamma_eff)}) == 1
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
        and isinstance(raw_gate_report, Mapping)
        and raw_gate_report.get("complete_proposed_retained_domain_mev_fm3")
        == [float(raw_epsilon[0]), float(raw_epsilon[-1])]
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
    anchor_matches = np.flatnonzero(epsilon == baseline.anchor.energy_density_mev_fm3)
    anchor_index = int(anchor_matches[0]) if len(anchor_matches) == 1 else -1
    below = slice(0, anchor_index)
    exact_below_match = bool(
        retained_aligned
        and anchor_index == baseline.anchor_index
        and np.array_equal(epsilon[below], baseline.epsilon[below])
        and np.array_equal(pressure[below], baseline.pressure[below])
        and np.array_equal(cs2[below], baseline.cs2[below])
        and np.array_equal(baryon_density[below], baseline.baryon_density[below])
        and np.array_equal(
            chemical_potential[below], baseline.chemical_potential[below]
        )
    )
    matching_residuals = {
        "pressure_mev_fm3": (
            float(pressure[anchor_index] - baseline.pressure[anchor_index])
            if retained_aligned and anchor_index >= 0
            else None
        ),
        "cs2": (
            float(cs2[anchor_index] - baseline.cs2[anchor_index])
            if retained_aligned and anchor_index >= 0
            else None
        ),
        "baryon_density_fm3": (
            float(baryon_density[anchor_index] - baseline.baryon_density[anchor_index])
            if retained_aligned and anchor_index >= 0
            else None
        ),
        "chemical_potential_mev": (
            float(
                chemical_potential[anchor_index]
                - baseline.chemical_potential[anchor_index]
            )
            if retained_aligned and anchor_index >= 0
            else None
        ),
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
    baryon_density_positive = bool(finite_state and np.all(baryon_density > 0.0))
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
        "effective_baryon_density_strictly_monotone": (baryon_density_monotone),
        "effective_chemical_potential_positive": (chemical_potential_positive),
    }
    passed = bool(all(independent_checks.values()))
    failed_checks = [name for name, value in independent_checks.items() if not value]
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
            "causality_margin": (float(1.0 - np.max(cs2)) if finite_state else None),
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
                "status": ("pass" if np.array_equal(observed, expected) else "fail"),
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


def _maximum_absolute_residual(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return a finite comparison residual without warning on all-NaN data."""

    if not len(left):
        return None
    with np.errstate(invalid="ignore"):
        absolute_difference = np.abs(left - right)
    if bool(np.isnan(absolute_difference).all()):
        return None
    return float(np.nanmax(absolute_difference))


def _raw_gate_frame(
    *,
    case_id: str,
    deformation: BSk24WindowedDeformation,
    baseline: BSk24ConsistentBaseline,
    epsilon: np.ndarray,
    raw_cs2: np.ndarray,
    status: str,
) -> pd.DataFrame:
    epsilon_t = baseline.anchor.energy_density_mev_fm3
    delta_pressure = np.asarray(
        windowed_gaussian_pressure_primitive(
            epsilon,
            deformation,
            epsilon_t_mev_fm3=epsilon_t,
        ),
        dtype=float,
    )
    direct_pressure = np.asarray(
        baseline.eos.published_fit_pressure_from_energy_density(epsilon),
        dtype=float,
    )
    return pd.DataFrame(
        {
            "case_id": case_id,
            "amplitude": deformation.amplitude,
            "epsilon0_mev_fm3": deformation.epsilon0_mev_fm3,
            "sigma_mev_fm3": deformation.sigma_mev_fm3,
            "delta_mev_fm3": deformation.delta_mev_fm3,
            "epsilon_mev_fm3": epsilon,
            "window": np.asarray(
                smootherstep_window(
                    epsilon,
                    epsilon_t_mev_fm3=epsilon_t,
                    delta_mev_fm3=deformation.delta_mev_fm3,
                ),
                dtype=float,
            ),
            "gaussian": np.asarray(gaussian_profile(epsilon, deformation), dtype=float),
            "delta_cs2": np.asarray(
                windowed_gaussian_delta_cs2(
                    epsilon, deformation, epsilon_t_mev_fm3=epsilon_t
                ),
                dtype=float,
            ),
            "direct_pressure_mev_fm3": direct_pressure,
            "delta_pressure_mev_fm3": delta_pressure,
            "raw_pressure_mev_fm3": direct_pressure + delta_pressure,
            "raw_cs2": raw_cs2,
            "gate_status": status,
        }
    )


def _thermodynamic_profile_frame(
    baseline: BSk24ConsistentBaseline,
    generated: Mapping[str, BSk24WindowedEos],
) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {
                "case_id": "direct",
                "amplitude": np.nan,
                "delta_mev_fm3": np.nan,
                "epsilon_mev_fm3": baseline.epsilon,
                "pressure_mev_fm3": baseline.pressure,
                "cs2": baseline.cs2,
                "delta_cs2": np.zeros_like(baseline.epsilon),
                "baryon_density_fm3": baseline.baryon_density,
                "effective_baryon_enthalpy_mev": baseline.chemical_potential,
                "gamma_eff": baseline.adiabatic_index,
                "energy_per_baryon_minus_neutron_rest_mev": (
                    baseline.energy_per_baryon_minus_neutron_rest
                ),
                "pressure_relative_to_direct": np.zeros_like(baseline.epsilon),
                "baryon_density_relative_to_direct": np.zeros_like(baseline.epsilon),
                "enthalpy_relative_to_direct": np.zeros_like(baseline.epsilon),
            }
        )
    ]
    epsilon_t = baseline.anchor.energy_density_mev_fm3
    for case_id, eos in generated.items():
        epsilon = eos.epsilon
        direct_pressure = np.asarray(
            baseline.eos.published_fit_pressure_from_energy_density(epsilon),
            dtype=float,
        )
        if epsilon[-1] <= baseline.epsilon[-1]:
            direct_density = np.asarray(
                baseline.consistent_baryon_density_from_energy_density(epsilon),
                dtype=float,
            )
        else:
            anchor_matches = np.flatnonzero(
                epsilon == baseline.anchor.energy_density_mev_fm3
            )
            if len(anchor_matches) != 1:
                raise ValueError("extended direct comparison requires the exact anchor")
            anchor_index = int(anchor_matches[0])
            upper_density = _bidirectional_baryon_reconstruction(
                epsilon[anchor_index:],
                direct_pressure[anchor_index:],
                anchor_index=0,
                anchor_density_fm3=baseline.anchor.baryon_density_fm3,
            )
            direct_density = np.concatenate(
                (baseline.baryon_density[:anchor_index], upper_density)
            )
        direct_enthalpy = (epsilon + direct_pressure) / direct_density
        frames.append(
            pd.DataFrame(
                {
                    "case_id": case_id,
                    "amplitude": eos.deformation.amplitude,
                    "delta_mev_fm3": eos.deformation.delta_mev_fm3,
                    "epsilon_mev_fm3": epsilon,
                    "pressure_mev_fm3": eos.pressure,
                    "cs2": eos.cs2,
                    "delta_cs2": np.asarray(
                        windowed_gaussian_delta_cs2(
                            epsilon,
                            eos.deformation,
                            epsilon_t_mev_fm3=epsilon_t,
                        ),
                        dtype=float,
                    ),
                    "baryon_density_fm3": eos.baryon_density,
                    "effective_baryon_enthalpy_mev": eos.chemical_potential,
                    "gamma_eff": eos.adiabatic_index,
                    "energy_per_baryon_minus_neutron_rest_mev": (
                        eos.energy_per_baryon_minus_neutron_rest
                    ),
                    "pressure_relative_to_direct": (eos.pressure - direct_pressure)
                    / direct_pressure,
                    "baryon_density_relative_to_direct": (
                        eos.baryon_density - direct_density
                    )
                    / direct_density,
                    "enthalpy_relative_to_direct": (
                        eos.chemical_potential - direct_enthalpy
                    )
                    / direct_enthalpy,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _thermodynamic_residual_frame(
    generated: Mapping[str, BSk24WindowedEos],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for case_id, eos in generated.items():
        frame = pd.DataFrame(
            {
                "case_id": case_id,
                "amplitude": eos.deformation.amplitude,
                "delta_mev_fm3": eos.deformation.delta_mev_fm3,
                "epsilon_mev_fm3": eos.epsilon,
            }
        )
        for name, values in eos.residuals.items():
            frame[name] = values
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


_THERMODYNAMIC_CONVERGENCE_METRICS = (
    "r_p_independent_normalized",
    "r_mu_independent_normalized",
    "first_law_normalized",
    "r_c",
)


_THERMODYNAMIC_RESIDUAL_REGIONS = (
    "global_all_nodes",
    "global_excluding_boundaries",
    "interior_excluding_sensitive_bands",
    "anchor_and_ramp_sensitive_band",
    "phase_transition_sensitive_bands",
    "boundary_exclusion",
)


_REFINEMENT_ULP_ALLOWANCE = 64


def _refinement_pair_allowance(left: float, right: float) -> float:
    """Return a scale-local floating-point allowance, measured only in ulps."""

    scale = max(abs(float(left)), abs(float(right)))
    return float(_REFINEMENT_ULP_ALLOWANCE * math.ulp(scale))


def _classify_refinement_series(
    ordered_values: Sequence[float | None],
) -> dict[str, Any]:
    """Classify one ordered refinement series without an absolute tolerance."""

    values = [
        None if value is None or not math.isfinite(float(value)) else float(value)
        for value in ordered_values
    ]
    finite_complete = all(value is not None for value in values)
    pairwise: list[dict[str, Any]] = []
    if finite_complete:
        for left, right in zip(values[:-1], values[1:]):
            assert left is not None and right is not None
            allowance = _refinement_pair_allowance(left, right)
            signed_change = float(right - left)
            if signed_change > allowance:
                relation = "increase"
            elif signed_change < -allowance:
                relation = "decrease"
            else:
                relation = "floating_equivalent"
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "signed_change": signed_change,
                    "absolute_change": abs(signed_change),
                    "floating_allowance": allowance,
                    "relation": relation,
                }
            )

    relations = [item["relation"] for item in pairwise]
    if not finite_complete:
        monotonicity_status = "nonfinite_or_missing_evidence"
    elif any(relation == "increase" for relation in relations):
        monotonicity_status = "mixed_or_nonmonotone_refinement"
    elif relations and all(relation == "decrease" for relation in relations):
        monotonicity_status = "strictly_decreasing"
    else:
        monotonicity_status = "nonincreasing_with_floating_equivalent_stages"

    absolute_changes = [item["absolute_change"] for item in pairwise]
    contraction_pairs: list[dict[str, Any]] = []
    if finite_complete and len(absolute_changes) >= 2:
        for previous, current in zip(absolute_changes[:-1], absolute_changes[1:]):
            allowance = _refinement_pair_allowance(previous, current)
            signed_change = float(current - previous)
            if signed_change < -allowance:
                relation = "contracting"
            elif signed_change > allowance:
                relation = "expanding"
            else:
                relation = "floating_equivalent"
            contraction_pairs.append(
                {
                    "previous_absolute_change": previous,
                    "current_absolute_change": current,
                    "signed_change": signed_change,
                    "floating_allowance": allowance,
                    "relation": relation,
                }
            )
    if not finite_complete or len(absolute_changes) < 2:
        contraction_status = "not_assessable"
    elif any(item["relation"] == "expanding" for item in contraction_pairs):
        contraction_status = "not_contracting"
    elif all(item["relation"] == "contracting" for item in contraction_pairs):
        contraction_status = "contracting_changes"
    else:
        contraction_status = "contraction_not_resolved"

    if len(values) < 3:
        status = "insufficient_stages"
    elif not finite_complete:
        status = "nonfinite_or_missing_evidence"
    elif monotonicity_status == "mixed_or_nonmonotone_refinement":
        status = "mixed_or_nonmonotone_refinement"
    elif (
        monotonicity_status == "strictly_decreasing"
        and contraction_status == "contracting_changes"
    ):
        status = "pass_monotonically_decreasing_interior_residuals"
    elif monotonicity_status != "strictly_decreasing":
        status = "no_strict_decrease_floating_equivalent_refinement"
    else:
        status = "strictly_decreasing_without_meaningful_contraction"

    finite_changes = [
        item["absolute_change"]
        for item in pairwise
        if math.isfinite(item["absolute_change"])
    ]
    return {
        "ordered_values": values,
        "stage_count": len(values),
        "finite_complete": finite_complete,
        "pairwise_refinement": pairwise,
        "successive_absolute_changes": finite_changes,
        "measured_refinement_envelope": (
            max(finite_changes) if finite_complete and finite_changes else None
        ),
        "monotonicity_status": monotonicity_status,
        "contraction_status": contraction_status,
        "contraction_pairs": contraction_pairs,
        "status": status,
    }


def _thermodynamic_convergence(
    stage_cases: Mapping[str, Mapping[str, BSk24WindowedEos]],
) -> dict[str, Any]:
    stage_names = tuple(stage_cases)
    required_case_ids = sorted(
        set().union(*(set(cases) for cases in stage_cases.values()))
        if stage_cases
        else set()
    )
    cases: dict[str, Any] = {}
    for case_id in required_case_ids:
        cases[case_id] = {}
        for metric in _THERMODYNAMIC_CONVERGENCE_METRICS:
            values: dict[str, dict[str, float | None]] = {}
            missing_evidence: list[dict[str, str]] = []
            for stage in stage_names:
                stage_values: dict[str, float | None] = {}
                eos = stage_cases[stage].get(case_id)
                try:
                    summary = eos.diagnostics["residual_summary"]["summaries"][metric]
                except (AttributeError, KeyError, TypeError):
                    summary = {}
                for region in _THERMODYNAMIC_RESIDUAL_REGIONS:
                    record = summary.get(region)
                    raw_value = (
                        record.get("maximum_absolute")
                        if isinstance(record, Mapping)
                        else None
                    )
                    try:
                        numeric = None if raw_value is None else float(raw_value)
                    except (TypeError, ValueError):
                        numeric = None
                    if numeric is None or not math.isfinite(numeric):
                        numeric = None
                        missing_evidence.append(
                            {
                                "stage": stage,
                                "region": region,
                                "reason": (
                                    "case_missing_from_stage"
                                    if eos is None
                                    else "missing_or_nonfinite_residual_maximum"
                                ),
                            }
                        )
                    stage_values[region] = numeric
                values[stage] = stage_values
            interior = [
                values[stage]["interior_excluding_sensitive_bands"]
                for stage in stage_names
            ]
            classification = _classify_refinement_series(interior)
            if missing_evidence and len(stage_names) >= 3:
                classification["status"] = "nonfinite_or_missing_evidence"
            cases[case_id][metric] = {
                "region_maxima_by_stage": values,
                "missing_evidence": missing_evidence,
                **classification,
            }

    record_statuses = [
        metric["status"] for case in cases.values() for metric in case.values()
    ]
    if len(stage_names) < 3:
        status = "insufficient_stages"
    elif not record_statuses or any(
        item == "nonfinite_or_missing_evidence" for item in record_statuses
    ):
        status = "nonfinite_or_missing_evidence"
    elif any(item == "mixed_or_nonmonotone_refinement" for item in record_statuses):
        status = "mixed_or_nonmonotone_refinement"
    elif any(
        item == "no_strict_decrease_floating_equivalent_refinement"
        for item in record_statuses
    ):
        status = "no_strict_decrease_floating_equivalent_refinement"
    elif any(
        item == "strictly_decreasing_without_meaningful_contraction"
        for item in record_statuses
    ):
        status = "strictly_decreasing_without_meaningful_contraction"
    else:
        status = "pass_monotonically_decreasing_interior_residuals"
    return {
        "schema_id": "eos_generation_thermodynamic_convergence_v1",
        "stages": list(stage_names),
        "stage_count": len(stage_names),
        "minimum_stages_for_decreasing_refinement_pass": 3,
        "required_case_ids": required_case_ids,
        "required_metrics": list(_THERMODYNAMIC_CONVERGENCE_METRICS),
        "required_regions": list(_THERMODYNAMIC_RESIDUAL_REGIONS),
        "floating_comparison_policy": {
            "method": "64 ulps at the magnitude of each compared pair",
            "ulp_count": _REFINEMENT_ULP_ALLOWANCE,
            "no_absolute_scientific_tolerance": True,
        },
        "closure_interpretation": {
            "related_forms": [
                "r_p_independent_normalized",
                "r_mu_independent_normalized",
                "first_law_normalized",
            ],
            "meaning": (
                "algebraically related forms of one PCHIP derivative-consistency discrepancy"
            ),
            "distinct_check": "r_c = cs2 - dP/d_epsilon",
        },
        "cases": cases,
        "status": status,
    }


FM3_TO_KM3 = 1.0e54


G_CGS = 6.67430e-8


KM_TO_CM = 1.0e5


MEV_TO_ERG = MEV_FM3_TO_ERG_CM3 * 1.0e-39


SOLAR_MASS_G_FROM_PROJECT_LENGTH = (
    DEFAULT_CONFIG.units.solar_mass_length_km * KM_TO_CM * C_LIGHT_CM_S**2 / G_CGS
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
        + effective_rtol * np.maximum(np.abs(values[:-1]), np.abs(values[1:]))
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
        (float(np.max(-differences[reversals])) if np.any(reversals) else 0.0),
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
    metadata_required = tov_core._discontinuity_metadata_is_required(eos_callable)
    try:
        discontinuities = tov_core._resolved_discontinuities(eos_callable)
    except (TypeError, ValueError):
        if metadata_required:
            raise
        discontinuities = ()
    if metadata_required:
        tov_core._validate_declared_branch_values(eos_callable, discontinuities)
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
        if metadata_required:
            raise
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
        raise ValueError(
            "mass and observable arrays must be aligned one-dimensional data"
        )
    if (
        masses.size < 2
        or not np.all(np.isfinite(masses))
        or not np.all(np.isfinite(observable))
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
        raise ValueError(
            "radial baryon integral requires aligned profiles of at least 3 points"
        )
    if (
        not np.all(np.isfinite(radius))
        or not np.all(np.isfinite(mass))
        or not np.all(np.isfinite(density))
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
    center = 4.0 * math.pi * central_density * FM3_TO_KM3 * radius[0] ** 3 / 3.0
    baryon_number = center + radial
    baryonic_mass = (
        baryon_number * NEUTRON_REST_MASS_G / SOLAR_MASS_G_FROM_PROJECT_LENGTH
    )
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
                    "radial_span_fraction": (outer_radius - inner_radius)
                    / stellar_radius,
                    "enclosed_mass_span_fraction": (outer_mass - inner_mass)
                    / stellar_mass,
                }
            )
        radial_span = float(sum(item["radial_span_fraction"] for item in intervals))
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


def _diagnostic_case_ids(config, generated):
    """Direct control and signed endpoints within one requested geometry."""
    eligible = [
        (case_id, eos)
        for case_id, eos in generated.items()
        if eos.deformation.amplitude != 0
    ]
    negative = sorted(
        (eos.deformation.amplitude, case_id)
        for case_id, eos in eligible
        if eos.deformation.amplitude < 0
    )
    positive = sorted(
        (
            (eos.deformation.amplitude, case_id)
            for case_id, eos in eligible
            if eos.deformation.amplitude > 0
        ),
        reverse=True,
    )
    selected = ["direct"]
    if config.extended_stellar_diagnostics_case_policy == "all-accepted":
        selected.extend(case_id for _, case_id in (*negative, *positive))
    else:
        if negative:
            selected.append(negative[0][1])
        if positive:
            selected.append(positive[0][1])
    return tuple(selected)


def _extended_diagnostics(
    *,
    packet: Path,
    config: BSk24TrialConfig,
    baseline: BSk24ConsistentBaseline,
    generated: Mapping[str, BSk24WindowedEos],
    sequences: pd.DataFrame,
    fixed: pd.DataFrame,
    stars: Mapping[tuple[str, str, float], Any],
    collect=None,
) -> dict[str, str]:
    """Create bounded, definition-free diagnostics from already planned stars.

    Turning-point refinement, outside-support controls, and matched-area cases
    remain unavailable here because they require separately constructed cases.
    """

    def save_table(frame, path):
        if collect is None:
            write_csv_atomic(frame, path)
        else:
            collect(frame, path.name)

    created: dict[str, str] = {}
    selected = _diagnostic_case_ids(config, generated)
    if not selected:
        return created
    reference_stage = config.tov_stages[-1]
    eos_map: dict[str, Any] = {"direct": baseline.eos, **generated}
    profile_frames: list[pd.DataFrame] = []
    support_rows: list[dict[str, Any]] = []
    baryon_rows: list[dict[str, Any]] = []
    for case_id in selected:
        eos = eos_map[case_id]
        deformation = None if case_id == "direct" else generated[case_id].deformation
        realized_peak_absolute_delta_cs2 = None
        if deformation is not None:
            characterization = window_characterization(baseline, deformation)
            realized_peak_absolute_delta_cs2 = max(
                abs(float(characterization["realized_delta_cs2_minimum"])),
                abs(float(characterization["realized_delta_cs2_maximum"])),
            )
        for mass in config.fixed_masses_msun:
            star = stars.get((case_id, reference_stage.name, mass))
            if star is None:
                continue
            settings = _tov_settings(eos, config, reference_stage)
            pressure = np.asarray(
                pressure_profile_from_solved_star(
                    eos,
                    star,
                    settings=settings,
                    rtol=reference_stage.rtol,
                    atol=reference_stage.atol,
                ),
                dtype=float,
            )
            epsilon = np.asarray(
                eos.energy_density_from_pressure(pressure), dtype=float
            )
            if case_id == "direct":
                density = np.asarray(
                    baseline.consistent_baryon_density_from_energy_density(epsilon),
                    dtype=float,
                )
                central_density = float(
                    baseline.consistent_baryon_density_from_energy_density(
                        star.central_energy_density
                    )
                )
                delta_cs2 = np.zeros_like(epsilon)
            else:
                density = np.asarray(
                    eos.baryon_density_from_energy_density(epsilon), dtype=float
                )
                central_density = float(
                    eos.baryon_density_from_energy_density(star.central_energy_density)
                )
                delta_cs2 = np.asarray(
                    windowed_gaussian_delta_cs2(
                        epsilon,
                        deformation,
                        epsilon_t_mev_fm3=(baseline.anchor.energy_density_mev_fm3),
                    ),
                    dtype=float,
                )
            cs2 = np.asarray([eos(float(value))[1] for value in pressure], dtype=float)
            radius = np.asarray(star.radius_profile, dtype=float)
            enclosed_mass = np.asarray(star.mass_profile, dtype=float)
            profile_frames.append(
                pd.DataFrame(
                    {
                        "case_id": case_id,
                        "target_mass_msun": mass,
                        "radius_km": radius,
                        "radius_over_R": radius / star.radius,
                        "enclosed_mass_msun": enclosed_mass,
                        "enclosed_mass_over_M": enclosed_mass / star.mass,
                        "pressure_mev_fm3": pressure,
                        "energy_density_mev_fm3": epsilon,
                        "baryon_density_fm3": density,
                        "cs2": cs2,
                        "delta_cs2": delta_cs2,
                    }
                )
            )
            baryon = baryon_number_from_profile(
                radius,
                enclosed_mass,
                density,
                central_baryon_density_fm3=central_density,
                gravitational_mass_msun=star.mass,
                solver_rtol=reference_stage.rtol,
                solver_atol=reference_stage.atol,
            )
            baryon_rows.append(
                {
                    "case_id": case_id,
                    "target_mass_msun": mass,
                    "stage": reference_stage.name,
                    **baryon.to_dict(),
                }
            )
            if deformation is not None:
                support = radial_deformation_support(
                    radius,
                    enclosed_mass,
                    delta_cs2,
                    total_radius_km=star.radius,
                    total_mass_msun=star.mass,
                    realized_peak_absolute_delta_cs2=(realized_peak_absolute_delta_cs2),
                    solver_rtol=reference_stage.rtol,
                    solver_atol=reference_stage.atol,
                )
                for fraction, item in support["thresholds"].items():
                    support_rows.append(
                        {
                            "case_id": case_id,
                            "stage": reference_stage.name,
                            "amplitude": deformation.amplitude,
                            "delta_mev_fm3": deformation.delta_mev_fm3,
                            "target_mass_msun": mass,
                            "threshold_fraction": float(fraction),
                            "threshold_label": (
                                "FWHM"
                                if math.isclose(float(fraction), 0.5)
                                else f"{float(fraction):.0%}_of_realized_peak"
                            ),
                            "realized_peak_absolute_delta_cs2": support[
                                "realized_peak_absolute_delta_cs2"
                            ],
                            "stored_profile_peak_absolute_delta_cs2": support[
                                "stored_profile_peak_absolute_delta_cs2"
                            ],
                            "profile_solver_rtol": support["profile_solver_rtol"],
                            "profile_solver_atol": support["profile_solver_atol"],
                            "bounded_mass_reversal_count": support[
                                "bounded_mass_reversal_count"
                            ],
                            "maximum_bounded_mass_reversal_msun": support[
                                "maximum_bounded_mass_reversal_msun"
                            ],
                            "raw_mass_profile_preserved": support[
                                "raw_mass_profile_preserved"
                            ],
                            "threshold_absolute_delta_cs2": item[
                                "threshold_absolute_delta_cs2"
                            ],
                            "status": item["status"],
                            "interval_count": item.get("interval_count", 0),
                            "crossing_count": item.get("crossing_count", 0),
                            "reaches_profile_inner_boundary": item.get(
                                "reaches_profile_inner_boundary", False
                            ),
                            "reaches_profile_outer_boundary": item.get(
                                "reaches_profile_outer_boundary", False
                            ),
                            "inner_radius_fraction": item.get("inner_radius_fraction"),
                            "radial_span_fraction": item.get("radial_span_fraction"),
                            "inner_enclosed_mass_fraction": item.get(
                                "inner_enclosed_mass_fraction"
                            ),
                            "enclosed_mass_span_fraction": item.get(
                                "enclosed_mass_span_fraction"
                            ),
                            "outer_radius_fraction": item.get("outer_radius_fraction"),
                            "outer_enclosed_mass_fraction": item.get(
                                "outer_enclosed_mass_fraction"
                            ),
                        }
                    )
    if profile_frames:
        save_table(
            pd.concat(profile_frames, ignore_index=True),
            packet / "radial_profiles.csv",
        )
        created["radial_profiles.csv"] = (
            "fixed-mass profiles selected by the "
            f"{config.extended_stellar_diagnostics_case_policy} case policy"
        )
    if support_rows:
        save_table(
            pd.DataFrame(support_rows),
            packet / "deformation_support_fractions.csv",
        )
        created["deformation_support_fractions.csv"] = (
            "continuous global-peak deformation thresholds including FWHM"
        )
    baryonic = pd.DataFrame(baryon_rows)
    if not baryonic.empty:
        write_csv_atomic(baryonic, packet / "baryonic_observables.csv")
        created["baryonic_observables.csv"] = (
            "neutron-rest-mass-convention baryon integral"
        )
        direct = baryonic.loc[baryonic.case_id == "direct"].set_index(
            "target_mass_msun"
        )
        responses: list[dict[str, Any]] = []
        for row in baryonic.itertuples(index=False):
            if row.case_id == "direct" or row.target_mass_msun not in direct.index:
                continue
            reference = direct.loc[row.target_mass_msun]
            responses.append(
                {
                    "case_id": row.case_id,
                    "mass_msun": row.target_mass_msun,
                    "delta_baryonic_mass_msun": (
                        row.baryonic_mass_msun - float(reference.baryonic_mass_msun)
                    ),
                    "delta_binding_energy_erg": (
                        row.binding_energy_erg - float(reference.binding_energy_erg)
                    ),
                }
            )
        if responses:
            save_table(
                pd.DataFrame(responses),
                packet / "baryonic_response_across_mass.csv",
            )
            created["baryonic_response_across_mass.csv"] = (
                "within-common-fixed-mass response"
            )
    response_rows: list[dict[str, Any]] = []
    sequence_reference = sequences.loc[
        (sequences.stage == reference_stage.name)
        & (sequences.calculation_status == "success")
    ]
    stable: dict[str, pd.DataFrame] = {}
    for case_id in selected:
        rows = sequence_reference.loc[
            sequence_reference.case_id == case_id
        ].sort_values("P_Central")
        if rows.empty:
            continue
        peak = int(np.argmax(rows.Mass.to_numpy(dtype=float)))
        stable[case_id] = rows.iloc[: peak + 1]
    if set(selected).issubset(stable):
        lower = max(float(frame.Mass.min()) for frame in stable.values())
        upper = min(float(frame.Mass.max()) for frame in stable.values())
        if upper > lower:
            masses = np.linspace(lower, upper, 80)
            direct_values = {
                column: interpolate_within_common_support(
                    stable["direct"].Mass,
                    stable["direct"][column],
                    masses,
                )
                for column in ("Radius", "k2", "Lambda", "Eps_Central")
            }
            for case_id, frame in stable.items():
                values = {
                    column: interpolate_within_common_support(
                        frame.Mass, frame[column], masses
                    )
                    for column in ("Radius", "k2", "Lambda", "Eps_Central")
                }
                for index, mass in enumerate(masses):
                    response_rows.append(
                        {
                            "case_id": case_id,
                            "mass_msun": mass,
                            "delta_radius_km": values["Radius"][index]
                            - direct_values["Radius"][index],
                            "delta_k2": values["k2"][index]
                            - direct_values["k2"][index],
                            "delta_lambda": values["Lambda"][index]
                            - direct_values["Lambda"][index],
                            "central_epsilon_mev_fm3": values["Eps_Central"][index],
                            "interpolation": (
                                "PCHIP_within_common_successful_stable_prefix"
                            ),
                        }
                    )
    if response_rows:
        save_table(
            pd.DataFrame(response_rows),
            packet / "stellar_response_across_mass.csv",
        )
        created["stellar_response_across_mass.csv"] = (
            "common successful stable-prefix response"
        )
    odd_rows: list[dict[str, Any]] = []
    reference_fixed = fixed.loc[
        (fixed.stage == reference_stage.name) & (fixed.status == "bracketed_and_solved")
    ]
    zero_cases = reference_fixed.loc[
        np.isclose(reference_fixed.amplitude, 0.0, equal_nan=False)
    ]
    for delta in config.deltas_mev_fm3:
        zero = zero_cases.loc[np.isclose(zero_cases.delta_mev_fm3, delta)]
        for amplitude in sorted(
            {abs(value) for value in config.effective_amplitudes if value != 0.0}
        ):
            plus = reference_fixed.loc[
                np.isclose(reference_fixed.amplitude, amplitude)
                & np.isclose(reference_fixed.delta_mev_fm3, delta)
            ]
            minus = reference_fixed.loc[
                np.isclose(reference_fixed.amplitude, -amplitude)
                & np.isclose(reference_fixed.delta_mev_fm3, delta)
            ]
            if zero.empty or plus.empty or minus.empty:
                continue
            for mass in config.fixed_masses_msun:
                z = zero.loc[np.isclose(zero.target_mass_msun, mass)]
                p = plus.loc[np.isclose(plus.target_mass_msun, mass)]
                m = minus.loc[np.isclose(minus.target_mass_msun, mass)]
                if z.empty or p.empty or m.empty:
                    continue
                for observable in (
                    "radius_km",
                    "k2",
                    "lambda_dimensionless",
                ):
                    values = (
                        float(p.iloc[0][observable]),
                        float(m.iloc[0][observable]),
                        float(z.iloc[0][observable]),
                    )
                    if not np.all(np.isfinite(values)):
                        continue
                    item = odd_even_response(
                        *values,
                        amplitude=amplitude,
                        numerical_envelope=(
                            np.finfo(float).eps * max(1.0, abs(values[2]))
                        ),
                    )
                    odd_rows.append(
                        {
                            "delta_mev_fm3": delta,
                            "target_mass_msun": mass,
                            "observable": observable,
                            **item,
                        }
                    )
    if odd_rows:
        save_table(pd.DataFrame(odd_rows), packet / "odd_even_response.csv")
        created["odd_even_response.csv"] = "paired amplitudes with A=0"
    error_rows: list[dict[str, Any]] = []
    if len(config.tov_stages) >= 2:
        for (case_id, target_mass), rows in fixed.loc[
            fixed.status == "bracketed_and_solved"
        ].groupby(["case_id", "target_mass_msun"]):
            for observable in (
                "radius_km",
                "k2",
                "lambda_dimensionless",
            ):
                values = pd.to_numeric(rows[observable], errors="coerce").dropna()
                if len(values) >= 2:
                    error_rows.append(
                        {
                            "label": f"{case_id}:{target_mass:g}:{observable}",
                            "observable": observable,
                            "numerical_envelope": float(values.max() - values.min()),
                            "reference_value": float(values.iloc[-1]),
                        }
                    )
    if error_rows:
        save_table(pd.DataFrame(error_rows), packet / "numerical_error_summary.csv")
        created["numerical_error_summary.csv"] = "same-case stage spans"
    return created


def write_diagnostics(
    *, packet, configs, baseline, generated, sequences, fixed, stars, baseline_id
):
    """Collect the existing diagnostic computation once per declared geometry."""
    from collections import defaultdict

    frames = defaultdict(list)
    description = {}
    geometry_cases = {}
    for index, config in enumerate(configs, 1):
        geometry = (
            config.epsilon0_mev_fm3,
            config.sigma_mev_fm3,
            config.deltas_mev_fm3[0],
        )
        selected = {
            key: eos
            for key, eos in generated.items()
            if eos.deformation.amplitude != 0
            and (
                eos.deformation.epsilon0_mev_fm3,
                eos.deformation.sigma_mev_fm3,
                eos.deformation.delta_mev_fm3,
            )
            == geometry
        }
        sequence = (
            sequences.loc[sequences.case_id.isin([baseline_id, *selected])]
            .replace({"case_id": {baseline_id: "direct"}})
            .copy()
        )
        targets = (
            fixed.loc[fixed.case_id.isin([baseline_id, *selected])]
            .replace({"case_id": {baseline_id: "direct"}})
            .copy()
        )
        targets.loc[targets.case_id.eq("direct"), "amplitude"] = 0.0
        targets.loc[targets.case_id.eq("direct"), "delta_mev_fm3"] = (
            config.deltas_mev_fm3[0]
        )
        geometry_cases[str(index)] = [
            baseline_id if case_id == "direct" else case_id
            for case_id in _diagnostic_case_ids(config, selected)
        ]

        def collect(frame, name):
            frame = frame.copy()
            frame.insert(0, "geometry_index", index)
            for column in frame:
                if "case_id" in column:
                    frame[column] = frame[column].replace("direct", baseline_id)
            frames[name].append(frame)

        description.update(
            _extended_diagnostics(
                packet=packet,
                config=config,
                baseline=baseline,
                generated=selected,
                sequences=sequence,
                fixed=targets,
                stars=stars,
                collect=collect,
            )
        )
    for name, tables in frames.items():
        write_csv_atomic(pd.concat(tables, ignore_index=True), packet / name)
    return {
        "case_policy": "direct_control_and_signed_endpoints_per_geometry",
        "selected_case_ids_by_geometry": geometry_cases,
        "tables": description,
    }

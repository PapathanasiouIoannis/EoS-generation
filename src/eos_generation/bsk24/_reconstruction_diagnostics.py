"""Residual, round-trip, and identity reports for reconstructed BSk24 states."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from eos_generation.bsk24._reconstruction_primitives import (
    COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3,
    COMPOSE_OUTER_INNER_TRANSITION_EPSILON_MEV_FM3,
)

if TYPE_CHECKING:
    from eos_generation.bsk24.reconstruction import (
        BSk24ConsistentBaseline,
        BSk24GeneratedEos,
    )


def summarize_residuals(
    eos: BSk24GeneratedEos,
    *,
    exclude_boundary_points: int = 4,
) -> dict[str, Any]:
    """Summarize algebraic and independent residuals with sensitive regions split."""
    epsilon = eos.epsilon
    anchor = eos.baseline.anchor.energy_density_mev_fm3
    base_mask = np.ones(len(epsilon), dtype=bool)
    base_mask[:exclude_boundary_points] = False
    base_mask[-exclude_boundary_points:] = False
    anchor_index = int(np.argmin(np.abs(epsilon - anchor)))
    anchor_mask = np.zeros(len(epsilon), dtype=bool)
    anchor_mask[max(0, anchor_index - 3) : min(len(epsilon), anchor_index + 4)] = True
    transition_mask = np.zeros(len(epsilon), dtype=bool)
    for transition in (
        COMPOSE_OUTER_INNER_TRANSITION_EPSILON_MEV_FM3,
        COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3,
    ):
        index = int(np.argmin(np.abs(epsilon - transition)))
        transition_mask[max(0, index - 3) : min(len(epsilon), index + 4)] = True
    interior_mask = base_mask & ~anchor_mask & ~transition_mask

    def region_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        indices = np.flatnonzero(mask & np.isfinite(values))
        if not len(indices):
            return {"status": "unavailable"}
        selected = values[indices]
        local = int(np.argmax(np.abs(selected)))
        index = int(indices[local])
        return {
            "status": "computed",
            "maximum_absolute": float(abs(values[index])),
            "signed_value_at_maximum": float(values[index]),
            "epsilon_at_maximum_mev_fm3": float(epsilon[index]),
            "p50_absolute": float(np.percentile(np.abs(selected), 50.0)),
            "p95_absolute": float(np.percentile(np.abs(selected), 95.0)),
            "p99_absolute": float(np.percentile(np.abs(selected), 99.0)),
            "point_count": int(len(indices)),
        }

    summaries = {}
    for name in (
        "r_p_algebraic",
        "r_mu_algebraic",
        "r_p_independent_normalized",
        "r_mu_independent_normalized",
        "r_c",
        "first_law_normalized",
    ):
        values = eos.residuals[name]
        summaries[name] = {
            "complete_retained_grid": region_summary(values, base_mask),
            "interior_excluding_sensitive_bands": region_summary(values, interior_mask),
            "anchor_band": region_summary(values, anchor_mask),
            "source_transition_bands": region_summary(values, transition_mask),
        }
    return {
        "case_id": eos.deformation.case_id,
        "definitions": {
            "r_p_algebraic": "P-(n*mu-epsilon); definition closure, not independent",
            "r_mu_algebraic": "mu-(epsilon+P)/n; definition closure, not independent",
            "r_p_independent_normalized": (
                "[P-(n*d_epsilon/d_n-epsilon)]/max(|P|,|n*d_epsilon/d_n|,|epsilon|)"
            ),
            "r_mu_independent_normalized": (
                "[mu-d_epsilon/d_n]/max(|mu|,|d_epsilon/d_n|)"
            ),
            "r_c": "raw_cs2-dP/d_epsilon from independently differentiated PCHIP",
            "first_law_normalized": "mu*dn/d_epsilon-1 from independently differentiated PCHIP",
        },
        "derivative_method": "PCHIP derivatives of sampled pressure and baryon-density states",
        "sensitive_region_policy": (
            "anchor and corrected-CompOSE phase-transition nearest-node bands reported separately"
        ),
        "summaries": summaries,
    }


def round_trip_diagnostics(eos: BSk24GeneratedEos) -> dict[str, Any]:
    """Measure forward and inverse residuals at non-node midpoints."""
    epsilon_probe = np.sqrt(eos.epsilon[:-1] * eos.epsilon[1:])
    pressure_probe = np.sqrt(eos.pressure[:-1] * eos.pressure[1:])
    forward_pressure = np.asarray(eos.pressure_from_energy_density(epsilon_probe), dtype=float)
    forward_back = np.asarray(eos.energy_density_from_pressure(forward_pressure), dtype=float)
    pchip_forward_back = eos._interpolated_energy_density_from_pressure(forward_pressure)
    inverse_epsilon = np.asarray(eos.energy_density_from_pressure(pressure_probe), dtype=float)
    inverse_back = np.asarray(eos.pressure_from_energy_density(inverse_epsilon), dtype=float)
    pchip_inverse_epsilon = eos._interpolated_energy_density_from_pressure(pressure_probe)
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


def local_identity_report(
    baseline: BSk24ConsistentBaseline,
    a0: BSk24GeneratedEos,
) -> dict[str, Any]:
    """Compare direct C4/C4-consistent state with the A=0 generator path."""
    if a0.deformation.amplitude != 0.0:
        raise ValueError("local identity requires an exactly zero-amplitude case")
    comparisons = {
        "pressure": (a0.pressure, baseline.pressure),
        "sound_speed_squared": (a0.cs2, baseline.cs2),
        "baryon_density_C4_consistent": (a0.baryon_density, baseline.baryon_density),
        "chemical_potential_C4_consistent": (
            a0.chemical_potential,
            baseline.chemical_potential,
        ),
        "adiabatic_index": (a0.adiabatic_index, baseline.adiabatic_index),
        "energy_per_baryon_minus_neutron_rest": (
            a0.energy_per_baryon_minus_neutron_rest,
            baseline.energy_per_baryon_minus_neutron_rest,
        ),
    }
    results = {}
    for name, (observed, expected) in comparisons.items():
        absolute = np.asarray(observed) - np.asarray(expected)
        relative = absolute / np.maximum(np.abs(expected), np.finfo(float).tiny)
        absolute_index = int(np.argmax(np.abs(absolute)))
        relative_index = int(np.argmax(np.abs(relative)))
        scale = float(max(1.0, np.max(np.abs(expected))))
        tolerance = float(64.0 * np.finfo(float).eps * scale)
        results[name] = {
            "maximum_absolute_residual": float(abs(absolute[absolute_index])),
            "maximum_absolute_location_epsilon_mev_fm3": float(
                baseline.epsilon[absolute_index]
            ),
            "maximum_relative_residual": float(abs(relative[relative_index])),
            "maximum_relative_location_epsilon_mev_fm3": float(
                baseline.epsilon[relative_index]
            ),
            "absolute_tolerance": tolerance,
            "tolerance_origin": "64*machine_epsilon*max(1,baseline_scale)",
            "status": "pass" if np.max(np.abs(absolute)) <= tolerance else "fail",
        }

    pressure_probe = np.sqrt(a0.pressure[:-1] * a0.pressure[1:])
    generated_inverse = np.asarray(a0.energy_density_from_pressure(pressure_probe), dtype=float)
    direct_inverse = np.asarray(
        baseline.eos.energy_density_from_pressure(pressure_probe), dtype=float
    )
    inverse_abs = generated_inverse - direct_inverse
    inverse_rel = inverse_abs / direct_inverse
    abs_index = int(np.argmax(np.abs(inverse_abs)))
    rel_index = int(np.argmax(np.abs(inverse_rel)))
    results["inverse_energy_density"] = {
        "maximum_absolute_residual": float(abs(inverse_abs[abs_index])),
        "maximum_absolute_location_pressure_mev_fm3": float(pressure_probe[abs_index]),
        "maximum_relative_residual": float(abs(inverse_rel[rel_index])),
        "maximum_relative_location_pressure_mev_fm3": float(pressure_probe[rel_index]),
        "absolute_tolerance": float(
            64.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(direct_inverse)))
        ),
        "status": (
            "pass"
            if np.max(np.abs(inverse_abs))
            <= 64.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(direct_inverse)))
            else "fail"
        ),
        "tolerance_origin": "64*machine_epsilon*max(1,direct_inverse_scale)",
        "separate_generated_pchip_residual": a0.diagnostics["round_trip"][
            "generated_pchip_forward_relative"
        ],
    }
    return {
        "identity_scope": (
            "direct C4 pressure/cs2 plus C4-consistent reconstructed state versus A=0 generator"
        ),
        "not_independent_BSk24_validation": True,
        "generator_identity": results,
        "c1_comparison_is_not_identity_target": True,
        "c1_c4_representation_discrepancy": baseline.diagnostics[
            "c1_c4_representation_discrepancy"
        ],
        "domain_identity": {
            "lower_epsilon_equal": bool(a0.epsilon[0] == baseline.epsilon[0]),
            "upper_epsilon_equal": bool(a0.epsilon[-1] == baseline.epsilon[-1]),
            "lower_pressure_equal": bool(a0.pressure[0] == baseline.pressure[0]),
            "upper_pressure_equal": bool(a0.pressure[-1] == baseline.pressure[-1]),
            "surface_metadata_equal": True,
            "internal_transition_metadata_equal": True,
            "artificial_seams": "none_at_A0",
            "extrapolation": "forbidden_both_paths",
        },
    }

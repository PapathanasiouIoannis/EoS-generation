"""Thermodynamic reporting helpers for governed BSk24 trials.

This internal module is deliberately independent of the public experiment
facade and performs no scientific work at import time.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from eos_generation._internal.planning import (
    BSk24TrialConfig,
    BSk24TrialPlan,
)
from eos_generation.bsk24.reconstruction import BSk24ConsistentBaseline
from eos_generation.bsk24.deformation import (
    BSk24WindowedDeformation,
    BSk24WindowedEos,
    gaussian_profile,
    smootherstep_window,
    windowed_a0_identity_report,
    windowed_gaussian_delta_cs2,
)


def _maximum_absolute_residual(
    left: np.ndarray, right: np.ndarray
) -> float | None:
    """Return a finite comparison residual without warning on all-NaN data."""

    if not len(left):
        return None
    with np.errstate(invalid="ignore"):
        absolute_difference = np.abs(left - right)
    if bool(np.isnan(absolute_difference).all()):
        return None
    return float(np.nanmax(absolute_difference))


def _deformations(
    plan: BSk24TrialPlan,
) -> dict[str, BSk24WindowedDeformation]:
    return {
        str(row.case_id): BSk24WindowedDeformation(
            case_id=str(row.case_id),
            amplitude=float(row.amplitude),
            epsilon0_mev_fm3=float(row.epsilon0_mev_fm3),
            sigma_mev_fm3=float(row.sigma_mev_fm3),
            delta_mev_fm3=float(row.delta_mev_fm3),
        )
        for row in plan.case_table.itertuples(index=False)
    }


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
            baseline.eos.pressure_from_energy_density(epsilon), dtype=float
        )
        direct_density = np.asarray(
            baseline.consistent_baryon_density_from_energy_density(epsilon), dtype=float
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
                    "pressure_relative_to_direct": (
                        eos.pressure - direct_pressure
                    )
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
        None
        if value is None or not math.isfinite(float(value))
        else float(value)
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
        item["absolute_change"] for item in pairwise if math.isfinite(item["absolute_change"])
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

def _a0_identity_table(
    *,
    baseline: BSk24ConsistentBaseline,
    generated: Mapping[str, BSk24WindowedEos],
    config: BSk24TrialConfig,
    sequences: pd.DataFrame | None,
    fixed: pd.DataFrame | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = (
        "scope",
        "delta_mev_fm3",
        "stage",
        "quantity",
        "maximum_absolute_residual",
        "array_equal",
        "status",
    )
    a0_cases = {
        eos.deformation.delta_mev_fm3: eos
        for eos in generated.values()
        if eos.deformation.amplitude == 0.0
    }
    if not a0_cases:
        local = {
            "schema_id": "bsk24_windowed_a0_identity_v1",
            "status": "unavailable",
            "reason": "the injected A=0 proposal did not pass the raw local-physics gate",
            "deltas": {},
        }
        report = {
            "schema_id": "eos_generation_a0_identity_v1",
            "identity_target": (
                "direct C4 pressure/sound speed plus C4-consistent reconstruction, "
                "C1-normalized at the anchor"
            ),
            "a0_was_injected": config.a0_was_injected,
            "local_thermodynamic_identity": local,
            "stellar_identity_status": "unavailable_due_to_raw_gate_rejection",
            "status": "unavailable",
        }
        return report, pd.DataFrame(columns=columns)

    local = windowed_a0_identity_report(baseline, a0_cases)
    rows: list[dict[str, Any]] = []
    for delta, report in local["deltas"].items():
        for quantity, item in report.items():
            rows.append(
                {
                    "scope": "thermodynamic",
                    "delta_mev_fm3": float(delta),
                    "stage": "refined",
                    "quantity": quantity,
                    "maximum_absolute_residual": item["maximum_absolute_residual"],
                    "array_equal": item["array_equal"],
                    "status": item["status"],
                }
            )
    stellar_status = "not_requested"
    if sequences is not None and fixed is not None and not sequences.empty:
        stellar_status = "pass"
        for delta, eos in a0_cases.items():
            case_id = eos.deformation.case_id
            for stage in config.tov_stages:
                direct = sequences.loc[
                    (sequences.case_id == "direct") & (sequences.stage == stage.name)
                ].reset_index(drop=True)
                a0 = sequences.loc[
                    (sequences.case_id == case_id) & (sequences.stage == stage.name)
                ].reset_index(drop=True)
                for quantity in ("Mass", "Radius", "Lambda", "P_Central", "Eps_Central"):
                    left = direct[quantity].to_numpy(dtype=float)
                    right = a0[quantity].to_numpy(dtype=float)
                    equal = bool(np.array_equal(left, right, equal_nan=True))
                    residual = _maximum_absolute_residual(left, right)
                    status = "pass" if equal else "fail"
                    if status == "fail":
                        stellar_status = "fail"
                    rows.append(
                        {
                            "scope": "stellar_sequence",
                            "delta_mev_fm3": delta,
                            "stage": stage.name,
                            "quantity": quantity,
                            "maximum_absolute_residual": residual,
                            "array_equal": equal,
                            "status": status,
                        }
                    )
    report = {
        "schema_id": "eos_generation_a0_identity_v1",
        "identity_target": (
            "direct C4 pressure/sound speed plus C4-consistent reconstruction, "
            "C1-normalized at the anchor"
        ),
        "a0_was_injected": config.a0_was_injected,
        "local_thermodynamic_identity": local,
        "stellar_identity_status": stellar_status,
        "status": (
            "pass"
            if local["status"] == "pass" and stellar_status in {"pass", "not_requested"}
            else "fail"
        ),
    }
    return report, pd.DataFrame(rows, columns=columns)

__all__ = [
    "_REFINEMENT_ULP_ALLOWANCE",
    "_THERMODYNAMIC_CONVERGENCE_METRICS",
    "_THERMODYNAMIC_RESIDUAL_REGIONS",
    "_a0_identity_table",
    "_classify_refinement_series",
    "_deformations",
    "_raw_gate_frame",
    "_refinement_pair_allowance",
    "_thermodynamic_convergence",
    "_thermodynamic_profile_frame",
    "_thermodynamic_residual_frame",
]

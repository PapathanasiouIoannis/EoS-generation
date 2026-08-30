"""CFL-specific adapters and saved thermodynamic evidence.

This module deliberately contains no hadronic/crust assumptions.  It maps the
shared packet lifecycle onto the frozen, bare, self-bound CFL surface and
retains physical case identities separately from logical Cartesian aliases.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from eos_generation._internal.thermodynamics import (
    _classify_refinement_series,
)
from eos_generation.cfl.baseline import (
    ENERGY_DENSITY_SURFACE_MEV_FM3,
    FROZEN_PARAMETER_SET_SHA256,
)
from eos_generation.cfl.deformation import (
    CFLWindowedDeformation,
    gaussian_profile,
    smootherstep_window,
    windowed_gaussian_delta_cs2,
)


_RESIDUAL_METRICS = (
    "r_p_independent_normalized",
    "r_mu_independent_normalized",
    "first_law_normalized",
    "r_c",
)


def _cfl_deformations(plan: Any) -> dict[str, CFLWindowedDeformation]:
    """Resolve one owned deformation per physical identity."""

    result: dict[str, CFLWindowedDeformation] = {}
    for row in plan.case_table.itertuples(index=False):
        if not bool(getattr(row, "planned_for_execution", True)):
            continue
        physical_case_id = str(getattr(row, "physical_case_id", row.case_id))
        deformation = CFLWindowedDeformation(
            case_id=physical_case_id,
            amplitude=float(row.amplitude),
            center_mev_fm3=float(row.epsilon0_mev_fm3),
            width_mev_fm3=float(row.sigma_mev_fm3),
            ramp_width_mev_fm3=float(row.delta_mev_fm3),
        )
        previous = result.get(physical_case_id)
        if previous is not None and previous != deformation:
            raise RuntimeError(
                f"physical CFL case collision for {physical_case_id!r}"
            )
        result[physical_case_id] = deformation
    return result


def _cfl_raw_gate_frame(
    *,
    case_id: str,
    deformation: CFLWindowedDeformation,
    baseline: Any,
    epsilon: np.ndarray,
    raw_cs2: np.ndarray,
    status: str,
) -> pd.DataFrame:
    epsilon = np.asarray(epsilon, dtype=float)
    return pd.DataFrame(
        {
            "case_id": case_id,
            "physical_case_id": case_id,
            "matter_model": "cfl",
            "baseline_parameter_set_sha256": FROZEN_PARAMETER_SET_SHA256,
            "amplitude": deformation.amplitude,
            "epsilon0_mev_fm3": deformation.center_mev_fm3,
            "sigma_mev_fm3": deformation.width_mev_fm3,
            "delta_mev_fm3": deformation.ramp_width_mev_fm3,
            "epsilon_mev_fm3": epsilon,
            "window": np.asarray(
                smootherstep_window(
                    epsilon,
                    ramp_width_mev_fm3=deformation.ramp_width_mev_fm3,
                ),
                dtype=float,
            ),
            "gaussian": np.asarray(
                gaussian_profile(epsilon, deformation), dtype=float
            ),
            "delta_cs2": np.asarray(
                windowed_gaussian_delta_cs2(epsilon, deformation), dtype=float
            ),
            "raw_cs2": np.asarray(raw_cs2, dtype=float),
            "gate_status": status,
        }
    )


def _safe_relative(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0.0)
    return result


def _gamma(epsilon: np.ndarray, pressure: np.ndarray, cs2: np.ndarray) -> np.ndarray:
    result = np.full_like(epsilon, np.nan, dtype=float)
    np.divide(
        (epsilon + pressure) * cs2,
        pressure,
        out=result,
        where=pressure > 0.0,
    )
    return result


def _profile_rows(
    *,
    case_id: str,
    amplitude: float | None,
    delta: float | None,
    epsilon: np.ndarray,
    pressure: np.ndarray,
    cs2: np.ndarray,
    density: np.ndarray,
    mu_b: np.ndarray,
    direct_pressure: np.ndarray,
    direct_density: np.ndarray,
    direct_mu_b: np.ndarray,
    delta_cs2: np.ndarray,
    quark_mu: np.ndarray | None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": case_id,
            "physical_case_id": case_id,
            "matter_model": "cfl",
            "amplitude": np.nan if amplitude is None else amplitude,
            "delta_mev_fm3": np.nan if delta is None else delta,
            "epsilon_mev_fm3": epsilon,
            "pressure_mev_fm3": pressure,
            "cs2": cs2,
            "delta_cs2": delta_cs2,
            "baryon_density_fm3": density,
            "effective_baryon_enthalpy_mev": mu_b,
            "baryon_chemical_potential_mev": mu_b,
            "quark_chemical_potential_mev": (
                np.full_like(epsilon, np.nan) if quark_mu is None else quark_mu
            ),
            "gamma_eff": _gamma(epsilon, pressure, cs2),
            "energy_per_baryon_minus_neutron_rest_mev": (
                epsilon / density - 939.5654
            ),
            "pressure_relative_to_direct": _safe_relative(
                pressure - direct_pressure, direct_pressure
            ),
            "baryon_density_relative_to_direct": (
                (density - direct_density) / direct_density
            ),
            "enthalpy_relative_to_direct": (
                (mu_b - direct_mu_b) / direct_mu_b
            ),
        }
    )


def _cfl_thermodynamic_profile_frame(
    baseline: Any,
    generated: Mapping[str, Any],
) -> pd.DataFrame:
    epsilon = np.asarray(baseline.epsilon, dtype=float)
    pressure = np.asarray(baseline.pressure, dtype=float)
    density = np.asarray(baseline.baryon_density, dtype=float)
    mu_b = np.asarray(baseline.baryon_chemical_potential, dtype=float)
    frames = [
        _profile_rows(
            case_id="direct",
            amplitude=None,
            delta=None,
            epsilon=epsilon,
            pressure=pressure,
            cs2=np.asarray(baseline.cs2, dtype=float),
            density=density,
            mu_b=mu_b,
            direct_pressure=pressure,
            direct_density=density,
            direct_mu_b=mu_b,
            delta_cs2=np.zeros_like(epsilon),
            quark_mu=np.asarray(
                baseline.quark_chemical_potential, dtype=float
            ),
        )
    ]
    case_count = len(generated)
    for case_index, (case_id, eos) in enumerate(generated.items(), start=1):
        epsilon = np.asarray(eos.epsilon, dtype=float)
        # All three direct-reference columns share the same governed
        # epsilon->mu_q inversion.  The former separate public calls solved
        # this identical Brent problem three times for every retained node.
        direct_quark_mu = np.asarray(
            baseline.quark_chemical_potential_from_energy_density(epsilon),
            dtype=float,
        )
        direct_pressure = np.asarray(
            baseline.pressure_from_quark_chemical_potential(direct_quark_mu),
            dtype=float,
        )
        direct_density = np.asarray(
            baseline.baryon_density_from_quark_chemical_potential(
                direct_quark_mu
            ),
            dtype=float,
        )
        direct_mu_b = np.asarray(
            baseline.baryon_chemical_potential_from_quark_chemical_potential(
                direct_quark_mu
            ),
            dtype=float,
        )
        frames.append(
            _profile_rows(
                case_id=case_id,
                amplitude=eos.deformation.amplitude,
                delta=eos.deformation.ramp_width_mev_fm3,
                epsilon=epsilon,
                pressure=np.asarray(eos.pressure, dtype=float),
                cs2=np.asarray(eos.cs2, dtype=float),
                density=np.asarray(eos.baryon_density, dtype=float),
                mu_b=np.asarray(eos.baryon_chemical_potential, dtype=float),
                direct_pressure=direct_pressure,
                direct_density=direct_density,
                direct_mu_b=direct_mu_b,
                delta_cs2=np.asarray(
                    windowed_gaussian_delta_cs2(epsilon, eos.deformation),
                    dtype=float,
                ),
                quark_mu=None,
            )
        )
        if os.environ.get("EOS_GENERATION_PROGRESS") == "1":
            print(
                f"[CFL] thermodynamic profile {case_index}/{case_count}: "
                f"{case_id}",
                flush=True,
            )
    return pd.concat(frames, ignore_index=True)


def _residual_arrays(eos: Any) -> dict[str, np.ndarray]:
    stored = getattr(eos, "residuals", None)
    if isinstance(stored, Mapping) and all(
        name in stored for name in _RESIDUAL_METRICS
    ):
        return {
            name: np.asarray(stored[name], dtype=float)
            for name in _RESIDUAL_METRICS
        }
    epsilon = np.asarray(eos.epsilon, dtype=float)
    pressure = np.asarray(eos.pressure, dtype=float)
    cs2 = np.asarray(eos.cs2, dtype=float)
    density = np.asarray(eos.baryon_density, dtype=float)
    mu_b = np.asarray(eos.baryon_chemical_potential, dtype=float)
    edge_order = 2 if len(epsilon) >= 3 else 1
    dp_de = np.gradient(pressure, epsilon, edge_order=edge_order)
    dn_de = np.gradient(density, epsilon, edge_order=edge_order)
    scale = np.maximum.reduce(
        (np.abs(pressure), np.abs(density * mu_b), np.abs(epsilon))
    )
    return {
        "r_p_independent_normalized": (
            pressure - (density * mu_b - epsilon)
        )
        / scale,
        "r_mu_independent_normalized": (
            mu_b - (epsilon + pressure) / density
        )
        / mu_b,
        "first_law_normalized": 1.0 - mu_b * dn_de,
        "r_c": cs2 - dp_de,
    }


def _cfl_thermodynamic_residual_frame(
    generated: Mapping[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for case_id, eos in generated.items():
        frame = pd.DataFrame(
            {
                "case_id": case_id,
                "physical_case_id": case_id,
                "matter_model": "cfl",
                "amplitude": eos.deformation.amplitude,
                "delta_mev_fm3": eos.deformation.ramp_width_mev_fm3,
                "epsilon_mev_fm3": eos.epsilon,
            }
        )
        for name, values in _residual_arrays(eos).items():
            frame[name] = values
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _interior_maximum(values: np.ndarray) -> float | None:
    interior = np.asarray(values, dtype=float)[1:-1]
    finite = interior[np.isfinite(interior)]
    return None if not len(finite) else float(np.max(np.abs(finite)))


def _cfl_thermodynamic_convergence(
    stage_cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    stages = tuple(stage_cases)
    case_ids = sorted(
        set().union(*(set(cases) for cases in stage_cases.values()))
        if stage_cases
        else set()
    )
    residuals_by_stage_case = {
        (stage, case_id): _residual_arrays(eos)
        for stage, cases in stage_cases.items()
        for case_id, eos in cases.items()
    }
    reports: dict[str, Any] = {}
    statuses: list[str] = []
    for case_id in case_ids:
        reports[case_id] = {}
        for metric in _RESIDUAL_METRICS:
            maxima: dict[str, float | None] = {}
            for stage in stages:
                eos = stage_cases[stage].get(case_id)
                maxima[stage] = (
                    None
                    if eos is None
                    else _interior_maximum(
                        residuals_by_stage_case[(stage, case_id)][metric]
                    )
                )
            classification = _classify_refinement_series(
                [maxima[stage] for stage in stages]
            )
            statuses.append(str(classification["status"]))
            reports[case_id][metric] = {
                "interior_maximum_absolute_by_stage": maxima,
                **classification,
            }
    if not reports:
        status = "no_accepted_cases"
    elif len(stages) < 3:
        status = "insufficient_stages"
    elif any(value == "nonfinite_or_missing_evidence" for value in statuses):
        status = "nonfinite_or_missing_evidence"
    elif any(value == "mixed_or_nonmonotone_refinement" for value in statuses):
        status = "mixed_or_nonmonotone_refinement"
    elif all(
        value == "pass_monotonically_decreasing_interior_residuals"
        for value in statuses
    ):
        status = "pass_monotonically_decreasing_interior_residuals"
    else:
        status = "measured_refinement_without_strict_contraction"
    return {
        "schema_id": "eos_generation_cfl_thermodynamic_convergence_v1",
        "matter_model": "cfl",
        "stages": list(stages),
        "stage_count": len(stages),
        "required_metrics": list(_RESIDUAL_METRICS),
        "finite_difference_evidence": (
            "independent numpy gradient on governed reconstruction nodes"
        ),
        "cases": reports,
        "status": status,
    }


def _maximum_absolute(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _cfl_a0_identity_table(
    *,
    baseline: Any,
    generated: Mapping[str, Any],
    config: Any,
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
    zero_cases = {
        case_id: eos
        for case_id, eos in generated.items()
        if eos.deformation.amplitude == 0.0
    }
    owner = bool(getattr(config, "zero_amplitude_control_owner", True))
    if not zero_cases:
        passed = not owner
        report = {
            "schema_id": "eos_generation_cfl_a0_identity_v1",
            "identity_target": "one globally owned analytic CFL baseline",
            "zero_amplitude_control_owner": owner,
            "physical_zero_case_ids": [],
            "stellar_identity_status": (
                "not_applicable_no_owned_a0_case"
                if passed
                else "fail_missing_or_rejected_owned_a0_case"
            ),
            "status": "pass" if passed else "fail",
        }
        return report, pd.DataFrame(columns=columns)

    if not owner or len(zero_cases) != 1:
        report = {
            "schema_id": "eos_generation_cfl_a0_identity_v1",
            "identity_target": "one globally owned analytic CFL baseline",
            "zero_amplitude_control_owner": owner,
            "physical_zero_case_ids": sorted(zero_cases),
            "stellar_identity_status": (
                "fail_unexpected_nonowner_physical_a0_case"
                if not owner
                else "fail_multiple_owned_physical_a0_cases"
            ),
            "status": "fail",
        }
        return report, pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    comparisons = {
        "epsilon": np.asarray(baseline.epsilon),
        "pressure": np.asarray(baseline.pressure),
        "cs2": np.asarray(baseline.cs2),
        "baryon_density": np.asarray(baseline.baryon_density),
        "baryon_chemical_potential": np.asarray(
            baseline.baryon_chemical_potential
        ),
    }
    local_pass = True
    for eos in zero_cases.values():
        generated_values = {
            "epsilon": eos.epsilon,
            "pressure": eos.pressure,
            "cs2": eos.cs2,
            "baryon_density": eos.baryon_density,
            "baryon_chemical_potential": eos.baryon_chemical_potential,
        }
        for quantity, direct in comparisons.items():
            candidate = np.asarray(generated_values[quantity])
            equal = bool(np.array_equal(direct, candidate, equal_nan=True))
            local_pass = local_pass and equal
            rows.append(
                {
                    "scope": "thermodynamic",
                    "delta_mev_fm3": eos.deformation.ramp_width_mev_fm3,
                    "stage": "reference",
                    "quantity": quantity,
                    "maximum_absolute_residual": _maximum_absolute(
                        direct, candidate
                    ),
                    "array_equal": equal,
                    "status": "pass" if equal else "fail",
                }
            )
    stellar_requested = bool(
        getattr(config, "background_tov_requested", False)
    )
    if not stellar_requested:
        stellar_status = "not_requested"
    elif sequences is None or sequences.empty:
        stellar_status = "fail_duplicate_or_missing_direct_solution"
    else:
        zero_ids = set(zero_cases)
        saved_ids = set(sequences.get("case_id", pd.Series(dtype=str)).astype(str))
        stellar_status = (
            "pass_shared_direct_solution_alias"
            if "direct" in saved_ids and zero_ids.isdisjoint(saved_ids)
            else "fail_duplicate_or_missing_direct_solution"
        )
    passed = local_pass and stellar_status in {
        "not_requested",
        "pass_shared_direct_solution_alias",
    }
    report = {
        "schema_id": "eos_generation_cfl_a0_identity_v1",
        "identity_target": (
            "formula-derived frozen CFL arrays and delegated analytic evaluators"
        ),
        "floating_point_policy": "numpy.array_equal_binary64",
        "zero_amplitude_control_owner": owner,
        "physical_zero_case_ids": sorted(zero_cases),
        "stellar_identity_status": stellar_status,
        "duplicate_zero_amplitude_stellar_solver_calls": 0,
        "status": "pass" if passed else "fail",
    }
    return report, pd.DataFrame(rows, columns=columns)


__all__ = [
    "_cfl_a0_identity_table",
    "_cfl_deformations",
    "_cfl_raw_gate_frame",
    "_cfl_thermodynamic_convergence",
    "_cfl_thermodynamic_profile_frame",
    "_cfl_thermodynamic_residual_frame",
]

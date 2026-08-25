"""Stellar execution and saved-table reporting for BSk24 trials.

This internal module is deliberately independent of the public experiment
facade and performs no solver work at import time.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from eos_generation._internal.config import DEFAULT_CONFIG
from eos_generation.stellar.tov import (
    LAMBDA_FRAMEWORK_CAPABILITY,
    TIDAL_NOT_REQUESTED_STATUS,
    TovSequenceEvidence,
    refine_maximum_mass_from_sequence,
    solve_sequence,
    solve_star,
)
from eos_generation._internal.planning import (
    BSk24TOVStage,
    BSk24TrialConfig,
    _json_records,
)
from eos_generation._internal.saved_tables import (
    classify_saved_tidal_rows,
)
from eos_generation.bsk24.reconstruction import BSk24ConsistentBaseline
from eos_generation._internal.sequence_tables import (
    _sequence_frame,
)
from eos_generation.bsk24.deformation import BSk24WindowedEos


_MAXIMUM_AUTOMATIC_STELLAR_WORKERS = 4
_OUTER_NOTEBOOK_WORKER_ENV = "BSK24_NOTEBOOK_OUTER_WORKER"
_PRODUCTION_REFINE_MAXIMUM_FROM_SEQUENCE = refine_maximum_mass_from_sequence
_PRODUCTION_SOLVE_SEQUENCE = solve_sequence
_PRODUCTION_SOLVE_STAR = solve_star


def _tov_settings(eos: Any, config: BSk24TrialConfig, stage: BSk24TOVStage):
    pressure_min = float(eos.pressure_min_mev_fm3)
    return replace(
        DEFAULT_CONFIG.tov,
        sequence_points=stage.sequence_points,
        grid_pressure_min_log=config.central_pressure_min_mev_fm3,
        pressure_min_safe=pressure_min,
        surface_pressure_cutoff=pressure_min,
        dense_profile_points=stage.radial_profile_points,
    )


def _pressure_max(eos: Any) -> float:
    if hasattr(eos, "pressure_max_mev_fm3"):
        return float(eos.pressure_max_mev_fm3)
    return float(eos.pressure_max_causal_mev_fm3)


def _declared_pressure_max(eos: Any) -> float | None:
    """Return an explicit retained endpoint when the EoS exposes one."""

    for name in ("pressure_max_mev_fm3", "pressure_max_causal_mev_fm3"):
        if hasattr(eos, name):
            value = float(getattr(eos, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("retained EoS pressure endpoint is invalid")
            return value
    return None


def _fixed_mass_result(
    eos: Any,
    evidence: TovSequenceEvidence,
    target_mass: float,
    config: BSk24TrialConfig,
    stage: BSk24TOVStage,
) -> tuple[dict[str, Any], Any | None]:
    stable = np.asarray(evidence.stable_sequence, dtype=float)
    if stable.ndim != 2 or len(stable) < 2:
        return {
            "status": "unavailable_not_bracketed",
            "target_mass_msun": target_mass,
            "reason": "stable sequence has fewer than two successful configurations",
            "tidal_status": None,
            "tidal_failure_reason": None,
        }, None
    endpoint = _declared_pressure_max(eos)
    if endpoint is not None and (
        not np.all(np.isfinite(stable[:, 3]))
        or np.any(stable[:, 3] > endpoint)
    ):
        return {
            "status": "unavailable_outside_retained_eos_domain",
            "target_mass_msun": target_mass,
            "reason": (
                "stable-sequence fixed-mass evidence exceeds the retained "
                "EoS pressure endpoint"
            ),
            "tidal_status": None,
            "tidal_failure_reason": None,
        }, None
    masses = stable[:, 0]
    crossings = np.flatnonzero(
        (masses[:-1] - target_mass) * (masses[1:] - target_mass) <= 0.0
    )
    if not len(crossings):
        return {
            "status": "unavailable_not_bracketed",
            "target_mass_msun": target_mass,
            "reason": "target mass is outside the successful stable prefix",
            "tidal_status": None,
            "tidal_failure_reason": None,
        }, None
    index = int(crossings[0])
    lower, upper = float(stable[index, 3]), float(stable[index + 1, 3])
    if endpoint is not None and (lower > endpoint or upper > endpoint):
        return {
            "status": "unavailable_outside_retained_eos_domain",
            "target_mass_msun": target_mass,
            "reason": "fixed-mass bracket exceeds the retained EoS pressure endpoint",
            "tidal_status": None,
            "tidal_failure_reason": None,
        }, None
    settings = _tov_settings(eos, config, stage)
    # The successful stable-prefix sequence already contains the exact
    # bracket-endpoint masses from this EoS, stage and tolerance pair.  Reuse
    # them instead of solving both endpoint stars again.  Interior Brent
    # evaluations need only M(Pc), so they remain background-only; one final
    # solve at the root supplies the governed tidal/profile observables.
    known_masses = {
        lower: float(stable[index, 0]),
        upper: float(stable[index + 1, 0]),
    }
    background_cache: dict[float, Any] = {}

    def background_star_at(pressure: float):
        key = float(pressure)
        if key not in background_cache:
            background_cache[key] = solve_star(
                eos,
                key,
                rtol=stage.rtol,
                atol=stage.atol,
                settings=settings,
                calculate_tidal=False,
                retain_profile=False,
            )
        return background_cache[key]

    def mass_residual(pressure: float) -> float:
        key = float(pressure)
        mass = (
            known_masses[key]
            if key in known_masses
            else float(background_star_at(key).mass)
        )
        return mass - target_mass

    root = brentq(
        mass_residual,
        lower,
        upper,
        xtol=config.fixed_mass_root_xtol_mev_fm3,
        rtol=4.0 * np.finfo(float).eps,
    )
    if endpoint is not None and float(root) > endpoint:
        return {
            "status": "unavailable_outside_retained_eos_domain",
            "target_mass_msun": target_mass,
            "reason": "fixed-mass root exceeds the retained EoS pressure endpoint",
            "tidal_status": None,
            "tidal_failure_reason": None,
        }, None
    star = solve_star(
        eos,
        float(root),
        rtol=stage.rtol,
        atol=stage.atol,
        settings=settings,
        calculate_tidal=True,
        retain_profile=True,
    )
    tidal = star.lambda_diagnostic
    return {
        "status": "bracketed_and_solved",
        "target_mass_msun": target_mass,
        "mass_msun": float(star.mass),
        "mass_residual_msun": float(star.mass - target_mass),
        "radius_km": float(star.radius),
        "central_pressure_mev_fm3": float(root),
        "central_energy_density_mev_fm3": float(star.central_energy_density),
        "central_sound_speed_squared": float(star.central_sound_speed_squared),
        "k2": None if tidal.k2 is None else float(tidal.k2),
        "lambda_dimensionless": (
            None if tidal.lambda_dimensionless is None else float(tidal.lambda_dimensionless)
        ),
        "tidal_status": tidal.scientific_status,
        "tidal_failure_reason": tidal.failure_reason,
        "bracket_pressure_mev_fm3": [lower, upper],
        "root_xtol_mev_fm3": config.fixed_mass_root_xtol_mev_fm3,
        "root_evaluation_count": len(background_cache) + 1,
    }, star


_PRODUCTION_FIXED_MASS_RESULT = _fixed_mass_result

def _fixed_mass_observable_convergence(
    rows: pd.DataFrame,
    *,
    observable: str,
    requested_stages: Sequence[str],
    tidal_observable: bool,
) -> dict[str, Any]:
    """Summarize one observable only when every requested stage is valid."""

    ordered_stages = tuple(str(stage) for stage in requested_stages)
    values_by_stage: dict[str, float | None] = {}
    stage_evidence: dict[str, dict[str, Any]] = {}
    missing_or_failed: list[dict[str, Any]] = []
    for stage in ordered_stages:
        stage_rows = (
            rows.loc[rows["stage"].astype(str) == stage]
            if "stage" in rows
            else pd.DataFrame()
        )
        if len(stage_rows) != 1:
            reason = "requested_stage_row_missing" if stage_rows.empty else "duplicate_stage_rows"
            values_by_stage[stage] = None
            evidence = {
                "background_status": None,
                "tidal_status": None,
                "tidal_failure_reason": None,
                "value_status": reason,
            }
            stage_evidence[stage] = evidence
            missing_or_failed.append({"stage": stage, "reason": reason})
            continue

        row = stage_rows.iloc[0]
        background_status = row.get("status")
        tidal_status = row.get("tidal_status")
        tidal_failure_reason = row.get("tidal_failure_reason")
        tidal_classification = classify_saved_tidal_rows(
            stage_rows, schema="fixed_mass"
        ).iloc[0]
        tidal_row_valid = bool(tidal_classification["tidal_valid"])
        tidal_validity_reason = str(
            tidal_classification["tidal_validity_reason"]
        )
        if pd.isna(background_status):
            background_status = None
        if pd.isna(tidal_status):
            tidal_status = None
        if pd.isna(tidal_failure_reason):
            tidal_failure_reason = None
        raw_value = row.get(observable)
        try:
            value = None if pd.isna(raw_value) else float(raw_value)
        except (TypeError, ValueError):
            value = None
        if value is not None and not math.isfinite(value):
            value = None

        if background_status != "bracketed_and_solved":
            value_status = "background_not_bracketed_and_solved"
        elif tidal_observable and not tidal_row_valid:
            value_status = (
                "tidal_failed_closed"
                if tidal_status == "failed_closed"
                else tidal_validity_reason
            )
        elif value is None:
            value_status = "missing_or_nonfinite_observable"
        else:
            value_status = "valid"

        values_by_stage[stage] = value if value_status == "valid" else None
        evidence = {
            "background_status": background_status,
            "tidal_status": tidal_status,
            "tidal_failure_reason": tidal_failure_reason,
            "tidal_validity_reason": tidal_validity_reason,
            "value_status": value_status,
        }
        stage_evidence[stage] = evidence
        if value_status != "valid":
            missing_or_failed.append(
                {
                    "stage": stage,
                    "reason": value_status,
                    "background_status": background_status,
                    "tidal_status": tidal_status,
                    "tidal_failure_reason": tidal_failure_reason,
                    "tidal_validity_reason": tidal_validity_reason,
                }
            )

    finite_values = [value for value in values_by_stage.values() if value is not None]
    requested_count = len(ordered_stages)
    contributing_count = len(finite_values)
    complete = contributing_count == requested_count
    completeness_status = (
        "complete_all_requested_stages" if complete else "incomplete_failed_closed"
    )
    if not complete:
        envelope_status = "unavailable_incomplete_failed_closed"
        envelope = None
    elif requested_count < 2:
        envelope_status = "unavailable_single_stage"
        envelope = None
    else:
        envelope_status = "available_complete_all_requested_stages"
        envelope = float(max(finite_values) - min(finite_values))
    return {
        "observable_kind": "tidal" if tidal_observable else "background",
        "ordered_requested_stages": list(ordered_stages),
        "values_by_stage": values_by_stage,
        "requested_stage_count": requested_count,
        "contributing_stage_count": contributing_count,
        "missing_or_failed_stages": missing_or_failed,
        "stage_evidence_by_stage": stage_evidence,
        "completeness_status": completeness_status,
        "convergence_envelope_status": envelope_status,
        "measured_numerical_envelope": envelope,
        "tidal_valid_status_required": (
            LAMBDA_FRAMEWORK_CAPABILITY if tidal_observable else None
        ),
    }

def _sampled_peak_convergence(
    rows: pd.DataFrame,
    *,
    requested_stages: Sequence[str],
) -> dict[str, Any]:
    """Report sampled-peak spans only from complete multi-stage evidence."""

    ordered_stages = tuple(str(stage) for stage in requested_stages)
    values: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for stage in ordered_stages:
        stage_rows = (
            rows.loc[rows["stage"].astype(str).eq(stage)].copy()
            if "stage" in rows.columns
            else pd.DataFrame()
        )
        if "is_sampled_peak" in stage_rows.columns:
            peak_mask = stage_rows["is_sampled_peak"].map(
                lambda value: value is True
                or str(value).strip().lower() == "true"
            )
            stage_rows = stage_rows.loc[peak_mask]
        else:
            stage_rows = stage_rows.iloc[0:0]
        if "calculation_status" in stage_rows.columns:
            stage_rows = stage_rows.loc[
                stage_rows["calculation_status"].astype(str).eq("success")
            ]
        if len(stage_rows) != 1:
            missing.append(
                {
                    "stage": stage,
                    "reason": (
                        "sampled_peak_row_missing"
                        if stage_rows.empty
                        else "duplicate_sampled_peak_rows"
                    ),
                }
            )
            continue
        row = stage_rows.iloc[0]
        mass = pd.to_numeric(pd.Series([row.get("Mass")]), errors="coerce").iloc[0]
        pressure = pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        "central_pressure_mev_fm3",
                        row.get("P_Central"),
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]
        if not math.isfinite(float(mass)) or not math.isfinite(float(pressure)):
            missing.append(
                {
                    "stage": stage,
                    "reason": "sampled_peak_mass_or_pressure_nonfinite",
                }
            )
            continue
        values.append(
            {
                "stage": stage,
                "mass_msun": float(mass),
                "central_pressure_mev_fm3": float(pressure),
                "classification": "sampled_peak_not_Mmax",
            }
        )

    requested_count = len(ordered_stages)
    contributing_count = len(values)
    complete = contributing_count == requested_count
    if not complete:
        completeness = "incomplete_sampled_peak_stage_evidence"
        envelope_status = "unavailable_incomplete_stage_evidence"
        mass_envelope = None
        pressure_envelope = None
    elif requested_count < 2:
        completeness = "complete_single_stage_sampled_peak_evidence"
        envelope_status = "unavailable_single_stage"
        mass_envelope = None
        pressure_envelope = None
    else:
        completeness = "complete_all_requested_sampled_peak_stages"
        envelope_status = "available_complete_all_requested_stages"
        mass_envelope = float(
            max(item["mass_msun"] for item in values)
            - min(item["mass_msun"] for item in values)
        )
        pressure_envelope = float(
            max(item["central_pressure_mev_fm3"] for item in values)
            - min(item["central_pressure_mev_fm3"] for item in values)
        )
    return {
        "classification": "sampled_peak_not_Mmax",
        "ordered_requested_stages": list(ordered_stages),
        "requested_stage_count": requested_count,
        "contributing_stage_count": contributing_count,
        "missing_or_failed_stages": missing,
        "completeness_status": completeness,
        "convergence_envelope_status": envelope_status,
        "values_by_stage": values,
        "mass_envelope_msun": mass_envelope,
        "central_pressure_envelope_mev_fm3": pressure_envelope,
    }

def _stellar_convergence_from_saved_tables(
    sequences: pd.DataFrame,
    fixed: pd.DataFrame,
    config: BSk24TrialConfig,
    *,
    case_ids: Sequence[str],
) -> dict[str, Any]:
    """Rebuild convergence reporting from saved stellar rows without solving."""

    requested_stages = tuple(stage.name for stage in config.tov_stages)
    convergence: dict[str, Any] = {
        "schema_id": "eos_generation_stellar_convergence_v1",
        "maximum_mass_policy": (
            "sampled peaks only; not M_max unless a separate governed turning-point "
            "refinement satisfies the repository policy"
        ),
        "ordered_requested_stages": list(requested_stages),
        "requested_stage_count": len(requested_stages),
        "tidal_valid_status_required": LAMBDA_FRAMEWORK_CAPABILITY,
        "cases": {},
    }
    incomplete_background = False
    incomplete_tidal = False
    incomplete_sampled_peak = False
    for case_id in case_ids:
        case_report: dict[str, Any] = {"fixed_masses": {}, "sampled_peak": {}}
        for target_mass in config.fixed_masses_msun:
            rows = fixed.loc[
                fixed["case_id"].astype(str).eq(str(case_id))
                & np.isclose(
                    pd.to_numeric(fixed["target_mass_msun"], errors="coerce"),
                    target_mass,
                )
            ]
            case_report["fixed_masses"][str(target_mass)] = {}
            for observable, is_tidal in (
                ("radius_km", False),
                ("central_energy_density_mev_fm3", False),
                ("k2", True),
                ("lambda_dimensionless", True),
            ):
                report = _fixed_mass_observable_convergence(
                    rows,
                    observable=observable,
                    requested_stages=requested_stages,
                    tidal_observable=is_tidal,
                )
                case_report["fixed_masses"][str(target_mass)][observable] = report
                if report["completeness_status"] != "complete_all_requested_stages":
                    if is_tidal:
                        incomplete_tidal = True
                    else:
                        incomplete_background = True
        peak = _sampled_peak_convergence(
            sequences.loc[
                sequences["case_id"].astype(str).eq(str(case_id))
            ],
            requested_stages=requested_stages,
        )
        case_report["sampled_peak"] = peak
        if peak["completeness_status"] == "incomplete_sampled_peak_stage_evidence":
            incomplete_sampled_peak = True
        convergence["cases"][str(case_id)] = case_report

    if len(config.tov_stages) < 2:
        convergence["status"] = "single_stage_no_numerical_envelope"
    elif incomplete_background:
        convergence["status"] = "incomplete_background_stages"
    elif incomplete_tidal:
        convergence["status"] = "partial_tidal_incomplete_failed_closed"
    elif incomplete_sampled_peak:
        convergence["status"] = "partial_sampled_peak_stage_evidence"
    else:
        convergence["status"] = "complete_all_requested_stages"
    return convergence


def _automatic_stellar_worker_count(case_count: int) -> int:
    if case_count < 1:
        return 1
    if os.environ.get(_OUTER_NOTEBOOK_WORKER_ENV) == "1":
        return 1
    logical = max(1, int(os.cpu_count() or 1))
    return min(
        case_count,
        _MAXIMUM_AUTOMATIC_STELLAR_WORKERS,
        max(1, logical // 2),
    )


def _case_worker_is_safe(config: BSk24TrialConfig, eos: Any) -> bool:
    if (
        solve_sequence is not _PRODUCTION_SOLVE_SEQUENCE
        or solve_star is not _PRODUCTION_SOLVE_STAR
        or refine_maximum_mass_from_sequence
        is not _PRODUCTION_REFINE_MAXIMUM_FROM_SEQUENCE
        or _fixed_mass_result is not _PRODUCTION_FIXED_MASS_RESULT
    ):
        return False
    try:
        pickle.dumps((config, "pickling_probe", eos))
    except Exception:
        return False
    return True


def _run_case_job(
    config: BSk24TrialConfig,
    case_id: str,
    eos: Any,
) -> dict[str, Any]:
    """Run full sampled tidal sequences for one ordinary experiment case."""

    if multiprocessing.current_process().name != "MainProcess":
        os.environ[_OUTER_NOTEBOOK_WORKER_ENV] = "1"
    started = time.perf_counter()
    stages: dict[str, dict[str, Any]] = {}
    for stage in config.tov_stages:
        settings = _tov_settings(eos, config, stage)
        retained_endpoint = _pressure_max(eos)
        evidence = solve_sequence(
            eos,
            p_max_causal=retained_endpoint,
            rtol=stage.rtol,
            atol=stage.atol,
            settings=settings,
            return_tidal_diagnostics=True,
            return_sequence_evidence=True,
        )
        if not isinstance(evidence, TovSequenceEvidence):
            raise TypeError(
                "shared solve_sequence did not return TovSequenceEvidence"
            )
        attempted_pressures = np.asarray(
            evidence.attempted_central_pressures, dtype=float
        )
        successful_pressures = np.asarray(
            evidence.successful_central_pressures, dtype=float
        )
        if (
            np.any(~np.isfinite(attempted_pressures))
            or np.any(~np.isfinite(successful_pressures))
            or np.any(attempted_pressures > retained_endpoint)
            or np.any(successful_pressures > retained_endpoint)
        ):
            raise ValueError(
                "stellar sequence contains a central pressure outside the "
                "retained EoS domain"
            )
        maximum = refine_maximum_mass_from_sequence(
            eos,
            evidence,
            maximum_mass_threshold_msun=config.maximum_mass_threshold_msun,
            local_points=config.maximum_mass_initial_points,
            rtol=stage.rtol,
            atol=stage.atol,
            settings=settings,
        )
        if (
            maximum.central_pressure_mev_fm3 is not None
            and float(maximum.central_pressure_mev_fm3) > retained_endpoint
        ):
            raise ValueError(
                "maximum-mass refinement exceeded the retained EoS endpoint"
            )
        maximum_row = {
            "case_id": case_id,
            "stage": stage.name,
            "status": maximum.status,
            "maximum_mass_resolved": maximum.maximum_mass_resolved,
            "maximum_mass_availability_status": (
                "resolved_bracketed_and_refined"
                if maximum.maximum_mass_resolved
                else f"unavailable_{maximum.status}"
            ),
            "maximum_mass_msun": maximum.maximum_mass_msun,
            "maximum_mass_threshold_msun": (
                maximum.maximum_mass_threshold_msun
            ),
            "passes_maximum_mass_threshold": (
                maximum.passes_maximum_mass_threshold
            ),
            "central_pressure_mev_fm3": (
                maximum.central_pressure_mev_fm3
            ),
            "central_energy_density_mev_fm3": (
                maximum.central_energy_density_mev_fm3
            ),
            "central_sound_speed_squared": (
                maximum.central_sound_speed_squared
            ),
            "radius_km": maximum.radius_km,
            "turning_point_count": len(maximum.turning_point_brackets),
            "positive_left_secant": maximum.positive_left_secant,
            "negative_right_secant": maximum.negative_right_secant,
            "eos_endpoint_pressure_mev_fm3": (
                maximum.eos_endpoint_pressure_mev_fm3
            ),
            "endpoint_limitation": maximum.endpoint_limitation,
            "refinement_status": maximum.refinement_status,
            "sampled_sequence_model_count": len(evidence.full_sequence),
            "local_background_solver_call_count": maximum.solver_call_count,
            "tidal_solver_calls_for_maximum_mass": 0,
        }
        frame = _sequence_frame(case_id, stage.name, evidence)
        if case_id == "direct":
            amplitude = None
            delta = None
        else:
            amplitude = eos.deformation.amplitude
            delta = eos.deformation.delta_mev_fm3
        maximum_row["amplitude"] = amplitude
        maximum_row["delta_mev_fm3"] = delta
        # Keep nullable numerical sequence coordinates explicitly float-typed.
        # Assigning ``None`` to the direct frame makes these columns object
        # dtype and triggers pandas' deprecated all-NA concat inference when
        # numerical deformation frames are appended.
        frame["amplitude"] = np.nan if amplitude is None else amplitude
        frame["delta_mev_fm3"] = np.nan if delta is None else delta
        frame["tov_rtol"] = stage.rtol
        frame["tov_atol"] = stage.atol
        frame["sequence_points_requested"] = stage.sequence_points
        fixed_rows: list[dict[str, Any]] = []
        stars: dict[tuple[str, str, float], Any] = {}
        for target_mass in config.fixed_masses_msun:
            result, star = _fixed_mass_result(
                eos, evidence, target_mass, config, stage
            )
            fixed_rows.append(
                {
                    "case_id": case_id,
                    "stage": stage.name,
                    "amplitude": amplitude,
                    "delta_mev_fm3": delta,
                    **result,
                }
            )
            if star is not None:
                stars[(case_id, stage.name, target_mass)] = star
        stages[stage.name] = {
            "sequence_frame": frame,
            "fixed_rows": fixed_rows,
            "stars": stars,
            "maximum_row": maximum_row,
            "maximum_report": maximum.to_dict(),
        }
    return {
        "case_id": case_id,
        "stages": stages,
        "worker_pid": os.getpid(),
        "worker_wall_seconds": time.perf_counter() - started,
    }

def _run_stellar(
    *,
    config: BSk24TrialConfig,
    baseline: BSk24ConsistentBaseline,
    generated: Mapping[str, BSk24WindowedEos],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[tuple[str, str, float], Any]]:
    eos_map: dict[str, Any] = {"direct": baseline.eos, **generated}
    sequence_frames: list[pd.DataFrame] = []
    fixed_rows: list[dict[str, Any]] = []
    stars: dict[tuple[str, str, float], Any] = {}
    maximum_rows: list[dict[str, Any]] = []
    maximum_reports: dict[str, Any] = {}
    selected_workers = _automatic_stellar_worker_count(len(eos_map))
    production_payloads = all(
        _case_worker_is_safe(config, eos) for eos in eos_map.values()
    )
    use_processes = bool(selected_workers > 1 and production_payloads)
    case_results: dict[str, dict[str, Any]] = {}
    if use_processes:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=selected_workers,
            mp_context=context,
        ) as pool:
            futures = {
                pool.submit(_run_case_job, config, case_id, eos): case_id
                for case_id, eos in eos_map.items()
            }
            try:
                for future in as_completed(futures):
                    case_id = futures[future]
                    result = dict(future.result())
                    if result.get("case_id") != case_id:
                        raise RuntimeError(
                            "stellar worker returned a mismatched case ID"
                        )
                    case_results[case_id] = result
            except Exception:
                for future in futures:
                    future.cancel()
                raise
    else:
        selected_workers = 1
        for case_id, eos in eos_map.items():
            case_results[case_id] = _run_case_job(
                config, case_id, eos
            )

    for stage in config.tov_stages:
        for case_id in eos_map:
            stage_result = case_results[case_id]["stages"][stage.name]
            sequence_frames.append(stage_result["sequence_frame"])
            fixed_rows.extend(stage_result["fixed_rows"])
            stars.update(stage_result["stars"])
            maximum_rows.append(stage_result["maximum_row"])
            maximum_reports[f"{case_id}:{stage.name}"] = stage_result[
                "maximum_report"
            ]
    sequences = (
        pd.concat(sequence_frames, ignore_index=True)
        if sequence_frames
        else pd.DataFrame()
    )
    fixed = pd.DataFrame(fixed_rows)
    convergence = _stellar_convergence_from_saved_tables(
        sequences,
        fixed,
        config,
        case_ids=tuple(eos_map),
    )
    resolved_count = sum(
        bool(row["maximum_mass_resolved"]) for row in maximum_rows
    )
    convergence.update(
        {
            "maximum_mass_policy": (
                "M_max requires one sampled positive-to-negative dM/dP_c "
                "bracket followed by local background-only refinement"
            ),
            "maximum_mass_case_stage_count": len(maximum_rows),
            "resolved_maximum_mass_case_stage_count": resolved_count,
            "maximum_mass_rows": maximum_rows,
            "maximum_mass_reports": maximum_reports,
            "background_solver_call_count": sum(
                int(row["local_background_solver_call_count"])
                for row in maximum_rows
            ),
        }
    )
    convergence["parallel_execution"] = {
        "mode": "spawned_case_processes" if use_processes else "serial",
        "policy": "automatic_bounded_spawned_processes_v1",
        "maximum_workers": _MAXIMUM_AUTOMATIC_STELLAR_WORKERS,
        "selected_worker_count": selected_workers,
        "case_job_count": len(case_results),
        "worker_process_ids": sorted(
            {
                int(result["worker_pid"])
                for result in case_results.values()
            }
        ),
        "case_worker_wall_seconds": {
            case_id: float(case_results[case_id]["worker_wall_seconds"])
            for case_id in eos_map
        },
        "deterministic_parent_merge_order": "stage_major_case_major",
        "nested_process_pool_disabled": bool(
            os.environ.get(_OUTER_NOTEBOOK_WORKER_ENV) == "1"
        ),
    }
    return sequences, fixed, convergence, stars

def _stellar_status_summary(
    sequences: pd.DataFrame,
    fixed: pd.DataFrame,
    config: BSk24TrialConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize background and exact-status tidal outcomes without new solves."""

    def reason_counts(frame: pd.DataFrame, column: str, mask: pd.Series) -> str:
        if column not in frame.columns:
            return "{}"
        values = frame.loc[mask, column].dropna().astype(str)
        values = values.loc[values.str.len() > 0]
        counts = {key: int(value) for key, value in values.value_counts().sort_index().items()}
        return json.dumps(counts, sort_keys=True)

    rows: list[dict[str, Any]] = []
    fixed_failures: list[dict[str, Any]] = []
    for scope, frame, background_column, background_success in (
        ("sequence", sequences, "calculation_status", "success"),
        ("fixed_mass", fixed, "status", "bracketed_and_solved"),
    ):
        for stage in (item.name for item in config.tov_stages):
            stage_rows = frame.loc[frame["stage"].astype(str) == stage].copy()
            requested = int(len(stage_rows))
            background_ok = (
                stage_rows[background_column].astype(str).eq(background_success)
                if requested and background_column in stage_rows
                else pd.Series(False, index=stage_rows.index, dtype=bool)
            )
            tidal_status = (
                stage_rows["tidal_status"].astype(str)
                if requested and "tidal_status" in stage_rows
                else pd.Series("", index=stage_rows.index, dtype=str)
            )
            tidal_classification = classify_saved_tidal_rows(
                stage_rows,
                schema="sequence" if scope == "sequence" else "fixed_mass",
            )
            tidal_valid = tidal_classification["tidal_valid"]
            tidal_failed = background_ok & tidal_status.eq("failed_closed")
            background_success_count = int(background_ok.sum())
            tidal_validated_count = int(tidal_valid.sum())
            tidal_failed_count = int(tidal_failed.sum())
            tidal_unavailable_count = int(
                requested - tidal_validated_count - tidal_failed_count
            )
            if requested == 0:
                background_completeness = "unavailable_no_requested_rows"
                tidal_completeness = "unavailable_no_requested_rows"
            else:
                background_completeness = (
                    "complete_all_requested_backgrounds"
                    if background_success_count == requested
                    else "partial_background_failures"
                )
                if tidal_validated_count == requested:
                    tidal_completeness = "complete_all_requested_tides_validated"
                elif tidal_validated_count:
                    tidal_completeness = "partial_tidal_validation"
                else:
                    tidal_completeness = "unavailable_no_validated_tides"
            row = {
                "scope": scope,
                "stage": stage,
                "requested_count": requested,
                "background_success_count": background_success_count,
                "background_failure_count": int(requested - background_success_count),
                "tidal_validated_count": tidal_validated_count,
                "tidal_failed_closed_count": tidal_failed_count,
                "tidal_unavailable_count": tidal_unavailable_count,
                "background_completeness_status": background_completeness,
                "tidal_completeness_status": tidal_completeness,
                "tidal_valid_status_required": LAMBDA_FRAMEWORK_CAPABILITY,
                "background_failure_reasons_json": reason_counts(
                    stage_rows, "failure_reason", ~background_ok
                ),
                "tidal_failure_reasons_json": reason_counts(
                    stage_rows,
                    "tidal_failure_reason",
                    background_ok & ~tidal_valid,
                ),
                "tidal_invalidity_reasons_json": json.dumps(
                    {
                        str(key): int(value)
                        for key, value in tidal_classification.loc[
                            background_ok & ~tidal_valid,
                            "tidal_validity_reason",
                        ]
                        .value_counts()
                        .sort_index()
                        .items()
                    },
                    sort_keys=True,
                ),
            }
            rows.append(row)
            if scope == "fixed_mass" and requested:
                failure_rows = stage_rows.loc[background_ok & ~tidal_valid]
                for failure_index, failure in failure_rows.iterrows():
                    fixed_failures.append(
                        {
                            "case_id": str(failure["case_id"]),
                            "stage": stage,
                            "target_mass_msun": float(
                                failure["target_mass_msun"]
                            ),
                            "status": str(failure["status"]),
                            "tidal_status": (
                                None
                                if pd.isna(failure.get("tidal_status"))
                                else str(failure["tidal_status"])
                            ),
                            "tidal_failure_reason": (
                                None
                                if pd.isna(failure.get("tidal_failure_reason"))
                                else str(failure["tidal_failure_reason"])
                            ),
                            "tidal_validity_reason": str(
                                tidal_classification.loc[
                                    failure_index,
                                    "tidal_validity_reason",
                                ]
                            ),
                        }
                    )

    summary = pd.DataFrame(rows)
    count_columns = (
        "requested_count",
        "background_success_count",
        "background_failure_count",
        "tidal_validated_count",
        "tidal_failed_closed_count",
        "tidal_unavailable_count",
    )
    totals_by_scope: dict[str, dict[str, int]] = {}
    for scope, scope_rows in summary.groupby("scope", sort=False):
        totals_by_scope[str(scope)] = {
            column: int(scope_rows[column].sum()) for column in count_columns
        }
    if summary.empty:
        publication_status = "unavailable_no_stellar_rows"
    elif int(summary["background_failure_count"].sum()) > 0:
        publication_status = "partial_background_failures"
    elif (
        int(summary["tidal_failed_closed_count"].sum()) > 0
        or int(summary["tidal_unavailable_count"].sum()) > 0
    ):
        publication_status = "partial_tidal_validation"
    else:
        publication_status = "complete_background_and_tidal"
    payload = {
        "schema_id": "eos_generation_stellar_status_summary_v1",
        "tidal_valid_status_required": LAMBDA_FRAMEWORK_CAPABILITY,
        "rows": _json_records(summary),
        "totals_by_scope": totals_by_scope,
        "fixed_mass_tidal_failures": fixed_failures,
        "publication_interpretation_status": publication_status,
    }
    return summary, payload

__all__ = [
    "_fixed_mass_observable_convergence",
    "_fixed_mass_result",
    "_pressure_max",
    "_run_stellar",
    "_sampled_peak_convergence",
    "_stellar_convergence_from_saved_tables",
    "_stellar_status_summary",
    "_tov_settings",
]

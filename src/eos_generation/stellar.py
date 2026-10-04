"""Stellar for controlled BSk24 / BSk25 experiments."""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import time
from .numerics import BSk24TOVStage, BSk24TrialConfig, DEFAULT_CONFIG, TovConfig
from .thermodynamics import BSk24ConsistentBaseline, BSk24WindowedEos
from .tov import (
    BARE_SELF_BOUND_SEQUENCE_POLICY,
    LAMBDA_FRAMEWORK_CAPABILITY,
    SEED_PRESERVING_LOCAL_REFINEMENT_POLICY,
    TOV_SEQUENCE_FIELDS,
    TOV_TIDAL_DIAGNOSTIC_FIELDS,
    TovConvergenceError,
    TovFailureDetail,
    TovLambdaDiagnostic,
    TovMassSecantEvidence,
    TovMaximumMassResult,
    TovSequenceEvidence,
    TovStarResult,
    _ABSOLUTE_P_MAX_FALLBACK,
    _A_CONV,
    _BUCHDAHL_LIMIT,
    _DEFAULT_TOV,
    _MAXIMUM_AUTOMATIC_SEQUENCE_WORKERS,
    _MIN_MASS_CUTOFF,
    _MIN_RADIUS_CUTOFF,
    _OUTER_PARALLEL_WORKER_ENV,
    _freeze_profiles,
    _freeze_rows,
    _require_finite,
    _resolved_discontinuities,
    solve_star,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from scipy.optimize import brentq, minimize_scalar
from types import SimpleNamespace
from typing import Any, Callable, Literal, Mapping, Sequence


logger = logging.getLogger("eos_generation.stellar.tov")


_PRODUCTION_SOLVE_STAR = solve_star


_SEQUENCE_WORKER_STATE: (
    tuple[
        Callable,
        float | None,
        float | None,
        TovConfig,
        bool,
        bool,
    ]
    | None
) = None


def _initialize_sequence_worker(
    eos_callable: Callable,
    rtol: float | None,
    atol: float | None,
    settings: TovConfig,
    calculate_tidal: bool,
    retain_profiles: bool,
) -> None:
    global _SEQUENCE_WORKER_STATE
    os.environ[_OUTER_PARALLEL_WORKER_ENV] = "1"
    _SEQUENCE_WORKER_STATE = (
        eos_callable,
        rtol,
        atol,
        settings,
        calculate_tidal,
        retain_profiles,
    )


def _solve_sequence_pressure_worker(
    index: int,
    central_pressure: float,
) -> tuple[int, TovStarResult | None, str | None]:
    if _SEQUENCE_WORKER_STATE is None:
        raise RuntimeError("TOV sequence worker was not initialized")
    eos_callable, rtol, atol, settings, calculate_tidal, retain_profiles = (
        _SEQUENCE_WORKER_STATE
    )
    try:
        kwargs = {
            "rtol": rtol,
            "atol": atol,
            "settings": settings,
            "calculate_tidal": calculate_tidal,
        }
        if not retain_profiles:
            kwargs["retain_profile"] = False
        star = solve_star(eos_callable, central_pressure, **kwargs)
        return index, star, None
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        return index, None, str(exc)


def _automatic_sequence_worker_count(pressure_count: int) -> int:
    if pressure_count < 17:
        return 1
    if os.environ.get(_OUTER_PARALLEL_WORKER_ENV) == "1":
        return 1
    logical = max(1, int(os.cpu_count() or 1))
    return min(
        pressure_count,
        _MAXIMUM_AUTOMATIC_SEQUENCE_WORKERS,
        max(1, logical // 2),
    )


def _sequence_process_worker_is_safe(
    eos_callable: Callable,
    settings: TovConfig,
    rtol: float | None,
    atol: float | None,
    calculate_tidal: bool,
    retain_profiles: bool = True,
) -> bool:
    if solve_star is not _PRODUCTION_SOLVE_STAR:
        return False
    module_name = str(
        getattr(
            eos_callable,
            "__module__",
            type(eos_callable).__module__,
        )
    )
    if module_name not in {
        "eos_generation.baseline",
        "eos_generation.thermodynamics",
    } and not bool(getattr(eos_callable, "allow_parallel_tov_sequence", False)):
        return False
    try:
        pickle.dumps(
            (eos_callable, settings, rtol, atol, calculate_tidal, retain_profiles)
        )
    except Exception:
        return False
    return True


def _tidal_diagnostic_rows(
    diagnostics: list[dict[str, float]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            float("nan") if row[field] is None else float(row[field])
            for field in TOV_TIDAL_DIAGNOSTIC_FIELDS
        )
        for row in diagnostics
    )


def _sampled_mass_secants(
    full_sequence: tuple[tuple[float, ...], ...],
) -> tuple[TovMassSecantEvidence, ...]:
    secants = []
    for lower_index in range(len(full_sequence) - 1):
        upper_index = lower_index + 1
        lower = full_sequence[lower_index]
        upper = full_sequence[upper_index]
        delta_pressure = upper[3] - lower[3]
        delta_mass = upper[0] - lower[0]
        slope = delta_mass / delta_pressure
        sign = (
            "positive"
            if delta_mass > 0.0
            else "negative" if delta_mass < 0.0 else "zero"
        )
        secants.append(
            TovMassSecantEvidence(
                lower_index=lower_index,
                upper_index=upper_index,
                lower_central_pressure=lower[3],
                upper_central_pressure=upper[3],
                lower_mass=lower[0],
                upper_mass=upper[0],
                delta_mass=delta_mass,
                slope=slope,
                sign=sign,
            )
        )
    return tuple(secants)


def _successful_pressure_ordering(successful_pressures: tuple[float, ...]) -> str:
    if not successful_pressures:
        return "unavailable"
    if len(successful_pressures) == 1:
        return "single_sample"
    if np.all(np.diff(successful_pressures) > 0.0):
        return "strictly_increasing"
    return "invalid"


def _build_sequence_evidence(
    *,
    full_sequence: tuple[tuple[float, ...], ...],
    stable_sequence: Any,
    full_dense_profiles: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...],
    stable_dense_profiles: Any,
    full_tidal_diagnostics: tuple[tuple[float, ...], ...] | None,
    stable_tidal_diagnostics: Any,
    full_lambda_diagnostics: tuple[TovLambdaDiagnostic, ...] | None,
    stable_lambda_diagnostics: Any,
    attempted_central_pressures: list[float],
    failed_central_pressures: list[TovFailureDetail],
    sampled_peak_index: int | None,
    sampled_secants: tuple[TovMassSecantEvidence, ...],
    eos_endpoint_pressure: float | None,
    max_mass_stable: float,
) -> TovSequenceEvidence:
    successful_pressures = tuple(row[3] for row in full_sequence)
    if sampled_peak_index is None:
        sampled_peak_row = None
        domain_end_row = None
        peak_is_interior = False
        pre_peak_slopes = ()
        post_peak_slopes = ()
    else:
        sampled_peak_row = full_sequence[sampled_peak_index]
        domain_end_row = full_sequence[-1]
        peak_is_interior = 0 < sampled_peak_index < len(full_sequence) - 1
        pre_peak_slopes = sampled_secants[:sampled_peak_index]
        post_peak_slopes = sampled_secants[sampled_peak_index:]

    if eos_endpoint_pressure is None or domain_end_row is None:
        endpoint_margin = None
        endpoint_contact = None
    else:
        endpoint_margin = float(eos_endpoint_pressure - domain_end_row[3])
        endpoint_contact = bool(endpoint_margin == 0.0)

    return TovSequenceEvidence(
        full_sequence=full_sequence,
        stable_sequence=stable_sequence,
        full_dense_profiles=full_dense_profiles,
        stable_dense_profiles=stable_dense_profiles,
        full_tidal_diagnostics=full_tidal_diagnostics,
        stable_tidal_diagnostics=stable_tidal_diagnostics,
        full_lambda_diagnostics=full_lambda_diagnostics,
        stable_lambda_diagnostics=stable_lambda_diagnostics,
        attempted_central_pressures=tuple(attempted_central_pressures),
        successful_central_pressures=successful_pressures,
        central_pressure_ordering=_successful_pressure_ordering(successful_pressures),
        failed_central_pressures=tuple(failed_central_pressures),
        sampled_peak_index=sampled_peak_index,
        sampled_peak_row=sampled_peak_row,
        domain_end_row=domain_end_row,
        sampled_peak_is_interior=peak_is_interior,
        pre_peak_slopes=pre_peak_slopes,
        post_peak_slopes=post_peak_slopes,
        eos_endpoint_pressure=eos_endpoint_pressure,
        eos_endpoint_margin=endpoint_margin,
        final_available_model_contacts_eos_endpoint=endpoint_contact,
        max_mass_stable=max_mass_stable,
    )


def solve_sequence(
    eos_callable: Callable,
    p_max_causal: float = None,
    rtol: float = None,
    atol: float = None,
    return_tidal_diagnostics: bool = False,
    *,
    settings: TovConfig | None = None,
    return_sequence_evidence: bool = False,
    calculate_tidal: bool = True,
    retain_profiles: bool = True,
) -> tuple | TovSequenceEvidence:
    """Integrate a sequence, preserving the historical tuple unless evidence is requested."""
    resolved = _DEFAULT_TOV if settings is None else settings
    if not isinstance(calculate_tidal, bool):
        raise ValueError("calculate_tidal must be boolean")
    if not isinstance(retain_profiles, bool):
        raise ValueError("retain_profiles must be boolean")
    p_max = p_max_causal if p_max_causal is not None else _ABSOLUTE_P_MAX_FALLBACK
    try:
        p_max = float(p_max)
    except (TypeError, ValueError) as exc:
        raise ValueError("p_max_causal must be a finite positive pressure") from exc
    if not np.isfinite(p_max) or p_max <= 0.0:
        raise ValueError("p_max_causal must be a finite positive pressure")
    pressure_floor = float(resolved.grid_pressure_min_log)
    if not np.isfinite(pressure_floor) or pressure_floor <= 0.0:
        raise ValueError("sequence central-pressure floor must be finite and positive")
    if p_max_causal is not None and p_max <= pressure_floor:
        reason = (
            "retained EoS endpoint does not exceed the configured "
            "central-pressure floor"
        )
        if return_sequence_evidence:
            return _build_sequence_evidence(
                full_sequence=(),
                stable_sequence=(),
                full_dense_profiles=(),
                stable_dense_profiles=(),
                full_tidal_diagnostics=() if return_tidal_diagnostics else None,
                stable_tidal_diagnostics=() if return_tidal_diagnostics else None,
                full_lambda_diagnostics=() if return_tidal_diagnostics else None,
                stable_lambda_diagnostics=() if return_tidal_diagnostics else None,
                attempted_central_pressures=[p_max],
                failed_central_pressures=[
                    TovFailureDetail(
                        central_pressure=p_max,
                        category="eos_endpoint_below_sequence_floor",
                        reason=reason,
                        solver_status=None,
                    )
                ],
                sampled_peak_index=None,
                sampled_secants=(),
                eos_endpoint_pressure=p_max,
                max_mass_stable=0.0,
            )
        if return_tidal_diagnostics:
            return [], [], 0.0, []
        return [], [], 0.0
    pressures = np.geomspace(
        pressure_floor,
        p_max if p_max_causal is not None else 1000.0,
        resolved.sequence_points,
    )
    if p_max_causal is not None:
        pressures[-1] = p_max
        if np.any(pressures > p_max) or not np.all(np.diff(pressures) > 0.0):
            raise ValueError(
                "central-pressure sequence is not strictly increasing within "
                "the retained EoS endpoint"
            )

    curve_data = []
    dense_profiles = []
    tidal_diagnostics = []
    lambda_diagnostic_objects = []
    attempted_pressures = []
    failure_details = []
    eps_surf = getattr(eos_callable, "eps_surf", 0.0)
    sequence_policy = getattr(eos_callable, "stellar_sequence_policy", None)
    if sequence_policy not in (None, BARE_SELF_BOUND_SEQUENCE_POLICY):
        raise ValueError(f"unsupported stellar sequence policy: {sequence_policy!r}")
    bare_self_bound = sequence_policy == BARE_SELF_BOUND_SEQUENCE_POLICY
    if bare_self_bound:
        joins = _resolved_discontinuities(eos_callable)
        if len(joins) != 1 or joins[0].kind != "surface" or float(eps_surf) <= 0.0:
            raise ValueError(
                "bare self-bound sequence policy requires one finite-density vacuum surface"
            )

    def record_failure(
        central_pressure: float,
        category: str,
        reason: str,
        *,
        solver_status: int | None = None,
    ) -> None:
        if return_sequence_evidence:
            failure_details.append(
                TovFailureDetail(
                    central_pressure=central_pressure,
                    category=category,
                    reason=reason,
                    solver_status=solver_status,
                )
            )

    parallel_outcomes: dict[int, tuple[TovStarResult | None, str | None]] | None = None
    sequence_workers = _automatic_sequence_worker_count(len(pressures))
    if sequence_workers > 1 and _sequence_process_worker_is_safe(
        eos_callable,
        resolved,
        rtol,
        atol,
        calculate_tidal,
        retain_profiles,
    ):
        parallel_outcomes = {}
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=sequence_workers,
            mp_context=context,
            initializer=_initialize_sequence_worker,
            initargs=(
                eos_callable,
                rtol,
                atol,
                resolved,
                calculate_tidal,
                retain_profiles,
            ),
        ) as pool:
            futures = {
                pool.submit(
                    _solve_sequence_pressure_worker,
                    index,
                    float(pc),
                ): index
                for index, pc in enumerate(pressures)
            }
            try:
                for future in as_completed(futures):
                    expected_index = futures[future]
                    index, star, failure = future.result()
                    if index != expected_index:
                        raise RuntimeError(
                            "TOV sequence worker returned a mismatched index"
                        )
                    parallel_outcomes[index] = (star, failure)
            except Exception:
                for future in futures:
                    future.cancel()
                raise

    for attempted_index, pc in enumerate(pressures):
        if return_sequence_evidence:
            attempted_pressures.append(float(pc))
        try:
            if parallel_outcomes is None:
                kwargs = {
                    "rtol": rtol,
                    "atol": atol,
                    "settings": resolved,
                    "calculate_tidal": calculate_tidal,
                }
                if not retain_profiles:
                    kwargs["retain_profile"] = False
                star = solve_star(eos_callable, float(pc), **kwargs)
            else:
                star, failure = parallel_outcomes[attempted_index]
                if failure is not None or star is None:
                    record_failure(pc, "domain_error", failure or "unknown failure")
                    logger.error(
                        "ODE Solver failed in spawned sequence worker at Pc=%r: %s",
                        float(pc),
                        failure,
                    )
                    continue
            if bare_self_bound and (
                not np.isfinite(star.radius)
                or not np.isfinite(star.mass)
                or star.radius <= 0.0
                or star.mass <= 0.0
            ):
                record_failure(
                    pc,
                    "invalid_self_bound_mass_or_radius",
                    f"surface mass={star.mass!r}, radius={star.radius!r}",
                )
                continue
            # Bare self-bound matter has a physical low-mass M~R^3 branch;
            # legacy hadronic display cutoffs are not a validity condition.
            # The explicit policy changes no legacy BSk24 behavior.
            if not bare_self_bound and (
                star.radius < _MIN_RADIUS_CUTOFF or star.mass < _MIN_MASS_CUTOFF
            ):
                record_failure(
                    pc,
                    "minimum_mass_or_radius_cutoff",
                    f"surface mass={star.mass!r}, radius={star.radius!r}",
                )
                continue
            compactness = star.mass * _A_CONV / star.radius
            if compactness >= _BUCHDAHL_LIMIT:
                record_failure(
                    pc,
                    "buchdahl_compactness_cutoff",
                    f"compactness={compactness!r}",
                )
                continue
            curve_data.append(star.curve_row)
            dense_profiles.append(
                (
                    np.asarray(star.radius_profile, dtype=float),
                    np.asarray(star.mass_profile, dtype=float),
                )
                if retain_profiles
                else ((), ())
            )
            if return_tidal_diagnostics:
                diagnostic = star.lambda_diagnostic.to_dict()
                diagnostic.update(
                    {
                        "Mass": star.mass,
                        "Radius": star.radius,
                        "P_Central": star.central_pressure,
                        "Eps_Central": star.central_energy_density,
                        "CS2_Central": star.central_sound_speed_squared,
                        "Compactness": compactness,
                        "eps_surf": star.surface_energy_density,
                    }
                )
                tidal_diagnostics.append(diagnostic)
                lambda_diagnostic_objects.append(star.lambda_diagnostic)
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            record_failure(pc, "domain_error", str(exc))
            try:
                raise TovConvergenceError(pc=pc, reason=str(exc)) from exc
            except TovConvergenceError:
                logger.exception("ODE Solver failed due to domain error")
                continue

    if not curve_data:
        if return_sequence_evidence:
            return _build_sequence_evidence(
                full_sequence=(),
                stable_sequence=(),
                full_dense_profiles=(),
                stable_dense_profiles=(),
                full_tidal_diagnostics=() if return_tidal_diagnostics else None,
                stable_tidal_diagnostics=() if return_tidal_diagnostics else None,
                full_lambda_diagnostics=() if return_tidal_diagnostics else None,
                stable_lambda_diagnostics=() if return_tidal_diagnostics else None,
                attempted_central_pressures=attempted_pressures,
                failed_central_pressures=failure_details,
                sampled_peak_index=None,
                sampled_secants=(),
                eos_endpoint_pressure=(
                    float(p_max_causal) if p_max_causal is not None else None
                ),
                max_mass_stable=0.0,
            )
        if return_tidal_diagnostics:
            return [], [], 0.0, []
        return [], [], 0.0

    curve_array = np.array(curve_data)
    mass_array = curve_array[:, 0]
    radius_array = curve_array[:, 1]
    lambda_array = curve_array[:, 2]
    pressure_array = curve_array[:, 3]
    density_array = curve_array[:, 4]
    cs2_array = curve_array[:, 5]
    eps_surf_array = curve_array[:, 6]
    max_mass_index = int(np.argmax(mass_array))

    if return_sequence_evidence:
        full_sequence_evidence = _freeze_rows(
            curve_data,
            fields=TOV_SEQUENCE_FIELDS,
            name="full_sequence",
            allow_nan_fields=("Lambda",),
        )
        full_dense_profiles_evidence = _freeze_profiles(
            dense_profiles,
            name="full_dense_profiles",
        )
        full_tidal_evidence = (
            _tidal_diagnostic_rows(tidal_diagnostics)
            if return_tidal_diagnostics
            else None
        )
        sampled_secants = _sampled_mass_secants(full_sequence_evidence)

    mass_stable = mass_array[: max_mass_index + 1]
    radius_stable = radius_array[: max_mass_index + 1]
    lambda_stable = lambda_array[: max_mass_index + 1]
    pressure_stable = pressure_array[: max_mass_index + 1]
    density_stable = density_array[: max_mass_index + 1]
    cs2_stable = cs2_array[: max_mass_index + 1]
    eps_surf_stable = eps_surf_array[: max_mass_index + 1]
    curve_stable = [
        [mass, radius, lambda_value, pressure, density, cs2, surface_density]
        for mass, radius, lambda_value, pressure, density, cs2, surface_density in zip(
            mass_stable,
            radius_stable,
            lambda_stable,
            pressure_stable,
            density_stable,
            cs2_stable,
            eps_surf_stable,
        )
    ]
    dense_profiles_stable = dense_profiles[: max_mass_index + 1]
    max_mass_stable = float(mass_stable[max_mass_index])
    if return_sequence_evidence:
        stable_tidal_diagnostics = (
            tidal_diagnostics[: max_mass_index + 1]
            if return_tidal_diagnostics
            else None
        )
        return _build_sequence_evidence(
            full_sequence=full_sequence_evidence,
            stable_sequence=curve_stable,
            full_dense_profiles=full_dense_profiles_evidence,
            stable_dense_profiles=dense_profiles_stable,
            full_tidal_diagnostics=full_tidal_evidence,
            stable_tidal_diagnostics=(
                _tidal_diagnostic_rows(stable_tidal_diagnostics)
                if stable_tidal_diagnostics is not None
                else None
            ),
            full_lambda_diagnostics=(
                tuple(lambda_diagnostic_objects) if return_tidal_diagnostics else None
            ),
            stable_lambda_diagnostics=(
                tuple(lambda_diagnostic_objects[: max_mass_index + 1])
                if return_tidal_diagnostics
                else None
            ),
            attempted_central_pressures=attempted_pressures,
            failed_central_pressures=failure_details,
            sampled_peak_index=max_mass_index,
            sampled_secants=sampled_secants,
            eos_endpoint_pressure=(
                float(p_max_causal) if p_max_causal is not None else None
            ),
            max_mass_stable=max_mass_stable,
        )
    if return_tidal_diagnostics:
        return (
            curve_stable,
            dense_profiles_stable,
            max_mass_stable,
            tidal_diagnostics[: max_mass_index + 1],
        )
    return curve_stable, dense_profiles_stable, max_mass_stable


def _prefer_highest_evaluated_candidate(
    cache: dict[float, Any],
    *,
    lower_pressure: float,
    upper_pressure: float,
    refined_pressure: float,
    refined_star: Any,
) -> tuple[float, Any]:
    """Return the highest-mass model evaluated inside one turning bracket.

    A bounded optimizer can finish microscopically away from an already
    evaluated, marginally higher model on a numerically flat maximum.  Keep
    the optimizer candidate unless a successful in-bracket evaluation has a
    strictly greater mass.  This preserves the invariant that a resolved
    maximum is never smaller than its own saved refinement evidence without
    weakening downstream validation.
    """

    candidates = [
        (float(pressure), star)
        for pressure, star in cache.items()
        if lower_pressure < float(pressure) < upper_pressure
    ]
    if not candidates:
        return float(refined_pressure), refined_star
    best_pressure, best_star = max(
        candidates,
        key=lambda item: float(item[1].mass),
    )
    if float(best_star.mass) > float(refined_star.mass):
        return best_pressure, best_star
    return float(refined_pressure), refined_star


def resolve_maximum_mass(
    eos_callable: Callable,
    *,
    pressure_min_mev_fm3: float,
    pressure_max_mev_fm3: float,
    maximum_mass_threshold_msun: float = 1.95,
    initial_points: int = 17,
    refinement_pressure_rtol: float = 1.0e-7,
    rtol: float | None = None,
    atol: float | None = None,
    settings: TovConfig | None = None,
    star_solver: Callable[..., Any] | None = None,
) -> TovMaximumMassResult:
    """Resolve one nonrotating maximum mass from a true turning-point bracket.

    The search uses background-only TOV models, globally refines the pressure
    sampling at least once, requires a positive-to-negative ``dM/dP_c``
    secant transition, and then refines the interior maximum in log pressure.
    A sampled argmax is never promoted to ``M_max``.  Multiple sign changes,
    solver gaps, and contact with the EoS endpoint all fail closed.
    """

    pressure_min = _require_finite("pressure_min_mev_fm3", pressure_min_mev_fm3)
    pressure_max = _require_finite("pressure_max_mev_fm3", pressure_max_mev_fm3)
    threshold = _require_finite(
        "maximum_mass_threshold_msun", maximum_mass_threshold_msun
    )
    pressure_tolerance = _require_finite(
        "refinement_pressure_rtol", refinement_pressure_rtol
    )
    if not 0.0 < pressure_min < pressure_max:
        raise ValueError("maximum-mass pressure bounds must satisfy 0 < min < max")
    if threshold <= 0.0:
        raise ValueError("maximum_mass_threshold_msun must be positive")
    if pressure_tolerance <= 0.0:
        raise ValueError("refinement_pressure_rtol must be positive")
    if (
        isinstance(initial_points, bool)
        or not isinstance(initial_points, int)
        or initial_points < 9
        or initial_points % 2 == 0
    ):
        raise ValueError("initial_points must be an odd integer of at least 9")

    solver = solve_star if star_solver is None else star_solver
    cache: dict[float, Any] = {}
    failures: dict[float, str] = {}

    def evaluate(pressure: float) -> Any:
        key = float(pressure)
        if key in cache:
            return cache[key]
        if key in failures:
            raise RuntimeError(failures[key])
        try:
            solver_kwargs = {
                "rtol": rtol,
                "atol": atol,
                "settings": settings,
                "calculate_tidal": False,
            }
            # Preserve the established custom-star-solver contract.  The
            # lightweight profile flag is supplied only when this function
            # selected the production solver itself.
            if star_solver is None:
                solver_kwargs["retain_profile"] = False
            star = solver(eos_callable, key, **solver_kwargs)
            values = (
                float(star.mass),
                float(star.radius),
                float(star.central_energy_density),
                float(star.central_sound_speed_squared),
            )
            if not np.all(np.isfinite(values)) or values[0] <= 0.0 or values[1] <= 0.0:
                raise ValueError("background star returned invalid finite state")
            cache[key] = star
            return star
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            failures[key] = str(exc)
            raise

    pressures = tuple(
        float(value)
        for value in np.geomspace(pressure_min, pressure_max, initial_points)
    )
    global_rounds = 0
    transitions: list[tuple[int, str]] = []
    ordered_pressures: list[float] = []
    masses = np.asarray([], dtype=float)

    for target_round in range(3):
        for pressure in pressures:
            try:
                evaluate(pressure)
            except (ValueError, RuntimeError, ArithmeticError):
                pass
        if failures:
            break
        ordered_pressures = sorted(cache)
        masses = np.asarray(
            [float(cache[pressure].mass) for pressure in ordered_pressures],
            dtype=float,
        )
        slopes = np.diff(masses) / np.diff(np.asarray(ordered_pressures))
        transitions = []
        for index in range(len(slopes) - 1):
            if slopes[index] > 0.0 and slopes[index + 1] < 0.0:
                transitions.append((index, "positive_to_negative"))
            elif slopes[index] < 0.0 and slopes[index + 1] > 0.0:
                transitions.append((index, "negative_to_positive"))
            elif slopes[index] == 0.0 or slopes[index + 1] == 0.0:
                transitions.append((index, "zero_or_flat_ambiguous"))
        if target_round >= 1 and transitions:
            break
        if target_round == 2:
            break
        refined = set(ordered_pressures)
        for lower, upper in zip(ordered_pressures[:-1], ordered_pressures[1:]):
            refined.add(math.sqrt(lower * upper))
        pressures = tuple(sorted(refined))
        global_rounds += 1

    sampled_models = tuple(
        (
            pressure,
            float(cache[pressure].mass),
            float(cache[pressure].radius),
            float(cache[pressure].central_energy_density),
            float(cache[pressure].central_sound_speed_squared),
        )
        for pressure in sorted(cache)
    )
    endpoint_reached = pressure_max in cache

    def bracket_rows() -> tuple[tuple[float, ...], ...]:
        rows: list[tuple[float, ...]] = []
        if not ordered_pressures or len(masses) < 3:
            return ()
        for index, _kind in transitions:
            lower_pressure = ordered_pressures[index]
            middle_pressure = ordered_pressures[index + 1]
            upper_pressure = ordered_pressures[index + 2]
            lower_mass = float(masses[index])
            middle_mass = float(masses[index + 1])
            upper_mass = float(masses[index + 2])
            rows.append(
                (
                    lower_pressure,
                    middle_pressure,
                    upper_pressure,
                    lower_mass,
                    middle_mass,
                    upper_mass,
                    (middle_mass - lower_mass) / (middle_pressure - lower_pressure),
                    (upper_mass - middle_mass) / (upper_pressure - middle_pressure),
                )
            )
        return tuple(rows)

    brackets = bracket_rows()

    def unresolved(
        status: str,
        *,
        endpoint_limitation: str,
        stable_models: tuple[tuple[float, ...], ...] = (),
        refinement_status: str = "not_started",
        iterations: int = 0,
    ) -> TovMaximumMassResult:
        return TovMaximumMassResult(
            status=status,
            maximum_mass_resolved=False,
            maximum_mass_threshold_msun=threshold,
            passes_maximum_mass_threshold=None,
            maximum_mass_msun=None,
            central_pressure_mev_fm3=None,
            central_energy_density_mev_fm3=None,
            central_sound_speed_squared=None,
            radius_km=None,
            turning_point_brackets=brackets,
            selected_bracket=None,
            stable_branch_models=stable_models,
            sampled_models=sampled_models,
            positive_left_secant=None,
            negative_right_secant=None,
            eos_endpoint_pressure_mev_fm3=pressure_max,
            endpoint_reached=endpoint_reached,
            endpoint_limitation=endpoint_limitation,
            refinement_status=refinement_status,
            refinement_iterations=iterations,
            global_refinement_rounds=global_rounds,
            solver_call_count=len(cache) + len(failures),
            solver_failure_count=len(failures),
            solver_failures=tuple(sorted(failures.items())),
        )

    if failures:
        return unresolved(
            "unresolved_background_solver_failure",
            endpoint_limitation="one_or_more_background_models_failed",
        )
    if len(transitions) != 1 or transitions[0][1] != "positive_to_negative":
        if len(transitions) > 1 or (
            transitions and transitions[0][1] != "positive_to_negative"
        ):
            status = "unresolved_multiple_or_ambiguous_turning_points"
            limitation = "turning_point_structure_is_ambiguous"
        else:
            sampled_peak_index = int(np.argmax(masses)) if len(masses) else -1
            if 0 < sampled_peak_index < len(masses) - 1:
                status = "unresolved_sampled_peak_without_turning_point_bracket"
                limitation = "sampled_argmax_is_not_resolved_Mmax"
            else:
                status = "unresolved_no_turning_point_before_eos_endpoint"
                limitation = "eos_endpoint_reached_without_bracket"
        return unresolved(status, endpoint_limitation=limitation)

    selected = brackets[0]
    lower_pressure, _middle_pressure, upper_pressure = selected[:3]
    try:
        optimization = minimize_scalar(
            lambda log_pressure: -float(evaluate(math.exp(float(log_pressure))).mass),
            bounds=(math.log(lower_pressure), math.log(upper_pressure)),
            method="bounded",
            options={"xatol": pressure_tolerance, "maxiter": 128},
        )
        refined_pressure = math.exp(float(optimization.x))
        maximum_star = evaluate(refined_pressure)
        refined_pressure, maximum_star = _prefer_highest_evaluated_candidate(
            cache,
            lower_pressure=lower_pressure,
            upper_pressure=upper_pressure,
            refined_pressure=refined_pressure,
            refined_star=maximum_star,
        )
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        return unresolved(
            "unresolved_turning_point_refinement_failure",
            endpoint_limitation=f"bounded_refinement_failed:{exc}",
            refinement_status="failed",
        )
    lower_star = evaluate(lower_pressure)
    upper_star = evaluate(upper_pressure)
    left_secant = (float(maximum_star.mass) - float(lower_star.mass)) / (
        refined_pressure - lower_pressure
    )
    right_secant = (float(upper_star.mass) - float(maximum_star.mass)) / (
        upper_pressure - refined_pressure
    )
    refinement_iterations = int(getattr(optimization, "nfev", 0))
    if (
        not bool(optimization.success)
        or not lower_pressure < refined_pressure < upper_pressure
        or not left_secant > 0.0
        or not right_secant < 0.0
    ):
        return unresolved(
            "unresolved_turning_point_refinement_failure",
            endpoint_limitation=(
                "refinement_did_not_preserve_positive_to_negative_secants"
            ),
            refinement_status="failed_sign_validation",
            iterations=refinement_iterations,
        )
    maximum_model = (
        refined_pressure,
        float(maximum_star.mass),
        float(maximum_star.radius),
        float(maximum_star.central_energy_density),
        float(maximum_star.central_sound_speed_squared),
    )
    stable_models = tuple(
        sorted(
            [row for row in sampled_models if row[0] < refined_pressure]
            + [maximum_model],
            key=lambda row: row[0],
        )
    )
    return TovMaximumMassResult(
        status="resolved_unique_turning_point",
        maximum_mass_resolved=True,
        maximum_mass_threshold_msun=threshold,
        passes_maximum_mass_threshold=bool(maximum_star.mass >= threshold),
        maximum_mass_msun=float(maximum_star.mass),
        central_pressure_mev_fm3=refined_pressure,
        central_energy_density_mev_fm3=float(maximum_star.central_energy_density),
        central_sound_speed_squared=float(maximum_star.central_sound_speed_squared),
        radius_km=float(maximum_star.radius),
        turning_point_brackets=brackets,
        selected_bracket=selected,
        stable_branch_models=stable_models,
        sampled_models=sampled_models,
        positive_left_secant=float(left_secant),
        negative_right_secant=float(right_secant),
        eos_endpoint_pressure_mev_fm3=pressure_max,
        endpoint_reached=endpoint_reached,
        endpoint_limitation=None,
        refinement_status="converged_bounded_log_pressure",
        refinement_iterations=refinement_iterations,
        global_refinement_rounds=global_rounds,
        solver_call_count=len(cache) + len(failures),
        solver_failure_count=len(failures),
        solver_failures=tuple(sorted(failures.items())),
    )


def _local_refinement_pressures(
    lower: float,
    middle: float,
    upper: float,
    points: int,
    policy: str | None,
) -> np.ndarray:
    if policy is None:
        # Preserve the established hadronic grid byte for byte.
        return np.geomspace(lower, upper, points)
    if policy != SEED_PRESERVING_LOCAL_REFINEMENT_POLICY:
        raise ValueError("unknown stellar local-refinement policy")
    if not 0.0 < lower < middle < upper or points < 7 or points % 2 == 0:
        raise ValueError("invalid seed-preserving local-refinement bracket")
    half_points = (points + 1) // 2
    left = np.geomspace(lower, middle, half_points)
    right = np.geomspace(middle, upper, half_points)
    # Use the original three solved nodes as exact endpoints of the two
    # subdivisions. Recomputing the central node via exp/log can produce a
    # second, near-equal pressure and a spurious mass secant. No tolerance-
    # based merging, smoothing, or change to the sign test is involved.
    left[0], left[-1] = lower, middle
    right[0], right[-1] = middle, upper
    pressures = np.concatenate((left[:-1], right))
    if not np.all(np.diff(pressures) > 0.0):
        raise ValueError("local-refinement nodes are not representably distinct")
    return pressures


def refine_maximum_mass_from_sequence(
    eos_callable: Callable,
    evidence: TovSequenceEvidence,
    *,
    maximum_mass_threshold_msun: float = 1.95,
    local_points: int = 9,
    refinement_pressure_rtol: float = 5.0e-4,
    rtol: float | None = None,
    atol: float | None = None,
    settings: TovConfig | None = None,
    star_solver: Callable[..., Any] | None = None,
) -> TovMaximumMassResult:
    """Resolve ``M_max`` by refining an existing sampled turning bracket.

    The sequence remains the global search.  A unique sampled
    positive-to-negative mass secant transition is populated with a small odd
    log-pressure grid, then refined with background-only stars.  Previously
    calculated sequence points are reused exactly; no tidal calculation or
    dense radial profile is repeated.  An endpoint argmax is never promoted
    to ``M_max``.
    """

    if not isinstance(evidence, TovSequenceEvidence):
        raise TypeError("evidence must be TovSequenceEvidence")
    threshold = _require_finite(
        "maximum_mass_threshold_msun", maximum_mass_threshold_msun
    )
    pressure_tolerance = _require_finite(
        "refinement_pressure_rtol", refinement_pressure_rtol
    )
    if threshold <= 0.0:
        raise ValueError("maximum_mass_threshold_msun must be positive")
    if pressure_tolerance <= 0.0:
        raise ValueError("refinement_pressure_rtol must be positive")
    if (
        isinstance(local_points, bool)
        or not isinstance(local_points, int)
        or local_points < 7
        or local_points % 2 == 0
    ):
        raise ValueError("local_points must be an odd integer of at least 7")

    rows = tuple(evidence.full_sequence)
    endpoint_pressure = float(
        evidence.eos_endpoint_pressure
        if evidence.eos_endpoint_pressure is not None
        else (rows[-1][3] if rows else 0.0)
    )
    endpoint_reached = bool(evidence.final_available_model_contacts_eos_endpoint)
    seed_cache = {
        float(row[3]): SimpleNamespace(
            mass=float(row[0]),
            radius=float(row[1]),
            central_energy_density=float(row[4]),
            central_sound_speed_squared=float(row[5]),
        )
        for row in rows
    }
    cache = dict(seed_cache)
    failures: dict[float, str] = {}
    solver = solve_star if star_solver is None else star_solver

    def model_rows() -> tuple[tuple[float, ...], ...]:
        return tuple(
            (
                pressure,
                float(cache[pressure].mass),
                float(cache[pressure].radius),
                float(cache[pressure].central_energy_density),
                float(cache[pressure].central_sound_speed_squared),
            )
            for pressure in sorted(cache)
        )

    def evaluate(pressure: float) -> Any:
        key = float(pressure)
        if key in cache:
            return cache[key]
        if key in failures:
            raise RuntimeError(failures[key])
        try:
            kwargs = {
                "rtol": rtol,
                "atol": atol,
                "settings": settings,
                "calculate_tidal": False,
            }
            if star_solver is None:
                kwargs["retain_profile"] = False
            star = solver(eos_callable, key, **kwargs)
            values = (
                float(star.mass),
                float(star.radius),
                float(star.central_energy_density),
                float(star.central_sound_speed_squared),
            )
            if not np.all(np.isfinite(values)) or values[0] <= 0.0 or values[1] <= 0.0:
                raise ValueError("background star returned invalid finite state")
            cache[key] = star
            return star
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            failures[key] = str(exc)
            raise

    def transition_brackets(
        pressures: list[float],
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[str, ...]]:
        masses = np.asarray(
            [float(cache[pressure].mass) for pressure in pressures],
            dtype=float,
        )
        slopes = np.diff(masses) / np.diff(np.asarray(pressures, dtype=float))
        brackets: list[tuple[float, ...]] = []
        kinds: list[str] = []
        for index in range(len(slopes) - 1):
            if slopes[index] > 0.0 and slopes[index + 1] < 0.0:
                kind = "positive_to_negative"
            elif slopes[index] < 0.0 and slopes[index + 1] > 0.0:
                kind = "negative_to_positive"
            elif slopes[index] == 0.0 or slopes[index + 1] == 0.0:
                kind = "zero_or_flat_ambiguous"
            else:
                continue
            lower = pressures[index]
            middle = pressures[index + 1]
            upper = pressures[index + 2]
            brackets.append(
                (
                    lower,
                    middle,
                    upper,
                    float(masses[index]),
                    float(masses[index + 1]),
                    float(masses[index + 2]),
                    float(slopes[index]),
                    float(slopes[index + 1]),
                )
            )
            kinds.append(kind)
        return tuple(brackets), tuple(kinds)

    original_pressures = sorted(seed_cache)
    brackets, kinds = (
        transition_brackets(original_pressures) if len(rows) >= 3 else ((), ())
    )

    def unresolved(
        status: str,
        limitation: str,
        *,
        selected: tuple[float, ...] | None = None,
        refinement_status: str = "not_started",
        iterations: int = 0,
    ) -> TovMaximumMassResult:
        sampled = model_rows()
        return TovMaximumMassResult(
            status=status,
            maximum_mass_resolved=False,
            maximum_mass_threshold_msun=threshold,
            passes_maximum_mass_threshold=None,
            maximum_mass_msun=None,
            central_pressure_mev_fm3=None,
            central_energy_density_mev_fm3=None,
            central_sound_speed_squared=None,
            radius_km=None,
            turning_point_brackets=brackets,
            selected_bracket=selected,
            stable_branch_models=(),
            sampled_models=sampled,
            positive_left_secant=None,
            negative_right_secant=None,
            eos_endpoint_pressure_mev_fm3=endpoint_pressure,
            endpoint_reached=endpoint_reached,
            endpoint_limitation=limitation,
            refinement_status=refinement_status,
            refinement_iterations=iterations,
            global_refinement_rounds=0,
            solver_call_count=len(cache) - len(seed_cache) + len(failures),
            solver_failure_count=len(failures),
            solver_failures=tuple(sorted(failures.items())),
        )

    if evidence.failed_central_pressures:
        return unresolved(
            "unresolved_sampled_sequence_solver_failure",
            "sampled_sequence_contains_solver_gaps",
        )
    if len(brackets) != 1 or kinds != ("positive_to_negative",):
        if len(brackets) > 1 or any(kind != "positive_to_negative" for kind in kinds):
            return unresolved(
                "unresolved_multiple_or_ambiguous_turning_points",
                "sampled_turning_point_structure_is_ambiguous",
            )
        return unresolved(
            "unresolved_no_turning_point_before_eos_endpoint",
            "eos_endpoint_reached_without_sampled_turning_bracket",
        )

    selected = brackets[0]
    lower_pressure, middle_pressure, upper_pressure = selected[:3]
    local_grid = _local_refinement_pressures(
        lower_pressure,
        middle_pressure,
        upper_pressure,
        local_points,
        getattr(eos_callable, "stellar_local_refinement_policy", None),
    )
    for pressure in local_grid:
        try:
            evaluate(float(pressure))
        except (ValueError, RuntimeError, ArithmeticError):
            pass
    if failures:
        return unresolved(
            "unresolved_turning_point_refinement_failure",
            "one_or_more_local_background_models_failed",
            selected=selected,
            refinement_status="failed_local_grid",
        )

    local_pressures = [
        pressure
        for pressure in sorted(cache)
        if lower_pressure <= pressure <= upper_pressure
    ]
    local_brackets, local_kinds = transition_brackets(local_pressures)
    brackets = local_brackets
    if len(local_brackets) != 1 or local_kinds != ("positive_to_negative",):
        return unresolved(
            "unresolved_multiple_or_ambiguous_turning_points",
            "local_refinement_did_not_preserve_one_turning_point",
            refinement_status="failed_local_sign_structure",
        )
    selected = local_brackets[0]
    lower_pressure, _middle_pressure, upper_pressure = selected[:3]
    try:
        optimization = minimize_scalar(
            lambda log_pressure: -float(evaluate(math.exp(float(log_pressure))).mass),
            bounds=(math.log(lower_pressure), math.log(upper_pressure)),
            method="bounded",
            options={"xatol": pressure_tolerance, "maxiter": 32},
        )
        refined_pressure = math.exp(float(optimization.x))
        maximum_star = evaluate(refined_pressure)
        refined_pressure, maximum_star = _prefer_highest_evaluated_candidate(
            cache,
            lower_pressure=lower_pressure,
            upper_pressure=upper_pressure,
            refined_pressure=refined_pressure,
            refined_star=maximum_star,
        )
        lower_star = evaluate(lower_pressure)
        upper_star = evaluate(upper_pressure)
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        return unresolved(
            "unresolved_turning_point_refinement_failure",
            f"bounded_local_refinement_failed:{exc}",
            selected=selected,
            refinement_status="failed_bounded_refinement",
        )
    left_secant = (float(maximum_star.mass) - float(lower_star.mass)) / (
        refined_pressure - lower_pressure
    )
    right_secant = (float(upper_star.mass) - float(maximum_star.mass)) / (
        upper_pressure - refined_pressure
    )
    iterations = int(getattr(optimization, "nfev", 0))
    if (
        not bool(optimization.success)
        or not lower_pressure < refined_pressure < upper_pressure
        or not left_secant > 0.0
        or not right_secant < 0.0
    ):
        return unresolved(
            "unresolved_turning_point_refinement_failure",
            "refinement_did_not_preserve_positive_to_negative_secants",
            selected=selected,
            refinement_status="failed_sign_validation",
            iterations=iterations,
        )

    maximum_model = (
        refined_pressure,
        float(maximum_star.mass),
        float(maximum_star.radius),
        float(maximum_star.central_energy_density),
        float(maximum_star.central_sound_speed_squared),
    )
    sampled = model_rows()
    stable = tuple(
        sorted(
            [row for row in sampled if row[0] < refined_pressure] + [maximum_model],
            key=lambda row: row[0],
        )
    )
    return TovMaximumMassResult(
        status="resolved_unique_turning_point_local_sequence_refinement",
        maximum_mass_resolved=True,
        maximum_mass_threshold_msun=threshold,
        passes_maximum_mass_threshold=bool(maximum_star.mass >= threshold),
        maximum_mass_msun=float(maximum_star.mass),
        central_pressure_mev_fm3=refined_pressure,
        central_energy_density_mev_fm3=float(maximum_star.central_energy_density),
        central_sound_speed_squared=float(maximum_star.central_sound_speed_squared),
        radius_km=float(maximum_star.radius),
        turning_point_brackets=brackets,
        selected_bracket=selected,
        stable_branch_models=stable,
        sampled_models=sampled,
        positive_left_secant=float(left_secant),
        negative_right_secant=float(right_secant),
        eos_endpoint_pressure_mev_fm3=endpoint_pressure,
        endpoint_reached=endpoint_reached,
        endpoint_limitation=None,
        refinement_status="converged_local_bounded_log_pressure",
        refinement_iterations=iterations,
        global_refinement_rounds=0,
        solver_call_count=len(cache) - len(seed_cache) + len(failures),
        solver_failure_count=len(failures),
        solver_failures=tuple(sorted(failures.items())),
    )


def _tidal_jump_evidence_columns(diagnostic: Any) -> dict[str, Any]:
    """Return strict, table-safe discontinuity evidence for one tidal solve."""

    surface_jumps = tuple(
        item for item in diagnostic.applied_jumps if item.kind == "surface"
    )
    surface = surface_jumps[0] if len(surface_jumps) == 1 else None
    payload = diagnostic.to_dict()
    return {
        "tidal_expected_jump_count": diagnostic.expected_jump_count,
        "tidal_applied_jump_count": diagnostic.applied_jump_count,
        "tidal_surface_jump_count": len(surface_jumps),
        "tidal_surface_delta_y": None if surface is None else surface.delta_y,
        "tidal_surface_y_before": None if surface is None else surface.y_before,
        "tidal_surface_y_after": None if surface is None else surface.y_after,
        "tidal_surface_event_pressure_mev_fm3": (diagnostic.surface_event_pressure),
        "tidal_jump_evidence_json": json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }


def _sequence_frame(
    case_id: str,
    stage: str,
    evidence: TovSequenceEvidence,
    *,
    retain_jump_evidence: bool = False,
) -> pd.DataFrame:
    success = list(evidence.full_sequence)
    diagnostics = list(evidence.full_lambda_diagnostics or ())
    successful_by_pressure = {float(row[3]): (row, i) for i, row in enumerate(success)}
    failures = {
        float(item.central_pressure): item for item in evidence.failed_central_pressures
    }
    rows: list[dict[str, Any]] = []
    segment = 0
    for attempted_index, pressure in enumerate(evidence.attempted_central_pressures):
        pressure = float(pressure)
        if pressure in failures:
            failure = failures[pressure]
            rows.append(
                {
                    "case_id": case_id,
                    "stage": stage,
                    "attempted_index": attempted_index,
                    "segment_id": segment,
                    "calculation_status": "failed",
                    "failure_category": failure.category,
                    "failure_reason": failure.reason,
                    "central_pressure_mev_fm3": pressure,
                }
            )
            segment += 1
            continue
        if pressure not in successful_by_pressure:
            continue
        row, success_index = successful_by_pressure[pressure]
        record = {
            "case_id": case_id,
            "stage": stage,
            "attempted_index": attempted_index,
            "segment_id": segment,
            "calculation_status": "success",
            "failure_category": None,
            "failure_reason": None,
            **{field: float(value) for field, value in zip(TOV_SEQUENCE_FIELDS, row)},
            "central_pressure_mev_fm3": float(row[3]),
            "is_sampled_peak": (success_index == evidence.sampled_peak_index),
            "is_domain_end": success_index == len(success) - 1,
            "is_on_successful_stable_prefix": success_index
            < len(evidence.stable_sequence),
        }
        if success_index < len(diagnostics):
            diagnostic = diagnostics[success_index]
            record.update(
                {
                    "k2": diagnostic.k2,
                    "tidal_status": diagnostic.scientific_status,
                    "tidal_failure_reason": diagnostic.failure_reason,
                }
            )
            if retain_jump_evidence:
                record.update(_tidal_jump_evidence_columns(diagnostic))
        rows.append(record)
    if not rows:
        return pd.DataFrame(
            columns=[
                "case_id",
                "stage",
                "attempted_index",
                "segment_id",
                "calculation_status",
                "failure_category",
                "failure_reason",
                "central_pressure_mev_fm3",
                *TOV_SEQUENCE_FIELDS,
                "k2",
                "tidal_status",
                "tidal_failure_reason",
                "is_on_successful_stable_prefix",
            ]
        )
    return pd.DataFrame(rows)


SavedStellarSchema = Literal["sequence", "fixed_mass"]


ResponseCoordinate = Literal["amplitude", "delta"]


@dataclass(frozen=True)
class _TidalSchema:
    background_column: str
    background_success: str
    lambda_column: str


_TIDAL_SCHEMAS: dict[SavedStellarSchema, _TidalSchema] = {
    "sequence": _TidalSchema(
        background_column="calculation_status",
        background_success="success",
        lambda_column="Lambda",
    ),
    "fixed_mass": _TidalSchema(
        background_column="status",
        background_success="bracketed_and_solved",
        lambda_column="lambda_dimensionless",
    ),
}


def classify_saved_tidal_rows(
    frame: pd.DataFrame,
    *,
    schema: SavedStellarSchema,
) -> pd.DataFrame:
    """Classify every saved row using the repository's full tidal definition.

    A row is valid only when its table-specific background succeeded, its
    status is exactly ``validated_lambda_validation_v1``, ``k2`` and Lambda
    are finite, and Lambda is strictly positive.
    """

    definition = _TIDAL_SCHEMAS[schema]
    result = pd.DataFrame(index=frame.index)
    result["background_success"] = False
    result["tidal_valid"] = False
    result["tidal_validity_reason"] = "unclassified"
    required = (
        definition.background_column,
        "tidal_status",
        "k2",
        definition.lambda_column,
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        result["tidal_validity_reason"] = "missing_required_column:" + ",".join(missing)
        return result

    background = (
        frame[definition.background_column]
        .astype(str)
        .eq(definition.background_success)
    )
    status = frame["tidal_status"].astype(str).eq(LAMBDA_FRAMEWORK_CAPABILITY)
    k2 = pd.to_numeric(frame["k2"], errors="coerce")
    lambda_value = pd.to_numeric(frame[definition.lambda_column], errors="coerce")
    finite_k2 = pd.Series(np.isfinite(k2), index=frame.index, dtype=bool)
    finite_lambda = pd.Series(np.isfinite(lambda_value), index=frame.index, dtype=bool)
    positive_lambda = lambda_value.gt(0.0)
    valid = background & status & finite_k2 & finite_lambda & positive_lambda

    reason = pd.Series("valid", index=frame.index, dtype=object)
    reason.loc[~background] = "background_not_successful"
    reason.loc[background & ~status] = "tidal_status_not_validated"
    reason.loc[background & status & ~finite_k2] = "k2_nonfinite"
    reason.loc[background & status & finite_k2 & ~finite_lambda] = "lambda_nonfinite"
    reason.loc[background & status & finite_k2 & finite_lambda & ~positive_lambda] = (
        "lambda_not_strictly_positive"
    )

    result["background_success"] = background
    result["tidal_valid"] = valid
    result["tidal_validity_reason"] = reason
    return result


_MAXIMUM_AUTOMATIC_STELLAR_WORKERS = 6


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
        not np.all(np.isfinite(stable[:, 3])) or np.any(stable[:, 3] > endpoint)
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
        retain_profile=config.retained_stellar_profiles_requested,
    )
    tidal = star.lambda_diagnostic
    result = {
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
            None
            if tidal.lambda_dimensionless is None
            else float(tidal.lambda_dimensionless)
        ),
        "tidal_status": tidal.scientific_status,
        "tidal_failure_reason": tidal.failure_reason,
        "bracket_pressure_mev_fm3": [lower, upper],
        "root_xtol_mev_fm3": config.fixed_mass_root_xtol_mev_fm3,
        "root_evaluation_count": len(background_cache) + 1,
    }
    if bool(getattr(eos, "requires_discontinuity_metadata", False)):
        result.update(_tidal_jump_evidence_columns(tidal))
    return result, star


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
            reason = (
                "requested_stage_row_missing"
                if stage_rows.empty
                else "duplicate_stage_rows"
            )
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
        tidal_validity_reason = str(tidal_classification["tidal_validity_reason"])
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
                lambda value: value is True or str(value).strip().lower() == "true"
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
        for target_mass in (
            config.fixed_masses_msun if config.fixed_mass_background_requested else ()
        ):
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
            sequences.loc[sequences["case_id"].astype(str).eq(str(case_id))],
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
            retain_profiles=bool(
                getattr(config, "retained_stellar_profiles_requested", True)
            ),
        )
        if not isinstance(evidence, TovSequenceEvidence):
            raise TypeError("shared solve_sequence did not return TovSequenceEvidence")
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
        maximum_row: dict[str, Any] | None = None
        maximum_report: dict[str, Any] | None = None
        if getattr(
            config,
            "maximum_mass_requested",
            config.background_tov_requested,
        ):
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
                "maximum_mass_threshold_msun": (maximum.maximum_mass_threshold_msun),
                "passes_maximum_mass_threshold": (
                    maximum.passes_maximum_mass_threshold
                ),
                "central_pressure_mev_fm3": (maximum.central_pressure_mev_fm3),
                "central_energy_density_mev_fm3": (
                    maximum.central_energy_density_mev_fm3
                ),
                "central_sound_speed_squared": (maximum.central_sound_speed_squared),
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
            maximum_report = maximum.to_dict()
        frame = _sequence_frame(
            case_id,
            stage.name,
            evidence,
            retain_jump_evidence=bool(
                getattr(eos, "requires_discontinuity_metadata", False)
            ),
        )
        if case_id == "direct":
            amplitude = None
            delta = None
        else:
            amplitude = eos.deformation.amplitude
            delta = eos.deformation.delta_mev_fm3
        if maximum_row is not None:
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
        for target_mass in (
            config.fixed_masses_msun if config.fixed_mass_background_requested else ()
        ):
            result, star = _fixed_mass_result(eos, evidence, target_mass, config, stage)
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
            "sequence_evidence": {
                "attempted_count": len(evidence.attempted_central_pressures),
                "successful_count": len(evidence.full_sequence),
                "failed_count": len(evidence.failed_central_pressures),
                "stable_prefix_count": len(evidence.stable_sequence),
                "sampled_peak_index": evidence.sampled_peak_index,
                "eos_endpoint_pressure": evidence.eos_endpoint_pressure,
            },
            "sequence_frame": frame,
            "fixed_rows": fixed_rows,
            "stars": stars,
            "maximum_row": maximum_row,
            "maximum_report": maximum_report,
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
) -> tuple[
    pd.DataFrame, pd.DataFrame, dict[str, Any], dict[tuple[str, str, float], Any]
]:
    direct_eos = getattr(baseline, "eos", baseline)
    owner = getattr(config, "zero_amplitude_control_owner", None)
    include_direct = not (isinstance(owner, bool) and owner is False)
    eos_map: dict[str, Any] = {
        **({"direct": direct_eos} if include_direct else {}),
        **generated,
    }
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
                completed = 0
                for future in as_completed(futures):
                    case_id = futures[future]
                    result = dict(future.result())
                    if result.get("case_id") != case_id:
                        raise RuntimeError(
                            "stellar worker returned a mismatched case ID"
                        )
                    case_results[case_id] = result
                    completed += 1
                    if os.environ.get("EOS_GENERATION_PROGRESS") == "1":
                        print(
                            f"[{getattr(config, 'matter_model', 'bsk24').upper()}] "
                            f"stellar case {completed}/{len(futures)}: {case_id} "
                            f"({float(result['worker_wall_seconds']):.1f} s worker)",
                            flush=True,
                        )
            except Exception:
                for future in futures:
                    future.cancel()
                raise
    else:
        selected_workers = 1
        for case_id, eos in eos_map.items():
            case_results[case_id] = _run_case_job(config, case_id, eos)
            if os.environ.get("EOS_GENERATION_PROGRESS") == "1":
                print(
                    f"[{getattr(config, 'matter_model', 'bsk24').upper()}] "
                    f"stellar case {len(case_results)}/{len(eos_map)}: {case_id} "
                    f"({float(case_results[case_id]['worker_wall_seconds']):.1f} s worker)",
                    flush=True,
                )

    for stage in config.tov_stages:
        for case_id in eos_map:
            stage_result = case_results[case_id]["stages"][stage.name]
            sequence_frames.append(stage_result["sequence_frame"])
            fixed_rows.extend(stage_result["fixed_rows"])
            stars.update(stage_result["stars"])
            if stage_result["maximum_row"] is not None:
                maximum_rows.append(stage_result["maximum_row"])
            if stage_result["maximum_report"] is not None:
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
    resolved_count = sum(bool(row["maximum_mass_resolved"]) for row in maximum_rows)
    convergence.update(
        {
            "sequence_evidence": {
                f"{case_id}:{stage.name}": case_results[case_id]["stages"][stage.name][
                    "sequence_evidence"
                ]
                for stage in config.tov_stages
                for case_id in eos_map
            },
            "maximum_mass_policy": (
                "not_requested_curve_only"
                if not getattr(
                    config,
                    "maximum_mass_requested",
                    config.background_tov_requested,
                )
                else "M_max requires one sampled positive-to-negative dM/dP_c "
                "bracket followed by local background-only refinement"
            ),
            "maximum_mass_case_stage_count": len(maximum_rows),
            "resolved_maximum_mass_case_stage_count": resolved_count,
            "maximum_mass_rows": maximum_rows,
            "maximum_mass_reports": maximum_reports,
            "background_solver_call_count": sum(
                int(row["local_background_solver_call_count"]) for row in maximum_rows
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
            {int(result["worker_pid"]) for result in case_results.values()}
        ),
        "case_worker_wall_seconds": {
            case_id: float(case_results[case_id]["worker_wall_seconds"])
            for case_id in eos_map
        },
        "deterministic_parent_merge_order": "stage_major_case_major",
        "nested_process_pool_disabled": bool(
            use_processes or os.environ.get(_OUTER_NOTEBOOK_WORKER_ENV) == "1"
        ),
    }
    return sequences, fixed, convergence, stars

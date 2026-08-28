"""TOV sequence workers, sampling, and immutable evidence assembly."""

from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np

from eos_generation._internal.config import TovConfig
from eos_generation.stellar._tov_integration import solve_star, _resolved_discontinuities
from eos_generation.stellar.discontinuities import BARE_SELF_BOUND_SEQUENCE_POLICY
from eos_generation.stellar._tov_types import (
    TOV_SEQUENCE_FIELDS,
    TOV_TIDAL_DIAGNOSTIC_FIELDS,
    TovConvergenceError,
    TovFailureDetail,
    TovLambdaDiagnostic,
    TovMassSecantEvidence,
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
)

logger = logging.getLogger("eos_generation.stellar.tov")

_PRODUCTION_SOLVE_STAR = solve_star
_SEQUENCE_WORKER_STATE: tuple[
    Callable,
    float | None,
    float | None,
    TovConfig,
    bool,
] | None = None


def _initialize_sequence_worker(
    eos_callable: Callable,
    rtol: float | None,
    atol: float | None,
    settings: TovConfig,
    calculate_tidal: bool,
) -> None:
    global _SEQUENCE_WORKER_STATE
    os.environ[_OUTER_PARALLEL_WORKER_ENV] = "1"
    _SEQUENCE_WORKER_STATE = (
        eos_callable,
        rtol,
        atol,
        settings,
        calculate_tidal,
    )


def _solve_sequence_pressure_worker(
    index: int,
    central_pressure: float,
) -> tuple[int, TovStarResult | None, str | None]:
    if _SEQUENCE_WORKER_STATE is None:
        raise RuntimeError("TOV sequence worker was not initialized")
    eos_callable, rtol, atol, settings, calculate_tidal = (
        _SEQUENCE_WORKER_STATE
    )
    try:
        star = solve_star(
            eos_callable,
            central_pressure,
            rtol=rtol,
            atol=atol,
            settings=settings,
            calculate_tidal=calculate_tidal,
        )
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
    if not module_name.startswith("eos_generation.bsk24") and not bool(
        getattr(eos_callable, "allow_parallel_tov_sequence", False)
    ):
        return False
    try:
        pickle.dumps(
            (eos_callable, settings, rtol, atol, calculate_tidal)
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
        sign = "positive" if delta_mass > 0.0 else "negative" if delta_mass < 0.0 else "zero"
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
) -> tuple | TovSequenceEvidence:
    """Integrate a sequence, preserving the historical tuple unless evidence is requested."""
    resolved = _DEFAULT_TOV if settings is None else settings
    if not isinstance(calculate_tidal, bool):
        raise ValueError("calculate_tidal must be boolean")
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
            raise ValueError("bare self-bound sequence policy requires one finite-density vacuum surface")

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

    parallel_outcomes: dict[
        int, tuple[TovStarResult | None, str | None]
    ] | None = None
    sequence_workers = _automatic_sequence_worker_count(len(pressures))
    if sequence_workers > 1 and _sequence_process_worker_is_safe(
        eos_callable,
        resolved,
        rtol,
        atol,
        calculate_tidal,
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
                star = solve_star(
                    eos_callable,
                    float(pc),
                    rtol=rtol,
                    atol=atol,
                    settings=resolved,
                    calculate_tidal=calculate_tidal,
                )
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
                not np.isfinite(star.radius) or not np.isfinite(star.mass)
                or star.radius <= 0.0 or star.mass <= 0.0
            ):
                record_failure(pc, "invalid_self_bound_mass_or_radius", f"surface mass={star.mass!r}, radius={star.radius!r}")
                continue
            # Bare self-bound matter has a physical low-mass M~R^3 branch;
            # legacy hadronic display cutoffs are not a validity condition.
            # The explicit policy changes no legacy BSk24 behavior.
            if not bare_self_bound and (star.radius < _MIN_RADIUS_CUTOFF or star.mass < _MIN_MASS_CUTOFF):
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
            eos_endpoint_pressure=(float(p_max_causal) if p_max_causal is not None else None),
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

_PUBLIC_MODULE = "eos_generation.stellar.tov"
for _compatibility_function in (
    _initialize_sequence_worker,
    _solve_sequence_pressure_worker,
    _automatic_sequence_worker_count,
    _sequence_process_worker_is_safe,
    _tidal_diagnostic_rows,
    _sampled_mass_secants,
    _successful_pressure_ordering,
    _build_sequence_evidence,
    solve_sequence,
):
    _compatibility_function.__module__ = _PUBLIC_MODULE
del _compatibility_function

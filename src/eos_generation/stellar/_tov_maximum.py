"""Fail-closed turning-point and maximum-mass resolution."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize_scalar

from eos_generation._internal.config import TovConfig
from eos_generation.stellar._tov_algebra import _require_finite
from eos_generation.stellar._tov_integration import solve_star
from eos_generation.stellar.discontinuities import (
    SEED_PRESERVING_LOCAL_REFINEMENT_POLICY,
)
from eos_generation.stellar._tov_types import (
    TovMaximumMassResult,
    TovSequenceEvidence,
)


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

    pressure_min = _require_finite(
        "pressure_min_mev_fm3", pressure_min_mev_fm3
    )
    pressure_max = _require_finite(
        "pressure_max_mev_fm3", pressure_max_mev_fm3
    )
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
                    (middle_mass - lower_mass)
                    / (middle_pressure - lower_pressure),
                    (upper_mass - middle_mass)
                    / (upper_pressure - middle_pressure),
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
            lambda log_pressure: -float(
                evaluate(math.exp(float(log_pressure))).mass
            ),
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
    left_secant = (
        float(maximum_star.mass) - float(lower_star.mass)
    ) / (refined_pressure - lower_pressure)
    right_secant = (
        float(upper_star.mass) - float(maximum_star.mass)
    ) / (upper_pressure - refined_pressure)
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
        central_energy_density_mev_fm3=float(
            maximum_star.central_energy_density
        ),
        central_sound_speed_squared=float(
            maximum_star.central_sound_speed_squared
        ),
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
    lower: float, middle: float, upper: float, points: int, policy: str | None,
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
    endpoint_reached = bool(
        evidence.final_available_model_contacts_eos_endpoint
    )
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
            if (
                not np.all(np.isfinite(values))
                or values[0] <= 0.0
                or values[1] <= 0.0
            ):
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
    brackets, kinds = transition_brackets(original_pressures) if len(rows) >= 3 else ((), ())

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
        lower_pressure, middle_pressure, upper_pressure, local_points,
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
            lambda log_pressure: -float(
                evaluate(math.exp(float(log_pressure))).mass
            ),
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
    left_secant = (
        float(maximum_star.mass) - float(lower_star.mass)
    ) / (refined_pressure - lower_pressure)
    right_secant = (
        float(upper_star.mass) - float(maximum_star.mass)
    ) / (upper_pressure - refined_pressure)
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
            [row for row in sampled if row[0] < refined_pressure]
            + [maximum_model],
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
        central_energy_density_mev_fm3=float(
            maximum_star.central_energy_density
        ),
        central_sound_speed_squared=float(
            maximum_star.central_sound_speed_squared
        ),
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

_PUBLIC_MODULE = "eos_generation.stellar.tov"
for _compatibility_function in (
    resolve_maximum_mass,
    refine_maximum_mass_from_sequence,
):
    _compatibility_function.__module__ = _PUBLIC_MODULE
del _compatibility_function

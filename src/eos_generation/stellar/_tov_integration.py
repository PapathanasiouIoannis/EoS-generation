"""Segmented background and tidal integration for individual TOV stars."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

from eos_generation._internal.config import TovConfig
from eos_generation.stellar._tov_algebra import (
    _require_finite,
    _tov_equations_with_limit,
    taylor_expansion,
    tidal_algebra,
    tidal_jump_delta_y,
)
from eos_generation.stellar._tov_types import (
    LAMBDA_SCIENTIFIC_STATUS,
    TIDAL_NOT_REQUESTED_STATUS,
    AppliedTidalJump,
    TovLambdaDiagnostic,
    TovStarResult,
    _A_CONV,
    _DEFAULT_TOV,
    _G_CONV,
    _TIDAL_SEGMENT_BOUNDARY_ULPS,
)
from eos_generation.stellar.discontinuities import (
    EosDiscontinuity,
    validate_discontinuity_sequence,
)

@dataclass
class _BackgroundSegment:
    solution: Any
    upper_discontinuity: EosDiscontinuity | None
    lower_discontinuity: EosDiscontinuity | None

    @property
    def radius_start(self) -> float:
        return float(self.solution.t[0])

    @property
    def radius_end(self) -> float:
        return float(self.solution.t_events[0][0])

    @property
    def event_state(self) -> np.ndarray:
        return np.asarray(self.solution.y_events[0][0], dtype=float)


def _tidal_segment_bounds(
    segment: _BackgroundSegment,
) -> tuple[float, float, float]:
    """Return validated radial bounds and their floating-point allowance."""
    radius_start = segment.radius_start
    radius_end = segment.radius_end
    if not math.isfinite(radius_start) or not math.isfinite(radius_end):
        raise ValueError("tidal background segment bounds must be finite")
    if radius_end <= radius_start:
        raise ValueError("tidal background segment must have positive radial span")
    allowance = _TIDAL_SEGMENT_BOUNDARY_ULPS * math.ulp(
        max(abs(radius_start), abs(radius_end), 1.0)
    )
    return radius_start, radius_end, allowance


def _bounded_tidal_first_step(
    segment: _BackgroundSegment,
    settings: TovConfig,
    *,
    scale: float = 1.0,
) -> float:
    """Return a positive RK45 first step bounded by the background segment.

    The nominal scale is the existing center/start radius (``radius_min_km``).
    The private ``scale`` argument exists only for numerical sensitivity tests.
    """
    radius_start, radius_end, _allowance = _tidal_segment_bounds(segment)
    multiplier = float(scale)
    nominal = float(settings.radius_min_km) * multiplier
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("tidal first-step scale must be finite and positive")
    if not math.isfinite(nominal) or nominal <= 0.0:
        raise ValueError("tidal first-step nominal radius must be finite and positive")
    first_step = min(radius_end - radius_start, nominal)
    if not math.isfinite(first_step) or first_step <= 0.0:
        raise ValueError("tidal first step must be finite and positive")
    return float(first_step)


def _assert_tidal_radius_on_segment(
    radius: float,
    segment: _BackgroundSegment,
) -> float:
    """Fail before dense-background evaluation when a radius leaves its segment."""
    requested = float(radius)
    if not math.isfinite(requested):
        raise ValueError("tidal RHS radius must be finite")
    radius_start, radius_end, allowance = _tidal_segment_bounds(segment)
    if requested < radius_start - allowance or requested > radius_end + allowance:
        raise ValueError(
            "tidal RHS radius is outside its background segment: "
            f"requested={requested!r}, bounds=({radius_start!r}, {radius_end!r}), "
            f"allowance={allowance!r}"
        )
    return requested


def _discontinuity_metadata_is_required(eos_callable: Callable) -> bool:
    """Return whether this EoS must use its declared discontinuity path."""
    try:
        surface_density = float(getattr(eos_callable, "eps_surf", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("eps_surf metadata must be finite and nonnegative") from exc
    if not math.isfinite(surface_density) or surface_density < 0.0:
        raise ValueError("eps_surf metadata must be finite and nonnegative")
    return bool(
        getattr(eos_callable, "requires_discontinuity_metadata", False)
    ) or surface_density > 0.0


def _resolved_discontinuities(eos_callable: Callable) -> tuple[EosDiscontinuity, ...]:
    required = _discontinuity_metadata_is_required(eos_callable)
    surface_density = float(getattr(eos_callable, "eps_surf", 0.0))
    declared = getattr(eos_callable, "discontinuities", None)
    if declared is None:
        if required:
            raise ValueError("required EoS discontinuity metadata are absent")
        return ()
    resolved = validate_discontinuity_sequence(declared)
    if required and not resolved:
        raise ValueError("required EoS discontinuity metadata are empty")
    surfaces = [item for item in resolved if item.kind == "surface"]
    if surface_density > 0.0:
        if len(surfaces) != 1:
            raise ValueError("a finite eps_surf requires one declared bare surface")
        if not math.isclose(
            surfaces[0].inner_energy_density,
            surface_density,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("declared surface density disagrees with eps_surf")
    elif surfaces:
        raise ValueError("a declared bare surface requires positive eps_surf metadata")
    return resolved


def _segment_pressure_floor(
    *,
    lower_discontinuity: EosDiscontinuity | None,
    settings: TovConfig,
) -> float:
    """Use the physical vacuum boundary on an explicit bare-surface segment."""
    if lower_discontinuity is not None and lower_discontinuity.kind == "surface":
        return 0.0
    return float(settings.pressure_min_safe)


def _applicable_discontinuities(
    discontinuities: tuple[EosDiscontinuity, ...],
    central_pressure: float,
) -> tuple[tuple[EosDiscontinuity, ...], tuple[str, ...]]:
    applicable = []
    skipped = []
    for item in discontinuities:
        if item.kind == "surface" or item.pressure < central_pressure:
            applicable.append(item)
        else:
            skipped.append(item.identifier)
    return tuple(applicable), tuple(skipped)


def _branch_pressure(
    pressure: float,
    *,
    upper_discontinuity: EosDiscontinuity | None,
    lower_discontinuity: EosDiscontinuity | None,
    settings: TovConfig,
) -> float:
    pressure_floor = _segment_pressure_floor(
        lower_discontinuity=lower_discontinuity,
        settings=settings,
    )
    candidate = max(float(pressure), pressure_floor)
    if upper_discontinuity is not None and candidate >= upper_discontinuity.pressure:
        candidate = float(np.nextafter(upper_discontinuity.pressure, -math.inf))
    if (
        lower_discontinuity is not None
        and lower_discontinuity.kind == "internal"
        and candidate <= lower_discontinuity.pressure
    ):
        candidate = float(np.nextafter(lower_discontinuity.pressure, math.inf))
    return max(candidate, pressure_floor)


def _evaluate_branch(
    eos_callable: Callable,
    pressure: float,
    *,
    upper_discontinuity: EosDiscontinuity | None,
    lower_discontinuity: EosDiscontinuity | None,
    settings: TovConfig,
) -> tuple[float, float]:
    evaluation_pressure = _branch_pressure(
        pressure,
        upper_discontinuity=upper_discontinuity,
        lower_discontinuity=lower_discontinuity,
        settings=settings,
    )
    epsilon, cs2_local = eos_callable(evaluation_pressure)
    epsilon = float(epsilon)
    cs2_local = float(cs2_local)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("EoS branch returned nonfinite or nonpositive energy density")
    if not math.isfinite(cs2_local):
        raise ValueError("EoS branch returned nonfinite sound speed")
    if cs2_local <= 0.0:
        raise ValueError("EoS branch returned nonpositive sound speed")
    return epsilon, cs2_local


def _background_rhs(
    radius: float,
    state: np.ndarray,
    eos_callable: Callable,
    *,
    upper_discontinuity: EosDiscontinuity | None,
    lower_discontinuity: EosDiscontinuity | None,
    settings: TovConfig,
) -> list[float]:
    radius = max(float(radius), 1.0e-10)
    mass, pressure = map(float, state)
    epsilon, _cs2 = _evaluate_branch(
        eos_callable,
        pressure,
        upper_discontinuity=upper_discontinuity,
        lower_discontinuity=lower_discontinuity,
        settings=settings,
    )
    pressure_safe = max(
        pressure,
        _segment_pressure_floor(
            lower_discontinuity=lower_discontinuity,
            settings=settings,
        ),
    )
    dm_dr = radius**2 * epsilon * _G_CONV
    if radius <= settings.center_radius_limit:
        dpressure_dr = (
            -_A_CONV
            * _G_CONV
            * (epsilon + pressure_safe)
            * (epsilon / 3.0 + pressure_safe)
            * radius
        )
        return [dm_dr, dpressure_dr]
    denominator = radius * (radius - 2.0 * mass * _A_CONV)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("background TOV state reached the Schwarzschild-radius boundary")
    dpressure_dr = (
        -_A_CONV
        * (epsilon + pressure_safe)
        * (mass + radius**3 * pressure_safe * _G_CONV)
        / denominator
    )
    if not math.isfinite(dm_dr) or not math.isfinite(dpressure_dr):
        raise ValueError("background TOV derivative is nonfinite")
    return [dm_dr, dpressure_dr]


def _pressure_event(target_pressure: float):
    def event(_radius: float, state: np.ndarray) -> float:
        return float(state[1] - target_pressure)

    event.terminal = True
    event.direction = -1
    return event


def _integrate_background(
    eos_callable: Callable,
    central_pressure: float,
    central_energy_density: float,
    discontinuities: tuple[EosDiscontinuity, ...],
    *,
    settings: TovConfig,
    rtol: float,
    atol: float,
    dense_output: bool = True,
) -> tuple[list[_BackgroundSegment], tuple[str, ...]]:
    applicable, skipped = _applicable_discontinuities(discontinuities, central_pressure)
    internal = [item for item in applicable if item.kind == "internal"]
    surface = next((item for item in applicable if item.kind == "surface"), None)
    targets: list[EosDiscontinuity | None] = [*internal, surface]

    radius_start = float(settings.radius_min_km)
    mass_start = radius_start**3 * central_energy_density * (_G_CONV / 3.0)
    state_start = np.asarray([mass_start, central_pressure], dtype=float)
    upper: EosDiscontinuity | None = None
    segments: list[_BackgroundSegment] = []
    for lower in targets:
        target_pressure = settings.surface_pressure_cutoff if lower is None else lower.pressure
        event = _pressure_event(target_pressure)

        def rhs(radius: float, state: np.ndarray) -> list[float]:
            return _background_rhs(
                radius,
                state,
                eos_callable,
                upper_discontinuity=upper,
                lower_discontinuity=lower,
                settings=settings,
            )

        solution = solve_ivp(
            rhs,
            (radius_start, settings.radius_max_km),
            state_start,
            events=event,
            method="RK45",
            dense_output=dense_output,
            rtol=rtol,
            atol=atol,
        )
        event_count = len(solution.t_events[0]) if solution.t_events else 0
        if solution.status != 1 or event_count != 1:
            identifier = "surface" if lower is None else lower.identifier
            raise RuntimeError(
                f"background event {identifier!r} was not reached exactly once "
                f"(status={solution.status}, count={event_count})"
            )
        solution.y_events[0][0][1] = target_pressure
        segment = _BackgroundSegment(solution, upper, lower)
        event_state = segment.event_state
        if not np.all(np.isfinite(event_state)):
            raise ValueError("background event state is nonfinite")
        if segments and segment.radius_start < segments[-1].radius_end:
            raise ValueError("background events are out of radial order")
        segments.append(segment)
        radius_start = segment.radius_end
        state_start = event_state.copy()
        state_start[1] = target_pressure
        upper = lower if lower is not None and lower.kind == "internal" else upper
    return segments, skipped


def _profile_from_segments(
    segments: list[_BackgroundSegment],
    *,
    points: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    radius_start = segments[0].radius_start
    radius_end = segments[-1].radius_end
    radii = np.linspace(radius_start, radius_end, int(points))
    masses = np.empty_like(radii)
    segment_index = 0
    for index, radius in enumerate(radii):
        while (
            segment_index < len(segments) - 1
            and radius > segments[segment_index].radius_end
        ):
            segment_index += 1
        masses[index] = float(segments[segment_index].solution.sol(radius)[0])
    return tuple(float(value) for value in radii), tuple(float(value) for value in masses)


def _validate_declared_branch_values(
    eos_callable: Callable,
    discontinuities: tuple[EosDiscontinuity, ...],
) -> None:
    for item in discontinuities:
        if item.kind == "internal":
            inner_pressure = float(np.nextafter(item.pressure, math.inf))
            outer_pressure = float(np.nextafter(item.pressure, -math.inf))
            evaluated_inner = float(eos_callable(inner_pressure)[0])
            evaluated_outer = float(eos_callable(outer_pressure)[0])
        else:
            evaluated_inner = float(eos_callable(0.0)[0])
            evaluated_outer = 0.0
        if not math.isfinite(evaluated_inner) or not math.isfinite(evaluated_outer):
            raise ValueError(f"declared jump {item.identifier} has nonfinite evaluated sides")
        for side, declared, evaluated in (
            ("inner", item.inner_energy_density, evaluated_inner),
            ("outer", item.outer_energy_density, evaluated_outer),
        ):
            if not math.isclose(declared, evaluated, rel_tol=1.0e-8, abs_tol=1.0e-10):
                raise ValueError(
                    f"declared jump {item.identifier} {side} density disagrees with EoS branch: "
                    f"declared={declared!r}, evaluated={evaluated!r}"
                )


def _tidal_rhs_on_segment(
    radius: float,
    y_state: np.ndarray,
    segment: _BackgroundSegment,
    eos_callable: Callable,
    settings: TovConfig,
) -> list[float]:
    radius = _assert_tidal_radius_on_segment(radius, segment)
    mass, pressure = map(float, segment.solution.sol(radius))
    epsilon, cs2_local = _evaluate_branch(
        eos_callable,
        pressure,
        upper_discontinuity=segment.upper_discontinuity,
        lower_discontinuity=segment.lower_discontinuity,
        settings=settings,
    )
    pressure_safe = max(
        pressure,
        _segment_pressure_floor(
            lower_discontinuity=segment.lower_discontinuity,
            settings=settings,
        ),
    )
    y_tidal = float(y_state[0])
    if radius <= settings.center_radius_limit:
        derivative = taylor_expansion(
            radius,
            pressure_safe,
            epsilon,
            cs2_local,
            _G_CONV,
            _A_CONV,
        )[2]
    else:
        derivative = _tov_equations_with_limit(
            radius,
            mass,
            pressure_safe,
            y_tidal,
            epsilon,
            cs2_local,
            _G_CONV,
            _A_CONV,
            0.0,
        )[2]
    if not math.isfinite(derivative):
        raise ValueError("tidal derivative is nonfinite")
    return [float(derivative)]


def _integrate_tidal(
    eos_callable: Callable,
    central_pressure: float,
    segments: list[_BackgroundSegment],
    discontinuities: tuple[EosDiscontinuity, ...],
    skipped_ids: tuple[str, ...],
    *,
    settings: TovConfig,
    rtol: float,
    atol: float,
) -> TovLambdaDiagnostic:
    applicable, expected_skipped = _applicable_discontinuities(
        discontinuities, central_pressure
    )
    if expected_skipped != skipped_ids:
        raise ValueError("background and tidal phase selection disagree")
    expected = len(applicable)
    y_value = 2.0
    applied: list[AppliedTidalJump] = []
    y_surface_interior = None
    for segment in segments:
        first_step = _bounded_tidal_first_step(segment, settings)
        solution = solve_ivp(
            lambda radius, state: _tidal_rhs_on_segment(
                radius, state, segment, eos_callable, settings
            ),
            (segment.radius_start, segment.radius_end),
            [y_value],
            method="RK45",
            rtol=rtol,
            atol=atol,
            first_step=first_step,
        )
        if not solution.success or len(solution.t) == 0:
            raise RuntimeError(f"tidal segment integration failed: {solution.message}")
        y_value = float(solution.y[0, -1])
        if not math.isfinite(y_value):
            raise ValueError("tidal segment endpoint is nonfinite")
        lower = segment.lower_discontinuity
        if lower is None:
            y_surface_interior = y_value
            continue
        event_mass = float(segment.event_state[0])
        event_radius = segment.radius_end
        if lower.kind == "surface":
            y_surface_interior = y_value
        delta_y, denominator = tidal_jump_delta_y(
            radius_km=event_radius,
            mass_msun=event_mass,
            pressure_mev_fm3=lower.pressure,
            delta_energy_density_mev_fm3=lower.delta_energy_density,
        )
        if lower.kind == "surface" and delta_y >= 0.0:
            raise ValueError("a bare-surface tidal correction must be negative")
        y_after = y_value + delta_y
        if not math.isfinite(y_after):
            raise ValueError("post-jump tidal state is nonfinite")
        applied.append(
            AppliedTidalJump(
                identifier=lower.identifier,
                kind=lower.kind,
                pressure=lower.pressure,
                radius=event_radius,
                mass=event_mass,
                inner_energy_density=lower.inner_energy_density,
                outer_energy_density=lower.outer_energy_density,
                delta_energy_density=lower.delta_energy_density,
                correction_denominator=denominator,
                y_before=y_value,
                delta_y=delta_y,
                y_after=y_after,
                provenance=lower.provenance,
            )
        )
        y_value = y_after

    if y_surface_interior is None:
        raise ValueError("surface interior tidal value is unavailable")
    if len(applied) != expected:
        raise ValueError(
            f"expected {expected} discontinuity corrections but applied {len(applied)}"
        )
    expected_identifiers = tuple(item.identifier for item in applicable)
    applied_identifiers = tuple(item.identifier for item in applied)
    if applied_identifiers != expected_identifiers:
        raise ValueError(
            "applied discontinuity identities do not match the required sequence: "
            f"expected={expected_identifiers!r}, applied={applied_identifiers!r}"
        )
    final_state = segments[-1].event_state
    mass = float(final_state[0])
    radius = segments[-1].radius_end
    surface_pressure = float(final_state[1])
    explicit_bare_surface = any(item.kind == "surface" for item in applicable)
    expected_surface_pressure = 0.0 if explicit_bare_surface else settings.surface_pressure_cutoff
    surface_tolerance = max(atol, abs(expected_surface_pressure) * 1.0e-6, 1.0e-14)
    if not math.isfinite(surface_pressure) or not math.isclose(
        surface_pressure,
        expected_surface_pressure,
        rel_tol=0.0,
        abs_tol=surface_tolerance,
    ):
        raise ValueError(
            "surface event was not localized at the declared policy pressure: "
            f"expected={expected_surface_pressure!r}, observed={surface_pressure!r}"
        )
    compactness = mass * _A_CONV / radius
    algebra = tidal_algebra(compactness, y_value)
    return TovLambdaDiagnostic(
        central_pressure=central_pressure,
        mass=mass,
        radius=radius,
        compactness=compactness,
        expected_jump_count=expected,
        applied_jumps=tuple(applied),
        skipped_discontinuity_ids=skipped_ids,
        surface_event_pressure=surface_pressure,
        y_surface_interior=y_surface_interior,
        y_surface_vacuum=y_value,
        y_supplied_to_k2=y_value,
        k2=algebra.k2,
        lambda_dimensionless=algebra.lambda_dimensionless,
        scientific_status=LAMBDA_SCIENTIFIC_STATUS,
    )


def _failed_lambda_diagnostic(
    *,
    central_pressure: float,
    mass: float,
    radius: float,
    surface_pressure: float,
    expected_jump_count: int,
    skipped_ids: tuple[str, ...],
    reason: str,
) -> TovLambdaDiagnostic:
    return TovLambdaDiagnostic(
        central_pressure=central_pressure,
        mass=mass,
        radius=radius,
        compactness=mass * _A_CONV / radius,
        expected_jump_count=expected_jump_count,
        applied_jumps=(),
        skipped_discontinuity_ids=skipped_ids,
        surface_event_pressure=surface_pressure,
        y_surface_interior=None,
        y_surface_vacuum=None,
        y_supplied_to_k2=None,
        k2=None,
        lambda_dimensionless=None,
        scientific_status="failed_closed",
        failure_reason=reason,
    )


def _background_only_lambda_diagnostic(
    *,
    central_pressure: float,
    mass: float,
    radius: float,
    surface_pressure: float,
    skipped_ids: tuple[str, ...],
) -> TovLambdaDiagnostic:
    """Record that no tidal ODE was requested for a valid background star."""

    return TovLambdaDiagnostic(
        central_pressure=central_pressure,
        mass=mass,
        radius=radius,
        compactness=mass * _A_CONV / radius,
        expected_jump_count=0,
        applied_jumps=(),
        skipped_discontinuity_ids=skipped_ids,
        surface_event_pressure=surface_pressure,
        y_surface_interior=None,
        y_surface_vacuum=None,
        y_supplied_to_k2=None,
        k2=None,
        lambda_dimensionless=None,
        scientific_status=TIDAL_NOT_REQUESTED_STATUS,
        failure_reason=None,
    )


def solve_star(
    eos_callable: Callable,
    central_pressure: float,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    settings: TovConfig | None = None,
    calculate_tidal: bool = True,
    retain_profile: bool = True,
) -> TovStarResult:
    """Solve one star, preserving a valid background when only Lambda fails.

    ``retain_profile=False`` is an exact-work optimization for internal
    searches that consume only the surface and central state.  It does not
    change the ODE, tolerances, events, or returned scalar observables.  The
    default remains ``True`` for public compatibility.

    Required discontinuity metadata and its segmented background are part of
    background validity.  Their failure therefore returns no star result.
    """
    resolved = _DEFAULT_TOV if settings is None else settings
    pc = _require_finite("central_pressure", central_pressure)
    if pc <= 0.0:
        raise ValueError("central_pressure must be positive")
    effective_rtol = resolved.ode_rtol if rtol is None else _require_finite("rtol", rtol)
    effective_atol = resolved.ode_atol if atol is None else _require_finite("atol", atol)
    if effective_rtol <= 0.0 or effective_atol <= 0.0:
        raise ValueError("TOV tolerances must be positive")
    if not isinstance(calculate_tidal, bool):
        raise ValueError("calculate_tidal must be boolean")
    if not isinstance(retain_profile, bool):
        raise ValueError("retain_profile must be boolean")

    metadata_required = _discontinuity_metadata_is_required(eos_callable)
    metadata_error = None
    try:
        discontinuities = _resolved_discontinuities(eos_callable)
    except (TypeError, ValueError) as exc:
        if metadata_required:
            raise ValueError(f"required_discontinuity_metadata:{exc}") from exc
        discontinuities = ()
        metadata_error = f"metadata_validation:{exc}"

    if metadata_required:
        try:
            _validate_declared_branch_values(eos_callable, discontinuities)
        except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
            raise ValueError(f"required_discontinuity_metadata:{exc}") from exc

    eps_init, cs2_init = map(float, eos_callable(pc))
    if not math.isfinite(eps_init) or eps_init <= 0.0:
        raise ValueError("initial energy density must be finite and positive")
    if not math.isfinite(cs2_init):
        raise ValueError("initial sound speed must be finite")

    try:
        segments, skipped = _integrate_background(
            eos_callable,
            pc,
            eps_init,
            discontinuities,
            settings=resolved,
            rtol=effective_rtol,
            atol=effective_atol,
            dense_output=bool(calculate_tidal or retain_profile),
        )
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        if metadata_required:
            raise RuntimeError(f"segmented_background:{exc}") from exc
        if not discontinuities:
            raise
        metadata_error = f"segmented_background:{exc}"
        segments, skipped = _integrate_background(
            eos_callable,
            pc,
            eps_init,
            (),
            settings=resolved,
            rtol=effective_rtol,
            atol=effective_atol,
            dense_output=bool(calculate_tidal or retain_profile),
        )

    final_state = segments[-1].event_state
    mass = float(final_state[0])
    radius = segments[-1].radius_end
    surface_pressure = float(final_state[1])
    if not math.isfinite(mass) or not math.isfinite(radius) or mass <= 0.0 or radius <= 0.0:
        raise ValueError("background surface mass/radius is invalid")
    if (
        any(item.kind == "surface" for item in discontinuities)
        and surface_pressure != 0.0
    ):
        raise ValueError("a declared bare surface must terminate exactly at P=0")
    if retain_profile:
        radius_profile, mass_profile = _profile_from_segments(
            segments, points=resolved.dense_profile_points
        )
    else:
        radius_profile, mass_profile = (), ()

    if not calculate_tidal:
        _applicable, skipped_for_record = _applicable_discontinuities(
            discontinuities, pc
        )
        lambda_diagnostic = _background_only_lambda_diagnostic(
            central_pressure=pc,
            mass=mass,
            radius=radius,
            surface_pressure=surface_pressure,
            skipped_ids=skipped_for_record,
        )
    elif metadata_error is None:
        try:
            if not metadata_required:
                _validate_declared_branch_values(eos_callable, discontinuities)
            lambda_diagnostic = _integrate_tidal(
                eos_callable,
                pc,
                segments,
                discontinuities,
                skipped,
                settings=resolved,
                rtol=effective_rtol,
                atol=effective_atol,
            )
        except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
            applicable, skipped = _applicable_discontinuities(discontinuities, pc)
            lambda_diagnostic = _failed_lambda_diagnostic(
                central_pressure=pc,
                mass=mass,
                radius=radius,
                surface_pressure=surface_pressure,
                expected_jump_count=len(applicable),
                skipped_ids=skipped,
                reason=f"tidal_integration:{exc}",
            )
    else:
        lambda_diagnostic = _failed_lambda_diagnostic(
            central_pressure=pc,
            mass=mass,
            radius=radius,
            surface_pressure=surface_pressure,
            expected_jump_count=0,
            skipped_ids=(),
            reason=metadata_error,
        )

    return TovStarResult(
        central_pressure=pc,
        central_energy_density=eps_init,
        central_sound_speed_squared=cs2_init,
        mass=mass,
        radius=radius,
        lambda_dimensionless=lambda_diagnostic.lambda_dimensionless,
        surface_energy_density=float(getattr(eos_callable, "eps_surf", 0.0)),
        radius_profile=radius_profile,
        mass_profile=mass_profile,
        lambda_diagnostic=lambda_diagnostic,
    )

_PUBLIC_MODULE = "eos_generation.stellar.tov"
for _compatibility_object in (
    _BackgroundSegment,
    _tidal_segment_bounds,
    _bounded_tidal_first_step,
    _assert_tidal_radius_on_segment,
    _discontinuity_metadata_is_required,
    _resolved_discontinuities,
    _segment_pressure_floor,
    _applicable_discontinuities,
    _branch_pressure,
    _evaluate_branch,
    _background_rhs,
    _pressure_event,
    _integrate_background,
    _profile_from_segments,
    _validate_declared_branch_values,
    _tidal_rhs_on_segment,
    _integrate_tidal,
    _failed_lambda_diagnostic,
    _background_only_lambda_diagnostic,
    solve_star,
):
    _compatibility_object.__module__ = _PUBLIC_MODULE
del _compatibility_object

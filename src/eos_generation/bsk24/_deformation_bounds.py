"""Continuous-extremum helpers for physical deformation bounds."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar


def _log_windowed_gaussian_shape_scalar(
    epsilon_mev_fm3: float,
    *,
    epsilon0_mev_fm3: float,
    sigma_mev_fm3: float,
    delta_mev_fm3: float,
    epsilon_match_mev_fm3: float,
) -> float:
    """Return log(G W), with ``-inf`` representing the exact ``f=0`` set."""

    epsilon = float(epsilon_mev_fm3)
    if epsilon <= epsilon_match_mev_fm3:
        return -math.inf
    ramp_end = epsilon_match_mev_fm3 + delta_mev_fm3
    if epsilon < ramp_end:
        x = (epsilon - epsilon_match_mev_fm3) / delta_mev_fm3
        window = x**3 * (10.0 + x * (-15.0 + 6.0 * x))
        if window <= 0.0:
            return -math.inf
        log_window = math.log(window)
    else:
        log_window = 0.0
    z = (epsilon - epsilon0_mev_fm3) / sigma_mev_fm3
    return -0.5 * z * z + log_window


def _continuous_local_minima(
    grid: np.ndarray,
    sampled_values: np.ndarray,
    function,
    *,
    mandatory_points: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    """Discover sampled basins and refine each one with bounded minimization."""

    if len(grid) < 3 or len(grid) != len(sampled_values):
        raise ValueError("continuous-extremum discovery requires matching grids")
    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(sampled_values)):
        raise ValueError("continuous-extremum discovery values must be finite")
    indices = np.flatnonzero(
        (sampled_values[1:-1] <= sampled_values[:-2])
        & (sampled_values[1:-1] <= sampled_values[2:])
    ) + 1
    candidates: list[tuple[float, float]] = [
        (float(sampled_values[0]), float(grid[0])),
        (float(sampled_values[-1]), float(grid[-1])),
    ]
    for index in indices:
        result = minimize_scalar(
            function,
            bounds=(float(grid[index - 1]), float(grid[index + 1])),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        if not result.success or not math.isfinite(float(result.fun)):
            raise ValueError("bounded continuous-extremum refinement failed")
        candidates.append((float(result.fun), float(result.x)))
    for point in mandatory_points:
        if float(grid[0]) <= point <= float(grid[-1]):
            value = float(function(point))
            if math.isfinite(value):
                candidates.append((value, float(point)))
    candidates.sort(key=lambda item: (item[1], item[0]))
    deduplicated: list[tuple[float, float]] = []
    for value, location in candidates:
        if deduplicated and math.isclose(
            location,
            deduplicated[-1][1],
            rel_tol=0.0,
            abs_tol=2.0e-9,
        ):
            if value < deduplicated[-1][0]:
                deduplicated[-1] = (value, location)
        else:
            deduplicated.append((value, location))
    return tuple(deduplicated)

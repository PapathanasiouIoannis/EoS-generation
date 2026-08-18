"""Pure TOV differential equations and tidal algebra."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from typing import Callable

import numba
import numpy as np

from eos_generation._internal.config import TovConfig
from eos_generation.stellar._tov_types import (
    DENOMINATOR_FLOOR,
    TidalAlgebraResult,
    _A_CONV,
    _DEFAULT_TOV,
    _G_CONV,
    _TOV_SINGULARITY_LIMIT,
)

@numba.njit
def taylor_expansion(
    r: float,
    P_safe: float,
    epsilon: float,
    cs2_local: float,
    G_CONV: float,
    A_CONV: float,
) -> list:
    dm_dr = (r**2) * epsilon * G_CONV
    dP_dr = -A_CONV * G_CONV * (epsilon + P_safe) * (epsilon / 3.0 + P_safe) * r
    dy_dr = (
        -(2.0 / 7.0)
        * A_CONV
        * G_CONV
        * r
        * (11.0 * P_safe + epsilon / 3.0 + (epsilon + P_safe) / cs2_local)
    )
    return [dm_dr, dP_dr, dy_dr]


@numba.njit
def _tov_equations_with_limit(
    r: float,
    m: float,
    P_safe: float,
    y_tidal: float,
    epsilon: float,
    cs2_local: float,
    G_CONV: float,
    A_CONV: float,
    singularity_limit: float,
) -> list:
    term_1 = epsilon + P_safe
    term_2 = m + (r**3 * P_safe * G_CONV)
    term_3 = r * (r - 2.0 * m * A_CONV)

    if abs(term_3) < singularity_limit:
        return [0.0, 0.0, 0.0]

    dP_dr = -A_CONV * (term_1 * term_2) / term_3
    dm_dr = (r**2) * epsilon * G_CONV

    exp_lambda = 1.0 / (1.0 - 2.0 * A_CONV * m / r)
    F = (1.0 - A_CONV * G_CONV * (r**2) * (epsilon - P_safe)) * exp_lambda
    matter_source = (
        A_CONV
        * G_CONV
        * (5.0 * epsilon + 9.0 * P_safe + (epsilon + P_safe) / cs2_local)
        * (r**2)
    )
    pressure_mass_source = A_CONV * (m + (r**3 * P_safe * G_CONV))
    nu_prime_squared_source = 4.0 * (
        pressure_mass_source / (r * (1.0 - 2.0 * A_CONV * m / r))
    ) ** 2
    Q = (matter_source - 6.0) * exp_lambda - nu_prime_squared_source
    dy_dr = -(y_tidal**2 + y_tidal * F + Q) / r
    return [dm_dr, dP_dr, dy_dr]


@numba.njit
def tov_equations(
    r: float,
    m: float,
    P_safe: float,
    y_tidal: float,
    epsilon: float,
    cs2_local: float,
    G_CONV: float,
    A_CONV: float,
) -> list:
    return _tov_equations_with_limit(
        r,
        m,
        P_safe,
        y_tidal,
        epsilon,
        cs2_local,
        G_CONV,
        A_CONV,
        _TOV_SINGULARITY_LIMIT,
    )


def tov_rhs(
    r: float,
    y_state: list,
    eos_callable: Callable,
    settings: TovConfig | None = None,
) -> list:
    """Return historical TOV/tidal derivatives for one radius and state."""
    resolved = _DEFAULT_TOV if settings is None else settings
    r = max(r, 1.0e-10)
    m, pressure, y_tidal = y_state
    if pressure < resolved.surface_pressure_cutoff:
        return [0.0, 0.0, 0.0]

    pressure_safe = max(pressure, resolved.pressure_min_safe)
    epsilon, cs2_local = eos_callable(pressure_safe)
    if epsilon <= 0:
        return [0.0, 0.0, 0.0]
    if cs2_local < 1.0e-10:
        cs2_local = 1.0e-10
    if r <= resolved.center_radius_limit:
        return taylor_expansion(r, pressure_safe, epsilon, cs2_local, _G_CONV, _A_CONV)
    if resolved.singularity_limit == _TOV_SINGULARITY_LIMIT:
        return tov_equations(r, m, pressure_safe, y_tidal, epsilon, cs2_local, _G_CONV, _A_CONV)
    return _tov_equations_with_limit(
        r,
        m,
        pressure_safe,
        y_tidal,
        epsilon,
        cs2_local,
        _G_CONV,
        _A_CONV,
        resolved.singularity_limit,
    )


def surface_event(_radius, state, *args):
    """Return pressure minus the configured historical surface cutoff."""
    settings = args[-1] if args and isinstance(args[-1], TovConfig) else _DEFAULT_TOV
    return state[1] - settings.surface_pressure_cutoff


surface_event.terminal = True
surface_event.direction = -1


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _love_number_k2_decimal(compactness: float, y_r: float) -> float:
    """Evaluate the corrected Hinderer expression with decimal guard digits."""
    with localcontext() as context:
        context.prec = 100
        c = Decimal(str(compactness))
        y = Decimal(str(y_r))
        one = Decimal(1)
        two = Decimal(2)
        three = Decimal(3)
        four = Decimal(4)
        five = Decimal(5)
        eight = Decimal(8)
        factor = one - two * c
        numerator = (
            (eight / five)
            * factor**2
            * c**5
            * (two * c * (y - one) - y + two)
        )
        denominator = (
            two * c * (Decimal(6) - three * y + three * c * (five * y - eight))
            + four
            * c**3
            * (
                Decimal(13)
                - Decimal(11) * y
                + c * (three * y - two)
                + two * c**2 * (one + y)
            )
            + three
            * factor**2
            * (two - y + two * c * (y - one))
            * factor.ln()
        )
        if denominator == 0:
            raise ValueError("tidal Love-number denominator is zero")
        value = numerator / denominator
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("tidal Love number is nonfinite")
    return result


def love_number_k2(
    compactness: float,
    y_r: float,
    *,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> float:
    """Return the corrected Hinderer dimensionless tidal Love number ``k2``.

    Postnikov et al. explicitly identify severe cancellation for ``C < 0.1``.
    That range is evaluated with standard-library decimal guard digits rather
    than imposing an unsupported compactness floor.
    """
    c = _require_finite("compactness", compactness)
    y = _require_finite("y_r", y_r)
    floor = abs(_require_finite("denominator_floor", denominator_floor))
    if c <= 0.0 or c >= 0.5:
        raise ValueError("compactness must satisfy 0 < C < 0.5 for the log term")
    if c < 0.1:
        return _love_number_k2_decimal(c, y)

    numerator = (
        (8.0 / 5.0)
        * (1.0 - 2.0 * c) ** 2
        * c**5
        * (2.0 * c * (y - 1.0) - y + 2.0)
    )
    denominator_term1 = 2.0 * c * (6.0 - 3.0 * y + 3.0 * c * (5.0 * y - 8.0))
    denominator_term2 = (
        4.0
        * c**3
        * (13.0 - 11.0 * y + c * (3.0 * y - 2.0) + 2.0 * c**2 * (1.0 + y))
    )
    denominator_term3 = (
        3.0
        * (1.0 - 2.0 * c) ** 2
        * (2.0 - y + 2.0 * c * (y - 1.0))
        * np.log1p(-2.0 * c)
    )
    denominator = denominator_term1 + denominator_term2 + denominator_term3
    if abs(denominator) < floor:
        raise ValueError("tidal Love-number denominator is too close to zero")
    return float(numerator / denominator)


def dimensionless_lambda(compactness: float, k2: float) -> float:
    """Return historical dimensionless tidal deformability from ``C`` and ``k2``."""
    c = _require_finite("compactness", compactness)
    k2_value = _require_finite("k2", k2)
    if c <= 0.0:
        raise ValueError("compactness must be positive")
    return float((2.0 / 3.0) * k2_value * c**-5)


def tidal_algebra(
    compactness: float,
    y_r: float,
    *,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> TidalAlgebraResult:
    """Return historical ``k2`` and ``Lambda`` for one surface state."""
    k2_value = love_number_k2(compactness, y_r, denominator_floor=denominator_floor)
    return TidalAlgebraResult(
        compactness=float(compactness),
        y_r=float(y_r),
        k2=k2_value,
        lambda_dimensionless=dimensionless_lambda(compactness, k2_value),
    )


def tidal_jump_delta_y(
    *,
    radius_km: float,
    mass_msun: float,
    pressure_mev_fm3: float,
    delta_energy_density_mev_fm3: float,
) -> tuple[float, float]:
    """Return ``(delta_y, denominator)`` in repository solver units.

    The geometric matching condition of Takatsy and Kovacs (2020), Eq. 11,
    reduces here because ``dm/dr = G_CONV * epsilon * r**2`` and mass is in
    solar masses. The finite-pressure term is retained for internal joins.
    """
    radius = _require_finite("radius_km", radius_km)
    mass = _require_finite("mass_msun", mass_msun)
    pressure = _require_finite("pressure_mev_fm3", pressure_mev_fm3)
    delta_epsilon = _require_finite(
        "delta_energy_density_mev_fm3", delta_energy_density_mev_fm3
    )
    if radius <= 0.0 or mass <= 0.0 or pressure < 0.0:
        raise ValueError("jump radius/mass must be positive and pressure nonnegative")
    pressure_mass = _G_CONV * radius**3 * pressure
    denominator = mass + pressure_mass
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("tidal jump correction denominator must be finite and positive")
    delta_y = -(_G_CONV * radius**3 * delta_epsilon) / denominator
    if not math.isfinite(delta_y):
        raise ValueError("tidal jump correction must be finite")
    return float(delta_y), float(denominator)

_PUBLIC_MODULE = "eos_generation.stellar.tov"
for _compatibility_function in (
    taylor_expansion,
    _tov_equations_with_limit,
    tov_equations,
    tov_rhs,
    surface_event,
    _require_finite,
    _love_number_k2_decimal,
    love_number_k2,
    dimensionless_lambda,
    tidal_algebra,
    tidal_jump_delta_y,
):
    _compatibility_function.__module__ = _PUBLIC_MODULE
del _compatibility_function

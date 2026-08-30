"""Explicit opt-in independent CFL stellar check; not collected by pytest.

No eos_generation imports or generated reference fixtures. Thermodynamics
are transcribed from Lugones & Horvath, PRD 66, 074017 (2002), Eqs. 2-6:
https://doi.org/10.1103/PhysRevD.66.074017
The massive free-gas integral is evaluated by QUADPACK, not the production
closed primitive. TOV and tidal equations are transcribed from Postnikov,
Prakash & Lattimer, PRD 82, 024016 (2010), Eqs. 2, 6, 9-11:
https://arxiv.org/abs/1004.5098
We integrate in h=ln(mu/mu_surface), with x=r^2 and z=m/r, using DOP853;
production integrates in radius using RK45 and its own EoS inversion.
The vacuum surface jump is -4*pi*R^3*epsilon_surface/M (geometric units).

This checks implementation independence, not independent authorship, QCD
truth, or agreement with a convention-matched published CFL star table.
Run only after explicit scientific-work authorization. The command reads a
packet and prints JSON; it never writes or modifies that packet.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import math
from pathlib import Path

from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar


# Deliberately transcribed from the approved contract, not imported from the
# implementation under test. Geometric energy densities below are km^-2.
MS = 100.0
GAP = 100.0
HC = 197.3269804
BAG = 57.5 * HC**3
MU_SURFACE = 249.31780807778472
MU_MAX = 600.0
SOLAR_LENGTH = 1.4766
GEOMETRIC_DENSITY = SOLAR_LENGTH * 1.124e-5 / (4.0 * math.pi)


def _raw_pressure(mu: float) -> float:
    nu = 2.0 * mu - math.sqrt(mu * mu + MS * MS / 3.0)
    strange_energy = quad(lambda p: math.sqrt(p * p + MS * MS) * p * p, 0.0, nu, epsabs=1e-6, epsrel=1e-12)[0]
    free_energy = (1.5 * nu**4 + 3.0 * strange_energy) / math.pi**2
    return (3.0 * mu * nu**3 / math.pi**2 + 3.0 * GAP**2 * mu**2 / math.pi**2 - BAG - free_energy) / HC**3


@lru_cache(maxsize=1)
def _surface_offset() -> float:
    return _raw_pressure(MU_SURFACE)


def thermodynamics(mu: float) -> tuple[float, float, float]:
    if not MU_SURFACE <= mu <= MU_MAX:
        raise ValueError("reference chemical potential outside approved domain")
    nu = 2.0 * mu - math.sqrt(mu * mu + MS * MS / 3.0)
    nu_prime = 2.0 - mu / math.sqrt(mu * mu + MS * MS / 3.0)
    density = (nu**3 + 2.0 * GAP**2 * mu) / (math.pi**2 * HC**3)
    pressure = _raw_pressure(mu) - _surface_offset()
    epsilon = 3.0 * mu * density - pressure
    cs2 = (nu**3 + 2.0 * GAP**2 * mu) / (mu * (3.0 * nu**2 * nu_prime + 2.0 * GAP**2))
    return pressure, epsilon, cs2


def _love_number(c: float, y: float) -> float:
    # Independently transcribed corrected algebra, with guard digits for the
    # low-compactness cancellation; not the production tidal helper.
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 60
        b, y = Decimal(str(c)), Decimal(str(y))
        t = 2 - y + 2 * b * (y - 1)
        d = 2 * b * (6 - 3 * y + 3 * b * (5 * y - 8))
        d += 4 * b**3 * (13 - 11 * y + b * (3 * y - 2) + 2 * b**2 * (1 + y))
        d += 3 * (1 - 2 * b)**2 * t * (1 - 2 * b).ln()
        return float(Decimal(8) / 5 * b**5 * (1 - 2 * b)**2 * t / d)


def star(mu_c: float, *, refined: bool = False) -> dict[str, float]:
    pc, ec, cs = thermodynamics(mu_c)
    if pc <= 0.0:
        raise ValueError("reference central pressure must be positive")
    hc = math.log(mu_c / MU_SURFACE)
    step = min(1e-9 if refined else 1e-8, hc / 100.0)
    p0, e0 = pc * GEOMETRIC_DENSITY, ec * GEOMETRIC_DENSITY
    x0 = 3.0 * step / (2.0 * math.pi * (e0 + 3.0 * p0))
    z0 = 4.0 * math.pi * e0 * x0 / 3.0
    # Regular y(0)=2; varying the finite center offset independently checks
    # the omitted O(r^2) starting correction.
    state0 = [x0, z0, 2.0]

    def rhs(h, state):
        pressure, epsilon, cs2 = thermodynamics(MU_SURFACE * math.exp(h))
        p, e = pressure * GEOMETRIC_DENSITY, epsilon * GEOMETRIC_DENSITY
        x, z, y = state
        if x <= 0.0 or z <= 0.0 or z >= 0.5:
            raise ValueError("invalid reference interior state")
        f = (1.0 - 4.0 * math.pi * x * (e - p)) / (1.0 - 2.0 * z)
        q = (4.0 * math.pi * x * (5.0 * e + 9.0 * p + (e + p) / cs2) - 6.0) / (1.0 - 2.0 * z)
        q -= 4.0 * ((z + 4.0 * math.pi * x * p) / (1.0 - 2.0 * z))**2
        scale = (1.0 - 2.0 * z) / (z + 4.0 * math.pi * x * p)
        return [-2.0 * x * scale, -(4.0 * math.pi * x * e - z) * scale, (y * y + y * f + q) * scale]

    solution = solve_ivp(rhs, (hc - step, 0.0), state0, method="DOP853",
                         rtol=2e-12 if refined else 2e-10, atol=2e-14 if refined else 2e-12)
    if not solution.success or solution.t[-1] != 0.0:
        raise RuntimeError("reference surface not reached")
    x, compactness, y_inner = solution.y[:, -1]
    radius = math.sqrt(x)
    mass_length = compactness * radius
    epsilon_surface = thermodynamics(MU_SURFACE)[1]
    jump = -4.0 * math.pi * GEOMETRIC_DENSITY * epsilon_surface * x / compactness
    k2 = _love_number(compactness, y_inner + jump)
    return dict(mass_msun=mass_length / SOLAR_LENGTH, radius_km=radius, k2=k2,
                lambda_dimensionless=2.0 * k2 / (3.0 * compactness**5), central_pressure_mev_fm3=pc,
                mu_c_mev=mu_c, surface_delta_y=jump)


def fixed_mass(target: float = 1.4, *, refined: bool = False) -> dict[str, float]:
    # Frozen B=57.5, m_s=100, Delta=100 baseline: the stable-branch 1.4-M_sun
    # crossing lies between 280 and 300 MeV.  Keep the bracket explicit so a
    # future refreeze fails closed instead of silently searching a new branch.
    lower, upper = 280.0, 300.0
    f = lambda mu: star(mu, refined=refined)["mass_msun"] - target
    if not f(lower) < 0.0 < f(upper):
        raise ValueError("reference target has no declared stable-branch bracket")
    return star(brentq(f, lower, upper, xtol=1e-10), refined=refined)


def maximum_mass() -> dict[str, float]:
    result = minimize_scalar(lambda mu: -star(mu, refined=True)["mass_msun"], bounds=(350.0, 600.0),
                             method="bounded", options={"xatol": 1e-6})
    if not result.success or not 350.01 < result.x < 599.99:
        raise RuntimeError("reference maximum not interior")
    peak = star(result.x, refined=True)
    if not all(star(result.x + offset, refined=True)["mass_msun"] < peak["mass_msun"] for offset in (-0.01, 0.01)):
        raise RuntimeError("reference maximum lacks positive/negative local slopes")
    return peak


def compare_packet(packet: Path, stage: str) -> dict:
    config = json.loads((packet / "complete_configuration.json").read_text(encoding="utf-8"))
    if config.get("baseline_parameter_set_sha256") != "3991cb8615d2d29617ccb90c6dc54b23aae64bcc752856d07f17f99abc048307":
        raise ValueError("reference does not match packet's frozen CFL constants")
    with (packet / "fixed_mass_observables.csv").open(newline="", encoding="utf-8") as stream:
        matches = [r for r in csv.DictReader(stream) if r["case_id"] == "direct" and r["stage"] == stage and float(r["target_mass_msun"]) == 1.4]
    if len(matches) != 1 or matches[0]["status"] != "bracketed_and_solved" or matches[0]["tidal_status"] != "validated_lambda_validation_v1":
        raise ValueError("packet has no validated direct baseline at the requested stage/mass")
    coarse, reference = fixed_mass(), fixed_mass(refined=True)
    fields = ("radius_km", "k2", "lambda_dimensionless")
    convergence = {key: abs(coarse[key] / reference[key] - 1.0) for key in fields}
    if max(convergence.values()) > 1e-7:
        raise AssertionError(f"independent center/ODE refinement failed: {convergence}")
    residuals = {key: abs(float(matches[0][key]) / reference[key] - 1.0) for key in fields}
    if max(residuals.values()) > 1e-5:
        raise AssertionError(f"production/reference mismatch exceeds declared 1e-5: {residuals}")
    peak = maximum_mass()
    with (packet / "maximum_mass_screening.csv").open(newline="", encoding="utf-8") as stream:
        maxima = [r for r in csv.DictReader(stream) if r["case_id"] == "direct" and r["stage"] == stage]
    if len(maxima) != 1 or maxima[0]["maximum_mass_resolved"].lower() != "true":
        raise ValueError("packet has no resolved direct maximum at the requested stage")
    peak_residual = abs(float(maxima[0]["maximum_mass_msun"]) / peak["mass_msun"] - 1.0)
    if peak_residual > 1e-5:
        raise AssertionError(f"maximum mass mismatch exceeds 1e-5: {peak_residual}")
    return dict(status="pass", source="independent_h_enthalpy_DOP853_QUADPACK_reference_v1", stage=stage,
                fixed_mass_reference=reference, reference_refinement_relative_errors=convergence,
                production_relative_errors=residuals, maximum_mass_reference=peak, maximum_mass_relative_error=peak_residual,
                comparison_relative_tolerance=1e-5, reference_refinement_tolerance=1e-7,
                packet_writes=0, production_solver_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--execute", action="store_true", help="authorize about 30 independent reference stars")
    args = parser.parse_args()
    if not args.execute:
        parser.error("independent stellar work requires --execute after reviewing its cost")
    print(json.dumps(compare_packet(args.packet, args.stage), indent=2, allow_nan=False))

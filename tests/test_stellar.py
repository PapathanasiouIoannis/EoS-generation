from __future__ import annotations

import math
from dataclasses import replace
import unittest

from eos_generation._internal.config import DEFAULT_CONFIG
from eos_generation.stellar.discontinuities import (
    EosDiscontinuity,
    validate_discontinuity_sequence,
)
from eos_generation.stellar.tov import (
    love_number_k2,
    solve_star,
    tidal_jump_delta_y,
)


def continuous_eos(pressure: float) -> tuple[float, float]:
    p = max(float(pressure), 1.0e-300)
    return math.sqrt(p / 1.0e-3), 2.0 * math.sqrt(1.0e-3 * p)


continuous_eos.eps_surf = 0.0


class StellarContracts(unittest.TestCase):
    def test_newtonian_n1_polytrope_weak_field_limit(self) -> None:
        k_value = 1.0e-3
        b_value = 1.0e-3
        settings = replace(DEFAULT_CONFIG.tov, radius_max_km=100.0)
        star = solve_star(
            continuous_eos,
            b_value**2 / k_value,
            rtol=1.0e-11,
            atol=1.0e-13,
            settings=settings,
        )
        length = DEFAULT_CONFIG.units.solar_mass_length_km
        gravity = DEFAULT_CONFIG.units.gravity_conversion
        expected_radius = math.pi * math.sqrt(
            2.0 * k_value / (length * gravity)
        )
        expected_mass = (
            math.pi
            * gravity
            * star.central_energy_density
            * (2.0 * k_value / (length * gravity)) ** 1.5
        )
        self.assertLess(abs(star.radius / expected_radius - 1.0), 0.01)
        self.assertLess(abs(star.mass / expected_mass - 1.0), 0.01)

    def test_binnington_poisson_published_n1_rows(self) -> None:
        # Binnington & Poisson, arXiv:0906.1366, n=1 and K=1 table.
        published = (
            (0.0488888889, 0.1586178173, 0.14594601117),
            (0.1358024691, 0.3264977638, 0.068656738911),
            (0.1955555556, 0.3971100356, 0.046672564713),
        )
        k_value = 1.0e-3
        settings = replace(DEFAULT_CONFIG.tov, radius_max_km=100.0)
        for b_value, expected_two_c, expected_k2 in published:
            with self.subTest(b=b_value):
                star = solve_star(
                    continuous_eos,
                    b_value**2 / k_value,
                    rtol=1.0e-11,
                    atol=1.0e-13,
                    settings=settings,
                )
                observed_two_c = 2.0 * star.lambda_diagnostic.compactness
                self.assertAlmostEqual(observed_two_c, expected_two_c, delta=2.0e-6)
                self.assertAlmostEqual(
                    star.lambda_diagnostic.k2, expected_k2, delta=2.0e-6
                )

    def test_discontinuities_are_ordered_and_immutable(self) -> None:
        inner = EosDiscontinuity.from_sides(
            identifier="inner",
            kind="internal",
            pressure=80.0,
            inner_energy_density=420.0,
            outer_energy_density=360.0,
            provenance="synthetic contract",
        )
        outer = EosDiscontinuity.from_sides(
            identifier="outer",
            kind="internal",
            pressure=30.0,
            inner_energy_density=260.0,
            outer_energy_density=210.0,
            provenance="synthetic contract",
        )
        self.assertEqual((inner, outer), validate_discontinuity_sequence((inner, outer)))
        with self.assertRaises(ValueError):
            validate_discontinuity_sequence((outer, inner))

    def test_tidal_jump_uses_the_exact_finite_pressure_algebra(self) -> None:
        radius = 10.0
        mass = 1.4
        pressure = 30.0
        delta_epsilon = 50.0
        observed, denominator = tidal_jump_delta_y(
            radius_km=radius,
            mass_msun=mass,
            pressure_mev_fm3=pressure,
            delta_energy_density_mev_fm3=delta_epsilon,
        )
        gravity = DEFAULT_CONFIG.units.gravity_conversion
        expected_denominator = mass + gravity * radius**3 * pressure
        expected = -(gravity * radius**3 * delta_epsilon) / expected_denominator
        self.assertEqual(expected_denominator, denominator)
        self.assertEqual(expected, observed)

    def test_postnikov_uniform_density_surface_jump_is_exactly_three(self) -> None:
        radius = 10.0
        epsilon_surface = 200.0
        gravity = DEFAULT_CONFIG.units.gravity_conversion
        mass = gravity * epsilon_surface * radius**3 / 3.0
        delta_y, _ = tidal_jump_delta_y(
            radius_km=radius,
            mass_msun=mass,
            pressure_mev_fm3=0.0,
            delta_energy_density_mev_fm3=epsilon_surface,
        )
        self.assertAlmostEqual(-delta_y, 3.0, places=12)
        self.assertAlmostEqual(2.0 + delta_y, -1.0, places=12)

    def test_continuous_background_and_tides_converge(self) -> None:
        coarse = solve_star(continuous_eos, 100.0, rtol=1.0e-7, atol=1.0e-9)
        medium = solve_star(continuous_eos, 100.0, rtol=1.0e-9, atol=1.0e-11)
        fine = solve_star(continuous_eos, 100.0, rtol=1.0e-11, atol=1.0e-13)
        for getter in (
            lambda star: star.mass,
            lambda star: star.radius,
            lambda star: star.lambda_dimensionless,
        ):
            self.assertLess(
                abs(getter(medium) - getter(fine)),
                abs(getter(coarse) - getter(fine)),
            )

    def test_explicit_finite_surface_cutoff_is_reported(self) -> None:
        settings = replace(DEFAULT_CONFIG.tov, surface_pressure_cutoff=1.0e-10)
        star = solve_star(
            continuous_eos,
            100.0,
            rtol=1.0e-10,
            atol=1.0e-12,
            settings=settings,
        )
        self.assertAlmostEqual(
            settings.surface_pressure_cutoff,
            star.lambda_diagnostic.surface_event_pressure,
            delta=1.0e-14,
        )
        self.assertEqual(0, star.lambda_diagnostic.applied_jump_count)

    def test_love_number_is_finite_on_a_representative_state(self) -> None:
        value = love_number_k2(0.16, 0.8)
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 0.0)


if __name__ == "__main__":
    unittest.main()

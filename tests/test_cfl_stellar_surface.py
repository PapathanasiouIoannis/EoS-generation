from __future__ import annotations

from dataclasses import replace
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from eos_generation._internal.config import DEFAULT_CONFIG
from eos_generation.stellar import _tov_integration as tov_integration
from eos_generation.stellar import tov as tov_core
from eos_generation.stellar._tov_maximum import _local_refinement_pressures
from eos_generation.stellar.diagnostics import pressure_profile_from_solved_star
from eos_generation.stellar.discontinuities import EosDiscontinuity
from eos_generation.stellar.discontinuities import BARE_SELF_BOUND_SEQUENCE_POLICY
from eos_generation.stellar.discontinuities import SEED_PRESERVING_LOCAL_REFINEMENT_POLICY
from eos_generation.stellar.tov import (
    LAMBDA_FRAMEWORK_CAPABILITY,
    _build_sequence_evidence,
    _sampled_mass_secants,
    refine_maximum_mass_from_sequence,
    solve_star,
    solve_sequence,
    tidal_jump_delta_y,
)


class _SyntheticSelfBoundEos:
    """Minimal CSS-like EoS used only to exercise the bare-surface contract."""

    sound_speed_squared = 1.0 / 3.0

    def __init__(
        self,
        *,
        surface_energy_density: float = 200.0,
        branch_surface_energy_density: float | None = None,
        requires_metadata: bool = True,
        declare_surface: bool = True,
    ) -> None:
        self.eps_surf = float(surface_energy_density)
        self.requires_discontinuity_metadata = bool(requires_metadata)
        self.branch_surface_energy_density = float(
            self.eps_surf
            if branch_surface_energy_density is None
            else branch_surface_energy_density
        )
        self.evaluated_pressures: list[float] = []
        if declare_surface:
            self.discontinuities = (
                EosDiscontinuity.from_sides(
                    identifier="synthetic_bare_surface",
                    kind="surface",
                    pressure=0.0,
                    inner_energy_density=self.eps_surf,
                    outer_energy_density=0.0,
                    provenance="synthetic_css_surface_v1",
                ),
            )

    def __call__(self, pressure: float) -> tuple[float, float]:
        resolved_pressure = float(pressure)
        self.evaluated_pressures.append(resolved_pressure)
        if resolved_pressure < 0.0:
            raise ValueError("synthetic self-bound EoS is undefined below P=0")
        return (
            self.branch_surface_energy_density
            + resolved_pressure / self.sound_speed_squared,
            self.sound_speed_squared,
        )


class BareSelfBoundStellarContracts(unittest.TestCase):
    def test_css_density_rescaling_preserves_dimensionless_tides(self) -> None:
        # For P=s(epsilon-epsilon_s), multiplying epsilon_s and P_c by a
        # factor a in the TOV/tidal equations gives r,m -> (r,m)/sqrt(a).
        # Compactness, y (including its surface jump), k2, and Lambda are
        # invariant. This is an analytic scaling law, not a stored fixture.
        stars = [
            solve_star(
                _SyntheticSelfBoundEos(surface_energy_density=200.0 * factor),
                100.0 * factor, rtol=1e-10, atol=1e-12, retain_profile=False,
            )
            for factor in (1.0, 4.0)
        ]
        first, scaled = stars
        ratios = (
            2.0 * scaled.mass / first.mass,
            2.0 * scaled.radius / first.radius,
            scaled.lambda_dimensionless / first.lambda_dimensionless,
            scaled.lambda_diagnostic.k2 / first.lambda_diagnostic.k2,
            scaled.lambda_diagnostic.applied_jumps[0].delta_y
            / first.lambda_diagnostic.applied_jumps[0].delta_y,
        )
        # 1e-7 is a predeclared dimensionless comparison bound, with margin
        # over the ODE tolerances and finite-radius center initialization.
        for ratio in ratios:
            self.assertAlmostEqual(1.0, ratio, delta=1e-7)
        self.assertTrue(all(s.lambda_diagnostic.applied_jump_count == 1 for s in stars))

    def test_local_refinement_reuses_original_pressure_nodes_exactly(self) -> None:
        lower, upper = 400.0, 480.0
        # A distinct binary64 representation of the intended log midpoint:
        # no computed fixture or production EoS is used in this regression.
        old = np.geomspace(lower, upper, 17)
        middle = float(np.nextafter(old[8], np.inf))
        policy = SEED_PRESERVING_LOCAL_REFINEMENT_POLICY
        grid = _local_refinement_pressures(lower, middle, upper, 17, policy)
        self.assertEqual(17, len(grid))
        self.assertEqual((lower, middle, upper), tuple(grid[[0, 8, 16]]))
        self.assertTrue(np.all(np.diff(grid) > 0.0))
        self.assertNotIn(old[8], grid)
        np.testing.assert_array_equal(
            old, _local_refinement_pressures(lower, middle, upper, 17, None)
        )
        with self.assertRaisesRegex(ValueError, "unknown"):
            _local_refinement_pressures(lower, middle, upper, 17, "unknown")

        peak_pressure = middle * 1.01

        def analytic_star(pressure: float) -> SimpleNamespace:
            return SimpleNamespace(
                mass=2.0 - math.log(pressure / peak_pressure)**2,
                radius=10.0, central_energy_density=pressure + 200.0,
                central_sound_speed_squared=1.0 / 3.0,
            )

        seeds = (lower, middle, upper)
        rows = tuple(
            (analytic_star(p).mass, 10.0, math.nan, p, p + 200.0, 1.0 / 3.0, 200.0)
            for p in seeds
        )
        profiles = tuple(((), ()) for _ in rows)
        evidence = _build_sequence_evidence(
            full_sequence=rows, stable_sequence=rows[:2],
            full_dense_profiles=profiles, stable_dense_profiles=profiles[:2],
            full_tidal_diagnostics=None, stable_tidal_diagnostics=None,
            full_lambda_diagnostics=None, stable_lambda_diagnostics=None,
            attempted_central_pressures=list(seeds), failed_central_pressures=[],
            sampled_peak_index=1, sampled_secants=_sampled_mass_secants(rows),
            eos_endpoint_pressure=upper, max_mass_stable=rows[1][0],
        )
        calls: list[float] = []

        def solver(_eos, pressure, **_kwargs):
            # A second evaluation at an already solved (or one-ULP-near)
            # seed would introduce independent ODE noise into a zero-width
            # intended interval. It is forbidden, not averaged or repaired.
            self.assertTrue(all(abs(pressure - p) > 8 * math.ulp(p) for p in seeds))
            calls.append(pressure)
            return analytic_star(pressure)

        resolved = refine_maximum_mass_from_sequence(
            SimpleNamespace(stellar_local_refinement_policy=policy), evidence,
            local_points=17, refinement_pressure_rtol=1e-10, star_solver=solver,
        )
        self.assertTrue(calls)
        self.assertTrue(resolved.maximum_mass_resolved)
        self.assertAlmostEqual(2.0, resolved.maximum_mass_msun, places=10)
        self.assertAlmostEqual(peak_pressure, resolved.central_pressure_mev_fm3, delta=1e-5)
        self.assertGreater(resolved.positive_left_secant, 0.0)
        self.assertLess(resolved.negative_right_secant, 0.0)

    def test_low_mass_self_bound_sequence_has_no_hadronic_size_cut(self) -> None:
        eos = _SyntheticSelfBoundEos()
        eos.stellar_sequence_policy = BARE_SELF_BOUND_SEQUENCE_POLICY
        # P_c/epsilon_s <= 2e-5 puts the finite-pressure corrections well
        # below the 0.2% Newtonian-limit comparison, without relaxing it.
        settings = replace(DEFAULT_CONFIG.tov, grid_pressure_min_log=0.002, sequence_points=3, dense_profile_points=8)
        evidence = solve_sequence(
            eos, p_max_causal=0.004, settings=settings, rtol=1e-9, atol=1e-11,
            return_tidal_diagnostics=True, return_sequence_evidence=True,
        )
        self.assertEqual(3, len(evidence.full_sequence))
        self.assertFalse(evidence.failed_central_pressures)
        for row, diagnostic in zip(evidence.full_sequence, evidence.full_lambda_diagnostics):
            mass, radius = row[:2]
            self.assertLess(mass, 0.05)
            self.assertLess(radius, 3.0)
            # Independent Newtonian uniform-density limit, not a generated
            # fixture: dm/dr = G_CONV epsilon_s r^2 and k2 -> 3/4.
            uniform_mass = DEFAULT_CONFIG.units.gravity_conversion * eos.eps_surf * radius**3 / 3.0
            self.assertLess(abs(mass / uniform_mass - 1.0), 2e-3)
            self.assertLess(abs(diagnostic.k2 / 0.75 - 1.0), 2e-3)
            self.assertEqual(1, diagnostic.applied_jump_count)

        del eos.stellar_sequence_policy
        legacy = solve_sequence(eos, p_max_causal=0.004, settings=settings, calculate_tidal=False, return_sequence_evidence=True)
        self.assertFalse(legacy.full_sequence)
        self.assertEqual({"minimum_mass_or_radius_cutoff"}, {f.category for f in legacy.failed_central_pressures})

    def test_self_bound_sequence_policy_requires_surface_metadata(self) -> None:
        eos = _SyntheticSelfBoundEos(declare_surface=False)
        eos.stellar_sequence_policy = BARE_SELF_BOUND_SEQUENCE_POLICY
        with self.assertRaisesRegex(ValueError, "metadata are absent"):
            solve_sequence(eos, settings=replace(DEFAULT_CONFIG.tov, sequence_points=3))

    def test_segmented_tidal_branch_never_clips_small_positive_sound_speed(self) -> None:
        expected_cs2 = 5.0e-12

        epsilon, observed_cs2 = tov_integration._evaluate_branch(
            lambda _pressure: (250.0, expected_cs2),
            10.0,
            upper_discontinuity=None,
            lower_discontinuity=None,
            settings=DEFAULT_CONFIG.tov,
        )

        self.assertEqual(250.0, epsilon)
        self.assertEqual(expected_cs2, observed_cs2)
        with self.assertRaisesRegex(ValueError, "nonpositive sound speed"):
            tov_integration._evaluate_branch(
                lambda _pressure: (250.0, 0.0),
                10.0,
                upper_discontinuity=None,
                lower_discontinuity=None,
                settings=DEFAULT_CONFIG.tov,
            )

    def test_certified_background_branch_skips_unused_sound_speed(self) -> None:
        class CertifiedEos:
            _background_energy_only_is_certified = True

            def __init__(self) -> None:
                self.energy_calls = 0
                self.full_calls = 0

            def energy_density_from_pressure(self, pressure: float) -> float:
                self.energy_calls += 1
                return 250.0 + pressure

            def __call__(self, pressure: float) -> tuple[float, float]:
                self.full_calls += 1
                return 250.0 + pressure, 1.0 / 3.0

        eos = CertifiedEos()
        epsilon = tov_integration._evaluate_background_branch(
            eos,
            10.0,
            upper_discontinuity=None,
            lower_discontinuity=None,
            settings=DEFAULT_CONFIG.tov,
        )

        self.assertEqual(260.0, epsilon)
        self.assertEqual(1, eos.energy_calls)
        self.assertEqual(0, eos.full_calls)

    def test_uniform_density_surface_jump_is_exactly_minus_three(self) -> None:
        radius = 10.0
        epsilon_surface = 200.0
        gravity = DEFAULT_CONFIG.units.gravity_conversion
        mass = gravity * epsilon_surface * radius**3 / 3.0

        delta_y, denominator = tidal_jump_delta_y(
            radius_km=radius,
            mass_msun=mass,
            pressure_mev_fm3=0.0,
            delta_energy_density_mev_fm3=epsilon_surface,
        )

        self.assertEqual(mass, denominator)
        self.assertAlmostEqual(-3.0, delta_y, places=14)
        self.assertAlmostEqual(-1.0, 2.0 + delta_y, places=14)

    def test_bare_surface_is_zero_pressure_and_jump_is_applied_once(self) -> None:
        eos = _SyntheticSelfBoundEos()
        settings = replace(
            DEFAULT_CONFIG.tov,
            pressure_min_safe=1.0,
            surface_pressure_cutoff=2.0,
            dense_profile_points=32,
        )

        star = solve_star(
            eos,
            100.0,
            rtol=1.0e-9,
            atol=1.0e-11,
            settings=settings,
            retain_profile=False,
        )

        diagnostic = star.lambda_diagnostic
        self.assertEqual(LAMBDA_FRAMEWORK_CAPABILITY, diagnostic.scientific_status)
        self.assertEqual(0.0, diagnostic.surface_event_pressure)
        self.assertEqual(1, diagnostic.expected_jump_count)
        self.assertEqual(1, diagnostic.applied_jump_count)
        self.assertEqual(0.0, min(eos.evaluated_pressures))
        self.assertTrue(all(value >= 0.0 for value in eos.evaluated_pressures))

        jump = diagnostic.applied_jumps[0]
        self.assertEqual("surface", jump.kind)
        self.assertEqual(0.0, jump.pressure)
        self.assertEqual(eos.eps_surf, jump.inner_energy_density)
        self.assertEqual(0.0, jump.outer_energy_density)
        self.assertEqual(eos.eps_surf, jump.delta_energy_density)
        self.assertEqual(star.mass, jump.correction_denominator)
        self.assertLess(jump.delta_y, 0.0)
        expected_delta_y = -(
            DEFAULT_CONFIG.units.gravity_conversion
            * star.radius**3
            * eos.eps_surf
            / star.mass
        )
        self.assertEqual(expected_delta_y, jump.delta_y)
        self.assertEqual(jump.y_before, diagnostic.y_surface_interior)
        self.assertEqual(jump.y_before + jump.delta_y, jump.y_after)
        self.assertEqual(jump.y_after, diagnostic.y_surface_vacuum)
        self.assertEqual(jump.y_after, diagnostic.y_supplied_to_k2)
        self.assertTrue(math.isfinite(star.lambda_dimensionless))

        payload = diagnostic.to_dict()
        self.assertTrue(payload["calculation_lambda_validated"])
        self.assertEqual(1, payload["applied_jump_count"])
        self.assertEqual("surface", payload["applied_jumps"][0]["type"])
        self.assertIn("synthetic_css_surface_v1", json.dumps(payload, allow_nan=False))

    def test_required_metadata_failure_produces_no_star_observables(self) -> None:
        cases = (
            _SyntheticSelfBoundEos(
                surface_energy_density=200.0,
                requires_metadata=False,
                declare_surface=False,
            ),
            _SyntheticSelfBoundEos(
                surface_energy_density=0.0,
                requires_metadata=True,
                declare_surface=False,
            ),
        )
        for eos in cases:
            with self.subTest(
                eps_surf=eos.eps_surf,
                required=eos.requires_discontinuity_metadata,
            ), patch.object(
                tov_integration,
                "_integrate_background",
                side_effect=AssertionError("background integration must not start"),
            ) as background:
                with self.assertRaisesRegex(
                    ValueError,
                    "required_discontinuity_metadata:required EoS discontinuity metadata",
                ):
                    solve_star(
                        eos,
                        100.0,
                        calculate_tidal=False,
                        retain_profile=False,
                    )
                background.assert_not_called()

    def test_surface_branch_mismatch_fails_before_background_integration(self) -> None:
        eos = _SyntheticSelfBoundEos(branch_surface_energy_density=201.0)
        with patch.object(
            tov_integration,
            "_integrate_background",
            side_effect=AssertionError("background integration must not start"),
        ) as background:
            with self.assertRaisesRegex(
                ValueError,
                "required_discontinuity_metadata:declared jump synthetic_bare_surface",
            ):
                solve_star(
                    eos,
                    100.0,
                    calculate_tidal=False,
                    retain_profile=False,
                )
            background.assert_not_called()

    def test_required_segment_failure_has_no_unsegmented_fallback(self) -> None:
        eos = _SyntheticSelfBoundEos()
        with patch.object(
            tov_integration,
            "_integrate_background",
            side_effect=RuntimeError("synthetic segmented failure"),
        ) as background:
            with self.assertRaisesRegex(
                RuntimeError,
                "segmented_background:synthetic segmented failure",
            ):
                solve_star(
                    eos,
                    100.0,
                    calculate_tidal=False,
                    retain_profile=False,
                )
            self.assertEqual(1, background.call_count)
            self.assertEqual(eos.discontinuities, background.call_args.args[3])

    def test_required_profile_resampling_has_no_unsegmented_fallback(self) -> None:
        eos = _SyntheticSelfBoundEos()
        star = SimpleNamespace(
            central_pressure=100.0,
            central_energy_density=500.0,
            radius_profile=(1.0e-4, 1.0),
        )
        with patch.object(
            tov_core,
            "_integrate_background",
            side_effect=RuntimeError("synthetic profile segment failure"),
        ) as background:
            with self.assertRaisesRegex(
                RuntimeError,
                "synthetic profile segment failure",
            ):
                pressure_profile_from_solved_star(
                    eos,
                    star,
                    settings=DEFAULT_CONFIG.tov,
                    rtol=1.0e-9,
                    atol=1.0e-11,
                )
            self.assertEqual(1, background.call_count)


if __name__ == "__main__":
    unittest.main()

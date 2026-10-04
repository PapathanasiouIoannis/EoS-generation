from __future__ import annotations
from eos_generation.stellar import (
    _build_sequence_evidence,
    _sampled_mass_secants,
    refine_maximum_mass_from_sequence,
    resolve_maximum_mass,
    solve_sequence,
)

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from eos_generation import stellar as internal_stellar
from eos_generation.numerics import (
    BSk24TOVStage,
    BSk24ThermodynamicStage,
    BSk24TrialConfig,
)
from eos_generation import (
    deformation,
    thermodynamics as reconstruction,
    assessment,
    diagnostics,
)
from eos_generation.tov import (
    LAMBDA_FRAMEWORK_CAPABILITY,
)


class EffectiveReconstructionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = reconstruction.build_consistent_baseline(
            reconstruction.BSk24GridSettings(
                lower_points=1025,
                upper_points=2049,
            )
        )

    def test_accepted_nonzero_reconstruction_closes_cold_identities(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "accepted-nonzero",
            0.01,
            200.0,
            50.0,
            40.0,
        )
        raw_gate, _, _ = assessment.raw_local_physics_gate(
            self.baseline,
            proposal,
            dense_lower_points=257,
            dense_upper_points=1025,
        )
        self.assertEqual("accepted_raw_local_physics_gate", raw_gate["status"])
        self.assertFalse(raw_gate["full_retained_domain_passed"])

        eos = reconstruction.build_windowed_eos(
            self.baseline,
            proposal,
            raw_gate_report=raw_gate,
            require_full_domain=False,
        )
        self.assertNotEqual(0.0, eos.deformation.amplitude)
        self.assertEqual(
            "accepted_selected_domain_thermodynamic_gate",
            eos.diagnostics["retained_domain_thermodynamic_admissibility"]["status"],
        )

        pressure_from_euler = eos.baryon_density * eos.chemical_potential - eos.epsilon
        mu_from_euler = (eos.epsilon + eos.pressure) / eos.baryon_density
        np.testing.assert_allclose(
            eos.pressure,
            pressure_from_euler,
            rtol=2.0e-15,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            eos.chemical_potential,
            mu_from_euler,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            eos.residuals["r_p_algebraic"],
            eos.pressure - pressure_from_euler,
        )
        np.testing.assert_array_equal(
            eos.residuals["r_mu_algebraic"],
            eos.chemical_potential - mu_from_euler,
        )

        # This is the independent PCHIP derivative check
        # mu_B * dn_B/d(epsilon) - 1 on a compact governed grid.
        first_law = np.abs(eos.residuals["first_law_normalized"])
        self.assertTrue(np.all(np.isfinite(first_law)))
        self.assertLess(float(np.max(first_law)), 2.0e-4)
        self.assertLess(float(np.percentile(first_law, 99.0)), 5.0e-6)


class StellarDecisionContracts(unittest.TestCase):
    @staticmethod
    def _fixed_mass_star(pressure: float) -> SimpleNamespace:
        mass = 0.8 + 0.04 * float(pressure)
        tidal = SimpleNamespace(
            k2=0.09,
            lambda_dimensionless=350.0,
            scientific_status=LAMBDA_FRAMEWORK_CAPABILITY,
            failure_reason=None,
        )
        return SimpleNamespace(
            mass=mass,
            radius=12.0 - 0.01 * float(pressure),
            central_energy_density=100.0 + float(pressure),
            central_sound_speed_squared=0.5,
            lambda_diagnostic=tidal,
        )

    def test_fixed_mass_requires_a_true_stable_prefix_bracket(self) -> None:
        evidence = SimpleNamespace(
            stable_sequence=(
                (1.2, 12.0, 400.0, 10.0, 110.0, 0.5, 0.0),
                (1.6, 11.8, 300.0, 20.0, 120.0, 0.5, 0.0),
            )
        )
        config = BSk24TrialConfig(amplitudes=(0.0,))
        stage = BSk24TOVStage("synthetic", 5, 1.0e-8, 1.0e-10, 3)
        eos = SimpleNamespace(pressure_min_mev_fm3=1.0e-9)

        solver_calls: list[tuple[float, bool]] = []

        def fake_solve_star(
            _eos: object,
            pressure: float,
            **kwargs: object,
        ) -> SimpleNamespace:
            solver_calls.append((float(pressure), bool(kwargs.get("calculate_tidal"))))
            return self._fixed_mass_star(float(pressure))

        with patch.object(
            internal_stellar,
            "solve_star",
            side_effect=fake_solve_star,
        ):
            solved, star = internal_stellar._fixed_mass_result(
                eos,
                evidence,
                1.4,
                config,
                stage,
            )

        self.assertEqual("bracketed_and_solved", solved["status"])
        self.assertEqual([10.0, 20.0], solved["bracket_pressure_mev_fm3"])
        self.assertAlmostEqual(15.0, solved["central_pressure_mev_fm3"])
        self.assertAlmostEqual(0.0, solved["mass_residual_msun"], places=12)
        self.assertIsNotNone(star)
        self.assertTrue(solver_calls)
        self.assertTrue(all(10.0 <= pressure <= 20.0 for pressure, _ in solver_calls))
        self.assertTrue(solver_calls[-1][1])

        with patch.object(internal_stellar, "solve_star") as forbidden_solver:
            unavailable, missing_star = internal_stellar._fixed_mass_result(
                eos,
                evidence,
                1.9,
                config,
                stage,
            )
        self.assertEqual("unavailable_not_bracketed", unavailable["status"])
        self.assertEqual(
            "target mass is outside the successful stable prefix",
            unavailable["reason"],
        )
        self.assertIsNone(missing_star)
        forbidden_solver.assert_not_called()

    def test_fixed_mass_rejects_a_bracket_beyond_the_retained_endpoint(self) -> None:
        evidence = SimpleNamespace(
            stable_sequence=(
                (1.2, 12.0, 400.0, 10.0, 110.0, 0.5, 0.0),
                (1.6, 11.8, 300.0, 20.0, 120.0, 0.5, 0.0),
            )
        )
        config = BSk24TrialConfig(amplitudes=(0.0,))
        stage = BSk24TOVStage("synthetic", 5, 1.0e-8, 1.0e-10, 3)
        eos = SimpleNamespace(
            pressure_min_mev_fm3=1.0e-9,
            pressure_max_mev_fm3=15.0,
        )
        with patch.object(internal_stellar, "solve_star") as forbidden_solver:
            unavailable, star = internal_stellar._fixed_mass_result(
                eos,
                evidence,
                1.4,
                config,
                stage,
            )
        self.assertEqual(
            "unavailable_outside_retained_eos_domain",
            unavailable["status"],
        )
        self.assertIn("retained EoS pressure endpoint", unavailable["reason"])
        self.assertIsNone(star)
        forbidden_solver.assert_not_called()

    def test_sequence_endpoint_below_pressure_floor_never_calls_solver(self) -> None:
        config = BSk24TrialConfig(amplitudes=(0.0,))
        stage = BSk24TOVStage("synthetic", 5, 1.0e-8, 1.0e-10, 3)
        eos = SimpleNamespace(pressure_min_mev_fm3=1.0e-9)
        settings = internal_stellar._tov_settings(eos, config, stage)
        with patch(
            "eos_generation.stellar.solve_star",
            side_effect=AssertionError("solver must not run"),
        ) as forbidden_solver:
            evidence = solve_sequence(
                object(),
                p_max_causal=1.0,
                settings=settings,
                return_sequence_evidence=True,
            )
        self.assertEqual((), evidence.full_sequence)
        self.assertEqual((1.0,), evidence.attempted_central_pressures)
        self.assertEqual(
            "eos_endpoint_below_sequence_floor",
            evidence.failed_central_pressures[0].category,
        )
        forbidden_solver.assert_not_called()

    def test_turning_point_is_refined_but_sampled_endpoint_peak_is_not_mmax(
        self,
    ) -> None:
        def parabolic_mass_solver(
            _eos: object,
            pressure: float,
            **_kwargs: object,
        ) -> SimpleNamespace:
            log_offset = math.log(float(pressure) / 10.0)
            return SimpleNamespace(
                mass=2.1 - 0.05 * log_offset**2,
                radius=12.0,
                central_energy_density=100.0 + float(pressure),
                central_sound_speed_squared=0.5,
            )

        resolved = resolve_maximum_mass(
            object(),
            pressure_min_mev_fm3=1.0,
            pressure_max_mev_fm3=100.0,
            initial_points=9,
            refinement_pressure_rtol=1.0e-10,
            star_solver=parabolic_mass_solver,
        )
        self.assertTrue(resolved.maximum_mass_resolved)
        self.assertEqual("resolved_unique_turning_point", resolved.status)
        self.assertAlmostEqual(2.1, resolved.maximum_mass_msun, places=10)
        self.assertAlmostEqual(
            10.0,
            resolved.central_pressure_mev_fm3,
            delta=1.0e-6,
        )
        self.assertGreater(resolved.positive_left_secant, 0.0)
        self.assertLess(resolved.negative_right_secant, 0.0)

        turning_rows = (
            (1.8, 12.2, math.nan, 1.0, 101.0, 0.5, 0.0),
            (2.1, 12.0, math.nan, 10.0, 110.0, 0.5, 0.0),
            (1.9, 11.8, math.nan, 100.0, 200.0, 0.5, 0.0),
        )
        turning_profiles = tuple(((), ()) for _ in turning_rows)
        turning_evidence = _build_sequence_evidence(
            full_sequence=turning_rows,
            stable_sequence=turning_rows[:2],
            full_dense_profiles=turning_profiles,
            stable_dense_profiles=turning_profiles[:2],
            full_tidal_diagnostics=None,
            stable_tidal_diagnostics=None,
            full_lambda_diagnostics=None,
            stable_lambda_diagnostics=None,
            attempted_central_pressures=[1.0, 10.0, 100.0],
            failed_central_pressures=[],
            sampled_peak_index=1,
            sampled_secants=_sampled_mass_secants(turning_rows),
            eos_endpoint_pressure=100.0,
            max_mass_stable=2.1,
        )
        displaced_optimizer = SimpleNamespace(
            x=math.log(10.001),
            success=True,
            nfev=1,
        )
        with patch(
            "eos_generation.stellar.minimize_scalar",
            return_value=displaced_optimizer,
        ):
            retained_sample = refine_maximum_mass_from_sequence(
                object(),
                turning_evidence,
                refinement_pressure_rtol=1.0e-10,
                star_solver=parabolic_mass_solver,
            )
        self.assertTrue(retained_sample.maximum_mass_resolved)
        self.assertEqual(2.1, retained_sample.maximum_mass_msun)
        self.assertEqual(10.0, retained_sample.central_pressure_mev_fm3)
        self.assertEqual(
            retained_sample.maximum_mass_msun,
            max(row[1] for row in retained_sample.stable_branch_models),
        )

        sampled_rows = (
            (1.0, 12.0, math.nan, 1.0, 101.0, 0.5, 0.0),
            (1.5, 11.8, math.nan, 10.0, 110.0, 0.5, 0.0),
            (1.8, 11.5, math.nan, 100.0, 200.0, 0.5, 0.0),
        )
        sampled_profiles = tuple(((), ()) for _ in sampled_rows)
        sampled_evidence = _build_sequence_evidence(
            full_sequence=sampled_rows,
            stable_sequence=sampled_rows,
            full_dense_profiles=sampled_profiles,
            stable_dense_profiles=sampled_profiles,
            full_tidal_diagnostics=None,
            stable_tidal_diagnostics=None,
            full_lambda_diagnostics=None,
            stable_lambda_diagnostics=None,
            attempted_central_pressures=[1.0, 10.0, 100.0],
            failed_central_pressures=[],
            sampled_peak_index=2,
            sampled_secants=_sampled_mass_secants(sampled_rows),
            eos_endpoint_pressure=100.0,
            max_mass_stable=1.8,
        )
        forbidden_solver = Mock(side_effect=AssertionError("solver must not run"))
        unresolved = refine_maximum_mass_from_sequence(
            object(),
            sampled_evidence,
            star_solver=forbidden_solver,
        )
        self.assertFalse(unresolved.maximum_mass_resolved)
        self.assertEqual(
            "unresolved_no_turning_point_before_eos_endpoint",
            unresolved.status,
        )
        self.assertIsNone(unresolved.maximum_mass_msun)
        self.assertIsNone(unresolved.passes_maximum_mass_threshold)
        self.assertIsNone(unresolved.to_dict()["passes_maximum_mass_threshold"])
        self.assertEqual(1.8, max(row[1] for row in unresolved.sampled_models))
        self.assertFalse(unresolved.to_dict()["sampled_argmax_is_maximum_mass"])
        forbidden_solver.assert_not_called()

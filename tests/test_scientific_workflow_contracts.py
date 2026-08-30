from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from eos_generation._internal import lifecycle as internal_lifecycle
from eos_generation._internal import stellar as internal_stellar
from eos_generation._internal import summary as internal_summary
from eos_generation._internal.execution import RunCallbacks, run_bsk24_trial
from eos_generation._internal.planning import (
    BSk24TOVStage,
    BSk24ThermodynamicStage,
    BSk24TrialConfig,
)
from eos_generation.bsk24 import deformation, reconstruction
from eos_generation.stellar.tov import (
    LAMBDA_FRAMEWORK_CAPABILITY,
    _build_sequence_evidence,
    _sampled_mass_secants,
    refine_maximum_mass_from_sequence,
    resolve_maximum_mass,
    solve_sequence,
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
        raw_gate, _, _ = deformation.raw_local_physics_gate(
            self.baseline,
            proposal,
            dense_lower_points=257,
            dense_upper_points=1025,
        )
        self.assertEqual("accepted_raw_local_physics_gate", raw_gate["status"])
        self.assertTrue(raw_gate["full_retained_domain_passed"])

        eos = deformation.build_windowed_eos(
            self.baseline,
            proposal,
            raw_gate_report=raw_gate,
            require_full_domain=True,
        )
        self.assertNotEqual(0.0, eos.deformation.amplitude)
        self.assertEqual(
            "accepted_full_domain_thermodynamic_gate",
            eos.diagnostics["full_domain_thermodynamic_admissibility"]["status"],
        )

        pressure_from_euler = (
            eos.baryon_density * eos.chemical_potential - eos.epsilon
        )
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
            solver_calls.append(
                (float(pressure), bool(kwargs.get("calculate_tidal")))
            )
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
            "eos_generation.stellar._tov_sequence.solve_star",
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
        self.assertIsNone(
            unresolved.to_dict()["passes_maximum_mass_threshold"]
        )
        self.assertEqual(1.8, max(row[1] for row in unresolved.sampled_models))
        self.assertFalse(unresolved.to_dict()["sampled_argmax_is_maximum_mass"])
        forbidden_solver.assert_not_called()


class LifecycleAvailabilityContracts(unittest.TestCase):
    def test_fixed_mass_success_remains_student_eligible_when_mmax_unavailable(
        self,
    ) -> None:
        stage = BSk24TOVStage("reporting", 5, 1.0e-8, 1.0e-10, 3)
        config = BSk24TrialConfig(
            amplitudes=(0.0, 0.2),
            fixed_masses_msun=(1.4,),
            tov_stages=(stage,),
            stellar_enabled=True,
        )
        case_table = pd.DataFrame(
            (
                {
                    "case_id": "early-causal",
                    "amplitude": 0.2,
                    "epsilon0_mev_fm3": 200.0,
                    "sigma_mev_fm3": 50.0,
                    "delta_mev_fm3": 40.0,
                },
                {
                    "case_id": "unresolved-case",
                    "amplitude": -1.0,
                    "epsilon0_mev_fm3": 200.0,
                    "sigma_mev_fm3": 50.0,
                    "delta_mev_fm3": 40.0,
                },
            )
        )
        plan = SimpleNamespace(config=config, case_table=case_table)
        gate_reports = {
            "early-causal": {
                "status": "accepted_raw_local_physics_gate",
                "complete_raw_proposal_causal_through_direct_endpoint": False,
                "retained_domain": {
                    "endpoint_reason": "first_continuous_causal_crossing",
                    "epsilon_max_mev_fm3": 600.0,
                    "pressure_max_mev_fm3": 250.0,
                },
                "first_failure": None,
            },
            "unresolved-case": {
                "status": "unresolved_raw_local_physics_gate",
                "complete_raw_proposal_causal_through_direct_endpoint": False,
                "retained_domain": {
                    "endpoint_reason": (
                        "unavailable_unresolved_continuous_assessment"
                    ),
                    "epsilon_max_mev_fm3": None,
                    "pressure_max_mev_fm3": None,
                },
                "first_failure": {"reason": "synthetic_unresolved"},
            },
        }
        fixed = pd.DataFrame(
            (
                {
                    "case_id": "early-causal",
                    "stage": stage.name,
                    "target_mass_msun": 1.4,
                    "status": "bracketed_and_solved",
                },
            )
        )
        maximum = pd.DataFrame(
            (
                {
                    "case_id": "early-causal",
                    "stage": stage.name,
                    "maximum_mass_availability_status": (
                        "unavailable_unresolved_no_turning_point_before_eos_endpoint"
                    ),
                },
            )
        )

        ledger = internal_lifecycle._case_lifecycle_ledger(
            plan,
            accepted_case_ids=("early-causal",),
            gate_reports=gate_reports,
            completed_stellar_case_ids={"early-causal"},
            fixed_mass_rows=fixed,
            maximum_mass_rows=maximum,
        ).set_index("case_id")

        accepted = ledger.loc["early-causal"]
        self.assertEqual(
            "through_first_continuous_causal_crossing",
            accepted["acceptance_domain"],
        )
        self.assertEqual(
            "assessed_noncausal_beyond_first_retained_crossing",
            accepted["full_domain_gate_status"],
        )
        self.assertEqual(
            "accepted_selected_retained_domain",
            accepted["selected_domain_status"],
        )
        self.assertEqual(
            "all_requested_fixed_masses_succeeded",
            accepted["requested_fixed_masses_status"],
        )
        self.assertEqual(
            "unavailable_unresolved_no_turning_point_before_eos_endpoint",
            accepted["maximum_mass_availability_status"],
        )
        self.assertEqual(
            "eligible_all_requested_fixed_masses_succeeded",
            accepted["student_view_eligibility_status"],
        )
        self.assertEqual(600.0, accepted["retained_epsilon_max_mev_fm3"])
        rejected = ledger.loc["unresolved-case"]
        self.assertEqual("rejected", rejected["status"])
        self.assertEqual("none", rejected["acceptance_domain"])
        self.assertEqual(
            "assessed_unresolved", rejected["full_domain_gate_status"]
        )
        self.assertEqual(
            "unresolved_no_selected_retained_domain",
            rejected["selected_domain_status"],
        )
        self.assertEqual(
            "evidence_only_raw_gate_not_accepted",
            rejected["student_view_eligibility_status"],
        )
        self.assertEqual(
            "skipped_due_to_raw_gate_rejection",
            rejected["stellar_calculation"],
        )


class SavedSummaryContracts(unittest.TestCase):
    def test_saved_evidence_builds_and_renders_through_the_stable_facade(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="synthetic-summary-evidence-",
        ) as temporary:
            evidence = Path(temporary)

            def write_json(name: str, payload: object) -> None:
                (evidence / name).write_text(
                    json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )

            write_json(
                "metadata.json",
                {
                    "schema_id": internal_summary.PACKET_SCHEMA_ID,
                    "packet_status": "complete",
                    "configuration_hash": "a" * 64,
                    "identity_status": "pass",
                },
            )
            write_json(
                "complete_configuration.json",
                {
                    "amplitudes": [0.0, 0.01],
                    "epsilon0_values_mev_fm3": [200.0],
                    "sigma_values_mev_fm3": [50.0],
                    "deltas_mev_fm3": [40.0],
                    "epsilon_match_mev_fm3": None,
                    "stellar_enabled": False,
                },
            )
            write_json(
                "raw_gate_report.json",
                {
                    "cases": {
                        "accepted-case": {
                            "status": "accepted_raw_local_physics_gate",
                            "finite_values": True,
                            "positive_energy_density": True,
                            "positive_pressure": True,
                            "strictly_monotone_pressure_implied": True,
                            "full_retained_domain_passed": True,
                        },
                        "rejected-case": {
                            "status": "rejected_raw_local_physics_gate",
                            "finite_values": True,
                            "positive_energy_density": True,
                            "positive_pressure": False,
                            "strictly_monotone_pressure_implied": False,
                            "full_retained_domain_passed": False,
                            "first_failure": {
                                "quantity": "pressure",
                                "value": -0.25,
                            },
                        },
                    }
                },
            )
            write_json(
                "thermodynamic_convergence.json",
                {
                    "status": "complete_all_requested_stages",
                    "uncertainty_envelope": {"pressure": 0.01},
                },
            )
            write_json(
                "reproduction.json",
                {
                    "portable_plan_command": "bsk24-trial plan --json",
                    "portable_run_command": "bsk24-trial run --execute",
                    "portable_plan_hash": "b" * 64,
                },
            )
            (evidence / "case_ledger.csv").write_text(
                "case_id,status,rejection_reason,amplitude,epsilon0_mev_fm3,"
                "sigma_mev_fm3,delta_mev_fm3,pressure_reconstruction,"
                "stellar_calculation,clipping_or_repair\n"
                "accepted-case,accepted,,0.0,200.0,50.0,40.0,complete,"
                "not_requested,none\n"
                "rejected-case,rejected,,0.01,200.0,50.0,40.0,"
                "skipped_due_to_raw_gate_rejection,"
                "skipped_due_to_raw_gate_rejection,none\n",
                encoding="utf-8",
                newline="\n",
            )

            model = internal_summary.build_summary_model(evidence)
            self.assertEqual("mixed", model["outcome"])
            self.assertEqual(1, model["cases"]["accepted"])
            self.assertEqual(1, model["cases"]["rejected"])
            self.assertEqual(
                '{"quantity":"pressure","value":-0.25}',
                model["cases"]["rows"][1]["rejection_reason"],
            )
            self.assertTrue(
                model["physical_assessment"][
                    "rejected_proposals_received_no_reconstruction_or_stellar_work"
                ]
            )
            self.assertEqual(
                "available_in_saved_convergence_evidence",
                model["numerical"]["saved_uncertainty_status"],
            )

            rendered = internal_summary.render_summary_markdown(model)
            self.assertIn("**Outcome: MIXED.**", rendered)
            self.assertIn("Exact rejection reason", rendered)
            self.assertIn("bsk24-trial plan --json", rendered)

            writer = Mock()
            with patch.object(internal_summary, "_write_text_atomic", writer):
                summary_path = internal_summary.write_packet_summary(evidence)
            self.assertEqual(evidence / "summary.md", summary_path)
            writer.assert_called_once_with(rendered, summary_path)
            self.assertFalse(summary_path.exists())


class FailClosedOrchestrationContracts(unittest.TestCase):
    def test_rejected_case_never_crosses_downstream_case_boundaries(self) -> None:
        reconstructed_case_ids: list[str] = []
        stellar_case_ids: set[str] = set()

        class DownstreamProbeComplete(RuntimeError):
            pass

        with tempfile.TemporaryDirectory(
            prefix="synthetic-orchestration-",
        ) as temporary:
            packet = Path(temporary) / "packet"
            config = BSk24TrialConfig(
                amplitudes=(0.0, -1.0, -0.5, 0.3),
                deltas_mev_fm3=(40.0,),
                thermodynamic_stages=(
                    BSk24ThermodynamicStage("synthetic", 17, 17),
                ),
                tov_stages=(
                    BSk24TOVStage("synthetic", 5, 1.0e-8, 1.0e-10, 3),
                ),
                raw_gate_lower_points=17,
                raw_gate_upper_points=17,
                stellar_enabled=True,
            )
            case_table = pd.DataFrame(
                (
                    {
                        "case_id": "accepted-case",
                        "amplitude": 0.0,
                        "epsilon0_mev_fm3": 200.0,
                        "sigma_mev_fm3": 50.0,
                        "delta_mev_fm3": 40.0,
                    },
                    {
                        "case_id": "rejected-case",
                        "amplitude": -1.0,
                        "epsilon0_mev_fm3": 200.0,
                        "sigma_mev_fm3": 50.0,
                        "delta_mev_fm3": 40.0,
                    },
                    {
                        "case_id": "unresolved-case",
                        "amplitude": -0.5,
                        "epsilon0_mev_fm3": 200.0,
                        "sigma_mev_fm3": 50.0,
                        "delta_mev_fm3": 40.0,
                    },
                    {
                        "case_id": "tabulation-unresolved-case",
                        "amplitude": 0.3,
                        "epsilon0_mev_fm3": 200.0,
                        "sigma_mev_fm3": 50.0,
                        "delta_mev_fm3": 40.0,
                    },
                )
            )
            plan = SimpleNamespace(
                output_path=packet,
                case_table=case_table,
                to_dict=lambda: {"schema_id": "synthetic_passive_plan"},
            )
            baseline = SimpleNamespace()

            def raw_gate(
                _baseline: object,
                proposal: deformation.BSk24WindowedDeformation,
                **_kwargs: object,
            ) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
                statuses = {
                    "accepted-case": "accepted_raw_local_physics_gate",
                    "rejected-case": "rejected_raw_local_physics_gate",
                    "unresolved-case": "unresolved_raw_local_physics_gate",
                    "tabulation-unresolved-case": (
                        "accepted_raw_local_physics_gate"
                    ),
                }
                return (
                    {
                        "case_id": proposal.case_id,
                        "status": statuses[proposal.case_id],
                    },
                    np.asarray([1.0]),
                    np.asarray([0.5]),
                )

            def build_windowed(
                _baseline: object,
                proposal: deformation.BSk24WindowedDeformation,
                **_kwargs: object,
            ) -> SimpleNamespace:
                if proposal.case_id == "tabulation-unresolved-case":
                    raise deformation.BSk24MechanicalStabilityError(
                        {
                            "case_id": proposal.case_id,
                            "status": "unresolved_tabulation_resolution",
                            "failure_reason": "synthetic_resolution_gap",
                        }
                    )
                reconstructed_case_ids.append(proposal.case_id)
                return SimpleNamespace(deformation=proposal)

            def stellar_probe(**kwargs: object) -> None:
                generated = kwargs["generated"]
                assert isinstance(generated, dict)
                stellar_case_ids.update(generated)
                raise DownstreamProbeComplete

            callbacks = RunCallbacks(
                prepare_trial=lambda _config: plan,
                load_trial=Mock(),
                generate_plots=Mock(),
                validate_packet=Mock(),
                build_consistent_baseline=lambda *_args, **_kwargs: baseline,
                raw_local_physics_gate=raw_gate,
                raw_gate_frame=lambda **kwargs: pd.DataFrame(
                    {"case_id": [kwargs["case_id"]]}
                ),
                build_windowed_eos=build_windowed,
                thermodynamic_profile_frame=lambda *_args: pd.DataFrame(),
                thermodynamic_residual_frame=lambda *_args: pd.DataFrame(),
                window_characterization=lambda *_args: {},
                thermodynamic_convergence=lambda *_args: {"status": "synthetic"},
                run_stellar=stellar_probe,
            )

            with (
                patch(
                    "eos_generation._internal.execution.write_json_atomic"
                ) as write_json,
                patch("eos_generation._internal.execution.write_csv_atomic"),
                self.assertRaises(DownstreamProbeComplete),
            ):
                run_bsk24_trial(config, callbacks=callbacks)

        self.assertEqual(["accepted-case"], reconstructed_case_ids)
        self.assertEqual({"accepted-case"}, stellar_case_ids)
        self.assertNotIn("rejected-case", reconstructed_case_ids)
        self.assertNotIn("rejected-case", stellar_case_ids)
        self.assertNotIn("unresolved-case", reconstructed_case_ids)
        self.assertNotIn("unresolved-case", stellar_case_ids)
        self.assertNotIn("tabulation-unresolved-case", reconstructed_case_ids)
        self.assertNotIn("tabulation-unresolved-case", stellar_case_ids)
        raw_payloads = [
            call.args[0]
            for call in write_json.call_args_list
            if call.args[1].name == "raw_gate_report.json"
        ]
        self.assertEqual(2, len(raw_payloads))
        self.assertEqual("eos_generation_raw_gate_v2", raw_payloads[-1]["schema_id"])
        self.assertEqual(
            ["unresolved-case", "tabulation-unresolved-case"],
            raw_payloads[-1]["unresolved_case_ids"],
        )
        self.assertEqual(
            ["rejected-case"], raw_payloads[-1]["hard_rejected_case_ids"]
        )


if __name__ == "__main__":
    unittest.main()

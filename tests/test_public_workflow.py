from __future__ import annotations

import json
import io
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import eos_generation
from eos_generation.cli import main as cli_main
from eos_generation.experiment import (
    ExperimentSettings,
    plan_experiment,
    run_experiment,
)


ROOT = Path(__file__).resolve().parents[1]


class PublicWorkflowTests(unittest.TestCase):
    def test_public_surface_is_small_and_defined_at_the_package_root(self) -> None:
        expected = {
            "Experiment",
            "ExperimentPlan",
            "ExperimentResult",
            "ExperimentSettings",
            "load_experiment",
            "plan_experiment",
            "run_experiment",
            "validate_experiment",
        }
        self.assertEqual(expected, set(eos_generation.__all__))
        for name in expected:
            self.assertIs(getattr(eos_generation, name), getattr(eos_generation.experiment, name))

    def test_quickstart_loads_and_planning_is_passive(self) -> None:
        settings = ExperimentSettings.from_json(ROOT / "configs" / "quickstart.json")
        contract = json.loads(
            (
                ROOT
                / "tests"
                / "fixtures"
                / "bsk24_contract_v1"
                / "lifecycle_contract.json"
            ).read_text(encoding="utf-8")
        )["quickstart"]
        output_root = ROOT / "runs" / "test-passive-plan"
        self.assertFalse(output_root.exists())
        first = plan_experiment(settings, output_root=output_root)
        second = plan_experiment(settings, output_root=output_root)
        self.assertEqual(first.to_dict(), second.to_dict())
        payload = first.to_dict()
        self.assertTrue(payload["planning_is_passive"])
        self.assertEqual(0, payload["scientific_solver_calls"])
        self.assertEqual(0, payload["filesystem_writes"])
        self.assertFalse(output_root.exists())
        contracts = payload["source_identity"]["project_contract_sha256"]
        self.assertEqual({"environment.yml", "pyproject.toml"}, set(contracts))
        self.assertTrue(all(len(value) == 64 for value in contracts.values()))
        self.assertEqual(
            contract["configuration_hash"],
            first.child_plans[0].config.deterministic_hash(),
        )
        self.assertEqual(
            contract["case_ids_in_order"],
            first.case_table["case_id"].tolist(),
        )
        for name, expected in contract["estimates"].items():
            self.assertEqual(expected, first.estimates[name])
        child = payload["children"][0]
        expanded = child["expanded_configuration"]
        self.assertEqual(257, expanded["raw_gate_lower_points"])
        self.assertEqual(1025, expanded["raw_gate_upper_points"])
        self.assertEqual(17, expanded["maximum_mass_initial_points"])
        self.assertEqual(1.0e-7, expanded["fixed_mass_root_xtol_mev_fm3"])
        self.assertEqual(
            {"output_packet_name", "output_path", "resume_policy"},
            set(child["operational_destination"]),
        )

    def test_internal_plan_objects_are_pickle_and_spawn_safe(self) -> None:
        settings = ExperimentSettings.from_json(ROOT / "configs" / "quickstart.json")
        plan = plan_experiment(settings, output_root=ROOT / "runs" / "test-pickle")
        recovered = pickle.loads(pickle.dumps(plan.child_plans[0].config))
        self.assertEqual(plan.child_plans[0].config, recovered)

    def test_plan_injects_the_exact_zero_control(self) -> None:
        settings = ExperimentSettings.from_values(amplitudes=(0.01,), ramp_width=40.0)
        plan = plan_experiment(settings, output_root=ROOT / "runs" / "test-a0")
        amplitudes = plan.case_table["amplitude"].tolist()
        self.assertEqual([0.0, 0.01], amplitudes)
        self.assertTrue(plan.child_plans[0].a0_identity_control_injected)

    def test_below_anchor_geometry_plans_passively_when_its_tail_overlaps(
        self,
    ) -> None:
        output_root = ROOT / "runs" / "test-below-anchor-plan"
        self.assertFalse(output_root.exists())
        settings = ExperimentSettings.from_values(
            amplitudes=(0.0, 0.01),
            center=140.0,
            width=5.0,
            ramp_width=40.0,
        )
        plan = plan_experiment(settings, output_root=output_root)
        payload = plan.to_dict()
        expanded = payload["children"][0]["expanded_configuration"]
        self.assertLess(
            expanded["epsilon0_mev_fm3"],
            expanded["effective_epsilon_match_mev_fm3"],
        )
        self.assertTrue(payload["planning_is_passive"])
        self.assertEqual(0, payload["scientific_solver_calls"])
        self.assertEqual(0, payload["filesystem_writes"])
        self.assertFalse(output_root.exists())

    def test_execution_requires_a_reviewed_plan_and_explicit_gate(self) -> None:
        settings = ExperimentSettings.from_values(amplitudes=(0.0,), precision="quick")
        with self.assertRaises(TypeError):
            run_experiment(settings, execute=True)  # type: ignore[arg-type]
        plan = plan_experiment(settings, output_root=ROOT / "runs" / "test-gate")
        with self.assertRaises(PermissionError):
            run_experiment(plan)
        self.assertFalse(plan.experiment_path.exists())

    def test_reviewed_plan_is_bound_to_governed_source_and_runtime(self) -> None:
        settings = ExperimentSettings.from_values(amplitudes=(0.0,), precision="quick")
        plan = plan_experiment(settings, output_root=ROOT / "runs" / "test-drift")
        with patch(
            "eos_generation.experiment._active_source_identity",
            return_value=(
                plan.source_inventory_id,
                "0" * 64,
                plan.source_file_count,
                plan.source_contracts,
            ),
        ):
            with self.assertRaises(RuntimeError):
                run_experiment(plan, execute=True)
        self.assertFalse(plan.experiment_path.exists())

    def test_strict_json_rejects_unknown_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text('{"amplitudes":[NaN]}', encoding="utf-8")
            with self.assertRaises(ValueError):
                ExperimentSettings.from_json(path)
            path.write_text('{"amplitudes":[0],"secret_stage":1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                ExperimentSettings.from_json(path)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required field"):
                ExperimentSettings.from_json(path)
            payload = json.loads(
                (ROOT / "configs" / "quickstart.json").read_text(encoding="utf-8")
            )
            payload["amplitudes"] = 0.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "amplitudes must be an array"):
                ExperimentSettings.from_json(path)

    def test_cli_plan_creates_no_output(self) -> None:
        output_root = ROOT / "runs" / "test-cli-plan"
        with redirect_stdout(io.StringIO()):
            status = cli_main(
                [
                    "plan",
                    "--config",
                    str(ROOT / "configs" / "quickstart.json"),
                    "--output-root",
                    str(output_root),
                    "--json",
                ]
            )
        self.assertEqual(0, status)
        self.assertFalse(output_root.exists())

    def test_cli_run_requires_the_reviewed_plan_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "unreviewed-run"
            errors = io.StringIO()
            with redirect_stderr(errors):
                status = cli_main(
                    [
                        "run",
                        "--config",
                        str(ROOT / "configs" / "quickstart.json"),
                        "--output-root",
                        str(output_root),
                        "--execute",
                    ]
                )
            self.assertEqual(2, status)
            self.assertIn("--plan-hash", errors.getvalue())
            self.assertFalse(output_root.exists())

    def test_cli_validation_failure_is_visible_without_json(self) -> None:
        report = {
            "status": "fail",
            "experiment_path": "runs/example",
            "child_packet_count": 1,
            "failures": ["aggregate manifest mismatch"],
            "children": [
                {
                    "status": "fail",
                    "packet_path": "geometry_001",
                    "failures": ["raw gate record is incomplete"],
                }
            ],
        }
        output = io.StringIO()
        with patch(
            "eos_generation.experiment.validate_experiment", return_value=report
        ), redirect_stdout(output):
            status = cli_main(["validate", "runs/example"])
        self.assertEqual(1, status)
        self.assertIn("aggregate manifest mismatch", output.getvalue())
        self.assertIn("raw gate record is incomplete", output.getvalue())
        self.assertIn("--json", output.getvalue())

    def test_settings_round_trip_and_cartesian_geometry_are_explicit(self) -> None:
        original = ExperimentSettings.from_values(
            amplitudes=(-0.01, 0.0, 0.01),
            center=(200.0, 240.0),
            width=50.0,
            ramp_width=(30.0, 40.0),
            precision="strict",
        )
        recovered = ExperimentSettings.from_dict(
            json.loads(json.dumps(original.to_dict(), allow_nan=False))
        )
        self.assertEqual(original, recovered)
        plan = plan_experiment(
            recovered, output_root=ROOT / "runs" / "test-cartesian"
        )
        self.assertEqual(4, len(plan.child_plans))
        self.assertEqual({1, 2, 3, 4}, set(plan.case_table["geometry_index"]))

    def test_bsk24_cartesian_zero_control_has_one_physical_owner(self) -> None:
        settings = ExperimentSettings.from_values(
            amplitudes=(0.0, 0.01, -0.01),
            center=(240.0, 200.0),
            width=50.0,
            ramp_width=(40.0, 30.0),
            precision="quick",
        )
        plan = plan_experiment(
            settings, output_root=ROOT / "runs" / "test-bsk24-a0-owner"
        )
        self.assertEqual(12, len(plan.case_table))
        self.assertEqual(
            9, sum(len(child.case_table) for child in plan.child_plans)
        )
        self.assertEqual(
            3, sum(len(child.logical_alias_table) for child in plan.child_plans)
        )
        owners = [
            child
            for child in plan.child_plans
            if child.config.zero_amplitude_control_owner
        ]
        self.assertEqual(1, len(owners))
        owner = owners[0].config
        self.assertEqual((200.0, 50.0, 30.0), (
            owner.epsilon0_mev_fm3,
            owner.sigma_mev_fm3,
            owner.deltas_mev_fm3[0],
        ))
        zero_rows = plan.case_table.loc[plan.case_table["amplitude"].eq(0.0)]
        self.assertEqual(4, len(zero_rows))
        self.assertEqual(1, zero_rows["physical_case_id"].nunique())
        self.assertEqual(3, plan.estimates["deduplicated_logical_case_aliases"])
        self.assertEqual(9, plan.estimates["physical_deformation_cases"])
        self.assertTrue(plan.to_dict()["planning_is_passive"])
        self.assertEqual(0, plan.to_dict()["scientific_solver_calls"])

        from eos_generation._internal.planning import BSk24TrialConfig

        saved_owner = owners[0].config.to_dict()
        recovered_owner = BSk24TrialConfig.from_dict(saved_owner)
        self.assertEqual(owners[0].config, recovered_owner)
        tampered = dict(
            saved_owner,
            zero_amplitude_physical_case_id="bsk24_baseline_tampered",
        )
        with self.assertRaisesRegex(ValueError, "disagrees with its inputs"):
            BSk24TrialConfig.from_dict(tampered)

    def test_legacy_direct_bsk24_trial_does_not_enable_alias_mode(self) -> None:
        from eos_generation._internal.planning import (
            BSk24TrialConfig,
            prepare_bsk24_trial,
        )

        config = BSk24TrialConfig(amplitudes=(0.0, 0.01))
        self.assertIsNone(config.zero_amplitude_control_owner)
        self.assertNotIn("zero_amplitude_control_owner", config.to_dict())
        self.assertNotIn("logical_amplitudes", config.to_dict())
        child = prepare_bsk24_trial(config)
        self.assertNotIn("logical_alias_table", child.to_dict())
        self.assertTrue(child.logical_alias_table.empty)

    def test_accidental_cartesian_explosions_fail_before_planning(self) -> None:
        with self.assertRaisesRegex(ValueError, "public planning limit is 256"):
            ExperimentSettings.from_values(center=tuple(range(1, 258)))
        with self.assertRaisesRegex(ValueError, "public planning limit is 4096"):
            ExperimentSettings.from_values(amplitudes=tuple(range(1, 4097)))
        with self.assertRaisesRegex(ValueError, "at most 32 targets"):
            ExperimentSettings.from_values(
                fixed_masses=tuple(index / 10.0 for index in range(1, 34))
            )

    def test_planning_rejects_output_paths_outside_runs(self) -> None:
        from eos_generation._internal.planning import BSk24TrialConfig

        settings = ExperimentSettings.from_values(amplitudes=(0.0,))
        for output_root in (ROOT / "outside-runs", ROOT / "runs" / ".." / "escape"):
            with self.subTest(output_root=output_root), self.assertRaisesRegex(
                ValueError, "outside runs"
            ):
                plan_experiment(settings, output_root=output_root)
        with self.assertRaisesRegex(ValueError, "outside runs"):
            BSk24TrialConfig(output_path=ROOT / "runs")

    def test_planning_rejects_a_runs_symlink_that_escapes(self) -> None:
        settings = ExperimentSettings.from_values(amplitudes=(0.0,))
        runs = ROOT / "runs"
        runs_preexisted = runs.exists()
        with tempfile.TemporaryDirectory() as temporary:
            link = runs / ".containment-test-link"
            try:
                runs.mkdir(exist_ok=True)
                link.symlink_to(Path(temporary), target_is_directory=True)
            except OSError:
                if link.is_symlink():
                    link.unlink()
                if not runs_preexisted and runs.exists() and not any(runs.iterdir()):
                    runs.rmdir()
                self.skipTest("directory symlinks are unavailable on this platform")
            try:
                with self.assertRaisesRegex(ValueError, "outside runs"):
                    plan_experiment(settings, output_root=link)
            finally:
                link.unlink(missing_ok=True)
                if not runs_preexisted and runs.exists() and not any(runs.iterdir()):
                    runs.rmdir()

    def test_default_and_leading_runs_paths_are_checkout_relative(self) -> None:
        settings = ExperimentSettings.from_values(amplitudes=(0.0,))
        runs = ROOT / "runs"
        runs_preexisted = runs.exists()
        entries_before = set(runs.iterdir()) if runs_preexisted else set()
        previous = Path.cwd()
        try:
            os.chdir(ROOT / "docs")
            with patch.object(Path, "mkdir", side_effect=AssertionError("planning wrote a directory")), \
                 patch.object(Path, "write_text", side_effect=AssertionError("planning wrote text")), \
                 patch.object(Path, "write_bytes", side_effect=AssertionError("planning wrote bytes")):
                default = plan_experiment(settings)
                explicit = plan_experiment(settings, output_root="runs/from-docs")
        finally:
            os.chdir(previous)
        self.assertEqual(ROOT / "runs", default.experiment_path.parent)
        self.assertEqual(ROOT / "runs" / "from-docs", explicit.experiment_path.parent)
        # Existing user runs are allowed and must never be removed for a test.
        # The same assertion still forbids creating runs/ in a clean checkout.
        self.assertEqual(runs_preexisted, runs.exists())
        self.assertEqual(entries_before, set(runs.iterdir()) if runs.exists() else set())


if __name__ == "__main__":
    unittest.main()

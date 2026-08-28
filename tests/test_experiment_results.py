from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any, Iterator
import unittest
from unittest.mock import patch

from eos_generation._internal.summary import PACKET_SCHEMA_ID
from eos_generation._internal._summary_evidence import (
    _maximum_mass_group_evidence,
    _validation_model,
)
from eos_generation._internal.planning import prepare_bsk24_trial
from eos_generation._internal.status import build_bsk24_trial_status
from eos_generation.experiment import (
    ExperimentSettings,
    load_experiment,
    plan_experiment,
    run_experiment,
    validate_experiment,
)
from eos_generation.reporting._validation_io import _Layer
from eos_generation.reporting._validation_cases import (
    RAW_GATE_SCHEMA,
    _validate_case_consistency,
    _validate_raw_gate_profiles,
)
from eos_generation.reporting._validation_scientific import (
    _validate_fixed_mass_completeness,
    _validate_final_lifecycle,
    _validate_maximum_mass_artifacts,
    _validate_sequence_completeness,
    _validate_thermodynamic_outputs,
)
from eos_generation.reporting.validation import (
    validate_bsk24_trial_packet_layers,
)


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_DOCUMENTS = (
    "experiment.json",
    "experiment_config.json",
    "reproduction_plan.json",
    "reviewed_plan.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seal_aggregate(experiment_path: Path, child_names: tuple[str, ...]) -> None:
    files = {name: experiment_path / name for name in AGGREGATE_DOCUMENTS}
    files.update(
        {
            f"{name}/SHA256SUMS.txt": experiment_path / name / "SHA256SUMS.txt"
            for name in child_names
        }
    )
    manifest = "".join(
        f"{_sha256(path)}  {relative}\n"
        for relative, path in sorted(files.items())
    )
    (experiment_path / "SHA256SUMS.txt").write_text(
        manifest, encoding="utf-8", newline="\n"
    )


@contextmanager
def _temporary_runs_root() -> Iterator[Path]:
    runs = ROOT / "runs"
    runs_preexisted = runs.exists()
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aggregate-contract-", dir=runs) as directory:
        yield Path(directory)
    if not runs_preexisted:
        try:
            runs.rmdir()
        except OSError:
            pass


def _synthetic_trial_result(config: Any) -> SimpleNamespace:
    packet = Path(config.output_path).resolve(strict=False)
    packet.mkdir(parents=True, exist_ok=False)
    payload = packet / "synthetic-child.txt"
    payload.write_text("synthetic child; no scientific solver was called\n", encoding="utf-8")
    configuration = packet / "complete_configuration.json"
    _write_json(configuration, config.to_dict())
    metadata = packet / "metadata.json"
    _write_json(
        metadata,
        {
            "schema_id": PACKET_SCHEMA_ID,
            "identity_status": "synthetic_not_assessed",
            "numerical_convergence_status": "synthetic_not_assessed",
        },
    )
    trial_plan = packet / "trial_plan.json"
    _write_json(trial_plan, prepare_bsk24_trial(config).to_dict())
    child_files = (configuration, metadata, payload, trial_plan)
    (packet / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in sorted(child_files, key=lambda item: item.name)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return SimpleNamespace(packet_path=packet, config=config)


def _loaded_trial(
    config_by_path: dict[Path, Any], packet_path: str | Path
) -> SimpleNamespace:
    packet = Path(packet_path).resolve(strict=False)
    return SimpleNamespace(packet_path=packet, config=config_by_path[packet])


def _layered_child_validation(
    *,
    validity: str = "pass",
    availability: str = "complete",
) -> dict[str, Any]:
    limitations = (
        []
        if availability == "complete"
        else ["maximum_mass:unavailable:1"]
    )
    hard_failures = [] if validity == "pass" else ["synthetic_hard_failure"]
    return {
        "status": "pass" if validity == "pass" else "fail",
        "failures": hard_failures,
        "warnings": limitations,
        "internal_packet_integrity": {"status": "pass"},
        "current_source_equivalence": {"status": "equivalent"},
        "scientific_output_validity": {
            "status": validity,
            "failures": hard_failures,
        },
        "scientific_output_availability": {
            "status": availability,
            "limitations": limitations,
        },
        "scientific_output_completeness": {
            "status": (
                "invalid"
                if validity != "pass"
                else availability
            ),
            "hard_validity": {
                "status": validity,
                "failures": hard_failures,
            },
            "availability": {
                "status": availability,
                "limitations": limitations,
            },
        },
    }


def _unresolved_maximum_row() -> dict[str, Any]:
    status = "unresolved_no_turning_point_before_eos_endpoint"
    return {
        "case_id": "direct",
        "stage": "synthetic",
        "status": status,
        "maximum_mass_resolved": False,
        "maximum_mass_availability_status": f"unavailable_{status}",
        "maximum_mass_msun": "",
        "maximum_mass_threshold_msun": 2.0,
        "passes_maximum_mass_threshold": "",
        "central_pressure_mev_fm3": "",
        "central_energy_density_mev_fm3": "",
        "central_sound_speed_squared": "",
        "radius_km": "",
        "turning_point_count": 0,
        "positive_left_secant": "",
        "negative_right_secant": "",
        "eos_endpoint_pressure_mev_fm3": 100.0,
        "endpoint_limitation": "eos_endpoint_reached_without_sampled_turning_bracket",
        "refinement_status": "not_started",
        "sampled_sequence_model_count": 0,
        "local_background_solver_call_count": 0,
        "tidal_solver_calls_for_maximum_mass": 0,
    }


def _unresolved_maximum_report() -> dict[str, Any]:
    return {
        "schema_id": "bsk24_maximum_mass_reports_v2",
        "cases": {
            "direct:synthetic": {
                "schema_id": "tov_resolved_maximum_mass_v2",
                "status": "unresolved_no_turning_point_before_eos_endpoint",
                "maximum_mass_resolved": False,
                "maximum_mass_threshold_msun": 2.0,
                "passes_maximum_mass_threshold": None,
                "maximum_mass_msun": None,
                "central_pressure_mev_fm3": None,
                "central_energy_density_mev_fm3": None,
                "central_sound_speed_squared": None,
                "radius_km": None,
                "decision_basis": "fail_closed_no_resolved_turning_point",
                "sampled_argmax_is_maximum_mass": False,
                "turning_point_count": 0,
                "turning_point_brackets": [],
                "selected_bracket": None,
                "positive_left_secant": None,
                "negative_right_secant": None,
                "stable_branch_extent": {
                    "model_count": 0,
                    "maximum_central_pressure_mev_fm3": None,
                    "models": [],
                },
                "sampled_models": [],
                "eos_endpoint": {
                    "pressure_mev_fm3": 100.0,
                    "reached_by_search": False,
                    "limitation": (
                        "eos_endpoint_reached_without_sampled_turning_bracket"
                    ),
                },
                "convergence": {
                    "refinement_status": "not_started",
                    "refinement_iterations": 0,
                    "global_refinement_rounds": 0,
                    "solver_call_count": 0,
                    "solver_failure_count": 0,
                    "solver_failures": [],
                },
                "tidal_calculations_performed": 0,
            }
        },
    }


def _make_synthetic_experiment(output_root: Path) -> tuple[Any, Any, dict[Path, Any]]:
    settings = ExperimentSettings.from_values(amplitudes=(0.0,), precision="quick")
    source_identity = (
        "synthetic-test-inventory-v1",
        "1" * 64,
        2,
        (("environment.yml", "2" * 64), ("pyproject.toml", "3" * 64)),
    )
    runtime_identity = (("python_version", "synthetic-test-runtime"),)
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        with (
            patch(
                "eos_generation.experiment._active_source_identity",
                return_value=source_identity,
            ),
            patch(
                "eos_generation.experiment._active_runtime_identity",
                return_value=runtime_identity,
            ),
            patch(
                "eos_generation._internal.runtime.execute_trial",
                side_effect=_synthetic_trial_result,
            ) as execute_trial,
        ):
            plan = plan_experiment(settings, output_root=output_root)
            result = run_experiment(plan, execute=True)
            execute_trial.assert_called_once_with(plan.child_plans[0].config)
    finally:
        os.chdir(previous)

    expected_configs = {
        Path(child.config.output_path).resolve(strict=False): child.config
        for child in plan.child_plans
    }
    return plan, result, expected_configs


@contextmanager
def _patched_child_loading(
    config_by_path: dict[Path, Any],
    *,
    expected_repository_root: Path = ROOT,
) -> Iterator[None]:
    def validate(
        path: str | Path,
        *,
        repository_root: str | Path | None = None,
    ) -> dict[str, Any]:
        packet = Path(path).resolve(strict=False)
        if packet not in config_by_path:
            raise AssertionError(f"unexpected child packet: {packet}")
        if Path(repository_root).resolve(strict=False) != expected_repository_root:
            raise AssertionError(f"unexpected owning repository: {repository_root}")
        return {"status": "pass", "failures": []}

    with patch(
        "eos_generation._internal.runtime.validate_trial",
        side_effect=validate,
    ):
        yield


class ExperimentResultContractTests(unittest.TestCase):
    def test_synthetic_run_writes_exact_manifest_and_loads_successfully(self) -> None:
        with _temporary_runs_root() as output_root:
            plan, result, configs = _make_synthetic_experiment(output_root)
            packet = plan.experiment_path

            self.assertTrue(result.completed)
            self.assertEqual(packet, result.experiment_path)
            self.assertEqual(
                {
                    "SHA256SUMS.txt",
                    "experiment.json",
                    "experiment_config.json",
                    "geometry_001",
                    "reproduction_plan.json",
                    "reviewed_plan.json",
                },
                {entry.name for entry in packet.iterdir()},
            )

            expected_files = {
                name: packet / name for name in AGGREGATE_DOCUMENTS
            }
            expected_files["geometry_001/SHA256SUMS.txt"] = (
                packet / "geometry_001" / "SHA256SUMS.txt"
            )
            expected_lines = [
                f"{_sha256(path)}  {relative}"
                for relative, path in sorted(expected_files.items())
            ]
            observed_lines = (packet / "SHA256SUMS.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(expected_lines, observed_lines)

            with _patched_child_loading(configs):
                loaded = load_experiment(packet)
            self.assertTrue(loaded.completed)
            self.assertEqual((packet / "geometry_001",), loaded.packet_paths)
            self.assertEqual(plan.settings, loaded.settings)

    def test_failed_child_keeps_only_reproduction_documents(self) -> None:
        with _temporary_runs_root() as output_root:
            settings = ExperimentSettings.from_values(amplitudes=(0.0,))
            source_identity = (
                "synthetic-test-inventory-v1",
                "1" * 64,
                2,
                (("environment.yml", "2" * 64), ("pyproject.toml", "3" * 64)),
            )
            runtime_identity = (("python_version", "synthetic-test-runtime"),)
            with (
                patch(
                    "eos_generation.experiment._active_source_identity",
                    return_value=source_identity,
                ),
                patch(
                    "eos_generation.experiment._active_runtime_identity",
                    return_value=runtime_identity,
                ),
            ):
                plan = plan_experiment(settings, output_root=output_root)
                with patch(
                    "eos_generation._internal.runtime.execute_trial",
                    side_effect=RuntimeError("synthetic child failure"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "synthetic child failure"):
                        run_experiment(plan, execute=True)

            self.assertEqual(
                {
                    "experiment_config.json",
                    "reproduction_plan.json",
                    "reviewed_plan.json",
                },
                {entry.name for entry in plan.experiment_path.iterdir()},
            )
            self.assertFalse((plan.experiment_path / "experiment.json").exists())
            self.assertFalse((plan.experiment_path / "SHA256SUMS.txt").exists())

    def test_reproduction_plan_is_portable_and_checked_semantically(self) -> None:
        with _temporary_runs_root() as output_root:
            plan, _, configs = _make_synthetic_experiment(output_root)
            packet = plan.experiment_path
            reproduction_path = packet / "reproduction_plan.json"
            reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))

            configuration_file = (
                packet / "experiment_config.json"
            ).relative_to(ROOT).as_posix()
            self.assertEqual(configuration_file, reproduction["configuration_file"])
            self.assertEqual("runs/reproductions", reproduction["output_root"])
            self.assertEqual(
                f'bsk24-trial plan --config "{configuration_file}" '
                '--output-root "runs/reproductions" --json',
                reproduction["plan_command"],
            )
            self.assertEqual(
                f'bsk24-trial run --config "{configuration_file}" '
                '--output-root "runs/reproductions" '
                f'--plan-hash {reproduction["plan_hash"]} --execute',
                reproduction["run_command"],
            )
            self.assertFalse(Path(reproduction["configuration_file"]).is_absolute())
            self.assertFalse(Path(reproduction["output_root"]).is_absolute())

            reproduction["run_command"] += " --tampered"
            _write_json(reproduction_path, reproduction)
            metadata_path = packet / "experiment.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["document_sha256"]["reproduction_plan.json"] = _sha256(
                reproduction_path
            )
            _write_json(metadata_path, metadata)
            _seal_aggregate(packet, ("geometry_001",))

            with _patched_child_loading(configs):
                with self.assertRaisesRegex(
                    ValueError, "reproduction run command mismatch"
                ):
                    load_experiment(packet)

    def test_saved_child_absolute_and_parent_traversal_paths_are_rejected(self) -> None:
        malicious_paths = (
            str((ROOT.parent / "outside" / "geometry_001").resolve(strict=False)),
            "../geometry_001",
        )
        for malicious in malicious_paths:
            with self.subTest(child_packet=malicious), _temporary_runs_root() as output_root:
                plan, _, configs = _make_synthetic_experiment(output_root)
                packet = plan.experiment_path
                metadata_path = packet / "experiment.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["child_packets"] = [malicious]
                _write_json(metadata_path, metadata)
                _seal_aggregate(packet, ("geometry_001",))

                with _patched_child_loading(configs):
                    with self.assertRaisesRegex(
                        ValueError, "saved child packet paths disagree|unsafe saved child"
                    ):
                        load_experiment(packet)

    def test_portable_paths_load_independently_of_current_working_directory(self) -> None:
        with _temporary_runs_root() as output_root:
            plan, _, configs = _make_synthetic_experiment(output_root)
            previous = Path.cwd()
            with tempfile.TemporaryDirectory(prefix="foreign-cwd-") as foreign:
                try:
                    os.chdir(foreign)
                    with _patched_child_loading(configs):
                        loaded = load_experiment(plan.experiment_path)
                finally:
                    os.chdir(previous)
            self.assertEqual(plan.experiment_path, loaded.experiment_path)
            self.assertTrue(loaded.completed)
            self.assertEqual(ROOT, loaded.repository_root)

    def test_nested_runs_path_rebases_from_the_complete_saved_path(self) -> None:
        with _temporary_runs_root() as output_root:
            nested_output = output_root / "nested" / "runs" / "experiment-output"
            plan, _, configs = _make_synthetic_experiment(nested_output)
            previous = Path.cwd()
            with tempfile.TemporaryDirectory(prefix="foreign-cwd-") as foreign:
                try:
                    os.chdir(foreign)
                    with _patched_child_loading(configs):
                        loaded = load_experiment(plan.experiment_path)
                finally:
                    os.chdir(previous)
            self.assertEqual(ROOT, loaded.repository_root)
            self.assertEqual(plan.experiment_path, loaded.experiment_path)

    def test_moved_packet_has_no_original_machine_path_and_rebases_child_config(
        self,
    ) -> None:
        with _temporary_runs_root() as output_root:
            plan, _, configs = _make_synthetic_experiment(output_root)
            packet = plan.experiment_path
            serialized = "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted(packet.rglob("*.json"))
            )
            self.assertNotIn(str(ROOT), serialized)
            self.assertNotIn("johns", serialized.lower())

            with tempfile.TemporaryDirectory(prefix="moved-repository-") as moved:
                moved_root = Path(moved).resolve(strict=False)
                moved_packet = moved_root / packet.relative_to(ROOT)
                moved_packet.parent.mkdir(parents=True)
                shutil.copytree(packet, moved_packet)
                moved_child = moved_packet / "geometry_001"
                moved_configs = {moved_child: next(iter(configs.values()))}
                previous = Path.cwd()
                with tempfile.TemporaryDirectory(prefix="foreign-cwd-") as foreign:
                    try:
                        os.chdir(foreign)
                        with _patched_child_loading(
                            moved_configs,
                            expected_repository_root=moved_root,
                        ):
                            loaded = load_experiment(moved_packet)
                    finally:
                        os.chdir(previous)

                self.assertEqual(moved_root, loaded.repository_root)
                self.assertEqual((moved_child,), loaded.packet_paths)
                self.assertEqual(
                    moved_child,
                    Path(loaded.child_results[0].config.output_path),
                )

    def test_foreign_cwd_validation_reaches_the_child_validator(self) -> None:
        with _temporary_runs_root() as output_root:
            plan, _, _ = _make_synthetic_experiment(output_root)
            previous = Path.cwd()
            with tempfile.TemporaryDirectory(prefix="foreign-cwd-") as foreign:
                try:
                    os.chdir(foreign)
                    validation = validate_experiment(plan.experiment_path)
                finally:
                    os.chdir(previous)
            self.assertEqual(1, validation["child_packet_count"])
            self.assertEqual("fail", validation["status"])
            self.assertNotIn("failures", validation)

    def test_foreign_cwd_child_status_accepts_the_explicit_owner(self) -> None:
        with _temporary_runs_root() as output_root:
            plan, _, _ = _make_synthetic_experiment(output_root)
            child = plan.experiment_path / "geometry_001"
            previous = Path.cwd()
            with tempfile.TemporaryDirectory(prefix="foreign-cwd-") as foreign:
                try:
                    os.chdir(foreign)
                    status = build_bsk24_trial_status(
                        child,
                        repository_root=ROOT,
                    )
                finally:
                    os.chdir(previous)
            self.assertEqual(str(child), status["packet_path"])
            self.assertEqual("invalid", status["packet_validity"])

    def test_foreign_cwd_authorized_plot_path_uses_the_explicit_owner(self) -> None:
        from eos_generation._internal.artifacts import (
            ensure_within_runs,
            project_root,
        )
        from eos_generation._internal.runtime import generate_trial_plots

        with _temporary_runs_root() as output_root:
            plan, _, _ = _make_synthetic_experiment(output_root)
            child = plan.experiment_path / "geometry_001"

            def validate(path: str | Path) -> dict[str, Any]:
                self.assertEqual(ROOT, project_root())
                self.assertEqual(child, ensure_within_runs(path))
                return {"status": "pass", "failures": []}

            def generate(path: str | Path, **_: Any) -> Any:
                self.assertEqual(ROOT, project_root())
                self.assertEqual(child, ensure_within_runs(path))
                return None

            previous = Path.cwd()
            with tempfile.TemporaryDirectory(prefix="foreign-cwd-") as foreign:
                try:
                    os.chdir(foreign)
                    with (
                        patch(
                            "eos_generation._internal.runtime.validate_trial",
                            side_effect=validate,
                        ),
                        patch(
                            "eos_generation._internal.runtime.generate_bsk24_trial_plots",
                            side_effect=generate,
                        ),
                    ):
                        generate_trial_plots(
                            child,
                            authorize_plot_overwrite=True,
                            repository_root=ROOT,
                        )
                finally:
                    os.chdir(previous)

    def test_validation_reports_aggregate_and_child_failures_without_raising(self) -> None:
        missing = ROOT / "runs" / "definitely-missing-aggregate-contract-test"
        aggregate_failure = validate_experiment(missing)
        self.assertEqual("fail", aggregate_failure["status"])
        self.assertEqual(0, aggregate_failure["child_packet_count"])
        self.assertEqual([], aggregate_failure["children"])
        self.assertRegex(
            aggregate_failure["failures"][0],
            r"^aggregate_load:FileNotFoundError:",
        )

        with _temporary_runs_root() as output_root:
            plan, _, configs = _make_synthetic_experiment(output_root)
            child_failure = {
                "status": "fail",
                "failures": ["synthetic_child:failed"],
            }
            with (
                patch(
                    "eos_generation._internal.runtime.validate_trial",
                    return_value=child_failure,
                ),
                patch(
                    "eos_generation._internal.runtime.load_trial",
                    side_effect=lambda path, **_: _loaded_trial(configs, path),
                ),
            ):
                validation = validate_experiment(plan.experiment_path)
            self.assertEqual("fail", validation["status"])
            self.assertEqual(1, validation["child_packet_count"])
            self.assertEqual([child_failure], validation["children"])

    def test_partial_scientific_availability_remains_loadable(self) -> None:
        with _temporary_runs_root() as output_root:
            plan, _, configs = _make_synthetic_experiment(output_root)
            partial = _layered_child_validation(availability="partial")
            with (
                patch(
                    "eos_generation._internal.runtime.validate_trial",
                    return_value=partial,
                ),
                patch(
                    "eos_generation._internal.runtime.load_trial",
                    side_effect=lambda path, **_: _loaded_trial(configs, path),
                ),
            ):
                loaded = load_experiment(plan.experiment_path)
                validation = validate_experiment(plan.experiment_path)
            self.assertTrue(loaded.completed)
            self.assertEqual("pass", validation["status"])
            self.assertEqual(
                "partial", validation["scientific_availability_status"]
            )
            summary_validation = _validation_model(partial)
            self.assertIsNotNone(summary_validation)
            self.assertEqual("valid", summary_validation["result_status"])
            self.assertEqual(
                "partial",
                summary_validation["scientific_output_availability"],
            )

    def test_status_marks_hard_valid_partial_packet_valid(self) -> None:
        partial = _layered_child_validation(availability="partial")
        with _temporary_runs_root() as packet:
            _write_json(
                packet / "metadata.json",
                {"schema_id": PACKET_SCHEMA_ID},
            )
            with (
                patch(
                    "eos_generation._internal.status._verify_packet_manifest_exact",
                    return_value=(),
                ),
                patch(
                    "eos_generation._internal.status._preflight_source_equivalence",
                    return_value=({"status": "equivalent"}, {}),
                ),
            ):
                status = build_bsk24_trial_status(
                    packet,
                    validate_packet=lambda _: partial,
                    summary_builder=lambda *_args, **_kwargs: {},
                )
        self.assertEqual("valid", status["packet_validity"])
        self.assertEqual(
            "partial", status["scientific_availability"]["status"]
        )

    def test_layered_validation_keeps_availability_out_of_hard_failures(self) -> None:
        internal = {
            "status": "pass",
            "failures": [],
            "warnings": [],
            "checks": {},
        }
        source = {
            "status": "equivalent",
            "failures": [],
            "warnings": [],
            "checks": {},
        }
        scientific = {
            "status": "partial",
            "failures": [],
            "limitations": ["maximum_mass:unavailable:1"],
            "warnings": [],
            "checks": {},
            "hard_validity": {
                "status": "pass",
                "failures": [],
                "warnings": [],
                "checks": {},
            },
            "availability": {
                "status": "partial",
                "limitations": ["maximum_mass:unavailable:1"],
                "warnings": [],
                "checks": {},
            },
        }
        context = {
            "source_hashes": {},
            "configuration": {},
            "metadata": {},
            "accepted_case_ids": set(),
        }
        with _temporary_runs_root() as packet:
            with (
                patch(
                    "eos_generation.reporting.validation._validate_internal",
                    return_value=(internal, context),
                ),
                patch(
                    "eos_generation.reporting.validation._validate_source_equivalence",
                    return_value=source,
                ),
                patch(
                    "eos_generation.reporting.validation._validate_scientific_completeness",
                    return_value=scientific,
                ),
            ):
                report = validate_bsk24_trial_packet_layers(
                    packet, current_source_hashes={}
                )
        self.assertEqual("pass", report["status"])
        self.assertEqual([], report["failures"])
        self.assertEqual(
            "pass", report["scientific_output_validity"]["status"]
        )
        self.assertEqual(
            "partial", report["scientific_output_availability"]["status"]
        )
        self.assertIn(
            "scientific_output_availability:maximum_mass:unavailable:1",
            report["warnings"],
        )

    def test_hard_scientific_invalidity_blocks_loading_even_with_top_pass(self) -> None:
        with _temporary_runs_root() as output_root:
            plan, _, configs = _make_synthetic_experiment(output_root)
            invalid = _layered_child_validation(validity="fail")
            invalid["status"] = "pass"
            with (
                patch(
                    "eos_generation._internal.runtime.validate_trial",
                    return_value=invalid,
                ),
                patch(
                    "eos_generation._internal.runtime.load_trial",
                    side_effect=lambda path, **_: _loaded_trial(configs, path),
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "saved child packet failed validation"
                ):
                    load_experiment(plan.experiment_path)

    def test_fail_closed_fixed_mass_unavailable_is_not_hard_invalid(self) -> None:
        configuration = {
            "tov_stages": [{"name": "synthetic"}],
            "fixed_masses_msun": [1.4],
        }
        base_row = {
            "case_id": "direct",
            "stage": "synthetic",
            "status": "unavailable_outside_retained_eos_domain",
            "reason": "target bracket exceeds retained endpoint",
            "target_mass_msun": 1.4,
            "mass_msun": "",
            "mass_residual_msun": "",
            "radius_km": "",
            "central_pressure_mev_fm3": "",
            "central_energy_density_mev_fm3": "",
            "central_sound_speed_squared": "",
            "k2": "",
            "lambda_dimensionless": "",
            "root_evaluation_count": "",
            "tidal_status": "",
            "tidal_failure_reason": "",
        }
        with tempfile.TemporaryDirectory(prefix="fixed-unavailable-") as temporary:
            packet = Path(temporary)
            _write_csv_row(packet / "fixed_mass_observables.csv", base_row)
            hard = _Layer()
            availability = _Layer()
            _validate_fixed_mass_completeness(
                packet,
                configuration,
                set(),
                hard,
                availability=availability,
                retained_pressure_endpoints={"direct": 1.0},
            )
            self.assertEqual([], hard.failures)
            self.assertEqual(
                ["fixed_mass:background_unavailable:1"],
                availability.failures,
            )

            contaminated = dict(base_row, radius_km=12.0)
            _write_csv_row(
                packet / "fixed_mass_observables.csv", contaminated
            )
            hard = _Layer()
            availability = _Layer()
            _validate_fixed_mass_completeness(
                packet,
                configuration,
                set(),
                hard,
                availability=availability,
            )
            self.assertTrue(
                any(
                    "unavailable_row_has_observables" in failure
                    for failure in hard.failures
                )
            )
            self.assertEqual([], availability.failures)

            coupled_to_maximum = dict(
                base_row,
                status="unavailable_maximum_mass_not_resolved",
                reason="legacy maximum-mass coupling",
            )
            _write_csv_row(
                packet / "fixed_mass_observables.csv", coupled_to_maximum
            )
            hard = _Layer()
            availability = _Layer()
            _validate_fixed_mass_completeness(
                packet,
                configuration,
                set(),
                hard,
                availability=availability,
            )
            self.assertTrue(
                any(
                    "invalid_background_status" in failure
                    for failure in hard.failures
                )
            )

            solved_outside_endpoint = {
                **base_row,
                "status": "bracketed_and_solved",
                "reason": "",
                "mass_msun": 1.4,
                "mass_residual_msun": 0.0,
                "radius_km": 12.0,
                "central_pressure_mev_fm3": 11.0,
                "central_energy_density_mev_fm3": 300.0,
                "central_sound_speed_squared": 0.5,
                "bracket_pressure_mev_fm3": "[10.0,12.0]",
                "root_xtol_mev_fm3": 0.001,
                "root_evaluation_count": 3,
                "tidal_status": "not_requested_background_only",
            }
            _write_csv_row(
                packet / "fixed_mass_observables.csv",
                solved_outside_endpoint,
            )
            hard = _Layer()
            availability = _Layer()
            _validate_fixed_mass_completeness(
                packet,
                configuration,
                set(),
                hard,
                availability=availability,
                require_tides=False,
                retained_pressure_endpoints={"direct": 10.0},
            )
            self.assertTrue(
                any(
                    "invalid_solved_background_row" in failure
                    for failure in hard.failures
                )
            )

    def test_endpoint_below_sequence_floor_is_availability_not_invalidity(self) -> None:
        configuration = {
            "tov_stages": [
                {"name": "synthetic", "sequence_points": 5}
            ]
        }
        row = {
            "case_id": "direct",
            "stage": "synthetic",
            "attempted_index": 0,
            "calculation_status": "failed",
            "failure_category": "eos_endpoint_below_sequence_floor",
            "failure_reason": "retained endpoint is below the sequence floor",
            "central_pressure_mev_fm3": 1.0,
            "Mass": "",
            "Radius": "",
            "Lambda": "",
            "P_Central": "",
            "Eps_Central": "",
            "CS2_Central": "",
            "eps_surf": "",
            "k2": "",
            "is_sampled_peak": "",
            "is_domain_end": "",
            "tidal_status": "",
            "tidal_failure_reason": "",
        }
        with tempfile.TemporaryDirectory(prefix="sequence-unavailable-") as temporary:
            packet = Path(temporary)
            _write_csv_row(packet / "stellar_sequences.csv", row)
            hard = _Layer()
            availability = _Layer()
            _validate_sequence_completeness(
                packet,
                configuration,
                set(),
                hard,
                availability=availability,
                retained_pressure_endpoints={"direct": 1.0},
            )
            outside_hard = _Layer()
            _validate_sequence_completeness(
                packet,
                configuration,
                set(),
                outside_hard,
                availability=_Layer(),
                retained_pressure_endpoints={"direct": 0.5},
            )
        self.assertEqual([], hard.failures)
        self.assertIn(
            "stellar_sequences:endpoint_below_sequence_floor:direct:synthetic",
            availability.failures,
        )
        self.assertIn(
            "stellar_sequences:background_unavailable:1",
            availability.failures,
        )
        self.assertTrue(
            any(
                "invalid_failed_background_row" in failure
                for failure in outside_hard.failures
            )
        )

    def test_thermodynamic_validation_separates_core_from_diagnostics(self) -> None:
        def profile_row(
            case_id: str,
            epsilon: float,
            pressure: float,
            density: float,
            *,
            gamma: float = -1.0e200,
        ) -> dict[str, Any]:
            return {
                "case_id": case_id,
                "amplitude": "" if case_id == "direct" else 0.1,
                "delta_mev_fm3": "" if case_id == "direct" else 10.0,
                "epsilon_mev_fm3": epsilon,
                "pressure_mev_fm3": pressure,
                "cs2": 0.5,
                "delta_cs2": 0.0,
                "baryon_density_fm3": density,
                "effective_baryon_enthalpy_mev": 1000.0,
                "gamma_eff": gamma,
                "energy_per_baryon_minus_neutron_rest_mev": -10.0,
                "pressure_relative_to_direct": 1.0e200,
                "baryon_density_relative_to_direct": -1.0e200,
                "enthalpy_relative_to_direct": 1.0e200,
            }

        residual_columns = (
            "r_p_algebraic",
            "r_mu_algebraic",
            "r_p_independent",
            "r_p_independent_normalized",
            "r_mu_independent",
            "r_mu_independent_normalized",
            "r_c",
            "first_law_normalized",
            "dP_dEpsilon_independent",
            "mu_from_dEpsilon_dn_independent",
        )

        def residual_row(epsilon: float) -> dict[str, Any]:
            return {
                "case_id": "accepted-case",
                "amplitude": 0.1,
                "delta_mev_fm3": 10.0,
                "epsilon_mev_fm3": epsilon,
                **{column: 1.0e200 for column in residual_columns},
            }

        profiles = [
            profile_row("direct", 100.0, 1.0, 0.1),
            profile_row("direct", 200.0, 2.0, 0.2),
            profile_row("accepted-case", 100.0, 1.1, 0.11),
            profile_row("accepted-case", 200.0, 2.1, 0.21),
        ]
        residuals = [residual_row(100.0), residual_row(200.0)]
        with tempfile.TemporaryDirectory(prefix="thermo-validity-") as temporary:
            packet = Path(temporary)
            _write_json(
                packet / "raw_gate_report.json",
                {
                    "cases": {
                        "a0": {
                            "status": "accepted_raw_local_physics_gate",
                            "parameters": {"amplitude": 0.0},
                            "complete_proposed_retained_domain_mev_fm3": [
                                100.0,
                                200.0,
                            ],
                            "retained_domain": {
                                "endpoint_reason": (
                                    "direct_bsk24_causal_endpoint"
                                ),
                                "epsilon_min_mev_fm3": 100.0,
                                "epsilon_max_mev_fm3": 200.0,
                                "pressure_max_mev_fm3": 2.0,
                            },
                        },
                        "accepted-case": {
                            "status": "accepted_raw_local_physics_gate",
                            "parameters": {"amplitude": 0.1},
                            "complete_proposed_retained_domain_mev_fm3": [
                                100.0,
                                200.0,
                            ],
                        },
                    }
                },
            )
            _write_csv_rows(
                packet / "case_ledger.csv",
                [
                    {
                        "case_id": "accepted-case",
                        "retained_epsilon_max_mev_fm3": 200.0,
                        "retained_pressure_max_mev_fm3": 2.1,
                    }
                ],
            )
            _write_csv_rows(packet / "thermodynamic_profiles.csv", profiles)
            _write_csv_rows(packet / "thermodynamic_residuals.csv", residuals)
            layer = _Layer()
            _validate_thermodynamic_outputs(
                packet,
                accepted={"accepted-case"},
                layer=layer,
            )
            self.assertEqual([], layer.failures)

            invalid_profiles = [dict(row) for row in profiles]
            invalid_profiles[-1]["pressure_mev_fm3"] = 1.0
            _write_csv_rows(
                packet / "thermodynamic_profiles.csv", invalid_profiles
            )
            layer = _Layer()
            _validate_thermodynamic_outputs(
                packet,
                accepted={"accepted-case"},
                layer=layer,
            )
            self.assertTrue(
                any(
                    "nonmonotone_profile_coordinate" in failure
                    for failure in layer.failures
                )
            )

            zero_pressure_profiles = [dict(row) for row in profiles]
            zero_pressure_profiles[2]["pressure_mev_fm3"] = 0.0
            _write_csv_rows(
                packet / "thermodynamic_profiles.csv",
                zero_pressure_profiles,
            )
            layer = _Layer()
            _validate_thermodynamic_outputs(
                packet,
                accepted={"accepted-case"},
                layer=layer,
            )
            self.assertTrue(
                any(
                    "invalid_profile_core_state:accepted-case:0" in failure
                    for failure in layer.failures
                )
            )

            nonfinite_profiles = [dict(row) for row in profiles]
            nonfinite_profiles[-1]["gamma_eff"] = "nan"
            _write_csv_rows(
                packet / "thermodynamic_profiles.csv", nonfinite_profiles
            )
            nonfinite_residuals = [dict(row) for row in residuals]
            nonfinite_residuals[-1]["r_c"] = "inf"
            _write_csv_rows(
                packet / "thermodynamic_residuals.csv", nonfinite_residuals
            )
            layer = _Layer()
            _validate_thermodynamic_outputs(
                packet,
                accepted={"accepted-case"},
                layer=layer,
            )
            self.assertTrue(
                any(
                    "nonfinite_or_missing_profile_value" in failure
                    for failure in layer.failures
                )
            )
            self.assertTrue(
                any(
                    "nonfinite_or_missing_residual_value" in failure
                    for failure in layer.failures
                )
            )

    def test_v2_raw_profiles_require_complete_structured_evidence(self) -> None:
        layer = _Layer()
        _validate_raw_gate_profiles(
            [{"case_id": "accepted-case"}],
            raw_gate={
                "schema_id": RAW_GATE_SCHEMA,
                "cases": {"accepted-case": {}},
            },
            raw_schema=RAW_GATE_SCHEMA,
            accepted={"accepted-case"},
            rejected=set(),
            layer=layer,
        )
        self.assertTrue(
            any("missing_v2_columns" in failure for failure in layer.failures)
        )

    def test_bsk24_raw_profiles_can_use_physical_alias_identities(self) -> None:
        physical_id = "bsk24-baseline-physical"
        report = {
            "parameters": {
                "amplitude": 0.0,
                "epsilon0_mev_fm3": 500.0,
                "sigma_mev_fm3": 300.0,
                "delta_mev_fm3": 175.0,
            },
            "complete_proposed_retained_domain_mev_fm3": [100.0, 200.0],
            "dense_grid_points": 2,
            "status": "accepted_raw_local_physics_gate",
            "finite_values": True,
        }
        physical_raw_gate = {
            "schema_id": RAW_GATE_SCHEMA,
            "cases": {physical_id: report},
            "accepted_case_ids": [physical_id],
            "rejected_case_ids": [],
        }
        logical_raw_gate = {
            "schema_id": RAW_GATE_SCHEMA,
            "executed_before_reconstruction_and_TOV": True,
            "complete_raw_proposal_assessment_authoritative": True,
            "selected_retained_domain_authoritative": True,
            "selected_domain_policy": (
                "prefix_through_first_continuous_cs2_equals_one"
            ),
            "accepted_case_ids": [],
            "rejected_case_ids": [],
            "hard_rejected_case_ids": [],
            "unresolved_case_ids": [],
            "cases": {},
        }
        rows = [
            {
                "case_id": physical_id,
                "amplitude": "0.0",
                "epsilon0_mev_fm3": "500.0",
                "sigma_mev_fm3": "300.0",
                "delta_mev_fm3": "175.0",
                "epsilon_mev_fm3": str(epsilon),
                "window": "1.0",
                "gaussian": "1.0",
                "delta_cs2": "0.0",
                "direct_pressure_mev_fm3": str(pressure),
                "delta_pressure_mev_fm3": "0.0",
                "raw_pressure_mev_fm3": str(pressure),
                "raw_cs2": "0.5",
                "gate_status": "accepted_raw_local_physics_gate",
            }
            for epsilon, pressure in ((100.0, 1.0), (200.0, 2.0))
        ]
        with tempfile.TemporaryDirectory(
            prefix="bsk24-physical-profile-"
        ) as temporary:
            packet = Path(temporary)
            _write_csv_rows(packet / "raw_gate_profiles.csv", rows)
            layer = _Layer()
            _validate_case_consistency(
                packet,
                matter_model="bsk24",
                case_plan=[],
                case_ledger=[],
                raw_gate=logical_raw_gate,
                accepted_rejected=None,
                metadata=None,
                layer=layer,
                bsk24_raw_profile_evidence=(
                    physical_raw_gate,
                    {physical_id},
                    set(),
                ),
            )
        self.assertEqual([], layer.failures)

    def test_case_lifecycle_recomputes_student_and_rejection_semantics(self) -> None:
        case_id = "accepted-case"
        report = {
            "status": "accepted_raw_local_physics_gate",
            "complete_raw_proposal_assessed": True,
            "finite_values": True,
            "positive_energy_density": True,
            "positive_pressure": True,
            "complete_raw_proposal_mechanically_stable": True,
            "complete_raw_pressure_numerically_usable": True,
            "strictly_monotone_pressure_implied": True,
            "selected_retained_domain_authoritative": True,
            "selected_retained_domain_passed": True,
            "complete_raw_proposal_causal_through_direct_endpoint": True,
            "complete_proposed_retained_domain_mev_fm3": [100.0, 200.0],
            "continuous_resolution_certificate": {
                "status": "resolved_geometry_aware_sampling"
            },
            "raw_pressure_reconstruction_certificate": {
                "status": "resolved_strictly_increasing_raw_pressure"
            },
            "retained_tabulation_resolution_certificate": {
                "status": "resolved_tabulation_resolution"
            },
            "first_failure": None,
            "retained_domain": {
                "endpoint_reason": "direct_bsk24_causal_endpoint",
                "epsilon_min_mev_fm3": 100.0,
                "epsilon_max_mev_fm3": 200.0,
                "pressure_max_mev_fm3": 2.0,
                "cs2_at_endpoint": 0.9,
                "first_causal_crossing": None,
                "passed": True,
                "resolution_certified": True,
            },
        }
        raw_gate = {
            "schema_id": RAW_GATE_SCHEMA,
            "executed_before_reconstruction_and_TOV": True,
            "complete_raw_proposal_assessment_authoritative": True,
            "selected_retained_domain_authoritative": True,
            "selected_domain_policy": (
                "prefix_through_first_continuous_cs2_equals_one"
            ),
            "accepted_case_ids": [case_id],
            "rejected_case_ids": [],
            "hard_rejected_case_ids": [],
            "unresolved_case_ids": [],
            "cases": {case_id: report},
        }
        ledger = [
            {
                "case_id": case_id,
                "status": "accepted",
                "acceptance_domain": "full_retained_domain",
                "raw_gate_status": "accepted_raw_local_physics_gate",
                "full_domain_gate_status": (
                    "assessed_causal_through_direct_endpoint"
                ),
                "selected_domain_status": "accepted_selected_retained_domain",
                "complete_raw_proposal_causal_through_direct_endpoint": "True",
                "retained_epsilon_max_mev_fm3": "200.0",
                "retained_pressure_max_mev_fm3": "2.0",
                "retained_endpoint_reason": "direct_bsk24_causal_endpoint",
                "requested_fixed_masses_status": (
                    "all_requested_fixed_masses_succeeded"
                ),
                "maximum_mass_availability_status": (
                    "unavailable_unresolved_no_turning_point_before_eos_endpoint"
                ),
                "student_view_eligibility_status": "bogus",
                "rejection_reason": '{"bogus": true}',
                "pressure_reconstruction": "completed",
                "stellar_calculation": "completed",
                "clipping_or_repair": "none",
            }
        ]
        with tempfile.TemporaryDirectory(prefix="case-validity-") as temporary:
            layer = _Layer()
            _validate_case_consistency(
                Path(temporary),
                case_plan=[{"case_id": case_id}],
                case_ledger=ledger,
                raw_gate=raw_gate,
                accepted_rejected=None,
                metadata=None,
                layer=layer,
            )
        self.assertTrue(
            any("student_eligibility_mismatch" in item for item in layer.failures)
        )
        self.assertTrue(
            any("accepted_has_rejection_reason" in item for item in layer.failures)
        )

    def test_final_lifecycle_is_recomputed_from_saved_availability_rows(self) -> None:
        configuration = {
            "stellar_enabled": True,
            "fixed_masses_msun": [1.4],
            "tov_stages": [
                {"name": "reporting", "sequence_points": 5}
            ],
        }
        ledger = {
            "case_id": "accepted-case",
            "status": "accepted",
            "stellar_calculation": "incomplete_or_failed",
            "pressure_reconstruction": "completed",
            "requested_fixed_masses_status": (
                "all_requested_fixed_masses_succeeded"
            ),
            "maximum_mass_availability_status": (
                "resolved_bracketed_and_refined"
            ),
            "student_view_eligibility_status": (
                "eligible_all_requested_fixed_masses_succeeded"
            ),
        }
        fixed = {
            "case_id": "accepted-case",
            "stage": "reporting",
            "target_mass_msun": 1.4,
            "status": "unavailable_not_bracketed",
        }
        maximum = {
            "case_id": "accepted-case",
            "stage": "reporting",
            "maximum_mass_availability_status": (
                "unavailable_unresolved_no_turning_point_before_eos_endpoint"
            ),
        }
        with tempfile.TemporaryDirectory(prefix="lifecycle-recompute-") as temporary:
            packet = Path(temporary)
            _write_csv_row(packet / "case_ledger.csv", ledger)
            _write_csv_row(packet / "fixed_mass_observables.csv", fixed)
            _write_csv_row(packet / "maximum_mass_screening.csv", maximum)
            layer = _Layer()
            _validate_final_lifecycle(
                packet,
                configuration,
                {"accepted-case"},
                layer,
            )
        self.assertTrue(
            any("fixed_mass_status_mismatch" in item for item in layer.failures)
        )
        self.assertTrue(
            any("maximum_mass_status_mismatch" in item for item in layer.failures)
        )
        self.assertTrue(
            any("student_eligibility_mismatch" in item for item in layer.failures)
        )

    def test_cfl_final_lifecycle_resolves_baseline_alias_to_direct_rows(self) -> None:
        physical_id = "cfl-baseline-physical"
        logical_id = "cfl-logical-a0"
        configuration = {
            "matter_model": "cfl",
            "zero_amplitude_physical_case_id": physical_id,
            "stellar_enabled": True,
            "background_tov_requested": True,
            "fixed_mass_background_requested": True,
            "fixed_masses_msun": [1.4],
            "tov_stages": [{"name": "reporting", "sequence_points": 1}],
        }
        ledger = {
            "case_id": logical_id,
            "physical_case_id": physical_id,
            "status": "accepted",
            "stellar_calculation": "completed",
            "pressure_reconstruction": "completed",
            "requested_fixed_masses_status": (
                "all_requested_fixed_masses_succeeded"
            ),
            "maximum_mass_availability_status": (
                "resolved_bracketed_and_refined"
            ),
            "student_view_eligibility_status": (
                "eligible_all_requested_fixed_masses_succeeded"
            ),
        }
        sequence = {
            "case_id": "direct",
            "stage": "reporting",
            "calculation_status": "success",
            "tidal_status": "validated_lambda_validation_v1",
            "k2": 0.09,
            "Lambda": 400.0,
        }
        fixed = {
            "case_id": "direct",
            "stage": "reporting",
            "target_mass_msun": 1.4,
            "status": "bracketed_and_solved",
            "tidal_status": "validated_lambda_validation_v1",
            "k2": 0.09,
            "lambda_dimensionless": 400.0,
        }
        maximum = {
            "case_id": "direct",
            "stage": "reporting",
            "maximum_mass_availability_status": (
                "resolved_bracketed_and_refined"
            ),
        }
        with tempfile.TemporaryDirectory(prefix="cfl-lifecycle-") as temporary:
            packet = Path(temporary)
            _write_csv_row(packet / "case_ledger.csv", ledger)
            _write_csv_row(packet / "stellar_sequences.csv", sequence)
            _write_csv_row(packet / "fixed_mass_observables.csv", fixed)
            _write_csv_row(packet / "maximum_mass_screening.csv", maximum)
            layer = _Layer()
            _validate_final_lifecycle(
                packet,
                configuration,
                {logical_id},
                layer,
                expected_stellar_case_ids={"direct"},
            )
        self.assertEqual([], layer.failures)

    def test_unresolved_maximum_has_unavailable_threshold_not_failure(self) -> None:
        configuration = {"tov_stages": [{"name": "synthetic"}]}
        row = _unresolved_maximum_row()
        with tempfile.TemporaryDirectory(prefix="maximum-unavailable-") as temporary:
            packet = Path(temporary)
            _write_csv_row(packet / "maximum_mass_screening.csv", row)
            _write_json(
                packet / "maximum_mass_reports.json",
                _unresolved_maximum_report(),
            )
            hard = _Layer()
            availability = _Layer()
            _validate_maximum_mass_artifacts(
                packet,
                configuration=configuration,
                accepted=set(),
                layer=hard,
                availability=availability,
                retained_pressure_endpoints={"direct": 100.0},
            )
            self.assertEqual([], hard.failures)
            self.assertEqual(
                ["maximum_mass:unavailable:1"], availability.failures
            )

            contradictory = _unresolved_maximum_report()
            contradictory_case = contradictory["cases"]["direct:synthetic"]
            contradictory_case["eos_endpoint"] = {
                "pressure_mev_fm3": 999.0,
                "reached_by_search": False,
                "limitation": "contradictory_endpoint_claim",
            }
            contradictory_case["convergence"]["solver_call_count"] = 1
            _write_json(
                packet / "maximum_mass_reports.json",
                contradictory,
            )
            corrupt_hard = _Layer()
            _validate_maximum_mass_artifacts(
                packet,
                configuration=configuration,
                accepted=set(),
                layer=corrupt_hard,
                availability=_Layer(),
                retained_pressure_endpoints={"direct": 100.0},
            )
            self.assertTrue(
                any(
                    "malformed_json_evidence" in failure
                    and "eos_endpoint_csv_json" in failure
                    and "convergence_summary" in failure
                    for failure in corrupt_hard.failures
                )
            )

            evidence = _maximum_mass_group_evidence([row])
            self.assertEqual(0, evidence["mass_threshold_fail_count"])
            self.assertEqual(1, evidence["mass_threshold_unavailable_count"])

            _write_json(
                packet / "maximum_mass_reports.json",
                _unresolved_maximum_report(),
            )
            endpoint_mismatch = _Layer()
            _validate_maximum_mass_artifacts(
                packet,
                configuration=configuration,
                accepted=set(),
                layer=endpoint_mismatch,
                availability=_Layer(),
                retained_pressure_endpoints={"direct": 99.0},
            )
            self.assertTrue(
                any(
                    "retained_endpoint_mismatch" in failure
                    for failure in endpoint_mismatch.failures
                )
            )

            contaminated = dict(
                row, passes_maximum_mass_threshold=False
            )
            _write_csv_row(
                packet / "maximum_mass_screening.csv", contaminated
            )
            hard = _Layer()
            availability = _Layer()
            _validate_maximum_mass_artifacts(
                packet,
                configuration=configuration,
                accepted=set(),
                layer=hard,
                availability=availability,
            )
            self.assertTrue(
                any(
                    "malformed_unresolved_row" in failure
                    for failure in hard.failures
                )
            )

    def test_resolved_maximum_requires_one_consistent_turning_bracket(self) -> None:
        configuration = {"tov_stages": [{"name": "synthetic"}]}
        lower_pressure, middle_pressure, upper_pressure = 10.0, 20.0, 30.0
        lower_mass, middle_mass, upper_mass = 1.8, 2.1, 2.0
        bracket_left = (middle_mass - lower_mass) / (
            middle_pressure - lower_pressure
        )
        bracket_right = (upper_mass - middle_mass) / (
            upper_pressure - middle_pressure
        )
        maximum_pressure, maximum_mass = 22.0, 2.11
        maximum_left = (maximum_mass - lower_mass) / (
            maximum_pressure - lower_pressure
        )
        maximum_right = (upper_mass - maximum_mass) / (
            upper_pressure - maximum_pressure
        )

        def model(pressure: float, mass: float) -> dict[str, float]:
            return {
                "central_pressure_mev_fm3": pressure,
                "mass_msun": mass,
                "radius_km": 11.0,
                "central_energy_density_mev_fm3": 500.0 + pressure,
                "central_sound_speed_squared": 0.6,
            }

        bracket = {
            "lower_pressure_mev_fm3": lower_pressure,
            "middle_pressure_mev_fm3": middle_pressure,
            "upper_pressure_mev_fm3": upper_pressure,
            "lower_mass_msun": lower_mass,
            "middle_mass_msun": middle_mass,
            "upper_mass_msun": upper_mass,
            "left_dM_dPc_secant": bracket_left,
            "right_dM_dPc_secant": bracket_right,
        }
        maximum_model = model(maximum_pressure, maximum_mass)
        sampled = [
            model(lower_pressure, lower_mass),
            model(middle_pressure, middle_mass),
            maximum_model,
            model(upper_pressure, upper_mass),
        ]
        stable = sampled[:3]
        status = "resolved_unique_turning_point_local_sequence_refinement"
        row = {
            "case_id": "direct",
            "stage": "synthetic",
            "status": status,
            "maximum_mass_resolved": True,
            "maximum_mass_availability_status": "resolved_bracketed_and_refined",
            "maximum_mass_msun": maximum_mass,
            "maximum_mass_threshold_msun": 2.0,
            "passes_maximum_mass_threshold": True,
            "central_pressure_mev_fm3": maximum_pressure,
            "central_energy_density_mev_fm3": maximum_model[
                "central_energy_density_mev_fm3"
            ],
            "central_sound_speed_squared": 0.6,
            "radius_km": 11.0,
            "turning_point_count": 1,
            "positive_left_secant": maximum_left,
            "negative_right_secant": maximum_right,
            "eos_endpoint_pressure_mev_fm3": 100.0,
            "endpoint_limitation": "",
            "refinement_status": "converged_local_bounded_log_pressure",
            "sampled_sequence_model_count": 3,
            "local_background_solver_call_count": 1,
            "tidal_solver_calls_for_maximum_mass": 0,
        }
        report = {
            "schema_id": "bsk24_maximum_mass_reports_v2",
            "cases": {
                "direct:synthetic": {
                    "schema_id": "tov_resolved_maximum_mass_v2",
                    "status": status,
                    "maximum_mass_resolved": True,
                    "decision_basis": (
                        "refined_positive_to_negative_dM_dPc_turning_point"
                    ),
                    "sampled_argmax_is_maximum_mass": False,
                    "maximum_mass_threshold_msun": 2.0,
                    "passes_maximum_mass_threshold": True,
                    "maximum_mass_msun": maximum_mass,
                    "central_pressure_mev_fm3": maximum_pressure,
                    "central_energy_density_mev_fm3": maximum_model[
                        "central_energy_density_mev_fm3"
                    ],
                    "central_sound_speed_squared": 0.6,
                    "radius_km": 11.0,
                    "turning_point_count": 1,
                    "turning_point_brackets": [bracket],
                    "selected_bracket": bracket,
                    "positive_left_secant": maximum_left,
                    "negative_right_secant": maximum_right,
                    "stable_branch_extent": {
                        "model_count": len(stable),
                        "maximum_central_pressure_mev_fm3": maximum_pressure,
                        "models": stable,
                    },
                    "sampled_models": sampled,
                    "eos_endpoint": {
                        "pressure_mev_fm3": 100.0,
                        "reached_by_search": False,
                        "limitation": None,
                    },
                    "convergence": {
                        "refinement_status": (
                            "converged_local_bounded_log_pressure"
                        ),
                        "refinement_iterations": 1,
                        "global_refinement_rounds": 0,
                        "solver_call_count": 1,
                        "solver_failure_count": 0,
                        "solver_failures": [],
                    },
                    "tidal_calculations_performed": 0,
                }
            },
        }
        with tempfile.TemporaryDirectory(prefix="maximum-resolved-") as temporary:
            packet = Path(temporary)
            _write_csv_row(packet / "maximum_mass_screening.csv", row)
            _write_json(packet / "maximum_mass_reports.json", report)
            hard = _Layer()
            availability = _Layer()
            _validate_maximum_mass_artifacts(
                packet,
                configuration=configuration,
                accepted=set(),
                layer=hard,
                availability=availability,
                retained_pressure_endpoints={"direct": 100.0},
            )
            self.assertEqual([], hard.failures)
            self.assertEqual([], availability.failures)

            corrupt = json.loads(json.dumps(report))
            corrupt_case = corrupt["cases"]["direct:synthetic"]
            corrupt_case["turning_point_brackets"].append(dict(bracket))
            corrupt_case["turning_point_count"] = 2
            _write_json(packet / "maximum_mass_reports.json", corrupt)
            corrupt_hard = _Layer()
            _validate_maximum_mass_artifacts(
                packet,
                configuration=configuration,
                accepted=set(),
                layer=corrupt_hard,
                availability=_Layer(),
                retained_pressure_endpoints={"direct": 100.0},
            )
            self.assertTrue(
                any(
                    "resolved_scientific_evidence" in failure
                    for failure in corrupt_hard.failures
                )
            )



if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from contextlib import contextmanager
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
from eos_generation._internal.planning import prepare_bsk24_trial
from eos_generation._internal.status import build_bsk24_trial_status
from eos_generation.experiment import (
    ExperimentSettings,
    load_experiment,
    plan_experiment,
    run_experiment,
    validate_experiment,
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


if __name__ == "__main__":
    unittest.main()

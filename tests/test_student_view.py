from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from eos_generation._internal import student_view as student_view_module
from eos_generation._internal.student_view import create_student_view


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seal_child(packet: Path) -> None:
    manifest = packet / "SHA256SUMS.txt"
    files = sorted(
        path for path in packet.rglob("*") if path.is_file() and path != manifest
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(packet).as_posix()}" for path in files
    ]
    (packet / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _seal_aggregate(experiment: Path, children: tuple[str, ...]) -> None:
    files = {
        name: experiment / name
        for name in (
            "experiment.json",
            "experiment_config.json",
            "reproduction_plan.json",
            "reviewed_plan.json",
        )
    }
    files.update(
        {
            f"{child}/SHA256SUMS.txt": experiment / child / "SHA256SUMS.txt"
            for child in children
        }
    )
    (experiment / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(path)}  {relative}\n"
            for relative, path in sorted(files.items())
        ),
        encoding="utf-8",
    )


def _make_experiment(
    root: Path,
    *,
    session: str = "session",
    child_count: int = 1,
    stellar: bool = True,
) -> tuple[Path, dict[str, object]]:
    experiment = root / "runs" / session / "experiment_abc123"
    children = tuple(f"geometry_{index:03d}" for index in range(1, child_count + 1))
    experiment.mkdir(parents=True)
    settings_hash = "a" * 64
    _write_json(
        experiment / "experiment.json",
        {
            "schema_id": "eos_generation_experiment_v1",
            "status": "complete",
            "settings": {"calculation": "stellar" if stellar else "thermodynamics"},
            "settings_hash": settings_hash,
            "child_packets": list(children),
        },
    )
    for name in (
        "experiment_config.json",
        "reproduction_plan.json",
        "reviewed_plan.json",
    ):
        _write_json(experiment / name, {})

    for index, child in enumerate(children, start=1):
        packet = experiment / child
        (packet / "plots").mkdir(parents=True)
        (packet / "case_ledger.csv").write_text(
            "case_id,amplitude,epsilon0_mev_fm3,sigma_mev_fm3,"
            "delta_mev_fm3,status,reason\n"
            "case_0001,-0.1,300,50,20,accepted,\n",
            encoding="utf-8",
        )
        (packet / "thermodynamic_profiles.csv").write_text(
            "case_id,epsilon_mev_fm3,pressure_mev_fm3,cs2\n"
            f"direct,100,{index},0.2\n"
            f"case_0001,100,{index + 0.1},0.19\n",
            encoding="utf-8",
        )
        (packet / "raw_gate_profiles.csv").write_text(
            "case_id,epsilon_mev_fm3,raw_cs2,gate_status\n"
            "case_0001,100,0.2,accepted\n",
            encoding="utf-8",
        )
        if stellar:
            (packet / "stellar_sequences.csv").write_text(
                "case_id,mass_msun,radius_km\ncase_0001,1.4,12\n",
                encoding="utf-8",
            )
            (packet / "fixed_mass_observables.csv").write_text(
                "case_id,target_mass_msun,radius_km,status\n"
                "case_0001,1.4,12,available\n",
                encoding="utf-8",
            )
            (packet / "maximum_mass_screening.csv").write_text(
                "case_id,maximum_mass_msun,status\ncase_0001,2.0,resolved\n",
                encoding="utf-8",
            )
        (packet / "plots" / "pressure_response.png").write_bytes(
            b"\x89PNG\r\n\x1a\nsynthetic"
        )
        _seal_child(packet)
    _seal_aggregate(experiment, children)
    report: dict[str, object] = {
        "schema_id": "eos_generation_validation_v1",
        "status": "pass",
        "experiment_path": str(experiment.resolve(strict=False)),
        "child_packet_count": len(children),
        "children": [{"status": "pass"} for _ in children],
    }
    return experiment, report


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class StudentViewTests(unittest.TestCase):
    def test_copies_saved_artifacts_without_solver_calls_or_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, report = _make_experiment(root, child_count=2)
            before = _snapshot(experiment)
            destination = experiment.parent / "STUDENT_VIEW"
            forbidden_modules = {
                "eos_generation._internal.execution": None,
                "eos_generation._internal.stellar": None,
                "eos_generation.reporting.plot_orchestration": None,
            }
            with patch.dict(sys.modules, forbidden_modules):
                view = create_student_view(
                    experiment,
                    validation_report=report,
                    repository_root=root,
                    destination=destination,
                )
            self.assertEqual(before, _snapshot(experiment))
            self.assertEqual(destination.resolve(strict=False), view.path)
            for child in ("geometry_001", "geometry_002"):
                self.assertTrue(
                    (view.primary_data / child / "thermodynamic_profiles.csv").is_file()
                )
                self.assertTrue(
                    (
                        view.optional_diagnostics
                        / child
                        / "raw_gate_profiles.csv"
                    ).is_file()
                )
            self.assertEqual(destination.parent / "plots", view.plots)
            self.assertFalse((view.path / "02_PLOTS").exists())
            self.assertFalse(any(view.path.rglob("*.png")))
            self.assertFalse(any(view.path.rglob("*.json")))
            readme = view.readme.read_text(encoding="utf-8")
            self.assertIn("derived, non-authoritative", readme)
            self.assertIn("Canonical configuration hash: `" + "a" * 64, readme)
            self.assertIn("thermodynamic_profiles.csv", readme)
            self.assertIn("sibling `plots/` folder", readme)
            self.assertIn("stellar_sequences.csv", readme)
            self.assertIn("A CSV row is not a new EoS", readme)
            self.assertIn("`case_id = direct`", readme)
            self.assertIn("one distinct deformed EoS", readme)
            self.assertIn(
                "copy both `case_ledger.csv` and `thermodynamic_profiles.csv`",
                readme,
            )
            self.assertIn("Do not compare cases by spreadsheet row number", readme)
            dictionary = view.data_dictionary.read_text(encoding="utf-8")
            self.assertIn("`epsilon_mev_fm3`", dictionary)
            self.assertIn("`mass_msun`", dictionary)
            self.assertIn(
                "One row represents one sampled energy-density point",
                dictionary,
            )
            self.assertIn("Do not use row numbers as scientific identifiers", dictionary)

    def test_rejects_failed_or_incomplete_sources_and_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, report = _make_experiment(root)
            destination = experiment.parent / "STUDENT_VIEW"
            failed = dict(report)
            failed["status"] = "fail"
            with self.assertRaisesRegex(ValueError, "passing experiment validation"):
                create_student_view(
                    experiment,
                    validation_report=failed,
                    repository_root=root,
                    destination=destination,
                )

            metadata_path = experiment / "experiment.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["status"] = "running"
            _write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, "completed authoritative experiment"):
                create_student_view(
                    experiment,
                    validation_report=report,
                    repository_root=root,
                    destination=destination,
                )

            metadata["status"] = "complete"
            _write_json(metadata_path, metadata)
            _seal_aggregate(experiment, ("geometry_001",))

            profile = experiment / "geometry_001" / "thermodynamic_profiles.csv"
            original_profile = profile.read_bytes()
            profile.write_bytes(original_profile + b"tampered\n")
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                create_student_view(
                    experiment,
                    validation_report=report,
                    repository_root=root,
                    destination=destination,
                )
            profile.write_bytes(original_profile)
            _seal_child(experiment / "geometry_001")
            _seal_aggregate(experiment, ("geometry_001",))
            create_student_view(
                experiment,
                validation_report=report,
                repository_root=root,
                destination=destination,
            )
            with self.assertRaises(FileExistsError):
                create_student_view(
                    experiment,
                    validation_report=report,
                    repository_root=root,
                    destination=destination,
                )

    def test_transient_permission_error_retries_same_volume_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, report = _make_experiment(root)
            destination = experiment.parent / "STUDENT_VIEW"
            original_rename = student_view_module.os.rename
            attempts: list[tuple[Path, Path]] = []

            def transient_rename(stage: Path, target: Path) -> None:
                source = Path(stage)
                published = Path(target)
                attempts.append((source, published))
                self.assertEqual(source.parent, published.parent)
                if len(attempts) == 1:
                    raise PermissionError(
                        13, "synthetic transient Windows sharing violation"
                    )
                original_rename(source, published)

            with (
                patch.object(
                    student_view_module.os,
                    "rename",
                    side_effect=transient_rename,
                ) as rename,
                patch.object(student_view_module.time, "sleep") as sleep,
            ):
                view = create_student_view(
                    experiment,
                    validation_report=report,
                    repository_root=root,
                    destination=destination,
                )

            self.assertEqual(destination.resolve(strict=False), view.path)
            self.assertTrue(destination.is_dir())
            self.assertEqual(2, rename.call_count)
            sleep.assert_called_once_with(
                student_view_module._PUBLICATION_RETRY_DELAYS_SECONDS[0]
            )
            self.assertFalse(
                any(destination.parent.glob(f".{destination.name}.*.tmp"))
            )

    def test_target_created_during_retry_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, report = _make_experiment(root)
            destination = experiment.parent / "STUDENT_VIEW"

            def racing_rename(_stage: Path, target: Path) -> None:
                published = Path(target)
                published.mkdir()
                (published / "owner.txt").write_text(
                    "unrelated destination\n", encoding="utf-8"
                )
                raise PermissionError(
                    13, "synthetic Windows destination race"
                )

            with (
                patch.object(
                    student_view_module.os,
                    "rename",
                    side_effect=racing_rename,
                ) as rename,
                patch.object(student_view_module.time, "sleep") as sleep,
                self.assertRaisesRegex(FileExistsError, "already exists"),
            ):
                create_student_view(
                    experiment,
                    validation_report=report,
                    repository_root=root,
                    destination=destination,
                )

            self.assertEqual(1, rename.call_count)
            sleep.assert_not_called()
            self.assertEqual(
                "unrelated destination\n",
                (destination / "owner.txt").read_text(encoding="utf-8"),
            )
            self.assertFalse(
                any(destination.parent.glob(f".{destination.name}.*.tmp"))
            )

    def test_persistent_permission_error_is_bounded_and_cleans_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, report = _make_experiment(root)
            destination = experiment.parent / "STUDENT_VIEW"

            def denied_rename(_stage: Path, _target: Path) -> None:
                raise PermissionError(
                    13, "synthetic persistent Windows sharing violation"
                )

            with (
                patch.object(
                    student_view_module.os,
                    "rename",
                    side_effect=denied_rename,
                ) as rename,
                patch.object(student_view_module.time, "sleep") as sleep,
                self.assertRaisesRegex(PermissionError, "persistent Windows"),
            ):
                create_student_view(
                    experiment,
                    validation_report=report,
                    repository_root=root,
                    destination=destination,
                )

            delays = student_view_module._PUBLICATION_RETRY_DELAYS_SECONDS
            self.assertEqual(len(delays) + 1, rename.call_count)
            self.assertEqual([call(delay) for delay in delays], sleep.call_args_list)
            self.assertFalse(destination.exists())
            self.assertFalse(
                any(destination.parent.glob(f".{destination.name}.*.tmp"))
            )

    def test_non_permission_rename_error_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, report = _make_experiment(root)
            destination = experiment.parent / "STUDENT_VIEW"
            with (
                patch.object(
                    student_view_module.os,
                    "rename",
                    side_effect=OSError("synthetic non-permission failure"),
                ) as rename,
                patch.object(student_view_module.time, "sleep") as sleep,
                self.assertRaisesRegex(OSError, "non-permission"),
            ):
                create_student_view(
                    experiment,
                    validation_report=report,
                    repository_root=root,
                    destination=destination,
                )

            self.assertEqual(1, rename.call_count)
            sleep.assert_not_called()
            self.assertFalse(destination.exists())
            self.assertFalse(
                any(destination.parent.glob(f".{destination.name}.*.tmp"))
            )

    def test_checksums_and_file_order_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, first_report = _make_experiment(root, session="first")
            second, second_report = _make_experiment(root, session="second")
            first_view = create_student_view(
                first,
                validation_report=first_report,
                repository_root=root,
                destination=first.parent / "STUDENT_VIEW",
            )
            second_view = create_student_view(
                second,
                validation_report=second_report,
                repository_root=root,
                destination=second.parent / "STUDENT_VIEW",
            )
            self.assertEqual(_snapshot(first_view.path), _snapshot(second_view.path))

            lines = (first_view.path / "SHA256SUMS.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            relatives = [line.split("  ", 1)[1] for line in lines]
            self.assertEqual(sorted(relatives), relatives)
            expected = {
                path.relative_to(first_view.path).as_posix()
                for path in first_view.path.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS.txt"
            }
            self.assertEqual(expected, set(relatives))
            for line in lines:
                digest, relative = line.split("  ", 1)
                self.assertEqual(digest, _sha256(first_view.path / relative))

    def test_missing_non_applicable_stellar_tables_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, report = _make_experiment(root, stellar=False)
            view = create_student_view(
                experiment,
                validation_report=report,
                repository_root=root,
                destination=experiment.parent / "STUDENT_VIEW",
            )
            primary_names = {
                path.name for path in view.primary_data.rglob("*.csv")
            }
            self.assertEqual(
                {"case_ledger.csv", "thermodynamic_profiles.csv"}, primary_names
            )
            readme = view.readme.read_text(encoding="utf-8")
            self.assertIn(
                "Stellar sequences: not applicable or unavailable", readme
            )
            self.assertIn(
                "Fixed-mass observables: not applicable or unavailable", readme
            )

    def test_paths_must_remain_below_runs_and_outside_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment, report = _make_experiment(root)
            for destination in (
                root / "outside" / "STUDENT_VIEW",
                experiment / "STUDENT_VIEW",
            ):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    create_student_view(
                        experiment,
                        validation_report=report,
                        repository_root=root,
                        destination=destination,
                    )

            outside = root / "outside" / "experiment_abc123"
            outside.parent.mkdir()
            outside.mkdir()
            with self.assertRaisesRegex(ValueError, "outside runs"):
                create_student_view(
                    outside,
                    validation_report=report,
                    repository_root=root,
                )

if __name__ == "__main__":
    unittest.main()

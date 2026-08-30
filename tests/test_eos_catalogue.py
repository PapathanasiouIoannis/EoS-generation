from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "notebooks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CATALOGUE = load_script("eos_catalogue")
PLOTS = load_script("build_experiment_plots")


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def seal(packet):
    files = sorted(path for path in packet.iterdir() if path.name != "SHA256SUMS.txt")
    (packet / "SHA256SUMS.txt").write_text(
        "".join(f"{CATALOGUE.sha256(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )


def fixture(
    root,
    name,
    amplitude=0.12,
    precision="quick",
    center=500.0,
    physics="a",
    physical_zero_id=None,
    matter_model="bsk24",
):
    experiment = root / "runs" / name / "experiment_test"
    packet = experiment / "geometry_001"
    packet.mkdir(parents=True)
    if matter_model == "bsk24":
        source_hashes = {
            f"src/eos_generation/bsk24/{name}.py": physics * 64
            for name in ("baseline", "deformation", "reconstruction")
        }
        source_hashes["src/eos_generation/_internal/config.py"] = "b" * 64
    elif matter_model == "cfl":
        source_hashes = {
            f"src/eos_generation/cfl/{name}.py": physics * 64
            for name in ("baseline", "deformation", "reconstruction")
        }
        source_hashes["src/eos_generation/_internal/cfl_thermodynamics.py"] = "b" * 64
    else:
        raise ValueError(matter_model)
    source_hashes["src/eos_generation/reporting/plotting.py"] = ("c" if precision == "quick" else "d") * 64
    write_json(packet / "source_hashes.json", source_hashes)
    baseline_validation_status = (
        "literature_supported_frozen_design_contract"
        if matter_model == "cfl"
        else "pass"
    )
    write_json(packet / "metadata.json", {
        "packet_status": "complete", "configuration_hash": "e" * 64,
        "baseline_validation_status": baseline_validation_status,
        "identity_status": "pass",
    })
    write_json(packet / "run_state.json", {"packet_status": "complete", "configuration_hash": "e" * 64})
    config = {
        "amplitudes": [0.0, amplitude, -0.99], "epsilon_match_mev_fm3": 80.0,
        "epsilon0_mev_fm3": center, "sigma_mev_fm3": 500.0,
        "deltas_mev_fm3": [350.0], "diagnostic_delta_mev_fm3": 350.0,
        "fixed_masses_msun": [1.4], "tov_stages": [{"name": "final"}],
    }
    write_json(packet / "complete_configuration.json", config)
    ledger = []
    for case_id, value, status in (("zero", 0.0, "accepted"), ("case", amplitude, "accepted"), ("rejected", -0.99, "rejected")):
        row = {
            "case_id": case_id, "amplitude": value, "epsilon_match_mev_fm3": 80.0,
            "epsilon0_mev_fm3": center, "sigma_mev_fm3": 500.0,
            "delta_mev_fm3": 350.0, "status": status, "anchor_mode": "exploratory",
            "maximum_mass_availability_status": "unavailable_unresolved_no_turning_point_before_eos_endpoint",
            "rejection_reason": "" if status == "accepted" else '{"reason":"nonpositive_cs2"}',
        }
        if case_id == "zero" and physical_zero_id is not None:
            row.update({
                "physical_case_id": physical_zero_id,
                "is_physical_case_alias": True,
            })
        ledger.append(row)
    write_csv(packet / "case_ledger.csv", ledger)
    profiles = []
    for case_id in ("direct", physical_zero_id or "zero", "case"):
        for epsilon in (80.0, 200.0, 500.0):
            row = {
                "case_id": case_id,
                "epsilon_mev_fm3": epsilon,
                "pressure_mev_fm3": epsilon / 10,
                "cs2": 0.1,
            }
            if matter_model == "cfl":
                row["matter_model"] = "cfl"
            profiles.append(row)
    write_csv(packet / "thermodynamic_profiles.csv", profiles)
    sequences, fixed, maximum = [], [], []
    for case_id in ("direct", "zero", "case"):
        for stage in ("coarse", "final"):
            for index in range(3):
                failed = case_id == "case" and index == 0
                sequences.append({
                    "case_id": case_id, "stage": stage, "attempted_index": index,
                    "segment_id": 1 if case_id == "case" and index > 0 else 0,
                    "calculation_status": "failed" if failed else "success",
                    "Mass": "" if failed else 1.0 + index / 3,
                    "Radius": "" if failed else 13 - index,
                    "failure_reason": "surface not reached" if failed else "",
                    "is_sampled_peak": index == 2,
                })
            fixed.append({"case_id": case_id, "stage": stage, "status": "bracketed_and_solved", "target_mass_msun": 1.4, "radius_km": 12.0})
            maximum.append({"case_id": case_id, "stage": stage, "maximum_mass_resolved": "False", "maximum_mass_msun": "", "status": "unresolved_no_turning_point_before_eos_endpoint"})
    write_csv(packet / "stellar_sequences.csv", sequences)
    write_csv(packet / "fixed_mass_observables.csv", fixed)
    write_csv(packet / "maximum_mass_screening.csv", maximum)
    seal(packet)
    settings = {"calculation": "stellar", "precision": precision}
    if matter_model != "bsk24":
        settings["matter_model"] = matter_model
    write_json(experiment / "experiment.json", {
        "status": "complete", "settings": settings,
        "settings_hash": "f" * 64, "child_packets": ["geometry_001"],
        "child_configuration_hashes": ["e" * 64],
    })
    return experiment


def snapshot(folder):
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


class EosCatalogueTests(unittest.TestCase):
    def test_cli_root_and_no_clobber_publication_are_fail_closed(self):
        self.assertEqual(ROOT, CATALOGUE.trusted_repository_root(ROOT))
        self.assertEqual(ROOT, CATALOGUE.trusted_repository_root(str(ROOT)))
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            with self.assertRaisesRegex(ValueError, "reviewed checkout"):
                CATALOGUE.trusted_repository_root(parent)
            with self.assertRaisesRegex(ValueError, "must be absolute"):
                CATALOGUE.confined(Path("relative"), parent)
            with self.assertRaisesRegex(ValueError, "allowed parent"):
                CATALOGUE.confined(parent.with_name(f"{parent.name}-sibling"), parent)

            stage = parent / ".stage"
            target = parent / "target"
            stage.mkdir()
            target.mkdir()
            with self.assertRaises(FileExistsError):
                CATALOGUE.publish_directory(stage, target)
            self.assertTrue(stage.is_dir())

            target.rmdir()
            lock = parent / ".target.publish.lock"
            lock.write_text("occupied\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                CATALOGUE.publish_directory(stage, target)
            self.assertTrue(stage.is_dir())
            self.assertEqual("occupied\n", lock.read_text(encoding="utf-8"))

    def build(self, root, experiment, destination=None):
        with patch.object(CATALOGUE, "validate_source") as validation:
            result = CATALOGUE.build_eos_data(root, experiment, destination or experiment.parent / "EOS_DATA")
        validation.assert_called_once_with(experiment)
        return result

    def test_aliases_continue_across_signs_and_reuse_across_precision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positive = fixture(root, "positive")
            before = snapshot(positive)
            first = self.build(root, positive)
            aliases = CATALOGUE.csv_rows(Path(first["data_path"]) / "case_aliases.csv")
            by_id = {row["source_case_id"]: row for row in aliases}
            self.assertEqual("H000000", by_id["direct"]["eos_id"])
            self.assertEqual("H000000", by_id["zero"]["eos_id"])
            self.assertEqual("H000001", by_id["case"]["eos_id"])
            self.assertEqual("", by_id["rejected"]["eos_id"])
            self.assertEqual(before, snapshot(positive))
            first_registration = (root / "runs/eos_catalogue/registration_000001.json").read_bytes()
            strict = fixture(root, "strict", precision="strict")
            strict_result = self.build(root, strict)
            strict_aliases = CATALOGUE.csv_rows(Path(strict_result["data_path"]) / "case_aliases.csv")
            self.assertEqual(by_id["case"]["eos_id"], next(row["eos_id"] for row in strict_aliases if row["source_case_id"] == "case"))
            self.assertEqual(first["catalogue_id"], strict_result["catalogue_id"])
            self.assertEqual(1, len(list((root / "runs/eos_catalogue").glob("registration_*.json"))))
            negative = fixture(root, "negative", amplitude=-0.12, center=950.0)
            last = self.build(root, negative)
            last_rows = CATALOGUE.csv_rows(Path(last["data_path"]) / "case_aliases.csv")
            self.assertEqual("H000002", next(row["eos_id"] for row in last_rows if row["source_case_id"] == "case"))
            self.assertEqual("H000000", next(row["eos_id"] for row in last_rows if row["source_case_id"] == "zero"))
            self.assertEqual(first_registration, (root / "runs/eos_catalogue/registration_000001.json").read_bytes())

    def test_physical_a0_profile_id_maps_to_logical_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            physical_id = "bsk24_baseline_physical"
            experiment = fixture(
                root,
                "physical-a0",
                physical_zero_id=physical_id,
            )
            result = self.build(root, experiment)
            output = Path(result["data_path"])
            aliases = CATALOGUE.csv_rows(output / "case_aliases.csv")
            self.assertNotIn(
                physical_id,
                {row["source_case_id"] for row in aliases},
            )
            zero = next(
                row for row in aliases if row["source_case_id"] == "zero"
            )
            self.assertEqual(physical_id, zero["physical_case_id"])
            self.assertEqual("H000000", zero["eos_id"])
            physical_rows = [
                row
                for row in CATALOGUE.csv_rows(
                    output / "thermodynamic_profiles.csv"
                )
                if row["case_id"] == physical_id
            ]
            self.assertTrue(physical_rows)
            self.assertEqual(
                {"zero"},
                {row["source_case_id"] for row in physical_rows},
            )
            self.assertEqual(
                {"H000000"},
                {row["eos_id"] for row in physical_rows},
            )

    def test_cfl_and_bsk24_share_one_registry_without_identity_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hadronic = self.build(root, fixture(root, "hadronic"))
            cfl = self.build(root, fixture(root, "quark", matter_model="cfl"))
            hadronic_rows = CATALOGUE.csv_rows(Path(hadronic["data_path"]) / "case_aliases.csv")
            cfl_rows = CATALOGUE.csv_rows(Path(cfl["data_path"]) / "case_aliases.csv")
            self.assertEqual(
                {"H000000", "H000001"},
                {row["eos_id"] for row in hadronic_rows if row["eos_id"]},
            )
            self.assertEqual(
                {"C000000", "C000001"},
                {row["eos_id"] for row in cfl_rows if row["eos_id"]},
            )
            self.assertEqual({"bsk24"}, {row["matter_model"] for row in hadronic_rows})
            self.assertEqual({"cfl"}, {row["matter_model"] for row in cfl_rows})
            self.assertEqual(hadronic["catalogue_id"], cfl["catalogue_id"])
            entries, _, _, count = CATALOGUE.read_registry(root / "runs/eos_catalogue")
            self.assertEqual(4, len(entries))
            self.assertEqual(2, count)

    def test_model_specific_baseline_validation_status_is_fail_closed(self):
        wrong_statuses = {
            "bsk24": "literature_supported_frozen_design_contract",
            "cfl": "pass",
        }
        for matter_model, wrong_status in wrong_statuses.items():
            with self.subTest(matter_model=matter_model), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                experiment = fixture(root, matter_model, matter_model=matter_model)
                packet = experiment / "geometry_001"
                metadata = CATALOGUE.read_json(packet / "metadata.json")
                metadata["baseline_validation_status"] = wrong_status
                write_json(packet / "metadata.json", metadata)
                seal(packet)
                with self.assertRaisesRegex(
                    ValueError,
                    "packet identity, baseline, or completion check failed",
                ):
                    CATALOGUE.collect_sources(root, experiment)

    def test_authoritative_matter_model_must_match_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = fixture(root, "cfl-model-column", matter_model="cfl")
            packet = experiment / "geometry_001"
            path = packet / "thermodynamic_profiles.csv"
            rows = CATALOGUE.csv_rows(path)
            rows[0]["matter_model"] = "bsk24"
            write_csv(path, rows)
            seal(packet)
            with self.assertRaisesRegex(ValueError, "matter_model disagrees"):
                CATALOGUE.collect_sources(root, experiment)

    def test_cfl_direct_baseline_occurrence_belongs_only_to_declared_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = fixture(root, "multi-geometry-cfl", matter_model="cfl")
            owner = experiment / "geometry_001"
            nonowner = experiment / "geometry_002"
            owner_config = CATALOGUE.read_json(owner / "complete_configuration.json")
            owner_config["zero_amplitude_control_owner"] = True
            write_json(owner / "complete_configuration.json", owner_config)
            seal(owner)
            shutil.copytree(owner, nonowner)
            nonowner_config = CATALOGUE.read_json(
                nonowner / "complete_configuration.json"
            )
            nonowner_config["zero_amplitude_control_owner"] = False
            write_json(nonowner / "complete_configuration.json", nonowner_config)
            for name in (
                "thermodynamic_profiles.csv",
                "stellar_sequences.csv",
                "fixed_mass_observables.csv",
                "maximum_mass_screening.csv",
            ):
                rows = CATALOGUE.csv_rows(nonowner / name)
                write_csv(
                    nonowner / name,
                    [row for row in rows if row["case_id"] not in {"direct", "zero"}],
                )
            seal(nonowner)
            document = CATALOGUE.read_json(experiment / "experiment.json")
            # Deliberately list the non-owner first: the baseline representative
            # must still be the actual direct evaluation in the owner packet.
            document["child_packets"] = ["geometry_002", "geometry_001"]
            document["child_configuration_hashes"] = ["e" * 64, "e" * 64]
            write_json(experiment / "experiment.json", document)
            result = self.build(root, experiment)
            aliases = CATALOGUE.csv_rows(Path(result["data_path"]) / "case_aliases.csv")
            direct = [row for row in aliases if row["source_case_id"] == "direct"]
            self.assertEqual(1, len(direct))
            self.assertEqual("geometry_001", direct[0]["geometry_id"])
            catalogue_rows = CATALOGUE.csv_rows(
                Path(result["data_path"]) / "eos_catalogue.csv"
            )
            baseline = next(row for row in catalogue_rows if row["eos_id"] == "C000000")
            self.assertEqual("direct", baseline["source_case_id"])
            self.assertEqual("geometry_001", baseline["geometry_id"])

    def test_legacy_bsk24_registry_chain_accepts_a_versioned_cfl_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build(root, fixture(root, "legacy"))
            first = root / "runs/eos_catalogue/registration_000001.json"
            transaction = CATALOGUE.read_json(first)
            transaction.pop("sha256")
            transaction["schema_id"] = CATALOGUE.LEGACY_SCHEMA
            transaction["sha256"] = CATALOGUE.digest(transaction)
            write_json(first, transaction)
            result = self.build(root, fixture(root, "cfl", matter_model="cfl"))
            rows = CATALOGUE.csv_rows(Path(result["data_path"]) / "case_aliases.csv")
            self.assertEqual(
                {"C000000", "C000001"},
                {row["eos_id"] for row in rows if row["eos_id"]},
            )
            second = CATALOGUE.read_json(
                root / "runs/eos_catalogue/registration_000002.json"
            )
            self.assertEqual(CATALOGUE.SCHEMA, second["schema_id"])

    def test_values_failures_all_stages_and_original_ids_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = fixture(root, "data")
            result = self.build(root, experiment)
            output = Path(result["data_path"])
            for name in CATALOGUE.PRIMARY_TABLES:
                original = CATALOGUE.csv_rows(experiment / "geometry_001" / name)
                labelled = CATALOGUE.csv_rows(output / name)
                self.assertEqual(original, [{key: row[key] for key in original[0]} for row in labelled])
            maximum = CATALOGUE.csv_rows(output / "maximum_mass_screening.csv")
            self.assertTrue(all(row["maximum_mass_msun"] == "" for row in maximum))
            self.assertEqual({"coarse", "final"}, {row["stage"] for row in maximum})
            self.assertTrue(all(row["eos_id"] for row in maximum))
            for name, expected in CATALOGUE.manifest(output).items():
                self.assertEqual(expected, CATALOGUE.sha256(output / name))
            self.assertEqual(0, result["solver_calls"])

    def test_no_overwrite_and_invalid_source_never_assigns_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = fixture(root, "invalid")
            target = experiment.parent / "EOS_DATA"
            with patch.object(CATALOGUE, "validate_source", side_effect=ValueError("invalid packet")):
                with self.assertRaisesRegex(ValueError, "invalid packet"):
                    CATALOGUE.build_eos_data(root, experiment, target)
            self.assertFalse((root / "runs/eos_catalogue").exists())
            packet = experiment / "geometry_001"
            (packet / "case_ledger.csv").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.build(root, experiment)
            self.assertFalse((root / "runs/eos_catalogue").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = fixture(root, "valid")
            self.build(root, experiment)
            before = snapshot(root)
            with self.assertRaises(FileExistsError):
                self.build(root, experiment)
            self.assertEqual(before, snapshot(root))

    def test_bad_identity_and_unknown_ids_fail_before_registration(self):
        for corruption in ("identity", "unknown"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                experiment = fixture(root, "invalid")
                packet = experiment / "geometry_001"
                if corruption == "identity":
                    metadata = CATALOGUE.read_json(packet / "metadata.json")
                    metadata["identity_status"] = "fail"
                    write_json(packet / "metadata.json", metadata)
                else:
                    rows = CATALOGUE.csv_rows(packet / "stellar_sequences.csv")
                    rows[0]["case_id"] = "unknown"
                    write_csv(packet / "stellar_sequences.csv", rows)
                seal(packet)
                with self.assertRaises(ValueError):
                    self.build(root, experiment)
                self.assertFalse((root / "runs/eos_catalogue").exists())

    def test_physics_change_gets_new_identity_without_renumbering_old_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build(root, fixture(root, "one"))
            changed = self.build(root, fixture(root, "two", physics="9"))
            rows = CATALOGUE.csv_rows(Path(changed["data_path"]) / "case_aliases.csv")
            self.assertEqual({"H000002", "H000003"}, {row["eos_id"] for row in rows if row["eos_id"]})
            entries, _, _, count = CATALOGUE.read_registry(root / "runs/eos_catalogue")
            self.assertEqual(4, len(entries))
            self.assertEqual(2, count)

    def test_registry_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build(root, fixture(root, "one"))
            path = root / "runs/eos_catalogue/registration_000001.json"
            transaction = CATALOGUE.read_json(path)
            transaction["entries"][0]["eos_id"] = "H999999"
            write_json(path, transaction)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.build(root, fixture(root, "two", amplitude=-0.12))

    def test_concurrent_registration_is_unique_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = fixture(root, "source")
            definitions = CATALOGUE.collect_sources(root, experiment)["definitions"]
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: CATALOGUE.register_definitions(root, definitions), range(8)))
            self.assertEqual(1, len({result["catalogue_id"] for result in results}))
            entries, _, _, count = CATALOGUE.read_registry(root / "runs/eos_catalogue")
            self.assertEqual(2, len(entries))
            self.assertEqual(1, count)

    def test_alias_snapshot_is_bound_to_experiment_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = fixture(root, "one")
            result = self.build(root, experiment)
            output = Path(result["data_path"])
            self.assertTrue(CATALOGUE.load_aliases(root, experiment, output))
            other = fixture(root, "two")
            with self.assertRaisesRegex(ValueError, "different experiment"):
                CATALOGUE.load_aliases(root, other, output)
            with (experiment / "geometry_001/SHA256SUMS.txt").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                CATALOGUE.load_aliases(root, experiment, output)

    def test_output_paths_cannot_escape_runs_or_overlap_packets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = fixture(root, "one")
            for destination in (root / "outside", root / "runs", experiment, experiment / "EOS_DATA", root / "runs/eos_catalogue"):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    self.build(root, experiment, destination)
            self.assertFalse((root / "runs/eos_catalogue").exists())

    def test_presentation_retry_reuses_reserved_ids_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = fixture(root, "one")
            before = snapshot(experiment)
            with patch.object(CATALOGUE, "publish_directory", side_effect=PermissionError("synthetic publication failure")):
                with self.assertRaises(PermissionError):
                    self.build(root, experiment)
            registration = snapshot(root / "runs/eos_catalogue")
            result = self.build(root, experiment)
            self.assertEqual(registration, snapshot(root / "runs/eos_catalogue"))
            self.assertEqual(before, snapshot(experiment))
            self.assertEqual(2, result["unique_eos_count"])
            self.assertFalse(list(experiment.parent.glob(".eos_data_*")))

    def test_plot_builder_consumes_aliases_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = fixture(root, "plots")
            result = self.build(root, experiment)
            before = snapshot(experiment)
            captured = []

            def renderer(packet, figures, *, config):
                captured.append(config.eos_aliases)
                frame = CATALOGUE.csv_rows(packet / "stellar_sequences.csv")
                self.assertEqual({"H000000", "H000001"}, {row["eos_id"] for row in frame})
                return ()

            with patch.object(PLOTS, "render_dense_saved_figures", side_effect=renderer):
                plotted = PLOTS.build_experiment_plots(root, experiment, experiment.parent / "plots", eos_data_path=Path(result["data_path"]))
            self.assertTrue(plotted["friendly_eos_labels"])
            self.assertTrue(captured)
            self.assertEqual(before, snapshot(experiment))
            self.assertTrue((experiment.parent / "plots/case_aliases.csv").is_file())

    def test_small_plot_legends_use_aliases_and_restore_renderer_state(self):
        import pandas as pd
        from types import SimpleNamespace
        from eos_generation.reporting import _plotting_style, _plotting_stellar, plotting

        original = _plotting_style._style_rows
        threshold = _plotting_style._AMPLITUDE_COLORBAR_THRESHOLD
        frame = pd.DataFrame([
            {"case_id": "direct", "amplitude": 0.0, "delta_mev_fm3": 350.0},
            {"case_id": "case", "amplitude": 0.12, "delta_mev_fm3": 350.0},
        ])

        def render(packet, figures, *, config):
            self.assertEqual("H000001", _plotting_stellar._style_rows(frame)["case"]["label"])
            self.assertEqual("H000000 (BSk24)", _plotting_stellar._concise_mass_radius_legend_label("Direct BSk24"))
            raise RuntimeError("synthetic rendering failure")

        config = SimpleNamespace(eos_aliases={"direct": "H000000", "case": "H000001"})
        with patch.object(plotting, "render_trial_figures", side_effect=render):
            with self.assertRaisesRegex(RuntimeError, "synthetic rendering failure"):
                PLOTS.render_dense_saved_figures(Path("unused"), (), config=config)
        self.assertIs(original, _plotting_style._style_rows)
        self.assertIs(original, _plotting_stellar._style_rows)
        self.assertEqual(threshold, _plotting_style._AMPLITUDE_COLORBAR_THRESHOLD)


if __name__ == "__main__":
    unittest.main()

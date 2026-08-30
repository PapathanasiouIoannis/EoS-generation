from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "notebooks" / "build_combined_hadronic_dataset.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("build_combined_hadronic_dataset", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"could not load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _catalogue_frame(root: Path, run: str, rank: int, rows: list[dict]) -> pd.DataFrame:
    run_root = root / "runs" / run
    packet = run_root / "experiment_test" / "geometry_001"
    packet.mkdir(parents=True)
    records = []
    for row in rows:
        records.append(
            {
                "eos_id": row["eos_id"],
                "catalogue_id": "catalogue",
                "physical_model_key": row["key"],
                "packet_path": packet.relative_to(root).as_posix(),
                "geometry_id": "geometry_001",
                "source_case_id": row["case_id"],
                "case_id": row["case_id"],
                "amplitude": row["amplitude"],
                "status": "baseline" if row["amplitude"] == 0 else "accepted",
                "epsilon_match_mev_fm3": 80.0 if row["amplitude"] else float("nan"),
                "epsilon0_mev_fm3": 200.0 if row["amplitude"] else float("nan"),
                "sigma_mev_fm3": 150.0 if row["amplitude"] else float("nan"),
                "delta_mev_fm3": 175.0 if row["amplitude"] else float("nan"),
                "_source_rank": rank,
                "_source_run": run,
                "_run_root": str(run_root),
                "_eos_data": str(run_root / "EOS_DATA"),
                "_manifest_sha256": str(rank) * 64,
            }
        )
    return pd.DataFrame(records)


def _direct_curve_run(
    root: Path, run: str, case_id: str, amplitude: float
) -> Path:
    run_root = root / "runs" / run
    experiment = run_root / "experiment_test"
    packet = experiment / "geometry_001"
    packet.mkdir(parents=True)
    (packet / "complete_configuration.json").write_bytes(
        MODULE.catalogue.canonical(
            {
                "curve_only_output": True,
                "tov_stages": [{"name": "dataset_40"}],
            }
        )
        + b"\n"
    )
    pd.DataFrame(
        [
            {
                "case_id": case_id,
                "physical_case_id": case_id,
                "status": "accepted",
                "amplitude": amplitude,
                "epsilon_match_mev_fm3": 80.0,
                "epsilon0_mev_fm3": 450.0 if amplitude > 0 else 800.0,
                "sigma_mev_fm3": 500.0,
                "delta_mev_fm3": 225.0,
            }
        ]
    ).to_csv(packet / "case_ledger.csv", index=False)
    pd.DataFrame({"case_id": ["direct", case_id]}).to_csv(
        packet / "stellar_sequences.csv", index=False
    )
    MODULE._write_manifest(packet)
    (experiment / "experiment.json").write_bytes(
        MODULE.catalogue.canonical({"child_packets": ["geometry_001"]}) + b"\n"
    )
    (experiment / "SHA256SUMS.txt").write_text(
        "synthetic aggregate marker\n", encoding="utf-8"
    )
    return run_root


class CombinedHadronicDatasetTests(unittest.TestCase):
    def test_finalize_manifest_is_confined_to_repository_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            destination = runs / "derived"
            outside = root / "outside"
            for folder in (destination, outside):
                folder.mkdir(parents=True)
                (folder / "provenance.json").write_text(
                    '{"schema_id":"bsk24_combined_hadronic_ml_dataset_v1"}\n',
                    encoding="utf-8",
                )

            result = MODULE.finalize_manifest(root, destination)

            self.assertEqual(destination.resolve(), Path(result["destination"]))
            self.assertTrue((destination / "SHA256SUMS.txt").is_file())
            with self.assertRaisesRegex(ValueError, "allowed parent"):
                MODULE.finalize_manifest(root, outside)
            self.assertFalse((outside / "SHA256SUMS.txt").exists())

    def test_build_destination_rejects_a_symlink_escape_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            outside = root / "outside"
            runs.mkdir()
            outside.mkdir()
            linked_destination = runs / "linked-destination"
            try:
                linked_destination.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "allowed parent"):
                MODULE.build_combined_dataset(root, [], linked_destination)
            self.assertEqual([], list(outside.iterdir()))

    def test_manifest_rejects_linked_inputs_and_fixed_temp_symlinks(self) -> None:
        for link_name in ("outside-link.csv", ".SHA256SUMS.tmp"):
            with self.subTest(link_name=link_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                packet = root / "packet"
                packet.mkdir()
                (packet / "data.csv").write_text("value\n1\n", encoding="utf-8")
                sentinel = root / "sentinel.txt"
                sentinel.write_text("unchanged\n", encoding="utf-8")
                linked = packet / link_name
                try:
                    linked.symlink_to(sentinel)
                except OSError as exc:
                    self.skipTest(f"file symlinks are unavailable: {exc}")

                with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                    MODULE._write_manifest(packet)
                self.assertEqual("unchanged\n", sentinel.read_text(encoding="utf-8"))
                self.assertFalse((packet / "SHA256SUMS.txt").exists())

    def test_nested_declared_packet_and_formula_run_label_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _direct_curve_run(root, "positive", "p", 0.2)
            experiment = run / "experiment_test"
            document = MODULE.catalogue.read_json(experiment / "experiment.json")
            document["child_packets"] = ["nested/geometry_001"]
            (experiment / "experiment.json").write_bytes(
                MODULE.catalogue.canonical(document) + b"\n"
            )
            (experiment / "nested").mkdir()
            (experiment / "geometry_001").rename(
                experiment / "nested" / "geometry_001"
            )
            with patch.object(MODULE.catalogue, "validate_source"):
                with self.assertRaisesRegex(ValueError, "direct experiment child"):
                    MODULE._source_catalogue(root, run, 0)

        for label in ("=run", "+run", "-run", "@run", "run\nname"):
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, "unsafe source run label"
            ):
                MODULE._csv_label(label, "source run")

    def test_direct_curve_experiments_need_no_eos_data_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positive = _direct_curve_run(root, "positive", "p", 0.2)
            negative = _direct_curve_run(root, "negative", "n", -0.1)
            with patch.object(MODULE.catalogue, "validate_source") as validate:
                mapping, sources, duplicates = MODULE.build_name_mapping(
                    root, [positive, negative]
                )

            self.assertEqual(2, validate.call_count)
            self.assertEqual(1, duplicates)
            self.assertEqual(
                ["baseline", "positive", "negative"],
                mapping["regime"].tolist(),
            )
            self.assertTrue(
                all(
                    source["source_mode"] == "sealed_curve_experiment"
                    for source in sources
                )
            )
            self.assertTrue(
                all("eos_data_path" not in source for source in sources)
            )

    def test_deduplicates_baseline_and_assigns_compact_regime_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positive = _catalogue_frame(
                root,
                "positive",
                0,
                [
                    {"eos_id": "H000000", "key": "baseline", "case_id": "direct", "amplitude": 0.0},
                    {"eos_id": "H000010", "key": "positive", "case_id": "p", "amplitude": 0.1},
                ],
            )
            negative = _catalogue_frame(
                root,
                "negative",
                1,
                [
                    {"eos_id": "H000000", "key": "baseline", "case_id": "direct", "amplitude": 0.0},
                    {"eos_id": "H000020", "key": "negative", "case_id": "n", "amplitude": -0.1},
                ],
            )
            with patch.object(MODULE, "_source_catalogue", side_effect=[positive, negative]):
                mapping, sources, duplicates = MODULE.build_name_mapping(
                    root, [root / "runs/positive", root / "runs/negative"]
                )

            self.assertEqual(1, duplicates)
            self.assertEqual(3, len(mapping))
            self.assertEqual(["H_0001", "H_0002", "H_0003"], mapping["name"].tolist())
            self.assertEqual(["baseline", "positive", "negative"], mapping["regime"].tolist())
            self.assertEqual(2, len(sources))

    def test_sequence_prefix_requires_and_stops_at_first_sampled_peak(self) -> None:
        rows = pd.DataFrame(
            {
                "attempted_index": [0, 1, 2, 3],
                "is_sampled_peak": [False, False, True, False],
            }
        )
        selected, has_peak = MODULE._sequence_prefix(rows)
        self.assertTrue(has_peak)
        self.assertEqual([0, 1, 2], selected["attempted_index"].tolist())

        selected, has_peak = MODULE._sequence_prefix(
            rows.assign(is_sampled_peak=False)
        )
        self.assertFalse(has_peak)
        self.assertTrue(selected.empty)


if __name__ == "__main__":
    unittest.main()

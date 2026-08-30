from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "notebooks" / "build_combined_all_data.py"
SPEC = importlib.util.spec_from_file_location("build_combined_all_data", BUILDER)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"could not load {BUILDER}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EXPERIMENT_BUILDER = ROOT / "notebooks" / "build_experiment_plots.py"
EXPERIMENT_SPEC = importlib.util.spec_from_file_location(
    "build_experiment_plots", EXPERIMENT_BUILDER
)
if EXPERIMENT_SPEC is None or EXPERIMENT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"could not load {EXPERIMENT_BUILDER}")
EXPERIMENT_MODULE = importlib.util.module_from_spec(EXPERIMENT_SPEC)
EXPERIMENT_SPEC.loader.exec_module(EXPERIMENT_MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _make_experiment(root: Path, name: str, *, precision: str) -> Path:
    experiment = root / "runs" / name / "experiment_abc123"
    packet = experiment / "geometry_001"
    packet.mkdir(parents=True)
    configuration_hash = "a" * 64
    _write_json(
        packet / "complete_configuration.json",
        {
            "stellar_enabled": True,
            "tov_stages": [{"name": "pilot_background"}],
        },
    )
    _write_json(
        packet / "metadata.json",
        {
            "packet_status": "complete",
            "configuration_hash": configuration_hash,
            "baseline_validation_status": "pass",
            "identity_status": "pass",
        },
    )
    _write_json(
        packet / "run_state.json",
        {"packet_status": "complete", "configuration_hash": configuration_hash},
    )
    _write_json(packet / "source_hashes.json", {"scientific.py": "source-v1"})
    pd.DataFrame(
        [
            {
                "case_id": "a0",
                "amplitude": 0.0,
                "epsilon_match_mev_fm3": 152.4912472062717,
                "epsilon0_mev_fm3": 300.0,
                "sigma_mev_fm3": 50.0,
                "delta_mev_fm3": 20.0,
                "anchor_mode": "standard",
                "status": "accepted",
            },
            {
                "case_id": "positive",
                "amplitude": 0.1,
                "epsilon_match_mev_fm3": 152.4912472062717,
                "epsilon0_mev_fm3": 300.0,
                "sigma_mev_fm3": 50.0,
                "delta_mev_fm3": 20.0,
                "anchor_mode": "standard",
                "status": "accepted",
            },
            {
                "case_id": "rejected",
                "amplitude": -0.5,
                "epsilon_match_mev_fm3": 152.4912472062717,
                "epsilon0_mev_fm3": 300.0,
                "sigma_mev_fm3": 50.0,
                "delta_mev_fm3": 20.0,
                "anchor_mode": "standard",
                "status": "rejected",
            },
        ]
    ).to_csv(packet / "case_ledger.csv", index=False)

    raw_rows = []
    for case_id, offset in (("a0", 0.0), ("positive", 0.1), ("rejected", -0.5)):
        for epsilon, direct_cs2 in ((80.0, 0.08), (300.0, 0.32), (1500.0, 0.92)):
            raw_rows.append(
                {
                    "case_id": case_id,
                    "epsilon_mev_fm3": epsilon,
                    "raw_cs2": direct_cs2 + offset,
                }
            )
    pd.DataFrame(raw_rows).to_csv(packet / "raw_gate_profiles.csv", index=False)

    sequence_rows = []
    for case_id, masses, radii in (
        ("direct", [0.8, 1.4, 2.0], [13.0, 12.0, 10.8]),
        ("a0", [0.8, 1.4, 2.0], [13.0, 12.0, 10.8]),
        ("positive", [0.8, 1.4, 2.1], [13.4, 12.5, 11.0]),
    ):
        for attempted_index, (mass, radius) in enumerate(zip(masses, radii)):
            sequence_rows.append(
                {
                    "case_id": case_id,
                    "stage": "pilot_background",
                    "attempted_index": attempted_index,
                    "segment_id": 0,
                    "calculation_status": "success",
                    "Mass": mass,
                    "Radius": radius,
                }
            )
    pd.DataFrame(sequence_rows).to_csv(packet / "stellar_sequences.csv", index=False)
    pd.DataFrame(
        [
            {
                "case_id": case_id,
                "stage": "pilot_background",
                "status": "bracketed_and_solved",
                "mass_msun": 1.4,
                "radius_km": radius,
            }
            for case_id, radius in (("direct", 12.0), ("a0", 12.0), ("positive", 12.5))
        ]
    ).to_csv(packet / "fixed_mass_observables.csv", index=False)

    required = [
        "case_ledger.csv",
        "complete_configuration.json",
        "fixed_mass_observables.csv",
        "metadata.json",
        "raw_gate_profiles.csv",
        "run_state.json",
        "source_hashes.json",
        "stellar_sequences.csv",
    ]
    (packet / "SHA256SUMS.txt").write_text(
        "\n".join(f"{_sha256(packet / relative)}  {relative}" for relative in required)
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        experiment / "experiment.json",
        {
            "status": "complete",
            "settings": {"calculation": "stellar", "precision": precision},
            "settings_hash": "b" * 64,
            "child_packets": ["geometry_001"],
            "child_configuration_hashes": [configuration_hash],
        },
    )
    return experiment


class CombinedAllDataTests(unittest.TestCase):
    def test_derived_destinations_cannot_escape_or_overlap_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = _make_experiment(root, "quick_run", precision="quick")
            destinations = (
                root / "outside",
                root / "runs",
                experiment / "derived",
                root / "runs" / "eos_catalogue" / "derived",
            )
            for destination in destinations:
                with self.subTest(builder="combined", destination=destination):
                    with self.assertRaises(ValueError):
                        MODULE.build_combined_snapshot(
                            root,
                            experiment,
                            destination,
                            precision="quick",
                        )
                with self.subTest(builder="experiment", destination=destination):
                    with self.assertRaises(ValueError):
                        EXPERIMENT_MODULE.build_experiment_plots(
                            root,
                            experiment,
                            destination,
                        )

            alias_folder = root / "runs" / "saved_aliases"
            alias_folder.mkdir()
            with self.assertRaisesRegex(ValueError, "saved-data source"):
                EXPERIMENT_MODULE.build_experiment_plots(
                    root,
                    experiment,
                    alias_folder / "plots",
                    eos_data_path=alias_folder,
                )

    def test_builds_immutable_same_precision_saved_table_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quick = _make_experiment(root, "quick_run", precision="quick")
            _make_experiment(root, "strict_run", precision="strict")
            destination = quick.parent / "COMBINED_ALL_DATA"

            result = MODULE.build_combined_snapshot(
                root,
                quick,
                destination,
                precision="quick",
            )

            self.assertEqual(0, result["solver_calls"])
            self.assertFalse(result["authoritative_packets_modified"])
            self.assertEqual(1, result["experiment_count"])
            self.assertEqual(1, result["packet_count"])
            self.assertEqual(3, result["curve_occurrence_count"])
            self.assertEqual(2, result["unique_curve_count"])
            self.assertEqual(1, result["exact_duplicate_curve_occurrence_count"])
            self.assertEqual(2, result["accepted_proposal_occurrence_count"])
            self.assertEqual(1, result["rejected_proposal_occurrence_count"])
            self.assertEqual(3, result["sound_speed_curve_occurrence_count"])
            self.assertEqual(3, result["unique_sound_speed_curve_count"])
            plot = destination / "plots" / "combined_all_stellar_mr.png"
            self.assertTrue(plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            sound_plot = destination / "plots" / "combined_all_speed_of_sound.png"
            self.assertTrue(sound_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue((destination / "combined_curve_index.csv").is_file())
            self.assertTrue((destination / "curve_occurrences.csv").is_file())
            self.assertTrue((destination / "sound_speed_curve_index.csv").is_file())
            self.assertTrue((destination / "sound_speed_curve_occurrences.csv").is_file())
            self.assertTrue((destination / "included_packets.csv").is_file())
            self.assertTrue((destination / "plot_generation_provenance.json").is_file())
            self.assertTrue((destination / "SHA256SUMS.txt").is_file())
            with self.assertRaises(FileExistsError):
                MODULE.build_combined_snapshot(
                    root,
                    quick,
                    destination,
                    precision="quick",
                )

    def test_current_experiment_collection_is_accepted_only_and_not_cumulative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quick = _make_experiment(root, "quick_run", precision="quick")
            _make_experiment(root, "another_quick_run", precision="quick")

            collected = EXPERIMENT_MODULE.collect_experiment(root, quick)

            self.assertEqual(1, len(collected["packet_index"]))
            self.assertEqual(2, collected["accepted_count"])
            self.assertEqual(1, collected["rejected_count"])
            raw = collected["frames"]["raw_gate_profiles.csv"]
            self.assertNotIn("rejected", set(raw["source_case_id"]))
            self.assertEqual(
                {"a0", "positive"}, set(raw["source_case_id"])
            )
            stellar = collected["frames"]["stellar_sequences.csv"]
            self.assertNotIn("rejected", set(stellar["source_case_id"]))

    def test_current_experiment_collection_fails_closed_on_consumed_table_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quick = _make_experiment(root, "quick_run", precision="quick")
            packet = quick / "geometry_001"
            with (packet / "raw_gate_profiles.csv").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("tampered\n")

            with self.assertRaisesRegex(
                ValueError, "sealed plot source checksum mismatch"
            ):
                EXPERIMENT_MODULE.collect_experiment(root, quick)

    def test_checksum_mismatch_fails_closed_for_current_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quick = _make_experiment(root, "quick_run", precision="quick")
            packet = quick / "geometry_001"
            with (packet / "stellar_sequences.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")

            with self.assertRaisesRegex(RuntimeError, "every declared sealed stellar packet"):
                MODULE.build_combined_snapshot(
                    root,
                    quick,
                    quick.parent / "COMBINED_ALL_DATA",
                    precision="quick",
                )


if __name__ == "__main__":
    unittest.main()

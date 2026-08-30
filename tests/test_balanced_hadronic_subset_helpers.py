from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

import materialize_balanced_hadronic_subset as materializer  # noqa: E402
import replot_balanced_hadronic_subset as replotter  # noqa: E402
import select_combined_hadronic_subset as selector  # noqa: E402


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(selector._canonical(value) + b"\n")


def _seal(folder: Path) -> None:
    selector._write_manifest(folder)


def _thermodynamic_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epsilon_mev_fm3": [80.0, 500.0, 1_000.0, 2_000.0],
            "pressure_mev_fm3": [5.0, 80.0, 220.0, 620.0],
            "cs2": [0.08, 0.25, 0.42, 0.70],
        }
    )


def _stellar_rows(name: str) -> pd.DataFrame:
    mass = np.array([0.5, 1.0, 1.4, 2.0, 2.5, 3.3])
    return pd.DataFrame(
        {
            "name": name,
            "M": mass,
            "R": 13.2 - 0.6 * mass,
            "Lambda": 1_200.0 / mass**5,
            "k2": 0.12 - 0.015 * mass,
            "P_c": 50.0 + 100.0 * mass,
        }
    )


class _SyntheticCombiner:
    def __init__(self, curves: dict[str, pd.DataFrame] | None = None) -> None:
        self.curves = curves or {}

    @staticmethod
    def _direct_physical_key(**values: float) -> str:
        return "|".join(f"{key}={values[key]:.17g}" for key in sorted(values))

    def collect_thermodynamic_curves(
        self,
        root: Path,
        mapping: pd.DataFrame,
        resolved_cases: dict[str, str],
    ) -> list[dict[str, Any]]:
        del root, resolved_cases
        generic = _thermodynamic_rows()
        return [
            {
                "name": str(name),
                "rows": self.curves.get(str(name), generic),
            }
            for name in mapping["name"].astype(str)
        ]


def _make_selection_parent(root: Path) -> tuple[Path, _SyntheticCombiner]:
    runs = root / "runs"
    runs.mkdir(parents=True)
    notebooks = root / "notebooks"
    notebooks.mkdir()
    builder = notebooks / "build_combined_hadronic_dataset.py"
    shutil.copy2(NOTEBOOKS / builder.name, builder)

    experiment = runs / "source" / "experiment_synthetic"
    packet = experiment / "geometry_001"
    packet.mkdir(parents=True)
    (experiment / "SHA256SUMS.txt").write_text(
        "synthetic sealed-experiment marker\n", encoding="utf-8", newline="\n"
    )

    parent = runs / "combined_parent"
    ml_data = parent / "ML_DATA"
    ml_data.mkdir(parents=True)
    names = ["H_0001", "H_0002", "H_0003", "H_0004"]
    amplitudes = [0.0, 0.1, 0.2, 0.3]
    mapping = pd.DataFrame(
        {
            "name": names,
            "source_eos_id": [
                f"source:geometry_001:case_{index}" for index in range(len(names))
            ],
            "regime": ["baseline", "positive", "positive", "positive"],
            "amplitude": amplitudes,
            "epsilon_match_mev_fm3": 80.0,
            "center_mev_fm3": 450.0,
            "width_mev_fm3": 500.0,
            "ramp_width_mev_fm3": 225.0,
            "source_run": "source",
        }
    )
    mapping.to_csv(ml_data / "eos_name_mapping.csv", index=False)
    pd.concat([_stellar_rows(name) for name in names], ignore_index=True).to_csv(
        ml_data / "hadronic_stellar_data.csv", index=False
    )
    provenance = {
        "schema_id": "bsk24_combined_hadronic_ml_dataset_v1",
        "unique_eos_count": len(names),
        "duplicate_physical_occurrence_count": 0,
        "builder_sha256": selector._sha256(builder),
        "source_runs": [
            {
                "run": "source",
                "source_mode": "sealed_curve_experiment",
                "experiment_path": experiment.relative_to(root).as_posix(),
                "experiment_manifest_sha256": selector._sha256(
                    experiment / "SHA256SUMS.txt"
                ),
            }
        ],
    }
    _write_json(parent / "provenance.json", provenance)
    _seal(parent)
    curves = {name: _thermodynamic_rows() for name in names}
    return parent, _SyntheticCombiner(curves)


def _make_materialization_inputs(root: Path) -> tuple[Path, Path]:
    runs = root / "runs"
    runs.mkdir(parents=True)
    parent = runs / "combined_parent"
    ml_data = parent / "ML_DATA"
    ml_data.mkdir(parents=True)

    names = [f"H_{index:04d}" for index in range(1, 2_001)]
    mapping = pd.DataFrame(
        {
            "name": names,
            "source_eos_id": [
                f"source:geometry_001:case_{index:04d}" for index in range(1, 2_001)
            ],
            "regime": ["baseline", *("positive" for _ in range(1, 2_000))],
            "amplitude": [0.0, *(index / 10_000 for index in range(1, 2_000))],
            "epsilon_match_mev_fm3": 80.0,
            "center_mev_fm3": 450.0,
            "width_mev_fm3": 500.0,
            "ramp_width_mev_fm3": 225.0,
            "source_run": "source",
        }
    )
    mapping.to_csv(ml_data / "eos_name_mapping.csv", index=False)
    pd.DataFrame(
        {
            "name": names,
            "M": 1.4,
            "R": 12.0,
            "Lambda": 300.0,
            "k2": 0.08,
            "P_c": 120.0,
        }
    ).to_csv(ml_data / "hadronic_stellar_data.csv", index=False)
    _write_json(parent / "provenance.json", {"source_runs": []})
    _seal(parent)

    dryrun = runs / "balanced_dryrun"
    dryrun.mkdir()
    selected = mapping.copy()
    selected["physical_model_key"] = [f"physical-{index:04d}" for index in range(2_000)]
    selected.to_csv(dryrun / "selected_eos_2000.csv", index=False)
    selected.assign(selected=True).to_csv(
        dryrun / "selection_manifest.csv", index=False
    )
    metrics = {
        "geometry_group_count_selected": 1,
        "r14_selected": {
            "minimum_km": 12.0,
            "maximum_km": 12.0,
            "largest_adjacent_gap_km": 0.0,
        },
        "r14_density": {"selected_count_coefficient_of_variation": 0.0},
        "mr_raster_below_2msun_retention_fraction": 1.0,
        "mr_raster_cell_retention_fraction": 1.0,
    }
    gates = {"synthetic_saved_data_contract": True}
    _write_json(dryrun / "coverage_metrics.json", metrics)
    _write_json(dryrun / "validation_gates.json", gates)
    _write_json(
        dryrun / "provenance.json",
        {
            "all_validation_gates_passed": True,
            "selection_policy": "synthetic reviewed water-filling selection",
            "selector_sha256": selector._sha256(Path(selector.__file__)),
            "parent_manifest_sha256": selector._sha256(
                parent / "SHA256SUMS.txt"
            ),
        },
    )
    _seal(dryrun)
    return parent, dryrun


def _fake_internal_mapping(
    root: Path,
    mapping: pd.DataFrame,
    parent_provenance: dict[str, Any],
    combiner: Any,
) -> tuple[pd.DataFrame, dict[str, str]]:
    del root, parent_provenance, combiner
    names = mapping["name"].astype(str)
    return mapping.copy(), {name: f"case-{name}" for name in names}


def _placeholder_inventory(
    plots: Path, filenames: tuple[str, ...], selected_names: list[str]
) -> list[dict[str, Any]]:
    inventory = []
    for filename in filenames:
        (plots / filename).write_bytes(PNG_SIGNATURE)
        inventory.append(
            {"figure": filename, "plotted_eos_count": len(selected_names)}
        )
    return inventory


def _fake_stellar_plots(
    curves: dict[str, pd.DataFrame],
    selected_names: list[str],
    baseline: str,
    plots: Path,
) -> list[dict[str, Any]]:
    del curves, baseline
    return _placeholder_inventory(plots, tuple(materializer.PLOT_FILES[2:]), selected_names)


def _fake_thermodynamic_plots(
    curves: dict[str, pd.DataFrame],
    selected_names: list[str],
    baseline: str,
    plots: Path,
) -> list[dict[str, Any]]:
    del curves, baseline
    return _placeholder_inventory(plots, tuple(materializer.PLOT_FILES[:2]), selected_names)


class BalancedHadronicSubsetHelperTests(unittest.TestCase):
    def test_combiner_loader_rejects_a_caller_selected_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebooks = root / "notebooks"
            notebooks.mkdir()
            sentinel = root / "executed.txt"
            (notebooks / "build_combined_hadronic_dataset.py").write_text(
                f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "reviewed checkout"):
                selector._load_combiner(root)
            self.assertFalse(sentinel.exists())

    def test_direct_child_rejects_traversal_and_nested_schema_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            packet = experiment / "geometry_001"
            packet.mkdir(parents=True)

            self.assertEqual(packet, selector._direct_child(experiment, "geometry_001"))
            for unsafe in ("../geometry_001", "nested/geometry_001", "nested\\geometry_001"):
                with self.subTest(unsafe=unsafe):
                    with self.assertRaises(ValueError):
                        selector._direct_child(experiment, unsafe)

    def test_direct_child_rejects_a_symlink_escape_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            outside = root / "outside"
            experiment.mkdir()
            outside.mkdir()
            linked = experiment / "geometry_link"
            try:
                linked.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")

            with self.assertRaises(ValueError):
                selector._direct_child(experiment, linked.name)

    def test_each_helper_rejects_source_destination_overlap_before_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            parent = runs / "parent"
            dryrun = runs / "dryrun"
            source = runs / "source"
            parent.mkdir(parents=True)
            dryrun.mkdir()
            source.mkdir()

            with self.assertRaisesRegex(ValueError, "overlap"):
                selector.run(root, parent, parent / "selection", target_count=3)
            with self.assertRaisesRegex(ValueError, "overlap"):
                materializer.materialize(root, parent, dryrun, dryrun / "final")
            with self.assertRaisesRegex(ValueError, "overlap"):
                replotter.replot(root, source, source / "plots")

            catalogue_destination = runs / "eos_catalogue" / "derived"
            for helper, call in (
                (
                    "select",
                    lambda: selector.run(
                        root, parent, catalogue_destination, target_count=3
                    ),
                ),
                (
                    "materialize",
                    lambda: materializer.materialize(
                        root, parent, dryrun, catalogue_destination
                    ),
                ),
                (
                    "replot",
                    lambda: replotter.replot(
                        root, source, catalogue_destination
                    ),
                ),
            ):
                with self.subTest(helper=helper), self.assertRaisesRegex(
                    ValueError, "overlap"
                ):
                    call()

    def test_shared_publisher_preserves_a_raced_in_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            stage = parent / ".stage"
            target = parent / "published"
            stage.mkdir()
            (stage / "payload.txt").write_text("payload", encoding="utf-8")

            def race_in_destination(source: Path, destination: Path) -> None:
                del source
                destination.mkdir()
                (destination / "owner.txt").write_text("other", encoding="utf-8")
                raise PermissionError("synthetic rename race")

            with (
                patch.object(
                    selector.catalogue.os,
                    "rename",
                    side_effect=race_in_destination,
                ),
                patch.object(selector.catalogue.time, "sleep"),
                self.assertRaises(FileExistsError),
            ):
                selector.catalogue.publish_directory(stage, target)

            self.assertEqual("other", (target / "owner.txt").read_text(encoding="utf-8"))
            self.assertEqual("payload", (stage / "payload.txt").read_text(encoding="utf-8"))
            self.assertFalse((parent / ".published.publish.lock").exists())

    def test_source_experiment_resolution_fails_closed_without_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runs").mkdir()
            records = [
                {
                    "run": "missing",
                    "experiment_path": "runs/missing/experiment_missing",
                }
            ]

            with self.assertRaises(FileNotFoundError):
                selector._source_experiments(root, records)

    def test_manifest_verification_fails_closed_for_changes_and_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet = root / "packet"
            packet.mkdir()
            data = packet / "saved.csv"
            data.write_text("value\n1\n", encoding="utf-8", newline="\n")
            _seal(packet)

            manifest = selector._read_manifest(packet)
            self.assertEqual(data, selector._verified(packet, "saved.csv", manifest))
            data.write_text("value\n2\n", encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                selector._verified(packet, "saved.csv", manifest)

            outside = root / "outside.csv"
            outside.write_text("value\n1\n", encoding="utf-8", newline="\n")
            with self.assertRaises(ValueError):
                selector._verified(
                    packet,
                    "../outside.csv",
                    {"../outside.csv": selector._sha256(outside)},
                )

    def test_balanced_selection_is_deterministic_and_prefers_new_raster_cells(self) -> None:
        names = list("ABCDEFGH")
        features = np.array([[0.0], [0.1], [0.2], [0.3], [1.0], [0.9], [0.8], [0.7]])
        bins = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        cells = [
            {(0, 0)},
            {(1, 0)},
            {(0, 0)},
            {(3, 0)},
            {(4, 0)},
            {(5, 0)},
            {(6, 0)},
            {(7, 0)},
        ]

        first = selector._select_balanced(
            names, features, {"A", "E"}, bins, cells, target_count=7
        )
        second = selector._select_balanced(
            names, features, {"A", "E"}, bins, cells, target_count=7
        )

        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])
        self.assertFalse(first[0][2])
        self.assertTrue(first[0][6])
        self.assertEqual("r14_waterfill_raster_diversity", first[2]["G"])
        self.assertEqual({0: 1, 1: 1}, first[3])

    def test_selector_writes_a_new_verified_report_without_changing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, combiner = _make_selection_parent(root)
            destination = root / "runs" / "selection_report"
            parent_manifest_before = selector._sha256(parent / "SHA256SUMS.txt")

            with (
                patch.object(selector, "TARGET_COUNT", 3),
                patch.object(selector, "_load_combiner", return_value=combiner),
            ):
                result = selector.run(root, parent, destination, target_count=3)

            self.assertTrue(result["all_validation_gates_passed"])
            self.assertEqual(0, json.loads((destination / "provenance.json").read_text())["solver_calls"])
            selected = pd.read_csv(destination / "selected_eos_2000.csv")
            self.assertEqual(["H_0001", "H_0002", "H_0004"], selected["name"].tolist())
            self.assertEqual(
                parent_manifest_before,
                selector._sha256(parent / "SHA256SUMS.txt"),
            )
            output_manifest = selector._read_manifest(destination)
            selector._verified(destination, "coverage_metrics.json", output_manifest)
            with self.assertRaises(FileExistsError):
                selector.run(root, parent, destination, target_count=3)

    def test_materialize_and_replot_are_atomic_checksum_bound_derivatives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, dryrun = _make_materialization_inputs(root)
            destination = root / "runs" / "balanced_final"
            parent_before = selector._sha256(parent / "SHA256SUMS.txt")
            dryrun_before = selector._sha256(dryrun / "SHA256SUMS.txt")
            combiner = _SyntheticCombiner()

            with (
                patch.object(selector, "_load_combiner", return_value=combiner),
                patch.object(
                    materializer,
                    "_selected_internal_mapping",
                    side_effect=_fake_internal_mapping,
                ),
                patch.object(
                    materializer, "_draw_stellar", side_effect=_fake_stellar_plots
                ),
                patch.object(
                    materializer,
                    "_draw_thermodynamic",
                    side_effect=_fake_thermodynamic_plots,
                ),
            ):
                materialized = materializer.materialize(
                    root, parent, dryrun, destination
                )

            self.assertEqual(0, materialized["solver_calls"])
            self.assertFalse(materialized["authoritative_packets_modified"])
            self.assertEqual(parent_before, selector._sha256(parent / "SHA256SUMS.txt"))
            self.assertEqual(dryrun_before, selector._sha256(dryrun / "SHA256SUMS.txt"))
            self.assertEqual(
                list(materializer.STELLAR_COLUMNS),
                pd.read_csv(destination / "ML_DATA" / "hadronic_stellar_data.csv").columns.tolist(),
            )
            self.assertEqual(
                set(materializer.PLOT_FILES),
                {path.name for path in (destination / "plots").glob("*.png")},
            )
            materialized_manifest = selector._read_manifest(destination)
            selector._verified(destination, "provenance.json", materialized_manifest)
            with self.assertRaises(FileExistsError):
                materializer.materialize(root, parent, dryrun, destination)

            replot_destination = root / "runs" / "balanced_blue_replot"
            source_before = selector._sha256(destination / "SHA256SUMS.txt")
            with (
                patch.object(selector, "_load_combiner", return_value=combiner),
                patch.object(
                    materializer,
                    "_selected_internal_mapping",
                    side_effect=_fake_internal_mapping,
                ),
                patch.object(replotter, "_draw_stellar", side_effect=_fake_stellar_plots),
                patch.object(
                    replotter,
                    "_draw_thermodynamic",
                    side_effect=_fake_thermodynamic_plots,
                ),
            ):
                replotted = replotter.replot(
                    root,
                    destination,
                    replot_destination,
                    parent_override=parent,
                )

            self.assertEqual(0, replotted["solver_calls"])
            self.assertFalse(replotted["scientific_data_changed"])
            self.assertFalse(replotted["source_dataset_modified"])
            self.assertEqual(
                source_before, selector._sha256(destination / "SHA256SUMS.txt")
            )
            self.assertEqual(
                set(replotter.PLOT_FILES),
                {path.name for path in replot_destination.glob("*.png")},
            )
            replot_manifest = selector._read_manifest(replot_destination)
            selector._verified(replot_destination, "provenance.json", replot_manifest)
            self.assertFalse(
                list((root / "runs").glob(".balanced_2000_final_*"))
                or list((root / "runs").glob(".balanced_2000_thick_*"))
            )
            with self.assertRaises(FileExistsError):
                replotter.replot(
                    root,
                    destination,
                    replot_destination,
                    parent_override=parent,
                )

    def test_materialize_and_replot_render_exact_saved_data_plot_sets(self) -> None:
        names = ["baseline", "deformed"]
        stellar = {name: _stellar_rows(name) for name in names}
        thermo = {name: _thermodynamic_rows() for name in names}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for module_name, module in (
                ("materialize", materializer),
                ("replot", replotter),
            ):
                plots = root / module_name
                plots.mkdir()
                inventory = module._draw_stellar(stellar, names, names[0], plots)
                inventory.extend(
                    module._draw_thermodynamic(thermo, names, names[0], plots)
                )

                self.assertEqual(set(module.PLOT_FILES), {row["figure"] for row in inventory})
                self.assertTrue(
                    all(row["plotted_eos_count"] == 2 for row in inventory)
                )
                for filename in module.PLOT_FILES:
                    self.assertTrue((plots / filename).read_bytes().startswith(PNG_SIGNATURE))


if __name__ == "__main__":
    unittest.main()

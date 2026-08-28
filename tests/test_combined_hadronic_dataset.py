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


class CombinedHadronicDatasetTests(unittest.TestCase):
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

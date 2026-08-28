"""Passive contracts for the promoted six-worker / tight 40-point defaults."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eos_generation._experiment_planning import _precision_profile
from eos_generation._internal import stellar
from eos_generation import notebook
from test_notebook import _Harness, _settings


ROOT = Path(__file__).resolve().parents[1]


class NotebookDefaultsTests(unittest.TestCase):
    def test_both_notebooks_select_tight_40_and_remain_passive(self):
        for name in ("bsk24_dataset.ipynb", "bsk24_experiment.ipynb"):
            with self.subTest(notebook=name):
                document = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
                cell = next(c for c in document["cells"] if c["id"] == "user-settings")
                tree = ast.parse("".join(cell["source"]))
                values = {
                    node.targets[0].id: ast.literal_eval(node.value)
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                }
                self.assertEqual("dataset_40", values["PRECISION"])
                self.assertEqual("stellar", values["CALCULATION"])
                self.assertEqual("off", values["DIAGNOSTICS"])
                self.assertEqual([1.4], values["FIXED_MASSES"])
                self.assertIs(False, values["EXECUTE_REVIEWED_PLAN"])
                stage, = _precision_profile(values["PRECISION"], values["CALCULATION"])["tov_stages"]
                self.assertEqual((40, 1e-10, 1e-12, 1201),
                                 (stage.sequence_points, stage.rtol, stage.atol, stage.radial_profile_points))
                for code in (c for c in document["cells"] if c["cell_type"] == "code"):
                    compile("".join(code["source"]), name + ":" + code["id"], "exec")

    def test_preview_and_production_share_six_worker_cpu_bound(self):
        for logical, expected in ((None, 1), (1, 1), (4, 2), (8, 4), (12, 6), (32, 6)):
            with self.subTest(logical=logical), patch.object(os, "cpu_count", return_value=logical), \
                 patch.dict(os.environ, {stellar._OUTER_NOTEBOOK_WORKER_ENV: "0"}):
                self.assertEqual(expected, notebook._default_worker_count())
                self.assertEqual(expected, stellar._automatic_stellar_worker_count(20))
        self.assertEqual(6, notebook._MAX_WORKERS)
        self.assertEqual(6, stellar._MAXIMUM_AUTOMATIC_STELLAR_WORKERS)

    def test_small_batches_do_not_spawn_idle_case_workers(self):
        with patch.object(os, "cpu_count", return_value=12), \
             patch.dict(os.environ, {stellar._OUTER_NOTEBOOK_WORKER_ENV: "0"}):
            for count, expected in ((0, 1), (1, 1), (2, 2), (5, 5), (6, 6), (10, 6)):
                self.assertEqual(expected, stellar._automatic_stellar_worker_count(count))

    def test_nested_case_and_sequence_pools_stay_disabled(self):
        from eos_generation.stellar._tov_sequence import _automatic_sequence_worker_count
        with patch.object(os, "cpu_count", return_value=12), \
             patch.dict(os.environ, {stellar._OUTER_NOTEBOOK_WORKER_ENV: "1"}):
            self.assertEqual(1, stellar._automatic_stellar_worker_count(20))
            self.assertEqual(1, _automatic_sequence_worker_count(40))

    def test_preview_prints_and_binds_budget_without_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _Harness(root)
            harness.workers = 6
            session = harness.session()
            settings = _settings(calculation="stellar", precision="dataset_40")
            run = session.prepare(settings, record_preview=True)
            self.assertEqual(6, run.to_dict()["worker_count"])
            self.assertIn("Automatic stellar worker limit: 6", run.summary_text())
            self.assertIn("40-model stellar stage", run.summary_text())
            self.assertIn("rtol=1e-10, atol=1e-12", run.summary_text())
            self.assertIn("Not a STRICT certificate", run.summary_text())
            self.assertEqual(0, harness.runner_calls)
            self.assertFalse((root / "runs").exists())
            harness.workers = 4
            with self.assertRaisesRegex(RuntimeError, "process budget changed"):
                session.execute(run, current_settings=settings, execute=True)
            self.assertEqual(0, harness.runner_calls)
            self.assertFalse((root / "runs").exists())


if __name__ == "__main__":
    unittest.main()

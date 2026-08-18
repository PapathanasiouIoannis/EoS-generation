from __future__ import annotations

import json
import re
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from eos_generation.notebook import (
    NotebookRun,
    NotebookSession,
    NotebookSettings,
    get_notebook_session,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "bsk24_experiment.ipynb"
FIXED_NOW = datetime(2099, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _Plan:
    settings: Any
    output_root: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": "synthetic_experiment_plan_v1",
            "settings": self.settings,
            "output_root": str(self.output_root),
            "scientific_solver_calls": 0,
            "filesystem_writes": 0,
        }

    def summary_text(self) -> str:
        return "Synthetic production-plan details"


def _settings(**overrides: Any) -> NotebookSettings:
    values = {
        "amplitudes": [-0.01, 0.0, 0.01],
        "epsilon_match": "standard",
        "center": 300.0,
        "width": 50.0,
        "ramp_width": 20.0,
        "calculation": "thermodynamics",
        "fixed_masses": [1.4],
        "precision": "strict",
        "diagnostics": "off",
    }
    values.update(overrides)
    return NotebookSettings.from_values(**values)


class _Harness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.planner_calls = 0
        self.runner_calls = 0
        self.validator_calls = 0
        self.source = {"source.py": "source-v1"}
        self.environment = {"python": "test", "numpy": "test"}
        self.workers = 2

    def planner(self, settings: Any, *, output_root: Path) -> _Plan:
        self.planner_calls += 1
        if output_root.exists():
            raise AssertionError("planner received an occupied destination")
        return _Plan(settings=settings, output_root=output_root)

    def runner(self, plan: _Plan, *, execute: bool) -> SimpleNamespace:
        self.runner_calls += 1
        if execute is not True:
            raise AssertionError("runner did not receive the explicit execute gate")
        plan.output_root.mkdir(parents=True)
        (plan.output_root / "synthetic.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(output_root=plan.output_root)

    def loader(self, path: Path) -> SimpleNamespace:
        return SimpleNamespace(output_root=path)

    def validator(self, path: Path) -> dict[str, str]:
        self.validator_calls += 1
        return {"status": "pass" if (path / "synthetic.json").is_file() else "fail"}

    def session(self) -> NotebookSession:
        return NotebookSession(
            self.root,
            planner=self.planner,
            runner=self.runner,
            loader=self.loader,
            validator=self.validator,
            settings_factory=lambda settings: settings.to_dict(),
            source_state=lambda: dict(self.source),
            environment_state=lambda: dict(self.environment),
            worker_count=lambda: self.workers,
            now=lambda: FIXED_NOW,
        )


class NotebookSettingsTests(unittest.TestCase):
    def test_scalars_and_lists_define_a_normalized_cartesian_grid(self) -> None:
        settings = _settings(
            amplitudes=[-0.02, 0.0, 0.02],
            epsilon_match=200.0,
            center=[250, 300],
            width=50,
            ramp_width=[10, 20],
        )
        self.assertEqual((-0.02, 0.0, 0.02), settings.amplitudes)
        self.assertEqual(200.0, settings.epsilon_match)
        self.assertEqual((250.0, 300.0), settings.centers_mev_fm3)
        self.assertEqual(4, settings.geometry_count)
        self.assertEqual(12, settings.requested_case_count)
        self.assertEqual("eos_generation_notebook_settings_v1", settings.to_dict()["schema_id"])

    def test_small_public_choices_fail_closed(self) -> None:
        for values, message in (
            ({"calculation": "tov"}, "CALCULATION"),
            ({"precision": "relaxed"}, "PRECISION"),
            ({"diagnostics": "radial-support"}, "DIAGNOSTICS"),
            ({"width": 0.0}, "WIDTH"),
            ({"amplitudes": [0.0, -0.0]}, "duplicate"),
        ):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                _settings(**values)
        with self.assertRaisesRegex(ValueError, "requires CALCULATION='stellar'"):
            _settings(diagnostics="on")
        with self.assertRaisesRegex(ValueError, "one matching anchor"):
            _settings(epsilon_match=["standard", 200.0])
        self.assertEqual(
            "on", _settings(calculation="stellar", diagnostics="on").diagnostics
        )


class NotebookSessionTests(unittest.TestCase):
    def test_preview_is_solver_free_and_write_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _Harness(root)
            session = harness.session()

            run = session.prepare(_settings(), record_preview=True)

            self.assertIsInstance(run, NotebookRun)
            self.assertEqual(1, harness.planner_calls)
            self.assertEqual(0, harness.runner_calls)
            self.assertFalse((root / "runs").exists())
            self.assertFalse(run.output_root.exists())
            self.assertEqual(0, run.to_dict()["scientific_solver_calls"])
            self.assertEqual(0, run.to_dict()["filesystem_writes"])

    def test_second_pass_executes_only_the_exact_reviewed_plan_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _Harness(root)
            session = harness.session()
            settings = _settings()
            preview = session.prepare(settings, record_preview=True)

            second_pass = session.prepare(settings, record_preview=False)
            self.assertIs(preview, second_pass)
            result = session.execute(
                second_pass, current_settings=settings, execute=True
            )

            self.assertEqual(preview.output_root, result.output_root)
            self.assertEqual(1, harness.planner_calls)
            self.assertEqual(1, harness.runner_calls)
            self.assertEqual(1, harness.validator_calls)
            self.assertTrue(preview.output_root.is_dir())
            with self.assertRaisesRegex(RuntimeError, "already been consumed"):
                session.execute(preview, current_settings=settings, execute=True)

    def test_execute_without_preview_or_after_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _Harness(root)
            session = harness.session()
            settings = _settings()
            unreviewed = session.prepare(settings, record_preview=False)
            with self.assertRaisesRegex(RuntimeError, "was not reviewed"):
                session.execute(unreviewed, current_settings=settings, execute=True)

            reviewed = session.prepare(settings, record_preview=True)
            with self.assertRaisesRegex(RuntimeError, "settings changed"):
                session.execute(
                    reviewed,
                    current_settings=_settings(amplitudes=[0.0, 0.02]),
                    execute=True,
                )
            harness.source["source.py"] = "source-v2"
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                session.execute(reviewed, current_settings=settings, execute=True)
            self.assertEqual(0, harness.runner_calls)
            self.assertFalse((root / "runs").exists())

    def test_false_execute_flag_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _Harness(root)
            session = harness.session()
            settings = _settings()
            reviewed = session.prepare(settings, record_preview=True)
            self.assertIsNone(
                session.execute(reviewed, current_settings=settings, execute=False)
            )
            self.assertEqual(0, harness.runner_calls)
            self.assertFalse((root / "runs").exists())


class NotebookArtifactTests(unittest.TestCase):
    def test_saved_notebook_executes_passively_from_repository_root(self) -> None:
        try:
            import nbformat
            from nbclient import NotebookClient
        except ImportError:
            self.skipTest("notebook optional dependencies are not installed")

        runs = ROOT / "runs"

        def snapshot() -> tuple[tuple[str, bool, int, int], ...]:
            if not runs.exists():
                return ()
            return tuple(
                (
                    path.relative_to(runs).as_posix(),
                    path.is_dir(),
                    0 if path.is_dir() else path.stat().st_size,
                    path.stat().st_mtime_ns,
                )
                for path in sorted(runs.rglob("*"))
            )

        before_exists = runs.exists()
        before = snapshot()
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        executed = NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
        ).execute(cwd=str(ROOT))
        after = snapshot()
        self.assertEqual(before_exists, runs.exists())
        self.assertEqual(before, after)
        output_text = "\n".join(
            str(output.get("text", ""))
            for cell in executed.cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "stream"
        )
        self.assertIn("0 solver calls", output_text)
        self.assertIn("0 filesystem writes", output_text)
        self.assertIn("EXECUTE_REVIEWED_PLAN=False", output_text)

    def test_saved_notebook_has_one_editable_settings_cell_and_no_outputs(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        editable = [
            cell for cell in notebook["cells"] if cell.get("metadata", {}).get("editable", True)
        ]
        self.assertEqual(["user-settings"], [cell["id"] for cell in editable])
        self.assertTrue(all(cell["execution_count"] is None for cell in code_cells))
        self.assertTrue(all(not cell["outputs"] for cell in code_cells))
        self.assertEqual("python3", notebook["metadata"]["kernelspec"]["name"])

        settings = next(cell for cell in code_cells if cell["id"] == "user-settings")
        settings_source = "".join(settings["source"])
        for name in (
            "AMPLITUDES",
            "EPSILON_MATCH",
            "CENTER",
            "WIDTH",
            "RAMP_WIDTH",
            "CALCULATION",
            "FIXED_MASSES",
            "PRECISION",
            "DIAGNOSTICS",
            "EXECUTE_REVIEWED_PLAN",
        ):
            self.assertEqual(
                1,
                len(re.findall(rf"(?m)^{name}\s*=", settings_source)),
                name,
            )
        self.assertIn("EXECUTE_REVIEWED_PLAN = False", settings_source)
        self.assertNotIn("NotebookSettings", settings_source)

        locked_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in code_cells
            if cell["id"] != "user-settings"
        )
        self.assertEqual(1, locked_source.count("notebook_session.prepare("))
        self.assertEqual(1, locked_source.count("notebook_session.execute("))
        self.assertIn("record_preview=not EXECUTE_REVIEWED_PLAN", locked_source)
        self.assertIn("execute=EXECUTE_REVIEWED_PLAN", locked_source)
        self.assertNotIn("ipywidgets", locked_source)
        self.assertNotIn("framework.", locked_source)
        for cell in code_cells:
            compile("".join(cell["source"]), f"{NOTEBOOK}:{cell['id']}", "exec")

    def test_public_notebook_api_is_intentionally_small(self) -> None:
        import eos_generation.notebook as notebook_module

        self.assertEqual(
            {
                "NotebookSettings",
                "NotebookSession",
                "NotebookRun",
                "get_notebook_session",
            },
            set(notebook_module.__all__),
        )
        self.assertIsInstance(get_notebook_session(ROOT), NotebookSession)

    def test_real_preview_serialization_contains_no_machine_paths(self) -> None:
        session = get_notebook_session(ROOT)
        run = session.prepare(_settings(precision="quick"), record_preview=False)
        serialized = json.dumps(run.to_dict(), sort_keys=True, allow_nan=False)
        self.assertNotIn(str(ROOT), serialized)
        self.assertNotIn("python_executable", serialized)
        self.assertIn('"planning_root": "runs/', serialized)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from eos_generation.reporting.plot_helpers import finalize_figure


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _temporary_runs_root() -> Iterator[Path]:
    runs = ROOT / "runs"
    runs_preexisted = runs.exists()
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plot-helper-", dir=runs) as directory:
        yield Path(directory)
    if not runs_preexisted:
        try:
            runs.rmdir()
        except OSError:
            pass


class PlotHelperTests(unittest.TestCase):
    def test_finalize_figure_syncs_a_writable_temporary_file(self) -> None:
        with _temporary_runs_root() as directory:
            output = directory / "figure.png"
            temporary = output.with_name(output.name + ".tmp")
            observed_modes: list[str] = []
            original_open = Path.open

            def tracked_open(
                path: Path,
                mode: str = "r",
                buffering: int = -1,
                encoding: str | None = None,
                errors: str | None = None,
                newline: str | None = None,
            ):
                if path == temporary:
                    observed_modes.append(mode)
                return original_open(
                    path,
                    mode=mode,
                    buffering=buffering,
                    encoding=encoding,
                    errors=errors,
                    newline=newline,
                )

            figure = Figure()
            figure.subplots().plot([0.0, 1.0], [0.0, 1.0])
            with patch.object(Path, "open", new=tracked_open):
                finalized = finalize_figure(figure, output, dpi=72)

            self.assertEqual(finalized, output.resolve())
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertFalse(temporary.exists())
            self.assertIn("r+b", observed_modes)


if __name__ == "__main__":
    unittest.main()

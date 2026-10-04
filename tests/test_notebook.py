from pathlib import Path
import os
import ast

import nbformat
from nbclient import NotebookClient
import pytest

ROOT = Path(__file__).resolve().parents[1]


def configured_code(source, **values):
    lines = source.splitlines(keepends=True)
    for node in reversed(ast.parse(source).body):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name in values:
                lines[node.lineno - 1 : node.end_lineno] = [
                    f"{name} = {values[name]!r}\n"
                ]
    return "".join(lines)


@pytest.mark.parametrize("matter_model", ["bsk24", "bsk25"])
@pytest.mark.parametrize("cwd", [ROOT, ROOT / "notebooks"])
def test_notebook_passive_kernel(cwd, matter_model):
    notebook = nbformat.read(ROOT / "notebooks/bsk24_experiment.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert [cell.cell_type for cell in notebook.cells] == ["markdown", "code"]
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "EXECUTE_REVIEWED_PLAN = False" in source
    assert '_execute_notebook_plan(globals().get("reviewed_plan"), settings)' in source
    assert "NotebookSession" not in source
    for option in (
        "observables",
        "TABLES",
        "TABLE_ROWS",
        "SHOW_GUIDES",
        "baryonic_observables",
        "odd_even_response",
        "numerical_error_summary",
        "LAMBDA_SCALE",
        "LAMBDA_MASS_LIMITS",
        "LAMBDA_LIMITS",
    ):
        assert option in source
    if matter_model == "bsk25":
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "NOTEBOOK_SETTINGS")
        controls = ast.literal_eval(node.value)
        controls["matter_model"] = matter_model
        # The user's BSk24 numeric anchor can be invalid after switching models.
        controls["epsilon_match"] = "standard"
        notebook.cells[1].source = configured_code(source, NOTEBOOK_SETTINGS=controls)
    guard = nbformat.v4.new_code_cell(
        """from eos_generation import thermodynamics, stellar
def forbidden(*args, **kwargs):
    raise AssertionError("passive notebook attempted scientific execution")
thermodynamics.build_consistent_baseline = forbidden
thermodynamics.build_windowed_eos = forbidden
stellar.solve_star = forbidden
stellar.solve_sequence = forbidden
"""
    )
    notebook.cells.insert(0, guard)
    runs = ROOT / "runs"
    before = {p.relative_to(runs) for p in runs.rglob("*")} if runs.exists() else set()
    client = NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd)}},
    )
    client.execute(env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    after = {p.relative_to(runs) for p in runs.rglob("*")} if runs.exists() else set()
    assert before == after
    assert "No calculation requested." in "".join(
        output.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )


def test_single_cell_execution_without_review_is_refused(tmp_path):
    notebook = nbformat.read(ROOT / "notebooks/bsk24_experiment.ipynb", as_version=4)
    notebook.cells[1].source = configured_code(
        notebook.cells[1].source,
        ACTION="execute",
        PLOTS="none",
        TABLES="none",
        SHOW_GUIDES=False,
    )
    notebook.cells[1].source = notebook.cells[1].source.replace(
        'RUNS = ROOT / "runs"', f'RUNS = Path({str(tmp_path / "runs")!r})'
    )
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            """from eos_generation import experiment
def forbidden(*args, **kwargs):
    raise AssertionError("unreviewed execution attempted a calculation")
experiment.run_experiment = forbidden
"""
        ),
    )
    NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(ROOT)}},
    ).execute(env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    error = next(
        output for output in notebook.cells[-1].outputs if output.output_type == "error"
    )
    assert error.ename == "ValueError" and "review" in error.evalue
    assert not list(tmp_path.iterdir())

"""Passive notebook and synthetic saved-table presentation contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from eos_generation import plan_experiment
from eos_generation.notebook import NotebookSession, NotebookSettings
from eos_generation.reporting.notebook_results import (
    _collect_saved_data,
    _plot_runs,
    cfl_notebook_view,
)
from eos_generation.stellar.tov import LAMBDA_FRAMEWORK_CAPABILITY


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "cfl_experiment.ipynb"
DATASET_NOTEBOOK = ROOT / "notebooks" / "cfl_dataset.ipynb"


def _settings(**overrides):
    values = dict(matter_model="cfl", epsilon_match="surface", amplitudes=[-0.02, 0.0, 0.02], center=800.0, width=150.0,
                  ramp_width=100.0, calculation="stellar", precision="quick")
    return NotebookSettings.from_values(**{**values, **overrides})


def test_model_discriminator_preserves_legacy_serialization_and_tiny_cfl_amplitudes():
    cfl = _settings(amplitudes=[0.0, 1e-18])
    assert cfl.amplitudes == (0.0, 1e-18)
    assert cfl.to_dict()["schema_id"] == "eos_generation_cfl_notebook_settings_v1"
    assert cfl.to_experiment_settings().matter_model == "cfl"
    legacy = _settings(matter_model="bsk24", epsilon_match="standard", amplitudes=[1e-18])
    assert legacy.amplitudes == (0.0,)
    assert "matter_model" not in legacy.to_dict()
    assert "matter_model" not in legacy.to_experiment_settings().to_dict()
    assert legacy.to_dict()["schema_id"] == "eos_generation_notebook_settings_v1"
    with pytest.raises(ValueError, match="surface"):
        _settings(epsilon_match="standard")
    with pytest.raises(ValueError, match="unavailable"):
        _settings(diagnostics="on")


@pytest.mark.parametrize("precision,stage_names,points", [
    ("quick", ["pilot_background"], [17]),
    ("strict", ["current", "finer_grid", "tighter_ode"], [61, 121, 121]),
    ("dataset_40", ["dataset_40"], [40]),
])
def test_notebook_uses_exact_governed_profiles_and_full_sequences(precision, stage_names, points, monkeypatch):
    import eos_generation.cfl.baseline as baseline

    monkeypatch.setattr(baseline, "brentq", lambda *a, **k: pytest.fail("planning called a root solver"))
    settings = _settings(precision=precision)
    plan = plan_experiment(settings.to_experiment_settings())
    config = plan.child_plans[0].config
    assert [s.name for s in config.tov_stages] == stage_names
    assert [s.sequence_points for s in config.tov_stages] == points
    assert config.fixed_masses_msun == (1.4,)
    assert plan.estimates["stellar_case_stage_evaluations"] == 3 * len(points)
    assert plan.estimates["sampled_sequence_tidal_targets"] == 3 * sum(points)
    assert plan.estimates["fixed_mass_root_targets"] == 3 * len(points)
    assert plan.to_dict()["scientific_solver_calls"] == 0
    if precision == "dataset_40":
        assert [(stage.lower_points, stage.upper_points) for stage in config.thermodynamic_stages] == [
            (1025, 2049), (2049, 4097), (4097, 8193),
        ]
        assert config.raw_gate_lower_points == 4097
        assert config.raw_gate_upper_points == 16385
        assert config.maximum_mass_initial_points == 17
        assert config.tov_stages[0].rtol == 1e-10
        assert config.tov_stages[0].atol == 1e-12
        assert config.tov_stages[0].radial_profile_points == 1201
        assert config.requested_plot_groups == ("none",)


def test_cfl_preview_has_own_destination_and_retains_one_shot_guard():
    session = NotebookSession(ROOT)
    run = session.prepare(_settings(), record_preview=True)
    assert run.planning_root.name.startswith("cfl_")
    assert not run.planning_root.exists()
    assert session.execute(run, current_settings=_settings(), execute=False) is None
    with pytest.raises(RuntimeError, match="settings changed"):
        session.execute(run, current_settings=_settings(precision="strict"), execute=True)
    assert not run.planning_root.exists()


def test_cfl_notebook_has_one_editable_cell_and_no_outputs_or_physics():
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert document["metadata"]["kernelspec"] == {
        "display_name": "Python (eos-generation)",
        "language": "python",
        "name": "eos-generation",
    }
    editable = []
    sources = []
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, str(NOTEBOOK), "exec")
            sources.append(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
        if cell["metadata"].get("editable"):
            editable.append(cell["id"])
    assert editable == ["user-settings"]
    code = "\n".join(sources)
    assert 'PRECISION = "quick"' in code
    assert 'CALCULATION = "stellar"' in code
    assert "EXECUTE_REVIEWED_PLAN = False" in code
    assert 'matter_model="cfl"' in code
    for forbidden in ("solve_ivp", "solve_star", "subprocess", "sys.path", "np.clip"):
        assert forbidden not in code


def test_cfl_dataset_notebook_is_passive_by_default_and_uses_only_combined_reporting():
    document = json.loads(DATASET_NOTEBOOK.read_text(encoding="utf-8"))
    assert document["metadata"]["kernelspec"] == {
        "display_name": "Python (eos-generation)",
        "language": "python",
        "name": "eos-generation",
    }
    editable = []
    sources = []
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, str(DATASET_NOTEBOOK), "exec")
            sources.append(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
        if cell["metadata"].get("editable"):
            editable.append(cell["id"])
    assert editable == ["user-settings"]
    code = "\n".join(sources)
    assert 'matter_model="cfl"' in code
    assert 'epsilon_match="surface"' in code
    assert 'PRECISION = "dataset_40"' in code
    assert "EXECUTE_REVIEWED_PLAN = False" in code
    assert "cfl_dataset.py" in code
    assert "build_cfl_dataset_output" in code
    assert "create_student_view=False" in code
    assert 'planning_root / "CFL_DATASET"' in code
    assert "eos_catalogue.py" not in code
    assert "build_dataset_plots.py" not in code
    assert "subprocess" not in code
    assert "notebook_session.present" not in code
    assert "BUILD_SAVED_PLOTS" not in code


@pytest.mark.parametrize("working_directory", [ROOT, ROOT / "notebooks"])
def test_cfl_dataset_notebook_run_all_is_passive(working_directory):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    def snapshot():
        runs = ROOT / "runs"
        return {
            str(path): (path.is_dir(), 0 if path.is_dir() else path.stat().st_size, path.stat().st_mtime_ns)
            for path in runs.rglob("*")
        } if runs.exists() else {}

    before = snapshot()
    notebook = nbformat.read(DATASET_NOTEBOOK, as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        "import os\n"
        "os.environ.pop('CONDA_PREFIX', None)\n"
        "from unittest.mock import patch\n"
        "def forbidden(*args, **kwargs):\n    raise AssertionError('passive notebook attempted execution or reporting')\n"
        "guards = [patch('eos_generation.experiment.run_experiment', forbidden), "
        "patch('eos_generation.cfl.baseline.brentq', forbidden), "
        "patch('subprocess.run', forbidden)]\n"
        "for guard in guards:\n    guard.start()\n"
    ))
    nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(working_directory)}},
    ).execute()
    assert snapshot() == before
    output = "".join(
        out.get("text", "") for cell in notebook.cells for out in cell.get("outputs", [])
    )
    assert "EXECUTE_REVIEWED_PLAN=False" in output
    assert "dataset_40" in output
    assert "40 pressures" in output
    assert "no per-case stellar refinement envelope" in output


@pytest.mark.parametrize("working_directory", [ROOT, ROOT / "notebooks"])
def test_cfl_notebook_run_all_is_passive(working_directory):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    def snapshot():
        return {str(p): (p.is_dir(), 0 if p.is_dir() else p.stat().st_size, p.stat().st_mtime_ns)
                for p in (ROOT / "runs").rglob("*")}

    before = snapshot()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    # Guard the executed kernel itself, rather than assuming a lack of output
    # proves a lack of solver calls. Imports remain permitted.
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        "import os\n"
        "os.environ.pop('CONDA_PREFIX', None)\n"
        "from unittest.mock import patch\n"
        "def forbidden(*args, **kwargs):\n    raise AssertionError('passive notebook attempted scientific work')\n"
        "guards = [patch('eos_generation.experiment.run_experiment', forbidden), "
        "patch('eos_generation.cfl.baseline.brentq', forbidden), "
        "patch('eos_generation.reporting.notebook_results.cfl_notebook_view', forbidden)]\n"
        "for guard in guards:\n    guard.start()\n"
    ))
    nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(working_directory)}},
    ).execute()
    assert snapshot() == before
    output = "".join(out.get("text", "") for cell in notebook.cells for out in cell.get("outputs", []))
    assert "Preview only" in output
    assert "CFL experiment plan" in output
    assert "17 pressures" in output


def _saved_result(tmp_path, monkeypatch):
    """Synthetic saved rows exercise reporting, never serve as physics fixtures."""
    import eos_generation.experiment as experiment

    root = tmp_path / "workspace"
    path = root / "runs" / "session" / "experiment_synthetic"
    path.mkdir(parents=True)
    (path / "SHA256SUMS.txt").write_text("synthetic aggregate\n", encoding="utf-8")
    children = []
    for index in (1, 2):
        packet = path / f"geometry_{index:03d}"
        packet.mkdir()
        (packet / "SHA256SUMS.txt").write_text(f"synthetic child {index}\n", encoding="utf-8")
        ledger = pd.DataFrame([
            dict(case_id=f"zero-{index}", physical_case_id="baseline", amplitude=0.0, status="accepted"),
            dict(case_id=f"positive-{index}", physical_case_id=f"positive-{index}", amplitude=0.02, status="accepted"),
            dict(case_id=f"bad-{index}", physical_case_id=f"bad-{index}", amplitude=2.0, status="rejected"),
        ])
        for name, value in (("epsilon0_mev_fm3", 800.0 + index), ("sigma_mev_fm3", 150.0), ("delta_mev_fm3", 100.0)):
            ledger[name] = value
        ledger.to_csv(packet / "case_ledger.csv", index=False)
        records = []
        for case_id in ("direct", "baseline", f"positive-{index}", f"bad-{index}"):
            for attempt in range(4):
                records.append(dict(case_id=case_id, stage="final", attempted_index=attempt, segment_id=0,
                                    calculation_status="success", Mass=1.0 + attempt * 0.1, Radius=10.0 + attempt,
                                    k2=0.1, Lambda=100.0, tidal_status=LAMBDA_FRAMEWORK_CAPABILITY if attempt != 1 else "failed_closed"))
        pd.DataFrame(records).to_csv(packet / "stellar_sequences.csv", index=False)
        fixed = pd.DataFrame([dict(case_id="baseline", stage="final", target_mass_msun=1.4, status="bracketed_and_solved",
                                   radius_km=10.0, k2=0.1, lambda_dimensionless=100.0, tidal_status=LAMBDA_FRAMEWORK_CAPABILITY)])
        fixed.to_csv(packet / "fixed_mass_observables.csv", index=False)
        children.append(SimpleNamespace(packet_path=packet, config=SimpleNamespace(tov_stages=[SimpleNamespace(name="final")]),
                                       table=lambda name, packet=packet: pd.read_csv(packet / name)))
    result = SimpleNamespace(experiment_path=path, packet_paths=tuple(c.packet_path for c in children), repository_root=root,
                             child_results=tuple(children), settings=_settings().to_experiment_settings())
    monkeypatch.setattr(experiment, "load_experiment", lambda location: result)
    return result


def test_saved_collection_filters_rejections_and_deduplicates_physical_baseline(tmp_path, monkeypatch):
    result = _saved_result(tmp_path, monkeypatch)
    ledger, tables = _collect_saved_data(result)
    assert len(ledger) == 6
    assert ledger.loc[ledger.status.eq("rejected"), "eos_label"].eq("").all()
    assert set(tables["stellar"].physical_case_id) == {"baseline", "positive-1", "positive-2"}
    assert tables["stellar"].physical_case_id.eq("baseline").sum() == 4
    runs = _plot_runs(tables["stellar"].loc[tables["stellar"].physical_case_id.eq("baseline")], "stellar", "Mass", "Lambda")
    assert [list(run.attempted_index) for run in runs] == [[0], [2, 3]]


def test_stellar_plotting_never_bridges_missing_attempts_or_background_failures():
    frame = pd.DataFrame(dict(attempted_index=[0, 1, 2, 4], segment_id=[0, 0, 1, 1], calculation_status=["success", "failed", "success", "success"],
                              Mass=[1.0, 1.1, 1.2, 1.3], Radius=[10.0] * 4, Lambda=[100.0] * 4, k2=[0.1] * 4,
                              tidal_status=[LAMBDA_FRAMEWORK_CAPABILITY] * 4))
    for y in ("Mass", "Lambda", "k2"):
        assert [list(run.attempted_index) for run in _plot_runs(frame, "stellar", "Radius" if y == "Mass" else "Mass", y)] == [[0], [2], [4]]


def test_saved_direct_star_is_the_only_physical_a0_stellar_alias(tmp_path, monkeypatch):
    result = _saved_result(tmp_path, monkeypatch)
    for packet in result.packet_paths:
        for name in ("stellar_sequences.csv", "fixed_mass_observables.csv"):
            frame = pd.read_csv(packet / name)
            if name == "stellar_sequences.csv":
                frame = frame.loc[~frame.case_id.eq("baseline")]
            else:
                frame["case_id"] = "direct"
            frame.to_csv(packet / name, index=False)
    _, tables = _collect_saved_data(result)
    assert tables["stellar"].physical_case_id.eq("baseline").sum() == 4
    assert tables["fixed"].physical_case_id.eq("baseline").sum() == 1
    assert tables["fixed"].case_id.iloc[0] == "direct"


def test_saved_view_is_explicit_atomic_reusable_and_leaves_packets_untouched(tmp_path, monkeypatch):
    result = _saved_result(tmp_path, monkeypatch)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in result.experiment_path.rglob("*") if p.is_file()}
    assert cfl_notebook_view(result, create=False) is None
    view = cfl_notebook_view(result, create=True)
    assert view["scientific_solver_calls"] == view["packet_writes"] == 0
    assert view["physical_eos_count"] == 3 and view["rejected_case_count"] == 2
    assert view["path"] == result.experiment_path.parent / "plots"
    assert set(pd.read_csv(view["path"] / "case_catalogue.csv").eos_label) >= {
        "C000000",
        "C000001",
    }
    for name in ("mass_radius.png", "lambda_mass.png", "k2_mass.png"):
        assert (view["path"] / name).is_file()
    for name in ("thermodynamic_profiles.csv", "stellar_sequences.csv", "README.md"):
        assert (view["path"] / name).is_file()
    inventory = pd.read_csv(view["path"] / "plot_inventory.csv")
    assert inventory.set_index("figure").loc["lambda_mass.png", "omitted_sequence_or_profile_rows"] == 3
    assert inventory.set_index("figure").loc["pressure.png", "status"] == "unavailable"
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in result.experiment_path.rglob("*") if p.is_file()}
    assert after == before
    saved = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in view["path"].iterdir()}
    assert cfl_notebook_view(result, create=True) == view
    assert saved == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in view["path"].iterdir()}
    (view["path"] / "mass_radius.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        cfl_notebook_view(result, create=True)


def test_render_failure_never_publishes_a_partial_view(tmp_path, monkeypatch):
    result = _saved_result(tmp_path, monkeypatch)
    import eos_generation.reporting.notebook_results as reporting

    monkeypatch.setattr(reporting, "_render_view", lambda *args: (_ for _ in ()).throw(RuntimeError("synthetic render failure")))
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        cfl_notebook_view(result, create=True)
    assert sorted(p.name for p in result.experiment_path.parent.iterdir()) == [result.experiment_path.name]


def test_atomic_png_sync_uses_a_writable_descriptor(tmp_path, monkeypatch):
    import os
    import matplotlib.pyplot as plt
    from eos_generation._internal.artifacts import repository_root_scope
    from eos_generation.reporting.plot_helpers import finalize_figure

    original = os.fsync
    descriptors = []

    def writable_sync(descriptor):
        assert os.write(descriptor, b"") == 0
        descriptors.append(descriptor)
        original(descriptor)

    monkeypatch.setattr(os, "fsync", writable_sync)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with repository_root_scope(tmp_path):
        path = finalize_figure(fig, tmp_path / "runs" / "test.png", dpi=50)
    assert descriptors and path.read_bytes().startswith(b"\x89PNG")
    assert not path.with_name("test.png.tmp").exists()

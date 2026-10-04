from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from eos_generation import (
    ExperimentSettings,
    plan_experiment,
    run_experiment,
    load_experiment,
    validate_experiment,
)
from eos_generation import assessment, thermodynamics, stellar
from eos_generation.storage import strict_json, sha256


def test_planning_is_passive_and_deterministic(tmp_path):
    settings = ExperimentSettings.from_values(center=(250, 200), amplitudes=(-0.1, 0.1))
    destination = tmp_path / "runs/study"
    with (
        patch.object(
            thermodynamics,
            "build_consistent_baseline",
            side_effect=AssertionError("solver"),
        ),
        patch.object(stellar, "solve_star", side_effect=AssertionError("solver")),
    ):
        first = plan_experiment(settings, output_path=destination)
        second = plan_experiment(settings, output_path=destination)
    assert not list(tmp_path.iterdir())
    assert first.to_dict() == second.to_dict()
    assert first.estimates["physical_case_count"] == 5
    assert first.estimates["baseline_constructions"] == 2
    assert (
        first.case_table.loc[
            first.case_table.amplitude.eq(0), "physical_case_id"
        ].nunique()
        == 1
    )
    # Changing a detached copy cannot mutate the reviewed plan.
    first.to_dict()["cases"].clear()
    assert len(first.case_table) == 6
    with pytest.raises(ValueError, match="execute=True"):
        run_experiment(first)


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    destination = tmp_path_factory.mktemp("flat_run") / "runs/study"
    plan = plan_experiment(
        ExperimentSettings.from_values(center=(200, 250), amplitudes=(0, -2, 0.01)),
        output_path=destination,
    )
    seen = []
    original = thermodynamics.build_windowed_eos

    def reconstruct(base, proposal, **kwargs):
        # All complete raw evidence is already on disk before any reconstruction.
        raw = pd.read_csv(destination / "data/raw.csv")
        assert raw.case_id.nunique() == 5
        assert proposal.amplitude != -2
        seen.append(proposal.case_id)
        return original(base, proposal, **kwargs)

    with (
        patch.object(
            thermodynamics,
            "build_consistent_baseline",
            wraps=thermodynamics.build_consistent_baseline,
        ) as baseline,
        patch.object(thermodynamics, "build_windowed_eos", side_effect=reconstruct),
        patch.object(
            stellar,
            "_run_stellar",
            side_effect=AssertionError("stellar work forbidden"),
        ),
    ):
        result = run_experiment(plan, execute=True)
    assert baseline.call_count == 2
    assert len(seen) == 6
    return plan, result


def test_flat_packet_raw_rejections_and_no_overwrite(saved_run):
    plan, result = saved_run
    assert {p.name for p in result.data_path.iterdir()} == {
        "run.json",
        "cases.csv",
        "raw.csv",
        "eos.csv",
        "SHA256SUMS.txt",
    }
    cases = result.case_table
    assert set(cases.loc[cases.amplitude.eq(-2), "status"]) == {"rejected"}
    assert cases.loc[cases.amplitude.eq(-2), "failure_reason"].notna().all()
    assert set(result.thermodynamic_profiles.case_id) == set(
        cases.loc[cases.status.eq("accepted"), "physical_case_id"]
    )
    assert {"baryon_density_fm3", "effective_baryon_enthalpy_mev"}.issubset(
        result.thermodynamic_profiles
    )
    assert validate_experiment(result.experiment_path)["passed"]
    before = sha256(result.data_path / "SHA256SUMS.txt")
    with pytest.raises(FileExistsError):
        run_experiment(plan, execute=True)
    assert sha256(result.data_path / "SHA256SUMS.txt") == before


def test_loading_source_drift_is_read_only(saved_run):
    _, result = saved_run
    before = {p.name: sha256(p) for p in result.data_path.iterdir()}
    from eos_generation import storage

    with patch.object(storage, "source_identity", return_value={"sha256": "different"}):
        loaded = load_experiment(result.experiment_path)
        report = validate_experiment(result.experiment_path)
        assert loaded.metadata["status"] == "complete"
        assert report["passed"] and report["source_equivalence"] == "different"
        assert not storage.validate_run(
            result.experiment_path, require_source_equivalence=True
        )["passed"]
    assert before == {p.name: sha256(p) for p in result.data_path.iterdir()}


def test_failure_seals_raw_evidence_and_cannot_load_as_complete(tmp_path):
    destination = tmp_path / "runs/failed"
    plan = plan_experiment(
        ExperimentSettings.from_values(amplitudes=(0, 0.01)), output_path=destination
    )
    with (
        patch.object(
            thermodynamics,
            "build_windowed_eos",
            side_effect=RuntimeError("injected reconstruction failure"),
        ),
        pytest.raises(RuntimeError, match="injected"),
    ):
        run_experiment(plan, execute=True)
    assert strict_json(destination / "data/run.json")["status"] == "failed"
    assert (destination / "data/raw.csv").is_file()
    with pytest.raises(ValueError, match="failed"):
        load_experiment(destination)


def test_stale_plan_does_not_create_destination(tmp_path):
    destination = tmp_path / "runs/stale"
    plan = plan_experiment(ExperimentSettings(), output_path=destination)
    from eos_generation import experiment

    with (
        patch.object(
            experiment, "environment_identity", return_value={"changed": "yes"}
        ),
        pytest.raises(ValueError, match="changed"),
    ):
        run_experiment(plan, execute=True)
    assert not destination.exists()


def test_settings_products_and_profile_authority():
    from eos_generation.numerics import precision_profile

    strict = precision_profile("strict", "stellar")
    assert [
        (s.lower_points, s.upper_points) for s in strict["thermodynamic_stages"]
    ] == [(1025, 2049), (2049, 4097), (4097, 8193)]
    assert [
        (s.sequence_points, s.rtol, s.atol, s.radial_profile_points)
        for s in strict["tov_stages"]
    ] == [(61, 1e-8, 1e-10, 601), (121, 1e-8, 1e-10, 601), (121, 1e-10, 1e-12, 1201)]
    assert (
        strict["raw_gate_lower_points"] == 4097
        and strict["raw_gate_upper_points"] == 16385
    )
    expected = {
        "dataset": (61, 1e-10, 1e-12),
        "dataset_10_tighter": (10, 1e-11, 1e-13),
        "dataset_20": (20, 1e-10, 1e-12),
        "dataset_40": (40, 1e-10, 1e-12),
        "dataset_40_curves": (40, 1e-10, 1e-12),
        "dataset_relaxed": (61, 1e-8, 1e-10),
        "dataset_relaxed_80": (80, 1e-8, 1e-10),
    }
    for name, values in expected.items():
        profile = precision_profile(name, "stellar")
        (stage,) = profile["tov_stages"]
        assert (stage.sequence_points, stage.rtol, stage.atol) == values
        assert stage.radial_profile_points == 1201
        assert len(profile["thermodynamic_stages"]) == (
            1 if name == "dataset_40_curves" else 3
        )
    settings = ExperimentSettings.from_values(
        calculation="stellar", precision="strict", observables=("sequence",)
    )
    assert settings.requested_observables == ("sequence",)
    assert ExperimentSettings.from_dict(settings.to_dict()) == settings
    with pytest.raises(ValueError, match="unknown"):
        ExperimentSettings.from_dict({"rtol": 1e-3})


def test_scientific_corruption_fails_even_after_resealing(saved_run, tmp_path):
    import shutil
    from eos_generation.storage import seal

    _, result = saved_run
    destination = tmp_path / "corrupt"
    shutil.copytree(result.experiment_path, destination)
    eos = pd.read_csv(destination / "data/eos.csv", float_precision="round_trip")
    eos.loc[0, "cs2"] = -0.1
    eos.to_csv(destination / "data/eos.csv", index=False)
    seal(destination / "data")
    report = validate_experiment(destination)
    assert not report["passed"] and "physical predicate" in report["errors"][0]


def test_saved_plotting_keeps_data_manifest_and_calls_no_solver(saved_run):
    from eos_generation.plotting import plot_experiment

    _, result = saved_run
    before = sha256(result.data_path / "SHA256SUMS.txt")
    with (
        patch.object(stellar, "solve_star", side_effect=AssertionError("solver")),
        patch.object(
            thermodynamics,
            "build_consistent_baseline",
            side_effect=AssertionError("solver"),
        ),
    ):
        figures = plot_experiment(result.experiment_path, figures=("pressure", "cs2"))
    assert strict_json(figures / "figures.json")["status"] == "complete"
    assert sha256(result.data_path / "SHA256SUMS.txt") == before
    repeated = plot_experiment(result.experiment_path, figures=("pressure", "cs2"))
    assert repeated == figures
    changed_view = plot_experiment(
        result.experiment_path, figures=("pressure", "cs2"),
        lambda_scale="linear", lambda_mass_limits=(1, 2), lambda_limits=(0, 2000),
    )
    assert changed_view != figures
    assert plot_experiment(
        result.experiment_path, figures=("pressure", "cs2"),
        lambda_scale="linear", lambda_mass_limits=[1, 2], lambda_limits=[0, 2000],
    ) == changed_view
    assert sha256(result.data_path / "SHA256SUMS.txt") == before
    regenerated = plot_experiment(
        result.experiment_path, figures=("pressure", "cs2"), regenerate=True
    )
    assert regenerated != figures and figures.is_dir()

    # A damaged cached image is never reused or repaired in place.
    image = next(figures.glob("*.png"))
    image.write_bytes(b"damaged")
    replacement = plot_experiment(result.experiment_path, figures=("pressure", "cs2"))
    assert replacement == regenerated
    assert image.read_bytes() == b"damaged"
    another_image = next(regenerated.glob("*.png"))
    another_image.write_bytes(b"also damaged")
    fresh = plot_experiment(result.experiment_path, figures=("pressure", "cs2"))
    assert fresh not in (figures, regenerated)
    record = strict_json(fresh / "figures.json")
    assert record["scientific_solver_calls"] == 0 and record["geometry_colors"]
    assert another_image.read_bytes() == b"also damaged"


def test_notebook_destinations_and_repeat_execution_are_passive(saved_run, tmp_path):
    from eos_generation import experiment

    plan, result = saved_run
    first = experiment._new_run_destination(plan.settings, tmp_path / "runs")
    second = experiment._new_run_destination(plan.settings, tmp_path / "runs")
    assert first != second and not list(tmp_path.iterdir())
    with patch.object(
        experiment, "run_experiment", side_effect=AssertionError("solver")
    ):
        repeated = experiment._execute_notebook_plan(plan, plan.settings)
    assert repeated.experiment_path == result.experiment_path
    with pytest.raises(ValueError, match="review"):
        experiment._execute_notebook_plan(None, plan.settings)
    with pytest.raises(ValueError, match="changed"):
        experiment._execute_notebook_plan(plan, ExperimentSettings())
    unfinished = tmp_path / "runs/incomplete"
    unfinished.mkdir(parents=True)
    incomplete_plan = plan_experiment(plan.settings, output_path=unfinished)
    with pytest.raises(ValueError, match="incomplete"):
        experiment._execute_notebook_plan(incomplete_plan, plan.settings)
    assert list(unfinished.iterdir()) == []
    with pytest.raises(ValueError, match="study name"):
        experiment._new_run_destination(plan.settings, tmp_path / "runs", "../escape")


def test_saved_run_selection_is_independent_and_read_only(saved_run):
    from eos_generation import experiment

    _, result = saved_run
    runs = result.experiment_path.parent
    before = {p: sha256(p) for p in runs.rglob("*") if p.is_file()}
    with patch.object(
        experiment, "run_experiment", side_effect=AssertionError("solver")
    ):
        listing = experiment._saved_runs(runs)
        assert listing.iloc[0].status == "complete"
        loaded = experiment._load_saved_run(runs)
        assert loaded.settings == result.settings
        assert (
            experiment._load_saved_run(runs, "study").experiment_path
            == result.experiment_path
        )
        with pytest.raises(ValueError, match="Cannot load"):
            experiment._load_saved_run(runs, "missing")
        with pytest.raises(ValueError, match="folder name"):
            experiment._load_saved_run(runs, "../escape")
    assert before == {p: sha256(p) for p in runs.rglob("*") if p.is_file()}


def test_every_supported_profile_product_diagnostic_combination_is_passive(tmp_path):
    from itertools import product
    from eos_generation import settings as settings_module

    combinations = 0
    with (
        patch.object(
            thermodynamics,
            "build_consistent_baseline",
            side_effect=AssertionError("solver"),
        ),
        patch.object(stellar, "_run_stellar", side_effect=AssertionError("solver")),
    ):
        for precision, calculation in product(
            settings_module._PRECISIONS, ("thermodynamics", "stellar")
        ):
            choices = (
                [()]
                if calculation == "thermodynamics"
                else [
                    ("sequence",),
                    ("sequence", "fixed_mass"),
                    ("sequence", "maximum_mass"),
                    ("sequence", "fixed_mass", "maximum_mass"),
                ]
            )
            for observables, diagnostics in product(choices, ("off", "on")):
                values = dict(
                    precision=precision,
                    calculation=calculation,
                    observables=observables,
                    diagnostics=diagnostics,
                    center=(200, 250),
                    width=(50, 100),
                    ramp_width=(40, 60),
                    amplitudes=(-0.01, 0.01),
                    fixed_masses=(1.2, 1.4),
                )
                if diagnostics == "on" and "fixed_mass" not in observables:
                    with pytest.raises(ValueError, match="requires"):
                        ExperimentSettings.from_values(**values)
                    continue
                settings = ExperimentSettings.from_values(**values)
                planned = plan_experiment(settings, output_path=tmp_path / "runs/study")
                assert planned.settings == ExperimentSettings.from_dict(
                    settings.to_dict()
                )
                assert planned.case_table.geometry_index.nunique() == 8
                assert planned.estimates["physical_case_count"] == 17
                assert planned.to_dict()["scientific_solver_calls"] == 0
                assert planned.to_dict()["filesystem_writes"] == 0
                combinations += 1
    assert combinations == 63
    assert not list(tmp_path.iterdir())


def test_renderer_failure_and_unknown_plot_leave_no_files(saved_run, tmp_path):
    from types import SimpleNamespace
    from eos_generation import plotting

    _, result = saved_run
    with (
        patch.object(plotting, "_RENDER_CHECKED", False),
        patch.object(
            plotting.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=1, stderr="native failure"),
        ),
        pytest.raises(RuntimeError, match="isolated check"),
    ):
        result.generate_plots(output_path=tmp_path / "figures")
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="Mass–radius"):
        result.generate_plots(
            figures=["radius", "mass"], output_path=tmp_path / "figures"
        )
    assert not list(tmp_path.iterdir())


def test_notebook_loads_after_restart_without_current_json(saved_run):
    import nbformat
    from nbclient import NotebookClient
    import os

    _, result = saved_run
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/bsk24_experiment.ipynb", as_version=4)
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            """from eos_generation import thermodynamics, stellar
def forbidden(*args, **kwargs):
    raise AssertionError("load attempted scientific execution")
thermodynamics.build_consistent_baseline = forbidden
thermodynamics.build_windowed_eos = forbidden
stellar._run_stellar = forbidden
stellar.solve_star = forbidden
"""
        ),
    )
    control = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "NOTEBOOK_SETTINGS = {" in cell.source
    )
    control.source = control.source.replace(
        'ACTION = "plan"', 'ACTION = "load"'
    ).replace('PLOTS = "auto"', 'PLOTS = "none"')
    control.source = control.source.replace(
        'RUNS = ROOT / "runs"',
        "RUNS = Path(" + repr(str(result.experiment_path.parent)) + ")",
    )
    control.source = control.source.replace(
        'CONFIG_FILE = "configs/quickstart.json"', 'CONFIG_FILE = "missing-config.json"'
    )
    control.source = control.source.replace(
        'SETTINGS_SOURCE = "notebook"', 'SETTINGS_SOURCE = "json"'
    )
    before = {p: sha256(p) for p in result.experiment_path.rglob("*") if p.is_file()}
    NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute(env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert before == {
        p: sha256(p) for p in result.experiment_path.rglob("*") if p.is_file()
    }


def test_two_cell_notebook_can_review_and_reload_completed_plan(saved_run):
    import os
    import nbformat
    from nbclient import NotebookClient
    from test_notebook import configured_code

    plan, result = saved_run
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/bsk24_experiment.ipynb", as_version=4)
    source = configured_code(
        notebook.cells[1].source,
        NOTEBOOK_SETTINGS=plan.settings.to_dict(),
        PLOTS="none",
        TABLES="none",
        SHOW_GUIDES=False,
    )
    source = source.replace(
        "_new_run_destination(settings, RUNS, STUDY)",
        f"Path({str(result.experiment_path)!r})",
    )
    source = source.replace(
        'RUNS = ROOT / "runs"', f"RUNS = Path({str(result.experiment_path.parent)!r})"
    )
    guard = nbformat.v4.new_code_cell(
        """from eos_generation import experiment
def forbidden(*args, **kwargs):
    raise AssertionError("completed-plan notebook attempted scientific execution")
experiment.run_experiment = forbidden
"""
    )
    notebook.cells = [
        notebook.cells[0],
        guard,
        nbformat.v4.new_code_cell(source),
        nbformat.v4.new_code_cell(configured_code(source, ACTION="execute")),
        nbformat.v4.new_code_cell(configured_code(source, ACTION="execute")),
    ]
    before = {p: sha256(p) for p in result.experiment_path.rglob("*") if p.is_file()}
    NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute(env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert before == {
        p: sha256(p) for p in result.experiment_path.rglob("*") if p.is_file()
    }


def test_notebook_catalogue_matches_profiles_products_and_saved_tables(tmp_path):
    from eos_generation import experiment
    from eos_generation.numerics import precision_profile
    from eos_generation.plotting import FIGURES

    settings = ExperimentSettings.from_values(
        calculation="stellar", precision="dataset_40_curves"
    )
    with patch.object(stellar, "solve_star", side_effect=AssertionError("solver")):
        guides = experiment._notebook_option_tables(settings)
    assert set(guides["plots"].key) == set(FIGURES)
    assert len(guides["tables"]) == 13
    assert not guides["products"].set_index("product").loc["fixed_mass", "requested"]
    for row in guides["profiles"].itertuples():
        profile = precision_profile(row.precision, "stellar")
        assert row.sequence_attempts_per_case == sum(
            stage.sequence_points for stage in profile["tov_stages"]
        )
    (tmp_path / "runs/interrupted").mkdir(parents=True)
    listing = experiment._saved_runs(tmp_path / "runs")
    assert listing.iloc[0].status == "incomplete"
    assert {"geometry", "amplitudes", "observables", "diagnostics"}.issubset(listing)


def test_diagnostic_table_access_is_read_only_and_rejects_paths(tmp_path):
    from eos_generation import ExperimentResult

    (tmp_path / "data").mkdir()
    path = tmp_path / "data/baryonic_observables.csv"
    pd.DataFrame({"baryonic_mass_msun": [1.5], "binding_energy_erg": [1.0e53]}).to_csv(
        path, index=False
    )
    result = ExperimentResult(tmp_path, {})
    before = sha256(path)
    assert result.table("baryonic_observables").baryonic_mass_msun.iloc[0] == 1.5
    assert result.table("odd_even_response").empty
    with pytest.raises(ValueError, match="unknown table"):
        result.table("../../outside")
    assert sha256(path) == before


@pytest.mark.parametrize(
    "values", [("radius", "auto", 12), ("auto", ["unknown"], 12), ("auto", "auto", 0)]
)
def test_invalid_notebook_display_options_fail_before_execution(values):
    from eos_generation.experiment import _notebook_display_options

    with pytest.raises(ValueError):
        _notebook_display_options(*values)


@pytest.mark.parametrize("matter_model", ["bsk24", "bsk25"])
def test_flat_stellar_route_keeps_failure_gaps_and_requested_products(tmp_path, matter_model):
    settings = ExperimentSettings.from_values(
        matter_model=matter_model, calculation="stellar", amplitudes=(0, -2, 0.01), observables=("sequence",)
    )
    plan = plan_experiment(settings, output_path=tmp_path / "runs/stellar")

    def stellar_tables(*, config, baseline, generated):
        assert config.matter_model == baseline.eos.matter_model == matter_model
        assert set(eos.deformation.amplitude for eos in generated.values()) == {0.01}
        assert (
            not config.fixed_mass_background_requested
            and not config.maximum_mass_requested
        )
        assert not config.retained_stellar_profiles_requested
        rows = []
        evidence = {}
        for case_id in ("direct", *generated):
            stage = config.tov_stages[-1].name
            for index in range(3):
                rows.append(
                    {
                        "case_id": case_id,
                        "stage": stage,
                        "attempted_index": index,
                        "segment_id": 0 if index < 2 else 1,
                        "calculation_status": "failed" if index == 1 else "success",
                        "failure_reason": "synthetic_gap" if index == 1 else None,
                        "central_pressure_mev_fm3": index + 2.0,
                        "Mass": None if index == 1 else 1.0 + index * 0.1,
                        "Radius": None if index == 1 else 12.0,
                        "Lambda": None if index == 1 else 500.0,
                        "k2": None if index == 1 else 0.1,
                        "tidal_status": (
                            None if index == 1 else "validated_lambda_validation_v1"
                        ),
                        "is_on_successful_stable_prefix": index != 1,
                    }
                )
            evidence[case_id + ":" + stage] = {
                "attempted_count": 3,
                "successful_count": 2,
                "failed_count": 1,
                "stable_prefix_count": 2,
            }
        return (
            pd.DataFrame(rows),
            pd.DataFrame(),
            {
                "maximum_mass_rows": [],
                "maximum_mass_reports": {},
                "sequence_evidence": evidence,
            },
            {},
        )

    with patch.object(stellar, "_run_stellar", side_effect=stellar_tables):
        result = run_experiment(plan, execute=True)
    assert (result.data_path / "stars.csv").exists()
    assert not (result.data_path / "fixed_mass.csv").exists()
    assert not (result.data_path / "maximum_mass.csv").exists()
    assert result.stellar_sequences.calculation_status.eq("failed").sum() == 2
    assert result.stellar_sequences.case_id.nunique() == 2


def test_diagnostics_are_scoped_per_geometry_without_new_solver_calls(tmp_path):
    from types import SimpleNamespace
    from eos_generation import diagnostics, experiment

    settings = ExperimentSettings.from_values(
        calculation="stellar",
        diagnostics="on",
        center=(200, 250),
        amplitudes=(-0.1, 0.1),
    )
    configs = experiment._configs(settings)
    generated = {}
    rows = []
    for index, config in enumerate(configs):
        for amplitude in (-0.1, 0.1):
            key = f"geometry-{index}-{amplitude}"
            generated[key] = SimpleNamespace(
                deformation=SimpleNamespace(
                    amplitude=amplitude,
                    epsilon0_mev_fm3=config.epsilon0_mev_fm3,
                    sigma_mev_fm3=config.sigma_mev_fm3,
                    delta_mev_fm3=config.deltas_mev_fm3[0],
                )
            )
            rows.append({"case_id": key, "amplitude": amplitude, "delta_mev_fm3": 40.0})
    frame = pd.DataFrame(
        [{"case_id": "baseline", "amplitude": 0.0, "delta_mev_fm3": 40.0}, *rows]
    )
    observed = []

    def diagnostic_tables(*, config, generated, sequences, fixed, collect, **kwargs):
        assert len(generated) == 2
        assert {eos.deformation.epsilon0_mev_fm3 for eos in generated.values()} == {
            config.epsilon0_mev_fm3
        }
        assert set(fixed.case_id) == {"direct", *generated}
        observed.append(config.epsilon0_mev_fm3)
        collect(
            pd.DataFrame({"case_id": ["direct", *generated], "value": [0, 1, 2]}),
            "radial_profiles.csv",
        )
        return {"radial_profiles.csv": "synthetic geometry-scoped evidence"}

    with (
        patch.object(
            diagnostics, "_extended_diagnostics", side_effect=diagnostic_tables
        ),
        patch.object(stellar, "solve_star", side_effect=AssertionError("solver")),
    ):
        evidence = diagnostics.write_diagnostics(
            packet=tmp_path,
            configs=configs,
            baseline=None,
            generated=generated,
            sequences=frame,
            fixed=frame,
            stars={},
            baseline_id="baseline",
        )
    assert observed == [200, 250]
    saved = pd.read_csv(tmp_path / "radial_profiles.csv")
    assert saved.geometry_index.value_counts().to_dict() == {1: 3, 2: 3}
    assert set(saved.case_id) == {"baseline", *generated}
    assert evidence["selected_case_ids_by_geometry"]["1"][0] == "baseline"

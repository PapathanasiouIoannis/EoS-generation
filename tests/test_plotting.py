"""Saved-table selection tests use synthetic statuses, never stellar solvers."""

from types import SimpleNamespace

import pandas as pd
import pytest

from eos_generation import ExperimentSettings
from eos_generation.plotting import _prepare, _names, _render, _geometry_colors, _view_options


@pytest.fixture
def saved_tables(tmp_path):
    cases = pd.DataFrame(
        [
            dict(
                geometry_index=g,
                epsilon0_mev_fm3=200 + 50 * g,
                sigma_mev_fm3=50,
                delta_mev_fm3=40,
                amplitude=a,
                physical_case_id="baseline" if a == 0 else f"g{g}-{a}",
                status="rejected" if a == -2 else "accepted",
            )
            for g in (1, 2)
            for a in (0, 0.1, -2)
        ]
    )
    ids = ["baseline", "g1-0.1", "g2-0.1"]
    stars = pd.DataFrame(
        [
            dict(
                case_id=case,
                stage=stage,
                attempted_index=i,
                calculation_status="failed" if i == 1 else "success",
                Mass=1 + i / 10,
                Radius=12,
                Lambda=500,
                k2=0.1,
                tidal_status=(
                    "failed_closed" if i == 2 else "validated_lambda_validation_v1"
                ),
                is_on_successful_stable_prefix=i == 0,
            )
            for case in ids
            for stage in ("coarse", "fine")
            for i in range(3)
        ]
    )
    fixed = pd.DataFrame(
        [
            dict(
                case_id=case,
                stage="fine",
                target_mass_msun=mass,
                radius_km=12,
                lambda_dimensionless=500,
                k2=0.1,
                status="bracketed_and_solved" if mass == 1.2 else "not_bracketed",
                tidal_status="validated_lambda_validation_v1",
            )
            for case in ids
            for mass in (1.2, 1.4)
        ]
    )
    maximum = pd.DataFrame(
        [
            dict(
                case_id=case,
                stage="fine",
                maximum_mass_msun=2.1,
                maximum_mass_resolved=case == "baseline",
            )
            for case in ids
        ]
    )
    response = pd.DataFrame(
        [
            dict(case_id=case, mass_msun=1.2, delta_lambda=10, delta_k2=0.001)
            for case in ids
        ]
    )
    for name, table in dict(
        stars=stars,
        fixed_mass=fixed,
        maximum_mass=maximum,
        stellar_response_across_mass=response,
    ).items():
        table.to_csv(tmp_path / (name + ".csv"), index=False)
    return SimpleNamespace(
        case_table=cases,
        data_path=tmp_path,
        settings=ExperimentSettings.from_values(
            calculation="stellar", fixed_masses=(1.2, 1.4)
        ),
        metadata={
            "numerical_profile": {"tov_stages": [{"name": "coarse"}, {"name": "fine"}]}
        },
    )


def test_failure_statuses_mask_finite_values_and_preserve_gaps(saved_tables):
    cases, frames, report, selection = _prepare(
        saved_tables, geometry=2, amplitudes=0.1
    )
    assert set(cases.geometry_index) == {2}
    assert set(frames["mr"].case_id) == {"baseline", "g2-0.1"}
    assert set(frames["mr"].stage) == {"fine"}
    assert len(frames["mr"]) == 6
    assert frames["mr"].Mass.isna().sum() == 2
    assert frames["lambda"].Lambda.isna().sum() == 4
    assert frames["fixed_radius"].radius_km.isna().sum() == 2
    assert frames["maximum_mass"].maximum_mass_msun.notna().sum() == 1
    assert frames["maximum_mass"].amplitude.iloc[0] == 0
    availability = {item["name"]: item for item in report}
    assert availability["mr"]["omitted_rows"] == 2
    assert not availability["pressure"]["available"]
    assert selection["geometry"] == [2]


def test_unavailable_mass_and_stage_are_explicit(saved_tables):
    _, frames, report, _ = _prepare(saved_tables, fixed_masses=1.4)
    assert frames["fixed_radius"].radius_km.isna().all()
    assert not next(item for item in report if item["name"] == "fixed_radius")[
        "available"
    ]
    _, frames, _, _ = _prepare(saved_tables, stage="coarse")
    assert set(frames["mr"].stage) == {"coarse"}
    assert frames["tidal_response"].delta_lambda.isna().all()
    for selection in (
        dict(geometry=3),
        dict(amplitudes=[]),
        dict(fixed_masses=2),
        dict(stage="missing"),
    ):
        with pytest.raises(ValueError):
            _prepare(saved_tables, **selection)


def test_tidal_responses_require_valid_baseline_capability(saved_tables):
    stars_path = saved_tables.data_path / "stars.csv"
    stars = pd.read_csv(stars_path)
    stars.loc[stars.case_id.eq("baseline"), "tidal_status"] = "failed_closed"
    stars.to_csv(stars_path, index=False)
    _, frames, _, _ = _prepare(saved_tables)
    assert frames["tidal_response"].delta_lambda.isna().all()
    stars.drop(columns="is_on_successful_stable_prefix").to_csv(stars_path, index=False)
    _, frames, _, _ = _prepare(saved_tables)
    assert frames["love_response"].delta_k2.isna().all()


def test_readable_plot_names_are_unambiguous():
    assert _names(
        ["Mass-radius", "Tidal deformability", "Fixed-mass radius", "mr"]
    ) == ["mr", "lambda", "fixed_radius"]
    with pytest.raises(ValueError, match="Unknown figure"):
        _names("radius")


def test_overlay_has_geometry_colors_one_baseline_and_preserves_failed_gaps(saved_tables, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.axes import Axes

    cases, frames, _, _ = _prepare(saved_tables)
    calls = []
    original = Axes.plot

    def capture(axis, x, y, **kwargs):
        calls.append((np.asarray(x), np.asarray(y), kwargs))
        return original(axis, x, y, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture)
    assert _render("mr", frames["mr"], cases, saved_tables.data_path) == ["mr.png"]
    assert len(list(saved_tables.data_path.glob("*.png"))) == 1
    baseline = [call for call in calls if call[2]["label"].startswith("BSk24")]
    assert len(baseline) == 1 and baseline[0][2]["color"] == "black"
    deformed = [call for call in calls if call[2]["label"].startswith("G")]
    assert {call[2]["label"] for call in deformed} == {"G1; A=0.1", "G2; A=0.1"}
    assert len({call[2]["color"] for call in deformed}) == 2
    for x, y, _ in calls:
        assert list(x) == [12, 12, 12]  # radius remains on x
        assert y[0] == 1 and np.isnan(y[1]) and y[2] == 1.2
    assert not plt.get_fignums()


def test_response_curves_do_not_connect_geometries_or_target_masses(saved_tables, monkeypatch):
    from matplotlib.axes import Axes
    import numpy as np

    cases, frames, _, _ = _prepare(saved_tables)
    calls = []
    original = Axes.plot

    def capture(axis, x, y, **kwargs):
        calls.append((np.asarray(x), np.asarray(y), kwargs))
        return original(axis, x, y, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture)
    _render("fixed_radius", frames["fixed_radius"], cases, saved_tables.data_path)
    curves = [c for c in calls if c[2].get("label", "").startswith("G")]
    assert {c[2]["label"] for c in curves} == {
        f"G{geometry}; M={mass:g} M☉" for geometry in (1, 2) for mass in (1.2, 1.4)
    }
    for x, y, options in curves:
        assert list(x) == [0, 0.1]
        if "M=1.4" in options["label"]:
            assert np.isnan(y).all()
    colors = _geometry_colors(cases.geometry_index)
    assert all(c[2]["color"] == colors[int(c[2]["label"][1])] for c in curves)


def test_lambda_log_and_zoom_are_view_only_and_signed_response_stays_linear(saved_tables, monkeypatch):
    import matplotlib.pyplot as plt
    import numpy as np

    cases, frames, _, _ = _prepare(saved_tables)
    frame = frames["lambda"]
    frame.loc[frame.case_id.eq("baseline"), "Lambda"] = [1e7, np.nan, 100]
    before = frame.copy(deep=True)
    axes = []
    original = plt.close

    def capture_close(figure):
        if hasattr(figure, "axes"):
            axes.append(figure.axes[0])
        original(figure)

    monkeypatch.setattr(plt, "close", capture_close)
    _render("lambda", frame, cases, saved_tables.data_path)
    assert axes[-1].get_yscale() == "log"
    assert axes[-1].lines[0].get_ydata()[0] == 1e7
    _render("lambda", frame, cases, saved_tables.data_path,
            lambda_scale="linear", lambda_mass_limits=(1, 2), lambda_limits=(0, 2000))
    assert axes[-1].get_yscale() == "linear"
    assert axes[-1].get_xlim() == (1, 2) and axes[-1].get_ylim() == (0, 2000)
    assert "Display limits selected" in axes[-1].get_title()
    pd.testing.assert_frame_equal(frame, before)
    _render("tidal_response", frames["tidal_response"], cases, saved_tables.data_path)
    assert axes[-1].get_yscale() == "linear"


@pytest.mark.parametrize("view", [
    dict(lambda_scale="symlog"), dict(lambda_limits=(0, 2000)),
    dict(lambda_limits=(100, 10)), dict(lambda_limits=(True, 2000)),
    dict(lambda_mass_limits=(1, float("inf"))), dict(lambda_mass_limits=(-1, 2)),
    dict(lambda_limits=[1]),
])
def test_invalid_lambda_view_is_refused_before_any_write(view, tmp_path):
    from eos_generation.plotting import plot_experiment

    with pytest.raises(ValueError):
        plot_experiment(tmp_path / "nonexistent", output_path=tmp_path / "figures", **view)
    assert not list(tmp_path.iterdir())


def test_view_normalizes_tuple_limits_for_portable_cache():
    assert _view_options("linear", (1, 2), (0, 2000)) == _view_options("linear", [1, 2], [0, 2000])


def test_filter_keeps_global_geometry_color_and_legacy_controls_distinct(saved_tables, monkeypatch):
    from matplotlib.axes import Axes

    cases, frames, _, _ = _prepare(saved_tables, geometry=2)
    baseline = frames["mr"].loc[frames["mr"].case_id.eq("baseline")].copy()
    frame = frames["mr"].copy()
    frame["legacy_execution_role"] = "original"
    baseline["legacy_execution_role"] = "repeated_control"
    frame = pd.concat([frame, baseline], ignore_index=True)
    colors = _geometry_colors(range(1, 27))
    calls = []
    original = Axes.plot

    def capture(axis, x, y, **kwargs):
        calls.append(kwargs)
        return original(axis, x, y, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture)
    _render("mr", frame, cases, saved_tables.data_path, geometry_colors=colors)
    baselines = [c for c in calls if c["label"].startswith("BSk24")]
    assert {c["label"] for c in baselines} == {
        "BSk24 (A=0); original", "BSk24 (A=0); repeated_control"
    }
    assert next(c for c in calls if c["label"].startswith("G2"))["color"] == colors[2]


def test_many_geometry_overlay_uses_one_image_with_color_key(saved_tables):
    cases, frames, _, _ = _prepare(saved_tables)
    declarations, rows = [], []
    for geometry in range(1, 27):
        case = cases.loc[cases.geometry_index.eq(1)].copy()
        case["geometry_index"] = geometry
        case.loc[case.amplitude.ne(0), "physical_case_id"] = f"g{geometry}-0.1"
        declarations.append(case)
        frame = frames["mr"].loc[frames["mr"].case_id.eq("g1-0.1")].copy()
        frame["case_id"] = f"g{geometry}-0.1"
        rows.append(frame)
    colors = _geometry_colors(range(1, 27))
    assert len(set(colors.values())) == 26
    assert _render("mr", pd.concat(rows), pd.concat(declarations), saved_tables.data_path) == ["mr.png"]

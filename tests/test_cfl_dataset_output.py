from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from eos_generation.reporting.cfl_dataset import (
    EOS_DATA_FILE,
    OUTPUT_FILES,
    STELLAR_DATA_FILE,
    build_cfl_dataset_output,
)


class _SavedChild:
    def __init__(
        self,
        packet_path: Path,
        amplitude: float,
        suffix: str,
        *,
        direct_baseline: bool,
    ) -> None:
        self.packet_path = packet_path
        self.config = SimpleNamespace(
            tov_stages=(SimpleNamespace(name="dataset_40"),)
        )
        self.calls: dict[str, int] = {}
        case_id = f"accepted_{suffix}"
        zero_id = f"zero_{suffix}"
        self.tables = {
            "case_ledger.csv": pd.DataFrame(
                [
                    {
                        "case_id": zero_id,
                        "amplitude": 0.0,
                        "epsilon0_mev_fm3": 450.0,
                        "sigma_mev_fm3": 300.0,
                        "delta_mev_fm3": 150.0,
                        "status": "accepted",
                    },
                    {
                        "case_id": case_id,
                        "amplitude": amplitude,
                        "epsilon0_mev_fm3": 450.0,
                        "sigma_mev_fm3": 300.0,
                        "delta_mev_fm3": 150.0,
                        "status": "accepted",
                    },
                    {
                        "case_id": f"rejected_{suffix}",
                        "amplitude": 0.9,
                        "epsilon0_mev_fm3": 450.0,
                        "sigma_mev_fm3": 300.0,
                        "delta_mev_fm3": 150.0,
                        "status": "rejected",
                    },
                ]
            ),
            "thermodynamic_profiles.csv": pd.DataFrame(
                [
                    {
                        "case_id": source,
                        "epsilon_mev_fm3": epsilon,
                        "pressure_mev_fm3": pressure + offset,
                        "cs2": 0.35 + offset / 100.0,
                    }
                    for source, offset in (
                        (("direct", 0.0),) if direct_baseline else ()
                    ) + ((zero_id, 0.0), (case_id, amplitude))
                    for epsilon, pressure in (
                        (200.0, 0.0),
                        (400.0, 60.0),
                        (800.0, 210.0),
                    )
                ]
            ),
            "stellar_sequences.csv": pd.DataFrame(
                [
                    {
                        "case_id": source,
                        "stage": "dataset_40",
                        "attempted_index": index,
                        "calculation_status": "success",
                        "failure_category": "",
                        "failure_reason": "",
                        "central_pressure_mev_fm3": pressure,
                        "Mass": mass + offset,
                        "Radius": radius,
                        "Lambda": tidal,
                        "k2": k2,
                        "tidal_status": "validated_lambda_validation_v1",
                        "tidal_failure_reason": "",
                        "is_sampled_peak": index == 2,
                    }
                    for source, offset in (
                        (("direct", 0.0),) if direct_baseline else ()
                    ) + ((case_id, amplitude),)
                    for index, pressure, mass, radius, tidal, k2 in (
                        (0, 2.0, 0.1, 5.0, 1.0e7, 0.55),
                        (1, 20.0, 1.0, 9.0, 1.0e3, 0.20),
                        (2, 100.0, 2.0, 10.0, 50.0, 0.08),
                    )
                ]
            ),
        }

    def table(self, name: str) -> pd.DataFrame:
        self.calls[name] = self.calls.get(name, 0) + 1
        return self.tables[name].copy()


def test_minimal_cfl_dataset_is_exactly_two_tables_and_five_plots(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs" / "cfl_test"
    experiment = run_root / "experiment_abc"
    run_root.mkdir(parents=True)
    children = (
        _SavedChild(
            experiment / "geometry_001", -0.1, "one", direct_baseline=True
        ),
        _SavedChild(
            experiment / "geometry_002", 0.1, "two", direct_baseline=False
        ),
    )
    result = SimpleNamespace(
        repository_root=tmp_path,
        experiment_path=experiment,
        settings=SimpleNamespace(matter_model="cfl", calculation="stellar"),
        completed=True,
        child_results=children,
    )
    destination = run_root / "CFL_DATASET"

    summary = build_cfl_dataset_output(result, destination)

    assert summary["solver_calls"] == 0
    assert summary["eos_count"] == 3
    assert {path.name for path in destination.iterdir()} == set(OUTPUT_FILES)
    assert all(path.is_file() for path in destination.iterdir())
    eos = pd.read_csv(destination / EOS_DATA_FILE)
    stellar = pd.read_csv(destination / STELLAR_DATA_FILE)
    assert list(dict.fromkeys(eos["label"])) == ["cfl_0", "cfl_1", "cfl_2"]
    assert set(stellar["label"]) == {"cfl_0", "cfl_1", "cfl_2"}
    assert not eos["case_id"].astype(str).str.startswith("zero_").any()
    assert not eos["case_id"].astype(str).str.startswith("rejected_").any()
    assert not stellar["case_id"].astype(str).str.startswith("rejected_").any()
    for name in set(OUTPUT_FILES) - {EOS_DATA_FILE, STELLAR_DATA_FILE}:
        assert (destination / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for child in children:
        assert child.calls == {
            "case_ledger.csv": 1,
            "thermodynamic_profiles.csv": 1,
            "stellar_sequences.csv": 1,
        }

    reused = build_cfl_dataset_output(result, destination)
    assert reused["reused"] is True
    for child in children:
        assert all(count == 1 for count in child.calls.values())


def _result_with_direct_baselines(
    tmp_path: Path, direct_baselines: tuple[bool, bool]
) -> SimpleNamespace:
    experiment = tmp_path / "runs" / "cfl_test" / "experiment_abc"
    experiment.parent.mkdir(parents=True)
    children = tuple(
        _SavedChild(
            experiment / f"geometry_{index:03d}",
            amplitude,
            suffix,
            direct_baseline=direct_baseline,
        )
        for index, amplitude, suffix, direct_baseline in (
            (1, -0.1, "one", direct_baselines[0]),
            (2, 0.1, "two", direct_baselines[1]),
        )
    )
    return SimpleNamespace(
        repository_root=tmp_path,
        experiment_path=experiment,
        settings=SimpleNamespace(matter_model="cfl", calculation="stellar"),
        completed=True,
        child_results=children,
    )


def test_direct_baseline_owner_may_be_later_child(tmp_path: Path) -> None:
    result = _result_with_direct_baselines(tmp_path, (False, True))
    destination = result.experiment_path.parent / "CFL_DATASET"

    summary = build_cfl_dataset_output(result, destination)

    assert summary["eos_count"] == 3
    eos = pd.read_csv(destination / EOS_DATA_FILE)
    stellar = pd.read_csv(destination / STELLAR_DATA_FILE)
    assert list(dict.fromkeys(eos["label"])) == ["cfl_0", "cfl_1", "cfl_2"]
    assert list(dict.fromkeys(stellar["label"])) == ["cfl_0", "cfl_1", "cfl_2"]


def test_missing_direct_baseline_fails_closed(tmp_path: Path) -> None:
    result = _result_with_direct_baselines(tmp_path, (False, False))

    with pytest.raises(ValueError, match="no direct baseline data"):
        build_cfl_dataset_output(
            result, result.experiment_path.parent / "CFL_DATASET"
        )


def test_duplicate_direct_baseline_packets_fail_closed(tmp_path: Path) -> None:
    result = _result_with_direct_baselines(tmp_path, (True, True))

    with pytest.raises(ValueError, match="duplicate direct baseline packets"):
        build_cfl_dataset_output(
            result, result.experiment_path.parent / "CFL_DATASET"
        )


def test_baseline_only_experiment_exports_one_eos(tmp_path: Path) -> None:
    result = _result_with_direct_baselines(tmp_path, (True, False))
    owner, nonowner = result.child_results
    owner.tables["case_ledger.csv"] = owner.tables["case_ledger.csv"].loc[
        owner.tables["case_ledger.csv"]["amplitude"].eq(0.0)
    ]
    owner.tables["thermodynamic_profiles.csv"] = owner.tables[
        "thermodynamic_profiles.csv"
    ].loc[lambda frame: frame["case_id"].eq("direct")]
    owner.tables["stellar_sequences.csv"] = owner.tables[
        "stellar_sequences.csv"
    ].loc[lambda frame: frame["case_id"].eq("direct")]
    nonowner.tables["case_ledger.csv"] = nonowner.tables[
        "case_ledger.csv"
    ].loc[lambda frame: frame["amplitude"].eq(0.0)]
    nonowner.tables["thermodynamic_profiles.csv"] = nonowner.tables[
        "thermodynamic_profiles.csv"
    ].iloc[0:0]
    nonowner.tables["stellar_sequences.csv"] = nonowner.tables[
        "stellar_sequences.csv"
    ].iloc[0:0]
    destination = result.experiment_path.parent / "CFL_DATASET"

    summary = build_cfl_dataset_output(result, destination)

    assert summary["eos_count"] == 1
    assert set(pd.read_csv(destination / EOS_DATA_FILE)["label"]) == {"cfl_0"}
    assert set(pd.read_csv(destination / STELLAR_DATA_FILE)["label"]) == {"cfl_0"}


def test_existing_output_rejects_invalid_png(tmp_path: Path) -> None:
    result = _result_with_direct_baselines(tmp_path, (True, False))
    destination = result.experiment_path.parent / "CFL_DATASET"
    build_cfl_dataset_output(result, destination)
    (destination / "mass_radius.png").write_bytes(b"not a PNG")

    with pytest.raises(FileExistsError, match="invalid PNG"):
        build_cfl_dataset_output(result, destination)


def test_existing_output_rejects_child_symlink(tmp_path: Path) -> None:
    result = _result_with_direct_baselines(tmp_path, (True, False))
    destination = result.experiment_path.parent / "CFL_DATASET"
    build_cfl_dataset_output(result, destination)
    outside = tmp_path / "outside.csv"
    outside.write_text("experiment\nwrong\n", encoding="utf-8")
    linked = destination / EOS_DATA_FILE
    linked.unlink()
    try:
        linked.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    with pytest.raises(FileExistsError, match="regular-file"):
        build_cfl_dataset_output(result, destination)

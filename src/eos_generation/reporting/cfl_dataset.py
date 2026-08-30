"""Minimal CFL dataset export from one completed, validated experiment.

The authoritative child packets retain the governed scientific evidence.  This
adapter reads each required saved table once and publishes only two labelled
CSV tables and the five requested figures.  It never calls a scientific
solver, validates a packet again, allocates a global catalogue ID, or copies
technical tables.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EOS_DATA_FILE = "cfl_eos_data.csv"
STELLAR_DATA_FILE = "cfl_stellar_data.csv"
FIGURE_FILES = (
    "mass_radius.png",
    "lambda_mass.png",
    "k2_mass.png",
    "pressure_energy_density.png",
    "speed_of_sound.png",
)
OUTPUT_FILES = (EOS_DATA_FILE, STELLAR_DATA_FILE, *FIGURE_FILES)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_TIDAL_VALID_STATUS = "validated_lambda_validation_v1"


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _portable(path: Path, root: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    table: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required column {missing[0]!r}")


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.lower().eq("true")


def _sequence_xy(
    rows: pd.DataFrame,
    x_name: str,
    y_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return saved pressure-order values through the sampled peak with gaps."""

    if rows.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    ordered = rows.copy()
    ordered["attempted_index"] = pd.to_numeric(
        ordered["attempted_index"], errors="raise"
    )
    ordered = ordered.sort_values("attempted_index", kind="stable")
    peaks = _truthy(ordered["is_sampled_peak"])
    if not peaks.any():
        return np.array([], dtype=float), np.array([], dtype=float)
    peak_index = ordered.loc[peaks, "attempted_index"].min()
    ordered = ordered.loc[ordered["attempted_index"].le(peak_index)]
    valid = ordered["calculation_status"].astype(str).eq("success")
    if y_name in {"k2", "Lambda"}:
        valid &= ordered["tidal_status"].astype(str).eq(
            _TIDAL_VALID_STATUS
        )
    x = pd.to_numeric(ordered[x_name], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(ordered[y_name], errors="coerce").to_numpy(dtype=float)
    finite = valid.to_numpy() & np.isfinite(x) & np.isfinite(y)
    if y_name == "Lambda":
        finite &= y > 0.0
    x[~finite] = np.nan
    y[~finite] = np.nan
    return x, y


def _case_metadata(
    row: pd.Series,
    *,
    label: str,
    geometry_id: str,
    experiment: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "experiment": experiment,
        "geometry_id": geometry_id,
        "case_id": str(row["case_id"]),
        "status": "accepted",
        "amplitude": float(row["amplitude"]),
        "epsilon0_mev_fm3": float(row["epsilon0_mev_fm3"]),
        "sigma_mev_fm3": float(row["sigma_mev_fm3"]),
        "delta_mev_fm3": float(row["delta_mev_fm3"]),
    }


def _label_rows(
    rows: pd.DataFrame,
    metadata: dict[str, Any],
    value_columns: tuple[str, ...],
) -> pd.DataFrame:
    output = rows.loc[:, list(value_columns)].copy()
    for name, value in reversed(tuple(metadata.items())):
        output.insert(0, name, value)
    return output


def _collect_tables(result: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    repository_root = Path(result.repository_root).resolve(strict=False)
    experiment_path = Path(result.experiment_path).resolve(strict=False)
    experiment = _portable(experiment_path, repository_root)
    eos_frames: list[pd.DataFrame] = []
    stellar_frames: list[pd.DataFrame] = []
    baseline_eos: pd.DataFrame | None = None
    baseline_stellar: pd.DataFrame | None = None
    next_label = 1

    for geometry_index, child in enumerate(result.child_results, start=1):
        geometry_id = Path(child.packet_path).name or f"geometry_{geometry_index:03d}"
        ledger = child.table("case_ledger.csv")
        profiles = child.table("thermodynamic_profiles.csv")
        sequences = child.table("stellar_sequences.csv")
        _require_columns(
            ledger,
            {
                "case_id",
                "amplitude",
                "epsilon0_mev_fm3",
                "sigma_mev_fm3",
                "delta_mev_fm3",
                "status",
            },
            table="case_ledger.csv",
        )
        _require_columns(
            profiles,
            {"case_id", "epsilon_mev_fm3", "pressure_mev_fm3", "cs2"},
            table="thermodynamic_profiles.csv",
        )
        _require_columns(
            sequences,
            {
                "case_id",
                "stage",
                "attempted_index",
                "calculation_status",
                "failure_category",
                "failure_reason",
                "central_pressure_mev_fm3",
                "Mass",
                "Radius",
                "Lambda",
                "k2",
                "tidal_status",
                "tidal_failure_reason",
                "is_sampled_peak",
            },
            table="stellar_sequences.csv",
        )
        if ledger["case_id"].astype(str).duplicated().any():
            raise ValueError("case_ledger.csv contains duplicate case IDs")
        final_stage = str(child.config.tov_stages[-1].name)
        sequences = sequences.loc[
            sequences["stage"].astype(str).eq(final_stage)
        ].copy()

        direct_profile = profiles.loc[
            profiles["case_id"].astype(str).eq("direct")
        ]
        direct_sequence = sequences.loc[
            sequences["case_id"].astype(str).eq("direct")
        ]
        if direct_profile.empty != direct_sequence.empty:
            raise ValueError("a CFL packet has incomplete direct baseline data")
        if not direct_profile.empty:
            if baseline_eos is not None or baseline_stellar is not None:
                raise ValueError(
                    "completed CFL experiment contains duplicate direct baseline packets"
                )
            baseline_metadata = {
                "label": "cfl_0",
                "experiment": experiment,
                "geometry_id": "baseline",
                "case_id": "direct",
                "status": "accepted",
                "amplitude": 0.0,
                "epsilon0_mev_fm3": np.nan,
                "sigma_mev_fm3": np.nan,
                "delta_mev_fm3": np.nan,
            }
            baseline_eos = _label_rows(
                direct_profile.sort_values(
                    "epsilon_mev_fm3", kind="stable"
                ),
                baseline_metadata,
                (
                    "epsilon_mev_fm3",
                    "pressure_mev_fm3",
                    "cs2",
                ),
            )
            baseline_stellar = _label_rows(
                direct_sequence.sort_values(
                    "attempted_index", kind="stable"
                ),
                baseline_metadata,
                (
                    "stage",
                    "attempted_index",
                    "calculation_status",
                    "failure_category",
                    "failure_reason",
                    "central_pressure_mev_fm3",
                    "Mass",
                    "Radius",
                    "tidal_status",
                    "tidal_failure_reason",
                    "k2",
                    "Lambda",
                    "is_sampled_peak",
                ),
            )

        accepted = ledger.loc[
            ledger["status"].astype(str).eq("accepted")
            & pd.to_numeric(ledger["amplitude"], errors="raise").ne(0.0)
        ]
        for _, row in accepted.iterrows():
            case_id = str(row["case_id"])
            case_profile = profiles.loc[
                profiles["case_id"].astype(str).eq(case_id)
            ]
            case_sequence = sequences.loc[
                sequences["case_id"].astype(str).eq(case_id)
            ]
            if case_profile.empty or case_sequence.empty:
                raise ValueError(
                    f"accepted CFL case {case_id!r} has incomplete saved data"
                )
            metadata = _case_metadata(
                row,
                label=f"cfl_{next_label}",
                geometry_id=geometry_id,
                experiment=experiment,
            )
            next_label += 1
            eos_frames.append(
                _label_rows(
                    case_profile.sort_values(
                        "epsilon_mev_fm3", kind="stable"
                    ),
                    metadata,
                    (
                        "epsilon_mev_fm3",
                        "pressure_mev_fm3",
                        "cs2",
                    ),
                )
            )
            stellar_frames.append(
                _label_rows(
                    case_sequence.sort_values(
                        "attempted_index", kind="stable"
                    ),
                    metadata,
                    (
                        "stage",
                        "attempted_index",
                        "calculation_status",
                        "failure_category",
                        "failure_reason",
                        "central_pressure_mev_fm3",
                        "Mass",
                        "Radius",
                        "tidal_status",
                        "tidal_failure_reason",
                        "k2",
                        "Lambda",
                        "is_sampled_peak",
                    ),
                )
            )

    if baseline_eos is None or baseline_stellar is None:
        raise ValueError("completed CFL experiment contains no direct baseline data")
    return (
        pd.concat([baseline_eos, *eos_frames], ignore_index=True),
        pd.concat([baseline_stellar, *stellar_frames], ignore_index=True),
    )


def _curve_colors(eos_data: pd.DataFrame) -> tuple[Any, Any, Any]:
    import matplotlib

    amplitudes = pd.to_numeric(eos_data["amplitude"], errors="raise")
    scale = max(float(amplitudes.abs().max()), 0.01)
    return (
        matplotlib.colors.Normalize(-scale, scale),
        matplotlib.colormaps["coolwarm"],
        amplitudes,
    )


def _write_figures(
    eos_data: pd.DataFrame,
    stellar_data: pd.DataFrame,
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm, cmap, _amplitudes = _curve_colors(eos_data)
    labels = list(dict.fromkeys(eos_data["label"].astype(str)))
    metadata = (
        eos_data.loc[:, ["label", "amplitude"]]
        .drop_duplicates("label", keep="first")
        .set_index("label")
    )
    figures = (
        (
            "pressure_energy_density.png",
            "thermodynamic",
            "epsilon_mev_fm3",
            "pressure_mev_fm3",
            r"Energy density [MeV fm$^{-3}$]",
            r"Pressure [MeV fm$^{-3}$]",
        ),
        (
            "speed_of_sound.png",
            "thermodynamic",
            "epsilon_mev_fm3",
            "cs2",
            r"Energy density [MeV fm$^{-3}$]",
            r"$c_s^2/c^2$",
        ),
        (
            "mass_radius.png",
            "stellar",
            "Radius",
            "Mass",
            "Radius [km]",
            r"Mass [$M_\odot$]",
        ),
        (
            "lambda_mass.png",
            "stellar",
            "Mass",
            "Lambda",
            r"Mass [$M_\odot$]",
            r"$\Lambda$",
        ),
        (
            "k2_mass.png",
            "stellar",
            "Mass",
            "k2",
            r"Mass [$M_\odot$]",
            r"$k_2$",
        ),
    )
    for filename, source, x_name, y_name, x_label, y_label in figures:
        fig, axis = plt.subplots(figsize=(8.4, 6.1), layout="constrained")
        plotted = 0
        try:
            for label in labels:
                amplitude = float(metadata.loc[label, "amplitude"])
                if source == "thermodynamic":
                    rows = eos_data.loc[eos_data["label"].astype(str).eq(label)]
                    x = pd.to_numeric(rows[x_name], errors="coerce").to_numpy(
                        dtype=float
                    )
                    y = pd.to_numeric(rows[y_name], errors="coerce").to_numpy(
                        dtype=float
                    )
                else:
                    rows = stellar_data.loc[
                        stellar_data["label"].astype(str).eq(label)
                    ]
                    x, y = _sequence_xy(rows, x_name, y_name)
                if not np.any(np.isfinite(x) & np.isfinite(y)):
                    continue
                axis.plot(
                    x,
                    y,
                    color="black" if amplitude == 0.0 else cmap(norm(amplitude)),
                    linewidth=1.5 if amplitude == 0.0 else 1.0,
                    alpha=0.88,
                    label=label if len(labels) <= 12 else None,
                )
                plotted += 1
            axis.set(xlabel=x_label, ylabel=y_label)
            axis.grid(True, alpha=0.2)
            if filename == "speed_of_sound.png":
                axis.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
            if filename == "lambda_mass.png":
                axis.set_yscale("log")
            if plotted and len(labels) <= 12:
                axis.legend(fontsize=8, ncol=2)
            elif plotted:
                fig.colorbar(
                    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=axis,
                    label="Amplitude A",
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No available valid curves",
                    transform=axis.transAxes,
                    ha="center",
                )
            fig.savefig(destination / filename, dpi=160)
        finally:
            plt.close(fig)


def _existing_summary(target: Path, experiment: str) -> dict[str, Any]:
    items = tuple(target.iterdir())
    observed = {item.name for item in items}
    expected = set(OUTPUT_FILES)
    if (
        observed != expected
        or any(item.is_symlink() for item in items)
        or any(not (target / name).is_file() for name in expected)
    ):
        raise FileExistsError(
            "existing CFL_DATASET is not an exact seven-file regular-file "
            f"output: {target}"
        )
    for name in FIGURE_FILES:
        with (target / name).open("rb") as stream:
            if stream.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
                raise FileExistsError(
                    f"existing CFL_DATASET has an invalid PNG file: {name}"
                )
    eos_data = pd.read_csv(target / EOS_DATA_FILE)
    stellar_data = pd.read_csv(target / STELLAR_DATA_FILE)
    for frame in (eos_data, stellar_data):
        if "experiment" not in frame or set(frame["experiment"].astype(str)) != {
            experiment
        }:
            raise FileExistsError(
                "existing CFL_DATASET belongs to a different experiment"
            )
    return {
        "path": target,
        "files": tuple(target / name for name in OUTPUT_FILES),
        "eos_count": int(eos_data["label"].nunique()),
        "thermodynamic_rows": int(len(eos_data)),
        "stellar_rows": int(len(stellar_data)),
        "reused": True,
        "solver_calls": 0,
    }


def build_cfl_dataset_output(
    result: Any,
    destination: str | Path,
    *,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """Publish exactly two labelled tables and five plots from saved data."""

    if not isinstance(reuse_existing, bool):
        raise TypeError("reuse_existing must be boolean")
    settings = getattr(result, "settings", None)
    if getattr(settings, "matter_model", None) != "cfl":
        raise ValueError("minimal CFL dataset output requires a CFL experiment")
    if getattr(settings, "calculation", None) != "stellar":
        raise ValueError("minimal CFL dataset output requires stellar data")
    if not bool(getattr(result, "completed", False)):
        raise ValueError("minimal CFL dataset output requires a completed result")

    repository_root = Path(result.repository_root).resolve(strict=False)
    runs_root = (repository_root / "runs").resolve(strict=False)
    experiment_path = Path(result.experiment_path).resolve(strict=False)
    target = Path(destination).expanduser().resolve(strict=False)
    if target == runs_root or not _is_below(target, runs_root):
        raise ValueError("CFL dataset output must remain below runs/")
    if (
        target == experiment_path
        or _is_below(target, experiment_path)
        or _is_below(experiment_path, target)
    ):
        raise ValueError("CFL dataset output must be separate from the experiment")
    if target.is_symlink():
        raise ValueError("CFL dataset output may not be a symbolic link")
    experiment = _portable(experiment_path, repository_root)
    if target.exists():
        if reuse_existing and target.is_dir():
            return _existing_summary(target, experiment)
        raise FileExistsError(target)

    eos_data, stellar_data = _collect_tables(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=".cfl_dataset_", dir=target.parent)
    ).resolve(strict=True)
    try:
        eos_data.to_csv(stage / EOS_DATA_FILE, index=False)
        stellar_data.to_csv(stage / STELLAR_DATA_FILE, index=False)
        _write_figures(eos_data, stellar_data, stage)
        observed = {item.name for item in stage.iterdir()}
        if observed != set(OUTPUT_FILES):
            raise RuntimeError("minimal CFL dataset staging inventory drifted")
        stage.replace(target)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        "path": target,
        "files": tuple(target / name for name in OUTPUT_FILES),
        "eos_count": int(eos_data["label"].nunique()),
        "thermodynamic_rows": int(len(eos_data)),
        "stellar_rows": int(len(stellar_data)),
        "reused": False,
        "solver_calls": 0,
    }


__all__ = [
    "EOS_DATA_FILE",
    "FIGURE_FILES",
    "OUTPUT_FILES",
    "STELLAR_DATA_FILE",
    "build_cfl_dataset_output",
]

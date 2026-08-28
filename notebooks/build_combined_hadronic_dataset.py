"""Combine accepted hadronic dataset runs without calling scientific solvers.

This presentation/data adapter consumes sealed saved tables only.  It assigns a
compact export name to each unique physical EoS, writes a six-column stellar
ML table, and renders five flat plots containing every selected physical EoS.
Authoritative packets and existing presentation folders are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eos_catalogue as catalogue


SCHEMA_ID = "bsk24_combined_hadronic_ml_dataset_v1"
VALID_TIDAL_STATUS = "validated_lambda_validation_v1"
CLEAN_COLUMNS = ("name", "M", "R", "Lambda", "k2", "P_c")
MAPPING_COLUMNS = (
    "name",
    "source_eos_id",
    "regime",
    "amplitude",
    "epsilon_match_mev_fm3",
    "center_mev_fm3",
    "width_mev_fm3",
    "ramp_width_mev_fm3",
    "source_run",
)
PLOT_FILES = (
    "eos_pressure.png",
    "speed_of_sound.png",
    "mass_radius.png",
    "k2_mass.png",
    "lambda_mass.png",
)
MAX_THERMODYNAMIC_PLOT_POINTS = 5000


def _portable(path: Path, root: Path) -> str:
    return path.resolve(strict=False).relative_to(root).as_posix()


def _finite(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)


def _true(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().eq("true")


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_below(path: Path, parent: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(parent.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"path escapes its allowed parent: {path}") from exc
    return resolved


def _source_catalogue(root: Path, run_root: Path, rank: int) -> pd.DataFrame:
    runs = (root / "runs").resolve(strict=True)
    run = _resolve_below(run_root, runs)
    eos_data = _resolve_below(run / "EOS_DATA", run)
    checksums = catalogue.manifest(eos_data)
    table = pd.read_csv(
        catalogue.verified(eos_data, "eos_catalogue.csv", checksums),
        low_memory=False,
    )
    required = {
        "eos_id",
        "catalogue_id",
        "physical_model_key",
        "packet_path",
        "geometry_id",
        "source_case_id",
        "amplitude",
        "status",
        "epsilon_match_mev_fm3",
        "epsilon0_mev_fm3",
        "sigma_mev_fm3",
        "delta_mev_fm3",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"source EoS catalogue is missing columns: {sorted(missing)}")
    if table.empty or not table["status"].astype(str).isin({"accepted", "baseline"}).all():
        raise ValueError("source EoS catalogue contains a non-accepted physical entry")
    table = table.copy()
    table["_source_rank"] = rank
    table["_source_run"] = run.name
    table["_run_root"] = str(run)
    table["_eos_data"] = str(eos_data)
    table["_manifest_sha256"] = catalogue.sha256(eos_data / "SHA256SUMS.txt")
    return table


def build_name_mapping(
    root: Path, source_runs: list[Path]
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    if len(source_runs) < 2:
        raise ValueError("at least two source runs are required")
    frames = [
        _source_catalogue(root, source, rank)
        for rank, source in enumerate(source_runs)
    ]
    catalogue_ids = set(pd.concat(frames)["catalogue_id"].astype(str))
    if len(catalogue_ids) != 1:
        raise ValueError("source runs do not share one persistent EoS catalogue")

    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    occurrence_count = 0
    for frame in frames:
        for row in frame.to_dict(orient="records"):
            key = (str(row["catalogue_id"]), str(row["physical_model_key"]))
            occurrence_count += 1
            previous = chosen.get(key)
            if previous is None:
                chosen[key] = row
                continue
            numeric = (
                "amplitude",
                "epsilon_match_mev_fm3",
                "epsilon0_mev_fm3",
                "sigma_mev_fm3",
                "delta_mev_fm3",
            )
            for column in numeric:
                left, right = previous.get(column), row.get(column)
                if pd.isna(left) and pd.isna(right):
                    continue
                if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0):
                    raise ValueError("duplicate physical identity has inconsistent parameters")

    records = list(chosen.values())
    baseline = [row for row in records if math.isclose(float(row["amplitude"]), 0.0)]
    if len(baseline) != 1:
        raise ValueError("combined sources must contain exactly one physical baseline")

    def order(row: dict[str, Any]) -> tuple[Any, ...]:
        amplitude = float(row["amplitude"])
        if math.isclose(amplitude, 0.0):
            regime, amplitude_order = 0, 0.0
        elif amplitude > 0.0:
            regime, amplitude_order = 1, amplitude
        else:
            regime, amplitude_order = 2, abs(amplitude)
        return (
            regime,
            amplitude_order,
            float(row["epsilon0_mev_fm3"]),
            float(row["sigma_mev_fm3"]),
            float(row["delta_mev_fm3"]),
            int(row["_source_rank"]),
            str(row["physical_model_key"]),
        )

    records.sort(key=order)
    digits = max(4, len(str(len(records))))
    for index, row in enumerate(records, start=1):
        amplitude = float(row["amplitude"])
        row["name"] = f"H_{index:0{digits}d}"
        row["regime"] = (
            "baseline" if math.isclose(amplitude, 0.0)
            else "positive" if amplitude > 0.0
            else "negative"
        )
        packet = _resolve_below(root / str(row["packet_path"]), Path(row["_run_root"]))
        row["_packet"] = str(packet)

    sources = []
    for frame in frames:
        first = frame.iloc[0]
        sources.append(
            {
                "run": str(first["_source_run"]),
                "path": _portable(Path(first["_run_root"]), root),
                "eos_data_path": _portable(Path(first["_eos_data"]), root),
                "eos_data_manifest_sha256": str(first["_manifest_sha256"]),
                "catalogue_entry_count": int(len(frame)),
            }
        )
    mapping = pd.DataFrame.from_records(records)
    return mapping, sources, occurrence_count - len(mapping)


def _resolve_case_id(rows: pd.DataFrame, record: pd.Series) -> str:
    available = set(rows["case_id"].astype(str))
    candidates = []
    for column in ("source_case_id", "case_id", "physical_case_id"):
        value = record.get(column)
        if value is not None and not pd.isna(value) and str(value) not in candidates:
            candidates.append(str(value))
    selected = [value for value in candidates if value in available]
    if len(selected) != 1:
        raise ValueError(
            f"could not resolve exactly one saved case for {record['name']}: {selected}"
        )
    return selected[0]


def _sequence_prefix(rows: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    ordered = rows.sort_values("attempted_index").copy()
    peaks = _true(ordered["is_sampled_peak"])
    if not peaks.any():
        return ordered.iloc[0:0].copy(), False
    peak = pd.to_numeric(
        ordered.loc[peaks, "attempted_index"], errors="raise"
    ).min()
    attempted = pd.to_numeric(ordered["attempted_index"], errors="raise")
    return ordered.loc[attempted.le(peak)].copy(), True


def collect_stellar(
    root: Path, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, str], list[str]]:
    required = {
        "case_id",
        "stage",
        "attempted_index",
        "calculation_status",
        "Mass",
        "Radius",
        "Lambda",
        "k2",
        "P_Central",
        "is_sampled_peak",
        "tidal_status",
    }
    clean_parts: list[pd.DataFrame] = []
    curves: list[dict[str, Any]] = []
    resolved_cases: dict[str, str] = {}
    without_peak: list[str] = []
    packet_groups = list(mapping.groupby("_packet", sort=False))
    for packet_index, (packet_text, selected) in enumerate(packet_groups, start=1):
        packet = _resolve_below(Path(packet_text), root / "runs")
        checksums = catalogue.manifest(packet)
        config = catalogue.read_json(
            catalogue.verified(packet, "complete_configuration.json", checksums)
        )
        stages = config.get("tov_stages")
        if not isinstance(stages, list) or not stages or not isinstance(stages[-1], dict):
            raise ValueError(f"packet has no final TOV stage: {packet}")
        final_stage = str(stages[-1].get("name", ""))
        sequence = pd.read_csv(
            catalogue.verified(packet, "stellar_sequences.csv", checksums),
            low_memory=False,
        )
        if not required.issubset(sequence.columns):
            raise ValueError(f"stellar sequence schema is incomplete: {packet}")
        sequence = sequence.loc[sequence["stage"].astype(str).eq(final_stage)].copy()
        if sequence.empty:
            raise ValueError(f"packet has no final-stage stellar rows: {packet}")
        for _, record in selected.iterrows():
            case_id = _resolve_case_id(sequence, record)
            resolved_cases[str(record["name"])] = case_id
            rows, has_peak = _sequence_prefix(
                sequence.loc[sequence["case_id"].astype(str).eq(case_id)]
            )
            if not has_peak:
                without_peak.append(str(record["name"]))
                continue
            success = rows["calculation_status"].astype(str).eq("success")
            mr_valid = success.to_numpy() & _finite(rows, ("Mass", "Radius"))
            tidal = rows["tidal_status"].astype(str).eq(VALID_TIDAL_STATUS)
            tidal_valid = (
                success.to_numpy()
                & tidal.to_numpy()
                & _finite(rows, ("Mass", "Lambda", "k2"))
            )
            clean_valid = (
                success.to_numpy()
                & tidal.to_numpy()
                & _finite(rows, ("Mass", "Radius", "Lambda", "k2", "P_Central"))
            )
            clean = rows.loc[
                clean_valid, ["Mass", "Radius", "Lambda", "k2", "P_Central"]
            ].copy()
            clean.insert(0, "name", str(record["name"]))
            clean.rename(
                columns={"Mass": "M", "Radius": "R", "P_Central": "P_c"},
                inplace=True,
            )
            clean_parts.append(clean.loc[:, list(CLEAN_COLUMNS)])
            curves.append(
                {
                    "name": str(record["name"]),
                    "amplitude": float(record["amplitude"]),
                    "rows": rows,
                    "mr_valid": mr_valid,
                    "tidal_valid": tidal_valid,
                }
            )
        if packet_index % 10 == 0 or packet_index == len(packet_groups):
            print(
                f"stellar packets: {packet_index}/{len(packet_groups)} verified and read",
                flush=True,
            )
    if not clean_parts:
        raise RuntimeError("no valid combined stellar rows were found")
    clean = pd.concat(clean_parts, ignore_index=True)
    return clean, curves, resolved_cases, without_peak


def _plot_color(amplitude: float, norm: matplotlib.colors.Normalize):
    if math.isclose(amplitude, 0.0):
        return "#111827"
    return matplotlib.colormaps["coolwarm"](norm(amplitude))


def _normalization(mapping: pd.DataFrame) -> matplotlib.colors.Normalize:
    amplitudes = pd.to_numeric(mapping["amplitude"], errors="raise")
    lower, upper = float(amplitudes.min()), float(amplitudes.max())
    if not lower < 0.0 < upper:
        raise ValueError("combined plot requires both positive and negative amplitudes")
    return matplotlib.colors.TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)


def _finish_plot(
    fig,
    ax,
    output: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    norm: matplotlib.colors.Normalize,
    log_y: bool = False,
) -> None:
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, color="#d1d5db", linewidth=0.55, alpha=0.55)
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax, label="Signed deformation amplitude A", shrink=0.84)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_stellar_plots(
    curves: list[dict[str, Any]], plots: Path, norm: matplotlib.colors.Normalize
) -> list[dict[str, Any]]:
    definitions = (
        ("mass_radius.png", "Radius", "Mass", "Radius R [km]", r"Mass $M/M_\odot$", False),
        ("k2_mass.png", "Mass", "k2", r"Mass $M/M_\odot$", r"Tidal Love number $k_2$", False),
        ("lambda_mass.png", "Mass", "Lambda", r"Mass $M/M_\odot$", r"Dimensionless tidal deformability $\Lambda$", True),
    )
    inventory = []
    for filename, x, y, xlabel, ylabel, log_y in definitions:
        fig, ax = plt.subplots(figsize=(9.6, 6.8))
        plotted = 0
        for curve in curves:
            rows = curve["rows"]
            mask = curve["mr_valid"] if filename == "mass_radius.png" else curve["tidal_valid"]
            xs = pd.to_numeric(rows[x], errors="coerce").to_numpy(dtype=float)
            ys = pd.to_numeric(rows[y], errors="coerce").to_numpy(dtype=float)
            xs[~mask], ys[~mask] = np.nan, np.nan
            if not np.isfinite(xs).any() or not np.isfinite(ys).any():
                continue
            amplitude = float(curve["amplitude"])
            ax.plot(
                xs,
                ys,
                color=_plot_color(amplitude, norm),
                linewidth=1.5 if math.isclose(amplitude, 0.0) else 0.7,
                alpha=0.95 if math.isclose(amplitude, 0.0) else 0.28,
                zorder=5 if math.isclose(amplitude, 0.0) else 2,
            )
            plotted += 1
        _finish_plot(
            fig,
            ax,
            plots / filename,
            xlabel=xlabel,
            ylabel=ylabel,
            title="Accepted combined hadronic EoSs — saved data",
            norm=norm,
            log_y=log_y,
        )
        inventory.append({"figure": filename, "plotted_eos_count": plotted})
    return inventory


def _decimate(rows: pd.DataFrame) -> pd.DataFrame:
    if len(rows) <= MAX_THERMODYNAMIC_PLOT_POINTS:
        return rows
    positions = np.unique(
        np.linspace(0, len(rows) - 1, MAX_THERMODYNAMIC_PLOT_POINTS, dtype=int)
    )
    return rows.iloc[positions]


def collect_thermodynamic_curves(
    root: Path, mapping: pd.DataFrame, resolved_cases: dict[str, str]
) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    packet_groups = list(mapping.groupby("_packet", sort=False))
    required = {"case_id", "epsilon_mev_fm3", "pressure_mev_fm3", "cs2"}
    for packet_index, (packet_text, selected) in enumerate(packet_groups, start=1):
        packet = _resolve_below(Path(packet_text), root / "runs")
        checksums = catalogue.manifest(packet)
        profile = pd.read_csv(
            catalogue.verified(packet, "thermodynamic_profiles.csv", checksums),
            usecols=lambda column: column in required,
            low_memory=False,
        )
        if not required.issubset(profile.columns):
            raise ValueError(f"thermodynamic profile schema is incomplete: {packet}")
        for _, record in selected.iterrows():
            name = str(record["name"])
            case_id = resolved_cases[name]
            rows = profile.loc[profile["case_id"].astype(str).eq(case_id)].copy()
            if rows.empty:
                raise ValueError(f"missing thermodynamic profile for {name} in {packet}")
            rows.sort_values("epsilon_mev_fm3", inplace=True)
            finite = _finite(rows, ("epsilon_mev_fm3", "pressure_mev_fm3", "cs2"))
            rows = _decimate(rows.loc[finite])
            if rows.empty:
                raise ValueError(f"thermodynamic profile has no finite plot rows for {name}")
            curves.append(
                {
                    "name": name,
                    "amplitude": float(record["amplitude"]),
                    "rows": rows,
                }
            )
        if packet_index % 10 == 0 or packet_index == len(packet_groups):
            print(
                f"thermodynamic packets: {packet_index}/{len(packet_groups)} verified and read",
                flush=True,
            )
    return curves


def draw_thermodynamic_plots(
    curves: list[dict[str, Any]], plots: Path, norm: matplotlib.colors.Normalize
) -> list[dict[str, Any]]:
    definitions = (
        (
            "eos_pressure.png",
            "pressure_mev_fm3",
            r"Pressure $P$ [MeV fm$^{-3}$]",
        ),
        (
            "speed_of_sound.png",
            "cs2",
            r"Dimensionless sound speed squared $c_s^2$ ($c=1$)",
        ),
    )
    inventory = []
    for filename, y, ylabel in definitions:
        fig, ax = plt.subplots(figsize=(9.6, 6.8))
        plotted = 0
        for curve in curves:
            rows = curve["rows"]
            amplitude = float(curve["amplitude"])
            ax.plot(
                rows["epsilon_mev_fm3"],
                rows[y],
                color=_plot_color(amplitude, norm),
                linewidth=1.5 if math.isclose(amplitude, 0.0) else 0.62,
                alpha=0.95 if math.isclose(amplitude, 0.0) else 0.24,
                zorder=5 if math.isclose(amplitude, 0.0) else 2,
            )
            plotted += 1
        if filename == "speed_of_sound.png":
            ax.axhline(1.0, color="#4b5563", linestyle="--", linewidth=0.9)
            ax.axhline(1.0 / 3.0, color="#6b7280", linestyle=":", linewidth=0.8)
        _finish_plot(
            fig,
            ax,
            plots / filename,
            xlabel=r"Energy density $\varepsilon$ [MeV fm$^{-3}$]",
            ylabel=ylabel,
            title="Accepted combined hadronic EoSs — saved data",
            norm=norm,
        )
        inventory.append({"figure": filename, "plotted_eos_count": plotted})
    return inventory


def _write_manifest(folder: Path) -> None:
    files = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    text = "".join(
        f"{_checksum(path)}  {path.relative_to(folder).as_posix()}\n" for path in files
    )
    temporary = folder / ".SHA256SUMS.tmp"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, folder / "SHA256SUMS.txt")


def finalize_manifest(destination: str | Path) -> dict[str, Any]:
    target = Path(destination).resolve(strict=True)
    provenance = catalogue.read_json(target / "provenance.json")
    if provenance.get("schema_id") != SCHEMA_ID:
        raise ValueError("destination is not a combined hadronic dataset")
    _write_manifest(target)
    return {
        "destination": str(target),
        "manifest_sha256": catalogue.sha256(target / "SHA256SUMS.txt"),
    }


def build_combined_dataset(
    repository_root: str | Path,
    source_runs: list[str | Path],
    destination: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    runs = (root / "runs").resolve(strict=True)
    target = Path(destination).resolve(strict=False)
    try:
        target.relative_to(runs)
    except ValueError as exc:
        raise ValueError("destination must remain below runs/") from exc
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    source_paths = [_resolve_below(Path(path), runs) for path in source_runs]
    if any(target == path or path in target.parents or target in path.parents for path in source_paths):
        raise ValueError("destination overlaps a source run")

    mapping, sources, duplicate_count = build_name_mapping(root, source_paths)
    regimes = mapping["regime"].value_counts().to_dict()
    print(
        f"physical EoSs: {len(mapping)} unique; {duplicate_count} duplicate occurrence(s); "
        f"regimes={regimes}",
        flush=True,
    )
    clean, stellar_curves, resolved_cases, without_peak = collect_stellar(root, mapping)
    valid_names = set(clean["name"].astype(str))
    if valid_names != set(mapping["name"].astype(str)):
        missing = sorted(set(mapping["name"].astype(str)) - valid_names)
        raise RuntimeError(f"accepted EoSs without complete clean stellar rows: {missing[:12]}")

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".combined_hadronic_", dir=target.parent))
    ml_data = stage / "ML_DATA"
    plots = stage / "plots"
    ml_data.mkdir()
    plots.mkdir()
    try:
        clean.to_csv(
            ml_data / "hadronic_stellar_data.csv",
            index=False,
            columns=list(CLEAN_COLUMNS),
            float_format="%.17g",
            lineterminator="\n",
        )
        public_mapping = pd.DataFrame(
            {
                "name": mapping["name"],
                "source_eos_id": mapping["eos_id"],
                "regime": mapping["regime"],
                "amplitude": mapping["amplitude"],
                "epsilon_match_mev_fm3": mapping["epsilon_match_mev_fm3"],
                "center_mev_fm3": mapping["epsilon0_mev_fm3"],
                "width_mev_fm3": mapping["sigma_mev_fm3"],
                "ramp_width_mev_fm3": mapping["delta_mev_fm3"],
                "source_run": mapping["_source_run"],
            }
        )
        public_mapping.to_csv(
            ml_data / "eos_name_mapping.csv",
            index=False,
            columns=list(MAPPING_COLUMNS),
            float_format="%.17g",
            lineterminator="\n",
        )
        norm = _normalization(mapping)
        inventory = draw_stellar_plots(stellar_curves, plots, norm)
        thermo_curves = collect_thermodynamic_curves(root, mapping, resolved_cases)
        if len(thermo_curves) != len(mapping):
            raise RuntimeError("thermodynamic curve count does not match physical EoS count")
        inventory.extend(draw_thermodynamic_plots(thermo_curves, plots, norm))
        inventory.sort(key=lambda row: PLOT_FILES.index(row["figure"]))
        pd.DataFrame(inventory).to_csv(plots / "plot_inventory.csv", index=False)

        provenance = {
            "schema_id": SCHEMA_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "destination": _portable(target, root),
            "source_runs": sources,
            "source_catalogue_occurrence_count": int(len(mapping) + duplicate_count),
            "duplicate_physical_occurrence_count": int(duplicate_count),
            "unique_eos_count": int(len(mapping)),
            "regime_counts": {str(key): int(value) for key, value in regimes.items()},
            "clean_stellar_row_count": int(len(clean)),
            "clean_stellar_eos_count": int(clean["name"].nunique()),
            "clean_stellar_columns": list(CLEAN_COLUMNS),
            "generated_plot_count": len(PLOT_FILES),
            "plot_inventory": inventory,
            "solver_calls": 0,
            "authoritative_packets_modified": False,
            "case_policy": "accepted/baseline physical EoSs only; rejected proposals excluded",
            "deduplication_policy": "catalogue_id plus physical_model_key; first explicit source owns duplicates",
            "naming_policy": "H_0001 baseline; positive amplitudes next; negative amplitudes last; parameters sorted within regimes",
            "stellar_row_policy": (
                "final saved stage through first sampled peak; calculation_status=success; "
                f"tidal_status={VALID_TIDAL_STATUS}; all six exported fields finite"
            ),
            "plot_branch_policy": "final saved stage through first sampled peak; failed gaps are not bridged",
            "thermodynamic_plot_policy": (
                f"all physical EoSs included; each retained profile is deterministically thinned to at most "
                f"{MAX_THERMODYNAMIC_PLOT_POINTS} points for raster presentation only"
            ),
            "eos_without_sampled_peak": without_peak,
            "builder_sha256": catalogue.sha256(Path(__file__)),
        }
        (stage / "provenance.json").write_bytes(catalogue.canonical(provenance) + b"\n")
        (ml_data / "README.md").write_text(
            "# Combined hadronic stellar ML data\n\n"
            "`hadronic_stellar_data.csv` is the machine-facing table. It has exactly six "
            "columns: `name`, `M`, `R`, `Lambda`, `k2`, and `P_c`. Units are solar masses "
            "for M, km for R, dimensionless for Lambda and k2, and MeV fm^-3 for P_c. "
            "Rows retain saved pressure order through the first sampled peak. No ten-point "
            "subsampling has been applied.\n\n"
            "`eos_name_mapping.csv` is a separate audit lookup. H_0001 is the undeformed "
            "BSk24 baseline. The main ML table deliberately contains no hashes or paths.\n\n"
            "These are derived saved-table outputs. Original packets and EOS_DATA folders "
            "remain authoritative and unchanged.\n",
            encoding="utf-8",
            newline="\n",
        )
        (plots / "README.md").write_text(
            "# Combined accepted hadronic plots\n\n"
            "Every PNG combines all accepted unique physical EoSs from both explicit source "
            "runs. The shared baseline is drawn once, rejected proposals are excluded, and "
            "no solver is called. Stellar figures retain all saved sequence points selected "
            "by the branch/status policy. Thermodynamic curves are deterministically thinned "
            "only for raster rendering; the original saved tables are untouched.\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_manifest(stage)
        catalogue.publish_directory(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        **provenance,
        "destination_path": str(target),
        "ml_data_path": str(target / "ML_DATA"),
        "plots_path": str(target / "plots"),
        "manifest_sha256": catalogue.sha256(target / "SHA256SUMS.txt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--source-run", type=Path, action="append")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--finalize-manifest", action="store_true")
    args = parser.parse_args()
    if args.finalize_manifest:
        print(json.dumps(finalize_manifest(args.destination), sort_keys=True))
        return 0
    if args.repository_root is None or not args.source_run:
        parser.error("--repository-root and at least two --source-run values are required")
    result = build_combined_dataset(
        args.repository_root, args.source_run, args.destination
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

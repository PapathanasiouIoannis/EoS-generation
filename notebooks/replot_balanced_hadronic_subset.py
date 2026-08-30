from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import materialize_balanced_hadronic_subset as materializer
import select_combined_hadronic_subset as selection


PLOT_FILES = (
    "eos_pressure.png",
    "speed_of_sound.png",
    "mass_radius.png",
    "k2_mass.png",
    "lambda_mass.png",
)
PLOT_COLOR = "#0057B8"
STELLAR_LINEWIDTH = 1.70
THERMO_LINEWIDTH = 1.60
BASELINE_LINEWIDTH = 3.60
STELLAR_ALPHA = 0.38
THERMO_ALPHA = 0.32


def _finish(
    fig: plt.Figure,
    ax: plt.Axes,
    path: Path,
    xlabel: str,
    ylabel: str,
    log_y: bool = False,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Balanced 2,000 accepted hadronic EoSs — double-thick blue replot")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _draw_stellar(
    curves: dict[str, pd.DataFrame],
    selected_names: list[str],
    baseline: str,
    plots: Path,
) -> list[dict[str, Any]]:
    definitions = (
        ("mass_radius.png", "R", "M", "Radius R [km]", r"Mass M [$M_\odot$]", False),
        ("k2_mass.png", "M", "k2", r"Mass M [$M_\odot$]", r"Love number $k_2$", False),
        ("lambda_mass.png", "M", "Lambda", r"Mass M [$M_\odot$]", r"Tidal deformability $\Lambda$", True),
    )
    inventory: list[dict[str, Any]] = []
    for filename, x, y, xlabel, ylabel, log_y in definitions:
        fig, ax = plt.subplots(figsize=(10.0, 7.0))
        for name in selected_names:
            rows = curves[name]
            is_baseline = name == baseline
            ax.plot(
                rows[x],
                rows[y],
                color=PLOT_COLOR,
                linewidth=BASELINE_LINEWIDTH if is_baseline else STELLAR_LINEWIDTH,
                alpha=1.0 if is_baseline else STELLAR_ALPHA,
                zorder=5 if is_baseline else 2,
            )
        _finish(fig, ax, plots / filename, xlabel, ylabel, log_y=log_y)
        inventory.append({"figure": filename, "plotted_eos_count": len(selected_names)})
    return inventory


def _draw_thermodynamic(
    curves: dict[str, pd.DataFrame],
    selected_names: list[str],
    baseline: str,
    plots: Path,
) -> list[dict[str, Any]]:
    definitions = (
        ("eos_pressure.png", "pressure_mev_fm3", r"Pressure P [MeV fm$^{-3}$]"),
        ("speed_of_sound.png", "cs2", r"Dimensionless sound speed squared $c_s^2$ ($c=1$)"),
    )
    inventory: list[dict[str, Any]] = []
    for filename, column, ylabel in definitions:
        fig, ax = plt.subplots(figsize=(10.0, 7.0))
        for name in selected_names:
            rows = curves[name]
            positions = np.unique(
                np.linspace(0, len(rows) - 1, min(len(rows), 600), dtype=int)
            )
            part = rows.iloc[positions]
            is_baseline = name == baseline
            ax.plot(
                part["epsilon_mev_fm3"],
                part[column],
                color=PLOT_COLOR,
                linewidth=BASELINE_LINEWIDTH if is_baseline else THERMO_LINEWIDTH,
                alpha=1.0 if is_baseline else THERMO_ALPHA,
                zorder=5 if is_baseline else 2,
            )
        if column == "cs2":
            ax.axhline(1.0, color="#4b5563", linestyle="--", linewidth=0.9)
            ax.axhline(1.0 / 3.0, color="#6b7280", linestyle=":", linewidth=0.8)
        _finish(
            fig,
            ax,
            plots / filename,
            r"Energy density $\varepsilon$ [MeV fm$^{-3}$]",
            ylabel,
        )
        inventory.append({"figure": filename, "plotted_eos_count": len(selected_names)})
    return inventory


def replot(
    root: Path,
    source: Path,
    destination: Path,
    parent_override: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    runs = (root / "runs").resolve(strict=True)
    catalogue_root = selection.catalogue.confined(runs / "eos_catalogue", runs)
    source = selection.catalogue.confined(source, runs)
    destination = selection.catalogue.confined(destination, runs)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination == runs:
        raise ValueError("replot destination must be below runs/")
    selection.catalogue.require_disjoint(destination, source, catalogue_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    source_manifest = selection._read_manifest(source)
    source_provenance_path = selection._verified(
        source, "provenance.json", source_manifest
    )
    mapping_path = selection._verified(
        source, "ML_DATA/eos_name_mapping.csv", source_manifest
    )
    stellar_path = selection._verified(
        source, "ML_DATA/hadronic_stellar_data.csv", source_manifest
    )
    source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8"))
    if int(source_provenance.get("selected_eos_count", -1)) != 2000:
        raise ValueError("source is not the validated 2,000-EoS packet")
    if source_provenance.get("solver_calls") != 0:
        raise ValueError("unexpected source solver-call provenance")

    parent = selection.catalogue.confined(
        parent_override
        if parent_override is not None
        else root / str(source_provenance["parent_dataset"]),
        runs,
    )
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    selection.catalogue.require_disjoint(destination, parent)
    if selection._sha256(parent / "SHA256SUMS.txt") != str(
        source_provenance["parent_manifest_sha256"]
    ):
        raise ValueError("authoritative parent manifest changed")
    parent_manifest = selection._read_manifest(parent)
    parent_provenance_path = selection._verified(
        parent, "provenance.json", parent_manifest
    )
    parent_provenance = json.loads(parent_provenance_path.read_text(encoding="utf-8"))
    source_records = parent_provenance.get("source_runs")
    if not isinstance(source_records, list):
        raise ValueError("parent source-run provenance is malformed")
    source_experiments = selection._source_experiments(root, source_records)
    selection.catalogue.require_disjoint(
        destination, *source_experiments.values()
    )

    mapping = pd.read_csv(mapping_path, low_memory=False)
    stellar = pd.read_csv(stellar_path, low_memory=False)
    selected_names = mapping["name"].astype(str).tolist()
    if len(selected_names) != 2000 or len(set(selected_names)) != 2000:
        raise ValueError("source mapping is not exactly 2,000 unique EoSs")
    if set(stellar["name"].astype(str)) != set(selected_names):
        raise ValueError("stellar and mapping identities differ")
    baseline_names = mapping.loc[
        mapping["regime"].astype(str).eq("baseline"), "name"
    ].astype(str).tolist()
    if len(baseline_names) != 1:
        raise ValueError("source must contain exactly one baseline")

    combiner = selection._load_combiner(root)
    internal_mapping, resolved_cases = materializer._selected_internal_mapping(
        root, mapping, parent_provenance, combiner
    )
    thermo_list = combiner.collect_thermodynamic_curves(
        root, internal_mapping, resolved_cases
    )
    thermo = {
        str(curve["name"]): curve["rows"].reset_index(drop=True)
        for curve in thermo_list
    }
    stellar_curves = {
        str(name): rows.reset_index(drop=True)
        for name, rows in stellar.groupby("name", sort=False)
    }
    if set(thermo) != set(selected_names) or set(stellar_curves) != set(selected_names):
        raise ValueError("saved curve collection is incomplete")

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".balanced_2000_thick_", dir=destination.parent))
    try:
        inventory = _draw_stellar(
            stellar_curves, selected_names, baseline_names[0], stage
        )
        inventory.extend(
            _draw_thermodynamic(thermo, selected_names, baseline_names[0], stage)
        )
        inventory.sort(key=lambda row: PLOT_FILES.index(row["figure"]))
        pd.DataFrame(inventory).to_csv(stage / "plot_inventory.csv", index=False)
        pngs = sorted(path.name for path in stage.glob("*.png"))
        if tuple(pngs) != tuple(sorted(PLOT_FILES)):
            raise ValueError(f"replot inventory is not exactly five PNGs: {pngs}")

        provenance = {
            "schema_id": "bsk24_balanced_hadronic_blue_double_thick_replot_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": source.relative_to(root).as_posix(),
            "source_manifest_sha256": selection._sha256(source / "SHA256SUMS.txt"),
            "resolved_parent_dataset": parent.relative_to(root).as_posix(),
            "selected_eos_count": 2000,
            "generated_plot_count": len(PLOT_FILES),
            "plot_inventory": inventory,
            "plot_policy": "all EoS curves blue; exactly twice the preceding thick replot widths; baseline emphasized; saved data only",
            "curve_color_hex": PLOT_COLOR,
            "stellar_linewidth_points": STELLAR_LINEWIDTH,
            "thermodynamic_linewidth_points": THERMO_LINEWIDTH,
            "baseline_linewidth_points": BASELINE_LINEWIDTH,
            "stellar_alpha": STELLAR_ALPHA,
            "thermodynamic_alpha": THERMO_ALPHA,
            "solver_calls": 0,
            "scientific_data_changed": False,
            "source_dataset_modified": False,
            "replot_script_sha256": selection._sha256(Path(__file__)),
        }
        (stage / "provenance.json").write_bytes(selection._canonical(provenance) + b"\n")
        (stage / "README.md").write_text(
            "# Double-thick blue replot of the final balanced 2,000-EoS dataset\n\n"
            "These five PNGs re-render the checksum-verified final saved data with blue "
            "curves at exactly twice the preceding thick-replot widths. No scientific solver "
            "was called and no source data was changed.\n",
            encoding="utf-8",
            newline="\n",
        )
        selection._write_manifest(stage)
        selection.catalogue.publish_directory(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        **provenance,
        "destination_path": str(destination),
        "manifest_sha256": selection._sha256(destination / "SHA256SUMS.txt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--parent-override", type=Path)
    args = parser.parse_args()
    root = selection.catalogue.trusted_repository_root(args.repository_root)
    result = replot(
        root,
        args.source,
        args.destination,
        parent_override=args.parent_override,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

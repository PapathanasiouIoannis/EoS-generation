"""Materialize a validated balanced BSk24 subset without scientific solver calls."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import select_combined_hadronic_subset as selection


SCHEMA_ID = "bsk24_balanced_hadronic_ml_dataset_v1"
PLOT_FILES = (
    "eos_pressure.png",
    "speed_of_sound.png",
    "mass_radius.png",
    "k2_mass.png",
    "lambda_mass.png",
)
STELLAR_COLUMNS = ("name", "M", "R", "Lambda", "k2", "P_c")
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


def _finish(
    fig: plt.Figure,
    ax: plt.Axes,
    path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    log_y: bool = False,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _draw_stellar(
    curves: dict[str, pd.DataFrame], selected_names: list[str], baseline: str, plots: Path
) -> list[dict[str, Any]]:
    definitions = (
        ("mass_radius.png", "R", "M", "Radius R [km]", r"Mass M [$M_\odot$]", False),
        ("k2_mass.png", "M", "k2", r"Mass M [$M_\odot$]", r"Love number $k_2$", False),
        ("lambda_mass.png", "M", "Lambda", r"Mass M [$M_\odot$]", r"Tidal deformability $\Lambda$", True),
    )
    inventory: list[dict[str, Any]] = []
    for filename, x, y, xlabel, ylabel, log_y in definitions:
        fig, ax = plt.subplots(figsize=(10.0, 7.0))
        plotted = 0
        for name in selected_names:
            rows = curves[name]
            is_baseline = name == baseline
            ax.plot(
                rows[x],
                rows[y],
                color="#000000",
                linewidth=1.25 if is_baseline else 0.42,
                alpha=1.0 if is_baseline else 0.30,
                zorder=5 if is_baseline else 2,
            )
            plotted += 1
        _finish(
            fig,
            ax,
            plots / filename,
            xlabel,
            ylabel,
            "Balanced 2,000 accepted hadronic EoSs — saved data",
            log_y=log_y,
        )
        inventory.append({"figure": filename, "plotted_eos_count": plotted})
    return inventory


def _draw_thermodynamic(
    curves: dict[str, pd.DataFrame], selected_names: list[str], baseline: str, plots: Path
) -> list[dict[str, Any]]:
    definitions = (
        ("eos_pressure.png", "pressure_mev_fm3", r"Pressure P [MeV fm$^{-3}$]"),
        ("speed_of_sound.png", "cs2", r"Dimensionless sound speed squared $c_s^2$ ($c=1$)"),
    )
    inventory: list[dict[str, Any]] = []
    for filename, column, ylabel in definitions:
        fig, ax = plt.subplots(figsize=(10.0, 7.0))
        plotted = 0
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
                color="#000000",
                linewidth=1.25 if is_baseline else 0.40,
                alpha=1.0 if is_baseline else 0.22,
                zorder=5 if is_baseline else 2,
            )
            plotted += 1
        if column == "cs2":
            ax.axhline(1.0, color="#4b5563", linestyle="--", linewidth=0.9)
            ax.axhline(1.0 / 3.0, color="#6b7280", linestyle=":", linewidth=0.8)
        _finish(
            fig,
            ax,
            plots / filename,
            r"Energy density $\varepsilon$ [MeV fm$^{-3}$]",
            ylabel,
            "Balanced 2,000 accepted hadronic EoSs — saved data",
        )
        inventory.append({"figure": filename, "plotted_eos_count": plotted})
    return inventory


def _selected_internal_mapping(
    root: Path,
    mapping: pd.DataFrame,
    parent_provenance: dict[str, Any],
    combiner: Any,
) -> tuple[pd.DataFrame, dict[str, str]]:
    source_records = parent_provenance.get("source_runs")
    if not isinstance(source_records, list):
        raise ValueError("parent source-run provenance is malformed")
    source_documents = {str(item.get("run", "")): item for item in source_records}
    source_experiments = selection._source_experiments(root, source_records)
    packet_paths: list[str] = []
    case_ids: list[str] = []
    verified_experiments: set[str] = set()
    for row in mapping.to_dict(orient="records"):
        source_run = str(row["source_run"])
        source = source_documents[source_run]
        if str(source.get("source_mode")) != "sealed_curve_experiment":
            raise ValueError("final subset requires sealed curve-experiment sources")
        experiment = source_experiments[source_run]
        if str(experiment) not in verified_experiments:
            if selection._sha256(experiment / "SHA256SUMS.txt") != str(
                source["experiment_manifest_sha256"]
            ):
                raise ValueError(f"source experiment manifest changed: {experiment}")
            verified_experiments.add(str(experiment))
        parts = str(row["source_eos_id"]).split(":", 2)
        if len(parts) != 3 or parts[0] != source_run:
            raise ValueError(f"malformed source identity: {row['source_eos_id']}")
        packet = selection._direct_child(experiment, parts[1])
        packet_paths.append(str(packet))
        case_ids.append(parts[2])
    internal = mapping.rename(
        columns={
            "source_eos_id": "eos_id",
            "center_mev_fm3": "epsilon0_mev_fm3",
            "width_mev_fm3": "sigma_mev_fm3",
            "ramp_width_mev_fm3": "delta_mev_fm3",
        }
    ).copy()
    internal["source_case_id"] = case_ids
    internal["_packet"] = packet_paths
    return internal, dict(zip(mapping["name"].astype(str), case_ids))


def materialize(root: Path, parent: Path, dryrun: Path, destination: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    runs = (root / "runs").resolve(strict=True)
    catalogue_root = selection.catalogue.confined(runs / "eos_catalogue", runs)
    parent = selection.catalogue.confined(parent, runs)
    dryrun = selection.catalogue.confined(dryrun, runs)
    destination = selection.catalogue.confined(destination, runs)
    if not parent.is_dir() or not dryrun.is_dir():
        raise FileNotFoundError("parent and balanced dry run must both exist")
    if destination == runs:
        raise ValueError("materialized destination must be below runs/")
    selection.catalogue.require_disjoint(
        destination, parent, dryrun, catalogue_root
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    parent_manifest = selection._read_manifest(parent)
    dryrun_manifest = selection._read_manifest(dryrun)
    parent_provenance_path = selection._verified(
        parent, "provenance.json", parent_manifest
    )
    parent_mapping_path = selection._verified(
        parent, "ML_DATA/eos_name_mapping.csv", parent_manifest
    )
    parent_stellar_path = selection._verified(
        parent, "ML_DATA/hadronic_stellar_data.csv", parent_manifest
    )
    dryrun_provenance_path = selection._verified(
        dryrun, "provenance.json", dryrun_manifest
    )
    selected_path = selection._verified(
        dryrun, "selected_eos_2000.csv", dryrun_manifest
    )
    selection_manifest_path = selection._verified(
        dryrun, "selection_manifest.csv", dryrun_manifest
    )
    metrics_path = selection._verified(
        dryrun, "coverage_metrics.json", dryrun_manifest
    )
    gates_path = selection._verified(
        dryrun, "validation_gates.json", dryrun_manifest
    )
    parent_provenance = json.loads(parent_provenance_path.read_text(encoding="utf-8"))
    dryrun_provenance = json.loads(dryrun_provenance_path.read_text(encoding="utf-8"))
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_records = parent_provenance.get("source_runs")
    if not isinstance(source_records, list):
        raise ValueError("parent source-run provenance is malformed")
    source_experiments = selection._source_experiments(root, source_records)
    selection.catalogue.require_disjoint(
        destination, *source_experiments.values()
    )
    if not dryrun_provenance.get("all_validation_gates_passed") or not all(gates.values()):
        raise ValueError("balanced dry run did not pass every validation gate")
    if dryrun_provenance.get("selection_policy", "").find("water-filling") < 0:
        raise ValueError("dry run is not the reviewed balanced selection")
    if selection._sha256(Path(selection.__file__)) != str(
        dryrun_provenance["selector_sha256"]
    ):
        raise ValueError("selector changed after the validated balanced dry run")
    if selection._sha256(parent / "SHA256SUMS.txt") != str(
        dryrun_provenance["parent_manifest_sha256"]
    ):
        raise ValueError("parent manifest changed after balanced selection")

    public_mapping = pd.read_csv(parent_mapping_path, low_memory=False)
    stellar = pd.read_csv(parent_stellar_path, low_memory=False)
    selected_audit = pd.read_csv(selected_path, low_memory=False)
    selected_names = selected_audit["name"].astype(str).tolist()
    if len(selected_names) != 2000 or len(set(selected_names)) != 2000:
        raise ValueError("validated selection is not exactly 2,000 unique EoSs")
    if selected_audit["physical_model_key"].astype(str).nunique() != 2000:
        raise ValueError("validated selection contains duplicate physical identities")
    selected_set = set(selected_names)
    mapping = public_mapping.loc[public_mapping["name"].astype(str).isin(selected_set)].copy()
    if mapping["name"].astype(str).tolist() != selected_names:
        # The dry-run list follows parent order; any mismatch is evidence of a changed source.
        raise ValueError("selected-name order no longer matches the parent mapping")
    clean = stellar.loc[stellar["name"].astype(str).isin(selected_set), list(STELLAR_COLUMNS)].copy()
    if clean["name"].nunique() != 2000:
        raise ValueError("selected stellar table is missing a complete EoS")

    combiner = selection._load_combiner(root)
    internal_mapping, resolved_cases = _selected_internal_mapping(
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
        for name, rows in clean.groupby("name", sort=False)
    }
    if set(thermo) != selected_set or set(stellar_curves) != selected_set:
        raise ValueError("final curve collection is incomplete")
    baseline_names = mapping.loc[mapping["regime"].astype(str).eq("baseline"), "name"].astype(str).tolist()
    if len(baseline_names) != 1:
        raise ValueError("final subset must contain exactly one baseline")

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".balanced_2000_final_", dir=destination.parent))
    try:
        ml_data = stage / "ML_DATA"
        plots = stage / "plots"
        audit = stage / "AUDIT"
        ml_data.mkdir()
        plots.mkdir()
        audit.mkdir()
        clean.to_csv(
            ml_data / "hadronic_stellar_data.csv",
            index=False,
            columns=list(STELLAR_COLUMNS),
            float_format="%.17g",
            lineterminator="\n",
        )
        mapping.to_csv(
            ml_data / "eos_name_mapping.csv",
            index=False,
            columns=list(MAPPING_COLUMNS),
            float_format="%.17g",
            lineterminator="\n",
        )
        shutil.copy2(selection_manifest_path, audit / "selection_manifest.csv")
        shutil.copy2(metrics_path, audit / "coverage_metrics.json")
        shutil.copy2(gates_path, audit / "validation_gates.json")

        inventory = _draw_stellar(
            stellar_curves, selected_names, baseline_names[0], plots
        )
        inventory.extend(
            _draw_thermodynamic(thermo, selected_names, baseline_names[0], plots)
        )
        inventory.sort(key=lambda row: PLOT_FILES.index(row["figure"]))
        pd.DataFrame(inventory).to_csv(plots / "plot_inventory.csv", index=False)
        pngs = sorted(path.name for path in plots.glob("*.png"))
        if tuple(pngs) != tuple(sorted(PLOT_FILES)):
            raise ValueError(f"final plot inventory is not exactly five PNGs: {pngs}")

        provenance = {
            "schema_id": SCHEMA_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "destination": destination.relative_to(root).as_posix(),
            "parent_dataset": parent.relative_to(root).as_posix(),
            "parent_manifest_sha256": selection._sha256(parent / "SHA256SUMS.txt"),
            "balanced_dryrun": dryrun.relative_to(root).as_posix(),
            "balanced_dryrun_manifest_sha256": selection._sha256(
                dryrun / "SHA256SUMS.txt"
            ),
            "selected_eos_manifest_sha256": selection._sha256(selected_path),
            "selection_policy": dryrun_provenance["selection_policy"],
            "selected_eos_count": 2000,
            "clean_stellar_row_count": int(len(clean)),
            "clean_stellar_columns": list(STELLAR_COLUMNS),
            "regime_counts": {
                str(key): int(value)
                for key, value in mapping["regime"].value_counts().items()
            },
            "geometry_group_count": int(metrics["geometry_group_count_selected"]),
            "r14_range_km": [
                float(metrics["r14_selected"]["minimum_km"]),
                float(metrics["r14_selected"]["maximum_km"]),
            ],
            "r14_largest_adjacent_gap_km": float(
                metrics["r14_selected"]["largest_adjacent_gap_km"]
            ),
            "r14_density_coefficient_of_variation": float(
                metrics["r14_density"]["selected_count_coefficient_of_variation"]
            ),
            "mr_raster_below_2msun_retention_fraction": float(
                metrics["mr_raster_below_2msun_retention_fraction"]
            ),
            "mr_raster_all_mass_retention_fraction": float(
                metrics["mr_raster_cell_retention_fraction"]
            ),
            "generated_plot_count": len(PLOT_FILES),
            "plot_inventory": inventory,
            "plot_policy": "all selected curves black; baseline emphasized; saved tables only",
            "name_policy": "parent H identifiers preserved; no renumbering",
            "row_policy": "whole selected EoS sequences retained; no row-level subsampling",
            "solver_calls": 0,
            "authoritative_packets_modified": False,
            "all_selection_validation_gates_passed": True,
            "selector_sha256": selection._sha256(Path(selection.__file__)),
            "materializer_sha256": selection._sha256(Path(__file__)),
        }
        (stage / "provenance.json").write_bytes(selection._canonical(provenance) + b"\n")
        (ml_data / "README.md").write_text(
            "# Balanced BSk24 hadronic stellar dataset\n\n"
            "`hadronic_stellar_data.csv` contains exactly six columns: `name`, `M`, `R`, "
            "`Lambda`, `k2`, and `P_c`. Complete saved sequences are retained for exactly "
            "2,000 selected physical EoSs; no curve rows were independently subsampled. "
            "`eos_name_mapping.csv` preserves the parent H identifiers and source provenance.\n",
            encoding="utf-8",
            newline="\n",
        )
        (plots / "README.md").write_text(
            "# Final balanced saved-data plots\n\n"
            "These five PNGs contain all 2,000 selected EoSs in black. They were rendered "
            "from checksum-verified saved tables and profiles without scientific solver calls.\n",
            encoding="utf-8",
            newline="\n",
        )
        (stage / "README.md").write_text(
            "# Final balanced 2,000-EoS BSk24 dataset\n\n"
            "This immutable derivative was selected from the 3,124-EoS combined master using "
            "mandatory physical anchors, R1.4-bin water-filling, and sub-2-solar-mass M-R "
            "coverage rescue. The master and sealed source packets remain unchanged.\n",
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
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--dryrun", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = selection.catalogue.trusted_repository_root(args.repository_root)
    result = materialize(
        root, args.parent, args.dryrun, args.destination
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

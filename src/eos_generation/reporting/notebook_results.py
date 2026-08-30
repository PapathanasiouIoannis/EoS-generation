"""CFL notebook presentation built exclusively from validated saved tables.

The derived, independently hashed ``plots/`` view is a sibling of the sealed
experiment, never an addition to it. No reconstruction, interpolation, or
stellar solve is permitted here. Local display labels do not replace
scientific case IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from ._plotting_data import (
    _contiguous_valid_runs,
    _successful_sequence_runs,
    _validated_tidal_runs,
)
from .._internal.saved_tables import saved_tidal_valid_mask


REPORT_SCHEMA = "eos_generation_cfl_notebook_view_v1"
_PLOTS = (
    ("mass_radius.png", "stellar", "Radius", "Mass", "R [km]", r"$M/M_\odot$", "Mass–radius"),
    ("lambda_mass.png", "stellar", "Mass", "Lambda", r"$M/M_\odot$", r"$\Lambda$", "Tidal deformability"),
    ("k2_mass.png", "stellar", "Mass", "k2", r"$M/M_\odot$", r"$k_2$", "Tidal Love number"),
    ("sound_speed.png", "thermo", "epsilon_mev_fm3", "cs2", r"$\varepsilon$ [MeV fm$^{-3}$]", r"$c_s^2$", "Sound speed"),
    ("pressure.png", "thermo", "epsilon_mev_fm3", "pressure_mev_fm3", r"$\varepsilon$ [MeV fm$^{-3}$]", r"$P$ [MeV fm$^{-3}$]", "Equation of state"),
    ("baryon_density.png", "thermo", "epsilon_mev_fm3", "baryon_density_fm3", r"$\varepsilon$ [MeV fm$^{-3}$]", r"$n_B$ [fm$^{-3}$]", "Baryon density"),
    ("chemical_potential.png", "thermo", "epsilon_mev_fm3", "baryon_chemical_potential_mev", r"$\varepsilon$ [MeV fm$^{-3}$]", r"$\mu_B$ [MeV]", "Baryon chemical potential"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_identity(result: Any) -> dict[str, str]:
    return {
        path.relative_to(result.experiment_path).as_posix(): _sha(path)
        for path in (
            result.experiment_path / "SHA256SUMS.txt",
            *(packet / "SHA256SUMS.txt" for packet in result.packet_paths),
        )
    }


def _view_path(result: Any) -> Path:
    return result.experiment_path.parent / "plots"


def _optional_table(child: Any, name: str) -> pd.DataFrame:
    if not (Path(child.packet_path) / name).is_file():
        return pd.DataFrame()
    return child.table(name)


def _collect_saved_data(result: Any) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Keep all case outcomes; draw only accepted physical cases, A=0 once."""

    ledgers: list[pd.DataFrame] = []
    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in ("thermo", "stellar", "fixed", "maximum")}
    table_names = {
        "thermo": "thermodynamic_profiles.csv",
        "stellar": "stellar_sequences.csv",
        "fixed": "fixed_mass_observables.csv",
        "maximum": "maximum_mass_screening.csv",
    }
    seen: dict[str, set[str]] = {key: set() for key in frames}
    for index, child in enumerate(result.child_results, 1):
        ledger = child.table("case_ledger.csv").copy()
        ledger.insert(0, "geometry_index", index)
        if not {"physical_case_id", "status", "amplitude"}.issubset(ledger.columns):
            raise ValueError("CFL case ledger lacks physical identity or outcome")
        ledgers.append(ledger)
        accepted = set(ledger.loc[ledger.status.eq("accepted"), "physical_case_id"].astype(str))
        zero_ids = set(ledger.loc[ledger.status.eq("accepted") & ledger.amplitude.eq(0.0), "physical_case_id"].astype(str))
        for key, name in table_names.items():
            frame = _optional_table(child, name)
            if frame.empty:
                continue
            if key != "thermo":
                if not child.config.tov_stages:
                    raise ValueError("saved stellar rows have no declared stellar stage")
                frame = frame.loc[frame.stage.eq(child.config.tov_stages[-1].name)]
            # Stellar A=0 is calculated only as 'direct'. Resolve that saved
            # alias from the authoritative accepted zero row, not by a curve
            # comparison or by constructing a new baseline.
            for case_id, rows in frame.groupby("case_id", sort=False):
                case_id = str(case_id)
                if case_id == "direct" and key != "thermo":
                    if len(zero_ids) != 1:
                        raise ValueError("saved direct CFL star lacks one accepted A=0 identity")
                    case_id = next(iter(zero_ids))
                if case_id not in accepted or case_id in seen[key]:
                    continue
                rows = rows.copy()
                rows.insert(0, "geometry_index", index)
                rows["physical_case_id"] = case_id
                frames[key].append(rows)
                seen[key].add(case_id)
    ledger = pd.concat(ledgers, ignore_index=True)
    accepted_rows = ledger.loc[ledger.status.eq("accepted")].drop_duplicates("physical_case_id")
    ordered = accepted_rows.sort_values(
        ["amplitude", "epsilon0_mev_fm3", "sigma_mev_fm3", "delta_mev_fm3"], kind="stable"
    )
    labels: dict[str, str] = {}
    next_deformation = 1
    for row in ordered.itertuples(index=False):
        if row.amplitude == 0.0:
            label = "C000000"
        else:
            label = f"C{next_deformation:06d}"
            next_deformation += 1
        labels[str(row.physical_case_id)] = label
    ledger.insert(0, "eos_label", ledger.physical_case_id.astype(str).map(labels).fillna(""))
    tables: dict[str, pd.DataFrame] = {}
    for key, parts in frames.items():
        table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["physical_case_id"])
        table.insert(0, "eos_label", table.physical_case_id.astype(str).map(labels))
        tables[key] = table
    return ledger, tables


def _plot_runs(frame: pd.DataFrame, kind: str, x: str, y: str) -> tuple[pd.DataFrame, ...]:
    if frame.empty or not {x, y}.issubset(frame.columns):
        return ()
    if kind == "stellar":
        # Each segment is separated by an actual failed attempted model.
        groups = frame.groupby("segment_id", sort=False) if "segment_id" in frame else ((0, frame),)
        return tuple(
            run
            for _, segment in groups
            for run in (_successful_sequence_runs(segment) if y == "Mass" else _validated_tidal_runs(segment))
        )
    valid = np.isfinite(pd.to_numeric(frame[x], errors="coerce")) & np.isfinite(pd.to_numeric(frame[y], errors="coerce"))
    return _contiguous_valid_runs(frame, valid)


def _render_view(destination: Path, ledger: pd.DataFrame, tables: dict[str, pd.DataFrame], precision: str) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    accepted = ledger.loc[ledger.status.eq("accepted")].drop_duplicates("physical_case_id")
    amplitude = dict(zip(accepted.physical_case_id.astype(str), accepted.amplitude.astype(float)))
    values = list(amplitude.values()) or [0.0]
    norm = Normalize(min(values) if min(values) != max(values) else -1.0, max(values) if min(values) != max(values) else 1.0)
    cmap = plt.get_cmap("coolwarm")
    inventory: list[dict[str, Any]] = []
    for filename, kind, x, y, xlabel, ylabel, title in _PLOTS:
        fig, ax = plt.subplots(figsize=(7.5, 5.3), layout="constrained")
        plotted_rows = 0
        curve_count = 0
        frame = tables[kind]
        for case_id, rows in frame.groupby("physical_case_id", sort=False):
            runs = _plot_runs(rows, kind, x, y)
            if not runs:
                continue
            amp = amplitude[str(case_id)]
            color = "#111827" if amp == 0.0 else cmap(norm(amp))
            label = str(rows.eos_label.iloc[0])
            if amp != 0.0:
                label += f" (A={amp:g})"
            for run_index, run in enumerate(runs):
                ax.plot(run[x], run[y], color=color, linestyle="--" if amp == 0.0 else "-", linewidth=1.7,
                        marker="." if len(run) == 1 else None,
                        label=label if run_index == 0 and (len(accepted) <= 12 or amp == 0.0) else None)
                plotted_rows += len(run)
            curve_count += 1
            fixed = tables["fixed"]
            if kind == "stellar" and not fixed.empty:
                points = fixed.loc[fixed.physical_case_id.astype(str).eq(str(case_id))]
                if y == "Mass":
                    points = points.loc[points.status.eq("bracketed_and_solved")]
                    px, py = "radius_km", "target_mass_msun"
                else:
                    points = points.loc[saved_tidal_valid_mask(points, schema="fixed_mass")]
                    px, py = "target_mass_msun", "lambda_dimensionless" if y == "Lambda" else "k2"
                ax.scatter(points[px], points[py], color=color, s=18, zorder=4)
        if plotted_rows:
            ax.set(xlabel=xlabel, ylabel=ylabel, title=f"CFL · {title} · {precision}")
            ax.grid(alpha=0.2)
            if y == "Lambda":
                ax.set_yscale("log")
            if len(accepted) <= 12:
                ax.legend(fontsize=8, frameon=False)
            else:
                fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Deformation amplitude A")
                if any(value == 0.0 for value in values):
                    ax.legend(fontsize=8, frameon=False)
            fig.savefig(destination / filename, dpi=160)
        plt.close(fig)
        eligible_rows = len(frame)
        inventory.append({
            "figure": filename,
            "status": "generated" if plotted_rows else "unavailable",
            "reason": "saved final-stage rows only; gaps retained" if plotted_rows else "no eligible saved rows; no solver was run",
            "physical_curve_count": curve_count,
            "plotted_sequence_or_profile_rows": plotted_rows,
            "omitted_sequence_or_profile_rows": eligible_rows - plotted_rows,
            "partial": bool(plotted_rows and plotted_rows < eligible_rows),
        })
    return pd.DataFrame(inventory)


def _load_view(destination: Path, result: Any) -> dict[str, Any]:
    path = destination / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_id") != REPORT_SCHEMA or report.get("status") != "complete":
        raise ValueError("unrecognized or incomplete CFL notebook view")
    if report.get("input_manifests") != _input_identity(result):
        raise ValueError("CFL notebook view belongs to different saved inputs")
    expected = report.get("files", {})
    if not expected or set(p.name for p in destination.iterdir()) != set(expected) | {"report.json", "SHA256SUMS.txt"}:
        raise ValueError("CFL notebook view file inventory mismatch")
    for name, digest in expected.items():
        if Path(name).name != name or (destination / name).is_symlink() or _sha(destination / name) != digest:
            raise ValueError(f"CFL notebook view hash mismatch: {name}")
    hashes = {**expected, "report.json": _sha(path)}
    manifest = "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))
    if (destination / "SHA256SUMS.txt").read_text(encoding="utf-8") != manifest:
        raise ValueError("CFL notebook view manifest mismatch")
    return {**report, "path": destination}


def cfl_notebook_view(result: Any, *, create: bool = False) -> dict[str, Any] | None:
    """Validate/reuse a view, or explicitly build it once from saved tables."""

    from ..experiment import load_experiment

    if not isinstance(create, bool):
        raise TypeError("create must be boolean")
    result = load_experiment(result.experiment_path)
    if result.settings.matter_model != "cfl":
        raise ValueError("this notebook view requires a CFL experiment")
    destination = _view_path(result)
    if destination.exists():
        return _load_view(destination, result)
    if not create:
        return None
    root = result.repository_root.resolve()
    if not destination.is_relative_to(root / "runs") or destination.is_symlink():
        raise ValueError("CFL notebook views must remain below the owning runs directory")
    inputs = _input_identity(result)
    ledger, tables = _collect_saved_data(result)
    with tempfile.TemporaryDirectory(prefix=".cfl-view-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        inventory = _render_view(stage, ledger, tables, result.settings.precision)
        ledger.to_csv(stage / "case_catalogue.csv", index=False)
        tables["thermo"].to_csv(stage / "thermodynamic_profiles.csv", index=False)
        tables["stellar"].to_csv(stage / "stellar_sequences.csv", index=False)
        tables["fixed"].to_csv(stage / "fixed_mass_observables.csv", index=False)
        tables["maximum"].to_csv(stage / "maximum_mass_screening.csv", index=False)
        inventory.to_csv(stage / "plot_inventory.csv", index=False)
        (stage / "README.md").write_text(
            "# CFL saved-table view\n\n"
            "This derived folder contains accepted physical CFL EoSs from the "
            "current experiment only. `C000000` is the single undeformed CFL "
            "baseline; later C labels are deterministic within this experiment. "
            "Canonical `physical_case_id` values and `geometry_index` remain in "
            "every combined table. Figures and CSVs were built only from sealed "
            "saved tables and used zero scientific solver calls.\n",
            encoding="utf-8",
            newline="\n",
        )
        files = {p.name: _sha(p) for p in sorted(stage.iterdir())}
        report = {
            "schema_id": REPORT_SCHEMA,
            "status": "complete",
            "matter_model": "cfl",
            "precision": result.settings.precision,
            "settings_hash": result.settings.deterministic_hash(),
            "input_manifests": inputs,
            "reporting_source_sha256": {
                "reporting/notebook_results.py": _sha(Path(__file__)),
                "reporting/_plotting_data.py": _sha(Path(__file__).with_name("_plotting_data.py")),
                "_internal/saved_tables.py": _sha(Path(__file__).parents[1] / "_internal" / "saved_tables.py"),
            },
            "scientific_solver_calls": 0,
            "packet_writes": 0,
            "physical_eos_count": int(ledger.loc[ledger.status.eq("accepted"), "physical_case_id"].nunique()),
            "rejected_case_count": int(ledger.status.eq("rejected").sum()),
            "label_policy": "experiment_local_C_six_digit_labels_C000000_baseline_canonical_physical_ids_retained",
            "sequence_policy": "full_saved_sequence_not_a_stability_claim_gaps_never_bridged",
            "stage_policy": "final_requested_stage_only_all_stages_remain_in_packet",
            "figures": inventory.to_dict(orient="records"),
            "files": files,
        }
        (stage / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        hashes = {**files, "report.json": _sha(stage / "report.json")}
        (stage / "SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())), encoding="utf-8", newline="\n")
        # Revalidate authoritative bytes after rendering. A concurrent change
        # must never produce a successfully sealed derivative of mixed inputs.
        load_experiment(result.experiment_path)
        if inputs != _input_identity(result):
            raise RuntimeError("saved experiment changed while building its view")
        if destination.exists():
            raise FileExistsError(f"CFL view destination became occupied: {destination}")
        os.rename(stage, destination)
    return _load_view(destination, result)


def present_cfl_results(result: Any, *, create: bool = False) -> dict[str, Any]:
    """Render friendly tables and existing PNGs inline; never run a solver."""

    view = cfl_notebook_view(result, create=create)
    from IPython.display import Image, Markdown, display

    print(result.summary_text())
    if view is None:
        print("No combined saved-table view exists. Set BUILD_SAVED_PLOTS=True to create it without rerunning science.")
        return {"status": "view_not_created", "scientific_solver_calls": 0}
    destination = view["path"]
    link = quote(os.path.relpath(destination, Path.cwd()).replace(os.sep, "/"), safe="/.")
    display(Markdown(f"### Saved CFL results\n\n[Open combined plots and CSV tables]({link}/) · {view['physical_eos_count']} physical EoSs · {view['rejected_case_count']} rejected proposals\n\n`C000000` is the undeformed baseline. C labels are local to this experiment; canonical IDs remain in `case_catalogue.csv`. Lines are saved sampled sequences, not proof of stability. Dots mark successfully bracketed requested masses. Gaps and missing tides are retained."))
    for filename, columns in (
        ("case_catalogue.csv", ("eos_label", "geometry_index", "amplitude", "epsilon0_mev_fm3", "sigma_mev_fm3", "delta_mev_fm3", "status", "stellar_calculation", "rejection_reason")),
        ("fixed_mass_observables.csv", ("eos_label", "stage", "target_mass_msun", "status", "radius_km", "k2", "lambda_dimensionless", "tidal_status", "reason")),
        ("maximum_mass_screening.csv", ("eos_label", "stage", "status", "maximum_mass_resolved", "maximum_mass_msun", "radius_km", "passes_maximum_mass_threshold", "endpoint_limitation")),
    ):
        frame = pd.read_csv(destination / filename)
        if not frame.empty:
            display(Markdown(f"**{filename}**"))
            display(frame.loc[:, [name for name in columns if name in frame]])
    display(pd.read_csv(destination / "plot_inventory.csv"))
    for item in view["figures"]:
        if item["status"] == "generated":
            display(Image(filename=str(destination / item["figure"]), width=800))
    return view

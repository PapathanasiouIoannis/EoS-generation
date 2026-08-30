"""Five flat, accepted-only dataset figures from validated saved tables.

This presentation adapter never calls scientific solvers. Every curve retains
its source identity; failed attempts break lines and tides require valid status.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eos_catalogue as catalogue

FIGURES = (
    ("eos_pressure.png", "epsilon_mev_fm3", "pressure_mev_fm3", "Energy density [MeV fm$^{-3}$]", "Pressure [MeV fm$^{-3}$]"),
    ("speed_of_sound.png", "epsilon_mev_fm3", "cs2", "Energy density [MeV fm$^{-3}$]", "$c_s^2/c^2$"),
    ("mass_radius.png", "Radius", "Mass", "Radius [km]", "Mass [$M_\\odot$]"),
    ("k2_mass.png", "Mass", "k2", "Mass [$M_\\odot$]", "$k_2$"),
    ("lambda_mass.png", "Mass", "Lambda", "Mass [$M_\\odot$]", "$\\Lambda$"),
)


def sequence_xy(rows: pd.DataFrame, x: str, y: str) -> tuple[np.ndarray, np.ndarray]:
    """Keep pressure order and failed-attempt gaps through the sampled peak.

    This plotting prefix is not a radial-stability certificate or resolved Mmax.
    A missing peak does not authorize inventing a branch.
    """
    rows = rows.sort_values("attempted_index").copy()
    peaks = rows["is_sampled_peak"].astype(str).str.lower().eq("true")
    if not peaks.any():
        return np.array([]), np.array([])
    end = rows.loc[peaks, "attempted_index"].min()
    rows = rows.loc[rows.attempted_index.le(end)]
    valid = rows.calculation_status.eq("success")
    if y in {"k2", "Lambda"}:
        valid &= rows.tidal_status.eq("validated_lambda_validation_v1")
    xs = pd.to_numeric(rows[x], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(rows[y], errors="coerce").to_numpy(dtype=float)
    mask = valid.to_numpy() & np.isfinite(xs) & np.isfinite(ys)
    xs[~mask], ys[~mask] = np.nan, np.nan
    # Reindexing is unnecessary: sealed sequence validation requires every
    # attempted index, including explicit failed rows, to be present.
    return xs, ys


def collect_curves(root: Path, experiment: Path, eos_data: Path) -> tuple[list, list]:
    catalogue.validate_source(experiment)
    aliases = catalogue.load_aliases(root, experiment, eos_data)
    by_case = {}
    for row in aliases:
        for case_id in catalogue.case_identity_keys(
            row, source_field="source_case_id"
        ):
            key = (row["geometry_id"], case_id)
            if key in by_case:
                raise ValueError("duplicate source identity in aliases")
            by_case[key] = row
    document = catalogue.read_json(experiment / "experiment.json")
    curves, seen = [], set()
    for child in document["child_packets"]:
        packet = catalogue.confined(experiment / child, experiment)
        checksums = catalogue.manifest(packet)
        config = catalogue.read_json(catalogue.verified(packet, "complete_configuration.json", checksums))
        profile = pd.read_csv(catalogue.verified(packet, "thermodynamic_profiles.csv", checksums))
        sequence = pd.read_csv(catalogue.verified(packet, "stellar_sequences.csv", checksums))
        sequence = sequence.loc[sequence.stage.eq(config["tov_stages"][-1]["name"])]
        case_groups = list(profile.groupby("case_id", sort=False))
        # CFL packets retain both the direct baseline and its physical A=0
        # alias.  Prefer the direct row so the deduplicated baseline keeps the
        # stellar sequence stored under case_id="direct".
        case_groups.sort(key=lambda item: str(item[0]) != "direct")
        for case_id, rows in case_groups:
            alias = by_case[(child, str(case_id))]
            if not alias["eos_id"] or alias["status"] not in {"accepted", "baseline"}:
                raise ValueError("reconstructed curve does not have an accepted identity")
            key = (alias["catalogue_id"], alias["physical_model_key"])
            if key in seen:
                continue
            seen.add(key)
            curves.append({
                "alias": alias,
                "profile": rows.sort_values("epsilon_mev_fm3"),
                "sequence": sequence.loc[sequence.case_id.eq(case_id)].copy(),
            })
    return curves, aliases


def collect_curves_direct(experiment: Path) -> tuple[list, list]:
    """Collect validated packet curves without creating a duplicate catalogue."""

    catalogue.validate_source(experiment)
    document = catalogue.read_json(experiment / "experiment.json")
    curves: list[dict] = []
    aliases: list[dict] = []
    for child_index, child in enumerate(document["child_packets"], start=1):
        packet = catalogue.confined(experiment / child, experiment)
        checksums = catalogue.manifest(packet)
        config = catalogue.read_json(
            catalogue.verified(packet, "complete_configuration.json", checksums)
        )
        profile = pd.read_csv(
            catalogue.verified(packet, "thermodynamic_profiles.csv", checksums)
        )
        sequence = pd.read_csv(
            catalogue.verified(packet, "stellar_sequences.csv", checksums)
        )
        ledger = pd.read_csv(
            catalogue.verified(packet, "case_ledger.csv", checksums)
        )
        final_stage = config["tov_stages"][-1]["name"]
        sequence = sequence.loc[sequence.stage.eq(final_stage)]
        direct_present = profile.case_id.astype(str).eq("direct").any()
        for case_id, rows in profile.groupby("case_id", sort=False):
            case_id = str(case_id)
            matches = ledger.iloc[0:0]
            if case_id == "direct":
                amplitude = 0.0
            else:
                matches = ledger.loc[
                    ledger.case_id.astype(str).eq(case_id)
                    | ledger.get("physical_case_id", pd.Series("", index=ledger.index))
                    .astype(str)
                    .eq(case_id)
                ]
                amplitude_values = (
                    pd.to_numeric(rows["amplitude"], errors="coerce").dropna()
                    if "amplitude" in rows.columns
                    else pd.Series(dtype=float)
                )
                if amplitude_values.empty and len(matches) != 1:
                    raise ValueError(
                        f"curve {child}/{case_id} has no unique saved amplitude"
                    )
                amplitude = (
                    float(amplitude_values.iloc[0])
                    if not amplitude_values.empty
                    else float(matches.iloc[0]["amplitude"])
                )
            # The owned A=0 deformation is an exact alias of the direct
            # baseline and deliberately has no duplicate stellar solve.
            if case_id != "direct" and amplitude == 0.0 and direct_present:
                continue
            if case_id == "direct":
                status = "baseline"
            else:
                if len(matches) != 1 or str(matches.iloc[0]["status"]) != "accepted":
                    raise ValueError(
                        f"reconstructed curve {child}/{case_id} is not uniquely accepted"
                    )
                status = "accepted"
            geometry_label = f"G{child_index:03d}"
            eos_id = (
                f"{geometry_label} baseline"
                if case_id == "direct"
                else f"{geometry_label} A={amplitude:+.6g}"
            )
            alias = {
                "geometry_id": child,
                "source_case_id": case_id,
                "eos_id": eos_id,
                "amplitude": amplitude,
                "status": status,
                "matter_model": config.get("matter_model", "bsk24"),
            }
            aliases.append(alias)
            curves.append(
                {
                    "alias": alias,
                    "profile": rows.sort_values("epsilon_mev_fm3"),
                    "sequence": sequence.loc[
                        sequence.case_id.astype(str).eq(case_id)
                    ].copy(),
                }
            )
    return curves, aliases


def build_dataset_plots(
    root: Path,
    experiment: Path,
    destination: Path,
    eos_data: Path | None = None,
) -> dict:
    root = root.resolve(strict=True)
    runs = (root / "runs").resolve(strict=True)
    experiment = catalogue.confined(experiment, runs)
    target = catalogue.confined(destination, runs)
    eos_data = (
        None if eos_data is None else catalogue.confined(eos_data, runs)
    )
    protected = tuple(
        path
        for path in (experiment, eos_data, runs / "eos_catalogue")
        if path is not None
    )
    if target == runs or any(target == p or p in target.parents or target in p.parents for p in protected):
        raise ValueError("plot destination overlaps protected data")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    curves, aliases = (
        collect_curves_direct(experiment)
        if eos_data is None
        else collect_curves(root, experiment, eos_data)
    )
    # Presentation folders produced before the shared H/C schema are BSk24 by
    # construction and have no matter_model column.
    matter_models = {row.get("matter_model", "bsk24") for row in aliases}
    if len(matter_models) != 1 or next(iter(matter_models)) not in catalogue.MODEL_CONTRACTS:
        raise ValueError("dataset plots require exactly one recognized matter model")
    matter_model = next(iter(matter_models))
    family_title = "CFL" if matter_model == "cfl" else "BSk24"
    amplitudes = [float(c["alias"]["amplitude"]) for c in curves]
    scale = max([abs(v) for v in amplitudes] + [0.01])
    norm = matplotlib.colors.Normalize(-scale, scale)
    cmap = matplotlib.colormaps["coolwarm"]
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".dataset_plots_", dir=target.parent)).resolve()
    inventory = []
    try:
        for name, x, y, xlabel, ylabel in FIGURES:
            fig, ax = plt.subplots(figsize=(8.4, 6.1), layout="constrained")
            plotted = 0
            try:
                for curve in curves:
                    alias = curve["alias"]
                    amplitude = float(alias["amplitude"])
                    if x == "epsilon_mev_fm3":
                        xs = curve["profile"][x].to_numpy(dtype=float)
                        ys = curve["profile"][y].to_numpy(dtype=float)
                    else:
                        xs, ys = sequence_xy(curve["sequence"], x, y)
                    if not np.any(np.isfinite(xs) & np.isfinite(ys)):
                        continue
                    ax.plot(xs, ys, color="black" if amplitude == 0 else cmap(norm(amplitude)),
                            linewidth=1.4 if amplitude == 0 else 0.9, alpha=0.85,
                            label=alias["eos_id"] if len(curves) <= 12 else None)
                    plotted += 1
                ax.set(xlabel=xlabel, ylabel=ylabel)
                if name == "eos_pressure.png":
                    ax.set(xscale="linear", yscale="linear")
                if name == "lambda_mass.png":
                    ax.set_yscale("log")
                if name == "speed_of_sound.png":
                    ax.axhline(1, color="grey", linestyle=":", linewidth=0.8)
                ax.grid(True, alpha=0.2)
                ax.set_title(f"Accepted {family_title}-family EoSs — saved data")
                if plotted and len(curves) <= 12:
                    ax.legend(fontsize=8, ncol=2)
                elif plotted:
                    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Amplitude A")
                if not plotted:
                    ax.text(0.5, 0.5, "No available valid curves", transform=ax.transAxes, ha="center")
                fig.savefig(stage / name, dpi=160)
            finally:
                plt.close(fig)
            inventory.append({"figure": name, "plotted_eos_count": plotted})
        if eos_data is not None:
            catalogue.write_rows(
                stage / "accepted_case_index.csv",
                [c["alias"] for c in curves],
            )
            catalogue.write_rows(stage / "case_aliases.csv", aliases)
            pd.DataFrame(inventory).to_csv(
                stage / "plot_inventory.csv", index=False
            )
        provenance = {
            "schema_id": "eos_generation_dataset_plots_v1",
            "plots_path": str(target), "generated_plot_count": len(FIGURES),
            "accepted_case_occurrence_count": sum(r["status"] == "accepted" for r in aliases),
            "excluded_rejected_case_occurrence_count": sum(r["status"] == "rejected" for r in aliases),
            "matter_model": matter_model,
            "unique_eos_count": len(curves), "solver_calls": 0,
            "experiment": experiment.relative_to(root).as_posix(),
            "experiment_manifest_sha256": catalogue.sha256(experiment / "SHA256SUMS.txt"),
            "eos_data_manifest_sha256": (
                None
                if eos_data is None
                else catalogue.sha256(eos_data / "SHA256SUMS.txt")
            ),
            "builder_sha256": catalogue.sha256(Path(__file__)),
            "catalogue_builder_sha256": catalogue.sha256(Path(catalogue.__file__)),
            "axis_policy": {"eos_pressure": "linear-linear", "lambda_mass_y": "log"},
            "branch_policy": "attempted-pressure order through sampled peak; gaps preserved; not a stability or Mmax certificate",
        }
        (stage / "provenance.json").write_bytes(catalogue.canonical(provenance) + b"\n")
        if eos_data is not None:
            (stage / "README.md").write_text(
                "# Focused dataset plots\n\nFive figures combine accepted EoSs "
                "from this experiment only. No scientific solvers were run by "
                "plotting.\n",
                encoding="utf-8",
            )
        (stage / "SHA256SUMS.txt").write_text(
            "".join(f"{catalogue.sha256(p)}  {p.name}\n" for p in sorted(stage.iterdir())), encoding="utf-8"
        )
        catalogue.publish_directory(stage, target)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--eos-data", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_dataset_plots(args.repository_root, args.experiment, args.destination, args.eos_data)))

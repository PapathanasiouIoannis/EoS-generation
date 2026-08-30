"""Create one cumulative M-R plot from sealed notebook/CLI result packets.

The builder is deliberately outside the scientific package.  It reads only
completed, checksum-matching CSV/JSON artifacts below ``runs/`` and never
imports or calls thermodynamic, TOV, tidal, or maximum-mass solvers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def _load_path_adapter():
    """Load the audited sibling boundary, never code below a caller root."""

    module_path = Path(__file__).resolve(strict=True).with_name("eos_catalogue.py")
    spec = importlib.util.spec_from_file_location(
        "eos_catalogue_paths_for_combined_all", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PATHS = _load_path_adapter()


_SCHEMA_ID = "eos_generation_combined_all_data_mr_v1"
_REQUIRED_PACKET_FILES = (
    "case_ledger.csv",
    "complete_configuration.json",
    "fixed_mass_observables.csv",
    "metadata.json",
    "raw_gate_profiles.csv",
    "run_state.json",
    "source_hashes.json",
    "stellar_sequences.csv",
)
_CURVE_COLUMNS = (
    "attempted_index",
    "segment_id",
    "calculation_status",
    "Mass",
    "Radius",
)
_SOUND_COLUMNS = (
    "epsilon_mev_fm3",
    "raw_cs2",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable(path: Path, repository_root: Path) -> str:
    return path.resolve(strict=False).relative_to(repository_root).as_posix()


def _safe_child(parent: Path, name: str) -> Path:
    if not name or Path(name).is_absolute():
        raise ValueError(f"unsafe child packet name {name!r}")
    candidate = (parent / name).resolve(strict=False)
    try:
        candidate.relative_to(parent.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"child packet escapes experiment directory: {name!r}") from exc
    return candidate


def _checksum_manifest(packet: Path) -> dict[str, str]:
    manifest = packet / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise ValueError("missing SHA256SUMS.txt")
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64 or not relative:
            raise ValueError("malformed SHA256SUMS.txt entry")
        entries[relative] = digest.lower()
    return entries


def _verify_plot_sources(packet: Path) -> None:
    checksums = _checksum_manifest(packet)
    for relative in _REQUIRED_PACKET_FILES:
        path = packet / relative
        expected = checksums.get(relative)
        if expected is None or not path.is_file():
            raise ValueError(f"sealed plot source is missing: {relative}")
        if _sha256(path) != expected:
            raise ValueError(f"sealed plot source checksum mismatch: {relative}")


def _curve_fingerprint(rows: pd.DataFrame) -> str:
    ordered = rows.loc[:, list(_CURVE_COLUMNS)].copy()
    ordered["attempted_index"] = pd.to_numeric(
        ordered["attempted_index"], errors="raise"
    ).astype(int)
    ordered["segment_id"] = pd.to_numeric(
        ordered["segment_id"], errors="raise"
    ).astype(int)
    for name in ("Mass", "Radius"):
        ordered[name] = pd.to_numeric(ordered[name], errors="coerce")
    ordered.sort_values(["segment_id", "attempted_index"], inplace=True)
    payload = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sound_fingerprint(rows: pd.DataFrame) -> str:
    ordered = rows.loc[:, list(_SOUND_COLUMNS)].copy()
    for name in _SOUND_COLUMNS:
        ordered[name] = pd.to_numeric(ordered[name], errors="raise")
    ordered.sort_values("epsilon_mev_fm3", inplace=True)
    payload = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity(meta: dict[str, Any]) -> str:
    if meta["role"] == "direct":
        return "direct"
    values = (
        meta.get("amplitude"),
        meta.get("epsilon_match_mev_fm3"),
        meta.get("epsilon0_mev_fm3"),
        meta.get("sigma_mev_fm3"),
        meta.get("delta_mev_fm3"),
    )
    return "|".join("nan" if pd.isna(value) else f"{float(value):.17g}" for value in values)


def _case_metadata(case_id: str, ledger: pd.DataFrame) -> dict[str, Any]:
    if case_id == "direct":
        return {
            "case_id": "direct",
            "role": "direct",
            "status": "accepted",
            "amplitude": 0.0,
            "epsilon_match_mev_fm3": np.nan,
            "epsilon0_mev_fm3": np.nan,
            "sigma_mev_fm3": np.nan,
            "delta_mev_fm3": np.nan,
            "anchor_mode": "direct",
        }
    selected = ledger.loc[ledger["case_id"].astype(str).eq(case_id)]
    if len(selected) != 1:
        raise ValueError(f"case ledger cardinality mismatch for {case_id!r}")
    row = selected.iloc[0]
    amplitude = float(row["amplitude"])
    return {
        "case_id": case_id,
        "role": "zero_control" if math.isclose(amplitude, 0.0, abs_tol=0.0) else "deformation",
        "status": str(row["status"]),
        "amplitude": amplitude,
        "epsilon_match_mev_fm3": float(row["epsilon_match_mev_fm3"]),
        "epsilon0_mev_fm3": float(row["epsilon0_mev_fm3"]),
        "sigma_mev_fm3": float(row["sigma_mev_fm3"]),
        "delta_mev_fm3": float(row["delta_mev_fm3"]),
        "anchor_mode": str(row.get("anchor_mode", "unknown")),
    }


def _successful_runs(rows: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = rows.sort_values("attempted_index").copy()
    ordered["attempted_index"] = pd.to_numeric(
        ordered["attempted_index"], errors="raise"
    ).astype(int)
    successful = ordered.loc[
        ordered["calculation_status"].astype(str).eq("success")
    ].copy()
    if successful.empty:
        return []
    groups = successful["attempted_index"].diff().fillna(1).ne(1).cumsum()
    return [part for _, part in successful.groupby(groups, sort=False)]


def _amplitude_norm(values: list[float]) -> mpl.colors.Normalize:
    if not values:
        return mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    lower, upper = min(values), max(values)
    if lower < 0.0 < upper:
        return mpl.colors.TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
    if math.isclose(lower, upper):
        padding = max(abs(lower) * 0.1, 1.0e-6)
        lower -= padding
        upper += padding
    return mpl.colors.Normalize(vmin=lower, vmax=upper)


def _draw_plot(
    curves: dict[str, dict[str, Any]],
    fixed_rows: pd.DataFrame,
    output: Path,
    *,
    precision: str,
    experiment_count: int,
    packet_count: int,
    accepted_occurrences: int,
    rejected_occurrences: int,
    source_count: int,
) -> None:
    deformation_amplitudes = [
        float(item["meta"]["amplitude"])
        for item in curves.values()
        if item["meta"]["role"] == "deformation"
    ]
    norm = _amplitude_norm(deformation_amplitudes)
    cmap = mpl.colormaps["coolwarm"]
    fig, ax = plt.subplots(figsize=(11.8, 7.6))

    role_priority = {"deformation": 0, "zero_control": 1, "direct": 2}
    ordered_curves = sorted(
        curves.values(), key=lambda item: role_priority[item["meta"]["role"]]
    )
    plotted_segments = 0
    for item in ordered_curves:
        meta = item["meta"]
        rows = item["rows"]
        if meta["role"] == "direct":
            color, linewidth, alpha, linestyle, zorder = "#111827", 2.4, 1.0, "-", 5
        elif meta["role"] == "zero_control":
            color, linewidth, alpha, linestyle, zorder = "#4b5563", 1.2, 0.65, "-.", 4
        else:
            color = cmap(norm(float(meta["amplitude"])))
            linewidth, alpha, zorder = 0.78, 0.25, 2
            linestyle = "--" if meta["anchor_mode"] == "standard" else "-"
        for _, segment in rows.groupby("segment_id", sort=True):
            for run in _successful_runs(segment):
                mass = pd.to_numeric(run["Mass"], errors="coerce")
                radius = pd.to_numeric(run["Radius"], errors="coerce")
                finite = np.isfinite(mass) & np.isfinite(radius)
                if finite.any():
                    ax.plot(
                        radius[finite],
                        mass[finite],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                        linestyle=linestyle,
                        zorder=zorder,
                    )
                    plotted_segments += 1

    if not fixed_rows.empty:
        solved = fixed_rows.loc[
            fixed_rows["status"].astype(str).eq("bracketed_and_solved")
        ].drop_duplicates(
            ["physical_identity", "source_fingerprint", "mass_msun", "radius_km"]
        )
        deformation = solved.loc[solved["role"].eq("deformation")]
        if not deformation.empty:
            ax.scatter(
                deformation["radius_km"],
                deformation["mass_msun"],
                c=[cmap(norm(float(value))) for value in deformation["amplitude"]],
                s=17,
                alpha=0.62,
                linewidths=0,
                zorder=6,
            )
        baseline = solved.loc[solved["role"].isin(["direct", "zero_control"])]
        if not baseline.empty:
            ax.scatter(
                baseline["radius_km"],
                baseline["mass_msun"],
                color="#111827",
                marker="D",
                s=28,
                alpha=0.85,
                linewidths=0,
                zorder=7,
            )

    ax.set_xlabel("Radius R [km]")
    ax.set_ylabel(r"Gravitational mass $M/M_\odot$")
    ax.set_title(f"Combined all-data mass-radius sequences ({precision.upper()})")
    ax.grid(True, color="#d1d5db", linewidth=0.55, alpha=0.65)
    handles = [
        Line2D([0], [0], color="#111827", lw=2.4, label="direct BSk24"),
        Line2D([0], [0], color="#4b5563", lw=1.2, ls="-.", label="distinct A=0 control"),
        Line2D([0], [0], color="#6b7280", lw=1.1, ls="-", label="numeric anchor"),
        Line2D([0], [0], color="#6b7280", lw=1.1, ls="--", label="standard anchor"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6b7280", markersize=5, label="saved fixed-mass solution"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=3)
    if deformation_amplitudes:
        scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        fig.colorbar(scalar, ax=ax, label="signed deformation amplitude A", shrink=0.82)
    fig.text(
        0.01,
        0.008,
        (
            f"{experiment_count} sealed {precision.upper()} stellar experiments; "
            f"{packet_count} packets; {len(curves)} unique M-R curves; "
            f"{accepted_occurrences} accepted and {rejected_occurrences} rejected proposal occurrences; "
            f"{source_count} recorded source inventories. Saved tables only; 0 solver calls. "
            f"{plotted_segments} contiguous successful segments; failed gaps are not bridged."
        ),
        fontsize=7.2,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_sound_speed_plot(
    curves: dict[str, dict[str, Any]],
    output: Path,
    *,
    precision: str,
    experiment_count: int,
    packet_count: int,
    source_count: int,
) -> None:
    deformation_amplitudes = [
        float(item["meta"]["amplitude"])
        for item in curves.values()
        if item["meta"]["role"] == "deformation"
    ]
    norm = _amplitude_norm(deformation_amplitudes)
    cmap = mpl.colormaps["coolwarm"]
    finite_epsilon: list[np.ndarray] = []
    finite_cs2: list[np.ndarray] = []
    for item in curves.values():
        epsilon = pd.to_numeric(item["rows"]["epsilon_mev_fm3"], errors="coerce").to_numpy()
        cs2 = pd.to_numeric(item["rows"]["raw_cs2"], errors="coerce").to_numpy()
        finite = np.isfinite(epsilon) & np.isfinite(cs2)
        if finite.any():
            finite_epsilon.append(epsilon[finite])
            finite_cs2.append(cs2[finite])
    if not finite_epsilon:
        raise RuntimeError("no finite saved sound-speed rows were found")
    all_epsilon = np.concatenate(finite_epsilon)
    all_cs2 = np.concatenate(finite_cs2)
    x_min, x_max = float(all_epsilon.min()), float(all_epsilon.max())
    y_min, y_max = float(all_cs2.min()), float(all_cs2.max())
    y_padding = max(0.04 * (y_max - y_min), 0.025)

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.0), sharey=True)
    role_priority = {"deformation": 0, "zero_control": 1, "direct": 2}
    ordered_curves = sorted(
        curves.values(), key=lambda item: role_priority[item["meta"]["role"]]
    )
    accepted_unique = 0
    rejected_unique = 0
    for item in ordered_curves:
        meta = item["meta"]
        rows = item["rows"].sort_values("epsilon_mev_fm3")
        if meta["role"] in {"direct", "zero_control"}:
            color, linewidth, alpha, linestyle, zorder = "#111827", 2.2, 1.0, "-", 5
        else:
            color = cmap(norm(float(meta["amplitude"])))
            linewidth, zorder = 0.72, 2
            if meta["status"] == "accepted":
                accepted_unique += 1
                alpha = 0.24
                linestyle = "--" if meta["anchor_mode"] == "standard" else "-"
            else:
                rejected_unique += 1
                alpha = 0.48
                linestyle = ":"
        for ax in axes:
            ax.plot(
                rows["epsilon_mev_fm3"],
                rows["raw_cs2"],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                zorder=zorder,
            )

    core_min = max(x_min, 50.0)
    core_max = min(x_max, 1500.0)
    if core_max <= core_min:
        core_min, core_max = x_min, x_max
    axes[0].set_xlim(x_min, x_max)
    axes[1].set_xlim(core_min, core_max)
    axes[0].set_title("Complete saved raw-assessment domain")
    axes[1].set_title("Anchor and stellar-core zoom")
    for ax in axes:
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.set_xlabel(r"Total energy density $\varepsilon$ [MeV fm$^{-3}$]")
        ax.grid(True, color="#d1d5db", linewidth=0.55, alpha=0.65)
        ax.axhline(0.0, color="#991b1b", linewidth=0.8, alpha=0.85)
        ax.axhline(1.0 / 3.0, color="#6b7280", linewidth=0.8, linestyle=":")
        ax.axhline(1.0, color="#4b5563", linewidth=0.9, linestyle="--")
    axes[0].set_ylabel(r"Raw proposed dimensionless sound speed squared $c_s^2$ ($c=1$)")
    fig.suptitle(f"Combined all-data sound-speed proposals ({precision.upper()})")
    handles = [
        Line2D([0], [0], color="#111827", lw=2.2, label="direct/A=0 baseline"),
        Line2D([0], [0], color="#6b7280", lw=1.1, ls="-", label="accepted, numeric anchor"),
        Line2D([0], [0], color="#6b7280", lw=1.1, ls="--", label="accepted, standard anchor"),
        Line2D([0], [0], color="#6b7280", lw=1.1, ls=":", label="scientifically rejected raw proposal"),
        Line2D([0], [0], color="#4b5563", lw=0.9, ls="--", label=r"causal boundary $c_s^2=1$"),
        Line2D([0], [0], color="#6b7280", lw=0.8, ls=":", label=r"conformal reference $c_s^2=1/3$"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7.8, ncol=2)
    if deformation_amplitudes:
        scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        colorbar_axis = fig.add_axes([0.905, 0.19, 0.018, 0.58])
        fig.colorbar(
            scalar,
            cax=colorbar_axis,
            label="signed deformation amplitude A",
        )
    fig.text(
        0.5,
        0.058,
        (
            "The full-domain panel retains raw assessment evidence after a case's first causal "
            "endpoint; those post-endpoint samples are not a reopened usable EoS branch."
        ),
        ha="center",
        fontsize=7.4,
        color="#374151",
    )
    fig.text(
        0.01,
        0.008,
        (
            f"{experiment_count} sealed {precision.upper()} stellar experiments; "
            f"{packet_count} packets; {len(curves)} unique raw sound-speed curves "
            f"({accepted_unique} accepted deformations, {rejected_unique} rejected proposals, "
            f"plus exact-deduplicated A=0 controls); {source_count} recorded source inventories. "
            "Saved tables only; 0 solver calls."
        ),
        fontsize=7.2,
        color="#374151",
    )
    fig.subplots_adjust(bottom=0.15, top=0.89, wspace=0.16, right=0.88)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_combined_snapshot(
    repository_root: str | Path,
    current_experiment: str | Path,
    destination: str | Path,
    *,
    precision: str,
) -> dict[str, Any]:
    """Build a new cumulative snapshot without modifying any result packet."""

    root = Path(repository_root).resolve(strict=True)
    runs_root = (root / "runs").resolve(strict=True)
    current = _PATHS.confined(Path(current_experiment), runs_root)
    if not current.is_dir():
        raise FileNotFoundError(current)
    target = _PATHS.confined(Path(destination), runs_root)
    if precision not in {"quick", "strict"}:
        raise ValueError("precision must be 'quick' or 'strict'")
    if target == runs_root:
        raise ValueError("combined snapshot destination overlaps an authoritative packet")
    if target.exists():
        raise FileExistsError(f"combined snapshot destination already exists: {target}")
    catalogue_root = _PATHS.confined(runs_root / "eos_catalogue", runs_root)
    try:
        _PATHS.require_disjoint(target, current, catalogue_root)
    except ValueError as exc:
        raise ValueError(
            "combined snapshot destination overlaps an authoritative packet"
        ) from exc

    experiments = sorted(
        path.parent
        for path in runs_root.rglob("experiment.json")
        if path.parent.name.startswith("experiment_")
    )
    try:
        _PATHS.require_disjoint(target, *experiments)
    except ValueError as exc:
        raise ValueError(
            "combined snapshot destination overlaps a discovered experiment"
        ) from exc
    curves: dict[str, dict[str, Any]] = {}
    sound_curves: dict[str, dict[str, Any]] = {}
    occurrences: list[dict[str, Any]] = []
    sound_occurrences: list[dict[str, Any]] = []
    fixed_records: list[dict[str, Any]] = []
    included_experiments: list[dict[str, Any]] = []
    included_packets: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    accepted_occurrences = 0
    rejected_occurrences = 0
    physical_identities: set[str] = set()
    rejected_identities: set[str] = set()
    source_fingerprints: set[str] = set()
    current_packet_count = 0
    current_declared_packet_count: int | None = None

    for experiment in experiments:
        try:
            document = _read_json(experiment / "experiment.json")
            settings = document.get("settings")
            if not isinstance(settings, dict):
                raise ValueError("experiment settings are missing")
            if document.get("status") != "complete":
                raise ValueError("experiment status is not complete")
            if settings.get("calculation") != "stellar":
                continue
            if settings.get("precision") != precision:
                continue
            child_names = document.get("child_packets")
            child_hashes = document.get("child_configuration_hashes")
            if not isinstance(child_names, list) or not isinstance(child_hashes, list):
                raise ValueError("experiment child packet declaration is malformed")
            if len(child_names) != len(child_hashes) or len(set(child_names)) != len(child_names):
                raise ValueError("experiment child packet cardinality mismatch")
            if experiment == current:
                current_declared_packet_count = len(child_names)
        except Exception as exc:
            excluded.append(
                {"path": _portable(experiment, root), "scope": "experiment", "reason": str(exc)}
            )
            continue

        experiment_packet_count = 0
        experiment_curve_occurrences = 0
        for child_name, expected_hash in zip(child_names, child_hashes):
            try:
                packet = _safe_child(experiment, str(child_name))
                _verify_plot_sources(packet)
                run_state = _read_json(packet / "run_state.json")
                metadata = _read_json(packet / "metadata.json")
                config = _read_json(packet / "complete_configuration.json")
                if run_state.get("packet_status") != "complete" or metadata.get("packet_status") != "complete":
                    raise ValueError("packet status is not complete")
                if run_state.get("configuration_hash") != expected_hash or metadata.get("configuration_hash") != expected_hash:
                    raise ValueError("packet configuration hash disagrees with experiment")
                if metadata.get("baseline_validation_status") != "pass" or metadata.get("identity_status") != "pass":
                    raise ValueError("packet baseline or identity evidence did not pass")
                if config.get("stellar_enabled") is not True:
                    raise ValueError("packet is not a stellar calculation")
                stages = config.get("tov_stages")
                if not isinstance(stages, list) or not stages or not isinstance(stages[-1], dict):
                    raise ValueError("packet has no final TOV stage")
                final_stage = str(stages[-1].get("name", ""))
                if not final_stage:
                    raise ValueError("packet final TOV stage is unnamed")
                ledger = pd.read_csv(packet / "case_ledger.csv", low_memory=False)
                raw_profiles = pd.read_csv(packet / "raw_gate_profiles.csv", low_memory=False)
                sequences = pd.read_csv(packet / "stellar_sequences.csv", low_memory=False)
                fixed = pd.read_csv(packet / "fixed_mass_observables.csv", low_memory=False)
                required_ledger = {
                    "case_id", "amplitude", "epsilon_match_mev_fm3", "epsilon0_mev_fm3",
                    "sigma_mev_fm3", "delta_mev_fm3", "status",
                }
                if not required_ledger.issubset(ledger.columns) or not set(_CURVE_COLUMNS).issubset(sequences.columns):
                    raise ValueError("packet saved-table schema is incomplete")
                if not {"case_id", *_SOUND_COLUMNS}.issubset(raw_profiles.columns):
                    raise ValueError("packet raw sound-speed table schema is incomplete")
                source_fingerprint = _sha256(packet / "source_hashes.json")
                final_sequences = sequences.loc[sequences["stage"].astype(str).eq(final_stage)].copy()
                if final_sequences.empty:
                    raise ValueError("packet has no rows at the final TOV stage")

                accepted = ledger.loc[ledger["status"].astype(str).eq("accepted")]
                rejected = ledger.loc[~ledger["status"].astype(str).eq("accepted")]
                packet_identities: set[str] = set()
                packet_rejected_identities: set[str] = set()
                for row in ledger.itertuples(index=False):
                    meta = _case_metadata(str(row.case_id), ledger)
                    identity = _identity(meta)
                    packet_identities.add(identity)
                    if meta["status"] != "accepted":
                        packet_rejected_identities.add(identity)

                packet_occurrences: list[dict[str, Any]] = []
                packet_curve_candidates: list[
                    tuple[str, dict[str, Any], pd.DataFrame, dict[str, Any]]
                ] = []
                for case_id, rows in final_sequences.groupby("case_id", sort=False):
                    meta = _case_metadata(str(case_id), ledger)
                    if meta["status"] != "accepted":
                        raise ValueError(f"rejected case has stellar rows: {case_id}")
                    fingerprint = _curve_fingerprint(rows)
                    identity = _identity(meta)
                    record = {
                        "experiment_path": _portable(experiment, root),
                        "packet_path": _portable(packet, root),
                        "case_id": meta["case_id"],
                        "role": meta["role"],
                        "physical_identity": identity,
                        "curve_fingerprint": fingerprint,
                        "source_fingerprint": source_fingerprint,
                        "final_stage": final_stage,
                        "amplitude": meta["amplitude"],
                        "epsilon_match_mev_fm3": meta["epsilon_match_mev_fm3"],
                        "epsilon0_mev_fm3": meta["epsilon0_mev_fm3"],
                        "sigma_mev_fm3": meta["sigma_mev_fm3"],
                        "delta_mev_fm3": meta["delta_mev_fm3"],
                        "saved_row_count": len(rows),
                    }
                    packet_occurrences.append(record)
                    packet_curve_candidates.append(
                        (fingerprint, meta, rows.copy(), record)
                    )

                packet_sound_occurrences: list[dict[str, Any]] = []
                packet_sound_candidates: list[
                    tuple[str, dict[str, Any], pd.DataFrame, dict[str, Any]]
                ] = []
                for case_id, rows in raw_profiles.groupby("case_id", sort=False):
                    meta = _case_metadata(str(case_id), ledger)
                    fingerprint = _sound_fingerprint(rows)
                    record = {
                        "experiment_path": _portable(experiment, root),
                        "packet_path": _portable(packet, root),
                        "case_id": meta["case_id"],
                        "role": meta["role"],
                        "status": meta["status"],
                        "physical_identity": _identity(meta),
                        "sound_speed_fingerprint": fingerprint,
                        "source_fingerprint": source_fingerprint,
                        "amplitude": meta["amplitude"],
                        "epsilon_match_mev_fm3": meta["epsilon_match_mev_fm3"],
                        "epsilon0_mev_fm3": meta["epsilon0_mev_fm3"],
                        "sigma_mev_fm3": meta["sigma_mev_fm3"],
                        "delta_mev_fm3": meta["delta_mev_fm3"],
                        "saved_row_count": len(rows),
                    }
                    packet_sound_occurrences.append(record)
                    packet_sound_candidates.append(
                        (fingerprint, meta, rows.copy(), record)
                    )

                final_fixed = fixed.loc[fixed["stage"].astype(str).eq(final_stage)].copy()
                packet_fixed_records: list[dict[str, Any]] = []
                for row in final_fixed.itertuples(index=False):
                    meta = _case_metadata(str(row.case_id), ledger)
                    packet_fixed_records.append(
                        {
                            "experiment_path": _portable(experiment, root),
                            "packet_path": _portable(packet, root),
                            "case_id": meta["case_id"],
                            "role": meta["role"],
                            "physical_identity": _identity(meta),
                            "source_fingerprint": source_fingerprint,
                            "amplitude": meta["amplitude"],
                            "status": str(row.status),
                            "mass_msun": float(row.mass_msun) if pd.notna(row.mass_msun) else np.nan,
                            "radius_km": float(row.radius_km) if pd.notna(row.radius_km) else np.nan,
                        }
                    )

                # Commit only after the entire packet has passed every read,
                # schema, identity, and sealed-checksum check above.
                source_fingerprints.add(source_fingerprint)
                accepted_occurrences += len(accepted)
                rejected_occurrences += len(rejected)
                physical_identities.update(packet_identities)
                rejected_identities.update(packet_rejected_identities)
                occurrences.extend(packet_occurrences)
                sound_occurrences.extend(packet_sound_occurrences)
                fixed_records.extend(packet_fixed_records)
                experiment_curve_occurrences += len(packet_occurrences)
                for fingerprint, meta, rows, record in packet_curve_candidates:
                    representative = curves.get(fingerprint)
                    if representative is None or (
                        meta["role"] == "direct"
                        and representative["meta"]["role"] != "direct"
                    ):
                        curves[fingerprint] = {
                            "meta": meta,
                            "rows": rows,
                            "record": record,
                        }
                for fingerprint, meta, rows, record in packet_sound_candidates:
                    representative = sound_curves.get(fingerprint)
                    if representative is None or (
                        meta["role"] in {"direct", "zero_control"}
                        and representative["meta"]["role"] == "deformation"
                    ):
                        sound_curves[fingerprint] = {
                            "meta": meta,
                            "rows": rows,
                            "record": record,
                        }

                included_packets.append(
                    {
                        "experiment_path": _portable(experiment, root),
                        "packet_path": _portable(packet, root),
                        "configuration_hash": expected_hash,
                        "source_fingerprint": source_fingerprint,
                        "final_stage": final_stage,
                        "accepted_proposal_count": len(accepted),
                        "rejected_proposal_count": len(rejected),
                        "curve_occurrence_count": int(final_sequences["case_id"].nunique()),
                        "integrity_status": "required_saved_tables_match_sealed_sha256_manifest",
                    }
                )
                experiment_packet_count += 1
                if experiment == current:
                    current_packet_count += 1
            except Exception as exc:
                excluded.append(
                    {
                        "path": (
                            _portable(experiment, root)
                            + "/"
                            + str(child_name).replace("\\", "/")
                        ),
                        "scope": "packet",
                        "reason": str(exc),
                    }
                )

        if experiment_packet_count:
            included_experiments.append(
                {
                    "experiment_path": _portable(experiment, root),
                    "settings_hash": document.get("settings_hash"),
                    "precision": precision,
                    "declared_packet_count": len(child_names),
                    "included_packet_count": experiment_packet_count,
                    "curve_occurrence_count": experiment_curve_occurrences,
                    "all_declared_packets_included": experiment_packet_count == len(child_names),
                }
            )

    if current_declared_packet_count is None:
        raise RuntimeError("the just-completed experiment was not discovered as an in-scope stellar experiment")
    if current_packet_count != current_declared_packet_count:
        raise RuntimeError(
            "the just-completed experiment did not contribute every declared sealed stellar packet: "
            f"{current_packet_count}/{current_declared_packet_count}"
        )
    if not curves:
        raise RuntimeError("no sealed stellar M-R curves were found")
    if not sound_curves:
        raise RuntimeError("no sealed raw sound-speed curves were found")

    occurrence_counts = Counter(row["curve_fingerprint"] for row in occurrences)
    curve_index = []
    for fingerprint, item in sorted(curves.items()):
        row = dict(item["record"])
        row["duplicate_occurrence_count"] = occurrence_counts[fingerprint] - 1
        curve_index.append(row)
    fixed_frame = pd.DataFrame(fixed_records)
    sound_occurrence_counts = Counter(
        row["sound_speed_fingerprint"] for row in sound_occurrences
    )
    sound_curve_index = []
    for fingerprint, item in sorted(sound_curves.items()):
        row = dict(item["record"])
        row["duplicate_occurrence_count"] = sound_occurrence_counts[fingerprint] - 1
        sound_curve_index.append(row)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".combined_all_data_", dir=target.parent))
    try:
        plots = temporary / "plots"
        plots.mkdir()
        plot_path = plots / "combined_all_stellar_mr.png"
        sound_speed_plot_path = plots / "combined_all_speed_of_sound.png"
        _draw_plot(
            curves,
            fixed_frame,
            plot_path,
            precision=precision,
            experiment_count=len(included_experiments),
            packet_count=len(included_packets),
            accepted_occurrences=accepted_occurrences,
            rejected_occurrences=rejected_occurrences,
            source_count=len(source_fingerprints),
        )
        _draw_sound_speed_plot(
            sound_curves,
            sound_speed_plot_path,
            precision=precision,
            experiment_count=len(included_experiments),
            packet_count=len(included_packets),
            source_count=len(source_fingerprints),
        )
        pd.DataFrame(curve_index).to_csv(temporary / "combined_curve_index.csv", index=False)
        pd.DataFrame(occurrences).to_csv(temporary / "curve_occurrences.csv", index=False)
        pd.DataFrame(sound_curve_index).to_csv(
            temporary / "sound_speed_curve_index.csv", index=False
        )
        pd.DataFrame(sound_occurrences).to_csv(
            temporary / "sound_speed_curve_occurrences.csv", index=False
        )
        pd.DataFrame(fixed_records).to_csv(temporary / "fixed_mass_occurrences.csv", index=False)
        pd.DataFrame(included_experiments).to_csv(temporary / "included_experiments.csv", index=False)
        pd.DataFrame(included_packets).to_csv(temporary / "included_packets.csv", index=False)
        pd.DataFrame(excluded, columns=["path", "scope", "reason"]).to_csv(
            temporary / "excluded_sources.csv", index=False
        )
        provenance = {
            "schema_id": _SCHEMA_ID,
            "current_experiment": _portable(current, root),
            "precision": precision,
            "experiment_count": len(included_experiments),
            "packet_count": len(included_packets),
            "source_inventory_count": len(source_fingerprints),
            "accepted_proposal_occurrence_count": accepted_occurrences,
            "rejected_proposal_occurrence_count": rejected_occurrences,
            "unique_physical_identity_count": len(physical_identities),
            "unique_rejected_physical_identity_count": len(rejected_identities),
            "curve_occurrence_count": len(occurrences),
            "unique_curve_count": len(curves),
            "exact_duplicate_curve_occurrence_count": len(occurrences) - len(curves),
            "sound_speed_curve_occurrence_count": len(sound_occurrences),
            "unique_sound_speed_curve_count": len(sound_curves),
            "exact_duplicate_sound_speed_curve_occurrence_count": (
                len(sound_occurrences) - len(sound_curves)
            ),
            "excluded_source_count": len(excluded),
            "integrity_policy": "required plot-source files match each packet's sealed SHA256SUMS.txt",
            "failed_sequence_gap_policy": "successful contiguous attempted-index runs only; gaps are never bridged",
            "rejected_case_policy": "retained in counts/index provenance; no M-R curve exists to draw",
            "solver_calls": 0,
            "authoritative_packets_modified": False,
            "plot_builder_sha256": _sha256(Path(__file__).resolve()),
            "plot_relative_path": "plots/combined_all_stellar_mr.png",
            "sound_speed_plot_relative_path": "plots/combined_all_speed_of_sound.png",
        }
        (temporary / "plot_generation_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            "# Combined all-data M-R snapshot\n\n"
            f"This immutable derived snapshot combines every completed, sealed {precision.upper()} "
            "stellar result found below `runs/` at the time of the current notebook run. "
            "It reads saved tables only and made zero scientific solver calls.\n\n"
            "The two PNGs overlay every unique saved M-R curve and every unique saved raw "
            "sound-speed proposal. Exact duplicate curves—including "
            "deterministically repeated direct/A=0 baselines—are drawn once and remain counted in "
            "the corresponding occurrence indexes. Rejected cases remain explicit in the "
            "sound-speed plot and provenance counts but cannot have M-R curves because governed "
            "stellar work is not performed for them. "
            "Failed attempted-index gaps are never connected. Different recorded source inventories "
            "are retained together and counted explicitly; use the CSV indexes for provenance.\n\n"
            "This folder is derived and non-authoritative. The sealed experiment packets are not changed.\n",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            relative = path.relative_to(temporary).as_posix()
            checksum_lines.append(f"{_sha256(path)}  {relative}")
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        _PATHS.publish_directory(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    result = dict(provenance)
    result["snapshot_path"] = str(target)
    result["plot_path"] = str(target / "plots" / "combined_all_stellar_mr.png")
    result["sound_speed_plot_path"] = str(
        target / "plots" / "combined_all_speed_of_sound.png"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--current-experiment", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--precision", required=True, choices=("quick", "strict"))
    args = parser.parse_args()
    root = _PATHS.trusted_repository_root(args.repository_root)
    result = build_combined_snapshot(
        root,
        args.current_experiment,
        args.destination,
        precision=args.precision,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create one flat accepted-only plot folder for one completed experiment.

Only checksum-verified saved tables are read.  No scientific solver is
imported or called, and authoritative experiment packets are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_catalogue_adapter():
    """Load the trusted sibling saved-data/path adapter."""

    module_path = Path(__file__).resolve(strict=True).with_name("eos_catalogue.py")
    spec = importlib.util.spec_from_file_location(
        "eos_catalogue_plot_adapter", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("friendly EoS catalogue adapter is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CATALOGUE = _load_catalogue_adapter()


SCHEMA_ID = "eos_generation_experiment_plots_v1"
STANDARD_FIGURES = (
    "window_profiles.png",
    "gaussian_realization.png",
    "raw_cs2_full_domain.png",
    "raw_cs2_anchor_core_zoom.png",
    "delta_cs2.png",
    "pressure_response.png",
    "baryon_density_response.png",
    "effective_baryon_enthalpy_response.png",
    "gamma_eff_response.png",
    "thermodynamic_residuals.png",
    "stellar_mr_k2_lambda.png",
    "observable_response_vs_amplitude.png",
    "observable_response_vs_delta.png",
    "a0_identity.png",
)
OPTIONAL_FIGURES = (
    "deformation_support_fractions.png",
    "outside_support_control.png",
    "turning_point_sequences.png",
    "turning_point_derivatives.png",
    "baryonic_mass_vs_mass.png",
    "binding_energy_vs_mass.png",
    "stellar_response_across_mass.png",
    "baryonic_response_across_mass.png",
    "odd_even_response.png",
    "matched_area_comparison.png",
    "numerical_error_summary.png",
)
TABLES = (
    "raw_gate_profiles.csv",
    "thermodynamic_profiles.csv",
    "thermodynamic_residuals.csv",
    "stellar_sequences.csv",
    "fixed_mass_observables.csv",
    "a0_identity_table.csv",
    "case_plan.csv",
    "radial_profiles.csv",
    "deformation_support_fractions.csv",
    "outside_support_control.csv",
    "turning_point_sequences.csv",
    "baryonic_observables.csv",
    "baryonic_response_across_mass.csv",
    "odd_even_response.csv",
    "matched_area_comparison.csv",
    "numerical_error_summary.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def checksum_manifest(packet: Path) -> dict[str, str]:
    path = packet / "SHA256SUMS.txt"
    if not path.is_file():
        raise ValueError("missing SHA256SUMS.txt")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64 or not relative:
            raise ValueError("malformed SHA256SUMS.txt")
        result[relative] = digest.lower()
    return result


def verified_file(packet: Path, name: str, manifest: dict[str, str]) -> Path:
    path = packet / name
    expected = manifest.get(name)
    if expected is None or not path.is_file():
        raise ValueError(f"sealed plot source is missing: {name}")
    if sha256(path) != expected:
        raise ValueError(f"sealed plot source checksum mismatch: {name}")
    return path


def verified_csv(
    packet: Path,
    name: str,
    manifest: dict[str, str],
    *,
    required: bool = False,
) -> pd.DataFrame:
    if not (packet / name).is_file():
        if required:
            raise ValueError(f"sealed plot source is missing: {name}")
        return pd.DataFrame()
    return pd.read_csv(verified_file(packet, name, manifest), low_memory=False)


class CombinedConfig(SimpleNamespace):
    def deterministic_hash(self) -> str:
        return "0" * 64


def safe_child(experiment: Path, name: str) -> Path:
    child = (experiment / name).resolve(strict=False)
    if not name or Path(name).is_absolute() or child.parent != experiment:
        raise ValueError(f"unsafe child packet: {name!r}")
    return child


def apply_coordinates(
    frame: pd.DataFrame,
    *,
    geometry: str,
    ledger: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty or "case_id" not in frame.columns:
        result = frame.copy()
        if not result.empty:
            result["geometry_id"] = geometry
        return result
    result = frame.copy()
    result["source_case_id"] = result["case_id"].astype(str)
    result["geometry_id"] = geometry
    lookup = ledger.copy()
    lookup["case_id"] = lookup["case_id"].astype(str)
    lookup = lookup.set_index("case_id", drop=False)
    fields = (
        "amplitude",
        "epsilon_match_mev_fm3",
        "epsilon0_mev_fm3",
        "sigma_mev_fm3",
        "delta_mev_fm3",
        "anchor_mode",
    )
    values = {name: [] for name in fields}
    global_ids: list[str] = []
    for source_id in result["source_case_id"]:
        if source_id == "direct":
            global_ids.append("direct")
            row = {
                "amplitude": 0.0,
                "epsilon_match_mev_fm3": config.get("epsilon_match_mev_fm3"),
                "epsilon0_mev_fm3": config.get("epsilon0_mev_fm3"),
                "sigma_mev_fm3": config.get("sigma_mev_fm3"),
                "delta_mev_fm3": config.get("diagnostic_delta_mev_fm3"),
                "anchor_mode": "direct",
            }
        else:
            if source_id not in lookup.index:
                raise ValueError(f"saved-table case absent from ledger: {source_id}")
            row = lookup.loc[source_id]
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"duplicate ledger case: {source_id}")
            global_ids.append(f"{geometry}::{source_id}")
        for name in fields:
            values[name].append(row.get(name))
    result["case_id"] = global_ids
    for name, column in values.items():
        result[name] = column
    return result


def deduplicate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated direct/A=0 saved rows without merging distinct cases."""

    if frame.empty:
        return frame.copy()
    if "case_id" not in frame.columns:
        return frame.drop_duplicates().reset_index(drop=True)
    data_columns = [
        name
        for name in frame.columns
        if name
        not in {
            "geometry_id",
            "source_case_id",
            "case_id",
            "epsilon_match_mev_fm3",
            "epsilon0_mev_fm3",
            "sigma_mev_fm3",
            "delta_mev_fm3",
            "anchor_mode",
        }
    ]
    if not data_columns:
        return frame.copy()
    pieces: list[pd.DataFrame] = []
    fingerprints: set[str] = set()
    group_columns = ["case_id"]
    if "geometry_id" in frame.columns:
        group_columns.insert(0, "geometry_id")
    for _, rows in frame.groupby(group_columns, sort=False, dropna=False):
        ordered = rows.loc[:, data_columns].copy()
        payload = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()


def collect_experiment(root: Path, experiment: Path) -> dict[str, Any]:
    document = read_json(experiment / "experiment.json")
    if document.get("status") != "complete":
        raise ValueError("experiment status is not complete")
    settings = document.get("settings")
    children = document.get("child_packets")
    child_hashes = document.get("child_configuration_hashes")
    if not isinstance(settings, dict):
        raise ValueError("experiment settings are missing")
    if (
        not isinstance(children, list)
        or not children
        or not isinstance(child_hashes, list)
        or len(children) != len(child_hashes)
        or len(set(children)) != len(children)
    ):
        raise ValueError("experiment child declaration is malformed")

    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}
    accepted_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    generated_source_figures: set[str] = set()
    configs: list[dict[str, Any]] = []
    accepted_count = 0
    rejected_count = 0
    for child_name, expected_hash in zip(children, child_hashes, strict=True):
        packet = safe_child(experiment, str(child_name))
        manifest = checksum_manifest(packet)
        for name in (
            "complete_configuration.json",
            "metadata.json",
            "run_state.json",
            "source_hashes.json",
            "case_ledger.csv",
        ):
            verified_file(packet, name, manifest)
        config = read_json(packet / "complete_configuration.json")
        metadata = read_json(packet / "metadata.json")
        run_state = read_json(packet / "run_state.json")
        if metadata.get("packet_status") != "complete" or run_state.get("packet_status") != "complete":
            raise ValueError(f"incomplete child packet: {child_name}")
        if metadata.get("configuration_hash") != expected_hash or run_state.get("configuration_hash") != expected_hash:
            raise ValueError(f"configuration hash mismatch: {child_name}")
        if metadata.get("baseline_validation_status") != "pass" or metadata.get("identity_status") != "pass":
            raise ValueError(f"baseline or identity validation did not pass: {child_name}")
        ledger = verified_csv(packet, "case_ledger.csv", manifest, required=True)
        required = {
            "case_id",
            "status",
            "amplitude",
            "epsilon_match_mev_fm3",
            "epsilon0_mev_fm3",
            "sigma_mev_fm3",
            "delta_mev_fm3",
        }
        if not required.issubset(ledger.columns):
            raise ValueError("case ledger schema is incomplete")
        if bool(ledger["case_id"].astype(str).duplicated().any()):
            raise ValueError("case ledger contains duplicate IDs")
        accepted = set(
            ledger.loc[ledger["status"].astype(str).eq("accepted"), "case_id"].astype(str)
        )
        rejected = set(ledger["case_id"].astype(str)) - accepted
        accepted_count += len(accepted)
        rejected_count += len(rejected)
        configs.append(config)
        for row in ledger.loc[ledger["case_id"].astype(str).isin(accepted)].to_dict(orient="records"):
            accepted_rows.append(
                {
                    "geometry_id": child_name,
                    "packet_path": packet.relative_to(root).as_posix(),
                    "source_case_id": str(row["case_id"]),
                    "case_id": f"{child_name}::{row['case_id']}",
                    **{name: row.get(name) for name in required - {"case_id", "status"}},
                    "anchor_mode": row.get("anchor_mode", "unknown"),
                    "status": "accepted",
                }
            )
        for name in TABLES:
            table = verified_csv(packet, name, manifest)
            if table.empty:
                continue
            if "case_id" in table.columns:
                ids = table["case_id"].astype(str)
                table = table.loc[ids.eq("direct") | ids.isin(accepted)].copy()
            parts[name].append(
                apply_coordinates(
                    table,
                    geometry=str(child_name),
                    ledger=ledger,
                    config=config,
                )
            )
        inventory = verified_csv(packet, "plot_inventory.csv", manifest)
        if not inventory.empty and {"figure", "status"}.issubset(inventory.columns):
            generated_source_figures.update(
                inventory.loc[inventory["status"].astype(str).eq("generated"), "figure"].astype(str)
            )
        packet_rows.append(
            {
                "geometry_id": child_name,
                "packet_path": packet.relative_to(root).as_posix(),
                "configuration_hash": expected_hash,
                "accepted_case_count": len(accepted),
                "rejected_case_count": len(rejected),
                "integrity_status": "all consumed tables matched sealed SHA256SUMS.txt",
            }
        )
    frames = {
        name: deduplicate_rows(pd.concat(items, ignore_index=True)) if items else pd.DataFrame()
        for name, items in parts.items()
    }
    identity = frames["a0_identity_table.csv"]
    if not identity.empty:
        identity_columns = [name for name in identity.columns if name != "geometry_id"]
        frames["a0_identity_table.csv"] = identity.drop_duplicates(
            identity_columns
        ).reset_index(drop=True)
    return {
        "document": document,
        "settings": settings,
        "configs": configs,
        "frames": frames,
        "accepted_index": pd.DataFrame(accepted_rows),
        "packet_index": pd.DataFrame(packet_rows),
        "generated_source_figures": generated_source_figures,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
    }


def proxy_config(configs: list[dict[str, Any]]) -> CombinedConfig:
    amplitudes = sorted(
        {
            float(value)
            for config in configs
            for value in config.get("effective_amplitudes", config.get("amplitudes", []))
        }
    )
    deltas = sorted(
        {
            float(value)
            for config in configs
            for value in config.get("deltas_mev_fm3", [])
        }
    )
    masses = sorted(
        {
            float(value)
            for config in configs
            for value in config.get("fixed_masses_msun", [])
        }
    )
    stages = {
        str(config["tov_stages"][-1]["name"])
        for config in configs
        if config.get("tov_stages")
    }
    if len(stages) > 1:
        raise ValueError("child packets disagree on final TOV stage")
    stage = next(iter(stages), None)
    epsilon_matches = {
        float(config["epsilon_match_mev_fm3"])
        for config in configs
        if config.get("epsilon_match_mev_fm3") is not None
    }
    if len(epsilon_matches) > 1:
        raise ValueError("child packets disagree on matching anchor")
    epsilon_match = next(iter(epsilon_matches), 0.0)
    # NaN deliberately suppresses geometry-specific center/width guides: the
    # combined curves contain several geometries, so drawing one would lie.
    return CombinedConfig(
        effective_amplitudes=tuple(amplitudes),
        amplitudes=tuple(amplitudes),
        deltas_mev_fm3=tuple(deltas),
        fixed_masses_msun=tuple(masses),
        diagnostic_delta_mev_fm3=float("nan"),
        effective_epsilon_match_mev_fm3=epsilon_match,
        epsilon_match_mev_fm3=epsilon_match,
        epsilon0_mev_fm3=float("nan"),
        sigma_mev_fm3=float("nan"),
        tov_stages=(SimpleNamespace(name=stage),) if stage else (),
        thermodynamic_stages=(),
        background_tov_requested=bool(stage),
    )


def write_virtual_packet(
    folder: Path,
    collected: dict[str, Any],
) -> Path:
    virtual = folder / "_merged_saved_tables"
    virtual.mkdir()
    (virtual / "plots").mkdir()
    for name, frame in collected["frames"].items():
        if not frame.empty:
            frame.to_csv(virtual / name, index=False)
    ledger = collected["accepted_index"].copy()
    if not ledger.empty:
        ledger.to_csv(virtual / "case_ledger.csv", index=False)
    (virtual / "metadata.json").write_text(
        json.dumps(
            {
                "configuration_hash": "0" * 64,
                "accepted_case_ids": list(ledger.get("case_id", pd.Series(dtype=str))),
                "rejected_case_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return virtual


def amplitude_norm(frame: pd.DataFrame):
    values = [
        float(value)
        for value in pd.to_numeric(frame.get("amplitude"), errors="coerce").dropna()
    ]
    if not values:
        return mpl.colors.Normalize(-1.0, 1.0)
    lower, upper = min(values), max(values)
    if lower < 0.0 < upper:
        return mpl.colors.TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
    if lower == upper:
        upper = lower + max(abs(lower), 1.0) * 1.0e-12
    return mpl.colors.Normalize(vmin=lower, vmax=upper)


def finish(fig, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_dense_saved_figures(
    packet: Path,
    figures: tuple[str, ...],
    *,
    config: CombinedConfig,
) -> tuple[str, ...]:
    """Use the governed renderers with a combined-family colorbar policy."""

    from eos_generation.reporting import (
        _plotting_style, _plotting_thermodynamic, _plotting_stellar,
        _plotting_diagnostics,
    )
    from eos_generation.reporting.plotting import render_trial_figures

    previous = _plotting_style._AMPLITUDE_COLORBAR_THRESHOLD
    previous_scales = _plotting_thermodynamic._scales
    aliases = getattr(config, "eos_aliases", {})
    compact_aliases = bool(aliases) and len(set(aliases.values())) <= 12
    original_styles = _plotting_style._style_rows
    original_legend_label = _plotting_stellar._concise_mass_radius_legend_label
    style_modules = tuple(
        module for module in (
            _plotting_style, _plotting_thermodynamic, _plotting_stellar,
            _plotting_diagnostics,
        ) if getattr(module, "_style_rows", None) is original_styles
    )

    def labelled_styles(frame):
        styles = original_styles(frame)
        for case_id, style in styles.items():
            if case_id in aliases and (compact_aliases or case_id == "direct"):
                style["label"] = aliases[case_id] + (" (BSk24)" if case_id == "direct" else "")
        return styles

    def labelled_legend(label):
        if label == "Direct BSk24" and "direct" in aliases:
            return f"{aliases['direct']} (BSk24)"
        return original_legend_label(label)

    def combined_scales(axis, active_config, delta=None) -> None:
        del delta
        axis.axvline(
            active_config.effective_epsilon_match_mev_fm3,
            color="#4b5563",
            linestyle=":",
            linewidth=0.9,
            label=r"shared $\varepsilon_t$",
        )

    _plotting_style._AMPLITUDE_COLORBAR_THRESHOLD = 100000 if compact_aliases else 0
    _plotting_thermodynamic._scales = combined_scales
    _plotting_stellar._concise_mass_radius_legend_label = labelled_legend
    for module in style_modules:
        module._style_rows = labelled_styles
    try:
        return render_trial_figures(packet, figures, config=config)
    finally:
        _plotting_style._AMPLITUDE_COLORBAR_THRESHOLD = previous
        _plotting_thermodynamic._scales = previous_scales
        _plotting_stellar._concise_mass_radius_legend_label = original_legend_label
        for module in style_modules:
            module._style_rows = original_styles


def draw_geometry_curves(
    frame: pd.DataFrame,
    output: Path,
    *,
    y_fields: tuple[tuple[str, str], ...],
    title: str,
) -> bool:
    required = {"case_id", "epsilon_mev_fm3", "epsilon0_mev_fm3", *(name for name, _ in y_fields)}
    if frame.empty or not required.issubset(frame.columns):
        return False
    centers = pd.to_numeric(frame["epsilon0_mev_fm3"], errors="coerce")
    lower, upper = float(centers.min()), float(centers.max())
    norm = mpl.colors.Normalize(lower, upper if upper != lower else upper + 1.0)
    cmap = mpl.colormaps["viridis"]
    fig, axes = plt.subplots(1, len(y_fields), figsize=(6.0 * len(y_fields), 5.0), squeeze=False)
    for _, rows in frame.groupby("case_id", sort=False):
        color = cmap(norm(float(rows["epsilon0_mev_fm3"].iloc[0])))
        for axis, (field, ylabel) in zip(axes.flat, y_fields, strict=True):
            axis.plot(rows["epsilon_mev_fm3"], rows[field], color=color, linewidth=0.85, alpha=0.48)
            axis.set(xlabel=r"$\varepsilon$ [MeV fm$^{-3}$]", ylabel=ylabel)
            axis.grid(True, alpha=0.25)
    fig.suptitle(title)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.colorbar(
        scalar,
        ax=list(axes.flat),
        label=r"deformation center $\varepsilon_0$ [MeV fm$^{-3}$]",
        fraction=0.035,
        pad=0.025,
    )
    finish(fig, output)
    return True


def draw_window_profiles(raw: pd.DataFrame, output: Path) -> bool:
    if "window" not in raw.columns:
        return False
    shapes = raw.drop_duplicates(
        ["geometry_id", "epsilon_mev_fm3", "window"]
    )
    return draw_geometry_curves(
        shapes,
        output,
        y_fields=(("window", r"$W(\varepsilon)$"),),
        title="All accepted unique smootherstep windows",
    )


def draw_gaussian_realization(raw: pd.DataFrame, output: Path) -> bool:
    if not {"gaussian", "window"}.issubset(raw.columns):
        return False
    shapes = raw.drop_duplicates(
        ["geometry_id", "epsilon_mev_fm3", "gaussian", "window"]
    ).copy()
    shapes["realized_shape"] = shapes["gaussian"] * shapes["window"]
    return draw_geometry_curves(
        shapes,
        output,
        y_fields=(
            ("gaussian", "nominal Gaussian"),
            ("realized_shape", "windowed realized shape"),
        ),
        title="All accepted unique deformation geometries",
    )


def draw_sound_speed(
    raw: pd.DataFrame,
    thermo: pd.DataFrame,
    output: Path,
    *,
    zoom: bool,
) -> bool:
    required = {"case_id", "epsilon_mev_fm3", "raw_cs2", "amplitude"}
    if raw.empty or not required.issubset(raw.columns):
        return False
    fig, axis = plt.subplots(figsize=(9.4, 5.7))
    norm = amplitude_norm(raw)
    cmap = mpl.colormaps["coolwarm"]
    direct = thermo.loc[
        thermo.get("case_id", pd.Series(dtype=str)).astype(str).eq("direct")
    ]
    if not direct.empty and {"epsilon_mev_fm3", "cs2"}.issubset(direct.columns):
        axis.plot(
            direct["epsilon_mev_fm3"],
            direct["cs2"],
            color="#111827",
            linewidth=2.4,
            label=(
                f"{direct['eos_id'].iloc[0]} (BSk24)"
                if "eos_id" in direct else "direct BSk24"
            ),
            zorder=5,
        )
    for case_id, rows in raw.groupby("case_id", sort=False):
        amplitude = float(rows["amplitude"].iloc[0])
        if np.isclose(amplitude, 0.0):
            color, linestyle, linewidth, alpha = "#4b5563", "-.", 1.1, 0.55
        else:
            color = cmap(norm(amplitude))
            linestyle = "--" if str(rows["anchor_mode"].iloc[0]) == "standard" else "-"
            linewidth, alpha = 0.8, 0.28
        axis.plot(
            rows["epsilon_mev_fm3"],
            rows["raw_cs2"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=(
                str(rows["eos_id"].iloc[0])
                if "eos_id" in raw and raw["eos_id"].nunique() <= 12
                else "_nolegend_"
            ),
        )
    axis.axhline(0.0, color="#991b1b", linewidth=0.8)
    axis.axhline(1.0 / 3.0, color="#6b7280", linestyle=":", linewidth=0.8, label=r"$c_s^2=1/3$")
    axis.axhline(1.0, color="#4b5563", linestyle="--", linewidth=0.9, label=r"$c_s^2=1$")
    if zoom:
        lower = max(
            0.0,
            float(pd.to_numeric(raw["epsilon_match_mev_fm3"], errors="coerce").min()) - 20.0,
        )
        upper = float(
            (
                pd.to_numeric(raw["epsilon0_mev_fm3"], errors="coerce")
                + 3.0 * pd.to_numeric(raw["sigma_mev_fm3"], errors="coerce")
            ).max()
        )
        axis.set_xlim(lower, upper)
        title = "All accepted raw sound-speed profiles: anchor and core zoom"
    else:
        title = "All accepted raw sound-speed profiles: complete saved domain"
    axis.set(
        xlabel=r"$\varepsilon$ [MeV fm$^{-3}$]",
        ylabel=r"raw $c_s^2$",
        title=title,
    )
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.colorbar(scalar, ax=axis, label="signed deformation amplitude A")
    finish(fig, output)
    return True


def draw_observable_response(
    fixed: pd.DataFrame,
    output: Path,
    *,
    versus: str,
) -> bool:
    needed = {
        "source_case_id",
        "status",
        "target_mass_msun",
        "amplitude",
        "delta_mev_fm3",
        "epsilon_match_mev_fm3",
        "epsilon0_mev_fm3",
        "sigma_mev_fm3",
        "radius_km",
        "central_energy_density_mev_fm3",
    }
    if fixed.empty or not needed.issubset(fixed.columns):
        return False
    rows = fixed.loc[
        ~fixed["source_case_id"].astype(str).eq("direct")
        & fixed["status"].astype(str).eq("bracketed_and_solved")
    ].copy()
    masses = pd.to_numeric(rows["target_mass_msun"], errors="coerce").dropna().unique()
    if rows.empty or not len(masses):
        return False
    target = float(min(masses, key=lambda value: abs(float(value) - 1.4)))
    rows = rows.loc[np.isclose(rows["target_mass_msun"], target)]
    if versus == "amplitude":
        xcolumn, xlabel = "amplitude", "A"
        groups = ["epsilon_match_mev_fm3", "epsilon0_mev_fm3", "sigma_mev_fm3", "delta_mev_fm3"]
        title = "All accepted observables versus amplitude"
    else:
        xcolumn, xlabel = "delta_mev_fm3", r"$\Delta$ [MeV fm$^{-3}$]"
        groups = ["epsilon_match_mev_fm3", "epsilon0_mev_fm3", "sigma_mev_fm3", "amplitude"]
        title = "All accepted observables versus ramp width"
    families = [
        group.sort_values(xcolumn)
        for _, group in rows.groupby(groups, dropna=False)
        if group[xcolumn].nunique() >= 2
    ]
    if not families:
        return False
    observables = (
        ("radius_km", rf"$R_{{{target:g}}}$ [km]"),
        ("k2", rf"$k_{{2,{target:g}}}$"),
        ("lambda_dimensionless", rf"$\Lambda_{{{target:g}}}$"),
        ("central_energy_density_mev_fm3", r"$\varepsilon_c$ [MeV fm$^{-3}$]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    norm = amplitude_norm(rows)
    cmap = mpl.colormaps["coolwarm"]
    for axis, (field, ylabel) in zip(axes.flat, observables, strict=True):
        if field not in rows.columns:
            axis.set_axis_off()
            continue
        for family in families:
            valid = np.isfinite(pd.to_numeric(family[field], errors="coerce"))
            plot = family.loc[valid]
            if plot.empty:
                continue
            color = (
                cmap(norm(float(plot["amplitude"].iloc[0])))
                if versus != "amplitude"
                else "#4b5563"
            )
            axis.plot(
                plot[xcolumn],
                plot[field],
                marker="o",
                markersize=3,
                linewidth=0.8,
                alpha=0.45,
                color=color,
            )
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(True, alpha=0.25)
    fig.suptitle(f"{title} at {target:g} solar masses")
    finish(fig, output)
    return True


def draw_radial_profiles(radial: pd.DataFrame, output: Path) -> bool:
    fields = (
        ("energy_density_mev_fm3", r"$\varepsilon$ [MeV fm$^{-3}$]"),
        ("pressure_mev_fm3", r"$P$ [MeV fm$^{-3}$]"),
        ("cs2", r"$c_s^2$"),
        ("enclosed_mass_over_M", r"$m(r)/M$"),
    )
    required = {"case_id", "radius_over_R", *(name for name, _ in fields)}
    if radial.empty or not required.issubset(radial.columns):
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))
    norm = amplitude_norm(radial)
    cmap = mpl.colormaps["coolwarm"]
    for _, rows in radial.groupby(["case_id", "target_mass_msun"], sort=False):
        rows = rows.sort_values("radius_over_R")
        color = "#111827" if str(rows["case_id"].iloc[0]) == "direct" else cmap(norm(float(rows["amplitude"].iloc[0])))
        for axis, (field, ylabel) in zip(axes.flat, fields, strict=True):
            axis.plot(rows["radius_over_R"], rows[field], color=color, linewidth=0.75, alpha=0.35)
            axis.set(xlabel=r"$r/R$", ylabel=ylabel)
            axis.grid(True, alpha=0.25)
    fig.suptitle("All accepted saved fixed-mass radial profiles")
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.colorbar(scalar, ax=list(axes.flat), label="signed deformation amplitude A")
    finish(fig, output)
    return True


def write_checksums(folder: Path) -> None:
    manifest = folder / "SHA256SUMS.txt"
    files = sorted(path for path in folder.iterdir() if path.is_file() and path != manifest)
    manifest.write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in files) + "\n",
        encoding="utf-8",
    )


def build_experiment_plots(
    repository_root: str | Path,
    experiment_path: str | Path,
    destination: str | Path,
    *,
    eos_data_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a new immutable, flat, current-experiment plot folder."""

    root = Path(repository_root).resolve(strict=True)
    runs_root = (root / "runs").resolve(strict=True)
    experiment = _CATALOGUE.confined(Path(experiment_path), runs_root)
    if not experiment.is_dir():
        raise FileNotFoundError(experiment)
    target = _CATALOGUE.confined(Path(destination), runs_root)
    if target == runs_root:
        raise ValueError("plot destination overlaps an authoritative experiment")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"plot destination already exists: {target}")
    catalogue_root = _CATALOGUE.confined(runs_root / "eos_catalogue", runs_root)
    sources = [experiment, catalogue_root]
    alias_folder = None
    if eos_data_path is not None:
        alias_folder = _CATALOGUE.confined(Path(eos_data_path), runs_root)
        if not alias_folder.is_dir():
            raise FileNotFoundError(alias_folder)
        sources.append(alias_folder)
    try:
        _CATALOGUE.require_disjoint(target, *sources)
    except ValueError as exc:
        raise ValueError(
            "plot destination overlaps an authoritative experiment or saved-data source"
        ) from exc

    collected = collect_experiment(root, experiment)
    alias_rows = None
    if alias_folder is not None:
        alias_rows = _CATALOGUE.load_aliases(root, experiment, alias_folder)
        by_case = {(row["geometry_id"], row["source_case_id"]): row for row in alias_rows}
        if len(by_case) != len(alias_rows):
            raise ValueError("duplicate friendly EoS mapping")
        for frame in [collected["accepted_index"], *collected["frames"].values()]:
            if frame.empty or not {"geometry_id", "source_case_id"}.issubset(frame.columns):
                continue
            selected = [by_case[(str(g), str(c))] for g, c in zip(frame["geometry_id"], frame["source_case_id"], strict=True)]
            if any(not row["eos_id"] for row in selected):
                raise ValueError("accepted plot case has no friendly EoS ID")
            for column in ("eos_id", "catalogue_id", "physical_model_key"):
                frame[column] = [row[column] for row in selected]
    frames = collected["frames"]
    raw = frames["raw_gate_profiles.csv"]
    if "gate_status" in raw.columns:
        raw = raw.loc[
            raw["gate_status"].astype(str).eq("accepted_raw_local_physics_gate")
        ].copy()
        frames["raw_gate_profiles.csv"] = raw
    final_stages = {
        str(config["tov_stages"][-1]["name"])
        for config in collected["configs"]
        if config.get("tov_stages")
    }
    for name in ("stellar_sequences.csv", "fixed_mass_observables.csv"):
        frame = frames[name]
        if not frame.empty and final_stages and "stage" in frame.columns:
            frames[name] = frame.loc[frame["stage"].astype(str).isin(final_stages)].copy()

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".experiment_plots_", dir=target.parent))
    inventory: list[dict[str, Any]] = []
    try:
        virtual = write_virtual_packet(temporary, collected)
        config = proxy_config(collected["configs"])
        if alias_rows is not None:
            config.eos_aliases = {
                ("direct" if row["source_case_id"] == "direct" else f"{row['geometry_id']}::{row['source_case_id']}"): row["eos_id"]
                for row in alias_rows if row["eos_id"]
            }
        existing_renderers = tuple(
            name
            for name in STANDARD_FIGURES
            if name
            not in {
                "window_profiles.png",
                "gaussian_realization.png",
                "raw_cs2_full_domain.png",
                "raw_cs2_anchor_core_zoom.png",
                "observable_response_vs_amplitude.png",
                "observable_response_vs_delta.png",
            }
        )
        generated = set(
            render_dense_saved_figures(
                virtual, existing_renderers, config=config
            )
        )
        for name in generated:
            shutil.move(str(virtual / "plots" / name), temporary / name)

        custom = {
            "window_profiles.png": draw_window_profiles(
                raw, temporary / "window_profiles.png"
            ),
            "gaussian_realization.png": draw_gaussian_realization(
                raw, temporary / "gaussian_realization.png"
            ),
            "raw_cs2_full_domain.png": draw_sound_speed(
                raw,
                frames["thermodynamic_profiles.csv"],
                temporary / "raw_cs2_full_domain.png",
                zoom=False,
            ),
            "raw_cs2_anchor_core_zoom.png": draw_sound_speed(
                raw,
                frames["thermodynamic_profiles.csv"],
                temporary / "raw_cs2_anchor_core_zoom.png",
                zoom=True,
            ),
            "observable_response_vs_amplitude.png": draw_observable_response(
                frames["fixed_mass_observables.csv"],
                temporary / "observable_response_vs_amplitude.png",
                versus="amplitude",
            ),
            "observable_response_vs_delta.png": draw_observable_response(
                frames["fixed_mass_observables.csv"],
                temporary / "observable_response_vs_delta.png",
                versus="delta",
            ),
        }
        generated.update(name for name, status in custom.items() if status)

        if "radial_structure_profiles.png" in collected["generated_source_figures"]:
            if draw_radial_profiles(
                frames["radial_profiles.csv"],
                temporary / "radial_structure_profiles.png",
            ):
                generated.add("radial_structure_profiles.png")
        optional_requested = tuple(
            name
            for name in OPTIONAL_FIGURES
            if name != "radial_structure_profiles.png"
            and name in collected["generated_source_figures"]
        )
        optional_generated = set(
            render_dense_saved_figures(
                virtual, optional_requested, config=config
            )
        )
        for name in optional_generated:
            shutil.move(str(virtual / "plots" / name), temporary / name)
        generated.update(optional_generated)
        shutil.rmtree(virtual)

        for name in (*STANDARD_FIGURES, *optional_requested):
            inventory.append(
                {
                    "figure": name,
                    "status": "generated" if name in generated else "skipped",
                    "reason": (
                        "all accepted current-experiment saved rows combined"
                        if name in generated
                        else "required accepted saved rows were unavailable or insufficient"
                    ),
                    "relative_path": name if name in generated else "",
                }
            )
        pd.DataFrame(inventory).to_csv(temporary / "plot_inventory.csv", index=False)
        collected["accepted_index"].to_csv(
            temporary / "accepted_case_index.csv", index=False
        )
        collected["packet_index"].to_csv(
            temporary / "included_packets.csv", index=False
        )
        if alias_rows is not None:
            pd.DataFrame(alias_rows).to_csv(temporary / "case_aliases.csv", index=False)
        provenance = {
            "schema_id": SCHEMA_ID,
            "experiment_path": experiment.relative_to(root).as_posix(),
            "settings_hash": collected["document"].get("settings_hash"),
            "calculation": collected["settings"].get("calculation"),
            "precision": collected["settings"].get("precision"),
            "packet_count": len(collected["packet_index"]),
            "accepted_case_occurrence_count": collected["accepted_count"],
            "excluded_rejected_case_occurrence_count": collected["rejected_count"],
            "generated_plot_count": len(generated),
            "generated_plots": sorted(generated),
            "plot_scope": "current completed experiment only",
            "case_policy": "accepted ledger cases plus direct BSk24; rejected cases excluded",
            "flat_plot_folder": True,
            "solver_calls": 0,
            "authoritative_packets_modified": False,
            "friendly_eos_labels": alias_rows is not None,
            "eos_data_path": (
                Path(eos_data_path).resolve().relative_to(root).as_posix()
                if eos_data_path is not None else None
            ),
            "builder_sha256": sha256(Path(__file__)),
        }
        (temporary / "plot_generation_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# Combined accepted experiment plots\n\n"
            "Every PNG in this flat folder combines accepted/valid EoS data from "
            "the single completed experiment named in the provenance JSON. Rejected "
            "cases are excluded. Inputs were checksum-verified saved tables; zero "
            "scientific solver calls were made and authoritative packets were not changed.\n\n"
            "When friendly labels are supplied, accepted_case_index.csv and case_aliases.csv "
            "map them to the original case IDs. Small EoS families use short legend labels; "
            "dense plots retain colorbars instead of thousands of unreadable legend entries.\n",
            encoding="utf-8",
        )
        if any(path.is_dir() for path in temporary.iterdir()):
            raise RuntimeError("flat plot-folder contract was violated")
        write_checksums(temporary)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"plot destination already exists: {target}")
        _CATALOGUE.publish_directory(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**provenance, "plots_path": str(target)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--eos-data", type=Path)
    args = parser.parse_args()
    root = _CATALOGUE.trusted_repository_root(args.repository_root)
    result = build_experiment_plots(
        root,
        args.experiment,
        args.destination,
        eos_data_path=args.eos_data,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

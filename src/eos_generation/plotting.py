"""Lazy figures from validated saved tables; scientific data stays sealed."""

from __future__ import annotations

import os
import tempfile
import json
import subprocess
import sys
from uuid import uuid4
from pathlib import Path


FIGURES = {
    "pressure": (
        "eos",
        "epsilon_mev_fm3",
        "pressure_mev_fm3",
        "Energy density (MeV fm⁻³)",
        "Pressure (MeV fm⁻³)",
    ),
    "cs2": (
        "eos",
        "epsilon_mev_fm3",
        "cs2",
        "Energy density (MeV fm⁻³)",
        "Sound speed squared",
    ),
    "density": (
        "eos",
        "epsilon_mev_fm3",
        "baryon_density_fm3",
        "Energy density (MeV fm⁻³)",
        "Effective baryon density (fm⁻³)",
    ),
    "chemical_potential": (
        "eos",
        "epsilon_mev_fm3",
        "effective_baryon_enthalpy_mev",
        "Energy density (MeV fm⁻³)",
        "Effective chemical potential (MeV)",
    ),
    "mr": ("stars", "Radius", "Mass", "Radius (km)", "Gravitational mass (M☉)"),
    "lambda": (
        "stars",
        "Mass",
        "Lambda",
        "Gravitational mass (M☉)",
        "Tidal deformability",
    ),
    "k2": ("stars", "Mass", "k2", "Gravitational mass (M☉)", "Love number k₂"),
}


FIGURES.update(
    {
        "fixed_radius": (
            "fixed_mass",
            "amplitude",
            "radius_km",
            "Deformation amplitude A",
            "Radius at fixed mass (km)",
        ),
        "fixed_lambda": (
            "fixed_mass",
            "amplitude",
            "lambda_dimensionless",
            "Deformation amplitude A",
            "Tidal deformability at fixed mass",
        ),
        "fixed_k2": (
            "fixed_mass",
            "amplitude",
            "k2",
            "Deformation amplitude A",
            "Love number at fixed mass",
        ),
        "maximum_mass": (
            "maximum_mass",
            "amplitude",
            "maximum_mass_msun",
            "Deformation amplitude A",
            "Resolved maximum mass (M☉)",
        ),
        "radial_pressure": (
            "radial_profiles",
            "radius_over_R",
            "pressure_mev_fm3",
            "Radius / stellar radius",
            "Pressure (MeV fm⁻³)",
        ),
        "radial_density": (
            "radial_profiles",
            "radius_over_R",
            "baryon_density_fm3",
            "Radius / stellar radius",
            "Effective baryon density (fm⁻³)",
        ),
        "radial_cs2": (
            "radial_profiles",
            "radius_over_R",
            "cs2",
            "Radius / stellar radius",
            "Sound speed squared",
        ),
        "radial_mass": (
            "radial_profiles",
            "radius_over_R",
            "enclosed_mass_over_M",
            "Radius / stellar radius",
            "Enclosed mass / stellar mass",
        ),
        "baryonic_response": (
            "baryonic_response_across_mass",
            "mass_msun",
            "delta_baryonic_mass_msun",
            "Gravitational mass (M☉)",
            "Change in baryonic mass (M☉)",
        ),
        "binding_response": (
            "baryonic_response_across_mass",
            "mass_msun",
            "delta_binding_energy_erg",
            "Gravitational mass (M☉)",
            "Change in binding energy (erg)",
        ),
        "radius_response": (
            "stellar_response_across_mass",
            "mass_msun",
            "delta_radius_km",
            "Gravitational mass (M☉)",
            "Change in radius (km)",
        ),
        "tidal_response": (
            "stellar_response_across_mass",
            "mass_msun",
            "delta_lambda",
            "Gravitational mass (M☉)",
            "Change in tidal deformability",
        ),
        "love_response": (
            "stellar_response_across_mass",
            "mass_msun",
            "delta_k2",
            "Gravitational mass (M☉)",
            "Change in Love number",
        ),
        "support_fraction": (
            "deformation_support_fractions",
            "amplitude",
            "radial_span_fraction",
            "Deformation amplitude A",
            "Deformation radial support / stellar radius",
        ),
    }
)

LABELS = {
    "pressure": "Pressure",
    "cs2": "Sound speed squared",
    "density": "Baryon density",
    "chemical_potential": "Chemical potential",
    "mr": "Mass–radius",
    "lambda": "Tidal deformability",
    "k2": "Love number",
    "fixed_radius": "Fixed-mass radius",
    "fixed_lambda": "Fixed-mass tidal deformability",
    "fixed_k2": "Fixed-mass Love number",
    "maximum_mass": "Maximum mass",
    "radial_pressure": "Radial pressure",
    "radial_density": "Radial baryon density",
    "radial_cs2": "Radial sound speed",
    "radial_mass": "Enclosed mass profile",
    "baryonic_response": "Baryonic mass response",
    "binding_response": "Binding energy response",
    "radius_response": "Radius response",
    "tidal_response": "Tidal response",
    "love_response": "Love number response",
    "support_fraction": "Deformation support",
}
_RENDER_CHECKED = False


def _names(figures):
    if isinstance(figures, str):
        figures = (figures,)
    aliases = {
        label.casefold().replace("–", "-"): name for name, label in LABELS.items()
    }
    names = []
    for item in figures:
        name = aliases.get(str(item).casefold().replace("–", "-"), item)
        if name not in FIGURES:
            raise ValueError(
                f"Unknown figure {item!r}. Use Mass–radius (mr) for mass/radius, or choose from: "
                + ", ".join(LABELS.values())
            )
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("Select at least one figure or use 'auto'")
    return names


def _selector(values, name, declared):
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a number or a nonempty list of numbers")
    try:
        items = list(values)
    except TypeError:
        items = [values]
    if not items or any(item not in declared for item in items):
        raise ValueError(f"{name} must use saved values from {sorted(declared)}")
    return items


def _prepare(
    result, *, geometry=None, amplitudes=None, fixed_masses=None, stage="final"
):
    import numpy as np
    import pandas as pd
    from .stellar import classify_saved_tidal_rows

    cases = result.case_table
    geometry = _selector(geometry, "geometry", set(cases.geometry_index))
    amplitudes = _selector(amplitudes, "amplitudes", set(cases.amplitude))
    fixed_masses = _selector(
        fixed_masses, "fixed_masses", set(result.settings.fixed_masses)
    )
    if geometry is not None:
        cases = cases.loc[cases.geometry_index.isin(geometry)]
    if amplitudes is not None:
        cases = cases.loc[cases.amplitude.isin([0.0, *amplitudes])]
    if cases.empty:
        raise ValueError("No cases match the requested geometry/amplitude combination")
    accepted = cases.loc[cases.status.eq("accepted")]
    declarations = accepted.drop_duplicates("physical_case_id").set_index(
        "physical_case_id"
    )
    tables = {}
    for table in {spec[0] for spec in FIGURES.values()}:
        path = result.data_path / (table + ".csv")
        tables[table] = (
            pd.read_csv(path, float_precision="round_trip")
            if path.is_file()
            else pd.DataFrame()
        )
    stellar_stages = list(
        dict.fromkeys(tables["stars"].get("stage", pd.Series(dtype=str)))
    )
    governed = result.metadata["numerical_profile"].get("tov_stages", [])
    final = (
        governed[-1]["name"]
        if governed
        else (stellar_stages[-1] if stellar_stages else None)
    )
    if stage != "final" and stage not in stellar_stages:
        raise ValueError(
            f"stage must be 'final' or a saved stage from {stellar_stages}"
        )
    selected_stage = final if stage == "final" else stage
    prepared, report = {}, []
    for name, (table, x, y, *_labels) in FIGURES.items():
        frame = tables[table].copy()
        required = {y} if x == "amplitude" else {x, y}
        if not required.issubset(frame) or "case_id" not in frame:
            report.append(
                {
                    "name": name,
                    "label": LABELS[name],
                    "available": False,
                    "valid_rows": 0,
                    "omitted_rows": 0,
                    "reason": _missing_plot_reason(result.settings, table),
                }
            )
            continue
        frame = frame.loc[frame.case_id.isin(declarations.index)].copy()
        if "geometry_index" in frame:
            frame = frame.loc[frame.geometry_index.isin(cases.geometry_index.unique())]
        if "stage" in frame:
            frame = frame.loc[frame.stage.eq(selected_stage)].copy()
        if fixed_masses is not None and "target_mass_msun" in frame:
            frame = frame.loc[frame.target_mass_msun.isin(fixed_masses)].copy()
        elif fixed_masses is not None and table == "baryonic_response_across_mass":
            frame = frame.loc[frame.mass_msun.isin(fixed_masses)].copy()
        frame["amplitude"] = frame.case_id.map(declarations.amplitude)
        valid = pd.Series(True, index=frame.index)
        if table == "stars":
            valid &= frame.calculation_status.eq("success")
            if name in {"lambda", "k2"}:
                valid &= classify_saved_tidal_rows(frame, schema="sequence").tidal_valid
        elif table == "fixed_mass":
            valid &= frame.status.eq("bracketed_and_solved")
            if name in {"fixed_lambda", "fixed_k2"}:
                valid &= classify_saved_tidal_rows(
                    frame, schema="fixed_mass"
                ).tidal_valid
        elif table == "maximum_mass":
            valid &= frame.maximum_mass_resolved.eq(True)
        elif name in {"tidal_response", "love_response"}:
            stars = tables["stars"]
            valid_cases = set()
            if not stars.empty:
                saved = stars.loc[
                    stars.stage.eq(final)
                    & stars.calculation_status.eq("success")
                    & stars.get(
                        "is_on_successful_stable_prefix",
                        pd.Series(False, index=stars.index),
                    ).eq(True)
                ]
                tidal = classify_saved_tidal_rows(saved, schema="sequence")
                valid_cases = {
                    case_id
                    for case_id, group in saved.groupby("case_id")
                    if tidal.loc[group.index, "tidal_valid"].all()
                }
                baseline_ids = set(declarations.loc[declarations.amplitude.eq(0)].index)
                if not baseline_ids or not baseline_ids.issubset(valid_cases):
                    valid_cases.clear()
            valid &= frame.case_id.isin(valid_cases)
        finite = np.isfinite(pd.to_numeric(frame[x], errors="coerce")) & np.isfinite(
            pd.to_numeric(frame[y], errors="coerce")
        )
        valid &= finite
        if (
            table not in {"eos", "stars", "fixed_mass", "maximum_mass"}
            and selected_stage != final
        ):
            valid &= False
        frame.loc[~valid, y] = np.nan
        prepared[name] = frame
        count = int(valid.sum())
        report.append(
            {
                "name": name,
                "label": LABELS[name],
                "available": count > 0,
                "valid_rows": count,
                "omitted_rows": int((~valid).sum()),
                "reason": (
                    f"{int((~valid).sum())} unavailable values omitted; inspect {table}.csv and its saved statuses"
                    if count and (~valid).any()
                    else (
                        ""
                        if count
                        else _unavailable_plot_reason(
                            name, table, frame, selected_stage, final
                        )
                    )
                ),
            }
        )
    selection = {
        "geometry": geometry,
        "amplitudes": amplitudes,
        "fixed_masses": fixed_masses,
        "stage": stage,
    }
    return cases, prepared, report, selection


def _missing_plot_reason(settings, table):
    if table in {"stars", "fixed_mass", "maximum_mass"}:
        product = "sequence" if table == "stars" else table
        if product not in settings.requested_observables:
            return f"Not requested in this saved run. For a new run choose calculation='stellar' and include '{product}' in observables."
    if table not in {"eos", "stars", "fixed_mass", "maximum_mass"}:
        if settings.diagnostics != "on":
            return "Diagnostics were off. For a new run set diagnostics='on' with stellar sequence and fixed_mass products."
        return "This diagnostic was not saved. It needs the governed selected cases and successful common fixed-mass/mass support; inspect the saved diagnostic scope."
    return "Required quantities were not saved in this table; inspect the saved product/capability reports."


def _unavailable_plot_reason(name, table, frame, selected_stage, final):
    if (
        table not in {"eos", "stars", "fixed_mass", "maximum_mass"}
        and selected_stage != final
    ):
        return f"Diagnostics refer to the retained final stellar stage ({final}); choose STAGE='final'."
    if frame.empty:
        return "No saved rows match these filters. Choose from the displayed saved geometry, amplitude, target-mass and stage values."
    if table == "maximum_mass":
        return "No resolved maximum for this selection. Inspect maximum_mass.csv for turning-point or retained-endpoint limitations; a sampled peak is not a resolved maximum."
    if table == "fixed_mass" and not frame.status.eq("bracketed_and_solved").any():
        return "No selected target mass was bracketed and solved on the successful stable prefix. Inspect fixed_mass.csv for the exact root status."
    if name in {
        "lambda",
        "k2",
        "fixed_lambda",
        "fixed_k2",
        "tidal_response",
        "love_response",
    }:
        return "Validated tidal values are unavailable for this selection; inspect the saved background/tidal capability statuses and failure reasons."
    if table == "stars":
        return "No successful saved stellar values for this selection; inspect stars.csv calculation_status and failure_reason."
    return "No valid saved values for this selection; inspect case and diagnostic status/support tables."


def available_figures(path, **selection):
    """Explain saved plot availability without importing a renderer or solving."""
    from .experiment import load_experiment

    cases, _frames, report, filters = _prepare(load_experiment(path), **selection)
    return {
        "plots": report,
        "selection": filters,
        "case_status_counts": cases.status.value_counts().to_dict(),
    }


def _check_renderer():
    """Probe native libraries in a child process before risking the kernel."""
    global _RENDER_CHECKED
    if _RENDER_CHECKED:
        return
    command = "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as p; f,a=p.subplots(); a.plot([0,1],[0,1]); f.canvas.draw(); p.close(f)"
    try:
        probe = subprocess.run(
            [sys.executable, "-c", command], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "Plot renderer could not start. Select a correctly activated Python kernel and restart it."
        ) from exc
    if probe.returncode:
        raise RuntimeError(
            "Plot renderer failed its isolated check; no figures were written. On Windows select the activated Deformation_EoS kernel, or launch Jupyter from 'conda activate eos-generation'. "
            + probe.stderr[-600:]
        )
    _RENDER_CHECKED = True


def _cached(directory, request, manifest, plotter):
    from .storage import strict_json, sha256

    try:
        record = strict_json(directory / "figures.json")
        if (
            record.get("status") != "complete"
            or record.get("request") != request
            or record.get("run_manifest_sha256") != manifest
            or record.get("plotter_source_sha256") != plotter
        ):
            return False
        images = record["figures"]
        return bool(images) and all(
            Path(name).name == name
            and not (directory / name).is_symlink()
            and sha256(directory / name) == digest
            for name, digest in images.items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _curve_groups(frame, *, response):
    columns = [
        column
        for column in ("target_mass_msun", "threshold_label", "legacy_execution_role")
        if column in frame
    ]
    if not response:
        columns.insert(0, "case_id")
    if not columns:
        yield (), frame
    else:
        yield from frame.groupby(columns, sort=False, dropna=False)


def _view_options(lambda_scale="log", lambda_mass_limits=None, lambda_limits=None):
    """Normalize presentation-only controls before any figure writes/execution."""
    import math
    from numbers import Real

    if lambda_scale not in ("linear", "log"):
        raise ValueError("lambda_scale must be 'linear' or 'log'")
    limits = {}
    for name, value in (
        ("lambda_mass_limits", lambda_mass_limits),
        ("lambda_limits", lambda_limits),
    ):
        if value is None:
            limits[name] = None
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be None or two increasing finite numbers")
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in value):
            raise ValueError(f"{name} must contain finite numbers")
        lower, upper = map(float, value)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError(f"{name} must contain two increasing finite numbers")
        if lower < 0 or (
            name == "lambda_limits" and lambda_scale == "log" and lower == 0
        ):
            raise ValueError(
                f"{name} lower bound must be nonnegative, and positive for log Λ"
            )
        limits[name] = [lower, upper]
    return {"lambda_scale": lambda_scale, **limits}


def _geometry_colors(geometry_ids):
    """Bind colors to the full saved geometry inventory, before display filtering."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    ids = sorted(set(geometry_ids))
    palette = "tab10" if len(ids) <= 10 else "tab20" if len(ids) <= 20 else "turbo"
    cmap = plt.get_cmap(palette)
    return {
        int(geometry): to_hex(
            cmap(index if len(ids) <= 20 else (index + 0.5) / len(ids))
        )
        for index, geometry in enumerate(ids)
    }


def _curve_label(group, *, response):
    labels = [] if response else [f"A={group.amplitude.iloc[0]:g}"]
    if "target_mass_msun" in group:
        labels.append(f"M={group.target_mass_msun.iloc[0]:g} M☉")
    for column in ("threshold_label", "legacy_execution_role"):
        if column in group:
            labels.append(str(group[column].iloc[0]))
    return "; ".join(labels)


def _render(
    name, frame, cases, destination, *, matter_model="bsk24", geometry_colors=None,
    lambda_scale="log", lambda_mass_limits=None, lambda_limits=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D

    _table, x, y, xlabel, ylabel = FIGURES[name]
    response = x == "amplitude"
    colors = geometry_colors or _geometry_colors(cases.geometry_index)
    curves = []
    seen_baselines = set()
    for geometry, declarations in cases.groupby("geometry_index", sort=True):
        ids = set(declarations.physical_case_id)
        data = frame.loc[frame.case_id.isin(ids)].copy()
        if "geometry_index" in data:
            data = data.loc[data.geometry_index.eq(geometry)]
        if data.empty or not np.isfinite(data[y]).any():
            continue
        for key, group in _curve_groups(data, response=response):
            order = "attempted_index" if "attempted_index" in group else x
            group = group.sort_values(order)
            baseline = not response and group.amplitude.iloc[0] == 0
            if baseline:
                identity = (group.case_id.iloc[0], _curve_label(group, response=True))
                if identity in seen_baselines:
                    continue
                seen_baselines.add(identity)
            curves.append(
                (geometry, group, baseline, _curve_label(group, response=response))
            )
    if not curves:
        return []
    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    variants = list(
        dict.fromkeys(label for _, _, baseline, label in curves if not baseline)
    )
    styles = {
        label: ("-", "--", "-.", ":")[i % 4] for i, label in enumerate(variants)
    }
    plotted_geometries, baseline_handles = set(), {}
    seen_zero_points = set()
    try:
        for geometry, group, baseline, detail in curves:
            color = "black" if baseline else colors[int(geometry)]
            label = f"BSk{matter_model[-2:]} (A=0)" if baseline else f"G{geometry}; {detail}"
            if baseline:
                extra = _curve_label(group, response=True)
                label += ("; " + extra) if extra else ""
            values = group[y].to_numpy(float, copy=True)
            if name == "lambda" and lambda_scale == "log":
                # A log view cannot display nonpositive values; retain a gap,
                # never clamp/repair a saved quantity or alter its status.
                values[values <= 0] = np.nan
            axis.plot(
                group[x], values,
                linewidth=1.8 if baseline else 1.2,
                marker="o" if response else None,
                markersize=3, label=label, color=color,
                linestyle="-" if baseline else styles[detail],
                zorder=3 if baseline else 2,
            )
            if baseline:
                baseline_handles[label] = Line2D([], [], color="black", label=label)
            else:
                plotted_geometries.add(int(geometry))
            if response:
                zero = group.loc[group.amplitude.eq(0)]
                for _, row in zero.iterrows():
                    identity = (row.case_id, detail)
                    if identity not in seen_zero_points and np.isfinite(row[y]):
                        axis.plot(
                            [0], [row[y]], color="black", marker="o",
                            linestyle="none", markersize=4, zorder=4,
                        )
                        seen_zero_points.add(identity)
                if seen_zero_points:
                    label = f"BSk{matter_model[-2:]} (A=0)"
                    baseline_handles[label] = Line2D(
                        [], [], color="black", marker="o", linestyle="none", label=label
                    )
        definitions = cases.drop_duplicates("geometry_index").set_index("geometry_index")
        geometry_ids = sorted(plotted_geometries)
        geometry_handles = []
        for geometry in geometry_ids:
            d = definitions.loc[geometry]
            geometry_handles.append(
                Line2D(
                    [], [], color=colors[geometry],
                    label=f"G{geometry}: ε₀={d.epsilon0_mev_fm3:g}, "
                    f"σ={d.sigma_mev_fm3:g}, Δ={d.delta_mev_fm3:g}",
                )
            )
        title = LABELS[name]
        if cases.geometry_index.nunique() == 1:
            d = cases.iloc[0]
            title += f" — G{d.geometry_index}: ε₀={d.epsilon0_mev_fm3:g}, σ={d.sigma_mev_fm3:g}, Δ={d.delta_mev_fm3:g} MeV fm⁻³"
        else:
            title += f" — {cases.geometry_index.nunique()} geometries (ε₀, σ, Δ in MeV fm⁻³)"
        axis.set(
            xlabel=xlabel, ylabel=ylabel, title=title,
        )
        if name == "lambda":
            axis.set_yscale(
                lambda_scale,
                **({"nonpositive": "mask"} if lambda_scale == "log" else {}),
            )
            if lambda_mass_limits is not None:
                axis.set_xlim(*lambda_mass_limits)
            if lambda_limits is not None:
                axis.set_ylim(*lambda_limits)
            if lambda_mass_limits is not None or lambda_limits is not None:
                axis.set_title(
                    title + "\nDisplay limits selected; full values remain in stars.csv",
                    fontsize=10,
                )
        axis.grid(alpha=0.2)
        handles = list(baseline_handles.values())
        if len(geometry_ids) <= 12:
            handles += geometry_handles
        else:
            cmap = ListedColormap([colors[g] for g in geometry_ids])
            norm = BoundaryNorm(
                np.arange(len(geometry_ids) + 1) - 0.5, len(geometry_ids)
            )
            ticks = np.unique(np.linspace(0, len(geometry_ids) - 1, 12).astype(int))
            fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=axis, ticks=ticks, label="Geometry index (full color key in figures.json)",
            ).set_ticklabels([str(geometry_ids[i]) for i in ticks])
        if len(variants) <= 8:
            handles += [
                Line2D(
                    [], [], color="0.45", linestyle=styles[label],
                    label=label or "Amplitude response",
                )
                for label in variants
            ]
        else:
            axis.text(
                0.99, 0.98,
                f"{len(variants)} curve variants; filter amplitudes/masses for detail",
                transform=axis.transAxes, ha="right", va="top", fontsize=7,
            )
        if handles:
            axis.legend(
                handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1),
            )
        filename = name + ".png"
        fd, temporary = tempfile.mkstemp(suffix=".png", dir=destination)
        os.close(fd)
        try:
            fig.savefig(temporary, dpi=180)
            os.replace(temporary, destination / filename)
        finally:
            Path(temporary).unlink(missing_ok=True)
    finally:
        plt.close(fig)
    return [filename]


def plot_experiment(
    path,
    *,
    figures="auto",
    output_path=None,
    regenerate=False,
    geometry=None,
    amplitudes=None,
    fixed_masses=None,
    stage="final",
    lambda_scale="log",
    lambda_mass_limits=None,
    lambda_limits=None,
) -> Path:
    """Reuse matching figures or publish a new figure version from saved tables."""
    from .experiment import load_experiment
    from .storage import MANIFEST, sha256, write_json_atomic

    view = _view_options(lambda_scale, lambda_mass_limits, lambda_limits)
    result = load_experiment(path)
    cases, frames, availability, selection = _prepare(
        result,
        geometry=geometry,
        amplitudes=amplitudes,
        fixed_masses=fixed_masses,
        stage=stage,
    )
    auto = figures == "auto" or figures == ["auto"] or figures == ("auto",)
    names = (
        [item["name"] for item in availability if item["available"]]
        if auto
        else _names(figures)
    )
    available = {item["name"] for item in availability if item["available"]}
    skipped = {
        item["name"]: item["reason"]
        for item in availability
        if item["name"] in names and not item["available"]
    }
    rendered = [name for name in names if name in available]
    if not rendered:
        raise ValueError(
            "No requested plots have valid saved data. " + "; ".join(skipped.values())
        )
    base = Path(output_path or result.experiment_path / "figures").resolve()
    if (
        base == result.experiment_path
        or base == result.data_path
        or result.data_path in base.parents
        or base in result.data_path.parents
    ):
        raise ValueError(
            "figures must be outside the sealed data directory and cannot replace a run directory"
        )
    request = {"names": names, "selection": selection, "view": view}
    manifest, plotter = sha256(result.data_path / MANIFEST), sha256(Path(__file__))
    if not regenerate:
        for candidate in [
            base,
            *sorted(base.parent.glob(base.name + "-*"), reverse=True),
        ]:
            if candidate.is_dir() and _cached(candidate, request, manifest, plotter):
                return candidate
    _check_renderer()
    import matplotlib

    matplotlib.use("Agg")
    colors = _geometry_colors(result.case_table.geometry_index)
    destination = (
        base
        if not base.exists()
        else base.with_name(base.name + "-" + uuid4().hex[:10])
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    record = {
        "status": "running",
        "request": request,
        "run_manifest_sha256": manifest,
        "plotter_source_sha256": plotter,
        "figures": {},
        "skipped": skipped,
        "availability": availability,
        "scientific_solver_calls": 0,
        "geometry_colors": {
            str(g): colors[int(g)] for g in sorted(cases.geometry_index.unique())
        },
    }
    write_json_atomic(record, destination / "figures.json")
    try:
        for name in rendered:
            images = _render(
                name,
                frames[name],
                cases,
                destination,
                matter_model=result.settings.matter_model,
                geometry_colors=colors,
                **view,
            )
            record["figures"].update(
                {filename: sha256(destination / filename) for filename in images}
            )
        record["status"] = "complete"
        write_json_atomic(record, destination / "figures.json")
    except BaseException as exc:
        record.update(status="failed", failure=str(exc))
        write_json_atomic(record, destination / "figures.json")
        raise
    return destination

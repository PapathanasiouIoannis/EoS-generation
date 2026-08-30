"""Extended saved-table diagnostic figures."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from eos_generation._internal.saved_tables import (
    exact_fixed_mass_response_to_direct,
)
from eos_generation.reporting._plotting_data import (
    _contiguous_valid_runs,
    _reindex_saved_mass_grid,
    _saved_case_style_rows,
    _table,
)
from eos_generation.reporting._plotting_style import (
    _finalize,
    _footer,
    _publication_case_styles,
    _style_rows,
)


def _radial_case_role(case_id: str, coordinates: pd.Series) -> str:
    if case_id == "direct":
        return "direct"
    amplitude = pd.to_numeric(
        pd.Series([coordinates.get("amplitude")]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(amplitude) or np.isclose(float(amplitude), 0.0):
        return "a0"
    return "positive_endpoint" if float(amplitude) > 0.0 else "negative_endpoint"


def _radial_case_title(
    case_id: str, coordinates: pd.Series, *, matter_model: str = "bsk24"
) -> str:
    role = _radial_case_role(case_id, coordinates)
    if role == "direct":
        model = "CFL" if matter_model == "cfl" else "BSk24"
        return f"Direct {model} baseline radial structure"
    amplitude = float(coordinates["amplitude"])
    delta = float(coordinates["delta_mev_fm3"])
    if role == "a0":
        return rf"$A=0$ identity-control radial structure ($\Delta={delta:g}$ MeV fm$^{{-3}}$)"
    sign = "Positive" if amplitude > 0.0 else "Negative"
    return (
        rf"{sign} endpoint radial structure: $A={amplitude:+.4g}$, "
        rf"$\Delta={delta:g}$ MeV fm$^{{-3}}$"
    )


def _radial(packet, config, plt, axis_style) -> bool:
    frame = _table(packet, "radial_profiles.csv")
    if frame.empty:
        return False
    masses = sorted(frame.target_mass_msun.unique())
    coordinate_rows = _saved_case_style_rows(packet, frame).drop_duplicates("case_id")
    coordinate_lookup = {
        str(row.case_id): row
        for _, row in coordinate_rows.iterrows()
    }
    coordinate_lookup.setdefault(
        "direct", pd.Series({"case_id": "direct", "amplitude": np.nan, "delta_mev_fm3": np.nan})
    )
    quantities = (
        ("energy_density_mev_fm3", r"$\varepsilon$ [MeV fm$^{-3}$]"),
        ("pressure_mev_fm3", r"$P$ [MeV fm$^{-3}$]"),
        ("cs2", r"$c_s^2$"),
        ("enclosed_mass_over_M", r"$m(r)/M$"),
    )
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from eos_generation.reporting.plot_helpers import finalize_figure

    mass_min, mass_max = float(min(masses)), float(max(masses))
    if mass_min == mass_max:
        padding = max(abs(mass_min), 1.0) * np.finfo(float).eps
        norm = Normalize(vmin=mass_min - padding, vmax=mass_max + padding)
    else:
        norm = Normalize(vmin=mass_min, vmax=mass_max)
    palette = colormaps["viridis"]

    case_ids = [str(value) for value in frame.case_id.dropna().unique()]
    nonzero_cases = []
    for case_id in case_ids:
        coordinates = coordinate_lookup.get(case_id, pd.Series(dtype=object))
        amplitude = pd.to_numeric(
            pd.Series([coordinates.get("amplitude")]), errors="coerce"
        ).iloc[0]
        if np.isfinite(amplitude) and not np.isclose(float(amplitude), 0.0):
            nonzero_cases.append((case_id, float(amplitude)))
    primary_case = (
        max(nonzero_cases, key=lambda item: (abs(item[1]), item[1] > 0.0))[0]
        if nonzero_cases
        else next(
            (
                case_id
                for case_id in case_ids
                if _radial_case_role(case_id, coordinate_lookup[case_id]) == "a0"
            ),
            case_ids[0],
        )
    )

    generated = 0
    used_roles: dict[str, int] = {}
    for case_id in case_ids:
        coordinates = coordinate_lookup.get(case_id, pd.Series(dtype=object))
        role = _radial_case_role(case_id, coordinates)
        used_roles[role] = used_roles.get(role, 0) + 1
        suffix = role if used_roles[role] == 1 else f"{role}_{used_roles[role]}"
        case_frame = frame.loc[frame.case_id.astype(str).eq(case_id)]
        if case_frame.empty:
            continue
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11.8, 8.4),
            squeeze=False,
            layout="constrained",
        )
        for mass in masses:
            rows = case_frame.loc[
                np.isclose(case_frame.target_mass_msun, mass)
            ].sort_values("radius_over_R", kind="stable")
            if rows.empty:
                continue
            color = palette(norm(float(mass)))
            for ax, (quantity, _) in zip(axes.flat, quantities):
                ax.plot(
                    rows.radius_over_R,
                    rows[quantity],
                    color=color,
                    linewidth=1.55,
                    alpha=0.95,
                )
        for ax, (_, ylabel) in zip(axes.flat, quantities):
            axis_style(ax, xlabel=r"$r/R$", ylabel=ylabel)
            ax.margins(x=0.01)
        sound_speed = pd.to_numeric(case_frame["cs2"], errors="coerce")
        if bool(sound_speed.notna().all()) and bool((sound_speed >= 0.0).all()):
            axes.flat[2].set_ylim(bottom=0.0)
        fig.suptitle(
            _radial_case_title(
                case_id,
                coordinates,
                matter_model=getattr(config, "matter_model", "bsk24"),
            ),
            fontsize=13.0,
        )
        mappable = ScalarMappable(norm=norm, cmap=palette)
        mappable.set_array(np.asarray(masses, dtype=float))
        colorbar = fig.colorbar(
            mappable,
            ax=list(axes.flat),
            fraction=0.035,
            pad=0.025,
            shrink=0.94,
        )
        colorbar.set_label(r"Target mass $M/M_\odot$")
        _footer(
            fig,
            config,
            "Each curve is one exact fixed-mass solved profile; colour encodes target mass.",
        )
        companion = packet / "plots" / f"radial_structure_profiles_{suffix}.png"
        rendered_companion = Path(finalize_figure(fig, companion))
        if case_id == primary_case:
            primary = packet / "plots" / "radial_structure_profiles.png"
            shutil.copyfile(rendered_companion, primary)
        generated += 1
    return generated > 0


def _support(packet, config, plt, axis_style) -> bool:
    frame = _table(packet, "deformation_support_fractions.csv")
    if frame.empty:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), layout="constrained")
    styles = _publication_case_styles(packet, frame)
    for (case_id, threshold), rows in frame.groupby(
        ["case_id", "threshold_fraction"]
    ):
        style = dict(styles.get(str(case_id), {"label": str(case_id)}))
        threshold_style = {
            0.01: {"linestyle": ":", "marker": "^"},
            0.10: {"linestyle": "--", "marker": "s"},
            0.50: {"linestyle": "-", "marker": "o"},
        }.get(float(threshold), {"marker": "o"})
        style.update(
            threshold_style,
            label=f"{style['label']}, {threshold:.0%} peak",
        )
        rows = _reindex_saved_mass_grid(rows, config.fixed_masses_msun)
        reached = rows["status"].astype(str).eq("reached")
        for axis, field in zip(
            axes,
            ("radial_span_fraction", "enclosed_mass_span_fraction"),
            strict=True,
        ):
            valid = reached & np.isfinite(
                pd.to_numeric(rows[field], errors="coerce")
            )
            for index, run in enumerate(_contiguous_valid_runs(rows, valid)):
                axis.plot(
                    run.target_mass_msun,
                    run[field],
                    **{
                        **style,
                        "label": style["label"] if index == 0 else None,
                    },
                )
    axis_style(axes[0], xlabel=r"$M/M_\odot$", ylabel="support radial span / R")
    axis_style(axes[1], xlabel=r"$M/M_\odot$", ylabel="support enclosed-mass span / M")
    axes[1].legend(fontsize=6.5)
    fig.suptitle("Radial support of the sound-speed deformation", fontsize=12.5)
    _footer(
        fig,
        config,
        "Thresholds use the global realized |delta cs2| peak; 50% is the FWHM "
        "diagnostic. Unavailable masses remain visible as line gaps.",
    )
    _finalize(fig, packet / "plots/deformation_support_fractions.png")
    return True


def _baryonic(packet, config, plt, axis_style, *, binding: bool) -> bool:
    frame = _table(packet, "baryonic_observables.csv")
    if frame.empty or frame.target_mass_msun.nunique() < 2:
        return False
    styles = _publication_case_styles(packet, frame)
    if binding:
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), layout="constrained")
        for case_id, rows in frame.groupby("case_id"):
            rows = _reindex_saved_mass_grid(rows, config.fixed_masses_msun)
            style = dict(styles.get(str(case_id), {"label": str(case_id)}))
            for axis, field in zip(
                axes,
                ("mass_excess_msun", "fractional_binding"),
                strict=True,
            ):
                valid = np.isfinite(
                    pd.to_numeric(rows[field], errors="coerce")
                )
                for index, run in enumerate(
                    _contiguous_valid_runs(rows, valid)
                ):
                    axis.plot(
                        run.target_mass_msun,
                        run[field],
                        **{
                            **style,
                            "label": style["label"] if index == 0 else None,
                        },
                    )
        axis_style(axes[0], xlabel=r"$M/M_\odot$", ylabel=r"$(M_b-M)/M_\odot$")
        axis_style(axes[1], xlabel=r"$M/M_\odot$", ylabel=r"$(M_b-M)/M_b$")
        axes[1].legend(fontsize=6.5)
        fig.suptitle("Gravitational binding at fixed mass", fontsize=12.5)
        name = "binding_energy_vs_mass.png"
    else:
        fig, ax = plt.subplots(figsize=(7.0, 5.0), layout="constrained")
        for case_id, rows in frame.groupby("case_id"):
            rows = _reindex_saved_mass_grid(rows, config.fixed_masses_msun)
            style = dict(styles.get(str(case_id), {"label": str(case_id)}))
            valid = np.isfinite(
                pd.to_numeric(rows["baryonic_mass_msun"], errors="coerce")
            )
            for index, run in enumerate(_contiguous_valid_runs(rows, valid)):
                ax.plot(
                    run.target_mass_msun,
                    run.baryonic_mass_msun,
                    **{
                        **style,
                        "label": style["label"] if index == 0 else None,
                    },
                )
        axis_style(ax, xlabel=r"$M/M_\odot$", ylabel=r"$M_b/M_\odot$")
        ax.legend(fontsize=6.5)
        ax.set_title("Baryonic mass at fixed gravitational mass")
        name = "baryonic_mass_vs_mass.png"
    _footer(
        fig,
        config,
        "Absolute M_b and binding quantities use M_b=m_n N_B (neutron-rest-mass "
        "convention). Unavailable masses remain visible as line gaps.",
    )
    _finalize(fig, packet / "plots" / name)
    return True


def _response(packet, config, plt, axis_style, *, baryonic: bool) -> bool:
    if baryonic:
        frame = _table(packet, "baryonic_response_across_mass.csv")
        if frame.empty:
            return False
        fields = (
            ("delta_baryonic_mass_msun", r"$\Delta M_b/M_\odot$"),
            ("delta_binding_energy_erg", r"$\Delta E_{\rm bind}$ [erg]"),
        )
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
        filename = "baryonic_response_across_mass.png"
        styles = _publication_case_styles(packet, frame)
    else:
        fixed = _table(packet, "fixed_mass_observables.csv")
        radial = _table(packet, "radial_profiles.csv")
        if fixed.empty or radial.empty or not config.tov_stages:
            return False
        frame = exact_fixed_mass_response_to_direct(
            fixed,
            final_stage=config.tov_stages[-1].name,
            case_ids=tuple(str(value) for value in radial.case_id.unique()),
        )
        amplitude = pd.to_numeric(frame.get("amplitude"), errors="coerce")
        frame = frame.loc[
            np.isfinite(amplitude)
            & ~np.isclose(amplitude, 0.0, equal_nan=False)
        ].copy()
        if frame.empty:
            return False
        fields = (
            ("fractional_delta_radius", r"$\Delta R/R$ [\%]"),
            ("fractional_delta_k2", r"$\Delta k_2/k_2$ [\%]"),
            ("fractional_delta_lambda", r"$\Delta\Lambda/\Lambda$ [\%]"),
        )
        fig, axes = plt.subplots(
            1, 3, figsize=(14.0, 4.5), layout="constrained"
        )
        filename = "stellar_response_across_mass.png"
        styles = _style_rows(_saved_case_style_rows(packet, frame))
    for ax, (field, ylabel) in zip(np.asarray(axes).flat, fields):
        ax.axhline(0.0, color="#222222", lw=0.8, zorder=0.5)
        for case_id, rows in frame.groupby("case_id"):
            configured_masses = getattr(
                config,
                "fixed_masses_msun",
                tuple(
                    sorted(
                        pd.to_numeric(
                            frame["mass_msun"], errors="coerce"
                        ).dropna().unique()
                    )
                ),
            )
            rows = _reindex_saved_mass_grid(
                rows,
                configured_masses,
                mass_column="mass_msun",
            )
            style = dict(styles.get(str(case_id), {"label": str(case_id)}))
            if baryonic:
                values = pd.to_numeric(rows[field], errors="coerce")
            else:
                values = 100.0 * pd.to_numeric(rows[field], errors="coerce")
                amplitude_rows = pd.to_numeric(
                    rows["amplitude"], errors="coerce"
                ).dropna()
                if amplitude_rows.empty:
                    continue
                amplitude_value = float(amplitude_rows.iloc[0])
                style.update(
                    color="#b91c1c" if amplitude_value > 0.0 else "#1d4ed8",
                    linewidth=2.0,
                    marker="o",
                    markersize=4.8,
                )
            valid = np.isfinite(values)
            plot_rows = rows.copy()
            plot_rows["_response_value"] = values
            for index, run in enumerate(
                _contiguous_valid_runs(plot_rows, valid)
            ):
                ax.plot(
                    run.mass_msun,
                    run["_response_value"],
                    **{
                        **style,
                        "label": style["label"] if index == 0 else None,
                    },
                )
        axis_style(ax, xlabel=r"$M/M_\odot$", ylabel=ylabel)
    np.asarray(axes).flat[-1].legend(fontsize=6.5)
    if not baryonic:
        fig.suptitle("Exact fixed-mass stellar response", fontsize=13.0)
    else:
        model = (
            "CFL"
            if getattr(config, "matter_model", "bsk24") == "cfl"
            else "BSk24"
        )
        fig.suptitle(
            f"Baryonic response relative to direct {model}", fontsize=12.5
        )
    _footer(
        fig,
        config,
        (
            "Exact differences of independently bracketed fixed-mass solves; "
            "markers are computed points and connecting lines are visual guides; "
            "unavailable masses remain line gaps."
            if not baryonic
            else "Interpolation is restricted to common successful stable support; "
            "unavailable masses remain line gaps."
        )
        + (
            " Absolute baryonic quantities use the neutron-rest-mass convention."
            if baryonic
            else ""
        ),
    )
    _finalize(fig, packet / "plots" / filename)
    return True


def _odd_even(packet, config, plt, axis_style) -> bool:
    frame = _table(packet, "odd_even_response.csv")
    required = {
        "delta_mev_fm3",
        "target_mass_msun",
        "observable",
        "amplitude",
        "odd_response",
        "even_response",
    }
    if frame.empty or not required.issubset(frame.columns):
        return False
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)
    planned_magnitudes = tuple(
        sorted(
            {
                abs(float(value))
                for value in getattr(config, "effective_amplitudes", ())
                if not np.isclose(float(value), 0.0)
            }
        )
    )
    for (delta, mass, observable), rows in frame.groupby(
        ["delta_mev_fm3", "target_mass_msun", "observable"],
        sort=False,
    ):
        rows = rows.copy()
        rows["amplitude"] = pd.to_numeric(rows["amplitude"], errors="coerce")
        if bool(rows.duplicated("amplitude", keep=False).any()):
            raise ValueError("odd-even response rows are not unique by amplitude")
        axis_values = planned_magnitudes or tuple(
            sorted(rows.amplitude.dropna().unique())
        )
        rows = (
            rows.set_index("amplitude")
            .reindex(axis_values)
            .rename_axis("amplitude")
            .reset_index()
        )
        series_label = (
            f"{observable}, M={float(mass):g}, "
            rf"$\Delta={float(delta):g}$"
        )
        for axis, field, marker in (
            (axes[0], "odd_response", "o"),
            (axes[1], "even_response", "s"),
        ):
            valid = np.isfinite(pd.to_numeric(rows[field], errors="coerce"))
            for index, run in enumerate(_contiguous_valid_runs(rows, valid)):
                axis.plot(
                    run.amplitude,
                    run[field],
                    marker=marker,
                    label=series_label if index == 0 else None,
                )
    axis_style(axes[0], ylabel="odd response")
    axis_style(axes[1], xlabel=r"$|A|$", ylabel="even response")
    axes[0].legend(fontsize=7, ncol=3)
    _footer(fig, config, "Unavailable planned amplitudes remain line gaps.")
    _finalize(fig, packet / "plots/odd_even_response.png")
    return True


def _errors(packet, config, plt, axis_style) -> bool:
    frame = _table(packet, "numerical_error_summary.csv")
    if frame.empty:
        return False
    label_parts = frame["label"].astype(str).str.rsplit(":", n=2, expand=True)
    if label_parts.shape[1] != 3:
        raise ValueError("numerical-error labels must be case:mass:observable")
    plot_frame = frame.copy()
    plot_frame["target_mass_msun"] = pd.to_numeric(label_parts[1], errors="coerce")
    scale = np.maximum(
        np.abs(pd.to_numeric(plot_frame.reference_value, errors="coerce")), 1.0
    )
    plot_frame["normalized_envelope"] = (
        pd.to_numeric(plot_frame.numerical_envelope, errors="coerce") / scale
    )
    plot_frame = plot_frame.loc[
        np.isfinite(plot_frame.target_mass_msun)
        & np.isfinite(plot_frame.normalized_envelope)
        & (plot_frame.normalized_envelope > 0.0)
    ]
    if plot_frame.empty:
        return False
    summary = (
        plot_frame.groupby(["observable", "target_mass_msun"], as_index=False)
        .agg(
            median=("normalized_envelope", "median"),
            percentile_95=("normalized_envelope", lambda values: values.quantile(0.95)),
            maximum=("normalized_envelope", "max"),
        )
        .sort_values(["observable", "target_mass_msun"], kind="stable")
    )
    observables = (
        ("radius_km", r"Radius $R$"),
        ("k2", r"Love number $k_2$"),
        ("lambda_dimensionless", r"Tidal deformability $\Lambda$"),
    )
    available = tuple(item for item in observables if item[0] in set(summary.observable))
    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(4.55 * len(available), 4.5),
        squeeze=False,
        layout="constrained",
    )
    for ax, (observable, title) in zip(axes.flat, available):
        rows = summary.loc[summary.observable.eq(observable)]
        rows = _reindex_saved_mass_grid(rows, config.fixed_masses_msun)
        for field, marker, linestyle, label, linewidth in (
            ("median", "o", "-", "median across cases", None),
            ("percentile_95", "s", "--", "95th percentile", None),
            ("maximum", "^", ":", "maximum (worst case)", 1.8),
        ):
            values = pd.to_numeric(rows[field], errors="coerce")
            valid = np.isfinite(values) & (values > 0.0)
            for index, run in enumerate(
                _contiguous_valid_runs(rows, valid)
            ):
                kwargs = {
                    "marker": marker,
                    "linestyle": linestyle,
                    "label": label if index == 0 else None,
                }
                if linewidth is not None:
                    kwargs["linewidth"] = linewidth
                ax.semilogy(
                    run.target_mass_msun,
                    run[field],
                    **kwargs,
                )
        axis_style(
            ax,
            xlabel=r"Target mass $M/M_\odot$",
            ylabel=r"stage envelope / $\max(|O|,1)$",
        )
        ax.set_title(title)
    axes.flat[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Saved-stage numerical envelope across the amplitude family", fontsize=12.5)
    fig._bsk24_numerical_error_summary = summary
    _footer(fig, config, "Unavailable configured masses remain line gaps.")
    _finalize(fig, packet / "plots/numerical_error_summary.png")
    return True


def _unavailable_saved_table_plot(packet, config, plt, axis_style, filename: str) -> bool:
    """Render simple saved-table diagnostics that the core runner does not construct."""
    table_name = {
        "outside_support_control.png": "outside_support_control.csv",
        "turning_point_sequences.png": "turning_point_sequences.csv",
        "turning_point_derivatives.png": "turning_point_sequences.csv",
        "matched_area_comparison.png": "matched_area_comparison.csv",
    }[filename]
    frame = _table(packet, table_name)
    if frame.empty:
        return False
    if filename == "outside_support_control.png":
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
        for ax, field, ylabel in zip(
            axes,
            ("delta_radius_km", "delta_k2", "delta_lambda"),
            (r"$\Delta R$ [km]", r"$\Delta k_2$", r"$\Delta\Lambda$"),
        ):
            for case_id, rows in frame.groupby("case_id"):
                ax.plot(rows.target_mass_msun, rows[field], marker="o", label=case_id)
            axis_style(ax, xlabel=r"$M/M_\odot$", ylabel=ylabel)
        axes[-1].legend(fontsize=7)
    elif filename.startswith("turning_point"):
        derivative = filename == "turning_point_derivatives.png"
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
        for case_id, rows in frame.groupby("case_id"):
            if derivative:
                axes[0].plot(rows.central_pressure_mev_fm3, rows.dM_dPc, label=case_id)
                axes[1].plot(rows.central_energy_density_mev_fm3, rows.dM_dEpsilon_c, label=case_id)
            else:
                axes[0].plot(rows.central_pressure_mev_fm3, rows.mass_msun, label=case_id)
                axes[1].plot(rows.central_energy_density_mev_fm3, rows.mass_msun, label=case_id)
        if derivative:
            for ax in axes:
                ax.axhline(0.0, color="#222222", lw=0.8)
            axis_style(axes[0], xlabel=r"$P_c$", ylabel=r"$dM/dP_c$")
            axis_style(axes[1], xlabel=r"$\varepsilon_c$", ylabel=r"$dM/d\varepsilon_c$")
        else:
            axis_style(axes[0], xlabel=r"$P_c$", ylabel=r"$M/M_\odot$")
            axis_style(axes[1], xlabel=r"$\varepsilon_c$", ylabel=r"$M/M_\odot$")
        axes[-1].legend(fontsize=7)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
        fields = (
            ("radius_km", r"$R_{1.4}$ [km]"),
            ("lambda_dimensionless", r"$\Lambda_{1.4}$"),
            ("baryonic_mass_msun", r"$M_{b,1.4}/M_\odot$"),
            ("binding_energy_erg", r"$E_{\rm bind,1.4}$ [erg]"),
        )
        for ax, (field, ylabel) in zip(axes.flat, fields):
            for keys, rows in frame.groupby(["comparison_type", "sign"]):
                rows = rows.sort_values("delta_mev_fm3")
                ax.plot(rows.delta_mev_fm3, rows[field], marker="o", label=", ".join(keys))
            axis_style(ax, xlabel=r"$\Delta$", ylabel=ylabel)
        axes.flat[0].legend(fontsize=7)
    _footer(fig, config)
    _finalize(fig, packet / "plots" / filename)
    return True

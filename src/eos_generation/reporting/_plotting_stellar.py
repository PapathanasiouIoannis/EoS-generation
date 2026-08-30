"""Stellar and fixed-mass response figures from saved packet tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eos_generation._internal.saved_tables import (
    summarize_fixed_mass_response_population,
)
from eos_generation.reporting._plotting_data import (
    _contiguous_valid_runs,
    _saved_case_style_rows,
    _successful_sequence_runs,
    _table,
    _validated_fixed_tidal_mask,
    _validated_tidal_runs,
)
from eos_generation.reporting._plotting_data import (
    _summarize_stellar_tidal_plot_completeness as summarize_stellar_tidal_plot_completeness,
)
from eos_generation.reporting._plotting_style import (
    _add_amplitude_colorbar,
    _concise_mass_radius_legend_label,
    _draw_mass_radius_compactness_guides,
    _finalize,
    _footer,
    _style_rows,
)
from eos_generation.stellar.tov import LAMBDA_FRAMEWORK_CAPABILITY


def _stellar(packet, config, plt, axis_style) -> bool:
    frame = _table(packet, "stellar_sequences.csv")
    if frame.empty:
        return False
    stage = config.tov_stages[-1].name
    stage_rows = frame.loc[frame.stage == stage]
    selected = stage_rows.loc[stage_rows.calculation_status == "success"]
    if selected.empty:
        return False
    style_rows = _saved_case_style_rows(packet, selected)
    styles = _style_rows(style_rows)
    if "direct" in styles:
        styles["direct"]["label"] = (
            "Direct CFL"
            if getattr(config, "matter_model", "bsk24") == "cfl"
            else "Direct BSk24"
        )
    fixed = _table(packet, "fixed_mass_observables.csv")
    fixed_stage = fixed.loc[fixed.stage == stage]
    fixed = fixed_stage.loc[fixed_stage.status == "bracketed_and_solved"]
    tidal_summary = summarize_stellar_tidal_plot_completeness(stage_rows, fixed_stage)
    has_sequence_tides = bool(
        tidal_summary["sequence_tidal_validated_count"]
    )
    if has_sequence_tides:
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.15))
    else:
        fig, mass_radius_axis = plt.subplots(1, 1, figsize=(6.4, 4.9))
        axes = np.asarray((mass_radius_axis,), dtype=object)
    for case_id, rows in stage_rows.groupby("case_id"):
        if str(case_id) not in styles:
            continue
        background_label_used = False
        for _, segment in rows.groupby("segment_id"):
            segment = segment.sort_values("attempted_index", kind="stable")
            style = dict(styles[str(case_id)])
            for run in _successful_sequence_runs(segment):
                axes[0].plot(
                    run.Radius,
                    run.Mass,
                    **{
                        **style,
                        "label": (
                            style.get("label")
                            if not background_label_used
                            else None
                        ),
                    },
                )
                background_label_used = True
            if has_sequence_tides:
                for run in _validated_tidal_runs(segment):
                    axes[1].plot(run.Mass, run.k2, **{**style, "label": None})
                    axes[2].semilogy(run.Mass, run.Lambda, **{**style, "label": None})
        points = fixed.loc[fixed.case_id == case_id]
        axes[0].scatter(
            points.radius_km,
            points.target_mass_msun,
            color=styles[str(case_id)]["color"],
            s=24,
        )
        tidal_points = points.loc[_validated_fixed_tidal_mask(points)]
        if has_sequence_tides:
            axes[1].scatter(
                tidal_points.target_mass_msun,
                tidal_points.k2,
                color=styles[str(case_id)]["color"],
                s=24,
            )
            axes[2].scatter(
                tidal_points.target_mass_msun,
                tidal_points.lambda_dimensionless,
                color=styles[str(case_id)]["color"],
                s=24,
            )
        peaks = rows.loc[
            rows.calculation_status.astype(str).eq("success")
            & (rows.is_sampled_peak == True)  # noqa: E712
        ]
        axes[0].scatter(
            peaks.Radius,
            peaks.Mass,
            facecolors="none",
            edgecolors=styles[str(case_id)]["color"],
            s=34,
        )
    axes[0].scatter([], [], facecolors="none", edgecolors="#111827", label="sampled peak (not Mmax)")
    compactness_artists = _draw_mass_radius_compactness_guides(
        axes[0], selected, fixed
    )
    handles, labels = axes[0].get_legend_handles_labels()
    display_labels = [_concise_mass_radius_legend_label(label) for label in labels]
    axes[0].legend_.remove() if axes[0].legend_ is not None else None
    fig.subplots_adjust(bottom=0.23, wspace=0.28)
    fig.legend(
        handles,
        display_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        frameon=False,
        fontsize=7.2,
    )
    axis_style(axes[0], xlabel="R [km]", ylabel=r"$M/M_\odot$")
    if has_sequence_tides:
        axis_style(axes[1], xlabel=r"$M/M_\odot$", ylabel=r"$k_2$")
        axis_style(axes[2], xlabel=r"$M/M_\odot$", ylabel=r"$\Lambda$")
    if has_sequence_tides:
        axes[1].set_title(r"$k_2(M)$")
        axes[2].set_title(r"$\Lambda(M)$")
    axes[0].set_title("Mass–radius")
    _footer(
        fig,
        config,
        (
            "M-R uses successful backgrounds. Tidal panels require exact status "
            f"{LAMBDA_FRAMEWORK_CAPABILITY}; omitted "
            f"{tidal_summary['sequence_tidal_omitted_count']} sequence and "
            f"{tidal_summary['fixed_mass_tidal_omitted_count']} fixed-mass rows. "
            "Lines break at every failed attempted-index gap."
        ),
    )
    fig._bsk24_tidal_completeness = tidal_summary
    fig._bsk24_compactness_guides = tuple(compactness_artists)
    _add_amplitude_colorbar(fig, axes, style_rows)
    _finalize(fig, packet / "plots/stellar_mr_k2_lambda.png")
    return True


def _observable_response(packet, config, plt, axis_style, *, versus: str) -> bool:
    frame = _table(packet, "fixed_mass_observables.csv")
    if frame.empty:
        return False
    stage = config.tov_stages[-1].name
    target = min(config.fixed_masses_msun, key=lambda value: abs(value - 1.4))
    plot_frame, population = summarize_fixed_mass_response_population(
        frame,
        final_stage=stage,
        target_mass_msun=target,
        versus=versus,
    )
    if plot_frame.empty:
        return False
    planned = _table(packet, "case_plan.csv")
    planned_coordinates = pd.DataFrame()
    if {"amplitude", "delta_mev_fm3"}.issubset(planned.columns):
        planned_coordinates = planned[["amplitude", "delta_mev_fm3"]].copy()
        for coordinate in ("amplitude", "delta_mev_fm3"):
            planned_coordinates[coordinate] = pd.to_numeric(
                planned_coordinates[coordinate], errors="coerce"
            )
        planned_coordinates = planned_coordinates.loc[
            np.isfinite(planned_coordinates["amplitude"])
            & np.isfinite(planned_coordinates["delta_mev_fm3"])
        ].drop_duplicates()
    observables = (
        ("radius_km", rf"$R_{{{target:g}}}$ [km]"),
        ("k2", rf"$k_{{2,{target:g}}}$"),
        ("lambda_dimensionless", rf"$\Lambda_{{{target:g}}}$"),
        ("central_energy_density_mev_fm3", r"$\varepsilon_c$ [MeV fm$^{-3}$]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    if versus == "amplitude":
        groups = plot_frame.groupby("delta_mev_fm3")
        xcolumn, xlabel = "amplitude", "A"
        filename = "observable_response_vs_amplitude.png"
        title = "Observable response versus amplitude"
    else:
        groups = plot_frame.groupby("amplitude")
        xcolumn, xlabel = "delta_mev_fm3", r"$\Delta$ [MeV fm$^{-3}$]"
        filename = "observable_response_vs_delta.png"
        title = "Observable response versus window width"
    group_column = (
        "delta_mev_fm3" if versus == "amplitude" else "amplitude"
    )
    fallback_axis = tuple(
        float(value)
        for value in (
            getattr(config, "effective_amplitudes", ())
            if versus == "amplitude"
            else getattr(config, "deltas_mev_fm3", ())
        )
    )
    planned_row_count = 0
    for key, _rows in groups:
        if planned_coordinates.empty:
            planned_row_count += len(fallback_axis)
            continue
        matching = np.isclose(
            planned_coordinates[group_column].to_numpy(dtype=float),
            float(key),
            rtol=0.0,
            atol=32.0 * np.finfo(float).eps * max(1.0, abs(float(key))),
        )
        planned_row_count += int(
            planned_coordinates.loc[matching, xcolumn].nunique()
        )
    planned_row_count = max(planned_row_count, int(len(plot_frame)))
    population = dict(population)
    population["saved_deformation_row_count"] = int(len(plot_frame))
    population["planned_deformation_row_count"] = planned_row_count
    population["background_unavailable_count"] = (
        planned_row_count - int(population["background_success_count"])
    )
    population["tidal_omitted_count"] = (
        planned_row_count - int(population["tidal_validated_count"])
    )
    for ax, (observable, ylabel) in zip(axes.flat, observables):
        for key, rows in groups:
            rows = rows.copy()
            rows[xcolumn] = pd.to_numeric(rows[xcolumn], errors="coerce")
            if bool(rows.duplicated(xcolumn, keep=False).any()):
                raise ValueError(
                    f"saved response rows are not unique by {xcolumn}"
                )
            if planned_coordinates.empty:
                axis_values = fallback_axis
            else:
                matching = np.isclose(
                    planned_coordinates[group_column].to_numpy(dtype=float),
                    float(key),
                    rtol=0.0,
                    atol=(
                        32.0
                        * np.finfo(float).eps
                        * max(1.0, abs(float(key)))
                    ),
                )
                axis_values = tuple(
                    sorted(
                        planned_coordinates.loc[matching, xcolumn].unique()
                    )
                )
            if not axis_values:
                axis_values = tuple(sorted(rows[xcolumn].dropna().unique()))
            rows = (
                rows.set_index(xcolumn)
                .reindex(axis_values)
                .rename_axis(xcolumn)
                .reset_index()
            )
            if observable in {"k2", "lambda_dimensionless"}:
                valid = _validated_fixed_tidal_mask(rows)
            else:
                valid = rows["status"].astype(str).eq(
                    "bracketed_and_solved"
                )
            valid &= np.isfinite(
                pd.to_numeric(rows[observable], errors="coerce")
            )
            runs = _contiguous_valid_runs(rows, valid)
            for index, run in enumerate(runs):
                ax.plot(
                    run[xcolumn],
                    run[observable],
                    marker="o",
                    label=f"{key:g}" if index == 0 else None,
                )
        axis_style(ax, xlabel=xlabel, ylabel=ylabel)
        if observable in {"k2", "lambda_dimensionless"}:
            ax.set_title(
                f"validated tides {population['tidal_validated_count']}/"
                f"{population['planned_deformation_row_count']}; "
                f"omitted {population['tidal_omitted_count']}",
                fontsize=8.0,
            )
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(title)
    _footer(
        fig,
        config,
        (
            f"Values use the saved {stage} stage at {target:g} solar masses. "
            f"Tidal panels require exact status {LAMBDA_FRAMEWORK_CAPABILITY}; "
            "failed or unavailable rows are omitted without zero filling or gap bridging."
        ),
    )
    fig._bsk24_response_population = population
    _finalize(fig, packet / "plots" / filename)
    return True

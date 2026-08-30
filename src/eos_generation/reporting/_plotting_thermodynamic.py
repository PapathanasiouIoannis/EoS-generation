"""Thermodynamic figures rendered exclusively from saved packet tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eos_generation.reporting._plotting_data import (
    _effective_plot_epsilon_match,
    _table,
)
from eos_generation.reporting._plotting_style import (
    _add_amplitude_colorbar,
    _dense_panel_note,
    _draw_gamma_reference,
    _draw_pressure_energy_guides,
    _draw_sound_speed_guides,
    _finalize,
    _footer,
    _scales,
    _style_rows,
    _uses_amplitude_colorbar,
)
from eos_generation.reporting.plot_layout import bounded_representative_values


def _window_profiles(packet, config, plt, axis_style) -> bool:
    frame = _table(packet, "raw_gate_profiles.csv")
    if frame.empty:
        return False
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    all_deltas = sorted(frame.delta_mev_fm3.dropna().unique())
    displayed_deltas, omitted = bounded_representative_values(
        all_deltas,
        preferred=(config.diagnostic_delta_mev_fm3,),
    )
    selected = (
        frame.loc[frame.delta_mev_fm3.isin(displayed_deltas)]
        .sort_values("amplitude")
        .drop_duplicates("delta_mev_fm3")
    )
    for delta, row in selected.groupby("delta_mev_fm3"):
        case = frame.loc[frame.case_id == row.case_id.iloc[0]]
        ax.plot(case.epsilon_mev_fm3, case.window, lw=2.0, label=rf"$\Delta={delta:g}$")
        ax.axvline(
            _effective_plot_epsilon_match(config) + float(delta),
            color="#9ca3af",
            ls=":",
            lw=0.7,
        )
    axis_style(
        ax,
        xlabel=r"$\varepsilon$ [MeV fm$^{-3}$]",
        ylabel=r"$W(\varepsilon)$",
        title="Quintic smootherstep windows",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    _footer(
        fig,
        config,
        _dense_panel_note(displayed_deltas, omitted, len(all_deltas)),
    )
    _finalize(fig, packet / "plots/window_profiles.png")
    return True


def _gaussian_realization(packet, config, plt, axis_style) -> bool:
    frame = _table(packet, "raw_gate_profiles.csv")
    all_deltas = sorted(frame.delta_mev_fm3.dropna().unique())
    deltas, omitted = bounded_representative_values(
        all_deltas,
        preferred=(config.diagnostic_delta_mev_fm3,),
    )
    if frame.empty or not deltas:
        return False
    fig, axes = plt.subplots(1, len(deltas), figsize=(5.0 * len(deltas), 4.6), squeeze=False)
    for ax, delta in zip(axes.flat, deltas):
        case_id = (
            frame.loc[np.isclose(frame.delta_mev_fm3, delta)]
            .sort_values("amplitude")
            .case_id.iloc[-1]
        )
        rows = frame.loc[frame.case_id == case_id]
        ax.plot(rows.epsilon_mev_fm3, rows.gaussian, color="#111827", label="G")
        ax.plot(
            rows.epsilon_mev_fm3,
            rows.gaussian * rows.window,
            color="#b91c1c",
            lw=2.0,
            label="realized G W",
        )
        _scales(ax, config, float(delta))
        axis_style(
            ax,
            xlabel=r"$\varepsilon$ [MeV fm$^{-3}$]",
            ylabel="unit deformation shape",
            title=rf"$\Delta={delta:g}$",
        )
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Nominal Gaussian and realized smootherstep-windowed shape")
    _footer(fig, config, _dense_panel_note(deltas, omitted, len(all_deltas)))
    _finalize(fig, packet / "plots/gaussian_realization.png")
    return True


def _raw_sound_speed(packet, config, plt, axis_style, *, zoom: bool) -> bool:
    raw = _table(packet, "raw_gate_profiles.csv")
    profile = _table(packet, "thermodynamic_profiles.csv")
    accepted = raw.loc[raw.gate_status == "accepted_raw_local_physics_gate"]
    if accepted.empty:
        return False
    styles = _style_rows(pd.concat((profile, accepted), ignore_index=True, sort=False))
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    direct = profile.loc[profile.case_id == "direct"]
    if not direct.empty:
        ax.plot(direct.epsilon_mev_fm3, direct.cs2, **styles["direct"])
    for case_id, rows in accepted.groupby("case_id"):
        ax.plot(rows.epsilon_mev_fm3, rows.raw_cs2, **styles[str(case_id)])
    _scales(ax, config, config.diagnostic_delta_mev_fm3)
    if zoom:
        epsilon_match = _effective_plot_epsilon_match(config)
        ax.set_xlim(
            max(0.0, epsilon_match - 20.0),
            config.epsilon0_mev_fm3 + 3.0 * config.sigma_mev_fm3,
        )
        title = "Raw sound-speed proposals: anchor and core zoom"
        name = "raw_cs2_anchor_core_zoom.png"
    else:
        title = "Raw sound-speed proposals: complete retained domain"
        name = "raw_cs2_full_domain.png"
    _draw_sound_speed_guides(ax)
    axis_style(
        ax,
        xlabel=r"$\varepsilon$ [MeV fm$^{-3}$]",
        ylabel=r"raw $c_s^2$",
        title=title,
    )
    ax.legend(
        fontsize=6.8,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
    )
    fig.subplots_adjust(bottom=0.22)
    _footer(fig, config, "Only raw-gate accepted proposals are shown.")
    _add_amplitude_colorbar(
        fig,
        ax,
        pd.concat((profile, accepted), ignore_index=True, sort=False),
    )
    _finalize(fig, packet / "plots" / name)
    return True


def _delta_cs2(packet, config, plt, axis_style) -> bool:
    frame = _table(packet, "raw_gate_profiles.csv")
    frame = frame.loc[frame.gate_status == "accepted_raw_local_physics_gate"]
    all_deltas = sorted(frame.delta_mev_fm3.dropna().unique())
    deltas, omitted = bounded_representative_values(
        all_deltas,
        preferred=(config.diagnostic_delta_mev_fm3,),
    )
    if frame.empty:
        return False
    displayed = frame.loc[frame.delta_mev_fm3.isin(deltas)]
    styles = _style_rows(displayed)
    fig, axes = plt.subplots(1, len(deltas), figsize=(5.0 * len(deltas), 4.6), squeeze=False)
    for ax, delta in zip(axes.flat, deltas):
        for case_id, rows in frame.loc[np.isclose(frame.delta_mev_fm3, delta)].groupby("case_id"):
            ax.plot(rows.epsilon_mev_fm3, rows.delta_cs2, **styles[str(case_id)])
        _scales(ax, config, float(delta))
        ax.axhline(0.0, color="#111827", lw=0.7)
        axis_style(
            ax,
            xlabel=r"$\varepsilon$ [MeV fm$^{-3}$]",
            ylabel=r"$\Delta c_s^2$",
            title=rf"$\Delta={delta:g}$",
        )
    axes.flat[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Realized windowed sound-speed deformation")
    _footer(fig, config, _dense_panel_note(deltas, omitted, len(all_deltas)))
    _add_amplitude_colorbar(fig, axes, displayed)
    _finalize(fig, packet / "plots/delta_cs2.png")
    return True


def _thermo_pair(
    packet,
    config,
    plt,
    axis_style,
    *,
    filename: str,
    value: str,
    relative: str,
    ylabel: str,
    title: str,
    log_value: bool = False,
) -> bool:
    frame = _table(packet, "thermodynamic_profiles.csv")
    if frame.empty or value not in frame:
        return False
    styles = _style_rows(frame)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for case_id, rows in frame.groupby("case_id"):
        # These response figures intentionally use a linear energy-density
        # axis so the window, Gaussian center, and stellar-core region remain
        # spatially legible.  Pressure retains its logarithmic value axis.
        plot = axes[0].semilogy if log_value else axes[0].plot
        plot(rows.epsilon_mev_fm3, rows[value], **styles[str(case_id)])
        axes[1].plot(rows.epsilon_mev_fm3, rows[relative], **styles[str(case_id)])
    if value == "pressure_mev_fm3":
        _draw_pressure_energy_guides(axes[0])
    axes[0].legend(fontsize=6.8, ncol=2)
    _scales(axes[1], config, config.diagnostic_delta_mev_fm3)
    if _uses_amplitude_colorbar(frame):
        axes[1].legend(fontsize=6.8, ncol=2)
    axis_style(
        axes[0],
        xlabel=r"$\varepsilon$ [MeV fm$^{-3}$]",
        ylabel=ylabel,
    )
    axis_style(
        axes[1],
        xlabel=r"$\varepsilon$ [MeV fm$^{-3}$]",
        ylabel="relative response",
    )
    fig.suptitle(title)
    _footer(
        fig,
        config,
        "Generated microscopic composition/species chemical potentials are unavailable.",
    )
    _add_amplitude_colorbar(fig, axes, frame)
    _finalize(fig, packet / "plots" / filename)
    return True


def _gamma(packet, config, plt, axis_style) -> bool:
    frame = _table(packet, "thermodynamic_profiles.csv")
    if frame.empty:
        return False
    styles = _style_rows(frame)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for case_id, rows in frame.groupby("case_id"):
        ax.plot(rows.epsilon_mev_fm3, rows.gamma_eff, **styles[str(case_id)])
    _scales(ax, config, config.diagnostic_delta_mev_fm3)
    _draw_gamma_reference(ax)
    axis_style(
        ax,
        xlabel=r"$\varepsilon$ [MeV fm$^{-3}$]",
        ylabel=r"$\Gamma_{\rm eff}$",
        title="Effective-barotrope adiabatic-index response",
    )
    ax.legend(fontsize=6.8, ncol=2)
    _footer(fig, config)
    _add_amplitude_colorbar(fig, ax, frame)
    _finalize(fig, packet / "plots/gamma_eff_response.png")
    return True


def _residuals(packet, config, plt, axis_style) -> bool:
    frame = _table(packet, "thermodynamic_residuals.csv")
    if frame.empty:
        return False
    styles = _style_rows(frame)
    metrics = (
        ("r_p_independent_normalized", r"normalized $r_P$"),
        ("r_mu_independent_normalized", r"normalized $r_h$"),
        ("first_law_normalized", "normalized first-law form"),
        ("r_c", r"$|c_s^2-dP/d\varepsilon|$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))
    epsilon_match = _effective_plot_epsilon_match(config)
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for case_id, rows in frame.groupby("case_id"):
            ax.loglog(
                rows.epsilon_mev_fm3,
                np.maximum(np.abs(rows[metric]), np.finfo(float).tiny),
                **styles[str(case_id)],
            )
        ax.axvspan(
            epsilon_match - 0.5,
            epsilon_match + config.diagnostic_delta_mev_fm3 + 0.5,
            color="#f59e0b",
            alpha=0.08,
            label="anchor/ramp-sensitive band",
        )
        for transition in (0.253215574967, 76.5591451931):
            ax.axvspan(transition * 0.98, transition * 1.02, color="#64748b", alpha=0.08)
        axis_style(
            ax,
            xlabel=r"$\varepsilon$ [MeV fm$^{-3}$]",
            ylabel=ylabel,
        )
    axes.flat[0].legend(fontsize=6.0, ncol=2)
    fig.suptitle(
        "PCHIP derivative-consistency closure family and distinct dP/d-epsilon check"
    )
    _footer(fig, config, "Global and sensitive-region maxima are stored in JSON.")
    _add_amplitude_colorbar(fig, axes, frame)
    _finalize(fig, packet / "plots/thermodynamic_residuals.png")
    return True


def _identity(packet, config, plt, axis_style) -> bool:
    frame = _table(packet, "a0_identity_table.csv")
    if frame.empty:
        return False
    display = frame[
        ["scope", "delta_mev_fm3", "stage", "quantity", "maximum_absolute_residual", "status"]
    ].copy()
    display["maximum_absolute_residual"] = display["maximum_absolute_residual"].map(
        lambda value: "—" if pd.isna(value) else f"{float(value):.3e}"
    )
    fig, ax = plt.subplots(figsize=(13.0, max(4.2, 0.25 * len(display) + 1.8)))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.25)
    ax.set_title("A=0 identity control")
    _footer(fig, config)
    _finalize(fig, packet / "plots/a0_identity.png")
    return True

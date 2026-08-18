"""Lazy saved-table plotting for saved :mod:`eos_generation` trial packets.

This private module is imported only by an explicit plotting action.  It does
not run EoS reconstruction, TOV integration, or diagnostic calculations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from eos_generation._internal.artifacts import ensure_within_runs
from eos_generation.reporting._plotting_data import (
    _ACTIVE_REPORTING_CONTEXT,
    _ACTIVE_TABLE_CACHE,
    _saved_plot_reporting_context,
    _summarize_stellar_tidal_plot_completeness,
    _table,
)
from eos_generation.reporting._plotting_diagnostics import (
    _baryonic,
    _errors,
    _odd_even,
    _radial,
    _response,
    _support,
    _unavailable_saved_table_plot,
)
from eos_generation.reporting._plotting_stellar import (
    _observable_response,
    _stellar,
)
from eos_generation.reporting._plotting_thermodynamic import (
    _delta_cs2,
    _gamma,
    _gaussian_realization,
    _identity,
    _raw_sound_speed,
    _residuals,
    _thermo_pair,
    _window_profiles,
)


def summarize_stellar_tidal_plot_completeness(
    sequence_rows: pd.DataFrame,
    fixed_mass_rows: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize what the M-R/tidal figure can truthfully display."""

    return _summarize_stellar_tidal_plot_completeness(
        sequence_rows,
        fixed_mass_rows,
    )


def stellar_tidal_plot_completeness(packet_path: str | Path, *, config) -> dict[str, Any]:
    """Return saved-table tidal completeness for plot-inventory integration."""

    packet = ensure_within_runs(packet_path)
    stage = config.tov_stages[-1].name
    sequences = _table(packet, "stellar_sequences.csv")
    fixed = _table(packet, "fixed_mass_observables.csv")
    if "stage" in sequences.columns:
        sequences = sequences.loc[sequences.stage == stage]
    if "stage" in fixed.columns:
        fixed = fixed.loc[fixed.stage == stage]
    return summarize_stellar_tidal_plot_completeness(sequences, fixed)


def render_trial_figures(
    packet_path: str | Path,
    figures: Sequence[str],
    *,
    config,
) -> tuple[str, ...]:
    """Render requested applicable figures from saved trial tables."""
    packet = ensure_within_runs(packet_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eos_generation.reporting.plot_helpers import apply_axis_style

    renderers: Mapping[str, Callable[[], bool]] = {
        "window_profiles.png": lambda: _window_profiles(packet, config, plt, apply_axis_style),
        "gaussian_realization.png": lambda: _gaussian_realization(packet, config, plt, apply_axis_style),
        "raw_cs2_full_domain.png": lambda: _raw_sound_speed(packet, config, plt, apply_axis_style, zoom=False),
        "raw_cs2_anchor_core_zoom.png": lambda: _raw_sound_speed(packet, config, plt, apply_axis_style, zoom=True),
        "delta_cs2.png": lambda: _delta_cs2(packet, config, plt, apply_axis_style),
        "pressure_response.png": lambda: _thermo_pair(
            packet,
            config,
            plt,
            apply_axis_style,
            filename="pressure_response.png",
            value="pressure_mev_fm3",
            relative="pressure_relative_to_direct",
            ylabel=r"$P$ [MeV fm$^{-3}$]",
            title="Pressure and relative pressure response",
            log_value=False,
        ),
        "baryon_density_response.png": lambda: _thermo_pair(
            packet,
            config,
            plt,
            apply_axis_style,
            filename="baryon_density_response.png",
            value="baryon_density_fm3",
            relative="baryon_density_relative_to_direct",
            ylabel=r"$n_B$ [fm$^{-3}$]",
            title="C4-consistent reconstruction, C1-normalized at the anchor",
        ),
        "effective_baryon_enthalpy_response.png": lambda: _thermo_pair(
            packet,
            config,
            plt,
            apply_axis_style,
            filename="effective_baryon_enthalpy_response.png",
            value="effective_baryon_enthalpy_mev",
            relative="enthalpy_relative_to_direct",
            ylabel=r"$h_B=(\varepsilon+P)/n_B$ [MeV]",
            title=(
                "Effective baryon chemical potential / enthalpy per baryon "
                "from the C4-consistent reconstruction"
            ),
        ),
        "gamma_eff_response.png": lambda: _gamma(packet, config, plt, apply_axis_style),
        "thermodynamic_residuals.png": lambda: _residuals(packet, config, plt, apply_axis_style),
        "stellar_mr_k2_lambda.png": lambda: _stellar(packet, config, plt, apply_axis_style),
        "observable_response_vs_amplitude.png": lambda: _observable_response(
            packet, config, plt, apply_axis_style, versus="amplitude"
        ),
        "observable_response_vs_delta.png": lambda: _observable_response(
            packet, config, plt, apply_axis_style, versus="delta"
        ),
        "a0_identity.png": lambda: _identity(packet, config, plt, apply_axis_style),
        "radial_structure_profiles.png": lambda: _radial(packet, config, plt, apply_axis_style),
        "deformation_support_fractions.png": lambda: _support(packet, config, plt, apply_axis_style),
        "baryonic_mass_vs_mass.png": lambda: _baryonic(packet, config, plt, apply_axis_style, binding=False),
        "binding_energy_vs_mass.png": lambda: _baryonic(packet, config, plt, apply_axis_style, binding=True),
        "stellar_response_across_mass.png": lambda: _response(packet, config, plt, apply_axis_style, baryonic=False),
        "baryonic_response_across_mass.png": lambda: _response(packet, config, plt, apply_axis_style, baryonic=True),
        "odd_even_response.png": lambda: _odd_even(packet, config, plt, apply_axis_style),
        "numerical_error_summary.png": lambda: _errors(packet, config, plt, apply_axis_style),
        "outside_support_control.png": lambda: _unavailable_saved_table_plot(
            packet, config, plt, apply_axis_style, "outside_support_control.png"
        ),
        "turning_point_sequences.png": lambda: _unavailable_saved_table_plot(
            packet, config, plt, apply_axis_style, "turning_point_sequences.png"
        ),
        "turning_point_derivatives.png": lambda: _unavailable_saved_table_plot(
            packet, config, plt, apply_axis_style, "turning_point_derivatives.png"
        ),
        "matched_area_comparison.png": lambda: _unavailable_saved_table_plot(
            packet, config, plt, apply_axis_style, "matched_area_comparison.png"
        ),
    }
    generated: list[str] = []
    reporting_token = _ACTIVE_REPORTING_CONTEXT.set(
        _saved_plot_reporting_context(packet, config)
    )
    table_token = _ACTIVE_TABLE_CACHE.set({})
    try:
        for name in figures:
            renderer = renderers.get(name)
            if renderer is not None and renderer():
                generated.append(name)
    finally:
        _ACTIVE_TABLE_CACHE.reset(table_token)
        _ACTIVE_REPORTING_CONTEXT.reset(reporting_token)
    return tuple(generated)


__all__ = [
    "render_trial_figures",
    "stellar_tidal_plot_completeness",
    "summarize_stellar_tidal_plot_completeness",
]

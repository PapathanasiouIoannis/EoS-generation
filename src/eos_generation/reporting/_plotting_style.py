"""Shared plot styling, guides, labels, and figure finalization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eos_generation.reporting._plotting_data import (
    _ACTIVE_REPORTING_CONTEXT,
    _effective_plot_epsilon_match,
    _saved_case_style_rows,
)

_AMPLITUDE_COLORBAR_THRESHOLD = 8


def _amplitude_values(frame: pd.DataFrame) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                float(value)
                for value in frame.get(
                    "amplitude", pd.Series(dtype=float)
                ).dropna()
                if math.isfinite(float(value))
            }
        )
    )


def _amplitude_color_scale(frame: pd.DataFrame):
    amplitudes = _amplitude_values(frame)
    if not amplitudes:
        return None

    import matplotlib
    from matplotlib.colors import Normalize, TwoSlopeNorm

    minimum, maximum = min(amplitudes), max(amplitudes)
    if minimum == maximum:
        padding = max(abs(minimum), 1.0) * np.finfo(float).eps
        norm = Normalize(vmin=minimum - padding, vmax=maximum + padding)
    elif minimum < 0.0 < maximum:
        norm = TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
    else:
        norm = Normalize(vmin=minimum, vmax=maximum)
    return matplotlib.colormaps["coolwarm"], norm


def _uses_amplitude_colorbar(frame: pd.DataFrame) -> bool:
    return len(_amplitude_values(frame)) > _AMPLITUDE_COLORBAR_THRESHOLD


def _add_amplitude_colorbar(fig, axes, frame: pd.DataFrame) -> bool:
    if not _uses_amplitude_colorbar(frame):
        return False
    scale = _amplitude_color_scale(frame)
    if scale is None:
        return False

    from matplotlib.cm import ScalarMappable

    palette, norm = scale
    mappable = ScalarMappable(norm=norm, cmap=palette)
    mappable.set_array(np.asarray(_amplitude_values(frame), dtype=float))
    colorbar = fig.colorbar(
        mappable,
        ax=list(np.asarray(axes, dtype=object).flat),
        fraction=0.035,
        pad=0.025,
    )
    colorbar.set_label(r"Amplitude $A$")
    return True


def _style_rows(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    case_ids = [str(value) for value in frame.get("case_id", pd.Series(dtype=str)).dropna().unique()]
    amplitudes = _amplitude_values(frame)
    deltas = sorted(
        {
            float(value)
            for value in frame.get("delta_mev_fm3", pd.Series(dtype=float)).dropna()
            if math.isfinite(float(value))
        }
    )
    colors = {}
    if amplitudes:
        palette, norm = _amplitude_color_scale(frame)
        colors = {value: palette(norm(value)) for value in amplitudes}
    dense_amplitude_family = _uses_amplitude_colorbar(frame)
    line_styles = ("-", "--", "-.", ":")
    delta_styles = {
        value: line_styles[index % len(line_styles)] for index, value in enumerate(deltas)
    }
    styles: dict[str, dict[str, Any]] = {
        "direct": {
            "color": "#111827",
            "linestyle": "-",
            "linewidth": 2.4,
            "label": "Direct BSk24 baseline",
        }
    }
    for case_id in case_ids:
        if case_id == "direct":
            continue
        row = frame.loc[frame.case_id == case_id].iloc[0]
        amplitude = float(row.amplitude)
        delta = float(row.delta_mev_fm3)
        styles[case_id] = {
            "color": colors.get(amplitude, "#4b5563"),
            "linestyle": delta_styles.get(delta, "-"),
            "linewidth": 1.15 if dense_amplitude_family else 1.8,
            "alpha": 0.85 if dense_amplitude_family else 1.0,
            "label": (
                "_nolegend_"
                if dense_amplitude_family
                else rf"$A={amplitude:+.4g}$, $\Delta={delta:g}$"
            ),
        }
    return styles


def _publication_case_styles(
    packet: Path,
    frame: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Return physical labels and visible identity-control markers.

    Saved case IDs remain the immutable table keys.  They are deliberately not
    used as publication legend text.  The direct and A=0 curves coincide by
    the governed identity contract, so A=0 is rendered as markers only instead
    of hiding the direct curve with a second line.
    """

    coordinates = _saved_case_style_rows(packet, frame)
    styles = _style_rows(coordinates)
    lookup = (
        coordinates.drop_duplicates("case_id")
        .set_index("case_id", drop=False)
        if not coordinates.empty
        else pd.DataFrame()
    )
    for case_id in tuple(styles):
        style = styles[case_id]
        if case_id == "direct":
            style.update(label="Direct BSk24", marker=None, zorder=2.0)
            continue
        row = lookup.loc[case_id] if case_id in lookup.index else pd.Series(dtype=object)
        amplitude = pd.to_numeric(
            pd.Series([row.get("amplitude")]), errors="coerce"
        ).iloc[0]
        if np.isfinite(amplitude) and np.isclose(float(amplitude), 0.0):
            style.update(
                color="#d97706",
                linestyle="none",
                linewidth=0.0,
                marker="x",
                markersize=5.2,
                markeredgewidth=1.1,
                label=r"$A=0$ identity",
                zorder=3.0,
            )
        else:
            style.update(marker="o", markersize=4.2, zorder=2.5)
    return styles


def _concise_mass_radius_legend_label(label: str) -> str:
    """Shorten governed compactness labels for a non-overlapping legend."""

    if label.startswith("Schwarzschild horizon:"):
        return r"Schwarzschild horizon: $R=2GM/c^2$"
    if label.startswith("Buchdahl boundary (conditional):"):
        return r"Buchdahl bound (conditional): $R=9GM/(4c^2)$"
    if label.startswith("Causal compactness diagnostic (conditional):"):
        return r"Causal compactness (conditional): $R=2.824GM/c^2$"
    if label == "sampled peak (not Mmax)":
        return r"Sampled sequence peak (not $M_{\max}$)"
    return label


def _footer(
    fig,
    config,
    extra: str = "",
    *,
    context: Mapping[str, str] | None = None,
) -> Any:
    # Publication-facing canvases contain only scientific axes, concise
    # titles, legends, and colour bars.  The same traceability data remain in
    # metadata.json, summary.md, plot_inventory.*, and
    # plot_generation_provenance.json instead of being rendered as a large
    # paragraph below every panel.
    reporting = dict(context or _ACTIVE_REPORTING_CONTEXT.get() or {})
    fig._bsk24_off_canvas_reporting = {
        **reporting,
        "note": " ".join(extra.split()) if extra else "",
    }
    return None


def _scales(ax, config, delta: float | None = None) -> None:
    epsilon_t = _effective_plot_epsilon_match(config)
    ax.axvline(epsilon_t, color="#4b5563", ls=":", lw=0.9, label=r"$\varepsilon_t$")
    if delta is not None:
        ax.axvline(
            epsilon_t + delta,
            color="#6b7280",
            ls=(0, (3, 2)),
            lw=0.9,
            label=r"$\varepsilon_t+\Delta$",
        )
    ax.axvline(
        config.epsilon0_mev_fm3,
        color="#7c3aed",
        ls="--",
        lw=0.9,
        label=r"$\varepsilon_0$",
    )
    ax.axvspan(
        config.epsilon0_mev_fm3 - config.sigma_mev_fm3,
        config.epsilon0_mev_fm3 + config.sigma_mev_fm3,
        color="#7c3aed",
        alpha=0.06,
        label=r"$\varepsilon_0\pm\sigma$",
    )


def _draw_sound_speed_guides(ax) -> dict[str, Any]:
    """Draw direct hard and conformal references on a ``c_s^2`` axis.

    These guides apply to the declared continuous cold phase.  In particular,
    the conformal value is deliberately styled as a neutral reference rather
    than as an acceptance boundary.
    """

    current_lower, current_upper = (float(value) for value in ax.get_ylim())
    x_lower, x_upper = (float(value) for value in ax.get_xlim())
    x_values = np.asarray((x_lower, x_upper), dtype=float)
    padding = max(0.05, 0.04 * max(current_upper - current_lower, 1.0))
    lower = min(current_lower, -padding)
    upper = max(current_upper, 1.0 + padding)
    mechanical_region = ax.fill_between(
        x_values,
        np.full_like(x_values, lower),
        np.zeros_like(x_values),
        color="#b91c1c",
        alpha=0.08,
        linewidth=0.0,
        label=r"$c_s^2<0$ mechanically unstable [hard]",
        zorder=0,
    )
    acausal_region = ax.fill_between(
        x_values,
        np.ones_like(x_values),
        np.full_like(x_values, upper),
        color="#b91c1c",
        alpha=0.08,
        linewidth=0.0,
        label=r"$c_s^2>1$ acausal [hard]",
        zorder=0,
    )
    lower_boundary = ax.plot(
        x_values,
        np.zeros_like(x_values),
        color="#b91c1c",
        linestyle=":",
        linewidth=0.9,
        label="_nolegend_",
        zorder=1,
    )[0]
    upper_boundary = ax.plot(
        x_values,
        np.ones_like(x_values),
        color="#b91c1c",
        linestyle="--",
        linewidth=0.9,
        label="_nolegend_",
        zorder=1,
    )[0]
    conformal_reference = ax.plot(
        x_values,
        np.full_like(x_values, 1.0 / 3.0),
        color="#64748b",
        linestyle="-.",
        linewidth=0.9,
        label=r"$c_s^2=1/3$ conformal reference",
        zorder=1,
    )[0]
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(lower, upper)
    return {
        "mechanical_region": mechanical_region,
        "acausal_region": acausal_region,
        "lower_boundary": lower_boundary,
        "upper_boundary": upper_boundary,
        "conformal_reference": conformal_reference,
    }


def _draw_gamma_reference(ax) -> Any:
    """Draw the local four-thirds reference without implying mode stability."""

    x_lower, x_upper = (float(value) for value in ax.get_xlim())
    line = ax.plot(
        (x_lower, x_upper),
        (4.0 / 3.0, 4.0 / 3.0),
        color="#d97706",
        linestyle="--",
        linewidth=0.9,
        label=r"$\Gamma_{\rm eff}=4/3$ reference (not a radial-stability gate)",
    )[0]
    ax.set_xlim(x_lower, x_upper)
    return line


def _draw_pressure_energy_guides(ax) -> dict[str, Any]:
    """Draw conditional DEC and neutral conformal lines on a P-epsilon axis."""

    x_lower, x_upper = (float(value) for value in ax.get_xlim())
    epsilon = np.linspace(max(0.0, x_lower), x_upper, 256)
    dominant_energy_boundary = ax.plot(
        epsilon,
        epsilon,
        color="#d97706",
        linestyle="--",
        linewidth=0.9,
        label=r"$P=\varepsilon$ DEC boundary [conditional]",
        zorder=1,
    )[0]
    conformal_reference = ax.plot(
        epsilon,
        epsilon / 3.0,
        color="#64748b",
        linestyle="-.",
        linewidth=0.9,
        label=r"$P=\varepsilon/3$ conformal reference",
        zorder=1,
    )[0]
    ax.set_xlim(x_lower, x_upper)
    return {
        "dominant_energy_boundary": dominant_energy_boundary,
        "conformal_reference": conformal_reference,
    }


def _draw_mass_radius_compactness_guides(
    ax,
    sequence_rows: pd.DataFrame,
    fixed_rows: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Draw governed compactness regions without touching tidal axes."""

    from eos_generation.reporting.plot_helpers import draw_compactness_boundaries

    def numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
        if name not in frame:
            return pd.Series(dtype=float)
        return pd.to_numeric(frame[name], errors="coerce")

    radii = pd.concat(
        (
            numeric_column(sequence_rows, "Radius"),
            numeric_column(fixed_rows, "radius_km"),
        ),
        ignore_index=True,
    ).to_numpy(dtype=float)
    masses = pd.concat(
        (
            numeric_column(sequence_rows, "Mass"),
            numeric_column(fixed_rows, "target_mass_msun"),
        ),
        ignore_index=True,
    ).to_numpy(dtype=float)
    radii = radii[np.isfinite(radii) & (radii > 0.0)]
    masses = masses[np.isfinite(masses) & (masses >= 0.0)]
    if radii.size == 0 or masses.size == 0:
        return {}

    radius_lower = max(0.0, min(5.0, 0.8 * float(np.min(radii))))
    radius_upper = max(16.0, 1.05 * float(np.max(radii)))
    mass_upper = max(3.0, 1.08 * float(np.max(masses)))
    guide_masses = np.linspace(0.0, mass_upper, 256)
    artists = draw_compactness_boundaries(
        ax,
        guide_masses,
        show_horizon=True,
        show_buchdahl=True,
        show_causal=True,
        shade_regions=True,
        linewidth=1.05,
    )
    # The immutable boundary labels already carry the scientific meaning.
    # Suppress duplicate region entries while retaining their governed fills.
    for payload in artists.values():
        payload["line"].set_zorder(1)
        region = payload.get("region")
        if region is not None:
            region.set_label("_nolegend_")
            region.set_zorder(0)
    ax.set_xlim(radius_lower, radius_upper)
    ax.set_ylim(0.0, mass_upper)
    return artists


def _finalize(fig, path: Path):
    from eos_generation.reporting.plot_helpers import finalize_figure

    return finalize_figure(fig, path)


def _dense_panel_note(
    displayed: Sequence[float], omitted: int, total: int
) -> str:
    if omitted <= 0:
        return ""
    labels = ", ".join(f"{value:g}" for value in displayed)
    return (
        f"Presentation-only bounded panel selection shows {len(displayed)} of "
        f"{total} Delta values ({labels} MeV fm^-3); complete saved tables "
        f"retain all values. {omitted} panels are omitted from this figure."
    )

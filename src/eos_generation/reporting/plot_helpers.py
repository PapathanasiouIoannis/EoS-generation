"""Small shared helpers for saved-table figures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from eos_generation._internal.artifacts import ensure_within_runs
from eos_generation._internal.config import DEFAULT_CONFIG


def apply_axis_style(
    ax: Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid_alpha: float = 0.24,
    show_grid: bool = True,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if show_grid:
        ax.grid(True, alpha=grid_alpha)


def finalize_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 320,
    bbox_inches: str = "tight",
    close: bool = True,
) -> Path:
    output = ensure_within_runs(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    image_format = output.suffix.lstrip(".") or "png"
    try:
        fig.savefig(
            temporary,
            format=image_format,
            dpi=dpi,
            bbox_inches=bbox_inches,
        )
        # Windows _commit/os.fsync requires a writable descriptor. Reopening
        # the completed temporary image read/write does not change its bytes.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
        if close:
            plt.close(fig)
    return output


def _compactness_boundaries(masses_msun: Any) -> dict[str, dict[str, Any]]:
    masses = np.asarray(masses_msun, dtype=float)
    if masses.ndim != 1 or masses.size == 0:
        raise ValueError("masses_msun must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("masses_msun must contain finite nonnegative values")
    length = float(DEFAULT_CONFIG.units.solar_mass_length_km)
    specifications = (
        (
            "schwarzschild_horizon",
            "Schwarzschild horizon: R = 2GM/c^2",
            2.0,
        ),
        (
            "buchdahl_boundary",
            "Buchdahl boundary (conditional): R = 9GM/(4c^2)",
            9.0 / 4.0,
        ),
        (
            "causal_compactness_boundary",
            "Causal compactness diagnostic (conditional): R = 2.824GM/c^2",
            2.824,
        ),
    )
    return {
        identifier: {
            "label": label,
            "masses": masses,
            "radii": factor * length * masses,
        }
        for identifier, label, factor in specifications
    }


def draw_compactness_boundaries(
    ax: Axes,
    masses_msun: Any,
    *,
    show_horizon: bool = True,
    show_buchdahl: bool = True,
    show_causal: bool = True,
    shade_regions: bool = True,
    linewidth: float = 1.25,
) -> dict[str, dict[str, Any]]:
    boundaries = _compactness_boundaries(masses_msun)
    enabled = {
        "schwarzschild_horizon": show_horizon,
        "buchdahl_boundary": show_buchdahl,
        "causal_compactness_boundary": show_causal,
    }
    colors = {
        "schwarzschild_horizon": "#202020",
        "buchdahl_boundary": "#b84a4a",
        "causal_compactness_boundary": "#d07a32",
    }
    styles = {
        "schwarzschild_horizon": "-",
        "buchdahl_boundary": "--",
        "causal_compactness_boundary": ":",
    }
    alphas = {
        "schwarzschild_horizon": 0.16,
        "buchdahl_boundary": 0.11,
        "causal_compactness_boundary": 0.08,
    }
    lower = {
        "schwarzschild_horizon": np.zeros_like(boundaries["schwarzschild_horizon"]["radii"]),
        "buchdahl_boundary": boundaries["schwarzschild_horizon"]["radii"],
        "causal_compactness_boundary": boundaries["buchdahl_boundary"]["radii"],
    }
    artists: dict[str, dict[str, Any]] = {}
    for identifier, boundary in boundaries.items():
        if not enabled[identifier]:
            continue
        line = ax.plot(
            boundary["radii"],
            boundary["masses"],
            color=colors[identifier],
            linestyle=styles[identifier],
            linewidth=linewidth,
            label=boundary["label"],
        )[0]
        region = None
        if shade_regions:
            region = ax.fill_betweenx(
                boundary["masses"],
                lower[identifier],
                boundary["radii"],
                color=colors[identifier],
                alpha=alphas[identifier],
                linewidth=0.0,
            )
        artists[identifier] = {"line": line, "region": region}
    return artists


__all__ = ["apply_axis_style", "draw_compactness_boundaries", "finalize_figure"]

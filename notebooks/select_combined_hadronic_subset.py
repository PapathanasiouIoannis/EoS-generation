"""Select a deterministic coverage-preserving subset of a combined BSk24 dataset.

This is a saved-data adapter.  It verifies the immutable parent dataset and its
authoritative source packets, performs no thermodynamic or stellar solver work,
and writes a separate dry-run selection report.  Whole EoSs are selected; curve
rows are never sampled independently.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eos_catalogue as catalogue


SCHEMA_ID = "bsk24_combined_hadronic_subset_dryrun_v1"
TARGET_COUNT = 2000
R14_BIN_WIDTH_KM = 0.01
MASS_ENVELOPE_GRID = np.round(np.arange(0.50, 3.3001, 0.05), 10)
MASS_LOW_GRID = np.round(np.arange(0.50, 2.0001, 0.05), 10)
MASS_HIGH_GRID = np.round(np.arange(2.05, 3.3001, 0.10), 10)
EPSILON_ENVELOPE_GRID = np.round(np.arange(80.0, 2000.0001, 10.0), 10)
EPSILON_FEATURE_GRID = np.linspace(80.0, 2000.0, 65)

FEATURE_WEIGHTS = {
    "r14": 8.0,
    "radius_below_2msun": 4.0,
    "log10_lambda_below_2msun": 2.0,
    "k2_below_2msun": 2.0,
    "radius_above_2msun": 0.50,
    "log10_lambda_above_2msun": 0.25,
    "k2_above_2msun": 0.25,
    "log10_pressure": 2.0,
    "cs2": 2.0,
    "sampled_peak_mass": 0.25,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _read_manifest(folder: Path) -> dict[str, str]:
    return catalogue.manifest(folder)


def _verified(folder: Path, relative: str, manifest: dict[str, str]) -> Path:
    return catalogue.verified(folder, relative, manifest)


def _direct_child(parent: Path, name: str) -> Path:
    """Resolve one existing schema-declared child without traversal."""

    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).is_absolute()
    ):
        raise ValueError(f"unsafe direct-child name: {name!r}")
    resolved_parent = parent.resolve(strict=True)
    requested = resolved_parent / name
    is_junction = getattr(requested, "is_junction", lambda: False)
    if requested.is_symlink() or is_junction():
        raise ValueError(f"schema child must not be a linked directory: {name!r}")
    child = catalogue.confined(requested, resolved_parent)
    if child.parent != resolved_parent or not child.is_dir():
        raise ValueError(f"schema child is not a direct directory: {name!r}")
    return child


def _source_experiments(
    root: Path, source_records: list[dict[str, Any]]
) -> dict[str, Path]:
    """Resolve declared source experiments once beneath the trusted runs tree."""

    runs = (root / "runs").resolve(strict=True)
    experiments: dict[str, Path] = {}
    for record in source_records:
        run = str(record.get("run", ""))
        if not run or run in experiments:
            raise ValueError("source-run declarations require unique non-empty names")
        declared = record.get("experiment_path")
        if not isinstance(declared, str) or not declared:
            raise ValueError(f"source run has no experiment path: {run!r}")
        experiment = catalogue.confined(root / declared, runs)
        if not experiment.is_dir():
            raise FileNotFoundError(experiment)
        experiments[run] = experiment
    return experiments


def _load_combiner(root: Path):
    trusted_root = catalogue.trusted_repository_root(root)
    module_path = Path(__file__).resolve(strict=True).with_name(
        "build_combined_hadronic_dataset.py"
    )
    if module_path.parent != trusted_root / "notebooks":
        raise ValueError("combined builder is not owned by the trusted checkout")
    spec = importlib.util.spec_from_file_location("combined_builder_for_subset", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _interp(rows: pd.DataFrame, x: str, y: str, grid: np.ndarray) -> np.ndarray:
    values = rows.loc[:, [x, y]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        return np.full(len(grid), np.nan)
    order = np.argsort(values[:, 0], kind="mergesort")
    values = values[order]
    unique_x, unique_indices = np.unique(values[:, 0], return_index=True)
    unique_y = values[unique_indices, 1]
    result = np.full(len(grid), np.nan)
    valid = (grid >= unique_x[0]) & (grid <= unique_x[-1])
    if len(unique_x) == 1:
        result[valid & np.isclose(grid, unique_x[0])] = unique_y[0]
    else:
        result[valid] = np.interp(grid[valid], unique_x, unique_y)
    return result


def _matrix(
    curves: dict[str, pd.DataFrame],
    names: list[str],
    x: str,
    y: str,
    grid: np.ndarray,
) -> np.ndarray:
    return np.vstack([_interp(curves[name], x, y, grid) for name in names])


def _arg_extreme(column: np.ndarray, minimum: bool) -> int | None:
    finite = np.flatnonzero(np.isfinite(column))
    if not len(finite):
        return None
    local = np.argmin(column[finite]) if minimum else np.argmax(column[finite])
    return int(finite[local])


def _add_envelope_anchors(
    reasons: dict[str, set[str]],
    names: list[str],
    values: np.ndarray,
    family: str,
    grid: np.ndarray,
) -> None:
    for column_index, coordinate in enumerate(grid):
        column = values[:, column_index]
        for label, minimum in (("min", True), ("max", False)):
            index = _arg_extreme(column, minimum)
            if index is not None:
                reasons[names[index]].add(f"{family}_{label}@{coordinate:g}")


def _feature_family(
    values: np.ndarray,
    total_weight: float,
    transform: str = "identity",
) -> np.ndarray:
    data = values.astype(float, copy=True)
    if transform == "log10":
        data[data <= 0.0] = np.nan
        data = np.log10(data)
    finite = np.isfinite(data)
    columns = data.shape[1]
    scaled = np.full_like(data, 0.5)
    for column_index in range(columns):
        valid = finite[:, column_index]
        if not valid.any():
            continue
        low = float(np.nanmin(data[valid, column_index]))
        high = float(np.nanmax(data[valid, column_index]))
        if high > low:
            scaled[valid, column_index] = (
                data[valid, column_index] - low
            ) / (high - low)
        else:
            scaled[valid, column_index] = 0.5
    scaled *= math.sqrt(total_weight / max(columns, 1))
    if finite.all():
        return scaled
    availability = finite.astype(float) * math.sqrt(
        0.20 * total_weight / max(columns, 1)
    )
    return np.hstack((scaled, availability))


def _scalar_feature(values: np.ndarray, weight: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("scalar feature has no finite values")
    low = float(np.nanmin(values[finite]))
    high = float(np.nanmax(values[finite]))
    scaled = np.full(len(values), 0.5)
    if high > low:
        scaled[finite] = (values[finite] - low) / (high - low)
    return (scaled * math.sqrt(weight))[:, None]


def _initial_distances(features: np.ndarray, selected: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(selected)
    result = np.full(len(features), np.inf, dtype=np.float64)
    norms = np.einsum("ij,ij->i", features, features, dtype=np.float64)
    for start in range(0, len(indices), 128):
        block_indices = indices[start : start + 128]
        block = features[block_indices]
        block_norms = norms[block_indices]
        distances = (
            norms[:, None]
            + block_norms[None, :]
            - 2.0 * np.asarray(features @ block.T, dtype=np.float64)
        )
        np.maximum(distances, 0.0, out=distances)
        result = np.minimum(result, distances.min(axis=1))
    result[selected] = 0.0
    return result


def _targets(regimes: pd.Series, anchor_regimes: pd.Series, target: int) -> dict[str, int]:
    total_nonbaseline = int((regimes != "baseline").sum())
    negative = int((regimes == "negative").sum())
    available = target - 1
    negative_target = int(round(available * negative / total_nonbaseline))
    desired = {
        "baseline": 1,
        "negative": negative_target,
        "positive": available - negative_target,
    }
    anchor_counts = anchor_regimes.value_counts().to_dict()
    for regime in ("negative", "positive"):
        if anchor_counts.get(regime, 0) > desired[regime]:
            overflow = anchor_counts[regime] - desired[regime]
            other = "positive" if regime == "negative" else "negative"
            desired[regime] += overflow
            desired[other] -= overflow
    if sum(desired.values()) != target:
        raise AssertionError(desired)
    return desired


def _select_diversity(
    names: list[str],
    regimes: pd.Series,
    features: np.ndarray,
    anchor_names: set[str],
    target_count: int,
) -> tuple[np.ndarray, dict[str, int], dict[str, float]]:
    name_to_index = {name: index for index, name in enumerate(names)}
    selected = np.zeros(len(names), dtype=bool)
    for name in anchor_names:
        selected[name_to_index[name]] = True
    if int(selected.sum()) > target_count:
        raise ValueError("mandatory anchors exceed requested subset size")
    targets = _targets(regimes, regimes.loc[selected], target_count)
    counts = regimes.loc[selected].value_counts().to_dict()
    min_distance = _initial_distances(features, selected)
    selection_distance: dict[str, float] = {}
    while int(selected.sum()) < target_count:
        allowed = ~selected
        for regime in ("baseline", "negative", "positive"):
            if counts.get(regime, 0) >= targets[regime]:
                allowed &= regimes.to_numpy() != regime
        candidates = np.flatnonzero(allowed)
        if not len(candidates):
            raise RuntimeError(f"no candidate can satisfy quotas: {counts} -> {targets}")
        candidate_distances = min_distance[candidates]
        best_distance = float(candidate_distances.max())
        tied = candidates[np.isclose(candidate_distances, best_distance, rtol=0.0, atol=1e-14)]
        best = int(tied[0])  # mapping is deterministically ordered by name
        selected[best] = True
        regime = str(regimes.iloc[best])
        counts[regime] = counts.get(regime, 0) + 1
        selection_distance[names[best]] = math.sqrt(max(best_distance, 0.0))
        delta = features - features[best]
        distance = np.einsum("ij,ij->i", delta, delta, dtype=np.float64)
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = 0.0
        if int(selected.sum()) % 100 == 0:
            print(
                f"selection progress: {int(selected.sum())}/{target_count}; "
                f"counts={counts}; covering_radius={math.sqrt(float(min_distance.max())):.6g}",
                flush=True,
            )
    final_counts = {regime: int(counts.get(regime, 0)) for regime in targets}
    if final_counts != targets:
        raise AssertionError(
            f"final counts do not match quotas: {final_counts} != {targets}"
        )
    selection_distance["__covering_radius__"] = math.sqrt(float(min_distance.max()))
    return selected, targets, selection_distance


def _select_balanced(
    names: list[str],
    features: np.ndarray,
    anchor_names: set[str],
    r14_bins: np.ndarray,
    mr_cell_sets_below_2msun: list[set[tuple[int, int]]],
    target_count: int,
) -> tuple[np.ndarray, dict[str, float], dict[str, str], dict[int, int]]:
    """Water-fill R1.4 bins, then maximize new low-mass M-R coverage."""

    name_to_index = {name: index for index, name in enumerate(names)}
    selected = np.zeros(len(names), dtype=bool)
    for name in anchor_names:
        selected[name_to_index[name]] = True
    if int(selected.sum()) > target_count:
        raise ValueError("mandatory anchors exceed requested subset size")

    bin_members = {
        int(bin_index): np.flatnonzero(r14_bins == bin_index)
        for bin_index in np.unique(r14_bins)
    }
    anchor_counts = {
        bin_index: int(selected[indices].sum())
        for bin_index, indices in bin_members.items()
    }
    bin_counts = dict(anchor_counts)
    min_distance = _initial_distances(features, selected)
    selection_distance: dict[str, float] = {}
    selection_stage: dict[str, str] = {}
    covered_cells: set[tuple[int, int]] = set()
    for index in np.flatnonzero(selected):
        covered_cells.update(mr_cell_sets_below_2msun[index])

    def choose(candidates: np.ndarray, use_raster_gain: bool) -> int:
        if not len(candidates):
            raise RuntimeError("balanced selection has no eligible candidate")
        if use_raster_gain:
            gains = np.array(
                [len(mr_cell_sets_below_2msun[index] - covered_cells) for index in candidates],
                dtype=int,
            )
            candidates = candidates[gains == gains.max()]
        distances = min_distance[candidates]
        best_distance = float(distances.max())
        tied = candidates[np.isclose(distances, best_distance, rtol=0.0, atol=1e-14)]
        return int(tied[0])

    def add(index: int, stage: str) -> None:
        selected[index] = True
        bin_index = int(r14_bins[index])
        bin_counts[bin_index] += 1
        selection_distance[names[index]] = math.sqrt(max(float(min_distance[index]), 0.0))
        selection_stage[names[index]] = stage
        covered_cells.update(mr_cell_sets_below_2msun[index])
        delta = features - features[index]
        distance = np.einsum("ij,ij->i", delta, delta, dtype=np.float64)
        np.minimum(min_distance, distance, out=min_distance)
        min_distance[selected] = 0.0

    # First take every candidate in genuinely sparse bins and reach three in
    # every bin that can supply three.  This step never prefers a dense bin.
    base_targets = {
        bin_index: min(3, len(indices)) for bin_index, indices in bin_members.items()
    }
    while True:
        deficit_bins = {
            bin_index
            for bin_index, target in base_targets.items()
            if bin_counts[bin_index] < target
        }
        if not deficit_bins:
            break
        candidates = np.flatnonzero(
            (~selected) & np.isin(r14_bins, np.fromiter(sorted(deficit_bins), dtype=int))
        )
        add(choose(candidates, use_raster_gain=False), "r14_waterfill_base")

    # Spend the remaining budget level by level.  At each level, only bins
    # with the lowest selected occupancy are eligible; within that fair pool,
    # prefer a curve adding new M-R cells below 2 Msun, then maximum feature
    # distance.  Mapping order provides the final deterministic tie-break.
    while int(selected.sum()) < target_count:
        fillable = {
            bin_index
            for bin_index, indices in bin_members.items()
            if bin_counts[bin_index] < len(indices)
        }
        if not fillable:
            raise RuntimeError("all parent candidates were exhausted before reaching target")
        minimum_level = min(bin_counts[bin_index] for bin_index in fillable)
        eligible_bins = {
            bin_index for bin_index in fillable if bin_counts[bin_index] == minimum_level
        }
        candidates = np.flatnonzero(
            (~selected) & np.isin(r14_bins, np.fromiter(sorted(eligible_bins), dtype=int))
        )
        add(choose(candidates, use_raster_gain=True), "r14_waterfill_raster_diversity")
        if int(selected.sum()) % 100 == 0:
            print(
                f"balanced selection progress: {int(selected.sum())}/{target_count}; "
                f"bin_level={minimum_level}; covered_low_mass_cells={len(covered_cells)}; "
                f"covering_radius={math.sqrt(float(min_distance.max())):.6g}",
                flush=True,
            )

    selection_distance["__covering_radius__"] = math.sqrt(float(min_distance.max()))
    return selected, selection_distance, selection_stage, anchor_counts


def _finite_extrema(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        return np.nanmin(values, axis=0), np.nanmax(values, axis=0)


def _envelope_error(full: np.ndarray, selected: np.ndarray, log10: bool = False) -> dict[str, float]:
    left = full.astype(float, copy=True)
    right = selected.astype(float, copy=True)
    if log10:
        left[left <= 0.0] = np.nan
        right[right <= 0.0] = np.nan
        left = np.log10(left)
        right = np.log10(right)
    full_min, full_max = _finite_extrema(left)
    subset_min, subset_max = _finite_extrema(right)
    return {
        "minimum_envelope_max_abs_error": float(np.nanmax(np.abs(full_min - subset_min))),
        "maximum_envelope_max_abs_error": float(np.nanmax(np.abs(full_max - subset_max))),
    }


def _gap_metrics(values: np.ndarray) -> dict[str, float]:
    ordered = np.sort(np.asarray(values, dtype=float))
    gaps = np.diff(ordered)
    index = int(np.argmax(gaps))
    return {
        "minimum_km": float(ordered[0]),
        "maximum_km": float(ordered[-1]),
        "largest_adjacent_gap_km": float(gaps[index]),
        "largest_gap_lower_km": float(ordered[index]),
        "largest_gap_upper_km": float(ordered[index + 1]),
        "median_adjacent_gap_km": float(np.median(gaps)),
    }


def _mr_cell_sets(
    curves: dict[str, pd.DataFrame], names: Iterable[str], maximum_mass: float
) -> list[set[tuple[int, int]]]:
    mass_grid = np.round(np.arange(0.20, maximum_mass + 0.0001, 0.025), 10)
    result: list[set[tuple[int, int]]] = []
    for name in names:
        radii = _interp(curves[name], "M", "R", mass_grid)
        valid = np.isfinite(radii)
        mass_indices = np.flatnonzero(valid)
        radius_indices = np.floor(radii[valid] / 0.025 + 1e-12).astype(int)
        result.append(set(zip(mass_indices.tolist(), radius_indices.tolist())))
    return result


def _union_cells(cell_sets: Iterable[set[tuple[int, int]]]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for cells in cell_sets:
        result.update(cells)
    return result


def _density_metrics(
    parent_bins: np.ndarray,
    selected_mask: np.ndarray,
    anchor_mask: np.ndarray,
) -> dict[str, Any]:
    bins = np.unique(parent_bins)
    parent_counts = np.array([int((parent_bins == value).sum()) for value in bins])
    selected_counts = np.array(
        [int(((parent_bins == value) & selected_mask).sum()) for value in bins]
    )
    anchor_counts = np.array(
        [int(((parent_bins == value) & anchor_mask).sum()) for value in bins]
    )
    eligible_three = parent_counts >= 3
    sparse = parent_counts < 3
    excess = selected_counts > 4
    mean = float(selected_counts.mean())
    return {
        "occupied_bin_count": int(len(bins)),
        "selected_count_min": int(selected_counts.min()),
        "selected_count_median": float(np.median(selected_counts)),
        "selected_count_max": int(selected_counts.max()),
        "selected_count_coefficient_of_variation": float(selected_counts.std() / mean),
        "bins_with_at_least_three_parent_candidates": int(eligible_three.sum()),
        "eligible_bins_below_three_selected": int(
            ((selected_counts < 3) & eligible_three).sum()
        ),
        "sparse_bins_not_fully_retained": int(
            ((selected_counts != parent_counts) & sparse).sum()
        ),
        "bins_above_four_selected": int(excess.sum()),
        "bins_above_four_not_explained_by_anchors": int(
            (excess & (selected_counts != anchor_counts)).sum()
        ),
        "selected_count_histogram": {
            str(int(value)): int((selected_counts == value).sum())
            for value in np.unique(selected_counts)
        },
    }


def _write_manifest(folder: Path) -> None:
    paths = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    text = "".join(
        f"{_sha256(path)}  {path.relative_to(folder).as_posix()}\n" for path in paths
    )
    temporary = folder / ".SHA256SUMS.tmp"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, folder / "SHA256SUMS.txt")


def _plot_mr(
    stellar: dict[str, pd.DataFrame], names: list[str], selected: set[str], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    for name in names:
        rows = stellar[name]
        ax.plot(rows["R"], rows["M"], color="#cbd5e1", alpha=0.09, linewidth=0.45)
    for name in names:
        if name not in selected:
            continue
        rows = stellar[name]
        ax.plot(rows["R"], rows["M"], color="#000000", alpha=0.20, linewidth=0.55)
    ax.set_xlabel("Radius R [km]")
    ax.set_ylabel(r"Mass M [$M_\odot$]")
    ax.set_title("Dry-run coverage check: all 3,124 (grey) vs selected 2,000 (black)")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_r14(
    r14: np.ndarray, names: list[str], selected: set[str], path: Path
) -> None:
    mask = np.array([name in selected for name in names])
    bins = np.arange(math.floor(r14.min() * 50) / 50, math.ceil(r14.max() * 50) / 50 + 0.0201, 0.02)
    fig, (hist, spacing) = plt.subplots(2, 1, figsize=(10.0, 7.5), height_ratios=(2.0, 1.0))
    hist.hist(r14, bins=bins, color="#cbd5e1", label="All 3,124", alpha=0.85)
    hist.hist(r14[mask], bins=bins, histtype="step", color="#000000", linewidth=1.1, label="Selected 2,000")
    hist.set_ylabel("EoS count")
    hist.legend()
    hist.grid(alpha=0.18)
    full = np.sort(r14)
    subset = np.sort(r14[mask])
    spacing.plot(full[:-1], np.diff(full), color="#94a3b8", linewidth=0.75, label="All")
    spacing.plot(subset[:-1], np.diff(subset), color="#000000", linewidth=0.75, label="Selected")
    spacing.axhline(0.03, color="#dc2626", linestyle="--", linewidth=0.8, label="0.03 km gate")
    spacing.set_xlabel(r"$R_{1.4}$ [km]")
    spacing.set_ylabel("Adjacent gap [km]")
    spacing.grid(alpha=0.18)
    spacing.legend(ncol=3, fontsize=8)
    fig.suptitle(r"$R_{1.4}$ coverage retained by the dry-run subset")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_thermo(
    curves: dict[str, pd.DataFrame],
    names: list[str],
    selected: set[str],
    column: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    for name in names:
        rows = curves[name]
        positions = np.unique(np.linspace(0, len(rows) - 1, min(len(rows), 350), dtype=int))
        part = rows.iloc[positions]
        ax.plot(part["epsilon_mev_fm3"], part[column], color="#cbd5e1", alpha=0.045, linewidth=0.4)
    for name in names:
        if name not in selected:
            continue
        rows = curves[name]
        positions = np.unique(np.linspace(0, len(rows) - 1, min(len(rows), 350), dtype=int))
        part = rows.iloc[positions]
        ax.plot(part["epsilon_mev_fm3"], part[column], color="#000000", alpha=0.11, linewidth=0.48)
    if column == "cs2":
        ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=0.8)
        ax.axhline(1.0 / 3.0, color="#64748b", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"Energy density $\varepsilon$ [MeV fm$^{-3}$]")
    ax.set_ylabel(ylabel)
    ax.set_title("Dry-run coverage check: all (grey) vs selected (black)")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_envelopes(
    radius: np.ndarray,
    tidal: np.ndarray,
    k2: np.ndarray,
    selected_mask: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 10.0), sharex=True)
    definitions = (
        (axes[0], radius, "R [km]", False),
        (axes[1], tidal, r"$\Lambda$", True),
        (axes[2], k2, r"$k_2$", False),
    )
    for ax, values, ylabel, log_y in definitions:
        full_min, full_max = _finite_extrema(values)
        sub_min, sub_max = _finite_extrema(values[selected_mask])
        ax.fill_between(MASS_ENVELOPE_GRID, full_min, full_max, color="#cbd5e1", alpha=0.65, label="All envelope")
        ax.plot(MASS_ENVELOPE_GRID, sub_min, color="#000000", linewidth=0.9)
        ax.plot(MASS_ENVELOPE_GRID, sub_max, color="#000000", linewidth=0.9, label="Selected envelope")
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel(r"Mass M [$M_\odot$]")
    fig.suptitle("Stellar observable envelopes: full master vs dry-run subset")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(
    root: Path,
    parent: Path,
    destination: Path,
    target_count: int,
    selection_policy: str = "diversity",
    comparison_selection: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    runs = (root / "runs").resolve(strict=True)
    catalogue_root = catalogue.confined(runs / "eos_catalogue", runs)
    parent = catalogue.confined(parent, runs)
    destination = catalogue.confined(destination, runs)
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    if destination == runs:
        raise ValueError("selection destination must be below runs/")
    catalogue.require_disjoint(destination, parent, catalogue_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if target_count != TARGET_COUNT:
        raise ValueError(f"this reviewed dry-run policy requires exactly {TARGET_COUNT} EoSs")
    if selection_policy not in {"diversity", "balanced"}:
        raise ValueError(f"unsupported selection policy: {selection_policy}")
    if comparison_selection is not None:
        comparison_selection = catalogue.confined(comparison_selection, runs)
        if not comparison_selection.is_file():
            raise FileNotFoundError(comparison_selection)
        catalogue.require_disjoint(destination, comparison_selection.parent)

    parent_manifest = _read_manifest(parent)
    provenance_path = _verified(parent, "provenance.json", parent_manifest)
    public_mapping_path = _verified(parent, "ML_DATA/eos_name_mapping.csv", parent_manifest)
    public_stellar_path = _verified(parent, "ML_DATA/hadronic_stellar_data.csv", parent_manifest)
    parent_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if parent_provenance.get("schema_id") != "bsk24_combined_hadronic_ml_dataset_v1":
        raise ValueError("parent is not a supported combined hadronic dataset")
    if int(parent_provenance.get("unique_eos_count", -1)) <= target_count:
        raise ValueError("parent does not contain more EoSs than the requested subset")

    combiner = _load_combiner(root)
    if _sha256(root / "notebooks" / "build_combined_hadronic_dataset.py") != str(
        parent_provenance.get("builder_sha256", "")
    ):
        raise ValueError("current saved-data combiner does not match the parent builder hash")
    public_mapping = pd.read_csv(public_mapping_path, low_memory=False)
    if len(public_mapping) != int(parent_provenance["unique_eos_count"]):
        raise ValueError("parent mapping row count does not match provenance")

    source_records = parent_provenance.get("source_runs")
    if not isinstance(source_records, list):
        raise ValueError("parent source-run provenance is malformed")
    source_documents = {str(item.get("run", "")): item for item in source_records}
    if not source_documents or any(
        str(item.get("source_mode")) != "sealed_curve_experiment"
        for item in source_documents.values()
    ):
        raise ValueError("this reviewed selector requires sealed curve-experiment sources")
    source_experiments = _source_experiments(root, source_records)
    catalogue.require_disjoint(destination, *source_experiments.values())
    packet_paths: list[str] = []
    case_ids: list[str] = []
    physical_keys: list[str] = []
    verified_experiments: set[str] = set()
    for row in public_mapping.to_dict(orient="records"):
        source_run = str(row["source_run"])
        source = source_documents.get(source_run)
        if source is None:
            raise ValueError(f"mapping references an undeclared source run: {source_run}")
        experiment = source_experiments[source_run]
        experiment_key = str(experiment)
        if experiment_key not in verified_experiments:
            if _sha256(experiment / "SHA256SUMS.txt") != str(
                source["experiment_manifest_sha256"]
            ):
                raise ValueError(f"source experiment manifest changed: {experiment}")
            verified_experiments.add(experiment_key)
        identity_parts = str(row["source_eos_id"]).split(":", 2)
        if len(identity_parts) != 3 or identity_parts[0] != source_run:
            raise ValueError(f"malformed source EoS identity: {row['source_eos_id']}")
        geometry_id, case_id = identity_parts[1], identity_parts[2]
        packet = _direct_child(experiment, geometry_id)
        packet_paths.append(str(packet))
        case_ids.append(case_id)
        physical_keys.append(
            combiner._direct_physical_key(
                amplitude=float(row["amplitude"]),
                epsilon_match=float(row["epsilon_match_mev_fm3"]),
                center=float(row["center_mev_fm3"]),
                width=float(row["width_mev_fm3"]),
                ramp_width=float(row["ramp_width_mev_fm3"]),
            )
        )
    internal_mapping = public_mapping.rename(
        columns={
            "source_eos_id": "eos_id",
            "center_mev_fm3": "epsilon0_mev_fm3",
            "width_mev_fm3": "sigma_mev_fm3",
            "ramp_width_mev_fm3": "delta_mev_fm3",
        }
    ).copy()
    internal_mapping["source_case_id"] = case_ids
    internal_mapping["physical_model_key"] = physical_keys
    internal_mapping["_packet"] = packet_paths
    resolved_cases = dict(zip(public_mapping["name"].astype(str), case_ids))

    stellar_table = pd.read_csv(public_stellar_path, low_memory=False)
    if set(stellar_table["name"].astype(str)) != set(public_mapping["name"].astype(str)):
        raise ValueError("parent stellar names do not match the name mapping")
    thermo_curves_list = combiner.collect_thermodynamic_curves(root, internal_mapping, resolved_cases)
    names = public_mapping["name"].astype(str).tolist()
    stellar = {name: rows.reset_index(drop=True) for name, rows in stellar_table.groupby("name", sort=False)}
    thermo = {str(curve["name"]): curve["rows"].reset_index(drop=True) for curve in thermo_curves_list}
    if set(stellar) != set(names) or set(thermo) != set(names):
        raise ValueError("saved curve collection is incomplete")

    radius_envelope = _matrix(stellar, names, "M", "R", MASS_ENVELOPE_GRID)
    lambda_envelope = _matrix(stellar, names, "M", "Lambda", MASS_ENVELOPE_GRID)
    k2_envelope = _matrix(stellar, names, "M", "k2", MASS_ENVELOPE_GRID)
    pressure_envelope = _matrix(
        thermo, names, "epsilon_mev_fm3", "pressure_mev_fm3", EPSILON_ENVELOPE_GRID
    )
    cs2_envelope = _matrix(thermo, names, "epsilon_mev_fm3", "cs2", EPSILON_ENVELOPE_GRID)
    r14 = _matrix(stellar, names, "M", "R", np.array([1.4]))[:, 0]
    if not np.isfinite(r14).all():
        raise ValueError("not every parent EoS supports R at 1.4 solar masses")

    reasons: dict[str, set[str]] = defaultdict(set)
    baseline_names = public_mapping.loc[public_mapping["regime"].eq("baseline"), "name"].astype(str).tolist()
    if len(baseline_names) != 1:
        raise ValueError("parent must contain exactly one baseline")
    reasons[baseline_names[0]].add("baseline")

    geometry_columns = [
        "regime",
        "epsilon_match_mev_fm3",
        "center_mev_fm3",
        "width_mev_fm3",
        "ramp_width_mev_fm3",
    ]
    deformed = public_mapping.loc[~public_mapping["regime"].eq("baseline")].copy()
    for _, group in deformed.groupby(geometry_columns, dropna=False, sort=False):
        amplitudes = pd.to_numeric(group["amplitude"], errors="raise")
        for label, value in (("geometry_amplitude_min", amplitudes.min()), ("geometry_amplitude_max", amplitudes.max())):
            row = group.loc[np.isclose(amplitudes, value, rtol=0.0, atol=0.0)].sort_values("name").iloc[0]
            reasons[str(row["name"])].add(label)

    r14_bins = np.floor(r14 / R14_BIN_WIDTH_KM + 1e-12).astype(int)
    for bin_index in np.unique(r14_bins):
        indices = np.flatnonzero(r14_bins == bin_index)
        center = (bin_index + 0.5) * R14_BIN_WIDTH_KM
        distances = np.abs(r14[indices] - center)
        choice = int(indices[np.argmin(distances)])
        reasons[names[choice]].add(f"r14_bin_{bin_index}")
    reasons[names[int(np.argmin(r14))]].add("r14_global_min")
    reasons[names[int(np.argmax(r14))]].add("r14_global_max")

    _add_envelope_anchors(reasons, names, radius_envelope, "radius_envelope", MASS_ENVELOPE_GRID)
    _add_envelope_anchors(reasons, names, lambda_envelope, "lambda_envelope", MASS_ENVELOPE_GRID)
    _add_envelope_anchors(reasons, names, k2_envelope, "k2_envelope", MASS_ENVELOPE_GRID)
    _add_envelope_anchors(reasons, names, pressure_envelope, "pressure_envelope", EPSILON_ENVELOPE_GRID)
    _add_envelope_anchors(reasons, names, cs2_envelope, "cs2_envelope", EPSILON_ENVELOPE_GRID)

    radius_low = _matrix(stellar, names, "M", "R", MASS_LOW_GRID)
    lambda_low = _matrix(stellar, names, "M", "Lambda", MASS_LOW_GRID)
    k2_low = _matrix(stellar, names, "M", "k2", MASS_LOW_GRID)
    radius_high = _matrix(stellar, names, "M", "R", MASS_HIGH_GRID)
    lambda_high = _matrix(stellar, names, "M", "Lambda", MASS_HIGH_GRID)
    k2_high = _matrix(stellar, names, "M", "k2", MASS_HIGH_GRID)
    pressure_features = _matrix(
        thermo, names, "epsilon_mev_fm3", "pressure_mev_fm3", EPSILON_FEATURE_GRID
    )
    cs2_features = _matrix(thermo, names, "epsilon_mev_fm3", "cs2", EPSILON_FEATURE_GRID)
    peak_mass = np.array([pd.to_numeric(stellar[name]["M"], errors="raise").max() for name in names])
    features = np.hstack(
        (
            _scalar_feature(r14, FEATURE_WEIGHTS["r14"]),
            _feature_family(radius_low, FEATURE_WEIGHTS["radius_below_2msun"]),
            _feature_family(lambda_low, FEATURE_WEIGHTS["log10_lambda_below_2msun"], "log10"),
            _feature_family(k2_low, FEATURE_WEIGHTS["k2_below_2msun"]),
            _feature_family(radius_high, FEATURE_WEIGHTS["radius_above_2msun"]),
            _feature_family(lambda_high, FEATURE_WEIGHTS["log10_lambda_above_2msun"], "log10"),
            _feature_family(k2_high, FEATURE_WEIGHTS["k2_above_2msun"]),
            _feature_family(pressure_features, FEATURE_WEIGHTS["log10_pressure"], "log10"),
            _feature_family(cs2_features, FEATURE_WEIGHTS["cs2"]),
            _scalar_feature(peak_mass, FEATURE_WEIGHTS["sampled_peak_mass"]),
        )
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("normalized diversity feature matrix is not finite")

    regimes = public_mapping["regime"].astype(str).reset_index(drop=True)
    anchor_names = set(reasons)
    anchor_mask = np.array([name in anchor_names for name in names], dtype=bool)
    mr_cell_sets = _mr_cell_sets(stellar, names, maximum_mass=3.4)
    mr_cell_sets_below_2msun = _mr_cell_sets(stellar, names, maximum_mass=2.0)
    if selection_policy == "balanced":
        selected_mask, selection_distance, selection_stage, anchor_bin_counts = _select_balanced(
            names,
            features,
            anchor_names,
            r14_bins,
            mr_cell_sets_below_2msun,
            target_count,
        )
        targets: dict[str, int] = {}
    else:
        selected_mask, targets, selection_distance = _select_diversity(
            names, regimes, features, anchor_names, target_count
        )
        selection_stage = {
            name: "diversity_fill"
            for name, selected in zip(names, selected_mask)
            if selected and name not in anchor_names
        }
        anchor_bin_counts = {
            int(bin_index): int((anchor_mask & (r14_bins == bin_index)).sum())
            for bin_index in np.unique(r14_bins)
        }
    selected_names = {name for name, selected in zip(names, selected_mask) if selected}

    full_bins = set(r14_bins.tolist())
    subset_bins = set(r14_bins[selected_mask].tolist())
    full_cells = _union_cells(mr_cell_sets)
    subset_cells = _union_cells(
        cells for cells, selected in zip(mr_cell_sets, selected_mask) if selected
    )
    full_cells_below_2msun = _union_cells(mr_cell_sets_below_2msun)
    subset_cells_below_2msun = _union_cells(
        cells for cells, selected in zip(mr_cell_sets_below_2msun, selected_mask) if selected
    )
    geometry_keys = deformed[geometry_columns].astype(str).agg("|".join, axis=1)
    selected_deformed = public_mapping.loc[selected_mask & ~public_mapping["regime"].eq("baseline")]
    selected_geometry_keys = selected_deformed[geometry_columns].astype(str).agg("|".join, axis=1)
    density = _density_metrics(r14_bins, selected_mask, anchor_mask)

    comparison: dict[str, Any] = {}
    if comparison_selection is not None:
        comparison_table = pd.read_csv(comparison_selection, usecols=["name"])
        comparison_names = comparison_table["name"].astype(str).tolist()
        if len(comparison_names) != target_count or len(set(comparison_names)) != target_count:
            raise ValueError("comparison selection does not contain exactly 2,000 unique names")
        unknown = sorted(set(comparison_names) - set(names))
        if unknown:
            raise ValueError(f"comparison selection references unknown parent names: {unknown[:10]}")
        comparison_mask = np.array([name in set(comparison_names) for name in names], dtype=bool)
        comparison_low_cells = _union_cells(
            cells
            for cells, selected in zip(mr_cell_sets_below_2msun, comparison_mask)
            if selected
        )
        comparison = {
            "selection_path": str(comparison_selection),
            "r14_density": _density_metrics(r14_bins, comparison_mask, anchor_mask),
            "mr_raster_below_2msun_retention_fraction": float(
                len(full_cells_below_2msun & comparison_low_cells)
                / len(full_cells_below_2msun)
            ),
            "r14": _gap_metrics(r14[comparison_mask]),
        }

    metrics = {
        "selection_policy": selection_policy,
        "parent_eos_count": int(len(names)),
        "selected_eos_count": int(selected_mask.sum()),
        "mandatory_anchor_count": int(len(anchor_names)),
        "diversity_fill_count": int(selected_mask.sum() - len(anchor_names)),
        "selected_regime_counts": {
            str(key): int(value) for key, value in regimes.loc[selected_mask].value_counts().items()
        },
        "r14_density": density,
        "r14_full": _gap_metrics(r14),
        "r14_selected": _gap_metrics(r14[selected_mask]),
        "r14_occupied_0p01km_bins_full": int(len(full_bins)),
        "r14_occupied_0p01km_bins_selected": int(len(subset_bins)),
        "r14_bin_retention_fraction": float(len(full_bins & subset_bins) / len(full_bins)),
        "geometry_group_count_full": int(geometry_keys.nunique()),
        "geometry_group_count_selected": int(selected_geometry_keys.nunique()),
        "mr_raster_cells_full": int(len(full_cells)),
        "mr_raster_cells_selected": int(len(subset_cells)),
        "mr_raster_cell_retention_fraction": float(len(full_cells & subset_cells) / len(full_cells)),
        "mr_raster_cells_below_2msun_full": int(len(full_cells_below_2msun)),
        "mr_raster_cells_below_2msun_selected": int(len(subset_cells_below_2msun)),
        "mr_raster_below_2msun_retention_fraction": float(
            len(full_cells_below_2msun & subset_cells_below_2msun)
            / len(full_cells_below_2msun)
        ),
        "stellar_envelope_error": {
            "radius_km": _envelope_error(radius_envelope, radius_envelope[selected_mask]),
            "log10_lambda": _envelope_error(lambda_envelope, lambda_envelope[selected_mask], log10=True),
            "k2": _envelope_error(k2_envelope, k2_envelope[selected_mask]),
        },
        "thermodynamic_envelope_error": {
            "log10_pressure": _envelope_error(pressure_envelope, pressure_envelope[selected_mask], log10=True),
            "cs2": _envelope_error(cs2_envelope, cs2_envelope[selected_mask]),
        },
        "normalized_feature_covering_radius": float(selection_distance["__covering_radius__"]),
        "comparison": comparison,
    }
    if targets:
        metrics["target_regime_counts"] = targets
    gates = {
        "exactly_2000_whole_eos": int(selected_mask.sum()) == target_count,
        "baseline_retained": baseline_names[0] in selected_names,
        "all_r14_bins_retained": full_bins == subset_bins,
        "r14_global_extrema_retained": bool(
            math.isclose(float(r14[selected_mask].min()), float(r14.min()), rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(r14[selected_mask].max()), float(r14.max()), rel_tol=0.0, abs_tol=1e-12)
        ),
        "r14_largest_gap_at_most_0p03km": metrics["r14_selected"]["largest_adjacent_gap_km"] <= 0.03,
        "all_geometry_groups_retained": geometry_keys.nunique() == selected_geometry_keys.nunique(),
        "all_feasible_r14_bins_have_at_least_three": density[
            "eligible_bins_below_three_selected"
        ] == 0,
        "all_sparse_r14_bins_fully_retained": density[
            "sparse_bins_not_fully_retained"
        ] == 0,
        "mr_raster_below_2msun_at_least_0p99": metrics[
            "mr_raster_below_2msun_retention_fraction"
        ] >= 0.99,
        "stellar_grid_envelopes_exact": all(
            value <= 1e-12
            for family in metrics["stellar_envelope_error"].values()
            for value in family.values()
        ),
        "thermodynamic_grid_envelopes_exact": all(
            value <= 1e-12
            for family in metrics["thermodynamic_envelope_error"].values()
            for value in family.values()
        ),
        "no_solver_calls": True,
        "authoritative_packets_unchanged": True,
    }
    if comparison:
        gates["r14_density_cv_improved_over_comparison"] = density[
            "selected_count_coefficient_of_variation"
        ] < comparison["r14_density"]["selected_count_coefficient_of_variation"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".subset_2000_dryrun_", dir=destination.parent))
    try:
        plots = stage / "plots"
        plots.mkdir()
        selected_rank = {name: rank for rank, name in enumerate((name for name in names if name in selected_names), start=1)}
        manifest = public_mapping.copy()
        manifest.insert(1, "selected", selected_mask)
        manifest.insert(2, "selected_rank", [selected_rank.get(name, "") for name in names])
        manifest.insert(
            3,
            "selection_stage",
            [
                "mandatory_anchor"
                if name in anchor_names
                else selection_stage.get(name, "not_selected")
                for name in names
            ],
        )
        manifest.insert(4, "selection_reasons", [";".join(sorted(reasons.get(name, ()))) for name in names])
        manifest.insert(5, "diversity_distance_at_selection", [selection_distance.get(name, "") for name in names])
        manifest.insert(6, "r14_km", r14)
        manifest["physical_model_key"] = internal_mapping["physical_model_key"].astype(str).to_numpy()
        manifest.to_csv(stage / "selection_manifest.csv", index=False, float_format="%.17g", lineterminator="\n")
        manifest.loc[manifest["selected"]].to_csv(
            stage / "selected_eos_2000.csv", index=False, float_format="%.17g", lineterminator="\n"
        )
        (stage / "coverage_metrics.json").write_bytes(_canonical(metrics) + b"\n")
        (stage / "validation_gates.json").write_bytes(_canonical(gates) + b"\n")
        _plot_mr(stellar, names, selected_names, plots / "qa_mass_radius_overlay.png")
        _plot_r14(r14, names, selected_names, plots / "qa_r14_coverage.png")
        _plot_thermo(
            thermo, names, selected_names, "pressure_mev_fm3",
            r"Pressure P [MeV fm$^{-3}$]", plots / "qa_pressure_overlay.png"
        )
        _plot_thermo(
            thermo, names, selected_names, "cs2",
            r"Dimensionless sound speed squared $c_s^2$", plots / "qa_speed_of_sound_overlay.png"
        )
        _plot_envelopes(
            radius_envelope, lambda_envelope, k2_envelope, selected_mask,
            plots / "qa_stellar_envelopes.png"
        )
        provenance = {
            "schema_id": SCHEMA_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "parent_dataset": parent.relative_to(root).as_posix(),
            "parent_manifest_sha256": _sha256(parent / "SHA256SUMS.txt"),
            "parent_provenance_sha256": _sha256(provenance_path),
            "target_eos_count": target_count,
            "selection_policy": (
                "whole-EoS mandatory anchors (baseline, per-geometry amplitude endpoints, "
                "every occupied 0.01-km R1.4 bin, exact saved-data stellar and thermodynamic "
                "grid envelopes), followed by deterministic R1.4-bin water-filling and low-mass "
                "M-R raster/feature-diversity rescue"
                if selection_policy == "balanced"
                else "whole-EoS mandatory anchors followed by deterministic proportional-regime "
                "farthest-point coverage"
            ),
            "feature_weights": FEATURE_WEIGHTS,
            "mass_envelope_grid_msun": MASS_ENVELOPE_GRID.tolist(),
            "epsilon_envelope_grid_mev_fm3": EPSILON_ENVELOPE_GRID.tolist(),
            "source_runs": parent_provenance["source_runs"],
            "duplicate_physical_occurrence_count": int(
                parent_provenance["duplicate_physical_occurrence_count"]
            ),
            "solver_calls": 0,
            "authoritative_packets_modified": False,
            "dry_run_only": True,
            "selector_sha256": _sha256(Path(__file__)),
            "all_validation_gates_passed": bool(all(gates.values())),
        }
        (stage / "provenance.json").write_bytes(_canonical(provenance) + b"\n")
        (stage / "README.md").write_text(
            "# BSk24 2,000-EoS coverage-preserving selection — dry run\n\n"
            "This directory is a non-authoritative selection report. The immutable 3,124-EoS "
            "parent and all sealed source packets were read and checksum-verified but not changed. "
            "No thermodynamic, TOV, or tidal solver was called. `selected_eos_2000.csv` identifies "
            "complete EoSs proposed for the derivative packet; it is not a row-level curve sample.\n\n"
            "Review `coverage_metrics.json`, `validation_gates.json`, and all QA plots before "
            "materializing a new combined dataset.\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_manifest(stage)
        catalogue.publish_directory(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "destination": str(destination),
        "metrics": metrics,
        "validation_gates": gates,
        "all_validation_gates_passed": bool(all(gates.values())),
        "manifest_sha256": _sha256(destination / "SHA256SUMS.txt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument(
        "--selection-policy",
        choices=("diversity", "balanced"),
        default="diversity",
    )
    parser.add_argument("--comparison-selection", type=Path)
    args = parser.parse_args()
    root = catalogue.trusted_repository_root(args.repository_root)
    result = run(
        root,
        args.parent,
        args.destination,
        args.target_count,
        selection_policy=args.selection_policy,
        comparison_selection=args.comparison_selection,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

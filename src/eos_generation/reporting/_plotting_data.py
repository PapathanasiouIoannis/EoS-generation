"""Saved-table data, validation masks, and reporting context for plotting."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eos_generation._internal.saved_tables import saved_tidal_valid_mask
from eos_generation.reporting.plot_style import resolve_saved_case_style_rows
from eos_generation.stellar.tov import LAMBDA_FRAMEWORK_CAPABILITY

_ACTIVE_TABLE_CACHE: ContextVar[
    dict[tuple[Path, str], pd.DataFrame] | None
] = ContextVar("bsk24_saved_plot_table_cache", default=None)


def _table(packet: Path, name: str) -> pd.DataFrame:
    path = packet / name
    cache = _ACTIVE_TABLE_CACHE.get()
    key = (packet, name)
    if cache is not None and key in cache:
        return cache[key]
    frame = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if cache is not None:
        cache[key] = frame
    return frame


def _effective_plot_epsilon_match(config: Any) -> float:
    """Return the governed reconstruction/window anchor used by one packet."""

    value = getattr(config, "effective_epsilon_match_mev_fm3", None)
    if value is not None:
        return float(value)
    selected = getattr(config, "epsilon_match_mev_fm3", None)
    if selected is not None:
        return float(selected)
    from eos_generation.bsk24.deformation import (
        BSK24_RETAINED_EPSILON_MATCH_MEV_FM3,
    )

    return float(BSK24_RETAINED_EPSILON_MATCH_MEV_FM3)


def _validated_tidal_mask(frame: pd.DataFrame) -> pd.Series:
    """Select only complete, explicitly validated tidal rows.

    A finite value is not sufficient evidence of tidal validity: the exact
    per-calculation repository status is authoritative.  Requiring both
    observables to be finite also prevents either tidal panel from presenting
    a partially populated row.
    """

    return saved_tidal_valid_mask(frame, schema="sequence")


def _validated_fixed_tidal_mask(frame: pd.DataFrame) -> pd.Series:
    """Fixed-mass equivalent of :func:`_validated_tidal_mask`."""

    return saved_tidal_valid_mask(frame, schema="fixed_mass")


def _contiguous_valid_runs(
    frame: pd.DataFrame,
    valid: pd.Series,
    *,
    order_column: str | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Return valid runs without bridging an invalid or missing attempted row."""

    if frame.empty:
        return ()
    ordered = (
        frame.sort_values(order_column, kind="stable")
        if order_column is not None and order_column in frame.columns
        else frame.copy()
    )
    ordered_valid = valid.reindex(ordered.index, fill_value=False).to_numpy(dtype=bool)
    order_values = (
        pd.to_numeric(ordered[order_column], errors="coerce").to_numpy(dtype=float)
        if order_column is not None and order_column in ordered.columns
        else np.arange(len(ordered), dtype=float)
    )
    runs: list[pd.DataFrame] = []
    start: int | None = None
    for position, is_valid in enumerate(ordered_valid):
        continues = (
            start is not None
            and position > 0
            and is_valid
            and ordered_valid[position - 1]
            and np.isfinite(order_values[position])
            and np.isfinite(order_values[position - 1])
            and order_values[position] == order_values[position - 1] + 1.0
        )
        if is_valid and start is None:
            start = position
        elif is_valid and not continues:
            if start is not None:
                runs.append(ordered.iloc[start:position])
            start = position
        elif not is_valid and start is not None:
            runs.append(ordered.iloc[start:position])
            start = None
    if start is not None:
        runs.append(ordered.iloc[start:])
    return tuple(run for run in runs if not run.empty)


def _validated_tidal_runs(frame: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Validated sequence runs split at every failed attempted index."""

    return _contiguous_valid_runs(
        frame,
        _validated_tidal_mask(frame),
        order_column="attempted_index",
    )


def _summarize_stellar_tidal_plot_completeness(
    sequence_rows: pd.DataFrame,
    fixed_mass_rows: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize what the M-R/tidal figure can truthfully display."""

    sequence_background = (
        sequence_rows.loc[sequence_rows["calculation_status"].eq("success")]
        if "calculation_status" in sequence_rows.columns
        else sequence_rows.iloc[0:0]
    )
    fixed_background = (
        fixed_mass_rows.loc[fixed_mass_rows["status"].eq("bracketed_and_solved")]
        if "status" in fixed_mass_rows.columns
        else fixed_mass_rows.iloc[0:0]
    )
    sequence_valid = int(_validated_tidal_mask(sequence_background).sum())
    fixed_valid = int(_validated_fixed_tidal_mask(fixed_background).sum())
    sequence_background_count = int(len(sequence_background))
    fixed_background_count = int(len(fixed_background))
    sequence_omitted = sequence_background_count - sequence_valid
    fixed_omitted = fixed_background_count - fixed_valid
    total_background = sequence_background_count + fixed_background_count
    total_valid = sequence_valid + fixed_valid
    total_omitted = sequence_omitted + fixed_omitted
    if total_background == 0:
        status = "unavailable_no_successful_backgrounds"
    elif total_omitted == 0:
        status = "complete_background_and_tidal"
    elif total_valid:
        status = "partial_tidal_data"
    else:
        status = "background_only_no_validated_tides"
    return {
        "status": status,
        "tidal_valid_status_required": LAMBDA_FRAMEWORK_CAPABILITY,
        "sequence_background_success_count": sequence_background_count,
        "sequence_tidal_validated_count": sequence_valid,
        "sequence_tidal_omitted_count": sequence_omitted,
        "fixed_mass_background_success_count": fixed_background_count,
        "fixed_mass_tidal_validated_count": fixed_valid,
        "fixed_mass_tidal_omitted_count": fixed_omitted,
    }


_ACTIVE_REPORTING_CONTEXT: ContextVar[Mapping[str, str] | None] = ContextVar(
    "bsk24_plot_reporting_context", default=None
)


def _successful_sequence_runs(frame: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Successful M-R runs split at every failed attempted index."""

    if not {
        "calculation_status",
        "Mass",
        "Radius",
    }.issubset(frame.columns):
        return ()
    mass = pd.to_numeric(frame["Mass"], errors="coerce")
    radius = pd.to_numeric(frame["Radius"], errors="coerce")
    valid = (
        frame["calculation_status"].astype(str).eq("success")
        & np.isfinite(mass)
        & np.isfinite(radius)
    )
    return _contiguous_valid_runs(
        frame,
        valid,
        order_column="attempted_index",
    )


def _reindex_saved_mass_grid(
    frame: pd.DataFrame,
    masses: Iterable[float],
    *,
    mass_column: str = "target_mass_msun",
) -> pd.DataFrame:
    """Retain every configured fixed-mass coordinate as a plot gap if absent."""

    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result[mass_column] = pd.to_numeric(
        result[mass_column], errors="coerce"
    )
    result = result.loc[np.isfinite(result[mass_column])].copy()
    if bool(result.duplicated(mass_column, keep=False).any()):
        raise ValueError("saved fixed-mass plot rows must be unique by mass")
    coordinates = np.asarray(tuple(float(value) for value in masses), dtype=float)
    return (
        result.set_index(mass_column)
        .reindex(coordinates)
        .rename_axis(mass_column)
        .reset_index()
    )


def _saved_case_style_rows(packet: Path, frame: pd.DataFrame) -> pd.DataFrame:
    has_deformed = any(
        str(value) != "direct"
        for value in frame.get("case_id", pd.Series(dtype=str)).dropna()
    )
    screening = pd.DataFrame()
    if has_deformed:
        screening_path = packet / "screening_results.csv"
        screening = _table(packet, screening_path.name)
        if not screening_path.is_file():
            # Normalized-q packets require screening_results.csv because A is
            # resolved from q at execution time.  Raw-A packets do not create
            # that table; their authoritative saved A/Delta coordinates live
            # in case_ledger.csv instead.
            screening = _table(packet, "case_ledger.csv")
    return resolve_saved_case_style_rows(frame, screening)


def _saved_plot_reporting_context(packet: Path, config) -> dict[str, str]:
    """Build concise labels from packet metadata without inventing evidence."""

    metadata: Mapping[str, Any] = {}
    metadata_path = packet / "metadata.json"
    if metadata_path.is_file():
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise TypeError("metadata.json must contain a JSON object")
        metadata = loaded

    configuration_hash = metadata.get("configuration_hash")
    if not isinstance(configuration_hash, str) or len(configuration_hash) != 64:
        hash_method = getattr(config, "deterministic_hash", None)
        configuration_hash = hash_method() if callable(hash_method) else "unavailable"

    accepted = metadata.get("accepted_case_ids", [])
    rejected = metadata.get("rejected_case_ids", [])
    accepted_ids = [str(value) for value in accepted] if isinstance(accepted, list) else []
    rejected_ids = [str(value) for value in rejected] if isinstance(rejected, list) else []
    case_ids = list(dict.fromkeys((*accepted_ids, *rejected_ids)))
    if not case_ids:
        case_label = "unavailable"
    elif len(case_ids) <= 3:
        case_label = ",".join(case_ids)
    else:
        case_label = f"{case_ids[0]},{case_ids[1]},+{len(case_ids) - 2}"

    if accepted_ids and rejected_ids:
        outcome = "mixed"
    elif rejected_ids:
        outcome = "rejected"
    elif accepted_ids:
        outcome = "accepted"
    else:
        outcome = "unavailable"

    stages: list[str] = []
    thermodynamic = tuple(getattr(config, "thermodynamic_stages", ()))
    if thermodynamic:
        stages.append(f"thermo:{thermodynamic[-1].name}")
    if bool(getattr(config, "background_tov_requested", False)):
        stellar = tuple(getattr(config, "tov_stages", ()))
        if stellar:
            stages.append(f"stellar:{stellar[-1].name}")
    return {
        "configuration_hash": configuration_hash[:12],
        "case_ids": case_label,
        "outcome": outcome,
        "stage": "/".join(stages) if stages else "saved",
    }

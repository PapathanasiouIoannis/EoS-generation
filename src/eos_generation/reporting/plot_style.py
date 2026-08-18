"""Fail-closed saved-coordinate resolution for BSk24 plot styling."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def resolve_saved_case_style_rows(
    frame: pd.DataFrame,
    screening: pd.DataFrame,
) -> pd.DataFrame:
    """Return one saved amplitude/Delta pair per plotted physical case.

    Stellar sequence tables intentionally contain only sequence quantities.
    Normalized-q amplitudes are geometry-dependent and must therefore come
    from the saved resolved screening table, never from the declared input.
    """

    case_ids = tuple(
        dict.fromkeys(
            str(value)
            for value in frame.get("case_id", pd.Series(dtype=str)).dropna()
        )
    )
    deformed = tuple(case_id for case_id in case_ids if case_id != "direct")
    if not deformed:
        return pd.DataFrame(
            {
                "case_id": ["direct"],
                "amplitude": [np.nan],
                "delta_mev_fm3": [np.nan],
            }
        )

    if {"amplitude", "delta_mev_fm3"}.issubset(frame.columns):
        inline = frame.loc[frame.case_id.astype(str).isin(deformed)].copy()
        amplitudes = pd.to_numeric(inline["amplitude"], errors="coerce")
        deltas = pd.to_numeric(inline["delta_mev_fm3"], errors="coerce")
        if bool(np.isfinite(amplitudes).all() and np.isfinite(deltas).all()):
            return frame

    required = {"amplitude", "delta_mev_fm3"}
    if not required.issubset(screening.columns):
        missing = ", ".join(sorted(required - set(screening.columns)))
        raise ValueError(
            "screening_results.csv cannot label stellar cases; missing " + missing
        )
    identifier = (
        "physical_case_id"
        if "physical_case_id" in screening.columns
        else "case_id"
    )
    resolved: list[dict[str, float | str]] = [
        {"case_id": "direct", "amplitude": np.nan, "delta_mev_fm3": np.nan}
    ]
    for case_id in deformed:
        matches = screening.loc[screening[identifier].astype(str) == case_id]
        coordinates = {
            (float(amplitude), float(delta))
            for amplitude, delta in zip(
                pd.to_numeric(matches["amplitude"], errors="coerce"),
                pd.to_numeric(matches["delta_mev_fm3"], errors="coerce"),
            )
            if math.isfinite(float(amplitude)) and math.isfinite(float(delta))
        }
        if len(coordinates) != 1:
            raise ValueError(
                "screening_results.csv must provide exactly one finite amplitude/"
                f"Delta pair for stellar case {case_id!r}; found {len(coordinates)}"
            )
        amplitude, delta = next(iter(coordinates))
        resolved.append(
            {
                "case_id": case_id,
                "amplitude": amplitude,
                "delta_mev_fm3": delta,
            }
        )
    return pd.DataFrame(resolved)


__all__ = ["resolve_saved_case_style_rows"]

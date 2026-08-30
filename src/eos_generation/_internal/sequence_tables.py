"""Stellar sequence-table serialization for BSk24 trials."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from eos_generation.stellar.tov import TOV_SEQUENCE_FIELDS, TovSequenceEvidence


def _tidal_jump_evidence_columns(diagnostic: Any) -> dict[str, Any]:
    """Return strict, table-safe discontinuity evidence for one tidal solve."""

    surface_jumps = tuple(
        item for item in diagnostic.applied_jumps if item.kind == "surface"
    )
    surface = surface_jumps[0] if len(surface_jumps) == 1 else None
    payload = diagnostic.to_dict()
    return {
        "tidal_expected_jump_count": diagnostic.expected_jump_count,
        "tidal_applied_jump_count": diagnostic.applied_jump_count,
        "tidal_surface_jump_count": len(surface_jumps),
        "tidal_surface_delta_y": None if surface is None else surface.delta_y,
        "tidal_surface_y_before": None if surface is None else surface.y_before,
        "tidal_surface_y_after": None if surface is None else surface.y_after,
        "tidal_surface_event_pressure_mev_fm3": (
            diagnostic.surface_event_pressure
        ),
        "tidal_jump_evidence_json": json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }


def _sequence_frame(
    case_id: str,
    stage: str,
    evidence: TovSequenceEvidence,
    *,
    retain_jump_evidence: bool = False,
) -> pd.DataFrame:
    success = list(evidence.full_sequence)
    diagnostics = list(evidence.full_lambda_diagnostics or ())
    successful_by_pressure = {
        float(row[3]): (row, i) for i, row in enumerate(success)
    }
    failures = {
        float(item.central_pressure): item
        for item in evidence.failed_central_pressures
    }
    rows: list[dict[str, Any]] = []
    segment = 0
    for attempted_index, pressure in enumerate(
        evidence.attempted_central_pressures
    ):
        pressure = float(pressure)
        if pressure in failures:
            failure = failures[pressure]
            rows.append(
                {
                    "case_id": case_id,
                    "stage": stage,
                    "attempted_index": attempted_index,
                    "segment_id": segment,
                    "calculation_status": "failed",
                    "failure_category": failure.category,
                    "failure_reason": failure.reason,
                    "central_pressure_mev_fm3": pressure,
                }
            )
            segment += 1
            continue
        if pressure not in successful_by_pressure:
            continue
        row, success_index = successful_by_pressure[pressure]
        record = {
            "case_id": case_id,
            "stage": stage,
            "attempted_index": attempted_index,
            "segment_id": segment,
            "calculation_status": "success",
            "failure_category": None,
            "failure_reason": None,
            **{
                field: float(value)
                for field, value in zip(TOV_SEQUENCE_FIELDS, row)
            },
            "central_pressure_mev_fm3": float(row[3]),
            "is_sampled_peak": (
                success_index == evidence.sampled_peak_index
            ),
            "is_domain_end": success_index == len(success) - 1,
        }
        if success_index < len(diagnostics):
            diagnostic = diagnostics[success_index]
            record.update(
                {
                    "k2": diagnostic.k2,
                    "tidal_status": diagnostic.scientific_status,
                    "tidal_failure_reason": diagnostic.failure_reason,
                }
            )
            if retain_jump_evidence:
                record.update(_tidal_jump_evidence_columns(diagnostic))
        rows.append(record)
    return pd.DataFrame(rows)


__all__ = ["_sequence_frame", "_tidal_jump_evidence_columns"]

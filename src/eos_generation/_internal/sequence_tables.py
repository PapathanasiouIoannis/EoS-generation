"""Stellar sequence-table serialization for BSk24 trials."""

from __future__ import annotations

from typing import Any

import pandas as pd

from eos_generation.stellar.tov import TOV_SEQUENCE_FIELDS, TovSequenceEvidence


def _sequence_frame(
    case_id: str,
    stage: str,
    evidence: TovSequenceEvidence,
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
        rows.append(record)
    return pd.DataFrame(rows)


__all__ = ["_sequence_frame"]

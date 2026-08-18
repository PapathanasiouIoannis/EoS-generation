"""Case-lifecycle reporting for governed BSk24 trials.

This internal module preserves the established saved-stellar completeness
predicate and the final accepted/rejected lifecycle artifacts.  It is
deliberately independent of the public experiment facade and scientific
solvers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from eos_generation._internal.artifacts import write_csv_atomic, write_json_atomic
from eos_generation._internal.planning import (
    BSk24TrialConfig,
    BSk24TrialPlan,
    _json_records,
)
from eos_generation._internal.saved_tables import (
    saved_tidal_valid_mask,
)


def _completed_stellar_case_ids(
    sequences: pd.DataFrame | None,
    fixed: pd.DataFrame | None,
    config: BSk24TrialConfig,
    *,
    accepted_case_ids: Sequence[str],
) -> set[str]:
    """Return accepted cases with complete, valid saved stellar evidence."""

    if (
        not config.background_tov_requested
        or sequences is None
        or fixed is None
        or not config.tov_stages
    ):
        return set()
    completed: set[str] = set()
    for case_id in accepted_case_ids:
        case_sequence = sequences.loc[
            sequences["case_id"].astype(str).eq(str(case_id))
        ]
        sequence_complete = True
        for stage in config.tov_stages:
            stage_rows = case_sequence.loc[
                case_sequence["stage"].astype(str).eq(stage.name)
            ]
            stage_complete = bool(
                len(stage_rows) == stage.sequence_points
                and saved_tidal_valid_mask(
                    stage_rows, schema="sequence"
                ).all()
            )
            if not stage_complete:
                sequence_complete = False
                break

        if not config.fixed_mass_background_requested:
            fixed_complete = True
        else:
            case_fixed = fixed.loc[
                fixed["case_id"].astype(str).eq(str(case_id))
            ]
            expected_fixed = len(config.tov_stages) * len(
                config.fixed_masses_msun
            )
            background_complete = bool(
                len(case_fixed) == expected_fixed
                and case_fixed["status"].astype(str).eq(
                    "bracketed_and_solved"
                ).all()
            )
            fixed_complete = bool(
                background_complete
                and saved_tidal_valid_mask(
                    case_fixed, schema="fixed_mass"
                ).all()
            )
        if sequence_complete and fixed_complete:
            completed.add(str(case_id))
    return completed


def _case_lifecycle_ledger(
    plan: BSk24TrialPlan,
    *,
    accepted_case_ids: Sequence[str],
    gate_reports: Mapping[str, Mapping[str, Any]],
    completed_stellar_case_ids: set[str],
) -> pd.DataFrame:
    """Build truthful final lifecycle rows for accepted and rejected cases."""

    accepted_set = set(accepted_case_ids)
    rows: list[dict[str, Any]] = []
    for row in plan.case_table.itertuples(index=False):
        case_id = str(row.case_id)
        physical_case_id = str(
            getattr(row, "physical_case_id", case_id)
        )
        accepted = case_id in accepted_set
        gate_report = gate_reports[case_id]
        failure = gate_report.get("first_failure")
        gate_status = gate_report.get("status")
        if accepted and gate_status not in {
            None,
            "accepted",
            "accepted_raw_local_physics_gate",
            "accepted_exact_baseline_alias",
        }:
            raise ValueError(
                f"accepted case {case_id!r} did not pass the full-domain raw gate"
            )
        if not accepted:
            stellar_calculation = "skipped_due_to_raw_gate_rejection"
        elif not plan.config.background_tov_requested:
            stellar_calculation = "disabled"
        elif physical_case_id in completed_stellar_case_ids:
            stellar_calculation = "completed"
        else:
            stellar_calculation = "incomplete_or_failed"
        record = {
                "case_id": case_id,
                "amplitude": row.amplitude,
                "epsilon_match_mev_fm3": getattr(
                    row,
                    "epsilon_match_mev_fm3",
                    plan.config.effective_epsilon_match_mev_fm3,
                ),
                "anchor_mode": getattr(
                    row,
                    "anchor_mode",
                    "exploratory"
                    if plan.config.exploratory_anchor_requested
                    else "standard",
                ),
                "epsilon0_mev_fm3": row.epsilon0_mev_fm3,
                "sigma_mev_fm3": row.sigma_mev_fm3,
                "delta_mev_fm3": row.delta_mev_fm3,
                "status": "accepted" if accepted else "rejected",
                "acceptance_domain": (
                    "full_retained_domain" if accepted else "none"
                ),
                "full_domain_gate_status": gate_report.get("status"),
                "selected_domain_status": gate_report.get(
                    "screening_status", gate_report.get("status")
                ),
                "rejection_reason": (
                    None
                    if accepted
                    else json.dumps(failure, sort_keys=True)
                ),
                "pressure_reconstruction": (
                    "completed"
                    if accepted
                    else "skipped_due_to_raw_gate_rejection"
                ),
                "stellar_calculation": stellar_calculation,
                "clipping_or_repair": "none",
            }
        rows.append(record)
    return pd.DataFrame(rows)


def _write_case_lifecycle(
    packet: Path,
    ledger: pd.DataFrame,
) -> None:
    write_csv_atomic(ledger, packet / "case_ledger.csv")
    write_json_atomic(
        {
            "accepted": _json_records(
                ledger.loc[ledger["status"].eq("accepted")]
            ),
            "rejected": _json_records(
                ledger.loc[ledger["status"].eq("rejected")]
            ),
            "rejected_cases_received_no_reconstruction_or_stellar_work": True,
        },
        packet / "accepted_rejected_cases.json",
    )


__all__ = [
    "_case_lifecycle_ledger",
    "_completed_stellar_case_ids",
    "_write_case_lifecycle",
]

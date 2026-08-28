"""Case-lifecycle reporting for governed BSk24 trials.

This internal module preserves the established saved-stellar completeness
predicate and the final accepted/rejected lifecycle artifacts.  It is
deliberately independent of the public experiment facade and scientific
solvers.
"""

from __future__ import annotations

import json
import math
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


def _zero_amplitude_physical_case_id(config: Any) -> str | None:
    value = getattr(config, "zero_amplitude_physical_case_id", None)
    if isinstance(value, str) and value:
        return value
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        saved = to_dict().get("zero_amplitude_physical_case_id")
        if isinstance(saved, str) and saved:
            return saved
    return None


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
        saved_case_id = str(case_id)
        if saved_case_id == _zero_amplitude_physical_case_id(config):
            saved_case_id = "direct"
        case_sequence = sequences.loc[
            sequences["case_id"].astype(str).eq(saved_case_id)
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
                fixed["case_id"].astype(str).eq(saved_case_id)
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
    fixed_mass_rows: pd.DataFrame | None = None,
    maximum_mass_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build truthful final lifecycle rows for accepted and rejected cases."""

    accepted_set = set(accepted_case_ids)
    rows: list[dict[str, Any]] = []
    for row in plan.case_table.itertuples(index=False):
        case_id = str(row.case_id)
        physical_case_id = str(
            getattr(row, "physical_case_id", case_id)
        )
        accepted = case_id in accepted_set or physical_case_id in accepted_set
        gate_report = gate_reports.get(
            physical_case_id, gate_reports.get(case_id)
        )
        if gate_report is None:
            raise KeyError(
                f"missing raw-gate evidence for logical case {case_id!r} "
                f"and physical case {physical_case_id!r}"
            )
        failure = gate_report.get("first_failure")
        gate_status = gate_report.get("status")
        if accepted and gate_status != "accepted_raw_local_physics_gate":
            raise ValueError(
                f"accepted case {case_id!r} did not pass the selected-domain raw gate"
            )
        retained = gate_report.get("retained_domain")
        retained = retained if isinstance(retained, Mapping) else {}
        endpoint_reason = retained.get("endpoint_reason")
        if accepted and endpoint_reason not in {
            "direct_bsk24_causal_endpoint",
            "published_bsk24_fit_endpoint",
            "first_continuous_causal_crossing",
            "formula_derived_cfl_domain_endpoint",
        }:
            raise ValueError(
                f"accepted case {case_id!r} has no resolved retained endpoint"
            )
        retained_epsilon = retained.get("epsilon_max_mev_fm3")
        retained_pressure = retained.get("pressure_max_mev_fm3")
        if accepted and (
            not isinstance(retained_epsilon, (int, float))
            or isinstance(retained_epsilon, bool)
            or not math.isfinite(float(retained_epsilon))
            or not isinstance(retained_pressure, (int, float))
            or isinstance(retained_pressure, bool)
            or not math.isfinite(float(retained_pressure))
        ):
            raise ValueError(
                f"accepted case {case_id!r} has an invalid retained endpoint"
            )
        complete_raw_causal = gate_report.get(
            "complete_raw_proposal_causal_through_direct_endpoint"
        )
        if gate_status == "accepted_raw_local_physics_gate":
            if not isinstance(complete_raw_causal, bool):
                raise ValueError(
                    f"accepted case {case_id!r} has no complete-domain causal status"
                )
            full_domain_status = (
                "assessed_causal_through_direct_endpoint"
                if complete_raw_causal
                else "assessed_noncausal_beyond_first_retained_crossing"
            )
            selected_domain_status = "accepted_selected_retained_domain"
        elif gate_status == "rejected_raw_local_physics_gate":
            full_domain_status = "assessed_hard_rejected"
            selected_domain_status = "rejected_no_selected_retained_domain"
        elif gate_status == "unresolved_raw_local_physics_gate":
            full_domain_status = "assessed_unresolved"
            selected_domain_status = "unresolved_no_selected_retained_domain"
        else:
            raise ValueError(
                f"case {case_id!r} has unsupported raw-gate status {gate_status!r}"
            )
        if not accepted:
            stellar_calculation = "skipped_due_to_raw_gate_rejection"
        elif not plan.config.background_tov_requested:
            stellar_calculation = "disabled"
        elif physical_case_id in completed_stellar_case_ids:
            stellar_calculation = "completed"
        else:
            stellar_calculation = "incomplete_or_failed"
        saved_stellar_case_id = physical_case_id
        if physical_case_id == _zero_amplitude_physical_case_id(plan.config):
            saved_stellar_case_id = "direct"
        fixed_mass_status = _requested_fixed_masses_status(
            plan.config,
            saved_stellar_case_id,
            accepted=accepted,
            fixed_mass_rows=fixed_mass_rows,
        )
        maximum_mass_status = _maximum_mass_availability_status(
            plan.config,
            saved_stellar_case_id,
            accepted=accepted,
            maximum_mass_rows=maximum_mass_rows,
        )
        if not accepted:
            student_view_status = "evidence_only_raw_gate_not_accepted"
        elif not plan.config.background_tov_requested:
            student_view_status = "eligible_thermodynamic_case"
        elif fixed_mass_status == "all_requested_fixed_masses_succeeded":
            student_view_status = (
                "eligible_all_requested_fixed_masses_succeeded"
            )
        else:
            student_view_status = (
                "ineligible_requested_fixed_masses_incomplete"
            )
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
                    (
                        "through_first_continuous_causal_crossing"
                        if endpoint_reason
                        == "first_continuous_causal_crossing"
                        else "full_retained_domain"
                    )
                    if accepted
                    else "none"
                ),
                "raw_gate_status": gate_status,
                "full_domain_gate_status": full_domain_status,
                "selected_domain_status": selected_domain_status,
                "complete_raw_proposal_causal_through_direct_endpoint": (
                    complete_raw_causal
                ),
                "retained_epsilon_max_mev_fm3": (
                    float(retained_epsilon) if accepted else None
                ),
                "retained_pressure_max_mev_fm3": (
                    float(retained_pressure) if accepted else None
                ),
                "retained_endpoint_reason": (
                    str(endpoint_reason) if accepted else None
                ),
                "requested_fixed_masses_status": fixed_mass_status,
                "maximum_mass_availability_status": maximum_mass_status,
                "student_view_eligibility_status": student_view_status,
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
        if hasattr(row, "physical_case_id"):
            record["physical_case_id"] = physical_case_id
            record["is_physical_case_alias"] = bool(
                getattr(row, "is_physical_case_alias", physical_case_id != case_id)
            )
        rows.append(record)
    if rows:
        return pd.DataFrame(rows)
    columns = [
        "case_id",
        "amplitude",
        "epsilon_match_mev_fm3",
        "anchor_mode",
        "epsilon0_mev_fm3",
        "sigma_mev_fm3",
        "delta_mev_fm3",
        "status",
        "acceptance_domain",
        "full_domain_gate_status",
        "selected_domain_status",
        "rejection_reason",
        "pressure_reconstruction",
        "stellar_calculation",
        "clipping_or_repair",
    ]
    if "physical_case_id" in plan.case_table.columns:
        columns.extend(("physical_case_id", "is_physical_case_alias"))
    return pd.DataFrame(columns=columns)


def _final_stage_rows(
    frame: pd.DataFrame | None,
    *,
    case_id: str,
    stage: str,
) -> pd.DataFrame:
    if (
        frame is None
        or frame.empty
        or not {"case_id", "stage"}.issubset(frame.columns)
    ):
        return pd.DataFrame()
    return frame.loc[
        frame["case_id"].astype(str).eq(case_id)
        & frame["stage"].astype(str).eq(stage)
    ]


def _requested_fixed_masses_status(
    config: BSk24TrialConfig,
    case_id: str,
    *,
    accepted: bool,
    fixed_mass_rows: pd.DataFrame | None,
) -> str:
    if not accepted:
        return "not_applicable_raw_gate_not_accepted"
    if not config.fixed_mass_background_requested:
        return "not_requested"
    if not config.tov_stages:
        return "unavailable_no_reporting_stage"
    rows = _final_stage_rows(
        fixed_mass_rows,
        case_id=case_id,
        stage=config.tov_stages[-1].name,
    )
    required = {"target_mass_msun", "status"}
    if rows.empty or not required.issubset(rows.columns):
        return "unavailable_missing_assessment"
    observed_targets = [float(value) for value in rows["target_mass_msun"]]
    expected_targets = [float(value) for value in config.fixed_masses_msun]
    targets_match = bool(
        len(observed_targets) == len(expected_targets)
        and all(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for observed, expected in zip(
                sorted(observed_targets), sorted(expected_targets), strict=True
            )
        )
    )
    succeeded = int(
        rows["status"].astype(str).eq("bracketed_and_solved").sum()
    )
    if targets_match and succeeded == len(expected_targets):
        return "all_requested_fixed_masses_succeeded"
    if succeeded:
        return "partial_requested_fixed_masses_succeeded"
    return "requested_fixed_masses_unavailable"


def _maximum_mass_availability_status(
    config: BSk24TrialConfig,
    case_id: str,
    *,
    accepted: bool,
    maximum_mass_rows: pd.DataFrame | None,
) -> str:
    if not accepted:
        return "not_applicable_raw_gate_not_accepted"
    if not config.background_tov_requested:
        return "not_requested"
    if not config.tov_stages:
        return "unavailable_no_reporting_stage"
    rows = _final_stage_rows(
        maximum_mass_rows,
        case_id=case_id,
        stage=config.tov_stages[-1].name,
    )
    if len(rows) != 1 or "maximum_mass_availability_status" not in rows:
        return "unavailable_missing_assessment"
    value = str(rows.iloc[0]["maximum_mass_availability_status"])
    if value == "resolved_bracketed_and_refined" or value.startswith(
        "unavailable_"
    ):
        return value
    raise ValueError(
        f"case {case_id!r} has an invalid maximum-mass availability status"
    )


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

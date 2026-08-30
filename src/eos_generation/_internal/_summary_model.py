"""Deterministic summary-model assembly from saved packet evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eos_generation._internal._summary_evidence import (
    _COMPLETE_FIXED_MASS_STATUSES,
    _COMPLETE_STELLAR_CONVERGENCE_STATUSES,
    _COMPLETE_TIDAL_STATUSES,
    _TABLE_CANDIDATES,
    _bool_request,
    _case_rows,
    _condition_record,
    _contains_saved_uncertainty,
    _deformation_declaration,
    _mapping_or_empty,
    _maximum_mass_evidence,
    _optional_json,
    _packet_outcome,
    _plot_rows,
    _raw_gate_reports,
    _read_csv_rows,
    _stable_rejection_category,
    _status_counts,
    _text,
    _validation_model,
)


PACKET_SCHEMA_ID = "eos_generation_trial_packet_v1"
CFL_PACKET_SCHEMA_ID = "eos_generation_cfl_trial_packet_v1"
SUPPORTED_PACKET_SCHEMA_IDS = (PACKET_SCHEMA_ID, CFL_PACKET_SCHEMA_ID)
SUMMARY_SCHEMA_ID = "eos_generation_trial_summary_v1"
MAX_SUMMARY_CASE_ROWS = 20


def build_summary_model(
    packet: Path,
    *,
    validation_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared summary model from saved packet evidence only."""

    packet = Path(packet)
    metadata = _mapping_or_empty(_optional_json(packet, "metadata.json"))
    configuration = _mapping_or_empty(
        _optional_json(packet, "complete_configuration.json")
    )
    raw_gate = _mapping_or_empty(_optional_json(packet, "raw_gate_report.json"))
    thermodynamic_convergence = _mapping_or_empty(
        _optional_json(packet, "thermodynamic_convergence.json")
    )
    stellar_convergence = _mapping_or_empty(
        _optional_json(packet, "stellar_convergence.json")
    )
    stellar_summary = _mapping_or_empty(
        _optional_json(packet, "stellar_status_summary.json")
    )
    plot_inventory = _mapping_or_empty(
        _optional_json(packet, "plot_inventory.json")
    )
    reproduction = _mapping_or_empty(
        _optional_json(packet, "reproduction.json")
    )

    ledger = _read_csv_rows(packet, "case_ledger.csv")
    cases = _case_rows(ledger, raw_gate)
    accepted = sum(row["outcome"] == "accepted" for row in cases)
    rejected = sum(row["outcome"] == "rejected" for row in cases)
    if not cases:
        accepted = int(metadata.get("accepted_case_count", 0) or 0)
        rejected = int(metadata.get("rejected_case_count", 0) or 0)
    category_counts = Counter(
        _stable_rejection_category(row["rejection_reason"])
        for row in cases
        if row["outcome"] == "rejected"
    )
    rejected_rows = [row for row in cases if row["outcome"] == "rejected"]
    rejected_no_downstream = bool(rejected_rows) and all(
        row["pressure_reconstruction"]
        == "skipped_due_to_raw_gate_rejection"
        and row["stellar_calculation"]
        == "skipped_due_to_raw_gate_rejection"
        for row in rejected_rows
    )

    raw_reports = _raw_gate_reports(raw_gate)
    assessed = [
        _condition_record(raw_reports, "finite_values", "finite raw proposal"),
        _condition_record(
            raw_reports,
            "positive_energy_density",
            "positive energy density on the declared domain",
        ),
        _condition_record(
            raw_reports,
            "positive_pressure",
            "non-negative pressure on the declared domain",
        ),
        _condition_record(
            raw_reports,
            "strictly_monotone_pressure_implied",
            "mechanical stability (0 < dP/d epsilon)",
        ),
        _condition_record(
            raw_reports,
            "full_retained_domain_passed",
            "causality and raw local-physics gate on the retained domain",
        ),
    ]

    background_requested = _bool_request(configuration, "background_tov_enabled")
    fixed_requested = _bool_request(
        configuration, "fixed_mass_background_enabled"
    )
    tidal_requested = _bool_request(configuration, "tidal_enabled")
    fixed_rows = _read_csv_rows(packet, "fixed_mass_observables.csv")
    maximum_rows = _read_csv_rows(packet, "maximum_mass_screening.csv")
    maximum_mass = _maximum_mass_evidence(
        maximum_rows, requested=background_requested
    )
    saved_uncertainty_sources = [
        name
        for name, payload in (
            ("thermodynamic_convergence.json", thermodynamic_convergence),
            ("stellar_convergence.json", stellar_convergence),
        )
        if _contains_saved_uncertainty(payload)
    ]
    figures = _plot_rows(plot_inventory)
    extended_tables = _mapping_or_empty(metadata.get("extended_tables"))
    table_candidates = list(_TABLE_CANDIDATES)
    for value in extended_tables.values():
        if isinstance(value, str) and value not in table_candidates:
            table_candidates.append(value)
    tables = [name for name in table_candidates if (packet / name).is_file()]

    warnings: list[str] = []
    if not metadata:
        warnings.append("metadata.json is unavailable")
    if not configuration:
        warnings.append("complete_configuration.json is unavailable")
    if not ledger:
        warnings.append("case_ledger.csv is unavailable or empty")
    if rejected:
        warnings.append(f"{rejected} proposal(s) were physically rejected")
    thermodynamic_status = _text(
        thermodynamic_convergence.get("status"), "unavailable"
    )
    if thermodynamic_status not in {
        "pass",
        "pass_monotonically_decreasing_interior_residuals",
        "complete_all_requested_stages",
    }:
        warnings.append(
            f"thermodynamic convergence status is {thermodynamic_status}"
        )
    stellar_status = (
        _text(stellar_convergence.get("status"))
        if background_requested
        else "not_requested"
    )
    if (
        background_requested
        and stellar_status not in _COMPLETE_STELLAR_CONVERGENCE_STATUSES
    ):
        warnings.append(
            "requested stellar convergence is not complete: "
            f"{stellar_status}"
        )
    if (
        background_requested
        and maximum_mass["resolution_status"] != "all_case_stages_resolved"
    ):
        warnings.append(
            "maximum-mass resolution is incomplete or unassessed: "
            f"{maximum_mass['resolution_status']} "
            f"({maximum_mass['resolved_count']}/"
            f"{maximum_mass['case_stage_count']} case-stage rows resolved)"
        )
    if maximum_mass["mass_threshold_fail_count"]:
        warnings.append(
            "maximum-mass threshold failed for "
            f"{maximum_mass['mass_threshold_fail_count']} saved case-stage row(s)"
        )
    fixed_status_counts = _status_counts(fixed_rows, "status")
    incomplete_fixed_statuses = sorted(
        set(fixed_status_counts) - _COMPLETE_FIXED_MASS_STATUSES
    )
    if fixed_requested and (not fixed_rows or incomplete_fixed_statuses):
        warnings.append(
            "requested fixed-mass evidence is incomplete: "
            + (
                ", ".join(incomplete_fixed_statuses)
                if incomplete_fixed_statuses
                else "no saved fixed-mass rows"
            )
        )
    tidal_status = (
        _text(
            stellar_summary.get("publication_interpretation_status"),
            "unavailable",
        )
        if tidal_requested
        else "not_requested"
    )
    if tidal_requested and tidal_status not in _COMPLETE_TIDAL_STATUSES:
        warnings.append(
            f"requested tidal evidence is not complete: {tidal_status}"
        )
    partial_figures = [
        row["figure"] for row in figures if row["status"] == "generated_partial"
    ]
    if partial_figures:
        warnings.append(
            "partial figure evidence: " + ", ".join(partial_figures)
        )
    validation = _validation_model(validation_report)
    if validation is not None and validation["result_status"] == "invalid":
        warnings.append("hard scientific validity failed")
    elif (
        validation is not None
        and validation["scientific_output_availability"] == "partial"
    ):
        warnings.append(
            "scientific results are hard-valid but only partially available"
        )

    packet_schema = _text(metadata.get("schema_id"), PACKET_SCHEMA_ID)
    matter_model = _text(
        metadata.get("matter_model", configuration.get("matter_model")),
        "bsk24",
    )
    model: dict[str, Any] = {
        "schema_id": SUMMARY_SCHEMA_ID,
        "packet_schema": packet_schema,
        "packet_status": _text(metadata.get("packet_status")),
        "outcome": _packet_outcome(accepted, rejected),
        "configuration_hash": _text(metadata.get("configuration_hash")),
        "deformation": _deformation_declaration(configuration),
        "cases": {
            "total": accepted + rejected,
            "accepted": accepted,
            "rejected": rejected,
            "rows_included": len(cases) <= MAX_SUMMARY_CASE_ROWS,
            "rows": cases if len(cases) <= MAX_SUMMARY_CASE_ROWS else [],
            "rejection_reason_totals": [
                {"category": category, "count": category_counts[category]}
                for category in sorted(category_counts)
            ],
            "complete_ledger": "case_ledger.csv",
        },
        "physical_assessment": {
            "assessed_conditions": assessed,
            "unassessed_conditions": [
                {
                    "condition": "microscopic composition",
                    "status": "unavailable",
                },
                {
                    "condition": "species chemical potentials",
                    "status": "unavailable",
                },
                {
                    "condition": "microscopic beta equilibrium",
                    "status": "unassessed",
                },
            ],
            "rejected_proposals_received_no_reconstruction_or_stellar_work": (
                rejected_no_downstream
            ),
        },
        "numerical": {
            "identity_status": _text(metadata.get("identity_status")),
            "thermodynamic_convergence_status": thermodynamic_status,
            "saved_uncertainty_status": (
                "available_in_saved_convergence_evidence"
                if saved_uncertainty_sources
                else "not_available"
            ),
            "saved_uncertainty_sources": saved_uncertainty_sources,
        },
        "stellar_tidal": {
            "background_requested": background_requested,
            "stellar_convergence_status": stellar_status,
            "resolved_maximum_mass_status": maximum_mass[
                "resolution_status"
            ],
            "maximum_mass_resolution_evidence": maximum_mass,
            "fixed_mass_requested": fixed_requested,
            "fixed_mass_status_counts": fixed_status_counts,
            "tidal_requested": tidal_requested,
            "tidal_status": tidal_status,
            "tidal_totals_by_scope": _mapping_or_empty(
                stellar_summary.get("totals_by_scope")
            ),
        },
        "artifacts": {"figures": figures, "tables": tables},
        "warnings": warnings,
        "reproduction": {
            "plan_command": _text(reproduction.get("portable_plan_command")),
            "run_command": _text(reproduction.get("portable_run_command")),
            "plan_hash": _text(reproduction.get("portable_plan_hash")),
            "configuration_file": _text(
                reproduction.get("configuration_file"),
                "unavailable",
            ),
        },
    }
    if matter_model == "cfl":
        model["matter_model"] = "cfl"
    if validation is not None:
        model["validation"] = validation
    return model

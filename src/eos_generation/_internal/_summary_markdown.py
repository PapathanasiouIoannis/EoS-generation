"""Canonical Markdown rendering for deterministic packet summaries."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from eos_generation._internal._summary_evidence import (
    _mapping_or_empty,
    _sequence_or_empty,
    _text,
)


def _markdown_escape(value: Any) -> str:
    return _text(value).replace("\\", "\\\\").replace("|", "\\|").replace(
        "\r", " "
    ).replace("\n", " ")


def _format_values(values: Sequence[Any], unit: str = "") -> str:
    if not values:
        return "not declared"
    suffix = f" {unit}" if unit else ""
    return ", ".join(f"`{value}`{suffix}" for value in values)


def render_summary_markdown(model: Mapping[str, Any]) -> str:
    """Render canonical Markdown for a summary model."""

    cases = _mapping_or_empty(model.get("cases"))
    deformation = _mapping_or_empty(model.get("deformation"))
    physical = _mapping_or_empty(model.get("physical_assessment"))
    numerical = _mapping_or_empty(model.get("numerical"))
    stellar = _mapping_or_empty(model.get("stellar_tidal"))
    artifacts = _mapping_or_empty(model.get("artifacts"))
    reproduction = _mapping_or_empty(model.get("reproduction"))
    lines = [
        "# BSk24 trial result summary",
        "",
        f"<!-- schema_id: {_text(model.get('schema_id'))} -->",
        "",
        f"**Outcome: {_text(model.get('outcome')).upper()}.**",
        "",
        f"- Packet schema: `{_text(model.get('packet_schema'))}`",
        f"- Packet status: `{_text(model.get('packet_status'))}`",
        f"- Configuration SHA-256: `{_text(model.get('configuration_hash'))}`",
        (
            "- Cases: "
            f"{cases.get('total', 0)} total; {cases.get('accepted', 0)} accepted; "
            f"{cases.get('rejected', 0)} rejected"
        ),
        "",
        "## Deformation declaration",
        "",
        "- Amplitude: " + _format_values(
            _sequence_or_empty(deformation.get("amplitudes"))
        ),
        "- Center `epsilon0`: " + _format_values(
            _sequence_or_empty(deformation.get("epsilon0_values_mev_fm3")),
            "MeV fm^-3",
        ),
        "- Width `sigma`: " + _format_values(
            _sequence_or_empty(deformation.get("sigma_values_mev_fm3")),
            "MeV fm^-3",
        ),
        "- Compensation geometry `delta`: " + _format_values(
            _sequence_or_empty(deformation.get("deltas_mev_fm3")),
            "MeV fm^-3",
        ),
        f"- Onset: {_text(deformation.get('onset'))}.",
        "",
        "## Case outcomes",
        "",
    ]
    if bool(cases.get("rows_included")):
        lines.extend(
            [
                "| Case | Outcome | Deformation (A; epsilon0; sigma; delta) | "
                "Exact rejection reason | Reconstruction | Stellar/tidal |",
                "|---|---|---|---|---|---|",
            ]
        )
        for row_value in _sequence_or_empty(cases.get("rows")):
            row = _mapping_or_empty(row_value)
            parameters = _mapping_or_empty(row.get("deformation"))
            amplitude = parameters.get("amplitude")
            leading = f"A={_text(amplitude)}"
            parameter_text = (
                f"{leading}; epsilon0={_text(parameters.get('epsilon0_mev_fm3'))}; "
                f"sigma={_text(parameters.get('sigma_mev_fm3'))}; "
                f"delta={_text(parameters.get('delta_mev_fm3'))} MeV fm^-3"
            )
            reason = row.get("rejection_reason") or "not applicable"
            lines.append(
                "| "
                + " | ".join(
                    _markdown_escape(value)
                    for value in (
                        row.get("case_id"),
                        row.get("outcome"),
                        parameter_text,
                        reason,
                        row.get("pressure_reconstruction"),
                        row.get("stellar_calculation"),
                    )
                )
                + " |"
            )
    else:
        lines.append(
            f"This packet has {cases.get('total', 0)} cases. The complete case "
            "table is in [case_ledger.csv](case_ledger.csv)."
        )
        reasons = _sequence_or_empty(cases.get("rejection_reason_totals"))
        if reasons:
            lines.extend(["", "Rejection-reason totals:", ""])
            for raw in reasons:
                item = _mapping_or_empty(raw)
                lines.append(
                    f"- {item.get('count', 0)} x "
                    f"`{_markdown_escape(item.get('category'))}`"
                )

    lines.extend(["", "## Physical assessment", "", "Assessed conditions:", ""])
    for raw in _sequence_or_empty(physical.get("assessed_conditions")):
        condition = _mapping_or_empty(raw)
        lines.append(
            f"- {_text(condition.get('condition'))}: "
            f"`{_text(condition.get('status'))}` "
            f"({condition.get('passed_case_count', 0)} passed, "
            f"{condition.get('failed_case_count', 0)} failed)."
        )
    lines.extend(["", "Explicitly unassessed or unavailable conditions:", ""])
    for raw in _sequence_or_empty(physical.get("unassessed_conditions")):
        condition = _mapping_or_empty(raw)
        lines.append(
            f"- {_text(condition.get('condition'))}: "
            f"`{_text(condition.get('status'))}`."
        )
    if cases.get("rejected", 0):
        confirmation = bool(
            physical.get(
                "rejected_proposals_received_no_reconstruction_or_stellar_work"
            )
        )
        lines.extend(
            [
                "",
                (
                    "Rejected proposals received no reconstruction or stellar/tidal "
                    f"work: **{'confirmed' if confirmation else 'not confirmed'}**."
                ),
            ]
        )

    maximum_evidence = _mapping_or_empty(
        stellar.get("maximum_mass_resolution_evidence")
    )
    lines.extend(
        [
            "",
            "## Numerical, stellar, and tidal evidence",
            "",
            f"- A=0 identity: `{_text(numerical.get('identity_status'))}`",
            (
                "- Thermodynamic convergence: "
                f"`{_text(numerical.get('thermodynamic_convergence_status'))}`"
            ),
            (
                "- Saved numerical uncertainty: "
                f"`{_text(numerical.get('saved_uncertainty_status'))}`"
            ),
            (
                "- Stellar convergence: "
                f"`{_text(stellar.get('stellar_convergence_status'))}`"
            ),
            (
                "- Resolved maximum mass: "
                f"`{_text(stellar.get('resolved_maximum_mass_status'))}`"
            ),
            (
                "- Maximum-mass resolution evidence: "
                f"{maximum_evidence.get('resolved_count', 0)}/"
                f"{maximum_evidence.get('case_stage_count', 0)} case-stage rows "
                "resolved; "
                f"{maximum_evidence.get('unresolved_count', 0)} unresolved; "
                f"{maximum_evidence.get('resolution_unrecorded_count', 0)} "
                "unrecorded"
            ),
            (
                "- Maximum-mass row status counts: `"
                + json.dumps(
                    _mapping_or_empty(
                        maximum_evidence.get("status_counts")
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "`"
            ),
            (
                "- Maximum-mass availability status counts: `"
                + json.dumps(
                    _mapping_or_empty(
                        maximum_evidence.get("availability_status_counts")
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "`"
            ),
            (
                "- Maximum-mass threshold evidence: "
                f"{maximum_evidence.get('mass_threshold_pass_count', 0)} passed; "
                f"{maximum_evidence.get('mass_threshold_fail_count', 0)} failed; "
                f"{maximum_evidence.get('mass_threshold_unavailable_count', 0)} "
                "unavailable because M_max was unresolved; "
                f"{maximum_evidence.get('mass_threshold_unrecorded_count', 0)} "
                "unrecorded; "
                f"{maximum_evidence.get('mass_threshold_inconsistent_count', 0)} "
                "inconsistent"
            ),
            (
                "- Maximum-mass resolution by stage: `"
                + json.dumps(
                    _mapping_or_empty(
                        maximum_evidence.get("by_stage")
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "`"
            ),
            (
                "- Fixed-mass status counts: `"
                + json.dumps(
                    _mapping_or_empty(stellar.get("fixed_mass_status_counts")),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "`"
            ),
            f"- Tidal evidence: `{_text(stellar.get('tidal_status'))}`",
        ]
    )
    validation = model.get("validation")
    if isinstance(validation, Mapping):
        lines.extend(
            [
                "",
                "## Validation status",
                "",
                f"**Result is {_text(validation.get('result_status')).upper()}.**",
                "",
                (
                    "Hard scientific validity: "
                    f"`{_text(validation.get('scientific_output_validity'))}`."
                ),
                (
                    "Scientific availability: "
                    f"`{_text(validation.get('scientific_output_availability'))}`."
                ),
            ]
        )
        for failure in _sequence_or_empty(validation.get("failures")):
            lines.append(f"- Failure: `{_markdown_escape(failure)}`")
        for limitation in _sequence_or_empty(validation.get("limitations")):
            lines.append(
                f"- Availability limitation: `{_markdown_escape(limitation)}`"
            )

    lines.extend(["", "## Warnings", ""])
    warnings = _sequence_or_empty(model.get("warnings"))
    if warnings:
        lines.extend(f"- {_text(item)}" for item in warnings)
    else:
        lines.append("- None recorded in the saved summary evidence.")

    lines.extend(["", "## Available figures and tables", ""])
    figures = _sequence_or_empty(artifacts.get("figures"))
    if figures:
        lines.extend(
            [
                "| Figure | Status | Notes |",
                "|---|---|---|",
            ]
        )
        for raw in figures:
            figure = _mapping_or_empty(raw)
            relative = _text(figure.get("relative_path"))
            figure_name = _markdown_escape(figure.get("figure"))
            status = _text(figure.get("status"))
            display = (
                f"[{figure_name}]({relative})"
                if status in {"generated", "generated_partial"}
                else f"`{figure_name}`"
            )
            lines.append(
                f"| {display} | "
                f"{_markdown_escape(status)} | "
                f"{_markdown_escape(figure.get('reason'))} |"
            )
    else:
        lines.append("No figure inventory is available.")
    tables = _sequence_or_empty(artifacts.get("tables"))
    lines.extend(["", "Tables:", ""])
    if tables:
        lines.extend(f"- [{name}]({name})" for name in tables)
    else:
        lines.append("- No saved result tables are available.")

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "From the repository root, first reproduce and review the fresh plan:",
            "",
            "```text",
            _text(reproduction.get("plan_command")),
            "```",
            "",
            "Then execute that exact destination-bound plan hash:",
            "",
            "```text",
            _text(reproduction.get("run_command")),
            "```",
            "",
        ]
    )
    return "\n".join(lines)

"""Deterministic human summaries for saved BSk24 trial packets.

The summary layer is deliberately passive.  It reads only saved result
evidence, never imports a scientific solver, and does not consult the current
clock, source tree, packet manifest, or an existing ``summary.md``.  This
makes the rendered bytes suitable for packet-integrity validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eos_generation._internal._summary_evidence import (
    _COMPLETE_FIXED_MASS_STATUSES,
    _COMPLETE_STELLAR_CONVERGENCE_STATUSES,
    _COMPLETE_TIDAL_STATUSES,
    _NUMERIC_TOKEN,
    _TABLE_CANDIDATES,
    _bool_request,
    _canonical_reason,
    _case_report,
    _case_rows,
    _condition_record,
    _contains_saved_uncertainty,
    _deformation_declaration,
    _mapping_or_empty,
    _maximum_mass_evidence,
    _maximum_mass_group_evidence,
    _optional_json,
    _packet_outcome,
    _parameter,
    _plot_rows,
    _raw_gate_reports,
    _read_csv_rows,
    _rejection_reason,
    _saved_boolean,
    _sequence_or_empty,
    _stable_rejection_category,
    _status_counts,
    _text,
    _validation_model,
)
from eos_generation._internal._summary_markdown import (
    _format_values,
    _markdown_escape,
    render_summary_markdown as _render_summary_markdown_impl,
)
from eos_generation._internal._summary_model import (
    MAX_SUMMARY_CASE_ROWS,
    PACKET_SCHEMA_ID,
    SUMMARY_SCHEMA_ID,
    build_summary_model as _build_summary_model_impl,
)
from eos_generation._internal.packet_integrity import _write_text_atomic


def build_summary_model(
    packet: Path,
    *,
    validation_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared summary model from saved packet evidence only."""

    return _build_summary_model_impl(
        packet,
        validation_report=validation_report,
    )


def render_summary_markdown(model: Mapping[str, Any]) -> str:
    """Render canonical Markdown for a summary model."""

    return _render_summary_markdown_impl(model)


def write_packet_summary(
    packet: Path,
    *,
    validation_report: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write the canonical ``summary.md`` for a saved packet."""

    packet = Path(packet)
    summary_path = packet / "summary.md"
    model = build_summary_model(packet, validation_report=validation_report)
    _write_text_atomic(render_summary_markdown(model), summary_path)
    return summary_path


# Private aliases keep call sites concise.
_build_summary_model = build_summary_model
_render_summary_markdown = render_summary_markdown
_write_packet_summary = write_packet_summary


__all__ = [
    "MAX_SUMMARY_CASE_ROWS",
    "PACKET_SCHEMA_ID",
    "SUMMARY_SCHEMA_ID",
    "_build_summary_model",
    "_render_summary_markdown",
    "_write_packet_summary",
    "build_summary_model",
    "render_summary_markdown",
    "write_packet_summary",
]

"""Case lifecycle and plot-inventory consistency checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eos_generation.reporting._validation_io import (
    _Layer,
    _read_csv,
    _safe_packet_relative,
)

RAW_GATE_SCHEMA = "eos_generation_raw_gate_v1"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

EXTENDED_REQUIRED_FILES = (
    "radial_profiles.csv",
    "deformation_support_fractions.csv",
    "outside_support_control.csv",
    "turning_point_sequences.csv",
    "baryonic_observables.csv",
    "stellar_response_across_mass.csv",
    "baryonic_response_across_mass.csv",
    "odd_even_response.csv",
    "matched_area_comparison.csv",
    "numerical_error_summary.csv",
)


def _validate_ledger(
    packet: Path,
    actual: set[str],
    ledger: Any,
    layer: _Layer,
) -> None:
    if not isinstance(ledger, dict):
        if ledger is not None:
            layer.fail("manual_ledger:not_an_object")
        return
    normalized: dict[str, list[str]] = {}
    for key in ("files_created", "files_modified"):
        values = ledger.get(key)
        if not isinstance(values, list):
            layer.fail(f"manual_ledger:{key}:not_a_list")
            continue
        paths: list[str] = []
        for index, value in enumerate(values):
            safe = _safe_packet_relative(
                value,
                packet,
                layer,
                context=f"manual_ledger:{key}:{index}",
            )
            if safe is not None:
                paths.append(safe[0])
        if len(paths) != len(set(paths)):
            layer.fail(f"manual_ledger:{key}:duplicates")
        normalized[key] = paths
    created = set(normalized.get("files_created", []))
    modified = set(normalized.get("files_modified", []))
    overlap = created & modified
    for relative in sorted(overlap):
        layer.fail(f"manual_ledger:created_modified_overlap:{relative}")
    covered = created | modified
    for relative in sorted(actual - covered):
        layer.fail(f"manual_ledger:coverage_missing:{relative}")
    for relative in sorted(covered - actual):
        layer.fail(f"manual_ledger:lists_missing_file:{relative}")
    layer.checks["manual_ledger_entries"] = len(covered)
    layer.checks["actual_packet_files"] = len(actual)


def _case_ids(rows: list[dict[str, str]] | None, status: str | None = None) -> set[str]:
    if rows is None:
        return set()
    return {
        row.get("case_id", "")
        for row in rows
        if row.get("case_id") and (status is None or row.get("status") == status)
    }


def _ids_from_records(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    result: set[str] = set()
    for item in value:
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, dict) and isinstance(item.get("case_id"), str):
            result.add(item["case_id"])
    return result


def _validate_case_consistency(
    packet: Path,
    *,
    case_plan: list[dict[str, str]] | None,
    case_ledger: list[dict[str, str]] | None,
    raw_gate: Any,
    accepted_rejected: Any,
    metadata: Any,
    layer: _Layer,
) -> tuple[set[str], set[str]]:
    if case_ledger is None:
        return set(), set()
    ledger_ids = [row.get("case_id", "") for row in case_ledger]
    if not all(ledger_ids) or len(ledger_ids) != len(set(ledger_ids)):
        layer.fail("case_ledger:missing_or_duplicate_case_id")
    invalid_status = sorted(
        {
            row.get("status", "")
            for row in case_ledger
            if row.get("status") not in {"accepted", "rejected"}
        }
    )
    if invalid_status:
        layer.fail(f"case_ledger:invalid_statuses:{invalid_status}")
    accepted = _case_ids(case_ledger, "accepted")
    rejected = _case_ids(case_ledger, "rejected")
    if accepted & rejected:
        layer.fail(f"case_ledger:accepted_rejected_overlap:{sorted(accepted & rejected)}")
    planned = _case_ids(case_plan)
    if case_plan is not None and planned != accepted | rejected:
        layer.fail("case_plan_case_ledger_mismatch")

    if isinstance(raw_gate, dict):
        if raw_gate.get("schema_id") != RAW_GATE_SCHEMA:
            layer.fail("raw_gate:unsupported_schema")
        raw_accepted = set(raw_gate.get("accepted_case_ids", []))
        raw_rejected = set(raw_gate.get("rejected_case_ids", []))
        raw_cases = set(raw_gate.get("cases", {}))
        if raw_accepted & raw_rejected:
            layer.fail("raw_gate:accepted_rejected_overlap")
        if raw_accepted != accepted:
            layer.fail("raw_gate:accepted_case_ids_mismatch")
        if raw_rejected != rejected:
            layer.fail("raw_gate:rejected_case_ids_mismatch")
        if raw_cases != accepted | rejected:
            layer.fail("raw_gate:cases_mismatch")
        if raw_gate.get("executed_before_reconstruction_and_TOV") is not True:
            layer.fail("raw_gate:ordering_not_confirmed")
        if raw_gate.get("full_domain_gate_authoritative") is not True:
            layer.fail("raw_gate:full_domain_authority_not_confirmed")
        if raw_gate.get("selected_domain_policy") != "full_retained_domain_only":
            layer.fail("raw_gate:invalid_selected_domain_policy")
        case_reports = raw_gate.get("cases", {})
        ledger_by_id = {
            str(row.get("case_id", "")): row for row in case_ledger
        }
        for case_id in accepted:
            report = (
                case_reports.get(case_id)
                if isinstance(case_reports, dict)
                else None
            )
            if not isinstance(report, dict):
                layer.fail(f"raw_gate:missing_case_report:{case_id}")
                continue
            if report.get("status") == "accepted_raw_local_physics_gate":
                if ledger_by_id[case_id].get("acceptance_domain") not in (
                    None,
                    "",
                    "full_retained_domain",
                ):
                    layer.fail(
                        f"raw_gate:invalid_full_domain_selection:{case_id}"
                    )
                continue
            layer.fail(
                f"raw_gate:accepted_case_failed_full_domain_gate:{case_id}"
            )
    elif raw_gate is not None:
        layer.fail("raw_gate:not_an_object")

    if isinstance(accepted_rejected, dict):
        if _ids_from_records(accepted_rejected.get("accepted")) != accepted:
            layer.fail("accepted_rejected_file:accepted_mismatch")
        if _ids_from_records(accepted_rejected.get("rejected")) != rejected:
            layer.fail("accepted_rejected_file:rejected_mismatch")
        if (
            accepted_rejected.get(
                "rejected_cases_received_no_reconstruction_or_stellar_work"
            )
            is not True
        ):
            layer.fail("accepted_rejected_file:rejected_work_exclusion_not_confirmed")
        json_lifecycle = {
            str(item["case_id"]): item
            for key in ("accepted", "rejected")
            for item in accepted_rejected.get(key, [])
            if isinstance(item, dict) and isinstance(item.get("case_id"), str)
        }
        for row in case_ledger:
            case_id = row.get("case_id", "")
            counterpart = json_lifecycle.get(case_id)
            if counterpart is None:
                continue
            for field in (
                "status",
                "pressure_reconstruction",
                "stellar_calculation",
                "clipping_or_repair",
            ):
                if _inventory_scalar(row.get(field)) != _inventory_scalar(
                    counterpart.get(field)
                ):
                    layer.fail(
                        "accepted_rejected_file:lifecycle_mismatch:"
                        f"{case_id}:{field}"
                    )
    elif accepted_rejected is not None:
        layer.fail("accepted_rejected_file:not_an_object")

    if isinstance(metadata, dict):
        if set(metadata.get("accepted_case_ids", [])) != accepted:
            layer.fail("metadata:accepted_case_ids_mismatch")
        if set(metadata.get("rejected_case_ids", [])) != rejected:
            layer.fail("metadata:rejected_case_ids_mismatch")
        if metadata.get("accepted_case_count") != len(accepted):
            layer.fail("metadata:accepted_case_count_mismatch")
        if metadata.get("rejected_case_count") != len(rejected):
            layer.fail("metadata:rejected_case_count_mismatch")

    reconstructed_tables = (
        "thermodynamic_profiles.csv",
        "thermodynamic_residuals.csv",
        "window_characterization.csv",
        "stellar_sequences.csv",
        "fixed_mass_observables.csv",
        *EXTENDED_REQUIRED_FILES,
    )
    for relative in reconstructed_tables:
        if not (packet / relative).is_file():
            continue
        rows = _read_csv(packet, relative, layer)
        present = _case_ids(rows)
        leaked = present & rejected
        if leaked:
            layer.fail(f"rejected_case_in_output:{relative}:{sorted(leaked)}")
    layer.checks["accepted_case_count"] = len(accepted)
    layer.checks["rejected_case_count"] = len(rejected)
    return accepted, rejected


def _inventory_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _csv_json_records_equal(
    csv_rows: list[dict[str, str]],
    json_rows: list[dict[str, Any]],
) -> bool:
    if len(csv_rows) != len(json_rows):
        return False
    for csv_row, json_row in zip(csv_rows, json_rows, strict=True):
        if set(csv_row) != set(json_row):
            return False
        for key in csv_row:
            if _inventory_scalar(csv_row.get(key)) != _inventory_scalar(
                json_row.get(key)
            ):
                return False
    return True


def _inventory_records(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return None
    return list(value)


def _validate_plot_inventory(
    packet: Path,
    csv_rows: list[dict[str, str]] | None,
    payload: Any,
    layer: _Layer,
) -> None:
    if csv_rows is None:
        return
    if not isinstance(payload, dict):
        if payload is not None:
            layer.fail("plot_inventory_json:not_an_object")
        return
    json_rows = _inventory_records(payload.get("figures"))
    if json_rows is None:
        layer.fail("plot_inventory_json:figures_not_a_record_list")
        return
    csv_figures = [row.get("figure", "") for row in csv_rows]
    json_figures = [_inventory_scalar(row.get("figure")) for row in json_rows]
    if (
        not all(csv_figures)
        or len(csv_figures) != len(set(csv_figures))
        or len(json_figures) != len(set(json_figures))
    ):
        layer.fail("plot_inventory:missing_or_duplicate_figure")
    csv_by_figure = {row.get("figure", ""): row for row in csv_rows}
    json_by_figure = {
        _inventory_scalar(row.get("figure")): row for row in json_rows
    }
    if set(csv_by_figure) != set(json_by_figure):
        layer.fail("plot_inventory:csv_json_figure_set_mismatch")
    comparison_fields = {
        "figure",
        "group",
        "status",
        "reason",
        "prerequisite",
        "relative_path",
        "tidal_completeness_status",
        "tidal_validated_count",
        "tidal_omitted_count",
        "population_stage",
        "population_target_mass_msun",
        "eligible_response_row_count",
    }
    for figure in sorted(set(csv_by_figure) & set(json_by_figure)):
        csv_row = csv_by_figure[figure]
        json_row = json_by_figure[figure]
        for field in comparison_fields:
            if field not in csv_row and field not in json_row:
                continue
            if _inventory_scalar(csv_row.get(field)) != _inventory_scalar(
                json_row.get(field)
            ):
                layer.fail(f"plot_inventory:csv_json_mismatch:{figure}:{field}")

    generated_statuses = {"generated", "generated_partial", "partial"}
    allowed_statuses = generated_statuses | {"skipped"}
    for row in csv_rows:
        figure = row.get("figure", "")
        status = row.get("status", "")
        if status not in allowed_statuses:
            layer.fail(f"plot_inventory:invalid_status:{figure}:{status}")
            continue
        safe = _safe_packet_relative(
            row.get("relative_path", ""),
            packet,
            layer,
            context=f"plot_inventory:{figure}",
        )
        if safe is None:
            continue
        relative, path = safe
        if not relative.lower().endswith(".png"):
            layer.fail(f"plot_inventory:non_png_path:{figure}:{relative}")
        if status in generated_statuses:
            if not path.is_file():
                layer.fail(f"plot_inventory:generated_file_missing:{figure}")
            else:
                try:
                    with path.open("rb") as handle:
                        signature = handle.read(len(PNG_SIGNATURE))
                    if signature != PNG_SIGNATURE:
                        layer.fail(f"plot_inventory:invalid_png_signature:{figure}")
                except Exception as exc:
                    layer.fail(
                        f"plot_inventory:png_read:{figure}:{type(exc).__name__}:{exc}"
                    )

    for key, statuses in (
        ("generated", generated_statuses),
        ("skipped", {"skipped"}),
        ("partial", {"generated_partial", "partial"}),
    ):
        if key not in payload:
            continue
        subset = _inventory_records(payload.get(key))
        if subset is None:
            layer.fail(f"plot_inventory_json:{key}_not_a_record_list")
            continue
        expected = {
            row.get("figure", "")
            for row in csv_rows
            if row.get("status", "") in statuses
        }
        actual = {_inventory_scalar(row.get("figure")) for row in subset}
        if actual != expected:
            layer.fail(f"plot_inventory_json:{key}_subset_mismatch")
    layer.checks["plot_inventory_entries"] = len(csv_rows)

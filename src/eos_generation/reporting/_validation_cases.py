"""Case lifecycle and plot-inventory consistency checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from eos_generation.reporting._validation_io import (
    _Layer,
    _read_csv,
    _safe_packet_relative,
)

RAW_GATE_SCHEMA = "eos_generation_raw_gate_v2"
LEGACY_RAW_GATE_SCHEMA = "eos_generation_raw_gate_v1"
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

_RAW_GATE_PROFILE_V2_COLUMNS = frozenset(
    {
        "case_id",
        "amplitude",
        "epsilon0_mev_fm3",
        "sigma_mev_fm3",
        "delta_mev_fm3",
        "epsilon_mev_fm3",
        "window",
        "gaussian",
        "delta_cs2",
        "direct_pressure_mev_fm3",
        "delta_pressure_mev_fm3",
        "raw_pressure_mev_fm3",
        "raw_cs2",
        "gate_status",
    }
)

_CFL_RAW_GATE_PROFILE_COLUMNS = frozenset(
    {
        "case_id",
        "physical_case_id",
        "matter_model",
        "baseline_parameter_set_sha256",
        "amplitude",
        "epsilon0_mev_fm3",
        "sigma_mev_fm3",
        "delta_mev_fm3",
        "epsilon_mev_fm3",
        "window",
        "gaussian",
        "delta_cs2",
        "raw_cs2",
        "gate_status",
    }
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


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _validate_raw_gate_profiles(
    rows: list[dict[str, str]] | None,
    *,
    raw_gate: Any,
    raw_schema: Any,
    accepted: set[str],
    rejected: set[str],
    layer: _Layer,
) -> None:
    if rows is None:
        return
    expected_ids = accepted | rejected
    present_ids = _case_ids(rows)
    if present_ids != expected_ids:
        layer.fail("raw_gate_profiles:case_ids_mismatch")
    if raw_schema != RAW_GATE_SCHEMA:
        return
    if not rows:
        if expected_ids:
            layer.fail("raw_gate_profiles:missing_v2_rows")
        return
    missing_columns = sorted(_RAW_GATE_PROFILE_V2_COLUMNS - set(rows[0]))
    if missing_columns:
        layer.fail(f"raw_gate_profiles:missing_v2_columns:{missing_columns}")
        return
    reports = raw_gate.get("cases") if isinstance(raw_gate, dict) else None
    if not isinstance(reports, dict):
        layer.fail("raw_gate_profiles:case_reports_unavailable")
        return

    blocks: dict[str, list[dict[str, str]]] = {}
    closed: set[str] = set()
    current: str | None = None
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if case_id != current:
            if current is not None:
                closed.add(current)
            if case_id in closed:
                layer.fail(f"raw_gate_profiles:noncontiguous_case:{case_id}")
            current = case_id
        blocks.setdefault(case_id, []).append(row)

    geometry_fields = (
        "amplitude",
        "epsilon0_mev_fm3",
        "sigma_mev_fm3",
        "delta_mev_fm3",
    )
    for case_id in sorted(expected_ids):
        case_rows = blocks.get(case_id, [])
        report = reports.get(case_id)
        if not case_rows or not isinstance(report, dict):
            layer.fail(f"raw_gate_profiles:missing_case_evidence:{case_id}")
            continue
        parameters = report.get("parameters")
        domain = report.get("complete_proposed_retained_domain_mev_fm3")
        if not isinstance(parameters, dict):
            layer.fail(f"raw_gate_profiles:parameters_unavailable:{case_id}")
            continue
        if not isinstance(domain, list) or len(domain) != 2:
            layer.fail(f"raw_gate_profiles:domain_unavailable:{case_id}")
            continue
        lower = _finite_float(domain[0])
        upper = _finite_float(domain[1])
        epsilon = [
            _finite_float(row.get("epsilon_mev_fm3")) for row in case_rows
        ]
        if (
            lower is None
            or upper is None
            or lower >= upper
            or any(value is None for value in epsilon)
            or any(
                right <= left
                for left, right in zip(epsilon[:-1], epsilon[1:])
            )
            or epsilon[0] != lower
            or epsilon[-1] != upper
        ):
            layer.fail(f"raw_gate_profiles:incomplete_raw_domain:{case_id}")
        expected_count = report.get("dense_grid_points")
        if (
            not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or len(case_rows) != expected_count
        ):
            layer.fail(f"raw_gate_profiles:point_count_mismatch:{case_id}")

        expected_status = report.get("status")
        accepted_status = expected_status == "accepted_raw_local_physics_gate"
        report_finite = report.get("finite_values") is True
        accepted_pressures: list[float] = []
        accepted_cs2: list[float] = []
        for position, row in enumerate(case_rows):
            if row.get("gate_status") != expected_status:
                layer.fail(
                    f"raw_gate_profiles:gate_status_mismatch:"
                    f"{case_id}:{position}"
                )
            for field in geometry_fields:
                saved = _finite_float(row.get(field))
                expected = _finite_float(parameters.get(field))
                if saved is None or expected is None or saved != expected:
                    layer.fail(
                        f"raw_gate_profiles:geometry_mismatch:"
                        f"{case_id}:{position}:{field}"
                    )
            direct_pressure = _finite_float(
                row.get("direct_pressure_mev_fm3")
            )
            delta_pressure = _finite_float(row.get("delta_pressure_mev_fm3"))
            raw_pressure = _finite_float(row.get("raw_pressure_mev_fm3"))
            raw_cs2 = _finite_float(row.get("raw_cs2"))
            auxiliary = tuple(
                _finite_float(row.get(column))
                for column in ("window", "gaussian", "delta_cs2")
            )
            if (
                direct_pressure is not None
                and delta_pressure is not None
                and raw_pressure is not None
            ):
                if raw_pressure != direct_pressure + delta_pressure:
                    layer.fail(
                        f"raw_gate_profiles:pressure_identity_mismatch:"
                        f"{case_id}:{position}"
                    )
            elif report_finite or accepted_status:
                layer.fail(
                    f"raw_gate_profiles:nonfinite_pressure_evidence:"
                    f"{case_id}:{position}"
                )
            if (report_finite or accepted_status) and raw_cs2 is None:
                layer.fail(
                    f"raw_gate_profiles:nonfinite_cs2_evidence:"
                    f"{case_id}:{position}"
                )
            if accepted_status:
                if any(value is None for value in auxiliary):
                    layer.fail(
                        f"raw_gate_profiles:nonfinite_analytical_evidence:"
                        f"{case_id}:{position}"
                    )
                if raw_pressure is not None:
                    accepted_pressures.append(raw_pressure)
                if raw_cs2 is not None:
                    accepted_cs2.append(raw_cs2)
        if accepted_status and (
            len(accepted_pressures) != len(case_rows)
            or len(accepted_cs2) != len(case_rows)
            or any(value <= 0.0 for value in accepted_pressures)
            or any(value <= 0.0 for value in accepted_cs2)
            or any(
                right <= left
                for left, right in zip(
                    accepted_pressures[:-1], accepted_pressures[1:]
                )
            )
        ):
            layer.fail(f"raw_gate_profiles:accepted_core_state_invalid:{case_id}")


def _validate_cfl_raw_gate_profiles(
    rows: list[dict[str, str]] | None,
    *,
    raw_gate: Any,
    accepted: set[str],
    rejected: set[str],
    layer: _Layer,
) -> None:
    """Validate complete-domain CFL raw evidence in its physical identity space."""

    if rows is None:
        return
    expected_ids = accepted | rejected
    present_ids = _case_ids(rows)
    if present_ids != expected_ids:
        layer.fail("cfl_raw_gate_profiles:physical_case_ids_mismatch")
    if not rows:
        if expected_ids:
            layer.fail("cfl_raw_gate_profiles:missing_rows")
        return
    missing_columns = sorted(_CFL_RAW_GATE_PROFILE_COLUMNS - set(rows[0]))
    if missing_columns:
        layer.fail(
            f"cfl_raw_gate_profiles:missing_columns:{missing_columns}"
        )
        return
    reports = raw_gate.get("cases") if isinstance(raw_gate, dict) else None
    if not isinstance(reports, dict):
        layer.fail("cfl_raw_gate_profiles:case_reports_unavailable")
        return

    blocks: dict[str, list[dict[str, str]]] = {}
    closed: set[str] = set()
    current: str | None = None
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if case_id != current:
            if current is not None:
                closed.add(current)
            if case_id in closed:
                layer.fail(
                    f"cfl_raw_gate_profiles:noncontiguous_case:{case_id}"
                )
            current = case_id
        blocks.setdefault(case_id, []).append(row)

    geometry_fields = (
        ("amplitude", "amplitude"),
        ("epsilon0_mev_fm3", "center_mev_fm3"),
        ("sigma_mev_fm3", "width_mev_fm3"),
        ("delta_mev_fm3", "ramp_width_mev_fm3"),
    )
    for case_id in sorted(expected_ids):
        case_rows = blocks.get(case_id, [])
        report = reports.get(case_id)
        if not case_rows or not isinstance(report, dict):
            layer.fail(
                f"cfl_raw_gate_profiles:missing_case_evidence:{case_id}"
            )
            continue
        parameters = report.get("parameters")
        domain = report.get("complete_declared_domain_mev_fm3")
        if not isinstance(parameters, dict):
            layer.fail(f"cfl_raw_gate_profiles:parameters_unavailable:{case_id}")
            continue
        if not isinstance(domain, list) or len(domain) != 2:
            layer.fail(f"cfl_raw_gate_profiles:domain_unavailable:{case_id}")
            continue
        lower = _finite_float(domain[0])
        upper = _finite_float(domain[1])
        epsilon = [
            _finite_float(row.get("epsilon_mev_fm3")) for row in case_rows
        ]
        if (
            lower is None
            or upper is None
            or not (0.0 < lower < upper)
            or any(value is None for value in epsilon)
            or any(
                right <= left
                for left, right in zip(epsilon[:-1], epsilon[1:])
            )
            or epsilon[0] != lower
            or epsilon[-1] != upper
        ):
            layer.fail(f"cfl_raw_gate_profiles:incomplete_domain:{case_id}")
        expected_count = report.get("dense_grid_points")
        if (
            not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or len(case_rows) != expected_count
        ):
            layer.fail(f"cfl_raw_gate_profiles:point_count_mismatch:{case_id}")

        expected_status = report.get("status")
        accepted_status = case_id in accepted
        expected_accepted_status = (
            expected_status == "accepted_raw_local_physics_gate"
        )
        if accepted_status is not expected_accepted_status:
            layer.fail(f"cfl_raw_gate_profiles:outcome_mismatch:{case_id}")
        report_finite = report.get("finite_values") is True
        baseline_hash = report.get("baseline_parameter_set_sha256")
        for position, row in enumerate(case_rows):
            if row.get("case_id") != case_id or row.get("physical_case_id") != case_id:
                layer.fail(
                    f"cfl_raw_gate_profiles:identity_mismatch:{case_id}:{position}"
                )
            if row.get("matter_model") != "cfl":
                layer.fail(
                    f"cfl_raw_gate_profiles:matter_model_mismatch:{case_id}:{position}"
                )
            if row.get("baseline_parameter_set_sha256") != baseline_hash:
                layer.fail(
                    f"cfl_raw_gate_profiles:baseline_hash_mismatch:{case_id}:{position}"
                )
            if row.get("gate_status") != expected_status:
                layer.fail(
                    f"cfl_raw_gate_profiles:gate_status_mismatch:{case_id}:{position}"
                )
            for saved_field, parameter_field in geometry_fields:
                saved = _finite_float(row.get(saved_field))
                expected = _finite_float(parameters.get(parameter_field))
                if saved is None or expected is None or saved != expected:
                    layer.fail(
                        "cfl_raw_gate_profiles:geometry_mismatch:"
                        f"{case_id}:{position}:{saved_field}"
                    )
            window = _finite_float(row.get("window"))
            gaussian = _finite_float(row.get("gaussian"))
            delta_cs2 = _finite_float(row.get("delta_cs2"))
            raw_cs2 = _finite_float(row.get("raw_cs2"))
            amplitude = _finite_float(row.get("amplitude"))
            if report_finite or accepted_status:
                if None in (window, gaussian, delta_cs2, raw_cs2, amplitude):
                    layer.fail(
                        f"cfl_raw_gate_profiles:nonfinite_evidence:{case_id}:{position}"
                    )
                    continue
                assert window is not None
                assert gaussian is not None
                assert delta_cs2 is not None
                assert raw_cs2 is not None
                assert amplitude is not None
                shape_roundoff = 8.0 * math.ulp(1.0)
                if not (
                    -shape_roundoff <= window <= 1.0 + shape_roundoff
                    and -shape_roundoff
                    <= gaussian
                    <= 1.0 + shape_roundoff
                ):
                    layer.fail(
                        f"cfl_raw_gate_profiles:shape_bounds_invalid:{case_id}:{position}"
                    )
                if delta_cs2 != amplitude * (gaussian * window):
                    layer.fail(
                        f"cfl_raw_gate_profiles:delta_identity_mismatch:{case_id}:{position}"
                    )
                if accepted_status and not (0.0 < raw_cs2 <= 1.0):
                    layer.fail(
                        f"cfl_raw_gate_profiles:accepted_cs2_invalid:{case_id}:{position}"
                    )


def _validate_case_consistency(
    packet: Path,
    *,
    matter_model: str = "bsk24",
    case_plan: list[dict[str, str]] | None,
    case_ledger: list[dict[str, str]] | None,
    raw_gate: Any,
    accepted_rejected: Any,
    metadata: Any,
    layer: _Layer,
    bsk24_raw_profile_evidence: (
        tuple[Any, set[str], set[str]] | None
    ) = None,
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

    raw_schema: Any = None
    case_reports: Any = {}
    if isinstance(raw_gate, dict):
        raw_schema = raw_gate.get("schema_id")
        if raw_schema not in {RAW_GATE_SCHEMA, LEGACY_RAW_GATE_SCHEMA}:
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
        if raw_schema == RAW_GATE_SCHEMA:
            if (
                raw_gate.get(
                    "complete_raw_proposal_assessment_authoritative"
                )
                is not True
            ):
                layer.fail("raw_gate:complete_raw_authority_not_confirmed")
            if raw_gate.get("selected_retained_domain_authoritative") is not True:
                layer.fail("raw_gate:selected_domain_authority_not_confirmed")
            if raw_gate.get("selected_domain_policy") != (
                "prefix_through_first_continuous_cs2_equals_one"
            ):
                layer.fail("raw_gate:invalid_selected_domain_policy")
            hard_rejected = set(raw_gate.get("hard_rejected_case_ids", []))
            unresolved = set(raw_gate.get("unresolved_case_ids", []))
            if hard_rejected & unresolved or hard_rejected | unresolved != raw_rejected:
                layer.fail("raw_gate:rejected_subsets_mismatch")
        else:
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
            if report.get("status") != "accepted_raw_local_physics_gate":
                layer.fail(
                    f"raw_gate:accepted_case_failed_selected_domain_gate:{case_id}"
                )
                continue
            if matter_model == "cfl":
                required_true = (
                    "evaluation_precedes_reconstruction_and_stellar_work",
                    "amplitude_interval_passed",
                    "finite_values",
                    "positive_energy_density",
                    "nonnegative_pressure_including_zero_surface",
                    "full_declared_domain_authoritative",
                    "full_declared_domain_passed",
                )
                if any(report.get(field) is not True for field in required_true):
                    layer.fail(f"raw_gate:cfl_accepted_flags_invalid:{case_id}")
                if report.get("schema_version") != "cfl_raw_local_physics_gate_v1":
                    layer.fail(f"raw_gate:cfl_report_schema_invalid:{case_id}")
                if report.get("first_failure") is not None:
                    layer.fail(f"raw_gate:cfl_accepted_case_has_failure:{case_id}")
                if (
                    report.get("clipping_clamping_smoothing_posthoc_repair")
                    != "none"
                ):
                    layer.fail(f"raw_gate:cfl_repair_claim:{case_id}")
                domain = report.get("complete_declared_domain_mev_fm3")
                surface = report.get("surface")
                try:
                    lower = float(domain[0])
                    upper = float(domain[1])
                    pressure_max = float(
                        report.get("raw_maximum_pressure_mev_fm3")
                    )
                    domain_valid = bool(
                        math.isfinite(lower)
                        and math.isfinite(upper)
                        and math.isfinite(pressure_max)
                        and 0.0 < lower < upper
                        and pressure_max > 0.0
                    )
                except (TypeError, ValueError, IndexError):
                    lower = upper = pressure_max = math.nan
                    domain_valid = False
                surface_valid = bool(
                    isinstance(surface, dict)
                    and surface.get("preserved_exactly") is True
                    and _finite_float(surface.get("epsilon_mev_fm3")) == lower
                    and _finite_float(surface.get("pressure_mev_fm3")) == 0.0
                    and _finite_float(surface.get("window")) == 0.0
                    and _finite_float(surface.get("deformation_delta_cs2"))
                    == 0.0
                )
                if not domain_valid or not surface_valid:
                    layer.fail(f"raw_gate:cfl_domain_or_surface_invalid:{case_id}")
                ledger_row = ledger_by_id[case_id]
                expected_ledger = {
                    "acceptance_domain": "full_retained_domain",
                    "full_domain_gate_status": (
                        "assessed_causal_through_direct_endpoint"
                    ),
                    "selected_domain_status": (
                        "accepted_selected_retained_domain"
                    ),
                    "retained_endpoint_reason": (
                        "formula_derived_cfl_domain_endpoint"
                    ),
                }
                for field, expected in expected_ledger.items():
                    if ledger_row.get(field) != expected:
                        layer.fail(
                            f"raw_gate:cfl_ledger_mismatch:{case_id}:{field}"
                        )
                if str(
                    ledger_row.get(
                        "complete_raw_proposal_causal_through_direct_endpoint",
                        "",
                    )
                ).lower() not in {"true", "1"}:
                    layer.fail(
                        f"raw_gate:cfl_complete_domain_status_invalid:{case_id}"
                    )
                for field, expected in (
                    ("retained_epsilon_max_mev_fm3", upper),
                    ("retained_pressure_max_mev_fm3", pressure_max),
                ):
                    if _finite_float(ledger_row.get(field)) != expected:
                        layer.fail(
                            f"raw_gate:cfl_ledger_endpoint_mismatch:{case_id}:{field}"
                        )
                continue
            if raw_schema == LEGACY_RAW_GATE_SCHEMA:
                if ledger_by_id[case_id].get("acceptance_domain") not in (
                    None,
                    "",
                    "full_retained_domain",
                ):
                    layer.fail(
                        f"raw_gate:invalid_full_domain_selection:{case_id}"
                    )
                continue
            required_hard_flags = (
                "finite_values",
                "positive_energy_density",
                "positive_pressure",
                "complete_raw_proposal_mechanically_stable",
                "complete_raw_pressure_numerically_usable",
                "strictly_monotone_pressure_implied",
            )
            if any(report.get(field) is not True for field in required_hard_flags):
                layer.fail(f"raw_gate:accepted_hard_flags_not_passed:{case_id}")
            if report.get("first_failure") is not None:
                layer.fail(f"raw_gate:accepted_case_has_failure:{case_id}")
            continuous = report.get("continuous_resolution_certificate")
            pressure_certificate = report.get(
                "raw_pressure_reconstruction_certificate"
            )
            retained_certificate = report.get(
                "retained_tabulation_resolution_certificate"
            )
            continuous_status = (
                continuous.get("status")
                if isinstance(continuous, dict)
                else None
            )
            pressure_status = (
                pressure_certificate.get("status")
                if isinstance(pressure_certificate, dict)
                else None
            )
            retained_status = (
                retained_certificate.get("status")
                if isinstance(retained_certificate, dict)
                else None
            )
            if continuous_status not in {
                "resolved_geometry_aware_sampling",
                "resolved_exact_zero_amplitude_identity_sampling",
            }:
                layer.fail(f"raw_gate:accepted_continuous_resolution_invalid:{case_id}")
            if pressure_status != "resolved_strictly_increasing_raw_pressure":
                layer.fail(f"raw_gate:accepted_pressure_resolution_invalid:{case_id}")
            if retained_status not in {
                "resolved_tabulation_resolution",
                "resolved_exact_baseline_identity_grid",
            }:
                layer.fail(f"raw_gate:accepted_tabulation_resolution_invalid:{case_id}")
            retained = report.get("retained_domain")
            if not isinstance(retained, dict):
                layer.fail(f"raw_gate:missing_retained_domain:{case_id}")
                continue
            reason = retained.get("endpoint_reason")
            expected_domain = {
                "direct_bsk24_causal_endpoint": "full_retained_domain",
                "published_bsk24_fit_endpoint": "full_retained_domain",
                "first_continuous_causal_crossing": (
                    "through_first_continuous_causal_crossing"
                ),
            }.get(reason)
            if (
                expected_domain is None
                or report.get("complete_raw_proposal_assessed") is not True
                or report.get("selected_retained_domain_authoritative") is not True
                or report.get("selected_retained_domain_passed") is not True
                or retained.get("passed") is not True
                or retained.get("resolution_certified") is not True
            ):
                layer.fail(f"raw_gate:invalid_retained_domain:{case_id}")
                continue
            ledger_row = ledger_by_id[case_id]
            if ledger_row.get("acceptance_domain") != expected_domain:
                layer.fail(f"raw_gate:ledger_domain_mismatch:{case_id}")
            complete_raw_causal = report.get(
                "complete_raw_proposal_causal_through_direct_endpoint"
            )
            if not isinstance(complete_raw_causal, bool):
                layer.fail(f"raw_gate:complete_domain_status_missing:{case_id}")
            else:
                expected_full_status = (
                    "assessed_causal_through_direct_endpoint"
                    if complete_raw_causal
                    else "assessed_noncausal_beyond_first_retained_crossing"
                )
                if (
                    ledger_row.get("full_domain_gate_status")
                    != expected_full_status
                ):
                    layer.fail(
                        f"raw_gate:ledger_full_domain_status_mismatch:{case_id}"
                    )
            raw_domain = report.get(
                "complete_proposed_retained_domain_mev_fm3"
            )
            try:
                raw_lower, raw_upper = (
                    float(raw_domain[0]),
                    float(raw_domain[1]),
                )
                retained_lower = float(retained.get("epsilon_min_mev_fm3"))
                retained_upper = float(retained.get("epsilon_max_mev_fm3"))
                retained_cs2 = float(retained.get("cs2_at_endpoint"))
                retained_pressure = float(retained.get("pressure_max_mev_fm3"))
                endpoints_finite = all(
                    math.isfinite(value)
                    for value in (
                        raw_lower,
                        raw_upper,
                        retained_lower,
                        retained_upper,
                        retained_cs2,
                        retained_pressure,
                    )
                )
            except (TypeError, ValueError, IndexError):
                endpoints_finite = False
                raw_lower = raw_upper = retained_lower = retained_upper = math.nan
                retained_cs2 = retained_pressure = math.nan
            crossing = retained.get("first_causal_crossing")
            endpoint_structure_valid = bool(
                endpoints_finite
                and 0.0 < raw_lower < raw_upper
                and retained_lower == raw_lower
                and raw_lower < retained_upper <= raw_upper
                and 0.0 < retained_cs2 <= 1.0
                and retained_pressure > 0.0
            )
            if reason == "direct_bsk24_causal_endpoint":
                endpoint_structure_valid = bool(
                    endpoint_structure_valid
                    and complete_raw_causal is True
                    and retained_upper == raw_upper
                    and crossing is None
                )
            elif reason == "published_bsk24_fit_endpoint":
                endpoint_structure_valid = bool(
                    endpoint_structure_valid
                    and complete_raw_causal is True
                    and retained_upper == raw_upper
                    and crossing is None
                    and report.get("declared_assessment_endpoint")
                    == "published_bsk24_fit_endpoint"
                )
            elif reason == "first_continuous_causal_crossing":
                crossing_epsilon = (
                    _finite_float(crossing.get("epsilon_mev_fm3"))
                    if isinstance(crossing, dict)
                    else None
                )
                crossing_cs2 = (
                    _finite_float(crossing.get("cs2_at_endpoint"))
                    if isinstance(crossing, dict)
                    else None
                )
                endpoint_structure_valid = bool(
                    endpoint_structure_valid
                    and complete_raw_causal is False
                    and isinstance(crossing, dict)
                    and crossing.get("status")
                    == "resolved_first_continuous_causal_crossing"
                    and crossing.get("continuous_crossing_bracketed") is True
                    and crossing.get(
                        "crossing_included_to_governed_tolerance"
                    )
                    is True
                    and crossing.get("cs2_values_modified") is False
                    and crossing_epsilon == retained_upper
                    and crossing_cs2 == retained_cs2
                )
            if not endpoint_structure_valid:
                layer.fail(f"raw_gate:invalid_retained_endpoint_structure:{case_id}")
            if ledger_row.get("selected_domain_status") != (
                "accepted_selected_retained_domain"
            ):
                layer.fail(
                    f"raw_gate:ledger_selected_domain_status_mismatch:{case_id}"
                )
            if ledger_row.get("retained_endpoint_reason") != reason:
                layer.fail(f"raw_gate:ledger_endpoint_reason_mismatch:{case_id}")
            for ledger_field, report_field in (
                ("retained_epsilon_max_mev_fm3", "epsilon_max_mev_fm3"),
                ("retained_pressure_max_mev_fm3", "pressure_max_mev_fm3"),
            ):
                saved = ledger_row.get(ledger_field)
                expected = retained.get(report_field)
                try:
                    matches = float(saved) == float(expected)
                except (TypeError, ValueError):
                    matches = False
                if not matches:
                    layer.fail(
                        f"raw_gate:ledger_endpoint_value_mismatch:{case_id}:"
                        f"{ledger_field}"
                    )
        if raw_schema == RAW_GATE_SCHEMA and matter_model == "cfl":
            for case_id in rejected:
                report = (
                    case_reports.get(case_id)
                    if isinstance(case_reports, dict)
                    else None
                )
                if not isinstance(report, dict):
                    continue
                status = report.get("status")
                if status not in {
                    "rejected_raw_local_physics_gate",
                    "unresolved_raw_local_physics_gate",
                }:
                    layer.fail(f"raw_gate:cfl_invalid_rejected_status:{case_id}")
                    continue
                if report.get("full_declared_domain_authoritative") is not True:
                    layer.fail(f"raw_gate:cfl_rejected_domain_not_assessed:{case_id}")
                if report.get("full_declared_domain_passed") is not False:
                    layer.fail(f"raw_gate:cfl_rejected_domain_passed:{case_id}")
                if not isinstance(report.get("first_failure"), dict):
                    layer.fail(f"raw_gate:cfl_rejected_failure_missing:{case_id}")
                ledger_row = ledger_by_id[case_id]
                expected_full = (
                    "assessed_hard_rejected"
                    if status == "rejected_raw_local_physics_gate"
                    else "assessed_unresolved"
                )
                expected_selected = (
                    "rejected_no_selected_retained_domain"
                    if status == "rejected_raw_local_physics_gate"
                    else "unresolved_no_selected_retained_domain"
                )
                if ledger_row.get("acceptance_domain") != "none":
                    layer.fail(f"raw_gate:cfl_rejected_domain_not_none:{case_id}")
                if ledger_row.get("full_domain_gate_status") != expected_full:
                    layer.fail(f"raw_gate:cfl_rejected_full_status:{case_id}")
                if ledger_row.get("selected_domain_status") != expected_selected:
                    layer.fail(
                        f"raw_gate:cfl_rejected_selected_status:{case_id}"
                    )
        elif raw_schema == RAW_GATE_SCHEMA:
            for case_id in rejected:
                report = (
                    case_reports.get(case_id)
                    if isinstance(case_reports, dict)
                    else None
                )
                if not isinstance(report, dict):
                    continue
                if report.get("status") not in {
                    "rejected_raw_local_physics_gate",
                    "unresolved_raw_local_physics_gate",
                }:
                    layer.fail(f"raw_gate:invalid_rejected_status:{case_id}")
                if report.get("complete_raw_proposal_assessed") is not True:
                    layer.fail(f"raw_gate:rejected_raw_not_preserved:{case_id}")
                if ledger_by_id[case_id].get("acceptance_domain") != "none":
                    layer.fail(f"raw_gate:rejected_domain_not_none:{case_id}")
                expected_statuses = {
                    "rejected_raw_local_physics_gate": (
                        "assessed_hard_rejected",
                        "rejected_no_selected_retained_domain",
                    ),
                    "unresolved_raw_local_physics_gate": (
                        "assessed_unresolved",
                        "unresolved_no_selected_retained_domain",
                    ),
                }.get(report.get("status"))
                if expected_statuses is None:
                    continue
                expected_full_status, expected_selected_status = expected_statuses
                ledger_row = ledger_by_id[case_id]
                if ledger_row.get("full_domain_gate_status") != expected_full_status:
                    layer.fail(
                        f"raw_gate:ledger_full_domain_status_mismatch:{case_id}"
                    )
                if (
                    ledger_row.get("selected_domain_status")
                    != expected_selected_status
                ):
                    layer.fail(
                        f"raw_gate:ledger_selected_domain_status_mismatch:{case_id}"
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
                "acceptance_domain",
                "raw_gate_status",
                "full_domain_gate_status",
                "selected_domain_status",
                "complete_raw_proposal_causal_through_direct_endpoint",
                "retained_epsilon_max_mev_fm3",
                "retained_pressure_max_mev_fm3",
                "retained_endpoint_reason",
                "requested_fixed_masses_status",
                "maximum_mass_availability_status",
                "student_view_eligibility_status",
                "rejection_reason",
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
        if raw_schema == RAW_GATE_SCHEMA:
            unresolved = set(raw_gate.get("unresolved_case_ids", []))
            if set(metadata.get("unresolved_case_ids", [])) != unresolved:
                layer.fail("metadata:unresolved_case_ids_mismatch")
            if metadata.get("unresolved_case_count") != len(unresolved):
                layer.fail("metadata:unresolved_case_count_mismatch")

    for row in case_ledger:
        case_id = str(row.get("case_id") or "")
        accepted_row = row.get("status") == "accepted"
        fixed_status = row.get("requested_fixed_masses_status")
        eligibility = row.get("student_view_eligibility_status")
        maximum_status = row.get("maximum_mass_availability_status")
        if not accepted_row:
            expected_eligibility = "evidence_only_raw_gate_not_accepted"
        elif fixed_status == "not_requested":
            expected_eligibility = "eligible_thermodynamic_case"
        elif fixed_status == "all_requested_fixed_masses_succeeded":
            expected_eligibility = (
                "eligible_all_requested_fixed_masses_succeeded"
            )
        else:
            expected_eligibility = "ineligible_requested_fixed_masses_incomplete"
        if eligibility != expected_eligibility:
            layer.fail(
                f"case_ledger:student_eligibility_mismatch:{case_id}"
            )
        report = case_reports.get(case_id) if isinstance(case_reports, dict) else None
        if isinstance(report, dict) and row.get("raw_gate_status") != report.get(
            "status"
        ):
            layer.fail(f"case_ledger:raw_gate_status_mismatch:{case_id}")
        if accepted_row:
            if _inventory_scalar(row.get("rejection_reason")):
                layer.fail(f"case_ledger:accepted_has_rejection_reason:{case_id}")
            if row.get("pressure_reconstruction") != "completed":
                layer.fail(
                    f"case_ledger:accepted_reconstruction_status_mismatch:{case_id}"
                )
            if row.get("clipping_or_repair") != "none":
                layer.fail(f"case_ledger:accepted_repair_claim:{case_id}")
        else:
            expected_reason = (
                json.dumps(report.get("first_failure"), sort_keys=True)
                if isinstance(report, dict)
                else None
            )
            if row.get("rejection_reason") != expected_reason:
                layer.fail(f"case_ledger:rejection_reason_mismatch:{case_id}")
            for field, expected in (
                ("requested_fixed_masses_status", "not_applicable_raw_gate_not_accepted"),
                ("maximum_mass_availability_status", "not_applicable_raw_gate_not_accepted"),
                ("pressure_reconstruction", "skipped_due_to_raw_gate_rejection"),
                ("stellar_calculation", "skipped_due_to_raw_gate_rejection"),
                ("clipping_or_repair", "none"),
            ):
                if row.get(field) != expected:
                    layer.fail(
                        f"case_ledger:rejected_lifecycle_mismatch:"
                        f"{case_id}:{field}"
                    )
        if maximum_status and maximum_status not in {
            "not_requested",
            "not_applicable_raw_gate_not_accepted",
            "unavailable_no_reporting_stage",
            "unavailable_missing_assessment",
            "resolved_bracketed_and_refined",
        } and not maximum_status.startswith("unavailable_"):
            layer.fail(
                "case_ledger:invalid_maximum_mass_availability_status:"
                f"{row.get('case_id', '')}"
            )

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
    raw_rows = _read_csv(packet, "raw_gate_profiles.csv", layer)
    if matter_model == "bsk24":
        profile_raw_gate = raw_gate
        profile_accepted = accepted
        profile_rejected = rejected
        if bsk24_raw_profile_evidence is not None:
            (
                profile_raw_gate,
                profile_accepted,
                profile_rejected,
            ) = bsk24_raw_profile_evidence
        profile_raw_schema = (
            profile_raw_gate.get("schema_id")
            if isinstance(profile_raw_gate, dict)
            else None
        )
        _validate_raw_gate_profiles(
            raw_rows,
            raw_gate=profile_raw_gate,
            raw_schema=profile_raw_schema,
            accepted=profile_accepted,
            rejected=profile_rejected,
            layer=layer,
        )
    layer.checks["accepted_case_count"] = len(accepted)
    layer.checks["rejected_case_count"] = len(rejected)
    return accepted, rejected


def _inventory_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower()
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

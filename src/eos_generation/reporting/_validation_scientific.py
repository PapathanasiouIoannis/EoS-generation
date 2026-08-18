"""Stellar-output and scientific-completeness validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from eos_generation._internal.saved_tables import (
    classify_saved_tidal_rows,
    summarize_fixed_mass_response_population,
)
from eos_generation.reporting._validation_cases import (
    _case_ids,
    _csv_json_records_equal,
    _inventory_records,
)
from eos_generation.reporting._validation_io import (
    _Layer,
    _load_json,
    _read_csv,
)

STELLAR_STATUS_SCHEMA = "eos_generation_stellar_status_summary_v1"

_FIXED_MASS_FAIL_CLOSED_STATUSES = frozenset(
    {
        "unavailable_failed_maximum_mass_screen",
        "unavailable_maximum_mass_not_resolved",
        "unavailable_not_bracketed",
    }
)
_FIXED_MASS_RESULT_COLUMNS = (
    "mass_msun",
    "mass_residual_msun",
    "radius_km",
    "central_pressure_mev_fm3",
    "central_energy_density_mev_fm3",
    "central_sound_speed_squared",
    "k2",
    "lambda_dimensionless",
    "root_evaluation_count",
)

STELLAR_REQUIRED_FILES = (
    "stellar_sequences.csv",
    "fixed_mass_observables.csv",
    "stellar_convergence.json",
)

CURRENT_STELLAR_STATUS_FILES = (
    "stellar_status_summary.csv",
    "stellar_status_summary.json",
)

EXTENDED_CORE_REQUIRED_FILES = (
    "radial_profiles.csv",
    "deformation_support_fractions.csv",
)


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _extended_output_claimed(configuration: Any, metadata: Any) -> bool:
    if isinstance(configuration, dict) and configuration.get(
        "extended_stellar_diagnostics_enabled"
    ) is True:
        return True
    if not isinstance(metadata, dict):
        return False
    extended = metadata.get("extended_tables")
    if isinstance(extended, bool):
        return extended
    if isinstance(extended, dict):
        return bool(extended)
    if isinstance(extended, list):
        return bool(extended)
    return False


def _validate_fixed_mass_completeness(
    packet: Path,
    configuration: Mapping[str, Any],
    accepted: set[str],
    layer: _Layer,
    *,
    require_tides: bool = True,
) -> None:
    rows = _read_csv(packet, "fixed_mass_observables.csv", layer)
    if rows is None:
        return
    import pandas as pd

    frame = pd.DataFrame(rows)
    tidal_classification = classify_saved_tidal_rows(
        frame, schema="fixed_mass"
    )
    stages = [
        str(item.get("name"))
        for item in configuration.get("tov_stages", [])
        if isinstance(item, dict) and item.get("name")
    ]
    masses = [
        value
        for item in configuration.get("fixed_masses_msun", [])
        if (value := _float_or_none(item)) is not None
    ]
    case_ids = {"direct", *accepted}
    expected = {
        (case_id, stage, mass.hex())
        for case_id in case_ids
        for stage in stages
        for mass in masses
    }
    actual: set[tuple[str, str, str]] = set()
    duplicates: set[tuple[str, str, str]] = set()
    background_unavailable = 0
    unavailable_reasons: dict[str, int] = {}
    tidal_failures = 0
    missing_reason = 0
    has_tidal_failure_reason = not rows or "tidal_failure_reason" in rows[0]
    if not has_tidal_failure_reason:
        layer.fail("fixed_mass:tidal_failure_reason_column_missing")
    for position, row in enumerate(rows):
        mass = _float_or_none(row.get("target_mass_msun"))
        if mass is None:
            layer.fail(f"fixed_mass:invalid_target_mass:{row.get('case_id', '')}")
            continue
        key = (row.get("case_id", ""), row.get("stage", ""), mass.hex())
        if key in actual:
            duplicates.add(key)
        actual.add(key)
        classification = tidal_classification.iloc[position]
        background_ok = bool(classification["background_success"])
        if not background_ok:
            background_unavailable += 1
            status = str(row.get("status", ""))
            reason = str(row.get("reason") or "").strip()
            if status not in _FIXED_MASS_FAIL_CLOSED_STATUSES:
                layer.fail(
                    "fixed_mass:invalid_background_status:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}:{status}"
                )
            elif not reason:
                layer.fail(
                    "fixed_mass:unavailable_reason_missing:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}:{status}"
                )
            else:
                unavailable_reasons[reason] = (
                    unavailable_reasons.get(reason, 0) + 1
                )
            contaminated = [
                column
                for column in _FIXED_MASS_RESULT_COLUMNS
                if _float_or_none(row.get(column)) is not None
            ]
            if contaminated:
                layer.fail(
                    "fixed_mass:unavailable_row_has_observables:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}:"
                    f"{','.join(contaminated)}"
                )
            if str(row.get("tidal_status") or "").strip() or str(
                row.get("tidal_failure_reason") or ""
            ).strip():
                layer.fail(
                    "fixed_mass:unavailable_row_has_tidal_claim:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}"
                )
        tidal_ok = bool(classification["tidal_valid"])
        if background_ok and require_tides and not tidal_ok:
            tidal_failures += 1
            reason = str(classification["tidal_validity_reason"])
            layer.fail(
                "fixed_mass:invalid_tidal_row:"
                f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                f"{row.get('target_mass_msun', '')}:{reason}"
            )
            if (
                row.get("tidal_status") == "failed_closed"
                and not row.get("tidal_failure_reason")
            ):
                missing_reason += 1
        elif (
            background_ok
            and not require_tides
            and row.get("tidal_status") != "not_requested_background_only"
        ):
            layer.fail(
                "fixed_mass:unexpected_tidal_work:"
                f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                f"{row.get('target_mass_msun', '')}"
            )
    if duplicates:
        layer.fail(f"fixed_mass:duplicate_targets:{len(duplicates)}")
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        layer.fail(f"fixed_mass:missing_requested_rows:{len(missing)}")
    if unexpected:
        layer.fail(f"fixed_mass:unexpected_rows:{len(unexpected)}")
    if tidal_failures:
        layer.fail(f"fixed_mass:tidal_incomplete_failed_closed:{tidal_failures}")
    if missing_reason:
        layer.fail(f"fixed_mass:tidal_failure_reason_missing:{missing_reason}")
    layer.checks["fixed_mass_requested_rows"] = len(expected)
    layer.checks["fixed_mass_present_rows"] = len(actual)
    layer.checks["fixed_mass_background_failures"] = background_unavailable
    layer.checks["fixed_mass_fail_closed_reasons"] = dict(
        sorted(unavailable_reasons.items())
    )
    layer.checks["fixed_mass_tidal_failures"] = tidal_failures
    layer.checks["fixed_mass_tidal_invalidity_reasons"] = {
        str(key): int(value)
        for key, value in tidal_classification.loc[
            tidal_classification["background_success"]
            & ~tidal_classification["tidal_valid"],
            "tidal_validity_reason",
        ]
        .value_counts()
        .sort_index()
        .items()
    }


def _validate_sequence_completeness(
    packet: Path,
    configuration: Mapping[str, Any],
    accepted: set[str],
    layer: _Layer,
    *,
    require_tides: bool = True,
    require_configured_count: bool = True,
) -> None:
    rows = _read_csv(packet, "stellar_sequences.csv", layer)
    if rows is None:
        return
    import pandas as pd

    frame = pd.DataFrame(rows)
    tidal_classification = classify_saved_tidal_rows(
        frame, schema="sequence"
    )
    expected_groups: dict[tuple[str, str], int] = {}
    for case_id in {"direct", *accepted}:
        for stage in configuration.get("tov_stages", []):
            if not isinstance(stage, dict) or not stage.get("name"):
                continue
            points = stage.get("sequence_points")
            if isinstance(points, int) and not isinstance(points, bool):
                expected_groups[(case_id, str(stage["name"]))] = points
    actual_groups: dict[tuple[str, str], int] = {}
    background_failures = 0
    tidal_failures = 0
    tidal_failures_without_reason = 0
    for position, row in enumerate(rows):
        key = (row.get("case_id", ""), row.get("stage", ""))
        actual_groups[key] = actual_groups.get(key, 0) + 1
        classification = tidal_classification.iloc[position]
        background_ok = bool(classification["background_success"])
        tidal_ok = bool(classification["tidal_valid"])
        if not background_ok:
            background_failures += 1
        elif require_tides and not tidal_ok:
            tidal_failures += 1
            reason = str(classification["tidal_validity_reason"])
            layer.fail(
                "stellar_sequences:invalid_tidal_row:"
                f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                f"{row.get('attempted_index', '')}:{reason}"
            )
            if (
                row.get("tidal_status") == "failed_closed"
                and not row.get("tidal_failure_reason")
            ):
                tidal_failures_without_reason += 1
        elif (
            not require_tides
            and row.get("tidal_status") != "not_requested_background_only"
        ):
            layer.fail(
                "stellar_sequences:unexpected_tidal_work:"
                f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                f"{row.get('attempted_index', '')}"
            )
    for key, expected_count in expected_groups.items():
        actual_count = actual_groups.get(key, 0)
        if require_configured_count and actual_count != expected_count:
            layer.fail(
                "stellar_sequences:requested_count_mismatch:"
                f"{key[0]}:{key[1]}:{actual_count}:{expected_count}"
            )
        elif not require_configured_count and actual_count == 0:
            layer.fail(
                "stellar_sequences:background_group_missing:"
                f"{key[0]}:{key[1]}"
            )
    for key in sorted(set(actual_groups) - set(expected_groups)):
        layer.fail(f"stellar_sequences:unexpected_group:{key[0]}:{key[1]}")
    if background_failures:
        # Failed rows may be valid fail-closed evidence, but the requested
        # sequence is not a complete set of solved backgrounds.
        layer.fail(f"stellar_sequences:background_failures:{background_failures}")
    if tidal_failures_without_reason:
        layer.fail(
            "stellar_sequences:tidal_failures_without_reason:"
            f"{tidal_failures_without_reason}"
        )
    if tidal_failures:
        layer.fail(f"stellar_sequences:tidal_incomplete_failed_closed:{tidal_failures}")
    layer.checks["stellar_sequence_requested_rows"] = (
        sum(expected_groups.values())
        if require_configured_count
        else len(rows)
    )
    layer.checks["stellar_sequence_present_rows"] = len(rows)
    layer.checks["stellar_sequence_background_failures"] = background_failures
    layer.checks["stellar_sequence_tidal_failures"] = tidal_failures
    layer.checks["stellar_sequence_tidal_invalidity_reasons"] = {
        str(key): int(value)
        for key, value in tidal_classification.loc[
            tidal_classification["background_success"]
            & ~tidal_classification["tidal_valid"],
            "tidal_validity_reason",
        ]
        .value_counts()
        .sort_index()
        .items()
    }


def _integer_or_none(value: Any) -> int | None:
    numeric = _float_or_none(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _configuration_background_tov_requested(
    configuration: Mapping[str, Any],
) -> bool:
    """Return the ordinary packet's single stellar-execution switch."""

    return configuration.get("stellar_enabled") is True


def _configuration_fixed_mass_requested(
    configuration: Mapping[str, Any],
) -> bool:
    return configuration.get("stellar_enabled") is True


def _configuration_tidal_requested(
    configuration: Mapping[str, Any],
) -> bool:
    return configuration.get("stellar_enabled") is True


def _saved_lifecycle_configuration(
    configuration: Mapping[str, Any],
) -> SimpleNamespace:
    """Expose only the serialized controls used by lifecycle completeness."""

    stages = tuple(
        SimpleNamespace(
            name=str(item.get("name", "")),
            sequence_points=int(item.get("sequence_points", 0)),
        )
        for item in configuration.get("tov_stages", ())
        if isinstance(item, Mapping) and item.get("name")
    )
    return SimpleNamespace(
        background_tov_requested=(
            _configuration_background_tov_requested(configuration)
        ),
        fixed_mass_background_requested=(
            _configuration_fixed_mass_requested(configuration)
        ),
        fixed_masses_msun=tuple(
            value
            for item in configuration.get("fixed_masses_msun", ())
            if (value := _float_or_none(item)) is not None
        ),
        tov_stages=stages,
    )


def _validate_final_lifecycle(
    packet: Path,
    configuration: Mapping[str, Any],
    accepted: set[str],
    layer: _Layer,
) -> None:
    rows = _read_csv(packet, "case_ledger.csv", layer)
    if rows is None:
        return
    stellar_enabled = _configuration_background_tov_requested(configuration)
    completed_stellar: set[str] = set()
    if stellar_enabled:
        import pandas as pd

        from eos_generation._internal.lifecycle import (
            _completed_stellar_case_ids,
        )

        sequences = (
            pd.read_csv(packet / "stellar_sequences.csv")
            if (packet / "stellar_sequences.csv").is_file()
            else None
        )
        fixed = (
            pd.read_csv(packet / "fixed_mass_observables.csv")
            if (packet / "fixed_mass_observables.csv").is_file()
            else None
        )
        saved_config = _saved_lifecycle_configuration(configuration)
        accepted_physical_ids = tuple(
            dict.fromkeys(
                str(row.get("physical_case_id") or row.get("case_id", ""))
                for row in rows
                if row.get("case_id", "") in accepted
            )
        )
        completed_stellar = _completed_stellar_case_ids(
            sequences,
            fixed,
            saved_config,
            accepted_case_ids=accepted_physical_ids,
        )
    for row in rows:
        case_id = row.get("case_id", "")
        accepted_row = case_id in accepted
        physical_case_id = str(
            row.get("physical_case_id") or case_id
        )
        expected_stellar = (
            "completed"
            if (
                accepted_row
                and stellar_enabled
                and physical_case_id in completed_stellar
            )
            else "incomplete_or_failed"
            if accepted_row and stellar_enabled
            else "disabled"
            if accepted_row
            else "skipped_due_to_raw_gate_rejection"
        )
        if row.get("stellar_calculation") != expected_stellar:
            layer.fail(
                "case_lifecycle:stellar_status_mismatch:"
                f"{case_id}:{row.get('stellar_calculation', '')}:{expected_stellar}"
            )
        expected_reconstruction = (
            "completed"
            if accepted_row
            else "skipped_due_to_raw_gate_rejection"
        )
        if row.get("pressure_reconstruction") != expected_reconstruction:
            layer.fail(
                "case_lifecycle:reconstruction_status_mismatch:"
                f"{case_id}:{row.get('pressure_reconstruction', '')}:"
                f"{expected_reconstruction}"
            )


def _validate_stellar_status_reporting(
    packet: Path,
    configuration: Mapping[str, Any],
    metadata: Any,
    layer: _Layer,
) -> None:
    import pandas as pd

    summary_rows = _read_csv(packet, "stellar_status_summary.csv", layer)
    sequence_rows = _read_csv(packet, "stellar_sequences.csv", layer)
    fixed_rows = _read_csv(packet, "fixed_mass_observables.csv", layer)
    if summary_rows is None or sequence_rows is None or fixed_rows is None:
        return
    summary = pd.DataFrame(summary_rows)
    frames = {
        "sequence": pd.DataFrame(sequence_rows),
        "fixed_mass": pd.DataFrame(fixed_rows),
    }
    totals: dict[str, dict[str, int]] = {}
    count_fields = (
        "requested_count",
        "background_success_count",
        "background_failure_count",
        "tidal_validated_count",
        "tidal_failed_closed_count",
        "tidal_unavailable_count",
    )
    stages = [
        str(item["name"])
        for item in configuration.get("tov_stages", [])
        if isinstance(item, dict) and item.get("name")
    ]
    total_background_failures = 0
    total_failed_closed = 0
    total_unavailable = 0
    for scope, frame in frames.items():
        classification = classify_saved_tidal_rows(frame, schema=scope)
        scope_totals = {field: 0 for field in count_fields}
        for stage in stages:
            stage_mask = frame["stage"].astype(str).eq(stage)
            stage_rows = frame.loc[stage_mask]
            stage_classification = classification.loc[stage_rows.index]
            requested = int(len(stage_rows))
            background = int(stage_classification["background_success"].sum())
            validated = int(stage_classification["tidal_valid"].sum())
            tidal_status = (
                stage_rows["tidal_status"].astype(str)
                if "tidal_status" in stage_rows.columns
                else pd.Series("", index=stage_rows.index, dtype=str)
            )
            failed_closed = int(
                (
                    stage_classification["background_success"]
                    & tidal_status.eq("failed_closed")
                ).sum()
            )
            expected = {
                "requested_count": requested,
                "background_success_count": background,
                "background_failure_count": requested - background,
                "tidal_validated_count": validated,
                "tidal_failed_closed_count": failed_closed,
                "tidal_unavailable_count": requested - validated - failed_closed,
            }
            stored = summary.loc[
                summary["scope"].astype(str).eq(scope)
                & summary["stage"].astype(str).eq(stage)
            ]
            if len(stored) != 1:
                layer.fail(
                    f"stellar_status_summary:missing_or_duplicate:{scope}:{stage}"
                )
            else:
                record = stored.iloc[0]
                for field, expected_value in expected.items():
                    if _integer_or_none(record.get(field)) != expected_value:
                        layer.fail(
                            "stellar_status_summary:count_mismatch:"
                            f"{scope}:{stage}:{field}"
                        )
            for field, expected_value in expected.items():
                scope_totals[field] += expected_value
            total_background_failures += expected["background_failure_count"]
            total_failed_closed += expected["tidal_failed_closed_count"]
            total_unavailable += expected["tidal_unavailable_count"]
        totals[scope] = scope_totals

    expected_publication = (
        "partial_background_failures"
        if total_background_failures
        else "partial_tidal_validation"
        if total_failed_closed or total_unavailable
        else "complete_background_and_tidal"
    )
    if isinstance(metadata, dict):
        metadata_summary = metadata.get("stellar_status_summary")
        if not isinstance(metadata_summary, dict):
            layer.fail("metadata:stellar_status_summary_missing")
        else:
            if (
                metadata_summary.get("publication_interpretation_status")
                != expected_publication
            ):
                layer.fail(
                    "metadata:stellar_publication_interpretation_status_mismatch"
                )
            if metadata_summary.get("totals_by_scope") != totals:
                layer.fail("metadata:stellar_totals_by_scope_mismatch")
    layer.checks["recomputed_stellar_status_totals"] = totals


def _validate_response_population_reporting(
    packet: Path,
    configuration: Mapping[str, Any],
    metadata: Any,
    layer: _Layer,
) -> None:
    import pandas as pd

    fixed_rows = _read_csv(packet, "fixed_mass_observables.csv", layer)
    inventory_rows = _read_csv(packet, "plot_inventory.csv", layer)
    if fixed_rows is None or inventory_rows is None:
        return
    stages = [
        str(item["name"])
        for item in configuration.get("tov_stages", [])
        if isinstance(item, dict) and item.get("name")
    ]
    masses = [
        value
        for item in configuration.get("fixed_masses_msun", [])
        if (value := _float_or_none(item)) is not None
    ]
    if not stages or not masses:
        return
    final_stage = stages[-1]
    target_mass = min(masses, key=lambda value: abs(value - 1.4))
    fixed = pd.DataFrame(fixed_rows)
    inventory = {
        str(row.get("figure", "")): row for row in inventory_rows
    }
    metadata_rows = {}
    if isinstance(metadata, dict):
        value = metadata.get("plot_tidal_completeness", [])
        if isinstance(value, list):
            metadata_rows = {
                str(item.get("figure", "")): item
                for item in value
                if isinstance(item, dict)
            }
    checks: dict[str, Any] = {}
    for versus, figure in (
        ("amplitude", "observable_response_vs_amplitude.png"),
        ("delta", "observable_response_vs_delta.png"),
    ):
        _, population = summarize_fixed_mass_response_population(
            fixed,
            final_stage=final_stage,
            target_mass_msun=target_mass,
            versus=versus,
        )
        checks[versus] = population
        stored = inventory.get(figure)
        if stored is None:
            layer.fail(f"plot_inventory:response_figure_missing:{figure}")
            continue
        expected_count = int(population["eligible_deformation_row_count"])
        if expected_count:
            comparisons = {
                "eligible_response_row_count": expected_count,
                "tidal_validated_count": int(
                    population["tidal_validated_count"]
                ),
                "tidal_omitted_count": int(population["tidal_omitted_count"]),
            }
            for field, expected in comparisons.items():
                if _integer_or_none(stored.get(field)) != expected:
                    layer.fail(
                        f"plot_inventory:response_population_mismatch:{figure}:{field}"
                    )
            if stored.get("population_stage") != final_stage:
                layer.fail(
                    f"plot_inventory:response_population_mismatch:{figure}:population_stage"
                )
            stored_mass = _float_or_none(
                stored.get("population_target_mass_msun")
            )
            if stored_mass is None or stored_mass != target_mass:
                layer.fail(
                    f"plot_inventory:response_population_mismatch:{figure}:target_mass"
                )
            metadata_record = metadata_rows.get(figure)
            if metadata_record is None:
                layer.fail(
                    f"metadata:plot_tidal_completeness_missing:{figure}"
                )
            else:
                for field, expected in comparisons.items():
                    if _integer_or_none(metadata_record.get(field)) != expected:
                        layer.fail(
                            f"metadata:response_population_mismatch:{figure}:{field}"
                        )
                if metadata_record.get("population_stage") != final_stage:
                    layer.fail(
                        f"metadata:response_population_mismatch:{figure}:population_stage"
                    )
                metadata_mass = _float_or_none(
                    metadata_record.get("population_target_mass_msun")
                )
                if metadata_mass is None or metadata_mass != target_mass:
                    layer.fail(
                        f"metadata:response_population_mismatch:{figure}:target_mass"
                    )
        elif stored.get("status") != "skipped":
            layer.fail(
                f"plot_inventory:inapplicable_response_not_skipped:{figure}"
            )
    layer.checks["response_populations"] = checks




def _validate_maximum_mass_artifacts(
    packet: Path,
    *,
    configuration: Mapping[str, Any],
    accepted: set[str],
    layer: _Layer,
) -> None:
    """Validate one complete set of saved maximum-mass evidence."""

    for relative in (
        "maximum_mass_screening.csv",
        "maximum_mass_reports.json",
    ):
        if not (packet / relative).is_file():
            layer.fail(f"missing_stellar_output:{relative}")
    maximum_rows = _read_csv(packet, "maximum_mass_screening.csv", layer)
    reports = _load_json(packet, "maximum_mass_reports.json", layer)
    stage_names = tuple(
        str(stage.get("name", ""))
        for stage in configuration.get("tov_stages", ())
        if isinstance(stage, dict) and stage.get("name")
    )
    expected_pairs = {
        (case_id, stage)
        for case_id in {"direct", *accepted}
        for stage in stage_names
    }
    observed_pairs: set[tuple[str, str]] = set()
    if maximum_rows is not None:
        for row in maximum_rows:
            pair = (
                str(row.get("case_id", "")),
                str(row.get("stage", "")),
            )
            observed_pairs.add(pair)
            resolved = (
                str(row.get("maximum_mass_resolved", "")).lower() == "true"
            )
            mass = _float_or_none(row.get("maximum_mass_msun"))
            pressure = _float_or_none(row.get("central_pressure_mev_fm3"))
            left = _float_or_none(row.get("positive_left_secant"))
            right = _float_or_none(row.get("negative_right_secant"))
            if resolved and (
                mass is None
                or pressure is None
                or left is None
                or right is None
                or not left > 0.0
                or not right < 0.0
            ):
                layer.fail(
                    "maximum_mass:resolved_without_turning_point_evidence:"
                    f"{pair[0]}:{pair[1]}"
                )
            if not resolved and mass is not None:
                layer.fail(
                    "maximum_mass:unresolved_row_claims_mass:"
                    f"{pair[0]}:{pair[1]}"
                )
            maximum_mass_tidal_calls = _integer_or_none(
                row.get("tidal_solver_calls_for_maximum_mass")
            )
            if (
                maximum_mass_tidal_calls is not None
                and maximum_mass_tidal_calls != 0
            ):
                layer.fail(
                    "maximum_mass:refinement_used_tidal_solver:"
                    f"{pair[0]}:{pair[1]}"
                )
        if observed_pairs != expected_pairs:
            layer.fail(
                "maximum_mass:case_stage_coverage_mismatch:"
                f"missing={sorted(expected_pairs - observed_pairs)}:"
                f"extra={sorted(observed_pairs - expected_pairs)}"
            )
    if isinstance(reports, dict):
        report_cases = reports.get("cases")
        expected_report_ids = {
            f"{case_id}:{stage}" for case_id, stage in expected_pairs
        }
        if not isinstance(report_cases, dict) or set(report_cases) != expected_report_ids:
            layer.fail("maximum_mass:report_coverage_mismatch")
        elif isinstance(report_cases, dict):
            for report_id, maximum_report in report_cases.items():
                if not isinstance(maximum_report, dict):
                    continue
                report_tidal_calls = _integer_or_none(
                    maximum_report.get("tidal_calculations_performed")
                )
                if report_tidal_calls is not None and report_tidal_calls != 0:
                    layer.fail(
                        "maximum_mass:refinement_used_tidal_solver:"
                        f"{report_id}"
                    )


def _validate_scientific_completeness(
    packet: Path,
    *,
    configuration: Any,
    metadata: Any,
    accepted: set[str],
) -> dict[str, Any]:
    layer = _Layer()
    if not isinstance(configuration, dict):
        layer.fail("configuration_unavailable")
    else:
        stellar_enabled = configuration.get("stellar_enabled") is True
        background_requested = _configuration_background_tov_requested(
            configuration
        )
        tidal_requested = _configuration_tidal_requested(configuration)
        extended_claimed = _extended_output_claimed(configuration, metadata)
        layer.checks["stellar_enabled"] = stellar_enabled
        layer.checks["background_tov_requested"] = background_requested
        layer.checks["extended_outputs_claimed"] = extended_claimed
        if background_requested:
            required_stellar_files = (
                *STELLAR_REQUIRED_FILES,
                *CURRENT_STELLAR_STATUS_FILES,
                "maximum_mass_screening.csv",
                "maximum_mass_reports.json",
            )
            for relative in required_stellar_files:
                if not (packet / relative).is_file():
                    layer.fail(f"missing_stellar_output:{relative}")

            _validate_maximum_mass_artifacts(
                packet,
                configuration=configuration,
                accepted=accepted,
                layer=layer,
            )
            summary_csv = _read_csv(
                packet, "stellar_status_summary.csv", layer
            )
            summary_json = _load_json(
                packet, "stellar_status_summary.json", layer
            )
            if isinstance(summary_json, dict) and summary_json.get(
                "schema_id"
            ) != STELLAR_STATUS_SCHEMA:
                layer.fail("stellar_status_summary:unsupported_schema")
            if summary_csv is not None and isinstance(summary_json, dict):
                summary_rows = _inventory_records(summary_json.get("rows"))
                if summary_rows is None:
                    layer.fail("stellar_status_summary:json_rows_invalid")
                elif not _csv_json_records_equal(summary_csv, summary_rows):
                    layer.fail("stellar_status_summary:csv_json_rows_mismatch")

            if all(
                (packet / relative).is_file()
                for relative in required_stellar_files
            ):
                _validate_fixed_mass_completeness(
                    packet,
                    configuration,
                    accepted,
                    layer,
                    require_tides=tidal_requested,
                )
                _validate_sequence_completeness(
                    packet,
                    configuration,
                    accepted,
                    layer,
                    require_tides=tidal_requested,
                    require_configured_count=True,
                )
                _validate_stellar_status_reporting(
                    packet, configuration, metadata, layer
                )
                _validate_response_population_reporting(
                    packet, configuration, metadata, layer
                )
            _validate_final_lifecycle(
                packet, configuration, accepted, layer
            )
        else:
            _validate_final_lifecycle(
                packet, configuration, accepted, layer
            )
        if extended_claimed:
            for relative in EXTENDED_CORE_REQUIRED_FILES:
                if not (packet / relative).is_file():
                    layer.fail(f"missing_claimed_extended_output:{relative}")

        for relative, direct_expected in (
            ("thermodynamic_profiles.csv", True),
            ("thermodynamic_residuals.csv", False),
            ("window_characterization.csv", False),
        ):
            if (
                relative == "thermodynamic_residuals.csv"
                and accepted
                and not (packet / relative).is_file()
            ):
                layer.fail(f"missing_thermodynamic_output:{relative}")
                continue
            rows = _read_csv(packet, relative, layer)
            if rows is None:
                continue
            present = _case_ids(rows)
            missing_accepted = accepted - present
            if missing_accepted:
                layer.fail(
                    f"thermodynamic_output:missing_accepted_cases:{relative}:"
                    f"{sorted(missing_accepted)}"
                )
            if direct_expected and "direct" not in present:
                layer.fail(f"thermodynamic_output:direct_baseline_missing:{relative}")

    identity = _load_json(packet, "identity_report.json", layer)
    if isinstance(identity, dict) and identity.get("status") != "pass":
        layer.fail(f"identity_not_pass:{identity.get('status')}")

    reproduction = _load_json(packet, "reproduction.json", layer)
    if isinstance(reproduction, dict):
        if "packet_interpreter_path" in reproduction:
            layer.fail("reproduction:absolute_interpreter_path_must_not_be_recorded")
        command = str(reproduction.get("command", "")).lstrip()
        if command.startswith("py "):
            layer.warn("reproduction:ambient_py_interpreter_is_ambiguous")

    environment = _load_json(packet, "environment.json", layer)
    if isinstance(environment, dict):
        forbidden_environment_fields = {
            "interpreter_path",
            "conda_prefix",
            "conda_executable",
        }
        leaked = sorted(forbidden_environment_fields.intersection(environment))
        if leaked:
            layer.fail(
                "environment:absolute_machine_paths_must_not_be_recorded:"
                + ",".join(leaked)
            )

    status = "complete" if not layer.failures else "partial"
    return {
        "status": status,
        "failures": layer.failures,
        "warnings": layer.warnings,
        "checks": layer.checks,
    }

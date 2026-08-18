"""Passive normalization of saved packet evidence for result summaries."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eos_generation._internal.packet_integrity import _strict_json_payload


_COMPLETE_STELLAR_CONVERGENCE_STATUSES = frozenset(
    {
        "complete_all_requested_stages",
        "complete_all_resolved_and_mass_accepted",
    }
)
_COMPLETE_TIDAL_STATUSES = frozenset({"complete_background_and_tidal"})
_COMPLETE_FIXED_MASS_STATUSES = frozenset({"bracketed_and_solved"})
_NUMERIC_TOKEN = re.compile(
    r"(?<![A-Za-z_^])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"(?![A-Za-z_])"
)

_TABLE_CANDIDATES = (
    "case_ledger.csv",
    "raw_gate_profiles.csv",
    "thermodynamic_profiles.csv",
    "thermodynamic_residuals.csv",
    "thermodynamic_convergence.json",
    "window_characterization.csv",
    "a0_identity_table.csv",
    "stellar_sequences.csv",
    "fixed_mass_observables.csv",
    "stellar_status_summary.csv",
    "maximum_mass_screening.csv",
    "screening_results.csv",
    "baryonic_observables.csv",
    "radial_structure_profiles.csv",
    "deformation_support_fractions.csv",
    "outside_support_control.csv",
    "turning_point_sequences.csv",
    "turning_point_derivatives.csv",
    "stellar_response_across_mass.csv",
    "baryonic_response_across_mass.csv",
    "odd_even_response.csv",
    "matched_area_comparison.csv",
    "numerical_error_summary.csv",
)


def _optional_json(packet: Path, name: str) -> Any | None:
    path = packet / name
    if not path.is_file():
        return None
    return _strict_json_payload(path)


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_or_empty(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _read_csv_rows(packet: Path, name: str) -> list[dict[str, str]]:
    path = packet / name
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"saved CSV has no header: {name}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"saved CSV has duplicate columns: {name}")
        return [dict(row) for row in reader]


def _text(value: Any, default: str = "unavailable") -> str:
    if value is None:
        return default
    result = str(value).strip()
    return result if result else default


def _bool_request(
    configuration: Mapping[str, Any], explicit_key: str
) -> bool:
    explicit = configuration.get(explicit_key)
    if explicit is None:
        return bool(configuration.get("stellar_enabled", False))
    return bool(explicit)


def _packet_outcome(accepted: int, rejected: int) -> str:
    if accepted and rejected:
        return "mixed"
    if rejected:
        return "rejected"
    if accepted:
        return "accepted"
    return "unavailable"


def _case_report(
    raw_gate: Mapping[str, Any], case_id: str
) -> Mapping[str, Any]:
    cases = _mapping_or_empty(raw_gate.get("cases"))
    if case_id in cases and isinstance(cases[case_id], Mapping):
        return cases[case_id]
    physical_cases = _mapping_or_empty(raw_gate.get("physical_cases"))
    aliases = _mapping_or_empty(raw_gate.get("declared_case_aliases"))
    physical_id = _text(aliases.get(case_id), case_id)
    report = physical_cases.get(physical_id)
    return report if isinstance(report, Mapping) else {}


def _canonical_reason(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _rejection_reason(
    row: Mapping[str, Any], gate_report: Mapping[str, Any]
) -> str:
    saved = _canonical_reason(row.get("rejection_reason"))
    if saved is not None:
        return saved
    for key in (
        "first_failure",
        "rejection_reason",
        "full_domain_reconstruction",
        "failure_reason",
        "reason",
    ):
        reason = _canonical_reason(gate_report.get(key))
        if reason is not None:
            return reason
    gate_status = _canonical_reason(gate_report.get("status"))
    return gate_status or "missing_saved_rejection_reason"


def _stable_rejection_category(reason: str) -> str:
    """Reduce exact case evidence to a stable aggregate rejection category."""

    try:
        payload = json.loads(reason)
    except (TypeError, ValueError):
        payload = reason
    if isinstance(payload, Mapping):
        for key in (
            "reason",
            "rejection_reason",
            "failure_category",
            "category",
            "quantity",
            "failed_check",
            "status",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return _NUMERIC_TOKEN.sub("<value>", value.strip())
        failed_checks = payload.get("failed_checks")
        if isinstance(failed_checks, Sequence) and not isinstance(
            failed_checks, (str, bytes)
        ):
            stable = sorted(
                {
                    _NUMERIC_TOKEN.sub("<value>", str(value).strip())
                    for value in failed_checks
                    if str(value).strip()
                }
            )
            if stable:
                return "failed_checks:" + ",".join(stable)
        return "structured_rejection_without_saved_category"
    if isinstance(payload, str) and payload.strip():
        return _NUMERIC_TOKEN.sub("<value>", payload.strip())
    return "missing_saved_rejection_reason"


def _parameter(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _case_rows(
    ledger: Sequence[Mapping[str, Any]], raw_gate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for saved in ledger:
        case_id = _text(saved.get("case_id"))
        outcome = _text(saved.get("status"))
        gate = _case_report(raw_gate, case_id)
        rejected = outcome == "rejected"
        rows.append(
            {
                "case_id": case_id,
                "physical_case_id": (
                    None
                    if saved.get("physical_case_id") in {None, ""}
                    else _text(saved.get("physical_case_id"))
                ),
                "outcome": outcome,
                "rejection_reason": (
                    _rejection_reason(saved, gate) if rejected else None
                ),
                "deformation": {
                    "amplitude": _parameter(saved, "amplitude"),
                    "epsilon0_mev_fm3": _parameter(
                        saved, "epsilon0_mev_fm3"
                    ),
                    "sigma_mev_fm3": _parameter(
                        saved, "sigma_mev_fm3"
                    ),
                    "delta_mev_fm3": _parameter(
                        saved, "delta_mev_fm3"
                    ),
                },
                "raw_gate_status": _text(gate.get("status")),
                "pressure_reconstruction": _text(
                    saved.get("pressure_reconstruction")
                ),
                "stellar_calculation": _text(
                    saved.get("stellar_calculation")
                ),
                "clipping_or_repair": _text(
                    saved.get("clipping_or_repair"), "not_recorded"
                ),
            }
        )
    return rows


def _condition_record(
    reports: Sequence[Mapping[str, Any]], key: str, label: str
) -> dict[str, Any]:
    observed = [report[key] for report in reports if isinstance(report.get(key), bool)]
    passed = sum(value is True for value in observed)
    failed = sum(value is False for value in observed)
    if not observed:
        status = "unavailable"
    elif failed == 0:
        status = "pass_all_evaluated_cases"
    elif passed == 0:
        status = "fail_all_evaluated_cases"
    else:
        status = "mixed"
    return {
        "condition": label,
        "status": status,
        "evaluated_case_count": len(observed),
        "passed_case_count": passed,
        "failed_case_count": failed,
        "evidence": "raw_gate_report.json",
    }


def _raw_gate_reports(raw_gate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    collection = _mapping_or_empty(raw_gate.get("cases"))
    if not collection:
        collection = _mapping_or_empty(raw_gate.get("physical_cases"))
    return [value for value in collection.values() if isinstance(value, Mapping)]


def _status_counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(_text(row.get(key)) for row in rows)
    return {name: counts[name] for name in sorted(counts)}


def _saved_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _maximum_mass_group_evidence(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    resolved_values = [_saved_boolean(row.get("maximum_mass_resolved")) for row in rows]
    threshold_values = [
        _saved_boolean(row.get("passes_maximum_mass_threshold")) for row in rows
    ]
    return {
        "case_stage_count": len(rows),
        "resolved_count": sum(value is True for value in resolved_values),
        "unresolved_count": sum(value is False for value in resolved_values),
        "resolution_unrecorded_count": sum(
            value is None for value in resolved_values
        ),
        "mass_threshold_pass_count": sum(
            value is True for value in threshold_values
        ),
        "mass_threshold_fail_count": sum(
            value is False for value in threshold_values
        ),
        "mass_threshold_unrecorded_count": sum(
            value is None for value in threshold_values
        ),
        "status_counts": _status_counts(rows, "status"),
    }


def _maximum_mass_evidence(
    rows: Sequence[Mapping[str, Any]], *, requested: bool
) -> dict[str, Any]:
    evidence = _maximum_mass_group_evidence(rows)
    total = int(evidence["case_stage_count"])
    resolved = int(evidence["resolved_count"])
    unrecorded = int(evidence["resolution_unrecorded_count"])
    if not requested:
        resolution_status = "not_requested"
    elif total == 0:
        resolution_status = "not_assessed_sampled_peaks_only"
    elif unrecorded:
        resolution_status = "incomplete_resolution_evidence"
    elif resolved == total:
        resolution_status = "all_case_stages_resolved"
    elif resolved:
        resolution_status = "partial_case_stages_resolved"
    else:
        resolution_status = "no_case_stages_resolved"
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_text(row.get("stage")), []).append(row)
    return {
        "resolution_status": resolution_status,
        **evidence,
        "by_stage": {
            stage: _maximum_mass_group_evidence(grouped[stage])
            for stage in sorted(grouped)
        },
        "evidence": (
            "maximum_mass_screening.csv"
            if rows
            else "no saved maximum_mass_screening.csv rows"
        ),
    }


def _contains_saved_uncertainty(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if (
                ("uncertainty" in lowered or "envelope" in lowered)
                and item is not None
            ):
                return True
            if _contains_saved_uncertainty(item):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_saved_uncertainty(item) for item in value)
    return False


def _deformation_declaration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    epsilon0_values = configuration.get("epsilon0_values_mev_fm3")
    if epsilon0_values is None:
        epsilon0_values = [configuration.get("epsilon0_mev_fm3")]
    sigma_values = configuration.get("sigma_values_mev_fm3")
    if sigma_values is None:
        sigma_values = [configuration.get("sigma_mev_fm3")]
    epsilon_match = configuration.get("epsilon_match_mev_fm3")
    return {
        "amplitudes": list(_sequence_or_empty(configuration.get("amplitudes"))),
        "epsilon0_values_mev_fm3": list(_sequence_or_empty(epsilon0_values)),
        "sigma_values_mev_fm3": list(_sequence_or_empty(sigma_values)),
        "deltas_mev_fm3": list(
            _sequence_or_empty(configuration.get("deltas_mev_fm3"))
        ),
        "epsilon_match_mev_fm3": epsilon_match,
        "onset": (
            "standard C1 anchor at n_B=0.16 fm^-3"
            if epsilon_match is None
            else (
                "exploratory full thermodynamic anchor at selected "
                f"epsilon_match={epsilon_match} MeV fm^-3"
            )
        ),
    }


def _plot_rows(plot_inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _sequence_or_empty(plot_inventory.get("figures")):
        if not isinstance(raw, Mapping):
            continue
        rows.append(
            {
                "figure": _text(raw.get("figure")),
                "relative_path": _text(raw.get("relative_path")),
                "status": _text(raw.get("status")),
                "reason": _text(raw.get("reason")),
                "tidal_completeness_status": _text(
                    raw.get("tidal_completeness_status"), "not_applicable"
                ),
            }
        )
    return rows


def _validation_model(
    validation_report: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if validation_report is None:
        return None
    scientific = _mapping_or_empty(
        validation_report.get("scientific_output_completeness")
    )
    status = _text(scientific.get("status"), "unavailable")
    failures = [str(item) for item in _sequence_or_empty(scientific.get("failures"))]
    warnings = [str(item) for item in _sequence_or_empty(scientific.get("warnings"))]
    return {
        "result_status": "valid" if status == "complete" else "invalid",
        "scientific_output_completeness": status,
        "failures": failures,
        "warnings": warnings,
    }

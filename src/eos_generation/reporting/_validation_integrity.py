"""Manifest, configuration, provenance, and reproduction validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from eos_generation._internal.artifacts import project_root
from eos_generation._internal.provenance import (
    SOURCE_INVENTORY_ID,
    _source_inventory_id_for_paths,
)
from eos_generation._internal.summary import (
    PACKET_SCHEMA_ID,
    build_summary_model,
    render_summary_markdown,
)
from eos_generation.bsk24.deformation import (
    BSK24_RETAINED_EPSILON_MATCH_MEV_FM3,
)
from eos_generation.reporting._validation_cases import (
    _validate_case_consistency,
    _validate_ledger,
    _validate_plot_inventory,
)
from eos_generation.reporting._validation_io import (
    _Layer,
    _load_json,
    _read_csv,
    _safe_packet_relative,
    _sha256,
)

TRIAL_PLAN_SCHEMA = "eos_generation_trial_plan_v1"
PLOT_PROVENANCE_SCHEMA = "eos_generation_plot_generation_provenance_v1"

CORE_REQUIRED_FILES = (
    "complete_configuration.json",
    "trial_plan.json",
    "case_plan.csv",
    "case_ledger.csv",
    "accepted_rejected_cases.json",
    "raw_gate_report.json",
    "raw_gate_profiles.csv",
    "thermodynamic_profiles.csv",
    "window_characterization.csv",
    "thermodynamic_convergence.json",
    "identity_report.json",
    "a0_identity_table.csv",
    "plot_inventory.csv",
    "plot_inventory.json",
    "metadata.json",
    "source_hashes.json",
    "environment.json",
    "reproduction.json",
    "commands_used.md",
    "methods_and_results.md",
    "plot_generation_provenance.json",
    "manual_file_ledger.json",
    "run_state.json",
    "SHA256SUMS.txt",
)

_MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def _actual_packet_files(packet: Path, layer: _Layer) -> set[str]:
    actual: set[str] = set()
    for path in packet.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(packet).as_posix()
        safe = _safe_packet_relative(
            relative, packet, layer, context="actual_packet_file"
        )
        if safe is not None:
            actual.add(safe[0])
    return actual


def _validate_manifest(packet: Path, actual: set[str], layer: _Layer) -> set[str]:
    manifest = packet / "SHA256SUMS.txt"
    listed: dict[str, str] = {}
    if not manifest.is_file():
        return set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        layer.fail(f"manifest_read:{type(exc).__name__}:{exc}")
        return set()
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            layer.fail(f"manifest_malformed_line:{line_number}")
            continue
        digest, raw_relative = match.groups()
        safe = _safe_packet_relative(
            raw_relative,
            packet,
            layer,
            context=f"manifest_line_{line_number}",
        )
        if safe is None:
            continue
        relative, path = safe
        if relative == "SHA256SUMS.txt":
            layer.fail("manifest_must_not_hash_itself")
            continue
        if relative in listed:
            layer.fail(f"manifest_duplicate:{relative}")
            continue
        listed[relative] = digest
        if not path.is_file():
            layer.fail(f"manifest_missing_file:{relative}")
        elif _sha256(path) != digest:
            layer.fail(f"manifest_hash_mismatch:{relative}")
    expected = actual - {"SHA256SUMS.txt"}
    for relative in sorted(expected - set(listed)):
        layer.fail(f"manifest_coverage_missing:{relative}")
    for relative in sorted(set(listed) - expected):
        layer.fail(f"manifest_lists_nonpacket_file:{relative}")
    layer.checks["manifest_entries"] = len(listed)
    layer.checks["manifest_expected_entries"] = len(expected)
    return set(listed)


def _configuration_hash_evidence(
    configuration: Any,
    trial_plan: Any,
    metadata: Any,
    run_state: Any,
    layer: _Layer,
    configuration_hash_fn: Callable[[Mapping[str, Any]], str],
) -> str | None:
    if not isinstance(configuration, dict):
        if configuration is not None:
            layer.fail("configuration:not_an_object")
        return None
    try:
        calculated = configuration_hash_fn(configuration)
    except Exception as exc:
        layer.fail(f"configuration_hash_calculation:{type(exc).__name__}:{exc}")
        return None
    if not isinstance(calculated, str) or not re.fullmatch(r"[0-9a-f]{64}", calculated):
        layer.fail("configuration_hash_calculation:invalid_digest")
        return None
    evidence = {
        "calculated_from_complete_configuration": calculated,
        "trial_plan": trial_plan.get("configuration_hash")
        if isinstance(trial_plan, dict)
        else None,
        "metadata": metadata.get("configuration_hash")
        if isinstance(metadata, dict)
        else None,
        "run_state": run_state.get("configuration_hash")
        if isinstance(run_state, dict)
        else None,
    }
    for source, digest in evidence.items():
        if digest != calculated:
            layer.fail(f"configuration_hash_mismatch:{source}")
    layer.checks["configuration_hash"] = calculated
    layer.checks["configuration_hash_evidence"] = evidence
    return calculated


def _validate_saved_output_paths(
    packet: Path,
    configuration: Any,
    trial_plan: Any,
    layer: _Layer,
) -> None:
    """Require child destination fields to be portable and packet-exact."""

    repository_root = project_root().resolve(strict=False)
    try:
        expected = packet.resolve(strict=False).relative_to(
            repository_root
        ).as_posix()
    except ValueError:
        layer.fail("saved_output_path:packet_outside_owning_repository")
        return

    def check(value: Any, context: str) -> None:
        if not isinstance(value, str) or not value or "\\" in value:
            layer.fail(f"{context}:malformed_or_nonportable")
            return
        posix = PurePosixPath(value)
        windows = PureWindowsPath(value)
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or bool(windows.drive)
            or any(part in {"", ".", ".."} for part in posix.parts)
            or not posix.parts
            or posix.parts[0] != "runs"
            or posix.as_posix() != value
        ):
            layer.fail(f"{context}:unsafe_repository_path:{value}")
            return
        resolved = repository_root.joinpath(*posix.parts).resolve(strict=False)
        if resolved != packet.resolve(strict=False) or value != expected:
            layer.fail(f"{context}:packet_path_mismatch")

    if isinstance(configuration, dict):
        check(configuration.get("output_path"), "configuration:output_path")
    if isinstance(trial_plan, dict):
        check(trial_plan.get("output_path"), "trial_plan:output_path")
        operational = trial_plan.get("operational_destination")
        if not isinstance(operational, dict):
            layer.fail("trial_plan:operational_destination:not_an_object")
        else:
            check(
                operational.get("output_path"),
                "trial_plan:operational_destination:output_path",
            )


def _validate_anchor_selection(
    configuration: Any,
    metadata: Any,
    layer: _Layer,
) -> None:
    """Bind the current packet to its selected derived anchor."""

    if not isinstance(configuration, dict) or not isinstance(metadata, dict):
        return
    record = metadata.get("anchor_selection")
    if not isinstance(record, dict):
        layer.fail("anchor_selection:missing_for_current_packet")
        return
    configured = configuration.get("epsilon_match_mev_fm3")
    exploratory = configured is not None
    expected = (
        float(configured)
        if exploratory
        else BSK24_RETAINED_EPSILON_MATCH_MEV_FM3
    )
    selected = record.get("selected_epsilon_match_mev_fm3")
    if (
        isinstance(selected, bool)
        or not isinstance(selected, (int, float))
        or not math.isfinite(float(selected))
        or float(selected) != expected
    ):
        layer.fail("anchor_selection:selected_epsilon_mismatch")
    if record.get("exploratory") is not exploratory:
        layer.fail("anchor_selection:mode_mismatch")
    derived = record.get("derived_state")
    if not isinstance(derived, dict):
        layer.fail("anchor_selection:derived_state_missing")
    elif derived.get("energy_density_mev_fm3") != expected:
        layer.fail("anchor_selection:derived_state_energy_mismatch")
    if record.get("window_and_reconstruction_share_this_anchor") is not True:
        layer.fail("anchor_selection:shared_anchor_declaration_missing")
    layer.checks["anchor_selection"] = {
        "exploratory": exploratory,
        "selected_epsilon_match_mev_fm3": expected,
    }


def _validate_reproduction(
    packet: Path,
    reproduction: Any,
    source_hashes: Any,
    child_configuration_hash: str | None,
    layer: _Layer,
) -> None:
    """Validate the concrete fresh-plan, hash-bound reproduction contract."""

    if not isinstance(reproduction, dict):
        if reproduction is not None:
            layer.fail("reproduction:not_an_object")
        return
    required_strings = (
        "schema_id",
        "configuration_file",
        "child_configuration_file",
        "child_configuration_hash",
        "source_inventory_id",
        "configuration_hash",
        "portable_reproduction_working_directory",
        "portable_configuration_file",
        "portable_output_root",
        "portable_configuration_hash",
        "portable_reproduction_plan_file",
        "portable_plan_hash",
        "portable_plan_command",
        "portable_run_command",
        "reproduction_scope",
        "notebook",
    )
    for key in required_strings:
        value = reproduction.get(key)
        if not isinstance(value, str) or not value.strip():
            layer.fail(f"reproduction:missing_or_empty:{key}")
    if any(
        not isinstance(reproduction.get(key), str)
        or not reproduction[key].strip()
        for key in required_strings
    ):
        return
    for obsolete in (
        "command",
        "portable_reproduction_command",
        "portable_output_directory",
        "reopen_command",
    ):
        if obsolete in reproduction:
            layer.fail(f"reproduction:obsolete_field:{obsolete}")

    if reproduction["schema_id"] != "eos_generation_reproduction_v1":
        layer.fail("reproduction:schema_mismatch")
    if reproduction["child_configuration_file"] != "complete_configuration.json":
        layer.fail("reproduction:child_configuration_file_mismatch")
    if (
        child_configuration_hash is None
        or reproduction["child_configuration_hash"] != child_configuration_hash
    ):
        layer.fail("reproduction:child_configuration_hash_mismatch")
    recognized_inventory = (
        _source_inventory_id_for_paths(source_hashes)
        if isinstance(source_hashes, dict)
        else None
    )
    if recognized_inventory != SOURCE_INVENTORY_ID:
        layer.fail("reproduction:source_inventory_unrecognized")
    if reproduction["source_inventory_id"] != SOURCE_INVENTORY_ID:
        layer.fail("reproduction:source_inventory_id_mismatch")

    try:
        repository_root = project_root().resolve(strict=False)
        expected_configuration_file = (
            packet.parent / "experiment_config.json"
        ).relative_to(repository_root).as_posix()
        expected_reproduction_plan_file = (
            packet.parent / "reproduction_plan.json"
        ).relative_to(repository_root).as_posix()
    except (IndexError, ValueError):
        layer.fail("reproduction:parent_document_outside_runs")
        return
    if not expected_configuration_file.startswith("runs/"):
        layer.fail("reproduction:experiment_configuration_outside_runs")

    experiment_configuration = _load_json(
        packet.parent, "experiment_config.json", layer
    )
    parent_record = _load_json(packet.parent, "reproduction_plan.json", layer)
    if not isinstance(experiment_configuration, dict):
        layer.fail("reproduction:experiment_configuration_missing_or_invalid")
        return
    if not isinstance(parent_record, dict):
        layer.fail("reproduction:parent_plan_missing_or_invalid")
        return
    try:
        expected_configuration_hash = hashlib.sha256(
            json.dumps(
                experiment_configuration,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        layer.fail(
            "reproduction:experiment_configuration_not_canonical_json:"
            f"{type(exc).__name__}"
        )
        return
    settings_payload = dict(experiment_configuration)
    settings_payload.pop("$schema", None)
    expected_settings_hash = hashlib.sha256(
        json.dumps(
            settings_payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    expected_output_root = "runs/reproductions"
    plan_hash = parent_record.get("plan_hash")
    plan = parent_record.get("plan")
    if not isinstance(plan_hash, str) or re.fullmatch(r"[0-9a-f]{64}", plan_hash) is None:
        layer.fail("reproduction:parent_plan_hash_invalid")
        return
    expected_plan_command = (
        "bsk24-trial plan "
        f'--config "{expected_configuration_file}" '
        f'--output-root "{expected_output_root}" --json'
    )
    expected_run_command = (
        "bsk24-trial run "
        f'--config "{expected_configuration_file}" '
        f'--output-root "{expected_output_root}" '
        f"--plan-hash {plan_hash} --execute"
    )
    parent_expected = {
        "schema_id": "eos_generation_reproduction_plan_v1",
        "configuration_file": expected_configuration_file,
        "output_root": expected_output_root,
        "plan_hash": plan_hash,
        "plan_command": expected_plan_command,
        "run_command": expected_run_command,
    }
    for key, value in parent_expected.items():
        if parent_record.get(key) != value:
            layer.fail(f"reproduction:parent_plan_{key}_mismatch")
    expected_experiment_path = (
        repository_root
        / expected_output_root
        / f"experiment_{expected_settings_hash[:12]}"
    ).resolve(strict=False)
    if not isinstance(plan, dict):
        layer.fail("reproduction:parent_plan_payload_invalid")
    else:
        if plan.get("schema_id") != "eos_generation_plan_v1":
            layer.fail("reproduction:parent_plan_payload_schema_mismatch")
        if plan.get("plan_hash") != plan_hash:
            layer.fail("reproduction:parent_plan_payload_hash_mismatch")
        if plan.get("settings") != settings_payload:
            layer.fail("reproduction:parent_plan_payload_settings_mismatch")
        if plan.get("settings_hash") != expected_settings_hash:
            layer.fail("reproduction:parent_plan_payload_settings_hash_mismatch")
        resolved_plan_path = _safe_packet_relative(
            plan.get("experiment_path"),
            repository_root,
            layer,
            context="reproduction:parent_plan_payload_destination",
        )
        if (
            resolved_plan_path is not None
            and resolved_plan_path[1].resolve(strict=False) != expected_experiment_path
        ):
            layer.fail("reproduction:parent_plan_payload_destination_mismatch")
        if (
            plan.get("planning_is_passive") is not True
            or plan.get("scientific_solver_calls") != 0
            or plan.get("filesystem_writes") != 0
        ):
            layer.fail("reproduction:parent_plan_not_passive")

    packet_expected = {
        "configuration_file": expected_configuration_file,
        "configuration_hash": expected_configuration_hash,
        "portable_configuration_file": expected_configuration_file,
        "portable_configuration_hash": expected_configuration_hash,
        "portable_output_root": expected_output_root,
        "portable_reproduction_plan_file": expected_reproduction_plan_file,
        "portable_plan_hash": plan_hash,
        "portable_plan_command": expected_plan_command,
        "portable_run_command": expected_run_command,
        "portable_reproduction_working_directory": "repository_root",
        "reproduction_scope": "aggregate_experiment",
        "notebook": "notebooks/bsk24_experiment.ipynb",
    }
    for key, value in packet_expected.items():
        if reproduction.get(key) != value:
            layer.fail(f"reproduction:{key}_mismatch")
    for command_field in ("portable_plan_command", "portable_run_command"):
        command = reproduction[command_field]
        if "--output-dir" in command or "--resume" in command:
            layer.fail(f"reproduction:forbidden_command_option:{command_field}")
    layer.checks["reproduction_scope"] = "aggregate_experiment"
    layer.checks["reproduction_plan_hash"] = plan_hash




def _validate_packet_schema_and_summary(
    packet: Path,
    metadata: Any,
    layer: _Layer,
) -> str | None:
    """Require the one packet schema written by the current runtime."""

    schema = metadata.get("schema_id") if isinstance(metadata, dict) else None
    layer.checks["packet_schema_id"] = schema
    if schema != PACKET_SCHEMA_ID:
        layer.fail(f"packet_schema:unrecognized:{schema!r}")
        return None

    summary_path = packet / "summary.md"
    if not summary_path.is_file():
        layer.fail("packet_schema:missing_required_file:summary.md")
        return schema
    try:
        observed = summary_path.read_text(encoding="utf-8")
    except Exception as exc:
        layer.fail(f"summary_read:{type(exc).__name__}:{exc}")
        return schema
    try:
        expected = render_summary_markdown(build_summary_model(packet))
    except Exception as exc:
        layer.fail(f"summary_canonical_render:{type(exc).__name__}:{exc}")
        return schema
    if observed != expected:
        layer.fail("summary:canonical_content_mismatch")
    else:
        layer.checks["summary_canonical"] = True
    return schema


def _validate_internal(
    packet: Path,
    *,
    configuration_hash_fn: Callable[[Mapping[str, Any]], str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    layer = _Layer()
    for relative in CORE_REQUIRED_FILES:
        if not (packet / relative).is_file():
            layer.fail(f"missing_required_file:{relative}")

    actual = _actual_packet_files(packet, layer)
    _validate_manifest(packet, actual, layer)

    named_json = {
        "complete_configuration.json",
        "trial_plan.json",
        "metadata.json",
        "run_state.json",
        "raw_gate_report.json",
        "accepted_rejected_cases.json",
        "plot_inventory.json",
        "plot_generation_provenance.json",
        "reproduction.json",
        "source_hashes.json",
        "manual_file_ledger.json",
    }
    additional_json = sorted(
        relative
        for relative in actual
        if relative.endswith(".json") and relative not in named_json
    )
    for relative in additional_json:
        _load_json(packet, relative, layer)
    layer.checks["strict_json_artifact_count"] = sum(
        relative.endswith(".json") for relative in actual
    )

    configuration = _load_json(packet, "complete_configuration.json", layer)
    trial_plan = _load_json(packet, "trial_plan.json", layer)
    metadata = _load_json(packet, "metadata.json", layer)
    run_state = _load_json(packet, "run_state.json", layer)
    raw_gate = _load_json(packet, "raw_gate_report.json", layer)
    accepted_rejected = _load_json(packet, "accepted_rejected_cases.json", layer)
    inventory_json = _load_json(packet, "plot_inventory.json", layer)
    plot_provenance = _load_json(
        packet, "plot_generation_provenance.json", layer
    )
    reproduction = _load_json(packet, "reproduction.json", layer)
    source_hashes = _load_json(packet, "source_hashes.json", layer)
    ledger = _load_json(packet, "manual_file_ledger.json", layer)

    configuration_hash = _configuration_hash_evidence(
        configuration,
        trial_plan,
        metadata,
        run_state,
        layer,
        configuration_hash_fn,
    )
    _validate_saved_output_paths(packet, configuration, trial_plan, layer)
    if not isinstance(trial_plan, dict) or trial_plan.get(
        "schema_id"
    ) != TRIAL_PLAN_SCHEMA:
        layer.fail("trial_plan:unsupported_schema")
    if not isinstance(plot_provenance, dict) or plot_provenance.get(
        "schema_id"
    ) != PLOT_PROVENANCE_SCHEMA:
        layer.fail("plot_generation_provenance:unsupported_schema")
    _validate_anchor_selection(
        configuration,
        metadata,
        layer,
    )
    for relative, payload in (("metadata.json", metadata), ("run_state.json", run_state)):
        if isinstance(payload, dict) and payload.get("packet_status") != "complete":
            layer.fail(f"packet_completion:{relative}:not_complete")

    _validate_ledger(packet, actual, ledger, layer)
    case_plan = _read_csv(packet, "case_plan.csv", layer)
    case_ledger = _read_csv(packet, "case_ledger.csv", layer)
    accepted, rejected = _validate_case_consistency(
        packet,
        case_plan=case_plan,
        case_ledger=case_ledger,
        raw_gate=raw_gate,
        accepted_rejected=accepted_rejected,
        metadata=metadata,
        layer=layer,
    )
    inventory_csv = _read_csv(packet, "plot_inventory.csv", layer)
    _validate_plot_inventory(packet, inventory_csv, inventory_json, layer)
    _validate_reproduction(
        packet,
        reproduction,
        source_hashes,
        configuration_hash,
        layer,
    )
    _validate_packet_schema_and_summary(packet, metadata, layer)

    if isinstance(source_hashes, dict):
        for relative, digest in source_hashes.items():
            if (
                not isinstance(relative, str)
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                layer.fail(f"source_hashes:invalid_entry:{relative!r}")
    elif source_hashes is not None:
        layer.fail("source_hashes:not_an_object")

    result = {
        "status": "pass" if not layer.failures else "fail",
        "failures": layer.failures,
        "warnings": layer.warnings,
        "checks": layer.checks,
    }
    context = {
        "configuration": configuration,
        "metadata": metadata,
        "run_state": run_state,
        "source_hashes": source_hashes,
        "accepted_case_ids": accepted,
        "rejected_case_ids": rejected,
        "plot_inventory": inventory_csv,
    }
    return result, context


def _validate_source_equivalence(
    packet_source_hashes: Any,
    current_source_hashes: Mapping[str, str] | None,
    *,
    required_source_paths: tuple[str, ...],
) -> dict[str, Any]:
    """Compare one packet with the exact current active-source inventory."""

    if not isinstance(packet_source_hashes, dict):
        return {
            "status": "unavailable",
            "failures": ["packet_source_hashes_unavailable"],
            "warnings": [],
            "matching": [],
            "drifted": {},
            "missing_from_packet": [],
            "missing_from_current": [],
        }
    if current_source_hashes is None:
        return {
            "status": "unavailable",
            "failures": ["current_source_hashes_unavailable"],
            "warnings": [],
            "matching": [],
            "drifted": {},
            "missing_from_packet": [],
            "missing_from_current": [],
        }
    packet_hashes = {str(key): str(value) for key, value in packet_source_hashes.items()}
    current_hashes = {str(key): str(value) for key, value in current_source_hashes.items()}
    packet_keys = set(packet_hashes)
    current_keys = set(current_hashes)
    packet_inventory_id = (
        _source_inventory_id_for_paths(packet_hashes) or "unrecognized"
    )
    current_inventory_id = (
        _source_inventory_id_for_paths(current_hashes)
        or "unrecognized"
    )
    missing_from_packet = sorted(current_keys - packet_keys)
    missing_from_current = sorted(packet_keys - current_keys)
    drifted = {
        relative: {
            "packet_sha256": packet_hashes[relative],
            "current_sha256": current_hashes[relative],
        }
        for relative in sorted(packet_keys & current_keys)
        if packet_hashes[relative] != current_hashes[relative]
    }
    required_missing = sorted(set(required_source_paths) - packet_keys)
    matching = sorted(
        relative
        for relative in packet_keys & current_keys
        if relative not in drifted
    )
    failures: list[str] = []
    if packet_inventory_id != SOURCE_INVENTORY_ID:
        failures.append("packet_source_inventory_unrecognized")
    if current_inventory_id != SOURCE_INVENTORY_ID:
        failures.append("current_source_inventory_unrecognized")
    if drifted:
        failures.append("current_source_drift")
    if missing_from_packet or required_missing:
        failures.append("packet_source_coverage_incomplete")
    if missing_from_current:
        failures.append("current_source_coverage_incomplete")
    return {
        "status": (
            "equivalent"
            if not failures
            else "drift_detected_or_incomplete_coverage"
        ),
        "failures": failures,
        "warnings": [],
        "packet_source_inventory_id": packet_inventory_id,
        "current_source_inventory_id": current_inventory_id,
        "inventory_compatibility_mode": "exact_current_inventory_only",
        "matching": matching,
        "drifted": drifted,
        "missing_from_packet": missing_from_packet,
        "missing_from_current": missing_from_current,
        "required_missing_from_packet": required_missing,
        "packet_entry_count": len(packet_hashes),
        "current_entry_count": len(current_hashes),
    }

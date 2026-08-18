"""Fail-closed, read-only status reporting for saved BSk24 trial packets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from eos_generation._internal.artifacts import (
    ensure_within_runs,
    repository_root_scope,
)
from eos_generation._internal.packet_integrity import (
    _strict_json_payload,
    _verify_packet_manifest_exact,
)


STATUS_SCHEMA_ID = "eos_generation_trial_status_v1"
VALIDATION_SCHEMA_ID = "eos_generation_trial_packet_validation_v1"
_REQUIRED_SOURCE_PATHS = (
    "src/eos_generation/stellar/discontinuities.py",
    "src/eos_generation/stellar/tov.py",
    "src/eos_generation/_internal/sequence_tables.py",
)


def _blocked_validation_report(
    packet: Path,
    *,
    failure: str,
) -> dict[str, Any]:
    """Represent a manifest preflight failure without inspecting packet tables."""

    return {
        "schema_id": VALIDATION_SCHEMA_ID,
        "status": "fail",
        "failures": [f"internal_packet_integrity:{failure}"],
        "warnings": [],
        "packet_path": str(packet),
        "manifest_entries": 0,
        "internal_packet_integrity": {
            "status": "fail",
            "failures": [failure],
            "warnings": [],
            "checks": {},
        },
        "current_source_equivalence": {
            "status": "not_assessed",
            "failures": [],
            "warnings": [],
            "checks": {},
        },
        "scientific_output_completeness": {
            "status": "not_assessed",
            "failures": [],
            "warnings": [],
            "checks": {},
        },
    }


def _source_blocked_validation_report(
    packet: Path,
    *,
    manifest_entries: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    failures = [
        f"current_source_equivalence:{item}"
        for item in source.get("failures", [])
    ]
    if not failures:
        failures = ["current_source_equivalence:not_equivalent"]
    return {
        "schema_id": VALIDATION_SCHEMA_ID,
        "status": "fail",
        "failures": failures,
        "warnings": [
            f"current_source_equivalence:{item}"
            for item in source.get("warnings", [])
        ],
        "packet_path": str(packet),
        "manifest_entries": manifest_entries,
        "internal_packet_integrity": {
            "status": "not_assessed_after_manifest_preflight",
            "failures": [],
            "warnings": [],
            "checks": {
                "manifest_exact": True,
                "manifest_entries": manifest_entries,
            },
        },
        "current_source_equivalence": dict(source),
        "scientific_output_completeness": {
            "status": "not_assessed",
            "failures": [],
            "warnings": [],
            "checks": {},
        },
    }


def _status_payload(
    packet: Path,
    *,
    validation: Mapping[str, Any],
    packet_validity: str,
    summary_source: str | None,
    summary: Mapping[str, Any] | None,
    summary_blocked_reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_id": STATUS_SCHEMA_ID,
        "packet_path": str(packet),
        "packet_validity": packet_validity,
        "validation": dict(validation),
        "summary_source": summary_source,
        "summary": None if summary is None else dict(summary),
    }
    if summary is None:
        payload["summary_blocked_reason"] = (
            summary_blocked_reason or "summary_unavailable"
        )
    return payload


def _preflight_source_equivalence(
    packet: Path,
) -> tuple[dict[str, Any], Mapping[str, str] | None]:
    """Check manifest-trusted source provenance without loading CSV tables."""

    from eos_generation._internal.provenance import _source_hashes
    from eos_generation.reporting.validation import (
        _validate_source_equivalence,
    )

    packet_hashes = _strict_json_payload(packet / "source_hashes.json")
    try:
        current_hashes: Mapping[str, str] | None = _source_hashes()
    except Exception:
        current_hashes = None
    report = _validate_source_equivalence(
        packet_hashes,
        current_hashes,
        required_source_paths=_REQUIRED_SOURCE_PATHS,
    )
    return report, current_hashes


def _default_validator(
    packet: Path,
    *,
    current_source_hashes: Mapping[str, str] | None,
) -> dict[str, Any]:
    from eos_generation._internal.planning import BSk24TrialConfig
    from eos_generation.reporting.validation import (
        validate_bsk24_trial_packet_layers,
    )

    return validate_bsk24_trial_packet_layers(
        packet,
        current_source_hashes=current_source_hashes,
        configuration_hash_fn=lambda payload: BSk24TrialConfig.from_dict(
            payload
        ).deterministic_hash(),
        required_source_paths=_REQUIRED_SOURCE_PATHS,
    )


def _default_summary_builder(
    packet: Path,
    *,
    validation_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from eos_generation._internal.summary import build_summary_model

    return build_summary_model(packet, validation_report=validation_report)


def build_bsk24_trial_status(
    packet_path: str | Path,
    *,
    validate_packet: Callable[[Path], Mapping[str, Any]] | None = None,
    summary_builder: Callable[..., Mapping[str, Any]] | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect one packet without trusting or loading tables prematurely.

    Exact manifest coverage and hashes are checked before the layered packet
    validator runs.  The shared summary model is built only after both
    internal integrity and recognized current-source compatibility pass.
    Scientifically incomplete packets may then be summarized, but are marked
    invalid for status-command exit-code purposes.
    """

    if repository_root is not None:
        with repository_root_scope(repository_root):
            return build_bsk24_trial_status(
                packet_path,
                validate_packet=validate_packet,
                summary_builder=summary_builder,
            )

    packet = ensure_within_runs(packet_path).resolve(strict=False)
    if not packet.is_dir():
        raise FileNotFoundError(f"BSk24 trial packet does not exist: {packet}")

    try:
        manifest_entries = _verify_packet_manifest_exact(packet)
    except (OSError, TypeError, ValueError) as exc:
        failure = f"manifest_preflight:{type(exc).__name__}:{exc}"
        validation = _blocked_validation_report(packet, failure=failure)
        return _status_payload(
            packet,
            validation=validation,
            packet_validity="invalid",
            summary_source=None,
            summary=None,
            summary_blocked_reason=f"internal_packet_integrity:{failure}",
        )

    try:
        source_preflight, current_source_hashes = (
            _preflight_source_equivalence(packet)
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        source_preflight = {
            "status": "unavailable",
            "failures": [
                f"source_preflight:{type(exc).__name__}:{exc}"
            ],
            "warnings": [],
        }
        current_source_hashes = None
    if source_preflight.get("status") != "equivalent":
        validation = _source_blocked_validation_report(
            packet,
            manifest_entries=len(manifest_entries),
            source=source_preflight,
        )
        return _status_payload(
            packet,
            validation=validation,
            packet_validity="invalid",
            summary_source=None,
            summary=None,
            summary_blocked_reason=(
                "current_source_equivalence:not_equivalent"
            ),
        )

    try:
        validation = dict(
            validate_packet(packet)
            if validate_packet is not None
            else _default_validator(
                packet,
                current_source_hashes=current_source_hashes,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        failure = f"layered_validation:{type(exc).__name__}:{exc}"
        blocked = _blocked_validation_report(packet, failure=failure)
        return _status_payload(
            packet,
            validation=blocked,
            packet_validity="invalid",
            summary_source=None,
            summary=None,
            summary_blocked_reason=f"internal_packet_integrity:{failure}",
        )

    internal = validation.get("internal_packet_integrity")
    if not isinstance(internal, Mapping) or internal.get("status") != "pass":
        return _status_payload(
            packet,
            validation=validation,
            packet_validity="invalid",
            summary_source=None,
            summary=None,
            summary_blocked_reason="internal_packet_integrity:not_pass",
        )
    source = validation.get("current_source_equivalence")
    if not isinstance(source, Mapping) or source.get("status") != "equivalent":
        return _status_payload(
            packet,
            validation=validation,
            packet_validity="invalid",
            summary_source=None,
            summary=None,
            summary_blocked_reason="current_source_equivalence:not_equivalent",
        )

    try:
        metadata = _strict_json_payload(packet / "metadata.json")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata.json must contain a JSON object")
        from eos_generation._internal.summary import PACKET_SCHEMA_ID

        packet_schema = metadata.get("schema_id")
        if packet_schema == PACKET_SCHEMA_ID:
            summary_source = "saved_current"
        else:
            raise ValueError(f"unrecognized packet schema {packet_schema!r}")
        builder = summary_builder or _default_summary_builder
        summary = dict(
            builder(packet, validation_report=validation)
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return _status_payload(
            packet,
            validation=validation,
            packet_validity="invalid",
            summary_source=None,
            summary=None,
            summary_blocked_reason=(
                f"summary_model:{type(exc).__name__}:{exc}"
            ),
        )

    scientific = validation.get("scientific_output_completeness")
    complete = (
        validation.get("status") == "pass"
        and isinstance(scientific, Mapping)
        and scientific.get("status") == "complete"
    )
    return _status_payload(
        packet,
        validation=validation,
        packet_validity="valid" if complete else "invalid",
        summary_source=summary_source,
        summary=summary,
    )


def render_bsk24_trial_status_text(status: Mapping[str, Any]) -> str:
    """Render the machine status payload as a concise human report."""

    validity = str(status.get("packet_validity", "invalid"))
    lines = [
        f"BSk24 trial packet: {validity.upper()}",
        f"Packet: {status.get('packet_path', '<unknown>')}",
    ]
    summary = status.get("summary")
    if not isinstance(summary, Mapping):
        lines.append(
            "Summary unavailable: "
            + str(status.get("summary_blocked_reason", "unknown reason"))
        )
    else:
        lines.append(f"Summary source: {status.get('summary_source')}")
        from eos_generation._internal.summary import (
            render_summary_markdown,
        )

        rendered = render_summary_markdown(summary).strip()
        if rendered:
            lines.extend(("", rendered))

    validation = status.get("validation")
    if isinstance(validation, Mapping):
        failures = validation.get("failures")
        if isinstance(failures, list) and failures:
            lines.extend(("", "Validation failures:"))
            lines.extend(f"- {item}" for item in failures)
        warnings = validation.get("warnings")
        if isinstance(warnings, list) and warnings:
            lines.extend(("", "Validation warnings:"))
            lines.extend(f"- {item}" for item in warnings)
    return "\n".join(lines)


__all__ = [
    "STATUS_SCHEMA_ID",
    "build_bsk24_trial_status",
    "render_bsk24_trial_status_text",
]

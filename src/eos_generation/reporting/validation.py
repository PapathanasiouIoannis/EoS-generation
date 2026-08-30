"""Layered, read-only validation for BSk24 trial packets.

The module imports authoritative configuration and provenance helpers from
their private implementation modules. The public experiment API can therefore
delegate packet validation here without reversing the private-to-facade
dependency direction or making its otherwise-passive import load plotting
code.

The three validation layers have different meanings:

``internal_packet_integrity``
    The retained files agree with one another and with the packet manifest.
``current_source_equivalence``
    The retained source checksums agree with the currently checked-out source.
``scientific_output_validity``
    Saved scientific evidence is structurally consistent, finite where
    required, and fail-closed where a result is unavailable.
``scientific_output_availability``
    Requested observables are complete or have explicit scientifically valid
    unavailable outcomes.

Source drift is never reported as packet corruption.  Likewise, a correctly
recorded failed-closed scientific result can make availability partial without
making the packet scientifically invalid or unloadable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from eos_generation._internal.artifacts import ensure_within_runs
from eos_generation.reporting._validation_integrity import (
    _validate_internal,
    _validate_source_equivalence,
)
from eos_generation.reporting._validation_io import (
    _default_configuration_hash,
    _default_current_source_hashes,
)
from eos_generation.reporting._validation_scientific import (
    _validate_scientific_completeness,
)

SCHEMA_ID = "eos_generation_trial_packet_validation_v1"


def validate_bsk24_trial_packet_layers(
    packet_path: str | Path,
    *,
    current_source_hashes: (
        Mapping[str, str] | Callable[[], Mapping[str, str]] | None
    ) = None,
    configuration_hash_fn: Callable[[Mapping[str, Any]], str] | None = None,
    required_source_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Validate a completed packet without modifying it.

    Parameters
    ----------
    packet_path:
        Child packet directory below ``runs/``.
    current_source_hashes:
        Current authority checksums or a callback producing them.  If omitted,
        the authoritative private provenance ``_source_hashes`` helper is used.
    configuration_hash_fn:
        Optional deterministic hash callback for isolated tests.  Production
        calls use ``BSk24TrialConfig.from_dict(...).deterministic_hash()``.
    required_source_paths:
        Authorities that a packet must cover in addition to the current source
        mapping.
    """

    packet = ensure_within_runs(packet_path)
    if not packet.is_dir():
        raise FileNotFoundError(f"BSk24 trial packet does not exist: {packet}")
    config_hash = configuration_hash_fn or _default_configuration_hash
    internal, context = _validate_internal(
        packet,
        configuration_hash_fn=config_hash,
    )
    if current_source_hashes is None:
        try:
            current: Mapping[str, str] | None = _default_current_source_hashes()
        except Exception:
            current = None
    elif callable(current_source_hashes):
        try:
            current = current_source_hashes()
        except Exception:
            current = None
    else:
        current = current_source_hashes
    source = _validate_source_equivalence(
        context["source_hashes"],
        current,
        required_source_paths=required_source_paths,
    )
    scientific = _validate_scientific_completeness(
        packet,
        configuration=context["configuration"],
        metadata=context["metadata"],
        accepted=context["accepted_case_ids"],
    )
    validity = scientific["hard_validity"]
    availability = scientific["availability"]
    valid = (
        internal["status"] == "pass"
        and source["status"] == "equivalent"
        and validity["status"] == "pass"
    )
    failures = [
        *(f"internal_packet_integrity:{item}" for item in internal["failures"]),
        *(f"current_source_equivalence:{item}" for item in source["failures"]),
        *(f"scientific_output_validity:{item}" for item in validity["failures"]),
    ]
    return {
        "schema_id": SCHEMA_ID,
        # Backward compatible for existing callers that require exactly
        # ``status == 'pass'`` before accepting a newly written packet.
        "status": "pass" if valid else "fail",
        "failures": failures,
        "warnings": [
            *(f"internal_packet_integrity:{item}" for item in internal["warnings"]),
            *(f"current_source_equivalence:{item}" for item in source["warnings"]),
            *(f"scientific_output_validity:{item}" for item in validity["warnings"]),
            *(
                f"scientific_output_availability:{item}"
                for item in availability["limitations"]
            ),
            *(
                f"scientific_output_availability:{item}"
                for item in availability["warnings"]
            ),
        ],
        "packet_path": str(packet),
        "manifest_entries": internal["checks"].get("manifest_entries", 0),
        "internal_packet_integrity": internal,
        "current_source_equivalence": source,
        "scientific_output_validity": validity,
        "scientific_output_availability": availability,
        "scientific_output_completeness": scientific,
    }


__all__ = ["validate_bsk24_trial_packet_layers"]

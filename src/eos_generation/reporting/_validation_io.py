"""Safe packet paths and strict JSON/CSV loading for packet validation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from eos_generation._internal.planning import BSk24TrialConfig
from eos_generation._internal.provenance import _source_hashes


class _Layer:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.checks: dict[str, Any] = {}

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_packet_relative(
    raw: Any,
    packet: Path,
    layer: _Layer,
    *,
    context: str,
) -> tuple[str, Path] | None:
    """Resolve one packet-relative path without accepting ambiguous syntax."""

    if not isinstance(raw, str) or not raw:
        layer.fail(f"{context}:empty_or_nonstring_path")
        return None
    if "\\" in raw:
        layer.fail(f"{context}:backslash_path:{raw}")
        return None
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        layer.fail(f"{context}:unsafe_path:{raw}")
        return None
    normalized = posix.as_posix()
    if normalized != raw:
        layer.fail(f"{context}:noncanonical_path:{raw}")
        return None
    target = packet.joinpath(*posix.parts)
    resolved = target.resolve(strict=False)
    if not _is_relative_to(resolved, packet):
        layer.fail(f"{context}:resolved_outside_packet:{raw}")
        return None
    return normalized, target


def _load_json(packet: Path, relative: str, layer: _Layer) -> Any | None:
    path = packet / relative
    if not path.is_file():
        return None

    def reject_nonstandard_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {token!r}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_nonstandard_constant,
        )
    except Exception as exc:
        layer.fail(f"json_parse:{relative}:{type(exc).__name__}:{exc}")
        return None


def _read_csv(packet: Path, relative: str, layer: _Layer) -> list[dict[str, str]] | None:
    path = packet / relative
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                layer.fail(f"csv_header_missing:{relative}")
                return None
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                layer.fail(f"csv_duplicate_columns:{relative}")
            return [dict(row) for row in reader]
    except Exception as exc:
        layer.fail(f"csv_parse:{relative}:{type(exc).__name__}:{exc}")
        return None


def _default_configuration_hash(configuration: Mapping[str, Any]) -> str:
    matter_model = configuration.get("matter_model", "bsk24")
    if matter_model == "cfl":
        # Keep the established BSk24 validation import path unchanged.  CFL
        # planning is imported only for a packet that declares that model.
        from eos_generation.cfl.planning import CFLTrialConfig

        return CFLTrialConfig.from_dict(configuration).deterministic_hash()
    if matter_model != "bsk24":
        raise ValueError(f"unsupported saved matter_model: {matter_model!r}")
    return BSk24TrialConfig.from_dict(configuration).deterministic_hash()


def _default_current_source_hashes() -> Mapping[str, str]:
    return _source_hashes()

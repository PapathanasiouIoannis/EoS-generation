"""Low-level packet-integrity helpers for BSk24 trial packets."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from eos_generation._internal.artifacts import ensure_within_runs
from eos_generation._internal.provenance import _hash_file


def _write_text_atomic(text: str, path: Path) -> None:
    target = ensure_within_runs(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        Path(temporary).replace(target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _refresh_manifest(packet: Path) -> None:
    manifest = packet / "SHA256SUMS.txt"
    lines = []
    for path in sorted(
        item for item in packet.rglob("*") if item.is_file() and item != manifest
    ):
        lines.append(f"{_hash_file(path)}  {path.relative_to(packet).as_posix()}")
    _write_text_atomic("\n".join(lines) + "\n", manifest)


def _verify_packet_manifest_exact(packet: Path) -> dict[str, str]:
    """Verify manifest hashes and exact non-manifest file coverage read-only."""

    packet = ensure_within_runs(packet).resolve()
    manifest = packet / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise ValueError(f"packet manifest is missing: {manifest}")
    listed: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
        ):
            raise ValueError(f"malformed manifest line {line_number}")
        digest, raw_relative = parts
        posix = PurePosixPath(raw_relative)
        windows = PureWindowsPath(raw_relative)
        if (
            "\\" in raw_relative
            or posix.is_absolute()
            or windows.is_absolute()
            or bool(windows.drive)
            or any(part in {"", ".", ".."} for part in posix.parts)
            or posix.as_posix() != raw_relative
            or raw_relative in listed
        ):
            raise ValueError(
                f"unsafe or duplicate manifest path on line {line_number}"
            )
        path = packet.joinpath(*posix.parts).resolve(strict=False)
        try:
            path.relative_to(packet)
        except ValueError as exc:
            raise ValueError(
                f"manifest path resolves outside packet on line {line_number}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"manifest file is missing: {raw_relative}")
        actual_digest = _hash_file(path)
        if actual_digest != digest:
            raise ValueError(f"manifest hash mismatch: {raw_relative}")
        listed[raw_relative] = digest
    actual = {
        path.relative_to(packet).as_posix()
        for path in packet.rglob("*")
        if path.is_file() and path != manifest
    }
    if set(listed) != actual:
        missing = sorted(actual - set(listed))
        extra = sorted(set(listed) - actual)
        raise ValueError(
            f"manifest coverage mismatch: missing={missing}, extra={extra}"
        )
    return listed


def _strict_json_payload(path: Path) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {token!r}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject,
    )


def _strict_json_artifact_count(packet: Path) -> int:
    count = 0
    for path in sorted(packet.rglob("*.json")):
        _strict_json_payload(path)
        count += 1
    return count


__all__ = [
    "_refresh_manifest",
    "_strict_json_artifact_count",
    "_strict_json_payload",
    "_verify_packet_manifest_exact",
    "_write_text_atomic",
]

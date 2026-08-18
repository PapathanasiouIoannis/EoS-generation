"""Exact aggregate-manifest writing and validation for experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from ._experiment_io import _sha256_file


AGGREGATE_MANIFEST = "SHA256SUMS.txt"
_AGGREGATE_DOCUMENTS = (
    "experiment.json",
    "experiment_config.json",
    "reproduction_plan.json",
    "reviewed_plan.json",
)


def _aggregate_manifest_files(
    experiment_path: Path, child_names: Sequence[str]
) -> dict[str, Path]:
    files = {name: experiment_path / name for name in _AGGREGATE_DOCUMENTS}
    for child in child_names:
        files[f"{child}/{AGGREGATE_MANIFEST}"] = (
            experiment_path / child / AGGREGATE_MANIFEST
        )
    return files


def _write_aggregate_manifest(
    experiment_path: Path, child_names: Sequence[str]
) -> None:
    files = _aggregate_manifest_files(experiment_path, child_names)
    missing = [relative for relative, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"aggregate manifest input is missing: {missing[0]}")
    text = "".join(
        f"{_sha256_file(path)}  {relative}\n"
        for relative, path in sorted(files.items())
    )
    destination = experiment_path / AGGREGATE_MANIFEST
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_aggregate_manifest(
    experiment_path: Path, child_names: Sequence[str]
) -> None:
    manifest = experiment_path / AGGREGATE_MANIFEST
    if not manifest.is_file():
        raise ValueError("aggregate SHA256SUMS.txt is unavailable")
    expected_files = _aggregate_manifest_files(experiment_path, child_names)
    observed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.count("  ") != 1:
            raise ValueError("aggregate manifest line is malformed")
        digest, relative = line.split("  ", 1)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in observed
        ):
            raise ValueError("aggregate manifest entry is malformed")
        observed[relative] = digest
    if set(observed) != set(expected_files):
        raise ValueError("aggregate manifest inventory mismatch")
    for relative, path in expected_files.items():
        if not path.is_file() or _sha256_file(path) != observed[relative]:
            raise ValueError(f"aggregate manifest hash mismatch: {relative}")

    allowed_files = set(_AGGREGATE_DOCUMENTS) | {AGGREGATE_MANIFEST}
    allowed_directories = set(child_names)
    for entry in experiment_path.iterdir():
        if entry.is_symlink():
            raise ValueError(f"aggregate experiment contains a symlink: {entry.name}")
        if entry.is_file() and entry.name not in allowed_files:
            raise ValueError(f"unexpected aggregate file: {entry.name}")
        if entry.is_dir() and entry.name not in allowed_directories:
            raise ValueError(f"unexpected aggregate directory: {entry.name}")

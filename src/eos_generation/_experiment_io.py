"""Strict serialization, paths, and runtime identity for experiments."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


_PLAN_DEPENDENCIES = ("numpy", "scipy", "pandas", "matplotlib", "numba")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _portable_path(value: str | Path) -> str:
    """Represent an in-repository path without binding it to one checkout."""

    from ._internal.artifacts import project_root

    root = project_root().resolve(strict=False)
    raw = Path(value).expanduser()
    if not raw.is_absolute() and raw.parts and raw.parts[0] == "runs":
        resolved = (root / raw).resolve(strict=False)
    else:
        resolved = raw.resolve(strict=False)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_portable_path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("saved portable path is malformed")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("saved portable path is not a safe repository-relative path")
    resolved = root.joinpath(*relative.parts).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError("saved portable path escapes the repository") from exc
    return resolved


def _owning_repository_root(
    experiment_path: Path,
    saved_experiment_path: Any,
) -> Path:
    """Rebase a saved portable path onto its supplied absolute packet path.

    The complete saved path is used, rather than searching for a directory
    named ``runs``.  This remains unambiguous when an output path itself has a
    nested ``runs/`` component.
    """

    if (
        not isinstance(saved_experiment_path, str)
        or not saved_experiment_path
        or "\\" in saved_experiment_path
        or ":" in saved_experiment_path
    ):
        raise ValueError("reviewed plan experiment path is malformed")
    relative = PurePosixPath(saved_experiment_path)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or not relative.parts
        or relative.parts[0] != "runs"
    ):
        raise ValueError(
            "reviewed plan experiment path is not a safe runs-relative path"
        )

    root = experiment_path
    for _ in relative.parts:
        root = root.parent
    root = root.resolve(strict=False)
    if _resolve_portable_path(saved_experiment_path, root) != experiment_path:
        raise ValueError("reviewed plan experiment path mismatch")

    allowed = (root / "runs").resolve(strict=False)
    try:
        experiment_path.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("saved experiment is outside its owning runs directory") from exc
    if experiment_path == allowed:
        raise ValueError("saved experiment may not be the runs directory itself")
    return root


def _active_source_identity() -> tuple[
    str, str, int, tuple[tuple[str, str], ...]
]:
    """Return a deterministic digest of every governed package source file."""

    from ._internal.artifacts import project_root
    from ._internal.provenance import SOURCE_INVENTORY_ID, _source_hashes

    hashes = _source_hashes()
    root = project_root().resolve(strict=False)
    contracts: dict[str, str] = {}
    for relative in ("environment.yml", "pyproject.toml"):
        path = root / relative
        contracts[relative] = _sha256_file(path) if path.is_file() else "unavailable"
    aggregate = {**hashes, **contracts}
    return (
        SOURCE_INVENTORY_ID,
        _hash_payload(aggregate),
        len(aggregate),
        tuple(sorted(contracts.items())),
    )


def _active_runtime_identity() -> tuple[tuple[str, str], ...]:
    """Return the stable execution-environment fields bound into a plan."""

    values: dict[str, str] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in _PLAN_DEPENDENCIES:
        try:
            values[f"dependency:{distribution}"] = importlib_metadata.version(
                distribution
            )
        except importlib_metadata.PackageNotFoundError:
            values[f"dependency:{distribution}"] = "unavailable"
    return tuple(sorted(values.items()))


def _finite_float(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must contain real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite numbers")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must contain positive numbers")
    return 0.0 if result == 0.0 else result


def _number_tuple(
    name: str,
    values: float | Sequence[float],
    *,
    positive: bool = False,
) -> tuple[float, ...]:
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        source: Iterable[Any] = (values,)
    elif isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a number or a sequence of numbers")
    else:
        source = values
    normalized = tuple(
        _finite_float(name, value, positive=positive) for value in source
    )
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate values")
    return normalized


def _strict_json_object(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid experiment configuration {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("an experiment configuration must be one JSON object")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

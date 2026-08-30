"""Private output-path and deterministic serialization helpers.

Every generated packet is contained below the checkout-local ``runs/``
directory.  Planning only resolves paths; directories are created exclusively
by explicit execution and plotting operations.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


_ACTIVE_REPOSITORY_ROOT: ContextVar[Path | None] = ContextVar(
    "eos_generation_active_repository_root", default=None
)


def _find_checkout_root() -> Path:
    starts = (Path.cwd().resolve(strict=False), Path(__file__).resolve())
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
    return Path.cwd().resolve(strict=False)


def project_root() -> Path:
    """Return the active checkout, or the current directory when installed."""
    active = _ACTIVE_REPOSITORY_ROOT.get()
    if active is not None:
        return active
    return _find_checkout_root()


@contextmanager
def repository_root_scope(root: str | Path) -> Iterator[Path]:
    """Temporarily bind saved-packet operations to their owning repository.

    New planning and execution never enter this scope and therefore retain the
    checkout-local ``runs/`` boundary.  Existing-packet loaders derive and
    verify ``root`` from the supplied absolute packet path plus its saved
    portable path before entering the scope.
    """

    resolved = Path(root).expanduser().resolve(strict=False)
    token = _ACTIVE_REPOSITORY_ROOT.set(resolved)
    try:
        yield resolved
    finally:
        _ACTIVE_REPOSITORY_ROOT.reset(token)


def runs_root() -> Path:
    """Return the only generated-output root."""
    return project_root() / "runs"


def ensure_within_runs(path: str | Path) -> Path:
    """Resolve ``path`` and require containment below ``runs/``."""

    allowed = Path(os.path.realpath(runs_root().resolve(strict=False)))
    resolved = Path(os.path.realpath(Path(path).expanduser()))
    try:
        common = Path(os.path.commonpath((str(allowed), str(resolved))))
    except ValueError as exc:
        raise ValueError(f"Generated path is outside runs/: {resolved}") from exc
    if resolved == allowed or common != allowed:
        raise ValueError(f"Generated path is outside runs/: {resolved}")
    return resolved


def resolve_runs_path(path: str | Path) -> Path:
    """Resolve an explicit output path under ``runs/`` without writing."""
    raw = Path(path).expanduser()
    if raw.is_absolute():
        candidate = raw
    elif raw.parts and raw.parts[0] == "runs":
        candidate = project_root() / raw
    else:
        candidate = Path.cwd() / raw
    return ensure_within_runs(candidate)


def write_json_atomic(data: dict[str, Any], path: str | Path) -> Path:
    target = ensure_within_runs(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
        Path(temporary).replace(target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def write_csv_atomic(frame: pd.DataFrame, path: str | Path) -> Path:
    target = ensure_within_runs(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        Path(temporary).replace(target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def _relative_path(path: Path, root: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return resolved.as_posix()


def json_clean(
    value: Any,
    *,
    path_root: str | Path | None = None,
    path_posix: bool = True,
    nonfinite_float: Any = None,
    preserve_infinite: bool = False,
) -> Any:
    """Convert NumPy, path, and non-finite values to strict JSON values."""
    if isinstance(value, Path):
        root = project_root() if path_root is None else Path(path_root)
        result = _relative_path(value, root)
        return result if path_posix else result.replace("/", os.sep)
    if isinstance(value, np.generic):
        return json_clean(
            value.item(),
            path_root=path_root,
            path_posix=path_posix,
            nonfinite_float=nonfinite_float,
            preserve_infinite=preserve_infinite,
        )
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {
            str(key): json_clean(
                item,
                path_root=path_root,
                path_posix=path_posix,
                nonfinite_float=nonfinite_float,
                preserve_infinite=preserve_infinite,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            json_clean(
                item,
                path_root=path_root,
                path_posix=path_posix,
                nonfinite_float=nonfinite_float,
                preserve_infinite=preserve_infinite,
            )
            for item in value
        ]
    if isinstance(value, float) and not math.isfinite(value):
        if preserve_infinite and math.isinf(value):
            return value
        return nonfinite_float
    return value


def canonical_json(value: Any, **kwargs: Any) -> str:
    cleaned = json_clean(value, **kwargs)
    return json.dumps(
        cleaned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


__all__ = [
    "canonical_json",
    "ensure_within_runs",
    "json_clean",
    "project_root",
    "repository_root_scope",
    "resolve_runs_path",
    "runs_root",
    "write_csv_atomic",
    "write_json_atomic",
]

"""Current-package source and execution-environment provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping

from eos_generation._internal.artifacts import project_root


METHODS_RUNTIME_SPECIFICATION = "environment.yml"
SOURCE_INVENTORY_ID = "eos_generation_active_source_inventory_v1"
SEQUENCE_TABLE_SOURCE_PATH = (
    "src/eos_generation/_internal/sequence_tables.py"
)
# There is no historical-packet compatibility layer in the clean package.
LEGACY_SEQUENCE_TABLE_SOURCE_PATH = SEQUENCE_TABLE_SOURCE_PATH

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PATHS = (
    "src/eos_generation/__init__.py",
    "src/eos_generation/experiment.py",
    "src/eos_generation/_experiment_integrity.py",
    "src/eos_generation/_experiment_io.py",
    "src/eos_generation/_experiment_planning.py",
    "src/eos_generation/cli.py",
    "src/eos_generation/notebook.py",
    "src/eos_generation/bsk24/__init__.py",
    "src/eos_generation/bsk24/baseline.py",
    "src/eos_generation/bsk24/_deformation_bounds.py",
    "src/eos_generation/bsk24/_deformation_core.py",
    "src/eos_generation/bsk24/_deformation_diagnostics.py",
    "src/eos_generation/bsk24/_deformation_gate.py",
    "src/eos_generation/bsk24/_reconstruction_diagnostics.py",
    "src/eos_generation/bsk24/_reconstruction_primitives.py",
    "src/eos_generation/bsk24/_reconstruction_profiles.py",
    "src/eos_generation/bsk24/reconstruction.py",
    "src/eos_generation/bsk24/deformation.py",
    "src/eos_generation/bsk24/source_manifest.json",
    "src/eos_generation/stellar/__init__.py",
    "src/eos_generation/stellar/discontinuities.py",
    "src/eos_generation/stellar/_tov_algebra.py",
    "src/eos_generation/stellar/_tov_integration.py",
    "src/eos_generation/stellar/_tov_maximum.py",
    "src/eos_generation/stellar/_tov_sequence.py",
    "src/eos_generation/stellar/_tov_types.py",
    "src/eos_generation/stellar/tov.py",
    "src/eos_generation/stellar/diagnostics.py",
    "src/eos_generation/_internal/__init__.py",
    "src/eos_generation/_internal/config.py",
    "src/eos_generation/_internal/artifacts.py",
    "src/eos_generation/_internal/planning.py",
    "src/eos_generation/_internal/execution.py",
    "src/eos_generation/_internal/runtime.py",
    "src/eos_generation/_internal/lifecycle.py",
    "src/eos_generation/_internal/loading.py",
    "src/eos_generation/_internal/packet_documents.py",
    "src/eos_generation/_internal/packet_integrity.py",
    "src/eos_generation/_internal/provenance.py",
    "src/eos_generation/_internal/saved_tables.py",
    "src/eos_generation/_internal/sequence_tables.py",
    "src/eos_generation/_internal/status.py",
    "src/eos_generation/_internal/stellar.py",
    "src/eos_generation/_internal/_summary_evidence.py",
    "src/eos_generation/_internal/_summary_markdown.py",
    "src/eos_generation/_internal/_summary_model.py",
    "src/eos_generation/_internal/summary.py",
    "src/eos_generation/_internal/thermodynamics.py",
    "src/eos_generation/_internal/diagnostics.py",
    "src/eos_generation/reporting/__init__.py",
    "src/eos_generation/reporting/_validation_cases.py",
    "src/eos_generation/reporting/_validation_integrity.py",
    "src/eos_generation/reporting/_validation_io.py",
    "src/eos_generation/reporting/_validation_scientific.py",
    "src/eos_generation/reporting/_plotting_data.py",
    "src/eos_generation/reporting/_plotting_diagnostics.py",
    "src/eos_generation/reporting/_plotting_stellar.py",
    "src/eos_generation/reporting/_plotting_style.py",
    "src/eos_generation/reporting/_plotting_thermodynamic.py",
    "src/eos_generation/reporting/plot_helpers.py",
    "src/eos_generation/reporting/plot_layout.py",
    "src/eos_generation/reporting/plot_style.py",
    "src/eos_generation/reporting/plotting.py",
    "src/eos_generation/reporting/plot_orchestration.py",
    "src/eos_generation/reporting/validation.py",
)
_PROVENANCE_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "numba",
)


def _portable_conda_environment_name() -> str | None:
    """Return only a simple environment label, never an activated prefix."""

    value = os.environ.get("CONDA_DEFAULT_ENV")
    if not value or value in {".", ".."}:
        return None
    if any(character in value for character in ("/", "\\", ":")):
        return None
    if any(ord(character) < 32 for character in value):
        return None
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _installed_source_path(relative: str) -> Path:
    prefix = "src/eos_generation/"
    if not relative.startswith(prefix):
        raise ValueError(f"source path is outside the package: {relative}")
    return _PACKAGE_ROOT / relative[len(prefix) :]


def _source_hashes_for_inventory(inventory_id: str) -> dict[str, str]:
    if inventory_id != SOURCE_INVENTORY_ID:
        raise ValueError(f"unrecognized source inventory: {inventory_id}")
    return {
        relative: _hash_file(_installed_source_path(relative))
        for relative in _SOURCE_PATHS
    }


def _source_hashes() -> dict[str, str]:
    return _source_hashes_for_inventory(SOURCE_INVENTORY_ID)


def _source_inventory_id_for_paths(
    paths: Iterable[str] | Mapping[str, str],
) -> str | None:
    if tuple(str(path) for path in paths) == _SOURCE_PATHS:
        return SOURCE_INVENTORY_ID
    return None


def _source_inventory_relation(inventory_id: str) -> str | None:
    return "current" if inventory_id == SOURCE_INVENTORY_ID else None


def _source_inventory_variant_for_hashes(
    hashes: Mapping[str, str],
) -> tuple[str, str] | None:
    # Current-only inventories have no alternate historical digest variants.
    return None


def _environment_record() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in _PROVENANCE_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "environment_role": "packet_execution_environment",
        "methods_runtime_specification": METHODS_RUNTIME_SPECIFICATION,
        "conda_environment_name": _portable_conda_environment_name(),
        "git_used": False,
    }


def _portable_reproduction_record(packet: Path) -> dict[str, str]:
    """Describe a concrete, freshly planned aggregate reproduction."""
    root = project_root().resolve(strict=False)
    experiment_config = packet.parent / "experiment_config.json"
    if not experiment_config.is_file():
        raise FileNotFoundError(
            "child packet reproduction requires parent experiment_config.json"
        )
    configuration_file = experiment_config.resolve(strict=False).relative_to(
        root
    ).as_posix()
    payload = json.loads(experiment_config.read_text(encoding="utf-8"))
    configuration_hash = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    settings_payload = dict(payload)
    settings_payload.pop("$schema", None)
    settings_hash = hashlib.sha256(
        json.dumps(
            settings_payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    reproduction_plan_path = packet.parent / "reproduction_plan.json"
    if not reproduction_plan_path.is_file():
        raise FileNotFoundError(
            "child packet reproduction requires parent reproduction_plan.json"
        )
    reproduction_plan = json.loads(
        reproduction_plan_path.read_text(encoding="utf-8")
    )
    if not isinstance(reproduction_plan, dict):
        raise ValueError("reproduction_plan.json must contain one object")
    output_root = "runs/reproductions"
    plan_hash = reproduction_plan.get("plan_hash")
    plan = reproduction_plan.get("plan")
    if not isinstance(plan_hash, str) or len(plan_hash) != 64:
        raise ValueError("reproduction plan_hash must be a SHA-256 string")
    if not isinstance(plan, dict) or plan.get("plan_hash") != plan_hash:
        raise ValueError("reproduction plan payload/hash mismatch")
    expected_experiment_path = (
        root / output_root / f"experiment_{settings_hash[:12]}"
    ).resolve(strict=False)
    expected_plan_command = (
        "bsk24-trial plan "
        f'--config "{configuration_file}" '
        f'--output-root "{output_root}" --json'
    )
    expected_run_command = (
        "bsk24-trial run "
        f'--config "{configuration_file}" '
        f'--output-root "{output_root}" '
        f"--plan-hash {plan_hash} --execute"
    )
    expected = {
        "schema_id": "eos_generation_reproduction_plan_v1",
        "configuration_file": configuration_file,
        "output_root": output_root,
        "plan_hash": plan_hash,
        "plan_command": expected_plan_command,
        "run_command": expected_run_command,
    }
    for key, value in expected.items():
        if reproduction_plan.get(key) != value:
            raise ValueError(f"reproduction plan {key} mismatch")
    if plan.get("settings") != settings_payload:
        raise ValueError("reproduction plan settings mismatch")
    if plan.get("settings_hash") != settings_hash:
        raise ValueError("reproduction plan settings hash mismatch")
    saved_experiment_path = plan.get("experiment_path")
    if not isinstance(saved_experiment_path, str) or "\\" in saved_experiment_path:
        raise ValueError("reproduction plan experiment path is malformed")
    posix_path = PurePosixPath(saved_experiment_path)
    windows_path = PureWindowsPath(saved_experiment_path)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise ValueError("reproduction plan experiment path is unsafe")
    if root.joinpath(*posix_path.parts).resolve(strict=False) != expected_experiment_path:
        raise ValueError("reproduction plan destination mismatch")
    reproduction_plan_file = reproduction_plan_path.resolve(
        strict=False
    ).relative_to(root).as_posix()
    return {
        "portable_reproduction_working_directory": "repository_root",
        "portable_configuration_file": configuration_file,
        "portable_output_root": output_root,
        "portable_configuration_hash": configuration_hash,
        "portable_reproduction_plan_file": reproduction_plan_file,
        "portable_plan_hash": plan_hash,
        "portable_plan_command": expected_plan_command,
        "portable_run_command": expected_run_command,
        "reproduction_scope": "aggregate_experiment",
    }


__all__ = [
    "LEGACY_SEQUENCE_TABLE_SOURCE_PATH",
    "METHODS_RUNTIME_SPECIFICATION",
    "SEQUENCE_TABLE_SOURCE_PATH",
    "SOURCE_INVENTORY_ID",
    "_environment_record",
    "_hash_file",
    "_portable_reproduction_record",
    "_source_hashes",
    "_source_hashes_for_inventory",
    "_source_inventory_id_for_paths",
    "_source_inventory_relation",
    "_source_inventory_variant_for_hashes",
]

"""Passive reconstruction of saved governed trial result handles."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from eos_generation._internal.artifacts import ensure_within_runs, resolve_runs_path
from eos_generation._internal.planning import (
    PLOT_REGISTRY,
    BSk24TrialConfig,
)
from eos_generation._internal.summary import (
    CFL_PACKET_SCHEMA_ID,
    PACKET_SCHEMA_ID,
)
from eos_generation.reporting.plot_orchestration import ALL_FIGURES


def load_trial_packet(
    packet_path: str | Path,
    *,
    result_factory: Callable[[Path, Any, dict[str, Any], pd.DataFrame], Any],
) -> Any:
    """Open a completed BSk24 or CFL packet without scientific execution."""
    packet = ensure_within_runs(packet_path)
    config_path = packet / "complete_configuration.json"
    metadata_path = packet / "metadata.json"
    if not config_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("not a completed governed trial packet")
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    matter_model = str(config_payload.get("matter_model", "bsk24"))
    if matter_model == "cfl":
        from eos_generation.cfl.planning import CFLTrialConfig

        config = CFLTrialConfig.from_dict(config_payload)
        expected_schema = CFL_PACKET_SCHEMA_ID
    elif matter_model == "bsk24":
        config = BSk24TrialConfig.from_dict(config_payload)
        expected_schema = PACKET_SCHEMA_ID
    else:
        raise ValueError(f"unsupported saved matter_model: {matter_model!r}")
    if config.output_path is None:
        raise ValueError("saved child configuration has no output path")
    saved_output = resolve_runs_path(config.output_path)
    if saved_output != packet.resolve(strict=False):
        raise ValueError(
            "saved child configuration output path does not identify its packet"
        )
    config = replace(config, output_path=packet)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_id") != expected_schema:
        raise ValueError(
            "packet schema is not supported by this eos_generation release"
        )
    if matter_model == "cfl" and metadata.get("matter_model") != "cfl":
        raise ValueError("CFL packet metadata is missing its matter-model identity")
    inventory_path = packet / "plot_inventory.csv"
    if inventory_path.is_file():
        inventory = pd.read_csv(inventory_path)
    else:
        rows = []
        for filename in ALL_FIGURES:
            path = packet / "plots" / filename
            rows.append(
                {
                    "figure": filename,
                    "group": next(
                        spec.group
                        for spec in PLOT_REGISTRY
                        if spec.filename == filename
                    ),
                    "status": "generated" if path.is_file() else "skipped",
                    "reason": (
                        "existing saved figure"
                        if path.is_file()
                        else "figure not present in packet"
                    ),
                    "prerequisite": next(
                        spec.prerequisite
                        for spec in PLOT_REGISTRY
                        if spec.filename == filename
                    ),
                    "relative_path": f"plots/{filename}",
                }
            )
        inventory = pd.DataFrame(rows)
    return result_factory(packet, config, metadata, inventory)


def load_bsk24_trial(
    packet_path: str | Path,
    *,
    result_factory: Callable[
        [Path, BSk24TrialConfig, dict[str, Any], pd.DataFrame], Any
    ],
) -> Any:
    """Backward-compatible name for the passive governed packet loader."""

    return load_trial_packet(packet_path, result_factory=result_factory)


__all__ = ["load_bsk24_trial", "load_trial_packet"]

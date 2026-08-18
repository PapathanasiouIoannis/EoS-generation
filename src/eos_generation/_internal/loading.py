"""Passive reconstruction of saved BSk24 trial result handles."""

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
from eos_generation._internal.summary import PACKET_SCHEMA_ID
from eos_generation.reporting.plot_orchestration import ALL_FIGURES


def load_bsk24_trial(
    packet_path: str | Path,
    *,
    result_factory: Callable[
        [Path, BSk24TrialConfig, dict[str, Any], pd.DataFrame], Any
    ],
) -> Any:
    """Open a completed trial packet without rerunning calculations."""
    packet = ensure_within_runs(packet_path)
    config_path = packet / "complete_configuration.json"
    metadata_path = packet / "metadata.json"
    if not config_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("not a completed BSk24 trial packet")
    config = BSk24TrialConfig.from_dict(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    if config.output_path is None:
        raise ValueError("saved child configuration has no output path")
    saved_output = resolve_runs_path(config.output_path)
    if saved_output != packet.resolve(strict=False):
        raise ValueError(
            "saved child configuration output path does not identify its packet"
        )
    config = replace(config, output_path=packet)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_id") != PACKET_SCHEMA_ID:
        raise ValueError(
            "packet schema is not supported by this eos_generation release"
        )
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


__all__ = ["load_bsk24_trial"]

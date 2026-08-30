"""Packet-document writers for governed BSk24 trials."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eos_generation._internal.artifacts import write_json_atomic
from eos_generation._internal.packet_integrity import (
    _write_text_atomic,
)
from eos_generation._internal.planning import (
    PURE_GAUSSIAN_GENERATOR_ID,
    WINDOWED_GAUSSIAN_GENERATOR_ID,
    BSk24TrialConfig,
)


def _write_packet_ledger(packet: Path) -> None:
    files = {
        path.relative_to(packet).as_posix()
        for path in packet.rglob("*")
        if path.is_file()
    }
    files.update({"manual_file_ledger.json", "SHA256SUMS.txt"})
    write_json_atomic(
        {
            "files_created": sorted(files),
            "files_modified": [],
        },
        packet / "manual_file_ledger.json",
    )


def _write_methods(
    packet: Path,
    config: BSk24TrialConfig,
    metadata: Mapping[str, Any],
) -> None:
    anchor_record = metadata.get("anchor_selection", {})
    if not bool(anchor_record.get("exploratory", False)):
        anchor_clause = (
            "This packet uses the analytical BSk24 C4 pressure barotrope, the C1-normalized\n"
            "anchor at n_t=0.16 fm^-3, the approved quintic smootherstep window, the raw\n"
            "local-physics gate before reconstruction or stellar work, strict\n"
            "non-extrapolation, and the unchanged shared TOV/tidal equations."
        )
    else:
        anchor_clause = (
            "This packet uses the analytical BSk24 C4 pressure barotrope, an explicitly\n"
            "exploratory C1/C4-derived homogeneous-core anchor at epsilon_match="
            f"{anchor_record.get('selected_epsilon_match_mev_fm3')} MeV fm^-3, the approved\n"
            "quintic smootherstep window, the raw local-physics gate before reconstruction\n"
            "or stellar work, strict non-extrapolation, and the unchanged shared TOV/tidal\n"
            "equations."
        )
    text = f"""# BSk24 experiment packet

Generator: `{WINDOWED_GAUSSIAN_GENERATOR_ID}`. The unwindowed
`{PURE_GAUSSIAN_GENERATOR_ID}` construction was not invoked.

{anchor_clause}

Accepted proposals: {metadata['accepted_case_count']}. Rejected proposals:
{metadata['rejected_case_count']}. Rejected proposals were not reconstructed
and were not sent to TOV, tidal, baryon, or extended diagnostic calculations.

A=0 identity status: `{metadata['identity_status']}`. Numerical convergence
status: `{metadata['numerical_convergence_status']}`. Highest sampled stellar
masses, where present, remain `sampled_peak_not_Mmax`; no turning-point or
radial-mode stability calculation is implied.

Generated microscopic composition and species chemical potentials are
unavailable; beta equilibrium is unassessed. Effective reconstructed
one-fluid quantities must not be described as microscopic composition.

Configuration hash: `{config.deterministic_hash()}`. Exact-machine provenance
and the fresh-plan, hash-bound public reproduction commands are recorded in
`reproduction.json`.
"""
    _write_text_atomic(text, packet / "methods_and_results.md")


__all__ = ["_write_methods", "_write_packet_ledger"]

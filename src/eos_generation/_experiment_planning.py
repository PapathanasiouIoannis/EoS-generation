"""Governed passive plan expansion for the public experiment workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._experiment_io import _canonical_json, _hash_payload, _portable_path


PLAN_SCHEMA = "eos_generation_plan_v1"


def _plan_child_document(child: Any) -> dict[str, Any]:
    document = json.loads(_canonical_json(child.to_dict()))
    document["output_path"] = _portable_path(document["output_path"])
    operational = document.get("operational_destination")
    if isinstance(operational, dict) and operational.get("output_path") is not None:
        operational["output_path"] = _portable_path(operational["output_path"])
    return document


def _precision_profile(name: str, calculation: str) -> dict[str, Any]:
    from ._internal.planning import BSk24TOVStage, BSk24ThermodynamicStage

    if name == "quick":
        if calculation == "thermodynamics":
            # Exact retained quickstart_thermodynamic profile.
            return {
                "thermodynamic_stages": (
                    BSk24ThermodynamicStage(
                        "smoke", 129, 257, 1.0e-12, 1.0e-12
                    ),
                    BSk24ThermodynamicStage(
                        "smoke_refined", 257, 513, 1.0e-12, 1.0e-12
                    ),
                ),
                "tov_stages": (),
                "raw_gate_lower_points": 257,
                "raw_gate_upper_points": 1025,
                "maximum_mass_initial_points": 17,
            }
        # Exact retained one-cell notebook relaxed stellar profile.
        return {
            "thermodynamic_stages": (
                BSk24ThermodynamicStage("pilot", 257, 513, 1.0e-12, 1.0e-12),
            ),
            "tov_stages": (
                BSk24TOVStage("pilot_background", 17, 1.0e-8, 1.0e-10, 301),
            ),
            "raw_gate_lower_points": 1025,
            "raw_gate_upper_points": 4097,
            "maximum_mass_initial_points": 9,
        }
    if name == "strict":
        return {
            "thermodynamic_stages": (
                BSk24ThermodynamicStage("coarse", 1025, 2049),
                BSk24ThermodynamicStage("standard", 2049, 4097),
                BSk24ThermodynamicStage("refined", 4097, 8193),
            ),
            "tov_stages": (
                BSk24TOVStage("current", 61, 1.0e-8, 1.0e-10, 601),
                BSk24TOVStage("finer_grid", 121, 1.0e-8, 1.0e-10, 601),
                BSk24TOVStage("tighter_ode", 121, 1.0e-10, 1.0e-12, 1201),
            ),
            "raw_gate_lower_points": 4097,
            "raw_gate_upper_points": 16385,
            "maximum_mass_initial_points": 17,
        }
    raise ValueError(f"unknown precision {name!r}")


def _internal_configs(settings: Any, experiment_path: Path) -> tuple[Any, ...]:
    from ._internal.planning import BSk24TrialConfig

    profile = _precision_profile(settings.precision, settings.calculation)
    anchor = None if settings.epsilon_match == "standard" else settings.epsilon_match
    stellar = settings.calculation == "stellar"
    configs: list[Any] = []
    geometry_count = len(settings.center) * len(settings.width) * len(settings.ramp_width)
    geometry_index = 0
    for center in settings.center:
        for width in settings.width:
            for ramp_width in settings.ramp_width:
                geometry_index += 1
                child_name = f"geometry_{geometry_index:03d}"
                configs.append(
                    BSk24TrialConfig(
                        amplitudes=settings.amplitudes,
                        epsilon_match_mev_fm3=anchor,
                        epsilon0_mev_fm3=center,
                        sigma_mev_fm3=width,
                        deltas_mev_fm3=(ramp_width,),
                        fixed_masses_msun=settings.fixed_masses,
                        thermodynamic_stages=profile["thermodynamic_stages"],
                        tov_stages=(profile["tov_stages"] if stellar else ()),
                        raw_gate_lower_points=profile["raw_gate_lower_points"],
                        raw_gate_upper_points=profile["raw_gate_upper_points"],
                        stellar_enabled=stellar,
                        maximum_mass_initial_points=profile["maximum_mass_initial_points"],
                        extended_stellar_diagnostics_enabled=(settings.diagnostics == "on"),
                        extended_stellar_diagnostics_case_policy="endpoints",
                        diagnostic_delta_mev_fm3=ramp_width,
                        requested_plot_groups=("all-applicable",),
                        output_path=experiment_path / child_name,
                        output_packet_name=None,
                        resume_policy="error",
                    )
                )
    if len(configs) != geometry_count:
        raise RuntimeError("geometry expansion count mismatch")
    return tuple(configs)


def _plan_digest(
    settings: Any,
    experiment_path: Path,
    children: Sequence[Any],
    *,
    source_inventory_id: str,
    source_digest: str,
    source_file_count: int,
    source_contracts: Sequence[tuple[str, str]],
    runtime_identity: Sequence[tuple[str, str]],
    runtime_digest: str,
) -> str:
    payload = {
        "schema_id": PLAN_SCHEMA,
        "settings": settings.to_dict(),
        "experiment_path": _portable_path(experiment_path),
        "source_identity": {
            "inventory_id": source_inventory_id,
            "file_count": source_file_count,
            "sha256": source_digest,
            "project_contract_sha256": dict(source_contracts),
        },
        "runtime_identity": {
            "values": dict(runtime_identity),
            "sha256": runtime_digest,
        },
        "children": [_plan_child_document(child) for child in children],
    }
    return _hash_payload(payload)


def _saved_plan_digest(payload: Mapping[str, Any]) -> str:
    required = {
        "schema_id",
        "settings",
        "experiment_path",
        "source_identity",
        "runtime_identity",
        "children",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"reviewed plan is missing required field {missing[0]!r}")
    return _hash_payload({name: payload[name] for name in sorted(required)})

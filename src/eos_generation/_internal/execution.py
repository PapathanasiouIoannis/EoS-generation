"""High-level execution orchestration for governed BSk24 trials."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from eos_generation._internal.artifacts import (
    write_csv_atomic,
    write_json_atomic,
)
from eos_generation.stellar.tov import (
    LAMBDA_FRAMEWORK_CAPABILITY,
    TIDAL_NOT_REQUESTED_STATUS,
)
from eos_generation._internal.diagnostics import (
    _extended_diagnostics,
)
from eos_generation._internal.lifecycle import (
    _case_lifecycle_ledger,
    _completed_stellar_case_ids,
    _write_case_lifecycle,
)
from eos_generation._internal.packet_documents import (
    _write_methods,
    _write_packet_ledger,
)
from eos_generation._internal.packet_integrity import (
    _refresh_manifest,
    _write_text_atomic,
)
from eos_generation._internal.planning import (
    BSk24TrialConfig,
    _json_records,
)
from eos_generation.reporting.plot_orchestration import (
    _actual_plot_inventory,
)
from eos_generation._internal.provenance import (
    METHODS_RUNTIME_SPECIFICATION,
    SOURCE_INVENTORY_ID,
    _environment_record,
    _portable_reproduction_record,
    _source_hashes,
)
from eos_generation._internal.stellar import (
    _stellar_status_summary,
)
from eos_generation._internal.summary import (
    PACKET_SCHEMA_ID,
    write_packet_summary,
)
from eos_generation._internal.thermodynamics import (
    _a0_identity_table,
    _deformations,
)
from eos_generation.bsk24.baseline import (
    MODEL_NAME,
    VALIDATION_SCOPE,
    VALIDATION_STATUS,
)
from eos_generation.bsk24.deformation import (
    PURE_GAUSSIAN_GENERATOR_ID,
    WINDOWED_GAUSSIAN_GENERATOR_ID,
    BSk24WindowedEos,
)


TRIAL_PACKET_SCHEMA = PACKET_SCHEMA_ID


@dataclass(frozen=True)
class RunCallbacks:
    """Facade-owned seams whose runtime patch behavior must remain stable."""

    prepare_trial: Callable[..., Any]
    load_trial: Callable[..., Any]
    generate_plots: Callable[..., pd.DataFrame]
    validate_packet: Callable[..., dict[str, Any]]
    build_consistent_baseline: Callable[..., Any]
    raw_local_physics_gate: Callable[..., Any]
    raw_gate_frame: Callable[..., pd.DataFrame]
    build_windowed_eos: Callable[..., BSk24WindowedEos]
    thermodynamic_profile_frame: Callable[..., pd.DataFrame]
    thermodynamic_residual_frame: Callable[..., pd.DataFrame]
    window_characterization: Callable[..., dict[str, Any]]
    thermodynamic_convergence: Callable[..., dict[str, Any]]
    run_stellar: Callable[..., Any]


def run_bsk24_trial(
    config: BSk24TrialConfig,
    *,
    callbacks: RunCallbacks,
) -> Any:
    """Explicitly execute one governed trial and persist an immutable packet."""
    execution_started = time.perf_counter()
    plan_started = execution_started
    plan = callbacks.prepare_trial(config)
    phase_wall_seconds: dict[str, float] = {
        "passive_plan_and_preflight": time.perf_counter() - plan_started
    }
    phase_started = time.perf_counter()

    def finish_phase(name: str) -> None:
        nonlocal phase_started
        finished = time.perf_counter()
        phase_wall_seconds[name] = finished - phase_started
        phase_started = finished

    packet = plan.output_path
    if packet.exists():
        if config.resume_policy == "resume-completed":
            existing = callbacks.load_trial(packet)
            if (
                existing.config.deterministic_hash()
                != config.deterministic_hash()
            ):
                raise ValueError("resume target configuration does not match")
            if existing.metadata.get("packet_status") != "complete":
                raise RuntimeError(
                    "only completed trial packets can be resumed"
                )
            return existing
        raise FileExistsError(
            f"output packet already exists: {packet}; choose a new path or "
            "use resume_policy='resume-completed'"
        )
    packet.mkdir(parents=True)
    (packet / "plots").mkdir()
    source_hashes_start = _source_hashes()
    write_json_atomic(
        config.to_dict(), packet / "complete_configuration.json"
    )
    write_json_atomic(plan.to_dict(), packet / "trial_plan.json")
    write_csv_atomic(plan.case_table, packet / "case_plan.csv")
    write_json_atomic(
        {
            "packet_status": "running",
            "configuration_hash": config.deterministic_hash(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
        packet / "run_state.json",
    )
    finish_phase("packet_initialization")

    if config.epsilon_match_mev_fm3 is None:
        # Preserve the exact established call shape for standard packets and
        # repository test doubles.
        stages = {
            stage.name: callbacks.build_consistent_baseline(
                stage.grid_settings()
            )
            for stage in config.thermodynamic_stages
        }
    else:
        stages = {
            stage.name: callbacks.build_consistent_baseline(
                stage.grid_settings(),
                anchor_energy_density_mev_fm3=(
                    config.epsilon_match_mev_fm3
                ),
            )
            for stage in config.thermodynamic_stages
        }
    reference_stage = config.thermodynamic_stages[-1].name
    baseline = stages[reference_stage]
    finish_phase("baseline_construction")
    deformations = _deformations(plan)
    finish_phase("case_resolution")
    gate_reports: dict[str, Any] = {}
    raw_frames: list[pd.DataFrame] = []
    accepted_ids: list[str] = []
    rejected_ids: list[str] = []
    for case_id, deformation in deformations.items():
        gate_kwargs: dict[str, Any] = {
            "dense_lower_points": config.raw_gate_lower_points,
            "dense_upper_points": config.raw_gate_upper_points,
        }
        report, epsilon, raw_cs2 = callbacks.raw_local_physics_gate(
            baseline,
            deformation,
            **gate_kwargs,
        )
        gate_reports[case_id] = report
        raw_frames.append(
            callbacks.raw_gate_frame(
                case_id=case_id,
                deformation=deformation,
                baseline=baseline,
                epsilon=np.asarray(epsilon, dtype=float),
                raw_cs2=np.asarray(raw_cs2, dtype=float),
                status=str(report["status"]),
            )
        )
        if report["status"] == "accepted_raw_local_physics_gate":
            accepted_ids.append(case_id)
        else:
            rejected_ids.append(case_id)
    write_json_atomic(
        {
            "schema_id": "eos_generation_raw_gate_v1",
            "executed_before_reconstruction_and_TOV": True,
            "cases": gate_reports,
            "accepted_case_ids": accepted_ids,
            "rejected_case_ids": rejected_ids,
        },
        packet / "raw_gate_report.json",
    )
    write_csv_atomic(
        (
            pd.concat(raw_frames, ignore_index=True)
            if raw_frames
            else pd.DataFrame(
                columns=(
                    "case_id",
                    "amplitude",
                    "epsilon0_mev_fm3",
                    "sigma_mev_fm3",
                    "delta_mev_fm3",
                    "epsilon_mev_fm3",
                    "window",
                    "gaussian",
                    "delta_cs2",
                    "raw_cs2",
                    "gate_status",
                )
            )
        ),
        packet / "raw_gate_profiles.csv",
    )
    finish_phase("raw_local_physics_gate")

    full_domain_accepted_ids = list(accepted_ids)
    full_domain_rejected_ids = list(rejected_ids)

    stage_cases: dict[str, dict[str, BSk24WindowedEos]] = {
        stage_name: {} for stage_name in stages
    }
    for stage_name, stage_baseline in stages.items():
        for case_id in accepted_ids:
            stage_cases[stage_name][case_id] = callbacks.build_windowed_eos(
                stage_baseline,
                deformations[case_id],
                raw_gate_report=gate_reports[case_id],
            )
    # The complete raw proposal is the authoritative acceptance domain.  No
    # rejected case is reconstructed or sent to a stellar solver.
    write_json_atomic(
        {
            "schema_id": "eos_generation_raw_gate_v1",
            "executed_before_reconstruction_and_TOV": True,
            "full_domain_gate_authoritative": True,
            "selected_domain_policy": "full_retained_domain_only",
            "cases": gate_reports,
            "full_domain_accepted_case_ids": full_domain_accepted_ids,
            "full_domain_rejected_case_ids": full_domain_rejected_ids,
            "accepted_case_ids": accepted_ids,
            "rejected_case_ids": rejected_ids,
        },
        packet / "raw_gate_report.json",
    )
    generated = stage_cases[reference_stage]
    declared_gate_reports = gate_reports
    lifecycle_accepted_ids = list(accepted_ids)
    lifecycle_rejected_ids = list(rejected_ids)
    finish_phase("thermodynamic_reconstruction")
    profile = callbacks.thermodynamic_profile_frame(baseline, generated)
    residuals = callbacks.thermodynamic_residual_frame(generated)
    write_csv_atomic(profile, packet / "thermodynamic_profiles.csv")
    if not residuals.empty:
        write_csv_atomic(
            residuals, packet / "thermodynamic_residuals.csv"
        )
    characterization = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "amplitude": deformations[case_id].amplitude,
                "delta_mev_fm3": deformations[case_id].delta_mev_fm3,
                **{
                    key: value
                    for key, value in callbacks.window_characterization(
                        baseline, deformations[case_id]
                    ).items()
                    if key not in {"case_id", "parameters", "quadrature"}
                    and not isinstance(value, (dict, list))
                },
            }
            for case_id in accepted_ids
        ]
    )
    write_csv_atomic(
        characterization, packet / "window_characterization.csv"
    )
    thermo_convergence = callbacks.thermodynamic_convergence(stage_cases)
    write_json_atomic(
        thermo_convergence, packet / "thermodynamic_convergence.json"
    )
    finish_phase("thermodynamic_tables_and_characterization")

    sequences: pd.DataFrame | None = None
    fixed: pd.DataFrame | None = None
    stellar_convergence: dict[str, Any] = {
        "status": "not_requested",
        "maximum_mass_policy": "sampled peaks are not M_max",
    }
    stellar_status_payload: dict[str, Any] = {
        "schema_id": "eos_generation_stellar_status_summary_v1",
        "publication_interpretation_status": "not_requested",
        "rows": [],
        "totals_by_scope": {},
        "fixed_mass_tidal_failures": [],
        "tidal_valid_status_required": LAMBDA_FRAMEWORK_CAPABILITY,
    }
    stars: dict[tuple[str, str, float], Any] = {}
    maximum_mass_frame: pd.DataFrame | None = None
    if config.background_tov_requested:
        sequences, fixed, stellar_convergence, stars = callbacks.run_stellar(
            config=config, baseline=baseline, generated=generated
        )
        maximum_rows = stellar_convergence.get("maximum_mass_rows", [])
        maximum_reports = stellar_convergence.get(
            "maximum_mass_reports", {}
        )
        maximum_mass_frame = pd.DataFrame(maximum_rows)
        write_csv_atomic(
            maximum_mass_frame,
            packet / "maximum_mass_screening.csv",
        )
        write_json_atomic(
            {
                "schema_id": "bsk24_maximum_mass_reports_v1",
                "cases": dict(maximum_reports),
            },
            packet / "maximum_mass_reports.json",
        )
        write_csv_atomic(sequences, packet / "stellar_sequences.csv")
        write_csv_atomic(fixed, packet / "fixed_mass_observables.csv")
        write_json_atomic(
            stellar_convergence, packet / "stellar_convergence.json"
        )
        if fixed is not None and not fixed.empty:
            stellar_status_frame, stellar_status_payload = (
                _stellar_status_summary(sequences, fixed, config)
            )
        else:
            stellar_status_frame = pd.DataFrame()
            stellar_status_payload = {
                "schema_id": "eos_generation_stellar_status_summary_v1",
                "publication_interpretation_status": (
                    "background_only_no_fixed_mass_rows"
                ),
                "rows": [],
                "totals_by_scope": {},
                "fixed_mass_tidal_failures": [],
                "tidal_valid_status_required": LAMBDA_FRAMEWORK_CAPABILITY,
            }
        write_csv_atomic(
            stellar_status_frame, packet / "stellar_status_summary.csv"
        )
        write_json_atomic(
            stellar_status_payload, packet / "stellar_status_summary.json"
        )

    finish_phase("stellar_calculations")

    completed_stellar_ids = _completed_stellar_case_ids(
        sequences,
        fixed,
        config,
        accepted_case_ids=accepted_ids,
    )
    lifecycle = _case_lifecycle_ledger(
        plan,
        accepted_case_ids=lifecycle_accepted_ids,
        gate_reports=declared_gate_reports,
        completed_stellar_case_ids=completed_stellar_ids,
    )
    _write_case_lifecycle(packet, lifecycle)

    identity_report, identity_table = _a0_identity_table(
        baseline=baseline,
        generated=generated,
        config=config,
        sequences=sequences,
        fixed=fixed,
    )
    write_json_atomic(identity_report, packet / "identity_report.json")
    write_csv_atomic(identity_table, packet / "a0_identity_table.csv")

    extended_tables: dict[str, str] = {}
    if (
        config.extended_stellar_diagnostics_enabled
        and sequences is not None
        and fixed is not None
    ):
        extended_tables = _extended_diagnostics(
            packet=packet,
            config=config,
            baseline=baseline,
            generated=generated,
            sequences=sequences,
            fixed=fixed,
            stars=stars,
        )
    finish_phase("lifecycle_identity_and_extended_diagnostics")

    source_hashes_end = _source_hashes()
    if source_hashes_start != source_hashes_end:
        raise RuntimeError("source files changed during trial execution")
    write_json_atomic(source_hashes_end, packet / "source_hashes.json")
    environment = _environment_record()
    write_json_atomic(environment, packet / "environment.json")
    portable_reproduction = _portable_reproduction_record(
        packet,
    )
    reproduction = {
        "schema_id": "eos_generation_reproduction_v1",
        "configuration_file": portable_reproduction[
            "portable_configuration_file"
        ],
        "child_configuration_file": "complete_configuration.json",
        "child_configuration_hash": config.deterministic_hash(),
        "source_inventory_id": SOURCE_INVENTORY_ID,
        "methods_runtime_specification": METHODS_RUNTIME_SPECIFICATION,
        "configuration_hash": portable_reproduction[
            "portable_configuration_hash"
        ],
        "portable_environment_creation_command": (
            f"conda env create --file {METHODS_RUNTIME_SPECIFICATION}"
        ),
        "canonical_environment": (
            "packet execution environment recorded in environment.json"
        ),
        "secondary_compatibility_environment": (
            "reported separately by the bounded cross-environment regression; "
            "not used to define packet provenance"
        ),
        "notebook": "notebooks/bsk24_experiment.ipynb",
        **portable_reproduction,
    }
    write_json_atomic(reproduction, packet / "reproduction.json")
    _write_text_atomic(
        "# Commands used\n\n"
        "From the repository root, first reproduce and review the fresh "
        "destination-bound plan:\n\n"
        f"`{reproduction['portable_plan_command']}`\n\n"
        "Then execute exactly that reviewed plan hash:\n\n"
        f"`{reproduction['portable_run_command']}`\n",
        packet / "commands_used.md",
    )
    inventory = _actual_plot_inventory(
        packet, config, groups=config.requested_plot_groups
    )
    write_csv_atomic(inventory, packet / "plot_inventory.csv")
    write_json_atomic(
        {
            "figures": _json_records(inventory),
            "generated": [],
            "skipped": _json_records(
                inventory.loc[inventory.status == "skipped"]
            ),
        },
        packet / "plot_inventory.json",
    )
    metadata = {
        "schema_id": TRIAL_PACKET_SCHEMA,
        "packet_status": "calculations_complete_plots_pending",
        "generator_id": WINDOWED_GAUSSIAN_GENERATOR_ID,
        "preserved_generator_id": PURE_GAUSSIAN_GENERATOR_ID,
        "source_baseline_identifier": MODEL_NAME,
        "baseline_validation_status": VALIDATION_STATUS,
        "baseline_validation_scope": VALIDATION_SCOPE,
        "configuration_hash": config.deterministic_hash(),
        "anchor_selection": {
            "mode": (
                "exploratory_selected_epsilon_match"
                if config.exploratory_anchor_requested
                else "standard_n_b_0p16_fm3"
            ),
            "exploratory": config.exploratory_anchor_requested,
            "selected_epsilon_match_mev_fm3": (
                config.effective_epsilon_match_mev_fm3
            ),
            "derived_state": baseline.anchor.to_dict(),
            "window_and_reconstruction_share_this_anchor": True,
            "microscopic_composition_status": "unassessed",
        },
        "accepted_case_count": len(accepted_ids),
        "rejected_case_count": len(rejected_ids),
        "accepted_case_ids": accepted_ids,
        "rejected_case_ids": rejected_ids,
        "identity_status": identity_report["status"],
        "numerical_convergence_status": {
            "thermodynamic": thermo_convergence["status"],
            "stellar": stellar_convergence["status"],
        },
        "stellar_status_summary": {
            "publication_interpretation_status": stellar_status_payload[
                "publication_interpretation_status"
            ],
            "totals_by_scope": stellar_status_payload["totals_by_scope"],
            "fixed_mass_tidal_failure_count": len(
                stellar_status_payload["fixed_mass_tidal_failures"]
            ),
            "tidal_valid_status_required": LAMBDA_FRAMEWORK_CAPABILITY,
        },
        "maximum_mass_status": (
            str(stellar_convergence["status"])
            if maximum_mass_frame is not None
            else "not_assessed_sampled_peaks_only"
        ),
        "composition_policy": {
            "microscopic_composition": "unavailable",
            "species_chemical_potentials": "unavailable",
            "beta_equilibrium": "unassessed",
        },
        "extended_tables": extended_tables,
    }
    _write_methods(packet, config, metadata)
    write_json_atomic(metadata, packet / "metadata.json")
    finish_phase("provenance_and_packet_documents")
    callbacks.generate_plots(
        packet,
        groups=config.requested_plot_groups,
        _initial_packet_generation=True,
    )
    finish_phase("saved_table_plot_generation")
    sequence_rows = 0 if sequences is None else int(len(sequences))
    fixed_rows = 0 if fixed is None else int(len(fixed))
    tidal_rows = 0
    if sequences is not None and "tidal_status" in sequences.columns:
        tidal_rows += int(
            sequences["tidal_status"]
            .fillna(TIDAL_NOT_REQUESTED_STATUS)
            .astype(str)
            .ne(TIDAL_NOT_REQUESTED_STATUS)
            .sum()
        )
    if fixed is not None and "tidal_status" in fixed.columns:
        tidal_rows += int(
            fixed["tidal_status"]
            .fillna(TIDAL_NOT_REQUESTED_STATUS)
            .astype(str)
            .ne(TIDAL_NOT_REQUESTED_STATUS)
            .sum()
        )
    parallel_execution = stellar_convergence.get(
        "parallel_execution",
        {
            "mode": "serial",
            "selected_worker_count": 1,
            "worker_process_ids": [os.getpid()],
        },
    )
    runtime_performance = {
        "schema_id": "bsk24_runtime_performance_v1",
        "configuration_hash": config.deterministic_hash(),
        "anchor_selection": {
            "mode": (
                "exploratory_selected_epsilon_match"
                if config.exploratory_anchor_requested
                else "standard_n_b_0p16_fm3"
            ),
            "exploratory": config.exploratory_anchor_requested,
            "selected_epsilon_match_mev_fm3": (
                config.effective_epsilon_match_mev_fm3
            ),
            "derived_state": baseline.anchor.to_dict(),
            "window_and_reconstruction_share_this_anchor": True,
            "microscopic_composition_status": "unassessed",
        },
        "clock": "time.perf_counter",
        "measurement_boundary": (
            "start of passive execution preflight through saved-table plot "
            "generation; excludes final summary, ledgers, manifest hashing, "
            "and final packet validation"
        ),
        "measured_wall_seconds": time.perf_counter() - execution_started,
        "phase_wall_seconds": phase_wall_seconds,
        "work_observed": {
            "thermodynamic_stage_count": len(config.thermodynamic_stages),
            "tov_stage_count": len(config.tov_stages),
            "declared_case_count": int(len(plan.case_table)),
            "raw_gate_physical_case_count": int(len(deformations)),
            "accepted_physical_case_count": int(len(accepted_ids)),
            "rejected_physical_case_count": int(len(rejected_ids)),
            "saved_stellar_sequence_row_count": sequence_rows,
            "saved_fixed_mass_row_count": fixed_rows,
            "saved_attempted_tidal_row_count": tidal_rows,
            "reported_maximum_mass_background_solver_calls": int(
                stellar_convergence.get("background_solver_call_count", 0)
            ),
        },
        "parallel_execution": parallel_execution,
        "scientific_policy": {
            "equations_changed_for_performance": False,
            "grids_changed_for_performance": False,
            "tolerances_changed_for_performance": False,
            "acceptance_predicates_changed_for_performance": False,
        },
    }
    write_json_atomic(
        runtime_performance, packet / "runtime_performance.json"
    )
    final = json.loads(
        (packet / "metadata.json").read_text(encoding="utf-8")
    )
    final["packet_status"] = "complete"
    final["completed_utc"] = datetime.now(timezone.utc).isoformat()
    final["runtime_performance"] = {
        "schema_id": runtime_performance["schema_id"],
        "relative_path": "runtime_performance.json",
        "measured_wall_seconds": runtime_performance[
            "measured_wall_seconds"
        ],
    }
    write_json_atomic(final, packet / "metadata.json")
    write_json_atomic(
        {
            "packet_status": "complete",
            "configuration_hash": config.deterministic_hash(),
            "completed_utc": final["completed_utc"],
        },
        packet / "run_state.json",
    )
    write_packet_summary(packet)
    _write_packet_ledger(packet)
    _refresh_manifest(packet)
    validation = callbacks.validate_packet(packet)
    if (
        validation["internal_packet_integrity"]["status"] != "pass"
        or validation["current_source_equivalence"]["status"]
        != "equivalent"
    ):
        raise RuntimeError(
            f"trial packet validation failed: {validation}"
        )
    return callbacks.load_trial(packet)


__all__ = ["RunCallbacks", "TRIAL_PACKET_SCHEMA", "run_bsk24_trial"]

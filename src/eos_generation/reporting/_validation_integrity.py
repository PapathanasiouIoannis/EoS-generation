"""Manifest, configuration, provenance, and reproduction validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from eos_generation._internal.artifacts import project_root
from eos_generation._internal.provenance import (
    SOURCE_INVENTORY_ID,
    _source_inventory_id_for_paths,
)
from eos_generation._internal.summary import (
    CFL_PACKET_SCHEMA_ID,
    PACKET_SCHEMA_ID,
    build_summary_model,
    render_summary_markdown,
)
from eos_generation.bsk24.deformation import (
    BSK24_RETAINED_EPSILON_MATCH_MEV_FM3,
)
from eos_generation.reporting._validation_cases import (
    _validate_cfl_raw_gate_profiles,
    _validate_case_consistency,
    _validate_ledger,
    _validate_plot_inventory,
)
from eos_generation.reporting._validation_io import (
    _Layer,
    _load_json,
    _read_csv,
    _safe_packet_relative,
    _sha256,
)
from eos_generation.stellar.tov import (
    TIDAL_CORRECTION_VERSION,
    TIDAL_JUMP_FORMULA,
)

TRIAL_PLAN_SCHEMA = "eos_generation_trial_plan_v1"
CFL_TRIAL_PLAN_SCHEMA = "eos_generation_cfl_trial_plan_v1"
PLOT_PROVENANCE_SCHEMA = "eos_generation_plot_generation_provenance_v1"

CORE_REQUIRED_FILES = (
    "complete_configuration.json",
    "trial_plan.json",
    "case_plan.csv",
    "case_ledger.csv",
    "accepted_rejected_cases.json",
    "raw_gate_report.json",
    "raw_gate_profiles.csv",
    "thermodynamic_profiles.csv",
    "window_characterization.csv",
    "thermodynamic_convergence.json",
    "identity_report.json",
    "a0_identity_table.csv",
    "plot_inventory.csv",
    "plot_inventory.json",
    "metadata.json",
    "source_hashes.json",
    "environment.json",
    "reproduction.json",
    "commands_used.md",
    "methods_and_results.md",
    "plot_generation_provenance.json",
    "manual_file_ledger.json",
    "run_state.json",
    "SHA256SUMS.txt",
)

_CURVE_ONLY_OMITTED_CORE_FILES = frozenset(
    {
        "window_characterization.csv",
        "thermodynamic_convergence.json",
    }
)

_MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def _actual_packet_files(packet: Path, layer: _Layer) -> set[str]:
    actual: set[str] = set()
    for path in packet.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(packet).as_posix()
        safe = _safe_packet_relative(
            relative, packet, layer, context="actual_packet_file"
        )
        if safe is not None:
            actual.add(safe[0])
    return actual


def _validate_manifest(packet: Path, actual: set[str], layer: _Layer) -> set[str]:
    manifest = packet / "SHA256SUMS.txt"
    listed: dict[str, str] = {}
    if not manifest.is_file():
        return set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        layer.fail(f"manifest_read:{type(exc).__name__}:{exc}")
        return set()
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            layer.fail(f"manifest_malformed_line:{line_number}")
            continue
        digest, raw_relative = match.groups()
        safe = _safe_packet_relative(
            raw_relative,
            packet,
            layer,
            context=f"manifest_line_{line_number}",
        )
        if safe is None:
            continue
        relative, path = safe
        if relative == "SHA256SUMS.txt":
            layer.fail("manifest_must_not_hash_itself")
            continue
        if relative in listed:
            layer.fail(f"manifest_duplicate:{relative}")
            continue
        listed[relative] = digest
        if not path.is_file():
            layer.fail(f"manifest_missing_file:{relative}")
        elif _sha256(path) != digest:
            layer.fail(f"manifest_hash_mismatch:{relative}")
    expected = actual - {"SHA256SUMS.txt"}
    for relative in sorted(expected - set(listed)):
        layer.fail(f"manifest_coverage_missing:{relative}")
    for relative in sorted(set(listed) - expected):
        layer.fail(f"manifest_lists_nonpacket_file:{relative}")
    layer.checks["manifest_entries"] = len(listed)
    layer.checks["manifest_expected_entries"] = len(expected)
    return set(listed)


def _configuration_hash_evidence(
    configuration: Any,
    trial_plan: Any,
    metadata: Any,
    run_state: Any,
    layer: _Layer,
    configuration_hash_fn: Callable[[Mapping[str, Any]], str],
) -> str | None:
    if not isinstance(configuration, dict):
        if configuration is not None:
            layer.fail("configuration:not_an_object")
        return None
    try:
        calculated = configuration_hash_fn(configuration)
    except Exception as exc:
        layer.fail(f"configuration_hash_calculation:{type(exc).__name__}:{exc}")
        return None
    if not isinstance(calculated, str) or not re.fullmatch(r"[0-9a-f]{64}", calculated):
        layer.fail("configuration_hash_calculation:invalid_digest")
        return None
    evidence = {
        "calculated_from_complete_configuration": calculated,
        "trial_plan": trial_plan.get("configuration_hash")
        if isinstance(trial_plan, dict)
        else None,
        "metadata": metadata.get("configuration_hash")
        if isinstance(metadata, dict)
        else None,
        "run_state": run_state.get("configuration_hash")
        if isinstance(run_state, dict)
        else None,
    }
    for source, digest in evidence.items():
        if digest != calculated:
            layer.fail(f"configuration_hash_mismatch:{source}")
    layer.checks["configuration_hash"] = calculated
    layer.checks["configuration_hash_evidence"] = evidence
    return calculated


def _validate_saved_output_paths(
    packet: Path,
    configuration: Any,
    trial_plan: Any,
    layer: _Layer,
) -> None:
    """Require child destination fields to be portable and packet-exact."""

    repository_root = project_root().resolve(strict=False)
    try:
        expected = packet.resolve(strict=False).relative_to(
            repository_root
        ).as_posix()
    except ValueError:
        layer.fail("saved_output_path:packet_outside_owning_repository")
        return

    def check(value: Any, context: str) -> None:
        if not isinstance(value, str) or not value or "\\" in value:
            layer.fail(f"{context}:malformed_or_nonportable")
            return
        posix = PurePosixPath(value)
        windows = PureWindowsPath(value)
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or bool(windows.drive)
            or any(part in {"", ".", ".."} for part in posix.parts)
            or not posix.parts
            or posix.parts[0] != "runs"
            or posix.as_posix() != value
        ):
            layer.fail(f"{context}:unsafe_repository_path:{value}")
            return
        resolved = repository_root.joinpath(*posix.parts).resolve(strict=False)
        if resolved != packet.resolve(strict=False) or value != expected:
            layer.fail(f"{context}:packet_path_mismatch")

    if isinstance(configuration, dict):
        check(configuration.get("output_path"), "configuration:output_path")
    if isinstance(trial_plan, dict):
        check(trial_plan.get("output_path"), "trial_plan:output_path")
        operational = trial_plan.get("operational_destination")
        if not isinstance(operational, dict):
            layer.fail("trial_plan:operational_destination:not_an_object")
        else:
            check(
                operational.get("output_path"),
                "trial_plan:operational_destination:output_path",
            )


def _validate_bsk24_anchor_selection(
    configuration: Any,
    metadata: Any,
    layer: _Layer,
) -> None:
    """Bind the current packet to its selected derived anchor."""

    if not isinstance(configuration, dict) or not isinstance(metadata, dict):
        return
    record = metadata.get("anchor_selection")
    if not isinstance(record, dict):
        layer.fail("anchor_selection:missing_for_current_packet")
        return
    configured = configuration.get("epsilon_match_mev_fm3")
    exploratory = configured is not None
    expected = (
        float(configured)
        if exploratory
        else BSK24_RETAINED_EPSILON_MATCH_MEV_FM3
    )
    selected = record.get("selected_epsilon_match_mev_fm3")
    if (
        isinstance(selected, bool)
        or not isinstance(selected, (int, float))
        or not math.isfinite(float(selected))
        or float(selected) != expected
    ):
        layer.fail("anchor_selection:selected_epsilon_mismatch")
    if record.get("exploratory") is not exploratory:
        layer.fail("anchor_selection:mode_mismatch")
    derived = record.get("derived_state")
    if not isinstance(derived, dict):
        layer.fail("anchor_selection:derived_state_missing")
    elif derived.get("energy_density_mev_fm3") != expected:
        layer.fail("anchor_selection:derived_state_energy_mismatch")
    if record.get("window_and_reconstruction_share_this_anchor") is not True:
        layer.fail("anchor_selection:shared_anchor_declaration_missing")
    layer.checks["anchor_selection"] = {
        "exploratory": exploratory,
        "selected_epsilon_match_mev_fm3": expected,
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _cfl_profile_sha256(profile: Mapping[str, Any]) -> str | None:
    payload = dict(profile)
    stored = payload.pop("parameter_set_sha256", None)
    if not isinstance(stored, str):
        return None
    try:
        calculated = hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
    except (TypeError, ValueError, UnicodeEncodeError):
        return None
    return calculated if calculated == stored else None


def _validate_cfl_anchor_selection(
    configuration: Mapping[str, Any],
    metadata: Mapping[str, Any],
    layer: _Layer,
) -> None:
    """Bind a CFL packet to its canonical finite-density vacuum surface."""

    if configuration.get("epsilon_match") != "surface":
        layer.fail("cfl_anchor:epsilon_match_not_surface")
    profile = configuration.get("baseline_profile")
    if not isinstance(profile, Mapping):
        layer.fail("cfl_anchor:baseline_profile_missing")
        return

    parameter_hash = profile.get("parameter_set_sha256")
    if (
        not isinstance(parameter_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", parameter_hash) is None
        or _cfl_profile_sha256(profile) != parameter_hash
    ):
        layer.fail("cfl_anchor:baseline_profile_hash_mismatch")
    for field in (
        "parameter_set_id",
        "parameter_set_sha256",
        "formulation_id",
        "formulation_version",
    ):
        configured_field = (
            "baseline_parameter_set_id"
            if field == "parameter_set_id"
            else "baseline_parameter_set_sha256"
            if field == "parameter_set_sha256"
            else field
        )
        if configuration.get(configured_field) != profile.get(field):
            layer.fail(f"cfl_anchor:configuration_profile_mismatch:{field}")

    surface = profile.get("surface")
    if not isinstance(surface, Mapping):
        layer.fail("cfl_anchor:surface_profile_missing")
        return
    required_surface_fields = (
        "energy_density_mev_fm3",
        "pressure_mev_fm3",
        "baryon_density_fm3",
        "quark_chemical_potential_mev",
        "baryon_chemical_potential_mev",
        "common_fermi_momentum_mev",
        "sound_speed_squared",
        "energy_per_baryon_mev",
    )
    values = {
        field: _finite_number(surface.get(field))
        for field in required_surface_fields
    }
    for field, value in values.items():
        if value is None:
            layer.fail(f"cfl_anchor:surface_nonfinite:{field}")
    if any(value is None for value in values.values()):
        return

    epsilon = float(values["energy_density_mev_fm3"])
    pressure = float(values["pressure_mev_fm3"])
    density = float(values["baryon_density_fm3"])
    mu_q = float(values["quark_chemical_potential_mev"])
    mu_b = float(values["baryon_chemical_potential_mev"])
    fermi_momentum = float(values["common_fermi_momentum_mev"])
    sound_speed = float(values["sound_speed_squared"])
    energy_per_baryon = float(values["energy_per_baryon_mev"])
    if epsilon <= 0.0:
        layer.fail("cfl_anchor:surface_energy_not_positive")
    if pressure != 0.0:
        layer.fail("cfl_anchor:surface_pressure_not_exact_zero")
    if density <= 0.0:
        layer.fail("cfl_anchor:surface_baryon_density_not_positive")
    if mu_q <= 0.0 or mu_b <= 0.0 or fermi_momentum <= 0.0:
        layer.fail("cfl_anchor:surface_chemical_state_not_positive")
    if not 0.0 < sound_speed <= 1.0:
        layer.fail("cfl_anchor:surface_sound_speed_not_causal_stable")
    if mu_b != 3.0 * mu_q:
        layer.fail("cfl_anchor:mu_b_not_three_mu_q")
    if energy_per_baryon != mu_b:
        layer.fail("cfl_anchor:surface_energy_per_baryon_mismatch")
    if epsilon != density * mu_b - pressure:
        layer.fail("cfl_anchor:surface_euler_identity_mismatch")
    if configuration.get("epsilon_match_mev_fm3") != epsilon:
        layer.fail("cfl_anchor:configuration_surface_energy_mismatch")
    complete_domain = configuration.get("complete_domain_mev_fm3")
    if (
        not isinstance(complete_domain, list)
        or len(complete_domain) != 2
        or complete_domain[0] != epsilon
        or _finite_number(complete_domain[1]) is None
        or float(complete_domain[1]) <= epsilon
    ):
        layer.fail("cfl_anchor:complete_domain_mismatch")

    frozen = metadata.get("frozen_cfl_parameters")
    if not isinstance(frozen, Mapping):
        layer.fail("cfl_anchor:metadata_frozen_profile_missing")
    elif dict(frozen) != dict(profile):
        layer.fail("cfl_anchor:metadata_frozen_profile_mismatch")
    for field in (
        "baseline_parameter_set_id",
        "baseline_parameter_set_sha256",
        "formulation_id",
        "formulation_version",
    ):
        if metadata.get(field) != configuration.get(field):
            layer.fail(f"cfl_anchor:metadata_provenance_mismatch:{field}")
    for field in (
        "deformation_profile_id",
        "deformation_profile_version",
        "reconstruction_profile_id",
        "reconstruction_schema_version",
        "pressure_primitive_policy",
        "stellar_sequence_policy",
        "stellar_local_refinement_policy",
        "domain_id",
    ):
        if metadata.get(field) != configuration.get(field):
            layer.fail(f"cfl_anchor:metadata_provenance_mismatch:{field}")
    from eos_generation.stellar.discontinuities import (
        BARE_SELF_BOUND_SEQUENCE_POLICY,
        SEED_PRESERVING_LOCAL_REFINEMENT_POLICY,
    )

    if configuration.get("stellar_sequence_policy") != BARE_SELF_BOUND_SEQUENCE_POLICY:
        layer.fail("cfl_anchor:stellar_sequence_policy_mismatch")
    if configuration.get("stellar_local_refinement_policy") != SEED_PRESERVING_LOCAL_REFINEMENT_POLICY:
        layer.fail("cfl_anchor:stellar_local_refinement_policy_mismatch")
    surface_tidal_policy = metadata.get("surface_tidal_policy")
    expected_surface_tidal_policy = {
        "finite_surface_energy_density": True,
        "required_correction": TIDAL_JUMP_FORMULA,
        "correction_version": TIDAL_CORRECTION_VERSION,
        "application_count": "exactly_once_per_successful_tidal_star",
    }
    if not isinstance(surface_tidal_policy, Mapping):
        layer.fail("cfl_anchor:surface_tidal_policy_missing")
    else:
        for field, expected in expected_surface_tidal_policy.items():
            if surface_tidal_policy.get(field) != expected:
                layer.fail(
                    f"cfl_anchor:surface_tidal_policy_mismatch:{field}"
                )
        saved_evidence = surface_tidal_policy.get("saved_evidence")
        if not isinstance(saved_evidence, str) or not saved_evidence.strip():
            layer.fail("cfl_anchor:surface_tidal_saved_evidence_missing")

    record = metadata.get("anchor_selection")
    if not isinstance(record, Mapping):
        layer.fail("anchor_selection:missing_for_current_packet")
        return
    if record.get("mode") != "bare_self_bound_zero_pressure_surface":
        layer.fail("anchor_selection:mode_mismatch")
    if record.get("exploratory") is not False:
        layer.fail("anchor_selection:mode_mismatch")
    if record.get("selected_epsilon_match_mev_fm3") != epsilon:
        layer.fail("anchor_selection:selected_epsilon_mismatch")
    derived = record.get("derived_state")
    if not isinstance(derived, Mapping):
        layer.fail("anchor_selection:derived_state_missing")
    else:
        for field in (
            "energy_density_mev_fm3",
            "pressure_mev_fm3",
            "baryon_density_fm3",
            "quark_chemical_potential_mev",
            "baryon_chemical_potential_mev",
            "sound_speed_squared",
        ):
            if derived.get(field) != surface.get(field):
                layer.fail(f"anchor_selection:derived_state_mismatch:{field}")
    for field in (
        "window_and_reconstruction_share_this_anchor",
        "surface_pressure_preserved_exactly",
        "surface_baryon_density_preserved_exactly",
        "surface_baryon_chemical_potential_preserved_exactly",
    ):
        if record.get(field) is not True:
            layer.fail(f"anchor_selection:missing_exact_declaration:{field}")
    if record.get("surface_exterior") != "vacuum":
        layer.fail("anchor_selection:surface_exterior_not_vacuum")
    if record.get("crust_or_hadronic_envelope") != "absent":
        layer.fail("anchor_selection:crust_or_envelope_not_absent")
    layer.checks["cfl_surface_anchor"] = {
        "energy_density_mev_fm3": epsilon,
        "pressure_mev_fm3": pressure,
        "baryon_density_fm3": density,
        "baryon_chemical_potential_mev": mu_b,
        "baseline_parameter_set_sha256": parameter_hash,
        "formulation_id": profile.get("formulation_id"),
        "formulation_version": profile.get("formulation_version"),
    }


def _validate_anchor_selection(
    configuration: Any,
    metadata: Any,
    layer: _Layer,
) -> None:
    if not isinstance(configuration, dict) or not isinstance(metadata, dict):
        return
    if configuration.get("matter_model", "bsk24") == "cfl":
        _validate_cfl_anchor_selection(configuration, metadata, layer)
        return
    _validate_bsk24_anchor_selection(configuration, metadata, layer)


def _validate_reproduction(
    packet: Path,
    reproduction: Any,
    source_hashes: Any,
    child_configuration_hash: str | None,
    layer: _Layer,
) -> None:
    """Validate the concrete fresh-plan, hash-bound reproduction contract."""

    if not isinstance(reproduction, dict):
        if reproduction is not None:
            layer.fail("reproduction:not_an_object")
        return
    required_strings = (
        "schema_id",
        "configuration_file",
        "child_configuration_file",
        "child_configuration_hash",
        "source_inventory_id",
        "configuration_hash",
        "portable_reproduction_working_directory",
        "portable_configuration_file",
        "portable_output_root",
        "portable_configuration_hash",
        "portable_reproduction_plan_file",
        "portable_plan_hash",
        "portable_plan_command",
        "portable_run_command",
        "reproduction_scope",
        "notebook",
    )
    for key in required_strings:
        value = reproduction.get(key)
        if not isinstance(value, str) or not value.strip():
            layer.fail(f"reproduction:missing_or_empty:{key}")
    if any(
        not isinstance(reproduction.get(key), str)
        or not reproduction[key].strip()
        for key in required_strings
    ):
        return
    for obsolete in (
        "command",
        "portable_reproduction_command",
        "portable_output_directory",
        "reopen_command",
    ):
        if obsolete in reproduction:
            layer.fail(f"reproduction:obsolete_field:{obsolete}")

    if reproduction["schema_id"] != "eos_generation_reproduction_v1":
        layer.fail("reproduction:schema_mismatch")
    if reproduction["child_configuration_file"] != "complete_configuration.json":
        layer.fail("reproduction:child_configuration_file_mismatch")
    if (
        child_configuration_hash is None
        or reproduction["child_configuration_hash"] != child_configuration_hash
    ):
        layer.fail("reproduction:child_configuration_hash_mismatch")
    recognized_inventory = (
        _source_inventory_id_for_paths(source_hashes)
        if isinstance(source_hashes, dict)
        else None
    )
    if recognized_inventory != SOURCE_INVENTORY_ID:
        layer.fail("reproduction:source_inventory_unrecognized")
    if reproduction["source_inventory_id"] != SOURCE_INVENTORY_ID:
        layer.fail("reproduction:source_inventory_id_mismatch")

    try:
        repository_root = project_root().resolve(strict=False)
        expected_configuration_file = (
            packet.parent / "experiment_config.json"
        ).relative_to(repository_root).as_posix()
        expected_reproduction_plan_file = (
            packet.parent / "reproduction_plan.json"
        ).relative_to(repository_root).as_posix()
    except (IndexError, ValueError):
        layer.fail("reproduction:parent_document_outside_runs")
        return
    if not expected_configuration_file.startswith("runs/"):
        layer.fail("reproduction:experiment_configuration_outside_runs")

    experiment_configuration = _load_json(
        packet.parent, "experiment_config.json", layer
    )
    parent_record = _load_json(packet.parent, "reproduction_plan.json", layer)
    if not isinstance(experiment_configuration, dict):
        layer.fail("reproduction:experiment_configuration_missing_or_invalid")
        return
    if not isinstance(parent_record, dict):
        layer.fail("reproduction:parent_plan_missing_or_invalid")
        return
    try:
        expected_configuration_hash = hashlib.sha256(
            json.dumps(
                experiment_configuration,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        layer.fail(
            "reproduction:experiment_configuration_not_canonical_json:"
            f"{type(exc).__name__}"
        )
        return
    settings_payload = dict(experiment_configuration)
    settings_payload.pop("$schema", None)
    expected_settings_hash = hashlib.sha256(
        json.dumps(
            settings_payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    expected_output_root = "runs/reproductions"
    plan_hash = parent_record.get("plan_hash")
    plan = parent_record.get("plan")
    if not isinstance(plan_hash, str) or re.fullmatch(r"[0-9a-f]{64}", plan_hash) is None:
        layer.fail("reproduction:parent_plan_hash_invalid")
        return
    expected_plan_command = (
        "bsk24-trial plan "
        f'--config "{expected_configuration_file}" '
        f'--output-root "{expected_output_root}" --json'
    )
    expected_run_command = (
        "bsk24-trial run "
        f'--config "{expected_configuration_file}" '
        f'--output-root "{expected_output_root}" '
        f"--plan-hash {plan_hash} --execute"
    )
    parent_expected = {
        "schema_id": "eos_generation_reproduction_plan_v1",
        "configuration_file": expected_configuration_file,
        "output_root": expected_output_root,
        "plan_hash": plan_hash,
        "plan_command": expected_plan_command,
        "run_command": expected_run_command,
    }
    for key, value in parent_expected.items():
        if parent_record.get(key) != value:
            layer.fail(f"reproduction:parent_plan_{key}_mismatch")
    expected_experiment_path = (
        repository_root
        / expected_output_root
        / f"experiment_{expected_settings_hash[:12]}"
    ).resolve(strict=False)
    if not isinstance(plan, dict):
        layer.fail("reproduction:parent_plan_payload_invalid")
    else:
        if plan.get("schema_id") != "eos_generation_plan_v1":
            layer.fail("reproduction:parent_plan_payload_schema_mismatch")
        if plan.get("plan_hash") != plan_hash:
            layer.fail("reproduction:parent_plan_payload_hash_mismatch")
        if plan.get("settings") != settings_payload:
            layer.fail("reproduction:parent_plan_payload_settings_mismatch")
        if plan.get("settings_hash") != expected_settings_hash:
            layer.fail("reproduction:parent_plan_payload_settings_hash_mismatch")
        resolved_plan_path = _safe_packet_relative(
            plan.get("experiment_path"),
            repository_root,
            layer,
            context="reproduction:parent_plan_payload_destination",
        )
        if (
            resolved_plan_path is not None
            and resolved_plan_path[1].resolve(strict=False) != expected_experiment_path
        ):
            layer.fail("reproduction:parent_plan_payload_destination_mismatch")
        if (
            plan.get("planning_is_passive") is not True
            or plan.get("scientific_solver_calls") != 0
            or plan.get("filesystem_writes") != 0
        ):
            layer.fail("reproduction:parent_plan_not_passive")

    packet_expected = {
        "configuration_file": expected_configuration_file,
        "configuration_hash": expected_configuration_hash,
        "portable_configuration_file": expected_configuration_file,
        "portable_configuration_hash": expected_configuration_hash,
        "portable_output_root": expected_output_root,
        "portable_reproduction_plan_file": expected_reproduction_plan_file,
        "portable_plan_hash": plan_hash,
        "portable_plan_command": expected_plan_command,
        "portable_run_command": expected_run_command,
        "portable_reproduction_working_directory": "repository_root",
        "reproduction_scope": "aggregate_experiment",
        "notebook": "notebooks/bsk24_experiment.ipynb",
    }
    for key, value in packet_expected.items():
        if reproduction.get(key) != value:
            layer.fail(f"reproduction:{key}_mismatch")
    for command_field in ("portable_plan_command", "portable_run_command"):
        command = reproduction[command_field]
        if "--output-dir" in command or "--resume" in command:
            layer.fail(f"reproduction:forbidden_command_option:{command_field}")
    layer.checks["reproduction_scope"] = "aggregate_experiment"
    layer.checks["reproduction_plan_hash"] = plan_hash




def _validate_packet_schema_and_summary(
    packet: Path,
    metadata: Any,
    layer: _Layer,
    *,
    matter_model: str,
) -> str | None:
    """Require the packet schema paired with the serialized matter model."""

    schema = metadata.get("schema_id") if isinstance(metadata, dict) else None
    layer.checks["packet_schema_id"] = schema
    expected_schema = (
        CFL_PACKET_SCHEMA_ID if matter_model == "cfl" else PACKET_SCHEMA_ID
    )
    if schema != expected_schema:
        layer.fail(f"packet_schema:unrecognized:{schema!r}")
        return None

    summary_path = packet / "summary.md"
    if not summary_path.is_file():
        layer.fail("packet_schema:missing_required_file:summary.md")
        return schema
    try:
        observed = summary_path.read_text(encoding="utf-8")
    except Exception as exc:
        layer.fail(f"summary_read:{type(exc).__name__}:{exc}")
        return schema
    try:
        expected = render_summary_markdown(build_summary_model(packet))
    except Exception as exc:
        layer.fail(f"summary_canonical_render:{type(exc).__name__}:{exc}")
        return schema
    if observed != expected:
        layer.fail("summary:canonical_content_mismatch")
    else:
        layer.checks["summary_canonical"] = True
    return schema


def _csv_boolean(value: Any) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _csv_finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _validate_cfl_case_aliases(
    configuration: Mapping[str, Any],
    case_plan: list[dict[str, str]] | None,
    case_ledger: list[dict[str, str]] | None,
    layer: _Layer,
) -> None:
    """Verify logical-to-physical CFL identities before completeness checks."""

    if case_plan is None or case_ledger is None:
        return
    baseline_id = configuration.get("zero_amplitude_physical_case_id")
    model = str(configuration.get("matter_model", "bsk24"))
    expected_prefix = (
        "cfl_baseline_" if model == "cfl" else "bsk24_baseline_"
    )
    if not isinstance(baseline_id, str) or not baseline_id.startswith(
        expected_prefix
    ):
        layer.fail("cfl_case_aliases:baseline_physical_id_invalid")
        return
    plan_by_id = {
        str(row.get("case_id", "")): row
        for row in case_plan
        if row.get("case_id")
    }
    physical_ids: set[str] = set()
    alias_count = 0
    for row in case_ledger:
        case_id = str(row.get("case_id", ""))
        planned = plan_by_id.get(case_id)
        if planned is None:
            continue
        physical_id = str(row.get("physical_case_id", ""))
        planned_physical_id = str(planned.get("physical_case_id", ""))
        if not physical_id or physical_id != planned_physical_id:
            layer.fail(f"cfl_case_aliases:physical_id_mismatch:{case_id}")
            continue
        physical_ids.add(physical_id)
        declared_alias = _csv_boolean(row.get("is_physical_case_alias"))
        planned_alias = _csv_boolean(planned.get("is_physical_case_alias"))
        expected_alias = physical_id != case_id
        if (
            declared_alias is not expected_alias
            or planned_alias is not expected_alias
        ):
            layer.fail(f"cfl_case_aliases:alias_flag_mismatch:{case_id}")
        if expected_alias:
            alias_count += 1
        amplitude = _csv_finite_number(planned.get("amplitude"))
        if amplitude is None:
            layer.fail(f"cfl_case_aliases:amplitude_invalid:{case_id}")
        elif amplitude == 0.0:
            if physical_id != baseline_id:
                layer.fail(f"cfl_case_aliases:a0_physical_id_mismatch:{case_id}")
        elif physical_id != case_id:
            layer.fail(f"cfl_case_aliases:nonzero_alias_forbidden:{case_id}")
    layer.checks["cfl_case_aliases"] = {
        "logical_case_count": len(case_ledger),
        "physical_case_count": len(physical_ids),
        "alias_count": alias_count,
        "zero_amplitude_physical_case_id": baseline_id,
    }


def _validate_cfl_plan_aliases(
    configuration: Mapping[str, Any],
    trial_plan: Mapping[str, Any],
    case_plan: list[dict[str, str]] | None,
    saved_aliases: list[dict[str, str]] | None,
    layer: _Layer,
) -> None:
    """Validate planned-but-unexecuted logical A=0 alias declarations."""

    if case_plan is None or saved_aliases is None:
        return
    planned_cases = trial_plan.get("case_table")
    aliases = trial_plan.get("logical_alias_table")
    if not isinstance(planned_cases, list):
        layer.fail("cfl_plan_aliases:case_table_missing")
        return
    if not isinstance(aliases, list):
        layer.fail("cfl_plan_aliases:logical_alias_table_missing")
        return
    saved_case_ids = {str(row.get("case_id", "")) for row in case_plan}
    serialized_case_ids = {
        str(row.get("case_id", ""))
        for row in planned_cases
        if isinstance(row, Mapping)
    }
    if len(serialized_case_ids) != len(planned_cases):
        layer.fail("cfl_plan_aliases:serialized_case_identity_invalid")
    if saved_case_ids != serialized_case_ids:
        layer.fail("cfl_plan_aliases:serialized_case_table_mismatch")

    saved_alias_by_id = {
        str(row.get("case_id", "")): row
        for row in saved_aliases
        if row.get("case_id")
    }
    if len(saved_alias_by_id) != len(saved_aliases):
        layer.fail("cfl_plan_aliases:saved_alias_identity_invalid")

    baseline_id = configuration.get("zero_amplitude_physical_case_id")
    alias_ids: set[str] = set()
    for item in aliases:
        if not isinstance(item, Mapping):
            layer.fail("cfl_plan_aliases:alias_not_object")
            continue
        case_id = str(item.get("case_id", ""))
        if not case_id or case_id in alias_ids or case_id in saved_case_ids:
            layer.fail(f"cfl_plan_aliases:alias_identity_invalid:{case_id}")
            continue
        alias_ids.add(case_id)
        amplitude = _finite_number(item.get("amplitude"))
        if amplitude != 0.0:
            layer.fail(f"cfl_plan_aliases:nonzero_alias:{case_id}")
        if item.get("physical_case_id") != baseline_id:
            layer.fail(f"cfl_plan_aliases:physical_id_mismatch:{case_id}")
        if item.get("is_physical_case_alias") is not True:
            layer.fail(f"cfl_plan_aliases:alias_flag_missing:{case_id}")
        if item.get("planned_for_execution") is not False:
            layer.fail(f"cfl_plan_aliases:execution_flag_invalid:{case_id}")
        if item.get("physical_case_owner") is not False:
            layer.fail(f"cfl_plan_aliases:owner_flag_invalid:{case_id}")
        saved = saved_alias_by_id.get(case_id)
        if saved is None:
            layer.fail(f"cfl_plan_aliases:saved_alias_missing:{case_id}")
        else:
            if saved.get("physical_case_id") != item.get("physical_case_id"):
                layer.fail(
                    f"cfl_plan_aliases:saved_physical_id_mismatch:{case_id}"
                )
            for field in (
                "is_physical_case_alias",
                "planned_for_execution",
                "physical_case_owner",
            ):
                if _csv_boolean(saved.get(field)) is not item.get(field):
                    layer.fail(
                        f"cfl_plan_aliases:saved_flag_mismatch:{case_id}:{field}"
                    )
            for field in (
                "amplitude",
                "epsilon0_mev_fm3",
                "sigma_mev_fm3",
                "delta_mev_fm3",
            ):
                if _csv_finite_number(saved.get(field)) != _finite_number(
                    item.get(field)
                ):
                    layer.fail(
                        f"cfl_plan_aliases:saved_geometry_mismatch:{case_id}:{field}"
                    )
    if set(saved_alias_by_id) != alias_ids:
        layer.fail("cfl_plan_aliases:saved_alias_id_mismatch")

    deltas = configuration.get("deltas_mev_fm3")
    owner = configuration.get("zero_amplitude_control_owner")
    if not isinstance(deltas, list) or not isinstance(owner, bool):
        layer.fail("cfl_plan_aliases:configuration_geometry_invalid")
    else:
        expected_alias_count = len(deltas) - (1 if owner else 0)
        if len(alias_ids) != expected_alias_count:
            layer.fail(
                "cfl_plan_aliases:alias_count_mismatch:"
                f"{len(alias_ids)}:{expected_alias_count}"
            )
    layer.checks["cfl_logical_alias_count"] = len(alias_ids)


def _normalize_cfl_raw_gate_identities(
    configuration: Mapping[str, Any],
    raw_gate: Any,
    case_plan: list[dict[str, str]] | None,
    metadata: Any,
    layer: _Layer,
) -> tuple[Any, set[str], set[str]]:
    """Validate physical raw-gate IDs and map them to lifecycle logical IDs."""

    if not isinstance(raw_gate, dict) or case_plan is None:
        return raw_gate, set(), set()
    logical_to_physical = {
        str(row.get("case_id", "")): str(row.get("physical_case_id", ""))
        for row in case_plan
        if row.get("case_id")
    }
    if not logical_to_physical:
        cases = raw_gate.get("cases")
        empty_outcome_fields = (
            "accepted_case_ids",
            "rejected_case_ids",
        )
        optional_empty_outcome_fields = (
            "full_domain_accepted_case_ids",
            "full_domain_rejected_case_ids",
        )
        valid_alias_only_child = bool(
            configuration.get("zero_amplitude_control_owner") is False
            and configuration.get("effective_amplitudes") == []
            and isinstance(cases, dict)
            and not cases
            and all(raw_gate.get(field) == [] for field in empty_outcome_fields)
            and all(
                field not in raw_gate or raw_gate.get(field) == []
                for field in optional_empty_outcome_fields
            )
        )
        if not valid_alias_only_child:
            layer.fail("cfl_raw_gate_identity:logical_physical_mapping_invalid")
            return raw_gate, set(), set()
        if isinstance(metadata, Mapping):
            for outcome in ("accepted", "rejected"):
                if metadata.get(f"{outcome}_physical_case_ids") != []:
                    layer.fail(
                        f"metadata:{outcome}_physical_case_ids_mismatch"
                    )
                if metadata.get(f"{outcome}_physical_case_count") != 0:
                    layer.fail(
                        f"metadata:{outcome}_physical_case_count_mismatch"
                    )
        layer.checks["cfl_raw_gate_physical_case_count"] = 0
        layer.checks["cfl_raw_gate_accepted_physical_case_count"] = 0
        layer.checks["cfl_raw_gate_rejected_physical_case_count"] = 0
        return dict(raw_gate), set(), set()
    if any(
        not physical_id for physical_id in logical_to_physical.values()
    ):
        layer.fail("cfl_raw_gate_identity:logical_physical_mapping_invalid")
        return raw_gate, set(), set()
    physical_to_logical = {
        physical_id: logical_id
        for logical_id, physical_id in logical_to_physical.items()
    }
    if len(physical_to_logical) != len(logical_to_physical):
        layer.fail("cfl_raw_gate_identity:duplicate_owned_physical_id")
        return raw_gate, set(), set()

    cases = raw_gate.get("cases")
    if not isinstance(cases, dict):
        layer.fail("cfl_raw_gate_identity:cases_not_object")
        return raw_gate, set(), set()
    raw_case_ids = set(cases)
    expected_physical_ids = set(physical_to_logical)
    if raw_case_ids != expected_physical_ids:
        layer.fail(
            "cfl_raw_gate_identity:case_coverage_mismatch:"
            f"missing={sorted(expected_physical_ids - raw_case_ids)}:"
            f"extra={sorted(raw_case_ids - expected_physical_ids)}"
        )
    accepted_physical = set(raw_gate.get("accepted_case_ids", []))
    rejected_physical = set(raw_gate.get("rejected_case_ids", []))
    if accepted_physical & rejected_physical:
        layer.fail("cfl_raw_gate_identity:accepted_rejected_overlap")
    if accepted_physical | rejected_physical != raw_case_ids:
        layer.fail("cfl_raw_gate_identity:outcome_coverage_mismatch")
    model = str(configuration.get("matter_model", "bsk24"))
    for physical_id, report in cases.items():
        if not isinstance(report, Mapping):
            layer.fail(f"cfl_raw_gate_identity:report_not_object:{physical_id}")
            continue
        if report.get("case_id") != physical_id:
            layer.fail(
                f"cfl_raw_gate_identity:report_case_id_mismatch:{physical_id}"
            )
        if model == "cfl":
            expected_report_fields = {
                "schema_version": "cfl_raw_local_physics_gate_v1",
                "profile_id": configuration.get("deformation_profile_id"),
                "profile_version": configuration.get(
                    "deformation_profile_version"
                ),
                "pressure_primitive_policy": configuration.get(
                    "pressure_primitive_policy"
                ),
                "baseline_parameter_set_id": configuration.get(
                    "baseline_parameter_set_id"
                ),
                "baseline_parameter_set_sha256": configuration.get(
                    "baseline_parameter_set_sha256"
                ),
            }
            for field, expected in expected_report_fields.items():
                if report.get(field) != expected:
                    layer.fail(
                        "cfl_raw_gate_identity:report_provenance_mismatch:"
                        f"{physical_id}:{field}"
                    )
            report_payload = dict(report)
            report_sha256 = report_payload.pop("report_sha256", None)
            try:
                calculated_report_sha256 = hashlib.sha256(
                    json.dumps(
                        report_payload,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                ).hexdigest()
            except (TypeError, ValueError, UnicodeEncodeError):
                calculated_report_sha256 = None
            if (
                not isinstance(report_sha256, str)
                or calculated_report_sha256 != report_sha256
            ):
                layer.fail(
                    f"cfl_raw_gate_identity:report_hash_mismatch:{physical_id}"
                )

    if isinstance(metadata, Mapping):
        for outcome, observed in (
            ("accepted", accepted_physical),
            ("rejected", rejected_physical),
        ):
            saved_ids = metadata.get(f"{outcome}_physical_case_ids")
            if not isinstance(saved_ids, list) or set(saved_ids) != observed:
                layer.fail(
                    f"metadata:{outcome}_physical_case_ids_mismatch"
                )
            if metadata.get(f"{outcome}_physical_case_count") != len(observed):
                layer.fail(
                    f"metadata:{outcome}_physical_case_count_mismatch"
                )

    normalized = dict(raw_gate)
    normalized["cases"] = {
        physical_to_logical.get(physical_id, physical_id): report
        for physical_id, report in cases.items()
    }
    for field in (
        "accepted_case_ids",
        "rejected_case_ids",
        "hard_rejected_case_ids",
        "unresolved_case_ids",
        "full_domain_accepted_case_ids",
        "full_domain_rejected_case_ids",
    ):
        value = raw_gate.get(field)
        if isinstance(value, list):
            normalized[field] = [
                physical_to_logical.get(str(physical_id), str(physical_id))
                for physical_id in value
            ]
    layer.checks["cfl_raw_gate_physical_case_count"] = len(raw_case_ids)
    layer.checks["cfl_raw_gate_accepted_physical_case_count"] = len(
        accepted_physical
    )
    layer.checks["cfl_raw_gate_rejected_physical_case_count"] = len(
        rejected_physical
    )
    return normalized, accepted_physical, rejected_physical


def _validate_cfl_identity_metadata(
    configuration: Mapping[str, Any],
    metadata: Any,
    case_plan: list[dict[str, str]] | None,
    saved_aliases: list[dict[str, str]] | None,
    accepted_physical: set[str],
    rejected_physical: set[str],
    layer: _Layer,
) -> None:
    if (
        not isinstance(metadata, Mapping)
        or case_plan is None
        or saved_aliases is None
    ):
        return
    expected = {
        "zero_amplitude_physical_case_id": configuration.get(
            "zero_amplitude_physical_case_id"
        ),
        "logical_case_count": len(case_plan) + len(saved_aliases),
        "logical_alias_count": len(saved_aliases),
        "physical_case_count": len(accepted_physical | rejected_physical),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            layer.fail(f"metadata:cfl_identity_mismatch:{field}")
    owner = configuration.get("zero_amplitude_control_owner")
    baseline_id = configuration.get("zero_amplitude_physical_case_id")
    if owner is True:
        if baseline_id not in accepted_physical:
            layer.fail("metadata:cfl_owner_baseline_not_accepted")
        if baseline_id in rejected_physical:
            layer.fail("metadata:cfl_owner_baseline_rejected")
    elif owner is False:
        if baseline_id in accepted_physical or baseline_id in rejected_physical:
            layer.fail("metadata:cfl_nonowner_contains_physical_baseline")
    else:
        layer.fail("metadata:cfl_zero_amplitude_owner_invalid")
    layer.checks["cfl_logical_case_count"] = expected["logical_case_count"]
    layer.checks["cfl_physical_case_count"] = expected["physical_case_count"]


def _validate_internal(
    packet: Path,
    *,
    configuration_hash_fn: Callable[[Mapping[str, Any]], str],
    expected_matter_model: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    layer = _Layer()
    configuration = _load_json(
        packet, "complete_configuration.json", layer
    )
    curve_only = bool(
        isinstance(configuration, Mapping)
        and configuration.get("curve_only_output") is True
    )
    required_files = tuple(
        relative
        for relative in CORE_REQUIRED_FILES
        if not curve_only or relative not in _CURVE_ONLY_OMITTED_CORE_FILES
    )
    for relative in required_files:
        if not (packet / relative).is_file():
            layer.fail(f"missing_required_file:{relative}")

    actual = _actual_packet_files(packet, layer)
    _validate_manifest(packet, actual, layer)

    named_json = {
        "complete_configuration.json",
        "trial_plan.json",
        "metadata.json",
        "run_state.json",
        "raw_gate_report.json",
        "accepted_rejected_cases.json",
        "plot_inventory.json",
        "plot_generation_provenance.json",
        "reproduction.json",
        "source_hashes.json",
        "manual_file_ledger.json",
    }
    additional_json = sorted(
        relative
        for relative in actual
        if relative.endswith(".json") and relative not in named_json
    )
    for relative in additional_json:
        _load_json(packet, relative, layer)
    layer.checks["strict_json_artifact_count"] = sum(
        relative.endswith(".json") for relative in actual
    )

    trial_plan = _load_json(packet, "trial_plan.json", layer)
    metadata = _load_json(packet, "metadata.json", layer)
    run_state = _load_json(packet, "run_state.json", layer)
    raw_gate = _load_json(packet, "raw_gate_report.json", layer)
    accepted_rejected = _load_json(packet, "accepted_rejected_cases.json", layer)
    inventory_json = _load_json(packet, "plot_inventory.json", layer)
    plot_provenance = _load_json(
        packet, "plot_generation_provenance.json", layer
    )
    reproduction = _load_json(packet, "reproduction.json", layer)
    source_hashes = _load_json(packet, "source_hashes.json", layer)
    ledger = _load_json(packet, "manual_file_ledger.json", layer)

    matter_model = "bsk24"
    if isinstance(configuration, dict):
        declared_model = configuration.get("matter_model", "bsk24")
        if declared_model not in {"bsk24", "cfl"}:
            layer.fail(f"matter_model:unsupported:{declared_model!r}")
        else:
            matter_model = str(declared_model)
    if (
        expected_matter_model is not None
        and matter_model != expected_matter_model
    ):
        layer.fail(
            "matter_model:validator_mismatch:"
            f"{matter_model}:{expected_matter_model}"
        )
    a0_alias_mode = matter_model == "cfl" or (
        matter_model == "bsk24"
        and isinstance(
            configuration.get("zero_amplitude_control_owner")
            if isinstance(configuration, Mapping)
            else None,
            bool,
        )
    )
    if a0_alias_mode and not (
        packet / "logical_case_aliases.csv"
    ).is_file():
        layer.fail("missing_required_file:logical_case_aliases.csv")

    configuration_hash = _configuration_hash_evidence(
        configuration,
        trial_plan,
        metadata,
        run_state,
        layer,
        configuration_hash_fn,
    )
    _validate_saved_output_paths(packet, configuration, trial_plan, layer)
    expected_plan_schema = (
        CFL_TRIAL_PLAN_SCHEMA if matter_model == "cfl" else TRIAL_PLAN_SCHEMA
    )
    if not isinstance(trial_plan, dict) or trial_plan.get(
        "schema_id"
    ) != expected_plan_schema:
        layer.fail("trial_plan:unsupported_schema")
    if (
        matter_model == "cfl"
        and isinstance(trial_plan, dict)
        and trial_plan.get("matter_model") != "cfl"
    ):
        layer.fail("trial_plan:matter_model_mismatch")
    if matter_model == "cfl" and (
        not isinstance(metadata, dict)
        or metadata.get("matter_model") != "cfl"
    ):
        layer.fail("metadata:matter_model_mismatch")
    if not isinstance(plot_provenance, dict) or plot_provenance.get(
        "schema_id"
    ) != PLOT_PROVENANCE_SCHEMA:
        layer.fail("plot_generation_provenance:unsupported_schema")
    _validate_anchor_selection(
        configuration,
        metadata,
        layer,
    )
    for relative, payload in (("metadata.json", metadata), ("run_state.json", run_state)):
        if isinstance(payload, dict) and payload.get("packet_status") != "complete":
            layer.fail(f"packet_completion:{relative}:not_complete")

    _validate_ledger(packet, actual, ledger, layer)
    case_plan = _read_csv(packet, "case_plan.csv", layer)
    case_ledger = _read_csv(packet, "case_ledger.csv", layer)
    logical_aliases = (
        _read_csv(packet, "logical_case_aliases.csv", layer)
        if a0_alias_mode
        else None
    )
    raw_gate_for_consistency = raw_gate
    accepted_physical: set[str] = set()
    rejected_physical: set[str] = set()
    if a0_alias_mode:
        (
            raw_gate_for_consistency,
            accepted_physical,
            rejected_physical,
        ) = _normalize_cfl_raw_gate_identities(
            configuration,
            raw_gate,
            case_plan,
            metadata,
            layer,
        )
    accepted, rejected = _validate_case_consistency(
        packet,
        matter_model=matter_model,
        case_plan=case_plan,
        case_ledger=case_ledger,
        raw_gate=raw_gate_for_consistency,
        accepted_rejected=accepted_rejected,
        metadata=metadata,
        layer=layer,
        bsk24_raw_profile_evidence=(
            # Deduplicated BSk24 lifecycle rows use logical IDs, while the
            # complete raw-gate evidence remains keyed by physical identity.
            (raw_gate, accepted_physical, rejected_physical)
            if matter_model == "bsk24" and a0_alias_mode
            else None
        ),
    )
    if a0_alias_mode and isinstance(configuration, dict):
        if matter_model == "cfl":
            _validate_cfl_raw_gate_profiles(
                _read_csv(packet, "raw_gate_profiles.csv", layer),
                raw_gate=raw_gate,
                accepted=accepted_physical,
                rejected=rejected_physical,
                layer=layer,
            )
        if isinstance(trial_plan, dict):
            _validate_cfl_plan_aliases(
                configuration,
                trial_plan,
                case_plan,
                logical_aliases,
                layer,
            )
        _validate_cfl_case_aliases(
            configuration,
            case_plan,
            case_ledger,
            layer,
        )
        _validate_cfl_identity_metadata(
            configuration,
            metadata,
            case_plan,
            logical_aliases,
            accepted_physical,
            rejected_physical,
            layer,
        )
    inventory_csv = _read_csv(packet, "plot_inventory.csv", layer)
    _validate_plot_inventory(packet, inventory_csv, inventory_json, layer)
    _validate_reproduction(
        packet,
        reproduction,
        source_hashes,
        configuration_hash,
        layer,
    )
    _validate_packet_schema_and_summary(
        packet,
        metadata,
        layer,
        matter_model=matter_model,
    )

    if isinstance(source_hashes, dict):
        for relative, digest in source_hashes.items():
            if (
                not isinstance(relative, str)
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                layer.fail(f"source_hashes:invalid_entry:{relative!r}")
    elif source_hashes is not None:
        layer.fail("source_hashes:not_an_object")

    result = {
        "status": "pass" if not layer.failures else "fail",
        "failures": layer.failures,
        "warnings": layer.warnings,
        "checks": layer.checks,
    }
    context = {
        "configuration": configuration,
        "metadata": metadata,
        "run_state": run_state,
        "source_hashes": source_hashes,
        "accepted_case_ids": accepted,
        "rejected_case_ids": rejected,
        "accepted_physical_case_ids": accepted_physical,
        "rejected_physical_case_ids": rejected_physical,
        "case_ledger": case_ledger,
        "matter_model": matter_model,
        "plot_inventory": inventory_csv,
    }
    return result, context


def _validate_source_equivalence(
    packet_source_hashes: Any,
    current_source_hashes: Mapping[str, str] | None,
    *,
    required_source_paths: tuple[str, ...],
) -> dict[str, Any]:
    """Compare one packet with the exact current active-source inventory."""

    if not isinstance(packet_source_hashes, dict):
        return {
            "status": "unavailable",
            "failures": ["packet_source_hashes_unavailable"],
            "warnings": [],
            "matching": [],
            "drifted": {},
            "missing_from_packet": [],
            "missing_from_current": [],
        }
    if current_source_hashes is None:
        return {
            "status": "unavailable",
            "failures": ["current_source_hashes_unavailable"],
            "warnings": [],
            "matching": [],
            "drifted": {},
            "missing_from_packet": [],
            "missing_from_current": [],
        }
    packet_hashes = {str(key): str(value) for key, value in packet_source_hashes.items()}
    current_hashes = {str(key): str(value) for key, value in current_source_hashes.items()}
    packet_keys = set(packet_hashes)
    current_keys = set(current_hashes)
    packet_inventory_id = (
        _source_inventory_id_for_paths(packet_hashes) or "unrecognized"
    )
    current_inventory_id = (
        _source_inventory_id_for_paths(current_hashes)
        or "unrecognized"
    )
    missing_from_packet = sorted(current_keys - packet_keys)
    missing_from_current = sorted(packet_keys - current_keys)
    drifted = {
        relative: {
            "packet_sha256": packet_hashes[relative],
            "current_sha256": current_hashes[relative],
        }
        for relative in sorted(packet_keys & current_keys)
        if packet_hashes[relative] != current_hashes[relative]
    }
    required_missing = sorted(set(required_source_paths) - packet_keys)
    matching = sorted(
        relative
        for relative in packet_keys & current_keys
        if relative not in drifted
    )
    failures: list[str] = []
    if packet_inventory_id != SOURCE_INVENTORY_ID:
        failures.append("packet_source_inventory_unrecognized")
    if current_inventory_id != SOURCE_INVENTORY_ID:
        failures.append("current_source_inventory_unrecognized")
    if drifted:
        failures.append("current_source_drift")
    if missing_from_packet or required_missing:
        failures.append("packet_source_coverage_incomplete")
    if missing_from_current:
        failures.append("current_source_coverage_incomplete")
    return {
        "status": (
            "equivalent"
            if not failures
            else "drift_detected_or_incomplete_coverage"
        ),
        "failures": failures,
        "warnings": [],
        "packet_source_inventory_id": packet_inventory_id,
        "current_source_inventory_id": current_inventory_id,
        "inventory_compatibility_mode": "exact_current_inventory_only",
        "matching": matching,
        "drifted": drifted,
        "missing_from_packet": missing_from_packet,
        "missing_from_current": missing_from_current,
        "required_missing_from_packet": required_missing,
        "packet_entry_count": len(packet_hashes),
        "current_entry_count": len(current_hashes),
    }

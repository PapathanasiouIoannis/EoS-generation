"""Stellar-output and scientific-completeness validation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from eos_generation._internal.saved_tables import (
    classify_saved_tidal_rows,
    summarize_fixed_mass_response_population,
)
from eos_generation.stellar.tov import (
    EOS_DISCONTINUITY_CONTRACT_VERSION,
    LAMBDA_FRAMEWORK_CAPABILITY,
    TIDAL_CORRECTION_VERSION,
    TIDAL_JUMP_FORMULA,
)
from eos_generation.reporting._validation_cases import (
    _case_ids,
    _csv_json_records_equal,
    _inventory_records,
)
from eos_generation.reporting._validation_io import (
    _Layer,
    _load_json,
    _read_csv,
)

STELLAR_STATUS_SCHEMA = "eos_generation_stellar_status_summary_v1"

_FIXED_MASS_FAIL_CLOSED_STATUSES = frozenset(
    {
        "unavailable_not_bracketed",
        "unavailable_outside_retained_eos_domain",
    }
)
_FIXED_MASS_RESULT_COLUMNS = (
    "mass_msun",
    "mass_residual_msun",
    "radius_km",
    "central_pressure_mev_fm3",
    "central_energy_density_mev_fm3",
    "central_sound_speed_squared",
    "k2",
    "lambda_dimensionless",
    "bracket_pressure_mev_fm3",
    "root_xtol_mev_fm3",
    "root_evaluation_count",
)
_FIXED_MASS_BACKGROUND_COLUMNS = (
    "mass_msun",
    "mass_residual_msun",
    "radius_km",
    "central_pressure_mev_fm3",
    "central_energy_density_mev_fm3",
    "central_sound_speed_squared",
    "root_xtol_mev_fm3",
    "root_evaluation_count",
)
_SEQUENCE_BACKGROUND_COLUMNS = (
    "Mass",
    "Radius",
    "P_Central",
    "Eps_Central",
    "CS2_Central",
    "eps_surf",
    "central_pressure_mev_fm3",
)
_SEQUENCE_RESULT_COLUMNS = (
    *_SEQUENCE_BACKGROUND_COLUMNS,
    "Lambda",
    "k2",
    "is_sampled_peak",
    "is_domain_end",
)

STELLAR_REQUIRED_FILES = (
    "stellar_sequences.csv",
    "fixed_mass_observables.csv",
    "stellar_convergence.json",
)

_THERMODYNAMIC_PROFILE_REQUIRED_NUMERIC_COLUMNS = (
    "epsilon_mev_fm3",
    "pressure_mev_fm3",
    "cs2",
    "delta_cs2",
    "baryon_density_fm3",
    "effective_baryon_enthalpy_mev",
    "gamma_eff",
    "energy_per_baryon_minus_neutron_rest_mev",
    "pressure_relative_to_direct",
    "baryon_density_relative_to_direct",
    "enthalpy_relative_to_direct",
)
_THERMODYNAMIC_RESIDUAL_REQUIRED_NUMERIC_COLUMNS = (
    "amplitude",
    "delta_mev_fm3",
    "epsilon_mev_fm3",
    "r_p_algebraic",
    "r_mu_algebraic",
    "r_p_independent",
    "r_p_independent_normalized",
    "r_mu_independent",
    "r_mu_independent_normalized",
    "r_c",
    "first_law_normalized",
    "dP_dEpsilon_independent",
    "mu_from_dEpsilon_dn_independent",
)

CURRENT_STELLAR_STATUS_FILES = (
    "stellar_status_summary.csv",
    "stellar_status_summary.json",
)

EXTENDED_CORE_REQUIRED_FILES = (
    "radial_profiles.csv",
    "deformation_support_fractions.csv",
)


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _has_saved_value(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


def _boolean_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if not _has_saved_value(value):
        return None
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _two_floats_or_none(value: Any) -> tuple[float, float] | None:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        return None
    lower = _float_or_none(parsed[0])
    upper = _float_or_none(parsed[1])
    if lower is None or upper is None:
        return None
    return lower, upper


def _a0_alias_mode(configuration: Mapping[str, Any]) -> bool:
    return configuration.get("matter_model", "bsk24") == "cfl" or isinstance(
        configuration.get("zero_amplitude_control_owner"), bool
    )


def _cfl_direct_baseline_expected(configuration: Mapping[str, Any]) -> bool:
    if not _a0_alias_mode(configuration):
        return True
    return configuration.get("zero_amplitude_control_owner") is True


def _validate_cfl_a0_identity(
    configuration: Mapping[str, Any],
    identity: Any,
    rows: list[dict[str, str]] | None,
    layer: _Layer,
) -> None:
    """Validate CFL A=0 ownership and exact binary64 identity evidence."""

    if configuration.get("matter_model", "bsk24") != "cfl":
        return
    prefix = "cfl_a0_identity"
    if not isinstance(identity, Mapping):
        layer.fail(f"{prefix}:report_missing_or_invalid")
        return
    owner = configuration.get("zero_amplitude_control_owner")
    baseline_id = configuration.get("zero_amplitude_physical_case_id")
    if not isinstance(owner, bool):
        layer.fail(f"{prefix}:owner_flag_invalid")
        return
    if identity.get("schema_id") != "eos_generation_cfl_a0_identity_v1":
        layer.fail(f"{prefix}:schema_invalid")
    if identity.get("zero_amplitude_control_owner") is not owner:
        layer.fail(f"{prefix}:owner_flag_mismatch")
    physical_ids = identity.get("physical_zero_case_ids")

    if not owner:
        if physical_ids != []:
            layer.fail(f"{prefix}:nonowner_physical_cases_not_empty")
        if identity.get("stellar_identity_status") != (
            "not_applicable_no_owned_a0_case"
        ):
            layer.fail(f"{prefix}:nonowner_status_invalid")
        if rows is None:
            layer.fail(f"{prefix}:table_missing")
        elif rows:
            layer.fail(f"{prefix}:nonowner_table_not_empty")
        return

    if not isinstance(baseline_id, str) or not baseline_id:
        layer.fail(f"{prefix}:baseline_physical_id_invalid")
    elif physical_ids != [baseline_id]:
        layer.fail(f"{prefix}:owned_physical_case_mismatch")
    if identity.get("floating_point_policy") != "numpy.array_equal_binary64":
        layer.fail(f"{prefix}:floating_point_policy_invalid")
    expected_stellar_status = (
        "pass_shared_direct_solution_alias"
        if _configuration_background_tov_requested(configuration)
        else "not_requested"
    )
    if identity.get("stellar_identity_status") != expected_stellar_status:
        layer.fail(f"{prefix}:stellar_identity_status_invalid")
    if rows is None:
        layer.fail(f"{prefix}:table_missing")
        return
    expected_quantities = {
        "epsilon",
        "pressure",
        "cs2",
        "baryon_density",
        "baryon_chemical_potential",
    }
    observed_quantities = [row.get("quantity", "") for row in rows]
    if len(rows) != len(expected_quantities) or set(observed_quantities) != (
        expected_quantities
    ):
        layer.fail(f"{prefix}:table_quantity_coverage_invalid")
    for index, row in enumerate(rows):
        context = f"{prefix}:row_{index}"
        residual = _float_or_none(row.get("maximum_absolute_residual"))
        if (
            row.get("scope") != "thermodynamic"
            or row.get("stage") != "reference"
            or row.get("array_equal") != "True"
            or row.get("status") != "pass"
            or residual != 0.0
        ):
            layer.fail(f"{context}:not_exact_binary64_identity")


def _validate_bsk24_a0_identity(
    configuration: Mapping[str, Any],
    identity: Any,
    rows: list[dict[str, str]] | None,
    layer: _Layer,
) -> None:
    """Validate deduplicated BSk24 A=0 identity and owner semantics."""

    if (
        configuration.get("matter_model", "bsk24") != "bsk24"
        or not isinstance(
            configuration.get("zero_amplitude_control_owner"), bool
        )
    ):
        return
    prefix = "bsk24_a0_identity"
    if not isinstance(identity, Mapping):
        layer.fail(f"{prefix}:report_missing_or_invalid")
        return
    owner = configuration["zero_amplitude_control_owner"]
    baseline_id = configuration.get("zero_amplitude_physical_case_id")
    if identity.get("schema_id") != "eos_generation_bsk24_a0_identity_v2":
        layer.fail(f"{prefix}:schema_invalid")
    if identity.get("zero_amplitude_control_owner") is not owner:
        layer.fail(f"{prefix}:owner_flag_mismatch")
    if identity.get("zero_amplitude_physical_case_id") != baseline_id:
        layer.fail(f"{prefix}:baseline_physical_id_mismatch")
    if identity.get("duplicate_zero_amplitude_stellar_solver_calls") != 0:
        layer.fail(f"{prefix}:duplicate_stellar_call_evidence_invalid")
    local = identity.get("local_thermodynamic_identity")
    if not isinstance(local, Mapping):
        layer.fail(f"{prefix}:local_report_invalid")
    if rows is None:
        layer.fail(f"{prefix}:table_missing")
        return
    if not owner:
        if rows:
            layer.fail(f"{prefix}:nonowner_table_not_empty")
        if not isinstance(local, Mapping) or local.get("status") != (
            "not_applicable_no_owned_a0_case"
        ):
            layer.fail(f"{prefix}:nonowner_local_status_invalid")
        expected_nonowner_stellar = (
            "not_applicable_no_owned_a0_case"
            if _configuration_background_tov_requested(configuration)
            else "not_requested"
        )
        if identity.get("stellar_identity_status") != expected_nonowner_stellar:
            layer.fail(f"{prefix}:nonowner_stellar_status_invalid")
        return
    if not isinstance(baseline_id, str) or not baseline_id.startswith(
        "bsk24_baseline_"
    ):
        layer.fail(f"{prefix}:baseline_physical_id_invalid")
    if not isinstance(local, Mapping) or local.get("status") != "pass":
        layer.fail(f"{prefix}:local_status_invalid")
    expected_stellar_status = (
        "pass_shared_direct_solution_alias"
        if _configuration_background_tov_requested(configuration)
        else "not_requested"
    )
    if identity.get("stellar_identity_status") != expected_stellar_status:
        layer.fail(f"{prefix}:stellar_identity_status_invalid")
    if not rows:
        layer.fail(f"{prefix}:owned_table_empty")
    local_deltas = local.get("deltas") if isinstance(local, Mapping) else None
    expected_rows: set[tuple[float, str]] = set()
    if isinstance(local_deltas, Mapping):
        for delta, quantities in local_deltas.items():
            delta_value = _float_or_none(delta)
            if delta_value is None or not isinstance(quantities, Mapping):
                layer.fail(f"{prefix}:local_delta_evidence_invalid")
                continue
            expected_rows.update(
                (delta_value, str(quantity)) for quantity in quantities
            )
    else:
        layer.fail(f"{prefix}:local_delta_evidence_invalid")
    observed_rows = {
        (_float_or_none(row.get("delta_mev_fm3")), str(row.get("quantity", "")))
        for row in rows
    }
    if None in {item[0] for item in observed_rows} or observed_rows != expected_rows:
        layer.fail(f"{prefix}:table_quantity_coverage_invalid")
    for index, row in enumerate(rows):
        residual = _float_or_none(row.get("maximum_absolute_residual"))
        if (
            row.get("scope") != "thermodynamic"
            or row.get("stage") != "refined"
            or row.get("array_equal") != "True"
            or row.get("status") != "pass"
            or residual != 0.0
        ):
            layer.fail(f"{prefix}:row_{index}:not_exact_binary64_identity")


def _cfl_surface_energy_density(
    configuration: Mapping[str, Any],
) -> float | None:
    if configuration.get("matter_model", "bsk24") != "cfl":
        return None
    profile = configuration.get("baseline_profile")
    surface = profile.get("surface") if isinstance(profile, Mapping) else None
    if not isinstance(surface, Mapping):
        return None
    value = _float_or_none(surface.get("energy_density_mev_fm3"))
    return value if value is not None and value > 0.0 else None


def _accepted_physical_output_case_ids(
    configuration: Mapping[str, Any],
    accepted: set[str],
    case_ledger: list[dict[str, str]] | None,
    layer: _Layer,
) -> set[str]:
    """Resolve accepted lifecycle IDs to unique reconstructed identities."""

    if not _a0_alias_mode(configuration):
        return set(accepted)
    if case_ledger is None:
        layer.fail("cfl_case_aliases:case_ledger_unavailable")
        return set()
    physical_ids: list[str] = []
    covered: set[str] = set()
    for row in case_ledger:
        case_id = str(row.get("case_id", ""))
        if case_id not in accepted:
            continue
        covered.add(case_id)
        physical_id = str(row.get("physical_case_id", ""))
        if not physical_id:
            layer.fail(f"cfl_case_aliases:physical_id_missing:{case_id}")
            continue
        if physical_id not in physical_ids:
            physical_ids.append(physical_id)
    if covered != accepted:
        layer.fail(
            "cfl_case_aliases:accepted_ledger_coverage_mismatch:"
            f"{sorted(accepted - covered)}"
        )
    physical = set(physical_ids)
    layer.checks["cfl_accepted_logical_case_count"] = len(accepted)
    layer.checks["cfl_accepted_physical_case_count"] = len(physical)
    return physical


def _expected_saved_stellar_case_ids(
    configuration: Mapping[str, Any],
    accepted: set[str],
    case_ledger: list[dict[str, str]] | None,
    layer: _Layer,
    *,
    accepted_physical_case_ids: set[str] | None = None,
) -> set[str]:
    """Resolve logical CFL aliases to the unique saved stellar identities."""

    if not _a0_alias_mode(configuration):
        return {"direct", *accepted}
    if not isinstance(
        configuration.get("zero_amplitude_control_owner"), bool
    ):
        layer.fail("cfl_case_aliases:zero_amplitude_control_owner_invalid")
    baseline_id = configuration.get("zero_amplitude_physical_case_id")
    if not isinstance(baseline_id, str) or not baseline_id:
        layer.fail("cfl_case_aliases:baseline_physical_id_invalid")
        return set()
    physical_ids = (
        _accepted_physical_output_case_ids(
            configuration,
            accepted,
            case_ledger,
            layer,
        )
        if accepted_physical_case_ids is None
        else accepted_physical_case_ids
    )
    owner = configuration.get("zero_amplitude_control_owner") is True
    if not owner and baseline_id in physical_ids:
        layer.fail("cfl_case_aliases:nonowner_claims_physical_a0")
    expected = {
        physical_id
        for physical_id in physical_ids
        if physical_id != baseline_id
    }
    if owner:
        expected.add("direct")
    layer.checks["cfl_saved_stellar_case_ids"] = sorted(expected)
    return expected


def _strict_json_object(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, "missing_json"

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token!r}")

    try:
        payload = json.loads(value, parse_constant=reject_constant)
    except (TypeError, ValueError):
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, "json_not_object"
    try:
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return None, "json_not_strict"
    if canonical != value:
        return None, "json_not_canonical"
    return payload, None


def _validate_cfl_tidal_jump_row(
    row: Mapping[str, Any],
    *,
    schema: str,
    surface_energy_density: float,
    layer: _Layer,
    context: str,
) -> None:
    """Require one saved, negative, exactly-once bare-surface correction."""

    prefix = f"cfl_tidal_surface_jump:{schema}:{context}"

    def fail(reason: str) -> None:
        layer.fail(f"{prefix}:{reason}")

    for field in (
        "tidal_expected_jump_count",
        "tidal_applied_jump_count",
        "tidal_surface_jump_count",
    ):
        if _integer_or_none(row.get(field)) != 1:
            fail(f"{field}_not_one")
    delta_y = _float_or_none(row.get("tidal_surface_delta_y"))
    y_before = _float_or_none(row.get("tidal_surface_y_before"))
    y_after = _float_or_none(row.get("tidal_surface_y_after"))
    surface_pressure = _float_or_none(
        row.get("tidal_surface_event_pressure_mev_fm3")
    )
    if delta_y is None or delta_y >= 0.0:
        fail("delta_y_not_finite_negative")
    if y_before is None or y_after is None:
        fail("surface_y_values_not_finite")
    elif delta_y is not None and y_after != y_before + delta_y:
        fail("surface_y_algebra_mismatch")
    if surface_pressure != 0.0:
        fail("surface_event_pressure_not_exact_zero")

    payload, json_error = _strict_json_object(
        row.get("tidal_jump_evidence_json")
    )
    if payload is None:
        fail(str(json_error))
        return
    if payload.get("schema_version") != "tov_lambda_diagnostic_v1":
        fail("diagnostic_schema_mismatch")
    if payload.get("scientific_status") != LAMBDA_FRAMEWORK_CAPABILITY:
        fail("diagnostic_status_not_validated")
    if payload.get("calculation_lambda_validated") is not True:
        fail("diagnostic_validation_claim_missing")
    if payload.get("expected_jump_count") != 1:
        fail("diagnostic_expected_jump_count_not_one")
    if payload.get("applied_jump_count") != 1:
        fail("diagnostic_applied_jump_count_not_one")
    if payload.get("skipped_discontinuity_ids") != []:
        fail("diagnostic_skipped_discontinuities_present")
    if payload.get("surface_event_pressure_MeV_fm3") != 0.0:
        fail("diagnostic_surface_pressure_not_exact_zero")
    if payload.get("correction_formula") != TIDAL_JUMP_FORMULA:
        fail("diagnostic_correction_formula_mismatch")
    if payload.get("correction_version") != TIDAL_CORRECTION_VERSION:
        fail("diagnostic_correction_version_mismatch")
    if (
        payload.get("discontinuity_contract_version")
        != EOS_DISCONTINUITY_CONTRACT_VERSION
    ):
        fail("diagnostic_discontinuity_contract_mismatch")
    row_identity_fields = (
        (
            "central_pressure_MeV_fm3",
            "central_pressure_mev_fm3",
        ),
        ("Mass", "Mass" if schema == "sequence" else "mass_msun"),
        ("Radius", "Radius" if schema == "sequence" else "radius_km"),
    )
    for payload_field, row_field in row_identity_fields:
        row_value = _float_or_none(row.get(row_field))
        if row_value is None or payload.get(payload_field) != row_value:
            fail(f"diagnostic_row_identity_mismatch:{payload_field}")

    jumps = payload.get("applied_jumps")
    if not isinstance(jumps, list) or len(jumps) != 1:
        fail("diagnostic_jump_list_not_exactly_one")
        return
    jump = jumps[0]
    if not isinstance(jump, Mapping) or jump.get("type") != "surface":
        fail("diagnostic_only_jump_not_surface")
        return
    if jump.get("pressure_MeV_fm3") != 0.0:
        fail("jump_pressure_not_exact_zero")
    if jump.get("inner_energy_density_MeV_fm3") != surface_energy_density:
        fail("jump_inner_energy_not_frozen_surface")
    if jump.get("outer_energy_density_MeV_fm3") != 0.0:
        fail("jump_outer_energy_not_vacuum")
    if (
        jump.get("signed_outward_delta_energy_density_MeV_fm3")
        != surface_energy_density
    ):
        fail("jump_energy_difference_mismatch")
    denominator = _float_or_none(jump.get("correction_denominator_Msun"))
    if denominator is None or denominator <= 0.0:
        fail("jump_denominator_not_positive")
    if jump.get("delta_y") != delta_y:
        fail("jump_delta_y_row_mismatch")
    if jump.get("y_before") != y_before or jump.get("y_after") != y_after:
        fail("jump_y_row_mismatch")
    if payload.get("y_surface_interior") != y_before:
        fail("diagnostic_interior_y_mismatch")
    if payload.get("y_surface_vacuum") != y_after:
        fail("diagnostic_vacuum_y_mismatch")
    if (
        payload.get("y_supplied_to_k2") != y_after
        or payload.get("y_R") != y_after
    ):
        fail("diagnostic_k2_y_mismatch")
    lambda_column = "Lambda" if schema == "sequence" else "lambda_dimensionless"
    if payload.get("k2") != _float_or_none(row.get("k2")):
        fail("diagnostic_k2_row_mismatch")
    if payload.get("Lambda") != _float_or_none(row.get(lambda_column)):
        fail("diagnostic_lambda_row_mismatch")


def _extended_output_claimed(configuration: Any, metadata: Any) -> bool:
    if isinstance(configuration, dict) and configuration.get(
        "extended_stellar_diagnostics_enabled"
    ) is True:
        return True
    if not isinstance(metadata, dict):
        return False
    extended = metadata.get("extended_tables")
    if isinstance(extended, bool):
        return extended
    if isinstance(extended, dict):
        return bool(extended)
    if isinstance(extended, list):
        return bool(extended)
    return False


def _validate_fixed_mass_completeness(
    packet: Path,
    configuration: Mapping[str, Any],
    accepted: set[str],
    layer: _Layer,
    *,
    availability: _Layer,
    require_tides: bool = True,
    retained_pressure_endpoints: Mapping[str, float] | None = None,
    expected_case_ids: set[str] | None = None,
) -> None:
    rows = _read_csv(packet, "fixed_mass_observables.csv", layer)
    if rows is None:
        return
    import pandas as pd

    frame = pd.DataFrame(rows)
    tidal_classification = classify_saved_tidal_rows(
        frame, schema="fixed_mass"
    )
    stages = [
        str(item.get("name"))
        for item in configuration.get("tov_stages", [])
        if isinstance(item, dict) and item.get("name")
    ]
    masses = [
        value
        for item in configuration.get("fixed_masses_msun", [])
        if (value := _float_or_none(item)) is not None
    ]
    case_ids = (
        {"direct", *accepted}
        if expected_case_ids is None
        else expected_case_ids
    )
    if retained_pressure_endpoints is not None:
        missing_endpoints = case_ids - set(retained_pressure_endpoints)
        if missing_endpoints:
            layer.fail(
                "fixed_mass:retained_endpoints_missing:"
                f"{sorted(missing_endpoints)}"
            )
    expected = {
        (case_id, stage, mass.hex())
        for case_id in case_ids
        for stage in stages
        for mass in masses
    }
    actual: set[tuple[str, str, str]] = set()
    duplicates: set[tuple[str, str, str]] = set()
    background_unavailable = 0
    valid_background_unavailable = 0
    unavailable_reasons: dict[str, int] = {}
    tidal_unavailable = 0
    hard_tidal_invalid = 0
    missing_reason = 0
    has_tidal_failure_reason = not rows or "tidal_failure_reason" in rows[0]
    is_cfl = configuration.get("matter_model", "bsk24") == "cfl"
    cfl_surface_energy = _cfl_surface_energy_density(configuration)
    if is_cfl and cfl_surface_energy is None:
        layer.fail("cfl_tidal_surface_jump:frozen_surface_energy_unavailable")
    if not has_tidal_failure_reason:
        layer.fail("fixed_mass:tidal_failure_reason_column_missing")
    for position, row in enumerate(rows):
        mass = _float_or_none(row.get("target_mass_msun"))
        if mass is None:
            layer.fail(f"fixed_mass:invalid_target_mass:{row.get('case_id', '')}")
            continue
        key = (row.get("case_id", ""), row.get("stage", ""), mass.hex())
        if key in actual:
            duplicates.add(key)
        actual.add(key)
        classification = tidal_classification.iloc[position]
        background_ok = bool(classification["background_success"])
        if (
            is_cfl
            and row.get("tidal_status") == LAMBDA_FRAMEWORK_CAPABILITY
            and cfl_surface_energy is not None
        ):
            _validate_cfl_tidal_jump_row(
                row,
                schema="fixed_mass",
                surface_energy_density=cfl_surface_energy,
                layer=layer,
                context=(
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}"
                ),
            )
        if not background_ok:
            background_unavailable += 1
            status = str(row.get("status", ""))
            reason = str(row.get("reason") or "").strip()
            unavailable_row_valid = True
            if status not in _FIXED_MASS_FAIL_CLOSED_STATUSES:
                unavailable_row_valid = False
                layer.fail(
                    "fixed_mass:invalid_background_status:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}:{status}"
                )
            elif not reason:
                unavailable_row_valid = False
                layer.fail(
                    "fixed_mass:unavailable_reason_missing:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}:{status}"
                )
            else:
                unavailable_reasons[reason] = (
                    unavailable_reasons.get(reason, 0) + 1
                )
            contaminated = [
                column
                for column in _FIXED_MASS_RESULT_COLUMNS
                if _has_saved_value(row.get(column))
            ]
            if contaminated:
                unavailable_row_valid = False
                layer.fail(
                    "fixed_mass:unavailable_row_has_observables:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}:"
                    f"{','.join(contaminated)}"
                )
            if str(row.get("tidal_status") or "").strip() or str(
                row.get("tidal_failure_reason") or ""
            ).strip():
                unavailable_row_valid = False
                layer.fail(
                    "fixed_mass:unavailable_row_has_tidal_claim:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}"
                )
            if unavailable_row_valid:
                valid_background_unavailable += 1
        else:
            invalid_background_columns = [
                column
                for column in _FIXED_MASS_BACKGROUND_COLUMNS
                if _float_or_none(row.get(column)) is None
            ]
            mass_value = _float_or_none(row.get("mass_msun"))
            radius = _float_or_none(row.get("radius_km"))
            pressure = _float_or_none(row.get("central_pressure_mev_fm3"))
            energy = _float_or_none(
                row.get("central_energy_density_mev_fm3")
            )
            sound_speed = _float_or_none(
                row.get("central_sound_speed_squared")
            )
            bracket = _two_floats_or_none(
                row.get("bracket_pressure_mev_fm3")
            )
            root_xtol = _float_or_none(row.get("root_xtol_mev_fm3"))
            evaluations = _integer_or_none(row.get("root_evaluation_count"))
            endpoint = (
                retained_pressure_endpoints.get(str(row.get("case_id") or ""))
                if retained_pressure_endpoints is not None
                else None
            )
            if (
                invalid_background_columns
                or mass_value is None
                or mass_value <= 0.0
                or radius is None
                or radius <= 0.0
                or pressure is None
                or pressure <= 0.0
                or energy is None
                or energy <= 0.0
                or sound_speed is None
                or not 0.0 < sound_speed <= 1.0
                or bracket is None
                or bracket[0] <= 0.0
                or bracket[0] >= bracket[1]
                or not bracket[0] <= pressure <= bracket[1]
                or root_xtol is None
                or root_xtol <= 0.0
                or evaluations is None
                or evaluations <= 0
                or _has_saved_value(row.get("reason"))
                or (
                    retained_pressure_endpoints is not None
                    and (
                        endpoint is None
                        or pressure > endpoint
                        or bracket[1] > endpoint
                    )
                )
            ):
                layer.fail(
                    "fixed_mass:invalid_solved_background_row:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}:"
                    f"{','.join(invalid_background_columns)}"
                )
        tidal_ok = bool(classification["tidal_valid"])
        if background_ok and require_tides and not tidal_ok:
            reason = str(classification["tidal_validity_reason"])
            tidal_status = str(row.get("tidal_status") or "").strip()
            tidal_failure_reason = str(
                row.get("tidal_failure_reason") or ""
            ).strip()
            tidal_claims = [
                column
                for column in ("k2", "lambda_dimensionless")
                if _has_saved_value(row.get(column))
            ]
            if (
                tidal_status == "failed_closed"
                and tidal_failure_reason
                and not tidal_claims
            ):
                tidal_unavailable += 1
            else:
                hard_tidal_invalid += 1
                layer.fail(
                    "fixed_mass:invalid_tidal_row:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}:{reason}:"
                    f"claims={','.join(tidal_claims)}"
                )
            if tidal_status == "failed_closed" and not tidal_failure_reason:
                missing_reason += 1
        elif background_ok and require_tides and tidal_ok:
            if _has_saved_value(row.get("tidal_failure_reason")):
                layer.fail(
                    "fixed_mass:validated_tidal_row_has_failure_reason:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}"
                )
        elif background_ok and not require_tides:
            unexpected_tidal = bool(
                row.get("tidal_status") != "not_requested_background_only"
                or _has_saved_value(row.get("tidal_failure_reason"))
                or _has_saved_value(row.get("k2"))
                or _has_saved_value(row.get("lambda_dimensionless"))
            )
            if unexpected_tidal:
                layer.fail(
                    "fixed_mass:unexpected_tidal_work:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('target_mass_msun', '')}"
                )
    if duplicates:
        layer.fail(f"fixed_mass:duplicate_targets:{len(duplicates)}")
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        layer.fail(f"fixed_mass:missing_requested_rows:{len(missing)}")
    if unexpected:
        layer.fail(f"fixed_mass:unexpected_rows:{len(unexpected)}")
    if missing_reason:
        layer.fail(f"fixed_mass:tidal_failure_reason_missing:{missing_reason}")
    if valid_background_unavailable:
        availability.fail(
            "fixed_mass:background_unavailable:"
            f"{valid_background_unavailable}"
        )
    if tidal_unavailable:
        availability.fail(
            f"fixed_mass:tidal_unavailable_failed_closed:{tidal_unavailable}"
        )
    layer.checks["fixed_mass_requested_rows"] = len(expected)
    layer.checks["fixed_mass_present_rows"] = len(actual)
    layer.checks["fixed_mass_background_failures"] = background_unavailable
    layer.checks["fixed_mass_valid_background_unavailable"] = (
        valid_background_unavailable
    )
    layer.checks["fixed_mass_fail_closed_reasons"] = dict(
        sorted(unavailable_reasons.items())
    )
    layer.checks["fixed_mass_tidal_unavailable"] = tidal_unavailable
    layer.checks["fixed_mass_hard_tidal_invalid"] = hard_tidal_invalid
    layer.checks["fixed_mass_tidal_invalidity_reasons"] = {
        str(key): int(value)
        for key, value in tidal_classification.loc[
            tidal_classification["background_success"]
            & ~tidal_classification["tidal_valid"],
            "tidal_validity_reason",
        ]
        .value_counts()
        .sort_index()
        .items()
    }


def _validate_sequence_completeness(
    packet: Path,
    configuration: Mapping[str, Any],
    accepted: set[str],
    layer: _Layer,
    *,
    availability: _Layer,
    require_tides: bool = True,
    require_configured_count: bool = True,
    retained_pressure_endpoints: Mapping[str, float] | None = None,
    expected_case_ids: set[str] | None = None,
) -> None:
    rows = _read_csv(packet, "stellar_sequences.csv", layer)
    if rows is None:
        return
    import pandas as pd

    frame = pd.DataFrame(rows)
    tidal_classification = classify_saved_tidal_rows(
        frame, schema="sequence"
    )
    case_ids = (
        {"direct", *accepted}
        if expected_case_ids is None
        else expected_case_ids
    )
    expected_groups: dict[tuple[str, str], int] = {}
    if retained_pressure_endpoints is not None:
        missing_endpoints = case_ids - set(retained_pressure_endpoints)
        if missing_endpoints:
            layer.fail(
                "stellar_sequences:retained_endpoints_missing:"
                f"{sorted(missing_endpoints)}"
            )
    for case_id in case_ids:
        for stage in configuration.get("tov_stages", []):
            if not isinstance(stage, dict) or not stage.get("name"):
                continue
            points = stage.get("sequence_points")
            if isinstance(points, int) and not isinstance(points, bool):
                expected_groups[(case_id, str(stage["name"]))] = points
    actual_groups: dict[tuple[str, str], int] = {}
    background_failures = 0
    valid_background_unavailable = 0
    tidal_unavailable = 0
    hard_tidal_invalid = 0
    endpoint_below_floor_groups: set[tuple[str, str]] = set()
    invalid_endpoint_below_floor_groups: set[tuple[str, str]] = set()
    is_cfl = configuration.get("matter_model", "bsk24") == "cfl"
    cfl_surface_energy = _cfl_surface_energy_density(configuration)
    if is_cfl and cfl_surface_energy is None:
        layer.fail("cfl_tidal_surface_jump:frozen_surface_energy_unavailable")
    for position, row in enumerate(rows):
        key = (row.get("case_id", ""), row.get("stage", ""))
        actual_groups[key] = actual_groups.get(key, 0) + 1
        classification = tidal_classification.iloc[position]
        background_ok = bool(classification["background_success"])
        tidal_ok = bool(classification["tidal_valid"])
        if (
            is_cfl
            and row.get("tidal_status") == LAMBDA_FRAMEWORK_CAPABILITY
            and cfl_surface_energy is not None
        ):
            _validate_cfl_tidal_jump_row(
                row,
                schema="sequence",
                surface_energy_density=cfl_surface_energy,
                layer=layer,
                context=(
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('attempted_index', '')}"
                ),
            )
        if not background_ok:
            background_failures += 1
            status = str(row.get("calculation_status") or "").strip()
            category = str(row.get("failure_category") or "").strip()
            reason = str(row.get("failure_reason") or "").strip()
            pressure = _float_or_none(
                row.get("central_pressure_mev_fm3")
            )
            endpoint = (
                retained_pressure_endpoints.get(str(row.get("case_id") or ""))
                if retained_pressure_endpoints is not None
                else None
            )
            contaminated = [
                column
                for column in _SEQUENCE_RESULT_COLUMNS
                if column != "central_pressure_mev_fm3"
                and _has_saved_value(row.get(column))
            ]
            tidal_claimed = any(
                _has_saved_value(row.get(column))
                for column in ("tidal_status", "tidal_failure_reason")
            )
            valid_failure = bool(
                status == "failed"
                and category
                and reason
                and pressure is not None
                and pressure > 0.0
                and (
                    retained_pressure_endpoints is None
                    or (endpoint is not None and pressure <= endpoint)
                )
                and not contaminated
                and not tidal_claimed
            )
            if valid_failure:
                valid_background_unavailable += 1
                if category == "eos_endpoint_below_sequence_floor":
                    endpoint_below_floor_groups.add(key)
            else:
                layer.fail(
                    "stellar_sequences:invalid_failed_background_row:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('attempted_index', '')}:status={status}:"
                    f"category={category}:claims={','.join(contaminated)}"
                )
                if category == "eos_endpoint_below_sequence_floor":
                    invalid_endpoint_below_floor_groups.add(key)
            continue
        if str(row.get("calculation_status") or "").strip() != "success":
            layer.fail(
                "stellar_sequences:invalid_background_status:"
                f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                f"{row.get('attempted_index', '')}:"
                f"{row.get('calculation_status', '')}"
            )
            continue
        invalid_background_columns = [
            column
            for column in _SEQUENCE_BACKGROUND_COLUMNS
            if _float_or_none(row.get(column)) is None
        ]
        mass_value = _float_or_none(row.get("Mass"))
        radius = _float_or_none(row.get("Radius"))
        pressure = _float_or_none(row.get("P_Central"))
        labelled_pressure = _float_or_none(
            row.get("central_pressure_mev_fm3")
        )
        energy = _float_or_none(row.get("Eps_Central"))
        sound_speed = _float_or_none(row.get("CS2_Central"))
        surface_energy = _float_or_none(row.get("eps_surf"))
        endpoint = (
            retained_pressure_endpoints.get(str(row.get("case_id") or ""))
            if retained_pressure_endpoints is not None
            else None
        )
        if (
            invalid_background_columns
            or mass_value is None
            or mass_value <= 0.0
            or radius is None
            or radius <= 0.0
            or pressure is None
            or pressure <= 0.0
            or labelled_pressure != pressure
            or energy is None
            or energy <= 0.0
            or sound_speed is None
            or not 0.0 < sound_speed <= 1.0
            or surface_energy is None
            or surface_energy < 0.0
            or (
                retained_pressure_endpoints is not None
                and (endpoint is None or pressure > endpoint)
            )
            or _has_saved_value(row.get("failure_category"))
            or _has_saved_value(row.get("failure_reason"))
        ):
            layer.fail(
                "stellar_sequences:invalid_successful_background_row:"
                f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                f"{row.get('attempted_index', '')}:"
                f"{','.join(invalid_background_columns)}"
            )
        elif require_tides and not tidal_ok:
            reason = str(classification["tidal_validity_reason"])
            tidal_status = str(row.get("tidal_status") or "").strip()
            tidal_failure_reason = str(
                row.get("tidal_failure_reason") or ""
            ).strip()
            tidal_claims = [
                column
                for column in ("k2", "Lambda")
                if _has_saved_value(row.get(column))
            ]
            if (
                tidal_status == "failed_closed"
                and tidal_failure_reason
                and not tidal_claims
            ):
                tidal_unavailable += 1
            else:
                hard_tidal_invalid += 1
                layer.fail(
                    "stellar_sequences:invalid_tidal_row:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('attempted_index', '')}:{reason}:"
                    f"claims={','.join(tidal_claims)}"
                )
        elif require_tides and tidal_ok:
            if _has_saved_value(row.get("tidal_failure_reason")):
                layer.fail(
                    "stellar_sequences:validated_tidal_row_has_failure_reason:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('attempted_index', '')}"
                )
        elif not require_tides:
            unexpected_tidal = bool(
                row.get("tidal_status") != "not_requested_background_only"
                or _has_saved_value(row.get("tidal_failure_reason"))
                or _has_saved_value(row.get("k2"))
                or _has_saved_value(row.get("Lambda"))
            )
            if unexpected_tidal:
                layer.fail(
                    "stellar_sequences:unexpected_tidal_work:"
                    f"{row.get('case_id', '')}:{row.get('stage', '')}:"
                    f"{row.get('attempted_index', '')}"
                )
    for key, expected_count in expected_groups.items():
        actual_count = actual_groups.get(key, 0)
        if require_configured_count and actual_count != expected_count:
            if (
                actual_count == 1
                and key in endpoint_below_floor_groups
                and key not in invalid_endpoint_below_floor_groups
            ):
                availability.fail(
                    "stellar_sequences:endpoint_below_sequence_floor:"
                    f"{key[0]}:{key[1]}"
                )
            else:
                layer.fail(
                    "stellar_sequences:requested_count_mismatch:"
                    f"{key[0]}:{key[1]}:{actual_count}:{expected_count}"
                )
        elif not require_configured_count and actual_count == 0:
            layer.fail(
                "stellar_sequences:background_group_missing:"
                f"{key[0]}:{key[1]}"
            )
    for key in sorted(set(actual_groups) - set(expected_groups)):
        layer.fail(f"stellar_sequences:unexpected_group:{key[0]}:{key[1]}")
    if valid_background_unavailable:
        availability.fail(
            "stellar_sequences:background_unavailable:"
            f"{valid_background_unavailable}"
        )
    if tidal_unavailable:
        availability.fail(
            "stellar_sequences:tidal_unavailable_failed_closed:"
            f"{tidal_unavailable}"
        )
    layer.checks["stellar_sequence_requested_rows"] = (
        sum(expected_groups.values())
        if require_configured_count
        else len(rows)
    )
    layer.checks["stellar_sequence_present_rows"] = len(rows)
    layer.checks["stellar_sequence_background_failures"] = background_failures
    layer.checks["stellar_sequence_valid_background_unavailable"] = (
        valid_background_unavailable
    )
    layer.checks["stellar_sequence_tidal_unavailable"] = tidal_unavailable
    layer.checks["stellar_sequence_hard_tidal_invalid"] = hard_tidal_invalid
    layer.checks["stellar_sequence_tidal_invalidity_reasons"] = {
        str(key): int(value)
        for key, value in tidal_classification.loc[
            tidal_classification["background_success"]
            & ~tidal_classification["tidal_valid"],
            "tidal_validity_reason",
        ]
        .value_counts()
        .sort_index()
        .items()
    }


def _integer_or_none(value: Any) -> int | None:
    numeric = _float_or_none(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _configuration_background_tov_requested(
    configuration: Mapping[str, Any],
) -> bool:
    """Return the ordinary packet's single stellar-execution switch."""

    return configuration.get("stellar_enabled") is True


def _configuration_fixed_mass_requested(
    configuration: Mapping[str, Any],
) -> bool:
    return (
        configuration.get("stellar_enabled") is True
        and configuration.get("curve_only_output") is not True
    )


def _configuration_maximum_mass_requested(
    configuration: Mapping[str, Any],
) -> bool:
    return (
        configuration.get("stellar_enabled") is True
        and configuration.get("curve_only_output") is not True
    )


def _configuration_tidal_requested(
    configuration: Mapping[str, Any],
) -> bool:
    return configuration.get("stellar_enabled") is True


def _saved_lifecycle_configuration(
    configuration: Mapping[str, Any],
) -> SimpleNamespace:
    """Expose only the serialized controls used by lifecycle completeness."""

    stages = tuple(
        SimpleNamespace(
            name=str(item.get("name", "")),
            sequence_points=int(item.get("sequence_points", 0)),
        )
        for item in configuration.get("tov_stages", ())
        if isinstance(item, Mapping) and item.get("name")
    )
    return SimpleNamespace(
        zero_amplitude_physical_case_id=configuration.get(
            "zero_amplitude_physical_case_id"
        ),
        background_tov_requested=(
            _configuration_background_tov_requested(configuration)
        ),
        fixed_mass_background_requested=(
            _configuration_fixed_mass_requested(configuration)
        ),
        maximum_mass_requested=(
            _configuration_maximum_mass_requested(configuration)
        ),
        fixed_masses_msun=tuple(
            value
            for item in configuration.get("fixed_masses_msun", ())
            if (value := _float_or_none(item)) is not None
        ),
        tov_stages=stages,
    )


def _validate_final_lifecycle(
    packet: Path,
    configuration: Mapping[str, Any],
    accepted: set[str],
    layer: _Layer,
    *,
    expected_stellar_case_ids: set[str] | None = None,
) -> None:
    rows = _read_csv(packet, "case_ledger.csv", layer)
    if rows is None:
        return
    import pandas as pd

    from eos_generation._internal.lifecycle import (
        _completed_stellar_case_ids,
        _maximum_mass_availability_status,
        _requested_fixed_masses_status,
    )

    stellar_enabled = _configuration_background_tov_requested(configuration)
    saved_config = _saved_lifecycle_configuration(configuration)
    fixed = (
        pd.read_csv(packet / "fixed_mass_observables.csv")
        if (packet / "fixed_mass_observables.csv").is_file()
        else None
    )
    maximum = (
        pd.read_csv(packet / "maximum_mass_screening.csv")
        if (packet / "maximum_mass_screening.csv").is_file()
        else None
    )
    completed_stellar: set[str] = set()
    if stellar_enabled:
        sequences = (
            pd.read_csv(packet / "stellar_sequences.csv")
            if (packet / "stellar_sequences.csv").is_file()
            else None
        )
        if expected_stellar_case_ids is None:
            accepted_physical_ids = tuple(
                dict.fromkeys(
                    str(row.get("physical_case_id") or row.get("case_id", ""))
                    for row in rows
                    if row.get("case_id", "") in accepted
                )
            )
        else:
            accepted_physical_ids = tuple(sorted(expected_stellar_case_ids))
        completed_stellar = _completed_stellar_case_ids(
            sequences,
            fixed,
            saved_config,
            accepted_case_ids=accepted_physical_ids,
        )
    for row in rows:
        case_id = row.get("case_id", "")
        accepted_row = case_id in accepted
        physical_case_id = str(
            row.get("physical_case_id") or case_id
        )
        saved_stellar_case_id = physical_case_id
        if (
            _a0_alias_mode(configuration)
            and physical_case_id
            == configuration.get("zero_amplitude_physical_case_id")
        ):
            saved_stellar_case_id = "direct"
        expected_stellar = (
            "completed"
            if (
                accepted_row
                and stellar_enabled
                and saved_stellar_case_id in completed_stellar
            )
            else "incomplete_or_failed"
            if accepted_row and stellar_enabled
            else "disabled"
            if accepted_row
            else "skipped_due_to_raw_gate_rejection"
        )
        if row.get("stellar_calculation") != expected_stellar:
            layer.fail(
                "case_lifecycle:stellar_status_mismatch:"
                f"{case_id}:{row.get('stellar_calculation', '')}:{expected_stellar}"
            )
        expected_reconstruction = (
            "completed"
            if accepted_row
            else "skipped_due_to_raw_gate_rejection"
        )
        if row.get("pressure_reconstruction") != expected_reconstruction:
            layer.fail(
                "case_lifecycle:reconstruction_status_mismatch:"
                f"{case_id}:{row.get('pressure_reconstruction', '')}:"
                f"{expected_reconstruction}"
            )
        try:
            expected_fixed = _requested_fixed_masses_status(
                saved_config,
                saved_stellar_case_id,
                accepted=accepted_row,
                fixed_mass_rows=fixed,
            )
            expected_maximum = _maximum_mass_availability_status(
                saved_config,
                saved_stellar_case_id,
                accepted=accepted_row,
                maximum_mass_rows=maximum,
            )
        except Exception as exc:
            layer.fail(
                "case_lifecycle:availability_derivation_failed:"
                f"{case_id}:{type(exc).__name__}:{exc}"
            )
            continue
        if row.get("requested_fixed_masses_status") != expected_fixed:
            layer.fail(
                "case_lifecycle:fixed_mass_status_mismatch:"
                f"{case_id}:{row.get('requested_fixed_masses_status', '')}:"
                f"{expected_fixed}"
            )
        if row.get("maximum_mass_availability_status") != expected_maximum:
            layer.fail(
                "case_lifecycle:maximum_mass_status_mismatch:"
                f"{case_id}:{row.get('maximum_mass_availability_status', '')}:"
                f"{expected_maximum}"
            )
        expected_eligibility = (
            "evidence_only_raw_gate_not_accepted"
            if not accepted_row
            else "eligible_thermodynamic_case"
            if not stellar_enabled or expected_fixed == "not_requested"
            else "eligible_all_requested_fixed_masses_succeeded"
            if expected_fixed == "all_requested_fixed_masses_succeeded"
            else "ineligible_requested_fixed_masses_incomplete"
        )
        if row.get("student_view_eligibility_status") != expected_eligibility:
            layer.fail(
                "case_lifecycle:student_eligibility_mismatch:"
                f"{case_id}:{row.get('student_view_eligibility_status', '')}:"
                f"{expected_eligibility}"
            )


def _validate_stellar_status_reporting(
    packet: Path,
    configuration: Mapping[str, Any],
    metadata: Any,
    layer: _Layer,
) -> None:
    import pandas as pd

    summary_rows = _read_csv(packet, "stellar_status_summary.csv", layer)
    sequence_rows = _read_csv(packet, "stellar_sequences.csv", layer)
    fixed_rows = _read_csv(packet, "fixed_mass_observables.csv", layer)
    if summary_rows is None or sequence_rows is None or fixed_rows is None:
        return
    summary = pd.DataFrame(summary_rows)
    frames = {
        "sequence": pd.DataFrame(sequence_rows),
        "fixed_mass": pd.DataFrame(fixed_rows),
    }
    totals: dict[str, dict[str, int]] = {}
    count_fields = (
        "requested_count",
        "background_success_count",
        "background_failure_count",
        "tidal_validated_count",
        "tidal_failed_closed_count",
        "tidal_unavailable_count",
    )
    stages = [
        str(item["name"])
        for item in configuration.get("tov_stages", [])
        if isinstance(item, dict) and item.get("name")
    ]
    total_background_failures = 0
    total_failed_closed = 0
    total_unavailable = 0
    for scope, frame in frames.items():
        classification = classify_saved_tidal_rows(frame, schema=scope)
        scope_totals = {field: 0 for field in count_fields}
        for stage in stages:
            stage_mask = frame["stage"].astype(str).eq(stage)
            stage_rows = frame.loc[stage_mask]
            stage_classification = classification.loc[stage_rows.index]
            requested = int(len(stage_rows))
            background = int(stage_classification["background_success"].sum())
            validated = int(stage_classification["tidal_valid"].sum())
            tidal_status = (
                stage_rows["tidal_status"].astype(str)
                if "tidal_status" in stage_rows.columns
                else pd.Series("", index=stage_rows.index, dtype=str)
            )
            failed_closed = int(
                (
                    stage_classification["background_success"]
                    & tidal_status.eq("failed_closed")
                ).sum()
            )
            expected = {
                "requested_count": requested,
                "background_success_count": background,
                "background_failure_count": requested - background,
                "tidal_validated_count": validated,
                "tidal_failed_closed_count": failed_closed,
                "tidal_unavailable_count": requested - validated - failed_closed,
            }
            stored = summary.loc[
                summary["scope"].astype(str).eq(scope)
                & summary["stage"].astype(str).eq(stage)
            ]
            if len(stored) != 1:
                layer.fail(
                    f"stellar_status_summary:missing_or_duplicate:{scope}:{stage}"
                )
            else:
                record = stored.iloc[0]
                for field, expected_value in expected.items():
                    if _integer_or_none(record.get(field)) != expected_value:
                        layer.fail(
                            "stellar_status_summary:count_mismatch:"
                            f"{scope}:{stage}:{field}"
                        )
            for field, expected_value in expected.items():
                scope_totals[field] += expected_value
            total_background_failures += expected["background_failure_count"]
            total_failed_closed += expected["tidal_failed_closed_count"]
            total_unavailable += expected["tidal_unavailable_count"]
        totals[scope] = scope_totals

    expected_publication = (
        "partial_background_failures"
        if total_background_failures
        else "partial_tidal_validation"
        if total_failed_closed or total_unavailable
        else "complete_background_and_tidal"
    )
    if isinstance(metadata, dict):
        metadata_summary = metadata.get("stellar_status_summary")
        if not isinstance(metadata_summary, dict):
            layer.fail("metadata:stellar_status_summary_missing")
        else:
            if (
                metadata_summary.get("publication_interpretation_status")
                != expected_publication
            ):
                layer.fail(
                    "metadata:stellar_publication_interpretation_status_mismatch"
                )
            if metadata_summary.get("totals_by_scope") != totals:
                layer.fail("metadata:stellar_totals_by_scope_mismatch")
    layer.checks["recomputed_stellar_status_totals"] = totals


def _validate_response_population_reporting(
    packet: Path,
    configuration: Mapping[str, Any],
    metadata: Any,
    layer: _Layer,
) -> None:
    import pandas as pd

    fixed_rows = _read_csv(packet, "fixed_mass_observables.csv", layer)
    inventory_rows = _read_csv(packet, "plot_inventory.csv", layer)
    if fixed_rows is None or inventory_rows is None:
        return
    stages = [
        str(item["name"])
        for item in configuration.get("tov_stages", [])
        if isinstance(item, dict) and item.get("name")
    ]
    masses = [
        value
        for item in configuration.get("fixed_masses_msun", [])
        if (value := _float_or_none(item)) is not None
    ]
    if not stages or not masses:
        return
    final_stage = stages[-1]
    target_mass = min(masses, key=lambda value: abs(value - 1.4))
    fixed = pd.DataFrame(fixed_rows)
    inventory = {
        str(row.get("figure", "")): row for row in inventory_rows
    }
    metadata_rows = {}
    if isinstance(metadata, dict):
        value = metadata.get("plot_tidal_completeness", [])
        if isinstance(value, list):
            metadata_rows = {
                str(item.get("figure", "")): item
                for item in value
                if isinstance(item, dict)
            }
    checks: dict[str, Any] = {}
    for versus, figure in (
        ("amplitude", "observable_response_vs_amplitude.png"),
        ("delta", "observable_response_vs_delta.png"),
    ):
        _, population = summarize_fixed_mass_response_population(
            fixed,
            final_stage=final_stage,
            target_mass_msun=target_mass,
            versus=versus,
        )
        checks[versus] = population
        stored = inventory.get(figure)
        if stored is None:
            layer.fail(f"plot_inventory:response_figure_missing:{figure}")
            continue
        expected_count = int(population["eligible_deformation_row_count"])
        if expected_count:
            comparisons = {
                "eligible_response_row_count": expected_count,
                "tidal_validated_count": int(
                    population["tidal_validated_count"]
                ),
                "tidal_omitted_count": int(population["tidal_omitted_count"]),
            }
            for field, expected in comparisons.items():
                if _integer_or_none(stored.get(field)) != expected:
                    layer.fail(
                        f"plot_inventory:response_population_mismatch:{figure}:{field}"
                    )
            if stored.get("population_stage") != final_stage:
                layer.fail(
                    f"plot_inventory:response_population_mismatch:{figure}:population_stage"
                )
            stored_mass = _float_or_none(
                stored.get("population_target_mass_msun")
            )
            if stored_mass is None or stored_mass != target_mass:
                layer.fail(
                    f"plot_inventory:response_population_mismatch:{figure}:target_mass"
                )
            metadata_record = metadata_rows.get(figure)
            if metadata_record is None:
                layer.fail(
                    f"metadata:plot_tidal_completeness_missing:{figure}"
                )
            else:
                for field, expected in comparisons.items():
                    if _integer_or_none(metadata_record.get(field)) != expected:
                        layer.fail(
                            f"metadata:response_population_mismatch:{figure}:{field}"
                        )
                if metadata_record.get("population_stage") != final_stage:
                    layer.fail(
                        f"metadata:response_population_mismatch:{figure}:population_stage"
                    )
                metadata_mass = _float_or_none(
                    metadata_record.get("population_target_mass_msun")
                )
                if metadata_mass is None or metadata_mass != target_mass:
                    layer.fail(
                        f"metadata:response_population_mismatch:{figure}:target_mass"
                    )
        elif stored.get("status") != "skipped":
            layer.fail(
                f"plot_inventory:inapplicable_response_not_skipped:{figure}"
            )
    layer.checks["response_populations"] = checks

_MAXIMUM_MODEL_FIELDS = (
    "central_pressure_mev_fm3",
    "mass_msun",
    "radius_km",
    "central_energy_density_mev_fm3",
    "central_sound_speed_squared",
)
_MAXIMUM_BRACKET_FIELDS = (
    "lower_pressure_mev_fm3",
    "middle_pressure_mev_fm3",
    "upper_pressure_mev_fm3",
    "lower_mass_msun",
    "middle_mass_msun",
    "upper_mass_msun",
    "left_dM_dPc_secant",
    "right_dM_dPc_secant",
)


def _maximum_model_tuple(
    value: Any,
    *,
    eos_endpoint: float,
) -> tuple[float, float, float, float, float] | None:
    if not isinstance(value, Mapping):
        return None
    resolved = tuple(_float_or_none(value.get(field)) for field in _MAXIMUM_MODEL_FIELDS)
    if any(item is None for item in resolved):
        return None
    pressure, mass, radius, energy, sound_speed = (
        float(item) for item in resolved if item is not None
    )
    if (
        not 0.0 < pressure <= eos_endpoint
        or mass <= 0.0
        or radius <= 0.0
        or energy <= 0.0
        or not 0.0 < sound_speed <= 1.0
    ):
        return None
    return pressure, mass, radius, energy, sound_speed


def _maximum_bracket_tuple(
    value: Any,
    *,
    eos_endpoint: float,
) -> tuple[float, float, float, float, float, float, float, float] | None:
    if not isinstance(value, Mapping):
        return None
    resolved = tuple(
        _float_or_none(value.get(field)) for field in _MAXIMUM_BRACKET_FIELDS
    )
    if any(item is None for item in resolved):
        return None
    bracket = tuple(float(item) for item in resolved if item is not None)
    lower, middle, upper, lower_mass, middle_mass, upper_mass, left, right = bracket
    if (
        not 0.0 < lower < middle < upper <= eos_endpoint
        or lower_mass <= 0.0
        or middle_mass <= 0.0
        or upper_mass <= 0.0
    ):
        return None
    expected_left = (middle_mass - lower_mass) / (middle - lower)
    expected_right = (upper_mass - middle_mass) / (upper - middle)
    tolerance = 128.0 * math.ulp(1.0)
    if (
        not math.isclose(
            left,
            expected_left,
            rel_tol=tolerance,
            abs_tol=tolerance * max(1.0, abs(expected_left)),
        )
        or not math.isclose(
            right,
            expected_right,
            rel_tol=tolerance,
            abs_tol=tolerance * max(1.0, abs(expected_right)),
        )
    ):
        return None
    return bracket


def _optional_evidence_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_maximum_mass_artifacts(
    packet: Path,
    *,
    configuration: Mapping[str, Any],
    accepted: set[str],
    layer: _Layer,
    availability: _Layer,
    retained_pressure_endpoints: Mapping[str, float] | None = None,
    expected_case_ids: set[str] | None = None,
) -> None:
    """Validate one complete set of saved maximum-mass evidence."""

    for relative in (
        "maximum_mass_screening.csv",
        "maximum_mass_reports.json",
    ):
        if not (packet / relative).is_file():
            layer.fail(f"missing_stellar_output:{relative}")
    maximum_rows = _read_csv(packet, "maximum_mass_screening.csv", layer)
    reports = _load_json(packet, "maximum_mass_reports.json", layer)
    stage_names = tuple(
        str(stage.get("name", ""))
        for stage in configuration.get("tov_stages", ())
        if isinstance(stage, dict) and stage.get("name")
    )
    expected_pairs = {
        (case_id, stage)
        for case_id in (
            {"direct", *accepted}
            if expected_case_ids is None
            else expected_case_ids
        )
        for stage in stage_names
    }
    if retained_pressure_endpoints is not None:
        expected_case_ids = {case_id for case_id, _stage in expected_pairs}
        missing_endpoints = expected_case_ids - set(retained_pressure_endpoints)
        if missing_endpoints:
            layer.fail(
                "maximum_mass:retained_endpoints_missing:"
                f"{sorted(missing_endpoints)}"
            )
    observed_pairs: set[tuple[str, str]] = set()
    duplicate_pairs: set[tuple[str, str]] = set()
    rows_by_report_id: dict[str, Mapping[str, Any]] = {}
    unresolved_count = 0
    unresolved_statuses: dict[str, int] = {}
    if maximum_rows is not None:
        for row in maximum_rows:
            pair = (
                str(row.get("case_id", "")),
                str(row.get("stage", "")),
            )
            if pair in observed_pairs:
                duplicate_pairs.add(pair)
            observed_pairs.add(pair)
            rows_by_report_id[f"{pair[0]}:{pair[1]}"] = row
            status = str(row.get("status") or "").strip()
            availability_status = str(
                row.get("maximum_mass_availability_status") or ""
            ).strip()
            resolved = _boolean_or_none(row.get("maximum_mass_resolved"))
            if resolved is None:
                layer.fail(
                    "maximum_mass:invalid_resolution_boolean:"
                    f"{pair[0]}:{pair[1]}"
                )
                continue
            mass = _float_or_none(row.get("maximum_mass_msun"))
            pressure = _float_or_none(row.get("central_pressure_mev_fm3"))
            energy = _float_or_none(
                row.get("central_energy_density_mev_fm3")
            )
            sound_speed = _float_or_none(
                row.get("central_sound_speed_squared")
            )
            radius = _float_or_none(row.get("radius_km"))
            threshold = _float_or_none(
                row.get("maximum_mass_threshold_msun")
            )
            threshold_claimed = _has_saved_value(
                row.get("passes_maximum_mass_threshold")
            )
            threshold_result = _boolean_or_none(
                row.get("passes_maximum_mass_threshold")
            )
            left = _float_or_none(row.get("positive_left_secant"))
            right = _float_or_none(row.get("negative_right_secant"))
            endpoint = _float_or_none(
                row.get("eos_endpoint_pressure_mev_fm3")
            )
            expected_endpoint = (
                retained_pressure_endpoints.get(pair[0])
                if retained_pressure_endpoints is not None
                else None
            )
            if threshold is None or threshold <= 0.0:
                layer.fail(
                    "maximum_mass:invalid_threshold:"
                    f"{pair[0]}:{pair[1]}"
                )
            if threshold_claimed and threshold_result is None:
                layer.fail(
                    "maximum_mass:invalid_threshold_boolean:"
                    f"{pair[0]}:{pair[1]}"
                )
            if endpoint is None or endpoint <= 0.0:
                layer.fail(
                    "maximum_mass:invalid_eos_endpoint:"
                    f"{pair[0]}:{pair[1]}"
                )
            elif (
                retained_pressure_endpoints is not None
                and endpoint != expected_endpoint
            ):
                layer.fail(
                    "maximum_mass:retained_endpoint_mismatch:"
                    f"{pair[0]}:{pair[1]}"
                )
            if resolved and (
                mass is None
                or mass <= 0.0
                or pressure is None
                or pressure <= 0.0
                or energy is None
                or energy <= 0.0
                or sound_speed is None
                or not 0.0 < sound_speed <= 1.0
                or radius is None
                or radius <= 0.0
                or endpoint is None
                or endpoint <= 0.0
                or pressure > endpoint
                or left is None
                or right is None
                or not left > 0.0
                or not right < 0.0
                or threshold_result is None
                or not status.startswith("resolved_")
                or availability_status != "resolved_bracketed_and_refined"
            ):
                layer.fail(
                    "maximum_mass:resolved_without_turning_point_evidence:"
                    f"{pair[0]}:{pair[1]}"
                )
            if not resolved:
                expected_availability = f"unavailable_{status}"
                contaminated = [
                    column
                    for column in (
                        "maximum_mass_msun",
                        "central_pressure_mev_fm3",
                        "central_energy_density_mev_fm3",
                        "central_sound_speed_squared",
                        "radius_km",
                        "positive_left_secant",
                        "negative_right_secant",
                    )
                    if _has_saved_value(row.get(column))
                ]
                unresolved_row_valid = bool(
                    status.startswith("unresolved_")
                    and availability_status == expected_availability
                    and not contaminated
                    and not threshold_claimed
                    and str(row.get("endpoint_limitation") or "").strip()
                    and str(row.get("refinement_status") or "").strip()
                    and endpoint is not None
                    and endpoint > 0.0
                )
                if not unresolved_row_valid:
                    layer.fail(
                        "maximum_mass:malformed_unresolved_row:"
                        f"{pair[0]}:{pair[1]}:status={status}:"
                        f"availability={availability_status}:"
                        f"claims={','.join(contaminated)}"
                    )
                else:
                    unresolved_count += 1
                    unresolved_statuses[status] = (
                        unresolved_statuses.get(status, 0) + 1
                    )
            turning_points = _integer_or_none(
                row.get("turning_point_count")
            )
            if (
                turning_points is None
                or turning_points < 0
                or (resolved and turning_points < 1)
            ):
                layer.fail(
                    "maximum_mass:invalid_turning_point_count:"
                    f"{pair[0]}:{pair[1]}"
                )
            maximum_mass_tidal_calls = _integer_or_none(
                row.get("tidal_solver_calls_for_maximum_mass")
            )
            if maximum_mass_tidal_calls != 0:
                layer.fail(
                    "maximum_mass:refinement_used_tidal_solver:"
                    f"{pair[0]}:{pair[1]}"
                )
        if duplicate_pairs:
            layer.fail(
                "maximum_mass:duplicate_case_stage_rows:"
                f"{sorted(duplicate_pairs)}"
            )
        if observed_pairs != expected_pairs:
            layer.fail(
                "maximum_mass:case_stage_coverage_mismatch:"
                f"missing={sorted(expected_pairs - observed_pairs)}:"
                f"extra={sorted(observed_pairs - expected_pairs)}"
            )
    if unresolved_count:
        availability.fail(
            f"maximum_mass:unavailable:{unresolved_count}"
        )
    availability.checks["maximum_mass_unavailable_status_counts"] = dict(
        sorted(unresolved_statuses.items())
    )
    if isinstance(reports, dict):
        expected_schema = (
            "eos_generation_cfl_maximum_mass_reports_v1"
            if configuration.get("matter_model", "bsk24") == "cfl"
            else "bsk24_maximum_mass_reports_v2"
        )
        if reports.get("schema_id") != expected_schema:
            layer.fail("maximum_mass:unsupported_report_schema")
        report_cases = reports.get("cases")
        expected_report_ids = {
            f"{case_id}:{stage}" for case_id, stage in expected_pairs
        }
        if not isinstance(report_cases, dict) or set(report_cases) != expected_report_ids:
            layer.fail("maximum_mass:report_coverage_mismatch")
        elif isinstance(report_cases, dict):
            for report_id, maximum_report in report_cases.items():
                if not isinstance(maximum_report, dict):
                    layer.fail(
                        f"maximum_mass:malformed_report:{report_id}"
                    )
                    continue
                row = rows_by_report_id.get(report_id)
                if row is None:
                    continue
                problems: list[str] = []

                required_fields = {
                    "schema_id",
                    "status",
                    "maximum_mass_resolved",
                    "decision_basis",
                    "sampled_argmax_is_maximum_mass",
                    "maximum_mass_threshold_msun",
                    "passes_maximum_mass_threshold",
                    "maximum_mass_msun",
                    "central_pressure_mev_fm3",
                    "central_energy_density_mev_fm3",
                    "central_sound_speed_squared",
                    "radius_km",
                    "turning_point_count",
                    "turning_point_brackets",
                    "selected_bracket",
                    "positive_left_secant",
                    "negative_right_secant",
                    "stable_branch_extent",
                    "sampled_models",
                    "eos_endpoint",
                    "convergence",
                    "tidal_calculations_performed",
                }
                missing = sorted(required_fields - set(maximum_report))
                if missing:
                    problems.append(f"missing_fields={missing}")

                report_resolved = maximum_report.get(
                    "maximum_mass_resolved"
                )
                row_resolved = _boolean_or_none(
                    row.get("maximum_mass_resolved")
                )
                if (
                    maximum_report.get("schema_id")
                    != "tov_resolved_maximum_mass_v2"
                    or maximum_report.get("status") != row.get("status")
                    or not isinstance(report_resolved, bool)
                    or report_resolved != row_resolved
                    or maximum_report.get("sampled_argmax_is_maximum_mass")
                    is not False
                ):
                    problems.append("identity_or_resolution")

                report_threshold = maximum_report.get(
                    "passes_maximum_mass_threshold"
                )
                row_threshold = _boolean_or_none(
                    row.get("passes_maximum_mass_threshold")
                )
                if report_threshold is not row_threshold:
                    problems.append("threshold_decision")

                for field in (
                    "maximum_mass_threshold_msun",
                    "maximum_mass_msun",
                    "central_pressure_mev_fm3",
                    "central_energy_density_mev_fm3",
                    "central_sound_speed_squared",
                    "radius_km",
                    "positive_left_secant",
                    "negative_right_secant",
                ):
                    if _float_or_none(maximum_report.get(field)) != _float_or_none(
                        row.get(field)
                    ):
                        problems.append(f"csv_json_{field}")

                endpoint = _float_or_none(
                    row.get("eos_endpoint_pressure_mev_fm3")
                )
                endpoint_evidence = maximum_report.get("eos_endpoint")
                endpoint_reached: bool | None = None
                if not isinstance(endpoint_evidence, Mapping):
                    problems.append("eos_endpoint_object")
                else:
                    endpoint_reached_value = endpoint_evidence.get(
                        "reached_by_search"
                    )
                    if isinstance(endpoint_reached_value, bool):
                        endpoint_reached = endpoint_reached_value
                    else:
                        problems.append("endpoint_reached_boolean")
                    if (
                        _float_or_none(
                            endpoint_evidence.get("pressure_mev_fm3")
                        )
                        != endpoint
                        or _optional_evidence_text(
                            endpoint_evidence.get("limitation")
                        )
                        != _optional_evidence_text(
                            row.get("endpoint_limitation")
                        )
                    ):
                        problems.append("eos_endpoint_csv_json")

                brackets: list[tuple[float, ...]] = []
                bracket_values = maximum_report.get(
                    "turning_point_brackets"
                )
                if endpoint is None or endpoint <= 0.0 or not isinstance(
                    bracket_values, list
                ):
                    problems.append("turning_point_brackets")
                else:
                    for value in bracket_values:
                        parsed = _maximum_bracket_tuple(
                            value,
                            eos_endpoint=endpoint,
                        )
                        if parsed is None:
                            problems.append("invalid_turning_point_bracket")
                            break
                        brackets.append(parsed)
                report_turning_count = _integer_or_none(
                    maximum_report.get("turning_point_count")
                )
                row_turning_count = _integer_or_none(
                    row.get("turning_point_count")
                )
                if (
                    report_turning_count is None
                    or report_turning_count != row_turning_count
                    or report_turning_count != len(brackets)
                ):
                    problems.append("turning_point_count")

                selected_value = maximum_report.get("selected_bracket")
                selected: tuple[float, ...] | None = None
                if selected_value is not None:
                    if endpoint is None or endpoint <= 0.0:
                        problems.append("selected_bracket_without_endpoint")
                    else:
                        selected = _maximum_bracket_tuple(
                            selected_value,
                            eos_endpoint=endpoint,
                        )
                        if selected is None or selected not in brackets:
                            problems.append("selected_bracket")

                sampled: list[tuple[float, ...]] = []
                sampled_values = maximum_report.get("sampled_models")
                if endpoint is None or endpoint <= 0.0 or not isinstance(
                    sampled_values, list
                ):
                    problems.append("sampled_models")
                else:
                    for value in sampled_values:
                        parsed = _maximum_model_tuple(
                            value,
                            eos_endpoint=endpoint,
                        )
                        if parsed is None:
                            problems.append("invalid_sampled_model")
                            break
                        sampled.append(parsed)
                if any(
                    right[0] <= left[0]
                    for left, right in zip(sampled[:-1], sampled[1:])
                ):
                    problems.append("sampled_model_ordering")
                sampled_sequence_count = _integer_or_none(
                    row.get("sampled_sequence_model_count")
                )
                if (
                    sampled_sequence_count is None
                    or sampled_sequence_count < 0
                    or sampled_sequence_count > len(sampled)
                ):
                    problems.append("sampled_sequence_model_count")
                if endpoint_reached is not None and endpoint is not None:
                    sampled_reaches_endpoint = any(
                        model[0] == endpoint for model in sampled
                    )
                    if endpoint_reached != sampled_reaches_endpoint:
                        problems.append("endpoint_reach_model_consistency")

                stable: list[tuple[float, ...]] = []
                stable_extent = maximum_report.get(
                    "stable_branch_extent"
                )
                if not isinstance(stable_extent, Mapping):
                    problems.append("stable_branch_extent")
                else:
                    stable_values = stable_extent.get("models")
                    if endpoint is None or endpoint <= 0.0 or not isinstance(
                        stable_values, list
                    ):
                        problems.append("stable_branch_models")
                    else:
                        for value in stable_values:
                            parsed = _maximum_model_tuple(
                                value,
                                eos_endpoint=endpoint,
                            )
                            if parsed is None:
                                problems.append("invalid_stable_model")
                                break
                            stable.append(parsed)
                    if any(
                        right[0] <= left[0]
                        for left, right in zip(stable[:-1], stable[1:])
                    ):
                        problems.append("stable_model_ordering")
                    stable_count = _integer_or_none(
                        stable_extent.get("model_count")
                    )
                    stable_maximum_pressure = _float_or_none(
                        stable_extent.get(
                            "maximum_central_pressure_mev_fm3"
                        )
                    )
                    expected_stable_maximum = stable[-1][0] if stable else None
                    if (
                        stable_count != len(stable)
                        or stable_maximum_pressure
                        != expected_stable_maximum
                    ):
                        problems.append("stable_branch_summary")

                convergence = maximum_report.get("convergence")
                if not isinstance(convergence, Mapping):
                    problems.append("convergence_object")
                else:
                    refinement_iterations = _integer_or_none(
                        convergence.get("refinement_iterations")
                    )
                    global_rounds = _integer_or_none(
                        convergence.get("global_refinement_rounds")
                    )
                    solver_calls = _integer_or_none(
                        convergence.get("solver_call_count")
                    )
                    solver_failure_count = _integer_or_none(
                        convergence.get("solver_failure_count")
                    )
                    solver_failures = convergence.get("solver_failures")
                    if (
                        convergence.get("refinement_status")
                        != row.get("refinement_status")
                        or refinement_iterations is None
                        or refinement_iterations < 0
                        or global_rounds is None
                        or global_rounds < 0
                        or solver_calls is None
                        or solver_calls < 0
                        or solver_calls
                        != _integer_or_none(
                            row.get("local_background_solver_call_count")
                        )
                        or solver_failure_count is None
                        or solver_failure_count < 0
                        or not isinstance(solver_failures, list)
                        or solver_failure_count
                        != (
                            len(solver_failures)
                            if isinstance(solver_failures, list)
                            else -1
                        )
                    ):
                        problems.append("convergence_summary")
                    if isinstance(solver_failures, list):
                        failure_pressures: list[float] = []
                        for failure in solver_failures:
                            if not isinstance(failure, Mapping):
                                problems.append("solver_failure_record")
                                break
                            failure_pressure = _float_or_none(
                                failure.get("central_pressure_mev_fm3")
                            )
                            if (
                                endpoint is None
                                or failure_pressure is None
                                or not 0.0 < failure_pressure <= endpoint
                                or not _optional_evidence_text(
                                    failure.get("reason")
                                )
                            ):
                                problems.append("solver_failure_record")
                                break
                            failure_pressures.append(failure_pressure)
                        if len(set(failure_pressures)) != len(failure_pressures):
                            problems.append("duplicate_solver_failure_pressure")

                report_tidal_calls = _integer_or_none(
                    maximum_report.get("tidal_calculations_performed")
                )
                if report_tidal_calls != 0:
                    problems.append("tidal_solver_use")

                report_mass = _float_or_none(
                    maximum_report.get("maximum_mass_msun")
                )
                report_pressure = _float_or_none(
                    maximum_report.get("central_pressure_mev_fm3")
                )
                report_radius = _float_or_none(
                    maximum_report.get("radius_km")
                )
                report_energy = _float_or_none(
                    maximum_report.get(
                        "central_energy_density_mev_fm3"
                    )
                )
                report_sound_speed = _float_or_none(
                    maximum_report.get("central_sound_speed_squared")
                )
                left_secant = _float_or_none(
                    maximum_report.get("positive_left_secant")
                )
                right_secant = _float_or_none(
                    maximum_report.get("negative_right_secant")
                )
                report_threshold_msun = _float_or_none(
                    maximum_report.get("maximum_mass_threshold_msun")
                )
                maximum_model = (
                    report_pressure,
                    report_mass,
                    report_radius,
                    report_energy,
                    report_sound_speed,
                )
                resolved_secants_match = False
                if (
                    selected is not None
                    and report_pressure is not None
                    and report_mass is not None
                    and left_secant is not None
                    and right_secant is not None
                    and selected[0] < report_pressure < selected[2]
                ):
                    expected_left_secant = (
                        report_mass - selected[3]
                    ) / (report_pressure - selected[0])
                    expected_right_secant = (
                        selected[5] - report_mass
                    ) / (selected[2] - report_pressure)
                    secant_tolerance = 128.0 * math.ulp(1.0)
                    resolved_secants_match = bool(
                        math.isclose(
                            left_secant,
                            expected_left_secant,
                            rel_tol=secant_tolerance,
                            abs_tol=secant_tolerance
                            * max(1.0, abs(expected_left_secant)),
                        )
                        and math.isclose(
                            right_secant,
                            expected_right_secant,
                            rel_tol=secant_tolerance,
                            abs_tol=secant_tolerance
                            * max(1.0, abs(expected_right_secant)),
                        )
                    )
                if report_resolved is True:
                    if (
                        maximum_report.get("decision_basis")
                        != "refined_positive_to_negative_dM_dPc_turning_point"
                        or report_mass is None
                        or report_pressure is None
                        or report_radius is None
                        or report_energy is None
                        or report_sound_speed is None
                        or report_threshold_msun is None
                        or report_threshold
                        is not (report_mass >= report_threshold_msun)
                        or len(brackets) != 1
                        or selected != brackets[0]
                        or selected[6] <= 0.0
                        or selected[7] >= 0.0
                        or not selected[0] < report_pressure < selected[2]
                        or left_secant is None
                        or left_secant <= 0.0
                        or right_secant is None
                        or right_secant >= 0.0
                        or not resolved_secants_match
                        or _optional_evidence_text(
                            maximum_report.get("eos_endpoint", {}).get(
                                "limitation"
                            )
                            if isinstance(
                                maximum_report.get("eos_endpoint"), Mapping
                            )
                            else None
                        )
                        is not None
                        or not stable
                        or stable[-1] != maximum_model
                        or any(model[1] > report_mass for model in stable)
                        or any(
                            model not in sampled and model != maximum_model
                            for model in stable
                        )
                    ):
                        problems.append("resolved_scientific_evidence")
                elif report_resolved is False:
                    if (
                        maximum_report.get("decision_basis")
                        != "fail_closed_no_resolved_turning_point"
                        or not str(
                            maximum_report.get("status") or ""
                        ).startswith("unresolved_")
                        or report_threshold is not None
                        or any(
                            value is not None
                            for value in maximum_model
                        )
                        or left_secant is not None
                        or right_secant is not None
                        or not _optional_evidence_text(
                            endpoint_evidence.get("limitation")
                            if isinstance(endpoint_evidence, Mapping)
                            else None
                        )
                    ):
                        problems.append("unresolved_scientific_evidence")

                if problems:
                    layer.fail(
                        "maximum_mass:malformed_json_evidence:"
                        f"{report_id}:{','.join(sorted(set(problems)))}"
                    )


def _ordered_case_blocks(
    rows: list[dict[str, str]],
    *,
    relative: str,
    layer: _Layer,
) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    """Return contiguous case blocks while recording malformed ordering."""

    order: list[str] = []
    blocks: dict[str, list[dict[str, str]]] = {}
    current: str | None = None
    closed: set[str] = set()
    for position, row in enumerate(rows):
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            layer.fail(f"thermodynamic_output:missing_case_id:{relative}:{position}")
            continue
        if case_id != current:
            if current is not None:
                closed.add(current)
            if case_id in closed:
                layer.fail(
                    f"thermodynamic_output:noncontiguous_case_block:"
                    f"{relative}:{case_id}"
                )
            if case_id not in blocks:
                order.append(case_id)
            current = case_id
        blocks.setdefault(case_id, []).append(row)
    return order, blocks


def _strictly_increasing(values: list[float]) -> bool:
    return bool(values) and all(
        right > left for left, right in zip(values[:-1], values[1:])
    )


def _validate_cfl_retained_pressure_endpoints(
    packet: Path,
    *,
    expected_stellar_case_ids: set[str],
    layer: _Layer,
) -> dict[str, tuple[float, float]]:
    """Validate CFL saved endpoints without importing BSk24 anchor semantics."""

    rows = _read_csv(packet, "thermodynamic_profiles.csv", layer)
    if rows is None:
        layer.fail("missing_thermodynamic_output:thermodynamic_profiles.csv")
        return {}
    _order, blocks = _ordered_case_blocks(
        rows,
        relative="thermodynamic_profiles.csv",
        layer=layer,
    )
    endpoints: dict[str, tuple[float, float]] = {}
    for case_id in sorted(expected_stellar_case_ids):
        case_rows = blocks.get(case_id)
        if not case_rows:
            layer.fail(f"cfl_thermodynamic_endpoint:profile_missing:{case_id}")
            continue
        epsilon = [_float_or_none(row.get("epsilon_mev_fm3")) for row in case_rows]
        pressure = [_float_or_none(row.get("pressure_mev_fm3")) for row in case_rows]
        if any(value is None for value in (*epsilon, *pressure)):
            layer.fail(f"cfl_thermodynamic_endpoint:nonfinite:{case_id}")
            continue
        epsilon_values = [float(value) for value in epsilon if value is not None]
        pressure_values = [float(value) for value in pressure if value is not None]
        if (
            not epsilon_values
            or not pressure_values
            or epsilon_values[0] <= 0.0
            or pressure_values[0] != 0.0
            or not _strictly_increasing(epsilon_values)
            or not _strictly_increasing(pressure_values)
            or pressure_values[-1] <= 0.0
        ):
            layer.fail(f"cfl_thermodynamic_endpoint:invalid_profile:{case_id}")
            continue
        endpoints[case_id] = (epsilon_values[-1], pressure_values[-1])
    unexpected = set(endpoints) - expected_stellar_case_ids
    if unexpected:
        layer.fail(
            "cfl_thermodynamic_endpoint:unexpected_stellar_cases:"
            f"{sorted(unexpected)}"
        )
    return endpoints


def _validate_thermodynamic_outputs(
    packet: Path,
    *,
    accepted: set[str],
    direct_expected: bool = True,
    curve_only: bool = False,
    layer: _Layer,
) -> dict[str, tuple[float, float]]:
    """Validate saved reconstructed state without judging finite diagnostics.

    Core state, interpolation coordinates, and inversion outputs must remain
    finite and physically ordered.  Finite values of auxiliary diagnostics and
    residuals are retained as evidence but are not compared with a magnitude,
    sign, or trend threshold here.
    """

    expected_raw_lower: float | None = None
    expected_direct_pressure: float | None = None
    raw_gate = _load_json(packet, "raw_gate_report.json", layer)
    raw_cases = raw_gate.get("cases") if isinstance(raw_gate, dict) else None
    if isinstance(raw_cases, dict) and raw_cases:
        raw_domains: set[tuple[float, float]] = set()
        zero_amplitude_endpoints: set[tuple[float, float, float]] = set()
        for report in raw_cases.values():
            if not isinstance(report, Mapping):
                continue
            domain = report.get("complete_proposed_retained_domain_mev_fm3")
            if isinstance(domain, list) and len(domain) == 2:
                lower = _float_or_none(domain[0])
                upper = _float_or_none(domain[1])
                if (
                    lower is not None
                    and upper is not None
                    and 0.0 < lower < upper
                ):
                    raw_domains.add((lower, upper))
            parameters = report.get("parameters")
            retained = report.get("retained_domain")
            amplitude = (
                _float_or_none(parameters.get("amplitude"))
                if isinstance(parameters, Mapping)
                else None
            )
            if amplitude == 0.0 and isinstance(retained, Mapping):
                epsilon_min = _float_or_none(
                    retained.get("epsilon_min_mev_fm3")
                )
                epsilon_max = _float_or_none(
                    retained.get("epsilon_max_mev_fm3")
                )
                pressure_max = _float_or_none(
                    retained.get("pressure_max_mev_fm3")
                )
                if (
                    report.get("status") == "accepted_raw_local_physics_gate"
                    and retained.get("endpoint_reason")
                    == "direct_bsk24_causal_endpoint"
                    and epsilon_min is not None
                    and epsilon_max is not None
                    and pressure_max is not None
                ):
                    zero_amplitude_endpoints.add(
                        (epsilon_min, epsilon_max, pressure_max)
                    )
        raw_lowers = {domain[0] for domain in raw_domains}
        if len(raw_lowers) == 1:
            expected_raw_lower = next(iter(raw_lowers))
        else:
            layer.fail("thermodynamic_output:raw_domain_lower_not_authoritative")
        if len(zero_amplitude_endpoints) == 1:
            zero_endpoint = next(iter(zero_amplitude_endpoints))
            if expected_raw_lower == zero_endpoint[0]:
                expected_direct_pressure = zero_endpoint[2]
            else:
                layer.fail(
                    "thermodynamic_output:a0_direct_lower_endpoint_mismatch"
                )
        elif direct_expected:
            layer.fail("thermodynamic_output:a0_direct_endpoint_not_authoritative")
    else:
        layer.fail("thermodynamic_output:raw_gate_cases_unavailable")

    ledger_endpoints: dict[str, tuple[float, float]] = {}
    ledger_rows = _read_csv(packet, "case_ledger.csv", layer)
    if ledger_rows is not None:
        for case_id in accepted:
            matches = [
                row
                for row in ledger_rows
                if str(
                    row.get("physical_case_id")
                    or row.get("case_id")
                    or ""
                )
                == case_id
            ]
            epsilon_endpoint = (
                _float_or_none(matches[0].get("retained_epsilon_max_mev_fm3"))
                if len(matches) == 1
                else None
            )
            pressure_endpoint = (
                _float_or_none(matches[0].get("retained_pressure_max_mev_fm3"))
                if len(matches) == 1
                else None
            )
            if (
                epsilon_endpoint is None
                or epsilon_endpoint <= 0.0
                or pressure_endpoint is None
                or pressure_endpoint <= 0.0
            ):
                layer.fail(
                    "thermodynamic_output:invalid_ledger_retained_endpoint:"
                    f"{case_id}"
                )
            else:
                ledger_endpoints[case_id] = (
                    epsilon_endpoint,
                    pressure_endpoint,
                )

    profile_relative = "thermodynamic_profiles.csv"
    profile_rows = _read_csv(packet, profile_relative, layer)
    profile_blocks: dict[str, list[dict[str, str]]] = {}
    profile_endpoints: dict[str, tuple[float, float]] = {}
    if profile_rows is None:
        layer.fail(f"missing_thermodynamic_output:{profile_relative}")
    else:
        profile_order, profile_blocks = _ordered_case_blocks(
            profile_rows,
            relative=profile_relative,
            layer=layer,
        )
        expected = set(accepted)
        if direct_expected:
            expected.add("direct")
        present = set(profile_blocks)
        if present != expected:
            layer.fail(
                "thermodynamic_output:profile_case_set_mismatch:"
                f"missing={sorted(expected - present)}:"
                f"unexpected={sorted(present - expected)}"
            )
        if direct_expected and profile_order and profile_order[0] != "direct":
            layer.fail("thermodynamic_output:direct_baseline_not_first")

        for case_id, case_rows in profile_blocks.items():
            epsilon: list[float] = []
            pressure: list[float] = []
            density: list[float] = []
            for position, row in enumerate(case_rows):
                required_profile_columns = (
                    ("epsilon_mev_fm3", "pressure_mev_fm3", "cs2")
                    if curve_only
                    else _THERMODYNAMIC_PROFILE_REQUIRED_NUMERIC_COLUMNS
                )
                missing_columns = sorted(
                    set(required_profile_columns)
                    - set(row)
                )
                invalid_columns = [
                    column
                    for column in required_profile_columns
                    if column in row and _float_or_none(row.get(column)) is None
                ]
                if case_id != "direct":
                    for column in ("amplitude", "delta_mev_fm3"):
                        if column not in row:
                            missing_columns.append(column)
                        elif _float_or_none(row.get(column)) is None:
                            invalid_columns.append(column)
                if missing_columns or invalid_columns:
                    layer.fail(
                        "thermodynamic_output:nonfinite_or_missing_profile_value:"
                        f"{case_id}:{position}:"
                        f"missing={sorted(set(missing_columns))}:"
                        f"invalid={sorted(set(invalid_columns))}"
                    )
                    continue

                epsilon_value = float(row["epsilon_mev_fm3"])
                pressure_value = float(row["pressure_mev_fm3"])
                cs2_value = float(row["cs2"])
                density_value = (
                    None
                    if curve_only
                    else float(row["baryon_density_fm3"])
                )
                enthalpy_value = (
                    None
                    if curve_only
                    else float(row["effective_baryon_enthalpy_mev"])
                )
                if (
                    epsilon_value <= 0.0
                    or pressure_value <= 0.0
                    or not 0.0 < cs2_value <= 1.0
                    or (density_value is not None and density_value <= 0.0)
                    or (enthalpy_value is not None and enthalpy_value <= 0.0)
                ):
                    layer.fail(
                        "thermodynamic_output:invalid_profile_core_state:"
                        f"{case_id}:{position}"
                    )
                epsilon.append(epsilon_value)
                pressure.append(pressure_value)
                if density_value is not None:
                    density.append(density_value)

            if len(epsilon) != len(case_rows):
                continue
            coordinates = [
                ("epsilon_mev_fm3", epsilon),
                ("pressure_mev_fm3", pressure),
            ]
            if not curve_only:
                coordinates.append(("baryon_density_fm3", density))
            for coordinate, values in coordinates:
                if not _strictly_increasing(values):
                    layer.fail(
                        "thermodynamic_output:nonmonotone_profile_coordinate:"
                        f"{case_id}:{coordinate}"
                    )
            if expected_raw_lower is not None and epsilon[0] != expected_raw_lower:
                layer.fail(
                    "thermodynamic_output:profile_lower_endpoint_mismatch:"
                    f"{case_id}"
                )
            if case_id == "direct" and zero_amplitude_endpoints:
                zero_endpoint = next(iter(zero_amplitude_endpoints))
                if (
                    epsilon[-1] != zero_endpoint[1]
                    or pressure[-1] != expected_direct_pressure
                ):
                    layer.fail(
                        "thermodynamic_output:direct_retained_endpoint_mismatch"
                    )
            profile_endpoints[case_id] = (epsilon[-1], pressure[-1])

        for case_id in accepted:
            ledger_endpoint = ledger_endpoints.get(case_id)
            profile_endpoint = profile_endpoints.get(case_id)
            if (
                ledger_endpoint is None
                or profile_endpoint is None
                or profile_endpoint != ledger_endpoint
            ):
                layer.fail(
                    "thermodynamic_output:profile_retained_endpoint_mismatch:"
                    f"{case_id}"
                )

    residual_relative = "thermodynamic_residuals.csv"
    residual_rows = (
        None
        if curve_only
        else _read_csv(packet, residual_relative, layer)
    )
    if not curve_only and residual_rows is None:
        if accepted:
            layer.fail(f"missing_thermodynamic_output:{residual_relative}")
    elif residual_rows is not None:
        _, residual_blocks = _ordered_case_blocks(
            residual_rows,
            relative=residual_relative,
            layer=layer,
        )
        if set(residual_blocks) != accepted:
            layer.fail(
                "thermodynamic_output:residual_case_set_mismatch:"
                f"missing={sorted(accepted - set(residual_blocks))}:"
                f"unexpected={sorted(set(residual_blocks) - accepted)}"
            )
        for case_id, case_rows in residual_blocks.items():
            residual_epsilon: list[float] = []
            for position, row in enumerate(case_rows):
                missing_columns = sorted(
                    set(_THERMODYNAMIC_RESIDUAL_REQUIRED_NUMERIC_COLUMNS)
                    - set(row)
                )
                invalid_columns = [
                    column
                    for column in _THERMODYNAMIC_RESIDUAL_REQUIRED_NUMERIC_COLUMNS
                    if column in row and _float_or_none(row.get(column)) is None
                ]
                if missing_columns or invalid_columns:
                    layer.fail(
                        "thermodynamic_output:nonfinite_or_missing_residual_value:"
                        f"{case_id}:{position}:"
                        f"missing={missing_columns}:invalid={invalid_columns}"
                    )
                    continue
                residual_epsilon.append(float(row["epsilon_mev_fm3"]))
            if len(residual_epsilon) != len(case_rows):
                continue
            if not _strictly_increasing(residual_epsilon):
                layer.fail(
                    "thermodynamic_output:nonmonotone_residual_coordinate:"
                    f"{case_id}"
                )
            profile_epsilon = [
                _float_or_none(row.get("epsilon_mev_fm3"))
                for row in profile_blocks.get(case_id, [])
            ]
            if profile_epsilon != residual_epsilon:
                layer.fail(
                    "thermodynamic_output:profile_residual_grid_mismatch:"
                    f"{case_id}"
                )
    retained_endpoints = {
        case_id: endpoint
        for case_id, endpoint in profile_endpoints.items()
        if case_id == "direct" or ledger_endpoints.get(case_id) == endpoint
    }
    layer.checks["thermodynamic_profile_case_count"] = len(profile_blocks)
    layer.checks["retained_endpoint_case_count"] = len(retained_endpoints)
    return retained_endpoints


def _validate_scientific_completeness(
    packet: Path,
    *,
    configuration: Any,
    metadata: Any,
    accepted: set[str],
    case_ledger: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    layer = _Layer()
    availability = _Layer()
    if not isinstance(configuration, dict):
        layer.fail("configuration_unavailable")
    else:
        accepted_output_case_ids = _accepted_physical_output_case_ids(
            configuration,
            accepted,
            case_ledger,
            layer,
        )
        expected_stellar_case_ids = _expected_saved_stellar_case_ids(
            configuration,
            accepted,
            case_ledger,
            layer,
            accepted_physical_case_ids=accepted_output_case_ids,
        )
        stellar_enabled = configuration.get("stellar_enabled") is True
        curve_only = configuration.get("curve_only_output") is True
        background_requested = _configuration_background_tov_requested(
            configuration
        )
        fixed_mass_requested = _configuration_fixed_mass_requested(
            configuration
        )
        maximum_mass_requested = _configuration_maximum_mass_requested(
            configuration
        )
        tidal_requested = _configuration_tidal_requested(configuration)
        extended_claimed = _extended_output_claimed(configuration, metadata)
        layer.checks["stellar_enabled"] = stellar_enabled
        layer.checks["background_tov_requested"] = background_requested
        layer.checks["extended_outputs_claimed"] = extended_claimed
        retained_endpoints = (
            _validate_cfl_retained_pressure_endpoints(
                packet,
                expected_stellar_case_ids=expected_stellar_case_ids,
                layer=layer,
            )
            if configuration.get("matter_model", "bsk24") == "cfl"
            else _validate_thermodynamic_outputs(
                packet,
                accepted=accepted_output_case_ids,
                direct_expected=_cfl_direct_baseline_expected(
                    configuration
                ),
                curve_only=curve_only,
                layer=layer,
            )
        )
        retained_pressure_endpoints = {
            case_id: endpoint[1]
            for case_id, endpoint in retained_endpoints.items()
        }
        if curve_only:
            for relative in (
                "thermodynamic_residuals.csv",
                "window_characterization.csv",
                "fixed_mass_observables.csv",
                "maximum_mass_screening.csv",
                "maximum_mass_reports.json",
                "stellar_status_summary.csv",
                "stellar_status_summary.json",
                "radial_profiles.csv",
            ):
                if (packet / relative).exists():
                    layer.fail(f"curve_only_output:unexpected_output:{relative}")
        if background_requested:
            required_stellar_files = (
                "stellar_sequences.csv",
                "stellar_convergence.json",
                *(
                    (
                        "fixed_mass_observables.csv",
                        *CURRENT_STELLAR_STATUS_FILES,
                    )
                    if fixed_mass_requested
                    else ()
                ),
                *(
                    (
                        "maximum_mass_screening.csv",
                        "maximum_mass_reports.json",
                    )
                    if maximum_mass_requested
                    else ()
                ),
            )
            for relative in required_stellar_files:
                if not (packet / relative).is_file():
                    layer.fail(f"missing_stellar_output:{relative}")

            if maximum_mass_requested:
                _validate_maximum_mass_artifacts(
                    packet,
                    configuration=configuration,
                    accepted=accepted,
                    layer=layer,
                    availability=availability,
                    retained_pressure_endpoints=retained_pressure_endpoints,
                    expected_case_ids=expected_stellar_case_ids,
                )
            if fixed_mass_requested:
                summary_csv = _read_csv(
                    packet, "stellar_status_summary.csv", layer
                )
                summary_json = _load_json(
                    packet, "stellar_status_summary.json", layer
                )
                if isinstance(summary_json, dict) and summary_json.get(
                    "schema_id"
                ) != STELLAR_STATUS_SCHEMA:
                    layer.fail("stellar_status_summary:unsupported_schema")
                if summary_csv is not None and isinstance(summary_json, dict):
                    summary_rows = _inventory_records(summary_json.get("rows"))
                    if summary_rows is None:
                        layer.fail("stellar_status_summary:json_rows_invalid")
                    elif not _csv_json_records_equal(summary_csv, summary_rows):
                        layer.fail("stellar_status_summary:csv_json_rows_mismatch")

            if all(
                (packet / relative).is_file()
                for relative in required_stellar_files
            ):
                if fixed_mass_requested:
                    _validate_fixed_mass_completeness(
                        packet,
                        configuration,
                        accepted,
                        layer,
                        availability=availability,
                        require_tides=tidal_requested,
                        retained_pressure_endpoints=retained_pressure_endpoints,
                        expected_case_ids=expected_stellar_case_ids,
                    )
                _validate_sequence_completeness(
                    packet,
                    configuration,
                    accepted,
                    layer,
                    availability=availability,
                    require_tides=tidal_requested,
                    require_configured_count=True,
                    retained_pressure_endpoints=retained_pressure_endpoints,
                    expected_case_ids=expected_stellar_case_ids,
                )
                if fixed_mass_requested:
                    _validate_stellar_status_reporting(
                        packet, configuration, metadata, layer
                    )
                    _validate_response_population_reporting(
                        packet, configuration, metadata, layer
                    )
            _validate_final_lifecycle(
                packet,
                configuration,
                accepted,
                layer,
                expected_stellar_case_ids=expected_stellar_case_ids,
            )
        else:
            _validate_final_lifecycle(
                packet,
                configuration,
                accepted,
                layer,
                expected_stellar_case_ids=expected_stellar_case_ids,
            )
        if extended_claimed:
            for relative in EXTENDED_CORE_REQUIRED_FILES:
                if not (packet / relative).is_file():
                    layer.fail(f"missing_claimed_extended_output:{relative}")

        thermodynamic_outputs = (
            ("thermodynamic_profiles.csv", True),
            ("thermodynamic_residuals.csv", False),
            ("window_characterization.csv", False),
        )
        for relative, direct_expected in thermodynamic_outputs:
            if curve_only and relative != "thermodynamic_profiles.csv":
                if (packet / relative).exists():
                    layer.fail(
                        f"curve_only_output:unexpected_thermodynamic_output:{relative}"
                    )
                continue
            if (
                relative == "thermodynamic_residuals.csv"
                and accepted
                and not (packet / relative).is_file()
            ):
                layer.fail(f"missing_thermodynamic_output:{relative}")
                continue
            rows = _read_csv(packet, relative, layer)
            if rows is None:
                continue
            present = _case_ids(rows)
            missing_accepted = accepted_output_case_ids - present
            if missing_accepted:
                layer.fail(
                    "thermodynamic_output:missing_accepted_cases:"
                    f"{relative}:"
                    f"{sorted(missing_accepted)}"
                )
            if _a0_alias_mode(configuration):
                duplicated_logical_aliases = (
                    accepted - accepted_output_case_ids
                ) & present
                if duplicated_logical_aliases:
                    layer.fail(
                        "thermodynamic_output:duplicated_logical_aliases:"
                        f"{relative}:{sorted(duplicated_logical_aliases)}"
                    )
            require_direct = (
                direct_expected
                and _cfl_direct_baseline_expected(configuration)
            )
            if require_direct and "direct" not in present:
                layer.fail(
                    f"thermodynamic_output:direct_baseline_missing:{relative}"
                )
            if (
                _a0_alias_mode(configuration)
                and not _cfl_direct_baseline_expected(configuration)
                and "direct" in present
            ):
                layer.fail(
                    f"thermodynamic_output:unexpected_nonowner_direct:{relative}"
                )

    identity = _load_json(packet, "identity_report.json", layer)
    identity_rows = _read_csv(packet, "a0_identity_table.csv", layer)
    if isinstance(configuration, dict):
        _validate_cfl_a0_identity(
            configuration,
            identity,
            identity_rows,
            layer,
        )
        _validate_bsk24_a0_identity(
            configuration,
            identity,
            identity_rows,
            layer,
        )
    if isinstance(identity, dict) and identity.get("status") != "pass":
        layer.fail(f"identity_not_pass:{identity.get('status')}")

    reproduction = _load_json(packet, "reproduction.json", layer)
    if isinstance(reproduction, dict):
        if "packet_interpreter_path" in reproduction:
            layer.fail("reproduction:absolute_interpreter_path_must_not_be_recorded")
        command = str(reproduction.get("command", "")).lstrip()
        if command.startswith("py "):
            layer.warn("reproduction:ambient_py_interpreter_is_ambiguous")

    environment = _load_json(packet, "environment.json", layer)
    if isinstance(environment, dict):
        forbidden_environment_fields = {
            "interpreter_path",
            "conda_prefix",
            "conda_executable",
        }
        leaked = sorted(forbidden_environment_fields.intersection(environment))
        if leaked:
            layer.fail(
                "environment:absolute_machine_paths_must_not_be_recorded:"
                + ",".join(leaked)
            )

    validity = {
        "status": "pass" if not layer.failures else "fail",
        "failures": layer.failures,
        "warnings": layer.warnings,
        "checks": layer.checks,
    }
    availability_report = {
        "status": "complete" if not availability.failures else "partial",
        "limitations": availability.failures,
        "warnings": availability.warnings,
        "checks": availability.checks,
    }
    status = (
        "invalid"
        if layer.failures
        else availability_report["status"]
    )
    return {
        "status": status,
        "failures": layer.failures,
        "limitations": availability.failures,
        "warnings": [*layer.warnings, *availability.warnings],
        "checks": layer.checks,
        "hard_validity": validity,
        "availability": availability_report,
    }

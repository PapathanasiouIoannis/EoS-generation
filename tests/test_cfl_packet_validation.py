from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eos_generation._internal.planning import (
    BSk24ThermodynamicStage,
    BSk24TrialConfig,
)
from eos_generation._internal.summary import (
    CFL_PACKET_SCHEMA_ID,
    PACKET_SCHEMA_ID,
)
from eos_generation.cfl.planning import CFLTrialConfig, prepare_cfl_trial
from eos_generation.reporting._validation_integrity import (
    _normalize_cfl_raw_gate_identities,
    _validate_anchor_selection,
    _validate_cfl_case_aliases,
    _validate_cfl_identity_metadata,
    _validate_cfl_plan_aliases,
    _validate_packet_schema_and_summary,
)
from eos_generation.reporting._validation_cases import (
    _validate_cfl_raw_gate_profiles,
)
from eos_generation.reporting._validation_io import (
    _Layer,
    _default_configuration_hash,
)
from eos_generation.reporting._validation_scientific import (
    _cfl_direct_baseline_expected,
    _expected_saved_stellar_case_ids,
    _validate_bsk24_a0_identity,
    _validate_cfl_a0_identity,
    _validate_cfl_tidal_jump_row,
)
from eos_generation.reporting.validation import (
    validate_bsk24_trial_packet_layers,
    validate_cfl_trial_packet_layers,
    validate_trial_packet_layers,
)
from eos_generation.stellar.tov import (
    EOS_DISCONTINUITY_CONTRACT_VERSION,
    LAMBDA_FRAMEWORK_CAPABILITY,
    TIDAL_CORRECTION_VERSION,
    TIDAL_JUMP_FORMULA,
)


def _cfl_config(*, owner: bool = True) -> CFLTrialConfig:
    return CFLTrialConfig(
        amplitudes=(0.0, 0.01),
        epsilon0_mev_fm3=600.0,
        sigma_mev_fm3=100.0,
        deltas_mev_fm3=(40.0,),
        zero_amplitude_control_owner=owner,
        thermodynamic_stages=(
            BSk24ThermodynamicStage("synthetic", 17, 17),
        ),
        stellar_enabled=False,
    )


def _anchor_metadata(configuration: dict[str, object]) -> dict[str, object]:
    profile = configuration["baseline_profile"]
    assert isinstance(profile, dict)
    surface = profile["surface"]
    assert isinstance(surface, dict)
    return {
        "matter_model": "cfl",
        "frozen_cfl_parameters": deepcopy(profile),
        "baseline_parameter_set_id": configuration[
            "baseline_parameter_set_id"
        ],
        "baseline_parameter_set_sha256": configuration[
            "baseline_parameter_set_sha256"
        ],
        "formulation_id": configuration["formulation_id"],
        "formulation_version": configuration["formulation_version"],
        "deformation_profile_id": configuration[
            "deformation_profile_id"
        ],
        "deformation_profile_version": configuration[
            "deformation_profile_version"
        ],
        "reconstruction_profile_id": configuration[
            "reconstruction_profile_id"
        ],
        "reconstruction_schema_version": configuration[
            "reconstruction_schema_version"
        ],
        "pressure_primitive_policy": configuration[
            "pressure_primitive_policy"
        ],
        "stellar_sequence_policy": configuration["stellar_sequence_policy"],
        "stellar_local_refinement_policy": configuration["stellar_local_refinement_policy"],
        "domain_id": configuration["domain_id"],
        "surface_tidal_policy": {
            "finite_surface_energy_density": True,
            "required_correction": TIDAL_JUMP_FORMULA,
            "correction_version": TIDAL_CORRECTION_VERSION,
            "application_count": "exactly_once_per_successful_tidal_star",
            "saved_evidence": (
                "stellar_sequences.csv and fixed_mass_observables.csv"
            ),
        },
        "anchor_selection": {
            "mode": "bare_self_bound_zero_pressure_surface",
            "exploratory": False,
            "selected_epsilon_match_mev_fm3": surface[
                "energy_density_mev_fm3"
            ],
            "derived_state": {
                field: surface[field]
                for field in (
                    "energy_density_mev_fm3",
                    "pressure_mev_fm3",
                    "baryon_density_fm3",
                    "quark_chemical_potential_mev",
                    "baryon_chemical_potential_mev",
                    "sound_speed_squared",
                )
            },
            "window_and_reconstruction_share_this_anchor": True,
            "surface_pressure_preserved_exactly": True,
            "surface_baryon_density_preserved_exactly": True,
            "surface_baryon_chemical_potential_preserved_exactly": True,
            "surface_exterior": "vacuum",
            "crust_or_hadronic_envelope": "absent",
        },
    }


def _jump_row(surface_energy_density: float) -> dict[str, object]:
    y_before = 0.8
    delta_y = -0.3
    y_after = y_before + delta_y
    k2 = 0.09
    lambda_value = 450.0
    jump = {
        "identifier": "cfl_bare_self_bound_surface_v1",
        "type": "surface",
        "pressure_MeV_fm3": 0.0,
        "radius_km": 11.2,
        "mass_Msun": 1.4,
        "inner_energy_density_MeV_fm3": surface_energy_density,
        "outer_energy_density_MeV_fm3": 0.0,
        "signed_outward_delta_energy_density_MeV_fm3": (
            surface_energy_density
        ),
        "correction_denominator_Msun": 1.4,
        "y_before": y_before,
        "delta_y": delta_y,
        "y_after": y_after,
        "provenance": "synthetic saved-evidence contract",
    }
    payload = {
        "schema_version": "tov_lambda_diagnostic_v1",
        "central_pressure_MeV_fm3": 100.0,
        "Mass": 1.4,
        "Radius": 11.2,
        "Compactness": 0.18,
        "expected_jump_count": 1,
        "applied_jump_count": 1,
        "applied_jumps": [jump],
        "skipped_discontinuity_ids": [],
        "surface_event_pressure_MeV_fm3": 0.0,
        "y_surface_interior": y_before,
        "y_surface_vacuum": y_after,
        "y_supplied_to_k2": y_after,
        "y_R": y_after,
        "k2": k2,
        "Lambda": lambda_value,
        "correction_formula": TIDAL_JUMP_FORMULA,
        "correction_version": TIDAL_CORRECTION_VERSION,
        "correction_status": "validated_conditional_per_calculation",
        "correction_sources": [],
        "discontinuity_contract_version": (
            EOS_DISCONTINUITY_CONTRACT_VERSION
        ),
        "framework_lambda_capability": LAMBDA_FRAMEWORK_CAPABILITY,
        "calculation_lambda_validated": True,
        "scientific_status": LAMBDA_FRAMEWORK_CAPABILITY,
        "failure_reason": None,
    }
    return {
        "case_id": "direct",
        "stage": "synthetic",
        "attempted_index": "0",
        "central_pressure_mev_fm3": "100.0",
        "Mass": "1.4",
        "Radius": "11.2",
        "tidal_status": LAMBDA_FRAMEWORK_CAPABILITY,
        "k2": str(k2),
        "Lambda": str(lambda_value),
        "tidal_expected_jump_count": "1",
        "tidal_applied_jump_count": "1",
        "tidal_surface_jump_count": "1",
        "tidal_surface_delta_y": str(delta_y),
        "tidal_surface_y_before": str(y_before),
        "tidal_surface_y_after": str(y_after),
        "tidal_surface_event_pressure_mev_fm3": "0.0",
        "tidal_jump_evidence_json": json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }


class CFLPacketValidationTests(unittest.TestCase):
    def test_bsk24_a0_owner_and_nonowner_identity_contracts(self) -> None:
        owner_configuration = BSk24TrialConfig(
            amplitudes=(0.0, 0.1),
            deltas_mev_fm3=(40.0,),
            zero_amplitude_control_owner=True,
            thermodynamic_stages=(
                BSk24ThermodynamicStage("synthetic", 17, 17),
            ),
        ).to_dict()
        baseline_id = owner_configuration[
            "zero_amplitude_physical_case_id"
        ]
        owner_identity = {
            "schema_id": "eos_generation_bsk24_a0_identity_v2",
            "zero_amplitude_control_owner": True,
            "zero_amplitude_physical_case_id": baseline_id,
            "duplicate_zero_amplitude_stellar_solver_calls": 0,
            "local_thermodynamic_identity": {
                "status": "pass",
                "deltas": {"40.0": {"pressure_mev_fm3": {}}},
            },
            "stellar_identity_status": "not_requested",
            "status": "pass",
        }
        rows = [
            {
                "scope": "thermodynamic",
                "stage": "refined",
                "delta_mev_fm3": "40.0",
                "quantity": "pressure_mev_fm3",
                "maximum_absolute_residual": "0.0",
                "array_equal": "True",
                "status": "pass",
            }
        ]
        owner_layer = _Layer()
        _validate_bsk24_a0_identity(
            owner_configuration, owner_identity, rows, owner_layer
        )
        self.assertEqual([], owner_layer.failures)

        nonowner_configuration = BSk24TrialConfig(
            amplitudes=(0.0, 0.1),
            deltas_mev_fm3=(40.0,),
            zero_amplitude_control_owner=False,
            thermodynamic_stages=(
                BSk24ThermodynamicStage("synthetic", 17, 17),
            ),
        ).to_dict()
        nonowner_identity = {
            "schema_id": "eos_generation_bsk24_a0_identity_v2",
            "zero_amplitude_control_owner": False,
            "zero_amplitude_physical_case_id": nonowner_configuration[
                "zero_amplitude_physical_case_id"
            ],
            "duplicate_zero_amplitude_stellar_solver_calls": 0,
            "local_thermodynamic_identity": {
                "status": "not_applicable_no_owned_a0_case"
            },
            "stellar_identity_status": "not_requested",
            "status": "pass",
        }
        nonowner_layer = _Layer()
        _validate_bsk24_a0_identity(
            nonowner_configuration, nonowner_identity, [], nonowner_layer
        )
        self.assertEqual([], nonowner_layer.failures)

    def test_cfl_raw_profiles_use_physical_ids_and_complete_domain(self) -> None:
        physical_id = "cfl-physical"
        baseline_hash = "a" * 64
        report = {
            "parameters": {
                "amplitude": 0.02,
                "center_mev_fm3": 500.0,
                "width_mev_fm3": 100.0,
                "ramp_width_mev_fm3": 40.0,
            },
            "complete_declared_domain_mev_fm3": [300.0, 900.0],
            "dense_grid_points": 2,
            "status": "accepted_raw_local_physics_gate",
            "finite_values": True,
            "baseline_parameter_set_sha256": baseline_hash,
        }
        rows = [
            {
                "case_id": physical_id,
                "physical_case_id": physical_id,
                "matter_model": "cfl",
                "baseline_parameter_set_sha256": baseline_hash,
                "amplitude": "0.02",
                "epsilon0_mev_fm3": "500.0",
                "sigma_mev_fm3": "100.0",
                "delta_mev_fm3": "40.0",
                "epsilon_mev_fm3": str(epsilon),
                "window": str(window),
                "gaussian": "0.5",
                "delta_cs2": str(0.02 * (0.5 * window)),
                "raw_cs2": "0.35",
                "gate_status": "accepted_raw_local_physics_gate",
            }
            for epsilon, window in ((300.0, 0.0), (900.0, 1.0))
        ]
        layer = _Layer()
        _validate_cfl_raw_gate_profiles(
            rows,
            raw_gate={"cases": {physical_id: report}},
            accepted={physical_id},
            rejected=set(),
            layer=layer,
        )
        self.assertEqual([], layer.failures)

        corrupted = deepcopy(rows)
        corrupted[-1]["raw_cs2"] = "1.0000001"
        failed = _Layer()
        _validate_cfl_raw_gate_profiles(
            corrupted,
            raw_gate={"cases": {physical_id: report}},
            accepted={physical_id},
            rejected=set(),
            layer=failed,
        )
        self.assertIn(
            "cfl_raw_gate_profiles:accepted_cs2_invalid:cfl-physical:1",
            failed.failures,
        )

    def test_cfl_a0_owner_requires_exact_identity_rows(self) -> None:
        configuration = _cfl_config().to_dict()
        baseline_id = configuration["zero_amplitude_physical_case_id"]
        identity = {
            "schema_id": "eos_generation_cfl_a0_identity_v1",
            "zero_amplitude_control_owner": True,
            "physical_zero_case_ids": [baseline_id],
            "floating_point_policy": "numpy.array_equal_binary64",
            "stellar_identity_status": "not_requested",
            "status": "pass",
        }
        rows = [
            {
                "scope": "thermodynamic",
                "stage": "reference",
                "quantity": quantity,
                "maximum_absolute_residual": "0.0",
                "array_equal": "True",
                "status": "pass",
            }
            for quantity in (
                "epsilon",
                "pressure",
                "cs2",
                "baryon_density",
                "baryon_chemical_potential",
            )
        ]
        layer = _Layer()
        _validate_cfl_a0_identity(configuration, identity, rows, layer)
        self.assertEqual([], layer.failures)

        missing_layer = _Layer()
        missing = dict(identity, physical_zero_case_ids=[])
        _validate_cfl_a0_identity(
            configuration,
            missing,
            [],
            missing_layer,
        )
        self.assertIn(
            "cfl_a0_identity:owned_physical_case_mismatch",
            missing_layer.failures,
        )
        self.assertIn(
            "cfl_a0_identity:table_quantity_coverage_invalid",
            missing_layer.failures,
        )

    def test_configuration_hash_dispatch_preserves_bsk_and_accepts_cfl(self) -> None:
        bsk = BSk24TrialConfig(stellar_enabled=False)
        self.assertEqual(
            bsk.deterministic_hash(),
            _default_configuration_hash(bsk.to_dict()),
        )
        cfl = _cfl_config()
        self.assertEqual(
            cfl.deterministic_hash(),
            _default_configuration_hash(cfl.to_dict()),
        )

    def test_both_packet_schemas_are_accepted_for_their_matter_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="packet-schema-") as temporary:
            packet = Path(temporary)
            (packet / "summary.md").write_text("canonical\n", encoding="utf-8")
            with (
                patch(
                    "eos_generation.reporting._validation_integrity."
                    "build_summary_model",
                    return_value={},
                ),
                patch(
                    "eos_generation.reporting._validation_integrity."
                    "render_summary_markdown",
                    return_value="canonical\n",
                ),
            ):
                for matter_model, schema in (
                    ("bsk24", PACKET_SCHEMA_ID),
                    ("cfl", CFL_PACKET_SCHEMA_ID),
                ):
                    layer = _Layer()
                    observed = _validate_packet_schema_and_summary(
                        packet,
                        {"schema_id": schema},
                        layer,
                        matter_model=matter_model,
                    )
                    self.assertEqual(schema, observed)
                    self.assertEqual([], layer.failures)

    def test_cfl_anchor_requires_the_hashed_finite_density_zero_pressure_state(
        self,
    ) -> None:
        configuration = _cfl_config().to_dict()
        metadata = _anchor_metadata(configuration)
        layer = _Layer()
        _validate_anchor_selection(configuration, metadata, layer)
        self.assertEqual([], layer.failures)

        corrupted = deepcopy(metadata)
        anchor = corrupted["anchor_selection"]
        assert isinstance(anchor, dict)
        derived = anchor["derived_state"]
        assert isinstance(derived, dict)
        derived["pressure_mev_fm3"] = 1.0e-9
        failed = _Layer()
        _validate_anchor_selection(configuration, corrupted, failed)
        self.assertIn(
            "anchor_selection:derived_state_mismatch:pressure_mev_fm3",
            failed.failures,
        )

    def test_aliases_deduplicate_a0_to_owner_direct_stellar_case(self) -> None:
        owner = _cfl_config(owner=True).to_dict()
        baseline_id = str(owner["zero_amplitude_physical_case_id"])
        ledger = [
            {
                "case_id": "logical-a0",
                "physical_case_id": baseline_id,
                "status": "accepted",
            },
            {
                "case_id": "physical-nonzero",
                "physical_case_id": "physical-nonzero",
                "status": "accepted",
            },
        ]
        layer = _Layer()
        observed = _expected_saved_stellar_case_ids(
            owner,
            {"logical-a0", "physical-nonzero"},
            ledger,
            layer,
        )
        self.assertEqual({"direct", "physical-nonzero"}, observed)
        self.assertTrue(_cfl_direct_baseline_expected(owner))
        self.assertEqual([], layer.failures)

        nonowner = _cfl_config(owner=False).to_dict()
        nonowner_layer = _Layer()
        observed_nonowner = _expected_saved_stellar_case_ids(
            nonowner,
            {"physical-nonzero"},
            [ledger[1]],
            nonowner_layer,
        )
        self.assertEqual({"physical-nonzero"}, observed_nonowner)
        self.assertFalse(_cfl_direct_baseline_expected(nonowner))
        self.assertEqual([], nonowner_layer.failures)

        bsk_owner = BSk24TrialConfig(
            amplitudes=(0.0, 0.01),
            deltas_mev_fm3=(40.0,),
            zero_amplitude_control_owner=True,
            thermodynamic_stages=(
                BSk24ThermodynamicStage("synthetic", 17, 17),
            ),
        ).to_dict()
        bsk_baseline_id = str(
            bsk_owner["zero_amplitude_physical_case_id"]
        )
        bsk_ledger = [
            {
                "case_id": "bsk-logical-a0",
                "physical_case_id": bsk_baseline_id,
                "status": "accepted",
            },
            {
                "case_id": "bsk-nonzero",
                "physical_case_id": "bsk-nonzero",
                "status": "accepted",
            },
        ]
        bsk_layer = _Layer()
        bsk_observed = _expected_saved_stellar_case_ids(
            bsk_owner,
            {"bsk-logical-a0", "bsk-nonzero"},
            bsk_ledger,
            bsk_layer,
        )
        self.assertEqual({"direct", "bsk-nonzero"}, bsk_observed)
        self.assertEqual([], bsk_layer.failures)

    def test_cfl_case_plan_and_ledger_retain_the_same_physical_alias(self) -> None:
        configuration = _cfl_config().to_dict()
        baseline_id = str(configuration["zero_amplitude_physical_case_id"])
        case_plan = [
            {
                "case_id": "logical-a0",
                "physical_case_id": baseline_id,
                "is_physical_case_alias": "True",
                "amplitude": "0.0",
            }
        ]
        case_ledger = [
            {
                **case_plan[0],
                "status": "accepted",
            }
        ]
        layer = _Layer()
        _validate_cfl_case_aliases(
            configuration,
            case_plan,
            case_ledger,
            layer,
        )
        self.assertEqual([], layer.failures)

        case_ledger[0]["is_physical_case_alias"] = "False"
        failed = _Layer()
        _validate_cfl_case_aliases(
            configuration,
            case_plan,
            case_ledger,
            failed,
        )
        self.assertIn(
            "cfl_case_aliases:alias_flag_mismatch:logical-a0",
            failed.failures,
        )

    def test_nonowner_logical_a0_is_retained_only_as_a_plan_alias(self) -> None:
        config = _cfl_config(owner=False)
        plan = prepare_cfl_trial(config).to_dict()
        serialized_cases = plan["case_table"]
        assert isinstance(serialized_cases, list)
        case_plan = [
            {"case_id": str(item["case_id"])}
            for item in serialized_cases
        ]
        aliases = plan["logical_alias_table"]
        assert isinstance(aliases, list)
        saved_aliases = [
            {key: str(value) for key, value in item.items()}
            for item in aliases
        ]
        layer = _Layer()
        _validate_cfl_plan_aliases(
            config.to_dict(),
            plan,
            case_plan,
            saved_aliases,
            layer,
        )
        self.assertEqual([], layer.failures)
        self.assertEqual(1, layer.checks["cfl_logical_alias_count"])

        aliases[0]["planned_for_execution"] = True
        failed = _Layer()
        _validate_cfl_plan_aliases(
            config.to_dict(),
            plan,
            case_plan,
            saved_aliases,
            failed,
        )
        self.assertTrue(
            any("execution_flag_invalid" in item for item in failed.failures)
        )

    def test_physical_raw_gate_ids_normalize_to_logical_lifecycle_ids(self) -> None:
        config = _cfl_config(owner=True)
        configuration = config.to_dict()
        plan = prepare_cfl_trial(config)
        case_plan = [
            {key: str(value) for key, value in row.items()}
            for row in plan.case_table.to_dict(orient="records")
        ]
        zero = next(row for row in case_plan if float(row["amplitude"]) == 0.0)
        physical_id = zero["physical_case_id"]
        logical_id = zero["case_id"]
        report = {
            "schema_version": "cfl_raw_local_physics_gate_v1",
            "profile_id": configuration["deformation_profile_id"],
            "profile_version": configuration["deformation_profile_version"],
            "pressure_primitive_policy": configuration[
                "pressure_primitive_policy"
            ],
            "case_id": physical_id,
            "baseline_parameter_set_id": configuration[
                "baseline_parameter_set_id"
            ],
            "baseline_parameter_set_sha256": configuration[
                "baseline_parameter_set_sha256"
            ],
            "status": "accepted_raw_local_physics_gate",
        }
        report["report_sha256"] = hashlib.sha256(
            json.dumps(
                report,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        raw_gate = {
            "schema_id": "eos_generation_raw_gate_v1",
            "cases": {physical_id: report},
            "accepted_case_ids": [physical_id],
            "rejected_case_ids": [],
            "full_domain_accepted_case_ids": [physical_id],
            "full_domain_rejected_case_ids": [],
        }
        # Limit the synthetic plan to the one physical identity represented by
        # this raw-gate fragment.
        case_plan = [zero]
        metadata = {
            "accepted_physical_case_ids": [physical_id],
            "rejected_physical_case_ids": [],
            "accepted_physical_case_count": 1,
            "rejected_physical_case_count": 0,
        }
        layer = _Layer()
        normalized, accepted, rejected = _normalize_cfl_raw_gate_identities(
            configuration,
            raw_gate,
            case_plan,
            metadata,
            layer,
        )
        self.assertEqual([], layer.failures)
        self.assertEqual({physical_id}, accepted)
        self.assertEqual(set(), rejected)
        self.assertEqual({logical_id}, set(normalized["cases"]))
        self.assertEqual([logical_id], normalized["accepted_case_ids"])

    def test_alias_only_child_accepts_a_governed_empty_raw_gate(self) -> None:
        config = CFLTrialConfig(
            amplitudes=(0.0,),
            epsilon0_mev_fm3=600.0,
            sigma_mev_fm3=100.0,
            deltas_mev_fm3=(40.0,),
            zero_amplitude_control_owner=False,
            thermodynamic_stages=(
                BSk24ThermodynamicStage("synthetic", 17, 17),
            ),
            stellar_enabled=False,
        )
        plan = prepare_cfl_trial(config)
        self.assertTrue(plan.case_table.empty)
        self.assertEqual(1, len(plan.logical_alias_table))
        raw_gate = {
            "schema_id": "eos_generation_raw_gate_v1",
            "cases": {},
            "accepted_case_ids": [],
            "rejected_case_ids": [],
            "full_domain_accepted_case_ids": [],
            "full_domain_rejected_case_ids": [],
        }
        metadata = {
            "accepted_physical_case_ids": [],
            "rejected_physical_case_ids": [],
            "accepted_physical_case_count": 0,
            "rejected_physical_case_count": 0,
        }
        layer = _Layer()
        normalized, accepted, rejected = _normalize_cfl_raw_gate_identities(
            config.to_dict(),
            raw_gate,
            [],
            metadata,
            layer,
        )
        self.assertEqual([], layer.failures)
        self.assertEqual(raw_gate, normalized)
        self.assertEqual(set(), accepted)
        self.assertEqual(set(), rejected)

    def test_cfl_identity_metadata_cross_links_owned_baseline_outcome(self) -> None:
        configuration = _cfl_config(owner=True).to_dict()
        baseline_id = str(configuration["zero_amplitude_physical_case_id"])
        case_plan = [{"case_id": "logical-a0"}]
        metadata = {
            "zero_amplitude_physical_case_id": baseline_id,
            "logical_case_count": 1,
            "logical_alias_count": 0,
            "physical_case_count": 1,
        }
        accepted = _Layer()
        _validate_cfl_identity_metadata(
            configuration,
            metadata,
            case_plan,
            [],
            {baseline_id},
            set(),
            accepted,
        )
        self.assertEqual([], accepted.failures)

        rejected = _Layer()
        _validate_cfl_identity_metadata(
            configuration,
            metadata,
            case_plan,
            [],
            set(),
            {baseline_id},
            rejected,
        )
        self.assertIn(
            "metadata:cfl_owner_baseline_not_accepted",
            rejected.failures,
        )
        self.assertIn(
            "metadata:cfl_owner_baseline_rejected",
            rejected.failures,
        )

    def test_successful_cfl_tide_requires_one_negative_canonical_surface_jump(
        self,
    ) -> None:
        configuration = _cfl_config().to_dict()
        profile = configuration["baseline_profile"]
        assert isinstance(profile, dict)
        surface = profile["surface"]
        assert isinstance(surface, dict)
        epsilon = float(surface["energy_density_mev_fm3"])
        row = _jump_row(epsilon)
        layer = _Layer()
        _validate_cfl_tidal_jump_row(
            row,
            schema="sequence",
            surface_energy_density=epsilon,
            layer=layer,
            context="direct:synthetic:0",
        )
        self.assertEqual([], layer.failures)

        fixed_row = dict(row)
        fixed_row["mass_msun"] = fixed_row["Mass"]
        fixed_row["radius_km"] = fixed_row["Radius"]
        fixed_row["lambda_dimensionless"] = fixed_row["Lambda"]
        fixed_layer = _Layer()
        _validate_cfl_tidal_jump_row(
            fixed_row,
            schema="fixed_mass",
            surface_energy_density=epsilon,
            layer=fixed_layer,
            context="direct:synthetic:1.4",
        )
        self.assertEqual([], fixed_layer.failures)

        corrupted = dict(row)
        corrupted["tidal_applied_jump_count"] = "2"
        failed = _Layer()
        _validate_cfl_tidal_jump_row(
            corrupted,
            schema="sequence",
            surface_energy_density=epsilon,
            layer=failed,
            context="direct:synthetic:0",
        )
        self.assertTrue(
            any("tidal_applied_jump_count_not_one" in item for item in failed.failures)
        )

        positive = _jump_row(epsilon)
        evidence = json.loads(str(positive["tidal_jump_evidence_json"]))
        evidence["applied_jumps"][0]["delta_y"] = 0.3
        positive["tidal_surface_delta_y"] = "0.3"
        positive["tidal_jump_evidence_json"] = json.dumps(
            evidence,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        positive_layer = _Layer()
        _validate_cfl_tidal_jump_row(
            positive,
            schema="sequence",
            surface_energy_density=epsilon,
            layer=positive_layer,
            context="direct:synthetic:0",
        )
        self.assertTrue(
            any("delta_y_not_finite_negative" in item for item in positive_layer.failures)
        )

    def test_generic_and_model_specific_entry_points_remain_exposed(self) -> None:
        self.assertTrue(callable(validate_trial_packet_layers))
        self.assertTrue(callable(validate_cfl_trial_packet_layers))
        self.assertTrue(callable(validate_bsk24_trial_packet_layers))


if __name__ == "__main__":
    unittest.main()

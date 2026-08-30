from __future__ import annotations

import json
import math
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

import jsonschema

from eos_generation import ExperimentSettings, plan_experiment
from eos_generation.cfl.baseline import (
    CFL_DEFORMATION_PROFILE_ID,
    CFL_DOMAIN_ID,
    ENERGY_DENSITY_MAX_MEV_FM3,
    ENERGY_DENSITY_SURFACE_MEV_FM3,
    FROZEN_CFL_PARAMETERS,
    FROZEN_PARAMETER_SET_ID,
    FROZEN_PARAMETER_SET_SHA256,
)
from eos_generation.cfl.planning import (
    CFL_TRIAL_PLAN_SCHEMA,
    CFLTrialConfig,
    frozen_cfl_physical_baseline_id,
)


ROOT = Path(__file__).resolve().parents[1]


def _cfl_settings(**overrides: object) -> ExperimentSettings:
    values: dict[str, object] = {
        "matter_model": "cfl",
        "epsilon_match": "surface",
        "amplitudes": (0.0, 0.01),
        "center": 600.0,
        "width": 100.0,
        "ramp_width": 50.0,
        "calculation": "thermodynamics",
        "precision": "quick",
    }
    values.update(overrides)
    return ExperimentSettings.from_values(**values)


class CFLPublicSettingsTests(unittest.TestCase):
    def test_legacy_bsk24_serialization_and_settings_hash_are_exact(self) -> None:
        expected = {
            "amplitudes": [0.0, 0.01],
            "epsilon_match": "standard",
            "center": 200.0,
            "width": 50.0,
            "ramp_width": 40.0,
            "calculation": "thermodynamics",
            "precision": "quick",
            "fixed_masses": [1.4],
            "diagnostics": "off",
        }
        legacy = ExperimentSettings.from_json(ROOT / "configs" / "quickstart.json")
        explicit = ExperimentSettings.from_dict(
            {"matter_model": "bsk24", **expected}
        )
        self.assertEqual("bsk24", legacy.matter_model)
        self.assertEqual(expected, legacy.to_dict())
        self.assertEqual(expected, explicit.to_dict())
        self.assertNotIn("matter_model", legacy.to_dict())
        self.assertEqual(
            "f22216234e8554f2bbb2b72838eb58cd3fc13e3b17c66ca1a23dde540ec4736b",
            legacy.deterministic_hash(),
        )
        self.assertEqual(legacy.deterministic_hash(), explicit.deterministic_hash())

    def test_cfl_discriminator_and_surface_anchor_are_explicit(self) -> None:
        settings = _cfl_settings()
        self.assertEqual("cfl", settings.matter_model)
        self.assertEqual("surface", settings.epsilon_match)
        self.assertEqual("cfl", settings.to_dict()["matter_model"])
        self.assertEqual(settings, ExperimentSettings.from_dict(settings.to_dict()))

        with self.assertRaisesRegex(ValueError, "epsilon_match='surface'"):
            ExperimentSettings.from_values(matter_model="cfl")
        with self.assertRaisesRegex(ValueError, "epsilon_match='surface'"):
            ExperimentSettings.from_values(
                matter_model="cfl", epsilon_match="standard"
            )
        with self.assertRaisesRegex(ValueError, "matter_model"):
            ExperimentSettings.from_values(matter_model="hybrid")
        with self.assertRaisesRegex(ValueError, "extended diagnostics are unavailable"):
            _cfl_settings(calculation="stellar", diagnostics="on")
        dataset = _cfl_settings(
            calculation="stellar",
            precision="dataset_40",
            diagnostics="off",
        )
        self.assertEqual("dataset_40", dataset.precision)
        with self.assertRaisesRegex(ValueError, "other dataset profiles"):
            _cfl_settings(
                calculation="stellar",
                precision="dataset_20",
                diagnostics="off",
            )
        with self.assertRaises((TypeError, ValueError)):
            ExperimentSettings.from_values(epsilon_match="surface")

    def test_json_schema_dispatches_anchor_semantics_by_model(self) -> None:
        schema = json.loads(
            (ROOT / "configs" / "schema.json").read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
        legacy = json.loads(
            (ROOT / "configs" / "quickstart.json").read_text(encoding="utf-8")
        )
        validator.validate(legacy)
        validator.validate({**legacy, "matter_model": "bsk24"})
        cfl = {
            **legacy,
            "matter_model": "cfl",
            "epsilon_match": "surface",
            "center": 600.0,
            "width": 100.0,
            "ramp_width": 50.0,
        }
        validator.validate(cfl)
        validator.validate(
            {**cfl, "calculation": "stellar", "precision": "dataset_40"}
        )
        invalid = (
            {**cfl, "epsilon_match": "standard"},
            {**cfl, "epsilon_match": 500.0},
            {**legacy, "matter_model": "bsk24", "epsilon_match": "surface"},
            {**legacy, "epsilon_match": "surface"},
            {**cfl, "calculation": "stellar", "diagnostics": "on"},
            {**cfl, "calculation": "stellar", "precision": "dataset_20"},
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(
                jsonschema.ValidationError
            ):
                validator.validate(payload)


class CFLPassivePlanningTests(unittest.TestCase):
    def test_plan_is_passive_and_carries_the_complete_frozen_contract(self) -> None:
        settings = _cfl_settings()
        output_root = ROOT / "runs" / "test-cfl-passive-plan"
        self.assertFalse(output_root.exists())
        with patch(
            "eos_generation.cfl.baseline.brentq",
            side_effect=AssertionError("planning called a root solver"),
        ):
            first = plan_experiment(settings, output_root=output_root)
            second = plan_experiment(settings, output_root=output_root)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertFalse(output_root.exists())
        self.assertEqual(
            "b36098eda836c8af284e0848efa5cce587be2a9009f88a928c672f765bd87027",
            settings.deterministic_hash(),
        )
        self.assertEqual(1, len(first.child_plans))
        child = first.child_plans[0]
        self.assertEqual(
            # Explicit CFL selection and seed-preserving refinement policies
            # enter expanded numerical identity; public settings are intact.
            "1ff5f30434d46bad0a42e8e4efff65f7a208e1780677a9f20442c41c3f9f2092",
            child.config.deterministic_hash(),
        )
        document = child.to_dict()
        expanded = document["expanded_configuration"]
        self.assertEqual(CFL_TRIAL_PLAN_SCHEMA, document["schema_id"])
        self.assertTrue(document["planning_is_passive"])
        self.assertEqual(0, document["scientific_solver_calls"])
        self.assertEqual(0, document["filesystem_writes"])
        self.assertEqual("cfl", expanded["matter_model"])
        self.assertEqual("surface", expanded["epsilon_match"])
        self.assertEqual(
            ENERGY_DENSITY_SURFACE_MEV_FM3,
            expanded["epsilon_match_mev_fm3"],
        )
        self.assertEqual(
            [ENERGY_DENSITY_SURFACE_MEV_FM3, ENERGY_DENSITY_MAX_MEV_FM3],
            expanded["complete_domain_mev_fm3"],
        )
        self.assertEqual(FROZEN_PARAMETER_SET_ID, expanded["baseline_parameter_set_id"])
        self.assertEqual(
            FROZEN_PARAMETER_SET_SHA256,
            expanded["baseline_parameter_set_sha256"],
        )
        self.assertEqual(
            "3991cb8615d2d29617ccb90c6dc54b23aae64bcc752856d07f17f99abc048307",
            FROZEN_PARAMETER_SET_SHA256,
        )
        self.assertEqual(FROZEN_CFL_PARAMETERS.to_dict(), expanded["baseline_profile"])
        self.assertEqual(CFL_DOMAIN_ID, expanded["domain_id"])
        self.assertEqual(
            CFL_DEFORMATION_PROFILE_ID, expanded["deformation_profile_id"]
        )
        self.assertEqual(
            "cfl_surface_anchored_first_law_v1",
            expanded["reconstruction_profile_id"],
        )
        self.assertEqual(
            "cfl_windowed_eos_v1",
            expanded["reconstruction_schema_version"],
        )
        self.assertEqual(
            "normalized_segmented_gauss_legendre_64_with_stable_normal_tail_v1",
            expanded["pressure_primitive_policy"],
        )
        self.assertEqual(
            "seed_preserving_split_log_pressure_v1",
            expanded["stellar_local_refinement_policy"],
        )
        recovered = CFLTrialConfig.from_dict(child.config.to_dict())
        self.assertEqual(child.config, recovered)
        self.assertEqual(
            child.config.deterministic_hash(), recovered.deterministic_hash()
        )
        self.assertEqual(child.config, pickle.loads(pickle.dumps(child.config)))

    def test_logical_cartesian_zero_controls_share_one_physical_baseline(self) -> None:
        settings = _cfl_settings(
            center=(600.0, 800.0),
            ramp_width=(50.0, 100.0),
        )
        plan = plan_experiment(
            settings, output_root=ROOT / "runs" / "test-cfl-zero-dedup"
        )
        cases = plan.case_table
        self.assertEqual(4, len(plan.child_plans))
        self.assertEqual(8, len(cases))
        self.assertEqual(5, sum(len(child.case_table) for child in plan.child_plans))
        self.assertEqual(
            3, sum(len(child.logical_alias_table) for child in plan.child_plans)
        )
        zeros = cases[cases["amplitude"] == 0.0]
        nonzeros = cases[cases["amplitude"] != 0.0]
        self.assertEqual(4, len(zeros))
        self.assertEqual(4, zeros["case_id"].nunique())
        self.assertEqual(1, zeros["physical_case_id"].nunique())
        self.assertEqual(
            frozen_cfl_physical_baseline_id(), zeros.iloc[0]["physical_case_id"]
        )
        self.assertTrue(zeros["is_physical_case_alias"].all())
        self.assertEqual(1, int(zeros["planned_for_execution"].sum()))
        self.assertEqual(1, int(zeros["physical_case_owner"].sum()))
        self.assertTrue(
            (nonzeros["case_id"] == nonzeros["physical_case_id"]).all()
        )
        self.assertFalse(nonzeros["is_physical_case_alias"].any())
        self.assertTrue(nonzeros["planned_for_execution"].all())
        repeated = plan_experiment(
            settings, output_root=ROOT / "runs" / "test-cfl-zero-dedup"
        )
        self.assertEqual(
            [
                "cfl_dp50_a0_6f021e47cda9",
                "cfl_dp50_ap0p01_10cad7a186ad",
                "cfl_dp100_a0_fe21c55a5ac1",
                "cfl_dp100_ap0p01_e07d5eb24dd7",
                "cfl_dp50_a0_c71df554e391",
                "cfl_dp50_ap0p01_2b4aa0d8c4df",
                "cfl_dp100_a0_dfddee44865f",
                "cfl_dp100_ap0p01_198aca8eefcb",
            ],
            cases["case_id"].tolist(),
        )
        self.assertEqual(
            cases["case_id"].tolist(), repeated.case_table["case_id"].tolist()
        )
        self.assertEqual(8, plan.estimates["logical_deformation_cases"])
        self.assertEqual(5, plan.estimates["physical_deformation_cases"])
        self.assertEqual(5, plan.estimates["proposed_deformation_cases"])
        self.assertEqual(3, plan.estimates["deduplicated_logical_case_aliases"])
        self.assertEqual(1, plan.estimates["direct_baseline_cases"])
        self.assertEqual(10, plan.estimates["thermodynamic_case_stage_evaluations"])

    def test_missing_zero_is_injected_for_each_logical_geometry_then_deduplicated(self) -> None:
        settings = _cfl_settings(
            amplitudes=(0.01,),
            center=(600.0, 800.0),
        )
        plan = plan_experiment(
            settings, output_root=ROOT / "runs" / "test-cfl-zero-injected"
        )
        zeros = plan.case_table[plan.case_table["amplitude"] == 0.0]
        self.assertEqual(2, len(zeros))
        self.assertTrue(zeros["identity_control_injected"].all())
        self.assertEqual(1, zeros["physical_case_id"].nunique())
        self.assertEqual(1, int(zeros["planned_for_execution"].sum()))
        self.assertEqual(
            [True, False],
            [child.config.zero_amplitude_control_owner for child in plan.child_plans],
        )
        self.assertEqual(
            [True, False],
            [child.a0_identity_control_injected for child in plan.child_plans],
        )

    def test_zero_owner_is_lexicographic_not_input_order_dependent(self) -> None:
        settings = _cfl_settings(
            center=(800.0, 600.0),
            ramp_width=(100.0, 50.0),
        )
        plan = plan_experiment(
            settings, output_root=ROOT / "runs" / "test-cfl-owner-order"
        )
        owners = [
            child.config
            for child in plan.child_plans
            if child.config.zero_amplitude_control_owner
        ]
        self.assertEqual(1, len(owners))
        self.assertEqual(600.0, owners[0].epsilon0_mev_fm3)
        self.assertEqual((50.0,), owners[0].deltas_mev_fm3)
        zeros = plan.case_table[plan.case_table["amplitude"] == 0.0]
        self.assertEqual(4, len(zeros))
        self.assertEqual(1, int(zeros["planned_for_execution"].sum()))
        self.assertEqual(
            1,
            sum(
                child.estimates["direct_baseline_cases"]
                for child in plan.child_plans
            ),
        )

    def test_all_zero_cartesian_request_plans_one_executed_baseline(self) -> None:
        settings = _cfl_settings(amplitudes=(0.0,), center=(600.0, 800.0))
        plan = plan_experiment(
            settings, output_root=ROOT / "runs" / "test-cfl-all-zero"
        )
        self.assertEqual(2, len(plan.case_table))
        self.assertEqual(1, sum(len(child.case_table) for child in plan.child_plans))
        self.assertEqual(
            1, sum(len(child.logical_alias_table) for child in plan.child_plans)
        )
        self.assertEqual(1, plan.estimates["proposed_deformation_cases"])
        self.assertEqual(2, plan.estimates["logical_deformation_cases"])
        self.assertEqual(1, plan.estimates["physical_deformation_cases"])
        self.assertEqual(1, plan.estimates["deduplicated_logical_case_aliases"])
        self.assertEqual(1, plan.estimates["direct_baseline_cases"])
        self.assertEqual(2, plan.estimates["thermodynamic_case_stage_evaluations"])

    def test_config_rejects_domain_escape_and_frozen_contract_tampering(self) -> None:
        with self.assertRaisesRegex(ValueError, "representably distinct"):
            plan_experiment(
                _cfl_settings(ramp_width=1.0e-14),
                output_root=ROOT / "runs" / "test-cfl-unrepresentable-ramp",
            )
        with self.assertRaisesRegex(ValueError, "both directions"):
            plan_experiment(
                _cfl_settings(width=1.0e-14),
                output_root=ROOT / "runs" / "test-cfl-unrepresentable-width",
            )
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            plan_experiment(
                _cfl_settings(center=ENERGY_DENSITY_SURFACE_MEV_FM3),
                output_root=ROOT / "runs" / "test-cfl-bad-center",
            )
        domain_span = (
            ENERGY_DENSITY_MAX_MEV_FM3 - ENERGY_DENSITY_SURFACE_MEV_FM3
        )
        endpoint_plan = plan_experiment(
            _cfl_settings(ramp_width=domain_span),
            output_root=ROOT / "runs" / "test-cfl-endpoint-ramp",
        )
        self.assertEqual(
            (domain_span,), endpoint_plan.child_plans[0].config.deltas_mev_fm3
        )
        with self.assertRaisesRegex(ValueError, "ramp width"):
            plan_experiment(
                _cfl_settings(ramp_width=math.nextafter(domain_span, math.inf)),
                output_root=ROOT / "runs" / "test-cfl-bad-ramp",
            )
        plan = plan_experiment(
            _cfl_settings(), output_root=ROOT / "runs" / "test-cfl-tamper"
        )
        saved = plan.child_plans[0].config.to_dict()
        saved["baseline_parameter_set_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen contract"):
            CFLTrialConfig.from_dict(saved)
        unsupported = plan.child_plans[0].config.to_dict()
        unsupported["extended_stellar_diagnostics_enabled"] = True
        with self.assertRaisesRegex(ValueError, "extended diagnostics are unavailable"):
            CFLTrialConfig.from_dict(unsupported)
        derived_tamper = plan.child_plans[0].config.to_dict()
        derived_tamper["effective_amplitudes"] = [0.0, 0.5]
        with self.assertRaisesRegex(ValueError, "disagrees with its inputs"):
            CFLTrialConfig.from_dict(derived_tamper)


if __name__ == "__main__":
    unittest.main()

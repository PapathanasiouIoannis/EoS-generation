from __future__ import annotations

import json
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch

import jsonschema

import eos_generation
from eos_generation import ExperimentSettings, plan_experiment
from eos_generation.cfl.baseline import (
    ENERGY_DENSITY_MAX_MEV_FM3,
    ENERGY_DENSITY_SURFACE_MEV_FM3,
    FROZEN_PARAMETER_SET_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]


class CFLDocumentationAndConfigTests(unittest.TestCase):
    def test_cfl_quickstart_validates_and_plans_without_writes_or_solvers(self) -> None:
        schema = json.loads(
            (ROOT / "configs" / "schema.json").read_text(encoding="utf-8")
        )
        payload = json.loads(
            (ROOT / "configs" / "cfl_quickstart.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.Draft202012Validator(schema).validate(payload)

        settings = ExperimentSettings.from_json(
            ROOT / "configs" / "cfl_quickstart.json"
        )
        self.assertEqual("cfl", settings.matter_model)
        self.assertEqual("surface", settings.epsilon_match)
        self.assertEqual("off", settings.diagnostics)

        output_root = ROOT / "runs" / "test-cfl-documented-quickstart"
        self.assertFalse(output_root.exists())
        with patch(
            "eos_generation.cfl.baseline.brentq",
            side_effect=AssertionError("passive planning called a root solver"),
        ):
            plan = plan_experiment(settings, output_root=output_root)
        self.assertFalse(output_root.exists())
        self.assertEqual(2, len(plan.case_table))
        self.assertEqual(0, plan.estimates["deduplicated_logical_case_aliases"])

    def test_legacy_bsk24_examples_keep_the_omitted_discriminator_contract(self) -> None:
        for name in (
            "quickstart.json",
            "custom_experiment.json",
            "stellar_example.json",
        ):
            with self.subTest(name=name):
                payload = json.loads(
                    (ROOT / "configs" / name).read_text(encoding="utf-8")
                )
                self.assertNotIn("matter_model", payload)
                settings = ExperimentSettings.from_json(ROOT / "configs" / name)
                self.assertEqual("bsk24", settings.matter_model)
                self.assertNotIn("matter_model", settings.to_dict())

    def test_cfl_contract_documents_authoritative_identity_and_scope(self) -> None:
        contract = (ROOT / "docs" / "cfl.md").read_text(encoding="utf-8")
        for required in (
            "cfl_bag_full_ms_delta2_v1",
            FROZEN_PARAMETER_SET_SHA256,
            repr(ENERGY_DENSITY_SURFACE_MEV_FM3),
            repr(ENERGY_DENSITY_MAX_MEV_FM3),
            "190.218176006531",
            "4008.8172440269",
            "two-flavor",
            "no nuclear or electron crust",
            "applied exactly once",
            "independent solver",
            "cfl_dataset.ipynb",
            "dataset_40",
            "exactly five combined plots",
            "https://doi.org/10.1103/PhysRevD.66.074017",
            "https://doi.org/10.1103/PhysRevD.102.028501",
        ):
            with self.subTest(required=required):
                self.assertIn(required, contract)

    def test_release_metadata_and_cfl_manifest_packaging_are_aligned(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual("1.2.0", project["project"]["version"])
        self.assertEqual(project["project"]["version"], eos_generation.__version__)
        self.assertEqual(
            ["source_manifest.json"],
            project["tool"]["setuptools"]["package-data"]["eos_generation.cfl"],
        )
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("\nversion: 1.2.0\n", f"\n{citation}")
        self.assertIn("## 1.2.0 - 2026-08-30", changelog)


if __name__ == "__main__":
    unittest.main()

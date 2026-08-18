from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib
import unittest
from unittest.mock import patch

import eos_generation
from eos_generation._internal.provenance import _SOURCE_PATHS, _environment_record


ROOT = Path(__file__).resolve().parents[1]


class CompactProvenanceTests(unittest.TestCase):
    def test_saved_environment_provenance_contains_no_machine_paths(self) -> None:
        record = _environment_record()
        self.assertFalse(
            {"interpreter_path", "conda_prefix", "conda_executable"}.intersection(
                record
            )
        )
        self.assertEqual("packet_execution_environment", record["environment_role"])
        self.assertEqual("environment.yml", record["methods_runtime_specification"])

    def test_path_activated_conda_environment_is_not_serialized(self) -> None:
        with patch.dict(
            os.environ,
            {"CONDA_DEFAULT_ENV": r"C:\Users\example\envs\private-methods"},
        ):
            record = _environment_record()
        self.assertIsNone(record["conda_environment_name"])

        with patch.dict(
            os.environ,
            {"CONDA_DEFAULT_ENV": "eos-generation"},
        ):
            named = _environment_record()
        self.assertEqual("eos-generation", named["conda_environment_name"])

    def test_release_version_is_identical_across_public_metadata(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        version = project["project"]["version"]
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertEqual("1.0.0", version)
        self.assertEqual(version, eos_generation.__version__)
        self.assertIn(f"\nversion: {version}\n", f"\n{citation}")
        self.assertIn(f"## {version} -", changelog)

    def test_source_inventory_covers_every_shipped_package_module(self) -> None:
        package = ROOT / "src" / "eos_generation"
        observed = {
            path.relative_to(ROOT).as_posix()
            for path in package.rglob("*.py")
            if path.is_file()
        }
        observed.add("src/eos_generation/bsk24/source_manifest.json")
        self.assertEqual(set(_SOURCE_PATHS), observed)

    def test_packaged_bsk24_source_manifest_is_json_safe_and_specific(self) -> None:
        path = ROOT / "src" / "eos_generation" / "bsk24" / "source_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            "BSK24_ANALYTIC_PEARSON2018_CORR2019", payload["model_identifier"]
        )
        self.assertEqual(64, len(payload["independent_implementation_oracle"]["sha256"]))
        self.assertEqual(64, len(payload["underlying_tabulated_oracle"]["sha256"]))
        self.assertFalse(payload["source_storage_policy"]["raw_third_party_artifacts_committed"])
        json.dumps(payload, allow_nan=False)

    def test_compact_fixture_ledger_covers_every_fixture(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "bsk24_contract_v1"
        ledger = fixture / "SHA256SUMS.txt"
        expected: dict[str, str] = {}
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            expected[name.lstrip("*")] = digest.lower()
        observed = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in fixture.iterdir()
            if path.is_file() and path.name != ledger.name
        }
        self.assertEqual(expected, observed)

    def test_repository_tracks_no_runtime_output(self) -> None:
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT
        ).decode("utf-8").split("\0")
        self.assertFalse(
            any(path == "runs" or path.startswith("runs/") for path in tracked)
        )


if __name__ == "__main__":
    unittest.main()

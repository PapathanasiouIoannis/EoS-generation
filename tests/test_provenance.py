from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
import unittest
from unittest.mock import patch

import eos_generation
from eos_generation._internal.provenance import _SOURCE_PATHS, _environment_record


ROOT = Path(__file__).resolve().parents[1]


def _archive_content_paths(root: Path) -> tuple[str, ...]:
    content: list[str] = []
    for directory, child_directories, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        child_directories[:] = sorted(
            name for name in child_directories if name != ".git"
        )
        parent = Path(directory)
        content.extend(
            (parent / name).relative_to(root).as_posix()
            for name in child_directories
        )
        content.extend(
            (parent / name).relative_to(root).as_posix()
            for name in sorted(filenames)
            if name != ".git"
        )
    return tuple(sorted(content))


def _repository_content_paths(root: Path) -> tuple[str, tuple[str, ...]]:
    """List checkout-tracked or Git-free archive content from the exact root."""

    resolved = root.resolve(strict=False)
    required = (
        resolved / "pyproject.toml",
        resolved / "src" / "eos_generation",
        resolved / "tests",
    )
    if not all(path.exists() for path in required):
        raise AssertionError(f"repository root markers are incomplete: {resolved}")
    try:
        discovered = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=resolved,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "git_free_archive", _archive_content_paths(resolved)
    if not discovered or Path(discovered).resolve(strict=False) != resolved:
        return "git_free_archive", _archive_content_paths(resolved)
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=resolved, text=False
    ).decode("utf-8").split("\0")
    return "exact_git_checkout", tuple(path for path in tracked if path)


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
        contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
        developer = (ROOT / "docs" / "developer.md").read_text(encoding="utf-8")
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        bug_report = (
            ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
        ).read_text(encoding="utf-8")
        self.assertEqual("1.1.0", version)
        self.assertEqual(version, eos_generation.__version__)
        release = re.search(
            rf"(?m)^## {re.escape(version)} - (\d{{4}}-\d{{2}}-\d{{2}})$",
            changelog,
        )
        self.assertIsNotNone(release)
        release_date = release.group(1)
        self.assertRegex(
            citation,
            rf"(?m)^version:\s*{re.escape(version)}\s*$",
        )
        self.assertRegex(
            citation,
            rf"(?m)^date-released:\s*{re.escape(release_date)}\s*$",
        )
        wheel = f"dist/eos_generation-{version}-py3-none-any.whl"
        self.assertIn(wheel, contributing)
        self.assertIn(wheel, developer)
        self.assertIn(
            f'placeholder: "{version} or a full commit SHA"', bug_report
        )
        self.assertIn(
            f'assert version("eos-generation") '
            f'== eos_generation.__version__ == "{version}"',
            ci,
        )
        self.assertIn("| `1.1.x` | Supported |", security)
        self.assertIn("| `1.0.x` and earlier | Not supported |", security)
        for relative in (
            "CODE_OF_CONDUCT.md",
            "SECURITY.md",
            ".github/PULL_REQUEST_TEMPLATE.md",
            ".github/ISSUE_TEMPLATE/config.yml",
            ".github/ISSUE_TEMPLATE/scientific_change.yml",
            ".github/dependabot.yml",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

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
        source, content = _repository_content_paths(ROOT)
        self.assertIn(source, {"exact_git_checkout", "git_free_archive"})
        self.assertFalse(
            any(path == "runs" or path.startswith("runs/") for path in content)
        )

    def test_git_free_archive_fallback_does_not_require_git(self) -> None:
        with patch.object(
            subprocess,
            "check_output",
            side_effect=FileNotFoundError("synthetic missing Git executable"),
        ) as command:
            source, content = _repository_content_paths(ROOT)
        self.assertEqual("git_free_archive", source)
        self.assertEqual(1, command.call_count)
        self.assertIn("pyproject.toml", content)
        self.assertIn("src/eos_generation/__init__.py", content)

    def test_enclosing_git_repository_is_not_used_for_an_archive(self) -> None:
        with patch.object(
            subprocess,
            "check_output",
            return_value=str(ROOT.parent) + os.linesep,
        ) as command:
            source, content = _repository_content_paths(ROOT)
        self.assertEqual("git_free_archive", source)
        self.assertEqual(1, command.call_count)
        self.assertIn("tests/test_provenance.py", content)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def repository_files(root):
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if Path(top).resolve() != root.resolve():
            raise ValueError("foreign Git root")
        names = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ]
            )
            .decode()
            .split("\0")
        )
        return [root / name for name in names if name and (root / name).is_file()]
    except (OSError, ValueError, subprocess.CalledProcessError):
        ignored = {
            ".git",
            "runs",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "build",
            "dist",
            ".codex_tmp",
        }
        return [
            p
            for p in root.rglob("*")
            if p.is_file()
            and not set(p.relative_to(root).parts) & ignored
            and not any(
                part.endswith(".egg-info") for part in p.relative_to(root).parts
            )
        ]


def assert_hygiene(root):
    for path in repository_files(root):
        relative = path.relative_to(root)
        assert not set(relative.parts) & {
            "runs",
            "__pycache__",
            ".pytest_cache",
            ".ipynb_checkpoints",
        }, relative
        assert path.suffix not in {".pyc", ".pyo", ".png", ".zip", ".pdf"}, relative
        assert path.stat().st_size < 2_000_000, relative


def test_repository_and_git_free_archive_hygiene(tmp_path):
    assert_hygiene(ROOT)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/example.py").write_text("pass\n")
    assert repository_files(tmp_path) == [tmp_path / "src/example.py"]
    assert_hygiene(tmp_path)


def test_reference_fixture_manifest_is_unchanged():
    fixture = ROOT / "tests/fixtures/bsk24_contract_v1"
    for line in (fixture / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((fixture / name).read_bytes()).hexdigest() == digest


def test_packaged_scientific_source_and_runtime_contracts(tmp_path):
    import eos_generation
    import zipfile
    from eos_generation.storage import archive_source, source_identity

    package = Path(eos_generation.__file__).parent
    source = json.loads((package / "source_manifest.json").read_text())
    assert source
    contracts = ("environment.yml", "pyproject.toml", "README.md", "LICENSE")
    for name in contracts:
        assert (package / "runtime_contracts" / name).read_bytes() == (
            ROOT / name
        ).read_bytes()
    run = tmp_path / "runs" / "archive-check"
    run.mkdir(parents=True)
    inventory = source_identity()
    archive = archive_source(run, inventory)
    with zipfile.ZipFile(run / archive["path_relative_to_run"]) as saved:
        assert set(saved.namelist()) == set(inventory["files"])
        for name in contracts:
            assert saved.read(name) == (ROOT / name).read_bytes()
            assert (
                saved.read("src/eos_generation/runtime_contracts/" + name)
                == (ROOT / name).read_bytes()
            )
    from eos_generation import experiment, settings

    assert (
        eos_generation.ExperimentSettings
        is experiment.ExperimentSettings
        is settings.ExperimentSettings
    )


def test_configs_match_editor_schema():
    import jsonschema
    from eos_generation import ExperimentSettings

    schema = json.loads((ROOT / "configs/schema.json").read_text())
    for path in (ROOT / "configs").glob("*.json"):
        if path.name != "schema.json":
            jsonschema.validate(json.loads(path.read_text()), schema)
            ExperimentSettings.from_json(path)

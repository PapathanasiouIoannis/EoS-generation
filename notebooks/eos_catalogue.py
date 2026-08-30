"""Persistent presentation aliases and labelled copies of sealed EoS tables.

This is a reporting adapter, not a scientific workflow. It never runs solvers,
changes canonical IDs, or edits an authoritative packet. Registration is
append-only; an OS file lock serializes writers and is released on process exit.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any
import uuid


LEGACY_SCHEMA = "bsk24_friendly_eos_catalogue_v1"
SCHEMA = "eos_generation_friendly_eos_catalogue_v2"
IDENTITY_POLICY = "exact_coordinates_and_saved_eos_source_v1"
MODEL_CONTRACTS = {
    "bsk24": {
        "prefix": "H",
        "baseline_family": "BSk24",
        "deformed_family": "deformed_BSk24",
        "baseline_validation_status": "pass",
    },
    "cfl": {
        "prefix": "C",
        "baseline_family": "CFL",
        "deformed_family": "deformed_CFL",
        "baseline_validation_status": (
            "literature_supported_frozen_design_contract"
        ),
    },
}
PRIMARY_TABLES = (
    "case_ledger.csv", "thermodynamic_profiles.csv", "stellar_sequences.csv",
    "fixed_mass_observables.csv", "maximum_mass_screening.csv",
)
COORDINATES = (
    "amplitude", "epsilon_match_mev_fm3", "epsilon0_mev_fm3",
    "sigma_mev_fm3", "delta_mev_fm3",
)
CONTEXT = (
    "eos_id", "catalogue_id", "physical_model_key", "matter_model", "model_family",
    "experiment_path", "experiment_hash", "geometry_id", "packet_path",
    "configuration_hash", "precision", "source_case_id",
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def trusted_repository_root(requested: str | Path) -> Path:
    """Bind command-line use to the checkout that owns this script.

    Library-level builders still accept explicit roots so their pure saved-data
    behavior can be tested in temporary repositories.  Command-line entry
    points must call this boundary before importing or opening repository code.
    """

    trusted = Path(__file__).resolve(strict=True).parents[1]
    if requested != trusted and requested != str(trusted):
        raise ValueError(
            "--repository-root must identify the reviewed checkout that owns "
            f"this script: {trusted}"
        )
    return trusted


def confined(path: Path, parent: Path) -> Path:
    """Reject symlink/junction escapes, including not-yet-created destinations."""

    allowed = Path(os.path.realpath(parent.resolve(strict=True)))
    allowed_text = str(allowed)
    allowed_prefix = allowed_text.rstrip(os.sep) + os.sep
    lexical = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if lexical == allowed_text:
        checked = allowed_text
    elif lexical.startswith(allowed_prefix):
        checked = lexical
    else:
        raise ValueError(f"path escapes its allowed parent: {path}")
    resolved = Path(os.path.realpath(checked))
    try:
        common = Path(os.path.commonpath((str(allowed), str(resolved))))
    except ValueError as exc:
        raise ValueError(f"path escapes its allowed parent: {path}") from exc
    if common != allowed:
        raise ValueError(f"path escapes its allowed parent: {path}")
    return resolved


def require_disjoint(destination: Path, *sources: Path) -> None:
    """Reject a derived output equal to, inside, or containing any source."""

    for source in sources:
        if (
            destination == source
            or destination in source.parents
            or source in destination.parents
        ):
            raise ValueError(
                "derived destination overlaps an authoritative source: "
                f"{destination} and {source}"
            )


def manifest(folder: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (folder / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        checksum, separator, name = line.partition("  ")
        if (separator != "  " or not re.fullmatch(r"[0-9a-f]{64}", checksum)
                or not name or name in result):
            raise ValueError("malformed or duplicate checksum entry")
        confined(folder / name, folder)
        result[name] = checksum
    return result


def verified(folder: Path, name: str, checksums: dict[str, str]) -> Path:
    path = confined(folder / name, folder)
    if name not in checksums or not path.is_file() or sha256(path) != checksums[name]:
        raise ValueError(f"sealed source checksum mismatch: {path}")
    return path


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def case_identity_keys(
    row: dict[str, Any], *, source_field: str
) -> tuple[str, ...]:
    """Return governed logical and physical lookup IDs for one occurrence."""

    logical_id = row.get(source_field)
    if not isinstance(logical_id, str) or not logical_id:
        raise ValueError("missing source case identity")
    flag = row.get("is_physical_case_alias")
    if flag is None or flag is False or str(flag).strip().lower() in {"", "false"}:
        return (logical_id,)
    if flag is not True and str(flag).strip().lower() != "true":
        raise ValueError("invalid physical case alias flag")
    physical_id = row.get("physical_case_id")
    try:
        amplitude = float(row.get("amplitude"))
    except (TypeError, ValueError):
        amplitude = math.nan
    if (
        not isinstance(physical_id, str)
        or not physical_id
        or physical_id == logical_id
        or amplitude != 0.0
        or row.get("status") not in {"accepted", "baseline"}
    ):
        raise ValueError("invalid physical A=0 case alias")
    return logical_id, physical_id


def _model_from_family(family: str) -> str:
    for matter_model, contract in MODEL_CONTRACTS.items():
        if family in {contract["baseline_family"], contract["deformed_family"]}:
            return matter_model
    raise ValueError(f"unrecognized EoS model family: {family!r}")


def physical_definition(
    row: dict[str, Any], source_hashes: dict[str, str], matter_model: str = "bsk24"
) -> dict[str, Any]:
    """Precision-independent, conservative identity; no curve-similarity merging.

Changes to any saved EoS implementation/config source create a new model key.
Stellar solvers and reporting changes are evaluation provenance, not EoS identity.
The baseline collapses geometry only after the caller has checked identity pass.
"""
    if matter_model not in MODEL_CONTRACTS:
        raise ValueError(f"unsupported matter model in EoS identity: {matter_model!r}")
    if matter_model == "bsk24":
        physics = {
            name: checksum for name, checksum in source_hashes.items()
            if name.startswith("src/eos_generation/bsk24/")
            or name == "src/eos_generation/_internal/config.py"
        }
        required = {
            "src/eos_generation/bsk24/baseline.py",
            "src/eos_generation/bsk24/deformation.py",
            "src/eos_generation/bsk24/reconstruction.py",
            "src/eos_generation/_internal/config.py",
        }
    else:
        physics = {
            name: checksum for name, checksum in source_hashes.items()
            if name.startswith("src/eos_generation/cfl/")
            or name == "src/eos_generation/_internal/cfl_thermodynamics.py"
        }
        required = {
            "src/eos_generation/cfl/baseline.py",
            "src/eos_generation/cfl/deformation.py",
            "src/eos_generation/cfl/reconstruction.py",
            "src/eos_generation/_internal/cfl_thermodynamics.py",
        }
    if not required.issubset(physics) or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in physics.values()
    ):
        raise ValueError("saved EoS source identity is incomplete or malformed")
    amplitude = float(row["amplitude"])
    if not math.isfinite(amplitude):
        raise ValueError("nonfinite amplitude in EoS identity")
    baseline = amplitude == 0.0
    contract = MODEL_CONTRACTS[matter_model]
    coordinates = {}
    if not baseline:
        for name in COORDINATES:
            value = float(row[name])
            if not math.isfinite(value) or (name != "amplitude" and value <= 0):
                raise ValueError(f"invalid physical coordinate: {name}")
            coordinates[name] = value.hex()
    return {
        "identity_policy": IDENTITY_POLICY,
        "model_family": (
            contract["baseline_family"] if baseline else contract["deformed_family"]
        ),
        "eos_source_hashes": physics,
        "coordinates_hex": coordinates,
    }


@contextmanager
def registry_lock(folder: Path, timeout: float = 10.0):
    """An abandoned process cannot leave a held OS lock behind."""
    path = confined(folder / "catalogue.lock", folder)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("EoS catalogue is busy; no IDs were reassigned")
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_registry(folder: Path) -> tuple[dict[str, dict[str, Any]], str | None, str | None, int]:
    entries: dict[str, dict[str, Any]] = {}
    catalogue_id = previous = None
    transactions = sorted(folder.glob("registration_*.json"))
    for number, path in enumerate(transactions, 1):
        confined(path, folder)
        if path.name != f"registration_{number:06d}.json":
            raise ValueError("EoS catalogue registration sequence has a gap")
        transaction = read_json(path)
        checksum = transaction.pop("sha256", None)
        if checksum != digest(transaction):
            raise ValueError("EoS catalogue checksum mismatch")
        schema = transaction.get("schema_id")
        if (schema not in {LEGACY_SCHEMA, SCHEMA} or transaction.get("number") != number
                or transaction.get("previous_sha256") != previous):
            raise ValueError("EoS catalogue chain mismatch")
        if catalogue_id is None:
            catalogue_id = transaction.get("catalogue_id")
            if not isinstance(catalogue_id, str) or not re.fullmatch(r"[0-9a-f]{32}", catalogue_id):
                raise ValueError("malformed EoS catalogue namespace")
        if transaction.get("catalogue_id") != catalogue_id:
            raise ValueError("mixed EoS catalogue namespaces")
        additions = transaction.get("entries")
        if not isinstance(additions, list) or not additions:
            raise ValueError("empty or malformed EoS registration")
        prefix_counts = {
            contract["prefix"]: sum(
                item.get("eos_id", "").startswith(contract["prefix"])
                for item in entries.values()
            )
            for contract in MODEL_CONTRACTS.values()
        }
        for entry in additions:
            key = entry.get("physical_model_key")
            definition = entry.get("definition")
            if not isinstance(definition, dict):
                raise ValueError("malformed EoS catalogue definition")
            matter_model = _model_from_family(str(definition.get("model_family", "")))
            contract = MODEL_CONTRACTS[matter_model]
            prefix = contract["prefix"]
            if (key != digest(definition) or key in entries
                    or entry.get("eos_id") != f"{prefix}{prefix_counts[prefix]:06d}"):
                raise ValueError("duplicate or inconsistent EoS catalogue assignment")
            if (prefix_counts[prefix] == 0
                    and definition.get("model_family") != contract["baseline_family"]):
                raise ValueError(f"the first {matter_model} EoS must be its validated baseline")
            if schema == LEGACY_SCHEMA and matter_model != "bsk24":
                raise ValueError("legacy catalogue transactions may contain only BSk24 entries")
            entries[key] = entry
            prefix_counts[prefix] += 1
        previous = checksum
    return entries, catalogue_id, previous, len(transactions)


def register_definitions(root: Path, definitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    runs = (root / "runs").resolve(strict=True)
    requested = runs / "eos_catalogue"
    if requested.is_symlink() or requested.is_junction():
        raise ValueError("the shared EoS catalogue must not be a linked directory")
    folder = confined(requested, runs)
    folder.mkdir(exist_ok=True)
    with registry_lock(folder):
        entries, catalogue_id, previous, count = read_registry(folder)
        def registration_order(key: str) -> tuple[str, bool, str]:
            family = str(definitions[key].get("model_family", ""))
            matter_model = _model_from_family(family)
            contract = MODEL_CONTRACTS[matter_model]
            return contract["prefix"], family != contract["baseline_family"], key

        new_keys = sorted(set(definitions) - set(entries), key=registration_order)
        prefix_counts = {
            contract["prefix"]: sum(
                item.get("eos_id", "").startswith(contract["prefix"])
                for item in entries.values()
            )
            for contract in MODEL_CONTRACTS.values()
        }
        additions = []
        for key in new_keys:
            definition = definitions[key]
            if digest(definition) != key:
                raise ValueError("physical model key mismatch")
            matter_model = _model_from_family(str(definition.get("model_family", "")))
            contract = MODEL_CONTRACTS[matter_model]
            prefix = contract["prefix"]
            if (prefix_counts[prefix] == 0
                    and definition.get("model_family") != contract["baseline_family"]):
                raise ValueError(f"initial {matter_model} registration requires its baseline")
            entry = {
                "eos_id": f"{prefix}{prefix_counts[prefix]:06d}",
                "physical_model_key": key,
                "definition": definition,
            }
            entries[key] = entry
            additions.append(entry)
            prefix_counts[prefix] += 1
        if additions:
            catalogue_id = catalogue_id or uuid.uuid4().hex
            transaction = {
                "schema_id": SCHEMA, "catalogue_id": catalogue_id,
                "number": count + 1, "previous_sha256": previous, "entries": additions,
            }
            previous = digest(transaction)
            transaction["sha256"] = previous
            # Publish a fully flushed new file; never replace an old registration.
            target = folder / f"registration_{count + 1:06d}.json"
            descriptor, temporary_name = tempfile.mkstemp(prefix=".registration_", dir=folder)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(canonical(transaction) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, target)  # atomic no-clobber publication on NTFS/POSIX
            finally:
                temporary.unlink(missing_ok=True)
        return {
            "entries": entries, "catalogue_id": catalogue_id,
            "registration_sha256": previous, "catalogue_path": str(folder),
        }


def validate_source(experiment: Path) -> None:
    from eos_generation import validate_experiment

    report = validate_experiment(experiment)
    if report.get("status") != "pass":
        reasons = list(report.get("failures", []))
        for child in report.get("children", []):
            reasons.extend(child.get("failures", []))
        raise ValueError(
            "EoS aliases require a completed, hard-valid experiment; "
            + "; ".join(dict.fromkeys(str(reason) for reason in reasons))[:1500]
        )


def collect_sources(root: Path, experiment: Path) -> dict[str, Any]:
    document = read_json(experiment / "experiment.json")
    if document.get("status") != "complete":
        raise ValueError("EoS aliases require a completed experiment")
    settings = document.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("experiment settings are missing")
    matter_model = str(settings.get("matter_model", "bsk24"))
    if matter_model not in MODEL_CONTRACTS:
        raise ValueError(f"unsupported experiment matter model: {matter_model!r}")
    children = document.get("child_packets", [])
    hashes = document.get("child_configuration_hashes", [])
    if not children or len(children) != len(hashes) or len(set(children)) != len(children):
        raise ValueError("malformed experiment child declaration")
    definitions, occurrences, packets = {}, [], []
    for child, expected in zip(children, hashes, strict=True):
        packet = confined(experiment / child, experiment)
        if packet.parent != experiment:
            raise ValueError("unsafe geometry packet path")
        checksums = manifest(packet)
        for name in (
            "metadata.json", "run_state.json", "source_hashes.json",
            "complete_configuration.json", "case_ledger.csv",
        ):
            verified(packet, name, checksums)
        metadata = read_json(packet / "metadata.json")
        state = read_json(packet / "run_state.json")
        configuration = read_json(packet / "complete_configuration.json")
        if (metadata.get("packet_status") != "complete" or state.get("packet_status") != "complete"
                or metadata.get("configuration_hash") != expected
                or state.get("configuration_hash") != expected
                or metadata.get("baseline_validation_status")
                != MODEL_CONTRACTS[matter_model]["baseline_validation_status"]
                or metadata.get("identity_status") != "pass"):
            raise ValueError("packet identity, baseline, or completion check failed")
        sources = read_json(packet / "source_hashes.json")
        ledger = csv_rows(packet / "case_ledger.csv")
        if len({row["case_id"] for row in ledger}) != len(ledger):
            raise ValueError("duplicate canonical case IDs")
        zero_owner = configuration.get("zero_amplitude_control_owner", True)
        if not isinstance(zero_owner, bool):
            raise ValueError("malformed zero-amplitude control ownership")
        direct_rows = (
            [{"case_id": "direct", "amplitude": "0", "status": "baseline"}]
            if zero_owner else []
        )
        by_case = {}
        for row in [*direct_rows, *ledger]:
            case_id = row["case_id"]
            if case_id in by_case:
                raise ValueError("duplicate baseline case ID")
            accepted = row["status"] in {"accepted", "baseline"}
            definition = physical_definition(row, sources, matter_model) if accepted else None
            key = digest(definition) if accepted else ""
            if accepted:
                definitions[key] = definition
            occurrence = {
                **row, "eos_id": "", "catalogue_id": "", "physical_model_key": key,
                "matter_model": matter_model,
                "model_family": definition["model_family"] if accepted else "",
                "experiment_path": experiment.relative_to(root).as_posix(),
                "experiment_hash": document["settings_hash"], "geometry_id": child,
                "packet_path": packet.relative_to(root).as_posix(),
                "configuration_hash": expected, "precision": document["settings"]["precision"],
                "source_case_id": case_id,
                "case_role": "direct" if case_id == "direct" else (
                    "identity_control" if float(row["amplitude"]) == 0 else "deformation"
                ),
            }
            for identity in case_identity_keys(row, source_field="case_id"):
                if identity in by_case:
                    raise ValueError("duplicate baseline case ID")
                by_case[identity] = occurrence
            occurrences.append(occurrence)
        tables = {}
        for name in PRIMARY_TABLES:
            if (packet / name).is_file():
                tables[name] = verified(packet, name, checksums)
        if "thermodynamic_profiles.csv" not in tables:
            raise ValueError("accepted EoS profiles are missing")
        # Discover schema/unknown IDs before any persistent assignment is made.
        for path in tables.values():
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or "case_id" not in reader.fieldnames:
                    raise ValueError("primary table has no case_id")
                presentation_only = set(CONTEXT) - {"matter_model"}
                if presentation_only.intersection(reader.fieldnames):
                    raise ValueError("authoritative table already contains presentation columns")
                for row in reader:
                    if row["case_id"] not in by_case:
                        raise ValueError("primary table contains an unknown case ID")
                    if (
                        "matter_model" in row
                        and row["matter_model"] != matter_model
                    ):
                        raise ValueError(
                            "authoritative table matter_model disagrees with experiment"
                        )
        packets.append({"tables": tables, "by_case": by_case, "checksums": checksums, "path": packet})
    return {"definitions": definitions, "occurrences": occurrences, "packets": packets, "document": document}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys([*CONTEXT, *(key for row in rows for key in row)]))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def publish_directory(stage: Path, target: Path) -> None:
    """Publish a same-parent stage without overwriting a destination.

    The exclusive sidecar serializes cooperating publishers.  The target is
    checked again immediately before every rename attempt; ``os.replace`` is
    intentionally forbidden for immutable result packets.
    """

    if stage.parent != target.parent:
        raise ValueError("presentation stage must share the destination parent")
    lock = target.with_name(f".{target.name}.publish.lock")
    acquired_lock = False
    try:
        with lock.open("x", encoding="utf-8") as stream:
            acquired_lock = True
            stream.write(f"stage={stage.name}\n")
            stream.flush()
            for attempt in range(4):
                if target.exists() or target.is_symlink():
                    raise FileExistsError(
                        f"presentation destination already exists: {target}"
                    )
                try:
                    os.rename(stage, target)
                    return
                except PermissionError:
                    if attempt == 3:
                        raise
                    time.sleep(0.1 * (attempt + 1))
    finally:
        if acquired_lock:
            lock.unlink(missing_ok=True)


def build_eos_data(root: Path, experiment: Path, destination: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    runs = (root / "runs").resolve(strict=True)
    experiment = confined(experiment, runs)
    target = confined(destination, runs)
    catalogue_root = confined(runs / "eos_catalogue", runs)
    if target == runs:
        raise ValueError("labelled output overlaps an authoritative or catalogue directory")
    try:
        require_disjoint(target, experiment, catalogue_root)
    except ValueError as exc:
        raise ValueError(
            "labelled output overlaps an authoritative or catalogue directory"
        ) from exc
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"labelled output already exists: {target}")
    validate_source(experiment)
    collected = collect_sources(root, experiment)
    registry = register_definitions(root, collected["definitions"])
    for row in collected["occurrences"]:
        key = row["physical_model_key"]
        row["catalogue_id"] = registry["catalogue_id"]
        if key:
            row["eos_id"] = registry["entries"][key]["eos_id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".eos_data_", dir=target.parent)).resolve()
    try:
        write_rows(stage / "case_aliases.csv", collected["occurrences"])
        unique = {}
        for row in collected["occurrences"]:
            eos_id = row["eos_id"]
            if eos_id and (
                eos_id not in unique
                or (
                    row["case_role"] == "direct"
                    and unique[eos_id]["case_role"] != "direct"
                )
            ):
                unique[eos_id] = row
        write_rows(stage / "eos_catalogue.csv", [unique[key] for key in sorted(unique)])
        for name in PRIMARY_TABLES:
            packets = [packet for packet in collected["packets"] if name in packet["tables"]]
            if not packets:
                continue
            columns = list(CONTEXT)
            for packet in packets:
                with packet["tables"][name].open(encoding="utf-8", newline="") as stream:
                    columns.extend(key for key in next(csv.reader(stream)) if key not in columns)
            with (stage / name).open("x", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                for packet in packets:
                    verified(packet["path"], name, packet["checksums"])
                    with packet["tables"][name].open(encoding="utf-8", newline="") as stream:
                        for row in csv.DictReader(stream):
                            context = packet["by_case"][row["case_id"]]
                            writer.writerow({**{key: context[key] for key in CONTEXT}, **row})
        provenance = {
            "schema_id": SCHEMA, "identity_policy": IDENTITY_POLICY,
            "experiment_path": experiment.relative_to(root).as_posix(),
            "settings_hash": collected["document"]["settings_hash"],
            "catalogue_id": registry["catalogue_id"],
            "catalogue_path": Path(registry["catalogue_path"]).relative_to(root).as_posix(),
            "registration_sha256": registry["registration_sha256"],
            "matter_models": sorted({row["matter_model"] for row in collected["occurrences"]}),
            "unique_eos_count": len(unique), "scope": "current experiment; global aliases",
            "builder_sha256": sha256(Path(__file__)),
            "source_manifests": {
                packet["path"].relative_to(root).as_posix(): sha256(packet["path"] / "SHA256SUMS.txt")
                for packet in collected["packets"]
            },
            "solver_calls": 0, "authoritative_packets_modified": False,
        }
        (stage / "provenance.json").write_bytes(canonical(provenance) + b"\n")
        (stage / "README.md").write_text(
            "# Friendly EoS data\n\n"
            "eos_catalogue.csv has one row per physical model in this experiment; its "
            "source columns identify a representative evaluation, not all evaluations. "
            "case_aliases.csv maps every original case/geometry, including repeated "
            "controls and rejected proposals. Rejected proposals have blank eos_id.\n\n"
            "Primary CSVs add aliases/provenance but preserve original case_id, values, "
            "row order, all numerical stages, missing entries and failure statuses. "
            "They are derived copies, not replacements for sealed packets. Group by "
            "eos_id AND evaluation provenance; QUICK/STRICT and repeated controls are "
            "not independent EoSs. Use the final stage and valid observable statuses "
            "for analysis. Missing M_max is not zero or the largest sampled mass.\n\n"
            "H labels identify BSk24 EoSs and C labels identify CFL EoSs; neither is a "
            "claim about observational acceptance. IDs are not ML features. Each model "
            "family begins with its validated baseline at H000000 or C000000. Archive "
            "runs/eos_catalogue with your data: numbering is local to its catalogue_id. "
            "Never delete, renumber, or reset it.\n\n"
            "Identical exact coordinates and saved EoS source signatures reuse IDs "
            "across precision and stellar solver changes. EoS/config source changes "
            "conservatively create new identities, even for a new baseline version. "
            "No existing packets or older presentation folders were changed.\n",
            encoding="utf-8",
        )
        (stage / "SHA256SUMS.txt").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(stage.iterdir())),
            encoding="utf-8",
        )
        publish_directory(stage, target)
    except Exception:
        shutil.rmtree(stage)
        raise
    return {**provenance, "data_path": str(target), "catalogue_path": registry["catalogue_path"]}


def load_aliases(root: Path, experiment: Path, folder: Path) -> list[dict[str, str]]:
    """Read a sealed derived mapping, bound to the exact source packets."""
    folder = confined(folder, root / "runs")
    checksums = manifest(folder)
    provenance = read_json(verified(folder, "provenance.json", checksums))
    if provenance.get("experiment_path") != experiment.relative_to(root).as_posix():
        raise ValueError("EoS aliases belong to a different experiment")
    for name, expected in provenance["source_manifests"].items():
        packet = confined(root / name, experiment)
        if sha256(packet / "SHA256SUMS.txt") != expected:
            raise ValueError("EoS aliases are stale for the source packet")
    return csv_rows(verified(folder, "case_aliases.csv", checksums))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = trusted_repository_root(args.repository_root)
    print(json.dumps(build_eos_data(root, args.experiment, args.destination), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

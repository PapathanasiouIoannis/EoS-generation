"""Atomic storage, strict JSON, integrity and portable source provenance."""

from __future__ import annotations

import hashlib
import json
import math
import numpy as np
import os
import pandas as pd
import platform
import tempfile
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any


RUN_SCHEMA = "eos_generation_run_v2"
MANIFEST = "SHA256SUMS.txt"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def resolve_runs_path(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve()
    # A portable installed package may execute in any working directory.
    # Require a runs ancestor without binding data to the source checkout.
    if not any(part.lower() == "runs" for part in result.parts[:-1]):
        raise ValueError("the destination must be below a runs directory")
    return result


def json_clean(value: Any, **_kwargs: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: str | Path) -> dict[str, Any]:
    def reject(value):
        raise ValueError(f"non-finite JSON value {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    result = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject,
        object_pairs_hook=unique,
    )
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json_atomic(value: Any, path: Path) -> None:
    _atomic_bytes(
        Path(path),
        (
            json.dumps(json_clean(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    _atomic_bytes(
        Path(path), frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    )


def seal(data: Path) -> None:
    files = sorted(p for p in data.iterdir() if p.name != MANIFEST)
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError("the data directory must contain regular files only")
    content = "".join(f"{sha256(p)}  {p.name}\n" for p in files)
    _atomic_bytes(data / MANIFEST, content.encode("utf-8"))


def verify_manifest(data: Path) -> dict[str, str]:
    declared = {}
    for line in (data / MANIFEST).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid manifest digest")
        if (
            name in declared
            or PurePosixPath(name).name != name
            or "\\" in name
            or name == MANIFEST
        ):
            raise ValueError("unsafe or duplicate manifest path")
        declared[name] = digest
    actual = {p.name for p in data.iterdir() if p.name != MANIFEST}
    if set(declared) != actual:
        raise ValueError("manifest coverage differs from the saved data")
    for name, digest in declared.items():
        path = data / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"manifest mismatch: {name}")
    return declared


def source_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    paths = {
        "src/eos_generation/" + p.name: p
        for p in package.iterdir()
        if p.suffix in {".py", ".json"}
    }
    paths.update(
        {
            "src/eos_generation/runtime_contracts/" + p.name: p
            for p in (package / "runtime_contracts").iterdir()
            if p.is_file()
        }
    )
    for name in ("pyproject.toml", "environment.yml", "README.md", "LICENSE"):
        path = project_root() / name
        if not path.is_file():
            path = package / "runtime_contracts" / name
        if path.is_file():
            paths[name] = path
    hashes = {name: sha256(path) for name, path in sorted(paths.items())}
    return {"sha256": hash_payload(hashes), "files": hashes}


def environment_identity() -> dict[str, str]:
    values = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for name in ("numpy", "scipy", "pandas", "matplotlib", "numba"):
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = "unavailable"
    return values


def archive_source(run_path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    runs = next(parent for parent in run_path.parents if parent.name.lower() == "runs")
    archive = runs / "_sources" / (identity["sha256"] + ".zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parent
    if not archive.exists():
        fd, temporary = tempfile.mkstemp(suffix=".zip", dir=archive.parent)
        os.close(fd)
        try:
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED
            ) as handle:
                for name, digest in identity["files"].items():
                    path = (
                        package / name.removeprefix("src/eos_generation/")
                        if name.startswith("src/eos_generation/")
                        else project_root() / name
                    )
                    if not path.is_file() and not name.startswith(
                        "src/eos_generation/"
                    ):
                        path = package / "runtime_contracts" / name
                    if sha256(path) != digest:
                        raise ValueError("source changed during archival")
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    handle.writestr(info, path.read_bytes())
            # Exclusive publication keeps simultaneous runs from overwriting.
            try:
                os.link(temporary, archive)
            except FileExistsError:
                pass
        finally:
            Path(temporary).unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as handle:
        hashes = {
            name: hashlib.sha256(handle.read(name)).hexdigest()
            for name in handle.namelist()
        }
        if hashes != identity["files"]:
            raise ValueError("source archive does not match the reviewed source")
    return {
        "path_relative_to_run": os.path.relpath(archive, run_path).replace("\\", "/"),
        "sha256": sha256(archive),
    }


def validate_run(
    run_path: Path, *, require_source_equivalence: bool = False
) -> dict[str, Any]:
    """Read-only validation; no analytical evaluation, interpolation or solver."""
    errors = []
    availability = {}
    source_status = "unavailable"
    metadata = {}
    try:
        data = run_path / "data"
        manifest = verify_manifest(data)
        metadata = strict_json(data / "run.json")
        if metadata.get("schema_id") != RUN_SCHEMA:
            raise ValueError("unsupported run schema")
        if metadata.get("status") != "complete":
            raise ValueError(f"run status is {metadata.get('status')!r}")
        from .settings import ExperimentSettings

        settings = ExperimentSettings.from_dict(metadata["settings"])
        if settings.deterministic_hash() != metadata.get("settings_hash"):
            raise ValueError("settings hash mismatch")
        if not metadata.get("imported_legacy") and hash_payload(
            metadata["environment_identity"]
        ) != metadata.get("environment_sha256"):
            raise ValueError("environment identity hash mismatch")
        source = metadata["source_identity"]
        if hash_payload(source["files"]) != source["sha256"]:
            raise ValueError("source inventory hash mismatch")
        source_status = (
            "equivalent"
            if source_identity()["sha256"] == source["sha256"]
            else "different"
        )
        if require_source_equivalence and source_status != "equivalent":
            raise ValueError("exact-source validation requires the recorded source")
        required = {"run.json", "cases.csv", "raw.csv", "eos.csv"}
        for observable in settings.requested_observables:
            required.add(
                {
                    "sequence": "stars",
                    "fixed_mass": "fixed_mass",
                    "maximum_mass": "maximum_mass",
                }[observable]
                + ".csv"
            )
        if not required.issubset(manifest):
            raise ValueError("missing requested scientific tables")
        if metadata.get("raw_gate_before_downstream") is not True:
            raise ValueError("raw-before-downstream evidence is missing")
        cases = pd.read_csv(
            data / "cases.csv", float_precision="round_trip", keep_default_na=False
        )
        raw = pd.read_csv(data / "raw.csv", float_precision="round_trip")
        eos = pd.read_csv(data / "eos.csv", float_precision="round_trip")
        if cases.empty or cases.case_id.duplicated().any():
            raise ValueError("empty or duplicate logical case declarations")
        if not set(cases.status).issubset({"accepted", "rejected", "unresolved"}):
            raise ValueError("unfinished or unknown case status")
        certificates = metadata["certificates"]
        gates = certificates["raw_gate"]
        physical = set(cases.physical_case_id)
        if physical != set(gates) or physical != set(raw.case_id):
            raise ValueError("raw evidence does not cover the physical cases exactly")
        accepted = set(cases.loc[cases.status.eq("accepted"), "physical_case_id"])
        if set(eos.case_id) != accepted:
            raise ValueError(
                "reconstruction coverage differs from accepted physical cases"
            )
        for case_id, group in cases.groupby("physical_case_id", sort=False):
            if group.status.nunique() != 1 or group.failure_reason.nunique() != 1:
                raise ValueError("logical aliases disagree on outcome")
            gate = gates[case_id]
            expected = {
                "accepted": "accepted_raw_local_physics_gate",
                "rejected": "rejected_raw_local_physics_gate",
                "unresolved": "unresolved_raw_local_physics_gate",
            }[group.status.iloc[0]]
            if (
                gate.get("status") != expected
                or gate.get("complete_raw_proposal_assessed") is not True
            ):
                raise ValueError(f"inconsistent complete raw assessment: {case_id}")
            if gate.get("clipping_clamping_smoothing_repair") != "none":
                raise ValueError("raw proposal was repaired")
            samples = raw.loc[raw.case_id.eq(case_id)]
            eps = samples.epsilon_mev_fm3.to_numpy(float)
            domain = gate["complete_proposed_retained_domain_mev_fm3"]
            if (
                len(eps) != gate["dense_grid_points"]
                or not np.all(np.isfinite(eps))
                or not np.all(np.diff(eps) > 0)
                or eps[0] != domain[0]
                or eps[-1] != domain[1]
            ):
                raise ValueError(
                    "saved raw grid differs from the complete assessed grid"
                )
            if not samples.gate_status.eq(expected).all():
                raise ValueError("raw table status differs from the certificate")
            parameters = gate["parameters"]
            for name in (
                "amplitude",
                "epsilon0_mev_fm3",
                "sigma_mev_fm3",
                "delta_mev_fm3",
            ):
                if not samples[name].eq(parameters[name]).all():
                    raise ValueError("raw parameters differ from the gate declaration")
            if not group.amplitude.eq(parameters["amplitude"]).all():
                raise ValueError("logical requests differ from their physical evidence")
            if group.status.iloc[0] != "accepted":
                reason = (gate.get("first_failure") or {}).get("reason")
                if not reason or not group.failure_reason.eq(reason).all():
                    raise ValueError(
                        "rejected/unresolved case lacks its exact failure reason"
                    )
                continue
            retained = gate["retained_domain"]
            raw_values = samples[["raw_pressure_mev_fm3", "raw_cs2"]].to_numpy(float)
            prefix = eps <= retained["epsilon_max_mev_fm3"]
            if (
                not np.isfinite(raw_values).all()
                or np.any(raw_values[:, 0] < 0)
                or np.any(raw_values[:, 1] <= 0)
                or np.any(raw_values[prefix, 1] > 1)
            ):
                raise ValueError(
                    "accepted raw evidence violates the assessed cold-phase predicates"
                )
            if (
                gate.get("selected_retained_domain_authoritative") is not True
                or gate.get("selected_retained_domain_passed") is not True
                or retained.get("passed") is not True
                or retained.get("resolution_certified") is not True
            ):
                raise ValueError("accepted case lacks a certified retained prefix")
            table = eos.loc[eos.case_id.eq(case_id)]
            columns = [
                "epsilon_mev_fm3",
                "pressure_mev_fm3",
                "cs2",
                "baryon_density_fm3",
                "effective_baryon_enthalpy_mev",
            ]
            full_state = set(columns).issubset(table)
            if not full_state and not metadata.get("imported_legacy"):
                raise ValueError("missing effective cold-state columns")
            values = table[columns if full_state else columns[:3]].to_numpy(float)
            epsilon, pressure, cs2 = values[:, :3].T
            if (
                not np.isfinite(values).all()
                or not np.all(epsilon > 0)
                or not np.all(pressure >= 0)
                or not np.all((cs2 > 0) & (cs2 <= 1))
                or not np.all(np.diff(epsilon) > 0)
                or not np.all(np.diff(pressure) > 0)
            ):
                raise ValueError(
                    "accepted cold-phase state violates a hard physical predicate"
                )
            if (
                epsilon[0] != retained["epsilon_min_mev_fm3"]
                or epsilon[-1] != retained["epsilon_max_mev_fm3"]
                or pressure[-1] != retained["pressure_max_mev_fm3"]
            ):
                raise ValueError(
                    "saved reconstruction differs from its retained domain"
                )
            if not full_state:
                availability["legacy_effective_cold_state"] = (
                    "not_saved; pressure/sound-speed curves and raw certificates checked"
                )
                continue
            density, mu = values[:, 3:].T
            if (
                not np.all(density > 0)
                or not np.all(mu > 0)
                or not np.all(np.diff(density) > 0)
            ):
                raise ValueError(
                    "accepted effective density/chemical potential violates a hard physical predicate"
                )
            tolerance = (
                64
                * np.finfo(float).eps
                * np.maximum(1, np.abs(epsilon) + np.abs(pressure))
            )
            if np.any(np.abs(density * mu - epsilon - pressure) > tolerance):
                raise ValueError(
                    "saved Euler identity does not close under the floating-point policy"
                )
            if np.any(
                np.abs(mu - (epsilon + pressure) / density)
                > 64 * np.finfo(float).eps * np.maximum(1, np.abs(mu))
            ):
                raise ValueError("saved chemical-potential identity does not close")
            if metadata.get("imported_legacy"):
                availability["legacy_reconstruction_certificate"] = (
                    "not_saved; raw certificates and saved cold identities checked"
                )
                continue
            stages = certificates["reconstruction"].get(case_id, {})
            expected_stages = {
                stage["name"]
                for stage in metadata["numerical_profile"]["thermodynamic_stages"]
            }
            if set(stages) != expected_stages:
                raise ValueError("missing reconstruction certificates")
            for evidence in stages.values():
                admissibility = evidence.get(
                    "retained_domain_thermodynamic_admissibility", {}
                )
                if (
                    admissibility.get("status")
                    not in {
                        "accepted_selected_domain_thermodynamic_gate",
                        "accepted_full_domain_thermodynamic_gate",
                    }
                    or admissibility.get("failed_checks")
                    or not admissibility.get("independent_checks")
                    or not all(
                        value is True
                        for value in admissibility["independent_checks"].values()
                    )
                ):
                    raise ValueError(
                        "reconstruction cold-identity/matching certificate failed"
                    )
                if (
                    evidence.get("tabulation_resolution", {}).get(
                        "interpolation_inversion_status"
                    )
                    != "resolved_finite_monotone_nonextrapolating"
                ):
                    raise ValueError(
                        "reconstruction interpolation/inversion certificate failed"
                    )
        identity = certificates["a0_identity"]
        if (
            identity.get("status") != "pass"
            or not identity.get("deltas")
            or any(
                value.get("status") != "pass" or value.get("array_equal") is not True
                for delta in identity["deltas"].values()
                for value in delta.values()
            )
        ):
            raise ValueError("zero-amplitude identity certificate failed")
        endpoints = {
            key: float(gates[key]["retained_domain"]["pressure_max_mev_fm3"])
            for key in accepted
        }
        saved_sequences = (
            pd.read_csv(data / "stars.csv", float_precision="round_trip")
            if (data / "stars.csv").is_file()
            else pd.DataFrame()
        )
        if settings.calculation == "stellar" and not metadata.get("imported_legacy"):
            sequence_evidence = certificates["stellar"]["sequence_evidence"]
            stage_names = {
                stage["name"] for stage in metadata["numerical_profile"]["tov_stages"]
            }
            if set(sequence_evidence) != {
                case_id + ":" + stage for case_id in accepted for stage in stage_names
            }:
                raise ValueError("missing declared stellar case/stage evidence")
            for key, evidence in sequence_evidence.items():
                case_id, stage = key.rsplit(":", 1)
                rows = saved_sequences.loc[
                    saved_sequences.case_id.eq(case_id)
                    & saved_sequences.stage.eq(stage)
                ]
                if (
                    len(rows) != evidence["attempted_count"]
                    or rows.calculation_status.eq("success").sum()
                    != evidence["successful_count"]
                    or rows.calculation_status.eq("failed").sum()
                    != evidence["failed_count"]
                ):
                    raise ValueError(
                        "stellar attempts are missing or differ from the sequence evidence"
                    )
                if rows.attempted_index.duplicated().any() or not np.all(
                    np.diff(
                        rows.sort_values(
                            "attempted_index"
                        ).central_pressure_mev_fm3.to_numpy(float)
                    )
                    > 0
                ):
                    raise ValueError(
                        "stellar attempts are duplicate or not ordered by central pressure"
                    )
        for name in ("stars", "fixed_mass", "maximum_mass"):
            path = data / (name + ".csv")
            if not path.is_file():
                continue
            frame = pd.read_csv(path, float_precision="round_trip")
            if not set(frame.case_id).issubset(accepted):
                raise ValueError("stellar work references a rejected or undeclared EoS")
            if "central_pressure_mev_fm3" in frame:
                finite = frame.central_pressure_mev_fm3.notna()
                pressures = frame.loc[finite, "central_pressure_mev_fm3"].to_numpy(
                    float
                )
                limits = frame.loc[finite, "case_id"].map(endpoints).to_numpy(float)
                if (
                    np.any(~np.isfinite(pressures))
                    or np.any(pressures > limits)
                    or np.any(pressures <= 0)
                ):
                    raise ValueError(
                        "stellar work exceeded the retained pressure domain"
                    )
            if name == "stars":
                if not set(frame.calculation_status).issubset({"success", "failed"}):
                    raise ValueError("unknown stellar attempt status")
                failed = frame.loc[frame.calculation_status.eq("failed")]
                if not failed.empty and failed.failure_reason.isna().any():
                    raise ValueError("stellar failure lacks its exact reason")
                from .stellar import classify_saved_tidal_rows

                tidal = classify_saved_tidal_rows(frame, schema="sequence")
                availability["sequence_tidal_status_counts"] = (
                    tidal.tidal_validity_reason.value_counts().to_dict()
                )
                claimed = frame.get(
                    "tidal_status", pd.Series(index=frame.index, dtype=str)
                ).eq("validated_lambda_validation_v1") & frame.calculation_status.eq(
                    "success"
                )
                if (claimed & ~tidal.tidal_valid).any():
                    raise ValueError(
                        "claimed validated tidal output lacks finite positive Lambda and finite k2"
                    )
            elif name == "fixed_mass":
                for row in frame.loc[
                    frame.status.eq("bracketed_and_solved")
                ].itertuples(index=False):
                    bracket = ast_literal_bracket(row.bracket_pressure_mev_fm3)
                    if (
                        not bracket[0] < bracket[1]
                        or not bracket[0] <= row.central_pressure_mev_fm3 <= bracket[1]
                        or bracket[1] > endpoints[row.case_id]
                    ):
                        raise ValueError(
                            "fixed mass lacks a true retained-domain pressure bracket"
                        )
                    population = saved_sequences.loc[
                        saved_sequences.case_id.eq(row.case_id)
                        & saved_sequences.stage.eq(row.stage)
                    ].copy()
                    if hasattr(row, "legacy_execution_role"):
                        population = population.loc[
                            population.legacy_execution_role.eq(
                                row.legacy_execution_role
                            )
                        ]
                    population = population.loc[
                        population.calculation_status.eq("success")
                    ].sort_values("central_pressure_mev_fm3")
                    if "is_on_successful_stable_prefix" in population:
                        population = population.loc[
                            population.is_on_successful_stable_prefix.eq(True)
                        ]
                    else:
                        peaks = np.flatnonzero(
                            population.is_sampled_peak.to_numpy(bool)
                        )
                        if len(peaks) != 1:
                            raise ValueError(
                                "legacy successful stable-prefix evidence is missing"
                            )
                        population = population.iloc[: peaks[0] + 1]
                    masses = population.Mass.to_numpy(float)
                    if (
                        len(masses) < 2
                        or not np.all(np.isfinite(masses))
                        or not np.all(np.diff(masses) > 0)
                    ):
                        raise ValueError(
                            "fixed mass does not have a monotone successful stable prefix"
                        )
                    pressures = population.central_pressure_mev_fm3.to_numpy(float)
                    lower = np.flatnonzero(pressures == bracket[0])
                    if (
                        len(lower) != 1
                        or lower[0] + 1 >= len(pressures)
                        or pressures[lower[0] + 1] != bracket[1]
                        or not masses[lower[0]]
                        <= row.target_mass_msun
                        <= masses[lower[0] + 1]
                    ):
                        raise ValueError(
                            "fixed mass is not bracketed by adjacent stable-prefix samples"
                        )
                from .stellar import classify_saved_tidal_rows

                tidal = classify_saved_tidal_rows(frame, schema="fixed_mass")
                claimed = frame.get(
                    "tidal_status", pd.Series(index=frame.index, dtype=str)
                ).eq("validated_lambda_validation_v1") & frame.status.eq(
                    "bracketed_and_solved"
                )
                if (claimed & ~tidal.tidal_valid).any():
                    raise ValueError(
                        "claimed validated fixed-mass tidal result is unusable"
                    )
                availability["fixed_mass_tidal_status_counts"] = (
                    tidal.tidal_validity_reason.value_counts().to_dict()
                )
            else:
                for row in frame.loc[frame.maximum_mass_resolved.eq(True)].itertuples(
                    index=False
                ):
                    if (
                        row.status
                        not in {
                            "resolved_unique_turning_point",
                            "resolved_unique_turning_point_local_sequence_refinement",
                        }
                        or not row.positive_left_secant > 0
                        or not row.negative_right_secant < 0
                        or row.turning_point_count != 1
                    ):
                        raise ValueError(
                            "maximum mass lacks bracketed/refined turning-point evidence"
                        )
            status_column = "calculation_status" if name == "stars" else "status"
            availability[name + "_status_counts"] = (
                frame[status_column].value_counts().to_dict()
            )
        archive = metadata.get("source_archive")
        availability["source_archive_available"] = False
        if archive:
            path = (run_path / archive["path_relative_to_run"]).resolve()
            if path.is_file():
                if sha256(path) != archive["sha256"]:
                    raise ValueError("source archive integrity mismatch")
                with zipfile.ZipFile(path) as handle:
                    hashes = {
                        name: hashlib.sha256(handle.read(name)).hexdigest()
                        for name in handle.namelist()
                    }
                    if hashes != source["files"]:
                        raise ValueError("source archive content mismatch")
                availability["source_archive_available"] = True
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        pd.errors.EmptyDataError,
    ) as exc:
        errors.append(str(exc))
    return {
        "passed": not errors,
        "schema_id": "eos_generation_validation_v2",
        "interpretation_version": "saved_cold_phase_certificates_v1",
        "status": metadata.get("status", "unavailable"),
        "errors": errors,
        "source_equivalence": source_status,
        "scientific_output_availability": availability,
        "scientific_solver_calls": 0,
        "filesystem_writes": 0,
    }


def ast_literal_bracket(value: Any) -> tuple[float, float]:
    import ast

    result = ast.literal_eval(value) if isinstance(value, str) else value
    if (
        not isinstance(result, (tuple, list))
        or len(result) != 2
        or not all(math.isfinite(float(x)) for x in result)
    ):
        raise ValueError("invalid saved pressure bracket")
    return tuple(float(x) for x in result)


def _legacy_manifest(directory: Path) -> dict[str, str]:
    """Verify all nested files in a legacy packet without current-source gates."""
    declarations = {}
    for line in (directory / MANIFEST).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or name in declarations
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("unsafe legacy manifest")
        file = directory.joinpath(*relative.parts)
        if file.is_symlink() or not file.is_file() or sha256(file) != digest:
            raise ValueError(f"legacy manifest mismatch: {name}")
        declarations[name] = digest
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p.name != MANIFEST
    }
    if set(declarations) != actual:
        raise ValueError("legacy manifest coverage mismatch")
    return declarations


def import_legacy(source: str | Path, destination: str | Path):
    """Copy scientific evidence into a new flat run; never write to the source.

    Historical reconstruction certificates that were not saved remain explicitly
    unavailable. Original evidence, source identities and interpretation survive
    in one run document; historical calculations are never rerun or recertified.
    """
    source = Path(source).resolve()
    destination = resolve_runs_path(destination)
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("legacy source and destination must be separate")
    aggregate = strict_json(source / "experiment.json")
    if (
        aggregate.get("schema_id") != "eos_generation_experiment_v1"
        or aggregate.get("status") != "complete"
    ):
        raise ValueError("import requires a complete v1 public experiment")
    # The v1 aggregate seals its own documents; each child seals its full packet.
    for name, digest in aggregate["document_sha256"].items():
        if PurePosixPath(name).name != name or sha256(source / name) != digest:
            raise ValueError("legacy aggregate document mismatch")
    evidence = []
    frames = {
        name: []
        for name in ("cases", "raw", "eos", "stars", "fixed_mass", "maximum_mass")
    }
    gates = {}
    identity = {"status": "pass", "deltas": {}}
    hashes = {}
    children = []
    for name in aggregate["child_packets"]:
        if PurePosixPath(name).name != name or "\\" in name:
            raise ValueError("unsafe legacy child path")
        child = source / name
        manifest = _legacy_manifest(child)
        documents = {p.name: strict_json(p) for p in child.glob("*.json")}
        if documents["metadata.json"].get("packet_status") != "complete":
            raise ValueError("legacy child is incomplete")
        gate_document = documents["raw_gate_report.json"]
        if (
            gate_document.get("schema_id") != "eos_generation_raw_gate_v2"
            or gate_document.get("executed_before_reconstruction_and_TOV") is not True
        ):
            raise ValueError(
                "legacy complete-domain raw-gate capability is not established"
            )
        for key, report in gate_document["cases"].items():
            if key in gates:
                raise ValueError("duplicate physical legacy gate evidence")
            gates[key] = report
        children.append((name, child, manifest, documents))
    for index, (name, child, manifest, documents) in enumerate(children, 1):
        ledger = pd.read_csv(
            child / "case_ledger.csv",
            float_precision="round_trip",
            keep_default_na=False,
        )
        if ledger.amplitude.eq(0).sum() != 1:
            raise ValueError(
                "import currently requires one zero control per v1 geometry"
            )
        if "physical_case_id" not in ledger:
            ledger["physical_case_id"] = ledger.case_id
        a0 = str(ledger.loc[ledger.amplitude.eq(0), "physical_case_id"].iloc[0])
        for row in ledger.to_dict("records"):
            key = row["physical_case_id"]
            report = gates[key]
            status = {
                "accepted_raw_local_physics_gate": "accepted",
                "rejected_raw_local_physics_gate": "rejected",
                "unresolved_raw_local_physics_gate": "unresolved",
            }[report["status"]]
            frames["cases"].append(
                {
                    **row,
                    "geometry_index": index,
                    "physical_case_id": key,
                    "planned_for_execution": row.get("planned_for_execution", True),
                    "is_alias": row.get("is_alias", False),
                    "status": status,
                    "failure_reason": (report.get("first_failure") or {}).get(
                        "reason", ""
                    ),
                }
            )
        for old, new in (
            ("raw_gate_profiles", "raw"),
            ("thermodynamic_profiles", "eos"),
            ("stellar_sequences", "stars"),
            ("fixed_mass_observables", "fixed_mass"),
            ("maximum_mass_screening", "maximum_mass"),
        ):
            path = child / (old + ".csv")
            if not path.is_file():
                continue
            frame = pd.read_csv(path, float_precision="round_trip")
            if new == "eos":
                frame = frame.loc[~frame.case_id.eq("direct")].copy()
            else:
                # Preserve repeated historical calculations with their role.
                frame["legacy_execution_role"] = np.where(
                    frame.case_id.eq("direct"), "direct_control", "proposal"
                )
                frame["case_id"] = frame.case_id.replace("direct", a0)
            frames[new].append(frame)
        local = documents["identity_report.json"].get(
            "local_thermodynamic_identity", {}
        )
        if not local.get("deltas") and not documents["raw_gate_report.json"][
            "cases"
        ].get(a0):
            local = {"status": "pass", "deltas": {}}
        if local.get("status") != "pass":
            raise ValueError("legacy zero-amplitude identity failed")
        identity["deltas"].update(
            {name + ":" + key: value for key, value in local["deltas"].items()}
        )
        hashes.update(
            {
                name + "/" + key: value
                for key, value in documents["source_hashes.json"].items()
            }
        )
        evidence.append(
            {"geometry": name, "manifest": manifest, "documents": documents}
        )
    settings = dict(aggregate["settings"])
    record = {
        "schema_id": RUN_SCHEMA,
        "status": "running",
        "imported_legacy": True,
        "settings": settings,
        "settings_hash": hash_payload(settings),
        "plan_hash": aggregate["plan_hash"],
        "numerical_profile": {
            "version": "legacy_saved_expansions",
            "children": [
                item["documents"]["complete_configuration.json"] for item in evidence
            ],
        },
        "source_identity": {"files": hashes, "sha256": hash_payload(hashes)},
        "raw_gate_before_downstream": True,
        "certificates": {"raw_gate": gates, "a0_identity": identity},
        "legacy_evidence": {
            "aggregate": aggregate,
            "children": evidence,
            "importer_source": source_identity(),
        },
        "source_archive_availability": "not provided by legacy format",
        "interpretation": "historical saved raw/identity/stellar evidence; missing certificates remain unavailable",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    data = destination / "data"
    data.mkdir()
    write_json_atomic(record, data / "run.json")
    try:
        write_csv_atomic(pd.DataFrame(frames.pop("cases")), data / "cases.csv")
        for name, tables in frames.items():
            if tables:
                write_csv_atomic(
                    pd.concat(tables, ignore_index=True), data / (name + ".csv")
                )
        record["status"] = "complete"
        write_json_atomic(record, data / "run.json")
        seal(data)
        from .experiment import load_experiment

        return load_experiment(destination)
    except BaseException as exc:
        record.update(
            status="failed", failure={"type": type(exc).__name__, "message": str(exc)}
        )
        write_json_atomic(record, data / "run.json")
        seal(data)
        raise

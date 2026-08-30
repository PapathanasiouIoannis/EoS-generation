"""Derived, non-authoritative student views of validated experiment packets.

The view is deliberately outside the sealed aggregate experiment.  It copies
saved CSV and PNG artifacts byte-for-byte and never imports or calls a
scientific or plotting implementation.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from eos_generation._experiment_integrity import _verify_aggregate_manifest
from eos_generation._internal.artifacts import (
    ensure_within_runs,
    repository_root_scope,
)
from eos_generation._internal.packet_integrity import (
    _strict_json_payload,
    _verify_packet_manifest_exact,
)


_PRIMARY_TABLES = (
    "case_ledger.csv",
    "thermodynamic_profiles.csv",
    "stellar_sequences.csv",
    "fixed_mass_observables.csv",
    "maximum_mass_screening.csv",
)
_REQUIRED_TABLES = frozenset(_PRIMARY_TABLES[:2])
_PASS_STATUSES = frozenset({"pass", "complete", "validated"})
_PUBLICATION_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4, 0.8)
_TABLE_PURPOSES = {
    "case_ledger.csv": "Saved accepted/rejected lifecycle outcome and deformation coordinates for every declared case.",
    "thermodynamic_profiles.csv": "Saved effective one-fluid EoS profile samples for the direct baseline and reconstructed cases.",
    "stellar_sequences.csv": "Saved stellar-sequence model attempts when stellar work was requested and available.",
    "fixed_mass_observables.csv": "Saved observables at requested, truly bracketed gravitational masses when available.",
    "maximum_mass_screening.csv": "Saved maximum-mass screening and turning-point status when available.",
}
_TABLE_GUIDANCE = {
    "case_ledger.csv": (
        "One row represents one declared deformation proposal. Use `case_id` to "
        "join its amplitude, geometry, and accepted/rejected status to downstream "
        "tables. The analytical `direct` baseline is not a deformation proposal "
        "and therefore need not have a ledger row."
    ),
    "thermodynamic_profiles.csv": (
        "One row represents one sampled energy-density point for one `case_id`; "
        "all rows with the same `case_id` belong to the same EoS. Use "
        "`epsilon_mev_fm3` as the physical horizontal coordinate rather than the "
        "spreadsheet row number."
    ),
    "stellar_sequences.csv": (
        "One row represents one stellar-model attempt for one `case_id` and "
        "numerical stage at its saved central coordinate. Filter by the saved "
        "calculation and capability status before using mass, radius, or tidal "
        "values."
    ),
    "fixed_mass_observables.csv": (
        "One row represents one requested target mass for one `case_id` and "
        "numerical stage when a true stable-branch bracket was available. Preserve "
        "the saved status when selecting rows."
    ),
    "maximum_mass_screening.csv": (
        "One row represents one maximum-mass assessment for one `case_id` and "
        "numerical stage. A largest sampled mass is not a resolved maximum unless "
        "the saved turning-point and resolution status says so."
    ),
}


@dataclass(frozen=True)
class StudentView:
    """Paths created for one immutable derived presentation view."""

    path: Path
    readme: Path
    plots: Path
    primary_data: Path
    optional_diagnostics: Path
    data_dictionary: Path
    authoritative_experiment: Path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_children(
    source: Path,
    metadata: Mapping[str, Any],
    validation_report: Mapping[str, Any],
) -> tuple[str, ...]:
    if validation_report.get("status") != "pass":
        raise ValueError("student view requires a passing experiment validation")
    report_path = validation_report.get("experiment_path")
    if report_path is not None and Path(str(report_path)).resolve(strict=False) != source:
        raise ValueError("validation report belongs to a different experiment")

    raw_children = metadata.get("child_packets")
    if not isinstance(raw_children, list) or not raw_children:
        raise ValueError("completed experiment has no child packets")
    children: list[str] = []
    for value in raw_children:
        if (
            not isinstance(value, str)
            or not value.startswith("geometry_")
            or "/" in value
            or "\\" in value
            or value in {".", ".."}
        ):
            raise ValueError(f"unsafe child packet path: {value!r}")
        child = (source / value).resolve(strict=False)
        if child.parent != source or child.is_symlink() or not child.is_dir():
            raise ValueError(f"unsafe or missing child packet: {value!r}")
        children.append(value)
    if len(set(children)) != len(children):
        raise ValueError("completed experiment contains duplicate child packets")

    reports = validation_report.get("children")
    if (
        not isinstance(reports, list)
        or len(reports) != len(children)
        or validation_report.get("child_packet_count") != len(children)
    ):
        raise ValueError("validation report does not cover every child packet")
    for report in reports:
        if not isinstance(report, Mapping):
            raise ValueError("child validation report is malformed")
        status = report.get("status")
        overall = report.get("overall_status")
        if status not in _PASS_STATUSES and overall not in _PASS_STATUSES:
            raise ValueError("student view requires every child packet to pass validation")
    return tuple(children)


def _safe_saved_file(path: Path, packet: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"saved artifact is unsafe or missing: {path.name}")
    resolved = path.resolve(strict=False)
    if not _is_relative_to(resolved, packet):
        raise ValueError(f"saved artifact escapes its packet: {path.name}")
    return resolved


def _copy_saved(source: Path, destination: Path, packet: Path) -> None:
    artifact = _safe_saved_file(source, packet)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact, destination)


def _table_columns(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        try:
            header = next(rows)
        except StopIteration as exc:
            raise ValueError(f"saved CSV has no header: {path.name}") from exc
    if not header or any(not value for value in header):
        raise ValueError(f"saved CSV has a malformed header: {path.name}")
    return tuple(header)


def _relative_link(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start=start)).as_posix()


def _render_readme(
    *,
    stage: Path,
    source: Path,
    destination: Path,
    settings_hash: str,
    copied_tables: tuple[Path, ...],
    copied_plots: tuple[Path, ...],
) -> str:
    relative_source = _relative_link(source, destination)
    by_name = {
        name: tuple(
            sorted(
                path.relative_to(stage).as_posix()
                for path in copied_tables
                if path.name == name
            )
        )
        for name in _PRIMARY_TABLES
    }

    def locations(name: str) -> str:
        paths = by_name[name]
        if not paths:
            return "not applicable or unavailable for this validated run"
        return ", ".join(f"[`{path}`]({path})" for path in paths)

    lines = [
        "# Read me first",
        "",
        "This is a derived, non-authoritative student view made only from saved artifacts after the authoritative experiment passed validation. No thermodynamic, TOV, tidal, or plotting calculation was rerun. Files in the authoritative packet were not changed.",
        "",
        f"- Authoritative technical packet: [`{relative_source}`]({relative_source})",
        f"- Canonical configuration hash: `{settings_hash}`",
        f"- EoS data: {locations('thermodynamic_profiles.csv')}",
        f"- Case acceptance/rejection ledger: {locations('case_ledger.csv')}",
        f"- Stellar sequences: {locations('stellar_sequences.csv')}",
        f"- Fixed-mass observables: {locations('fixed_mass_observables.csv')}",
        f"- Maximum-mass screening: {locations('maximum_mass_screening.csv')}",
        f"- Generated plots: [`02_PLOTS/`](02_PLOTS/) ({len(copied_plots)} PNG files)",
        "- Units and exact saved columns: [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md)",
        "- Checksums for every file in this view: [`SHA256SUMS.txt`](SHA256SUMS.txt)",
        "",
        "## Folder guide",
        "",
        "- `02_PLOTS/` contains copies of generated PNGs.",
        "- `03_PRIMARY_DATA/` contains the main saved result tables.",
        "- `04_OPTIONAL_DIAGNOSTICS/` contains other saved CSV diagnostics, when present.",
        "- Each `geometry_NNN/` directory corresponds exactly to one authoritative geometry child packet; rows are not merged across geometries.",
        "",
        "## How the data is organized",
        "",
        "A CSV row is not a new EoS. The hierarchy is:",
        "",
        "```text",
        "experiment (one canonical configuration hash)",
        "└── geometry_NNN (one center, width, ramp-width, and anchor combination)",
        "    └── case_id (one baseline or one deformation amplitude)",
        "        └── rows (sampled thermodynamic points or stellar models)",
        "```",
        "",
        "- `case_id = direct` is the undeformed analytical BSk24 baseline saved for comparison.",
        "- A deformation with amplitude `A = 0` is a separate identity-control case. It passes through the reconstruction workflow and is expected to reproduce the baseline under the governed identity policy.",
        "- Each nonzero-amplitude `case_id` is one distinct deformed EoS within that geometry.",
        "- Read amplitude and geometry values from `case_ledger.csv`; do not try to recover the complete scientific identity from the readable part of the case-ID text. Its final hexadecimal suffix is a deterministic collision-resistant digest of the deformation coordinates.",
        "- A rejected ledger row is a completed scientific outcome. It deliberately has no reconstructed thermodynamic profile or stellar sequence.",
        "- `student_view_eligibility_status` is the saved case-level selection status. For stellar work, `eligible_all_requested_fixed_masses_succeeded` requires every configured target at the final reporting stage, but it does not require a resolved maximum mass.",
        "- Read `requested_fixed_masses_status` and `maximum_mass_availability_status` independently. An unavailable maximum mass does not erase valid fixed-mass rows inside the retained EoS domain.",
        "",
        "## Which files to keep together",
        "",
        "For portable EoS analysis, copy both `case_ledger.csv` and `thermodynamic_profiles.csv` from the same `geometry_NNN/` directory. The profiles contain the sampled quantities and case IDs; the ledger supplies the complete deformation coordinates and accepted/rejected status. For stellar work, keep that same ledger with the applicable stellar CSVs.",
        "",
        "When combining several geometry directories, add the `geometry_NNN` folder name as a column in your own analysis. When combining separate experiments, also retain the canonical configuration hash and authoritative packet location. A `case_id` identifies a deformation proposal, not the complete source-code and runtime provenance of an experiment.",
        "",
        "## Minimal analysis workflow",
        "",
        "1. Open `case_ledger.csv` and select an accepted `case_id` with the `student_view_eligibility_status` appropriate to your analysis.",
        "2. Open the result table needed for the question and filter its `case_id` column to that exact value.",
        "3. For an EoS curve, plot `epsilon_mev_fm3` on the horizontal axis and a saved quantity such as `pressure_mev_fm3` or `cs2` on the vertical axis.",
        "4. Treat `direct` as the baseline comparison. Do not compare cases by spreadsheet row number alone; use the saved physical coordinate and status columns.",
        "5. Blank or unavailable values are not zero. Preserve the saved status and reason columns when exporting or filtering data.",
        "",
        "These are ordinary UTF-8 CSV files and can be opened in Excel, LibreOffice, R, pandas, Julia, MATLAB, Mathematica, or another CSV-capable tool. [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) records the exact columns present in this particular run.",
        "",
        "## Interpretation boundary",
        "",
        "Energy density and pressure include the conventions recorded by the authoritative method: energy density includes rest-mass energy, energy-density/pressure columns with `_mev_fm3` are in MeV fm^-3, `dP/dε = c_s^2` is dimensionless with `c = 1`, fixed masses are gravitational masses in solar masses, and radius columns with `_km` are in kilometres. A rejected case remains rejected and has no reconstructed or stellar values. The reconstructed state is an effective one-fluid cold barotrope; this view does not establish microscopic composition, species chemical potentials, or beta equilibrium.",
        "",
        "Use the authoritative packet and its validation/provenance documents for reproduction or technical review. This convenience view is not part of that packet's manifest.",
    ]
    return "\n".join(lines) + "\n"


def _render_dictionary(stage: Path, tables: tuple[Path, ...]) -> str:
    lines = [
        "# Data dictionary",
        "",
        "This dictionary lists the exact headers of the CSV files copied from the validated authoritative packet. It does not rename, merge, derive, or reinterpret columns.",
        "",
        "## Unit cues encoded in saved column names",
        "",
        "- `_mev_fm3`: MeV fm^-3",
        "- `_msun`: solar masses",
        "- `_km`: kilometres",
        "- `cs2` and named fractions: dimensionless",
        "- Status, reason, case, stage, and capability columns are categorical bookkeeping; preserve their saved text exactly.",
        "",
        "## How to identify a data point",
        "",
        "- `case_id` groups rows belonging to the same baseline or deformation. A change of spreadsheet row is not by itself a change of EoS.",
        "- Within a thermodynamic case, `epsilon_mev_fm3` identifies the sampled total-energy-density coordinate.",
        "- Within a stellar case, the saved stage and central-pressure/central-energy-density columns identify the attempted stellar model.",
        "- Do not use row numbers as scientific identifiers. When copying a subset, keep its `case_id`, physical coordinate, stage, and status columns.",
        "- Empty fields mean unavailable or non-applicable unless the table explicitly defines otherwise; they must not be replaced with zero.",
        "",
    ]
    for path in sorted(tables, key=lambda item: item.relative_to(stage).as_posix()):
        relative = path.relative_to(stage).as_posix()
        purpose = _TABLE_PURPOSES.get(
            path.name,
            "Optional saved diagnostic or bookkeeping table from the authoritative packet.",
        )
        guidance = _TABLE_GUIDANCE.get(path.name)
        lines.extend(
            (
                f"## `{relative}`",
                "",
                purpose,
                "",
            )
        )
        if guidance is not None:
            lines.extend((f"Row meaning: {guidance}", ""))
        lines.extend(("Columns in saved order:", ""))
        lines.extend(f"- `{column}`" for column in _table_columns(path))
        lines.append("")
    return "\n".join(lines)


def _write_checksums(stage: Path) -> None:
    manifest = stage / "SHA256SUMS.txt"
    files = sorted(
        path for path in stage.rglob("*") if path.is_file() and path != manifest
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(stage).as_posix()}" for path in files
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _publish_staged_directory(stage: Path, target: Path) -> None:
    """Atomically publish one same-directory stage without overwriting.

    Windows can transiently deny a directory rename while an editor or virus
    scanner holds a sharing handle.  Only ``PermissionError`` is retried, for
    one bounded schedule.  Every attempt rechecks the no-overwrite condition;
    if a target appears during a retry, it wins and publication fails closed.
    """

    if stage.parent != target.parent:
        raise ValueError("student view publication stage must share the target parent")
    for attempt in range(len(_PUBLICATION_RETRY_DELAYS_SECONDS) + 1):
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"student view destination already exists: {target}"
            )
        try:
            os.rename(stage, target)
            return
        except FileExistsError as exc:
            raise FileExistsError(
                f"student view destination already exists: {target}"
            ) from exc
        except PermissionError as exc:
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    f"student view destination already exists: {target}"
                ) from exc
            if attempt == len(_PUBLICATION_RETRY_DELAYS_SECONDS):
                raise
            time.sleep(_PUBLICATION_RETRY_DELAYS_SECONDS[attempt])


def create_student_view(
    experiment_path: str | Path,
    *,
    validation_report: Mapping[str, Any],
    repository_root: str | Path,
    destination: str | Path | None = None,
) -> StudentView:
    """Copy a validated completed experiment into a separate student view.

    ``validation_report`` must be the fresh report produced by the production
    aggregate validator immediately before this call.  Existing destinations
    are always rejected; no overwrite policy is provided by this layer.
    """

    root = Path(repository_root).expanduser().resolve(strict=False)
    with repository_root_scope(root):
        source = ensure_within_runs(experiment_path).resolve(strict=False)
        if source.is_symlink() or not source.is_dir():
            raise ValueError("authoritative experiment is unsafe or missing")
        target = ensure_within_runs(
            destination
            if destination is not None
            else source.with_name(f"{source.name}_STUDENT_VIEW")
        ).resolve(strict=False)
        if _is_relative_to(target, source) or _is_relative_to(source, target):
            raise ValueError("student view must be outside the authoritative experiment")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"student view destination already exists: {target}")

        metadata_path = source / "experiment.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError("completed experiment metadata is unavailable")
        metadata = _strict_json_payload(metadata_path)
        if not isinstance(metadata, Mapping) or metadata.get("status") != "complete":
            raise ValueError("student view requires a completed authoritative experiment")
        settings_hash = metadata.get("settings_hash")
        if (
            not isinstance(settings_hash, str)
            or len(settings_hash) != 64
            or any(character not in "0123456789abcdef" for character in settings_hash)
        ):
            raise ValueError("authoritative configuration hash is malformed")
        children = _validated_children(source, metadata, validation_report)
        _verify_aggregate_manifest(source, children)
        for child_name in children:
            _verify_packet_manifest_exact(source / child_name)

        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        ).resolve(strict=False)
        ensure_within_runs(stage)
        copied_tables: list[Path] = []
        copied_plots: list[Path] = []
        try:
            for folder in (
                "02_PLOTS",
                "03_PRIMARY_DATA",
                "04_OPTIONAL_DIAGNOSTICS",
            ):
                (stage / folder).mkdir()

            for child_name in children:
                packet = (source / child_name).resolve(strict=False)
                csv_files = sorted(packet.glob("*.csv"), key=lambda path: path.name)
                available = {path.name for path in csv_files}
                missing = sorted(_REQUIRED_TABLES - available)
                if missing:
                    raise ValueError(
                        f"validated child packet is missing required student table: {missing[0]}"
                    )
                for saved in csv_files:
                    category = (
                        "03_PRIMARY_DATA"
                        if saved.name in _PRIMARY_TABLES
                        else "04_OPTIONAL_DIAGNOSTICS"
                    )
                    copied = stage / category / child_name / saved.name
                    _copy_saved(saved, copied, packet)
                    copied_tables.append(copied)

                plots = packet / "plots"
                if plots.exists():
                    if plots.is_symlink() or not plots.is_dir():
                        raise ValueError(f"unsafe plots directory in {child_name}")
                    for saved in sorted(
                        plots.rglob("*.png"),
                        key=lambda path: path.relative_to(plots).as_posix(),
                    ):
                        copied = (
                            stage
                            / "02_PLOTS"
                            / child_name
                            / saved.relative_to(plots)
                        )
                        _copy_saved(saved, copied, packet)
                        copied_plots.append(copied)

            tables = tuple(copied_tables)
            plots = tuple(copied_plots)
            (stage / "01_READ_ME_FIRST.md").write_text(
                _render_readme(
                    stage=stage,
                    source=source,
                    destination=target,
                    settings_hash=settings_hash,
                    copied_tables=tables,
                    copied_plots=plots,
                ),
                encoding="utf-8",
                newline="\n",
            )
            (stage / "DATA_DICTIONARY.md").write_text(
                _render_dictionary(stage, tables),
                encoding="utf-8",
                newline="\n",
            )
            _write_checksums(stage)
            _publish_staged_directory(stage, target)
        except Exception:
            if stage.exists():
                shutil.rmtree(stage)
            raise

    return StudentView(
        path=target,
        readme=target / "01_READ_ME_FIRST.md",
        plots=target / "02_PLOTS",
        primary_data=target / "03_PRIMARY_DATA",
        optional_diagnostics=target / "04_OPTIONAL_DIAGNOSTICS",
        data_dictionary=target / "DATA_DICTIONARY.md",
        authoritative_experiment=source,
    )

__all__ = ["StudentView", "create_student_view"]

"""One passive plan and one explicit executor for a flat scientific run."""

from __future__ import annotations

import numpy as np
import pandas as pd
from .settings import ExperimentSettings
from .storage import (
    RUN_SCHEMA,
    archive_source,
    environment_identity,
    hash_payload,
    json_clean,
    resolve_runs_path,
    seal,
    source_identity,
    write_csv_atomic,
    write_json_atomic,
)
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "eos_generation_plan_v2"

_TABLE_DESCRIPTIONS = {
    "cases": "Case identity, geometry, acceptance and exact rejection reasons",
    "raw": "Complete raw proposals: window, Gaussian, sound-speed and pressure changes",
    "eos": "Accepted reconstructed effective state and changes relative to the selected baseline",
    "stars": "Stellar sequence attempts, central conditions and background/tidal statuses",
    "fixed_mass": "Requested masses, true brackets, solved observables and failure reasons",
    "maximum_mass": "Resolved turning points, endpoint limitations and refinement evidence",
    "radial_profiles": "Retained fixed-mass stellar profiles at the final stage",
    "deformation_support_fractions": "Radial and enclosed-mass support of the deformation",
    "baryonic_observables": "Absolute baryon number, baryonic mass and binding energy",
    "baryonic_response_across_mass": "Baryonic mass and binding-energy changes relative to the selected baseline",
    "stellar_response_across_mass": "Radius, Love-number and tidal changes on common mass support",
    "odd_even_response": "Paired positive/negative amplitude responses with a zero control",
    "numerical_error_summary": "Same-case fixed-mass spreads across saved stellar stages",
}


def _notebook_option_tables(settings):
    """Build discovery tables from the governed profiles and saved-plot registry."""
    from .numerics import precision_profile
    from .settings import _PRECISIONS
    from .plotting import FIGURES, LABELS

    purposes = {
        "quick": "Exploratory pilot; does not establish convergence",
        "strict": "Compare sampling and ODE refinement across three stellar stages",
        "dataset": "61-point sequence at the governed tight ODE tolerances",
        "dataset_10_tighter": "10-point sequence with tighter ODE tolerances",
        "dataset_20": "20-point sequence at tight ODE tolerances",
        "dataset_40": "40-point sequence with three thermodynamic stages",
        "dataset_40_curves": "40-point curves with only the final thermodynamic stage",
        "dataset_relaxed": "61-point sequence with relaxed ODE tolerances",
        "dataset_relaxed_80": "80-point sequence with relaxed ODE tolerances",
    }
    profiles = []
    for name in _PRECISIONS:
        profile = precision_profile(name, "stellar")
        stages = profile["tov_stages"]
        profiles.append(
            {
                "precision": name,
                "purpose": purposes[name],
                "thermodynamic_stages": len(profile["thermodynamic_stages"]),
                "stellar_stages": ", ".join(s.name for s in stages),
                "sequence_attempts_per_case": sum(s.sequence_points for s in stages),
                "ODE_tolerances": "; ".join(
                    f"{s.name}: rtol={s.rtol:g}, atol={s.atol:g}" for s in stages
                ),
                "default_products": ", ".join(
                    ExperimentSettings.from_values(
                        calculation="stellar", precision=name
                    ).requested_observables
                ),
            }
        )
    requirements = {
        "eos": "Thermodynamics; accepted reconstructed EoS",
        "stars": "stellar + sequence; successful points and validated tides where needed",
        "fixed_mass": "stellar + fixed_mass; solved brackets and validated tides where needed",
        "maximum_mass": "stellar + maximum_mass; resolved turning point",
        "radial_profiles": "diagnostics on; retained final-stage fixed-mass profiles",
        "deformation_support_fractions": "diagnostics on; retained final-stage fixed-mass profiles",
        "baryonic_response_across_mass": "diagnostics on; common successful fixed masses with baseline",
        "stellar_response_across_mass": "diagnostics on; common successful mass support with baseline",
    }
    plots = pd.DataFrame(
        [
            {
                "key": key,
                "plot": LABELS[key],
                "requires": requirements[spec[0]],
                "horizontal_axis": spec[3],
                "vertical_axis": spec[4],
            }
            for key, spec in FIGURES.items()
        ]
    )
    products = pd.DataFrame(
        [
            {"product": label, "requested": enabled, "meaning": meaning}
            for label, enabled, meaning in (
                (
                    "thermodynamics",
                    True,
                    "Assess raw proposals and reconstruct accepted EoS",
                ),
                (
                    "sequence",
                    "sequence" in settings.requested_observables,
                    "M-R and available tidal curves",
                ),
                (
                    "fixed_mass",
                    "fixed_mass" in settings.requested_observables,
                    "Solve the requested gravitational masses",
                ),
                (
                    "maximum_mass",
                    "maximum_mass" in settings.requested_observables,
                    "Bracket and refine a turning point",
                ),
                (
                    "diagnostics",
                    settings.diagnostics == "on",
                    "Retain governed radial, baryonic and response evidence",
                ),
            )
        ]
    )
    return {
        "profiles": pd.DataFrame(profiles),
        "plots": plots,
        "products": products,
        "tables": pd.DataFrame(
            [
                {"table": name, "meaning": description}
                for name, description in _TABLE_DESCRIPTIONS.items()
            ]
        ),
    }


def _notebook_details(title, frame):
    """Keep the discovery catalogues compact in the single-cell output."""
    from html import escape
    from IPython.display import HTML, display

    display(
        HTML(
            f"<details><summary>{escape(title)}</summary>"
            f"{frame.to_html(index=False, escape=True)}</details>"
        )
    )


def _notebook_display_options(plots, tables, table_rows, *, view=None):
    """Validate presentation controls before any requested execution."""
    from .plotting import _names, _view_options

    _view_options(**(view or {}))

    if not (isinstance(plots, str) and plots in ("auto", "none")):
        _names(plots)
    if (
        isinstance(table_rows, bool)
        or not isinstance(table_rows, int)
        or not 1 <= table_rows <= 200
    ):
        raise ValueError("TABLE_ROWS must be an integer from 1 to 200")
    if isinstance(tables, str) and tables not in ("auto", "all", "none"):
        raise ValueError(
            "TABLES must be 'auto', 'all', 'none', or a list of table names"
        )
    names = (
        list(_TABLE_DESCRIPTIONS)
        if tables == "all"
        else (
            ["cases", "fixed_mass", "maximum_mass", *list(_TABLE_DESCRIPTIONS)[6:]]
            if tables == "auto"
            else [] if tables == "none" else list(tables)
        )
    )
    if any(name not in _TABLE_DESCRIPTIONS for name in names):
        raise ValueError("Choose TABLES from " + ", ".join(_TABLE_DESCRIPTIONS))
    return names


def _notebook_inspect(
    result, *, selection, plots="auto", tables="auto", table_rows=12, regenerate=False,
    view=None,
):
    """Display saved evidence, selections and figures without scientific work."""
    from IPython.display import Image, display
    from .storage import strict_json

    names = _notebook_display_options(plots, tables, table_rows, view=view)
    report = validate_experiment(result.experiment_path)
    if not report["passed"]:
        raise ValueError("Saved run failed validation: " + "; ".join(report["errors"]))
    print("Viewing saved run:", result.experiment_path)
    print("Saved settings are authoritative for this view:")
    display(result.settings.to_dict())
    display(
        pd.DataFrame(
            [
                {
                    "validation": "passed",
                    "source_equivalence": report["source_equivalence"],
                    "source_archive_available": report[
                        "scientific_output_availability"
                    ].get("source_archive_available", False),
                }
            ]
        )
    )
    if report["source_equivalence"] != "equivalent":
        print(
            "Saved evidence is intact; current source differs. Exact replay requires the recorded source archive and runtime."
        )
    print("Saved output status counts:")
    display(report["scientific_output_availability"])
    cases = result.case_table
    geometries = cases[
        [
            "geometry_index",
            "epsilon0_mev_fm3",
            "sigma_mev_fm3",
            "delta_mev_fm3",
            "epsilon_match_mev_fm3",
        ]
    ].drop_duplicates()
    print("Available geometry indices (energy-density parameters in MeV fm^-3):")
    display(geometries)
    stages = [
        s["name"] for s in result.metadata["numerical_profile"].get("tov_stages", [])
    ]
    print("Saved amplitudes:", sorted(cases.amplitude.unique().tolist()))
    print("Saved fixed-mass targets (M_sun):", list(result.settings.fixed_masses))
    print("Valid STAGE choices:", ["final", *stages])
    print("Diagnostic scope:", result.metadata.get("diagnostics", "off"))
    inventory = []
    frames = {}
    for name, meaning in _TABLE_DESCRIPTIONS.items():
        saved = (result.data_path / (name + ".csv")).is_file()
        inventory.append({"table": name, "saved": saved, "meaning": meaning})
        if name in names and saved:
            frame = result.table(name)
            frames[name] = frame
            print(
                f"{name}: showing the first {min(table_rows, len(frame))} of {len(frame)} saved rows"
            )
            print("Columns:", ", ".join(frame.columns))
            display(frame.head(table_rows))
    _notebook_details(
        "Saved table inventory; use TABLES to choose previews", pd.DataFrame(inventory)
    )
    availability = result.available_plots(**selection)
    print("Plot availability for the selected saved values:")
    display(pd.DataFrame(availability["plots"]))
    if plots == "none":
        print("Plotting disabled. No scientific calculation was requested.")
        return frames
    folder = result.generate_plots(
        figures=plots, regenerate=regenerate, **selection, **(view or {})
    )
    record = strict_json(folder / "figures.json")
    print("Figure folder:", folder)
    print("Combined plots: geometry colors; line styles distinguish curve variants. Shared zero control is black.")
    print("Λ display settings:", record["request"]["view"])
    color_key = geometries.loc[
        geometries.geometry_index.astype(str).isin(record["geometry_colors"])
    ].copy()
    color_key["plot_color"] = color_key.geometry_index.astype(str).map(record["geometry_colors"])
    _notebook_details("Geometry color key for these figures", color_key)
    if record["skipped"]:
        display(record["skipped"])
    for name in record["figures"]:
        display(Image(filename=str(folder / name)))
    return frames


def _new_run_destination(settings, runs_directory, study="study") -> Path:
    """Choose a fresh destination without creating anything; bind it in a plan."""
    import re
    from uuid import uuid4

    if not isinstance(study, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", study):
        raise ValueError(
            "study name must contain 1–60 letters, numbers, underscores or hyphens"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return resolve_runs_path(
        Path(runs_directory)
        / f"run-{study}-{stamp}-{settings.deterministic_hash()[:8]}-{uuid4().hex[:6]}"
    )


def _saved_runs(runs_directory) -> pd.DataFrame:
    """List complete, failed and interrupted run folders without calculation."""
    from .storage import strict_json

    rows = []
    for folder in sorted(Path(runs_directory).glob("*")):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        try:
            metadata = strict_json(folder / "data/run.json")
            settings = metadata.get("settings", {})
            status = metadata.get("status", "incomplete")
            detail = metadata.get("failure", {}).get("message", "")
            changed = (folder / "data/run.json").stat().st_mtime
        except (OSError, ValueError, TypeError, AttributeError):
            metadata = {}
            settings, status, detail, changed = (
                {},
                "incomplete",
                "Saved run metadata is missing or unreadable",
                folder.stat().st_mtime,
            )
        rows.append(
            {
                "run": folder.name,
                "path": str(folder.resolve()),
                "status": status,
                "matter_model": settings.get("matter_model", "bsk24") if settings else "unknown",
                "calculation": settings.get("calculation", "unknown"),
                "precision": settings.get("precision", "unknown"),
                "amplitudes": str(settings.get("amplitudes", "unknown")),
                "geometry": "; ".join(
                    f"{key}={settings.get(key, '?')}"
                    for key in ("center", "width", "ramp_width")
                ),
                "observables": ", ".join(metadata.get("requested_observables", [])),
                "diagnostics": settings.get("diagnostics", "unknown"),
                "detail": detail,
                "modified": changed,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "run",
            "path",
            "status",
            "matter_model",
            "calculation",
            "precision",
            "amplitudes",
            "geometry",
            "observables",
            "diagnostics",
            "detail",
            "modified",
        ],
    )


def _load_saved_run(runs_directory, selection="latest"):
    """Load a selected saved run; a missing/corrupt run never falls back silently."""
    root = Path(runs_directory).resolve()
    if selection == "latest":
        candidates = _saved_runs(root)
        candidates = candidates.loc[candidates.status.eq("complete")]
        if candidates.empty:
            raise ValueError(
                "No completed saved runs are available. Review a new plan and execute it first; failed/incomplete runs cannot be plotted."
            )
        path = Path(candidates.sort_values("modified").iloc[-1].path)
    else:
        path = (root / selection).resolve()
        if path.parent != root:
            raise ValueError("Select a run folder name from the saved-run list")
    try:
        return load_experiment(path)
    except ValueError as exc:
        raise ValueError(
            f"Cannot load {path.name}: {exc}. Select another completed run or execute a new reviewed plan; no calculation was started."
        ) from exc


def _notebook_environment() -> dict:
    """Read the installed runtime contract; render checks belong to plotting."""
    import sys
    import tomllib
    from importlib.metadata import version

    contract = Path(__file__).parent / "runtime_contracts/pyproject.toml"
    expected = tomllib.loads(contract.read_text(encoding="utf-8"))["project"]
    installed = environment_identity()
    issues = []
    for dependency in expected["dependencies"]:
        if "==" in dependency:
            name, pin = dependency.split("==", 1)
            if installed.get(name) != pin:
                issues.append(f"{name}: expected {pin}, found {installed.get(name)}")
    if not installed["python"].startswith("3.12."):
        issues.append("The declared runtime requires Python 3.12")
    return {
        "python": sys.executable,
        "package_version": version("eos-generation"),
        "package_source": str(Path(__file__).parent),
        "runtime_issues": issues,
    }


def _execute_notebook_plan(plan, settings):
    """Execute a reviewed plan once; repeating its cell loads its intact result."""
    if not isinstance(plan, ExperimentPlan):
        raise ValueError(
            "Choose ACTION='plan' and review its output before choosing 'execute'"
        )
    fresh = plan_experiment(settings, output_path=plan.experiment_path)
    if fresh.to_dict() != plan.to_dict():
        raise ValueError(
            "Settings, source, environment or workers changed. Choose ACTION='plan' and review a fresh plan before execution."
        )
    if plan.experiment_path.exists():
        try:
            result = load_experiment(plan.experiment_path)
        except ValueError as exc:
            raise ValueError(
                "This destination contains an incomplete, failed or changed run. Choose ACTION='plan' for a new destination; existing evidence was preserved."
            ) from exc
        if result.metadata.get("plan_hash") != plan.plan_hash:
            raise ValueError(
                "This destination belongs to another plan. Choose ACTION='plan' to start a new run."
            )
        return result
    return run_experiment(plan, execute=True)


def _configs(settings: ExperimentSettings):
    from .numerics import BSk24TrialConfig, precision_profile

    profile = precision_profile(settings.precision, settings.calculation)
    anchor = None if settings.epsilon_match == "standard" else settings.epsilon_match
    geometries = list(product(settings.center, settings.width, settings.ramp_width))
    owner = min(geometries)
    return tuple(
        BSk24TrialConfig(
            matter_model=settings.matter_model,
            amplitudes=settings.amplitudes,
            epsilon_match_mev_fm3=anchor,
            epsilon0_mev_fm3=center,
            sigma_mev_fm3=width,
            deltas_mev_fm3=(ramp,),
            fixed_masses_msun=settings.fixed_masses,
            thermodynamic_stages=profile["thermodynamic_stages"],
            tov_stages=profile["tov_stages"],
            raw_gate_lower_points=profile["raw_gate_lower_points"],
            raw_gate_upper_points=profile["raw_gate_upper_points"],
            maximum_mass_initial_points=profile["maximum_mass_initial_points"],
            stellar_enabled=settings.calculation == "stellar",
            requested_observables=settings.requested_observables,
            extended_stellar_diagnostics_enabled=settings.diagnostics == "on",
            diagnostic_delta_mev_fm3=ramp,
            zero_amplitude_control_owner=(center, width, ramp) == owner,
        )
        for center, width, ramp in geometries
    )


def _cases(configs) -> list[dict[str, Any]]:
    from .numerics import deterministic_case_id

    rows = []
    seen = set()
    for index, config in enumerate(configs, 1):
        for amplitude in config.logical_amplitudes:
            logical_id = deterministic_case_id(
                amplitude=amplitude,
                delta_mev_fm3=config.deltas_mev_fm3[0],
                epsilon0_mev_fm3=config.epsilon0_mev_fm3,
                sigma_mev_fm3=config.sigma_mev_fm3,
                epsilon_match_mev_fm3=config.epsilon_match_mev_fm3,
                matter_model=config.matter_model,
            )
            physical_id = (
                config.zero_amplitude_physical_case_id if amplitude == 0 else logical_id
            )
            execute = physical_id not in seen and (
                amplitude != 0 or config.zero_amplitude_control_owner
            )
            if execute:
                seen.add(physical_id)
            rows.append(
                {
                    "geometry_index": index,
                    "case_id": logical_id,
                    "physical_case_id": physical_id,
                    "amplitude": amplitude,
                    "epsilon0_mev_fm3": config.epsilon0_mev_fm3,
                    "sigma_mev_fm3": config.sigma_mev_fm3,
                    "delta_mev_fm3": config.deltas_mev_fm3[0],
                    "epsilon_match_mev_fm3": config.effective_epsilon_match_mev_fm3,
                    "planned_for_execution": execute,
                    "is_alias": amplitude == 0 and not execute,
                }
            )
    return rows


@dataclass(frozen=True)
class ExperimentPlan:
    settings: ExperimentSettings
    experiment_path: Path
    plan_hash: str
    _document: str

    def to_dict(self) -> dict[str, Any]:
        import json

        return json.loads(self._document)

    @property
    def case_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict()["cases"])

    @property
    def estimates(self) -> dict[str, int]:
        return self.to_dict()["estimates"]

    def summary_text(self) -> str:
        d = self.to_dict()
        lines = [
            f"BSk{self.settings.matter_model[-2:]} experiment plan",
            f"Plan hash: {self.plan_hash}",
            f"Destination: {self.experiment_path}",
            f"Calculation: {self.settings.calculation}; precision: {self.settings.precision}",
            "Requested observables: "
            + (", ".join(self.settings.requested_observables) or "thermodynamics"),
            "Planning is passive: 0 solver calls, 0 filesystem writes",
        ]
        lines.extend(f"{name}: {value}" for name, value in d["estimates"].items())
        lines.append(
            "Thermodynamic stages: "
            + ", ".join(
                f"{x['name']} ({x['lower_points']}/{x['upper_points']})"
                for x in d["numerical_profile"]["thermodynamic_stages"]
            )
        )
        profile = d["numerical_profile"]
        lines.append(
            "Stellar stages: "
            + (
                ", ".join(
                    f"{stage['name']} ({stage['sequence_points']} pressures, rtol={stage['rtol']:g}, atol={stage['atol']:g}, radial points={stage['radial_profile_points']})"
                    for stage in profile["tov_stages"]
                )
                or "disabled"
            )
        )
        lines.append(
            f"Raw-gate grids: {profile['raw_gate_lower_points']}/{profile['raw_gate_upper_points']}"
        )
        lines.append("Execution controls: " + str(d["execution_controls"]))
        lines.append(
            "Stellar root and maximum-mass calls are adaptive; the declared pressure count is not a total-call budget."
        )
        return "\n".join(lines)


def plan_experiment(
    settings: ExperimentSettings, *, output_path: str | Path | None = None
) -> ExperimentPlan:
    """Validate and expand settings without scientific calls or writes."""
    import json
    from .numerics import precision_profile
    from .stellar import (
        _automatic_stellar_worker_count,
        _automatic_sequence_worker_count,
    )

    if not isinstance(settings, ExperimentSettings):
        raise TypeError("settings must be ExperimentSettings")
    destination = resolve_runs_path(
        output_path
        or Path.cwd() / "runs" / ("experiment_" + settings.deterministic_hash()[:12])
    )
    configs = _configs(settings)
    cases = _cases(configs)
    profile = json_clean(precision_profile(settings.precision, settings.calculation))
    # Dataclasses in profiles have one serialization authority.
    for key in ("thermodynamic_stages", "tov_stages"):
        profile[key] = [asdict(stage) for stage in getattr(configs[0], key)]
    physical_count = sum(row["planned_for_execution"] for row in cases)
    stellar = settings.calculation == "stellar"
    case_workers = _automatic_stellar_worker_count(physical_count) if stellar else 1
    document = {
        "schema_id": PLAN_SCHEMA,
        "settings": settings.to_dict(),
        "settings_hash": settings.deterministic_hash(),
        "experiment_path": destination.as_posix(),
        "numerical_profile": profile,
        "requested_observables": list(settings.requested_observables),
        "diagnostics": settings.diagnostics,
        "execution_controls": {
            "central_pressure_min_mev_fm3": configs[0].central_pressure_min_mev_fm3,
            "fixed_mass_root_xtol_mev_fm3": configs[0].fixed_mass_root_xtol_mev_fm3,
            "maximum_mass_threshold_msun": configs[0].maximum_mass_threshold_msun,
            "diagnostics_case_policy": configs[
                0
            ].extended_stellar_diagnostics_case_policy,
            "worker_count": case_workers,
            "standalone_sequence_workers_by_stage": {
                stage.name: (
                    _automatic_sequence_worker_count(stage.sequence_points)
                    if case_workers == 1
                    else 1
                )
                for stage in configs[0].tov_stages
            },
        },
        "cases": cases,
        "source_identity": source_identity(),
        "environment_identity": environment_identity(),
        "estimates": {
            "geometry_count": len(configs),
            "logical_case_count": len(cases),
            "physical_case_count": physical_count,
            "baseline_constructions": len(configs[0].thermodynamic_stages),
            "raw_gate_cases": physical_count,
            "maximum_sequence_pressure_attempts": physical_count
            * sum(s.sequence_points for s in configs[0].tov_stages),
            "fixed_mass_targets": (
                physical_count * len(configs[0].tov_stages) * len(settings.fixed_masses)
                if "fixed_mass" in settings.requested_observables
                else 0
            ),
        },
        "planning_is_passive": True,
        "scientific_solver_calls": 0,
        "filesystem_writes": 0,
    }
    document["environment_sha256"] = hash_payload(document["environment_identity"])
    digest = hash_payload(document)
    document["plan_hash"] = digest
    return ExperimentPlan(
        settings,
        destination,
        digest,
        json.dumps(document, sort_keys=True, allow_nan=False),
    )


@dataclass(frozen=True)
class ExperimentResult:
    experiment_path: Path
    metadata: dict[str, Any]

    @property
    def data_path(self) -> Path:
        return self.experiment_path / "data"

    @property
    def settings(self) -> ExperimentSettings:
        return ExperimentSettings.from_dict(self.metadata["settings"])

    def table(self, name: str) -> pd.DataFrame:
        if name not in _TABLE_DESCRIPTIONS:
            raise ValueError(f"unknown table {name!r}")
        path = self.data_path / (name + ".csv")
        return (
            pd.read_csv(path, float_precision="round_trip")
            if path.is_file()
            else pd.DataFrame()
        )

    @property
    def case_table(self):
        return self.table("cases")

    @property
    def thermodynamic_profiles(self):
        return self.table("eos")

    @property
    def stellar_sequences(self):
        return self.table("stars")

    @property
    def fixed_mass_results(self):
        return self.table("fixed_mass")

    def available_plots(self, **selection):
        from .plotting import available_figures

        return available_figures(self.experiment_path, **selection)

    def generate_plots(
        self, *, figures="auto", output_path=None, regenerate=False, **selection
    ):
        from .plotting import plot_experiment

        return plot_experiment(
            self.experiment_path,
            figures=figures,
            output_path=output_path,
            regenerate=regenerate,
            **selection,
        )


@dataclass(frozen=True)
class Experiment:
    settings: ExperimentSettings

    def plan(self, *, output_path=None):
        return plan_experiment(self.settings, output_path=output_path)


def _outcomes(rows, reports):
    result = []
    for row in rows:
        report = reports.get(row["physical_case_id"], {})
        status = report.get("status", "pending")
        label = {
            "accepted_raw_local_physics_gate": "accepted",
            "rejected_raw_local_physics_gate": "rejected",
            "unresolved_raw_local_physics_gate": "unresolved",
        }.get(status, "pending")
        failure = report.get("first_failure") or {}
        result.append(
            {
                **row,
                "status": label,
                "failure_reason": failure.get("reason", ""),
                "retained_endpoint_epsilon_mev_fm3": (
                    report.get("retained_domain") or {}
                ).get("epsilon_max_mev_fm3"),
            }
        )
    return pd.DataFrame(result)


def run_experiment(plan: ExperimentPlan, *, execute: bool = False) -> ExperimentResult:
    """Execute exactly a fresh reviewed plan into a new, no-overwrite run."""
    if not isinstance(plan, ExperimentPlan):
        raise TypeError("execution requires an ExperimentPlan")
    if execute is not True:
        raise ValueError("execution requires execute=True after reviewing the plan")
    fresh = plan_experiment(plan.settings, output_path=plan.experiment_path)
    if fresh.plan_hash != plan.plan_hash or fresh.to_dict() != plan.to_dict():
        raise ValueError(
            "settings, source, environment, workers or destination changed; review a fresh plan"
        )
    document = plan.to_dict()
    destination = plan.experiment_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    data = destination / "data"
    data.mkdir()
    record = {
        "schema_id": RUN_SCHEMA,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "settings": plan.settings.to_dict(),
        "settings_hash": document["settings_hash"],
        "plan_hash": plan.plan_hash,
        "numerical_profile": document["numerical_profile"],
        "execution_controls": document["execution_controls"],
        "requested_observables": document["requested_observables"],
        "source_identity": document["source_identity"],
        "environment_identity": document["environment_identity"],
        "environment_sha256": document["environment_sha256"],
        "units": {
            "epsilon": "MeV fm^-3 including rest mass",
            "pressure": "MeV fm^-3",
            "cs2": "dimensionless, c=1",
            "mass": "gravitational solar masses",
            "n_B": "fm^-3",
            "mu_B": "MeV",
        },
        "interpretation": "effective one-fluid cold barotrope; microscopic composition and beta equilibrium are not established",
        "certificates": {},
        "raw_gate_before_downstream": True,
        "reproduction": {
            "settings_file": "data/run.json (settings field)",
            "plan": "bsk24-trial plan --config data/run.json --output runs/reproduction",
            "run": "bsk24-trial run --config data/run.json --output runs/reproduction --plan-hash <fresh reviewed plan hash> --execute",
        },
    }
    write_json_atomic(record, data / "run.json")
    reports = {}
    try:
        from .baseline import make_baseline_eos
        from .assessment import raw_local_physics_gate
        from .deformation import BSk24WindowedDeformation
        from .diagnostics import (
            _raw_gate_frame,
            _thermodynamic_profile_frame,
            _thermodynamic_convergence,
            windowed_a0_identity_report,
            write_diagnostics,
        )
        from .thermodynamics import (
            build_consistent_baseline,
            build_windowed_eos,
            BSk24MechanicalStabilityError,
        )
        from .stellar import _run_stellar

        record["source_archive"] = archive_source(
            destination, document["source_identity"]
        )
        configs = _configs(plan.settings)
        config = configs[0]
        stages = {
            stage.name: build_consistent_baseline(
                stage.grid_settings(),
                eos=make_baseline_eos(plan.settings.matter_model),
                **(
                    {"anchor_energy_density_mev_fm3": config.epsilon_match_mev_fm3}
                    if config.epsilon_match_mev_fm3 is not None
                    else {}
                ),
            )
            for stage in config.thermodynamic_stages
        }
        baseline = stages[config.thermodynamic_stages[-1].name]
        record["baseline"] = {
            "anchor": baseline.anchor.to_dict(),
            "diagnostics": baseline.diagnostics,
            **({"provenance": baseline.eos.provenance()} if plan.settings.matter_model == "bsk25" else {}),
        }
        rows = document["cases"]
        deformations = {
            row["physical_case_id"]: BSk24WindowedDeformation(
                row["physical_case_id"],
                row["amplitude"],
                row["epsilon0_mev_fm3"],
                row["sigma_mev_fm3"],
                row["delta_mev_fm3"],
            )
            for row in rows
            if row["planned_for_execution"]
        }
        raw = []
        # Complete raw evidence for every proposal precedes all reconstruction.
        for case_id, deformation in deformations.items():
            report, epsilon, cs2 = raw_local_physics_gate(
                baseline,
                deformation,
                dense_lower_points=config.raw_gate_lower_points,
                dense_upper_points=config.raw_gate_upper_points,
            )
            if report.get("status") not in {
                "accepted_raw_local_physics_gate",
                "rejected_raw_local_physics_gate",
                "unresolved_raw_local_physics_gate",
            }:
                raise ValueError("unknown raw-gate status")
            reports[case_id] = report
            raw.append(
                _raw_gate_frame(
                    case_id=case_id,
                    deformation=deformation,
                    baseline=baseline,
                    epsilon=np.asarray(epsilon),
                    raw_cs2=np.asarray(cs2),
                    status=report["status"],
                )
            )
        raw_frame = pd.concat(raw, ignore_index=True)
        record["certificates"]["raw_gate"] = reports
        write_csv_atomic(raw_frame, data / "raw.csv")
        write_csv_atomic(_outcomes(rows, reports), data / "cases.csv")
        write_json_atomic(record, data / "run.json")
        stage_cases = {name: {} for name in stages}
        for case_id, deformation in deformations.items():
            if reports[case_id]["status"] != "accepted_raw_local_physics_gate":
                continue
            try:
                built = {
                    name: build_windowed_eos(
                        base, deformation, raw_gate_report=reports[case_id]
                    )
                    for name, base in stages.items()
                }
            except BSk24MechanicalStabilityError as exc:
                if exc.diagnostics.get("status") != "unresolved_tabulation_resolution":
                    raise
                report = reports[case_id]
                report.update(
                    status="unresolved_raw_local_physics_gate",
                    selected_retained_domain_passed=False,
                    pre_reconstruction_tabulation_resolution=exc.diagnostics,
                    first_failure={
                        "reason": "unresolved_tabulation_resolution",
                        "detail": exc.diagnostics,
                    },
                )
                report["retained_domain"].update(
                    passed=False, resolution_certified=False
                )
                raw_frame.loc[raw_frame.case_id.eq(case_id), "gate_status"] = report[
                    "status"
                ]
                continue
            for name, eos in built.items():
                stage_cases[name][case_id] = eos
        generated = stage_cases[config.thermodynamic_stages[-1].name]
        a0 = config.zero_amplitude_physical_case_id
        if a0 not in generated:
            raise ValueError("the zero-amplitude identity control was not accepted")
        record["certificates"]["a0_identity"] = windowed_a0_identity_report(
            baseline, {generated[a0].deformation.delta_mev_fm3: generated[a0]}
        )
        if record["certificates"]["a0_identity"]["status"] != "pass":
            raise ValueError("zero-amplitude identity failed")
        record["certificates"]["thermodynamic_convergence"] = (
            _thermodynamic_convergence(stage_cases)
        )
        record["certificates"]["reconstruction"] = {
            case_id: {
                name: {
                    key: value
                    for key, value in eos.diagnostics.items()
                    if key != "raw_gate_report"
                }
                for name, eos in (
                    (stage, stage_cases[stage][case_id]) for stage in stages
                )
            }
            for case_id in generated
        }
        eos_frame = _thermodynamic_profile_frame(baseline, generated)
        eos_frame = eos_frame.loc[~eos_frame.case_id.eq("direct")].reset_index(
            drop=True
        )
        write_csv_atomic(eos_frame, data / "eos.csv")
        write_csv_atomic(raw_frame, data / "raw.csv")
        write_csv_atomic(_outcomes(rows, reports), data / "cases.csv")
        write_json_atomic(record, data / "run.json")
        if config.stellar_enabled:
            from dataclasses import replace

            # The direct analytical control supplies the historical stellar
            # baseline path once; logical controls reference its physical ID.
            stellar_config = replace(config, zero_amplitude_control_owner=True)
            sequences, fixed, convergence, stars = _run_stellar(
                config=stellar_config,
                baseline=baseline,
                generated={key: eos for key, eos in generated.items() if key != a0},
            )
            maximum = pd.DataFrame(convergence.pop("maximum_mass_rows"))
            for frame in (sequences, fixed, maximum):
                if "case_id" in frame:
                    frame.loc[frame.case_id.eq("direct"), "case_id"] = a0
            reports_max = convergence.get("maximum_mass_reports", {})
            convergence["maximum_mass_reports"] = {
                key.replace("direct:", a0 + ":", 1): value
                for key, value in reports_max.items()
            }
            convergence["sequence_evidence"] = {
                key.replace("direct:", a0 + ":", 1): value
                for key, value in convergence.get("sequence_evidence", {}).items()
            }
            for key in ("worker_process_ids", "case_worker_wall_seconds"):
                convergence.get("parallel_execution", {}).pop(key, None)
            record["certificates"]["stellar"] = convergence
            write_csv_atomic(sequences, data / "stars.csv")
            if "fixed_mass" in plan.settings.requested_observables:
                write_csv_atomic(fixed, data / "fixed_mass.csv")
            if "maximum_mass" in plan.settings.requested_observables:
                write_csv_atomic(maximum, data / "maximum_mass.csv")
            if plan.settings.diagnostics == "on":
                record["certificates"]["diagnostics"] = write_diagnostics(
                    packet=data,
                    configs=configs,
                    baseline=baseline,
                    generated=generated,
                    sequences=sequences,
                    fixed=fixed,
                    stars=stars,
                    baseline_id=a0,
                )
        record.update(
            status="complete", completed_utc=datetime.now(timezone.utc).isoformat()
        )
        write_json_atomic(record, data / "run.json")
        seal(data)
        # Integrity/scientific validation remains read-only and source drift
        # does not prevent loading an intact saved run.
        return load_experiment(destination)
    except BaseException as exc:
        record.update(
            status="failed",
            failure={"type": type(exc).__name__, "message": str(exc)},
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
        record["certificates"]["raw_gate"] = reports
        write_json_atomic(record, data / "run.json")
        seal(data)
        raise


def load_experiment(path: str | Path) -> ExperimentResult:
    from .storage import strict_json, validate_run

    destination = Path(path).resolve()
    report = validate_run(destination)
    if not report["passed"]:
        raise ValueError("invalid run: " + "; ".join(report["errors"]))
    return ExperimentResult(destination, strict_json(destination / "data/run.json"))


def validate_experiment(path: str | Path) -> dict[str, Any]:
    from .storage import validate_run

    return validate_run(Path(path).resolve())

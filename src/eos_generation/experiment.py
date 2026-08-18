"""Small public interface for controlled analytical BSk24 experiments.

The public settings deliberately describe scientific intent rather than the
internal numerical machinery.  ``plan_experiment`` expands ``quick`` or
``strict`` into the governed numerical stages and remains calculation-free
and write-free.  ``run_experiment`` accepts only an exact reviewed plan and an
explicit execution gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ._experiment_integrity import (
    AGGREGATE_MANIFEST,
    _AGGREGATE_DOCUMENTS,
    _aggregate_manifest_files,
    _verify_aggregate_manifest,
    _write_aggregate_manifest,
)
from ._experiment_io import (
    _PLAN_DEPENDENCIES,
    _active_runtime_identity,
    _active_source_identity,
    _canonical_json,
    _finite_float,
    _hash_payload,
    _number_tuple,
    _owning_repository_root,
    _portable_path,
    _resolve_portable_path,
    _sha256_file,
    _strict_json_object,
    _write_json_atomic,
)
from ._experiment_planning import (
    PLAN_SCHEMA,
    _internal_configs,
    _plan_child_document,
    _plan_digest,
    _precision_profile,
    _saved_plan_digest,
)


EXPERIMENT_SCHEMA = "eos_generation_experiment_v1"
REPRODUCTION_PLAN_SCHEMA = "eos_generation_reproduction_plan_v1"
CONFIG_SCHEMA_URL = (
    "https://raw.githubusercontent.com/PapathanasiouIoannis/"
    "EoS-generation/main/configs/schema.json"
)
_CALCULATIONS = ("thermodynamics", "stellar")
_PRECISIONS = ("quick", "strict")
_DIAGNOSTICS = ("off", "on")
_MAX_GEOMETRIES = 256
_MAX_EXPANDED_CASES = 4096
_MAX_FIXED_MASSES = 32


@dataclass(frozen=True)
class ExperimentSettings:
    """User-facing scientific choices for one experiment.

    Geometry values may be scalars or small sequences.  Sequences are expanded
    as an explicit Cartesian product during passive planning; they are not a
    hidden campaign mode.
    """

    amplitudes: tuple[float, ...] = (0.0, 0.01)
    epsilon_match: str | float = "standard"
    center: tuple[float, ...] = (200.0,)
    width: tuple[float, ...] = (50.0,)
    ramp_width: tuple[float, ...] = (40.0,)
    calculation: str = "thermodynamics"
    precision: str = "quick"
    fixed_masses: tuple[float, ...] = (1.4,)
    diagnostics: str = "off"

    def __post_init__(self) -> None:
        object.__setattr__(self, "amplitudes", _number_tuple("amplitudes", self.amplitudes))
        object.__setattr__(self, "center", _number_tuple("center", self.center, positive=True))
        object.__setattr__(self, "width", _number_tuple("width", self.width, positive=True))
        object.__setattr__(
            self,
            "ramp_width",
            _number_tuple("ramp_width", self.ramp_width, positive=True),
        )
        masses = _number_tuple("fixed_masses", self.fixed_masses, positive=True)
        if any(value >= 10.0 for value in masses):
            raise ValueError("fixed_masses must be below 10 solar masses")
        if len(masses) > _MAX_FIXED_MASSES:
            raise ValueError(
                f"fixed_masses may contain at most {_MAX_FIXED_MASSES} targets"
            )
        object.__setattr__(self, "fixed_masses", masses)
        geometry_count = len(self.center) * len(self.width) * len(self.ramp_width)
        if geometry_count > _MAX_GEOMETRIES:
            raise ValueError(
                f"settings expand to {geometry_count} geometries; the public "
                f"planning limit is {_MAX_GEOMETRIES}"
            )
        amplitude_count = len(self.amplitudes) + (
            0 if any(value == 0.0 for value in self.amplitudes) else 1
        )
        expanded_cases = geometry_count * amplitude_count
        if expanded_cases > _MAX_EXPANDED_CASES:
            raise ValueError(
                f"settings expand to {expanded_cases} cases including the zero "
                f"control; the public planning limit is {_MAX_EXPANDED_CASES}"
            )
        if self.epsilon_match != "standard":
            object.__setattr__(
                self,
                "epsilon_match",
                _finite_float("epsilon_match", self.epsilon_match, positive=True),
            )
        if self.calculation not in _CALCULATIONS:
            raise ValueError(f"calculation must be one of {_CALCULATIONS}")
        if self.precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {_PRECISIONS}")
        if self.diagnostics not in _DIAGNOSTICS:
            raise ValueError(f"diagnostics must be one of {_DIAGNOSTICS}")
        if self.diagnostics == "on" and self.calculation != "stellar":
            raise ValueError("diagnostics='on' requires calculation='stellar'")

    @classmethod
    def from_values(
        cls,
        *,
        amplitudes: float | Sequence[float] = (0.0, 0.01),
        epsilon_match: str | float = "standard",
        center: float | Sequence[float] = 200.0,
        width: float | Sequence[float] = 50.0,
        ramp_width: float | Sequence[float] = 40.0,
        calculation: str = "thermodynamics",
        fixed_masses: float | Sequence[float] = (1.4,),
        precision: str = "quick",
        diagnostics: str = "off",
    ) -> "ExperimentSettings":
        return cls(
            amplitudes=_number_tuple("amplitudes", amplitudes),
            epsilon_match=epsilon_match,
            center=_number_tuple("center", center, positive=True),
            width=_number_tuple("width", width, positive=True),
            ramp_width=_number_tuple("ramp_width", ramp_width, positive=True),
            calculation=calculation,
            fixed_masses=_number_tuple("fixed_masses", fixed_masses, positive=True),
            precision=precision,
            diagnostics=diagnostics,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentSettings":
        if not isinstance(payload, Mapping):
            raise TypeError("settings payload must be a mapping")
        values = dict(payload)
        values.pop("$schema", None)
        allowed = {
            "amplitudes",
            "epsilon_match",
            "center",
            "width",
            "ramp_width",
            "calculation",
            "precision",
            "fixed_masses",
            "diagnostics",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown experiment setting {unknown[0]!r}")
        return cls.from_values(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentSettings":
        payload = _strict_json_object(path)
        required = {
            "$schema",
            "amplitudes",
            "epsilon_match",
            "center",
            "width",
            "ramp_width",
            "calculation",
            "precision",
            "fixed_masses",
            "diagnostics",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                f"experiment configuration is missing required field {missing[0]!r}"
            )
        schema = payload.get("$schema")
        if not isinstance(schema, str) or not schema.strip():
            raise ValueError("experiment configuration $schema must be a non-empty string")
        for name in ("amplitudes", "fixed_masses"):
            if not isinstance(payload.get(name), list):
                raise ValueError(f"experiment configuration {name} must be an array")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "amplitudes": list(self.amplitudes),
            "epsilon_match": self.epsilon_match,
            "center": list(self.center) if len(self.center) > 1 else self.center[0],
            "width": list(self.width) if len(self.width) > 1 else self.width[0],
            "ramp_width": (
                list(self.ramp_width)
                if len(self.ramp_width) > 1
                else self.ramp_width[0]
            ),
            "calculation": self.calculation,
            "precision": self.precision,
            "fixed_masses": list(self.fixed_masses),
            "diagnostics": self.diagnostics,
        }

    def deterministic_hash(self) -> str:
        return _hash_payload(self.to_dict())


@dataclass(frozen=True)
class ExperimentPlan:
    """Calculation-free, write-free expansion of reviewed settings."""

    settings: ExperimentSettings
    child_plans: tuple[Any, ...]
    experiment_path: Path
    source_inventory_id: str
    source_digest: str
    source_file_count: int
    source_contracts: tuple[tuple[str, str], ...]
    runtime_identity: tuple[tuple[str, str], ...]
    runtime_digest: str
    plan_hash: str

    @property
    def case_table(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for index, child in enumerate(self.child_plans, start=1):
            frame = child.case_table.copy()
            frame.insert(0, "geometry_index", index)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @property
    def estimates(self) -> dict[str, int]:
        result: dict[str, int] = {"geometry_count": len(self.child_plans)}
        for child in self.child_plans:
            for key, value in child.estimates.items():
                result[key] = result.get(key, 0) + int(value)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": PLAN_SCHEMA,
            "settings": self.settings.to_dict(),
            "settings_hash": self.settings.deterministic_hash(),
            "experiment_path": _portable_path(self.experiment_path),
            "source_identity": {
                "inventory_id": self.source_inventory_id,
                "file_count": self.source_file_count,
                "sha256": self.source_digest,
                "project_contract_sha256": dict(self.source_contracts),
            },
            "runtime_identity": {
                "values": dict(self.runtime_identity),
                "sha256": self.runtime_digest,
            },
            "children": [_plan_child_document(child) for child in self.child_plans],
            "estimates": self.estimates,
            "planning_is_passive": True,
            "scientific_solver_calls": 0,
            "filesystem_writes": 0,
            "plan_hash": self.plan_hash,
        }

    def summary_text(self) -> str:
        estimates = self.estimates
        lines = [
            "BSk24 experiment plan",
            f"Plan hash: {self.plan_hash}",
            f"Calculation: {self.settings.calculation}",
            f"Precision: {self.settings.precision}",
            f"Geometries: {len(self.child_plans)}",
            f"Cases: {len(self.case_table)}",
            f"Destination: {self.experiment_path}",
            "Planning is passive: yes (0 solver calls, 0 filesystem writes)",
        ]
        for key in sorted(estimates):
            if key != "geometry_count":
                lines.append(f"{key.replace('_', ' ').capitalize()}: {estimates[key]}")
        for index, child in enumerate(self.child_plans, start=1):
            config = child.config
            thermo = ", ".join(
                f"{stage.name} ({stage.lower_points}/{stage.upper_points})"
                for stage in config.thermodynamic_stages
            )
            stellar = ", ".join(
                f"{stage.name} ({stage.sequence_points} pressures, "
                f"rtol={stage.rtol:.0e}, atol={stage.atol:.0e})"
                for stage in config.tov_stages
            )
            lines.extend(
                (
                    "",
                    f"Geometry {index}: center={config.epsilon0_mev_fm3:g}, "
                    f"width={config.sigma_mev_fm3:g}, "
                    f"ramp={config.deltas_mev_fm3[0]:g} MeV fm^-3",
                    f"  Thermodynamic stages: {thermo}",
                    f"  Stellar stages: {stellar or 'disabled'}",
                    f"  Raw-gate grids: {config.raw_gate_lower_points}/"
                    f"{config.raw_gate_upper_points} points",
                    f"  Central-pressure floor: "
                    f"{config.central_pressure_min_mev_fm3:g} MeV fm^-3",
                    f"  Fixed-mass root tolerance: "
                    f"{config.fixed_mass_root_xtol_mev_fm3:.0e} MeV fm^-3",
                    f"  Maximum-mass screen: {config.maximum_mass_initial_points} "
                    f"initial points; threshold "
                    f"{config.maximum_mass_threshold_msun:g} M_sun",
                    "  Cases: " + ", ".join(child.case_table["case_id"].astype(str)),
                )
            )
        if self.settings.calculation == "stellar":
            lines.append(
                "Maximum-mass refinement calls after the declared local screens "
                "are adaptive and are not included in the fixed target totals."
            )
        return "\n".join(lines)


@dataclass
class ExperimentResult:
    """Loaded handle to one completed experiment and its child packets."""

    experiment_path: Path
    settings: ExperimentSettings
    child_results: tuple[Any, ...]
    metadata: dict[str, Any]
    repository_root: Path

    @property
    def completed(self) -> bool:
        return bool(self.child_results) and self.metadata.get("status") == "complete"

    @property
    def packet_paths(self) -> tuple[Path, ...]:
        return tuple(Path(item.packet_path) for item in self.child_results)

    @property
    def accepted_cases(self) -> pd.DataFrame:
        return _combined_result_frame(self.child_results, "accepted_cases")

    @property
    def rejected_cases(self) -> pd.DataFrame:
        return _combined_result_frame(self.child_results, "rejected_cases")

    @property
    def figures(self) -> tuple[Path, ...]:
        return tuple(path for item in self.child_results for path in item.figures)

    @property
    def plot_inventory(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for index, item in enumerate(self.child_results, start=1):
            frame = item.figure_inventory()
            if "relative_path" in frame.columns:
                for value in frame["relative_path"].dropna():
                    item._packet_artifact(value)
            frame.insert(0, "geometry_index", index)
            frame.insert(1, "packet", Path(item.packet_path).name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def table(self, relative_path: str, *, geometry_index: int = 1) -> pd.DataFrame:
        if geometry_index < 1 or geometry_index > len(self.child_results):
            raise IndexError("geometry_index is outside the completed experiment")
        return self.child_results[geometry_index - 1].table(relative_path)

    def plot(
        self,
        *,
        overwrite: bool = False,
        groups: Sequence[str] = ("all-applicable",),
    ) -> "ExperimentResult":
        from ._internal.runtime import generate_trial_plots

        for packet in self.packet_paths:
            generate_trial_plots(
                packet,
                groups=groups,
                authorize_plot_overwrite=overwrite,
                repository_root=self.repository_root,
            )
        _write_aggregate_manifest(
            self.experiment_path,
            tuple(path.name for path in self.packet_paths),
        )
        return load_experiment(self.experiment_path)

    def show_gallery(self) -> tuple[Path, ...]:
        for item in self.child_results:
            item.show_gallery()
        return self.figures

    def summary_text(self) -> str:
        identity = sorted(
            {str(getattr(item, "identity_status", "unavailable")) for item in self.child_results}
        )
        convergence = sorted(
            {
                str(getattr(item, "convergence_status", "unavailable"))
                for item in self.child_results
            }
        )
        rejected = self.rejected_cases
        reason_column = next(
            (
                name
                for name in ("reason", "failure_reason", "status_reason")
                if name in rejected.columns
            ),
            None,
        )
        lines = [
            "BSk24 experiment: COMPLETE" if self.completed else "BSk24 experiment: INCOMPLETE",
            f"Path: {self.experiment_path}",
            f"Calculation: {self.settings.calculation}",
            f"Precision: {self.settings.precision}",
            f"Geometries / child packets: {len(self.child_results)}",
            f"Accepted cases: {len(self.accepted_cases)}",
            f"Rejected cases: {len(rejected)}",
            f"A=0 identity status: {', '.join(identity)}",
            f"Numerical convergence status: {', '.join(convergence)}",
            f"Stellar capability: {'requested' if self.settings.calculation == 'stellar' else 'not requested'}",
            f"Figures: {len(self.figures)}",
        ]
        if reason_column is not None and not rejected.empty:
            counts = rejected[reason_column].fillna("unspecified").astype(str).value_counts()
            lines.append(
                "Rejection reasons: "
                + "; ".join(f"{reason} ({count})" for reason, count in counts.items())
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class Experiment:
    """Convenient stateful front door around one immutable settings object."""

    settings: ExperimentSettings

    def __post_init__(self) -> None:
        if not isinstance(self.settings, ExperimentSettings):
            raise TypeError("settings must be ExperimentSettings")

    @classmethod
    def from_json(cls, path: str | Path) -> "Experiment":
        return cls(ExperimentSettings.from_json(path))

    def plan(self, output_root: str | Path | None = None) -> ExperimentPlan:
        return plan_experiment(self.settings, output_root=output_root)

    def run(self, plan: ExperimentPlan, *, execute: bool = False) -> ExperimentResult:
        if plan.settings != self.settings:
            raise ValueError("the reviewed plan belongs to different settings")
        return run_experiment(plan, execute=execute)

    @staticmethod
    def load(path: str | Path) -> ExperimentResult:
        return load_experiment(path)


def _combined_result_frame(results: Sequence[Any], attribute: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for index, item in enumerate(results, start=1):
        frame = getattr(item, attribute).copy()
        if not frame.empty:
            frame.insert(0, "geometry_index", index)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plan_experiment(
    settings: ExperimentSettings,
    output_root: str | Path | None = None,
) -> ExperimentPlan:
    """Expand and validate settings with zero calculations and zero writes."""

    if not isinstance(settings, ExperimentSettings):
        raise TypeError("settings must be ExperimentSettings")
    from ._internal.planning import prepare_bsk24_trial

    from ._internal.artifacts import project_root, runs_root

    if output_root is None:
        root = runs_root().resolve(strict=False)
    else:
        raw_root = Path(output_root).expanduser()
        if raw_root.is_absolute():
            root = raw_root.resolve(strict=False)
        elif raw_root.parts and raw_root.parts[0] == "runs":
            root = (project_root() / raw_root).resolve(strict=False)
        else:
            root = (Path.cwd() / raw_root).resolve(strict=False)
    experiment_path = root / f"experiment_{settings.deterministic_hash()[:12]}"
    configs = _internal_configs(settings, experiment_path)
    children = tuple(prepare_bsk24_trial(config) for config in configs)
    (
        source_inventory_id,
        source_digest,
        source_file_count,
        source_contracts,
    ) = _active_source_identity()
    runtime_identity = _active_runtime_identity()
    runtime_digest = _hash_payload(dict(runtime_identity))
    digest = _plan_digest(
        settings,
        experiment_path,
        children,
        source_inventory_id=source_inventory_id,
        source_digest=source_digest,
        source_file_count=source_file_count,
        source_contracts=source_contracts,
        runtime_identity=runtime_identity,
        runtime_digest=runtime_digest,
    )
    return ExperimentPlan(
        settings,
        children,
        experiment_path,
        source_inventory_id,
        source_digest,
        source_file_count,
        source_contracts,
        runtime_identity,
        runtime_digest,
        digest,
    )


def _reproduction_plan_record(plan: ExperimentPlan) -> dict[str, Any]:
    from ._internal.artifacts import project_root

    root = project_root().resolve(strict=False)
    output_root = root / "runs" / "reproductions"
    reproduction = plan_experiment(plan.settings, output_root=output_root)
    configuration_file = (
        plan.experiment_path / "experiment_config.json"
    ).resolve(strict=False).relative_to(root).as_posix()
    portable_output_root = output_root.relative_to(root).as_posix()
    plan_command = (
        "bsk24-trial plan "
        f'--config "{configuration_file}" '
        f'--output-root "{portable_output_root}" --json'
    )
    run_command = (
        "bsk24-trial run "
        f'--config "{configuration_file}" '
        f'--output-root "{portable_output_root}" '
        f"--plan-hash {reproduction.plan_hash} --execute"
    )
    return {
        "schema_id": REPRODUCTION_PLAN_SCHEMA,
        "configuration_file": configuration_file,
        "output_root": portable_output_root,
        "plan_hash": reproduction.plan_hash,
        "plan": reproduction.to_dict(),
        "plan_command": plan_command,
        "run_command": run_command,
    }


def run_experiment(plan: ExperimentPlan, *, execute: bool = False) -> ExperimentResult:
    """Execute an exact reviewed plan after an explicit authorization gate."""

    if not isinstance(plan, ExperimentPlan):
        raise TypeError("run_experiment requires an ExperimentPlan, not raw settings")
    if execute is not True:
        raise PermissionError("execution requires execute=True after reviewing the plan")
    if plan.experiment_path.exists():
        raise FileExistsError(f"experiment destination already exists: {plan.experiment_path}")
    reviewed = plan_experiment(plan.settings, output_root=plan.experiment_path.parent)
    if reviewed.plan_hash != plan.plan_hash or reviewed.to_dict() != plan.to_dict():
        raise RuntimeError("reviewed plan is stale or has changed; preview it again")

    from ._internal.runtime import execute_trial

    # These two small public documents are written before the first child so a
    # failed calculation remains reproducible without being mislabeled as a
    # completed experiment.
    config_document = {"$schema": CONFIG_SCHEMA_URL, **plan.settings.to_dict()}
    reproduction_document = _reproduction_plan_record(plan)
    config_path = plan.experiment_path / "experiment_config.json"
    reviewed_plan_path = plan.experiment_path / "reviewed_plan.json"
    reproduction_plan_path = plan.experiment_path / "reproduction_plan.json"
    _write_json_atomic(config_path, config_document)
    _write_json_atomic(reviewed_plan_path, plan.to_dict())
    _write_json_atomic(reproduction_plan_path, reproduction_document)

    results: list[Any] = []
    try:
        for child in plan.child_plans:
            results.append(execute_trial(child.config))
    except Exception:
        # Child packets deliberately retain fail-closed evidence.  The absence
        # of experiment.json makes the aggregate status unambiguously incomplete.
        raise

    metadata = {
        "schema_id": EXPERIMENT_SCHEMA,
        "status": "complete",
        "settings": plan.settings.to_dict(),
        "settings_hash": plan.settings.deterministic_hash(),
        "plan_hash": plan.plan_hash,
        "child_packets": [
            Path(item.packet_path).relative_to(plan.experiment_path).as_posix()
            for item in results
        ],
        "child_configuration_hashes": [
            child.config.deterministic_hash() for child in plan.child_plans
        ],
        "document_sha256": {
            "experiment_config.json": _sha256_file(config_path),
            "reviewed_plan.json": _sha256_file(reviewed_plan_path),
            "reproduction_plan.json": _sha256_file(reproduction_plan_path),
        },
    }
    _write_json_atomic(plan.experiment_path / "experiment.json", metadata)
    _write_aggregate_manifest(
        plan.experiment_path,
        tuple(Path(item.packet_path).name for item in results),
    )
    from ._internal.artifacts import project_root

    return ExperimentResult(
        plan.experiment_path,
        plan.settings,
        tuple(results),
        metadata,
        project_root().resolve(strict=False),
    )


def load_experiment(
    path: str | Path,
    *,
    _require_child_validation: bool = True,
) -> ExperimentResult:
    """Load completed saved results without rerunning any calculation."""

    experiment_path = Path(path).expanduser().resolve(strict=False)
    metadata_path = experiment_path / "experiment.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"completed experiment metadata not found: {metadata_path}")
    metadata = _strict_json_object(metadata_path)
    if metadata.get("schema_id") != EXPERIMENT_SCHEMA:
        raise ValueError("unsupported experiment schema")
    if metadata.get("status") != "complete":
        raise ValueError("saved experiment is not complete")
    document_hashes = metadata.get("document_sha256")
    expected_documents = {
        "experiment_config.json",
        "reviewed_plan.json",
        "reproduction_plan.json",
    }
    if not isinstance(document_hashes, Mapping) or set(document_hashes) != expected_documents:
        raise ValueError("saved experiment document inventory is incomplete")
    for name in sorted(expected_documents):
        document = experiment_path / name
        if not document.is_file() or _sha256_file(document) != document_hashes[name]:
            raise ValueError(f"saved experiment document hash mismatch: {name}")

    config_path = experiment_path / "experiment_config.json"
    settings = ExperimentSettings.from_json(config_path)
    if settings.to_dict() != metadata.get("settings"):
        raise ValueError("saved experiment config and metadata disagree")
    if settings.deterministic_hash() != metadata.get("settings_hash"):
        raise ValueError("saved experiment settings hash mismatch")

    reviewed = _strict_json_object(experiment_path / "reviewed_plan.json")
    if reviewed.get("schema_id") != PLAN_SCHEMA:
        raise ValueError("unsupported reviewed-plan schema")
    if reviewed.get("settings") != settings.to_dict():
        raise ValueError("reviewed plan settings disagree with the saved config")
    if reviewed.get("settings_hash") != settings.deterministic_hash():
        raise ValueError("reviewed plan settings hash mismatch")
    if reviewed.get("plan_hash") != metadata.get("plan_hash"):
        raise ValueError("saved experiment plan hash mismatch")
    if _saved_plan_digest(reviewed) != reviewed.get("plan_hash"):
        raise ValueError("reviewed plan contents do not match its hash")
    source_identity = reviewed.get("source_identity")
    runtime_identity = reviewed.get("runtime_identity")
    if (
        not isinstance(source_identity, Mapping)
        or not isinstance(source_identity.get("inventory_id"), str)
        or not isinstance(source_identity.get("file_count"), int)
        or not isinstance(source_identity.get("sha256"), str)
        or len(source_identity["sha256"]) != 64
        or not isinstance(source_identity.get("project_contract_sha256"), Mapping)
        or set(source_identity["project_contract_sha256"])
        != {"environment.yml", "pyproject.toml"}
        or any(
            value != "unavailable"
            and (not isinstance(value, str) or len(value) != 64)
            for value in source_identity["project_contract_sha256"].values()
        )
    ):
        raise ValueError("reviewed plan source identity is malformed")
    if (
        not isinstance(runtime_identity, Mapping)
        or not isinstance(runtime_identity.get("values"), Mapping)
        or runtime_identity.get("sha256")
        != _hash_payload(dict(runtime_identity.get("values", {})))
    ):
        raise ValueError("reviewed plan runtime identity is malformed")

    reproduction = _strict_json_object(experiment_path / "reproduction_plan.json")
    if reproduction.get("schema_id") != REPRODUCTION_PLAN_SCHEMA:
        raise ValueError("unsupported reproduction-plan schema")
    for name in (
        "configuration_file",
        "output_root",
        "plan_hash",
        "plan_command",
        "run_command",
    ):
        if not isinstance(reproduction.get(name), str) or not reproduction[name]:
            raise ValueError(f"reproduction plan field {name!r} is malformed")
    reproduction_plan = reproduction.get("plan")
    if not isinstance(reproduction_plan, Mapping):
        raise ValueError("reproduction plan payload is malformed")
    if reproduction_plan.get("plan_hash") != reproduction["plan_hash"]:
        raise ValueError("reproduction plan payload/hash mismatch")
    if _saved_plan_digest(reproduction_plan) != reproduction["plan_hash"]:
        raise ValueError("reproduction plan contents do not match its hash")
    if reproduction_plan.get("settings") != settings.to_dict():
        raise ValueError("reproduction plan settings mismatch")
    root = _owning_repository_root(
        experiment_path,
        reviewed.get("experiment_path"),
    )
    expected_configuration = config_path.resolve(strict=False).relative_to(root).as_posix()
    expected_output_root = "runs/reproductions"
    expected_experiment_path = (
        root / expected_output_root / f"experiment_{settings.deterministic_hash()[:12]}"
    ).resolve(strict=False)
    if reproduction["configuration_file"] != expected_configuration:
        raise ValueError("reproduction configuration path mismatch")
    if reproduction["output_root"] != expected_output_root:
        raise ValueError("reproduction output root mismatch")
    if _resolve_portable_path(
        reproduction_plan.get("experiment_path"), root
    ) != expected_experiment_path:
        raise ValueError("reproduction experiment path mismatch")
    expected_plan_command = (
        "bsk24-trial plan "
        f'--config "{expected_configuration}" '
        f'--output-root "{expected_output_root}" --json'
    )
    expected_run_command = (
        "bsk24-trial run "
        f'--config "{expected_configuration}" '
        f'--output-root "{expected_output_root}" '
        f'--plan-hash {reproduction["plan_hash"]} --execute'
    )
    if reproduction["plan_command"] != expected_plan_command:
        raise ValueError("reproduction plan command mismatch")
    if reproduction["run_command"] != expected_run_command:
        raise ValueError("reproduction run command mismatch")

    children = metadata.get("child_packets")
    if not isinstance(children, list) or not children:
        raise ValueError("saved experiment has no child packets")
    for value in children:
        if (
            not isinstance(value, str)
            or not value
            or "/" in value
            or "\\" in value
            or value in {".", ".."}
            or not value.startswith("geometry_")
        ):
            raise ValueError(f"unsafe saved child packet path: {value!r}")
        child_path = (experiment_path / value).resolve(strict=False)
        if child_path.parent != experiment_path:
            raise ValueError(f"saved child packet escapes the experiment: {value!r}")
    reviewed_children = reviewed.get("children")
    if not isinstance(reviewed_children, list) or len(reviewed_children) != len(children):
        raise ValueError("saved child packets disagree with the reviewed plan")
    expected_children: list[str] = []
    expected_hashes: list[str] = []
    for child in reviewed_children:
        if not isinstance(child, Mapping):
            raise ValueError("reviewed child plan is malformed")
        output_path = child.get("output_path")
        configuration_hash = child.get("configuration_hash")
        if not isinstance(output_path, str) or not output_path:
            raise ValueError("reviewed child output path is malformed")
        reviewed_child_path = _resolve_portable_path(output_path, root)
        expected_child = reviewed_child_path.name
        if reviewed_child_path != (experiment_path / expected_child).resolve(
            strict=False
        ):
            raise ValueError("reviewed child output path escapes its experiment")
        expected_children.append(expected_child)
        if not isinstance(configuration_hash, str) or len(configuration_hash) != 64:
            raise ValueError("reviewed child configuration hash is malformed")
        expected_hashes.append(configuration_hash)
    if children != expected_children or len(set(children)) != len(children):
        raise ValueError("saved child packet paths disagree with the reviewed plan")
    if metadata.get("child_configuration_hashes") != expected_hashes:
        raise ValueError("saved child configuration hashes disagree with the plan")
    _verify_aggregate_manifest(experiment_path, children)

    from ._internal.runtime import load_trial, validate_trial

    results_list: list[Any] = []
    for value, configuration_hash in zip(children, expected_hashes):
        child_path = (experiment_path / value).resolve(strict=False)
        if _require_child_validation:
            validation = validate_trial(child_path, repository_root=root)
            if validation.get("status") != "pass":
                failures = validation.get("failures", [])
                detail = failures[0] if failures else "unspecified validation failure"
                raise ValueError(f"saved child packet failed validation: {detail}")
        result = load_trial(child_path, repository_root=root)
        if Path(result.packet_path).resolve(strict=False) != child_path:
            raise ValueError("loaded child packet path mismatch")
        if result.config.deterministic_hash() != configuration_hash:
            raise ValueError("loaded child configuration disagrees with the reviewed plan")
        results_list.append(result)
    results = tuple(results_list)
    return ExperimentResult(experiment_path, settings, results, metadata, root)


def validate_experiment(path: str | Path) -> dict[str, Any]:
    """Read-only integrity/scientific validation of every saved child packet."""

    try:
        result = load_experiment(path, _require_child_validation=False)
    except Exception as exc:
        return {
            "schema_id": "eos_generation_validation_v1",
            "status": "fail",
            "experiment_path": str(Path(path).expanduser().resolve(strict=False)),
            "child_packet_count": 0,
            "failures": [f"aggregate_load:{type(exc).__name__}:{exc}"],
            "children": [],
        }
    from ._internal.runtime import validate_trial

    children = [
        validate_trial(packet, repository_root=result.repository_root)
        for packet in result.packet_paths
    ]
    passed = all(
        item.get("status") in {"pass", "complete", "validated"}
        or item.get("overall_status") in {"pass", "complete", "validated"}
        for item in children
    )
    return {
        "schema_id": "eos_generation_validation_v1",
        "status": "pass" if passed else "fail",
        "experiment_path": str(result.experiment_path),
        "child_packet_count": len(children),
        "children": children,
    }


__all__ = [
    "Experiment",
    "ExperimentPlan",
    "ExperimentResult",
    "ExperimentSettings",
    "load_experiment",
    "plan_experiment",
    "run_experiment",
    "validate_experiment",
]

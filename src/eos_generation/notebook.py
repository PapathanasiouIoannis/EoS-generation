"""Small, passive-by-default notebook adapter for BSk24 experiments.

The notebook contains no equations or solver code.  It translates one plain
settings cell into :mod:`eos_generation.experiment`, records an exact passive
preview, and permits that preview to be executed once on a later ``Run All``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from numbers import Real
from pathlib import Path
from typing import Any


_SETTINGS_SCHEMA = "eos_generation_notebook_settings_v1"
_RUN_SCHEMA = "eos_generation_notebook_run_v1"
_CALCULATIONS = ("thermodynamics", "stellar")
_PRECISIONS = ("quick", "strict")
_DIAGNOSTICS = ("off", "on")
_MAX_WORKERS = 4


def _as_items(value: Any) -> tuple[Any, ...]:
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(value)
    return (value,)


def _numeric_axis(name: str, value: Any, *, positive: bool = False) -> tuple[float, ...]:
    items = _as_items(value)
    if not items:
        raise ValueError(f"{name} must contain at least one value")
    parsed: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{name} values must be real numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} values must be finite")
        if positive and number <= 0.0:
            raise ValueError(f"{name} values must be greater than zero")
        # Canonicalize harmless floating-point roundoff around an intended zero.
        if name == "AMPLITUDES" and abs(number) <= len(items) * math.ulp(1.0):
            number = 0.0
        if number in parsed:
            raise ValueError(f"{name} contains duplicate value {number:.12g}")
        parsed.append(number)
    return tuple(parsed)


def _match_value(value: Any) -> str | float:
    items = _as_items(value)
    if len(items) != 1:
        raise ValueError("EPSILON_MATCH must be one matching anchor")
    item = items[0]
    if item is None or (
        isinstance(item, str) and item.strip().lower() == "standard"
    ):
        return "standard"
    if isinstance(item, bool) or not isinstance(item, Real):
        raise ValueError("EPSILON_MATCH must be 'standard' or a finite number")
    number = float(item)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(
            "numeric EPSILON_MATCH must be finite and greater than zero"
        )
    return number


def _choice(name: str, value: Any, allowed: tuple[str, ...]) -> str:
    selected = str(value).strip().lower()
    if selected not in allowed:
        raise ValueError(f"{name} must be one of {allowed}")
    return selected


@dataclass(frozen=True)
class NotebookSettings:
    """Normalized values from the notebook's only editable cell.

    Scalar and sequence inputs are both accepted by :meth:`from_values`.
    Multiple values define a Cartesian grid; the production experiment layer
    remains the authority for physical domains and numerical profiles.
    """

    amplitudes: tuple[float, ...]
    epsilon_match: str | float
    centers_mev_fm3: tuple[float, ...]
    widths_mev_fm3: tuple[float, ...]
    ramp_widths_mev_fm3: tuple[float, ...]
    calculation: str
    fixed_masses_msun: tuple[float, ...]
    precision: str
    diagnostics: str

    def __post_init__(self) -> None:
        # Direct dataclass construction receives the same validation as the
        # canonical notebook constructor.
        object.__setattr__(
            self, "amplitudes", _numeric_axis("AMPLITUDES", self.amplitudes)
        )
        object.__setattr__(self, "epsilon_match", _match_value(self.epsilon_match))
        object.__setattr__(
            self,
            "centers_mev_fm3",
            _numeric_axis("CENTER", self.centers_mev_fm3, positive=True),
        )
        object.__setattr__(
            self,
            "widths_mev_fm3",
            _numeric_axis("WIDTH", self.widths_mev_fm3, positive=True),
        )
        object.__setattr__(
            self,
            "ramp_widths_mev_fm3",
            _numeric_axis("RAMP_WIDTH", self.ramp_widths_mev_fm3, positive=True),
        )
        object.__setattr__(
            self,
            "fixed_masses_msun",
            _numeric_axis("FIXED_MASSES", self.fixed_masses_msun, positive=True),
        )
        object.__setattr__(
            self,
            "calculation",
            _choice("CALCULATION", self.calculation, _CALCULATIONS),
        )
        object.__setattr__(
            self,
            "precision",
            _choice("PRECISION", self.precision, _PRECISIONS),
        )
        object.__setattr__(
            self,
            "diagnostics",
            _choice("DIAGNOSTICS", self.diagnostics, _DIAGNOSTICS),
        )
        if self.diagnostics == "on" and self.calculation != "stellar":
            raise ValueError("DIAGNOSTICS='on' requires CALCULATION='stellar'")

    @classmethod
    def from_values(
        cls,
        *,
        amplitudes: Any,
        epsilon_match: Any,
        center: Any,
        width: Any,
        ramp_width: Any,
        calculation: str = "thermodynamics",
        fixed_masses: Any = (1.4,),
        precision: str = "strict",
        diagnostics: str = "off",
    ) -> "NotebookSettings":
        return cls(
            amplitudes=_numeric_axis("AMPLITUDES", amplitudes),
            epsilon_match=_match_value(epsilon_match),
            centers_mev_fm3=_numeric_axis("CENTER", center, positive=True),
            widths_mev_fm3=_numeric_axis("WIDTH", width, positive=True),
            ramp_widths_mev_fm3=_numeric_axis(
                "RAMP_WIDTH", ramp_width, positive=True
            ),
            calculation=calculation,
            fixed_masses_msun=_numeric_axis(
                "FIXED_MASSES", fixed_masses, positive=True
            ),
            precision=precision,
            diagnostics=diagnostics,
        )

    @property
    def geometry_count(self) -> int:
        return (
            len(self.centers_mev_fm3)
            * len(self.widths_mev_fm3)
            * len(self.ramp_widths_mev_fm3)
        )

    @property
    def requested_case_count(self) -> int:
        return len(self.amplitudes) * self.geometry_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": _SETTINGS_SCHEMA,
            "amplitudes": list(self.amplitudes),
            "epsilon_match": self.epsilon_match,
            "center_mev_fm3": list(self.centers_mev_fm3),
            "width_mev_fm3": list(self.widths_mev_fm3),
            "ramp_width_mev_fm3": list(self.ramp_widths_mev_fm3),
            "calculation": self.calculation,
            "fixed_masses_msun": list(self.fixed_masses_msun),
            "precision": self.precision,
            "diagnostics": self.diagnostics,
        }

    def deterministic_hash(self) -> str:
        return _digest(self.to_dict())

    def to_experiment_settings(self) -> Any:
        """Build the governed production settings without duplicating policy."""

        from .experiment import ExperimentSettings

        return ExperimentSettings.from_values(
            amplitudes=self.amplitudes,
            epsilon_match=self.epsilon_match,
            center=self.centers_mev_fm3,
            width=self.widths_mev_fm3,
            ramp_width=self.ramp_widths_mev_fm3,
            calculation=self.calculation,
            fixed_masses=self.fixed_masses_msun,
            precision=self.precision,
            diagnostics=self.diagnostics,
        )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return repr(value)


def _digest(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _plan_document(plan: Any) -> Any:
    to_dict = getattr(plan, "to_dict", None)
    return to_dict() if callable(to_dict) else plan


def _default_source_state(repository_root: Path) -> Mapping[str, str]:
    package_root = Path(__file__).resolve().parent
    result = {
        f"src/eos_generation/{path.relative_to(package_root).as_posix()}":
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_root.rglob("*.py"))
        if path.is_file()
    }
    for name in ("environment.yml", "pyproject.toml"):
        candidate = repository_root / name
        if candidate.is_file():
            result[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return result


def _default_environment_state() -> Mapping[str, Any]:
    versions: dict[str, str] = {}
    for distribution in ("numpy", "scipy", "pandas", "matplotlib", "numba"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": versions,
    }


def _default_worker_count() -> int:
    logical = max(1, int(os.cpu_count() or 1))
    return min(_MAX_WORKERS, max(1, logical // 2))


def _repository_root(value: str | Path | None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve(strict=False)
    current = Path.cwd().resolve(strict=False)
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current


def _result_path(result: Any) -> Path | None:
    for name in (
        "experiment_path",
        "output_root",
        "output_path",
        "packet_path",
        "path",
    ):
        value = getattr(result, name, None)
        if value is not None:
            return Path(value).resolve(strict=False)
    return None


@dataclass(frozen=True)
class NotebookRun:
    """An immutable, hash-bound preview of one production experiment."""

    settings: NotebookSettings
    experiment_settings: Any
    plan: Any
    planning_root: Path
    output_root: Path
    settings_hash: str
    plan_hash: str
    authorization_token: str
    source_state: Mapping[str, str]
    environment_state: Mapping[str, Any]
    worker_count: int

    def to_dict(self) -> dict[str, Any]:
        repository_root = self.planning_root.parents[1]

        def portable(path: Path) -> str:
            return path.resolve(strict=False).relative_to(repository_root).as_posix()

        return {
            "schema_id": _RUN_SCHEMA,
            "settings": self.settings.to_dict(),
            "settings_hash": self.settings_hash,
            "plan_hash": self.plan_hash,
            "authorization_token": self.authorization_token,
            "planning_root": portable(self.planning_root),
            "output_root": portable(self.output_root),
            "worker_count": self.worker_count,
            "source_state": dict(self.source_state),
            "environment_state": dict(self.environment_state),
            "experiment_plan": _plan_document(self.plan),
            "planning_is_passive": True,
            "scientific_solver_calls": 0,
            "filesystem_writes": 0,
        }

    def summary_text(self) -> str:
        plan_summary = getattr(self.plan, "summary_text", None)
        details = plan_summary() if callable(plan_summary) else ""
        lines = [
            "Settings valid; exact experiment preview recorded.",
            f"Calculation: {self.settings.calculation}",
            f"Requested cases: {self.settings.requested_case_count}",
            f"Numerical precision: {self.settings.precision}",
            f"Diagnostics: {self.settings.diagnostics}",
            f"Destination: {self.output_root}",
            "Preview cost: 0 solver calls, 0 filesystem writes.",
            "Execution requires a second Run All with EXECUTE_REVIEWED_PLAN=True.",
        ]
        if self.settings.precision == "quick":
            lines.append(
                "Quick changes numerical resolution only; physical acceptance gates are unchanged."
            )
        if details:
            lines.extend(("", str(details)))
        return "\n".join(lines)


class NotebookSession:
    """Kernel-local two-pass preview and execution gate.

    Planning is always delegated to :func:`plan_experiment` and is required to
    be passive.  Execution is one-shot and fails closed if settings, plan,
    source, environment, process budget, or destination changed after review.
    """

    def __init__(
        self,
        repository_root: str | Path | None = None,
        *,
        planner: Callable[..., Any] | None = None,
        runner: Callable[..., Any] | None = None,
        loader: Callable[[Path], Any] | None = None,
        validator: Callable[[Path], Any] | None = None,
        settings_factory: Callable[[NotebookSettings], Any] | None = None,
        source_state: Callable[[], Mapping[str, str]] | None = None,
        environment_state: Callable[[], Mapping[str, Any]] | None = None,
        worker_count: Callable[[], int] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository_root = _repository_root(repository_root)
        self.runs_root = (self.repository_root / "runs").resolve(strict=False)
        self._planner = planner
        self._runner = runner
        self._loader = loader
        self._validator = validator
        self._settings_factory = settings_factory
        self._source_state = source_state or (
            lambda: _default_source_state(self.repository_root)
        )
        self._environment_state = environment_state or _default_environment_state
        self._worker_count = worker_count or _default_worker_count
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._runs: dict[str, NotebookRun] = {}
        self._reviewed: dict[str, str] = {}
        self._consumed: set[str] = set()

    def _dependencies(self) -> tuple[Callable[..., Any], Callable[..., Any], Callable[[Path], Any], Callable[[Path], Any]]:
        if all(
            dependency is not None
            for dependency in (self._planner, self._runner, self._loader, self._validator)
        ):
            return self._planner, self._runner, self._loader, self._validator  # type: ignore[return-value]
        from .experiment import (
            load_experiment,
            plan_experiment,
            run_experiment,
            validate_experiment,
        )

        return (
            self._planner or plan_experiment,
            self._runner or run_experiment,
            self._loader or load_experiment,
            self._validator or validate_experiment,
        )

    def _experiment_settings(self, settings: NotebookSettings) -> Any:
        if self._settings_factory is not None:
            return self._settings_factory(settings)
        return settings.to_experiment_settings()

    def _new_output_root(self, settings_hash: str) -> Path:
        timestamp = self._now().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"bsk24_{timestamp}_{settings_hash[:12]}"
        candidate = self.runs_root / stem
        suffix = 2
        while candidate.exists():
            candidate = self.runs_root / f"{stem}_{suffix:02d}"
            suffix += 1
        return candidate.resolve(strict=False)

    def _assert_destination(self, path: Path) -> None:
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self.runs_root)
        except ValueError as exc:
            raise RuntimeError(f"notebook output must remain below {self.runs_root}") from exc
        if resolved == self.runs_root:
            raise RuntimeError("the runs root itself cannot be an experiment destination")

    def _build(self, settings: NotebookSettings, planning_root: Path) -> NotebookRun:
        planner, _, _, _ = self._dependencies()
        self._assert_destination(planning_root)
        if planning_root.exists():
            raise FileExistsError(f"planned destination already exists: {planning_root}")
        experiment_settings = self._experiment_settings(settings)
        plan = planner(experiment_settings, output_root=planning_root)
        if planning_root.exists():
            raise RuntimeError("plan_experiment wrote to its planned destination")
        plan_destination = _result_path(plan) or planning_root
        self._assert_destination(plan_destination)
        if plan_destination.exists():
            raise RuntimeError("plan_experiment wrote to its experiment destination")
        settings_hash = settings.deterministic_hash()
        source_state = dict(self._source_state())
        environment_state = dict(self._environment_state())
        worker_count = int(self._worker_count())
        if worker_count < 1:
            raise RuntimeError("automatic worker count must be positive")
        plan_hash = _digest(_plan_document(plan))
        token = _digest(
            {
                "settings_hash": settings_hash,
                "plan_hash": plan_hash,
                "planning_root": str(planning_root),
                "output_root": str(plan_destination),
                "source_state": source_state,
                "environment_state": environment_state,
                "worker_count": worker_count,
            }
        )
        return NotebookRun(
            settings=settings,
            experiment_settings=experiment_settings,
            plan=plan,
            planning_root=planning_root,
            output_root=plan_destination,
            settings_hash=settings_hash,
            plan_hash=plan_hash,
            authorization_token=token,
            source_state=source_state,
            environment_state=environment_state,
            worker_count=worker_count,
        )

    def prepare(
        self, settings: NotebookSettings, *, record_preview: bool
    ) -> NotebookRun:
        """Passively plan, optionally recording the exact plan as reviewed."""

        if not isinstance(settings, NotebookSettings):
            raise TypeError("settings must be NotebookSettings from the editable cell")
        if not isinstance(record_preview, bool):
            raise TypeError("record_preview must be boolean")
        settings_hash = settings.deterministic_hash()
        existing = self._runs.get(settings_hash)
        current_source = dict(self._source_state())
        current_environment = dict(self._environment_state())
        current_workers = int(self._worker_count())
        reusable = bool(
            existing is not None
            and not existing.planning_root.exists()
            and not existing.output_root.exists()
            and dict(existing.source_state) == current_source
            and dict(existing.environment_state) == current_environment
            and existing.worker_count == current_workers
            and _digest(_plan_document(existing.plan)) == existing.plan_hash
        )
        run = (
            existing
            if reusable
            else self._build(settings, self._new_output_root(settings_hash))
        )
        self._runs[settings_hash] = run
        if record_preview:
            self._reviewed.clear()
            self._reviewed[settings_hash] = run.authorization_token
        return run

    def execute(
        self,
        run: NotebookRun,
        *,
        current_settings: NotebookSettings,
        execute: bool,
    ) -> Any | None:
        """Execute one unchanged reviewed plan, once, only when explicitly true."""

        if not isinstance(execute, bool):
            raise TypeError("EXECUTE_REVIEWED_PLAN must be exactly False or True")
        if not execute:
            return None
        if not isinstance(run, NotebookRun):
            raise TypeError("run must be the NotebookRun returned by prepare")
        current_hash = current_settings.deterministic_hash()
        if current_hash != run.settings_hash:
            raise RuntimeError(
                "settings changed after preview; set EXECUTE_REVIEWED_PLAN=False and Run All again"
            )
        if run.authorization_token in self._consumed:
            raise RuntimeError("this reviewed plan has already been consumed")
        if self._reviewed.get(current_hash) != run.authorization_token:
            raise RuntimeError(
                "this exact plan was not reviewed; first Run All with EXECUTE_REVIEWED_PLAN=False"
            )
        if _digest(_plan_document(run.plan)) != run.plan_hash:
            raise RuntimeError("the experiment plan changed after preview; preview again")
        if dict(self._source_state()) != dict(run.source_state):
            raise RuntimeError("governed source changed after preview; preview again")
        if dict(self._environment_state()) != dict(run.environment_state):
            raise RuntimeError("the execution environment changed after preview; preview again")
        if int(self._worker_count()) != run.worker_count:
            raise RuntimeError("the automatic process budget changed after preview; preview again")
        self._assert_destination(run.output_root)
        self._assert_destination(run.planning_root)
        if run.planning_root.exists() or run.output_root.exists():
            occupied = (
                run.planning_root if run.planning_root.exists() else run.output_root
            )
            raise FileExistsError(f"planned destination already exists: {occupied}")

        _, runner, _, validator = self._dependencies()
        self._consumed.add(run.authorization_token)
        self._reviewed.pop(current_hash, None)
        result = runner(run.plan, execute=True)
        result_path = _result_path(result)
        if result_path is not None and result_path != run.output_root:
            raise RuntimeError(
                "run_experiment returned a result outside the reviewed destination: "
                f"{result_path} != {run.output_root}"
            )
        report = validator(run.output_root)
        status = (
            report.get("status")
            if isinstance(report, Mapping)
            else getattr(report, "status", None)
        )
        if status not in (None, "pass", "valid", "complete"):
            raise RuntimeError(f"completed experiment failed validation: {status}")
        return result

    def load(self, path: str | Path) -> Any:
        """Passively load an existing experiment through the public facade."""

        _, _, loader, _ = self._dependencies()
        return loader(Path(path))

    def validate(self, path: str | Path) -> Any:
        """Read-only validation through the public facade."""

        _, _, _, validator = self._dependencies()
        return validator(Path(path))


_SESSIONS: dict[Path, NotebookSession] = {}


def get_notebook_session(
    repository_root: str | Path | None = None,
) -> NotebookSession:
    """Return the kernel-local session used by both notebook passes."""

    root = _repository_root(repository_root)
    session = _SESSIONS.get(root)
    if session is None:
        session = NotebookSession(root)
        _SESSIONS[root] = session
    return session


__all__ = [
    "NotebookRun",
    "NotebookSession",
    "NotebookSettings",
    "get_notebook_session",
]

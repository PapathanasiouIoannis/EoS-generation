"""Settings for controlled BSk24 / BSk25 experiments."""

from __future__ import annotations

import math
from .storage import hash_payload as _hash_payload, strict_json as _strict_json_object
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONFIG_SCHEMA_URL = (
    "https://raw.githubusercontent.com/PapathanasiouIoannis/"
    "EoS-generation/main/configs/schema.json"
)


_CALCULATIONS = ("thermodynamics", "stellar")


_MATTER_MODELS = ("bsk24", "bsk25")


_PRECISIONS = (
    "quick",
    "strict",
    "dataset",
    "dataset_10_tighter",
    "dataset_20",
    "dataset_40",
    "dataset_40_curves",
    "dataset_relaxed",
    "dataset_relaxed_80",
)


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

    matter_model: str = "bsk24"
    amplitudes: tuple[float, ...] = (0.0, 0.01)
    epsilon_match: str | float = "standard"
    center: tuple[float, ...] = (200.0,)
    width: tuple[float, ...] = (50.0,)
    ramp_width: tuple[float, ...] = (40.0,)
    calculation: str = "thermodynamics"
    precision: str = "quick"
    fixed_masses: tuple[float, ...] = (1.4,)
    diagnostics: str = "off"
    observables: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.matter_model not in _MATTER_MODELS:
            raise ValueError(f"matter_model must be one of {_MATTER_MODELS}")
        object.__setattr__(
            self, "amplitudes", _number_tuple("amplitudes", self.amplitudes)
        )
        object.__setattr__(
            self, "center", _number_tuple("center", self.center, positive=True)
        )
        object.__setattr__(
            self, "width", _number_tuple("width", self.width, positive=True)
        )
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
        if self.observables is not None:
            items = tuple(self.observables)
            if len(items) != len(set(items)) or any(
                x not in ("sequence", "fixed_mass", "maximum_mass") for x in items
            ):
                raise ValueError(
                    "observables must contain unique sequence, fixed_mass or maximum_mass names"
                )
            if items and self.calculation != "stellar":
                raise ValueError("stellar observables require calculation='stellar'")
            if self.calculation == "stellar" and "sequence" not in items:
                raise ValueError("stellar calculation requires the sequence observable")
            object.__setattr__(self, "observables", items)

        if self.diagnostics == "on" and "fixed_mass" not in self.requested_observables:
            raise ValueError("diagnostics='on' requires the fixed_mass observable")

    @property
    def requested_observables(self) -> tuple[str, ...]:
        if self.observables is not None:
            return self.observables
        if self.calculation != "stellar":
            return ()
        # Preserve historical default work while separating new explicit requests.
        return (
            ("sequence",)
            if self.precision == "dataset_40_curves"
            else ("sequence", "fixed_mass", "maximum_mass")
        )

    @classmethod
    def from_values(
        cls,
        *,
        matter_model: str = "bsk24",
        amplitudes: float | Sequence[float] = (0.0, 0.01),
        epsilon_match: str | float = "standard",
        center: float | Sequence[float] = 200.0,
        width: float | Sequence[float] = 50.0,
        ramp_width: float | Sequence[float] = 40.0,
        calculation: str = "thermodynamics",
        fixed_masses: float | Sequence[float] = (1.4,),
        precision: str = "quick",
        diagnostics: str = "off",
        observables: Sequence[str] | None = None,
    ) -> "ExperimentSettings":
        return cls(
            matter_model=matter_model,
            amplitudes=_number_tuple("amplitudes", amplitudes),
            epsilon_match=epsilon_match,
            center=_number_tuple("center", center, positive=True),
            width=_number_tuple("width", width, positive=True),
            ramp_width=_number_tuple("ramp_width", ramp_width, positive=True),
            calculation=calculation,
            fixed_masses=_number_tuple("fixed_masses", fixed_masses, positive=True),
            precision=precision,
            diagnostics=diagnostics,
            observables=None if observables is None else tuple(observables),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentSettings":
        if not isinstance(payload, Mapping):
            raise TypeError("settings payload must be a mapping")
        values = dict(payload)
        values.pop("$schema", None)
        allowed = {
            "observables",
            "matter_model",
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
            raise ValueError(
                "experiment configuration $schema must be a non-empty string"
            )
        for name in ("amplitudes", "fixed_masses"):
            if not isinstance(payload.get(name), list):
                raise ValueError(f"experiment configuration {name} must be an array")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        data = {
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
        if self.matter_model != "bsk24":
            data["matter_model"] = self.matter_model
        if self.observables is not None:
            data["observables"] = list(self.observables)
        return data

    def deterministic_hash(self) -> str:
        return _hash_payload(self.to_dict())


def _finite_float(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must contain real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite numbers")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must contain positive numbers")
    return 0.0 if result == 0.0 else result


def _number_tuple(
    name: str,
    values: float | Sequence[float],
    *,
    positive: bool = False,
) -> tuple[float, ...]:
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        source: Iterable[Any] = (values,)
    elif isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a number or a sequence of numbers")
    else:
        source = values
    normalized = tuple(
        _finite_float(name, value, positive=positive) for value in source
    )
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate values")
    return normalized

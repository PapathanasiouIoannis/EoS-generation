"""Immutable EoS discontinuity metadata shared by domain and TOV code."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal


EOS_DISCONTINUITY_CONTRACT_VERSION = "eos_discontinuity_v1"
DISCONTINUITY_KINDS = ("internal", "surface")
BARE_SELF_BOUND_SEQUENCE_POLICY = "bare_self_bound_positive_mass_radius_v1"
SEED_PRESERVING_LOCAL_REFINEMENT_POLICY = "seed_preserving_split_log_pressure_v1"


def _finite(name: str, value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class EosDiscontinuity:
    """One explicitly declared outward energy-density discontinuity.

    ``inner_energy_density - outer_energy_density`` is the signed outward
    ``delta_energy_density``. Negative internal seams are valid and retained.
    """

    identifier: str
    kind: Literal["internal", "surface"]
    pressure: float
    inner_energy_density: float
    outer_energy_density: float
    delta_energy_density: float
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise ValueError("discontinuity identifier must be a non-empty string")
        if self.kind not in DISCONTINUITY_KINDS:
            raise ValueError(f"discontinuity kind must be one of {DISCONTINUITY_KINDS}")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("discontinuity provenance must be a non-empty string")

        pressure = _finite("discontinuity pressure", self.pressure)
        inner = _finite("inner energy density", self.inner_energy_density)
        outer = _finite("outer energy density", self.outer_energy_density)
        delta = _finite("signed outward delta energy density", self.delta_energy_density)
        if pressure < 0.0:
            raise ValueError("discontinuity pressure must be nonnegative")
        if inner < 0.0 or outer < 0.0:
            raise ValueError("one-sided energy densities must be nonnegative")
        if self.kind == "internal" and pressure <= 0.0:
            raise ValueError("internal discontinuities require positive pressure")
        if self.kind == "surface":
            if pressure != 0.0:
                raise ValueError("surface discontinuities must be declared at P=0")
            if inner <= 0.0 or outer != 0.0:
                raise ValueError("a bare surface requires positive inner density and vacuum outside")

        expected = inner - outer
        tolerance = 8.0 * math.ulp(max(abs(inner), abs(outer), 1.0))
        if not math.isclose(delta, expected, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                "delta_energy_density must equal inner_energy_density - outer_energy_density"
            )
        object.__setattr__(self, "pressure", pressure)
        object.__setattr__(self, "inner_energy_density", inner)
        object.__setattr__(self, "outer_energy_density", outer)
        object.__setattr__(self, "delta_energy_density", delta)

    @classmethod
    def from_sides(
        cls,
        *,
        identifier: str,
        kind: Literal["internal", "surface"],
        pressure: float,
        inner_energy_density: float,
        outer_energy_density: float,
        provenance: str,
    ) -> "EosDiscontinuity":
        inner = float(inner_energy_density)
        outer = float(outer_energy_density)
        return cls(
            identifier=identifier,
            kind=kind,
            pressure=pressure,
            inner_energy_density=inner,
            outer_energy_density=outer,
            delta_energy_density=inner - outer,
            provenance=provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EOS_DISCONTINUITY_CONTRACT_VERSION,
            "identifier": self.identifier,
            "type": self.kind,
            "pressure_MeV_fm3": self.pressure,
            "inner_energy_density_MeV_fm3": self.inner_energy_density,
            "outer_energy_density_MeV_fm3": self.outer_energy_density,
            "signed_outward_delta_energy_density_MeV_fm3": self.delta_energy_density,
            "provenance": self.provenance,
        }


def validate_discontinuity_sequence(
    values: Iterable[EosDiscontinuity],
) -> tuple[EosDiscontinuity, ...]:
    """Return a validated descending-pressure discontinuity tuple."""
    resolved = tuple(values)
    if any(not isinstance(item, EosDiscontinuity) for item in resolved):
        raise TypeError("discontinuities must contain only EosDiscontinuity values")
    identifiers = [item.identifier for item in resolved]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("discontinuity identifiers must be unique")
    pressures = [item.pressure for item in resolved]
    if any(upper <= lower for upper, lower in zip(pressures, pressures[1:])):
        raise ValueError("discontinuities must be in strictly descending pressure order")
    surface_indices = [index for index, item in enumerate(resolved) if item.kind == "surface"]
    if len(surface_indices) > 1:
        raise ValueError("at most one bare surface discontinuity may be declared")
    if surface_indices and surface_indices[0] != len(resolved) - 1:
        raise ValueError("a surface discontinuity must be the final ordered entry")
    return resolved


__all__ = [
    "DISCONTINUITY_KINDS",
    "EOS_DISCONTINUITY_CONTRACT_VERSION",
    "EosDiscontinuity",
    "validate_discontinuity_sequence",
]

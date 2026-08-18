"""Bounded extended stellar diagnostics for governed BSk24 experiments.

The helpers consume already planned stellar solutions. They do not define
new acceptance criteria or run merely by importing this module.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from eos_generation._internal.artifacts import write_csv_atomic
from eos_generation._internal.planning import BSk24TrialConfig
from eos_generation._internal.stellar import _tov_settings
from eos_generation.bsk24.deformation import (
    BSk24WindowedEos,
    window_characterization,
    windowed_gaussian_delta_cs2,
)
from eos_generation.bsk24.reconstruction import BSk24ConsistentBaseline
from eos_generation.stellar.diagnostics import (
    baryon_number_from_profile,
    interpolate_within_common_support,
    odd_even_response,
    pressure_profile_from_solved_star,
    radial_deformation_support,
)


def _diagnostic_case_ids(
    config: BSk24TrialConfig,
    generated: Mapping[str, BSk24WindowedEos],
) -> tuple[str, ...]:
    """Select deterministic post-gate cases for extended radial diagnostics.

    ``generated`` is the trial's reconstructed mapping, so a proposal that
    failed the complete retained-domain raw gate cannot enter either policy.
    The default preserves the established endpoint-only selection. The opt-in
    policy extends each signed branch from its strongest accepted case toward
    zero without changing acceptance, reconstruction, or stellar solving.
    """
    eligible = [
        (case_id, eos)
        for case_id, eos in generated.items()
        if eos.deformation.delta_mev_fm3 == config.diagnostic_delta_mev_fm3
    ]
    a0 = next(
        (
            case_id
            for case_id, eos in eligible
            if eos.deformation.amplitude == 0.0
        ),
        None,
    )
    negative = sorted(
        (
            (eos.deformation.amplitude, case_id)
            for case_id, eos in eligible
            if eos.deformation.amplitude < 0.0
        )
    )
    positive = sorted(
        (
            (eos.deformation.amplitude, case_id)
            for case_id, eos in eligible
            if eos.deformation.amplitude > 0.0
        ),
        reverse=True,
    )
    if a0 is None:
        return ()
    policy = getattr(
        config, "extended_stellar_diagnostics_case_policy", "endpoints"
    )
    if policy == "all-accepted":
        return tuple(
            [
                "direct",
                a0,
                *(case_id for _, case_id in negative),
                *(case_id for _, case_id in positive),
            ]
        )
    if not (negative or positive):
        return ()
    selected = ["direct", a0]
    if negative:
        selected.append(negative[0][1])
    if positive:
        selected.append(positive[0][1])
    return tuple(selected)


def _extended_diagnostics(
    *,
    packet: Path,
    config: BSk24TrialConfig,
    baseline: BSk24ConsistentBaseline,
    generated: Mapping[str, BSk24WindowedEos],
    sequences: pd.DataFrame,
    fixed: pd.DataFrame,
    stars: Mapping[tuple[str, str, float], Any],
) -> dict[str, str]:
    """Create bounded, definition-free diagnostics from already planned stars.

    Turning-point refinement, outside-support controls, and matched-area cases
    remain unavailable here because they require separately constructed cases.
    """
    created: dict[str, str] = {}
    selected = _diagnostic_case_ids(config, generated)
    if not selected:
        return created
    reference_stage = config.tov_stages[-1]
    eos_map: dict[str, Any] = {"direct": baseline.eos, **generated}
    profile_frames: list[pd.DataFrame] = []
    support_rows: list[dict[str, Any]] = []
    baryon_rows: list[dict[str, Any]] = []
    for case_id in selected:
        eos = eos_map[case_id]
        deformation = (
            None if case_id == "direct" else generated[case_id].deformation
        )
        realized_peak_absolute_delta_cs2 = None
        if deformation is not None:
            characterization = window_characterization(baseline, deformation)
            realized_peak_absolute_delta_cs2 = max(
                abs(float(characterization["realized_delta_cs2_minimum"])),
                abs(float(characterization["realized_delta_cs2_maximum"])),
            )
        for mass in config.fixed_masses_msun:
            star = stars.get((case_id, reference_stage.name, mass))
            if star is None:
                continue
            settings = _tov_settings(eos, config, reference_stage)
            pressure = np.asarray(
                pressure_profile_from_solved_star(
                    eos,
                    star,
                    settings=settings,
                    rtol=reference_stage.rtol,
                    atol=reference_stage.atol,
                ),
                dtype=float,
            )
            epsilon = np.asarray(
                eos.energy_density_from_pressure(pressure), dtype=float
            )
            if case_id == "direct":
                density = np.asarray(
                    baseline.consistent_baryon_density_from_energy_density(
                        epsilon
                    ),
                    dtype=float,
                )
                central_density = float(
                    baseline.consistent_baryon_density_from_energy_density(
                        star.central_energy_density
                    )
                )
                delta_cs2 = np.zeros_like(epsilon)
            else:
                density = np.asarray(
                    eos.baryon_density_from_energy_density(epsilon), dtype=float
                )
                central_density = float(
                    eos.baryon_density_from_energy_density(
                        star.central_energy_density
                    )
                )
                delta_cs2 = np.asarray(
                    windowed_gaussian_delta_cs2(
                        epsilon,
                        deformation,
                        epsilon_t_mev_fm3=(
                            baseline.anchor.energy_density_mev_fm3
                        ),
                    ),
                    dtype=float,
                )
            cs2 = np.asarray(
                [eos(float(value))[1] for value in pressure], dtype=float
            )
            radius = np.asarray(star.radius_profile, dtype=float)
            enclosed_mass = np.asarray(star.mass_profile, dtype=float)
            profile_frames.append(
                pd.DataFrame(
                    {
                        "case_id": case_id,
                        "target_mass_msun": mass,
                        "radius_km": radius,
                        "radius_over_R": radius / star.radius,
                        "enclosed_mass_msun": enclosed_mass,
                        "enclosed_mass_over_M": enclosed_mass / star.mass,
                        "pressure_mev_fm3": pressure,
                        "energy_density_mev_fm3": epsilon,
                        "baryon_density_fm3": density,
                        "cs2": cs2,
                        "delta_cs2": delta_cs2,
                    }
                )
            )
            baryon = baryon_number_from_profile(
                radius,
                enclosed_mass,
                density,
                central_baryon_density_fm3=central_density,
                gravitational_mass_msun=star.mass,
                solver_rtol=reference_stage.rtol,
                solver_atol=reference_stage.atol,
            )
            baryon_rows.append(
                {
                    "case_id": case_id,
                    "target_mass_msun": mass,
                    "stage": reference_stage.name,
                    **baryon.to_dict(),
                }
            )
            if deformation is not None:
                support = radial_deformation_support(
                    radius,
                    enclosed_mass,
                    delta_cs2,
                    total_radius_km=star.radius,
                    total_mass_msun=star.mass,
                    realized_peak_absolute_delta_cs2=(
                        realized_peak_absolute_delta_cs2
                    ),
                    solver_rtol=reference_stage.rtol,
                    solver_atol=reference_stage.atol,
                )
                for fraction, item in support["thresholds"].items():
                    support_rows.append(
                        {
                            "case_id": case_id,
                            "stage": reference_stage.name,
                            "amplitude": deformation.amplitude,
                            "delta_mev_fm3": deformation.delta_mev_fm3,
                            "target_mass_msun": mass,
                            "threshold_fraction": float(fraction),
                            "threshold_label": (
                                "FWHM"
                                if math.isclose(float(fraction), 0.5)
                                else f"{float(fraction):.0%}_of_realized_peak"
                            ),
                            "realized_peak_absolute_delta_cs2": support[
                                "realized_peak_absolute_delta_cs2"
                            ],
                            "stored_profile_peak_absolute_delta_cs2": support[
                                "stored_profile_peak_absolute_delta_cs2"
                            ],
                            "profile_solver_rtol": support[
                                "profile_solver_rtol"
                            ],
                            "profile_solver_atol": support[
                                "profile_solver_atol"
                            ],
                            "bounded_mass_reversal_count": support[
                                "bounded_mass_reversal_count"
                            ],
                            "maximum_bounded_mass_reversal_msun": support[
                                "maximum_bounded_mass_reversal_msun"
                            ],
                            "raw_mass_profile_preserved": support[
                                "raw_mass_profile_preserved"
                            ],
                            "threshold_absolute_delta_cs2": item[
                                "threshold_absolute_delta_cs2"
                            ],
                            "status": item["status"],
                            "interval_count": item.get("interval_count", 0),
                            "crossing_count": item.get("crossing_count", 0),
                            "reaches_profile_inner_boundary": item.get(
                                "reaches_profile_inner_boundary", False
                            ),
                            "reaches_profile_outer_boundary": item.get(
                                "reaches_profile_outer_boundary", False
                            ),
                            "inner_radius_fraction": item.get(
                                "inner_radius_fraction"
                            ),
                            "radial_span_fraction": item.get(
                                "radial_span_fraction"
                            ),
                            "inner_enclosed_mass_fraction": item.get(
                                "inner_enclosed_mass_fraction"
                            ),
                            "enclosed_mass_span_fraction": item.get(
                                "enclosed_mass_span_fraction"
                            ),
                            "outer_radius_fraction": item.get(
                                "outer_radius_fraction"
                            ),
                            "outer_enclosed_mass_fraction": item.get(
                                "outer_enclosed_mass_fraction"
                            ),
                        }
                    )
    if profile_frames:
        write_csv_atomic(
            pd.concat(profile_frames, ignore_index=True),
            packet / "radial_profiles.csv",
        )
        created["radial_profiles.csv"] = (
            "fixed-mass profiles selected by the "
            f"{config.extended_stellar_diagnostics_case_policy} case policy"
        )
    if support_rows:
        write_csv_atomic(
            pd.DataFrame(support_rows),
            packet / "deformation_support_fractions.csv",
        )
        created["deformation_support_fractions.csv"] = (
            "continuous global-peak deformation thresholds including FWHM"
        )
    baryonic = pd.DataFrame(baryon_rows)
    if not baryonic.empty:
        write_csv_atomic(baryonic, packet / "baryonic_observables.csv")
        created["baryonic_observables.csv"] = (
            "neutron-rest-mass-convention baryon integral"
        )
        direct = baryonic.loc[baryonic.case_id == "direct"].set_index(
            "target_mass_msun"
        )
        responses: list[dict[str, Any]] = []
        for row in baryonic.itertuples(index=False):
            if (
                row.case_id == "direct"
                or row.target_mass_msun not in direct.index
            ):
                continue
            reference = direct.loc[row.target_mass_msun]
            responses.append(
                {
                    "case_id": row.case_id,
                    "mass_msun": row.target_mass_msun,
                    "delta_baryonic_mass_msun": (
                        row.baryonic_mass_msun
                        - float(reference.baryonic_mass_msun)
                    ),
                    "delta_binding_energy_erg": (
                        row.binding_energy_erg
                        - float(reference.binding_energy_erg)
                    ),
                }
            )
        if responses:
            write_csv_atomic(
                pd.DataFrame(responses),
                packet / "baryonic_response_across_mass.csv",
            )
            created["baryonic_response_across_mass.csv"] = (
                "within-common-fixed-mass response"
            )
    response_rows: list[dict[str, Any]] = []
    sequence_reference = sequences.loc[
        (sequences.stage == reference_stage.name)
        & (sequences.calculation_status == "success")
    ]
    stable: dict[str, pd.DataFrame] = {}
    for case_id in selected:
        rows = sequence_reference.loc[
            sequence_reference.case_id == case_id
        ].sort_values("P_Central")
        if rows.empty:
            continue
        peak = int(np.argmax(rows.Mass.to_numpy(dtype=float)))
        stable[case_id] = rows.iloc[: peak + 1]
    if set(selected).issubset(stable):
        lower = max(float(frame.Mass.min()) for frame in stable.values())
        upper = min(float(frame.Mass.max()) for frame in stable.values())
        if upper > lower:
            masses = np.linspace(lower, upper, 80)
            direct_values = {
                column: interpolate_within_common_support(
                    stable["direct"].Mass,
                    stable["direct"][column],
                    masses,
                )
                for column in ("Radius", "k2", "Lambda", "Eps_Central")
            }
            for case_id, frame in stable.items():
                values = {
                    column: interpolate_within_common_support(
                        frame.Mass, frame[column], masses
                    )
                    for column in ("Radius", "k2", "Lambda", "Eps_Central")
                }
                for index, mass in enumerate(masses):
                    response_rows.append(
                        {
                            "case_id": case_id,
                            "mass_msun": mass,
                            "delta_radius_km": values["Radius"][index]
                            - direct_values["Radius"][index],
                            "delta_k2": values["k2"][index]
                            - direct_values["k2"][index],
                            "delta_lambda": values["Lambda"][index]
                            - direct_values["Lambda"][index],
                            "central_epsilon_mev_fm3": values["Eps_Central"][
                                index
                            ],
                            "interpolation": (
                                "PCHIP_within_common_successful_stable_prefix"
                            ),
                        }
                    )
    if response_rows:
        write_csv_atomic(
            pd.DataFrame(response_rows),
            packet / "stellar_response_across_mass.csv",
        )
        created["stellar_response_across_mass.csv"] = (
            "common successful stable-prefix response"
        )
    odd_rows: list[dict[str, Any]] = []
    reference_fixed = fixed.loc[
        (fixed.stage == reference_stage.name)
        & (fixed.status == "bracketed_and_solved")
    ]
    zero_cases = reference_fixed.loc[
        np.isclose(reference_fixed.amplitude, 0.0, equal_nan=False)
    ]
    for delta in config.deltas_mev_fm3:
        zero = zero_cases.loc[np.isclose(zero_cases.delta_mev_fm3, delta)]
        for amplitude in sorted(
            {
                abs(value)
                for value in config.effective_amplitudes
                if value != 0.0
            }
        ):
            plus = reference_fixed.loc[
                np.isclose(reference_fixed.amplitude, amplitude)
                & np.isclose(reference_fixed.delta_mev_fm3, delta)
            ]
            minus = reference_fixed.loc[
                np.isclose(reference_fixed.amplitude, -amplitude)
                & np.isclose(reference_fixed.delta_mev_fm3, delta)
            ]
            if zero.empty or plus.empty or minus.empty:
                continue
            for mass in config.fixed_masses_msun:
                z = zero.loc[np.isclose(zero.target_mass_msun, mass)]
                p = plus.loc[np.isclose(plus.target_mass_msun, mass)]
                m = minus.loc[np.isclose(minus.target_mass_msun, mass)]
                if z.empty or p.empty or m.empty:
                    continue
                for observable in (
                    "radius_km",
                    "k2",
                    "lambda_dimensionless",
                ):
                    values = (
                        float(p.iloc[0][observable]),
                        float(m.iloc[0][observable]),
                        float(z.iloc[0][observable]),
                    )
                    if not np.all(np.isfinite(values)):
                        continue
                    item = odd_even_response(
                        *values,
                        amplitude=amplitude,
                        numerical_envelope=(
                            np.finfo(float).eps * max(1.0, abs(values[2]))
                        ),
                    )
                    odd_rows.append(
                        {
                            "delta_mev_fm3": delta,
                            "target_mass_msun": mass,
                            "observable": observable,
                            **item,
                        }
                    )
    if odd_rows:
        write_csv_atomic(
            pd.DataFrame(odd_rows), packet / "odd_even_response.csv"
        )
        created["odd_even_response.csv"] = "paired amplitudes with A=0"
    error_rows: list[dict[str, Any]] = []
    if len(config.tov_stages) >= 2:
        for (case_id, target_mass), rows in fixed.loc[
            fixed.status == "bracketed_and_solved"
        ].groupby(["case_id", "target_mass_msun"]):
            for observable in (
                "radius_km",
                "k2",
                "lambda_dimensionless",
            ):
                values = pd.to_numeric(
                    rows[observable], errors="coerce"
                ).dropna()
                if len(values) >= 2:
                    error_rows.append(
                        {
                            "label": f"{case_id}:{target_mass:g}:{observable}",
                            "observable": observable,
                            "numerical_envelope": float(
                                values.max() - values.min()
                            ),
                            "reference_value": float(values.iloc[-1]),
                        }
                    )
    if error_rows:
        write_csv_atomic(
            pd.DataFrame(error_rows), packet / "numerical_error_summary.csv"
        )
        created["numerical_error_summary.csv"] = "same-case stage spans"
    return created


__all__ = ["_diagnostic_case_ids", "_extended_diagnostics"]

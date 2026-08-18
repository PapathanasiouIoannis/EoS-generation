"""Shared, fail-closed selection for saved BSk24 stellar tables.

This module is deliberately independent of the public experiment and plotting
modules.  It is the single authority for saved-row tidal validity and for the
fixed-mass populations used by response-plot applicability, rendering, and
provenance reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from eos_generation.stellar.tov import LAMBDA_FRAMEWORK_CAPABILITY


SavedStellarSchema = Literal["sequence", "fixed_mass"]
ResponseCoordinate = Literal["amplitude", "delta"]


@dataclass(frozen=True)
class _TidalSchema:
    background_column: str
    background_success: str
    lambda_column: str


_TIDAL_SCHEMAS: dict[SavedStellarSchema, _TidalSchema] = {
    "sequence": _TidalSchema(
        background_column="calculation_status",
        background_success="success",
        lambda_column="Lambda",
    ),
    "fixed_mass": _TidalSchema(
        background_column="status",
        background_success="bracketed_and_solved",
        lambda_column="lambda_dimensionless",
    ),
}


def classify_saved_tidal_rows(
    frame: pd.DataFrame,
    *,
    schema: SavedStellarSchema,
) -> pd.DataFrame:
    """Classify every saved row using the repository's full tidal definition.

    A row is valid only when its table-specific background succeeded, its
    status is exactly ``validated_lambda_validation_v1``, ``k2`` and Lambda
    are finite, and Lambda is strictly positive.
    """

    definition = _TIDAL_SCHEMAS[schema]
    result = pd.DataFrame(index=frame.index)
    result["background_success"] = False
    result["tidal_valid"] = False
    result["tidal_validity_reason"] = "unclassified"
    required = (
        definition.background_column,
        "tidal_status",
        "k2",
        definition.lambda_column,
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        result["tidal_validity_reason"] = (
            "missing_required_column:" + ",".join(missing)
        )
        return result

    background = frame[definition.background_column].astype(str).eq(
        definition.background_success
    )
    status = frame["tidal_status"].astype(str).eq(
        LAMBDA_FRAMEWORK_CAPABILITY
    )
    k2 = pd.to_numeric(frame["k2"], errors="coerce")
    lambda_value = pd.to_numeric(frame[definition.lambda_column], errors="coerce")
    finite_k2 = pd.Series(np.isfinite(k2), index=frame.index, dtype=bool)
    finite_lambda = pd.Series(
        np.isfinite(lambda_value), index=frame.index, dtype=bool
    )
    positive_lambda = lambda_value.gt(0.0)
    valid = background & status & finite_k2 & finite_lambda & positive_lambda

    reason = pd.Series("valid", index=frame.index, dtype=object)
    reason.loc[~background] = "background_not_successful"
    reason.loc[background & ~status] = "tidal_status_not_validated"
    reason.loc[background & status & ~finite_k2] = "k2_nonfinite"
    reason.loc[background & status & finite_k2 & ~finite_lambda] = (
        "lambda_nonfinite"
    )
    reason.loc[
        background & status & finite_k2 & finite_lambda & ~positive_lambda
    ] = "lambda_not_strictly_positive"

    result["background_success"] = background
    result["tidal_valid"] = valid
    result["tidal_validity_reason"] = reason
    return result


def saved_tidal_valid_mask(
    frame: pd.DataFrame,
    *,
    schema: SavedStellarSchema,
) -> pd.Series:
    """Return the shared fail-closed tidal-validity mask."""

    return classify_saved_tidal_rows(frame, schema=schema)["tidal_valid"]


def select_fixed_mass_response_population(
    frame: pd.DataFrame,
    *,
    final_stage: str,
    target_mass_msun: float,
    versus: ResponseCoordinate,
) -> pd.DataFrame:
    """Select exactly the grouped deformation rows eligible for one response.

    The direct baseline is excluded because it has no finite deformation
    coordinates.  Groups without at least two distinct response coordinates
    are excluded, so applicability and rendering cannot disagree.
    """

    if versus not in {"amplitude", "delta"}:
        raise ValueError("versus must be 'amplitude' or 'delta'")
    required = {
        "case_id",
        "stage",
        "status",
        "target_mass_msun",
        "amplitude",
        "delta_mev_fm3",
    }
    if frame.empty or not required.issubset(frame.columns):
        return frame.iloc[0:0].copy()

    target_mass = pd.to_numeric(frame["target_mass_msun"], errors="coerce")
    amplitude = pd.to_numeric(frame["amplitude"], errors="coerce")
    delta = pd.to_numeric(frame["delta_mev_fm3"], errors="coerce")
    selected = frame.loc[
        frame["stage"].astype(str).eq(str(final_stage))
        & ~frame["case_id"].astype(str).eq("direct")
        & np.isclose(target_mass, float(target_mass_msun))
        & np.isfinite(amplitude)
        & np.isfinite(delta)
    ].copy()
    if selected.empty:
        return selected
    selected["amplitude"] = pd.to_numeric(
        selected["amplitude"], errors="coerce"
    )
    selected["delta_mev_fm3"] = pd.to_numeric(
        selected["delta_mev_fm3"], errors="coerce"
    )
    x_column = "amplitude" if versus == "amplitude" else "delta_mev_fm3"
    group_column = (
        "delta_mev_fm3" if versus == "amplitude" else "amplitude"
    )
    eligible_groups = (
        selected.groupby(group_column, dropna=False)[x_column]
        .nunique(dropna=True)
        .loc[lambda values: values >= 2]
        .index
    )
    selected = selected.loc[selected[group_column].isin(eligible_groups)].copy()
    return selected.sort_values(
        [group_column, x_column, "case_id"], kind="stable"
    ).reset_index(drop=True)


def summarize_fixed_mass_response_population(
    frame: pd.DataFrame,
    *,
    final_stage: str,
    target_mass_msun: float,
    versus: ResponseCoordinate,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return the shared response rows and their exact tidal provenance."""

    selected = select_fixed_mass_response_population(
        frame,
        final_stage=final_stage,
        target_mass_msun=target_mass_msun,
        versus=versus,
    )
    background = (
        selected["status"].astype(str).eq("bracketed_and_solved")
        if "status" in selected.columns
        else pd.Series(False, index=selected.index, dtype=bool)
    )
    valid = saved_tidal_valid_mask(selected, schema="fixed_mass")
    validated = int(valid.sum())
    background_count = int(background.sum())
    total = int(len(selected))
    return selected, {
        "final_stage": str(final_stage),
        "target_mass_msun": float(target_mass_msun),
        "response_coordinate": versus,
        "eligible_deformation_row_count": total,
        "planned_deformation_row_count": total,
        "background_success_count": background_count,
        "background_unavailable_count": total - background_count,
        "tidal_validated_count": validated,
        "tidal_omitted_count": total - validated,
        "tidal_completeness_status": (
            "not_applicable_no_grouped_response_population"
            if total == 0
            else "complete_background_and_tidal"
            if validated == total
            else "partial_tidal_data"
            if validated
            else "background_only_no_validated_tides"
        ),
    }


def exact_fixed_mass_response_to_direct(
    frame: pd.DataFrame,
    *,
    final_stage: str,
    case_ids: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Return exact fixed-mass response rows relative to the direct baseline.

    Every returned point is a difference between two independently bracketed
    and solved stars at the same requested mass.  No stellar-sequence
    interpolation is performed.  Tidal differences fail closed unless both
    the deformed and direct rows satisfy the full saved-row tidal definition.
    """

    columns = (
        "case_id",
        "stage",
        "amplitude",
        "delta_mev_fm3",
        "mass_msun",
        "delta_radius_km",
        "fractional_delta_radius",
        "delta_k2",
        "fractional_delta_k2",
        "delta_lambda",
        "fractional_delta_lambda",
        "central_epsilon_mev_fm3",
        "tidal_pair_valid",
        "response_source",
    )
    required = {
        "case_id",
        "stage",
        "status",
        "target_mass_msun",
        "amplitude",
        "delta_mev_fm3",
        "radius_km",
        "central_energy_density_mev_fm3",
        "k2",
        "lambda_dimensionless",
        "tidal_status",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    selected = frame.loc[
        frame["stage"].astype(str).eq(str(final_stage))
        & frame["status"].astype(str).eq("bracketed_and_solved")
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)

    selected["target_mass_msun"] = pd.to_numeric(
        selected["target_mass_msun"], errors="coerce"
    )
    selected = selected.loc[np.isfinite(selected["target_mass_msun"])].copy()
    direct = selected.loc[selected["case_id"].astype(str).eq("direct")].copy()
    deformed = selected.loc[~selected["case_id"].astype(str).eq("direct")].copy()
    if case_ids is not None:
        requested = {str(value) for value in case_ids if str(value) != "direct"}
        deformed = deformed.loc[
            deformed["case_id"].astype(str).isin(requested)
        ].copy()
    if direct.empty or deformed.empty:
        return pd.DataFrame(columns=columns)

    direct_duplicate = direct.duplicated("target_mass_msun", keep=False)
    deformed_duplicate = deformed.duplicated(
        ["case_id", "target_mass_msun"], keep=False
    )
    if bool(direct_duplicate.any()) or bool(deformed_duplicate.any()):
        raise ValueError(
            "exact fixed-mass response requires one saved solved row per "
            "case and requested mass"
        )

    reference_columns = {
        "status": "reference_status",
        "radius_km": "reference_radius_km",
        "k2": "reference_k2",
        "lambda_dimensionless": "reference_lambda_dimensionless",
        "tidal_status": "reference_tidal_status",
    }
    reference = direct[
        ["target_mass_msun", *reference_columns]
    ].rename(columns=reference_columns)
    paired = deformed.merge(
        reference,
        on="target_mass_msun",
        how="inner",
        validate="many_to_one",
    )
    if paired.empty:
        return pd.DataFrame(columns=columns)

    numeric = (
        "amplitude",
        "delta_mev_fm3",
        "radius_km",
        "reference_radius_km",
        "k2",
        "reference_k2",
        "lambda_dimensionless",
        "reference_lambda_dimensionless",
        "central_energy_density_mev_fm3",
    )
    for column in numeric:
        paired[column] = pd.to_numeric(paired[column], errors="coerce")

    deformed_tidal_valid = saved_tidal_valid_mask(
        paired, schema="fixed_mass"
    ).to_numpy(dtype=bool)
    reference_tidal_frame = pd.DataFrame(
        {
            "status": paired["reference_status"],
            "tidal_status": paired["reference_tidal_status"],
            "k2": paired["reference_k2"],
            "lambda_dimensionless": paired["reference_lambda_dimensionless"],
        },
        index=paired.index,
    )
    reference_tidal_valid = saved_tidal_valid_mask(
        reference_tidal_frame, schema="fixed_mass"
    ).to_numpy(dtype=bool)
    tidal_pair_valid = deformed_tidal_valid & reference_tidal_valid

    radius = paired["radius_km"].to_numpy(dtype=float)
    reference_radius = paired["reference_radius_km"].to_numpy(dtype=float)
    radius_pair_valid = (
        np.isfinite(radius)
        & np.isfinite(reference_radius)
        & (reference_radius != 0.0)
    )
    delta_radius = np.where(radius_pair_valid, radius - reference_radius, np.nan)
    fractional_delta_radius = np.where(
        radius_pair_valid, delta_radius / reference_radius, np.nan
    )

    k2 = paired["k2"].to_numpy(dtype=float)
    reference_k2 = paired["reference_k2"].to_numpy(dtype=float)
    lambda_value = paired["lambda_dimensionless"].to_numpy(dtype=float)
    reference_lambda = paired["reference_lambda_dimensionless"].to_numpy(
        dtype=float
    )
    k2_pair_valid = tidal_pair_valid & (reference_k2 != 0.0)
    lambda_pair_valid = tidal_pair_valid & (reference_lambda != 0.0)
    delta_k2 = np.where(k2_pair_valid, k2 - reference_k2, np.nan)
    delta_lambda = np.where(
        lambda_pair_valid, lambda_value - reference_lambda, np.nan
    )

    result = pd.DataFrame(
        {
            "case_id": paired["case_id"].astype(str),
            "stage": paired["stage"].astype(str),
            "amplitude": paired["amplitude"],
            "delta_mev_fm3": paired["delta_mev_fm3"],
            "mass_msun": paired["target_mass_msun"],
            "delta_radius_km": delta_radius,
            "fractional_delta_radius": fractional_delta_radius,
            "delta_k2": delta_k2,
            "fractional_delta_k2": np.where(
                k2_pair_valid, delta_k2 / reference_k2, np.nan
            ),
            "delta_lambda": delta_lambda,
            "fractional_delta_lambda": np.where(
                lambda_pair_valid, delta_lambda / reference_lambda, np.nan
            ),
            "central_epsilon_mev_fm3": paired[
                "central_energy_density_mev_fm3"
            ],
            "tidal_pair_valid": tidal_pair_valid,
            "response_source": "exact_fixed_mass_bracketed_solve_no_interpolation",
        }
    )
    return result.loc[:, columns].sort_values(
        ["case_id", "mass_msun"], kind="stable"
    ).reset_index(drop=True)


__all__ = [
    "classify_saved_tidal_rows",
    "exact_fixed_mass_response_to_direct",
    "saved_tidal_valid_mask",
    "select_fixed_mass_response_population",
    "summarize_fixed_mass_response_population",
]

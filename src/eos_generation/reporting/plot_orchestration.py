"""Saved-table plot inventory and generation orchestration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from eos_generation._internal.artifacts import write_csv_atomic, write_json_atomic
from eos_generation._internal.packet_documents import (
    _write_packet_ledger,
)
from eos_generation._internal.packet_integrity import (
    _refresh_manifest,
)
from eos_generation._internal.planning import (
    PLOT_REGISTRY,
    BSk24TrialConfig,
    _assert_writable_packet_path,
    _json_records,
    _selected_groups,
)
from eos_generation._internal.provenance import _source_hashes
from eos_generation._internal.saved_tables import (
    summarize_fixed_mass_response_population,
)
from eos_generation._internal.summary import PACKET_SCHEMA_ID, write_packet_summary
from eos_generation._internal.summary import CFL_PACKET_SCHEMA_ID


WINDOWED_FIGURES = (
    "window_profiles.png",
    "gaussian_realization.png",
    "raw_cs2_full_domain.png",
    "raw_cs2_anchor_core_zoom.png",
    "delta_cs2.png",
    "pressure_response.png",
    "baryon_density_response.png",
    "effective_baryon_enthalpy_response.png",
    "gamma_eff_response.png",
    "thermodynamic_residuals.png",
    "stellar_mr_k2_lambda.png",
    "observable_response_vs_amplitude.png",
    "observable_response_vs_delta.png",
    "a0_identity.png",
)
EXTENDED_FIGURES = (
    "radial_structure_profiles.png",
    "deformation_support_fractions.png",
    "outside_support_control.png",
    "turning_point_sequences.png",
    "turning_point_derivatives.png",
    "baryonic_mass_vs_mass.png",
    "binding_energy_vs_mass.png",
    "stellar_response_across_mass.png",
    "baryonic_response_across_mass.png",
    "odd_even_response.png",
    "matched_area_comparison.png",
    "numerical_error_summary.png",
)
ALL_FIGURES = WINDOWED_FIGURES + EXTENDED_FIGURES
_TIDAL_PLOT_SCOPES: Mapping[str, tuple[str, ...]] = {
    "stellar_mr_k2_lambda.png": ("sequence", "fixed_mass"),
    "observable_response_vs_amplitude.png": ("fixed_mass",),
    "observable_response_vs_delta.png": ("fixed_mass",),
    "outside_support_control.png": ("fixed_mass",),
    "stellar_response_across_mass.png": ("fixed_mass",),
    "odd_even_response.png": ("fixed_mass",),
    "matched_area_comparison.png": ("fixed_mass",),
    "numerical_error_summary.png": ("fixed_mass",),
}


def _actual_plot_inventory(
    packet: Path,
    config: Any,
    *,
    groups: Sequence[str],
) -> pd.DataFrame:
    selected = set(_selected_groups(groups))
    rows: list[dict[str, Any]] = []
    for spec in PLOT_REGISTRY:
        status = "applicable"
        reason = "all prerequisites satisfied"
        tidal_completeness_status = "not_applicable"
        tidal_validated_count: int | None = None
        tidal_omitted_count: int | None = None
        population_stage: str | None = None
        population_target_mass_msun: float | None = None
        eligible_response_row_count: int | None = None
        if spec.group not in selected:
            status = "skipped"
            reason = f"plot group {spec.group!r} was not requested"
        else:
            missing = [
                name
                for name in spec.required_tables
                if not (packet / name).is_file()
            ]
            if missing:
                status = "skipped"
                reason = (
                    f"missing prerequisite table(s): {', '.join(missing)}; "
                    f"requires {spec.prerequisite}"
                )
        # Saved response-population evidence is mandatory even when rendering
        # is disabled. Keep an unrequested figure skipped, but record the same
        # final-stage population that the unchanged validator recomputes.
        if (packet / "fixed_mass_observables.csv").is_file() and spec.filename in {
            "observable_response_vs_amplitude.png",
            "observable_response_vs_delta.png",
        }:
            table = pd.read_csv(packet / "fixed_mass_observables.csv")
            if not config.tov_stages or not config.fixed_masses_msun:
                status, reason = (
                    "skipped",
                    "requires a final requested TOV stage and fixed target mass",
                )
            else:
                versus = (
                    "amplitude"
                    if spec.filename
                    == "observable_response_vs_amplitude.png"
                    else "delta"
                )
                population_stage = config.tov_stages[-1].name
                population_target_mass_msun = min(
                    config.fixed_masses_msun,
                    key=lambda value: abs(value - 1.4),
                )
                population, population_summary = (
                    summarize_fixed_mass_response_population(
                        table,
                        final_stage=population_stage,
                        target_mass_msun=population_target_mass_msun,
                        versus=versus,
                    )
                )
                eligible_response_row_count = int(len(population))
                tidal_validated_count = int(
                    population_summary["tidal_validated_count"]
                )
                tidal_omitted_count = int(
                    population_summary["tidal_omitted_count"]
                )
                tidal_completeness_status = str(
                    population_summary["tidal_completeness_status"]
                )
                if population.empty and status == "applicable":
                    status, reason = (
                        "skipped",
                        "no final-stage fixed-mass deformation group has at "
                        f"least two distinct {versus} values",
                    )
                elif tidal_omitted_count and status == "applicable":
                    status = "applicable_partial"
                    reason = (
                        f"shared {versus} response population retains "
                        f"{tidal_validated_count} and omits "
                        f"{tidal_omitted_count} tidal row(s)"
                    )
        if status == "applicable" and spec.filename in {
            "baryonic_mass_vs_mass.png",
            "binding_energy_vs_mass.png",
        }:
            table = pd.read_csv(packet / "baryonic_observables.csv")
            if table.target_mass_msun.nunique() < 2:
                status, reason = (
                    "skipped",
                    "requires baryon integration at two or more fixed masses",
                )
        if status == "applicable" and spec.filename == "stellar_mr_k2_lambda.png":
            from eos_generation.reporting.plotting import (
                stellar_tidal_plot_completeness,
            )

            completeness = stellar_tidal_plot_completeness(
                packet, config=config
            )
            tidal_completeness_status = str(completeness["status"])
            tidal_validated_count = int(
                completeness["sequence_tidal_validated_count"]
                + completeness["fixed_mass_tidal_validated_count"]
            )
            tidal_omitted_count = int(
                completeness["sequence_tidal_omitted_count"]
                + completeness["fixed_mass_tidal_omitted_count"]
            )
            if tidal_completeness_status in {
                "partial_tidal_data",
                "background_only_no_validated_tides",
            }:
                status = "applicable_partial"
                reason = (
                    "background M-R remains applicable; exact-status tidal plotting "
                    f"retains {tidal_validated_count} and omits "
                    f"{tidal_omitted_count} row(s)"
                )
        if (
            status == "applicable"
            and spec.filename in _TIDAL_PLOT_SCOPES
            and spec.filename
            not in {
                "observable_response_vs_amplitude.png",
                "observable_response_vs_delta.png",
            }
        ):
            summary_path = packet / "stellar_status_summary.csv"
            if summary_path.is_file() and config.tov_stages:
                summary = pd.read_csv(summary_path)
                stage = config.tov_stages[-1].name
                relevant = summary.loc[
                    summary["stage"].astype(str).eq(stage)
                    & summary["scope"]
                    .astype(str)
                    .isin(_TIDAL_PLOT_SCOPES[spec.filename])
                ]
                failed = int(
                    relevant.get(
                        "tidal_failed_closed_count", pd.Series(dtype=int)
                    ).sum()
                )
                unavailable = int(
                    relevant.get(
                        "tidal_unavailable_count", pd.Series(dtype=int)
                    ).sum()
                )
                tidal_validated_count = int(
                    relevant.get(
                        "tidal_validated_count", pd.Series(dtype=int)
                    ).sum()
                )
                tidal_omitted_count = failed + unavailable
                tidal_completeness_status = (
                    "complete_background_and_tidal"
                    if tidal_omitted_count == 0
                    else "partial_tidal_data"
                )
                if failed or unavailable:
                    status = "applicable_partial"
                    reason = (
                        f"background panels remain applicable; reference stage {stage!r} "
                        f"omits {failed} failed-closed and {unavailable} unavailable "
                        "tidal row(s)"
                    )
        rows.append(
            {
                "figure": spec.filename,
                "group": spec.group,
                "status": status,
                "reason": reason,
                "prerequisite": spec.prerequisite,
                "relative_path": f"plots/{spec.filename}",
                "tidal_completeness_status": tidal_completeness_status,
                "tidal_validated_count": tidal_validated_count,
                "tidal_omitted_count": tidal_omitted_count,
                "population_stage": population_stage,
                "population_target_mass_msun": population_target_mass_msun,
                "eligible_response_row_count": eligible_response_row_count,
            }
        )
    return pd.DataFrame(rows)


def _append_radial_companion_inventory(
    packet: Path,
    inventory: pd.DataFrame,
    *,
    radial_generated: bool,
) -> pd.DataFrame:
    """Record every dynamically named per-case radial figure."""

    if not radial_generated:
        return inventory
    primary_rows = inventory.loc[
        inventory.figure.astype(str).eq("radial_structure_profiles.png")
    ]
    if primary_rows.empty:
        return inventory
    known = set(inventory.figure.astype(str))
    additions: list[dict[str, Any]] = []
    template = primary_rows.iloc[0].to_dict()
    for path in sorted(
        (packet / "plots").glob("radial_structure_profiles_*.png")
    ):
        if path.name in known:
            continue
        row = dict(template)
        row.update(
            {
                "figure": path.name,
                "status": "generated",
                "reason": "per-case radial structure generated from saved profiles",
                "relative_path": path.relative_to(packet).as_posix(),
            }
        )
        additions.append(row)
        known.add(path.name)
    if not additions:
        return inventory
    return pd.concat((inventory, pd.DataFrame(additions)), ignore_index=True)


def generate_trial_plots_from_saved_tables(
    packet_path: str | Path,
    *,
    groups: Sequence[str] = ("all-applicable",),
    authorize_plot_overwrite: bool = False,
    _initial_packet_generation: bool = False,
) -> pd.DataFrame:
    """Generate applicable figures from saved tables under write authority."""
    packet = _assert_writable_packet_path(packet_path)
    metadata_path = packet / "metadata.json"
    metadata: dict[str, Any] | None = None
    if _initial_packet_generation:
        if not metadata_path.is_file():
            raise RuntimeError(
                "initial plot generation requires packet metadata"
            )
        initial_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        metadata = initial_metadata
        if initial_metadata.get("packet_status") not in {
            "calculations_complete_plots_pending",
            "saved_tables_promoted_plots_pending",
        }:
            raise RuntimeError(
                "private initial plot generation is restricted to an "
                "in-progress fresh packet"
            )
    elif not authorize_plot_overwrite:
        raise PermissionError(
            "plot regeneration would mutate an existing packet; pass the dedicated "
            "authorize_plot_overwrite=True permission"
        )
    elif not metadata_path.is_file():
        raise RuntimeError("plot regeneration requires packet metadata")
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    packet_schema = (
        metadata.get("schema_id") if isinstance(metadata, dict) else None
    )
    if packet_schema not in {PACKET_SCHEMA_ID, CFL_PACKET_SCHEMA_ID}:
        raise ValueError(
            "plot generation requires a recognized packet schema; found "
            f"{packet_schema!r}"
        )
    config_path = packet / "complete_configuration.json"
    if not config_path.is_file():
        raise ValueError(
            "plot regeneration requires a trial complete_configuration.json"
        )
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    matter_model = str(config_payload.get("matter_model", "bsk24"))
    if matter_model == "cfl":
        from eos_generation.cfl.planning import CFLTrialConfig

        if packet_schema != CFL_PACKET_SCHEMA_ID:
            raise ValueError("CFL configuration and packet schema disagree")
        config = CFLTrialConfig.from_dict(config_payload)
    elif matter_model == "bsk24":
        if packet_schema != PACKET_SCHEMA_ID:
            raise ValueError("BSk24 configuration and packet schema disagree")
        config = BSk24TrialConfig.from_dict(config_payload)
    else:
        raise ValueError(f"unsupported saved matter_model: {matter_model!r}")
    inventory = _actual_plot_inventory(packet, config, groups=groups)
    applicable = inventory.loc[
        inventory.status.isin(("applicable", "applicable_partial")), "figure"
    ].tolist()
    partial_applicable = set(
        inventory.loc[inventory.status == "applicable_partial", "figure"]
    )
    from eos_generation.reporting.plotting import (
        render_trial_figures,
    )

    generated = render_trial_figures(packet, applicable, config=config)
    generated_set = set(generated)
    generated_complete = generated_set - partial_applicable
    generated_partial = generated_set & partial_applicable
    inventory.loc[
        inventory.figure.isin(generated_complete), "status"
    ] = "generated"
    inventory.loc[
        inventory.figure.isin(generated_partial), "status"
    ] = "generated_partial"
    inventory.loc[
        inventory.figure.isin(generated_set), "reason"
    ] = inventory.loc[inventory.figure.isin(generated_set)].apply(
        lambda row: (
            row["reason"]
            if row["status"] == "generated_partial"
            else "generated from saved packet tables"
        ),
        axis=1,
    )
    not_rendered = inventory.status.isin(
        ("applicable", "applicable_partial")
    ) & ~inventory.figure.isin(generated_set)
    inventory.loc[not_rendered, "status"] = "skipped"
    inventory.loc[
        not_rendered, "reason"
    ] = "renderer reported no nonempty valid data"
    inventory = _append_radial_companion_inventory(
        packet,
        inventory,
        radial_generated="radial_structure_profiles.png" in generated_set,
    )
    write_csv_atomic(inventory, packet / "plot_inventory.csv")
    write_json_atomic(
        {
            "figures": _json_records(inventory),
            "generated": _json_records(
                inventory.loc[
                    inventory.status.isin(
                        ("generated", "generated_partial")
                    )
                ]
            ),
            "generated_complete": _json_records(
                inventory.loc[inventory.status == "generated"]
            ),
            "generated_partial": _json_records(
                inventory.loc[inventory.status == "generated_partial"]
            ),
            "skipped": _json_records(
                inventory.loc[inventory.status == "skipped"]
            ),
        },
        packet / "plot_inventory.json",
    )
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["plot_inventory_status"] = {
            "generated": int(
                inventory.status.isin(
                    ("generated", "generated_partial")
                ).sum()
            ),
            "generated_complete": int(
                (inventory.status == "generated").sum()
            ),
            "generated_partial": int(
                (inventory.status == "generated_partial").sum()
            ),
            "skipped": int((inventory.status == "skipped").sum()),
        }
        metadata["plot_tidal_completeness"] = _json_records(
            inventory.loc[
                inventory["tidal_completeness_status"] != "not_applicable",
                [
                    "figure",
                    "status",
                    "tidal_completeness_status",
                    "tidal_validated_count",
                    "tidal_omitted_count",
                    "population_stage",
                    "population_target_mass_msun",
                    "eligible_response_row_count",
                ],
            ]
        )
        write_json_atomic(metadata, metadata_path)
    provenance_path = packet / "plot_generation_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
    else:
        provenance = {
            "schema_id": "eos_generation_plot_generation_provenance_v1",
            "calculation_source_hashes_remain_in": "source_hashes.json",
            "events": [],
        }
    provenance["events"].append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "saved_table_plot_generation",
            "groups": list(groups),
            "generated_figures": sorted(generated_set),
            "generated_partial_figures": sorted(generated_partial),
            "tidal_completeness_source": "stellar_status_summary.csv",
            "plot_canvas_annotation_policy": (
                "scientific_axes_titles_legends_and_colorbars_only_v1"
            ),
            "detailed_reporting_locations": [
                "metadata.json",
                "summary.md",
                "plot_inventory.csv",
                "plot_inventory.json",
                "plot_generation_provenance.json",
            ],
            "physics_reconstruction_run": False,
            "tov_or_tidal_run": False,
            "plot_source_hashes": {
                relative: digest
                for relative, digest in _source_hashes().items()
                if relative
                in {
                    "src/eos_generation/experiment.py",
                    "src/eos_generation/reporting/plot_orchestration.py",
                    "src/eos_generation/_internal/saved_tables.py",
                    "src/eos_generation/reporting/plotting.py",
                }
            },
        }
    )
    write_json_atomic(provenance, provenance_path)
    if not _initial_packet_generation:
        write_packet_summary(packet)
        _write_packet_ledger(packet)
        _refresh_manifest(packet)
    return inventory


def generate_bsk24_trial_plots(
    packet_path: str | Path,
    *,
    groups: Sequence[str] = ("all-applicable",),
    authorize_plot_overwrite: bool = False,
    _initial_packet_generation: bool = False,
) -> pd.DataFrame:
    """Backward-compatible name for saved-table governed plot generation."""

    return generate_trial_plots_from_saved_tables(
        packet_path,
        groups=groups,
        authorize_plot_overwrite=authorize_plot_overwrite,
        _initial_packet_generation=_initial_packet_generation,
    )


__all__ = [
    "ALL_FIGURES",
    "EXTENDED_FIGURES",
    "WINDOWED_FIGURES",
    "_actual_plot_inventory",
    "generate_bsk24_trial_plots",
    "generate_trial_plots_from_saved_tables",
]

# CSV data guide

This guide explains how the saved CSV files relate to one another, what one
row represents, how rows are ordered, and how to prepare the data for plots or
machine learning. It describes saved results; it does not introduce a second
calculation or reinterpret the physical model.

Use CSV files only from an experiment that has completed and passed
validation. For a general-notebook run, `STUDENT_VIEW/` is the easiest place
to find the main tables. Its CSV files are byte-for-byte copies of artifacts
in the authoritative geometry packets, and its generated
`DATA_DICTIONARY.md` lists the exact headers present in that run. The focused
notebooks do not create `STUDENT_VIEW/`; use their derived output described
below or read the authoritative geometry packets directly.

## Start with the primary tables

The authoritative geometry packets always use canonical `case_id` values.
Derived notebook views add convenience labels differently:

- the general BSk24 notebook builds `EOS_DATA/` with persistent H labels,
  `case_aliases.csv`, and `eos_catalogue.csv` backed by the append-only
  `runs/eos_catalogue/` registry;
- the focused BSk24 dataset notebook builds only five plots and creates no
  `EOS_DATA/` or friendly labels;
- the general CFL notebook uses run-local C labels in its combined view; and
- the focused CFL dataset route creates only
  `CFL_DATASET/cfl_eos_data.csv` and `cfl_stellar_data.csv`, with run-local
  `cfl_0`, `cfl_1`, ... labels.

All derived tables retain canonical experiment, geometry, and case provenance.
Friendly identifiers never replace packet identity and must not be used as
machine-learning features. Persistent BSk24 labels reuse an exact physical
model definition across compatible evaluations, but stages and evaluation
provenance remain separate. Rejected proposals have no accepted-EoS label,
and missing maximum mass does not erase an independently accepted EoS.
Archive the BSk24 registry with its data; do not delete, edit, or reset it.
Full identity rules are in [`results.md`](results.md#persistent-bsk24-eos_data).

The focused CFL export is intentionally smaller than the packet tables. Its
ordered headers are:

```text
cfl_eos_data.csv:
label,experiment,geometry_id,case_id,status,amplitude,epsilon0_mev_fm3,sigma_mev_fm3,delta_mev_fm3,epsilon_mev_fm3,pressure_mev_fm3,cs2

cfl_stellar_data.csv:
label,experiment,geometry_id,case_id,status,amplitude,epsilon0_mev_fm3,sigma_mev_fm3,delta_mev_fm3,stage,attempted_index,calculation_status,failure_category,failure_reason,central_pressure_mev_fm3,Mass,Radius,tidal_status,tidal_failure_reason,k2,Lambda,is_sampled_peak
```

`cfl_0` is the accepted analytical `direct` baseline, with
`geometry_id=baseline`, zero amplitude, and blank geometry coordinates.
Accepted nonzero cases receive `cfl_1`, `cfl_2`, ... in stable geometry/case
order. Only the final configured numerical stage is exported, but all of its
attempted rows, failures, and gaps are retained. The five figures stop at the
sampled peak; `k2` and `Lambda` points require
`validated_lambda_validation_v1`. That sampled peak is not a resolved
maximum mass.

These derived CSVs omit packet auxiliaries, surface-jump evidence, fixed-mass
rows, and maximum-mass evidence. They never replace the sealed packets; use
the authoritative packet when those fields are needed.

Each `geometry_NNN/` directory in `STUDENT_VIEW/03_PRIMARY_DATA/` contains the
tables applicable to that geometry and calculation:

| File | What it answers |
|---|---|
| `case_ledger.csv` | Which deformation does a `case_id` represent, was it hard-valid, and which requested results are available? |
| `thermodynamic_profiles.csv` | What are the saved pressure, sound-speed, and reconstructed thermodynamic values along each EoS? |
| `stellar_sequences.csv` | What stellar models were attempted along each saved central-pressure sequence? |
| `fixed_mass_observables.csv` | Was each requested gravitational mass bracketed, and what observables were saved? |
| `maximum_mass_screening.csv` | Was a maximum-mass turning point resolved, and what evidence supports that status? |

A thermodynamics-only run correctly contains only the first two primary
tables. Keep `case_ledger.csv` with every downstream table copied from the
same geometry. The profile table alone does not repeat the Gaussian center,
Gaussian width, anchor, or complete experiment provenance on every row.

## Identity hierarchy

The saved hierarchy is:

```text
experiment (one canonical settings hash)
└── geometry_NNN (one anchor, center, width, and ramp-width combination)
    └── case_id (one baseline or one deformation amplitude)
        └── sampled rows (thermodynamic points or stellar models)
```

### Experiment identity

One experiment is bound to canonical settings, source and runtime identities,
and a reviewed plan. The canonical settings hash is shown in
`STUDENT_VIEW/01_READ_ME_FIRST.md` and saved in the authoritative experiment
documents. A `case_id` is not a substitute for that experiment identity.

When combining experiments, retain at least:

- the canonical settings hash;
- the authoritative experiment location or archival identifier;
- the `geometry_NNN` name;
- the `case_id`;
- the numerical `stage`, when the table has one.

### Geometry order

The public settings form a Cartesian product without numerically sorting the
declared values. Geometry numbering uses this stable nesting order:

1. `center` in its declared order;
2. within each center, `width` in its declared order;
3. within each center and width, `ramp_width` in its declared order.

For example:

```text
center     = [200, 300]
width      = [40, 50]
ramp_width = [20, 30]

geometry_001 = center 200, width 40, ramp width 20
geometry_002 = center 200, width 40, ramp width 30
geometry_003 = center 200, width 50, ramp width 20
geometry_004 = center 200, width 50, ramp width 30
geometry_005 = center 300, width 40, ramp width 20
...
```

Use the numerical coordinates in `case_ledger.csv` rather than inferring a
geometry solely from its folder number.

### Case identity and order

Inside one geometry, deformation proposals follow the declared amplitude
order. They are not sorted from negative to positive or by magnitude. If the
user did not declare zero, the governed `A = 0` identity control is inserted
before the declared amplitudes.

There are three important case types:

- `case_id = direct` is the undeformed analytical baseline for the selected
  matter model (BSk24 or frozen CFL). It appears
  in thermodynamic and stellar result tables, but it is not a deformation
  proposal and therefore has no row in `case_ledger.csv`.
- The deterministic case ID with `amplitude = 0` is a logical identity
  control. In a public Cartesian experiment, exactly one geometry owns its
  physical reconstruction; all other geometry rows alias that same physical
  baseline. The owner is expected to reproduce `direct` under the saved
  identity policy. The owning thermodynamic packet can retain an identical
  block keyed by the ledger's `physical_case_id`; nonowner packets have no
  zero-amplitude profile block. Stellar output always reuses the one
  `direct` solution.
- Every accepted nonzero-amplitude `case_id` is one distinct deformed EoS for
  that geometry.

Rejected or unresolved proposals remain in `case_ledger.csv` and the
applicable raw-gate diagnostics. Their complete raw proposals are retained as
evidence, but they deliberately have no reconstructed thermodynamic profile
or stellar sequence.

A readable ID such as `dp20_am0p1_<digest>` includes a ramp-width and amplitude
prefix, but the final digest is calculated from the complete deformation
coordinates. Always use the ledger as the mapping authority; do not reverse
engineer scientific metadata from the filename-like text.

## Row ordering and EoS boundaries

The CSV files preserve deterministic block order, but they are not all
globally sorted by a physical quantity.

| Table | Outer block order | Order within one block |
|---|---|---|
| `case_ledger.csv` | Declared effective amplitude order | One row per proposal |
| `thermodynamic_profiles.csv` | `direct`, then accepted cases in proposal order | Strictly increasing `epsilon_mev_fm3`; pressure is also strictly increasing within an accepted EoS |
| `stellar_sequences.csv` | Numerical stage, then `direct`, then accepted cases | Increasing `attempted_index` and central pressure |
| `fixed_mass_observables.csv` | Numerical stage, then `direct`, then accepted cases | Requested fixed masses in their declared order |
| `maximum_mass_screening.csv` | Numerical stage, then `direct`, then accepted cases | One assessment per case and stage |

In `thermodynamic_profiles.csv`, all rows for one case are contiguous. The
first case block is `direct`; when that block ends, the next case restarts at
the lower energy-density coordinate. The complete CSV is therefore not
globally pressure-increasing. A new spreadsheet row is not a new EoS—a change
in `case_id` is the EoS boundary.

Within one thermodynamic case, energy density is the independent saved
coordinate. BSk24 reconstruction grids are nonuniform: they are logarithmic
below the anchor and linear from the anchor to the case endpoint. If BSk24
reaches `c_s^2 = 1`, the refined first crossing is the included last point, so
accepted BSk24 cases can end at different coordinates. CFL accepted profiles
instead span the complete formula-derived surface-to-endpoint domain; a causal
failure anywhere rejects the proposal. Never substitute row number for energy
density. Join or compare cases using saved coordinates on a common valid
domain.

Within one stellar case and stage, central pressure increases. Mass need not
increase over the complete sequence: models after the sampled mass peak can
belong to the decreasing-mass branch. Do not sort a stellar sequence by mass
and then infer stability from the new order. Every attempted and solved
central pressure must remain inside that case's retained EoS endpoint.

## Keys and table relationships

Use these logical keys within one geometry:

| Table | Logical key or grouping key |
|---|---|
| `case_ledger.csv` | `case_id` |
| `thermodynamic_profiles.csv` | `case_id`, `epsilon_mev_fm3` |
| `stellar_sequences.csv` | `case_id`, `stage`, `attempted_index` |
| `fixed_mass_observables.csv` | `case_id`, `stage`, `target_mass_msun` |
| `maximum_mass_screening.csv` | `case_id`, `stage` |

Join downstream nonzero deformed-case rows to the ledger with `case_id`;
their logical and physical IDs are identical. A left join correctly leaves
the analytical `direct` baseline without proposal metadata. Do not use that
join for amplitude zero: the owning thermodynamic block is keyed by
`physical_case_id`, while several logical zero rows can alias it, and stellar
tables use `direct`. Preserve those logical controls separately rather than
duplicating the physical baseline. Across geometries or experiments, first
add the geometry name and experiment hash to the user's analysis table;
`case_id` alone is not the complete provenance key.

## `case_ledger.csv`

One row represents one declared or injected deformation proposal and its
final lifecycle status. The analytical `direct` baseline is the intentional
exception and has no ledger row.

| Column | Meaning |
|---|---|
| `case_id` | Deterministic identifier for the complete deformation coordinates |
| `amplitude` | Dimensionless additive amplitude `A` applied to `c_s^2` |
| `epsilon_match_mev_fm3` | Total-energy-density reconstruction anchor in MeV fm^-3 |
| `anchor_mode` | Saved categorical anchor selection: `standard` or `exploratory` for BSk24, and `self_bound_surface` for CFL |
| `epsilon0_mev_fm3` | Gaussian center in MeV fm^-3 |
| `sigma_mev_fm3` | Gaussian standard deviation in MeV fm^-3 |
| `delta_mev_fm3` | Smootherstep activation-ramp width in MeV fm^-3; this is not a pressure difference |
| `status` | Final `accepted` or `rejected` lifecycle status |
| `acceptance_domain` | Model-specific accepted domain: a BSk24 retained prefix or the complete governed CFL domain |
| `raw_gate_status` | Exact accepted, rejected, or unresolved raw-local-physics gate outcome |
| `full_domain_gate_status` | Summary of complete raw assessment. BSk24 can record raw noncausality beyond an accepted first crossing; CFL acceptance requires a full-domain pass. |
| `selected_domain_status` | Separate acceptance, rejection, or unresolved status of the selected usable domain; preserve it with the complete raw evidence |
| `complete_raw_proposal_causal_through_direct_endpoint` | Boolean complete-domain causality result for an accepted proposal; it can be false for a BSk24 case accepted only through its first continuous causal crossing |
| `retained_epsilon_max_mev_fm3` | Included upper total-energy-density endpoint of the accepted reconstructed branch; blank for a rejected or unresolved proposal |
| `retained_pressure_max_mev_fm3` | Pressure at that included retained endpoint in MeV fm^-3; blank when no retained reconstruction exists |
| `retained_endpoint_reason` | One of `direct_bsk24_causal_endpoint`, `published_bsk24_fit_endpoint`, `first_continuous_causal_crossing`, or `formula_derived_cfl_domain_endpoint` for an accepted row |
| `rejection_reason` | Blank for accepted cases; strict JSON text describing the first saved failure for rejected cases |
| `pressure_reconstruction` | Whether reconstruction completed or was skipped after raw-gate rejection |
| `stellar_calculation` | Whether stellar work was disabled, completed with its saved availability statuses, incomplete/failed, or skipped after rejection |
| `requested_fixed_masses_status` | Final-reporting-stage availability of all explicitly requested fixed-mass rows; it is independent of maximum-mass resolution |
| `maximum_mass_availability_status` | Final-reporting-stage maximum-mass availability, including an explicit unresolved or endpoint-limited outcome |
| `student_view_eligibility_status` | Whether the case is eligible for student-facing use; for a stellar case this requires every configured target row at the final reporting stage to be uniquely present and `bracketed_and_solved`, but does not require maximum-mass availability |
| `clipping_or_repair` | Records that no failed result was made acceptable by clipping or repair |

Public Cartesian BSk24 and CFL ledgers also include `physical_case_id` and
`is_physical_case_alias`. These make the shared physical identity of
zero-amplitude logical controls explicit; nonzero cases use their own
`case_id` as the physical ID.

Choose EoSs for downstream regression or plotting only from ledger rows whose
saved status is appropriate for that task. Rejection is a valid labelled
outcome, not a missing positive example.

## `thermodynamic_profiles.csv`

One row is one sampled total-energy-density point belonging to the EoS named
by `case_id`. The full-profile file contains `direct` plus every accepted
reconstructed case; rejected and unresolved cases are absent. Accepted BSk24
cases stop at their included case-specific endpoint. Accepted CFL cases span
the complete frozen domain. Complete raw evidence, including BSk24 values
after an earlier retained crossing, belongs in `raw_gate_profiles.csv` rather
than this reconstructed table.

| Column | Meaning |
|---|---|
| `case_id` | Baseline or deformation grouping key |
| `amplitude` | Proposal amplitude; blank for `direct` |
| `delta_mev_fm3` | Activation-ramp width; blank for `direct` |
| `epsilon_mev_fm3` | Total energy density including rest-mass energy, in MeV fm^-3 |
| `pressure_mev_fm3` | Pressure in MeV fm^-3 |
| `cs2` | Dimensionless `dP/dε = c_s^2` in units with `c = 1` |
| `delta_cs2` | Saved pointwise deformation contribution to `c_s^2`; zero for `direct` |
| `baryon_density_fm3` | Effective baryon number density in fm^-3 |
| `effective_baryon_enthalpy_mev` | Effective `(ε + P) / n_B`, equivalently the reconstructed baryon chemical potential, in MeV |
| `gamma_eff` | Effective adiabatic index `((ε + P) / P) c_s^2`, dimensionless |
| `energy_per_baryon_minus_neutron_rest_mev` | `ε / n_B` minus the neutron rest energy, in MeV |
| `pressure_relative_to_direct` | Fractional `(case - direct) / direct` pressure response at the saved energy density |
| `baryon_density_relative_to_direct` | Fractional effective baryon-density response relative to `direct` |
| `enthalpy_relative_to_direct` | Fractional effective baryon-enthalpy response relative to `direct` |

The three relative columns are fractions, not percentages. A value of `0.02`
means a two-percent relative change. The baryon-density and enthalpy responses
are zero for `direct`. The direct pressure response is zero wherever direct
pressure is nonzero; at the exact CFL zero-pressure surface its denominator is
zero, so the saved value is blank/NaN rather than a manufactured zero.

CFL full profiles additionally include `physical_case_id`, `matter_model`,
`baryon_chemical_potential_mev`, and
`quark_chemical_potential_mev`. The baryon chemical potential is the same
effective one-fluid quantity as `effective_baryon_enthalpy_mev`. The frozen
analytical baseline provides the quark chemical potential; reconstructed
deformations deliberately leave that microscopic column blank.

The `dataset_40_curves` route intentionally writes a reduced profile with
only `case_id`, `amplitude`, `delta_mev_fm3`, `epsilon_mev_fm3`,
`pressure_mev_fm3`, and `cs2`. Do not expect the auxiliary full-profile
columns from a curve-only packet.

Finite auxiliary columns such as `gamma_eff` and effective baryon-enthalpy or
chemical-potential trends are diagnostic data. Their finite magnitudes do not
alone decide hard acceptance. Non-finite or unusable reconstruction and
broken matching, interpolation, or inversion remain fail-closed conditions.

The reconstructed state is an effective one-fluid cold barotrope. These
columns do not establish particle fractions, species chemical potentials, or
beta equilibrium.

## `stellar_sequences.csv`

This table exists only when stellar work was requested. One row represents
one attempted stellar model for one EoS and one numerical stage. The table
retains failed attempts with a reason; fields that require a successful model
remain blank in those rows.

| Column | Meaning |
|---|---|
| `case_id` | Baseline or deformation grouping key |
| `stage` | Governed numerical stage name |
| `attempted_index` | Zero-based position in the increasing central-pressure attempt grid |
| `segment_id` | Counter separating contiguous success segments; it advances after a saved background failure |
| `calculation_status` | `success` or `failed` for the background stellar model |
| `failure_category` | Saved failure category for a failed attempt |
| `failure_reason` | Saved detailed reason for a failed attempt |
| `Mass` | Gravitational mass in solar masses for a successful model |
| `Radius` | Radius in kilometres |
| `Lambda` | Dimensionless tidal deformability |
| `P_Central` | Central pressure in MeV fm^-3 |
| `Eps_Central` | Central total energy density in MeV fm^-3 |
| `CS2_Central` | Dimensionless central sound-speed squared |
| `eps_surf` | Saved surface energy-density convention in MeV fm^-3 |
| `central_pressure_mev_fm3` | Explicitly unit-labelled central-pressure coordinate; it duplicates `P_Central` for successful rows and is retained for failed attempts |
| `is_sampled_peak` | Marks the largest mass on the sampled successful sequence; it is not by itself a resolved maximum mass |
| `is_domain_end` | Marks the final successful saved sequence model |
| `k2` | Dimensionless quadrupolar Love number when the tidal calculation is valid |
| `tidal_status` | Saved tidal capability status |
| `tidal_failure_reason` | Reason a tidal value is unavailable or failed closed |
| `amplitude` | Proposal amplitude; blank for `direct` |
| `delta_mev_fm3` | Activation-ramp width; blank for `direct` |
| `tov_rtol` | Saved relative ODE solver tolerance |
| `tov_atol` | Saved absolute ODE solver tolerance |
| `sequence_points_requested` | Number of central-pressure attempts requested for that stage |

CFL sequence rows also retain the surface-discontinuity evidence used by the
tidal calculation: `tidal_expected_jump_count`,
`tidal_applied_jump_count`, `tidal_surface_jump_count`,
`tidal_surface_delta_y`, `tidal_surface_y_before`,
`tidal_surface_y_after`, `tidal_surface_event_pressure_mev_fm3`, and the
strict `tidal_jump_evidence_json` payload. A usable CFL tidal result requires
one expected, applied, and surface jump with internally consistent values.

Use only rows with the required background and tidal status. The sampled
stable prefix ends at the sampled-peak row used by the workflow, but that row
must not be reported as a resolved maximum mass. Maximum-mass resolution is
recorded separately in `maximum_mass_screening.csv`. No attempted or
successful central pressure may exceed the case-specific retained EoS
endpoint; an endpoint below the sequence floor produces explicit unavailable
evidence instead of an extrapolated model.

Strict calculations retain several stellar stages for convergence evidence.
The final configured stage is the reporting and plotting reference; earlier
stages are not independent physical EoSs or extra training examples. The
authoritative configuration and `plot_inventory.csv` identify the final
population stage for a particular packet.

## `fixed_mass_observables.csv`

One row represents one requested gravitational mass for one EoS and one
numerical stage. The row is saved whether the target was solved or explicitly
unavailable. This makes `status` part of the logical data, not an optional
annotation.

| Column | Meaning |
|---|---|
| `case_id` | Baseline or deformation grouping key |
| `stage` | Governed numerical stage name |
| `amplitude` | Proposal amplitude; blank for `direct` |
| `delta_mev_fm3` | Activation-ramp width; blank for `direct` |
| `status` | `bracketed_and_solved` or an explicit unavailable status |
| `target_mass_msun` | Requested gravitational mass in solar masses |
| `reason` | Reason an unavailable row could not be solved; present when applicable |
| `mass_msun` | Gravitational mass of the solved model |
| `mass_residual_msun` | `mass_msun - target_mass_msun` |
| `radius_km` | Radius of the solved model in kilometres |
| `central_pressure_mev_fm3` | Solved central pressure in MeV fm^-3 |
| `central_energy_density_mev_fm3` | Solved central total energy density in MeV fm^-3 |
| `central_sound_speed_squared` | Dimensionless central `c_s^2` |
| `k2` | Dimensionless quadrupolar Love number when valid |
| `lambda_dimensionless` | Dimensionless tidal deformability when valid |
| `tidal_status` | Saved tidal capability status |
| `tidal_failure_reason` | Reason a tidal observable is unavailable |
| `bracket_pressure_mev_fm3` | Two saved central-pressure bracket endpoints, serialized as a CSV field |
| `root_xtol_mev_fm3` | Absolute central-pressure root tolerance in MeV fm^-3 |
| `root_evaluation_count` | Number of governed root/final evaluations recorded for the solution |

Some result columns can be absent from a run-specific CSV header when no row
could populate them. In a general-notebook `STUDENT_VIEW`, the generated
`DATA_DICTIONARY.md` records the columns actually present; otherwise inspect
the packet or focused-export CSV header itself. Never replace an unavailable
observable with zero. For tidal analysis, require both a solved background
status and the saved valid tidal capability status. An unavailable status can
specifically record that the stable evidence, bracket, or root would lie
outside the retained EoS domain; other fixed-mass rows solved inside the
domain remain valid.

Solved CFL fixed-mass tidal rows carry the same eight surface-jump evidence
columns listed for `stellar_sequences.csv`. A valid bare-surface result has
expected, applied, and surface jump counts of one, a negative
`tidal_surface_delta_y`, event pressure zero, and
`tidal_surface_y_after = tidal_surface_y_before + tidal_surface_delta_y`.

## `maximum_mass_screening.csv`

One row represents one maximum-mass assessment for one EoS and numerical
stage. An unresolved row intentionally does not claim the largest sampled
mass as the maximum.

| Column | Meaning |
|---|---|
| `case_id` | Baseline or deformation grouping key |
| `stage` | Governed numerical stage name |
| `status` | Exact turning-point/refinement outcome |
| `maximum_mass_resolved` | Boolean declaring whether a maximum was validly bracketed and refined |
| `maximum_mass_availability_status` | Explicit resolved or unavailable status, independent of fixed-mass availability |
| `maximum_mass_msun` | Refined maximum gravitational mass in solar masses; blank when unresolved |
| `maximum_mass_threshold_msun` | Governed screening threshold in solar masses |
| `passes_maximum_mass_threshold` | Saved threshold result when assessable |
| `central_pressure_mev_fm3` | Refined central pressure at the maximum, when resolved |
| `central_energy_density_mev_fm3` | Refined central total energy density at the maximum, when resolved |
| `central_sound_speed_squared` | Dimensionless central `c_s^2` at the maximum |
| `radius_km` | Radius at the resolved maximum in kilometres |
| `turning_point_count` | Number of admissible sampled turning-point brackets found |
| `positive_left_secant` | Saved positive-slope evidence on the left of the selected bracket |
| `negative_right_secant` | Saved negative-slope evidence on the right of the selected bracket |
| `eos_endpoint_pressure_mev_fm3` | Upper valid EoS pressure relevant to endpoint assessment |
| `endpoint_limitation` | Saved indication that the EoS domain or sequence endpoint limited resolution |
| `refinement_status` | Detailed local turning-point refinement status |
| `sampled_sequence_model_count` | Number of successful sampled models supporting the assessment |
| `local_background_solver_call_count` | Additional background-only calls used by local refinement |
| `tidal_solver_calls_for_maximum_mass` | Tidal calls used for maximum-mass refinement; the governed procedure records zero |
| `amplitude` | Proposal amplitude; blank for `direct` |
| `delta_mev_fm3` | Activation-ramp width; blank for `direct` |

Use `maximum_mass_msun` only when `maximum_mass_resolved` is true and the
status is consistent with a bracketed, refined turning point. When unresolved,
the threshold result is unavailable rather than false. Use
`stellar_sequences.csv` to inspect the sampled curve, not to invent a maximum
for an unresolved row. Endpoint-limited maximum-mass availability does not
erase solved rows in `fixed_mass_observables.csv`.

## Common analysis tasks

### Plot one EoS against the baseline

In Excel or LibreOffice:

1. Open `case_ledger.csv` and choose a row with `status = accepted` and
   nonzero `amplitude`.
2. Copy its exact `case_id`.
3. Open `thermodynamic_profiles.csv` and filter `case_id` to that ID.
4. Create an XY scatter plot with `epsilon_mev_fm3` on the horizontal axis and
   `pressure_mev_fm3` or `cs2` on the vertical axis.
5. Add a second series filtered to `case_id = direct`.

Use an XY plot, not spreadsheet row number as the horizontal coordinate.

In pandas, set the working directory to the validated `STUDENT_VIEW/`
directory (or prefix `data` with its actual location):

```python
from pathlib import Path

import pandas as pd

data = Path("03_PRIMARY_DATA/geometry_001")
ledger = pd.read_csv(data / "case_ledger.csv")
profiles = pd.read_csv(data / "thermodynamic_profiles.csv")

accepted_ids = ledger.loc[
    ledger["status"].eq("accepted") & ledger["amplitude"].ne(0.0),
    "case_id",
]
if accepted_ids.empty:
    raise ValueError("this geometry has no accepted nonzero deformation")
case_id = accepted_ids.iloc[0]

eos = (
    profiles.loc[profiles["case_id"].eq(case_id)]
    .sort_values("epsilon_mev_fm3")
)
direct = (
    profiles.loc[profiles["case_id"].eq("direct")]
    .sort_values("epsilon_mev_fm3")
)
```

### Compare fractional thermodynamic responses

For an accepted case, plot the saved `pressure_relative_to_direct`,
`baryon_density_relative_to_direct`, or `enthalpy_relative_to_direct` against
`epsilon_mev_fm3`. These are already dimensionless fractional responses. Do
not multiply them by 100 unless the plot is explicitly labelled percent.

### Make a mass-radius curve

Select one `stage` and `case_id`, retain successful stellar rows in
`attempted_index` order, and plot `Mass` against `Radius`. If the task requires
the stable prefix, stop at the sampled-peak marker and retain the separate
maximum-mass resolution status. Do not use tidal values whose `tidal_status`
is invalid or unavailable.

### Compare fixed-mass observables

Filter to one target mass and the final reporting stage. Keep rows with
`status = bracketed_and_solved`, then separately require valid tidal status
for `k2` or `lambda_dimensionless`. Radius and non-tidal background values can
be usable even when a tidal value is unavailable, provided the saved statuses
support that use. Do not discard these rows merely because the separate
maximum-mass assessment is unavailable.

### Compare maximum masses

Filter to the final reporting stage and rows with
`maximum_mass_resolved = true`. Preserve unresolved rows as labelled missing
outcomes when studying coverage; do not convert them to zero or replace them
with the largest sampled mass. Use the explicit availability status to
distinguish an endpoint-limited branch from another unresolved outcome.

## Preparing data for machine learning

The combined long-format tables are a better ML foundation than thousands of
separate files, but they are normalized: proposal metadata lives in the
ledger, while repeated sampled values live in downstream tables.

### Decide what one sample means

Different tasks have different sample identities:

- Pointwise EoS regression: one row may map energy density plus deformation
  coordinates to pressure or `c_s^2`.
- Curve-level modelling: one complete `(experiment, geometry, case_id)` group
  is one sequence sample.
- Stellar-sequence modelling: one `(experiment, geometry, case_id, stage)`
  group is one sequence sample.
- Acceptance classification: one ledger proposal is one labelled sample; raw
  gate diagnostics may be features, while reconstructed profiles must not be
  used because rejected cases never receive them.

### Join proposal metadata explicitly

For deformed thermodynamic rows, use a many-to-one join from profile rows to
the ledger:

```python
metadata_columns = [
    "case_id",
    "epsilon_match_mev_fm3",
    "epsilon0_mev_fm3",
    "sigma_mev_fm3",
    "status",
]

model_rows = (
    profiles.loc[
        profiles["amplitude"].notna() & profiles["amplitude"].ne(0.0)
    ]
    .merge(
        ledger[metadata_columns],
        on="case_id",
        how="left",
        validate="many_to_one",
    )
)
model_rows.insert(0, "geometry_name", "geometry_001")
```

The profile table already contains `amplitude` and `delta_mev_fm3`; the join
adds the remaining geometry and lifecycle fields. The explicit nonzero filter
also excludes the separately keyed physical zero-amplitude identity block.
Keep `direct` as the analytical baseline case type, and preserve logical
amplitude-zero ledger rows and their physical-ID mapping as separate identity
evidence.

### Prevent train/test leakage

Do not randomly split thermodynamic or stellar rows. Neighbouring rows from
one EoS are highly correlated; putting rows from the same curve in both train
and test sets measures interpolation within a known EoS rather than
generalization to a new one.

Create splits at the intended generalization level:

- hold out complete `case_id` groups to test new amplitudes at known geometry;
- hold out complete geometries to test new centers or widths;
- hold out complete experiments to test transfer across a different governed
  configuration;
- include experiment hash and geometry in the grouping key when data from
  several runs are combined.

Repeated `direct` baselines and identical `A = 0` controls are especially easy
to leak across partitions. Deduplicate or group them according to the
scientific question before assigning folds. Numerical stages are convergence
views of the same physical case, not independent examples.

### Respect the physical coordinate and missingness

The thermodynamic grid is nonuniform. Include `epsilon_mev_fm3` as a feature
for pointwise models. For fixed-length curve models, any interpolation or
resampling is new preprocessing outside the authoritative packet: record the
target grid, method, domain, masks, and extrapolation policy. Do not
extrapolate past a case's saved domain.

Keep unavailable values missing and retain status/reason columns or explicit
masks. Replacing missing radius, tidal, or maximum-mass values with zero gives
them a false physical meaning.

Finally, prevent target leakage. For example,
`pressure_relative_to_direct` contains the deformed pressure and should not be
used as an input feature when pressure is the prediction target. Select
features and targets from their definitions, not merely from numeric dtype.

## Optional diagnostic CSV files

For a general-notebook run, `STUDENT_VIEW/04_OPTIONAL_DIAGNOSTICS/` contains
the remaining copied CSVs for each geometry. In any route, the authoritative
versions remain in the geometry packet. The exact set depends on the
calculation, precision, accepted cases, and diagnostics setting.

| File | Main use |
|---|---|
| `case_plan.csv` | Planned case identities, identity-control injection, and requested stages |
| `raw_gate_profiles.csv` | Complete raw deformation evidence and gate status; for BSk24 this can include values beyond an accepted earlier crossing, while CFL uses the complete domain as its acceptance boundary |
| `thermodynamic_residuals.csv` | Reconstruction and derivative-consistency residuals for accepted cases |
| `window_characterization.csv` | One-row summaries of nominal and realized deformation geometry |
| `a0_identity_table.csv` | Saved zero-amplitude identity evidence |
| `stellar_status_summary.csv` | Background and tidal completeness counts and reasons |
| `plot_inventory.csv` | Which figures were produced or omitted and why |
| `radial_profiles.csv` | Saved radial stellar profiles when extended diagnostics apply |
| `deformation_support_fractions.csv` | Where the deformation support lies within applicable stars |
| Other extended stellar tables | Governed endpoint, response, pairing, and numerical-error diagnostics when requested |

Use raw-gate data when the question includes why a proposal was rejected. Do
not mix raw proposal values with reconstructed accepted profiles as if they
were the same lifecycle stage. For an accepted early-causal BSk24 case, raw
rows above the first `c_s^2 = 1` crossing are evidence outside the usable
branch; a later return below one does not extend reconstruction. Its raw table
retains direct pressure, integrated pressure change, resulting raw pressure,
and raw `c_s^2` at its saved coordinates. Read it together with the gate
report's continuous-resolution and first-crossing evidence; the CSV sampling
alone is not the continuous certificate.

CFL has no accepted early-causal prefix: a crossing rejects the complete
proposal. Its raw table instead carries case/physical/model identity, the
frozen baseline hash, geometry, energy density, window/Gaussian values,
`delta_cs2`, `raw_cs2`, and `gate_status`; it does not contain the BSk24 raw
pressure-array fields.

## Combining geometries or experiments

Before concatenating tables, add explicit provenance columns to the user's
analysis copy:

```text
experiment_hash
geometry_name
authoritative_packet_id_or_location
```

Then retain the original `case_id`, physical coordinate, `stage`, and status
columns. Do not depend on directory order, spreadsheet row numbers, or a
shortened case-ID prefix. Recheck schemas when combining runs: optional or
status-dependent columns may differ, and an absent column is not a column of
physical zeros.

## Common mistakes and interpretation boundaries

- The complete profile CSV is not globally pressure-sorted; ordering restarts
  when `case_id` changes.
- `delta_mev_fm3` is the activation-ramp width, while `delta_cs2` is the
  pointwise sound-speed deformation.
- `direct` is the saved solver identity for the one physical `A = 0` baseline;
  geometry-specific zero-amplitude IDs remain logical lifecycle identities.
- Rejected proposals have no reconstructed or stellar rows by design.
- Accepted BSk24 cases can have different retained endpoints; compare only
  their common saved domains and never extrapolate a shorter case. Accepted
  CFL cases span the complete frozen domain.
- Blank and unavailable values are not zero.
- A sampled stellar peak is not automatically a resolved maximum mass.
- Maximum-mass unavailability does not invalidate independently solved
  requested fixed masses.
- Background success does not automatically make a tidal result valid.
- Fixed masses are gravitational masses in solar masses.
- Energy density includes rest-mass energy.
- The reconstructed state is an effective one-fluid cold barotrope, not a
  microscopic composition calculation.
- CSV convenience does not replace the authoritative packet, validation
  report, exact manifest, configuration hash, or source provenance.

For the physical construction and acceptance rules, read
[`method.md`](method.md). For packet validation, figures, and lifecycle
interpretation, read [`results.md`](results.md).

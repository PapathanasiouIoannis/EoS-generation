# Results

Every authorized execution writes one self-contained result packet below the
selected `runs/` destination. The destination must be new; a result is not
silently overwritten.

## Start with validation and status

```powershell
bsk24-trial validate runs/my-experiment
bsk24-trial status runs/my-experiment
```

Validation is read-only. It checks the packet schema, strict JSON, exact file
manifest, configuration and source identities, and hard scientific validity
before tables are treated as trusted saved results. Availability is reported
separately: a structurally sound, hard-valid packet can be scientifically
partial when a requested observable is explicitly unavailable. Such a packet
remains loadable; the status fields and reasons determine which observables
may be used.

In a child validation report, `scientific_output_validity` is the hard gate
and `scientific_output_availability` is `complete` or `partial`. The aggregate
experiment report summarizes the latter as `scientific_availability_status`.
An availability limitation is reported as a warning/limitation, not rewritten
as a hard-validity failure.

`status` gives a compact health summary: matter model, calculation and
precision, geometry count, identity and convergence status,
accepted/rejected counts, and bounded rejection-reason counts. Detailed
observables remain in the saved tables and plots. Read the summary before
opening them.

## Packet layers

The exact files depend on `calculation` and `diagnostics`, but a packet records
these layers:

| Layer | Contents |
|---|---|
| Definition | Canonical user settings, matter model, expanded numerical profile, deterministic hash, plan, physical case IDs, and any logical aliases |
| Lifecycle | Run state and one accepted/rejected outcome with an exact reason for every executed physical case; nonexecuting logical aliases retain their mapping |
| Thermodynamics | Complete raw gate profiles, continuous-resolution evidence, case-specific retained endpoints, reconstructed profiles, residuals, identity checks, and convergence status |
| Stellar | Domain-bounded sequences, fixed-mass observables, tidal capability status, and independent turning-point availability when requested |
| Diagnostics | Applicable radial profiles and deformation-support tables when enabled |
| Figures | Plots rendered from saved tables and their inventory |
| Provenance | Environment versions, source hashes, two-step reproduction commands, and exact SHA-256 manifest; machine-specific executable and environment paths are deliberately omitted |

The packet retains the fully expanded profile behind `quick` or `strict`.
The short public configuration therefore remains auditable.

A CFL packet additionally retains the full frozen baseline record, parameter
SHA-256, formulation and domain IDs, formula-derived surface and endpoint
values, and the surface-anchored deformation profile version. Absence of
`matter_model` remains the canonical legacy BSk24 representation; validators
must not rewrite an old packet merely to insert the default.

Reproduction remains an explicit two-step action: first run the saved passive
plan command, then copy its hash into the saved `run --plan-hash ...
--execute` command. The hash binds authorization to the reviewed settings and
destination; it is not an optional label.

## Accepted and rejected cases

An accepted case passed the hard checks that were actually requested and
reached every applicable hard-valid stage. It is not a promise that every
optional observable is available, and it is not a claim of microscopic
composition or observational preference. In particular, a case can retain
valid requested fixed-mass observables while its maximum mass is unavailable
because the causal endpoint is reached before a turning point is established.

A rejected or unresolved case is a valid recorded outcome. It includes the
failed gate or unresolved certificate and available location evidence. The
complete raw proposal is preserved, but the case receives no reconstructed or
stellar values; missing entries must remain explicitly unavailable rather
than being filled with zero, interpolated across a failure, or repaired for
presentation.

An accepted proposal can have a shorter or longer domain than direct BSk24.
Each nonzero proposal is assessed within the published analytical-fit domain,
and its retained endpoint is the first continuously resolved combined
`c_s^2 = 1` crossing, included in the profile. Raw samples after that crossing remain evidence only; a later
return below one does not reopen the usable branch. All reconstructed and
stellar values must lie at or below the saved case-specific endpoint.

For a BSk24 or CFL Cartesian sweep, distinguish logical cases from executed
physical cases. Each geometry has a logical `A = 0` control, but only the
deterministic owner geometry executes the undeformed baseline. The other zero
rows must point to that physical case as nonexecuting aliases. Counts and
estimates are incorrect if aliases are treated as repeated scientific work
or as missing outcomes.

## Reading thermodynamic plots

The deformation is local in `c_s^2`, but pressure is an integral of
`dP/dε`. A pressure difference can therefore persist above the main
deformation region. Overlapping curves at small amplitudes are not by
themselves a failure; use the saved differences and residual tables to judge
scale.

Plots showing deformation support locate where a chosen fraction of the
windowed Gaussian is present in the star. Radial fraction and enclosed-mass
fraction answer different questions: one measures geometric position and the
other measures how much stellar mass lies inside that position.

## Reading stellar results

Only successful stable-prefix rows may be used for fixed-mass interpolation.
A requested mass without a true bracket, or whose required central pressure
would exceed the retained EoS endpoint, is unavailable. Other requested masses
that were validly solved inside the endpoint remain usable.

The maximum-mass status must distinguish:

- a bracketed and refined turning point;
- a sequence truncated by its valid domain;
- solver failure before a turning point;
- a merely highest sampled mass.

Do not reinterpret the last category as a resolved maximum. Tidal quantities
must likewise carry an explicit valid capability status; a background TOV
solution alone does not guarantee a valid tidal result.

Maximum-mass resolution is an availability result, not a substitute for EoS
validity. When the retained endpoint prevents a turning-point bracket, the
maximum-mass value and threshold comparison remain unavailable rather than
false, while independently solved fixed-mass rows are preserved.

For a bare CFL star, the zero-pressure surface has finite inner energy
density and vacuum outside. A successful CFL tidal row must show exactly one
negative outward `y` jump and retain the jump count, surface pressure, inner
and outer energy densities, `y_before`, `delta_y`, and `y_after`. The corrected
vacuum-side `y` is the value used for `k2`. Missing, repeated, wrong-sign, or
algebraically inconsistent evidence invalidates the tidal capability. CFL
extended radial diagnostics are unsupported in 1.2; their absence is expected
and must carry the declared capability status rather than be inferred from a
BSk24 diagnostic table.

## Saved plots

The CFL notebook adds a separate `plots/` sibling beside
the sealed experiment. It contains sweep-wide M–R, Λ–M, and k₂–M PNGs,
thermodynamic PNGs, a case catalogue, final-stage fixed-mass and maximum-mass
tables, an availability inventory, reporting-source hashes, and its own
manifest. No scientific packet bytes are changed. A missing tidal capability
produces an explicit unavailable/partial status, not an invented curve.

Local labels `C000000`, `C000001`, etc. are presentation aids only. Retain the
canonical physical IDs when comparing experiments. The A=0 physical case is
drawn once; rejected cases remain in the catalogue but not accepted-EoS
plots. All numerical stages and logical aliases remain in the source packet.
Full sampled curves may include unstable configurations; only governed
stable-prefix brackets support fixed-mass results. A maximum-mass threshold
comparison is distinct from physical EoS validity.

The focused `cfl_dataset.ipynb` route deliberately does not build this
seven-figure CFL experiment view or any scientific packet PNGs. Its child
plot inventories mark all technical figures skipped while retaining mandatory
population/completeness evidence. After all calculations validate, it creates
one flat `CFL_DATASET/` directory containing two labelled CSVs and the five
requested figures: pressure, sound speed, M–R, k2–M, and Lambda–M. There is no
student view, global catalogue, copied-table set, reporting manifest, or extra
inventory. The adapter consumes the already validated tables once and performs
zero scientific solver calls.

```powershell
bsk24-trial plot runs/my-experiment
```

The passive form lists or inspects plots already derived from saved tables.
Plotting must not call the thermodynamic or stellar solver. If a requested
plot family is absent, first check whether its required calculation and
capability exist in the packet.

To regenerate applicable figures from the already saved, manifest-verified
tables, authorize that mutation explicitly:

```powershell
bsk24-trial plot runs/my-experiment --overwrite
```

This can replace saved figure files and reseal their manifests, but it still
does not rerun thermodynamics or stellar structure.

Local packets are intentionally ignored by Git. Preserve important runs in
your own archival or publication workflow together with their manifest; do
not add large generated results to the source repository.

## Student view from the notebook

After a notebook execution completes and the authoritative experiment passes
validation, the notebook creates a separate `STUDENT_VIEW/` beside the sealed
experiment directory. It contains copied PNGs, primary CSV data, optional CSV
diagnostics, a data dictionary, and its own exact checksum manifest. The
notebook prints clickable locations for the view and the authoritative
packet. Because the notebook file lives below `notebooks/`, these links use
`../runs/...` paths so they resolve correctly in both VS Code and Jupyter.

The student view is derived and non-authoritative. It never changes the
experiment packet, is not included in the packet manifest, and is created
from saved artifacts without rerunning thermodynamics, stellar structure,
tidal work, or plotting. An existing student-view destination is rejected
rather than overwritten.

Its student-facing structure is:

```text
STUDENT_VIEW/
├── 01_READ_ME_FIRST.md
├── 03_PRIMARY_DATA/
│   └── geometry_NNN/
├── 04_OPTIONAL_DIAGNOSTICS/
│   └── geometry_NNN/
├── DATA_DICTIONARY.md
└── SHA256SUMS.txt
```

The CSV files are byte-for-byte copies of saved artifacts. Packet-level PNGs
remain in the sealed technical packets; the notebook-facing combined figures
are published only in the sibling `plots/` folder described below. The two
Markdown guides and `SHA256SUMS.txt` describe and checksum the student view;
they do not become part of the authoritative experiment. For a stellar case,
student-facing eligibility requires all explicitly requested fixed masses to
have succeeded at the reporting stage; it does not require a resolved maximum
mass. Fixed-mass and maximum-mass availability remain separate explicit
statuses in the copied data.

### Automatic combined accepted plot folder

After every successful notebook execution, the notebook creates one flat
`plots/` directory beside `STUDENT_VIEW/` and the authoritative experiment:

```text
plots/
├── window_profiles.png
├── gaussian_realization.png
├── raw_cs2_full_domain.png
├── raw_cs2_anchor_core_zoom.png
├── delta_cs2.png
├── pressure_response.png
├── baryon_density_response.png
├── effective_baryon_enthalpy_response.png
├── gamma_eff_response.png
├── thermodynamic_residuals.png
├── stellar_mr_k2_lambda.png                 # stellar runs only
├── observable_response_vs_amplitude.png     # when applicable
├── observable_response_vs_delta.png         # when applicable
├── a0_identity.png
├── accepted_case_index.csv
├── included_packets.csv
├── plot_inventory.csv
├── plot_generation_provenance.json
├── README.md
└── SHA256SUMS.txt
```

Applicable optional diagnostic figures are added to this same flat directory;
no per-case or per-geometry plot subdirectories are created there. Every PNG
combines accepted cases from the current completed experiment only. Historical
experiments are not mixed automatically. Rejected cases are excluded, while
their count remains explicit in provenance. Deterministically identical
direct/A=0 curves are drawn once.

The builder verifies every consumed table against its packet's sealed
`SHA256SUMS.txt`. Stellar plots use only final-stage saved rows and preserve
failed attempted-index gaps. Plotting makes zero thermodynamic, TOV, tidal, or
maximum-mass solver calls and never changes authoritative packets. Existing
destinations are rejected rather than overwritten.

Publication of the completed view remains an atomic same-volume directory
rename with no overwrite. On Windows, bounded retries handle transient sharing
violations from an editor or scanner; every retry rechecks that the destination
has not appeared, and a persistent failure removes the unpublished staging
directory without changing the sealed packet.

For the exact geometry and case ordering, complete primary-column meanings,
analysis examples, and leakage-safe machine-learning preparation, read the
dedicated [`csv-data-guide.md`](csv-data-guide.md).

## From an experiment to one CSV row

### Persistent friendly EoS labels

This shared-registry section applies to the standard notebook views and the
BSk24 dataset route. The focused CFL dataset route instead uses run-local
`cfl_0`, `cfl_1`, ... labels in its two `CFL_DATASET/` tables and creates no
catalogue registry or alias CSV.

After validation, the notebook publishes a new flat `EOS_DATA/` directory
beside `STUDENT_VIEW/`, `plots/`, and the authoritative experiment. It contains:

- `eos_catalogue.csv`: one row per accepted physical model in this experiment,
  including the baseline; source/status columns describe its first listed
  evaluation, not an aggregate over every evaluation;
- `case_aliases.csv`: every source case/geometry, its original ID, friendly
  `eos_id`, coordinates, precision, provenance and availability statuses;
- labelled copies of the five applicable primary CSVs, preserving every source
  value, row order, numerical stage, failure and missing entry;
- `README.md`, `provenance.json`, and an exact `SHA256SUMS.txt` manifest.

`H000000` is the first registered BSk24 baseline and `C000000` is the first
registered CFL baseline. Subsequent model-specific labels are allocated once
from the shared `runs/eos_catalogue/` registry. Positive/negative batches
do not restart numbering. A validated A=0 identity control shares the baseline
label regardless of geometry. Rejected proposals have blank `eos_id`; an
accepted EoS with unavailable maximum mass still receives a label.

The model key uses exact saved deformation coordinates and the saved
model-specific EoS/config source signature, not approximate similarity of curves.
QUICK/STRICT, fixed-mass requests and stellar-solver/reporting changes do not
change that physical key. Changes to EoS/config source conservatively create a
new key, even for a new baseline version; this is not an automatic scientific
equivalence judgement. Keep `precision`, experiment/packet identity, `case_id`
and `stage` when comparing evaluations. The `matter_model` column and `H`/`C`
prefix distinguish BSk24 and CFL; neither prefix is a microscopic-composition
certificate or observational selection.

Registry registrations are checksum-chained, append-only files, serialized by
an OS lock. IDs reserved before a presentation-publication failure stay reserved
and are reused on retry. Existing packets, presentations and registrations are
never overwritten. The catalogue has a unique `catalogue_id`: archive the
entire `runs/eos_catalogue/` directory with your data and never reset, edit or
renumber it. Independent checkouts/catalogues must not be joined on H/C labels
alone. Preview and read-only validation do not register or write labels.

The `plots/accepted_case_index.csv` and `plots/case_aliases.csv` carry the same
labels. Small EoS families use short legend labels; dense plots retain colour
bars rather than thousands of overlapping labels. These additions do not
change the current-experiment-only plot scope.

### Authoritative table hierarchy

The result hierarchy is:

```text
experiment -> geometry_NNN -> case_id -> sampled row
```

The experiment is one canonical settings hash. A `geometry_NNN` child is one
deterministically expanded combination of matching anchor, Gaussian center,
Gaussian width, and activation-ramp width. Inside it, a `case_id` identifies
one baseline or one deformation proposal. The rows for that case are samples
of that EoS or stellar sequence; each row is not a new deformation.

`case_id = direct` is the undeformed analytical BSk24 baseline saved for
comparison. The amplitude-zero case has its own deterministic case ID because
it passes through the same proposal and reconstruction route as the nonzero
cases. It is the governed identity control and is expected to reproduce the
direct baseline under the saved identity policy. Every accepted nonzero
amplitude case is a distinct deformed EoS.

A readable case ID can resemble `dp20_am0p2_<digest>`: the readable prefixes
encode ramp width and amplitude, while the final hexadecimal suffix is a
collision-resistant digest of the complete deformation coordinates. Do not
decode a case ID as a substitute for the ledger. `case_ledger.csv` is the
saved mapping from `case_id` to amplitude, geometry, anchor, and lifecycle
status. It also records the case-specific retained energy-density and pressure
endpoints and their reason, final-stage requested-fixed-mass status,
maximum-mass availability, and student-view eligibility. These statuses are
separate: all requested fixed masses can succeed, and the case can remain
student-view eligible, while maximum mass is unavailable.

Case IDs are useful grouping keys inside an experiment, but they are not
complete provenance identities. When combining separate experiments, retain
the canonical configuration hash and authoritative packet location as well.

## What one row means in each primary table

| File | Row meaning | Main grouping or coordinate |
|---|---|---|
| `case_ledger.csv` | One declared deformation proposal with lifecycle, retained-endpoint, and final availability statuses | `case_id` |
| `thermodynamic_profiles.csv` | One sampled total-energy-density point for the direct baseline or one reconstructed case | `case_id`, `epsilon_mev_fm3` |
| `stellar_sequences.csv` | One stellar-model attempt at a saved central coordinate | `case_id`, stage, central pressure/energy density, calculation status |
| `fixed_mass_observables.csv` | One requested fixed-mass outcome, either solved from a true stable-branch bracket or explicitly unavailable | `case_id`, stage, `target_mass_msun`, status |
| `maximum_mass_screening.csv` | One maximum-mass/turning-point assessment | `case_id`, stage, saved resolution status |

The physical row coordinate matters. Do not compare “row 100” between two
cases merely because the spreadsheet positions match. Filter or group by
`case_id`, then use the saved energy density, central pressure, target mass,
stage, and status columns appropriate to that table. Empty or unavailable
values are not zero.

A rejected proposal remains in `case_ledger.csv` with its exact reason but
has no reconstructed profile or stellar sequence. That downstream absence is
intentional.

## Which CSV files a student should copy

For EoS work, keep these two files from the same geometry directory together:

- `case_ledger.csv` supplies deformation coordinates and lifecycle status;
- `thermodynamic_profiles.csv` supplies the sampled EoS quantities.

For stellar work, keep that ledger together with whichever of
`stellar_sequences.csv`, `fixed_mass_observables.csv`, and
`maximum_mass_screening.csv` applies to the run. A thermodynamics-only run
correctly has no stellar tables.

If several `geometry_NNN` directories are combined, add the geometry folder
name as a column in the user's analysis table. Do not concatenate rows and
discard their original geometry or experiment identity.

### Excel or LibreOffice

Open `case_ledger.csv` first, choose an accepted `case_id`, then filter the
same column in `thermodynamic_profiles.csv`. For a conventional EoS curve,
make an XY plot with `epsilon_mev_fm3` on the horizontal axis and
`pressure_mev_fm3` or `cs2` on the vertical axis. Use `direct` as the baseline
series.

### pandas

Run this from the `STUDENT_VIEW` directory, changing the geometry directory
when required:

```python
from pathlib import Path

import pandas as pd

data = Path("03_PRIMARY_DATA/geometry_001")
ledger = pd.read_csv(data / "case_ledger.csv")
profiles = pd.read_csv(data / "thermodynamic_profiles.csv")

accepted = ledger.loc[ledger["status"] == "accepted", "case_id"]
case_id = accepted.iloc[0]
eos = (
    profiles.loc[profiles["case_id"] == case_id]
    .sort_values("epsilon_mev_fm3")
)
baseline = (
    profiles.loc[profiles["case_id"] == "direct"]
    .sort_values("epsilon_mev_fm3")
)
```

Plot `eos` and `baseline` as separate series against their saved
`epsilon_mev_fm3` coordinates. The same standard CSV files can be used in R,
Julia, MATLAB, Mathematica, or any other CSV-capable tool.

The frozen CFL equations and stability boundary are documented in
[`cfl.md`](cfl.md). A valid packet is evidence that the requested governed
software checks passed, not by itself an independent published CFL stellar
benchmark. Publication-level interpretation requires the strict convergence
and independent-solver work listed there.

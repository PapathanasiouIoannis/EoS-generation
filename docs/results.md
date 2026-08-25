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

`status` gives a compact health summary: calculation and precision, geometry
count, identity and convergence status, accepted/rejected counts, and bounded
rejection-reason counts. Detailed observables remain in the saved tables and
plots. Read the summary before opening them.

## Packet layers

The exact files depend on `calculation` and `diagnostics`, but a packet records
these layers:

| Layer | Contents |
|---|---|
| Definition | Canonical user settings, expanded numerical profile, deterministic hash, plan, and case IDs |
| Lifecycle | Run state and one accepted/rejected outcome with an exact reason for every case |
| Thermodynamics | Complete raw gate profiles, continuous-resolution evidence, case-specific retained endpoints, reconstructed profiles, residuals, identity checks, and convergence status |
| Stellar | Domain-bounded sequences, fixed-mass observables, tidal capability status, and independent turning-point availability when requested |
| Diagnostics | Applicable radial profiles and deformation-support tables when enabled |
| Figures | Plots rendered from saved tables and their inventory |
| Provenance | Environment versions, source hashes, two-step reproduction commands, and exact SHA-256 manifest; machine-specific executable and environment paths are deliberately omitted |

The packet retains the fully expanded profile behind `quick` or `strict`.
The short public configuration therefore remains auditable.

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

An accepted proposal can have a shorter domain than direct BSk24. Its retained
endpoint is the first continuously resolved `c_s^2 = 1` crossing, included in
the profile. Raw samples after that crossing remain evidence only; a later
return below one does not reopen the usable branch. All reconstructed and
stellar values must lie at or below the saved case-specific endpoint.

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

## Saved plots

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
├── 02_PLOTS/
│   └── geometry_NNN/
├── 03_PRIMARY_DATA/
│   └── geometry_NNN/
├── 04_OPTIONAL_DIAGNOSTICS/
│   └── geometry_NNN/
├── DATA_DICTIONARY.md
└── SHA256SUMS.txt
```

The PNG and CSV files are byte-for-byte copies of saved artifacts. The two
Markdown guides and `SHA256SUMS.txt` describe and checksum this derived view;
they do not become part of the authoritative experiment. For a stellar case,
student-facing eligibility requires all explicitly requested fixed masses to
have succeeded at the reporting stage; it does not require a resolved maximum
mass. Fixed-mass and maximum-mass availability remain separate explicit
statuses in the copied data.

Publication of the completed view remains an atomic same-volume directory
rename with no overwrite. On Windows, bounded retries handle transient sharing
violations from an editor or scanner; every retry rechecks that the destination
has not appeared, and a persistent failure removes the unpublished staging
directory without changing the sealed packet.

For the exact geometry and case ordering, complete primary-column meanings,
analysis examples, and leakage-safe machine-learning preparation, read the
dedicated [`csv-data-guide.md`](csv-data-guide.md).

## From an experiment to one CSV row

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

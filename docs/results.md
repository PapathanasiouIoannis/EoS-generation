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
manifest, configuration and source identities, and declared scientific
completeness before tables are treated as trusted saved results.

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
| Thermodynamics | Raw gate profiles, reconstructed profiles, residuals, identity checks, and convergence status |
| Stellar | Successful sequences, fixed-mass observables, tidal capability status, and turning-point resolution when requested |
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

An accepted case passed the checks that were actually requested and reached
the applicable downstream stages. It is not a claim of microscopic
composition or observational preference.

A rejected case is a valid recorded outcome. It should include the failed
gate and location. It receives no reconstructed or stellar values; missing
entries must remain explicitly unavailable rather than being filled with
zero, interpolated across a failure, or repaired for presentation.

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
A requested mass without a true bracket is unavailable.

The maximum-mass status must distinguish:

- a bracketed and refined turning point;
- a sequence truncated by its valid domain;
- solver failure before a turning point;
- a merely highest sampled mass.

Do not reinterpret the last category as a resolved maximum. Tidal quantities
must likewise carry an explicit valid capability status; a background TOV
solution alone does not guarantee a valid tidal result.

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
notebook prints clickable locations for the view and the authoritative packet.

The student view is derived and non-authoritative. It never changes the
experiment packet, is not included in the packet manifest, and is created
from saved artifacts without rerunning thermodynamics, stellar structure,
tidal work, or plotting. An existing student-view destination is rejected
rather than overwritten.

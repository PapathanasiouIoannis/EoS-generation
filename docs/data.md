# Data and migration

## Settings and numerical identity

The public JSON schema is `configs/schema.json`. Geometry axes expand as a
Cartesian product; zero amplitude is injected when absent. Every logical request
has its stable case ID. One physical zero control is shared across geometries;
the lexicographically first geometry owns its execution. Settings hashes and
BSk24 nonzero case IDs retain their v1 normalization. BSk25 settings always
record `matter_model="bsk25"` and use separate model-bound logical and physical
IDs; identical deformation numbers across the two models do not share runs.
Its source discrepancies, selected anchor and model identifier remain in the
saved baseline provenance. Existing BSk24 packets remain loadable. The v2 plan and run schemas are
explicitly different and do not reuse v1 aggregate hashes.

`observables`, when supplied, contains `sequence`, optionally `fixed_mass` and
`maximum_mass`. Stellar calculation requires sequence. Omission preserves v1
defaults: thermodynamics requests no stellar work, `dataset_40_curves` requests
sequence only, and other stellar profiles request all three products. Diagnostics
and figures are separate. The historical `dataset_40_curves` final thermodynamic
stage remains distinct from `dataset_40`; its saved EoS now includes the full
effective state.

`numerics.py` is the profile authority. Its data table preserves every historical
stage, grid, root tolerance and integration tolerance. Expanded profiles and
operational solver controls are bound into plans and saved with results.

## Flat run

| File | Meaning |
|---|---|
| `data/run.json` | Canonical settings/hash, expanded numerical profile, reviewed plan identity, units/anchor/domain, exact scientific certificates, source/environment identity, outcome and reproduction instructions |
| `data/cases.csv` | Logical case declarations, physical IDs, geometry, accepted/rejected/unresolved outcome and exact reason; aliases reference their shared physical evidence |
| `data/raw.csv` | Full assessed analytical proposals, including rejected/unresolved values and the superluminal continuation outside an accepted retained prefix |
| `data/eos.csv` | Accepted physical barotropes, including total energy density, pressure, sound speed squared, effective baryon density and chemical potential |
| `data/stars.csv` | Every declared sequence attempt, background failures/reasons, ordered segments and tidal capability status |
| `data/fixed_mass.csv` | Requested gravitational masses, true successful-stable-prefix pressure brackets, root results and tidal availability |
| `data/maximum_mass.csv` | Bracketed/refined maximum status, signed secants, endpoint limitation and refinement call evidence |
| `data/SHA256SUMS.txt` | Exact coverage of the data files, excluding the manifest itself |

Units: energy density includes rest mass and is in MeV fm^-3; pressure uses the
same units; sound speed squared is dimensionless (`c=1`); baryon density is in
fm^-3; effective chemical potential is in MeV; stellar mass is gravitational solar
mass and radius is in km. The chemical-potential column is historically named
`effective_baryon_enthalpy_mev` and means `(epsilon + P)/n_B`.

Thermodynamic certificates and convergence evidence are retained by numerical
stage. Full raw-gate reports have one authority in `run.json`. A source archive
contains the exact active package and available runtime/package contracts, even
for uncommitted edits. Archives are content-addressed and shared under
`runs/_sources/`; moving an entire study including `_sources` preserves replay
availability. Missing archives are reported separately from saved-data validity.
Keep archives when moving results. Restore the archive and its declared runtime
before an exact replay; an environment hash alone cannot install that runtime.

CLI `plan` and `run` accept `data/run.json` as their configuration input. For
portable reproduction, use a fresh destination and review a fresh hash:

```text
bsk24-trial plan --config old-study/data/run.json --output runs/reproduction
bsk24-trial run --config old-study/data/run.json --output runs/reproduction --plan-hash <fresh-reviewed-hash> --execute
```

Ordinary loading checks integrity and scientific evidence without requiring the
current source to match. Source equivalence is diagnostic, and `--exact-source`
adds that requirement. Validation performs no solver calls or writes. A failed or
interrupted run is never loaded as complete. Recover available failure evidence
by reading its saved tables and `run.json`; execute a new reviewed plan into a new
destination to retry.

## Optional diagnostics and figures

With diagnostics on, saved radial, baryonic and response tables retain their
scientific definitions and numerical envelope evidence. Diagnostic scope is
recorded explicitly; it does not alter raw acceptance. Figures read validated
saved data, use the final requested stellar stage and retain failure gaps. Valid
tidal curves require the saved capability status and finite valid values.
`figures/figures.json` records the consumed data manifest, plotter hash and image
hashes and the complete plot selection. Matching complete image hashes are reused
without writes. A changed selection or damaged/incomplete figure record creates
a new version; `--regenerate` explicitly requests another version. Data is never
resealed. The default selection is `auto`; readable plot labels and the historical
keys are accepted. `--geometry`, `--amplitudes`, `--fixed-masses` and `--stage`
select saved values. Baseline comparisons remain visible when amplitudes are
filtered. Each plot combines selected geometries, assigns colors from the full
saved geometry inventory (stable when filtering), and draws a shared physical
zero control once in black. Curves remain grouped by geometry, case and saved
mass/threshold/legacy-role qualifiers; ordered failure gaps prevent connecting
unrelated cases. Line styles distinguish amplitudes or response qualifiers; use
filters for detailed comparisons when many curves overlap. The figure record
and notebook color key identify each geometry's color.

The sequence Λ plot defaults to a logarithmic y axis over the full saved range,
so very large low-mass values do not flatten the higher-mass region. Set
`lambda_scale="linear"`, `lambda_mass_limits=(1.0, 2.2)` and/or
`lambda_limits=(0, 2000)` to choose a focused linear view. On a log axis the
lower Λ limit must be positive. CLI equivalents are `--lambda-scale`,
`--lambda-mass-limits` and `--lambda-limits`; notebook controls are documented
inline. Limits affect only the sequence Λ view, are noted on the figure, and
participate in figure caching. They never change or thin saved scientific rows.
Nonpositive Λ values cannot appear on a log axis and retain gaps rather than
being replaced. Signed tidal-response plots retain their linear axes.

Unresolved maximum masses, failed fixed-mass roots and
unvalidated tidal values are omitted and reported by the availability API.

Notebook previews choose unique paths without creating files. The public Python
and CLI default destination remains deterministic; specify a fresh destination
for another CLI/Python run. Re-executing a completed notebook plan loads the
intact matching result. A failed/incomplete destination instead requires another
preview, preserving its evidence. Notebook load mode is independent of the
current configuration and requires no in-memory result from an earlier session.

The notebook has one Markdown introduction and one executable cell. Its documented
top section chooses the active settings source, products, plots and saved-table
previews. The dictionary is active by default; JSON mode is an explicit alternative.
The output includes expandable profile/plot/table guides, enabled products, saved
run parameters, validation/source status and actual geometry/stage choices. Bad
display options are rejected before scientific execution. Plot availability
distinguishes products not requested, diagnostics disabled, missing support,
failed roots, unresolved maxima and unavailable tidal capabilities.

Diagnostics retain the governed baseline and accepted signed endpoint cases per
geometry for radial/baryonic profiles. Paired responses require matching signed
amplitudes and successful fixed-mass results; numerical-error summaries require
multiple stellar stages. Setting diagnostics on does not guarantee those tables.
All retained diagnostic tables, including absolute baryonic observables, paired
responses and numerical-error summaries, are inspectable through the notebook
table selector and `ExperimentResult.table()`.

## Migration

V2 removes child result packets, duplicate summaries/ledgers, automatic plotting,
student-view copies, persistent friendly-ID registries and notebook helper
scripts. The eight public import identities and `bsk24-trial` remain. Private
module paths, `ExperimentPlan.child_plans`, child results and old figure-group
arguments are intentionally retired. `ExperimentResult.table()` reads the six
primary tables and seven known optional diagnostic tables; its saved thermodynamic/stellar/fixed-mass properties remain
available.

The explicit legacy importer verifies original documents and exact packet
manifests, copies scientific tables to a new flat destination and retains old
JSON evidence in one run document. It does not rewrite settings, regenerate
fixtures, infer missing certificates or retroactively claim convergence. Repeated
historical stellar controls retain a `legacy_execution_role`. Legacy source
equivalence normally differs, and archives not supplied by the old format remain
unavailable. Imported results retain the saved raw-gate capability and cold
identity checks; a missing reconstruction certificate is reported as unavailable.
Unsupported schemas or missing complete-domain capability fail closed.

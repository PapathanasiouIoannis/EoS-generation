# Results

Every authorized execution writes one new aggregate experiment below the
checkout-local `runs/` tree. Existing destinations are never silently
overwritten. Generated results are ignored by Git and do not belong in the
source repository.

## Validate before interpreting

Use the path printed by execution:

```powershell
bsk24-trial validate runs/experiment_0123456789ab
bsk24-trial status runs/experiment_0123456789ab
```

`validate` is read-only and is the diagnostic entry point. It checks aggregate
and child schemas, strict JSON, exact manifests, configuration/source
identity, hard scientific validity, and observable-availability evidence. Add
`--json` for the complete report.

`status` strictly loads a completed, currently valid experiment and prints a
compact summary. It is not a replacement for the failure detail in
`validate`.

Hard validity and availability are separate:

- `scientific_output_validity` is the child hard gate;
- `scientific_output_availability` is `complete` or `partial`; and
- the aggregate reports `scientific_availability_status`.

A hard-valid packet may be scientifically partial when an explicitly
requested observable is unavailable. That limitation remains visible without
being rewritten as an invalid EoS.

## Aggregate and child packets

The saved hierarchy is:

```text
experiment_<settings-hash-prefix>/
├── experiment.json
├── experiment_config.json
├── reviewed_plan.json
├── reproduction_plan.json
├── SHA256SUMS.txt
└── geometry_NNN/
    ├── metadata.json
    ├── complete_configuration.json
    ├── raw_gate_report.json
    ├── case_ledger.csv
    ├── thermodynamic_profiles.csv
    ├── ... calculation-dependent evidence ...
    └── SHA256SUMS.txt
```

Exact child files depend on calculation, precision, diagnostics, accepted
cases, and requested plot groups. The packet layers are:

| Layer | Saved evidence |
|---|---|
| Definition | Canonical settings, matter model, expanded numerical profile, plan/configuration hashes, physical case IDs, and logical aliases |
| Lifecycle | One accepted, rejected, or unresolved outcome with a reason for each executed physical proposal, plus nonexecuting alias mappings |
| Thermodynamics | Complete raw gate profiles, model-specific domain evidence, reconstructed accepted profiles, identity checks, and requested convergence/residual data |
| Stellar | Domain-bounded sequences, fixed-mass outcomes, tidal capability evidence, and independent maximum-mass availability when requested |
| Diagnostics | Applicable extended BSk24 radial/support tables when enabled |
| Figures | Packet plots rendered from saved tables and an explicit plot inventory |
| Provenance | Source and runtime identities, portable two-step reproduction commands, and exact SHA-256 manifests |

CFL packets additionally retain the frozen formulation/parameter IDs and
hashes, formula-derived surface/domain values, complete-domain gate evidence,
and finite-density surface-jump records.

## Cases, identities, and aliases

Inside each `geometry_NNN` packet:

```text
geometry -> case_id -> sampled row
```

`case_id = direct` identifies the saved analytical baseline in applicable
thermodynamic and stellar tables. A deterministic amplitude-zero proposal is
the identity control. Across a Cartesian experiment, one geometry owns the
physical zero-amplitude calculation and all other zero rows are nonexecuting
logical aliases to it. The baseline is therefore calculated and plotted once,
not once per geometry.

Nonzero case IDs encode readable fragments plus a digest of the complete
deformation coordinates. Use `case_ledger.csv` as the mapping authority; do
not infer complete scientific identity from the readable prefix. Across
experiments, retain the experiment/settings hash and geometry identity as
well as `case_id`.

## Accepted, rejected, and unresolved proposals

An accepted proposal passed the requested model-specific hard checks and
reached every applicable hard-valid stage. Acceptance is not an observational
preference, a microscopic composition claim, or a guarantee that every
optional observable is available.

A rejected or unresolved proposal remains a valid recorded outcome. Its raw
proposal and failure evidence are saved, but it receives no reconstructed
profile or stellar sequence. Missing downstream values must remain missing;
do not fill, smooth, clip, or extrapolate across the failure.

The causal policy is model-specific:

- For BSk24, an otherwise valid proposal may retain the certified prefix
  through its first continuous `c_s^2 = 1` crossing. The crossing is included,
  and raw values after it are evidence outside the usable branch. A later
  return below one does not reopen the EoS.
- For CFL, the entire formula-derived domain is authoritative. A mechanical or
  causal failure anywhere rejects the whole proposal; CFL is not truncated to
  a passing prefix.

## Stellar availability

Only successful stable-prefix evidence may support fixed-mass interpolation.
A target is unavailable when it lacks a true bracket, the solver evidence is
invalid, or its required central pressure would exceed the EoS domain. Other
targets solved inside that domain remain usable.

Maximum-mass assessment must distinguish:

- a bracketed and refined turning point;
- an EoS/domain endpoint reached before a turning point;
- solver failure before resolution; and
- the largest sampled mass.

Only the first is a resolved maximum mass. The largest sampled model must not
be substituted for an unresolved result. For BSk24, a shortened causal branch
can preserve valid fixed-mass results while maximum mass remains unavailable.

Tidal quantities require an explicit valid capability status in addition to a
successful background TOV model. For a bare CFL star, the solver must record
one negative outward `y` jump at the finite-density zero-pressure surface and
use the corrected vacuum-side value for `k2`. Missing, repeated, wrong-sign,
or inconsistent jump evidence fails the CFL tidal capability closed.

CFL extended radial diagnostics are unsupported in 1.2.0. Their absence is
expected and must not be interpreted using BSk24 diagnostic semantics.

## Primary CSV tables

| File | One row represents |
|---|---|
| `case_ledger.csv` | One logical deformation proposal with lifecycle, retained-domain, and final availability statuses |
| `thermodynamic_profiles.csv` | One sampled total-energy-density point for `direct` or an accepted reconstructed case |
| `stellar_sequences.csv` | One attempted stellar model for a case and numerical stage |
| `fixed_mass_observables.csv` | One requested target-mass outcome, solved or explicitly unavailable |
| `maximum_mass_screening.csv` | One turning-point/maximum-mass assessment for a case and stage |

A thermodynamics-only packet correctly has no stellar tables. The
`dataset_40_curves` profile intentionally omits fixed-mass and maximum-mass
tables because those calculations are not requested.

Never compare spreadsheet row numbers across cases. Group by case and use the
saved physical coordinate, stage, and status. Complete column definitions,
ordering, join keys, and examples are in the [CSV data guide](csv-data-guide.md).

## Plotting

Inspect the packet plot inventory without writing:

```powershell
bsk24-trial plot runs/experiment_0123456789ab
```

Explicitly regenerate applicable packet figures from validated saved tables:

```powershell
bsk24-trial plot runs/experiment_0123456789ab --overwrite
```

The latter may replace figure files and reseal manifests, but it does not call
thermodynamic, TOV, tidal, fixed-mass, or maximum-mass solvers.

The Python result API has a deliberately different shape. Read
`result.plot_inventory` for passive inspection. Calling `result.plot()` asks
the packet plotter to generate applicable figures and reseal manifests; it
does not act as an inventory getter. Existing figure replacement requires
`result.plot(overwrite=True)`. Neither API plotting path reruns scientific
solvers.

Plot interpretation remains status-driven. Curves can overlap at small
amplitude; pressure responses can persist above the local `c_s^2` deformation
because pressure is an integral. Stellar sequence plots can contain attempted
models beyond the sampled peak; a displayed curve is not by itself a
turning-point or radial-stability certificate.

## Notebook-derived views

Notebook views are derived, non-authoritative siblings of the sealed
experiment. Their inputs are validated saved tables, and their reporting steps
make zero scientific solver calls.

| Route | Derived presentation |
|---|---|
| General BSk24 notebook | `STUDENT_VIEW/`, persistent-label `EOS_DATA/`, and combined `plots/` |
| Focused BSk24 dataset notebook | Exactly five combined figures in `plots/`; no `STUDENT_VIEW/` or `EOS_DATA/` |
| General CFL notebook | `STUDENT_VIEW/` plus a separately manifested combined `plots/` view with run-local C labels |
| Focused CFL dataset notebook | `CFL_DATASET/` containing exactly `cfl_eos_data.csv`, `cfl_stellar_data.csv`, and five combined figures |

### `STUDENT_VIEW/`

Where created, `STUDENT_VIEW/` contains copied primary CSVs, optional
diagnostic CSVs, guidance, and its own exact manifest. It never changes the
authoritative experiment and rejects an existing destination. For stellar
cases, eligibility requires every explicitly requested fixed mass to succeed
at the final reporting stage; it does not require maximum-mass availability.

### Persistent BSk24 `EOS_DATA/`

The general BSk24 notebook can publish friendly H labels and labelled primary
table copies through an append-only shared registry under
`runs/eos_catalogue/`. `H000000` is the first registered BSk24 baseline.
Precision and reporting changes do not make an evaluation a new physical EoS,
but source-definition changes conservatively do.

Archive the registry with its data. Do not reset, edit, or renumber it, and do
not join independent checkouts on an H label alone. Friendly labels never
replace canonical case IDs or packet provenance. The focused BSk24 dataset
notebook does not build this view.

### CFL labels

The general CFL notebook's C labels are local presentation labels. The focused
CFL dataset route instead uses deterministic run-local `cfl_0`, `cfl_1`, ...
labels in its two export tables. Neither scheme replaces canonical experiment,
geometry, and case identities.

## Reproduction and archival

Each completed experiment saves portable two-step reproduction commands:

1. create and inspect a fresh passive plan from the saved configuration; and
2. copy that fresh plan's hash into the saved `run --plan-hash ... --execute`
   command.

The hash binds authorization to settings, source/runtime identity, and
destination. Do not edit a packet to make source hashes agree, reuse a stale
hash, or overwrite the original. Reproduction creates a new packet below
`runs/reproductions/`.

Newly generated child packets record the general notebook appropriate to their
matter model in the legacy `notebook` compatibility field:
`notebooks/bsk24_experiment.ipynb` or `notebooks/cfl_experiment.ipynb`. CFL
packets already sealed by release 1.2.0 may still name
`notebooks/bsk24_experiment.ipynb`; do not edit or reseal them. For every
packet, the saved matter model, configuration, and two hash-bound reproduction
commands are authoritative; use the focused dataset notebook when reproducing
that presentation route.

Archive an important result as a complete experiment tree with all child
packets and manifests. Derived views are useful but do not replace that
authoritative evidence.

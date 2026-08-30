# Dataset workflows

Dataset-family profiles are experimental single-stage stellar workflows. They
preserve the governed equations, model-specific raw gates, units, surface
conventions, and failure semantics, but they are not renamed `strict` runs and
do not provide a per-case stellar refinement envelope.

Do not infer convergence, machine-learning suitability, or publication
readiness from a profile name. Review the complete passive plan and qualify the
profile against matched evidence for the intended scientific domain.

## Focused BSk24 notebook

[`../notebooks/bsk24_dataset.ipynb`](../notebooks/bsk24_dataset.ipynb) is the
focused BSk24 curve route. Its active checked-in settings are a large negative-
amplitude campaign: 90 geometries and 11 declared nonzero amplitudes. With the
injected logical identity control, the passive plan contains 1,080 logical
cases and deduplicates the physical baseline. It is not a starter notebook.

The default `dataset_40_curves` profile requests:

- the final 4,097/8,193-node thermodynamic stage;
- one 40-pressure stellar stage at rtol `1e-10`, atol `1e-12`, with 1,201
  radial integration samples and tides at every attempted sequence point;
- complete raw-gate and reconstruction-admissibility evidence; and
- only the five curve families: P–ε, sound speed, M–R, k2–M, and Lambda–M.

It does not request repeated thermodynamic residual processing, fixed-mass
roots, maximum-mass refinement, or retained radial profiles. The
`FIXED_MASSES` field remains in the public settings but is not solved by this
profile.

After validation, the notebook builds a separate `plots/` directory with
exactly five figures from saved packet tables. It creates no `STUDENT_VIEW/`,
`EOS_DATA/`, persistent labels, or duplicate data tree. Reporting makes zero
scientific solver calls.

## Focused CFL notebook

[`../notebooks/cfl_dataset.ipynb`](../notebooks/cfl_dataset.ipynb) is the
parallel pure, bare, self-bound CFL route. Its checked-in settings contain
three amplitudes at one geometry and use `dataset_40`, the only dataset-family
profile enabled for CFL.

The profile retains the strict-family thermodynamic/raw-domain stages, one
40-pressure stellar stage at rtol `1e-10`, atol `1e-12`, fixed-mass roots,
adaptive maximum-mass refinement, all-node tides, and the exact-once
finite-density surface-jump evidence. Per-child scientific PNG groups are
disabled; tables and statuses remain authoritative.

After aggregate validation, the notebook creates one `CFL_DATASET/` directory
with exactly seven files:

```text
CFL_DATASET/
├── cfl_eos_data.csv
├── cfl_stellar_data.csv
├── pressure_energy_density.png
├── speed_of_sound.png
├── mass_radius.png
├── k2_mass.png
└── lambda_mass.png
```

`cfl_0` is the direct baseline. Accepted nonzero deformations receive
`cfl_1`, `cfl_2`, ... in deterministic geometry/case order. Both CSVs retain
the canonical experiment, geometry, and case identities. Rejected proposals
remain in the sealed packet and are not exported as accepted EoSs.

The reporting adapter reads required saved tables once and calls no solver. A
failed presentation must not trigger a scientific rerun; the notebook retains
the completed in-memory result so its reporting cell can be retried.

## Passive-to-explicit procedure

For either focused notebook:

1. Activate the `eos-generation` environment and restart the kernel after
   changing installed package source.
2. Leave `EXECUTE_REVIEWED_PLAN = False` and run all cells.
3. Review geometry/case counts, physical aliases, expanded profile, worker
   budget, solver targets, and the new destination.
4. Change only the execution flag to `True` and run all again in the same
   kernel.

The preview makes zero solver calls and writes nothing. Source, settings,
environment, worker, kernel, plan, or destination drift requires a new
disabled preview. Up to six case workers may be selected, bounded by case
count and available logical CPUs; nested pools remain disabled.

## Dataset-family profiles

All profiles below require `calculation = "stellar"` and
`diagnostics = "off"`.

| Profile | Models | Single stellar stage |
|---|---|---|
| `dataset` | BSk24 | 61 pressures, rtol `1e-10`, atol `1e-12` |
| `dataset_10_tighter` | BSk24 | 10 pressures, rtol `1e-11`, atol `1e-13`; sampling and tolerance change together |
| `dataset_20` | BSk24 | 20 pressures, rtol `1e-10`, atol `1e-12` |
| `dataset_40` | BSk24, CFL | 40 pressures, rtol `1e-10`, atol `1e-12` |
| `dataset_40_curves` | BSk24 | Final thermodynamic stage and 40-pressure curve-only output at rtol `1e-10`, atol `1e-12` |
| `dataset_relaxed` | BSk24 | 61 pressures, rtol `1e-8`, atol `1e-10` |
| `dataset_relaxed_80` | BSk24 | 80 pressures, rtol `1e-8`, atol `1e-10` |

Except for `dataset_40_curves`, these profiles retain fixed-mass roots and
adaptive maximum-mass assessment. Every profile requests tides at all sequence
nodes. The pressure grids for different point counts are not necessarily
nested. Tighter ODE tolerances do not compensate for sparse curve sampling,
and more points do not tighten the integration of an individual star.

## Saved-data campaign adapters

The scripts below consume completed, checksum-bound saved data only. They do
not run thermodynamic, TOV, tidal, fixed-mass, or maximum-mass solvers, and they
do not modify source packets. All sources and new destinations must remain
below the same checkout's ignored `runs/` directory.

### Combine BSk24 dataset runs

At least two validated source-run directories are required:

```powershell
python notebooks/build_combined_hadronic_dataset.py --repository-root . --source-run runs/RUN_A --source-run runs/RUN_B --destination runs/COMBINED_BSK24
```

The adapter writes a separate manifested combined dataset, a six-column
stellar ML table, mappings/provenance, and five figures. It preserves original
packets and rejects overwrite.

### Build a current cumulative snapshot

The cumulative adapter uses a completed current experiment and the matching
`quick` or `strict` evaluation scope:

```powershell
python notebooks/build_combined_all_data.py --repository-root . --current-experiment runs/CURRENT_EXPERIMENT --destination runs/COMBINED_SNAPSHOT --precision strict
```

It publishes a non-authoritative saved-data snapshot with cumulative M–R and
sound-speed plots and explicit source indexes.

### Select and materialize a balanced BSk24 subset

Selection is deliberately two-stage: first publish a dry-run selection report,
then materialize only a report whose validation gates passed.

```powershell
python notebooks/select_combined_hadronic_subset.py --repository-root . --parent runs/COMBINED_BSK24 --destination runs/SELECTION_DRYRUN --target-count 2000 --selection-policy balanced
python notebooks/materialize_balanced_hadronic_subset.py --repository-root . --parent runs/COMBINED_BSK24 --dryrun runs/SELECTION_DRYRUN --destination runs/BALANCED_DERIVATIVE
python notebooks/replot_balanced_hadronic_subset.py --repository-root . --source runs/BALANCED_DERIVATIVE --destination runs/BALANCED_REPLOT
```

Whole EoSs are selected; curve rows are not sampled independently. Parent,
dry-run, derivative, and replot destinations must be distinct and unoccupied.
Preserve all inputs for audit and recovery.

## Qualification boundary

Before using a dataset profile for a large study:

- compare matched geometries against archived `strict` evidence;
- quantify errors for every required thermodynamic and stellar observable,
  including between-node curve behavior;
- preserve rejected cases, failed-attempt gaps, and explicit unavailable
  statuses;
- verify maximum-mass availability rather than substituting sampled peaks;
- document the accepted accuracy budget and split ML data by whole physical
  EoSs/geometries to avoid leakage; and
- measure runtime on the intended hardware without promising a universal
  speedup.

The CFL `dataset_40` route has software/passivity coverage but no matched
campaign-level `strict` qualification. Its output must remain experimental
until that comparison is performed.

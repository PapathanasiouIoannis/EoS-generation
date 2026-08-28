# Developer guide

## Architecture

The package follows a narrow public shell around the scientific
implementation:

| Path | Responsibility |
|---|---|
| `src/eos_generation/__init__.py` | Stable public imports and version |
| `src/eos_generation/experiment.py` | Public settings, planning, execution, loading, and validation facade |
| `src/eos_generation/cli.py` | `bsk24-trial` command adapter |
| `src/eos_generation/notebook.py` | Passive-by-default notebook adapter |
| `src/eos_generation/bsk24/` | Analytical BSk24, smooth deformation, and effective reconstruction |
| `src/eos_generation/cfl/` | Frozen CFL baseline, surface-anchored deformation, reconstruction, planning contract, and primary-source manifest |
| `src/eos_generation/stellar/` | TOV, tidal, discontinuity, and stellar diagnostics logic |
| `src/eos_generation/reporting/` | Saved-table plotting and plot orchestration |
| `src/eos_generation/_internal/` | Configuration expansion, execution lifecycle, packet integrity, provenance, and validation details |

Users should not need private modules. Private modules must not import back
through `experiment.py` or the package facade in a way that creates a cycle.

## Public API

The supported objects are:

```python
Experiment
ExperimentSettings
ExperimentPlan
ExperimentResult
plan_experiment
run_experiment
load_experiment
validate_experiment
```

The command and notebook are adapters to this API, not independent scientific
implementations. Preserve object import identity and keep all planning paths
passive. `bsk24-trial` is retained as the compatibility command name and
dispatches from the settings model. Separate BSk24 and CFL notebooks share
the two-pass adapter. The CFL discriminator uses its own notebook-settings
schema; the legacy BSk24 settings document remains byte-for-byte equivalent.
Only CFL preserves tiny nonzero amplitudes exactly instead of using the
legacy notebook's near-zero normalization.

The CLI execution gate is deliberately two-part: `run` requires both the
hash from the exact reviewed passive plan and the explicit `--execute` flag.
Any settings, source, environment, or destination change requires a new plan.

The public configuration has the required `$schema` annotation, the nine
legacy scientific settings, and the optional `matter_model` discriminator in
`configs/schema.json`. Omission must continue to serialize exactly as legacy
BSk24; explicit CFL must carry `matter_model = "cfl"` and
`epsilon_match = "surface"`. Normalize scalar/list geometry deterministically,
expand `quick` or `strict` to its complete internal settings, and hash the
resolved scientific configuration canonically. Destination and execution
authorization are operational controls rather than hidden scientific
settings.

The existing stage dataclasses are intentionally reused as model-neutral
numerical-profile containers for CFL. Their serialized fields describe grids,
tolerances, and requested work rather than BSk24 physics. Do not rename or
wrap them merely for taxonomy: that would risk changing established BSk24
hashes without adding a scientific distinction.

## Scientific separation

Keep these layers explicit:

1. passive settings validation and work planning;
2. baseline and raw windowed proposal construction;
3. complete raw-evidence assessment, continuous-extremum search, and
   resolution certification;
4. first-causal-crossing selection and effective thermodynamic reconstruction
   on the accepted retained branch;
5. optional stellar and tidal work;
6. deterministic serialization and manifest sealing;
7. read-only loading, validation, status, and saved-table plotting.

A rejected raw proposal must not leak into downstream layers. Debug-only
transformations must retain the original arrays and cannot change acceptance.
Do not infer the causal endpoint from an ordinary output-grid sample: locate
and refine the first continuous `c_s^2 = 1` crossing, include it in the
retained table, and treat all later values as raw evidence outside the usable
branch. Geometry-aware discovery must cover every relevant extremum basin and
must fail closed when the analytical deformation cannot be resolved.

Do not split tightly coupled TOV/tidal equations merely for cosmetic file
size. Refactor only across boundaries that preserve units, interpolation and
inversion authorities, jump corrections, surface conditions, root brackets,
stable-branch logic, and error semantics.

For CFL, preserve a further model boundary: the full immutable baseline
profile is imported from one authority, not duplicated in planning or runtime.
Its formulation ID is `cfl_bag_full_ms_delta2_v1`, parameter hash is
`3991cb8615d2d29617ccb90c6dc54b23aae64bcc752856d07f17f99abc048307`,
and its authoritative formula-derived binary64 energy-density domain is
`[190.2181760065314, 4008.81724402691] MeV fm^-3`. Rounded design-review
values are documentation aids only and must never enter comparison, hashing,
or serialization.

The CFL raw gate covers that complete domain and reconstruction is anchored at
the undeformed finite-density surface. No below-surface EoS, crust, or
hadronic matching is permitted. For every public BSk24 or CFL Cartesian
sweep, exactly one lexicographically first geometry owns the physical
`A = 0` baseline; non-owner logical controls must be stable nonexecuting
aliases. The BSk24 physical ID includes its effective matching anchor.
Estimates and executors must count physical work, while public case tables
retain logical traceability. Directly constructed legacy
`BSk24TrialConfig` objects with no owner flag retain their established local
identity-control behavior and serialization.

The bare-CFL tidal surface jump has a negative outward sign and must be
applied exactly once before `k2`. Preserve its recorded count and before/after
evidence through execution, serialization, loading, and validation. CFL
extended radial diagnostics remain a fail-closed unsupported capability in
1.1.

## Numerical profiles

`quick` and `strict` are governed names, not informal presets. The `quick`
expansion selects the retained thermodynamic quickstart or relaxed stellar
profile according to `calculation`; `strict` selects the retained strict
profile. Their exact grids, tolerances, refinement stages, and diagnostics
settings live in one internal authority. They must be:

- expanded during passive planning;
- included in deterministic configuration identity;
- unchanged between reviewed plan and execution;
- serialized with every result;
- protected by regression tests.

Changing either profile is a scientific change. Do not add source-level or
environment-variable overrides that bypass the public settings contract.

The separately named experimental `dataset` profile is documented in
`docs/dataset.md`. Never change QUICK/STRICT to implement its optimization.
Protect the retained thermodynamic settings/tolerances and historical STRICT
configuration hash with regression tests. The focused notebook must remain
passive by default and its five-figure adapter must consume validated saved
data, preserve failure gaps, and perform zero solver calls.

The two BSk24 notebook settings cells and the dedicated CFL dataset notebook
select `dataset_40` by default. This is not a
redefinition of `strict`. The shared production case-worker cap and notebook
preview budget are six, bounded by case count and half the logical CPU count.
This shared executor policy also applies to CLI/API runs; do not introduce a
hidden notebook-only override. Nested pools remain disabled inside case
workers. The standalone sequence-worker fallback is unchanged. Bind the new
budget to a fresh preview, preserve deterministic merge order, and regression
test preview/production budget agreement. The six-worker, 40-point benchmark
preserved scientific values and statuses; operational timing/PID metadata
naturally differs. Historical source archives and packets must remain intact.

## Result integrity

Writes belong below a user-selected new path under `runs/`. Use atomic writes,
strict finite JSON, deterministic table order, exact manifest coverage, and
stable case IDs. Preserve accepted/rejected status and exact failure reason
for every declared case.

Loading and validation are passive. Keep hard packet/scientific validity
separate from observable availability so a well-formed endpoint-limited
packet remains loadable with explicit partial statuses. Finite auxiliary
thermodynamic diagnostics remain evidence rather than acceptance predicates;
non-finite reconstruction and broken matching, interpolation, or inversion
remain hard failures. Plotting reads saved tables only. Source provenance must
include every active calculation and reporting module; update the inventory
and its test when a source path changes.

The layered report names these scientific sections
`scientific_output_validity` and `scientific_output_availability`; only the
former participates in the packet pass/fail gate. Aggregate validation carries
the child availability result forward as `scientific_availability_status`.

The notebook may create a derived student presentation view only after the
authoritative aggregate and every child packet pass validation. That view must
remain outside the sealed experiment, copy saved CSV/PNG artifacts only, have
its own exact checksum manifest, reject overwrite, and never become part of
the canonical packet or public API. Publish it with an atomic same-volume
rename. Bounded Windows `PermissionError` retries may accommodate transient
share violations, but every attempt must recheck no-overwrite and any failed
stage must be cleaned up.

`reporting/notebook_results.py` is the CFL saved-table presentation adapter.
It validates the sealed experiment, overlays accepted physical cases from
all geometries, draws the A=0 control once, and writes an independently
manifested sibling `plots/` view. It never mutates the
authoritative experiment, infers tidal validity from finiteness, fills a gap,
or reruns a solver. Its catalogue labels are experiment-local, not a global
registry. Existing views are hash-checked and reused without writes; missing
views require explicit creation authority. Failed builds publish no view.

The large-run `cfl_dataset.ipynb` intentionally bypasses that seven-figure
adapter. It uses `requested_plot_groups=("none",)` in every CFL child, then
invokes the shared `eos_catalogue.py` and `build_dataset_plots.py` adapters
after aggregate validation. Those adapters publish complete labelled table
copies and exactly five combined figures with zero solver calls. The shared
registry assigns independent H/BSk24 and C/CFL counters in one append-only
checksum chain and must continue to read legacy BSk24-only transactions.

Routine notebook tests use synthetic saved tables and guarded passive kernels
from both repository-root and notebook working directories. Real quick/strict
acceptance runs require separately reviewed cost and authorization and do not
belong in the routine CI suite.

## Tests

Friendly IDs are a notebook presentation concern, implemented in
`notebooks/eos_catalogue.py`. They must never enter scientific settings hashes,
canonical case IDs, acceptance gates, or authoritative packets. The notebook
preview binds both presentation builders' source hashes; derived outputs also
record their builder hashes and consumed source manifests. Registry identity
excludes precision, stellar solver and reporting source, but conservatively
includes the saved model-specific EoS/config source signatures. Registration uses an OS lock,
append-only checksum-chained transactions and atomic no-clobber publication.
All labelled primary columns preserve their source values; only provenance
and friendly-ID columns are added. Keep student-view copies byte-identical.

Test this reporting path with synthetic sealed tables, never new stellar
calculations. Cover mixed signs, QUICK/STRICT reuse, A=0 geometry collapse,
physics-version separation, rejected/unresolved semantics, concurrent writers,
corrupt inputs/registrations, no overwrite, value preservation, and passive
notebook execution. A failed derived export must not trigger a solver rerun.

Install the declared environment and editable package, then run the focused
suite:

```powershell
conda env create -f environment.yml
conda activate eos-generation
python -m pip install -e ".[notebook]"
python -m pytest -q
```

During development, select the narrowest relevant test. The suite should
cover at least:

- public config normalization, hash stability, and passive planning;
- BSk24 analytical values and zero-amplitude identity;
- frozen CFL thermodynamics, formula-derived endpoints, full-profile hash,
  surface anchoring, and zero-amplitude identity;
- smooth-window geometry and raw-gate behavior;
- first continuous causal crossing, narrow-pocket/island detection, retained
  tabulation resolution, and below-anchor four-sigma support overlap;
- compact independent continuous-star TOV/tidal regression;
- retained-domain central-pressure bounds and fixed-mass/maximum-mass partial
  availability;
- uniform-density and CFL finite-surface jump sign/count regressions;
- result integrity and read-only validation;
- student-view eligibility and transient Windows publication recovery;
- notebook passivity, `../runs/...` result links, and delegation to the
  production API.

Do not regenerate a reference fixture with the implementation it checks. Do
not weaken a tolerance, gate, or expected status to obtain a pass.

Repository-hygiene and provenance tests must also work from a source archive
that has no `.git` directory. When Git metadata is present, first verify that
the discovered top level is this checkout before using tracked-file output;
otherwise apply the explicit archive/file-tree policy. CI should exercise the
Git-free archive path directly rather than skipping it.

## Installed-wheel check

Before publishing a release, test the built artifact rather than relying only
on editable imports:

```powershell
python -m pip install build
python -m build --wheel
python -m pip install --force-reinstall dist/eos_generation-1.1.0-py3-none-any.whl
```

From outside the checkout, import the public objects, run `bsk24-trial
--help`, and make separate passive BSk24 and CFL plans with absolute config
paths. The plans must leave their working directory empty. Confirm that the
wheel includes both packaged source manifests.

Passing the repository suite is not sufficient evidence for publication-level
CFL stellar claims. Release review must keep the distinction between analytic
unit tests and the still-required strict convergence study,
convention-matched published pure-CFL sequence, and independently implemented
stellar solver comparison. Never create the expected benchmark fixture with
the implementation under test.

## Change review

Report software behavior, numerical behavior, and physical interpretation as
separate concerns. Include exact commands, test results, units, tolerances,
and unresolved scientific decisions. Do not launch an expensive stellar
calculation for routine packaging, documentation, import, or passivity checks.

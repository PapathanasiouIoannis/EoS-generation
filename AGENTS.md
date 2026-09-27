# AGENTS.md

## Scope

This repository implements controlled smooth sound-speed deformations of
analytical BSk24, fail-closed thermodynamic assessment, effective one-fluid
reconstruction, and optional TOV/tidal calculations. Supported user surfaces
are settings under `configs/`, passive `bsk24-trial plan` or
`plan_experiment`, explicit `bsk24-trial run ... --execute` or
`run_experiment`, passive loading/validation/status, saved-table plotting, and
the BSk24 notebooks `notebooks/bsk24_experiment.ipynb` and
`notebooks/bsk24_dataset.ipynb`. The first supports the full `dataset_40`
campaign with fixed-mass and maximum-mass outputs; the second supports the
focused `dataset_40_curves` campaign and five saved-table figures.

Local packets belong under ignored `runs/`. Never commit results, caches, or
large generated data.

## Before changes

Read relevant `docs/`, trace the public entry point and private execution
path, inspect configuration expansion, provenance, focused tests, and a
representative result shape. Distinguish demonstrated facts, inference,
assumptions, and open scientific questions. For reviews, make no edits and run
no scientific calculations. For implementation, make the smallest coherent
change and preserve scientific behavior unless explicitly in scope.

## Public contract

`pyproject.toml` defines packaging and dependencies; `environment.yml` pins
the scientific runtime. The distribution, import, and command names are
`eos-generation`, `eos_generation`, and `bsk24-trial`. Top-level Python
exports are `Experiment`, `ExperimentSettings`, `ExperimentPlan`,
`ExperimentResult`, `plan_experiment`, `run_experiment`, `load_experiment`, and
`validate_experiment`. Preserve import identity, deterministic settings
hashes, case IDs, schemas, manifests, and reproduction commands. Private
modules must not import through the public facade and create a cycle.

Public JSON fields are limited to `configs/schema.json`. `quick` and `strict`
expand to governed numerical profiles; expanded settings participate in
planning, hashing, and saved results. Preserve the BSk24 dataset profiles and
large-run routes. Do not expose hidden numerical overrides.

## Execution safety and science

Planning must call no scientific solver and write no files. A run requires a
separate operation and explicit authorization. Notebooks remain passive when
`EXECUTE_REVIEWED_PLAN = False`. Do not run expensive stellar calculations for
routine verification; review planned cost and destination first. Validation
is read-only, plotting uses saved tables, and writes are atomic without silent
overwrite.

Energy density includes rest-mass energy and pressure is in MeV fm^-3;
`dP/dε = c_s^2` is dimensionless with `c = 1`; fixed masses are gravitational
solar masses. Preserve coefficients, equations, constants, conversions,
domains, anchors, solvers, grids, tolerances, and acceptance predicates without
direct scientific scope and independent support. On every assessed continuous
cold phase, require `ε > 0`, `P >= 0`, and `0 < dP/dε <= 1` on the accepted
domain. Where available, preserve `dε = μ_B dn_B`, `P = n_B μ_B - ε`, and
`μ_B = (ε + P) / n_B`.

Assess the raw proposal over its complete declared domain before
reconstruction. Rejected proposals receive no reconstruction or stellar
work; retain raw values and exact reasons. Never clip, clamp, smooth, or
repair a failed raw value into acceptance. Zero amplitude reproduces direct
BSk24 under the governed floating-point identity policy. Reconstruction is
an effective one-fluid barotrope, not a microscopic composition claim.

Distinguish fitting seams, composition thresholds, physical transitions,
self-bound surfaces, and unknown discontinuities. Require physical support
for a physical label, apply each required jump correction exactly once, and
fail closed when capability is unestablished. Fixed-mass observables need a
true bracket on the successful stable prefix. A sampled peak is not a
resolved maximum mass without a bracketed and refined turning point.

## Reproducibility and verification

Executed results retain canonical settings and hash, expanded profile, stable
case identities and statuses, exact failures and capability statuses,
source/environment hashes, calculation/reporting provenance, strict JSON,
exact manifest, and portable reproduction commands. Never regenerate a
reference fixture from the implementation under test, weaken a scientific
tolerance or predicate, or turn failure into a skip. Use independent published
values, source tables, solvers, convergence studies, and benchmarks for
scientific changes. Record commands, results, and justified tolerances.

Run narrow passive or regression checks while developing, then
`python -m pytest -q` before publication. `.github/workflows/ci.yml` governs
clean installed-wheel, passivity, notebook, regression, and hygiene checks.

## Git

Begin implementation from clean, current `main` on one focused branch.
Preserve unrelated changes, stage only intended paths, and inspect the full
diff before publication. Do not rewrite history, force-push, bypass CI, use
destructive Git commands, commit, or mutate a remote without the user's
explicit publication request.

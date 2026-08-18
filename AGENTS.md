# AGENTS.md

## Repository purpose

This repository implements one reproducible method: controlled smooth
sound-speed deformations of analytical BSk24, fail-closed thermodynamic
assessment, effective one-fluid reconstruction, and optional TOV/tidal
calculations.

The supported workflow is intentionally small:

- user settings under `configs/`;
- passive planning through `bsk24-trial plan` or `plan_experiment`;
- explicit execution through `bsk24-trial run ... --execute` or
  `run_experiment`;
- passive loading, validation, status, and saved-table plotting;
- the single notebook `notebooks/bsk24_experiment.ipynb`.

Local result packets belong below the ignored `runs/` directory. Do not add
generated results, caches, or large data files to Git.

## Familiarize yourself before changes

Before a significant change:

1. Read the relevant file under `docs/`.
2. Trace the actual public entry point and private execution path.
3. Inspect configuration expansion, focused tests, source provenance, and a
   representative result shape.
4. Separate demonstrated facts, reasonable inferences, assumptions, and open
   scientific questions.
5. Run the narrowest relevant passive or regression checks.

For an inspection or review task, do not edit files or run scientific
calculations. For an implementation task, make the smallest coherent change
and preserve scientific behavior unless the task explicitly changes it.

## Canonical user contract

`pyproject.toml` defines the package and dependencies. `environment.yml` is
the pinned scientific runtime. Do not duplicate dependency versions in prose
or change them to solve an unrelated problem.

The supported distribution, import, and command names are:

```text
eos-generation
eos_generation
bsk24-trial
```

The public Python objects exported by `eos_generation` are:

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

Preserve public import identity, deterministic settings hashes, case IDs,
schemas, result manifests, and reproduction commands during refactors.
Private modules must not import back through the public facade in a way that
creates a cycle.

The public JSON surface is limited to the fields in `configs/schema.json`.
`quick` and `strict` expand to governed internal numerical profiles; those
expanded settings must be included in planning, hashing, and saved results.
Do not expose hidden numerical overrides as an undocumented alternative.

## Execution safety

The safe first operation is:

```text
bsk24-trial plan --config configs/quickstart.json
```

Planning must perform zero scientific solver calls and zero filesystem
writes. Scientific execution requires a separate run operation and explicit
authorization. Notebook execution must remain passive when
`EXECUTE_REVIEWED_PLAN = False`.

Do not run expensive stellar calculations for import checks, documentation,
planning, notebook validation, packaging, or routine repository verification.
Run them only when the task explicitly authorizes the scientific calculation
and the planned cost and destination have been reviewed.

Result validation is read-only. Plotting must consume saved tables and remain
lazy; it must not rerun a scientific solver. Writes must be atomic and must
not overwrite an existing result without a separately declared policy.

## Scientific invariants

Scientific correctness takes precedence over code elegance, runtime, visual
smoothness, or passing tests.

For every quantity, preserve its physical meaning, units, normalization,
total-energy convention, and valid domain. In this package:

- energy density includes rest-mass energy and is expressed in MeV fm^-3;
- pressure is expressed in MeV fm^-3;
- `dP/dε = c_s^2` is dimensionless with `c = 1`;
- fixed masses are gravitational masses in solar masses.

Do not change BSk24 coefficients, equations, conversions, domains, anchors,
physical constants, solvers, grids, root brackets, tolerances, surface
conditions, or acceptance predicates without direct scientific scope and
independent support.

On every assessed continuous cold phase, preserve checks such as:

```text
epsilon > 0
P >= 0
0 < dP/depsilon <= 1
```

Where baryon density and chemical potential are available, preserve:

```text
d epsilon = mu_B d n_B
P = n_B mu_B - epsilon
mu_B = (epsilon + P) / n_B
```

Assess the raw windowed proposal on the complete declared domain before
reconstruction. Rejected proposals receive neither reconstruction nor stellar
work. Retain raw values and exact rejection reasons. Never clip, clamp,
smooth, regularize, or replace a failed raw result and then classify the
modified value as accepted.

The zero-amplitude case must reproduce baseline BSk24 under the governed
floating-point identity policy. The reconstructed state is an effective
one-fluid cold barotrope; it does not establish microscopic composition,
species chemical potentials, or beta equilibrium.

Distinguish numerical fitting seams, composition thresholds, physical
first-order transitions, self-bound surfaces, and unknown discontinuities.
A physical label requires physical support. Apply each required stellar or
tidal jump correction exactly once, and fail closed when the capability is
not established.

Fixed-mass observables require a true bracket on the successful stable prefix.
Do not report a sampled peak as a resolved maximum mass unless the governed
turning point has been bracketed and refined.

## Reproducibility and tests

Every executed result must retain:

- canonical user settings and their deterministic hash;
- the expanded internal numerical profile;
- stable case identity and accepted/rejected status;
- exact failure reasons and solver capability statuses;
- source and environment hashes;
- calculation and reporting provenance;
- strict JSON and an exact file manifest;
- a portable reproduction command.

Do not regenerate a reference fixture from the implementation under test.
Never weaken a scientific tolerance or predicate, convert a failure to a
skip, hide a failure by repair, or use finiteness alone as proof of physical
correctness.

Prefer independently published values, authoritative source tables,
independently implemented solvers, convergence studies, and analytical or
semi-analytical benchmarks. Record exact commands, results, and justified
tolerances for scientific changes.

Use the narrowest relevant test during development and run `python -m pytest
-q` before publishing. `.github/workflows/ci.yml` is the authority for clean
installed-wheel, passivity, notebook, regression, and repository-hygiene
checks.

## Git discipline

Begin implementation from a clean, current `main` and use one focused branch
per coherent change. Preserve unrelated user changes, stage only intended
paths, and inspect the complete diff before publishing.

Do not rewrite history, force-push, bypass CI, or use destructive Git commands.
Do not commit anything or mutate a remote unless the user explicitly requests
publication.

More specific nested `AGENTS.md` files may add durable local requirements but
must not weaken these repository-wide protections.

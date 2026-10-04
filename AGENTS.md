# AGENTS.md

This repository implements controlled smooth BSk24/BSk25 sound-speed deformation,
fail-closed raw assessment, effective one-fluid reconstruction, and optional
TOV/tidal work. The workflow is configs, passive planning, explicit reviewed
execution, read-only loading/validation/status, saved-table plotting, and the
single passive `notebooks/bsk24_experiment.ipynb`. Local results belong below
ignored `runs/`; never add generated results, caches or large data to Git.

Before significant changes, read relevant docs, trace the public and scientific
execution paths, inspect expansion/tests/provenance/result shape and run the
narrowest relevant checks. Reviews perform no edits or scientific calculations.
Implement the smallest coherent change while preserving scientific behavior.

Package/dependency authority is `pyproject.toml`; scientific runtime authority is
`environment.yml`. Names are `eos-generation`, `eos_generation`, `bsk24-trial`.
Preserve the eight public exports, deterministic settings hashes, stable case
IDs, versioned schemas/manifests and reproduction commands. Private modules must
not import through the public facade and create cycles. Public JSON fields are
governed by `configs/schema.json`; expanded numerical profiles participate in
planning, identity and results. No undocumented numerical overrides.

Planning makes zero solver calls and writes zero files. Execution is a separate
operation requiring explicit reviewed authorization. Notebook execution stays
passive with `EXECUTE_REVIEWED_PLAN = False`. No expensive stellar work for
routine verification, packaging, documentation or import checks. Validation is
read-only; plotting consumes saved tables. Writes are atomic and refuse existing
destinations without a separately declared policy.

Preserve physical meaning, units, total energy/rest-mass convention, valid domains,
equations, coefficients, anchors, constants, solver grids, root brackets,
tolerances, surface conditions, jump corrections and acceptance predicates.
Require epsilon>0, P>=0, 0<dP/depsilon<=1 on every accepted continuous cold phase;
retain d epsilon=mu_B d n_B, P=n_B mu_B-epsilon, mu_B=(epsilon+P)/n_B where
available. Energy/pressure use MeV fm^-3 and fixed masses are gravitational solar
masses. Assess the complete raw domain first. Rejected/unresolved proposals get
no reconstruction or stellar work; preserve raw values and exact reasons. Never
clip, repair or smooth a failed raw value into acceptance. A=0 obeys the exact
governed baseline identity. Effective one-fluid reconstruction does not establish
composition, species chemical potentials or beta equilibrium.

Distinguish numerical fitting seams, composition thresholds, physical transitions,
self-bound surfaces and unknown discontinuities; physical labels require support.
Apply required stellar/tidal jump corrections exactly once and fail closed for
unestablished capabilities. Fixed masses require a true bracket on the successful
stable prefix. Maximum mass requires a bracketed and refined turning point.

Executed runs retain canonical settings/hash, expanded profile, stable logical and
physical identities/statuses, exact failures, capability statuses, actual source
and environment identities, source archive availability, calculation/reporting
provenance, strict JSON, exact manifests and portable commands. Source equivalence
is distinct from saved scientific integrity. Historical imports never infer
missing certificates or recalculate science. Preserve immutable fixtures; never
regenerate them from the implementation, weaken scientific tests/tolerances,
convert failure to skip or use finiteness alone as physical proof. Prefer published
values, independent solvers and convergence studies for scientific changes.
Record exact commands, results and justified tolerances for scientific changes.

Use narrow checks while developing and `python -m pytest -q` before publication.
CI governs installed-wheel, passivity, notebook, regression, archive and hygiene
checks. Begin implementation from clean current main on one focused branch.
Preserve unrelated changes, stage only intended paths and inspect the full diff.
Never rewrite history, force-push, bypass CI or use destructive Git commands.
Do not commit, publish or mutate remotes without explicit user authorization.
Nested instructions may strengthen these protections.

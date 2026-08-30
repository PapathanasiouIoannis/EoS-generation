# CFL notebook acceptance evidence

This records acceptance work for the active frozen CFL baseline and preserves
the earlier, now-superseded baseline evidence for auditability. Scientific
execution was explicitly authorized. No result packet or generated reference
fixture is part of the tracked source tree.

The accepted quick/strict interface is
[`cfl_experiment.ipynb`](../notebooks/cfl_experiment.ipynb), using the
unchanged public experiment facade. Release 1.2 also distributes
[`cfl_dataset.ipynb`](../notebooks/cfl_dataset.ipynb), but that extension is
explicitly experimental. Focused tests cover its passive plan, packet
validation, and saved-table reporting contract; those software checks do not
extend the quick/strict acceptance evidence below to `dataset_40`. A separately
authorized matched STRICT comparison and dataset execution review are still
required before qualifying results from that route.
See [quickstart](quickstart.md#notebook-route)
for launch, preview, execution, and passive reopening instructions, and
[the CFL contract](cfl.md) for frozen microphysics, domains, units, and
limitations. Runtime versions remain defined by `environment.yml` and
`pyproject.toml`, not by this report.

## Current active baseline acceptance (2026-08-27)

The active pure, bare, zero-temperature CFL baseline is frozen as follows:

| Quantity | Governed value |
|---|---:|
| Bag constant `B` | `57.5 MeV fm^-3` |
| Derived `B^(1/4)` | `144.97957215191494 MeV` |
| Strange-quark mass `m_s` | `100 MeV` |
| CFL gap `Delta` | `100 MeV` |
| Perturbative multiplier `a4` | `1` (no separate correction) |
| Quark-chemical-potential domain | `[249.31780807778472, 600] MeV` |
| Parameter-set SHA-256 | `3991cb8615d2d29617ccb90c6dc54b23aae64bcc752856d07f17f99abc048307` |

`B=57.5 MeV fm^-3` is the authoritative input; `B^(1/4)` is derived using
the governed `hbar*c` conversion. At the zero-pressure bare surface,
`epsilon=190.2181760065314 MeV fm^-3`, `n_B=0.2543182099894835 fm^-3`,
`mu_B=747.9534242333541 MeV`, and `c_s^2=0.3842809221857484`.
The surface pressure is represented as exact zero after the governed root is
accepted; the retained raw binary64 residual is
`2.3272441558063732e-14 MeV fm^-3`.

This value of `B` is intentionally close to the conventional lower edge set
by the requirement that neutral bulk two-flavor quark matter not destabilize
ordinary nuclei. That constraint is not demonstrated by the frozen CFL-only
equation set and remains an explicit external assumption. In particular,
small changes in the nuclear reference energy, light-quark/electron model,
or conversion constants can move a quoted boundary across `57.5`. The
software therefore does not label this parameter set as independently
passing a two-flavor stability test.

### Reviewed baseline calculations

One undeformed physical case was executed through the public experiment
facade at each governed precision. Geometry values are identity-irrelevant
for the direct `A=0` baseline but were declared as center `600`, width `100`,
and ramp width `50 MeV fm^-3`. Each run requested the fixed gravitational
mass `1.4 M_sun` and retained the exact bare-surface tidal-jump evidence.

| Run | Local experiment path | Reviewed plan SHA-256 | Measured pipeline time |
|---|---|---|---:|
| Quick | `runs/cfl_baseline_b57p5_ms100_d100_quick_validation_v2/experiment_8e7ee419331f` | `83ca2a41b07804153e61c87d54adb91c3ee18a968c59ffc21af6534bad28a0f0` | `26.12 s` |
| Strict | `runs/cfl_baseline_b57p5_ms100_d100_strict_validation/experiment_dbb050b27652` | `1b7f34210a32ccaebc0c3486e903f6b9760140e973888fa8b763e8c25df3991a` | `219.17 s` |

Both sealed experiments pass the public read-only validator, contain one
accepted and zero rejected cases, and pass the deterministic `A=0` identity
check. The first attempted quick launch from standard input was not compatible
with Windows spawned multiprocessing; its unsealed partial directory was
retained as failure evidence and was not overwritten or relabeled.

| Observable | Quick `pilot_background` | Strict `tighter_ode` |
|---|---:|---:|
| Refined `M_max` (`M_sun`) | `2.3466940198160042` | `2.3466940191683263` |
| Radius at `M_max` (km) | `12.43375780351352` | `12.43374842724906` |
| `R_1.4` (km) | `11.885078914951567` | `11.885078909788211` |
| `k2_1.4` | `0.2009435283682922` | `0.20094352816465091` |
| `Lambda_1.4` | `841.4699560289916` | `841.4699533584725` |

All three strict fixed-mass stages expected one vacuum-surface jump and
applied it exactly once. The final strict jump is negative under the governed
outward convention. The requested undeformed target, `M_max` approximately
`2.3467 M_sun`, is therefore reproduced without changing any deformation
parameter or stellar acceptance predicate.

Strict thermodynamic finite-difference first-law residuals decrease from
`1.6946e-7` to `4.2385e-8` to `1.0598e-8`; the sound-speed derivative residuals
decrease from `1.9135e-8` to `4.7878e-9` to `1.1972e-9`. The aggregate status
remains truthfully `mixed_or_nonmonotone_refinement` because two algebraic
identity residuals at the approximately `1e-16` binary64 floor change by a
few `1e-19` rather than strictly decreasing. Packet validation still passes;
the status was neither repaired nor weakened.

### Independent comparison

The no-production-import enthalpy/DOP853/QUADPACK reference solver was updated
only for the new transcribed constants and an explicit stable-branch bracket.
Its script SHA-256 is
`b5f02a8f4b2684da385e256dc1328cb72b98ed53335480006b4dffb630e0ff94`.
It gives `M_max=2.34669401915468 M_sun`, `R_1.4=11.88507890975522 km`,
`k2_1.4=0.20094352816364938`, and
`Lambda_1.4=841.4699533378679`. Relative differences from the strict
production result are respectively `5.82e-12`, `2.78e-12`, `4.98e-12`, and
`2.45e-11`. The quick comparison also passes the predeclared `1e-5` bound.
These are software cross-check residuals, not global physical-error bars.

### Regression status

These results predate the CFL dataset extension described above and do not
qualify that path. Release 1.2 adds focused synthetic regression and passivity
coverage for the extension, but no synthetic test is scientific acceptance
evidence for an executed dataset campaign. All then-existing CFL-focused tests
pass, including two passive run-all notebook executions
from both supported working directories. The broad source regression produced
`229 passed, 1 skipped, 1 deselected`, with three failures confined to
pre-existing user-edited BSk24 notebook state: saved outputs/execution counts
and a triple-quoted alternate settings block. Those hadronic notebook edits
and their active runs were deliberately not modified by the CFL refreeze.
The repository-hygiene test that requires no local `runs/` directory is
separately exercised in a clean copy so existing user results need not be
deleted. After passivating only those two hadronic notebooks inside the
disposable copy, the complete clean-tree result is **233 passed, 1 skipped**;
the skip is the pre-existing Windows directory-symlink capability check.

## Superseded historical baseline record (2026-08-26)

All remaining sections below refer to the superseded
`B^(1/4)=165 MeV`, `m_s=150 MeV`, `Delta=100 MeV` baseline and its isolated
worktree evidence. They are retained to prevent old packets from being
silently migrated or misidentified as results of the active parameter set.

## Reviewed work and provenance

The notebook's default bounded stellar test used:

```text
matter_model = cfl
amplitudes = [-0.02, 0.0, 0.02]
center = 800 MeV fm^-3
width = 150 MeV fm^-3
ramp_width = 100 MeV fm^-3
epsilon_match = surface
calculation = stellar
fixed_masses = [1.4] gravitational M_sun
diagnostics = off
```

Only `precision` changed between quick and strict. No microphysical constant
was adjusted to improve a test or a mass-threshold flag. The frozen
parameter-set hash is
`521f4a242ae13393fe264daff6fe81a5c3cab9a14cef4dc124853b5b80200e76`.

The actual code cells were read from the notebook and executed in their
declared order in one namespace: a passive preview first, then the unchanged
reviewed plan with `EXECUTE_REVIEWED_PLAN=True`. The pinned Conda environment
was activated and imports scoped to this worktree with `PYTHONPATH=src`.
Neither a shared editable installation nor the active hadronic process was
changed. All paths below are relative to this worktree and point to ignored
local artifacts, not portable benchmark fixtures.

| Run | Local experiment path | Reviewed plan SHA-256 |
|---|---|---|
| Quick notebook | `runs/cfl_20260826T065607Z_ebb48dec224c/experiment_5f0489353869` | `0667bff02c24b69790cb3b1fbd8ca3f3b55ad5ffc73b814059cff5041a7c9211` |
| Strict notebook | `runs/cfl_20260826T065704Z_04ac04643509/experiment_2519d4a3a0ab` | `96d6e66a468f6da62bc61e091e9a522cba101fd4fb696b3158b9bcd185fede2f` |
| Cartesian rejection check | `runs/cfl_cartesian_final_acceptance_20260826/experiment_f4dc06e54ec4` | `47f041787d6e0daf0c1fa585c3352debb88f945f0dc82043c1786871dae8b070` |
| All-zero alias check | `runs/cfl_zero_alias_final_acceptance_20260826/experiment_bbd66e1601af` | `292f07a392d4c54bfacee4e673ab620b84203adf10d9f7c1d71d91cd3f175247` |

The sealed packets retain exact settings, expanded profiles, source and
environment hashes, failures, scientific statuses, and manifests. Their
sibling `plots/` directories have separate input/reporting
hashes and manifests. The public validation operation is read-only.

## Quick stellar workflow

The completed quick run has three accepted physical EoSs, no rejected
proposals, 51 successful sampled tidal models, three successfully bracketed
fixed-mass points, and three refined turning-point maximum masses. The
physical A=0 case reuses the single saved analytic `direct` stellar solution;
there is no second supposedly independent baseline solve. The thermodynamic
A=0 identity check passes.

| Amplitude | R at 1.4 M_sun (km) | k2 at 1.4 M_sun | Lambda at 1.4 M_sun | Refined M_max (M_sun) |
|---|---:|---:|---:|---:|
| -0.02 | 9.3933701511 | 0.1190881368 | 153.79064706 | 1.6743954932 |
| 0 | 9.3996585901 | 0.1196010411 | 154.97070208 | 1.6947036158 |
| +0.02 | 9.4054534899 | 0.1200732006 | 156.06266902 | 1.7146117131 |

These are computed local results, not independent expected fixtures. The
configured `1.95 M_sun` maximum-mass comparison flag is false for all three
cases and remains false in the saved results. This flag is separate from
thermodynamic validity and fixed-mass availability; the frozen baseline was
not retuned to pass it.

Seven combined PNGs were generated: M-R, Lambda-M, k2-M, sound speed,
pressure, baryon density, and baryon chemical potential. All applicable rows
were retained, with no missing-data repair. The three main stellar images
were visually inspected. The full sequence can extend beyond the stable
branch; plotting it is not a claim that every point is stable.

Quick retains the truthful convergence statuses `insufficient_stages`
(thermodynamics) and `single_stage_no_numerical_envelope` (stellar). A
successful quick workflow is not a convergence certificate.

## Independent baseline solver

[`tests/cfl_reference_solver.py`](../tests/cfl_reference_solver.py) is an
explicit opt-in acceptance tool, not an ordinary pytest workload. It imports
no production EoS or stellar code and creates no reference fixture. The
finite-mass free-gas integral is evaluated independently with QUADPACK.
The tested script SHA-256 is
`a83527260a3f12b8811991684597d1f15025f74ad260044ecd158f04eed3633b`.
The stellar equations use enthalpy `h=ln(mu/mu_surface)`, state variables
`r^2`, `m/r`, and `y`, and DOP853, whereas production uses radius-based RK45,
its own pressure inversion, and a separately integrated tidal equation.

Equations are transcribed from Lugones and Horvath,
[PRD 66, 074017](https://doi.org/10.1103/PhysRevD.66.074017), and Postnikov,
Prakash, and Lattimer,
[PRD 82, 024016, Eqs. 2, 6, 9-11](https://arxiv.org/abs/1004.5098).
The independently evaluated vacuum jump is
`Delta y=-4 pi R^3 epsilon_surface/M` in geometric units. The Love-number
expression is evaluated with guard digits to avoid cancellation at small
compactness.

The acceptance tolerance was declared as relative error `1e-5` in R, k2,
Lambda at fixed gravitational mass and in the refined maximum mass.
Independent center-offset/ODE refinement must agree within `1e-7` relative.
These tolerances are comparison bounds, not certified global error bars.
No scientific predicate or tolerance was relaxed to obtain the results.

Exact quick comparison command, after activating the pinned environment:

```powershell
python -u tests/cfl_reference_solver.py --packet runs/cfl_20260826T065607Z_ebb48dec224c/experiment_5f0489353869/geometry_001 --stage pilot_background --execute
```

The comparison passed. The refined independent reference gives
`R_1.4=9.399658585160575 km`, `k2_1.4=0.11960104099278994`,
`Lambda_1.4=154.97070149889907`, and
`M_max=1.6947036155720576 M_sun`. Its own fixed-mass refinement changes
R, k2, and Lambda by at most `7.79e-13` relative. The quick production
relative differences are respectively `5.22e-10`, `1.18e-9`, `3.77e-9`,
and `1.42e-10` for maximum mass.

This is implementation independence, not independent authorship or
agreement with a published convention-matched stellar table. It checks the
undeformed baseline; nonzero deformations additionally need their own
reconstruction and grid/ODE convergence evidence.

The final strict comparison also passed, using:

```powershell
python -u tests/cfl_reference_solver.py --packet runs/cfl_20260826T065704Z_04ac04643509/experiment_2519d4a3a0ab/geometry_001 --stage tighter_ode --execute
```

Its relative differences from the independent reference are `4.23e-12`
in R, `1.11e-11` in k2, `3.44e-11` in Lambda at 1.4 M_sun, and `5.25e-12`
in maximum mass. These are observed comparison residuals, not a claim of
equivalent physical accuracy in the bag-model approximation.

## Strict stellar and tidal evidence

The final strict notebook completed all three stellar stages: 61 pressure
samples, 121 samples at the same ODE tolerances, then 121 with tighter ODE
tolerances. It saved **909 successful sampled stars, nine successfully
bracketed fixed-mass points, and nine refined maximum masses** across the
three physical EoSs. All 918 sampled/fixed-mass tidal records have exactly
one negative surface jump at exactly zero pressure. The baseline final-stage
maximum is `1.694703615580952 M_sun`; no sampled peak was substituted for it.

The seven combined final-stage plots were created and the M-R, Lambda-M,
and k2-M images visually checked. Every eligible saved row was retained.
The same-mass quick/strict relative differences across the three cases are
at most `5.18e-10` in R, `2.11e-9` in k2, and `3.74e-9` in Lambda.

The two 121-point stellar stages have exactly matching stored central
pressures. Comparing all 363 corresponding pairs gives these maximum
relative changes when tightening the ODE tolerances:

| Observable | Maximum relative change |
|---|---:|
| Mass | `8.12e-8` |
| Radius | `3.97e-8` |
| k2 | `9.10e-8` |
| Lambda | `1.28e-7` |

The fixed-mass three-stage measured envelopes are at most `5.15e-9 km`
in radius, `1.49e-10` in k2, and `6.13e-7` in Lambda. The saved stellar
status is `complete_all_requested_stages`: this means the requested
evidence is complete, not that an a priori global error bound was proved.
The original grids, ODE/root tolerances, and all missing-data rules are
retained in each packet.

The reported core-pipeline times on this machine were about 50.5 seconds
for quick and 653.4 seconds for strict. These exclude final sealing and
validation, as explicitly defined in `runtime_performance.json`; they are
observations, not runtime guarantees for other settings or machines.

## Strict thermodynamic evidence

The strict profile assesses and reconstructs the same three physical EoSs
at coarse, standard, and refined grids. Independent finite-difference
first-law and sound-speed residuals decrease at each refinement. At the
refined grid, the largest interior first-law residual is `5.65e-9`; the
largest sound-speed derivative residual is `4.75e-9`.

The Euler/chemical-potential algebraic residuals are at most `3.01e-16`,
or identically zero for the deformed chemical-potential relation. They do
not strictly decrease with grid size. The saved aggregate status is
therefore **`mixed_or_nonmonotone_refinement`**, not an unconditional pass.
The scale is consistent with binary64 roundoff, but that interpretation
does not alter any stored value or status. No convergence predicate was
weakened, and `strict` is not presented as automatic scientific validation.

## Cartesian grids, rejection, and reopening

The thermodynamics-only Cartesian check used amplitudes `[0, 0.02, 2]`,
centers `[700, 800]`, width `150`, and ramp width `100`, in quick precision.
It produced six logical rows, five physical cases, one nonexecuting zero
alias, three accepted EoSs, and two raw-proposal rejections. The rejected
cases have exact retained reasons and no reconstruction or stellar work.
Both children and the aggregate validate; the combined catalogue contains
the three accepted physical EoSs and both rejection records.

The separate all-zero check used amplitude `[0]` and the same two geometries.
It produced two logical rows and one physical baseline. The non-owning
geometry has no physical work. Both children, including the alias-only
child, validate; the combined view contains the baseline once.

The saved quick experiment was reopened through the actual notebook in a
real Jupyter kernel, from the `notebooks/` working directory. Test guards
made both scientific execution and new figure rendering raise if called.
Seven inline images appeared. Every file's size and modification time in
the packet and view remained unchanged: no solver calls, no rerendering,
and no packet writes.

## Runtime and installation checks

An isolated wheel installation was tested outside the checkout. It imports
from that wheel, includes both source manifests, plans both models without
writes or solver calls, and loads/validates the saved CFL packet and all
seven existing plots. Four installed-wheel packet portability tests pass.
The shared Conda environment's installed package was not replaced.

The full regression command was

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:MPLBACKEND = "Agg"
python -m pytest -q -p no:cacheprovider
```

Result: **150 passed, 1 skipped**. The skip is the pre-existing Windows
directory-symlink capability test, not a converted scientific failure.
Jupyter reports its Windows selector-thread warning; notebook execution
still completes. The test runner is the available development interpreter;
the scientific quick/strict runs and independent reference use the activated
pinned scientific environment. The path-planning regression now permits
pre-existing user results while explicitly forbidding directory/text/byte
writes and checking that the result-root entries do not change.

The CI metadata/configuration/local-link checks, prospective tracked-tree
allowlist (including intended new files), notebook syntax/empty-output
checks, and `git diff --check` also pass. No remote CI or publication is
claimed; no commit or remote mutation was performed.

The final wheel was built with
`python -m pip wheel . --no-deps --wheel-dir runs/cfl-wheel-check-20260826/final-wheel`.
Its SHA-256 is
`eefe886e8e8cdaf47009f98dd361819c7cf12f7d90a5071bb4e2145fdfc1c817`.
It was installed with `--no-deps` into a disposable virtual environment,
not into the active scientific environment.

Actual execution exposed three platform/model-boundary problems that small
mocked tests had not exercised:

- Windows `fsync` requires a writable file descriptor. Atomic plot
  finalization now opens its completed temporary PNG in `r+b` mode;
  a regression test checks the descriptor without changing image bytes.
- Legacy hadronic minimum-mass/radius cuts incorrectly removed valid
  small self-bound stars. The explicitly hashed CFL-only policy
  `bare_self_bound_positive_mass_radius_v1` now requires finite positive
  mass/radius and verified bare-surface metadata, without those hadronic
  cuts. Compactness, domain, and solver/tidal validity checks are unchanged.
  The legacy BSk24 path retains its previous selection. The independent
  Newtonian limits `M=G_CONV epsilon_surface R^3/3` and `k2 -> 3/4`, plus
  `Delta y=-3` for uniform density, test the self-bound behavior.
  A separate constant-sound-speed scaling test multiplies surface density
  and central pressure by four: mass and radius halve, while k2, Lambda,
  and the surface jump remain unchanged within a declared `1e-7` relative
  comparison bound. This follows directly by rescaling the TOV/tidal
  equations and uses no generated expected fixture.
- The pre-fix strict baseline maximum was correctly withheld because local
  log-grid construction generated a second representation of an original
  solved pressure: `428.35813091910745` versus `428.3581309191079` MeV fm^-3.
  Independent integration noise across that intended zero-width interval
  created a false negative secant and three apparent turning brackets.
  The CFL-only `seed_preserving_split_log_pressure_v1` grid subdivides each
  side of the original bracket, retaining all three seed pressures exactly.
  An analytic log-pressure parabola tests node reuse and the refined peak.
  Replaying only the affected maximum required 20 background solves and
  resolved `1.694703615580952 M_sun`, with positive/negative secants and no
  packet writes or tidal recalculation. The full notebook acceptance was
  then rerun with the new explicitly hashed numerical policy. Original
  tolerances, physical gates, and turning-point predicates were preserved.

Two initial direct-interpreter attempts failed because the pinned Conda
environment had not been activated and incompatible DLLs were inherited.
A subsequent pre-fix attempt failed at Windows plot finalization. Their
incomplete local directories were retained as failure evidence, not
sealed or relabeled as successful calculations. Activate Conda before
launching the notebook; see [troubleshooting](troubleshooting.md).

The pre-refinement-fix strict packet remains at
`runs/cfl_20260826T063657Z_04ac04643509/experiment_2519d4a3a0ab` with its
original unresolved maximum-mass evidence intact. It is a development
artifact under the previous numerical contract, not the final acceptance
packet and not silently migrated by the new loader.

## Scope of the evidence

This establishes a usable, guarded notebook workflow and local software/
equation acceptance. It does not establish microscopic CFL composition of
the effective deformed barotropes, the external two-flavor nuclear-stability
assumption, or observational acceptability of the selected frozen baseline.

Publication-level claims still require a convention-matched published
pure-CFL sequence comparison, independent scientific review, and convergence
assessment for the particular deformation domain being claimed. The local
results above must never be regenerated from production and promoted to
independent expected fixtures.

# Changelog

All notable changes to the supported package are recorded here.

## Unreleased

### Documentation

- Reconciled the landing page, quickstart, parameters, method, result/CSV,
  dataset, CFL status, troubleshooting, developer, and notebook guidance with
  the repository restored by pull request #18 and release 1.2.0.
- Corrected the model boundary: BSk24 can retain a certified prefix through
  its first continuous causal crossing, while CFL requires the complete
  formula-derived domain to pass.
- Corrected current notebook defaults and derived-output descriptions without
  changing code cells, scientific settings, result schemas, or calculations.

## 1.2.0 - 2026-08-30

### Added

- Governed `matter_model = "cfl"` support for pure, cold, bare, self-bound
  CFL quark stars using the frozen `cfl_bag_full_ms_delta2_v1` formulation.
- Surface-anchored CFL sound-speed deformation, complete-domain raw gating,
  cold first-law reconstruction, deterministic zero-amplitude aliasing, and
  bare-surface TOV/tidal handling with exactly-once `y`-jump evidence.
- A dedicated passive-by-default `notebooks/cfl_experiment.ipynb` workflow
  with editable deformation controls, governed quick/strict profiles, and
  saved-table M-R, Lambda-M, k2-M, and thermodynamic plots.
- A passive-by-default `notebooks/cfl_dataset.ipynb` workflow using the
  explicitly experimental 40-point profile at final STRICT ODE tolerances.
  Scientific child-packet PNG rendering is disabled; complete tables are
  retained and exactly five combined saved-table plots are built afterward.
- A frozen CFL source manifest, scientific contract, acceptance record,
  focused regression suite, and opt-in independent reference solver.

- Separate `dataset_10_tighter` combined sampling/tolerance pilot: 10 sequence
  stars with all-node tides at rtol=1e-11 / atol=1e-13. Existing profiles,
  notebook defaults, physical gates and maximum-mass rules remain unchanged.
  This single-stage experiment is not STRICT certification.

- Explicit `dataset_20` sampling pilot: 20 sequence stars with all-node tides
  at rtol=1e-10 / atol=1e-12. No changes to existing profiles, defaults,
  thermodynamics, physical gates, maximum-mass rules or validation.

- Separate `dataset_40` sampling candidate: 40 sequence stars and all-node
  tides at dataset's tight rtol=1e-10 / atol=1e-12. Existing profiles,
  notebook defaults, certification and maximum-mass logic are unchanged.

- Separate `dataset_relaxed_80` sampling candidate: 80 sequence stars with
  all-node tides at the existing dataset_relaxed tolerances (1e-8 / 1e-10).
  Existing profiles, notebook defaults and maximum-mass logic remain unchanged.

- Explicit experimental `dataset_relaxed` tolerance-only candidate, selectable
  in the focused notebook: rtol 1e-8 / atol 1e-10 with all 61 tidal sequence
  points and 1201 radial samples. Existing profiles, notebook defaults,
  physical gates, refinement rules and validators are unchanged.

- Experimental `dataset` profile and a separate focused notebook with five
  combined accepted-only figure families. STRICT/QUICK profiles, all physical
  gates, solvers and tolerances remain unchanged. Dataset uses one 61-model
  stellar stage at STRICT final tolerances and retains maximum-mass checks;
  it explicitly lacks a per-case stellar refinement envelope. See docs/dataset.md.

### Changed

- The frozen pure-CFL baseline is refrozen to the Lugones-Horvath high-mass
  family at authoritative `B = 57.5 MeV fm^-3`, `m_s = 100 MeV`, and
  `Delta = 100 MeV`. The parameter-set ID/hash, formula-derived self-bound
  surface and domain, source manifest, planning identities, and CFL reference
  evidence change accordingly. Historical packets with the former
  `B^(1/4) = 165 MeV`, `m_s = 150 MeV` identity are not migrated or relabeled.

- Public planning, configuration, provenance, validation, and reporting now
  retain the matter model and CFL frozen-parameter identity while omitted
  `matter_model` preserves legacy BSk24 serialization and deterministic IDs.
- CFL uses its explicit bare-self-bound low-mass policy and finite-density
  vacuum surface; it never imports the BSk24 crust, matching anchor, or
  hadronic display cuts. Frozen CFL microphysics is not a sweep axis.

- The general BSk24 experiment notebook now defaults to a large
  125-geometry-by-nine-amplitude `dataset_40` campaign. The focused BSk24
  dataset notebook instead selects the curve-only `dataset_40_curves` profile;
  both profiles use 40 sequence points at rtol=1e-10 / atol=1e-12. All tracked
  notebooks are output-clean and remain passive until a fresh preview is
  explicitly executed. The shared production case-worker cap and preview
  budget increase to six, still bounded by CPU/case count with nested pools
  disabled. Existing profile definitions, including multi-stage STRICT,
  scientific equations, acceptance gates, and validation are unchanged.

- Skipped response figures retain their mandatory final-stage population and
  tidal-completeness metadata. Disabling rendering does not disable evidence;
  the scientific validator is unchanged and still rejects omitted metadata.

- General BSk24 notebook runs publish persistent friendly `H` aliases and
  labelled primary-table copies in `EOS_DATA/`. An append-only catalogue
  shares BSk24 IDs across sign batches and matching QUICK/STRICT physical
  definitions while retaining evaluation provenance, rejected cases, and
  missing observables. General CFL presentation labels are run-local `C`
  labels rather than catalogue registrations.
- Derived notebook views remain separate from sealed packets. The focused
  BSk24 route publishes only its five checksum-manifested figures; the
  focused CFL route publishes `CFL_DATASET/` with two labelled CSVs and five
  figures. General notebook presentation routes create their documented
  saved-table views. Every route excludes rejected cases from accepted-EoS
  plots and makes zero solver calls while presenting saved results.

## 1.1.0 - 2026-08-25

### Added

- A derived, checksum-manifested notebook `STUDENT_VIEW` containing saved
  plots, primary CSV data, optional diagnostics, and schema-grounded guidance.
- Passive notebook protection against mixed Conda server/kernel environments.
- Focused Windows coverage for atomic plot finalization and the notebook
  presentation workflow.
- Continuous, geometry-aware raw-proposal assessment that refines the first
  causal crossing, detects narrow mechanical/causal excursions, and certifies
  analytical deformation resolution before reconstruction.
- Explicit retained-endpoint, hard-validity, fixed-mass availability,
  maximum-mass availability, and student-view eligibility evidence.
- Git-free GitHub archive compatibility coverage.

### Changed

- A mechanically valid deformation that becomes superluminal before direct
  BSk24 now retains the certified prefix through its first continuous
  `c_s^2 = 1` crossing instead of being rejected solely for a shorter causal
  domain. Later returns below one remain outside the usable branch.
- Stellar searches and roots are bounded by each retained EoS endpoint.
  Requested fixed-mass results remain available when valid even if the
  retained endpoint prevents a maximum-mass turning point from being
  established.
- `STUDENT_VIEW` eligibility now depends on all explicitly requested fixed
  masses, independently of maximum-mass availability.
- Finite auxiliary thermodynamic diagnostics remain visible and nonblocking;
  nonfinite/unusable reconstruction, interpolation, inversion, or matching
  evidence still fails closed.
- Deformation centers outside the deformable domain are supported when their
  four-sigma tail has meaningful in-domain overlap.

### Fixed

- Open temporary plot files in a writable mode before `fsync` on Windows.
- Retry transient Windows directory rename/share violations with bounded
  backoff, same-volume atomic publication, per-attempt no-overwrite checks,
  and staged-directory cleanup.
- Correct notebook result links and present causal endpoints, hard validity,
  fixed-mass availability, and maximum-mass availability separately while
  preserving passive-by-default execution.
- Reject forged retained endpoints and malformed or contradictory saved raw,
  thermodynamic, lifecycle, sequence, fixed-mass, and maximum-mass evidence.

### Scientific compatibility

- BSk24 coefficients, the deformation formula, physical constants, units,
  solvers, governed profile parameters/tolerances, public configuration
  schema, deterministic settings hashes, case-ID algorithm, case ordering,
  and the top-level `eos_generation_trial_packet_v1` schema are unchanged.
  Resolution certification adds deterministic geometry-scale samples without
  weakening a scientific predicate.
- Source hashes and source-bound reviewed plan hashes necessarily change with
  this implementation. Existing reviewed plan hashes must not be reused.
- Nested scientific evidence advances to `eos_generation_raw_gate_v2`,
  `bsk24_selected_domain_thermodynamic_gate_v2`,
  `bsk24_maximum_mass_reports_v2`, and
  `tov_resolved_maximum_mass_v2`. The v2 maximum-mass schema makes the
  unresolved threshold decision explicitly nullable; legacy v1 stellar
  evidence is not relabeled as v2.
- CSV and metadata additions are additive. `maximum_mass_status` keeps its
  legacy stellar-convergence meaning; the new maximum-mass availability
  summary is stored separately.

## 1.0.0 - 2026-08-18

This is the first release of the compact public interface.

### Added

- The installable `eos_generation` package and `bsk24-trial` command.
- Nine scientific experiment settings plus a required machine-readable
  `$schema` annotation.
- Passive planning, explicitly authorized execution, saved-result loading,
  validation, and plotting.
- A single passive-by-default notebook backed by the production API.
- Named `quick` and `strict` numerical profiles whose expanded settings are
  saved with each result.
- The MIT License.

### Changed

- The repository layout now follows the supported user workflow: configure,
  plan, run, validate, and inspect.
- Local calculations are written below the ignored `runs/` directory.
- The public Python interface is centered on `Experiment`,
  `ExperimentSettings`, `ExperimentPlan`, and `ExperimentResult`.

### Scientific compatibility

- The BSk24 analytical fit, smooth-window deformation, thermodynamic gates,
  effective one-fluid reconstruction, TOV/tidal solvers, units, and governed
  numerical profiles are unchanged by the packaging refactor.
- Invalid raw proposals still fail before reconstruction or stellar work.

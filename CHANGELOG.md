# Changelog

All notable changes to the supported package are recorded here.

## Unreleased

### Changed

- Restored the supported single-method BSk24 surface and canonical notebook
  after an unreleased bulk dataset and alternate-matter-model synchronization.
- Excluded repository-local publication scratch, generated output, website
  sources, archives, and root-level notebook copies from version control.

### Fixed

- Require an explicit successful validation status before the notebook treats
  an executed experiment as complete; missing validation status now fails
  closed.

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

# Changelog

All notable changes to the supported package are recorded here.

## Unreleased

### Added

- A derived, checksum-manifested notebook `STUDENT_VIEW` containing saved
  plots, primary CSV data, optional diagnostics, and schema-grounded guidance.
- Passive notebook protection against mixed Conda server/kernel environments.
- Focused Windows coverage for atomic plot finalization and the notebook
  presentation workflow.

### Fixed

- Open temporary plot files in a writable mode before `fsync` on Windows.

### Scientific compatibility

- No equation, coefficient, physical constant, numerical profile, tolerance,
  gate, solver, unit, configuration identity, source table, canonical packet,
  or manifest rule changed.

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

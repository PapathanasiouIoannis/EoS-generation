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
passive.

The CLI execution gate is deliberately two-part: `run` requires both the
hash from the exact reviewed passive plan and the explicit `--execute` flag.
Any settings, source, environment, or destination change requires a new plan.

The public configuration has the required `$schema` annotation and the nine
scientific settings in `configs/schema.json`.
Normalize scalar/list geometry deterministically, expand `quick` or `strict`
to its complete internal settings, and hash the resolved scientific
configuration canonically. Destination and execution authorization are
operational controls rather than hidden scientific settings.

## Scientific separation

Keep these layers explicit:

1. passive settings validation and work planning;
2. baseline and raw windowed proposal construction;
3. complete-domain physical gating;
4. effective thermodynamic reconstruction for accepted cases;
5. optional stellar and tidal work;
6. deterministic serialization and manifest sealing;
7. read-only loading, validation, status, and saved-table plotting.

A rejected raw proposal must not leak into downstream layers. Debug-only
transformations must retain the original arrays and cannot change acceptance.

Do not split tightly coupled TOV/tidal equations merely for cosmetic file
size. Refactor only across boundaries that preserve units, interpolation and
inversion authorities, jump corrections, surface conditions, root brackets,
stable-branch logic, and error semantics.

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

## Result integrity

Writes belong below a user-selected new path under `runs/`. Use atomic writes,
strict finite JSON, deterministic table order, exact manifest coverage, and
stable case IDs. Preserve accepted/rejected status and exact failure reason
for every declared case.

Loading and validation are passive. Plotting reads saved tables only. Source
provenance must include every active calculation and reporting module; update
the inventory and its test when a source path changes.

## Tests

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
- smooth-window geometry and raw-gate behavior;
- compact independent continuous-star TOV/tidal regression;
- result integrity and read-only validation;
- notebook passivity and delegation to the production API.

Do not regenerate a reference fixture with the implementation it checks. Do
not weaken a tolerance, gate, or expected status to obtain a pass.

## Installed-wheel check

Before publishing a release, test the built artifact rather than relying only
on editable imports:

```powershell
python -m pip install build
python -m build --wheel
python -m pip install --force-reinstall dist/eos_generation-1.0.0-py3-none-any.whl
```

From outside the checkout, import the public objects, run `bsk24-trial
--help`, and make a passive plan with an absolute config path. The plan must
leave its working directory empty.

## Change review

Report software behavior, numerical behavior, and physical interpretation as
separate concerns. Include exact commands, test results, units, tolerances,
and unresolved scientific decisions. Do not launch an expensive stellar
calculation for routine packaging, documentation, import, or passivity checks.

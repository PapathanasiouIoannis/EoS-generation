# EoS Generation

[![CI](https://github.com/PapathanasiouIoannis/EoS-generation/actions/workflows/ci.yml/badge.svg)](https://github.com/PapathanasiouIoannis/EoS-generation/actions/workflows/ci.yml)

EoS Generation is a Python 3.12 package for controlled sound-speed
deformations of analytical BSk24 neutron-star matter.

The package constructs a smooth change to dimensionless `c_s^2 = dP/dε`,
applies fail-closed checks, reconstructs an effective cold
one-fluid barotrope, and can optionally calculate TOV and tidal observables.
It does not model microscopic composition.

## Execution is opt-in

The supported workflow is:

```text
settings -> passive plan -> explicit execution -> validation -> saved-data reporting
```

`bsk24-trial plan` and `plan_experiment(...)` make zero scientific solver
calls and zero filesystem writes. A run begins only when both the reviewed
plan hash and the explicit execution gate are supplied. Results are written
below the checkout-local, Git-ignored `runs/` directory and are never silently
overwritten.

## Install from a source checkout

The project is not published on PyPI. Clone this repository, then run these
commands from the directory containing `pyproject.toml` and
`environment.yml`:

```powershell
conda env create -f environment.yml
conda activate eos-generation
python -m pip install -e . --no-deps
```

For the two Jupyter notebooks, install the optional notebook tools as well:

```powershell
python -m pip install -e ".[notebook]"
```

Confirm the package, version, and command:

```powershell
python -c "import eos_generation; print(eos_generation.__version__)"
bsk24-trial --help
```

The governed Python and numerical-library versions are declared in
[`environment.yml`](environment.yml) and [`pyproject.toml`](pyproject.toml).
See the [quickstart](docs/quickstart.md) for environment and notebook-kernel
checks, or [troubleshooting](docs/troubleshooting.md) if Conda or the command is
not found.

## Safe first run

Start with the small thermodynamics-only BSk24 configuration. Planning is the
safe first action:

```powershell
bsk24-trial plan --config configs/quickstart.json --output-root runs
```

Review the resolved settings, case list, expanded numerical profile, work
estimate, destination, and plan hash. Then replace `REVIEWED_PLAN_HASH` below
with the exact hash printed by that plan:

```powershell
bsk24-trial run --config configs/quickstart.json --output-root runs --plan-hash REVIEWED_PLAN_HASH --execute
```

The run prints a deterministic experiment path. Substitute that printed path
for the illustrative one below, then inspect it without rerunning science:

```powershell
bsk24-trial validate runs/experiment_0123456789ab
bsk24-trial status runs/experiment_0123456789ab
bsk24-trial plot runs/experiment_0123456789ab
```

## Supported scientific scope

BSk24 is a unified cold neutron-star EoS. Total energy density and pressure
are in MeV fm^-3. An otherwise valid proposal may retain the certified prefix
through its first continuous `c_s^2 = 1` crossing. The crossing is included;
later values remain raw evidence only.

Amplitudes are dimensionless additions to `c_s^2` in units with `c = 1`.
`center`, `width`, `ramp_width`, and numeric anchors are
total-energy-density coordinates in MeV fm^-3. Fixed masses are gravitational
masses in solar masses.

Accepted reconstructions are effective one-fluid cold barotropes. They do not
establish particle fractions, species chemical potentials, or beta
equilibrium. Rejected or unresolved proposals retain their raw evidence and
receive no downstream reconstruction or stellar work. Nothing is clipped,
smoothed, repaired, or extrapolated into acceptance.

For stellar runs, a fixed-mass result requires a true stable-branch bracket
inside the retained EoS domain. Maximum mass is reported only after a turning
point is bracketed and refined. Valid fixed-mass results can remain available
when a causal endpoint prevents maximum-mass resolution.

Read the [method](docs/method.md) before interpreting results.

## Configurations and entry points

The distribution name is `eos-generation`, the import package is
`eos_generation`, and the command is
`bsk24-trial`.

Public JSON settings are governed by [`configs/schema.json`](configs/schema.json).
Useful starting files are:

- [`configs/quickstart.json`](configs/quickstart.json): small BSk24
  thermodynamics check;
- [`configs/custom_experiment.json`](configs/custom_experiment.json): strict
  BSk24 thermodynamics template; and
- [`configs/stellar_example.json`](configs/stellar_example.json): strict BSk24
  stellar example.

[`configs/final_negative_dataset.json`](configs/final_negative_dataset.json)
is a retained campaign configuration, not a starter template. Review its large
Cartesian expansion passively before any use. The complete field and precision
profile contract is in [Parameters](docs/parameters.md).

### Python API

The supported top-level imports are:

```python
from eos_generation import (
    Experiment,
    ExperimentPlan,
    ExperimentResult,
    ExperimentSettings,
    load_experiment,
    plan_experiment,
    run_experiment,
    validate_experiment,
)
```

A minimal passive-to-explicit workflow is:

```python
settings = ExperimentSettings.from_json("configs/quickstart.json")
plan = plan_experiment(settings, output_root="runs/my-study")
print(plan.summary_text())

# Only after reviewing this exact plan:
result = run_experiment(plan, execute=True)
print(validate_experiment(result.experiment_path)["status"])
```

Use only new destinations below `runs/`. The command line and notebooks are
adapters to this same API.

## Notebook choices

Both tracked notebooks have empty outputs and execution counts, and both
default to `EXECUTE_REVIEWED_PLAN = False`. Their checked-in settings are not
equally suitable for a first run:

| Notebook | Checked-in scope | Derived user view |
|---|---|---|
| [`bsk24_experiment.ipynb`](notebooks/bsk24_experiment.ipynb) | Large 125-geometry, nine-amplitude `dataset_40` stellar campaign: 1,125 logical cases | `STUDENT_VIEW/`, persistent-label `EOS_DATA/`, and combined `plots/` |
| [`bsk24_dataset.ipynb`](notebooks/bsk24_dataset.ipynb) | Large focused BSk24 campaign using experimental `dataset_40_curves` | Exactly five combined plots; no `STUDENT_VIEW/` or `EOS_DATA/` |

For any notebook, run all cells once with execution disabled, review the exact
case count, work, profile, and destination, then change only the execution flag
and run the unchanged plan in the same kernel. A kernel restart or any settings,
source, environment, worker-budget, or destination change requires a fresh
passive preview. Use the JSON quickstarts above for onboarding; the BSk24
notebook defaults are campaign-scale.

See the [notebook route](docs/quickstart.md#notebook-route) and
[dataset qualification boundary](docs/dataset.md).

## Results and reproducibility

Each authorized execution produces a manifest-sealed aggregate experiment
with child geometry packets. Depending on the requested calculation, saved
evidence includes:

- canonical settings, expanded numerical profiles, plan/configuration hashes,
  physical case IDs, and logical zero-amplitude aliases;
- raw gates, accepted/rejected lifecycle status, reconstructed thermodynamic
  profiles, and convergence evidence;
- stellar sequences, fixed-mass outcomes, and turning-point availability; and
- source/environment provenance, exact manifests, and two-step reproduction
  commands.

Validation and status are read-only. `bsk24-trial plot` only inventories saved
plots; `--overwrite` explicitly regenerates applicable figures from validated
saved tables without calling a scientific solver.

Do not commit `runs/`, generated datasets, plots, caches, or executed notebook
outputs. Preserve an important result together with its complete packet and
manifest. See [Results](docs/results.md) and the [CSV data guide](docs/csv-data-guide.md).

## Documentation

| Guide | Purpose |
|---|---|
| [Quickstart](docs/quickstart.md) | Installation, first passive plan, explicit run, and notebook procedure |
| [Parameters](docs/parameters.md) | JSON fields, units, constraints, and governed precision profiles |
| [Method](docs/method.md) | Deformation, reconstruction, causal gates, stellar semantics, and limitations |
| [Results](docs/results.md) | Packet layout, validation/status, plotting, and observable availability |
| [CSV data guide](docs/csv-data-guide.md) | Table schemas, joins, ordering, plotting, and leakage-safe ML preparation |
| [Dataset workflows](docs/dataset.md) | Experimental dataset profiles, focused notebooks, saved-data helpers, and qualification limits |
| [Troubleshooting](docs/troubleshooting.md) | Environment, planning, validation, result, and reporting failures |
| [Developer guide](docs/developer.md) | Architecture, invariants, testing, packaging, and change review |

Release history is in [`CHANGELOG.md`](CHANGELOG.md). Contributions must follow
[`CONTRIBUTING.md`](CONTRIBUTING.md), and security reports must use the private
process in [`SECURITY.md`](SECURITY.md).

## Status, citation, and license

BSk24 workflow contracts are regression-tested, but the experimental dataset
profiles are not STRICT convergence certificates. Assess convergence and
independent scientific references before publication-level claims.

If you use the software, cite the release metadata in
[`CITATION.cff`](CITATION.cff). EoS Generation is distributed under the
[MIT License](LICENSE).

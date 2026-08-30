# EoS Generation

[![CI](https://github.com/PapathanasiouIoannis/EoS-generation/actions/workflows/ci.yml/badge.svg)](https://github.com/PapathanasiouIoannis/EoS-generation/actions/workflows/ci.yml)

EoS Generation is a focused Python package for controlled sound-speed
deformations of two separately governed cold-matter baselines: analytical
BSk24 neutron-star matter and one frozen, pure, bare, self-bound
color-flavor-locked (CFL) quark-matter model. It constructs a smooth proposal,
applies fail-closed thermodynamic checks, reconstructs an effective cold
one-fluid barotrope, and can optionally solve for stellar and tidal
observables.

The repository has one workflow:

```text
settings -> passive plan -> explicit run -> validation -> saved plots
```

Planning never calls a scientific solver and never writes a result. Execution
requires an explicit `--execute` gate. Local results go below the ignored
`runs/` directory.

## Install

Before starting, install either Miniconda or Anaconda. VS Code and JupyterLab
can use Conda environments, but neither application supplies the `conda`
command itself.

The governed Python and scientific-library versions are in
[`environment.yml`](environment.yml). Open the repository folder itself—the
folder containing `environment.yml`, `pyproject.toml`, `configs/`, and
`notebooks/`—then confirm the terminal is at that repository root:

```powershell
Test-Path .\environment.yml
Test-Path .\pyproject.toml
Test-Path .\notebooks\bsk24_experiment.ipynb
```

All three commands must print `True`. Next check Conda and the available
environments:

```powershell
conda --version
conda env list
```

If `conda` is not recognized, stop and follow
[`docs/troubleshooting.md`](docs/troubleshooting.md); a VS Code extension is
not the missing component. If `eos-generation` is absent from `conda env
list`, create it once:

```powershell
conda env create -f environment.yml
```

Whether the environment was existing or newly created, install this checkout
and its notebook tools into it:

```powershell
conda activate eos-generation
python -m pip install -e . --no-deps
python -m pip install -e ".[notebook]"
python -c "import sys, eos_generation; print(sys.executable); print(eos_generation.__version__)"
```

The printed executable must identify the `eos-generation` environment. Conda
provides the governed scientific runtime; a notebook *kernel* is simply the
Python interpreter selected to execute notebook cells.

See [`docs/quickstart.md`](docs/quickstart.md) for installation checks and a
complete first run.

## Choose an entry point

For a first run, use the small thermodynamic configuration in
[`configs/quickstart.json`](configs/quickstart.json), not a campaign notebook.
Its passive plan is fast to inspect, makes zero solver calls, and writes
nothing. The parallel CFL onboarding configuration is
[`configs/cfl_quickstart.json`](configs/cfl_quickstart.json).

The four tracked notebooks are passive by default, but their intended scopes
are different:

- `bsk24_experiment.ipynb` is preconfigured as a large stellar campaign:
  125 deformation geometries crossed with nine amplitudes under the
  experimental `dataset_40` profile. It is not a starter notebook.
- `cfl_experiment.ipynb` is the accepted CFL quick/strict experiment interface.
  Quick remains exploratory, and the recorded software/equation acceptance
  does not by itself establish publication-level physical claims.
- `bsk24_dataset.ipynb` and `cfl_dataset.ipynb` are focused dataset routes.
  Their single-stage `dataset_40_curves` and `dataset_40` profiles are
  experimental and are not STRICT convergence certificates.

Opening or passively previewing any notebook is safe. Setting
`EXECUTE_REVIEWED_PLAN = True` authorizes the exact reviewed plan, so inspect
the case count, solver work, destination, and profile before changing that
flag. See the [notebook route](docs/quickstart.md#notebook-route) for kernel and
two-pass instructions, and [`docs/dataset.md`](docs/dataset.md) for the dataset
qualification boundary.

## Plan before calculating

For the experimental five-figure dataset workflow, use the separate
[`notebooks/bsk24_dataset.ipynb`](notebooks/bsk24_dataset.ipynb) or
[`notebooks/cfl_dataset.ipynb`](notebooks/cfl_dataset.ipynb), and read
[`docs/dataset.md`](docs/dataset.md). The BSk24 route selects
`dataset_40_curves`; the CFL route selects `dataset_40`. Both use one 40-point
stellar stage at rtol=1e-10 / atol=1e-12, with up to six case workers. The
BSk24 curve-only profile omits residual reporting, fixed/max-mass refinements,
retained radial profiles, and duplicate presentation trees while preserving
the raw gate and final curve grids. Neither is a per-case STRICT convergence
certificate. Restart the kernel and review a fresh passive preview after
updating; no execution is enabled by default.

The safe first command is:

```powershell
bsk24-trial plan --config configs/quickstart.json --output-root runs
```

`bsk24-trial` remains the compatibility command name for both matter models.
For the passive CFL example, use:

```powershell
bsk24-trial plan --config configs/cfl_quickstart.json --output-root runs
```

The plan resolves defaults, expands the named numerical profile, lists every
case, estimates work, and shows the destination. Review it before running:

```powershell
$planHash = "COPY_HASH_FROM_PLAN"
bsk24-trial run --config configs/quickstart.json --output-root runs --plan-hash $planHash --execute
$experiment = "runs/experiment_COPY_HASH_FROM_RUN"
bsk24-trial validate $experiment
bsk24-trial status $experiment
bsk24-trial plot $experiment
```

Copy the reviewed hash exactly; the run fails if the settings or destination
no longer match it. The run prints its deterministic
`runs/experiment_<hash>` path; substitute that path in the three inspection
commands. Use the exact same config filename that produced the reviewed hash.
Both supplied quickstarts perform thermodynamics only. Stellar calculations
are deliberately a separate, more expensive choice.

## Configure an experiment

A complete legacy BSk24 JSON configuration has one required `$schema`
annotation and nine scientific settings:

```json
{
  "$schema": "./schema.json",
  "amplitudes": [0.0, 0.01, -0.01],
  "epsilon_match": "standard",
  "center": 200.0,
  "width": 50.0,
  "ramp_width": 40.0,
  "calculation": "thermodynamics",
  "precision": "strict",
  "fixed_masses": [1.4],
  "diagnostics": "off"
}
```

Omitting `matter_model` is the canonical legacy BSk24 contract. A CFL
configuration adds the explicit discriminator `"matter_model": "cfl"` and
must use `"epsilon_match": "surface"`; see
[`configs/cfl_quickstart.json`](configs/cfl_quickstart.json). The frozen CFL
microphysical constants are not public sweep fields.

`center`, `width`, and `ramp_width` accept either one number or a list. The
code expands their combinations deterministically. `precision` selects a
governed internal numerical profile; the fully expanded settings are recorded
with the result, so the simple input does not hide what was calculated.

Start from one of these files:

- [`configs/quickstart.json`](configs/quickstart.json): small passive and
  BSk24 thermodynamic smoke workflow;
- [`configs/cfl_quickstart.json`](configs/cfl_quickstart.json): small passive
  CFL thermodynamic workflow;
- [`configs/custom_experiment.json`](configs/custom_experiment.json): strict
  template for editing;
- [`configs/stellar_example.json`](configs/stellar_example.json): strict
  stellar and tidal example;
- [`configs/schema.json`](configs/schema.json): editor and validation schema.

All quantities named `center`, `width`, `ramp_width`, and a numeric BSk24
`epsilon_match` are total-energy-density coordinates in MeV fm^-3.
Amplitudes add directly to dimensionless `c_s^2` in units with `c = 1`.
Fixed masses are gravitational masses in solar masses.

## Notebooks

Choose the notebook for the intended scope:

```powershell
jupyter lab notebooks/bsk24_experiment.ipynb
jupyter lab notebooks/bsk24_dataset.ipynb
jupyter lab notebooks/cfl_experiment.ipynb
jupyter lab notebooks/cfl_dataset.ipynb
```

Edit only the settings cell. Leave `EXECUTE_REVIEWED_PLAN = False`, run all
cells, and inspect the passive plan. Set it to `True` only after the plan and
destination are correct. The notebook delegates to the same production API as
the command line.

The general BSk24 notebook is a campaign template, not the quickstart: its
checked-in settings expand 125 geometries across nine amplitudes and request
the experimental single-stage `dataset_40` stellar profile. The focused BSk24
dataset notebook uses `dataset_40_curves`, which retains the requested curve
grids and fail-closed gates while omitting fixed/max-mass refinement and other
non-curve products. Preview either plan carefully before authorizing it.

The CFL experiment notebook defaults to a small stellar experiment with
`PRECISION = "quick"`; change to `"strict"` and preview again for the governed
refinement stages. This quick/strict notebook is the accepted CFL interface.
Its frozen microphysics and self-bound surface are not editable sweep controls.

The CFL notebook automatically displays combined M–R, Λ–M, and k₂–M plots
from accepted saved sequences, plus thermodynamic curves and availability
tables. `FIXED_MASSES = [1.4]` adds a comparison point, not a restriction on
the full sequences. Reopen completed results with `LOAD_EXPERIMENT` and
execution disabled; loading and viewing existing plots are read-only. See
the [notebook quickstart](docs/quickstart.md#notebook-route).

For large CFL data collection, `cfl_dataset.ipynb` fixes the experimental
`dataset_40` profile, disables scientific packet PNG rendering, retains all
required evidence and statuses in the authoritative packet, and builds the
seven-file `CFL_DATASET/` view after successful validation. Reporting reads
each required saved table once and makes zero solver calls. The route is
passive by default, but it is outside the quick/strict acceptance evidence and
is not a substitute for a matched STRICT convergence study.

## Python interface

The supported imports are intentionally small:

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

Use the command line or notebook for ordinary work. The Python interface is
available for reproducible automation without reaching into private modules.
The destination must remain below the checkout-local `runs/` directory:

```python
settings = ExperimentSettings.from_json("configs/quickstart.json")
plan = plan_experiment(settings, output_root="runs/my-first-experiment")
print(plan.plan_hash)
print(plan.summary_text())

# Only after reviewing that exact plan:
result = run_experiment(plan, execute=True)
print(result.summary_text())
```

## Scientific boundary

For every proposed deformation, the complete raw result is assessed and saved
before reconstruction. The usable branch ends at the first continuously
resolved `c_s^2 = 1` crossing, which is included as that case's endpoint. A
deformation that reaches this endpoint before direct BSk24 is therefore not
rejected merely for having a shorter causal branch; any later return below one
lies outside the usable EoS. Geometry-aware sampling and bounded continuous
refinement search for narrow unstable or superluminal features, and an
unresolved assessment fails closed. Results are never made acceptable by
clipping, smoothing, repair, or extrapolation.

Stellar central pressures remain inside each case's retained branch. If that
endpoint prevents a maximum-mass turning point from being established, valid
fixed-mass results are kept and maximum-mass availability is reported
separately. Finite auxiliary thermodynamic diagnostics remain visible without
being promoted to rejection criteria; non-finite or unusable reconstruction,
matching, interpolation, or inversion still fails closed.

The reconstruction is an effective one-fluid cold barotrope. It does not
establish microscopic composition, species chemical potentials, or beta
equilibrium. A maximum mass is reported as resolved only when the governed
turning-point procedure succeeds, and fixed-mass observables require a true
stable-branch bracket.

The CFL model is zero-temperature ideal CFL matter with frozen
`m_s = 100 MeV`, `Delta = 100 MeV`, and authoritative
`B = 57.5 MeV fm^-3` (derived `B^(1/4) = 144.97957215191494 MeV`). It describes a
bare quark star joined directly from a finite-density zero-pressure surface to
vacuum, with no crust or hadronic envelope. The CFL-only phase does not test
the separate two-flavor stability condition needed to protect ordinary
nuclei. Extended CFL radial diagnostics are unsupported and fail closed.
Read the exact equations, formula-derived binary64 domain, tidal surface jump,
and validation status in [`docs/cfl.md`](docs/cfl.md) before interpreting a
CFL result.

## Repository map

| Path | Purpose |
|---|---|
| [`src/eos_generation/`](src/eos_generation/) | Installable implementation and public API |
| [`configs/`](configs/) | Five user configurations and their schema |
| [`notebooks/`](notebooks/) | Separate passive-by-default BSk24 and CFL notebooks |
| [`docs/`](docs/) | Short method and usage guides |
| [`tests/`](tests/) | Focused scientific and interface regression tests |

See [`docs/method.md`](docs/method.md) for the construction,
[`docs/results.md`](docs/results.md) for interpreting saved packets, and
[`docs/csv-data-guide.md`](docs/csv-data-guide.md) for using CSV data in
spreadsheets, Python, or machine-learning workflows. The frozen CFL contract
is in [`docs/cfl.md`](docs/cfl.md). Published CFL
stellar claims still require a strict convergence study, published benchmark,
and independent solver comparison; the quick example is exploratory only.
Problems and contribution requirements are covered by
[`docs/troubleshooting.md`](docs/troubleshooting.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md).

Security vulnerabilities must be reported privately as described in
[`SECURITY.md`](SECURITY.md), not through a public issue. Participation in the
project is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License and citation

The software is released under the [MIT License](LICENSE). Citation metadata
is provided in [`CITATION.cff`](CITATION.cff).

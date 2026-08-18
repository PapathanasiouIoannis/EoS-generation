# EoS Generation

[![CI](https://github.com/PapathanasiouIoannis/EoS-generation/actions/workflows/ci.yml/badge.svg)](https://github.com/PapathanasiouIoannis/EoS-generation/actions/workflows/ci.yml)

EoS Generation is a focused Python package for controlled sound-speed
deformations of the analytical BSk24 neutron-star equation of state. It
constructs a smooth proposal, applies fail-closed thermodynamic checks,
reconstructs an effective cold one-fluid barotrope, and can optionally solve
for stellar and tidal observables.

The repository has one workflow:

```text
settings -> passive plan -> explicit run -> validation -> saved plots
```

Planning never calls a scientific solver and never writes a result. Execution
requires an explicit `--execute` gate. Local results go below the ignored
`runs/` directory.

## Install

The governed Python and scientific-library versions are in
[`environment.yml`](environment.yml):

```powershell
conda env create -f environment.yml
conda activate eos-generation
python -m pip install -e . --no-deps
```

For the notebook, install the optional interface tools as well:

```powershell
python -m pip install -e ".[notebook]"
```

See [`docs/quickstart.md`](docs/quickstart.md) for installation checks and a
complete first run.

## Plan before calculating

The safe first command is:

```powershell
bsk24-trial plan --config configs/quickstart.json --output-root runs
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
commands. The supplied quickstart performs thermodynamics only. Stellar
calculations are deliberately a separate, more expensive choice.

## Configure an experiment

A complete JSON configuration has one required `$schema` annotation and nine
scientific settings:

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

`center`, `width`, and `ramp_width` accept either one number or a list. The
code expands their combinations deterministically. `precision` selects a
governed internal numerical profile; the fully expanded settings are recorded
with the result, so the simple input does not hide what was calculated.

Start from one of these files:

- [`configs/quickstart.json`](configs/quickstart.json): small passive and
  thermodynamic smoke workflow;
- [`configs/custom_experiment.json`](configs/custom_experiment.json): strict
  template for editing;
- [`configs/stellar_example.json`](configs/stellar_example.json): strict
  stellar and tidal example;
- [`configs/schema.json`](configs/schema.json): editor and validation schema.

All quantities named `center`, `width`, `ramp_width`, and a numeric
`epsilon_match` are total-energy-density coordinates in MeV fm^-3.
Amplitudes add directly to dimensionless `c_s^2` in units with `c = 1`.
Fixed masses are gravitational masses in solar masses.

## Notebook

Launch the single supported notebook:

```powershell
jupyter lab notebooks/bsk24_experiment.ipynb
```

Edit only the settings cell. Leave `EXECUTE_REVIEWED_PLAN = False`, run all
cells, and inspect the passive plan. Set it to `True` only after the plan and
destination are correct. The notebook delegates to the same production API as
the command line.

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

For every proposed deformation, the raw complete-domain result is assessed
before reconstruction. A proposal that violates the configured finite-domain
requirements, including mechanical stability or causality, is rejected with
its exact reason and receives no stellar calculation. Results are never made
acceptable by clipping or smoothing failed values.

The reconstruction is an effective one-fluid cold barotrope. It does not
establish microscopic composition, species chemical potentials, or beta
equilibrium. A maximum mass is reported as resolved only when the governed
turning-point procedure succeeds, and fixed-mass observables require a true
stable-branch bracket.

## Repository map

| Path | Purpose |
|---|---|
| [`src/eos_generation/`](src/eos_generation/) | Installable implementation and public API |
| [`configs/`](configs/) | Three user configurations and their schema |
| [`notebooks/`](notebooks/) | Passive-by-default experiment notebook |
| [`docs/`](docs/) | Short method and usage guides |
| [`tests/`](tests/) | Focused scientific and interface regression tests |

See [`docs/method.md`](docs/method.md) for the construction and
[`docs/results.md`](docs/results.md) for interpreting saved results. Problems
and contribution requirements are covered by
[`docs/troubleshooting.md`](docs/troubleshooting.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md).

Security vulnerabilities must be reported privately as described in
[`SECURITY.md`](SECURITY.md), not through a public issue. Participation in the
project is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License and citation

The software is released under the [MIT License](LICENSE). Citation metadata
is provided in [`CITATION.cff`](CITATION.cff).

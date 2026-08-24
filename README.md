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

Conda is not required merely because the project contains an `.ipynb` file.
It provides the governed Python and scientific-library environment used by
the command line, VS Code, and browser-based Jupyter alike. A notebook
*kernel* is simply the Python interpreter that executes its cells; select the
`eos-generation` environment rather than an unrelated base or system Python.

See [`docs/quickstart.md`](docs/quickstart.md) for installation checks and a
complete first run.

## First notebook run in VS Code

1. Use **File > Open Folder** and select the repository folder itself: the
   folder containing `environment.yml`, `pyproject.toml`, `configs/`, and
   `notebooks/`. Do not open only its parent download directory.
2. In the Explorer, right-click that top-level folder and select **Open in
   Integrated Terminal**. This avoids typing a machine-specific path.
3. Confirm the terminal is at the repository root:

   ```powershell
   Test-Path .\environment.yml
   Test-Path .\pyproject.toml
   Test-Path .\notebooks\bsk24_experiment.ipynb
   ```

   All three commands must print `True`.
4. Create the environment once, then install the checkout and notebook tools:

   ```powershell
   conda env create -f environment.yml
   conda activate eos-generation
   python -m pip install -e . --no-deps
   python -m pip install -e ".[notebook]"
   ```

   If the environment already exists, skip `conda env create` and activate
   it. If `conda` is not recognized, use **Anaconda Prompt** for these setup
   commands; [`docs/troubleshooting.md`](docs/troubleshooting.md) gives a
   path-free way to get that prompt into the correct folder.
5. Open `notebooks/bsk24_experiment.ipynb`. Select **Python: Select
   Interpreter**, then **Select Kernel**, and choose `eos-generation` in both
   places. The selected interpreter must be Python 3.12 from that environment.
6. Leave `EXECUTE_REVIEWED_PLAN = False` and choose **Run All**. This first
   pass only prints the passive plan and writes no result.
7. Review the settings, work count, cases, and new destination. Change only
   `EXECUTE_REVIEWED_PLAN` to `True`, then choose **Run All** again in the
   same kernel. Restarting the kernel discards the reviewed authorization and
   requires another `False` preview pass.

After a successful run, use the notebook's **Student result locations** links.
Start with `STUDENT_VIEW/01_READ_ME_FIRST.md`; primary CSV data is below
`STUDENT_VIEW/03_PRIMARY_DATA/` and the sealed technical packet remains
separate.

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

## Browser-based Jupyter Lab

The same notebook can be run in a browser instead of VS Code. Activate the
same environment at the repository root and launch:

```powershell
python -m jupyterlab notebooks/bsk24_experiment.ipynb
```

Edit only the settings cell. Leave `EXECUTE_REVIEWED_PLAN = False`, run all
cells, and inspect the passive plan. Set it to `True` only after the plan and
destination are correct, then run all again in the same kernel. On completion
it validates the sealed experiment and creates a separate derived
`STUDENT_VIEW/` with links to plots and primary CSV data.

If shell activation is uncertain on Windows, launch explicitly through the
governed environment:

```powershell
conda run -n eos-generation --no-capture-output python -m jupyterlab notebooks/bsk24_experiment.ipynb
```

VS Code and Jupyter Lab delegate to the same production API and create the
same result structure. They are different interfaces, not different
scientific workflows.

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

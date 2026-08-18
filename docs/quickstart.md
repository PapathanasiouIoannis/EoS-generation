# Quickstart

This guide performs a small thermodynamic BSk24 experiment. Planning is
passive; only the explicitly authorized `run` command performs scientific
work or creates a result directory.

## 1. Create the environment

Run these commands from the repository root:

```powershell
conda env create -f environment.yml
conda activate eos-generation
python -m pip install -e . --no-deps
```

Confirm that Python imports the installed package and that the command is
available:

```powershell
python -c "import eos_generation; print(eos_generation.__version__)"
bsk24-trial --help
```

The supported interpreter and numerical-library versions are declared only
in `environment.yml` and `pyproject.toml`.

## 2. Inspect the settings

Open [`../configs/quickstart.json`](../configs/quickstart.json). It requests
two amplitudes at one geometry, uses the small `quick` numerical profile, and
does not request stellar calculations.

The user-facing fields are explained in
[`parameters.md`](parameters.md). Their units matter: energy-density
coordinates are in MeV fm^-3, the amplitude is dimensionless, and fixed
masses are in solar masses.

## 3. Make a passive plan

```powershell
bsk24-trial plan --config configs/quickstart.json --output-root runs
```

Review the resolved settings, deterministic hash, expanded numerical profile,
case list, work count, and destination. This command must make zero solver
calls and zero filesystem writes. It is safe to use while learning the
interface.

If the plan is not what you intended, edit the JSON and plan again. Do not
edit implementation constants to configure an experiment.

## 4. Execute the reviewed plan

Use the reviewed output root and execute once:

```powershell
$planHash = "COPY_HASH_FROM_PLAN"
bsk24-trial run --config configs/quickstart.json --output-root runs --plan-hash $planHash --execute
```

Copy the hash from that exact reviewed plan; `--plan-hash` and `--execute` are
both required. Execution fails if the hash no longer matches the settings or
destination. The run prints the exact deterministic
`runs/experiment_<hash>` destination. An existing destination is not silently
overwritten. To repeat the same settings independently, plan and run with a
new output root such as `runs/repeat-01`.

Execution assesses every raw proposal before reconstruction. A rejected case
is a recorded scientific outcome, not automatically a software error. It
keeps its exact reason and receives no downstream calculation.

## 5. Validate and inspect

```powershell
$experiment = "runs/experiment_COPY_HASH_FROM_RUN"
bsk24-trial validate $experiment
bsk24-trial status $experiment
bsk24-trial plot $experiment
```

Validation is read-only and checks the result structure, hashes, and declared
scientific completeness. `status` presents the result summary. The passive
`plot` invocation reports the saved plot inventory; it does not rerun the
calculation.

Read [`results.md`](results.md) before interpreting accepted/rejected cases,
stellar branches, or missing observables.

## Notebook route

Install the optional interface tools and launch Jupyter:

```powershell
python -m pip install -e ".[notebook]"
jupyter lab notebooks/bsk24_experiment.ipynb
```

In the settings cell, leave:

```python
EXECUTE_REVIEWED_PLAN = False
```

Run all cells once and inspect the plan. Set the flag to `True` only after the
settings, work count, and new destination are correct. Authorization is bound
to that reviewed plan; settings, source, environment, worker, or destination
drift requires a new passive pass.

## Next steps

- Copy [`../configs/custom_experiment.json`](../configs/custom_experiment.json)
  to vary amplitudes or geometry with the strict profile.
- Inspect [`../configs/stellar_example.json`](../configs/stellar_example.json)
  before requesting the more expensive stellar route.
- Read [`method.md`](method.md) for the construction and physical gates.

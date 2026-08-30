# Quickstart

This guide performs a small thermodynamic BSk24 experiment. Planning is
passive; only the explicitly authorized `run` command performs scientific
work or creates a result directory.

There are three interfaces, not three different calculations:

| Interface | What the user opens | Execution environment |
|---|---|---|
| VS Code notebook | `notebooks/bsk24_experiment.ipynb` in VS Code | Selected `eos-generation` kernel |
| Jupyter Lab notebook | The same `.ipynb` in a browser | Selected `eos-generation` kernel |
| Command line | `bsk24-trial` in a terminal | Activated `eos-generation` environment |

All routes use the same package, configuration contract, solvers, validation,
and result format. Conda supplies the reproducible scientific environment;
the notebook kernel is that environment's Python interpreter.

## 0. Open the repository root

The repository root is the folder containing all three of these files:

```text
environment.yml
pyproject.toml
notebooks/bsk24_experiment.ipynb
```

In VS Code, use **File > Open Folder** and select that folder, not its parent.
Then right-click the top folder in the Explorer and choose **Open in
Integrated Terminal**. Verify it without entering a user-specific path:

```powershell
Test-Path .\environment.yml
Test-Path .\pyproject.toml
Test-Path .\notebooks\bsk24_experiment.ipynb
```

Continue only when all three results are `True`. Text such as “path to the
repository” in prose is an instruction to select or paste a real path; do not
type angle brackets such as `<path>` into `cd`.

## 1. Create the environment

Run these commands once from the repository root:

```powershell
conda env create -f environment.yml
conda activate eos-generation
python -m pip install -e . --no-deps
```

For either notebook interface, also install the notebook tools:

```powershell
python -m pip install -e ".[notebook]"
```

If `eos-generation` already exists, do not create a second copy merely to use
a notebook. Activate the existing environment, install the current checkout,
and select that same interpreter as the kernel. If `conda` is not recognized,
follow the Anaconda Prompt instructions in
[`troubleshooting.md`](troubleshooting.md).

Confirm that Python imports the installed package and that the command is
available:

```powershell
python -c "import sys, eos_generation; print(sys.executable); print(eos_generation.__version__)"
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

Execution assesses and saves every complete raw proposal before
reconstruction. It locates the first continuous `c_s^2 = 1` crossing and, for
an otherwise valid case, includes that crossing as the retained endpoint. A
later return below one is outside the usable branch. A rejected or unresolved
case is a recorded scientific outcome, not automatically a software error. It
keeps its raw evidence and exact reason and receives no downstream
calculation; no result is clipped, repaired, or extrapolated into acceptance.

## 5. Validate and inspect

```powershell
$experiment = "runs/experiment_COPY_HASH_FROM_RUN"
bsk24-trial validate $experiment
bsk24-trial status $experiment
bsk24-trial plot $experiment
```

Validation is read-only and checks the result structure, hashes, and hard
scientific validity while reporting observable availability separately.
`status` presents that summary. A valid packet may be scientifically partial,
for example when requested fixed masses were solved inside the retained EoS
domain but the endpoint prevented a maximum-mass turning point from being
resolved. The passive `plot` invocation reports the saved plot inventory; it
does not rerun the calculation.

Read [`results.md`](results.md) before interpreting accepted/rejected cases,
stellar branches, or missing observables. Read
[`csv-data-guide.md`](csv-data-guide.md) for the exact CSV ordering, column
meanings, analysis recipes, and machine-learning preparation.

## Notebook route

### VS Code

1. Open `notebooks/bsk24_experiment.ipynb`.
2. Use **Python: Select Interpreter**, then the notebook's **Select Kernel**
   menu, and choose `eos-generation` in both places. It must be the Python
   3.12 interpreter from the declared environment.
3. Edit only the settings cell.
4. Keep `EXECUTE_REVIEWED_PLAN = False` and choose **Run All**.
5. Review the printed settings, cases, work count, and destination.
6. Change only `EXECUTE_REVIEWED_PLAN` to `True` and choose **Run All** again
   in the same kernel.

The first pass performs zero solver calls and writes nothing. The second pass
executes only the exact plan retained in that kernel. If the kernel is
restarted or any governed input changes, repeat the `False` preview.

### Jupyter Lab in a browser

Activate the same environment and launch Jupyter from the repository root:

```powershell
python -m jupyterlab notebooks/bsk24_experiment.ipynb
```

On Windows, this explicit alternative also guarantees that both the Jupyter
server and kernel start from the governed environment:

```powershell
conda run -n eos-generation --no-capture-output python -m jupyterlab notebooks/bsk24_experiment.ipynb
```

Select the `eos-generation` kernel and use the same two-pass procedure. Verify
`sys.executable` identifies `envs\eos-generation\python.exe`, not the base
`anaconda3\python.exe`. In the settings cell, the first pass must leave:

```python
EXECUTE_REVIEWED_PLAN = False
```

Run all cells once and inspect the plan. Set the flag to `True` only after the
settings, work count, and new destination are correct. Authorization is bound
to that reviewed plan; settings, source, environment, worker, or destination
drift requires a new passive pass.

On a successful authorized pass, the notebook validates the completed
experiment and then creates a separate `STUDENT_VIEW/` with obvious links to
plots, primary CSV data, and the authoritative technical packet. This derived
view is outside the sealed packet and has its own checksum manifest. The
notebook uses `../runs/...` links because it is stored in `notebooks/`; those
links resolve correctly in both VS Code and browser Jupyter. A stellar case
remains student-view eligible when every explicitly requested fixed mass
succeeded even if maximum mass is explicitly unavailable. The copied
`case_ledger.csv` exposes the retained endpoint, requested-fixed-mass,
maximum-mass-availability, and student-view-eligibility statuses separately.

Start with `STUDENT_VIEW/01_READ_ME_FIRST.md`. For portable EoS analysis,
keep `case_ledger.csv` and `thermodynamic_profiles.csv` from the same
`geometry_NNN` directory together. The ledger says which deformation a
`case_id` represents; the profiles hold its sampled data points. The complete
row, ordering, column, and machine-learning guidance is documented in
[`csv-data-guide.md`](csv-data-guide.md); packet validation and lifecycle
interpretation remain in [`results.md`](results.md).

## Next steps

- Copy [`../configs/custom_experiment.json`](../configs/custom_experiment.json)
  to vary amplitudes or geometry with the strict profile.
- Inspect [`../configs/stellar_example.json`](../configs/stellar_example.json)
  before requesting the more expensive stellar route.
- Read [`method.md`](method.md) for the construction and physical gates.

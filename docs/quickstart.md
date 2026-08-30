# Quickstart

This guide performs a small thermodynamic BSk24 experiment and shows the
parallel passive entry point for the frozen CFL model. Planning is passive;
only the explicitly authorized `run` command performs scientific work or
creates a result directory. The historical `bsk24-trial` command name remains
the compatibility interface for both models.

The first-run route in this guide is the command line with
`configs/quickstart.json`. It is intentionally small. The notebooks use the
same package, configuration contract, solvers, validation, and result format,
but their checked-in settings serve different campaign purposes:

| Entry point | Checked-in scope |
|---|---|
| `configs/quickstart.json` through `bsk24-trial` | Small BSk24 thermodynamic onboarding run |
| `configs/cfl_quickstart.json` through `bsk24-trial` | Small CFL thermodynamic workflow check |
| `notebooks/bsk24_experiment.ipynb` | Large 125-geometry, nine-amplitude `dataset_40` stellar campaign |
| `notebooks/cfl_experiment.ipynb` | Accepted CFL quick/strict experiment interface |
| `notebooks/bsk24_dataset.ipynb` and `cfl_dataset.ipynb` | Experimental focused dataset routes |

Conda supplies the reproducible scientific environment; a notebook kernel is
that environment's Python interpreter. Passive notebook preview is safe, but
the general BSk24 notebook is not the small starter calculation documented in
steps 2--5 below.

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

The CFL counterpart is
[`../configs/cfl_quickstart.json`](../configs/cfl_quickstart.json). It declares
`"matter_model": "cfl"`, uses the required `"epsilon_match": "surface"`, and
also requests thermodynamics only. Its microphysical constants are frozen in
the versioned baseline profile and cannot be changed through settings. Read
[`cfl.md`](cfl.md) for its exact equation set and limitations.

The user-facing fields are explained in
[`parameters.md`](parameters.md). Their units matter: energy-density
coordinates are in MeV fm^-3, the amplitude is dimensionless, and fixed
masses are in solar masses.

## 3. Make a passive plan

```powershell
bsk24-trial plan --config configs/quickstart.json --output-root runs
```

To inspect CFL instead, substitute its config without changing any other
planning rule:

```powershell
bsk24-trial plan --config configs/cfl_quickstart.json --output-root runs
```

Review the resolved settings, deterministic hash, expanded numerical profile,
case list, work count, and destination. This command must make zero solver
calls and zero filesystem writes. It is safe to use while learning the
interface.

If the plan is not what you intended, edit the JSON and plan again. Do not
edit implementation constants to configure an experiment.

For any multi-geometry BSk24 or CFL plan, inspect both logical identity
aliases and the physical work count. Exactly one deterministic geometry owns
the physical zero-amplitude baseline; the other logical zero rows do not
repeat execution.

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
If you reviewed the CFL config, use that same CFL filename in the run command;
a BSk24 hash cannot authorize a CFL run and vice versa.

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

The CFL quickstart is an exploratory workflow check, not a validation of a
published CFL mass-radius sequence. Publication-level CFL stellar claims
require a reviewed strict run, convergence evidence, a convention-matched
published benchmark, and comparison with an independent solver.

## Notebook route

Notebook use is optional and follows the same reviewed-plan gate as the command
line. Do not use the checked-in `bsk24_experiment.ipynb` defaults as a first
run: they request 125 Cartesian geometries crossed with nine amplitudes under
the experimental, single-stage `dataset_40` stellar profile. Start with the
small JSON quickstart above; use this notebook section only after choosing and
reviewing an intentional campaign.

### General BSk24 campaign in VS Code

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

### General BSk24 campaign in JupyterLab

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

With the checked-in settings, the passive plan contains 1,125 logical rows
(125 geometries times nine amplitudes) and deduplicates the physical
zero-amplitude baseline. That preview is evidence of a large campaign, not an
invitation to execute it as a tutorial.

On a successful authorized pass, the notebook validates the completed
experiment, creates a separate `STUDENT_VIEW/` for primary CSV data and the
authoritative technical packet, and creates one flat experiment-level
`plots/` directory. Every applicable figure in that folder overlays all
accepted EoSs from the current sweep; rejected cases and historical
experiments are not mixed into the plots. The saved-table plot step makes zero
solver calls. The notebook uses `../runs/...` links because it is stored in
`notebooks/`; those links resolve correctly in both VS Code and browser
Jupyter. A stellar case remains student-view eligible when every explicitly
requested fixed mass succeeded even if maximum mass is explicitly unavailable.
The copied `case_ledger.csv` exposes the retained endpoint,
requested-fixed-mass, maximum-mass-availability, and student-view-eligibility
statuses separately.

Start with `STUDENT_VIEW/01_READ_ME_FIRST.md`. For portable EoS analysis,
keep `case_ledger.csv` and `thermodynamic_profiles.csv` from the same
`geometry_NNN` directory together. The ledger says which deformation a
`case_id` represents; the profiles hold its sampled data points. The complete
row, ordering, column, and machine-learning guidance is documented in
[`csv-data-guide.md`](csv-data-guide.md); packet validation and lifecycle
interpretation remain in [`results.md`](results.md).

For the focused BSk24 five-curve export, use
`notebooks/bsk24_dataset.ipynb`. It fixes the experimental
`dataset_40_curves` profile and intentionally omits fixed/max-mass refinement,
retained radial profiles, and duplicate presentation trees. It is a campaign
data route, not a replacement for quick/strict qualification; see
[`dataset.md`](dataset.md).

For pure bare CFL quark stars, open the separate notebook instead:

```powershell
jupyter lab notebooks/cfl_experiment.ipynb
```

The CFL notebook starts with three small signed amplitudes, one geometry,
`CALCULATION = "stellar"`, and `PRECISION = "quick"`. Its single editable
cell has the same deformation and precision controls as the hadronic route,
but no editable microphysical parameters or hadronic matching anchor. The
surface and `diagnostics = "off"` are fixed by the supported CFL contract.

The quick preview includes 51 sampled tidal models: three physical cases,
each at 17 central pressures. A=0 reuses the single analytic direct solution.
Fixed-mass roots and adaptive maximum-mass refinement add work. The strict
profile uses the governed three stellar stages; the preview itemizes its
larger cost. Switching precision requires a fresh passive pass.

After execution, combined M–R, Λ–M, k₂–M and thermodynamic plots appear
inline and in a separate flat `plots/` folder next to the
sealed experiment. Each graph overlays applicable accepted physical EoSs
from all geometries, with one baseline. Exact status and missing-data counts
are saved in the inventory. `FIXED_MASSES = [1.4]` only adds a comparison
point on the full sequence; you may choose other gravitational masses.

To reopen without scientific work, set `EXECUTE_REVIEWED_PLAN = False` and
paste the completed experiment path into `LOAD_EXPERIMENT`. The default
`BUILD_SAVED_PLOTS = False` makes this read-only. If the combined view is
missing, explicitly set that flag to `True` to create it from saved tables.
Existing views are verified, not overwritten. Archive the entire timestamped
run folder, including the independent view manifest.

For a large CFL dataset sweep, use the separate focused route:

```powershell
jupyter lab notebooks/cfl_dataset.ipynb
```

Its only supported precision is the experimental `dataset_40` profile. The
first Run All remains passive. The reviewed plan shows one 40-pressure stage
per physical EoS at `rtol=1e-10`, `atol=1e-12`, 1201 radial samples, all-node
tides, strict thermodynamic/raw gates, fixed-mass roots, and adaptive
maximum-mass refinement. Per-packet PNG groups are disabled. After successful
validation the notebook creates one `CFL_DATASET/` folder containing only two
labelled CSVs and exactly five combined plots. The labels are `cfl_0` for the
direct baseline and `cfl_1`, ... for accepted deformations. It skips
`STUDENT_VIEW`, `EOS_DATA`, the global catalogue, and copied technical tables;
reporting makes zero solver calls. Do not interpret the single-stage profile
as a STRICT convergence certificate.

## Next steps

- Copy [`../configs/custom_experiment.json`](../configs/custom_experiment.json)
  to vary amplitudes or geometry with the strict profile.
- Copy [`../configs/cfl_quickstart.json`](../configs/cfl_quickstart.json) to a
  new filename before defining a reviewed CFL geometry.
- Inspect [`../configs/stellar_example.json`](../configs/stellar_example.json)
  before requesting the more expensive stellar route.
- Read [`method.md`](method.md) for the construction and physical gates.

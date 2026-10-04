# EoS-generation

Controlled smooth sound-speed deformations of analytical BSk24 or BSk25, complete-domain
raw assessment, effective cold one-fluid reconstruction, and optional TOV/tidal
calculations. The numerical equations and governed profile values are retained
for BSk24 from v1.2.0; v2 replaces its aggregate/reporting workflow with one flat run.

Install the runtime defined by `environment.yml`, then the package:

```powershell
conda env create -f environment.yml
conda activate eos-generation
python -m pip install -e ".[notebook]"
```

Choose `matter_model="bsk24"` (default) or `"bsk25"` in the notebook controls.
Use `epsilon_match="standard"` when switching models to select its own anchor.
The BSk25 JSON example is `configs/bsk25.json`; the CLI remains `bsk24-trial`
for both models. [BSk25 sources and validation](docs/bsk25-validation.md)
documents the paper coefficients and the paper/Fortran discrepancy.

Edit `configs/quickstart.json` and review a passive plan:

```powershell
bsk24-trial plan --config configs/quickstart.json --output runs/my-study
```

Planning writes no result files and calls no scientific solver. Review the
destination, physical case count, expanded numerical stages, observable requests
and worker count. Copy the returned plan hash into the separate execution command:

```powershell
bsk24-trial run --config configs/quickstart.json --output runs/my-study --plan-hash <reviewed-hash> --execute
bsk24-trial validate runs/my-study
bsk24-trial status runs/my-study
bsk24-trial plot runs/my-study --figures pressure cs2
```

The destination must be a new path beneath a `runs` directory. Execution never
overwrites an existing run. Changes to settings, code, environment, worker budget
or destination require a fresh plan. Stellar calculations can be expensive;
planning reports the declared sequence attempts and identifies adaptive roots.

The same workflow is available in Python:

```python
from eos_generation import ExperimentSettings, plan_experiment, run_experiment
settings = ExperimentSettings.from_json("configs/quickstart.json")
plan = plan_experiment(settings, output_path="runs/my-study")
print(plan.summary_text())
# After reviewing this exact plan:
result = run_experiment(plan, execute=True)
```

`notebooks/bsk24_experiment.ipynb` has exactly two cells: a Markdown introduction
and one code cell. Edit the documented controls at the top of the code cell,
then run that cell with `ACTION = "plan"` to review the active settings, requested
products and cost. Change only `ACTION` to `"execute"` and rerun the same cell
to execute that reviewed plan. Planning always keeps
`EXECUTE_REVIEWED_PLAN = False`; execution requires the separate explicit action.
Changing settings requires another preview. Every preview chooses a new run
destination; repeating execution of the same completed plan loads its result.

Use `ACTION = "load"` after a restart to list and display saved results.
`SAVED_RUN = "latest"` selects the latest complete run, or supply a folder name.
Loading uses the saved settings even if the JSON configuration has since changed.
`PLOTS = "auto"` displays every available plot, with an availability report for
missing quantities. Use `"none"` to suppress figures or a list of readable names
such as `["Mass–radius", "Fixed-mass radius"]`. Geometry, amplitude, fixed-mass
and saved-stage filters affect only plotting. All parameter units and choices
are documented beside the editable controls. `NOTEBOOK_SETTINGS` is authoritative
by default; choose `SETTINGS_SOURCE = "json"` to use `CONFIG_FILE` instead.
The supplied Downloads copy starts with the user's existing JSON values copied
into the notebook dictionary; JSON files and saved results remain untouched.

Each figure combines the selected geometries, with a consistent color per
geometry and a black shared baseline. Line styles distinguish amplitudes or
response curves. The Λ sequence plot defaults to a logarithmic y axis; notebook
controls `LAMBDA_SCALE`, `LAMBDA_MASS_LIMITS` and `LAMBDA_LIMITS` also support a
focused linear view without another scientific run. Set `ACTION="load"` to
redraw saved results; existing figures and data remain intact.

Expandable guides compare all nine profiles and list all 21 plot types and 13
table choices, with their axes, meanings and prerequisites. `observables = None`
retains governed defaults: `dataset_40_curves` requests sequence only. Set an
explicit list to request fixed-mass or maximum-mass work; the enabled-product
table always shows the effective choice. `TABLES = "auto"` previews cases,
fixed/maximum-mass results and saved diagnostics; `"all"` also includes raw, EoS
and sequence tables. A list selects named tables; `TABLE_ROWS` limits displayed
rows and `table_frames` retains the full selected DataFrames. Plot filters do
not filter table previews. Loading displays validation, source/archive status,
diagnostic scope, and the saved values available for plot filters.

For stellar work, use `configs/stellar_example.json`; optional `observables`
separates sequence, fixed-mass and maximum-mass requests from precision. A figure
never starts a calculation or obtains quantities absent from a saved run.
On Windows, start Jupyter from the activated environment. The supplied Downloads
copy also supports the local `Python (Deformation_EoS)` kernel. A child-process
renderer check reports native-library failures before rendering in the kernel.

A thermodynamic run has five files under `data/`: `run.json`, `cases.csv`,
`raw.csv`, `eos.csv`, and `SHA256SUMS.txt`. Stellar observables add up to three
tables. Optional diagnostic tables are explicitly recorded. Figures and their
provenance live under `figures/` and never change the data manifest. Matching
verified figures are reused; different selections, regeneration, damaged images
or incomplete figure folders produce a fresh `figures-…` version. Source
archives are shared under `runs/_sources/` by content hash.

Intact results remain loadable after source changes; validation reports source
equivalence separately. `validate --exact-source` additionally requires the
current source inventory to match. Numerical correctness and observable
availability are separate: a retained endpoint can prevent a maximum-mass
resolution while valid fixed-mass results remain available.

V1 aggregate results can be imported without modifying or rerunning them:

```powershell
bsk24-trial import-legacy runs/old-experiment --output runs/imported-experiment
```

The importer verifies saved manifests and preserves historical evidence. It
requires complete-domain raw-gate v2 evidence; it fails closed for unsupported
historical capabilities and reports missing reconstruction certificates. Old
packets and the old implementation remain available in Git history.

See [method](docs/method.md), [data and migration](docs/data.md), and
[development](docs/developer.md). The effective barotrope does not establish
microscopic composition or beta equilibrium.

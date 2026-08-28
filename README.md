# EoS Generation
# Important Note: Currently attempting to speed up the notebooks for mass data generation. 
# Current state of the codebase isn't recommended for typical experimentation (26th August 2026)

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

## Run the experiment notebook

VS Code and JupyterLab are two interfaces to the same notebook and production
workflow. They use the same environment, calculations, validation, result
packets, and student view.

### Guide A — VS Code

1. Use **File > Open Folder** and select the repository folder itself: the
   folder containing `environment.yml`, `pyproject.toml`, `configs/`, and
   `notebooks/`. Do not open only its parent download directory.
2. In the Explorer, right-click that top-level folder and select **Open in
   Integrated Terminal**, then complete the installation block above one line
   at a time.
3. Open `notebooks/bsk24_experiment.ipynb`. Select **Python: Select
   Interpreter**, then **Select Kernel**, and choose `eos-generation` in both
   places. The selected interpreter must be Python 3.12 from that environment.
4. Leave `EXECUTE_REVIEWED_PLAN = False` and choose **Run All**. This first
   pass only prints the passive plan and writes no result.
5. Review the settings, work count, cases, and new destination. Change only
   `EXECUTE_REVIEWED_PLAN` to `True`, then choose **Run All** again in the
   same kernel. Restarting the kernel discards the reviewed authorization and
   requires another `False` preview pass.
6. Open the displayed **Student result locations** and start with **Read me
   first**.

### Guide B — JupyterLab in a browser

Open PowerShell at the repository root. Run these commands one line at a time;
they deliberately reuse the environment that worked in VS Code:

```powershell
conda activate eos-generation
python -c "import sys; print(sys.executable)"
python -m pip install -e . --no-deps
python -m pip install -e ".[notebook]"
python -m ipykernel install --user --name eos-generation --display-name "Python (eos-generation)"
python -m jupyterlab notebooks/bsk24_experiment.ipynb
```

The printed executable must identify the `eos-generation` environment. Keep
the launching terminal open while JupyterLab is running. In the browser:

1. Use **Kernel > Change Kernel** if necessary and select **Python
   (eos-generation)**.
2. Leave `EXECUTE_REVIEWED_PLAN = False` and choose **Run > Run All Cells**.
3. Review the passive plan, cases, work count, and new destination.
4. Change only `EXECUTE_REVIEWED_PLAN` to `True`, then choose **Run > Run All
   Cells** again without restarting or changing the kernel.
5. Open the displayed **Student result locations** and start with **Read me
   first**.

Do not launch this workflow from a base-Anaconda Jupyter shortcut or mix a
base Jupyter server with an `eos-generation` kernel. If activation is
uncertain, the explicit equivalent is:

```powershell
conda run -n eos-generation --no-capture-output python -m jupyterlab notebooks/bsk24_experiment.ipynb
```

After a successful run, use the notebook's **Student result locations** links.
Start with `STUDENT_VIEW/01_READ_ME_FIRST.md`; primary CSV data is below
`STUDENT_VIEW/03_PRIMARY_DATA/` and the sealed technical packet remains
separate. [`docs/csv-data-guide.md`](docs/csv-data-guide.md) explains exactly
which rows belong to each EoS, the saved column meanings, and safe preparation
for plotting or machine learning. After every successful notebook execution,
the same location list also includes one flat experiment-level `plots/`
folder. Every applicable standard figure combines all accepted EoSs from that
current sweep. Rejected cases are excluded, exact duplicate curves are drawn
once, and no per-case plot directories are created in this presentation
folder. The builder reads checksum-verified saved tables only, makes zero
solver calls, and never changes an authoritative packet. Separate packet-level
plots remain sealed technical evidence rather than the notebook's presentation
output.

Dataset notebook runs also create a flat `EOS_DATA/` folder with friendly
labels such as `H000001` for BSk24 and `C000001` for CFL, labelled primary
CSVs, a `matter_model` column, and an exact mapping back to every canonical
case ID. Numbering continues through the append-only shared
`runs/eos_catalogue/` registry. Each family begins with its baseline at
`H000000` or `C000000`; matching physical definitions reuse labels but keep
their evaluations separate. Archive the registry with your data and never
reset it. Existing packets and byte-for-byte `STUDENT_VIEW` copies are unchanged.
Small plot families show friendly legend labels; dense plots keep colour bars
and provide the ID mapping in `accepted_case_index.csv` and `case_aliases.csv`.

## Plan before calculating

For the experimental five-figure dataset workflow, use the separate
[`notebooks/bsk24_dataset.ipynb`](notebooks/bsk24_dataset.ipynb) or
[`notebooks/cfl_dataset.ipynb`](notebooks/cfl_dataset.ipynb), and read
[`docs/dataset.md`](docs/dataset.md). The focused dataset notebooks select `dataset_40`:
one 40-point stellar stage at rtol=1e-10 / atol=1e-12, with up to six case
workers. It retains STRICT thermodynamics and final integration tolerances,
but is not a per-case STRICT convergence certificate. The full multi-stage
`strict` profile remains unchanged. Restart the kernel and review a fresh
passive preview after updating; no execution is enabled by default.

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

Choose the notebook for your matter model:

```powershell
jupyter lab notebooks/bsk24_experiment.ipynb
jupyter lab notebooks/cfl_experiment.ipynb
jupyter lab notebooks/cfl_dataset.ipynb
```

Edit only the settings cell. Leave `EXECUTE_REVIEWED_PLAN = False`, run all
cells, and inspect the passive plan. Set it to `True` only after the plan and
destination are correct. The notebook delegates to the same production API as
the command line. The CFL notebook defaults to a small stellar experiment
with `PRECISION = "quick"`; change to `"strict"` and preview again for the
governed refinement stages. Its frozen microphysics and self-bound surface
are not editable sweep controls.

The CFL notebook automatically displays combined M–R, Λ–M, and k₂–M plots
from accepted saved sequences, plus thermodynamic curves and availability
tables. `FIXED_MASSES = [1.4]` adds a comparison point, not a restriction on
the full sequences. Reopen completed results with `LOAD_EXPERIMENT` and
execution disabled; loading and viewing existing plots are read-only. See
the [notebook quickstart](docs/quickstart.md#notebook-route).

For large CFL data collection, `cfl_dataset.ipynb` fixes the experimental
`dataset_40` profile, disables scientific packet PNG rendering, retains all
saved tables and statuses, and builds exactly five combined saved-data plots
plus `EOS_DATA/` after successful validation. It is passive by default and is
not a substitute for a matched STRICT convergence study.

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
| [`configs/`](configs/) | Four user configurations and their schema |
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

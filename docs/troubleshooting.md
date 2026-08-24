# Troubleshooting

## The terminal opened in the wrong folder

The terminal must be at the repository root: the folder containing
`environment.yml`, `pyproject.toml`, `configs/`, and `notebooks/`. In VS Code,
right-click that top-level folder in the Explorer and choose **Open in
Integrated Terminal**. Then check:

```powershell
Test-Path .\environment.yml
Test-Path .\pyproject.toml
Test-Path .\notebooks\bsk24_experiment.ipynb
```

All three results must be `True`. If they are not, the wrong folder was opened
in VS Code or the terminal belongs to its parent directory.

Do not type documentation placeholders such as `<path>` or the angle brackets
themselves into `cd`. To use Anaconda Prompt without manually typing a path,
type `cd /d ` including the trailing space, drag the repository folder from
File Explorer onto the prompt, and press Enter.

## `conda` is not recognized

This is not caused by the location of the notebook. It means that the current
terminal does not know where Conda is installed.

On Windows, open **Anaconda Prompt** or **Miniconda Prompt** from the Start
menu. Move to the repository root using the drag-and-drop method above, then
run:

```text
conda env create -f environment.yml
conda activate eos-generation
python -m pip install -e . --no-deps
python -m pip install -e ".[notebook]"
```

If `conda env create` reports that `eos-generation` already exists, keep that
environment, run `conda activate eos-generation`, and continue with the two
install commands. Return to VS Code and select it through the interpreter and
notebook-kernel menus.

## The `eos-generation` notebook kernel is missing

First install the notebook tools in the declared environment. If the kernel
still does not appear, register that interpreter explicitly:

```powershell
conda activate eos-generation
python -m ipykernel install --user --name eos-generation --display-name "Python (eos-generation)"
```

Then reload the VS Code window or refresh Jupyter's kernel list and choose
**Python (eos-generation)**. Verify the interpreter in a temporary notebook
cell if necessary:

```python
import sys
print(sys.executable)
```

The path should name the `eos-generation` environment and the interpreter
must be Python 3.12. Selecting `base`, a system Python, or an old environment
can produce missing-package errors or inconsistent worker behavior.

VS Code and Jupyter Lab are only different notebook interfaces. They run the
same project correctly when they use the same kernel.

## `bsk24-trial` is not found

Activate the declared environment and install the checkout:

```powershell
conda activate eos-generation
python -m pip install -e . --no-deps
bsk24-trial --help
```

Check which interpreter is active:

```powershell
python -c "import sys, eos_generation; print(sys.executable); print(eos_generation.__file__)"
```

The supported Python range and pinned scientific dependencies are defined in
`pyproject.toml` and `environment.yml`.

## Jupyter reports an environment mismatch or `BrokenProcessPool`

Do not run a base-Anaconda Jupyter server with an `eos-generation` kernel, or
the reverse. Native numerical libraries and spawned stellar workers inherit
the server environment on Windows; mixing environments can terminate a worker
without a Python exception. The notebook now rejects a detected mismatch
before planning or calculation.

Activate and launch through the governed interpreter:

```powershell
conda activate eos-generation
python -m jupyterlab notebooks/bsk24_experiment.ipynb
```

If shell activation is uncertain, use:

```powershell
conda run -n eos-generation --no-capture-output python -m jupyterlab notebooks/bsk24_experiment.ipynb
```

For VS Code, select `eos-generation` with **Python: Select Interpreter** and
again with the notebook's **Select Kernel** menu. Check the live kernel before
execution:

```python
import os
import sys

print(sys.executable)
print(sys.prefix)
print(os.environ.get("CONDA_PREFIX"))
```

The executable and prefix must identify `envs\eos-generation`. A missing
`CONDA_PREFIX` is acceptable when VS Code launches the interpreter directly;
if present, it must identify the same environment. Shut down old notebook
kernels after changing the selection.

After correcting the environment, restart the kernel. A restart discards the
reviewed notebook authorization, so run all once with
`EXECUTE_REVIEWED_PLAN = False`, review the new plan, then change only that
flag to `True` and run all again. Confirm that the new passive plan selects a
new destination; never reuse an incomplete packet as a write target.

If the same plan succeeds in VS Code with the declared kernel but fails in a
browser notebook, use the working VS Code route and report the failing
kernel's `sys.executable`, configuration hash, and traceback. Do not change
scientific settings or tolerances merely to hide a worker crash.

## The configuration is rejected

Validate the JSON syntax, then compare the fields with
[`../configs/schema.json`](../configs/schema.json) and
[`parameters.md`](parameters.md). Common causes are:

- an unknown or misspelled field;
- a nonpositive width or ramp width;
- a nonpositive center;
- diagnostics `on` with a thermodynamics-only calculation;
- a numeric anchor outside its allowed homogeneous-core interval;
- `NaN` or infinity, which JSON settings do not permit.

Use `bsk24-trial plan` after every edit. Do not bypass validation by changing
implementation constants.

## Planning created files or started a long calculation

Stop and report this as a bug. `plan_experiment` and `bsk24-trial plan` must
make zero scientific solver calls and zero filesystem writes.

Only a hash-bound `bsk24-trial run ... --plan-hash <reviewed-hash> --execute`
is authorized to calculate and create a packet. If either gate is missing,
make a fresh passive plan and copy its hash; do not guess or reuse a hash from
different settings or a different destination.

## The destination already exists

Results are fail-closed against accidental overwrite. Choose a new,
descriptive path below `runs/`, or inspect and archive the existing result
before deciding what to do with it. Do not delete an uncertain destination.

## A proposal was rejected

This often reflects a real raw-domain violation rather than a software
failure. Inspect the exact reason and raw gate table. The workflow rejects
before reconstruction when, for example, `dP/dε` becomes nonpositive or
exceeds the causal bound.

Do not clip the failed points, discard a small violating region, or infer that
an isolated feature is harmless without a scientific analysis of its origin
and convergence.

## Stellar output is missing

Check, in order:

1. `calculation` was `"stellar"`;
2. the raw proposal was accepted;
3. the reconstructed barotrope covered the required domain;
4. the background solver completed the successful stable prefix;
5. a requested fixed mass was truly bracketed;
6. the tidal capability status was valid.

Unavailable output must have an explicit reason. It should not be replaced by
zero or extrapolated past the supported branch.

## The mass-radius curve does not reach zero mass

A neutron-star sequence is sampled over a governed central-pressure interval;
it is not created by forcing the curve through the plot origin. The first
successful model may therefore have a positive mass. Changing an axis limit
does not create missing physical models.

Inspect the central-pressure lower bound, finite-domain surface convention,
and successful sequence prefix before deciding whether a separate numerical
study is warranted.

## Maximum mass is unresolved

The largest sampled model is not enough. Resolution requires a bracketed and
refined turning point on the successful branch. If the valid domain or solver
ends first, the packet must say that the maximum is unresolved.

A denser run is useful only when the current result indicates inadequate
sampling inside an otherwise valid bracket. It cannot repair domain
truncation or an unphysical proposal.

## Validation reports source drift

Validation compares the saved source inventory with the current installed
code. The execution environment and dependency versions are recorded as
provenance, but packet validation does not claim cross-platform runtime
equivalence. Reproduce in two steps: run the recorded passive plan command,
then use that plan's hash in the recorded `run --plan-hash ... --execute`
command. Use the declared dependency versions. Do not edit a completed packet
in place to make hashes agree.

## A plot looks empty or has crossed-out points

Read the plot inventory and the corresponding saved status table. Marked or
omitted points can represent rejected proposals, unavailable stellar
capabilities, unbracketed masses, or failed convergence. Confirm which status
applies before changing plot limits or rerunning.

If the table is valid but the rendering is unclear, report the packet path,
plot filename, configuration hash, and validation output.

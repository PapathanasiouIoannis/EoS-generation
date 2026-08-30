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

## Windows exits inside a NumPy operation

Activate the declared Conda environment before launching Python or Jupyter.
Calling another environment's `python.exe` directly while the base
environment's DLL paths are active can load incompatible native libraries.
Do not change numerical algorithms or dependency pins to work around that
launch error. Preserve any incomplete packet, correct environment activation,
and preview a new destination. A packet without completed aggregate metadata
is not a successful experiment.

## The configuration is rejected

Validate the JSON syntax, then compare the fields with
[`../configs/schema.json`](../configs/schema.json) and
[`parameters.md`](parameters.md). Common causes are:

- an unknown or misspelled field;
- `matter_model = "cfl"` without `epsilon_match = "surface"`, or a BSk24
  configuration using the CFL-only surface spelling;
- a nonpositive width or ramp width;
- a nonpositive center;
- a center/width pair whose nominal four-sigma support has no nonzero overlap
  with the deformable domain above the selected anchor;
- diagnostics `on` with a thermodynamics-only calculation;
- diagnostics `on` for CFL, whose extended radial capability is unsupported;
- a numeric anchor outside its allowed homogeneous-core interval;
- a CFL center outside its formula-derived energy-density domain, or a ramp
  whose surface-plus-width exceeds the upper endpoint;
- `NaN` or infinity, which JSON settings do not permit.

Use `bsk24-trial plan` after every edit. Do not bypass validation by changing
implementation constants.

The frozen CFL energy-density endpoints are
`190.2181760065314` and `4008.81724402691 MeV fm^-3`. Similar-looking
rounded design-review values are display references, not replacement
endpoints for equality or identity checks. See [`cfl.md`](cfl.md).

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

This often reflects a real raw-domain violation or an unresolved continuous
assessment rather than a software failure. Inspect the exact reason, complete
raw gate table, continuous-extremum evidence, and resolution certificate. The
workflow rejects before reconstruction when, for example, `dP/dε` becomes
nonpositive or a narrow feature cannot be resolved reliably.

Reaching `c_s^2 = 1` before direct BSk24 is not by itself a rejection. For an
otherwise valid proposal, the first continuously resolved crossing is included
as the case-specific endpoint. Raw values after that crossing remain saved
evidence, but even a later return below one does not reopen the usable branch.

Do not clip the failed points, discard a small violating region, or infer that
an isolated feature is harmless without a scientific analysis of its origin
and convergence.

For CFL, the raw gate covers the full frozen domain. A continuous causal
crossing or failure near the upper endpoint rejects the whole proposal; it is
not acceptable to shorten the EoS to the last passing grid point.

## Stellar output is missing

Check, in order:

1. `calculation` was `"stellar"`;
2. the raw proposal was accepted;
3. the reconstructed barotrope covered the required domain;
4. the background solver completed the successful stable prefix;
5. a requested fixed mass was truly bracketed at a central pressure inside the
   retained case-specific EoS endpoint;
6. the tidal capability status was valid.

Unavailable output must have an explicit reason. It should not be replaced by
zero or extrapolated past the supported branch.

For a CFL tidal result, also inspect the surface-jump evidence. The bare star
has finite energy density just inside `P = 0` and vacuum outside, so the
outward `y` correction must be negative and applied exactly once before
`k2`. A missing or duplicate jump correctly makes the capability unavailable.
CFL has no crust or hadronic envelope that can be substituted for this
surface rule.

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

Maximum-mass unavailability does not erase independently valid fixed-mass
results. Check the fixed-mass status at the final reporting stage and the
separate `maximum_mass_availability_status`. If every explicitly requested
fixed mass succeeded, `student_view_eligibility_status` can remain eligible
even though the turning point is unresolved.

## STUDENT_VIEW finalization reports `PermissionError` on Windows

VS Code, File Explorer, antivirus software, or an indexing service can briefly
hold a sharing handle on the staging directory. Publication uses a bounded
retry schedule for this transient Windows condition, keeps the rename on the
same volume, and rechecks no-overwrite before every attempt.

If all retries fail, the unpublished staging directory is cleaned up and the
already sealed authoritative experiment remains unchanged. Close previews or
Explorer windows holding the result directory and retry only the student-view
creation from that validated packet. Do not delete or edit the authoritative
packet, and do not reuse an existing `STUDENT_VIEW/` destination.

## Friendly EoS labels or EOS_DATA publication failed

The scientific experiment is already complete and validated before this
presentation step. Preserve it; do not execute the scientific settings again
just to recover labels. Rebuild the labelled view from that saved experiment
using the reporting-only adapter and a new, unoccupied destination:

```powershell
python notebooks/eos_catalogue.py --repository-root . --experiment runs/PATH_TO_COMPLETED_EXPERIMENT --destination runs/PATH_TO_RUN/EOS_DATA_retry
python notebooks/build_experiment_plots.py --repository-root . --experiment runs/PATH_TO_COMPLETED_EXPERIMENT --destination runs/PATH_TO_RUN/plots_retry --eos-data runs/PATH_TO_RUN/EOS_DATA_retry
```

For either focused dataset notebook, use the five-figure dataset adapter for
the second command instead:

```powershell
python notebooks/build_dataset_plots.py --repository-root . --experiment runs/PATH_TO_COMPLETED_EXPERIMENT --destination runs/PATH_TO_RUN/plots_retry --eos-data runs/PATH_TO_RUN/EOS_DATA_retry
```

Replace the paths with actual locations; neither command calls solvers. They
never overwrite an existing destination. A busy registry fails after a bounded
wait, and the OS releases its lock when a process exits. Retry only the reporting
operation after the competing writer finishes. A checksum/identity failure is
not transient: stop and investigate it. Never edit registrations, delete the
registry to reset numbering, or reassign an existing label. IDs allocated before
an interrupted presentation export are intentionally retained for reuse.

For `cfl_dataset.ipynb`, the science result is stored in memory before the
minimal presentation begins. Rerun only the execution/reporting cell in the
same kernel: it recognizes the consumed authorization token, reuses the
completed result, and rebuilds or reopens `CFL_DATASET/` without solver calls.
After a kernel restart, load the completed experiment and call
`eos_generation.reporting.cfl_dataset.build_cfl_dataset_output` with a new
unoccupied destination. The adapter publishes atomically, so a failed build
leaves no partial final directory.

## Validation reports source drift

Validation compares the saved source inventory with the current installed
code. The execution environment and dependency versions are recorded as
provenance, but packet validation does not claim cross-platform runtime
equivalence. Reproduce in two steps: run the recorded passive plan command,
then use that plan's hash in the recorded `run --plan-hash ... --execute`
command. Use the declared dependency versions. Do not edit a completed packet
in place to make hashes agree.

## A plot looks empty or has crossed-out points

For the CFL notebook, `LOAD_EXPERIMENT` with execution disabled reopens saved
results without calculations. If only the combined view is missing, explicitly
set `BUILD_SAVED_PLOTS = True`; the view is built from validated tables and
cannot rerun science. A damaged existing view fails hash validation and is
never silently overwritten. Preserve it for diagnosis and rebuild in a
separately archived copy rather than editing a scientific packet.

Read the plot inventory and the corresponding saved status table. Marked or
omitted points can represent rejected proposals, unavailable stellar
capabilities, unbracketed masses, or failed convergence. Confirm which status
applies before changing plot limits or rerunning.

If the table is valid but the rendering is unclear, report the packet path,
plot filename, configuration hash, and validation output.

## A plan shows several zero-amplitude rows

The owner's A=0 stellar case reuses the governed `direct` solution. It is
counted once in the work plan and drawn once in the notebook view, with the
physical baseline ID linked to that saved direct row.

The rows are logical identity controls for each requested geometry, not
repeated baseline calculations. Exactly one lexicographically first geometry
owns the physical `A = 0` case; the other zero rows should be marked as
nonexecuting aliases to its physical case ID. The physical work estimate must
count the baseline once. If the executor schedules every alias, stop and
report that as a deduplication bug.

## A quick CFL result disagrees with a published curve

First confirm that the publication uses the same full finite-`m_s` equation
set, `B`, gap, strange mass, surface convention, and no crust. Truncated bag
forms and a quoted `B` in `MeV fm^-3` are not interchangeable with
the current authoritative `B = 57.5 MeV fm^-3`; its derived fourth root is
`144.97957215191494 MeV`.

Even with matching conventions, the quick profile is exploratory. Do not tune
settings or tolerances until the result agrees visually. Publication-level
claims require a reviewed strict convergence study, a convention-matched
published benchmark, and an independent solver comparison; an expected table
generated by this implementation is not independent validation.

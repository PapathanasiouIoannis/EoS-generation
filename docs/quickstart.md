# Quickstart

This guide installs EoS Generation from a source checkout, creates a passive
plan, explicitly executes a small thermodynamic example, and inspects the
saved result. The command name is `bsk24-trial`.

Planning is always passive. Only a hash-bound `run` command with `--execute`
may start scientific work or create a result.

## 1. Open the repository root

Clone the repository or unpack the
[v1.2.0 source release](https://github.com/PapathanasiouIoannis/EoS-generation/releases/tag/v1.2.0).
Open a terminal in the directory containing these paths:

```text
environment.yml
pyproject.toml
configs/
notebooks/
```

On PowerShell, verify the location without using a machine-specific path:

```powershell
Test-Path .\environment.yml
Test-Path .\pyproject.toml
Test-Path .\configs\quickstart.json
Test-Path .\notebooks\bsk24_experiment.ipynb
```

All four commands must print `True`.

## 2. Create and install the environment

The supported scientific runtime is defined by `environment.yml` and
`pyproject.toml`:

```powershell
conda env create -f environment.yml
conda activate eos-generation
python -m pip install -e . --no-deps
```

If the `eos-generation` environment already exists, activate it and install
the current checkout; do not create a duplicate environment. For notebook use,
also install the optional interface dependencies:

```powershell
python -m pip install -e ".[notebook]"
```

Check the active interpreter, package, and CLI:

```powershell
python -c "import sys, eos_generation; print(sys.executable); print(eos_generation.__version__)"
bsk24-trial --help
```

The version must be `1.2.0`, and the executable should belong to the
`eos-generation` environment. If Conda, the package, command, or notebook
kernel is missing, see [Troubleshooting](troubleshooting.md).

## 3. Inspect a small configuration

[`../configs/quickstart.json`](../configs/quickstart.json) requests two BSk24
amplitudes at one geometry with the `quick`, thermodynamics-only profile. It
does not request a stellar calculation.

Field meanings, units, and relationships are documented in
[Parameters](parameters.md).

## 4. Make a passive plan

```powershell
bsk24-trial plan --config configs/quickstart.json --output-root runs
```

The plan resolves defaults, expands the named numerical profile, lists the
logical and physical cases, estimates fixed work, shows the destination, and
prints a plan hash. It makes zero solver calls and zero filesystem writes.

When several geometries are requested, every geometry has a logical
zero-amplitude identity row, but only the numerically lexicographically
smallest `(center, width, ramp_width)` tuple owns the physical baseline
calculation. Geometry folder numbers still follow declared Cartesian order,
so the owner need not be `geometry_001`. The plan reports nonexecuting aliases
and counts that baseline once.

If any setting, source file, environment identity, worker budget, output root,
or plan changes, preview again. Never guess a hash or copy one from a different
configuration.

## 5. Execute exactly the reviewed plan

Replace `REVIEWED_PLAN_HASH` with the complete hash printed by the matching
plan:

```powershell
bsk24-trial run --config configs/quickstart.json --output-root runs --plan-hash REVIEWED_PLAN_HASH --execute
```

The CLI rejects a missing execution flag, a missing or stale hash, settings drift,
and an existing destination.

Execution writes a deterministic directory such as
`runs/experiment_0123456789ab`. Do not type that example literally; use the
path printed by your run. The quickstart stops after thermodynamic gating and
reconstruction. It does not call the stellar solver.

Rejected and unresolved proposals are recorded outcomes. They retain raw gate
evidence and a reason, but receive no reconstruction or stellar work. An
otherwise valid proposal may retain a certified prefix through its first
continuous `c_s^2 = 1` crossing.

## 6. Validate, summarize, and inspect plots

Use the actual path printed by execution:

```powershell
bsk24-trial validate runs/experiment_0123456789ab
bsk24-trial status runs/experiment_0123456789ab
bsk24-trial plot runs/experiment_0123456789ab
```

`validate` is the diagnostic entry point. It reads the packet without changing
it and checks aggregate/child structure, manifests, source and configuration
identity, hard scientific validity, and observable availability. Use `--json`
for the complete report.

`status` strictly loads a completed, currently valid experiment and prints a
compact summary. If validation fails or the aggregate is incomplete, diagnose
with `validate` rather than expecting `status` to load it.

`plot` without `--overwrite` inventories existing saved plots. To regenerate
applicable packet figures from validated saved tables, explicitly authorize
the file replacement:

```powershell
bsk24-trial plot runs/experiment_0123456789ab --overwrite
```

Plot regeneration does not call thermodynamic, TOV, tidal, or maximum-mass
solvers. It can replace figure files and reseal their manifests.

Read [Results](results.md) before interpreting lifecycle or availability
statuses, and use the [CSV data guide](csv-data-guide.md) for table schemas.

## Notebook route

Notebook use is optional. Install the notebook extra, activate the same Conda
environment, and launch from the repository root:

```powershell
python -m jupyterlab notebooks/bsk24_experiment.ipynb
```

In VS Code, select the same `eos-generation` interpreter as both the Python
interpreter and notebook kernel. In either interface, the safe procedure is:

1. Open the intended notebook and edit only its settings cell.
2. Keep `EXECUTE_REVIEWED_PLAN = False` and run all cells.
3. Review the model, settings, logical/physical case counts, expanded profile,
   solver targets, worker budget, and new destination.
4. Change only `EXECUTE_REVIEWED_PLAN` to `True`.
5. Run all cells again in the same kernel.

The first pass makes zero solver calls and writes nothing. Authorization is
bound to the plan retained in that kernel. Restarting the kernel or changing a
governed input requires another disabled preview.

### Choose the notebook deliberately

| Notebook | Current checked-in settings | Output presentation |
|---|---|---|
| [`bsk24_experiment.ipynb`](../notebooks/bsk24_experiment.ipynb) | 125 geometries x 9 amplitudes, stellar `dataset_40`; 1,125 logical cases | `STUDENT_VIEW/`, persistent-label `EOS_DATA/`, combined `plots/` |
| [`bsk24_dataset.ipynb`](../notebooks/bsk24_dataset.ipynb) | Large negative-amplitude BSk24 campaign, stellar `dataset_40_curves` | Exactly five combined plots; no `STUDENT_VIEW/` or `EOS_DATA/` |

The BSk24 notebooks are campaign-scale and are not onboarding examples. Use
the JSON quickstart first. The focused dataset profiles are experimental and
are not substitutes for the multi-stage `strict` profile.

## Next steps

- Copy [`../configs/custom_experiment.json`](../configs/custom_experiment.json)
  to define a strict BSk24 thermodynamic study.
- Inspect [`../configs/stellar_example.json`](../configs/stellar_example.json)
  before requesting the more expensive stellar layer.
- Read [Method](method.md) for equations, units, and causal gates.
- Read [Dataset workflows](dataset.md) before using any dataset-family profile
  or campaign helper.

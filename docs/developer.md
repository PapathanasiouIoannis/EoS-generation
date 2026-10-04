# Development

The package has fourteen modules with one workflow: `settings` normalizes public
choices; `numerics` governs constants/profiles; `baseline`, `deformation`,
`assessment` and `thermodynamics` implement the cold EoS; `tov` contains coupled
background/tidal integration and discontinuities; `stellar` resolves sequences,
stable-prefix fixed masses and maximum mass; `diagnostics` computes explicit
scientific evidence; `experiment` plans/executes; `storage` writes, loads,
validates and imports historical data; `plotting` reads saved tables; `cli` adapts
the API; `__init__` preserves public identity.

Planning must write nothing and perform no scientific calculation. Execution
rechecks its source/environment/settings/destination/worker binding, exclusively
creates the destination, builds baselines once per stage for the whole sweep,
persists complete raw evidence before reconstruction and sends only accepted
retained barotropes to stellar work. Rejected and unresolved cases keep their
values and exact reasons. Exceptions seal a failed record and propagate.
Each data write uses a same-directory temporary file and atomic replacement;
existing run/figure destinations are never overwritten.

Notebook actions are plan, execute and load. Unique notebook destinations are
chosen passively; repeat execution validates and loads a matching complete run.
Saved-run listing and plot availability perform no solver calls or writes.
Plotting checks scientific capability statuses before selecting valid rows,
probes the renderer in a child process, and reuses only matching manifest,
plotter and image hashes. Incomplete or damaged figures require a fresh version.
The notebook contains exactly one Markdown cell and one code cell. Discovery
tables derive numerical counts from the governed profiles and plot choices from
the saved-plot registry. The editable dictionary is authoritative by default;
JSON is an explicit alternative. Presentation helpers remain lazy and scientific
execution still needs a reviewed plan. Tests exercise repeated runs of the same
cell, restart loading, unreviewed refusal and passive execution from both working
directories. Saved diagnostic previews perform only CSV reads.

Scientific equations, coefficients, constants, causal root policies, pressure
floors, stable-branch brackets, surface/jump corrections and acceptance predicates
are unchanged from v1.2.0. The immutable fixture files remain byte-identical.
Scientific tests moved with their defining routines; workflow/reporting tests now
exercise v2. Never regenerate reference fixtures or weaken a scientific tolerance.

Use the narrowest regression while editing, then run:

```text
python -m pytest -q
python -m build --wheel
```

CI installs the built wheel and runs tests from outside the checkout. It also
checks passive notebook execution from both working directories, passive planning,
Git-free source archives, package-data inclusion and repository hygiene. Hygiene
rejects results, caches and large generated files without an exact source-file
whitelist. `.github/workflows/ci.yml` is the authority. Routine verification uses
compact scientific regressions and synthetic stellar orchestration; do not launch
large stellar campaigns.

Source identity includes every active package module and the source manifest;
runtime/package contracts, build README and license are included. Installed wheels
package these files too, so a restored source archive can build an equivalent
wheel. A plan's hash changes after
any source change. Result loading distinguishes scientific integrity from current
source equivalence. Historical import is an interpretation of saved evidence,
never a scientific recalculation.

The default bounded case-worker policy remains six, half logical CPU count and
case count, with deterministic parent merge and no nested process pools. Preserve
this reviewed budget in every interface. Timing/PID details are development
metrics, not independent numerical evidence.

Keep generated runs and source archives ignored. Work on one focused branch from
current main, preserve unrelated edits, inspect the full diff, and do not publish
without the user's explicit instruction.

## Selecting a baseline

`baseline_definition` selects source-pinned coefficients, domains and phase
metadata without evaluations. `make_baseline_eos` instantiates that selection
only during explicit execution. Every anchor, grid, raw gate, reconstruction and
diagnostic uses that same model. BSk24 literal constants and arithmetic are
preserved. BSk25 follows paper C1/C4 with its documented discrepancies; its
compact reference generator has no production imports. Never substitute
CompOSE interpolation or silently move an invalid anchor.

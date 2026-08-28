# Focused dataset pilot

Open `notebooks/bsk24_dataset.ipynb` with the `eos-generation` kernel. Restart
the kernel after updating package source. Keep `EXECUTE_REVIEWED_PLAN=False`
for the first Run All, review the exact work/destination, then change only the
flag to True. The notebook delegates to the existing public experiment API.
Pure self-bound CFL data collection uses the parallel
`notebooks/cfl_dataset.ipynb` route; it fixes the CFL surface anchor and
microphysics while exposing the same deformation axes.

The focused dataset notebooks use `precision="dataset_40"`: one
40-pressure sequence at `rtol=1e-10`, `atol=1e-12`, with tides at every point.
The automatic case-worker cap is six (fewer for small batches or lower-core
machines), and nested pools are disabled. The preview displays the worker
budget. This is the governed tight single-stage configuration, not a renamed
full STRICT calculation. The existing BSk24 route has its prior evidence; the
new CFL route remains pending the deferred checks described below. Existing
`quick`, `strict`, and dataset-family
profile definitions are unchanged. Any saved notebook outputs are historical
until refreshed; existing results and parameter grids are preserved.

For CFL, `dataset_40` is the only enabled dataset-family profile. It uses the
same numerical expansion but remains explicitly experimental until a matched
CFL STRICT comparison is authorized. CFL child packets select the `none` plot
group, so no per-case or per-geometry scientific PNGs are rendered. All
thermodynamic, sequence, fixed-mass, maximum-mass, rejection, surface-jump,
provenance, and manifest data remain authoritative and complete.

## Numerical contract

`precision="dataset"` is a separate **experimental** profile, not a renamed
STRICT calculation or a certificate of ML suitability. It requires
`calculation="stellar"` and `diagnostics="off"`. QUICK and STRICT expansions
are unchanged. Dataset retains all three STRICT thermodynamic stages and
raw-domain certification grids; EoS equations, gates, units, causal endpoints,
TOV/tidal equations, surface conventions and failure semantics are unchanged.

It calculates one 61-pressure stellar sequence at `rtol=1e-10`, `atol=1e-12`,
with 1201 radial-profile samples (STRICT final-stage settings). It retains
fixed-mass roots and the original 17-point-initial maximum-mass refinement.
This replaces STRICT's 61/121/121 sequence repetitions, but does not weaken
solver tolerances or pretend that sampled maxima are resolved M_max.
The saved stellar status explicitly says `single_stage_no_numerical_envelope`.
Finite thermodynamic residual warnings also remain visible.

Every sampled star still receives a tidal solve. Selective tidal sampling and
the professor's ten aligned tuples are deferred pending separate validation.
This first candidate tests reduced stellar repetition without changing those
algorithms. A 61-point curve may undersample a feature even if its fixed-mass
root is accurate; comparisons must include between-node curve errors and
maximum-mass status changes, not only R(1.4).

## Output

### Explicit 10-point candidate at tighter tolerances

`precision="dataset_10_tighter"` uses one 10-pressure stellar stage at
rtol=1e-11, atol=1e-13 and 1201 radial samples, with tides at every point.
It changes both sequence sampling and ODE tolerances relative to `dataset`;
comparisons cannot attribute their combined effect to either change alone.
All thermodynamics/raw certification, pressure domain, surface convention,
fixed-mass root settings and maximum-mass refinement rules remain unchanged.
Ten sequence nodes do not mean ten total solver calls: fixed-mass and adaptive
maximum-mass searches add work. Sparse sampling is not compensated or certified
by tighter ODE tolerances. This single-stage pilot has no per-case stellar
refinement envelope and is not STRICT certification. Existing profiles remain
unchanged; this is an optional alternative to the `dataset_40` notebook default.

### Explicit sampling-only candidate at tight tolerances

`precision="dataset_20"` is the 20-point sampling-only candidate. Relative to
`dataset`, only the stage label and sequence count change. It retains
rtol=1e-10, atol=1e-12, 1201 radial samples, all-node tides, STRICT
thermodynamics/raw certification, fixed-mass roots and maximum-mass rules.
It is not full STRICT certification. Sparse curves can miss between-node
features despite tight integration; compare saved reference curves, not just
fixed-mass roots. Existing profiles are unchanged.

`precision="dataset_40"` changes only the dataset stellar stage name and
sequence count to 40 logarithmically spaced central pressures. It retains
rtol=1e-10, atol=1e-12, 1201 radial samples, tides at all 40 points, all STRICT
thermodynamic stages/raw certification, fixed-mass roots and maximum-mass rules.
It is single-stage experimental evidence, not full STRICT certification.
This is now the default in both notebooks. Its pressure grid is not nested in the
61-point grid; sparse-curve interpolation and turning-point availability need
separate review. Tight integration alone does not certify curve sampling.

### Explicit tolerance-only candidate

`precision="dataset_relaxed"` is a separately identified experimental variant
for the same notebook. It changes only the stellar stage name and integration
tolerances to `rtol=1e-8`, `atol=1e-10`. The 61 pressure nodes, 1201 radial
samples, all-node tides, thermodynamic stages/certification, fixed-mass root
settings and maximum-mass refinement rules remain those of `dataset`.
It is not STRICT certification and has no per-case refinement envelope.
The notebook default is `dataset_40`; changing the variant requires a fresh
passive preview. Validate timing and all required observables on matched
geometries before deciding whether this candidate suits an ML accuracy budget.
No existing QUICK, STRICT or dataset profile is redefined by this addition.

`precision="dataset_relaxed_80"` is the separately named 80-point variant.
Relative to `dataset_relaxed`, only the stellar stage label and sequence count
change: 80 logarithmically spaced central pressures instead of 61, with tides
at every point. It retains rtol=1e-8, atol=1e-10 and 1201 radial-profile samples.
This alternative does not redefine existing profiles. The two grids are not nested;
curve comparisons must distinguish directly matched pressure nodes from
interpolation. More samples do not tighten the integration of an individual
star or guarantee a resolved maximum mass. All unavailable statuses remain.

The authoritative experiment preserves all mandatory evidence, statuses,
configuration/source hashes and manifests. Technical packet plot inventories
record every unused plot as skipped via the explicit `none` group. No extra
technical PNGs are rendered. Notebook `STUDENT_VIEW` copies remain available.
Skipped response figures still record mandatory final-stage population and
tidal-completeness metadata. The unchanged validator checks this evidence even
when a figure is not rendered.

The notebook creates persistent model-specific labels in `EOS_DATA` (`H` for
BSk24 and `C` for CFL), retains a `matter_model` column, and creates exactly five PNGs
in one sibling `plots` folder: `eos_pressure.png`, `speed_of_sound.png`,
`mass_radius.png`, `k2_mass.png`, and `lambda_mass.png`. These overlay accepted
EoSs from the current experiment only, deduplicate the baseline, preserve
failed-attempt gaps and reject invalid tidal values. The plotted stellar
prefix ends at the sampled peak; this is not an independent stability proof.
The pressure-energy-density figure uses linear energy-density and pressure axes.
Raw rejected/superluminal evidence remains in packets, not accepted EoS plots.
Rendering reads checksum-verified saved tables and never invokes solvers.

Changing source necessarily changes source-bound plan identities. Never reuse
a pre-change reviewed plan or rewrite an old packet's hashes. Historical
packets require their archived matching source for source-equivalent validation.
Physical H labels still use exact EoS definitions, not the numerical profile.

## Qualification boundary

Do not assume suitability for a 2,000-EoS campaign from the profile name.
Benchmark against archived STRICT evidence over the intended sign/geometry
domain, quantify errors for all requested observables, retain unavailable
statuses, and agree an accuracy budget before final ML export. Where only
QUICK references exist, comparison is exploratory and cannot certify STRICT
accuracy. Runtime savings must be measured on matched cases; no universal
speedup factor is promised.

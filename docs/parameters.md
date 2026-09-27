# Parameters

Public JSON configurations are described by
[`../configs/schema.json`](../configs/schema.json). The schema supports editors
and CI validation; runtime constructors remain authoritative for finite-value,
cross-field, BSk24-domain, and output-path checks.

Every public JSON file must include a nonempty `$schema` annotation. That
annotation is removed before scientific settings are normalized and hashed.

## Public fields

| Field | Meaning | Allowed values and units |
|---|---|---|
| `matter_model` | Governed matter baseline | Omit for canonical BSk24 settings, or set explicitly to `"bsk24"`; canonical serialization omits it. |
| `amplitudes` | Additive coefficients multiplying the windowed Gaussian contribution to `c_s^2` | Nonempty array of distinct finite dimensionless numbers. If zero is absent, a logical identity control is inserted for each geometry. |
| `epsilon_match` | Reconstruction anchor and lower edge of the activation window | `"standard"` or a governed total-energy-density value in MeV fm^-3. |
| `center` | Gaussian center `ε0` | Positive finite scalar or nonempty array of distinct values, in MeV fm^-3. |
| `width` | Gaussian standard deviation `σ` | Positive finite scalar or nonempty array of distinct values, in MeV fm^-3. |
| `ramp_width` | Width `Δ` of the quintic smootherstep rise | Positive finite scalar or nonempty array of distinct values, in MeV fm^-3. |
| `calculation` | Requested calculation layer | `"thermodynamics"` or `"stellar"`. |
| `precision` | Governed numerical profile | `"quick"`, `"strict"`, or one of the experimental dataset-family names below. |
| `fixed_masses` | Requested fixed gravitational masses | Nonempty array of distinct positive values below 10, in solar masses; at most 32 values. Retained in thermodynamics-only settings but used only by stellar work. |
| `diagnostics` | Governed extended stellar diagnostics | `"off"` or `"on"`; `"on"` requires stellar calculation. |

## Cartesian expansion and identity controls

`center`, `width`, and `ramp_width` may each be a scalar or an array. The plan
forms their deterministic Cartesian product in this order:

1. center in declared order;
2. width in declared order; and
3. ramp width in declared order.

For each geometry, amplitudes keep their declared order. If zero was omitted,
the logical zero-amplitude control is inserted before them.

Every geometry has a logical zero-amplitude row, but the undeformed baseline
is one physical calculation. The lexicographically first geometry owns it;
other zero rows are nonexecuting aliases to that physical case. Nonzero
amplitudes remain one physical case per geometry. The passive plan exposes
logical/physical counts and alias mappings.

The public interface limits a plan to:

- 256 expanded geometries;
- 4,096 expanded logical geometry-amplitude cases, including injected zero
  controls; and
- 32 fixed-mass targets.

These are planning-safety bounds, not a recommendation to execute the largest
permitted grid.

## Geometry constraints

All amplitudes and geometry values must be finite, and each geometry requires:

```text
center > 0
width > 0
ramp_width > 0
```

### BSk24

A numeric `epsilon_match` must lie strictly inside the governed
homogeneous-core interval `(76.5591451931, 1508.9793344234) MeV fm^-3`.
The Gaussian center need not be above the anchor. Planning accepts a
center/width pair when its nominal open four-sigma support has a nonempty
overlap with the deformable domain above the anchor and below the direct BSk24
endpoint. A point contact is not overlap.

`ramp_width` controls the activation window and is not restricted by
`center - epsilon_match`.

## Calculation layers

`thermodynamics` evaluates the raw proposal and, only when accepted,
reconstructs the effective cold barotrope. It does not call a stellar solver.

`stellar` adds sampled TOV/tidal sequences, requested fixed-mass roots, and
turning-point maximum-mass assessment for accepted cases. It is substantially
more expensive.

A fixed-mass observable is available only when the successful stable branch
truly brackets the target and the required central pressure stays inside the
case EoS domain. Maximum-mass availability is independent: the largest sampled
mass is not promoted to a maximum without a bracketed, refined turning point.

An early retained causal endpoint can leave valid fixed-mass results while
making maximum mass unavailable.

With diagnostics `on`, the workflow retains governed endpoint radial
diagnostics for the zero-amplitude control and selected accepted sign
endpoints. Rejected cases do not reach that stage.

## Precision profiles

Profile names expand during passive planning. Their complete grids,
tolerances, stage names, requested work, and reporting groups contribute to
the plan/configuration identity and are saved with results. Public JSON does
not expose internal stage arrays.

For thermodynamics-only `quick`, the two reconstruction stages use
129/257 and 257/513 nodes, and the raw gate uses 257/1,025 points. Stellar
`quick` instead uses one 257/513-node thermodynamic pilot, a 17-pressure TOV
stage at rtol `1e-8`, atol `1e-10` with 301 radial samples, and a
1,025/4,097-point raw gate. `strict` uses 1,025/2,049, 2,049/4,097, and
4,097/8,193-node thermodynamic stages plus a 4,097/16,385-point raw gate.
Its stellar stages use 61, 121, and 121 pressures; the final stage uses rtol
`1e-10`, atol `1e-12`, and 1,201 radial samples. Dataset-family profiles
inherit the strict thermodynamic/raw-gate grids and replace the repeated
stellar sequence with the single stage shown below.

| Profile | Model | Governing intent |
|---|---|---|
| `quick` | BSk24 | Small exploratory profile. Thermodynamics-only work uses two stages; stellar work instead uses one thermodynamic pilot plus one 17-pressure stage at rtol `1e-8`, atol `1e-10`. Not a convergence certificate. |
| `strict` | BSk24 | Three thermodynamic stages and, for stellar work, 61/121/121-pressure stages with final rtol `1e-10`, atol `1e-12`. Saved convergence statuses still require interpretation. |
| `dataset` | BSk24 | One 61-pressure stellar stage at rtol `1e-10`, atol `1e-12`, while retaining strict-family thermodynamic stages. |
| `dataset_10_tighter` | BSk24 | One 10-pressure stage at rtol `1e-11`, atol `1e-13`; changes sampling and tolerance together. |
| `dataset_20` | BSk24 | One 20-pressure stage at rtol `1e-10`, atol `1e-12`. |
| `dataset_40` | BSk24 | One 40-pressure stage at rtol `1e-10`, atol `1e-12` |
| `dataset_40_curves` | BSk24 | Curve-only final thermodynamic stage plus the 40-pressure stellar grid; omits residual processing, fixed-mass roots, maximum-mass refinement, and retained radial profiles. |
| `dataset_relaxed` | BSk24 | One 61-pressure stage at rtol `1e-8`, atol `1e-10`. |
| `dataset_relaxed_80` | BSk24 | One 80-pressure stage at the `dataset_relaxed` tolerances. |

Every dataset-family profile requires `calculation = "stellar"` and
`diagnostics = "off"`. All are experimental single-stage stellar routes, not
renamed `strict` calculations or certificates of curve sampling, convergence,
machine-learning suitability, or publication readiness. All-node tides are
requested by these profiles; actual availability remains status-driven.

Read [Dataset workflows](dataset.md) before using them.

## Starting files

- [`../configs/quickstart.json`](../configs/quickstart.json): small BSk24
  thermodynamics onboarding configuration.
- [`../configs/custom_experiment.json`](../configs/custom_experiment.json):
  strict BSk24 thermodynamics template with signed amplitudes.
- [`../configs/stellar_example.json`](../configs/stellar_example.json): small
  strict BSk24 stellar example.
- [`../configs/final_negative_dataset.json`](../configs/final_negative_dataset.json):
  retained large BSk24 campaign input using `dataset_40`; not a starter or a
  qualification record.

Copy a suitable example to a new filename for a new study. Do not edit frozen
implementation constants to configure an experiment, and always inspect the
full passive expansion before execution.

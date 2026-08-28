# Parameters

The JSON contract is machine-readable in
[`../configs/schema.json`](../configs/schema.json). Runtime validation is
authoritative for relationships between fields and for finite-value checks.

## Public settings

| Field | Meaning | Allowed values and units |
|---|---|---|
| `matter_model` | Selects the governed matter baseline. | Optional `"bsk24"` or explicit `"cfl"`. Omission is the canonical legacy BSk24 form and remains omitted from its normalized serialization. |
| `amplitudes` | Additive coefficients `A` multiplying the windowed Gaussian contribution to `c_s^2`. | Nonempty list of finite dimensionless numbers. If zero is absent, an identity control is added internally. |
| `epsilon_match` | Model-specific thermodynamic reconstruction anchor and lower endpoint of the activation window. | BSk24: `"standard"` or one governed total-energy-density value in MeV fm^-3. CFL: exactly `"surface"`. |
| `center` | Gaussian center `ε0`. | Positive number or nonempty list, in MeV fm^-3. |
| `width` | Gaussian standard deviation `σ`. | Positive number or nonempty list, in MeV fm^-3. |
| `ramp_width` | Smootherstep rise width `Δ`; the window rises from zero at the anchor to one at `epsilon_match + Δ`. | Positive number or nonempty list, in MeV fm^-3. |
| `calculation` | Selects the calculation layer. | `"thermodynamics"` or `"stellar"`. |
| `precision` | Selects a governed numerical profile. | `"quick"`, `"strict"`, or experimental `"dataset"` / `"dataset_10_tighter"` / `"dataset_20"` / `"dataset_40"` / `"dataset_relaxed"` / `"dataset_relaxed_80"`. |
| `fixed_masses` | Gravitational masses requested from a stellar sequence. | Nonempty list of positive values below 10, in solar masses. |
| `diagnostics` | Requests governed endpoint extended stellar diagnostics. | BSk24: `"off"` or `"on"`, with `"on"` requiring `calculation = "stellar"`. CFL: exactly `"off"` in 1.1. |

`$schema` is required in public JSON files so editors and validators use the
same contract. It is removed before scientific settings are normalized and
hashed. `matter_model` is a discriminator, not a tunable physical parameter:
it is omitted from legacy BSk24 serialization and included explicitly in CFL
identity. The frozen CFL constants cannot be supplied through public JSON;
see [`cfl.md`](cfl.md).

## Scalar and list expansion

`center`, `width`, and `ramp_width` each accept a scalar or a list. Scalars
behave as one-element lists. The workflow forms the deterministic Cartesian
product of those geometry values and the declared amplitudes.

For example:

```json
{
  "$schema": "./schema.json",
  "amplitudes": [0.0, 0.01, -0.01],
  "epsilon_match": "standard",
  "center": [200.0, 300.0],
  "width": 50.0,
  "ramp_width": [30.0, 40.0],
  "calculation": "thermodynamics",
  "precision": "strict",
  "fixed_masses": [1.4],
  "diagnostics": "off"
}
```

requests twelve declared BSk24 geometry-amplitude combinations. If an
amplitude list omits zero, each geometry receives an internally added logical
identity control.

For both BSk24 and CFL, every geometry has a logical zero-amplitude row, but
the undeformed baseline is a single physical calculation. The
lexicographically first geometry is its deterministic owner; the remaining
logical zero rows are nonexecuting aliases to the same anchor-aware physical
case ID. The plan exposes the logical alias table and estimates only actual
physical work. Nonzero amplitudes remain one physical case per geometry.
Always inspect the passive plan for exact logical and physical case counts
and IDs.

To keep accidental pasted grids from exhausting memory during a supposedly
safe preview, the public interface permits at most 256 expanded geometries and
4096 expanded geometry-amplitude cases, including an injected zero control.
It also permits at most 32 fixed-mass targets per experiment. Split larger
studies into reviewed experiments with distinct output roots.

## Geometry constraints

For every expanded combination, runtime validation requires finite amplitudes,
positive geometry scales, and

```text
center > 0
width > 0
ramp_width > 0.
```

For BSk24, the schema restricts a numeric `epsilon_match` to the retained
homogeneous-core interval. The Gaussian center is not required to lie above
the activation anchor: the smootherstep window still makes the deformation
exactly zero below that anchor. A requested center and width are retained when
the Gaussian's nominal four-standard-deviation support has a nonempty open
intersection with the deformable domain above the anchor and below the direct
BSk24 endpoint. This allows a below-anchor center whose tail meaningfully
overlaps the domain and rejects, during passive planning, a geometry with no
meaningful in-domain support. A point contact is not an overlap. Likewise,
`ramp_width` controls the window rise and is not geometrically bounded by
`center - epsilon_match`. These checks and the fully expanded geometry occur
before scientific execution.

For CFL, the anchor is the fixed finite-density surface
`190.2181760065314 MeV fm^-3`, the center must lie strictly inside the
formula-derived domain ending at `4008.81724402691 MeV fm^-3`, and the
surface plus ramp width must not exceed that endpoint; equality is allowed.
The width is positive but need not fit inside a compact support: the Gaussian
is evaluated only on the governed domain. Rounded design-review endpoint
values are not accepted as identity replacements. These checks and the fully
expanded geometry occur before scientific execution.

## Calculation choices

`thermodynamics` constructs and assesses the raw proposal and, when accepted,
performs effective cold reconstruction. It does not run a stellar solver.

`stellar` includes the governed background sequence, fixed-mass observables,
and tidal calculation for accepted cases. It is substantially more expensive.
Requested fixed masses are reported only when they are truly bracketed on the
successful stable branch and their central pressures lie inside the retained
case-specific EoS domain. Maximum-mass availability is independent: an
endpoint-limited case can retain solved requested fixed masses while its
turning point remains unavailable.

The passive plan reports the declared sequence, fixed-mass, and initial local
maximum-mass screening targets. Any later turning-point refinement is adaptive,
so its exact solver-call count cannot be known before execution and is labelled
separately rather than hidden inside a misleading fixed total.

With diagnostics `on`, the workflow retains the zero-amplitude control and
the strongest accepted endpoint per available sign for the governed extended
radial diagnostics. Rejected cases never reach that stage.

That diagnostics capability has not been established for a bare self-bound
CFL surface. CFL settings with diagnostics `on` therefore fail closed during
public validation rather than reusing neutron-star support semantics.

## Precision choices

`quick` is a small governed profile for installation checks and workflow
familiarization. Its expansion is calculation-dependent: thermodynamics uses
the retained quickstart profile, while stellar uses the retained relaxed
notebook profile. It is not a substitute for strict numerical results.

`strict` expands to the governed refinement stages used for scientific
results. The public JSON deliberately does not expose stage arrays or solver
tolerances. The complete expansion appears in the passive plan, contributes
to the deterministic configuration hash, and is saved in the result packet.

## Choosing a starting file

The experimental `dataset` profile is stellar-only with diagnostics off.
It preserves STRICT thermodynamics and final integration tolerances, but uses
one 61-point stellar sequence with no per-case stellar refinement envelope.
See [dataset.md](dataset.md) for the exact contract and qualification limits.
The `dataset_10_tighter` candidate changes the stage label, sequence count to
10 and ODE tolerances to rtol=1e-11 / atol=1e-13. It retains all-node tides and
all other dataset settings; it is not STRICT certification. Tighter ODE
integration does not certify sparse-curve sampling.
The `dataset_40` candidate changes only the dataset sequence count to 40 and
the stage label, retaining rtol=1e-10, atol=1e-12 and all-node tides.
It is the only dataset-family profile exposed for CFL. CFL uses the identical
numerical expansion with its self-bound surface/tidal contract and suppresses
per-packet plots; it remains experimental pending a matched STRICT comparison.
The `dataset_20` candidate does the same with 20 points; sparse-curve sampling
needs separate qualification from integration accuracy.
The separately named `dataset_relaxed` candidate changes only the dataset
stellar integration tolerances to `rtol=1e-8`, `atol=1e-10`; it retains the
61-point grid and all-node tides, and is not a STRICT certificate.
The `dataset_relaxed_80` variant changes only that sequence count to 80 and
the stage label; tolerances and all other settings remain as in dataset_relaxed.

- [`../configs/quickstart.json`](../configs/quickstart.json) uses the existing
  small BSk24 thermodynamic geometry and `quick` profile.
- [`../configs/cfl_quickstart.json`](../configs/cfl_quickstart.json) uses an
  explicit CFL surface anchor and a small thermodynamics-only `quick` profile.
- [`../configs/custom_experiment.json`](../configs/custom_experiment.json)
  contains the existing signed amplitudes and three ramp widths with `strict`
  precision.
- [`../configs/stellar_example.json`](../configs/stellar_example.json) is a
  small signed-amplitude stellar example with `strict` precision.

Copy one to a new filename; do not edit the supplied examples in place when
you need a durable record of your own settings. The supplied CFL quickstart is
exploratory; it is not a published stellar benchmark or a substitute for a
strict convergence study.

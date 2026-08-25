# Parameters

The JSON contract is machine-readable in
[`../configs/schema.json`](../configs/schema.json). Runtime validation is
authoritative for relationships between fields and for finite-value checks.

## Public settings

| Field | Meaning | Allowed values and units |
|---|---|---|
| `amplitudes` | Additive coefficients `A` multiplying the windowed Gaussian contribution to `c_s^2`. | Nonempty list of finite dimensionless numbers. If zero is absent, the identity control is added internally. |
| `epsilon_match` | Thermodynamic reconstruction anchor and lower endpoint of the activation window. | `"standard"`, or one total-energy-density value in MeV fm^-3 strictly inside the retained homogeneous-core interval. |
| `center` | Gaussian center `ε0`. | Positive number or nonempty list, in MeV fm^-3. |
| `width` | Gaussian standard deviation `σ`. | Positive number or nonempty list, in MeV fm^-3. |
| `ramp_width` | Smootherstep rise width `Δ`; the window rises from zero at the anchor to one at `epsilon_match + Δ`. | Positive number or nonempty list, in MeV fm^-3. |
| `calculation` | Selects the calculation layer. | `"thermodynamics"` or `"stellar"`. |
| `precision` | Selects a governed numerical profile. | `"quick"` or `"strict"`. |
| `fixed_masses` | Gravitational masses requested from a stellar sequence. | Nonempty list of positive values below 10, in solar masses. |
| `diagnostics` | Requests governed endpoint extended stellar diagnostics. | `"off"` or `"on"`; `"on"` requires `calculation = "stellar"`. |

`$schema` is required in public JSON files so editors and validators use the
same contract. It is removed before scientific settings are normalized and
hashed, so it is not a tenth scientific control.

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

requests twelve declared geometry-amplitude combinations. If an amplitude
list omits zero, each geometry receives an internally added identity control.
Always inspect the passive plan for the exact case count and case IDs.

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

The schema also restricts a numeric `epsilon_match` to the retained
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

- [`../configs/quickstart.json`](../configs/quickstart.json) uses the existing
  small thermodynamic geometry and `quick` profile.
- [`../configs/custom_experiment.json`](../configs/custom_experiment.json)
  contains the existing signed amplitudes and three ramp widths with `strict`
  precision.
- [`../configs/stellar_example.json`](../configs/stellar_example.json) is a
  small signed-amplitude stellar example with `strict` precision.

Copy one to a new filename; do not edit the supplied examples in place when
you need a durable record of your own settings.

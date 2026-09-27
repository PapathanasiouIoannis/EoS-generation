# Method

## Baselines and conventions

The supported baseline is the analytical representation of the unified cold
BSk24 neutron-star equation of state. The optional `matter_model` field may
only name `"bsk24"`; omission is the canonical form.

The independent variable used by the deformation workflow is total energy
density, including rest-mass energy, in MeV fm^-3. Pressure
uses the same units and

```text
c_s^2 = dP/dε
```

is dimensionless in units with `c = 1`.

The direct BSk24 control retains its declared causal domain. For a nonzero
deformation, the raw analytical proposal is assessed through the documented
upper domain of the published BSk24 fit (`rho = 10^16 g cm^-3`). This is not
an extrapolation or a causal repair: the published C4 expression and its
analytical derivative are evaluated unchanged, and causality is applied to
the combined deformed sound speed. The implementation does not extrapolate
beyond its governed domain.

## Smooth sound-speed deformation

For amplitude `A`, center `ε0`, Gaussian width `σ`, and smootherstep ramp
width `Δ`, the raw proposal is

```text
c_s,raw^2(ε) = c_s,0^2(ε)
                 + A exp[-(ε - ε0)^2 / (2 σ^2)] W(ε).
```

Let `εt` denote the selected reconstruction anchor and
`x = (ε - εt) / Δ`. The compact activation window is

```text
W(ε) = 0                              for ε <= εt
     = 6 x^5 - 15 x^4 + 10 x^3       for εt < ε < εt + Δ
     = 1                              for ε >= εt + Δ.
```

This quintic smootherstep has continuous first and second derivatives at
both endpoints. It introduces the deformation smoothly above the anchor
without a corner in `c_s^2` at activation.

The anchor is the retained numeric BSk24 value or `standard`. Every requested
geometry must use the applicable retained anchor and positive geometry scales:

```text
ε0 > 0
Δ > 0
σ > 0.
```

There is deliberately no ordering constraint between the activation anchor,
the Gaussian center, and the end of the ramp. If the center lies inside or
below the ramp, the window suppresses the corresponding part of the Gaussian.
The geometry has meaningful in-domain support when the open intersection of
its nominal four-standard-deviation Gaussian support with the deformable
domain is nonempty. A center may therefore lie below the anchor when its tail
overlaps that domain. The passive plan rejects a geometry whose four-sigma
support has no such overlap and exposes every retained geometry exactly.

## Pressure and effective thermodynamics

The raw pressure response is fixed by integrating the sound-speed change from
the anchor:

```text
Praw(ε) = P0(ε) + integral[εt to ε] Δc_s^2(u) du.
```

Consequently, even a localized sound-speed change generally leaves an
integrated pressure offset above its main support. That pressure response is
why a local deformation can shift the central energy density, radius, Love
number, and tidal deformability of a star at fixed gravitational mass.

Only an accepted raw proposal is reconstructed, and only on its retained
causal branch. The effective baryon density is obtained from the cold
first-law relation

```text
dε = μ_B dn_B,
μ_B = (ε + P) / n_B,
```

with continuity at the selected anchor. The implementation checks the Euler
identity

```text
P = n_B μ_B - ε
```

and retains the relevant residuals. This is an effective one-fluid
reconstruction; it does not determine microscopic particle fractions or
species chemical potentials.

## Fail-closed assessment and causal policy

Every complete raw proposal is assessed and saved before reconstruction or
stellar work. Assessment is not limited to the ordinary output grid:
deterministic geometry-scale nodes resolve the smootherstep ramp and
four-sigma support, and bounded local refinement examines every discovered
extremum basin. Saved resolution evidence must certify the analytical
deformation and retained tabulation. Narrow negative-`c_s^2` pockets and
superluminal islands cannot disappear between ordinary grid points. An
unresolved proposal receives no downstream work.

Across the assessed raw domain, the finite and mechanical hard conditions
include

```text
ε > 0
P >= 0
0 < dP/dε.
```

The usable retained prefix additionally requires

```text
dP/dε <= 1,
```

with equality allowed at the included endpoint.

When the combined proposal reaches `c_s^2 = 1`, the first continuously
resolved crossing defines the case-specific BSk24 endpoint and is included in
the retained branch. A proposal may reach it before or after the direct BSk24
endpoint without being rejected solely for that domain change. A sufficiently
softened proposal can instead remain causal through the governed upper end of
the published fit; its retained endpoint records
`published_bsk24_fit_endpoint`. Values after a first crossing are outside the
usable branch even if the raw proposal later returns below one. The complete
raw BSk24 proposal remains saved as evidence.

Failed values are never clipped, replaced, extrapolated, or relabelled as
accepted. A rejected or unresolved case retains its raw result and exact
reason, and receives no reconstruction or stellar calculation. The
zero-amplitude case is an explicit identity control. It must reproduce the
selected baseline under its governed floating-point policy. In public
Cartesian sweeps, the lexicographically first geometry
owns the one physical zero-amplitude execution; the other logical geometry
controls are stable, nonexecuting aliases to it. Nonzero cases retain their
geometry-specific physical identities.

Hard validity is separate from auxiliary thermodynamic diagnostics. Finite
quantities such as `P/epsilon`, `Gamma_eff`, effective chemical-potential
trends, `dmu_eff/dn_B`, and finite diagnostic residual magnitudes remain saved
for interpretation but do not by themselves reject a case. Non-finite or
unusable reconstruction, broken matching, interpolation, or inversion, and
other genuine numerical invalidity still fail closed.

## Stellar calculation

With `calculation = "stellar"`, accepted barotropes enter the governed
background TOV, fixed-mass, and tidal workflow. Every attempted, bracket, and
refined central pressure remains at or below the case-specific retained EoS
endpoint; interpolation and inversion do not extend that domain. Each
discontinuity or surface correction is applied according to its classified
numerical or physical role,
and a tidal result remains unavailable if the required capability is not
established.

Fixed-mass observables require a true bracket on the successful stable branch.
A maximum mass is marked resolved only after the turning point is bracketed
and refined; the largest sampled mass is not automatically a maximum. An
early retained causal endpoint can leave valid requested fixed-mass
solutions available while maximum mass is reported as unavailable or
unresolved. That partial availability does not invalidate the EoS.

The named `quick` and `strict` profiles expand to fixed internal grids,
tolerances, and convergence stages. The plan shows those settings and the
result records them. Choosing a profile changes numerical effort, not the
physical definition of the deformation. `quick` is exploratory;
publication-level claims require reviewed convergence and independent
scientific support.

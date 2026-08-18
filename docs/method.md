# Method

## Baseline and conventions

The baseline is the analytical representation of the unified cold BSk24
neutron-star equation of state. The independent variable used by the
deformation workflow is total energy density, including rest-mass energy, in
MeV fm^-3. Pressure uses the same units and

```text
c_s^2 = dP/dε
```

is dimensionless in units with `c = 1`.

The implementation retains the declared finite BSk24 domain. It does not use
uncontrolled extrapolation beyond that domain.

## Smooth sound-speed deformation

For amplitude `A`, center `ε0`, Gaussian width `σ`, and smootherstep ramp
width `Δ`, the raw proposal is

```text
c_s,raw^2(ε) = c_s,BSk24^2(ε)
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

Every requested geometry must use a retained anchor (or `standard`) and
positive geometry scales:

```text
ε0 > 0
Δ > 0
σ > 0.
```

There is deliberately no ordering constraint between the activation anchor,
the Gaussian center, and the end of the ramp. If the center lies inside or
below the ramp, the window suppresses the corresponding part of the Gaussian;
the passive plan exposes that requested geometry exactly.

## Pressure and effective thermodynamics

The raw pressure response is fixed by integrating the sound-speed change from
the anchor:

```text
Praw(ε) = PBSk24(ε) + integral[εt to ε] Δc_s^2(u) du.
```

Consequently, even a localized sound-speed change generally leaves an
integrated pressure offset above its main support. That pressure response is
why a local deformation can shift the central energy density, radius, Love
number, and tidal deformability of a star at fixed gravitational mass.

Only an accepted raw proposal is reconstructed. The effective baryon density
is obtained from the cold first-law relation

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

## Fail-closed assessment

The complete raw proposal is assessed before reconstruction or stellar work.
On each assessed continuous interval, the workflow requires the declared
finite-domain conditions, including

```text
ε > 0
P >= 0
0 < dP/dε <= 1.
```

It also checks monotonic and thermodynamic consistency requirements where the
needed quantities are available. Failed values are never clipped, replaced,
or relabelled as accepted. A rejected case retains its raw result and exact
reason, and receives no downstream calculation.

The zero-amplitude case is an explicit identity control. It must reproduce
baseline BSk24 under the governed floating-point policy.

## Stellar calculation

With `calculation = "stellar"`, accepted barotropes enter the governed
background TOV, fixed-mass, and tidal workflow. Interpolation and inversion
remain restricted to the declared domain. Each discontinuity or surface
correction is applied according to its classified numerical or physical role,
and a tidal result remains unavailable if the required capability is not
established.

Fixed-mass observables require a true bracket on the successful stable branch.
A maximum mass is marked resolved only after the turning point is bracketed
and refined; the largest sampled mass is not automatically a maximum.

The named `quick` and `strict` profiles expand to fixed internal grids,
tolerances, and convergence stages. The plan shows those settings and the
result records them. Choosing a profile changes numerical effort, not the
physical definition of the deformation.

# Method

## Baselines and conventions

The public `matter_model` discriminator selects one of two separately
governed baselines:

- omitted or `"bsk24"`: the legacy analytical representation of the unified
  cold BSk24 neutron-star equation of state;
- explicit `"cfl"`: the frozen full finite-strange-mass bag-model CFL phase
  for a pure, bare, self-bound quark star.

The CFL equation set, constants, formula-derived binary64 domain, stability
assumptions, and primary literature are fixed in [`cfl.md`](cfl.md). In
particular, it has no crust or hadronic envelope and cannot be combined with
BSk24 as a hybrid construction.

For both baselines, the independent variable used by the deformation workflow
is total energy density, including rest-mass energy, in MeV fm^-3. Pressure
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
the combined deformed sound speed.
The CFL model likewise retains its formula-derived finite domain. Neither
implementation uses uncontrolled extrapolation beyond its governed domain.

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

The anchor is model-specific. BSk24 accepts its retained numeric anchor or
`standard`. CFL requires `surface`, which resolves to the undeformed
finite-density zero-pressure surface. Every requested geometry must use the
applicable retained anchor and positive geometry scales:

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
For CFL, by contrast, the center must
lie strictly inside its complete domain and the surface plus ramp width must
not exceed the upper endpoint. In either model, if the center lies inside
the ramp, the window suppresses the corresponding part of the Gaussian; the
passive plan exposes that requested geometry exactly.

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

with continuity at the selected anchor. CFL specifically retains its analytic
finite surface baryon density as the first-law anchor and defines no EoS below
that point. The implementation checks the Euler identity

```text
P = n_B μ_B - ε
```

and retains the relevant residuals. This is an effective one-fluid
reconstruction; it does not determine microscopic particle fractions or
species chemical potentials.

## Fail-closed assessment and causal endpoint

The complete zero-amplitude control over the direct-BSk24 causal domain, or
the complete nonzero raw proposal over the published analytical-fit domain,
is assessed and retained as evidence before reconstruction or stellar work. Assessment is
not limited to the ordinary output grid: deterministic geometry-scale nodes
resolve the smootherstep ramp and four-sigma support, and bounded local
refinement examines every discovered extremum basin. The saved resolution
certificate must show that this continuous assessment and the retained
tabulation resolve the analytical deformation. Narrow negative-`c_s^2`
pockets and narrow superluminal islands are therefore not allowed to disappear
between ordinary grid points. A proposal whose required resolution cannot be
certified is explicitly unresolved and receives no downstream work.

Across the complete assessed raw domain, the workflow requires the declared
finite and mechanical hard conditions, including

```text
ε > 0
P >= 0
0 < dP/dε.
```

On the usable retained prefix it additionally requires

```text
dP/dε <= 1,
```

with equality allowed at the included endpoint.

The first continuously resolved crossing of the combined deformed
`c_s^2 = 1` defines a
case-specific causal endpoint and is itself included in the retained branch.
A deformed proposal may reach that endpoint before or after direct BSk24
without being rejected solely for the changed domain. A softened proposal can
therefore remain usable above the direct endpoint when its combined
`c_s^2 <= 1`. Once the first crossing is reached,
all higher-energy-density values are outside the usable branch even if the raw
proposal later returns below one. The complete raw proposal and diagnostics
on both sides of the endpoint remain saved as evidence.

Failed values are never clipped, replaced, extrapolated, or relabelled as
accepted. A rejected or unresolved case retains its raw result and exact
reason, and receives no reconstruction or stellar calculation. The
zero-amplitude case is an explicit identity control. It must reproduce
the selected baseline under its governed floating-point policy. In both
BSk24 and CFL public Cartesian sweeps, the lexicographically first geometry
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
discontinuity or surface
correction is applied according to its classified numerical or physical role,
and a tidal result remains unavailable if the required capability is not
established.

For CFL, TOV integration reaches the bare `P = 0` surface at finite inner
energy density and then joins directly to vacuum. With outward
`Delta epsilon = epsilon_inner - epsilon_outer > 0`, the repository-unit
tidal correction is

```text
y_out - y_in
  = -G_CONV r^3 Delta epsilon / (m + G_CONV r^3 P).
```

It is applied exactly once before the exterior-side `y` is used for `k2`; the
packet retains auditable before/after evidence. Missing or inconsistent
evidence fails the CFL tidal capability closed. See [`cfl.md`](cfl.md) for the
sign convention and primary sources. Extended radial diagnostics are not yet
an established bare-CFL capability, so CFL rejects `diagnostics = "on"`.

Fixed-mass observables require a true bracket on the successful stable branch.
A maximum mass is marked resolved only after the turning point is bracketed
and refined; the largest sampled mass is not automatically a maximum. When
the retained causal endpoint is reached first, valid requested fixed-mass
solutions remain available while the maximum mass is reported as unavailable
or unresolved. That partial availability does not invalidate the EoS.

The named `quick` and `strict` profiles expand to fixed internal grids,
tolerances, and convergence stages. The plan shows those settings and the
result records them. Choosing a profile changes numerical effort, not the
physical definition of the deformation. `quick` is exploratory; publication-
level CFL stellar claims additionally require reviewed strict convergence,
an independent solver comparison, and a published convention-matched
benchmark.

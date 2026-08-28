# Frozen CFL scientific contract

This page defines the color-flavor-locked (CFL) matter model supported by
EoS Generation 1.1. It is a separate baseline behind the public
`matter_model` discriminator. Nothing on this page changes the legacy BSk24
definition, serialization, settings hash, or case identities.

## Physical scope

The implemented object is pure, cold (`T = 0`), bare, self-bound CFL quark
matter. A stellar model ends at its finite-density zero-pressure surface and
is joined directly to vacuum. There is no nuclear or electron crust, no
hadronic envelope, no hadron-quark matching construction, and no hybrid-star
branch. The frozen ideal-CFL phase is locally electrically neutral, so
`mu_e = 0` and there is no electron or muon contribution.
The bulk-neutrality convention follows Alford, Rajagopal, Reddy, and Wilczek,
[Physical Review D 64, 074017 (2001)](https://doi.org/10.1103/PhysRevD.64.074017)
([arXiv:hep-ph/0105009](https://arxiv.org/abs/hep-ph/0105009)).

This is a phenomenological bag-model realization of CFL matter. It does not
claim that the frozen microphysical parameters are uniquely preferred by QCD
or observation. Later sound-speed sweeps vary only the declared deformation
geometry and amplitude; none of the quantities in the frozen table below is
a sweep dimension.

## Primary equation set

The production authority is the full finite-strange-mass,
common-Fermi-momentum treatment in Lugones and Horvath,
[Physical Review D 66, 074017 (2002)](https://doi.org/10.1103/PhysRevD.66.074017),
Eqs. 2--6 ([arXiv:hep-ph/0211070](https://arxiv.org/abs/hep-ph/0211070)).
The implemented formulation ID is `cfl_bag_full_ms_delta2_v1`. It is not the
separately derived lowest-order strange-mass expansion in that paper.

`mu` below is the common quark chemical potential, not the baryon chemical
potential. The locked common Fermi momentum is

```text
nu(mu) = 2 mu - sqrt(mu^2 + m_s^2 / 3),
mu_B = 3 mu.
```

The two massless flavors have total spin-color-flavor degeneracy 12, giving
the coefficient `6/pi^2`; the strange flavor has spin-color degeneracy six,
giving `3/pi^2`. The natural-unit thermodynamic-potential density is

```text
Omega_CFL^(4)(mu)
  = (6/pi^2) integral[0 to nu] (p - mu) p^2 dp
  + (3/pi^2) integral[0 to nu] (sqrt(p^2 + m_s^2) - mu) p^2 dp
  - 3 Delta^2 mu^2 / pi^2
  + B.
```

For the frozen nonzero `m_s`, the two integrals are evaluated in the exact
closed forms

```text
I_0 = nu^4 / 4 - mu nu^3 / 3,

I_s = {nu sqrt(nu^2 + m_s^2) [2 nu^2 + m_s^2]
       - m_s^4 asinh(nu / m_s)} / 8
      - mu nu^3 / 3,

Omega_CFL^(4) = 6 I_0 / pi^2 + 3 I_s / pi^2
                - 3 Delta^2 mu^2 / pi^2 + B.
```

`Omega_CFL^(4)` and `B` are in `MeV^4`. Division by `(hbar c)^3` converts a
thermodynamic density to `MeV fm^-3`. The governed relations are

```text
P_raw(mu) = -Omega_CFL^(4)(mu) / (hbar c)^3,
P(mu)     = P_raw(mu) - P_raw(mu_surface),
n_B(mu)   = [nu^3 + 2 Delta^2 mu] / [pi^2 (hbar c)^3],
epsilon(mu) = -P(mu) + 3 mu n_B(mu),
mu_B(mu)    = 3 mu.
```

Energy density includes rest energy. `n_B` is baryon number density and the
total quark number density is `3 n_B`. The one constant pressure subtraction
is the governed binary64 surface normalization described below; it changes no
derivative.

The derivative is analytic. With

```text
nu'(mu) = 2 - mu / sqrt(mu^2 + m_s^2 / 3),

c_s^2 = dP/depsilon
      = [nu^3 + 2 Delta^2 mu]
        / {mu [3 nu^2 nu'(mu) + 2 Delta^2]}.
```

The implementation also evaluates numerical derivatives as an independent
consistency check. The governed thermodynamic identities are

```text
d epsilon = mu_B d n_B,
P = n_B mu_B - epsilon,
mu_B = (epsilon + P) / n_B.
```

Mixing the full equation set with coefficients from the truncated
`m_s^2`/`m_s^4` effective-bag expressions would define a different EoS and is
not supported by this parameter-set identity.

## Frozen constants and identity

The frozen high-mass pure-star parameter family follows Lugones and Horvath,
[Astronomy & Astrophysics 403, 173 (2003)](https://doi.org/10.1051/0004-6361:20030374)
([arXiv:astro-ph/0211638](https://arxiv.org/abs/astro-ph/0211638)). Their
Figs. 4--7 vary `B` directly in `MeV fm^-3`, include `m_s = 100 MeV` and
`Delta = 100 MeV` in the displayed families, and identify small `B` plus high
`Delta` as the direction producing pure CFL stars approaching `2.4 M_sun`.
The exact project choice `B = 57.5 MeV fm^-3` is a frozen point near the
paper's quoted lower stability edge; the paper is not treated as an
implementation-derived fixture.

| Quantity or convention | Frozen value |
|---|---|
| parameter-set ID | `cfl_full_finite_ms_bag_delta2_b57p5_ms100_delta100_v1` |
| parameter-set SHA-256 | `3991cb8615d2d29617ccb90c6dc54b23aae64bcc752856d07f17f99abc048307` |
| temperature | `T = 0` |
| light-quark masses | `m_u = m_d = 0 MeV` |
| strange-quark mass | `m_s = 100 MeV` |
| pairing gap | `Delta = 100 MeV`, retained through order `Delta^2` |
| authoritative bag constant | `B = 57.5 MeV fm^-3` |
| exact project conversions | `B = 441801570.2435963 MeV^4`; derived `B^(1/4) = 144.97957215191494 MeV` |
| unit constant | `hbar c = 197.3269804 MeV fm` |
| perturbative convention | no perturbative correction (`a4 = 1` convention); no renormalization scale |
| charge sector | ideal local CFL neutrality; `mu_e = 0`; no lepton term |
| omitted low-energy terms | no Goldstone-boson or kaon-condensate contribution |

The full immutable record and its hash are saved in planning and result
provenance. Altering any entry requires a new parameter-set ID and hash; it
cannot be done through Cartesian sweep expansion.

## Self-bound surface, domain, and stability

The approved surface quark chemical potential is
`mu_surface = 249.31780807778472 MeV`, so
`mu_B,surface = E/A = 747.9534242333541 MeV`. The governed pressure is exactly
zero there while the inner energy density is finite:

```text
nu_surface      = 242.7201950531192 MeV,
n_B,surface     = 0.2543182099894835 fm^-3,
epsilon_surface = 190.2181760065314 MeV fm^-3,
c_s,surface^2   = 0.3842809221857484.
```

These are the formula-derived binary64 values of the analytic expressions
above. The baryon surface density is retained at the reconstruction anchor
rather than extrapolated toward zero density. The CFL absolute-stability gate uses
`E/A <= 930 MeV` at zero pressure and therefore passes for this frozen phase.
The stability logic follows the strange-matter discussion of Farhi and Jaffe,
[Physical Review D 30, 2379 (1984)](https://doi.org/10.1103/PhysRevD.30.2379).

That CFL-only calculation cannot establish the distinct ordinary-nuclei
safety condition. This project records, but does not evaluate, the external
assumption that an independently defined two-flavor quark phase has
`E/A >= 934 MeV` at zero pressure. No result may report that constraint as
checked unless such a separately governed two-flavor model is added.

The governed closed chemical-potential and energy-density domains are

```text
mu       in [249.31780807778472, 600.0] MeV,
mu_B     in [747.9534242333541, 1800.0] MeV,
n_B      in [0.2543182099894835, 2.9673129144553494] fm^-3,
P        in [0.0, 1332.3460019927188] MeV fm^-3,
epsilon  in [190.2181760065314, 4008.81724402691] MeV fm^-3.
```

The energy-density endpoints above are the authoritative formula-derived
binary64 values. The corresponding review values
`[190.218176006531, 4008.8172440269]` are rounded display references only;
they never replace an endpoint or participate in equality, hashing, case
identity, or provenance. At the approved binary64 `mu_surface`, direct
formula evaluation gives
`P_raw = 2.3272441558063732e-14 MeV fm^-3`. Subtracting this single global
constant makes `P_surface` exactly zero and leaves all derivatives unchanged.

Across the complete domain the baseline must remain positive in energy and
baryon density, nonnegative in pressure, mechanically stable, monotone, and
causal. The fully gapped applicability check is
`m_s^2 / mu < 2 Delta`; its most restrictive point is the surface and the
frozen values pass it. This criterion follows Alford, Kouvaris, and Rajagopal,
[Physical Review Letters 92, 222001 (2004)](https://doi.org/10.1103/PhysRevLett.92.222001)
([arXiv:hep-ph/0311286](https://arxiv.org/abs/hep-ph/0311286)). Any out-of-domain
evaluation, failed positivity or monotonicity condition, nonpositive
`c_s^2`, or causal crossing fails closed; there is no extrapolation, clipping,
endpoint truncation, or post-hoc repair.

## Surface-anchored deformation and reconstruction

For CFL, the independent variable is total energy density over the complete
domain above. `epsilon_match` is fixed to `"surface"`; it resolves to the
undeformed finite-density surface `epsilon_surface`. For amplitude `A`,
center `epsilon_0`, Gaussian width `sigma`, and ramp width `Delta_epsilon`,

```text
c_s,raw^2(epsilon)
  = c_s,CFL^2(epsilon)
    + A exp[-(epsilon - epsilon_0)^2 / (2 sigma^2)] W(epsilon),

W = 0                                      at epsilon <= epsilon_surface,
W = 6x^5 - 15x^4 + 10x^3                  in the ramp,
W = 1                                      above the ramp,
x = (epsilon - epsilon_surface) / Delta_epsilon.
```

The center must lie strictly inside the CFL domain, both width and ramp width
must be positive and representably nonzero in binary64, and the ramp must not
end above the upper endpoint; equality at that endpoint is allowed.
The baseline surface point is preserved because `W(epsilon_surface) = 0` and

```text
P_A(epsilon)
  = P_CFL(epsilon)
    + integral[epsilon_surface to epsilon]
        A exp[-(u - epsilon_0)^2 / (2 sigma^2)] W(u) du.
```

The pressure primitive is evaluated in the dimensionless ramp coordinate by
a governed segmented 64-point Gauss-Legendre rule, with breakpoints tied to
the Gaussian center and width. Above the ramp, its Gaussian tail uses a
same-tail-cancellation-safe normal-CDF difference. This avoids the catastrophic
subtraction in raw Gaussian moments for narrow ramps. Multi-scale tests compare
the primitive to an independently integrated definition and verify
`d(P_A-P_CFL)/d epsilon = A G W`; the exact `A = 0` path does no quadrature.

The raw proposal is assessed over the entire declared domain before
reconstruction. A rejection retains the unmodified raw value, location, and
reason and receives no reconstruction or stellar work. Accepted cases use
the surface state as their first-law anchor:

```text
n_B(epsilon)
  = n_B,surface exp(integral[epsilon_surface to epsilon]
                      du / [u + P_A(u)]),
mu_B(epsilon) = [epsilon + P_A(epsilon)] / n_B(epsilon).
```

No EoS is manufactured below the surface. A continuous causal crossing or
other invalid endpoint rejects the full case; the domain is not shortened to
make it pass. `A = 0` delegates to the baseline arrays and evaluators under
the exact floating-point identity policy.

For a Cartesian geometry sweep, every geometry still has a logical
zero-amplitude identity row. Exactly one deterministic lexicographically
first geometry owns and executes the single physical baseline case. The
other zero-amplitude rows are nonexecuting aliases to that physical case.
Planning reports both logical rows and truthful deduplicated work estimates;
nonzero amplitudes retain one physical case per requested geometry.

## Bare-star stellar and tidal semantics

The TOV integration terminates at `P = 0`. The state just inside the surface
has `epsilon = epsilon_surface`; the exterior has `P = epsilon = 0`. A
hadronic low-density EoS is never attached.

In the repository's outward-jump convention,
`Delta epsilon = epsilon_inner - epsilon_outer > 0`. The finite-density
matching correction is

```text
y_out - y_in
  = -G_CONV r^3 Delta epsilon / (m + G_CONV r^3 P),
```

where repository solver units are defined by
`dm/dr = G_CONV epsilon r^2`. At the bare surface `P = 0`, so the correction
is negative. This is Eq. 11 of Takatsy and Kovacs,
[Physical Review D 102, 028501 (2020)](https://doi.org/10.1103/PhysRevD.102.028501)
([arXiv:2007.01139](https://arxiv.org/abs/2007.01139)), expressed in the
repository's mass and density units. It is consistent with the self-bound
surface treatment discussed by Postnikov, Prakash, and Lattimer,
[Physical Review D 82, 024016 (2010)](https://doi.org/10.1103/PhysRevD.82.024016).

The jump is applied exactly once, after the interior integration and before
the exterior-side `y` enters the corrected Hinderer `k2` expression. The
result records jump count, pressure, inner and outer energy densities,
`y_before`, `delta_y`, and `y_after`. Missing, repeated, positive-sign, or
inconsistent evidence invalidates the CFL tidal capability rather than
silently returning a Love number.

The Love-number algebra uses the corrected Hinderer expression; version 4 of
[arXiv:0711.2420](https://arxiv.org/abs/0711.2420) incorporates the published
erratum. That algebraic correction and the finite-density surface jump are
distinct requirements and neither substitutes for the other.

Background TOV sequences, fixed-gravitational-mass bracketing, successful
stable-prefix handling, and bracketed turning-point maximum-mass refinement
are supported under the same fail-closed rules as BSk24. CFL extended radial
diagnostics have not been audited for bare self-bound support and are not a
supported capability in 1.1; `diagnostics = "on"` is rejected for CFL.

The CFL-only stellar-selection policy is
`bare_self_bound_positive_mass_radius_v1`, retained in the expanded settings
hash and metadata. It requires a declared finite-density vacuum surface and
finite positive stellar mass and radius, with the existing compactness and
solver-validity checks. It does not impose the legacy hadronic minimum-mass
or minimum-radius cuts. This follows from the self-bound low-pressure limit:
`epsilon -> epsilon_surface` and
`M -> G_CONV epsilon_surface R^3 / 3`; no hadronic lower size bound follows.
The associated Newtonian tidal limit is `k2 -> 3/4` after the surface jump.
See Postnikov et al., Eqs. 8--11 and Sections IV and VI.B, cited above. The
governed central-pressure sampling floor itself is unchanged.

The physical A=0 stellar case reuses the single analytic `direct` solution;
thermodynamic arrays/evaluators still undergo exact identity checks. Saved
physical IDs and logical aliases make that reuse explicit. A duplicate solve
would not be independent validation.

CFL local maximum-mass refinement declares
`seed_preserving_split_log_pressure_v1` in expanded settings and metadata.
The existing sampled lower/middle/upper turning bracket is subdivided on
each side in log pressure, retaining those three solved pressure nodes
exactly. This prevents a recomputed, near-equal midpoint from creating a
spurious secant between two representations of the same intended node.
There is no tolerance-based merging or mass smoothing. The original
positive-to-negative sign checks, bounded refinement, and fail-closed
statuses remain unchanged. BSk24 retains its established local grid. This
numerical policy changes the expanded CFL hash, not public settings,
physical EoS identity, or frozen microphysics. Earlier development packets
without this declared policy are not silently migrated to the new contract.

## Notebook workflow

Open [`../notebooks/cfl_experiment.ipynb`](../notebooks/cfl_experiment.ipynb).
Its single editable settings cell supports scalar/list deformation geometry,
quick or strict precision, full stellar sequences, and fixed gravitational
masses as additional comparison points. The surface and microphysics remain
frozen. Run once with execution disabled, review the plan, then explicitly
execute the unchanged plan in the same kernel.

The result view combines accepted saved curves across geometries, including
M–R, Λ–M and k₂–M, and exposes exact availability/rejection statuses. It
has its own manifest outside the sealed experiment. Loading a saved result
and viewing existing plots are passive; creating a missing view requires the
separate `BUILD_SAVED_PLOTS` control. See [quickstart](quickstart.md#notebook-route)
and [acceptance evidence](cfl_acceptance.md).

For large data collection, open
[`../notebooks/cfl_dataset.ipynb`](../notebooks/cfl_dataset.ipynb). It fixes
the explicitly experimental `dataset_40` profile: one 40-pressure sequence
at `rtol=1e-10`, `atol=1e-12`, with 1201 radial samples and tides at every
sampled star. It retains all STRICT thermodynamic/raw-domain stages,
fixed-mass roots, maximum-mass refinement, and the bare CFL surface jump.
Scientific packet plot groups are `none`; after validation, reporting reads
the sealed tables once to create `EOS_DATA/` and exactly five combined plots.
No solver is called by that reporting step. This profile is not a full STRICT
certificate and requires a later matched CFL STRICT qualification.

## Verification and publication status

The implementation is protected by an independently transcribed closed-form
evaluation of the finite-`m_s` integrals, analytic and finite-difference
thermodynamic derivatives, exact thermodynamic identities, zero-amplitude
identity, complete-domain gate tests, and a uniform-density self-bound
surface-jump test. These are necessary software and equation checks, not an
independent published stellar validation. The executed notebook, independent
enthalpy solver, and numerical-refinement evidence are recorded separately in
[the acceptance report](cfl_acceptance.md), including non-passing diagnostic
flags. Convention-isolated massless and
zero-gap limits remain useful additional semi-analytic benchmarks when they
are added without changing the frozen production profile.

Before publication-level CFL mass-radius, maximum-mass, or tidal claims, a
reviewed strict run must demonstrate grid and solver convergence, an
independent solver comparison for TOV/tidal structure, and agreement with a
published pure self-bound CFL sequence using exactly matching conventions.
Lugones and Horvath,
[Astronomy & Astrophysics 403, 173 (2003)](https://doi.org/10.1051/0004-6361:20030374),
is the designated published sequence authority for that future benchmark.
Do not regenerate an expected fixture from this implementation and call it
independent validation. The `quick` profile and `configs/cfl_quickstart.json`
are exploratory workflow aids only.

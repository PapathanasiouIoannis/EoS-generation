# CFL verification status

This page summarizes the verification evidence and remaining qualification
boundary for the frozen CFL model distributed in EoS Generation 1.2.0. It is
not a result packet, a published stellar benchmark, or a claim that every
deformation is physically acceptable.

The scientific definition is authoritative in [the CFL contract](cfl.md).
Installation, preview, execution, and passive reopening instructions are in
the [quickstart](quickstart.md#notebook-route).

## Active frozen baseline

| Quantity | Governed value |
|---|---:|
| Formulation | `cfl_bag_full_ms_delta2_v1` |
| Bag constant `B` | `57.5 MeV fm^-3` |
| Derived `B^(1/4)` | `144.97957215191494 MeV` |
| Strange-quark mass `m_s` | `100 MeV` |
| CFL gap `Delta` | `100 MeV` |
| Perturbative convention | `a4 = 1` (no separate correction) |
| Quark-chemical-potential domain | `[249.31780807778472, 600] MeV` |
| Energy-density domain | `[190.2181760065314, 4008.81724402691] MeV fm^-3` |
| Parameter-set SHA-256 | `3991cb8615d2d29617ccb90c6dc54b23aae64bcc752856d07f17f99abc048307` |

At the finite-density zero-pressure surface, the governed binary64 state is:

```text
epsilon = 190.2181760065314 MeV fm^-3
n_B     = 0.2543182099894835 fm^-3
mu_B    = 747.9534242333541 MeV
c_s^2   = 0.3842809221857484
```

The frozen CFL phase passes the implemented `E/A <= 930 MeV` zero-pressure
criterion and the fully gapped applicability check. The separate two-flavor
ordinary-nuclei condition is an external assumption and is not evaluated by
this CFL-only model.

## What is covered

### Analytic and thermodynamic checks

The regression suite checks:

- independently transcribed closed-form finite-`m_s` integrals;
- analytic derivatives against numerical derivatives;
- first-law, Euler, and chemical-potential identities;
- exact formula-derived surface and endpoint states;
- positivity, monotonicity, mechanical stability, and causality across the
  complete frozen domain;
- the surface-anchored pressure primitive at several scales; and
- exact zero-amplitude identity and deterministic Cartesian aliasing.

The CFL gate requires the complete domain to pass. It does not reuse BSk24's
first-causal-crossing retained-prefix policy.

### Stellar and tidal checks

The suite checks the bare finite-density surface, positive self-bound
low-pressure scaling, TOV domain bounds, stable-prefix fixed-mass bracketing,
turning-point refinement, and the tidal jump sign/count/algebra. A valid CFL
tidal row must apply exactly one negative outward surface jump before `k2` and
Lambda are calculated.

The uniform-density self-bound limit and a constant-sound-speed scaling
relation provide additional checks without generating expected fixtures from
the production implementation.

### Reviewed undeformed quick/strict comparison

An explicitly authorized undeformed baseline was exercised through the public
experiment facade at both governed precisions. Both completed packets passed
read-only validation, accepted the one physical case, and passed the exact
zero-amplitude identity check.

Representative final results were:

| Observable | Quick | Strict |
|---|---:|---:|
| Refined `M_max` (`M_sun`) | `2.3466940198160042` | `2.3466940191683263` |
| Radius at `M_max` (km) | `12.43375780351352` | `12.43374842724906` |
| `R_1.4` (km) | `11.885078914951567` | `11.885078909788211` |
| `k2_1.4` | `0.2009435283682922` | `0.20094352816465091` |
| `Lambda_1.4` | `841.4699560289916` | `841.4699533584725` |

These are local verification results, not repository fixtures or global
physical-error bounds.

[`../tests/cfl_reference_solver.py`](../tests/cfl_reference_solver.py) is an
explicitly gated, no-production-EoS-import reference implementation using an
independent enthalpy/DOP853/QUADPACK route. For the undeformed strict baseline,
its recorded relative differences from production were below `2.5e-11` for
the compared `M_max`, `R_1.4`, `k2_1.4`, and `Lambda_1.4` values. This is an
implementation cross-check, not independent authorship or agreement with a
published convention-matched table.

Strict thermodynamic derivative residuals decreased across the three grids.
The aggregate residual status remained
`mixed_or_nonmonotone_refinement` because algebraic identities at the
binary64 floor were not strictly monotone. Validation passed without changing
that recorded status.

### Workflow, packet, and packaging checks

Routine tests cover:

- passive planning with zero writes and solver calls;
- the two-part execution gate;
- quick/strict notebook passivity from supported working directories;
- complete packet manifests, source/configuration identities, and read-only
  loading/validation;
- unavailable/rejected status preservation;
- saved-table reporting with no solver calls; and
- clean-wheel imports, packaged CFL/BSk24 source manifests, and packet
  portability outside the checkout.

The CI workflow performs these software checks without launching a real
expensive stellar campaign.

## Dataset route status

[`../notebooks/cfl_dataset.ipynb`](../notebooks/cfl_dataset.ipynb) and the
experimental `dataset_40` profile have focused passivity, planning, packet,
validation, and seven-file saved-table reporting coverage. Those tests verify
the software contract only.

No matched executed `dataset_40` versus CFL `strict` qualification is included
in the release. Dataset output must not be presented as STRICT convergence
evidence. See [Dataset workflows](dataset.md#qualification-boundary).

## What remains outside the evidence

The current evidence does not establish:

- that the frozen bag-model parameters are uniquely preferred by QCD or
  observation;
- the external two-flavor ordinary-nuclei stability condition;
- a crust, hadronic envelope, hybrid branch, or extended CFL radial
  diagnostics;
- microscopic composition for a deformed effective barotrope;
- a convention-matched comparison to a published pure-CFL stellar sequence;
  or
- independent-solver and matched strict-convergence coverage across the
  nonzero deformation domain intended for a publication.

Publication-level CFL mass-radius, maximum-mass, or tidal claims therefore
require a reviewed strict study over the claimed domain, a matching published
benchmark, and independent-solver comparison for the relevant nonzero
deformations. Do not regenerate an expected table from production and call it
independent validation.

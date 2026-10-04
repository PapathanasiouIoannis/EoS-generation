# BSk25 sources and validation

## Selection and authority

Set `matter_model="bsk25"` in the two-cell notebook, or use
`configs/bsk25.json`. The existing `bsk24-trial` command handles both models.
Use `epsilon_match="standard"` when switching models: it selects n_B=0.16
fm^-3 using that model's C1 total energy. For BSk25 this is
152.41651525155257 MeV fm^-3. A numeric anchor must lie strictly between
81.14923220504647 and 2137.2532580494963 MeV fm^-3. An anchor of 80 is
therefore invalid for BSk25 and is never silently changed.

Production pressure and its analytical derivative use equation C4 and the
BSk25 column of Table C2 in [Pearson et al. (2018), corrected 2019](https://arxiv.org/abs/1903.04981).
C1 and Table C1 establish the anchor normalization, including rest-mass energy.
The HTML rendering numbers these equations 75/78 and tables 20/21. The
existing unit conversion, constants, solvers, governed profiles and physical
predicates are preserved. No model is blended or substituted at runtime.

The direct BSk25 domain is rho=10^6 to 3.81e15 g cm^-3; the latter is the
published rounded causality limit. Its unmodified derivative is
0.9997865428384274 there. Nonzero proposals retain the existing complete raw
assessment through rho=10^16, followed by their own first causal endpoint.
The standard anchor, core-entry threshold, muon-onset label and sensitive
diagnostic bands are model-specific. Phase labels are source metadata and do
not establish microscopic composition for deformed effective barotropes.

## Source discrepancies

The source manifest records retrieval on 2026-10-03 and exact artifact/member
hashes. Three distinctions are explicit:

- Paper Table C1 has p8=2.54; the authors' `bskfit18.f` revision 2023-02-13
  uses 2.31 for BSk25. BSk25 production follows the paper.
- The paper's C1 exponent is exactly 7/6. The Fortran implementation uses
  1.16667 and a rounded mass-density conversion. BSk25 follows the exact
  exponent and the package's existing unit conversion. BSk24's previously
  governed representation remains unchanged.
- The fetched [CompOSE BSk25 archive](https://compose.obspm.fr/eos/257) has
  SHA256 `c72690daa670f91390382f22400417cc9302dd724b1aa7d70529f1dd000894b1`,
  which differs from its advertised checksum sidecar. Two archive fetches
  agreed; eos.nb, eos.thermo, eos.compo and eos.mr were also fetched individually
  and matched their archive members byte for byte. This is recorded as a
  discrepancy, not a successful sidecar verification.

The Ioffe source's hash matches the previously pinned artifact. No compiled
Fortran run is claimed. Its C4 coefficients agree with the paper; its C1
difference prevents treating it as an identical energy oracle. Full source
archives, PDFs and generated validation results remain outside Git.

## Independent equation references

`tests/fixtures/bsk25_contract_v1/reference.py` has no production imports.
It evaluates the paper equations in 60-digit Decimal arithmetic and takes a
symmetric numerical C4 derivative with step 1e-20 in log10 rho. Its immutable
JSON contains ten pressure/derivative nodes, seven C1 nodes, selected CompOSE
rows, source hashes and published stellar benchmarks. A checksum manifest
seals the reference; fixture creation refuses an existing destination.

The float64 comparisons allow 5e-13 relative error for pressure and sound
speed, with 5e-14 absolute sound-speed allowance, and 5e-14 relative error
for C1 total energy. These bounds cover arithmetic evaluation against the
independent reference. They do not measure the analytical fit's accuracy
against the underlying microscopic table. Inversions allow 5e-12 relative
error under the retained governed root tolerances. Tests also check causal
positivity, zero-amplitude identity, first-law/Euler closure, rejected-case
exclusion, settings round trips, separate identities and passive planning.

The current CompOSE rows differ from the analytical representation: the
sampled C4 pressure difference reaches +6.4851% at core entry; sampled C1
total-energy differences range from -0.1154% to +0.3501%. The paper describes
its original C4 fit errors as typically about 1%, reaching about 4% at phase
boundaries. Those limits are not demonstrated for today's CompOSE download.
Its README includes later references; why this current table differs from
the original fit remains an open source-comparison question. Production
implements the source-pinned published fit and does not repair it to match
the current table.

## Bounded stellar checks

Four undeformed configurations were evaluated with the existing production
surface boundary, at rho_c=7.46e14 and 2.26e15 g cm^-3, with the existing
rtol/atol pairs 1e-6/1e-8 and 1e-8/1e-10. These are fixed central densities
from Pearson Tables 17/16, not fixed-mass roots or a maximum search.

| Central density | Tighter mass (M_sun) | Tighter radius (km) | Lambda | k2 |
|---|---:|---:|---:|---:|
| 7.46e14 g cm^-3 | 1.398852665 | 12.400075579 | 490.470333623 | 0.094353179 |
| 2.26e15 g cm^-3 | 2.228330812 | 11.034804570 | 5.788866972 | 0.020467945 |

The published comparison values are (1.4 M_sun, 12.37 km) and
(2.224 M_sun, 11.05 km). The respective offsets are
(-0.001147335 M_sun, +0.030075579 km) and
(+0.004330812 M_sun, -0.015195430 km). They are recorded observations;
no scientific tolerance was weakened or introduced to label them exact.
In particular, this does not establish the paper's quoted fit/table maximum
mass agreement or resolve a turning point.

Between the two ODE tolerances, mass changes were below 1.5e-7 M_sun,
radius changes below 0.0106 km, and relative Lambda changes below 9e-6.
Both densities returned the existing validated tidal-framework status.
This is model-routing/convergence evidence; no independent published
numerical BSk25 tidal table was available for a scalar comparison.
Existing analytical tidal benchmarks remain in the regression suite.
Scientific use still requires appropriate convergence of the selected
profile, reconstruction and requested stellar products.

The reproducible, opt-in check performs four star solves and refuses an
existing output directory:

```powershell
python -B tests/validate_bsk25_stellar.py --output runs/bsk25-stellar-check --execute
```

Development checks used the pinned `eos-generation` environment with the
active worktree's `src` on PYTHONPATH:

```powershell
python -B -m pytest -q tests/test_bsk25.py
python -B -m pytest -q
```

The full suite covers both models' passive two-cell notebook execution,
saved plotting without solver calls, legacy BSk24 identities/fixtures, source
archives and schema/hygiene checks. No raw third-party archive or generated
scientific packet is included in the package.

Final source and isolated installed-wheel runs each passed 109 tests and three
subtests. All eighteen precision/calculation profile expansions matched the
prior simplified version exactly. Packaging and installed CLI planning also
passed; BSk24 fixture data were preserved from the original Git authority.

# Independent BSk25 reference, version 1

`reference.json` was created by `reference.py`, an independent 60-digit Decimal
implementation of Pearson Appendix C1/C4, using the paper parameter order and
a symmetric numerical C4 derivative (step 1e-20 in log10 rho). It imports no
production code. The fixture also extracts selected CompOSE rows and source
hashes. It is sealed by SHA256SUMS.txt and must never be regenerated from the
implementation under test.

Authorities: [corrected Pearson paper](https://arxiv.org/abs/1903.04981),
[CompOSE BSk25](https://compose.obspm.fr/eos/257), and the source-pinned
[Ioffe routine](http://www.ioffe.ru/astro/NSG/BSk/bskfit18.f).
Paper C1 p8=2.54 and exact 7/6 are authoritative here. The Ioffe routine uses
2.31 and 1.16667. The downloaded CompOSE ZIP differs from its checksum sidecar;
its relevant members were verified against separately downloaded files.
These differences are recorded explicitly in the JSON.

Initial creation command (refuses an existing destination):

```powershell
python -B tests/fixtures/bsk25_contract_v1/reference.py runs/bsk25-validation-sources tests/fixtures/bsk25_contract_v1/reference.json
```

C4 pressure and derivative comparisons allow 5e-13 relative error (sound speed
also 5e-14 absolute); C1 total energy allows 5e-14 relative error. These bounds
cover float64 equation evaluation versus Decimal, not scientific fit error.
The rho=1e16 derivative is deliberately superluminal and belongs only to the
published-fit evaluator. Production direct use ends at rho=3.81e15 g/cm^3.
CompOSE rows are source observations, not exact analytical-fit expectations.
See docs/bsk25-validation.md for fit differences and stellar validation scope.

# BSk24 public regression fixture, version 1

This is a deliberately small, read-only regression fixture. It is not a
scientific result packet and is never used at runtime. Its values were selected
from source-pinned, previously validated artifacts; they were not regenerated
by the implementation being tested.

Only contracts exercised by the public test suite are retained:

- `thermodynamic_rows.csv`: three direct analytical BSk24 pressure and sound-
  speed rows at retained energy-density nodes.
- `lifecycle_contract.json`: the exact passive quickstart configuration hash,
  case order, and work estimates.
- `provenance.json`: archival source identifiers, hashes, units, selection,
  and comparison policies for those two files.
- `SHA256SUMS.txt`: exact coverage of every other file in this directory.

Large validation packets, figures, profiles, campaigns, and unused historical
contracts are intentionally absent from the public repository.

"""Opt-in bounded BSk25 benchmark; never collected by pytest."""
import argparse
from dataclasses import asdict
from pathlib import Path

from eos_generation.baseline import make_baseline_eos
from eos_generation.numerics import BSk24TrialConfig
from eos_generation.stellar import _tov_settings
from eos_generation.tov import solve_star
from eos_generation.storage import (
    environment_identity, hash_payload, seal, source_identity, write_json_atomic,
)


def main():
    parser = argparse.ArgumentParser(description="Four undeformed BSk25 star solves; no sweep or maximum search.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    if "runs" not in destination.parts:
        parser.error("validation evidence must be below a runs directory")
    if not args.execute:
        parser.error("review the four-solve cost and output path, then supply --execute")
    # Never overwrite an existing benchmark; refuse before scientific work.
    destination.mkdir(parents=True, exist_ok=False)
    try:
        eos = make_baseline_eos("bsk25")
        config = BSk24TrialConfig(matter_model="bsk25")
        settings = _tov_settings(eos, config, config.tov_stages[-1])
        choices = {"matter_model": "bsk25", "rho_g_cm3": [7.46e14, 2.26e15],
                   "tolerances": [[1e-6, 1e-8], [1e-8, 1e-10]], "calculate_tidal": True}
        record = {"schema_id": "bsk25_stellar_validation_v1", "settings": choices,
                  "settings_hash": hash_payload(choices), "source_identity": source_identity(),
                  "environment_identity": environment_identity(), "baseline": eos.provenance(),
                  "expanded_solver_settings": asdict(settings),
                  "reproduction": "python -B tests/validate_bsk25_stellar.py --output runs/bsk25-stellar-reproduction --execute"}
        rows = []
        for rho, label in [(7.46e14, "published_1p4_central_density"), (2.26e15, "published_maximum_central_density")]:
            for rtol, atol in [(1e-6, 1e-8), (1e-8, 1e-10)]:
                star = solve_star(eos, float(eos.pressure_from_mass_density(rho)),
                                  settings=settings, rtol=rtol, atol=atol,
                                  calculate_tidal=True, retain_profile=False)
                row = dict(label=label, rho_g_cm3=rho, rtol=rtol, atol=atol,
                           mass_msun=star.mass, radius_km=star.radius,
                           lambda_dimensionless=star.lambda_dimensionless,
                           k2=star.lambda_diagnostic.k2,
                           tidal_status=star.lambda_diagnostic.scientific_status,
                           failure_reason=star.lambda_diagnostic.failure_reason)
                rows.append(row)
                print(row, flush=True)
        write_json_atomic({**record, "status": "complete", "rows": rows}, destination / "benchmarks.json")
        seal(destination)
    except BaseException as exc:
        write_json_atomic({"status": "failed", "failure": str(exc)}, destination / "failure.json")
        raise


if __name__ == "__main__":
    main()

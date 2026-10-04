"""Command adapter for the single experiment workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _settings(path):
    from .settings import ExperimentSettings
    from .storage import RUN_SCHEMA, strict_json

    payload = strict_json(path)
    return (
        ExperimentSettings.from_dict(payload["settings"])
        if payload.get("schema_id") == RUN_SCHEMA
        else ExperimentSettings.from_json(path)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="bsk24-trial",
        description="Plan, execute and inspect controlled BSk24 or BSk25 experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--output", "--output-path", dest="output", type=Path)
        p.add_argument("--json", action="store_true")
        if name == "run":
            p.add_argument("--plan-hash", required=True)
            p.add_argument("--execute", action="store_true")
    for name in ("status", "validate"):
        p = commands.add_parser(name)
        p.add_argument("path", type=Path)
        p.add_argument("--json", action="store_true")
        if name == "validate":
            p.add_argument("--exact-source", action="store_true")
    p = commands.add_parser("plot")
    p.add_argument("path", type=Path)
    p.add_argument("--figures", nargs="+", default=["auto"])
    p.add_argument("--output", type=Path)
    p.add_argument("--geometry", type=int, nargs="+")
    p.add_argument("--amplitudes", type=float, nargs="+")
    p.add_argument("--fixed-masses", type=float, nargs="+")
    p.add_argument("--stage", default="final")
    p.add_argument("--lambda-scale", choices=("log", "linear"), default="log")
    p.add_argument("--lambda-mass-limits", type=float, nargs=2, metavar=("MIN", "MAX"))
    p.add_argument("--lambda-limits", type=float, nargs=2, metavar=("MIN", "MAX"))
    p.add_argument("--regenerate", action="store_true")
    p = commands.add_parser("import-legacy")
    p.add_argument("path", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        from .experiment import plan_experiment, run_experiment
        from .storage import strict_json, validate_run

        if args.command in {"plan", "run"}:
            plan = plan_experiment(_settings(args.config), output_path=args.output)
            if args.command == "plan":
                print(
                    json.dumps(plan.to_dict(), indent=2, allow_nan=False)
                    if args.json
                    else plan.summary_text()
                )
            else:
                if not args.execute:
                    raise ValueError("run requires --execute after reviewing the plan")
                if args.plan_hash != plan.plan_hash:
                    raise ValueError(
                        "reviewed plan hash differs; make and review a fresh plan"
                    )
                result = run_experiment(plan, execute=True)
                print(
                    json.dumps(
                        {
                            "path": result.experiment_path.as_posix(),
                            "status": result.metadata["status"],
                        }
                    )
                    if args.json
                    else str(result.experiment_path)
                )
        elif args.command in {"validate", "status"}:
            report = validate_run(
                args.path.resolve(),
                require_source_equivalence=getattr(args, "exact_source", False),
            )
            if args.json:
                print(json.dumps(report, indent=2, allow_nan=False))
            else:
                print(
                    f"Status: {report['status']}; validation: {'pass' if report['passed'] else 'fail'}; source: {report['source_equivalence']}"
                )
                for error in report["errors"]:
                    print(error)
            return 0 if report["passed"] else 1
        elif args.command == "plot":
            from .plotting import plot_experiment

            print(
                plot_experiment(
                    args.path,
                    figures=args.figures,
                    output_path=args.output,
                    geometry=args.geometry,
                    amplitudes=args.amplitudes,
                    fixed_masses=args.fixed_masses,
                    stage=args.stage,
                    lambda_scale=args.lambda_scale,
                    lambda_mass_limits=args.lambda_mass_limits,
                    lambda_limits=args.lambda_limits,
                    regenerate=args.regenerate,
                )
            )
        else:
            from .storage import import_legacy

            print(import_legacy(args.path, args.output).experiment_path)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"bsk24-trial: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

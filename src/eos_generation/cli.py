"""Command-line front door for governed equation-of-state experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


class _UsageError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="bsk24-trial",
        description=(
            "Plan, run, inspect, and validate controlled BSk24 or CFL "
            "experiments."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="passively preview a configuration")
    _config_arguments(plan)

    run = subparsers.add_parser("run", help="execute an explicitly authorized plan")
    _config_arguments(run)
    run.add_argument(
        "--execute",
        action="store_true",
        help="required scientific-execution authorization gate",
    )
    run.add_argument(
        "--plan-hash",
        required=True,
        help="required hash copied from the reviewed passive plan",
    )

    status = subparsers.add_parser("status", help="summarize a saved experiment")
    status.add_argument("experiment", type=Path)
    status.add_argument("--json", action="store_true", dest="as_json")

    validate = subparsers.add_parser(
        "validate", help="read-only validation of a saved experiment"
    )
    validate.add_argument("experiment", type=Path)
    validate.add_argument("--json", action="store_true", dest="as_json")

    plot = subparsers.add_parser(
        "plot", help="inspect plots; regenerate only with --overwrite"
    )
    plot.add_argument("experiment", type=Path)
    plot.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly authorize replacement of existing plots",
    )
    plot.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "base directory inside the checkout-local runs/ tree "
            "(default: the checkout's runs/)"
        ),
    )
    parser.add_argument("--json", action="store_true", dest="as_json")


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _result_payload(result: Any) -> dict[str, Any]:
    inventory = result.plot_inventory
    return {
        "schema_id": "eos_generation_result_summary_v1",
        "status": "complete" if result.completed else "incomplete",
        "experiment_path": str(result.experiment_path),
        "child_packets": [str(path) for path in result.packet_paths],
        "accepted_cases": len(result.accepted_cases),
        "rejected_cases": len(result.rejected_cases),
        "figures": [str(path) for path in result.figures],
        "plot_inventory": (
            json.loads(inventory.to_json(orient="records"))
            if not inventory.empty
            else []
        ),
    }


def _plan(args: argparse.Namespace) -> int:
    from .experiment import ExperimentSettings, plan_experiment

    settings = ExperimentSettings.from_json(args.config)
    plan = plan_experiment(settings, output_root=args.output_root)
    if args.as_json:
        _json(plan.to_dict())
    else:
        print(plan.summary_text())
        print("\nNext step after review:")
        command = (
            f'bsk24-trial run --config "{args.config}" --execute '
            f"--plan-hash {plan.plan_hash}"
        )
        if args.output_root is not None:
            command += f' --output-root "{args.output_root}"'
        print(command)
    return 0


def _run(args: argparse.Namespace) -> int:
    from .experiment import ExperimentSettings, plan_experiment, run_experiment

    if args.execute is not True:
        raise _UsageError("run requires --execute after the passive plan was reviewed")
    settings = ExperimentSettings.from_json(args.config)
    plan = plan_experiment(settings, output_root=args.output_root)
    if args.plan_hash != plan.plan_hash:
        raise _UsageError(
            "--plan-hash does not match the current configuration, environment, "
            "or destination; run plan again"
        )
    result = run_experiment(plan, execute=True)
    if args.as_json:
        _json(_result_payload(result))
    else:
        print(result.summary_text())
    return 0


def _status(args: argparse.Namespace) -> int:
    from .experiment import load_experiment

    result = load_experiment(args.experiment)
    if args.as_json:
        _json(_result_payload(result))
    else:
        print(result.summary_text())
    return 0 if result.completed else 1


def _validate(args: argparse.Namespace) -> int:
    from .experiment import validate_experiment

    report = validate_experiment(args.experiment)
    if args.as_json:
        _json(report)
    else:
        print(f"EoS experiment validation: {str(report['status']).upper()}")
        print(f"Experiment: {report['experiment_path']}")
        print(f"Child packets: {report['child_packet_count']}")
        if report["status"] != "pass":
            for failure in list(report.get("failures", []))[:5]:
                print(f"Failure: {failure}")
            for index, child in enumerate(report.get("children", []), start=1):
                child_status = child.get("status", child.get("overall_status", "unknown"))
                if child_status in {"pass", "complete", "validated"}:
                    continue
                label = child.get("packet_path", child.get("output_path", f"child {index}"))
                failures = list(child.get("failures", child.get("errors", [])))
                if failures:
                    for failure in failures[:3]:
                        print(f"{label}: {failure}")
                else:
                    print(f"{label}: validation status {child_status}")
            print("Use --json for the complete validation report.")
    return 0 if report["status"] == "pass" else 1


def _plot(args: argparse.Namespace) -> int:
    from .experiment import load_experiment

    result = load_experiment(args.experiment)
    if args.overwrite:
        result = result.plot(overwrite=True)
    payload = _result_payload(result)
    if args.as_json:
        _json(payload)
    else:
        inventory = result.plot_inventory
        print(f"Plot inventory records: {len(inventory)}")
        for row in json.loads(inventory.to_json(orient="records")):
            reason = row.get("reason") or ""
            print(
                f"geometry {row['geometry_index']}: {row.get('figure')} - "
                f"{row.get('status')}"
                + (f" ({reason})" if reason else "")
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        handlers = {
            "plan": _plan,
            "run": _run,
            "status": _status,
            "validate": _validate,
            "plot": _plot,
        }
        return handlers[args.command](args)
    except _UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileExistsError, FileNotFoundError, PermissionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # scientific/runtime failures remain visible
        print(f"execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

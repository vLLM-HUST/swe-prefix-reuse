"""Offline workload research only; never connects to an inference endpoint."""

import argparse
import json
from pathlib import Path

from swe_prefix_reuse.prepare import read_json
from swe_prefix_reuse.session_arrivals import build_plan, simulate
from swe_prefix_reuse.session_audit import audit
from swe_prefix_reuse.session_profile import build_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile")
    profile.add_argument("--source", required=True)
    profile.add_argument("--output", required=True, help="new deterministic gzip file")
    plan = commands.add_parser("plan")
    plan.add_argument("--profile", required=True)
    plan.add_argument("--provider", required=True, choices=["claude", "codex"])
    plan.add_argument("--rate", type=float, required=True, help="new sessions/second")
    plan.add_argument("--duration", type=float, required=True)
    plan.add_argument("--seed", type=int, default=20261002)
    plan.add_argument("--pattern", choices=["poisson", "constant"], default="poisson")
    plan.add_argument("--max-sessions", type=int, default=100000)
    plan.add_argument("--max-calls", type=int, default=2000000)
    plan.add_argument("--output", required=True)
    check = commands.add_parser("audit")
    check.add_argument("--profile", required=True)
    check.add_argument("--max-context", type=int, default=262144)
    check.add_argument("--output", required=True)
    preview = commands.add_parser("preview")
    preview.add_argument("--plan", required=True)
    preview.add_argument("--service-seconds", type=float, required=True)
    preview.add_argument("--workers", type=int, required=True)
    preview.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    output = args.pop("output")
    if command == "profile":
        result = build_profile(output=output, **args)
    elif command == "plan":
        args["profile_path"] = args.pop("profile")
        result = build_plan(output=output, **args)
    else:
        if command == "audit":
            result = audit(read_json(args["profile"]), args["max_context"])
        else:
            result = simulate(
                read_json(args["plan"]),
                service_seconds=args["service_seconds"],
                workers=args["workers"],
            )
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with Path(output).open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()

"""Opt-in arrival workload commands; legacy closed-loop run remains unchanged."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .arrival_plan import build, digest, validate_plan
from .arrival_runner import replay
from .prepare import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--workload", required=True)
    prepare.add_argument("--profile", required=True)
    prepare.add_argument("--provider", choices=["claude", "codex"], required=True)
    prepare.add_argument("--rate", type=float, required=True)
    prepare.add_argument("--duration", type=float, required=True)
    prepare.add_argument("--seed", type=int, default=20261002)
    prepare.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--workload", required=True)
    run.add_argument("--plan", required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--chips", type=int, required=True)
    run.add_argument("--connections", type=int, default=256)
    run.add_argument("--timeout", type=float, default=1800)
    run.add_argument("--data-parallel-size", type=int)
    run.add_argument("--server-max-context", type=int, required=True)
    run.add_argument("--server-metadata")
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            json.dumps(
                build(
                    args.workload,
                    args.profile,
                    args.output,
                    provider=args.provider,
                    rate=args.rate,
                    duration=args.duration,
                    seed=args.seed,
                ),
                indent=2,
            )
        )
        return
    url = urlsplit(args.endpoint)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or not url.path.endswith("/v1/completions")
    ):
        raise ValueError("endpoint must be a credential-free http(s) /v1/completions URL")
    workload, plan = read_json(args.workload), read_json(args.plan)
    validate_plan(workload, plan)
    if digest(args.workload) != plan["workload_sha256"]:
        raise ValueError("prepared workload digest differs from frozen arrival plan")
    if args.server_max_context < max(s["max_context_tokens"] for s in workload["sessions"]):
        raise ValueError("workload exceeds declared server context")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    config = {
        **vars(args),
        "tool_version": __version__,
        "run_id": run_id,
        "plan_sha256": digest(args.plan),
        "workload_sha256": digest(args.workload),
        "tokenizer": workload.get("tokenizer"),
        "server_metadata": read_json(args.server_metadata) if args.server_metadata else None,
        "policy": plan["semantics"],
        "window": "stop new dispatch at deadline; drain in-flight; "
        "sleeping sessions remain incomplete, not finished; cold empty initial pool",
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    try:
        with (root / "requests.jsonl").open("w") as stream:
            summary, _, samples = asyncio.run(
                replay(
                    workload,
                    plan,
                    url=args.endpoint,
                    model=args.model,
                    chips=args.chips,
                    connections=args.connections,
                    timeout=args.timeout,
                    data_parallel_size=args.data_parallel_size,
                    run_id=run_id,
                    record_sink=lambda r: stream.write(json.dumps(r, separators=(",", ":")) + "\n"),
                )
            )
        (root / "client-samples.json").write_text(json.dumps(samples) + "\n")
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    except BaseException as exc:
        (root / "error.json").write_text(json.dumps({"valid": False, "error": type(exc).__name__}))
        raise
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()

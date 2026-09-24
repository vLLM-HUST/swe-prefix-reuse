import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="SWE prefix reuse: fixed shape, real generated history"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="compile source trajectories for a local tokenizer"
    )
    prepare_parser.add_argument("--source", required=True)
    prepare_parser.add_argument("--tokenizer", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--max-context", type=int, required=True)
    run_parser = commands.add_parser(
        "run", help="measure a running vLLM-compatible completions endpoint"
    )
    run_parser.add_argument("--workload", required=True)
    run_parser.add_argument("--endpoint", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--concurrency", type=int, required=True)
    run_parser.add_argument("--duration", type=float, default=60)
    run_parser.add_argument("--chips", type=int, required=True)
    run_parser.add_argument(
        "--data-parallel-size",
        type=int,
        help="Opt-in native vLLM rank affinity: lane modulo DP size; must match the server",
    )
    run_parser.add_argument("--server-max-context", type=int, required=True)
    run_parser.add_argument(
        "--server-metadata",
        help="JSON: engine revisions, model revision, precision, hardware, command",
    )
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--timeout", type=float, default=1800)
    run_parser.add_argument(
        "--output", required=True, help="new result directory; refuses overwrite"
    )
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare

        print(
            json.dumps(
                prepare(args.source, args.tokenizer, args.output, args.max_context), indent=2
            )
        )
    else:
        from .runner import run

        raise SystemExit(run(args))


if __name__ == "__main__":
    main()

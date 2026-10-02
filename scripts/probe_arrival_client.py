"""Bounded CPU-only HTTP/SSE protocol probe; local server is not an inference engine."""

import argparse
import asyncio
import json
from pathlib import Path

from aiohttp import web

from swe_prefix_reuse.arrival_plan import SCHEMA
from swe_prefix_reuse.arrival_runner import replay
from swe_prefix_reuse.prepare import SCHEMA as WORKLOAD_SCHEMA


def event(value):
    return ("data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n").encode()


async def probe(count, rate, prompt_tokens):
    async def handler(req):
        b = await req.json()
        n = b["max_tokens"]
        s = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await s.prepare(req)
        await s.write(
            event({"choices": [{"index": 0, "prompt_token_ids": b["prompt"], "token_ids": []}]})
        )
        await asyncio.sleep(0.002)
        await s.write(
            event({"choices": [{"index": 0, "token_ids": [7] * n, "finish_reason": "length"}]})
        )
        await s.write(
            event(
                {
                    "choices": [],
                    "usage": {"prompt_tokens": len(b["prompt"]), "completion_tokens": n},
                }
            )
        )
        await s.write(event("[DONE]"))
        await s.write_eof()
        return s

    app = web.Application(client_max_size=4 * 1024 * 1024)
    app.router.add_post("/v1/completions", handler)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    duration = count / rate
    workload = {
        "schema": WORKLOAD_SCHEMA,
        "max_context": prompt_tokens + 10,
        "sessions": [
            {
                "trajectory_id": f"fixture-{i}",
                "max_context_tokens": prompt_tokens + 9,
                "turns": [
                    {
                        "input_ids": [10] * prompt_tokens,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": 4,
                    },
                    {"input_ids": [11], "prompt_tokens": prompt_tokens + 5, "output_tokens": 4},
                ],
            }
            for i in range(count)
        ],
    }
    plan = {
        "schema": SCHEMA,
        "seed": 0,
        "duration_s": duration,
        "new_sessions_per_second": rate,
        "sessions": [
            {
                "trajectory_id": f"fixture-{i}",
                "arrival_s": i / rate,
                "delay_after_turn_s": [duration + 10],
            }
            for i in range(count)
        ],
    }
    try:
        summary, _, samples = await replay(
            workload,
            plan,
            url=f"http://127.0.0.1:{port}/v1/completions",
            model="protocol-fixture",
            chips=1,
        )
    finally:
        await server.cleanup()
    assert summary["valid"] and summary["launched_sessions"] == count, summary
    assert summary["incomplete_sessions_after_drain"] == count
    return {
        "scope": "Protocol only: fixture server shares this event loop/CPU; "
        "NOT standalone client capacity or model throughput",
        "sessions": count,
        "rate": rate,
        "prompt_tokens": prompt_tokens,
        "peak_sampled_waiting": max(s["waiting"] for s in samples),
        "summary": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=100)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not (
        1 <= args.sessions <= 5000 and 0 < args.rate <= 1000 and 1 <= args.prompt_tokens <= 32768
    ):
        parser.error("bounded probe: 1..5000 sessions, rate<=1000, 1..32768 prompt tokens")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        result = asyncio.run(probe(args.sessions, args.rate, args.prompt_tokens))
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

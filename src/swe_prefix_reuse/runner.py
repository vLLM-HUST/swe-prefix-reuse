"""Exactly C closed-loop lanes; no arrival schedule, tool delays or turn queue."""

import asyncio
import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

from . import __version__
from .client import request
from .prepare import read_json, validate_workload


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize(records, start, stop, concurrency, chips):
    duration = stop - start
    completed = [r for r in records if r["success"] and r["end"] <= stop]
    valid = bool(records) and all(r["success"] for r in records)
    tokens = sum(
        count for r in records for timestamp, count in r["chunks"] if start <= timestamp < stop
    )
    events = [(start, 0), (stop, 0)]
    for row in records:
        lo, hi = max(start, row["start"]), min(stop, row["end"])
        if hi > lo:
            events.extend([(lo, 1), (hi, -1)])
    occupancy = {str(i): 0.0 for i in range(concurrency + 1)}
    active, previous = 0, start
    for timestamp, change in sorted(events):
        occupancy[str(active)] += timestamp - previous
        active += change
        previous = timestamp
    speeds = [
        r["decode_tokens_per_second"]
        for r in completed
        if r["decode_tokens_per_second"] is not None
    ]
    return {
        "valid": valid,
        "measurement_seconds": duration,
        "requests_started": len(records),
        "max_prompt_tokens_observed": max((r.get("prompt_tokens", 0) for r in records), default=0),
        "prompt_tokens_p50": percentile([r.get("prompt_tokens", 0) for r in records], 0.5),
        "max_turn_index_reached": max((r.get("turn", 0) for r in records), default=0),
        "distinct_trajectories_started": len({r.get("trajectory_id") for r in records}),
        "sessions_completed": sum(r["success"] and r.get("last_turn", False) for r in records),
        "requests_completed_in_window": len(completed),
        "requests_drained": sum(r["end"] > stop for r in records),
        "failed_requests": sum(not r["success"] for r in records),
        "observed_output_tokens_in_window": tokens,
        "output_tokens_per_second": tokens / duration if valid else None,
        "output_tokens_per_second_per_chip": tokens / duration / chips if valid else None,
        "decode_tokens_per_second_p90": percentile(speeds, 0.9) if valid else None,
        "decode_speed_samples": len(speeds),
        "ttft_seconds_p95": percentile([r["ttft_seconds"] for r in completed], 0.95)
        if valid
        else None,
        "client_inflight_seconds": occupancy,
        "mean_client_inflight": sum(int(k) * v for k, v in occupancy.items()) / duration,
        "full_concurrency_fraction": occupancy[str(concurrency)] / duration,
        "drain_seconds": max([stop] + [r["end"] for r in records]) - stop,
    }


async def replay(
    workload, *, url, model, concurrency, duration, chips, seed=0, timeout=1800, run_id=None
):
    validate_workload(workload)
    if (
        concurrency < 1
        or chips < 1
        or duration <= 0
        or timeout <= 0
        or not math.isfinite(duration)
        or not math.isfinite(timeout)
    ):
        raise ValueError("concurrency, chips, duration and timeout must be positive")
    run_id = run_id or uuid.uuid4().hex
    records, aborted = [], asyncio.Event()
    next_session = 0
    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    async with aiohttp.ClientSession(
        connector=connector, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as http:
        start = time.perf_counter()
        stop = start + duration

        async def lane(lane_id):
            nonlocal next_session
            play = 0
            while time.perf_counter() < stop and not aborted.is_set():
                session_index = next_session % len(workload["sessions"])
                next_session += 1
                trace = workload["sessions"][session_index]
                # A fresh salt for each play prevents false hits on recycled source traces.
                salt = f"{run_id}:{lane_id}:{play}"
                prompt = []
                for turn_index, turn in enumerate(trace["turns"]):
                    if time.perf_counter() >= stop or aborted.is_set():
                        return
                    prompt.extend(turn["input_ids"])
                    if len(prompt) != turn["prompt_tokens"]:
                        raise ValueError("runtime prompt no longer matches compiled shape")
                    output = await request(
                        http,
                        url,
                        model,
                        prompt,
                        turn["output_tokens"],
                        salt,
                        seed + session_index + turn_index,
                    )
                    row = output.record(prompt)
                    row.update(
                        lane=lane_id,
                        play=play,
                        trajectory_id=trace["trajectory_id"],
                        turn=turn_index,
                        expected_output_tokens=turn["output_tokens"],
                        last_turn=(turn_index == len(trace["turns"]) - 1),
                    )
                    records.append(row)
                    if not output.success:
                        aborted.set()
                        return
                    prompt.extend(output.token_ids)
                play += 1

        tasks = [asyncio.create_task(lane(i)) for i in range(concurrency)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    # On failure the planned window may not finish; do not claim a full measurement.
    actual_stop = min(stop, time.perf_counter())
    summary = summarize(records, start, actual_stop, concurrency, chips)
    summary["aborted"] = aborted.is_set()
    summary["planned_measurement_seconds"] = duration
    # Store relative monotonic times; never imply these are wall-clock timestamps.
    for row in records:
        for key in ("start", "end", "first_token", "last_token"):
            if row[key] is not None:
                row[key] -= start
        row["chunks"] = [[timestamp - start, count] for timestamp, count in row["chunks"]]
    return summary, records


def run(args):
    parsed = urlsplit(args.endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/v1/completions")
    ):
        raise ValueError("endpoint must be a credential-free http(s) /v1/completions URL")
    workload = read_json(args.workload)
    validate_workload(workload)
    if args.server_max_context < max(s["max_context_tokens"] for s in workload["sessions"]):
        raise ValueError("selected workload exceeds declared server context capacity")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    config = {
        "tool_version": __version__,
        "schema": workload["schema"],
        "run_id": run_id,
        "started_at_unix": time.time(),
        "endpoint": args.endpoint,
        "model": args.model,
        "workload_sha256": hashlib.sha256(Path(args.workload).read_bytes()).hexdigest(),
        "tokenizer": workload["tokenizer"],
        "concurrency": args.concurrency,
        "duration": args.duration,
        "chips": args.chips,
        "seed": args.seed,
        "timeout": args.timeout,
        "server_max_context": args.server_max_context,
        "cache_policy": "unique salt per session play; reused within session; no cross-session reuse",
        "window_policy": "cold-start fixed duration; count streamed tokens in window; drain in-flight only",
        "server_metadata": read_json(args.server_metadata) if args.server_metadata else None,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    try:
        summary, records = asyncio.run(
            replay(
                workload,
                url=args.endpoint,
                model=args.model,
                concurrency=args.concurrency,
                duration=args.duration,
                chips=args.chips,
                seed=args.seed,
                timeout=args.timeout,
                run_id=run_id,
            )
        )
    except BaseException as exc:
        (output / "error.json").write_text(
            json.dumps({"valid": False, "error": type(exc).__name__})
        )
        raise
    with (output / "requests.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 1

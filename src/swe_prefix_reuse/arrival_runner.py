"""One task per live session, independent arrival clock, completion-relative waits."""

import asyncio
import json
import resource
import time
import uuid
from contextvars import ContextVar

import aiohttp

from .arrival_plan import validate_plan
from .client import request
from .runner import percentile


def distribution(values):
    return {
        "n": len(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values, default=None),
        "sum": sum(values),
    }


async def replay(
    workload,
    plan,
    *,
    url,
    model,
    chips,
    connections=256,
    timeout=1800,
    data_parallel_size=None,
    run_id=None,
    record_sink=None,
    monitor_interval=0.05,
):
    validate_plan(workload, plan)
    if type(chips) is not int or chips < 1 or type(connections) is not int or connections < 1:
        raise ValueError("chips and connections must be positive integers")
    if not 0 < timeout < float("inf") or not 0 < monitor_interval < float("inf"):
        raise ValueError("timeout and monitor interval must be finite and positive")
    if data_parallel_size is not None and (
        type(data_parallel_size) is not int or not 1 <= data_parallel_size <= chips
    ):
        raise ValueError("invalid data_parallel_size")
    run_id = run_id or uuid.uuid4().hex
    current = ContextVar("request_measurements", default=None)
    trace = aiohttp.TraceConfig()

    async def queued(_, ctx, __):
        ctx.queued_at = time.perf_counter()

    async def dequeued(_, ctx, __):
        current.get()["connection_queue_s"] += time.perf_counter() - ctx.queued_at

    async def headers_sent(*_):
        current.get()["headers_sent_s"] = time.perf_counter() - start

    trace.on_connection_queued_start.append(queued)
    trace.on_connection_queued_end.append(dequeued)
    trace.on_request_headers_sent.append(headers_sent)

    def serialize(payload):
        before = time.perf_counter()
        value = json.dumps(payload, separators=(",", ":"))
        current.get()["serialize_s"] += time.perf_counter() - before
        return value

    aborted = asyncio.Event()
    records, tasks, samples = [], [], []
    missed_due = 0
    sink_seconds = 0.0
    states = {"waiting": 0, "inflight": 0, "complete": 0}
    start = time.perf_counter()
    stop = start + plan["duration_s"]
    cpu_start = time.process_time()

    async def wait_until(when):
        delay = min(when, stop) - time.perf_counter()
        if delay > 0:
            try:
                await asyncio.wait_for(aborted.wait(), delay)
            except asyncio.TimeoutError:
                pass
        return not aborted.is_set() and time.perf_counter() < stop

    async def monitor():
        due = start + monitor_interval
        while await wait_until(due):
            now = time.perf_counter()
            samples.append({"time_s": now - start, "loop_lag_s": max(0, now - due), **states})
            due = now + monitor_interval

    connector = aiohttp.TCPConnector(limit=connections, limit_per_host=connections)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=timeout),
        trace_configs=[trace],
        json_serialize=serialize,
    ) as http:

        async def session_task(index):
            nonlocal missed_due, sink_seconds
            trace_row = workload["sessions"][index]
            schedule = plan["sessions"][index]
            prompt = []
            salt = f"{run_id}:session:{index}"
            rank = None if data_parallel_size is None else index % data_parallel_size
            planned = start + schedule["arrival_s"]
            states["waiting"] += 1
            try:
                for turn_index, turn in enumerate(trace_row["turns"]):
                    if not await wait_until(planned):
                        if planned < stop and not aborted.is_set():
                            missed_due += 1
                        return
                    before = time.perf_counter()
                    history_tokens = len(prompt)
                    prompt.extend(turn["input_ids"])
                    if len(prompt) != turn["prompt_tokens"]:
                        raise ValueError("runtime prompt differs from compiled shape")
                    metrics = {
                        "connection_queue_s": 0.0,
                        "serialize_s": 0.0,
                        "headers_sent_s": None,
                        "prepare_s": time.perf_counter() - before,
                    }
                    current.set(metrics)
                    states["waiting"] -= 1
                    states["inflight"] += 1
                    try:
                        result = await request(
                            http,
                            url,
                            model,
                            prompt,
                            turn["output_tokens"],
                            salt,
                            plan["seed"] + index + turn_index,
                            data_parallel_rank=rank,
                        )
                    finally:
                        states["inflight"] -= 1
                        states["waiting"] += 1
                    # The wait is anchored to validated stream completion, not disk logging.
                    next_due = result.end + (
                        schedule["delay_after_turn_s"][turn_index]
                        if turn_index < len(trace_row["turns"]) - 1
                        else 0
                    )
                    before = time.perf_counter()
                    row = result.record(prompt)
                    row.pop("token_ids")
                    chunks = row.pop("chunks")
                    row.update(
                        session=index,
                        trajectory_id=trace_row["trajectory_id"],
                        turn=turn_index,
                        last_turn=turn_index == len(trace_row["turns"]) - 1,
                        data_parallel_rank=rank,
                        planned_send_s=planned - start,
                        dispatch_lateness_s=max(0, result.start - planned),
                        history_tokens=history_tokens,
                        new_input_tokens=len(turn["input_ids"]),
                        window_output_tokens=sum(n for t, n in chunks if start <= t < stop),
                        expected_output_tokens=turn["output_tokens"],
                        **metrics,
                    )
                    row["offered_ttft_s"] = (
                        None if result.first_token is None else result.first_token - planned
                    )
                    row["headers_lateness_s"] = (
                        None
                        if metrics["headers_sent_s"] is None
                        else max(0, metrics["headers_sent_s"] - (planned - start))
                    )
                    for field in ("start", "end", "first_token", "last_token"):
                        if row[field] is not None:
                            row[field] -= start
                    row["record_build_s"] = time.perf_counter() - before
                    records.append(row)
                    if record_sink:
                        sink_start = time.perf_counter()
                        record_sink(row)
                        sink_seconds += time.perf_counter() - sink_start
                    if not result.success:
                        aborted.set()
                        return
                    prompt.extend(result.token_ids)
                    planned = next_due
                states["complete"] += 1
            except Exception:
                aborted.set()
                raise
            finally:
                states["waiting"] -= 1

        monitoring = asyncio.create_task(monitor())
        try:
            for index, schedule in enumerate(plan["sessions"]):
                if not await wait_until(start + schedule["arrival_s"]):
                    break
                tasks.append(asyncio.create_task(session_task(index), name=f"session-{index}"))
            await asyncio.gather(*tasks)
            # Preserve the declared observation window even when all sessions finish early.
            if not aborted.is_set():
                await wait_until(stop)
        finally:
            monitoring.cancel()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(monitoring, *tasks, return_exceptions=True)
    elapsed = time.perf_counter() - start
    duration = plan["duration_s"]
    valid = bool(records) and not aborted.is_set() and all(r["success"] for r in records)
    valid &= len(tasks) == len(plan["sessions"]) and missed_due == 0
    completed = [r for r in records if r["success"] and r["end"] <= duration]
    total_tokens = sum(r["window_output_tokens"] for r in records)
    summary = {
        "valid": valid,
        "missed_due_requests": missed_due,
        "record_sink_seconds": sink_seconds,
        "measurement_seconds": duration,
        "wall_seconds": elapsed,
        "cpu_seconds": time.process_time() - cpu_start,
        "max_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_units": "bytes on macOS, KiB on Linux; process lifetime high-water",
        "scheduled_sessions": len(plan["sessions"]),
        "launched_sessions": len(tasks),
        "completed_sessions": states["complete"],
        "incomplete_sessions_after_drain": len(tasks) - states["complete"],
        "requests_started": len(records),
        "requests_completed_in_window": len(completed),
        "requests_drained": sum(r["end"] > duration for r in records),
        "failed_requests": sum(not r["success"] for r in records),
        "output_tokens_per_second_per_chip": (total_tokens / duration / chips if valid else None),
        "ttft_seconds_p95": percentile([r["ttft_seconds"] for r in completed], 0.95)
        if valid
        else None,
        "offered_ttft_seconds_p95": percentile([r["offered_ttft_s"] for r in completed], 0.95)
        if valid
        else None,
        "event_loop_lag_s": distribution([s["loop_lag_s"] for s in samples]),
    }
    for name in (
        "dispatch_lateness_s",
        "headers_lateness_s",
        "connection_queue_s",
        "serialize_s",
        "prepare_s",
        "record_build_s",
    ):
        summary[name] = distribution([r[name] for r in records if r[name] is not None])
    summary["first_request_lateness_s"] = distribution(
        [r["dispatch_lateness_s"] for r in records if r["turn"] == 0]
    )
    summary["continuation_lateness_s"] = distribution(
        [r["dispatch_lateness_s"] for r in records if r["turn"] > 0]
    )
    return summary, records, samples

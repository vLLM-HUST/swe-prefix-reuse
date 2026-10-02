"""Reproducible open-loop session arrivals with observed continuation templates.

This is a timing/length scenario, not real prompt data or measured request arrivals.
First-output offsets become nominal release offsets by explicit modeling choice.
"""

import hashlib
import heapq
import json
import math
import random
from pathlib import Path

from .prepare import read_json
from .session_profile import CALL_FIELDS, describe
from .session_profile import SCHEMA as PROFILE_SCHEMA

SCHEMA = "session-arrival-plan/v1"


def positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def make_plan(
    profile,
    *,
    provider,
    rate,
    duration,
    seed=0,
    pattern="poisson",
    max_sessions=100000,
    max_calls=2000000,
):
    if profile.get("schema") != PROFILE_SCHEMA or profile.get("call_fields") != CALL_FIELDS:
        raise ValueError("unsupported session observation profile")
    positive(rate, "new sessions/s")
    positive(duration, "duration")
    if pattern not in {"poisson", "constant"}:
        raise ValueError("unknown arrival pattern")
    if type(seed) is not int or type(max_sessions) is not int or type(max_calls) is not int:
        raise ValueError("seed and safety limits must be integers")
    positive(max_sessions, "max_sessions")
    positive(max_calls, "max_calls")
    source = profile["providers"][provider]["sessions"]
    if not source:
        raise ValueError("empty provider")
    # Independent streams keep template draws stable when rate changes.
    arrivals, templates = random.Random(f"arrivals:{seed}"), random.Random(f"templates:{seed}")
    sessions, t, count = [], 0.0, 0
    while True:
        t += (arrivals.expovariate(1) if pattern == "poisson" else 1) / rate
        if t >= duration:
            break
        if len(sessions) >= max_sessions:
            raise ValueError(
                "max_sessions safety limit; shorten scenario or explicitly raise limit"
            )
        index = templates.randrange(len(source))
        sample = source[index]
        calls = sample["calls"]
        previous = -1
        if not calls or calls[0][0] != 0:
            raise ValueError("template must start at offset zero")
        for call in calls:
            if len(call) != len(CALL_FIELDS):
                raise ValueError("malformed call")
            offset, total, prefix, new, _output = call
            if not math.isfinite(offset) or offset < previous or offset < 0:
                raise ValueError("invalid template time order")
            if any(type(x) is not int or x < 0 for x in call[1:]) or total != prefix + new:
                raise ValueError("invalid template token accounting")
            previous = offset
        count += len(calls)
        if count > max_calls:
            raise ValueError("max_calls safety limit; no silent truncation")
        sessions.append(
            {
                "session_id": f"s{len(sessions):06d}",
                "arrival_s": t,
                "observation_index": index,
                "calls": calls,
            }
        )
    window = [c for s in sessions for c in s["calls"] if s["arrival_s"] + c[0] < duration]
    return {
        "schema": SCHEMA,
        "provider": provider,
        "new_sessions_per_second": rate,
        "duration_s": duration,
        "seed": seed,
        "arrival_pattern": pattern,
        "call_fields": CALL_FIELDS,
        "sessions": sessions,
        "provenance": profile["provenance"],
        "semantics": {
            "arrival": "independent wall-clock new-session releases; no concurrency feedback",
            "continuation": "eligible=max(session arrival+output-anchor offset, predecessor end)",
            "timing_proxy": "synthetic release offsets, NOT measured API arrivals or think time",
            "sampling": "whole observed sessions uniformly with replacement, one provider",
            "content": "length metadata only; no text/token IDs/prefix lineage inferred",
            "window": "cold-start, empty pool; retain all tails, do not claim stationary load",
            "identity": "fresh synthetic session IDs; no inter-session prefix reuse",
            "safety_limits": "fail rather than thin, truncate, or pause arrivals",
        },
        "offered_summary": {
            "new_sessions": len(sessions),
            "observed_calls_all_tails": count,
            "nominal_calls_in_window": len(window),
            "nominal_continuations_in_window": len(window) - len(sessions),
            "nominal_calls_after_window": count - len(window),
            "session_call_counts": describe([len(s["calls"]) for s in sessions]),
            "input_tokens_total_in_window": sum(c[1] for c in window),
            "api_cached_tokens_in_window": sum(c[2] for c in window),
            "api_uncached_tokens_in_window": sum(c[3] for c in window),
            "output_token_budget_in_window": sum(c[4] for c in window),
        },
    }


def build_plan(profile_path, output, **kwargs):
    plan = make_plan(read_json(profile_path), **kwargs)
    plan["provenance"] = dict(
        plan["provenance"],
        profile_sha256=hashlib.sha256(Path(profile_path).read_bytes()).hexdigest(),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(plan, separators=(",", ":")) + "\n")
    return plan["offered_summary"]


def simulate(plan, *, service_seconds, workers):
    """CPU-only protocol preview: fixed synthetic service; never an inference score.

    Independent arrival events are recorded even when all workers are occupied.
    Future continuations are released only when their predecessor is complete.
    At window end stop dispatching; only already-in-flight requests drain.
    """
    positive(service_seconds, "synthetic service seconds")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported plan")
    duration = plan["duration_s"]
    sessions = plan["sessions"]
    events = [(s["arrival_s"], 1, i, 0) for i, s in enumerate(sessions)]
    heapq.heapify(events)
    ready, running, records = [], {}, []
    arrived = peak_queue = completed_sessions = 0
    while events:
        now = events[0][0]
        # Process all same-time completions/releases before dispatching ready work.
        while events and events[0][0] == now:
            _, kind, i, turn = heapq.heappop(events)
            s = sessions[i]
            if kind == 0:
                record = running.pop(i)
                records.append(record)
                if turn + 1 == len(s["calls"]):
                    completed_sessions += 1
                else:
                    nominal = s["arrival_s"] + s["calls"][turn + 1][0]
                    release = max(nominal, now)
                    if release < duration:
                        heapq.heappush(events, (release, 1, i, turn + 1))
            else:
                if turn == 0:
                    arrived += 1
                nominal = s["arrival_s"] + s["calls"][turn][0]
                heapq.heappush(ready, (now, i, turn, nominal))
        if now >= duration:
            continue
        while ready and len(running) < workers:
            eligible, i, turn, nominal = heapq.heappop(ready)
            assert i not in running
            running[i] = {
                "session_id": sessions[i]["session_id"],
                "turn": turn,
                "nominal_s": nominal,
                "eligible_s": eligible,
                "start_s": now,
                "end_s": now + service_seconds,
                "predecessor_delay_s": eligible - nominal,
                "client_queue_s": now - eligible,
            }
            heapq.heappush(events, (now + service_seconds, 0, i, turn))
        peak_queue = max(peak_queue, len(ready))
    return {
        "scope": "CPU protocol preview with synthetic fixed service, NOT model performance",
        "service_seconds": service_seconds,
        "workers": workers,
        "offered_new_sessions": arrived,
        "pending_ready_at_stop": len(ready),
        "incomplete_sessions_after_drain": arrived - completed_sessions,
        "peak_ready_queue": peak_queue,
        "started_requests": len(records),
        "completed_in_window": sum(r["end_s"] <= duration for r in records),
        "drained_requests": sum(r["end_s"] > duration for r in records),
        "client_queue_s": describe([r["client_queue_s"] for r in records]),
        "records": records,
    }

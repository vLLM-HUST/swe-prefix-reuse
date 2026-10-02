"""Fixed-order SWE arrivals and frozen empirical post-completion delays."""

import hashlib
import json
import math
import random
from pathlib import Path

from .prepare import read_json, validate_workload
from .session_profile import SCHEMA as PROFILE_SCHEMA

SCHEMA = "swe-session-arrivals/v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrival_count(rate, duration):
    count = math.ceil(rate * duration)
    while count and (count - 1) / rate >= duration:
        count -= 1
    return count


def validate_plan(workload, plan):
    validate_workload(workload)
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported arrival plan")
    for key in ("new_sessions_per_second", "duration_s"):
        value = plan[key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if type(plan["seed"]) is not int:
        raise ValueError("seed must be an integer")
    expected = arrival_count(plan["new_sessions_per_second"], plan["duration_s"])
    if len(plan["sessions"]) != expected or expected > len(workload["sessions"]):
        raise ValueError("insufficient distinct trajectories or wrong arrival count; no recycling")
    for i, session in enumerate(plan["sessions"]):
        trace = workload["sessions"][i]
        if session["trajectory_id"] != trace["trajectory_id"]:
            raise ValueError("trajectory order differs from workload")
        if session["arrival_s"] != i / plan["new_sessions_per_second"]:
            raise ValueError("arrival is not on the fixed-rate clock")
        gaps = session["delay_after_turn_s"]
        if len(gaps) != len(trace["turns"]) - 1:
            raise ValueError("one frozen gap per continuation required")
        if any(isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in gaps):
            raise ValueError("delay must be nonnegative and finite")


def build(workload_path, profile_path, output, *, provider, rate, duration, seed=0):
    workload, profile = read_json(workload_path), read_json(profile_path)
    validate_workload(workload)
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError("unsupported observation profile")
    if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in (rate, duration)):
        raise ValueError("rate and duration must be positive and finite")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    count = arrival_count(rate, duration)
    if count > len(workload["sessions"]):
        raise ValueError(f"need {count} distinct trajectories; have {len(workload['sessions'])}")
    gaps = [
        b[0] - a[0]
        for s in profile["providers"][provider]["sessions"]
        for a, b in zip(s["calls"], s["calls"][1:])
    ]
    if not gaps or any(not math.isfinite(x) or x < 0 for x in gaps):
        raise ValueError("invalid or empty empirical gap distribution")
    sessions = []
    for i, trace in enumerate(workload["sessions"][:count]):
        rng = random.Random(f"{seed}:{trace['trajectory_id']}")
        sessions.append(
            {
                "trajectory_id": trace["trajectory_id"],
                "arrival_s": i / rate,
                "delay_after_turn_s": [rng.choice(gaps) for _ in trace["turns"][1:]],
            }
        )
    plan = {
        "schema": SCHEMA,
        "workload_sha256": digest(workload_path),
        "profile_sha256": digest(profile_path),
        "provider": provider,
        "new_sessions_per_second": rate,
        "duration_s": duration,
        "seed": seed,
        "sessions": sessions,
        "semantics": "Sequential SWE trajectories; fixed-rate arrivals starting at zero; "
        "empirical first-output gaps repurposed as synthetic POST-COMPLETION waits; "
        "not measured tool/user idle times; no recycling or gap clipping",
    }
    validate_plan(workload, plan)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(plan, stream, indent=2)
    return {"sessions": count, "continuations": sum(len(s["delay_after_turn_s"]) for s in sessions)}

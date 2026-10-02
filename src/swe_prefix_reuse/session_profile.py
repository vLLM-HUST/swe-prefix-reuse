"""TraceLab session observations adapted from ruijie-workshop, not idle times.

Namespaced identities are used only in memory. Export anonymous session ordinals,
relative first-output anchors, and call length accounting; no content or raw IDs.
"""

import collections
import gzip
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

SCHEMA = "session-observation-profile/v1"
SOURCE = "https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.2"
METHOD = "CubeLander/ruijie-workshop@ddcec06550b531b2cde1027c187d317d21b8da94"
OUTPUT_EVENTS = {"reasoning", "text", "tool_call", "usage_report"}
CALL_FIELDS = [
    "output_anchor_offset_s",
    "input_tokens_total",
    "prefix_tokens",
    "newly_append_tokens",
    "output_tokens",
]


def describe(values):
    values = sorted(values)
    if not values:
        return {"n": 0}

    def quantile(p):
        x = (len(values) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)

    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "maximum": values[-1],
        "zeros": values.count(0),
    }


def observe(rows):
    sessions = collections.defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["user"], row["provider"], row["session_id"])
        identity = (*key, row["round_id"])
        if identity in seen:
            raise ValueError("duplicate namespaced round ID")
        seen.add(identity)
        lengths = [row[k] for k in CALL_FIELDS[1:]]
        if any(type(x) is not int or x < 0 for x in lengths):
            raise ValueError("token counts must be nonnegative integers")
        total, prefix, new, _ = lengths
        if total != prefix + new:
            raise ValueError("input accounting identity violated")
        if key[1] == "claude" and prefix != row["claude_cache_read_input_tokens"]:
            raise ValueError("Claude cache-read identity violated")
        times, outputs = [], []
        for event in row["timing_events"]:
            if not event.get("timestamp"):
                continue
            dt = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            if dt.tzinfo is None:
                raise ValueError("timezone required")
            t = dt.timestamp()
            times.append(t)
            if event["event_type"] in OUTPUT_EVENTS:
                outputs.append(t)
        if not outputs:
            raise ValueError("missing first-output anchor")
        sessions[key].append((min(outputs), min(times), max(times), lengths))
    providers = {}
    for (_, provider, _), calls in sessions.items():
        target = providers.setdefault(provider, {"sessions": []})
        calls.sort(key=lambda c: c[0])  # Stable input order for equal-time observations.
        first = calls[0][0]
        target["sessions"].append(
            {
                "span_s": max(c[2] for c in calls) - min(c[1] for c in calls),
                "calls": [[c[0] - first, *c[3]] for c in calls],
            }
        )
    if not providers:
        raise ValueError("empty trace")
    for group in providers.values():
        all_calls = [c for s in group["sessions"] for c in s["calls"]]
        group["statistics"] = {
            "session_calls": describe([len(s["calls"]) for s in group["sessions"]]),
            "session_span_s": describe([s["span_s"] for s in group["sessions"]]),
            "output_anchor_gap_s": describe(
                [b[0] - a[0] for s in group["sessions"] for a, b in zip(s["calls"], s["calls"][1:])]
            ),
            **{
                field: describe([c[i] for c in all_calls])
                for i, field in enumerate(CALL_FIELDS)
                if i
            },
        }
    return {
        "schema": SCHEMA,
        "call_fields": CALL_FIELDS,
        "providers": providers,
        "provenance": {
            "release": SOURCE,
            "license": "CC-BY-4.0",
            "authors": "Kan Zhu et al. / UW TraceLab",
            "method": METHOD,
        },
        "semantics": {
            "timing": "first-model-output to first-model-output; NOT arrival or idle",
            "prefix_tokens": "API-accounted cache-read; NOT prefix identity or storage IO",
            "weighting": "uniform sessions within provider; includes singletons",
            "censoring": "observed histories; no complete-lifetime claim",
            "privacy": "no raw session/user IDs, content, or absolute timestamps exported",
        },
    }


def build_profile(source, output):
    source, output = Path(source), Path(output)
    with gzip.open(source, "rt") as stream:
        result = observe(json.loads(line) for line in stream)
    with source.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result["provenance"].update(raw_sha256=digest.hexdigest(), raw_bytes=source.stat().st_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Compressed output is deterministic (no wallclock header), never overwrites.
    with (
        output.open("xb") as stream,
        gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped,
    ):
        zipped.write((json.dumps(result, separators=(",", ":")) + "\n").encode())
    return {p: x["statistics"] for p, x in result["providers"].items()}

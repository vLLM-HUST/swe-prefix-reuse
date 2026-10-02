import asyncio
import copy

import pytest
from test_client import endpoint, reply
from test_runner import workload

from swe_prefix_reuse.arrival_plan import SCHEMA, validate_plan
from swe_prefix_reuse.arrival_runner import replay


def inputs(n=3, rate=30, duration=0.3, gap=0.01):
    w = workload()
    w["sessions"] = [dict(copy.deepcopy(w["sessions"][0]), trajectory_id=f"t{i}") for i in range(n)]
    p = {
        "schema": SCHEMA,
        "duration_s": duration,
        "new_sessions_per_second": rate,
        "seed": 17,
        "sessions": [
            {"trajectory_id": f"t{i}", "arrival_s": i / rate, "delay_after_turn_s": [gap]}
            for i in range(n)
        ],
    }
    return w, p


def test_fixed_arrivals_gap_after_completion_and_exact_prefix():
    async def check():
        seen = {}

        async def handler(req):
            b = await req.json()
            seen.setdefault(b["cache_salt"], []).append(b["prompt"])
            assert req.headers["X-data-parallel-rank"] == str(
                int(b["cache_salt"].split(":")[-1]) % 2
            )
            return await reply(req, delay=0.002)

        w, p = inputs(n=3, rate=10, duration=0.3, gap=0.015)
        async with endpoint(handler) as url:
            s, rows, samples = await replay(
                w,
                p,
                url=url,
                model="fixture",
                chips=2,
                data_parallel_size=2,
                monitor_interval=0.005,
            )
        assert s["valid"] and s["launched_sessions"] == 3
        assert s["completed_sessions"] == 3
        assert len(seen) == 3 and samples
        for prompts in seen.values():
            assert prompts[1] == prompts[0] + [700, 701, 702, 12]
        for i in range(3):
            a, b = [r for r in rows if r["session"] == i]
            assert b["planned_send_s"] == pytest.approx(a["end"] + 0.015)
            assert b["start"] >= b["planned_send_s"]
            assert "token_ids" not in b

    asyncio.run(check())


def test_connection_queue_does_not_throttle_new_session_launches():
    async def check():
        async def handler(req):
            return await reply(req, delay=0.02)

        w, p = inputs(n=10, rate=100, duration=0.1, gap=999)
        async with endpoint(handler) as url:
            s, rows, _ = await replay(w, p, url=url, model="fixture", chips=1, connections=1)
        assert s["valid"] and s["launched_sessions"] == 10
        assert s["connection_queue_s"]["max"] > 0.1
        assert s["incomplete_sessions_after_drain"] == 10
        assert all(r["turn"] == 0 for r in rows)
        assert s["requests_drained"] > 0
        assert s["output_tokens_per_second_per_chip"] < 300

    asyncio.run(check())


def test_long_sleep_does_not_extend_run_or_count_as_complete():
    async def check():
        async def handler(req):
            return await reply(req)

        w, p = inputs(n=1, rate=1, duration=0.04, gap=100000)
        async with endpoint(handler) as url:
            s, rows, _ = await replay(w, p, url=url, model="fixture", chips=1)
        assert s["valid"] and s["wall_seconds"] < 1
        assert s["incomplete_sessions_after_drain"] == 1 and len(rows) == 1

    asyncio.run(check())


def test_protocol_failure_suppresses_score_and_wakes_sleepers():
    async def check():
        async def handler(req):
            return await reply(req, fault="echo")

        w, p = inputs(n=1, rate=0.1, duration=5, gap=1000)
        async with endpoint(handler) as url:
            s, _, _ = await replay(w, p, url=url, model="fixture", chips=1)
        assert not s["valid"] and s["output_tokens_per_second_per_chip"] is None
        assert s["wall_seconds"] < 1

    asyncio.run(check())


def test_duplicate_recycling_and_bad_gaps_rejected():
    w, p = inputs(n=1, rate=10, duration=0.2)
    with pytest.raises(ValueError, match="insufficient"):
        validate_plan(w, p)
    w, p = inputs(n=1, rate=1, duration=0.2, gap=-1)
    with pytest.raises(ValueError, match="delay"):
        validate_plan(w, p)

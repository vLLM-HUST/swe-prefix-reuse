import pytest

from swe_prefix_reuse.session_arrivals import make_plan, simulate
from swe_prefix_reuse.session_profile import CALL_FIELDS, SCHEMA


def profile():
    return {
        "schema": SCHEMA,
        "call_fields": CALL_FIELDS,
        "provenance": {},
        "providers": {
            "test": {
                "sessions": [
                    {
                        "span_s": 3,
                        "calls": [[0, 10, 0, 10, 2], [0.1, 14, 12, 2, 2], [3, 17, 16, 1, 2]],
                    }
                ]
            }
        },
    }


def test_deterministic_and_no_tail_truncation():
    p = profile()
    a = make_plan(p, provider="test", rate=2, duration=2, seed=4)
    assert a == make_plan(p, provider="test", rate=2, duration=2, seed=4)
    assert all(len(s["calls"]) == 3 for s in a["sessions"])
    assert a["offered_summary"]["nominal_calls_after_window"] > 0
    b = make_plan(p, provider="test", rate=4, duration=2, seed=4)
    for x, y in zip(a["sessions"], b["sessions"]):
        assert x["arrival_s"] == y["arrival_s"] * 2
        assert x["observation_index"] == y["observation_index"]


def test_overload_does_not_throttle_new_arrivals_and_predecessor_gates():
    plan = make_plan(profile(), provider="test", rate=10, duration=2, pattern="constant")
    slow = simulate(plan, service_seconds=0.7, workers=1)
    fast = simulate(plan, service_seconds=0.01, workers=2)
    assert slow["offered_new_sessions"] == fast["offered_new_sessions"] == len(plan["sessions"])
    assert slow["pending_ready_at_stop"] > fast["pending_ready_at_stop"]
    assert slow["client_queue_s"]["maximum"] > 0
    records = slow["records"]
    for r in records:
        prev = [
            x for x in records if x["session_id"] == r["session_id"] and x["turn"] == r["turn"] - 1
        ]
        if prev:
            assert r["start_s"] >= prev[0]["end_s"]
        assert r["start_s"] >= r["eligible_s"] >= r["nominal_s"]
        assert r["start_s"] < plan["duration_s"]


@pytest.mark.parametrize("rate", [0, -1, float("inf"), float("nan"), True])
def test_invalid_rates(rate):
    with pytest.raises(ValueError):
        make_plan(profile(), provider="test", rate=rate, duration=1)


def test_safety_cap_fails_instead_of_resampling_or_clipping():
    with pytest.raises(ValueError, match="max_calls"):
        make_plan(profile(), provider="test", rate=2, duration=2, pattern="constant", max_calls=1)

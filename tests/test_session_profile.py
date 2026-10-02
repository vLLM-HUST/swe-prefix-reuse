import copy

import pytest

from swe_prefix_reuse.session_profile import observe


def row(user="u", provider="codex", session="s", round_id=0, second=0, output=True):
    return {
        "user": user,
        "provider": provider,
        "session_id": session,
        "round_id": round_id,
        "input_tokens_total": 10,
        "prefix_tokens": 8,
        "newly_append_tokens": 2,
        "output_tokens": 3,
        "claude_cache_read_input_tokens": 8,
        "timing_events": [
            {
                "event_type": "text" if output else "user_message",
                "timestamp": f"2026-01-01T00:00:{second:02d}Z",
            }
        ],
    }


def test_namespaces_singletons_equal_times_and_correlations():
    rows = [
        row(second=4),
        row(round_id=1, second=4),
        row(round_id=2, second=9),
        row(user="other"),
        row(provider="claude"),
    ]
    result = observe(rows)
    c = result["providers"]["codex"]
    assert len(c["sessions"]) == 2
    assert c["sessions"][0]["calls"] == [[0, 10, 8, 2, 3], [0, 10, 8, 2, 3], [5, 10, 8, 2, 3]]
    assert c["statistics"]["output_anchor_gap_s"]["zeros"] == 1
    assert c["statistics"]["session_calls"]["mean"] == 2
    assert set(c["sessions"][0]) == {"span_s", "calls"}
    assert c["statistics"]["session_span_s"]["maximum"] == 5


@pytest.mark.parametrize("change", ["duplicate", "missing", "accounting", "negative", "claude"])
def test_bad_rows_fail(change):
    r = row()
    rows = [r]
    if change == "duplicate":
        rows.append(copy.deepcopy(r))
    elif change == "missing":
        r["timing_events"] = []
    elif change == "accounting":
        r["input_tokens_total"] = 11
    elif change == "negative":
        r["output_tokens"] = -1
    else:
        r["provider"] = "claude"
        r["claude_cache_read_input_tokens"] = 0
    with pytest.raises(ValueError):
        observe(rows)

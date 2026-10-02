from swe_prefix_reuse.session_audit import audit


def test_length_checks_do_not_conflate_api_cache_with_prefix_identity():
    profile = {
        "providers": {
            "test": {
                "sessions": [{"calls": [[0, 10, 8, 2, 2], [1, 13, 13, 0, 4], [2, 10, 0, 10, 1]]}]
            }
        }
    }
    counts = audit(profile, 15)["providers"]["test"]["counts"]
    assert counts["append_length_feasible"] == 1
    assert counts["cached_gt_prev_input_output"] == 1
    assert counts["calls_over_context"] == 1
    assert counts["whole_session_append_feasible"] == 0
    assert counts["initial_cached_nonzero"] == 1

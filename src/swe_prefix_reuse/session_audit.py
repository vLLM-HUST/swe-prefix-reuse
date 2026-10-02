"""Length-only feasibility evidence; no claim of recovered prefix content identity."""

import itertools
from collections import Counter

from .session_profile import describe


def audit(profile, max_context=262144):
    if type(max_context) is not int or max_context < 1:
        raise ValueError("max_context must be a positive integer")
    result = {}
    for provider, group in profile["providers"].items():
        counts = Counter()
        deltas, negative, maxima, accepted_depth, first_inputs = [], [], [], [], []
        for session in group["sessions"]:
            calls = session["calls"]
            counts["sessions"] += 1
            counts["calls"] += len(calls)
            counts["singleton"] += len(calls) == 1
            counts["initial_cached_nonzero"] += calls[0][2] > 0
            first_inputs.append(calls[0][1])
            maxima.append(max(c[1] + c[4] for c in calls))
            appendable = fits = True
            for call in calls:
                counts["zero_input"] += call[1] == 0
                counts["zero_output"] += call[4] == 0
                over = call[1] + call[4] > max_context
                counts["calls_over_context"] += over
                fits &= not over
            for a, b in itertools.pairwise(calls):
                counts["pairs"] += 1
                delta = b[1] - a[1] - a[4]
                deltas.append(delta)
                counts["append_length_feasible"] += delta >= 0
                counts["input_decreases"] += b[1] < a[1]
                counts["cached_gt_prev_input_output"] += b[2] > a[1] + a[4]
                counts["next_cache_zero"] += b[2] == 0
                counts["zero_gap"] += b[0] == a[0]
                if delta < 0:
                    negative.append(-delta)
                    appendable = False
            counts["whole_session_append_feasible"] += appendable
            counts["whole_session_fits_context"] += fits
            counts["whole_session_both"] += appendable and fits
            counts["calls_in_whole_append_sessions"] += len(calls) if appendable else 0
            counts["calls_in_whole_fit_sessions"] += len(calls) if fits else 0
            if appendable and fits:
                accepted_depth.append(len(calls))
        result[provider] = {
            "counts": dict(counts),
            "signed_append_delta": describe(deltas),
            "negative_delta_magnitude": describe(negative),
            "max_context_by_session": describe(maxima),
            "initial_input": describe(first_inputs),
            "fully_appendable_in_context_depth": describe(accepted_depth),
        }
    return {
        "max_context": max_context,
        "providers": result,
        "scope": "length compatibility only; a negative delta does not prove compaction, "
        "and a nonnegative delta does not prove content-prefix identity",
    }

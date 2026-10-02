import copy
import gzip
import json

import pytest
from test_runner import workload

from swe_prefix_reuse.arrival_plan import build
from swe_prefix_reuse.session_profile import SCHEMA


def test_frozen_delays_order_and_independent_rate(tmp_path):
    w = workload()
    w["sessions"].append(dict(copy.deepcopy(w["sessions"][0]), trajectory_id="second"))
    wp, pp = tmp_path / "w.json", tmp_path / "p.json.gz"
    wp.write_text(json.dumps(w))
    with gzip.open(pp, "wt") as f:
        json.dump(
            {
                "schema": SCHEMA,
                "providers": {"codex": {"sessions": [{"calls": [[0], [0], [7], [5000]]}]}},
            },
            f,
        )
    paths = [tmp_path / f"{i}.json" for i in range(3)]
    for out, rate in zip(paths, [1, 1, 2]):
        build(wp, pp, out, provider="codex", rate=rate, duration=2 / rate, seed=7)
    a, b, c = [json.loads(x.read_text()) for x in paths]
    assert a == b
    assert [s["trajectory_id"] for s in a["sessions"]] == ["trace", "second"]
    assert [s["delay_after_turn_s"] for s in a["sessions"]] == [
        s["delay_after_turn_s"] for s in c["sessions"]
    ]
    assert c["sessions"][1]["arrival_s"] == 0.5
    with pytest.raises(ValueError, match="distinct"):
        build(wp, pp, tmp_path / "bad.json", provider="codex", rate=10, duration=2)
    with pytest.raises(FileExistsError):
        build(wp, pp, paths[0], provider="codex", rate=1, duration=2)

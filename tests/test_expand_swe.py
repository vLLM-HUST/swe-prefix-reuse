import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("expand_swe", scripts / "expand_swe.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_order_duplicates_metadata_and_no_partial_output(tmp_path):
    source = tmp_path / "source"
    shard = source / module.SHARD
    shard.parent.mkdir(parents=True)
    rows = [
        {
            "trajectory_id": i,
            "instance_id": i,
            "tools": [],
            "messages": [
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
            ],
        }
        for i in ["first", "first", "second"]
    ]
    pq.write_table(pa.Table.from_pylist(rows), shard)
    meta = source / ".cache/huggingface/download" / (module.SHARD + ".metadata")
    meta.parent.mkdir(parents=True)
    meta.write_text(
        module.REVISION + "\n" + hashlib.sha256(shard.read_bytes()).hexdigest() + "\n0\n"
    )
    out = tmp_path / "out.json.gz"
    report = module.expand(source, out, 2)
    assert report["sessions"] == 2 and report["excluded"] == 1
    with gzip.open(out, "rt") as stream:
        data = json.load(stream)
    assert [s["trajectory_id"] for s in data["sessions"]] == ["first", "second"]
    with pytest.raises(ValueError, match="next pinned shard"):
        module.expand(source, tmp_path / "missing.json.gz", 3)
    assert not (tmp_path / "missing.json.gz").exists()
    meta.write_text("wrong\nidentity\n")
    with pytest.raises(ValueError, match="revision"):
        module.expand(source, tmp_path / "bad.json.gz", 1)

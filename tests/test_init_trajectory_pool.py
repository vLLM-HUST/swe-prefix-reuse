import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_prepare import Tokenizer, source

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("jinja2")
scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "init_trajectory_pool", scripts / "init_trajectory_pool.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_shard(root, name, rows):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    meta = root / ".cache/huggingface/download" / (name + ".metadata")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(
        module.REVISION + "\n" + hashlib.sha256(path.read_bytes()).hexdigest() + "\n0\n"
    )


def rows():
    a = source()
    a["tools"] = []
    big = copy.deepcopy(a)
    big["trajectory_id"] = "large"
    big["messages"][0]["content"] = "x" * 5000
    b = copy.deepcopy(a)
    b["trajectory_id"] = "second"
    return [big, a, b]


def test_exact_qualified_count_after_rejection_and_download_only_needed_shard(tmp_path):
    fetched = []

    def fetch(name):
        fetched.append(name)
        write_shard(tmp_path / "cache", name, rows())

    out = tmp_path / "pool"
    receipt = module.initialize(
        tmp_path / "cache", out, 2, 1000, Tokenizer(), {"fixture": True}, fetch=fetch
    )
    assert receipt["status"] == "ready" and len(fetched) == 1
    workload = json.loads((out / "workload.json").read_text())
    assert [s["trajectory_id"] for s in workload["sessions"]] == ["t", "second"]
    assert len(workload["rejected"]) == 1
    assert json.loads((out / "receipt.json").read_text())["accepted_count"] == 2
    with pytest.raises(FileExistsError):
        module.initialize(tmp_path / "cache", out, 2, 1000, Tokenizer(), {})


def test_short_pool_is_failed_not_ready_or_recycled(tmp_path):
    from expand_swe import SHARD

    write_shard(tmp_path / "cache", SHARD, rows())
    out = tmp_path / "pool"
    with pytest.raises(ValueError, match="exhausted"):
        module.initialize(tmp_path / "cache", out, 3, 1000, Tokenizer(), {}, max_shards=1)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["accepted_count"] == 2
    assert not (out / "workload.json").exists()

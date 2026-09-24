"""Rebuild the small attributed sample from a pinned local Hugging Face download.

Selection uses serialized source size, never measured model performance. Source
text is inert. This deliberately bounded sample is not a population benchmark.
"""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq

REVISION = "fb0c0dccc7a5cce79b3f6de891848acdede36685"
SHARD = "data/minisweagent/qwen38_27b/scale-swe/train-00000-of-00022.parquet"


def credential_like_content(row):
    text = json.dumps(row, ensure_ascii=False)
    if re.search(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,}|AKIA[A-Z0-9]{16}|hf_[A-Za-z0-9]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{40,})\b",
        text,
    ):
        return True
    for match in re.finditer(r'https?://[^\s"<>\\`]+', text):
        try:
            url = urlsplit(match.group())
            if url.username and url.password:
                return True
        except ValueError:
            continue
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/open-swe-sample.json.gz"))
    args = parser.parse_args()
    source = args.dataset / SHARD
    metadata = args.dataset / ".cache/huggingface/download" / (SHARD + ".metadata")
    revision, etag, *_ = metadata.read_text().splitlines()
    if revision != REVISION:
        raise ValueError("local source is not the pinned dataset revision")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != etag:
        raise ValueError("source Parquet differs from pinned download content identity")
    rows = next(
        pq.ParquetFile(source).iter_batches(
            batch_size=32, columns=["messages", "tools", "trajectory_id", "instance_id"]
        )
    ).to_pylist()
    excluded = [i for i, row in enumerate(rows) if credential_like_content(row)]
    ranked = sorted(
        (i for i in range(len(rows)) if i not in excluded),
        key=lambda i: (len(json.dumps(rows[i], ensure_ascii=False)), i),
    )
    if len(ranked) < 8:
        raise ValueError("fewer than eight eligible whole trajectories")
    selected = [ranked[round(i * (len(ranked) - 1) / 7)] for i in range(8)]
    provenance = {
        "dataset": "nvidia/Open-SWE-Traces",
        "revision": REVISION,
        "url": f"https://huggingface.co/datasets/nvidia/Open-SWE-Traces/tree/{REVISION}",
        "creator": "NVIDIA",
        "license": "CC-BY-4.0",
        "source_shard": SHARD,
        "source_shard_sha256": etag,
        "source_rows_zero_based": selected,
        "selection": "8 evenly spaced size ranks among first 32 rows, after conservative credential-like-content exclusion",
        "excluded_rows_zero_based": excluded,
        "exclusion": "whole rows with credential-bearing URL examples or recognizable key patterns; no secret content logged",
        "changes": "subset; only messages, tools, trajectory_id and instance_id retained; no message text edits",
        "scope": "small size-spread example, not a representative population sample",
    }
    document = {"provenance": provenance, "sessions": [rows[i] for i in selected]}
    raw = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
    compressed = gzip.compress(raw, mtime=0)
    with args.output.open("xb") as stream:
        stream.write(compressed)
    print(json.dumps({"rows": selected, "raw_bytes": len(raw), "gzip_bytes": len(compressed)}))


if __name__ == "__main__":
    main()

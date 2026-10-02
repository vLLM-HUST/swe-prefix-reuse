"""Select complete pinned SWE trajectories in shard/row order, never randomly.

Source download uses Hugging Face local_dir metadata. No source content is executed.
This extends the existing bundle's screening, not its eight-row size-spread selection.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from bundle_traces import REVISION, SHARD, credential_like_content


def verified_rows(dataset, sources, max_shards=22, fetch=None):
    """Yield pinned rows in order; optional fetch downloads only the needed shard."""
    dataset = Path(dataset)
    for shard_index in range(max_shards):
        name = SHARD.replace("00000-of-", f"{shard_index:05d}-of-")
        if fetch:
            fetch(name)
        path = dataset / name
        metadata = dataset / ".cache/huggingface/download" / (name + ".metadata")
        if not path.exists() or not metadata.exists():
            raise ValueError(
                f"need next pinned shard and download metadata: {name}; "
                "download the pinned shard or enable downloading"
            )
        revision, etag, *_ = metadata.read_text().splitlines()
        if revision != REVISION:
            raise ValueError("dataset revision mismatch")
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != etag:
            raise ValueError("Parquet identity mismatch")
        source = {"shard": name, "sha256": etag, "rows_examined": 0}
        sources.append(source)
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=8, columns=["messages", "tools", "trajectory_id", "instance_id"]
        ):
            for row in batch.to_pylist():
                ordinal = source["rows_examined"]
                source["rows_examined"] += 1
                yield name, ordinal, row


def row_rejection(row, identities):
    if credential_like_content(row):
        return "credential-like content"
    if row["trajectory_id"] in identities:
        return "duplicate trajectory ID"
    if sum(m.get("role") == "assistant" for m in row["messages"]) < 2:
        return "fewer than two assistant turns"
    return None


def expand(dataset, output, count):
    if count < 1:
        raise ValueError("count must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    selected, exclusions, sources, identities = [], [], [], set()
    for name, ordinal, row in verified_rows(dataset, sources):
        reason = row_rejection(row, identities)
        if reason:
            exclusions.append({"shard": name, "row": ordinal, "reason": reason})
        else:
            identities.add(row["trajectory_id"])
            selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("exhausted source; no partial or recycled bundle emitted")
    provenance = {
        "dataset": "nvidia/Open-SWE-Traces",
        "revision": REVISION,
        "url": "https://huggingface.co/datasets/nvidia/Open-SWE-Traces",
        "authors": "NVIDIA",
        "license": "CC-BY-4.0",
        "sources": sources,
        "selection": "first eligible whole rows, contiguous shard order; no padding/truncation",
        "exclusions": exclusions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as z:
        z.write(json.dumps({"provenance": provenance, "sessions": selected}).encode())
    return {"sessions": len(selected), "excluded": len(exclusions), "shards": len(sources)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(expand(args.dataset, args.output, args.count)))

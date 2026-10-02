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


def expand(dataset, output, count):
    if count < 1:
        raise ValueError("count must be positive")
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(output)
    selected, exclusions, sources, identities = [], [], [], set()
    for shard_index in range(22):
        name = SHARD.replace("00000-of-", f"{shard_index:05d}-of-")
        path = dataset / name
        metadata = dataset / ".cache/huggingface/download" / (name + ".metadata")
        if not path.exists() or not metadata.exists():
            raise ValueError(
                f"need next pinned shard and download metadata: {name}; "
                f"have {len(selected)} eligible trajectories, need {count}"
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
        sources.append({"shard": name, "sha256": etag})
        ordinal = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=8, columns=["messages", "tools", "trajectory_id", "instance_id"]
        ):
            for row in batch.to_pylist():
                reason = None
                if credential_like_content(row):
                    reason = "credential-like content"
                elif row["trajectory_id"] in identities:
                    reason = "duplicate trajectory ID"
                elif sum(m.get("role") == "assistant" for m in row["messages"]) < 2:
                    reason = "fewer than two assistant turns"
                if reason:
                    exclusions.append({"shard": name, "row": ordinal, "reason": reason})
                else:
                    identities.add(row["trajectory_id"])
                    selected.append(row)
                ordinal += 1
                if len(selected) == count:
                    break
            if len(selected) == count:
                break
        sources[-1]["rows_examined"] = ordinal
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

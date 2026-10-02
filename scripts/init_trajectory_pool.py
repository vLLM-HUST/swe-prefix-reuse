"""Initialize exactly N qualified SWE trajectories, sequentially, with provenance.

Only source shards are downloaded. Tokenizer must already be local; model weights
are never downloaded or loaded. Failed initialization retains a failure receipt,
not a usable partial pool. Raw downloads stay cached for a subsequent attempt.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from expand_swe import REVISION, row_rejection, verified_rows
from jinja2 import TemplateError

from swe_prefix_reuse.prepare import SCHEMA, compile_session, validate_workload

REPO = "nvidia/Open-SWE-Traces"


def initialize(
    dataset, output, count, max_context, tokenizer, tokenizer_metadata, max_shards=22, fetch=None
):
    if count < 1 or max_context < 1 or not 1 <= max_shards <= 22:
        raise ValueError("positive count/context and 1..22 max-shards required")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    sources, accepted, original, rejected, identities = [], [], [], [], set()
    receipt = {
        "status": "initializing",
        "requested_count": count,
        "max_context": max_context,
        "dataset": REPO,
        "revision": REVISION,
        "max_shards": max_shards,
    }

    def save():
        (root / "receipt.json").write_text(
            json.dumps(
                {
                    **receipt,
                    "accepted_count": len(accepted),
                    "sources": sources,
                    "rejected": rejected,
                },
                indent=2,
            )
            + "\n"
        )

    save()
    try:
        for name, ordinal, row in verified_rows(dataset, sources, max_shards, fetch):
            reason = row_rejection(row, identities)
            if reason is None:
                identities.add(row["trajectory_id"])
                try:
                    compiled = compile_session(row, tokenizer)
                    if compiled["max_context_tokens"] > max_context:
                        reason = "whole trajectory exceeds context"
                except (ValueError, TypeError, KeyError, TemplateError) as exc:
                    reason = "token compilation rejected: " + type(exc).__name__
            if reason:
                rejected.append({"shard": name, "row": ordinal, "reason": reason})
            else:
                original.append(row)
                accepted.append(compiled)
            if len(accepted) == count:
                break
        if len(accepted) != count:
            raise ValueError("source/shard limit exhausted before requested qualified count")
        provenance = {
            "dataset": REPO,
            "revision": REVISION,
            "sources": sources,
            "authors": "NVIDIA",
            "license": "CC-BY-4.0",
            "url": "https://huggingface.co/datasets/" + REPO,
            "selection": "first tokenizer-qualified whole trajectories in shard/row order",
            "exclusions": rejected,
        }
        source = root / "source.json.gz"
        with (
            source.open("xb") as stream,
            gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped,
        ):
            zipped.write(json.dumps({"provenance": provenance, "sessions": original}).encode())
        workload = {
            "schema": SCHEMA,
            "source": provenance,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "tokenizer": tokenizer_metadata,
            "max_context": max_context,
            "policy": "independent input deltas, actual generated history; no truncation/padding",
            "sessions": accepted,
            "rejected": rejected,
        }
        validate_workload(workload)
        path = root / "workload.json"
        with path.open("x") as stream:
            json.dump(workload, stream, separators=(",", ":"))
        receipt.update(
            status="ready",
            workload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            turns=sum(len(s["turns"]) for s in accepted),
            max_context_observed=max(s["max_context_tokens"] for s in accepted),
        )
    except BaseException as exc:
        receipt.update(status="failed", error=type(exc).__name__)
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count", type=int, required=True, help="accepted pool size AFTER compilation"
    )
    parser.add_argument("--tokenizer", required=True, help="existing local tokenizer directory")
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument("--dataset-cache", default="prepared/open-swe-download")
    parser.add_argument("--max-shards", type=int, default=22, help="explicit download/scan bound")
    parser.add_argument(
        "--offline", action="store_true", help="only verified existing local shards"
    )
    parser.add_argument("--output", required=True, help="new pool directory; refuses overwrite")
    args = parser.parse_args()
    if args.count < 1 or args.max_context < 1 or not 1 <= args.max_shards <= 22:
        parser.error("invalid count, context or shard bound")
    if not Path(args.tokenizer).is_dir():
        parser.error("tokenizer must be an existing local directory")
    if Path(args.output).exists():
        parser.error("output already exists; preserve receipt and choose a new directory")
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    metadata = {
        "path": str(Path(args.tokenizer).resolve()),
        "enable_thinking": True,
        "transformers": transformers.__version__,
        "fingerprint": hashlib.sha256(
            (tokenizer.backend_tokenizer.to_str() + tokenizer.chat_template).encode()
        ).hexdigest(),
    }
    fetch = None
    if not args.offline:
        from huggingface_hub import hf_hub_download

        def fetch(name):
            hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                revision=REVISION,
                filename=name,
                local_dir=args.dataset_cache,
            )

    result = initialize(
        args.dataset_cache,
        args.output,
        args.count,
        args.max_context,
        tokenizer,
        metadata,
        args.max_shards,
        fetch,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

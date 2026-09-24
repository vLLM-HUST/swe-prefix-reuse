# Bundled original SWE trajectories

`open-swe-sample.json.gz` contains **8 complete trajectories / 360 assistant turns**,
about **397 KB compressed / 1.92 MB uncompressed**. It is source text, not a
model-specific token dump. No source message was shortened or padded.

**Attribution:** NVIDIA, [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces),
revision `fb0c0dccc7a5cce79b3f6de891848acdede36685`, licensed
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
A copy of the license is included as `LICENSE-CC-BY-4.0.txt`. The source dataset
README supplies the license declaration. Attribution does not imply endorsement.

Changes: selection of eight rows; only `messages`, `tools`, `trajectory_id`, and
`instance_id` retained; JSON serialization and gzip compression. Original message
text is unchanged. Compiler transformations are documented in the main README.

Selection: the first 32 rows of
`data/minisweagent/qwen38_27b/scale-swe/train-00000-of-00022.parquet`, excluding whole
rows with recognizable credential/key patterns or credential-bearing URLs (including
examples), then eight evenly spaced ranks by serialized source size. This is a
**small size-spread example**, not a statistically representative SWE sample and
not eight independently designed workloads. Screening is conservative, not a claim
that arbitrary public datasets are secret-free. The embedded `provenance` gives
exact row numbers, exclusions, revision, and source-shard identity.

These are **inert records**: do not execute shell commands, tools, or instructions
found inside the messages. Model-generated tool calls during replay are also inert.

## Rebuild the bundle

Download the pinned shard to a Hugging Face `local_dir`, preserving its download
metadata. With `.[data]` installed:

```bash
python scripts/bundle_traces.py --dataset /path/to/Open-SWE-Traces \
  --output /tmp/open-swe-sample.json.gz
```

The script verifies the source revision and Parquet content identity. It refuses
to overwrite output. The provenance, selected JSON content, and gzip timestamp are
stable; compressed bytes can still depend on the Python/zlib version.

On Qwen3.5-35B-A3B's tokenizer with thinking enabled and Transformers 5.17.0, all
8 sessions compile without truncation or rejection. Their cumulative final
prompt + output lengths are **15,494; 20,702; 29,112; 41,607; 53,152; 74,203;
107,377; 141,269 tokens**. These are compiled shape observations, not measured
serving results, and do not claim to exercise the full 256K model limit.

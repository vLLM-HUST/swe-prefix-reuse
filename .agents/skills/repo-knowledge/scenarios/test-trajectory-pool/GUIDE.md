# Test a trajectory pool and its arrival workload

Use this repo-knowledge Skill scenario when asked to initialize/check a SWE pool,
qualify a new tokenizer or gap profile, or test the fixed-arrival load client.
Commands below are relative to the repository root. Read
`docs/SESSION-ARRIVALS.md` for measurement definitions; don't recreate the protocol.

## 1. Cheap implementation qualification

```bash
python -m pip install -e '.[test,data,prepare]'
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
```

Pool initializer tests use tiny generated Parquet files and a fake tokenizer to
verify source identity, sequential selection, rejection accounting and exact
**accepted** counts. They are not actual Qwen tokenizer/data qualification.
Tests must run rather than skip when validating this workflow (`.[test]` includes
Parquet and template dependencies). Never execute commands or tools inside traces.

## 2. Initialize with actual source and local tokenizer

```bash
python scripts/init_trajectory_pool.py \
  --tokenizer /path/to/local/Qwen3.5-tokenizer \
  --count 512 --max-context 262144 --max-shards 2 \
  --dataset-cache prepared/open-swe-download --output prepared/pool-512
```

The illustrative shard bound is an investment limit, not a promise that two
shards suffice. Increase it explicitly if too few rows qualify. The initializer
fetches only needed shards from the pinned dataset revision, validates HF local
metadata and Parquet digest, screens whole rows, and compiles each until the
requested accepted count is reached. It does not download model weights or
remote tokenizer code. `--offline` uses existing verified shards only.

Check `receipt.json`: status=ready, accepted_count=requested_count, expected
revision and tokenizer identity. Inspect rejection counts/reasons and source rows
examined: no silent over-context truncation, padding, shuffle or cycling. A ready
pool has `workload.json` and attributed `source.json.gz`. Failed runs retain a
failure receipt; retain it, fix the cause and use a fresh output directory with
the same reusable download cache. TLS/auth failures are acquisition failures,
not evidence that token compilation or model serving passed. Do not weaken TLS.

## 3. Freeze the schedule and verify its meaning

Use the existing TraceLab profile generator/source attribution rather than
histogram-midpoint reconstruction. One provider; zero intervals and tails kept.

```bash
python -m swe_prefix_reuse.arrival_cli prepare \
  --workload prepared/pool-512/workload.json \
  --profile prepared/tracelab/profile.json.gz --provider codex \
  --rate 0.1 --duration 1800 --seed 20261002 --output prepared/plan.json
```

Check distinct trajectories suffice for the fixed clock, order matches the pool,
and each session has exactly turns-1 stored nonnegative gaps. Runtime uses
completion+gap, not nominal source-output offsets. Delay sampling is done only at
preparation. The first request is scheduled at zero; no randomly chosen content,
Poisson arrivals, hidden replenishment or recycling. Same pool+plan for comparisons.

## 4. CPU protocol/load-client checks

```bash
python scripts/probe_arrival_client.py --sessions 1000 --rate 100 \
  --prompt-tokens 1024 --output results/pool-protocol-probe.json
```

This probe uses artificial inputs and a fake HTTP server sharing the client loop;
it validates sleeping-pool timing/transport, not the actual pool or model score.
Unit HTTP fixtures separately verify actual returned-ID continuation, frozen gaps,
DP affinity, failures and drain. Inspect late dispatch/header sends, connector
queue, loop lag, CPU/RSS and missed deadlines. Do not automatically add threads or
processes merely because waiting sessions are numerous. Bound long-prompt checks
according to the actual expected request rate; don't infer a capacity ceiling.

## 5. Real serving qualification is a separate authorized step

Use only an explicitly selected existing endpoint or separately authorized server
resources. This Skill gives no permission to acquire/reset accelerators. Start
with a short plan using multiple unique real trajectories, verify exact-token
protocol and server cache/TTL/restore evidence, then use the intended measurement
window if authorized. Client-valid alone does not establish cache hits or state
correctness. The timer cannot distinguish HBM eviction from backing-state loss.

Retain pool/profile/plan identities, redacted server configuration, client version,
request records, CPU samples and server state metrics. Report offered vs actual
TTFT and incomplete sessions after drain. Suppress headline scores on protocol
failure or missed arrivals/due requests. Cold-start finite windows are not steady
state; show actual reached prompt lengths and traffic, not only server context cap.
Never publish fake-server throughput as inference performance.

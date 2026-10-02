# Fixed-rate SWE sessions with frozen completion-relative waits

This opt-in client is separate from the original closed-loop `run` protocol.
It does not launch or modify a model server. The earlier TraceLab-only
[research scaffold](session-arrivals/README.md) is not this accepted serving protocol.

## Protocol

1. Use a prepared SWE workload in its recorded order; no random trajectory choice.
2. New session `i` is scheduled at `i / new_sessions_per_second`, starting at zero.
   The arrival producer does not wait for inference completion. There is no fixed
   concurrency target and no automatic replacement of finished sessions.
3. Prepare one delay for each within-session transition by independently sampling
   the selected provider's empirical **adjacent-call-pair-weighted** TraceLab gap
   population, including zeros and the full tail. Freeze the delays in the plan.
   The seed and trajectory identity determine each delay sequence independently
   of other sessions or the offered arrival rate. Runtime does not sample.
4. After a response is complete and protocol-validated, send the next turn no
   earlier than **completion + stored delay**. One asyncio task owns one session,
   sleeps without polling, and retains the exact generated token history.
5. Every session play has a fresh cache salt and stable routing identity. Optional
   native DP affinity is session ordinal modulo DP size. No inter-session reuse.
6. The source pool must have `ceil(rate * duration)` distinct trajectories; fail
   before networking if insufficient. Do not silently loop the eight-row sample.
7. At the deadline, stop dispatching, wake waiting tasks and release their client
   memory; drain already in-flight calls under the HTTP timeout. Sleeping or
   partially completed sessions are reported as incomplete, not successful exits.
   Source exhaustion never changes the requested arrival rate.

**Modeling boundary:** TraceLab measures first-output-to-first-output intervals.
Fletcher explicitly selected using that empirical distribution as synthetic
post-completion wait times. These are not measured user/tool think times. This
changes the original timing interpretation intentionally; do not relabel the raw
statistics. Arrival order is fixed, while actual continuation times necessarily
vary with the backend's response completion times. Marginal gap sampling does not
preserve TraceLab's per-session timing correlations. SWE supplies content/depth.

Long waits create an inactive-session state working set. An expired HBM residency
can require restore; recomputation is required only when reusable backing state
is also unavailable. Waiting sessions alone do not prove a server is pressured.
Measure server-tier residency, transfers, TTL behavior, cache hits and recomputation
separately. Short SWE first prompts are reported honestly, not padded to make P busy.

## Prepare content and an immutable arrival plan

Existing content preparation and strict token protocol are unchanged. To expand
beyond the eight-row example, first obtain contiguous pinned Open-SWE shards and
Hugging Face `local_dir` metadata (revision in `data/README.md`). Then:

```bash
pip install -e '.[test,data,prepare]'
python scripts/expand_swe.py --dataset /path/to/Open-SWE-Traces \
  --count 512 --output prepared/swe-512.json.gz
swe-prefix-reuse prepare --source prepared/swe-512.json.gz \
  --tokenizer /path/to/Qwen3.5-35B-A3B --max-context 262144 \
  --output prepared/qwen35-expanded.json
# Inspect whole-trajectory rejections; the accepted count, not source count, matters.

# TraceLab profile preparation/source/license are documented in session-arrivals/README.md.
python -m swe_prefix_reuse.arrival_cli prepare \
  --workload prepared/qwen35-expanded.json --profile prepared/tracelab/profile.json.gz \
  --provider codex --rate 0.1 --duration 1800 --seed 20261002 \
  --output prepared/arrival-plan.json
```

`expand_swe.py` checks pinned shard content identity, walks contiguous shards/rows
in order, rejects credential-like content, duplicate identities and fewer-than-two-
turn records, and records exclusions. It does not execute source tools, truncate
or pad text, or claim secret screening is perfect. Dataset derivatives retain
NVIDIA Open-SWE's CC BY4.0 attribution. Runtime token compilation rejects whole
unsupported/over-context trajectories. No anonymous source reshuffle is allowed.

Local expansion status2026-10-02: the pinned HF download failed TLS handshakes
with both curl and Python; no expanded real-data bundle is claimed yet. The
expander is CPU-fixture-tested; it needs accessible source shards to complete
real-data preparation. The original eight rows are only protocol material.

## Run against an explicitly selected existing endpoint

```bash
python -m swe_prefix_reuse.arrival_cli run \
  --workload prepared/qwen35-expanded.json --plan prepared/arrival-plan.json \
  --endpoint http://127.0.0.1:8000/v1/completions --model MODEL \
  --server-max-context 262144 --chips 2 --connections 256 \
  --server-metadata /path/to/redacted-server-metadata.json \
  --output results/arrival-0.1
```

After reinstall, `swe-session-arrivals` is an equivalent entrypoint. Plan binds
prepared workload SHA256; output directories refuse overwrite. Credential-bearing
URLs are rejected; API key remains environment-only via the existing client.
No generated IDs are duplicated into result logs: exact submitted/returned IDs
are checked during replay and prompt digests retained. The live session owns its
history; terminated sessions release it. Scalar per-request records and periodic
samples remain in client memory, so arbitrarily long tests still need a memory
budget. There is no model-text semantic or agent correctness claim.

## Detect client-induced latency rather than hiding it

- Per-request planned send time; dispatch and headers-sent lateness, separately
  summarized for first calls and continuations.
- Connection-pool wait measured through aiohttp tracing. `connections` caps sockets,
  not new-session arrivals; queued HTTP requests remain visible as client delay.
  Reported in-flight includes tasks waiting for a connection, not server admission.
- JSON serialization time, prompt preparation, record-building/hash overhead,
  total record sink time, process CPU time and lifetime peak RSS (platform units).
- Event-loop timer lag and sampled waiting/in-flight/completed sessions.
- TTFT from request dispatch, plus **offered TTFT from planned send**, so a busy
  client cannot conceal its wait by resetting the latency clock.
- Missed due requests or missed planned session launches invalidate headline
  throughput; protocol/transport failures do too. In-flight drain tokens after
  the observation deadline are excluded from throughput.

JSON parsing/validation time is not yet separately profiled; it contributes to CPU
and loop lag. Request dispatch precedes encoding/socket acquisition; headers-sent
is an additional transport-stage observation, not server receipt. Samples are not
an exact integral of pool occupancy. No automatic concurrency or rate reduction.

## Bounded CPU qualification

`pytest -q` covers exact actual-output continuation, completion-relative delays,
immutable plan/order, insufficient pool rejection, DP affinity, connection queue,
long sleeping-tail cutoff, protocol failure, and the original closed-loop tests.

```bash
python scripts/probe_arrival_client.py --sessions 1000 --rate 100 \
  --prompt-tokens 1024 --output results/arrival-client-1000.json
python scripts/probe_arrival_client.py --sessions 100 --rate 20 \
  --prompt-tokens 32768 --output results/arrival-client-long-input.json
```

Observed on this Mac: first case1000/1000 successful first calls, sampled waiting
peak996, dispatch lateness P95~1.04ms, loop lag P95~0.77ms; max dispatch lateness
~70ms (do not hide this tail). Long-input case100/100 successful, dispatch P95~1.33ms,
loop lag P95~1.79ms, JSON encoding P95~1.58ms. Raw receipts are ignored under
`results/`. Both retain dormant sessions until deadline and deliberately do not
send second turns. Continuation behavior is covered separately by HTTP tests.

**These are protocol fixtures sharing the same event loop/CPU with the fake server,
not standalone client-capacity ceilings, model throughput, or deployment qualification.**
No accelerator experiment, PD backend modification, or public publication occurred.

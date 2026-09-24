# Qualify token continuation without confusing it with serving performance

Use when changing the compiler/client or qualifying another tokenizer/server. The
public protocol is in README.md and docs/UPSTREAM.md; read those first rather than
re-deriving AgentX or an independent request-rate runner.

## Paid observations

- Qwen3.5-35B-A3B's thinking-enabled template extended previous full histories for
  all 8 bundled mini-SWE traces (360 turns). With thinking disabled, the generation
  prefix adds an empty closing think block which is not a prefix of original
  reasoning-bearing assistant serialization. Do not fix this by dropping reasoning
  or retokenizing generated text. A different template policy is a workload change.
- Original `tools` entries and `tool_calls[*].function.arguments` may be JSON
  strings. The compiler normalizes them before Qwen rendering; leaving arguments
  as strings raises a Jinja mapping error.
- Original source Parquet files are nested under subset/scenario directories. The
  pinned download's `.cache/huggingface/download/*.metadata` records dataset commit
  and content identity. Running `git rev-parse` in a download directory can return
  an enclosing workspace commit: that is NOT the dataset revision.
- `return_token_ids` in the pinned vLLM protocol supplies both delta output IDs and
  a prompt-ID echo. Echo verification is useful but can inflate client/transport
  cost at long context. Do not silently drop IDs or compare with a text-only client.
- Trace recycling needs a new session salt; salt stays constant within a session.
  This tests within-session reuse only. A correct prefix is not proof the backend
  actually caches it, especially for hybrid/recurrent architectures.

## Cheap checks before accelerator work

1. Run `pytest -q` and Ruff. The real local HTTP/SSE fixture verifies framing,
   continuation, replenishment, strict failure and deadline/drain accounting.
   Its timings are NOT inference scores.
2. Compile the bundled data with the actual target tokenizer. Whole-history rewrite
   or over-context sessions must be rejected, never silently shortened. Inspect
   rejections and the actual observed request lengths, not just the 256K server flag.
3. Real-engine qualification is still required for a deployed engine/version. Use
   externally authorized idle resources, a short bounded probe, and preserve that
   deployment's acquisition/ownership rules. No benchmark task creates authority to
   disturb another server or import accelerator-initializing modules unleased.
4. Check strict protocol success, prompt identity, actual in-flight occupancy,
   server-side prefix-cache evidence, client bottlenecks and repeatability before
   comparing performance. The initial v0.1 release was CPU protocol + real
   tokenizer only; the bounded real-engine observation below is not universal
   server qualification.

## Real-engine continuation observation (2026-09-24)

Tool commit `59ea20a` (0.1.1), Qwen3.5-35B-A3B BF16, native vLLM 0.25.1 /
Ascend 0.25.1rc1 TP2 with natural MTP2 completed one 900-second C4 window:
367 requests, zero failures, all exact-token checks passed, 99.85% full client
concurrency and maximum observed prompt 80,630 tokens. Separate C2 qualification
observed cached prompt tokens; server prefix counters also increased during C4.
The server exited cleanly and the owned-card release was observed. This supports
this deployment's protocol/continuation compatibility, not SWE answer quality,
full-256K performance, repeatability or other engine versions.

Local retained evidence is
`/workspace/my-ascend-workspace/runs/swe-frontier-sequential-20260924/native8/`:
`qualification/`, `c4/`, `receipt.json`, metric snapshots and `release.json`.
The prepared-workload identity is in `c4/config.json`; never substitute an earlier
prepared sample when continuing the campaign.

For multi-rank HTTP relays, preserve **both** cache salt and routing affinity.
The qualified eight-chip relay requires `X-Correlation-ID`; 0.1.1 sends the
session salt as that header. A cache salt alone cannot prevent a load balancer
from routing successive turns to different replicas. Header support was verified
by the CPU protocol fixture; an eight-chip inference result must still establish
its own cache/ownership evidence before publication.

## Raising a serving limit also needs graph-coverage review

On the same pinned native TP2 deployment, raising only `max_num_seqs` from16 to32
retained a graph-capture maximum of48 tokens. With MTP2, a full32-lane decode batch
needs96 tokens; the pinned vLLM `v1/cudagraph_dispatcher.py` explicitly returns
`CUDAGraphMode.NONE` above the configured maximum. Check this before spending a
measurement window: an admitted request slot is not evidence of graph coverage.

Two separate900s observations retained the24.25GiB/chip KV budget and passed C32
protocol, prefix-reuse, no-preemption and cleanup gates. Original max48 capture
measured135.53 tokens/s/chip and P90 decode9.72; adding96 capture measured207.88
and21.01, with peak KV65.1%. This supports a consequential route/configuration
effect, not a precise causal speedup estimate on a shared host. Even the latter
window did not beat the capacity16 C16 observation (221.95 and38.93). Capacity,
throughput and per-request speed are different judgments; more slots need not
improve the observed tradeoff.

Retained evidence is in the same campaign root under `native32-c32/` and
`native32-c32-graph96/`; each keeps its actual graph configuration and result.
Do not join these points to a server-limit16 concurrency line. The capacity gate
observes actual running requests and preemption counters, fails closed on missing
counters, and does not certify32 simultaneous full-256K contexts.

Local preparation evidence from initial development lives outside the repository
at `/workspace/my-ascend-workspace/runs/swe-token-continuation/`. Portable provenance
and shape observations live in data/README.md; no local path is required to run the
public tool. Do not recover an obsolete unscreened sample from that local directory:
only the checked-in bundle is the accepted input.

## Native internal-DP affinity

The pinned vLLM OpenAI completions endpoint reads `X-data-parallel-rank`;
`X-Correlation-ID` is not its native routing contract. Two real Qwen35 native
DP2 deployments (expertTP2 andEP2) answered cold retrieval correctly but returned
zero cached tokens on the following warm request when the controller left rank
selection to internal load balancing. Do not mistake a stable cache salt for a
stable cache owner.

The opt-in `--data-parallel-size N` sends lane modulo N as that native header,
keeping a lane on the same rank through turns and recycled sessions. It preserves
prepared prompts, exact output budgets, timing and fresh-play salts. Artifacts
record declared DP size, policy and per-request rank. CPU HTTP fixtures verify
opt-in behavior, lane distribution and replacement continuity; real engine
cache/load acceptance remains required. Servers may ignore unsupported headers.

Bounded real-engine check: client860e7c9 (0.1.2), the same prepared SWE file,
Qwen35 native BF16/MTP2/256K DP2+EP2, C2/60s completed13 strict-protocol requests,
99.987% full client concurrency,5 requests with cached input (maximum6144).
Both rank-specific cache counters increased; server exited0. Evidence:
workspace `runs/qwen35-parallel-matrix-20260924/native-dp2-ep2-swe-protocol/`.
This qualifies this deployment's routing/protocol only, not other servers or a
Frontier performance window.

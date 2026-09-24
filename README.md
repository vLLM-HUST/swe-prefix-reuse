# SWE Prefix Reuse

A small closed-loop serving benchmark for **long conversations with real prefix
reuse**, adapted from [`vllm bench serve`](docs/UPSTREAM.md).

Keep original SWE input increments and fixed per-turn output budgets. Let the model
generate freely, append its **actual token IDs**, then add the next recorded input.
Maintain C active request lanes without replaying human/tool delays.

**Not SWE task solving, agent accuracy, or faithful online agent execution.** Recorded
tool observations need not agree with the newly generated answer. This deliberately
measures a fixed serving shape, not the correctness of an agent's decisions.

## Quick start

Requires Python 3.10+. The client does not require vLLM, torch, or accelerator drivers.
Preparation needs a locally available Hugging Face tokenizer, not model weights.

```bash
git clone https://github.com/vLLM-HUST/swe-prefix-reuse.git
cd swe-prefix-reuse
python -m venv .venv
source .venv/bin/activate
pip install -e '.[prepare]'

swe-prefix-reuse prepare \
  --source data/open-swe-sample.json.gz \
  --tokenizer /path/to/Qwen3.5-35B-A3B \
  --max-context 262144 \
  --output prepared/qwen35.json

swe-prefix-reuse run \
  --workload prepared/qwen35.json \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model YOUR_SERVED_MODEL_NAME \
  --server-max-context 262144 \
  --concurrency 8 --duration 60 --chips 2 \
  --server-metadata /path/to/server-metadata.json \
  --output results/c8-smoke
```

The run command **does not launch or reconfigure a server**. Set the actual server
capacity and tokenizer consistently with the prepared workload. Do not claim BF16,
MTP, prefix caching or a chip type from client flags: those belong to the server
configuration and recorded metadata. `--chips` is the total participating accelerator
count, not TP if there are multiple replicas. An API key may be supplied through
`OPENAI_API_KEY`; it is not saved. Credential-bearing endpoint URLs are rejected.

Server requirements: `/v1/completions` with integer token prompts, streaming delta
`token_ids` through `return_token_ids=true`, prompt-ID echo, final usage,
`ignore_eos=true`, and `cache_salt`. The source protocol was checked in vLLM
`752a3a5`; other versions/compatible servers must satisfy the same strict checks.
There is no text-retokenization fallback and no engine patch required for that API.

`--server-metadata` is optional for local diagnostics but necessary before sharing a
comparison. Supply a JSON object containing at least engine/version/commit, mod and
version, model/revision, precision, tokenizer identity, chip model/count, and the
**redacted** server launch command. Never include API keys or environment dumps.

## What is fixed?

Preparation uses the target tokenizer and thinking-enabled chat template to render
original histories. It extracts:

- initial input and each subsequent **new input segment**;
- each original assistant body's token count as that turn's generation budget;
- structural assistant closing delimiters, moved into the next fixed input segment.

Each segment is tokenized independently once. Templates that rewrite previous
history are rejected rather than quietly breaking the prefix. Tool argument JSON
strings are normalized to mappings for rendering. Whole sessions beyond the context
limit are rejected, never truncated. Rejections are recorded; all rejected means
preparation fails. Review them: a changed accepted subset is a changed workload.

At runtime:

```text
turn 0: initial input                            → actual output IDs, exactly N₀
turn 1: previous input + actual output + delta₁  → actual output IDs, exactly N₁
turn 2: previous input + actual output + delta₂  → ...
```

The historical prefix is never decoded/re-encoded or re-rendered. The prompt-ID echo
and server usage are checked against the exact submitted IDs. Early termination,
missing IDs, usage mismatch, an incomplete stream, or any request failure invalidates
the run and suppresses headline performance values.

Use the **same prepared file** for baseline/mod comparisons. Its SHA-256 and tokenizer
fingerprint are saved with results. Different tokenizers, compiler policies or accepted
sessions create different workloads, not directly comparable points. Reusing a prepared
file with an incompatible server tokenizer is an operator error: token echo alone
cannot prove vocabulary identity.

## Load and cache policy

- C sequential lanes; at most one in-flight request per lane. A next turn is sent
  immediately when the previous response has been validated.
- A finished session is replaced immediately from the fixed cyclic source queue.
  No spread, arrival-rate schedule, simulated tool execution or think time.
- Each session play has a fresh `cache_salt`, unchanged across its own turns. This
  preserves within-session reuse while preventing artificial full-cache hits on
  repeated traces. **Cross-session common-prefix reuse is intentionally excluded.**
- The run starts from fresh salts with **no warmup**. A fixed duration stops new
  requests, then drains already in-flight requests without starting another turn.
  Drain time/tokens are not secretly added to the measurement window. Timeout is
  per request (`--timeout`, default 1800 seconds), not a global run deadline.
- Client scheduling, JSON serialization, validation and network overhead still exist.
  Actual client-inflight occupancy is reported; C is not proof of C engine-active
  decodes. Do not describe an underfilled or client-limited run as engine peak throughput.

The bundled [source sample](data/README.md) is only ~397 KB. With the tested Qwen
tokenizer it spans final cumulative lengths of roughly 15K–141K. **A short smoke
may never reach its long turns.** Check `max_prompt_tokens_observed`, turn progress,
completed sessions and raw requests before making any long-context claim. The cold
start and finite window also change the mixture at different C: compare equal-C
baseline/mod runs first; use sufficient coverage before interpreting a C curve.

## Metrics and artifacts

`summary.json` reports:

- **Output tokens/s/chip:** actual generated IDs received within the timed window,
  divided by its wall-clock duration and chip count. Portions of drained requests
  received before the deadline count; portions after it do not.
- **Decode speed P90:** P90 across fully completed in-window requests of
  `(output_tokens - 1) / (last_token_time - first_token_time)`.
  This is not `1 / P90(TPOT)`, nor output tokens divided by prefill-inclusive latency.
  A single output chunk has no measurable decode interval and is excluded; an empty
  sample yields `null`, not a fabricated zero or infinity.
- TTFT P95, sample counts, drain duration, actual prompt lengths/turn coverage,
  mean client-inflight requests, and fraction of time with all C requests in flight.

Timing is client-observed at token-bearing SSE chunks. MTP may put several tokens
in one chunk; we count all IDs but do not invent individual token timestamps.
Returning token IDs and echoing a long prompt add transfer/JSON overhead; both
baseline and mod must use the same client/protocol. These are serving measurements,
not kernel-only counters. See [intentional differences from upstream](docs/UPSTREAM.md).

`config.json` saves workload identity and declared server/client configuration.
`requests.jsonl` saves actual output IDs, prompt fingerprints, per-chunk token counts
and relative monotonic timestamps, lane/play/turn identity, usage and errors. Together
with the prepared workload these allow exact prompt reconstruction. Results can be
large: requests are buffered until run end. Use bounded runs, not an unattended
multi-hour campaign. Treat artifacts as potentially sensitive generated content.

`valid=true` means the protocol/shape checks passed; it is **not** certification of
hardware identity, steady state, sufficient samples, real cache hits, or publication
readiness. Cache-hit behavior depends on the engine/model and must be checked with
server-side cache metrics when qualifying a deployment. Old AgentX scores must not
be relabelled as this workload. This tool does not submit to a leaderboard.

## Validation and development

```bash
pip install -e '.[test]'
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
```

Tests include a real local HTTP/SSE **protocol fixture**, not an inference model:
fragmented UTF-8/CRLF, budget/echo/usage failures, real-output prefix continuation,
C-way load, session replenishment, salt isolation and window/drain accounting.
The bundled trajectories have also been compiled with the actual Qwen3.5 tokenizer.
**No accelerator performance score or real-engine qualification is shipped in v0.1.**

Code: [Apache-2.0](LICENSE). Adapted upstream attribution: [NOTICE](NOTICE).
Bundled data: [CC-BY-4.0 with NVIDIA attribution](data/README.md).

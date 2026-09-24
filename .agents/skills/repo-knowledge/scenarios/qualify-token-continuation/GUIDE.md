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
   comparing performance. v0.1's published verification is CPU protocol + real
   tokenizer only, not accelerator qualification.

Local preparation evidence from initial development lives outside the repository
at `/workspace/my-ascend-workspace/runs/swe-token-continuation/`. Portable provenance
and shape observations live in data/README.md; no local path is required to run the
public tool. Do not recover an obsolete unscreened sample from that local directory:
only the checked-in bundle is the accepted input.

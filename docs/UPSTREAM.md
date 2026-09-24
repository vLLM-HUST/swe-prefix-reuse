# Adaptation boundary

This is a small, independent measurement client, **not an official vLLM benchmark
mode** and not a new inference engine. It does not import vLLM or require PyTorch.

Upstream baseline: **vLLM `752a3a504485790a2e8491cacbb35c137339ad34`**.

- [Streaming request implementation](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/benchmarks/lib/endpoint_request_func.py):
  `async_request_openai_completions`, `RequestFuncOutput`, `StreamedResponseHandler`.
  `client.py` adapts their aiohttp POST/stream loop, incremental UTF-8 buffering,
  monotonic timing, and request measurement shape. Upstream blob identity:
  `59cbc0e2e6c43211517ada2b171687405065b11b`.
- [Serving benchmark](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/benchmarks/serve.py):
  retain the no-arrival-delay, bounded-concurrency intent and the per-request
  `TPOT = (E2E - TTFT) / (output_tokens - 1)` definition.
- [Completions protocol](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/entrypoints/openai/completion/protocol.py):
  `return_token_ids` returns delta token IDs and an initial prompt-ID echo;
  `cache_salt` isolates independent session plays.

## Deliberate differences

1. Upstream prebuilds independent requests. We keep C live sequential sessions,
   constructing each next prompt from actual returned IDs plus a fixed delta.
   Finished sessions are replaced from a cyclic source queue, with no think/tool wait.
2. Our narrow client retains token IDs rather than generated text, and requires exact
   prompt echo, usage counts, output budget, `finish_reason=length`, and `[DONE]`.
   A failed request invalidates the entire run's headline metrics.
3. SSE framing waits for complete events, preserves transport whitespace, accepts
   CRLF, and does not dispatch merely because a partial read happens to parse as JSON.
   Tokenless choice chunks do not start TTFT or extend decode latency.
4. TTFT and decode intervals are **client-observed token-bearing chunk times**.
   Multiple tokens per chunk (e.g. MTP) are counted, not assigned imaginary individual
   emission times. We intentionally do not expose a per-token ITL percentile.
5. `bench serve`'s finite request batch uses completed tokens / batch duration. This
   tool counts tokens received inside a fixed window, including portions of requests
   that finish during drain, divided by the window duration. No post-window token
   is credited. These throughput definitions must not be silently mixed.
6. Speed P90 is the percentile of per-request `(N - 1) / (E2E - TTFT)` among requests
   fully completed inside the window, **not** `1 / P90(TPOT)`. N=1 or all-tokens-in-one-
   timestamp requests have no measurable decode interval and are excluded.

The client is intentionally adapted, not a verbatim vendor copy and not a wrapper
around private Python imports. This avoids installing a GPU/NPU engine just to
run an HTTP load client. Keep upstream attribution when changing it.

# Build fixed-arrival SWE workloads and diagnose client overhead

Enter for new sessions/s, inactive waiting pools, TraceLab statistics or the new
async client. Read `docs/SESSION-ARRIVALS.md` (repo-root relative) for the accepted
protocol, preparation/run commands and observed CPU qualification. Read
`docs/session-arrivals/README.md` only for the earlier synthesis feasibility work;
its absolute nominal-output-offset experiment is NOT the accepted HTTP protocol.

Fletcher selected fixed SWE trajectory order, uniform new-session clock, and
**completion + pre-sampled empirical delay** for continuations. One asyncio task
per live session; do not introduce one OS thread per sleeper or random trajectory
selection. A frozen plan binds the prepared workload and refuses an insufficient
pool; do not silently recycle the eight-row example. Gap source is first-output
intervals repurposed as synthetic post-completion waits, not measured think time.

Use `arrival_plan`, `arrival_runner`, `arrival_cli`; legacy closed-loop remains
unchanged. `scripts/session_workload.py profile` reuses workshop statistics on the
fixed TraceLab release and includes singleton/zero/tail observations. The full
profile/raw trace and generated plans remain ignored under `prepared/`; compact
source/statistics/audit are attributed in `docs/session-arrivals/`.

Run pytest and Ruff, then bounded `scripts/probe_arrival_client.py` only when a
new client-performance risk requires it. Fake server shares the client loop/CPU:
its output is protocol evidence, NOT model throughput or a capacity ceiling.
Inspect dispatch/headers lateness, connection queue, loop lag, CPU/RSS and logging
cost before adding processes. Long prompts and response parsing can cost more
than dormant timers. Never use these commands as authority to start an NPU server.

Content expansion uses `scripts/expand_swe.py` with pinned HF download metadata,
sequential complete rows, identity/credential screening and explicit exclusions.
Actual expanded source acquisition remained blocked by HF TLS on2026-10-02;
no expanded real-model run has been claimed. Don't mislabel a fixture or the
original8 trajectories as the expanded dataset. Preserve source licenses and do
not execute trace tool calls.

The initialization entrypoint is now `scripts/init_trajectory_pool.py`: downloads
on demand and counts AFTER actual tokenizer/context qualification. For repeatable
pool testing use the sibling [test-trajectory-pool](../test-trajectory-pool/GUIDE.md)
scenario rather than manually composing acquisition and acceptance rules.

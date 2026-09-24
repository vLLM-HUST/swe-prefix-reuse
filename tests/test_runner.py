import asyncio

import pytest
from test_client import endpoint, reply

from swe_prefix_reuse.prepare import SCHEMA
from swe_prefix_reuse.runner import percentile, replay, summarize


def workload():
    return {
        "schema": SCHEMA,
        "max_context": 100,
        "sessions": [
            {
                "trajectory_id": "trace",
                "instance_id": "task",
                "max_context_tokens": 9,
                "turns": [
                    {"input_ids": [10, 11], "prompt_tokens": 2, "output_tokens": 3},
                    {"input_ids": [12], "prompt_tokens": 6, "output_tokens": 3},
                ],
            }
        ],
    }


@pytest.mark.parametrize("dp_size", [None, 1, 2])
def test_native_dp_affinity_is_opt_in_and_survives_session_replacement(dp_size):
    async def check():
        seen = []

        async def handler(req):
            body = await req.json()
            seen.append((body["cache_salt"], req.headers.get("X-data-parallel-rank")))
            return await reply(req, delay=0.001)

        async with endpoint(handler) as url:
            summary, records = await replay(
                workload(),
                url=url,
                model="fixture",
                concurrency=4,
                duration=0.12,
                chips=2,
                data_parallel_size=dp_size,
                run_id="affinity-test",
            )
        assert summary["valid"]
        assert len({salt for salt, _ in seen}) > 4
        for salt, rank in seen:
            lane = int(salt.split(":")[1])
            assert rank == (None if dp_size is None else str(lane % dp_size))
        for row in records:
            assert row["data_parallel_rank"] == (None if dp_size is None else row["lane"] % dp_size)

    asyncio.run(check())


@pytest.mark.parametrize("size", [0, -1, 3, 1.5, True])
def test_invalid_dp_size_rejected_before_network(size):
    with pytest.raises(ValueError, match="data_parallel_size"):
        asyncio.run(
            replay(
                workload(),
                url="unused",
                model="fixture",
                concurrency=1,
                duration=1,
                chips=2,
                data_parallel_size=size,
            )
        )


def test_real_http_closed_loop_prefix_salts_and_concurrency():
    async def check():
        payloads, active, peak = [], 0, 0

        async def handler(req):
            nonlocal active, peak
            payloads.append(await req.json())
            active += 1
            peak = max(peak, active)
            try:
                return await reply(req, delay=0.002)
            finally:
                active -= 1

        async with endpoint(handler) as url:
            summary, records = await replay(
                workload(),
                url=url,
                model="fixture-not-a-model",
                concurrency=4,
                duration=0.15,
                chips=2,
            )
        assert summary["valid"]
        assert peak == 4
        assert summary["mean_client_inflight"] > 2
        assert summary["requests_started"] == len(records)
        assert summary["requests_drained"] <= 4
        assert (
            summary["output_tokens_per_second_per_chip"] == summary["output_tokens_per_second"] / 2
        )
        by_salt = {}
        for payload in payloads:
            by_salt.setdefault(payload["cache_salt"], []).append(payload)
            assert payload["ignore_eos"] is True
            assert payload["return_token_ids"] is True
        assert len(by_salt) > 4  # sessions are replenished, not only C finite traces
        pairs = [p for p in by_salt.values() if len(p) == 2]
        assert pairs
        assert all(len(p) <= 2 for p in by_salt.values())
        for first, second in pairs:
            assert second["prompt"] == first["prompt"] + [700, 701, 702] + [12]
        assert all(row["start"] < 0.15 for row in records)

    asyncio.run(check())


def test_window_uses_chunks_not_completed_request_total():
    row = {
        "success": True,
        "start": 0.1,
        "end": 1.5,
        "chunks": [(0.5, 5), (1.2, 7)],
        "ttft_seconds": 0.4,
        "decode_tokens_per_second": 10,
    }
    summary = summarize([row], 0, 1, 1, 2)
    assert summary["observed_output_tokens_in_window"] == 5
    assert summary["output_tokens_per_second_per_chip"] == 2.5
    assert summary["requests_completed_in_window"] == 0
    assert summary["decode_tokens_per_second_p90"] is None
    assert summary["full_concurrency_fraction"] == 0.9
    assert summary["drain_seconds"] == 0.5
    row["success"] = False
    assert summarize([row], 0, 1, 1, 2)["output_tokens_per_second"] is None


def test_percentile_is_of_speed_not_inverse_slow_latency_percentile():
    assert percentile([2, 10], 0.9) == 9.2
    assert percentile([], 0.9) is None


def test_protocol_failure_invalidates_run_and_stops_replenishment():
    async def check():
        async def handler(req):
            return await reply(req, fault="usage")

        async with endpoint(handler) as url:
            summary, records = await replay(
                workload(), url=url, model="fixture", concurrency=2, duration=10, chips=1
            )
        assert not summary["valid"]
        assert summary["aborted"]
        assert summary["output_tokens_per_second"] is None
        assert len(records) == 2

    asyncio.run(check())


def test_installed_cli_writes_auditable_artifacts(tmp_path):
    import json
    import sys

    async def check():
        fixture = workload()
        fixture["tokenizer"] = {"fingerprint": "protocol-fixture-only"}
        source = tmp_path / "workload.json"
        source.write_text(json.dumps(fixture))
        destination = tmp_path / "results"

        async def handler(req):
            return await reply(req, delay=0.001)

        async with endpoint(handler) as url:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "swe_prefix_reuse.cli",
                "run",
                "--workload",
                str(source),
                "--endpoint",
                url,
                "--model",
                "protocol-fixture",
                "--server-max-context",
                "100",
                "--concurrency",
                "2",
                "--duration",
                "0.08",
                "--chips",
                "1",
                "--output",
                str(destination),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            assert process.returncode == 0, stderr.decode()
            assert json.loads(stdout)["valid"]
        config = json.loads((destination / "config.json").read_text())
        summary = json.loads((destination / "summary.json").read_text())
        records = [
            json.loads(line) for line in (destination / "requests.jsonl").read_text().splitlines()
        ]
        assert config["concurrency"] == 2
        assert summary["valid"] and summary["max_prompt_tokens_observed"] == 6
        assert summary["sessions_completed"] > 0
        assert records and all(r["success"] for r in records)
        assert all(len(r["prompt_sha256"]) == 64 and len(r["token_ids"]) == 3 for r in records)

    asyncio.run(check())

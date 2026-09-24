# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 vLLM-HUST contributors
"""Adapted from vLLM's endpoint_request_func.py; see docs/UPSTREAM.md.

Completions-only: preserve actual token IDs, validate the complete stream, and
measure token-bearing chunks. No text retokenization or per-token timing fiction.
"""

import codecs
import hashlib
import json
import os
import time
from dataclasses import dataclass, field


class StreamedResponseHandler:
    """vLLM's incremental UTF-8/buffer design, with SSE boundary-only dispatch.

    Do not strip transport chunks: a separator or UTF-8 character can cross reads.
    CRLF and comment/keepalive events are accepted. JSON is parsed only after an
    event boundary, not as soon as a partial transport read happens to be JSON.
    """

    def __init__(self):
        self.buffer = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")()

    def add_chunk(self, chunk):
        self.buffer += self.decoder.decode(chunk)
        self.buffer = self.buffer.replace("\r\n", "\n")
        messages = []
        while "\n\n" in self.buffer:
            event, self.buffer = self.buffer.split("\n\n", 1)
            data = [line[5:].lstrip(" ") for line in event.split("\n") if line.startswith("data:")]
            if data:
                messages.append("\n".join(data))
        return messages

    def finish(self):
        self.buffer += self.decoder.decode(b"", final=True)
        if self.buffer.strip():
            raise ValueError("incomplete SSE event at EOF")


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


@dataclass
class RequestOutput:
    start: float = 0.0
    end: float = 0.0
    first_token: float | None = None
    last_token: float | None = None
    token_ids: list[int] = field(default_factory=list)
    chunks: list[tuple[float, int]] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    success: bool = False
    error: str = ""

    def record(self, prompt):
        ttft = None if self.first_token is None else self.first_token - self.start
        latency = None if self.last_token is None else self.last_token - self.start
        # Match bench serve's per-request TPOT denominator (N - 1), not N / E2E.
        decode_seconds = None if ttft is None else latency - ttft
        speed = (
            (len(self.token_ids) - 1) / decode_seconds
            if len(self.token_ids) > 1 and decode_seconds > 0
            else None
        )
        return {
            **self.__dict__,
            "prompt_tokens": len(prompt),
            "prompt_sha256": token_digest(prompt),
            "ttft_seconds": ttft,
            "e2e_seconds": latency,
            "decode_tokens_per_second": speed,
        }


async def request(session, url, model, prompt, budget, salt, seed):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": budget,
        "ignore_eos": True,
        "temperature": 0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_salt": salt,
    }
    headers = {}
    if key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    output = RequestOutput(start=time.perf_counter())
    done, echoed, finished = False, False, False
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}: {response.reason}")
            parser = StreamedResponseHandler()
            async for chunk in response.content.iter_any():
                for message in parser.add_chunk(chunk):
                    if done:
                        raise ValueError("data after DONE")
                    if message == "[DONE]":
                        done = True
                        continue
                    data = json.loads(message)
                    if data.get("error"):
                        raise ValueError("server returned a streaming error")
                    if data.get("usage"):
                        output.usage = data["usage"]
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    if len(choices) != 1 or choices[0].get("index", 0) != 0:
                        raise ValueError("expected exactly one completion")
                    choice = choices[0]
                    if choice.get("prompt_token_ids") is not None:
                        if choice["prompt_token_ids"] != prompt:
                            raise ValueError("server prompt token echo differs from exact input")
                        echoed = True
                    ids = choice.get("token_ids")
                    if ids is not None and (
                        not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids)
                    ):
                        raise ValueError("invalid generated token IDs")
                    if ids:
                        if finished:
                            raise ValueError("tokens after finish_reason")
                        timestamp = time.perf_counter()
                        if output.first_token is None:
                            output.first_token = timestamp
                        output.last_token = timestamp
                        output.token_ids.extend(ids)
                        output.chunks.append((timestamp, len(ids)))
                        if len(output.token_ids) > budget:
                            raise ValueError("generated more tokens than requested")
                    if choice.get("finish_reason") is not None:
                        output.finish_reason = choice["finish_reason"]
                        finished = True
            parser.finish()
        if not done or not echoed or output.finish_reason != "length":
            raise ValueError("missing DONE, exact prompt echo, or length finish")
        if len(output.token_ids) != budget or output.usage.get("completion_tokens") != budget:
            raise ValueError("output token IDs / usage do not match fixed budget")
        if output.usage.get("prompt_tokens") != len(prompt):
            raise ValueError("prompt usage differs from exact input length")
        output.success = True
    except Exception as exc:  # noqa: BLE001 — protocol/transport failures invalidate the run
        # Avoid logging response bodies/headers or credential-bearing URLs.
        output.error = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
    finally:
        output.end = time.perf_counter()
    return output

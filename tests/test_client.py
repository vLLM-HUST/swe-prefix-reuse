"""Protocol fixtures only. Nothing in these tests is a model performance result."""

import asyncio
import json
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web

from swe_prefix_reuse.client import StreamedResponseHandler, request


def event(data):
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return ("data: " + text + "\r\n\r\n").encode()


@asynccontextmanager
async def endpoint(handler):
    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1/completions"
    finally:
        await server.cleanup()


async def reply(req, *, fault=None, delay=0):
    body = await req.json()
    n = body["max_tokens"]
    stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await stream.prepare(req)
    echo = body["prompt"] if fault != "echo" else [999]
    await stream.write(
        event({"choices": [{"index": 0, "text": "", "token_ids": [], "prompt_token_ids": echo}]})
    )
    for i in range(n):
        await asyncio.sleep(delay)
        choice = {"index": 0, "text": "你", "token_ids": [700 + i]}
        if fault == "ids":
            choice.pop("token_ids")
        data = event({"choices": [choice]})
        # Exercise arbitrary byte splits, including UTF-8 and CRLF separators.
        for j in range(0, len(data), 7):
            await stream.write(data[j : j + 7])
    await stream.write(
        event(
            {
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "token_ids": [],
                        "finish_reason": "stop" if fault == "finish" else "length",
                    }
                ]
            }
        )
    )
    await stream.write(
        event(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": len(body["prompt"]),
                    "completion_tokens": n + 1 if fault == "usage" else n,
                },
            }
        )
    )
    if fault != "done":
        await stream.write(event("[DONE]"))
    await stream.write_eof()
    return stream


def test_stream_parser_every_byte_and_comments():
    parser = StreamedResponseHandler()
    output = []
    for byte in b": ping\r\n\r\n" + event({"text": "你好"}) + event("[DONE]"):
        output.extend(parser.add_chunk(bytes([byte])))
    parser.finish()
    assert json.loads(output[0]) == {"text": "你好"}
    assert output[1] == "[DONE]"
    with pytest.raises(ValueError, match="incomplete"):
        parser.add_chunk(b"data: {}")
        parser.finish()


@pytest.mark.parametrize("fault", [None, "echo", "ids", "finish", "usage", "done"])
def test_strict_protocol(fault):
    async def check():
        async def handler(req):
            return await reply(req, fault=fault, delay=0.001)

        async with endpoint(handler) as url, aiohttp.ClientSession() as http:
            output = await request(http, url, "fixture-not-a-model", [1, 2], 3, "salt", 0)
            assert output.success == (fault is None)
            if fault is None:
                assert output.token_ids == [700, 701, 702]
                row = output.record([1, 2])
                assert row["decode_tokens_per_second"] > 0
                assert row["ttft_seconds"] > 0
            else:
                assert output.error

    asyncio.run(check())

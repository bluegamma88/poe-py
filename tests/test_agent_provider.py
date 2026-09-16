import asyncio
import copy
import json
import ssl

import httpx
import pytest

from poe.agent import Agent
from poe.config import Config
from poe.events import Event
from poe.provider import OpenRouter, ProviderError
from poe.sessions import Session, SessionStore


def chunk(delta=None, finish=None, **extra):
    return {"choices": [{"delta": delta or {}, "finish_reason": finish}], **extra}


def response(*chunks, done=True):
    text = ": heartbeat\n\n" + "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    if done:
        text += "data: [DONE]\n\n"
    return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


class FailingStream(httpx.AsyncByteStream):
    def __init__(self, *chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        raise httpx.ReadError("TLS record failed")

    async def aclose(self):
        pass


async def ignore(event):
    pass


async def test_full_agent_loop_assembles_stream_and_executes_tools(tmp_path):
    (tmp_path / "hello.txt").write_text("old\n")
    (tmp_path / "AGENTS.md").write_text("Keep edits small.")
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(copy.deepcopy(body))
        if len(requests) == 1:
            return response(
                chunk(
                    {
                        "content": "I will update the file.",
                        "reasoning_details": [
                            {"type": "reasoning.text", "index": 0, "text": "Check "},
                        ],
                    }
                ),
                chunk(
                    {
                        "reasoning_details": [
                            {"type": "reasoning.text", "index": 0, "text": "file."}
                        ],
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "edit_file",
                                    "arguments": '{"file_path": "hello.txt",',
                                },
                            }
                        ],
                    }
                ),
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": '"search": "old", "replace": "new"}',
                                },
                            }
                        ]
                    },
                    "tool_calls",
                ),
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            )
        return response(chunk({"content": "Updated."}, "stop"))

    config = Config(api_key="test-key")
    session = Session(cwd=str(tmp_path), model=config.model)
    agent = Agent(
        config,
        session,
        SessionStore(tmp_path / "sessions"),
        OpenRouter(config, transport=httpx.MockTransport(handler)),
    )
    events = []

    async def emit(event):
        events.append(event)

    await agent.run("Update hello.txt", emit)
    assert (tmp_path / "hello.txt").read_text() == "new\n"
    assert any(tool["function"]["name"] == "edit_file" for tool in requests[0]["tools"])
    assert "Keep edits small" in requests[0]["messages"][0]["content"]
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert requests[1]["messages"][-2]["reasoning_details"][0]["text"] == "Check file."
    assert session.messages[-1]["content"] == "Updated."
    assert [e.kind for e in events].count("tool_start") == 1
    assert any(e.kind == "usage" for e in events)
    assert not agent.running


@pytest.mark.parametrize(
    "reply",
    [
        response(chunk({"content": "partial"}), done=False),
        response(chunk({"content": "too long"}, "length")),
        response({"error": {"message": "upstream failed"}}),
        httpx.Response(401, json={"error": {"message": "bad key"}}),
        httpx.Response(200, text="data: invalid-json\n\n"),
    ],
)
async def test_stream_and_http_failures_are_reported_without_retry(reply):
    calls = []

    def handler(request):
        calls.append(request)
        return reply

    provider = OpenRouter(Config(api_key="test"), transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError):
        await provider.complete([], [], ignore)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 524, 529])
async def test_retries_rejected_requests_only(status):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(status, headers={"retry-after": "0"})
        return response(chunk({"content": "hello"}, "stop"))

    provider = OpenRouter(Config(api_key="test"), transport=httpx.MockTransport(handler))
    assert (await provider.complete([], [], ignore))["content"] == "hello"
    assert len(calls) == 2


async def test_retries_connect_failure_before_response(monkeypatch):
    calls = []
    events = []
    monkeypatch.setattr("poe.provider.random.uniform", lambda start, end: 0)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("TLS handshake failed", request=request)
        return response(chunk({"content": "hello"}, "stop"))

    async def record(event):
        events.append(event)

    provider = OpenRouter(Config(api_key="test"), transport=httpx.MockTransport(handler))
    assert (await provider.complete([], [], record))["content"] == "hello"
    assert len(calls) == 2
    assert any("Connection interrupted before response" in event.text for event in events)


async def test_retries_read_failure_before_first_sse_event(monkeypatch):
    calls = []
    monkeypatch.setattr("poe.provider.random.uniform", lambda start, end: 0)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                200,
                stream=FailingStream(b": heartbeat\n\n"),
                headers={"content-type": "text/event-stream"},
            )
        return response(chunk({"content": "hello"}, "stop"))

    provider = OpenRouter(Config(api_key="test"), transport=httpx.MockTransport(handler))
    assert (await provider.complete([], [], ignore))["content"] == "hello"
    assert len(calls) == 2


async def test_network_debug_log_records_safe_metadata(monkeypatch, tmp_path):
    path = tmp_path / "network.jsonl"
    events = []
    monkeypatch.setenv("POE_NETWORK_DEBUG", "1")
    monkeypatch.setenv("POE_NETWORK_DEBUG_PATH", str(path))

    reply = response(chunk({"content": "hello"}, "stop", id="gen-test"))
    reply.headers["CF-Ray"] = "ray-test-SJC"
    reply.headers["X-Generation-Id"] = "gen-test"

    async def record(event):
        events.append(event)

    provider = OpenRouter(
        Config(api_key="secret-api-key", model="test/model"),
        transport=httpx.MockTransport(lambda request: reply),
    )
    message = await provider.complete(
        [{"role": "user", "content": "secret prompt"}],
        [{"function": {"name": "secret_tool"}}],
        record,
    )

    assert message["content"] == "hello"
    text = path.read_text()
    assert "secret-api-key" not in text
    assert "secret prompt" not in text
    assert "secret_tool" not in text
    assert path.stat().st_mode & 0o777 == 0o600
    records = [json.loads(line) for line in text.splitlines()]
    assert [record["event"] for record in records] == [
        "request_start",
        "attempt_start",
        "response_headers",
        "attempt_complete",
    ]
    assert len({record["request_id"] for record in records}) == 1
    assert all(record["model"] == "test/model" for record in records)
    headers = records[2]["response"]["headers"]
    assert headers == {"cf-ray": "ray-test-SJC", "x-generation-id": "gen-test"}
    assert records[3]["stream"]["sse_events"] == 2
    assert records[3]["stream"]["generation_id"] == "gen-test"
    assert any(event.kind == "status" and str(path) in event.text for event in events)


async def test_network_debug_distinguishes_pre_stream_failure_phases(monkeypatch, tmp_path):
    path = tmp_path / "network.jsonl"
    calls = []
    monkeypatch.setenv("POE_NETWORK_DEBUG", "true")
    monkeypatch.setenv("POE_NETWORK_DEBUG_PATH", str(path))
    monkeypatch.setattr("poe.provider.random.uniform", lambda start, end: 0)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("TLS handshake failed", request=request)
        if len(calls) == 2:
            return httpx.Response(
                200,
                stream=FailingStream(b": heartbeat\n\n"),
                headers={"content-type": "text/event-stream"},
            )
        return response(chunk({"content": "hello"}, "stop"))

    provider = OpenRouter(Config(api_key="test"), transport=httpx.MockTransport(handler))
    assert (await provider.complete([], [], ignore))["content"] == "hello"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    failures = [record for record in records if record["event"] == "transport_error"]
    assert [record["phase"] for record in failures] == ["before_headers", "before_sse"]
    assert all(record["will_retry"] for record in failures)
    assert failures[0]["exceptions"][0]["type"] == "ConnectError"
    assert failures[1]["exceptions"][0]["type"] == "ReadError"


async def test_retries_raw_ssl_alert_and_records_transport_trace(monkeypatch, tmp_path):
    path = tmp_path / "network.jsonl"
    calls = []
    monkeypatch.setenv("POE_NETWORK_DEBUG", "1")
    monkeypatch.setenv("POE_NETWORK_DEBUG_PATH", str(path))
    monkeypatch.setattr("poe.provider.random.uniform", lambda start, end: 0)

    async def handler(request):
        calls.append(request)
        trace = request.extensions["trace"]
        await trace("http11.send_request_headers.started", {"request": request})
        if len(calls) == 1:
            error = ssl.SSLError(1, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac")
            await trace("http11.receive_response_headers.failed", {"exception": error})
            raise error
        return response(chunk({"content": "hello"}, "stop"))

    provider = OpenRouter(Config(api_key="secret-api-key"), transport=httpx.MockTransport(handler))
    assert (await provider.complete([], [], ignore))["content"] == "hello"
    assert len(calls) == 2

    text = path.read_text()
    assert "secret-api-key" not in text
    records = [json.loads(line) for line in text.splitlines()]
    failure = next(record for record in records if record["event"] == "transport_error")
    assert failure["phase"] == "before_headers"
    assert failure["will_retry"]
    assert failure["exceptions"][0]["type"] == "SSLError"
    assert failure["last_trace"] == "http11.receive_response_headers.failed"
    traces = [record["operation"] for record in records if record["event"] == "transport_trace"]
    assert traces == [
        "http11.send_request_headers.started",
        "http11.receive_response_headers.failed",
        "http11.send_request_headers.started",
    ]


async def test_network_debug_records_unexpected_attempt_exception(monkeypatch, tmp_path):
    path = tmp_path / "network.jsonl"
    calls = []
    monkeypatch.setenv("POE_NETWORK_DEBUG", "1")
    monkeypatch.setenv("POE_NETWORK_DEBUG_PATH", str(path))

    def handler(request):
        calls.append(request)
        raise RuntimeError("unexpected transport failure")

    provider = OpenRouter(Config(api_key="test"), transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="unexpected transport failure"):
        await provider.complete([], [], ignore)
    assert len(calls) == 1
    records = [json.loads(line) for line in path.read_text().splitlines()]
    failures = [record for record in records if record["event"] == "unexpected_error"]
    assert len(failures) == 1
    assert failures[0]["phase"] == "before_headers"
    assert failures[0]["exceptions"][0]["type"] == "RuntimeError"


async def test_does_not_retry_read_failure_after_sse_event(monkeypatch, tmp_path):
    calls = []
    events = []
    path = tmp_path / "network.jsonl"
    monkeypatch.setenv("POE_NETWORK_DEBUG", "on")
    monkeypatch.setenv("POE_NETWORK_DEBUG_PATH", str(path))
    monkeypatch.setattr("poe.provider.random.uniform", lambda start, end: 0)
    partial = f"data: {json.dumps(chunk({'content': 'partial'}))}\n\n".encode()

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            stream=FailingStream(partial),
            headers={"content-type": "text/event-stream"},
        )

    async def record(event):
        events.append(event)

    provider = OpenRouter(Config(api_key="test"), transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="after response streaming began.*ReadError"):
        await provider.complete([], [], record)
    assert len(calls) == 1
    assert [(event.kind, event.text) for event in events if event.kind == "text"] == [
        ("text", "partial")
    ]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    failure = next(record for record in records if record["event"] == "transport_error")
    assert failure["phase"] == "streaming"
    assert not failure["will_retry"]
    assert failure["stream"]["sse_events"] == 1


class ToolModel:
    def __init__(self, calls):
        self.calls = calls
        self.requests = []

    async def complete(self, messages, tools, emit):
        self.requests.append(copy.deepcopy(messages))
        if len(self.requests) == 1:
            return {"role": "assistant", "content": None, "tool_calls": self.calls}
        await emit(Event("text", "Done"))
        return {"role": "assistant", "content": "Done"}


def call(id, name, arguments):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


async def test_invalid_tool_json_goes_back_to_model(tmp_path):
    model = ToolModel([call("bad", "read_file", "invalid json")])
    agent = Agent(
        Config(),
        Session(cwd=str(tmp_path), model="test"),
        SessionStore(tmp_path / "sessions"),
        model,
    )
    await agent.run("read", ignore)
    assert "Invalid tool arguments" in model.requests[1][-1]["content"]


async def test_cancellation_repairs_all_pending_calls_and_allows_next_turn(tmp_path):
    started = asyncio.Event()
    model = ToolModel(
        [
            call("a", "shell", '{"command": "sleep 30"}'),
            call("b", "write_file", '{"file_path": "should-not-exist", "content": "no"}'),
        ]
    )
    agent = Agent(
        Config(),
        Session(cwd=str(tmp_path), model="test"),
        SessionStore(tmp_path / "sessions"),
        model,
    )

    async def emit(event):
        if event.kind == "tool_start":
            started.set()
            await asyncio.sleep(0.01)

    task = asyncio.create_task(agent.run("work", emit))
    await asyncio.wait_for(started.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not (tmp_path / "should-not-exist").exists()
    results = [m for m in agent.session.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["a", "b"]
    await agent.run("continue", ignore)
    assert agent.session.messages[-1]["content"] == "Done"


async def test_tool_round_limit_stops_execution(tmp_path):
    class Repeating:
        async def complete(self, messages, tools, emit):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    call(str(len(messages)), "shell", '{"command": "echo x >> count"}'),
                ],
            }

    agent = Agent(
        Config(max_tool_rounds=1),
        Session(cwd=str(tmp_path), model="test"),
        SessionStore(tmp_path / "sessions"),
        Repeating(),
    )
    with pytest.raises(RuntimeError, match="round limit"):
        await agent.run("work", ignore)
    assert (tmp_path / "count").read_text() == "x\n"
    assert not agent.running


async def test_reasoning_is_streamed_to_the_interface():
    events = []

    async def record(event):
        events.append((event.kind, event.text))

    def handler(request):
        return response(
            chunk({"reasoning": "First I "}),
            chunk({"reasoning": "check the file."}),
            chunk({"content": "Done."}, "stop"),
        )

    provider = OpenRouter(Config(api_key="test"), transport=httpx.MockTransport(handler))
    message = await provider.complete([], [], record)
    assert [text for kind, text in events if kind == "reasoning"] == [
        "First I ",
        "check the file.",
    ]
    assert message["reasoning"] == "First I check the file."


async def test_reasoning_details_stream_once_without_duplication():
    events = []

    async def record(event):
        events.append((event.kind, event.text))

    def handler(request):
        return response(
            chunk(
                {"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "Think "}]}
            ),
            # Providers that mirror both shapes must not render the thinking twice.
            chunk(
                {
                    "reasoning": "harder.",
                    "reasoning_details": [
                        {"type": "reasoning.text", "index": 0, "text": "harder."}
                    ],
                }
            ),
            chunk({"content": "Done."}, "stop"),
        )

    provider = OpenRouter(Config(api_key="test"), transport=httpx.MockTransport(handler))
    message = await provider.complete([], [], record)
    assert [text for kind, text in events if kind == "reasoning"] == ["Think ", "harder."]
    assert message["reasoning_details"][0]["text"] == "Think harder."

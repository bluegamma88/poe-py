import asyncio
import copy
import json

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
        await provider.complete([], ignore)
    assert len(calls) == 1


async def test_retries_rejected_requests_only():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return response(chunk({"content": "hello"}, "stop"))

    provider = OpenRouter(Config(api_key="test"), transport=httpx.MockTransport(handler))
    assert (await provider.complete([], ignore))["content"] == "hello"
    assert len(calls) == 2


class ToolModel:
    def __init__(self, calls):
        self.calls = calls
        self.requests = []

    async def complete(self, messages, emit):
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
        async def complete(self, messages, emit):
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
    message = await provider.complete([], record)
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
    message = await provider.complete([], record)
    assert [text for kind, text in events if kind == "reasoning"] == ["Think ", "harder."]
    assert message["reasoning_details"][0]["text"] == "Think harder."

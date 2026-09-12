import asyncio

import pytest
from mcp.server.mcpserver import MCPServer
from test_mcp import ApprovalBackend, ApprovalModel
from textual.widgets import Collapsible, Markdown, Static

from poe.agent import Agent
from poe.app import ApprovalScreen, Composer, PoeApp
from poe.config import Config, McpServerConfig
from poe.events import Event
from poe.mcp import McpToolBackend
from poe.sessions import Session, SessionStore
from poe.tooling import ToolRegistry


class FakeModel:
    async def complete(self, messages, tools, emit):
        await emit(Event("text", "Hello **from Poe**."))
        return {"role": "assistant", "content": "Hello **from Poe**."}


def app_for(tmp_path, model=None, prompt=""):
    return PoeApp(
        Agent(
            Config(api_key="test"),
            Session(cwd=str(tmp_path), model="test"),
            SessionStore(tmp_path / "sessions"),
            model or FakeModel(),
        ),
        initial_prompt=prompt,
    )


@pytest.mark.parametrize("size", [(100, 35), (60, 20)])
async def test_submit_stream_new_chat_and_multiline(tmp_path, size):
    app = app_for(tmp_path)
    async with app.run_test(size=size) as pilot:
        composer = app.query_one(Composer)
        composer.text = "hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one(Markdown).source == "Hello **from Poe**."
        assert not app.busy
        assert composer.text == ""
        session_id = app.agent.session.id
        await pilot.press("ctrl+n")
        assert app.agent.session.id != session_id
        assert app.agent.session.messages == []
        await pilot.press("a", "shift+enter", "b")
        assert composer.text == "a\nb"
        await pilot.press("ctrl+j", "c")
        assert composer.text == "a\nb\nc"
        assert app.agent.session.messages == []


async def test_escape_cancels_stream_and_next_turn_works(tmp_path):
    started = asyncio.Event()

    class SlowModel(FakeModel):
        async def complete(self, messages, tools, emit):
            if not started.is_set():
                started.set()
                await emit(Event("text", "Partial"))
                await asyncio.sleep(30)
            return await super().complete(messages, tools, emit)

    app = app_for(tmp_path, SlowModel(), "start")
    async with app.run_test() as pilot:
        await asyncio.wait_for(started.wait(), 3)
        await pilot.press("escape")
        await pilot.press("escape")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.busy
        app.query_one(Composer).text = "again"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert app.agent.session.messages[-1]["content"] == "Hello **from Poe**."


async def test_tool_panels_show_results(tmp_path):
    class FileModel(FakeModel):
        async def complete(self, messages, tools, emit):
            if messages[-1]["role"] != "tool":
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"file_path": "hello.txt", "content": "hello"}',
                            },
                        }
                    ],
                }
            return await super().complete(messages, tools, emit)

    app = app_for(tmp_path, FileModel(), "create hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert (tmp_path / "hello.txt").read_text() == "hello"
        panel = app.query_one(Collapsible)
        assert panel.title.startswith("✓")
        assert panel.collapsed
        await pilot.click("CollapsibleTitle")
        assert not panel.collapsed


async def test_quit_while_model_running_saves_session(tmp_path):
    started = asyncio.Event()

    class SlowModel:
        async def complete(self, messages, tools, emit):
            started.set()
            await asyncio.sleep(30)

    app = app_for(tmp_path, SlowModel(), "start")
    async with app.run_test() as pilot:
        await asyncio.wait_for(started.wait(), 3)
        await pilot.press("ctrl+q")
    assert not app.agent.running
    assert app.agent.store.load("latest").messages[-1]["content"] == "start"


async def test_thinking_panel_streams_and_folds_away_after_the_answer(tmp_path):
    class ThinkingModel(FakeModel):
        async def complete(self, messages, tools, emit):
            await emit(Event("reasoning", "Weighing "))
            await emit(Event("reasoning", "the options."))
            return await super().complete(messages, tools, emit)

    app = app_for(tmp_path, ThinkingModel(), "start")
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        panel = app.query_one(".thinking", Collapsible)
        assert str(app.query_one(".thought", Static).content) == "Weighing the options."
        assert panel.collapsed  # The answer arrived, so the thinking folds away.
        await pilot.click("CollapsibleTitle")
        assert not panel.collapsed


async def test_thinking_stays_open_when_a_round_produces_no_answer(tmp_path):
    class SilentModel:
        async def complete(self, messages, tools, emit):
            await emit(Event("reasoning", "No conclusion reached."))
            return {"role": "assistant", "content": None}

    app = app_for(tmp_path, SilentModel(), "start")
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.query_one(".thinking", Collapsible).collapsed
        assert str(app.query_one(".thought", Static).content) == "No conclusion reached."


async def test_resumed_session_replays_saved_thinking(tmp_path):
    session = Session(cwd=str(tmp_path), model="test")
    session.messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "Hello.",
            "reasoning_details": [{"type": "reasoning.text", "text": "Recalled context."}],
        },
    ]
    app = PoeApp(
        Agent(Config(api_key="test"), session, SessionStore(tmp_path / "sessions"), FakeModel())
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(".thinking", Collapsible)
        assert panel.collapsed
        assert str(app.query_one(".thought", Static).content) == "Recalled context."


async def test_mcp_approval_modal_allows_one_call(tmp_path):
    backend = ApprovalBackend()
    agent = Agent(
        Config(),
        Session(cwd=str(tmp_path), model="test"),
        SessionStore(tmp_path / "sessions"),
        ApprovalModel(),
        tools=ToolRegistry([backend]),
    )
    app = PoeApp(agent, initial_prompt="send hello")
    async with app.run_test() as pilot:
        async with asyncio.timeout(3):
            while not isinstance(app.screen, ApprovalScreen):
                await pilot.pause()
        await pilot.click("#allow")
        await app.workers.wait_for_complete()
        assert backend.calls == [("send", {"text": "hello"})]


async def test_app_owns_mcp_client_across_worker_and_shutdown(tmp_path):
    calls = []
    server = MCPServer("app-test")

    @server.tool()
    def send(text: str) -> str:
        """Record text."""
        calls.append(text)
        return "sent"

    backend = McpToolBackend(
        McpServerConfig(
            name="remote",
            transport="stdio",
            command="unused",
            approval="never",
        ),
        tmp_path,
        target=server,
    )
    agent = Agent(
        Config(),
        Session(cwd=str(tmp_path), model="test"),
        SessionStore(tmp_path / "sessions"),
        ApprovalModel(),
        tools=ToolRegistry([backend]),
    )
    app = PoeApp(agent, initial_prompt="send hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert calls == ["hello"]
    assert backend._client is None

import asyncio

import pytest
from textual.widgets import Collapsible, Markdown

from poe.agent import Agent
from poe.app import Composer, PoeApp
from poe.config import Config
from poe.events import Event
from poe.sessions import Session, SessionStore


class FakeModel:
    async def complete(self, messages, emit):
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
        async def complete(self, messages, emit):
            if not started.is_set():
                started.set()
                await emit(Event("text", "Partial"))
                await asyncio.sleep(30)
            return await super().complete(messages, emit)

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
        async def complete(self, messages, emit):
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
            return await super().complete(messages, emit)

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
        async def complete(self, messages, emit):
            started.set()
            await asyncio.sleep(30)

    app = app_for(tmp_path, SlowModel(), "start")
    async with app.run_test() as pilot:
        await asyncio.wait_for(started.wait(), 3)
        await pilot.press("ctrl+q")
    assert not app.agent.running
    assert app.agent.store.load("latest").messages[-1]["content"] == "start"

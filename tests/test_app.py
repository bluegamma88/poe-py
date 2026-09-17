import asyncio
import io
import re
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer
from rich.console import Console
from test_mcp import ApprovalBackend, ApprovalModel
from textual.widgets import Collapsible, Markdown, Static
from textual.widgets.markdown import MarkdownBlock, MarkdownFence

from poe.agent import Agent
from poe.app import (
    CURSOR_THEME,
    SPINNER_FRAMES,
    ApprovalScreen,
    Composer,
    ContextUsageScreen,
    LatexBlock,
    LatexMarkdown,
    PoeApp,
    Starfield,
    display_path,
    estimate_context_breakdown,
    format_arguments,
    render_latex,
)
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


async def test_starfield_shows_only_while_conversation_is_empty(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        starfield = app.query_one(Starfield)
        assert starfield.display
        assert starfield.stars
        app.query_one(Composer).text = "hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert not starfield.display
        await pilot.press("ctrl+n")
        assert starfield.display


async def test_starfield_hidden_for_resumed_session(tmp_path):
    session = Session(cwd=str(tmp_path), model="test")
    session.messages = [{"role": "user", "content": "hi"}]
    app = PoeApp(
        Agent(Config(api_key="test"), session, SessionStore(tmp_path / "sessions"), FakeModel())
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.query_one(Starfield).display


async def test_shooting_star_crosses_the_sky_and_burns_out(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        starfield = app.query_one(Starfield)
        starfield.launch_meteor()
        seen_trail = False
        for _ in range(200):
            starfield.tick()
            cells = starfield.meteor_cells()
            assert len({y for _, y in cells}) <= 1
            seen_trail = seen_trail or "".join(glyph for glyph, _ in cells.values()) == "✦━━──·"
            if starfield.meteor_head is None:
                break
        assert seen_trail
        assert starfield.meteor_head is None


def test_render_latex_uses_unicode_and_preserves_unsupported_commands():
    assert render_latex(r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}") == ("(-b±√(b²-4ac))/(2a)")
    assert render_latex(r"\unknown{x}") == r"\unknown{x}"
    assert render_latex(r"\frac{") == r"\frac{"


async def test_latex_renders_inline_display_and_bracket_delimiters(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test() as pilot:
        body = await app.add_message(
            "assistant",
            "Inline $x^2 + \\alpha$ or \\(y_1\\).\n\n"
            "$$\\frac{1}{2}$$\n\n"
            "\\[\\sqrt{4} = 2\\]\n\n"
            "`$leave_code_alone$`",
        )
        await pilot.pause()

        assert isinstance(body, LatexMarkdown)
        blocks = list(body.query(MarkdownBlock))
        assert blocks[0].content.plain == "Inline x²+α or y₁."
        assert isinstance(blocks[1], LatexBlock)
        assert blocks[1].content.plain == "½"
        assert isinstance(blocks[2], LatexBlock)
        assert blocks[2].content.plain == "√4=2"
        assert blocks[3].content.plain == "$leave_code_alone$"


async def test_latex_renders_when_delimiters_arrive_across_stream_chunks(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test() as pilot:
        await app.handle_event(Event("text", "Result: $x"))
        await app.handle_event(Event("text", "^2$.\n\n$$\\frac"))
        await app.handle_event(Event("text", "{3}{4}$$"))
        await app.handle_event(Event("assistant_done"))
        await pilot.pause()

        body = app.query_one(LatexMarkdown)
        blocks = list(body.query(MarkdownBlock))
        assert body.source == "Result: $x^2$.\n\n$$\\frac{3}{4}$$"
        assert blocks[0].content.plain == "Result: x²."
        assert isinstance(blocks[1], LatexBlock)
        assert blocks[1].content.plain == "¾"


async def test_latex_renders_math_only_latex_fences(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test() as pilot:
        body = await app.add_message(
            "assistant",
            "```latex\n"
            "\\[\n"
            "J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta}\n"
            "\\left[\\sum_{t=0}^{T-1} \\gamma^t r_t\\right]\n"
            "\\]\n\n"
            "\\[\n"
            "\\nabla_\\theta J(\\theta) = \\mathbb{E}[G_t]\n"
            "\\]\n"
            "```",
        )
        await pilot.pause()

        blocks = list(body.query(MarkdownBlock))
        assert len(blocks) == 1
        assert isinstance(blocks[0], LatexBlock)
        assert blocks[0].content.plain == ("J(θ)=𝔼[τ∼π[θ]][(∑[t=0])ᵀ⁻¹γᵗr[t]]\n\n∇[θ]J(θ)=𝔼[G[t]]")


async def test_latex_source_code_fence_remains_syntax_highlighted(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test() as pilot:
        body = await app.add_message(
            "assistant",
            "```latex\n\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}\n```",
        )
        await pilot.pause()

        assert isinstance(body.query_one(MarkdownBlock), MarkdownFence)


async def test_composer_grows_and_shrinks_with_multiline_prompt(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test(size=(100, 35)) as pilot:
        composer = app.query_one(Composer)
        box = app.query_one("#composer-box")
        assert box.outer_size.height == Composer.MIN_HEIGHT + 2

        composer.text = "one\ntwo\nthree"
        await pilot.pause()
        assert box.outer_size.height == 5

        composer.text = "\n".join(str(line) for line in range(20))
        await pilot.pause()
        assert box.outer_size.height == Composer.MAX_HEIGHT + 2

        composer.clear()
        await pilot.pause()
        assert box.outer_size.height == Composer.MIN_HEIGHT + 2


async def test_composer_grows_with_soft_wrapped_prompt(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test(size=(20, 35)) as pilot:
        composer = app.query_one(Composer)
        composer.text = "word " * 20
        await pilot.pause()

        assert composer.document.line_count == 1
        assert composer.wrapped_document.height > 1
        assert app.query_one("#composer-box").outer_size.height == Composer.MAX_HEIGHT + 2


async def test_header_and_status_keep_layout_stable(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test(size=(60, 20)) as pilot:
        assert str(app.query_one("#header-path", Static).content) == display_path(str(tmp_path))
        assert str(app.query_one("#header-model", Static).content) == app.agent.config.model

        status = app.query_one("#status", Static)
        dock = app.query_one("#composer-dock")
        idle_height = dock.outer_size.height
        assert not status.visible
        app.set_status("Thinking…")
        await pilot.pause()
        assert status.visible
        assert str(status.content) == "Thinking…"
        assert dock.outer_size.height == idle_height
        app.set_status("Ready")
        await pilot.pause()
        assert not status.visible
        assert dock.outer_size.height == idle_height

        await app.handle_event(Event("usage", data={"prompt_tokens": 10, "completion_tokens": 2}))
        await pilot.pause()
        assert app.query_one("#usage", Static).visible
        assert dock.outer_size.height == idle_height


async def test_status_summarizes_tokens_cache_and_cost(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test() as pilot:
        await app.handle_event(
            Event(
                "usage",
                data={
                    "prompt_tokens": 1_200,
                    "completion_tokens": 30,
                    "total_tokens": 1_230,
                    "prompt_tokens_details": {
                        "cached_tokens": 1_000,
                        "cache_write_tokens": 100,
                    },
                    "cost": 0.0012,
                },
            )
        )
        await app.handle_event(
            Event(
                "usage",
                data={
                    "prompt_tokens": 300,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 250},
                    "cost": "0.0003",
                },
            )
        )
        app.set_status("Ready")
        await pilot.pause()

        status = app.query_one("#status", Static)
        usage = app.query_one("#usage", Static)
        assert not status.visible
        assert usage.visible
        assert str(usage.visual) == (
            "context 320 tokens · usage 1,500 in / 50 out · "
            "cache 1,250 read / 100 write · cost $0.0015"
        )

        await app.action_new_chat()
        await pilot.pause()
        assert not usage.visible
        assert str(usage.content) == ""


async def test_status_omits_usage_details_not_reported_by_provider(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test() as pilot:
        await app.handle_event(Event("usage", data={"prompt_tokens": 10, "completion_tokens": 2}))
        app.set_status("Ready")
        await pilot.pause()

        usage = app.query_one("#usage", Static)
        assert str(usage.visual) == "context 12 tokens · usage 10 in / 2 out"


async def test_clicking_context_opens_usage_breakdown(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test(size=(100, 35)) as pilot:
        await app.handle_event(
            Event(
                "usage",
                data={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            )
        )
        app.set_status("Ready")
        await pilot.pause()

        assert await pilot.click("#usage", offset=(10, 0))
        await pilot.pause()
        assert isinstance(app.screen, ContextUsageScreen)
        assert "Reported prompt 100 · latest response 20 · cached prompt 40" == str(
            app.screen.query_one("#context-summary", Static).content
        )
        assert "Total context" in render_text(
            app.screen.query_one("#context-table Static", Static).content
        )

        await pilot.click("#context-dialog", offset=(1, 0))
        await pilot.pause()
        assert isinstance(app.screen, ContextUsageScreen), "clicks inside the dialog keep it open"

        await pilot.click(offset=(1, 1))
        await pilot.pause()
        assert not isinstance(app.screen, ContextUsageScreen), "backdrop click dismisses"


async def test_context_modal_closes_from_close_button(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test(size=(100, 35)) as pilot:
        await app.handle_event(
            Event("usage", data={"prompt_tokens": 100, "completion_tokens": 20})
        )
        await pilot.pause()
        app.action_show_context()
        await pilot.pause()
        assert isinstance(app.screen, ContextUsageScreen)

        await pilot.click("#context-close")
        await pilot.pause()
        assert not isinstance(app.screen, ContextUsageScreen)


def test_display_path_abbreviates_home_directory():
    home = Path.home()
    assert display_path(str(home)) == "~"
    assert display_path(str(home / "code" / "poe")) == str(Path("~/code/poe"))
    assert display_path("/etc/hosts") == "/etc/hosts"
    assert display_path(f"{home}-backup/notes") == f"{home}-backup/notes"


async def test_header_shows_home_relative_path(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    app = app_for(tmp_path / "work")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert str(app.query_one("#header-path", Static).content) == str(Path("~/work"))


def test_context_breakdown_categories_sum_to_reported_tokens():
    breakdown = estimate_context_breakdown(
        [
            {"role": "system", "content": "Follow the project instructions."},
            {"role": "user", "content": "Update the file."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "print('hello')"},
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from the workspace.",
                    "parameters": {"type": "object"},
                },
            }
        ],
        prompt_tokens=1_000,
        completion_tokens=50,
        cached_tokens=400,
    )

    categories = dict(breakdown.categories)
    assert set(categories) == {
        "System prompt",
        "User messages",
        "Assistant messages",
        "Tool calls",
        "Tool results",
        "Tool definitions",
        "Message overhead",
    }
    assert sum(categories.values()) == 1_000
    assert breakdown.total_tokens == 1_050
    assert breakdown.cached_tokens == 400


async def test_composer_matches_app_background(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test():
        composer = app.query_one(Composer)
        dock = app.query_one("#composer-dock")

        assert dock.styles.background == composer.styles.background == app.screen.styles.background
        assert dock.styles.border_top[0] == ""


async def test_cursor_theme_is_fixed_and_palette_is_disabled(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test():
        assert app.theme == CURSOR_THEME.name
        assert app.current_theme is CURSOR_THEME
        assert CURSOR_THEME.background == "#14120b"
        assert CURSOR_THEME.foreground == "#edecec"
        assert CURSOR_THEME.accent == "#9fbbe0"
        assert CURSOR_THEME.warning == "#f54e00"
        assert not app.use_command_palette
        assert "ctrl+p" not in app.active_bindings
        assert "ctrl+q" not in app.active_bindings
        assert app.active_bindings["ctrl+d"].binding.action == "quit"


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
        assert re.fullmatch(r"✓ Write hello\.txt · \d+\.\ds", panel.title)
        assert panel.has_class("success")
        assert panel.collapsed
        await pilot.click("CollapsibleTitle")
        assert not panel.collapsed


async def test_running_and_failed_tools_use_compact_activity_states(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test() as pilot:
        await app.show_tool(
            {
                "id": "failed-read",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"file_path": "missing.txt"}',
                },
            }
        )
        await pilot.pause()
        panel = app.tool_panels["failed-read"].panel
        assert panel.title[0] in SPINNER_FRAMES
        assert panel.has_class("running")

        app.finish_tool("failed-read", "Error: file not found", False)
        assert re.fullmatch(r"⚠ Read missing\.txt · \d+\.\ds", panel.title)
        assert panel.has_class("failure")
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
        await pilot.press("ctrl+d")
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
        assert re.fullmatch(r"✓ Thinking · \d+\.\ds", panel.title)
        assert panel.has_class("success")
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


async def test_failed_reasoning_uses_warning_state(tmp_path):
    class FailingModel:
        async def complete(self, messages, tools, emit):
            await emit(Event("reasoning", "Trying an approach."))
            raise RuntimeError("model failed")

    app = app_for(tmp_path, FailingModel(), "start")
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        panel = app.query_one(".thinking", Collapsible)
        assert re.fullmatch(r"⚠ Thinking · \d+\.\ds", panel.title)
        assert panel.has_class("failure")
        assert not panel.collapsed


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


async def test_mcp_approval_modal_denies_with_keyboard(tmp_path):
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
        assert app.screen.query_one("#approval-dialog").border_title == "Tool approval"
        await pilot.press("d")
        await app.workers.wait_for_complete()
        assert backend.calls == []


async def test_mcp_approval_modal_escape_denies_without_cancelling_turn(tmp_path):
    backend = ApprovalBackend()
    model = ApprovalModel()
    agent = Agent(
        Config(),
        Session(cwd=str(tmp_path), model="test"),
        SessionStore(tmp_path / "sessions"),
        model,
        tools=ToolRegistry([backend]),
    )
    app = PoeApp(agent, initial_prompt="send hello")
    async with app.run_test() as pilot:
        async with asyncio.timeout(3):
            while not isinstance(app.screen, ApprovalScreen):
                await pilot.pause()
        await pilot.press("escape")
        await app.workers.wait_for_complete()
        assert backend.calls == []
        assert app.turn_worker is not None and not app.turn_worker.is_cancelled
        # The denial is reported to the model, which finishes the turn normally.
        assert len(model.requests) == 2
        transcript = [str(widget.content) for widget in app.query(".notice")]
        assert not any("cancelled" in line.lower() for line in transcript)


def render_text(renderable, width=60):
    console = Console(width=width, file=io.StringIO())
    console.print(renderable)
    return console.file.getvalue()


def test_format_arguments_aligns_values_under_one_column():
    rendered = render_text(format_arguments({"text": "hello", "count": 2, "path": "a/b"}))
    lines = rendered.splitlines()
    values = ["hello", "2", "a/b"]
    starts = {line.index(value) for line, value in zip(lines, values, strict=True)}
    assert len(starts) == 1
    assert "No arguments" in render_text(format_arguments({}))


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

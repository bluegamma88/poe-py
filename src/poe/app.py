"""Textual chat interface; model work runs in a cancellable async worker."""

from __future__ import annotations

import asyncio
import contextlib
import json

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, Footer, Header, Markdown, Static, TextArea
from textual.widgets.markdown import MarkdownStream
from textual.worker import Worker, WorkerCancelled, WorkerFailed, WorkerState

from poe.agent import Agent
from poe.events import Event
from poe.sessions import Session
from poe.tools import clip

HELP = (
    "Enter sends · Shift+Enter adds a newline · Escape cancels · Ctrl+N starts a new chat · "
    "Ctrl+Q quits\nCommands: /new, /help, /quit. Click a tool panel to expand its output."
)


class Composer(TextArea):
    BINDINGS = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("shift+enter", "newline", "Newline", key_display="Shift+Enter", priority=True),
        Binding("ctrl+j", "newline", "Newline", show=False, priority=True),
    ]

    class Submitted(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

    def action_submit(self) -> None:
        if self.text.strip():
            self.post_message(self.Submitted(self.text.strip()))

    def action_newline(self) -> None:
        self.insert("\n")


class PoeApp(App):
    TITLE = "Poe"
    CSS = """
    Screen { background: #111820; }
    Header { background: #19252e; color: #c4e5df; }
    #transcript { width: 100%; padding: 1 0; scrollbar-size: 1 1; }
    #transcript > .message, #transcript > .notice,
    #transcript > .error, #transcript > Collapsible { margin-left: 3; margin-right: 3; }
    .message { height: auto; margin-bottom: 1; }
    .label { color: #7ccfbe; text-style: bold; margin-bottom: 1; }
    .user .label { color: #c3b5f3; }
    .message Markdown { padding: 0; margin: 0; background: transparent; }
    .message Static { height: auto; }
    .notice { color: #96a7b5; margin-bottom: 1; }
    .error { color: #f1a1a1; margin-bottom: 1; }
    Collapsible { background: #18222c; margin-bottom: 1; padding: 0 1; }
    .arguments { color: #96a7b5; margin-bottom: 1; }
    #status { height: 1; padding: 0 3; color: #7ccfbe; }
    #composer { height: 5; max-height: 10; margin: 1 2 0 2;
                border: round #435662; background: #18222c; }
    #composer:focus { border: round #7ccfbe; }
    Footer { background: #111820; }
    """
    BINDINGS = [
        Binding("escape", "cancel_turn", "Cancel", priority=True),
        Binding("ctrl+n", "new_chat", "New chat", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self, agent: Agent, *, initial_prompt: str = ""):
        super().__init__()
        self.agent = agent
        self.initial_prompt = initial_prompt
        self.busy = False
        self.turn_worker: Worker | None = None
        self.markdown_stream: MarkdownStream | None = None
        self.tool_panels: dict[str, tuple[Collapsible, Static, str]] = {}
        self.input_tokens = 0
        self.output_tokens = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="transcript")
        yield Static("Ready", id="status", markup=False)
        yield Composer(
            id="composer", placeholder="Ask Poe to explore, change, or test this project…"
        )
        yield Footer()

    async def on_mount(self) -> None:
        self.theme = "textual-dark"
        self.sub_title = self.agent.session.cwd
        self.query_one("#transcript", VerticalScroll).anchor()
        await self.notice(f"{self.agent.config.model} · {self.agent.session.cwd}\n{HELP}")
        for message in self.agent.session.messages:
            role = message["role"]
            if role in {"user", "assistant"} and message.get("content"):
                await self.add_message(role, message["content"])
            for call in message.get("tool_calls", []):
                await self.show_tool(call)
            if role == "tool":
                self.finish_tool(message["tool_call_id"], message.get("content", ""), None)
        self.set_status("Ready")
        self.query_one(Composer).focus()
        if self.initial_prompt:
            self.query_one(Composer).text = self.initial_prompt
            self.query_one(Composer).action_submit()

    def set_status(self, text: str) -> None:
        usage = (
            f" · tokens {self.input_tokens:,} in / {self.output_tokens:,} out"
            if (self.input_tokens or self.output_tokens)
            else ""
        )
        self.query_one("#status", Static).update(
            f"{text} · {self.agent.config.model} · {self.agent.session.id[:8]}{usage}"
        )

    async def notice(self, text: str, *, error: bool = False) -> None:
        await self.query_one("#transcript", VerticalScroll).mount(
            Static(text, markup=False, classes="error" if error else "notice")
        )

    async def add_message(self, role: str, text: str) -> Markdown | Static:
        body = Markdown(text) if role == "assistant" else Static(text, markup=False)
        await self.query_one("#transcript", VerticalScroll).mount(
            Vertical(
                Static("Poe" if role == "assistant" else "You", classes="label"),
                body,
                classes=f"message {role}",
            )
        )
        return body

    async def show_tool(self, call: dict) -> None:
        function = call["function"]
        raw = function.get("arguments", "")
        try:
            args = json.loads(raw)
            formatted = json.dumps(args, indent=2, ensure_ascii=False)
            target = args.get("command") or args.get("file_path") or args.get("dir_path", ".")
        except (ValueError, AttributeError, TypeError):
            formatted, target = raw, ""
        title = escape(f"{function['name']}  {clip(str(target).replace(chr(10), ' '), 90)}")
        output = Static("Running…", markup=False)
        panel = Collapsible(
            Static(clip(formatted), markup=False, classes="arguments"),
            output,
            title=f"● {title}",
            collapsed=True,
        )
        self.tool_panels[call["id"]] = panel, output, title
        await self.query_one("#transcript", VerticalScroll).mount(panel)

    def finish_tool(self, call_id: str, content: str, success: bool | None) -> None:
        if call_id in self.tool_panels:
            panel, output, title = self.tool_panels[call_id]
            symbol = "✓" if success is True else "!" if success is False else "·"
            panel.title = f"{symbol} {title}"
            output.update(content)
            if success is False:
                panel.collapsed = False

    async def finish_stream(self) -> None:
        if self.markdown_stream is not None:
            stream, self.markdown_stream = self.markdown_stream, None
            await stream.stop()

    async def handle_event(self, event: Event) -> None:
        if event.kind == "text":
            if self.markdown_stream is None:
                body = await self.add_message("assistant", "")
                assert isinstance(body, Markdown)
                self.markdown_stream = Markdown.get_stream(body)
            await self.markdown_stream.write(event.text)
            self.set_status("Responding…")
        elif event.kind == "assistant_done":
            await self.finish_stream()
        elif event.kind == "tool_start":
            self.set_status(f"Running {event.text}…")
            await self.show_tool(event.data)
        elif event.kind == "tool_result":
            self.finish_tool(event.data["id"], event.text, event.data["success"])
        elif event.kind == "status":
            self.set_status(event.text)
        elif event.kind == "usage":
            self.input_tokens += event.data.get("prompt_tokens", 0) or 0
            self.output_tokens += event.data.get("completion_tokens", 0) or 0
        elif event.kind == "done":
            self.set_status("Ready")

    async def on_composer_submitted(self, event: Composer.Submitted) -> None:
        if self.busy:
            self.notify("A turn is running. Press Escape to cancel it.")
            return
        text = event.text
        self.query_one(Composer).clear()
        if text == "/quit":
            await self.action_quit()
        elif text == "/new":
            await self.action_new_chat()
        elif text == "/help":
            await self.notice(HELP)
        else:
            self.busy = True
            await self.add_message("user", text)
            self.turn_worker = self.run_turn(text)

    @work(group="turn", exclusive=True, exit_on_error=False)
    async def run_turn(self, prompt: str) -> None:
        try:
            await self.agent.run(prompt, self.handle_event)
        except asyncio.CancelledError:
            await self.notice("Turn cancelled. Completed changes are saved.")
            # Reconcile any running panel with the repaired, resumable transcript.
            for message in self.agent.session.messages:
                if message["role"] == "tool" and message["content"].startswith("Interrupted."):
                    self.finish_tool(message["tool_call_id"], message["content"], False)
            raise
        except Exception as exc:
            await self.notice(str(exc), error=True)
        finally:
            await self.finish_stream()
            self.busy = False
            self.set_status("Ready")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker is self.turn_worker and event.state in {
            WorkerState.SUCCESS,
            WorkerState.ERROR,
            WorkerState.CANCELLED,
        }:
            self.busy = False

    def action_cancel_turn(self) -> None:
        if self.turn_worker is not None and self.busy and not self.turn_worker.is_cancelled:
            self.set_status("Cancelling…")
            self.turn_worker.cancel()

    async def action_new_chat(self) -> None:
        if self.busy:
            self.notify("Press Escape to cancel the current turn first.")
            return
        self.agent.session = Session(cwd=self.agent.session.cwd, model=self.agent.config.model)
        self.tool_panels.clear()
        self.input_tokens = self.output_tokens = 0
        await self.query_one("#transcript", VerticalScroll).remove_children()
        await self.notice("New conversation. " + HELP)
        self.set_status("Ready")
        self.query_one(Composer).focus()

    async def action_quit(self) -> None:
        if self.turn_worker is not None and self.busy:
            if not self.turn_worker.is_cancelled:
                self.turn_worker.cancel()
            with contextlib.suppress(WorkerCancelled, WorkerFailed):
                await self.turn_worker.wait()
        self.exit()

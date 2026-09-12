"""Textual chat interface; model work runs in a cancellable async worker."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from time import monotonic

from rich.console import RenderableType
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, Collapsible, Markdown, Static, TextArea
from textual.widgets.markdown import MarkdownStream
from textual.worker import Worker, WorkerCancelled, WorkerFailed, WorkerState

from poe.agent import Agent
from poe.events import Event
from poe.provider import reasoning_text
from poe.sessions import Session
from poe.tooling import ToolRoute
from poe.tools import clip

HELP = (
    "Enter sends · Shift+Enter adds a newline · Escape cancels · Ctrl+N starts a new chat · "
    "Ctrl+D quits\nCommands: /new, /help, /quit. "
    "Click a tool or thinking panel to expand its output."
)

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
TOOL_ACTIONS = {
    "read_file": "Read",
    "list_dir": "List",
    "edit_file": "Edit",
    "write_file": "Write",
    "shell": "Run",
}

CURSOR_THEME = Theme(
    name="cursor",
    primary="#9fbbe0",
    secondary="#9fbbe0",
    accent="#9fbbe0",
    warning="#f54e00",
    error="#cf2d56",
    success="#1f8a65",
    foreground="#edecec",
    background="#14120b",
    surface="#1b1913",
    panel="#201e18",
    dark=True,
    variables={
        "border": "#edecec 10%",
        "border-blurred": "#edecec 5%",
        "input-selection-background": "#9fbbe0 30%",
        "block-cursor-background": "#edecec",
        "block-cursor-foreground": "#14120b",
        "button-color-foreground": "#14120b",
    },
)


@dataclass
class ToolActivity:
    panel: Collapsible
    output: Static
    label: str
    started_at: float | None
    running: bool = True


def activity_label(name: str, target: object = "") -> str:
    """Turn a tool call into a short, readable activity label."""
    action = TOOL_ACTIONS.get(name, name if "/" in name else name.replace("_", " ").capitalize())
    return f"{action} {target}".strip()


ARGUMENT_LIMIT = 1_000


def format_arguments(args: dict) -> RenderableType:
    """Lay tool arguments out as a key/value grid so wrapped values stay aligned."""
    if not args:
        return Text("No arguments", style="italic #edecec 60%")
    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(style="bold #9fbbe0", overflow="fold")
    grid.add_column(style="#edecec", ratio=1, overflow="fold")
    for key, value in args.items():
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        grid.add_row(key, clip(str(rendered), ARGUMENT_LIMIT))
    return grid


class ApprovalScreen(ModalScreen[bool]):
    """Confirmation gate for MCP tools configured with approval='always'."""

    DEFAULT_CSS = """
    ApprovalScreen { align: center middle; background: $background 70%; }
    #approval-dialog { width: 80%; max-width: 100; height: auto; padding: 1 2;
                       border: round #edecec 20%; background: #1b1913;
                       border-title-color: #f54e00; border-title-align: left; }
    #approval-tool { color: #edecec; text-style: bold; }
    #approval-server { color: #edecec 60%; margin-bottom: 1; }
    #approval-arguments { height: auto; max-height: 14; padding: 0 1; scrollbar-size: 1 1;
                          border-left: solid #edecec 20%; }
    #approval-arguments Static { height: auto; }
    #approval-footer { height: 1; margin-top: 1; }
    #approval-hint { width: 1fr; color: #edecec 40%; }
    #approval-buttons { width: auto; height: 1; }
    #approval-buttons Button { height: 1; min-width: 14; border: none; padding: 0 2;
                               margin-left: 1; background: transparent; text-style: none; }
    #approval-buttons Button:hover { background: #edecec 10%; }
    #approval-buttons Button:focus { background: #edecec 15%; text-style: bold; }
    #approval-buttons #deny { color: #cf2d56; }
    #approval-buttons #allow { color: #1f8a65; }
    """
    BINDINGS = [
        Binding("escape", "deny", "Deny", show=False),
        Binding("d,n", "deny", "Deny", show=False),
        Binding("a,y", "allow", "Allow once", show=False),
    ]

    def __init__(self, route: ToolRoute, args: dict):
        super().__init__()
        self.route = route
        self.args = args

    def compose(self) -> ComposeResult:
        server, _, tool = self.route.display_name.rpartition("/")
        source = server or self.route.backend.source_id
        yield Vertical(
            Static(tool or self.route.display_name, id="approval-tool", markup=False),
            Static(
                f"MCP server · {source}" if self.route.backend.namespaced else source,
                id="approval-server",
                markup=False,
            ),
            VerticalScroll(Static(format_arguments(self.args)), id="approval-arguments"),
            Horizontal(
                Static("a allow · d deny · esc deny", id="approval-hint", markup=False),
                Horizontal(
                    Button("Deny", id="deny"),
                    Button("Allow once", id="allow"),
                    id="approval-buttons",
                ),
                id="approval-footer",
            ),
            id="approval-dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#approval-dialog", Vertical).border_title = "Tool approval"
        self.query_one("#deny", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class Composer(TextArea):
    MIN_HEIGHT = 3
    MAX_HEIGHT = 10

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

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Grow with explicit prompt lines, then shrink again as lines are removed."""
        content_height = self.document.line_count + 2  # Account for the top and bottom border.
        self.styles.height = max(self.MIN_HEIGHT, min(self.MAX_HEIGHT, content_height))


class PoeApp(App, inherit_bindings=False):
    TITLE = "Poe"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { background: #14120b; color: #edecec; }
    #app-header { height: 1; width: 100%; background: #14120b; }
    #header-path { width: 1fr; padding: 0 1; color: #edecec 60%; text-align: left;
                   text-overflow: ellipsis; overflow: hidden; }
    #header-model { width: auto; max-width: 40%; padding: 0 1; color: #edecec;
                    text-align: right;
                    text-overflow: ellipsis; overflow: hidden; }
    #transcript { width: 100%; padding: 1 0; scrollbar-size: 1 1; }
    #transcript > .message, #transcript > .notice,
    #transcript > .error, #transcript > Collapsible { margin: 0 3 1 3; }
    .message { height: auto; }
    .assistant { padding: 0 1 1 1; }
    .user { padding: 1 2; background: #1b1913; border-left: solid #9fbbe0; }
    .label { color: #edecec; text-style: bold; margin-bottom: 1; }
    .user .label { color: #9fbbe0; }
    .message Markdown { padding: 0; margin: 0; background: transparent; }
    .message Static { height: auto; }
    .notice { color: #edecec 60%; }
    .error { color: #cf2d56; }
    Collapsible { background: transparent; border-top: none; padding: 0 1; }
    .activity CollapsibleTitle { color: #edecec 60%; }
    .activity.running CollapsibleTitle { color: #9fbbe0; }
    .activity.success CollapsibleTitle { color: #1f8a65; }
    .activity.failure CollapsibleTitle { color: #cf2d56; }
    .arguments { color: #edecec 60%; margin-bottom: 1; }
    .thought { color: #edecec 60%; text-style: italic; }
    #composer-dock { height: auto; padding: 0 2; background: #14120b;
                     border-top: solid #edecec 10%; }
    #status { display: none; height: 1; padding: 0 1; color: #9fbbe0;
              background: transparent; }
    #composer { height: 3; max-height: 10; margin: 0;
                border: round #edecec 10%; background: #1b1913; color: #edecec; }
    #composer:focus { border: round #9fbbe0; }
    """
    BINDINGS = [
        Binding("escape", "cancel_turn", "Cancel", priority=True),
        Binding("ctrl+n", "new_chat", "New chat", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
    ]

    def __init__(self, agent: Agent, *, initial_prompt: str = ""):
        super().__init__()
        self.register_theme(CURSOR_THEME)
        self.theme = CURSOR_THEME.name
        self.agent = agent
        self.initial_prompt = initial_prompt
        self.busy = False
        self.turn_worker: Worker | None = None
        self.markdown_stream: MarkdownStream | None = None
        self.tool_panels: dict[str, ToolActivity] = {}
        self.running_tool_ids: set[str] = set()
        self.thinking: Static | None = None
        self.thinking_panel: Collapsible | None = None
        self.thinking_started_at: float | None = None
        self.thinking_parts: list[str] = []
        self.thinking_dirty = False
        self.spinner_index = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static(self.agent.session.cwd, id="header-path", markup=False),
            Static(self.agent.config.model, id="header-model", markup=False),
            id="app-header",
        )
        yield VerticalScroll(id="transcript")
        yield Vertical(
            Static("Ready", id="status", markup=False),
            Composer(
                id="composer", placeholder="Ask Poe to explore, change, or test this project…"
            ),
            id="composer-dock",
        )

    async def on_mount(self) -> None:
        self.query_one("#transcript", VerticalScroll).anchor()
        self.set_interval(0.1, self.refresh_activity_rows)
        for message in self.agent.session.messages:
            role = message["role"]
            if role == "assistant" and (thought := reasoning_text(message)):
                await self.add_reasoning_panel(thought, collapsed=True)
            if role in {"user", "assistant"} and message.get("content"):
                await self.add_message(role, message["content"])
            for call in message.get("tool_calls", []):
                await self.show_tool(call, track_time=False)
            if role == "tool":
                self.finish_tool(message["tool_call_id"], message.get("content", ""), None)
        self.set_status("Connecting tools…")
        connected = True
        try:
            await self.agent.start()
        except Exception as exc:
            connected = False
            await self.notice(str(exc), error=True)
        self.set_status("Ready" if connected else "MCP unavailable")
        self.query_one(Composer).focus()
        if self.initial_prompt and connected:
            self.query_one(Composer).text = self.initial_prompt
            self.query_one(Composer).action_submit()

    async def on_unmount(self) -> None:
        await self.agent.close()

    def set_status(self, text: str) -> None:
        status = self.query_one("#status", Static)
        status.update(text)
        status.display = text != "Ready"

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

    async def show_tool(self, call: dict, *, track_time: bool = True) -> None:
        function = call["function"]
        route = self.agent.tools.route(function["name"])
        name = route.display_name if route is not None else function["name"]
        raw = function.get("arguments", "")
        try:
            args = json.loads(raw)
            formatted = json.dumps(args, indent=2, ensure_ascii=False)
            target = args.get("command") or args.get("file_path") or args.get("dir_path") or ""
            if function["name"] == "list_dir" and not target:
                target = "."
        except (ValueError, AttributeError, TypeError):
            formatted, target = raw, ""
        target = clip(str(target).replace(chr(10), " "), 90)
        label = escape(activity_label(name, target))
        output = Static("Running…", markup=False)
        panel = Collapsible(
            Static(clip(formatted), markup=False, classes="arguments"),
            output,
            title=f"{SPINNER_FRAMES[0]} {label}",
            collapsed=True,
            classes="activity tool running",
        )
        self.tool_panels[call["id"]] = ToolActivity(
            panel,
            output,
            label,
            monotonic() if track_time else None,
        )
        self.running_tool_ids.add(call["id"])
        await self.query_one("#transcript", VerticalScroll).mount(panel)

    def finish_tool(self, call_id: str, content: str, success: bool | None) -> None:
        if call_id not in self.tool_panels:
            return
        activity = self.tool_panels[call_id]
        if success is None:
            success = not content.startswith(("Error:", "Interrupted."))
        activity.running = False
        self.running_tool_ids.discard(call_id)
        activity.output.update(content)
        self.complete_activity(
            activity.panel,
            activity.label,
            activity.started_at,
            success=success,
        )
        activity.panel.collapsed = success

    def complete_activity(
        self,
        panel: Collapsible,
        label: str,
        started_at: float | None,
        *,
        success: bool,
    ) -> None:
        """Settle a running row into its compact completed state."""
        panel.remove_class("running")
        panel.add_class("success" if success else "failure")
        elapsed = f" · {monotonic() - started_at:.1f}s" if started_at is not None else ""
        panel.title = f"{'✓' if success else '⚠'} {label}{elapsed}"

    async def add_reasoning_panel(
        self, text: str = "", *, collapsed: bool = False, running: bool = False
    ) -> tuple[Collapsible, Static]:
        output = Static(text, markup=False, classes="thought")
        panel = Collapsible(
            output,
            title=f"{SPINNER_FRAMES[0] if running else '✓'} Thinking",
            collapsed=collapsed,
            classes=f"activity thinking {'running' if running else 'success'}",
        )
        await self.query_one("#transcript", VerticalScroll).mount(panel)
        return panel, output

    async def append_reasoning(self, text: str) -> None:
        if self.thinking is None:
            self.thinking_panel, self.thinking = await self.add_reasoning_panel(running=True)
            self.thinking_started_at = monotonic()
            self.thinking_parts.clear()
        self.thinking_parts.append(text)
        self.thinking_dirty = True
        self.set_status("Thinking…")

    def flush_thinking(self) -> None:
        """Repaint on a timer; thinking arrives token by token."""
        if self.thinking_dirty and self.thinking is not None:
            self.thinking.update("".join(self.thinking_parts))
            self.thinking_dirty = False

    def refresh_activity_rows(self) -> None:
        """Flush streamed reasoning and animate each active compact row."""
        self.flush_thinking()
        self.spinner_index = (self.spinner_index + 1) % len(SPINNER_FRAMES)
        frame = SPINNER_FRAMES[self.spinner_index]
        for call_id in self.running_tool_ids:
            activity = self.tool_panels[call_id]
            activity.panel.title = f"{frame} {activity.label}"
        if self.thinking_panel is not None and self.thinking_started_at is not None:
            self.thinking_panel.title = f"{frame} Thinking"

    def end_reasoning(self, *, collapse: bool, success: bool = True) -> None:
        """Close the round's panel, folding it away only once something follows it."""
        self.flush_thinking()
        if self.thinking_panel is not None:
            self.complete_activity(
                self.thinking_panel,
                "Thinking",
                self.thinking_started_at,
                success=success,
            )
            if collapse:
                self.thinking_panel.collapsed = True
        self.thinking = self.thinking_panel = None
        self.thinking_started_at = None
        self.thinking_parts = []
        self.thinking_dirty = False

    async def finish_stream(self) -> None:
        if self.markdown_stream is not None:
            stream, self.markdown_stream = self.markdown_stream, None
            await stream.stop()

    async def handle_event(self, event: Event) -> None:
        if event.kind == "text":
            self.end_reasoning(collapse=True)
            if self.markdown_stream is None:
                body = await self.add_message("assistant", "")
                assert isinstance(body, Markdown)
                self.markdown_stream = Markdown.get_stream(body)
            await self.markdown_stream.write(event.text)
            self.set_status("Responding…")
        elif event.kind == "reasoning":
            await self.append_reasoning(event.text)
        elif event.kind == "assistant_done":
            self.flush_thinking()
            await self.finish_stream()
        elif event.kind == "tool_start":
            self.end_reasoning(collapse=True)
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
            self.end_reasoning(collapse=False)
            self.set_status("Ready")

    async def approve_tool(self, route: ToolRoute, args: dict) -> bool:
        self.set_status(f"Waiting for approval: {route.display_name}…")
        return bool(await self.push_screen_wait(ApprovalScreen(route, args)))

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
        reasoning_succeeded = True
        try:
            await self.agent.run(prompt, self.handle_event, self.approve_tool)
        except asyncio.CancelledError:
            reasoning_succeeded = False
            await self.notice("Turn cancelled. Completed changes are saved.")
            # Reconcile any running panel with the repaired, resumable transcript.
            for message in self.agent.session.messages:
                if message["role"] == "tool" and message["content"].startswith("Interrupted."):
                    self.finish_tool(message["tool_call_id"], message["content"], False)
            raise
        except Exception as exc:
            reasoning_succeeded = False
            await self.notice(str(exc), error=True)
        finally:
            self.end_reasoning(collapse=False, success=reasoning_succeeded)
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

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Yield Escape to a modal; app-level priority bindings are checked first."""
        if action == "cancel_turn" and isinstance(self.screen, ModalScreen):
            return False
        return True

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
        self.running_tool_ids.clear()
        self.end_reasoning(collapse=False)
        self.input_tokens = self.output_tokens = 0
        await self.query_one("#transcript", VerticalScroll).remove_children()
        await self.notice("New conversation.")
        self.set_status("Ready")
        self.query_one(Composer).focus()

    async def action_quit(self) -> None:
        if self.turn_worker is not None and self.busy:
            if not self.turn_worker.is_cancelled:
                self.turn_worker.cancel()
            with contextlib.suppress(WorkerCancelled, WorkerFailed):
                await self.turn_worker.wait()
        await self.agent.close()
        self.exit()

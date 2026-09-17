"""Textual chat interface; model work runs in a cancellable async worker."""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import monotonic

import flatlatex
from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.texmath import texmath_plugin
from rich.console import RenderableType
from rich.markup import escape
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.events import Click, Key
from textual.message import Message
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Markdown, Static, TextArea
from textual.widgets.markdown import MarkdownBlock, MarkdownStream
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

_LATEX_CONVERTER = flatlatex.converter()
_INLINE_MATH_TOKENS = {"math_inline", "math_inline_double", "math_single"}
_LATEX_FENCE_LANGUAGES = {"katex", "latex", "math", "tex"}


def _normalize_latex(source: str) -> str:
    """Remove presentation-only commands that flatlatex doesn't understand."""
    expression = re.sub(r"\s+", " ", source).strip()
    expression = re.sub(r"\\(?:left|right)(?=[()[\]{}|.])", "", expression)
    expression = re.sub(r"\\(?:qquad|quad)\b", " ", expression)
    expression = re.sub(r"\\[,;:!]", " ", expression)
    return re.sub(r"\\(log|ln|exp)\b", r"\1", expression)


def render_latex(source: str) -> str:
    """Convert a LaTeX math expression to terminal-friendly Unicode."""
    expression = source.strip()
    try:
        rendered = _LATEX_CONVERTER.convert(_normalize_latex(expression))
    except Exception:
        # A malformed expression should never interrupt response streaming.
        return expression
    # flatlatex removes braces even when it doesn't recognize a command. Keeping
    # the original expression is more useful than displaying a corrupted one.
    return expression if not rendered or "\\" in rendered else rendered


def render_latex_fence(source: str) -> str | None:
    """Render a fenced block only when all of its content is delimited math."""
    tokens = math_markdown_parser().parse(source)
    if not tokens or any(not token.type.startswith("math_block") for token in tokens):
        return None
    return "\n\n".join(render_latex(token.content) for token in tokens)


def math_markdown_parser() -> MarkdownIt:
    """Build the Markdown parser with common inline and display math delimiters."""
    parser = MarkdownIt("gfm-like").use(dollarmath_plugin)
    parser.use(texmath_plugin, delimiters="brackets")
    return parser


class LatexBlock(MarkdownBlock):
    """A display-math block rendered as centered terminal text."""

    DEFAULT_CSS = """
    LatexBlock {
        height: auto;
        margin: 0 0 1 0;
        text-align: center;
        color: $text-accent;
    }
    """

    def __init__(self, markdown: Markdown, token: Token) -> None:
        super().__init__(markdown, token)
        self.rendered = token.meta.get("rendered_latex", render_latex(token.content))
        self.set_content(Content(self.rendered))

    async def _update_from_block(self, block: MarkdownBlock) -> None:
        if isinstance(block, LatexBlock):
            self.rendered = block.rendered
            self.set_content(Content(self.rendered))
            self._copy_context(block)
        else:
            await super()._update_from_block(block)


class LatexMarkdown(Markdown):
    """Markdown widget that renders LaTeX math as Unicode."""

    def __init__(self, markdown: str | None = None, **kwargs) -> None:
        super().__init__(markdown, parser_factory=math_markdown_parser, **kwargs)

    def _parse_markdown(self, tokens: Iterable[Token]) -> Iterable[MarkdownBlock]:
        parsed_tokens = list(tokens)
        for token in parsed_tokens:
            if token.type == "inline" and token.children is not None:
                for child in token.children:
                    if child.type in _INLINE_MATH_TOKENS:
                        child.type = "text"
                        child.content = render_latex(child.content)
            elif token.type == "fence" and token.info.strip().lower() in _LATEX_FENCE_LANGUAGES:
                rendered = render_latex_fence(token.content)
                if rendered is not None:
                    token.type = "math_fence"
                    token.meta["rendered_latex"] = rendered
        yield from super()._parse_markdown(parsed_tokens)

    def unhandled_token(self, token: Token) -> MarkdownBlock | None:
        if token.type.startswith("math_block") or token.type == "math_fence":
            return LatexBlock(self, token)
        return super().unhandled_token(token)


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


@dataclass(frozen=True)
class ContextBreakdown:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None
    categories: tuple[tuple[str, int], ...]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def activity_label(name: str, target: object = "") -> str:
    """Turn a tool call into a short, readable activity label."""
    action = TOOL_ACTIONS.get(name, name if "/" in name else name.replace("_", " ").capitalize())
    return f"{action} {target}".strip()


def display_path(path: str) -> str:
    """Abbreviate the user's home directory as ~ the way a shell prompt does."""
    try:
        relative = Path(path).relative_to(Path.home())
    except (ValueError, RuntimeError):
        return path
    return "~" if relative == Path(".") else str(Path("~") / relative)


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


def _token_weight(value: object) -> int:
    """Return a model-agnostic size estimate for one request fragment."""
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return max(1, (len(serialized.encode()) + 3) // 4)


def estimate_context_breakdown(
    messages: list[dict],
    tools: list[dict],
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int | None = None,
) -> ContextBreakdown:
    """Estimate prompt categories, scaling them to the provider-reported total."""
    weights = {
        "System prompt": 0,
        "User messages": 0,
        "Assistant messages": 0,
        "Tool calls": 0,
        "Tool results": 0,
        "Tool definitions": 0,
        "Message overhead": 3 + 3 * len(messages),
    }
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            assistant = {key: value for key, value in message.items() if key != "tool_calls"}
            weights["Assistant messages"] += _token_weight(assistant)
            if message.get("tool_calls"):
                weights["Tool calls"] += _token_weight(message["tool_calls"])
        elif role == "system":
            weights["System prompt"] += _token_weight(message)
        elif role == "user":
            weights["User messages"] += _token_weight(message)
        elif role == "tool":
            weights["Tool results"] += _token_weight(message)
    if tools:
        weights["Tool definitions"] = _token_weight(tools)

    present = [(label, weight) for label, weight in weights.items() if weight]
    total_weight = sum(weight for _, weight in present)
    allocated = [prompt_tokens * weight // total_weight for _, weight in present]
    remainders = [prompt_tokens * weight % total_weight for _, weight in present]
    missing = prompt_tokens - sum(allocated)
    for index in sorted(range(len(present)), key=remainders.__getitem__, reverse=True)[:missing]:
        allocated[index] += 1

    return ContextBreakdown(
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        tuple((label, tokens) for (label, _), tokens in zip(present, allocated, strict=True)),
    )


def format_context_breakdown(breakdown: ContextBreakdown) -> RenderableType:
    """Lay out the current context distribution as a compact table."""
    table = Table(box=None, expand=True, padding=(0, 1))
    table.add_column("Category", style="#edecec")
    table.add_column("Tokens", justify="right", style="bold #9fbbe0")
    table.add_column("Share", justify="right", style="dim #edecec")
    rows = [*breakdown.categories, ("Latest response", breakdown.completion_tokens)]
    for label, tokens in rows:
        share = tokens / breakdown.total_tokens if breakdown.total_tokens else 0
        table.add_row(label, f"{tokens:,}", f"{share:.1%}")
    table.add_section()
    table.add_row("Total context", f"{breakdown.total_tokens:,}", "100.0%")
    return table


class ContextUsageScreen(ModalScreen[None]):
    """Current context composition, with reported totals and estimated categories."""

    DEFAULT_CSS = """
    ContextUsageScreen { align: center middle; background: $background 70%; }
    #context-dialog { width: 80%; max-width: 86; height: auto; max-height: 85%; padding: 1 2;
                      border: round #edecec 20%; background: #1b1913;
                      border-title-color: #9fbbe0; border-title-align: left; }
    #context-summary { color: #edecec; margin-bottom: 1; }
    #context-table { height: auto; max-height: 16; scrollbar-size: 1 1; }
    #context-table Static { height: auto; }
    #context-note { color: #edecec 60%; margin-top: 1; }
    #context-footer { height: 1; margin-top: 1; align-horizontal: right; }
    #context-close { height: 1; min-width: 10; border: none; padding: 0 2;
                     background: transparent; color: #9fbbe0; }
    #context-close:hover, #context-close:focus { background: #edecec 10%; text-style: bold; }
    """
    BINDINGS = [Binding("escape", "close", "Close", show=False)]

    def __init__(self, breakdown: ContextBreakdown):
        super().__init__()
        self.breakdown = breakdown

    def compose(self) -> ComposeResult:
        breakdown = self.breakdown
        summary = (
            f"Reported prompt {breakdown.prompt_tokens:,} · "
            f"latest response {breakdown.completion_tokens:,}"
        )
        if breakdown.cached_tokens is not None:
            summary += f" · cached prompt {breakdown.cached_tokens:,}"
        yield Vertical(
            Static(summary, id="context-summary", markup=False),
            VerticalScroll(
                Static(format_context_breakdown(breakdown)),
                id="context-table",
            ),
            Static(
                "Category counts are estimates based on the request payload, scaled to the "
                "provider-reported prompt total. Cached tokens overlap the prompt categories.",
                id="context-note",
                markup=False,
            ),
            Horizontal(Button("Close", id="context-close"), id="context-footer"),
            id="context-dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#context-dialog", Vertical).border_title = "Context tokens"
        self.query_one("#context-close", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "context-close":
            self.dismiss()

    def on_click(self, event: Click) -> None:
        """Dismiss when the click lands on the dimmed backdrop rather than the dialog."""
        if self.get_widget_at(*event.screen_offset)[0] is self:
            self.dismiss()

    def action_close(self) -> None:
        self.dismiss()


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
    MIN_HEIGHT = 1
    MAX_HEIGHT = 8

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
        """Grow with visible prompt lines, then shrink again as content is removed."""
        content_height = self.wrapped_document.height
        self.styles.height = max(self.MIN_HEIGHT, min(self.MAX_HEIGHT, content_height))


STAR_FAINT = Style(color="#55534e")
STAR_MID = Style(color="#969592")
STAR_BRIGHT = Style(color="#edecec")
STAR_ACCENT = Style(color="#9fbbe0")
# Glyph and style by distance from the meteor's head.
METEOR_TRAIL = (
    ("✦", STAR_BRIGHT + Style(bold=True)),
    ("━", STAR_BRIGHT),
    ("━", STAR_MID),
    ("─", STAR_MID),
    ("─", STAR_FAINT),
    ("·", STAR_FAINT),
)
METEOR_SPEED = 2


class Starfield(Widget):
    """Empty-state night sky with twinkling stars and an occasional shooting star."""

    DEFAULT_CSS = "Starfield { height: 1fr; width: 100%; padding: 1 3 0 3; }"
    TICK_SECONDS = 0.1
    CELLS_PER_STAR = 28

    def __init__(self, *, id: str | None = None):
        super().__init__(id=id)
        self.rng = random.Random()
        self.stars: dict[tuple[int, int], tuple[str, Style]] = {}
        self.twinkles: dict[tuple[int, int], int] = {}
        # The meteor flies left along one row with its trail to the right of the head.
        self.meteor_head: tuple[int, int] | None = None
        self.meteor_fuel = 0
        self.meteor_length = 0
        self.ticks_until_meteor = self.rng.randint(5, 15)

    def on_mount(self) -> None:
        self.set_interval(self.TICK_SECONDS, self.tick)

    def on_resize(self) -> None:
        self.scatter_stars()

    def scatter_stars(self) -> None:
        width, height = self.size
        self.stars.clear()
        self.twinkles.clear()
        self.meteor_head = None
        for _ in range(width * height // self.CELLS_PER_STAR):
            position = (self.rng.randrange(width), self.rng.randrange(height))
            roll = self.rng.random()
            if roll < 0.7:
                self.stars[position] = ("·", STAR_FAINT)
            elif roll < 0.92:
                self.stars[position] = ("⋆", STAR_MID)
            else:
                self.stars[position] = ("✦", STAR_ACCENT if roll > 0.97 else STAR_BRIGHT)

    def tick(self) -> None:
        # Hidden once a conversation starts; skip the work until it's shown again.
        if not self.display or not self.stars:
            return
        self.twinkles = {pos: ticks - 1 for pos, ticks in self.twinkles.items() if ticks > 1}
        if self.rng.random() < 0.2:
            self.twinkles[self.rng.choice(list(self.stars))] = self.rng.randint(3, 8)
        if self.meteor_head is not None:
            self.advance_meteor()
        else:
            self.ticks_until_meteor -= 1
            if self.ticks_until_meteor <= 0:
                self.launch_meteor()
        self.refresh()

    def launch_meteor(self) -> None:
        width, height = self.size
        self.meteor_head = (
            self.rng.randrange(width // 3, width),
            self.rng.randrange(height * 2 // 3 + 1),
        )
        self.meteor_fuel = self.rng.randint(max(6, width // 4), max(8, width // 2))
        self.meteor_length = 0
        self.ticks_until_meteor = self.rng.randint(15, 35)

    def advance_meteor(self) -> None:
        """Move the head left, then let the trail burn out behind it."""
        assert self.meteor_head is not None
        if self.meteor_fuel > 0:
            step = min(METEOR_SPEED, self.meteor_fuel)
            self.meteor_fuel -= step
            x, y = self.meteor_head
            self.meteor_head = (x - step, y)
            self.meteor_length = min(self.meteor_length + step, len(METEOR_TRAIL))
            if x - step < 0:
                self.meteor_fuel = 0
        else:
            self.meteor_length -= METEOR_SPEED
            if self.meteor_length <= 0:
                self.meteor_head = None

    def meteor_cells(self) -> dict[tuple[int, int], tuple[str, Style]]:
        """Lay out the trail, shifting to fainter glyphs as it burns out.

        Cells may fall off the left edge; render() crops them.
        """
        if self.meteor_head is None:
            return {}
        x, y = self.meteor_head
        fade = 0 if self.meteor_fuel > 0 else len(METEOR_TRAIL) - self.meteor_length
        return {
            (x + distance, y): METEOR_TRAIL[fade + distance]
            for distance in range(self.meteor_length)
        }

    def render(self) -> RenderableType:
        width, height = self.size
        cells = dict(self.stars)
        for position in self.twinkles:
            if position in cells:
                cells[position] = (cells[position][0], STAR_BRIGHT)
        cells.update(self.meteor_cells())
        rows: list[list[tuple[int, str, Style]]] = [[] for _ in range(height)]
        for (x, y), (glyph, style) in cells.items():
            if 0 <= x < width and 0 <= y < height:
                rows[y].append((x, glyph, style))
        text = Text(no_wrap=True, overflow="crop")
        for y, row in enumerate(rows):
            column = 0
            for x, glyph, style in sorted(row, key=lambda cell: cell[0]):
                text.append(" " * (x - column))
                text.append(glyph, style)
                column = x + 1
            if y < height - 1:
                text.append("\n")
        return text


class PoeApp(App, inherit_bindings=False):
    TITLE = "Poe"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { background: #14120b; color: #edecec; }
    #app-header { height: 2; width: 100%; padding: 1 3 0 3; background: #14120b; }
    #header-path { width: 1fr; color: #edecec 60%; text-align: left;
                   text-overflow: ellipsis; overflow: hidden; }
    #header-model { width: auto; max-width: 40%; color: #edecec;
                    text-align: right;
                    text-overflow: ellipsis; overflow: hidden; }
    #transcript { width: 100%; padding: 1 0; scrollbar-size: 1 1; }
    #transcript.empty { height: auto; max-height: 50%; padding: 0; }
    #transcript > .message, #transcript > .notice,
    #transcript > .error, #transcript > Collapsible { margin: 0 3 1 3; }
    .message { height: auto; }
    .assistant { padding: 0 1 0 1; }
    #transcript > .assistant { margin: 0 3 0 3; }
    .user { padding: 1 2; background: #1b1913; border-left: solid #9fbbe0; }
    .message Markdown { padding: 0; margin: 0; background: transparent; }
    .message Static { height: auto; }
    .notice { color: #edecec 60%; }
    .error { color: #cf2d56; }
    Collapsible { background: transparent; border-top: none; padding: 0 1; }
    .activity CollapsibleTitle { color: #edecec 60%; padding: 0 1; }
    .activity.running CollapsibleTitle { color: #9fbbe0; }
    .activity.success CollapsibleTitle { color: #1f8a65; }
    .activity.failure CollapsibleTitle { color: #cf2d56; }
    .arguments { color: #edecec 60%; margin-bottom: 1; }
    .thought { color: #edecec 60%; text-style: italic; }
    #composer-dock { height: auto; padding: 0 2; background: #14120b; }
    #status { visibility: hidden; height: 1; padding: 0 1; color: #9fbbe0;
              background: transparent; text-overflow: ellipsis; overflow: hidden; }
    #usage { visibility: hidden; height: 1; padding: 0 1; color: #edecec 60%;
             background: transparent; text-overflow: ellipsis; overflow: hidden;
             link-color: #edecec 60%; link-style: underline;
             link-style-hover: bold underline; }
    #composer-box { height: auto; border: round #edecec 10%; background: #14120b; }
    #composer-box:focus-within { border: round #9fbbe0; }
    #composer-prompt { width: 2; height: 1; padding: 0 0 0 1; color: #9fbbe0; text-style: bold; }
    #composer { height: 1; max-height: 8; margin: 0; padding: 0 1; border: none;
                background: #14120b; color: #edecec; }
    #composer:focus { border: none; }
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
        self.context_tokens = 0
        self.context_usage_reported = False
        self.context_breakdown: ContextBreakdown | None = None
        self.cached_tokens = 0
        self.cache_write_tokens = 0
        self.cache_usage_reported = False
        self.cost = Decimal()
        self.cost_reported = False

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static(display_path(self.agent.session.cwd), id="header-path", markup=False),
            Static(self.agent.config.model, id="header-model", markup=False),
            id="app-header",
        )
        yield Starfield(id="starfield")
        yield VerticalScroll(id="transcript")
        yield Vertical(
            Static("Ready", id="status"),
            Horizontal(
                Static("❯", id="composer-prompt", markup=False),
                Composer(id="composer"),
                id="composer-box",
            ),
            Static("", id="usage", markup=True),
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
        self.show_empty_state(not self.agent.session.messages)
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

    def on_key(self, event: Key) -> None:
        """Route printable keystrokes to the composer when it isn't focused."""
        if isinstance(self.screen, ModalScreen):
            return
        composer = self.query_one(Composer)
        if self.focused is composer:
            return
        if event.is_printable and event.character:
            composer.focus()
            composer.insert(event.character)
            event.stop()

    def show_empty_state(self, show: bool) -> None:
        """Show the night sky until the conversation has something in it."""
        self.query_one(Starfield).display = show
        self.query_one("#transcript", VerticalScroll).set_class(show, "empty")

    def set_status(self, text: str) -> None:
        status = self.query_one("#status", Static)
        status.update(escape(text))
        status.visible = text != "Ready"
        self.refresh_usage()

    def refresh_usage(self) -> None:
        """Render the context and pricing line that sits below the composer."""
        usage = self.query_one("#usage", Static)
        summary = self.usage_summary(link_context=True)
        usage.update(summary)
        usage.visible = bool(summary)
        usage.tooltip = "Show context token breakdown" if self.context_breakdown else None

    def usage_summary(self, *, link_context: bool = False) -> str:
        """Format cumulative usage fields reported by OpenRouter for this chat."""
        parts = []
        if self.context_usage_reported:
            context = f"context {self.context_tokens:,} tokens"
            if link_context and self.context_breakdown is not None:
                context = f"[@click=app.show_context]{context}[/]"
            parts.append(context)
        if self.input_tokens or self.output_tokens:
            parts.append(f"usage {self.input_tokens:,} in / {self.output_tokens:,} out")
        if self.cache_usage_reported:
            cache = f"cache {self.cached_tokens:,} read"
            if self.cache_write_tokens:
                cache += f" / {self.cache_write_tokens:,} write"
            parts.append(cache)
        if self.cost_reported:
            parts.append(f"cost ${self.cost:f}")
        return " · ".join(parts)

    def add_usage(self, usage: dict) -> None:
        """Accumulate one completion's optional usage accounting fields."""
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens

        if usage.get("total_tokens") is not None:
            self.context_tokens = usage["total_tokens"]
            self.context_usage_reported = True
        elif "prompt_tokens" in usage or "completion_tokens" in usage:
            self.context_tokens = prompt_tokens + completion_tokens
            self.context_usage_reported = True

        details = usage.get("prompt_tokens_details")
        latest_cached_tokens = None
        if isinstance(details, dict) and (
            "cached_tokens" in details or "cache_write_tokens" in details
        ):
            self.cache_usage_reported = True
            latest_cached_tokens = details.get("cached_tokens", 0) or 0
            self.cached_tokens += latest_cached_tokens
            self.cache_write_tokens += details.get("cache_write_tokens", 0) or 0

        reported_prompt_tokens = prompt_tokens
        if "prompt_tokens" not in usage and usage.get("total_tokens") is not None:
            reported_prompt_tokens = max(0, usage["total_tokens"] - completion_tokens)
        self.context_breakdown = estimate_context_breakdown(
            self.agent.session.messages,
            self.agent.tools.definitions(),
            reported_prompt_tokens,
            completion_tokens,
            latest_cached_tokens,
        )

        if usage.get("cost") is not None:
            try:
                self.cost += Decimal(str(usage["cost"]))
            except (InvalidOperation, TypeError, ValueError):
                pass
            else:
                self.cost_reported = True

        self.refresh_usage()

    async def notice(self, text: str, *, error: bool = False) -> None:
        await self.query_one("#transcript", VerticalScroll).mount(
            Static(text, markup=False, classes="error" if error else "notice")
        )

    async def add_message(self, role: str, text: str) -> Markdown | Static:
        body = LatexMarkdown(text) if role == "assistant" else Static(text, markup=False)
        await self.query_one("#transcript", VerticalScroll).mount(
            Vertical(
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
            self.add_usage(event.data)
        elif event.kind == "done":
            self.end_reasoning(collapse=False)
            self.set_status("Ready")

    async def approve_tool(self, route: ToolRoute, args: dict) -> bool:
        self.set_status(f"Waiting for approval: {route.display_name}…")
        return bool(await self.push_screen_wait(ApprovalScreen(route, args)))

    def action_show_context(self) -> None:
        if self.context_breakdown is not None:
            self.push_screen(ContextUsageScreen(self.context_breakdown))

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
            self.show_empty_state(False)
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
        self.context_tokens = 0
        self.context_usage_reported = False
        self.context_breakdown = None
        self.cached_tokens = self.cache_write_tokens = 0
        self.cache_usage_reported = False
        self.cost = Decimal()
        self.cost_reported = False
        await self.query_one("#transcript", VerticalScroll).remove_children()
        self.show_empty_state(True)
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

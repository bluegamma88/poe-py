"""Provider-neutral tool definitions, routing, and execution results."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def openrouter_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class ToolResult:
    content: str
    success: bool = True


class ToolBackend(Protocol):
    @property
    def source_id(self) -> str: ...

    @property
    def namespaced(self) -> bool: ...

    @property
    def approval(self) -> Literal["always", "never"]: ...

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    def specs(self) -> list[ToolSpec]: ...

    async def run(self, name: str, args: Any) -> ToolResult: ...


@dataclass(frozen=True)
class ToolRoute:
    backend: ToolBackend
    original_name: str
    display_name: str


def _component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    return normalized or "tool"


def mcp_tool_name(server_id: str, tool_name: str) -> str:
    """Build a stable OpenAI-compatible name, retaining uniqueness when truncated."""
    raw = f"mcp_{_component(server_id)}_{_component(tool_name)}"
    if len(raw) <= 64:
        return raw
    digest = hashlib.sha256(f"{server_id}\0{tool_name}".encode()).hexdigest()[:8]
    return f"{raw[:55]}_{digest}"


class ToolRegistry:
    def __init__(self, backends: list[ToolBackend]):
        self.backends = backends
        self._routes: dict[str, ToolRoute] = {}
        self._specs: list[ToolSpec] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        started: list[ToolBackend] = []
        try:
            for backend in self.backends:
                await backend.start()
                started.append(backend)
            self._rebuild()
        except BaseException:
            for backend in reversed(started):
                await backend.close()
            raise
        self._started = True

    async def close(self) -> None:
        if not self._started:
            return
        error: BaseException | None = None
        try:
            for backend in reversed(self.backends):
                try:
                    await backend.close()
                except BaseException as exc:
                    error = error or exc
        finally:
            self._started = False
            self._routes.clear()
            self._specs.clear()
        if error is not None:
            raise error

    def _rebuild(self) -> None:
        routes: dict[str, ToolRoute] = {}
        specs: list[ToolSpec] = []
        for backend in self.backends:
            for spec in backend.specs():
                exposed = (
                    spec.name
                    if not backend.namespaced
                    else mcp_tool_name(backend.source_id, spec.name)
                )
                if exposed in routes:
                    other = routes[exposed]
                    raise ValueError(
                        f"Tool name collision for {exposed!r}: "
                        f"{other.display_name} and {backend.source_id}/{spec.name}"
                    )
                description = spec.description
                if backend.namespaced:
                    description = f"[MCP server: {backend.source_id}] {description}"
                routes[exposed] = ToolRoute(
                    backend=backend,
                    original_name=spec.name,
                    display_name=(
                        spec.name if not backend.namespaced else f"{backend.source_id}/{spec.name}"
                    ),
                )
                specs.append(replace(spec, name=exposed, description=description))
        self._routes = routes
        self._specs = specs

    def definitions(self) -> list[dict[str, Any]]:
        return [spec.openrouter_definition() for spec in self._specs]

    def route(self, name: str) -> ToolRoute | None:
        return self._routes.get(name)

    async def run(self, name: str, args: Any) -> ToolResult:
        route = self.route(name)
        if route is None:
            return ToolResult(f"Error: Unknown tool: {name}", success=False)
        return await route.backend.run(route.original_name, args)

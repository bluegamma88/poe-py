"""MCP client backends for stdio and Streamable HTTP servers."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
from mcp import Client, Implementation, MCPError, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.streamable_http import streamable_http_client

from poe import __version__
from poe.config import McpServerConfig
from poe.tooling import ToolResult, ToolSpec
from poe.tools import clip


def _resource_link(block: mcp_types.ResourceLink) -> str:
    label = block.title or block.name
    details = f" ({block.mime_type})" if block.mime_type else ""
    return f"[MCP resource link] {label}: {block.uri}{details}"


def render_result(result: mcp_types.CallToolResult) -> ToolResult:
    """Convert MCP's content-block result into Poe's text-only tool result."""
    parts: list[str] = []
    unsupported: list[str] = []
    for block in result.content:
        if isinstance(block, mcp_types.TextContent):
            parts.append(block.text)
        elif isinstance(block, mcp_types.ResourceLink):
            parts.append(_resource_link(block))
        elif isinstance(block, mcp_types.EmbeddedResource):
            resource = block.resource
            if isinstance(resource, mcp_types.TextResourceContents):
                parts.append(f"[MCP resource {resource.uri}]\n{resource.text}")
            else:
                unsupported.append(f"binary resource {resource.uri}")
        elif isinstance(block, mcp_types.ImageContent):
            unsupported.append(f"image ({block.mime_type})")
        elif isinstance(block, mcp_types.AudioContent):
            unsupported.append(f"audio ({block.mime_type})")

    if not parts and result.structured_content is not None:
        parts.append(
            json.dumps(result.structured_content, ensure_ascii=False, sort_keys=True, default=str)
        )
    if unsupported:
        parts.append("[Unsupported MCP content omitted: " + ", ".join(unsupported) + "]")
    if not parts:
        parts.append("[MCP tool returned no content]")
    unsupported_only = (
        bool(result.content)
        and len(unsupported) == len(result.content)
        and result.structured_content is None
    )
    success = not result.is_error and not unsupported_only
    return ToolResult(clip("\n\n".join(parts)), success)


@dataclass
class _Call:
    name: str
    args: dict[str, Any]
    future: asyncio.Future[mcp_types.CallToolResult]
    cancelled: asyncio.Event


class McpToolBackend:
    def __init__(
        self,
        config: McpServerConfig,
        workspace: Path,
        *,
        target: Any = None,
    ):
        self.config = config
        self.workspace = workspace
        self.source_id = config.name
        self.namespaced = True
        self.approval = config.approval
        self._target = target
        self._client: Client | None = None
        self._specs: list[ToolSpec] = []
        self._queue: asyncio.Queue[_Call | None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._active: _Call | None = None

    def _env(self) -> dict[str, str] | None:
        missing = [name for name in self.config.env_from if not os.environ.get(name)]
        if missing:
            raise ValueError(
                f"MCP server {self.source_id!r} requires environment variable(s): "
                + ", ".join(missing)
            )
        return {name: os.environ[name] for name in self.config.env_from} or None

    def _cwd(self) -> Path | None:
        if self.config.cwd is None:
            return self.workspace
        configured = Path(self.config.cwd).expanduser()
        path = configured if configured.is_absolute() else self.workspace / configured
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"MCP server {self.source_id!r} cwd is not a directory: {path}")
        return path

    async def _make_target(self, stack: AsyncExitStack) -> Any:
        if self._target is not None:
            return self._target
        if self.config.transport == "stdio":
            assert self.config.command is not None
            return StdioServerParameters(
                command=self.config.command,
                args=list(self.config.args),
                env=self._env(),
                cwd=self._cwd(),
            )
        assert self.config.url is not None
        headers = {}
        missing = []
        for header, variable in self.config.header_env:
            if not os.environ.get(variable):
                missing.append(variable)
            else:
                headers[header] = os.environ[variable]
        if missing:
            raise ValueError(
                f"MCP server {self.source_id!r} requires environment variable(s): "
                + ", ".join(missing)
            )
        if not headers:
            return self.config.url
        http = await stack.enter_async_context(httpx2.AsyncClient(headers=headers))
        return streamable_http_client(self.config.url, http_client=http)

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._serve(ready), name=f"poe-mcp-{self.source_id}")
        try:
            await ready
        except BaseException as exc:
            if self._task is not None and not self._task.done():
                self._task.cancel()
            if self._task is not None:
                with contextlib.suppress(BaseException):
                    await self._task
            self._reset()
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError(f"Could not start MCP server {self.source_id!r}: {exc}") from exc

    async def _serve(self, ready: asyncio.Future[None]) -> None:
        stack = AsyncExitStack()
        try:
            target = await self._make_target(stack)
            client = Client(
                target,
                client_info=Implementation(name="poe-py", version=__version__),
                read_timeout_seconds=self.config.tool_timeout_seconds,
            )
            self._client = await stack.enter_async_context(client)
            self._specs = await self._discover(self._client)
            ready.set_result(None)
            assert self._queue is not None
            while request := await self._queue.get():
                await self._execute(self._client, request)
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
        finally:
            await stack.aclose()
            self._client = None

    async def _discover(self, client: Client) -> list[ToolSpec]:
        tools: list[mcp_types.Tool] = []
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                break
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError(f"MCP server {self.source_id!r} returned duplicate tool names")
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description or tool.title or f"Run {tool.name}.",
                input_schema=tool.input_schema,
            )
            for tool in tools
        ]

    async def _execute(self, client: Client, request: _Call) -> None:
        self._active = request
        call = asyncio.create_task(
            client.call_tool(
                request.name,
                request.args,
                read_timeout_seconds=self.config.tool_timeout_seconds,
            )
        )
        cancelled = asyncio.create_task(request.cancelled.wait())
        try:
            done, _ = await asyncio.wait({call, cancelled}, return_when=asyncio.FIRST_COMPLETED)
            if cancelled in done:
                call.cancel()
                with contextlib.suppress(BaseException):
                    await call
                if not request.future.done():
                    request.future.cancel()
                return
            cancelled.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancelled
            if not request.future.done():
                try:
                    request.future.set_result(call.result())
                except BaseException as exc:
                    request.future.set_exception(exc)
        finally:
            self._active = None

    async def close(self) -> None:
        task = self._task
        queue = self._queue
        if task is None:
            return
        if self._active is not None:
            self._active.cancelled.set()
        if queue is not None and not task.done():
            await queue.put(None)
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._reset()

    def _reset(self) -> None:
        self._client = None
        self._specs.clear()
        self._queue = None
        self._task = None
        self._active = None

    def specs(self) -> list[ToolSpec]:
        return self._specs.copy()

    async def run(self, name: str, args: Any) -> ToolResult:
        if self._client is None or self._queue is None or self._task is None:
            return ToolResult(f"Error: MCP server {self.source_id!r} is not connected", False)
        if not isinstance(args, dict):
            return ToolResult("Error: Tool arguments must be a JSON object", False)
        loop = asyncio.get_running_loop()
        request = _Call(name, args, loop.create_future(), asyncio.Event())
        try:
            await self._queue.put(request)
            result = await request.future
            return render_result(result)
        except asyncio.CancelledError:
            request.cancelled.set()
            raise
        except MCPError as exc:
            return ToolResult(
                clip(f"MCP server {self.source_id!r} rejected the call: {exc}"), False
            )
        except Exception as exc:
            return ToolResult(clip(f"MCP server {self.source_id!r} call failed: {exc}"), False)

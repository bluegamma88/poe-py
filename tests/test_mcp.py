import asyncio
import json
import sys

import pytest
from mcp import types as mcp_types
from mcp.server.mcpserver import MCPServer

from poe.agent import Agent
from poe.config import Config, McpServerConfig
from poe.mcp import McpToolBackend, render_result
from poe.sessions import Session, SessionStore
from poe.tooling import ToolRegistry, ToolResult, ToolSpec, mcp_tool_name


async def ignore(event):
    pass


async def test_in_process_mcp_server_is_discovered_namespaced_and_called(tmp_path):
    server = MCPServer("test-server")

    @server.tool(name="echo.text")
    def echo(text: str) -> str:
        """Echo text from the test server."""
        return f"remote: {text}"

    config = McpServerConfig(
        name="demo",
        transport="stdio",
        command="unused-for-in-process-test",
        approval="never",
    )
    backend = McpToolBackend(config, tmp_path, target=server)
    registry = ToolRegistry([backend])
    await registry.start()
    try:
        definitions = registry.definitions()
        assert definitions[0]["function"]["name"] == "mcp_demo_echo_text"
        assert definitions[0]["function"]["parameters"]["required"] == ["text"]
        assert definitions[0]["function"]["description"].startswith("[MCP server: demo]")
        result = await registry.run("mcp_demo_echo_text", {"text": "hello"})
        assert result.success
        assert result.content == "remote: hello"
    finally:
        await registry.close()


async def test_stdio_mcp_server_lifecycle_and_call(tmp_path):
    server_path = tmp_path / "server.py"
    server_path.write_text(
        """
from mcp.server.mcpserver import MCPServer

server = MCPServer("stdio-test")

@server.tool()
def add(left: int, right: int) -> int:
    \"\"\"Add two integers.\"\"\"
    return left + right

server.run(transport="stdio")
"""
    )
    backend = McpToolBackend(
        McpServerConfig(
            name="math",
            transport="stdio",
            command=sys.executable,
            args=(str(server_path),),
            approval="never",
        ),
        tmp_path,
    )
    registry = ToolRegistry([backend])
    await registry.start()
    try:
        result = await registry.run("mcp_math_add", {"left": 2, "right": 3})
        assert result.success
        assert result.content == "5"
    finally:
        await registry.close()


async def test_cancelling_mcp_call_propagates_and_connection_closes(tmp_path):
    started = asyncio.Event()
    server = MCPServer("slow-test")

    @server.tool()
    async def slow() -> str:
        """Wait until cancelled."""
        started.set()
        await asyncio.sleep(30)
        return "late"

    backend = McpToolBackend(
        McpServerConfig(
            name="slow",
            transport="stdio",
            command="unused",
            approval="never",
        ),
        tmp_path,
        target=server,
    )
    registry = ToolRegistry([backend])
    await registry.start()
    call = asyncio.create_task(registry.run("mcp_slow_slow", {}))
    await asyncio.wait_for(started.wait(), 3)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    await asyncio.wait_for(registry.close(), 3)


def test_mcp_result_conversion_handles_structured_and_unsupported_content():
    structured = render_result(
        mcp_types.CallToolResult(content=[], structuredContent={"answer": 42})
    )
    assert structured == ToolResult('{"answer": 42}', True)

    image = render_result(
        mcp_types.CallToolResult(
            content=[mcp_types.ImageContent(data="ignored", mimeType="image/png")]
        )
    )
    assert not image.success
    assert "Unsupported MCP content" in image.content
    assert "ignored" not in image.content


def test_mcp_tool_names_are_stable_and_bounded():
    first = mcp_tool_name("server.with.dots", "tool/with spaces")
    assert first == "mcp_server_with_dots_tool_with_spaces"
    long_name = mcp_tool_name("server", "x" * 100)
    assert len(long_name) == 64
    assert long_name == mcp_tool_name("server", "x" * 100)
    assert long_name != mcp_tool_name("server", "y" * 100)


async def test_mcp_server_named_local_is_still_namespaced():
    backend = ApprovalBackend()
    backend.source_id = "local"
    registry = ToolRegistry([backend])
    await registry.start()
    try:
        assert registry.definitions()[0]["function"]["name"] == "mcp_local_send"
    finally:
        await registry.close()


def test_mcp_config_parses_both_transports_without_resolving_secrets(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
        [mcp.servers.local]
        transport = "stdio"
        command = "uvx"
        args = ["example-server"]
        env_from = ["EXAMPLE_TOKEN"]
        approval = "never"

        [mcp.servers.remote]
        transport = "streamable-http"
        url = "https://example.com/mcp/"
        header_env = { Authorization = "REMOTE_AUTH" }
        tool_timeout_seconds = 12.5
        """
    )
    config = Config.load(path, dotenv_path=tmp_path / "missing.env")
    local, remote = config.mcp_servers
    assert local.args == ("example-server",)
    assert local.env_from == ("EXAMPLE_TOKEN",)
    assert local.approval == "never"
    assert remote.url == "https://example.com/mcp"
    assert remote.header_env == (("Authorization", "REMOTE_AUTH"),)
    assert "secret-value" not in repr(config)


async def test_missing_stdio_environment_is_an_actionable_startup_error(tmp_path, monkeypatch):
    monkeypatch.delenv("POE_TEST_MCP_TOKEN", raising=False)
    backend = McpToolBackend(
        McpServerConfig(
            name="private",
            transport="stdio",
            command="unused",
            env_from=("POE_TEST_MCP_TOKEN",),
        ),
        tmp_path,
    )
    with pytest.raises(RuntimeError, match="POE_TEST_MCP_TOKEN"):
        await backend.start()
    assert backend._task is None


class ApprovalBackend:
    source_id = "remote"
    namespaced = True
    approval = "always"

    def __init__(self):
        self.calls = []

    async def start(self):
        pass

    async def close(self):
        pass

    def specs(self):
        return [
            ToolSpec(
                "send",
                "Send something.",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ]

    async def run(self, name, args):
        self.calls.append((name, args))
        return ToolResult("sent")


class ApprovalModel:
    def __init__(self):
        self.requests = []

    async def complete(self, messages, tools, emit):
        self.requests.append((json.loads(json.dumps(messages)), tools))
        if len(self.requests) == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "mcp_remote_send",
                            "arguments": '{"text": "hello"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "Not sent."}


async def test_agent_denies_mcp_tool_without_running_backend(tmp_path):
    backend = ApprovalBackend()
    model = ApprovalModel()
    agent = Agent(
        Config(),
        Session(cwd=str(tmp_path), model="test"),
        SessionStore(tmp_path / "sessions"),
        model,
        tools=ToolRegistry([backend]),
    )

    async def deny(route, args):
        assert route.display_name == "remote/send"
        return False

    await agent.run("send hello", ignore, deny)
    assert backend.calls == []
    assert "Denied" in model.requests[1][0][-1]["content"]


async def test_agent_supplies_registry_definitions_to_model(tmp_path):
    backend = ApprovalBackend()
    backend.approval = "never"
    model = ApprovalModel()
    agent = Agent(
        Config(),
        Session(cwd=str(tmp_path), model="test"),
        SessionStore(tmp_path / "sessions"),
        model,
        tools=ToolRegistry([backend]),
    )
    await agent.run("send hello", ignore)
    assert model.requests[0][1][0]["function"]["name"] == "mcp_remote_send"
    assert backend.calls == [("send", {"text": "hello"})]

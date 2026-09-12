"""Read the same OpenRouter settings as the Rust implementation."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    transport: Literal["stdio", "streamable-http"]
    command: str | None = None
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env_from: tuple[str, ...] = ()
    url: str | None = None
    header_env: tuple[tuple[str, str], ...] = ()
    tool_timeout_seconds: float = 60.0
    approval: Literal["always", "never"] = "always"


@dataclass(frozen=True)
class Config:
    model: str = "openai/gpt-oss-120b"
    api_key: str = field(default="", repr=False)
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    max_tool_rounds: int = 50
    mcp_servers: tuple[McpServerConfig, ...] = ()

    @classmethod
    def load(
        cls, path: Path | None = None, *, model: str | None = None, dotenv_path: Path | None = None
    ) -> Config:
        path = path or Path.home() / ".poe" / "config.toml"
        try:
            data = tomllib.loads(path.read_text()) if path.exists() else {}
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Could not read {path}: {exc}") from exc
        provider = data.get("openrouter", {})
        if not isinstance(provider, dict):
            raise ValueError("[openrouter] must be a TOML table")

        def string(value: object, name: str, *, empty: bool = False) -> str:
            if not isinstance(value, str) or (not empty and not value.strip()):
                raise ValueError(f"{name} must be a {'possibly empty ' if empty else ''}string")
            return value.strip()

        env = string(provider.get("api_key_env", cls.api_key_env), "openrouter.api_key_env")
        key = string(provider.get("api_key", ""), "openrouter.api_key", empty=True)
        dotenv_path = dotenv_path or Path.cwd() / ".env"
        try:
            local_env = dotenv_values(dotenv_path, interpolate=False)
        except OSError as exc:
            raise ValueError(f"Could not read {dotenv_path}: {exc}") from exc
        rounds = data.get("max_tool_rounds", cls.max_tool_rounds)
        if type(rounds) is not int or not 1 <= rounds <= 250:
            raise ValueError("max_tool_rounds must be an integer between 1 and 250")
        base_url = string(provider.get("base_url", cls.base_url), "openrouter.base_url")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("openrouter.base_url must start with https:// or http://")
        mcp_servers = cls._mcp_servers(data.get("mcp", {}), string)
        return cls(
            model=string(model if model is not None else data.get("model", cls.model), "model"),
            api_key=key or os.environ.get(env, "").strip() or (local_env.get(env) or "").strip(),
            api_key_env=env,
            base_url=base_url.rstrip("/"),
            max_tool_rounds=rounds,
            mcp_servers=mcp_servers,
        )

    @staticmethod
    def _mcp_servers(mcp: object, string) -> tuple[McpServerConfig, ...]:
        if not isinstance(mcp, dict):
            raise ValueError("[mcp] must be a TOML table")
        unknown_mcp = mcp.keys() - {"servers"}
        if unknown_mcp:
            raise ValueError(f"Unknown MCP setting(s): {sorted(unknown_mcp)}")
        servers = mcp.get("servers", {})
        if not isinstance(servers, dict):
            raise ValueError("[mcp.servers] must be a TOML table")
        parsed = []
        allowed = {
            "transport",
            "command",
            "args",
            "cwd",
            "env_from",
            "url",
            "header_env",
            "tool_timeout_seconds",
            "approval",
        }

        def strings(value: object, name: str) -> tuple[str, ...]:
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise ValueError(f"{name} must be an array of non-empty strings")
            return tuple(item.strip() for item in value)

        for name, raw in servers.items():
            label = f"mcp.servers.{name}"
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ValueError("MCP server names may contain letters, digits, _, -, and .")
            if not isinstance(raw, dict):
                raise ValueError(f"[{label}] must be a TOML table")
            unknown = raw.keys() - allowed
            if unknown:
                raise ValueError(f"Unknown {label} setting(s): {sorted(unknown)}")
            transport = string(raw.get("transport", ""), f"{label}.transport")
            if transport not in {"stdio", "streamable-http"}:
                raise ValueError(f"{label}.transport must be 'stdio' or 'streamable-http'")
            approval = string(raw.get("approval", "always"), f"{label}.approval")
            if approval not in {"always", "never"}:
                raise ValueError(f"{label}.approval must be 'always' or 'never'")
            timeout = raw.get("tool_timeout_seconds", 60)
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError(f"{label}.tool_timeout_seconds must be a positive number")
            args = strings(raw.get("args", []), f"{label}.args")
            env_from = strings(raw.get("env_from", []), f"{label}.env_from")
            header_env = raw.get("header_env", {})
            if not isinstance(header_env, dict) or not all(
                isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
                for key, value in header_env.items()
            ):
                raise ValueError(f"{label}.header_env must map header names to environment names")
            command = raw.get("command")
            cwd = raw.get("cwd")
            url = raw.get("url")
            for value, key in ((command, "command"), (cwd, "cwd"), (url, "url")):
                if value is not None:
                    string(value, f"{label}.{key}")
            if transport == "stdio":
                if command is None:
                    raise ValueError(f"{label}.command is required for stdio")
                if url is not None or header_env:
                    raise ValueError(f"{label}: url/header_env are only valid for streamable-http")
            else:
                if url is None:
                    raise ValueError(f"{label}.url is required for streamable-http")
                if not url.startswith(("https://", "http://")):
                    raise ValueError(f"{label}.url must start with https:// or http://")
                if command is not None or args or cwd is not None or env_from:
                    raise ValueError(f"{label}: command/args/cwd/env_from are only valid for stdio")
            parsed.append(
                McpServerConfig(
                    name=name,
                    transport=transport,
                    command=command.strip() if command is not None else None,
                    args=args,
                    cwd=cwd.strip() if cwd is not None else None,
                    env_from=env_from,
                    url=url.rstrip("/") if url is not None else None,
                    header_env=tuple(
                        (key.strip(), value.strip()) for key, value in header_env.items()
                    ),
                    tool_timeout_seconds=float(timeout),
                    approval=approval,
                )
            )
        return tuple(parsed)

    def require_api_key(self) -> None:
        if not self.api_key:
            raise ValueError(
                f"Set {self.api_key_env} in your environment or .env, "
                "or [openrouter].api_key in ~/.poe/config.toml."
            )

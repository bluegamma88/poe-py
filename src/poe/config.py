"""Read the same OpenRouter settings as the Rust implementation."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class Config:
    model: str = "openai/gpt-oss-120b"
    api_key: str = field(default="", repr=False)
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    max_tool_rounds: int = 50

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
        return cls(
            model=string(model if model is not None else data.get("model", cls.model), "model"),
            api_key=key or os.environ.get(env, "").strip() or (local_env.get(env) or "").strip(),
            api_key_env=env,
            base_url=base_url.rstrip("/"),
            max_tool_rounds=rounds,
        )

    def require_api_key(self) -> None:
        if not self.api_key:
            raise ValueError(
                f"Set {self.api_key_env} in your environment or .env, "
                "or [openrouter].api_key in ~/.poe/config.toml."
            )

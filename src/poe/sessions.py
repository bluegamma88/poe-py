"""Private, atomic JSON transcripts; intentionally separate from Rust sessions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from poe.tools import atomic_write


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Session:
    cwd: str
    model: str
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    messages: list[dict] = field(default_factory=list)
    version: int = 1

    @property
    def title(self) -> str:
        return next((m["content"] for m in self.messages if m["role"] == "user"), "New session")

    def repair_pending_tools(self) -> None:
        """Pair every saved call with a result after cancellation or a process crash."""
        repaired: list[dict] = []
        pending: list[dict] = []

        def flush() -> None:
            for call in pending:
                repaired.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": "Interrupted. Execution may have started; inspect the workspace "
                        "before retrying. No result was recorded.",
                    }
                )
            pending.clear()

        for message in self.messages:
            if message["role"] != "tool":
                flush()
            else:
                pending = [call for call in pending if call["id"] != message.get("tool_call_id")]
            repaired.append(message)
            if message.get("tool_calls"):
                pending.extend(message["tool_calls"])
        flush()
        self.messages = repaired


class SessionStore:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or Path.home() / ".poe" / "python-sessions"

    def save(self, session: Session) -> None:
        if not re.fullmatch(r"[a-f0-9]{32}", session.id):
            raise ValueError("Invalid session ID")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        session.updated_at = now()
        atomic_write(self.directory / f"{session.id}.json", json.dumps(asdict(session), indent=2))

    def _read(self, path: Path) -> Session:
        try:
            data = json.loads(path.read_text())
            if data.get("version") != 1:
                raise ValueError("unsupported session version")
            session = Session(**data)
            if not re.fullmatch(r"[a-f0-9]{32}", session.id) or session.id != path.stem:
                raise ValueError("invalid session ID")
            if (
                not all(
                    isinstance(value, str)
                    for value in (
                        session.cwd,
                        session.model,
                        session.created_at,
                        session.updated_at,
                    )
                )
                or not Path(session.cwd).is_absolute()
            ):
                raise ValueError("invalid session metadata")
            if not isinstance(session.messages, list):
                raise ValueError("invalid messages")
            for message in session.messages:
                if not isinstance(message, dict) or message.get("role") not in {
                    "system",
                    "user",
                    "assistant",
                    "tool",
                }:
                    raise ValueError("invalid message")
                if not isinstance(message.get("content", ""), (str, type(None))):
                    raise ValueError("invalid message content")
                if message["role"] == "user" and not isinstance(message.get("content"), str):
                    raise ValueError("invalid user message")
                for call in message.get("tool_calls", []):
                    if not isinstance(call["id"], str) or not isinstance(call["function"], dict):
                        raise ValueError("invalid tool call")
            session.repair_pending_tools()
            return session
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"Could not load session {path.name}: {exc}") from exc

    def list(self) -> list[Session]:
        sessions = []
        for path in self.directory.glob("*.json"):
            try:
                sessions.append(self._read(path))
            except ValueError:
                continue  # One damaged transcript must not hide the other sessions.
        return sorted(sessions, key=lambda s: (s.updated_at, s.created_at, s.id), reverse=True)

    def load(self, selector: str) -> Session:
        if selector != "latest" and not re.fullmatch(r"[a-f0-9]{1,32}", selector):
            raise ValueError("Use 'latest' or a session ID prefix from --sessions")
        sessions = self.list()
        matches = (
            sessions[:1]
            if selector == "latest"
            else [session for session in sessions if session.id.startswith(selector)]
        )
        if not matches:
            raise ValueError(f"No saved session matches {selector!r}")
        if len(matches) != 1:
            raise ValueError(f"Session prefix {selector!r} is ambiguous; use a longer ID")
        return matches[0]

"""The five local tools. Shell execution is trusted, not sandboxed."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import stat
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Literal

from poe.tooling import ToolResult, ToolSpec

MAX_FILE_BYTES = 2_000_000
MAX_OUTPUT = 20_000


def schema(name: str, description: str, properties: dict, required: list[str]) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )


PATH = {"type": "string", "description": "Path relative to the workspace, or absolute within it."}
LOCAL_TOOL_SPECS = [
    schema(
        "read_file",
        "Read a UTF-8 text file with line numbers (up to 2 MB).",
        {
            "file_path": PATH,
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
        },
        ["file_path"],
    ),
    schema(
        "list_dir",
        "List a directory. Common generated directories are not traversed.",
        {
            "dir_path": PATH,
            "depth": {"type": "integer", "minimum": 1, "maximum": 4},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        [],
    ),
    schema(
        "edit_file",
        "Replace exact text in an existing file. Read the file first. "
        "The match count must equal expected_replacements (default 1).",
        {
            "file_path": PATH,
            "search": {"type": "string", "minLength": 1},
            "replace": {"type": "string"},
            "expected_replacements": {"type": "integer", "minimum": 1},
        },
        ["file_path", "search", "replace"],
    ),
    schema(
        "write_file",
        "Create a UTF-8 file and any missing parent directories. "
        "Existing files require overwrite=true; prefer edit_file for changes.",
        {"file_path": PATH, "content": {"type": "string"}, "overwrite": {"type": "boolean"}},
        ["file_path", "content"],
    ),
    schema(
        "shell",
        "Run a non-interactive shell command, including searches and tests. "
        "Commands run with the user's permissions, not in a sandbox. "
        "Output is capped at 20000 bytes; default timeout is 30 seconds.",
        {
            "command": {"type": "string", "minLength": 1},
            "cwd": PATH,
            "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 300_000},
        },
        ["command"],
    ),
]


TOOL_DEFINITIONS = [spec.openrouter_definition() for spec in LOCAL_TOOL_SPECS]


def clip(text: str, limit: int = MAX_OUTPUT) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[output truncated]"


def atomic_write(path: Path, text: str, mode: int = 0o600) -> None:
    """Replace a file atomically; private by default for session transcripts."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ToolRunner:
    source_id = "local"
    namespaced = False
    approval: Literal["never"] = "never"

    def __init__(self, cwd: Path):
        self.cwd = cwd.resolve()

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def specs(self) -> list[ToolSpec]:
        return LOCAL_TOOL_SPECS.copy()

    def path(self, value: str) -> Path:
        candidate = Path(value)
        path = (self.cwd / candidate).resolve()
        if not path.is_relative_to(self.cwd):
            raise ValueError("Path must stay within the workspace")
        return path

    def validate(self, name: str, args: Any) -> None:
        spec = next((tool for tool in LOCAL_TOOL_SPECS if tool.name == name), None)
        if spec is None:
            raise ValueError(f"Unknown tool: {name}")
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be a JSON object")
        parameters = spec.input_schema
        missing = set(parameters["required"]) - args.keys()
        extra = args.keys() - parameters["properties"].keys()
        if missing or extra:
            raise ValueError(
                f"Invalid arguments: missing={sorted(missing)}, unknown={sorted(extra)}"
            )
        types = {"string": str, "integer": int, "boolean": bool}
        for key, value in args.items():
            prop = parameters["properties"][key]
            if type(value) is not types[prop["type"]]:
                raise ValueError(f"{key} must be {prop['type']}")
            if "minimum" in prop and value < prop["minimum"]:
                raise ValueError(f"{key} must be >= {prop['minimum']}")
            if "maximum" in prop and value > prop["maximum"]:
                raise ValueError(f"{key} must be <= {prop['maximum']}")
            if "minLength" in prop and len(value) < prop["minLength"]:
                raise ValueError(f"{key} must not be empty")

    async def run(self, name: str, args: Any) -> ToolResult:
        try:
            self.validate(name, args)
            if name == "shell":
                return await self.shell(**args)
            method = {
                "read_file": self.read_file,
                "list_dir": self.list_dir,
                "edit_file": self.edit_file,
                "write_file": self.write_file,
            }[name]
            return ToolResult(clip(method(**args)))
        except (ValueError, OSError, UnicodeError) as exc:
            return ToolResult(f"Error: {exc}", success=False)

    def read_text(self, path: Path) -> str:
        if not path.is_file():
            raise ValueError(f"Not a regular file: {path}")
        with path.open("rb") as handle:
            content = handle.read(MAX_FILE_BYTES + 1)
        if len(content) > MAX_FILE_BYTES:
            raise ValueError("File exceeds 2 MB; inspect it with a targeted shell command")
        if b"\x00" in content:
            raise ValueError("Binary files are not supported")
        return content.decode("utf-8")

    def read_file(self, file_path: str, offset: int = 1, limit: int = 2000) -> str:
        lines = self.read_text(self.path(file_path)).splitlines()
        selected = lines[offset - 1 : offset - 1 + limit]
        result = "\n".join(f"{n}: {clip(line, 2000)}" for n, line in enumerate(selected, offset))
        if offset - 1 + limit < len(lines):
            result += f"\n[more lines; next offset: {offset + limit}]"
        return result or "[empty file or offset beyond end]"

    def list_dir(self, dir_path: str = ".", depth: int = 2, limit: int = 200) -> str:
        root = self.path(dir_path)
        queue = deque([(root, 0)])
        entries: list[str] = []
        skip = {".git", ".venv", "node_modules", "target", "__pycache__"}
        while queue:
            directory, level = queue.popleft()
            # Bound traversal and memory even in directories with millions of entries.
            with os.scandir(directory) as scan:
                for entry in scan:
                    if len(entries) >= limit:
                        return "\n".join(sorted(entries)) + "\n[entry limit reached]"
                    path = Path(entry.path)
                    is_dir = entry.is_dir(follow_symlinks=False)
                    suffix = "@" if entry.is_symlink() else "/" if is_dir else ""
                    entries.append(str(path.relative_to(root)) + suffix)
                    if is_dir and entry.name not in skip and level + 1 < depth:
                        queue.append((path, level + 1))
        return "\n".join(sorted(entries)) or "[empty directory]"

    def edit_file(
        self, file_path: str, search: str, replace: str, expected_replacements: int = 1
    ) -> str:
        path = self.path(file_path)
        original = self.read_text(path)
        count = original.count(search)
        if count != expected_replacements:
            raise ValueError(
                f"Expected {expected_replacements} matches, found {count}; file unchanged"
            )
        updated = original.replace(search, replace)
        if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError("Resulting file would exceed 2 MB")
        atomic_write(path, updated, stat.S_IMODE(path.stat().st_mode))
        return f"Edited {file_path}: {count} replacement(s)"

    def write_file(self, file_path: str, content: str, overwrite: bool = False) -> str:
        path = self.path(file_path)
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError("Content exceeds 2 MB")
        if path.exists() and not overwrite:
            raise ValueError("File exists; use edit_file or set overwrite=true")
        if path.exists() and not path.is_file():
            raise ValueError("Target is not a regular file")
        path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
            atomic_write(path, content, mode)
        else:
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(content)
        return f"Wrote {file_path} ({len(content.encode('utf-8'))} bytes)"

    async def shell(self, command: str, cwd: str = ".", timeout_ms: int = 30_000) -> ToolResult:
        if not command.strip():
            raise ValueError("command must not be blank")
        spawn = asyncio.create_task(
            asyncio.create_subprocess_shell(
                command,
                cwd=self.path(cwd),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
        )
        cancelled_during_spawn = False
        try:
            process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            # Obtain the process handle before propagating cancellation so that
            # cancellation during startup cannot lose a newly created child.
            process = await spawn
            cancelled_during_spawn = True
        output = bytearray()
        truncated = False

        async def drain() -> None:
            nonlocal truncated
            assert process.stdout is not None
            while chunk := await process.stdout.read(8192):
                remaining = MAX_OUTPUT - len(output)
                output.extend(chunk[:remaining])
                truncated |= len(chunk) > remaining
            await process.wait()

        task = asyncio.create_task(drain())
        timed_out = False
        try:
            if cancelled_during_spawn:
                raise asyncio.CancelledError
            await asyncio.wait_for(asyncio.shield(task), timeout_ms / 1000)
        except (TimeoutError, asyncio.CancelledError) as exc:
            # Kill the group even if the shell itself has already exited: a child
            # may still own stdout. This also bounds commands that fork children.
            with contextlib.suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            await process.wait()
            await task
            if isinstance(exc, asyncio.CancelledError):
                raise
            timed_out = True
        content = output.decode("utf-8", errors="replace")
        if truncated:
            content += "\n[output truncated]"
        status = (
            f"Timed out after {timeout_ms} ms" if timed_out else f"Exit code: {process.returncode}"
        )
        return ToolResult(f"{status}\n{content}", not timed_out and process.returncode == 0)

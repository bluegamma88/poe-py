import asyncio
import os
import shlex
import stat
import sys
import time

import pytest

from poe.tools import MAX_OUTPUT, ToolRunner


async def test_file_tools_create_read_edit_and_preserve_mode(tmp_path):
    tools = ToolRunner(tmp_path)
    result = await tools.run(
        "write_file", {"file_path": "src/demo.py", "content": "print('old')\n"}
    )
    assert result.success
    path = tmp_path / "src/demo.py"
    path.chmod(0o755)
    assert (await tools.run("read_file", {"file_path": "src/demo.py"})).content == "1: print('old')"
    result = await tools.run(
        "edit_file",
        {
            "file_path": "src/demo.py",
            "search": "old",
            "replace": "new",
        },
    )
    assert result.success
    assert path.read_text() == "print('new')\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o755
    assert "src/demo.py" in (await tools.run("list_dir", {})).content


async def test_edits_require_exact_counts_and_overwrite_is_explicit(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("same same\n")
    tools = ToolRunner(tmp_path)
    result = await tools.run(
        "edit_file", {"file_path": "a.txt", "search": "same", "replace": "new"}
    )
    assert not result.success
    assert "found 2" in result.content
    assert path.read_text() == "same same\n"
    result = await tools.run("write_file", {"file_path": "a.txt", "content": "lost"})
    assert not result.success
    assert path.read_text() == "same same\n"
    assert (
        await tools.run(
            "edit_file",
            {
                "file_path": "a.txt",
                "search": "same",
                "replace": "new",
                "expected_replacements": 2,
            },
        )
    ).success
    assert path.read_text() == "new new\n"


async def test_preserves_crlf(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"one\r\ntwo\r\n")
    result = await ToolRunner(tmp_path).run(
        "edit_file",
        {
            "file_path": "a.txt",
            "search": "one",
            "replace": "three",
        },
    )
    assert result.success
    assert path.read_bytes() == b"three\r\ntwo\r\n"


async def test_rejects_workspace_escape_and_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (workspace / "link").symlink_to(outside)
    tools = ToolRunner(workspace)
    for path in ("../outside.txt", str(outside), "link"):
        assert not (await tools.run("read_file", {"file_path": path})).success
        assert not (
            await tools.run(
                "write_file",
                {
                    "file_path": path,
                    "content": "overwritten",
                    "overwrite": True,
                },
            )
        ).success
    assert outside.read_text() == "private"


@pytest.mark.parametrize(
    "name,args",
    [
        ("unknown", {}),
        ("read_file", []),
        ("read_file", {}),
        ("read_file", {"file_path": "a", "offset": 0}),
        ("read_file", {"file_path": "a", "offset": True}),
        ("list_dir", {"depth": 100}),
        ("list_dir", {"unknown": 1}),
        ("edit_file", {"file_path": "a", "search": "", "replace": "x"}),
        ("shell", {"command": "echo ok", "timeout_ms": 0}),
    ],
)
async def test_invalid_tool_arguments_are_recoverable(tmp_path, name, args):
    assert not (await ToolRunner(tmp_path).run(name, args)).success


async def test_binary_and_oversized_reads_are_rejected(tmp_path):
    (tmp_path / "binary").write_bytes(b"a\x00b")
    (tmp_path / "large").write_bytes(b"x" * 2_000_001)
    tools = ToolRunner(tmp_path)
    for path in ("binary", "large"):
        assert not (await tools.run("read_file", {"file_path": path})).success


async def test_shell_exit_status_and_bounded_output(tmp_path):
    tools = ToolRunner(tmp_path)
    result = await tools.run("shell", {"command": "printf failure >&2; exit 7"})
    assert not result.success
    assert "Exit code: 7" in result.content and "failure" in result.content
    code = "print('x' * 100000)"
    result = await tools.run(
        "shell", {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"}
    )
    assert result.success
    assert "output truncated" in result.content
    assert len(result.content) < MAX_OUTPUT + 100


async def test_shell_timeout_including_descendant_pipe(tmp_path):
    start = time.monotonic()
    result = await ToolRunner(tmp_path).run(
        "shell",
        {
            "command": "sleep 30 & wait",
            "timeout_ms": 50,
        },
    )
    assert not result.success
    assert "Timed out" in result.content
    assert time.monotonic() - start < 3


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group cleanup")
async def test_shell_cancel_kills_process(tmp_path):
    tools = ToolRunner(tmp_path)
    task = asyncio.create_task(tools.run("shell", {"command": "echo $$ > pid; exec sleep 30"}))
    async with asyncio.timeout(3):
        while not (tmp_path / "pid").exists():  # noqa: ASYNC110 — readiness from an external process
            await asyncio.sleep(0.01)
    pid = int((tmp_path / "pid").read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group cleanup")
async def test_cancel_during_process_startup_still_cleans_up(tmp_path, monkeypatch):
    real_spawn = asyncio.create_subprocess_shell
    started = asyncio.Event()
    release = asyncio.Event()
    children = []

    async def slow_spawn(*args, **kwargs):
        process = await real_spawn(*args, **kwargs)
        children.append(process)
        started.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_shell", slow_spawn)
    task = asyncio.create_task(ToolRunner(tmp_path).run("shell", {"command": "sleep 30"}))
    await asyncio.wait_for(started.wait(), 3)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert children[0].returncode is not None

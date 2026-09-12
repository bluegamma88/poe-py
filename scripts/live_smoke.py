"""Opt-in live integration test. Sends a tiny fixture to OpenRouter and uses API credits."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values

from poe.agent import Agent
from poe.app import Composer, PoeApp
from poe.config import Config
from poe.provider import OpenRouter
from poe.sessions import Session, SessionStore


class SmokeApp(PoeApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.errors: list[str] = []

    async def notice(self, text: str, *, error: bool = False) -> None:
        if error:
            self.errors.append(text.replace(self.agent.config.api_key, "[redacted]"))
        await super().notice(text, error=error)


async def run(model: str) -> None:
    # Explicitly prefer the .env key for this opt-in test. Never print credentials
    # or put them in the agent's workspace, prompt, or conversation metadata.
    key = dotenv_values(Path.cwd() / ".env", interpolate=False).get("OPENROUTER_API_KEY")
    config = Config(
        model=model, api_key=key or os.environ.get("OPENROUTER_API_KEY", ""), max_tool_rounds=8
    )
    config.require_api_key()
    with tempfile.TemporaryDirectory(prefix="poe-live-") as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "calculator.py").write_text("def add(a, b):\n    return a - b\n")
        (workspace / "check.py").write_text(
            "from calculator import add\n"
            "assert add(2, 3) == 5\n"
            "assert add(-1, 1) == 0\n"
            "print('CHECK_OK')\n"
        )
        store = SessionStore(root / "sessions")
        session = Session(cwd=str(workspace), model=model)
        app = SmokeApp(
            Agent(config, session, store, OpenRouter(config)),
            initial_prompt=(
                "Fix the add function in calculator.py. Read calculator.py and check.py first. "
                "Use edit_file for the change, then run "
                f"{shlex.quote(sys.executable)} check.py with the shell tool. "
                "Do not change check.py. Keep your final answer to one sentence."
            ),
        )
        print(f"Testing {model}: Textual → read → edit → shell → saved conversation", flush=True)
        async with asyncio.timeout(180), app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert not app.errors, app.errors
            transcript = session.messages
            calls = [call for message in transcript for call in message.get("tool_calls", [])]
            names = [call["function"]["name"] for call in calls]
            assert {"read_file", "edit_file", "shell"} <= set(names), names
            assert "return a + b" in (workspace / "calculator.py").read_text()
            tool_output = [m["content"] for m in transcript if m["role"] == "tool"]
            assert any("Exit code: 0" in output and "CHECK_OK" in output for output in tool_output)
            assert transcript[-1]["role"] == "assistant", "Agent did not complete the turn"
            saved = store.load("latest")
            assert config.api_key not in json.dumps(saved.messages)
            print(f"First turn passed; tools: {', '.join(names)}", flush=True)

        # Use a new app and provider instance to exercise a genuine saved-session continuation.
        resumed = SmokeApp(Agent(config, saved, store, OpenRouter(config)))
        async with asyncio.timeout(90), resumed.run_test(size=(80, 24)) as pilot:
            resumed.query_one(Composer).text = (
                "What exact success marker did the previous check print? "
                "Answer from the conversation in one sentence; no tools needed."
            )
            await pilot.press("enter")
            await resumed.workers.wait_for_complete()
            assert not resumed.errors, resumed.errors
            assert saved.messages[-1]["role"] == "assistant"
            assert "CHECK_OK" in saved.messages[-1]["content"]
            print("Resumed conversation passed; live smoke test complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=Config.model)
    arguments = parser.parse_args()
    asyncio.run(run(arguments.model))

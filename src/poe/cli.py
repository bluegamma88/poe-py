"""The installed poe-py console entry point."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from poe import __version__
from poe.agent import Agent
from poe.config import Config
from poe.provider import OpenRouter
from poe.sessions import Session, SessionStore


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Poe: a small Textual coding agent")
    result.add_argument("prompt", nargs="?", default="", help="optional opening prompt")
    result.add_argument("-C", "--cwd", type=Path, help="workspace directory (default: current)")
    result.add_argument("--model", help="OpenRouter model slug")
    result.add_argument("--config", type=Path, help="config file (default: ~/.poe/config.toml)")
    result.add_argument("--version", action="version", version=f"poe-py {__version__}")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--sessions", action="store_true", help="list saved conversations")
    mode.add_argument("--resume", metavar="ID", help="resume a session ID prefix, or 'latest'")
    return result


def main(argv: list[str] | None = None) -> None:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    store = SessionStore()
    try:
        if args.sessions:
            sessions = store.list()
            if not sessions:
                print("No saved conversations yet.")
            for session in sessions:
                title = " ".join(session.title.split())[:70]
                print(f"{session.id[:8]}  {session.updated_at}  {session.cwd}  {title}")
            return
        if args.config is not None and not args.config.expanduser().is_file():
            raise ValueError(f"Config file does not exist: {args.config}")
        config = Config.load(args.config.expanduser() if args.config else None, model=args.model)
        config.require_api_key()
        if args.resume:
            session = store.load(args.resume)
            config = replace(config, model=args.model or session.model)
            # Resume into a new transcript, preserving the original as a checkpoint.
            session = Session(
                cwd=session.cwd,
                model=config.model,
                messages=[message.copy() for message in session.messages],
            )
        else:
            session = Session(cwd=str(Path.cwd()), model=config.model)
        cwd = (args.cwd.expanduser() if args.cwd else Path(session.cwd)).resolve()
        if not cwd.is_dir():
            raise ValueError(f"Workspace is not a directory: {cwd}")
        session.cwd = str(cwd)
        agent = Agent(config, session, store, OpenRouter(config))
        if args.resume:
            # Refresh workspace instructions, especially when -C overrides the old path.
            session.messages = [m for m in session.messages if m["role"] != "system"]
            session.messages.insert(0, agent.system_message())
        from poe.app import PoeApp

        PoeApp(agent, initial_prompt=args.prompt).run()
    except (OSError, ValueError) as exc:
        arg_parser.exit(2, f"poe-py: {exc}\n")


if __name__ == "__main__":
    main()

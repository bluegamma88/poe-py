# Poe (Python)

A small coding agent built with Python and [Textual](https://textual.textualize.io/).

## Install

Requires Python 3.11 or newer. From a checkout:

```sh
uv tool install .
poe-py
```

Once this directory is published as a GitHub repository:

```sh
uv tool install git+https://github.com/bluegamm88/poe-py.git
```

## Configure

Poe reads `~/.poe/config.toml`

```toml
model = "openai/gpt-oss-120b"
# max_tool_rounds = 50

[openrouter]
api_key_env = "OPENROUTER_API_KEY"
# api_key = "sk-or-..."
# base_url = "https://openrouter.ai/api/v1"
```

Alternatively, export `OPENROUTER_API_KEY`; no configuration file is required.
You can also put `OPENROUTER_API_KEY=sk-or-...` in a `.env` file in the directory
where you launch `poe-py`. Key precedence is config file, environment variable,
then `.env`. Only the configured key variable is read from `.env`; it does not
modify your process environment or search parent directories. `.env` files are
ignored by Git and excluded from the built package.
Choose an [OpenRouter model that supports tools](https://openrouter.ai/models?supported_parameters=tools).

## Use

```sh
poe-py                              # Start in the current directory
poe-py "Explain this project"       # Start with a prompt
poe-py -C /path/to/project          # Choose the workspace
poe-py --model provider/model       # Override the configured model
poe-py --config /path/config.toml   # Use another configuration file
poe-py --sessions                   # List saved conversations
poe-py --resume latest              # Continue the latest conversation
poe-py --resume a1b2c3d4            # Continue by unique ID prefix
```

Enter sends a prompt; Shift+Enter inserts a newline. Ctrl+J also works as a fallback
for terminals that cannot distinguish Shift+Enter from Enter. Escape cancels the current turn,
Ctrl+N starts a new conversation, and Ctrl+Q quits. `/new`, `/help`, and `/quit`
are also supported. Tool calls and their output appear in expandable panels.

The agent can list directories, read text files, make exact text replacements,
write files, and execute shell commands. It reads the workspace's root
`AGENTS.md` at the start of a conversation. File tools stay within the chosen
workspace and reject paths that escape it, including symlinks. Edits require
an exact match count, and overwriting a file requires an explicit tool argument.

Shell commands run automatically with your user permissions in the workspace;
this is a trusted local coding tool, **not an OS sandbox**. Commands can access
paths outside the workspace. Cancellation stops active requests and, on POSIX,
the shell process group; completed file edits are not undone.

Conversations, including file contents and tool output sent to the model, are
saved locally in `~/.poe/python-sessions/`. Credentials are not stored in the
session metadata. This separate format does not import Rust session files.
Resuming uses the original workspace and model unless overridden by CLI flags.
`--sessions` and `--help` work without an API key.

## Develop

```sh
uv sync --dev
uv run poe-py
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

For an opt-in live smoke test using the key in your local `.env` (uses API credits):

```sh
uv run python scripts/live_smoke.py
```

This launches the actual Textual app headlessly, asks the agent to fix and test a
small program in a temporary workspace, verifies the result, and checks a follow-up
turn from the saved conversation. It does not send this repository to the model.

The package uses a `src/` layout and a standard console entry point, so `uv tool
install` builds the same wheel as a local install. The Rust reference is excluded
from distribution artifacts.

The core is deliberately small: `provider.py` handles OpenRouter streaming,
`agent.py` owns the tool loop, `tools.py` implements local operations,
`sessions.py` stores transcripts, and `app.py` provides the Textual interface.
Tests use a fake HTTP transport and Textual's headless pilot; they require no
API credentials. There is no MCP, multi-agent orchestration, automatic context
compaction, or custom terminal renderer in this first version. Use `/new` when
a conversation grows too large for your model.

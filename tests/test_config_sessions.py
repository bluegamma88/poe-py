import json
import stat

import pytest

from poe.cli import main
from poe.config import Config
from poe.sessions import Session, SessionStore


def test_rust_config_compatibility_and_key_precedence(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('model = "anthropic/example"\n[openrouter]\napi_key = "file-key"\n')
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    config = Config.load(path)
    assert config.model == "anthropic/example"
    assert config.api_key == "file-key"
    assert "file-key" not in repr(config)
    assert Config.load(path, model="custom/model").model == "custom/model"


def test_custom_key_env_and_empty_literal(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[openrouter]\napi_key = ""\napi_key_env = "TEST_POE_KEY"\n')
    monkeypatch.setenv("TEST_POE_KEY", "custom-key")
    assert Config.load(path).api_key == "custom-key"


def test_dotenv_loading_and_precedence_without_environment_mutation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text('OPENROUTER_API_KEY="local-key"\nUNRELATED=ignored\n')
    missing_config = tmp_path / "config.toml"
    assert Config.load(missing_config).api_key == "local-key"
    import os

    assert "OPENROUTER_API_KEY" not in os.environ
    assert "UNRELATED" not in os.environ
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    assert Config.load(missing_config).api_key == "env-key"
    missing_config.write_text('[openrouter]\napi_key="config-key"\n')
    assert Config.load(missing_config).api_key == "config-key"


def test_dotenv_does_not_search_parent_directories(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=parent-key\n")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert Config.load(child / "config.toml").api_key == ""


@pytest.mark.parametrize(
    "text",
    [
        "not valid toml",
        "model = 123",
        "max_tool_rounds = true",
        'openrouter = "bad"',
        "max_tool_rounds = 0",
    ],
)
def test_config_validation(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    with pytest.raises(ValueError):
        Config.load(path)


def test_session_roundtrip_private_permissions_and_no_key(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = Session(
        cwd=str(tmp_path),
        model="test/model",
        messages=[
            {"role": "user", "content": "fix this"},
        ],
    )
    store.save(session)
    path = store.directory / f"{session.id}.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "api_key" not in json.loads(path.read_text())
    assert store.load(session.id[:8]) == session
    assert store.load("latest") == session
    assert store.list()[0].title == "fix this"


def test_repairs_interrupted_tool_batch(tmp_path):
    session = Session(
        cwd=str(tmp_path),
        model="test",
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "function": {"name": "shell", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "read_file", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "completed"},
        ],
    )
    store = SessionStore(tmp_path / "sessions")
    store.save(session)
    restored = store.load("latest")
    assert restored.messages[1]["content"] == "completed"
    assert restored.messages[2]["tool_call_id"] == "b"
    assert "Interrupted" in restored.messages[2]["content"]
    restored.repair_pending_tools()
    assert len(restored.messages) == 3


def test_bad_sessions_do_not_hide_good_sessions(tmp_path):
    store = SessionStore(tmp_path)
    store.save(Session(cwd=str(tmp_path), model="test"))
    (tmp_path / "broken.json").write_text("not json")
    assert len(store.list()) == 1
    with pytest.raises(ValueError):
        store.load("../outside")


def test_cli_help_version_and_sessions_do_not_need_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("poe.cli.SessionStore", lambda: SessionStore(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for flag in ("--help", "--version"):
        with pytest.raises(SystemExit) as exc:
            main([flag])
        assert exc.value.code == 0
    main(["--sessions"])
    assert "No saved conversations" in capsys.readouterr().out


def test_cli_missing_config_is_actionable(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(tmp_path / "missing.toml")])
    assert exc.value.code == 2
    assert "Config file does not exist" in capsys.readouterr().err

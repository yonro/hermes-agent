"""Tests for the XMemo memory provider plugin."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pytest

from plugins.memory.xmemo import XMemoMemoryProvider


class FakeXMemoClient:
    """Fake synchronous XMemo REST client for unit tests."""

    def __init__(
        self,
        search_results: Optional[List[Dict[str, Any]]] = None,
        recall_context: Optional[Dict[str, Any]] = None,
    ):
        self.search_results = search_results or []
        self.recall_context_response = recall_context or {}
        self.captured_calls: List[Dict[str, Any]] = []

    def _record(self, method: str, **kwargs):
        self.captured_calls.append({"method": method, **kwargs})

    def health(self):
        self._record("health")
        return {"status": "ok"}

    def recall_context(self, **kwargs):
        self._record("recall_context", **kwargs)
        return self.recall_context_response

    def search(self, **kwargs):
        self._record("search", **kwargs)
        return self.search_results

    def remember(self, **kwargs):
        self._record("remember", **kwargs)
        return {"id": "mem-test-123"}

    def update_state(self, **kwargs):
        self._record("update_state", **kwargs)
        return {"state_key": kwargs.get("state_key", "active_task"), "id": "state-123"}

    def record_event(self, **kwargs):
        self._record("record_event", **kwargs)
        return {"id": "event-123"}

    def create_restart_snapshot(self, **kwargs):
        self._record("create_restart_snapshot", **kwargs)
        return {"id": "snapshot-123"}

    def create_reminder(self, **kwargs):
        self._record("create_reminder", **kwargs)
        return {"id": "reminder-123"}

    def list_reminders(self, **kwargs):
        self._record("list_reminders", **kwargs)
        return self.search_results

    def complete_reminder(self, **kwargs):
        self._record("complete_reminder", **kwargs)
        return {"id": kwargs.get("todo_id", "reminder-123")}

    def mark_used(self, **kwargs):
        self._record("mark_used", **kwargs)
        return {"id": kwargs.get("memory_id", "mem-123")}

    def forget(self, **kwargs):
        self._record("forget", **kwargs)
        return {"id": kwargs.get("target", "mem-123")}

    def close(self):
        self._record("close")


@pytest.fixture
def provider_with_config(monkeypatch, tmp_path):
    """Create an initialized provider with a fake client."""
    monkeypatch.setenv("XMEMO_KEY", "test-key")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XMEMO_AGENT_INSTANCE_ID", "test-instance")

    provider = XMemoMemoryProvider()
    provider.initialize("test-session")
    return provider


class TestAvailability:
    """is_available must be fast and network-free."""

    def test_not_available_without_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XMEMO_KEY", raising=False)
        monkeypatch.delenv("MEMORY_OS_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        assert provider.is_available() is False

    def test_available_with_env_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        assert provider.is_available() is True

    def test_available_with_legacy_env_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XMEMO_KEY", raising=False)
        monkeypatch.setenv("MEMORY_OS_API_KEY", "legacy-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        assert provider.is_available() is True


class TestLifecycle:
    """Initialization and shutdown behavior."""

    def test_initialize_loads_config(self, provider_with_config):
        assert provider_with_config._config["api_key"] == "test-key"
        assert provider_with_config._config["agent_id"] == "hermes"
        assert provider_with_config._session_id == "test-session"

    def test_system_prompt_block_active(self, provider_with_config):
        block = provider_with_config.system_prompt_block()
        assert "XMemo Memory" in block
        assert "xmemo_search" in block
        assert "xmemo_remember" in block

    def test_system_prompt_block_empty_when_not_configured(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XMEMO_KEY", raising=False)
        monkeypatch.delenv("MEMORY_OS_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("test-session")
        assert provider.system_prompt_block() == ""

    def test_name_property(self):
        provider = XMemoMemoryProvider()
        assert provider.name == "xmemo"


class TestTools:
    """Tool handlers route to the correct API calls."""

    def test_search_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient(
            search_results=[
                {"content": "user prefers dark mode", "memory_type": "semantic", "similarity": 0.92},
            ]
        )
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call("xmemo_search", {"query": "preferences"})
        )

        assert result["count"] == 1
        assert result["results"][0]["content"] == "user prefers dark mode"
        assert fake.captured_calls[0]["method"] == "search"
        assert fake.captured_calls[0]["query"] == "preferences"

    def test_search_tool_missing_query(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(provider_with_config.handle_tool_call("xmemo_search", {}))
        assert "error" in result

    def test_remember_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_remember",
                {"content": "user likes small PRs", "path": "hermes/preferences"},
            )
        )

        assert result["result"] == "Saved to XMemo."
        assert result["memory_id"] == "mem-test-123"
        assert fake.captured_calls[0]["method"] == "remember"
        assert fake.captured_calls[0]["content"] == "user likes small PRs"

    def test_remember_tool_missing_content(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_remember", {"path": "hermes/preferences"}
            )
        )
        assert "error" in result

    def test_update_state_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_update_state",
                {"current_task": "Implement XMemo plugin", "next_action": "Write tests"},
            )
        )

        assert result["result"] == "Working state saved to XMemo."
        assert fake.captured_calls[0]["method"] == "update_state"
        assert fake.captured_calls[0]["current_task"] == "Implement XMemo plugin"

    def test_update_state_tool_missing_fields(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call("xmemo_update_state", {})
        )
        assert "error" in result

    def test_recall_context_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient(
            recall_context={
                "context_text": "User prefers concise answers.",
                "items": [{"content": "User prefers concise answers."}],
            }
        )
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_recall_context", {"query": "style preferences"}
            )
        )

        assert "concise answers" in result["context"]
        assert fake.captured_calls[0]["method"] == "recall_context"
        assert fake.captured_calls[0]["query"] == "style preferences"

    def test_recall_context_tool_missing_query(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call("xmemo_recall_context", {})
        )
        assert "error" in result

    def test_record_event_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_record_event",
                {"content": "Migrated to new memory backend", "event_type": "milestone"},
            )
        )

        assert result["result"] == "Event recorded in XMemo timeline."
        assert result["event_id"] == "event-123"
        assert fake.captured_calls[0]["method"] == "record_event"
        assert fake.captured_calls[0]["event_type"] == "milestone"

    def test_create_reminder_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_create_reminder",
                {"content": "Write migration docs", "due_at": "2026-06-20T10:00:00Z"},
            )
        )

        assert result["result"] == "Reminder saved to XMemo."
        assert result["todo_id"] == "reminder-123"
        assert fake.captured_calls[0]["method"] == "create_reminder"
        assert fake.captured_calls[0]["due_at"] == "2026-06-20T10:00:00Z"

    def test_list_reminders_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient(
            search_results=[
                {"content": "Write migration docs", "item_status": "open", "id": "reminder-123"},
            ]
        )
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_list_reminders", {"item_status": "open"}
            )
        )

        assert result["count"] == 1
        assert result["items"][0]["content"] == "Write migration docs"
        assert fake.captured_calls[0]["method"] == "list_reminders"
        assert fake.captured_calls[0]["item_status"] == "open"

    def test_complete_reminder_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_complete_reminder",
                {"todo_id": "reminder-123", "note": "Done in PR #42"},
            )
        )

        assert result["result"] == "Reminder marked completed."
        assert result["todo_id"] == "reminder-123"
        assert fake.captured_calls[0]["method"] == "complete_reminder"
        assert fake.captured_calls[0]["note"] == "Done in PR #42"

    def test_complete_reminder_tool_missing_id(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call("xmemo_complete_reminder", {})
        )
        assert "error" in result

    def test_mark_used_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_mark_used",
                {"memory_id": "mem-456", "context": "Used to answer style question"},
            )
        )

        assert result["result"] == "Memory usage recorded in XMemo."
        assert result["memory_id"] == "mem-456"
        assert fake.captured_calls[0]["method"] == "mark_used"
        assert fake.captured_calls[0]["context"] == "Used to answer style question"

    def test_forget_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_forget",
                {"target": "mem-789", "reason": "Outdated preference"},
            )
        )

        assert result["result"] == "Memory deleted from XMemo."
        assert result["memory_id"] == "mem-789"
        assert fake.captured_calls[0]["method"] == "forget"
        assert fake.captured_calls[0]["reason"] == "Outdated preference"

    def test_forget_tool_missing_target(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(provider_with_config.handle_tool_call("xmemo_forget", {}))
        assert "error" in result

    def test_tool_schemas_include_new_tools(self, provider_with_config):
        names = {s["name"] for s in provider_with_config.get_tool_schemas()}
        assert names == {
            "xmemo_search",
            "xmemo_remember",
            "xmemo_update_state",
            "xmemo_recall_context",
            "xmemo_record_event",
            "xmemo_create_reminder",
            "xmemo_list_reminders",
            "xmemo_complete_reminder",
            "xmemo_mark_used",
            "xmemo_forget",
        }


class TestPrefetch:
    """Background recall and prefetch behavior."""

    def test_queue_prefetch_populates_result(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient(
            recall_context={
                "context_text": "User is working on the XMemo integration.",
                "items": [{"content": "User is working on the XMemo integration."}],
            }
        )
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        provider_with_config.queue_prefetch("current task")
        provider_with_config._prefetch_thread.join(timeout=2)

        result = provider_with_config.prefetch("current task")
        assert "XMemo integration" in result
        assert fake.captured_calls[0]["method"] == "recall_context"
        assert fake.captured_calls[0]["query"] == "current task"

    def test_prefetch_skips_trivial_prompts(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        provider_with_config.queue_prefetch("ok")
        # No thread should have been started
        assert provider_with_config._prefetch_thread is None
        assert provider_with_config.prefetch("ok") == ""

    def test_prefetch_returns_empty_when_no_result(self, provider_with_config):
        result = provider_with_config.prefetch("anything")
        assert result == ""


class TestSyncTurn:
    """Turn synchronization records lightweight events."""

    def test_sync_turn_records_event(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        provider_with_config.sync_turn("hello", "hi there", session_id="s1")
        provider_with_config._sync_thread.join(timeout=2)

        assert len(fake.captured_calls) == 1
        assert fake.captured_calls[0]["method"] == "record_event"
        assert fake.captured_calls[0]["session_id"] == "s1"


class TestCircuitBreaker:
    """Consecutive failures should pause API calls temporarily."""

    def test_circuit_breaker_trips(self, provider_with_config, monkeypatch):
        class FailingClient:
            def search(self, **kwargs):
                raise RuntimeError("network down")

            def close(self):
                pass

        monkeypatch.setattr(provider_with_config, "_get_client", lambda: FailingClient())

        # Trigger enough failures to trip the breaker
        for _ in range(6):
            provider_with_config.handle_tool_call("xmemo_search", {"query": "x"})

        assert provider_with_config._is_breaker_open() is True

        # After the breaker is open, calls return immediately without hitting client
        result = json.loads(
            provider_with_config.handle_tool_call("xmemo_search", {"query": "y"})
        )
        assert "temporarily unavailable" in result["error"]


class TestConfigSchema:
    """Setup wizard integration."""

    def test_config_schema_has_api_key(self):
        provider = XMemoMemoryProvider()
        schema = provider.get_config_schema()
        keys = {field["key"] for field in schema}
        assert "api_key" in keys
        assert "base_url" in keys
        assert "scope" in keys

    def test_save_config_does_not_persist_api_key(self, tmp_path):
        provider = XMemoMemoryProvider()
        provider.save_config({"api_key": "secret", "scope": "hermes/test"}, str(tmp_path))

        config_file = tmp_path / "xmemo.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert "api_key" not in data
        assert data["scope"] == "hermes/test"


class TestSetupWizard:
    """post_setup() and cli.py write config files correctly."""

    def test_post_setup_writes_config_and_env(self, monkeypatch, tmp_path, capsys):
        from hermes_cli.config import load_config, save_config as save_global_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Seed an empty global config in the temp home
        save_global_config({"memory": {}})

        # Mock curses select for bucket choice -> "work"
        monkeypatch.setattr(
            "plugins.memory.xmemo.cli._curses_select", lambda title, choices, default=0: default
        )
        # Mock secret prompt and regular stdin prompt
        monkeypatch.setattr(
            "plugins.memory.xmemo.cli.masked_secret_prompt", lambda prompt: "xmemo-token-123"
        )
        # Answers for: base_url, agent_id, scope, timeout (all keep defaults)
        answers = iter(["", "", "", ""])
        monkeypatch.setattr("sys.stdin.readline", lambda: next(answers) + "\n")

        provider = XMemoMemoryProvider()
        config = load_config()
        provider.post_setup(str(tmp_path), config)

        # Verify config.yaml activation
        updated = load_config()
        assert updated.get("memory", {}).get("provider") == "xmemo"

        # Verify .env contains the secret
        env_file = tmp_path / ".env"
        assert env_file.exists()
        env_text = env_file.read_text()
        assert "XMEMO_KEY=xmemo-token-123" in env_text

        # Verify xmemo.json does NOT contain the secret
        xmemo_file = tmp_path / "xmemo.json"
        assert xmemo_file.exists()
        xmemo_data = json.loads(xmemo_file.read_text())
        assert "api_key" not in xmemo_data
        assert xmemo_data["bucket"] == "work"

        captured = capsys.readouterr()
        assert "Memory provider: xmemo" in captured.out

    def test_post_setup_preserves_existing_secret(self, monkeypatch, tmp_path):
        from hermes_cli.config import load_config, save_config as save_global_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("XMEMO_KEY", "existing-secret")
        save_global_config({"memory": {}})

        monkeypatch.setattr(
            "plugins.memory.xmemo.cli._curses_select", lambda title, choices, default=0: default
        )
        # User presses Enter for everything (keep existing secret)
        monkeypatch.setattr("sys.stdin.readline", lambda: "\n")

        provider = XMemoMemoryProvider()
        config = load_config()
        provider.post_setup(str(tmp_path), config)

        env_file = tmp_path / ".env"
        if env_file.exists():
            # Wizard should not write an empty value when user keeps existing secret
            assert "existing-secret" in env_file.read_text() or "XMEMO_KEY=" not in env_file.read_text()


class TestProfileIsolation:
    """Different Hermes profiles should use different XMemo scopes."""

    def test_scope_derived_from_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("test-session", agent_identity="coder")

        assert provider._config["scope"] == "hermes/coder"
